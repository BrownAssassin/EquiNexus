from __future__ import annotations

import csv
import hashlib
import itertools
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .classical import ClassicalConfig, compute_salience_zmap, predictions_from_zmap
from .data import inventory, load_dataset
from .predictor import read_grayscale
from .runner import (
    _git_status,
    _resolve_data_root,
    _write_observation_scores,
    _write_overlays,
    _write_predictions,
)
from .runtime import hardware_info, package_versions
from .scorer import score_validation
from .split import create_observation_split
from .visuals import write_comparison_overlays


def run_classical_validation(
    *,
    repo_root: str | Path,
    data_root: str | Path | None,
    run_name: str,
    observation_limit: int | None = None,
    single_config: ClassicalConfig | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    dataset = load_dataset(_resolve_data_root(data_root, repo_root))
    config = json.loads((repo_root / "configs" / "validation-baseline.json").read_text())
    config_sha256 = hashlib.sha256(
        (repo_root / "configs" / "validation-baseline.json").read_bytes()
    ).hexdigest()
    split = create_observation_split(
        dataset,
        repo_root / "data" / "splits",
        seed=int(config["seed"]),
        validation_fraction=float(config["validation_fraction"]),
    )
    if split.sha256 != config["expected_split_sha256"]:
        raise RuntimeError("The classical experiment requires the frozen split")
    all_validation_names = list(split.validation_names)
    validation_names = all_validation_names
    if observation_limit is not None:
        if observation_limit < 1:
            raise ValueError("observation_limit must be positive")
        if observation_limit < len(all_validation_names):
            if observation_limit == 1:
                validation_names = [all_validation_names[len(all_validation_names) // 2]]
            else:
                indices = sorted(
                    {
                        round(
                            index * (len(all_validation_names) - 1)
                            / (observation_limit - 1)
                        )
                        for index in range(observation_limit)
                    }
                )
                validation_names = [all_validation_names[index] for index in indices]
    run_root = repo_root / "artifacts" / "classical-experiments" / run_name
    run_root.mkdir(parents=True, exist_ok=True)
    repository_info = _git_status(repo_root)
    cache_dir = run_root / "salience-zmaps"
    cache_dir.mkdir(exist_ok=True)
    started = time.perf_counter()
    for index, file_name in enumerate(validation_names, start=1):
        cache_path = cache_dir / f"{Path(file_name).stem}.npz"
        if cache_path.is_file():
            continue
        image = read_grayscale(dataset.train_image_dir / file_name)
        z_map, disk = compute_salience_zmap(image)
        np.savez_compressed(
            cache_path,
            z_map=z_map.astype(np.float16),
            disk=disk.astype(np.uint8),
        )
        if index % 20 == 0 or index == len(validation_names):
            print(f"Cached {index}/{len(validation_names)} classical salience maps", flush=True)
    preprocessing_seconds = time.perf_counter() - started

    candidates = (
        [single_config]
        if single_config is not None
        else [
            ClassicalConfig(
                z_threshold=z_threshold,
                bridge_size=bridge_size,
                min_instance_area=min_area,
            )
            for z_threshold, bridge_size, min_area in itertools.product(
                (2.9, 3.2), (5, 9, 13), (250, 500)
            )
        ]
    )
    sweep_results = []
    best_pq = -1.0
    scoring_started = time.perf_counter()
    for index, candidate in enumerate(candidates, start=1):
        predictions = {}
        for file_name in validation_names:
            cache_path = cache_dir / f"{Path(file_name).stem}.npz"
            with np.load(cache_path) as cached:
                predictions[file_name] = predictions_from_zmap(
                    cached["z_map"].astype(np.float32),
                    cached["disk"].astype(bool),
                    file_name,
                    candidate,
                )
        metrics, rows = score_validation(
            dataset,
            predictions,
            validation_names,
            iou_threshold=float(config["iou_threshold"]),
            diagnostic_overlap_threshold=float(config["diagnostic_overlap_threshold"]),
        )
        global_metrics = metrics["global"]
        row = {
            "config": candidate.__dict__,
            "pq": global_metrics["pq"],
            "tp": global_metrics["tp"],
            "fp": global_metrics["fp"],
            "fn": global_metrics["fn"],
            "predictions_across_views": global_metrics[
                "prediction_count_across_annotator_views"
            ],
            "metrics": metrics,
        }
        sweep_results.append(row)
        print(
            f"[{index}/{len(candidates)}] z={candidate.z_threshold:.2f} "
            f"bridge={candidate.bridge_size} area={candidate.min_instance_area} "
            f"PQ={row['pq']:.6f} TP={row['tp']} FP={row['fp']} FN={row['fn']}",
            flush=True,
        )
        if row["pq"] > best_pq:
            best_pq = row["pq"]
            best_root = run_root / (
                "best-screen" if observation_limit is not None else "best"
            )
            best_root.mkdir(exist_ok=True)
            (best_root / "metrics.json").write_text(
                json.dumps(
                    {"split_sha256": split.sha256, "config": candidate.__dict__, "metrics": metrics},
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            _write_predictions(best_root / "predictions.csv", predictions)
            _write_observation_scores(best_root / "per_observation.csv", rows, predictions)
            _write_overlays(dataset, predictions, best_root / "overlays", 12)
            write_comparison_overlays(
                dataset, predictions, best_root / "comparison-overlays", 12
            )
    scoring_seconds = time.perf_counter() - scoring_started

    sweep_results.sort(key=lambda item: item["pq"], reverse=True)
    sweep_stem = "screen-sweep" if observation_limit is not None else "sweep"
    (run_root / f"{sweep_stem}.json").write_text(
        json.dumps(sweep_results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (run_root / f"{sweep_stem}.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = (
            *ClassicalConfig().__dict__.keys(),
            "pq",
            "tp",
            "fp",
            "fn",
        )
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in sweep_results:
            writer.writerow(
                {
                    **row["config"],
                    **{
                        field: row[field]
                        for field in fields[len(row["config"]) :]
                    },
                }
            )

    result: dict[str, Any] = {
        "status": "complete" if observation_limit is None else "validation_subset",
        "run_name": run_name,
        "repository": repository_info,
        "configuration_sha256": config_sha256,
        "dataset_inventory": inventory(dataset),
        "split_sha256": split.sha256,
        "validation_observations": len(validation_names),
        "full_holdout_observations": len(all_validation_names),
        "method": "independent implementation of public radial-normalization, multiscale black-hat, robust z-score, hysteresis, and component filtering approach",
        "reference": "https://www.kaggle.com/code/robinamirbahar/solar-filament-panoptic-morphology-segmentation",
        "best": sweep_results[0],
        "preprocessing_seconds": preprocessing_seconds,
        "sweep_and_scoring_seconds": scoring_seconds,
        "runtime": {**hardware_info(), "package_versions": package_versions()},
        "artifacts": {
            "sweep_csv": f"{sweep_stem}.csv",
            "sweep_json": f"{sweep_stem}.json",
            "best_dir": "best-screen" if observation_limit is not None else "best",
        },
    }
    (run_root / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Best classical PQ on {len(validation_names)} observations: {sweep_results[0]['pq']:.6f} "
        f"with {sweep_results[0]['config']}; outputs {run_root}",
        flush=True,
    )
    return result
