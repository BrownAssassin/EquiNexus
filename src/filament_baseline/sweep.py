from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from .data import load_dataset
from .inference import InstanceConfig, instances_from_probabilities
from .runner import _resolve_data_root, _write_observation_scores, _write_overlays, _write_predictions
from .scorer import score_validation
from .split import create_observation_split
from .visuals import write_comparison_overlays


def run_postprocessing_sweep(
    *,
    repo_root: str | Path,
    data_root: str | Path | None,
    run_root: str | Path,
    observation_limit: int | None = None,
) -> list[dict[str, Any]]:
    repo_root = Path(repo_root).resolve()
    run_root = Path(run_root).resolve()
    dataset = load_dataset(_resolve_data_root(data_root, repo_root))
    config = json.loads((repo_root / "configs" / "validation-baseline.json").read_text())
    split = create_observation_split(
        dataset,
        repo_root / "data" / "splits",
        seed=int(config["seed"]),
        validation_fraction=float(config["validation_fraction"]),
    )
    if split.sha256 != config["expected_split_sha256"]:
        raise RuntimeError("The postprocessing sweep requires the frozen validation split")
    base_instance_config = InstanceConfig()
    metrics_path = run_root / "metrics.json"
    if metrics_path.is_file():
        run_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        configured = run_metrics.get("method", {}).get("instance_config")
        if isinstance(configured, dict):
            fields = InstanceConfig.__dataclass_fields__
            base_instance_config = InstanceConfig(
                **{name: configured[name] for name in fields if name in configured}
            )
    validation_names = list(split.validation_names)
    if observation_limit is not None:
        validation_names = validation_names[:observation_limit]
    cache_dir = run_root / "probabilities"
    for file_name in validation_names:
        cache_path = cache_dir / f"{Path(file_name).stem}.npz"
        if not cache_path.is_file():
            raise FileNotFoundError(f"Missing cached validation probability map: {cache_path}")

    configurations = [
        InstanceConfig(
            foreground_threshold=foreground_threshold,
            boundary_threshold=boundary_threshold,
            boundary_weight=base_instance_config.boundary_weight,
            seed_threshold=base_instance_config.seed_threshold,
            seed_min_area=base_instance_config.seed_min_area,
            min_instance_area=base_instance_config.min_instance_area,
            solar_radius=base_instance_config.solar_radius,
        )
        for foreground_threshold, boundary_threshold in itertools.product(
            (0.25, 0.35, 0.45, 0.55, 0.65),
            (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90),
        )
    ]
    results = []
    for index, instance_config in enumerate(configurations, start=1):
        predictions = {}
        for file_name in validation_names:
            cache_path = cache_dir / f"{Path(file_name).stem}.npz"
            with np.load(cache_path) as cache:
                predictions[file_name] = instances_from_probabilities(
                    cache["foreground"].astype(np.float32),
                    cache["boundary"].astype(np.float32),
                    file_name,
                    instance_config,
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
            "configuration": instance_config.__dict__,
            "pq": global_metrics["pq"],
            "tp": global_metrics["tp"],
            "fp": global_metrics["fp"],
            "fn": global_metrics["fn"],
            "prediction_count_across_views": global_metrics[
                "prediction_count_across_annotator_views"
            ],
            "metrics": metrics,
        }
        results.append(row)
        print(
            f"[{index}/{len(configurations)}] fg={instance_config.foreground_threshold:.2f} "
            f"boundary={instance_config.boundary_threshold:.2f} "
            f"PQ={global_metrics['pq']:.6f} TP={global_metrics['tp']} "
            f"FP={global_metrics['fp']} FN={global_metrics['fn']}",
            flush=True,
        )

        if index == 1 or row["pq"] > max(item["pq"] for item in results[:-1]):
            best_dir = run_root / "best-postprocessing"
            best_dir.mkdir(parents=True, exist_ok=True)
            (best_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "split_sha256": split.sha256,
                        "base_instance_configuration": base_instance_config.__dict__,
                        "configuration": instance_config.__dict__,
                        "metrics": metrics,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            _write_predictions(best_dir / "predictions.csv", predictions)
            _write_observation_scores(best_dir / "per_observation.csv", rows, predictions)
            _write_overlays(dataset, predictions, best_dir / "overlays", 12)
            write_comparison_overlays(
                dataset, predictions, best_dir / "comparison-overlays", 12
            )

    result_path = run_root / "postprocessing_sweep.json"
    result_path.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (run_root / "postprocessing_sweep.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fields = (
            *InstanceConfig().__dict__.keys(),
            "pq",
            "tp",
            "fp",
            "fn",
            "prediction_count_across_views",
        )
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    **row["configuration"],
                    **{key: row[key] for key in fields[len(InstanceConfig().__dict__) :]},
                }
            )
    return sorted(results, key=lambda item: item["pq"], reverse=True)
