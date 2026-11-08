# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from .data import load_dataset
from .inference import InstanceConfig, instances_from_probabilities
from .runner import (
    _resolve_data_root,
    _write_observation_scores,
    _write_overlays,
    _write_predictions,
)
from .scorer import score_validation
from .split import create_observation_split
from .visuals import write_comparison_overlays


def run_instance_structure_sweep(
    *,
    repo_root: str | Path,
    data_root: str | Path | None,
    run_root: str | Path,
    foreground_threshold: float = 0.55,
    boundary_threshold: float = 0.60,
    boundary_weight: float = 1.5,
    boundary_weights: tuple[float, ...] | None = None,
    seed_threshold: float = 0.45,
) -> list[dict[str, Any]]:
    repo_root = Path(repo_root).resolve()
    run_root = Path(run_root).resolve()
    dataset = load_dataset(_resolve_data_root(data_root, repo_root))
    validation_config = json.loads(
        (repo_root / "configs" / "validation-baseline.json").read_text(
            encoding="utf-8"
        )
    )
    split = create_observation_split(
        dataset,
        repo_root / "data" / "splits",
        seed=int(validation_config["seed"]),
        validation_fraction=float(validation_config["validation_fraction"]),
    )
    if split.sha256 != validation_config["expected_split_sha256"]:
        raise RuntimeError("The structure sweep requires the frozen validation split")

    validation_names = list(split.validation_names)
    probability_dir = run_root / "probabilities"
    for file_name in validation_names:
        cache_path = probability_dir / f"{Path(file_name).stem}.npz"
        if not cache_path.is_file():
            raise FileNotFoundError(f"Missing probability cache: {cache_path}")

    weights = boundary_weights or (boundary_weight,)
    configurations = [
        InstanceConfig(
            foreground_threshold=foreground_threshold,
            boundary_threshold=boundary_threshold,
            boundary_weight=weight,
            seed_threshold=seed_threshold,
            seed_min_area=seed_area,
            min_instance_area=instance_area,
        )
        for weight, (instance_area, seed_area) in itertools.product(
            weights,
            itertools.product(
                (50, 100, 200, 300, 400, 500, 600, 700, 800, 1_000, 1_200),
                (8, 32, 128),
            ),
        )
    ]
    results = []
    best_pq = -1.0
    for index, candidate in enumerate(configurations, start=1):
        predictions = {}
        for file_name in validation_names:
            with np.load(probability_dir / f"{Path(file_name).stem}.npz") as cache:
                predictions[file_name] = instances_from_probabilities(
                    cache["foreground"].astype(np.float32),
                    cache["boundary"].astype(np.float32),
                    file_name,
                    candidate,
                )
        metrics, observations = score_validation(
            dataset,
            predictions,
            validation_names,
            iou_threshold=float(validation_config["iou_threshold"]),
            diagnostic_overlap_threshold=float(
                validation_config["diagnostic_overlap_threshold"]
            ),
        )
        global_metrics = metrics["global"]
        row = {
            "configuration": candidate.__dict__,
            "pq": global_metrics["pq"],
            "tp": global_metrics["tp"],
            "fp": global_metrics["fp"],
            "fn": global_metrics["fn"],
            "prediction_count_across_annotator_views": global_metrics[
                "prediction_count_across_annotator_views"
            ],
            "metrics": metrics,
        }
        results.append(row)
        print(
            f"[{index}/{len(configurations)}] min_area={candidate.min_instance_area} "
            f"seed_area={candidate.seed_min_area} PQ={row['pq']:.6f} "
            f"TP={row['tp']} FP={row['fp']} FN={row['fn']}",
            flush=True,
        )
        if row["pq"] > best_pq:
            best_pq = row["pq"]
            best_dir = run_root / "best-instance-structure"
            best_dir.mkdir(parents=True, exist_ok=True)
            (best_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "split_sha256": split.sha256,
                        "configuration": candidate.__dict__,
                        "metrics": metrics,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            _write_predictions(best_dir / "predictions.csv", predictions)
            _write_observation_scores(
                best_dir / "per_observation.csv", observations, predictions
            )
            _write_overlays(dataset, predictions, best_dir / "overlays", 12)
            write_comparison_overlays(
                dataset, predictions, best_dir / "comparison-overlays", 12
            )

    results.sort(key=lambda item: item["pq"], reverse=True)
    (run_root / "instance-structure-sweep.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (run_root / "instance-structure-sweep.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fields = (
            *InstanceConfig().__dict__.keys(),
            "pq",
            "tp",
            "fp",
            "fn",
            "prediction_count_across_annotator_views",
        )
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in results:
            config_values = row["configuration"]
            writer.writerow(
                {
                    **config_values,
                    **{
                        field: row[field]
                        for field in fields[len(config_values) :]
                    },
                }
            )
    return results
