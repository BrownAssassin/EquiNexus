# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import hashlib
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools import mask as mask_utils

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from filament_baseline.data import load_dataset
from filament_baseline.predictor import PredictedInstance
from filament_baseline.scorer import score_validation
from filament_baseline.split import create_observation_split


def _decode_counts(value: str | bytes) -> bytes:
    return value if isinstance(value, bytes) else value.encode("ascii")


def score_detector_predictions(
    dataset,
    validation_names: tuple[str, ...],
    predictions: dict[str, list[dict[str, Any]]],
    *,
    confidence_threshold: float,
    minimum_area: int,
) -> tuple[dict[str, Any], list[Any], int]:
    by_name: dict[str, list[PredictedInstance]] = {}
    prediction_count = 0
    for file_name in validation_names:
        stem = Path(file_name).stem
        retained = []
        for index, item in enumerate(predictions.get(stem, []), start=1):
            confidence = float(item["score"])
            rle = {"size": [2048, 2048], "counts": _decode_counts(item["counts"])}
            area = int(mask_utils.area(rle))
            if confidence < confidence_threshold or area < minimum_area:
                continue
            retained.append(
                PredictedInstance(
                    filament_id=f"{stem}_{index}",
                    image_stem=stem,
                    area=area,
                    rle=rle,
                )
            )
        by_name[file_name] = retained
        prediction_count += len(retained)
    metrics, rows = score_validation(
        dataset,
        by_name,
        validation_names,
        iou_threshold=0.5,
        diagnostic_overlap_threshold=0.0,
    )
    return metrics, rows, prediction_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score full-resolution YOLO masks against the frozen official holdout."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--data-root", type=Path, default=REPO_ROOT / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026"
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence-thresholds", default="0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5")
    parser.add_argument("--minimum-areas", default="0,100,200,400,600")
    args = parser.parse_args()

    repo_root = args.repo_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    predictions_path = args.predictions.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not output_path.is_relative_to(repo_root):
        raise ValueError(f"Score output must stay inside the workspace: {output_path}")
    config = json.loads((repo_root / "configs" / "validation-baseline.json").read_text())
    dataset = load_dataset(data_root)
    split = create_observation_split(
        dataset,
        repo_root / "data" / "splits",
        seed=int(config["seed"]),
        validation_fraction=float(config["validation_fraction"]),
    )
    if split.sha256 != config["expected_split_sha256"]:
        raise RuntimeError(f"Frozen split changed: {split.sha256}")
    prediction_payload = json.loads(predictions_path.read_text(encoding="utf-8"))
    confidence_thresholds = [float(value) for value in args.confidence_thresholds.split(",")]
    minimum_areas = [int(value) for value in args.minimum_areas.split(",")]
    if any(value < 0 or value > 1 for value in confidence_thresholds):
        raise ValueError("Confidence thresholds must be between 0 and 1")
    if any(value < 0 for value in minimum_areas):
        raise ValueError("Minimum areas must be nonnegative")

    results = []
    best_rows = None
    best_observations = None
    for confidence in confidence_thresholds:
        for minimum_area in minimum_areas:
            metrics, rows, prediction_count = score_detector_predictions(
                dataset,
                split.validation_names,
                prediction_payload,
                confidence_threshold=confidence,
                minimum_area=minimum_area,
            )
            row = {
                "confidence_threshold": confidence,
                "minimum_area": minimum_area,
                "prediction_count": prediction_count,
                "metrics": metrics,
            }
            results.append(row)
            print(
                f"confidence={confidence:.2f} min_area={minimum_area} "
                f"PQ={metrics['global']['pq']:.6f} "
                f"TP={metrics['global']['tp']} FP={metrics['global']['fp']} "
                f"FN={metrics['global']['fn']}",
                flush=True,
            )
            if best_rows is None or metrics["global"]["pq"] > best_rows["metrics"]["global"]["pq"]:
                best_rows = row
                best_observations = [
                    {
                        "file_name": item.file_name,
                        "year": item.year,
                        "site": item.site,
                        "annotator_views": item.annotator_views,
                        "gt_count": item.gt_count,
                        "prediction_count": item.prediction_count,
                        "tp": item.tp,
                        "tp_iou_sum": item.tp_iou_sum,
                        "fp": item.fp,
                        "fn": item.fn,
                        "pq": item.pq,
                    }
                    for item in rows
                ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "split_sha256": split.sha256,
        "validation_observations": len(split.validation_names),
        "ground_truth_source": "competition-provided training annotations only",
        "prediction_source": str(predictions_path),
        "scorer_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "best": best_rows,
        "per_observation": best_observations,
        "sweep": results,
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Best: {json.dumps(best_rows, sort_keys=True)}")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
