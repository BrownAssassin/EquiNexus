# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Audit component-selector targets and its frozen-holdout score."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_component_selector_oof import (  # noqa: E402
    load_prediction_rows,
    predictions_from_masks,
    target_marginal_pq,
)
from verify_official_scorer import compare_one_observation, official_namespace  # noqa: E402
from filament_baseline.data import load_dataset  # noqa: E402
from filament_baseline.predictor import PredictedInstance  # noqa: E402
from filament_baseline.scorer import score_observation  # noqa: E402
from filament_baseline.split import create_observation_split  # noqa: E402


def _global_subset(metrics: dict[str, Any]) -> dict[str, float | int]:
    fields = ("pq", "tp", "tp_iou_sum", "fp", "fn", "annotator_view_count")
    return {key: metrics["global"][key] for key in fields}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=REPO_ROOT / "artifacts/model-experiments/component-selector-official-pq-20261109-rle-fixed")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--target-observations", type=int, default=5)
    parser.add_argument("--official-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7001)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "artifacts/verification/component-selector-audit-20261109.json")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    data_root = args.data_root.resolve()
    if not run_dir.is_relative_to((REPO_ROOT / "artifacts").resolve()):
        raise ValueError("Run directory must remain inside artifacts/")
    if not data_root.is_relative_to((REPO_ROOT / "data").resolve()):
        raise ValueError("Dataset must remain inside data/")
    if not args.output.resolve().is_relative_to((REPO_ROOT / "artifacts").resolve()):
        raise ValueError("Audit output must remain inside artifacts/")

    dataset = load_dataset(data_root)
    split = create_observation_split(dataset, REPO_ROOT / "data/splits")
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    if metrics["split_sha256"] != split.sha256:
        raise RuntimeError("Component-selector result uses a different frozen split")

    training_names = sorted(split.training_names)
    holdout_names = sorted(split.validation_names)
    training_masks = load_prediction_rows(
        REPO_ROOT / "artifacts/model-experiments/oof-fusion-weight25-th035-a200/best-oof-predictions.csv",
        {Path(name).stem for name in training_names},
    )
    rng = np.random.default_rng(args.seed)
    target_names = sorted(
        rng.choice(training_names, size=min(args.target_observations, len(training_names)), replace=False).tolist()
    )
    checked_components = 0
    max_target_error = 0.0
    for file_name in target_names:
        stem = Path(file_name).stem
        rles = training_masks[stem]
        target_rows = target_marginal_pq(dataset, file_name, rles)
        predictions = [
            PredictedInstance(
                filament_id=f"{stem}_{index + 1}",
                image_stem=stem,
                area=int(mask_utils.area(rle)),
                rle=rle,
            )
            for index, rle in enumerate(rles)
        ]
        full_counts, _ = score_observation(dataset, file_name, predictions)
        for index, target in enumerate(target_rows):
            reduced = [item for item_index, item in enumerate(predictions) if item_index != index]
            reduced_counts, _ = score_observation(dataset, file_name, reduced)
            direct_delta = full_counts.pq - reduced_counts.pq
            error = abs(direct_delta - float(target["keep_delta_pq"]))
            max_target_error = max(max_target_error, error)
            checked_components += 1
            if error > 1e-7 or int(direct_delta >= -1e-12) != int(target["target_keep"]):
                raise AssertionError(
                    f"Marginal target mismatch: {file_name}, component {index + 1}, "
                    f"direct={direct_delta}, target={target}, error={error}"
                )

    selected_masks = load_prediction_rows(
        run_dir / "selector-holdout-predictions.csv",
        {Path(name).stem for name in holdout_names},
    )
    selected = predictions_from_masks(holdout_names, selected_masks)
    recorded = metrics["fit"]["holdout_selected"]
    per_observation_path = run_dir / "selector-holdout-per-observation.csv"
    per_observation = pd.read_csv(per_observation_path)
    if set(per_observation.file_name) != set(holdout_names) or len(per_observation) != len(holdout_names):
        raise RuntimeError("Per-observation holdout score coverage differs from the frozen holdout")
    for row in per_observation.itertuples():
        actual_count = len(selected_masks[Path(row.file_name).stem])
        if actual_count != int(row.predicted_instances_per_observation):
            raise AssertionError(f"Prediction row count mismatch for {row.file_name}")
    tp = int(per_observation.tp_edges.sum())
    fp = int(per_observation.fp.sum())
    fn = int(per_observation.fn.sum())
    tp_iou_sum = float(per_observation.tp_iou_sum.sum())
    denominator = tp + 0.5 * fp + 0.5 * fn
    aggregate = {
        "pq": tp_iou_sum / denominator if denominator else 0.0,
        "tp": tp,
        "tp_iou_sum": tp_iou_sum,
        "fp": fp,
        "fn": fn,
        "annotator_view_count": int(per_observation.annotator_views.sum()),
    }
    for key, value in aggregate.items():
        tolerance = 1e-7 if key == "tp_iou_sum" else (2e-8 if key == "pq" else 0.0)
        if not np.isclose(
            float(value), float(recorded["global"][key]), rtol=0.0, atol=tolerance
        ):
            raise AssertionError(
                f"Per-observation aggregate mismatch for {key}: "
                f"{value} != {recorded['global'][key]}"
            )

    # Recompute the full local scorer on a deterministic sample of holdout observations.
    rng = np.random.default_rng(args.seed + 1)
    sampled_holdout = sorted(
        rng.choice(holdout_names, size=min(args.official_samples, len(holdout_names)), replace=False).tolist()
    )
    local_sample_results = []
    for file_name in sampled_holdout:
        counts, _ = score_observation(dataset, file_name, selected[file_name])
        saved = per_observation.loc[per_observation.file_name == file_name].iloc[0]
        current = {
            "pq": counts.pq,
            "tp": counts.tp,
            "tp_iou_sum": counts.tp_iou_sum,
            "fp": counts.fp,
            "fn": counts.fn,
        }
        for key, value in current.items():
            expected_value = float(saved.pq if key == "pq" else saved.tp_iou_sum if key == "tp_iou_sum" else getattr(saved, "tp_edges" if key == "tp" else key))
            if not np.isclose(float(value), expected_value, rtol=0.0, atol=2e-8 if key in {"pq", "tp_iou_sum"} else 0.0):
                raise AssertionError(
                    f"Local scorer sample mismatch for {file_name} / {key}: "
                    f"recomputed={value}, saved={expected_value}"
                )
        local_sample_results.append({"file_name": file_name, **current})

    # Select nonempty prediction sets spanning the holdout's sorted order.
    nonempty = [name for name in holdout_names if selected[name] and len(selected[name]) <= 16]
    if not nonempty:
        raise RuntimeError("Selector produced no holdout predictions")
    sample_indices = np.linspace(
        0, len(nonempty) - 1, num=min(args.official_samples, len(nonempty)), dtype=int
    )
    official = official_namespace(REPO_ROOT / "data/reference/self-evaluation-notebook.ipynb")
    official_results = []
    for index in sample_indices:
        file_name = nonempty[int(index)]
        predictions = selected[file_name]
        result = compare_one_observation(
            dataset,
            file_name,
            predictions,
            official,
            full_prediction_count=len(predictions),
            selected_match_index=-1,
        )
        if float(result["absolute_difference"]) > 1e-6:
            raise AssertionError(f"Organizer scorer mismatch: {result}")
        official_results.append(result)

    result = {
        "status": "complete",
        "split_sha256": split.sha256,
        "target_observation_count": len(target_names),
        "target_component_count": checked_components,
        "target_max_absolute_error": max_target_error,
        "holdout_per_observation_aggregate": aggregate,
        "holdout_recorded": _global_subset(recorded),
        "local_scorer_sample": local_sample_results,
        "official_scorer_sample": official_results,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
