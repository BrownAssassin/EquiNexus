# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Screen convex blends of cross-fitted component-selector probabilities."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_component_selector_oof import (  # noqa: E402
    load_prediction_rows,
    predictions_from_masks,
)
from filament_baseline.data import load_dataset  # noqa: E402
from filament_baseline.scorer import score_validation  # noqa: E402
from filament_baseline.split import create_observation_split  # noqa: E402


FROZEN_SPLIT_SHA256 = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"
DEFAULT_MODEL_SCREEN = REPO_ROOT / "artifacts/model-experiments/component-selector-model-screen-20261109"
DEFAULT_SELECTOR_RUN = REPO_ROOT / "artifacts/model-experiments/component-selector-official-pq-20261109-rle-fixed"
DEFAULT_OOF_PREDICTIONS = REPO_ROOT / "artifacts/model-experiments/oof-fusion-weight25-th035-a200/best-oof-predictions.csv"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path,
        default=REPO_ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026",
    )
    parser.add_argument("--model-screen", type=Path, default=DEFAULT_MODEL_SCREEN)
    parser.add_argument("--selector-run", type=Path, default=DEFAULT_SELECTOR_RUN)
    parser.add_argument("--oof-predictions", type=Path, default=DEFAULT_OOF_PREDICTIONS)
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "artifacts/model-experiments/component-selector-probability-blends-20261109",
    )
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    model_screen = args.model_screen.resolve()
    selector_run = args.selector_run.resolve()
    oof_predictions_path = args.oof_predictions.resolve()
    output_dir = args.output_dir.resolve()
    if not data_root.is_relative_to((REPO_ROOT / "data").resolve()):
        raise ValueError("Competition data must stay under data/")
    for path, label in (
        (model_screen, "model screen"),
        (selector_run, "selector run"),
        (oof_predictions_path, "OOF predictions"),
    ):
        if not path.is_relative_to((REPO_ROOT / "artifacts").resolve()):
            raise ValueError(f"{label} must stay under artifacts/")
    if not output_dir.is_relative_to((REPO_ROOT / "artifacts/model-experiments").resolve()):
        raise ValueError("Experiment outputs must stay under artifacts/model-experiments/")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)

    started = time.perf_counter()
    dataset = load_dataset(data_root)
    split = create_observation_split(dataset, REPO_ROOT / "data/splits")
    selector_metrics = json.loads((selector_run / "metrics.json").read_text(encoding="utf-8"))
    model_metrics = json.loads((model_screen / "metrics.json").read_text(encoding="utf-8"))
    if split.sha256 != FROZEN_SPLIT_SHA256:
        raise RuntimeError(f"Frozen competition split changed: {split.sha256}")
    if selector_metrics.get("split_sha256") != split.sha256:
        raise RuntimeError("Selector feature source uses a different frozen split")
    if model_metrics.get("split_sha256") != split.sha256:
        raise RuntimeError("Selector probability source uses a different frozen split")
    if model_metrics.get("holdout_used") is not False:
        raise RuntimeError("Model screen metadata does not confirm holdout exclusion")

    names = sorted(split.training_names)
    expected_stems = {Path(name).stem for name in names}
    masks = load_prediction_rows(oof_predictions_path, expected_stems)
    feature_rows = pd.read_csv(selector_run / "oof-component-features-targets.csv")
    probability_rows = pd.read_csv(model_screen / "oof-model-probabilities.csv")
    required = {"file_name", "component_id", "logistic_c1.0", "extra_trees"}
    if required - set(probability_rows.columns):
        raise ValueError(f"Missing probability columns: {sorted(required - set(probability_rows.columns))}")
    if set(feature_rows["file_name"].astype(str)) - set(names):
        raise ValueError("Feature table includes observations outside the OOF training partition")
    feature_keys = set(zip(
        feature_rows["file_name"].astype(str),
        feature_rows["component_id"].astype(int),
    ))
    probability_keys = set(zip(
        probability_rows["file_name"].astype(str),
        probability_rows["component_id"].astype(int),
    ))
    if probability_keys != feature_keys:
        raise ValueError("Model probabilities and component features have different keys")
    if "oof_fold" not in feature_rows or set(feature_rows["oof_fold"].astype(int)) != set(range(5)):
        raise ValueError("Expected probabilities from all five frozen OOF folds")

    keys = list(zip(
        probability_rows["file_name"].astype(str),
        probability_rows["component_id"].astype(int),
    ))
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate file/component keys in model-screen probabilities")
    logistic = probability_rows["logistic_c1.0"].to_numpy(dtype=np.float64)
    extra_trees = probability_rows["extra_trees"].to_numpy(dtype=np.float64)
    if not np.isfinite(logistic).all() or not np.isfinite(extra_trees).all():
        raise ValueError("Non-finite component selector probabilities")

    output_dir.mkdir(parents=True, exist_ok=True)
    weight_grid = np.linspace(0.0, 1.0, 11)
    threshold_grid = np.round(np.arange(0.25, 0.5001, 0.025), 3)
    rows: list[dict[str, Any]] = []
    baseline = selector_metrics["fit"]["oof_selected"]["global"]["pq"]

    for extra_weight in weight_grid:
        blended = (1.0 - extra_weight) * logistic + extra_weight * extra_trees
        keyed = {
            key: float(value)
            for key, value in zip(keys, blended)
        }
        for threshold in threshold_grid:
            predictions = predictions_from_masks(names, masks, keyed, float(threshold))
            metrics, _ = score_validation(dataset, predictions, names)
            global_metrics = metrics["global"]
            row = {
                "extra_trees_weight": float(extra_weight),
                "logistic_weight": float(1.0 - extra_weight),
                "threshold": float(threshold),
                "pq": float(global_metrics["pq"]),
                "tp": int(global_metrics["tp"]),
                "fp": int(global_metrics["fp"]),
                "fn": int(global_metrics["fn"]),
                "prediction_count": int(global_metrics["prediction_count_across_annotator_views"]),
            }
            rows.append(row)
        print(
            f"screened ExtraTrees weight {extra_weight:.1f}; "
            f"best PQ={max(row['pq'] for row in rows if row['extra_trees_weight'] == extra_weight):.6f}",
            flush=True,
        )

    results = sorted(rows, key=lambda row: (row["pq"], -row["extra_trees_weight"]), reverse=True)
    pd.DataFrame(rows).to_csv(output_dir / "blend-sweep.csv", index=False)
    report = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split_sha256": split.sha256,
        "oof_manifest_sha256": selector_metrics.get("oof_manifest_sha256"),
        "oof_observations": len(names),
        "component_rows": len(keys),
        "holdout_used": False,
        "test_images_or_labels_used": False,
        "training_labels": "competition-provided training annotations only",
        "metric_note": "Cross-fitted grouped OOF selector probabilities are blended and ranked with the official scorer; ranking is selection-aware.",
        "baseline_logistic_c1_threshold_040_pq": float(baseline),
        "probability_blend": "(1 - extra_trees_weight) * logistic_c1.0 + extra_trees_weight * extra_trees",
        "weight_grid": [float(value) for value in weight_grid],
        "threshold_grid": [float(value) for value in threshold_grid],
        "top_candidates": results[:15],
        "input_sha256": {
            "model_screen_probabilities": file_sha256(model_screen / "oof-model-probabilities.csv"),
            "selector_features": file_sha256(selector_run / "oof-component-features-targets.csv"),
            "base_oof_predictions": file_sha256(oof_predictions_path),
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"baseline_pq": baseline, "top_candidates": results[:10], "output": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
