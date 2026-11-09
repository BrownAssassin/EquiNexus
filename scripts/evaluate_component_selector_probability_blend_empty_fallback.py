# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Evaluate a fixed highest-score fallback for empty selector outputs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from evaluate_component_selector_probability_blend_holdout import (  # noqa: E402
    EXTRA_TREES_WEIGHT,
    THRESHOLD,
    file_sha256,
    paired_bootstrap,
)
from run_component_selector_oof import load_prediction_rows, predictions_from_masks  # noqa: E402
from screen_component_selector_models import _fit_model  # noqa: E402
from filament_baseline.data import load_dataset  # noqa: E402
from filament_baseline.predictor import PredictedInstance  # noqa: E402
from filament_baseline.scorer import score_validation  # noqa: E402
from filament_baseline.split import create_observation_split  # noqa: E402


FROZEN_SPLIT_SHA256 = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"
DEFAULT_SELECTOR_RUN = REPO_ROOT / "artifacts/model-experiments/component-selector-official-pq-20261109-rle-fixed"
DEFAULT_MODEL_SCREEN = REPO_ROOT / "artifacts/model-experiments/component-selector-model-screen-20261109"
DEFAULT_OOF_PREDICTIONS = REPO_ROOT / "artifacts/model-experiments/oof-fusion-weight25-th035-a200/best-oof-predictions.csv"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/model-experiments/component-selector-probability-blend-empty-fallback-20261109"


def _fallback_empty_outputs(
    names: list[str],
    masks: dict[str, list[dict[str, Any]]],
    probability_by_key: dict[tuple[str, int], float],
    predictions: dict[str, list[PredictedInstance]],
) -> tuple[dict[str, list[PredictedInstance]], list[dict[str, Any]]]:
    result = {name: list(predictions[name]) for name in names}
    recovered: list[dict[str, Any]] = []
    for name in names:
        stem = Path(name).stem
        components = masks[stem]
        if result[name] or not components:
            continue
        scores = [probability_by_key[(name, index)] for index in range(1, len(components) + 1)]
        best_index = int(np.argmax(scores))
        rle = components[best_index]
        result[name] = [
            PredictedInstance(
                filament_id=f"{stem}_1",
                image_stem=stem,
                area=int(mask_utils.area(rle)),
                rle=rle,
            )
        ]
        recovered.append(
            {
                "file_name": name,
                "component_id": best_index + 1,
                "probability": float(scores[best_index]),
            }
        )
    return result, recovered


def _component_probabilities(frame: pd.DataFrame) -> dict[tuple[str, int], float]:
    keys = list(zip(frame.file_name.astype(str), frame.component_id.astype(int)))
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate observation/component probability keys")
    values = (
        (1.0 - EXTRA_TREES_WEIGHT) * frame["logistic_c1.0"].to_numpy(dtype=np.float64)
        + EXTRA_TREES_WEIGHT * frame["extra_trees"].to_numpy(dtype=np.float64)
    )
    if not np.isfinite(values).all():
        raise ValueError("Non-finite blend probabilities")
    return {key: float(value) for key, value in zip(keys, values)}


def _global(metrics: dict[str, Any]) -> dict[str, Any]:
    return metrics["global"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--selector-run", type=Path, default=DEFAULT_SELECTOR_RUN)
    parser.add_argument("--model-screen", type=Path, default=DEFAULT_MODEL_SCREEN)
    parser.add_argument("--oof-predictions", type=Path, default=DEFAULT_OOF_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    selector_run = args.selector_run.resolve()
    model_screen = args.model_screen.resolve()
    oof_predictions_path = args.oof_predictions.resolve()
    output_dir = args.output_dir.resolve()
    if not data_root.is_relative_to((REPO_ROOT / "data").resolve()):
        raise ValueError("Competition data must stay under data/")
    for path, label in (
        (selector_run, "selector run"),
        (model_screen, "OOF model screen"),
        (oof_predictions_path, "OOF predictions"),
    ):
        if not path.is_relative_to((REPO_ROOT / "artifacts").resolve()):
            raise ValueError(f"{label} must stay under artifacts/")
    if not output_dir.is_relative_to((REPO_ROOT / "artifacts/model-experiments").resolve()):
        raise ValueError("Evaluation outputs must stay under artifacts/model-experiments/")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)

    started = time.perf_counter()
    dataset = load_dataset(data_root)
    split = create_observation_split(dataset, REPO_ROOT / "data/splits")
    if split.sha256 != FROZEN_SPLIT_SHA256:
        raise RuntimeError(f"Frozen split changed: {split.sha256}")
    selector_metrics = json.loads((selector_run / "metrics.json").read_text(encoding="utf-8"))
    screen_metrics = json.loads((model_screen / "metrics.json").read_text(encoding="utf-8"))
    if selector_metrics.get("split_sha256") != split.sha256 or screen_metrics.get("split_sha256") != split.sha256:
        raise RuntimeError("Component-selector inputs use a different frozen split")
    if screen_metrics.get("holdout_used") is not False:
        raise RuntimeError("The OOF probability source must be certified holdout-free")
    if screen_metrics.get("test_images_or_labels_used") is not False:
        raise RuntimeError("The OOF probability screen reports test data access")

    train_features = pd.read_csv(selector_run / "oof-component-features-targets.csv")
    holdout_features = pd.read_csv(selector_run / "holdout-component-features.csv")
    oof_probabilities = pd.read_csv(model_screen / "oof-model-probabilities.csv")
    required = {"file_name", "component_id", "logistic_c1.0", "extra_trees"}
    if required - set(oof_probabilities.columns):
        raise ValueError(f"Missing OOF probabilities: {sorted(required - set(oof_probabilities.columns))}")
    expected_keys = set(zip(train_features.file_name.astype(str), train_features.component_id.astype(int)))
    oof_probability_keys = set(zip(oof_probabilities.file_name.astype(str), oof_probabilities.component_id.astype(int)))
    if expected_keys != oof_probability_keys:
        raise ValueError("OOF probabilities do not exactly cover selector training components")

    oof_names = sorted(split.training_names)
    oof_stems = {Path(name).stem for name in oof_names}
    oof_masks = load_prediction_rows(oof_predictions_path, oof_stems)
    oof_probability_by_key = _component_probabilities(oof_probabilities)
    oof_plain_predictions = predictions_from_masks(
        oof_names, oof_masks, oof_probability_by_key, THRESHOLD
    )
    oof_fallback_predictions, oof_recovered = _fallback_empty_outputs(
        oof_names, oof_masks, oof_probability_by_key, oof_plain_predictions
    )
    oof_plain_metrics, _ = score_validation(dataset, oof_plain_predictions, oof_names)
    oof_fallback_metrics, _ = score_validation(dataset, oof_fallback_predictions, oof_names)

    holdout_names = sorted(split.validation_names)
    holdout_stems = {Path(name).stem for name in holdout_names}
    holdout_masks = load_prediction_rows(selector_run / "baseline-holdout-predictions.csv", holdout_stems)
    logistic = _fit_model("logistic_c1.0", train_features, holdout_features)
    trees = _fit_model("extra_trees", train_features, holdout_features)
    holdout_keys = list(zip(holdout_features.file_name.astype(str), holdout_features.component_id.astype(int)))
    holdout_probability = (1.0 - EXTRA_TREES_WEIGHT) * logistic + EXTRA_TREES_WEIGHT * trees
    holdout_probability_by_key = {
        key: float(value) for key, value in zip(holdout_keys, holdout_probability)
    }
    holdout_plain_predictions = predictions_from_masks(
        holdout_names, holdout_masks, holdout_probability_by_key, THRESHOLD
    )
    holdout_fallback_predictions, holdout_recovered = _fallback_empty_outputs(
        holdout_names, holdout_masks, holdout_probability_by_key, holdout_plain_predictions
    )
    holdout_plain_metrics, holdout_plain_rows = score_validation(
        dataset, holdout_plain_predictions, holdout_names
    )
    holdout_fallback_metrics, holdout_fallback_rows = score_validation(
        dataset, holdout_fallback_predictions, holdout_names
    )
    bootstrap = paired_bootstrap(holdout_plain_rows, holdout_fallback_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split_sha256": split.sha256,
        "oof_manifest_sha256": selector_metrics.get("oof_manifest_sha256"),
        "selection": {
            "logistic_weight": 1.0 - EXTRA_TREES_WEIGHT,
            "extra_trees_weight": EXTRA_TREES_WEIGHT,
            "threshold": THRESHOLD,
            "fallback": "when the selected set is empty but base components exist, retain the highest blend-probability component",
            "fallback_rule_selected_using_holdout": False,
        },
        "oof": {
            "observation_count": len(oof_names),
            "fallback_observations": oof_recovered,
            "plain": _global(oof_plain_metrics),
            "fallback": _global(oof_fallback_metrics),
        },
        "holdout": {
            "observation_count": len(holdout_names),
            "fallback_observations": holdout_recovered,
            "plain": _global(holdout_plain_metrics),
            "fallback": _global(holdout_fallback_metrics),
            "paired_observation_bootstrap": bootstrap,
        },
        "data_boundary": {
            "training_labels": "competition-provided training annotations only",
            "test_images_or_labels_used": False,
            "frozen_holdout_used_for_training_or_selection": False,
        },
        "input_sha256": {
            "oof_probabilities": file_sha256(model_screen / "oof-model-probabilities.csv"),
            "oof_component_features": file_sha256(selector_run / "oof-component-features-targets.csv"),
            "holdout_component_features": file_sha256(selector_run / "holdout-component-features.csv"),
            "oof_base_predictions": file_sha256(oof_predictions_path),
            "holdout_base_predictions": file_sha256(selector_run / "baseline-holdout-predictions.csv"),
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
