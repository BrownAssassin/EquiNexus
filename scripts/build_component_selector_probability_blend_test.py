# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Build a full-fit test CSV from the OOF-selected selector probability blend."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from evaluate_component_selector_probability_blend_holdout import (  # noqa: E402
    EXTRA_TREES_WEIGHT,
    THRESHOLD,
    file_sha256,
)
from evaluate_component_selector_probability_blend_empty_fallback import (  # noqa: E402
    _fallback_empty_outputs,
)
from run_component_selector_oof import load_prediction_rows, predictions_from_masks  # noqa: E402
from screen_component_selector_models import _fit_model  # noqa: E402
from filament_baseline.data import load_dataset  # noqa: E402
from filament_baseline.runner import _write_predictions  # noqa: E402
from filament_baseline.runtime import git_info, hardware_info, package_versions  # noqa: E402
from filament_baseline.split import create_observation_split  # noqa: E402
from filament_baseline.submission import _check_submission  # noqa: E402


FROZEN_SPLIT_SHA256 = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"
DEFAULT_SELECTOR_RUN = REPO_ROOT / "artifacts/model-experiments/component-selector-official-pq-20261109-rle-fixed"
DEFAULT_MODEL_SCREEN = REPO_ROOT / "artifacts/model-experiments/component-selector-model-screen-20261109"
DEFAULT_BLEND_SCREEN = REPO_ROOT / "artifacts/model-experiments/component-selector-probability-blends-20261109"
DEFAULT_HOLDOUT_CHECK = REPO_ROOT / "artifacts/model-experiments/component-selector-probability-blend-holdout-20261109"
DEFAULT_TEST_FEATURE_RUN = REPO_ROOT / "artifacts/submissions/component-selector-fullfit-test-20261109"
DEFAULT_BASE_SUBMISSION = REPO_ROOT / "artifacts/submissions/final-yolo11s-fullfit-p25-20261108/submission.csv"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/submissions/component-selector-probability-blend-fullfit-test-20261109"
DEFAULT_EMPTY_FALLBACK_METRICS = REPO_ROOT / "artifacts/model-experiments/component-selector-probability-blend-empty-fallback-20261109/metrics.json"


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--selector-run", type=Path, default=DEFAULT_SELECTOR_RUN)
    parser.add_argument("--model-screen", type=Path, default=DEFAULT_MODEL_SCREEN)
    parser.add_argument("--blend-screen", type=Path, default=DEFAULT_BLEND_SCREEN)
    parser.add_argument("--holdout-check", type=Path, default=DEFAULT_HOLDOUT_CHECK)
    parser.add_argument("--test-feature-run", type=Path, default=DEFAULT_TEST_FEATURE_RUN)
    parser.add_argument("--base-submission", type=Path, default=DEFAULT_BASE_SUBMISSION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--empty-fallback", action="store_true", help="keep the highest-scoring base component when selection is empty")
    parser.add_argument("--empty-fallback-metrics", type=Path, default=DEFAULT_EMPTY_FALLBACK_METRICS)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    selector_run = args.selector_run.resolve()
    model_screen = args.model_screen.resolve()
    blend_screen = args.blend_screen.resolve()
    holdout_check = args.holdout_check.resolve()
    test_feature_run = args.test_feature_run.resolve()
    base_submission = args.base_submission.resolve()
    output_dir = args.output_dir.resolve()
    empty_fallback_metrics_path = args.empty_fallback_metrics.resolve()
    if not data_root.is_relative_to((REPO_ROOT / "data").resolve()):
        raise ValueError("Competition data must stay under data/")
    for path, label in (
        (selector_run, "selector run"),
        (model_screen, "OOF model screen"),
        (blend_screen, "OOF blend screen"),
        (holdout_check, "fixed-holdout check"),
        (test_feature_run, "test feature run"),
        (base_submission, "base submission"),
    ):
        if not path.is_relative_to((REPO_ROOT / "artifacts").resolve()):
            raise ValueError(f"{label} must stay under artifacts/")
    if not output_dir.is_relative_to((REPO_ROOT / "artifacts/submissions").resolve()):
        raise ValueError("Submission outputs must stay under artifacts/submissions/")
    if args.empty_fallback and not empty_fallback_metrics_path.is_relative_to((REPO_ROOT / "artifacts/model-experiments").resolve()):
        raise ValueError("Empty-fallback evidence must stay under artifacts/model-experiments/")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)

    started = time.perf_counter()
    dataset = load_dataset(data_root)
    split = create_observation_split(dataset, REPO_ROOT / "data/splits")
    if split.sha256 != FROZEN_SPLIT_SHA256:
        raise RuntimeError(f"Frozen split changed: {split.sha256}")
    selector_metrics = json.loads((selector_run / "metrics.json").read_text(encoding="utf-8"))
    model_metrics = json.loads((model_screen / "metrics.json").read_text(encoding="utf-8"))
    blend_metrics = json.loads((blend_screen / "metrics.json").read_text(encoding="utf-8"))
    holdout_metrics = json.loads((holdout_check / "metrics.json").read_text(encoding="utf-8"))
    test_metrics = json.loads((test_feature_run / "metrics.json").read_text(encoding="utf-8"))
    for metrics, label in (
        (selector_metrics, "selector"),
        (model_metrics, "model screen"),
        (blend_metrics, "blend screen"),
        (holdout_metrics, "holdout check"),
        (test_metrics, "test feature run"),
    ):
        if metrics.get("split_sha256") != split.sha256:
            raise RuntimeError(f"{label} uses a different frozen split")
    if model_metrics.get("holdout_used") is not False or blend_metrics.get("holdout_used") is not False:
        raise RuntimeError("The probability blend was not screened exclusively on OOF rows")
    if holdout_metrics.get("test_images_or_labels_used") is not False:
        raise RuntimeError("The fixed-holdout check reports test image or label use")
    if test_metrics.get("data_boundary", {}).get("test_labels_used") is not False:
        raise RuntimeError("Test feature source reports test-label access")

    top = blend_metrics["top_candidates"][0]
    if abs(float(top["extra_trees_weight"]) - EXTRA_TREES_WEIGHT) > 1e-8 or abs(float(top["threshold"]) - THRESHOLD) > 1e-8:
        raise RuntimeError("Configured blend does not match the OOF-screened top candidate")
    holdout_candidate = holdout_metrics["candidate"]
    if (
        abs(float(holdout_candidate["extra_trees_weight"]) - EXTRA_TREES_WEIGHT) > 1e-8
        or abs(float(holdout_candidate["threshold"]) - THRESHOLD) > 1e-8
    ):
        raise RuntimeError("Configured blend does not match the fixed-holdout check")
    if test_metrics.get("source_hashes", {}).get("base_submission") != file_sha256(base_submission):
        raise RuntimeError("Test features were built from a different base submission")

    test_names = sorted(path.name for path in dataset.test_image_dir.iterdir() if path.is_file())
    if len(test_names) != 180:
        raise RuntimeError(f"Expected 180 supplied H-alpha test images, got {len(test_names)}")
    test_stems = {Path(name).stem for name in test_names}
    masks = load_prediction_rows(base_submission, test_stems)
    test_features_path = test_feature_run / "test-component-features.csv"
    test_features = pd.read_csv(test_features_path)
    train_features_path = selector_run / "oof-component-features-targets.csv"
    train_features = pd.read_csv(train_features_path)
    if int(test_metrics.get("selection", {}).get("component_count", -1)) != len(test_features):
        raise RuntimeError("Cached test feature rows differ from the source run metadata")
    if set(test_features.file_name.astype(str)) - set(test_names):
        raise ValueError("Test component features contain unknown test observations")
    if len(test_features) != sum(len(masks[Path(name).stem]) for name in test_names):
        raise RuntimeError("Test feature rows do not align with base prediction components")

    logistic_probability = _fit_model("logistic_c1.0", train_features, test_features)
    trees_probability = _fit_model("extra_trees", train_features, test_features)
    if len(logistic_probability) != len(test_features) or len(trees_probability) != len(test_features):
        raise RuntimeError("Fitted selector models returned an unexpected test row count")
    blended_probability = (
        (1.0 - EXTRA_TREES_WEIGHT) * logistic_probability
        + EXTRA_TREES_WEIGHT * trees_probability
    )
    keys = zip(test_features.file_name.astype(str), test_features.component_id.astype(int))
    probability_by_key = {key: float(value) for key, value in zip(keys, blended_probability)}
    predictions = predictions_from_masks(test_names, masks, probability_by_key, THRESHOLD)
    fallback_metrics = None
    fallback_observations = []
    if args.empty_fallback:
        fallback_metrics = json.loads(empty_fallback_metrics_path.read_text(encoding="utf-8"))
        expected_fallback_inputs = {
            "oof_probabilities": file_sha256(model_screen / "oof-model-probabilities.csv"),
            "oof_component_features": file_sha256(train_features_path),
            "holdout_component_features": file_sha256(selector_run / "holdout-component-features.csv"),
            "oof_base_predictions": file_sha256(REPO_ROOT / "artifacts/model-experiments/oof-fusion-weight25-th035-a200/best-oof-predictions.csv"),
            "holdout_base_predictions": file_sha256(selector_run / "baseline-holdout-predictions.csv"),
        }
        if fallback_metrics.get("split_sha256") != split.sha256:
            raise RuntimeError("Empty-fallback evidence uses a different frozen split")
        if fallback_metrics.get("selection", {}).get("threshold") != THRESHOLD or fallback_metrics.get("selection", {}).get("extra_trees_weight") != EXTRA_TREES_WEIGHT:
            raise RuntimeError("Empty-fallback evidence uses a different blend configuration")
        if fallback_metrics.get("input_sha256") != expected_fallback_inputs:
            raise RuntimeError("Empty-fallback evidence does not match the selected model inputs")
        if fallback_metrics.get("data_boundary", {}).get("test_images_or_labels_used") is not False:
            raise RuntimeError("Empty-fallback evidence reports test data access")
        if fallback_metrics.get("selection", {}).get("fallback_rule_selected_using_holdout") is not False:
            raise RuntimeError("Empty-fallback rule was selected using holdout evidence")
        predictions, fallback_observations = _fallback_empty_outputs(
            test_names, masks, probability_by_key, predictions
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = output_dir / "submission.csv"
    _write_predictions(submission_path, predictions)
    coverage = _check_submission(submission_path, data_root, test_names, predictions)
    test_features.to_csv(output_dir / "test-component-features.csv", index=False)

    source_files = {
        "selector_oof_features": train_features_path,
        "test_component_features": test_features_path,
        "base_submission": base_submission,
        "selector_metrics": selector_run / "metrics.json",
        "model_screen_metrics": model_screen / "metrics.json",
        "blend_screen_metrics": blend_screen / "metrics.json",
        "fixed_holdout_metrics": holdout_check / "metrics.json",
        "test_feature_metrics": test_feature_run / "metrics.json",
        "scorer": REPO_ROOT / "src/filament_baseline/scorer.py",
    }
    if args.empty_fallback:
        source_files["empty_fallback_metrics"] = empty_fallback_metrics_path
    report: dict[str, Any] = {
        "status": "complete",
        "run_name": output_dir.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split_sha256": split.sha256,
        "oof_manifest_sha256": selector_metrics.get("oof_manifest_sha256"),
        "data_boundary": {
            "training_labels": "competition-provided training annotations only",
            "inference_inputs": "180 supplied competition H-alpha test images only",
            "test_labels_used": False,
            "test_images_used_for_training": False,
            "selector_fit_rows": "cross-fitted OOF training observations only",
        },
        "selection": {
            "method": "0.70 standardized L2 logistic probability + 0.30 ExtraTrees probability",
            "threshold": THRESHOLD,
            "extra_trees_weight": EXTRA_TREES_WEIGHT,
            "oof_blend_pq": float(top["pq"]),
            "oof_screen_pq": float(
                fallback_metrics["oof"]["fallback"]["pq"] if fallback_metrics
                else top["pq"]
            ),
            "holdout_blend_pq": float(holdout_candidate["global_metrics"]["pq"]),
            "holdout_pq": float(
                fallback_metrics["holdout"]["fallback"]["pq"] if fallback_metrics
                else holdout_candidate["global_metrics"]["pq"]
            ),
            "holdout_paired_delta_ci_95": holdout_metrics["paired_observation_bootstrap"]["paired_delta_candidate_minus_baseline"]["bootstrap_ci_95"],
            "empty_fallback_holdout_delta": (
                fallback_metrics["holdout"]["paired_observation_bootstrap"]["paired_delta_candidate_minus_baseline"]
                if fallback_metrics else None
            ),
            "base_prediction_count": int(sum(len(items) for items in masks.values())),
            "selected_prediction_count": int(sum(len(items) for items in predictions.values())),
            "training_feature_rows": int(len(train_features)),
            "test_feature_rows": int(len(test_features)),
            "empty_fallback": bool(args.empty_fallback),
            "empty_fallback_observations": fallback_observations,
        },
        "coverage_validation": coverage,
        "submission_sha256": file_sha256(submission_path),
        "input_sha256": {name: file_sha256(path) for name, path in source_files.items()},
        "source_code_sha256": file_sha256(Path(__file__).resolve()),
        "source_repository": git_info(REPO_ROOT),
        "runtime": {
            "device_used": "cpu",
            "hardware": hardware_info(),
            "package_versions": package_versions(),
            "peak_gpu_memory_bytes": 0,
        },
        "runtime_seconds": time.perf_counter() - started,
        "artifacts": {
            "submission": "submission.csv",
            "test_component_features": "test-component-features.csv",
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "selection": report["selection"], "coverage": coverage, "sha256": report["submission_sha256"], "output": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
