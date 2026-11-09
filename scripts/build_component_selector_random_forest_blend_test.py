# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Build a hashable test CSV for the OOF-selected random-forest blend."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from evaluate_component_selector_random_forest_blend import (  # noqa: E402
    CORE_EXTRA_TREES_WEIGHT,
    RF_WEIGHT,
    THRESHOLD,
)
from evaluate_component_selector_probability_blend_holdout import file_sha256  # noqa: E402
from run_component_selector_oof import (  # noqa: E402
    FEATURES,
    load_prediction_rows,
    predictions_from_masks,
)
from screen_component_selector_models import _fit_model  # noqa: E402
from filament_baseline.data import load_dataset  # noqa: E402
from filament_baseline.runner import _write_predictions  # noqa: E402
from filament_baseline.runtime import git_info, hardware_info, package_versions  # noqa: E402
from filament_baseline.split import create_observation_split  # noqa: E402
from filament_baseline.submission import _check_submission  # noqa: E402


FROZEN_SPLIT_SHA256 = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"
SELECTOR_RUN = REPO_ROOT / "artifacts/model-experiments/component-selector-official-pq-20261109-rle-fixed"
MODEL_SCREEN = REPO_ROOT / "artifacts/model-experiments/component-selector-model-screen-20261109"
RF_SCREEN = REPO_ROOT / "artifacts/model-experiments/component-selector-random-forest-oof-20261109"
RF_BLEND_SCREEN = REPO_ROOT / "artifacts/model-experiments/component-selector-rf-blend-oof-repro-20261109"
RF_HOLDOUT = REPO_ROOT / "artifacts/model-experiments/component-selector-rf-blend-holdout-20261109-final"
TEST_FEATURE_RUN = REPO_ROOT / "artifacts/submissions/component-selector-probability-blend-fullfit-test-20261109-rerun"
BASE_SUBMISSION = REPO_ROOT / "artifacts/submissions/final-yolo11s-fullfit-p25-20261108/submission.csv"
OUTPUT_DIR = REPO_ROOT / "artifacts/submissions/component-selector-rf-blend-fullfit-test-20261109-final"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    data_root, output_dir = args.data_root.resolve(), args.output_dir.resolve()
    if not data_root.is_relative_to((REPO_ROOT / "data").resolve()):
        raise ValueError("Competition data must remain under data/")
    if not output_dir.is_relative_to((REPO_ROOT / "artifacts/submissions").resolve()):
        raise ValueError("Submission outputs must remain under artifacts/submissions/")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)

    started = time.perf_counter()
    dataset = load_dataset(data_root)
    split = create_observation_split(dataset, REPO_ROOT / "data/splits")
    if split.sha256 != FROZEN_SPLIT_SHA256:
        raise RuntimeError("Frozen grouped split digest changed")
    selector_metrics = json.loads((SELECTOR_RUN / "metrics.json").read_text(encoding="utf-8"))
    model_metrics = json.loads((MODEL_SCREEN / "metrics.json").read_text(encoding="utf-8"))
    rf_metrics = json.loads((RF_SCREEN / "metrics.json").read_text(encoding="utf-8"))
    blend_metrics = json.loads((RF_BLEND_SCREEN / "metrics.json").read_text(encoding="utf-8"))
    holdout_metrics = json.loads((RF_HOLDOUT / "metrics.json").read_text(encoding="utf-8"))
    for metrics, label in (
        (selector_metrics, "selector"), (model_metrics, "classifier"),
        (rf_metrics, "forest"), (blend_metrics, "blend"), (holdout_metrics, "holdout"),
    ):
        if metrics.get("split_sha256") != split.sha256:
            raise RuntimeError(f"{label} source uses a different frozen split")
    if model_metrics.get("holdout_used") is not False or rf_metrics.get("holdout_used") is not False:
        raise RuntimeError("Model probabilities were not screened with grouped OOF predictions")
    if blend_metrics.get("holdout_used") is not False or holdout_metrics.get("test_images_or_labels_used") is not False:
        raise RuntimeError("Candidate provenance reports holdout/test leakage")
    selected = blend_metrics["top_candidates"][0]
    if abs(float(selected["rf_weight"]) - RF_WEIGHT) > 1e-8 or abs(float(selected["threshold"]) - THRESHOLD) > 1e-8:
        raise RuntimeError("Configured test blend differs from OOF-selected settings")

    train_features_path = SELECTOR_RUN / "oof-component-features-targets.csv"
    test_features_path = TEST_FEATURE_RUN / "test-component-features.csv"
    train = pd.read_csv(train_features_path)
    test_features = pd.read_csv(test_features_path)
    if set(train.file_name.astype(str)) - set(split.training_names):
        raise RuntimeError("Selector training table crosses into the frozen holdout")
    test_names = sorted(path.name for path in dataset.test_image_dir.iterdir() if path.is_file())
    if len(test_names) != 180:
        raise RuntimeError(f"Expected 180 supplied H-alpha images, found {len(test_names)}")
    if set(test_features.file_name.astype(str)) - set(test_names):
        raise RuntimeError("Test features contain unknown observations")
    test_stems = {Path(name).stem for name in test_names}
    masks = load_prediction_rows(BASE_SUBMISSION, test_stems)
    if sum(len(masks[Path(name).stem]) for name in test_names) != len(test_features):
        raise RuntimeError("Test features do not align with the base submission components")
    test_feature_metrics = json.loads((TEST_FEATURE_RUN / "metrics.json").read_text(encoding="utf-8"))
    if test_feature_metrics.get("data_boundary", {}).get("test_labels_used") is not False:
        raise RuntimeError("Test feature source reports test-label use")
    if test_feature_metrics.get("input_sha256", {}).get("base_submission") != file_sha256(BASE_SUBMISSION):
        raise RuntimeError("Test features were built from a different base submission")

    logistic = _fit_model("logistic_c1.0", train, test_features)
    extra_trees = _fit_model("extra_trees", train, test_features)
    group_sizes = train.groupby("file_name").component_id.transform("count").to_numpy(float)
    sample_weight = 1.0 / np.maximum(group_sizes, 1.0)
    rf_model = RandomForestClassifier(
        n_estimators=400, max_depth=8, min_samples_leaf=10, max_features=0.8,
        class_weight=None, n_jobs=4, random_state=42,
    )
    rf_model.fit(
        train.loc[:, FEATURES].to_numpy(dtype=np.float32),
        train.target_keep.to_numpy(dtype=np.int8),
        sample_weight=sample_weight,
    )
    rf_probability = rf_model.predict_proba(test_features.loc[:, FEATURES].to_numpy(dtype=np.float32))[:, 1]
    core = (1.0 - CORE_EXTRA_TREES_WEIGHT) * logistic + CORE_EXTRA_TREES_WEIGHT * extra_trees
    blended = (1.0 - RF_WEIGHT) * core + RF_WEIGHT * rf_probability
    keyed = {
        (str(name), int(component_id)): float(probability)
        for name, component_id, probability in zip(
            test_features.file_name.astype(str), test_features.component_id.astype(int), blended
        )
    }
    predictions = predictions_from_masks(test_names, masks, keyed, THRESHOLD)

    output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = output_dir / "submission.csv"
    _write_predictions(submission_path, predictions)
    coverage = _check_submission(submission_path, data_root, test_names, predictions)
    test_features.to_csv(output_dir / "test-component-features.csv", index=False)
    metrics = {
        "status": "complete",
        "run_name": output_dir.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split_sha256": split.sha256,
        "oof_manifest_sha256": selector_metrics.get("oof_manifest_sha256"),
        "data_boundary": {
            "training_labels": "competition-provided annotations in the 565-observation grouped OOF training partition",
            "inference_inputs": "180 supplied competition H-alpha test images only",
            "test_labels_used": False,
            "test_images_used_for_training": False,
            "selector_fit_rows": "grouped OOF component features only",
        },
        "selection": {
            "core": "0.70 logistic C1 + 0.30 ExtraTrees cross-fitted probability",
            "random_forest": "400 trees, max_depth 8, min_samples_leaf 10, max_features 0.8",
            "rf_weight": RF_WEIGHT,
            "threshold": THRESHOLD,
            "oof_pq": float(selected["pq"]),
            "holdout_pq": float(holdout_metrics["candidate"]["global"]["pq"]),
            "holdout_paired_delta_ci_95": holdout_metrics["paired_bootstrap_vs_core_blend"]["paired_delta_candidate_minus_baseline"]["bootstrap_ci_95"],
            "predicted_instances": int(coverage["predicted_instances"]),
        },
        "coverage_validation": coverage,
        "submission_sha256": file_sha256(submission_path),
        "input_sha256": {
            "selector_oof_features": file_sha256(train_features_path),
            "test_component_features": file_sha256(test_features_path),
            "base_submission": file_sha256(BASE_SUBMISSION),
            "selector_metrics": file_sha256(SELECTOR_RUN / "metrics.json"),
            "rf_screen_metrics": file_sha256(RF_SCREEN / "metrics.json"),
            "rf_blend_screen_metrics": file_sha256(RF_BLEND_SCREEN / "metrics.json"),
            "rf_holdout_metrics": file_sha256(RF_HOLDOUT / "metrics.json"),
            "scorer": file_sha256(REPO_ROOT / "src/filament_baseline/scorer.py"),
        },
        "source_code_sha256": file_sha256(Path(__file__).resolve()),
        "source_repository": git_info(REPO_ROOT),
        "runtime": {
            "device_used": "cpu",
            "hardware": hardware_info(),
            "package_versions": package_versions(),
            "peak_gpu_memory_bytes": 0,
        },
        "runtime_seconds": time.perf_counter() - started,
        "artifacts": {"submission": submission_path.name, "test_component_features": "test-component-features.csv"},
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": metrics["status"], "coverage": coverage,
        "submission_sha256": metrics["submission_sha256"],
        "runtime_seconds": metrics["runtime_seconds"], "output_dir": str(output_dir),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
