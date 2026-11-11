# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Build a hash-locked batch from the next-best grouped-OOF RF selector settings.

The frozen holdout is deliberately not loaded or scored. The blend weights and
thresholds come only from the existing five-fold grouped-OOF sweep.
"""

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

from run_component_selector_oof import (  # noqa: E402
    FEATURES,
    load_prediction_rows,
    predictions_from_masks,
)
from screen_component_selector_models import _fit_model  # noqa: E402
from filament_baseline.data import load_dataset  # noqa: E402
from filament_baseline.runner import _write_predictions  # noqa: E402
from filament_baseline.runtime import hardware_info, package_versions  # noqa: E402
from filament_baseline.split import create_observation_split  # noqa: E402
from filament_baseline.submission import _check_submission  # noqa: E402


FROZEN_SPLIT_SHA256 = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"
SELECTOR_RUN = REPO_ROOT / "artifacts/model-experiments/component-selector-official-pq-20261109-rle-fixed"
MODEL_SCREEN = REPO_ROOT / "artifacts/model-experiments/component-selector-model-screen-20261109"
RF_SCREEN = REPO_ROOT / "artifacts/model-experiments/component-selector-random-forest-oof-20261109"
BLEND_SCREEN = REPO_ROOT / "artifacts/model-experiments/component-selector-rf-blend-oof-repro-20261109"
TEST_FEATURE_RUN = REPO_ROOT / "artifacts/submissions/component-selector-probability-blend-fullfit-test-20261109-rerun"
BASE_SUBMISSION = REPO_ROOT / "artifacts/submissions/final-yolo11s-fullfit-p25-20261108/submission.csv"
PREVIOUS_BATCH = REPO_ROOT / "artifacts/submissions/filament-20261110-component-selector-batch.json"
OUTPUT_DIR = REPO_ROOT / "artifacts/submissions/component-selector-rf-oof-batch-20261111"
MANIFEST_PATH = REPO_ROOT / "artifacts/submissions/filament-20261111-component-selector-rf-oof-batch.json"
CORE_EXTRA_TREES_WEIGHT = 0.30


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    import subprocess

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--blend-screen", type=Path, default=BLEND_SCREEN)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--candidate-count", type=int, default=5)
    parser.add_argument("--submission-day-utc", default="2026-11-11")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    blend_screen = args.blend_screen.resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = args.manifest.resolve()
    if not data_root.is_relative_to((REPO_ROOT / "data").resolve()):
        raise ValueError("Competition data must remain under data/")
    submissions_root = (REPO_ROOT / "artifacts/submissions").resolve()
    if not blend_screen.is_relative_to((REPO_ROOT / "artifacts/model-experiments").resolve()):
        raise ValueError("Blend sweep inputs must remain under artifacts/model-experiments/")
    if not output_dir.is_relative_to(submissions_root) or not manifest_path.is_relative_to(submissions_root):
        raise ValueError("Submission outputs and manifest must remain under artifacts/submissions/")
    if args.candidate_count < 1:
        raise ValueError("candidate-count must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    if manifest_path.exists():
        raise FileExistsError(manifest_path)

    started = time.perf_counter()
    dataset = load_dataset(data_root)
    split = create_observation_split(dataset, REPO_ROOT / "data/splits")
    if split.sha256 != FROZEN_SPLIT_SHA256:
        raise RuntimeError("Frozen grouped split digest changed")

    selector_metrics_path = SELECTOR_RUN / "metrics.json"
    model_metrics_path = MODEL_SCREEN / "metrics.json"
    rf_metrics_path = RF_SCREEN / "metrics.json"
    blend_metrics_path = blend_screen / "metrics.json"
    test_metrics_path = TEST_FEATURE_RUN / "metrics.json"
    selector_metrics = json.loads(selector_metrics_path.read_text(encoding="utf-8"))
    model_metrics = json.loads(model_metrics_path.read_text(encoding="utf-8"))
    rf_metrics = json.loads(rf_metrics_path.read_text(encoding="utf-8"))
    blend_metrics = json.loads(blend_metrics_path.read_text(encoding="utf-8"))
    test_metrics = json.loads(test_metrics_path.read_text(encoding="utf-8"))
    for label, metrics in (
        ("selector", selector_metrics),
        ("model screen", model_metrics),
        ("random-forest screen", rf_metrics),
        ("blend screen", blend_metrics),
        ("test features", test_metrics),
    ):
        if metrics.get("split_sha256") != split.sha256:
            raise RuntimeError(f"{label} source does not match the frozen split")
    if model_metrics.get("holdout_used") is not False or rf_metrics.get("holdout_used") is not False:
        raise RuntimeError("Selector probabilities were not screened on grouped OOF predictions")
    if blend_metrics.get("holdout_used") is not False:
        raise RuntimeError("RF blend settings were not selected using OOF-only evidence")
    if blend_metrics.get("test_images_or_labels_used") is not False:
        raise RuntimeError("RF blend sweep reports test image or label access")
    if test_metrics.get("data_boundary", {}).get("test_labels_used") is not False:
        raise RuntimeError("Test feature run reports test-label use")
    if test_metrics.get("data_boundary", {}).get("test_images_used_for_training") is not False:
        raise RuntimeError("Test feature run reports test images in training")

    history_paths = {PREVIOUS_BATCH}
    history_paths.update(
        path
        for path in submissions_root.glob("filament-*-component-selector-rf-oof-batch*.json")
        if path.resolve() != manifest_path
    )
    previous_hashes: set[str] = set()
    previous_settings: set[tuple[float, float]] = set()
    for history_path in sorted(history_paths):
        if not history_path.is_file():
            continue
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if history.get("competition") != "filament-segmentation-2026":
            continue
        if history.get("split_sha256") != split.sha256:
            raise RuntimeError(f"Submission history uses a different split: {history_path}")
        for row in history.get("candidates", []):
            if row.get("sha256"):
                previous_hashes.add(str(row["sha256"]))
            selection = row.get("selection")
            if isinstance(selection, dict) and {"rf_weight", "threshold"}.issubset(selection):
                previous_settings.add(
                    (round(float(selection["rf_weight"]), 8), round(float(selection["threshold"]), 8))
                )
            # The first component-selector batch predates explicit selection metadata.
            message = str(row.get("message", ""))
            if "70:30 selector probability blend;" in message:
                previous_settings.add((0.0, 0.45))
            if "selector + 30% random forest;" in message:
                previous_settings.add((0.30, 0.45))

    sweep_path = blend_screen / "blend-sweep.csv"
    if sweep_path.is_file():
        sweep = pd.read_csv(sweep_path)
        required_sweep_columns = {"rf_weight", "threshold", "pq"}
        if not required_sweep_columns.issubset(sweep.columns):
            raise RuntimeError(f"Blend sweep is missing columns: {sorted(required_sweep_columns - set(sweep.columns))}")
        candidates = sweep.to_dict(orient="records")
    else:
        candidates = blend_metrics["top_candidates"]
    candidates = sorted(
        candidates,
        key=lambda row: (-float(row["pq"]), float(row["rf_weight"]), float(row["threshold"])),
    )
    selected: list[dict[str, float]] = []
    selected_keys: set[tuple[float, float]] = set()
    for row in candidates:
        rf_weight = float(row["rf_weight"])
        threshold = float(row["threshold"])
        key = (round(rf_weight, 8), round(threshold, 8))
        if key in previous_settings or key in selected_keys:
            continue
        selected.append({"rf_weight": rf_weight, "threshold": threshold, "oof_pq": float(row["pq"])})
        selected_keys.add(key)
    if len(selected) < args.candidate_count:
        raise RuntimeError(f"OOF sweep provides only {len(selected)} unseen candidates")

    train_features_path = SELECTOR_RUN / "oof-component-features-targets.csv"
    test_features_path = TEST_FEATURE_RUN / "test-component-features.csv"
    train = pd.read_csv(train_features_path)
    test_features = pd.read_csv(test_features_path)
    if set(train.file_name.astype(str)) - set(split.training_names):
        raise RuntimeError("Selector training table crosses into the fixed holdout")
    if "oof_fold" not in train or set(train.oof_fold.astype(int)) != set(range(5)):
        raise RuntimeError("Selector training table is missing the five grouped OOF folds")
    if train.groupby("file_name").oof_fold.nunique().max() != 1:
        raise RuntimeError("Annotator views of an observation cross OOF folds")

    test_names = sorted(path.name for path in dataset.test_image_dir.iterdir() if path.is_file())
    if len(test_names) != 180:
        raise RuntimeError(f"Expected 180 supplied H-alpha test images, found {len(test_names)}")
    if set(test_features.file_name.astype(str)) - set(test_names):
        raise RuntimeError("Test features contain unknown observations")
    test_stems = {Path(name).stem for name in test_names}
    masks = load_prediction_rows(BASE_SUBMISSION, test_stems)
    if sum(len(masks[Path(name).stem]) for name in test_names) != len(test_features):
        raise RuntimeError("Test features do not align with the base submission components")
    if test_metrics.get("input_sha256", {}).get("base_submission") != sha256(BASE_SUBMISSION):
        raise RuntimeError("Test features were built from different base predictions")

    logistic = _fit_model("logistic_c1.0", train, test_features)
    extra_trees = _fit_model("extra_trees", train, test_features)
    group_sizes = train.groupby("file_name").component_id.transform("count").to_numpy(float)
    sample_weight = 1.0 / np.maximum(group_sizes, 1.0)
    forest = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=10,
        max_features=0.8,
        class_weight=None,
        n_jobs=4,
        random_state=42,
    )
    forest.fit(
        train.loc[:, FEATURES].to_numpy(dtype=np.float32),
        train.target_keep.to_numpy(dtype=np.int8),
        sample_weight=sample_weight,
    )
    rf_probability = forest.predict_proba(test_features.loc[:, FEATURES].to_numpy(dtype=np.float32))[:, 1]
    core = (1.0 - CORE_EXTRA_TREES_WEIGHT) * logistic + CORE_EXTRA_TREES_WEIGHT * extra_trees
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_candidates: list[dict[str, object]] = []
    used_hashes = set(previous_hashes)

    for setting in selected:
        candidate_index = len(manifest_candidates) + 1
        rf_weight = setting["rf_weight"]
        threshold = setting["threshold"]
        probabilities = (1.0 - rf_weight) * core + rf_weight * rf_probability
        keyed = {
            (str(name), int(component_id)): float(probability)
            for name, component_id, probability in zip(
                test_features.file_name.astype(str), test_features.component_id.astype(int), probabilities
            )
        }
        predictions = predictions_from_masks(test_names, masks, keyed, threshold)
        candidate_dir = output_dir / f"candidate-{candidate_index}"
        candidate_dir.mkdir(parents=True, exist_ok=False)
        submission_path = candidate_dir / "submission.csv"
        _write_predictions(submission_path, predictions)
        coverage = _check_submission(submission_path, data_root, test_names, predictions)
        submission_hash = sha256(submission_path)
        if submission_hash in used_hashes:
            submission_path.unlink()
            candidate_dir.rmdir()
            continue
        used_hashes.add(submission_hash)

        submission_day = datetime.fromisoformat(args.submission_day_utc).strftime("%b %d")
        message = (
            f"EquiNexus {submission_day} {candidate_index}/{args.candidate_count}; RF selector weight {rf_weight:.2f}; "
            f"threshold {threshold:.3f}; grouped OOF PQ {setting['oof_pq']:.6f}"
        )
        metrics = {
            "status": "complete",
            "run_name": candidate_dir.name,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "split_sha256": split.sha256,
            "oof_manifest_sha256": selector_metrics.get("oof_manifest_sha256"),
            "data_boundary": {
                "training_labels": "competition-provided annotations from the 565-observation grouped OOF training partition",
                "inference_inputs": "180 supplied competition H-alpha test images only",
                "test_labels_used": False,
                "test_images_used_for_training": False,
                "holdout_scored_for_this_candidate": False,
            },
            "selection": {
                "source": "top RF weight/threshold combinations from the grouped OOF blend sweep",
                "core_extra_trees_weight": CORE_EXTRA_TREES_WEIGHT,
                "rf_weight": rf_weight,
                "threshold": threshold,
                "oof_pq": setting["oof_pq"],
                "holdout_pq": None,
                "predicted_instances": int(coverage["predicted_instances"]),
            },
            "coverage_validation": coverage,
            "submission_sha256": submission_hash,
            "input_sha256": {
                "selector_oof_features": sha256(train_features_path),
                "test_component_features": sha256(test_features_path),
                "base_submission": sha256(BASE_SUBMISSION),
                "selector_metrics": sha256(selector_metrics_path),
                "model_screen_metrics": sha256(model_metrics_path),
                "rf_screen_metrics": sha256(rf_metrics_path),
                "blend_screen_metrics": sha256(blend_metrics_path),
                "blend_sweep": sha256(sweep_path) if sweep_path.is_file() else None,
                "test_feature_metrics": sha256(test_metrics_path),
                "scorer": sha256(REPO_ROOT / "src/filament_baseline/scorer.py"),
            },
            "source_code_sha256": sha256(Path(__file__).resolve()),
            "source_commit_at_generation": git_commit(),
            "runtime": {
                "device_used": "cpu",
                "hardware": hardware_info(),
                "package_versions": package_versions(),
                "peak_gpu_memory_bytes": 0,
            },
            "message": message,
            "submission": str(submission_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        }
        (candidate_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest_candidates.append(
            {
                "path": metrics["submission"],
                "sha256": submission_hash,
                "message": message,
                "oof_pq": setting["oof_pq"],
                "holdout_pq": None,
                "predicted_instances": int(coverage["predicted_instances"]),
                "test_images": int(coverage["test_images"]),
                "selection": {"rf_weight": rf_weight, "threshold": threshold},
            }
        )
        if len(manifest_candidates) == args.candidate_count:
            break

    if len(manifest_candidates) != args.candidate_count:
        raise RuntimeError(
            f"OOF sweep produced only {len(manifest_candidates)} unique submission hashes; "
            f"expected {args.candidate_count}"
        )

    manifest = {
        "batch": f"EquiNexus {datetime.fromisoformat(args.submission_day_utc).strftime('%b %d')} grouped-OOF RF selector variants",
        "candidate_count": len(manifest_candidates),
        "candidates": manifest_candidates,
        "competition": "filament-segmentation-2026",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": git_commit(),
        "split_sha256": split.sha256,
        "submission_day_utc": args.submission_day_utc,
        "blend_sweep": str(sweep_path.relative_to(REPO_ROOT)) if sweep_path.is_file() else None,
        "selection_policy": "Top unseen RF weight/threshold settings from the frozen five-fold OOF sweep; no holdout rescoring.",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "complete",
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "candidate_count": len(manifest_candidates),
        "split_sha256": split.sha256,
        "runtime_seconds": time.perf_counter() - started,
        "candidates": [
            {
                "message": c["message"],
                "oof_pq": c["oof_pq"],
                "path": c["path"],
                "sha256": c["sha256"],
            }
            for c in manifest_candidates
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
