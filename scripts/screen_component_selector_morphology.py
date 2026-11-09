# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Screen skeleton- and contrast-aware component features with grouped OOF."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import sklearn
from pycocotools import mask as mask_utils
from skimage.morphology import skeletonize
from sklearn.ensemble import RandomForestClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_component_selector_oof import (  # noqa: E402
    FEATURES,
    load_prediction_rows,
    predictions_from_masks,
)
from screen_component_selector_random_forest import (  # noqa: E402
    DEFAULT_BASE_PREDICTIONS,
    DEFAULT_SELECTOR_RUN,
    MODEL_CONFIGS,
)
from filament_baseline.data import load_dataset  # noqa: E402
from filament_baseline.scorer import score_validation  # noqa: E402
from filament_baseline.split import create_observation_split  # noqa: E402


FROZEN_SPLIT_SHA256 = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"
MODEL_SCREEN = REPO_ROOT / "artifacts/model-experiments/component-selector-model-screen-20261109"
RF_SCREEN = REPO_ROOT / "artifacts/model-experiments/component-selector-random-forest-oof-20261109"
OUTPUT_DIR = REPO_ROOT / "artifacts/model-experiments/component-selector-morphology-20261109"
NEW_FEATURES = (
    "log_skeleton_length",
    "skeleton_density",
    "log_mean_width",
    "endpoint_fraction",
    "branch_fraction",
    "perimeter_area_ratio",
    "inside_intensity_q10",
    "inside_intensity_q90",
    "near_ring_contrast",
    "far_ring_contrast",
)
THRESHOLDS = (0.35, 0.375, 0.40, 0.425, 0.45, 0.475, 0.50)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def morphology_features(gray: np.ndarray, rle: dict[str, Any]) -> dict[str, float]:
    mask = mask_utils.decode(rle).astype(np.uint8)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("Empty component in OOF predictions")
    area = len(xs)
    y0, y1 = max(0, int(ys.min()) - 16), min(mask.shape[0], int(ys.max()) + 17)
    x0, x1 = max(0, int(xs.min()) - 16), min(mask.shape[1], int(xs.max()) + 17)
    local_mask = mask[y0:y1, x0:x1]
    local_gray = gray[y0:y1, x0:x1].astype(np.float32) / 255.0

    skeleton = skeletonize(local_mask != 0)
    skeleton_length = int(skeleton.sum())
    if skeleton_length == 0:
        skeleton_length = 1
    neighbor_count = cv2.filter2D(
        skeleton.astype(np.uint8), cv2.CV_16S, np.ones((3, 3), dtype=np.uint8)
    ) - skeleton.astype(np.int16)
    endpoints = int(np.count_nonzero(skeleton & (neighbor_count == 1)))
    branches = int(np.count_nonzero(skeleton & (neighbor_count >= 3)))

    contours, _ = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
    inside = local_gray[local_mask != 0]
    near_dilated = cv2.dilate(
        local_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    )
    far_dilated = cv2.dilate(
        local_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    )
    near_ring = (near_dilated != 0) & (local_mask == 0)
    far_ring = (far_dilated != 0) & (near_dilated == 0)
    near_values = local_gray[near_ring]
    far_values = local_gray[far_ring]
    inside_mean = float(inside.mean())

    return {
        "log_skeleton_length": float(np.log1p(skeleton_length)),
        "skeleton_density": float(skeleton_length / area),
        "log_mean_width": float(np.log1p(area / skeleton_length)),
        "endpoint_fraction": float(endpoints / skeleton_length),
        "branch_fraction": float(branches / skeleton_length),
        "perimeter_area_ratio": float(perimeter / area),
        "inside_intensity_q10": float(np.quantile(inside, 0.10)),
        "inside_intensity_q90": float(np.quantile(inside, 0.90)),
        "near_ring_contrast": float(near_values.mean() - inside_mean) if len(near_values) else 0.0,
        "far_ring_contrast": float(far_values.mean() - inside_mean) if len(far_values) else 0.0,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--selector-run", type=Path, default=DEFAULT_SELECTOR_RUN)
    parser.add_argument("--base-predictions", type=Path, default=DEFAULT_BASE_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    selector_run = args.selector_run.resolve()
    base_path = args.base_predictions.resolve()
    output_dir = args.output_dir.resolve()
    for path, label, root in (
        (data_root, "dataset", (REPO_ROOT / "data").resolve()),
        (selector_run, "selector run", (REPO_ROOT / "artifacts").resolve()),
        (base_path, "base predictions", (REPO_ROOT / "artifacts").resolve()),
        (output_dir, "output", (REPO_ROOT / "artifacts/model-experiments").resolve()),
    ):
        if not path.is_relative_to(root):
            raise ValueError(f"{label} must remain inside {root}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)

    started = time.perf_counter()
    dataset = load_dataset(data_root)
    split = create_observation_split(dataset, REPO_ROOT / "data/splits")
    if split.sha256 != FROZEN_SPLIT_SHA256:
        raise RuntimeError(f"Frozen split changed: {split.sha256}")
    selector_metrics_path = selector_run / "metrics.json"
    selector_metrics = json.loads(selector_metrics_path.read_text(encoding="utf-8"))
    if selector_metrics.get("split_sha256") != split.sha256:
        raise RuntimeError("Selector feature source uses another split")
    core_metrics = json.loads((MODEL_SCREEN / "metrics.json").read_text(encoding="utf-8"))
    rf_metrics = json.loads((RF_SCREEN / "metrics.json").read_text(encoding="utf-8"))
    for metrics, label in ((core_metrics, "core selector"), (rf_metrics, "random forest")):
        if metrics.get("split_sha256") != split.sha256 or metrics.get("holdout_used") is not False:
            raise RuntimeError(f"{label} probabilities are not frozen grouped OOF predictions")
        if metrics.get("test_images_or_labels_used") is not False:
            raise RuntimeError(f"{label} metrics report test-image or test-label use")

    features_path = selector_run / "oof-component-features-targets.csv"
    features = pd.read_csv(features_path)
    training_names = set(split.training_names)
    if set(features.file_name.astype(str)) - training_names:
        raise RuntimeError("OOF rows include observations outside the grouped training partition")
    folds = features.oof_fold.to_numpy(dtype=int)
    if set(np.unique(folds)) != set(range(5)):
        raise RuntimeError("Expected all five frozen grouped folds")
    masks = load_prediction_rows(base_path, {Path(name).stem for name in training_names})

    features["file_name"] = features.file_name.astype(str)
    features["component_id"] = features.component_id.astype(int)
    added: list[dict[str, Any]] = []
    for file_name, group in features.groupby("file_name", sort=True):
        stem = Path(file_name).stem
        image = cv2.imread(str(dataset.train_image_dir / file_name), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (2048, 2048):
            raise ValueError(f"Invalid training image: {file_name}")
        components = masks[stem]
        for row in group.itertuples(index=False):
            component_id = int(row.component_id)
            if component_id < 1 or component_id > len(components):
                raise ValueError(f"Missing OOF component {stem}_{component_id}")
            values = morphology_features(image, components[component_id - 1])
            values.update({"file_name": file_name, "component_id": component_id})
            added.append(values)
        print(f"features: {stem} components={len(group)}", flush=True)

    added_frame = pd.DataFrame(added)
    enriched = features.merge(added_frame, on=["file_name", "component_id"], validate="one_to_one")
    if len(enriched) != len(features) or not np.isfinite(enriched[list(NEW_FEATURES)].to_numpy()).all():
        raise RuntimeError("Morphology feature alignment or finite-value check failed")

    output_dir.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_dir / "oof-enriched-features-targets.csv", index=False)

    core_prob_path = MODEL_SCREEN / "oof-model-probabilities.csv"
    rf_prob_path = RF_SCREEN / "oof-model-probabilities.csv"
    core_prob = pd.read_csv(core_prob_path)
    rf_prob = pd.read_csv(rf_prob_path)
    keys = ["file_name", "component_id"]
    expected = pd.MultiIndex.from_frame(enriched[keys])
    expected_keys = set(zip(enriched.file_name.astype(str), enriched.component_id.astype(int), strict=True))
    for table, label in ((core_prob, "core"), (rf_prob, "random forest")):
        got_keys = set(zip(table.file_name.astype(str), table.component_id.astype(int), strict=True))
        if got_keys != expected_keys:
            raise RuntimeError(f"{label} probability rows do not match morphology feature rows")
    core_prob = core_prob.set_index(keys).loc[expected]
    rf_prob = rf_prob.set_index(keys).loc[expected]
    core = 0.70 * core_prob["logistic_c1.0"].to_numpy() + 0.30 * core_prob["extra_trees"].to_numpy()
    old_rf = rf_prob["rf_depth8_leaf10"].to_numpy()

    names = sorted(training_names)
    prediction_masks = load_prediction_rows(base_path, {Path(name).stem for name in names})
    probabilities = enriched[keys].copy()
    scores = np.full(len(enriched), np.nan, dtype=np.float64)
    model_name = "rf_depth8_leaf10"
    model_config = dict(MODEL_CONFIGS[model_name])
    model_config.update({"n_jobs": 4, "random_state": 42})
    group_sizes = enriched.groupby("file_name").component_id.transform("count").to_numpy(float)
    sample_weights = 1.0 / np.maximum(group_sizes, 1.0)
    y = enriched.target_keep.to_numpy(dtype=np.int8)
    x = enriched.loc[:, [*FEATURES, *NEW_FEATURES]].to_numpy(dtype=np.float32)
    for fold in range(5):
        held = folds == fold
        model = RandomForestClassifier(**model_config)
        model.fit(x[~held], y[~held], sample_weight=sample_weights[~held])
        scores[held] = model.predict_proba(x[held])[:, 1]
        print(f"model: fold={fold} train={int((~held).sum())} held={int(held.sum())}", flush=True)
    if not np.isfinite(scores).all():
        raise RuntimeError("Enriched OOF predictions are incomplete")
    probabilities["morphology_rf"] = scores
    probabilities.to_csv(output_dir / "oof-model-probabilities.csv", index=False)

    # Candidate selection is restricted to cross-fitted training predictions.
    candidates: list[dict[str, Any]] = []
    for label, raw in (
        ("core", core),
        ("core_plus_original_rf", 0.70 * core + 0.30 * old_rf),
        ("core_plus_morphology_rf_30", 0.70 * core + 0.30 * scores),
        ("core_plus_morphology_rf_50", 0.50 * core + 0.50 * scores),
        ("rf_pair_plus_morphology_rf_50", 0.35 * core + 0.15 * old_rf + 0.50 * scores),
    ):
        keyed = {
            (str(file_name), int(component_id)): float(probability)
            for file_name, component_id, probability in zip(
                enriched.file_name, enriched.component_id, raw, strict=True
            )
        }
        for threshold in THRESHOLDS:
            predictions = predictions_from_masks(names, prediction_masks, keyed, threshold)
            metrics, _ = score_validation(dataset, predictions, names)
            candidates.append({
                "candidate": label,
                "threshold": threshold,
                "pq": float(metrics["global"]["pq"]),
                "tp": int(metrics["global"]["tp"]),
                "fp": int(metrics["global"]["fp"]),
                "fn": int(metrics["global"]["fn"]),
            })
    candidates.sort(key=lambda row: row["pq"], reverse=True)
    pd.DataFrame(candidates).to_csv(output_dir / "candidate-sweep.csv", index=False)
    report = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split_sha256": split.sha256,
        "oof_manifest_sha256": selector_metrics["oof_manifest_sha256"],
        "training_observations": len(training_names),
        "component_rows": len(enriched),
        "holdout_used": False,
        "test_images_or_labels_used": False,
        "training_labels": "competition-provided annotations for grouped OOF observations only",
        "features": list(NEW_FEATURES),
        "model": model_config,
        "baseline_rf_blend_oof_pq": 0.4247906680196568,
        "top_candidates": candidates[:15],
        "input_sha256": {
            "source_features": sha256(features_path),
            "base_oof_predictions": sha256(base_path),
            "core_oof_probabilities": sha256(core_prob_path),
            "random_forest_oof_probabilities": sha256(rf_prob_path),
        },
        "versions": {"opencv": cv2.__version__, "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "python": platform.python_version(),
        "runtime_seconds": time.perf_counter() - started,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"top_candidates": candidates[:10], "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
