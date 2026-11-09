# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Train and independently validate a scorer-aligned component keep/drop model.

Features use only the image, frozen base predictions, and probability maps.
Training targets measure each component's leave-one-out marginal PQ effect
under the organizer's all-qualifying-pairs scorer across every annotator view.
The selector is fit only on cross-fitted training-partition predictions; the
frozen 142-observation holdout is reserved for the final comparison.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from filament_baseline.data import load_dataset  # noqa: E402
from filament_baseline.predictor import PredictedInstance, encode_binary_mask  # noqa: E402
from filament_baseline.scorer import (  # noqa: E402
    annotation_to_rle,
    pairwise_iou,
    score_validation,
)
from filament_baseline.split import create_observation_split  # noqa: E402


FEATURES = (
    "log_area",
    "log_aspect",
    "bbox_fill",
    "eccentricity",
    "radius_mean",
    "radius_max",
    "intensity_mean",
    "intensity_std",
    "local_background_mean",
    "dark_contrast",
    "foreground_mean",
    "foreground_q10",
    "foreground_q50",
    "foreground_q90",
    "boundary_mean",
    "boundary_q50",
    "boundary_q90",
    "detector_score_max",
    "detector_iou_max",
    "detector_coverage_max",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_inside(path: Path, root: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"{label} must remain inside {root}: {resolved}")
    return resolved


def coco_rle(counts: str | bytes) -> dict[str, Any]:
    if isinstance(counts, str):
        counts = counts.encode("ascii")
    return {"size": [2048, 2048], "counts": counts}


def load_prediction_rows(path: Path, expected_stems: set[str]) -> dict[str, list[dict[str, Any]]]:
    predictions: dict[str, list[dict[str, Any]]] = {stem: [] for stem in expected_stems}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["filament_id", "segmentation_rle"]:
            raise ValueError(f"Unexpected prediction schema: {reader.fieldnames}")
        for row in reader:
            filament_id = str(row["filament_id"])
            stem = filament_id.rsplit("_", maxsplit=1)[0]
            if stem not in predictions:
                raise ValueError(f"Unexpected prediction observation: {stem}")
            predictions[stem].append(coco_rle(str(row["segmentation_rle"])))
    return predictions


def load_detector_rows(path: Path, expected_stems: set[str]) -> dict[str, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if set(payload) != expected_stems:
        raise ValueError(
            f"Detector coverage differs: expected {len(expected_stems)}, found {len(payload)}"
        )
    result = {}
    for stem, items in payload.items():
        rows = []
        for item in items:
            rows.append(
                {
                    "score": float(item["score"]),
                    "rle": coco_rle(item["counts"]),
                    "area": float(mask_utils.area(coco_rle(item["counts"]))),
                }
            )
        result[stem] = rows
    return result


def load_probability_maps(
    file_name: str,
    *,
    fold: int | None,
    first_root: Path,
    second_root: Path,
) -> tuple[np.ndarray, np.ndarray]:
    stem = Path(file_name).stem
    if fold is None:
        first_path = first_root / f"{stem}.npz"
        second_path = second_root / f"{stem}.npz"
    else:
        first_path = first_root / f"fold-{fold}" / "probabilities" / f"{stem}.npz"
        second_path = second_root / f"fold-{fold}" / "probabilities" / f"{stem}.npz"
    with np.load(first_path) as first, np.load(second_path) as second:
        foreground = (
            0.25 * first["foreground"].astype(np.float32)
            + 0.75 * second["foreground"].astype(np.float32)
        )
        boundary = (
            0.25 * first["boundary"].astype(np.float32)
            + 0.75 * second["boundary"].astype(np.float32)
        )
    if foreground.shape != (2048, 2048) or boundary.shape != (2048, 2048):
        raise ValueError(f"Unexpected probability map shape for {file_name}")
    return foreground, boundary


def extract_observation_features(
    dataset,
    file_name: str,
    predictions: list[dict[str, Any]],
    detector_rows: list[dict[str, Any]],
    foreground: np.ndarray,
    boundary: np.ndarray,
    image_dir: Path | None = None,
) -> list[dict[str, Any]]:
    image_path = (image_dir or dataset.train_image_dir) / file_name
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None or gray.shape != (2048, 2048):
        raise ValueError(f"Could not read expected 2048x2048 image: {image_path}")
    image_float = gray.astype(np.float32) / 255.0
    local_background = cv2.GaussianBlur(image_float, (31, 31), 0)
    detector_rles = [row["rle"] for row in detector_rows]
    detector_scores = np.asarray([row["score"] for row in detector_rows], dtype=np.float32)
    detector_areas = np.asarray([row["area"] for row in detector_rows], dtype=np.float32)
    output = []
    stem = Path(file_name).stem
    for component_id, rle in enumerate(predictions, start=1):
        mask = mask_utils.decode(rle).astype(np.uint8)
        ys, xs = np.nonzero(mask)
        if not len(xs):
            raise ValueError(f"Empty prediction component {stem}_{component_id}")
        area = int(len(xs))
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        width, height = max(1, x1 - x0), max(1, y1 - y0)
        cx, cy = float(xs.mean()), float(ys.mean())
        dx = xs.astype(np.float32) - cx
        dy = ys.astype(np.float32) - cy
        cxx = float(np.mean(dx * dx))
        cyy = float(np.mean(dy * dy))
        cxy = float(np.mean(dx * dy))
        trace = cxx + cyy
        discriminant = max(0.0, (cxx - cyy) ** 2 + 4.0 * cxy * cxy)
        major = max(1e-6, 0.5 * (trace + discriminant**0.5))
        minor = max(1e-6, 0.5 * (trace - discriminant**0.5))
        eccentricity = float(np.sqrt(max(0.0, 1.0 - minor / major)))
        radius = np.sqrt((xs.astype(np.float32) - 1023.5) ** 2 + (ys.astype(np.float32) - 1023.5) ** 2)
        inside_intensity = image_float[ys, xs]
        local_intensity = local_background[ys, xs]
        foreground_values = foreground[ys, xs]
        boundary_values = boundary[ys, xs]
        if detector_rles:
            ious = np.asarray(mask_utils.iou([rle], detector_rles, [0] * len(detector_rles)))[0]
            intersections = ious * (area + detector_areas) / np.maximum(1.0 + ious, 1e-6)
            coverage = intersections / max(area, 1)
            detector_index = int(np.argmax(ious))
            max_iou = float(ious[detector_index])
            max_coverage = float(np.max(coverage))
            max_score = float(np.max(detector_scores[coverage >= 0.05])) if np.any(coverage >= 0.05) else 0.0
        else:
            max_iou = max_coverage = max_score = 0.0
        row = {
            "file_name": file_name,
            "component_id": component_id,
            "log_area": float(np.log1p(area)),
            "log_aspect": float(np.log(max(width, height) / min(width, height))),
            "bbox_fill": float(area / (width * height)),
            "eccentricity": eccentricity,
            "radius_mean": float(radius.mean()),
            "radius_max": float(radius.max()),
            "intensity_mean": float(inside_intensity.mean()),
            "intensity_std": float(inside_intensity.std()),
            "local_background_mean": float(local_intensity.mean()),
            "dark_contrast": float(local_intensity.mean() - inside_intensity.mean()),
            "foreground_mean": float(foreground_values.mean()),
            "foreground_q10": float(np.quantile(foreground_values, 0.10)),
            "foreground_q50": float(np.quantile(foreground_values, 0.50)),
            "foreground_q90": float(np.quantile(foreground_values, 0.90)),
            "boundary_mean": float(boundary_values.mean()),
            "boundary_q50": float(np.quantile(boundary_values, 0.50)),
            "boundary_q90": float(np.quantile(boundary_values, 0.90)),
            "detector_score_max": max_score,
            "detector_iou_max": max_iou,
            "detector_coverage_max": max_coverage,
        }
        values = np.asarray([row[name] for name in FEATURES], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite feature for {stem}_{component_id}")
        output.append(row)
    return output


def target_marginal_pq(dataset, file_name: str, predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return exact per-component keep-vs-delete effects over this observation's views."""
    component_count = len(predictions)
    if component_count == 0:
        return []
    tp_sum_by_component = np.zeros(component_count, dtype=np.float64)
    tp_count_by_component = np.zeros(component_count, dtype=np.int64)
    fp_by_component = np.zeros(component_count, dtype=np.int64)
    fn_by_component = np.zeros(component_count, dtype=np.int64)
    total_tp_iou = 0.0
    total_tp = total_fp = total_fn = 0
    for view in dataset.views_by_observation[file_name]:
        annotations = dataset.annotations_by_image_id[view.image_id]
        gt_rles = [annotation_to_rle(item, view.height, view.width) for item in annotations]
        ious = pairwise_iou(gt_rles, predictions)
        hits = ious > 0.5
        row_degree = hits.sum(axis=1)
        col_degree = hits.sum(axis=0)
        total_tp_iou += float(ious[hits].sum())
        total_tp += int(hits.sum())
        total_fp += int(np.count_nonzero(col_degree == 0))
        total_fn += int(np.count_nonzero(row_degree == 0))
        tp_sum_by_component += np.where(hits, ious, 0.0).sum(axis=0)
        tp_count_by_component += hits.sum(axis=0)
        fp_by_component += (col_degree == 0).astype(np.int64)
        fn_by_component += ((row_degree == 1)[:, None] & hits).sum(axis=0)
    denominator = total_tp + 0.5 * total_fp + 0.5 * total_fn
    pq_keep = total_tp_iou / denominator if denominator > 0 else 0.0
    output = []
    for index in range(component_count):
        tp_after = total_tp - int(tp_count_by_component[index])
        fp_after = total_fp - int(fp_by_component[index])
        fn_after = total_fn + int(fn_by_component[index])
        iou_after = total_tp_iou - float(tp_sum_by_component[index])
        denominator = tp_after + 0.5 * fp_after + 0.5 * fn_after
        pq_after = iou_after / denominator if denominator > 0 else 0.0
        # A no-hit prediction can only reduce FP when removed.  The component
        # is useful when retaining it does not lower this observation's PQ.
        keep_delta = pq_keep - pq_after
        output.append(
            {
                "file_name": file_name,
                "component_id": index + 1,
                "keep_delta_pq": float(keep_delta),
                "target_keep": int(keep_delta >= 0.0),
            }
        )
    return output


def fit_predict(
    train_rows: pd.DataFrame,
    eval_rows: pd.DataFrame,
    *,
    c_value: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    x_train = train_rows.loc[:, FEATURES].to_numpy(dtype=np.float64)
    x_eval = eval_rows.loc[:, FEATURES].to_numpy(dtype=np.float64)
    y_train = train_rows.target_keep.to_numpy(dtype=np.int8)
    group_sizes = train_rows.groupby("file_name").component_id.transform("count").to_numpy(float)
    weights = 1.0 / np.maximum(group_sizes, 1.0)
    if len(np.unique(y_train)) < 2:
        probability = float(y_train[0]) if len(y_train) else 1.0
        return np.full(len(eval_rows), probability), {"constant_class": probability, "train_components": len(train_rows)}
    scaler = StandardScaler().fit(x_train, sample_weight=weights)
    if c_value <= 0:
        raise ValueError("c_value must be positive")
    model = LogisticRegression(C=c_value, solver="liblinear", max_iter=1000, random_state=42)
    model.fit(scaler.transform(x_train), y_train, sample_weight=weights)
    probabilities = model.predict_proba(scaler.transform(x_eval))[:, 1]
    metadata = {
        "family": "standardized_l2_logistic",
        "C": float(c_value),
        "train_components": int(len(train_rows)),
        "positive_fraction": float(y_train.mean()),
        "feature_names": list(FEATURES),
        "feature_mean": [float(v) for v in scaler.mean_],
        "feature_scale": [float(v) for v in scaler.scale_],
        "coefficients": [float(v) for v in model.coef_[0]],
        "intercept": float(model.intercept_[0]),
    }
    return probabilities.astype(float), metadata


def predictions_from_masks(
    names: list[str],
    masks_by_name: dict[str, list[dict[str, Any]]],
    probabilities_by_key: dict[tuple[str, int], float] | None = None,
    threshold: float | None = None,
) -> dict[str, list[PredictedInstance]]:
    result: dict[str, list[PredictedInstance]] = {}
    for file_name in names:
        stem = Path(file_name).stem
        rendered = []
        for index, rle in enumerate(masks_by_name[Path(file_name).stem], start=1):
            if probabilities_by_key is not None and threshold is not None:
                if probabilities_by_key[(file_name, index)] < threshold:
                    continue
            rendered.append(
                PredictedInstance(
                    filament_id=f"{stem}_{len(rendered) + 1}",
                    image_stem=stem,
                    area=int(mask_utils.area(rle)),
                    rle=rle,
                )
            )
        result[file_name] = rendered
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--run-name", default="component-selector-official-pq-20261109-rle-fixed")
    parser.add_argument("--oof-predictions", type=Path, default=REPO_ROOT / "artifacts/model-experiments/oof-fusion-weight25-th035-a200/best-oof-predictions.csv")
    parser.add_argument("--holdout-predictions", type=Path, default=REPO_ROOT / "artifacts/model-experiments/yolo11s-25perann-75consensus-foldmeans-valid-holdout-th035-a200/predictions.csv")
    parser.add_argument("--oof-detector-predictions", type=Path, default=REPO_ROOT / "artifacts/detector-oof/yolo11s-seg-1280-40ep-coco-oof-tta8/dedup-iou-030/oof_detections.json")
    parser.add_argument("--holdout-detector-predictions", type=Path, default=REPO_ROOT / "artifacts/detector-experiments/yolo11s-seg-1280-oof-holdout-ensemble-tta8/dedup-iou-030/oof_detections.json")
    parser.add_argument("--oof-perann-root", type=Path, default=REPO_ROOT / "artifacts/oof-experiments/oof-perann1024-10k-tta8")
    parser.add_argument("--oof-consensus-root", type=Path, default=REPO_ROOT / "artifacts/kaggle/oof-consensus-download/EquiNexus/artifacts/oof-experiments/oof-consensus512-5k-tta4")
    parser.add_argument("--holdout-perann-root", type=Path, default=REPO_ROOT / "artifacts/kaggle/fold-ensemble/remote-v3-output/EquiNexus/oof-fivefold-mean-v1/probabilities/holdout")
    parser.add_argument("--holdout-consensus-root", type=Path, default=REPO_ROOT / "artifacts/model-experiments/oof-consensus512-foldmean-valid-holdout/probabilities")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    data_root = validate_inside(args.data_root, repo_root / "data", "data root")
    output_root = validate_inside(repo_root / "artifacts" / "model-experiments" / args.run_name, repo_root / "artifacts", "output")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(output_root)
    path_args = {
        "oof_predictions": args.oof_predictions,
        "holdout_predictions": args.holdout_predictions,
        "oof_detector_predictions": args.oof_detector_predictions,
        "holdout_detector_predictions": args.holdout_detector_predictions,
        "oof_perann_root": args.oof_perann_root,
        "oof_consensus_root": args.oof_consensus_root,
        "holdout_perann_root": args.holdout_perann_root,
        "holdout_consensus_root": args.holdout_consensus_root,
    }
    resolved = {
        key: validate_inside(value, repo_root / "artifacts", key)
        for key, value in path_args.items()
    }

    config_path = repo_root / "configs/validation-baseline.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset = load_dataset(data_root)
    split = create_observation_split(
        dataset,
        repo_root / "data/splits",
        seed=int(config["seed"]),
        validation_fraction=float(config["validation_fraction"]),
    )
    if split.sha256 != config["expected_split_sha256"]:
        raise RuntimeError("Frozen validation split digest changed")
    manifest_path = repo_root / "data/splits/observation_oof_seed42_5fold.csv"
    with manifest_path.open("r", encoding="utf-8", newline="") as stream:
        manifest = list(csv.DictReader(stream))
    fold_by_name = {row["file_name"]: int(row["oof_fold"]) for row in manifest}
    oof_names = sorted(split.training_names)
    holdout_names = sorted(split.validation_names)
    if len(fold_by_name) != len(oof_names) or set(fold_by_name) != set(oof_names):
        raise RuntimeError("Five-fold OOF manifest does not cover the frozen training partition")

    oof_stems = {Path(name).stem for name in oof_names}
    holdout_stems = {Path(name).stem for name in holdout_names}
    oof_masks = load_prediction_rows(resolved["oof_predictions"], oof_stems)
    holdout_masks = load_prediction_rows(resolved["holdout_predictions"], holdout_stems)
    oof_detectors = load_detector_rows(resolved["oof_detector_predictions"], oof_stems)
    holdout_detectors = load_detector_rows(resolved["holdout_detector_predictions"], holdout_stems)
    if set(holdout_detectors) != holdout_stems:
        raise RuntimeError("Holdout detector prediction coverage mismatch")

    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    feature_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    for index, file_name in enumerate(oof_names, start=1):
        foreground, boundary = load_probability_maps(
            file_name,
            fold=fold_by_name[file_name],
            first_root=resolved["oof_perann_root"],
            second_root=resolved["oof_consensus_root"],
        )
        feature_rows.extend(
            extract_observation_features(
                dataset,
                file_name,
                oof_masks[Path(file_name).stem],
                oof_detectors[Path(file_name).stem],
                foreground,
                boundary,
            )
        )
        target_rows.extend(target_marginal_pq(dataset, file_name, oof_masks[Path(file_name).stem]))
        if index % 50 == 0 or index == len(oof_names):
            print(f"Built OOF component features/targets {index}/{len(oof_names)}", flush=True)
    oof_features = pd.DataFrame(feature_rows)
    oof_targets = pd.DataFrame(target_rows)
    oof = oof_features.merge(oof_targets, on=["file_name", "component_id"], validate="one_to_one")
    if len(oof) != len(oof_features) or list(oof.loc[:, FEATURES].columns) != list(FEATURES):
        raise RuntimeError("OOF component feature/target coverage mismatch")
    oof["oof_fold"] = oof.file_name.map(fold_by_name).astype(int)
    oof.to_csv(output_root / "oof-component-features-targets.csv", index=False)

    holdout_features_list: list[dict[str, Any]] = []
    for index, file_name in enumerate(holdout_names, start=1):
        foreground, boundary = load_probability_maps(
            file_name,
            fold=None,
            first_root=resolved["holdout_perann_root"],
            second_root=resolved["holdout_consensus_root"],
        )
        holdout_features_list.extend(
            extract_observation_features(
                dataset,
                file_name,
                holdout_masks[Path(file_name).stem],
                holdout_detectors[Path(file_name).stem],
                foreground,
                boundary,
            )
        )
        if index % 25 == 0 or index == len(holdout_names):
            print(f"Built holdout component features {index}/{len(holdout_names)}", flush=True)
    holdout_features = pd.DataFrame(holdout_features_list)
    holdout_features.to_csv(output_root / "holdout-component-features.csv", index=False)

    # Generate selector probabilities only from folds other than each row's
    # source prediction fold. Threshold selection therefore sees no in-fold
    # classifier scores, and every base prediction is itself grouped OOF.
    oof_probability = np.full(len(oof), np.nan, dtype=np.float64)
    model_records = []
    for fold in range(5):
        train = oof[oof.oof_fold != fold]
        evaluate = oof[oof.oof_fold == fold]
        probabilities, metadata = fit_predict(train, evaluate)
        oof_probability[oof.oof_fold.to_numpy() == fold] = probabilities
        model_records.append({"heldout_fold": fold, **metadata})
        print(f"Cross-fit selector fold {fold}: train={len(train)} evaluate={len(evaluate)}", flush=True)
    if not np.isfinite(oof_probability).all():
        raise RuntimeError("Missing cross-fitted selector probabilities")
    oof["selector_probability"] = oof_probability

    from filament_baseline.runner import _write_observation_scores, _write_predictions  # noqa: PLC0415

    threshold_rows = []
    thresholds = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80)
    baseline_oof_predictions = predictions_from_masks(oof_names, oof_masks)
    baseline_oof_metrics, baseline_oof_rows = score_validation(dataset, baseline_oof_predictions, oof_names)
    for threshold in thresholds:
        probabilities_by_key = {
            (str(row.file_name), int(row.component_id)): float(row.selector_probability)
            for row in oof.itertuples()
        }
        predictions = predictions_from_masks(oof_names, oof_masks, probabilities_by_key, threshold)
        metrics, _ = score_validation(dataset, predictions, oof_names)
        threshold_rows.append({"threshold": threshold, "metrics": metrics})
        print(f"OOF selector threshold={threshold:.2f} PQ={metrics['global']['pq']:.6f}", flush=True)
    best_row = max(threshold_rows, key=lambda row: row["metrics"]["global"]["pq"])
    selected_threshold = float(best_row["threshold"])

    # Fit once on all grouped OOF component rows, then evaluate exactly once on
    # the untouched 142-observation holdout.
    holdout_probability, final_model = fit_predict(oof, holdout_features)
    holdout_features["selector_probability"] = holdout_probability
    holdout_probabilities_by_key = {
        (str(row.file_name), int(row.component_id)): float(row.selector_probability)
        for row in holdout_features.itertuples()
    }
    baseline_holdout_predictions = predictions_from_masks(holdout_names, holdout_masks)
    baseline_holdout_metrics, _ = score_validation(dataset, baseline_holdout_predictions, holdout_names)
    selected_holdout_predictions = predictions_from_masks(
        holdout_names, holdout_masks, holdout_probabilities_by_key, selected_threshold
    )
    selected_holdout_metrics, holdout_rows = score_validation(dataset, selected_holdout_predictions, holdout_names)

    _write_predictions(output_root / "selector-holdout-predictions.csv", selected_holdout_predictions)
    _write_observation_scores(
        output_root / "selector-holdout-per-observation.csv", holdout_rows, selected_holdout_predictions
    )
    _write_predictions(output_root / "baseline-holdout-predictions.csv", baseline_holdout_predictions)
    _write_observation_scores(
        output_root / "baseline-holdout-per-observation.csv", _, baseline_holdout_predictions
    )

    # Save the cross-fitted OOF prediction table at the selected threshold for
    # later official-scorer and paired-bootstrap comparisons.
    selected_oof_probabilities = {
        (str(row.file_name), int(row.component_id)): float(row.selector_probability)
        for row in oof.itertuples()
    }
    selected_oof_predictions = predictions_from_masks(oof_names, oof_masks, selected_oof_probabilities, selected_threshold)
    selected_oof_metrics, selected_oof_rows = score_validation(dataset, selected_oof_predictions, oof_names)
    _write_predictions(output_root / "selector-oof-predictions.csv", selected_oof_predictions)
    _write_observation_scores(output_root / "selector-oof-per-observation.csv", selected_oof_rows, selected_oof_predictions)

    provenance_paths = {
        name: {"path": str(path.relative_to(repo_root)), "sha256": sha256(path)}
        for name, path in resolved.items()
        if path.is_file()
    }
    result = {
        "status": "complete",
        "run_name": args.run_name,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split_sha256": split.sha256,
        "oof_manifest_sha256": sha256(manifest_path),
        "scope": {
            "oof_observations": len(oof_names),
            "holdout_observations": len(holdout_names),
            "oof_component_count": len(oof),
            "holdout_component_count": len(holdout_features),
            "training_labels": "competition-provided training annotations only",
            "external_ground_truth": False,
            "test_images_used": False,
            "test_labels_used": False,
        },
        "target": "component leave-one-out marginal PQ on all annotator views using strict IoU > 0.5 all-pairs scoring",
        "feature_names": list(FEATURES),
        "fit": {
            "crossfit_models": model_records,
            "final_model": final_model,
            "selected_threshold": selected_threshold,
            "thresholds": threshold_rows,
            "oof_baseline": baseline_oof_metrics,
            "oof_selected": selected_oof_metrics,
            "holdout_baseline": baseline_holdout_metrics,
            "holdout_selected": selected_holdout_metrics,
        },
        "sources": provenance_paths,
        "runtime_seconds": time.perf_counter() - started,
        "artifacts": {
            "oof_features_targets": "oof-component-features-targets.csv",
            "holdout_features": "holdout-component-features.csv",
            "selector_oof_predictions": "selector-oof-predictions.csv",
            "selector_oof_per_observation": "selector-oof-per-observation.csv",
            "selector_holdout_predictions": "selector-holdout-predictions.csv",
            "selector_holdout_per_observation": "selector-holdout-per-observation.csv",
            "baseline_holdout_predictions": "baseline-holdout-predictions.csv",
            "baseline_holdout_per_observation": "baseline-holdout-per-observation.csv",
        },
    }
    (output_root / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "selected_threshold": selected_threshold,
        "oof_baseline_pq": baseline_oof_metrics["global"]["pq"],
        "oof_selected_pq": selected_oof_metrics["global"]["pq"],
        "holdout_baseline_pq": baseline_holdout_metrics["global"]["pq"],
        "holdout_selected_pq": selected_holdout_metrics["global"]["pq"],
        "output": str(output_root),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
