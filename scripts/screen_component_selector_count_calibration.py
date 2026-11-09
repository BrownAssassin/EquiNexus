# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Screen observation-level prediction-count calibration on grouped OOF data."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
SELECTOR_RUN = REPO_ROOT / "artifacts/model-experiments/component-selector-official-pq-20261109-rle-fixed"
MODEL_SCREEN = REPO_ROOT / "artifacts/model-experiments/component-selector-model-screen-20261109"
RF_SCREEN = REPO_ROOT / "artifacts/model-experiments/component-selector-random-forest-oof-20261109"
BASE_PREDICTIONS = REPO_ROOT / "artifacts/model-experiments/oof-fusion-weight25-th035-a200/best-oof-predictions.csv"
OOF_MANIFEST = REPO_ROOT / "data/splits/observation_oof_seed42_5fold.csv"
OUTPUT_DIR = REPO_ROOT / "artifacts/model-experiments/component-selector-count-calibration-20261109"
BLENDS = {"core": (1.0, 0.0), "core_rf30": (0.70, 0.30)}
COUNT_SCALES = tuple(float(value) for value in np.arange(0.60, 1.401, 0.10))
REGRESSORS = ("ridge_a1", "ridge_a10", "rf_depth3_leaf12", "hist_leaf5")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_count_model(name: str, train: pd.DataFrame, evaluate: pd.DataFrame) -> np.ndarray:
    numeric = [column for column in train.columns if column.startswith("x_")]
    categories = ["site"]
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)
    encoded_train = encoder.fit_transform(train[categories])
    encoded_eval = encoder.transform(evaluate[categories])
    x_train = train[numeric].to_numpy(dtype=np.float64)
    x_eval = evaluate[numeric].to_numpy(dtype=np.float64)
    y_train = train.expected_gt_count.to_numpy(dtype=np.float64)

    if name.startswith("ridge_"):
        alpha = float(name.removeprefix("ridge_a"))
        scaler = StandardScaler().fit(x_train)
        x_train = np.column_stack((scaler.transform(x_train), encoded_train))
        x_eval = np.column_stack((scaler.transform(x_eval), encoded_eval))
        model = Ridge(alpha=alpha)
    elif name == "rf_depth3_leaf12":
        x_train = np.column_stack((x_train, encoded_train))
        x_eval = np.column_stack((x_eval, encoded_eval))
        model = RandomForestRegressor(
            n_estimators=300, max_depth=3, min_samples_leaf=12,
            max_features=0.8, n_jobs=4, random_state=42,
        )
    elif name == "hist_leaf5":
        x_train = np.column_stack((x_train, encoded_train))
        x_eval = np.column_stack((x_eval, encoded_eval))
        model = HistGradientBoostingRegressor(
            learning_rate=0.04, max_iter=100, max_leaf_nodes=5,
            min_samples_leaf=12, l2_regularization=3.0,
            early_stopping=False, random_state=42,
        )
    else:
        raise ValueError(f"Unknown count model: {name}")
    model.fit(x_train, y_train)
    return np.maximum(model.predict(x_eval), 0.0)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--selector-run", type=Path, default=SELECTOR_RUN)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    selector_run = args.selector_run.resolve()
    output_dir = args.output_dir.resolve()
    for path, label, root in (
        (data_root, "dataset", (REPO_ROOT / "data").resolve()),
        (selector_run, "selector features", (REPO_ROOT / "artifacts").resolve()),
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
        raise RuntimeError(f"Frozen holdout split changed: {split.sha256}")
    selector_metrics_path = selector_run / "metrics.json"
    selector_metrics = json.loads(selector_metrics_path.read_text(encoding="utf-8"))
    if selector_metrics.get("split_sha256") != split.sha256:
        raise RuntimeError("Selector feature file uses another holdout split")
    core_metrics = json.loads((MODEL_SCREEN / "metrics.json").read_text(encoding="utf-8"))
    rf_metrics = json.loads((RF_SCREEN / "metrics.json").read_text(encoding="utf-8"))
    for metrics, label in ((core_metrics, "core selector"), (rf_metrics, "random forest")):
        if metrics.get("split_sha256") != split.sha256 or metrics.get("holdout_used") is not False:
            raise RuntimeError(f"{label} scores are not holdout-free grouped OOF predictions")
        if metrics.get("test_images_or_labels_used") is not False:
            raise RuntimeError(f"{label} run reports test-image or test-label use")

    training_names = sorted(split.training_names)
    features_path = selector_run / "oof-component-features-targets.csv"
    features = pd.read_csv(features_path)
    keys = ["file_name", "component_id"]
    core_prob_path = MODEL_SCREEN / "oof-model-probabilities.csv"
    rf_prob_path = RF_SCREEN / "oof-model-probabilities.csv"
    core_prob = pd.read_csv(core_prob_path)
    rf_prob = pd.read_csv(rf_prob_path)
    expected_keys = set(zip(features.file_name.astype(str), features.component_id.astype(int), strict=True))
    for table, label in ((core_prob, "core"), (rf_prob, "forest")):
        got_keys = set(zip(table.file_name.astype(str), table.component_id.astype(int), strict=True))
        if got_keys != expected_keys:
            raise RuntimeError(f"{label} probabilities do not align with OOF feature rows")

    core_lookup = core_prob.set_index(keys)
    rf_lookup = rf_prob.set_index(keys)
    core_scores = 0.70 * core_lookup.loc[
        pd.MultiIndex.from_frame(features[keys]), "logistic_c1.0"
    ].to_numpy(dtype=np.float64) + 0.30 * core_lookup.loc[
        pd.MultiIndex.from_frame(features[keys]), "extra_trees"
    ].to_numpy(dtype=np.float64)
    forest_scores = rf_lookup.loc[
        pd.MultiIndex.from_frame(features[keys]), "rf_depth8_leaf10"
    ].to_numpy(dtype=np.float64)
    core_scores_by_key = {
        (str(name), int(component)): float(value)
        for name, component, value in zip(features.file_name, features.component_id, core_scores, strict=True)
    }
    forest_scores_by_key = {
        (str(name), int(component)): float(value)
        for name, component, value in zip(features.file_name, features.component_id, forest_scores, strict=True)
    }

    oof_manifest = pd.read_csv(OOF_MANIFEST)
    if set(oof_manifest.file_name.astype(str)) != set(training_names):
        raise RuntimeError("Five-fold manifest does not cover the training partition")
    fold_by_name = dict(zip(oof_manifest.file_name.astype(str), oof_manifest.oof_fold.astype(int), strict=True))
    if set(fold_by_name.values()) != set(range(5)):
        raise RuntimeError("Expected all five grouped folds")
    feature_folds = features.set_index("file_name").oof_fold.astype(int)
    for name, group in features.groupby("file_name"):
        if int(group.oof_fold.iloc[0]) != fold_by_name[str(name)]:
            raise RuntimeError(f"Component fold does not match observation manifest: {name}")

    masks = load_prediction_rows(BASE_PREDICTIONS, {Path(name).stem for name in training_names})
    rows: list[dict[str, Any]] = []
    for name in training_names:
        group = features.loc[features.file_name.astype(str) == name]
        component_count = len(masks[Path(name).stem])
        if component_count != len(group):
            raise RuntimeError(f"Component feature/prediction count mismatch for {name}")
        views = dataset.views_by_observation[name]
        gt_counts = [len(dataset.annotations_by_image_id[view.image_id]) for view in views]
        row: dict[str, Any] = {
            "file_name": name,
            "oof_fold": fold_by_name[name],
            "site": Path(name).stem[-2:],
            "expected_gt_count": float(np.mean(gt_counts)),
            "component_count": component_count,
        }
        component_ids = range(1, component_count + 1)
        p_core = np.asarray([core_scores_by_key[(name, i)] for i in component_ids], dtype=np.float64)
        p_forest = np.asarray([forest_scores_by_key[(name, i)] for i in component_ids], dtype=np.float64)
        p_blend = 0.70 * p_core + 0.30 * p_forest
        for prefix, values in (("core", p_core), ("rf", p_forest), ("blend", p_blend)):
            row[f"x_{prefix}_sum"] = float(values.sum()) if len(values) else 0.0
            row[f"x_{prefix}_max"] = float(values.max()) if len(values) else 0.0
            row[f"x_{prefix}_mean"] = float(values.mean()) if len(values) else 0.0
            row[f"x_{prefix}_std"] = float(values.std()) if len(values) else 0.0
            for q in (0.25, 0.5, 0.75):
                row[f"x_{prefix}_q{int(q * 100)}"] = float(np.quantile(values, q)) if len(values) else 0.0
            for threshold in (0.3, 0.4, 0.5, 0.6):
                row[f"x_{prefix}_n_gt_{int(threshold * 10)}"] = int(np.count_nonzero(values > threshold))
        for column in ("log_area", "eccentricity", "radius_mean", "dark_contrast", "detector_score_max"):
            values = group[column].to_numpy(dtype=np.float64)
            row[f"x_component_{column}_mean"] = float(values.mean()) if len(values) else 0.0
            row[f"x_component_{column}_max"] = float(values.max()) if len(values) else 0.0
        row["x_year"] = int(Path(name).stem[:4])
        rows.append(row)

    image_features = pd.DataFrame(rows)
    numeric_columns = [column for column in image_features if column.startswith("x_")]
    if not np.isfinite(image_features[numeric_columns].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("Non-finite image-level feature")
    output_dir.mkdir(parents=True, exist_ok=True)
    image_features.to_csv(output_dir / "oof-image-count-features.csv", index=False)

    fold_ids = image_features.oof_fold.to_numpy(dtype=int)
    count_predictions: dict[str, np.ndarray] = {}
    for name in REGRESSORS:
        values = np.full(len(image_features), np.nan, dtype=np.float64)
        for fold in range(5):
            held = fold_ids == fold
            values[held] = fit_count_model(name, image_features.loc[~held], image_features.loc[held])
            print(f"{name}: fold={fold} train={int((~held).sum())} held={int(held.sum())}", flush=True)
        if not np.isfinite(values).all():
            raise RuntimeError(f"Missing out-of-fold count predictions for {name}")
        count_predictions[name] = values

    names = training_names
    all_candidate_rows: list[dict[str, Any]] = []
    prediction_specs: dict[str, dict[tuple[str, int], float]] = {}
    for blend_name, (core_weight, rf_weight) in BLENDS.items():
        probability_lookup = {
            key: core_weight * core_scores_by_key[key] + rf_weight * forest_scores_by_key[key]
            for key in expected_keys
        }
        prediction_specs[blend_name] = probability_lookup
        for regressor_name, predicted_count in count_predictions.items():
            for scale in COUNT_SCALES:
                topk_scores: dict[tuple[str, int], float] = {}
                for row_index, row in image_features.iterrows():
                    file_name = str(row.file_name)
                    n_components = int(row.component_count)
                    keep_count = min(n_components, max(0, int(np.rint(predicted_count[row_index] * scale))))
                    ranked = sorted(
                        range(1, n_components + 1),
                        key=lambda component: probability_lookup[(file_name, component)],
                        reverse=True,
                    )
                    keep = set(ranked[:keep_count])
                    for component in range(1, n_components + 1):
                        topk_scores[(file_name, component)] = 1.0 if component in keep else 0.0
                predictions = predictions_from_masks(names, masks, topk_scores, 0.5)
                metrics, _ = score_validation(dataset, predictions, names)
                global_metrics = metrics["global"]
                all_candidate_rows.append({
                    "blend": blend_name,
                    "count_regressor": regressor_name,
                    "count_scale": scale,
                    "pq": float(global_metrics["pq"]),
                    "tp": int(global_metrics["tp"]),
                    "fp": int(global_metrics["fp"]),
                    "fn": int(global_metrics["fn"]),
                })
                print(f"{blend_name}/{regressor_name}/scale={scale:.2f}: OOF PQ={global_metrics['pq']:.6f}", flush=True)

    all_candidate_rows.sort(key=lambda row: row["pq"], reverse=True)
    pd.DataFrame(all_candidate_rows).to_csv(output_dir / "candidate-sweep.csv", index=False)
    model_outputs = pd.DataFrame({"file_name": image_features.file_name, "expected_gt_count": image_features.expected_gt_count})
    for name, values in count_predictions.items():
        model_outputs[name] = values
    model_outputs.to_csv(output_dir / "oof-count-predictions.csv", index=False)

    baseline_rows: list[dict[str, Any]] = []
    for blend_name in BLENDS:
        raw = prediction_specs[blend_name]
        for threshold in (0.40, 0.425, 0.45, 0.475, 0.50):
            predictions = predictions_from_masks(names, masks, raw, threshold)
            metrics, _ = score_validation(dataset, predictions, names)
            baseline_rows.append({
                "blend": blend_name, "threshold": threshold,
                "pq": float(metrics["global"]["pq"]),
                "tp": int(metrics["global"]["tp"]),
                "fp": int(metrics["global"]["fp"]),
                "fn": int(metrics["global"]["fn"]),
            })
    best_candidate = all_candidate_rows[0]
    best_regressor = best_candidate["count_regressor"]
    best_blend = best_candidate["blend"]
    best_scale = best_candidate["count_scale"]
    best_count = count_predictions[best_regressor]
    best_probability_lookup = prediction_specs[best_blend]
    best_topk: dict[tuple[str, int], float] = {}
    for row_index, row in image_features.iterrows():
        file_name = str(row.file_name)
        n_components = int(row.component_count)
        keep_count = min(n_components, max(0, int(np.rint(best_count[row_index] * best_scale))))
        ranked = sorted(range(1, n_components + 1), key=lambda component: best_probability_lookup[(file_name, component)], reverse=True)
        keep = set(ranked[:keep_count])
        for component in range(1, n_components + 1):
            best_topk[(file_name, component)] = 1.0 if component in keep else 0.0
    fold_deltas: list[dict[str, Any]] = []
    for fold in range(5):
        fold_names = sorted(image_features.loc[fold_ids == fold, "file_name"].astype(str))
        topk_predictions = predictions_from_masks(fold_names, masks, best_topk, 0.5)
        baseline_predictions = predictions_from_masks(fold_names, masks, best_probability_lookup, 0.45)
        topk_metrics, _ = score_validation(dataset, topk_predictions, fold_names)
        baseline_metrics, _ = score_validation(dataset, baseline_predictions, fold_names)
        fold_deltas.append({
            "fold": fold,
            "count_calibrated_pq": float(topk_metrics["global"]["pq"]),
            "global_threshold_pq": float(baseline_metrics["global"]["pq"]),
            "delta": float(topk_metrics["global"]["pq"] - baseline_metrics["global"]["pq"]),
        })

    report = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split_sha256": split.sha256,
        "oof_manifest_sha256": selector_metrics["oof_manifest_sha256"],
        "oof_observations": len(training_names),
        "component_rows": len(features),
        "holdout_used": False,
        "test_images_or_labels_used": False,
        "training_labels": "competition-provided annotation counts for grouped OOF observations only",
        "baseline_rf_blend_oof_pq": 0.4247906680196568,
        "best_global_thresholds": sorted(baseline_rows, key=lambda row: row["pq"], reverse=True)[:8],
        "top_count_calibrated_candidates": all_candidate_rows[:15],
        "selected_candidate_fold_deltas": fold_deltas,
        "count_prediction_rmse_by_model": {
            name: float(np.sqrt(np.mean((values - image_features.expected_gt_count.to_numpy()) ** 2)))
            for name, values in count_predictions.items()
        },
        "regressors": list(REGRESSORS),
        "count_scales": list(COUNT_SCALES),
        "input_sha256": {
            "selector_metrics": sha256(selector_metrics_path),
            "component_features": sha256(features_path),
            "core_probabilities": sha256(core_prob_path),
            "rf_probabilities": sha256(rf_prob_path),
            "base_oof_predictions": sha256(BASE_PREDICTIONS),
            "oof_manifest": sha256(OOF_MANIFEST),
        },
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "python": platform.python_version()},
        "runtime_seconds": time.perf_counter() - started,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"best_global_thresholds": report["best_global_thresholds"], "top_count_calibrated_candidates": report["top_count_calibrated_candidates"][:10], "fold_deltas": fold_deltas, "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
