# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Cross-fit continuous component-impact regressors with the official scorer."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_component_selector_oof import (  # noqa: E402
    FEATURES,
    load_prediction_rows,
    predictions_from_masks,
)
from filament_baseline.data import load_dataset  # noqa: E402
from filament_baseline.runner import _write_observation_scores, _write_predictions  # noqa: E402
from filament_baseline.scorer import score_validation  # noqa: E402
from filament_baseline.split import create_observation_split  # noqa: E402


LIGHTGBM_REGRESSOR_CONFIGS: dict[str, dict[str, Any]] = {
    "lightgbm_leaf3_min25": {
        "n_estimators": 250, "learning_rate": 0.03, "num_leaves": 3,
        "max_depth": 2, "min_child_samples": 25, "reg_lambda": 4.0,
    },
    "lightgbm_leaf7_min30": {
        "n_estimators": 250, "learning_rate": 0.03, "num_leaves": 7,
        "max_depth": 3, "min_child_samples": 30, "reg_lambda": 5.0,
    },
    "lightgbm_leaf15_min50": {
        "n_estimators": 250, "learning_rate": 0.03, "num_leaves": 15,
        "max_depth": 4, "min_child_samples": 50, "reg_lambda": 8.0,
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observation_weights(rows: pd.DataFrame) -> np.ndarray:
    group_sizes = rows.groupby("file_name").component_id.transform("count").to_numpy(float)
    return 1.0 / np.maximum(group_sizes, 1.0)


def fit_regressor(
    name: str,
    train: pd.DataFrame,
    evaluate: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    x_train = train.loc[:, FEATURES].to_numpy(dtype=np.float64)
    x_eval = evaluate.loc[:, FEATURES].to_numpy(dtype=np.float64)
    y_train = train.keep_delta_pq.to_numpy(dtype=np.float64)
    weights = observation_weights(train)

    if name == "ridge":
        scaler = StandardScaler().fit(x_train, sample_weight=weights)
        model = Ridge(alpha=1.0)
        model.fit(scaler.transform(x_train), y_train, sample_weight=weights)
        result = model.predict(scaler.transform(x_eval))
        metadata = {"family": name, "alpha": 1.0}
    elif name == "hist_gradient_boosting":
        model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=160,
            max_leaf_nodes=7,
            min_samples_leaf=25,
            l2_regularization=2.0,
            early_stopping=False,
            random_state=42,
        )
        model.fit(x_train, y_train, sample_weight=weights)
        result = model.predict(x_eval)
        metadata = {"family": name, "max_iter": 160, "max_leaf_nodes": 7}
    elif name == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=15,
            max_features=0.8,
            n_jobs=4,
            random_state=42,
        )
        model.fit(x_train, y_train, sample_weight=weights)
        result = model.predict(x_eval)
        metadata = {"family": name, "n_estimators": 300, "max_depth": 6}
    elif name in LIGHTGBM_REGRESSOR_CONFIGS:
        config = LIGHTGBM_REGRESSOR_CONFIGS[name]
        model = lgb.LGBMRegressor(
            **config,
            objective="regression_l2",
            n_jobs=4,
            random_state=42,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
        )
        model.fit(x_train, y_train, sample_weight=weights)
        result = model.predict(x_eval)
        metadata = {"family": name, **config, "objective": "regression_l2"}
    else:
        raise ValueError(f"Unknown regressor: {name}")

    return np.asarray(result, dtype=np.float64), metadata


def counts(rows: list[Any]) -> np.ndarray:
    return np.asarray(
        [[row.tp_iou_sum, row.tp, row.fp, row.fn] for row in rows], dtype=np.float64
    )


def aggregate_pq(samples: np.ndarray) -> np.ndarray:
    iou_sum = samples[..., 0].sum(axis=1)
    tp = samples[..., 1].sum(axis=1)
    fp = samples[..., 2].sum(axis=1)
    fn = samples[..., 3].sum(axis=1)
    denominator = tp + 0.5 * fp + 0.5 * fn
    return np.divide(iou_sum, denominator, out=np.zeros_like(iou_sum), where=denominator > 0)


def paired_bootstrap(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    resamples: int = 20_000,
    seed: int = 42,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(baseline), size=(resamples, len(baseline)))
    deltas = aggregate_pq(candidate[indices]) - aggregate_pq(baseline[indices])
    return {
        "delta": float(aggregate_pq(candidate[None, ...])[0] - aggregate_pq(baseline[None, ...])[0]),
        "paired_delta_ci_95": [float(value) for value in np.quantile(deltas, [0.025, 0.975])],
        "probability_candidate_better": float(np.mean(deltas > 0)),
        "resamples": resamples,
        "seed": seed,
    }


def read_counts(path: Path) -> np.ndarray:
    frame = pd.read_csv(path).sort_values("file_name")
    return frame[["tp_iou_sum", "tp_edges", "fp", "fn"]].to_numpy(dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--selector-run", type=Path, default=REPO_ROOT / "artifacts/model-experiments/component-selector-official-pq-20261109-rle-fixed")
    parser.add_argument("--run-name", default="component-selector-regression-screen-20261109")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resamples", type=int, default=20_000)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    selector_run = args.selector_run.resolve()
    if not data_root.is_relative_to((REPO_ROOT / "data").resolve()):
        raise ValueError("Dataset must remain inside data/")
    if not selector_run.is_relative_to((REPO_ROOT / "artifacts").resolve()):
        raise ValueError("Selector inputs must remain inside artifacts/")
    output_dir = (args.output_dir or (REPO_ROOT / "artifacts/model-experiments" / args.run_name)).resolve()
    if not output_dir.is_relative_to((REPO_ROOT / "artifacts/model-experiments").resolve()):
        raise ValueError("Screen outputs must remain inside artifacts/model-experiments/")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)

    started = time.perf_counter()
    dataset = load_dataset(data_root)
    split = create_observation_split(dataset, REPO_ROOT / "data/splits")
    selector_metrics_path = selector_run / "metrics.json"
    selector_metrics = json.loads(selector_metrics_path.read_text(encoding="utf-8"))
    if split.sha256 != selector_metrics.get("split_sha256"):
        raise RuntimeError("Selector input does not use the frozen holdout split")

    oof_names = sorted(split.training_names)
    holdout_names = sorted(split.validation_names)
    features_path = selector_run / "oof-component-features-targets.csv"
    holdout_features_path = selector_run / "holdout-component-features.csv"
    features = pd.read_csv(features_path)
    holdout_features = pd.read_csv(holdout_features_path)
    if set(features.file_name) - set(oof_names):
        raise RuntimeError("OOF component features include observations outside the training partition")
    if set(holdout_features.file_name) - set(holdout_names):
        raise RuntimeError("Holdout component features include observations outside the frozen holdout")
    folds = features.oof_fold.to_numpy(dtype=int)
    if set(np.unique(folds)) != set(range(5)):
        raise RuntimeError("Expected all five observation-grouped OOF folds")

    oof_mask_path = REPO_ROOT / "artifacts/model-experiments/oof-fusion-weight25-th035-a200/best-oof-predictions.csv"
    holdout_mask_path = REPO_ROOT / "artifacts/model-experiments/yolo11s-25perann-75consensus-foldmeans-valid-holdout-th035-a200/predictions.csv"
    oof_masks = load_prediction_rows(oof_mask_path, {Path(name).stem for name in oof_names})
    holdout_masks = load_prediction_rows(holdout_mask_path, {Path(name).stem for name in holdout_names})
    model_names = (
        "ridge", "hist_gradient_boosting", "extra_trees",
        *LIGHTGBM_REGRESSOR_CONFIGS.keys(),
    )
    quantiles = np.linspace(0.0, 1.0, 11)
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_rows: list[dict[str, Any]] = []
    fitted_oof: dict[str, np.ndarray] = {}
    for model_name in model_names:
        score_vector = np.full(len(features), np.nan, dtype=np.float64)
        for fold in range(5):
            held = folds == fold
            score_vector[held], _ = fit_regressor(
                model_name, features.loc[~held], features.loc[held]
            )
            print(f"{model_name}: fold={fold} train={(~held).sum()} held={held.sum()}", flush=True)
        if not np.isfinite(score_vector).all():
            raise RuntimeError(f"Missing cross-fitted scores for {model_name}")
        fitted_oof[model_name] = score_vector
        score_by_key = {
            (str(row.file_name), int(row.component_id)): float(score)
            for row, score in zip(features.itertuples(), score_vector)
        }
        thresholds = sorted({float(np.quantile(score_vector, q)) for q in quantiles} | {0.0})
        for threshold in thresholds:
            predictions = predictions_from_masks(oof_names, oof_masks, score_by_key, threshold)
            metrics, _ = score_validation(dataset, predictions, oof_names)
            sweep_rows.append(
                {
                    "model": model_name,
                    "threshold": threshold,
                    "oof_pq": float(metrics["global"]["pq"]),
                    "tp": int(metrics["global"]["tp"]),
                    "fp": int(metrics["global"]["fp"]),
                    "fn": int(metrics["global"]["fn"]),
                    "selected_components": int(sum(len(v) for v in predictions.values())),
                }
            )
        print(f"{model_name}: scorer threshold sweep complete", flush=True)

    sweep = pd.DataFrame(sweep_rows).sort_values(["oof_pq", "model", "threshold"], ascending=[False, True, False])
    sweep.to_csv(output_dir / "oof-threshold-sweep.csv", index=False)
    best = sweep.iloc[0]
    best_model = str(best.model)
    best_threshold = float(best.threshold)
    best_oof_scores = fitted_oof[best_model]
    oof_score_by_key = {
        (str(row.file_name), int(row.component_id)): float(score)
        for row, score in zip(features.itertuples(), best_oof_scores)
    }
    best_oof_predictions = predictions_from_masks(oof_names, oof_masks, oof_score_by_key, best_threshold)
    best_oof_metrics, best_oof_rows = score_validation(dataset, best_oof_predictions, oof_names)

    holdout_scores, model_metadata = fit_regressor(best_model, features, holdout_features)
    holdout_score_by_key = {
        (str(row.file_name), int(row.component_id)): float(score)
        for row, score in zip(holdout_features.itertuples(), holdout_scores)
    }
    holdout_predictions = predictions_from_masks(holdout_names, holdout_masks, holdout_score_by_key, best_threshold)
    holdout_metrics, holdout_rows = score_validation(dataset, holdout_predictions, holdout_names)
    _write_predictions(output_dir / "best-regression-oof-predictions.csv", best_oof_predictions)
    _write_observation_scores(output_dir / "best-regression-oof-per-observation.csv", best_oof_rows, best_oof_predictions)
    _write_predictions(output_dir / "best-regression-holdout-predictions.csv", holdout_predictions)
    _write_observation_scores(output_dir / "best-regression-holdout-per-observation.csv", holdout_rows, holdout_predictions)

    baseline_oof = read_counts(selector_run / "selector-oof-per-observation.csv")
    baseline_holdout = read_counts(selector_run / "selector-holdout-per-observation.csv")
    candidate_oof = counts(best_oof_rows)
    candidate_holdout = counts(holdout_rows)
    metrics_result = {
        "status": "complete",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split_sha256": split.sha256,
        "oof_manifest_sha256": selector_metrics["oof_manifest_sha256"],
        "data_boundary": {
            "training_labels": "competition-provided annotations for the 565-observation OOF training partition",
            "holdout_labels": "competition-provided annotations used once for the selected OOF candidate comparison",
            "test_images_or_labels_used": False,
        },
        "target": "continuous component marginal PQ contribution (keep_delta_pq)",
        "model_families": list(model_names),
        "lightgbm_version": lgb.__version__,
        "python_version": platform.python_version(),
        "execution_device": "CPU",
        "threshold_quantiles": [float(value) for value in quantiles],
        "selected_model": best_model,
        "selected_threshold": best_threshold,
        "selected_model_parameters": model_metadata,
        "primary_selector_oof_pq": float(selector_metrics["fit"]["oof_selected"]["global"]["pq"]),
        "regression_oof_metrics": best_oof_metrics,
        "regression_holdout_metrics": holdout_metrics,
        "paired_bootstrap_vs_primary_selector": {
            "oof": paired_bootstrap(baseline_oof, candidate_oof, resamples=args.resamples),
            "holdout": paired_bootstrap(baseline_holdout, candidate_holdout, resamples=args.resamples),
        },
        "runtime_seconds": time.perf_counter() - started,
        "source_sha256": {
            "selector_metrics": sha256(selector_metrics_path),
            "oof_component_features_targets": sha256(features_path),
            "script": sha256(Path(__file__).resolve()),
        },
        "artifacts": {
            "threshold_sweep": "oof-threshold-sweep.csv",
            "oof_predictions": "best-regression-oof-predictions.csv",
            "oof_per_observation": "best-regression-oof-per-observation.csv",
            "holdout_predictions": "best-regression-holdout-predictions.csv",
            "holdout_per_observation": "best-regression-holdout-per-observation.csv",
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "selected_model": best_model,
                "threshold": best_threshold,
                "oof_pq": best_oof_metrics["global"]["pq"],
                "holdout_pq": holdout_metrics["global"]["pq"],
                "runtime_seconds": metrics_result["runtime_seconds"],
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
