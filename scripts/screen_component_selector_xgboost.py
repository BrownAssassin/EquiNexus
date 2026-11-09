# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Cross-fit XGBoost component selectors and evaluate one frozen holdout choice."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from pycocotools import mask as mask_utils

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from evaluate_component_selector_probability_blend_holdout import (  # noqa: E402
    paired_bootstrap,
    per_observation_rows,
)
from run_component_selector_oof import (  # noqa: E402
    FEATURES,
    load_prediction_rows,
    predictions_from_masks,
)
from screen_component_selector_models import _fit_model  # noqa: E402
from filament_baseline.data import load_dataset  # noqa: E402
from filament_baseline.predictor import PredictedInstance  # noqa: E402
from filament_baseline.runner import _write_predictions  # noqa: E402
from filament_baseline.scorer import score_validation  # noqa: E402
from filament_baseline.split import create_observation_split  # noqa: E402
from filament_baseline.visuals import write_comparison_overlays  # noqa: E402


FROZEN_SPLIT_SHA256 = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"
SELECTOR_RUN = REPO_ROOT / "artifacts/model-experiments/component-selector-official-pq-20261109-rle-fixed"
MODEL_SCREEN = REPO_ROOT / "artifacts/model-experiments/component-selector-model-screen-20261109"
BASE_PREDICTIONS = REPO_ROOT / "artifacts/model-experiments/oof-fusion-weight25-th035-a200/best-oof-predictions.csv"
OUTPUT_DIR = REPO_ROOT / "artifacts/model-experiments/component-selector-xgboost-20261109"
RF_INCUMBENT_PREDICTIONS = REPO_ROOT / "artifacts/model-experiments/component-selector-rf-blend-holdout-20261109-final/candidate-predictions.csv"
CORE_LOGISTIC_WEIGHT = 0.70
BLEND_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
THRESHOLDS = (0.35, 0.375, 0.40, 0.425, 0.45, 0.475, 0.50)
MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "xgb_depth2": {
        "n_estimators": 350, "learning_rate": 0.03, "max_depth": 2,
        "min_child_weight": 2.0, "reg_lambda": 8.0, "max_bin": 128,
    },
    "xgb_depth3": {
        "n_estimators": 350, "learning_rate": 0.03, "max_depth": 3,
        "min_child_weight": 3.0, "reg_lambda": 12.0, "max_bin": 128,
    },
    "xgb_depth4": {
        "n_estimators": 350, "learning_rate": 0.03, "max_depth": 4,
        "min_child_weight": 5.0, "reg_lambda": 20.0, "max_bin": 128,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_predict(train: pd.DataFrame, evaluate: pd.DataFrame, name: str) -> np.ndarray:
    group_sizes = train.groupby("file_name").component_id.transform("count").to_numpy(float)
    weights = 1.0 / np.maximum(group_sizes, 1.0)
    model = xgb.XGBClassifier(
        **MODEL_CONFIGS[name],
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=4,
        random_state=42,
        verbosity=0,
        tree_method="hist",
        device="cpu",
        subsample=0.9,
        colsample_bytree=0.9,
    )
    model.fit(
        train.loc[:, FEATURES].to_numpy(dtype=np.float32),
        train.target_keep.to_numpy(dtype=np.int8),
        sample_weight=weights,
    )
    return model.predict_proba(evaluate.loc[:, FEATURES].to_numpy(dtype=np.float32))[:, 1]


def keyed_probabilities(features: pd.DataFrame, values: np.ndarray) -> dict[tuple[str, int], float]:
    return {
        (str(name), int(component_id)): float(value)
        for name, component_id, value in zip(
            features.file_name, features.component_id, values, strict=True
        )
    }


def visual_review_selection(
    baseline_rows: list[Any], candidate_rows: list[Any]
) -> list[dict[str, Any]]:
    baseline = {row.file_name: row for row in baseline_rows}
    candidate = {row.file_name: row for row in candidate_rows}
    if set(baseline) != set(candidate) or not baseline:
        raise ValueError("Visual review requires matching observation-level scorer rows")
    deltas = {name: float(candidate[name].pq - baseline[name].pq) for name in baseline}
    best = max(deltas, key=deltas.get)
    median = min(deltas, key=lambda name: abs(deltas[name]))
    worst = min(deltas, key=deltas.get)
    roles: dict[str, list[str]] = {best: ["largest_gain"], median: ["closest_to_zero"]}
    roles.setdefault(worst, []).append("largest_loss")
    return [
        {"file_name": name, "roles": assigned, "pq_delta": deltas[name]}
        for name, assigned in sorted(roles.items())
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--selector-run", type=Path, default=SELECTOR_RUN)
    parser.add_argument("--model-screen", type=Path, default=MODEL_SCREEN)
    parser.add_argument("--base-predictions", type=Path, default=BASE_PREDICTIONS)
    parser.add_argument("--rf-incumbent-predictions", type=Path, default=RF_INCUMBENT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    data_root, selector_run = args.data_root.resolve(), args.selector_run.resolve()
    model_screen, base_path = args.model_screen.resolve(), args.base_predictions.resolve()
    rf_incumbent_path = args.rf_incumbent_predictions.resolve()
    output_dir = args.output_dir.resolve()
    if not data_root.is_relative_to((REPO_ROOT / "data").resolve()):
        raise ValueError("Competition data must stay under data/")
    for path, label in ((selector_run, "selector run"), (model_screen, "model screen"), (base_path, "base predictions"), (rf_incumbent_path, "RF incumbent predictions")):
        if not path.is_relative_to((REPO_ROOT / "artifacts").resolve()):
            raise ValueError(f"{label} must stay under artifacts/")
    if not output_dir.is_relative_to((REPO_ROOT / "artifacts/model-experiments").resolve()):
        raise ValueError("Outputs must stay under artifacts/model-experiments/")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)

    started = time.perf_counter()
    dataset = load_dataset(data_root)
    split = create_observation_split(dataset, REPO_ROOT / "data/splits")
    if split.sha256 != FROZEN_SPLIT_SHA256:
        raise RuntimeError(f"Frozen split changed: {split.sha256}")
    selector_metrics_path = selector_run / "metrics.json"
    selector_metrics = json.loads(selector_metrics_path.read_text(encoding="utf-8"))
    model_metrics_path = model_screen / "metrics.json"
    model_metrics = json.loads(model_metrics_path.read_text(encoding="utf-8"))
    if selector_metrics.get("split_sha256") != split.sha256:
        raise RuntimeError("Selector features use a different split")
    if model_metrics.get("split_sha256") != split.sha256 or model_metrics.get("holdout_used") is not False:
        raise RuntimeError("Probability screen must be holdout-free and use the frozen split")

    names = sorted(split.training_names)
    holdout_names = sorted(split.validation_names)
    train = pd.read_csv(selector_run / "oof-component-features-targets.csv")
    holdout = pd.read_csv(selector_run / "holdout-component-features.csv")
    probability_path = model_screen / "oof-model-probabilities.csv"
    base_probabilities = pd.read_csv(probability_path)
    folds = train.oof_fold.to_numpy(dtype=int)
    if set(np.unique(folds)) != set(range(5)):
        raise ValueError("Expected all five observation-grouped OOF folds")
    if set(train.file_name.astype(str)) - set(names):
        raise ValueError("OOF selector rows include observations outside the training partition")
    if set(holdout.file_name.astype(str)) - set(holdout_names):
        raise ValueError("Holdout selector rows include non-holdout observations")
    keys = ["file_name", "component_id"]
    indexed = base_probabilities.set_index(keys)
    if set(zip(train.file_name.astype(str), train.component_id.astype(int))) != set(indexed.index):
        raise ValueError("Base OOF probabilities do not exactly match selector rows")
    aligned = indexed.loc[pd.MultiIndex.from_frame(train[keys]), ["logistic_c1.0", "extra_trees"]]
    core_oof = (
        CORE_LOGISTIC_WEIGHT * aligned["logistic_c1.0"].to_numpy(dtype=np.float64)
        + (1.0 - CORE_LOGISTIC_WEIGHT) * aligned["extra_trees"].to_numpy(dtype=np.float64)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    probability_rows = train.loc[:, keys + ["oof_fold"]].copy()
    probability_rows["core_blend"] = core_oof
    oof_probabilities: dict[str, np.ndarray] = {}
    for name in MODEL_CONFIGS:
        scores = np.full(len(train), np.nan, dtype=np.float64)
        for fold in range(5):
            held = folds == fold
            scores[held] = fit_predict(train.loc[~held], train.loc[held], name)
            print(f"{name}: fold={fold} rows={int(held.sum())}", flush=True)
        if not np.isfinite(scores).all():
            raise RuntimeError(f"Incomplete grouped OOF probabilities for {name}")
        oof_probabilities[name] = scores
        probability_rows[name] = scores

    train_names = sorted(split.training_names)
    train_masks = load_prediction_rows(base_path, {Path(name).stem for name in train_names})
    oof_results: list[dict[str, Any]] = []
    for model_name, xgb_prob in oof_probabilities.items():
        for xgb_weight in BLEND_WEIGHTS:
            mixed = (1.0 - xgb_weight) * core_oof + xgb_weight * xgb_prob
            keyed = keyed_probabilities(train, mixed)
            for threshold in THRESHOLDS:
                predicted = predictions_from_masks(train_names, train_masks, keyed, threshold)
                metrics, _ = score_validation(dataset, predicted, train_names)
                global_metrics = metrics["global"]
                oof_results.append({
                    "model": model_name,
                    "xgboost_weight": xgb_weight,
                    "threshold": threshold,
                    "pq": float(global_metrics["pq"]),
                    "tp": int(global_metrics["tp"]),
                    "fp": int(global_metrics["fp"]),
                    "fn": int(global_metrics["fn"]),
                })
    oof_results.sort(key=lambda row: row["pq"], reverse=True)
    selected = oof_results[0]
    if selected["xgboost_weight"] <= 0.0:
        report = {
            "status": "no_promotion",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "split_sha256": split.sha256,
            "oof_manifest_sha256": selector_metrics.get("oof_manifest_sha256"),
            "training_observations": len(names),
            "holdout_observations": len(holdout_names),
            "component_rows": len(train),
            "holdout_scored": False,
            "holdout_used_for_model_selection": False,
            "test_images_or_labels_used": False,
            "external_labels_used": False,
            "xgboost_version": xgb.__version__,
            "python_version": platform.python_version(),
            "execution_device": "CPU",
            "logical_cpu_count": os.cpu_count(),
            "source_script_sha256": sha256(Path(__file__).resolve()),
            "model_configs": MODEL_CONFIGS,
            "blend_weights": BLEND_WEIGHTS,
            "thresholds": THRESHOLDS,
            "selected_oof_candidate": selected,
            "top_oof_candidates": oof_results[:20],
            "input_sha256": {
                "selector_metrics": sha256(selector_metrics_path),
                "oof_features_targets": sha256(selector_run / "oof-component-features-targets.csv"),
                "model_metrics": sha256(model_metrics_path),
                "base_oof_probabilities": sha256(probability_path),
                "base_oof_predictions": sha256(base_path),
            },
            "runtime_seconds": time.perf_counter() - started,
        }
        probability_rows.to_csv(output_dir / "oof-probabilities.csv", index=False)
        pd.DataFrame(oof_results).to_csv(output_dir / "oof-screen.csv", index=False)
        (output_dir / "metrics.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps({"status": report["status"], "selected_oof_candidate": selected, "output_dir": str(output_dir)}, indent=2))
        return 0

    holdout_base_path = selector_run / "baseline-holdout-predictions.csv"
    holdout_masks = load_prediction_rows(holdout_base_path, {Path(name).stem for name in holdout_names})
    fitted_xgb = fit_predict(train, holdout, str(selected["model"]))
    logistic = _fit_model("logistic_c1.0", train, holdout)
    trees = _fit_model("extra_trees", train, holdout)
    core_holdout = CORE_LOGISTIC_WEIGHT * logistic + (1.0 - CORE_LOGISTIC_WEIGHT) * trees
    candidate_holdout = (1.0 - float(selected["xgboost_weight"])) * core_holdout + float(selected["xgboost_weight"]) * fitted_xgb
    baseline_keyed = keyed_probabilities(holdout, core_holdout)
    candidate_keyed = keyed_probabilities(holdout, candidate_holdout)
    baseline_predictions = predictions_from_masks(holdout_names, holdout_masks, baseline_keyed, 0.45)
    candidate_predictions = predictions_from_masks(holdout_names, holdout_masks, candidate_keyed, float(selected["threshold"]))
    baseline_metrics, baseline_rows = score_validation(dataset, baseline_predictions, holdout_names)
    candidate_metrics, candidate_rows = score_validation(dataset, candidate_predictions, holdout_names)
    bootstrap = paired_bootstrap(baseline_rows, candidate_rows)
    rf_masks = load_prediction_rows(rf_incumbent_path, {Path(name).stem for name in holdout_names})
    rf_predictions = {}
    for name in holdout_names:
        stem = Path(name).stem
        instances = []
        for index, rle in enumerate(rf_masks[stem], start=1):
            counts = rle["counts"]
            encoded = {"size": rle["size"], "counts": counts.encode("ascii") if isinstance(counts, str) else counts}
            instances.append(PredictedInstance(f"{stem}_{index}", stem, int(mask_utils.area(encoded)), rle))
        rf_predictions[name] = instances
    rf_metrics, rf_rows = score_validation(dataset, rf_predictions, holdout_names)
    rf_bootstrap = paired_bootstrap(rf_rows, candidate_rows)
    visual_selection = visual_review_selection(baseline_rows, candidate_rows)
    for observation in visual_selection:
        name = observation["file_name"]
        for label, predictions in (("baseline", baseline_predictions), ("xgboost", candidate_predictions)):
            write_comparison_overlays(
                dataset,
                {name: predictions[name]},
                output_dir / "overlays" / name / label,
                maximum=1,
            )
    pd.DataFrame(visual_selection).to_csv(output_dir / "visual-observation-selection.csv", index=False)
    _write_predictions(output_dir / "baseline-predictions.csv", baseline_predictions)
    _write_predictions(output_dir / "candidate-predictions.csv", candidate_predictions)
    pd.DataFrame(per_observation_rows(baseline_rows)).to_csv(output_dir / "baseline-per-observation.csv", index=False)
    pd.DataFrame(per_observation_rows(candidate_rows)).to_csv(output_dir / "candidate-per-observation.csv", index=False)
    probability_rows.to_csv(output_dir / "oof-probabilities.csv", index=False)
    pd.DataFrame(oof_results).to_csv(output_dir / "oof-screen.csv", index=False)
    report = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split_sha256": split.sha256,
        "oof_manifest_sha256": selector_metrics.get("oof_manifest_sha256"),
        "training_observations": len(train_names),
        "holdout_observations": len(holdout_names),
        "component_rows": len(train),
        "holdout_component_rows": len(holdout),
        "holdout_used_for_model_selection": False,
        "test_images_or_labels_used": False,
        "external_labels_used": False,
        "training_labels": "competition-provided grouped OOF training observations only",
        "xgboost_version": xgb.__version__,
        "python_version": platform.python_version(),
        "execution_device": "CPU",
        "logical_cpu_count": os.cpu_count(),
        "source_script_sha256": sha256(Path(__file__).resolve()),
        "model_configs": MODEL_CONFIGS,
        "blend_weights": BLEND_WEIGHTS,
        "thresholds": THRESHOLDS,
        "selected_oof_candidate": selected,
        "top_oof_candidates": oof_results[:20],
        "holdout_baseline": baseline_metrics["global"],
        "holdout_candidate": candidate_metrics["global"],
        "paired_observation_bootstrap": bootstrap,
        "rf_incumbent": rf_metrics["global"],
        "paired_bootstrap_vs_rf_incumbent": rf_bootstrap,
        "visual_review": {
            "selection": visual_selection,
            "overlay_scope": "first annotator view per selected observation; aggregate scores use every annotator view",
        },
        "input_sha256": {
            "selector_metrics": sha256(selector_metrics_path),
            "oof_features_targets": sha256(selector_run / "oof-component-features-targets.csv"),
            "holdout_features": sha256(selector_run / "holdout-component-features.csv"),
            "model_metrics": sha256(model_metrics_path),
            "base_oof_probabilities": sha256(probability_path),
            "base_oof_predictions": sha256(base_path),
            "holdout_base_predictions": sha256(holdout_base_path),
            "rf_incumbent_predictions": sha256(rf_incumbent_path),
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "selected_oof_candidate": selected,
        "holdout_baseline": baseline_metrics["global"],
        "holdout_candidate": candidate_metrics["global"],
        "bootstrap_delta": bootstrap["paired_delta_candidate_minus_baseline"],
        "rf_incumbent": rf_metrics["global"],
        "bootstrap_delta_vs_rf": rf_bootstrap["paired_delta_candidate_minus_baseline"],
        "output_dir": str(output_dir),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
