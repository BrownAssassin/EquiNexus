# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Screen alternative component selectors using grouped OOF observations only."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
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
from filament_baseline.scorer import score_validation  # noqa: E402
from filament_baseline.split import create_observation_split  # noqa: E402


def _fit_model(
    model_name: str,
    train: pd.DataFrame,
    evaluate: pd.DataFrame,
) -> np.ndarray:
    x_train = train.loc[:, FEATURES].to_numpy(dtype=np.float64)
    x_eval = evaluate.loc[:, FEATURES].to_numpy(dtype=np.float64)
    y = train.target_keep.to_numpy(dtype=np.int8)
    group_sizes = train.groupby("file_name").component_id.transform("count").to_numpy(float)
    weights = 1.0 / np.maximum(group_sizes, 1.0)

    if model_name.startswith("logistic_c"):
        c_value = float(model_name.removeprefix("logistic_c"))
        scaler = StandardScaler().fit(x_train, sample_weight=weights)
        model = LogisticRegression(
            C=c_value,
            solver="liblinear",
            max_iter=1000,
            random_state=42,
        )
        model.fit(scaler.transform(x_train), y, sample_weight=weights)
        return model.predict_proba(scaler.transform(x_eval))[:, 1]

    if model_name == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=160,
            max_leaf_nodes=7,
            min_samples_leaf=25,
            l2_regularization=2.0,
            early_stopping=False,
            random_state=42,
        )
    elif model_name == "extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=15,
            max_features=0.8,
            n_jobs=4,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model screen entry: {model_name}")
    model.fit(x_train, y, sample_weight=weights)
    return model.predict_proba(x_eval)[:, 1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--selector-run", type=Path, default=REPO_ROOT / "artifacts/model-experiments/component-selector-official-pq-20261109-rle-fixed")
    parser.add_argument("--run-name", default="component-selector-model-screen-20261109")
    parser.add_argument("--output-dir", type=Path, default=None)
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
    selector_metrics = json.loads((selector_run / "metrics.json").read_text(encoding="utf-8"))
    if split.sha256 != selector_metrics["split_sha256"]:
        raise RuntimeError("Selector input does not use the frozen split")
    names = sorted(split.training_names)
    feature_path = selector_run / "oof-component-features-targets.csv"
    features = pd.read_csv(feature_path)
    if set(features.file_name) - set(names):
        raise RuntimeError("OOF component features contain non-training observations")
    if set(features.oof_fold.astype(int).unique()) != set(range(5)):
        raise RuntimeError("Expected all five frozen OOF folds")
    base_path = REPO_ROOT / "artifacts/model-experiments/oof-fusion-weight25-th035-a200/best-oof-predictions.csv"
    masks = load_prediction_rows(base_path, {Path(name).stem for name in names})
    output_dir.mkdir(parents=True, exist_ok=True)
    model_names = ("logistic_c0.25", "logistic_c1.0", "logistic_c4.0", "hist_gradient_boosting", "extra_trees")
    probabilities = pd.DataFrame({"file_name": features.file_name, "component_id": features.component_id})
    threshold_grid = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
    results: dict[str, Any] = {}
    for model_name in model_names:
        score_vector = np.full(len(features), np.nan, dtype=np.float64)
        for fold in range(5):
            heldout = features.oof_fold.to_numpy(dtype=int) == fold
            train = features.loc[~heldout]
            evaluate = features.loc[heldout]
            score_vector[heldout] = _fit_model(model_name, train, evaluate)
            print(
                f"{model_name}: fold={fold} train={len(train)} evaluate={len(evaluate)}",
                flush=True,
            )
        if not np.isfinite(score_vector).all():
            raise RuntimeError(f"Missing cross-fitted probabilities for {model_name}")
        probabilities[model_name] = score_vector
        keyed = {
            (str(file_name), int(component_id)): float(value)
            for file_name, component_id, value in zip(
                probabilities.file_name,
                probabilities.component_id,
                score_vector,
            )
        }
        threshold_metrics = []
        for threshold in threshold_grid:
            predictions = predictions_from_masks(names, masks, keyed, threshold)
            metrics, _ = score_validation(dataset, predictions, names)
            threshold_metrics.append({"threshold": threshold, "metrics": metrics})
            print(
                f"{model_name}: threshold={threshold:.2f} "
                f"OOF_PQ={metrics['global']['pq']:.6f}",
                flush=True,
            )
        best = max(threshold_metrics, key=lambda row: row["metrics"]["global"]["pq"])
        results[model_name] = {
            "best_threshold": best["threshold"],
            "best_metrics": best["metrics"],
            "thresholds": threshold_metrics,
        }

    probabilities.to_csv(output_dir / "oof-model-probabilities.csv", index=False)
    baseline = selector_metrics["fit"]["oof_selected"]["global"]["pq"]
    report = {
        "status": "complete",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split_sha256": split.sha256,
        "oof_manifest_sha256": selector_metrics["oof_manifest_sha256"],
        "oof_observations": len(names),
        "component_rows": len(features),
        "holdout_used": False,
        "test_images_or_labels_used": False,
        "training_labels": "competition-provided training annotations only",
        "metric_note": "Threshold and model screening use the cross-fitted OOF partition; results are selection-aware.",
        "logistic_c1_baseline_pq": baseline,
        "models": results,
        "probability_artifact": "oof-model-probabilities.csv",
        "runtime_seconds": time.perf_counter() - started,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ranking = sorted(
        (
            (name, value["best_metrics"]["global"]["pq"], value["best_threshold"])
            for name, value in results.items()
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    print(json.dumps({"ranking": ranking, "output": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
