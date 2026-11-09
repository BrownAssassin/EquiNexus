# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Cross-fit random-forest component selectors on the frozen OOF partition."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
from filament_baseline.data import load_dataset  # noqa: E402
from filament_baseline.scorer import score_validation  # noqa: E402
from filament_baseline.split import create_observation_split  # noqa: E402


FROZEN_SPLIT_SHA256 = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"
DEFAULT_SELECTOR_RUN = REPO_ROOT / "artifacts/model-experiments/component-selector-official-pq-20261109-rle-fixed"
DEFAULT_BASE_PREDICTIONS = REPO_ROOT / "artifacts/model-experiments/oof-fusion-weight25-th035-a200/best-oof-predictions.csv"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/model-experiments/component-selector-random-forest-oof-20261109"
THRESHOLDS = tuple(float(value) for value in np.arange(0.25, 0.7501, 0.025))
MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "rf_depth8_leaf10": {
        "n_estimators": 400, "max_depth": 8, "min_samples_leaf": 10,
        "max_features": 0.8, "class_weight": None,
    },
    "rf_depth12_leaf20_balanced": {
        "n_estimators": 400, "max_depth": 12, "min_samples_leaf": 20,
        "max_features": 0.8, "class_weight": "balanced_subsample",
    },
    "rf_unbounded_leaf30": {
        "n_estimators": 400, "max_depth": None, "min_samples_leaf": 30,
        "max_features": 0.8, "class_weight": None,
    },
    "rf_depth10_leaf5_balanced": {
        "n_estimators": 400, "max_depth": 10, "min_samples_leaf": 5,
        "max_features": 0.8, "class_weight": "balanced_subsample",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fit_predict(
    model_name: str,
    train: pd.DataFrame,
    evaluate: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    config = dict(MODEL_CONFIGS[model_name])
    config.update({"n_jobs": 4, "random_state": 42})
    group_sizes = train.groupby("file_name").component_id.transform("count").to_numpy(float)
    weights = 1.0 / np.maximum(group_sizes, 1.0)
    model = RandomForestClassifier(**config)
    model.fit(
        train.loc[:, FEATURES].to_numpy(dtype=np.float32),
        train.target_keep.to_numpy(dtype=np.int8),
        sample_weight=weights,
    )
    return model.predict_proba(evaluate.loc[:, FEATURES].to_numpy(dtype=np.float32))[:, 1], config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--selector-run", type=Path, default=DEFAULT_SELECTOR_RUN)
    parser.add_argument("--base-predictions", type=Path, default=DEFAULT_BASE_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    selector_run = args.selector_run.resolve()
    base_path = args.base_predictions.resolve()
    output_dir = args.output_dir.resolve()
    if not data_root.is_relative_to((REPO_ROOT / "data").resolve()):
        raise ValueError("Competition data must remain under data/")
    for path, label in ((selector_run, "selector run"), (base_path, "base predictions")):
        if not path.is_relative_to((REPO_ROOT / "artifacts").resolve()):
            raise ValueError(f"{label} must remain under artifacts/")
    if not output_dir.is_relative_to((REPO_ROOT / "artifacts/model-experiments").resolve()):
        raise ValueError("Outputs must remain under artifacts/model-experiments/")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)

    started = time.perf_counter()
    dataset = load_dataset(data_root)
    split = create_observation_split(dataset, REPO_ROOT / "data/splits")
    selector_metrics_path = selector_run / "metrics.json"
    selector_metrics = json.loads(selector_metrics_path.read_text(encoding="utf-8"))
    if split.sha256 != FROZEN_SPLIT_SHA256 or selector_metrics.get("split_sha256") != split.sha256:
        raise RuntimeError("Selector source does not use the frozen grouped split")
    names = sorted(split.training_names)
    features_path = selector_run / "oof-component-features-targets.csv"
    features = pd.read_csv(features_path)
    if set(features.file_name.astype(str)) - set(names):
        raise RuntimeError("OOF features include observations outside the training partition")
    folds = features.oof_fold.to_numpy(dtype=int)
    if set(np.unique(folds)) != set(range(5)):
        raise RuntimeError("Expected all five grouped OOF folds")
    masks = load_prediction_rows(base_path, {Path(name).stem for name in names})
    if sum(len(masks[Path(name).stem]) for name in names) == 0:
        raise RuntimeError("Base prediction set is empty")

    output_dir.mkdir(parents=True, exist_ok=True)
    probability_rows = features.loc[:, ["file_name", "component_id"]].copy()
    model_results: dict[str, Any] = {}
    for model_name in MODEL_CONFIGS:
        scores = np.full(len(features), np.nan, dtype=np.float64)
        fitted_parameters = None
        for fold in range(5):
            held = folds == fold
            scores[held], fitted_parameters = _fit_predict(
                model_name, features.loc[~held], features.loc[held]
            )
            print(
                f"{model_name}: fold={fold} train={int((~held).sum())} held={int(held.sum())}",
                flush=True,
            )
        if not np.isfinite(scores).all():
            raise RuntimeError(f"Missing cross-fitted scores for {model_name}")
        probability_rows[model_name] = scores
        keyed = {
            (str(name), int(component_id)): float(score)
            for name, component_id, score in zip(
                probability_rows.file_name, probability_rows.component_id, scores
            )
        }
        thresholds = []
        for threshold in THRESHOLDS:
            predictions = predictions_from_masks(names, masks, keyed, threshold)
            metrics, _ = score_validation(dataset, predictions, names)
            global_metrics = metrics["global"]
            thresholds.append({
                "threshold": threshold,
                "pq": float(global_metrics["pq"]),
                "tp": int(global_metrics["tp"]),
                "fp": int(global_metrics["fp"]),
                "fn": int(global_metrics["fn"]),
                "prediction_count": int(global_metrics["prediction_count_across_annotator_views"]),
            })
        best = max(thresholds, key=lambda row: row["pq"])
        model_results[model_name] = {
            "parameters": fitted_parameters,
            "best_oof": best,
            "threshold_sweep": thresholds,
        }
        print(f"{model_name}: best OOF PQ={best['pq']:.6f} at {best['threshold']:.3f}", flush=True)

    probabilities_path = output_dir / "oof-model-probabilities.csv"
    probability_rows.to_csv(probabilities_path, index=False)
    report = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split_sha256": split.sha256,
        "oof_manifest_sha256": selector_metrics.get("oof_manifest_sha256"),
        "oof_observations": len(names),
        "component_rows": len(features),
        "holdout_used": False,
        "test_images_or_labels_used": False,
        "training_labels": "competition-provided annotations for the frozen OOF training partition only",
        "primary_logistic_selector_oof_pq": float(selector_metrics["fit"]["oof_selected"]["global"]["pq"]),
        "model_configs": MODEL_CONFIGS,
        "models": model_results,
        "threshold_grid": list(THRESHOLDS),
        "input_sha256": {
            "selector_metrics": sha256(selector_metrics_path),
            "oof_features_targets": sha256(features_path),
            "base_oof_predictions": sha256(base_path),
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    ranking = sorted(
        ((name, value["best_oof"]["pq"], value["best_oof"]["threshold"])
         for name, value in model_results.items()),
        key=lambda row: row[1], reverse=True,
    )
    print(json.dumps({"ranking": ranking, "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
