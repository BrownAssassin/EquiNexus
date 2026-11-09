# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Screen utility-weighted component classifiers on grouped OOF observations."""

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_component_selector_oof import FEATURES, load_prediction_rows, predictions_from_masks  # noqa: E402
from filament_baseline.data import load_dataset  # noqa: E402
from filament_baseline.scorer import score_validation  # noqa: E402
from filament_baseline.split import create_observation_split  # noqa: E402


FROZEN_SPLIT_SHA256 = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"
SELECTOR_RUN = REPO_ROOT / "artifacts/model-experiments/component-selector-official-pq-20261109-rle-fixed"
MODEL_SCREEN = REPO_ROOT / "artifacts/model-experiments/component-selector-model-screen-20261109"
RF_SCREEN = REPO_ROOT / "artifacts/model-experiments/component-selector-random-forest-oof-20261109"
BASE_PREDICTIONS = REPO_ROOT / "artifacts/model-experiments/oof-fusion-weight25-th035-a200/best-oof-predictions.csv"
OOF_MANIFEST = REPO_ROOT / "data/splits/observation_oof_seed42_5fold.csv"
OUTPUT_DIR = REPO_ROOT / "artifacts/model-experiments/component-selector-utility-weighted-20261109"
GAMMAS = (0.0, 0.25, 0.5, 1.0)
MODELS = ("logistic_c1", "rf_depth8_leaf10")
THRESHOLDS = (0.35, 0.375, 0.40, 0.425, 0.45, 0.475, 0.50)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def observation_weights(train: pd.DataFrame, gamma: float) -> np.ndarray:
    sizes = train.groupby("file_name").component_id.transform("count").to_numpy(dtype=np.float64)
    base = 1.0 / np.maximum(sizes, 1.0)
    if gamma == 0:
        return base
    utility = np.maximum(np.abs(train.keep_delta_pq.to_numpy(dtype=np.float64)), 0.005)
    cap = float(np.quantile(utility, 0.95))
    utility = np.minimum(utility, max(cap, 0.005)) ** gamma
    normalized = pd.Series(utility).groupby(train.file_name.reset_index(drop=True)).transform("mean").to_numpy()
    return base * utility / np.maximum(normalized, 1e-12)


def fit_predict(name: str, train: pd.DataFrame, evaluate: pd.DataFrame, gamma: float) -> np.ndarray:
    x_train = train.loc[:, FEATURES].to_numpy(dtype=np.float64)
    x_eval = evaluate.loc[:, FEATURES].to_numpy(dtype=np.float64)
    y = train.target_keep.to_numpy(dtype=np.int8)
    weights = observation_weights(train, gamma)
    if name == "logistic_c1":
        scaler = StandardScaler().fit(x_train, sample_weight=weights)
        model = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000, random_state=42)
        model.fit(scaler.transform(x_train), y, sample_weight=weights)
        return model.predict_proba(scaler.transform(x_eval))[:, 1]
    if name == "rf_depth8_leaf10":
        model = RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=10, max_features=0.8,
            n_jobs=4, random_state=42,
        )
        model.fit(x_train, y, sample_weight=weights)
        return model.predict_proba(x_eval)[:, 1]
    raise ValueError(f"Unknown classifier: {name}")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--selector-run", type=Path, default=SELECTOR_RUN)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    data_root, selector_run, output_dir = args.data_root.resolve(), args.selector_run.resolve(), args.output_dir.resolve()
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
        raise RuntimeError(f"Frozen split changed: {split.sha256}")
    selector_metrics_path = selector_run / "metrics.json"
    selector_metrics = json.loads(selector_metrics_path.read_text(encoding="utf-8"))
    if selector_metrics.get("split_sha256") != split.sha256:
        raise RuntimeError("Selector features use another split")
    for path, label in ((MODEL_SCREEN / "metrics.json", "core selector"), (RF_SCREEN / "metrics.json", "random forest")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        if metrics.get("split_sha256") != split.sha256 or metrics.get("holdout_used") is not False:
            raise RuntimeError(f"{label} source is not frozen, holdout-free OOF")
        if metrics.get("test_images_or_labels_used") is not False:
            raise RuntimeError(f"{label} source reports test use")

    names = sorted(split.training_names)
    features_path = selector_run / "oof-component-features-targets.csv"
    features = pd.read_csv(features_path)
    if set(features.file_name.astype(str)) - set(names):
        raise RuntimeError("OOF component rows include validation observations")
    fold_manifest = pd.read_csv(OOF_MANIFEST)
    fold_by_name = dict(zip(fold_manifest.file_name.astype(str), fold_manifest.oof_fold.astype(int), strict=True))
    if set(fold_by_name) != set(names) or set(fold_by_name.values()) != set(range(5)):
        raise RuntimeError("Frozen OOF manifest has incomplete observation coverage")
    folds = features.oof_fold.to_numpy(dtype=int)
    for file_name, group in features.groupby("file_name"):
        if int(group.oof_fold.iloc[0]) != fold_by_name[str(file_name)]:
            raise RuntimeError(f"Component OOF fold disagrees with observation manifest: {file_name}")
    masks = load_prediction_rows(BASE_PREDICTIONS, {Path(name).stem for name in names})

    # Align the incumbent probabilities by (observation, component), never by row order.
    core_path, rf_path = MODEL_SCREEN / "oof-model-probabilities.csv", RF_SCREEN / "oof-model-probabilities.csv"
    core_rows, rf_rows = pd.read_csv(core_path), pd.read_csv(rf_path)
    key_cols = ["file_name", "component_id"]
    expected = set(zip(features.file_name.astype(str), features.component_id.astype(int), strict=True))
    for rows, label in ((core_rows, "core"), (rf_rows, "forest")):
        if set(zip(rows.file_name.astype(str), rows.component_id.astype(int), strict=True)) != expected:
            raise RuntimeError(f"{label} probability rows do not match selector features")
    idx = pd.MultiIndex.from_frame(features[key_cols])
    core_lookup, rf_lookup = core_rows.set_index(key_cols), rf_rows.set_index(key_cols)
    core_prob = 0.70 * core_lookup.loc[idx, "logistic_c1.0"].to_numpy(dtype=np.float64) + 0.30 * core_lookup.loc[idx, "extra_trees"].to_numpy(dtype=np.float64)
    rf_prob = rf_lookup.loc[idx, "rf_depth8_leaf10"].to_numpy(dtype=np.float64)
    core_prob_map = {(str(n), int(c)): float(p) for n, c, p in zip(features.file_name, features.component_id, core_prob, strict=True)}
    rf_prob_map = {(str(n), int(c)): float(p) for n, c, p in zip(features.file_name, features.component_id, rf_prob, strict=True)}

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame({"file_name": features.file_name.astype(str), "component_id": features.component_id.astype(int)})
    candidates: list[dict[str, Any]] = []
    for model_name in MODELS:
        for gamma in GAMMAS:
            score_vector = np.full(len(features), np.nan, dtype=np.float64)
            for fold in range(5):
                held = folds == fold
                score_vector[held] = fit_predict(model_name, features.loc[~held], features.loc[held], gamma)
                print(f"{model_name}/gamma={gamma:.2f}: fold={fold}", flush=True)
            if not np.isfinite(score_vector).all():
                raise RuntimeError(f"Incomplete OOF predictions: {model_name}, gamma={gamma}")
            column = f"{model_name}_gamma{gamma:.2f}"
            predictions[column] = score_vector
            model_map = {(str(n), int(c)): float(p) for n, c, p in zip(features.file_name, features.component_id, score_vector, strict=True)}
            blend_names = ("weighted_model", "core_plus_weighted_30", "rfblend_plus_weighted_30")
            for blend_name in blend_names:
                if blend_name == "weighted_model":
                    combined = model_map
                elif blend_name == "core_plus_weighted_30":
                    combined = {key: 0.70 * core_prob_map[key] + 0.30 * model_map[key] for key in expected}
                else:
                    combined = {key: 0.49 * core_prob_map[key] + 0.21 * rf_prob_map[key] + 0.30 * model_map[key] for key in expected}
                for threshold in THRESHOLDS:
                    pred = predictions_from_masks(names, masks, combined, threshold)
                    metric, _ = score_validation(dataset, pred, names)
                    g = metric["global"]
                    candidates.append({
                        "model": model_name, "utility_gamma": gamma,
                        "blend": blend_name, "threshold": threshold,
                        "pq": float(g["pq"]), "tp": int(g["tp"]), "fp": int(g["fp"]), "fn": int(g["fn"]),
                    })
            print(f"{column}: screened", flush=True)

    candidates.sort(key=lambda row: row["pq"], reverse=True)
    predictions.to_csv(output_dir / "oof-model-probabilities.csv", index=False)
    pd.DataFrame(candidates).to_csv(output_dir / "candidate-sweep.csv", index=False)
    best = candidates[0]
    best_column = f"{best['model']}_gamma{best['utility_gamma']:.2f}"
    best_model_map = {
        (str(n), int(c)): float(p)
        for n, c, p in zip(features.file_name, features.component_id, predictions[best_column], strict=True)
    }
    if best["blend"] == "weighted_model":
        best_map = best_model_map
    elif best["blend"] == "core_plus_weighted_30":
        best_map = {key: 0.70 * core_prob_map[key] + 0.30 * best_model_map[key] for key in expected}
    else:
        best_map = {key: 0.49 * core_prob_map[key] + 0.21 * rf_prob_map[key] + 0.30 * best_model_map[key] for key in expected}
    fold_deltas = []
    for fold in range(5):
        fold_names = sorted(features.loc[folds == fold, "file_name"].astype(str).unique())
        best_predictions = predictions_from_masks(fold_names, masks, best_map, float(best["threshold"]))
        base_predictions = predictions_from_masks(fold_names, masks, {key: 0.70 * core_prob_map[key] + 0.30 * rf_prob_map[key] for key in expected}, 0.45)
        m_best, _ = score_validation(dataset, best_predictions, fold_names)
        m_base, _ = score_validation(dataset, base_predictions, fold_names)
        fold_deltas.append({
            "fold": fold,
            "candidate_pq": float(m_best["global"]["pq"]),
            "incumbent_pq": float(m_base["global"]["pq"]),
            "delta": float(m_best["global"]["pq"] - m_base["global"]["pq"]),
        })
    report = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split_sha256": split.sha256,
        "oof_manifest_sha256": selector_metrics["oof_manifest_sha256"],
        "oof_observations": len(names), "component_rows": len(features),
        "holdout_used": False, "test_images_or_labels_used": False,
        "training_labels": "marginal component utility derived only from competition-provided grouped OOF annotations",
        "utility_weighting": "per-component abs(keep_delta_pq)^gamma, clipped at the training-fold 95th percentile and normalized to equal total weight per observation",
        "gammas": list(GAMMAS), "models": list(MODELS), "thresholds": list(THRESHOLDS),
        "incumbent_oof_pq": 0.4247906680196568,
        "best_candidates": candidates[:15], "selected_candidate_fold_deltas": fold_deltas,
        "input_sha256": {
            "selector_metrics": sha256(selector_metrics_path),
            "component_features_targets": sha256(features_path),
            "base_oof_predictions": sha256(BASE_PREDICTIONS),
            "core_oof_probabilities": sha256(core_path),
            "rf_oof_probabilities": sha256(rf_path),
            "oof_manifest": sha256(OOF_MANIFEST),
        },
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "python": platform.python_version()},
        "runtime_seconds": time.perf_counter() - started,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"best_candidates": candidates[:10], "fold_deltas": fold_deltas, "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
