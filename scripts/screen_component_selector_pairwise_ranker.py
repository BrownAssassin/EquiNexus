# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Screen grouped pairwise component ranking with cross-fitted OOF scores."""

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
import xgboost
from scipy.special import expit
from xgboost import XGBRanker

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
OUTPUT_DIR = REPO_ROOT / "artifacts/model-experiments/component-selector-pairwise-ranker-20261109"
CONFIGS: dict[str, dict[str, Any]] = {
    "rank_depth2_400": {"n_estimators": 400, "max_depth": 2, "learning_rate": 0.03, "min_child_weight": 20, "reg_lambda": 10.0},
    "rank_depth3_300": {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.025, "min_child_weight": 30, "reg_lambda": 12.0},
    "rank_depth2_700": {"n_estimators": 700, "max_depth": 2, "learning_rate": 0.02, "min_child_weight": 25, "reg_lambda": 15.0},
}
BLENDS = ("rank_only", "core_rank30", "incumbent_rank30")
THRESHOLDS = tuple(float(value) for value in np.arange(0.35, 0.651, 0.025))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def grouped_relevance(rows: pd.DataFrame) -> np.ndarray:
    relevance = np.zeros(len(rows), dtype=np.int32)
    for _, indices in rows.groupby("file_name", sort=False).indices.items():
        if len(indices) < 2:
            continue
        values = rows.iloc[indices].keep_delta_pq.to_numpy(dtype=np.float64)
        percentile = pd.Series(values).rank(method="average", pct=True).to_numpy()
        relevance[indices] = np.minimum(4, np.floor(percentile * 5).astype(np.int32))
    return relevance


def sorted_group_data(frame: pd.DataFrame, relevance: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    order = np.lexsort((frame.component_id.to_numpy(dtype=int), frame.file_name.astype(str).to_numpy()))
    ordered = frame.iloc[order].reset_index(drop=True)
    labels = relevance[order]
    counts = ordered.groupby("file_name", sort=False).size().to_numpy(dtype=np.uint32)
    return ordered, labels, counts


def fit_ranker(config: dict[str, Any], train: pd.DataFrame, evaluate: pd.DataFrame) -> np.ndarray:
    relevance = grouped_relevance(train)
    ordered, labels, group_sizes = sorted_group_data(train, relevance)
    model = XGBRanker(
        objective="rank:pairwise",
        tree_method="hist",
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        n_jobs=4,
        random_state=42,
        **config,
    )
    model.fit(
        ordered.loc[:, FEATURES].to_numpy(dtype=np.float32),
        labels,
        group=group_sizes,
        verbose=False,
    )
    evaluate_order = np.lexsort((evaluate.component_id.to_numpy(dtype=int), evaluate.file_name.astype(str).to_numpy()))
    ordered_eval = evaluate.iloc[evaluate_order]
    raw_ordered = model.predict(ordered_eval.loc[:, FEATURES].to_numpy(dtype=np.float32))
    inverse = np.empty_like(evaluate_order)
    inverse[evaluate_order] = np.arange(len(evaluate_order))
    return expit(raw_ordered[inverse])


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
        raise RuntimeError("Selector feature file uses another holdout split")
    for path, label in ((MODEL_SCREEN / "metrics.json", "core classifier"), (RF_SCREEN / "metrics.json", "forest classifier")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        if metrics.get("split_sha256") != split.sha256 or metrics.get("holdout_used") is not False:
            raise RuntimeError(f"{label} probabilities are not frozen grouped OOF data")
        if metrics.get("test_images_or_labels_used") is not False:
            raise RuntimeError(f"{label} source reports test use")

    names = sorted(split.training_names)
    features_path = selector_run / "oof-component-features-targets.csv"
    features = pd.read_csv(features_path).sort_values(["file_name", "component_id"]).reset_index(drop=True)
    if set(features.file_name.astype(str)) - set(names):
        raise RuntimeError("Ranker input includes observations outside the OOF training partition")
    manifest = pd.read_csv(OOF_MANIFEST)
    fold_by_name = dict(zip(manifest.file_name.astype(str), manifest.oof_fold.astype(int), strict=True))
    if set(fold_by_name) != set(names) or set(fold_by_name.values()) != set(range(5)):
        raise RuntimeError("Frozen grouped OOF manifest is incomplete")
    folds = features.oof_fold.to_numpy(dtype=int)
    for name, group in features.groupby("file_name", sort=False):
        if int(group.oof_fold.iloc[0]) != fold_by_name[str(name)]:
            raise RuntimeError(f"Component fold differs from observation manifest: {name}")

    core_path = MODEL_SCREEN / "oof-model-probabilities.csv"
    rf_path = RF_SCREEN / "oof-model-probabilities.csv"
    core_rows, rf_rows = pd.read_csv(core_path), pd.read_csv(rf_path)
    keys = ["file_name", "component_id"]
    expected = set(zip(features.file_name.astype(str), features.component_id.astype(int), strict=True))
    for table, label in ((core_rows, "core"), (rf_rows, "forest")):
        if set(zip(table.file_name.astype(str), table.component_id.astype(int), strict=True)) != expected:
            raise RuntimeError(f"{label} probability rows do not match OOF components")
    index = pd.MultiIndex.from_frame(features[keys])
    core = core_rows.set_index(keys).loc[index]
    forest = rf_rows.set_index(keys).loc[index]
    core_prob = 0.70 * core["logistic_c1.0"].to_numpy(dtype=np.float64) + 0.30 * core["extra_trees"].to_numpy(dtype=np.float64)
    rf_prob = forest["rf_depth8_leaf10"].to_numpy(dtype=np.float64)
    core_map = {(str(n), int(c)): float(p) for n, c, p in zip(features.file_name, features.component_id, core_prob, strict=True)}
    rf_map = {(str(n), int(c)): float(p) for n, c, p in zip(features.file_name, features.component_id, rf_prob, strict=True)}
    masks = load_prediction_rows(BASE_PREDICTIONS, {Path(name).stem for name in names})

    output_dir.mkdir(parents=True, exist_ok=True)
    all_predictions = pd.DataFrame({"file_name": features.file_name, "component_id": features.component_id})
    sweep: list[dict[str, Any]] = []
    for config_name, config in CONFIGS.items():
        score_vector = np.full(len(features), np.nan, dtype=np.float64)
        for fold in range(5):
            held = folds == fold
            score_vector[held] = fit_ranker(config, features.loc[~held], features.loc[held])
            print(f"{config_name}: fold={fold} train={int((~held).sum())} held={int(held.sum())}", flush=True)
        if not np.isfinite(score_vector).all():
            raise RuntimeError(f"Incomplete cross-fitted rank scores: {config_name}")
        column = f"{config_name}_probability"
        all_predictions[column] = score_vector
        rank_map = {(str(n), int(c)): float(p) for n, c, p in zip(features.file_name, features.component_id, score_vector, strict=True)}
        blends = {
            "rank_only": rank_map,
            "core_rank30": {key: 0.70 * core_map[key] + 0.30 * rank_map[key] for key in expected},
            "incumbent_rank30": {key: 0.49 * core_map[key] + 0.21 * rf_map[key] + 0.30 * rank_map[key] for key in expected},
        }
        for blend_name, probabilities in blends.items():
            for threshold in THRESHOLDS:
                predictions = predictions_from_masks(names, masks, probabilities, threshold)
                metrics, _ = score_validation(dataset, predictions, names)
                global_metrics = metrics["global"]
                sweep.append({
                    "model": config_name,
                    "blend": blend_name,
                    "threshold": threshold,
                    "pq": float(global_metrics["pq"]),
                    "tp": int(global_metrics["tp"]),
                    "fp": int(global_metrics["fp"]),
                    "fn": int(global_metrics["fn"]),
                })
        print(f"{config_name}: best={max(row['pq'] for row in sweep if row['model']==config_name):.6f}", flush=True)

    sweep.sort(key=lambda row: row["pq"], reverse=True)
    all_predictions.to_csv(output_dir / "oof-ranker-probabilities.csv", index=False)
    pd.DataFrame(sweep).to_csv(output_dir / "candidate-sweep.csv", index=False)
    best = sweep[0]
    best_scores = all_predictions[f"{best['model']}_probability"].to_numpy(dtype=np.float64)
    best_rank_map = {(str(n), int(c)): float(p) for n, c, p in zip(features.file_name, features.component_id, best_scores, strict=True)}
    if best["blend"] == "rank_only":
        best_map = best_rank_map
    elif best["blend"] == "core_rank30":
        best_map = {key: 0.70 * core_map[key] + 0.30 * best_rank_map[key] for key in expected}
    else:
        best_map = {key: 0.49 * core_map[key] + 0.21 * rf_map[key] + 0.30 * best_rank_map[key] for key in expected}
    incumbent_map = {key: 0.70 * core_map[key] + 0.30 * rf_map[key] for key in expected}
    fold_deltas = []
    for fold in range(5):
        fold_names = sorted(name for name, fold_id in fold_by_name.items() if fold_id == fold)
        best_predictions = predictions_from_masks(fold_names, masks, best_map, float(best["threshold"]))
        base_predictions = predictions_from_masks(fold_names, masks, incumbent_map, 0.45)
        best_metrics, _ = score_validation(dataset, best_predictions, fold_names)
        base_metrics, _ = score_validation(dataset, base_predictions, fold_names)
        fold_deltas.append({
            "fold": fold,
            "candidate_pq": float(best_metrics["global"]["pq"]),
            "incumbent_pq": float(base_metrics["global"]["pq"]),
            "delta": float(best_metrics["global"]["pq"] - base_metrics["global"]["pq"]),
        })
    report = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split_sha256": split.sha256,
        "oof_manifest_sha256": selector_metrics["oof_manifest_sha256"],
        "oof_observations": len(names),
        "component_rows": len(features),
        "holdout_used": False,
        "test_images_or_labels_used": False,
        "training_labels": "within-observation ranks of marginal PQ contribution from competition-provided grouped OOF annotations",
        "relevance": "five quantized within-observation relevance levels based on keep_delta_pq ranks",
        "incumbent_oof_pq": 0.4247906680196568,
        "configs": CONFIGS,
        "thresholds": list(THRESHOLDS),
        "top_candidates": sweep[:15],
        "selected_candidate_fold_deltas": fold_deltas,
        "input_sha256": {
            "selector_metrics": sha256(selector_metrics_path),
            "component_features_targets": sha256(features_path),
            "core_oof_probabilities": sha256(core_path),
            "rf_oof_probabilities": sha256(rf_path),
            "base_oof_predictions": sha256(BASE_PREDICTIONS),
            "oof_manifest": sha256(OOF_MANIFEST),
        },
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__},
        "runtime_seconds": time.perf_counter() - started,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"top_candidates": sweep[:10], "fold_deltas": fold_deltas, "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
