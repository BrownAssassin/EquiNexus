# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Screen CatBoost pairwise rankers on the frozen grouped OOF selector data."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import catboost
import numpy as np
import pandas as pd
import sklearn
from scipy.special import expit
from catboost import CatBoostRanker

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_component_selector_oof import FEATURES, load_prediction_rows, predictions_from_masks  # noqa: E402
from filament_baseline.data import load_dataset  # noqa: E402
from filament_baseline.scorer import score_validation  # noqa: E402
from filament_baseline.split import create_observation_split  # noqa: E402

FROZEN_SPLIT_SHA256 = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"
SELECTOR_DIR = REPO_ROOT / "artifacts/model-experiments/component-selector-official-pq-20261109-rle-fixed"
MODEL_SCREEN = REPO_ROOT / "artifacts/model-experiments/component-selector-model-screen-20261109"
RF_SCREEN = REPO_ROOT / "artifacts/model-experiments/component-selector-random-forest-oof-20261109"
OOF_PREDICTIONS = REPO_ROOT / "artifacts/model-experiments/oof-fusion-weight25-th035-a200/best-oof-predictions.csv"
OOF_MANIFEST = REPO_ROOT / "data/splits/observation_oof_seed42_5fold.csv"
OUTPUT_DIR = REPO_ROOT / "artifacts/model-experiments/component-selector-catboost-ranker-20261109"

CONFIGS: dict[str, dict[str, Any]] = {
    "pairlogit_depth4_500": {"iterations": 500, "depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 8.0},
    "pairlogit_depth5_400": {"iterations": 400, "depth": 5, "learning_rate": 0.03, "l2_leaf_reg": 10.0},
    "pairlogit_depth6_300": {"iterations": 300, "depth": 6, "learning_rate": 0.025, "l2_leaf_reg": 12.0},
}
THRESHOLDS = tuple(float(value) for value in np.arange(0.35, 0.651, 0.025))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relevance_by_observation(rows: pd.DataFrame) -> np.ndarray:
    relevance = np.zeros(len(rows), dtype=np.int32)
    for _, indices in rows.groupby("file_name", sort=False).indices.items():
        values = rows.iloc[indices].keep_delta_pq.to_numpy(dtype=np.float64)
        pct = pd.Series(values).rank(method="average", pct=True).to_numpy()
        relevance[indices] = np.minimum(4, np.floor(pct * 5).astype(np.int32))
    return relevance


def ordered_groups(rows: pd.DataFrame, relevance: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    order = np.lexsort((rows.component_id.to_numpy(int), rows.file_name.astype(str).to_numpy()))
    ordered = rows.iloc[order].reset_index(drop=True)
    labels = relevance[order]
    groups = ordered.groupby("file_name", sort=False).size().to_numpy(dtype=np.uint32)
    return ordered, labels, groups


def fit_ranker(config: dict[str, Any], train: pd.DataFrame, evaluate: pd.DataFrame) -> np.ndarray:
    relevance = relevance_by_observation(train)
    ordered, labels, group_sizes = ordered_groups(train, relevance)
    group_ids = ordered.file_name.astype(str).to_numpy()
    model = CatBoostRanker(
        loss_function="PairLogitPairwise",
        eval_metric="NDCG:top=10",
        random_seed=42,
        thread_count=4,
        random_strength=0.25,
        verbose=False,
        allow_writing_files=False,
        **config,
    )
    model.fit(
        ordered.loc[:, FEATURES].to_numpy(dtype=np.float32),
        labels,
        group_id=group_ids,
        verbose=False,
    )
    evaluate_order = np.lexsort((evaluate.component_id.to_numpy(int), evaluate.file_name.astype(str).to_numpy()))
    ordered_eval = evaluate.iloc[evaluate_order]
    raw = model.predict(ordered_eval.loc[:, FEATURES].to_numpy(dtype=np.float32))
    inverse = np.empty_like(evaluate_order)
    inverse[evaluate_order] = np.arange(len(evaluate_order))
    return expit(np.asarray(raw[inverse], dtype=np.float64))


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    data_root, output_dir = args.data_root.resolve(), args.output_dir.resolve()
    if not data_root.is_relative_to((REPO_ROOT / "data").resolve()):
        raise ValueError("Competition data must stay under data/")
    if not output_dir.is_relative_to((REPO_ROOT / "artifacts/model-experiments").resolve()):
        raise ValueError("Experiment output must stay under artifacts/model-experiments/")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)

    started = time.perf_counter()
    dataset = load_dataset(data_root)
    split = create_observation_split(dataset, REPO_ROOT / "data/splits")
    if split.sha256 != FROZEN_SPLIT_SHA256:
        raise RuntimeError(f"Frozen split changed: {split.sha256}")
    selector_metrics_path = SELECTOR_DIR / "metrics.json"
    selector_metrics = json.loads(selector_metrics_path.read_text(encoding="utf-8"))
    if selector_metrics.get("split_sha256") != split.sha256:
        raise RuntimeError("Component feature source uses another holdout split")
    names = sorted(split.training_names)
    features_path = SELECTOR_DIR / "oof-component-features-targets.csv"
    features = pd.read_csv(features_path).sort_values(["file_name", "component_id"]).reset_index(drop=True)
    if set(features.file_name.astype(str)) - set(names):
        raise RuntimeError("OOF component features include observations outside training partition")
    manifest = pd.read_csv(OOF_MANIFEST)
    fold_by_name = dict(zip(manifest.file_name.astype(str), manifest.oof_fold.astype(int), strict=True))
    if set(fold_by_name) != set(names):
        raise RuntimeError("Frozen grouped OOF manifest is incomplete")
    for file_name, group in features.groupby("file_name", sort=False):
        if group.oof_fold.nunique() != 1 or int(group.oof_fold.iloc[0]) != fold_by_name[str(file_name)]:
            raise RuntimeError(f"Component rows cross observation folds: {file_name}")
    folds = features.oof_fold.to_numpy(dtype=int)
    core_metrics = json.loads((MODEL_SCREEN / "metrics.json").read_text(encoding="utf-8"))
    rf_metrics = json.loads((RF_SCREEN / "metrics.json").read_text(encoding="utf-8"))
    for label, metrics in (("core", core_metrics), ("random forest", rf_metrics)):
        if metrics.get("split_sha256") != split.sha256 or metrics.get("holdout_used") is not False:
            raise RuntimeError(f"{label} OOF probabilities do not certify holdout exclusion")
        if metrics.get("test_images_or_labels_used") is not False:
            raise RuntimeError(f"{label} OOF probabilities report test data use")

    keys = ["file_name", "component_id"]
    core_rows, rf_rows = pd.read_csv(MODEL_SCREEN / "oof-model-probabilities.csv"), pd.read_csv(RF_SCREEN / "oof-model-probabilities.csv")
    expected = set(zip(features.file_name.astype(str), features.component_id.astype(int), strict=True))
    for table, label in ((core_rows, "core"), (rf_rows, "RF")):
        if set(zip(table.file_name.astype(str), table.component_id.astype(int), strict=True)) != expected:
            raise RuntimeError(f"{label} probabilities do not match OOF component rows")
    index = pd.MultiIndex.from_frame(features[keys])
    core = core_rows.set_index(keys).loc[index]
    forest = rf_rows.set_index(keys).loc[index]
    core_prob = 0.70 * core["logistic_c1.0"].to_numpy(float) + 0.30 * core["extra_trees"].to_numpy(float)
    rf_prob = forest["rf_depth8_leaf10"].to_numpy(float)
    core_map = {(str(n), int(c)): float(p) for n, c, p in zip(features.file_name, features.component_id, core_prob, strict=True)}
    rf_map = {(str(n), int(c)): float(p) for n, c, p in zip(features.file_name, features.component_id, rf_prob, strict=True)}
    incumbent_map = {key: 0.70 * core_map[key] + 0.30 * rf_map[key] for key in expected}
    masks_path = OOF_PREDICTIONS
    masks = load_prediction_rows(masks_path, {Path(name).stem for name in names})

    output_dir.mkdir(parents=True, exist_ok=True)
    all_predictions = pd.DataFrame({"file_name": features.file_name, "component_id": features.component_id})
    sweep: list[dict[str, Any]] = []
    config_metadata: dict[str, Any] = {}
    for config_name, config in CONFIGS.items():
        scores = np.full(len(features), np.nan, dtype=np.float64)
        for fold in range(5):
            held = folds == fold
            scores[held] = fit_ranker(config, features.loc[~held], features.loc[held])
            print(f"{config_name}: fold={fold} train={int((~held).sum())} held={int(held.sum())}", flush=True)
        if not np.isfinite(scores).all():
            raise RuntimeError(f"Incomplete cross-fitted scores for {config_name}")
        all_predictions[f"{config_name}_probability"] = scores
        rank_map = {(str(n), int(c)): float(p) for n, c, p in zip(features.file_name, features.component_id, scores, strict=True)}
        blends = {
            "ranker_only": rank_map,
            "core_ranker30": {key: 0.70 * core_map[key] + 0.30 * rank_map[key] for key in expected},
            "incumbent_ranker30": {key: 0.49 * core_map[key] + 0.21 * rf_map[key] + 0.30 * rank_map[key] for key in expected},
        }
        config_metadata[config_name] = config
        for blend_name, probability_map in blends.items():
            for threshold in THRESHOLDS:
                predictions = predictions_from_masks(names, masks, probability_map, threshold)
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
        print(f"{config_name}: best PQ={max(row['pq'] for row in sweep if row['model']==config_name):.6f}", flush=True)

    sweep.sort(key=lambda row: row["pq"], reverse=True)
    all_predictions.to_csv(output_dir / "oof-ranker-probabilities.csv", index=False)
    pd.DataFrame(sweep).to_csv(output_dir / "candidate-sweep.csv", index=False)
    best = sweep[0]
    score_vector = all_predictions[f"{best['model']}_probability"].to_numpy(float)
    best_rank_map = {(str(n), int(c)): float(p) for n, c, p in zip(features.file_name, features.component_id, score_vector, strict=True)}
    if best["blend"] == "ranker_only":
        best_map = best_rank_map
    elif best["blend"] == "core_ranker30":
        best_map = {key: 0.70 * core_map[key] + 0.30 * best_rank_map[key] for key in expected}
    else:
        best_map = {key: 0.49 * core_map[key] + 0.21 * rf_map[key] + 0.30 * best_rank_map[key] for key in expected}

    fold_rows: list[dict[str, Any]] = []
    candidate_rows: list[Any] = []
    incumbent_rows: list[Any] = []
    for fold in range(5):
        fold_names = sorted(name for name, fold_id in fold_by_name.items() if fold_id == fold)
        candidate_predictions = predictions_from_masks(fold_names, masks, best_map, float(best["threshold"]))
        incumbent_predictions = predictions_from_masks(fold_names, masks, incumbent_map, 0.45)
        candidate_metrics, candidate_counts = score_validation(dataset, candidate_predictions, fold_names)
        incumbent_metrics, incumbent_counts = score_validation(dataset, incumbent_predictions, fold_names)
        candidate_rows.extend(candidate_counts)
        incumbent_rows.extend(incumbent_counts)
        fold_rows.append({
            "fold": fold,
            "candidate_pq": float(candidate_metrics["global"]["pq"]),
            "incumbent_pq": float(incumbent_metrics["global"]["pq"]),
            "delta": float(candidate_metrics["global"]["pq"] - incumbent_metrics["global"]["pq"]),
        })
    rng = np.random.default_rng(42)
    deltas = np.empty(20_000, dtype=float)
    def pq_for(rows: list[Any], indices: np.ndarray) -> float:
        sampled = [rows[int(i)] for i in indices]
        iou_sum = sum(row.tp_iou_sum for row in sampled)
        tp, fp, fn = sum(row.tp for row in sampled), sum(row.fp for row in sampled), sum(row.fn for row in sampled)
        denominator = tp + 0.5 * fp + 0.5 * fn
        return float(iou_sum / denominator) if denominator else 0.0
    def aggregate(rows: list[Any]) -> float:
        iou_sum = sum(row.tp_iou_sum for row in rows)
        tp, fp, fn = sum(row.tp for row in rows), sum(row.fp for row in rows), sum(row.fn for row in rows)
        denominator = tp + 0.5 * fp + 0.5 * fn
        return float(iou_sum / denominator) if denominator else 0.0
    for draw in range(len(deltas)):
        indices = rng.integers(0, len(candidate_rows), size=len(candidate_rows))
        deltas[draw] = pq_for(candidate_rows, indices) - pq_for(incumbent_rows, indices)
    if abs(aggregate(candidate_rows) - float(best["pq"])) > 1e-12:
        raise RuntimeError("Fold-wise scorer counts do not aggregate to the selected OOF score")
    report = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split_sha256": split.sha256,
        "oof_manifest_sha256": sha256(OOF_MANIFEST),
        "oof_observations": len(names),
        "component_rows": len(features),
        "holdout_used": False,
        "test_images_or_labels_used": False,
        "training_labels": "within-observation ranks of marginal component PQ contribution from competition-provided grouped OOF annotations",
        "incumbent_oof_pq": 0.4247906680196568,
        "configs": config_metadata,
        "thresholds": list(THRESHOLDS),
        "top_candidates": sweep[:20],
        "selected_candidate_fold_deltas": fold_rows,
        "paired_observation_bootstrap_vs_incumbent": {
            "resamples": len(deltas),
            "seed": 42,
            "point_delta": float(aggregate(candidate_rows) - aggregate(incumbent_rows)),
            "ci_95": [float(value) for value in np.quantile(deltas, [0.025, 0.975])],
            "probability_candidate_better": float(np.mean(deltas > 0)),
        },
        "input_sha256": {
            "selector_metrics": sha256(selector_metrics_path),
            "component_features_targets": sha256(features_path),
            "core_oof_probabilities": sha256(MODEL_SCREEN / "oof-model-probabilities.csv"),
            "rf_oof_probabilities": sha256(RF_SCREEN / "oof-model-probabilities.csv"),
            "base_oof_predictions": sha256(masks_path),
            "oof_manifest": sha256(OOF_MANIFEST),
        },
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "catboost": catboost.__version__},
        "runtime_seconds": time.perf_counter() - started,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"top_candidates": sweep[:10], "fold_deltas": fold_rows, "bootstrap": report["paired_observation_bootstrap_vs_incumbent"], "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
