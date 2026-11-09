# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Screen rank-normalized ensembles of grouped OOF component selectors."""

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
from scipy.stats import rankdata

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_component_selector_oof import load_prediction_rows, predictions_from_masks  # noqa: E402
from filament_baseline.data import load_dataset  # noqa: E402
from filament_baseline.scorer import score_validation  # noqa: E402
from filament_baseline.split import create_observation_split  # noqa: E402

FROZEN_SPLIT_SHA256 = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"
SELECTOR_DIR = REPO_ROOT / "artifacts/model-experiments/component-selector-official-pq-20261109-rle-fixed"
OOF_PREDICTIONS = REPO_ROOT / "artifacts/model-experiments/oof-fusion-weight25-th035-a200/best-oof-predictions.csv"
OOF_MANIFEST = REPO_ROOT / "data/splits/observation_oof_seed42_5fold.csv"
OUTPUT_DIR = REPO_ROOT / "artifacts/model-experiments/component-selector-rank-consensus-20261109"

SOURCES = {
    "core": ("component-selector-model-screen-20261109", "oof-model-probabilities.csv", None),
    "rf": ("component-selector-random-forest-oof-20261109", "oof-model-probabilities.csv", None),
    "lgbm": ("component-selector-lightgbm-20261109-reviewed", "oof-probabilities.csv", None),
    "xgb": ("component-selector-xgboost-20261109", "oof-probabilities.csv", None),
    "utility": ("component-selector-utility-weighted-20261109", "oof-model-probabilities.csv", None),
    "ranker": ("component-selector-pairwise-ranker-20261109", "oof-ranker-probabilities.csv", None),
    "morphology": ("component-selector-morphology-20261109", "oof-model-probabilities.csv", None),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_pq(rows: list[Any], selection: np.ndarray | None = None) -> float:
    if selection is None:
        chosen = rows
    else:
        chosen = [rows[int(index)] for index in selection]
    tp_ious = [value for row in chosen for value in row.tp_ious]
    fp = sum(row.fp for row in chosen)
    fn = sum(row.fn for row in chosen)
    denominator = len(tp_ious) + 0.5 * fp + 0.5 * fn
    return float(sum(tp_ious) / denominator) if denominator else 0.0


def paired_observation_bootstrap(
    candidate_rows: list[Any], baseline_rows: list[Any], *, resamples: int = 20_000, seed: int = 42
) -> dict[str, Any]:
    if len(candidate_rows) != len(baseline_rows):
        raise ValueError("Candidate and baseline observation counts do not align")
    rng = np.random.default_rng(seed)
    n = len(candidate_rows)
    deltas = np.empty(resamples, dtype=np.float64)
    for draw in range(resamples):
        indices = rng.integers(0, n, size=n)
        deltas[draw] = aggregate_pq(candidate_rows, indices) - aggregate_pq(baseline_rows, indices)
    return {
        "resamples": resamples,
        "seed": seed,
        "unit": "original observation; its annotator views stay grouped",
        "point_delta_candidate_minus_baseline": aggregate_pq(candidate_rows) - aggregate_pq(baseline_rows),
        "ci_95": [float(value) for value in np.quantile(deltas, [0.025, 0.975])],
        "probability_candidate_better": float(np.mean(deltas > 0)),
    }


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
        raise RuntimeError("Component feature source uses a different frozen split")
    names = sorted(split.training_names)
    expected_stems = {Path(name).stem for name in names}
    masks = load_prediction_rows(OOF_PREDICTIONS, expected_stems)
    features_path = SELECTOR_DIR / "oof-component-features-targets.csv"
    features = pd.read_csv(features_path).sort_values(["file_name", "component_id"]).reset_index(drop=True)
    if set(features.file_name.astype(str)) - set(names):
        raise RuntimeError("OOF feature rows include observations outside the frozen training partition")
    manifest = pd.read_csv(OOF_MANIFEST)
    fold_by_name = dict(zip(manifest.file_name.astype(str), manifest.oof_fold.astype(int), strict=True))
    if set(fold_by_name) != set(names):
        raise RuntimeError("Frozen OOF manifest is incomplete")
    for name, group in features.groupby("file_name", sort=False):
        if group.oof_fold.nunique() != 1 or int(group.oof_fold.iloc[0]) != fold_by_name[str(name)]:
            raise RuntimeError(f"Component fold differs from frozen observation manifest: {name}")

    keys = ["file_name", "component_id"]
    expected_keys = pd.MultiIndex.from_frame(features[keys])
    probabilities: dict[str, np.ndarray] = {}
    input_hashes: dict[str, str] = {
        "selector_metrics": sha256(selector_metrics_path),
        "selector_features": sha256(features_path),
        "base_oof_predictions": sha256(OOF_PREDICTIONS),
        "oof_manifest": sha256(OOF_MANIFEST),
    }
    for name, (folder, filename, _) in SOURCES.items():
        source_dir = REPO_ROOT / "artifacts/model-experiments" / folder
        metrics_path = source_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("split_sha256") != split.sha256 or metrics.get("test_images_or_labels_used") is not False:
            raise RuntimeError(f"{name} selector metadata reports an unexpected split or test use")
        path = source_dir / filename
        table = pd.read_csv(path)
        table_index = pd.MultiIndex.from_frame(table[keys])
        if not table_index.is_unique or len(table_index) != len(expected_keys) or set(table_index) != set(expected_keys):
            raise RuntimeError(f"{name} predictions do not match the complete OOF component key set")
        table = table.set_index(keys).loc[expected_keys]
        if name == "core":
            probabilities[name] = 0.70 * table["logistic_c1.0"].to_numpy(float) + 0.30 * table["extra_trees"].to_numpy(float)
        elif name == "rf":
            probabilities[name] = table["rf_depth8_leaf10"].to_numpy(float)
        elif name == "lgbm":
            probabilities[name] = table["lgbm_leaf7_min30"].to_numpy(float)
        elif name == "xgb":
            probabilities[name] = table["xgb_depth2"].to_numpy(float)
        elif name == "utility":
            probabilities[name] = table["rf_depth8_leaf10_gamma1.00"].to_numpy(float)
        elif name == "ranker":
            probabilities[name] = table["rank_depth3_300_probability"].to_numpy(float)
        else:
            probabilities[name] = table["morphology_rf"].to_numpy(float)
        if not all(np.isfinite(probabilities[name])):
            raise RuntimeError(f"{name} probabilities contain non-finite values")
        input_hashes[f"{name}_probabilities"] = sha256(path)
        input_hashes[f"{name}_metrics"] = sha256(metrics_path)

    raw = pd.DataFrame(probabilities)
    ranked = pd.DataFrame(index=raw.index)
    for column in raw:
        ranked[column] = raw.groupby(features.file_name, sort=False)[column].rank(method="average", pct=True)

    incumbent = 0.70 * probabilities["core"] + 0.30 * probabilities["rf"]
    all_rank = ranked.mean(axis=1).to_numpy(float)
    independent_rank = ranked[["rf", "lgbm", "xgb", "utility", "ranker", "morphology"]].mean(axis=1).to_numpy(float)
    strong_three_rank = ranked[["core", "rf", "utility"]].mean(axis=1).to_numpy(float)
    mean_probability = raw.mean(axis=1).to_numpy(float)
    candidates = {
        "rank_all_models": (all_rank, np.arange(0.40, 0.801, 0.025)),
        "rank_independent_models": (independent_rank, np.arange(0.40, 0.801, 0.025)),
        "rank_core_rf_utility": (strong_three_rank, np.arange(0.40, 0.801, 0.025)),
        "incumbent_rank30": (0.70 * incumbent + 0.30 * all_rank, np.arange(0.30, 0.651, 0.025)),
        "incumbent_rank50": (0.50 * incumbent + 0.50 * all_rank, np.arange(0.30, 0.651, 0.025)),
        "mean_probability": (mean_probability, np.arange(0.30, 0.651, 0.025)),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    score_frame = pd.DataFrame({"file_name": features.file_name, "component_id": features.component_id})
    for name, (scores, _) in candidates.items():
        score_frame[name] = scores
    score_frame.to_csv(output_dir / "oof-consensus-probabilities.csv", index=False)

    sweep: list[dict[str, Any]] = []
    score_maps: dict[str, dict[tuple[str, int], float]] = {}
    for name, (scores, thresholds) in candidates.items():
        score_maps[name] = {
            (str(file_name), int(component_id)): float(score)
            for file_name, component_id, score in zip(features.file_name, features.component_id, scores, strict=True)
        }
        for threshold in thresholds:
            predictions = predictions_from_masks(names, masks, score_maps[name], float(threshold))
            metrics, _ = score_validation(dataset, predictions, names)
            global_metrics = metrics["global"]
            sweep.append({
                "candidate": name,
                "threshold": float(threshold),
                "pq": float(global_metrics["pq"]),
                "tp": int(global_metrics["tp"]),
                "fp": int(global_metrics["fp"]),
                "fn": int(global_metrics["fn"]),
            })
        print(f"{name}: best PQ={max(row['pq'] for row in sweep if row['candidate'] == name):.6f}", flush=True)

    sweep.sort(key=lambda row: row["pq"], reverse=True)
    pd.DataFrame(sweep).to_csv(output_dir / "candidate-sweep.csv", index=False)
    best = sweep[0]
    baseline_map = {
        (str(file_name), int(component_id)): float(score)
        for file_name, component_id, score in zip(features.file_name, features.component_id, incumbent, strict=True)
    }
    fold_rows: list[dict[str, Any]] = []
    candidate_counts: list[Any] = []
    baseline_counts: list[Any] = []
    for fold in range(5):
        fold_names = sorted(name for name, fold_id in fold_by_name.items() if fold_id == fold)
        candidate_predictions = predictions_from_masks(fold_names, masks, score_maps[best["candidate"]], float(best["threshold"]))
        baseline_predictions = predictions_from_masks(fold_names, masks, baseline_map, 0.45)
        candidate_metrics, candidate_rows = score_validation(dataset, candidate_predictions, fold_names)
        baseline_metrics, baseline_rows = score_validation(dataset, baseline_predictions, fold_names)
        candidate_counts.extend(candidate_rows)
        baseline_counts.extend(baseline_rows)
        fold_rows.append({
            "fold": fold,
            "candidate_pq": float(candidate_metrics["global"]["pq"]),
            "incumbent_pq": float(baseline_metrics["global"]["pq"]),
            "delta": float(candidate_metrics["global"]["pq"] - baseline_metrics["global"]["pq"]),
        })
    paired_bootstrap = paired_observation_bootstrap(candidate_counts, baseline_counts)
    report = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split_sha256": split.sha256,
        "oof_manifest_sha256": sha256(OOF_MANIFEST),
        "oof_observations": len(names),
        "component_rows": len(features),
        "holdout_used": False,
        "test_images_or_labels_used": False,
        "training_labels": "competition-provided grouped OOF annotations used only to train the source selectors; rank consensus uses their cross-fitted OOF probabilities",
        "incumbent_oof_pq": aggregate_pq(baseline_counts),
        "selected_candidate": best,
        "selected_candidate_fold_deltas": fold_rows,
        "paired_observation_bootstrap_vs_incumbent": paired_bootstrap,
        "candidate_count": len(sweep),
        "top_candidates": sweep[:20],
        "probability_sources": {name: {"folder": spec[0], "file": spec[1]} for name, spec in SOURCES.items()},
        "input_sha256": input_hashes,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "runtime_seconds": time.perf_counter() - started,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"selected_candidate": best, "fold_deltas": fold_rows, "paired_bootstrap": paired_bootstrap, "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
