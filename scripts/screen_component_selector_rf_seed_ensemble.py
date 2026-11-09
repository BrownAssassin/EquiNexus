# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Screen seed averaging for the grouped OOF random-forest component selector."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier

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
from filament_baseline.data import load_dataset  # noqa: E402
from filament_baseline.scorer import score_validation  # noqa: E402
from filament_baseline.split import create_observation_split  # noqa: E402


FROZEN_SPLIT_SHA256 = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"
DEFAULT_SELECTOR_RUN = REPO_ROOT / "artifacts/model-experiments/component-selector-official-pq-20261109-rle-fixed"
DEFAULT_MODEL_SCREEN = REPO_ROOT / "artifacts/model-experiments/component-selector-model-screen-20261109"
DEFAULT_RF_SCREEN = REPO_ROOT / "artifacts/model-experiments/component-selector-random-forest-oof-20261109"
DEFAULT_RF_BLEND = REPO_ROOT / "artifacts/model-experiments/component-selector-rf-blend-oof-repro-20261109"
DEFAULT_BASE_PREDICTIONS = REPO_ROOT / "artifacts/model-experiments/oof-fusion-weight25-th035-a200/best-oof-predictions.csv"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/model-experiments/component-selector-rf-seed-ensemble-20261109"
SEEDS = (21, 42, 63, 84, 105)
RF_WEIGHT_GRID = (0.20, 0.30, 0.40)
THRESHOLD_GRID = (0.40, 0.425, 0.45, 0.475, 0.50)
BOOTSTRAP_RESAMPLES = 20_000


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_forest(train: pd.DataFrame, evaluate: pd.DataFrame, seed: int) -> np.ndarray:
    group_sizes = train.groupby("file_name").component_id.transform("count").to_numpy(np.float64)
    sample_weight = 1.0 / np.maximum(group_sizes, 1.0)
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=10,
        max_features=0.8,
        class_weight=None,
        n_jobs=4,
        random_state=seed,
    )
    model.fit(
        train.loc[:, FEATURES].to_numpy(dtype=np.float32),
        train.target_keep.to_numpy(dtype=np.int8),
        sample_weight=sample_weight,
    )
    return model.predict_proba(evaluate.loc[:, FEATURES].to_numpy(dtype=np.float32))[:, 1]


def score_probabilities(
    dataset: Any,
    names: list[str],
    masks: dict[str, Any],
    features: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[dict[str, Any], list[Any]]:
    keyed = {
        (str(name), int(component_id)): float(value)
        for name, component_id, value in zip(features.file_name, features.component_id, probabilities)
    }
    predictions = predictions_from_masks(names, masks, keyed, threshold)
    return score_validation(dataset, predictions, names)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--selector-run", type=Path, default=DEFAULT_SELECTOR_RUN)
    parser.add_argument("--model-screen", type=Path, default=DEFAULT_MODEL_SCREEN)
    parser.add_argument("--rf-screen", type=Path, default=DEFAULT_RF_SCREEN)
    parser.add_argument("--rf-blend", type=Path, default=DEFAULT_RF_BLEND)
    parser.add_argument("--base-predictions", type=Path, default=DEFAULT_BASE_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    paths = {
        "data_root": args.data_root.resolve(),
        "selector_run": args.selector_run.resolve(),
        "model_screen": args.model_screen.resolve(),
        "rf_screen": args.rf_screen.resolve(),
        "rf_blend": args.rf_blend.resolve(),
        "base_predictions": args.base_predictions.resolve(),
        "output_dir": args.output_dir.resolve(),
    }
    if not paths["data_root"].is_relative_to((REPO_ROOT / "data").resolve()):
        raise ValueError("Competition data must remain under data/")
    for label in ("selector_run", "model_screen", "rf_screen", "rf_blend", "base_predictions"):
        if not paths[label].is_relative_to((REPO_ROOT / "artifacts").resolve()):
            raise ValueError(f"{label} must remain under artifacts/")
    if not paths["output_dir"].is_relative_to((REPO_ROOT / "artifacts/model-experiments").resolve()):
        raise ValueError("Outputs must remain under artifacts/model-experiments/")
    if paths["output_dir"].exists() and any(paths["output_dir"].iterdir()):
        raise FileExistsError(paths["output_dir"])

    started = time.perf_counter()
    dataset = load_dataset(paths["data_root"])
    split = create_observation_split(dataset, REPO_ROOT / "data/splits")
    if split.sha256 != FROZEN_SPLIT_SHA256:
        raise RuntimeError("Frozen observation split digest changed")
    selector_metrics = json.loads((paths["selector_run"] / "metrics.json").read_text(encoding="utf-8"))
    model_metrics = json.loads((paths["model_screen"] / "metrics.json").read_text(encoding="utf-8"))
    rf_metrics = json.loads((paths["rf_screen"] / "metrics.json").read_text(encoding="utf-8"))
    blend_metrics = json.loads((paths["rf_blend"] / "metrics.json").read_text(encoding="utf-8"))
    for metrics, label in (
        (selector_metrics, "selector"), (model_metrics, "classifier"),
        (rf_metrics, "forest"), (blend_metrics, "RF blend"),
    ):
        if metrics.get("split_sha256") != split.sha256:
            raise RuntimeError(f"{label} artifact uses a different observation split")
    if model_metrics.get("holdout_used") is not False or rf_metrics.get("holdout_used") is not False:
        raise RuntimeError("Cross-fitted probability sources are not certified holdout-free")
    if any(metrics.get("test_images_or_labels_used") is True for metrics in (model_metrics, rf_metrics)):
        raise RuntimeError("A selector probability source reports test-image/label use")

    selector_features_path = paths["selector_run"] / "oof-component-features-targets.csv"
    model_probabilities_path = paths["model_screen"] / "oof-model-probabilities.csv"
    rf_probabilities_path = paths["rf_screen"] / "oof-model-probabilities.csv"
    features = pd.read_csv(selector_features_path)
    model_probabilities = pd.read_csv(model_probabilities_path)
    rf_probabilities = pd.read_csv(rf_probabilities_path)
    keys = ["file_name", "component_id"]
    feature_keys = pd.MultiIndex.from_frame(features[keys])
    for rows, label in ((model_probabilities, "classifier"), (rf_probabilities, "forest")):
        if set(zip(rows.file_name.astype(str), rows.component_id.astype(int))) != set(
            zip(features.file_name.astype(str), features.component_id.astype(int))
        ):
            raise RuntimeError(f"{label} probabilities do not align with OOF feature rows")
    if set(features.file_name.astype(str)) - set(split.training_names):
        raise RuntimeError("OOF component features cross into the fixed holdout")
    if set(features.oof_fold.astype(int)) != set(range(5)):
        raise RuntimeError("Expected all five grouped OOF folds")

    model_index = model_probabilities.set_index(keys).loc[feature_keys]
    rf_index = rf_probabilities.set_index(keys).loc[feature_keys]
    core = (
        0.70 * model_index["logistic_c1.0"].to_numpy(dtype=np.float64)
        + 0.30 * model_index["extra_trees"].to_numpy(dtype=np.float64)
    )
    incumbent_rf = rf_index["rf_depth8_leaf10"].to_numpy(dtype=np.float64)
    fold_ids = features.oof_fold.to_numpy(dtype=np.int32)
    seeds = np.full((len(features), len(SEEDS)), np.nan, dtype=np.float64)
    for seed_index, seed in enumerate(SEEDS):
        for fold in range(5):
            held = fold_ids == fold
            seeds[held, seed_index] = fit_forest(
                features.loc[~held], features.loc[held], seed
            )
            print(
                f"seed={seed} fold={fold} train={int((~held).sum())} held={int(held.sum())}",
                flush=True,
            )
    if not np.isfinite(seeds).all():
        raise RuntimeError("Some seed/fold probability predictions are missing")
    ensemble_rf = seeds.mean(axis=1)

    names = sorted(split.training_names)
    masks = load_prediction_rows(paths["base_predictions"], {Path(name).stem for name in names})
    if not any(masks.get(Path(name).stem) for name in names):
        raise RuntimeError("Cross-fitted base predictions are empty")
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    probabilities_path = paths["output_dir"] / "oof-seed-probabilities.csv"
    probability_rows = features.loc[:, keys].copy()
    for index, seed in enumerate(SEEDS):
        probability_rows[f"rf_seed_{seed}"] = seeds[:, index]
    probability_rows["rf_seed_mean"] = ensemble_rf
    probability_rows["incumbent_rf_seed42"] = incumbent_rf
    probability_rows.to_csv(probabilities_path, index=False)

    sweep: list[dict[str, Any]] = []
    for weight in RF_WEIGHT_GRID:
        blended = (1.0 - weight) * core + weight * ensemble_rf
        for threshold in THRESHOLD_GRID:
            metrics, _ = score_probabilities(dataset, names, masks, features, blended, threshold)
            sweep.append({
                "rf_weight": weight,
                "threshold": threshold,
                "pq": float(metrics["global"]["pq"]),
                "tp": int(metrics["global"]["tp"]),
                "fp": int(metrics["global"]["fp"]),
                "fn": int(metrics["global"]["fn"]),
            })
    sweep.sort(key=lambda row: row["pq"], reverse=True)
    pd.DataFrame(sweep).to_csv(paths["output_dir"] / "candidate-sweep.csv", index=False)

    incumbent_probability = 0.70 * core + 0.30 * incumbent_rf
    incumbent_metrics, incumbent_rows = score_probabilities(
        dataset, names, masks, features, incumbent_probability, 0.45
    )
    selected = sweep[0]
    selected_probability = (
        (1.0 - float(selected["rf_weight"])) * core
        + float(selected["rf_weight"]) * ensemble_rf
    )
    candidate_metrics, candidate_rows = score_probabilities(
        dataset, names, masks, features, selected_probability, float(selected["threshold"])
    )
    pd.DataFrame(per_observation_rows(incumbent_rows)).to_csv(
        paths["output_dir"] / "incumbent-per-observation.csv", index=False
    )
    pd.DataFrame(per_observation_rows(candidate_rows)).to_csv(
        paths["output_dir"] / "candidate-per-observation.csv", index=False
    )
    rf_blend_baseline = float(blend_metrics["top_candidates"][0]["pq"])
    if abs(float(incumbent_metrics["global"]["pq"]) - rf_blend_baseline) > 1e-9:
        raise RuntimeError("Reconstructed seed-42 incumbent did not reproduce the saved OOF blend")

    try:
        source_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        source_commit = "unknown"
    report = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "source_sha256": file_sha256(Path(__file__).resolve()),
        "split_sha256": split.sha256,
        "oof_manifest_sha256": selector_metrics.get("oof_manifest_sha256"),
        "oof_observation_count": len(names),
        "component_row_count": len(features),
        "holdout_used": False,
        "test_images_or_labels_used": False,
        "training_labels": "competition-provided annotations for grouped OOF training observations only",
        "model": {
            "type": "RandomForestClassifier",
            "n_estimators_per_seed": 400,
            "max_depth": 8,
            "min_samples_leaf": 10,
            "max_features": 0.8,
            "sample_weight": "inverse candidate component count within each observation",
            "seeds": list(SEEDS),
            "aggregation": "arithmetic mean of five cross-fitted probability predictions",
        },
        "incumbent": {
            "description": "0.70 * (0.70 logistic C1 + 0.30 ExtraTrees) + 0.30 RF seed 42 at threshold 0.45",
            "global": incumbent_metrics["global"],
        },
        "candidate_selection": {
            "selection_partition": "five-fold grouped OOF training partition only",
            "rf_weight_grid": list(RF_WEIGHT_GRID),
            "threshold_grid": list(THRESHOLD_GRID),
            "best": selected,
            "global": candidate_metrics["global"],
            "paired_bootstrap_vs_incumbent": paired_bootstrap(
                incumbent_rows, candidate_rows, resamples=BOOTSTRAP_RESAMPLES, seed=42
            ),
        },
        "probability_variability": {
            "mean_seed_probability_std": float(np.std(seeds, axis=1).mean()),
            "p95_seed_probability_std": float(np.quantile(np.std(seeds, axis=1), 0.95)),
        },
        "runtime_seconds": time.perf_counter() - started,
        "hardware": {
            "device": "CPU",
            "processor": platform.processor(),
            "logical_cpu_count": __import__("os").cpu_count(),
            "peak_gpu_memory_bytes": 0,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "input_sha256": {
            "selector_metrics": file_sha256(paths["selector_run"] / "metrics.json"),
            "selector_features_targets": file_sha256(selector_features_path),
            "classifier_probabilities": file_sha256(model_probabilities_path),
            "rf_probabilities": file_sha256(rf_probabilities_path),
            "rf_blend_metrics": file_sha256(paths["rf_blend"] / "metrics.json"),
            "base_oof_predictions": file_sha256(paths["base_predictions"]),
        },
    }
    (paths["output_dir"] / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "incumbent_oof_pq": incumbent_metrics["global"]["pq"],
        "candidate_oof_pq": candidate_metrics["global"]["pq"],
        "best": selected,
        "paired_bootstrap": report["candidate_selection"]["paired_bootstrap_vs_incumbent"],
        "output_dir": str(paths["output_dir"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
