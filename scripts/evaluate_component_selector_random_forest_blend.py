# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Evaluate the single OOF-selected random-forest blend on the frozen holdout."""

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

from evaluate_component_selector_probability_blend_holdout import (  # noqa: E402
    file_sha256,
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
from filament_baseline.runner import _write_predictions  # noqa: E402
from filament_baseline.scorer import score_validation  # noqa: E402
from filament_baseline.split import create_observation_split  # noqa: E402


FROZEN_SPLIT_SHA256 = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"
SELECTOR_RUN = REPO_ROOT / "artifacts/model-experiments/component-selector-official-pq-20261109-rle-fixed"
MODEL_SCREEN = REPO_ROOT / "artifacts/model-experiments/component-selector-model-screen-20261109"
RF_SCREEN = REPO_ROOT / "artifacts/model-experiments/component-selector-random-forest-oof-20261109"
RF_BLEND_SCREEN = REPO_ROOT / "artifacts/model-experiments/component-selector-rf-blend-oof-repro-20261109"
OUTPUT_DIR = REPO_ROOT / "artifacts/model-experiments/component-selector-rf-blend-holdout-20261109-final"
RF_WEIGHT = 0.30
CORE_EXTRA_TREES_WEIGHT = 0.30
THRESHOLD = 0.45


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    data_root, output_dir = args.data_root.resolve(), args.output_dir.resolve()
    if not data_root.is_relative_to((REPO_ROOT / "data").resolve()):
        raise ValueError("Competition data must remain under data/")
    if not output_dir.is_relative_to((REPO_ROOT / "artifacts/model-experiments").resolve()):
        raise ValueError("Outputs must remain under artifacts/model-experiments/")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)

    started = time.perf_counter()
    dataset = load_dataset(data_root)
    split = create_observation_split(dataset, REPO_ROOT / "data/splits")
    if split.sha256 != FROZEN_SPLIT_SHA256:
        raise RuntimeError("Frozen grouped split digest changed")
    selector_metrics = json.loads((SELECTOR_RUN / "metrics.json").read_text(encoding="utf-8"))
    oof_blend_metrics = json.loads((RF_BLEND_SCREEN / "metrics.json").read_text(encoding="utf-8"))
    if selector_metrics.get("split_sha256") != split.sha256:
        raise RuntimeError("Selector feature source uses a different split")
    if oof_blend_metrics.get("split_sha256") != split.sha256 or oof_blend_metrics.get("holdout_used") is not False:
        raise RuntimeError("Random-forest blend was not screened on holdout-free OOF data")
    selected = oof_blend_metrics["top_candidates"][0]
    if (abs(float(selected["rf_weight"]) - RF_WEIGHT) > 1e-8
            or abs(float(selected["threshold"]) - THRESHOLD) > 1e-8):
        raise RuntimeError("Holdout candidate differs from the OOF-selected blend")

    train = pd.read_csv(SELECTOR_RUN / "oof-component-features-targets.csv")
    holdout = pd.read_csv(SELECTOR_RUN / "holdout-component-features.csv")
    if set(train.file_name.astype(str)) - set(split.training_names):
        raise RuntimeError("Selector training table crosses into the fixed holdout")
    if set(holdout.file_name.astype(str)) - set(split.validation_names):
        raise RuntimeError("Holdout feature table contains non-holdout observations")
    if set(train.oof_fold.astype(int)) != set(range(5)):
        raise RuntimeError("Expected all five grouped OOF folds")

    logistic = _fit_model("logistic_c1.0", train, holdout)
    extra_trees = _fit_model("extra_trees", train, holdout)
    component_counts = train.groupby("file_name").component_id.transform("count").to_numpy(float)
    weights = 1.0 / np.maximum(component_counts, 1.0)
    rf_model = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=10,
        max_features=0.8,
        class_weight=None,
        n_jobs=4,
        random_state=42,
    )
    rf_model.fit(
        train.loc[:, FEATURES].to_numpy(dtype=np.float32),
        train.target_keep.to_numpy(dtype=np.int8),
        sample_weight=weights,
    )
    rf_probability = rf_model.predict_proba(holdout.loc[:, FEATURES].to_numpy(dtype=np.float32))[:, 1]
    core_probability = (1.0 - CORE_EXTRA_TREES_WEIGHT) * logistic + CORE_EXTRA_TREES_WEIGHT * extra_trees
    candidate_probability = (1.0 - RF_WEIGHT) * core_probability + RF_WEIGHT * rf_probability
    keys = list(zip(holdout.file_name.astype(str), holdout.component_id.astype(int)))
    core_keyed = {key: float(value) for key, value in zip(keys, core_probability)}
    candidate_keyed = {key: float(value) for key, value in zip(keys, candidate_probability)}
    names = sorted(split.validation_names)
    stems = {Path(name).stem for name in names}
    base_predictions = load_prediction_rows(SELECTOR_RUN / "baseline-holdout-predictions.csv", stems)
    core_predictions = predictions_from_masks(names, base_predictions, core_keyed, THRESHOLD)
    candidate_predictions = predictions_from_masks(names, base_predictions, candidate_keyed, THRESHOLD)
    core_metrics, core_rows = score_validation(dataset, core_predictions, names)
    candidate_metrics, candidate_rows = score_validation(dataset, candidate_predictions, names)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_predictions(output_dir / "baseline-predictions.csv", core_predictions)
    _write_predictions(output_dir / "candidate-predictions.csv", candidate_predictions)
    pd.DataFrame(per_observation_rows(core_rows)).to_csv(output_dir / "baseline-per-observation.csv", index=False)
    pd.DataFrame(per_observation_rows(candidate_rows)).to_csv(output_dir / "candidate-per-observation.csv", index=False)
    report = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split_sha256": split.sha256,
        "oof_manifest_sha256": selector_metrics.get("oof_manifest_sha256"),
        "holdout_observations": len(names),
        "test_images_or_labels_used": False,
        "training_labels": "competition-provided annotations for the OOF training partition only",
        "holdout_labels": "used for this single OOF-selected candidate comparison",
        "selection_policy": "RF weight 0.30 and threshold 0.45 selected from grouped OOF only; no holdout search",
        "candidate_config": {
            "score": "0.70 * (0.70 logistic C1 + 0.30 ExtraTrees) + 0.30 RandomForest",
            "random_forest": {
                "n_estimators": 400, "max_depth": 8, "min_samples_leaf": 10,
                "max_features": 0.8, "sample_weight": "inverse component count per observation",
            },
            "threshold": THRESHOLD,
        },
        "core_blend": {"global": core_metrics["global"], "metrics": core_metrics},
        "candidate": {"global": candidate_metrics["global"], "metrics": candidate_metrics},
        "paired_bootstrap_vs_core_blend": paired_bootstrap(core_rows, candidate_rows),
        "input_sha256": {
            "selector_features_targets": sha256(SELECTOR_RUN / "oof-component-features-targets.csv"),
            "holdout_features": sha256(SELECTOR_RUN / "holdout-component-features.csv"),
            "holdout_base_predictions": sha256(SELECTOR_RUN / "baseline-holdout-predictions.csv"),
            "rf_oof_screen": sha256(RF_SCREEN / "metrics.json"),
            "rf_blend_oof_screen": sha256(RF_BLEND_SCREEN / "metrics.json"),
            "script": sha256(Path(__file__).resolve()),
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "core_holdout_pq": core_metrics["global"]["pq"],
        "candidate_holdout_pq": candidate_metrics["global"]["pq"],
        "paired_bootstrap": report["paired_bootstrap_vs_core_blend"],
        "output_dir": str(output_dir),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
