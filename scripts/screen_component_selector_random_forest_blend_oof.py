# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Screen OOF blends between the selected probability selector and a forest."""

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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_component_selector_oof import load_prediction_rows, predictions_from_masks  # noqa: E402
from filament_baseline.data import load_dataset  # noqa: E402
from filament_baseline.scorer import score_validation  # noqa: E402
from filament_baseline.split import create_observation_split  # noqa: E402


FROZEN_SPLIT_SHA256 = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"
DEFAULT_SELECTOR_RUN = REPO_ROOT / "artifacts/model-experiments/component-selector-official-pq-20261109-rle-fixed"
DEFAULT_MODEL_SCREEN = REPO_ROOT / "artifacts/model-experiments/component-selector-model-screen-20261109"
DEFAULT_RF_SCREEN = REPO_ROOT / "artifacts/model-experiments/component-selector-random-forest-oof-20261109"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/model-experiments/component-selector-rf-blend-oof-repro-20261109"
RF_WEIGHTS = tuple(float(value) for value in np.arange(0.0, 1.0001, 0.1))
THRESHOLDS = (0.40, 0.425, 0.45, 0.475, 0.50)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--selector-run", type=Path, default=DEFAULT_SELECTOR_RUN)
    parser.add_argument("--model-screen", type=Path, default=DEFAULT_MODEL_SCREEN)
    parser.add_argument("--rf-screen", type=Path, default=DEFAULT_RF_SCREEN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    selector_run, model_screen, rf_screen = (
        args.selector_run.resolve(), args.model_screen.resolve(), args.rf_screen.resolve()
    )
    output_dir = args.output_dir.resolve()
    if not data_root.is_relative_to((REPO_ROOT / "data").resolve()):
        raise ValueError("Competition data must remain under data/")
    for path, label in ((selector_run, "selector run"), (model_screen, "model screen"), (rf_screen, "RF screen")):
        if not path.is_relative_to((REPO_ROOT / "artifacts").resolve()):
            raise ValueError(f"{label} must remain under artifacts/")
    if not output_dir.is_relative_to((REPO_ROOT / "artifacts/model-experiments").resolve()):
        raise ValueError("Outputs must remain under artifacts/model-experiments/")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)

    started = time.perf_counter()
    dataset = load_dataset(data_root)
    split = create_observation_split(dataset, REPO_ROOT / "data/splits")
    if split.sha256 != FROZEN_SPLIT_SHA256:
        raise RuntimeError("Frozen grouped split digest changed")
    selector_metrics = json.loads((selector_run / "metrics.json").read_text(encoding="utf-8"))
    model_metrics = json.loads((model_screen / "metrics.json").read_text(encoding="utf-8"))
    rf_metrics = json.loads((rf_screen / "metrics.json").read_text(encoding="utf-8"))
    for metrics, label in ((selector_metrics, "selector"), (model_metrics, "classifier"), (rf_metrics, "forest")):
        if metrics.get("split_sha256") != split.sha256:
            raise RuntimeError(f"{label} source uses a different frozen split")
    if model_metrics.get("holdout_used") is not False or rf_metrics.get("holdout_used") is not False:
        raise RuntimeError("Blend inputs are not certified as holdout-free OOF predictions")
    if rf_metrics.get("test_images_or_labels_used") is not False:
        raise RuntimeError("Forest screen reports test-image or test-label use")

    features_path = selector_run / "oof-component-features-targets.csv"
    model_prob_path = model_screen / "oof-model-probabilities.csv"
    rf_prob_path = rf_screen / "oof-model-probabilities.csv"
    features = pd.read_csv(features_path)
    model_probabilities = pd.read_csv(model_prob_path)
    rf_probabilities = pd.read_csv(rf_prob_path)
    keys = ["file_name", "component_id"]
    expected_keys = set(zip(features.file_name.astype(str), features.component_id.astype(int)))
    for rows, label in ((model_probabilities, "classifier"), (rf_probabilities, "forest")):
        got = set(zip(rows.file_name.astype(str), rows.component_id.astype(int)))
        if got != expected_keys:
            raise RuntimeError(f"{label} probabilities do not match the OOF feature rows")
    if set(features.file_name.astype(str)) - set(split.training_names):
        raise RuntimeError("OOF features contain observations outside the training partition")
    if set(features.oof_fold.astype(int)) != set(range(5)):
        raise RuntimeError("Expected all five grouped OOF folds")
    # Align explicitly by keys so row-order changes cannot silently corrupt predictions.
    classifier = model_probabilities.set_index(keys).loc[
        pd.MultiIndex.from_frame(features[keys]), ["logistic_c1.0", "extra_trees"]
    ]
    forest = rf_probabilities.set_index(keys).loc[
        pd.MultiIndex.from_frame(features[keys]), "rf_depth8_leaf10"
    ].to_numpy(dtype=np.float64)
    core = 0.70 * classifier["logistic_c1.0"].to_numpy(dtype=np.float64) + 0.30 * classifier["extra_trees"].to_numpy(dtype=np.float64)
    names = sorted(split.training_names)
    masks = load_prediction_rows(
        REPO_ROOT / "artifacts/model-experiments/oof-fusion-weight25-th035-a200/best-oof-predictions.csv",
        {Path(name).stem for name in names},
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for weight in RF_WEIGHTS:
        blended = (1.0 - weight) * core + weight * forest
        keyed = {
            (str(name), int(component_id)): float(value)
            for name, component_id, value in zip(features.file_name, features.component_id, blended)
        }
        for threshold in THRESHOLDS:
            predictions = predictions_from_masks(names, masks, keyed, threshold)
            metrics, _ = score_validation(dataset, predictions, names)
            global_metrics = metrics["global"]
            results.append({
                "rf_weight": weight,
                "threshold": threshold,
                "pq": float(global_metrics["pq"]),
                "tp": int(global_metrics["tp"]),
                "fp": int(global_metrics["fp"]),
                "fn": int(global_metrics["fn"]),
                "prediction_count": int(global_metrics["prediction_count_across_annotator_views"]),
            })
    results.sort(key=lambda row: row["pq"], reverse=True)
    pd.DataFrame(results).to_csv(output_dir / "blend-sweep.csv", index=False)
    report = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split_sha256": split.sha256,
        "oof_manifest_sha256": selector_metrics.get("oof_manifest_sha256"),
        "oof_observations": len(names),
        "component_rows": len(features),
        "holdout_used": False,
        "test_images_or_labels_used": False,
        "training_labels": "competition-provided annotations for the grouped OOF training partition only",
        "core_blend": "0.70 logistic C1 + 0.30 ExtraTrees cross-fitted probability",
        "forest_model": "rf_depth8_leaf10 cross-fitted probability",
        "rf_weight_grid": list(RF_WEIGHTS),
        "threshold_grid": list(THRESHOLDS),
        "baseline_oof_pq": 0.4242445362600016,
        "top_candidates": results[:10],
        "input_sha256": {
            "selector_metrics": sha256(selector_run / "metrics.json"),
            "selector_features": sha256(features_path),
            "classifier_probabilities": sha256(model_prob_path),
            "forest_metrics": sha256(rf_screen / "metrics.json"),
            "forest_probabilities": sha256(rf_prob_path),
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"top_candidates": results[:10], "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
