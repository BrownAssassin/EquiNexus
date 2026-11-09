# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Evaluate one OOF-selected selector-probability blend on the frozen holdout."""

from __future__ import annotations

import argparse
import csv
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

from run_component_selector_oof import (  # noqa: E402
    load_prediction_rows,
    predictions_from_masks,
)
from screen_component_selector_models import _fit_model  # noqa: E402
from run_component_selector_oof import FEATURES  # noqa: E402
from filament_baseline.data import load_dataset  # noqa: E402
from filament_baseline.scorer import score_validation  # noqa: E402
from filament_baseline.split import create_observation_split  # noqa: E402


FROZEN_SPLIT_SHA256 = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"
EXTRA_TREES_WEIGHT = 0.30
THRESHOLD = 0.45
BASELINE_THRESHOLD = 0.40
BOOTSTRAP_RESAMPLES = 20_000


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def per_observation_rows(rows: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "file_name": row.file_name,
            "tp_iou_sum": row.tp_iou_sum,
            "tp_edges": row.tp,
            "fp": row.fp,
            "fn": row.fn,
            "pq": row.pq,
        }
        for row in rows
    ]


def paired_bootstrap(
    baseline_rows: list[Any],
    candidate_rows: list[Any],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 42,
) -> dict[str, Any]:
    baseline = {row.file_name: row for row in baseline_rows}
    candidate = {row.file_name: row for row in candidate_rows}
    names = sorted(baseline)
    if set(names) != set(candidate) or not names:
        raise ValueError("Paired bootstrap requires matching nonempty observation sets")
    base = np.asarray(
        [[baseline[n].tp_iou_sum, baseline[n].tp, baseline[n].fp, baseline[n].fn] for n in names],
        dtype=np.float64,
    )
    cand = np.asarray(
        [[candidate[n].tp_iou_sum, candidate[n].tp, candidate[n].fp, candidate[n].fn] for n in names],
        dtype=np.float64,
    )
    indices = np.random.default_rng(seed).integers(0, len(names), size=(resamples, len(names)))

    def aggregate(samples: np.ndarray) -> np.ndarray:
        totals = samples.sum(axis=1)
        denominator = totals[:, 1] + 0.5 * totals[:, 2] + 0.5 * totals[:, 3]
        return np.divide(
            totals[:, 0], denominator,
            out=np.zeros_like(denominator), where=denominator > 0,
        )

    base_pq = aggregate(base[None, ...])[0]
    cand_pq = aggregate(cand[None, ...])[0]
    delta = aggregate(cand[indices]) - aggregate(base[indices])
    return {
        "resampling_unit": "original observation; annotator-view scorer counts are grouped",
        "observation_count": len(names),
        "resamples": resamples,
        "seed": seed,
        "baseline_pq": float(base_pq),
        "candidate_pq": float(cand_pq),
        "paired_delta_candidate_minus_baseline": {
            "point_estimate": float(cand_pq - base_pq),
            "bootstrap_ci_95": [float(value) for value in np.quantile(delta, [0.025, 0.975])],
            "probability_candidate_better": float(np.mean(delta > 0)),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path,
        default=REPO_ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026",
    )
    parser.add_argument(
        "--model-screen", type=Path,
        default=REPO_ROOT / "artifacts/model-experiments/component-selector-model-screen-20261109",
    )
    parser.add_argument(
        "--selector-run", type=Path,
        default=REPO_ROOT / "artifacts/model-experiments/component-selector-official-pq-20261109-rle-fixed",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "artifacts/model-experiments/component-selector-probability-blend-holdout-20261109",
    )
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    model_screen = args.model_screen.resolve()
    selector_run = args.selector_run.resolve()
    output_dir = args.output_dir.resolve()
    if not data_root.is_relative_to((REPO_ROOT / "data").resolve()):
        raise ValueError("Competition data must stay under data/")
    for path, label in ((model_screen, "model screen"), (selector_run, "selector run")):
        if not path.is_relative_to((REPO_ROOT / "artifacts").resolve()):
            raise ValueError(f"{label} must stay under artifacts/")
    if not output_dir.is_relative_to((REPO_ROOT / "artifacts/model-experiments").resolve()):
        raise ValueError("Experiment outputs must stay under artifacts/model-experiments/")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)

    started = time.perf_counter()
    dataset = load_dataset(data_root)
    split = create_observation_split(dataset, REPO_ROOT / "data/splits")
    if split.sha256 != FROZEN_SPLIT_SHA256:
        raise RuntimeError(f"Frozen split changed: {split.sha256}")
    screen_metrics_path = model_screen / "metrics.json"
    screen_metrics = json.loads(screen_metrics_path.read_text(encoding="utf-8"))
    selector_metrics_path = selector_run / "metrics.json"
    selector_metrics = json.loads(selector_metrics_path.read_text(encoding="utf-8"))
    if screen_metrics.get("split_sha256") != split.sha256 or screen_metrics.get("holdout_used") is not False:
        raise RuntimeError("OOF probability screen is not certified holdout-free")
    if selector_metrics.get("split_sha256") != split.sha256:
        raise RuntimeError("Selector source uses a different frozen split")

    train_features = pd.read_csv(selector_run / "oof-component-features-targets.csv")
    holdout_features = pd.read_csv(selector_run / "holdout-component-features.csv")
    model_probability_path = model_screen / "oof-model-probabilities.csv"
    model_probability_rows = pd.read_csv(model_probability_path)
    if "oof_fold" not in train_features or set(train_features.oof_fold.astype(int)) != set(range(5)):
        raise ValueError("Training features do not cover all five frozen OOF folds")
    keys_train = set(zip(train_features.file_name.astype(str), train_features.component_id.astype(int)))
    keys_probability = set(zip(model_probability_rows.file_name.astype(str), model_probability_rows.component_id.astype(int)))
    if keys_train != keys_probability:
        raise ValueError("OOF probability rows do not exactly match selector training features")
    names = sorted(split.validation_names)
    stems = {Path(name).stem for name in names}
    holdout_base_path = selector_run / "baseline-holdout-predictions.csv"
    masks = load_prediction_rows(holdout_base_path, stems)
    expected_feature_names = {name for name in names if masks.get(Path(name).stem)}
    if set(holdout_features.file_name.astype(str)) != expected_feature_names:
        raise ValueError("Holdout component features do not match nonempty base-prediction observations")
    logistic_probability = _fit_model("logistic_c1.0", train_features, holdout_features)
    trees_probability = _fit_model("extra_trees", train_features, holdout_features)
    if len(logistic_probability) != len(holdout_features) or len(trees_probability) != len(holdout_features):
        raise RuntimeError("Selector model returned the wrong number of holdout probabilities")

    keys_holdout = list(zip(
        holdout_features.file_name.astype(str),
        holdout_features.component_id.astype(int),
    ))
    logistic_keyed = {key: float(value) for key, value in zip(keys_holdout, logistic_probability)}
    blend_probability = (
        (1.0 - EXTRA_TREES_WEIGHT) * logistic_probability
        + EXTRA_TREES_WEIGHT * trees_probability
    )
    blend_keyed = {key: float(value) for key, value in zip(keys_holdout, blend_probability)}

    baseline_predictions = predictions_from_masks(names, masks, logistic_keyed, BASELINE_THRESHOLD)
    candidate_predictions = predictions_from_masks(names, masks, blend_keyed, THRESHOLD)
    baseline_metrics, baseline_rows = score_validation(dataset, baseline_predictions, names)
    candidate_metrics, candidate_rows = score_validation(dataset, candidate_predictions, names)
    bootstrap = paired_bootstrap(baseline_rows, candidate_rows)
    output_dir.mkdir(parents=True, exist_ok=True)

    for label, rows in (("baseline", baseline_rows), ("candidate", candidate_rows)):
        with (output_dir / f"{label}-per-observation.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["file_name", "tp_iou_sum", "tp_edges", "fp", "fn", "pq"])
            writer.writeheader()
            writer.writerows(per_observation_rows(rows))
    from filament_baseline.runner import _write_predictions

    _write_predictions(output_dir / "baseline-predictions.csv", baseline_predictions)
    _write_predictions(output_dir / "candidate-predictions.csv", candidate_predictions)
    report = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split_sha256": split.sha256,
        "oof_manifest_sha256": selector_metrics.get("oof_manifest_sha256"),
        "holdout_observations": len(names),
        "holdout_component_rows": len(holdout_features),
        "training_observations": len(split.training_names),
        "test_images_or_labels_used": False,
        "external_labels_used": False,
        "selection_policy": "The 0.30 ExtraTrees probability weight and 0.45 threshold were selected using only the separate cross-fitted OOF screen; this script makes no holdout search.",
        "candidate": {
            "logistic_weight": 1.0 - EXTRA_TREES_WEIGHT,
            "extra_trees_weight": EXTRA_TREES_WEIGHT,
            "threshold": THRESHOLD,
            "global_metrics": candidate_metrics["global"],
            "evaluation_metrics": candidate_metrics,
        },
        "baseline": {
            "model": "grouped-OOF-fitted standardized L2 logistic selector",
            "threshold": BASELINE_THRESHOLD,
            "global_metrics": baseline_metrics["global"],
            "evaluation_metrics": baseline_metrics,
        },
        "paired_observation_bootstrap": bootstrap,
        "input_sha256": {
            "oof_model_probabilities": file_sha256(model_probability_path),
            "oof_component_features": file_sha256(selector_run / "oof-component-features-targets.csv"),
            "holdout_component_features": file_sha256(selector_run / "holdout-component-features.csv"),
            "holdout_base_predictions": file_sha256(holdout_base_path),
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"baseline": baseline_metrics["global"], "candidate": candidate_metrics["global"], "paired_bootstrap": bootstrap["paired_delta_candidate_minus_baseline"], "output": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
