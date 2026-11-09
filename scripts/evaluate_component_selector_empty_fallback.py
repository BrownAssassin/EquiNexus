# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Evaluate the OOF-ranked best-component fallback on OOF and fixed holdout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_component_selector_oof import (  # noqa: E402
    fit_predict,
    load_prediction_rows,
    predictions_from_masks,
)
from filament_baseline.data import load_dataset  # noqa: E402
from filament_baseline.runner import _write_observation_scores, _write_predictions  # noqa: E402
from filament_baseline.scorer import score_validation  # noqa: E402
from filament_baseline.split import create_observation_split  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fallback_predictions(
    names: list[str],
    masks: dict[str, list[dict[str, Any]]],
    probability_by_key: dict[tuple[str, int], float],
    threshold: float,
) -> tuple[dict[str, list[Any]], list[str]]:
    adjusted = dict(probability_by_key)
    fallback_names = []
    for file_name in names:
        stem = Path(file_name).stem
        components = masks[stem]
        if not components:
            continue
        kept = [
            index
            for index in range(1, len(components) + 1)
            if adjusted[(file_name, index)] >= threshold
        ]
        if not kept:
            best = max(
                range(1, len(components) + 1),
                key=lambda index: adjusted[(file_name, index)],
            )
            adjusted[(file_name, best)] = threshold
            fallback_names.append(file_name)
    return predictions_from_masks(names, masks, adjusted, threshold), fallback_names


def count_array(rows: list[Any]) -> np.ndarray:
    return np.asarray(
        [[row.tp_iou_sum, row.tp, row.fp, row.fn] for row in rows], dtype=np.float64
    )


def aggregate_pq(samples: np.ndarray) -> np.ndarray:
    iou_sum = samples[..., 0].sum(axis=1)
    tp = samples[..., 1].sum(axis=1)
    fp = samples[..., 2].sum(axis=1)
    fn = samples[..., 3].sum(axis=1)
    denominator = tp + 0.5 * fp + 0.5 * fn
    return np.divide(
        iou_sum,
        denominator,
        out=np.zeros_like(iou_sum),
        where=denominator > 0,
    )


def read_saved_counts(path: Path) -> np.ndarray:
    rows = pd.read_csv(path).sort_values("file_name")
    return rows[["tp_iou_sum", "tp_edges", "fp", "fn"]].to_numpy(dtype=np.float64)


def paired_bootstrap(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    if baseline.shape != candidate.shape:
        raise ValueError("Paired bootstrap arrays have different shapes")
    baseline_score = float(aggregate_pq(baseline[None, ...])[0])
    candidate_score = float(aggregate_pq(candidate[None, ...])[0])
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(baseline), size=(resamples, len(baseline)))
    deltas = aggregate_pq(candidate[indices]) - aggregate_pq(baseline[indices])
    return {
        "baseline_pq": baseline_score,
        "fallback_pq": candidate_score,
        "delta": candidate_score - baseline_score,
        "paired_delta_ci_95": [float(value) for value in np.quantile(deltas, [0.025, 0.975])],
        "probability_fallback_better": float(np.mean(deltas > 0)),
        "resamples": resamples,
        "seed": seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--selector-run", type=Path, default=REPO_ROOT / "artifacts/model-experiments/component-selector-official-pq-20261109-rle-fixed")
    parser.add_argument("--run-name", default="component-selector-empty-fallback-20261109")
    parser.add_argument("--resamples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    selector_run = args.selector_run.resolve()
    if not data_root.is_relative_to((REPO_ROOT / "data").resolve()):
        raise ValueError("Dataset must remain inside data/")
    if not selector_run.is_relative_to((REPO_ROOT / "artifacts").resolve()):
        raise ValueError("Selector inputs must remain inside artifacts/")
    output_dir = (args.output_dir or (REPO_ROOT / "artifacts/verification" / args.run_name)).resolve()
    if not output_dir.is_relative_to((REPO_ROOT / "artifacts/verification").resolve()):
        raise ValueError("Verification outputs must remain inside artifacts/verification/")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)

    started = time.perf_counter()
    dataset = load_dataset(data_root)
    split = create_observation_split(dataset, REPO_ROOT / "data/splits")
    selector_metrics_path = selector_run / "metrics.json"
    selector_metrics = json.loads(selector_metrics_path.read_text(encoding="utf-8"))
    if split.sha256 != selector_metrics["split_sha256"]:
        raise RuntimeError("Selector metrics use a different frozen split")
    threshold = float(selector_metrics["fit"]["selected_threshold"])

    oof_names = sorted(split.training_names)
    holdout_names = sorted(split.validation_names)
    oof_features = pd.read_csv(selector_run / "oof-component-features-targets.csv")
    holdout_features = pd.read_csv(selector_run / "holdout-component-features.csv")
    oof_probability = np.full(len(oof_features), np.nan, dtype=np.float64)
    folds = oof_features.oof_fold.to_numpy(dtype=int)
    for fold in range(5):
        heldout = folds == fold
        oof_probability[heldout], _ = fit_predict(
            oof_features.loc[~heldout], oof_features.loc[heldout]
        )
    if not np.isfinite(oof_probability).all():
        raise RuntimeError("Missing cross-fitted OOF selector probabilities")
    oof_features["selector_probability"] = oof_probability
    holdout_features["selector_probability"], _ = fit_predict(oof_features, holdout_features)

    oof_keys = {
        (str(row.file_name), int(row.component_id)): float(row.selector_probability)
        for row in oof_features.itertuples()
    }
    holdout_keys = {
        (str(row.file_name), int(row.component_id)): float(row.selector_probability)
        for row in holdout_features.itertuples()
    }
    oof_masks = load_prediction_rows(
        REPO_ROOT / "artifacts/model-experiments/oof-fusion-weight25-th035-a200/best-oof-predictions.csv",
        {Path(name).stem for name in oof_names},
    )
    holdout_masks = load_prediction_rows(
        REPO_ROOT / "artifacts/model-experiments/yolo11s-25perann-75consensus-foldmeans-valid-holdout-th035-a200/predictions.csv",
        {Path(name).stem for name in holdout_names},
    )
    oof_predictions, oof_fallback_names = fallback_predictions(
        oof_names, oof_masks, oof_keys, threshold
    )
    holdout_predictions, holdout_fallback_names = fallback_predictions(
        holdout_names, holdout_masks, holdout_keys, threshold
    )
    oof_metrics, oof_rows = score_validation(dataset, oof_predictions, oof_names)
    holdout_metrics, holdout_rows = score_validation(dataset, holdout_predictions, holdout_names)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_predictions(output_dir / "fallback-oof-predictions.csv", oof_predictions)
    _write_observation_scores(output_dir / "fallback-oof-per-observation.csv", oof_rows, oof_predictions)
    _write_predictions(output_dir / "fallback-holdout-predictions.csv", holdout_predictions)
    _write_observation_scores(
        output_dir / "fallback-holdout-per-observation.csv", holdout_rows, holdout_predictions
    )

    baseline_oof = read_saved_counts(selector_run / "selector-oof-per-observation.csv")
    baseline_holdout = read_saved_counts(selector_run / "selector-holdout-per-observation.csv")
    report = {
        "status": "complete",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split_sha256": split.sha256,
        "oof_manifest_sha256": selector_metrics["oof_manifest_sha256"],
        "threshold": threshold,
        "policy": "If no component passes the threshold but base candidates exist, keep the highest-probability component.",
        "test_images_or_labels_used": False,
        "holdout_used_for_selection": False,
        "fallback_observations": {
            "oof": oof_fallback_names,
            "holdout": holdout_fallback_names,
        },
        "oof": {
            "baseline_pq": float(selector_metrics["fit"]["oof_selected"]["global"]["pq"]),
            "fallback_pq": float(oof_metrics["global"]["pq"]),
            "paired_bootstrap": paired_bootstrap(
                baseline_oof, count_array(oof_rows), resamples=args.resamples, seed=args.seed
            ),
        },
        "holdout": {
            "baseline_pq": float(selector_metrics["fit"]["holdout_selected"]["global"]["pq"]),
            "fallback_pq": float(holdout_metrics["global"]["pq"]),
            "paired_bootstrap": paired_bootstrap(
                baseline_holdout,
                count_array(holdout_rows),
                resamples=args.resamples,
                seed=args.seed,
            ),
        },
        "runtime_seconds": time.perf_counter() - started,
        "source_sha256": {
            "selector_metrics": sha256(selector_metrics_path),
            "script": sha256(Path(__file__).resolve()),
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
