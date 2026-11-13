#!/usr/bin/env python3
# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Paired, observation-grouped comparison of GONG-aux and native U-Net OOF maps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from filament_baseline.data import load_dataset
from filament_baseline.scorer import score_validation
from filament_baseline.split import create_observation_split
from train_native_unet_grouped import predictions_for_config

EXPECTED_SPLIT_SHA256 = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_summary(run_dir: Path, expected_mode: str) -> dict[str, Any]:
    summary_path = run_dir / "oof-summary.json"
    metadata_path = run_dir / "run-metadata-oof.json"
    if not summary_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Incomplete OOF result under {run_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("mode") != expected_mode or summary.get("split_sha256") != EXPECTED_SPLIT_SHA256:
        raise RuntimeError(f"Unexpected OOF mode or split digest under {run_dir}")
    return summary


def read_fold_map(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return {row["file_name"]: int(row["fold"]) for row in csv.DictReader(stream)}


def load_oof_predictions(run_dir: Path, summary: dict[str, Any]) -> tuple[list[str], dict[str, np.ndarray]]:
    cache_index = json.loads((run_dir / "cache" / "cache-index.json").read_text(encoding="utf-8"))
    names = list(cache_index["names"])
    epoch = int(summary["selected_oof_parameters"]["epoch"])
    relative_probability_path = Path(summary["prediction_files"][str(epoch)])
    probability_path = relative_probability_path if relative_probability_path.is_absolute() else REPO_ROOT / relative_probability_path
    probabilities = np.load(probability_path, mmap_mode="r")
    if probabilities.shape != (len(names), 1888, 1888) or probabilities.dtype != np.float16:
        raise RuntimeError(f"Unexpected OOF probability array: {probability_path}")
    return names, {name: probabilities[index] for index, name in enumerate(names)}


def paired_bootstrap(
    baseline_rows: list[Any],
    candidate_rows: list[Any],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    if len(baseline_rows) != len(candidate_rows) or not baseline_rows:
        raise ValueError("Paired OOF observation rows do not match")
    if [row.file_name for row in baseline_rows] != [row.file_name for row in candidate_rows]:
        raise ValueError("Paired OOF observations are not in the same order")
    baseline_iou = np.asarray([row.tp_iou_sum for row in baseline_rows], dtype=np.float64)
    baseline_tp = np.asarray([row.tp for row in baseline_rows], dtype=np.float64)
    baseline_fp = np.asarray([row.fp for row in baseline_rows], dtype=np.float64)
    baseline_fn = np.asarray([row.fn for row in baseline_rows], dtype=np.float64)
    candidate_iou = np.asarray([row.tp_iou_sum for row in candidate_rows], dtype=np.float64)
    candidate_tp = np.asarray([row.tp for row in candidate_rows], dtype=np.float64)
    candidate_fp = np.asarray([row.fp for row in candidate_rows], dtype=np.float64)
    candidate_fn = np.asarray([row.fn for row in candidate_rows], dtype=np.float64)

    rng = np.random.default_rng(seed)
    deltas: list[np.ndarray] = []
    n = len(baseline_rows)
    for start in range(0, resamples, 500):
        count = min(500, resamples - start)
        indices = rng.integers(0, n, size=(count, n))
        baseline_denominator = (
            baseline_tp[indices]
            + 0.5 * baseline_fp[indices]
            + 0.5 * baseline_fn[indices]
        ).sum(axis=1)
        candidate_denominator = (
            candidate_tp[indices]
            + 0.5 * candidate_fp[indices]
            + 0.5 * candidate_fn[indices]
        ).sum(axis=1)
        baseline_pq = np.divide(
            baseline_iou[indices].sum(axis=1),
            baseline_denominator,
            out=np.zeros(count, dtype=np.float64),
            where=baseline_denominator > 0,
        )
        candidate_pq = np.divide(
            candidate_iou[indices].sum(axis=1),
            candidate_denominator,
            out=np.zeros(count, dtype=np.float64),
            where=candidate_denominator > 0,
        )
        deltas.append(candidate_pq - baseline_pq)
    values = np.concatenate(deltas)
    return {
        "resamples": int(resamples),
        "seed": int(seed),
        "mean_delta": float(values.mean()),
        "median_delta": float(np.median(values)),
        "paired_delta_ci_95": [float(value) for value in np.quantile(values, [0.025, 0.975])],
        "p_candidate_better": float(np.mean(values > 0.0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "model-experiments" / "native-unet-grouped-20261111",
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "model-experiments" / "native-unet-gong-aux-20261112",
    )
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--split-dir", type=Path, default=REPO_ROOT / "data" / "splits")
    parser.add_argument("--resamples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.resamples < 100:
        parser.error("--resamples must be at least 100")
    baseline_dir = args.baseline_dir.expanduser().resolve()
    candidate_dir = args.candidate_dir.expanduser().resolve()
    for path in (baseline_dir, candidate_dir):
        if not path.is_relative_to((REPO_ROOT / "artifacts" / "model-experiments").resolve()):
            raise ValueError("OOF run directories must remain under artifacts/model-experiments")
    data_root = args.data_root.expanduser().resolve()
    split_dir = args.split_dir.expanduser().resolve()
    if not data_root.is_relative_to((REPO_ROOT / "data").resolve()):
        raise ValueError("Competition data must remain under data/")
    if not split_dir.is_relative_to((REPO_ROOT / "data" / "splits").resolve()):
        raise ValueError("Split manifests must remain under data/splits/")

    baseline_summary = read_summary(baseline_dir, "oof")
    candidate_summary = read_summary(candidate_dir, "oof")
    baseline_names, baseline_probability = load_oof_predictions(baseline_dir, baseline_summary)
    candidate_names, candidate_probability = load_oof_predictions(candidate_dir, candidate_summary)
    if baseline_names != candidate_names:
        raise RuntimeError("Baseline and GONG OOF observation lists differ")
    baseline_folds = read_fold_map(baseline_dir / "oof-folds.csv")
    candidate_folds = read_fold_map(candidate_dir / "oof-folds.csv")
    if baseline_folds != candidate_folds or set(baseline_folds) != set(baseline_names):
        raise RuntimeError("Baseline and GONG OOF fold assignments differ")

    dataset = load_dataset(data_root)
    split = create_observation_split(dataset, split_dir, seed=42, validation_fraction=0.2)
    if split.sha256 != EXPECTED_SPLIT_SHA256 or set(baseline_names) != set(split.training_names):
        raise RuntimeError("OOF scorer received observations outside the frozen training partition")
    gong_manifest_path = (
        REPO_ROOT
        / "data"
        / "reference"
        / "public-datasets"
        / "vhrabar-gong-solar-filaments"
        / "train-only-manifest.json"
    )
    gong_manifest = json.loads(gong_manifest_path.read_text(encoding="utf-8"))
    if sha256(gong_manifest_path) != candidate_summary.get("gong_target_manifest_sha256"):
        raise RuntimeError("GONG OOF summary uses a different target manifest")
    if gong_manifest.get("validation_target_files_downloaded") != 0:
        raise RuntimeError("GONG OOF target manifest reports holdout downloads")
    frozen_training = set(split.training_names)
    frozen_holdout = set(split.validation_names)
    for target in gong_manifest.get("downloaded_files", []):
        target_observation = Path(target["name"]).stem + ".jpeg"
        if target_observation not in frozen_training or target_observation in frozen_holdout:
            raise RuntimeError(f"GONG OOF target manifest contains a non-training observation: {target_observation}")
    baseline_config = baseline_summary["selected_oof_parameters"]
    candidate_config = candidate_summary["selected_oof_parameters"]
    baseline_predictions = predictions_for_config(
        baseline_probability,
        baseline_names,
        threshold=float(baseline_config["threshold"]),
        min_area=int(baseline_config["min_area"]),
        closing_kernel=int(baseline_config["closing_kernel"]),
    )
    candidate_predictions = predictions_for_config(
        candidate_probability,
        candidate_names,
        threshold=float(candidate_config["threshold"]),
        min_area=int(candidate_config["min_area"]),
        closing_kernel=int(candidate_config["closing_kernel"]),
    )
    baseline_metrics, baseline_rows = score_validation(dataset, baseline_predictions, baseline_names)
    candidate_metrics, candidate_rows = score_validation(dataset, candidate_predictions, candidate_names)
    per_observation_path = candidate_dir / "oof-comparison-per-observation.csv"
    with per_observation_path.open("w", encoding="utf-8", newline="") as stream:
        columns = [
            "file_name",
            "baseline_pq",
            "baseline_tp",
            "baseline_fp",
            "baseline_fn",
            "gong_pq",
            "gong_tp",
            "gong_fp",
            "gong_fn",
        ]
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for baseline_row, candidate_row in zip(baseline_rows, candidate_rows, strict=True):
            writer.writerow(
                {
                    "file_name": baseline_row.file_name,
                    "baseline_pq": f"{baseline_row.pq:.10f}",
                    "baseline_tp": baseline_row.tp,
                    "baseline_fp": baseline_row.fp,
                    "baseline_fn": baseline_row.fn,
                    "gong_pq": f"{candidate_row.pq:.10f}",
                    "gong_tp": candidate_row.tp,
                    "gong_fp": candidate_row.fp,
                    "gong_fn": candidate_row.fn,
                }
            )
    comparison = {
        "split_sha256": split.sha256,
        "partition": "cross-fitted frozen training observations; no holdout observations or GONG holdout targets",
        "observations": len(baseline_names),
        "annotator_views": sum(len(dataset.views_by_observation[name]) for name in baseline_names),
        "gong_target_manifest_sha256": candidate_summary["gong_target_manifest_sha256"],
        "baseline": {
            "run_dir": str(baseline_dir.relative_to(REPO_ROOT)),
            "selected_oof_parameters": baseline_config,
            "global_metrics": baseline_metrics["global"],
            "fold_manifest_sha256": sha256(baseline_dir / "oof-folds.csv"),
        },
        "gong_auxiliary": {
            "run_dir": str(candidate_dir.relative_to(REPO_ROOT)),
            "selected_oof_parameters": candidate_config,
            "global_metrics": candidate_metrics["global"],
            "fold_manifest_sha256": sha256(candidate_dir / "oof-folds.csv"),
        },
        "paired_bootstrap_candidate_vs_baseline": paired_bootstrap(
            baseline_rows, candidate_rows, resamples=args.resamples, seed=args.seed
        ),
        "per_observation_csv": str(per_observation_path.relative_to(REPO_ROOT)),
        "holdout_used": False,
        "test_data_used": False,
    }
    output_path = candidate_dir / "oof-comparison-vs-native.json"
    output_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path.relative_to(REPO_ROOT)), **comparison["paired_bootstrap_candidate_vs_baseline"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
