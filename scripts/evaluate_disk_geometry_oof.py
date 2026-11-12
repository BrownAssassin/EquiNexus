from __future__ import annotations

# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Evaluate image-fitted solar-disk clipping on cross-fitted OOF instances."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament_baseline.data import load_dataset
from filament_baseline.predictor import PredictedInstance, encode_binary_mask
from filament_baseline.scorer import score_validation
from filament_baseline.split import create_observation_split
from filament_baseline.vendor.ameypawar_disk import detect_disk, read_image


EXPECTED_SPLIT = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"
SHAPE = (2048, 2048)
DEFAULT_PREDICTIONS = ROOT / "artifacts/model-experiments/public-unet-yolo-family-vote-oof-conf025-20261111/oof_predictions.csv"
DEFAULT_OUTPUT = ROOT / "artifacts/model-experiments/public-disk-geometry-oof-20261111"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_oof_predictions(path: Path, observation_names: list[str]) -> dict[str, list[dict[str, Any]]]:
    observations = {Path(name).stem: [] for name in observation_names}
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["filament_id", "segmentation_rle"]:
            raise ValueError("Unexpected OOF prediction CSV columns")
        for row in reader:
            filament_id = str(row["filament_id"])
            stem, separator, suffix = filament_id.rpartition("_")
            if not separator or not suffix.isdigit() or stem not in observations:
                raise ValueError(f"Invalid or unexpected OOF filament id: {filament_id}")
            observations[stem].append({
                "filament_id": filament_id,
                "rle": {"size": list(SHAPE), "counts": str(row["segmentation_rle"]).encode("ascii")},
            })
    return observations


def as_predictions(
    names: list[str], observations: dict[str, list[dict[str, Any]]], disks: dict[str, dict[str, float]] | None = None,
    shrink: float | None = None,
) -> dict[str, list[PredictedInstance]]:
    predictions: dict[str, list[PredictedInstance]] = {}
    for name in names:
        stem = Path(name).stem
        disk_mask = None
        if disks is not None and shrink is not None:
            geometry = disks[stem]
            from filament_baseline.vendor.ameypawar_disk import Disk
            disk_mask = Disk(**geometry).mask(SHAPE, shrink=shrink)
        rows = []
        for row in observations[stem]:
            binary = mask_utils.decode(row["rle"]).astype(bool)
            if disk_mask is not None:
                binary &= disk_mask
            area = int(binary.sum())
            if area == 0:
                continue
            rle = row["rle"] if disk_mask is None else encode_binary_mask(binary)
            rows.append(PredictedInstance(
                filament_id=f"{stem}_{len(rows) + 1}", image_stem=stem, area=area, rle=rle,
            ))
        predictions[name] = rows
    return predictions


def paired_bootstrap(
    baseline_rows: list[Any], candidate_rows: list[Any], *, resamples: int = 20_000, seed: int = 42,
) -> dict[str, Any]:
    if [row.file_name for row in baseline_rows] != [row.file_name for row in candidate_rows]:
        raise ValueError("Baseline and candidate observation rows differ")
    baseline = [
        np.asarray([getattr(row, key) for row in rows], dtype=np.float64)
        for rows in (baseline_rows, candidate_rows)
        for key in ("tp_iou_sum", "tp", "fp", "fn")
    ]
    base_iou, base_tp, base_fp, base_fn, cand_iou, cand_tp, cand_fp, cand_fn = baseline
    rng = np.random.default_rng(seed)
    deltas = []
    count = len(baseline_rows)
    for start in range(0, resamples, 500):
        size = min(500, resamples - start)
        index = rng.integers(0, count, size=(size, count))
        base_den = (base_tp[index] + 0.5 * base_fp[index] + 0.5 * base_fn[index]).sum(axis=1)
        cand_den = (cand_tp[index] + 0.5 * cand_fp[index] + 0.5 * cand_fn[index]).sum(axis=1)
        base_pq = np.divide(base_iou[index].sum(axis=1), base_den, out=np.zeros(size), where=base_den > 0)
        cand_pq = np.divide(cand_iou[index].sum(axis=1), cand_den, out=np.zeros(size), where=cand_den > 0)
        deltas.append(cand_pq - base_pq)
    values = np.concatenate(deltas)
    return {
        "resamples": resamples, "seed": seed, "mean_delta": float(values.mean()),
        "paired_delta_ci_95": [float(value) for value in np.quantile(values, [0.025, 0.975])],
        "p_candidate_better": float(np.mean(values > 0)),
    }


def metric_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    global_metrics = metrics["global"]
    return {key: global_metrics[key] for key in ("pq", "tp", "fp", "fn", "prediction_count") if key in global_metrics}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shrinks", default="0.97,0.98,0.99,1.00,1.01")
    args = parser.parse_args()
    started = time.perf_counter()
    data_root = args.data_root.expanduser().resolve()
    predictions_path = args.predictions.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for label, path in (("data", data_root), ("predictions", predictions_path), ("output", output_dir)):
        if not path.is_relative_to(ROOT):
            raise ValueError(f"{label} path must remain inside the workspace: {path}")
    if not predictions_path.is_relative_to(ROOT / "artifacts"):
        raise ValueError("OOF predictions must be in artifacts/")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    shrinks = tuple(float(item) for item in args.shrinks.split(",") if item.strip())
    if not shrinks or any(not 0.9 <= value <= 1.05 for value in shrinks):
        raise ValueError("Disk shrink factors must be between 0.9 and 1.05")

    dataset = load_dataset(data_root)
    split = create_observation_split(dataset, ROOT / "data/splits", seed=42, validation_fraction=0.2)
    if split.sha256 != EXPECTED_SPLIT:
        raise RuntimeError(f"Frozen split digest changed: {split.sha256}")
    names = list(sorted(split.training_names))
    holdout = set(split.validation_names)
    if set(names) & holdout or len(names) != 565 or len(holdout) != 142:
        raise RuntimeError("Frozen observation partition failed its grouped size/disjointness check")
    observations = read_oof_predictions(predictions_path, names)
    disk_geometry: dict[str, dict[str, float]] = {}
    image_dir = dataset.train_image_dir
    for index, name in enumerate(names, start=1):
        image = read_image(image_dir / name)
        disk = detect_disk(image)
        disk_geometry[Path(name).stem] = {"cx": disk.cx, "cy": disk.cy, "radius": disk.radius}
        if index % 50 == 0 or index == len(names):
            print(f"[disk] fitted {index}/{len(names)} observation limbs", flush=True)

    baseline, baseline_rows = score_validation(dataset, as_predictions(names, observations), names)
    results: dict[str, Any] = {}
    per_observation: dict[str, list[dict[str, Any]]] = {
        row.file_name: [{"file_name": row.file_name, "baseline_pq": row.pq, "baseline_tp": row.tp,
                         "baseline_fp": row.fp, "baseline_fn": row.fn}]
        for row in baseline_rows
    }
    for shrink in shrinks:
        predictions = as_predictions(names, observations, disk_geometry, shrink)
        metrics, rows = score_validation(dataset, predictions, names)
        results[str(shrink)] = {
            "global_metrics": metric_summary(metrics),
            "paired_bootstrap_vs_unclipped": paired_bootstrap(baseline_rows, rows),
            "per_observation_count": len(rows),
        }
        for row in rows:
            per_observation[row.file_name][0][f"shrink_{shrink:g}_pq"] = row.pq
            per_observation[row.file_name][0][f"shrink_{shrink:g}_tp"] = row.tp
            per_observation[row.file_name][0][f"shrink_{shrink:g}_fp"] = row.fp
            per_observation[row.file_name][0][f"shrink_{shrink:g}_fn"] = row.fn
        print(f"[score] shrink={shrink:.3f} PQ={metrics['global']['pq']:.6f}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "disk-geometries.json").write_text(
        json.dumps(disk_geometry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "per-observation.csv").open("w", encoding="utf-8", newline="") as stream:
        fieldnames = list(next(iter(per_observation.values()))[0])
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(value[0] for value in per_observation.values())
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    report = {
        "method": "per-observation limb fit; clip cross-fitted family-vote instances to fitted circle",
        "split_sha256": split.sha256,
        "partition": "frozen training observations only (five-fold OOF predictions)",
        "training_observations": len(names),
        "holdout_observations": len(holdout),
        "holdout_labels_read": False,
        "baseline": metric_summary(baseline),
        "candidate_shrink_factors": list(shrinks),
        "candidates": results,
        "prediction_sha256": sha256(predictions_path),
        "disk_geometry_source": "Amey Pawar, filament-seg, MIT, upstream commit 81a739bb7786a44b989f5d76afbb6f9d22b5d1dd",
        "disk_geometry_module_sha256": sha256(ROOT / "src/filament_baseline/vendor/ameypawar_disk.py"),
        "source_commit": commit,
        "source_script_sha256": sha256(Path(__file__)),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "outputs": {"disk_geometries": "disk-geometries.json", "per_observation": "per-observation.csv"},
    }
    (output_dir / "run-metadata.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"baseline_pq": baseline["global"]["pq"], "candidates": results}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
