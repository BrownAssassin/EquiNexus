# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools import mask as mask_utils

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from filament_baseline.data import load_dataset
from filament_baseline.detector_fusion import (
    DetectorFusionConfig,
    fuse_detector_instances,
    solar_disk_mask_from_geometry,
)
from filament_baseline.predictor import PredictedInstance, encode_binary_mask
from filament_baseline.scorer import score_validation
from filament_baseline.split import create_observation_split


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _counts(value: str) -> bytes:
    return value.encode("ascii")


def _load_unet_masks(path: Path, stems: set[str]) -> dict[str, list[str]]:
    masks: dict[str, list[str]] = {stem: [] for stem in stems}
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["filament_id", "segmentation_rle"]:
            raise ValueError("U-Net OOF predictions must contain filament_id,segmentation_rle")
        for row in reader:
            stem = str(row["filament_id"]).rsplit("_", maxsplit=1)[0]
            if stem not in masks:
                raise ValueError(f"Unexpected U-Net OOF observation: {stem}")
            masks[stem].append(str(row["segmentation_rle"]))
    return masks


def _make_predictions(labels: np.ndarray, file_name: str) -> list[PredictedInstance]:
    stem = Path(file_name).stem
    predictions = []
    for label in range(1, int(labels.max()) + 1):
        binary = labels == label
        area = int(binary.sum())
        if area:
            predictions.append(
                PredictedInstance(
                    filament_id=f"{stem}_{len(predictions) + 1}",
                    image_stem=stem,
                    area=area,
                    rle=encode_binary_mask(binary),
                )
            )
    return predictions


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score cross-fitted YOLO plus U-Net instance fusion on training OOF groups."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--detector-predictions", type=Path, required=True)
    parser.add_argument("--unet-predictions", type=Path, required=True)
    parser.add_argument("--oof-manifest", type=Path, default=REPO_ROOT / "data" / "splits" / "observation_oof_seed42_5fold.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--detection-thresholds", default="0.15,0.20,0.25,0.30,0.40")
    parser.add_argument("--grow-radii", default="0,4,8")
    parser.add_argument("--minimum-areas", default="100,200,400")
    parser.add_argument("--closing-radii", default="3")
    parser.add_argument("--solar-radius", type=int, default=950)
    parser.add_argument("--disk-geometry-json", type=Path, default=None,
                        help="optional image-derived {stem: {cx, cy, radius}} map")
    parser.add_argument("--disk-shrink", type=float, default=0.98)
    args = parser.parse_args()

    repo_root = args.repo_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve() if args.data_root else repo_root / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026"
    detector_path = args.detector_predictions.expanduser().resolve()
    unet_path = args.unet_predictions.expanduser().resolve()
    manifest_path = args.oof_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for label, path, allowed_root in (
        ("data", data_root, repo_root / "data"),
        ("detector predictions", detector_path, repo_root / "artifacts"),
        ("U-Net predictions", unet_path, repo_root / "artifacts"),
        ("OOF manifest", manifest_path, repo_root / "data" / "splits"),
        ("output", output_dir, repo_root / "artifacts"),
    ):
        if not path.is_relative_to(allowed_root):
            raise ValueError(f"{label} must remain under {allowed_root}: {path}")
    if not detector_path.is_file() or not unet_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Missing detector OOF, U-Net OOF, or split manifest")
    geometry_path = args.disk_geometry_json.expanduser().resolve() if args.disk_geometry_json else None
    if geometry_path is not None and not geometry_path.is_relative_to(repo_root / "artifacts"):
        raise ValueError(f"Disk geometry must remain under artifacts: {geometry_path}")
    if not 0.8 <= args.disk_shrink <= 1.05:
        raise ValueError("disk-shrink must be in [0.8, 1.05]")

    config = json.loads((repo_root / "configs" / "validation-baseline.json").read_text())
    dataset = load_dataset(data_root)
    frozen = create_observation_split(
        dataset,
        repo_root / "data" / "splits",
        seed=int(config["seed"]),
        validation_fraction=float(config["validation_fraction"]),
    )
    if frozen.sha256 != config["expected_split_sha256"]:
        raise RuntimeError(f"Frozen holdout digest changed: {frozen.sha256}")
    with manifest_path.open("r", encoding="utf-8", newline="") as stream:
        manifest_rows = list(csv.DictReader(stream))
    if not manifest_rows or not {"file_name", "year", "site", "oof_fold"}.issubset(manifest_rows[0]):
        raise ValueError("OOF manifest must include file_name,year,site,oof_fold")
    names = tuple(sorted(row["file_name"] for row in manifest_rows))
    if len(names) != len(set(names)) or set(names) != set(frozen.training_names):
        raise ValueError("OOF manifest must cover the frozen training partition exactly once")
    if set(names) & set(frozen.validation_names):
        raise ValueError("The fixed validation holdout must be excluded from OOF scoring")
    for row in manifest_rows:
        view = dataset.views_by_observation[row["file_name"]][0]
        if int(row["year"]) != view.year or row["site"] != view.site:
            raise ValueError(f"OOF observation metadata mismatch: {row['file_name']}")

    with detector_path.open("r", encoding="utf-8") as stream:
        detections: dict[str, list[dict[str, Any]]] = json.load(stream)
    stems = {Path(name).stem for name in names}
    if set(detections) != stems:
        raise ValueError("Detector OOF predictions do not exactly cover the OOF observations")
    unet_masks = _load_unet_masks(unet_path, stems)
    disk_geometries = json.loads(geometry_path.read_text(encoding="utf-8")) if geometry_path else None
    if disk_geometries is not None and not stems.issubset(disk_geometries):
        raise ValueError("Disk geometry is missing OOF observations")
    shape = (2048, 2048)
    packed_width = (shape[0] * shape[1] + 7) // 8
    output_dir.mkdir(parents=True, exist_ok=True)
    packed_path = output_dir / "unet-oof-foreground-packed.bin"
    packed_maps = np.memmap(packed_path, dtype=np.uint8, mode="w+", shape=(len(names), packed_width))
    baseline_predictions: dict[str, list[PredictedInstance]] = {}
    for index, file_name in enumerate(names):
        union = np.zeros(shape, dtype=np.uint8)
        stem = Path(file_name).stem
        individual_predictions = []
        for mask_index, counts in enumerate(unet_masks[stem], start=1):
            rle = {"size": list(shape), "counts": _counts(counts)}
            decoded = mask_utils.decode(rle)
            union |= decoded.astype(np.uint8)
            individual_predictions.append(
                PredictedInstance(
                    filament_id=f"{stem}_{mask_index}",
                    image_stem=stem,
                    area=int(mask_utils.area(rle)),
                    rle=rle,
                )
            )
        packed_maps[index] = np.packbits(union.reshape(-1), bitorder="big")
        baseline_predictions[file_name] = individual_predictions
        if (index + 1) % 50 == 0 or index + 1 == len(names):
            print(f"Packed {index + 1}/{len(names)} U-Net OOF observations", flush=True)
    packed_maps.flush()
    baseline_metrics, _ = score_validation(dataset, baseline_predictions, names)

    grid = itertools.product(
        [float(value) for value in args.detection_thresholds.split(",")],
        [int(value) for value in args.grow_radii.split(",")],
        [int(value) for value in args.minimum_areas.split(",")],
        [int(value) for value in args.closing_radii.split(",")],
    )
    sweep = []
    best = None
    best_predictions = None
    started = time.perf_counter()
    for candidate_index, values in enumerate(grid, start=1):
        fusion_config = DetectorFusionConfig(
            foreground_threshold=0.5,
            detection_threshold=values[0],
            grow_radius=values[1],
            minimum_claimed_area=values[2],
            closing_radius=values[3],
            solar_radius=args.solar_radius,
        )
        predictions: dict[str, list[PredictedInstance]] = {}
        for index, file_name in enumerate(names):
            foreground = np.unpackbits(
                np.asarray(packed_maps[index]), count=shape[0] * shape[1], bitorder="big"
            ).reshape(shape)
            labels = fuse_detector_instances(
                foreground,
                detections[Path(file_name).stem],
                fusion_config,
                disk_mask=(
                    solar_disk_mask_from_geometry(
                        shape, disk_geometries[Path(file_name).stem], shrink=args.disk_shrink
                    )
                    if disk_geometries is not None
                    else None
                ),
            )
            predictions[file_name] = _make_predictions(labels, file_name)
        metrics, _ = score_validation(dataset, predictions, names)
        record = {
            "configuration": asdict(fusion_config),
            "metrics": metrics,
            "prediction_count": sum(len(items) for items in predictions.values()),
        }
        sweep.append(record)
        if best is None or metrics["global"]["pq"] > best["metrics"]["global"]["pq"]:
            best = record
            best_predictions = predictions
        print(
            f"{candidate_index}: PQ={metrics['global']['pq']:.6f} "
            f"TP={metrics['global']['tp']} FP={metrics['global']['fp']} "
            f"FN={metrics['global']['fn']} config={asdict(fusion_config)}",
            flush=True,
        )

    if best is None or best_predictions is None:
        raise RuntimeError("No fusion configurations were evaluated")
    from filament_baseline.runner import _write_observation_scores, _write_predictions

    _write_predictions(output_dir / "best-oof-predictions.csv", best_predictions)
    _, per_observation = score_validation(dataset, best_predictions, names)
    _write_observation_scores(output_dir / "best-oof-per-observation.csv", per_observation, best_predictions)
    result = {
        "split_sha256": frozen.sha256,
        "oof_manifest": str(manifest_path.relative_to(repo_root)),
        "oof_manifest_sha256": _sha256(manifest_path),
        "oof_observations": len(names),
        "oof_annotator_views": sum(len(dataset.views_by_observation[name]) for name in names),
        "ground_truth_source": "competition-provided training annotations only",
        "detector_predictions": str(detector_path.relative_to(repo_root)),
        "detector_predictions_sha256": _sha256(detector_path),
        "disk_geometry": str(geometry_path.relative_to(repo_root)) if geometry_path else None,
        "disk_geometry_sha256": _sha256(geometry_path) if geometry_path else None,
        "disk_shrink": args.disk_shrink if geometry_path else None,
        "unet_oof_predictions": str(unet_path.relative_to(repo_root)),
        "unet_oof_predictions_sha256": _sha256(unet_path),
        "unet_baseline_metrics": baseline_metrics,
        "best": best,
        "sweep_seconds": time.perf_counter() - started,
        "sweep": sweep,
    }
    (output_dir / "fusion-oof-sweep.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    packed_maps.flush()
    print(
        f"U-Net OOF PQ={baseline_metrics['global']['pq']:.6f}; "
        f"best fused OOF PQ={best['metrics']['global']['pq']:.6f}; output={output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
