# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import hashlib
import csv
import itertools
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from filament_baseline.data import load_dataset
from filament_baseline.detector_fusion import (
    DetectorFusionConfig,
    fuse_detector_instances,
    solar_disk_mask_from_geometry,
)
from filament_baseline.inference import InstanceConfig, instances_from_probabilities
from filament_baseline.predictor import PredictedInstance, encode_binary_mask
from filament_baseline.runner import _write_observation_scores, _write_overlays, _write_predictions
from filament_baseline.scorer import score_validation
from filament_baseline.split import create_observation_split
from filament_baseline.visuals import write_comparison_overlays


def _prediction_instances(labels: np.ndarray, file_name: str) -> list[PredictedInstance]:
    output = []
    for label in range(1, int(labels.max()) + 1):
        binary = labels == label
        area = int(binary.sum())
        if area == 0:
            continue
        output.append(
            PredictedInstance(
                filament_id=f"{Path(file_name).stem}_{len(output) + 1}",
                image_stem=Path(file_name).stem,
                area=area,
                rle=encode_binary_mask(binary),
            )
        )
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep detector-guided U-Net fusion with the frozen organizer-aligned scorer."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--detector-predictions", type=Path, required=True)
    parser.add_argument("--probability-dir", type=Path, required=True)
    parser.add_argument("--second-probability-dir", type=Path, default=None,
                        help="optional second U-Net cache; maps are probability-averaged with the first cache")
    parser.add_argument("--first-probability-weight", type=float, default=0.5,
                        help="weight for --probability-dir when a second cache is supplied")
    parser.add_argument("--disk-geometry-json", type=Path, default=None,
                        help="optional image-derived {stem: {cx, cy, radius}} map")
    parser.add_argument("--disk-shrink", type=float, default=0.98,
                        help="radius multiplier for --disk-geometry-json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--foreground-thresholds", default="0.45,0.50,0.55")
    parser.add_argument("--detection-thresholds", default="0.10,0.20,0.30,0.40")
    parser.add_argument("--grow-radii", default="0,4,8")
    parser.add_argument("--minimum-areas", default="50,200,400")
    parser.add_argument("--closing-radii", default="3")
    parser.add_argument("--keep-unclaimed-areas", default="0")
    parser.add_argument("--baseline-foreground-threshold", type=float, default=0.65)
    parser.add_argument("--baseline-boundary-threshold", type=float, default=0.70)
    parser.add_argument("--baseline-boundary-weight", type=float, default=1.5)
    parser.add_argument("--baseline-seed-threshold", type=float, default=0.45)
    parser.add_argument("--baseline-seed-min-area", type=int, default=8)
    parser.add_argument("--baseline-minimum-area", type=int, default=400)
    parser.add_argument("--overlay-count", type=int, default=8)
    args = parser.parse_args()

    repo_root = args.repo_root.expanduser().resolve()
    data_root = (
        args.data_root.expanduser().resolve()
        if args.data_root
        else repo_root / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026"
    )
    detections_path = args.detector_predictions.expanduser().resolve()
    probability_dir = args.probability_dir.expanduser().resolve()
    second_probability_dir = (
        args.second_probability_dir.expanduser().resolve()
        if args.second_probability_dir
        else None
    )
    output_dir = args.output_dir.expanduser().resolve()
    for label, path in (("data", data_root), ("detector predictions", detections_path),
                        ("probability cache", probability_dir), ("output", output_dir)):
        if not path.is_relative_to(repo_root):
            raise ValueError(f"Local {label} must remain inside the repository: {path}")
    if not detections_path.is_file() or not probability_dir.is_dir():
        raise FileNotFoundError("Missing detector predictions or U-Net probability cache")
    if second_probability_dir is not None:
        if not second_probability_dir.is_dir():
            raise FileNotFoundError(f"Missing second U-Net probability cache: {second_probability_dir}")
        if not second_probability_dir.is_relative_to(repo_root):
            raise ValueError(f"Second probability cache must remain inside the repository: {second_probability_dir}")
        if not 0.0 <= args.first_probability_weight <= 1.0:
            raise ValueError("first-probability-weight must be between zero and one")
    geometry_path = args.disk_geometry_json.expanduser().resolve() if args.disk_geometry_json else None
    if geometry_path is not None and not geometry_path.is_relative_to(repo_root / "artifacts"):
        raise ValueError(f"Disk geometry must remain under artifacts: {geometry_path}")
    if not 0.8 <= args.disk_shrink <= 1.05:
        raise ValueError("disk-shrink must be in [0.8, 1.05]")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((repo_root / "configs" / "validation-baseline.json").read_text())
    dataset = load_dataset(data_root)
    split = create_observation_split(
        dataset,
        repo_root / "data" / "splits",
        seed=int(config["seed"]),
        validation_fraction=float(config["validation_fraction"]),
    )
    if split.sha256 != config["expected_split_sha256"]:
        raise RuntimeError(f"Frozen split changed: {split.sha256}")
    with detections_path.open("r", encoding="utf-8") as stream:
        detector_predictions = json.load(stream)
    disk_geometries = None
    if geometry_path is not None:
        disk_geometries = json.loads(geometry_path.read_text(encoding="utf-8"))

    def image_disk_mask(file_name: str) -> np.ndarray | None:
        if disk_geometries is None:
            return None
        stem = Path(file_name).stem
        if stem not in disk_geometries:
            raise KeyError(f"No disk geometry for {stem}")
        return solar_disk_mask_from_geometry(
            (2048, 2048), disk_geometries[stem], shrink=args.disk_shrink
        )

    names = split.validation_names
    probability_maps = {}
    baseline_predictions = {}
    baseline_config = InstanceConfig(
        foreground_threshold=args.baseline_foreground_threshold,
        boundary_threshold=args.baseline_boundary_threshold,
        boundary_weight=args.baseline_boundary_weight,
        seed_threshold=args.baseline_seed_threshold,
        seed_min_area=args.baseline_seed_min_area,
        min_instance_area=args.baseline_minimum_area,
        solar_radius=950,
    )
    for index, file_name in enumerate(names, start=1):
        cache_path = probability_dir / f"{Path(file_name).stem}.npz"
        if not cache_path.is_file():
            raise FileNotFoundError(f"Missing U-Net probabilities: {cache_path}")
        with np.load(cache_path) as cache:
            foreground = cache["foreground"].astype(np.float32)
            boundary = cache["boundary"].astype(np.float32)
        if second_probability_dir is not None:
            second_cache_path = second_probability_dir / f"{Path(file_name).stem}.npz"
            if not second_cache_path.is_file():
                raise FileNotFoundError(f"Missing second U-Net probabilities: {second_cache_path}")
            with np.load(second_cache_path) as cache:
                second_foreground = cache["foreground"].astype(np.float32)
                second_boundary = cache["boundary"].astype(np.float32)
            if second_foreground.shape != foreground.shape or second_boundary.shape != boundary.shape:
                raise ValueError(f"Probability cache shape mismatch for {file_name}")
            second_weight = 1.0 - args.first_probability_weight
            foreground = args.first_probability_weight * foreground + second_weight * second_foreground
            boundary = args.first_probability_weight * boundary + second_weight * second_boundary
        probability_maps[file_name] = foreground
        baseline_predictions[file_name] = instances_from_probabilities(
            foreground,
            boundary,
            file_name,
            baseline_config,
            disk_mask=image_disk_mask(file_name),
        )
        if index % 25 == 0 or index == len(names):
            print(f"Loaded {index}/{len(names)} probability maps", flush=True)
    baseline_metrics, _ = score_validation(
        dataset,
        baseline_predictions,
        names,
        iou_threshold=float(config["iou_threshold"]),
        diagnostic_overlap_threshold=float(config["diagnostic_overlap_threshold"]),
    )

    grids = itertools.product(
        [float(x) for x in args.foreground_thresholds.split(",")],
        [float(x) for x in args.detection_thresholds.split(",")],
        [int(x) for x in args.grow_radii.split(",")],
        [int(x) for x in args.minimum_areas.split(",")],
        [int(x) for x in args.closing_radii.split(",")],
        [int(x) for x in args.keep_unclaimed_areas.split(",")],
    )
    sweep = []
    best = None
    best_predictions = None
    started = time.perf_counter()
    for index, values in enumerate(grids, start=1):
        fusion_config = DetectorFusionConfig(
            foreground_threshold=values[0],
            detection_threshold=values[1],
            grow_radius=values[2],
            minimum_claimed_area=values[3],
            closing_radius=values[4],
            keep_unclaimed_minimum_area=values[5],
            solar_radius=950,
        )
        predictions = {}
        for file_name in names:
            detections = detector_predictions.get(Path(file_name).stem, [])
            labels = fuse_detector_instances(
                probability_maps[file_name],
                detections,
                fusion_config,
                disk_mask=image_disk_mask(file_name),
            )
            predictions[file_name] = _prediction_instances(labels, file_name)
        metrics, rows = score_validation(
            dataset,
            predictions,
            names,
            iou_threshold=float(config["iou_threshold"]),
            diagnostic_overlap_threshold=float(config["diagnostic_overlap_threshold"]),
        )
        row = {
            "configuration": asdict(fusion_config),
            "metrics": metrics,
            "prediction_instances": sum(len(items) for items in predictions.values()),
        }
        sweep.append(row)
        if best is None or metrics["global"]["pq"] > best["metrics"]["global"]["pq"]:
            best = row
            best_predictions = predictions
        print(
            f"{index}: PQ={metrics['global']['pq']:.6f} "
            f"TP={metrics['global']['tp']} FP={metrics['global']['fp']} "
            f"FN={metrics['global']['fn']} config={asdict(fusion_config)}",
            flush=True,
        )

    result = {
        "split_sha256": split.sha256,
        "validation_observations": len(names),
        "validation_annotator_views": sum(
            len(dataset.views_by_observation[name]) for name in names
        ),
        "detector_predictions": str(detections_path.relative_to(repo_root)),
        "probability_dir": str(probability_dir.relative_to(repo_root)),
        "second_probability_dir": (
            str(second_probability_dir.relative_to(repo_root))
            if second_probability_dir is not None
            else None
        ),
        "first_probability_weight": (
            args.first_probability_weight if second_probability_dir is not None else 1.0
        ),
        "disk_geometry": str(geometry_path.relative_to(repo_root)) if geometry_path else None,
        "disk_geometry_sha256": _sha256(geometry_path) if geometry_path else None,
        "disk_shrink": args.disk_shrink if geometry_path else None,
        "baseline": {"configuration": asdict(baseline_config), "metrics": baseline_metrics},
        "best": best,
        "sweep": sweep,
        "sweep_seconds": time.perf_counter() - started,
    }
    (output_dir / "fusion-sweep.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    assert best_predictions is not None
    _write_predictions(output_dir / "predictions.csv", best_predictions)
    best_metrics, observation_rows = score_validation(
        dataset,
        best_predictions,
        names,
        iou_threshold=float(config["iou_threshold"]),
        diagnostic_overlap_threshold=float(config["diagnostic_overlap_threshold"]),
    )
    _write_observation_scores(
        output_dir / "per_observation.csv", observation_rows, best_predictions
    )
    _write_overlays(dataset, best_predictions, output_dir / "overlays", args.overlay_count)
    write_comparison_overlays(
        dataset, best_predictions, output_dir / "comparison-overlays", args.overlay_count
    )
    with (output_dir / "best-metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(best_metrics, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        f"Baseline PQ {baseline_metrics['global']['pq']:.6f}; "
        f"best fused PQ {best['metrics']['global']['pq']:.6f}; "
        f"results {output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
