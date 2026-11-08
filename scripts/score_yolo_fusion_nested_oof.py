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
)
from filament_baseline.predictor import PredictedInstance, encode_binary_mask
from filament_baseline.runner import _write_observation_scores, _write_predictions
from filament_baseline.scorer import score_validation
from filament_baseline.split import create_observation_split


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_masks(path: Path, stems: set[str]) -> dict[str, list[str]]:
    masks: dict[str, list[str]] = {stem: [] for stem in stems}
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["filament_id", "segmentation_rle"]:
            raise ValueError("U-Net predictions must contain filament_id,segmentation_rle")
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


def _pq(metrics: dict[str, Any]) -> float:
    denominator = metrics["tp"] + 0.5 * metrics["fp"] + 0.5 * metrics["fn"]
    return metrics["tp_iou_sum"] / denominator if denominator else 0.0


def _sum_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("tp", "fp", "fn", "tp_iou_sum", "gt_count_across_annotator_views", "prediction_count_across_annotator_views", "annotator_view_count")
    result = {key: sum(row[key] for row in rows) for key in keys}
    result["pq"] = _pq(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Nested five-fold selection of YOLO/U-Net fusion postprocessing."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--detector-predictions", type=Path, required=True)
    parser.add_argument("--unet-predictions", type=Path, required=True)
    parser.add_argument("--oof-manifest", type=Path, default=REPO_ROOT / "data" / "splits" / "observation_oof_seed42_5fold.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--detection-thresholds", default="0.20,0.25,0.30")
    parser.add_argument("--minimum-areas", default="200,400")
    parser.add_argument("--grow-radii", default="0")
    parser.add_argument("--closing-radius", type=int, default=3)
    parser.add_argument("--foreground-threshold", type=float, default=0.5)
    parser.add_argument("--solar-radius", type=int, default=950)
    args = parser.parse_args()

    repo_root = args.repo_root.expanduser().resolve()
    data_root = (args.data_root or repo_root / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026").expanduser().resolve()
    detector_path = args.detector_predictions.expanduser().resolve()
    unet_path = args.unet_predictions.expanduser().resolve()
    manifest_path = args.oof_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    allowed = (
        ("data", data_root, repo_root / "data"),
        ("detector predictions", detector_path, repo_root / "artifacts"),
        ("U-Net predictions", unet_path, repo_root / "artifacts"),
        ("OOF manifest", manifest_path, repo_root / "data" / "splits"),
        ("output", output_dir, repo_root / "artifacts"),
    )
    for label, path, root in allowed:
        if not path.is_relative_to(root):
            raise ValueError(f"{label} must remain under {root}: {path}")
    if not all(path.is_file() for path in (detector_path, unet_path, manifest_path)):
        raise FileNotFoundError("Missing detector predictions, U-Net predictions, or OOF manifest")

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
    oof_manifest_sha256 = _sha256(manifest_path)
    if oof_manifest_sha256 != "f856708222f5747cd35760932f535e4ae2752fd5161b9cd5981fbf4bbde26a3c":
        raise RuntimeError(f"OOF manifest digest changed: {oof_manifest_sha256}")
    names = tuple(sorted(row["file_name"] for row in manifest_rows))
    if len(names) != len(set(names)) or set(names) != set(frozen.training_names):
        raise ValueError("OOF manifest must cover exactly the 565-observation training partition")
    fold_by_name = {row["file_name"]: int(row["oof_fold"]) for row in manifest_rows}
    if set(fold_by_name.values()) != {0, 1, 2, 3, 4}:
        raise ValueError("Expected exactly five OOF folds")
    for row in manifest_rows:
        view = dataset.views_by_observation[row["file_name"]][0]
        if int(row["year"]) != view.year or row["site"] != view.site:
            raise ValueError(f"OOF metadata changed for {row['file_name']}")

    with detector_path.open("r", encoding="utf-8") as stream:
        detections: dict[str, list[dict[str, Any]]] = json.load(stream)
    stems = {Path(name).stem for name in names}
    if set(detections) != stems:
        raise ValueError("Detector predictions must exactly cover all OOF observations")
    unet_masks = _load_masks(unet_path, stems)

    shape = (2048, 2048)
    packed_width = (shape[0] * shape[1] + 7) // 8
    output_dir.mkdir(parents=True, exist_ok=True)
    packed_path = output_dir / "unet-foreground-packed.bin"
    packed_maps = np.memmap(packed_path, dtype=np.uint8, mode="w+", shape=(len(names), packed_width))
    unet_predictions: dict[str, list[PredictedInstance]] = {}
    for index, file_name in enumerate(names):
        stem = Path(file_name).stem
        union = np.zeros(shape, dtype=np.uint8)
        individual = []
        for mask_index, counts in enumerate(unet_masks[stem], start=1):
            rle = {"size": list(shape), "counts": counts.encode("ascii")}
            decoded = mask_utils.decode(rle)
            union |= decoded.astype(np.uint8)
            individual.append(
                PredictedInstance(
                    filament_id=f"{stem}_{mask_index}",
                    image_stem=stem,
                    area=int(mask_utils.area(rle)),
                    rle=rle,
                )
            )
        packed_maps[index] = np.packbits(union.reshape(-1), bitorder="big")
        unet_predictions[file_name] = individual
    packed_maps.flush()

    fold_names = {
        fold: tuple(name for name in names if fold_by_name[name] == fold)
        for fold in range(5)
    }
    unet_fold_metrics = {
        fold: score_validation(dataset, unet_predictions, subset)[0]["global"]
        for fold, subset in fold_names.items()
    }
    candidate_grid = list(
        itertools.product(
            [float(value) for value in args.detection_thresholds.split(",")],
            [int(value) for value in args.minimum_areas.split(",")],
            [int(value) for value in args.grow_radii.split(",")],
        )
    )
    if not candidate_grid:
        raise ValueError("At least one postprocessing configuration is required")

    best_inner_pq = {fold: float("-inf") for fold in range(5)}
    selected_configs: dict[int, dict[str, Any]] = {}
    nested_predictions: dict[str, list[PredictedInstance]] = {}
    candidate_results = []
    global_best: dict[str, Any] | None = None
    started = time.perf_counter()
    for index, (confidence, area, grow) in enumerate(candidate_grid, start=1):
        fusion_config = DetectorFusionConfig(
            foreground_threshold=args.foreground_threshold,
            detection_threshold=confidence,
            grow_radius=grow,
            minimum_claimed_area=area,
            closing_radius=args.closing_radius,
            solar_radius=args.solar_radius,
        )
        predictions: dict[str, list[PredictedInstance]] = {}
        for row_index, file_name in enumerate(names):
            foreground = np.unpackbits(
                np.asarray(packed_maps[row_index]), count=shape[0] * shape[1], bitorder="big"
            ).reshape(shape)
            labels = fuse_detector_instances(
                foreground,
                detections[Path(file_name).stem],
                fusion_config,
            )
            predictions[file_name] = _make_predictions(labels, file_name)

        fold_metrics = {
            fold: score_validation(dataset, predictions, subset)[0]["global"]
            for fold, subset in fold_names.items()
        }
        global_metrics = _sum_metrics(list(fold_metrics.values()))
        candidate = {
            "configuration": asdict(fusion_config),
            "global_oof": global_metrics,
            "per_fold": fold_metrics,
        }
        candidate_results.append(candidate)
        if global_best is None or global_metrics["pq"] > global_best["global_oof"]["pq"]:
            global_best = candidate

        for outer_fold in range(5):
            inner = _sum_metrics(
                [metrics for fold, metrics in fold_metrics.items() if fold != outer_fold]
            )
            if inner["pq"] > best_inner_pq[outer_fold]:
                best_inner_pq[outer_fold] = inner["pq"]
                selected_configs[outer_fold] = {
                    "configuration": asdict(fusion_config),
                    "inner_pq": inner["pq"],
                }
                for name in fold_names[outer_fold]:
                    nested_predictions[name] = predictions[name]
        fold_pq = ",".join(f"{fold_metrics[fold]['pq']:.4f}" for fold in range(5))
        print(
            f"{index}/{len(candidate_grid)} config={asdict(fusion_config)} "
            f"global_oof_PQ={global_metrics['pq']:.6f}; "
            f"fold_PQ={fold_pq}",
            flush=True,
        )

    if global_best is None or set(nested_predictions) != set(names):
        raise RuntimeError("Nested OOF selection did not cover every observation")
    nested_metrics, nested_per_observation = score_validation(dataset, nested_predictions, names)
    selected_outer_metrics = {
        fold: score_validation(dataset, nested_predictions, subset)[0]["global"]
        for fold, subset in fold_names.items()
    }
    _write_predictions(output_dir / "nested-oof-predictions.csv", nested_predictions)
    _write_observation_scores(
        output_dir / "nested-oof-per-observation.csv",
        nested_per_observation,
        nested_predictions,
    )
    result = {
        "status": "complete",
        "method": "Each outer OOF fold uses postprocessing selected on the other four folds.",
        "frozen_split_sha256": frozen.sha256,
        "oof_manifest_sha256": oof_manifest_sha256,
        "detector_predictions_sha256": _sha256(detector_path),
        "unet_predictions_sha256": _sha256(unet_path),
        "oof_observations": len(names),
        "oof_annotator_views": sum(len(dataset.views_by_observation[name]) for name in names),
        "candidate_count": len(candidate_grid),
        "elapsed_seconds": time.perf_counter() - started,
        "unet_only_global_oof": _sum_metrics(list(unet_fold_metrics.values())),
        "global_best_configuration": global_best,
        "nested_configuration_by_outer_fold": {
            fold: {
                **selected_configs[fold],
                "outer_metrics": selected_outer_metrics[fold],
            }
            for fold in range(5)
        },
        "nested_oof_metrics": nested_metrics["global"],
        "global_six_config_results": candidate_results,
    }
    (output_dir / "nested-oof-metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "nested_oof_pq": nested_metrics["global"]["pq"],
                "global_best_pq": global_best["global_oof"]["pq"],
                "unet_only_pq": result["unet_only_global_oof"]["pq"],
                "outer_fold_pq": {fold: selected_outer_metrics[fold]["pq"] for fold in range(5)},
                "elapsed_seconds": result["elapsed_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
