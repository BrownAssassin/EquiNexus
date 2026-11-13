# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Evaluate the public ten-model U-Net using only fold-held-out checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from workspace_runtime import configure_workspace_runtime

configure_workspace_runtime(REPO_ROOT)

import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_utils

from evaluate_component_selector_probability_blend_holdout import paired_bootstrap
from filament_baseline.data import CompetitionDataset, load_dataset
from filament_baseline.predictor import PredictedInstance
from filament_baseline.runtime import hardware_info, package_versions
from filament_baseline.scorer import annotation_to_rle, score_validation
from filament_baseline.split import create_observation_split
from run_public_unet_ensemble import (
    DISK_RADIUS,
    IMAGE_SIZE,
    KEEP_RULE,
    disk_statistics,
    load_model,
    predict_map,
    prediction_rows,
    read_gray,
    sha256,
)

EXPECTED_SPLIT_SHA256 = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"
SOURCE_DATASET = "https://www.kaggle.com/datasets/dariushafshar/solar-filament-unet-weights"
SOURCE_INFERENCE_NOTEBOOK = "https://www.kaggle.com/code/huendao/solar-filament-segmentation-challenge-2026"
SOURCE_TRAINING_NOTEBOOK = "https://www.kaggle.com/code/dariushafshar/solar-filament-u-net-oof-0-45-vs-public-0-39"


def git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def load_fold_assignments(path: Path, dataset: CompetitionDataset) -> dict[str, int]:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["stem", "fold"]:
        raise ValueError("Upstream folds.csv must contain exactly stem,fold columns")
    if frame["stem"].astype(str).duplicated().any():
        raise ValueError("Upstream folds.csv contains duplicate observation stems")
    assignments = {str(row.stem): int(row.fold) for row in frame.itertuples(index=False)}
    dataset_stems = {Path(name).stem for name in dataset.observation_names}
    if set(assignments) != dataset_stems:
        missing = sorted(dataset_stems - set(assignments))[:5]
        extra = sorted(set(assignments) - dataset_stems)[:5]
        raise ValueError(f"Fold map does not cover the competition observations: missing={missing}, extra={extra}")
    if set(assignments.values()) != {0, 1, 2, 3, 4}:
        raise ValueError("Expected five upstream held-out folds numbered 0 through 4")
    return assignments


def render_overlays(
    dataset: CompetitionDataset,
    predictions: dict[str, list[dict[str, Any]]],
    observation_rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    import cv2
    from PIL import Image, ImageDraw

    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(observation_rows, key=lambda row: (row["pq"], row["file_name"]))
    selected = [
        ("low-pq", ranked[0]),
        ("median-pq", ranked[len(ranked) // 2]),
        ("high-pq", ranked[-1]),
    ]
    for label, row in selected:
        name = row["file_name"]
        stem = Path(name).stem
        image = read_gray(dataset.train_image_dir / name)
        lo, hi = np.percentile(image, (1, 99))
        gray = np.clip((image.astype(np.float32) - lo) * 220 / max(float(hi - lo), 1), 0, 220).astype(np.uint8)
        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        height, width = image.shape
        view = dataset.views_by_observation[name][0]
        for annotation in dataset.annotations_by_image_id[view.image_id]:
            rle = annotation_to_rle(annotation, height, width)
            mask = mask_utils.decode(rle).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, (30, 255, 50), 2)
        for item in predictions.get(name, []):
            rle = {"size": item["size"], "counts": item["counts"].encode("ascii")}
            mask = mask_utils.decode(rle).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, (255, 55, 50), 2)
        canvas = cv2.resize(canvas, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        panel = Image.fromarray(canvas)
        draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 0, 900, 48), fill=(0, 0, 0))
        draw.text((12, 8), f"{label}: {name} | first annotator view | observation PQ={row['pq']:.3f}", fill="white")
        draw.text((12, 27), "Green=annotator ground truth; red=prediction", fill=(230, 230, 230))
        panel.save(output_dir / f"{label}-{stem}.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--weights-root", type=Path, default=REPO_ROOT / "data" / "reference" / "dariushafshar-unet-weights")
    parser.add_argument("--split-dir", type=Path, default=REPO_ROOT / "data" / "splits")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "model-experiments" / "dariushafshar-unet-fold-aware-holdout-20261111")
    parser.add_argument("--incumbent-observations", type=Path, default=REPO_ROOT / "artifacts" / "model-experiments" / "crossfold-medoid-unet-fusion-20261109" / "incumbent-per-observation.csv")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tta-views", type=int, choices=(1, 2, 4, 8), default=8)
    parser.add_argument("--views-per-pass", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--partition", choices=("holdout", "oof"), default="holdout")
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    weights_root = args.weights_root.expanduser().resolve()
    split_dir = args.split_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    receipt_path = weights_root / "solar_unet_receipt.json"
    folds_path = weights_root / "folds.csv"
    if not receipt_path.is_file() or not folds_path.is_file():
        raise FileNotFoundError("Public weights receipt or fold manifest is missing")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {output_dir}")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    dataset = load_dataset(data_root)
    split = create_observation_split(dataset, split_dir, seed=42, validation_fraction=0.2)
    if split.sha256 != EXPECTED_SPLIT_SHA256:
        raise ValueError(f"Frozen split digest mismatch: {split.sha256}")
    if args.partition == "holdout":
        validation_names = list(sorted(split.validation_names))
        if set(validation_names).intersection(split.training_names):
            raise ValueError("Observation overlap between training and validation")
    else:
        validation_names = list(sorted(split.training_names))
        if set(validation_names) != set(split.training_names):
            raise ValueError("OOF evaluation must cover the frozen training partition")
        if set(validation_names).intersection(split.validation_names):
            raise ValueError("Frozen holdout observations cannot be included in OOF scoring")

    fold_assignments = load_fold_assignments(folds_path, dataset)
    fold_manifest_hash = sha256(folds_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
    upstream_rule = receipt["ensemble"]["keep_rule"]
    if {key: upstream_rule[key] for key in KEEP_RULE} != KEEP_RULE:
        raise ValueError("The shared inference rule differs from the upstream receipt")
    models = list(receipt["ensemble"]["models"])
    models_by_fold: dict[int, list[dict[str, Any]]] = {fold: [] for fold in range(5)}
    for entry in models:
        models_by_fold[int(entry["fold"])].append(entry)
    if any(
        len(entries) != 2
        or {str(entry["family"]) for entry in entries} != {"resnet34", "efficientnetv2_s"}
        for entries in models_by_fold.values()
    ):
        raise ValueError("Each held-out fold must have exactly one checkpoint from each encoder family")

    names_by_fold: dict[int, list[str]] = {fold: [] for fold in range(5)}
    for name in validation_names:
        names_by_fold[fold_assignments[Path(name).stem]].append(name)
    if sum(map(len, names_by_fold.values())) != len(validation_names):
        raise ValueError("Some frozen holdout observations were not assigned to an upstream fold")

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats(device)

    yy, xx = np.ogrid[:IMAGE_SIZE, :IMAGE_SIZE]
    disk = ((yy - (IMAGE_SIZE - 1) / 2.0) ** 2 + (xx - (IMAGE_SIZE - 1) / 2.0) ** 2) <= DISK_RADIUS**2
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions: dict[str, list[dict[str, Any]]] = {}
    model_records: list[dict[str, Any]] = []
    started = time.perf_counter()
    partition_description = "frozen holdout" if args.partition == "holdout" else "cross-fitted training partition"
    print(f"Scoring {len(validation_names)} {partition_description} observations with only their source fold checkpoints", flush=True)

    for fold in range(5):
        fold_names = sorted(names_by_fold[fold])
        if not fold_names:
            continue
        print(f"Fold {fold}: {len(fold_names)} {partition_description} observations", flush=True)
        inputs = {
            name: (read_gray(dataset.train_image_dir / name), None)
            for name in fold_names
        }
        for name, (image, _) in list(inputs.items()):
            inputs[name] = (image, disk_statistics(image, disk))
        probability_sums = {name: np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint16) for name in fold_names}
        for entry in sorted(models_by_fold[fold], key=lambda item: str(item["family"])):
            model_started = time.perf_counter()
            model = load_model(entry, weights_root, device)
            for index, name in enumerate(fold_names, start=1):
                image, (mean, std) = inputs[name]
                probability = predict_map(
                    model,
                    image,
                    mean,
                    std,
                    device,
                    tta_views=args.tta_views,
                    views_per_pass=args.views_per_pass,
                )
                probability_sums[name] += probability.astype(np.uint16)
                if index % 10 == 0 or index == len(fold_names):
                    print(f"  {entry['family']} fold {fold}: {index}/{len(fold_names)}", flush=True)
            model_records.append({
                "file": entry["file"],
                "family": entry["family"],
                "fold": fold,
                "sha256": entry["sha256"],
                "seconds": round(time.perf_counter() - model_started, 3),
                "validation_observations": len(fold_names),
                "views_per_image": args.tta_views,
            })
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        for name in fold_names:
            mean_probability = np.rint(probability_sums[name].astype(np.float32) / len(models_by_fold[fold]))
            mean_probability = np.clip(mean_probability, 0, 255).astype(np.uint8)
            rows = prediction_rows(mean_probability, Path(name).stem)
            predictions[name] = [
                {"filament_id": filament_id, "size": [IMAGE_SIZE, IMAGE_SIZE], "counts": counts}
                for filament_id, counts in rows
            ]
        del inputs, probability_sums

    inference_seconds = time.perf_counter() - started
    output_prefix = "validation" if args.partition == "holdout" else "oof"
    predictions_path = output_dir / f"{output_prefix}_predictions.json"
    predictions_path.write_text(json.dumps(predictions, separators=(",", ":")), encoding="utf-8")
    prediction_rows_by_name: dict[str, list[PredictedInstance]] = {}
    csv_path = output_dir / f"{output_prefix}_predictions.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("filament_id", "segmentation_rle"))
        writer.writeheader()
        for name in validation_names:
            retained: list[PredictedInstance] = []
            for item in predictions[name]:
                rle = {"size": item["size"], "counts": item["counts"].encode("ascii")}
                area = int(mask_utils.area(rle))
                retained.append(PredictedInstance(item["filament_id"], Path(name).stem, area, rle))
                writer.writerow({"filament_id": item["filament_id"], "segmentation_rle": item["counts"]})
            prediction_rows_by_name[name] = retained

    metrics, observation_counts = score_validation(
        dataset, prediction_rows_by_name, validation_names,
        iou_threshold=0.5, diagnostic_overlap_threshold=0.0,
    )
    per_observation = [
        {
            "file_name": row.file_name,
            "year": row.year,
            "site": row.site,
            "annotator_views": row.annotator_views,
            "gt_count": row.gt_count,
            "prediction_count": row.prediction_count,
            "tp": row.tp,
            "tp_iou_sum": row.tp_iou_sum,
            "fp": row.fp,
            "fn": row.fn,
            "pq": row.pq,
        }
        for row in observation_counts
    ]
    render_overlays(dataset, predictions, per_observation, output_dir / "overlays")

    comparison = None
    incumbent_path = args.incumbent_observations.expanduser().resolve()
    if args.partition == "holdout" and incumbent_path.is_file():
        incumbent = pd.read_csv(incumbent_path)
        baseline_rows = [
            SimpleNamespace(file_name=row.file_name, tp_iou_sum=row.tp_iou_sum, tp=row.tp_edges, fp=row.fp, fn=row.fn)
            for row in incumbent.itertuples(index=False)
        ]
        candidate_rows = [
            SimpleNamespace(file_name=row["file_name"], tp_iou_sum=row["tp_iou_sum"], tp=row["tp"], fp=row["fp"], fn=row["fn"])
            for row in per_observation
        ]
        comparison = {
            "incumbent_per_observation_sha256": sha256(incumbent_path),
            "incumbent_per_observation_path": str(incumbent_path),
            "paired_bootstrap": paired_bootstrap(baseline_rows, candidate_rows, resamples=20_000, seed=42),
        }

    elapsed_seconds = time.perf_counter() - started
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    metadata = {
        "run_name": output_dir.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": SOURCE_DATASET,
        "source_training_notebook": SOURCE_TRAINING_NOTEBOOK,
        "source_inference_notebook": SOURCE_INFERENCE_NOTEBOOK,
        "source_license": "CC BY-NC-SA 4.0; attribution and ShareAlike terms apply",
        "source_training_provenance": "The dataset README states the checkpoints were trained only on competition training observations and labels, grouped into five folds by image; fold k checkpoint did not train on fold k images.",
        "evaluation_partition": "per-observation five-fold crossfit over the frozen holdout subset; not an independent holdout" if args.partition == "holdout" else "per-observation five-fold crossfit over the frozen training subset",
        "validation_protocol": "Each scored observation uses only the two checkpoints for its upstream held-out fold. Because upstream folds partition all 707 observations, other observations from this evaluation subset may be used to train other fold models; interpret aggregate metrics as crossfit, not independent holdout.",
        "target_observation_used_for_gradient_updates_by_its_predictor": False,
        "evaluation_peers_used_for_gradient_updates_in_other_fold_models": True,
        "strict_independent_frozen_holdout": False,
        "holdout_used_for_checkpoint_selection": False,
        "holdout_used_for_threshold_selection": False,
        "threshold_selection_source": "Upstream nested cross-validation; the keep rule is fixed and unchanged.",
        "test_ground_truth_used": False,
        "test_images_used": False,
        "ground_truth_source": "Competition-provided training annotations only",
        "training_observations": len(split.training_names),
        "validation_observations": len(validation_names),
        "evaluation_observations": len(validation_names),
        "validation_annotator_views": len({view.image_id for name in validation_names for view in dataset.views_by_observation[name]}),
        "competition_split_sha256": split.sha256,
        "upstream_fold_manifest": str(folds_path),
        "upstream_fold_manifest_sha256": fold_manifest_hash,
        "evaluation_observations_by_upstream_fold": {str(fold): len(names_by_fold[fold]) for fold in range(5)},
        "upstream_fold_exclusion_claim": "Each validation observation is predicted only by the two public checkpoints corresponding to the fold listed for it in folds.csv.",
        "inference_config": {"tta_views": args.tta_views, "views_per_pass": args.views_per_pass, "disk_radius": DISK_RADIUS, "image_size": IMAGE_SIZE, "keep_rule": KEEP_RULE, "models_per_observation": 2},
        "device": str(device),
        "hardware": hardware_info(),
        "package_versions": package_versions(),
        "python": sys.version,
        "inference_seconds": inference_seconds,
        "total_seconds": elapsed_seconds,
        "peak_gpu_memory_allocated_bytes": peak_memory,
        "model_records": model_records,
        "metrics": metrics,
        "per_observation": per_observation,
        "comparison_vs_incumbent": comparison,
        "comparison_interpretation": "Exploratory only: the public candidate is per-observation crossfit, while the incumbent was trained without any frozen-holdout groups; this is not an independent-holdout promotion gate.",
        "predictions": {"json": str(predictions_path), "json_sha256": sha256(predictions_path), "csv": str(csv_path), "csv_sha256": sha256(csv_path)},
        "runner_sha256": sha256(Path(__file__).resolve()),
        "source_commit": git_value("rev-parse", "HEAD"),
        "source_worktree_status": git_value("status", "--short"),
    }
    metadata_path = output_dir / "run-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "event": "complete",
        "official_pq": metrics["global"]["pq"],
        "tp": metrics["global"]["tp"],
        "fp": metrics["global"]["fp"],
        "fn": metrics["global"]["fn"],
        "inference_seconds": inference_seconds,
        "peak_gpu_memory_allocated_bytes": peak_memory,
        "comparison_vs_incumbent": comparison,
        "comparison_interpretation": "Exploratory only: the public candidate is per-observation crossfit, while the incumbent was trained without any frozen-holdout groups; this is not an independent-holdout promotion gate.",
        "metadata": str(metadata_path),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
