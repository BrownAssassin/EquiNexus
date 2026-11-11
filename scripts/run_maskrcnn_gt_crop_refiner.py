# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Train a crop U-Net from grouped training instances and refine Mask R-CNN masks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TORCH_HOME", str(REPO_ROOT / ".cache" / "torch"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from torch import nn
from torch.utils.data import DataLoader, Dataset

from filament_baseline.data import CompetitionDataset, ImageView, inventory, load_dataset
from filament_baseline.external_panoptic import CropRefiner, _refiner_probabilities, _square_crop
from filament_baseline.predictor import (
    PredictedInstance,
    encode_cropped_binary_mask,
    read_grayscale,
)
from filament_baseline.runtime import hardware_info, package_versions
from filament_baseline.scorer import annotation_to_rle, score_validation
from filament_baseline.split import create_observation_split
from filament_baseline.visuals import write_comparison_overlays
from run_maskrcnn_experiment import (
    merge_positive_overlaps,
    select_one_view_per_observation,
)
from score_detector_json import score_detector_predictions


FROZEN_SPLIT_SHA256 = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"
IMAGE_SHAPE = (2048, 2048)
CROP_SIZE = 256
REFINER_THRESHOLDS = (0.35, 0.50, 0.65)
DETECTOR_THRESHOLDS = (0.40, 0.50, 0.60)
MINIMUM_AREAS = (0, 200, 600)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str | None:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def seed_worker(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


class CropMaskDataset(Dataset):
    def __init__(self, image_path: Path, mask_path: Path, augment: bool) -> None:
        self.images = np.load(image_path, mmap_mode="r")
        self.masks = np.load(mask_path, mmap_mode="r")
        self.augment = augment
        if self.images.shape != self.masks.shape or self.images.ndim != 3:
            raise ValueError("Crop image and mask caches must have the same N x H x W shape")

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = np.array(self.images[index], dtype=np.uint8, copy=True)
        target = np.array(self.masks[index], dtype=np.uint8, copy=True)
        if self.augment:
            rotations = int(np.random.randint(0, 4))
            image = np.rot90(image, rotations)
            target = np.rot90(target, rotations)
            if random.random() < 0.5:
                image = np.flip(image, axis=1)
                target = np.flip(target, axis=1)
            intensity = image.astype(np.float32) / 255.0
            if random.random() < 0.5:
                intensity = intensity ** random.uniform(0.80, 1.25)
            if random.random() < 0.5:
                intensity = np.clip(
                    intensity * random.uniform(0.90, 1.10), 0.0, 1.0
                )
            image = np.clip(intensity * 255.0, 0, 255).astype(np.uint8)
        image_tensor = (
            torch.from_numpy(np.ascontiguousarray(image))
            .float()
            .div_(255.0)
            .unsqueeze(0)
            .repeat(3, 1, 1)
        )
        mask_tensor = torch.from_numpy(np.ascontiguousarray(target)).float().unsqueeze(0)
        return image_tensor, mask_tensor


def _decode_annotation_mask(
    annotation: dict[str, Any], view: ImageView
) -> np.ndarray:
    rle = annotation_to_rle(annotation, view.height, view.width)
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("ascii")
    return mask_utils.decode(rle).astype(np.uint8, copy=False)


def build_training_crop_cache(
    dataset: CompetitionDataset,
    training_views: tuple[ImageView, ...],
    cache_dir: Path,
) -> tuple[Path, Path, Path, int]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    image_path = cache_dir / "train-images-256.npy"
    mask_path = cache_dir / "train-masks-256.npy"
    manifest_path = cache_dir / "train-crop-index.csv"
    if any(path.exists() for path in (image_path, mask_path, manifest_path)):
        raise FileExistsError(f"Crop cache already exists: {cache_dir}")
    count = sum(
        len(dataset.annotations_by_image_id[view.image_id]) for view in training_views
    )
    images = np.lib.format.open_memmap(
        image_path, mode="w+", dtype=np.uint8, shape=(count, CROP_SIZE, CROP_SIZE)
    )
    masks = np.lib.format.open_memmap(
        mask_path, mode="w+", dtype=np.uint8, shape=(count, CROP_SIZE, CROP_SIZE)
    )
    position = 0
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("crop_index", "file_name", "image_id", "annotation_id", "bounds"),
        )
        writer.writeheader()
        for view in training_views:
            image = read_grayscale(dataset.train_image_dir / view.file_name)
            annotations = dataset.annotations_by_image_id[view.image_id]
            for annotation in annotations:
                target = _decode_annotation_mask(annotation, view)
                if not np.any(target):
                    continue
                x0, y0, x1, y1 = _square_crop(target.astype(bool), context=1.8, minimum=96)
                image_crop = image[y0:y1, x0:x1]
                target_crop = target[y0:y1, x0:x1]
                images[position] = cv2.resize(
                    image_crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR
                )
                masks[position] = cv2.resize(
                    target_crop,
                    (CROP_SIZE, CROP_SIZE),
                    interpolation=cv2.INTER_NEAREST,
                )
                writer.writerow(
                    {
                        "crop_index": position,
                        "file_name": view.file_name,
                        "image_id": view.image_id,
                        "annotation_id": annotation["id"],
                        "bounds": f"{x0},{y0},{x1},{y1}",
                    }
                )
                position += 1
    images.flush()
    masks.flush()
    del images, masks
    if position != count:
        raise RuntimeError(f"Expected {count} crops, wrote {position}")
    return image_path, mask_path, manifest_path, count


def loss_fn(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    probabilities = logits.sigmoid()
    intersection = (probabilities * target).sum((1, 2, 3))
    denominator = probabilities.sum((1, 2, 3)) + target.sum((1, 2, 3))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return 0.5 * bce + 0.5 * dice, dice


def load_detection_payload(path: Path, validation_names: tuple[str, ...]) -> dict[str, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {Path(name).stem for name in validation_names}
    if not expected.issubset(payload):
        missing = sorted(expected - set(payload))
        raise ValueError(f"Detector predictions are missing holdout observations: {missing[:5]}")
    return payload


def predict_refiner_probabilities(
    model: nn.Module,
    dataset: CompetitionDataset,
    validation_names: tuple[str, ...],
    detector_predictions: dict[str, list[dict[str, Any]]],
    device: torch.device,
    *,
    crop_size: int,
    minimum_detector_score: float = 0.05,
    batch_size: int = 24,
) -> dict[str, Any]:
    crops: list[np.ndarray] = []
    records: list[tuple[str, float, tuple[int, int, int, int]]] = []
    for file_name in validation_names:
        stem = Path(file_name).stem
        image = read_grayscale(dataset.train_image_dir / file_name)
        for item in detector_predictions.get(stem, []):
            score = float(item["score"])
            if score < minimum_detector_score:
                continue
            rle = {"size": list(IMAGE_SHAPE), "counts": item["counts"].encode("ascii")}
            mask = mask_utils.decode(rle).astype(bool)
            if not np.any(mask):
                continue
            bounds = _square_crop(mask, context=1.8, minimum=96)
            x0, y0, x1, y1 = bounds
            crops.append(
                cv2.resize(
                    image[y0:y1, x0:x1],
                    (crop_size, crop_size),
                    interpolation=cv2.INTER_LINEAR,
                )
            )
            records.append((stem, score, bounds))
    if not crops:
        raise RuntimeError("No detector masks were available to refine")
    crop_array = np.stack(crops).astype(np.uint8, copy=False)
    probabilities = _refiner_probabilities(
        model, crop_array, device, batch_size=batch_size
    )
    if probabilities.shape != (len(records), crop_size, crop_size):
        raise RuntimeError(f"Unexpected refiner probability shape: {probabilities.shape}")
    refined = {Path(name).stem: [] for name in validation_names}
    for (stem, score, bounds), probability in zip(records, probabilities, strict=True):
        refined[stem].append(
            {"score": score, "bounds": bounds, "probability": probability}
        )
    return refined


def encode_refined_predictions(
    refined_probabilities: dict[str, list[dict[str, Any]]],
    threshold: float,
    image_size: int = 2048,
) -> dict[str, list[dict[str, Any]]]:
    predictions: dict[str, list[dict[str, Any]]] = {}
    for stem, items in refined_probabilities.items():
        encoded_items: list[dict[str, Any]] = []
        for item in items:
            x0, y0, x1, y1 = item["bounds"]
            native_probability = cv2.resize(
                item["probability"],
                (x1 - x0, y1 - y0),
                interpolation=cv2.INTER_LINEAR,
            )
            binary = native_probability >= threshold
            if not np.any(binary):
                continue
            rle = encode_cropped_binary_mask(
                binary, x0, y0, (image_size, image_size)
            )
            encoded_items.append({"score": float(item["score"]), "counts": rle["counts"]})
        predictions[stem] = encoded_items
    return predictions


def score_refined_epoch(
    dataset: CompetitionDataset,
    validation_names: tuple[str, ...],
    refined_probabilities: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    sweep: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_predictions: dict[str, list[dict[str, Any]]] | None = None
    for refiner_threshold in REFINER_THRESHOLDS:
        raw = encode_refined_predictions(refined_probabilities, refiner_threshold)
        for merge_mode, predictions in (
            ("raw", raw),
            ("positive_iou_merge", {
                stem: merge_positive_overlaps(items) for stem, items in raw.items()
            }),
        ):
            for confidence in DETECTOR_THRESHOLDS:
                for minimum_area in MINIMUM_AREAS:
                    metrics, _, prediction_count = score_detector_predictions(
                        dataset,
                        validation_names,
                        predictions,
                        confidence_threshold=confidence,
                        minimum_area=minimum_area,
                    )
                    row = {
                        "refiner_threshold": refiner_threshold,
                        "merge_mode": merge_mode,
                        "confidence_threshold": confidence,
                        "minimum_area": minimum_area,
                        "prediction_count": prediction_count,
                        "metrics": metrics["global"],
                    }
                    sweep.append(row)
                    if best is None or row["metrics"]["pq"] > best["metrics"]["pq"]:
                        best = row
                        best_predictions = predictions
    if best is None or best_predictions is None:
        raise RuntimeError("Refiner sweep did not evaluate any configurations")
    return sweep, best, best_predictions


def _score_records(rows: list[Any]) -> list[dict[str, Any]]:
    return [
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
        for row in rows
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path,
        default=REPO_ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026",
    )
    parser.add_argument(
        "--detector-run",
        type=Path,
        default=REPO_ROOT / "artifacts/detector-experiments/maskrcnn-r50fpn-v2-1024-pq-holdout-20261110-8ep",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=REPO_ROOT / "artifacts/detector-experiments/maskrcnn-r50fpn-v2-gt-crop-refiner-256-holdout-20261110",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--crop-size", type=int, default=CROP_SIZE)
    parser.add_argument(
        "--training-view-strategy", choices=("selected", "all"), default="selected"
    )
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    detector_root = args.detector_run.expanduser().resolve()
    run_root = args.run_root.expanduser().resolve()
    if not data_root.is_relative_to(REPO_ROOT) or not detector_root.is_relative_to(REPO_ROOT):
        parser.error("Data and detector run paths must stay inside this repository")
    if not run_root.is_relative_to(REPO_ROOT) or run_root.exists() and any(run_root.iterdir()):
        parser.error("Run output must be inside the repository and empty or not yet created")
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs, patience, and batch size must be positive; workers nonnegative")
    if args.crop_size != CROP_SIZE:
        parser.error(f"This refiner pipeline uses fixed {CROP_SIZE}-pixel crops")
    detector_metadata_path = detector_root / "run-metadata.json"
    detector_predictions_path = detector_root / "best-validation-detections.json"
    if not detector_metadata_path.is_file() or not detector_predictions_path.is_file():
        parser.error("The detector run must contain metadata and best holdout detections")

    cv2.setNumThreads(0)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    if not torch.cuda.is_available():
        parser.error("CUDA is required for this training run")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(device)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    if args.smoke_only:
        model = CropRefiner().to(device).train()
        images = torch.rand((2, 3, CROP_SIZE, CROP_SIZE), device=device)
        targets = torch.rand((2, 1, CROP_SIZE, CROP_SIZE), device=device).round()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(images)
            loss, _ = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
        if not torch.isfinite(loss):
            raise RuntimeError("Crop-refiner smoke loss is not finite")
        print(json.dumps({
            "smoke_loss": float(loss.detach().cpu()),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "device": torch.cuda.get_device_name(device),
        }, indent=2))
        return 0

    run_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    dataset = load_dataset(data_root)
    split = create_observation_split(dataset, REPO_ROOT / "data/splits", seed=42)
    if split.sha256 != FROZEN_SPLIT_SHA256:
        raise RuntimeError(f"Frozen split changed: {split.sha256}")
    detector_metadata = json.loads(detector_metadata_path.read_text(encoding="utf-8"))
    if detector_metadata.get("split_sha256") != split.sha256:
        raise RuntimeError("Detector run uses a different grouped split")
    if detector_metadata.get("best_checkpoint_sha256") != sha256(detector_root / "best.pt"):
        raise RuntimeError("Detector checkpoint hash differs from its completed run metadata")

    if args.training_view_strategy == "selected":
        training_views = select_one_view_per_observation(dataset, split.training_names)
    else:
        training_views = tuple(
            sorted(
                (
                    view
                    for file_name in split.training_names
                    for view in dataset.views_by_observation[file_name]
                ),
                key=lambda view: (view.file_name, view.image_id),
            )
        )
    validation_names = split.validation_names
    train_names = {view.file_name for view in training_views}
    if train_names != set(split.training_names) or train_names & set(validation_names):
        raise AssertionError("Training and validation observations are not disjoint and complete")
    detector_predictions = load_detection_payload(detector_predictions_path, validation_names)
    crop_cache = run_root / "crop-cache"
    image_path, mask_path, crop_index_path, crop_count = build_training_crop_cache(
        dataset, training_views, crop_cache
    )
    crop_dataset = CropMaskDataset(image_path, mask_path, augment=True)
    generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(
        crop_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        collate_fn=None,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    model = CropRefiner().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.0001
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")
    metadata: dict[str, Any] = {
        "status": "running",
        "run_name": run_root.name,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_commit": git_commit(),
        "source_sha256": sha256(Path(__file__).resolve()),
        "split_sha256": split.sha256,
        "training_observations": len(train_names),
        "training_annotator_views": len(training_views),
        "validation_observations": len(validation_names),
        "validation_annotator_views": sum(
            len(dataset.views_by_observation[name]) for name in validation_names
        ),
        "training_crop_count": crop_count,
        "training_crop_index": str(crop_index_path.relative_to(REPO_ROOT)),
        "training_crop_index_sha256": sha256(crop_index_path),
        "detector_run": str(detector_root.relative_to(REPO_ROOT)),
        "detector_checkpoint_sha256": sha256(detector_root / "best.pt"),
        "detector_predictions_sha256": sha256(detector_predictions_path),
        "training_view_strategy": args.training_view_strategy,
        "training_view_selection": (
            "maximum annotation count, then total annotation area, then image id"
            if args.training_view_strategy == "selected"
            else "all annotator views, sorted by observation filename and image id"
        ),
        "training_labels": (
            f"competition-provided annotations from {len(training_views)} views of the "
            f"{len(train_names)} grouped training observations"
        ),
        "data_boundary": {
            "holdout_used_for_training": False,
            "holdout_used_for_checkpoint_and_postprocessing_selection": True,
            "test_images_used_for_training": False,
            "test_labels_used": False,
            "external_supervised_data_used_for_refiner_training": False,
            "external_pretrained_weights_used_for_refiner_training": False,
            "detector_checkpoint_uses_torchvision_coco_initialization": True,
        },
        "configuration": {
            "crop_size": CROP_SIZE,
            "crop_context": 1.8,
            "crop_minimum_side": 96,
            "epochs_requested": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "optimizer": f"AdamW, lr={args.learning_rate}, weight_decay=0.0001",
            "loss": "0.5 binary cross entropy with logits + 0.5 soft Dice loss",
            "encoder_initialization": "random ResNet-18",
            "augmentation": "random quarter-turns, horizontal flip, gamma, and brightness gain",
            "refiner_thresholds": REFINER_THRESHOLDS,
            "detector_thresholds": DETECTOR_THRESHOLDS,
            "minimum_areas": MINIMUM_AREAS,
        },
        "hardware": hardware_info(),
        "package_versions": package_versions(),
        "epochs": [],
    }
    (run_root / "run-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    best_pq = -1.0
    epochs_without_improvement = 0
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_started = time.perf_counter()
            total_loss = total_dice_loss = 0.0
            samples = 0
            for batch_index, (images, targets) in enumerate(train_loader, start=1):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(images)
                    loss, dice_loss = loss_fn(logits, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                batch_count = len(images)
                total_loss += float(loss.detach().cpu()) * batch_count
                total_dice_loss += float(dice_loss.detach().cpu()) * batch_count
                samples += batch_count
                if batch_index % 100 == 0 or batch_index == len(train_loader):
                    print(
                        f"epoch {epoch}/{args.epochs} batch {batch_index}/{len(train_loader)} "
                        f"loss={float(loss.detach().cpu()):.4f}",
                        flush=True,
                    )
            scheduler.step()
            model.eval()
            refined = predict_refiner_probabilities(
                model,
                dataset,
                validation_names,
                detector_predictions,
                device,
                crop_size=CROP_SIZE,
            )
            sweep, best, best_predictions = score_refined_epoch(
                dataset, validation_names, refined
            )
            row = {
                "epoch": epoch,
                "mean_training_loss": total_loss / max(samples, 1),
                "mean_soft_dice_loss": total_dice_loss / max(samples, 1),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "training_seconds": time.perf_counter() - epoch_started,
                "best_validation": best,
                "sweep": sweep,
            }
            metadata["epochs"].append(row)
            (run_root / "epoch-metrics.json").write_text(
                json.dumps(metadata["epochs"], indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            current_pq = float(best["metrics"]["pq"])
            print(
                f"epoch {epoch}/{args.epochs}: official PQ={current_pq:.6f}; "
                f"mode={best['merge_mode']} det_conf={best['confidence_threshold']:.2f} "
                f"mask_conf={best['refiner_threshold']:.2f} area={best['minimum_area']}; "
                f"TP={best['metrics']['tp']} FP={best['metrics']['fp']} "
                f"FN={best['metrics']['fn']}",
                flush=True,
            )
            if current_pq > best_pq:
                best_pq = current_pq
                epochs_without_improvement = 0
                torch.save({"epoch": epoch, "model": model.state_dict()}, run_root / "best.pt")
                (run_root / "best-validation-detections.json").write_text(
                    json.dumps(best_predictions, separators=(",", ":")),
                    encoding="utf-8",
                )
                metadata["best_epoch"] = epoch
                metadata["best_validation"] = best
            else:
                epochs_without_improvement += 1
            metadata["status"] = "running"
            metadata["peak_gpu_memory_bytes"] = int(
                torch.cuda.max_memory_allocated(device)
            )
            (run_root / "run-metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after {args.patience} epochs without PQ improvement", flush=True)
                break
    except KeyboardInterrupt:
        metadata["status"] = "interrupted"
        metadata["interrupted_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        (run_root / "run-metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise

    best_path = run_root / "best-validation-detections.json"
    best_predictions = json.loads(best_path.read_text(encoding="utf-8"))
    selection = metadata["best_validation"]
    metrics, observation_rows, prediction_count = score_detector_predictions(
        dataset,
        validation_names,
        best_predictions,
        confidence_threshold=float(selection["confidence_threshold"]),
        minimum_area=int(selection["minimum_area"]),
    )
    observation_scores = _score_records(observation_rows)
    with (run_root / "per-observation-scores.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(observation_scores[0]))
        writer.writeheader()
        writer.writerows(observation_scores)
    overlay_predictions: dict[str, list[PredictedInstance]] = {}
    selected_predictions = best_predictions
    for file_name in validation_names:
        stem = Path(file_name).stem
        retained = []
        for index, item in enumerate(selected_predictions.get(stem, []), start=1):
            counts = item["counts"].encode("ascii")
            rle = {"size": list(IMAGE_SHAPE), "counts": counts}
            area = int(mask_utils.area(rle))
            if (
                float(item["score"]) < float(selection["confidence_threshold"])
                or area < int(selection["minimum_area"])
            ):
                continue
            retained.append(
                PredictedInstance(f"{stem}_{index}", stem, area, rle)
            )
        overlay_predictions[file_name] = retained
    overlay_names = write_comparison_overlays(
        dataset, overlay_predictions, run_root / "overlays", maximum=8
    )
    evaluation = {
        "run_name": run_root.name,
        "split_sha256": split.sha256,
        "validation_observations": len(validation_names),
        "validation_annotator_views": metadata["validation_annotator_views"],
        "configuration": selection,
        "prediction_count": prediction_count,
        "official_scorer": metrics,
        "per_observation_csv": "per-observation-scores.csv",
        "overlays": overlay_names,
        "best_checkpoint_sha256": sha256(run_root / "best.pt"),
        "best_predictions_sha256": sha256(best_path),
        "scorer_sha256": sha256(REPO_ROOT / "src/filament_baseline/scorer.py"),
        "source_sha256": metadata["source_sha256"],
        "source_commit": metadata["source_commit"],
    }
    (run_root / "official-pq-evaluation.json").write_text(
        json.dumps(evaluation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata["status"] = "complete"
    metadata["elapsed_seconds"] = time.perf_counter() - started
    metadata["peak_gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
    metadata["best_checkpoint_sha256"] = sha256(run_root / "best.pt")
    metadata["best_predictions_sha256"] = sha256(best_path)
    metadata["evaluation_sha256"] = sha256(run_root / "official-pq-evaluation.json")
    (run_root / "run-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": metadata["status"],
                "run_root": str(run_root.relative_to(REPO_ROOT)),
                "best_epoch": metadata["best_epoch"],
                "best_validation": metadata["best_validation"],
                "elapsed_seconds": metadata["elapsed_seconds"],
                "peak_gpu_memory_bytes": metadata["peak_gpu_memory_bytes"],
                "overlay_count": len(overlay_names),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
