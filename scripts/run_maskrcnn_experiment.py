# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Train a grouped-holdout Mask R-CNN baseline and score it with official PQ."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
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
from torch.utils.data import DataLoader, Dataset

from filament_baseline.data import CompetitionDataset, ImageView, inventory, load_dataset
from filament_baseline.predictor import encode_binary_mask
from filament_baseline.runtime import hardware_info, package_versions
from filament_baseline.scorer import annotation_to_rle
from filament_baseline.split import create_observation_split
from score_detector_json import score_detector_predictions


FROZEN_SPLIT_SHA256 = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"
SCORE_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60)
MINIMUM_AREAS = (0, 100, 200, 400, 600)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
            text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def annotation_view_rank(
    dataset: CompetitionDataset, view: ImageView
) -> tuple[int, float, str]:
    annotations = dataset.annotations_by_image_id[view.image_id]
    return (
        len(annotations),
        sum(float(annotation.get("area", 0.0)) for annotation in annotations),
        view.image_id,
    )


def select_one_view_per_observation(
    dataset: CompetitionDataset, names: tuple[str, ...]
) -> tuple[ImageView, ...]:
    selected = []
    for file_name in names:
        views = dataset.views_by_observation[file_name]
        selected.append(max(views, key=lambda view: annotation_view_rank(dataset, view)))
    return tuple(selected)


def masks_to_boxes(masks: np.ndarray) -> np.ndarray:
    boxes: list[list[float]] = []
    for mask in masks:
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            continue
        boxes.append(
            [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
        )
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def encode_mask_rle(mask: np.ndarray) -> dict[str, Any]:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("ascii")
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": rle["counts"]}


def merge_positive_overlaps(
    detections: list[dict[str, Any]], image_size: int = 2048
) -> list[dict[str, Any]]:
    if len(detections) < 2:
        return detections
    rles = [
        {"size": [image_size, image_size], "counts": item["counts"].encode("ascii")}
        for item in detections
    ]
    adjacency = mask_utils.iou(rles, rles, [0] * len(rles)) > 0.0
    seen: set[int] = set()
    merged: list[dict[str, Any]] = []
    for start in range(len(detections)):
        if start in seen:
            continue
        stack, group = [start], []
        while stack:
            index = stack.pop()
            if index in seen:
                continue
            seen.add(index)
            group.append(index)
            stack.extend(np.flatnonzero(adjacency[index]).tolist())
        if len(group) == 1:
            merged.append(detections[group[0]])
            continue
        union = mask_utils.merge([rles[index] for index in group])
        if isinstance(union["counts"], bytes):
            counts = union["counts"].decode("ascii")
        else:
            counts = str(union["counts"])
        merged.append(
            {
                "score": max(float(detections[index]["score"]) for index in group),
                "counts": counts,
            }
        )
    return merged


class FilamentInstanceDataset(Dataset):
    def __init__(
        self,
        dataset: CompetitionDataset,
        views: tuple[ImageView, ...],
        image_size: int,
        augment: bool,
    ) -> None:
        self.dataset = dataset
        self.views = views
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.views)

    @staticmethod
    def photometric(image: np.ndarray) -> np.ndarray:
        valid = image > 0.01
        if np.any(valid) and random.random() < 0.7:
            image[valid] = image[valid] ** random.uniform(0.80, 1.25)
        if np.any(valid) and random.random() < 0.7:
            mean = float(image[valid].mean())
            contrast, gain = random.uniform(0.85, 1.15), random.uniform(0.90, 1.10)
            image[valid] = ((image[valid] - mean) * contrast + mean) * gain
        if np.any(valid) and random.random() < 0.20:
            image[valid] += np.random.normal(0.0, random.uniform(0.003, 0.015), valid.sum())
        return np.clip(image, 0.0, 1.0)

    def __getitem__(self, index: int):
        view = self.views[index]
        path = self.dataset.train_image_dir / view.file_name
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape != (view.height, view.width):
            raise ValueError(f"Could not read expected image {path}")
        image = cv2.resize(
            gray, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA
        ).astype(np.float32) / 255.0

        masks: list[np.ndarray] = []
        for annotation in self.dataset.annotations_by_image_id[view.image_id]:
            rle = annotation_to_rle(annotation, view.height, view.width)
            mask = mask_utils.decode(rle)
            if mask.shape != (self.image_size, self.image_size):
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (self.image_size, self.image_size),
                    interpolation=cv2.INTER_NEAREST,
                )
            mask = mask.astype(bool)
            if mask.any():
                masks.append(mask)
        if masks:
            mask_array = np.stack(masks).astype(np.uint8, copy=False)
        else:
            mask_array = np.zeros((0, self.image_size, self.image_size), dtype=np.uint8)

        if self.augment:
            rotations = random.randrange(4)
            image = np.rot90(image, rotations)
            mask_array = np.rot90(mask_array, rotations, axes=(1, 2))
            if random.random() < 0.5:
                image = np.fliplr(image)
                mask_array = np.flip(mask_array, axis=2)
            image = self.photometric(image)

        image = np.ascontiguousarray(image)
        mask_array = np.ascontiguousarray(mask_array)
        boxes = masks_to_boxes(mask_array)
        if len(boxes) != len(mask_array):
            raise ValueError(f"Empty instance mask in training observation {view.file_name}")
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.ones(len(mask_array), dtype=torch.int64),
            "masks": torch.as_tensor(mask_array, dtype=torch.uint8),
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": torch.as_tensor(mask_array.sum(axis=(1, 2)), dtype=torch.float32),
            "iscrowd": torch.zeros(len(mask_array), dtype=torch.int64),
        }
        tensor = torch.from_numpy(image).unsqueeze(0).repeat(3, 1, 1)
        return tensor, target, view.file_name


def collate_fn(batch):
    return tuple(zip(*batch))


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def build_model(image_size: int):
    from torchvision.models.detection import (
        MaskRCNN_ResNet50_FPN_V2_Weights,
        maskrcnn_resnet50_fpn_v2,
    )
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    from torchvision.ops import MultiScaleRoIAlign

    model = maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        weights_backbone=None,
    )
    box_features = model.roi_heads.box_predictor.cls_score.in_features
    mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_features = model.roi_heads.mask_predictor.conv5_mask.out_channels
    model.roi_heads.box_predictor = FastRCNNPredictor(box_features, 2)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_features, hidden_features, 2)
    model.rpn.anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )
    model.roi_heads.mask_roi_pool = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"], output_size=28, sampling_ratio=2
    )
    model.transform.min_size = (image_size,)
    model.transform.max_size = image_size
    model.roi_heads.detections_per_img = 200
    return model


@torch.inference_mode()
def predict_observations(
    model,
    dataset: CompetitionDataset,
    names: tuple[str, ...],
    image_size: int,
    device: torch.device,
) -> dict[str, list[dict[str, Any]]]:
    model.eval()
    predictions: dict[str, list[dict[str, Any]]] = {}
    for index, file_name in enumerate(names, start=1):
        path = dataset.train_image_dir / file_name
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape != (2048, 2048):
            raise ValueError(f"Expected a readable 2048x2048 image: {path}")
        resized = cv2.resize(
            gray, (image_size, image_size), interpolation=cv2.INTER_AREA
        ).astype(np.float32) / 255.0
        tensor = torch.from_numpy(resized).unsqueeze(0).repeat(3, 1, 1).to(device)
        output = model([tensor])[0]
        masks = output["masks"][:, 0].detach().cpu().numpy()
        scores = output["scores"].detach().cpu().numpy()
        detections: list[dict[str, Any]] = []
        for mask, score in zip(masks, scores, strict=True):
            binary = mask >= 0.5
            if binary.shape != (2048, 2048):
                binary = cv2.resize(
                    binary.astype(np.uint8),
                    (2048, 2048),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            if not binary.any():
                continue
            detections.append(
                {"score": float(score), "counts": encode_binary_mask(binary)["counts"]}
            )
        predictions[Path(file_name).stem] = detections
        if index % 20 == 0 or index == len(names):
            print(f"Predicted {index}/{len(names)} validation observations", flush=True)
    model.train()
    return predictions


def score_prediction_modes(
    dataset: CompetitionDataset,
    names: tuple[str, ...],
    raw_predictions: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    merged_predictions = {
        stem: merge_positive_overlaps(items) for stem, items in raw_predictions.items()
    }
    sweep: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for mode, predictions in (("raw", raw_predictions), ("positive_iou_merge", merged_predictions)):
        for confidence in SCORE_THRESHOLDS:
            for minimum_area in MINIMUM_AREAS:
                metrics, _, prediction_count = score_detector_predictions(
                    dataset,
                    names,
                    predictions,
                    confidence_threshold=confidence,
                    minimum_area=minimum_area,
                )
                row = {
                    "merge_mode": mode,
                    "confidence_threshold": confidence,
                    "minimum_area": minimum_area,
                    "prediction_count": prediction_count,
                    "metrics": metrics["global"],
                }
                sweep.append(row)
                if best is None or row["metrics"]["pq"] > best["metrics"]["pq"]:
                    best = row
    if best is None:
        raise RuntimeError("The detector validation sweep did not evaluate any settings")
    return sweep, best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=REPO_ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026"
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=REPO_ROOT
        / "artifacts/detector-experiments/maskrcnn-r50fpn-v2-1024-pq-holdout-20261110",
    )
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--training-view-strategy", choices=("selected", "all"), default="selected"
    )
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    run_root = args.run_root.expanduser().resolve()
    if not data_root.is_relative_to(REPO_ROOT / "data"):
        raise ValueError("Competition data must remain under data/")
    if not run_root.is_relative_to(REPO_ROOT / "artifacts"):
        raise ValueError("Experiment outputs must remain under artifacts/")
    if args.image_size < 512 or args.image_size % 32:
        raise ValueError("image-size must be a multiple of 32 and at least 512")
    if args.epochs < 1 or args.patience < 1 or args.workers < 0:
        raise ValueError("epochs and patience must be positive and workers nonnegative")
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"Run output already contains files: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    torch_home = Path(os.environ["TORCH_HOME"])
    torch_home.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(data_root)
    data_inventory = inventory(dataset)
    if data_inventory["observation_count"] != 707 or data_inventory["annotator_view_count"] != 1154:
        raise ValueError(f"Unexpected competition data inventory: {data_inventory}")
    split = create_observation_split(dataset, REPO_ROOT / "data/splits")
    if split.sha256 != FROZEN_SPLIT_SHA256:
        raise RuntimeError(f"Frozen validation split changed: {split.sha256}")
    if args.training_view_strategy == "selected":
        train_views = select_one_view_per_observation(dataset, split.training_names)
    else:
        train_views = tuple(
            sorted(
                (
                    view
                    for file_name in split.training_names
                    for view in dataset.views_by_observation[file_name]
                ),
                key=lambda view: (view.file_name, view.image_id),
            )
        )
    if {view.file_name for view in train_views} != set(split.training_names):
        raise AssertionError("Training annotations do not cover the frozen training observations")
    if {view.file_name for view in train_views} & set(split.validation_names):
        raise AssertionError("Frozen holdout observations crossed into detector training")

    manifest_path = run_root / "training_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        import csv

        writer = csv.DictWriter(
            stream, fieldnames=("file_name", "image_id", "year", "site", "annotation_count")
        )
        writer.writeheader()
        for view in train_views:
            writer.writerow(
                {
                    "file_name": view.file_name,
                    "image_id": view.image_id,
                    "year": view.year,
                    "site": view.site,
                    "annotation_count": len(dataset.annotations_by_image_id[view.image_id]),
                }
            )

    import torchvision
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights

    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    if not torch.cuda.is_available() and args.device != "cpu":
        raise RuntimeError("CUDA is not available; choose --device cpu only for a small smoke run")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    cv2.setNumThreads(0)
    torch.set_num_threads(min(8, os.cpu_count() or 1))

    started = time.perf_counter()
    model = build_model(args.image_size).to(device)
    metadata = {
        "status": "running",
        "run_name": run_root.name,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_commit": git_commit(),
        "source_sha256": sha256(Path(__file__).resolve()),
        "split_sha256": split.sha256,
        "training_manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "training_manifest_sha256": sha256(manifest_path),
        "data_inventory": data_inventory,
        "training_observations": len({view.file_name for view in train_views}),
        "training_annotator_views": len(train_views),
        "validation_observations": len(split.validation_names),
        "validation_annotator_views": sum(
            len(dataset.views_by_observation[name]) for name in split.validation_names
        ),
        "training_view_strategy": args.training_view_strategy,
        "training_view_selection": (
            "maximum annotation count, then total annotation area, then image id"
            if args.training_view_strategy == "selected"
            else "all annotator views, sorted by observation filename and image id"
        ),
        "training_labels": (
            f"competition-provided annotations from {len(train_views)} views of the "
            f"{len({view.file_name for view in train_views})} grouped training observations"
        ),
        "pretrained_weights": {
            "name": "torchvision Mask R-CNN ResNet50 FPN V2 COCO",
            "url": weights.url,
            "license": "Torchvision model weights; downloaded by torchvision into workspace .cache/torch",
        },
        "configuration": {
            "image_size": args.image_size,
            "epochs_requested": args.epochs,
            "patience": args.patience,
            "batch_size": 1,
            "workers": args.workers,
            "optimizer": "SGD, lr=0.005, momentum=0.9, weight_decay=0.0001",
            "amp": device.type == "cuda",
            "mask_roi_output_size": 28,
            "detections_per_image": 200,
        },
        "data_boundary": {
            "holdout_used_for_training": False,
            "test_images_used_for_training": False,
            "test_labels_used": False,
            "external_supervised_data": False,
        },
        "hardware": hardware_info(),
        "package_versions": {**package_versions(), "torchvision": torchvision.__version__},
        "epochs": [],
    }
    (run_root / "run-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if args.smoke_only:
        view = train_views[0]
        sample = FilamentInstanceDataset(dataset, (view,), args.image_size, False)[0]
        image, target, _ = sample
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            losses = model([image.to(device)], [{key: value.to(device) for key, value in target.items()}])
            loss = sum(losses.values())
        loss.backward()
        optimizer.step()
        metadata["status"] = "smoke_complete"
        metadata["smoke_loss"] = float(loss.detach().cpu())
        metadata["smoke_gpu_memory_bytes"] = (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        )
        (run_root / "run-metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
        return 0

    train_dataset = FilamentInstanceDataset(dataset, train_views, args.image_size, True)
    generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    optimizer = torch.optim.SGD(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0001,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=sorted(set((max(1, round(args.epochs * 0.7)), max(2, round(args.epochs * 0.9))))),
        gamma=0.1,
    )
    best_pq = -1.0
    epochs_without_improvement = 0

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_started = time.perf_counter()
            loss_sums: dict[str, float] = {}
            seen = 0
            for batch_index, (images, targets, _) in enumerate(train_loader, start=1):
                moved_images = [image.to(device, non_blocking=True) for image in images]
                moved_targets = [
                    {key: value.to(device, non_blocking=True) for key, value in target.items()}
                    for target in targets
                ]
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    losses = model(moved_images, moved_targets)
                    loss = sum(losses.values())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                for name, value in losses.items():
                    loss_sums[name] = loss_sums.get(name, 0.0) + float(value.detach().cpu())
                seen += 1
                if batch_index % 50 == 0 or batch_index == len(train_loader):
                    print(
                        f"epoch {epoch}/{args.epochs} batch {batch_index}/{len(train_loader)} "
                        f"loss {float(loss.detach().cpu()):.4f}",
                        flush=True,
                    )
            scheduler.step()

            prediction_started = time.perf_counter()
            raw_predictions = predict_observations(
                model, dataset, split.validation_names, args.image_size, device
            )
            prediction_seconds = time.perf_counter() - prediction_started
            sweep, best = score_prediction_modes(dataset, split.validation_names, raw_predictions)
            row = {
                "epoch": epoch,
                "mean_training_losses": {name: value / max(seen, 1) for name, value in loss_sums.items()},
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "training_seconds": time.perf_counter() - epoch_started - prediction_seconds,
                "prediction_seconds": prediction_seconds,
                "best_validation": best,
                "sweep": sweep,
            }
            metadata["epochs"].append(row)
            (run_root / "epoch-metrics.json").write_text(
                json.dumps(metadata["epochs"], indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
                run_root / "last.pt",
            )
            current_pq = float(best["metrics"]["pq"])
            print(
                f"epoch {epoch}/{args.epochs}: official PQ={current_pq:.6f}; "
                f"mode={best['merge_mode']} conf={best['confidence_threshold']:.2f} "
                f"area={best['minimum_area']}; TP={best['metrics']['tp']} "
                f"FP={best['metrics']['fp']} FN={best['metrics']['fn']}; "
                f"elapsed={row['training_seconds'] + prediction_seconds:.1f}s",
                flush=True,
            )
            if current_pq > best_pq:
                best_pq = current_pq
                epochs_without_improvement = 0
                best_payload = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "validation_selection": {
                        key: best[key]
                        for key in ("merge_mode", "confidence_threshold", "minimum_area")
                    },
                    "validation_metrics": best["metrics"],
                }
                torch.save(best_payload, run_root / "best.pt")
                selected = (
                    raw_predictions
                    if best["merge_mode"] == "raw"
                    else {
                        stem: merge_positive_overlaps(items)
                        for stem, items in raw_predictions.items()
                    }
                )
                (run_root / "best-validation-detections.json").write_text(
                    json.dumps(selected, separators=(",", ":")), encoding="utf-8"
                )
                metadata["best_epoch"] = epoch
                metadata["best_validation"] = best
            else:
                epochs_without_improvement += 1
            metadata["status"] = "running"
            metadata["peak_gpu_memory_bytes"] = (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            )
            (run_root / "run-metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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

    metadata["status"] = "complete"
    metadata["elapsed_seconds"] = time.perf_counter() - started
    metadata["peak_gpu_memory_bytes"] = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    metadata["best_checkpoint_sha256"] = sha256(run_root / "best.pt")
    metadata["last_checkpoint_sha256"] = sha256(run_root / "last.pt")
    (run_root / "run-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": metadata["status"],
                "run_root": str(run_root),
                "best_epoch": metadata.get("best_epoch"),
                "best_validation": metadata.get("best_validation"),
                "elapsed_seconds": metadata["elapsed_seconds"],
                "peak_gpu_memory_bytes": metadata["peak_gpu_memory_bytes"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
