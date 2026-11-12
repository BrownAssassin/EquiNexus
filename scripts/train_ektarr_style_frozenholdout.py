# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
import subprocess
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ["TORCH_HOME"] = str(REPO_ROOT / ".cache" / "torch")
os.environ["YOLO_CONFIG_DIR"] = str(REPO_ROOT / ".cache" / "ultralytics")
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pycocotools import mask as mask_utils
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO

from evaluate_ektarr_public_notebook import (
    SOURCE_CONFIG,
    CropUNet,
    digest,
    encode_mask,
    refine_masks,
    source_split,
    square_bounds,
)
from filament_baseline.data import load_dataset
from filament_baseline.predictor import PredictedInstance
from filament_baseline.runtime import hardware_info, package_versions
from filament_baseline.scorer import score_validation


BASE_WEIGHTS = REPO_ROOT / "data" / "reference" / "pretrained" / "ultralytics-yolo11" / "yolo11s-seg-v8.3.0.pt"
RUN_NAME = "ektarr-style-yolo11s-crop-refiner-frozenholdout-20261111"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def coco_polygon_to_yolo_line(polygon: list[float], width: int, height: int, class_id: int) -> str:
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    points[:, 0] = np.clip(points[:, 0] / width, 0, 1)
    points[:, 1] = np.clip(points[:, 1] / height, 0, 1)
    coordinates = " ".join(f"{value:.6f}" for value in points.flatten())
    return f"{class_id} {coordinates}"


def export_selected_views(dataset, selection: dict[str, str], partition: dict[str, str], out_dir: Path) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"Refusing to reuse an existing generated YOLO data directory: {out_dir}")
    images_dir = dataset.train_image_dir
    annotation_payload = json.loads(dataset.annotation_path.read_text(encoding="utf-8"))
    raw_images = {str(row["id"]): row for row in annotation_payload["images"]}
    raw_annotations: dict[str, list[dict[str, Any]]] = {}
    for row in annotation_payload["annotations"]:
        raw_annotations.setdefault(str(row["image_id"]), []).append(row)
    categories = sorted(annotation_payload.get("categories", []), key=lambda row: row["id"])
    class_id = {row["id"]: index for index, row in enumerate(categories)}
    names = [str(row["name"]) for row in categories]
    counts = {"train": {"observations": 0, "instances": 0}, "val": {"observations": 0, "instances": 0}}
    rows = []

    for filename in sorted(selection):
        split = "train" if partition[filename] == "train" else "val"
        image_id = selection[filename]
        image_info = raw_images[image_id]
        height, width = int(image_info["height"]), int(image_info["width"])
        src = images_dir / filename
        img_dir = out_dir / "images" / split
        label_dir = out_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        destination = img_dir / filename
        try:
            os.link(src, destination)
        except OSError:
            import shutil
            shutil.copy2(src, destination)

        label_lines = []
        for annotation in raw_annotations.get(image_id, []):
            if annotation.get("iscrowd", 0):
                continue
            target_class = class_id[annotation["category_id"]]
            segmentation = annotation.get("segmentation", [])
            if not isinstance(segmentation, list):
                continue
            for polygon in segmentation:
                if len(polygon) < 6:
                    continue
                label_lines.append(coco_polygon_to_yolo_line(polygon, width, height, target_class))
        (label_dir / f"{Path(filename).stem}.txt").write_text("\n".join(label_lines), encoding="utf-8")
        counts[split]["observations"] += 1
        counts[split]["instances"] += len(label_lines)
        rows.append({
            "file_name": filename,
            "partition": partition[filename],
            "selected_image_id": image_id,
            "source_view_count": len(dataset.views_by_observation[filename]),
            "selected_annotation_count": len(dataset.annotations_by_image_id[image_id]),
            "yolo_polygon_count": len(label_lines),
            "year": dataset.views_by_observation[filename][0].year,
            "site": dataset.views_by_observation[filename][0].site,
        })

    yaml_names = "\n".join(f"  {index}: {json.dumps(name)}" for index, name in enumerate(names))
    yaml = f"path: {out_dir.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{yaml_names}\n"
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(yaml, encoding="utf-8")
    manifest_path = out_dir / "selected-view-export.csv"
    manifest_path.write_text(pd.DataFrame(rows).to_csv(index=False, lineterminator="\n"), encoding="utf-8")
    return {"counts": counts, "data_yaml": yaml_path, "manifest": manifest_path, "classes": names}


def selected_crops(dataset, selected_ids: dict[str, str], names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for filename in names:
        image_id = selected_ids[filename]
        image = cv2.imread(str(dataset.train_image_dir / filename), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (2048, 2048):
            raise ValueError(f"Could not decode full-resolution training image {filename}")
        for annotation in dataset.annotations_by_image_id[image_id]:
            mask = np.zeros(image.shape, dtype=np.uint8)
            segmentation = annotation.get("segmentation", [])
            if not isinstance(segmentation, list):
                continue
            for polygon in segmentation:
                points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
                if len(points) >= 3:
                    cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
            if not mask.any():
                continue
            x0, y0, x1, y1 = square_bounds(mask)
            images.append(cv2.resize(image[y0:y1, x0:x1], (256, 256), interpolation=cv2.INTER_LINEAR))
            masks.append(cv2.resize(mask[y0:y1, x0:x1], (256, 256), interpolation=cv2.INTER_NEAREST))
    if not images:
        raise RuntimeError("No refiner crops were produced")
    return np.stack(images), np.stack(masks)


class CropDataset(Dataset):
    def __init__(self, images: np.ndarray, masks: np.ndarray, augment: bool):
        self.images = images
        self.masks = masks
        self.augment = augment

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = torch.from_numpy(self.images[index].copy()).float() / 255
        mask = torch.from_numpy(self.masks[index].copy()).float().unsqueeze(0)
        if self.augment:
            k = random.randrange(4)
            image = torch.rot90(image, k, (0, 1))
            mask = torch.rot90(mask, k, (1, 2))
            if random.random() < 0.5:
                image, mask = torch.flip(image, (1,)), torch.flip(mask, (2,))
            if random.random() < 0.5:
                image = image ** random.uniform(0.85, 1.20)
            if random.random() < 0.5:
                image = torch.clamp(image * random.uniform(0.90, 1.10), 0, 1)
        return image.unsqueeze(0).repeat(3, 1, 1), mask


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = logits.sigmoid()
    intersection = (probabilities * targets).sum((1, 2, 3))
    denominator = probabilities.sum((1, 2, 3)) + targets.sum((1, 2, 3))
    return (1 - (2 * intersection + 1) / (denominator + 1)).mean()


def train_refiner(
    model: CropUNet,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    output_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    best_dice = -1.0
    history = []
    started = time.perf_counter()
    for epoch in range(1, 9):
        model.train()
        loss_sum = 0.0
        batch_count = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = 0.5 * nn.functional.binary_cross_entropy_with_logits(logits, masks) + 0.5 * soft_dice_loss(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach())
            batch_count += 1

        model.eval()
        dice_sum = 0.0
        sample_count = 0
        with torch.inference_mode():
            for images, masks in validation_loader:
                images, masks = images.to(device), masks.to(device).bool()
                with torch.amp.autocast("cuda"):
                    predicted = model(images).sigmoid() >= SOURCE_CONFIG["refiner_threshold"]
                intersection = (predicted & masks).sum((1, 2, 3)).float()
                denominator = predicted.sum((1, 2, 3)) + masks.sum((1, 2, 3))
                dice_sum += float(((2 * intersection + 1e-7) / (denominator + 1e-7)).sum())
                sample_count += len(images)
        validation_dice = dice_sum / sample_count
        epoch_row = {
            "epoch": epoch,
            "mean_training_loss": loss_sum / max(batch_count, 1),
            "validation_patch_dice": validation_dice,
        }
        history.append(epoch_row)
        print(json.dumps({"event": "refiner_epoch", **epoch_row}), flush=True)
        if validation_dice > best_dice:
            best_dice = validation_dice
            torch.save(model.state_dict(), output_path)
        scheduler.step()
    return {
        "training_seconds": time.perf_counter() - started,
        "best_epoch": max(history, key=lambda row: row["validation_patch_dice"])["epoch"],
        "best_validation_patch_dice": best_dice,
        "history": history,
        "checkpoint": str(output_path),
        "checkpoint_sha256": digest(output_path),
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Train the public Ektarr YOLO+crop-refiner recipe on the frozen grouped split.")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "detector-experiments" / RUN_NAME)
    parser.add_argument("--yolo-data-dir", type=Path, default=REPO_ROOT / "data" / "processed" / "yolo" / RUN_NAME)
    parser.add_argument("--detector-batch-size", type=int, default=2)
    parser.add_argument("--refiner-batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    yolo_data_dir = args.yolo_data_dir.expanduser().resolve()
    if not output_dir.is_relative_to((REPO_ROOT / "artifacts").resolve()):
        raise ValueError("Training artifacts must stay under artifacts/")
    if not yolo_data_dir.is_relative_to((REPO_ROOT / "data" / "processed").resolve()):
        raise ValueError("Generated training data must stay under data/processed/")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to reuse a nonempty run directory: {output_dir}")
    if not BASE_WEIGHTS.is_file():
        raise FileNotFoundError(f"COCO initialization checkpoint missing: {BASE_WEIGHTS}")
    if digest(BASE_WEIGHTS) != "1caa81c0195412efa411b632bcfb8c184939dddb6ae41f6a80c41b211ff257c3":
        raise RuntimeError("COCO checkpoint SHA-256 differs from the audited MORDEN/KSO YOLO11s source weight")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(42)

    dataset = load_dataset(data_root)
    split_rows = pd.read_csv(REPO_ROOT / "data" / "splits" / "observation_split_seed42.csv")
    partition = dict(zip(split_rows["file_name"], split_rows["partition"], strict=True))
    if set(partition) != set(dataset.observation_names):
        raise RuntimeError("Frozen split manifest does not cover the competition observations")
    audit_rows, _, _ = source_split(dataset.annotation_path)
    selected_ids = {row["file_name"]: row["selected_image_id"] for row in audit_rows}
    export = export_selected_views(dataset, selected_ids, partition, yolo_data_dir)
    training_names = sorted(name for name, part in partition.items() if part == "train")
    validation_names = sorted(name for name, part in partition.items() if part == "validation")

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    detector_started = time.perf_counter()
    detector = YOLO(str(BASE_WEIGHTS))
    detector.train(
        data=str(export["data_yaml"]),
        epochs=30,
        imgsz=SOURCE_CONFIG["image_size"],
        batch=args.detector_batch_size,
        workers=args.workers,
        device=0,
        patience=SOURCE_CONFIG["patience"],
        seed=42,
        single_cls=False,
        mask_ratio=SOURCE_CONFIG["mask_ratio"],
        overlap_mask=False,
        hsv_h=0.0,
        hsv_s=0.0,
        fliplr=0.0,
        flipud=0.0,
        mosaic=0.3,
        degrees=10,
        scale=0.3,
        hsv_v=0.3,
        close_mosaic=5,
        project=str(output_dir / "detector-training"),
        name="frozen-split",
        exist_ok=False,
    )
    detector_training_seconds = time.perf_counter() - detector_started
    detector_path = output_dir / "detector-training" / "frozen-split" / "weights" / "best.pt"
    if not detector_path.is_file():
        raise FileNotFoundError(f"Ultralytics did not produce the expected best checkpoint: {detector_path}")

    crop_train_images, crop_train_masks = selected_crops(dataset, selected_ids, training_names)
    crop_validation_images, crop_validation_masks = selected_crops(dataset, selected_ids, validation_names)
    crop_train_loader = DataLoader(
        CropDataset(crop_train_images, crop_train_masks, augment=True),
        batch_size=args.refiner_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    crop_validation_loader = DataLoader(
        CropDataset(crop_validation_images, crop_validation_masks, augment=False),
        batch_size=max(args.refiner_batch_size, 8),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    refiner_path = output_dir / "best_refiner.pt"
    refiner_started = time.perf_counter()
    refiner = CropUNet(pretrained=True)
    resnet_weights_path = REPO_ROOT / ".cache" / "torch" / "hub" / "checkpoints" / "resnet18-f37072fd.pth"
    if not resnet_weights_path.is_file():
        raise FileNotFoundError("Expected torchvision's ImageNet ResNet18 checkpoint inside the workspace cache")
    resnet_weights_sha256 = digest(resnet_weights_path)
    refiner_result = train_refiner(refiner, crop_train_loader, crop_validation_loader, refiner_path, device)
    refiner_training_seconds = time.perf_counter() - refiner_started
    training_peak_gpu_memory_bytes = int(torch.cuda.max_memory_allocated())
    del crop_train_images, crop_train_masks, crop_validation_images, crop_validation_masks
    del crop_train_loader, crop_validation_loader, refiner
    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()
    detector = YOLO(str(detector_path))
    refiner = CropUNet(pretrained=False)
    refiner.load_state_dict(torch.load(refiner_path, map_location="cpu", weights_only=True), strict=True)
    refiner.to(device).eval()
    predictions: dict[str, list[dict[str, Any]]] = {Path(name).stem: [] for name in validation_names}
    inference_started = time.perf_counter()
    for index, filename in enumerate(validation_names, start=1):
        image_path = dataset.train_image_dir / filename
        image_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        result = detector.predict(
            str(image_path), imgsz=SOURCE_CONFIG["image_size"],
            conf=SOURCE_CONFIG["confidence_threshold"], agnostic_nms=True,
            retina_masks=True, verbose=False, device=0,
        )[0]
        if result.masks is None:
            continue
        masks = result.masks.data.detach().cpu().numpy().astype(np.uint8)
        scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
        if masks.shape[1:] != (2048, 2048) or masks.shape[0] != len(scores):
            raise RuntimeError(f"Unexpected YOLO output for {filename}: {masks.shape}, {scores.shape}")
        refined = refine_masks(image_gray, masks, refiner, device)
        for mask, score in zip(refined, scores, strict=True):
            if int(mask.sum()) >= SOURCE_CONFIG["minimum_component_size"]:
                predictions[Path(filename).stem].append({"score": float(score), **encode_mask(mask)})
        if index % 10 == 0 or index == len(validation_names):
            print(json.dumps({"event": "inference_progress", "completed": index, "total": len(validation_names)}), flush=True)
    inference_seconds = time.perf_counter() - inference_started
    prediction_path = output_dir / "validation_predictions.json"
    prediction_path.write_text(json.dumps(predictions, separators=(",", ":")), encoding="utf-8")

    predictions_by_name = {}
    for filename in validation_names:
        stem = Path(filename).stem
        retained = []
        for index, item in enumerate(predictions[stem], start=1):
            rle = {"size": item["size"], "counts": item["counts"].encode("ascii")}
            area = int(mask_utils.area(rle))
            if item["score"] >= SOURCE_CONFIG["confidence_threshold"] and area >= SOURCE_CONFIG["minimum_component_size"]:
                retained.append(PredictedInstance(f"{stem}_{index}", stem, area, rle))
        predictions_by_name[filename] = retained
    metrics, observation_rows = score_validation(
        dataset, predictions_by_name, validation_names,
        iou_threshold=0.5, diagnostic_overlap_threshold=0.0,
    )

    source_selection_csv = export["manifest"]
    split_digest = hashlib.sha256(
        (REPO_ROOT / "data" / "splits" / "observation_split_seed42.csv").read_bytes()
    ).hexdigest()
    metadata = {
        "run_name": RUN_NAME,
        "source_method": "Ektarr public YOLO + U-Net crop refiner notebook, retrained from public COCO/ImageNet initialization on the EquiNexus frozen split.",
        "source_notebook": "https://www.kaggle.com/code/ektarr/yolo-u-net-solar-filament-segmentation",
        "source_notebook_license": "Apache-2.0",
        "ultralytics_license": "AGPL-3.0",
        "external_initialization": {
            "yolo_checkpoint": str(BASE_WEIGHTS.relative_to(REPO_ROOT)),
            "yolo_license": "AGPL-3.0",
            "yolo_sha256": digest(BASE_WEIGHTS),
            "resnet18_weights": "torchvision ResNet18_Weights.DEFAULT (ImageNet-1K); downloaded only to workspace .cache/torch.",
            "resnet18_checkpoint": str(REPO_ROOT / ".cache" / "torch" / "hub" / "checkpoints" / "resnet18-f37072fd.pth"),
            "resnet18_checkpoint_sha256": resnet_weights_sha256,
        },
        "competition_split_sha256": split_digest,
        "training_observations": len(training_names),
        "validation_observations": len(validation_names),
        "selected_view_policy": "Choose the annotator view maximizing (annotation count, summed polygon area), preserving first COCO-view order on ties; one view per observation.",
        "selected_view_manifest": str(source_selection_csv.relative_to(REPO_ROOT)),
        "selected_view_manifest_sha256": digest(source_selection_csv),
        "yolo_export": {
            **export["counts"],
            "classes": export["classes"],
            "training_images_are_hardlinked_when_supported": True,
            "dataset_yaml": str(export["data_yaml"].relative_to(REPO_ROOT)),
        },
        "detector_config": {
            "model": "YOLO11s-seg COCO initialization",
            "epochs": 30,
            "image_size": SOURCE_CONFIG["image_size"],
            "batch_size": args.detector_batch_size,
            "workers": args.workers,
            "patience": SOURCE_CONFIG["patience"],
            "seed": 42,
            "augmentation": "Source recipe: mosaic .3, degrees 10, scale .3, HSV value .3; horizontal/vertical flips and hue/saturation disabled.",
        },
        "detector_checkpoint": {"path": str(detector_path.relative_to(REPO_ROOT)), "sha256": digest(detector_path)},
        "detector_training_seconds": detector_training_seconds,
        "refiner_config": {
            "architecture": "ResNet18 U-Net, ImageNet-1K initialization",
            "crop_size": 256,
            "context": 1.8,
            "minimum_crop": 96,
            "epochs": 8,
            "batch_size": args.refiner_batch_size,
            "optimizer": "AdamW(lr=3e-4, weight_decay=1e-4)",
            "loss": "0.5 BCEWithLogits + 0.5 soft Dice",
            "selection_metric": "mask Dice at probability threshold 0.65 on per-instance crops from the frozen holdout",
        },
        "refiner_result": refiner_result,
        "refiner_training_seconds_including_initialization": refiner_training_seconds,
        "scorer": {
            "name": "EquiNexus port of the official organizer scorer",
            "iou_threshold": 0.5,
            "diagnostic_overlap_threshold": 0.0,
            "prediction_confidence": SOURCE_CONFIG["confidence_threshold"],
            "minimum_mask_area": SOURCE_CONFIG["minimum_component_size"],
            "all_annotator_views_scored": True,
        },
        "official_pq": metrics,
        "per_observation": [
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
            for row in observation_rows
        ],
        "validation_predictions": {"path": str(prediction_path.relative_to(REPO_ROOT)), "sha256": digest(prediction_path)},
        "inference_seconds": inference_seconds,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "training_peak_gpu_memory_bytes": training_peak_gpu_memory_bytes,
        "device": torch.cuda.get_device_name(0),
        "hardware": hardware_info(),
        "python": platform.python_version(),
        "package_versions": package_versions(),
        "working_tree_source_commit": subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "runner_sha256": digest(Path(__file__).resolve()),
        "holdout_used_for_gradient_updates": False,
        "holdout_used_for_detector_checkpoint_selection": True,
        "holdout_used_for_refiner_checkpoint_selection": True,
        "holdout_used_for_official_pq_screening": True,
        "holdout_interpretation": "Candidate-screening result because checkpoint selection and model assessment use the same frozen holdout.",
        "training_ground_truth": "Competition-provided training annotations only; one selected annotator view per training observation.",
        "test_ground_truth_used": False,
    }
    metadata_path = output_dir / "run-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "event": "complete",
        "official_pq": metrics["global"]["pq"],
        "tp": metrics["global"]["tp"],
        "fp": metrics["global"]["fp"],
        "fn": metrics["global"]["fn"],
        "training_seconds": detector_training_seconds + refiner_training_seconds,
        "inference_seconds": inference_seconds,
        "peak_gpu_memory_bytes": metadata["peak_gpu_memory_bytes"],
        "metadata": str(metadata_path),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
