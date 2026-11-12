# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Train and evaluate a high-resolution, semantic U-Net with grouped observations.

The model is an independent implementation of a public native-resolution U-Net idea;
it uses only competition training annotations and timm's Apache-2.0 ImageNet encoder.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TORCH_HOME", str(REPO_ROOT / ".cache" / "torch"))
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".cache" / "matplotlib"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from pycocotools import mask as mask_utils
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

from filament_baseline.data import CompetitionDataset, load_dataset
from filament_baseline.predictor import PredictedInstance, encode_label_components
from filament_baseline.runtime import git_info, hardware_info, package_versions
from filament_baseline.scorer import annotation_to_rle, score_validation
from filament_baseline.split import create_observation_split


EXPECTED_SPLIT_SHA256 = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"
FULL_SIZE = 2048
CROP_SIZE = 1888
CROP_OFFSET = (FULL_SIZE - CROP_SIZE) // 2
NORM_RADIUS = 900
ENCODER_NAME = "efficientvit_b2.r288_in1k"
DECODER_CHANNELS = (192, 96, 48, 24, 12)
EPOCHS_DEFAULT = 24
TTA_DEFAULT = 8
SEED = 42
_Y, _X = np.ogrid[:CROP_SIZE, :CROP_SIZE]
_CROP_CENTER = (CROP_SIZE - 1) / 2.0
_NORM_DISK = (_Y - _CROP_CENTER) ** 2 + (_X - _CROP_CENTER) ** 2 <= NORM_RADIUS**2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_device(device: torch.device) -> dict[str, Any]:
    info = hardware_info()
    if device.type == "cuda":
        info["selected_device"] = torch.cuda.get_device_name(device)
        info["cuda_device_index"] = device.index or 0
        info["cuda_total_memory_bytes"] = int(
            torch.cuda.get_device_properties(device).total_memory
        )
    else:
        info["selected_device"] = str(device)
    return info


def centre_crop(array: np.ndarray) -> np.ndarray:
    return array[
        CROP_OFFSET : CROP_OFFSET + CROP_SIZE,
        CROP_OFFSET : CROP_OFFSET + CROP_SIZE,
    ]


def intensity_stats(crop: np.ndarray) -> tuple[float, float]:
    hist = np.bincount(crop[_NORM_DISK], minlength=256).astype(np.float64)
    cdf = np.cumsum(hist) / max(float(hist.sum()), 1.0)
    low, median, high = np.searchsorted(cdf, (0.16, 0.50, 0.84))
    return float(median), float(max(int(high) - int(low), 1))


def cache_paths(output_dir: Path) -> tuple[Path, Path, Path, Path]:
    cache_dir = output_dir / "cache"
    return (
        cache_dir / "images.npy",
        cache_dir / "union_masks.npy",
        cache_dir / "intensity_stats.npy",
        cache_dir / "cache-index.json",
    )


def build_training_cache(
    dataset: CompetitionDataset,
    names: list[str],
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_path, mask_path, stats_path, index_path = cache_paths(output_dir)
    expected_names = sorted(names)
    if index_path.is_file() and image_path.is_file() and mask_path.is_file() and stats_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if index.get("names") != expected_names:
            raise RuntimeError("Existing cache belongs to a different observation set")
        if index.get("split_sha256") != EXPECTED_SPLIT_SHA256:
            raise RuntimeError("Existing cache split digest differs from the frozen split")
        images = np.load(image_path, mmap_mode="r")
        masks = np.load(mask_path, mmap_mode="r")
        stats = np.load(stats_path, mmap_mode="r")
        expected_shape = (len(expected_names), CROP_SIZE, CROP_SIZE)
        if images.shape != expected_shape or masks.shape != expected_shape or stats.shape != (len(names), 2):
            raise RuntimeError("Existing cache has an unexpected shape")
        print(f"[cache] reused {len(expected_names)} training observations", flush=True)
        return images, masks, stats

    for path in (image_path, mask_path, stats_path, index_path):
        if path.exists():
            raise FileExistsError(f"Incomplete cache exists; refusing to overwrite {path}")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    images = np.lib.format.open_memmap(
        image_path, mode="w+", dtype=np.uint8, shape=(len(expected_names), CROP_SIZE, CROP_SIZE)
    )
    masks = np.lib.format.open_memmap(
        mask_path, mode="w+", dtype=np.uint8, shape=(len(expected_names), CROP_SIZE, CROP_SIZE)
    )
    stats = np.lib.format.open_memmap(
        stats_path, mode="w+", dtype=np.float32, shape=(len(expected_names), 2)
    )
    raw_crop_pixels = 0
    dropped_mask_pixels = 0
    started = time.perf_counter()
    print(f"[cache] rasterizing {len(expected_names)} training-only observation masks", flush=True)

    for index, file_name in enumerate(expected_names):
        image = cv2.imread(str(dataset.train_image_dir / file_name), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (FULL_SIZE, FULL_SIZE):
            raise ValueError(f"Expected a decodable {FULL_SIZE}x{FULL_SIZE} image: {file_name}")
        union = np.zeros((FULL_SIZE, FULL_SIZE), dtype=np.uint8)
        for view in dataset.views_by_observation[file_name]:
            for annotation in dataset.annotations_by_image_id[view.image_id]:
                rle = annotation_to_rle(annotation, view.height, view.width)
                decoded = mask_utils.decode(rle)
                np.maximum(union, decoded, out=union)
                del decoded
        mask_crop = centre_crop(union)
        dropped_mask_pixels += int(union.sum()) - int(mask_crop.sum())
        image_crop = centre_crop(image)
        images[index] = image_crop
        masks[index] = mask_crop
        stats[index] = intensity_stats(image_crop)
        raw_crop_pixels += int(mask_crop.sum())
        if (index + 1) % 50 == 0 or index + 1 == len(expected_names):
            print(f"[cache] {index + 1}/{len(expected_names)}", flush=True)
        del image, union, mask_crop, image_crop

    images.flush()
    masks.flush()
    stats.flush()
    cache_index = {
        "names": expected_names,
        "training_observations": len(expected_names),
        "split_sha256": EXPECTED_SPLIT_SHA256,
        "image_shape": [len(expected_names), CROP_SIZE, CROP_SIZE],
        "image_dtype": "uint8",
        "mask_dtype": "uint8",
        "supervision": "pixelwise union of all competition annotator masks for each observation",
        "dropped_ground_truth_pixels_outside_center_crop": dropped_mask_pixels,
        "kept_ground_truth_pixels": raw_crop_pixels,
        "cache_seconds": round(time.perf_counter() - started, 3),
    }
    index_path.write_text(json.dumps(cache_index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"[cache] ready in {cache_index['cache_seconds']:.1f}s; "
        f"crop-dropped foreground pixels={dropped_mask_pixels:,}",
        flush=True,
    )
    return np.load(image_path, mmap_mode="r"), np.load(mask_path, mmap_mode="r"), np.load(stats_path, mmap_mode="r")


class ObservationDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        masks: np.ndarray,
        stats: np.ndarray,
        indices: list[int],
        *,
        augment: bool,
        seed: int,
    ) -> None:
        self.images = images
        self.masks = masks
        self.stats = stats
        self.indices = indices
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = self.indices[item]
        image = np.array(self.images[index], copy=True)
        mask = np.array(self.masks[index], copy=True)
        if self.augment:
            k = int(self.rng.integers(0, 4))
            image = np.rot90(image, k)
            mask = np.rot90(mask, k)
            if self.rng.random() < 0.5:
                image = image[:, ::-1]
                mask = mask[:, ::-1]
            if self.rng.random() < 0.5:
                image = image[::-1, :]
                mask = mask[::-1, :]

        centre, scale = (float(value) for value in self.stats[index])
        image_float = image.astype(np.float32)
        if self.augment:
            image_float = image_float * float(self.rng.uniform(0.8, 1.2))
            image_float += float(self.rng.uniform(-12.0, 12.0))
        image_float = np.ascontiguousarray((image_float - centre) / scale)
        mask_float = np.ascontiguousarray(mask.astype(np.float32))
        return torch.from_numpy(image_float[None]), torch.from_numpy(mask_float[None])


class ConvBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_out, channels_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ContextBlock(nn.Module):
    def __init__(self, channels: int, inner_channels: int = 192) -> None:
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(channels, inner_channels, 1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
        )
        self.dilated = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(inner_channels, inner_channels, 3, padding=d, dilation=d, bias=False),
                    nn.BatchNorm2d(inner_channels),
                    nn.ReLU(inplace=True),
                )
                for d in (1, 2, 4)
            ]
        )
        self.expand = nn.Sequential(
            nn.Conv2d(inner_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        nn.init.zeros_(self.expand[1].weight)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.expand(self.dilated(self.reduce(x))))


class NativeResolutionUNet(nn.Module):
    def __init__(self, *, pretrained: bool) -> None:
        super().__init__()
        self.encoder = timm.create_model(
            ENCODER_NAME,
            pretrained=pretrained,
            features_only=True,
            in_chans=1,
        )
        channels = self.encoder.feature_info.channels()
        reductions = self.encoder.feature_info.reduction()
        channels_by_reduction = dict(zip(reductions, channels, strict=True))
        deepest = max(reductions)
        input_channels = channels[-1]
        blocks: list[nn.Module] = []
        self.skip_reductions: list[int | None] = []
        for level in range(int(math.log2(deepest))):
            reduction = deepest // (2 ** (level + 1))
            skip_channels = channels_by_reduction.get(reduction, 0)
            output_channels = DECODER_CHANNELS[min(level, len(DECODER_CHANNELS) - 1)]
            blocks.append(ConvBlock(input_channels + skip_channels, output_channels))
            self.skip_reductions.append(reduction if skip_channels else None)
            input_channels = output_channels
        self.blocks = nn.ModuleList(blocks)
        self.context = ContextBlock(channels[-1])
        self.head = nn.Conv2d(input_channels, 1, kernel_size=1)
        self.reductions = reductions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        pad_h, pad_w = (-height) % 32, (-width) % 32
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        features = self.encoder(x)
        feature_by_reduction = dict(zip(self.reductions, features, strict=True))
        output = self.context(features[-1])
        for block, reduction in zip(self.blocks, self.skip_reductions, strict=True):
            output = F.interpolate(output, scale_factor=2, mode="nearest")
            if reduction is not None:
                output = torch.cat((output, feature_by_reduction[reduction]), dim=1)
            output = block(output)
        output = self.head(output)
        return output[..., :height, :width]


def make_optimizer(model: NativeResolutionUNet) -> torch.optim.Optimizer:
    modules = (
        ("encoder", model.encoder, 1e-4),
        ("context", model.context, 5e-4),
        ("decoder", model.blocks, 5e-4),
        ("head", model.head, 5e-4),
    )
    groups = []
    for label, module, learning_rate in modules:
        decay = [parameter for parameter in module.parameters() if parameter.requires_grad and parameter.ndim > 1]
        no_decay = [parameter for parameter in module.parameters() if parameter.requires_grad and parameter.ndim <= 1]
        if decay:
            groups.append({"params": decay, "lr": learning_rate, "weight_decay": 0.02, "name": label})
        if no_decay:
            groups.append({"params": no_decay, "lr": learning_rate, "weight_decay": 0.0, "name": label})
    return torch.optim.AdamW(groups)


def binary_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    target = target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = torch.sigmoid(logits).flatten(1)
    target_flat = target.flatten(1)
    intersection = (probability * target_flat).sum(dim=1)
    dice = 1.0 - (2.0 * intersection + 1e-6) / (
        probability.sum(dim=1) + target_flat.sum(dim=1) + 1e-6
    )
    return 0.45 * bce + 0.55 * dice.mean()


def selected_epoch_checkpoints(epochs: int) -> tuple[int, ...]:
    candidates = [epoch for epoch in (16, 20, 24) if epoch <= epochs]
    if epochs not in candidates:
        candidates.append(epochs)
    return tuple(sorted(set(candidates)))


def epoch_checkpoint_path(last_checkpoint: Path, epoch: int) -> Path:
    return last_checkpoint.with_name(f"checkpoint-epoch-{epoch:02d}.pt")


def train_model(
    images: np.ndarray,
    masks: np.ndarray,
    stats: np.ndarray,
    names: list[str],
    train_indices: list[int],
    *,
    epochs: int,
    batch_size: int,
    workers: int,
    device: torch.device,
    checkpoint_path: Path,
    fold_label: str,
) -> tuple[NativeResolutionUNet, dict[str, Any]]:
    seed_everything(SEED + (int(fold_label) if fold_label.isdigit() else 1000))
    ordered_names = [names[index] for index in train_indices]
    snapshot_epochs = selected_epoch_checkpoints(epochs)
    snapshot_paths = {
        epoch: epoch_checkpoint_path(checkpoint_path, epoch)
        for epoch in snapshot_epochs
    }
    if checkpoint_path.is_file():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        saved_config = saved.get("config", {})
        if (
            int(saved.get("epoch", 0)) == epochs
            and saved.get("fold") == fold_label
            and saved.get("training_observations") == ordered_names
            and saved_config.get("encoder") == ENCODER_NAME
            and int(saved_config.get("crop_size", -1)) == CROP_SIZE
            and int(saved_config.get("epochs", -1)) == epochs
            and int(saved_config.get("batch_size", -1)) == batch_size
            and all(path.is_file() for path in snapshot_paths.values())
        ):
            model = NativeResolutionUNet(pretrained=False).to(device)
            model.load_state_dict(saved["state_dict"], strict=True)
            model = model.to(memory_format=torch.channels_last)
            runtime = {
                "training_seconds": None,
                "history": [],
                "checkpoint": str(checkpoint_path.resolve().relative_to(REPO_ROOT)),
                "checkpoint_sha256": sha256(checkpoint_path),
                "peak_gpu_memory_bytes": 0,
                "reused_completed_checkpoint": True,
                "snapshot_epochs": list(snapshot_epochs),
                "snapshot_paths": {
                    str(epoch): str(path.resolve().relative_to(REPO_ROOT))
                    for epoch, path in snapshot_paths.items()
                },
            }
            print(f"[train:{fold_label}] reusing a completed checkpoint", flush=True)
            return model, runtime

    dataset = ObservationDataset(
        images,
        masks,
        stats,
        train_indices,
        augment=True,
        seed=SEED + (int(fold_label) if fold_label.isdigit() else 1000),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=workers > 0,
    )
    model = NativeResolutionUNet(pretrained=True).to(device)
    model = model.to(memory_format=torch.channels_last)
    optimizer = make_optimizer(model)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    accumulation = 4
    steps_per_epoch = math.ceil(len(loader) / accumulation)
    warmup_steps = max(steps_per_epoch * 2, 1)
    total_steps = max(steps_per_epoch * epochs, 1)

    def lr_factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    print(
        f"[train:{fold_label}] {len(train_indices)} observations, {len(loader)} steps/epoch, "
        f"batch={batch_size}, accumulation={accumulation}, epochs={epochs}",
        flush=True,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        epoch_started = time.perf_counter()
        for step, (input_image, target) in enumerate(loader, start=1):
            input_image = input_image.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            target = target.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(input_image)
                loss = binary_loss(logits, target)
            scaler.scale(loss / accumulation).backward()
            if step % accumulation == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
            loss_sum += float(loss.detach())
            if step % 50 == 0 or step == len(loader):
                print(
                    f"[train:{fold_label}] epoch={epoch}/{epochs} step={step}/{len(loader)} "
                    f"mean_loss={loss_sum / step:.5f}",
                    flush=True,
                )
        epoch_row = {
            "epoch": epoch,
            "mean_training_loss": loss_sum / max(len(loader), 1),
            "epoch_seconds": round(time.perf_counter() - epoch_started, 3),
            "optimizer_steps": global_step,
            "learning_rates": {group.get("name", str(i)): float(group["lr"]) for i, group in enumerate(optimizer.param_groups)},
        }
        history.append(epoch_row)
        checkpoint = {
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "fold": fold_label,
            "training_observations": ordered_names,
            "config": {
                "encoder": ENCODER_NAME,
                "crop_size": CROP_SIZE,
                "epochs": epochs,
                "batch_size": batch_size,
                "accumulation": accumulation,
            },
        }
        torch.save(checkpoint, checkpoint_path)
        if epoch in snapshot_paths:
            torch.save(checkpoint, snapshot_paths[epoch])
        print(json.dumps({"event": "epoch", "fold": fold_label, **epoch_row}), flush=True)
    runtime = {
        "training_seconds": round(time.perf_counter() - started, 3),
        "history": history,
        "checkpoint": str(checkpoint_path.resolve().relative_to(REPO_ROOT)),
        "checkpoint_sha256": sha256(checkpoint_path),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "reused_completed_checkpoint": False,
        "snapshot_epochs": list(snapshot_epochs),
        "snapshot_paths": {
            str(epoch): str(path.resolve().relative_to(REPO_ROOT))
            for epoch, path in snapshot_paths.items()
        },
    }
    return model, runtime


@torch.inference_mode()
def predict_crop(
    model: NativeResolutionUNet,
    image_crop: np.ndarray,
    stats: tuple[float, float],
    *,
    device: torch.device,
    tta_views: int,
) -> np.ndarray:
    centre, scale = stats
    array = np.ascontiguousarray((image_crop.astype(np.float32) - centre) / scale)
    tensor = torch.from_numpy(array[None, None]).to(device)
    tensor = tensor.to(memory_format=torch.channels_last)
    probability = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
    view_count = 0
    for flip in range(2 if tta_views == 8 else 1):
        source = torch.flip(tensor, dims=(-1,)) if flip else tensor
        for rotation in range(4 if tta_views in (4, 8) else (2 if tta_views == 2 else 1)):
            transformed = torch.rot90(source, rotation, dims=(-2, -1))
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = torch.sigmoid(model(transformed.contiguous(memory_format=torch.channels_last)))
            output = torch.rot90(output, -rotation, dims=(-2, -1))
            if flip:
                output = torch.flip(output, dims=(-1,))
            probability += output[0, 0].float().cpu().numpy()
            view_count += 1
    probability /= max(view_count, 1)
    return probability.astype(np.float16)


def predict_observations(
    model: NativeResolutionUNet,
    dataset: CompetitionDataset,
    names: list[str],
    images: np.ndarray | None,
    stats: np.ndarray | None,
    cache_names: list[str] | None,
    *,
    device: torch.device,
    tta_views: int,
) -> dict[str, np.ndarray]:
    model.eval()
    cache_index = {name: index for index, name in enumerate(cache_names or [])}
    output: dict[str, np.ndarray] = {}
    started = time.perf_counter()
    for position, file_name in enumerate(sorted(names), start=1):
        if file_name in cache_index and images is not None and stats is not None:
            image_crop = np.asarray(images[cache_index[file_name]])
            image_stats = tuple(float(value) for value in stats[cache_index[file_name]])
        else:
            full = cv2.imread(str(dataset.train_image_dir / file_name), cv2.IMREAD_GRAYSCALE)
            if full is None or full.shape != (FULL_SIZE, FULL_SIZE):
                raise ValueError(f"Could not decode validation image: {file_name}")
            image_crop = centre_crop(full)
            image_stats = intensity_stats(image_crop)
        output[file_name] = predict_crop(
            model, image_crop, image_stats, device=device, tta_views=tta_views
        )
        if position % 10 == 0 or position == len(names):
            print(f"[predict] {position}/{len(names)} observations", flush=True)
    print(f"[predict] done in {time.perf_counter() - started:.1f}s", flush=True)
    return output


def components_to_instances(
    probability: np.ndarray,
    file_name: str,
    *,
    threshold: float,
    min_area: int,
    closing_kernel: int,
) -> list[PredictedInstance]:
    binary = (probability.astype(np.float32) > threshold).astype(np.uint8)
    if closing_kernel:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (closing_kernel, closing_kernel)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    kept_labels = [
        component_id
        for component_id in range(1, count)
        if int(stats[component_id, cv2.CC_STAT_AREA]) >= min_area
    ]
    if not kept_labels:
        return []
    remap = np.zeros(count, dtype=np.int32)
    remap[kept_labels] = np.arange(1, len(kept_labels) + 1, dtype=np.int32)
    full_labels = np.zeros((FULL_SIZE, FULL_SIZE), dtype=np.int32)
    crop_labels = remap[labels]
    full_labels[
        CROP_OFFSET : CROP_OFFSET + CROP_SIZE,
        CROP_OFFSET : CROP_OFFSET + CROP_SIZE,
    ] = crop_labels
    encoded = encode_label_components(full_labels, list(range(1, len(kept_labels) + 1)))
    instances: list[PredictedInstance] = []
    stem = Path(file_name).stem
    for label_id, component_id in enumerate(kept_labels, start=1):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        instances.append(
            PredictedInstance(
                filament_id=f"{stem}_{label_id}",
                image_stem=stem,
                area=area,
                rle=encoded[label_id],
            )
        )
    return instances


def predictions_for_config(
    probabilities: dict[str, np.ndarray],
    names: list[str],
    *,
    threshold: float,
    min_area: int,
    closing_kernel: int,
) -> dict[str, list[PredictedInstance]]:
    return {
        file_name: components_to_instances(
            probabilities[file_name],
            file_name,
            threshold=threshold,
            min_area=min_area,
            closing_kernel=closing_kernel,
        )
        for file_name in names
    }


def load_dataset_split(data_root: Path, split_dir: Path) -> tuple[CompetitionDataset, Any]:
    dataset = load_dataset(data_root)
    split = create_observation_split(dataset, split_dir, seed=SEED, validation_fraction=0.2)
    if split.sha256 != EXPECTED_SPLIT_SHA256:
        raise RuntimeError(f"Frozen split digest changed: {split.sha256}")
    if set(split.training_names) & set(split.validation_names):
        raise AssertionError("Frozen observation groups crossed partitions")
    if len(split.training_names) != 565 or len(split.validation_names) != 142:
        raise RuntimeError("Unexpected frozen split sizes")
    return dataset, split


def create_oof_fold_map(dataset: CompetitionDataset, training_names: list[str], path: Path) -> dict[str, int]:
    years = [dataset.views_by_observation[name][0].year for name in training_names]
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    assignments: dict[str, int] = {}
    for fold, (_, validation_indices) in enumerate(splitter.split(training_names, years)):
        for index in validation_indices:
            assignments[training_names[int(index)]] = fold
    if set(assignments) != set(training_names):
        raise AssertionError("OOF fold map does not cover each training observation once")
    rows = [
        {
            "file_name": name,
            "stem": Path(name).stem,
            "year": dataset.views_by_observation[name][0].year,
            "site": dataset.views_by_observation[name][0].site,
            "fold": assignments[name],
        }
        for name in sorted(assignments)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False, lineterminator="\n")
    for fold in range(5):
        if not any(value == fold for value in assignments.values()):
            raise AssertionError(f"OOF fold {fold} is empty")
    return assignments


def run_oof(args: argparse.Namespace, dataset: CompetitionDataset, split: Any, device: torch.device) -> dict[str, Any]:
    output_dir = args.output_dir
    training_names = list(split.training_names)
    images, masks, stats = build_training_cache(dataset, training_names, output_dir)
    index_path = cache_paths(output_dir)[3]
    cache_names = json.loads(index_path.read_text(encoding="utf-8"))["names"]
    name_to_index = {name: index for index, name in enumerate(cache_names)}
    fold_map_path = output_dir / "oof-folds.csv"
    fold_assignments = create_oof_fold_map(dataset, training_names, fold_map_path)
    expected_probability_shape = (len(cache_names), CROP_SIZE, CROP_SIZE)
    candidate_epochs = selected_epoch_checkpoints(args.epochs)
    probability_maps: dict[int, np.ndarray] = {}
    probability_paths: dict[int, Path] = {}
    for epoch in candidate_epochs:
        probability_path = output_dir / f"oof-probabilities-epoch{epoch:02d}.npy"
        probability_paths[epoch] = probability_path
        if probability_path.is_file():
            probability_maps[epoch] = np.load(probability_path, mmap_mode="r+")
            if (
                probability_maps[epoch].shape != expected_probability_shape
                or probability_maps[epoch].dtype != np.float16
            ):
                raise RuntimeError(f"OOF probability map has an unexpected shape or dtype: {probability_path}")
        else:
            probability_maps[epoch] = np.lib.format.open_memmap(
                probability_path,
                mode="w+",
                dtype=np.float16,
                shape=expected_probability_shape,
            )
    completed_folds: list[dict[str, Any]] = []
    started = time.perf_counter()
    for fold in range(5):
        fold_dir = output_dir / f"fold-{fold}"
        completion_path = fold_dir / "fold-complete.json"
        checkpoint_path = fold_dir / "last-checkpoint.pt"
        validation_names = sorted(name for name, value in fold_assignments.items() if value == fold)
        train_names = sorted(name for name, value in fold_assignments.items() if value != fold)
        if completion_path.is_file():
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            if (
                completion.get("validation_names") != validation_names
                or completion.get("snapshot_epochs") != list(candidate_epochs)
            ):
                raise RuntimeError(f"Fold {fold} completion manifest does not match current folds")
            completed_folds.append(completion)
            print(f"[oof] reused completed fold {fold}", flush=True)
            continue
        train_indices = [name_to_index[name] for name in train_names]
        model, training = train_model(
            images,
            masks,
            stats,
            cache_names,
            train_indices,
            epochs=args.epochs,
            batch_size=args.batch_size,
            workers=args.workers,
            device=device,
            checkpoint_path=checkpoint_path,
            fold_label=str(fold),
        )
        for epoch in candidate_epochs:
            snapshot_path = epoch_checkpoint_path(checkpoint_path, epoch)
            if not snapshot_path.is_file():
                raise FileNotFoundError(f"Missing epoch snapshot required for OOF selection: {snapshot_path}")
            saved = torch.load(snapshot_path, map_location="cpu", weights_only=True)
            model.load_state_dict(saved["state_dict"], strict=True)
            model.eval()
            for position, name in enumerate(validation_names, start=1):
                cache_index = name_to_index[name]
                predicted_probability = predict_crop(
                    model,
                    np.asarray(images[cache_index]),
                    tuple(float(value) for value in stats[cache_index]),
                    device=device,
                    tta_views=args.tta_views,
                )
                if not np.isfinite(predicted_probability).all():
                    raise RuntimeError(f"Non-finite OOF probability for {name} at epoch {epoch}")
                probability_maps[epoch][cache_index] = predicted_probability
                if position % 10 == 0 or position == len(validation_names):
                    print(
                        f"[oof] epoch {epoch}, fold {fold}: {position}/{len(validation_names)}",
                        flush=True,
                    )
            probability_maps[epoch].flush()
        completion = {
            "fold": fold,
            "train_observations": len(train_names),
            "validation_observations": len(validation_names),
            "train_names": train_names,
            "validation_names": validation_names,
            "training": training,
            "snapshot_epochs": list(candidate_epochs),
            "snapshot_paths": training["snapshot_paths"],
            "oof_prediction_files": {
                str(epoch): str(path.relative_to(REPO_ROOT))
                for epoch, path in probability_paths.items()
            },
        }
        completion_path.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        completed_folds.append(completion)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for probability_map in probability_maps.values():
        probability_map.flush()
    if len(completed_folds) != 5:
        raise RuntimeError(f"Only {len(completed_folds)} of five OOF folds completed")
    grid = [
        (threshold, min_area, closing)
        for threshold in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
        for min_area in (50, 150, 250, 400)
        for closing in (0, 5, 9)
    ]
    sweep_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    total_settings = len(grid) * len(candidate_epochs)
    evaluated = 0
    for epoch in candidate_epochs:
        probability_by_name = {
            name: np.asarray(probability_maps[epoch][name_to_index[name]])
            for name in training_names
        }
        for threshold, min_area, closing in grid:
            predictions = predictions_for_config(
                probability_by_name,
                training_names,
                threshold=threshold,
                min_area=min_area,
                closing_kernel=closing,
            )
            metrics, _ = score_validation(dataset, predictions, training_names)
            row = {
                "epoch": epoch,
                "threshold": threshold,
                "min_area": min_area,
                "closing_kernel": closing,
                **metrics["global"],
            }
            sweep_rows.append(row)
            if best is None or float(row["pq"]) > float(best["pq"]):
                best = row
            evaluated += 1
            if evaluated % 10 == 0 or evaluated == total_settings:
                print(
                    f"[oof] postprocess sweep {evaluated}/{total_settings}; "
                    f"best PQ={best['pq']:.6f} at epoch {best['epoch']}",
                    flush=True,
                )
            del predictions
        del probability_by_name
    sweep = pd.DataFrame(sweep_rows).sort_values("pq", ascending=False, kind="stable")
    sweep.to_csv(output_dir / "oof-postprocess-sweep.csv", index=False, lineterminator="\n")
    result = {
        "split_sha256": split.sha256,
        "partition": "cross-fitted frozen training partition; each observation excluded from its model fit",
        "training_observations": len(training_names),
        "annotator_views": sum(len(dataset.views_by_observation[name]) for name in training_names),
        "fold_manifest": str(fold_map_path.relative_to(REPO_ROOT)),
        "fold_manifest_sha256": sha256(fold_map_path),
        "folds": completed_folds,
        "tta_views": args.tta_views,
        "prediction_files": {
            str(epoch): str(path.relative_to(REPO_ROOT))
            for epoch, path in probability_paths.items()
        },
        "probability_map_dtype": "float16 on a 0-1 scale",
        "selected_oof_parameters": best,
        "candidate_epochs": list(candidate_epochs),
        "sweep_rows": total_settings,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "device": capture_device(device),
        "package_versions": {**package_versions(), "timm": importlib.metadata.version("timm")},
        "source": git_info(REPO_ROOT),
        "source_commit": git_commit(),
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    (output_dir / "oof-summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "oof_complete", "selected": best, "runtime_seconds": result["runtime_seconds"]}, flush=True), flush=True)
    return result


def run_holdout(args: argparse.Namespace, dataset: CompetitionDataset, split: Any, device: torch.device) -> dict[str, Any]:
    output_dir = args.output_dir
    oof_summary_path = output_dir / "oof-summary.json"
    if not oof_summary_path.is_file():
        raise FileNotFoundError("Run the OOF partition first to select postprocessing parameters")
    oof_summary = json.loads(oof_summary_path.read_text(encoding="utf-8"))
    oof_metadata_path = output_dir / "run-metadata-oof.json"
    if not oof_metadata_path.is_file():
        raise FileNotFoundError("OOF run metadata is missing")
    oof_config = json.loads(oof_metadata_path.read_text(encoding="utf-8"))["config"]
    if oof_config["epochs"] != args.epochs or oof_config["tta_views"] != args.tta_views:
        raise RuntimeError("Holdout configuration must match the OOF-selected training and TTA settings")
    selected = oof_summary["selected_oof_parameters"]
    training_names = list(split.training_names)
    holdout_names = list(split.validation_names)
    images, masks, stats = build_training_cache(dataset, training_names, output_dir)
    cache_names = json.loads(cache_paths(output_dir)[3].read_text(encoding="utf-8"))["names"]
    indices = list(range(len(cache_names)))
    checkpoint_path = output_dir / "frozen-train-full" / "last-checkpoint.pt"
    model, training = train_model(
        images,
        masks,
        stats,
        cache_names,
        indices,
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
        checkpoint_path=checkpoint_path,
        fold_label="full",
    )
    selected_epoch = int(selected["epoch"])
    selected_checkpoint = epoch_checkpoint_path(checkpoint_path, selected_epoch)
    if not selected_checkpoint.is_file():
        raise FileNotFoundError(f"Missing OOF-selected holdout checkpoint: {selected_checkpoint}")
    selected_state = torch.load(selected_checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(selected_state["state_dict"], strict=True)
    probabilities = predict_observations(
        model,
        dataset,
        holdout_names,
        None,
        None,
        None,
        device=device,
        tta_views=args.tta_views,
    )
    predictions = predictions_for_config(
        probabilities,
        holdout_names,
        threshold=float(selected["threshold"]),
        min_area=int(selected["min_area"]),
        closing_kernel=int(selected["closing_kernel"]),
    )
    metrics, counts = score_validation(dataset, predictions, holdout_names)
    observation_rows = [
        {
            "file_name": row.file_name,
            "year": row.year,
            "site": row.site,
            "annotator_views": row.annotator_views,
            "pq": row.pq,
            "tp": row.tp,
            "fp": row.fp,
            "fn": row.fn,
            "tp_iou_sum": row.tp_iou_sum,
            "gt_count": row.gt_count,
            "prediction_count": row.prediction_count,
        }
        for row in counts
    ]
    pd.DataFrame(observation_rows).to_csv(
        output_dir / "holdout-per-observation.csv", index=False, lineterminator="\n"
    )
    metrics_path = output_dir / "holdout-metrics.json"
    result = {
        "split_sha256": split.sha256,
        "training_observations": len(training_names),
        "holdout_observations": len(holdout_names),
        "holdout_annotator_views": sum(len(dataset.views_by_observation[name]) for name in holdout_names),
        "postprocessing_selected_from_oof": selected,
        "model_epoch_selected_from_oof": selected_epoch,
        "selected_checkpoint": str(selected_checkpoint.relative_to(REPO_ROOT)),
        "official_scorer_metrics": metrics,
        "training": training,
        "tta_views": args.tta_views,
        "device": capture_device(device),
        "package_versions": {**package_versions(), "timm": importlib.metadata.version("timm")},
        "source": git_info(REPO_ROOT),
        "source_commit": git_commit(),
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "holdout_complete", "global": metrics["global"]}, flush=True), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("oof", "holdout"), required=True)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--split-dir", type=Path, default=REPO_ROOT / "data" / "splits")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "model-experiments" / "native-unet-grouped-20261111")
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    parser.add_argument("--batch-size", type=int, choices=(1,), default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--tta-views", type=int, choices=(1, 2, 4, 8), default=TTA_DEFAULT)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.workers < 0:
        parser.error("--epochs must be positive and --workers nonnegative")
    data_root = args.data_root.expanduser().resolve()
    split_dir = args.split_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not output_dir.is_relative_to((REPO_ROOT / "artifacts" / "model-experiments").resolve()):
        parser.error("--output-dir must stay inside artifacts/model-experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Competition data folder not found: {data_root}")
    workspace_data = (REPO_ROOT / "data").resolve()
    if not data_root.is_relative_to(workspace_data):
        if os.name == "nt" or not data_root.is_relative_to(Path("/kaggle/input").resolve()):
            raise ValueError("Data must be under this workspace's data/ folder or Kaggle's read-only input mount")
    if not split_dir.is_relative_to((REPO_ROOT / "data" / "splits").resolve()):
        raise ValueError("Split manifests must remain under this workspace's data/splits folder")
    args.data_root = data_root
    args.split_dir = split_dir
    args.output_dir = output_dir
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
    seed_everything(SEED)
    cv2.setNumThreads(0)
    dataset, split = load_dataset_split(data_root, split_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_metadata = {
        "mode": args.mode,
        "config": {
            "epochs": args.epochs,
            "epoch_checkpoints": list(selected_epoch_checkpoints(args.epochs)),
            "batch_size": args.batch_size,
            "workers": args.workers,
            "tta_views": args.tta_views,
            "encoder": ENCODER_NAME,
            "crop_size": CROP_SIZE,
            "crop_offset": CROP_OFFSET,
            "seed": SEED,
            "split_sha256": split.sha256,
            "annotation_supervision": "all competition annotator masks unioned by original observation",
        },
        "dataset": {
            "root": str(dataset.root.relative_to(REPO_ROOT)) if dataset.root.is_relative_to(REPO_ROOT) else str(dataset.root),
            "training_observations": len(split.training_names),
            "holdout_observations": len(split.validation_names),
            "training_annotator_views": sum(len(dataset.views_by_observation[name]) for name in split.training_names),
            "holdout_annotator_views": sum(len(dataset.views_by_observation[name]) for name in split.validation_names),
        },
        "source_commit": git_commit(),
        "source_script_sha256": sha256(Path(__file__).resolve()),
        "device": capture_device(device),
        "package_versions": {
            **package_versions(),
            "timm": importlib.metadata.version("timm"),
        },
        "source": git_info(REPO_ROOT),
    }
    metadata_path = output_dir / f"run-metadata-{args.mode}.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("config") != run_metadata["config"]:
            raise FileExistsError(f"Run metadata exists with a different configuration: {metadata_path}")
    else:
        metadata_path.write_text(json.dumps(run_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.mode == "oof":
        run_oof(args, dataset, split, device)
    else:
        run_holdout(args, dataset, split, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
