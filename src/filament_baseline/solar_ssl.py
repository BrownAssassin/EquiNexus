# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .model import DINO_V2_REPOSITORY


class SolarContrastiveDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Two augmented views from a random disk crop of each unlabeled GONG image."""

    def __init__(self, data_root: str | Path, size: int = 224) -> None:
        self.data_root = Path(data_root).resolve()
        self.size = size
        manifest_path = self.data_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = manifest["files"]
        canonical = json.dumps(records, sort_keys=True, separators=(",", ":"))
        computed_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if computed_digest != manifest.get("content_manifest_sha256"):
            raise ValueError("GONG image manifest content digest does not match its records")
        self.manifest_digest = computed_digest
        self.records = records
        self.image_paths = [self.data_root / "images" / item["filename"] for item in records]
        self._disk_mask_cache: dict[tuple[int, int], np.ndarray] = {}
        missing = [path.name for path in self.image_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} GONG images; first: {missing[:3]}")
        if size < 28 or size % 14:
            raise ValueError("DINOv2 image size must be at least 28 and divisible by 14")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _sample_crop(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape
        radius = 860.0 * min(width, height) / 2048.0
        crop_size = int(radius * random.uniform(0.30, 0.58))
        angle = random.uniform(0.0, 2.0 * math.pi)
        distance = radius * math.sqrt(random.uniform(0.0, 0.54))
        center_x = width / 2.0 + math.cos(angle) * distance
        center_y = height / 2.0 + math.sin(angle) * distance
        crop_size = min(crop_size, width, height)
        x0 = min(max(int(round(center_x - crop_size / 2)), 0), width - crop_size)
        y0 = min(max(int(round(center_y - crop_size / 2)), 0), height - crop_size)
        crop = image[y0 : y0 + crop_size, x0 : x0 + crop_size]
        return cv2.resize(crop, (self.size, self.size), interpolation=cv2.INTER_AREA)

    def _augment(self, crop: np.ndarray) -> torch.Tensor:
        image = crop.astype(np.float32)
        image = (image - 127.5) * random.uniform(0.72, 1.30) + random.uniform(-20.0, 20.0) + 127.5
        image = np.clip(image, 0.0, 255.0)
        gamma = random.uniform(0.78, 1.25)
        image = 255.0 * np.power(image / 255.0, gamma)
        if random.random() < 0.20:
            image = cv2.GaussianBlur(image, (3, 3), sigmaX=random.uniform(0.2, 1.0))
        image = image.astype(np.uint8)
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
        if random.random() < 0.5:
            image = cv2.flip(image, 0)
        image = np.rot90(image, random.randrange(4)).copy()
        if random.random() < 0.15:
            noise = np.random.normal(0.0, 2.0, image.shape)
            image = np.clip(image.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)
        value = torch.from_numpy(image).float().div_(255.0).unsqueeze(0).expand(3, -1, -1)
        mean = value.new_tensor((0.485, 0.456, 0.406))[:, None, None]
        std = value.new_tensor((0.229, 0.224, 0.225))[:, None, None]
        return (value - mean) / std

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        with Image.open(self.image_paths[index]) as source:
            gray = np.asarray(source.convert("L"))
        height, width = gray.shape
        image_shape = (height, width)
        disk = self._disk_mask_cache.get(image_shape)
        if disk is None:
            radius = 860.0 * min(width, height) / 2048.0
            center_x, center_y = width / 2.0, height / 2.0
            y_coordinates, x_coordinates = np.ogrid[:height, :width]
            disk = (
                (x_coordinates - center_x) ** 2 + (y_coordinates - center_y) ** 2
                <= radius**2
            )
            self._disk_mask_cache[image_shape] = disk
        disk_median = np.median(gray[disk]).astype(gray.dtype)
        gray = gray.copy()
        gray[~disk] = disk_median
        crop = self._sample_crop(gray)
        return self._augment(crop), self._augment(crop)


class _ProjectionHead(nn.Module):
    def __init__(self, in_features: int = 384, hidden_features: int = 512, out_features: int = 1024) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_features, bias=False),
            nn.GELU(),
            nn.Linear(hidden_features, hidden_features // 2),
            nn.GELU(),
            nn.Linear(hidden_features // 2, out_features),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class _DinoSolarStudent(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.projector = _ProjectionHead()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(images)["x_norm_clstoken"]
        return self.projector(features)


def _freeze_early_blocks(backbone: nn.Module, trainable_blocks: int) -> None:
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    for block in backbone.blocks[-trainable_blocks:]:
        for parameter in block.parameters():
            parameter.requires_grad_(True)
    for parameter in backbone.norm.parameters():
        parameter.requires_grad_(True)


@torch.no_grad()
def _ema_update(teacher: nn.Module, student: nn.Module, momentum: float) -> None:
    for teacher_parameter, student_parameter in zip(
        teacher.parameters(), student.parameters(), strict=True
    ):
        teacher_parameter.lerp_(student_parameter, 1.0 - momentum)


def _distillation_loss(
    student_first: torch.Tensor,
    student_second: torch.Tensor,
    teacher_first: torch.Tensor,
    teacher_second: torch.Tensor,
    center: torch.Tensor,
    *,
    student_temperature: float = 0.10,
    teacher_temperature: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    teacher_first = teacher_first.float()
    teacher_second = teacher_second.float()
    center_batch = 0.5 * (teacher_first.mean(dim=0) + teacher_second.mean(dim=0))
    teacher_first = F.softmax((teacher_first - center) / teacher_temperature, dim=-1)
    teacher_second = F.softmax((teacher_second - center) / teacher_temperature, dim=-1)
    student_first = F.log_softmax(student_first.float() / student_temperature, dim=-1)
    student_second = F.log_softmax(student_second.float() / student_temperature, dim=-1)
    loss = -0.5 * (
        (teacher_first * student_second).sum(dim=-1).mean()
        + (teacher_second * student_first).sum(dim=-1).mean()
    )
    return loss, center_batch


def adapt_dinov2_on_solar_images(
    *,
    data_root: str | Path,
    output_path: str | Path,
    steps: int = 1_000,
    batch_size: int = 16,
    image_size: int = 224,
    num_workers: int = 2,
    learning_rate: float = 1e-5,
    projector_learning_rate: float = 5e-4,
    trainable_blocks: int = 4,
    seed: int = 314159,
    device_name: str = "auto",
    hub_repository: str = DINO_V2_REPOSITORY,
    hub_dir: str | Path | None = None,
) -> dict[str, Any]:
    if min(steps, batch_size, trainable_blocks) < 1 or num_workers < 0:
        raise ValueError("steps, batch size, and trainable block count must be positive; workers nonnegative")
    if device_name not in {"auto", "cpu", "cuda"}:
        raise ValueError("device_name must be auto, cpu, or cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for solar adaptation but is unavailable")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

    if hub_dir is not None:
        Path(hub_dir).mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(hub_dir))
    hub_checkpoints = Path(torch.hub.get_dir()) / "checkpoints"
    hub_checkpoints.mkdir(parents=True, exist_ok=True)
    original_checkpoint = hub_checkpoints / "dinov2_vits14_pretrain.pth"
    base_checkpoint = hub_checkpoints / "dinov2_vits14_pretrain.base.pth"
    if not original_checkpoint.is_file():
        torch.hub.load(
            hub_repository,
            "dinov2_vits14",
            pretrained=True,
            trust_repo=True,
        )
    if not base_checkpoint.is_file():
        shutil.copy2(original_checkpoint, base_checkpoint)
    base_state = torch.load(base_checkpoint, map_location="cpu", weights_only=True)
    device = (
        torch.device("cpu")
        if device_name == "cpu"
        else torch.device("cuda:0")
        if device_name == "cuda" or torch.cuda.is_available()
        else torch.device("cpu")
    )
    dataset = SolarContrastiveDataset(data_root, size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    if len(loader) == 0:
        raise ValueError("External GONG dataset must contain at least one complete batch")

    student_backbone = torch.hub.load(
        hub_repository,
        "dinov2_vits14",
        pretrained=False,
        trust_repo=True,
    )
    student_backbone.load_state_dict(base_state, strict=True)
    _freeze_early_blocks(student_backbone, min(trainable_blocks, len(student_backbone.blocks)))
    student = _DinoSolarStudent(student_backbone).to(device)
    teacher = deepcopy(student).to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    backbone_parameters = [p for p in student.backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": learning_rate},
            {"params": student.projector.parameters(), "lr": projector_learning_rate},
        ],
        weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    center = torch.zeros(1, 1024, device=device)
    started = time.perf_counter()
    iterator = iter(loader)
    student.train()

    for step in range(steps):
        try:
            first, second = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            first, second = next(iterator)
        first = first.to(device, non_blocking=True)
        second = second.to(device, non_blocking=True)
        progress = step / max(1, steps - 1)
        multiplier = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))
        optimizer.param_groups[0]["lr"] = learning_rate * multiplier
        optimizer.param_groups[1]["lr"] = projector_learning_rate * multiplier
        autocast_enabled = device.type == "cuda"
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
            student_first = student(first)
            student_second = student(second)
            with torch.no_grad():
                teacher_first = teacher(first)
                teacher_second = teacher(second)
        loss, batch_center = _distillation_loss(
            student_first,
            student_second,
            teacher_first,
            teacher_second,
            center,
        )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in student.parameters() if parameter.requires_grad], 1.0
        )
        scaler.step(optimizer)
        scaler.update()
        momentum = 0.996 + 0.003 * progress
        _ema_update(teacher, student, momentum)
        center.mul_(0.9).add_(batch_center.detach(), alpha=0.1)
        if (step + 1) % 100 == 0 or step + 1 == steps:
            print(
                json.dumps(
                    {
                        "step": step + 1,
                        "steps": steps,
                        "loss": float(loss.detach()),
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adapted_state = {key: value.detach().cpu() for key, value in student.backbone.state_dict().items()}
    torch.save(
        {"state_dict": adapted_state, "manifest_sha256": dataset.manifest_digest},
        output_path,
    )
    original_sha256 = hashlib.sha256(base_checkpoint.read_bytes()).hexdigest()
    torch.save(adapted_state, original_checkpoint)
    artifact_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
    hub_checkpoint_sha256 = hashlib.sha256(original_checkpoint.read_bytes()).hexdigest()
    adaptation = {
        "source_model": hub_repository,
        "model_license": "Apache-2.0",
        "source_checkpoint_sha256": original_sha256,
        "adapted_checkpoint_sha256": hub_checkpoint_sha256,
        "external_dataset": "NSO GONG H-alpha unlabeled public archive",
        "external_source_url": "https://nso.edu/data/nisp-data/h-alpha/",
        "external_image_credit": "NSO/AURA/NSF",
        "external_data_acknowledgement": (
            "Data were acquired by GONG instruments operated by NISP/NSO/AURA/NSF "
            "with contribution from NOAA."
        ),
        "external_image_use_policy": "https://nso.edu/about/image-use-policy/",
        "external_manifest_sha256": dataset.manifest_digest,
        "external_image_count": len(dataset),
        "external_ground_truth_used": False,
        "steps": steps,
        "batch_size": batch_size,
        "image_size": image_size,
        "trainable_transformer_blocks": min(trainable_blocks, len(student.backbone.blocks)),
        "seed": seed,
    }
    original_checkpoint.with_suffix(".json").write_text(
        json.dumps(adaptation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result = {
        "status": "complete",
        "source_model": hub_repository,
        "original_checkpoint_sha256": original_sha256,
        "adapted_checkpoint_sha256": hub_checkpoint_sha256,
        "adaptation_artifact_sha256": artifact_sha256,
        "external_dataset": "NSO GONG H-alpha unlabeled public archive",
        "external_source_url": "https://nso.edu/data/nisp-data/h-alpha/",
        "external_image_credit": "NSO/AURA/NSF",
        "external_data_acknowledgement": (
            "Data were acquired by GONG instruments operated by NISP/NSO/AURA/NSF "
            "with contribution from NOAA."
        ),
        "model_license": "Apache-2.0",
        "external_manifest_sha256": dataset.manifest_digest,
        "external_image_count": len(dataset),
        "external_ground_truth_used": False,
        "steps": steps,
        "batch_size": batch_size,
        "image_size": image_size,
        "trainable_transformer_blocks": min(trainable_blocks, len(student.backbone.blocks)),
        "seed": seed,
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "checkpoint": str(output_path),
        "hub_checkpoint": str(original_checkpoint),
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result
