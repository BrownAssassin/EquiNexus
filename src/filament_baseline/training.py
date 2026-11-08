# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .learning import FilamentPatchDataset, TargetMode, training_loss
from .model import DINO_V2_REPOSITORY, build_filament_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


@torch.no_grad()
def evaluate_patch_dice(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    max_batches: int = 64,
) -> float:
    model.eval()
    scores = []
    for index, (images, targets) in enumerate(loader):
        if index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(images)
        probabilities = logits[:, 0:1].float().sigmoid()
        expected = targets[:, 0:1]
        intersection = (probabilities * expected).sum(dim=(1, 2, 3))
        denominator = probabilities.sum(dim=(1, 2, 3)) + expected.sum(dim=(1, 2, 3))
        scores.extend(((2 * intersection + 1) / (denominator + 1)).cpu().tolist())
    return float(np.mean(scores)) if scores else 0.0


def train_model(
    *,
    train_dataset: FilamentPatchDataset,
    validation_dataset: FilamentPatchDataset | None,
    output_path: str | Path,
    steps: int,
    batch_size: int,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    seed: int = 42,
    validation_interval: int = 500,
    gradient_accumulation: int = 1,
    base_channels: int = 16,
    model_type: str = "unet",
    pretrained_backbone: bool = False,
    hub_repository: str = DINO_V2_REPOSITORY,
    hub_dir: str | Path | None = None,
    initial_weights: str | Path | None = None,
) -> dict[str, Any]:
    if steps < 1 or batch_size < 1 or gradient_accumulation < 1 or base_channels < 1:
        raise ValueError(
            "steps, batch_size, gradient_accumulation, and base_channels must be positive"
        )
    seed_everything(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_channels = int(getattr(train_dataset, "in_channels", 1))
    if model_type == "dinov2_vits14":
        patch_size = int(getattr(train_dataset, "patch_size", 0))
        if patch_size % 224:
            raise ValueError(
                f"DINOv2 patch size must be divisible by 224, received {patch_size}"
            )
    if hub_dir is not None:
        Path(hub_dir).mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(hub_dir))
    model = build_filament_model(
        model_type=model_type,
        in_channels=input_channels,
        base_channels=base_channels,
        pretrained_backbone=pretrained_backbone,
        hub_repository=hub_repository,
    ).to(device)
    if initial_weights is not None:
        initial_path = Path(initial_weights).expanduser().resolve()
        if not initial_path.is_file():
            raise FileNotFoundError(f"Initial model checkpoint does not exist: {initial_path}")
        checkpoint = torch.load(initial_path, map_location="cpu", weights_only=True)
        expected_config = {
            "model_type": model_type,
            "base_channels": base_channels,
            "in_channels": input_channels,
            "hub_repository": hub_repository,
        }
        if checkpoint.get("model_config") != expected_config:
            raise ValueError(
                "Initial checkpoint model configuration does not match the requested model: "
                f"{checkpoint.get('model_config')} != {expected_config}"
            )
        model.load_state_dict(checkpoint["model"], strict=True)
        print(f"Initialized model from {initial_path}", flush=True)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    warmup_steps = min(300, max(1, steps // 10))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return max(0.05, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, steps - warmup_steps)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    validation_loader = (
        DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        if validation_dataset is not None
        else None
    )
    if len(train_loader) == 0:
        raise ValueError("The training patch loader produced no batches")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history_path = output_path.with_suffix(".jsonl")
    best_dice = -1.0
    best_step = 0
    running_loss = 0.0
    logged_steps = 0
    started = time.perf_counter()
    iterator = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with history_path.open("w", encoding="utf-8") as history:
        for step in range(1, steps + 1):
            model.train()
            try:
                images, targets = next(iterator)
            except StopIteration:
                train_dataset.set_epoch(step // len(train_loader))
                iterator = iter(train_loader)
                images, targets = next(iterator)
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(images)
                loss = training_loss(logits, targets) / gradient_accumulation
            scaler.scale(loss).backward()
            if step % gradient_accumulation == 0 or step == steps:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            running_loss += float(loss.detach()) * gradient_accumulation
            logged_steps += 1

            should_validate = (
                step % validation_interval == 0 or step == steps
            )
            if should_validate:
                patch_dice = (
                    evaluate_patch_dice(model, validation_loader, device, max_batches=64)
                    if validation_loader is not None
                    else None
                )
                row = {
                    "step": step,
                    "mean_train_loss_since_log": running_loss
                    / max(1, logged_steps),
                    "validation_patch_dice": patch_dice,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "elapsed_seconds": time.perf_counter() - started,
                }
                history.write(json.dumps(row, sort_keys=True) + "\n")
                history.flush()
                print(json.dumps(row, sort_keys=True), flush=True)
                running_loss = 0.0
                logged_steps = 0
                if patch_dice is None or patch_dice > best_dice:
                    if patch_dice is not None:
                        best_dice = patch_dice
                    best_step = step
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "model_config": {
                                "model_type": model_type,
                                "base_channels": base_channels,
                                "in_channels": input_channels,
                                "hub_repository": hub_repository,
                            },
                            "target_mode": train_dataset.target_mode,
                            "step": step,
                            "validation_patch_dice": patch_dice,
                            "seed": seed,
                        },
                        output_path,
                    )

    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    return {
        "checkpoint": str(output_path),
        "history": str(history_path),
        "device": str(device),
        "steps": steps,
        "base_channels": base_channels,
        "model_type": model_type,
        "pretrained_backbone": pretrained_backbone,
        "hub_repository": hub_repository if model_type == "dinov2_vits14" else None,
        "seed": seed,
        "best_step": best_step,
        "best_validation_patch_dice": best_dice if best_dice >= 0.0 else None,
        "selection_mode": "best_validation_patch_dice"
        if validation_loader is not None
        else "final_step",
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": peak_memory,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
