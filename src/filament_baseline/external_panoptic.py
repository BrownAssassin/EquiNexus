# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Inference adapter for the public Apache-2.0 I079 detector/refiner weights.

The implementation is independent EquiNexus code. It loads only the public
checkpoint pair and operates on the supplied competition test images.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from torch import nn
from torchvision.models import resnet18
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign


FULL_SIZE = 2048
DETECTOR_SIZE = 1024
CROP_SIZE = 256
DETECTOR_SCORE_MIN = 0.70
DETECTOR_MASK_MIN = 0.50
REFINER_THRESHOLD = 0.70
REFINER_TTA_TRANSFORMS = ("identity", "horizontal", "vertical", "both")
TTA_AREA_RATIO = (0.60, 1.70)
TTA_MIN_DICE = 0.62
TTA_MIN_DICE_AREA = 32
TTA_MAX_COMPONENT_INCREASE = 1
MAX_SPLIT_TASKS = 1
SEED_MAX_IOU = 0.10
SEED_MIN_UNIQUE_PIXELS = 16
SEED_MIN_DISTANCE = 8.0

EXPECTED_MODEL_MD5 = {
    "final_detector.pt": "dcd3053561e40ba78718546c31525d26",
    "final_refiner.pt": "74991b8d95f7922f0ca6fd75f8ab4f5f",
}


def build_detector() -> nn.Module:
    """Construct the architecture required by the public detector checkpoint."""
    model = maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
    box_features = model.roi_heads.box_predictor.cls_score.in_features
    mask_predictor = model.roi_heads.mask_predictor
    mask_features = mask_predictor.conv5_mask.in_channels
    hidden_channels = mask_predictor.conv5_mask.out_channels
    model.roi_heads.box_predictor = FastRCNNPredictor(box_features, 2)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        mask_features, hidden_channels, 2
    )
    model.rpn.anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )
    model.roi_heads.mask_roi_pool = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"], output_size=28, sampling_ratio=2
    )
    model.transform.min_size = (DETECTOR_SIZE,)
    model.transform.max_size = DETECTOR_SIZE
    return model


class _DecoderBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class CropRefiner(nn.Module):
    """ResNet-18 encoder and compact U-Net decoder for detector crops."""

    def __init__(self) -> None:
        super().__init__()
        encoder = resnet18(weights=None)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4
        self.up4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec4 = _DecoderBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3 = _DecoderBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = _DecoderBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1 = _DecoderBlock(96, 32)
        self.up0 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.head = nn.Conv2d(16, 1, 1)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = (image - self.mean) / self.std
        stem = self.stem(image)
        level1 = self.layer1(self.pool(stem))
        level2 = self.layer2(level1)
        level3 = self.layer3(level2)
        level4 = self.layer4(level3)
        value = self.dec4(torch.cat((self.up4(level4), level3), dim=1))
        value = self.dec3(torch.cat((self.up3(value), level2), dim=1))
        value = self.dec2(torch.cat((self.up2(value), level1), dim=1))
        value = self.dec1(torch.cat((self.up1(value), stem), dim=1))
        return self.head(self.up0(value))


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.count_nonzero(first & second))
    union = int(np.count_nonzero(first | second))
    return intersection / union if union else 0.0


def _overlap_groups(masks: np.ndarray) -> list[list[int]]:
    """Group detector masks connected by any positive-area overlap."""
    count = len(masks)
    adjacency: list[list[int]] = [[] for _ in range(count)]
    for first in range(count):
        for second in range(first + 1, count):
            if np.any(masks[first] & masks[second]):
                adjacency[first].append(second)
                adjacency[second].append(first)
    groups: list[list[int]] = []
    visited: set[int] = set()
    for start in range(count):
        if start in visited:
            continue
        queue = deque([start])
        visited.add(start)
        group: list[int] = []
        while queue:
            member = queue.popleft()
            group.append(member)
            for neighbor in adjacency[member]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        groups.append(sorted(group))
    return groups


def _square_crop(mask: np.ndarray, context: float = 1.8, minimum: int = 96) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError("cannot crop an empty detector group")
    height, width = mask.shape
    center_x = (int(xs.min()) + int(xs.max()) + 1) / 2.0
    center_y = (int(ys.min()) + int(ys.max()) + 1) / 2.0
    side = min(
        max(int(np.ceil(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * context)), minimum),
        min(height, width),
    )
    x0 = int(round(center_x - side / 2.0))
    y0 = int(round(center_y - side / 2.0))
    x0 = min(max(x0, 0), width - side)
    y0 = min(max(y0, 0), height - side)
    return x0, y0, x0 + side, y0 + side


def _to_refiner_batch(crops: np.ndarray, device: torch.device) -> torch.Tensor:
    gray = torch.from_numpy(np.ascontiguousarray(crops)).float().unsqueeze(1) / 255.0
    return gray.repeat(1, 3, 1, 1).to(device, non_blocking=True)


def _refiner_probabilities(
    model: CropRefiner,
    crops: np.ndarray,
    device: torch.device,
    batch_size: int = 24,
) -> np.ndarray:
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(crops), batch_size):
            batch = _to_refiner_batch(crops[start : start + batch_size], device)
            logits = model(batch).sigmoid()[:, 0]
            output.append(logits.cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(output, axis=0)


def _transform_batch(crops: np.ndarray, transform: str) -> np.ndarray:
    if transform == "identity":
        return np.ascontiguousarray(crops)
    if transform == "horizontal":
        return np.ascontiguousarray(crops[:, :, ::-1])
    if transform == "vertical":
        return np.ascontiguousarray(crops[:, ::-1, :])
    if transform == "both":
        return np.ascontiguousarray(crops[:, ::-1, ::-1])
    raise ValueError(f"unknown crop transform: {transform}")


def _undo_transform(probabilities: np.ndarray, transform: str) -> np.ndarray:
    if transform == "identity":
        return probabilities
    if transform == "horizontal":
        return probabilities[:, :, ::-1]
    if transform == "vertical":
        return probabilities[:, ::-1, :]
    if transform == "both":
        return probabilities[:, ::-1, ::-1]
    raise ValueError(f"unknown crop transform: {transform}")


def _binary_dice(first: np.ndarray, second: np.ndarray) -> float:
    size = int(first.sum() + second.sum())
    return 1.0 if size == 0 else 2.0 * int(np.count_nonzero(first & second)) / size


def _safe_tta_mask(single: np.ndarray, averaged: np.ndarray) -> np.ndarray:
    base = single >= REFINER_THRESHOLD
    candidate = averaged >= REFINER_THRESHOLD
    base_area, candidate_area = int(base.sum()), int(candidate.sum())
    if base_area == 0 or candidate_area == 0:
        return single
    ratio = candidate_area / base_area
    if not TTA_AREA_RATIO[0] <= ratio <= TTA_AREA_RATIO[1]:
        return single
    base_components = cv2.connectedComponents(base.astype(np.uint8), connectivity=8)[0] - 1
    candidate_components = cv2.connectedComponents(candidate.astype(np.uint8), connectivity=8)[0] - 1
    if candidate_components > max(base_components + TTA_MAX_COMPONENT_INCREASE, 2):
        return single
    if base_area >= TTA_MIN_DICE_AREA and _binary_dice(base, candidate) < TTA_MIN_DICE:
        return single
    return averaged


def _restore_crop_mask(probability: np.ndarray, bounds: tuple[int, int, int, int], fallback: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = bounds
    resized = cv2.resize(probability, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    result = np.zeros_like(fallback, dtype=bool)
    result[y0:y1, x0:x1] = resized >= REFINER_THRESHOLD
    return result if np.any(result) else fallback.astype(bool, copy=True)


def _split_with_detector_seeds(
    refined: np.ndarray, seeds: list[np.ndarray], scores: np.ndarray
) -> list[np.ndarray]:
    if len(seeds) < 2:
        return [refined]
    order = sorted(range(len(seeds)), key=lambda i: (-float(scores[i]), i))
    accepted: list[int] = []
    for candidate in order:
        if all(_mask_iou(seeds[candidate], seeds[other]) <= SEED_MAX_IOU for other in accepted):
            accepted.append(candidate)
    if len(accepted) < 2:
        return [refined]
    points: list[tuple[int, int]] = []
    for index in accepted:
        other_union = np.logical_or.reduce([seeds[j] for j in accepted if j != index])
        unique = seeds[index] & refined & ~other_union
        ys, xs = np.where(unique)
        if len(xs) < SEED_MIN_UNIQUE_PIXELS:
            continue
        cy, cx = float(ys.mean()), float(xs.mean())
        nearest = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
        points.append((int(ys[nearest]), int(xs[nearest])))
    if len(points) < 2:
        return [refined]
    if any(
        np.hypot(points[a][0] - points[b][0], points[a][1] - points[b][1]) < SEED_MIN_DISTANCE
        for a in range(len(points)) for b in range(a + 1, len(points))
    ):
        return [refined]
    ys, xs = np.where(refined)
    distances = np.stack([(ys - py) ** 2 + (xs - px) ** 2 for py, px in points], axis=1)
    assignment = distances.argmin(axis=1)
    parts: list[np.ndarray] = []
    for index in range(len(points)):
        chosen = assignment == index
        if not np.any(chosen):
            return [refined]
        part = np.zeros_like(refined, dtype=bool)
        part[ys[chosen], xs[chosen]] = True
        parts.append(part)
    if not np.array_equal(np.logical_or.reduce(parts), refined):
        raise RuntimeError("seed partition changed the refined foreground")
    return parts


def _merge_overlapping(masks: list[np.ndarray]) -> list[np.ndarray]:
    if not masks:
        return []
    stack = np.stack(masks, axis=0)
    output: list[np.ndarray] = []
    for group in _overlap_groups(stack):
        output.append(np.logical_or.reduce([stack[index] for index in group]))
    output.sort(key=lambda mask: (
        int(np.where(mask)[0].min()),
        int(np.where(mask)[1].min()),
    ))
    return output


def predict_image(
    detector: nn.Module,
    refiner: CropRefiner,
    image: np.ndarray,
    device: torch.device,
) -> tuple[list[np.ndarray], dict[str, int]]:
    """Predict full-resolution instances for one grayscale test image."""
    if image.shape != (FULL_SIZE, FULL_SIZE) or image.dtype != np.uint8:
        raise ValueError(f"expected uint8 {FULL_SIZE}x{FULL_SIZE} grayscale input")
    detector_view = cv2.resize(image, (DETECTOR_SIZE, DETECTOR_SIZE), interpolation=cv2.INTER_AREA)
    channels = np.repeat(detector_view[None], 3, axis=0).astype(np.float32) / 255.0
    with torch.inference_mode():
        detected = detector([torch.from_numpy(channels).to(device)])[0]
    keep = detected["scores"] >= DETECTOR_SCORE_MIN
    scores = detected["scores"][keep].detach().cpu().numpy().astype(np.float32)
    masks_1024 = detected["masks"][keep, 0].detach().cpu().numpy() >= DETECTOR_MASK_MIN
    valid = np.flatnonzero(masks_1024.any(axis=(1, 2)))
    masks_1024 = np.ascontiguousarray(masks_1024[valid])
    scores = np.ascontiguousarray(scores[valid])
    if len(masks_1024) == 0:
        return [], {"detector_masks": 0, "refiner_tasks": 0, "split_tasks": 0}

    native_members: list[list[np.ndarray]] = []
    native_fallbacks: list[np.ndarray] = []
    bounds_list: list[tuple[int, int, int, int]] = []
    crops: list[np.ndarray] = []
    grouped_scores: list[np.ndarray] = []
    for group in _overlap_groups(masks_1024):
        group_union = np.logical_or.reduce([masks_1024[index] for index in group])
        fallback = cv2.resize(
            group_union.astype(np.uint8), (FULL_SIZE, FULL_SIZE), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        bounds = _square_crop(fallback)
        x0, y0, x1, y1 = bounds
        crops.append(cv2.resize(image[y0:y1, x0:x1], (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR))
        bounds_list.append(bounds)
        native_fallbacks.append(fallback)
        native_members.append([
            cv2.resize(mask.astype(np.uint8), (FULL_SIZE, FULL_SIZE), interpolation=cv2.INTER_NEAREST).astype(bool)
            for mask in masks_1024[group]
        ])
        grouped_scores.append(scores[group])

    crop_array = np.stack(crops)
    original = _refiner_probabilities(refiner, crop_array, device)
    transformed: list[np.ndarray] = []
    for transform in REFINER_TTA_TRANSFORMS:
        if transform == "identity":
            values = original
        else:
            values = _refiner_probabilities(refiner, _transform_batch(crop_array, transform), device)
            values = _undo_transform(values, transform)
        transformed.append(values.astype(np.float32, copy=False))
    tta_mean = np.mean(np.stack(transformed, axis=0), axis=0, dtype=np.float32)

    unsplit: list[np.ndarray] = []
    split_variants: list[list[np.ndarray]] = []
    for index in range(len(crops)):
        chosen_probability = _safe_tta_mask(original[index], tta_mean[index])
        refined = _restore_crop_mask(chosen_probability, bounds_list[index], native_fallbacks[index])
        unsplit.append(refined)
        split_variants.append(_split_with_detector_seeds(refined, native_members[index], grouped_scores[index]))

    proposed_splits = sum(len(parts) > 1 for parts in split_variants)
    use_splits = proposed_splits <= MAX_SPLIT_TASKS
    selected = [
        part
        for plain, split_parts in zip(unsplit, split_variants, strict=True)
        for part in (split_parts if use_splits else [plain])
    ]
    instances = _merge_overlapping(selected)
    diagnostics = {
        "detector_masks": int(len(masks_1024)),
        "refiner_tasks": int(len(crops)),
        "split_tasks": int(proposed_splits if use_splits else 0),
    }
    return instances, diagnostics


def encode_instance(mask: np.ndarray) -> str:
    if mask.shape != (FULL_SIZE, FULL_SIZE) or not mask.any():
        raise ValueError("cannot encode an empty or wrong-size instance mask")
    result = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return result["counts"].decode("ascii")


def load_public_checkpoints(
    detector_path: str | Path,
    refiner_path: str | Path,
    device: torch.device,
) -> tuple[nn.Module, CropRefiner]:
    detector_state = torch.load(Path(detector_path), map_location="cpu", weights_only=True)
    refiner_state = torch.load(Path(refiner_path), map_location="cpu", weights_only=True)
    if not isinstance(detector_state, dict) or set(detector_state) != {"epoch", "model"}:
        raise ValueError("unexpected detector checkpoint structure")
    if not isinstance(refiner_state, dict) or set(refiner_state) != {"epoch", "model"}:
        raise ValueError("unexpected refiner checkpoint structure")
    if int(detector_state["epoch"]) != 6 or int(refiner_state["epoch"]) != 8:
        raise ValueError("unexpected public checkpoint versions")
    detector = build_detector()
    refiner = CropRefiner()
    detector.load_state_dict(detector_state["model"], strict=True)
    refiner.load_state_dict(refiner_state["model"], strict=True)
    detector.to(device).eval()
    refiner.to(device).eval()
    return detector, refiner
