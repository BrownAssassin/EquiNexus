# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .data import CompetitionDataset, ImageView
from .features import InputMode, input_channel_count, prepare_input_channels

TargetMode = Literal["annotator", "consensus"]


@dataclass(frozen=True)
class InstancePolygons:
    polygons: tuple[np.ndarray, ...]
    bounds: tuple[float, float, float, float]


@dataclass(frozen=True)
class ViewTargets:
    view: ImageView
    instances: tuple[InstancePolygons, ...]


@dataclass(frozen=True)
class PatchRecord:
    file_name: str
    views: tuple[ViewTargets, ...]


def _view_targets(dataset: CompetitionDataset, view: ImageView) -> ViewTargets:
    instances = []
    for annotation in dataset.annotations_by_image_id[view.image_id]:
        segmentation = annotation["segmentation"]
        if not isinstance(segmentation, list):
            raise ValueError("Training polygons must follow the competition COCO polygon format")
        polygons = tuple(
            np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            for polygon in segmentation
            if len(polygon) >= 6
        )
        if not polygons:
            continue
        points = np.concatenate(polygons, axis=0)
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        instances.append(
            InstancePolygons(
                polygons,
                (float(x_min), float(y_min), float(x_max), float(y_max)),
            )
        )
    return ViewTargets(view, tuple(instances))


def records_for_observations(
    dataset: CompetitionDataset,
    observations: tuple[str, ...] | list[str],
    target_mode: TargetMode,
) -> tuple[PatchRecord, ...]:
    records = []
    for file_name in sorted(observations):
        views = tuple(
            _view_targets(dataset, view)
            for view in dataset.views_by_observation[file_name]
        )
        if target_mode == "annotator":
            records.extend(PatchRecord(file_name, (view,)) for view in views)
        elif target_mode == "consensus":
            records.append(PatchRecord(file_name, views))
        else:
            raise ValueError(f"Unsupported target mode: {target_mode}")
    return tuple(records)


def _instance_map(view: ViewTargets, x0: int, y0: int, size: int) -> np.ndarray:
    labels = np.zeros((size, size), dtype=np.uint16)
    x1, y1 = x0 + size, y0 + size
    for label, instance in enumerate(view.instances, start=1):
        left, top, right, bottom = instance.bounds
        if right < x0 or bottom < y0 or left >= x1 or top >= y1:
            continue
        local_polygons = []
        for polygon in instance.polygons:
            points = np.rint(polygon - np.array([x0, y0], dtype=np.float32))
            points = np.clip(points, -size, 2 * size).astype(np.int32)
            if len(points) >= 3:
                local_polygons.append(points)
        if local_polygons:
            cv2.fillPoly(labels, local_polygons, color=label, lineType=cv2.LINE_8)
    return labels


def _foreground_and_boundary(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    foreground = (labels > 0).astype(np.float32)
    boundary = np.zeros(labels.shape, dtype=bool)
    horizontal = labels[:, 1:] != labels[:, :-1]
    vertical = labels[1:, :] != labels[:-1, :]
    boundary[:, 1:] |= horizontal & ((labels[:, 1:] > 0) | (labels[:, :-1] > 0))
    boundary[:, :-1] |= horizontal & ((labels[:, 1:] > 0) | (labels[:, :-1] > 0))
    boundary[1:, :] |= vertical & ((labels[1:, :] > 0) | (labels[:-1, :] > 0))
    boundary[:-1, :] |= vertical & ((labels[1:, :] > 0) | (labels[:-1, :] > 0))
    return foreground, boundary.astype(np.float32)


class FilamentPatchDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """On-the-fly full-resolution crops from official annotation views."""

    def __init__(
        self,
        dataset: CompetitionDataset,
        observations: tuple[str, ...] | list[str],
        *,
        target_mode: TargetMode = "consensus",
        patch_size: int = 512,
        positive_fraction: float = 0.7,
        augment: bool = False,
        seed: int = 42,
        image_cache_size: int = 8,
        input_mode: InputMode = "raw",
    ) -> None:
        if patch_size < 128 or patch_size % 32:
            raise ValueError("patch_size must be at least 128 and divisible by 32")
        if not 0.0 <= positive_fraction <= 1.0:
            raise ValueError("positive_fraction must be between 0 and 1")
        self.records = records_for_observations(dataset, observations, target_mode)
        self.target_mode = target_mode
        self.image_dir = dataset.train_image_dir
        self.image_resize_to = dataset.image_resize_to
        self.patch_size = patch_size
        self.positive_fraction = positive_fraction
        self.augment = augment
        self.seed = seed
        self.input_mode = input_mode
        self.in_channels = input_channel_count(input_mode)
        self.epoch = 0
        self.image_cache_size = image_cache_size
        self._image_cache: OrderedDict[
            str, tuple[np.ndarray, float, float, np.ndarray | None]
        ] = OrderedDict()
        self._disk = np.zeros((2048, 2048), dtype=np.uint8)
        cv2.circle(self._disk, (1023, 1023), 950, color=1, thickness=-1)

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _image_and_stats(
        self, file_name: str
    ) -> tuple[np.ndarray, float, float, np.ndarray | None]:
        cached = self._image_cache.get(file_name)
        if cached is not None:
            self._image_cache.move_to_end(file_name)
            return cached
        image = cv2.imread(str(self.image_dir / file_name), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read a training image: {file_name}")
        if image.shape != (2048, 2048) and self.image_resize_to is not None:
            height, width = self.image_resize_to
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        if image.shape != (2048, 2048):
            raise FileNotFoundError(f"Could not read a 2048x2048 training image: {file_name}")
        scaled = image.astype(np.float32) / 255.0
        disk_values = scaled[self._disk > 0]
        mean = float(np.mean(disk_values))
        std = float(np.std(disk_values))
        auxiliary_channels = (
            prepare_input_channels(image, self.input_mode)[1:]
            if self.in_channels > 1
            else None
        )
        cached = (image, mean, max(std, 1e-3), auxiliary_channels)
        self._image_cache[file_name] = cached
        while len(self._image_cache) > self.image_cache_size:
            self._image_cache.popitem(last=False)
        return cached

    def _crop_origin(self, record: PatchRecord, rng: random.Random) -> tuple[int, int]:
        annotations = [
            instance
            for view in record.views
            for instance in view.instances
        ]
        if annotations and rng.random() < self.positive_fraction:
            instance = rng.choice(annotations)
            polygon = rng.choice(instance.polygons)
            center_x, center_y = polygon[rng.randrange(len(polygon))]
        else:
            center_x = rng.uniform(0, 2047)
            center_y = rng.uniform(0, 2047)
        half = self.patch_size // 2
        x0 = min(max(int(center_x) - half, 0), 2048 - self.patch_size)
        y0 = min(max(int(center_y) - half, 0), 2048 - self.patch_size)
        return x0, y0

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        rng = random.Random(self.seed + self.epoch * 1_000_003 + index * 97)
        image, mean, std, auxiliary_channels = self._image_and_stats(record.file_name)
        x0, y0 = self._crop_origin(record, rng)
        x1, y1 = x0 + self.patch_size, y0 + self.patch_size
        image_crop = image[y0:y1, x0:x1].astype(np.float32) / 255.0
        auxiliary_crop = (
            auxiliary_channels[:, y0:y1, x0:x1].copy()
            if auxiliary_channels is not None
            else None
        )

        per_view_foreground = []
        per_view_boundary = []
        for view in record.views:
            labels = _instance_map(view, x0, y0, self.patch_size)
            foreground, boundary = _foreground_and_boundary(labels)
            per_view_foreground.append(foreground)
            per_view_boundary.append(boundary)
        target_foreground = np.mean(per_view_foreground, axis=0, dtype=np.float32)
        target_boundary = np.mean(per_view_boundary, axis=0, dtype=np.float32)

        if self.augment:
            turns = rng.randrange(4)
            image_crop = np.rot90(image_crop, turns).copy()
            if auxiliary_crop is not None:
                auxiliary_crop = np.rot90(auxiliary_crop, turns, axes=(-2, -1)).copy()
            target_foreground = np.rot90(target_foreground, turns).copy()
            target_boundary = np.rot90(target_boundary, turns).copy()
            if rng.random() < 0.5:
                image_crop = np.fliplr(image_crop).copy()
                if auxiliary_crop is not None:
                    auxiliary_crop = np.flip(auxiliary_crop, axis=-1).copy()
                target_foreground = np.fliplr(target_foreground).copy()
                target_boundary = np.fliplr(target_boundary).copy()
            if rng.random() < 0.5:
                image_crop = np.flipud(image_crop).copy()
                if auxiliary_crop is not None:
                    auxiliary_crop = np.flip(auxiliary_crop, axis=-2).copy()
                target_foreground = np.flipud(target_foreground).copy()
                target_boundary = np.flipud(target_boundary).copy()
            image_crop = np.power(
                np.clip(image_crop, 0.0, 1.0), rng.uniform(0.8, 1.2)
            )
            image_crop = np.clip(
                image_crop * rng.uniform(0.9, 1.1) + rng.uniform(-0.04, 0.04),
                0.0,
                1.0,
            )

        normalized = np.clip((image_crop - mean) / std, -5.0, 5.0)
        input_channels = [normalized.astype(np.float32)]
        if auxiliary_crop is not None:
            input_channels.extend(
                channel.astype(np.float32) for channel in auxiliary_crop
            )
        image_tensor = torch.from_numpy(np.stack(input_channels))
        target_tensor = torch.from_numpy(
            np.stack((target_foreground, target_boundary)).astype(np.float32)
        )
        return image_tensor, target_tensor


def training_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    foreground_target = target[:, 0:1]
    boundary_target = target[:, 1:2]
    foreground_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:, 0:1], foreground_target
    )
    foreground_probability = logits[:, 0:1].sigmoid()
    intersection = (foreground_probability * foreground_target).sum(dim=(1, 2, 3))
    denominator = foreground_probability.sum(dim=(1, 2, 3)) + foreground_target.sum(
        dim=(1, 2, 3)
    )
    dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    boundary_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:, 1:2], boundary_target, pos_weight=torch.tensor(
            5.0, dtype=logits.dtype, device=logits.device
        )
    )
    return 0.45 * foreground_bce + 0.45 * dice_loss + 0.10 * boundary_bce
