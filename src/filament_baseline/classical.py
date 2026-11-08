# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .data import IMAGE_SIZE
from .predictor import PredictedInstance, encode_cropped_binary_mask


@dataclass(frozen=True)
class ClassicalConfig:
    z_threshold: float = 2.55
    bridge_size: int = 9
    min_instance_area: int = 165
    limb_fraction: float = 0.945


def _disk_geometry(gray: np.ndarray) -> tuple[int, int, int]:
    height, width = gray.shape
    reduced = cv2.resize(gray, (max(1, width // 4), max(1, height // 4)), interpolation=cv2.INTER_AREA)
    _, bright = cv2.threshold(reduced, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        (center_x, center_y), radius = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))
        center_x, center_y, radius = int(round(center_x * 4)), int(round(center_y * 4)), int(round(radius * 4))
        if 650 <= radius <= 1020:
            return center_x, center_y, radius
    return width // 2, height // 2, int(0.43 * min(height, width))


def compute_salience_zmap(gray_image: np.ndarray, limb_fraction: float = 0.945) -> tuple[np.ndarray, np.ndarray]:
    """Estimate disk-normalized dark filament response and a safe-disk mask."""
    if gray_image.ndim != 2 or gray_image.dtype != np.uint8:
        raise ValueError("Expected an 8-bit grayscale image")
    if gray_image.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"Expected {IMAGE_SIZE}x{IMAGE_SIZE}, got {gray_image.shape}")

    reduced = cv2.resize(gray_image, (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float32)
    center_x, center_y, radius = _disk_geometry(reduced.astype(np.uint8))
    yy, xx = np.ogrid[:1024, :1024]
    safe_disk = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= (limb_fraction * radius) ** 2

    broad = cv2.GaussianBlur(reduced, (0, 0), sigmaX=28.0)
    narrow = cv2.GaussianBlur(reduced, (0, 0), sigmaX=14.0)
    local_background = 0.65 * broad + 0.35 * narrow
    absorption = np.clip(
        (local_background - reduced) / np.maximum(local_background, 25.0), 0.0, 1.0
    )

    reduced_u8 = np.clip(reduced, 0, 255).astype(np.uint8)
    responses = []
    for size in (9, 19, 33):
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        response = cv2.morphologyEx(reduced_u8, cv2.MORPH_BLACKHAT, element)
        responses.append(response.astype(np.float32) / 255.0)
    salience = (
        0.50 * absorption
        + 0.15 * responses[0]
        + 0.20 * responses[1]
        + 0.15 * responses[2]
    )
    disk_values = salience[safe_disk]
    median = float(np.median(disk_values))
    robust_scale = 1.4826 * float(np.median(np.abs(disk_values - median))) + 1e-5
    z_map = (salience - median) / robust_scale
    z_map[~safe_disk] = -np.inf
    return z_map.astype(np.float32), safe_disk


def predictions_from_zmap(
    z_map: np.ndarray,
    safe_disk: np.ndarray,
    file_name: str,
    config: ClassicalConfig | None = None,
) -> list[PredictedInstance]:
    config = config or ClassicalConfig()
    if z_map.shape != (1024, 1024) or safe_disk.shape != z_map.shape:
        raise ValueError("Expected a 1024x1024 salience map and disk mask")
    if config.bridge_size < 1 or config.bridge_size % 2 == 0:
        raise ValueError("bridge_size must be a positive odd number")

    weak = ((z_map >= config.z_threshold) & safe_disk).astype(np.uint8)
    strong = (z_map >= config.z_threshold + 0.55) & safe_disk
    if config.bridge_size > 1:
        bridge = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (config.bridge_size, config.bridge_size)
        )
        weak = cv2.morphologyEx(weak, cv2.MORPH_CLOSE, bridge)
        weak[~safe_disk] = 0

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        weak, connectivity=8
    )
    min_reduced_area = max(20, config.min_instance_area // 4)
    image_stem = Path(file_name).stem
    predictions = []
    for component_id in range(1, component_count):
        reduced_area = int(stats[component_id, cv2.CC_STAT_AREA])
        if reduced_area < min_reduced_area:
            continue
        component = labels == component_id
        if int(np.count_nonzero(strong[component])) < 4:
            continue
        contours, _ = cv2.findContours(
            component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if contours:
            perimeter = max(float(cv2.arcLength(contours[0], True)), 1.0)
            compactness = 4.0 * np.pi * reduced_area / (perimeter * perimeter)
            if compactness > 0.78 and reduced_area < 180 and float(z_map[component].max()) > 6.5:
                continue

        x0 = int(stats[component_id, cv2.CC_STAT_LEFT])
        y0 = int(stats[component_id, cv2.CC_STAT_TOP])
        box_width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        box_height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        small_crop = component[y0 : y0 + box_height, x0 : x0 + box_width].astype(np.uint8)
        full_crop = cv2.resize(
            small_crop,
            (box_width * 2, box_height * 2),
            interpolation=cv2.INTER_LINEAR,
        ) > 0
        crop_y, crop_x = np.nonzero(full_crop)
        full_x = x0 * 2 + int(crop_x.min())
        full_y = y0 * 2 + int(crop_y.min())
        tight_crop = full_crop[
            crop_y.min() : crop_y.max() + 1,
            crop_x.min() : crop_x.max() + 1,
        ]
        area = int(np.count_nonzero(tight_crop))
        if area < config.min_instance_area:
            continue
        predictions.append(
            PredictedInstance(
                filament_id=f"{image_stem}_{len(predictions) + 1}",
                image_stem=image_stem,
                area=area,
                rle=encode_cropped_binary_mask(
                    tight_crop,
                    full_x,
                    full_y,
                    (IMAGE_SIZE, IMAGE_SIZE),
                ),
            )
        )
    return predictions


def predict_classical(
    gray_image: np.ndarray,
    file_name: str,
    config: ClassicalConfig | None = None,
) -> list[PredictedInstance]:
    config = config or ClassicalConfig()
    z_map, safe_disk = compute_salience_zmap(gray_image, config.limb_fraction)
    return predictions_from_zmap(z_map, safe_disk, file_name, config)
