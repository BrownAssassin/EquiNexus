# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from scipy import ndimage


@dataclass(frozen=True)
class DetectorFusionConfig:
    """Use detector instances to group foreground pixels from a full-resolution segmenter."""

    foreground_threshold: float = 0.5
    detection_threshold: float = 0.3
    grow_radius: int = 4
    minimum_claimed_area: int = 200
    closing_radius: int = 3
    solar_radius: int = 950
    keep_unclaimed_minimum_area: int = 0


def _decode_detection(counts: str | bytes, shape: tuple[int, int]) -> np.ndarray:
    encoded_counts = counts if isinstance(counts, bytes) else counts.encode("ascii")
    decoded = mask_utils.decode(
        {"size": list(shape), "counts": encoded_counts}
    )
    if decoded.ndim == 3:
        decoded = decoded[:, :, 0]
    return decoded.astype(bool)


def _disk_mask(shape: tuple[int, int], radius: int) -> np.ndarray:
    height, width = shape
    disk = np.zeros((height, width), dtype=np.uint8)
    center = ((width - 1) // 2, (height - 1) // 2)
    cv2.circle(disk, center, radius, color=1, thickness=-1)
    return disk.astype(bool)


def solar_disk_mask_from_geometry(
    shape: tuple[int, int],
    geometry: dict[str, float],
    *,
    shrink: float = 1.0,
) -> np.ndarray:
    """Build an image-derived solar disk mask from its fitted circle."""
    if not 0.0 < shrink <= 1.05:
        raise ValueError("shrink must be in (0, 1.05]")
    height, width = shape
    cx, cy = float(geometry["cx"]), float(geometry["cy"])
    radius = float(geometry["radius"]) * shrink
    if radius <= 0:
        raise ValueError("solar disk radius must be positive")
    yy, xx = np.ogrid[:height, :width]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2


def _grow_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or not np.any(mask):
        return mask
    ys, xs = np.nonzero(mask)
    y0, y1 = max(int(ys.min()) - radius, 0), min(int(ys.max()) + radius + 1, mask.shape[0])
    x0, x1 = max(int(xs.min()) - radius, 0), min(int(xs.max()) + radius + 1, mask.shape[1])
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    crop = mask[y0:y1, x0:x1].astype(np.uint8)
    grown = cv2.dilate(crop, kernel, iterations=1).astype(bool)
    output = np.zeros(mask.shape, dtype=bool)
    output[y0:y1, x0:x1] = grown
    return output


def fuse_detector_instances(
    foreground_probability: np.ndarray,
    detections: list[dict[str, Any]],
    config: DetectorFusionConfig | None = None,
    *,
    disk_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Assign U-Net foreground pixels to confident detector masks, one instance each.

    The detector supplies instance grouping and confidence; the foreground map
    supplies full-resolution boundaries. Overlapping detections claim pixels in
    descending confidence order, and each pixel can be assigned only once.
    """
    config = config or DetectorFusionConfig()
    if foreground_probability.ndim != 2:
        raise ValueError("foreground_probability must be a two-dimensional map")
    if not 0.0 <= config.foreground_threshold <= 1.0:
        raise ValueError("foreground_threshold must be between 0 and 1")
    if not 0.0 <= config.detection_threshold <= 1.0:
        raise ValueError("detection_threshold must be between 0 and 1")
    if min(
        config.grow_radius,
        config.minimum_claimed_area,
        config.closing_radius,
        config.keep_unclaimed_minimum_area,
    ) < 0:
        raise ValueError("Fusion sizes must be nonnegative")

    shape = foreground_probability.shape
    if disk_mask is None:
        disk = _disk_mask(shape, config.solar_radius)
    else:
        if disk_mask.shape != shape:
            raise ValueError("disk_mask must have the same shape as foreground_probability")
        disk = np.asarray(disk_mask, dtype=bool)
    binary = (foreground_probability >= config.foreground_threshold) & disk
    if config.closing_radius > 0:
        radius = config.closing_radius
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
        binary &= disk
    binary = ndimage.binary_fill_holes(binary) & disk

    labels = np.zeros(shape, dtype=np.int32)
    claimed = np.zeros(shape, dtype=bool)
    next_label = 1
    eligible = [
        item for item in detections if float(item["score"]) >= config.detection_threshold
    ]
    eligible.sort(key=lambda item: -float(item["score"]))
    for detection in eligible:
        region = _decode_detection(detection["counts"], shape)
        region = _grow_mask(region, config.grow_radius)
        pixels = binary & region & ~claimed
        area = int(pixels.sum())
        if area < config.minimum_claimed_area:
            continue
        labels[pixels] = next_label
        claimed[pixels] = True
        next_label += 1

    if config.keep_unclaimed_minimum_area > 0:
        components, component_count = ndimage.label(binary & ~claimed)
        sizes = np.bincount(components.ravel(), minlength=component_count + 1)
        for component_id in range(1, component_count + 1):
            if sizes[component_id] < config.keep_unclaimed_minimum_area:
                continue
            labels[components == component_id] = next_label
            next_label += 1
    return labels
