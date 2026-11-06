from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pycocotools import mask as mask_utils

from .data import IMAGE_SIZE


@dataclass(frozen=True)
class BaselineConfig:
    blur_kernel: int = 31
    solar_disk_radius_fraction: float = 0.44
    closing_kernel: int = 3
    component_connectivity: int = 8
    min_component_area: int = 50
    max_component_area: int = 50_000


@dataclass(frozen=True)
class PredictedInstance:
    filament_id: str
    image_stem: str
    area: int
    rle: dict[str, Any]


def encode_binary_mask(binary_mask: np.ndarray) -> dict[str, Any]:
    if binary_mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {binary_mask.shape}")
    contiguous = np.asfortranarray((binary_mask > 0).astype(np.uint8))
    encoded = mask_utils.encode(contiguous)
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": [int(binary_mask.shape[0]), int(binary_mask.shape[1])], "counts": counts}


def decode_rle(rle: dict[str, Any]) -> np.ndarray:
    encoded = dict(rle)
    if isinstance(encoded["counts"], str):
        encoded["counts"] = encoded["counts"].encode("ascii")
    return mask_utils.decode(encoded).astype(np.uint8, copy=False)


def encode_label_components(
    labels: np.ndarray,
    component_labels: list[int],
) -> dict[int, dict[str, Any]]:
    """Encode selected connected components from one label image in a single scan."""
    height, width = labels.shape
    flat_labels = labels.reshape(-1, order="F")
    foreground_positions = np.flatnonzero(flat_labels)
    if not component_labels or foreground_positions.size == 0:
        return {}
    foreground_labels = flat_labels[foreground_positions]
    order = np.argsort(foreground_labels, kind="stable")
    sorted_positions = foreground_positions[order]
    sorted_labels = foreground_labels[order]
    unique_labels, group_starts = np.unique(sorted_labels, return_index=True)
    group_ends = np.r_[group_starts[1:], len(sorted_labels)]
    result: dict[int, dict[str, Any]] = {}
    total_pixels = height * width

    for label in component_labels:
        group_index = int(np.searchsorted(unique_labels, label))
        if group_index >= len(unique_labels) or int(unique_labels[group_index]) != label:
            continue
        positions = sorted_positions[group_starts[group_index] : group_ends[group_index]]
        boundaries = np.flatnonzero(np.diff(positions) > 1) + 1
        # The exclusive ends are one past the last foreground pixel in each run.
        starts_at = np.r_[0, boundaries]
        ends_at = np.r_[boundaries, len(positions)]
        counts: list[int] = []
        cursor = 0
        for begin, end in zip(starts_at, ends_at):
            foreground_start = int(positions[begin])
            foreground_end = int(positions[end - 1]) + 1
            counts.append(foreground_start - cursor)
            counts.append(foreground_end - foreground_start)
            cursor = foreground_end
        tail = total_pixels - cursor
        if tail:
            counts.append(tail)
        uncompressed = {"size": [height, width], "counts": counts}
        encoded = mask_utils.frPyObjects(uncompressed, height, width)
        if isinstance(encoded, list):
            encoded = encoded[0]
        result[label] = _coco_encoded_rle(encoded)
    return result


def _coco_encoded_rle(encoded: dict[str, Any]) -> dict[str, Any]:
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": [int(encoded["size"][0]), int(encoded["size"][1])], "counts": counts}


def solar_disk_mask(height: int, width: int, radius_fraction: float) -> np.ndarray:
    radius = int(round(radius_fraction * min(height, width)))
    center = (width // 2, height // 2)
    disk = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(disk, center, radius, color=1, thickness=-1, lineType=cv2.LINE_8)
    return disk.astype(bool)


def predict_image(
    gray_image: np.ndarray,
    file_name: str,
    config: BaselineConfig | None = None,
) -> list[PredictedInstance]:
    config = config or BaselineConfig()
    if gray_image.ndim != 2 or gray_image.dtype != np.uint8:
        raise ValueError("Expected an 8-bit grayscale image")
    height, width = gray_image.shape
    if height != IMAGE_SIZE or width != IMAGE_SIZE:
        raise ValueError(f"Expected a {IMAGE_SIZE}x{IMAGE_SIZE} image, got {width}x{height}")
    if config.blur_kernel < 1 or config.blur_kernel % 2 == 0:
        raise ValueError("blur_kernel must be a positive odd integer")
    if config.closing_kernel < 1 or config.closing_kernel % 2 == 0:
        raise ValueError("closing_kernel must be a positive odd integer")

    blurred = cv2.GaussianBlur(gray_image, (config.blur_kernel, config.blur_kernel), 0)
    response = cv2.subtract(blurred, gray_image)
    disk = solar_disk_mask(height, width, config.solar_disk_radius_fraction)
    disk_values = response[disk]
    if disk_values.size == 0:
        raise ValueError("Solar disk region is empty")
    threshold, _ = cv2.threshold(
        disk_values.reshape(-1, 1),
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )
    binary = ((response > threshold) & disk).astype(np.uint8)
    kernel = np.ones((config.closing_kernel, config.closing_kernel), dtype=np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    closed[~disk] = 0

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        closed, connectivity=config.component_connectivity
    )
    accepted_labels = [
        label
        for label in range(1, count)
        if config.min_component_area
        <= int(stats[label, cv2.CC_STAT_AREA])
        <= config.max_component_area
    ]
    encoded_components = encode_label_components(labels, accepted_labels)
    image_stem = Path(file_name).stem
    predictions: list[PredictedInstance] = []
    for sequence, label in enumerate(accepted_labels, start=1):
        area = int(stats[label, cv2.CC_STAT_AREA])
        predictions.append(
            PredictedInstance(
                filament_id=f"{image_stem}_{sequence}",
                image_stem=image_stem,
                area=area,
                rle=encoded_components[label],
            )
        )
    return predictions


def read_grayscale(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if image.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"Expected a 2048x2048 image at {path}, got {image.shape}")
    return image


def write_prediction_overlay(
    gray_image: np.ndarray,
    predictions: list[PredictedInstance],
    output_path: str | Path,
) -> None:
    canvas = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    color_layer = canvas.copy()
    for index, prediction in enumerate(predictions):
        hue = (index * 37 + 11) % 180
        hsv = np.uint8([[[hue, 220, 255]]])
        color = tuple(int(value) for value in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])
        mask = decode_rle(prediction.rle)
        color_layer[mask > 0] = color
    canvas = cv2.addWeighted(canvas, 0.62, color_layer, 0.38, 0)
    cv2.putText(
        canvas,
        f"Predicted instances: {len(predictions)}",
        (32, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise OSError(f"Could not write prediction overlay: {output_path}")
