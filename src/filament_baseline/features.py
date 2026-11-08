# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import cv2
import numpy as np


InputMode = Literal["raw", "raw_local_contrast", "raw_local_contrast_radial"]
INPUT_MODES: tuple[InputMode, ...] = (
    "raw",
    "raw_local_contrast",
    "raw_local_contrast_radial",
)
LOCAL_CONTRAST_SIGMA = 8.0
RADIAL_PROFILE_BINS = 256
RADIAL_PROFILE_SAMPLE_STRIDE = 4


def input_channel_count(input_mode: InputMode) -> int:
    try:
        return {
            "raw": 1,
            "raw_local_contrast": 2,
            "raw_local_contrast_radial": 3,
        }[input_mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported input mode: {input_mode}") from exc


def input_mode_for_channel_count(channel_count: int) -> InputMode:
    try:
        return {
            1: "raw",
            2: "raw_local_contrast",
            3: "raw_local_contrast_radial",
        }[channel_count]
    except KeyError as exc:
        raise ValueError(f"Unsupported model input channel count: {channel_count}") from exc


@lru_cache(maxsize=4)
def _solar_geometry(
    height: int, width: int, solar_radius: int
) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.ogrid[:height, :width]
    center_y = float((height - 1) // 2)
    center_x = float((width - 1) // 2)
    radius_squared = (yy - center_y) ** 2 + (xx - center_x) ** 2
    disk = radius_squared <= solar_radius**2
    radius = np.sqrt(radius_squared)
    radial_bin = np.minimum(
        (radius * (RADIAL_PROFILE_BINS / max(solar_radius, 1))).astype(np.int16),
        RADIAL_PROFILE_BINS - 1,
    )
    disk.setflags(write=False)
    radial_bin.setflags(write=False)
    return disk, radial_bin


def _radial_contrast(
    scaled: np.ndarray, disk: np.ndarray, radial_bin: np.ndarray
) -> np.ndarray:
    """Normalize dark structure against the median intensity at each solar radius."""
    sampled_scaled = scaled[::RADIAL_PROFILE_SAMPLE_STRIDE, ::RADIAL_PROFILE_SAMPLE_STRIDE]
    sampled_disk = disk[::RADIAL_PROFILE_SAMPLE_STRIDE, ::RADIAL_PROFILE_SAMPLE_STRIDE]
    sampled_bins = radial_bin[
        ::RADIAL_PROFILE_SAMPLE_STRIDE, ::RADIAL_PROFILE_SAMPLE_STRIDE
    ]
    intensity = np.rint(sampled_scaled * 255.0).clip(0, 255).astype(np.uint8)
    packed_bins = (
        sampled_bins[sampled_disk].astype(np.int32) * 256
        + intensity[sampled_disk].astype(np.int32)
    )
    histogram = np.bincount(packed_bins, minlength=RADIAL_PROFILE_BINS * 256).reshape(
        RADIAL_PROFILE_BINS, 256
    )
    counts = histogram.sum(axis=1)
    cumulative = histogram.cumsum(axis=1)
    median_bins = np.argmax(cumulative >= ((counts + 1) // 2)[:, None], axis=1)
    populated = counts > 0
    profile = np.interp(
        np.arange(RADIAL_PROFILE_BINS),
        np.flatnonzero(populated),
        median_bins[populated].astype(np.float32) / 255.0,
    ).astype(np.float32)
    profile = cv2.GaussianBlur(profile[None, :], (0, 0), sigmaX=2.0)[0]

    radial_contrast = profile[radial_bin] - scaled
    disk_values = radial_contrast[disk]
    mean = float(np.mean(disk_values))
    std = max(float(np.std(disk_values)), 1e-3)
    normalized = np.clip((radial_contrast - mean) / std, -5.0, 5.0)
    normalized[~disk] = 0.0
    return normalized.astype(np.float32, copy=False)


def prepare_input_channels(
    gray_image: np.ndarray,
    input_mode: InputMode = "raw",
    *,
    solar_radius: int = 950,
) -> np.ndarray:
    """Build normalized channels from one 8-bit H-alpha observation."""
    if gray_image.ndim != 2 or gray_image.dtype != np.uint8:
        raise ValueError("Expected an 8-bit grayscale image")
    input_channel_count(input_mode)

    height, width = gray_image.shape
    disk, radial_bin = _solar_geometry(height, width, solar_radius)
    if not np.any(disk):
        raise ValueError("Solar disk mask is empty for the supplied image and radius")

    scaled = gray_image.astype(np.float32) / 255.0
    disk_values = scaled[disk]
    image_mean = float(np.mean(disk_values))
    image_std = max(float(np.std(disk_values)), 1e-3)
    raw_normalized = np.clip((scaled - image_mean) / image_std, -5.0, 5.0)
    if input_mode == "raw":
        return raw_normalized[None].astype(np.float32, copy=False)

    blurred = cv2.GaussianBlur(
        scaled, (0, 0), sigmaX=LOCAL_CONTRAST_SIGMA, sigmaY=LOCAL_CONTRAST_SIGMA
    )
    local_contrast = blurred - scaled
    contrast_values = local_contrast[disk]
    contrast_mean = float(np.mean(contrast_values))
    contrast_std = max(float(np.std(contrast_values)), 1e-3)
    contrast_normalized = np.clip(
        (local_contrast - contrast_mean) / contrast_std, -5.0, 5.0
    )
    channels = [raw_normalized, contrast_normalized]
    if input_mode == "raw_local_contrast_radial":
        channels.append(_radial_contrast(scaled, disk, radial_bin))
    return np.stack(channels).astype(np.float32, copy=False)
