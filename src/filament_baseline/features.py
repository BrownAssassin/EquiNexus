from __future__ import annotations

from typing import Literal

import cv2
import numpy as np


InputMode = Literal["raw", "raw_local_contrast"]
LOCAL_CONTRAST_SIGMA = 8.0


def prepare_input_channels(
    gray_image: np.ndarray,
    input_mode: InputMode = "raw",
    *,
    solar_radius: int = 950,
) -> np.ndarray:
    """Build normalized channels from one 8-bit H-alpha observation."""
    if gray_image.ndim != 2 or gray_image.dtype != np.uint8:
        raise ValueError("Expected an 8-bit grayscale image")
    if input_mode not in ("raw", "raw_local_contrast"):
        raise ValueError(f"Unsupported input mode: {input_mode}")

    height, width = gray_image.shape
    yy, xx = np.ogrid[:height, :width]
    center_y = float((height - 1) // 2)
    center_x = float((width - 1) // 2)
    disk = (yy - center_y) ** 2 + (xx - center_x) ** 2 <= solar_radius**2
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
    return np.stack((raw_normalized, contrast_normalized)).astype(np.float32, copy=False)
