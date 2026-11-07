from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from .data import IMAGE_SIZE
from .features import prepare_input_channels
from .model import FilamentUNet
from .predictor import PredictedInstance, encode_label_components


@dataclass(frozen=True)
class InstanceConfig:
    foreground_threshold: float = 0.35
    boundary_threshold: float = 0.5
    boundary_weight: float = 1.5
    seed_threshold: float = 0.45
    seed_min_area: int = 8
    min_instance_area: int = 50
    solar_radius: int = 950


def _tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if tile_size > length or stride < 1:
        raise ValueError("tile size must fit the image and stride must be positive")
    starts = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if not starts or starts[-1] != final:
        starts.append(final)
    return starts


def _transform(tensor: torch.Tensor, operation: int) -> torch.Tensor:
    if operation == 1:
        return torch.flip(tensor, dims=(-1,))
    if operation == 2:
        return torch.rot90(tensor, 2, dims=(-2, -1))
    if operation == 3:
        return torch.flip(tensor, dims=(-2,))
    if operation == 4:
        return torch.rot90(tensor, 1, dims=(-2, -1))
    if operation == 5:
        return torch.rot90(tensor, 3, dims=(-2, -1))
    if operation == 6:
        return torch.rot90(torch.flip(tensor, dims=(-1,)), 1, dims=(-2, -1))
    if operation == 7:
        return torch.rot90(torch.flip(tensor, dims=(-1,)), 3, dims=(-2, -1))
    return tensor


def _inverse_transform(tensor: torch.Tensor, operation: int) -> torch.Tensor:
    if operation == 4:
        return torch.rot90(tensor, 3, dims=(-2, -1))
    if operation == 5:
        return torch.rot90(tensor, 1, dims=(-2, -1))
    return _transform(tensor, operation)


@torch.inference_mode()
def predict_probabilities(
    model: FilamentUNet,
    gray_image: np.ndarray,
    *,
    device: torch.device | None = None,
    patch_size: int = 512,
    stride: int = 384,
    tta: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    if gray_image.ndim != 2 or gray_image.dtype != np.uint8:
        raise ValueError("Expected an 8-bit grayscale image")
    if gray_image.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"Expected {IMAGE_SIZE}x{IMAGE_SIZE}, got {gray_image.shape}")
    if patch_size % 32 or patch_size > IMAGE_SIZE:
        raise ValueError("patch_size must be divisible by 32 and no larger than the image")
    if tta not in (1, 2, 4, 8):
        raise ValueError("tta must be 1, 2, 4, or 8")
    device = device or next(model.parameters()).device
    model.eval()

    if model.in_channels not in (1, 2):
        raise ValueError(f"Unsupported model input channel count: {model.in_channels}")
    input_mode = "raw" if model.in_channels == 1 else "raw_local_contrast"
    normalized_channels = prepare_input_channels(
        gray_image, input_mode, solar_radius=950
    )

    one_dimensional = np.hanning(patch_size).astype(np.float32)
    one_dimensional = np.maximum(one_dimensional, 0.05)
    weight = np.outer(one_dimensional, one_dimensional)
    accumulation = np.zeros((2, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    weight_sum = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    operations = (0, 1, 2, 3, 4, 5, 6, 7)[:tta]
    gpu = device.type == "cuda"

    for y0 in _tile_starts(IMAGE_SIZE, patch_size, stride):
        for x0 in _tile_starts(IMAGE_SIZE, patch_size, stride):
            crop = normalized_channels[
                :, y0 : y0 + patch_size, x0 : x0 + patch_size
            ].copy()
            tensor = torch.from_numpy(crop[None]).to(device=device)
            if gpu:
                tensor = tensor.contiguous(memory_format=torch.channels_last)
            batch = torch.cat([_transform(tensor, op) for op in operations], dim=0)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=gpu,
            ):
                logits = model(batch)
            probabilities = logits.float().sigmoid().cpu()
            averaged = torch.zeros((2, patch_size, patch_size), dtype=torch.float32)
            for index, operation in enumerate(operations):
                averaged += _inverse_transform(probabilities[index], operation)
            averaged /= len(operations)
            region = np.s_[y0 : y0 + patch_size, x0 : x0 + patch_size]
            accumulation[:, region[0], region[1]] += averaged.numpy() * weight[None]
            weight_sum[region] += weight

    output = accumulation / np.maximum(weight_sum[None], 1e-6)
    return output[0], output[1]


def instances_from_probabilities(
    foreground_probability: np.ndarray,
    boundary_probability: np.ndarray,
    file_name: str,
    config: InstanceConfig | None = None,
) -> list[PredictedInstance]:
    config = config or InstanceConfig()
    if foreground_probability.shape != boundary_probability.shape:
        raise ValueError("Foreground and boundary probability maps must have equal shapes")
    height, width = foreground_probability.shape
    if (height, width) != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"Expected {IMAGE_SIZE}x{IMAGE_SIZE} probability maps")
    yy, xx = np.ogrid[:height, :width]
    disk = (yy - 1023.0) ** 2 + (xx - 1023.0) ** 2 <= config.solar_radius**2
    foreground = (foreground_probability >= config.foreground_threshold) & disk
    if not np.any(foreground):
        return []

    seed_mask = (
        foreground
        & (foreground_probability >= config.seed_threshold)
        & (boundary_probability < config.boundary_threshold)
    )
    markers, marker_count = ndi.label(seed_mask, structure=np.ones((3, 3), dtype=np.uint8))
    if marker_count:
        sizes = np.bincount(markers.ravel(), minlength=marker_count + 1)
        keep = sizes >= config.seed_min_area
        keep[0] = False
        seed_mask &= keep[markers]
        markers, marker_count = ndi.label(
            seed_mask, structure=np.ones((3, 3), dtype=np.uint8)
        )
    if marker_count == 0:
        markers, marker_count = ndi.label(
            foreground, structure=np.ones((3, 3), dtype=np.uint8)
        )
    elevation = -foreground_probability.astype(np.float32) + (
        config.boundary_weight * boundary_probability.astype(np.float32)
    )
    labels = watershed(
        elevation,
        markers.astype(np.int32),
        mask=foreground,
        connectivity=np.ones((3, 3), dtype=np.uint8),
        watershed_line=True,
    ).astype(np.int32, copy=False)
    counts = np.bincount(labels.ravel())
    accepted = [label for label in range(1, len(counts)) if counts[label] >= config.min_instance_area]
    if not accepted:
        return []

    first_by_label = np.full(len(counts), labels.size, dtype=np.int64)
    nonzero_positions = np.flatnonzero(labels.ravel())
    np.minimum.at(
        first_by_label,
        labels.ravel()[nonzero_positions],
        nonzero_positions,
    )
    first_positions = {label: int(first_by_label[label]) for label in accepted}
    accepted.sort(key=first_positions.__getitem__)
    encoded = encode_label_components(labels, accepted)
    image_stem = Path(file_name).stem
    return [
        PredictedInstance(
            filament_id=f"{image_stem}_{sequence}",
            image_stem=image_stem,
            area=int(counts[label]),
            rle=encoded[label],
        )
        for sequence, label in enumerate(accepted, start=1)
    ]
