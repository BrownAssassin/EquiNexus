# Adapted and modified by EquiNexus from the Apache-2.0 Kaggle notebook.
# https://www.kaggle.com/code/phuongncn/lb-0-70-topology-safe-solar-filaments-handoff
# EquiNexus added local/Kaggle paths, a CLI, run metadata, and output validation.
# See licenses/THIRD_PARTY_NOTICES.md. The checkpoints are Apache-2.0.
# Source model: phuongncn/solar-filament-i079-weights.
# This is image-conditioned inference on the official 180-image H-alpha test set.
# It never reads annotations or any test labels.
"""Inference-only I079 Mask R-CNN and refiner pipeline with four-view TTA.

Uses MD5-locked Apache-2.0 public model assets and the official competition
test images. It never reads training annotations or any test labels. The same
script runs locally and in Kaggle kernels.
"""

from __future__ import annotations

import hashlib
import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
import importlib.metadata as importlib_metadata
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from workspace_runtime import configure_workspace_runtime

configure_workspace_runtime(REPO_ROOT)

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
from pycocotools import mask as mask_utils
from torch import nn
from torchvision.models import resnet18
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign


CANDIDATE_ID = "I079-topology-safe-refiner-flip4-tta-v1"
LOCAL_DATA_ROOT = REPO_ROOT / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026"
LOCAL_MODEL_ROOT = (
    REPO_ROOT / "data" / "reference" / "public-models"
    / "phuongncn-solar-filament-i079-weights"
)
LOCAL_OUTPUT_ROOT = (
    REPO_ROOT / "artifacts" / "model-experiments"
    / "i079-topology-safe-inference-20261113"
)
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_OUTPUT_ROOT = Path("/kaggle/working")
MODEL_INPUT_ROOT = LOCAL_MODEL_ROOT
TEST_INPUT_ROOT = LOCAL_DATA_ROOT
OUTPUT_DIR = LOCAL_OUTPUT_ROOT
SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"
REPORT_PATH = OUTPUT_DIR / "run_report.json"
EXPECTED_TEST_IMAGES = 180
EXPECTED_SUBMISSION_MD5 = "09c0eae5fdff29a91f8a740475ecaf36"
IMAGE_SIZE = 1024
DETECTOR_SCORE_THRESHOLD = 0.70
DETECTOR_MASK_THRESHOLD = 0.50
ANCHOR_REFINER_THRESHOLD = 0.85
CANDIDATE_REFINER_THRESHOLD = 0.70
MAXIMUM_SEED_IOU = 0.10
MINIMUM_UNIQUE_SEED_PIXELS = 16
MINIMUM_SEED_DISTANCE = 8.0
MAXIMUM_SPLIT_TASKS_PER_OBSERVATION = 1
REFINER_TTA_TRANSFORMS = ("identity", "hflip", "vflip", "hvflip")
REFINER_TTA_THRESHOLD = CANDIDATE_REFINER_THRESHOLD
TTA_AREA_RATIO_MIN = 0.60
TTA_AREA_RATIO_MAX = 1.70
TTA_DICE_MIN = 0.62
TTA_MIN_CONTROL_AREA_FOR_DICE = 32
FORBIDDEN_SELECTED_PATH_TOKENS = (
    "magfilo",
    "contamination_audit",
    "public_annotations",
    "public-label",
    "public_label",
    "official_labels",
)
EXPECTED_MD5 = {
    "final_detector.pt": "dcd3053561e40ba78718546c31525d26",
    "final_refiner.pt": "74991b8d95f7922f0ca6fd75f8ab4f5f",
}


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fail(message: str) -> None:
    raise RuntimeError(message)


def dependency_report() -> dict[str, str | bool]:
    """Print and return the offline package/runtime contract."""
    versions: dict[str, str | bool] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "opencv": cv2.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    try:
        versions["pycocotools"] = importlib_metadata.version("pycocotools")
    except importlib_metadata.PackageNotFoundError:
        versions["pycocotools"] = "unknown"
    if torch.cuda.is_available():
        versions["cuda_device_0"] = torch.cuda.get_device_name(0)
    print(json.dumps({"event": "dependency_report", **versions}, sort_keys=True), flush=True)
    return versions


def assert_selected_path_is_safe_checkpoint(path: Path) -> None:
    """Reject a checkpoint path that clearly indicates a label or mirror source."""
    lowered = str(path).lower()
    hits = [token for token in FORBIDDEN_SELECTED_PATH_TOKENS if token in lowered]
    if hits:
        fail(f"selected asset path looks leakage-prone ({hits}): {path}")


def locate_exact_file(name: str, expected_md5: str) -> Path:
    candidates = [
        path
        for path in sorted(MODEL_INPUT_ROOT.rglob(name))
        if path.is_file() and md5_file(path) == expected_md5
    ]
    if len(candidates) != 1:
        fail(f"expected one MD5-matching {name}, found {len(candidates)}")
    assert_selected_path_is_safe_checkpoint(candidates[0])
    return candidates[0]


def locate_test_images() -> list[Path]:
    valid: list[list[Path]] = []
    if TEST_INPUT_ROOT.name == "test_images":
        directories = [TEST_INPUT_ROOT]
    elif TEST_INPUT_ROOT.name == "MAGFiLO_1.0_Kaggle_2026":
        directories = sorted(TEST_INPUT_ROOT.rglob("test_images"))
    else:
        directories = sorted(TEST_INPUT_ROOT.rglob("test_images"))
    for directory in directories:
        if not directory.is_dir():
            continue
        images = sorted(
            path
            for path in directory.iterdir()
            if path.suffix.lower() in {".jpeg", ".jpg", ".png"}
        )
        if len(images) == EXPECTED_TEST_IMAGES:
            valid.append(images)
    if len(valid) != 1:
        fail(f"expected one official 180-image test directory, found {len(valid)}")
    return valid[0]


def encode_mask(mask: np.ndarray) -> dict[str, Any]:
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))


def positive_overlap_groups(masks: np.ndarray) -> list[list[int]]:
    values = np.asarray(masks, dtype=bool)
    if values.ndim != 3:
        raise ValueError("masks must have shape [N, H, W]")
    if any(not np.any(mask) for mask in values):
        raise ValueError("grouping requires active masks")
    if len(values) < 2:
        return [[index] for index in range(len(values))]
    rles = [encode_mask(mask) for mask in values]
    adjacency = np.asarray(
        mask_utils.iou(rles, rles, [0] * len(rles)),
        dtype=np.float64,
    ) > 0
    groups: list[list[int]] = []
    seen: set[int] = set()
    for start in range(len(values)):
        if start in seen:
            continue
        stack = [start]
        group: list[int] = []
        while stack:
            index = stack.pop()
            if index in seen:
                continue
            seen.add(index)
            group.append(index)
            stack.extend(np.flatnonzero(adjacency[index]).tolist())
        groups.append(group)
    return groups


def merge_positive_overlap_rles(
    rles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(rles) < 2:
        return list(rles)
    adjacency = np.asarray(
        mask_utils.iou(rles, rles, [0] * len(rles)),
        dtype=np.float64,
    ) > 0
    groups: list[list[int]] = []
    seen: set[int] = set()
    for start in range(len(rles)):
        if start in seen:
            continue
        stack = [start]
        group: list[int] = []
        while stack:
            index = stack.pop()
            if index in seen:
                continue
            seen.add(index)
            group.append(index)
            stack.extend(np.flatnonzero(adjacency[index]).tolist())
        groups.append(group)
    return [
        mask_utils.merge([rles[index] for index in group])
        for group in groups
    ]


def square_bounds(
    mask: np.ndarray,
    *,
    context: float = 1.8,
    minimum: int = 96,
) -> tuple[int, int, int, int]:
    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2 or not np.any(value):
        raise ValueError("square_bounds requires one active 2D mask")
    height, width = value.shape
    ys, xs = np.where(value)
    center_x = (xs.min() + xs.max() + 1) / 2
    center_y = (ys.min() + ys.max() + 1) / 2
    side = int(
        np.ceil(
            max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
            * context
        )
    )
    side = min(max(side, minimum), min(height, width))
    x0 = int(round(center_x - side / 2))
    y0 = int(round(center_y - side / 2))
    x1, y1 = x0 + side, y0 + side
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > width:
        x0 -= x1 - width
        x1 = width
    if y1 > height:
        y0 -= y1 - height
        y1 = height
    return int(x0), int(y0), int(x1), int(y1)


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.sum(first & second))
    union = int(np.sum(first | second))
    return intersection / union if union else 0.0


def split_mask_with_detector_seeds(
    refined_mask: np.ndarray,
    member_masks: list[np.ndarray],
    member_scores: np.ndarray,
    *,
    maximum_seed_iou: float = MAXIMUM_SEED_IOU,
    minimum_unique_seed_pixels: int = MINIMUM_UNIQUE_SEED_PIXELS,
    minimum_seed_distance: float = MINIMUM_SEED_DISTANCE,
) -> list[np.ndarray]:
    refined = np.asarray(refined_mask, dtype=bool)
    masks = [np.asarray(mask, dtype=bool) for mask in member_masks]
    scores = np.asarray(member_scores, dtype=np.float32)
    if refined.ndim != 2 or not np.any(refined):
        raise ValueError("refined mask must be active and 2D")
    if scores.shape != (len(masks),):
        raise ValueError("member score/mask counts differ")
    accepted: list[int] = []
    for index in sorted(range(len(masks)), key=lambda item: (-scores[item], item)):
        if all(
            _mask_iou(masks[index], masks[other]) <= maximum_seed_iou
            for other in accepted
        ):
            accepted.append(index)
    if len(accepted) < 2:
        return [refined]

    seed_points: list[tuple[int, int]] = []
    for index in accepted:
        others = np.zeros_like(refined)
        for other in accepted:
            if other != index:
                others |= masks[other]
        unique = masks[index] & refined & ~others
        ys, xs = np.nonzero(unique)
        if len(ys) < minimum_unique_seed_pixels:
            continue
        center_y, center_x = float(np.mean(ys)), float(np.mean(xs))
        nearest = int(np.argmin((ys - center_y) ** 2 + (xs - center_x) ** 2))
        seed_points.append((int(ys[nearest]), int(xs[nearest])))
    if len(seed_points) < 2:
        return [refined]
    for first in range(len(seed_points)):
        for second in range(first + 1, len(seed_points)):
            if (
                np.hypot(
                    seed_points[first][0] - seed_points[second][0],
                    seed_points[first][1] - seed_points[second][1],
                )
                < minimum_seed_distance
            ):
                return [refined]

    foreground_y, foreground_x = np.nonzero(refined)
    distances = np.stack(
        [
            (foreground_y - seed_y) ** 2 + (foreground_x - seed_x) ** 2
            for seed_y, seed_x in seed_points
        ],
        axis=1,
    )
    assignment = np.argmin(distances, axis=1)
    output = []
    for seed_index in range(len(seed_points)):
        selected = assignment == seed_index
        if not np.any(selected):
            return [refined]
        part = np.zeros_like(refined)
        part[foreground_y[selected], foreground_x[selected]] = True
        output.append(part)
    if not np.array_equal(np.logical_or.reduce(output), refined):
        raise RuntimeError("seed partition changed the foreground union")
    return output


def select_crowding_output(
    control_rles: list[dict[str, Any]],
    candidate_rles: list[dict[str, Any]],
    *,
    proposed_split_tasks: int,
) -> tuple[list[dict[str, Any]], bool]:
    abstain = proposed_split_tasks > MAXIMUM_SPLIT_TASKS_PER_OBSERVATION
    return list(control_rles if abstain else candidate_rles), abstain


def build_detector() -> nn.Module:
    model = maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
    box_features = model.roi_heads.box_predictor.cls_score.in_features
    mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden = model.roi_heads.mask_predictor.conv5_mask.out_channels
    model.roi_heads.box_predictor = FastRCNNPredictor(box_features, 2)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        mask_features, hidden, 2
    )
    model.rpn.anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )
    model.roi_heads.mask_roi_pool = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=28,
        sampling_ratio=2,
    )
    model.transform.min_size = (IMAGE_SIZE,)
    model.transform.max_size = IMAGE_SIZE
    return model


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class CropUNet(nn.Module):
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
        self.dec4 = ConvBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1 = ConvBlock(96, 32)
        self.up0 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.head = nn.Conv2d(16, 1, 1)
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = (image - self.mean) / self.std
        x0 = self.stem(image)
        x1 = self.layer1(self.pool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        value = self.dec4(torch.cat([self.up4(x4), x3], 1))
        value = self.dec3(torch.cat([self.up3(value), x2], 1))
        value = self.dec2(torch.cat([self.up2(value), x1], 1))
        value = self.dec1(torch.cat([self.up1(value), x0], 1))
        return self.head(self.up0(value))


def _apply_crop_transform(crops: np.ndarray, transform: str) -> np.ndarray:
    if transform == "identity":
        return np.ascontiguousarray(crops)
    if transform == "hflip":
        return np.ascontiguousarray(crops[:, :, ::-1])
    if transform == "vflip":
        return np.ascontiguousarray(crops[:, ::-1, :])
    if transform == "hvflip":
        return np.ascontiguousarray(crops[:, ::-1, ::-1])
    raise ValueError(f"unknown crop transform: {transform}")


def _invert_probability_transform(probabilities: np.ndarray, transform: str) -> np.ndarray:
    if transform == "identity":
        return np.ascontiguousarray(probabilities)
    if transform == "hflip":
        return np.ascontiguousarray(probabilities[:, :, ::-1])
    if transform == "vflip":
        return np.ascontiguousarray(probabilities[:, ::-1, :])
    if transform == "hvflip":
        return np.ascontiguousarray(probabilities[:, ::-1, ::-1])
    raise ValueError(f"unknown probability transform: {transform}")


def _run_refiner_batches(
    refiner: nn.Module,
    crops: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    if crops.ndim != 3 or crops.shape[1:] != (256, 256):
        raise ValueError(f"refiner crops must have shape [N, 256, 256], got {crops.shape}")
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(crops), batch_size):
            chunk = np.ascontiguousarray(crops[start : start + batch_size])
            batch = (
                torch.from_numpy(chunk)
                .float()
                .unsqueeze(1)
                .repeat(1, 3, 1, 1)
                .to(device, non_blocking=True)
                / 255
            )
            outputs.append(refiner(batch).sigmoid().cpu().numpy()[:, 0].astype(np.float32))
    return np.concatenate(outputs, axis=0)


def _component_count_8(mask: np.ndarray) -> int:
    value = np.asarray(mask, dtype=np.uint8)
    if value.ndim != 2:
        raise ValueError("component count requires a 2D mask")
    return int(cv2.connectedComponents(value, connectivity=8)[0] - 1)


def _mask_dice(first: np.ndarray, second: np.ndarray) -> float:
    first_value = np.asarray(first, dtype=bool)
    second_value = np.asarray(second, dtype=bool)
    denominator = int(first_value.sum() + second_value.sum())
    if denominator == 0:
        return 1.0
    return 2.0 * int(np.logical_and(first_value, second_value).sum()) / denominator


def select_topology_safe_tta_probability(
    single_probability: np.ndarray,
    tta_probability: np.ndarray,
) -> tuple[np.ndarray, bool, str]:
    """Use TTA only when thresholded topology remains close to the single-view mask."""
    single_mask = np.asarray(single_probability, dtype=np.float32) >= np.float32(REFINER_TTA_THRESHOLD)
    tta_mask = np.asarray(tta_probability, dtype=np.float32) >= np.float32(REFINER_TTA_THRESHOLD)
    single_area = int(single_mask.sum())
    tta_area = int(tta_mask.sum())
    if tta_area == 0:
        return single_probability, False, "empty_tta_mask"
    if single_area == 0:
        return single_probability, False, "empty_single_view_mask"
    area_ratio = tta_area / max(single_area, 1)
    if area_ratio < TTA_AREA_RATIO_MIN:
        return single_probability, False, "area_shrink_guard"
    if area_ratio > TTA_AREA_RATIO_MAX:
        return single_probability, False, "area_growth_guard"
    single_components = _component_count_8(single_mask)
    tta_components = _component_count_8(tta_mask)
    if tta_components > max(single_components + 1, 2):
        return single_probability, False, "fragmentation_guard"
    if single_area >= TTA_MIN_CONTROL_AREA_FOR_DICE:
        dice = _mask_dice(single_mask, tta_mask)
        if dice < TTA_DICE_MIN:
            return single_probability, False, "overlap_guard"
    return tta_probability, True, "accepted"


def infer_refiner_probabilities_with_tta(
    refiner: nn.Module,
    crops: np.ndarray,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str | bool]]]:
    """Return single-view probabilities, topology-safe TTA probabilities, and decisions."""
    single_probabilities: np.ndarray | None = None
    accumulator: np.ndarray | None = None
    for transform in REFINER_TTA_TRANSFORMS:
        transformed = _apply_crop_transform(crops, transform)
        probabilities = _run_refiner_batches(refiner, transformed, device=device)
        probabilities = _invert_probability_transform(probabilities, transform)
        if transform == "identity":
            single_probabilities = probabilities.astype(np.float32, copy=True)
        accumulator = probabilities.astype(np.float32, copy=False) if accumulator is None else accumulator + probabilities
    if single_probabilities is None or accumulator is None:
        raise RuntimeError("refiner TTA produced no probabilities")
    mean_probabilities = (accumulator / np.float32(len(REFINER_TTA_TRANSFORMS))).astype(np.float32)
    selected: list[np.ndarray] = []
    decisions: list[dict[str, str | bool]] = []
    for single_probability, tta_probability in zip(single_probabilities, mean_probabilities):
        chosen, accepted, reason = select_topology_safe_tta_probability(single_probability, tta_probability)
        selected.append(np.asarray(chosen, dtype=np.float32))
        decisions.append({"accepted": bool(accepted), "reason": str(reason)})
    return single_probabilities, np.stack(selected).astype(np.float32), decisions


def _decode_task(
    probability: np.ndarray,
    bounds: tuple[int, int, int, int],
    fallback: np.ndarray,
    *,
    threshold: float,
) -> np.ndarray:
    x0, y0, x1, y1 = bounds
    crop_mask = cv2.resize(
        np.asarray(probability, dtype=np.float32),
        (x1 - x0, y1 - y0),
        interpolation=cv2.INTER_LINEAR,
    ) >= np.float32(threshold)
    if not np.any(crop_mask):
        return np.asarray(fallback, dtype=bool)
    full = np.zeros(fallback.shape, dtype=bool)
    full[y0:y1, x0:x1] = crop_mask
    return full


def infer_observation(
    detector: nn.Module,
    refiner: nn.Module,
    image: np.ndarray,
    *,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    height, width = image.shape
    resized = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)
    tensor = (
        torch.from_numpy(resized)
        .unsqueeze(0)
        .repeat(3, 1, 1)
        .to(device)
        / 255
    )
    with torch.inference_mode():
        output = detector([tensor])[0]
    keep = output["scores"] >= DETECTOR_SCORE_THRESHOLD
    scores = output["scores"][keep].detach().cpu().numpy().astype(np.float32)
    masks = (
        output["masks"][keep, 0]
        .detach()
        .cpu()
        .numpy()
        >= DETECTOR_MASK_THRESHOLD
    )
    active = np.flatnonzero(np.any(masks, axis=(1, 2)))
    masks = np.ascontiguousarray(masks[active], dtype=bool)
    scores = np.ascontiguousarray(scores[active], dtype=np.float32)
    if not len(masks):
        return [], [], [], {
            "detector_masks": 0,
            "refiner_tasks": 0,
            "eligible_tasks": 0,
            "attached_proposed_split_tasks": 0,
            "attached_applied_split_tasks": 0,
            "attached_abstained": False,
            "proposed_split_tasks": 0,
            "applied_split_tasks": 0,
            "abstained": False,
            "tta_total_tasks": 0,
            "tta_accepted_tasks": 0,
            "tta_rejected_tasks": 0,
            "tta_rejection_reasons": {},
        }

    tasks = []
    for group in positive_overlap_groups(masks):
        model_fallback = np.logical_or.reduce([masks[index] for index in group])
        native_fallback = cv2.resize(
            model_fallback.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        bounds = square_bounds(native_fallback)
        x0, y0, x1, y1 = bounds
        crop = cv2.resize(
            image[y0:y1, x0:x1],
            (256, 256),
            interpolation=cv2.INTER_LINEAR,
        )
        member_masks = [
            cv2.resize(
                masks[index].astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            for index in group
        ]
        tasks.append(
            {
                "bounds": bounds,
                "crop": crop,
                "fallback": native_fallback,
                "member_masks": member_masks,
                "member_scores": scores[group],
            }
        )

    crops = np.stack([task["crop"] for task in tasks])
    single_probabilities, tta_probabilities, tta_decisions = (
        infer_refiner_probabilities_with_tta(refiner, crops, device=device)
    )

    anchor_task_rles = []
    attached_control_rles = []
    attached_split_rles = []
    tta_control_rles = []
    tta_split_rles = []
    eligible_tasks = 0
    attached_proposed_split_tasks = 0
    tta_proposed_split_tasks = 0
    tta_accepted_tasks = 0
    tta_rejection_reasons: dict[str, int] = {}
    for task, single_probability, tta_probability, decision in zip(
        tasks, single_probabilities, tta_probabilities, tta_decisions
    ):
        anchor_mask = _decode_task(
            single_probability,
            task["bounds"],
            task["fallback"],
            threshold=ANCHOR_REFINER_THRESHOLD,
        )
        attached_mask = _decode_task(
            single_probability,
            task["bounds"],
            task["fallback"],
            threshold=CANDIDATE_REFINER_THRESHOLD,
        )
        tta_mask = _decode_task(
            tta_probability,
            task["bounds"],
            task["fallback"],
            threshold=CANDIDATE_REFINER_THRESHOLD,
        )
        anchor_task_rles.append(encode_mask(anchor_mask))
        attached_control_rles.append(encode_mask(attached_mask))
        tta_control_rles.append(encode_mask(tta_mask))
        if bool(decision["accepted"]):
            tta_accepted_tasks += 1
        else:
            reason = str(decision["reason"])
            tta_rejection_reasons[reason] = tta_rejection_reasons.get(reason, 0) + 1
        if len(task["member_masks"]) >= 2:
            eligible_tasks += 1
            attached_parts = split_mask_with_detector_seeds(
                attached_mask,
                task["member_masks"],
                task["member_scores"],
            )
            tta_parts = split_mask_with_detector_seeds(
                tta_mask,
                task["member_masks"],
                task["member_scores"],
            )
        else:
            attached_parts = [attached_mask]
            tta_parts = [tta_mask]
        attached_proposed_split_tasks += int(len(attached_parts) > 1)
        tta_proposed_split_tasks += int(len(tta_parts) > 1)
        attached_split_rles.extend(encode_mask(part) for part in attached_parts)
        tta_split_rles.extend(encode_mask(part) for part in tta_parts)

    anchor = merge_positive_overlap_rles(anchor_task_rles)
    attached_control = merge_positive_overlap_rles(attached_control_rles)
    attached_split = merge_positive_overlap_rles(attached_split_rles)
    attached_candidate, attached_abstained = select_crowding_output(
        attached_control,
        attached_split,
        proposed_split_tasks=attached_proposed_split_tasks,
    )
    tta_control = merge_positive_overlap_rles(tta_control_rles)
    tta_split = merge_positive_overlap_rles(tta_split_rles)
    candidate, abstained = select_crowding_output(
        tta_control,
        tta_split,
        proposed_split_tasks=tta_proposed_split_tasks,
    )
    key = lambda rle: (
        float(mask_utils.toBbox(rle)[1]),
        float(mask_utils.toBbox(rle)[0]),
    )
    anchor.sort(key=key)
    attached_candidate.sort(key=key)
    candidate.sort(key=key)
    return anchor, attached_candidate, candidate, {
        "detector_masks": int(len(masks)),
        "refiner_tasks": int(len(tasks)),
        "eligible_tasks": int(eligible_tasks),
        "attached_proposed_split_tasks": int(attached_proposed_split_tasks),
        "attached_applied_split_tasks": int(0 if attached_abstained else attached_proposed_split_tasks),
        "attached_abstained": bool(attached_abstained),
        "proposed_split_tasks": int(tta_proposed_split_tasks),
        "applied_split_tasks": int(0 if abstained else tta_proposed_split_tasks),
        "abstained": bool(abstained),
        "tta_total_tasks": int(len(tasks)),
        "tta_accepted_tasks": int(tta_accepted_tasks),
        "tta_rejected_tasks": int(len(tasks) - tta_accepted_tasks),
        "tta_rejection_reasons": tta_rejection_reasons,
    }


def prediction_rows(
    predictions: dict[str, list[dict[str, Any]]],
) -> list[dict[str, str]]:
    rows = []
    for observation_id in sorted(predictions):
        for index, rle in enumerate(predictions[observation_id], 1):
            counts = rle["counts"]
            rows.append(
                {
                    "filament_id": f"{observation_id}_{index}",
                    "segmentation_rle": (
                        counts.decode("ascii")
                        if isinstance(counts, bytes)
                        else str(counts)
                    ),
                }
            )
    return rows


def validate_submission(frame: pd.DataFrame, test_ids: set[str]) -> dict[str, int]:
    if list(frame.columns) != ["filament_id", "segmentation_rle"]:
        fail("candidate columns are invalid")
    if frame.empty or not frame["filament_id"].is_unique:
        fail("candidate must be non-empty with unique filament IDs")
    if frame.isna().any().any():
        fail("candidate contains missing values")
    observations = {
        str(value).rsplit("_", 1)[0] for value in frame["filament_id"]
    }
    if not observations.issubset(test_ids):
        fail("candidate contains an unknown test observation")
    for counts in frame["segmentation_rle"]:
        mask = mask_utils.decode(
            {"size": [2048, 2048], "counts": str(counts).encode("ascii")}
        )
        if mask.shape != (2048, 2048) or not np.any(mask):
            fail("candidate contains an invalid or empty RLE")
    return {
        "rows": int(len(frame)),
        "covered_images": int(len(observations)),
        "missing_images": int(len(test_ids - observations)),
    }


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a checkpoint on Kaggle across torch versions without network access."""
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        fail(f"checkpoint is not a dictionary: {path}")
    return value


def dataframe_difference_summary(
    first: pd.DataFrame,
    second: pd.DataFrame,
    *,
    first_name: str,
    second_name: str,
) -> dict[str, int | bool]:
    summary: dict[str, int | bool] = {
        f"{first_name}_rows": int(len(first)),
        f"{second_name}_rows": int(len(second)),
        "same_row_count": bool(len(first) == len(second)),
    }
    if len(first) == len(second):
        summary["same_filament_ids_by_position"] = int(
            np.sum(first["filament_id"].to_numpy() == second["filament_id"].to_numpy())
        )
        summary["same_rle_by_position"] = int(
            np.sum(first["segmentation_rle"].to_numpy() == second["segmentation_rle"].to_numpy())
        )
    else:
        summary["same_filament_ids_by_position"] = 0
        summary["same_rle_by_position"] = 0
    return summary


def merge_reason_counts(stats: list[dict[str, Any]]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for row in stats:
        reasons = row.get("tta_rejection_reasons", {})
        if isinstance(reasons, dict):
            for key, value in reasons.items():
                merged[str(key)] = merged.get(str(key), 0) + int(value)
    return merged


def repository_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def main() -> None:
    global MODEL_INPUT_ROOT, TEST_INPUT_ROOT, OUTPUT_DIR
    global SUBMISSION_PATH, REPORT_PATH

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--test-images", type=Path, default=None)
    parser.add_argument("--weights-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--allow-output-drift", action="store_true")
    args = parser.parse_args()

    kaggle_runtime = KAGGLE_INPUT_ROOT.is_dir()
    data_root = args.data_root
    if data_root is None:
        data_root = KAGGLE_INPUT_ROOT if kaggle_runtime else LOCAL_DATA_ROOT
    MODEL_INPUT_ROOT = args.weights_root or (
        KAGGLE_INPUT_ROOT if kaggle_runtime else LOCAL_MODEL_ROOT
    )
    if args.test_images is not None:
        TEST_INPUT_ROOT = args.test_images
    elif kaggle_runtime and args.data_root is None:
        TEST_INPUT_ROOT = KAGGLE_INPUT_ROOT
    else:
        TEST_INPUT_ROOT = data_root
    OUTPUT_DIR = args.output_dir or (
        KAGGLE_OUTPUT_ROOT if kaggle_runtime else LOCAL_OUTPUT_ROOT
    )
    if not kaggle_runtime and not OUTPUT_DIR.resolve().is_relative_to(
        (REPO_ROOT / "artifacts").resolve()
    ):
        parser.error("local --output-dir must stay inside artifacts/")
    SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"
    REPORT_PATH = OUTPUT_DIR / "run_report.json"

    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dependency_versions = dependency_report()
    os.environ["PYTHONHASHSEED"] = "42"
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True
    if not torch.cuda.is_available():
        fail("CUDA is required")
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    detector_path = locate_exact_file(
        "final_detector.pt", EXPECTED_MD5["final_detector.pt"]
    )
    refiner_path = locate_exact_file(
        "final_refiner.pt", EXPECTED_MD5["final_refiner.pt"]
    )
    test_files = locate_test_images()
    test_ids = {path.stem for path in test_files}
    asset_report = {
        "event": "asset_discovery",
        "detector_path": str(detector_path),
        "detector_md5": md5_file(detector_path),
        "refiner_path": str(refiner_path),
        "refiner_md5": md5_file(refiner_path),
        "test_images": len(test_files),
        "test_directory": str(test_files[0].parent if test_files else ""),
    }
    print(json.dumps(asset_report, sort_keys=True), flush=True)

    detector_checkpoint = load_checkpoint(detector_path)
    refiner_checkpoint = load_checkpoint(refiner_path)
    if (
        set(detector_checkpoint) != {"epoch", "model"}
        or int(detector_checkpoint["epoch"]) != 6
        or len(detector_checkpoint["model"]) != 432
        or set(refiner_checkpoint) != {"epoch", "model"}
        or int(refiner_checkpoint["epoch"]) != 8
        or len(refiner_checkpoint["model"]) != 182
    ):
        fail("owned checkpoint schema mismatch")
    detector = build_detector().to(device)
    refiner = CropUNet().to(device)
    detector.load_state_dict(detector_checkpoint["model"], strict=True)
    refiner.load_state_dict(refiner_checkpoint["model"], strict=True)
    detector.eval()
    refiner.eval()

    anchor_predictions: dict[str, list[dict[str, Any]]] = {}
    attached_predictions: dict[str, list[dict[str, Any]]] = {}
    candidate_predictions: dict[str, list[dict[str, Any]]] = {}
    stats = []
    for index, image_path in enumerate(test_files, 1):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (2048, 2048):
            fail(f"test image decode mismatch: {image_path.name}")
        anchor, attached, candidate, row = infer_observation(
            detector, refiner, image, device=device
        )
        anchor_predictions[image_path.stem] = anchor
        attached_predictions[image_path.stem] = attached
        candidate_predictions[image_path.stem] = candidate
        stats.append({"observation_id": image_path.stem, **row})
        if index % 10 == 0 or index == len(test_files):
            print(
                json.dumps(
                    {
                        "event": "heartbeat",
                        "images_completed": index,
                        "total_images": len(test_files),
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    anchor_frame = pd.DataFrame(
        prediction_rows(anchor_predictions),
        columns=["filament_id", "segmentation_rle"],
    )
    attached_frame = pd.DataFrame(
        prediction_rows(attached_predictions),
        columns=["filament_id", "segmentation_rle"],
    )
    anchor_validation = validate_submission(anchor_frame, test_ids)
    attached_validation = validate_submission(attached_frame, test_ids)

    candidate_frame = pd.DataFrame(
        prediction_rows(candidate_predictions),
        columns=["filament_id", "segmentation_rle"],
    )
    validation = validate_submission(candidate_frame, test_ids)
    anchor_csv = anchor_frame.to_csv(index=False).encode("utf-8")
    attached_csv = attached_frame.to_csv(index=False).encode("utf-8")
    candidate_csv = candidate_frame.to_csv(index=False).encode("utf-8")
    if candidate_csv == anchor_csv:
        fail("candidate unexpectedly equals the same-detector 0.85 control")
    if candidate_csv == attached_csv:
        fail("candidate unexpectedly equals the attached no-TTA champion control")
    candidate_frame.to_csv(SUBMISSION_PATH, index=False)
    actual_submission_md5 = md5_file(SUBMISSION_PATH)
    output_hash_matches_reference = actual_submission_md5 == EXPECTED_SUBMISSION_MD5
    if not output_hash_matches_reference and not args.allow_output_drift:
        fail(
            "reproduction identity mismatch: "
            f"expected {EXPECTED_SUBMISSION_MD5}, got {actual_submission_md5}"
        )

    report = {
        "candidate_id": CANDIDATE_ID,
        "candidate_mechanism": "topology-safe 4-view flip TTA on refiner crop probabilities",
        "verdict": "PASS-CANDIDATE",
        "submission_path": str(SUBMISSION_PATH),
        "submission_md5": md5_file(SUBMISSION_PATH),
        "submission_sha256": hashlib.sha256(SUBMISSION_PATH.read_bytes()).hexdigest(),
        "submission_md5_matches_reference": output_hash_matches_reference,
        "expected_submission_md5": EXPECTED_SUBMISSION_MD5,
        "validation": validation,
        "dependency_versions": dependency_versions,
        "asset_report": asset_report,
        "leakage_policy": {
            "reads_training_annotations": False,
            "reads_public_magfilo_or_mirror_labels": False,
            "uses_public_test_labels_or_hash_matches": False,
            "selected_assets_are_md5_locked": True,
            "offline_execution": True,
            "public_handoff_ready": True,
        },
        "anchor_control": {
            "detector_score_threshold": DETECTOR_SCORE_THRESHOLD,
            "refiner_threshold": ANCHOR_REFINER_THRESHOLD,
            "submission_md5": hashlib.md5(
                anchor_csv, usedforsecurity=False
            ).hexdigest(),
            "validation": anchor_validation,
        },
        "attached_champion_control": {
            "candidate_id": "I078-fususu-t070-seeds010-single-active-v1",
            "refiner_threshold": CANDIDATE_REFINER_THRESHOLD,
            "submission_md5": hashlib.md5(
                attached_csv, usedforsecurity=False
            ).hexdigest(),
            "validation": attached_validation,
            "difference_vs_candidate": dataframe_difference_summary(
                attached_frame, candidate_frame,
                first_name="attached",
                second_name="candidate",
            ),
        },
        "candidate_rows": int(len(candidate_frame)),
        "candidate_row_delta_vs_anchor": int(
            len(candidate_frame) - len(anchor_frame)
        ),
        "proposed_split_tasks": int(
            sum(int(row["proposed_split_tasks"]) for row in stats)
        ),
        "applied_split_tasks": int(
            sum(int(row["applied_split_tasks"]) for row in stats)
        ),
        "active_split_observations": int(
            sum(int(row["applied_split_tasks"]) > 0 for row in stats)
        ),
        "attached_proposed_split_tasks": int(
            sum(int(row["attached_proposed_split_tasks"]) for row in stats)
        ),
        "attached_applied_split_tasks": int(
            sum(int(row["attached_applied_split_tasks"]) for row in stats)
        ),
        "tta_total_tasks": int(sum(int(row["tta_total_tasks"]) for row in stats)),
        "tta_accepted_tasks": int(sum(int(row["tta_accepted_tasks"]) for row in stats)),
        "tta_rejected_tasks": int(sum(int(row["tta_rejected_tasks"]) for row in stats)),
        "tta_rejection_reasons": merge_reason_counts(stats),
        "abstained_observations": [
            str(row["observation_id"]) for row in stats if row["abstained"]
        ],
        "attached_abstained_observations": [
            str(row["observation_id"]) for row in stats if row["attached_abstained"]
        ],
        "policy": {
            "detector_score_threshold": DETECTOR_SCORE_THRESHOLD,
            "detector_mask_threshold": DETECTOR_MASK_THRESHOLD,
            "refiner_threshold": CANDIDATE_REFINER_THRESHOLD,
            "maximum_seed_iou": MAXIMUM_SEED_IOU,
            "minimum_unique_seed_pixels": MINIMUM_UNIQUE_SEED_PIXELS,
            "minimum_seed_distance": MINIMUM_SEED_DISTANCE,
            "maximum_split_tasks_per_observation": (
                MAXIMUM_SPLIT_TASKS_PER_OBSERVATION
            ),
            "refiner_tta_transforms": REFINER_TTA_TRANSFORMS,
            "tta_area_ratio_min": TTA_AREA_RATIO_MIN,
            "tta_area_ratio_max": TTA_AREA_RATIO_MAX,
            "tta_dice_min": TTA_DICE_MIN,
        },
        "source_md5": EXPECTED_MD5,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_commit": repository_commit(),
        "frozen_split_digest": None,
        "validation_status": "test-only inference; no EquiNexus holdout score",
        "elapsed_seconds": time.perf_counter() - started,
        "gpu": torch.cuda.get_device_name(0),
        "peak_gpu_memory_allocated_bytes": int(
            torch.cuda.max_memory_allocated(device)
        ),
        "peak_gpu_memory_reserved_bytes": int(
            torch.cuda.max_memory_reserved(device)
        ),
        "training_executed": False,
    }
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
