# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Train and run an observation-grouped YOLO-seeded crop refiner."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from .data import CompetitionDataset, IMAGE_SIZE, ImageView
from .predictor import PredictedInstance, encode_binary_mask, read_grayscale
from .scorer import annotation_to_rle


@dataclass(frozen=True)
class SeedRefinerConfig:
    crop_size: int = 512
    crop_context: float = 1.8
    seed_padding: float = 0.18
    detection_score_min: float = 0.10
    match_iou_min: float = 0.05
    validation_score_min: float = 0.25
    probability_threshold: float = 0.50
    minimum_instance_area: int = 200
    box_jitter_px: int = 8
    training_steps: int = 3000
    batch_size: int = 8
    learning_rate: float = 0.0002
    seed: int = 42


@dataclass(frozen=True)
class SeedCropSample:
    file_name: str
    image_id: str
    detection_counts: str
    detection_score: float
    bbox: tuple[float, float, float, float]
    target_segmentation: list[Any] | dict[str, Any] | None


def _rle(counts: str | bytes, shape: tuple[int, int] = (IMAGE_SIZE, IMAGE_SIZE)) -> dict[str, Any]:
    return {
        "size": [int(shape[0]), int(shape[1])],
        "counts": counts if isinstance(counts, bytes) else counts.encode("ascii"),
    }


def _polygon_rles(annotations: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    rles = []
    for annotation in annotations:
        rle = annotation_to_rle(annotation, IMAGE_SIZE, IMAGE_SIZE)
        rles.append(_rle(rle["counts"], (rle["size"][0], rle["size"][1])))
    return rles


def _build_training_samples(
    dataset: CompetitionDataset,
    observations: tuple[str, ...],
    detector_predictions: dict[str, list[dict[str, Any]]],
    config: SeedRefinerConfig,
) -> tuple[SeedCropSample, ...]:
    samples: list[SeedCropSample] = []
    for file_name in sorted(observations):
        detections = [
            item
            for item in detector_predictions.get(Path(file_name).stem, [])
            if float(item["score"]) >= config.detection_score_min
        ]
        if not detections:
            continue
        views: list[tuple[ImageView, tuple[dict[str, Any], ...], list[dict[str, Any]]]] = []
        for view in dataset.views_by_observation[file_name]:
            annotations = dataset.annotations_by_image_id[view.image_id]
            views.append((view, annotations, _polygon_rles(annotations)))
        for detection in detections:
            detection_rle = _rle(detection["counts"])
            bbox = tuple(float(value) for value in mask_utils.toBbox(detection_rle))
            if bbox[2] <= 0 or bbox[3] <= 0:
                continue
            for view, annotations, gt_rles in views:
                overlaps = (
                    mask_utils.iou([detection_rle], gt_rles, [0] * len(gt_rles))[0]
                    if gt_rles
                    else np.zeros(0, dtype=np.float64)
                )
                target_index = int(np.argmax(overlaps)) if overlaps.size else -1
                target = (
                    annotations[target_index]["segmentation"]
                    if target_index >= 0 and float(overlaps[target_index]) >= config.match_iou_min
                    else None
                )
                samples.append(
                    SeedCropSample(
                        file_name=file_name,
                        image_id=view.image_id,
                        detection_counts=str(detection["counts"]),
                        detection_score=float(detection["score"]),
                        bbox=bbox,
                        target_segmentation=target,
                    )
                )
    if not samples:
        raise ValueError("No training crop samples were constructed")
    return tuple(samples)


def _crop_bounds(
    bbox: tuple[float, float, float, float],
    image_shape: tuple[int, int],
    context: float,
) -> tuple[int, int, int]:
    x, y, width, height = bbox
    image_height, image_width = image_shape
    center_x, center_y = x + width / 2.0, y + height / 2.0
    side = min(
        max(int(math.ceil(max(width, height) * context)), 96),
        min(image_height, image_width),
    )
    x0 = min(max(int(round(center_x - side / 2.0)), 0), image_width - side)
    y0 = min(max(int(round(center_y - side / 2.0)), 0), image_height - side)
    return x0, y0, side


def _seed_box(
    bbox: tuple[float, float, float, float],
    image_shape: tuple[int, int],
    padding: float,
    jitter_x: float = 0.0,
    jitter_y: float = 0.0,
    scale: float = 1.0,
) -> tuple[float, float, float, float]:
    x, y, width, height = bbox
    center_x = x + width / 2.0 + jitter_x
    center_y = y + height / 2.0 + jitter_y
    width = max(1.0, width * (1.0 + 2.0 * padding) * scale)
    height = max(1.0, height * (1.0 + 2.0 * padding) * scale)
    image_height, image_width = image_shape
    x0 = max(0.0, center_x - width / 2.0)
    y0 = max(0.0, center_y - height / 2.0)
    x1 = min(float(image_width), center_x + width / 2.0)
    y1 = min(float(image_height), center_y + height / 2.0)
    return x0, y0, x1, y1


def _fill_segmentation_crop(
    segmentation: list[Any] | dict[str, Any] | None,
    bounds: tuple[int, int, int],
    output_size: int,
) -> np.ndarray:
    x0, y0, side = bounds
    if segmentation is None:
        return np.zeros((output_size, output_size), dtype=np.uint8)
    if isinstance(segmentation, dict):
        rle = _rle(segmentation["counts"], tuple(segmentation["size"]))
        full_mask = mask_utils.decode(rle).astype(np.uint8)
        crop = full_mask[y0 : y0 + side, x0 : x0 + side]
        return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_NEAREST)
    scale = output_size / float(side)
    output = np.zeros((output_size, output_size), dtype=np.uint8)
    polygons = []
    for polygon in segmentation:
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if len(points) < 3:
            continue
        points -= np.array([x0, y0], dtype=np.float32)
        points *= scale
        polygons.append(np.rint(points).astype(np.int32))
    if polygons:
        cv2.fillPoly(output, polygons, color=1, lineType=cv2.LINE_8)
    return output


class SeedRefinerDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        dataset: CompetitionDataset,
        samples: tuple[SeedCropSample, ...],
        config: SeedRefinerConfig,
        *,
        augment: bool,
        image_cache_size: int = 8,
    ) -> None:
        self.dataset = dataset
        self.samples = samples
        self.config = config
        self.augment = augment
        # Persistent DataLoader workers hold their own dataset copies. A shared
        # tensor makes epoch-based deterministic augmentation visible to them.
        self._epoch = torch.zeros(1, dtype=torch.int64).share_memory_()
        self.image_cache_size = image_cache_size
        self._image_cache: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.samples)

    def set_epoch(self, epoch: int) -> None:
        self._epoch.fill_(int(epoch))

    def sample_file_name(self, index: int) -> str:
        return self.samples[index].file_name

    def _image_channels(self, file_name: str) -> tuple[np.ndarray, np.ndarray]:
        cached = self._image_cache.get(file_name)
        if cached is not None:
            self._image_cache.move_to_end(file_name)
            return cached
        image = read_grayscale(self.dataset.train_image_dir / file_name)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
        cached = (image, clahe)
        self._image_cache[file_name] = cached
        while len(self._image_cache) > self.image_cache_size:
            self._image_cache.popitem(last=False)
        return cached

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image, clahe = self._image_channels(sample.file_name)
        bounds = _crop_bounds(sample.bbox, image.shape, self.config.crop_context)
        x0, y0, side = bounds
        image_crop = cv2.resize(
            image[y0 : y0 + side, x0 : x0 + side],
            (self.config.crop_size, self.config.crop_size),
            interpolation=cv2.INTER_AREA if side > self.config.crop_size else cv2.INTER_LINEAR,
        )
        clahe_crop = cv2.resize(
            clahe[y0 : y0 + side, x0 : x0 + side],
            (self.config.crop_size, self.config.crop_size),
            interpolation=cv2.INTER_AREA if side > self.config.crop_size else cv2.INTER_LINEAR,
        )
        rng = random.Random(self.config.seed + int(self._epoch.item()) * 1_000_003 + index * 97)
        jitter_scale = rng.uniform(0.9, 1.1) if self.augment else 1.0
        jitter_x = rng.uniform(-self.config.box_jitter_px, self.config.box_jitter_px) if self.augment else 0.0
        jitter_y = rng.uniform(-self.config.box_jitter_px, self.config.box_jitter_px) if self.augment else 0.0
        sx0, sy0, sx1, sy1 = _seed_box(
            sample.bbox,
            image.shape,
            self.config.seed_padding,
            jitter_x=jitter_x,
            jitter_y=jitter_y,
            scale=jitter_scale,
        )
        seed_crop = np.zeros((self.config.crop_size, self.config.crop_size), dtype=np.uint8)
        sx0 = (sx0 - x0) * self.config.crop_size / side
        sy0 = (sy0 - y0) * self.config.crop_size / side
        sx1 = (sx1 - x0) * self.config.crop_size / side
        sy1 = (sy1 - y0) * self.config.crop_size / side
        cv2.rectangle(
            seed_crop,
            (int(round(sx0)), int(round(sy0))),
            (int(round(sx1)), int(round(sy1))),
            color=1,
            thickness=-1,
        )
        target = _fill_segmentation_crop(
            sample.target_segmentation, bounds, self.config.crop_size
        )

        if self.augment:
            turns = rng.randrange(4)
            image_crop = np.rot90(image_crop, turns).copy()
            clahe_crop = np.rot90(clahe_crop, turns).copy()
            seed_crop = np.rot90(seed_crop, turns).copy()
            target = np.rot90(target, turns).copy()
            if rng.random() < 0.5:
                image_crop = np.fliplr(image_crop).copy()
                clahe_crop = np.fliplr(clahe_crop).copy()
                seed_crop = np.fliplr(seed_crop).copy()
                target = np.fliplr(target).copy()
            if rng.random() < 0.5:
                image_crop = np.flipud(image_crop).copy()
                clahe_crop = np.flipud(clahe_crop).copy()
                seed_crop = np.flipud(seed_crop).copy()
                target = np.flipud(target).copy()
            gamma = rng.uniform(0.85, 1.2)
            gain = rng.uniform(0.9, 1.1)
            bias = rng.uniform(-10.0, 10.0)
            image_crop = np.clip(
                np.power(image_crop.astype(np.float32) / 255.0, gamma) * gain * 255.0 + bias,
                0,
                255,
            ).astype(np.uint8)
            clahe_crop = np.clip(
                np.power(clahe_crop.astype(np.float32) / 255.0, gamma) * gain * 255.0 + bias,
                0,
                255,
            ).astype(np.uint8)

        channels = np.stack((image_crop, clahe_crop, seed_crop * 255), axis=0).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        inputs = np.clip((channels - mean) / std, -5.0, 5.0)
        return torch.from_numpy(inputs), torch.from_numpy(target[None].astype(np.float32))


class ObservationBatchSampler(Sampler[list[int]]):
    """Keep each minibatch on one source image to reuse its decoded pixels."""

    def __init__(self, dataset: SeedRefinerDataset, batch_size: int, seed: int) -> None:
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        grouped: dict[str, list[int]] = defaultdict(list)
        for index in range(len(dataset)):
            grouped[dataset.sample_file_name(index)].append(index)
        self.groups = dict(grouped)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch * 9973)
        file_names = list(self.groups)
        rng.shuffle(file_names)
        batches = []
        for file_name in file_names:
            indices = list(self.groups[file_name])
            rng.shuffle(indices)
            batches.extend(
                indices[start : start + self.batch_size]
                for start in range(0, len(indices), self.batch_size)
            )
        rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return sum(math.ceil(len(indices) / self.batch_size) for indices in self.groups.values())


def build_seed_refiner() -> nn.Module:
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise RuntimeError("Install requirements-segmentation.txt for the seed refiner") from exc
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )


def _seed_refiner_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    return 0.5 * bce + 0.5 * dice.mean()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train_seed_refiner(
    dataset: CompetitionDataset,
    samples: tuple[SeedCropSample, ...],
    config: SeedRefinerConfig,
    output_path: str | Path,
    *,
    device: torch.device,
    num_workers: int = 4,
) -> dict[str, Any]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)

    training_dataset = SeedRefinerDataset(dataset, samples, config, augment=True)
    batch_sampler = ObservationBatchSampler(training_dataset, config.batch_size, config.seed)
    loader = DataLoader(
        training_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    model = build_seed_refiner().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history_path = output_path.with_suffix(".jsonl")
    started = time.perf_counter()
    step = 0
    epoch = 0
    history = []

    while step < config.training_steps:
        epoch += 1
        training_dataset.set_epoch(epoch)
        batch_sampler.set_epoch(epoch)
        model.train()
        loss_window = []
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=use_amp,
            ):
                logits = model(inputs)
                loss = _seed_refiner_loss(logits.float(), targets.float())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            loss_window.append(float(loss.detach().cpu()))
            if step % 250 == 0 or step == config.training_steps:
                row = {
                    "step": step,
                    "epoch": epoch,
                    "elapsed_seconds": time.perf_counter() - started,
                    "mean_loss_since_log": float(np.mean(loss_window)),
                }
                history.append(row)
                with history_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                print(json.dumps(row, sort_keys=True), flush=True)
                loss_window.clear()
            if step >= config.training_steps:
                break

    peak_gpu_memory = int(torch.cuda.max_memory_allocated(device)) if use_amp else 0
    checkpoint = {
        "model": model.cpu().state_dict(),
        "config": asdict(config),
        "training_steps": step,
        "training_samples": len(samples),
        "epochs_started": epoch,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": peak_gpu_memory,
        "pretrained_backbone": None,
    }
    torch.save(checkpoint, output_path)
    return {
        "checkpoint": str(output_path),
        "checkpoint_sha256": _sha256(output_path),
        "training_samples": len(samples),
        "training_steps": step,
        "epochs_started": epoch,
        "training_seconds": checkpoint["elapsed_seconds"],
        "peak_gpu_memory_bytes": peak_gpu_memory,
        "history": history,
    }


def load_seed_refiner(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    model = build_seed_refiner()
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def _normalized_crop(
    image_crop: np.ndarray,
    clahe_crop: np.ndarray,
    seed_crop: np.ndarray,
) -> np.ndarray:
    channels = np.stack((image_crop, clahe_crop, seed_crop * 255), axis=0).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    return np.clip((channels - mean) / std, -5.0, 5.0)


def _component_count_8(mask: np.ndarray) -> int:
    return int(cv2.connectedComponents(np.asarray(mask, dtype=np.uint8), connectivity=8)[0] - 1)


def _select_topology_safe_tta(
    single_probability: np.ndarray,
    tta_probability: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, bool, str]:
    """Keep flip-4 TTA only when its thresholded mask stays close in topology."""
    single_mask = np.asarray(single_probability, dtype=np.float32) >= threshold
    tta_mask = np.asarray(tta_probability, dtype=np.float32) >= threshold
    single_area = int(single_mask.sum())
    tta_area = int(tta_mask.sum())
    if tta_area == 0:
        return single_probability, False, "empty_tta_mask"
    if single_area == 0:
        return single_probability, False, "empty_single_view_mask"
    area_ratio = tta_area / single_area
    if area_ratio < 0.60:
        return single_probability, False, "area_shrink_guard"
    if area_ratio > 1.70:
        return single_probability, False, "area_growth_guard"
    single_components = _component_count_8(single_mask)
    tta_components = _component_count_8(tta_mask)
    if tta_components > max(single_components + 1, 2):
        return single_probability, False, "fragmentation_guard"
    if single_area >= 32:
        denominator = single_area + tta_area
        dice = 2.0 * int(np.logical_and(single_mask, tta_mask).sum()) / denominator
        if dice < 0.62:
            return single_probability, False, "overlap_guard"
    return tta_probability, True, "accepted"


def predict_seed_refiner(
    model: nn.Module,
    image: np.ndarray,
    detections: list[dict[str, Any]],
    config: SeedRefinerConfig,
    *,
    device: torch.device,
    score_threshold: float | None = None,
    probability_threshold: float | None = None,
    box_jitter: bool = False,
    flip_tta: bool = False,
    topology_safe_tta: bool = True,
    tta_diagnostics: dict[str, int] | None = None,
) -> list[tuple[float, np.ndarray]]:
    if image.shape != (IMAGE_SIZE, IMAGE_SIZE) or image.dtype != np.uint8:
        raise ValueError("Seed-refiner inference expects a 2048x2048 uint8 grayscale image")
    score_threshold = (
        config.validation_score_min if score_threshold is None else score_threshold
    )
    probability_threshold = (
        config.probability_threshold
        if probability_threshold is None
        else probability_threshold
    )
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
    prepared = []
    for detection in sorted(detections, key=lambda value: -float(value["score"])):
        score = float(detection["score"])
        if score < score_threshold:
            continue
        rle = _rle(detection["counts"])
        bbox = tuple(float(value) for value in mask_utils.toBbox(rle))
        if bbox[2] <= 0 or bbox[3] <= 0:
            continue
        bounds = _crop_bounds(bbox, image.shape, config.crop_context)
        x0, y0, side = bounds
        image_crop = cv2.resize(
            image[y0 : y0 + side, x0 : x0 + side],
            (config.crop_size, config.crop_size),
            interpolation=cv2.INTER_AREA if side > config.crop_size else cv2.INTER_LINEAR,
        )
        clahe_crop = cv2.resize(
            clahe[y0 : y0 + side, x0 : x0 + side],
            (config.crop_size, config.crop_size),
            interpolation=cv2.INTER_AREA if side > config.crop_size else cv2.INTER_LINEAR,
        )
        shifts = [(0.0, 0.0)]
        if box_jitter and config.box_jitter_px > 0:
            amount = float(config.box_jitter_px)
            shifts.extend((dx, dy) for dx, dy in ((amount, 0), (-amount, 0), (0, amount), (0, -amount)))
        for dx, dy in shifts:
            seed_box = _seed_box(
                bbox,
                image.shape,
                config.seed_padding,
                jitter_x=dx,
                jitter_y=dy,
            )
            seed_crop = np.zeros((config.crop_size, config.crop_size), dtype=np.uint8)
            sx0, sy0, sx1, sy1 = seed_box
            sx0 = (sx0 - x0) * config.crop_size / side
            sy0 = (sy0 - y0) * config.crop_size / side
            sx1 = (sx1 - x0) * config.crop_size / side
            sy1 = (sy1 - y0) * config.crop_size / side
            cv2.rectangle(
                seed_crop,
                (int(round(sx0)), int(round(sy0))),
                (int(round(sx1)), int(round(sy1))),
                color=1,
                thickness=-1,
            )
            prepared.append((score, bounds, image_crop, clahe_crop, seed_crop))

    if not prepared:
        return []
    model.eval()
    single_probability_rows = []
    tta_probability_rows = []
    with torch.inference_mode():
        for start in range(0, len(prepared), 8):
            batch = torch.from_numpy(
                np.stack(
                    [
                        _normalized_crop(image_crop, clahe_crop, seed_crop)
                        for _, _, image_crop, clahe_crop, seed_crop in prepared[start : start + 8]
                    ]
                )
            ).to(device)
            def infer_probability(inputs: torch.Tensor) -> torch.Tensor:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    return torch.sigmoid(model(inputs).float())[:, 0]

            single_probabilities = infer_probability(batch)
            if flip_tta:
                tta_sum = single_probabilities.clone()
                for dimensions in ((-1,), (-2,), (-2, -1)):
                    transformed_probabilities = infer_probability(torch.flip(batch, dimensions))
                    tta_sum += torch.flip(transformed_probabilities, dimensions)
                tta_probabilities = tta_sum / 4.0
            else:
                tta_probabilities = single_probabilities
            single_probability_rows.extend(single_probabilities.cpu().numpy())
            tta_probability_rows.extend(tta_probabilities.cpu().numpy())
    single_probabilities = np.stack(single_probability_rows)
    tta_probabilities = np.stack(tta_probability_rows)
    single_by_detection: dict[int, list[np.ndarray]] = defaultdict(list)
    tta_by_detection: dict[int, list[np.ndarray]] = defaultdict(list)
    for index, (score, bounds, *_channels) in enumerate(prepared):
        detection_index = index // (5 if box_jitter and config.box_jitter_px > 0 else 1)
        single_by_detection[detection_index].append(single_probabilities[index])
        tta_by_detection[detection_index].append(tta_probabilities[index])

    output = []
    detection_count = len(single_by_detection)
    variants_per_detection = 5 if box_jitter and config.box_jitter_px > 0 else 1
    for detection_index in range(detection_count):
        single_probability = np.mean(single_by_detection[detection_index], axis=0)
        tta_probability = np.mean(tta_by_detection[detection_index], axis=0)
        if flip_tta and topology_safe_tta:
            probability, accepted, reason = _select_topology_safe_tta(
                single_probability,
                tta_probability,
                probability_threshold,
            )
            if tta_diagnostics is not None:
                tta_diagnostics[reason] = tta_diagnostics.get(reason, 0) + 1
        else:
            probability = tta_probability
            if flip_tta and tta_diagnostics is not None:
                tta_diagnostics["accepted"] = tta_diagnostics.get("accepted", 0) + 1
        binary_crop = (probability >= probability_threshold).astype(np.uint8)
        if int(binary_crop.sum()) < config.minimum_instance_area:
            continue
        score, bounds = prepared[detection_index * variants_per_detection][:2]
        x0, y0, side = bounds
        native_crop = cv2.resize(
            binary_crop,
            (side, side),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        full_mask = np.zeros(image.shape, dtype=bool)
        full_mask[y0 : y0 + side, x0 : x0 + side] = native_crop
        if int(full_mask.sum()) >= config.minimum_instance_area:
            output.append((score, full_mask))

    claimed = np.zeros(image.shape, dtype=bool)
    non_overlapping = []
    for score, mask in output:
        mask = mask & ~claimed
        if int(mask.sum()) < config.minimum_instance_area:
            continue
        claimed |= mask
        non_overlapping.append((score, mask))
    return non_overlapping


def predicted_instances(
    image_name: str,
    image: np.ndarray,
    detections: list[dict[str, Any]],
    model: nn.Module,
    config: SeedRefinerConfig,
    *,
    device: torch.device,
    box_jitter: bool = False,
    flip_tta: bool = False,
    topology_safe_tta: bool = True,
    tta_diagnostics: dict[str, int] | None = None,
) -> list[PredictedInstance]:
    masks = predict_seed_refiner(
        model,
        image,
        detections,
        config,
        device=device,
        box_jitter=box_jitter,
        flip_tta=flip_tta,
        topology_safe_tta=topology_safe_tta,
        tta_diagnostics=tta_diagnostics,
    )
    predictions = []
    for score, mask in masks:
        area = int(mask.sum())
        predictions.append(
            PredictedInstance(
                filament_id=f"{Path(image_name).stem}_{len(predictions) + 1}",
                image_stem=Path(image_name).stem,
                area=area,
                rle=encode_binary_mask(mask),
            )
        )
    return predictions
