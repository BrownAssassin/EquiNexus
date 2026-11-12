# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pycocotools import mask as mask_utils
from sklearn.model_selection import train_test_split
from torchvision.models import ResNet18_Weights, resnet18
from ultralytics import YOLO

from filament_baseline.data import load_dataset
from filament_baseline.predictor import PredictedInstance
from filament_baseline.runtime import hardware_info, package_versions
from filament_baseline.scorer import score_validation


SOURCE_NOTEBOOK = (
    "https://www.kaggle.com/code/ektarr/yolo-u-net-solar-filament-segmentation"
)
SOURCE_OUTPUTS = (
    REPO_ROOT / "data" / "reference" / "public-notebooks" / "ektarr-yolo-unet" / "outputs"
)
SOURCE_CONFIG = {
    "image_size": 1024,
    "batch_size": 4,
    "epochs": 30,
    "patience": 5,
    "confidence_threshold": 0.25,
    "minimum_component_size": 20,
    "mask_ratio": 2,
    "categories": True,
    "refiner_crop_size": 256,
    "refiner_threshold": 0.65,
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def source_split(annotation_path: Path) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Reproduce the public notebook's exact file order, view choice, and sklearn split."""
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = {item["id"]: item for item in payload["images"]}
    annotations: dict[Any, list[dict[str, Any]]] = {}
    for item in payload["annotations"]:
        annotations.setdefault(item["image_id"], []).append(item)
    file_to_ids: dict[str, list[Any]] = {}
    for image_id, info in images.items():
        file_to_ids.setdefault(info["file_name"], []).append(image_id)

    selected_ids: dict[str, Any] = {}
    for filename, image_ids in file_to_ids.items():
        selected_ids[filename] = max(
            image_ids,
            key=lambda image_id: (
                len(annotations.get(image_id, [])),
                sum(item.get("area", 0) for item in annotations.get(image_id, [])),
            ),
        )

    filenames = list(selected_ids)
    train_names, validation_names = train_test_split(
        filenames, test_size=0.2, random_state=42
    )
    train_set, validation_set = set(train_names), set(validation_names)
    if train_set & validation_set or len(train_set | validation_set) != len(filenames):
        raise RuntimeError("Reconstructed source split is not a disjoint partition")
    manifest = []
    for filename in sorted(filenames):
        chosen_id = selected_ids[filename]
        info = images[chosen_id]
        manifest.append({
            "file_name": filename,
            "partition": "validation" if filename in validation_set else "train",
            "selected_image_id": str(chosen_id),
            "view_count": len(file_to_ids[filename]),
            "selected_annotation_count": len(annotations.get(chosen_id, [])),
            "year": int(Path(filename).stem[:4]),
            "site": Path(filename).stem[-2:],
        })
    return manifest, list(train_names), list(validation_names)


def square_bounds(mask: np.ndarray, context: float = 1.8, minimum: int = 96) -> tuple[int, int, int, int]:
    height, width = mask.shape
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0, width, height
    center_x = (xs.min() + xs.max() + 1) / 2
    center_y = (ys.min() + ys.max() + 1) / 2
    side = int(np.ceil(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * context))
    side = min(max(side, minimum), min(height, width))
    x0, y0 = int(round(center_x - side / 2)), int(round(center_y - side / 2))
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


class CropUNet(nn.Module):
    """Inference-compatible copy of the Apache-2.0 public notebook refiner."""

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        encoder = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4
        self.up4, self.dec4 = nn.ConvTranspose2d(512, 256, 2, 2), self._block(512, 256)
        self.up3, self.dec3 = nn.ConvTranspose2d(256, 128, 2, 2), self._block(256, 128)
        self.up2, self.dec2 = nn.ConvTranspose2d(128, 64, 2, 2), self._block(128, 64)
        self.up1, self.dec1 = nn.ConvTranspose2d(64, 32, 2, 2), self._block(96, 32)
        self.up0 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.head = nn.Conv2d(16, 1, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = (image - self.mean) / self.std
        x0 = self.stem(image)
        x1 = self.layer1(self.pool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.dec4(torch.cat([self.up4(x4), x3], 1))
        x = self.dec3(torch.cat([self.up3(x), x2], 1))
        x = self.dec2(torch.cat([self.up2(x), x1], 1))
        x = self.dec1(torch.cat([self.up1(x), x0], 1))
        return self.head(self.up0(x))


@torch.inference_mode()
def refine_masks(
    image_gray: np.ndarray,
    masks: np.ndarray,
    refiner: CropUNet,
    device: torch.device,
) -> list[np.ndarray]:
    output = []
    for mask in masks:
        if not mask.any():
            output.append(mask)
            continue
        x0, y0, x1, y1 = square_bounds(mask)
        crop = cv2.resize(image_gray[y0:y1, x0:x1], (256, 256), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(crop).float().view(1, 1, 256, 256).repeat(1, 3, 1, 1).to(device) / 255
        probability = refiner(image).sigmoid()[0, 0].cpu().numpy()
        crop_mask = cv2.resize(
            probability, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR
        ) >= SOURCE_CONFIG["refiner_threshold"]
        if crop_mask.any():
            refined = np.zeros_like(mask, dtype=np.uint8)
            refined[y0:y1, x0:x1] = crop_mask.astype(np.uint8)
            output.append(refined)
        else:
            output.append(mask)
    return output


def encode_mask(mask: np.ndarray) -> dict[str, Any]:
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": [2048, 2048], "counts": encoded["counts"].decode("ascii")}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score the public Ektarr YOLO+crop-refiner notebook on its reconstructed source split."
    )
    parser.add_argument(
        "--data-root", type=Path, default=REPO_ROOT / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026"
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "external-data" / "ektarr-yolo-unet-source-validation-20261111")
    parser.add_argument("--detector", type=Path, default=SOURCE_OUTPUTS / "best_model.pt")
    parser.add_argument("--refiner", type=Path, default=SOURCE_OUTPUTS / "best_refiner.pt")
    parser.add_argument("--notebook", type=Path, default=SOURCE_OUTPUTS.parent / "yolo-u-net-solar-filament-segmentation.ipynb")
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not output_dir.is_relative_to((REPO_ROOT / "artifacts").resolve()):
        raise ValueError("All evaluation outputs must be kept under repository artifacts/")
    for path in (args.detector, args.refiner, args.notebook):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("This 1024-pixel full-resolution checkpoint evaluation requires CUDA")

    dataset = load_dataset(data_root)
    manifest, source_train, source_validation = source_split(dataset.annotation_path)
    source_validation = sorted(source_validation)
    validation_set = set(source_validation)
    frozen_rows = pd.read_csv(REPO_ROOT / "data" / "splits" / "observation_split_seed42.csv")
    frozen_validation = set(frozen_rows.loc[frozen_rows["partition"] == "validation", "file_name"])
    manifest_frame = pd.DataFrame(manifest)
    manifest_bytes = manifest_frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    split_hash = hashlib.sha256(manifest_bytes).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "data" / "splits" / "ektarr-yolo-u-net-random-seed42.csv").write_bytes(manifest_bytes)

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    detector = YOLO(str(args.detector))
    refiner = CropUNet()
    refiner_state = torch.load(args.refiner, map_location="cpu", weights_only=True)
    refiner.load_state_dict(refiner_state, strict=True)
    refiner.to(device).eval()

    predictions: dict[str, list[dict[str, Any]]] = {Path(name).stem: [] for name in source_validation}
    started = time.perf_counter()
    for index, filename in enumerate(source_validation, start=1):
        image_path = dataset.train_image_dir / filename
        image_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image_gray is None or image_gray.shape != (2048, 2048):
            raise ValueError(f"Expected a readable 2048x2048 image: {image_path}")
        result = detector.predict(
            str(image_path),
            imgsz=SOURCE_CONFIG["image_size"],
            conf=SOURCE_CONFIG["confidence_threshold"],
            agnostic_nms=True,
            retina_masks=True,
            verbose=False,
            device=0,
        )[0]
        if result.masks is None:
            continue
        raw_masks = result.masks.data.detach().cpu().numpy().astype(np.uint8)
        scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
        if raw_masks.shape[0] != len(scores) or raw_masks.shape[1:] != (2048, 2048):
            raise RuntimeError(f"Unexpected detector mask/score shapes for {filename}: {raw_masks.shape}, {scores.shape}")
        refined_masks = refine_masks(image_gray, raw_masks, refiner, device)
        for mask, score in zip(refined_masks, scores, strict=True):
            if int(mask.sum()) < SOURCE_CONFIG["minimum_component_size"]:
                continue
            predictions[Path(filename).stem].append({
                "score": float(score),
                **encode_mask(mask),
            })
        if index % 10 == 0 or index == len(source_validation):
            print(json.dumps({
                "event": "progress",
                "completed": index,
                "total": len(source_validation),
                "elapsed_seconds": round(time.perf_counter() - started, 2),
            }), flush=True)

    elapsed = time.perf_counter() - started
    predictions_path = output_dir / "validation_predictions.json"
    predictions_path.write_text(json.dumps(predictions, separators=(",", ":")), encoding="utf-8")

    predictions_by_name: dict[str, list[PredictedInstance]] = {}
    for filename in source_validation:
        stem = Path(filename).stem
        instances = []
        for index, item in enumerate(predictions[stem], start=1):
            rle = {"size": item["size"], "counts": item["counts"].encode("ascii")}
            area = int(mask_utils.area(rle))
            if float(item["score"]) < SOURCE_CONFIG["confidence_threshold"] or area < SOURCE_CONFIG["minimum_component_size"]:
                continue
            instances.append(PredictedInstance(
                filament_id=f"{stem}_{index}",
                image_stem=stem,
                area=area,
                rle=rle,
            ))
        predictions_by_name[filename] = instances

    metrics, observation_rows = score_validation(
        dataset,
        predictions_by_name,
        source_validation,
        iou_threshold=0.5,
        diagnostic_overlap_threshold=0.0,
    )
    frozen_overlap = len(validation_set & frozen_validation)
    report = {
        "run_name": "ektarr-public-yolo-unet-source-validation-20261111",
        "source": SOURCE_NOTEBOOK,
        "source_license": "Apache-2.0 notebook; Ultralytics detector/code and COCO checkpoint are AGPL-3.0; torchvision ResNet18 ImageNet initialization is recorded by the source notebook.",
        "source_kaggle_kernel_version": 1,
        "source_notebook_sha256": digest(args.notebook),
        "source_run_status": "Kaggle kernel complete; source log reports approximately 2015 seconds end-to-end.",
        "source_output_hashes": {
            "detector_sha256": digest(args.detector),
            "refiner_sha256": digest(args.refiner),
            "public_test_submission_sha256": digest(SOURCE_OUTPUTS / "submission.csv"),
            "kernel_log_sha256": digest(SOURCE_OUTPUTS / "yolo-u-net-solar-filament-segmentation.log"),
        },
        "source_configuration": SOURCE_CONFIG,
        "source_split": {
            "method": "Exact notebook behavior: first-seen COCO filename order, choose one view by max(annotation_count, annotation_area), sklearn train_test_split(test_size=0.2, random_state=42).",
            "sha256": split_hash,
            "training_observations": len(source_train),
            "validation_observations": len(source_validation),
            "observation_overlap": len(set(source_train) & validation_set),
            "frozen_challenge_validation_overlap": frozen_overlap,
            "frozen_challenge_validation_count": len(frozen_validation),
            "all_annotator_views_scored": True,
            "manifest": "data/splits/ektarr-yolo-u-net-random-seed42.csv",
        },
        "holdout_selection_disclosure": "The source notebook uses its validation partition for YOLO checkpoint selection and crop-refiner Dice checkpoint selection. Results are screening evidence on the source split, not an untouched final estimate.",
        "scorer": {
            "name": "EquiNexus port of the organizer scorer",
            "iou_threshold": 0.5,
            "diagnostic_overlap_threshold": 0.0,
            "prediction_confidence": 0.25,
            "minimum_mask_area": 20,
            "prediction_reused_for_each_annotator_view": True,
        },
        "official_pq": metrics,
        "per_observation": [
            {
                "file_name": row.file_name,
                "year": row.year,
                "site": row.site,
                "annotator_views": row.annotator_views,
                "gt_count": row.gt_count,
                "prediction_count": row.prediction_count,
                "tp": row.tp,
                "tp_iou_sum": row.tp_iou_sum,
                "fp": row.fp,
                "fn": row.fn,
                "pq": row.pq,
            }
            for row in observation_rows
        ],
        "validation_predictions": {
            "path": str(predictions_path),
            "sha256": digest(predictions_path),
            "count": sum(len(items) for items in predictions.values()),
        },
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "device": torch.cuda.get_device_name(device),
        "hardware": hardware_info(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "package_versions": package_versions(),
        "runner_sha256": digest(Path(__file__).resolve()),
        "ground_truth_source": "competition-provided training annotations only",
        "public_MAGFiLO_test_labels_read": False,
    }
    report_path = output_dir / "official-pq-source-validation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "event": "complete",
        "pq": metrics["global"]["pq"],
        "tp": metrics["global"]["tp"],
        "fp": metrics["global"]["fp"],
        "fn": metrics["global"]["fn"],
        "validation_observations": len(source_validation),
        "frozen_validation_overlap": frozen_overlap,
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": report["peak_gpu_memory_bytes"],
        "report": str(report_path),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
