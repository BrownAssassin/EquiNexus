# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Run HDJoJo's public YOLO26x-seg checkpoint on supplied competition test images.

The checkpoint is Apache-2.0. Ultralytics runtime use is AGPL-3.0. The public
model card reports training on the 707 competition training images, but does
not publish an observation-level split; this runner is test-only and never
loads training annotations or evaluates the frozen holdout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
WORK_CACHE = REPO_ROOT / ".cache"
os.environ["YOLO_CONFIG_DIR"] = str(WORK_CACHE / "ultralytics")
os.environ["TEMP"] = str(WORK_CACHE / "temp")
os.environ["TMP"] = str(WORK_CACHE / "temp")
os.environ["TMPDIR"] = str(WORK_CACHE / "temp")
(WORK_CACHE / "ultralytics").mkdir(parents=True, exist_ok=True)
(WORK_CACHE / "temp").mkdir(parents=True, exist_ok=True)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from pycocotools import mask as mask_utils  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "src"))
from filament_baseline.runtime import hardware_info, package_versions  # noqa: E402

MODEL_URL = "https://www.kaggle.com/models/hdjojo/yolo26x-seg-solar-filament"
SOURCE_NOTEBOOK = "https://www.kaggle.com/code/hdjojo/solar-filament-segmentation"
SPLIT_SHA256 = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"
IMAGE_SIZE = (2048, 2048)
MODEL_NAME = "yolo26x-seg-solar-filament-2048-1109.pt"
MODEL_SHA256 = "c5c02daec7b04b78481a22ade2780d55cf99755c0865a6e8e1dff22eda5d7390"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def require_within(path: Path, root: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"{description} must stay under {root}: {resolved}")
    return resolved


def encode_mask(binary_mask: np.ndarray) -> str:
    encoded = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    return encoded["counts"].decode("ascii")


def save_overlay(image: np.ndarray, masks: list[np.ndarray], destination: Path) -> None:
    base = np.repeat(image[:, :, None], 3, axis=2).astype(np.float32)
    colors = np.zeros((len(masks), 3), dtype=np.float32)
    for index in range(len(masks)):
        colors[index] = (
            (index * 71 + 47) % 256,
            (index * 131 + 89) % 256,
            (index * 197 + 151) % 256,
        )
    for index, mask in enumerate(masks):
        base[mask] = base[mask] * 0.58 + colors[index] * 0.42
    Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB").save(destination)


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=REPO_ROOT / "data" / "reference" / "kaggle-models" / "hdjojo-yolo26x" / MODEL_NAME,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "submissions" / "hdjojo-yolo26x-20261114",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, or CUDA device index such as 0")
    parser.add_argument("--max-images", type=int, help="limit inference for a local GPU smoke run")
    parser.add_argument("--half", action="store_true", help="use half precision when CUDA is selected")
    args = parser.parse_args()

    data_root = require_within(args.data_root, REPO_ROOT / "data", "Competition data")
    model_path = require_within(args.model_path, REPO_ROOT / "data" / "reference", "Public model checkpoint")
    output_root = require_within(args.output_dir, REPO_ROOT / "artifacts", "Run output")
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if sha256(model_path) != MODEL_SHA256:
        raise ValueError("YOLO26x checkpoint SHA-256 mismatch")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_root}")

    if args.device == "auto":
        device: str | int = 0 if torch.cuda.is_available() else "cpu"
    elif args.device.isdigit():
        device = int(args.device)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device was requested but CUDA is unavailable")
    else:
        device = args.device
    use_cuda = device != "cpu"
    if args.half and not use_cuda:
        raise ValueError("--half requires a CUDA device")

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    test_dir = data_root / "test" / "test_images"
    test_paths = sorted(test_dir.glob("*.jpeg"))
    if len(test_paths) != 180:
        raise ValueError(f"Expected 180 supplied competition test images, found {len(test_paths)}")
    if len({path.stem for path in test_paths}) != len(test_paths):
        raise ValueError("Test image filenames are not unique")
    if args.max_images is not None:
        if args.max_images < 1:
            raise ValueError("--max-images must be positive")
        test_paths = test_paths[: args.max_images]

    output_root.mkdir(parents=True, exist_ok=True)
    overlay_root = output_root / "overlays"
    overlay_root.mkdir()
    overlay_indices = set(
        np.linspace(0, len(test_paths) - 1, min(6, len(test_paths)), dtype=int).tolist()
    )
    rows: list[tuple[str, str]] = []
    image_results: list[dict[str, Any]] = []
    started = time.perf_counter()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    for image_index, path in enumerate(test_paths):
        with Image.open(path) as source_image:
            if source_image.size != IMAGE_SIZE:
                raise ValueError(f"Expected {IMAGE_SIZE} image at {path}, got {source_image.size}")
            gray = np.asarray(source_image.convert("L"), dtype=np.uint8)

        result = model.predict(
            str(path),
            imgsz=2048,
            conf=0.25,
            iou=0.0,
            max_det=100,
            verbose=False,
            device=device,
            half=args.half,
        )[0]
        surviving_masks: list[np.ndarray] = []
        surviving_scores: list[float] = []
        if result.masks is not None:
            raw_masks = result.masks.data.detach().to("cpu").numpy()
            scores = result.boxes.conf.detach().to("cpu").numpy()
            if len(raw_masks) != len(scores):
                raise ValueError(f"YOLO mask/score count mismatch for {path.name}")
            for prediction_index in np.argsort(scores)[::-1]:
                mask = raw_masks[int(prediction_index)] > 0.5
                if mask.shape != IMAGE_SIZE:
                    mask = cv2.resize(
                        mask.astype(np.uint8), IMAGE_SIZE, interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                clean = mask & ~np.logical_or.reduce(surviving_masks) if surviving_masks else mask
                if int(clean.sum()) < 5:
                    continue
                surviving_masks.append(clean)
                surviving_scores.append(float(scores[int(prediction_index)]))

        for instance_index, mask in enumerate(surviving_masks, start=1):
            rows.append((f"{path.stem}_{instance_index}", encode_mask(mask)))
        image_results.append(
            {
                "image": path.name,
                "detections_after_confidence_and_overlap_filter": len(surviving_masks),
                "mean_confidence": float(np.mean(surviving_scores)) if surviving_scores else None,
            }
        )
        if image_index in overlay_indices:
            save_overlay(gray, surviving_masks, overlay_root / f"{path.stem}.png")
        if (image_index + 1) % 20 == 0 or image_index + 1 == len(test_paths):
            print(f"Processed {image_index + 1}/{len(test_paths)} supplied test images", flush=True)

    if use_cuda:
        torch.cuda.synchronize()
    submission_path = output_root / "submission.csv"
    with submission_path.open("w", encoding="ascii", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("filament_id", "segmentation_rle"))
        writer.writerows(rows)
    elapsed = time.perf_counter() - started

    try:
        import ultralytics

        ultralytics_version = ultralytics.__version__
    except Exception:
        ultralytics_version = package_version("ultralytics")
    peak_gpu_memory_bytes = int(torch.cuda.max_memory_allocated()) if use_cuda else 0
    hardware = hardware_info()
    metadata = {
        "candidate_role": "test-only public model submission candidate; no holdout evaluation",
        "source_model": MODEL_URL,
        "source_notebook": SOURCE_NOTEBOOK,
        "model_author": "HDJoJo",
        "model_license": "Apache-2.0",
        "runtime_license": "AGPL-3.0 (Ultralytics)",
        "training_provenance": "Public model card states the 707 competition training images only; the exact observation split is not published. No public MAGFiLO test labels are reported or loaded by this runner.",
        "holdout_evaluation": "not performed because the published model may have trained on frozen holdout observations",
        "input_scope": "supplied competition test H-alpha images only",
        "competition_test_images": len(test_paths),
        "test_images_with_predictions": sum(int(item["detections_after_confidence_and_overlap_filter"] > 0) for item in image_results),
        "predicted_instances": len(rows),
        "image_results": image_results,
        "split_sha256": SPLIT_SHA256,
        "source_commit": git_value("rev-parse", "HEAD"),
        "working_tree": git_value("status", "--short"),
        "script_sha256": sha256(Path(__file__).resolve()),
        "checkpoint_sha256": sha256(model_path),
        "config": {
            "imgsz": 2048,
            "confidence_threshold": 0.25,
            "nms_iou_threshold": 0.0,
            "max_detections": 100,
            "overlap_resolution": "sort by model confidence, then subtract previously claimed mask pixels",
            "minimum_remaining_mask_area": 5,
            "half_precision": args.half,
        },
        "torch": torch.__version__,
        "ultralytics": ultralytics_version,
        "opencv": cv2.__version__,
        "device": str(device),
        "hardware": hardware,
        "package_versions": package_versions(),
        "runtime_seconds": round(elapsed, 3),
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "submission_path": str(submission_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "submission_sha256": sha256(submission_path),
        "overlay_paths": [str(path.relative_to(REPO_ROOT)).replace("\\", "/") for path in sorted(overlay_root.glob("*.png"))],
    }
    metadata_path = output_root / "run-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: metadata[key] for key in (
        "candidate_role", "competition_test_images", "test_images_with_predictions",
        "predicted_instances", "runtime_seconds", "peak_gpu_memory_bytes", "submission_sha256"
    )}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
