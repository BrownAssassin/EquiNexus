# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import cv2
import pandas as pd
import torch
from pycocotools import mask as mask_utils

from filament_baseline.data import find_dataset_root
from filament_baseline.external_panoptic import (
    EXPECTED_MODEL_MD5,
    encode_instance,
    load_public_checkpoints,
    predict_image,
)
from filament_baseline.runtime import hardware_info, package_versions


MODEL_SOURCE = "phuongncn/solar-filament-i079-weights/PyTorch/detector-refiner/1"


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _locate_checkpoint(search_root: Path, name: str, md5: str) -> Path:
    matches = [
        path for path in search_root.rglob(name)
        if path.is_file() and _digest(path, "md5") == md5
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one md5-verified {name} below {search_root}, found {len(matches)}")
    return matches[0]


def _validate_submission(frame: pd.DataFrame, test_stems: set[str]) -> dict[str, int]:
    if list(frame.columns) != ["filament_id", "segmentation_rle"]:
        raise ValueError("submission columns differ from the organizer format")
    if frame.empty or frame.isna().any().any() or not frame.filament_id.is_unique:
        raise ValueError("submission must contain non-empty, unique, complete rows")
    covered: set[str] = set()
    for row in frame.itertuples(index=False):
        image_stem = str(row.filament_id).rsplit("_", maxsplit=1)[0]
        if image_stem not in test_stems:
            raise ValueError(f"submission references unknown image {image_stem}")
        covered.add(image_stem)
        encoded = str(row.segmentation_rle).encode("ascii")
        decoded = mask_utils.decode({"size": [2048, 2048], "counts": encoded})
        if decoded.shape != (2048, 2048) or not decoded.any():
            raise ValueError(f"invalid or empty RLE for {row.filament_id}")
    return {
        "rows": int(len(frame)),
        "processed_images": int(len(test_stems)),
        "images_with_predictions": int(len(covered)),
        "images_without_predictions": int(len(test_stems - covered)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the public Apache-2.0 detector/refiner checkpoints on competition test images only."
    )
    parser.add_argument("--data-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--model-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/equinexus/public-i079"))
    parser.add_argument("--expected-test-images", type=int, default=180)
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    model_root = args.model_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    kaggle_runtime = Path("/kaggle/input").is_dir()
    allowed_output_root = Path("/kaggle/working").resolve() if kaggle_runtime else (REPO_ROOT / "artifacts").resolve()
    if not output_dir.is_relative_to(allowed_output_root):
        raise ValueError(f"outputs must remain under {allowed_output_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    competition_root = find_dataset_root(data_root)
    test_dir = competition_root / "test" / "test_images"
    test_files = sorted(path for path in test_dir.iterdir() if path.is_file())
    if len(test_files) != args.expected_test_images:
        raise ValueError(f"expected {args.expected_test_images} test images, found {len(test_files)}")
    detector_path = _locate_checkpoint(
        model_root, "final_detector.pt", EXPECTED_MODEL_MD5["final_detector.pt"]
    )
    refiner_path = _locate_checkpoint(
        model_root, "final_refiner.pt", EXPECTED_MODEL_MD5["final_refiner.pt"]
    )

    if not torch.cuda.is_available():
        raise RuntimeError("this public checkpoint pair requires a CUDA device")
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda:0")
    detector, refiner = load_public_checkpoints(detector_path, refiner_path, device)

    rows: list[dict[str, str]] = []
    diagnostic_rows = []
    started = time.perf_counter()
    for index, image_path in enumerate(test_files, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (2048, 2048):
            raise ValueError(f"test image failed full-resolution decode: {image_path.name}")
        masks, diagnostics = predict_image(detector, refiner, image, device)
        image_stem = image_path.stem
        for instance_index, mask in enumerate(masks, start=1):
            rows.append({
                "filament_id": f"{image_stem}_{instance_index}",
                "segmentation_rle": encode_instance(mask),
            })
        diagnostic_rows.append({"image": image_path.name, **diagnostics, "predictions": len(masks)})
        if index % 10 == 0 or index == len(test_files):
            print(json.dumps({
                "event": "inference_progress",
                "completed": index,
                "total": len(test_files),
                "elapsed_seconds": round(time.perf_counter() - started, 2),
            }), flush=True)

    frame = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    coverage = _validate_submission(frame, {path.stem for path in test_files})
    submission_path = output_dir / "submission.csv"
    frame.to_csv(submission_path, index=False, quoting=csv.QUOTE_MINIMAL)
    elapsed = time.perf_counter() - started
    report = {
        "run_name": "public-i079-apache2-test-inference",
        "candidate_id": "public-i079-detector-refiner-flip4-tta",
        "source_model": MODEL_SOURCE,
        "source_model_license": "Apache-2.0",
        "source_model_card": "Training used competition training images and annotations only; model publisher reports no public MAGFiLO/test-overlap labels.",
        "source_method_reference": "https://www.kaggle.com/code/robinamirbahar/solar-filament-panoptic-morphology-segmentation",
        "competition_root": str(competition_root),
        "training_annotations_read": False,
        "test_labels_read": False,
        "test_images_processed": len(test_files),
        "detector_checkpoint": {
            "name": detector_path.name,
            "md5": _digest(detector_path, "md5"),
            "sha256": _digest(detector_path, "sha256"),
            "bytes": detector_path.stat().st_size,
        },
        "refiner_checkpoint": {
            "name": refiner_path.name,
            "md5": _digest(refiner_path, "md5"),
            "sha256": _digest(refiner_path, "sha256"),
            "bytes": refiner_path.stat().st_size,
        },
        "thresholds": {
            "detector_score_min": 0.70,
            "detector_mask_min": 0.50,
            "refiner_threshold": 0.70,
            "tta_transforms": ["identity", "horizontal", "vertical", "both"],
        },
        "submission": {
            **coverage,
            "path": str(submission_path),
            "sha256": _digest(submission_path, "sha256"),
            "bytes": submission_path.stat().st_size,
        },
        "diagnostics": diagnostic_rows,
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "device": torch.cuda.get_device_name(0),
        "hardware": hardware_info(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "package_versions": package_versions(),
    }
    report_path = output_dir / "run_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "diagnostics"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
