# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Run the all-training-data YOLO checkpoint on the supplied test images with D4 TTA."""

from __future__ import annotations

import argparse
import cv2
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from filament_baseline.data import inventory, load_dataset
from scripts.reinfer_yolo_oof_tta import _predict_image, _sha256
from filament_baseline.split import create_observation_split


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--detector-run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=1280)
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--nms-iou", type=float, default=0.6)
    parser.add_argument("--max-detections", type=int, default=150)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    repo_root = args.repo_root.expanduser().resolve()
    data_root = (args.data_root or repo_root / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026").expanduser().resolve()
    detector_root = args.detector_run_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for label, path, allowed_root in (
        ("competition data", data_root, repo_root / "data"),
        ("full-fit detector", detector_root, repo_root / "artifacts" / "detector-experiments"),
        ("test predictions", output_dir, repo_root / "artifacts" / "detector-experiments"),
    ):
        if not path.is_relative_to(allowed_root.resolve()):
            raise ValueError(f"{label} must remain under {allowed_root}: {path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    config = json.loads((repo_root / "configs" / "validation-baseline.json").read_text(encoding="utf-8"))
    dataset = load_dataset(data_root)
    data_inventory = inventory(dataset)
    expected = {
        "observation_count": 707,
        "annotator_view_count": 1154,
        "annotation_count": 8199,
        "test_image_file_count": 180,
    }
    for key, value in expected.items():
        if data_inventory[key] != value:
            raise ValueError(f"Expected {value} {key}, found {data_inventory[key]}")
    split = create_observation_split(
        dataset,
        repo_root / "data" / "splits",
        seed=int(config["seed"]),
        validation_fraction=float(config["validation_fraction"]),
    )
    if split.sha256 != config["expected_split_sha256"]:
        raise RuntimeError("Frozen validation split digest changed")
    oof_manifest = repo_root / "data" / "splits" / "observation_oof_seed42_5fold.csv"
    if not oof_manifest.is_file():
        raise FileNotFoundError(oof_manifest)

    training_metadata_path = detector_root / "run-metadata.json"
    training_metadata = json.loads(training_metadata_path.read_text(encoding="utf-8"))
    if (
        training_metadata.get("status") != "complete"
        or training_metadata.get("training_observations") != 707
        or training_metadata.get("holdout_used_for_final_training") is not True
        or training_metadata.get("test_images_used_for_training") is not False
        or training_metadata.get("test_labels_used") is not False
    ):
        raise ValueError("Expected a completed all-707-group final fit with no test data in training")
    checkpoint = detector_root / "final.pt"
    checkpoint_hash = _sha256(checkpoint)
    if checkpoint_hash != training_metadata.get("checkpoint_sha256"):
        raise ValueError("Final-fit checkpoint hash does not match its training metadata")

    import torch
    import ultralytics
    from ultralytics import YOLO

    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select --device cpu only for a smoke run")
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(output_dir / "ultralytics-config")
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)

    test_names = sorted(path.name for path in dataset.test_image_dir.iterdir() if path.is_file())
    if len(test_names) != 180 or len({Path(name).stem for name in test_names}) != 180:
        raise ValueError("Expected exactly 180 unique supplied H-alpha test images")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model = YOLO(str(checkpoint))
    predictions: dict[str, list[dict[str, Any]]] = {}
    started = time.perf_counter()
    for index, file_name in enumerate(test_names, start=1):
        image_path = dataset.test_image_dir / file_name
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (2048, 2048):
            raise ValueError(f"Expected a readable 2048x2048 test image: {image_path}")
        predictions[Path(file_name).stem] = _predict_image(
            model,
            image,
            image_size=args.image_size,
            confidence=args.confidence,
            nms_iou=args.nms_iou,
            max_detections=args.max_detections,
            device=args.device,
        )
        if index % 20 == 0 or index == len(test_names):
            print(f"Predicted {index}/{len(test_names)} supplied test images", flush=True)

    predictions_path = output_dir / "test_detections_tta8_raw.json"
    predictions_path.write_text(json.dumps(predictions, separators=(",", ":")), encoding="utf-8")
    prediction_seconds = time.perf_counter() - started
    model_record = {
        "name": "full_training_yolo11s_seg",
        "checkpoint_sha256": checkpoint_hash,
        "prediction_observations": len(predictions),
        "prediction_instances": sum(len(items) for items in predictions.values()),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        "predictions_file": predictions_path.name,
    }
    metadata = {
        "status": "complete",
        "partition": "competition_test_inference",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "source_script_sha256": _sha256(Path(__file__).resolve()),
        "training_run_metadata_sha256": _sha256(training_metadata_path),
        "oof_manifest_sha256": _sha256(oof_manifest),
        "frozen_holdout_sha256": split.sha256,
        "full_training_final_fit": True,
        "training_observations": 707,
        "holdout_used_for_training": True,
        "training_data": "all 707 competition-provided training observations and 1,154 annotator views",
        "test_observations": len(predictions),
        "test_labels_used": False,
        "test_images_used_for_training": False,
        "prediction_instances": sum(len(items) for items in predictions.values()),
        "predictions_file": str(predictions_path.relative_to(repo_root)),
        "prediction_sha256": _sha256(predictions_path),
        "configuration": {
            "transforms": [
                "rot0", "rot0_hflip", "rot90", "rot90_hflip",
                "rot180", "rot180_hflip", "rot270", "rot270_hflip",
            ],
            "image_size": args.image_size,
            "confidence": args.confidence,
            "nms_iou": args.nms_iou,
            "max_detections_per_transform": args.max_detections,
            "device": args.device,
        },
        "runtime": {
            "prediction_seconds": round(prediction_seconds, 3),
            "peak_gpu_memory_bytes": model_record["peak_gpu_memory_bytes"],
            "python": platform.python_version(),
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "models": [model_record],
    }
    (output_dir / "run-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "partition": metadata["partition"],
                "test_observations": metadata["test_observations"],
                "prediction_instances": metadata["prediction_instances"],
                "prediction_seconds": metadata["runtime"]["prediction_seconds"],
                "peak_gpu_memory_bytes": metadata["runtime"]["peak_gpu_memory_bytes"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
