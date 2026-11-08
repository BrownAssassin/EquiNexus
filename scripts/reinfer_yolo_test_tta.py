# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Run the grouped YOLO checkpoints on supplied test images with D4 TTA."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from filament_baseline.data import inventory, load_dataset
from filament_baseline.split import create_observation_split
from scripts.reinfer_yolo_oof_tta import _predict_image, _sha256


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run each validated grouped YOLO checkpoint on the 180 competition test images."
    )
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
    data_root = (
        args.data_root or repo_root / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026"
    ).expanduser().resolve()
    detector_root = args.detector_run_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for label, path, allowed_root in (
        ("competition data", data_root, repo_root / "data"),
        ("detector checkpoints", detector_root, repo_root / "artifacts"),
        ("test predictions", output_dir, repo_root / "artifacts"),
    ):
        if not path.is_relative_to(allowed_root.resolve()):
            raise ValueError(f"{label} must remain under {allowed_root}: {path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    config_path = repo_root / "configs" / "validation-baseline.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset = load_dataset(data_root)
    expected_inventory = {
        "observation_count": 707,
        "annotator_view_count": 1154,
        "annotation_count": 8199,
        "test_image_file_count": 180,
    }
    data_inventory = inventory(dataset)
    for key, expected in expected_inventory.items():
        if data_inventory[key] != expected:
            raise ValueError(f"Expected {expected} {key}, found {data_inventory[key]}")

    frozen_split = create_observation_split(
        dataset,
        repo_root / "data" / "splits",
        seed=int(config["seed"]),
        validation_fraction=float(config["validation_fraction"]),
    )
    if frozen_split.sha256 != config["expected_split_sha256"]:
        raise RuntimeError("The frozen holdout split digest changed")
    oof_manifest = repo_root / "data" / "splits" / "observation_oof_seed42_5fold.csv"
    if not oof_manifest.is_file():
        raise FileNotFoundError(oof_manifest)

    parent_metadata_path = detector_root / "run-metadata.json"
    parent = json.loads(parent_metadata_path.read_text(encoding="utf-8"))
    if parent.get("status") != "complete":
        raise ValueError("The detector training run is incomplete")
    if parent.get("oof_manifest_sha256") != _sha256(oof_manifest):
        raise ValueError("Detector checkpoints do not match the local OOF manifest")
    if parent.get("frozen_holdout_sha256") != frozen_split.sha256:
        raise ValueError("Detector checkpoints do not match the frozen holdout")
    fold_entries = {int(item["fold"]): item for item in parent.get("folds", [])}
    if set(fold_entries) != set(range(5)):
        raise ValueError("Expected exactly five grouped YOLO checkpoints")

    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(output_dir / "ultralytics-config")
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_MODEL_DIR"] = str(repo_root / "artifacts" / "pretrained" / "ultralytics")

    import torch
    import ultralytics
    from ultralytics import YOLO

    if not torch.cuda.is_available() and args.device != "cpu":
        raise RuntimeError("CUDA is unavailable; select --device cpu only for a smoke run")

    test_names = sorted(path.name for path in dataset.test_image_dir.iterdir() if path.is_file())
    if len(test_names) != 180 or len({Path(name).stem for name in test_names}) != 180:
        raise ValueError("Expected 180 unique supplied competition test images")
    all_predictions = {Path(name).stem: [] for name in test_names}
    fold_records = []
    peak_allocations = []
    started = time.perf_counter()

    for fold in range(5):
        checkpoint = detector_root / f"fold-{fold}" / "best.pt"
        expected_hash = fold_entries[fold]["best_checkpoint_sha256"]
        checkpoint_hash = _sha256(checkpoint)
        if checkpoint_hash != expected_hash:
            raise ValueError(f"Fold {fold} checkpoint hash mismatch")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        model = YOLO(str(checkpoint))
        fold_predictions = {}
        for index, file_name in enumerate(test_names, start=1):
            image_path = dataset.test_image_dir / file_name
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (2048, 2048):
                raise ValueError(f"Expected a readable 2048x2048 test image: {image_path}")
            stem = Path(file_name).stem
            detections = _predict_image(
                model,
                image,
                image_size=args.image_size,
                confidence=args.confidence,
                nms_iou=args.nms_iou,
                max_detections=args.max_detections,
                device=args.device,
            )
            fold_predictions[stem] = detections
            all_predictions[stem].extend(detections)
            if index % 20 == 0 or index == len(test_names):
                print(f"Fold {fold}: predicted {index}/{len(test_names)} test images", flush=True)

        fold_path = output_dir / f"fold-{fold}-test-predictions.json"
        fold_path.write_text(json.dumps(fold_predictions, separators=(",", ":")), encoding="utf-8")
        peak_bytes = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        peak_allocations.append(peak_bytes)
        fold_records.append(
            {
                "fold": fold,
                "checkpoint_sha256": checkpoint_hash,
                "prediction_observations": len(fold_predictions),
                "prediction_instances": sum(len(items) for items in fold_predictions.values()),
                "peak_gpu_memory_bytes": peak_bytes,
                "predictions_file": fold_path.name,
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if set(all_predictions) != {Path(name).stem for name in test_names}:
        raise AssertionError("Detector predictions do not exactly cover the test images")
    predictions_path = output_dir / "test_detections_tta8_raw.json"
    predictions_path.write_text(
        json.dumps(all_predictions, separators=(",", ":")), encoding="utf-8"
    )
    metadata = {
        "status": "complete",
        "partition": "competition_test_inference",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "source_script_sha256": _sha256(Path(__file__).resolve()),
        "shared_tta_helper_sha256": _sha256(Path(__file__).with_name("reinfer_yolo_oof_tta.py")),
        "parent_detector_run_metadata_sha256": _sha256(parent_metadata_path),
        "oof_manifest_sha256": parent["oof_manifest_sha256"],
        "frozen_holdout_sha256": frozen_split.sha256,
        "test_observations": len(all_predictions),
        "test_labels_used": False,
        "test_images_used_for_training": False,
        "training_data": "five grouped checkpoints trained only on competition-provided train annotations",
        "prediction_instances": sum(len(items) for items in all_predictions.values()),
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
            "prediction_seconds": round(time.perf_counter() - started, 3),
            "peak_gpu_memory_bytes": max(peak_allocations, default=0),
            "python": platform.python_version(),
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "folds": fold_records,
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
