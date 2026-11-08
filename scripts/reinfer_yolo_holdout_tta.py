# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Predict the frozen holdout with all five grouped YOLO checkpoints and D4 TTA."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from filament_baseline.data import load_dataset
from filament_baseline.split import create_observation_split
from scripts.reinfer_yolo_oof_tta import _predict_image, _sha256


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run all five OOF YOLO checkpoints on the frozen holdout with D4 TTA."
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
    data_root = (args.data_root or repo_root / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026").expanduser().resolve()
    detector_root = args.detector_run_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not data_root.is_relative_to(repo_root / "data"):
        raise ValueError("Competition data must remain under data/")
    if not detector_root.is_relative_to(repo_root / "artifacts"):
        raise ValueError("Detector checkpoints must remain under artifacts/")
    if not output_dir.is_relative_to(repo_root / "artifacts"):
        raise ValueError("Holdout predictions must remain under artifacts/")

    config = json.loads((repo_root / "configs" / "validation-baseline.json").read_text())
    dataset = load_dataset(data_root)
    split = create_observation_split(
        dataset,
        repo_root / "data" / "splits",
        seed=int(config["seed"]),
        validation_fraction=float(config["validation_fraction"]),
    )
    if split.sha256 != config["expected_split_sha256"]:
        raise RuntimeError(f"Frozen split changed: {split.sha256}")

    manifest_path = repo_root / "data" / "splits" / "observation_oof_seed42_5fold.csv"
    parent_metadata_path = detector_root / "run-metadata.json"
    parent_metadata = json.loads(parent_metadata_path.read_text(encoding="utf-8"))
    if parent_metadata.get("oof_manifest_sha256") != _sha256(manifest_path):
        raise ValueError("Detector checkpoints do not match the grouped OOF manifest")
    if parent_metadata.get("frozen_holdout_sha256") != split.sha256:
        raise ValueError("Detector checkpoints record a different frozen holdout")

    import torch
    import ultralytics
    from ultralytics import YOLO

    if not torch.cuda.is_available() and args.device != "cpu":
        raise RuntimeError("CUDA is unavailable; select --device cpu only for a smoke run")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ultralytics-config").mkdir(exist_ok=True)
    import os

    os.environ["YOLO_CONFIG_DIR"] = str(output_dir / "ultralytics-config")
    torch_peaks: list[int] = []
    all_predictions: dict[str, list[dict]] = {
        Path(name).stem: [] for name in split.validation_names
    }
    fold_records = []
    started = time.perf_counter()
    holdout_names = tuple(sorted(split.validation_names))
    for fold in range(5):
        checkpoint = detector_root / f"fold-{fold}" / "best.pt"
        expected_hash = parent_metadata["folds"][fold]["best_checkpoint_sha256"]
        checkpoint_hash = _sha256(checkpoint)
        if checkpoint_hash != expected_hash:
            raise ValueError(f"Fold {fold} checkpoint hash does not match parent metadata")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        model = YOLO(str(checkpoint))
        fold_predictions: dict[str, list[dict]] = {}
        for index, file_name in enumerate(holdout_names, start=1):
            image_path = dataset.train_image_dir / file_name
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (2048, 2048):
                raise ValueError(f"Expected a readable 2048x2048 image: {image_path}")
            stem = Path(file_name).stem
            found = _predict_image(
                model,
                image,
                image_size=args.image_size,
                confidence=args.confidence,
                nms_iou=args.nms_iou,
                max_detections=args.max_detections,
                device=args.device,
            )
            fold_predictions[stem] = found
            all_predictions[stem].extend(found)
            if index % 20 == 0 or index == len(holdout_names):
                print(f"Fold {fold}: predicted {index}/{len(holdout_names)} holdout observations", flush=True)

        fold_path = output_dir / f"fold-{fold}-predictions.json"
        fold_path.write_text(json.dumps(fold_predictions, separators=(",", ":")), encoding="utf-8")
        peak_memory = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        torch_peaks.append(peak_memory)
        fold_records.append(
            {
                "fold": fold,
                "checkpoint_sha256": checkpoint_hash,
                "prediction_observations": len(fold_predictions),
                "prediction_instances": sum(map(len, fold_predictions.values())),
                "peak_gpu_memory_bytes": peak_memory,
                "predictions_file": fold_path.name,
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    expected_stems = {Path(name).stem for name in split.validation_names}
    if set(all_predictions) != expected_stems:
        raise AssertionError("Holdout predictions do not exactly cover the frozen observations")
    prediction_path = output_dir / "holdout_detections_raw.json"
    prediction_path.write_text(
        json.dumps(all_predictions, separators=(",", ":")), encoding="utf-8"
    )
    metadata = {
        "status": "complete",
        "partition": "frozen_validation_holdout",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_name": output_dir.name,
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "source_script_sha256": _sha256(Path(__file__).resolve()),
        "shared_tta_helper_sha256": _sha256(Path(__file__).with_name("reinfer_yolo_oof_tta.py")),
        "parent_detector_run_metadata_sha256": _sha256(parent_metadata_path),
        "detector_run_root": str(detector_root.relative_to(repo_root)),
        "oof_manifest_sha256": parent_metadata["oof_manifest_sha256"],
        "frozen_holdout_sha256": split.sha256,
        "holdout_observations": len(all_predictions),
        "holdout_used_for_training": False,
        "prediction_instances": sum(map(len, all_predictions.values())),
        "prediction_sha256": _sha256(prediction_path),
        "predictions_file": str(prediction_path.relative_to(repo_root)),
        "configuration": {
            "transforms": ["rot0", "rot0_hflip", "rot90", "rot90_hflip", "rot180", "rot180_hflip", "rot270", "rot270_hflip"],
            "image_size": args.image_size,
            "confidence": args.confidence,
            "nms_iou": args.nms_iou,
            "max_detections_per_transform": args.max_detections,
            "device": args.device,
            "training_data": "competition-provided training annotations only",
        },
        "runtime": {
            "prediction_seconds": round(time.perf_counter() - started, 3),
            "peak_gpu_memory_bytes": max(torch_peaks, default=0),
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
                "frozen_holdout_sha256": metadata["frozen_holdout_sha256"],
                "holdout_observations": metadata["holdout_observations"],
                "prediction_instances": metadata["prediction_instances"],
                "prediction_seconds": metadata["runtime"]["prediction_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
