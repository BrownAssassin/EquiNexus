# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from filament_baseline.data import load_dataset
from filament_baseline.runtime import hardware_info, package_versions
from filament_baseline.split import create_observation_split
from run_yolo_detector_experiment import _predict_validation, _sha256


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one saved detector checkpoint on the frozen grouped validation observations."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prediction-name", default="last")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--device", default="0")
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--nms-iou", type=float, default=0.6)
    parser.add_argument("--max-detections", type=int, default=150)
    args = parser.parse_args()

    repo_root = args.repo_root.expanduser().resolve()
    data_root = (
        args.data_root.expanduser().resolve()
        if args.data_root
        else repo_root / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026"
    )
    run_root = args.run_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    data_root.relative_to(repo_root / "data")
    run_root.relative_to(repo_root / "artifacts")
    checkpoint.relative_to(repo_root)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", args.prediction_name):
        raise ValueError("prediction-name must use only letters, digits, dot, underscore, or dash")

    run_root.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(run_root / "ultralytics-config")
    os.environ["TORCH_HOME"] = str(run_root / "torch-home")
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TORCH_HOME"]).mkdir(parents=True, exist_ok=True)

    import torch
    import ultralytics
    from ultralytics import YOLO

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

    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu only for a smoke run")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model = YOLO(str(checkpoint))
    started = time.perf_counter()
    predictions = _predict_validation(
        model,
        dataset,
        split.validation_names,
        image_size=args.image_size,
        device=args.device,
        confidence=args.confidence,
        nms_iou=args.nms_iou,
        max_detections=args.max_detections,
    )
    prediction_seconds = time.perf_counter() - started
    output_path = run_root / f"validation_detections_{args.prediction_name}.json"
    output_path.write_text(
        json.dumps(predictions, separators=(",", ":")), encoding="utf-8"
    )
    metadata = {
        "checkpoint": str(checkpoint.relative_to(repo_root)),
        "checkpoint_sha256": _sha256(checkpoint),
        "prediction_file": output_path.name,
        "prediction_name": args.prediction_name,
        "prediction_observations": len(predictions),
        "prediction_instances": sum(len(items) for items in predictions.values()),
        "split_sha256": split.sha256,
        "image_size": args.image_size,
        "device": args.device,
        "confidence": args.confidence,
        "nms_iou": args.nms_iou,
        "max_detections": args.max_detections,
        "prediction_seconds": prediction_seconds,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        "hardware": hardware_info(),
        "package_versions": {**package_versions(), "ultralytics": ultralytics.__version__},
    }
    (run_root / f"validation_{args.prediction_name}_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
