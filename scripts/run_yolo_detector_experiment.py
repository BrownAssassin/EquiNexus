# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from filament_baseline.data import find_dataset_root, inventory, load_dataset
from filament_baseline.predictor import encode_binary_mask
from filament_baseline.runtime import hardware_info, package_versions
from filament_baseline.split import (
    create_observation_split,
    load_observation_split_manifest,
)
from export_yolo_dataset import export_yolo_dataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _find_checkpoint(cache_root: Path, model_name: str) -> Path | None:
    for path in cache_root.rglob(model_name):
        if path.is_file():
            return path
    return None


def _predict_validation(
    model: Any,
    dataset,
    validation_names: tuple[str, ...],
    *,
    image_size: int,
    device: str,
    confidence: float,
    nms_iou: float,
    max_detections: int,
) -> dict[str, list[dict[str, Any]]]:
    predictions: dict[str, list[dict[str, Any]]] = {}
    for index, file_name in enumerate(validation_names, start=1):
        path = dataset.train_image_dir / file_name
        result = model.predict(
            source=str(path),
            imgsz=image_size,
            conf=confidence,
            iou=nms_iou,
            max_det=max_detections,
            retina_masks=True,
            device=device,
            verbose=False,
        )[0]
        found: list[dict[str, Any]] = []
        if result.masks is not None and result.boxes is not None:
            masks = result.masks.data.detach().cpu().numpy() > 0.5
            scores = result.boxes.conf.detach().cpu().numpy()
            for mask, score in zip(masks, scores, strict=True):
                if mask.shape != (2048, 2048):
                    import cv2
                    import numpy as np

                    mask = cv2.resize(
                        mask.astype(np.uint8), (2048, 2048), interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                if not mask.any():
                    continue
                rle = encode_binary_mask(mask)
                found.append({"score": float(score), "counts": rle["counts"]})
        predictions[Path(file_name).stem] = found
        if index % 20 == 0 or index == len(validation_names):
            print(f"Predicted {index}/{len(validation_names)} validation observations", flush=True)
    return predictions


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.expanduser().resolve()
    kaggle_runtime = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle/input").is_dir()
    if args.data_root is None and kaggle_runtime:
        data_root = find_dataset_root(Path("/kaggle/input"))
    else:
        data_root = (args.data_root or repo_root / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026").expanduser().resolve()
    if not kaggle_runtime and not _within(data_root, repo_root / "data"):
        raise ValueError(f"Local competition data must remain under data/: {data_root}")

    if args.run_root is None:
        run_root = (
            Path("/kaggle/working/equinexus") / args.run_name
            if kaggle_runtime
            else repo_root / "artifacts" / "detector-experiments" / args.run_name
        )
    else:
        run_root = args.run_root.expanduser().resolve()
    allowed_root = Path("/kaggle/working").resolve() if kaggle_runtime else repo_root
    if not _within(run_root, allowed_root):
        raise ValueError(f"Run artifacts must remain inside {allowed_root}: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)

    model_cache = run_root / "ultralytics-config"
    model_cache.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(model_cache)
    os.environ["TORCH_HOME"] = str(run_root / "torch-home")

    config = json.loads((repo_root / "configs" / "validation-baseline.json").read_text())
    dataset = load_dataset(data_root)
    data_inventory = inventory(dataset)
    expected_inventory = {"observation_count": 707, "annotator_view_count": 1154, "annotation_count": 8199}
    for key, expected in expected_inventory.items():
        if data_inventory[key] != expected:
            raise ValueError(f"Expected {expected} {key}, found {data_inventory[key]}")
    if args.split_manifest is not None:
        split_manifest = args.split_manifest.expanduser().resolve()
        if not _within(split_manifest, repo_root / "data" / "splits"):
            raise ValueError(f"Split manifests must remain under data/splits/: {split_manifest}")
        split = load_observation_split_manifest(
            dataset, split_manifest, allow_subset=True
        )
        split_mode = "explicit_grouped_manifest"
    else:
        split = create_observation_split(
            dataset,
            repo_root / "data" / "splits",
            seed=int(config["seed"]),
            validation_fraction=float(config["validation_fraction"]),
        )
        if split.sha256 != config["expected_split_sha256"]:
            raise RuntimeError(f"Frozen split changed: {split.sha256}")
        split_mode = "frozen_validation_holdout"

    export_root = repo_root / "data" / "processed" / "yolo" / args.run_name
    if not _within(export_root, repo_root / "data" / "processed"):
        raise ValueError(f"Exported detector data must remain under data/processed: {export_root}")
    export_manifest = export_yolo_dataset(dataset, split, export_root)
    export_manifest.pop("views", None)
    (run_root / "export_manifest.json").write_text(
        json.dumps(export_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    import torch
    import ultralytics
    from ultralytics import YOLO

    if not torch.cuda.is_available() and args.device != "cpu":
        raise RuntimeError("No CUDA GPU is available; choose --device cpu only for a smoke run")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    pretrained_name = Path(args.model).name
    detector = YOLO(args.model)
    candidate_paths = (
        Path(args.model).expanduser(),
        Path.cwd() / pretrained_name,
        model_cache / pretrained_name,
    )
    pretrained_path = next((path.resolve() for path in candidate_paths if path.is_file()), None)
    if pretrained_path is None:
        pretrained_path = _find_checkpoint(model_cache, pretrained_name)
    pretrained_hash = _sha256(pretrained_path) if pretrained_path else None

    train_started = time.perf_counter()
    detector.train(
        data=str(export_root / "data.yaml"),
        imgsz=args.image_size,
        epochs=args.epochs,
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        seed=int(config["seed"]),
        deterministic=False,
        single_cls=True,
        cos_lr=True,
        plots=False,
        fraction=args.fraction,
        val=True,
        mosaic=0.0,
        close_mosaic=0,
        mixup=0.0,
        copy_paste=0.0,
        fliplr=0.5,
        flipud=0.5,
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        translate=0.05,
        scale=0.2,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.2,
        overlap_mask=True,
        mask_ratio=args.mask_ratio,
        project=str(run_root / "train"),
        name="yolo-seg",
        exist_ok=True,
        save=True,
        save_period=max(1, args.epochs // 10),
        amp=True,
        cache=False,
        verbose=True,
    )
    training_seconds = time.perf_counter() - train_started

    weights_dir = run_root / "train" / "yolo-seg" / "weights"
    best_path = weights_dir / "best.pt"
    last_path = weights_dir / "last.pt"
    if not best_path.is_file():
        raise FileNotFoundError(f"YOLO training did not produce {best_path}")
    import shutil

    shutil.copy2(best_path, run_root / "best.pt")
    if last_path.is_file():
        shutil.copy2(last_path, run_root / "last.pt")
    prediction_started = time.perf_counter()
    validation_predictions: dict[str, dict[str, Any]] = {}
    for checkpoint_name, checkpoint_path in (("best", best_path), ("last", last_path)):
        if not checkpoint_path.is_file():
            continue
        detector_for_inference = YOLO(str(checkpoint_path))
        predictions = _predict_validation(
            detector_for_inference,
            dataset,
            split.validation_names,
            image_size=args.image_size,
            device=args.device,
            confidence=args.prediction_confidence,
            nms_iou=args.nms_iou,
            max_detections=args.max_detections,
        )
        checkpoint_predictions_path = run_root / f"validation_detections_{checkpoint_name}.json"
        checkpoint_predictions_path.write_text(
            json.dumps(predictions, separators=(",", ":")), encoding="utf-8"
        )
        validation_predictions[checkpoint_name] = {
            "file": checkpoint_predictions_path.name,
            "observations": len(predictions),
            "instances": sum(len(items) for items in predictions.values()),
        }
        print(
            f"Saved {checkpoint_name} checkpoint predictions for {len(predictions)} observations",
            flush=True,
        )
    if "best" not in validation_predictions:
        raise FileNotFoundError("The best checkpoint produced no validation predictions")
    predictions_path = run_root / "validation_detections.json"
    predictions_path.write_bytes(
        (run_root / validation_predictions["best"]["file"]).read_bytes()
    )
    prediction_seconds = time.perf_counter() - prediction_started
    run_metadata = {
        "status": "complete",
        "run_name": args.run_name,
        "source_commit": os.environ.get("EQUINEXUS_SOURCE_COMMIT"),
        "source_bundle_sha256": os.environ.get("EQUINEXUS_SOURCE_BUNDLE_SHA256"),
        "split_sha256": split.sha256,
        "split_mode": split_mode,
        "split_manifest": str(split.manifest_path.relative_to(repo_root)),
        "training_observations": len(split.training_names),
        "validation_observations": len(split.validation_names),
        "validation_annotator_views": sum(
            len(dataset.views_by_observation[name]) for name in split.validation_names
        ),
        "dataset_inventory": data_inventory,
        "training_data": os.environ.get(
            "EQUINEXUS_TRAINING_DATA_DESCRIPTION",
            "competition-provided training annotations only",
        ),
        "external_supervised_data": os.environ.get(
            "EQUINEXUS_EXTERNAL_SUPERVISED_DATA", "false"
        ).lower() == "true",
        "pretrained_model": {
            "name": pretrained_name,
            "source": os.environ.get(
                "EQUINEXUS_PRETRAINED_SOURCE",
                "Ultralytics YOLO11-seg COCO pretrained weights",
            ),
            "license": os.environ.get("EQUINEXUS_PRETRAINED_LICENSE", "AGPL-3.0"),
            "sha256": pretrained_hash,
        },
        "configuration": {
            "model": args.model,
            "image_size": args.image_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "device": args.device,
            "workers": args.workers,
            "patience": args.patience,
            "seed": int(config["seed"]),
            "fraction": args.fraction,
            "mask_ratio": args.mask_ratio,
            "prediction_confidence": args.prediction_confidence,
            "nms_iou": args.nms_iou,
            "max_detections": args.max_detections,
            "augmentation": "flips, mild translation/scale/brightness; mosaic/mixup disabled",
        },
        "best_checkpoint_sha256": _sha256(run_root / "best.pt"),
        "last_checkpoint_sha256": _sha256(run_root / "last.pt") if (run_root / "last.pt").is_file() else None,
        "training_seconds": training_seconds,
        "prediction_seconds": prediction_seconds,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        "hardware": hardware_info(),
        "package_versions": {**package_versions(), "ultralytics": ultralytics.__version__},
        "predictions_file": predictions_path.name,
        "prediction_observations": validation_predictions["best"]["observations"],
        "prediction_instances": validation_predictions["best"]["instances"],
        "validation_predictions": validation_predictions,
    }
    (run_root / "run-metadata.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: run_metadata[key] for key in (
        "run_name", "split_sha256", "training_seconds", "prediction_seconds",
        "peak_gpu_memory_bytes", "prediction_instances", "best_checkpoint_sha256",
    )}, indent=2, sort_keys=True), flush=True)
    return run_metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Train an instance detector on the frozen grouped split.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--run-name", default="yolo11s-seg-1024-step40")
    parser.add_argument("--model", default="yolo11s-seg.pt")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--device", default="0")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--mask-ratio", type=int, default=2)
    parser.add_argument("--prediction-confidence", type=float, default=0.05)
    parser.add_argument("--nms-iou", type=float, default=0.6)
    parser.add_argument("--max-detections", type=int, default=150)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not 0 < args.fraction <= 1:
        parser.error("epochs, batch size, and fraction must be positive")
    if args.image_size < 256 or args.mask_ratio < 1:
        parser.error("image size must be at least 256 and mask ratio must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
