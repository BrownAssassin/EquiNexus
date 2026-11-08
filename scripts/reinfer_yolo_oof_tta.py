# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Re-infer grouped YOLO OOF checkpoints with eight-way dihedral TTA."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from filament_baseline.data import load_dataset
from filament_baseline.predictor import encode_binary_mask
from filament_baseline.split import create_observation_split


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _transforms(image: np.ndarray):
    """Yield each D4 image transform and its inverse mask transform."""
    for quarter_turns in range(4):
        rotated = np.ascontiguousarray(np.rot90(image, quarter_turns))
        for horizontal_flip in (False, True):
            if horizontal_flip:
                transformed = cv2.flip(rotated, 1)
            else:
                transformed = rotated

            def invert(mask: np.ndarray, *, k=quarter_turns, flip=horizontal_flip):
                if flip:
                    mask = np.fliplr(mask)
                if k:
                    mask = np.rot90(mask, -k)
                return np.ascontiguousarray(mask)

            name = f"rot{quarter_turns * 90}" + ("_hflip" if horizontal_flip else "")
            yield name, np.ascontiguousarray(transformed), invert


def _predict_image(
    model: Any,
    image: np.ndarray,
    *,
    image_size: int,
    confidence: float,
    nms_iou: float,
    max_detections: int,
    device: str,
) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    for _, transformed, inverse in _transforms(image):
        result = model.predict(
            source=transformed,
            imgsz=image_size,
            conf=confidence,
            iou=nms_iou,
            max_det=max_detections,
            retina_masks=True,
            device=device,
            verbose=False,
        )[0]
        if result.masks is None or result.boxes is None:
            continue
        masks = result.masks.data.detach().cpu().numpy() > 0.5
        scores = result.boxes.conf.detach().cpu().numpy()
        for mask, score in zip(masks, scores, strict=True):
            if mask.shape != (2048, 2048):
                mask = cv2.resize(
                    mask.astype(np.uint8), (2048, 2048), interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            mask = inverse(mask)
            if mask.shape != (2048, 2048) or not mask.any():
                continue
            detections.append(
                {"score": float(score), "counts": encode_binary_mask(mask)["counts"]}
            )
    return detections


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-infer each five-fold YOLO checkpoint on its held-out groups with D4 TTA."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--detector-run-root", type=Path, required=True)
    parser.add_argument("--oof-manifest", type=Path, default=None)
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
    manifest_path = (
        args.oof_manifest
        or repo_root / "data" / "splits" / "observation_oof_seed42_5fold.csv"
    ).expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for label, path, allowed_root in (
        ("competition data", data_root, repo_root / "data"),
        ("detector checkpoints", detector_root, repo_root / "artifacts"),
        ("OOF manifest", manifest_path, repo_root / "data" / "splits"),
        ("outputs", output_dir, repo_root / "artifacts"),
    ):
        if not _inside(path, allowed_root):
            raise ValueError(f"{label} must remain under {allowed_root}: {path}")

    config = json.loads((repo_root / "configs" / "validation-baseline.json").read_text())
    dataset = load_dataset(data_root)
    frozen = create_observation_split(
        dataset,
        repo_root / "data" / "splits",
        seed=int(config["seed"]),
        validation_fraction=float(config["validation_fraction"]),
    )
    expected_frozen = config["expected_split_sha256"]
    if frozen.sha256 != expected_frozen:
        raise RuntimeError(f"Frozen split changed: {frozen.sha256}")

    with manifest_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"file_name", "year", "site", "oof_fold"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"OOF manifest must include {sorted(required)}")
    names = [row["file_name"] for row in rows]
    if len(names) != len(set(names)) or set(names) != set(frozen.training_names):
        raise ValueError("OOF manifest must cover the frozen training partition exactly once")
    if set(names) & set(frozen.validation_names):
        raise ValueError("Frozen holdout observations cannot be included in OOF inference")
    by_fold: dict[int, list[str]] = {fold: [] for fold in range(5)}
    for row in rows:
        fold = int(row["oof_fold"])
        if fold not in by_fold:
            raise ValueError(f"Unexpected OOF fold: {fold}")
        view = dataset.views_by_observation[row["file_name"]][0]
        if int(row["year"]) != view.year or row["site"] != view.site:
            raise ValueError(f"Manifest metadata mismatch for {row['file_name']}")
        by_fold[fold].append(row["file_name"])
    if any(len(group) != 113 for group in by_fold.values()):
        raise ValueError("Expected five OOF folds of 113 observations each")

    parent_metadata_path = detector_root / "run-metadata.json"
    if not parent_metadata_path.is_file():
        raise FileNotFoundError(f"Missing detector run metadata: {parent_metadata_path}")
    parent_metadata = json.loads(parent_metadata_path.read_text(encoding="utf-8"))
    if parent_metadata.get("oof_manifest_sha256") != _sha256(manifest_path):
        raise ValueError("Detector checkpoint run used a different OOF manifest")
    if parent_metadata.get("frozen_holdout_sha256") != frozen.sha256:
        raise ValueError("Detector checkpoint run records a different frozen holdout")

    import torch
    import ultralytics
    from ultralytics import YOLO

    if not torch.cuda.is_available() and args.device != "cpu":
        raise RuntimeError("CUDA is unavailable; select --device cpu only for a smoke run")
    output_dir.mkdir(parents=True, exist_ok=True)
    os_config = output_dir / "ultralytics-config"
    os_config.mkdir(exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(os_config)
    torch_metrics: list[int] = []
    fold_records = []
    all_predictions: dict[str, list[dict[str, Any]]] = {}
    started = time.perf_counter()
    for fold in range(5):
        checkpoint = detector_root / f"fold-{fold}" / "best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing fold-{fold} checkpoint: {checkpoint}")
        expected_hash = parent_metadata["folds"][fold]["best_checkpoint_sha256"]
        checkpoint_hash = _sha256(checkpoint)
        if checkpoint_hash != expected_hash:
            raise ValueError(f"Fold {fold} checkpoint hash does not match parent metadata")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        model = YOLO(str(checkpoint))
        fold_predictions: dict[str, list[dict[str, Any]]] = {}
        fold_names = sorted(by_fold[fold])
        for index, file_name in enumerate(fold_names, start=1):
            image_path = dataset.train_image_dir / file_name
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (2048, 2048):
                raise ValueError(f"Expected a readable 2048x2048 image: {image_path}")
            fold_predictions[Path(file_name).stem] = _predict_image(
                model,
                image,
                image_size=args.image_size,
                confidence=args.confidence,
                nms_iou=args.nms_iou,
                max_detections=args.max_detections,
                device=args.device,
            )
            if index % 10 == 0 or index == len(fold_names):
                print(f"Fold {fold}: predicted {index}/{len(fold_names)} observations", flush=True)

        fold_output = output_dir / f"fold-{fold}-predictions.json"
        fold_output.write_text(
            json.dumps(fold_predictions, separators=(",", ":")), encoding="utf-8"
        )
        all_predictions.update(fold_predictions)
        peak_memory = (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        )
        torch_metrics.append(peak_memory)
        fold_records.append(
            {
                "fold": fold,
                "checkpoint_sha256": checkpoint_hash,
                "prediction_observations": len(fold_predictions),
                "prediction_instances": sum(map(len, fold_predictions.values())),
                "peak_gpu_memory_bytes": peak_memory,
                "predictions_file": fold_output.name,
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    expected_stems = {Path(name).stem for name in frozen.training_names}
    if set(all_predictions) != expected_stems:
        raise AssertionError("TTA predictions do not cover the OOF groups exactly")
    predictions_path = output_dir / "oof_detections.json"
    predictions_path.write_text(
        json.dumps(all_predictions, separators=(",", ":")), encoding="utf-8"
    )
    metadata = {
        "status": "complete",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_name": output_dir.name,
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "source_script_sha256": _sha256(Path(__file__).resolve()),
        "parent_detector_run_metadata_sha256": _sha256(parent_metadata_path),
        "detector_run_root": str(detector_root.relative_to(repo_root)),
        "oof_manifest": str(manifest_path.relative_to(repo_root)),
        "oof_manifest_sha256": _sha256(manifest_path),
        "frozen_holdout_sha256": frozen.sha256,
        "frozen_holdout_used_for_training_or_selection": False,
        "oof_observations": len(all_predictions),
        "prediction_instances": sum(map(len, all_predictions.values())),
        "prediction_sha256": _sha256(predictions_path),
        "predictions_file": str(predictions_path.relative_to(repo_root)),
        "configuration": {
            "transforms": [name for name, _, _ in _transforms(np.zeros((1, 1, 3), dtype=np.uint8))],
            "image_size": args.image_size,
            "confidence": args.confidence,
            "nms_iou": args.nms_iou,
            "max_detections_per_transform": args.max_detections,
            "device": args.device,
            "training_data": "competition-provided training annotations only",
        },
        "runtime": {
            "prediction_seconds": round(time.perf_counter() - started, 3),
            "peak_gpu_memory_bytes": max(torch_metrics, default=0),
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
                "oof_manifest_sha256": metadata["oof_manifest_sha256"],
                "frozen_holdout_sha256": metadata["frozen_holdout_sha256"],
                "oof_observations": metadata["oof_observations"],
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
