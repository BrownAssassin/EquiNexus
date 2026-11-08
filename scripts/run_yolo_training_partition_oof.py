# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from filament_baseline.data import find_dataset_root, load_dataset
from filament_baseline.runner import _source_fingerprint
from filament_baseline.split import (
    create_observation_split,
    load_observation_split_manifest,
)
from run_yolo_detector_experiment import run as run_yolo_detector_experiment


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_fold_manifest(
    path: Path,
    dataset,
    assignments: dict[str, int],
    validation_fold: int,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("file_name", "year", "site", "partition"),
            lineterminator="\n",
        )
        writer.writeheader()
        for name in sorted(assignments):
            view = dataset.views_by_observation[name][0]
            writer.writerow(
                {
                    "file_name": name,
                    "year": view.year,
                    "site": view.site,
                    "partition": (
                        "validation" if assignments[name] == validation_fold else "train"
                    ),
                }
            )
    return _sha256(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.expanduser().resolve()
    kaggle_runtime = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path(
        "/kaggle/input"
    ).is_dir()
    if args.data_root is None and kaggle_runtime:
        data_root = find_dataset_root(Path("/kaggle/input"))
    else:
        data_root = (
            args.data_root or repo_root / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026"
        ).expanduser().resolve()
    manifest_path = (
        args.oof_manifest.expanduser().resolve()
        if args.oof_manifest is not None
        else repo_root / "data" / "splits" / "observation_oof_seed42_5fold.csv"
    )
    if not kaggle_runtime and not data_root.is_relative_to(repo_root / "data"):
        raise ValueError(f"Competition data must remain under data/: {data_root}")
    if not manifest_path.is_relative_to(repo_root / "data" / "splits"):
        raise ValueError(f"OOF split manifest must remain under data/splits/: {manifest_path}")
    if args.folds < 2:
        raise ValueError("At least two OOF folds are required")

    dataset = load_dataset(data_root)
    frozen_config = json.loads(
        (repo_root / "configs" / "validation-baseline.json").read_text(encoding="utf-8")
    )
    frozen_split = create_observation_split(
        dataset,
        repo_root / "data" / "splits",
        seed=int(frozen_config["seed"]),
        validation_fraction=float(frozen_config["validation_fraction"]),
    )
    if frozen_split.sha256 != frozen_config["expected_split_sha256"]:
        raise RuntimeError("Frozen holdout digest changed before detector OOF training")

    with manifest_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["file_name", "year", "site", "oof_fold"]:
            raise ValueError("OOF manifest must contain file_name, year, site, oof_fold")
        oof_rows = list(reader)
    oof_names = {row["file_name"] for row in oof_rows}
    if oof_names != set(frozen_split.training_names):
        raise ValueError("OOF manifest must cover exactly the frozen training partition")
    assignments = {row["file_name"]: int(row["oof_fold"]) for row in oof_rows}
    if set(assignments.values()) != set(range(args.folds)):
        raise ValueError("OOF manifest fold ids do not match the requested fold count")
    for row in oof_rows:
        view = dataset.views_by_observation[row["file_name"]][0]
        if int(row["year"]) != view.year or row["site"] != view.site:
            raise ValueError(f"OOF metadata differs from the original observation: {row['file_name']}")

    run_root = repo_root / "artifacts" / "detector-oof" / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)
    source_commit = os.environ.get("EQUINEXUS_SOURCE_COMMIT")
    if not source_commit:
        try:
            source_commit = subprocess.run(
                ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except subprocess.CalledProcessError:
            source_commit = "unavailable"
    source_tree_sha256 = _source_fingerprint(repo_root)
    os.environ["EQUINEXUS_SOURCE_COMMIT"] = source_commit
    os.environ["EQUINEXUS_SOURCE_BUNDLE_SHA256"] = source_tree_sha256
    fold_records: list[dict[str, Any]] = []
    all_predictions: dict[str, list[dict[str, Any]]] = {}
    for fold in range(args.folds):
        fold_name = f"{args.run_name}-fold-{fold}"
        train_names = {name for name, assigned in assignments.items() if assigned != fold}
        validation_names = {name for name, assigned in assignments.items() if assigned == fold}
        if train_names & validation_names or train_names | validation_names != oof_names:
            raise AssertionError(f"Observation leakage or coverage error in OOF fold {fold}")
        fold_manifest_path = repo_root / "data" / "splits" / f"{fold_name}.csv"
        fold_manifest_sha256 = _write_fold_manifest(
            fold_manifest_path, dataset, assignments, fold
        )
        checked_split = load_observation_split_manifest(
            dataset, fold_manifest_path, allow_subset=True
        )
        if set(checked_split.training_names) != train_names or set(
            checked_split.validation_names
        ) != validation_names:
            raise AssertionError(f"Fold manifest failed round-trip validation for fold {fold}")

        fold_run_root = run_root / f"fold-{fold}"
        fold_args = SimpleNamespace(
            repo_root=repo_root,
            data_root=data_root,
            run_root=fold_run_root,
            split_manifest=fold_manifest_path,
            run_name=fold_name,
            model=args.model,
            image_size=args.image_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            workers=args.workers,
            patience=args.patience,
            device=args.device,
            fraction=args.fraction,
            mask_ratio=args.mask_ratio,
            prediction_confidence=args.prediction_confidence,
            nms_iou=args.nms_iou,
            max_detections=args.max_detections,
        )
        print(
            f"Starting YOLO OOF fold {fold + 1}/{args.folds}: "
            f"{len(train_names)} train observations, "
            f"{len(validation_names)} validation observations; "
            f"split {fold_manifest_sha256}",
            flush=True,
        )
        metadata = run_yolo_detector_experiment(fold_args)
        fold_predictions_path = fold_run_root / "validation_detections.json"
        predictions = json.loads(fold_predictions_path.read_text(encoding="utf-8"))
        expected_stems = {Path(name).stem for name in validation_names}
        if set(predictions) != expected_stems:
            raise ValueError(f"YOLO OOF predictions do not cover fold {fold} exactly")
        overlap = set(all_predictions) & set(predictions)
        if overlap:
            raise ValueError(f"Duplicate OOF predictions across folds: {sorted(overlap)[:3]}")
        all_predictions.update(predictions)
        fold_records.append(
            {
                "fold": fold,
                "training_observations": len(train_names),
                "validation_observations": len(validation_names),
                "split_manifest": str(fold_manifest_path.relative_to(repo_root)),
                "split_manifest_sha256": fold_manifest_sha256,
                "detector_run_root": str(fold_run_root.relative_to(repo_root)),
                "predictions": str(fold_predictions_path.relative_to(repo_root)),
                "best_checkpoint_sha256": metadata["best_checkpoint_sha256"],
                "training_seconds": metadata["training_seconds"],
                "prediction_seconds": metadata["prediction_seconds"],
                "peak_gpu_memory_bytes": metadata["peak_gpu_memory_bytes"],
            }
        )
        print(
            f"Completed YOLO OOF fold {fold + 1}/{args.folds}: "
            f"{len(predictions)} observations, "
            f"{sum(map(len, predictions.values()))} instances",
            flush=True,
        )

    expected_stems = {Path(name).stem for name in oof_names}
    if set(all_predictions) != expected_stems:
        raise AssertionError("Detector OOF predictions do not cover all 565 training observations")
    predictions_path = run_root / "oof_detections.json"
    predictions_path.write_text(
        json.dumps(all_predictions, separators=(",", ":")), encoding="utf-8"
    )
    result = {
        "status": "complete",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_name": args.run_name,
        "source_commit": source_commit,
        "source_tree_sha256": source_tree_sha256,
        "oof_manifest": str(manifest_path.relative_to(repo_root)),
        "oof_manifest_sha256": _sha256(manifest_path),
        "frozen_holdout_sha256": frozen_split.sha256,
        "frozen_holdout_used_for_training_or_selection": False,
        "oof_observations": len(oof_names),
        "fold_count": args.folds,
        "training_data": os.environ.get(
            "EQUINEXUS_TRAINING_DATA_DESCRIPTION",
            "competition-provided training annotations only",
        ),
        "external_supervised_data": os.environ.get(
            "EQUINEXUS_EXTERNAL_SUPERVISED_DATA", "false"
        ).lower() == "true",
        "pretrained_model": {
            "name": Path(args.model).name,
            "source": os.environ.get(
                "EQUINEXUS_PRETRAINED_SOURCE",
                "Ultralytics YOLO11-seg COCO pretrained weights",
            ),
            "license": os.environ.get("EQUINEXUS_PRETRAINED_LICENSE", "AGPL-3.0"),
            "checkpoint_sha256": (
                _sha256(Path(args.model).expanduser().resolve())
                if Path(args.model).expanduser().is_file()
                else None
            ),
        },
        "configuration": {
            "image_size": args.image_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "device": args.device,
            "workers": args.workers,
            "patience": args.patience,
            "mask_ratio": args.mask_ratio,
        },
        "prediction_observations": len(all_predictions),
        "prediction_instances": sum(map(len, all_predictions.values())),
        "predictions_file": str(predictions_path.relative_to(repo_root)),
        "folds": fold_records,
    }
    (run_root / "run-metadata.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: result[key] for key in (
        "run_name", "oof_manifest_sha256", "frozen_holdout_sha256",
        "oof_observations", "prediction_instances", "fold_count",
    )}, indent=2, sort_keys=True), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train grouped YOLO detectors on five folds of the frozen training partition."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--oof-manifest", type=Path, default=None)
    parser.add_argument("--run-name", default="yolo11s-seg-1024-40ep-oof")
    parser.add_argument("--folds", type=int, default=5)
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
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
