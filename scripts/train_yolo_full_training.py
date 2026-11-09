# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Train a final YOLO-seg detector on every competition training observation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from export_yolo_dataset import export_yolo_dataset
from filament_baseline.data import inventory, load_dataset
from filament_baseline.runtime import hardware_info, package_versions
from filament_baseline.split import ObservationSplit


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_all_training_manifest(dataset, path: Path) -> tuple[ObservationSplit, str]:
    names = tuple(sorted(dataset.observation_names))
    rows = []
    assignments = {name: "train" for name in names}
    for name in names:
        view = dataset.views_by_observation[name][0]
        rows.append(
            {
                "file_name": name,
                "year": view.year,
                "site": view.site,
                "partition": "train",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("file_name", "year", "site", "partition"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    digest = _sha256(path)
    return ObservationSplit(assignments, path, digest), digest


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.expanduser().resolve()
    data_root = (args.data_root or repo_root / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026").expanduser().resolve()
    run_root = (args.run_root or repo_root / "artifacts" / "detector-experiments" / args.run_name).expanduser().resolve()
    export_root = repo_root / "data" / "processed" / "yolo" / args.run_name
    pretrained_path = args.model.expanduser().resolve()
    for label, path, allowed_root in (
        ("competition data", data_root, repo_root / "data"),
        ("detector output", run_root, repo_root / "artifacts" / "detector-experiments"),
        ("YOLO dataset export", export_root, repo_root / "data" / "processed"),
        ("pretrained checkpoint", pretrained_path, repo_root / "artifacts"),
    ):
        if not path.is_relative_to(allowed_root.resolve()):
            raise ValueError(f"{label} must remain under {allowed_root}: {path}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", args.run_name):
        raise ValueError("run-name must use only letters, digits, dot, underscore, or dash")
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {run_root}")
    if export_root.exists() and any(export_root.iterdir()):
        raise FileExistsError(f"YOLO export directory is not empty: {export_root}")
    if not pretrained_path.is_file():
        raise FileNotFoundError(pretrained_path)

    config_path = repo_root / "configs" / "validation-baseline.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset = load_dataset(data_root)
    data_inventory = inventory(dataset)
    expected = {
        "observation_count": 707,
        "annotator_view_count": 1154,
        "annotation_count": 8199,
        "train_image_file_count": 707,
        "test_image_file_count": 180,
    }
    for key, value in expected.items():
        if data_inventory[key] != value:
            raise ValueError(f"Expected {value} {key}, found {data_inventory[key]}")

    oof_manifest = repo_root / "data" / "splits" / "observation_oof_seed42_5fold.csv"
    if not oof_manifest.is_file():
        raise FileNotFoundError(oof_manifest)
    train_manifest = repo_root / "data" / "splits" / "observation_train_all_seed42.csv"
    split, train_manifest_sha256 = _write_all_training_manifest(dataset, train_manifest)
    if len(split.training_names) != 707 or split.validation_names:
        raise AssertionError("Final fit must train on all 707 groups and have no validation partition")
    export_manifest = export_yolo_dataset(dataset, split, export_root)
    export_manifest.pop("views", None)
    for partition in ("train", "val"):
        (export_root / "images" / partition).mkdir(parents=True, exist_ok=True)
        (export_root / "labels" / partition).mkdir(parents=True, exist_ok=True)
    data_yaml_path = export_root / "data.yaml"
    data_yaml = data_yaml_path.read_text(encoding="utf-8")
    if "val: images/val\n" not in data_yaml:
        raise ValueError("Unexpected exported YOLO dataset configuration")
    # Ultralytics constructs a validation loader even when val=False. Point it
    # at the training set for initialization only; val=False disables scoring.
    data_yaml_path.write_text(data_yaml.replace("val: images/val\n", "val: images/train\n"), encoding="utf-8")
    export_manifest_path = run_root / "export_manifest.json"
    run_root.mkdir(parents=True, exist_ok=True)
    export_manifest_path.write_text(
        json.dumps(export_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    os.environ["YOLO_CONFIG_DIR"] = str(run_root / "ultralytics-config")
    os.environ["TORCH_HOME"] = str(repo_root / "artifacts" / "pretrained" / "torch" / "hub")
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TORCH_HOME"]).mkdir(parents=True, exist_ok=True)

    import torch
    import ultralytics
    from ultralytics import YOLO

    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select --device cpu only for a smoke run")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model = YOLO(str(pretrained_path))
    run_metadata_path = run_root / "run-metadata.json"
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    metadata: dict[str, Any] = {
        "status": "running",
        "created_utc": _utc_now(),
        "run_name": args.run_name,
        "source_commit": source_commit,
        "source_script_sha256": _sha256(Path(__file__).resolve()),
        "pretrained_checkpoint": str(pretrained_path.relative_to(repo_root)),
        "pretrained_checkpoint_sha256": _sha256(pretrained_path),
        "training_data": "competition-provided training annotations only",
        "external_supervised_data": False,
        "test_labels_used": False,
        "test_images_used_for_training": False,
        "dataset_inventory": data_inventory,
        "training_observations": len(split.training_names),
        "training_annotator_views": data_inventory["annotator_view_count"],
        "training_annotations": data_inventory["annotation_count"],
        "final_training_manifest": str(train_manifest.relative_to(repo_root)),
        "final_training_manifest_sha256": train_manifest_sha256,
        "export_manifest": str(export_manifest_path.relative_to(repo_root)),
        "export_manifest_sha256": _sha256(export_manifest_path),
        "holdout_excluded": False,
        "holdout_used_for_final_training": True,
        "frozen_holdout_sha256": config["expected_split_sha256"],
        "oof_manifest_sha256": _sha256(oof_manifest),
        "configuration": {
            "model": str(pretrained_path.relative_to(repo_root)),
            "image_size": args.image_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "device": args.device,
            "seed": int(config["seed"]),
            "mask_ratio": args.mask_ratio,
            "single_class": True,
            "validation_during_final_fit": "disabled during epochs; Ultralytics still runs final-epoch diagnostics on the training images because its trainer always validates the final epoch",
            "validation_loader_source": "all training images; no independent validation data is used",
            "checkpoint_selection": "last epoch; final-epoch training-set diagnostics are ignored for selection",
            "augmentation": "flips, mild translation/scale/brightness; mosaic/mixup disabled",
        },
        "hardware": hardware_info(),
        "package_versions": {**package_versions(), "ultralytics": ultralytics.__version__},
    }
    run_metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    started = time.perf_counter()
    try:
        result = model.train(
            data=str(data_yaml_path),
            imgsz=args.image_size,
            epochs=args.epochs,
            batch=args.batch_size,
            device=args.device,
            workers=args.workers,
            patience=0,
            seed=int(config["seed"]),
            deterministic=False,
            single_cls=True,
            cos_lr=True,
            plots=False,
            fraction=1.0,
            val=False,
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
            name="yolo-seg-final",
            exist_ok=True,
            save=True,
            save_period=max(1, args.epochs // 10),
            amp=True,
            cache=False,
            verbose=True,
        )
    except Exception as error:
        metadata.update(
            {
                "status": "failed",
                "failed_utc": _utc_now(),
                "failure_reason": f"{type(error).__name__}: {error}",
                "elapsed_before_failure_seconds": time.perf_counter() - started,
            }
        )
        run_metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise
    training_seconds = time.perf_counter() - started
    train_dir = run_root / "train" / "yolo-seg-final"
    last_path = train_dir / "weights" / "last.pt"
    if not last_path.is_file():
        raise FileNotFoundError(f"Final YOLO training did not produce {last_path}")
    final_path = run_root / "final.pt"
    final_path.write_bytes(last_path.read_bytes())

    metadata.update(
        {
            "status": "complete",
            "completed_utc": _utc_now(),
            "checkpoint": str(final_path.relative_to(repo_root)),
            "checkpoint_sha256": _sha256(final_path),
            "last_checkpoint_sha256": _sha256(last_path),
            "training_seconds": training_seconds,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
            "training_results_dir": str(train_dir.relative_to(repo_root)),
            "trainer_result": str(getattr(result, "save_dir", train_dir)),
        }
    )
    run_metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": metadata["status"],
                "training_observations": metadata["training_observations"],
                "training_annotator_views": metadata["training_annotator_views"],
                "training_seconds": metadata["training_seconds"],
                "peak_gpu_memory_bytes": metadata["peak_gpu_memory_bytes"],
                "checkpoint_sha256": metadata["checkpoint_sha256"],
                "run_root": str(run_root),
            },
            indent=2,
        ),
        flush=True,
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--run-name", default="yolo11s-seg-1280-final-fulltrain-20261108")
    parser.add_argument("--model", type=Path, default=REPO_ROOT / "artifacts" / "yolo11s-seg.pt")
    parser.add_argument("--image-size", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mask-ratio", type=int, default=2)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
