# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from filament_baseline.data import load_dataset
from filament_baseline.external_data import load_swefil_dataset
from filament_baseline.runtime import hardware_info, package_versions
from filament_baseline.split import ObservationSplit, create_observation_split
from export_yolo_dataset import export_yolo_dataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _date_group_split(
    dataset,
    manifest_path: Path,
    *,
    seed: int,
    validation_fraction: float,
) -> ObservationSplit:
    days = {name: Path(name).stem[:8] for name in dataset.observation_names}
    unique_days = sorted(set(days.values()))
    validation_days = {
        day
        for day in unique_days
        if int.from_bytes(
            hashlib.sha256(f"swefil-yolo-date-split-{seed}:{day}".encode()).digest()[:8],
            "big",
        )
        / 2**64
        < validation_fraction
    }
    if not validation_days or validation_days == set(unique_days):
        validation_days = set(unique_days[::5])
    assignments = {
        name: ("validation" if days[name] in validation_days else "train")
        for name in dataset.observation_names
    }
    if set(assignments.values()) != {"train", "validation"}:
        raise ValueError("External date split must have nonempty train and validation groups")
    train_days = {days[name] for name, value in assignments.items() if value == "train"}
    val_days = {days[name] for name, value in assignments.items() if value == "validation"}
    if train_days & val_days:
        raise AssertionError("SWEFil observations from one calendar date crossed partitions")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
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
                    "partition": assignments[name],
                }
            )
    return ObservationSplit(
        assignments=assignments,
        manifest_path=manifest_path,
        sha256=_sha256(manifest_path),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.expanduser().resolve()
    data_root = (args.data_root or repo_root / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026").expanduser().resolve()
    swefil_root = (args.swefil_root or repo_root / "data" / "external" / "SWEFil").expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    run_root = (repo_root / "artifacts" / "detector-experiments" / args.run_name).resolve()
    export_root = (repo_root / "data" / "processed" / "yolo" / f"{args.run_name}-external-data").resolve()
    for label, path, parent in (
        ("initial model", model_path, repo_root),
        ("run artifacts", run_root, repo_root / "artifacts"),
        ("processed YOLO data", export_root, repo_root / "data" / "processed"),
    ):
        if not path.is_relative_to(parent):
            raise ValueError(f"{label} must remain under {parent}: {path}")
    if not data_root.is_dir():
        raise FileNotFoundError(f"Competition input directory does not exist: {data_root}")
    kaggle_input_root = Path("/kaggle/input").resolve()
    if not swefil_root.is_relative_to(repo_root / "data" / "external") and not swefil_root.is_relative_to(kaggle_input_root):
        raise ValueError(
            "SWEFil must be inside the workspace data/external folder or a read-only Kaggle input: "
            f"{swefil_root}"
        )
    if not swefil_root.is_dir():
        raise FileNotFoundError(f"SWEFil input directory does not exist: {swefil_root}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Initial model checkpoint is missing: {model_path}")
    run_root.mkdir(parents=True, exist_ok=True)
    config = json.loads((repo_root / "configs" / "validation-baseline.json").read_text())
    competition = load_dataset(data_root)
    frozen = create_observation_split(
        competition,
        repo_root / "data" / "splits",
        seed=int(config["seed"]),
        validation_fraction=float(config["validation_fraction"]),
    )
    if frozen.sha256 != config["expected_split_sha256"]:
        raise RuntimeError("Frozen holdout digest changed before external pretraining")
    external, provenance = load_swefil_dataset(
        swefil_root,
        competition,
        frozen,
        repository_root=repo_root,
        exclude_all_competition_dates=True,
        allow_readonly_external_root=swefil_root.is_relative_to(kaggle_input_root),
    )
    if provenance["included_observations"] != 229:
        raise ValueError(
            "Expected the audited 229-image, all-competition-date-disjoint SWEFil sample; "
            f"found {provenance['included_observations']}"
        )
    audit_path = swefil_root / "equinexus_audit_all_competition_dates_excluded.json"
    if not audit_path.is_file():
        raise FileNotFoundError(f"Strict SWEFil audit is missing: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("included_manifest_sha256") != provenance["included_manifest_sha256"]:
        raise ValueError("SWEFil strict date-disjoint audit does not match the training sample")

    split = _date_group_split(
        external,
        repo_root / "data" / "splits" / f"{args.run_name}-date-grouped.csv",
        seed=int(config["seed"]),
        validation_fraction=0.2,
    )
    export_manifest = export_yolo_dataset(external, split, export_root)
    export_manifest.pop("views", None)
    (run_root / "external-export-manifest.json").write_text(
        json.dumps(export_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    provenance.update(
        {
            "competition_holdout_sha256": frozen.sha256,
            "oof_safe_date_exclusion": True,
            "external_train_validation_split": {
                "manifest": str(split.manifest_path.relative_to(repo_root)),
                "sha256": split.sha256,
                "group_key": "UTC calendar date",
                "train_observations": len(split.training_names),
                "validation_observations": len(split.validation_names),
                "validation_days": len(
                    {Path(name).stem[:8] for name in split.validation_names}
                ),
            },
            "export": export_manifest,
        }
    )
    (run_root / "external-data-provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    model_cache = run_root / "ultralytics-config"
    model_cache.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(model_cache)
    os.environ["TORCH_HOME"] = str(run_root / "torch-home")

    import torch
    import ultralytics
    from ultralytics import YOLO

    if not torch.cuda.is_available() and args.device != "cpu":
        raise RuntimeError("CUDA is unavailable; use --device cpu only for a smoke run")
    random.seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config["seed"]))
        torch.cuda.reset_peak_memory_stats()
    model = YOLO(str(model_path))
    started = time.perf_counter()
    model.train(
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
        mask_ratio=2,
        project=str(run_root / "train"),
        name="yolo-seg",
        exist_ok=True,
        save=True,
        save_period=max(1, args.epochs // 10),
        amp=True,
        cache=False,
        verbose=True,
    )
    elapsed = time.perf_counter() - started
    best = run_root / "train" / "yolo-seg" / "weights" / "best.pt"
    last = run_root / "train" / "yolo-seg" / "weights" / "last.pt"
    if not best.is_file():
        raise FileNotFoundError(f"YOLO did not save the best checkpoint: {best}")
    import shutil

    shutil.copy2(best, run_root / "best.pt")
    if last.is_file():
        shutil.copy2(last, run_root / "last.pt")
    source_commit = os.environ.get("EQUINEXUS_SOURCE_COMMIT")
    if source_commit is None:
        source_commit = __import__("subprocess").run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    result = {
        "status": "complete",
        "run_name": args.run_name,
        "source_commit": source_commit,
        "source_bundle_sha256": os.environ.get("EQUINEXUS_SOURCE_BUNDLE_SHA256"),
        "frozen_holdout_sha256": frozen.sha256,
        "frozen_holdout_used_for_training_or_selection": False,
        "training_data": "SWEFil CC-BY-4.0 external annotations only",
        "training_observations": len(split.training_names),
        "validation_observations": len(split.validation_names),
        "external_data_provenance": str((run_root / "external-data-provenance.json").relative_to(repo_root)),
        "initial_model": {
            "name": model_path.name,
            "source": "Ultralytics YOLO11-seg COCO pretrained weights",
            "license": "AGPL-3.0",
            "checkpoint_sha256": _sha256(model_path),
        },
        "configuration": {
            "image_size": args.image_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "workers": args.workers,
            "device": args.device,
            "seed": int(config["seed"]),
            "augmentation": "flips, mild translation/scale/brightness; mosaic/mixup disabled",
        },
        "checkpoint": str((run_root / "best.pt").relative_to(repo_root)),
        "best_checkpoint_sha256": _sha256(run_root / "best.pt"),
        "training_seconds": elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        "hardware": hardware_info(),
        "package_versions": {**package_versions(), "ultralytics": ultralytics.__version__},
    }
    (run_root / "run-metadata.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pretrain YOLO11-seg on date-disjoint, separately licensed SWEFil masks."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--swefil-root", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=REPO_ROOT / "weights" / "yolo11s-seg.pt")
    parser.add_argument("--run-name", default="yolo11s-seg-swefil-pretrain")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
