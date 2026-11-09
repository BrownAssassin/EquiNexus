# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from filament_baseline.data import load_dataset  # noqa: E402
from filament_baseline.inference import predict_probabilities  # noqa: E402
from filament_baseline.model_runner import _git_status, _load_model  # noqa: E402
from filament_baseline.predictor import read_grayscale  # noqa: E402
from filament_baseline.runtime import hardware_info, package_versions  # noqa: E402
from filament_baseline.split import create_observation_split  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Average saved grouped-OOF U-Net fold predictions on the "
            "frozen holdout without training on any holdout observation."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()

    repo_root = args.repo_root.expanduser().resolve()
    data_root = (
        args.data_root.expanduser().resolve()
        if args.data_root
        else repo_root / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026"
    )
    source_run_root = args.source_run_root.expanduser().resolve()
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for label, path in (
        ("data", data_root),
        ("source run", source_run_root),
        ("checkpoint root", checkpoint_root),
        ("output", output_dir),
    ):
        if not path.is_relative_to(repo_root):
            raise ValueError(f"{label} must stay inside the repository: {path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    source_metrics_path = source_run_root / "metrics.json"
    source_metrics = json.loads(source_metrics_path.read_text(encoding="utf-8"))
    method = source_metrics.get("method", {})
    target_mode = method.get("target_mode")
    patch_size = method.get("patch_size")
    tta = method.get("tta")
    split_metadata = source_metrics.get("splits", {})
    if (
        target_mode not in {"consensus", "annotator"}
        or not isinstance(patch_size, int)
        or not isinstance(tta, int)
        or method.get("external_supervised_weights") is not False
    ):
        raise ValueError("Source run is not a supported scratch grouped-OOF U-Net model")
    if split_metadata.get("fold_count") != 5:
        raise ValueError("Source run must contain exactly five grouped OOF folds")

    config = json.loads(
        (repo_root / "configs" / "validation-baseline.json").read_text(encoding="utf-8")
    )
    dataset = load_dataset(data_root)
    split = create_observation_split(
        dataset,
        repo_root / "data" / "splits",
        seed=int(config["seed"]),
        validation_fraction=float(config["validation_fraction"]),
    )
    expected_split = config["expected_split_sha256"]
    if split.sha256 != expected_split or split_metadata.get("frozen_holdout_sha256") != expected_split:
        raise RuntimeError("The frozen holdout digest differs from the configured split")
    if len(split.validation_names) != 142:
        raise RuntimeError("Expected exactly 142 frozen holdout observations")

    oof_manifest_path = repo_root / "data" / "splits" / "observation_oof_seed42_5fold.csv"
    if _sha256(oof_manifest_path) != split_metadata.get("oof_manifest_sha256"):
        raise RuntimeError("Grouped OOF manifest digest differs from the source run")
    with oof_manifest_path.open("r", encoding="utf-8", newline="") as stream:
        oof_rows = list(csv.DictReader(stream))
    oof_names = {row["file_name"] for row in oof_rows}
    if len(oof_rows) != 565 or len(oof_names) != 565 or oof_names != set(split.training_names):
        raise RuntimeError("OOF manifest does not cover the frozen training partition exactly once")
    fold_counts = {
        fold: sum(int(row["oof_fold"]) == fold for row in oof_rows)
        for fold in range(5)
    }
    if fold_counts != {fold: 113 for fold in range(5)}:
        raise RuntimeError(f"Unexpected grouped OOF fold counts: {fold_counts}")

    fold_metadata = source_metrics.get("folds", [])
    if {int(item["fold"]) for item in fold_metadata} != set(range(5)):
        raise RuntimeError("Source run is missing one or more fold records")
    checkpoint_paths = [checkpoint_root / f"fold-{fold}" / "best_model.pt" for fold in range(5)]
    checkpoint_hashes = []
    for fold, checkpoint_path in enumerate(checkpoint_paths):
        fold_record = next(item for item in fold_metadata if int(item["fold"]) == fold)
        if (
            int(fold_record["training_observations"]) != 452
            or int(fold_record["validation_observations"]) != 113
        ):
            raise RuntimeError(f"Fold {fold} does not match the recorded grouped training split")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint_hashes.append(_sha256(checkpoint_path))

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda:0" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    if device.type == "cuda":
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(device)
    peak_gpu_memory_bytes = 0

    output_dir.mkdir(parents=True, exist_ok=True)
    probability_dir = output_dir / "probabilities"
    probability_dir.mkdir(parents=True, exist_ok=True)
    validation_names = sorted(split.validation_names)
    probability_hashes: dict[str, str] = {}
    summed_foreground = {
        Path(name).stem: np.zeros((2048, 2048), dtype=np.float32)
        for name in validation_names
    }
    summed_boundary = {stem: np.zeros_like(value) for stem, value in summed_foreground.items()}
    inference_started = time.perf_counter()
    for fold, checkpoint_path in enumerate(checkpoint_paths):
        print(f"Loading {target_mode} fold {fold + 1}/5", flush=True)
        model = _load_model(checkpoint_path, device)
        fold_started = time.perf_counter()
        for index, file_name in enumerate(validation_names, start=1):
            image = read_grayscale(dataset.train_image_dir / file_name)
            foreground, boundary = predict_probabilities(
                model,
                image,
                device=device,
                patch_size=patch_size,
                stride=int(patch_size * 0.75),
                tta=tta,
            )
            stem = Path(file_name).stem
            summed_foreground[stem] += foreground
            summed_boundary[stem] += boundary
            if index % 20 == 0 or index == len(validation_names):
                print(f"fold-{fold} holdout: {index}/{len(validation_names)}", flush=True)
        print(
            f"Completed fold {fold} in {time.perf_counter() - fold_started:.1f}s",
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_gpu_memory_bytes = max(
                peak_gpu_memory_bytes,
                int(torch.cuda.max_memory_allocated(device)),
            )
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for file_name in validation_names:
        stem = Path(file_name).stem
        path = probability_dir / f"{stem}.npz"
        np.savez_compressed(
            path,
            foreground=(summed_foreground[stem] / 5.0).astype(np.float16),
            boundary=(summed_boundary[stem] / 5.0).astype(np.float16),
        )
        probability_hashes[path.name] = _sha256(path)

    probability_manifest_digest = hashlib.sha256(
        "".join(f"{name},{probability_hashes[name]}\n" for name in sorted(probability_hashes)).encode("utf-8")
    ).hexdigest()
    result: dict[str, Any] = {
        "status": "complete",
        "run_name": output_dir.name,
        "method": {
            "name": f"five-fold mean of scratch {target_mode} U-Net probabilities",
            "target_mode": target_mode,
            "patch_size": patch_size,
            "stride": int(patch_size * 0.75),
            "tta": tta,
            "external_supervised_weights": False,
        },
        "data_boundary": {
            "training_labels": "competition-provided training annotations only",
            "test_observations_used": False,
            "test_labels_used": False,
            "frozen_holdout_used_for_training": False,
        },
        "source_run": {
            "run_name": source_metrics.get("run_name"),
            "metrics_path": str(source_metrics_path.relative_to(repo_root)),
            "metrics_sha256": _sha256(source_metrics_path),
            "oof_manifest_sha256": split_metadata["oof_manifest_sha256"],
            "oof_manifest_rows": len(oof_rows),
            "fold_training_observations": 452,
            "fold_validation_observations": 113,
        },
        "source_checkpoints": [
            {
                "fold": fold,
                "path": str(path.relative_to(repo_root)),
                "sha256": checkpoint_hashes[fold],
                "training_observations": 452,
                "validation_observations": 113,
            }
            for fold, path in enumerate(checkpoint_paths)
        ],
        "dataset_inventory": {
            "frozen_holdout_observations": len(validation_names),
            "annotator_views": sum(len(dataset.views_by_observation[name]) for name in validation_names),
        },
        "split": {
            "frozen_holdout_sha256": split.sha256,
            "oof_manifest_sha256": split_metadata["oof_manifest_sha256"],
            "holdout_observation_count": len(validation_names),
            "validation_is_excluded_from_every_fold_training_set": True,
        },
        "artifacts": {
            "probability_dir": str(probability_dir.relative_to(repo_root)),
            "probability_map_count": len(probability_hashes),
            "probability_manifest_sha256": probability_manifest_digest,
            "probability_sha256": probability_hashes,
        },
        "repository": _git_status(repo_root),
        "runtime": {
            **hardware_info(),
            "package_versions": package_versions(),
            "device": str(device),
            "inference_seconds": time.perf_counter() - inference_started,
            "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "split": result["split"], "runtime": result["runtime"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
