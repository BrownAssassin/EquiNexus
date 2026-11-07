from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import inventory, load_dataset
from .features import InputMode
from .inference import (
    InstanceConfig,
    instances_from_probabilities,
    predict_probabilities,
)
from .learning import FilamentPatchDataset, TargetMode
from .model import FilamentUNet
from .predictor import read_grayscale
from .runner import (
    _git_status,
    _resolve_data_root,
    _write_observation_scores,
    _write_overlays,
    _write_predictions,
)
from .runtime import hardware_info, package_versions
from .scorer import score_validation
from .split import create_observation_split
from .training import train_model
from .visuals import write_comparison_overlays


def _within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _is_kaggle_runtime() -> bool:
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or (
        Path("/kaggle/input").is_dir() and Path("/kaggle/working").is_dir()
    )


def _load_model(checkpoint_path: Path, device: torch.device) -> FilamentUNet:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = FilamentUNet(**checkpoint.get("model_config", {}))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)
    return model.eval()


def _save_probability_cache(
    output_dir: Path,
    observation: str,
    foreground: np.ndarray,
    boundary: np.ndarray,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{Path(observation).stem}.npz"
    np.savez_compressed(
        output_path,
        foreground=foreground.astype(np.float16),
        boundary=boundary.astype(np.float16),
    )
    return output_path


def run_model_experiment(
    *,
    repo_root: str | Path,
    data_root: str | Path | None,
    run_name: str,
    target_mode: TargetMode = "consensus",
    steps: int = 5_000,
    batch_size: int = 2,
    patch_size: int = 512,
    tta: int = 1,
    input_mode: InputMode = "raw",
    foreground_threshold: float = 0.35,
    boundary_threshold: float = 0.5,
    boundary_weight: float = 1.5,
    min_instance_area: int = 50,
    train_observation_limit: int | None = None,
    validation_limit: int | None = None,
    overlay_count: int = 8,
) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", run_name):
        raise ValueError("run_name must use only letters, digits, dot, underscore, or dash")
    repo_root = Path(repo_root).resolve()
    kaggle_runtime = _is_kaggle_runtime()
    dataset_root = _resolve_data_root(data_root, repo_root)
    if not kaggle_runtime and not _within(dataset_root, repo_root):
        raise ValueError(f"Local competition data must stay inside the repository: {dataset_root}")
    run_root = (
        Path("/kaggle/working") / "equinexus" / run_name
        if kaggle_runtime
        else repo_root / "artifacts" / "model-experiments" / run_name
    )
    if not kaggle_runtime and not _within(run_root, repo_root):
        raise ValueError(f"Local experiment outputs must stay in the repository: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    repository_info = _git_status(repo_root)

    config = json.loads((repo_root / "configs" / "validation-baseline.json").read_text())
    config_sha256 = hashlib.sha256(
        (repo_root / "configs" / "validation-baseline.json").read_bytes()
    ).hexdigest()
    dataset = load_dataset(dataset_root)
    dataset_inventory = inventory(dataset)
    expected_inventory = {
        "observation_count": 707,
        "annotator_view_count": 1154,
        "annotation_count": 8199,
    }
    for key, expected in expected_inventory.items():
        if dataset_inventory[key] != expected:
            raise ValueError(f"Expected {expected} {key}, found {dataset_inventory[key]}")

    split = create_observation_split(
        dataset,
        repo_root / "data" / "splits",
        seed=int(config["seed"]),
        validation_fraction=float(config["validation_fraction"]),
    )
    if split.sha256 != config["expected_split_sha256"]:
        raise RuntimeError(
            f"Frozen split digest changed: {split.sha256} != {config['expected_split_sha256']}"
        )
    train_names = list(split.training_names)
    validation_names = list(split.validation_names)
    if train_observation_limit is not None:
        train_names = train_names[:train_observation_limit]
    if validation_limit is not None:
        validation_names = validation_names[:validation_limit]
    if set(train_names) & set(validation_names):
        raise AssertionError("Training and validation observations overlap")

    train_dataset = FilamentPatchDataset(
        dataset,
        train_names,
        target_mode=target_mode,
        patch_size=patch_size,
        augment=True,
        seed=int(config["seed"]),
        input_mode=input_mode,
    )
    validation_dataset = FilamentPatchDataset(
        dataset,
        validation_names,
        target_mode=target_mode,
        patch_size=patch_size,
        augment=False,
        seed=int(config["seed"]),
        input_mode=input_mode,
    )
    checkpoint_path = run_root / "best_model.pt"
    started = time.perf_counter()
    print(
        f"Training {target_mode} U-Net: {len(train_names)} train groups, "
        f"{len(validation_names)} validation groups, {steps} steps, "
        f"patch {patch_size}, batch {batch_size}",
        flush=True,
    )
    training_result = train_model(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        output_path=checkpoint_path,
        steps=steps,
        batch_size=batch_size,
        seed=int(config["seed"]),
        validation_interval=max(100, min(500, steps // 5)),
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = _load_model(checkpoint_path, device)
    cache_dir = run_root / "probabilities"
    probability_cache_paths = {}
    inference_started = time.perf_counter()
    for index, file_name in enumerate(validation_names, start=1):
        image = read_grayscale(dataset.train_image_dir / file_name)
        foreground, boundary = predict_probabilities(
            model,
            image,
            device=device,
            patch_size=patch_size,
            stride=max(patch_size // 2, int(patch_size * 0.75)),
            tta=tta,
        )
        probability_cache_paths[file_name] = str(
            _save_probability_cache(cache_dir, file_name, foreground, boundary)
        )
        if index % 10 == 0 or index == len(validation_names):
            print(f"Predicted {index}/{len(validation_names)} validation observations", flush=True)
    inference_seconds = time.perf_counter() - inference_started

    instance_config = InstanceConfig(
        foreground_threshold=foreground_threshold,
        boundary_threshold=boundary_threshold,
        boundary_weight=boundary_weight,
        min_instance_area=min_instance_area,
    )
    predictions: dict[str, list] = {}
    for file_name, cache_path in probability_cache_paths.items():
        with np.load(cache_path) as cache:
            predictions[file_name] = instances_from_probabilities(
                cache["foreground"].astype(np.float32),
                cache["boundary"].astype(np.float32),
                file_name,
                instance_config,
            )

    scoring_started = time.perf_counter()
    metrics, observation_rows = score_validation(
        dataset,
        predictions,
        validation_names,
        iou_threshold=float(config["iou_threshold"]),
        diagnostic_overlap_threshold=float(config["diagnostic_overlap_threshold"]),
    )
    scoring_seconds = time.perf_counter() - scoring_started
    _write_predictions(run_root / "predictions.csv", predictions)
    _write_observation_scores(run_root / "per_observation.csv", observation_rows, predictions)
    overlays = _write_overlays(dataset, predictions, run_root / "overlays", overlay_count)
    comparison_overlays = write_comparison_overlays(
        dataset, predictions, run_root / "comparison-overlays", overlay_count
    )

    hardware = hardware_info()
    if os.environ.get("EQUINEXUS_SOURCE_COMMIT"):
        repository_info["commit"] = os.environ["EQUINEXUS_SOURCE_COMMIT"]
        repository_info["worktree_dirty"] = True
    repository_info["source_bundle_sha256"] = os.environ.get(
        "EQUINEXUS_SOURCE_BUNDLE_SHA256"
    )
    result = {
        "status": "complete" if validation_limit is None else "validation_subset",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_name": run_name,
        "repository": repository_info,
        "configuration_sha256": config_sha256,
        "kaggle_notebook_version": os.environ.get("EQUINEXUS_NOTEBOOK_VERSION"),
        "dataset_inventory": dataset_inventory,
        "split": {
            "sha256": split.sha256,
            "expected_sha256": config["expected_split_sha256"],
            "training_observations": len(split.training_names),
            "validation_observations": len(split.validation_names),
            "evaluated_validation_observations": len(validation_names),
        },
        "method": {
            "architecture": "from_scratch_five_level_unet_foreground_boundary",
            "base_channels": 16,
            "target_mode": target_mode,
            "patch_size": patch_size,
            "input_mode": input_mode,
            "steps": steps,
            "batch_size": batch_size,
            "seed": int(config["seed"]),
            "tta": tta,
            "instance_config": instance_config.__dict__,
            "external_supervised_weights": False,
            "external_ground_truth": False,
        },
        "training": training_result,
        "metrics": metrics,
        "runtime": {
            **hardware,
            "training_seconds": training_result["elapsed_seconds"],
            "inference_seconds": inference_seconds,
            "scoring_seconds": scoring_seconds,
            "end_to_end_seconds": time.perf_counter() - started,
            "package_versions": package_versions(),
        },
        "artifacts": {
            "checkpoint": checkpoint_path.name,
            "training_history": Path(training_result["history"]).name,
            "probability_cache_dir": "probabilities",
            "predictions": "predictions.csv",
            "per_observation": "per_observation.csv",
            "overlays": overlays,
            "comparison_overlays": comparison_overlays,
        },
    }
    (run_root / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Validation PQ: {metrics['global']['pq']:.6f}; "
        f"split {split.sha256}; outputs {run_root}",
        flush=True,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate a grouped filament U-Net.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--target-mode", choices=("annotator", "consensus"), default="consensus")
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--tta", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument(
        "--input-mode", choices=("raw", "raw_local_contrast"), default="raw"
    )
    parser.add_argument("--foreground-threshold", type=float, default=0.35)
    parser.add_argument("--boundary-threshold", type=float, default=0.5)
    parser.add_argument("--boundary-weight", type=float, default=1.5)
    parser.add_argument("--min-instance-area", type=int, default=50)
    parser.add_argument("--train-observation-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--overlay-count", type=int, default=8)
    args = parser.parse_args(argv)
    result = run_model_experiment(
        repo_root=args.repo_root,
        data_root=args.data_root,
        run_name=args.run_name,
        target_mode=args.target_mode,
        steps=args.steps,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        tta=args.tta,
        input_mode=args.input_mode,
        foreground_threshold=args.foreground_threshold,
        boundary_threshold=args.boundary_threshold,
        boundary_weight=args.boundary_weight,
        min_instance_area=args.min_instance_area,
        train_observation_limit=args.train_observation_limit,
        validation_limit=args.validation_limit,
        overlay_count=args.overlay_count,
    )
    print(json.dumps(result["metrics"]["global"], indent=2, sort_keys=True))
    return 0
