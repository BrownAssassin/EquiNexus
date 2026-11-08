# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import inventory, load_dataset
from .inference import InstanceConfig, instances_from_probabilities, predict_probabilities
from .model_runner import _git_status, _is_kaggle_runtime, _load_model, _resolve_data_root, _within
from .predictor import PredictedInstance, read_grayscale
from .runner import _write_observation_scores, _write_overlays, _write_predictions
from .runtime import hardware_info, package_versions
from .scorer import score_validation
from .split import create_observation_split
from .visuals import write_comparison_overlays


def reinfer_validation(
    *,
    repo_root: str | Path,
    data_root: str | Path | None,
    model_run_root: str | Path,
    tta: int,
    overlay_count: int = 12,
) -> dict[str, Any]:
    """Recompute holdout probability maps from a saved checkpoint with another TTA count."""
    repo_root = Path(repo_root).resolve()
    model_run_root = Path(model_run_root).resolve()
    kaggle_runtime = _is_kaggle_runtime()
    if not kaggle_runtime and not _within(model_run_root, repo_root):
        raise ValueError(f"Local model artifacts must stay inside the repository: {model_run_root}")
    if tta not in (1, 2, 4, 8):
        raise ValueError("tta must be 1, 2, 4, or 8")

    dataset_root = _resolve_data_root(data_root, repo_root)
    if not kaggle_runtime and not _within(dataset_root, repo_root):
        raise ValueError(f"Local competition data must stay inside the repository: {dataset_root}")
    model_metrics_path = model_run_root / "metrics.json"
    checkpoint_path = model_run_root / "best_model.pt"
    if not model_metrics_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("model_run_root must contain metrics.json and best_model.pt")
    model_metrics = json.loads(model_metrics_path.read_text(encoding="utf-8"))
    model_configuration = model_metrics.get("method", {})
    patch_size = int(model_configuration["patch_size"])
    dataset = load_dataset(dataset_root)
    data_inventory = inventory(dataset)
    config_path = repo_root / "configs" / "validation-baseline.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    split = create_observation_split(
        dataset,
        repo_root / "data" / "splits",
        seed=int(config["seed"]),
        validation_fraction=float(config["validation_fraction"]),
    )
    if split.sha256 != config["expected_split_sha256"]:
        raise RuntimeError("The frozen validation split digest changed")
    validation_names = list(split.validation_names)
    if len(validation_names) != 142:
        raise RuntimeError(f"Expected all 142 holdout observations, found {len(validation_names)}")

    output_root = model_run_root / f"tta-{tta}"
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"TTA output already exists; choose or clear a new run directory: {output_root}")
    probability_dir = output_root / "probabilities"
    probability_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = _load_model(checkpoint_path, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    predictions: dict[str, list[PredictedInstance]] = {}
    started = time.perf_counter()
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
        np.savez_compressed(
            probability_dir / f"{Path(file_name).stem}.npz",
            foreground=foreground.astype(np.float16),
            boundary=boundary.astype(np.float16),
        )
        predictions[file_name] = instances_from_probabilities(
            foreground,
            boundary,
            file_name,
            InstanceConfig(),
        )
        if index % 20 == 0 or index == len(validation_names):
            print(f"Re-inferred {index}/{len(validation_names)} observations with TTA={tta}", flush=True)
    inference_seconds = time.perf_counter() - inference_started
    peak_gpu_memory_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
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
    _write_predictions(output_root / "predictions.csv", predictions)
    _write_observation_scores(output_root / "per_observation.csv", observation_rows, predictions)
    _write_overlays(dataset, predictions, output_root / "overlays", overlay_count)
    comparison_overlays = write_comparison_overlays(
        dataset, predictions, output_root / "comparison-overlays", overlay_count
    )

    repository_info = _git_status(repo_root)
    result = {
        "status": "complete",
        "run_name": f"{model_run_root.name}-tta-{tta}",
        "parent_model_run": str(model_run_root.relative_to(repo_root))
        if _within(model_run_root, repo_root)
        else str(model_run_root),
        "parent_model_configuration": model_configuration,
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "tta": tta,
        "patch_size": patch_size,
        "default_instance_configuration": InstanceConfig().__dict__,
        "repository": repository_info,
        "configuration_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "dataset_inventory": data_inventory,
        "split": {
            "sha256": split.sha256,
            "training_observations": len(split.training_names),
            "validation_observations": len(validation_names),
        },
        "metrics": metrics,
        "runtime": {
            **hardware_info(),
            "package_versions": package_versions(),
            "device": str(device),
            "inference_seconds": inference_seconds,
            "scoring_seconds": scoring_seconds,
            "end_to_end_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
            "comparison_overlays": comparison_overlays,
        },
    }
    (output_root / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"run_name": result["run_name"], "metrics": metrics["global"], "split_sha256": split.sha256}, indent=2))
    return result
