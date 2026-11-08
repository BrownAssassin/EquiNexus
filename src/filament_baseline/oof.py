# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from .data import inventory, load_dataset
from .features import InputMode
from .inference import InstanceConfig, instances_from_probabilities, predict_probabilities
from .learning import FilamentPatchDataset, TargetMode
from .model import DINO_V2_REPOSITORY, model_architecture
from .model_runner import _load_model
from .predictor import PredictedInstance, read_grayscale
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


def _save_probability_cache(
    output_dir: Path,
    file_name: str,
    foreground: np.ndarray,
    boundary: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / f"{Path(file_name).stem}.npz",
        foreground=foreground.astype(np.float16),
        boundary=boundary.astype(np.float16),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_training_partition_oof(
    *,
    repo_root: str | Path,
    data_root: str | Path | None,
    run_name: str,
    target_mode: TargetMode = "consensus",
    folds: int = 5,
    steps: int = 5_000,
    batch_size: int = 2,
    patch_size: int = 512,
    base_channels: int = 16,
    model_type: str = "unet",
    pretrained_backbone: bool = False,
    hub_repository: str = DINO_V2_REPOSITORY,
    tta: int = 1,
    input_mode: InputMode = "raw",
    instance_config: InstanceConfig | None = None,
    initial_weights: str | Path | None = None,
    external_pretraining: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if folds < 2:
        raise ValueError("At least two OOF folds are required")
    repo_root = Path(repo_root).resolve()
    initial_weights_path = (
        Path(initial_weights).expanduser().resolve() if initial_weights is not None else None
    )
    if initial_weights_path is not None:
        if not initial_weights_path.is_file():
            raise FileNotFoundError(f"Initial OOF checkpoint does not exist: {initial_weights_path}")
        kaggle_runtime = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or (
            Path("/kaggle/input").is_dir() and Path("/kaggle/working").is_dir()
        )
        allowed_roots = [repo_root]
        if kaggle_runtime:
            allowed_roots.extend((Path("/kaggle/input"), Path("/kaggle/working")))
        if not any(
            initial_weights_path.is_relative_to(root.resolve()) for root in allowed_roots
        ):
            raise ValueError("Initial OOF checkpoints must remain inside the workspace or Kaggle paths")
    dataset = load_dataset(_resolve_data_root(data_root, repo_root))
    inventory_info = inventory(dataset)
    if inventory_info["observation_count"] != 707:
        raise ValueError("Expected 707 official training observations")
    if inventory_info["annotator_view_count"] != 1_154:
        raise ValueError("Expected 1,154 official annotator views")
    if inventory_info["annotation_count"] != 8_199:
        raise ValueError("Expected 8,199 competition-provided annotations")

    validation_config_path = repo_root / "configs" / "validation-baseline.json"
    validation_config = json.loads(
        validation_config_path.read_text(encoding="utf-8")
    )
    validation_config_sha256 = hashlib.sha256(
        validation_config_path.read_bytes()
    ).hexdigest()
    frozen_split = create_observation_split(
        dataset,
        repo_root / "data" / "splits",
        seed=int(validation_config["seed"]),
        validation_fraction=float(validation_config["validation_fraction"]),
    )
    if frozen_split.sha256 != validation_config["expected_split_sha256"]:
        raise RuntimeError("The frozen validation split digest changed")

    oof_names = list(frozen_split.training_names)
    if set(oof_names) & set(frozen_split.validation_names):
        raise AssertionError("Frozen validation observations entered the OOF training partition")
    years = np.asarray(
        [dataset.views_by_observation[name][0].year for name in oof_names],
        dtype=np.int32,
    )
    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=int(validation_config["seed"]),
    )
    fold_for_name: dict[str, int] = {}
    fold_splits: list[tuple[list[str], list[str]]] = []
    for fold_index, (train_indices, validation_indices) in enumerate(
        splitter.split(np.zeros(len(oof_names)), years), start=0
    ):
        fold_training_names = [oof_names[index] for index in train_indices]
        fold_validation_names = [oof_names[index] for index in validation_indices]
        if set(fold_training_names) & set(fold_validation_names):
            raise AssertionError(f"Observation overlap in OOF fold {fold_index}")
        fold_splits.append((fold_training_names, fold_validation_names))
        for name in fold_validation_names:
            if name in fold_for_name:
                raise AssertionError(f"OOF observation appears in multiple folds: {name}")
            fold_for_name[name] = fold_index
    if set(fold_for_name) != set(oof_names):
        raise AssertionError("OOF folds do not cover the complete training partition")

    manifest_path = repo_root / "data" / "splits" / f"observation_oof_seed{validation_config['seed']}_{folds}fold.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("file_name", "year", "site", "oof_fold"),
            lineterminator="\n",
        )
        writer.writeheader()
        for file_name in oof_names:
            view = dataset.views_by_observation[file_name][0]
            writer.writerow(
                {
                    "file_name": file_name,
                    "year": view.year,
                    "site": view.site,
                    "oof_fold": fold_for_name[file_name],
                }
            )
    oof_manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    run_root = repo_root / "artifacts" / "oof-experiments" / run_name
    run_root.mkdir(parents=True, exist_ok=True)
    kaggle_runtime = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or (
        Path("/kaggle/input").is_dir() and Path("/kaggle/working").is_dir()
    )
    hub_dir = (
        Path("/kaggle/working/EquiNexus/artifacts/pretrained/torch/hub")
        if kaggle_runtime
        else repo_root / "artifacts" / "pretrained" / "torch" / "hub"
    )
    repository_info = _git_status(repo_root)
    if os.environ.get("EQUINEXUS_SOURCE_COMMIT"):
        repository_info["commit"] = os.environ["EQUINEXUS_SOURCE_COMMIT"]
        repository_info["worktree_dirty"] = True
    repository_info["source_bundle_sha256"] = os.environ.get(
        "EQUINEXUS_SOURCE_BUNDLE_SHA256"
    )
    candidate_config = instance_config or InstanceConfig()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fold_results = []
    all_oof_predictions: dict[str, list[PredictedInstance]] = {}
    all_observation_rows = []
    total_training_seconds = 0.0
    total_inference_seconds = 0.0
    total_scoring_seconds = 0.0
    peak_gpu_memory = 0
    started = time.perf_counter()

    for fold_index, (fold_training_names, fold_validation_names) in enumerate(
        fold_splits, start=0
    ):
        fold_root = run_root / f"fold-{fold_index}"
        fold_root.mkdir(parents=True, exist_ok=True)
        training_dataset = FilamentPatchDataset(
            dataset,
            fold_training_names,
            target_mode=target_mode,
            patch_size=patch_size,
            augment=True,
            seed=int(validation_config["seed"]) + fold_index,
            input_mode=input_mode,
        )
        validation_dataset = FilamentPatchDataset(
            dataset,
            fold_validation_names,
            target_mode=target_mode,
            patch_size=patch_size,
            augment=False,
            seed=int(validation_config["seed"]) + fold_index,
            input_mode=input_mode,
        )
        checkpoint_path = fold_root / "best_model.pt"
        print(
            f"Starting OOF fold {fold_index + 1}/{folds}: "
            f"{len(fold_training_names)} train, {len(fold_validation_names)} validation groups",
            flush=True,
        )
        training_result = train_model(
            train_dataset=training_dataset,
            validation_dataset=validation_dataset,
            output_path=checkpoint_path,
            steps=steps,
            batch_size=batch_size,
            base_channels=base_channels,
            seed=int(validation_config["seed"]) + fold_index,
            validation_interval=max(100, min(500, steps // 5)),
            model_type=model_type,
            pretrained_backbone=pretrained_backbone,
            hub_repository=hub_repository,
            hub_dir=hub_dir,
            initial_weights=initial_weights_path,
        )
        total_training_seconds += float(training_result["elapsed_seconds"])
        peak_gpu_memory = max(
            peak_gpu_memory, int(training_result["peak_gpu_memory_bytes"])
        )

        model = _load_model(checkpoint_path, device, hub_dir=hub_dir)
        fold_predictions: dict[str, list[PredictedInstance]] = {}
        inference_started = time.perf_counter()
        for index, file_name in enumerate(fold_validation_names, start=1):
            image = read_grayscale(dataset.train_image_dir / file_name)
            foreground, boundary = predict_probabilities(
                model,
                image,
                device=device,
                patch_size=patch_size,
                stride=max(patch_size // 2, int(patch_size * 0.75)),
                tta=tta,
            )
            fold_predictions[file_name] = instances_from_probabilities(
                foreground, boundary, file_name, candidate_config
            )
            _save_probability_cache(
                fold_root / "probabilities", file_name, foreground, boundary
            )
            if index % 20 == 0 or index == len(fold_validation_names):
                print(
                    f"Fold {fold_index + 1}: predicted "
                    f"{index}/{len(fold_validation_names)} observations",
                    flush=True,
                )
        inference_seconds = time.perf_counter() - inference_started
        total_inference_seconds += inference_seconds

        scoring_started = time.perf_counter()
        fold_metrics, fold_observation_rows = score_validation(
            dataset,
            fold_predictions,
            fold_validation_names,
            iou_threshold=float(validation_config["iou_threshold"]),
            diagnostic_overlap_threshold=float(
                validation_config["diagnostic_overlap_threshold"]
            ),
        )
        scoring_seconds = time.perf_counter() - scoring_started
        total_scoring_seconds += scoring_seconds
        _write_predictions(fold_root / "predictions.csv", fold_predictions)
        _write_observation_scores(
            fold_root / "per_observation.csv",
            fold_observation_rows,
            fold_predictions,
        )
        fold_result = {
            "fold": fold_index,
            "training_observations": len(fold_training_names),
            "validation_observations": len(fold_validation_names),
            "training": training_result,
            "metrics": fold_metrics,
            "inference_seconds": inference_seconds,
            "scoring_seconds": scoring_seconds,
            "checkpoint": str(checkpoint_path.relative_to(run_root)),
        }
        (fold_root / "metrics.json").write_text(
            json.dumps(fold_result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        fold_results.append(fold_result)
        all_oof_predictions.update(fold_predictions)
        all_observation_rows.extend(fold_observation_rows)
        print(
            f"OOF fold {fold_index + 1}/{folds} PQ="
            f"{fold_metrics['global']['pq']:.6f}",
            flush=True,
        )

    if set(all_oof_predictions) != set(oof_names):
        raise AssertionError("OOF prediction coverage does not match the training partition")
    aggregate_tp = sum(item["metrics"]["global"]["tp"] for item in fold_results)
    aggregate_fp = sum(item["metrics"]["global"]["fp"] for item in fold_results)
    aggregate_fn = sum(item["metrics"]["global"]["fn"] for item in fold_results)
    aggregate_iou = sum(
        item["metrics"]["global"]["tp_iou_sum"] for item in fold_results
    )
    denominator = aggregate_tp + 0.5 * aggregate_fp + 0.5 * aggregate_fn
    aggregate_global = {
        "pq": aggregate_iou / denominator if denominator else 0.0,
        "tp": aggregate_tp,
        "fp": aggregate_fp,
        "fn": aggregate_fn,
        "tp_iou_sum": aggregate_iou,
        "gt_count_across_annotator_views": sum(
            item["metrics"]["global"]["gt_count_across_annotator_views"]
            for item in fold_results
        ),
        "prediction_count_across_annotator_views": sum(
            item["metrics"]["global"]["prediction_count_across_annotator_views"]
            for item in fold_results
        ),
        "annotator_view_count": sum(
            item["metrics"]["global"]["annotator_view_count"]
            for item in fold_results
        ),
    }
    _write_predictions(run_root / "oof-predictions.csv", all_oof_predictions)
    _write_observation_scores(
        run_root / "oof-per-observation.csv",
        all_observation_rows,
        all_oof_predictions,
    )
    _write_overlays(dataset, all_oof_predictions, run_root / "overlays", 12)
    write_comparison_overlays(
        dataset, all_oof_predictions, run_root / "comparison-overlays", 12
    )

    result: dict[str, Any] = {
        "status": "complete",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_name": run_name,
        "repository": repository_info,
        "configuration_sha256": validation_config_sha256,
        "dataset_inventory": inventory_info,
        "data_boundary": {
            "training_labels": (
                "competition-provided training annotations plus separately licensed external "
                "supervised pretraining labels"
                if external_pretraining is not None
                else "competition-provided training annotations only"
            ),
            "excluded_holdout": "all 142 observations in the frozen validation manifest",
            "external_ground_truth": external_pretraining is not None,
        },
        "splits": {
            "frozen_holdout_sha256": frozen_split.sha256,
            "oof_manifest": str(manifest_path),
            "oof_manifest_sha256": oof_manifest_sha256,
            "oof_observations": len(oof_names),
            "fold_count": folds,
        },
        "method": {
            "architecture": model_architecture(model_type),
            "model_type": model_type,
            "target_mode": target_mode,
            "patch_size": patch_size,
                "base_channels": base_channels,
            "input_mode": input_mode,
            "steps_per_fold": steps,
            "batch_size": batch_size,
            "tta": tta,
            "instance_config": candidate_config.__dict__,
            "pretrained_backbone": pretrained_backbone,
            "pretrained_backbone_repository": (
                hub_repository
                if pretrained_backbone and model_type == "dinov2_vits14"
                else "smp-hub/resnet34.imagenet@7a57b34f723329ff020b3f8bc41771163c519d0c"
                if pretrained_backbone and model_type == "smp_unet_resnet34"
                else None
            ),
            "external_supervised_weights": initial_weights_path is not None,
            "external_supervised_checkpoint_sha256": (
                _file_sha256(initial_weights_path) if initial_weights_path is not None else None
            ),
            "external_supervised_pretraining": external_pretraining,
        },
        "folds": fold_results,
        "aggregate_oof": aggregate_global,
        "runtime": {
            **hardware_info(),
            "training_seconds_total": total_training_seconds,
            "inference_seconds_total": total_inference_seconds,
            "scoring_seconds_total": total_scoring_seconds,
            "end_to_end_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": peak_gpu_memory,
            "package_versions": package_versions(),
        },
        "artifacts": {
            "oof_predictions": "oof-predictions.csv",
            "per_observation": "oof-per-observation.csv",
            "overlays": "overlays",
            "comparison_overlays": "comparison-overlays",
        },
    }
    (run_root / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Aggregate grouped {folds}-fold OOF PQ: {aggregate_global['pq']:.6f}; "
        f"split {oof_manifest_sha256}; outputs {run_root}",
        flush=True,
    )
    return result
