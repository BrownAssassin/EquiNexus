# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import IMAGE_SIZE, inventory, load_dataset
from .features import InputMode
from .inference import InstanceConfig, instances_from_probabilities, predict_probabilities
from .learning import FilamentPatchDataset, TargetMode
from .model_runner import _file_sha256, _is_kaggle_runtime, _load_model, _within
from .model import DINO_V2_REPOSITORY
from .predictor import decode_rle, read_grayscale
from .runner import _git_status, _resolve_data_root, _write_predictions
from .runtime import hardware_info, package_versions
from .training import train_model


def _check_submission(
    output_path: Path,
    dataset_root: Path,
    test_names: list[str],
    predictions: dict[str, list],
) -> dict[str, int]:
    expected_columns = ["filament_id", "segmentation_rle"]
    with output_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != expected_columns:
            raise ValueError(f"Unexpected submission columns: {reader.fieldnames}")
        rows = list(reader)

    expected_ids = {
        instance.filament_id
        for file_name in test_names
        for instance in predictions[file_name]
    }
    row_ids = [row["filament_id"] for row in rows]
    if len(row_ids) != len(set(row_ids)) or set(row_ids) != expected_ids:
        raise ValueError("Submission filament IDs are duplicated or differ from predictions")
    if any(not row["segmentation_rle"] for row in rows):
        raise ValueError("Submission contains an empty RLE string")

    instance_count = 0
    for file_name in test_names:
        image = read_grayscale(dataset_root / "test" / "test_images" / file_name)
        if image.shape != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"Unexpected test image size for {file_name}: {image.shape}")
        occupied = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
        seen_for_image = set()
        for instance in predictions[file_name]:
            if not instance.filament_id.startswith(f"{Path(file_name).stem}_"):
                raise ValueError(f"Filament ID does not belong to {file_name}")
            if instance.filament_id in seen_for_image:
                raise ValueError(f"Duplicate filament ID in {file_name}")
            seen_for_image.add(instance.filament_id)
            mask = decode_rle(instance.rle)
            if mask.shape != (IMAGE_SIZE, IMAGE_SIZE):
                raise ValueError(f"Invalid mask dimensions for {instance.filament_id}")
            if int(mask.sum()) != instance.area:
                raise ValueError(f"RLE area mismatch for {instance.filament_id}")
            if np.any(occupied & mask.astype(bool)):
                raise ValueError(f"Predicted instances overlap in {file_name}")
            occupied |= mask.astype(bool)
            instance_count += 1

    if len(rows) != instance_count:
        raise ValueError("Submission row count does not match validated test predictions")
    return {"test_images": len(test_names), "predicted_instances": instance_count}


def run_final_submission(
    *,
    repo_root: str | Path,
    data_root: str | Path | None,
    run_name: str,
    target_mode: TargetMode = "consensus",
    steps: int = 5_000,
    batch_size: int = 2,
    patch_size: int = 512,
    base_channels: int = 16,
    model_type: str = "unet",
    pretrained_backbone: bool = False,
    hub_repository: str = DINO_V2_REPOSITORY,
    tta: int = 1,
    input_mode: InputMode = "raw",
    foreground_threshold: float = 0.45,
    boundary_threshold: float = 0.65,
    boundary_weight: float = 1.5,
    seed_threshold: float = 0.45,
    seed_min_area: int = 8,
    min_instance_area: int = 50,
) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", run_name):
        raise ValueError("run_name must use only letters, digits, dot, underscore, or dash")
    repo_root = Path(repo_root).resolve()
    kaggle_runtime = _is_kaggle_runtime()
    dataset_root = _resolve_data_root(data_root, repo_root)
    if not kaggle_runtime and not _within(dataset_root, repo_root):
        raise ValueError(f"Local competition data must stay in the repository: {dataset_root}")
    run_root = (
        Path("/kaggle/working") / "equinexus" / run_name
        if kaggle_runtime
        else repo_root / "artifacts" / "submissions" / run_name
    )
    if not kaggle_runtime and not _within(run_root, repo_root):
        raise ValueError(f"Local outputs must stay in the repository: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    hub_dir = (
        Path("/kaggle/working/equinexus/pretrained/torch/hub")
        if kaggle_runtime
        else repo_root / "artifacts" / "pretrained" / "torch" / "hub"
    )
    repo_info = _git_status(repo_root)

    config_json = json.loads(
        (repo_root / "configs" / "validation-baseline.json").read_text(encoding="utf-8")
    )
    config_sha256 = hashlib.sha256(
        (repo_root / "configs" / "validation-baseline.json").read_bytes()
    ).hexdigest()
    dataset = load_dataset(dataset_root)
    data_inventory = inventory(dataset)
    if data_inventory["observation_count"] != 707:
        raise ValueError("Final training must use all 707 official competition observations")
    if data_inventory["annotator_view_count"] != 1_154:
        raise ValueError("Final training must use all 1,154 official annotator views")
    if data_inventory["annotation_count"] != 8_199:
        raise ValueError("Unexpected number of competition-provided training annotations")
    if data_inventory["test_image_file_count"] != 180:
        raise ValueError("Expected 180 supplied H-alpha test images")

    train_names = list(dataset.observation_names)
    train_dataset = FilamentPatchDataset(
        dataset,
        train_names,
        target_mode=target_mode,
        patch_size=patch_size,
        augment=True,
        seed=int(config_json["seed"]),
        input_mode=input_mode,
    )
    checkpoint_path = run_root / "final_model.pt"
    start_time = time.perf_counter()
    training = train_model(
        train_dataset=train_dataset,
        validation_dataset=None,
        output_path=checkpoint_path,
        steps=steps,
        batch_size=batch_size,
        base_channels=base_channels,
        seed=int(config_json["seed"]),
        validation_interval=max(100, min(500, steps // 5)),
        model_type=model_type,
        pretrained_backbone=pretrained_backbone,
        hub_repository=hub_repository,
        hub_dir=hub_dir,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = _load_model(checkpoint_path, device, hub_dir=hub_dir)
    pretrained_checkpoint = hub_dir / "checkpoints" / "dinov2_vits14_pretrain.pth"
    pretrained_checkpoint_sha256 = (
        _file_sha256(pretrained_checkpoint)
        if model_type == "dinov2_vits14" and pretrained_backbone
        else None
    )
    adaptation_metadata = None
    adaptation_metadata_path = pretrained_checkpoint.with_suffix(".json")
    if (
        model_type == "dinov2_vits14"
        and pretrained_backbone
        and adaptation_metadata_path.is_file()
    ):
        candidate_adaptation = json.loads(adaptation_metadata_path.read_text(encoding="utf-8"))
        if candidate_adaptation.get("adapted_checkpoint_sha256") == pretrained_checkpoint_sha256:
            adaptation_metadata = candidate_adaptation
    instance_config = InstanceConfig(
        foreground_threshold=foreground_threshold,
        boundary_threshold=boundary_threshold,
        boundary_weight=boundary_weight,
        seed_threshold=seed_threshold,
        seed_min_area=seed_min_area,
        min_instance_area=min_instance_area,
    )
    test_names = sorted(
        path.name for path in dataset.test_image_dir.iterdir() if path.is_file()
    )
    predictions: dict[str, list] = {}
    coverage_rows = []
    inference_start = time.perf_counter()
    for index, file_name in enumerate(test_names, start=1):
        image = read_grayscale(dataset.test_image_dir / file_name)
        foreground, boundary = predict_probabilities(
            model,
            image,
            device=device,
            patch_size=patch_size,
            stride=max(patch_size // 2, int(patch_size * 0.75)),
            tta=tta,
        )
        predictions[file_name] = instances_from_probabilities(
            foreground, boundary, file_name, instance_config
        )
        coverage_rows.append(
            {"file_name": file_name, "instance_count": len(predictions[file_name])}
        )
        if index % 10 == 0 or index == len(test_names):
            print(f"Predicted {index}/{len(test_names)} test images", flush=True)
    inference_seconds = time.perf_counter() - inference_start

    submission_path = run_root / "submission.csv"
    _write_predictions(submission_path, predictions)
    validation = _check_submission(
        submission_path,
        dataset.root,
        test_names,
        predictions,
    )
    coverage_path = run_root / "test_coverage.csv"
    with coverage_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("file_name", "instance_count"), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(coverage_rows)

    if os.environ.get("EQUINEXUS_SOURCE_COMMIT"):
        repo_info["commit"] = os.environ["EQUINEXUS_SOURCE_COMMIT"]
        repo_info["worktree_dirty"] = True
    repo_info["source_bundle_sha256"] = os.environ.get(
        "EQUINEXUS_SOURCE_BUNDLE_SHA256"
    )
    result = {
        "status": "complete",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_name": run_name,
        "repository": repo_info,
        "configuration_sha256": config_sha256,
        "kaggle_notebook_version": os.environ.get("EQUINEXUS_NOTEBOOK_VERSION"),
        "data_boundary": {
            "training_labels": "competition-provided train annotations only",
            "inference_inputs": "attached competition H-alpha test images only",
            "external_ground_truth": False,
        },
        "dataset_inventory": data_inventory,
        "validation_reference": {
            "split_sha256": config_json["expected_split_sha256"],
            "selection_note": "Postprocessing selected on the frozen observation-grouped holdout before refitting on all official training observations.",
        },
        "training": {**training, "training_observations": len(train_names)},
        "method": {
            "architecture": (
                "dinov2_vits14_frozen_backbone_cnn_decoder_foreground_boundary"
                if model_type == "dinov2_vits14"
                else "from_scratch_five_level_unet_foreground_boundary"
            ),
            "model_type": model_type,
            "target_mode": target_mode,
            "patch_size": patch_size,
            "base_channels": base_channels,
            "input_mode": input_mode,
            "steps": steps,
            "batch_size": batch_size,
            "tta": tta,
            "instance_config": instance_config.__dict__,
            "pretrained_backbone": pretrained_backbone,
            "pretrained_backbone_repository": hub_repository if pretrained_backbone else None,
            "pretrained_backbone_license": "Apache-2.0"
            if model_type == "dinov2_vits14" and pretrained_backbone
            else None,
            "pretrained_backbone_checkpoint_sha256": pretrained_checkpoint_sha256,
            "pretraining_data": "DINOv2 LVD-142M, self-supervised without labels"
            if pretrained_backbone
            else None,
            "external_unlabeled_data": adaptation_metadata,
            "external_supervised_weights": False,
        },
        "test_validation": validation,
        "runtime": {
            **hardware_info(),
            "inference_seconds": inference_seconds,
            "end_to_end_seconds": time.perf_counter() - start_time,
            "package_versions": package_versions(),
        },
        "artifacts": {
            "checkpoint": checkpoint_path.name,
            "training_history": checkpoint_path.with_suffix(".jsonl").name,
            "submission": submission_path.name,
            "coverage": coverage_path.name,
        },
    }
    (run_root / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Validated submission with {validation['predicted_instances']} instances "
        f"for {validation['test_images']} images: {submission_path}",
        flush=True,
    )
    return result
