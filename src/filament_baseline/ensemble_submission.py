from __future__ import annotations

import csv
import gc
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
from .model_runner import _is_kaggle_runtime, _load_model, _within
from .predictor import read_grayscale
from .runner import _git_status, _resolve_data_root, _write_predictions
from .runtime import hardware_info, package_versions
from .submission import _check_submission
from .training import train_model


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_probability_cache(
    output_path: Path,
    foreground: np.ndarray,
    boundary: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        foreground=foreground.astype(np.float16),
        boundary=boundary.astype(np.float16),
    )


def _fit_and_cache_member(
    *,
    dataset: Any,
    train_names: list[str],
    test_names: list[str],
    member: dict[str, Any],
    member_root: Path,
    seed: int,
) -> dict[str, Any]:
    target_mode: TargetMode = member["target_mode"]
    patch_size = int(member["patch_size"])
    batch_size = int(member["batch_size"])
    steps = int(member["steps"])
    tta = int(member["tta"])
    input_mode: InputMode = member.get("input_mode", "raw")
    if target_mode not in ("annotator", "consensus"):
        raise ValueError(f"Unsupported target mode for ensemble member: {target_mode}")
    if tta not in (1, 2, 4, 8):
        raise ValueError(f"Unsupported TTA count for ensemble member: {tta}")

    member_root.mkdir(parents=True, exist_ok=False)
    train_dataset = FilamentPatchDataset(
        dataset,
        train_names,
        target_mode=target_mode,
        patch_size=patch_size,
        augment=True,
        seed=seed,
        input_mode=input_mode,
    )
    checkpoint_path = member_root / "final_model.pt"
    training = train_model(
        train_dataset=train_dataset,
        validation_dataset=None,
        output_path=checkpoint_path,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        validation_interval=max(100, min(500, steps // 5)),
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = _load_model(checkpoint_path, device)
    cache_dir = member_root / "probabilities"
    inference_started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
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
        _save_probability_cache(
            cache_dir / f"{Path(file_name).stem}.npz", foreground, boundary
        )
        if index % 10 == 0 or index == len(test_names):
            print(
                f"{member['name']}: cached probabilities for {index}/{len(test_names)} test images",
                flush=True,
            )
    inference_seconds = time.perf_counter() - inference_started
    inference_peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    del model, train_dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    member_result = {
        "name": member["name"],
        "configuration": {
            "target_mode": target_mode,
            "steps": steps,
            "batch_size": batch_size,
            "patch_size": patch_size,
            "tta": tta,
            "input_mode": input_mode,
            "seed": seed,
            "external_supervised_weights": False,
        },
        "training": training,
        "inference_seconds": inference_seconds,
        "inference_peak_gpu_memory_bytes": inference_peak_memory,
        "checkpoint": str(checkpoint_path.name),
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "probability_cache": str(cache_dir.name),
    }
    (member_root / "metrics.json").write_text(
        json.dumps(member_result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return member_result


def run_final_ensemble_submission(
    *,
    repo_root: str | Path,
    data_root: str | Path | None,
    run_name: str,
    first_member: dict[str, Any],
    second_member: dict[str, Any],
    first_weight: float,
    instance_config: InstanceConfig,
) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", run_name):
        raise ValueError("run_name must use only letters, digits, dot, underscore, or dash")
    if not 0.0 <= first_weight <= 1.0:
        raise ValueError("first_weight must be between zero and one")
    if first_member.get("name") == second_member.get("name"):
        raise ValueError("Ensemble member names must differ")

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
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"Submission output already exists: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)

    config_path = repo_root / "configs" / "validation-baseline.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_sha256 = _file_sha256(config_path)
    dataset = load_dataset(dataset_root)
    data_inventory = inventory(dataset)
    expected = {
        "observation_count": 707,
        "annotator_view_count": 1_154,
        "annotation_count": 8_199,
        "test_image_file_count": 180,
    }
    for key, value in expected.items():
        if data_inventory[key] != value:
            raise ValueError(f"Expected {value} {key}, found {data_inventory[key]}")

    train_names = list(dataset.observation_names)
    test_names = sorted(path.name for path in dataset.test_image_dir.iterdir() if path.is_file())
    if len(test_names) != expected["test_image_file_count"]:
        raise ValueError("Test image inventory differs from the official 180-image set")
    if len({Path(name).stem for name in test_names}) != len(test_names):
        raise ValueError("Test image stems must be unique for stable filament IDs and caches")
    members_root = run_root / "members"
    members_root.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    repository = _git_status(repo_root)
    member_results = []
    for index, member in enumerate((first_member, second_member), start=1):
        name = member.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", name):
            raise ValueError("Each ensemble member needs a safe, unique name")
        print(f"Training final ensemble member {index}/2: {name}", flush=True)
        member_results.append(
            _fit_and_cache_member(
                dataset=dataset,
                train_names=train_names,
                test_names=test_names,
                member=member,
                member_root=members_root / name,
                seed=int(config["seed"]),
            )
        )

    first_cache = members_root / first_member["name"] / "probabilities"
    second_cache = members_root / second_member["name"] / "probabilities"
    predictions: dict[str, list] = {}
    coverage_rows = []
    postprocess_started = time.perf_counter()
    for index, file_name in enumerate(test_names, start=1):
        stem = Path(file_name).stem
        with np.load(first_cache / f"{stem}.npz") as first, np.load(
            second_cache / f"{stem}.npz"
        ) as second:
            first_foreground = first["foreground"].astype(np.float32)
            first_boundary = first["boundary"].astype(np.float32)
            second_foreground = second["foreground"].astype(np.float32)
            second_boundary = second["boundary"].astype(np.float32)
        expected_shape = (IMAGE_SIZE, IMAGE_SIZE)
        if not (
            first_foreground.shape
            == first_boundary.shape
            == second_foreground.shape
            == second_boundary.shape
            == expected_shape
        ):
            raise ValueError(f"Invalid ensemble probability dimensions for {file_name}")
        foreground = first_weight * first_foreground + (1.0 - first_weight) * second_foreground
        boundary = first_weight * first_boundary + (1.0 - first_weight) * second_boundary
        predictions[file_name] = instances_from_probabilities(
            foreground, boundary, file_name, instance_config
        )
        coverage_rows.append(
            {"file_name": file_name, "instance_count": len(predictions[file_name])}
        )
        if index % 10 == 0 or index == len(test_names):
            print(f"Postprocessed {index}/{len(test_names)} test images", flush=True)
    postprocess_seconds = time.perf_counter() - postprocess_started

    submission_path = run_root / "submission.csv"
    _write_predictions(submission_path, predictions)
    submission_validation = _check_submission(
        submission_path, dataset.root, test_names, predictions
    )
    coverage_path = run_root / "test_coverage.csv"
    with coverage_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("file_name", "instance_count"), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(coverage_rows)

    if os.environ.get("EQUINEXUS_SOURCE_COMMIT"):
        repository["commit"] = os.environ["EQUINEXUS_SOURCE_COMMIT"]
        repository["worktree_dirty"] = True
    repository["source_bundle_sha256"] = os.environ.get(
        "EQUINEXUS_SOURCE_BUNDLE_SHA256"
    )
    result = {
        "status": "complete",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_name": run_name,
        "repository": repository,
        "configuration_sha256": config_sha256,
        "kaggle_notebook_version": os.environ.get("EQUINEXUS_NOTEBOOK_VERSION"),
        "data_boundary": {
            "training_labels": "competition-provided train annotations only",
            "inference_inputs": "attached competition H-alpha test images only",
            "external_ground_truth": False,
        },
        "dataset_inventory": data_inventory,
        "validation_reference": {
            "split_sha256": config["expected_split_sha256"],
            "selection_note": "The frozen observation-grouped holdout selected the ensemble and postprocessing before final refitting on all official training observations.",
        },
        "ensemble": {
            "first_member": first_member,
            "second_member": second_member,
            "first_member_weight": first_weight,
            "second_member_weight": 1.0 - first_weight,
            "instance_config": instance_config.__dict__,
            "external_supervised_weights": False,
        },
        "members": member_results,
        "test_validation": submission_validation,
        "runtime": {
            **hardware_info(),
            "postprocess_seconds": postprocess_seconds,
            "end_to_end_seconds": time.perf_counter() - start_time,
            "package_versions": package_versions(),
            "max_member_training_gpu_memory_bytes": max(
                item["training"]["peak_gpu_memory_bytes"] for item in member_results
            ),
            "max_member_inference_gpu_memory_bytes": max(
                item["inference_peak_gpu_memory_bytes"] for item in member_results
            ),
        },
        "artifacts": {
            "submission": submission_path.name,
            "coverage": coverage_path.name,
            "members_root": str(members_root.name),
        },
    }
    (run_root / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Validated ensemble submission with {submission_validation['predicted_instances']} instances "
        f"for {submission_validation['test_images']} images: {submission_path}",
        flush=True,
    )
    return result
