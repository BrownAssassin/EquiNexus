# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from filament_baseline.data import IMAGE_SIZE, inventory, load_dataset  # noqa: E402
from filament_baseline.inference import InstanceConfig, instances_from_probabilities, predict_probabilities  # noqa: E402
from filament_baseline.model_runner import _is_kaggle_runtime, _load_model, _resolve_data_root, _within  # noqa: E402
from filament_baseline.predictor import read_grayscale  # noqa: E402
from filament_baseline.runner import _git_status, _write_observation_scores, _write_overlays, _write_predictions  # noqa: E402
from filament_baseline.runtime import hardware_info, package_versions  # noqa: E402
from filament_baseline.scorer import score_validation  # noqa: E402
from filament_baseline.split import create_observation_split  # noqa: E402
from filament_baseline.submission import _check_submission  # noqa: E402
from filament_baseline.visuals import write_comparison_overlays  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_probabilities(path: Path, foreground: np.ndarray, boundary: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        foreground=foreground.astype(np.float16),
        boundary=boundary.astype(np.float16),
    )


def _average_fold_probabilities(
    *, fold_cache_root: Path, averaged_cache_root: Path, subset: str, file_name: str
) -> tuple[np.ndarray, np.ndarray]:
    stem = Path(file_name).stem
    foreground_sum = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    boundary_sum = np.zeros_like(foreground_sum)
    for fold in range(5):
        with np.load(fold_cache_root / f"fold-{fold}" / subset / f"{stem}.npz") as cache:
            foreground = cache["foreground"].astype(np.float32)
            boundary = cache["boundary"].astype(np.float32)
        if foreground.shape != (IMAGE_SIZE, IMAGE_SIZE) or boundary.shape != foreground.shape:
            raise ValueError(f"Invalid fold-{fold} probability shape for {file_name}")
        foreground_sum += foreground
        boundary_sum += boundary
    foreground_sum /= 5.0
    boundary_sum /= 5.0
    _save_probabilities(
        averaged_cache_root / subset / f"{stem}.npz", foreground_sum, boundary_sum
    )
    return foreground_sum, boundary_sum


def run_fold_ensemble(
    *,
    repo_root: Path,
    data_root: Path | None,
    checkpoint_root: Path,
    run_name: str,
    instance_config: InstanceConfig,
    overlay_count: int = 12,
    keep_fold_probabilities: bool = False,
) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", run_name):
        raise ValueError("run_name must use only letters, digits, dot, underscore, or dash")
    repo_root = repo_root.resolve()
    checkpoint_root = checkpoint_root.resolve()
    kaggle_runtime = _is_kaggle_runtime()
    dataset_root = _resolve_data_root(data_root, repo_root)
    if not kaggle_runtime and not _within(dataset_root, repo_root):
        raise ValueError(f"Local competition data must stay inside the repository: {dataset_root}")
    if not kaggle_runtime and not _within(checkpoint_root, repo_root):
        raise ValueError(f"Local checkpoints must stay inside the repository: {checkpoint_root}")
    run_root = (
        Path("/kaggle/working") / "equinexus" / run_name
        if kaggle_runtime
        else repo_root / "artifacts" / "model-experiments" / run_name
    ).resolve()
    if not kaggle_runtime and not _within(run_root, repo_root):
        raise ValueError(f"Local outputs must stay inside the repository: {run_root}")
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"Run output already contains files: {run_root}")

    config_path = repo_root / "configs" / "validation-baseline.json"
    validation_config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset = load_dataset(dataset_root)
    dataset_inventory = inventory(dataset)
    expected_inventory = {
        "observation_count": 707,
        "annotator_view_count": 1154,
        "annotation_count": 8199,
        "test_image_file_count": 180,
    }
    for key, value in expected_inventory.items():
        if dataset_inventory[key] != value:
            raise ValueError(f"Expected {value} {key}, found {dataset_inventory[key]}")

    split = create_observation_split(
        dataset,
        repo_root / "data" / "splits",
        seed=int(validation_config["seed"]),
        validation_fraction=float(validation_config["validation_fraction"]),
    )
    if split.sha256 != validation_config["expected_split_sha256"]:
        raise RuntimeError("The frozen observation-grouped validation split changed")
    validation_names = sorted(split.validation_names)
    test_names = sorted(path.name for path in dataset.test_image_dir.iterdir() if path.is_file())
    if len(validation_names) != 142 or len(test_names) != 180:
        raise ValueError("Unexpected validation or test image count")
    if set(validation_names) & set(split.training_names):
        raise AssertionError("Frozen training and validation observations overlap")

    checkpoint_manifest_path = checkpoint_root / "checkpoint-manifest.json"
    checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
    oof_manifest_name = checkpoint_manifest.get(
        "oof_manifest_file", "observation_oof_seed42_5fold.csv"
    )
    oof_manifest_candidates = (
        checkpoint_root / oof_manifest_name,
        repo_root / "data" / "splits" / oof_manifest_name,
    )
    oof_manifest_path = next(
        (candidate for candidate in oof_manifest_candidates if candidate.is_file()), None
    )
    if oof_manifest_path is None:
        raise FileNotFoundError(
            "Grouped OOF manifest missing from checkpoint bundle and repository split folder"
        )
    oof_manifest_sha256 = _sha256(oof_manifest_path)
    if checkpoint_manifest.get("status") != "complete":
        raise ValueError("Checkpoint bundle is not complete")
    if checkpoint_manifest.get("frozen_split_sha256") != split.sha256:
        raise ValueError("Checkpoint bundle uses a different frozen holdout")
    if checkpoint_manifest.get("oof_manifest_sha256") != oof_manifest_sha256:
        raise ValueError("Checkpoint bundle uses a different grouped OOF manifest")
    training_policy = checkpoint_manifest.get("training", {})
    if (
        training_policy.get("labels") != "competition-provided train annotations only"
        or training_policy.get("external_supervised_weights") is not False
        or training_policy.get("test_observations_used") is not False
    ):
        raise ValueError("Checkpoint training provenance does not meet the competition data boundary")
    checkpoint_entries = checkpoint_manifest.get("checkpoints", [])
    if {int(entry["fold"]) for entry in checkpoint_entries} != set(range(5)):
        raise ValueError("Checkpoint manifest must contain exactly five fold models")
    for entry in checkpoint_entries:
        checkpoint_path = checkpoint_root / entry["file"]
        if not checkpoint_path.is_file() or _sha256(checkpoint_path) != entry["sha256"]:
            raise ValueError(f"Fold checkpoint integrity mismatch: {checkpoint_path}")

    run_root.mkdir(parents=True, exist_ok=True)
    git = _git_status(repo_root)
    fold_cache_root = run_root / "fold-probability-cache"
    averaged_cache_root = run_root / "probabilities"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("The five-fold inference run requires a CUDA GPU")
    torch.cuda.set_device(0)
    try:
        torch.cuda.reset_peak_memory_stats()
        peak_memory_stats_available = True
    except RuntimeError as error:
        # Kaggle's CUDA 13.0 / T4 runtime has intermittently rejected the
        # device-qualified stats reset even though CUDA inference is available.
        print(f"CUDA peak-memory reset unavailable; continuing inference: {error}", flush=True)
        peak_memory_stats_available = False
    inference_started = time.perf_counter()
    subsets = (
        ("holdout", validation_names, dataset.train_image_dir),
        ("test", test_names, dataset.test_image_dir),
    )

    for fold in range(5):
        checkpoint_path = checkpoint_root / f"fold-{fold}.pt"
        print(f"Loading fold {fold + 1}/5 checkpoint", flush=True)
        model = _load_model(checkpoint_path, device)
        for subset, names, image_dir in subsets:
            for index, file_name in enumerate(names, start=1):
                image = read_grayscale(image_dir / file_name)
                foreground, boundary = predict_probabilities(
                    model,
                    image,
                    device=device,
                    patch_size=1024,
                    stride=768,
                    tta=8,
                )
                _save_probabilities(
                    fold_cache_root / f"fold-{fold}" / subset / f"{Path(file_name).stem}.npz",
                    foreground,
                    boundary,
                )
                if index % 20 == 0 or index == len(names):
                    print(f"fold-{fold} {subset}: predicted {index}/{len(names)}", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    validation_predictions = {}
    for index, file_name in enumerate(validation_names, start=1):
        foreground, boundary = _average_fold_probabilities(
            fold_cache_root=fold_cache_root,
            averaged_cache_root=averaged_cache_root,
            subset="holdout",
            file_name=file_name,
        )
        validation_predictions[file_name] = instances_from_probabilities(
            foreground, boundary, file_name, instance_config
        )
        if index % 25 == 0 or index == len(validation_names):
            print(f"Averaged {index}/{len(validation_names)} holdout observations", flush=True)
    validation_metrics, observation_rows = score_validation(
        dataset,
        validation_predictions,
        validation_names,
        iou_threshold=float(validation_config["iou_threshold"]),
        diagnostic_overlap_threshold=float(validation_config["diagnostic_overlap_threshold"]),
    )
    _write_predictions(run_root / "holdout-predictions.csv", validation_predictions)
    _write_observation_scores(
        run_root / "holdout-per-observation.csv", observation_rows, validation_predictions
    )
    validation_overlays = _write_overlays(
        dataset, validation_predictions, run_root / "holdout-overlays", overlay_count
    )
    comparison_overlays = write_comparison_overlays(
        dataset,
        validation_predictions,
        run_root / "holdout-comparison-overlays",
        overlay_count,
    )

    test_predictions = {}
    coverage_rows = []
    for index, file_name in enumerate(test_names, start=1):
        foreground, boundary = _average_fold_probabilities(
            fold_cache_root=fold_cache_root,
            averaged_cache_root=averaged_cache_root,
            subset="test",
            file_name=file_name,
        )
        test_predictions[file_name] = instances_from_probabilities(
            foreground, boundary, file_name, instance_config
        )
        coverage_rows.append({"file_name": file_name, "instance_count": len(test_predictions[file_name])})
        if index % 25 == 0 or index == len(test_names):
            print(f"Averaged {index}/{len(test_names)} test observations", flush=True)

    submission_path = run_root / "submission.csv"
    _write_predictions(submission_path, test_predictions)
    submission_validation = _check_submission(
        submission_path, dataset.root, test_names, test_predictions
    )
    with (run_root / "test-coverage.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("file_name", "instance_count"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(coverage_rows)

    if not keep_fold_probabilities:
        safe_cache_root = fold_cache_root.resolve()
        if safe_cache_root.parent != run_root or not _within(safe_cache_root, run_root):
            raise RuntimeError(f"Refusing to remove unexpected fold cache path: {safe_cache_root}")
        shutil.rmtree(safe_cache_root)

    repository_info = git
    if os.environ.get("EQUINEXUS_SOURCE_COMMIT"):
        repository_info["commit"] = os.environ["EQUINEXUS_SOURCE_COMMIT"]
        repository_info["worktree_dirty"] = True
    repository_info["source_bundle_sha256"] = os.environ.get("EQUINEXUS_SOURCE_BUNDLE_SHA256")
    result = {
        "status": "complete",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_name": run_name,
        "repository": repository_info,
        "source_checkpoint_manifest_sha256": _sha256(checkpoint_manifest_path),
        "source_checkpoints": checkpoint_entries,
        "source_data_boundary": training_policy,
        "split": {
            "sha256": split.sha256,
            "observation_count": len(validation_names),
            "oof_manifest_sha256": oof_manifest_sha256,
            "validation_is_excluded_from_every_fold_training_set": True,
        },
        "dataset_inventory": dataset_inventory,
        "method": {
            "name": "five-fold per-annotator U-Net probability mean",
            "target_mode": "annotator",
            "patch_size": 1024,
            "tta": 8,
            "external_supervised_weights": False,
            "instance_config": instance_config.__dict__,
        },
        "holdout": {
            "metrics": validation_metrics,
            "artifacts": {
                "predictions": "holdout-predictions.csv",
                "per_observation": "holdout-per-observation.csv",
                "overlays": validation_overlays,
                "comparison_overlays": comparison_overlays,
            },
        },
        "test_validation": submission_validation,
        "data_boundary": {
            "training_labels": "competition-provided train annotations only",
            "inference_inputs": "official competition H-alpha test images only",
            "test_labels_read": False,
            "external_ground_truth": False,
        },
        "runtime": {
            **hardware_info(),
            "inference_seconds": time.perf_counter() - inference_started,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated()) if peak_memory_stats_available else None
            ),
            "package_versions": package_versions(),
        },
        "artifacts": {
            "submission": submission_path.name,
            "test_coverage": "test-coverage.csv",
            "averaged_probability_cache": "probabilities/",
        },
    }
    result["submission_sha256"] = _sha256(submission_path)
    (run_root / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "holdout_pq": validation_metrics["global"]["pq"], "submission_sha256": result["submission_sha256"], "test_validation": submission_validation}, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a validation-safe five-fold U-Net ensemble and prepare its submission.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--run-name", default="oof-fivefold-mean-v1")
    parser.add_argument("--foreground-threshold", type=float, default=0.65)
    parser.add_argument("--boundary-threshold", type=float, default=0.70)
    parser.add_argument("--boundary-weight", type=float, default=1.5)
    parser.add_argument("--seed-threshold", type=float, default=0.45)
    parser.add_argument("--seed-min-area", type=int, default=8)
    parser.add_argument("--min-instance-area", type=int, default=400)
    parser.add_argument("--solar-radius", type=int, default=950)
    parser.add_argument("--overlay-count", type=int, default=12)
    parser.add_argument("--keep-fold-probabilities", action="store_true")
    args = parser.parse_args()
    result = run_fold_ensemble(
        repo_root=args.repo_root,
        data_root=args.data_root,
        checkpoint_root=args.checkpoint_root,
        run_name=args.run_name,
        instance_config=InstanceConfig(
            foreground_threshold=args.foreground_threshold,
            boundary_threshold=args.boundary_threshold,
            boundary_weight=args.boundary_weight,
            seed_threshold=args.seed_threshold,
            seed_min_area=args.seed_min_area,
            min_instance_area=args.min_instance_area,
            solar_radius=args.solar_radius,
        ),
        overlay_count=args.overlay_count,
        keep_fold_probabilities=args.keep_fold_probabilities,
    )
    print(json.dumps({"status": result["status"], "holdout_pq": result["holdout"]["metrics"]["global"]["pq"], "submission_sha256": result["submission_sha256"]}, indent=2))
