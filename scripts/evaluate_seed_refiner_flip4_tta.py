# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from filament_baseline.data import load_dataset  # noqa: E402
from filament_baseline.predictor import read_grayscale  # noqa: E402
from filament_baseline.runner import _write_observation_scores, _write_predictions  # noqa: E402
from filament_baseline.runtime import hardware_info, package_versions  # noqa: E402
from filament_baseline.scorer import score_validation  # noqa: E402
from filament_baseline.seed_refiner import (  # noqa: E402
    SeedRefinerConfig,
    load_seed_refiner,
    predicted_instances,
)
from filament_baseline.split import create_observation_split  # noqa: E402
from filament_baseline.visuals import write_comparison_overlays  # noqa: E402


RUN_NAME = "2026-11-08-seedrefiner-yolo11s-crop640-flip4-topology-guard"
SOURCE_RUN_NAME = "2026-11-08-seedrefiner-yolo11s-crop640-10k-scratch"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate public flip-4 topology-guarded crop-refiner TTA on the fixed holdout."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    repo_root = args.repo_root.expanduser().resolve()
    data_root = (
        args.data_root.expanduser().resolve()
        if args.data_root
        else repo_root / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else repo_root / "artifacts" / "model-experiments" / RUN_NAME
    )
    if not data_root.is_relative_to((repo_root / "data").resolve()):
        raise ValueError("Competition data must remain inside the repository data directory")
    if not output_dir.is_relative_to((repo_root / "artifacts").resolve()):
        raise ValueError("Experiment output must remain inside the repository artifacts directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    source_dir = repo_root / "artifacts" / "model-experiments" / SOURCE_RUN_NAME
    source_metrics_path = source_dir / "metrics.json"
    threshold_metrics_path = source_dir / "threshold-sweep" / "best-holdout-screen" / "metrics.json"
    checkpoint_path = source_dir / "seed_refiner.pt"
    source_metrics = json.loads(source_metrics_path.read_text(encoding="utf-8-sig"))
    selected_screen = json.loads(threshold_metrics_path.read_text(encoding="utf-8-sig"))
    baseline = json.loads(
        (repo_root / "configs" / "validation-baseline.json").read_text(encoding="utf-8")
    )
    expected_split = baseline["expected_split_sha256"]
    if source_metrics.get("split_sha256") != expected_split:
        raise RuntimeError("Crop-refiner checkpoint was produced for a different frozen split")
    if source_metrics.get("training_observations") != 565:
        raise RuntimeError("Expected the refiner to train on the 565-observation partition only")
    if source_metrics.get("validation_observations") != 142:
        raise RuntimeError("The source run does not record the expected 142-observation holdout")
    if source_metrics.get("submission_test_ground_truth_used") is not False:
        raise RuntimeError("The source run does not certify the test-label boundary")
    if _sha256(checkpoint_path) != source_metrics["training"]["checkpoint_sha256"]:
        raise RuntimeError("Crop-refiner checkpoint hash differs from its source metadata")
    if selected_screen.get("split_sha256") != expected_split:
        raise RuntimeError("The selected baseline inference configuration uses another split")
    source_config = selected_screen.get("configuration", {})
    if source_config != {
        "validation_score_min": 0.45,
        "probability_threshold": 0.50,
        "minimum_instance_area": 400,
    }:
        raise RuntimeError(f"Unexpected fixed refiner screening configuration: {source_config}")

    dataset = load_dataset(data_root)
    split = create_observation_split(
        dataset,
        repo_root / "data" / "splits",
        seed=int(baseline["seed"]),
        validation_fraction=float(baseline["validation_fraction"]),
    )
    if split.sha256 != expected_split:
        raise RuntimeError("The regenerated holdout does not match the frozen split digest")
    if len(split.validation_names) != 142:
        raise RuntimeError("Expected exactly 142 holdout observations")

    validation_detection_path = repo_root / source_metrics["validation_detection_source"]
    if _sha256(validation_detection_path) != source_metrics["validation_detection_sha256"]:
        raise RuntimeError("Validation detector predictions do not match the recorded source hash")
    with validation_detection_path.open("r", encoding="utf-8") as stream:
        detections: dict[str, list[dict[str, Any]]] = json.load(stream)
    holdout_stems = {Path(name).stem for name in split.validation_names}
    if not holdout_stems.issubset(detections):
        raise RuntimeError("Validation detector predictions do not cover the frozen holdout")
    test_stems = {path.stem for path in dataset.test_image_dir.iterdir() if path.is_file()}
    if holdout_stems & test_stems:
        raise RuntimeError("A competition test image was found in the frozen training holdout")

    allowed_fields = {field.name for field in SeedRefinerConfig.__dataclass_fields__.values()}
    config = SeedRefinerConfig(
        **{key: value for key, value in source_metrics["configuration"].items() if key in allowed_fields}
    )
    config = SeedRefinerConfig(
        **{
            **config.__dict__,
            "validation_score_min": float(source_config["validation_score_min"]),
            "probability_threshold": float(source_config["probability_threshold"]),
            "minimum_instance_area": int(source_config["minimum_instance_area"]),
        }
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, _checkpoint = load_seed_refiner(checkpoint_path, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = {}
    diagnostics: Counter[str] = Counter()
    diagnostic_rows = []
    inference_started = time.perf_counter()
    for index, file_name in enumerate(sorted(split.validation_names), start=1):
        image = read_grayscale(dataset.train_image_dir / file_name)
        per_observation_diagnostics: dict[str, int] = {}
        predictions[file_name] = predicted_instances(
            file_name,
            image,
            detections.get(Path(file_name).stem, []),
            model,
            config,
            device=device,
            flip_tta=True,
            topology_safe_tta=True,
            tta_diagnostics=per_observation_diagnostics,
        )
        diagnostics.update(per_observation_diagnostics)
        diagnostic_rows.append(
            {
                "file_name": file_name,
                "accepted": per_observation_diagnostics.get("accepted", 0),
                "rejected": sum(
                    count
                    for reason, count in per_observation_diagnostics.items()
                    if reason != "accepted"
                ),
                "rejection_reasons": json.dumps(
                    {
                        reason: count
                        for reason, count in sorted(per_observation_diagnostics.items())
                        if reason != "accepted"
                    },
                    sort_keys=True,
                ),
            }
        )
        if index % 20 == 0 or index == len(split.validation_names):
            print(f"holdout flip-4 inference: {index}/{len(split.validation_names)}", flush=True)
    inference_seconds = time.perf_counter() - inference_started
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_gpu_memory_bytes = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_gpu_memory_bytes = 0

    scoring_started = time.perf_counter()
    metrics, observation_rows = score_validation(
        dataset,
        predictions,
        split.validation_names,
        iou_threshold=float(baseline["iou_threshold"]),
        diagnostic_overlap_threshold=float(baseline["diagnostic_overlap_threshold"]),
    )
    scoring_seconds = time.perf_counter() - scoring_started
    _write_predictions(output_dir / "predictions.csv", predictions)
    _write_observation_scores(output_dir / "per_observation.csv", observation_rows, predictions)
    with (output_dir / "tta-diagnostics.csv").open("w", encoding="utf-8", newline="") as stream:
        import csv

        writer = csv.DictWriter(
            stream,
            fieldnames=("file_name", "accepted", "rejected", "rejection_reasons"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(diagnostic_rows)
    overlays = write_comparison_overlays(
        dataset, predictions, output_dir / "comparison-overlays", maximum=12
    )

    report = {
        "status": "complete",
        "run_name": RUN_NAME,
        "method": "ResNet34 crop refiner with public topology-safe flip-4 probability TTA",
        "source_commit": source_metrics["source_commit"],
        "source_checkpoint_sha256": source_metrics["training"]["checkpoint_sha256"],
        "source_metrics_sha256": _sha256(source_metrics_path),
        "selected_screen_metrics_sha256": _sha256(threshold_metrics_path),
        "validation_detections_sha256": _sha256(validation_detection_path),
        "refiner_code_sha256": _sha256(repo_root / "src" / "filament_baseline" / "seed_refiner.py"),
        "data_boundary": {
            "training_observations": 565,
            "holdout_observations": 142,
            "frozen_holdout_used_for_training": False,
            "test_observations_used": False,
            "test_labels_used": False,
            "training_labels": "competition-provided annotations for the frozen training partition only",
        },
        "split_sha256": split.sha256,
        "scorer": {
            "iou_threshold": float(baseline["iou_threshold"]),
            "diagnostic_overlap_threshold": float(baseline["diagnostic_overlap_threshold"]),
        },
        "configuration": {
            **source_config,
            "flip_transforms": ["identity", "horizontal", "vertical", "horizontal+vertical"],
            "topology_guard": {
                "area_ratio_min": 0.60,
                "area_ratio_max": 1.70,
                "max_component_increase": 1,
                "minimum_single_area_for_dice": 32,
                "minimum_mask_dice": 0.62,
            },
        },
        "tta_diagnostics": dict(sorted(diagnostics.items())),
        "metrics": metrics,
        "observation_count": len(observation_rows),
        "overlays": overlays,
        "baseline_comparison": {
            "run_name": SOURCE_RUN_NAME,
            "metrics_path": str(threshold_metrics_path.relative_to(repo_root)),
            "global": selected_screen["metrics"]["global"],
            "candidate_is_holdout_screened": True,
        },
        "runtime": {
            **hardware_info(),
            "package_versions": package_versions(),
            "device": str(device),
            "inference_seconds": inference_seconds,
            "scoring_seconds": scoring_seconds,
            "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "run_name": RUN_NAME,
                "global": metrics["global"],
                "tta_diagnostics": dict(sorted(diagnostics.items())),
                "inference_seconds": inference_seconds,
                "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
