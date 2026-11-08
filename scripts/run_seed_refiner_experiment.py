# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

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
    _build_training_samples,
    load_seed_refiner,
    predicted_instances,
    train_seed_refiner,
)
from filament_baseline.split import create_observation_split  # noqa: E402
from filament_baseline.visuals import write_comparison_overlays  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(repo_root: Path) -> str | None:
    import subprocess

    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train an OOF-seeded crop refiner and score it on the frozen holdout."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--train-detector-predictions", type=Path, required=True)
    parser.add_argument("--validation-detector-predictions", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--validation-score-threshold", type=float, default=0.25)
    parser.add_argument("--probability-threshold", type=float, default=0.50)
    parser.add_argument("--minimum-area", type=int, default=200)
    parser.add_argument("--box-jitter-tta", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.expanduser().resolve()
    data_root = (
        args.data_root.expanduser().resolve()
        if args.data_root is not None
        else repo_root / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026"
    )
    train_detections_path = args.train_detector_predictions.expanduser().resolve()
    validation_detections_path = args.validation_detector_predictions.expanduser().resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", args.run_name):
        parser.error("run-name must use only letters, digits, dot, underscore, or dash")
    for path in (data_root, train_detections_path, validation_detections_path):
        if not path.is_relative_to(repo_root):
            parser.error(f"Local data and detector predictions must stay in the repository: {path}")
    for path in (train_detections_path, validation_detections_path):
        if not path.is_file():
            parser.error(f"Detector-prediction file does not exist: {path}")
    output_dir = repo_root / "artifacts" / "model-experiments" / args.run_name
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"Run output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.hub.set_dir(str(repo_root / "artifacts" / "pretrained" / "torch" / "hub"))
    config_path = repo_root / "configs" / "validation-baseline.json"
    baseline_config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset = load_dataset(data_root)
    split = create_observation_split(
        dataset,
        repo_root / "data" / "splits",
        seed=int(baseline_config["seed"]),
        validation_fraction=float(baseline_config["validation_fraction"]),
    )
    if split.sha256 != baseline_config["expected_split_sha256"]:
        raise RuntimeError("The seed-refiner run requires the frozen observation split")

    train_detections = json.loads(train_detections_path.read_text(encoding="utf-8"))
    validation_detections = json.loads(validation_detections_path.read_text(encoding="utf-8"))
    train_stems = {Path(name).stem for name in split.training_names}
    validation_stems = {Path(name).stem for name in split.validation_names}
    if not train_stems.issubset(train_detections):
        raise ValueError("OOF detector predictions do not cover the entire split training partition")
    if not validation_stems.issubset(validation_detections):
        raise ValueError("Validation detector predictions do not cover the frozen holdout")
    if train_stems & validation_stems:
        raise AssertionError("Training and validation observations overlap")

    config = SeedRefinerConfig(
        crop_size=args.crop_size,
        training_steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_score_min=args.validation_score_threshold,
        probability_threshold=args.probability_threshold,
        minimum_instance_area=args.minimum_area,
    )
    samples = _build_training_samples(
        dataset,
        split.training_names,
        train_detections,
        config,
    )
    print(
        json.dumps(
            {
                "training_observations": len(split.training_names),
                "validation_observations": len(split.validation_names),
                "annotator_view_crop_samples": len(samples),
                "matched_crop_samples": sum(
                    sample.target_segmentation is not None for sample in samples
                ),
                "split_sha256": split.sha256,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hardware = hardware_info()
    started = time.perf_counter()
    checkpoint_path = output_dir / "seed_refiner.pt"
    training_report = train_seed_refiner(
        dataset,
        samples,
        config,
        checkpoint_path,
        device=device,
        num_workers=args.workers,
    )
    model, _checkpoint = load_seed_refiner(checkpoint_path, device)

    predictions = {}
    for index, file_name in enumerate(split.validation_names, start=1):
        image = read_grayscale(dataset.train_image_dir / file_name)
        predictions[file_name] = predicted_instances(
            file_name,
            image,
            validation_detections.get(Path(file_name).stem, []),
            model,
            config,
            device=device,
            box_jitter=args.box_jitter_tta,
        )
        if index % 25 == 0 or index == len(split.validation_names):
            print(f"Predicted {index}/{len(split.validation_names)} validation observations", flush=True)

    metrics, observation_rows = score_validation(
        dataset,
        predictions,
        split.validation_names,
        iou_threshold=float(baseline_config["iou_threshold"]),
        diagnostic_overlap_threshold=float(baseline_config["diagnostic_overlap_threshold"]),
    )
    predictions_path = output_dir / "predictions.csv"
    observations_path = output_dir / "per_observation.csv"
    metrics_path = output_dir / "metrics.json"
    _write_predictions(predictions_path, predictions)
    _write_observation_scores(observations_path, observation_rows, predictions)
    overlays = write_comparison_overlays(
        dataset, predictions, output_dir / "comparison-overlays", maximum=12
    )
    result = {
        "run_name": args.run_name,
        "source_commit": _git_commit(repo_root),
        "training_code_sha256": _sha256(Path(__file__).resolve()),
        "seed_refiner_code_sha256": _sha256(
            repo_root / "src" / "filament_baseline" / "seed_refiner.py"
        ),
        "split_sha256": split.sha256,
        "split_manifest": str(split.manifest_path.relative_to(repo_root)),
        "official_scorer": {
            "iou_threshold": float(baseline_config["iou_threshold"]),
            "diagnostic_overlap_threshold": float(
                baseline_config["diagnostic_overlap_threshold"]
            ),
        },
        "training_observations": len(split.training_names),
        "validation_observations": len(split.validation_names),
        "training_labels": "competition-provided annotations only",
        "training_detection_source": str(train_detections_path.relative_to(repo_root)),
        "training_detection_sha256": _sha256(train_detections_path),
        "validation_detection_source": str(validation_detections_path.relative_to(repo_root)),
        "validation_detection_sha256": _sha256(validation_detections_path),
        "target_match_min_iou": config.match_iou_min,
        "training_crop_samples": len(samples),
        "matched_crop_samples": sum(sample.target_segmentation is not None for sample in samples),
        "configuration": {**asdict(config), "box_jitter_tta": args.box_jitter_tta},
        "training": training_report,
        "metrics": metrics,
        "per_observation_csv": str(observations_path.relative_to(repo_root)),
        "prediction_csv": str(predictions_path.relative_to(repo_root)),
        "overlays": overlays,
        "submission_test_ground_truth_used": False,
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
        "hardware": hardware,
        "package_versions": package_versions(),
    }
    metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "run-metadata.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics["global"], indent=2, sort_keys=True), flush=True)
    print(f"Metrics written to {metrics_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
