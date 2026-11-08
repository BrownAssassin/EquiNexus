# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from filament_baseline.data import IMAGE_SIZE, inventory, load_dataset
from filament_baseline.detector_fusion import DetectorFusionConfig, fuse_detector_instances
from filament_baseline.predictor import PredictedInstance, encode_binary_mask
from filament_baseline.runner import _git_status, _write_predictions
from filament_baseline.runtime import hardware_info, package_versions
from filament_baseline.submission import _check_submission


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_tree_sha256(repo_root: Path) -> tuple[str, dict[str, str]]:
    paths = [
        *sorted((repo_root / "src" / "filament_baseline").glob("*.py")),
        repo_root / "scripts" / "reinfer_yolo_oof_tta.py",
        repo_root / "scripts" / "reinfer_yolo_test_tta.py",
        repo_root / "scripts" / "deduplicate_yolo_tta_oof.py",
        repo_root / "scripts" / "build_yolo_fused_submission.py",
    ]
    file_hashes = {}
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(repo_root).as_posix()
        content_hash = _sha256(path)
        file_hashes[relative] = content_hash
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest(), file_hashes


def _prediction_instances(labels: np.ndarray, file_name: str) -> list[PredictedInstance]:
    stem = Path(file_name).stem
    output = []
    for label in range(1, int(labels.max()) + 1):
        binary = labels == label
        area = int(binary.sum())
        if area:
            output.append(
                PredictedInstance(
                    filament_id=f"{stem}_{len(output) + 1}",
                    image_stem=stem,
                    area=area,
                    rle=encode_binary_mask(binary),
                )
            )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fuse deduplicated YOLO test instances with the final two-member U-Net probability blend."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--unet-run-dir", type=Path, required=True)
    parser.add_argument("--detector-predictions", type=Path, required=True)
    parser.add_argument("--detector-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--first-member", default="annotator1024")
    parser.add_argument("--second-member", default="consensus512")
    parser.add_argument("--first-weight", type=float, default=0.5)
    parser.add_argument("--foreground-threshold", type=float, default=0.5)
    parser.add_argument("--detection-threshold", type=float, default=0.35)
    parser.add_argument("--minimum-claimed-area", type=int, default=200)
    parser.add_argument("--closing-radius", type=int, default=3)
    parser.add_argument("--solar-radius", type=int, default=950)
    parser.add_argument("--mask-nms-iou", type=float, default=0.30)
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", args.output_dir.name):
        raise ValueError("Output directory name must use only letters, digits, dot, underscore, or dash")
    if not 0.0 <= args.first_weight <= 1.0:
        raise ValueError("first-weight must be between 0 and 1")
    repo_root = args.repo_root.expanduser().resolve()
    data_root = (
        args.data_root or repo_root / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026"
    ).expanduser().resolve()
    unet_run = args.unet_run_dir.expanduser().resolve()
    detector_predictions_path = args.detector_predictions.expanduser().resolve()
    detector_metadata_path = args.detector_metadata.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for label, path, allowed_root in (
        ("competition data", data_root, repo_root / "data"),
        ("U-Net artifacts", unet_run, repo_root / "artifacts"),
        ("detector predictions", detector_predictions_path, repo_root / "artifacts"),
        ("detector metadata", detector_metadata_path, repo_root / "artifacts"),
        ("submission output", output_dir, repo_root / "artifacts"),
    ):
        if not path.is_relative_to(allowed_root.resolve()):
            raise ValueError(f"{label} must remain under {allowed_root}: {path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    dataset = load_dataset(data_root)
    data_inventory = inventory(dataset)
    if data_inventory["observation_count"] != 707 or data_inventory["annotator_view_count"] != 1154:
        raise ValueError("Unexpected competition training inventory")
    if data_inventory["annotation_count"] != 8199 or data_inventory["test_image_file_count"] != 180:
        raise ValueError("Unexpected annotation or test-image count")
    test_names = sorted(path.name for path in dataset.test_image_dir.iterdir() if path.is_file())
    if len(test_names) != 180 or len({Path(name).stem for name in test_names}) != 180:
        raise ValueError("Expected 180 unique supplied H-alpha test images")

    unet_metadata_path = unet_run / "metrics.json"
    unet_metadata = json.loads(unet_metadata_path.read_text(encoding="utf-8"))
    if unet_metadata.get("status") != "complete":
        raise ValueError("Final U-Net ensemble training did not complete")
    if unet_metadata.get("dataset_inventory", {}).get("observation_count") != 707:
        raise ValueError("Final U-Net ensemble was not trained against all supplied observations")
    expected_member_names = {args.first_member, args.second_member}
    member_results = {item["name"]: item for item in unet_metadata.get("members", [])}
    if set(member_results) != expected_member_names:
        raise ValueError("Final U-Net run is missing the requested ensemble members")
    expected_members = {
        args.first_member: ("annotator", 1024, 10000, 8),
        args.second_member: ("consensus", 512, 5000, 4),
    }
    for member_name, (target_mode, patch_size, steps, tta) in expected_members.items():
        training = member_results[member_name]
        member_config = training.get("configuration", {})
        if (
            member_config.get("target_mode") != target_mode
            or member_config.get("patch_size") != patch_size
            or member_config.get("steps") != steps
            or member_config.get("tta") != tta
        ):
            raise ValueError(f"Unexpected validated U-Net configuration for {member_name}")
        if member_config.get("external_supervised_weights") is not False:
            raise ValueError("U-Net members must not use external supervised weights")

    detector_metadata = json.loads(detector_metadata_path.read_text(encoding="utf-8"))
    if detector_metadata.get("status") != "complete":
        raise ValueError("YOLO mask deduplication did not complete")
    if detector_metadata.get("partition") != "competition_test_inference":
        raise ValueError("Detector predictions are not marked as supplied test inference")
    raw_predictions_path = (repo_root / detector_metadata["source_predictions"]).resolve()
    test_inference_metadata_path = raw_predictions_path.parent / "run-metadata.json"
    test_inference_metadata = json.loads(test_inference_metadata_path.read_text(encoding="utf-8"))
    if _sha256(raw_predictions_path) != detector_metadata.get("source_predictions_sha256"):
        raise ValueError("Deduplicated predictions do not match the recorded raw test predictions")
    if _sha256(test_inference_metadata_path) != detector_metadata.get("source_run_metadata_sha256"):
        raise ValueError("Deduplicated predictions do not match their test-inference metadata")
    if (
        test_inference_metadata.get("partition") != "competition_test_inference"
        or test_inference_metadata.get("test_labels_used") is not False
        or test_inference_metadata.get("test_images_used_for_training") is not False
        or test_inference_metadata.get("test_observations") != 180
        or test_inference_metadata.get("prediction_sha256") != _sha256(raw_predictions_path)
    ):
        raise ValueError("Test labels or images were marked as training inputs")
    if not np.isclose(args.mask_nms_iou, 0.30):
        raise ValueError("This candidate is validated only with mask NMS IoU 0.30")
    if not np.isclose(float(detector_metadata.get("mask_iou_suppression_threshold", -1.0)), args.mask_nms_iou):
        raise ValueError("Deduplicated predictions use a different mask-NMS threshold")
    detector_predictions = json.loads(detector_predictions_path.read_text(encoding="utf-8"))
    test_stems = {Path(name).stem for name in test_names}
    if set(detector_predictions) != test_stems:
        raise ValueError("Deduplicated detector predictions do not exactly cover the test set")

    config = DetectorFusionConfig(
        foreground_threshold=args.foreground_threshold,
        detection_threshold=args.detection_threshold,
        grow_radius=0,
        minimum_claimed_area=args.minimum_claimed_area,
        closing_radius=args.closing_radius,
        solar_radius=args.solar_radius,
    )
    first_cache = unet_run / "members" / args.first_member / "probabilities"
    second_cache = unet_run / "members" / args.second_member / "probabilities"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions: dict[str, list[PredictedInstance]] = {}
    coverage_rows = []
    started = time.perf_counter()
    for index, file_name in enumerate(test_names, start=1):
        stem = Path(file_name).stem
        first_path = first_cache / f"{stem}.npz"
        second_path = second_cache / f"{stem}.npz"
        if not first_path.is_file() or not second_path.is_file():
            raise FileNotFoundError(f"Missing U-Net test probability cache for {file_name}")
        with np.load(first_path) as first, np.load(second_path) as second:
            first_foreground = first["foreground"].astype(np.float32)
            second_foreground = second["foreground"].astype(np.float32)
        if first_foreground.shape != (IMAGE_SIZE, IMAGE_SIZE) or second_foreground.shape != first_foreground.shape:
            raise ValueError(f"Invalid U-Net probability dimensions for {file_name}")
        foreground = (
            args.first_weight * first_foreground
            + (1.0 - args.first_weight) * second_foreground
        )
        labels = fuse_detector_instances(
            foreground,
            detector_predictions[stem],
            config,
        )
        predictions[file_name] = _prediction_instances(labels, file_name)
        coverage_rows.append({"file_name": file_name, "instance_count": len(predictions[file_name])})
        if index % 20 == 0 or index == len(test_names):
            print(f"Fused {index}/{len(test_names)} test images", flush=True)

    submission_path = output_dir / "submission.csv"
    _write_predictions(submission_path, predictions)
    submission_validation = _check_submission(submission_path, dataset.root, test_names, predictions)
    with (output_dir / "test-coverage.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("file_name", "instance_count"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(coverage_rows)

    repository = _git_status(repo_root)
    source_tree_sha256, source_file_hashes = _source_tree_sha256(repo_root)
    if os_commit := os.environ.get("EQUINEXUS_SOURCE_COMMIT"):
        repository["commit"] = os_commit
        repository["worktree_dirty"] = True
    metadata = {
        "status": "complete",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_name": output_dir.name,
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "source_tree_sha256": source_tree_sha256,
        "source_file_sha256": source_file_hashes,
        "repository": repository,
        "data_boundary": {
            "training_labels": "competition-provided training annotations only",
            "inference_inputs": "180 supplied competition H-alpha test images only",
            "test_labels_used": False,
            "test_images_used_for_training": False,
            "external_ground_truth": False,
        },
        "dataset_inventory": data_inventory,
        "validation_reference": {
            "frozen_split_sha256": detector_metadata["frozen_holdout_sha256"],
            "oof_manifest_sha256": detector_metadata["oof_manifest_sha256"],
            "holdout_pq": 0.4097760614657074,
            "oof_pq": 0.4086201997023809,
            "selection_note": "Fixed deduplicated detector/U-Net configuration confirmed once on the observation-grouped holdout before test inference.",
        },
        "fusion": {
            "first_member": args.first_member,
            "second_member": args.second_member,
            "first_member_weight": args.first_weight,
            "second_member_weight": 1.0 - args.first_weight,
            "configuration": config.__dict__,
            "yolo_mask_nms_iou": args.mask_nms_iou,
            "tta": "eight rotations/reflections",
        },
        "provenance": {
            "unet_metrics_sha256": _sha256(unet_metadata_path),
            "unet_members": member_results,
            "detector_metadata_sha256": _sha256(detector_metadata_path),
            "test_inference_metadata_sha256": _sha256(test_inference_metadata_path),
            "detector_predictions_sha256": _sha256(detector_predictions_path),
            "detector_checkpoint_hashes": [
                item["checkpoint_sha256"] for item in test_inference_metadata["folds"]
            ],
        },
        "test_validation": submission_validation,
        "runtime": {
            **hardware_info(),
            "postprocess_seconds": round(time.perf_counter() - started, 3),
            "package_versions": package_versions(),
        },
        "artifacts": {
            "submission": submission_path.name,
            "coverage": "test-coverage.csv",
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", **submission_validation, "submission": str(submission_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
