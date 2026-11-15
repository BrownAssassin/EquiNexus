# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Reproduce the U-Net + YOLO11s final candidate from packaged weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from filament_baseline.runtime import hardware_info, package_versions  # noqa: E402

EXPECTED_SPLIT = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--weights-root", type=Path, default=ROOT / "data/reference/final-weights"
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "artifacts/final-reproduction"
    )
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    parser.add_argument("--expected-sha256", default=None)
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    weights_root = args.weights_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    for label, path, allowed_root in (
        ("packaged weights", weights_root, ROOT / "data/reference"),
        ("run outputs", output_root, ROOT / "artifacts"),
    ):
        if not path.is_relative_to(allowed_root.resolve()):
            raise ValueError(f"{label} must remain under {allowed_root}: {path}")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty run directory: {output_root}")
    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError("confidence must be between zero and one")
    if not data_root.is_dir():
        raise NotADirectoryError(data_root)
    if not (ROOT / "configs/validation-baseline.json").is_file():
        raise FileNotFoundError("The frozen split configuration is missing")
    config = json.loads((ROOT / "configs/validation-baseline.json").read_text())
    if config.get("expected_split_sha256") != EXPECTED_SPLIT:
        raise RuntimeError("The frozen observation split digest changed")
    config_path = ROOT / "configs/validation-baseline.json"

    unet_root = weights_root / "unet"
    detector_root = weights_root / "yolo"
    if not (detector_root / "run-metadata.json").is_file():
        detector_root = weights_root / "equinexus-yolo"
    if not (unet_root / "solar_unet_receipt.json").is_file():
        raise FileNotFoundError(unet_root / "solar_unet_receipt.json")
    if not (detector_root / "run-metadata.json").is_file():
        raise FileNotFoundError(detector_root / "run-metadata.json")

    output_root.mkdir(parents=True, exist_ok=True)
    unet_output = output_root / "unet"
    yolo_output = output_root / "yolo"
    dedup_output = yolo_output / "dedup-iou-030"
    fusion_output = output_root / "family-vote"

    started = time.perf_counter()
    run(
        [
            sys.executable,
            "scripts/run_public_unet_ensemble.py",
            "--data-root",
            str(data_root),
            "--weights-root",
            str(unet_root),
            "--output-root",
            str(unet_output),
            "--device",
            "cuda:0" if args.device != "cpu" else "cpu",
            "--tta-views",
            "8",
            "--views-per-pass",
            "2",
        ]
    )
    run(
        [
            sys.executable,
            "scripts/reinfer_yolo_test_tta.py",
            "--repo-root",
            str(ROOT),
            "--data-root",
            str(data_root),
            "--detector-run-root",
            str(detector_root),
            "--output-dir",
            str(yolo_output),
            "--image-size",
            "1280",
            "--confidence",
            "0.05",
            "--nms-iou",
            "0.6",
            "--max-detections",
            "150",
            "--device",
            args.device,
        ]
    )
    run(
        [
            sys.executable,
            "scripts/deduplicate_yolo_tta_oof.py",
            "--predictions",
            str(yolo_output / "test_detections_tta8_raw.json"),
            "--output-dir",
            str(dedup_output),
            "--mask-iou-threshold",
            "0.3",
        ]
    )
    run(
        [
            sys.executable,
            "scripts/evaluate_public_unet_yolo_family_fusion.py",
            "--data-root",
            str(data_root),
            "--unet-predictions",
            str(unet_output / "submission.csv"),
            "--detector-predictions",
            str(dedup_output / "test_detections_deduplicated.json"),
            "--partition",
            "test",
            "--output-dir",
            str(fusion_output),
            "--detector-confidence",
            f"{args.confidence:.2f}",
            "--minimum-family-votes",
            "2",
            "--skip-overlays",
            "--skip-paired-bootstrap",
        ]
    )
    submission = fusion_output / "submission.csv"
    if not submission.is_file():
        raise FileNotFoundError(submission)
    submission_hash = sha256(submission)
    if args.expected_sha256 and submission_hash != args.expected_sha256:
        raise RuntimeError(
            f"Reproduced CSV hash differs: {submission_hash} != {args.expected_sha256}"
        )

    validator_output = output_root / "rle-validation.json"
    run(
        [
            sys.executable,
            "scripts/validate_submission_rles.py",
            str(submission),
            "--test-image-dir",
            str(data_root / "test" / "test_images"),
            "--expected-test-images",
            "180",
            "--output-json",
            str(validator_output),
        ]
    )
    validation = json.loads(validator_output.read_text(encoding="utf-8"))
    if validation.get("csv_sha256") != submission_hash:
        raise RuntimeError("RLE validator recorded a different submission hash")
    if validation.get("processed_test_images") != 180:
        raise RuntimeError("RLE validator did not cover all 180 test images")
    unet_metadata_path = unet_output / "run-metadata.json"
    yolo_metadata_path = yolo_output / "run-metadata.json"
    unet_metadata = json.loads(unet_metadata_path.read_text(encoding="utf-8"))
    yolo_metadata = json.loads(yolo_metadata_path.read_text(encoding="utf-8"))
    yolo_runtime = yolo_metadata.get("runtime", {})
    component_peak_memory = [
        int(unet_metadata.get("peak_gpu_memory_bytes", 0)),
        int(yolo_runtime.get("peak_gpu_memory_bytes", 0)),
    ]
    metadata = {
        "status": "complete",
        "solution": "public 10-model U-Net + five-fold YOLO11s family vote",
        "confidence_threshold": args.confidence,
        "minimum_family_votes": 2,
        "support_iou_strictly_greater_than": 0.5,
        "minimum_retained_pixel_fraction": 0.5,
        "split_sha256": EXPECTED_SPLIT,
        "test_labels_used": False,
        "inference_inputs": "The 180 supplied competition H-alpha test images.",
        "source_script_sha256": sha256(Path(__file__).resolve()),
        "component_run_metadata": {
            "unet": {
                "path": str(unet_metadata_path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(unet_metadata_path),
                "runtime_seconds": unet_metadata.get("runtime_seconds"),
                "peak_gpu_memory_bytes": unet_metadata.get("peak_gpu_memory_bytes"),
            },
            "yolo": {
                "path": str(yolo_metadata_path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(yolo_metadata_path),
                "runtime_seconds": yolo_runtime.get("prediction_seconds"),
                "peak_gpu_memory_bytes": yolo_runtime.get("peak_gpu_memory_bytes"),
            },
        },
        "peak_gpu_memory_bytes": max(component_peak_memory),
        "configuration_sha256": sha256(config_path),
        "weights_manifest": str((weights_root / "weights-manifest.json").relative_to(ROOT)),
        "weights_manifest_sha256": sha256(weights_root / "weights-manifest.json"),
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "submission_path": str(submission.relative_to(ROOT)).replace("\\", "/"),
        "submission_sha256": submission_hash,
        "rle_validation": validation,
        "hardware": hardware_info(),
        "package_versions": package_versions(),
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    metadata_path = output_root / "reproduction-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
