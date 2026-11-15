# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Reproduce the audited three-family test-time consensus from packaged weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from filament_baseline.runtime import hardware_info, package_versions  # noqa: E402

EXPECTED_SPLIT = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"
HDJOJO_MODEL = "yolo26x-seg-solar-filament-2048-1109.pt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def require_under(path: Path, roots: tuple[Path, ...], label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not any(resolved.is_relative_to(root.resolve()) for root in roots):
        raise ValueError(f"{label} must stay under one of {roots}: {resolved}")
    return resolved


def make_candidate_manifest(
    staging: Path,
    component_paths: list[tuple[str, Path]],
) -> Path:
    candidates: list[dict[str, Any]] = []
    for index, (family, source) in enumerate(component_paths, start=1):
        destination = staging / f"family-{index}.csv"
        shutil.copy2(source, destination)
        candidates.append(
            {
                "message": f"Final reproduction component {index}: {family}",
                "path": destination.relative_to(ROOT).as_posix(),
                "sha256": sha256(destination),
                "validation_protocol": "Test-only inference on supplied competition H-alpha images; no test labels were read.",
            }
        )
    manifest = {
        "competition": "filament-segmentation-2026",
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "split_sha256": EXPECTED_SPLIT,
        "candidates": candidates,
    }
    path = staging / "component-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--weights-root", type=Path, default=ROOT / "data/reference/final-weights"
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "artifacts/final-reproduction"
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--expected-sha256", default=None)
    args = parser.parse_args()

    data_roots = [ROOT / "data"]
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        data_roots.append(kaggle_input)
    data_root = require_under(args.data_root, tuple(data_roots), "competition data")
    weights_root = require_under(
        args.weights_root,
        (ROOT / "data" / "reference",),
        "packaged weights",
    )
    output_root = require_under(args.output_root, (ROOT / "artifacts",), "run outputs")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty run directory: {output_root}")
    if not (data_root / "test" / "test_images").is_dir():
        raise FileNotFoundError(data_root / "test" / "test_images")
    if not (weights_root / "unet" / "solar_unet_receipt.json").is_file():
        raise FileNotFoundError(weights_root / "unet" / "solar_unet_receipt.json")
    if not (weights_root / "equinexus-yolo" / "run-metadata.json").is_file():
        raise FileNotFoundError(weights_root / "equinexus-yolo" / "run-metadata.json")
    if not (weights_root / "hdjojo" / HDJOJO_MODEL).is_file():
        raise FileNotFoundError(weights_root / "hdjojo" / HDJOJO_MODEL)
    if not (weights_root / "i079" / "final_detector.pt").is_file():
        raise FileNotFoundError(weights_root / "i079" / "final_detector.pt")

    staging = ROOT / "artifacts" / "submissions" / "final-consensus-reproduction-inputs"
    consensus_output = ROOT / "artifacts" / "submissions" / "final-consensus-reproduction-output"
    for path in (staging, consensus_output):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing path: {path}")
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    equinexus_root = output_root / "equinexus-family-vote"
    run(
        [
            sys.executable,
            "scripts/reproduce_final_family_vote.py",
            "--data-root",
            str(data_root),
            "--weights-root",
            str(weights_root),
            "--output-root",
            str(equinexus_root),
            "--confidence",
            "0.38",
            "--device",
            args.device,
        ]
    )

    hdjojo_root = output_root / "hdjojo-yolo26x"
    run(
        [
            sys.executable,
            "scripts/run_hdjojo_yolo26x_test_inference.py",
            "--data-root",
            str(data_root),
            "--model-path",
            str(weights_root / "hdjojo" / HDJOJO_MODEL),
            "--output-dir",
            str(hdjojo_root),
            "--device",
            args.device,
            "--match-public-kernel",
        ]
    )

    i079_root = output_root / "i079-topology-safe"
    run(
        [
            sys.executable,
            "scripts/run_i079_topology_safe_inference.py",
            "--data-root",
            str(data_root),
            "--weights-root",
            str(weights_root / "i079"),
            "--output-dir",
            str(i079_root),
            "--allow-output-drift",
        ]
    )

    staging.mkdir(parents=True)
    family_sources = [
        ("EquiNexus U-Net+YOLO11s", equinexus_root / "family-vote" / "submission.csv"),
        ("HDJoJo YOLO26x", hdjojo_root / "submission.csv"),
        ("I079 Mask R-CNN+refiner", i079_root / "submission.csv"),
    ]
    for family, source in family_sources:
        if not source.is_file():
            raise FileNotFoundError(f"{family} inference did not produce {source}")
    manifest_path = make_candidate_manifest(staging, family_sources)
    run(
        [
            sys.executable,
            "scripts/experiments/build_rule_compliant_test_consensus.py",
            "--manifest",
            str(manifest_path),
            "--source-indices",
            "1",
            "2",
            "3",
            "--families",
            "EquiNexus U-Net+YOLO11s",
            "HDJoJo YOLO26x",
            "I079 Mask R-CNN+refiner",
            "--output-dir",
            str(consensus_output),
            "--test-image-dir",
            str(data_root / "test" / "test_images"),
        ]
    )
    submission = consensus_output / "submission.csv"
    submission_hash = sha256(submission)
    if args.expected_sha256 and submission_hash != args.expected_sha256:
        raise RuntimeError(
            f"Reproduced CSV hash differs: {submission_hash} != {args.expected_sha256}"
        )
    final_csv = output_root / "submission.csv"
    shutil.copy2(submission, final_csv)
    validation_path = output_root / "rle-validation.json"
    run(
        [
            sys.executable,
            "scripts/validate_submission_rles.py",
            str(final_csv),
            "--test-image-dir",
            str(data_root / "test" / "test_images"),
            "--expected-test-images",
            "180",
            "--output-json",
            str(validation_path),
        ]
    )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("csv_sha256") != submission_hash:
        raise RuntimeError("RLE validator recorded a different submission hash")
    if validation.get("processed_test_images") != 180:
        raise RuntimeError("RLE validator did not cover all supplied test images")

    metadata = {
        "status": "complete",
        "solution": "label-free vote across EquiNexus U-Net+YOLO11s, HDJoJo YOLO26x, and I079 Mask R-CNN+refiner",
        "split_sha256": EXPECTED_SPLIT,
        "test_labels_used": False,
        "inference_inputs": "The 180 supplied competition H-alpha test images.",
        "component_submission_sha256": {
            family: sha256(source) for family, source in family_sources
        },
        "source_script_sha256": sha256(Path(__file__).resolve()),
        "weights_manifest": str((weights_root / "weights-manifest.json").relative_to(ROOT)),
        "weights_manifest_sha256": sha256(weights_root / "weights-manifest.json"),
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "submission_path": str(final_csv.relative_to(ROOT)).replace("\\", "/"),
        "submission_sha256": submission_hash,
        "rle_validation": validation,
        "hardware": hardware_info(),
        "package_versions": package_versions(),
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    (output_root / "reproduction-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
