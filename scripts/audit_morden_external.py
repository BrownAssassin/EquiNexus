# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from filament_baseline.data import load_dataset
from filament_baseline.external_data import (
    MORDEN_ARCHIVE_MD5,
    MORDEN_ARCHIVE_SHA256,
    MORDEN_DOI,
    MORDEN_IMAGE_PATTERN,
)
from filament_baseline.split import create_observation_split


SOURCE_ROOT = "https://zenodo.org/records/21582245"
REPOSITORY_URL = "https://github.com/irisaltHu/MORDEN"
PAPER_URL = "https://arxiv.org/abs/2607.24525"
SOURCE_CITATION = (
    "Hu, J. et al. A Modern ConvNet for Solar Filament Detection. arXiv:2607.24525 (2026)."
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def audit(
    *, repo_root: Path, competition_data_root: Path, external_root: Path
) -> dict[str, Any]:
    if not _inside(competition_data_root, repo_root / "data"):
        raise ValueError(f"Competition data must stay under data/: {competition_data_root}")
    if not _inside(external_root, repo_root / "data" / "external"):
        raise ValueError(f"External data must stay under data/external/: {external_root}")

    archive_path = external_root / "data.zip"
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    archive_sha256 = _hash_file(archive_path, "sha256")
    archive_md5 = _hash_file(archive_path, "md5")
    if archive_sha256 != MORDEN_ARCHIVE_SHA256 or archive_md5 != MORDEN_ARCHIVE_MD5:
        raise ValueError("MORDEN archive checksum does not match the verified Zenodo record")

    competition = load_dataset(competition_data_root)
    split = create_observation_split(
        competition, repo_root / "data" / "splits", seed=42, validation_fraction=0.2
    )
    expected_split = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"
    if split.sha256 != expected_split:
        raise RuntimeError(f"Frozen competition split changed: {split.sha256}")

    source_train = external_root / "extracted" / "AHAS" / "train"
    image_dir, mask_dir = source_train / "images", source_train / "masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError("Extract MORDEN data.zip under extracted/ before auditing")
    all_competition_files = sorted(
        path
        for folder in (competition.train_image_dir, competition.test_image_dir)
        for path in folder.iterdir()
        if path.is_file()
    )
    comp_bytes_hashes: set[str] = set()
    comp_pixel_hashes: set[str] = set()
    comp_names = {path.name for path in all_competition_files}
    comp_dates = {path.name[:8] for path in all_competition_files}
    for path in all_competition_files:
        payload = path.read_bytes()
        comp_bytes_hashes.add(_sha256_bytes(payload))
        image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not decode competition image: {path}")
        if image.shape != (2048, 2048):
            image = cv2.resize(image, (2048, 2048), interpolation=cv2.INTER_AREA)
        comp_pixel_hashes.add(_sha256_bytes(np.ascontiguousarray(image).tobytes()))

    dated_names = sorted(
        path.name
        for path in image_dir.iterdir()
        if path.is_file() and MORDEN_IMAGE_PATTERN.fullmatch(path.name)
    )
    observations: list[dict[str, Any]] = []
    external_bytes_hashes: set[str] = set()
    external_pixel_hashes: set[str] = set()
    annotation_count = 0
    dates: set[str] = set()
    for file_name in dated_names:
        image_path = image_dir / file_name
        mask_path = mask_dir / file_name
        if not mask_path.is_file():
            raise FileNotFoundError(f"MORDEN image has no paired mask: {file_name}")
        date = MORDEN_IMAGE_PATTERN.fullmatch(file_name).group(1)
        if date[:4] not in {"2024", "2025"}:
            raise ValueError(f"Unexpected source year in {file_name}")
        if date in comp_dates:
            raise ValueError(f"MORDEN observation date overlaps the competition: {date}")
        if file_name in comp_names:
            raise ValueError(f"MORDEN source filename overlaps a competition image: {file_name}")
        dates.add(date)

        image_bytes, mask_bytes = image_path.read_bytes(), mask_path.read_bytes()
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None or mask.shape != (2048, 2048):
            raise ValueError(f"Invalid image-mask pair in MORDEN AHAS train: {file_name}")
        if min(image.shape) < 1024 or max(image.shape) > 4096:
            raise ValueError(f"Unexpected source image dimensions for {file_name}: {image.shape}")
        if max(image.shape) / min(image.shape) > 1.02:
            raise ValueError(f"Unexpected source image aspect ratio for {file_name}: {image.shape}")
        normalized = image if image.shape == (2048, 2048) else cv2.resize(
            image, (2048, 2048), interpolation=cv2.INTER_AREA
        )
        binary = (mask > 127).astype(np.uint8)
        count, _ = cv2.connectedComponents(binary, connectivity=8)
        annotation_count += max(count - 1, 0)
        image_sha256 = _sha256_bytes(image_bytes)
        mask_sha256 = _sha256_bytes(mask_bytes)
        external_bytes_hashes.add(image_sha256)
        external_pixel_hashes.add(
            _sha256_bytes(np.ascontiguousarray(normalized).tobytes())
        )
        observations.append(
            {
                "date": date,
                "file_name": file_name,
                "image_shape": [int(image.shape[0]), int(image.shape[1])],
                "mask_shape": [int(mask.shape[0]), int(mask.shape[1])],
                "image_sha256": image_sha256,
                "mask_sha256": mask_sha256,
            }
        )

    if len(observations) != 695 or len(dates) != len(observations):
        raise ValueError(
            f"Expected 695 dated, unique daily AHAS training images, got "
            f"{len(observations)} observations and {len(dates)} dates"
        )
    exact_byte_overlaps = sorted(comp_bytes_hashes & external_bytes_hashes)
    decoded_pixel_overlaps = sorted(comp_pixel_hashes & external_pixel_hashes)
    if exact_byte_overlaps or decoded_pixel_overlaps:
        raise ValueError(
            "MORDEN images contain exact competition overlaps: "
            f"bytes={len(exact_byte_overlaps)}, pixels={len(decoded_pixel_overlaps)}"
        )

    manifest: dict[str, Any] = {
        "dataset": "MORDEN AHAS CHASE training data",
        "dataset_record": SOURCE_ROOT,
        "doi": MORDEN_DOI,
        "repository": REPOSITORY_URL,
        "paper": PAPER_URL,
        "citation": SOURCE_CITATION,
        "license": "CC-BY-4.0",
        "competition_rule_basis": "Public, no-cost external data under Rule 6",
        "archive": {
            "file": "data.zip",
            "bytes": archive_path.stat().st_size,
            "md5": archive_md5,
            "sha256": archive_sha256,
        },
        "source_provenance": {
            "observatory": "Chinese H-alpha Solar Explorer (CHASE)",
            "source_years": [2024, 2025],
            "paper_description": "One observation per day; labels are automated AHAS semantic masks.",
            "selected_partition": "AHAS/train",
            "excluded_partitions": [
                "AHAS/val",
                "AHAS/test",
                "MHAS/train",
                "MHAS/val",
                "MHAS/test",
            ],
            "anonymized_mhas_samples_in_ahas_train_excluded": 80,
        },
        "competition": {
            "split_sha256": split.sha256,
            "training_observations": len(split.training_names),
            "validation_observations": len(split.validation_names),
            "test_images": competition.test_image_count,
            "date_range_years": [
                min(path.name[:4] for path in all_competition_files),
                max(path.name[:4] for path in all_competition_files),
            ],
            "unique_dates": len(comp_dates),
        },
        "audit": {
            "selected_observations": len(observations),
            "unique_source_dates": len(dates),
            "connected_components_in_semantic_masks": annotation_count,
            "same_calendar_date_overlaps": 0,
            "exact_byte_identical_images": 0,
            "exact_decoded_2048_pixel_overlaps": 0,
            "competition_images_checked": len(all_competition_files),
            "source_images_checked": len(observations),
        },
        "training_scope": {
            "external_labels_used_for_pretraining": True,
            "competition_labels_used_for_finetuning": True,
            "public_magfilo_labels_used": False,
            "external_images_used_for_final_inference": False,
        },
        "observations": observations,
    }
    manifest["selection_manifest_sha256"] = _sha256_bytes(
        json.dumps(observations, sort_keys=True, separators=(",", ":")).encode()
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit CC-BY MORDEN AHAS training data against competition observations."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--competition-data-root",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026",
    )
    parser.add_argument(
        "--external-root",
        type=Path,
        default=REPO_ROOT / "data" / "external" / "morden-zenodo-21582245",
    )
    args = parser.parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    manifest = audit(
        repo_root=repo_root,
        competition_data_root=args.competition_data_root.expanduser().resolve(),
        external_root=args.external_root.expanduser().resolve(),
    )
    output_path = args.external_root.expanduser().resolve() / "audit_manifest.json"
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("archive", "source_provenance", "competition", "audit", "selection_manifest_sha256")}, indent=2))
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
