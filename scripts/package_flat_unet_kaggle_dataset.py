# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from filament_baseline.data import load_dataset  # noqa: E402
from filament_baseline.external_data import load_flat_unet_dataset  # noqa: E402
from filament_baseline.split import create_observation_split  # noqa: E402


DATASET_ID = "mrinanksivakumar/flat-unet-date-disjoint-2026"
DEFAULT_SOURCE = REPO_ROOT / "data" / "external" / "flat-unet-101545" / "kaggle-256-q90"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package(
    source_root: Path,
    competition_data_root: Path,
    repository_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    source_root = source_root.resolve()
    competition_data_root = competition_data_root.resolve()
    if not source_root.is_relative_to(repository_root / "data" / "external"):
        raise ValueError(f"Flat U-Net data must remain under data/external/: {source_root}")
    if not competition_data_root.is_relative_to(repository_root / "data"):
        raise ValueError(f"Competition data must remain under data/: {competition_data_root}")

    competition = load_dataset(competition_data_root)
    split = create_observation_split(
        competition,
        repository_root / "data" / "splits",
        seed=42,
        validation_fraction=0.2,
    )
    external, provenance = load_flat_unet_dataset(
        source_root, competition, repository_root=repository_root
    )
    if split.sha256 != "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e":
        raise ValueError("Frozen competition split digest changed before Kaggle packaging")
    if provenance["competition_date_overlap"] != 0 or len(external.observation_names) != 256:
        raise ValueError("Flat U-Net input failed date-overlap or observation-count checks")

    manifest = json.loads((source_root / "manifest.json").read_text(encoding="utf-8"))
    total_bytes = 0
    for row in manifest["observations"]:
        for kind, folder in (("image", "images"), ("mask", "masks")):
            file_name = row[kind]["file_name"]
            path = (source_root / folder / file_name).resolve()
            if not path.is_relative_to(source_root):
                raise ValueError(f"Flat U-Net manifest path escaped its data root: {file_name}")
            if _sha256(path) != row[kind]["processed_sha256"]:
                raise ValueError(f"Flat U-Net {kind} hash mismatch: {file_name}")
            total_bytes += path.stat().st_size

    metadata = {
        "title": "EquiNexus Flat U-Net Date-Disjoint Weak Labels",
        "id": DATASET_ID,
        "licenses": [{"name": "Other"}],
        "isPrivate": True,
        "keywords": ["solar", "filament", "BBSO", "H-alpha", "weak-labels"],
        "description": (
            "Private team compute input containing 256 date-disjoint BBSO H-alpha images "
            "and binary semantic filament masks from the public China-VO PaperData record "
            "https://doi.org/10.12149/101545. Source metadata states that dataset copyrights "
            "remain with the authors and does not provide a separate dataset license. This "
            "copy is used under Kaggle Competition Rule 6 with attribution and is not for "
            "public redistribution. All 801 calendar dates in the competition train and test "
            "sets are excluded; the source test_gd labels are not used. The masks are weak "
            "semantic targets, not per-filament instance labels. Images are normalized "
            "grayscale JPEG quality 90 derivatives of the local lossless PNG subset; their "
            "measured compression error is recorded in the manifest. The manifest stores source "
            "and processed file hashes, intensity normalization, citations, and the frozen "
            "competition split digest."
        ),
    }
    metadata_path = source_root / "dataset-metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "dataset_id": DATASET_ID,
        "private": True,
        "metadata_path": str(metadata_path),
        "source_archive_sha256": manifest["source_archive_sha256"],
        "selection_manifest_sha256": manifest["selection_manifest_sha256"],
        "observations": len(external.observation_names),
        "unique_dates": len({row["date"] for row in manifest["observations"]}),
        "competition_date_overlap": 0,
        "data_files_bytes": total_bytes,
        "frozen_competition_split_sha256": split.sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify and prepare the private date-disjoint Flat U-Net Kaggle input."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--competition-data-root",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026",
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    print(
        json.dumps(
            package(args.source_root, args.competition_data_root, args.repo_root),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
