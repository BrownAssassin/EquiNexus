# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "data" / "external" / "chrotel"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "kaggle" / "chrotel-external-dataset"
ARCHIVE_NAME = "chrotel-date-disjoint-weak-labels.tar.gz"
KAGGLE_DATASET_ID = "mrinanksivakumar/chrotel-date-disjoint-2026"
FROZEN_SPLIT_SHA256 = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package(source_root: Path, output_root: Path) -> dict[str, Any]:
    source_root = source_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not source_root.is_relative_to(REPO_ROOT / "data" / "external"):
        raise ValueError(f"ChroTel source must remain under data/external/: {source_root}")
    if not output_root.is_relative_to(REPO_ROOT / "artifacts"):
        raise ValueError(f"Kaggle upload package must remain under artifacts/: {output_root}")

    manifest_path = source_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"ChroTel preparation manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("ChroTel preparation is not complete")
    if manifest.get("all_competition_dates_excluded") is not True:
        raise ValueError("ChroTel manifest does not certify exclusion of all competition dates")
    if manifest.get("competition_date_overlap") != 0:
        raise ValueError("ChroTel manifest reports competition-date overlap")
    if manifest.get("selection", {}).get("frozen_competition_split_sha256") != FROZEN_SPLIT_SHA256:
        raise ValueError("ChroTel manifest uses a different frozen competition split")

    rows = manifest.get("observations", [])
    if not rows or len(rows) != manifest.get("prepared_observations"):
        raise ValueError("ChroTel manifest observation inventory is empty or inconsistent")
    for row in rows:
        filename = row["file_name"]
        for kind, directory in (("image", "images"), ("mask", "masks")):
            path = (source_root / directory / filename).resolve()
            if not path.is_relative_to(source_root):
                raise ValueError(f"ChroTel manifest path escaped data root: {filename}")
            if sha256_file(path) != row[f"{kind}_sha256"]:
                raise ValueError(f"ChroTel {kind} checksum mismatch: {filename}")

    output_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "title": "EquiNexus ChroTel Date-Disjoint Weak Labels",
        "id": KAGGLE_DATASET_ID,
        "licenses": [{"name": "Other"}],
        "isPrivate": True,
        "description": (
            "Private compute copy of a date-disjoint subset of KIS ChroTel H-alpha images "
            "and semi-supervised semantic filament masks. Source products are freely "
            "available for scientific and educational use. This dataset preserves KIS "
            "attribution, its segmentation-method citation, and SHA-256 provenance. It "
            "contains no MAGFiLO annotations. The source manifest certifies zero calendar "
            "date overlap with competition train and test images."
        ),
        "keywords": ["solar", "filament", "ChroTel", "H-alpha", "weak-labels"],
    }
    (output_root / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    archive_path = output_root / ARCHIVE_NAME
    with tarfile.open(archive_path, mode="w:gz", compresslevel=1) as archive:
        for relative in (
            Path("manifest.json"),
            Path("selection.json"),
            Path("images"),
            Path("masks"),
        ):
            path = source_root / relative
            if path.exists():
                archive.add(path, arcname=relative.as_posix(), recursive=True)
    return {
        "dataset_id": KAGGLE_DATASET_ID,
        "private": True,
        "source_manifest_sha256": sha256_file(manifest_path),
        "selection_manifest_sha256": manifest["selection_manifest_sha256"],
        "observations": len(rows),
        "archive": str(archive_path.relative_to(REPO_ROOT)),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256_file(archive_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Package the audited ChroTel weak-label subset as a private Kaggle input."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(package(args.source_root, args.output_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
