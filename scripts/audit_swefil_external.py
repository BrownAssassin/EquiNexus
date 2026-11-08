# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from filament_baseline.data import load_dataset
from filament_baseline.external_data import load_swefil_dataset
from filament_baseline.split import create_observation_split


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the public SWEFil external training data.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--competition-data-root",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026",
    )
    parser.add_argument(
        "--swefil-root", type=Path, default=REPO_ROOT / "data" / "external" / "SWEFil"
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--exclude-all-competition-dates",
        action="store_true",
        help="Use a date-disjoint sample suitable for grouped cross-fit experiments.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.expanduser().resolve()
    competition_root = args.competition_data_root.expanduser().resolve()
    swefil_root = args.swefil_root.expanduser().resolve()
    if not swefil_root.is_relative_to(repo_root):
        raise ValueError(f"SWEFil must stay inside the workspace: {swefil_root}")
    competition = load_dataset(competition_root)
    split = create_observation_split(
        competition,
        repo_root / "data" / "splits",
        seed=42,
        validation_fraction=0.2,
    )
    external, metadata = load_swefil_dataset(
        swefil_root,
        competition,
        split,
        repository_root=repo_root,
        exclude_all_competition_dates=args.exclude_all_competition_dates,
    )

    image_hashes = []
    total_bytes = 0
    competition_paths = {
        competition.train_image_dir / view.file_name
        for view in competition.image_views
    }
    competition_paths.update(
        path for path in competition.test_image_dir.iterdir() if path.is_file()
    )
    competition_byte_hashes = set()
    competition_pixel_hashes = set()
    for image_path in sorted(competition_paths):
        payload = image_path.read_bytes()
        competition_byte_hashes.add(hashlib.sha256(payload).hexdigest())
        with Image.open(image_path) as image:
            competition_pixel_hashes.add(
                hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
            )

    included_byte_hashes = set()
    included_pixel_hashes = set()
    for view in external.image_views:
        image_path = external.train_image_dir / view.file_name
        payload = image_path.read_bytes()
        with Image.open(image_path) as image:
            if image.format != "JPEG" or image.size != (2048, 2048):
                raise ValueError(f"Invalid SWEFil image: {image_path}")
            image.load()
            pixel_sha256 = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
        total_bytes += len(payload)
        byte_sha256 = hashlib.sha256(payload).hexdigest()
        included_byte_hashes.add(byte_sha256)
        included_pixel_hashes.add(pixel_sha256)
        image_hashes.append(
            {"file_name": view.file_name, "bytes": len(payload), "sha256": byte_sha256}
        )
    hashes_json = json.dumps(image_hashes, sort_keys=True, separators=(",", ":")).encode()
    metadata["included_image_file_count"] = len(image_hashes)
    metadata["included_image_bytes"] = total_bytes
    metadata["included_image_hashes_sha256"] = hashlib.sha256(hashes_json).hexdigest()
    metadata["exact_byte_overlap_with_competition_images"] = len(
        included_byte_hashes & competition_byte_hashes
    )
    metadata["exact_decoded_rgb_overlap_with_competition_images"] = len(
        included_pixel_hashes & competition_pixel_hashes
    )
    if (
        metadata["exact_byte_overlap_with_competition_images"]
        or metadata["exact_decoded_rgb_overlap_with_competition_images"]
    ):
        raise ValueError(
            "SWEFil includes pixel-identical competition images; exclude those observations before training"
        )
    metadata["competition_split_sha256"] = split.sha256

    default_name = (
        "equinexus_audit_all_competition_dates_excluded.json"
        if args.exclude_all_competition_dates
        else "equinexus_audit.json"
    )
    output_path = args.output.expanduser().resolve() if args.output else swefil_root / default_name
    if not output_path.is_relative_to(repo_root):
        raise ValueError(f"Audit output must stay inside the workspace: {output_path}")
    output_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
