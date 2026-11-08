# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]


def audit(data_root: Path, competition_root: Path) -> dict[str, object]:
    data_root = data_root.resolve()
    competition_root = competition_root.resolve()
    if not data_root.is_relative_to(REPO_ROOT) or not competition_root.is_relative_to(REPO_ROOT):
        raise ValueError("External and competition data must stay inside the repository")
    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    records = manifest["files"]
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if digest != manifest.get("content_manifest_sha256"):
        raise ValueError("Manifest record digest does not match content_manifest_sha256")
    competition_stems = {
        path.stem
        for path in competition_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    overlaps = sorted(
        Path(record["filename"]).stem
        for record in records
        if Path(record["filename"]).stem in competition_stems
    )
    if overlaps:
        raise ValueError(f"GONG data overlaps competition filenames: {overlaps[:5]}")

    sites: Counter[str] = Counter()
    total_bytes = 0
    external_hashes: set[str] = set()
    for index, record in enumerate(records, start=1):
        path = data_root / "images" / record["filename"]
        payload = path.read_bytes()
        if len(payload) != int(record["bytes"]):
            raise ValueError(f"File size mismatch for {path.name}")
        file_digest = hashlib.sha256(payload).hexdigest()
        if file_digest != record["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {path.name}")
        external_hashes.add(file_digest)
        with Image.open(path) as image:
            if image.format != "JPEG" or image.size != (2048, 2048):
                raise ValueError(f"Unexpected image format or dimensions for {path.name}")
            image.verify()
        sites[str(record["site"])] += 1
        total_bytes += len(payload)
        if index % 250 == 0 or index == len(records):
            print(f"Audited {index}/{len(records)} images", flush=True)

    if total_bytes != int(manifest["total_bytes"]):
        raise ValueError("Total byte count differs from the manifest")
    competition_hashes = {
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in competition_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    content_overlaps = external_hashes & competition_hashes
    if content_overlaps:
        raise ValueError(
            f"GONG data has {len(content_overlaps)} byte-identical competition image(s)"
        )
    return {
        "count": len(records),
        "total_bytes": total_bytes,
        "content_manifest_sha256": digest,
        "competition_overlap_count": len(overlaps),
        "competition_content_overlap_count": len(content_overlaps),
        "competition_file_count_checked": len(competition_stems),
        "image_size": [2048, 2048],
        "site_counts": dict(sorted(sites.items())),
        "start_date_utc": manifest["selection"]["start_date_utc"],
        "end_date_utc": manifest["selection"]["end_date_utc"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the unlabeled external GONG image set.")
    parser.add_argument(
        "--data-root", type=Path, default=REPO_ROOT / "data" / "external" / "gong_halpha_2026_11"
    )
    parser.add_argument(
        "--competition-root",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026",
    )
    args = parser.parse_args()
    result = audit(args.data_root, args.competition_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
