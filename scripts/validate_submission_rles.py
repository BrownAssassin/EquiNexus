# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Validate an instance-RLE competition submission against supplied test images."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from pycocotools import mask as mask_utils


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (REPO_ROOT / "data").resolve()
ARTIFACTS_ROOT = (REPO_ROOT / "artifacts").resolve()
IMAGE_SIZE = (2048, 2048)
EXPECTED_COLUMNS = ["filament_id", "segmentation_rle"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_within(path: Path, root: Path, description: str) -> None:
    if not path.is_relative_to(root):
        raise ValueError(f"{description} must remain under {root}: {path}")


def validate_submission(
    submission_path: Path,
    test_image_dir: Path,
    *,
    expected_test_images: int,
) -> dict[str, Any]:
    submission_path = submission_path.expanduser().resolve()
    test_image_dir = test_image_dir.expanduser().resolve()
    _require_within(submission_path, ARTIFACTS_ROOT, "Submission CSV")
    _require_within(test_image_dir, DATA_ROOT, "Test image directory")
    if not submission_path.is_file():
        raise FileNotFoundError(submission_path)
    if not test_image_dir.is_dir():
        raise NotADirectoryError(test_image_dir)

    test_images = sorted(test_image_dir.glob("*.jpeg"))
    if len(test_images) != expected_test_images:
        raise ValueError(
            f"Expected {expected_test_images} supplied test images, found {len(test_images)}"
        )
    test_stems = {path.stem for path in test_images}
    for image_path in test_images:
        with Image.open(image_path) as image:
            if image.size != IMAGE_SIZE:
                raise ValueError(f"Expected {IMAGE_SIZE} image at {image_path}, got {image.size}")

    frame = pd.read_csv(submission_path, dtype=str, keep_default_na=False)
    if frame.columns.tolist() != EXPECTED_COLUMNS:
        raise ValueError(f"Expected columns {EXPECTED_COLUMNS}, got {frame.columns.tolist()}")
    if frame.empty or frame.isna().any().any():
        raise ValueError("Submission must contain at least one complete prediction row")
    if frame[EXPECTED_COLUMNS].apply(lambda column: column.str.strip().eq("")).any().any():
        raise ValueError("Submission contains a blank ID or RLE")
    if not frame["filament_id"].is_unique:
        raise ValueError("filament_id values must be unique")

    rows_by_image: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for row in frame.itertuples(index=False):
        parts = row.filament_id.rsplit("_", maxsplit=1)
        if len(parts) != 2 or not parts[1].isdigit() or int(parts[1]) < 1:
            raise ValueError(f"Malformed filament ID: {row.filament_id}")
        image_stem, sequence = parts[0], int(parts[1])
        if image_stem not in test_stems:
            raise ValueError(f"Unknown supplied test image in filament ID: {row.filament_id}")
        rows_by_image[image_stem].append(
            (row.filament_id, row.segmentation_rle, sequence)
        )

    areas: list[int] = []
    exact_roundtrips = 0
    overlap_pixels = 0
    for image_stem, rows in rows_by_image.items():
        sequences = sorted(sequence for _, _, sequence in rows)
        if sequences != list(range(1, len(rows) + 1)):
            raise ValueError(f"Filament IDs for {image_stem} are not unique and contiguous from 1")

        claimed = np.zeros(IMAGE_SIZE, dtype=np.uint8)
        for filament_id, encoded_text, _ in rows:
            try:
                encoded = encoded_text.encode("ascii")
            except UnicodeEncodeError as exc:
                raise ValueError(f"Non-ASCII RLE for {filament_id}") from exc
            decoded = mask_utils.decode({"size": list(IMAGE_SIZE), "counts": encoded})
            if decoded.shape != IMAGE_SIZE or not decoded.any():
                raise ValueError(f"Invalid, wrong-sized, or empty RLE for {filament_id}")
            if not np.isin(decoded, (0, 1)).all():
                raise ValueError(f"Non-binary decoded mask for {filament_id}")
            roundtrip = mask_utils.encode(np.asfortranarray(decoded.astype(np.uint8)))
            if roundtrip["counts"].decode("ascii") != encoded_text:
                raise ValueError(f"RLE round trip changed the encoding for {filament_id}")
            exact_roundtrips += 1

            overlapping = int(np.count_nonzero(claimed & decoded))
            if overlapping:
                raise ValueError(f"{filament_id} overlaps {overlapping} claimed pixels")
            overlap_pixels += overlapping
            claimed |= decoded
            areas.append(int(decoded.sum()))

    submission_hash = sha256_file(submission_path)
    return {
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "submission_path": str(submission_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "csv_sha256": submission_hash,
        "rows": int(len(frame)),
        "processed_test_images": len(test_images),
        "images_with_predictions": len(rows_by_image),
        "images_without_predictions": len(test_stems - rows_by_image.keys()),
        "unique_ids": True,
        "rle_shape": list(IMAGE_SIZE),
        "rle_exact_roundtrips": exact_roundtrips,
        "all_masks_nonempty": True,
        "instances_overlap_pixels": overlap_pixels,
        "min_instance_area": min(areas),
        "median_instance_area": float(np.median(areas)),
        "max_instance_area": max(areas),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission_csv", type=Path)
    parser.add_argument(
        "--test-image-dir",
        type=Path,
        default=DATA_ROOT / "raw" / "MAGFiLO_1.0_Kaggle_2026" / "test" / "test_images",
    )
    parser.add_argument("--expected-test-images", type=int, default=180)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    summary = validate_submission(
        args.submission_csv,
        args.test_image_dir,
        expected_test_images=args.expected_test_images,
    )
    output_path = args.output_json
    if output_path is not None:
        output_path = output_path.expanduser().resolve()
        _require_within(output_path, ARTIFACTS_ROOT, "Validation report")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        summary["validation_report_path"] = str(output_path.relative_to(REPO_ROOT)).replace("\\", "/")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
