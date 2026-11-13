#!/usr/bin/env python3
"""Build a label-free test consensus from audited candidate submission CSVs.

The script reads only the selected CSVs and the supplied test image filenames
for coverage checking. It never opens test image pixels or any ground-truth
labels. Keep each input family independent; related model outputs must not get
multiple votes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools import mask as mask_utils


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SPLIT_SHA256 = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"
HEIGHT = WIDTH = 2048
PIXELS = HEIGHT * WIDTH
MATCH_IOU = 0.50  # strict: IoU must be greater than this value
MIN_FAMILIES = 2
STRICT_IOU = 0.60  # every supporter must reach this when exactly two families vote
MIN_RETAINED_PIXEL_FRACTION = 0.50
EXPECTED_COLUMNS = ["filament_id", "segmentation_rle"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_rle(rle_text: str) -> np.ndarray:
    rle = {"size": [HEIGHT, WIDTH], "counts": rle_text.encode("ascii")}
    decoded = mask_utils.decode(rle)
    if decoded.shape != (HEIGHT, WIDTH):
        raise ValueError(f"Unexpected decoded mask shape: {decoded.shape}")
    indices = np.flatnonzero(decoded.reshape(-1, order="F"))
    if indices.size == 0:
        raise ValueError("An input mask is empty")
    return indices


def encode_indices(indices: np.ndarray) -> str:
    flat = np.zeros(PIXELS, dtype=np.uint8)
    flat[indices] = 1
    rle = mask_utils.encode(np.asfortranarray(flat.reshape((HEIGHT, WIDTH), order="F")))
    return rle["counts"].decode("ascii")


def read_source(path: Path) -> tuple[dict[str, list[str]], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    by_image: dict[str, list[str]] = {}
    seen_ids: set[str] = set()
    row_count = 0
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(f"Unexpected columns in {path}: {reader.fieldnames}")
        for row in reader:
            identifier = str(row["filament_id"] or "")
            rle = str(row["segmentation_rle"] or "")
            parts = identifier.rsplit("_", 1)
            if not identifier or not rle or len(parts) != 2 or not parts[1].isdigit():
                raise ValueError(f"Malformed row {row_count + 2} in {path}")
            if identifier in seen_ids:
                raise ValueError(f"Duplicate filament ID in {path}: {identifier}")
            seen_ids.add(identifier)
            by_image.setdefault(parts[0], []).append(rle)
            row_count += 1
    return by_image, {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
        "md5": md5_file(path),
        "rows": row_count,
        "images_with_predictions": len(by_image),
    }


def build_voter(rles: list[str]) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    masks = [decode_rle(rle) for rle in rles]
    labels = np.zeros(PIXELS, dtype=np.int32)
    areas = np.empty(len(masks), dtype=np.int64)
    for mask_id, indices in enumerate(masks, start=1):
        if np.any(labels[indices]):
            raise ValueError("Input candidate contains overlapping masks in one image")
        labels[indices] = mask_id
        areas[mask_id - 1] = indices.size
    return masks, labels, areas


def best_iou(indices: np.ndarray, labels: np.ndarray, areas: np.ndarray) -> float:
    if indices.size == 0 or areas.size == 0:
        return 0.0
    hit_ids = labels[indices]
    hit_ids = hit_ids[hit_ids > 0]
    if hit_ids.size == 0:
        return 0.0
    counts = np.bincount(hit_ids, minlength=len(areas) + 1)
    matched_ids = np.flatnonzero(counts[1:])
    intersections = counts[matched_ids + 1]
    union = indices.size + areas[matched_ids] - intersections
    return float(np.max(intersections / union))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "artifacts/submissions/filament-20261114-component-selector-rf-oof-batch.json",
    )
    parser.add_argument(
        "--source-indices",
        type=int,
        nargs=3,
        default=[1, 3, 4],
        help="three 1-based manifest indices in priority order",
    )
    parser.add_argument(
        "--families",
        nargs=3,
        default=["EquiNexus U-Net+YOLO11s", "HDJoJo YOLO26x", "I079 Mask R-CNN+refiner"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/submissions/equinexus-three-family-consensus-20261114-v2",
    )
    parser.add_argument(
        "--test-image-dir",
        type=Path,
        default=ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026/test/test_images",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    test_image_dir = args.test_image_dir.resolve()
    submissions_root = (ROOT / "artifacts/submissions").resolve()
    if manifest_path.is_relative_to(output_dir):
        raise ValueError("Output directory cannot contain the input manifest")
    for path, label in ((manifest_path, "manifest"), (output_dir, "output directory")):
        if not path.is_relative_to(submissions_root):
            raise ValueError(f"{label} must remain under {submissions_root}")
    if not test_image_dir.is_relative_to((ROOT / "data").resolve()):
        raise ValueError("Test image directory must remain inside the repository data folder")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("competition") != "filament-segmentation-2026":
        raise ValueError("Manifest belongs to a different competition")
    if manifest.get("split_sha256") != EXPECTED_SPLIT_SHA256:
        raise ValueError("Manifest does not use the frozen validation split")
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("Manifest candidates must be a list")
    if len(args.source_indices) != 3 or len(set(args.source_indices)) != 3:
        raise ValueError("Exactly three distinct source indices are required")
    if len(args.families) != 3 or len(set(args.families)) != 3:
        raise ValueError("Each selected candidate must represent a distinct model family")

    test_files = sorted(path for path in test_image_dir.iterdir() if path.is_file())
    stems = [path.stem for path in test_files]
    if len(stems) != 180 or len(set(stems)) != 180:
        raise ValueError(f"Expected 180 unique supplied test image stems, found {len(stems)}")
    stem_set = set(stems)

    sources: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for source_index, family in zip(args.source_indices, args.families, strict=True):
        if not 1 <= source_index <= len(candidates):
            raise ValueError(f"Source index {source_index} is outside the manifest")
        candidate = candidates[source_index - 1]
        path = (ROOT / candidate["path"]).resolve()
        if not path.is_relative_to(submissions_root):
            raise ValueError(f"Input candidate escaped artifacts/submissions: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != candidate["sha256"]:
            raise ValueError(f"Input hash mismatch at manifest index {source_index}")
        by_image, provenance = read_source(path)
        unknown = set(by_image) - stem_set
        if unknown:
            raise ValueError(f"Input {source_index} includes unknown observations: {sorted(unknown)[:3]}")
        sources[family] = {
            "candidate": candidate,
            "by_image": by_image,
            "provenance": {
                **provenance,
                "manifest_index": source_index,
                "candidate_sha256": candidate["sha256"],
                "manifest_message": candidate["message"],
                "candidate_selection": candidate.get("selection"),
                "candidate_validation_protocol": candidate.get("validation_protocol"),
                "candidate_run_metadata": candidate.get("run_metadata"),
                "family": family,
            },
        }

    source_mask_counts = {family: 0 for family in sources}
    selected: list[tuple[str, str]] = []
    selected_by_image: dict[str, list[str]] = {}
    supporter_histogram: dict[str, int] = {}
    retained_by_source = {family: 0 for family in sources}
    clipped_count = 0
    fallback_images = 0
    for image_number, stem in enumerate(stems, start=1):
        voters: OrderedDict[str, tuple[list[np.ndarray], np.ndarray, np.ndarray]] = OrderedDict()
        for family, source in sources.items():
            voter = build_voter(source["by_image"].get(stem, []))
            source_mask_counts[family] += len(voter[0])
            voters[family] = voter
        taken = np.zeros(PIXELS, dtype=np.bool_)
        chosen: list[str] = []
        fallback: tuple[int, str, int] | None = None
        for family, (masks, _, _) in voters.items():
            for mask_index, indices in enumerate(masks):
                votes = {family: 1.0}
                for other_family, (_, labels, areas) in voters.items():
                    if other_family == family:
                        continue
                    iou = best_iou(indices, labels, areas)
                    if iou > MATCH_IOU:
                        votes[other_family] = iou
                if fallback is None or len(votes) > fallback[0]:
                    fallback = (len(votes), family, mask_index)
                vote_key = str(len(votes))
                supporter_histogram[vote_key] = supporter_histogram.get(vote_key, 0) + 1
                if len(votes) < MIN_FAMILIES:
                    continue
                if len(votes) == MIN_FAMILIES and min(votes.values()) < STRICT_IOU:
                    continue
                free = ~taken[indices]
                remaining = int(free.sum())
                if remaining < MIN_RETAINED_PIXEL_FRACTION * indices.size:
                    continue
                if remaining == indices.size:
                    rle = sources[family]["by_image"][stem][mask_index]
                else:
                    indices = indices[free]
                    rle = encode_indices(indices)
                    clipped_count += 1
                taken[indices] = True
                chosen.append(rle)
                retained_by_source[family] += 1
        if not chosen and fallback is not None:
            _, family, mask_index = fallback
            chosen.append(sources[family]["by_image"][stem][mask_index])
            retained_by_source[family] += 1
            fallback_images += 1
        selected.extend((f"{stem}_{index}", rle) for index, rle in enumerate(chosen, start=1))
        selected_by_image[stem] = chosen
        if image_number % 30 == 0:
            print(f"processed {image_number}/180 images; retained={len(selected)}", flush=True)

    output_dir.mkdir(parents=True)
    csv_path = output_dir / "submission.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(EXPECTED_COLUMNS)
        writer.writerows(selected)

    # Decode every output again and verify uniqueness, shape, and disjointness.
    output_ids: set[str] = set()
    decoded_count = 0
    for stem, rles in selected_by_image.items():
        occupied = np.zeros(PIXELS, dtype=np.bool_)
        for index, rle in enumerate(rles, start=1):
            identifier = f"{stem}_{index}"
            if identifier in output_ids:
                raise ValueError(f"Duplicate output identifier: {identifier}")
            output_ids.add(identifier)
            indices = decode_rle(rle)
            if np.any(occupied[indices]):
                raise ValueError(f"Output masks overlap for {stem}")
            occupied[indices] = True
            decoded_count += 1
    if len(selected_by_image) != 180 or decoded_count != len(selected):
        raise ValueError("Output coverage or decoded instance count mismatch")
    images_with_predictions = sum(bool(rles) for rles in selected_by_image.values())
    if not selected:
        raise ValueError("Consensus output is empty")

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    source_details = [source["provenance"] for source in sources.values()]
    metadata = {
        "candidate_id": "equinexus-three-family-consensus-20261114-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": head,
        "source_script": Path(__file__).resolve().relative_to(ROOT).as_posix(),
        "source_script_sha256": sha256_file(Path(__file__).resolve()),
        "manifest": manifest_path.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_source_commit": manifest.get("source_commit"),
        "split_sha256": EXPECTED_SPLIT_SHA256,
        "method": "Label-free vote across three distinct audited candidate families. A mask needs votes from at least two families at IoU > 0.5; when exactly two families support it, both IoUs must be at least 0.6. Keep at least half of a mask after overlap clipping; backfill an image with its highest-vote row only when no row passes.",
        "parameters": {
            "family_count": len(sources),
            "minimum_families": MIN_FAMILIES,
            "match_iou_strictly_greater_than": MATCH_IOU,
            "exactly_two_families_minimum_iou": STRICT_IOU,
            "minimum_retained_pixel_fraction": MIN_RETAINED_PIXEL_FRACTION,
            "priority_order": list(sources),
        },
        "sources": source_details,
        "training_executed": False,
        "competition_test_labels_read": False,
        "test_image_pixels_read": False,
        "test_images_expected": 180,
        "test_images_processed": len(selected_by_image),
        "test_images_covered": images_with_predictions,
        "source_mask_counts": source_mask_counts,
        "retained_rows_by_source": retained_by_source,
        "supporter_histogram_over_input_masks": supporter_histogram,
        "pixel_disjoint_output": True,
        "clipped_mask_count": clipped_count,
        "fallback_images": fallback_images,
        "rows": len(selected),
        "decoded_output_masks": decoded_count,
        "elapsed_seconds": time.perf_counter() - started,
        "submission_sha256": sha256_file(csv_path),
        "submission_md5": md5_file(csv_path),
    }
    validation = {
        "columns": EXPECTED_COLUMNS,
        "rows": len(selected),
        "unique_ids": len(output_ids),
        "test_images_expected": 180,
        "test_images_processed": len(selected_by_image),
        "test_images_covered": images_with_predictions,
        "decoded_masks": decoded_count,
        "mask_size": [HEIGHT, WIDTH],
        "all_masks_nonempty": True,
        "all_masks_pixel_disjoint_per_image": True,
        "submission_sha256": metadata["submission_sha256"],
        "submission_md5": metadata["submission_md5"],
    }
    (output_dir / "run-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "rle-validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: metadata[key] for key in (
        "rows", "test_images_processed", "test_images_covered", "source_mask_counts", "retained_rows_by_source",
        "supporter_histogram_over_input_masks", "clipped_mask_count", "fallback_images",
        "elapsed_seconds", "submission_sha256", "submission_md5",
    )}, indent=2), flush=True)
    return metadata


if __name__ == "__main__":
    run(parse_args())
