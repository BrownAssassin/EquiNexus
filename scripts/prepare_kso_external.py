# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Prepare a date-disjoint sample of KSO H-alpha images and filament maps."""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import io
import json
import re
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
from astropy.io import fits

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from filament_baseline.data import load_dataset

IMAGE_INDEX = "https://cesar.kso.ac.at/halpha3s/FITS/arch/2023/"
MASK_ROOT = "https://cesar.kso.ac.at/sniv/filaments/2023/"
IMAGE_PATTERN = re.compile(r"^kanz_halph_fi_(?P<stamp>2023\d{4}_\d{6})\.fts\.gz$")
ATTRIBUTION = (
    "H-alpha data / filament maps were provided by the Kanzelhoehe Observatory, "
    "University of Graz, Austria."
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _parse_mask(payload: bytes, file_name: str) -> tuple[np.ndarray, list[int]]:
    raw = gzip.decompress(payload)
    if len(raw) < 16:
        raise ValueError(f"KSO mask header is truncated: {file_name}")
    height, width, reserved_a, reserved_b = struct.unpack("<4I", raw[:16])
    if (height, width) != (2048, 2048) or reserved_a != 0 or reserved_b != 0:
        raise ValueError(f"Unexpected KSO mask header in {file_name}: {(height, width, reserved_a, reserved_b)}")
    values = np.frombuffer(raw, dtype="<f4", offset=16)
    if values.size != 2048 * 2048:
        raise ValueError(f"Unexpected KSO mask payload size in {file_name}: {values.size}")
    label_map = values.reshape(2048, 2048)
    if not np.isfinite(label_map).all() or np.any(label_map < 0):
        raise ValueError(f"KSO mask contains invalid values: {file_name}")
    rounded = np.rint(label_map)
    if not np.allclose(label_map, rounded, rtol=0.0, atol=1e-5):
        raise ValueError(f"KSO mask labels are not integer-valued: {file_name}")
    labels = [int(value) for value in np.unique(rounded) if value > 0]
    return rounded.astype(np.uint32), labels


def _read_image(payload: bytes, file_name: str) -> tuple[np.ndarray, dict[str, Any]]:
    fits_payload = gzip.decompress(payload)
    with fits.open(io.BytesIO(fits_payload), memmap=False) as hdus:
        arrays = [np.asarray(hdu.data) for hdu in hdus if hdu.data is not None]
        if not arrays:
            raise ValueError(f"KSO FITS image has no image plane: {file_name}")
        image = np.squeeze(arrays[0])
    if image.shape != (2048, 2048) or image.ndim != 2:
        raise ValueError(f"Expected a 2048x2048 KSO image, got {image.shape}: {file_name}")
    finite = np.isfinite(image)
    if float(finite.mean()) < 0.999:
        raise ValueError(f"KSO FITS image has non-finite pixels: {file_name}")
    low, high = np.percentile(image[finite].astype(np.float32), [0.5, 99.5])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError(f"KSO FITS image has invalid intensity range: {file_name}")
    scaled = np.clip((image.astype(np.float32) - low) * (255.0 / (high - low)), 0, 255)
    scaled[~finite] = 0
    return np.rint(scaled).astype(np.uint8), {
        "fits_dtype": str(image.dtype),
        "shape": list(image.shape),
        "percentile_clip": [0.5, 99.5],
        "normalization_low": float(low),
        "normalization_high": float(high),
        "finite_fraction": float(finite.mean()),
    }


def _head_mask(item: tuple[str, str]) -> tuple[str, str, int] | None:
    image_name, mask_url = item
    try:
        response = requests.head(mask_url, timeout=(10, 20), allow_redirects=True)
        if response.status_code != 200:
            return None
        return image_name, mask_url, int(response.headers.get("Content-Length", "0"))
    except requests.RequestException:
        return None


def _download(url: str, path: Path) -> tuple[str, int]:
    response = requests.get(url, timeout=(15, 90), stream=True)
    response.raise_for_status()
    digest = hashlib.sha256()
    size = 0
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as stream:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            stream.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    if temporary.stat().st_size != size:
        raise OSError(f"Downloaded byte count changed while writing {path}")
    temporary.replace(path)
    expected_length = response.headers.get("Content-Length")
    if expected_length is not None and int(expected_length) != size:
        raise OSError(f"Content-Length mismatch for {url}: expected {expected_length}, got {size}")
    return digest.hexdigest(), size


def prepare(
    *, repository_root: Path, competition_root: Path, output_root: Path, max_observations: int
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    competition_root = competition_root.resolve()
    output_root = output_root.resolve()
    if not competition_root.is_relative_to(repository_root / "data"):
        raise ValueError(f"Competition data must stay under data/: {competition_root}")
    if not output_root.is_relative_to(repository_root / "data" / "external"):
        raise ValueError(f"KSO data must stay under data/external/: {output_root}")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty data directory: {output_root}")
    if max_observations < 1:
        raise ValueError("max_observations must be positive")

    competition = load_dataset(competition_root)
    competition_names = set(competition.observation_names)
    competition_names.update(path.name for path in competition.test_image_dir.iterdir() if path.is_file())
    blocked_dates = {name[:8] for name in competition_names}
    response = requests.get(IMAGE_INDEX, timeout=(15, 45))
    response.raise_for_status()
    page = response.text
    image_names = sorted(set(re.findall(r'href="(kanz_halph_fi_2023\d{4}_\d{6}\.fts\.gz)"', page)))
    if not image_names:
        raise ValueError("No 2023 H-alpha images were listed by the KSO archive")

    pairs = []
    for image_name in image_names:
        match = IMAGE_PATTERN.fullmatch(image_name)
        if match is None:
            raise ValueError(f"Unexpected KSO image filename: {image_name}")
        date = match.group("stamp")[:8]
        if date in blocked_dates:
            raise ValueError(f"KSO source date overlaps competition data: {date}")
        mask_name = image_name[:-3] + ".filament.bin.gz"
        pairs.append((image_name, f"{MASK_ROOT}{date}/{mask_name}"))

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        available = [item for item in pool.map(_head_mask, pairs) if item is not None]
    if not available:
        raise ValueError("The KSO archive returned no matching image-mask pairs")
    selected_count = min(max_observations, len(available))
    indices = np.linspace(0, len(available) - 1, selected_count, dtype=int)
    selected = [available[int(index)] for index in indices]
    if len({item[0][14:22] for item in selected}) != len(selected):
        raise AssertionError("KSO selection repeated a calendar date")

    image_dir = output_root / "images"
    raw_image_dir = output_root / "raw" / "images"
    raw_mask_dir = output_root / "raw" / "masks"
    for path in (image_dir, raw_image_dir, raw_mask_dir):
        path.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    for index, (image_name, mask_url, expected_mask_bytes) in enumerate(selected, start=1):
        stamp = IMAGE_PATTERN.fullmatch(image_name).group("stamp")
        date = stamp[:8]
        mask_name = image_name[:-3] + ".filament.bin.gz"
        image_url = IMAGE_INDEX + image_name
        image_path = raw_image_dir / image_name
        mask_path = raw_mask_dir / mask_name
        image_hash, image_bytes = _download(image_url, image_path)
        mask_hash, mask_bytes = _download(mask_url, mask_path)
        if expected_mask_bytes and mask_bytes != expected_mask_bytes:
            raise OSError(f"Mask size changed since HEAD request: {mask_name}")
        image_array, image_metadata = _read_image(image_path.read_bytes(), image_name)
        labels, label_ids = _parse_mask(mask_path.read_bytes(), mask_name)
        foreground_pixels = int(np.count_nonzero(labels))
        processed_name = f"{stamp}_KSO.jpg"
        processed_path = image_dir / processed_name
        if not cv2.imwrite(str(processed_path), image_array, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"Could not write normalized KSO image: {processed_path}")
        processed_hash = _sha256(processed_path.read_bytes())
        manifest_rows.append(
            {
                "date": date,
                "timestamp": stamp,
                "file_name": processed_name,
                "source_image_name": image_name,
                "source_mask_name": mask_name,
                "source_image_url": image_url,
                "source_mask_url": mask_url,
                "image_path": str(image_path.relative_to(output_root)),
                "mask_path": str(mask_path.relative_to(output_root)),
                "processed_image_path": str(processed_path.relative_to(output_root)),
                "source_image_sha256": image_hash,
                "source_mask_sha256": mask_hash,
                "processed_image_sha256": processed_hash,
                "source_image_bytes": image_bytes,
                "source_mask_bytes": mask_bytes,
                "shape": [2048, 2048],
                "label_count": len(label_ids),
                "label_ids": label_ids,
                "foreground_pixels": foreground_pixels,
                "foreground_fraction": foreground_pixels / float(2048 * 2048),
                "image_processing": image_metadata,
            }
        )
        if index % 8 == 0 or index == len(selected):
            print(f"Prepared KSO pair {index}/{len(selected)}: {stamp}", flush=True)

    canonical = json.dumps(manifest_rows, sort_keys=True, separators=(",", ":")).encode()
    manifest = {
        "status": "complete",
        "name": "Kanzelhoehe Observatory 2023 H-alpha filament maps",
        "source_image_index": IMAGE_INDEX,
        "source_mask_root": MASK_ROOT,
        "source_archive_page_sha256": _sha256(page.encode("utf-8")),
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "license": "Non-profit scientific and educational use is granted; profit use is prohibited.",
        "source_policy": "https://cesar.kso.ac.at/info/datapolicy.php",
        "attribution": ATTRIBUTION,
        "source_notification_requested": True,
        "source_archive_product": "2k x 2k H-alpha synoptic FITS and per-image filament label maps",
        "annotation_type": "automatically detected per-filament integer-valued label maps; weak supervision",
        "source_observations_listed": len(image_names),
        "source_pairs_available": len(available),
        "selected_observations": len(manifest_rows),
        "requested_max_observations": max_observations,
        "competition_split_sha256": json.loads(
            (repository_root / "configs" / "validation-baseline.json").read_text(encoding="utf-8")
        )["expected_split_sha256"],
        "competition_train_test_dates_excluded": True,
        "blocked_competition_dates": len(blocked_dates),
        "exact_competition_name_overlap": 0,
        "competition_ground_truth_used": False,
        "public_MAGFiLO_test_annotations_used": False,
        "inference_uses_external_images": False,
        "selection_manifest_sha256": _sha256(canonical),
        "observations": manifest_rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "status": manifest["status"],
        "listed": len(image_names),
        "available": len(available),
        "selected": len(manifest_rows),
        "foreground_filaments": sum(row["label_count"] for row in manifest_rows),
        "foreground_pixels": sum(row["foreground_pixels"] for row in manifest_rows),
        "selection_manifest_sha256": manifest["selection_manifest_sha256"],
        "output_root": str(output_root),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--competition-root",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "data" / "external" / "kso-weak-2023",
    )
    parser.add_argument("--max-observations", type=int, default=128)
    args = parser.parse_args()
    result = prepare(
        repository_root=args.repo_root,
        competition_root=args.competition_root,
        output_root=args.output_root,
        max_observations=args.max_observations,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
