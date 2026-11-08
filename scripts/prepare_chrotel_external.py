# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import tarfile
import time
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np
from astropy.io import fits

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from filament_baseline.data import load_dataset  # noqa: E402
from filament_baseline.split import create_observation_split  # noqa: E402

API_ROOT = "https://archive.sdc.leibniz-kis.de/api/instruments/chrotel"
DATA_CALENDAR = "https://archive.sdc.leibniz-kis.de/data-calendar"
KIS_POLICY = "https://sdc.leibniz-kis.de/en/license-policy/citation"
METHOD_PAPER = "https://doi.org/10.1051/0004-6361/202348314"
PAGE_SIZE = 200
FILENAME_PATTERN = re.compile(
    r"^chrotel_(?P<date>\d{8})_(?P<time>\d{6})_ha$", re.IGNORECASE
)


def _urlopen(request: urllib.request.Request, *, timeout: int):
    """Open public KIS HTTPS endpoints using the platform's trusted CA store."""
    return urllib.request.urlopen(request, timeout=timeout)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _selection_digest(rows: list[dict[str, Any]]) -> str:
    canonical_rows = [
        {
            "observation_id": row["observation_id"],
            "date": row["date"],
            "file_name": row["file_name"],
            "source_timestamp": row["source_timestamp"],
        }
        for row in rows
    ]
    canonical = json.dumps(
        canonical_rows, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _validate_batch_archive(path: Path, observation_ids: list[str]) -> None:
    expected = set(observation_ids)
    products: dict[str, set[str]] = {}
    with tarfile.open(path, mode="r:") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            parts = PurePosixPath(member.name).parts
            if len(parts) < 4 or parts[0] not in expected or member.size <= 0:
                raise ValueError(f"Invalid member in ChroTel batch archive: {member.name}")
            if parts[1:3] == ("l2", "l2data"):
                kind = "image"
            elif parts[1:3] == ("l2", "l2mask"):
                kind = "mask"
            else:
                raise ValueError(f"Unexpected ChroTel product path: {member.name}")
            kinds = products.setdefault(parts[0], set())
            if kind in kinds:
                raise ValueError(f"Duplicate ChroTel {kind} product: {member.name}")
            kinds.add(kind)
    if set(products) != expected or any(kinds != {"image", "mask"} for kinds in products.values()):
        raise ValueError("ChroTel batch archive is incomplete")


def _api_records() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    page = 0
    while True:
        query = urllib.parse.urlencode(
            {
                "page": page,
                "size": PAGE_SIZE,
                "wavelengthFilters": "H-alpha",
                "highlevelFilters": "Filament-Segmentation",
                "sort": "metadata.header.date_beg,asc",
            }
        )
        request = urllib.request.Request(
            f"{API_ROOT}/observations?{query}",
            headers={"User-Agent": "EquiNexus challenge research/1.0"},
        )
        with _urlopen(request, timeout=45) as response:
            payload = json.load(response)
        content = payload.get("content", [])
        rows.extend(content)
        if len(rows) >= int(payload.get("totalElements", len(rows))):
            break
        if not content:
            raise RuntimeError("ChroTel API pagination stopped before its reported total")
        page += 1
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("ChroTel API returned duplicate observation IDs")
    return rows


def _prepare_selection(
    *,
    competition_root: Path,
    output_root: Path,
    max_observations: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    competition = load_dataset(competition_root)
    split = create_observation_split(
        competition,
        REPO_ROOT / "data" / "splits",
        seed=42,
        validation_fraction=0.2,
    )
    expected_split = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"
    if split.sha256 != expected_split:
        raise RuntimeError(f"Frozen competition split changed: {split.sha256}")

    blocked_dates = {name[:8] for name in competition.observation_names}
    blocked_dates.update(
        path.stem[:8]
        for path in competition.test_image_dir.iterdir()
        if path.is_file()
    )
    raw_rows = _api_records()
    eligible: list[dict[str, Any]] = []
    for row in raw_rows:
        observation_id = str(row["id"])
        match = FILENAME_PATTERN.fullmatch(observation_id)
        if match is None:
            raise ValueError(f"Unexpected ChroTel observation ID: {observation_id}")
        date = match.group("date")
        if date in blocked_dates:
            continue
        timestamp = str(row["metadata"]["header"]["dateBeg"])
        file_name = f"{observation_id}.png"
        eligible.append(
            {
                "observation_id": observation_id,
                "date": date,
                "file_name": file_name,
                "source_timestamp": timestamp,
            }
        )
    eligible.sort(key=lambda row: (row["date"], row["source_timestamp"], row["observation_id"]))
    if max_observations is not None and len(eligible) > max_observations:
        indices = np.linspace(0, len(eligible) - 1, max_observations, dtype=int)
        selected = [eligible[int(index)] for index in indices]
    else:
        selected = eligible
    if not selected:
        raise ValueError("No ChroTel observations remain after competition-date exclusion")

    selection = {
        "source": "KIS Science Data Centre ChroTel H-alpha Filament-Segmentation products",
        "api": API_ROOT,
        "calendar": DATA_CALENDAR,
        "policy": KIS_POLICY,
        "paper": METHOD_PAPER,
        "license_status": (
            "KIS SDC states that ChroTel data are freely available for scientific and educational use; "
            "citation and acknowledgement are required. This project retains attribution and does not redistribute source FITS files."
        ),
        "competition_rule_basis": "Publicly accessible at no charge; external weak-label pretraining under Rule 6.",
        "source_products_reported_by_api": len(raw_rows),
        "same_calendar_date_products_excluded": len(raw_rows) - len(eligible),
        "eligible_date_disjoint_products": len(eligible),
        "selected_observations": len(selected),
        "max_observations": max_observations,
        "selection_rule": "evenly sample sorted date-disjoint API records with rounded linspace indices",
        "blocked_competition_dates": len(blocked_dates),
        "competition_date_overlap": 0,
        "all_competition_dates_excluded": True,
        "frozen_competition_split_sha256": split.sha256,
        "observations": selected,
        "selection_manifest_sha256": _selection_digest(selected),
    }
    _write_json(output_root / "selection.json", selection)
    return selected, selection


def _download_batch(observation_ids: list[str], destination: Path) -> str:
    query = urllib.parse.urlencode(
        {
            "observations": ",".join(observation_ids),
            "buckets": "level2,segmentation",
        }
    )
    request = urllib.request.Request(
        f"{API_ROOT}/observations/download?{query}",
        headers={"User-Agent": "EquiNexus challenge research/1.0"},
    )
    temporary = destination.with_suffix(destination.suffix + ".partial")
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        digest = hashlib.sha256()
        try:
            with _urlopen(request, timeout=180) as response, temporary.open("wb") as stream:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    stream.write(block)
                    digest.update(block)
            _validate_batch_archive(temporary, observation_ids)
            os.replace(temporary, destination)
            return digest.hexdigest()
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def _decode_primary_fits(payload: bytes) -> tuple[np.ndarray, str]:
    with fits.open(io.BytesIO(payload), memmap=False) as hdul:
        data = np.asarray(hdul[0].data)
        timestamp = str(hdul[0].header.get("DATE-OBS", ""))
    return data, timestamp


def _to_uint8_image(data: np.ndarray) -> np.ndarray:
    values = np.asarray(data, dtype=np.float32)
    valid = np.isfinite(values) & (values > 0)
    if not np.any(valid):
        raise ValueError("ChroTel FITS image contains no positive finite solar pixels")
    low, high = np.quantile(values[valid], (0.01, 0.995))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("ChroTel FITS image has a degenerate intensity range")
    scaled = np.zeros(values.shape, dtype=np.uint8)
    scaled[valid] = np.round(
        np.clip((values[valid] - low) / (high - low), 0.0, 1.0) * 255
    ).astype(np.uint8)
    if scaled.shape != (2048, 2048):
        scaled = cv2.resize(scaled, (2048, 2048), interpolation=cv2.INTER_CUBIC)
    return scaled


def _to_binary_mask(data: np.ndarray) -> np.ndarray:
    values = np.asarray(data)
    if values.shape != (2000, 2000):
        raise ValueError(f"Unexpected ChroTel mask dimensions: {values.shape}")
    binary = (values > 0).astype(np.uint8) * 255
    return cv2.resize(binary, (2048, 2048), interpolation=cv2.INTER_NEAREST)


def _write_png(path: Path, array: np.ndarray) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.png")
    if not cv2.imwrite(str(temporary), array, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise OSError(f"Could not write ChroTel PNG: {path}")
    os.replace(temporary, path)
    return _sha256_file(path), path.stat().st_size


def _process_batch(
    *,
    archive_path: Path,
    selected_rows: list[dict[str, Any]],
    output_root: Path,
) -> list[dict[str, Any]]:
    selected_by_id = {row["observation_id"]: row for row in selected_rows}
    pairs: dict[str, dict[str, tuple[np.ndarray, str]]] = {}
    with tarfile.open(archive_path, mode="r:") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            parts = PurePosixPath(member.name).parts
            if len(parts) < 4 or parts[0] not in selected_by_id:
                raise ValueError(f"Unexpected member in ChroTel batch archive: {member.name}")
            if parts[1:3] == ("l2", "l2data"):
                kind = "image"
            elif parts[1:3] == ("l2", "l2mask"):
                kind = "mask"
            else:
                raise ValueError(f"Unexpected ChroTel product path: {member.name}")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"Could not read ChroTel product: {member.name}")
            array, timestamp = _decode_primary_fits(extracted.read())
            if array.shape != (2000, 2000):
                raise ValueError(f"Unexpected ChroTel {kind} dimensions in {member.name}: {array.shape}")
            pair = pairs.setdefault(parts[0], {})
            if kind in pair:
                raise ValueError(f"Duplicate ChroTel {kind} product for {parts[0]}")
            pair[kind] = (array, timestamp)
    if set(pairs) != set(selected_by_id):
        missing = sorted(set(selected_by_id) - set(pairs))
        extra = sorted(set(pairs) - set(selected_by_id))
        raise ValueError(f"ChroTel batch archive inventory mismatch; missing={missing}, extra={extra}")

    rows = []
    for observation_id in sorted(selected_by_id):
        pair = pairs[observation_id]
        if set(pair) != {"image", "mask"}:
            raise ValueError(f"ChroTel image-mask pair is incomplete: {observation_id}")
        image_data, image_timestamp = pair["image"]
        mask_data, mask_timestamp = pair["mask"]
        if not image_timestamp or image_timestamp != mask_timestamp:
            raise ValueError(f"ChroTel image and mask timestamps differ: {observation_id}")
        expected_timestamp = str(selected_by_id[observation_id]["source_timestamp"])
        if image_timestamp[:19] != expected_timestamp[:19]:
            raise ValueError(
                f"ChroTel FITS timestamp differs from API metadata for {observation_id}: "
                f"{image_timestamp} != {expected_timestamp}"
            )
        file_name = selected_by_id[observation_id]["file_name"]
        image_path = output_root / "images" / file_name
        mask_path = output_root / "masks" / file_name
        image_sha, image_bytes = _write_png(image_path, _to_uint8_image(image_data))
        mask = _to_binary_mask(mask_data)
        mask_sha, mask_bytes = _write_png(mask_path, mask)
        rows.append(
            {
                **selected_by_id[observation_id],
                "source_products": ["l2/l2data", "l2/l2mask"],
                "image_sha256": image_sha,
                "image_bytes": image_bytes,
                "mask_sha256": mask_sha,
                "mask_bytes": mask_bytes,
                "mask_area_fraction": float(np.mean(mask > 127)),
                "source_image_shape": [2000, 2000],
                "prepared_image_shape": [2048, 2048],
            }
        )
    return rows


def prepare(
    *,
    competition_root: Path,
    output_root: Path,
    max_observations: int | None,
    batch_size: int,
    dry_run: bool,
    delete_archives_after_extraction: bool = False,
) -> dict[str, Any]:
    repo_root = REPO_ROOT.resolve()
    competition_root = competition_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    kaggle_competition_root = (
        competition_root == Path("/kaggle/input")
        or Path("/kaggle/input") in competition_root.parents
    )
    if not competition_root.is_relative_to(repo_root / "data") and not kaggle_competition_root:
        raise ValueError(f"Competition data must remain under data/: {competition_root}")
    if not output_root.is_relative_to(repo_root / "data" / "external"):
        raise ValueError(f"ChroTel files must remain under data/external/: {output_root}")
    if batch_size < 1 or (max_observations is not None and max_observations < 1):
        raise ValueError("batch_size and max_observations must be positive")
    selected, selection = _prepare_selection(
        competition_root=competition_root,
        output_root=output_root,
        max_observations=max_observations,
    )
    if dry_run:
        return {**selection, "status": "selection-only"}

    archives_root = output_root / "archives"
    archives_root.mkdir(parents=True, exist_ok=True)
    completed: dict[str, dict[str, Any]] = {}
    existing_manifest_path = output_root / "manifest.json"
    if existing_manifest_path.is_file():
        existing = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        if existing.get("selection_manifest_sha256") != selection["selection_manifest_sha256"]:
            raise ValueError("Existing ChroTel preparation uses a different selection manifest")
        completed = {
            str(row["observation_id"]): row for row in existing.get("observations", [])
        }

    archive_records = []
    if existing_manifest_path.is_file():
        archive_records = list(existing.get("archives", []))
    for batch_index, offset in enumerate(range(0, len(selected), batch_size)):
        batch = selected[offset : offset + batch_size]
        pending = [row for row in batch if row["observation_id"] not in completed]
        if not pending:
            continue
        archive_path = archives_root / f"batch-{batch_index:04d}.tar"
        if archive_path.is_file():
            try:
                _validate_batch_archive(
                    archive_path, [row["observation_id"] for row in pending]
                )
            except (OSError, tarfile.TarError, ValueError):
                archive_path.unlink()
                archive_sha = _download_batch(
                    [row["observation_id"] for row in pending], archive_path
                )
            else:
                archive_sha = _sha256_file(archive_path)
        else:
            archive_sha = _download_batch(
                [row["observation_id"] for row in pending], archive_path
            )
        batch_rows = _process_batch(
            archive_path=archive_path,
            selected_rows=pending,
            output_root=output_root,
        )
        completed.update({row["observation_id"]: row for row in batch_rows})
        archive_records.append(
            {
                "file": archive_path.relative_to(output_root).as_posix(),
                "sha256": archive_sha,
                "bytes": archive_path.stat().st_size,
                "observations": [row["observation_id"] for row in pending],
            }
        )
        archive_bytes = archive_path.stat().st_size
        if delete_archives_after_extraction:
            archive_path.unlink()
        current = _build_manifest(selection, completed, archive_records, status="in_progress")
        _write_json(existing_manifest_path, current)
        print(
            json.dumps(
                {
                    "batch": batch_index,
                    "completed_observations": len(completed),
                    "target_observations": len(selected),
                    "archive_bytes": archive_bytes,
                    "archive_retained": not delete_archives_after_extraction,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if len(completed) != len(selected):
        raise RuntimeError(f"Only prepared {len(completed)} of {len(selected)} ChroTel observations")
    final = _build_manifest(selection, completed, archive_records, status="complete")
    _write_json(existing_manifest_path, final)
    return final


def _build_manifest(
    selection: dict[str, Any],
    completed: dict[str, dict[str, Any]],
    archive_records: list[dict[str, Any]],
    *,
    status: str,
) -> dict[str, Any]:
    rows = [completed[key] for key in sorted(completed)]
    return {
        "status": status,
        "dataset": selection["source"],
        "source_api": selection["api"],
        "source_calendar": selection["calendar"],
        "source_policy": selection["policy"],
        "paper_citation": selection["paper"],
        "license_status": selection["license_status"],
        "competition_rule_basis": selection["competition_rule_basis"],
        "preprocessing": {
            "input": "KIS SDC Level-2 ChroTel H-alpha intensity-corrected FITS",
            "target": "KIS SDC Level-2 semantic filament segmentation FITS",
            "source_shape": [2000, 2000],
            "prepared_shape": [2048, 2048],
            "image_scaling": "finite positive image pixels mapped between p1 and p99.5 to uint8; off-disk pixels set to zero",
            "image_resize": "OpenCV INTER_CUBIC",
            "mask_threshold": "source value greater than zero",
            "mask_resize": "OpenCV INTER_NEAREST",
            "prepared_format": "lossless grayscale PNG",
        },
        "selection_manifest_sha256": selection["selection_manifest_sha256"],
        "selection": {
            key: selection[key]
            for key in (
                "source_products_reported_by_api",
                "same_calendar_date_products_excluded",
                "eligible_date_disjoint_products",
                "selected_observations",
                "selection_rule",
                "blocked_competition_dates",
                "competition_date_overlap",
                "all_competition_dates_excluded",
                "frozen_competition_split_sha256",
            )
        },
        "blocked_competition_dates": selection["blocked_competition_dates"],
        "competition_date_overlap": selection["competition_date_overlap"],
        "all_competition_dates_excluded": selection["all_competition_dates_excluded"],
        "archives": archive_records,
        "observations": rows,
        "prepared_observations": len(rows),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Audit and prepare date-disjoint ChroTel H-alpha weak-label data under data/external/."
    )
    parser.add_argument(
        "--competition-root",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026",
    )
    parser.add_argument(
        "--output-root", type=Path, default=REPO_ROOT / "data" / "external" / "chrotel"
    )
    parser.add_argument("--max-observations", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delete-archives-after-extraction", action="store_true")
    arguments = parser.parse_args()
    result = prepare(
        competition_root=arguments.competition_root,
        output_root=arguments.output_root,
        max_observations=arguments.max_observations,
        batch_size=arguments.batch_size,
        dry_run=arguments.dry_run,
        delete_archives_after_extraction=arguments.delete_archives_after_extraction,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_observations": result.get("selected_observations")
                or result.get("selection", {}).get("selected_observations"),
                "prepared_observations": result.get("prepared_observations"),
                "competition_date_overlap": result.get("selection", {}).get(
                    "competition_date_overlap", result.get("competition_date_overlap")
                ),
                "selection_manifest_sha256": result.get("selection_manifest_sha256"),
            },
            indent=2,
            sort_keys=True,
        )
    )
