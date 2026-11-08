# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from astropy.io import fits
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from filament_baseline.data import load_dataset  # noqa: E402


SOURCE_DOI = "10.12149/101545"
SOURCE_PORTAL = "https://nadc.china-vo.org/article/20200519170135"
SOURCE_REPOSITORY = "https://doi.org/10.12149/101545"
SOURCE_PAPER = (
    "Zhu, G., Lin, G., Yang, X., & Zeng, C. (2025). Flat U-Net: An Efficient "
    "Ultralightweight Model for Solar Filament Segmentation in Full-disk H-alpha Images. "
    "The Astrophysical Journal, 980(2), 176. https://doi.org/10.3847/1538-4357/adadff. "
    "Dataset: China-VO PaperData, https://doi.org/10.12149/101545."
)
FROZEN_SPLIT_SHA256 = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"
IMAGE_PATTERN = re.compile(r"^bbso_halph_fl_(?P<date>\d{8})_(?P<time>\d{6})$", re.I)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _archive_members(
    archive_path: Path,
) -> tuple[dict[str, tarfile.TarInfo], dict[str, tarfile.TarInfo], set[str]]:
    fits_by_stem: dict[str, tarfile.TarInfo] = {}
    masks_by_stem: dict[str, tarfile.TarInfo] = {}
    nonempty_masks: set[str] = set()
    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            path = PurePosixPath(member.name.removeprefix("./"))
            parts = path.parts
            if len(parts) != 4 or parts[0:2] != ("data", "train"):
                continue
            if parts[2] == "fits" and path.suffix.lower() == ".fts":
                fits_by_stem[path.stem] = member
            elif parts[2] == "filament_masks" and path.name.lower().endswith("_ntimg.png"):
                stem = path.stem[: -len("_ntimg")]
                masks_by_stem[stem] = member
                payload = archive.extractfile(member).read()
                _, binary = _read_mask(payload, member.name)
                if np.any(binary):
                    nonempty_masks.add(stem)
    if not fits_by_stem or set(fits_by_stem) != set(masks_by_stem):
        missing_masks = sorted(set(fits_by_stem) - set(masks_by_stem))[:5]
        missing_images = sorted(set(masks_by_stem) - set(fits_by_stem))[:5]
        raise ValueError(
            "Flat U-Net training FITS/mask pairs do not match: "
            f"fits={len(fits_by_stem)}, masks={len(masks_by_stem)}, "
            f"missing_masks={missing_masks}, missing_fits={missing_images}"
        )
    return fits_by_stem, masks_by_stem, nonempty_masks


def _normalize_fits(payload: bytes, file_name: str) -> tuple[np.ndarray, dict[str, Any]]:
    with fits.open(io.BytesIO(payload), memmap=False) as hdus:
        arrays = [
            np.asarray(hdu.data)
            for hdu in hdus
            if hdu.data is not None and np.asarray(hdu.data).ndim >= 2
        ]
    if not arrays:
        raise ValueError(f"No image plane in Flat U-Net FITS file: {file_name}")
    image = np.squeeze(arrays[0])
    if image.ndim != 2 or image.shape != (2048, 2048):
        raise ValueError(f"Expected a 2048x2048 FITS image in {file_name}, got {image.shape}")
    finite = np.isfinite(image)
    if float(finite.mean()) < 0.999:
        raise ValueError(f"Too many non-finite pixels in Flat U-Net FITS image: {file_name}")
    low, high = np.percentile(image[finite].astype(np.float32), [0.5, 99.5])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError(f"Invalid robust intensity range in Flat U-Net FITS image: {file_name}")
    scaled = np.clip((image.astype(np.float32) - low) * (255.0 / (high - low)), 0, 255)
    scaled[~finite] = 0
    gray = np.rint(scaled).astype(np.uint8)
    return gray, {
        "fits_dtype": str(image.dtype),
        "fits_shape": list(image.shape),
        "normalization": "clip full-image finite pixels to p0.5-p99.5 and scale to uint8",
        "normalization_low": float(low),
        "normalization_high": float(high),
        "finite_fraction": float(finite.mean()),
    }


def _read_mask(payload: bytes, file_name: str) -> tuple[bytes, np.ndarray]:
    with Image.open(io.BytesIO(payload)) as image:
        mask = np.asarray(image.convert("L"))
    if mask.shape != (2048, 2048):
        raise ValueError(f"Expected a 2048x2048 mask in {file_name}, got {mask.shape}")
    values = np.unique(mask)
    if not set(int(value) for value in values).issubset({0, 255}):
        raise ValueError(f"Expected a binary 0/255 mask in {file_name}, got {values[:16]}")
    return payload, mask


def prepare(
    *,
    repository_root: Path,
    competition_root: Path,
    archive_path: Path,
    output_root: Path,
    max_observations: int,
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    competition_root = competition_root.resolve()
    archive_path = archive_path.resolve()
    output_root = output_root.resolve()
    if not archive_path.is_relative_to(repository_root / "data" / "external"):
        raise ValueError(f"Flat U-Net source archive must stay under data/external/: {archive_path}")
    if not output_root.is_relative_to(repository_root / "data" / "external"):
        raise ValueError(f"Processed data must stay under data/external/: {output_root}")
    if not competition_root.is_relative_to(repository_root / "data"):
        raise ValueError(f"Competition data must stay under data/: {competition_root}")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_root}")
    if max_observations < 1:
        raise ValueError("max_observations must be positive")

    competition = load_dataset(competition_root)
    competition_names = set(competition.observation_names)
    competition_names.update(
        path.name for path in competition.test_image_dir.iterdir() if path.is_file()
    )
    blocked_dates = {name[:8] for name in competition_names}
    fits_by_stem, masks_by_stem, nonempty_masks = _archive_members(archive_path)
    observations_by_date: dict[str, list[str]] = {}
    malformed: list[str] = []
    for stem in nonempty_masks:
        match = IMAGE_PATTERN.fullmatch(stem)
        if match is None:
            malformed.append(stem)
            continue
        observations_by_date.setdefault(match.group("date"), []).append(stem)
    if malformed:
        raise ValueError(f"Unexpected Flat U-Net training filename(s): {malformed[:5]}")

    eligible_dates = sorted(date for date in observations_by_date if date not in blocked_dates)
    if not eligible_dates:
        raise ValueError("No Flat U-Net training dates remain after competition-date exclusion")
    if max_observations > len(eligible_dates):
        raise ValueError(
            f"Requested {max_observations} unique dates, but only {len(eligible_dates)} are eligible"
        )
    selected_indices = np.linspace(0, len(eligible_dates) - 1, max_observations, dtype=int)
    selected_dates = [eligible_dates[int(index)] for index in selected_indices]
    if len(set(selected_dates)) != len(selected_dates):
        raise AssertionError("Evenly spaced selection repeated a calendar date")
    selected_stems = {
        date: min(observations_by_date[date]) for date in selected_dates
    }
    selected_stem_to_date = {stem: date for date, stem in selected_stems.items()}
    if set(selected_stems.values()) & set(competition.observation_names):
        raise AssertionError("Exact Flat U-Net observation overlap with competition training data")
    if set(selected_dates) & blocked_dates:
        raise AssertionError("Flat U-Net selection includes a competition calendar date")

    output_root.mkdir(parents=True, exist_ok=True)
    image_root, mask_root = output_root / "images", output_root / "masks"
    image_root.mkdir(parents=True, exist_ok=True)
    mask_root.mkdir(parents=True, exist_ok=True)
    archive_sha256 = _sha256_file(archive_path)
    expected_fits = {fits_by_stem[stem].name.removeprefix("./"): stem for stem in selected_stems.values()}
    expected_masks = {masks_by_stem[stem].name.removeprefix("./"): stem for stem in selected_stems.values()}
    image_records: dict[str, dict[str, Any]] = {}
    mask_records: dict[str, dict[str, Any]] = {}
    pending_masks: dict[str, tuple[str, bytes, np.ndarray]] = {}

    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            member_name = member.name.removeprefix("./")
            if member_name in expected_fits:
                stem = expected_fits[member_name]
                payload = archive.extractfile(member).read()
                gray, normalization = _normalize_fits(payload, member_name)
                file_name = f"{stem}.png"
                image_path = image_root / file_name
                Image.fromarray(gray).save(image_path, format="PNG", compress_level=3)
                image_records[stem] = {
                    "file_name": file_name,
                    "source_file": member_name,
                    "source_bytes": len(payload),
                    "source_sha256": hashlib.sha256(payload).hexdigest(),
                    "processed_sha256": _sha256_file(image_path),
                    "processed_bytes": image_path.stat().st_size,
                    **normalization,
                }
                if stem in pending_masks:
                    mask_source, mask_payload, binary = pending_masks.pop(stem)
                    mask_records[stem] = _save_mask(
                        mask_root, stem, mask_source, mask_payload, binary
                    )
            elif member_name in expected_masks:
                stem = expected_masks[member_name]
                payload = archive.extractfile(member).read()
                mask_payload, binary = _read_mask(payload, member_name)
                if stem in image_records:
                    mask_records[stem] = _save_mask(mask_root, stem, member_name, mask_payload, binary)
                else:
                    pending_masks[stem] = (member_name, mask_payload, binary)
            if len(image_records) == max_observations and len(mask_records) == max_observations:
                break

    if set(image_records) != set(selected_stems.values()) or set(mask_records) != set(selected_stems.values()):
        raise RuntimeError(
            "Archive extraction was incomplete: "
            f"images={len(image_records)}/{max_observations}, masks={len(mask_records)}/{max_observations}"
        )

    rows = []
    for date in selected_dates:
        stem = selected_stems[date]
        rows.append(
            {
                "date": date,
                "observation_id": stem,
                "file_name": f"{stem}.png",
                "image": image_records[stem],
                "mask": mask_records[stem],
            }
        )
    selection_payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    metadata = {
        "status": "complete",
        "name": "Flat U-Net Solar Filament Segmentation Dataset",
        "source_doi": SOURCE_DOI,
        "source_portal": SOURCE_PORTAL,
        "source_dataset": SOURCE_REPOSITORY,
        "source_paper_citation": SOURCE_PAPER,
        "source_rights_note": (
            "China-VO PaperData states that dataset copyrights remain with the paper authors; "
            "the source page lists no separate dataset license. Used as public, no-cost external "
            "competition data under Kaggle Rule 6 with attribution; do not redistribute the source data."
        ),
        "source_archive": archive_path.name,
        "source_archive_bytes": archive_path.stat().st_size,
        "source_archive_sha256": archive_sha256,
        "source_training_fits_count": len(fits_by_stem),
        "source_training_mask_count": len(masks_by_stem),
        "source_empty_masks_excluded": len(masks_by_stem) - len(nonempty_masks),
        "eligible_training_observations": sum(len(stems) for day, stems in observations_by_date.items() if day not in blocked_dates),
        "eligible_training_dates": len(eligible_dates),
        "competition_dates_excluded": len(blocked_dates),
        "excluded_competition_date_observations": sum(len(observations_by_date[day]) for day in observations_by_date if day in blocked_dates),
        "competition_date_overlap": 0,
        "all_competition_dates_excluded": True,
        "competition_split_sha256": FROZEN_SPLIT_SHA256,
        "selection_method": "evenly spaced unique calendar dates from sorted date-disjoint train observations; earliest filename per selected date",
        "prepared_observations": len(rows),
        "prepared_dates": len(selected_dates),
        "selection_manifest_sha256": hashlib.sha256(selection_payload).hexdigest(),
        "semantic_mask_note": "Source masks are binary semantic maps, not per-filament instance labels; used only as weak pretraining targets.",
        "source_test_gd_used": False,
        "inference_uses_external_images": False,
        "observations": rows,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    notes = (
        "# Flat U-Net external-data provenance\n\n"
        f"Source: {SOURCE_PORTAL} (DOI {SOURCE_DOI}).\n\n"
        f"Citation: {SOURCE_PAPER}\n\n"
        "The China-VO PaperData service states that copyrights remain with the authors; "
        "the dataset record does not state a separate license. This private competition "
        "experiment uses the public, no-cost source under Rule 6 and retains attribution. "
        "Do not redistribute the data or processed copies publicly. The source's designated "
        "test_gd labels are not used for pretraining. All competition train and test calendar "
        "dates are excluded. The weak masks are binary semantic maps rather than instance labels.\n"
    )
    (output_root / "SOURCE_NOTES.md").write_text(notes, encoding="utf-8")
    return {
        "output_root": str(output_root),
        "source_archive_sha256": archive_sha256,
        "source_training_observations": len(fits_by_stem),
        "eligible_training_observations": metadata["eligible_training_observations"],
        "eligible_training_dates": len(eligible_dates),
        "excluded_competition_date_observations": metadata["excluded_competition_date_observations"],
        "prepared_observations": len(rows),
        "selection_manifest_sha256": metadata["selection_manifest_sha256"],
    }


def _save_mask(
    mask_root: Path,
    stem: str,
    source_file: str,
    payload: bytes,
    binary: np.ndarray,
) -> dict[str, Any]:
    file_name = f"{stem}.png"
    path = mask_root / file_name
    path.write_bytes(payload)
    return {
        "file_name": file_name,
        "source_file": source_file,
        "source_bytes": len(payload),
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "processed_sha256": _sha256_file(path),
        "processed_bytes": path.stat().st_size,
        "shape": list(binary.shape),
        "values": [0, 255],
        "positive_fraction": float((binary > 127).mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a date-disjoint Flat U-Net weak-label subset under data/external/."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--competition-data-root",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=REPO_ROOT / "data" / "external" / "flat-unet-101545" / "solar-filament-segmentation-dataset.tar.gz",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "data" / "external" / "flat-unet-101545" / "processed-256",
    )
    parser.add_argument("--max-observations", type=int, default=256)
    args = parser.parse_args()
    result = prepare(
        repository_root=args.repo_root,
        competition_root=args.competition_data_root,
        archive_path=args.archive,
        output_root=args.output_root,
        max_observations=args.max_observations,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
