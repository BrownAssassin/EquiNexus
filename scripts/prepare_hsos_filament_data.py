# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import sys
import urllib.request
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from filament_baseline.data import load_dataset
from filament_baseline.split import create_observation_split


IMAGE_INDEX_URL = "https://sun.bao.ac.cn/hsos_data/download/filaments-unet-dataset/img/"
MASK_INDEX_URL = "https://sun.bao.ac.cn/hsos_data/download/filaments-unet-dataset/mask/"
IMAGE_PATTERN = re.compile(r"bbso_halph_fi_(\d{8})_(\d{6})_nimg\.jpg\Z", re.IGNORECASE)
PAPER_CITATION = (
    "Zhu, G., Lin, G., Wang, D. et al. Solar Filament Recognition Based on Deep Learning. "
    "Solar Physics 294, 117 (2019). https://doi.org/10.1007/s11207-019-1517-4"
)


class _DirectoryLinks(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.hrefs.add(Path(urlparse(href).path).name)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _read_listing(url: str) -> set[str]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "EquiNexus challenge research data audit"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    parser = _DirectoryLinks()
    parser.feed(payload.decode("utf-8", errors="replace"))
    return parser.hrefs


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _download_one(
    record: dict[str, str], output_root: Path
) -> dict[str, Any]:
    image_path = output_root / "images" / record["image_file"]
    mask_path = output_root / "masks" / record["mask_file"]
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {**record}
    dimensions: dict[str, tuple[int, int]] = {}
    for url_key, path in (("image_url", image_path), ("mask_url", mask_path)):
        if path.is_file():
            payload = path.read_bytes()
        else:
            request = urllib.request.Request(
                record[url_key],
                headers={"User-Agent": "EquiNexus challenge research data audit"},
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = response.read()
            temporary = path.with_suffix(path.suffix + ".partial")
            temporary.write_bytes(payload)
            temporary.replace(path)
        with Image.open(path) as image:
            if (
                image.format != "JPEG"
                or min(image.size) < 1024
                or max(image.size) > 2048
                or max(image.size) / min(image.size) > 1.02
            ):
                raise ValueError(
                    f"Unexpected image format or dimensions for {path.name}: "
                    f"{image.format} {image.size}"
                )
            image.load()
            dimensions[url_key.removesuffix("_url")] = image.size
        result[url_key.removesuffix("_url") + "_sha256"] = _sha256(payload)
        result[url_key.removesuffix("_url") + "_bytes"] = len(payload)
        result[url_key.removesuffix("_url") + "_path"] = str(path.relative_to(output_root))
    if dimensions["image"] != dimensions["mask"]:
        raise ValueError(
            f"Image and mask dimensions differ for {record['timestamp']}: {dimensions}"
        )
    result["width"], result["height"] = dimensions["image"]
    return result


def prepare(
    *,
    repo_root: Path,
    competition_data_root: Path,
    output_root: Path,
    samples_per_day: int,
    max_samples: int | None,
    workers: int,
    dry_run: bool,
) -> dict[str, Any]:
    if not _inside(output_root, repo_root):
        raise ValueError(f"External data must remain under the repository: {output_root}")
    if not _inside(competition_data_root, repo_root):
        raise ValueError(
            f"Competition data must remain under the repository: {competition_data_root}"
        )
    if samples_per_day < 1 or workers < 1:
        raise ValueError("samples_per_day and workers must be positive")
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive when supplied")

    dataset = load_dataset(competition_data_root)
    split = create_observation_split(
        dataset,
        repo_root / "data" / "splits",
        seed=42,
        validation_fraction=0.2,
    )
    if split.sha256 != "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e":
        raise RuntimeError(f"Frozen competition split digest changed: {split.sha256}")

    blocked_dates = {name[:8] for name in dataset.observation_names}
    blocked_dates.update(
        path.name[:8]
        for path in dataset.test_image_dir.iterdir()
        if path.is_file()
    )
    competition_timestamps = {
        Path(name).stem[:14] for name in dataset.observation_names
    }
    competition_timestamps.update(
        path.stem[:14]
        for path in dataset.test_image_dir.iterdir()
        if path.is_file()
    )

    image_names = _read_listing(IMAGE_INDEX_URL)
    mask_names = _read_listing(MASK_INDEX_URL)
    by_date: dict[str, list[dict[str, str]]] = defaultdict(list)
    blocked_date_image_count = 0
    exact_competition_timestamp_count = 0
    missing_masks = 0
    for image_name in sorted(image_names):
        match = IMAGE_PATTERN.fullmatch(image_name)
        if match is None:
            continue
        date, clock = match.groups()
        if date in blocked_dates:
            blocked_date_image_count += 1
            continue
        timestamp = date + clock
        if timestamp in competition_timestamps:
            exact_competition_timestamp_count += 1
            continue
        mask_name = image_name.replace("_nimg.jpg", "_ntimg.jpg")
        if mask_name not in mask_names:
            missing_masks += 1
            continue
        by_date[date].append(
            {
                "date": date,
                "timestamp": timestamp,
                "image_file": image_name,
                "mask_file": mask_name,
                "image_url": urljoin(IMAGE_INDEX_URL, image_name),
                "mask_url": urljoin(MASK_INDEX_URL, mask_name),
            }
        )

    selected: list[dict[str, str]] = []
    for date in sorted(by_date):
        daily = by_date[date]
        count = min(samples_per_day, len(daily))
        if count == len(daily):
            indices = list(range(len(daily)))
        else:
            indices = sorted(
                {round(index * (len(daily) - 1) / max(count - 1, 1)) for index in range(count)}
            )
        selected.extend(daily[index] for index in indices)
    if max_samples is not None:
        selected = selected[:max_samples]

    manifest: dict[str, Any] = {
        "dataset": "HSOS BBSO H-alpha filament-unet dataset",
        "source_image_index": IMAGE_INDEX_URL,
        "source_mask_index": MASK_INDEX_URL,
        "source_code": "https://github.com/GF-Zhu/Filament-Unet",
        "source_recommendation": "The source repository recommends manual review of the ground-truth masks.",
        "paper_citation": PAPER_CITATION,
        "license_status": "Public archive; no standalone dataset license is stated in the directory listing.",
        "competition_rule_basis": "Publicly accessible source data used as external training data under Rule 6.",
        "split_sha256": split.sha256,
        "sampling": {
            "samples_per_day": samples_per_day,
            "selected_samples": len(selected),
            "selected_dates": len({record["date"] for record in selected}),
            "blocked_competition_dates": len(blocked_dates),
            "blocked_images_on_competition_dates": blocked_date_image_count,
            "exact_competition_timestamps_excluded": exact_competition_timestamp_count,
            "missing_mask_pairs": missing_masks,
            "all_competition_dates_excluded": True,
        },
        "downloads": [],
    }
    canonical = json.dumps(selected, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest["selected_source_manifest_sha256"] = _sha256(canonical)
    if dry_run:
        return manifest

    output_root.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_download_one, record, output_root) for record in selected]
        manifest["downloads"] = [future.result() for future in futures]
    manifest["downloaded_image_bytes"] = sum(
        record["image_bytes"] for record in manifest["downloads"]
    )
    manifest["downloaded_mask_bytes"] = sum(
        record["mask_bytes"] for record in manifest["downloads"]
    )
    external_hashes = {record["image_sha256"] for record in manifest["downloads"]}
    competition_paths = [
        path
        for folder in (dataset.train_image_dir, dataset.test_image_dir)
        for path in folder.iterdir()
        if path.is_file()
    ]
    competition_hashes = {
        _sha256(path.read_bytes()) for path in competition_paths
    }
    overlaps = sorted(external_hashes & competition_hashes)
    manifest["overlap_audit"] = {
        "competition_image_files_hashed": len(competition_paths),
        "external_image_files_hashed": len(external_hashes),
        "exact_byte_identical_images": len(overlaps),
        "exact_overlap_sha256": overlaps,
        "all_competition_dates_excluded": True,
    }
    if overlaps:
        raise ValueError(f"HSOS images overlap competition image bytes: {overlaps}")
    output_path = output_root / "manifest.json"
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit or download date-disjoint samples from the public HSOS filament dataset."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--competition-data-root",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "data" / "external" / "hsos_filaments_unet_2013",
    )
    parser.add_argument("--samples-per-day", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = prepare(
        repo_root=args.repo_root.expanduser().resolve(),
        competition_data_root=args.competition_data_root.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
        samples_per_day=args.samples_per_day,
        max_samples=args.max_samples,
        workers=args.workers,
        dry_run=args.dry_run,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "downloads"}, indent=2, sort_keys=True))
    if not args.dry_run:
        print(f"Downloaded {len(result['downloads'])} image/mask pairs to {args.output_root}")
        print(f"Manifest SHA-256: {result['selected_source_manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
