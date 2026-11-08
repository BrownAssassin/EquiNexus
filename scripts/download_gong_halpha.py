"""Download a timestamp-audited, unlabeled sample from the public NSO GONG archive."""
# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only


from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import html.parser
import json
import re
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = "https://gong2.nso.edu/ftp/HA/hag"
SCIENCE_SITES = set("BCLMTU")
FILENAME_RE = re.compile(r"^(\d{14})([BCLMTU])h\.jpg$", re.IGNORECASE)


class _ImageLinks(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.names: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href and FILENAME_RE.fullmatch(href):
            self.names.append(href)


def _request(url: str, attempts: int = 4) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "EquiNexus-filament-challenge/1.0 (public archive research)"},
    )
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return response.read()
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(1.0 + attempt)
    assert last_error is not None
    raise last_error


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _date_range(start: date, end: date, step_days: int = 1):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=step_days)


def _select_daily_files(day: date, cadence_minutes: int, sites: set[str]) -> list[dict[str, str]]:
    folder = day.strftime("%Y%m/%Y%m%d")
    index_url = f"{ARCHIVE_ROOT}/{folder}/"
    parser = _ImageLinks()
    parser.feed(_request(index_url).decode("latin-1"))

    buckets: dict[tuple[str, int], dict[str, str]] = {}
    for name in sorted(parser.names):
        match = FILENAME_RE.fullmatch(name)
        if not match:
            continue
        stamp, site = match.groups()
        site = site.upper()
        if site not in sites:
            continue
        timestamp = datetime.strptime(stamp, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        bucket = int(timestamp.timestamp()) // (cadence_minutes * 60)
        buckets.setdefault((site, bucket), {"filename": name, "url": f"{index_url}{name}"})
    return sorted(buckets.values(), key=lambda item: item["filename"])


def _competition_stems(data_root: Path) -> set[str]:
    if not data_root.exists():
        return set()
    return {
        path.stem
        for path in data_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }


def _download_one(item: dict[str, str], image_dir: Path) -> dict[str, Any]:
    destination = image_dir / item["filename"]
    if destination.is_file() and destination.stat().st_size > 0:
        payload = destination.read_bytes()
    else:
        payload = _request(item["url"])
        if not payload.startswith(b"\xff\xd8\xff"):
            raise ValueError(f"Archive item is not a JPEG image: {item['url']}")
        temporary = destination.with_suffix(destination.suffix + ".part")
        temporary.write_bytes(payload)
        temporary.replace(destination)
    match = FILENAME_RE.fullmatch(item["filename"])
    assert match is not None
    timestamp = datetime.strptime(match.group(1), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    return {
        "filename": item["filename"],
        "url": item["url"],
        "timestamp_utc": timestamp.isoformat(),
        "site": match.group(2).upper(),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", type=_parse_date, required=True, help="Inclusive UTC date, YYYY-MM-DD")
    parser.add_argument("--end-date", type=_parse_date, required=True, help="Inclusive UTC date, YYYY-MM-DD")
    parser.add_argument(
        "--date-step-days",
        type=int,
        default=1,
        help="Index one archive day every N days (default: 1)",
    )
    parser.add_argument("--cadence-minutes", type=int, default=10, help="Keep at most one image/site/time bin")
    parser.add_argument("--sites", default="BCLMTU", help="GONG science site codes (default: all six)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/external/gong_halpha"),
        help="Output directory, relative to the repository unless absolute",
    )
    parser.add_argument(
        "--competition-data-root",
        type=Path,
        default=REPOSITORY_ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026",
        help="Read-only competition input root used to reject exact filename overlap",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent archive index requests and image downloads; keep this modest",
    )
    parser.add_argument(
        "--expected-content-manifest-sha256",
        default=None,
        help="Optional digest assertion for a previously audited file selection",
    )
    parser.add_argument("--dry-run", action="store_true", help="List the selected archive entries only")
    args = parser.parse_args()

    if args.end_date < args.start_date:
        parser.error("--end-date must be on or after --start-date")
    if args.cadence_minutes < 1:
        parser.error("--cadence-minutes must be at least 1")
    if args.date_step_days < 1:
        parser.error("--date-step-days must be at least 1")
    if args.workers < 1 or args.workers > 8:
        parser.error("--workers must be between 1 and 8")
    sites = set(args.sites.upper())
    if not sites or not sites <= SCIENCE_SITES:
        parser.error("--sites must contain only science-grade GONG sites: B, C, L, M, T, U")

    days = tuple(_date_range(args.start_date, args.end_date, args.date_step_days))
    selected: list[dict[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_select_daily_files, day, args.cadence_minutes, sites): day
            for day in days
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            day = futures[future]
            daily = future.result()
            selected.extend(daily)
            if index % 50 == 0 or index == len(futures):
                print(
                    f"Indexed {index}/{len(futures)} days; "
                    f"{day.isoformat()} selected {len(daily)} files",
                    flush=True,
                )
    selected.sort(key=lambda item: item["filename"])

    competition_stems = _competition_stems(args.competition_data_root)
    overlaps = sorted({Path(item["filename"]).stem for item in selected} & competition_stems)
    if overlaps:
        raise RuntimeError(f"External image timestamp(s) overlap competition files: {overlaps[:10]}")

    print(f"Selected total: {len(selected)}; exact filename overlap: 0")
    if args.dry_run:
        return

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = REPOSITORY_ROOT / output_dir
    output_dir = output_dir.resolve()
    if REPOSITORY_ROOT not in output_dir.parents:
        parser.error("--output-dir must stay inside the repository workspace")
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_download_one, item, image_dir) for item in selected]
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            records.append(future.result())
            if index % 100 == 0 or index == len(futures):
                print(f"Downloaded and hashed {index}/{len(futures)}")

    records.sort(key=lambda item: item["filename"])
    content_manifest = json.dumps(records, sort_keys=True, separators=(",", ":"))
    content_digest = hashlib.sha256(content_manifest.encode("utf-8")).hexdigest()
    if args.expected_content_manifest_sha256 and content_digest != args.expected_content_manifest_sha256:
        raise RuntimeError(
            "Downloaded archive content manifest changed: "
            f"expected {args.expected_content_manifest_sha256}, received {content_digest}"
        )
    payload = {
        "source": "NSF National Solar Observatory GONG H-alpha public archive",
        "archive_root": ARCHIVE_ROOT,
        "image_credit": "NSO/AURA/NSF",
        "data_acknowledgement": (
            "Data were acquired by GONG instruments operated by NISP/NSO/AURA/NSF "
            "with contribution from NOAA."
        ),
        "image_use_policy": "https://nso.edu/about/image-use-policy/",
        "accessed_utc": datetime.now(timezone.utc).isoformat(),
        "selection": {
            "start_date_utc": args.start_date.isoformat(),
            "end_date_utc": args.end_date.isoformat(),
            "date_step_days": args.date_step_days,
            "cadence_minutes_per_site": args.cadence_minutes,
            "sites": "".join(sorted(sites)),
            "selection_rule": "first archive filename per sampled UTC day, cadence bin, and site",
        },
        "purpose": "unlabeled external imagery for self-supervised solar-domain adaptation only",
        "competition_overlap": {"matched_exact_filenames": 0, "checked_files": len(competition_stems)},
        "count": len(records),
        "total_bytes": sum(item["bytes"] for item in records),
        "content_manifest_sha256": content_digest,
        "files": records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest: {manifest_path.relative_to(REPOSITORY_ROOT)}")
    print(f"Files: {len(records)}; bytes: {payload['total_bytes']}")


if __name__ == "__main__":
    main()
