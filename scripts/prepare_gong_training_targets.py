#!/usr/bin/env python3
# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Download GONG magnetogram targets only for the frozen training partition.

This helper never downloads targets for the frozen validation observations. It
uses the Kaggle-hosted public GONG dataset and stores all files under data/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from filament_baseline.data import load_dataset
from filament_baseline.split import create_observation_split

DATASET_SLUG = "vhrabar/gong-solar-filaments"
DATASET_URL = "https://www.kaggle.com/datasets/vhrabar/gong-solar-filaments"
EXPECTED_SPLIT_SHA256 = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"
TARGET_SIZE = 512
TARGET_CHANNELS = 3
_API: Any | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def kaggle_cli_path() -> str:
    candidates = (
        REPO_ROOT / ".venv" / "Scripts" / "kaggle.exe",
        REPO_ROOT / ".venv" / "bin" / "kaggle",
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    located = shutil.which("kaggle")
    if located:
        return located
    raise FileNotFoundError("Kaggle CLI not found in the workspace .venv or PATH")


def list_target_files(cli: str) -> list[dict[str, Any]]:
    token: str | None = None
    found_targets = False
    files: list[dict[str, Any]] = []
    page_count = 0
    while True:
        command = [cli, "datasets", "files", DATASET_SLUG, "--page-size", "200", "--format", "json"]
        if token:
            command.extend(("--page-token", token))
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(
                f"Kaggle file listing failed on page {page_count + 1}: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        output = completed.stdout + "\n" + completed.stderr
        page_token = re.search(r"(?m)^Next Page Token = ([^\r\n]+)", output)
        json_start = output.find("[")
        if json_start < 0:
            raise RuntimeError(f"Kaggle file listing page {page_count + 1} had no JSON array")
        rows = json.loads(output[json_start:])
        if not isinstance(rows, list):
            raise RuntimeError("Kaggle file listing response is not a list")
        page_count += 1
        page_finished_targets = False
        for row in rows:
            name = str(row.get("name", ""))
            if name.startswith("bzi_targets/") and name.lower().endswith(".npy"):
                files.append({"name": name, "size": int(row["size"])})
                found_targets = True
            elif name.startswith("bzi_targets/"):
                found_targets = True
            elif found_targets:
                page_finished_targets = True
                break
        print(
            f"[listing] page={page_count} rows={len(rows)} target_maps={len(files)}",
            flush=True,
        )
        if page_finished_targets:
            break
        token = page_token.group(1) if page_token else None
        if not token:
            break
    if not files:
        raise RuntimeError("The public GONG dataset returned no bzi_targets/*.npy files")
    return sorted(files, key=lambda item: item["name"])


def local_api() -> Any:
    global _API
    if _API is None:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        # The SDK prints the URL/license for every individual file even in quiet mode.
        api._print_dataset_url_and_license = lambda *args, **kwargs: None
        api.authenticate()
        _API = api
    return _API


def validate_target(path: Path) -> None:
    array = np.load(path, allow_pickle=False, mmap_mode="r")
    if array.shape != (TARGET_CHANNELS, TARGET_SIZE, TARGET_SIZE) or array.dtype != np.uint8:
        raise ValueError(f"Unexpected GONG target array at {path}: {array.shape} {array.dtype}")


def download_one(remote_name: str, data_root: Path) -> dict[str, Any]:
    destination = data_root / remote_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    downloaded = False
    if not destination.is_file():
        last_error: Exception | None = None
        for attempt in range(6):
            try:
                local_api().dataset_download_file(
                    DATASET_SLUG,
                    remote_name,
                    path=str(data_root),
                    quiet=True,
                    unzip=True,
                )
                downloaded = True
                time.sleep(2.0)
                break
            except Exception as error:  # Kaggle API may transiently fail between files.
                last_error = error
                if attempt < 5:
                    rate_limited = "429" in str(error)
                    delay = min(60 * (2**attempt), 300) if rate_limited else min(2**attempt, 30)
                    print(f"[retry] {remote_name} after {delay}s: {error}", flush=True)
                    time.sleep(delay)
        if not destination.is_file():
            raise RuntimeError(f"Failed to download {remote_name}: {last_error}")
    validate_target(destination)
    return {
        "name": remote_name,
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "downloaded_this_run": downloaded,
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--competition-data-root",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026",
    )
    parser.add_argument("--split-dir", type=Path, default=REPO_ROOT / "data" / "splits")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "data" / "reference" / "public-datasets" / "vhrabar-gong-solar-filaments",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Download only the first N eligible files for a smoke run")
    parser.add_argument("--manifest-only", action="store_true", help="Write the training-only plan without downloading")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--workers must be positive and --limit must be positive when supplied")

    workspace_data = (REPO_ROOT / "data").resolve()
    competition_root = args.competition_data_root.expanduser().resolve()
    split_dir = args.split_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not competition_root.is_relative_to(workspace_data):
        raise ValueError("Competition data must remain under this workspace's data/ folder")
    if not split_dir.is_relative_to((workspace_data / "splits").resolve()):
        raise ValueError("Split manifests must remain under this workspace's data/splits folder")
    if not output_root.is_relative_to((workspace_data / "reference").resolve()):
        raise ValueError("GONG files must remain under this workspace's data/reference folder")
    if args.workers != 1:
        raise ValueError("Use one Kaggle download worker; the dataset file endpoint throttles concurrent requests")

    dataset = load_dataset(competition_root)
    split = create_observation_split(dataset, split_dir, seed=42, validation_fraction=0.2)
    if split.sha256 != EXPECTED_SPLIT_SHA256:
        raise RuntimeError(f"Frozen split digest changed: {split.sha256}")
    training_names = set(split.training_names)
    validation_names = set(split.validation_names)
    if training_names & validation_names or (len(training_names), len(validation_names)) != (565, 142):
        raise RuntimeError("Frozen observation partitions failed their size or disjointness check")

    available = list_target_files(kaggle_cli_path())
    selected = [
        item
        for item in available
        if f"{Path(item['name']).stem}.jpeg" in training_names
    ]
    validation_matches = sum(
        f"{Path(item['name']).stem}.jpeg" in validation_names for item in available
    )
    if any(f"{Path(item['name']).stem}.jpeg" in validation_names for item in selected):
        raise AssertionError("A holdout observation entered the GONG download selection")
    if len(selected) != len({Path(item["name"]).stem for item in selected}):
        raise RuntimeError("GONG target listing contains duplicate observation files")
    if args.limit is not None:
        selected = selected[: args.limit]

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "train-only-manifest.json"
    manifest: dict[str, Any] = {
        "dataset": DATASET_SLUG,
        "dataset_url": DATASET_URL,
        "dataset_license_label": "other",
        "source_note": "Public GONG/NSO magnetogram-derived targets; dataset contains no MAGFiLO images, annotations, or pseudo-labels.",
        "required_acknowledgement": "Data were acquired by GONG instruments operated by NISP/NSO/AURA/NSF with contribution from NOAA.",
        "split_sha256": split.sha256,
        "partition": "frozen training observations only",
        "training_observations": len(training_names),
        "training_observation_names": sorted(training_names),
        "validation_observations": len(validation_names),
        "available_target_maps": len(available),
        "training_observations_with_targets": len(
            {Path(item["name"]).stem for item in available if f"{Path(item['name']).stem}.jpeg" in training_names}
        ),
        "validation_target_assets_seen_in_file_listing_only": validation_matches,
        "validation_target_files_downloaded": 0,
        "selected_target_maps": len(selected),
        "selected_target_bytes": sum(item["size"] for item in selected),
        "target_metadata": {
            "array_shape": [3, TARGET_SIZE, TARGET_SIZE],
            "dtype": "uint8",
            "channels": [
                "PIL proximity exp(-distance / 10 Mm), after 12 Mm smoothing",
                "strong-flux proxy tanh(abs(B) / 100 G)",
                "valid-disk mask",
            ],
            "crop_for_1888px_training_input": "indices 20:492 on each spatial axis",
        },
        "downloaded_files": [],
    }
    if args.manifest_only:
        atomic_json(manifest_path, manifest)
        print(f"[plan] {len(selected)} train-only targets; validation assets downloaded=0")
        print(f"[plan] wrote {manifest_path.relative_to(REPO_ROOT)}")
        return 0

    # Fail fast on missing Kaggle credentials before starting the concurrent batch.
    local_api()
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    for index, item in enumerate(selected, start=1):
        try:
            results.append(download_one(item["name"], output_root))
        except Exception as error:
            raise RuntimeError(f"GONG target download failed at {item['name']}: {error}") from error
        if index % 25 == 0 or index == len(selected):
            print(f"[download] {index}/{len(selected)} training-only targets", flush=True)

    results.sort(key=lambda item: item["name"])
    manifest["downloaded_files"] = results
    manifest["downloaded_target_count"] = len(results)
    manifest["downloaded_target_bytes"] = sum(item["bytes"] for item in results)
    manifest["downloaded_target_sha256"] = hashlib.sha256(
        "\n".join(f"{item['name']}:{item['sha256']}" for item in results).encode("utf-8")
    ).hexdigest()
    manifest["download_runtime_seconds"] = round(time.perf_counter() - started, 3)
    manifest["holdout_observation_names_in_downloaded_files"] = sorted(
        f"{Path(item['name']).stem}.jpeg" for item in results
        if f"{Path(item['name']).stem}.jpeg" in validation_names
    )
    if manifest["holdout_observation_names_in_downloaded_files"]:
        raise AssertionError("Holdout GONG targets were downloaded")
    atomic_json(manifest_path, manifest)
    print(
        f"[complete] {len(results)} train-only targets; "
        f"holdout target files downloaded=0; "
        f"runtime={manifest['download_runtime_seconds']:.1f}s",
        flush=True,
    )
    print(f"[complete] manifest={manifest_path.relative_to(REPO_ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
