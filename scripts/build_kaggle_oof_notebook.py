# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import base64
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "oof_consensus.ipynb"
KERNEL_DIR = REPO_ROOT / "artifacts" / "kaggle" / "oof-consensus"
KERNEL_ID = "mrinanksivakumar/equinexus-grouped-oof-consensus-u-net"


def _cell(cell_type: str, source: str) -> dict[str, Any]:
    return {
        "id": hashlib.sha1(f"{cell_type}\0{source}".encode("utf-8")).hexdigest()[:8],
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True),
        **({"execution_count": None, "outputs": []} if cell_type == "code" else {}),
    }


def build(*, input_mode: str = "raw") -> tuple[Path, Path]:
    if input_mode not in ("raw", "raw_local_contrast", "raw_local_contrast_radial"):
        raise ValueError(f"Unsupported OOF input mode: {input_mode}")
    paths = [
        *sorted((REPO_ROOT / "src" / "filament_baseline").glob("*.py")),
        REPO_ROOT / "scripts" / "run_training_partition_oof.py",
        REPO_ROOT / "configs" / "validation-baseline.json",
        REPO_ROOT / "requirements.txt",
    ]
    paths = [path for path in paths if path.is_file()]
    encoded: dict[str, str] = {}
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(REPO_ROOT).as_posix()
        content = path.read_bytes()
        encoded[relative] = base64.b64encode(content).decode("ascii")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    bundle_sha256 = digest.hexdigest()
    source_commit = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    markdown = """# EquiNexus grouped OOF: consensus U-Net

This notebook evaluates a from-scratch, 512-pixel consensus U-Net on five observation-grouped folds drawn only from the competition training partition. It uses the frozen seed-42/year-stratified holdout's 565 training observations, so none of the 142 holdout observations enter OOF fitting or scoring. All views of an observation stay together. The fixed postprocessing was selected on the holdout before this OOF run.

The source bundle is embedded from the local checkout. Outputs include fold checkpoints, full-resolution probability caches, fold and aggregate scorer metrics, and the exact OOF manifest under `/kaggle/working/EquiNexus/artifacts/oof-experiments/`.
"""
    source_cell = (
        "import base64\n"
        "from pathlib import Path\n"
        "REPOSITORY_PATH = Path('/kaggle/working/EquiNexus')\n"
        f"SOURCE_COMMIT = {source_commit!r}\n"
        f"SOURCE_BUNDLE_SHA256 = {bundle_sha256!r}\n"
        f"SOURCE_FILES = {json.dumps(encoded, separators=(',', ':'))}\n"
        "for relative, payload in SOURCE_FILES.items():\n"
        "    destination = REPOSITORY_PATH / relative\n"
        "    destination.parent.mkdir(parents=True, exist_ok=True)\n"
        "    destination.write_bytes(base64.b64decode(payload))\n"
        "print(f'Embedded {len(SOURCE_FILES)} source files from base commit {SOURCE_COMMIT}')\n"
        "print(f'Source bundle SHA-256: {SOURCE_BUNDLE_SHA256}')\n"
    )
    runtime_cell = (
        "import importlib.metadata\n"
        "import os\n"
        "import subprocess\n"
        "import sys\n"
        "requirements = REPOSITORY_PATH / 'requirements.txt'\n"
        "pins = [tuple(line.strip().split('==', 1)) for line in requirements.read_text(encoding='utf-8').splitlines() if line.strip() and not line.strip().startswith('#')]\n"
        "mismatches = []\n"
        "for package, expected in pins:\n"
        "    try:\n"
        "        installed = importlib.metadata.version(package)\n"
        "    except importlib.metadata.PackageNotFoundError:\n"
        "        installed = None\n"
        "    if installed != expected:\n"
        "        mismatches.append((package, installed, expected))\n"
        "if mismatches:\n"
        "    print('Installing pinned baseline packages:', mismatches)\n"
        "    subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '--no-cache-dir', '-r', str(requirements)], check=True)\n"
        "os.environ['EQUINEXUS_SOURCE_COMMIT'] = SOURCE_COMMIT\n"
        "os.environ['EQUINEXUS_SOURCE_BUNDLE_SHA256'] = SOURCE_BUNDLE_SHA256\n"
        "os.environ['EQUINEXUS_NOTEBOOK_VERSION'] = 'oof-consensus-v1'\n"
        "os.environ['PYTHONUNBUFFERED'] = '1'\n"
        "probe = \"import json, torch; print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'available':torch.cuda.is_available(),'devices':[{'name':torch.cuda.get_device_name(i),'total_memory_bytes':torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]})); assert torch.cuda.is_available(), 'Kaggle GPU is required'\"\n"
        "subprocess.run([sys.executable, '-c', probe], check=True)\n"
    )
    run_cell = (
        "command = [sys.executable, '-u', str(REPOSITORY_PATH / 'scripts' / 'run_training_partition_oof.py'), '--repo-root', str(REPOSITORY_PATH), '--data-root', '/kaggle/input', '--run-name', 'oof-consensus512-5k-tta4', '--target-mode', 'consensus', '--folds', '5', '--steps', '5000', '--batch-size', '8', '--patch-size', '512', '--tta', '4', '--input-mode', "
        f"{input_mode!r}, '--foreground-threshold', '0.65', '--boundary-threshold', '0.70', '--boundary-weight', '1.5', '--seed-threshold', '0.45', '--seed-min-area', '8', '--min-instance-area', '400']\n"
        "print('Running grouped consensus OOF:', ' '.join(command))\n"
        "subprocess.run(command, check=True)\n"
    )
    report_cell = (
        "from pathlib import Path\n"
        "run_root = REPOSITORY_PATH / 'artifacts' / 'oof-experiments' / 'oof-consensus512-5k-tta4'\n"
        "for name in ('metrics.json', 'oof-predictions.csv', 'oof-per-observation.csv'):\n"
        "    path = run_root / name\n"
        "    print(f'{name}: {path} exists={path.is_file()} bytes={path.stat().st_size if path.is_file() else 0}')\n"
        "metrics = run_root / 'metrics.json'\n"
        "if metrics.is_file(): print(metrics.read_text(encoding='utf-8'))\n"
    )
    notebook = {
        "cells": [
            _cell("markdown", markdown),
            _cell("code", source_cell),
            _cell("code", runtime_cell),
            _cell("code", run_cell),
            _cell("code", report_cell),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    (KERNEL_DIR / NOTEBOOK_PATH.name).write_bytes(NOTEBOOK_PATH.read_bytes())
    metadata = {
        "id": KERNEL_ID,
        "title": "EquiNexus Grouped OOF Consensus U-Net",
        "code_file": NOTEBOOK_PATH.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "machine_shape": "",
        "dataset_sources": [],
        "competition_sources": ["filament-segmentation-2026"],
        "kernel_sources": [],
        "model_sources": [],
    }
    metadata_path = KERNEL_DIR / "kernel-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Notebook: {NOTEBOOK_PATH}")
    print(f"Kaggle bundle: {KERNEL_DIR}")
    print(f"Source bundle SHA-256: {bundle_sha256}")
    return NOTEBOOK_PATH, metadata_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the Kaggle consensus OOF notebook for a selected input mode."
    )
    parser.add_argument(
        "--input-mode", choices=("raw", "raw_local_contrast", "raw_local_contrast_radial"), default="raw"
    )
    options = parser.parse_args()
    build(input_mode=options.input_mode)
