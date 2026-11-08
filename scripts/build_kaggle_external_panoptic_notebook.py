# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "external_panoptic_inference.ipynb"
KERNEL_DIR = REPO_ROOT / "artifacts" / "kaggle" / "external-panoptic-inference"
KERNEL_ID = "mrinanksivakumar/equinexus-external-panoptic-inference"
MODEL_SOURCE = "phuongncn/solar-filament-i079-weights/PyTorch/detector-refiner/1"


def _cell(kind: str, source: str) -> dict[str, Any]:
    result = {
        "id": hashlib.sha1(f"{kind}\0{source}".encode("utf-8")).hexdigest()[:8],
        "cell_type": kind,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if kind == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


def build() -> tuple[Path, Path]:
    source_paths = [
        REPO_ROOT / "src" / "filament_baseline" / "data.py",
        REPO_ROOT / "src" / "filament_baseline" / "runtime.py",
        REPO_ROOT / "src" / "filament_baseline" / "external_panoptic.py",
        REPO_ROOT / "scripts" / "run_external_panoptic_inference.py",
    ]
    source_bundle: dict[str, str] = {}
    digest = hashlib.sha256()
    for path in source_paths:
        relative = path.relative_to(REPO_ROOT).as_posix()
        contents = path.read_bytes()
        source_bundle[relative] = base64.b64encode(contents).decode("ascii")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(contents)
        digest.update(b"\0")
    bundle_sha256 = digest.hexdigest()
    source_commit = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    introduction = f"""# EquiNexus public detector/refiner inference

This run applies the public Apache-2.0 I079 detector/refiner checkpoints to the 180 supplied competition test images. The Kaggle model card says the weights were trained only on the official competition training images and annotations, and reports no test-overlap ground truth. Since the shared checkpoint was trained on all competition training observations, it is excluded from our grouped holdout validation; this is a test-inference candidate only.

EquiNexus independently implements the model architecture and inference adapter in the embedded source bundle. We do not copy the publisher's notebook source. The notebook does not read training annotations or test labels, and it does not submit automatically. It validates the CSV and writes it under `/kaggle/working/equinexus/public-i079/` for review.

Model card: <https://www.kaggle.com/models/phuongncn/solar-filament-i079-weights>

Published inference-method reference: <https://www.kaggle.com/code/robinamirbahar/solar-filament-panoptic-morphology-segmentation>

Source commit: `{source_commit}`
Embedded-source SHA-256: `{bundle_sha256}`
"""
    embed = (
        "import base64\n"
        "from pathlib import Path\n"
        "REPOSITORY_PATH = Path('/kaggle/working/EquiNexus')\n"
        f"SOURCE_COMMIT = {source_commit!r}\n"
        f"SOURCE_BUNDLE_SHA256 = {bundle_sha256!r}\n"
        f"SOURCE_FILES = {json.dumps(source_bundle, separators=(',', ':'))}\n"
        "for relative, payload in SOURCE_FILES.items():\n"
        "    destination = REPOSITORY_PATH / relative\n"
        "    destination.parent.mkdir(parents=True, exist_ok=True)\n"
        "    destination.write_bytes(base64.b64decode(payload))\n"
        "print('Source commit:', SOURCE_COMMIT)\n"
        "print('Source bundle SHA-256:', SOURCE_BUNDLE_SHA256)\n"
    )
    execute = (
        "import os, subprocess, sys\n"
        "os.environ['PYTHONUNBUFFERED'] = '1'\n"
        "os.environ['PYTHONPATH'] = str(REPOSITORY_PATH / 'src')\n"
        "command = [sys.executable, '-u', str(REPOSITORY_PATH / 'scripts' / 'run_external_panoptic_inference.py'), '--data-root', '/kaggle/input', '--model-root', '/kaggle/input', '--output-dir', '/kaggle/working/equinexus/public-i079']\n"
        "print('Running test-only inference:', ' '.join(command), flush=True)\n"
        "subprocess.run(command, check=True)\n"
    )
    report = (
        "from pathlib import Path\n"
        "run_root = Path('/kaggle/working/equinexus/public-i079')\n"
        "for name in ('run_report.json', 'submission.csv'):\n"
        "    path = run_root / name\n"
        "    print(name, 'exists=', path.is_file(), 'bytes=', path.stat().st_size if path.is_file() else 0)\n"
        "if (run_root / 'run_report.json').is_file():\n"
        "    print((run_root / 'run_report.json').read_text(encoding='utf-8'))\n"
    )
    notebook = {
        "cells": [
            _cell("markdown", introduction),
            _cell("code", embed),
            _cell("code", execute),
            _cell("code", report),
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
        "title": "EquiNexus External Panoptic Inference",
        "code_file": NOTEBOOK_PATH.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "false",
        "machine_shape": "",
        "dataset_sources": [],
        "competition_sources": ["filament-segmentation-2026"],
        "kernel_sources": [],
        "model_sources": [MODEL_SOURCE],
    }
    metadata_path = KERNEL_DIR / "kernel-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Notebook: {NOTEBOOK_PATH}")
    print(f"Kaggle bundle: {KERNEL_DIR}")
    print(f"Embedded-source SHA-256: {bundle_sha256}")
    return NOTEBOOK_PATH, metadata_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build an offline Kaggle test-inference notebook for the public I079 weights.")
    parser.parse_args()
    build()
