# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import base64
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "final_submission.ipynb"
KERNEL_DIR = REPO_ROOT / "artifacts" / "kaggle" / "final-submission"
KERNEL_ID = "mrinanksivakumar/equinexus-filament-final-ensemble"


def _cell(cell_type: str, source: str) -> dict[str, Any]:
    return {
        "id": hashlib.sha1(f"{cell_type}\0{source}".encode("utf-8")).hexdigest()[:8],
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True),
        **({"execution_count": None, "outputs": []} if cell_type == "code" else {}),
    }


def build(
    *,
    first_input_mode: str = "raw",
    second_input_mode: str = "raw",
    run_name: str = "final-ensemble-v1",
    foreground_threshold: float = 0.65,
    boundary_threshold: float = 0.70,
) -> tuple[Path, Path]:
    valid_input_modes = {"raw", "raw_local_contrast", "raw_local_contrast_radial"}
    if first_input_mode not in valid_input_modes or second_input_mode not in valid_input_modes:
        raise ValueError("Unsupported final ensemble input mode")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", run_name):
        raise ValueError("run_name must use only letters, digits, dot, underscore, or dash")
    if not 0.0 <= foreground_threshold <= 1.0 or not 0.0 <= boundary_threshold <= 1.0:
        raise ValueError("Postprocessing thresholds must be between zero and one")
    paths = [
        *sorted((REPO_ROOT / "src" / "filament_baseline").glob("*.py")),
        REPO_ROOT / "scripts" / "run_final_ensemble_submission.py",
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

    markdown = """# EquiNexus final ensemble submission

This notebook trains both selected U-Nets from scratch on all 707 official training observations and their competition-provided annotations. It predicts only the 180 attached competition H-alpha test images, blends the two probability maps, validates the 2048x2048 RLE output, and saves the submission under `/kaggle/working/equinexus/`.

No public MAGFiLO test labels, external ground-truth metadata, or pretrained weights are used. The fixed observation-grouped validation split selected the ensemble and postprocessing before final refitting. The local builder embeds a content-hashed source bundle, so this run does not depend on a GitHub push. The notebook creates the submission file but does not submit it; the validated file can be reviewed and submitted through the authenticated local Kaggle CLI.
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
        f"os.environ['EQUINEXUS_NOTEBOOK_VERSION'] = {run_name!r}\n"
        "os.environ['PYTHONUNBUFFERED'] = '1'\n"
        "probe = \"import json, torch; print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'available':torch.cuda.is_available(),'devices':[{'name':torch.cuda.get_device_name(i),'total_memory_bytes':torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]})); assert torch.cuda.is_available(), 'Kaggle GPU is required'\"\n"
        "subprocess.run([sys.executable, '-c', probe], check=True)\n"
    )
    run_cell = (
        "command = [sys.executable, '-u', str(REPOSITORY_PATH / 'scripts' / 'run_final_ensemble_submission.py'), '--repo-root', str(REPOSITORY_PATH), '--data-root', '/kaggle/input', '--run-name', "
        f"{run_name!r}, '--first-name', 'annotator1024', '--first-target-mode', 'annotator', '--first-steps', '10000', '--first-batch-size', '2', '--first-patch-size', '1024', '--first-tta', '8', '--first-input-mode', "
        f"{first_input_mode!r}, '--second-name', 'consensus512', '--second-target-mode', 'consensus', '--second-steps', '5000', '--second-batch-size', '8', '--second-patch-size', '512', '--second-tta', '4', '--second-input-mode', {second_input_mode!r}, '--first-weight', '0.5', '--foreground-threshold', '{foreground_threshold:.2f}', '--boundary-threshold', '{boundary_threshold:.2f}', '--boundary-weight', '1.5', '--seed-threshold', '0.45', '--seed-min-area', '8', '--min-instance-area', '400']\n"
        "print('Running final all-train ensemble:', ' '.join(command))\n"
        "subprocess.run(command, check=True)\n"
    )
    report_cell = (
        "from pathlib import Path\n"
        f"run_root = Path('/kaggle/working/equinexus/{run_name}')\n"
        "for name in ('metrics.json', 'test_coverage.csv', 'submission.csv'):\n"
        "    path = run_root / name\n"
        "    print(f'{name}: {path} exists={path.is_file()} bytes={path.stat().st_size if path.is_file() else 0}')\n"
        "if (run_root / 'metrics.json').is_file():\n"
        "    print((run_root / 'metrics.json').read_text(encoding='utf-8'))\n"
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
        "title": "EquiNexus Filament Final Ensemble",
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
        description="Build the Kaggle final-submission notebook with selected input modes."
    )
    parser.add_argument(
        "--first-input-mode", choices=("raw", "raw_local_contrast", "raw_local_contrast_radial"), default="raw"
    )
    parser.add_argument(
        "--second-input-mode", choices=("raw", "raw_local_contrast", "raw_local_contrast_radial"), default="raw"
    )
    parser.add_argument("--run-name", default="final-ensemble-v1")
    parser.add_argument("--foreground-threshold", type=float, default=0.65)
    parser.add_argument("--boundary-threshold", type=float, default=0.70)
    options = parser.parse_args()
    build(
        first_input_mode=options.first_input_mode,
        second_input_mode=options.second_input_mode,
        run_name=options.run_name,
        foreground_threshold=options.foreground_threshold,
        boundary_threshold=options.boundary_threshold,
    )
