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
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "detector_sprint.ipynb"
KERNEL_DIR = REPO_ROOT / "artifacts" / "kaggle" / "detector-sprint"
KERNEL_ID = "mrinanksivakumar/equinexus-filament-detector-sprint"


def _cell(cell_type: str, source: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": hashlib.sha1(f"{cell_type}\0{source}".encode("utf-8")).hexdigest()[:8],
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if cell_type == "code":
        result.update(execution_count=None, outputs=[])
    return result


def _source_files() -> list[Path]:
    paths = [
        *sorted((REPO_ROOT / "src" / "filament_baseline").glob("*.py")),
        REPO_ROOT / "scripts" / "export_yolo_dataset.py",
        REPO_ROOT / "scripts" / "run_yolo_detector_experiment.py",
        REPO_ROOT / "configs" / "validation-baseline.json",
        REPO_ROOT / "requirements-detector.txt",
    ]
    return [path for path in paths if path.is_file()]


def _bundle(paths: list[Path]) -> tuple[dict[str, str], str]:
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
    return encoded, digest.hexdigest()


def build(
    *,
    run_name: str = "yolo11s-seg-1024-40ep-coco",
    model: str = "yolo11s-seg.pt",
    image_size: int = 1024,
    epochs: int = 40,
    batch_size: int = 4,
    workers: int = 4,
    patience: int = 12,
    notebook_path: str | Path = NOTEBOOK_PATH,
    kernel_dir: str | Path = KERNEL_DIR,
) -> tuple[Path, Path]:
    if not run_name or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in run_name):
        raise ValueError("run_name must use only letters, digits, dot, underscore, or dash")
    if image_size < 256 or epochs < 1 or batch_size < 1 or workers < 0 or patience < 1:
        raise ValueError("image size, epochs, batch size, and patience must be positive")

    notebook_path = Path(notebook_path)
    if not notebook_path.is_absolute():
        notebook_path = REPO_ROOT / notebook_path
    kernel_dir = Path(kernel_dir)
    if not kernel_dir.is_absolute():
        kernel_dir = REPO_ROOT / kernel_dir
    if not notebook_path.resolve().is_relative_to(REPO_ROOT.resolve()):
        raise ValueError("Generated notebook must remain inside the repository")
    if not kernel_dir.resolve().is_relative_to(REPO_ROOT.resolve()):
        raise ValueError("Kaggle kernel files must remain inside the repository")

    encoded, bundle_sha256 = _bundle(_source_files())
    source_commit = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    intro = f"""# EquiNexus YOLO detector sprint

This notebook trains `{model}` using the frozen observation-grouped training partition and the competition-provided training annotations. It uses public COCO-pretrained Ultralytics YOLO11-seg weights (AGPL-3.0); no public MAGFiLO labels or competition test labels are read. The model is evaluated on each frozen validation observation exactly once, and its instance masks are saved for scoring and detector-guided fusion against the existing U-Net probability cache.

Run configuration: {image_size}px input, {epochs} epochs, batch size {batch_size}, patience {patience}, one Kaggle GPU. Source commit and embedded-source digest are recorded below. Training outputs are written under `/kaggle/working/equinexus/{run_name}`.
"""
    source_code = (
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
        "print(f'Embedded source SHA-256: {SOURCE_BUNDLE_SHA256}')\n"
    )
    runtime_code = (
        "import importlib.metadata\n"
        "import os, subprocess, sys\n"
        "os.environ['EQUINEXUS_SOURCE_COMMIT'] = SOURCE_COMMIT\n"
        "os.environ['EQUINEXUS_SOURCE_BUNDLE_SHA256'] = SOURCE_BUNDLE_SHA256\n"
        "os.environ['YOLO_CONFIG_DIR'] = '/kaggle/working/equinexus/yolo-config'\n"
        "os.environ['TORCH_HOME'] = '/kaggle/working/equinexus/torch-home'\n"
        "Path(os.environ['YOLO_CONFIG_DIR']).mkdir(parents=True, exist_ok=True)\n"
        "Path(os.environ['TORCH_HOME']).mkdir(parents=True, exist_ok=True)\n"
        "subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '--no-cache-dir', '-r', str(REPOSITORY_PATH / 'requirements-detector.txt')], check=True)\n"
        "import torch, ultralytics\n"
        "devices = [{'name': torch.cuda.get_device_name(i), 'total_memory_bytes': torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]\n"
        "print({'python': sys.version.split()[0], 'torch': torch.__version__, 'cuda': torch.version.cuda, 'ultralytics': ultralytics.__version__, 'cuda_available': torch.cuda.is_available(), 'devices': devices})\n"
        "assert torch.cuda.is_available(), 'A Kaggle GPU is required for this run'\n"
    )
    run_code = (
        "command = [sys.executable, '-u', str(REPOSITORY_PATH / 'scripts' / 'run_yolo_detector_experiment.py'), "
        "'--repo-root', str(REPOSITORY_PATH), '--run-name', "
        f"{run_name!r}, '--model', {model!r}, '--image-size', {str(image_size)!r}, "
        f"'--epochs', {str(epochs)!r}, '--batch-size', {str(batch_size)!r}, "
        f"'--workers', {str(workers)!r}, '--patience', {str(patience)!r}, '--device', '0']\n"
        "print('Starting detector training:', ' '.join(command), flush=True)\n"
        "subprocess.run(command, check=True)\n"
    )
    report_code = (
        "import json\n"
        f"run_root = Path('/kaggle/working/equinexus/{run_name}')\n"
        "metadata_path = run_root / 'run-metadata.json'\n"
        "print('run metadata:', metadata_path, 'exists=', metadata_path.is_file())\n"
        "if metadata_path.is_file():\n"
        "    print(json.dumps(json.loads(metadata_path.read_text()), indent=2))\n"
        "print('validation detections:', (run_root / 'validation_detections.json').stat().st_size if (run_root / 'validation_detections.json').is_file() else 'pending')\n"
    )
    notebook = {
        "cells": [
            _cell("markdown", intro),
            _cell("code", source_code),
            _cell("code", runtime_code),
            _cell("code", run_code),
            _cell("code", report_code),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / notebook_path.name).write_bytes(notebook_path.read_bytes())
    metadata = {
        "id": KERNEL_ID,
        "title": "EquiNexus Filament Detector Sprint",
        "code_file": notebook_path.name,
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
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return notebook_path, kernel_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Kaggle YOLO detector experiment notebook.")
    parser.add_argument("--run-name", default="yolo11s-seg-1024-40ep-coco")
    parser.add_argument("--model", default="yolo11s-seg.pt")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=12)
    args = parser.parse_args()
    notebook, kernel_dir = build(**vars(args))
    print(json.dumps({"notebook": str(notebook), "kernel_dir": str(kernel_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
