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
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "swefil_yolo_pretrain.ipynb"
KERNEL_DIR = REPO_ROOT / "artifacts" / "kaggle" / "swefil-yolo-pretrain"
KERNEL_ID = "mrinanksivakumar/equinexus-filament-swefil-yolo-pretrain"
SWEFIL_REVISION = "bc818c69a79266ad0fa91a539359485af57967cf"


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
    return [
        REPO_ROOT / "src" / "filament_baseline" / name
        for name in ("__init__.py", "data.py", "external_data.py", "runtime.py", "split.py")
    ] + [
        REPO_ROOT / "scripts" / "export_yolo_dataset.py",
        REPO_ROOT / "scripts" / "run_yolo_swefil_pretrain.py",
        REPO_ROOT / "configs" / "validation-baseline.json",
        REPO_ROOT / "requirements-detector.txt",
    ]


def _bundle(paths: list[Path]) -> tuple[dict[str, str], str]:
    encoded: dict[str, str] = {}
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Kaggle notebook source is missing: {path}")
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
    run_name: str = "yolo11s-seg-swefil-30ep",
    image_size: int = 1024,
    epochs: int = 30,
    batch_size: int = 4,
    workers: int = 4,
    patience: int = 8,
    notebook_path: str | Path = NOTEBOOK_PATH,
    kernel_dir: str | Path = KERNEL_DIR,
) -> tuple[Path, Path]:
    if not run_name or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for char in run_name
    ):
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
    intro = f"""# EquiNexus SWEFil YOLO warm-start

Pretrain YOLO11-seg on the separately released SWEFil CC-BY-4.0 filament masks, then use the resulting checkpoint to initialize later training on the competition's allowed training annotations. The attached private Kaggle input contains only the audited sample: all calendar dates present in either the competition train or test data, exact competition observation names, raw duplicates, sunspot labels, and filament class targets were removed before upload. This notebook rechecks the filtered manifest and exact image overlap before use. No MAGFiLO labels outside the competition-provided training set are read. Final test inference remains limited to the supplied H-alpha images.

SWEFil source revision: `{SWEFIL_REVISION}`. Run configuration: {image_size}px, {epochs} epochs, batch size {batch_size}, patience {patience}, one Kaggle GPU. Outputs are written under `/kaggle/working/equinexus/`; the best checkpoint is the reusable artifact.
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
        "print(f'Embedded {len(SOURCE_FILES)} source files from {SOURCE_COMMIT}')\n"
        "print(f'Embedded source SHA-256: {SOURCE_BUNDLE_SHA256}')\n"
    )
    setup_code = (
        "import os, subprocess, sys\n"
        "os.environ['EQUINEXUS_SOURCE_COMMIT'] = SOURCE_COMMIT\n"
        "os.environ['EQUINEXUS_SOURCE_BUNDLE_SHA256'] = SOURCE_BUNDLE_SHA256\n"
        "os.environ['YOLO_CONFIG_DIR'] = '/kaggle/working/equinexus/yolo-config'\n"
        "os.environ['TORCH_HOME'] = '/kaggle/working/equinexus/torch-home'\n"
        "for key in ('YOLO_CONFIG_DIR', 'TORCH_HOME'):\n"
        "    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)\n"
        "subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '--no-cache-dir', '-r', str(REPOSITORY_PATH / 'requirements-detector.txt')], check=True)\n"
        "import torch, ultralytics\n"
        "devices = [{'name': torch.cuda.get_device_name(i), 'total_memory_bytes': torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]\n"
        "print({'python': sys.version.split()[0], 'torch': torch.__version__, 'cuda': torch.version.cuda, 'ultralytics': ultralytics.__version__, 'cuda_available': torch.cuda.is_available(), 'devices': devices})\n"
        "assert torch.cuda.is_available(), 'A Kaggle GPU is required for this run'\n"
        "input_root = Path('/kaggle/input')\n"
        "revision_markers = sorted(input_root.rglob('equinexus_revision.txt'))\n"
        "if not revision_markers:\n"
        "    mounted_inputs = sorted(str(path) for path in input_root.iterdir()) if input_root.is_dir() else []\n"
        "    raise FileNotFoundError(f'No SWEFil revision marker found under {input_root}; mounted inputs: {mounted_inputs}')\n"
        "if len(revision_markers) != 1:\n"
        "    raise RuntimeError(f'Expected one attached SWEFil dataset, found revision markers: {revision_markers}')\n"
        "swefil_root = revision_markers[0].parent\n"
        "print('Mounted SWEFil dataset:', swefil_root)\n"
        f"assert (swefil_root / 'equinexus_revision.txt').read_text().strip() == {SWEFIL_REVISION!r}, 'Unexpected SWEFil revision'\n"
        "weights_dir = REPOSITORY_PATH / 'weights'\n"
        "weights_dir.mkdir(parents=True, exist_ok=True)\n"
        "from ultralytics import YOLO\n"
        "YOLO('yolo11s-seg.pt')\n"
        "import shutil\n"
        "downloaded = Path('yolo11s-seg.pt').resolve()\n"
        "if not downloaded.is_file():\n"
        "    downloaded = Path.home() / '.config' / 'Ultralytics' / 'yolo11s-seg.pt'\n"
        "assert downloaded.is_file(), f'Could not locate downloaded COCO checkpoint: {downloaded}'\n"
        "shutil.copy2(downloaded, weights_dir / 'yolo11s-seg.pt')\n"
        "print('Prepared SWEFil data and COCO initialization checkpoint')\n"
    )
    run_code = (
        "command = [sys.executable, '-u', str(REPOSITORY_PATH / 'scripts' / 'run_yolo_swefil_pretrain.py'), "
        "'--repo-root', str(REPOSITORY_PATH), '--data-root', '/kaggle/input', "
        "'--swefil-root', str(swefil_root), '--model', str(weights_dir / 'yolo11s-seg.pt'), "
        f"'--run-name', {run_name!r}, '--image-size', {str(image_size)!r}, "
        f"'--epochs', {str(epochs)!r}, '--batch-size', {str(batch_size)!r}, "
        f"'--workers', {str(workers)!r}, '--patience', {str(patience)!r}, '--device', '0']\n"
        "print('Starting external YOLO pretraining:', ' '.join(command), flush=True)\n"
        "subprocess.run(command, check=True)\n"
        "import json\n"
        f"metadata_path = REPOSITORY_PATH / 'artifacts' / 'detector-experiments' / {run_name!r} / 'run-metadata.json'\n"
        "print(json.dumps(json.loads(metadata_path.read_text()), indent=2))\n"
        "print('Best checkpoint:', metadata_path.parent / 'best.pt')\n"
    )
    notebook = {
        "cells": [
            _cell("markdown", intro),
            _cell("code", source_code),
            _cell("code", setup_code),
            _cell("code", run_code),
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
        "title": "EquiNexus Filament SWEFil YOLO Pretrain",
        "code_file": notebook_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "machine_shape": "",
        "dataset_sources": ["mrinanksivakumar/equinexus-swefil-date-disjoint"],
        "competition_sources": ["filament-segmentation-2026"],
        "kernel_sources": [],
        "model_sources": [],
    }
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return notebook_path, kernel_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a Kaggle notebook to pretrain YOLO11-seg on audited SWEFil masks."
    )
    parser.add_argument("--run-name", default="yolo11s-seg-swefil-30ep")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()
    notebook, kernel_dir = build(**vars(args))
    print(json.dumps({"notebook": str(notebook), "kernel_dir": str(kernel_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
