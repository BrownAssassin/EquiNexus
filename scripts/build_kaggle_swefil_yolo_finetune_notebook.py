# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from build_kaggle_detector_notebook import _bundle, _cell, _source_files

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "swefil_yolo_finetune_screen.ipynb"
KERNEL_DIR = REPO_ROOT / "artifacts" / "kaggle" / "swefil-yolo-finetune-screen"
KERNEL_ID = "mrinanksivakumar/equinexus-yolo-swefil-fine-tuning-screen"
PRETRAIN_KERNEL = "mrinanksivakumar/equinexus-filament-swefil-yolo-pretrain"
PRETRAIN_RUN = "yolo11s-seg-swefil-30ep"
PRETRAINED_SHA256 = "6e1eedabd75a98783d4e840be7b98f399f0930197886bba560758e7d0dbb1bbd"
SWEFIL_MANIFEST_SHA256 = "0a8f6e7f313d78bfff4935645e919e9825c17233e8dfa880cfae3f69628d3d8d"
FROZEN_HOLDOUT_SHA256 = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"


def build(
    *,
    run_name: str = "yolo11s-seg-swefil-ft-25ep",
    image_size: int = 1024,
    epochs: int = 25,
    batch_size: int = 4,
    workers: int = 4,
    patience: int = 8,
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
    intro = f"""# EquiNexus SWEFil to competition YOLO11-seg fine-tuning screen

This run initializes from the verified SWEFil-pretrained checkpoint produced by `{PRETRAIN_KERNEL}`. It verifies checkpoint SHA-256 `{PRETRAINED_SHA256}`, SWEFil manifest SHA-256 `{SWEFIL_MANIFEST_SHA256}`, and the frozen observation holdout digest `{FROZEN_HOLDOUT_SHA256}` before training. The data source is SWEFil, a public CC-BY-4.0 dataset, filtered to exclude every competition training and test date. Its attribution and citation are preserved in the attached pretraining run provenance.

The detector is fine-tuned using only the competition-provided training partition, then evaluated on the fixed grouped holdout by the shared scorer. Only organizer-provided training annotations are used for this fine-tuning run; inference uses the holdout images in the attached competition data. Run configuration: {image_size}px, {epochs} epochs, batch size {batch_size}, patience {patience}, one Kaggle GPU. Outputs go under `/kaggle/working/equinexus/{run_name}`.
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
        "import hashlib, importlib.metadata, json, os, subprocess, sys\n"
        "from pathlib import Path\n"
        "os.environ['EQUINEXUS_SOURCE_COMMIT'] = SOURCE_COMMIT\n"
        "os.environ['EQUINEXUS_SOURCE_BUNDLE_SHA256'] = SOURCE_BUNDLE_SHA256\n"
        "os.environ['EQUINEXUS_EXTERNAL_SUPERVISED_DATA'] = 'true'\n"
        "os.environ['EQUINEXUS_TRAINING_DATA_DESCRIPTION'] = 'SWEFil CC-BY-4.0 supervised pretraining, followed by fine-tuning on competition-provided training annotations only'\n"
        "os.environ['EQUINEXUS_PRETRAINED_SOURCE'] = 'Ultralytics YOLO11-seg initialized from COCO weights and supervised-pretrained on audited SWEFil CC-BY-4.0 masks'\n"
        "os.environ['EQUINEXUS_PRETRAINED_LICENSE'] = 'AGPL-3.0 model/code; SWEFil CC-BY-4.0 attribution retained in external-data provenance'\n"
        "os.environ['YOLO_CONFIG_DIR'] = '/kaggle/working/equinexus/yolo-config'\n"
        "os.environ['TORCH_HOME'] = '/kaggle/working/equinexus/torch-home'\n"
        "Path(os.environ['YOLO_CONFIG_DIR']).mkdir(parents=True, exist_ok=True)\n"
        "Path(os.environ['TORCH_HOME']).mkdir(parents=True, exist_ok=True)\n"
        "subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '--no-cache-dir', '-r', str(REPOSITORY_PATH / 'requirements-detector.txt')], check=True)\n"
        "import torch, ultralytics\n"
        "devices = [{'name': torch.cuda.get_device_name(i), 'total_memory_bytes': torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]\n"
        "print({'python': sys.version.split()[0], 'torch': torch.__version__, 'cuda': torch.version.cuda, 'ultralytics': ultralytics.__version__, 'cuda_available': torch.cuda.is_available(), 'devices': devices})\n"
        "assert torch.cuda.is_available(), 'A Kaggle GPU is required for this run'\n"
        "INPUT_ROOT = Path('/kaggle/input')\n"
        f"RUN_SLUG = {PRETRAIN_RUN!r}\n"
        f"EXPECTED_CHECKPOINT_SHA256 = {PRETRAINED_SHA256!r}\n"
        f"EXPECTED_EXTERNAL_MANIFEST_SHA256 = {SWEFIL_MANIFEST_SHA256!r}\n"
        f"EXPECTED_HOLDOUT_SHA256 = {FROZEN_HOLDOUT_SHA256!r}\n"
        "checkpoint_candidates = [p for p in INPUT_ROOT.rglob('best.pt') if p.parent.name == RUN_SLUG]\n"
        "if len(checkpoint_candidates) != 1:\n"
        "    raise FileNotFoundError(f'Expected one attached SWEFil checkpoint, found: {checkpoint_candidates}')\n"
        "checkpoint_path = checkpoint_candidates[0]\n"
        "sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()\n"
        "assert sha == EXPECTED_CHECKPOINT_SHA256, f'Unexpected SWEFil checkpoint hash: {sha}'\n"
        "pretrain_root = checkpoint_path.parent\n"
        "pretrain_metadata = json.loads((pretrain_root / 'run-metadata.json').read_text())\n"
        "external_provenance = json.loads((pretrain_root / 'external-data-provenance.json').read_text())\n"
        "assert pretrain_metadata['status'] == 'complete'\n"
        "assert pretrain_metadata['frozen_holdout_used_for_training_or_selection'] is False\n"
        "assert pretrain_metadata['frozen_holdout_sha256'] == EXPECTED_HOLDOUT_SHA256\n"
        "assert external_provenance['included_manifest_sha256'] == EXPECTED_EXTERNAL_MANIFEST_SHA256\n"
        "assert external_provenance['oof_safe_date_exclusion'] is True\n"
        "print({'checkpoint': str(checkpoint_path), 'checkpoint_sha256': sha, 'external_manifest_sha256': external_provenance['included_manifest_sha256']})\n"
    )
    run_code = (
        "import shutil\n"
        f"run_root = Path('/kaggle/working/equinexus/{run_name}')\n"
        "run_root.mkdir(parents=True, exist_ok=True)\n"
        "shutil.copy2(pretrain_root / 'external-data-provenance.json', run_root / 'pretrained-external-data-provenance.json')\n"
        "shutil.copy2(pretrain_root / 'run-metadata.json', run_root / 'swefil-pretraining-run-metadata.json')\n"
        "command = [sys.executable, '-u', str(REPOSITORY_PATH / 'scripts' / 'run_yolo_detector_experiment.py'), "
        "'--repo-root', str(REPOSITORY_PATH), '--run-name', "
        f"{run_name!r}, '--model', str(checkpoint_path), '--image-size', {str(image_size)!r}, "
        f"'--epochs', {str(epochs)!r}, '--batch-size', {str(batch_size)!r}, "
        f"'--workers', {str(workers)!r}, '--patience', {str(patience)!r}, '--device', '0']\n"
        "print('Starting SWEFil warm-start fine-tuning:', ' '.join(command), flush=True)\n"
        "subprocess.run(command, check=True)\n"
    )
    report_code = (
        "import json\n"
        "metadata_path = run_root / 'run-metadata.json'\n"
        "print(json.dumps(json.loads(metadata_path.read_text()), indent=2))\n"
        "for name in ('best.pt', 'validation_detections.json', 'pretrained-external-data-provenance.json'):\n"
        "    path = run_root / name\n"
        "    print(name, path.stat().st_size if path.is_file() else 'missing')\n"
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
        "title": "EquiNexus YOLO SWEFil Fine Tuning Screen",
        "code_file": notebook_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "machine_shape": "",
        "dataset_sources": [],
        "competition_sources": ["filament-segmentation-2026"],
        "kernel_sources": [PRETRAIN_KERNEL],
        "model_sources": [],
    }
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return notebook_path, kernel_dir


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build the Kaggle SWEFil YOLO fine-tuning screen notebook.")
    parser.add_argument("--run-name", default="yolo11s-seg-swefil-ft-25ep")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()
    notebook, kernel_dir = build(**vars(args))
    print(json.dumps({"notebook": str(notebook), "kernel_dir": str(kernel_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
