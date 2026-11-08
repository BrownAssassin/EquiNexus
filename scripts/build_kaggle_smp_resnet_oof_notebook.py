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
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "smp_resnet34_grouped_oof.ipynb"
KERNEL_DIR = REPO_ROOT / "artifacts" / "kaggle" / "smp-resnet34-grouped-oof"
KERNEL_ID = "mrinanksivakumar/equinexus-smp-resnet34-grouped-oof"
RUN_NAME = "smp-resnet34-consensus512-3k-oof"


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
        *sorted((REPO_ROOT / "src" / "filament_baseline").glob("*.py")),
        REPO_ROOT / "scripts" / "run_training_partition_oof.py",
        REPO_ROOT / "configs" / "validation-baseline.json",
        REPO_ROOT / "requirements.txt",
        REPO_ROOT / "requirements-segmentation.txt",
    ]


def _bundle(paths: list[Path]) -> tuple[dict[str, str], str]:
    encoded: dict[str, str] = {}
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        relative = path.relative_to(REPO_ROOT).as_posix()
        content = path.read_bytes()
        encoded[relative] = base64.b64encode(content).decode("ascii")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return encoded, digest.hexdigest()


def build(target_mode: str = "consensus", input_mode: str = "raw") -> tuple[Path, Path]:
    if target_mode not in {"annotator", "consensus"}:
        raise ValueError("target_mode must be 'annotator' or 'consensus'")
    if input_mode not in {"raw", "raw_local_contrast", "raw_local_contrast_radial"}:
        raise ValueError("unsupported input_mode")
    run_name = (
        RUN_NAME
        if target_mode == "consensus" and input_mode == "raw"
        else f"smp-resnet34-{target_mode}-{input_mode.replace('_', '-')}-512-3k-oof"
    )
    encoded, source_bundle_sha256 = _bundle(_source_files())
    source_commit = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    markdown = f"""# EquiNexus pretrained ResNet34 U-Net grouped OOF experiment

Train five observation-grouped folds using an SMP U-Net with a public ImageNet-pretrained ResNet34 encoder. The target mode is `{target_mode}` and the image input mode is `{input_mode}`; each fold predicts observations excluded from its training. The seed-42 20% holdout is excluded from all folds and scoring. Training uses no external labels or competition test images.

Run configuration: 512-pixel crops, 3,000 steps per fold, batch size 8, 5 folds, and one Kaggle GPU. The ResNet34 weights are obtained through the public torchvision/SMP ImageNet initialization. The training pipeline and scorer are the repository's shared implementation; source bundle SHA-256 is recorded in the report. All caches and artifacts are written under `/kaggle/working/EquiNexus/`.
"""
    source_cell = (
        "import base64\n"
        "from pathlib import Path\n"
        "REPOSITORY_PATH = Path('/kaggle/working/EquiNexus')\n"
        f"SOURCE_COMMIT = {source_commit!r}\n"
        f"SOURCE_BUNDLE_SHA256 = {source_bundle_sha256!r}\n"
        f"SOURCE_FILES = {json.dumps(encoded, separators=(',', ':'))}\n"
        "for relative, payload in SOURCE_FILES.items():\n"
        "    destination = REPOSITORY_PATH / relative\n"
        "    destination.parent.mkdir(parents=True, exist_ok=True)\n"
        "    destination.write_bytes(base64.b64decode(payload))\n"
        "print({'source_commit': SOURCE_COMMIT, 'source_bundle_sha256': SOURCE_BUNDLE_SHA256, 'files': len(SOURCE_FILES)})\n"
    )
    runtime_cell = (
        "import os, subprocess, sys\n"
        "os.environ['EQUINEXUS_SOURCE_COMMIT'] = SOURCE_COMMIT\n"
        "os.environ['EQUINEXUS_SOURCE_BUNDLE_SHA256'] = SOURCE_BUNDLE_SHA256\n"
        "os.environ['PIP_CACHE_DIR'] = str(REPOSITORY_PATH / '.cache' / 'pip')\n"
        "os.environ['TMPDIR'] = str(REPOSITORY_PATH / '.tmp')\n"
        "os.environ['TORCH_HOME'] = str(REPOSITORY_PATH / 'artifacts' / 'pretrained' / 'torch')\n"
        "os.environ['HF_HOME'] = str(REPOSITORY_PATH / '.cache' / 'huggingface')\n"
        "os.environ['HF_HUB_DISABLE_XET'] = '1'\n"
        "for key in ('PIP_CACHE_DIR', 'TMPDIR', 'TORCH_HOME', 'HF_HOME'):\n"
        "    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)\n"
        "for requirements in ('requirements.txt', 'requirements-segmentation.txt'):\n"
        "    subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '-r', str(REPOSITORY_PATH / requirements)], check=True)\n"
        "import torch, segmentation_models_pytorch as smp, timm\n"
        "devices = [{'name': torch.cuda.get_device_name(i), 'total_memory_bytes': torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]\n"
        "print({'python': sys.version.split()[0], 'torch': torch.__version__, 'torchvision': __import__('torchvision').__version__, 'smp': smp.__version__, 'timm': timm.__version__, 'devices': devices})\n"
        "assert torch.cuda.is_available(), 'A Kaggle GPU is required for this run'\n"
    )
    run_cell = (
        "command = [sys.executable, '-u', str(REPOSITORY_PATH / 'scripts' / 'run_training_partition_oof.py'), "
        "'--repo-root', str(REPOSITORY_PATH), '--data-root', '/kaggle/input', '--run-name', "
        f"{run_name!r}, '--target-mode', {target_mode!r}, '--folds', '5', '--steps', '3000', '--batch-size', '8', "
        "'--patch-size', '512', '--base-channels', '16', '--model-type', 'smp_unet_resnet34', '--pretrained-backbone', "
        f"'--input-mode', {input_mode!r}, "
        "'--tta', '4', '--foreground-threshold', '0.65', '--boundary-threshold', '0.70', '--boundary-weight', '1.5', "
        "'--seed-threshold', '0.45', '--seed-min-area', '8', '--min-instance-area', '400']\n"
        "print('Starting five-fold SMP ResNet34 U-Net OOF training:', ' '.join(command), flush=True)\n"
        "subprocess.run(command, check=True)\n"
    )
    report_cell = (
        "import json\n"
        f"run_root = REPOSITORY_PATH / 'artifacts' / 'oof-experiments' / {run_name!r}\n"
        "metrics_path = run_root / 'metrics.json'\n"
        "print('metrics:', metrics_path, 'exists=', metrics_path.is_file())\n"
        "if metrics_path.is_file(): print(json.dumps(json.loads(metrics_path.read_text()), indent=2))\n"
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
            "title": f"EquiNexus SMP ResNet34 {target_mode} Grouped OOF",
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
    print(f"Run: {run_name} ({target_mode}; {input_mode})")
    print(f"Source bundle SHA-256: {source_bundle_sha256}")
    return NOTEBOOK_PATH, metadata_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the SMP ResNet34 five-fold Kaggle OOF notebook.")
    parser.add_argument("--target-mode", choices=("annotator", "consensus"), default="consensus")
    parser.add_argument(
        "--input-mode",
        choices=("raw", "raw_local_contrast", "raw_local_contrast_radial"),
        default="raw",
    )
    args = parser.parse_args()
    build(target_mode=args.target_mode, input_mode=args.input_mode)
