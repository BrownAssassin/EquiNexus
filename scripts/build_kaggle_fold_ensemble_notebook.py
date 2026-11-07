from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "fold_ensemble.ipynb"
KERNEL_DIR = REPO_ROOT / "artifacts" / "kaggle" / "fold-ensemble"
KERNEL_ID = "mrinanksivakumar/equinexus-five-fold-u-net-ensemble"
CHECKPOINT_DATASET = "mrinanksivakumar/equinexus-perann-oof-checkpoints"


def _cell(cell_type: str, source: str) -> dict[str, Any]:
    return {
        "id": hashlib.sha1(f"{cell_type}\0{source}".encode("utf-8")).hexdigest()[:8],
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True),
        **({"execution_count": None, "outputs": []} if cell_type == "code" else {}),
    }


def build() -> tuple[Path, Path, str]:
    paths = [
        *sorted((REPO_ROOT / "src" / "filament_baseline").glob("*.py")),
        REPO_ROOT / "scripts" / "run_fold_ensemble.py",
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

    markdown = f"""# EquiNexus five-fold U-Net ensemble

This run loads five per-annotator U-Nets trained from scratch on grouped folds of the competition training partition. Every fold excluded all 142 observations in the frozen validation manifest. It averages their full-resolution probability maps, scores the frozen holdout with the official scorer, and predicts only the 180 attached competition H-alpha test images.

The checkpoint dataset is private and contains only model weights and a SHA-256 provenance manifest. No public MAGFiLO test labels, external labels, pretrained supervised weights, or test annotations are used. Checkpoint training used only the official competition training annotations. The ensemble and postprocessing remain candidates until the grouped holdout supports them.

Source commit: `{source_commit}`

Embedded source bundle SHA-256: `{bundle_sha256}`
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
        "os.environ['PYTHONUNBUFFERED'] = '1'\n"
        "probe = \"import json, torch; print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'available':torch.cuda.is_available(),'devices':[{'name':torch.cuda.get_device_name(i),'total_memory_bytes':torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]})); assert torch.cuda.is_available(), 'Kaggle GPU is required'\"\n"
        "subprocess.run([sys.executable, '-c', probe], check=True)\n"
    )
    run_cell = (
        "from pathlib import Path\n"
        "matches = list(Path('/kaggle/input').rglob('checkpoint-manifest.json'))\n"
        "if len(matches) != 1:\n"
        "    raise RuntimeError(f'Expected one private checkpoint manifest, found {matches}')\n"
        "CHECKPOINT_ROOT = matches[0].parent\n"
        "command = [sys.executable, '-u', str(REPOSITORY_PATH / 'scripts' / 'run_fold_ensemble.py'), '--repo-root', str(REPOSITORY_PATH), '--data-root', '/kaggle/input', '--checkpoint-root', str(CHECKPOINT_ROOT), '--run-name', 'oof-fivefold-mean-v1', '--foreground-threshold', '0.65', '--boundary-threshold', '0.70', '--boundary-weight', '1.5', '--seed-threshold', '0.45', '--seed-min-area', '8', '--min-instance-area', '400', '--solar-radius', '950']\n"
        "print('Running frozen-split five-fold inference:', ' '.join(command))\n"
        "subprocess.run(command, check=True)\n"
    )
    report_cell = (
        "from pathlib import Path\n"
        "run_root = Path('/kaggle/working/equinexus/oof-fivefold-mean-v1')\n"
        "for name in ('metrics.json', 'holdout-predictions.csv', 'holdout-per-observation.csv', 'submission.csv', 'test-coverage.csv'):\n"
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
        "title": "EquiNexus Five-Fold U-Net Ensemble",
        "code_file": NOTEBOOK_PATH.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [CHECKPOINT_DATASET],
        "competition_sources": ["filament-segmentation-2026"],
        "kernel_sources": [],
        "model_sources": [],
    }
    metadata_path = KERNEL_DIR / "kernel-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return NOTEBOOK_PATH, metadata_path, bundle_sha256


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the Kaggle five-fold inference notebook.")
    parser.parse_args()
    notebook_path, metadata_path, bundle_sha256 = build()
    print(f"Notebook: {notebook_path}")
    print(f"Kaggle bundle: {metadata_path.parent}")
    print(f"Source bundle SHA-256: {bundle_sha256}")
