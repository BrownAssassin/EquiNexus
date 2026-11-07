from __future__ import annotations

import base64
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "model_sprint.ipynb"
KERNEL_DIR = REPO_ROOT / "artifacts" / "kaggle" / "model-sprint"
KERNEL_ID = "mrinanksivakumar/equinexus-filament-model-sprint"


def _source_paths() -> list[Path]:
    paths = [
        *sorted((REPO_ROOT / "src" / "filament_baseline").glob("*.py")),
        REPO_ROOT / "scripts" / "run_model_experiment.py",
        REPO_ROOT / "scripts" / "run_final_submission.py",
        REPO_ROOT / "scripts" / "sweep_model_postprocessing.py",
        REPO_ROOT / "scripts" / "sweep_model_instance_structure.py",
        REPO_ROOT / "configs" / "validation-baseline.json",
        REPO_ROOT / "notebooks" / "validation_baseline.ipynb",
        REPO_ROOT / "requirements.txt",
        REPO_ROOT / "requirements-gpu.txt",
    ]
    return [path for path in paths if path.is_file()]


def _source_manifest(paths: list[Path]) -> tuple[dict[str, str], str]:
    encoded = {}
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


def _cell(cell_type: str, source: str, *, outputs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "id": hashlib.sha1(f"{cell_type}\0{source}".encode("utf-8")).hexdigest()[:8],
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if cell_type == "code":
        value["execution_count"] = None
        value["outputs"] = outputs or []
    return value


def build() -> tuple[Path, Path]:
    paths = _source_paths()
    encoded, bundle_sha256 = _source_manifest(paths)
    source_commit = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    markdown = """# EquiNexus model sprint\n\nThis GPU notebook runs the repository's from-scratch, per-annotator filament U-Net against the frozen observation-grouped split and organizer scorer. The model uses only competition-provided training annotations; test inference reads only the attached H-alpha test images. No external labels or pretrained weights are loaded.\n\nThe source files are embedded from the local checkout by `scripts/build_kaggle_model_notebook.py`, so the run does not depend on a GitHub push. This version validates a 1024-pixel patch model; final training is run only after selecting the validation-supported model and postprocessing.\n"""
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
    dependency_cell = (
        "import importlib.metadata\n"
        "import subprocess\n"
        "import sys\n"
        "requirements = REPOSITORY_PATH / 'requirements.txt'\n"
        "pins = []\n"
        "for line in requirements.read_text(encoding='utf-8').splitlines():\n"
        "    line = line.strip()\n"
        "    if line and not line.startswith('#') and not line.startswith('--'):\n"
        "        pins.append(tuple(line.split('==', 1)))\n"
        "mismatches = []\n"
        "for package, expected in pins:\n"
        "    try:\n"
        "        installed = importlib.metadata.version(package)\n"
        "    except importlib.metadata.PackageNotFoundError:\n"
        "        installed = None\n"
        "    if installed != expected:\n"
        "        mismatches.append((package, installed, expected))\n"
        "if mismatches:\n"
        "    print('Installing pinned non-Torch packages:', mismatches)\n"
        "    subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '--no-cache-dir', '-r', str(requirements)], check=True)\n"
        "print('Kaggle runtime inventory:')\n"
        "subprocess.run([sys.executable, '-c', \"import torch, numpy, cv2, skimage; print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'available', torch.cuda.is_available(), 'GPUs', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]); print('numpy', numpy.__version__, 'opencv', cv2.__version__, 'scikit-image', skimage.__version__)\"], check=True)\n"
        ""
    )
    hardware_cell = (
        "import os\n"
        "import subprocess\n"
        "import sys\n"
        "os.environ['EQUINEXUS_SOURCE_COMMIT'] = SOURCE_COMMIT\n"
        "os.environ['EQUINEXUS_SOURCE_BUNDLE_SHA256'] = SOURCE_BUNDLE_SHA256\n"
        "os.environ['EQUINEXUS_NOTEBOOK_VERSION'] = 'model-sprint-v2'\n"
        "os.environ['PYTHONUNBUFFERED'] = '1'\n"
        "probe = \"import json, torch; print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'available':torch.cuda.is_available(),'devices':[{'name':torch.cuda.get_device_name(i),'total_memory_bytes':torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]})); assert torch.cuda.is_available(), 'Kaggle GPU is required'\"\n"
        "subprocess.run([sys.executable, '-c', probe], check=True)\n"
    )
    model_cell = (
        "command = [sys.executable, '-u', str(REPOSITORY_PATH / 'scripts' / 'run_model_experiment.py'), '--repo-root', str(REPOSITORY_PATH), '--data-root', '/kaggle/input', '--run-name', 'unet-annotator-1024-kaggle-v2', '--target-mode', 'annotator', '--steps', '5000', '--batch-size', '2', '--patch-size', '1024', '--tta', '2', '--overlay-count', '12']\n"
        "print('Running:', ' '.join(command))\n"
        "subprocess.run(command, check=True)\n"
    )
    sweep_cell = (
        "sweep_command = [sys.executable, '-u', str(REPOSITORY_PATH / 'scripts' / 'sweep_model_postprocessing.py'), '--repo-root', str(REPOSITORY_PATH), '--data-root', '/kaggle/input', '--run-root', '/kaggle/working/equinexus/unet-annotator-1024-kaggle-v2']\n"
        "print('Sweeping validation postprocessing:', ' '.join(sweep_command))\n"
        "subprocess.run(sweep_command, check=True)\n"
        "import json\n"
        "from pathlib import Path\n"
        "best_config = json.loads(Path('/kaggle/working/equinexus/unet-annotator-1024-kaggle-v2/best-postprocessing/metrics.json').read_text(encoding='utf-8'))['configuration']\n"
        "structure_command = [sys.executable, '-u', str(REPOSITORY_PATH / 'scripts' / 'sweep_model_instance_structure.py'), '--repo-root', str(REPOSITORY_PATH), '--data-root', '/kaggle/input', '--run-root', '/kaggle/working/equinexus/unet-annotator-1024-kaggle-v2', '--foreground-threshold', str(best_config['foreground_threshold']), '--boundary-threshold', str(best_config['boundary_threshold']), '--boundary-weight', str(best_config['boundary_weight']), '--seed-threshold', str(best_config['seed_threshold'])]\n"
        "print('Sweeping instance-size and seed cutoffs:', ' '.join(structure_command))\n"
        "subprocess.run(structure_command, check=True)\n"
    )
    final_cell = (
        "# Enable only after reviewing the completed holdout sweeps and selecting this model.\n"
        "RUN_FINAL_SUBMISSION = False\n"
        "if RUN_FINAL_SUBMISSION:\n"
        "    selected = json.loads(Path('/kaggle/working/equinexus/unet-annotator-1024-kaggle-v2/best-instance-structure/metrics.json').read_text(encoding='utf-8'))['configuration']\n"
        "    final_command = [sys.executable, '-u', str(REPOSITORY_PATH / 'scripts' / 'run_final_submission.py'), '--repo-root', str(REPOSITORY_PATH), '--data-root', '/kaggle/input', '--run-name', 'final-annotator-1024-kaggle-v2', '--target-mode', 'annotator', '--steps', '5000', '--batch-size', '2', '--patch-size', '1024', '--tta', '2', '--foreground-threshold', str(selected['foreground_threshold']), '--boundary-threshold', str(selected['boundary_threshold']), '--boundary-weight', str(selected['boundary_weight']), '--seed-threshold', str(selected['seed_threshold']), '--seed-min-area', str(selected['seed_min_area']), '--min-instance-area', str(selected['min_instance_area'])]\n"
        "    print('Training on all official observations and writing the validated submission:', ' '.join(final_command))\n"
        "    subprocess.run(final_command, check=True)\n"
        "else:\n"
        "    print('Final training is deferred until the validation model and postprocessing are selected.')\n"
    )
    report_cell = (
        "from pathlib import Path\n"
        "for run in ('unet-annotator-1024-kaggle-v2', 'final-annotator-1024-kaggle-v2'):\n"
        "    metrics_path = Path('/kaggle/working/equinexus') / run / 'metrics.json'\n"
        "    if metrics_path.is_file():\n"
        "        print(f'\\n{run}:\\n{metrics_path.read_text(encoding=\"utf-8\")}')\n"
        "submission_path = Path('/kaggle/working/equinexus/final-annotator-1024-kaggle-v2/submission.csv')\n"
        "print('Submission path:', submission_path if submission_path.is_file() else 'not generated by this validation-only version')\n"
    )
    notebook = {
        "cells": [
            _cell("markdown", markdown),
            _cell("code", source_cell),
            _cell("code", dependency_cell),
            _cell("code", hardware_cell),
            _cell("code", model_cell),
            _cell("code", sweep_cell),
            _cell("code", final_cell),
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
    kernel_notebook = KERNEL_DIR / NOTEBOOK_PATH.name
    kernel_notebook.write_bytes(NOTEBOOK_PATH.read_bytes())
    metadata = {
        "id": KERNEL_ID,
        "title": "EquiNexus Filament Model Sprint",
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
    print(f"Embedded-source SHA-256: {bundle_sha256}")
    return NOTEBOOK_PATH, metadata_path


if __name__ == "__main__":
    build()
