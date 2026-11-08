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
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "model_sprint.ipynb"
KERNEL_DIR = REPO_ROOT / "artifacts" / "kaggle" / "model-sprint"
KERNEL_ID = "mrinanksivakumar/equinexus-filament-model-sprint"
INPUT_MODES = ("raw", "raw_local_contrast", "raw_local_contrast_radial")
MODEL_TYPES = ("unet", "dinov2_vits14")
DINO_V2_REPOSITORY = (
    "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8"
)
GONG_CONTENT_MANIFEST_SHA256 = (
    "e14073f882dccb945fda142371ef3a12119f27dc9c104fbbc38f98baddb653ab"
)
def _source_paths() -> list[Path]:
    package_paths = sorted((REPO_ROOT / "src" / "filament_baseline").glob("*.py"))
    paths = [
        *package_paths,
        REPO_ROOT / "scripts" / "run_model_experiment.py",
        REPO_ROOT / "scripts" / "run_final_submission.py",
        REPO_ROOT / "scripts" / "run_swefil_model_experiment.py",
        REPO_ROOT / "scripts" / "audit_swefil_external.py",
        REPO_ROOT / "scripts" / "download_gong_halpha.py",
        REPO_ROOT / "scripts" / "adapt_dinov2_solar.py",
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


def build(
    *,
    input_mode: str = "raw",
    target_mode: str = "annotator",
    run_name: str = "unet-annotator-1024-kaggle-v2",
    steps: int = 5_000,
    batch_size: int = 2,
    patch_size: int = 1_024,
    base_channels: int = 16,
    model_type: str = "unet",
    pretrained_backbone: bool = False,
    hub_repository: str = DINO_V2_REPOSITORY,
    solar_adaptation: bool = False,
    external_swefil: bool = False,
    external_pretrain_steps: int = 1_000,
    notebook_version: str = "model-sprint-v3",
    tta: int = 2,
    notebook_path: str | Path | None = None,
    kernel_dir: str | Path | None = None,
    kernel_id: str | None = None,
    kernel_title: str | None = None,
) -> tuple[Path, Path]:
    if input_mode not in INPUT_MODES:
        raise ValueError(f"Unsupported model input mode: {input_mode}")
    if target_mode not in ("annotator", "consensus"):
        raise ValueError("target_mode must be annotator or consensus")
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unsupported model type: {model_type}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", run_name):
        raise ValueError("run_name must use only letters, digits, dot, underscore, or dash")
    if steps < 1 or batch_size < 1 or patch_size < 128 or patch_size % 32 or base_channels < 1:
        raise ValueError("steps, batch size, patch size, and model width must be positive and valid")
    if model_type == "dinov2_vits14" and patch_size % 224:
        raise ValueError("DINOv2 patches must be divisible by 224")
    if pretrained_backbone and model_type != "dinov2_vits14":
        raise ValueError("pretrained_backbone is only supported for dinov2_vits14")
    if solar_adaptation and (model_type != "dinov2_vits14" or not pretrained_backbone):
        raise ValueError("solar_adaptation requires the pretrained dinov2_vits14 model")
    if tta not in (1, 2, 4, 8):
        raise ValueError("tta must be 1, 2, 4, or 8")
    notebook_path = Path(notebook_path) if notebook_path else NOTEBOOK_PATH
    if not notebook_path.is_absolute():
        notebook_path = REPO_ROOT / notebook_path
    kernel_dir = Path(kernel_dir) if kernel_dir else KERNEL_DIR
    if not kernel_dir.is_absolute():
        kernel_dir = REPO_ROOT / kernel_dir
    kernel_id = kernel_id or KERNEL_ID
    kernel_title = kernel_title or "EquiNexus Filament Model Sprint"
    paths = _source_paths()
    encoded, bundle_sha256 = _source_manifest(paths)
    source_commit = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    training_scope = (
        "Training uses the competition annotations plus the separately licensed SWEFil external filament masks. "
        if external_swefil
        else "Training uses only competition-provided annotations. "
    )
    markdown = (
        "# EquiNexus model sprint\n\n"
        f"This GPU notebook runs the repository's {model_type}, {target_mode} filament model "
        "against the frozen observation-grouped split and organizer scorer. "
        + training_scope
        + "Test inference reads only the attached H-alpha test images. External data and "
        "pretraining are pinned and recorded in run metadata.\n\n"
        "The source files are embedded from the local checkout by "
        "`scripts/build_kaggle_model_notebook.py`, so the run does not depend on a GitHub "
        f"push. This run uses `{input_mode}` inputs, {patch_size}-pixel patches, "
        f"{steps} training steps, base width {base_channels}, pretrained backbone "
        f"{pretrained_backbone}, and {tta}-view TTA. "
        "Final training is deferred until "
        "selecting the validation-supported model and postprocessing.\n"
        + (
            "Before supervised fitting, this notebook downloads the timestamp-audited, "
            "unlabeled NSO GONG H-alpha sample (image credit: NSO/AURA/NSF) and adapts "
            "the Apache-2.0 DINOv2 ViT-S/14 model by self-distillation. "
            "GONG data acknowledgment: 'Data were acquired by GONG instruments operated "
            "by NISP/NSO/AURA/NSF with contribution from NOAA.' This adaptation stage is "
            "label-free and no competition test labels are used.\n"
            if solar_adaptation
            else ""
        )
        + (
            "Before supervised fine-tuning, this notebook downloads the separately licensed "
            "SWEFil H-alpha filament dataset (CC-BY-4.0), pinned to revision "
            "`bc818c69a79266ad0fa91a539359485af57967cf`. It audits and excludes competition "
            "image overlaps and all frozen-validation/test dates, and uses only filament "
            "instance masks for external pretraining.\n"
            if external_swefil
            else ""
        )
    )
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
    gpu_smoke_code = (
        "import json, sys, torch\n"
        "sys.path.insert(0, '/kaggle/working/EquiNexus/src')\n"
        "torch.hub.set_dir('/kaggle/working/equinexus/pretrained/torch/hub')\n"
        "from filament_baseline.features import input_channel_count\n"
        "from filament_baseline.model import DINO_V2_REPOSITORY, build_filament_model\n"
        f"base_channels = {base_channels}\n"
        f"channels = input_channel_count({input_mode!r})\n"
        f"model = build_filament_model(model_type={model_type!r}, in_channels=channels, base_channels=base_channels, pretrained_backbone={pretrained_backbone!r}, hub_repository={hub_repository!r}).cuda().to(memory_format=torch.channels_last)\n"
        f"batch = torch.randn(({batch_size}, channels, {patch_size}, {patch_size}), device='cuda').contiguous(memory_format=torch.channels_last)\n"
        "torch.cuda.reset_peak_memory_stats()\n"
        "with torch.autocast(device_type='cuda', dtype=torch.float16):\n"
        "    logits = model(batch)\n"
        "    loss = logits.float().square().mean()\n"
        "assert torch.isfinite(loss)\n"
        "loss.backward()\n"
        "torch.cuda.synchronize()\n"
        "print(json.dumps({'smoke_loss': float(loss.detach()), 'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated()}))\n"
    )
    hardware_cell = (
        "import os\n"
        "import subprocess\n"
        "import sys\n"
        "os.environ['EQUINEXUS_SOURCE_COMMIT'] = SOURCE_COMMIT\n"
        "os.environ['EQUINEXUS_SOURCE_BUNDLE_SHA256'] = SOURCE_BUNDLE_SHA256\n"
        f"os.environ['EQUINEXUS_NOTEBOOK_VERSION'] = {notebook_version!r}\n"
        "os.environ['PYTHONUNBUFFERED'] = '1'\n"
        "inventory_probe = \"import json, torch; print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'available':torch.cuda.is_available(),'devices':[{'name':torch.cuda.get_device_name(i),'total_memory_bytes':torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]})); assert torch.cuda.is_available(), 'Kaggle GPU is required'\"\n"
        "subprocess.run([sys.executable, '-c', inventory_probe], check=True)\n"
        f"gpu_smoke_probe = {gpu_smoke_code!r}\n"
        "subprocess.run([sys.executable, '-c', gpu_smoke_probe], check=True)\n"
    )
    solar_adaptation_cell = (
        "external_data_root = REPOSITORY_PATH / 'data' / 'external' / 'gong_halpha_2026_11'\n"
        "download_command = [sys.executable, '-u', str(REPOSITORY_PATH / 'scripts' / 'download_gong_halpha.py'), '--start-date', '2026-11-01', '--end-date', '2026-11-06', '--cadence-minutes', '10', '--output-dir', str(external_data_root), '--competition-data-root', '/kaggle/input', '--expected-content-manifest-sha256', "
        f"{GONG_CONTENT_MANIFEST_SHA256!r}]\n"
        "print('Downloading unlabeled public NSO GONG H-alpha images:', ' '.join(download_command))\n"
        "subprocess.run(download_command, check=True)\n"
        "adapted_backbone_path = Path('/kaggle/working/equinexus/pretrained/dinov2_vits14_gong_adapted.pth')\n"
        "adaptation_command = [sys.executable, '-u', str(REPOSITORY_PATH / 'scripts' / 'adapt_dinov2_solar.py'), '--data-root', str(external_data_root), '--output-path', str(adapted_backbone_path), '--hub-dir', '/kaggle/working/equinexus/pretrained/torch/hub', '--hub-repository', "
        f"{hub_repository!r}, '--steps', '1000', '--batch-size', '16', '--image-size', '224', '--num-workers', '2', '--device', 'cuda']\n"
        "print('Running label-free DINOv2 adaptation:', ' '.join(adaptation_command))\n"
        "subprocess.run(adaptation_command, check=True)\n"
        "print('Adaptation metadata:', adapted_backbone_path.with_suffix('.json').read_text(encoding='utf-8'))\n"
    )
    model_cell = (
        "command = [sys.executable, '-u', str(REPOSITORY_PATH / 'scripts' / 'run_model_experiment.py'), '--repo-root', str(REPOSITORY_PATH), '--data-root', '/kaggle/input', '--run-name', "
        f"{run_name!r}, '--target-mode', {target_mode!r}, '--steps', {str(steps)!r}, '--batch-size', {str(batch_size)!r}, '--patch-size', {str(patch_size)!r}, '--base-channels', {str(base_channels)!r}, '--model-type', {model_type!r}, '--hub-repository', {hub_repository!r}, '--tta', {str(tta)!r}, '--input-mode', {input_mode!r}, '--overlay-count', '12']\n"
        f"command += ['--pretrained-backbone'] if {pretrained_backbone!r} else []\n"
        "print('Running:', ' '.join(command))\n"
        "subprocess.run(command, check=True)\n"
    )
    swefil_cell = (
        "import importlib.metadata\n"
        "import os\n"
        "import subprocess\n"
        "import sys\n"
        "try:\n"
        "    from huggingface_hub import snapshot_download\n"
        "except ImportError:\n"
        "    subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', 'huggingface_hub==0.35.3'], check=True)\n"
        "    from huggingface_hub import snapshot_download\n"
        "swefil_root = REPOSITORY_PATH / 'data' / 'external' / 'SWEFil'\n"
        "swefil_root.mkdir(parents=True, exist_ok=True)\n"
        "snapshot_download(repo_id='antonio-reche/SWEFil', repo_type='dataset', revision='bc818c69a79266ad0fa91a539359485af57967cf', local_dir=str(swefil_root), max_workers=8)\n"
        "(swefil_root / 'equinexus_revision.txt').write_text('bc818c69a79266ad0fa91a539359485af57967cf\\n', encoding='utf-8')\n"
        "audit_command = [sys.executable, '-u', str(REPOSITORY_PATH / 'scripts' / 'audit_swefil_external.py'), '--repo-root', str(REPOSITORY_PATH), '--competition-data-root', '/kaggle/input', '--swefil-root', str(swefil_root)]\n"
        "print('Auditing SWEFil external training data:', ' '.join(audit_command))\n"
        "subprocess.run(audit_command, check=True)\n"
        "pretrain_command = [sys.executable, '-u', str(REPOSITORY_PATH / 'scripts' / 'run_swefil_model_experiment.py'), '--repo-root', str(REPOSITORY_PATH), '--data-root', '/kaggle/input', '--swefil-root', str(swefil_root), '--run-name', "
        f"{run_name!r}, '--pretrain-steps', {str(external_pretrain_steps)!r}, '--steps', {str(steps)!r}, '--batch-size', {str(batch_size)!r}, '--patch-size', {str(patch_size)!r}, '--base-channels', {str(base_channels)!r}, '--model-type', {model_type!r}, '--tta', {str(tta)!r}, '--input-mode', {input_mode!r}, '--foreground-threshold', '0.55', '--boundary-threshold', '0.60', '--boundary-weight', '1.5', '--min-instance-area', '50']\n"
        "print('Running SWEFil supervised pretraining and competition fine-tuning:', ' '.join(pretrain_command))\n"
        "subprocess.run(pretrain_command, check=True)\n"
    )
    sweep_cell = (
        "run_root = Path('/kaggle/working/equinexus') / "
        f"{run_name!r}\n"
        "sweep_command = [sys.executable, '-u', str(REPOSITORY_PATH / 'scripts' / 'sweep_model_postprocessing.py'), '--repo-root', str(REPOSITORY_PATH), '--data-root', '/kaggle/input', '--run-root', "
        "str(run_root)]\n"
        "print('Sweeping validation postprocessing:', ' '.join(sweep_command))\n"
        "subprocess.run(sweep_command, check=True)\n"
        "import json\n"
        "from pathlib import Path\n"
        "best_config = json.loads((run_root / 'best-postprocessing' / 'metrics.json').read_text(encoding='utf-8'))['configuration']\n"
        "structure_command = [sys.executable, '-u', str(REPOSITORY_PATH / 'scripts' / 'sweep_model_instance_structure.py'), '--repo-root', str(REPOSITORY_PATH), '--data-root', '/kaggle/input', '--run-root', str(run_root), '--foreground-threshold', str(best_config['foreground_threshold']), '--boundary-threshold', str(best_config['boundary_threshold']), '--boundary-weight', str(best_config['boundary_weight']), '--seed-threshold', str(best_config['seed_threshold'])]\n"
        "print('Sweeping instance-size and seed cutoffs:', ' '.join(structure_command))\n"
        "subprocess.run(structure_command, check=True)\n"
    )
    final_cell = (
        "# Enable only after reviewing the completed holdout sweeps and selecting this model.\n"
        "RUN_FINAL_SUBMISSION = False\n"
        "if RUN_FINAL_SUBMISSION:\n"
        f"    selected = json.loads(Path({('/kaggle/working/equinexus/' + run_name + '/best-instance-structure/metrics.json')!r}).read_text(encoding='utf-8'))['configuration']\n"
        "    final_command = [sys.executable, '-u', str(REPOSITORY_PATH / 'scripts' / 'run_final_submission.py'), '--repo-root', str(REPOSITORY_PATH), '--data-root', '/kaggle/input', '--run-name', "
        f"{('final-' + run_name)!r}, '--target-mode', {target_mode!r}, '--steps', {str(steps)!r}, '--batch-size', {str(batch_size)!r}, '--patch-size', {str(patch_size)!r}, '--base-channels', {str(base_channels)!r}, '--model-type', {model_type!r}, '--hub-repository', {hub_repository!r}, '--tta', {str(tta)!r}, '--input-mode', {input_mode!r}, '--foreground-threshold', str(selected['foreground_threshold']), '--boundary-threshold', str(selected['boundary_threshold']), '--boundary-weight', str(selected['boundary_weight']), '--seed-threshold', str(selected['seed_threshold']), '--seed-min-area', str(selected['seed_min_area']), '--min-instance-area', str(selected['min_instance_area'])]\n"
        f"    final_command += ['--pretrained-backbone'] if {pretrained_backbone!r} else []\n"
        "    print('Training on all official observations and writing the validated submission:', ' '.join(final_command))\n"
        "    subprocess.run(final_command, check=True)\n"
        "else:\n"
        "    print('Final training is deferred until the validation model and postprocessing are selected.')\n"
    )
    report_cell = (
        "from pathlib import Path\n"
        f"for run in ({run_name!r}, {('final-' + run_name)!r}):\n"
        "    metrics_path = Path('/kaggle/working/equinexus') / run / 'metrics.json'\n"
        "    if metrics_path.is_file():\n"
        "        print(f'\\n{run}:\\n{metrics_path.read_text(encoding=\"utf-8\")}')\n"
        f"submission_path = Path({('/kaggle/working/equinexus/final-' + run_name + '/submission.csv')!r})\n"
        "print('Submission path:', submission_path if submission_path.is_file() else 'not generated by this validation-only version')\n"
    )
    cells = [
            _cell("markdown", markdown),
            _cell("code", source_cell),
            _cell("code", dependency_cell),
            _cell("code", hardware_cell),
        ]
    if solar_adaptation:
        cells.append(_cell("code", solar_adaptation_cell))
    if external_swefil:
        cells.append(_cell("code", swefil_cell))
    else:
        cells.append(_cell("code", model_cell))
    cells.extend(
        [
            _cell("code", sweep_cell),
            _cell("code", final_cell),
            _cell("code", report_cell),
        ]
    )
    notebook = {
        "cells": cells,
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
    kernel_notebook = kernel_dir / notebook_path.name
    kernel_notebook.write_bytes(notebook_path.read_bytes())
    metadata = {
        "id": kernel_id,
        "title": kernel_title,
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
    metadata_path = kernel_dir / "kernel-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Notebook: {notebook_path}")
    print(f"Kaggle bundle: {kernel_dir}")
    print(f"Embedded-source SHA-256: {bundle_sha256}")
    return notebook_path, metadata_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a reproducible Kaggle model experiment notebook.")
    parser.add_argument("--input-mode", choices=INPUT_MODES, default="raw")
    parser.add_argument("--target-mode", choices=("annotator", "consensus"), default="annotator")
    parser.add_argument("--run-name", default="unet-annotator-1024-kaggle-v2")
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=1_024)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--model-type", choices=MODEL_TYPES, default="unet")
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument("--hub-repository", default=DINO_V2_REPOSITORY)
    parser.add_argument("--solar-adaptation", action="store_true")
    parser.add_argument("--external-swefil", action="store_true")
    parser.add_argument("--external-pretrain-steps", type=int, default=1_000)
    parser.add_argument("--notebook-version", default="model-sprint-v3")
    parser.add_argument("--tta", type=int, choices=(1, 2, 4, 8), default=2)
    parser.add_argument("--notebook-path", default=None)
    parser.add_argument("--kernel-dir", default=None)
    parser.add_argument("--kernel-id", default=None)
    parser.add_argument("--kernel-title", default=None)
    build(**vars(parser.parse_args()))
