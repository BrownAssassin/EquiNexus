# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from build_kaggle_chrotel_notebook import REPO_ROOT, _cell, _source_bundle


NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "flat_unet_external_pretraining.ipynb"
KERNEL_DIR = REPO_ROOT / "artifacts" / "kaggle" / "flat-unet-external-pretraining"
KERNEL_ID = "mrinanksivakumar/equinexus-flat-unet-external-pretraining"
KAGGLE_DATASET_ID = "mrinanksivakumar/flat-unet-date-disjoint-2026"
SELECTION_SHA256 = "049bd1dda9c2d95d161b96f7fe1b29f0bb053c44ffb754b610f9b8e0f78c0a22"
FROZEN_SPLIT_SHA256 = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"


def build(
    *,
    pretrain_steps: int = 1_500,
    fine_tune_steps: int = 5_000,
    batch_size: int = 2,
    patch_size: int = 512,
    tta: int = 4,
) -> tuple[Path, Path]:
    values = (pretrain_steps, fine_tune_steps, batch_size, patch_size)
    if any(value < 1 for value in values) or patch_size % 32:
        raise ValueError("Run sizes must be positive and patch size must be divisible by 32")
    if tta not in (1, 2, 4, 8):
        raise ValueError("tta must be one of 1, 2, 4, or 8")

    sources, bundle_sha256, source_commit = _source_bundle()
    sources["scripts/prepare_flat_unet_external.py"] = __import__("base64").b64encode(
        (REPO_ROOT / "scripts" / "prepare_flat_unet_external.py").read_bytes()
    ).decode("ascii")
    sources["scripts/run_chrotel_model_experiment.py"] = __import__("base64").b64encode(
        (REPO_ROOT / "scripts" / "run_chrotel_model_experiment.py").read_bytes()
    ).decode("ascii")
    digest = hashlib.sha256()
    for relative, payload in sorted(sources.items()):
        import base64

        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(base64.b64decode(payload))
        digest.update(b"\0")
    bundle_sha256 = digest.hexdigest()

    intro = (
        "# EquiNexus Flat U-Net weak-label pretraining\n\n"
        "This private GPU experiment uses 256 date-disjoint BBSO H-alpha observations and "
        "their binary semantic masks from the public China-VO PaperData dataset "
        "(DOI 10.12149/101545). The source page says that data copyrights remain with the "
        "authors and does not list a separate dataset license; this is a private compute "
        "copy used under competition Rule 6 with attribution and no public redistribution. "
        "The preparation excludes every competition train and test calendar date and does "
        "not use the source test_gd labels. Masks are weak semantic targets; the model is "
        "fine-tuned only on supplied competition training annotations. Final test inference "
        "uses only supplied H-alpha images.\n\n"
        f"Configuration: {pretrain_steps} external steps, {fine_tune_steps} competition "
        f"fine-tuning steps, batch size {batch_size}, {patch_size}-pixel patches, and "
        f"{tta}-view TTA. Frozen competition split SHA-256: `{FROZEN_SPLIT_SHA256}`.\n"
    )
    source_code = (
        "import base64, hashlib, json, os, sys\n"
        "from pathlib import Path\n"
        "REPOSITORY_PATH = Path('/kaggle/working/equinexus')\n"
        "REPOSITORY_PATH.mkdir(parents=True, exist_ok=True)\n"
        "os.environ['PIP_CACHE_DIR'] = str(REPOSITORY_PATH / '.cache' / 'pip')\n"
        "os.environ['TMP'] = str(REPOSITORY_PATH / '.tmp')\n"
        "os.environ['TEMP'] = str(REPOSITORY_PATH / '.tmp')\n"
        "os.environ['TORCH_HOME'] = str(REPOSITORY_PATH / '.cache' / 'torch')\n"
        "os.environ['HF_HOME'] = str(REPOSITORY_PATH / '.cache' / 'huggingface')\n"
        "Path(os.environ['TMP']).mkdir(parents=True, exist_ok=True)\n"
        f"SOURCE_FILES = json.loads({json.dumps(json.dumps(sources))})\n"
        "for relative, payload in SOURCE_FILES.items():\n"
        "    destination = REPOSITORY_PATH / relative\n"
        "    destination.parent.mkdir(parents=True, exist_ok=True)\n"
        "    destination.write_bytes(base64.b64decode(payload))\n"
        f"SOURCE_COMMIT = {source_commit!r}\n"
        f"SOURCE_BUNDLE_SHA256 = {bundle_sha256!r}\n"
        "os.environ['EQUINEXUS_SOURCE_COMMIT'] = SOURCE_COMMIT\n"
        "os.environ['EQUINEXUS_SOURCE_BUNDLE_SHA256'] = SOURCE_BUNDLE_SHA256\n"
        "sys.path.insert(0, str(REPOSITORY_PATH / 'src'))\n"
        "os.chdir(REPOSITORY_PATH)\n"
        "print({'source_commit': SOURCE_COMMIT, 'source_bundle_sha256': SOURCE_BUNDLE_SHA256, 'embedded_files': len(SOURCE_FILES)})\n"
    )
    runtime_code = (
        "import json, sys, torch\n"
        "COMPETITION_ROOT = Path('/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026')\n"
        "assert COMPETITION_ROOT.is_dir(), f'Competition input missing: {COMPETITION_ROOT}'\n"
        "assert torch.cuda.is_available(), 'Enable a Kaggle GPU accelerator for this run'\n"
        "print({'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0), 'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory})\n"
    )
    prepare_code = (
        "import hashlib, json, shutil\n"
        "def sha256_path(path):\n"
        "    digest = hashlib.sha256()\n"
        "    with path.open('rb') as stream:\n"
        "        for block in iter(lambda: stream.read(1024 * 1024), b''):\n"
        "            digest.update(block)\n"
        "    return digest.hexdigest()\n"
        "EXTERNAL_ROOT = REPOSITORY_PATH / 'data' / 'external' / 'flat-unet-101545' / 'processed-256'\n"
        "EXTERNAL_ROOT.parent.mkdir(parents=True, exist_ok=True)\n"
        "manifest_paths = list(Path('/kaggle/input').rglob('manifest.json'))\n"
        "matching_manifests = []\n"
        "for path in manifest_paths:\n"
        "    try:\n"
        "        candidate = json.loads(path.read_text())\n"
        "    except (OSError, json.JSONDecodeError):\n"
        "        continue\n"
        f"    if candidate.get('source_doi') == '10.12149/101545' and candidate.get('selection_manifest_sha256') == {SELECTION_SHA256!r}:\n"
        "        matching_manifests.append((path, candidate))\n"
        "assert len(matching_manifests) == 1, f'Expected one matching Flat U-Net manifest, found {[str(path) for path, _ in matching_manifests]}'\n"
        "source_manifest_path, manifest = matching_manifests[0]\n"
        "source_root = source_manifest_path.parent\n"
        "assert manifest['status'] == 'complete' and manifest['all_competition_dates_excluded'] is True and manifest['competition_date_overlap'] == 0\n"
        f"assert manifest['competition_split_sha256'] == {FROZEN_SPLIT_SHA256!r}\n"
        "assert manifest['prepared_observations'] == 256 and manifest['prepared_dates'] == 256\n"
        "print({'attached_private_source': str(source_root), 'source_manifest_sha256': sha256_path(source_manifest_path)})\n"
        "shutil.copytree(source_root, EXTERNAL_ROOT, dirs_exist_ok=True)\n"
        "manifest = json.loads((EXTERNAL_ROOT / 'manifest.json').read_text())\n"
        "for row in manifest['observations']:\n"
        "    for kind, folder in (('image', 'images'), ('mask', 'masks')):\n"
        "        path = EXTERNAL_ROOT / folder / row[kind]['file_name']\n"
        "        assert hashlib.sha256(path.read_bytes()).hexdigest() == row[kind]['processed_sha256'], f'{kind} hash mismatch: {path.name}'\n"
        "print({'prepared_observations': manifest['prepared_observations'], 'selection_manifest_sha256': manifest['selection_manifest_sha256'], 'competition_date_overlap': manifest['competition_date_overlap'], 'source_archive_sha256': manifest['source_archive_sha256']})\n"
    )
    train_code = (
        "import json, shutil, subprocess, sys\n"
        "RUN_NAME = 'unet-flat-unet-weak-pretrain-finetune'\n"
        "RUN_ROOT = Path('/kaggle/working/equinexus') / RUN_NAME\n"
        "(RUN_ROOT / 'external-pretraining').mkdir(parents=True, exist_ok=True)\n"
        "shutil.copy2(EXTERNAL_ROOT / 'manifest.json', RUN_ROOT / 'external-pretraining' / 'flat-unet-source-manifest.json')\n"
        "train_command = [sys.executable, '-u', str(REPOSITORY_PATH / 'scripts' / 'run_chrotel_model_experiment.py'), "
        "'--repo-root', str(REPOSITORY_PATH), '--data-root', str(COMPETITION_ROOT), '--external-source', 'flat-unet', '--external-root', str(EXTERNAL_ROOT), "
        f"'--run-name', RUN_NAME, '--pretrain-steps', {str(pretrain_steps)!r}, '--steps', {str(fine_tune_steps)!r}, "
        f"'--batch-size', {str(batch_size)!r}, '--patch-size', {str(patch_size)!r}, '--tta', {str(tta)!r}]\n"
        "print('Training and scoring Flat U-Net weak-label warm start:', ' '.join(train_command), flush=True)\n"
        "subprocess.run(train_command, check=True)\n"
        "metrics_path = RUN_ROOT / 'metrics.json'\n"
        "metrics = json.loads(metrics_path.read_text())\n"
        "print(json.dumps({'status': metrics['status'], 'run_name': metrics['run_name'], 'split': metrics['split'], 'global': metrics['metrics']['global'], 'runtime': metrics['runtime']}, indent=2, sort_keys=True))\n"
        "shutil.rmtree(EXTERNAL_ROOT)\n"
        "print({'cleaned_external_working_data': str(EXTERNAL_ROOT), 'retained_run_artifacts': str(RUN_ROOT)})\n"
    )
    notebook = {
        "cells": [
            _cell("markdown", intro),
            _cell("code", source_code),
            _cell("code", runtime_code),
            _cell("code", prepare_code),
            _cell("code", train_code),
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
        "title": "EquiNexus Flat Unet External Pretraining",
        "code_file": NOTEBOOK_PATH.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "false",
        "machine_shape": "",
        "dataset_sources": [KAGGLE_DATASET_ID],
        "competition_sources": ["filament-segmentation-2026"],
        "kernel_sources": [],
        "model_sources": [],
    }
    (KERNEL_DIR / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return NOTEBOOK_PATH, KERNEL_DIR


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Kaggle Flat U-Net warm-start notebook.")
    parser.add_argument("--pretrain-steps", type=int, default=1_500)
    parser.add_argument("--fine-tune-steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--tta", type=int, choices=(1, 2, 4, 8), default=4)
    args = parser.parse_args()
    notebook, kernel_dir = build(**vars(args))
    print(json.dumps({"notebook": str(notebook), "kernel_dir": str(kernel_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
