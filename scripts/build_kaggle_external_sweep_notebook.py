# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from build_kaggle_chrotel_notebook import REPO_ROOT, _cell, _source_bundle


NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "external_postprocessing_sweeps.ipynb"
KERNEL_DIR = REPO_ROOT / "artifacts" / "kaggle" / "external-postprocessing-sweeps"
KERNEL_ID = "mrinanksivakumar/equinexus-external-postprocessing-sweeps"
FROZEN_SPLIT_SHA256 = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"


def build() -> tuple[Path, Path]:
    sources, bundle_sha256, source_commit = _source_bundle()
    intro = (
        "# EquiNexus external warm-start postprocessing sweeps\n\n"
        "This CPU-only notebook reuses validation probability caches from the completed "
        "ChroTel and Flat U-Net Kaggle runs. It searches foreground and boundary thresholds, "
        "then instance and seed area cutoffs, using only the frozen observation-grouped "
        "competition validation split and organizer-aligned scorer. It does not retrain "
        "models, use validation labels as training targets, or read public MAGFiLO test "
        "annotations. The held-out observations remain excluded from model fitting.\n\n"
        f"Frozen split SHA-256: `{FROZEN_SPLIT_SHA256}`. The source probability maps remain "
        "mounted read-only from the two private experiment kernels; only sweep results are "
        "written to `/kaggle/working/`.\n"
    )
    source_code = (
        "import base64, hashlib, json, os, sys\n"
        "from pathlib import Path\n"
        "REPOSITORY_PATH = Path('/kaggle/working/equinexus')\n"
        "REPOSITORY_PATH.mkdir(parents=True, exist_ok=True)\n"
        "os.environ['PIP_CACHE_DIR'] = str(REPOSITORY_PATH / '.cache' / 'pip')\n"
        "os.environ['TMP'] = str(REPOSITORY_PATH / '.tmp')\n"
        "os.environ['TEMP'] = str(REPOSITORY_PATH / '.tmp')\n"
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
        "import json, platform, sys, time\n"
        "COMPETITION_ROOT = Path('/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026')\n"
        "assert COMPETITION_ROOT.is_dir(), f'Competition input missing: {COMPETITION_ROOT}'\n"
        "print({'python': sys.version, 'platform': platform.platform(), 'competition_root': str(COMPETITION_ROOT), 'gpu_used': False})\n"
    )
    sweep_code = (
        "import hashlib, json, shutil, subprocess, sys, time\n"
        "from pathlib import Path\n"
        f"FROZEN_SPLIT_SHA256 = {FROZEN_SPLIT_SHA256!r}\n"
        "SOURCE_KERNELS = {\n"
        "    'chrotel': {'kernel': 'mrinanksivakumar/equinexus-chrotel-external-pretraining', 'run_name': 'unet-chrotel-weak-pretrain-finetune'},\n"
        "    'flat-unet': {'kernel': 'mrinanksivakumar/equinexus-flat-unet-external-pretraining', 'run_name': 'unet-flat-unet-weak-pretrain-finetune'},\n"
        "}\n"
        "INPUT_NOTEBOOKS = Path('/kaggle/input/notebooks')\n"
        "assert INPUT_NOTEBOOKS.is_dir(), f'Notebook sources missing: {INPUT_NOTEBOOKS}'\n"
        "def find_run(source_name, expected_run_name):\n"
        "    matches = []\n"
        "    for metrics_path in INPUT_NOTEBOOKS.rglob('metrics.json'):\n"
        "        try:\n"
        "            metrics = json.loads(metrics_path.read_text())\n"
        "        except (OSError, json.JSONDecodeError):\n"
        "            continue\n"
        "        if metrics.get('run_name') == expected_run_name:\n"
        "            matches.append((metrics_path.parent, metrics_path, metrics))\n"
        "    assert len(matches) == 1, f'{source_name}: expected one metrics file for {expected_run_name}, found {[str(row[1]) for row in matches]}'\n"
        "    return matches[0]\n"
        "def sha256_file(path):\n"
        "    digest = hashlib.sha256()\n"
        "    with path.open('rb') as stream:\n"
        "        for block in iter(lambda: stream.read(1024 * 1024), b''):\n"
        "            digest.update(block)\n"
        "    return digest.hexdigest()\n"
        "def run_command(command):\n"
        "    print('Running:', ' '.join(map(str, command)), flush=True)\n"
        "    subprocess.run(command, check=True)\n"
        "summaries = {}\n"
        "for source_name, source_info in SOURCE_KERNELS.items():\n"
        "    source_root, metrics_path, source_metrics = find_run(source_name, source_info['run_name'])\n"
        "    assert source_metrics['split']['sha256'] == FROZEN_SPLIT_SHA256, f'{source_name}: frozen split mismatch'\n"
        "    source_probabilities = source_root / 'probabilities'\n"
        "    assert source_probabilities.is_dir(), f'{source_name}: missing probability cache {source_probabilities}'\n"
        "    probability_files = sorted(source_probabilities.glob('*.npz'))\n"
        "    assert len(probability_files) == source_metrics['split']['evaluated_validation_observations'] == 142, f'{source_name}: expected 142 caches, found {len(probability_files)}'\n"
        "    run_root = REPOSITORY_PATH / 'artifacts' / 'external-postprocessing-sweeps' / source_name\n"
        "    run_root.mkdir(parents=True, exist_ok=True)\n"
        "    shutil.copy2(metrics_path, run_root / 'metrics.json')\n"
        "    cache_link = run_root / 'probabilities'\n"
        "    if cache_link.exists() or cache_link.is_symlink():\n"
        "        if cache_link.is_symlink(): cache_link.unlink()\n"
        "        else: shutil.rmtree(cache_link)\n"
        "    cache_link.symlink_to(source_probabilities, target_is_directory=True)\n"
        "    cache_bytes = sum(path.stat().st_size for path in probability_files)\n"
        "    print({'candidate': source_name, 'kernel': source_info['kernel'], 'source_run': str(source_root), 'cache_count': len(probability_files), 'cache_bytes': cache_bytes, 'source_metrics_sha256': sha256_file(metrics_path)}, flush=True)\n"
        "    started = time.perf_counter()\n"
        "    run_command([sys.executable, '-u', str(REPOSITORY_PATH / 'scripts' / 'sweep_model_postprocessing.py'), '--repo-root', str(REPOSITORY_PATH), '--data-root', str(COMPETITION_ROOT), '--run-root', str(run_root)])\n"
        "    threshold_rows = json.loads((run_root / 'postprocessing_sweep.json').read_text())\n"
        "    best_threshold = threshold_rows[0]['configuration']\n"
        "    run_command([sys.executable, '-u', str(REPOSITORY_PATH / 'scripts' / 'sweep_model_instance_structure.py'), '--repo-root', str(REPOSITORY_PATH), '--data-root', str(COMPETITION_ROOT), '--run-root', str(run_root), '--foreground-threshold', str(best_threshold['foreground_threshold']), '--boundary-threshold', str(best_threshold['boundary_threshold']), '--boundary-weight', str(best_threshold['boundary_weight']), '--seed-threshold', str(best_threshold['seed_threshold'])])\n"
        "    structure = json.loads((run_root / 'best-instance-structure' / 'metrics.json').read_text())\n"
        "    summaries[source_name] = {'source_kernel': source_info['kernel'], 'source_metrics_sha256': sha256_file(metrics_path), 'source_pq': source_metrics['metrics']['global']['pq'], 'source_instance_config': source_metrics['method']['instance_config'], 'best_threshold_sweep': threshold_rows[0], 'best_instance_structure': {'configuration': structure['configuration'], 'global': structure['metrics']['global']}, 'cache_count': len(probability_files), 'cache_bytes': cache_bytes, 'sweep_seconds': time.perf_counter() - started, 'split_sha256': FROZEN_SPLIT_SHA256}\n"
        "    (run_root / 'sweep-summary.json').write_text(json.dumps(summaries[source_name], indent=2, sort_keys=True) + '\\n')\n"
        "    print(json.dumps(summaries[source_name], indent=2, sort_keys=True), flush=True)\n"
        "summary_path = REPOSITORY_PATH / 'artifacts' / 'external-postprocessing-sweeps' / 'summary.json'\n"
        "summary_path.parent.mkdir(parents=True, exist_ok=True)\n"
        "summary_path.write_text(json.dumps({'status': 'complete', 'source_commit': SOURCE_COMMIT, 'source_bundle_sha256': SOURCE_BUNDLE_SHA256, 'split_sha256': FROZEN_SPLIT_SHA256, 'candidates': summaries}, indent=2, sort_keys=True) + '\\n')\n"
        "print({'summary_path': str(summary_path), 'candidates': list(summaries)}, flush=True)\n"
    )
    notebook = {
        "cells": [
            _cell("markdown", intro),
            _cell("code", source_code),
            _cell("code", runtime_code),
            _cell("code", sweep_code),
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
        "title": "EquiNexus External Postprocessing Sweeps",
        "code_file": NOTEBOOK_PATH.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "false",
        "enable_internet": "false",
        "machine_shape": "",
        "dataset_sources": [],
        "competition_sources": ["filament-segmentation-2026"],
        "kernel_sources": [
            "mrinanksivakumar/equinexus-chrotel-external-pretraining",
            "mrinanksivakumar/equinexus-flat-unet-external-pretraining",
        ],
        "model_sources": [],
    }
    (KERNEL_DIR / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "notebook": str(NOTEBOOK_PATH),
                "kernel_dir": str(KERNEL_DIR),
                "source_bundle_sha256": bundle_sha256,
                "source_commit": source_commit,
            },
            indent=2,
        )
    )
    return NOTEBOOK_PATH, KERNEL_DIR


if __name__ == "__main__":
    build()
