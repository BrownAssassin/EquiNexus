# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_SPLIT_SHA256 = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"
OOF_MANIFEST_SHA256 = "f856708222f5747cd35760932f535e4ae2752fd5161b9cd5981fbf4bbde26a3c"
OOF_MANIFEST_NAME = "observation_oof_seed42_5fold.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_checkpoints(*, repo_root: Path, source_root: Path, output_root: Path) -> dict:
    repo_root = repo_root.resolve()
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    for label, path in (("source", source_root), ("output", output_root)):
        if not path.is_relative_to(repo_root):
            raise ValueError(f"Local {label} path must stay in the repository: {path}")

    source_metrics_path = source_root / "metrics.json"
    source_metrics = json.loads(source_metrics_path.read_text(encoding="utf-8"))
    if source_metrics.get("status") != "complete":
        raise ValueError("Source OOF training run is not complete")
    if source_metrics.get("data_boundary", {}).get("external_ground_truth") is not False:
        raise ValueError("Source checkpoints do not certify competition-only ground truth")
    splits = source_metrics.get("splits", {})
    if splits.get("frozen_holdout_sha256") != FROZEN_SPLIT_SHA256:
        raise ValueError("Source checkpoints do not use the frozen holdout")
    if splits.get("oof_manifest_sha256") != OOF_MANIFEST_SHA256:
        raise ValueError("Source checkpoints do not use the frozen grouped OOF manifest")
    oof_manifest_path = repo_root / "data" / "splits" / OOF_MANIFEST_NAME
    if not oof_manifest_path.is_file() or _sha256(oof_manifest_path) != OOF_MANIFEST_SHA256:
        raise ValueError("The local grouped OOF manifest is missing or has changed")
    method = source_metrics.get("method", {})
    expected_method = {
        "target_mode": "annotator",
        "patch_size": 1024,
        "steps_per_fold": 10000,
        "tta": 8,
        "external_supervised_weights": False,
    }
    if any(method.get(key) != value for key, value in expected_method.items()):
        raise ValueError(f"Unexpected checkpoint training recipe: {method}")
    fold_summaries = {
        int(item["fold"]): item for item in source_metrics.get("folds", [])
    }
    if set(fold_summaries) != set(range(5)):
        raise ValueError("Source run must contain exactly five fold summaries")

    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Checkpoint bundle folder is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(oof_manifest_path, output_root / OOF_MANIFEST_NAME)
    checkpoints = []
    for fold in range(5):
        fold_root = source_root / f"fold-{fold}"
        fold_metrics_path = fold_root / "metrics.json"
        checkpoint_path = fold_root / "best_model.pt"
        fold_metrics = json.loads(fold_metrics_path.read_text(encoding="utf-8"))
        fold_summary = fold_summaries[fold]
        if (
            int(fold_metrics.get("fold", -1)) != fold
            or int(fold_summary.get("fold", -1)) != fold
        ):
            raise ValueError(f"Fold {fold} identity differs between run and fold metrics")
        if (
            fold_metrics.get("training_observations") != 452
            or fold_metrics.get("validation_observations") != 113
            or fold_summary.get("training_observations") != 452
            or fold_summary.get("validation_observations") != 113
        ):
            raise ValueError(f"Fold {fold} group counts are unexpected")
        for key in ("best_step", "seed", "steps", "selection_mode"):
            if fold_metrics.get("training", {}).get(key) != fold_summary.get("training", {}).get(key):
                raise ValueError(f"Fold {fold} training metadata differs for {key}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        output_path = output_root / f"fold-{fold}.pt"
        shutil.copy2(checkpoint_path, output_path)
        checkpoints.append(
            {
                "fold": fold,
                "file": output_path.name,
                "sha256": _sha256(output_path),
                "bytes": output_path.stat().st_size,
                "metrics_sha256": _sha256(fold_metrics_path),
                "training_observations": fold_metrics["training_observations"],
                "validation_observations": fold_metrics["validation_observations"],
                "best_step": fold_metrics.get("training", {}).get("best_step"),
                "seed": fold_metrics.get("training", {}).get("seed"),
                "best_validation_patch_dice": fold_metrics.get("training", {}).get("best_validation_patch_dice"),
            }
        )

    manifest = {
        "status": "complete",
        "source_run": str(source_root.relative_to(repo_root)),
        "source_metrics_sha256": _sha256(source_metrics_path),
        "source_repository": source_metrics.get("repository", {}),
        "frozen_split_sha256": FROZEN_SPLIT_SHA256,
        "oof_manifest_sha256": OOF_MANIFEST_SHA256,
        "oof_manifest_file": OOF_MANIFEST_NAME,
        "training": {
            "labels": "competition-provided train annotations only",
            "target_mode": "per-annotator views",
            "external_supervised_weights": False,
            "test_observations_used": False,
            "architecture": method.get("architecture"),
            "steps_per_fold": method["steps_per_fold"],
            "patch_size": method["patch_size"],
            "tta": method["tta"],
        },
        "checkpoints": checkpoints,
    }
    (output_root / "checkpoint-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package validated grouped-OOF checkpoint files for a private Kaggle dataset.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--source-root", type=Path, default=REPO_ROOT / "artifacts" / "oof-experiments" / "oof-perann1024-10k-tta8")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "artifacts" / "kaggle" / "fold-checkpoints")
    args = parser.parse_args()
    manifest = package_checkpoints(repo_root=args.repo_root, source_root=args.source_root, output_root=args.output_root)
    print(json.dumps({"status": manifest["status"], "checkpoints": manifest["checkpoints"], "frozen_split_sha256": manifest["frozen_split_sha256"], "oof_manifest_sha256": manifest["oof_manifest_sha256"]}, indent=2))
