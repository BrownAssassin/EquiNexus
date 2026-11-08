# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from filament_baseline.data import load_dataset
from filament_baseline.external_data import load_swefil_dataset
from filament_baseline.features import INPUT_MODES
from filament_baseline.learning import FilamentPatchDataset
from filament_baseline.model import DINO_V2_REPOSITORY
from filament_baseline.model_runner import _is_kaggle_runtime, run_model_experiment
from filament_baseline.split import create_observation_split
from filament_baseline.training import train_model


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Pretrain a DINOv2 filament decoder on disjoint public SWEFil labels, "
            "then fine-tune and score on the frozen competition split."
        )
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--swefil-root", default=None)
    parser.add_argument("--run-name", default="swefil-dinov2-pretrain-1000-finetune-5000")
    parser.add_argument("--pretrain-steps", type=int, default=1_000)
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument(
        "--model-type", choices=("unet", "dinov2_vits14"), default="unet"
    )
    parser.add_argument("--tta", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--input-mode", choices=INPUT_MODES, default="raw_local_contrast")
    parser.add_argument("--foreground-threshold", type=float, default=0.65)
    parser.add_argument("--boundary-threshold", type=float, default=0.70)
    parser.add_argument("--boundary-weight", type=float, default=1.5)
    parser.add_argument("--min-instance-area", type=int, default=400)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else None
    swefil_root = (
        Path(args.swefil_root).expanduser().resolve()
        if args.swefil_root
        else repo_root / "data" / "external" / "SWEFil"
    )
    if data_root is not None and not data_root.is_relative_to(repo_root / "data"):
        raise ValueError(f"Competition data must remain under data/: {data_root}")
    if not swefil_root.is_relative_to(repo_root / "data" / "external"):
        raise ValueError(f"SWEFil must remain under data/external/: {swefil_root}")

    config = json.loads((repo_root / "configs" / "validation-baseline.json").read_text())
    competition = load_dataset(data_root or repo_root / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026")
    split = create_observation_split(
        competition,
        repo_root / "data" / "splits",
        seed=int(config["seed"]),
        validation_fraction=float(config["validation_fraction"]),
    )
    expected_digest = config["expected_split_sha256"]
    if split.sha256 != expected_digest:
        raise RuntimeError(f"Frozen split digest changed: {split.sha256} != {expected_digest}")

    external_dataset, external_metadata = load_swefil_dataset(
        swefil_root,
        competition,
        split,
        repository_root=repo_root,
    )
    run_root = (
        Path("/kaggle/working/equinexus") / args.run_name
        if _is_kaggle_runtime()
        else repo_root / "artifacts" / "model-experiments" / args.run_name
    )
    pretrain_root = run_root / "external-pretraining"
    pretrain_root.mkdir(parents=True, exist_ok=True)
    (pretrain_root / "data-provenance.json").write_text(
        json.dumps(external_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    hub_dir = (
        Path("/kaggle/working/equinexus/pretrained/torch/hub")
        if _is_kaggle_runtime()
        else repo_root / "artifacts" / "pretrained" / "torch" / "hub"
    )
    adapted_checkpoint = hub_dir / "checkpoints" / "dinov2_vits14_pretrain.pth"
    adaptation_metadata_path = adapted_checkpoint.with_suffix(".json")
    if adapted_checkpoint.is_file() and adaptation_metadata_path.is_file():
        adaptation_metadata = json.loads(adaptation_metadata_path.read_text(encoding="utf-8"))
        external_metadata["unlabeled_adaptation"] = {
            "method": "DINOv2 self-distillation on public unlabeled GONG H-alpha",
            "repository": DINO_V2_REPOSITORY,
            "license": "Apache-2.0",
            "adapted_checkpoint_sha256": hashlib.sha256(adapted_checkpoint.read_bytes()).hexdigest(),
            "metadata": adaptation_metadata,
        }
    else:
        external_metadata["unlabeled_adaptation"] = None
    external_train = FilamentPatchDataset(
        external_dataset,
        external_dataset.observation_names,
        target_mode="annotator",
        patch_size=args.patch_size,
        augment=True,
        seed=int(config["seed"]),
        input_mode=args.input_mode,
    )
    competition_validation = FilamentPatchDataset(
        competition,
        split.validation_names,
        target_mode="annotator",
        patch_size=args.patch_size,
        augment=False,
        seed=int(config["seed"]),
        input_mode=args.input_mode,
    )

    print(
        f"External pretraining: {len(external_dataset.observation_names)} SWEFil observations; "
        f"competition holdout: {len(split.validation_names)} observations; "
        f"split {split.sha256}",
        flush=True,
    )
    pretrain_result = train_model(
        train_dataset=external_train,
        validation_dataset=competition_validation,
        output_path=pretrain_root / "best_model.pt",
        steps=args.pretrain_steps,
        batch_size=args.batch_size,
        base_channels=args.base_channels,
        seed=int(config["seed"]),
        validation_interval=max(100, min(500, args.pretrain_steps // 5)),
        model_type=args.model_type,
        pretrained_backbone=args.model_type == "dinov2_vits14",
        hub_repository=DINO_V2_REPOSITORY,
        hub_dir=hub_dir,
    )
    (pretrain_root / "training.json").write_text(
        json.dumps(pretrain_result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    external_metadata["pretraining"] = pretrain_result
    (pretrain_root / "data-provenance.json").write_text(
        json.dumps(external_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = run_model_experiment(
        repo_root=repo_root,
        data_root=data_root,
        run_name=args.run_name,
        target_mode="annotator",
        steps=args.steps,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        base_channels=args.base_channels,
        model_type=args.model_type,
        pretrained_backbone=False,
        hub_repository=DINO_V2_REPOSITORY,
        tta=args.tta,
        input_mode=args.input_mode,
        foreground_threshold=args.foreground_threshold,
        boundary_threshold=args.boundary_threshold,
        boundary_weight=args.boundary_weight,
        min_instance_area=args.min_instance_area,
        initial_weights=pretrain_root / "best_model.pt",
        external_pretraining=external_metadata,
    )
    print(json.dumps(result["metrics"]["global"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
