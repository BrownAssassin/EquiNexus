# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from filament_baseline.data import load_dataset
from filament_baseline.external_data import load_morden_ahas_dataset
from filament_baseline.features import INPUT_MODES
from filament_baseline.learning import FilamentPatchDataset
from filament_baseline.model_runner import _is_kaggle_runtime, run_model_experiment
from filament_baseline.split import create_observation_split
from filament_baseline.training import train_model


_MORDEN_DATE = re.compile(r"^RSM(\d{8})T", re.IGNORECASE)


def _external_date_split(
    observation_names: tuple[str, ...], validation_fraction: float = 0.2
) -> tuple[list[str], list[str]]:
    dates_by_name: dict[str, str] = {}
    for name in observation_names:
        match = _MORDEN_DATE.match(Path(name).name)
        if match is None:
            raise ValueError(f"Could not parse a CHASE date from external image {name}")
        dates_by_name[name] = match.group(1)
    dates = sorted(set(dates_by_name.values()))
    validation_dates = {
        date
        for date in dates
        if int.from_bytes(hashlib.sha256(date.encode()).digest()[:8], "big") / 2**64
        < validation_fraction
    }
    if not validation_dates or validation_dates == set(dates):
        validation_dates = set(dates[::5])
    train_names = [name for name in observation_names if dates_by_name[name] not in validation_dates]
    validation_names = [name for name in observation_names if dates_by_name[name] in validation_dates]
    if {dates_by_name[name] for name in train_names} & {
        dates_by_name[name] for name in validation_names
    }:
        raise AssertionError("MORDEN external pretraining split crossed source dates")
    return train_names, validation_names


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Pretrain on audited, date-disjoint MORDEN AHAS labels, fine-tune on the "
            "official training partition, and score the frozen competition holdout."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--morden-root", type=Path, default=None)
    parser.add_argument("--run-name", default="unet-morden-ahas-pretrain-finetune-local")
    parser.add_argument("--pretrain-steps", type=int, default=3_000)
    parser.add_argument("--steps", type=int, default=3_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--tta", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--input-mode", choices=INPUT_MODES, default="raw_local_contrast")
    parser.add_argument("--foreground-threshold", type=float, default=0.55)
    parser.add_argument("--boundary-threshold", type=float, default=0.60)
    parser.add_argument("--boundary-weight", type=float, default=1.5)
    parser.add_argument("--min-instance-area", type=int, default=50)
    args = parser.parse_args()

    repo_root = args.repo_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve() if args.data_root else None
    morden_root = (
        args.morden_root.expanduser().resolve()
        if args.morden_root
        else repo_root / "data" / "external" / "morden-zenodo-21582245"
    )
    if data_root is not None and not data_root.is_relative_to(repo_root / "data"):
        raise ValueError(f"Competition data must remain under data/: {data_root}")
    if not morden_root.is_relative_to(repo_root / "data" / "external"):
        raise ValueError(f"MORDEN data must remain under data/external/: {morden_root}")

    config = json.loads((repo_root / "configs" / "validation-baseline.json").read_text())
    competition = load_dataset(data_root or repo_root / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026")
    split = create_observation_split(
        competition,
        repo_root / "data" / "splits",
        seed=int(config["seed"]),
        validation_fraction=float(config["validation_fraction"]),
    )
    if split.sha256 != config["expected_split_sha256"]:
        raise RuntimeError(f"Frozen competition split changed: {split.sha256}")
    external, provenance = load_morden_ahas_dataset(
        morden_root, competition, repository_root=repo_root
    )
    external_train_names, external_validation_names = _external_date_split(
        external.observation_names
    )
    provenance["external_pretraining_split"] = {
        "group_key": "CHASE calendar date parsed from RSM filenames",
        "train_observations": len(external_train_names),
        "validation_observations": len(external_validation_names),
        "train_dates": len(external_train_names),
        "validation_dates": len(external_validation_names),
        "validation_fraction": 0.2,
        "split_algorithm": "SHA-256 date hash threshold",
    }

    run_root = (
        Path("/kaggle/working/equinexus") / args.run_name
        if _is_kaggle_runtime()
        else repo_root / "artifacts" / "model-experiments" / args.run_name
    )
    if not _is_kaggle_runtime() and not run_root.is_relative_to(repo_root):
        raise ValueError(f"Run artifacts must remain inside the workspace: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    external_root = run_root / "external-pretraining"
    external_root.mkdir(parents=True, exist_ok=True)
    (external_root / "data-provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    train_dataset = FilamentPatchDataset(
        external,
        external_train_names,
        target_mode="annotator",
        patch_size=args.patch_size,
        augment=True,
        seed=int(config["seed"]),
        input_mode=args.input_mode,
    )
    validation_dataset = FilamentPatchDataset(
        external,
        external_validation_names,
        target_mode="annotator",
        patch_size=args.patch_size,
        augment=False,
        seed=int(config["seed"]),
        input_mode=args.input_mode,
    )
    print(
        f"MORDEN/AHAS pretraining on {len(external_train_names)} observations; "
        f"date-disjoint validation {len(external_validation_names)}; "
        f"frozen split {split.sha256}",
        flush=True,
    )
    pretrain_result = train_model(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        output_path=external_root / "best_model.pt",
        steps=args.pretrain_steps,
        batch_size=args.batch_size,
        base_channels=args.base_channels,
        seed=int(config["seed"]),
        validation_interval=max(100, min(500, args.pretrain_steps // 5)),
    )
    (external_root / "training.json").write_text(
        json.dumps(pretrain_result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    provenance["pretraining"] = pretrain_result
    (external_root / "data-provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
        model_type="unet",
        tta=args.tta,
        input_mode=args.input_mode,
        foreground_threshold=args.foreground_threshold,
        boundary_threshold=args.boundary_threshold,
        boundary_weight=args.boundary_weight,
        min_instance_area=args.min_instance_area,
        initial_weights=external_root / "best_model.pt",
        external_pretraining=provenance,
    )
    print(json.dumps(result["metrics"]["global"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
