# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from filament_baseline.ensemble_submission import run_final_ensemble_submission  # noqa: E402
from filament_baseline.inference import InstanceConfig  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train two final models on official training data and create a probability-blended Kaggle submission."
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--first-name", default="annotator1024")
    parser.add_argument("--first-target-mode", choices=("annotator", "consensus"), default="annotator")
    parser.add_argument("--first-steps", type=int, default=5_000)
    parser.add_argument("--first-batch-size", type=int, default=2)
    parser.add_argument("--first-patch-size", type=int, default=1_024)
    parser.add_argument("--first-tta", type=int, choices=(1, 2, 4, 8), default=2)
    parser.add_argument(
        "--first-input-mode", choices=("raw", "raw_local_contrast", "raw_local_contrast_radial"), default="raw"
    )
    parser.add_argument("--second-name", default="consensus512")
    parser.add_argument("--second-target-mode", choices=("annotator", "consensus"), default="consensus")
    parser.add_argument("--second-steps", type=int, default=5_000)
    parser.add_argument("--second-batch-size", type=int, default=8)
    parser.add_argument("--second-patch-size", type=int, default=512)
    parser.add_argument("--second-tta", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument(
        "--second-input-mode", choices=("raw", "raw_local_contrast", "raw_local_contrast_radial"), default="raw"
    )
    parser.add_argument("--first-weight", type=float, default=0.5)
    parser.add_argument("--foreground-threshold", type=float, required=True)
    parser.add_argument("--boundary-threshold", type=float, required=True)
    parser.add_argument("--boundary-weight", type=float, required=True)
    parser.add_argument("--seed-threshold", type=float, required=True)
    parser.add_argument("--seed-min-area", type=int, required=True)
    parser.add_argument("--min-instance-area", type=int, required=True)
    args = parser.parse_args()

    result = run_final_ensemble_submission(
        repo_root=args.repo_root,
        data_root=args.data_root,
        run_name=args.run_name,
        first_member={
            "name": args.first_name,
            "target_mode": args.first_target_mode,
            "steps": args.first_steps,
            "batch_size": args.first_batch_size,
            "patch_size": args.first_patch_size,
            "tta": args.first_tta,
            "input_mode": args.first_input_mode,
        },
        second_member={
            "name": args.second_name,
            "target_mode": args.second_target_mode,
            "steps": args.second_steps,
            "batch_size": args.second_batch_size,
            "patch_size": args.second_patch_size,
            "tta": args.second_tta,
            "input_mode": args.second_input_mode,
        },
        first_weight=args.first_weight,
        instance_config=InstanceConfig(
            foreground_threshold=args.foreground_threshold,
            boundary_threshold=args.boundary_threshold,
            boundary_weight=args.boundary_weight,
            seed_threshold=args.seed_threshold,
            seed_min_area=args.seed_min_area,
            min_instance_area=args.min_instance_area,
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
