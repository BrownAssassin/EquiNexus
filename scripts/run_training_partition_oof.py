from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from filament_baseline.inference import InstanceConfig  # noqa: E402
from filament_baseline.oof import run_training_partition_oof  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run grouped out-of-fold validation within the official training partition."
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--target-mode", choices=("annotator", "consensus"), default="consensus")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--tta", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument(
        "--input-mode", choices=("raw", "raw_local_contrast"), default="raw"
    )
    parser.add_argument("--foreground-threshold", type=float, default=0.55)
    parser.add_argument("--boundary-threshold", type=float, default=0.60)
    parser.add_argument("--boundary-weight", type=float, default=1.5)
    parser.add_argument("--seed-threshold", type=float, default=0.45)
    parser.add_argument("--seed-min-area", type=int, default=8)
    parser.add_argument("--min-instance-area", type=int, default=50)
    args = parser.parse_args()
    result = run_training_partition_oof(
        repo_root=args.repo_root,
        data_root=args.data_root,
        run_name=args.run_name,
        target_mode=args.target_mode,
        folds=args.folds,
        steps=args.steps,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        tta=args.tta,
        input_mode=args.input_mode,
        instance_config=InstanceConfig(
            foreground_threshold=args.foreground_threshold,
            boundary_threshold=args.boundary_threshold,
            boundary_weight=args.boundary_weight,
            seed_threshold=args.seed_threshold,
            seed_min_area=args.seed_min_area,
            min_instance_area=args.min_instance_area,
        ),
    )
    print(json.dumps(result["aggregate_oof"], indent=2, sort_keys=True))
