from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from filament_baseline.structure_sweep import run_instance_structure_sweep  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sweep instance and seed area cutoffs over cached validation maps."
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--foreground-threshold", type=float, default=0.55)
    parser.add_argument("--boundary-threshold", type=float, default=0.60)
    parser.add_argument("--boundary-weight", type=float, default=1.5)
    parser.add_argument("--boundary-weights", type=float, nargs="+", default=None)
    parser.add_argument("--seed-threshold", type=float, default=0.45)
    args = parser.parse_args()
    result = run_instance_structure_sweep(
        repo_root=args.repo_root,
        data_root=args.data_root,
        run_root=args.run_root,
        foreground_threshold=args.foreground_threshold,
        boundary_threshold=args.boundary_threshold,
        boundary_weight=args.boundary_weight,
        boundary_weights=tuple(args.boundary_weights) if args.boundary_weights else None,
        seed_threshold=args.seed_threshold,
    )
    print(json.dumps(result[:5], indent=2, sort_keys=True))
