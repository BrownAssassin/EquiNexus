from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from filament_baseline.classical import ClassicalConfig  # noqa: E402
from filament_baseline.classical_runner import run_classical_validation  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-run a public classical approach on frozen validation.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--run-name", default="radial-blackhat-public-method")
    parser.add_argument("--observation-limit", type=int, default=None)
    parser.add_argument("--z-threshold", type=float, default=None)
    parser.add_argument("--bridge-size", type=int, default=None)
    parser.add_argument("--min-instance-area", type=int, default=None)
    arguments = parser.parse_args()
    selected = (
        arguments.z_threshold,
        arguments.bridge_size,
        arguments.min_instance_area,
    )
    if any(value is not None for value in selected) and not all(
        value is not None for value in selected
    ):
        parser.error(
            "--z-threshold, --bridge-size, and --min-instance-area must be set together"
        )
    candidate = (
        ClassicalConfig(
            z_threshold=arguments.z_threshold,
            bridge_size=arguments.bridge_size,
            min_instance_area=arguments.min_instance_area,
        )
        if all(value is not None for value in selected)
        else None
    )
    result = run_classical_validation(
        repo_root=arguments.repo_root,
        data_root=arguments.data_root,
        run_name=arguments.run_name,
        observation_limit=arguments.observation_limit,
        single_config=candidate,
    )
    print(json.dumps(result["best"], indent=2, sort_keys=True))
