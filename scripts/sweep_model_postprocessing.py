# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from filament_baseline.sweep import run_postprocessing_sweep  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep instance thresholds on cached validation maps.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--observation-limit", type=int, default=None)
    arguments = parser.parse_args()
    results = run_postprocessing_sweep(
        repo_root=arguments.repo_root,
        data_root=arguments.data_root,
        run_root=arguments.run_root,
        observation_limit=arguments.observation_limit,
    )
    print(json.dumps(results[:3], indent=2, sort_keys=True))
