from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from filament_baseline.reinference import reinfer_validation  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recompute a saved model's frozen validation predictions with another TTA setting."
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--model-run-root", required=True)
    parser.add_argument("--tta", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--overlay-count", type=int, default=12)
    args = parser.parse_args()
    result = reinfer_validation(
        repo_root=args.repo_root,
        data_root=args.data_root,
        model_run_root=args.model_run_root,
        tta=args.tta,
        overlay_count=args.overlay_count,
    )
    print(json.dumps(result["runtime"], indent=2, sort_keys=True))
