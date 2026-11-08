# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_counts(path: Path) -> dict[str, tuple[float, float, float, float]]:
    result = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            name = row["file_name"]
            if name in result:
                raise ValueError(f"Duplicate observation row in {path}: {name}")
            result[name] = (
                float(row["tp_iou_sum"]),
                float(row["tp_edges"]),
                float(row["fp"]),
                float(row["fn"]),
            )
    return result


def _aggregate_pq(samples: np.ndarray) -> np.ndarray:
    iou_sum = samples[..., 0].sum(axis=1)
    tp = samples[..., 1].sum(axis=1)
    fp = samples[..., 2].sum(axis=1)
    fn = samples[..., 3].sum(axis=1)
    denominator = tp + 0.5 * fp + 0.5 * fn
    return np.divide(iou_sum, denominator, out=np.zeros_like(iou_sum), where=denominator > 0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Paired bootstrap of scorer counts by original observation group."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repo_root = args.repo_root.expanduser().resolve()
    baseline_path = args.baseline.expanduser().resolve()
    candidate_path = args.candidate.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    for label, path in (("baseline", baseline_path), ("candidate", candidate_path), ("output", output_path)):
        if not path.is_relative_to(repo_root):
            parser.error(f"{label} file must remain inside the repository: {path}")
    if args.resamples < 100:
        parser.error("At least 100 bootstrap resamples are required")

    baseline = _read_counts(baseline_path)
    candidate = _read_counts(candidate_path)
    names = sorted(baseline)
    if not names or set(names) != set(candidate):
        raise ValueError("Baseline and candidate must contain the same nonempty observation groups")
    baseline_array = np.asarray([baseline[name] for name in names], dtype=np.float64)
    candidate_array = np.asarray([candidate[name] for name in names], dtype=np.float64)
    baseline_pq = float(_aggregate_pq(baseline_array[None, ...])[0])
    candidate_pq = float(_aggregate_pq(candidate_array[None, ...])[0])

    generator = np.random.default_rng(args.seed)
    indices = generator.integers(0, len(names), size=(args.resamples, len(names)))
    baseline_bootstrap = _aggregate_pq(baseline_array[indices])
    candidate_bootstrap = _aggregate_pq(candidate_array[indices])
    delta = candidate_bootstrap - baseline_bootstrap
    result = {
        "resampling_unit": "original observation; all annotator-view counts stay together",
        "observation_count": len(names),
        "resamples": args.resamples,
        "seed": args.seed,
        "baseline": {
            "path": str(baseline_path.relative_to(repo_root)),
            "pq": baseline_pq,
            "bootstrap_ci_95": [float(value) for value in np.quantile(baseline_bootstrap, [0.025, 0.975])],
        },
        "candidate": {
            "path": str(candidate_path.relative_to(repo_root)),
            "pq": candidate_pq,
            "bootstrap_ci_95": [float(value) for value in np.quantile(candidate_bootstrap, [0.025, 0.975])],
        },
        "paired_delta_candidate_minus_baseline": {
            "point_estimate": candidate_pq - baseline_pq,
            "bootstrap_ci_95": [float(value) for value in np.quantile(delta, [0.025, 0.975])],
            "probability_candidate_better": float(np.mean(delta > 0)),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
