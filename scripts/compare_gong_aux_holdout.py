#!/usr/bin/env python3
# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Paired observation-bootstrap comparison of native and GONG U-Net holdout results."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from compare_gong_aux_oof import EXPECTED_SPLIT_SHA256, paired_bootstrap


def load_rows(path: Path) -> list[SimpleNamespace]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [
        SimpleNamespace(
            file_name=row["file_name"],
            tp_iou_sum=float(row["tp_iou_sum"]),
            tp=int(row["tp"]),
            fp=int(row["fp"]),
            fn=int(row["fn"]),
            pq=float(row["pq"]),
        )
        for row in rows
    ]


def load_run(run_dir: Path) -> tuple[dict[str, Any], list[SimpleNamespace]]:
    metrics_path = run_dir / "holdout-metrics.json"
    rows_path = run_dir / "holdout-per-observation.csv"
    if not metrics_path.is_file() or not rows_path.is_file():
        raise FileNotFoundError(f"Incomplete holdout result under {run_dir}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("split_sha256") != EXPECTED_SPLIT_SHA256 or metrics.get("holdout_observations") != 142:
        raise RuntimeError(f"Unexpected frozen holdout metadata under {run_dir}")
    rows = load_rows(rows_path)
    if len(rows) != 142 or len({row.file_name for row in rows}) != 142:
        raise RuntimeError(f"Unexpected holdout observation rows under {run_dir}")
    return metrics, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "model-experiments" / "native-unet-grouped-20261111",
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "model-experiments" / "native-unet-gong-aux-20261112",
    )
    parser.add_argument("--resamples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.resamples < 100:
        parser.error("--resamples must be at least 100")
    baseline_dir = args.baseline_dir.expanduser().resolve()
    candidate_dir = args.candidate_dir.expanduser().resolve()
    for path in (baseline_dir, candidate_dir):
        if not path.is_relative_to((REPO_ROOT / "artifacts" / "model-experiments").resolve()):
            raise ValueError("Holdout run directories must remain under artifacts/model-experiments")
    baseline, baseline_rows = load_run(baseline_dir)
    candidate, candidate_rows = load_run(candidate_dir)
    baseline_rows.sort(key=lambda row: row.file_name)
    candidate_rows.sort(key=lambda row: row.file_name)
    if [row.file_name for row in baseline_rows] != [row.file_name for row in candidate_rows]:
        raise RuntimeError("Native and GONG holdout observation lists differ")

    per_observation_path = candidate_dir / "holdout-comparison-per-observation.csv"
    with per_observation_path.open("w", encoding="utf-8", newline="") as stream:
        fields = [
            "file_name",
            "native_pq",
            "native_tp",
            "native_fp",
            "native_fn",
            "gong_pq",
            "gong_tp",
            "gong_fp",
            "gong_fn",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for native, gong in zip(baseline_rows, candidate_rows, strict=True):
            writer.writerow(
                {
                    "file_name": native.file_name,
                    "native_pq": f"{native.pq:.10f}",
                    "native_tp": native.tp,
                    "native_fp": native.fp,
                    "native_fn": native.fn,
                    "gong_pq": f"{gong.pq:.10f}",
                    "gong_tp": gong.tp,
                    "gong_fp": gong.fp,
                    "gong_fn": gong.fn,
                }
            )

    comparison = {
        "split_sha256": EXPECTED_SPLIT_SHA256,
        "partition": "frozen observation holdout; competition annotations only",
        "observations": 142,
        "baseline": {
            "run_dir": str(baseline_dir.relative_to(REPO_ROOT)),
            "global_metrics": baseline["official_scorer_metrics"]["global"],
            "selected_oof_parameters": baseline["postprocessing_selected_from_oof"],
            "selected_oof_epoch": baseline["model_epoch_selected_from_oof"],
        },
        "gong_auxiliary": {
            "run_dir": str(candidate_dir.relative_to(REPO_ROOT)),
            "global_metrics": candidate["official_scorer_metrics"]["global"],
            "selected_oof_parameters": candidate["postprocessing_selected_from_oof"],
            "selected_oof_epoch": candidate["model_epoch_selected_from_oof"],
            "gong_target_manifest_sha256": candidate.get("gong_target_manifest_sha256"),
            "holdout_gong_targets_downloaded": 0,
        },
        "paired_bootstrap_candidate_vs_baseline": paired_bootstrap(
            baseline_rows, candidate_rows, resamples=args.resamples, seed=args.seed
        ),
        "per_observation_csv": str(per_observation_path.relative_to(REPO_ROOT)),
        "test_data_used": False,
    }
    output_path = candidate_dir / "holdout-comparison-vs-native.json"
    output_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path.relative_to(REPO_ROOT)),
                "baseline_pq": comparison["baseline"]["global_metrics"]["pq"],
                "gong_pq": comparison["gong_auxiliary"]["global_metrics"]["pq"],
                **comparison["paired_bootstrap_candidate_vs_baseline"],
            },
            flush=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
