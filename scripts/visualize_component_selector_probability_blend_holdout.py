# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Render fixed paired holdout examples for the OOF-selected selector blend."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from pycocotools import mask as mask_utils

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_component_selector_oof import load_prediction_rows  # noqa: E402
from filament_baseline.data import load_dataset  # noqa: E402
from filament_baseline.predictor import PredictedInstance  # noqa: E402
from filament_baseline.split import create_observation_split  # noqa: E402
from filament_baseline.visuals import write_comparison_overlays  # noqa: E402


FROZEN_SPLIT_SHA256 = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"
DEFAULT_RUN = REPO_ROOT / "artifacts/model-experiments/component-selector-probability-blend-holdout-20261109"


def as_instances(file_name: str, rles: list[dict[str, Any]]) -> list[PredictedInstance]:
    stem = Path(file_name).stem
    instances = []
    for index, rle in enumerate(rles, start=1):
        area = int(mask_utils.area(rle))
        if area <= 0:
            raise ValueError(f"Empty mask for {stem}_{index}")
        instances.append(
            PredictedInstance(
                filament_id=f"{stem}_{index}",
                image_stem=stem,
                area=area,
                rle=rle,
            )
        )
    return instances


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    run_dir = args.run_dir.resolve()
    if not data_root.is_relative_to((REPO_ROOT / "data").resolve()):
        raise ValueError("Competition data must stay under data/")
    if not run_dir.is_relative_to((REPO_ROOT / "artifacts/model-experiments").resolve()):
        raise ValueError("Run artifacts must stay under artifacts/model-experiments/")

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    dataset = load_dataset(data_root)
    split = create_observation_split(dataset, REPO_ROOT / "data/splits")
    if split.sha256 != FROZEN_SPLIT_SHA256 or metrics.get("split_sha256") != split.sha256:
        raise RuntimeError("The visual comparison must use the frozen holdout")

    baseline = pd.read_csv(run_dir / "baseline-per-observation.csv")
    candidate = pd.read_csv(run_dir / "candidate-per-observation.csv")
    joined = baseline.merge(candidate, on="file_name", suffixes=("_baseline", "_candidate"), validate="one_to_one")
    joined["delta_pq"] = joined["pq_candidate"] - joined["pq_baseline"]
    if set(joined.file_name) != set(split.validation_names):
        raise ValueError("Paired scores do not cover exactly the frozen holdout observations")

    ordered = joined.sort_values(["delta_pq", "file_name"], kind="mergesort")
    candidates = [
        ordered.iloc[-1],
        ordered.iloc[(len(ordered) - 1) // 2],
        ordered.iloc[0],
    ]
    selected = []
    seen: set[str] = set()
    for row in candidates:
        file_name = str(row["file_name"])
        if file_name not in seen:
            seen.add(file_name)
            selected.append(row)
    selected_names = [str(row["file_name"]) for row in selected]
    all_holdout_stems = {Path(name).stem for name in split.validation_names}
    selected_stems = {Path(name).stem for name in selected_names}
    baseline_all = load_prediction_rows(run_dir / "baseline-predictions.csv", all_holdout_stems)
    candidate_all = load_prediction_rows(run_dir / "candidate-predictions.csv", all_holdout_stems)
    baseline_by_stem = {stem: baseline_all[stem] for stem in selected_stems}
    candidate_by_stem = {stem: candidate_all[stem] for stem in selected_stems}
    baseline_predictions = {
        name: as_instances(name, baseline_by_stem[Path(name).stem]) for name in selected_names
    }
    candidate_predictions = {
        name: as_instances(name, candidate_by_stem[Path(name).stem]) for name in selected_names
    }

    overlay_root = run_dir / "overlays"
    overlay_root.mkdir(parents=True, exist_ok=True)
    write_comparison_overlays(dataset, baseline_predictions, overlay_root / "baseline", maximum=3)
    write_comparison_overlays(dataset, candidate_predictions, overlay_root / "candidate", maximum=3)
    with (run_dir / "selected-visual-observations.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "file_name", "delta_pq", "baseline_pq", "candidate_pq",
                "baseline_tp", "baseline_fp", "baseline_fn",
                "candidate_tp", "candidate_fp", "candidate_fn",
            ],
        )
        writer.writeheader()
        for row in selected:
            writer.writerow({
                "file_name": row["file_name"],
                "delta_pq": row["delta_pq"],
                "baseline_pq": row["pq_baseline"],
                "candidate_pq": row["pq_candidate"],
                "baseline_tp": row["tp_edges_baseline"],
                "baseline_fp": row["fp_baseline"],
                "baseline_fn": row["fn_baseline"],
                "candidate_tp": row["tp_edges_candidate"],
                "candidate_fp": row["fp_candidate"],
                "candidate_fn": row["fn_candidate"],
            })

    print(json.dumps({
        "split_sha256": split.sha256,
        "selected_observations": [
            {
                "file_name": str(row["file_name"]),
                "delta_pq": float(row["delta_pq"]),
                "baseline_pq": float(row["pq_baseline"]),
                "candidate_pq": float(row["pq_candidate"]),
            }
            for row in selected
        ],
        "overlay_root": str(overlay_root),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
