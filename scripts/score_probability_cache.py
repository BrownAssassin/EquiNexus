# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from filament_baseline.data import load_dataset  # noqa: E402
from filament_baseline.inference import InstanceConfig, instances_from_probabilities  # noqa: E402
from filament_baseline.runner import _write_observation_scores  # noqa: E402
from filament_baseline.scorer import score_validation  # noqa: E402
from filament_baseline.split import create_observation_split  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score cached holdout probability maps with one fixed postprocessing configuration."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--probability-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--observation-csv", type=Path, default=None)
    parser.add_argument("--foreground-threshold", type=float, default=0.65)
    parser.add_argument("--boundary-threshold", type=float, default=0.70)
    parser.add_argument("--boundary-weight", type=float, default=1.5)
    parser.add_argument("--seed-threshold", type=float, default=0.45)
    parser.add_argument("--seed-min-area", type=int, default=8)
    parser.add_argument("--min-instance-area", type=int, default=400)
    parser.add_argument("--solar-radius", type=int, default=950)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    data_root = (
        args.data_root.resolve()
        if args.data_root is not None
        else repo_root / "data/raw/MAGFiLO_1.0_Kaggle_2026"
    )
    probability_dir = args.probability_dir.resolve()
    for path in (data_root, probability_dir):
        if not path.is_relative_to(repo_root):
            parser.error(f"Local datasets and artifacts must stay inside the repository: {path}")
    config = json.loads(
        (repo_root / "configs" / "validation-baseline.json").read_text(encoding="utf-8")
    )
    dataset = load_dataset(data_root)
    split = create_observation_split(
        dataset,
        repo_root / "data/splits",
        seed=int(config["seed"]),
        validation_fraction=float(config["validation_fraction"]),
    )
    if split.sha256 != config["expected_split_sha256"]:
        raise RuntimeError("The fixed probability-cache scorer requires the frozen holdout")

    instance_config = InstanceConfig(
        foreground_threshold=args.foreground_threshold,
        boundary_threshold=args.boundary_threshold,
        boundary_weight=args.boundary_weight,
        seed_threshold=args.seed_threshold,
        seed_min_area=args.seed_min_area,
        min_instance_area=args.min_instance_area,
        solar_radius=args.solar_radius,
    )
    predictions = {}
    for index, file_name in enumerate(split.validation_names, start=1):
        cache_path = probability_dir / f"{Path(file_name).stem}.npz"
        if not cache_path.is_file():
            raise FileNotFoundError(f"Missing probability cache: {cache_path}")
        import numpy as np

        with np.load(cache_path) as cache:
            predictions[file_name] = instances_from_probabilities(
                cache["foreground"].astype(np.float32),
                cache["boundary"].astype(np.float32),
                file_name,
                instance_config,
            )
        if index % 25 == 0 or index == len(split.validation_names):
            print(f"Scored {index}/{len(split.validation_names)} observations", flush=True)

    started = time.perf_counter()
    metrics, observation_rows = score_validation(
        dataset,
        predictions,
        split.validation_names,
        iou_threshold=float(config["iou_threshold"]),
        diagnostic_overlap_threshold=float(config["diagnostic_overlap_threshold"]),
    )
    result = {
        "split_sha256": split.sha256,
        "probability_dir": str(probability_dir.relative_to(repo_root)),
        "configuration": instance_config.__dict__,
        "validation_observations": len(split.validation_names),
        "metrics": metrics,
        "scoring_seconds": time.perf_counter() - started,
    }
    output_json = args.output_json.resolve() if args.output_json else probability_dir.parent / "fixed-score.json"
    if not output_json.is_relative_to(repo_root):
        parser.error(f"Score output must stay inside the repository: {output_json}")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.observation_csv is not None:
        observation_csv = args.observation_csv.expanduser().resolve()
        if not observation_csv.is_relative_to(repo_root):
            parser.error(f"Per-observation output must stay inside the repository: {observation_csv}")
        _write_observation_scores(observation_csv, observation_rows, predictions)
        result["observation_metrics"] = str(observation_csv.relative_to(repo_root))
        output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics["global"], indent=2, sort_keys=True))
    print(f"Metrics written to {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
