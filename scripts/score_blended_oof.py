from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from filament_baseline.data import load_dataset  # noqa: E402
from filament_baseline.inference import InstanceConfig, instances_from_probabilities  # noqa: E402
from filament_baseline.predictor import PredictedInstance  # noqa: E402
from filament_baseline.runner import (  # noqa: E402
    _write_observation_scores,
    _write_overlays,
    _write_predictions,
)
from filament_baseline.scorer import score_validation  # noqa: E402
from filament_baseline.split import create_observation_split  # noqa: E402
from filament_baseline.visuals import write_comparison_overlays  # noqa: E402


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_blended_oof(
    *,
    repo_root: Path,
    data_root: Path,
    first_oof_root: Path,
    second_oof_root: Path,
    first_weight: float,
    run_name: str,
    instance_config: InstanceConfig,
    save_probability_cache: bool = False,
) -> dict:
    repo_root = repo_root.resolve()
    data_root = data_root.resolve()
    first_oof_root = first_oof_root.resolve()
    second_oof_root = second_oof_root.resolve()
    for path in (data_root, first_oof_root, second_oof_root):
        if not path.is_relative_to(repo_root):
            raise ValueError(f"Local data and OOF artifacts must stay inside the repository: {path}")
    if not 0.0 <= first_weight <= 1.0:
        raise ValueError("first_weight must be between zero and one")

    config_path = repo_root / "configs" / "validation-baseline.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset = load_dataset(data_root)
    split = create_observation_split(
        dataset,
        repo_root / "data" / "splits",
        seed=int(config["seed"]),
        validation_fraction=float(config["validation_fraction"]),
    )
    if split.sha256 != config["expected_split_sha256"]:
        raise RuntimeError("The frozen validation split changed")
    frozen_training = set(split.training_names)
    oof_manifest_path = repo_root / "data" / "splits" / f"observation_oof_seed{config['seed']}_5fold.csv"
    if not oof_manifest_path.is_file():
        raise FileNotFoundError(f"Missing grouped OOF manifest: {oof_manifest_path}")
    with oof_manifest_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    fold_by_name = {row["file_name"]: int(row["oof_fold"]) for row in rows}
    if len(fold_by_name) != len(rows) or set(fold_by_name) != frozen_training:
        raise RuntimeError("The OOF manifest must cover every training-partition observation exactly once")
    if set(fold_by_name) & set(split.validation_names):
        raise RuntimeError("The frozen 142-observation holdout cannot enter the OOF blend")
    oof_manifest_sha256 = _digest(oof_manifest_path)

    metrics_paths = (first_oof_root / "metrics.json", second_oof_root / "metrics.json")
    source_metrics = [json.loads(path.read_text(encoding="utf-8")) for path in metrics_paths]
    for source in source_metrics:
        source_splits = source.get("splits", {})
        if source_splits.get("frozen_holdout_sha256") != split.sha256:
            raise RuntimeError("Both OOF source models must exclude the frozen holdout")
        if source_splits.get("oof_manifest_sha256") != oof_manifest_sha256:
            raise RuntimeError("Both OOF models must use the same five-fold observation manifest")

    output_root = repo_root / "artifacts" / "oof-experiments" / run_name
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"OOF blend output already exists: {output_root}")
    probability_dir = output_root / "probabilities"
    if save_probability_cache:
        probability_dir.mkdir(parents=True, exist_ok=True)
    predictions: dict[str, list[PredictedInstance]] = {}
    started = time.perf_counter()
    for index, file_name in enumerate(sorted(frozen_training), start=1):
        fold = fold_by_name[file_name]
        file_stem = Path(file_name).stem
        with np.load(first_oof_root / f"fold-{fold}" / "probabilities" / f"{file_stem}.npz") as first, np.load(
            second_oof_root / f"fold-{fold}" / "probabilities" / f"{file_stem}.npz"
        ) as second:
            first_fg = first["foreground"].astype(np.float32)
            first_boundary = first["boundary"].astype(np.float32)
            second_fg = second["foreground"].astype(np.float32)
            second_boundary = second["boundary"].astype(np.float32)
        if not (
            first_fg.shape == first_boundary.shape == second_fg.shape == second_boundary.shape == (2048, 2048)
        ):
            raise ValueError(f"Invalid OOF probability dimensions for {file_name}")
        foreground = first_weight * first_fg + (1.0 - first_weight) * second_fg
        boundary = first_weight * first_boundary + (1.0 - first_weight) * second_boundary
        if save_probability_cache:
            np.savez_compressed(
                probability_dir / f"{file_stem}.npz",
                foreground=foreground.astype(np.float16),
                boundary=boundary.astype(np.float16),
            )
        predictions[file_name] = instances_from_probabilities(
            foreground, boundary, file_name, instance_config
        )
        if index % 25 == 0 or index == len(frozen_training):
            print(f"Blended {index}/{len(frozen_training)} OOF observations", flush=True)

    metrics, observation_rows = score_validation(
        dataset,
        predictions,
        sorted(frozen_training),
        iou_threshold=float(config["iou_threshold"]),
        diagnostic_overlap_threshold=float(config["diagnostic_overlap_threshold"]),
    )
    _write_predictions(output_root / "oof-predictions.csv", predictions)
    _write_observation_scores(output_root / "oof-per-observation.csv", observation_rows, predictions)
    _write_overlays(dataset, predictions, output_root / "overlays", 12)
    comparison_overlays = write_comparison_overlays(
        dataset, predictions, output_root / "comparison-overlays", 12
    )
    result = {
        "status": "complete",
        "run_name": run_name,
        "split_sha256": split.sha256,
        "oof_manifest_sha256": oof_manifest_sha256,
        "oof_observations": len(frozen_training),
        "configuration": {
            "first_weight": first_weight,
            "second_weight": 1.0 - first_weight,
            "instance": instance_config.__dict__,
            "saved_probability_cache": save_probability_cache,
        },
        "source_runs": [
            {
                "root": str(root.relative_to(repo_root)),
                "metrics_sha256": _digest(metrics_path),
            }
            for root, metrics_path in zip((first_oof_root, second_oof_root), metrics_paths, strict=True)
        ],
        "metrics": metrics,
        "artifacts": {
            "predictions": "oof-predictions.csv",
            "per_observation": "oof-per-observation.csv",
            "comparison_overlays": comparison_overlays,
        },
        "runtime": {"scoring_seconds": time.perf_counter() - started},
    }
    (output_root / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"aggregate_oof": metrics["global"], "run_name": run_name}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score a probability blend over grouped training-partition OOF maps.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--first-oof-root", type=Path, required=True)
    parser.add_argument("--second-oof-root", type=Path, required=True)
    parser.add_argument("--first-weight", type=float, default=0.5)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--foreground-threshold", type=float, required=True)
    parser.add_argument("--boundary-threshold", type=float, required=True)
    parser.add_argument("--boundary-weight", type=float, required=True)
    parser.add_argument("--seed-threshold", type=float, required=True)
    parser.add_argument("--seed-min-area", type=int, required=True)
    parser.add_argument("--min-instance-area", type=int, required=True)
    parser.add_argument("--save-probability-cache", action="store_true")
    args = parser.parse_args()
    run_blended_oof(
        repo_root=args.repo_root,
        data_root=args.data_root,
        first_oof_root=args.first_oof_root,
        second_oof_root=args.second_oof_root,
        first_weight=args.first_weight,
        run_name=args.run_name,
        instance_config=InstanceConfig(
            foreground_threshold=args.foreground_threshold,
            boundary_threshold=args.boundary_threshold,
            boundary_weight=args.boundary_weight,
            seed_threshold=args.seed_threshold,
            seed_min_area=args.seed_min_area,
            min_instance_area=args.min_instance_area,
        ),
        save_probability_cache=args.save_probability_cache,
    )
