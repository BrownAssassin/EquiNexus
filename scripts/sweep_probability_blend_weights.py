# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from filament_baseline.data import IMAGE_SIZE, load_dataset  # noqa: E402
from filament_baseline.inference import InstanceConfig, instances_from_probabilities  # noqa: E402
from filament_baseline.scorer import score_validation  # noqa: E402
from filament_baseline.split import create_observation_split  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_metrics_path(run_root: Path) -> Path:
    candidates = (
        run_root / "metrics.json",
        run_root / "best-instance-structure" / "metrics.json",
        run_root / "best-postprocessing" / "metrics.json",
        run_root / "fixed-score.json",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"No validation metrics found under {run_root}")


def _split_digest(metrics: dict) -> str | None:
    split = metrics.get("split")
    if isinstance(split, dict):
        return split.get("sha256") or split.get("expected_sha256")
    return metrics.get("split_sha256")


def sweep_weights(
    *,
    repo_root: Path,
    data_root: Path,
    first_root: Path,
    second_root: Path,
    weights: list[float],
    boundary_weights: list[float] | None,
    run_name: str,
    instance_config: InstanceConfig,
) -> dict:
    repo_root = repo_root.resolve()
    data_root = data_root.resolve()
    first_root = first_root.resolve()
    second_root = second_root.resolve()
    for label, path in (
        ("data", data_root),
        ("first run", first_root),
        ("second run", second_root),
    ):
        if not path.is_relative_to(repo_root):
            raise ValueError(f"{label} must stay inside the repository: {path}")
    if not run_name.replace("-", "").replace("_", "").isalnum():
        raise ValueError("run_name must contain only letters, digits, dashes, and underscores")
    if not weights or any(not 0.0 <= weight <= 1.0 for weight in weights):
        raise ValueError("Provide one or more blend weights in [0, 1]")
    if len(set(weights)) != len(weights):
        raise ValueError("Blend weights must be unique")
    boundary_weights = boundary_weights if boundary_weights is not None else weights
    if not boundary_weights or any(not 0.0 <= weight <= 1.0 for weight in boundary_weights):
        raise ValueError("Provide one or more boundary blend weights in [0, 1]")
    if len(set(boundary_weights)) != len(boundary_weights):
        raise ValueError("Boundary blend weights must be unique")

    config = json.loads(
        (repo_root / "configs" / "validation-baseline.json").read_text(encoding="utf-8")
    )
    dataset = load_dataset(data_root)
    split = create_observation_split(
        dataset,
        repo_root / "data" / "splits",
        seed=int(config["seed"]),
        validation_fraction=float(config["validation_fraction"]),
    )
    if split.sha256 != config["expected_split_sha256"]:
        raise RuntimeError("The frozen validation split changed")

    first_metrics_path = _run_metrics_path(first_root)
    second_metrics_path = _run_metrics_path(second_root)
    first_metrics = json.loads(first_metrics_path.read_text(encoding="utf-8"))
    second_metrics = json.loads(second_metrics_path.read_text(encoding="utf-8"))
    for label, source_metrics in (("first", first_metrics), ("second", second_metrics)):
        source_split = _split_digest(source_metrics)
        if source_split != split.sha256:
            raise RuntimeError(f"The {label} source run does not use the frozen holdout")

    first_cache = first_root / "probabilities"
    second_cache = second_root / "probabilities"
    expected_files = {f"{Path(name).stem}.npz" for name in split.validation_names}
    for label, cache_dir in (("first", first_cache), ("second", second_cache)):
        present = {path.name for path in cache_dir.glob("*.npz")}
        if present != expected_files:
            missing = len(expected_files - present)
            extra = len(present - expected_files)
            raise RuntimeError(
                f"The {label} cache must cover exactly the frozen holdout; "
                f"missing={missing}, extra={extra}"
            )

    rows = []
    started = time.perf_counter()
    for foreground_weight in weights:
        for boundary_weight in boundary_weights:
            predictions = {}
            for index, file_name in enumerate(split.validation_names, start=1):
                stem = Path(file_name).stem
                with np.load(first_cache / f"{stem}.npz") as first, np.load(
                    second_cache / f"{stem}.npz"
                ) as second:
                    arrays = (
                        first["foreground"].astype(np.float32),
                        first["boundary"].astype(np.float32),
                        second["foreground"].astype(np.float32),
                        second["boundary"].astype(np.float32),
                    )
                if any(array.shape != (IMAGE_SIZE, IMAGE_SIZE) for array in arrays):
                    raise ValueError(f"Invalid probability-map dimensions for {file_name}")
                if any(not np.isfinite(array).all() for array in arrays):
                    raise ValueError(f"Non-finite probabilities for {file_name}")
                foreground = foreground_weight * arrays[0] + (1.0 - foreground_weight) * arrays[2]
                boundary = boundary_weight * arrays[1] + (1.0 - boundary_weight) * arrays[3]
                predictions[file_name] = instances_from_probabilities(
                    foreground, boundary, file_name, instance_config
                )
                if index % 50 == 0 or index == len(split.validation_names):
                    print(
                        f"foreground_weight={foreground_weight:.3f} "
                        f"boundary_weight={boundary_weight:.3f}: postprocessed "
                        f"{index}/{len(split.validation_names)} observations",
                        flush=True,
                    )
            metrics, _ = score_validation(
                dataset,
                predictions,
                split.validation_names,
                iou_threshold=float(config["iou_threshold"]),
                diagnostic_overlap_threshold=float(config["diagnostic_overlap_threshold"]),
            )
            row = {
                "first_foreground_weight": foreground_weight,
                "second_foreground_weight": 1.0 - foreground_weight,
                "first_boundary_weight": boundary_weight,
                "second_boundary_weight": 1.0 - boundary_weight,
                "configuration": instance_config.__dict__,
                "metrics": metrics["global"],
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    rows.sort(key=lambda row: float(row["metrics"]["pq"]), reverse=True)
    output_root = repo_root / "artifacts" / "model-experiments" / "blend-weight-sweeps" / run_name
    output_root.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "complete",
        "run_name": run_name,
        "split_sha256": split.sha256,
        "validation_observations": len(split.validation_names),
        "source_runs": [
            {
                "root": str(first_root.relative_to(repo_root)),
                "metrics_path": str(first_metrics_path.relative_to(repo_root)),
                "metrics_sha256": _sha256(first_metrics_path),
            },
            {
                "root": str(second_root.relative_to(repo_root)),
                "metrics_path": str(second_metrics_path.relative_to(repo_root)),
                "metrics_sha256": _sha256(second_metrics_path),
            },
        ],
        "configuration": instance_config.__dict__,
        "foreground_weights_tested": weights,
        "boundary_weights_tested": boundary_weights,
        "results_by_pq": rows,
        "runtime_seconds": time.perf_counter() - started,
    }
    (output_root / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"best": rows[0], "output": str(output_root)}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Screen two holdout probability caches with the unchanged official scorer."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--data-root", type=Path, default=REPO_ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026"
    )
    parser.add_argument("--first-root", type=Path, required=True)
    parser.add_argument("--second-root", type=Path, required=True)
    parser.add_argument("--weights", type=float, nargs="+", default=[0.75, 0.80, 0.85, 0.90, 0.95])
    parser.add_argument("--boundary-weights", type=float, nargs="+", default=None)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--foreground-threshold", type=float, default=0.65)
    parser.add_argument("--boundary-threshold", type=float, default=0.70)
    parser.add_argument("--boundary-weight", type=float, default=1.5)
    parser.add_argument("--seed-threshold", type=float, default=0.45)
    parser.add_argument("--seed-min-area", type=int, default=8)
    parser.add_argument("--min-instance-area", type=int, default=500)
    parser.add_argument("--solar-radius", type=int, default=950)
    args = parser.parse_args()
    sweep_weights(
        repo_root=args.repo_root,
        data_root=args.data_root,
        first_root=args.first_root,
        second_root=args.second_root,
        weights=args.weights,
        boundary_weights=args.boundary_weights,
        run_name=args.run_name,
        instance_config=InstanceConfig(
            foreground_threshold=args.foreground_threshold,
            boundary_threshold=args.boundary_threshold,
            boundary_weight=args.boundary_weight,
            seed_threshold=args.seed_threshold,
            seed_min_area=args.seed_min_area,
            min_instance_area=args.min_instance_area,
            solar_radius=args.solar_radius,
        ),
    )
