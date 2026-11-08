# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from filament_baseline.data import load_dataset  # noqa: E402
from filament_baseline.split import create_observation_split  # noqa: E402


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _split_digest(metrics: dict) -> str:
    split = metrics.get("split", {})
    return str(split.get("sha256") or metrics.get("split_sha256") or "")


def _metrics_path(run_root: Path) -> Path:
    candidates = (
        run_root / "metrics.json",
        run_root / "best-postprocessing" / "metrics.json",
        run_root / "best-instance-structure" / "metrics.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No holdout metrics file found under {run_root}")


def blend_caches(
    *,
    repo_root: Path,
    data_root: Path,
    first_root: Path,
    second_root: Path,
    first_cache_dir: Path | None,
    second_cache_dir: Path | None,
    first_weight: float,
    run_name: str,
    observation_set: str = "validation",
) -> Path:
    repo_root = repo_root.resolve()
    data_root = data_root.resolve()
    first_root = first_root.resolve()
    second_root = second_root.resolve()
    if not 0.0 <= first_weight <= 1.0:
        raise ValueError("first_weight must be between zero and one")
    if observation_set not in {"validation", "test"}:
        raise ValueError("observation_set must be validation or test")
    first_cache_dir = (first_cache_dir or first_root / "probabilities").resolve()
    second_cache_dir = (second_cache_dir or second_root / "probabilities").resolve()
    for path in (data_root, first_root, second_root, first_cache_dir, second_cache_dir):
        if not path.is_relative_to(repo_root):
            raise ValueError(f"Local datasets and model artifacts must stay inside the repository: {path}")
    if not run_name.replace("-", "").replace("_", "").isalnum():
        raise ValueError("run_name must contain only letters, digits, dashes, and underscores")

    config = json.loads((repo_root / "configs" / "validation-baseline.json").read_text(encoding="utf-8"))
    dataset = load_dataset(data_root)
    split = create_observation_split(
        dataset,
        repo_root / "data" / "splits",
        seed=int(config["seed"]),
        validation_fraction=float(config["validation_fraction"]),
    )
    if split.sha256 != config["expected_split_sha256"]:
        raise RuntimeError("The frozen validation split digest changed")
    observation_names = (
        split.validation_names
        if observation_set == "validation"
        else tuple(sorted(path.name for path in dataset.test_image_dir.iterdir() if path.is_file()))
    )
    if not observation_names:
        raise ValueError(f"No observations found for {observation_set}")

    first_metrics_path = _metrics_path(first_root)
    second_metrics_path = _metrics_path(second_root)
    first_metrics = json.loads(first_metrics_path.read_text(encoding="utf-8"))
    second_metrics = json.loads(second_metrics_path.read_text(encoding="utf-8"))
    if _split_digest(first_metrics) != split.sha256 or _split_digest(second_metrics) != split.sha256:
        raise RuntimeError("Both source runs must use the frozen holdout split")
    first_cache = first_cache_dir
    second_cache = second_cache_dir
    output_root = repo_root / "artifacts" / "model-experiments" / run_name
    output_dir = output_root / "probabilities"
    if observation_set == "test":
        output_dir /= "test"
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Blend output already exists: {output_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, file_name in enumerate(observation_names, start=1):
        file_stem = Path(file_name).stem
        with np.load(first_cache / f"{file_stem}.npz") as first, np.load(
            second_cache / f"{file_stem}.npz"
        ) as second:
            first_fg = first["foreground"].astype(np.float32)
            second_fg = second["foreground"].astype(np.float32)
            first_boundary = first["boundary"].astype(np.float32)
            second_boundary = second["boundary"].astype(np.float32)
        if not (
            first_fg.shape == second_fg.shape == first_boundary.shape == second_boundary.shape
        ):
            raise ValueError(f"Probability map dimensions differ for {file_name}")
        if first_fg.shape != (2048, 2048):
            raise ValueError(f"Unexpected probability-map dimensions for {file_name}: {first_fg.shape}")
        blended_fg = first_weight * first_fg + (1.0 - first_weight) * second_fg
        blended_boundary = first_weight * first_boundary + (1.0 - first_weight) * second_boundary
        np.savez_compressed(
            output_dir / f"{file_stem}.npz",
            foreground=blended_fg.astype(np.float16),
            boundary=blended_boundary.astype(np.float16),
        )
        if index % 25 == 0 or index == len(observation_names):
            print(f"Blended {index}/{len(observation_names)} observations", flush=True)

    first_checkpoint = first_root / "best_model.pt"
    second_checkpoint = second_root / "best_model.pt"
    provenance = {
        "run_name": run_name,
        "split_sha256": split.sha256,
        "first_model_run": str(first_root.relative_to(repo_root)),
        "first_metrics_path": str(first_metrics_path.relative_to(first_root)),
        "first_probability_cache": str(first_cache.relative_to(repo_root)),
        "first_model_metrics_sha256": _digest(first_metrics_path),
        "first_checkpoint_sha256": _digest(first_checkpoint) if first_checkpoint.is_file() else None,
        "second_model_run": str(second_root.relative_to(repo_root)),
        "second_metrics_path": str(second_metrics_path.relative_to(second_root)),
        "second_probability_cache": str(second_cache.relative_to(repo_root)),
        "second_model_metrics_sha256": _digest(second_metrics_path),
        "second_checkpoint_sha256": _digest(second_checkpoint) if second_checkpoint.is_file() else None,
        "probability_weights": {
            "first_model": first_weight,
            "second_model": 1.0 - first_weight,
        },
        "observation_set": observation_set,
        "observations": len(observation_names),
        "training_labels": "competition-provided annotations only",
        "test_images_used_for_inference": observation_set == "test",
        "test_labels_used": False,
    }
    (output_root / "blend-provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2, sort_keys=True))
    return output_root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Blend two validation or test probability-map caches."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--data-root", type=Path, default=REPO_ROOT / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026"
    )
    parser.add_argument("--first-root", type=Path, required=True)
    parser.add_argument("--second-root", type=Path, required=True)
    parser.add_argument("--first-cache-dir", type=Path, default=None)
    parser.add_argument("--second-cache-dir", type=Path, default=None)
    parser.add_argument("--first-weight", type=float, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--observation-set", choices=("validation", "test"), default="validation"
    )
    args = parser.parse_args()
    blend_caches(
        repo_root=args.repo_root,
        data_root=args.data_root,
        first_root=args.first_root,
        second_root=args.second_root,
        first_cache_dir=args.first_cache_dir,
        second_cache_dir=args.second_cache_dir,
        first_weight=args.first_weight,
        run_name=args.run_name,
        observation_set=args.observation_set,
    )
