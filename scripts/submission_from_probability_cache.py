# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from filament_baseline.data import load_dataset  # noqa: E402
from filament_baseline.inference import InstanceConfig, instances_from_probabilities  # noqa: E402
from filament_baseline.runner import _write_predictions  # noqa: E402
from filament_baseline.split import create_observation_split  # noqa: E402
from filament_baseline.submission import _check_submission  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_submission_from_cache(
    *,
    repo_root: str | Path,
    data_root: str | Path,
    probability_dir: str | Path,
    run_name: str,
    instance_config: InstanceConfig,
) -> Path:
    repo_root = Path(repo_root).resolve()
    data_root = Path(data_root).resolve()
    probability_dir = Path(probability_dir).resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", run_name):
        raise ValueError("run_name must use only letters, digits, dot, underscore, or dash")
    for path in (data_root, probability_dir):
        if not path.is_relative_to(repo_root):
            raise ValueError(f"Local datasets and probability maps must stay in the repository: {path}")

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
        raise RuntimeError("The frozen observation split changed")
    test_names = sorted(
        path.name for path in dataset.test_image_dir.iterdir() if path.is_file()
    )
    if len(test_names) != 180:
        raise ValueError(f"Expected 180 supplied test images, found {len(test_names)}")
    missing = [
        file_name
        for file_name in test_names
        if not (probability_dir / f"{Path(file_name).stem}.npz").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing test probability maps for {len(missing)} images; first: {missing[:3]}"
        )

    predictions = {}
    for index, file_name in enumerate(test_names, start=1):
        with np.load(probability_dir / f"{Path(file_name).stem}.npz") as cache:
            foreground = cache["foreground"].astype(np.float32)
            boundary = cache["boundary"].astype(np.float32)
        if foreground.shape != (2048, 2048) or boundary.shape != (2048, 2048):
            raise ValueError(f"Unexpected probability-map dimensions for {file_name}")
        predictions[file_name] = instances_from_probabilities(
            foreground, boundary, file_name, instance_config
        )
        if index % 25 == 0 or index == len(test_names):
            print(f"Converted {index}/{len(test_names)} test images", flush=True)

    output_root = repo_root / "artifacts" / "submissions" / run_name
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Submission output already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    submission_path = output_root / "submission.csv"
    _write_predictions(submission_path, predictions)
    validation = _check_submission(
        submission_path, dataset.root, test_names, predictions
    )
    blend_provenance_path = probability_dir.parent.parent / "blend-provenance.json"
    blend_provenance = (
        json.loads(blend_provenance_path.read_text(encoding="utf-8"))
        if blend_provenance_path.is_file()
        else None
    )
    source_commit = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    metadata = {
        "run_name": run_name,
        "source_commit": source_commit,
        "split_sha256": split.sha256,
        "probability_directory": str(probability_dir.relative_to(repo_root)),
        "blend_provenance": blend_provenance,
        "observation_set": "supplied test H-alpha images",
        "test_ground_truth_used": False,
        "training_labels": "competition-provided annotations only",
        "instance_configuration": instance_config.__dict__,
        "submission_sha256": _sha256(submission_path),
        "submission_validation": validation,
    }
    (output_root / "run-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    return submission_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create and validate a submission from cached test probability maps."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026",
    )
    parser.add_argument("--probability-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--foreground-threshold", type=float, required=True)
    parser.add_argument("--boundary-threshold", type=float, required=True)
    parser.add_argument("--boundary-weight", type=float, default=1.5)
    parser.add_argument("--seed-threshold", type=float, default=0.45)
    parser.add_argument("--seed-min-area", type=int, default=8)
    parser.add_argument("--min-instance-area", type=int, required=True)
    parser.add_argument("--solar-radius", type=int, default=950)
    args = parser.parse_args()
    instance_config = InstanceConfig(
        foreground_threshold=args.foreground_threshold,
        boundary_threshold=args.boundary_threshold,
        boundary_weight=args.boundary_weight,
        seed_threshold=args.seed_threshold,
        seed_min_area=args.seed_min_area,
        min_instance_area=args.min_instance_area,
        solar_radius=args.solar_radius,
    )
    create_submission_from_cache(
        repo_root=args.repo_root,
        data_root=args.data_root,
        probability_dir=args.probability_dir,
        run_name=args.run_name,
        instance_config=instance_config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
