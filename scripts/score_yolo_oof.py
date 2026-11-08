# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from filament_baseline.data import load_dataset
from filament_baseline.split import create_observation_split
from score_detector_json import score_detector_predictions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score cross-fitted YOLO detections on the frozen training partition."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--oof-manifest",
        type=Path,
        default=REPO_ROOT / "data" / "splits" / "observation_oof_seed42_5fold.csv",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence-thresholds", default="0.10,0.15,0.20,0.25,0.30,0.40")
    parser.add_argument("--minimum-areas", default="0,100,200,400,600")
    args = parser.parse_args()

    repo_root = args.repo_root.expanduser().resolve()
    data_root = (
        args.data_root.expanduser().resolve()
        if args.data_root
        else repo_root / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026"
    )
    predictions_path = args.predictions.expanduser().resolve()
    manifest_path = args.oof_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for label, path, parent in (
        ("data", data_root, repo_root / "data"),
        ("predictions", predictions_path, repo_root / "artifacts"),
        ("OOF manifest", manifest_path, repo_root / "data" / "splits"),
        ("output", output_dir, repo_root / "artifacts"),
    ):
        if not path.is_relative_to(parent):
            raise ValueError(f"{label} must remain under {parent}: {path}")
    if not predictions_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Missing OOF detections or observation manifest")

    config = json.loads((repo_root / "configs" / "validation-baseline.json").read_text())
    dataset = load_dataset(data_root)
    frozen = create_observation_split(
        dataset,
        repo_root / "data" / "splits",
        seed=int(config["seed"]),
        validation_fraction=float(config["validation_fraction"]),
    )
    if frozen.sha256 != config["expected_split_sha256"]:
        raise RuntimeError(f"Frozen holdout digest changed: {frozen.sha256}")

    with manifest_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"file_name", "year", "site", "oof_fold"}.issubset(rows[0]):
        raise ValueError("OOF manifest must have file_name, year, site, and oof_fold columns")
    names = tuple(sorted(row["file_name"] for row in rows))
    if len(names) != len(set(names)):
        raise ValueError("OOF manifest contains duplicate observation names")
    if set(names) != set(frozen.training_names):
        raise ValueError("OOF observations must cover exactly the frozen training partition")
    if set(names) & set(frozen.validation_names):
        raise ValueError("Frozen holdout observations cannot be scored as training OOF")
    for row in rows:
        view = dataset.views_by_observation[row["file_name"]][0]
        if int(row["year"]) != view.year or row["site"] != view.site:
            raise ValueError(f"Observation metadata mismatch: {row['file_name']}")

    predictions: dict[str, list[dict[str, Any]]] = json.loads(
        predictions_path.read_text(encoding="utf-8")
    )
    expected_stems = {Path(name).stem for name in names}
    if set(predictions) != expected_stems:
        raise ValueError(
            f"OOF detections cover {len(predictions)} observations; expected {len(expected_stems)}"
        )
    confidence_thresholds = [float(v) for v in args.confidence_thresholds.split(",")]
    minimum_areas = [int(v) for v in args.minimum_areas.split(",")]
    if any(not 0 <= value <= 1 for value in confidence_thresholds):
        raise ValueError("Confidence thresholds must be in [0, 1]")
    if any(value < 0 for value in minimum_areas):
        raise ValueError("Minimum areas must be nonnegative")

    output_dir.mkdir(parents=True, exist_ok=True)
    sweep = []
    best = None
    best_rows = None
    for confidence in confidence_thresholds:
        for minimum_area in minimum_areas:
            metrics, per_observation, prediction_count = score_detector_predictions(
                dataset,
                names,
                predictions,
                confidence_threshold=confidence,
                minimum_area=minimum_area,
            )
            result = {
                "confidence_threshold": confidence,
                "minimum_area": minimum_area,
                "prediction_count": prediction_count,
                "metrics": metrics,
            }
            sweep.append(result)
            print(
                f"confidence={confidence:.2f} min_area={minimum_area} "
                f"PQ={metrics['global']['pq']:.6f} "
                f"TP={metrics['global']['tp']} FP={metrics['global']['fp']} "
                f"FN={metrics['global']['fn']}",
                flush=True,
            )
            if best is None or metrics["global"]["pq"] > best["metrics"]["global"]["pq"]:
                best = result
                best_rows = per_observation

    if best is None or best_rows is None:
        raise RuntimeError("No OOF scoring configuration was evaluated")
    metadata = {
        "split_sha256": frozen.sha256,
        "oof_manifest": str(manifest_path.relative_to(repo_root)),
        "oof_manifest_sha256": _sha256(manifest_path),
        "oof_observations": len(names),
        "oof_annotator_views": sum(len(dataset.views_by_observation[name]) for name in names),
        "ground_truth_source": "competition-provided training annotations only",
        "prediction_source": str(predictions_path.relative_to(repo_root)),
        "prediction_sha256": _sha256(predictions_path),
        "best": best,
        "sweep": sweep,
    }
    (output_dir / "oof-score.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if best_rows:
        serializable_rows = [asdict(row) for row in best_rows]
        with (output_dir / "oof-per-observation.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(serializable_rows[0].keys()))
            writer.writeheader()
            writer.writerows(serializable_rows)
    print(
        f"Best OOF PQ={best['metrics']['global']['pq']:.6f}; output={output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
