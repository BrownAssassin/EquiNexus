# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import fields, replace
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from filament_baseline.data import load_dataset  # noqa: E402
from filament_baseline.predictor import read_grayscale  # noqa: E402
from filament_baseline.runner import _write_observation_scores, _write_predictions  # noqa: E402
from filament_baseline.runtime import hardware_info, package_versions  # noqa: E402
from filament_baseline.scorer import score_validation  # noqa: E402
from filament_baseline.seed_refiner import (  # noqa: E402
    SeedRefinerConfig,
    load_seed_refiner,
    predicted_instances,
)
from filament_baseline.split import create_observation_split  # noqa: E402
from filament_baseline.visuals import write_comparison_overlays  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Screen crop-refiner settings on the frozen holdout.")
    parser.add_argument(
        "--run-name",
        default="2026-11-08-seedrefiner-yolo11s-oof-scratch",
        help="Existing model-experiment directory under artifacts/model-experiments.",
    )
    args = parser.parse_args()
    run_dir = REPO_ROOT / "artifacts" / "model-experiments" / args.run_name
    metadata = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    baseline = json.loads((REPO_ROOT / "configs" / "validation-baseline.json").read_text(encoding="utf-8"))
    dataset = load_dataset(REPO_ROOT / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026")
    split = create_observation_split(
        dataset,
        REPO_ROOT / "data" / "splits",
        seed=int(baseline["seed"]),
        validation_fraction=float(baseline["validation_fraction"]),
    )
    if split.sha256 != baseline["expected_split_sha256"] or split.sha256 != metadata["split_sha256"]:
        raise RuntimeError("Refiner sweep requires the unchanged frozen observation split")

    validation_path = REPO_ROOT / metadata["validation_detection_source"]
    validation_detections = json.loads(validation_path.read_text(encoding="utf-8"))
    if not {Path(name).stem for name in split.validation_names}.issubset(validation_detections):
        raise ValueError("Validation detections do not cover the frozen holdout")

    allowed = {field.name for field in fields(SeedRefinerConfig)}
    base_config = SeedRefinerConfig(
        **{key: value for key, value in metadata["configuration"].items() if key in allowed}
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, _checkpoint = load_seed_refiner(run_dir / "seed_refiner.pt", device)

    # This fixed, modest grid screens detector confidence, crop-mask threshold,
    # and minimum output area on the holdout. Any promotion requires grouped OOF.
    candidates = [
        (0.35, 0.50, 200),
        (0.45, 0.50, 200),
        (0.55, 0.50, 200),
        (0.65, 0.50, 200),
        (0.25, 0.40, 200),
        (0.25, 0.60, 200),
        (0.25, 0.70, 200),
        (0.45, 0.40, 200),
        (0.45, 0.60, 200),
        (0.45, 0.50, 50),
        (0.45, 0.50, 400),
        (0.45, 0.50, 800),
    ]
    output_dir = run_dir / "threshold-sweep"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    best = None
    started = time.perf_counter()
    for index, (score_threshold, probability_threshold, minimum_area) in enumerate(candidates, start=1):
        config = replace(
            base_config,
            validation_score_min=score_threshold,
            probability_threshold=probability_threshold,
            minimum_instance_area=minimum_area,
        )
        predictions = {}
        for file_name in split.validation_names:
            image = read_grayscale(dataset.train_image_dir / file_name)
            predictions[file_name] = predicted_instances(
                file_name,
                image,
                validation_detections.get(Path(file_name).stem, []),
                model,
                config,
                device=device,
            )
        metrics, observation_rows = score_validation(
            dataset,
            predictions,
            split.validation_names,
            iou_threshold=float(baseline["iou_threshold"]),
            diagnostic_overlap_threshold=float(baseline["diagnostic_overlap_threshold"]),
        )
        result = {
            "configuration": {
                "validation_score_min": score_threshold,
                "probability_threshold": probability_threshold,
                "minimum_instance_area": minimum_area,
            },
            "scorer_metrics": metrics,
            "global": metrics["global"],
            "by_site": metrics["by_site"],
            "observation_count": len(observation_rows),
        }
        candidate_dir = output_dir / f"candidate-{index:02d}"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        _write_observation_scores(
            candidate_dir / "per_observation.csv", observation_rows, predictions
        )
        results.append(result)
        if best is None or float(result["global"]["pq"]) > float(best["global"]["pq"]):
            best = result
        payload = {
            "source_commit": metadata["source_commit"],
            "source_script_sha256": _sha256(Path(__file__).resolve()),
            "model_code_sha256": metadata["seed_refiner_code_sha256"],
            "checkpoint_sha256": metadata["training"]["checkpoint_sha256"],
            "validation_detections_sha256": _sha256(validation_path),
            "split_sha256": split.sha256,
            "official_scorer": metadata["official_scorer"],
            "device": str(device),
            "hardware": hardware_info(),
            "package_versions": package_versions(),
            "completed_candidates": results,
            "best_configuration_so_far": best,
            "elapsed_seconds": time.perf_counter() - started,
        }
        (output_dir / "sweep-metrics.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "candidate": index,
                    "total": len(candidates),
                    **result,
                    "best_pq_so_far": best["global"]["pq"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    assert best is not None
    best_config = replace(
        base_config,
        validation_score_min=float(best["configuration"]["validation_score_min"]),
        probability_threshold=float(best["configuration"]["probability_threshold"]),
        minimum_instance_area=int(best["configuration"]["minimum_instance_area"]),
    )
    predictions = {}
    for file_name in split.validation_names:
        image = read_grayscale(dataset.train_image_dir / file_name)
        predictions[file_name] = predicted_instances(
            file_name,
            image,
            validation_detections.get(Path(file_name).stem, []),
            model,
            best_config,
            device=device,
        )
    metrics, observation_rows = score_validation(
        dataset,
        predictions,
        split.validation_names,
        iou_threshold=float(baseline["iou_threshold"]),
        diagnostic_overlap_threshold=float(baseline["diagnostic_overlap_threshold"]),
    )
    selected_dir = output_dir / "best-holdout-screen"
    selected_dir.mkdir(parents=True, exist_ok=True)
    _write_predictions(selected_dir / "predictions.csv", predictions)
    _write_observation_scores(selected_dir / "per_observation.csv", observation_rows, predictions)
    overlays = write_comparison_overlays(
        dataset, predictions, selected_dir / "comparison-overlays", maximum=12
    )
    (selected_dir / "metrics.json").write_text(
        json.dumps(
            {
                "configuration": best["configuration"],
                "metrics": metrics,
                "split_sha256": split.sha256,
                "overlays": overlays,
                "holdout_screen_only": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"selected_holdout_screen": best}, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
