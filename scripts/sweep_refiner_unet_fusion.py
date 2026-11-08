# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from filament_baseline.data import load_dataset  # noqa: E402
from filament_baseline.predictor import (  # noqa: E402
    PredictedInstance,
    decode_rle,
    encode_binary_mask,
)
from filament_baseline.runner import _write_observation_scores, _write_predictions  # noqa: E402
from filament_baseline.scorer import score_validation  # noqa: E402
from filament_baseline.split import create_observation_split  # noqa: E402
from filament_baseline.visuals import write_comparison_overlays  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_instances(path: Path) -> dict[str, list[PredictedInstance]]:
    grouped: dict[str, list[PredictedInstance]] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["filament_id", "segmentation_rle"]:
            raise ValueError(f"Unexpected prediction columns: {reader.fieldnames}")
        for row in reader:
            filament_id = row["filament_id"]
            stem = filament_id.rsplit("_", 1)[0]
            rle = {"size": [2048, 2048], "counts": row["segmentation_rle"]}
            mask = decode_rle(rle)
            grouped.setdefault(stem, []).append(
                PredictedInstance(filament_id, stem, int(mask.sum()), rle)
            )
    return grouped


def _run_variant(
    names: tuple[str, ...],
    base: dict[str, list[PredictedInstance]],
    probability_dir: Path,
    *,
    gate_threshold: float | None,
    growth_radius: int,
    growth_threshold: float,
    minimum_area: int,
) -> dict[str, list[PredictedInstance]]:
    predictions = {}
    for file_name in names:
        stem = Path(file_name).stem
        with np.load(probability_dir / f"{stem}.npz") as archive:
            foreground = archive["foreground"].astype(np.float32)
        claimed = np.zeros(foreground.shape, dtype=bool)
        outputs = []
        for instance in base.get(stem, []):
            mask = decode_rle(instance.rle).astype(bool)
            candidate = mask.copy()
            if gate_threshold is not None:
                candidate &= foreground >= gate_threshold
            if growth_radius > 0:
                kernel_size = 2 * growth_radius + 1
                expanded = cv2.dilate(
                    mask.astype(np.uint8),
                    np.ones((kernel_size, kernel_size), dtype=np.uint8),
                ).astype(bool)
                candidate |= expanded & (foreground >= growth_threshold)
            candidate &= ~claimed
            area = int(candidate.sum())
            if area < minimum_area:
                continue
            claimed |= candidate
            outputs.append(
                PredictedInstance(
                    filament_id=f"{stem}_{len(outputs) + 1}",
                    image_stem=stem,
                    area=area,
                    rle=encode_binary_mask(candidate),
                )
            )
        predictions[file_name] = outputs
    return predictions


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Screen crop-refiner and U-Net mask fusion on the frozen holdout."
    )
    parser.add_argument(
        "--run-name",
        default="2026-11-08-seedrefiner-yolo11s-oof-scratch",
        help="Existing model-experiment directory under artifacts/model-experiments.",
    )
    args = parser.parse_args()
    run_dir = REPO_ROOT / "artifacts" / "model-experiments" / args.run_name
    base_path = run_dir / "threshold-sweep" / "best-holdout-screen" / "predictions.csv"
    probability_dir = REPO_ROOT / "artifacts" / "model-experiments" / "fivefold-plus-blend-r05" / "probabilities"
    probability_provenance = REPO_ROOT / "artifacts" / "model-experiments" / "fivefold-plus-blend-r05" / "blend-provenance.json"
    baseline = json.loads((REPO_ROOT / "configs" / "validation-baseline.json").read_text(encoding="utf-8"))
    dataset = load_dataset(REPO_ROOT / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026")
    split = create_observation_split(
        dataset,
        REPO_ROOT / "data" / "splits",
        seed=int(baseline["seed"]),
        validation_fraction=float(baseline["validation_fraction"]),
    )
    if split.sha256 != baseline["expected_split_sha256"]:
        raise RuntimeError("Fusion sweep requires the unchanged frozen observation split")
    base = _load_instances(base_path)
    missing_maps = [
        Path(name).stem
        for name in split.validation_names
        if not (probability_dir / f"{Path(name).stem}.npz").is_file()
    ]
    if missing_maps:
        raise FileNotFoundError(f"U-Net probability maps are missing for {len(missing_maps)} holdout images")

    candidates = [
        {"name": "gate-025", "gate": 0.25, "radius": 0, "growth": 0.50, "minimum_area": 1000},
        {"name": "gate-035", "gate": 0.35, "radius": 0, "growth": 0.50, "minimum_area": 1000},
        {"name": "gate-045", "gate": 0.45, "radius": 0, "growth": 0.50, "minimum_area": 1000},
        {"name": "gate-055", "gate": 0.55, "radius": 0, "growth": 0.50, "minimum_area": 1000},
        {"name": "gate-065", "gate": 0.65, "radius": 0, "growth": 0.50, "minimum_area": 1000},
        {"name": "grow-r4-p060", "gate": None, "radius": 4, "growth": 0.60, "minimum_area": 1000},
        {"name": "grow-r8-p050", "gate": None, "radius": 8, "growth": 0.50, "minimum_area": 1000},
        {"name": "grow-r8-p060", "gate": None, "radius": 8, "growth": 0.60, "minimum_area": 1000},
        {"name": "grow-r12-p050", "gate": None, "radius": 12, "growth": 0.50, "minimum_area": 1000},
        {"name": "mix-g035-r8-p060", "gate": 0.35, "radius": 8, "growth": 0.60, "minimum_area": 1000},
        {"name": "mix-g045-r8-p060", "gate": 0.45, "radius": 8, "growth": 0.60, "minimum_area": 1000},
    ]
    output_dir = run_dir / "threshold-sweep" / "unet-fusion"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    best = None
    started = time.perf_counter()
    for index, config in enumerate(candidates, start=1):
        predictions = _run_variant(
            split.validation_names,
            base,
            probability_dir,
            gate_threshold=config["gate"],
            growth_radius=config["radius"],
            growth_threshold=config["growth"],
            minimum_area=config["minimum_area"],
        )
        metrics, observation_rows = score_validation(
            dataset,
            predictions,
            split.validation_names,
            iou_threshold=float(baseline["iou_threshold"]),
            diagnostic_overlap_threshold=float(baseline["diagnostic_overlap_threshold"]),
        )
        result = {"configuration": config, "metrics": metrics}
        results.append(result)
        if best is None or float(metrics["global"]["pq"]) > float(best["metrics"]["global"]["pq"]):
            best = result
        candidate_dir = output_dir / config["name"]
        candidate_dir.mkdir(parents=True, exist_ok=True)
        _write_observation_scores(candidate_dir / "per_observation.csv", observation_rows, predictions)
        payload = {
            "source_script_sha256": _sha256(Path(__file__).resolve()),
            "base_predictions_sha256": _sha256(base_path),
            "unet_probability_provenance_sha256": _sha256(probability_provenance),
            "split_sha256": split.sha256,
            "official_scorer": {
                "iou_threshold": float(baseline["iou_threshold"]),
                "diagnostic_overlap_threshold": float(baseline["diagnostic_overlap_threshold"]),
            },
            "completed_candidates": results,
            "best_configuration_so_far": best,
            "elapsed_seconds": time.perf_counter() - started,
            "holdout_screen_only": True,
        }
        (output_dir / "sweep-metrics.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "candidate": index,
                    "total": len(candidates),
                    "configuration": config,
                    "global": metrics["global"],
                    "best_pq_so_far": best["metrics"]["global"]["pq"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    assert best is not None
    selected_config = best["configuration"]
    selected = _run_variant(
        split.validation_names,
        base,
        probability_dir,
        gate_threshold=selected_config["gate"],
        growth_radius=selected_config["radius"],
        growth_threshold=selected_config["growth"],
        minimum_area=selected_config["minimum_area"],
    )
    metrics, observation_rows = score_validation(
        dataset,
        selected,
        split.validation_names,
        iou_threshold=float(baseline["iou_threshold"]),
        diagnostic_overlap_threshold=float(baseline["diagnostic_overlap_threshold"]),
    )
    selected_dir = output_dir / "best-holdout-screen"
    selected_dir.mkdir(parents=True, exist_ok=True)
    _write_predictions(selected_dir / "predictions.csv", selected)
    _write_observation_scores(selected_dir / "per_observation.csv", observation_rows, selected)
    overlays = write_comparison_overlays(
        dataset, selected, selected_dir / "comparison-overlays", maximum=12
    )
    (selected_dir / "metrics.json").write_text(
        json.dumps(
            {
                "configuration": selected_config,
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
    print(json.dumps({"best_holdout_screen": best}, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
