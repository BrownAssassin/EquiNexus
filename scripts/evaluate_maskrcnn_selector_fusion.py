# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Screen grouped-holdout fusion policies for RF selector and Mask R-CNN masks."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_component_selector_probability_blend_holdout import (  # noqa: E402
    paired_bootstrap,
    per_observation_rows,
)
from filament_baseline.data import load_dataset  # noqa: E402
from filament_baseline.predictor import PredictedInstance  # noqa: E402
from filament_baseline.scorer import score_validation  # noqa: E402
from filament_baseline.split import create_observation_split  # noqa: E402
from run_component_selector_oof import load_prediction_rows  # noqa: E402

FROZEN_SPLIT_SHA256 = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"
DEFAULT_SELECTOR = ROOT / "artifacts/model-experiments/component-selector-rf-blend-holdout-20261109-final/candidate-predictions.csv"
DEFAULT_DETECTOR = ROOT / "artifacts/detector-experiments/maskrcnn-r50fpn-v2-1024-all-annotator-views-holdout-20261110"
DEFAULT_OUTPUT = ROOT / "artifacts/detector-experiments/maskrcnn-allview-selector-fusion-20261110"
MODES = ("replace", "union", "intersection", "add_unmatched")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rle_bytes(rle: dict[str, Any]) -> dict[str, Any]:
    counts = rle["counts"]
    return {"size": [2048, 2048], "counts": counts.encode("ascii") if isinstance(counts, str) else counts}


def filter_detector_predictions(
    payload: dict[str, list[dict[str, Any]]],
    *,
    confidence_threshold: float,
    minimum_area: int,
) -> dict[str, list[dict[str, Any]]]:
    filtered: dict[str, list[dict[str, Any]]] = {}
    for stem, items in payload.items():
        retained = []
        for index, item in enumerate(items, start=1):
            rle = _rle_bytes({"counts": item["counts"]})
            area = int(mask_utils.area(rle))
            score = float(item["score"])
            if score >= confidence_threshold and area >= minimum_area:
                retained.append({"index": index, "score": score, "area": area, "rle": rle})
        filtered[stem] = retained
    return filtered


def fuse_one_observation(
    baseline: list[dict[str, Any]],
    detector: list[dict[str, Any]],
    *,
    matching_iou: float,
) -> dict[str, list[dict[str, Any]]]:
    if not baseline or not detector:
        return {mode: baseline for mode in MODES}

    overlaps = np.zeros((len(baseline), len(detector)), dtype=np.float32)
    baseline_rles = [_rle_bytes(item) for item in baseline]
    for detector_index, item in enumerate(detector):
        overlaps[:, detector_index] = mask_utils.iou(
            [item["rle"]], baseline_rles, [0] * len(baseline)
        )[0]

    edges = sorted(
        (
            (float(overlaps[base_index, detector_index]), base_index, detector_index)
            for base_index in range(len(baseline))
            for detector_index in range(len(detector))
        ),
        key=lambda edge: (-edge[0], edge[1], edge[2]),
    )
    matches: dict[int, int] = {}
    used_baseline: set[int] = set()
    used_detector: set[int] = set()
    for overlap, base_index, detector_index in edges:
        if overlap < matching_iou:
            break
        if base_index not in used_baseline and detector_index not in used_detector:
            matches[base_index] = detector_index
            used_baseline.add(base_index)
            used_detector.add(detector_index)

    result = {mode: [] for mode in MODES}
    for base_index, baseline_item in enumerate(baseline):
        base_rle = _rle_bytes(baseline_item)
        if base_index not in matches:
            for mode in MODES:
                result[mode].append(base_rle)
            continue

        detector_item = detector[matches[base_index]]
        detector_rle = detector_item["rle"]
        result["replace"].append(detector_rle)
        result["union"].append(mask_utils.merge([base_rle, detector_rle]))
        base_mask = mask_utils.decode(base_rle).astype(bool)
        detector_mask = mask_utils.decode(detector_rle).astype(bool)
        intersection = np.asfortranarray((base_mask & detector_mask).astype(np.uint8))
        result["intersection"].append(
            mask_utils.encode(intersection) if intersection.any() else base_rle
        )
        result["add_unmatched"].append(base_rle)

    for detector_index, item in enumerate(detector):
        if (
            detector_index not in used_detector
            and float(item["score"]) >= 0.95
            and not np.any(overlaps[:, detector_index] > 0.05)
        ):
            result["add_unmatched"].append(item["rle"])
    return result


def prediction_objects(
    dataset: Any,
    names: tuple[str, ...],
    masks_by_stem: dict[str, list[dict[str, Any]]],
) -> dict[str, list[PredictedInstance]]:
    predictions: dict[str, list[PredictedInstance]] = {}
    for file_name in names:
        stem = Path(file_name).stem
        predictions[file_name] = []
        for index, rle in enumerate(masks_by_stem[stem], start=1):
            predictions[file_name].append(
                PredictedInstance(
                    filament_id=f"{stem}_{index}",
                    image_stem=stem,
                    area=int(mask_utils.area(rle)),
                    rle=rle,
                )
            )
    return predictions


def serializable_masks(
    names: tuple[str, ...], masks_by_stem: dict[str, list[dict[str, Any]]]
) -> dict[str, list[dict[str, Any]]]:
    result = {}
    for file_name in names:
        stem = Path(file_name).stem
        result[stem] = [
            {"size": [2048, 2048], "counts": item["counts"].decode("ascii") if isinstance(item["counts"], bytes) else item["counts"]}
            for item in masks_by_stem[stem]
        ]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--selector-predictions", type=Path, default=DEFAULT_SELECTOR)
    parser.add_argument("--detector-run", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    selector_path = args.selector_predictions.expanduser().resolve()
    detector_root = args.detector_run.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not data_root.is_relative_to(ROOT / "data"):
        parser.error("Competition data must stay under data/")
    if not selector_path.is_relative_to(ROOT) or not detector_root.is_relative_to(ROOT):
        parser.error("Inputs must stay inside this repository")
    if not output.is_relative_to(ROOT / "artifacts"):
        parser.error("Fusion outputs must stay under artifacts/")
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory must be empty or not yet created")

    detector_predictions_path = detector_root / "best-validation-detections.json"
    detector_metadata_path = detector_root / "run-metadata.json"
    if not detector_predictions_path.is_file() or not detector_metadata_path.is_file():
        parser.error("Detector run must contain metadata and best validation detections")

    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    dataset = load_dataset(data_root)
    split = create_observation_split(dataset, ROOT / "data/splits")
    if split.sha256 != FROZEN_SPLIT_SHA256:
        raise RuntimeError(f"Frozen split changed: {split.sha256}")
    names = tuple(sorted(split.validation_names))
    stems = {Path(name).stem for name in names}
    base = load_prediction_rows(selector_path, stems)
    detector_payload = json.loads(detector_predictions_path.read_text(encoding="utf-8"))
    if set(detector_payload) != stems:
        raise ValueError("Detector predictions do not cover exactly the holdout observations")
    detector_metadata = json.loads(detector_metadata_path.read_text(encoding="utf-8"))
    selection = detector_metadata["best_validation"]
    detector = filter_detector_predictions(
        detector_payload,
        confidence_threshold=float(selection["confidence_threshold"]),
        minimum_area=int(selection["minimum_area"]),
    )

    results = []
    best_result: dict[str, Any] | None = None
    best_rows = None
    baseline_rows = None
    for matching_iou in (0.25, 0.50):
        mode_masks = {mode: {stem: [] for stem in stems} for mode in MODES}
        for stem in sorted(stems):
            variants = fuse_one_observation(base[stem], detector[stem], matching_iou=matching_iou)
            for mode in MODES:
                mode_masks[mode][stem] = variants[mode]
        for mode in MODES:
            predictions = prediction_objects(dataset, names, mode_masks[mode])
            metrics, rows = score_validation(dataset, predictions, names)
            row = {
                "matching_iou_minimum": matching_iou,
                "mode": mode,
                "global": metrics["global"],
                "by_site": metrics["by_site"],
                "by_year": metrics["by_year"],
            }
            results.append(row)
            if best_result is None or row["global"]["pq"] > best_result["global"]["pq"]:
                best_result = row
                best_rows = rows
                best_masks = mode_masks[mode]
        if baseline_rows is None:
            baseline_predictions = prediction_objects(dataset, names, base)
            _, baseline_rows = score_validation(dataset, baseline_predictions, names)

    assert best_result is not None and best_rows is not None and baseline_rows is not None
    bootstrap = paired_bootstrap(baseline_rows, best_rows)
    (output / "best-candidate-predictions.json").write_text(
        json.dumps(serializable_masks(names, best_masks), separators=(",", ":")),
        encoding="utf-8",
    )
    report = {
        "status": "complete",
        "split_sha256": split.sha256,
        "validation_observations": len(names),
        "validation_annotator_views": sum(len(dataset.views_by_observation[name]) for name in names),
        "test_images_or_labels_used": False,
        "selector_predictions_sha256": sha256(selector_path),
        "detector_predictions_sha256": sha256(detector_predictions_path),
        "detector_checkpoint_sha256": detector_metadata.get("best_checkpoint_sha256"),
        "detector_source_sha256": detector_metadata.get("source_sha256"),
        "detector_filter": {
            "confidence_threshold": selection["confidence_threshold"],
            "minimum_area": selection["minimum_area"],
        },
        "matching": "greedy one-to-one by detector-mask IoU",
        "modes": list(MODES),
        "results": results,
        "best_result": best_result,
        "paired_bootstrap_vs_selector": bootstrap,
        "best_per_observation": per_observation_rows(best_rows),
        "best_predictions": "best-candidate-predictions.json",
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for row in sorted(results, key=lambda item: item["global"]["pq"], reverse=True):
        print(
            f"IoU>={row['matching_iou_minimum']:.2f} {row['mode']}: "
            f"PQ={row['global']['pq']:.6f} TP={row['global']['tp']} "
            f"FP={row['global']['fp']} FN={row['global']['fn']}",
            flush=True,
        )
    print(json.dumps({"best_result": best_result, "paired_bootstrap_vs_selector": bootstrap}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
