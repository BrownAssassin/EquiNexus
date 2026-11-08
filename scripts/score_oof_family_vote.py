from __future__ import annotations

# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament_baseline.data import load_dataset
from filament_baseline.predictor import PredictedInstance, encode_binary_mask
from filament_baseline.runner import _write_predictions
from filament_baseline.scorer import score_validation
from filament_baseline.split import create_observation_split


DATA_ROOT = ROOT / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026"
MANIFEST = ROOT / "data" / "splits" / "observation_oof_seed42_5fold.csv"
OUTPUT = ROOT / "artifacts" / "model-experiments" / "public-notebook-oof-family-vote"
SHAPE = (2048, 2048)
EXPECTED_SPLIT = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"


def decode_rle(counts: str) -> dict:
    return {"size": list(SHAPE), "counts": counts.encode("ascii")}


def load_detector(path: Path, confidence: float, stems: set[str]) -> dict[str, list[dict]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if set(raw) != stems:
        raise ValueError(f"{path.name} observation coverage differs")
    result = {}
    for stem, detections in raw.items():
        rows = []
        for index, item in enumerate(detections):
            if float(item["score"]) < confidence:
                continue
            rle = decode_rle(str(item["counts"]))
            if tuple(rle["size"]) != SHAPE or int(mask_utils.area(rle)) == 0:
                continue
            rows.append({"family": "", "rle": rle, "score": float(item["score"]), "index": index})
        result[stem] = rows
    return result


def load_unet(path: Path, stems: set[str]) -> dict[str, list[dict]]:
    result = {stem: [] for stem in stems}
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["filament_id", "segmentation_rle"]:
            raise ValueError("Unexpected U-Net OOF CSV schema")
        for index, row in enumerate(reader):
            stem = str(row["filament_id"]).rsplit("_", maxsplit=1)[0]
            if stem not in result:
                raise ValueError(f"Unexpected U-Net observation {stem}")
            rle = decode_rle(str(row["segmentation_rle"]))
            if int(mask_utils.area(rle)):
                result[stem].append({"family": "unet", "rle": rle, "score": 0.0, "index": index})
    return result


def as_predictions(stem: str, rles: list[dict]) -> list[PredictedInstance]:
    items = []
    for index, rle in enumerate(rles, 1):
        items.append(PredictedInstance(
            filament_id=f"{stem}_{index}", image_stem=stem,
            area=int(mask_utils.area(rle)), rle=rle,
        ))
    return items


def main() -> None:
    started = time.perf_counter()
    dataset = load_dataset(DATA_ROOT)
    frozen = create_observation_split(dataset, ROOT / "data" / "splits", seed=42, validation_fraction=0.2)
    if frozen.sha256 != EXPECTED_SPLIT:
        raise ValueError(f"Frozen split changed: {frozen.sha256}")
    with MANIFEST.open("r", encoding="utf-8", newline="") as stream:
        names = tuple(sorted(row["file_name"] for row in csv.DictReader(stream)))
    if set(names) != set(frozen.training_names) or set(names) & set(frozen.validation_names):
        raise ValueError("OOF observations do not match the frozen training partition")
    stems = {Path(name).stem for name in names}

    sources = {
        "yolo11s": load_detector(
            ROOT / "artifacts/detector-oof/yolo11s-seg-1280-40ep-coco-oof-tta8/dedup-iou-030/oof_detections.json",
            0.35, stems,
        ),
        "unet": load_unet(
            ROOT / "artifacts/oof-experiments/oof-blend-perann10k-tta8-p50-consensus512-5k-tta4/oof-predictions.csv",
            stems,
        ),
        "yolo11m": load_detector(
            ROOT / "artifacts/detector-oof/yolo11m-seg-1024-30ep-coco-oof-local/oof_detections.json",
            0.20, stems,
        ),
    }
    quality_order = ["yolo11s", "unet", "yolo11m"]
    family_metrics = {
        "yolo11s": json.loads((ROOT / "artifacts/detector-oof/yolo11s-seg-1280-40ep-coco-oof-tta8/dedup-iou-030/scoring/oof-score.json").read_text(encoding="utf-8"))["best"]["metrics"]["global"],
        "unet": json.loads((ROOT / "artifacts/detector-oof/yolo11m-seg-1024-30ep-coco-oof-local/fusion-oof/fusion-oof-sweep.json").read_text(encoding="utf-8"))["unet_baseline_metrics"]["global"],
        "yolo11m": json.loads((ROOT / "artifacts/detector-oof/yolo11m-seg-1024-30ep-coco-oof-local/scoring/oof-score.json").read_text(encoding="utf-8"))["best"]["metrics"]["global"],
    }

    voted: dict[str, list[dict]] = {}
    vote_histogram: dict[str, int] = {}
    for name in names:
        stem = Path(name).stem
        candidates = []
        for family in quality_order:
            for item in sources[family][stem]:
                item["family"] = family
                candidates.append(item)
        other_rles = {
            family: [item["rle"] for item in sources[family][stem]]
            for family in quality_order
        }
        for item in candidates:
            family = item["family"]
            support = 1
            for other in quality_order:
                if other == family or not other_rles[other]:
                    continue
                overlaps = mask_utils.iou([item["rle"]], other_rles[other], [0] * len(other_rles[other]))
                support += int(bool(np.any(np.asarray(overlaps) > 0.5)))
            item["votes"] = support
        priority = {family: index for index, family in enumerate(quality_order)}
        candidates.sort(key=lambda item: (priority[item["family"]], -item["score"], item["index"]))
        selected = []
        taken = np.zeros(SHAPE, dtype=bool)
        for item in candidates:
            if item["votes"] < 2:
                continue
            mask = mask_utils.decode(item["rle"]).astype(bool)
            area = int(mask.sum())
            free = mask & ~taken
            free_area = int(free.sum())
            if free_area < 0.5 * area:
                continue
            if free_area != area:
                selected.append(encode_binary_mask(free))
                taken |= free
            else:
                selected.append(item["rle"])
                taken |= mask
        if not selected and candidates:
            fallback = max(candidates, key=lambda item: (item["votes"], -priority[item["family"]], item["score"], -item["index"]))
            selected = [fallback["rle"]]
            vote_histogram["fallback_images"] = vote_histogram.get("fallback_images", 0) + 1
        voted[stem] = selected
        for item in candidates:
            key = str(item["votes"])
            vote_histogram[key] = vote_histogram.get(key, 0) + 1

    predictions = {name: as_predictions(Path(name).stem, voted[Path(name).stem]) for name in names}
    metrics, rows = score_validation(dataset, predictions, names)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _write_predictions(OUTPUT / "family-vote-oof-predictions.csv", predictions)
    source_files = {
        "oof_manifest": MANIFEST,
        "yolo11s_deduplicated_oof": ROOT / "artifacts/detector-oof/yolo11s-seg-1280-40ep-coco-oof-tta8/dedup-iou-030/oof_detections.json",
        "unet_oof_csv": ROOT / "artifacts/oof-experiments/oof-blend-perann10k-tta8-p50-consensus512-5k-tta4/oof-predictions.csv",
        "yolo11m_oof": ROOT / "artifacts/detector-oof/yolo11m-seg-1024-30ep-coco-oof-local/oof_detections.json",
    }
    source_hashes = {
        key: hashlib.sha256(path.read_bytes()).hexdigest()
        for key, path in source_files.items()
    }
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    report = {
        "method": "three-family majority vote, keep at least two family matches at IoU > 0.5; clip overlaps; backfill empty observations",
        "families": quality_order,
        "confidence_thresholds": {"yolo11s": 0.35, "yolo11m": 0.20, "unet": "existing fixed OOF CSV"},
        "source_commit": git_commit,
        "source_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        "hardware": {
            "device": "CPU (scoring only; model predictions were precomputed)",
            "processor": platform.processor(),
            "logical_cpu_count": __import__("os").cpu_count(),
            "platform": platform.platform(),
        },
        "dependencies": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pycocotools": importlib.metadata.version("pycocotools"),
        },
        "data_boundary": {
            "training_labels": "competition-provided annotations only",
            "cross_fitted_predictions": True,
            "frozen_holdout_used_for_selection": False,
            "test_images_or_labels_used": False,
        },
        "input_sha256": source_hashes,
        "split_sha256": frozen.sha256,
        "oof_manifest_sha256": source_hashes["oof_manifest"],
        "oof_observations": len(names),
        "family_metrics": family_metrics,
        "vote_histogram": vote_histogram,
        "vote_metrics": metrics["global"],
        "vote_by_year": {k: v["pq"] for k, v in metrics["by_year"].items()},
        "prediction_count": sum(len(items) for items in predictions.values()),
        "per_observation_pq_mean": float(np.mean([row.pq for row in rows])),
        "per_observation_pq_median": float(np.median([row.pq for row in rows])),
    }
    (OUTPUT / "family-vote-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
