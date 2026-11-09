# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Rebuild selector features from the seed-42/seed-100 per-annotator map mean."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_component_selector_oof import (  # noqa: E402
    extract_observation_features,
    load_detector_rows,
    load_prediction_rows,
    target_marginal_pq,
)
from filament_baseline.data import load_dataset  # noqa: E402
from filament_baseline.split import create_observation_split  # noqa: E402


FROZEN_SPLIT_SHA256 = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"
OOF_MANIFEST_SHA256 = "f856708222f5747cd35760932f535e4ae2752fd5161b9cd5981fbf4bbde26a3c"
BASE_SELECTOR = REPO_ROOT / "artifacts/model-experiments/component-selector-official-pq-20261109-rle-fixed"
BASE_PREDICTIONS = REPO_ROOT / "artifacts/model-experiments/oof-fusion-weight25-th035-a200/best-oof-predictions.csv"
DETECTOR_PREDICTIONS = REPO_ROOT / "artifacts/detector-oof/yolo11s-seg-1280-40ep-coco-oof-tta8/dedup-iou-030/oof_detections.json"
PERANN_ENSEMBLE = REPO_ROOT / "artifacts/oof-experiments/oof-perann-seed42-seed100-ensemble-cache-20261109"
CONSENSUS = REPO_ROOT / "artifacts/kaggle/oof-consensus-download/EquiNexus/artifacts/oof-experiments/oof-consensus512-5k-tta4"
OUTPUT = REPO_ROOT / "artifacts/model-experiments/component-selector-seed100-input-features-20261109"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if not OUTPUT.is_relative_to((REPO_ROOT / "artifacts/model-experiments").resolve()):
        raise ValueError("Outputs must remain under artifacts/model-experiments/")
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(OUTPUT)

    config = json.loads((REPO_ROOT / "configs/validation-baseline.json").read_text(encoding="utf-8"))
    dataset = load_dataset(REPO_ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026")
    split = create_observation_split(dataset, REPO_ROOT / "data/splits")
    if split.sha256 != FROZEN_SPLIT_SHA256 or split.sha256 != config["expected_split_sha256"]:
        raise RuntimeError("Frozen validation split digest changed")
    manifest_path = REPO_ROOT / "data/splits/observation_oof_seed42_5fold.csv"
    if sha256(manifest_path) != OOF_MANIFEST_SHA256:
        raise RuntimeError("Five-fold OOF manifest digest changed")

    ensemble_metrics = json.loads((PERANN_ENSEMBLE / "metrics.json").read_text(encoding="utf-8-sig"))
    consensus_metrics = json.loads((CONSENSUS / "metrics.json").read_text(encoding="utf-8-sig"))
    if ensemble_metrics.get("split_sha256") != split.sha256 or ensemble_metrics.get("oof_manifest_sha256") != OOF_MANIFEST_SHA256:
        raise RuntimeError("Seed ensemble maps do not match the frozen grouped OOF partition")
    if consensus_metrics.get("splits", {}).get("frozen_holdout_sha256") != split.sha256:
        raise RuntimeError("Consensus maps do not certify frozen-holdout exclusion")
    if consensus_metrics.get("splits", {}).get("oof_manifest_sha256") != OOF_MANIFEST_SHA256:
        raise RuntimeError("Consensus maps do not match the grouped OOF manifest")
    if consensus_metrics.get("data_boundary", {}).get("external_ground_truth") is not False:
        raise RuntimeError("Consensus map source reports external ground-truth use")

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        fold_by_name = {row["file_name"]: int(row["oof_fold"]) for row in csv.DictReader(stream)}
    names = sorted(split.training_names)
    if set(fold_by_name) != set(names) or set(fold_by_name) & set(split.validation_names):
        raise RuntimeError("OOF folds must cover only the 565-observation training partition")
    stems = {Path(name).stem for name in names}
    predictions = load_prediction_rows(BASE_PREDICTIONS, stems)
    detectors = load_detector_rows(DETECTOR_PREDICTIONS, stems)

    started = time.perf_counter()
    feature_rows: list[dict] = []
    target_rows: list[dict] = []
    for index, file_name in enumerate(names, start=1):
        stem = Path(file_name).stem
        fold = fold_by_name[file_name]
        with (
            np.load(PERANN_ENSEMBLE / "probabilities" / f"{stem}.npz") as perann,
            np.load(CONSENSUS / f"fold-{fold}" / "probabilities" / f"{stem}.npz") as consensus,
        ):
            foreground = 0.25 * perann["foreground"].astype(np.float32) + 0.75 * consensus["foreground"].astype(np.float32)
            boundary = 0.25 * perann["boundary"].astype(np.float32) + 0.75 * consensus["boundary"].astype(np.float32)
        if foreground.shape != (2048, 2048) or boundary.shape != (2048, 2048):
            raise ValueError(f"Unexpected probability dimensions for {file_name}")
        observation_predictions = predictions[stem]
        feature_rows.extend(
            extract_observation_features(
                dataset,
                file_name,
                observation_predictions,
                detectors[stem],
                foreground,
                boundary,
            )
        )
        target_rows.extend(target_marginal_pq(dataset, file_name, observation_predictions))
        if index % 50 == 0 or index == len(names):
            print(f"Built alternate OOF feature rows {index}/{len(names)}", flush=True)

    features = pd.DataFrame(feature_rows)
    targets = pd.DataFrame(target_rows)
    merged = features.merge(targets, on=["file_name", "component_id"], validate="one_to_one")
    if len(merged) != len(features) or len(features) != len(targets):
        raise RuntimeError("Feature and marginal-PQ target rows do not align")
    merged["oof_fold"] = merged.file_name.map(fold_by_name).astype(int)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    feature_path = OUTPUT / "oof-component-features-targets.csv"
    merged.to_csv(feature_path, index=False)

    baseline_metrics = json.loads((BASE_SELECTOR / "metrics.json").read_text(encoding="utf-8-sig"))
    metrics = {
        "status": "complete",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split_sha256": split.sha256,
        "oof_manifest_sha256": OOF_MANIFEST_SHA256,
        "oof_observations": len(names),
        "component_rows": len(merged),
        "holdout_used": False,
        "test_images_or_labels_used": False,
        "training_labels": "competition-provided training annotations for the grouped OOF training partition only",
        "feature_map_policy": "25% 50:50 seed-42/seed-100 per-annotator U-Net map ensemble plus 75% five-fold consensus U-Net maps",
        "base_predictions": "fixed cross-fitted YOLO/U-Net fusion masks; unchanged from incumbent selector features",
        "fit": {"oof_selected": {"global": baseline_metrics["fit"]["oof_selected"]["global"]}},
        "input_sha256": {
            "oof_manifest": sha256(manifest_path),
            "base_predictions": sha256(BASE_PREDICTIONS),
            "detector_predictions": sha256(DETECTOR_PREDICTIONS),
            "seed42_seed100_perann_metrics": sha256(PERANN_ENSEMBLE / "metrics.json"),
            "consensus_metrics": sha256(CONSENSUS / "metrics.json"),
            "feature_target_csv": sha256(feature_path),
            "builder_script": sha256(Path(__file__).resolve()),
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"component_rows": len(merged), "output_dir": str(OUTPUT), "runtime_seconds": metrics["runtime_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
