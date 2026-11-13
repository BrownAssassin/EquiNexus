# SPDX-License-Identifier: AGPL-3.0-only
"""Render fixed representative clDice holdout overlays against the incumbent."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))
from workspace_runtime import configure_workspace_runtime  # noqa: E402

configure_workspace_runtime(REPO_ROOT)

import cv2  # noqa: E402
import torch  # noqa: E402

import train_native_unet_grouped as native  # noqa: E402


SPLIT_SHA256 = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"
DEFAULT_RUN_DIR = REPO_ROOT / "artifacts/model-experiments/native-unet-cldice-holdout-20261112"
DEFAULT_BASELINE_DIR = REPO_ROOT / "artifacts/model-experiments/public-unet-yolo-family-vote-holdout-tta8-conf025-20261111"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def union_rles(rles: list[dict[str, Any]]) -> np.ndarray:
    union = np.zeros((native.FULL_SIZE, native.FULL_SIZE), dtype=bool)
    for rle in rles:
        union |= native.mask_utils.decode(rle).astype(bool)
    return union


def read_baseline_predictions(path: Path, selected_stems: set[str]) -> dict[str, list[dict[str, Any]]]:
    result = {stem: [] for stem in selected_stems}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            filament_id = row["filament_id"]
            stem, _ = filament_id.rsplit("_", maxsplit=1)
            if stem in selected_stems:
                result[stem].append(
                    {"size": [native.FULL_SIZE, native.FULL_SIZE], "counts": row["segmentation_rle"].encode("ascii")}
                )
    return result


def overlay(gray: np.ndarray, gt: np.ndarray, prediction: np.ndarray, text: str) -> np.ndarray:
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    color = canvas.copy()
    both = gt & prediction
    color[gt & ~prediction] = (40, 190, 40)
    color[prediction & ~gt] = (30, 30, 235)
    color[both] = (0, 220, 255)
    canvas = cv2.addWeighted(canvas, 0.52, color, 0.48, 0)
    cv2.putText(canvas, text, (28, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    baseline_dir = args.baseline_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_dir / "overlays"
    )
    if not data_root.is_relative_to((REPO_ROOT / "data").resolve()):
        raise ValueError("Competition data must remain under the workspace data folder")
    for path, root in (
        (run_dir, REPO_ROOT / "artifacts/model-experiments"),
        (baseline_dir, REPO_ROOT / "artifacts/model-experiments"),
        (output_dir, REPO_ROOT / "artifacts/model-experiments"),
    ):
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"Artifact path must remain under {root}: {path}")

    metrics_path = run_dir / "holdout-metrics.json"
    rows_path = run_dir / "holdout-per-observation.csv"
    metadata_path = run_dir / "run-metadata.json"
    checkpoint_path = run_dir / "frozen-train-full" / "checkpoint-epoch-24.pt"
    baseline_metadata_path = baseline_dir / "run-metadata.json"
    baseline_predictions_path = baseline_dir / "validation_predictions.csv"
    for path in (metrics_path, rows_path, metadata_path, checkpoint_path, baseline_metadata_path, baseline_predictions_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required visual-review artifact missing: {path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    run_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    baseline_metadata = json.loads(baseline_metadata_path.read_text(encoding="utf-8"))
    if (
        metrics.get("split_sha256") != SPLIT_SHA256
        or baseline_metadata.get("fixed_holdout_split_sha256") != SPLIT_SHA256
        or run_metadata.get("config", {}).get("split_sha256") != SPLIT_SHA256
    ):
        raise RuntimeError("Visual-review inputs do not use the frozen grouped split")
    if (
        baseline_metadata.get("holdout_used_for_training") is not False
        or baseline_metadata.get("holdout_used_for_model_or_threshold_selection") is not False
        or run_metadata.get("config", {}).get("holdout_threshold_tuning") is not False
    ):
        raise RuntimeError("Visual-review inputs do not certify a fixed, independent holdout")

    with rows_path.open(newline="", encoding="utf-8") as stream:
        scored_rows = list(csv.DictReader(stream))
    if len(scored_rows) != 142:
        raise RuntimeError(f"Expected 142 scored observations, found {len(scored_rows)}")
    ranked = sorted(scored_rows, key=lambda row: (float(row["pq"]), row["file_name"]))
    positions = sorted({0, 1, 2, len(ranked) // 2, len(ranked) - 3, len(ranked) - 2, len(ranked) - 1})
    sample_rows = [ranked[position] for position in positions]
    names = [row["file_name"] for row in sample_rows]
    stems = {Path(name).stem for name in names}
    baseline_score_rows = {row["file_name"]: row for row in baseline_metadata["per_observation"]}
    if set(baseline_score_rows) != {row["file_name"] for row in scored_rows}:
        raise RuntimeError("Incumbent and clDice score summaries do not cover the same holdout observations")
    baseline_by_stem = read_baseline_predictions(baseline_predictions_path, stems)

    dataset, split = native.load_dataset_split(data_root, REPO_ROOT / "data/splits")
    if split.sha256 != SPLIT_SHA256 or not set(names).issubset(set(split.validation_names)):
        raise RuntimeError("Visual-review sample contains an observation outside the frozen holdout")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        int(checkpoint.get("epoch", -1)) != 24
        or checkpoint.get("fold") != "full"
        or set(checkpoint.get("training_observations", [])) != set(split.training_names)
    ):
        raise RuntimeError("clDice visual-review checkpoint has unexpected training provenance")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
    model = native.NativeResolutionUNet(pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = model.to(memory_format=torch.channels_last).eval()

    config = run_metadata["config"]
    selected = config["postprocessing"]
    started = time.perf_counter()
    probabilities = native.predict_observations(
        model, dataset, names, None, None, None, device=device, tta_views=int(config["tta_views"])
    )
    candidate_by_name = native.predictions_for_config(
        probabilities,
        names,
        threshold=float(selected["threshold"]),
        min_area=int(selected["min_area"]),
        closing_kernel=int(selected["closing_kernel"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_records = []
    count_rows = {row["file_name"]: row for row in scored_rows}
    for row in sample_rows:
        name = row["file_name"]
        stem = Path(name).stem
        views = dataset.views_by_observation[name]
        first_view = views[0]
        gt_annotations = dataset.annotations_by_image_id[first_view.image_id]
        gt = union_rles(
            [native.annotation_to_rle(annotation, native.FULL_SIZE, native.FULL_SIZE) for annotation in gt_annotations]
        )
        baseline_rles = baseline_by_stem.get(stem, [])
        baseline_mask = union_rles(baseline_rles)
        candidate_items = candidate_by_name[name]
        candidate_mask = union_rles([item.rle for item in candidate_items])
        gray = cv2.imread(str(dataset.train_image_dir / name), cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape != (native.FULL_SIZE, native.FULL_SIZE):
            raise ValueError(f"Could not read holdout image for overlay: {name}")
        for label, prediction, instances in (
            ("incumbent", baseline_mask, len(baseline_rles)),
            ("cldice", candidate_mask, len(candidate_items)),
        ):
            observation_pq = (
                float(baseline_score_rows[name]["pq"])
                if label == "incumbent"
                else float(count_rows[name]["pq"])
            )
            image = overlay(
                gray,
                gt,
                prediction,
                f"{stem} {label} | GT view 1={len(gt_annotations)} pred={instances} obs PQ={observation_pq:.3f}",
            )
            path = output_dir / f"{stem}-{label}.png"
            if not cv2.imwrite(str(path), image):
                raise OSError(f"Could not write overlay: {path}")
            output_records.append({"file_name": name, "role": label, "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"), "sha256": sha256_file(path)})

    manifest = {
        "split_sha256": SPLIT_SHA256,
        "sample_policy": "three lowest, median, and three highest observation PQs; visual review only, no parameter selection",
        "scorer": "official observation-grouped PQ; overlays compare predictions to the first annotation view",
        "holdout_labels_used_for_visualization": True,
        "test_images_or_labels_used": False,
        "training_observations": 565,
        "holdout_observations": 142,
        "cldice_run_metadata_sha256": sha256_file(metadata_path),
        "cldice_holdout_metrics_sha256": sha256_file(metrics_path),
        "cldice_checkpoint_sha256": sha256_file(checkpoint_path),
        "incumbent_run_metadata_sha256": sha256_file(baseline_metadata_path),
        "incumbent_predictions_sha256": sha256_file(baseline_predictions_path),
        "sample": [
            {
                "file_name": row["file_name"],
                "observation_pq": float(row["pq"]),
                "annotator_views": len(dataset.views_by_observation[row["file_name"]]),
                "gt_instances_in_visualized_view": len(dataset.annotations_by_image_id[dataset.views_by_observation[row["file_name"]][0].image_id]),
            }
            for row in sample_rows
        ],
        "overlays": output_records,
        "device": native.capture_device(device),
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    manifest_path = output_dir / "visualization-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "holdout_overlays_complete", "count": len(output_records), "manifest": str(manifest_path)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
