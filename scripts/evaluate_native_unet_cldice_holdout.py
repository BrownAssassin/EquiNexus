#!/usr/bin/env python3
"""Evaluate one topology-loss U-Net on the frozen holdout without threshold tuning."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from workspace_runtime import configure_workspace_runtime  # noqa: E402

configure_workspace_runtime(REPO_ROOT)

import cv2
import torch

import train_native_unet_grouped as native


CLDICE_WEIGHT = 0.25
EPOCHS = 24
TTA_VIEWS = 8
PRETRAINED_ENCODER_PROVENANCE = {
    "timm_model_name": "efficientvit_b2.r288_in1k",
    "huggingface_model": "timm/efficientvit_b2.r224_in1k",
    "huggingface_revision": "891530d0c5a400dd2d4549c81d433661b192abe1",
    "weight_file": "model.safetensors",
    "weight_file_sha256": "f5726851b5c55fab49f8302a79c3a898814e11bc20df76cc23cab28ae15c7328",
    "license": "Apache-2.0",
    "pretraining_dataset": "ImageNet-1K",
    "model_card": "https://huggingface.co/timm/efficientvit_b2.r224_in1k",
}
DEFAULT_BASELINE_DIR = REPO_ROOT / "artifacts" / "model-experiments" / "native-unet-grouped-20261111"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "model-experiments" / "native-unet-cldice-holdout-20261112"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--split-dir", type=Path, default=REPO_ROOT / "data" / "splits")
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    split_dir = args.split_dir.expanduser().resolve()
    baseline_dir = args.baseline_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    artifacts_root = (REPO_ROOT / "artifacts" / "model-experiments").resolve()
    if not output_dir.is_relative_to(artifacts_root):
        parser.error("--output-dir must stay inside artifacts/model-experiments")
    if not split_dir.is_relative_to((REPO_ROOT / "data" / "splits").resolve()):
        parser.error("--split-dir must stay under data/splits")
    workspace_data = (REPO_ROOT / "data").resolve()
    if not data_root.is_relative_to(workspace_data):
        if os.name == "nt" or not data_root.is_relative_to(Path("/kaggle/input").resolve()):
            parser.error("--data-root must stay inside workspace data/ or Kaggle's read-only input mount")
    if not data_root.is_dir():
        raise FileNotFoundError(f"Competition data folder not found: {data_root}")

    baseline_summary_path = baseline_dir / "oof-summary.json"
    if not baseline_summary_path.is_file():
        raise FileNotFoundError(f"Missing native OOF summary: {baseline_summary_path}")
    baseline_summary = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
    selected = baseline_summary["selected_oof_parameters"]
    if baseline_summary["split_sha256"] != native.EXPECTED_SPLIT_SHA256:
        raise RuntimeError("Baseline OOF summary has a different frozen split digest")
    if int(selected["epoch"]) != EPOCHS:
        raise RuntimeError("The topology-loss screen requires the OOF-selected epoch to remain 24")

    metrics_path = output_dir / "holdout-metrics.json"
    if metrics_path.exists():
        raise FileExistsError(f"Refusing to overwrite completed evaluation: {metrics_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
    native.seed_everything(native.SEED)
    cv2.setNumThreads(0)

    dataset, split = native.load_dataset_split(data_root, split_dir)
    if split.sha256 != native.EXPECTED_SPLIT_SHA256:
        raise RuntimeError("Loaded split differs from the frozen seed-42 holdout")
    training_names = list(split.training_names)
    holdout_names = list(split.validation_names)
    images, masks, stats = native.build_training_cache(dataset, training_names, output_dir)
    cache_names = json.loads(native.cache_paths(output_dir)[3].read_text(encoding="utf-8"))["names"]
    train_indices = list(range(len(cache_names)))
    checkpoint_path = output_dir / "frozen-train-full" / "last-checkpoint.pt"

    run_config = {
        "epochs": EPOCHS,
        "batch_size": 1,
        "workers": 0,
        "tta_views": TTA_VIEWS,
        "encoder": native.ENCODER_NAME,
        "cldice_weight": CLDICE_WEIGHT,
        "cldice_patch_size": native.CLDICE_PATCH_SIZE,
        "cldice_iterations": native.CLDICE_ITERATIONS,
        "split_sha256": split.sha256,
        "training_observations": len(training_names),
        "holdout_observations": len(holdout_names),
        "postprocessing_source": str(baseline_summary_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "postprocessing_source_sha256": native.sha256(baseline_summary_path),
        "postprocessing": {
            "epoch": int(selected["epoch"]),
            "threshold": float(selected["threshold"]),
            "min_area": int(selected["min_area"]),
            "closing_kernel": int(selected["closing_kernel"]),
        },
        "holdout_threshold_tuning": False,
    }
    metadata_path = output_dir / "run-metadata.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("config") != run_config:
            raise FileExistsError(f"Output directory has a different config: {metadata_path}")
    else:
        metadata_path.write_text(
            json.dumps(
                {
                    "config": run_config,
                    "source_commit": native.git_commit(),
                    "source_script_sha256": native.sha256(Path(__file__).resolve()),
                    "base_trainer_sha256": native.sha256(Path(native.__file__).resolve()),
                    "device": native.capture_device(device),
                    "package_versions": {
                        **native.package_versions(),
                        "timm": native.importlib.metadata.version("timm"),
                    },
                    "source": native.git_info(REPO_ROOT),
                    "official_scorer": "filament_baseline.scorer.score_validation",
                    "test_labels_read": False,
                    "external_data_used": True,
                    "external_pretrained_weights_used": True,
                    "external_segmentation_labels_used": False,
                    "pretrained_encoder_provenance": PRETRAINED_ENCODER_PROVENANCE,
                    "method_reference": "Shit et al., clDice, CVPR 2021",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    started = time.perf_counter()
    model, training = native.train_model(
        images,
        masks,
        stats,
        cache_names,
        train_indices,
        epochs=EPOCHS,
        batch_size=1,
        workers=0,
        device=device,
        checkpoint_path=checkpoint_path,
        fold_label="full",
        cldice_weight=CLDICE_WEIGHT,
    )
    selected_epoch = int(selected["epoch"])
    selected_checkpoint = native.epoch_checkpoint_path(checkpoint_path, selected_epoch)
    if not selected_checkpoint.is_file():
        raise FileNotFoundError(f"Missing OOF-selected checkpoint: {selected_checkpoint}")
    checkpoint = torch.load(selected_checkpoint, map_location="cpu", weights_only=True)
    if float(checkpoint.get("config", {}).get("cldice_weight", 0.0)) != CLDICE_WEIGHT:
        raise RuntimeError("Selected checkpoint does not contain the configured topology loss")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    probabilities = native.predict_observations(
        model,
        dataset,
        holdout_names,
        None,
        None,
        None,
        device=device,
        tta_views=TTA_VIEWS,
    )
    predictions = native.predictions_for_config(
        probabilities,
        holdout_names,
        threshold=float(selected["threshold"]),
        min_area=int(selected["min_area"]),
        closing_kernel=int(selected["closing_kernel"]),
    )
    metrics, counts = native.score_validation(dataset, predictions, holdout_names)
    rows = [
        {
            "file_name": row.file_name,
            "year": row.year,
            "site": row.site,
            "annotator_views": row.annotator_views,
            "pq": row.pq,
            "tp": row.tp,
            "fp": row.fp,
            "fn": row.fn,
            "tp_iou_sum": row.tp_iou_sum,
            "gt_count": row.gt_count,
            "prediction_count": row.prediction_count,
        }
        for row in counts
    ]
    observation_path = output_dir / "holdout-per-observation.csv"
    with observation_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    result: dict[str, Any] = {
        "split_sha256": split.sha256,
        "training_observations": len(training_names),
        "holdout_observations": len(holdout_names),
        "holdout_annotator_views": sum(len(dataset.views_by_observation[name]) for name in holdout_names),
        "postprocessing_selected_from_native_oof": selected,
        "model_epoch": selected_epoch,
        "cldice_weight": CLDICE_WEIGHT,
        "official_scorer_metrics": metrics,
        "training": training,
        "tta_views": TTA_VIEWS,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "device": native.capture_device(device),
        "source_commit": native.git_commit(),
        "source_script_sha256": native.sha256(Path(__file__).resolve()),
        "base_trainer_sha256": native.sha256(Path(native.__file__).resolve()),
        "test_labels_read": False,
    }
    metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "cldice_holdout_complete", "global": metrics["global"], "runtime_seconds": result["runtime_seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
