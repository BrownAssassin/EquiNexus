# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Compare single-view, 8-view, and topology-guarded native U-Net OOF maps.

The guard thresholds are transferred unchanged from the public I079 handoff;
this script evaluates them on our own cross-fitted observations and does not
use the I079 weights or predictions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from workspace_runtime import configure_workspace_runtime  # noqa: E402

configure_workspace_runtime(REPO_ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import train_native_unet_grouped as native
from filament_baseline.scorer import score_validation


GUARD_AREA_MIN = 0.60
GUARD_AREA_MAX = 1.70
GUARD_DICE_MIN = 0.62
GUARD_MIN_CONTROL_AREA = 32


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_fold_map(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return {row["file_name"]: int(row["fold"]) for row in csv.DictReader(stream)}


def select_probability(
    single: np.ndarray,
    tta: np.ndarray,
    *,
    threshold: float,
) -> tuple[np.ndarray, bool, str, dict[str, float | int]]:
    """Apply the public I079 topology guards to a full-observation mask."""
    single_mask = np.asarray(single, dtype=np.float32) >= np.float32(threshold)
    tta_mask = np.asarray(tta, dtype=np.float32) >= np.float32(threshold)
    single_area = int(single_mask.sum())
    tta_area = int(tta_mask.sum())
    diagnostics: dict[str, float | int] = {
        "single_area": single_area,
        "tta_area": tta_area,
        "single_components": 0,
        "tta_components": 0,
        "area_ratio": float("nan"),
        "mask_dice": float("nan"),
    }
    if tta_area == 0:
        return single, False, "empty_tta_mask", diagnostics
    if single_area == 0:
        return single, False, "empty_single_view_mask", diagnostics
    ratio = tta_area / single_area
    diagnostics["area_ratio"] = float(ratio)
    if ratio < GUARD_AREA_MIN:
        return single, False, "area_shrink_guard", diagnostics
    if ratio > GUARD_AREA_MAX:
        return single, False, "area_growth_guard", diagnostics

    single_binary = np.ascontiguousarray(single_mask.astype(np.uint8))
    tta_binary = np.ascontiguousarray(tta_mask.astype(np.uint8))
    single_count = int(cv2.connectedComponentsWithStats(single_binary, connectivity=8)[0] - 1)
    tta_count = int(cv2.connectedComponentsWithStats(tta_binary, connectivity=8)[0] - 1)
    diagnostics["single_components"] = single_count
    diagnostics["tta_components"] = tta_count
    if tta_count > max(single_count + 1, 2):
        return single, False, "fragmentation_guard", diagnostics

    if single_area >= GUARD_MIN_CONTROL_AREA:
        intersection = int(np.logical_and(single_mask, tta_mask).sum())
        dice = 2.0 * intersection / float(single_area + tta_area)
        diagnostics["mask_dice"] = float(dice)
        if dice < GUARD_DICE_MIN:
            return single, False, "overlap_guard", diagnostics
    return tta, True, "accepted", diagnostics


def paired_bootstrap(
    baseline_rows: list[Any],
    candidate_rows: list[Any],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    if len(baseline_rows) != len(candidate_rows) or not baseline_rows:
        raise ValueError("Paired OOF observation rows do not match")
    if [row.file_name for row in baseline_rows] != [row.file_name for row in candidate_rows]:
        raise ValueError("Paired OOF observations are not in the same order")
    values = []
    for field in ("tp_iou_sum", "tp", "fp", "fn"):
        values.append(
            np.asarray([float(getattr(row, field)) for row in baseline_rows], dtype=np.float64)
        )
    candidate_values = [
        np.asarray([float(getattr(row, field)) for row in candidate_rows], dtype=np.float64)
        for field in ("tp_iou_sum", "tp", "fp", "fn")
    ]
    rng = np.random.default_rng(seed)
    deltas: list[np.ndarray] = []
    count = len(baseline_rows)
    for start in range(0, resamples, 500):
        batch = min(500, resamples - start)
        indices = rng.integers(0, count, size=(batch, count))
        base_sum, base_tp, base_fp, base_fn = (array[indices].sum(axis=1) for array in values)
        cand_sum, cand_tp, cand_fp, cand_fn = (
            array[indices].sum(axis=1) for array in candidate_values
        )
        base_den = base_tp + 0.5 * base_fp + 0.5 * base_fn
        cand_den = cand_tp + 0.5 * cand_fp + 0.5 * cand_fn
        base_pq = np.divide(base_sum, base_den, out=np.zeros(batch), where=base_den > 0)
        cand_pq = np.divide(cand_sum, cand_den, out=np.zeros(batch), where=cand_den > 0)
        deltas.append(cand_pq - base_pq)
    delta = np.concatenate(deltas)
    return {
        "resamples": int(resamples),
        "seed": int(seed),
        "mean_delta": float(delta.mean()),
        "median_delta": float(np.median(delta)),
        "paired_delta_ci_95": [float(value) for value in np.quantile(delta, [0.025, 0.975])],
        "p_candidate_better": float(np.mean(delta > 0.0)),
    }


def read_or_create_single_map(
    map_path: Path,
    progress_path: Path,
    shape: tuple[int, int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    if map_path.exists() != progress_path.exists():
        raise FileExistsError("Single-view map and progress manifest must either both exist or both be absent")
    if map_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        values = np.load(map_path, mmap_mode="r+")
        if values.shape != shape or values.dtype != np.float16:
            raise RuntimeError(f"Unexpected single-view map shape or dtype: {map_path}")
        if progress.get("shape") != list(shape):
            raise RuntimeError("Single-view progress manifest has a different observation shape")
        return values, progress
    map_path.parent.mkdir(parents=True, exist_ok=True)
    values = np.lib.format.open_memmap(map_path, mode="w+", dtype=np.float16, shape=shape)
    progress = {"completed_folds": [], "shape": list(shape), "status": "in_progress"}
    progress_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return values, progress


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    experiments_root = (REPO_ROOT / "artifacts" / "model-experiments").resolve()
    if not run_dir.is_relative_to(experiments_root) or not output_dir.is_relative_to(experiments_root):
        raise ValueError("Run and output directories must remain under artifacts/model-experiments")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "oof-summary.json"
    metadata_path = run_dir / "run-metadata-oof.json"
    if not summary_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Complete native U-Net OOF summary and metadata are required")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("mode") != "oof" or metadata["config"].get("split_sha256") != native.EXPECTED_SPLIT_SHA256:
        raise RuntimeError("Native U-Net run is not the expected frozen-split OOF run")
    if int(metadata["config"].get("tta_views", 0)) != 8:
        raise RuntimeError("This comparison requires the existing 8-view OOF maps")
    if summary.get("split_sha256") != native.EXPECTED_SPLIT_SHA256:
        raise RuntimeError("OOF result uses a different frozen split")
    for fold in range(5):
        if not (run_dir / f"fold-{fold}" / "fold-complete.json").is_file():
            raise RuntimeError(f"Native U-Net OOF fold {fold} is not complete")

    dataset, split = native.load_dataset_split(args.data_root.resolve(), args.split_dir.resolve())
    names = sorted(split.training_names)
    fold_manifest_path = run_dir / "oof-folds.csv"
    fold_map = load_fold_map(fold_manifest_path)
    if set(fold_map) != set(names) or set(fold_map.values()) != set(range(5)):
        raise RuntimeError("OOF fold map does not cover the frozen training observations exactly")
    cache_names = json.loads(native.cache_paths(run_dir)[3].read_text(encoding="utf-8"))["names"]
    if cache_names != names:
        raise RuntimeError("OOF image cache and frozen training observations differ")
    image_path, _, stats_path, _ = native.cache_paths(run_dir)
    images = np.load(image_path, mmap_mode="r")
    stats = np.load(stats_path, mmap_mode="r")
    index_by_name = {name: index for index, name in enumerate(cache_names)}
    expected_shape = (len(names), native.CROP_SIZE, native.CROP_SIZE)
    epoch = int(summary["selected_oof_parameters"]["epoch"])
    relative_tta_path = Path(summary["prediction_files"][str(epoch)])
    tta_path = relative_tta_path if relative_tta_path.is_absolute() else REPO_ROOT / relative_tta_path
    tta_maps = np.load(tta_path, mmap_mode="r")
    if tta_maps.shape != expected_shape or tta_maps.dtype != np.float16:
        raise RuntimeError(f"Unexpected 8-view OOF probability map: {tta_path}")

    single_map_path = output_dir / f"single-probabilities-epoch{epoch:02d}.npy"
    progress_path = output_dir / f"single-progress-epoch{epoch:02d}.json"
    single_maps, progress = read_or_create_single_map(single_map_path, progress_path, expected_shape)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for fold in range(5):
        if fold in progress.get("completed_folds", []):
            print(f"[tta-guard] reused single-view fold {fold}", flush=True)
            continue
        fold_dir = run_dir / f"fold-{fold}"
        completion = json.loads((fold_dir / "fold-complete.json").read_text(encoding="utf-8"))
        validation_names = sorted(name for name in names if fold_map[name] == fold)
        if completion.get("validation_names") != validation_names:
            raise RuntimeError(f"Fold {fold} validation names differ from the completed OOF record")
        checkpoint_path = native.epoch_checkpoint_path(fold_dir / "last-checkpoint.pt", epoch)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing selected epoch checkpoint for fold {fold}: {checkpoint_path}")
        model = native.NativeResolutionUNet(pretrained=False).to(device)
        model = model.to(memory_format=torch.channels_last)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["state_dict"], strict=True)
        model.eval()
        for position, name in enumerate(validation_names, start=1):
            index = index_by_name[name]
            single_maps[index] = native.predict_crop(
                model,
                np.asarray(images[index]),
                tuple(float(value) for value in stats[index]),
                device=device,
                tta_views=1,
            )
            if position % 20 == 0 or position == len(validation_names):
                print(f"[tta-guard] fold {fold}: {position}/{len(validation_names)} single views", flush=True)
        single_maps.flush()
        progress["completed_folds"] = sorted(set(progress.get("completed_folds", [])) | {fold})
        progress_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if progress.get("completed_folds") != list(range(5)):
        raise RuntimeError("Single-view prediction generation did not complete all five folds")

    selected = summary["selected_oof_parameters"]
    threshold = float(selected["threshold"])
    min_area = int(selected["min_area"])
    closing_kernel = int(selected["closing_kernel"])
    tta_by_name = {name: tta_maps[index_by_name[name]] for name in names}
    single_by_name = {name: single_maps[index_by_name[name]] for name in names}
    common_config = {"threshold": threshold, "min_area": min_area, "closing_kernel": closing_kernel}
    tta_predictions = native.predictions_for_config(tta_by_name, names, **common_config)
    single_predictions = native.predictions_for_config(single_by_name, names, **common_config)

    guard_predictions: dict[str, list[native.PredictedInstance]] = {}
    decision_rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for name in names:
        index = index_by_name[name]
        chosen, accepted, reason, diagnostics = select_probability(
            single_maps[index], tta_maps[index], threshold=threshold
        )
        reason_counts[reason] += 1
        guard_predictions[name] = native.components_to_instances(
            chosen,
            name,
            threshold=threshold,
            min_area=min_area,
            closing_kernel=closing_kernel,
        )
        decision_rows.append(
            {"file_name": name, "fold": fold_map[name], "accepted_tta": accepted, "reason": reason, **diagnostics}
        )

    metrics: dict[str, Any] = {}
    observations: dict[str, list[Any]] = {}
    for label, predictions in (
        ("single_view", single_predictions),
        ("tta8", tta_predictions),
        ("topology_guarded", guard_predictions),
    ):
        metrics[label], observations[label] = score_validation(dataset, predictions, names)
    comparisons = {
        "topology_guarded_vs_tta8": paired_bootstrap(
            observations["tta8"], observations["topology_guarded"],
            resamples=args.resamples, seed=42,
        ),
        "single_view_vs_tta8": paired_bootstrap(
            observations["tta8"], observations["single_view"],
            resamples=args.resamples, seed=43,
        ),
    }
    observation_path = output_dir / "observation-results.csv"
    output_rows: list[dict[str, Any]] = []
    observation_maps = {
        label: {row.file_name: row for row in rows}
        for label, rows in observations.items()
    }
    for decision in decision_rows:
        row: dict[str, Any] = dict(decision)
        for label, values in observation_maps.items():
            record = values[decision["file_name"]]
            for field in ("pq", "tp", "fp", "fn", "tp_iou_sum"):
                row[f"{label}_{field}"] = getattr(record, field)
        output_rows.append(row)
    if output_rows:
        with observation_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(output_rows)

    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    report = {
        "split_sha256": split.sha256,
        "partition": "five-fold cross-fitted frozen training partition; no observation appears in its own model fit",
        "selection_caveat": (
            "The parent OOF run selected the epoch and post-processing parameters on these same cross-fitted "
            "observations. The paired comparison is exploratory, does not account for that selection, and must "
            "be confirmed on the independent frozen holdout before promotion."
        ),
        "training_observations": len(names),
        "annotator_views": sum(len(dataset.views_by_observation[name]) for name in names),
        "source_oof_run": str(run_dir.relative_to(REPO_ROOT)),
        "source_oof_summary_sha256": sha256(summary_path),
        "source_training_commit": metadata["source_commit"],
        "source_training_script_sha256": metadata["source_script_sha256"],
        "selected_epoch_and_postprocessing": selected,
        "guard_parameters_transferred_from": "Fususu public I079 topology-safe inference handoff, Apache-2.0",
        "guard_parameters": {
            "area_ratio_min": GUARD_AREA_MIN,
            "area_ratio_max": GUARD_AREA_MAX,
            "dice_min": GUARD_DICE_MIN,
            "minimum_control_area": GUARD_MIN_CONTROL_AREA,
            "maximum_component_count": "max(single_view_components + 1, 2)",
        },
        "guard_decisions": dict(sorted(reason_counts.items())),
        "metrics": metrics,
        "paired_comparisons": comparisons,
        "single_probability_path": str(single_map_path.relative_to(REPO_ROOT)),
        "single_probability_sha256": sha256(single_map_path),
        "tta_probability_path": str(tta_path.relative_to(REPO_ROOT)) if tta_path.is_relative_to(REPO_ROOT) else str(tta_path),
        "observation_results": str(observation_path.relative_to(REPO_ROOT)),
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "device": native.capture_device(device),
        "peak_gpu_memory_bytes": peak_memory,
        "source_commit": native.git_commit(),
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    report_path = output_dir / "tta-guard-summary.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    progress["status"] = "complete"
    progress["report_sha256"] = sha256(report_path)
    progress_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "tta_guard_complete", "report": str(report_path), "metrics": {key: value["global"] for key, value in metrics.items()}, "paired_comparisons": comparisons}, indent=2), flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "model-experiments" / "native-unet-grouped-20261111",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "model-experiments" / "native-unet-tta-guard-20261112",
    )
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--split-dir", type=Path, default=REPO_ROOT / "data" / "splits")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resamples", type=int, default=20_000)
    args = parser.parse_args()
    if args.resamples < 100:
        parser.error("--resamples must be at least 100")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
