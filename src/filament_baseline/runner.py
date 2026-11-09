# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .data import CompetitionDataset, find_dataset_root, inventory, load_dataset
from .predictor import BaselineConfig, PredictedInstance, predict_image, read_grayscale, write_prediction_overlay
from .runtime import git_info, hardware_info, package_versions
from .scorer import ObservationCounts, score_validation
from .split import create_observation_split

OFFICIAL_SCORER_URL = (
    "https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook"
)


def _within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _is_kaggle_runtime() -> bool:
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or (
        Path("/kaggle/input").is_dir() and Path("/kaggle/working").is_dir()
    )


def _source_fingerprint(repo_root: Path) -> str:
    paths = [
        *sorted((repo_root / "src").rglob("*.py")),
        *sorted((repo_root / "scripts").glob("*.py")),
        repo_root / "configs" / "validation-baseline.json",
        repo_root / "notebooks" / "validation_baseline.ipynb",
        repo_root / "requirements.txt",
        repo_root / "requirements-gpu.txt",
    ]
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            continue
        digest.update(path.relative_to(repo_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _load_config(repo_root: Path) -> tuple[dict[str, Any], BaselineConfig]:
    path = repo_root / "configs" / "validation-baseline.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    baseline = BaselineConfig(
        blur_kernel=int(payload["blur_kernel"]),
        solar_disk_radius_fraction=float(payload["solar_disk_radius_fraction"]),
        closing_kernel=int(payload["closing_kernel"]),
        component_connectivity=int(payload["component_connectivity"]),
        min_component_area=int(payload["min_component_area"]),
        max_component_area=int(payload["max_component_area"]),
    )
    return payload, baseline


def _resolve_data_root(data_root: str | Path | None, repo_root: Path) -> Path:
    candidates: list[Path] = []
    if data_root is not None:
        candidates.append(Path(data_root))
    env_root = os.environ.get("EQUINEXUS_DATA_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend(
        [
            repo_root / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026",
            Path("/kaggle/input"),
        ]
    )
    errors = []
    for candidate in candidates:
        try:
            return find_dataset_root(candidate)
        except (FileNotFoundError, RuntimeError) as exc:
            errors.append(f"{candidate}: {exc}")
    raise FileNotFoundError(
        "No competition dataset was found. Set EQUINEXUS_DATA_ROOT or pass --data-root.\n"
        + "\n".join(errors)
    )


def _write_predictions(
    output_path: Path,
    predictions: dict[str, list[PredictedInstance]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("filament_id", "segmentation_rle"))
        for file_name in sorted(predictions):
            for prediction in predictions[file_name]:
                counts = prediction.rle["counts"]
                if isinstance(counts, bytes):
                    counts = counts.decode("ascii")
                writer.writerow((prediction.filament_id, counts))


def _write_observation_scores(
    output_path: Path,
    rows: list[ObservationCounts],
    predictions: dict[str, list[PredictedInstance]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "file_name",
        "year",
        "site",
        "annotator_views",
        "ground_truth_instances_across_views",
        "predicted_instances_per_observation",
        "tp_edges",
        "tp_iou_sum",
        "fp",
        "fn",
        "pq",
    )
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "file_name": row.file_name,
                    "year": row.year,
                    "site": row.site,
                    "annotator_views": row.annotator_views,
                    "ground_truth_instances_across_views": row.gt_count,
                    "predicted_instances_per_observation": len(
                        predictions.get(row.file_name, [])
                    ),
                    "tp_edges": row.tp,
                    "tp_iou_sum": f"{row.tp_iou_sum:.9f}",
                    "fp": row.fp,
                    "fn": row.fn,
                    "pq": f"{row.pq:.9f}",
                }
            )


def _write_overlays(
    dataset: CompetitionDataset,
    predictions: dict[str, list[PredictedInstance]],
    output_dir: Path,
    maximum: int,
) -> list[str]:
    names = sorted(predictions)
    if not names or maximum <= 0:
        return []
    if len(names) <= maximum:
        selected = names
    elif maximum == 1:
        selected = [names[len(names) // 2]]
    else:
        indices = {round(index * (len(names) - 1) / (maximum - 1)) for index in range(maximum)}
        selected = [names[index] for index in sorted(indices)]
    output_dir.mkdir(parents=True, exist_ok=True)
    for file_name in selected:
        gray = read_grayscale(dataset.train_image_dir / file_name)
        output_path = output_dir / f"{Path(file_name).stem}_prediction.png"
        write_prediction_overlay(gray, predictions[file_name], output_path)
    return selected


def _git_status(repo_root: Path) -> dict[str, Any]:
    info = git_info(repo_root)
    info["source_tree_sha256"] = _source_fingerprint(repo_root)
    return info


def run_validation(
    *,
    repo_root: str | Path,
    data_root: str | Path | None = None,
    output_root: str | Path | None = None,
    limit_observations: int | None = None,
    overlay_count: int = 12,
) -> dict[str, Any]:
    run_started = time.perf_counter()
    repo_root = Path(repo_root).resolve()
    kaggle_runtime = _is_kaggle_runtime()
    dataset_root = _resolve_data_root(data_root, repo_root)
    if not kaggle_runtime and not _within(dataset_root, repo_root / "data"):
        raise ValueError(
            f"Local competition data must stay under data/: {dataset_root}"
        )

    if output_root is None:
        output_root = (
            Path("/kaggle/working/filament-validation-baseline")
            if kaggle_runtime
            else repo_root / "artifacts" / "validation-baseline"
        )
    output_root = Path(output_root)
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    output_root = output_root.resolve()
    if not kaggle_runtime and not _within(output_root, repo_root):
        raise ValueError(f"Local run output must stay inside the repository: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    config_json, baseline_config = _load_config(repo_root)
    dataset = load_dataset(dataset_root)
    input_inventory = inventory(dataset)
    if input_inventory["observation_count"] != 707:
        raise ValueError(
            "Expected 707 competition training observations, found "
            f"{input_inventory['observation_count']}"
        )
    if input_inventory["annotator_view_count"] != 1154:
        raise ValueError(
            "Expected 1,154 annotator views, found "
            f"{input_inventory['annotator_view_count']}"
        )
    if input_inventory["annotation_count"] != 8199:
        raise ValueError(
            f"Expected 8,199 training annotations, found {input_inventory['annotation_count']}"
        )

    split = create_observation_split(
        dataset,
        repo_root / "data" / "splits",
        seed=int(config_json["seed"]),
        validation_fraction=float(config_json["validation_fraction"]),
    )
    expected_split_sha256 = config_json.get("expected_split_sha256")
    if expected_split_sha256 and split.sha256 != expected_split_sha256:
        raise RuntimeError(
            "Observation split digest differs from the frozen baseline: "
            f"expected {expected_split_sha256}, got {split.sha256}"
        )
    validation_names = list(split.validation_names)
    full_holdout_count = len(validation_names)
    if limit_observations is not None:
        if limit_observations < 1:
            raise ValueError("limit_observations must be positive")
        validation_names = validation_names[:limit_observations]

    hardware = hardware_info()
    predictions: dict[str, list[PredictedInstance]] = {}
    prediction_start = time.perf_counter()
    for index, file_name in enumerate(validation_names, start=1):
        image = read_grayscale(dataset.train_image_dir / file_name)
        predictions[file_name] = predict_image(image, file_name, baseline_config)
        if index % 20 == 0 or index == len(validation_names):
            print(f"Predicted {index}/{len(validation_names)} validation observations")
    prediction_seconds = time.perf_counter() - prediction_start

    scorer_start = time.perf_counter()
    metric_result, observation_rows = score_validation(
        dataset,
        predictions,
        validation_names,
        iou_threshold=float(config_json["iou_threshold"]),
        diagnostic_overlap_threshold=float(config_json["diagnostic_overlap_threshold"]),
    )
    scoring_seconds = time.perf_counter() - scorer_start

    _write_predictions(output_root / "predictions.csv", predictions)
    _write_observation_scores(
        output_root / "per_observation.csv", observation_rows, predictions
    )
    overlay_names = _write_overlays(
        dataset, predictions, output_root / "overlays", overlay_count
    )
    split_balance: dict[str, dict[str, int]] = {"year": {}, "site": {}}
    for file_name in split.validation_names:
        view = dataset.views_by_observation[file_name][0]
        split_balance["year"][str(view.year)] = (
            split_balance["year"].get(str(view.year), 0) + 1
        )
        split_balance["site"][view.site] = split_balance["site"].get(view.site, 0) + 1

    total_seconds = time.perf_counter() - run_started
    try:
        import torch

        peak_gpu_bytes = (
            int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available()
            else 0
        )
    except ImportError:
        peak_gpu_bytes = 0

    run_result: dict[str, Any] = {
        "status": "complete" if limit_observations is None else "smoke_subset",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository": _git_status(repo_root),
        "kaggle_notebook_version": os.environ.get(
            "KAGGLE_KERNEL_VERSION", os.environ.get("EQUINEXUS_NOTEBOOK_VERSION")
        ),
        "data_root": str(dataset.root),
        "output_root": str(output_root),
        "dataset_inventory": input_inventory,
        "split": {
            "manifest": str(split.manifest_path),
            "sha256": split.sha256,
            "matches_expected_sha256": (
                split.sha256 == expected_split_sha256
                if expected_split_sha256
                else None
            ),
            "seed": int(config_json["seed"]),
            "validation_fraction": float(config_json["validation_fraction"]),
            "stratification": config_json["stratify_by"],
            "training_observations": len(split.training_names),
            "validation_observations": full_holdout_count,
            "evaluated_observations": len(validation_names),
            "validation_balance": split_balance,
        },
        "baseline": {
            **config_json,
            "name": "local_contrast_otsu_connected_components",
            "trained": False,
            "device_used": "cpu",
        },
        "scorer": {
            "name": "organizer_self_evaluation_notebook_semantics",
            "reference_url": OFFICIAL_SCORER_URL,
            "iou_threshold_is_strict": True,
            "one_to_one_assignment": False,
            "predictions_reused_for_each_annotator_view": True,
        },
        "runtime": {
            **hardware,
            "cuda_used_by_baseline": False,
            "peak_gpu_memory_bytes": peak_gpu_bytes,
            "prediction_seconds": prediction_seconds,
            "scoring_seconds": scoring_seconds,
            "total_seconds": total_seconds,
            "package_versions": package_versions(),
        },
        "metrics": metric_result,
        "prediction_instance_count": sum(len(items) for items in predictions.values()),
        "overlay_observations": overlay_names,
        "artifacts": {
            "predictions_csv": "predictions.csv",
            "per_observation_csv": "per_observation.csv",
            "overlays_dir": "overlays",
        },
    }
    (output_root / "metrics.json").write_text(
        json.dumps(run_result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "Validation PQ: "
        f"{metric_result['global']['pq']:.6f} "
        f"({metric_result['global']['tp']} TP edges, "
        f"{metric_result['global']['fp']} FP, {metric_result['global']['fn']} FN)"
    )
    print(f"Split SHA-256: {split.sha256}")
    print(f"Run artifacts: {output_root}")
    return run_result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the fixed grouped validation baseline.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--limit-observations", type=int, default=None)
    parser.add_argument("--overlay-count", type=int, default=12)
    args = parser.parse_args(argv)
    result = run_validation(
        repo_root=args.repo_root,
        data_root=args.data_root,
        output_root=args.output_root,
        limit_observations=args.limit_observations,
        overlay_count=args.overlay_count,
    )
    print(json.dumps(result["metrics"]["global"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
