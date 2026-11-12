# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Evaluate a public U-Net and YOLO family-vote fusion on grouped data partitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_component_selector_probability_blend_holdout import paired_bootstrap
from score_oof_family_vote import as_predictions, load_detector, load_unet
from filament_baseline.data import load_dataset
from filament_baseline.predictor import encode_binary_mask
from filament_baseline.runtime import hardware_info, package_versions
from filament_baseline.runner import _write_predictions
from filament_baseline.scorer import score_validation
from filament_baseline.split import create_observation_split
from filament_baseline.visuals import write_comparison_overlays

EXPECTED_SPLIT = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"
SHAPE = (2048, 2048)
DEFAULT_UNET = ROOT / "artifacts/model-experiments/dariushafshar-unet-fold-aware-holdout-20261111/validation_predictions.csv"
DEFAULT_DETECTOR = ROOT / "artifacts/detector-experiments/yolo11s-seg-1280-oof-holdout-ensemble-tta8/dedup-iou-030/oof_detections.json"
DEFAULT_INCUMBENT = ROOT / "artifacts/model-experiments/crossfold-medoid-unet-fusion-20261109/incumbent-per-observation.csv"
DEFAULT_UNET_OBSERVATIONS = ROOT / "artifacts/model-experiments/dariushafshar-unet-fold-aware-holdout-20261111/run-metadata.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--unet-predictions", type=Path, default=DEFAULT_UNET)
    parser.add_argument("--detector-predictions", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--incumbent-observations", type=Path, default=DEFAULT_INCUMBENT)
    parser.add_argument("--unet-observations", type=Path, default=DEFAULT_UNET_OBSERVATIONS)
    parser.add_argument("--partition", choices=("holdout", "oof", "test"), default="holdout")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/model-experiments/public-unet-yolo-family-vote-holdout-20261111",
    )
    parser.add_argument("--detector-confidence", type=float, default=0.35)
    parser.add_argument("--minimum-family-votes", type=int, choices=(1, 2), default=2)
    parser.add_argument("--strict-support-iou", type=float, default=None)
    parser.add_argument("--skip-paired-bootstrap", action="store_true", help="Skip baseline comparisons for screening sweeps.")
    parser.add_argument("--skip-overlays", action="store_true", help="Skip visual outputs for screening sweeps.")
    args = parser.parse_args()

    started = time.perf_counter()
    paths = {
        "data_root": args.data_root.expanduser().resolve(),
        "unet_predictions": args.unet_predictions.expanduser().resolve(),
        "detector_predictions": args.detector_predictions.expanduser().resolve(),
        "incumbent_observations": args.incumbent_observations.expanduser().resolve(),
        "unet_observations": args.unet_observations.expanduser().resolve(),
        "output_dir": args.output_dir.expanduser().resolve(),
    }
    for key, path in paths.items():
        if not path.is_relative_to(ROOT):
            raise ValueError(f"{key} must be inside this repository: {path}")
    for key in ("data_root", "unet_predictions", "detector_predictions", "incumbent_observations", "unet_observations"):
        if not paths[key].exists():
            raise FileNotFoundError(paths[key])
    if paths["output_dir"].exists() and any(paths["output_dir"].iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {paths['output_dir']}")
    if not 0.0 <= args.detector_confidence <= 1.0:
        raise ValueError("detector confidence must be between zero and one")
    if args.strict_support_iou is not None and not 0.0 <= args.strict_support_iou <= 1.0:
        raise ValueError("strict support IoU must be between zero and one")

    dataset = load_dataset(paths["data_root"])
    split = create_observation_split(dataset, ROOT / "data/splits", seed=42, validation_fraction=0.2)
    if split.sha256 != EXPECTED_SPLIT:
        raise RuntimeError(f"Frozen split changed: {split.sha256}")
    if args.partition == "holdout":
        names = tuple(sorted(split.validation_names))
    elif args.partition == "oof":
        names = tuple(sorted(split.training_names))
        if set(names) != set(split.training_names) or set(names) & set(split.validation_names):
            raise ValueError("OOF scoring must cover only the frozen training partition")
    else:
        names = tuple(sorted(path.name for path in dataset.test_image_dir.iterdir() if path.is_file()))
        if len(names) != 180:
            raise ValueError(f"Expected all 180 supplied test H-alpha images, found {len(names)}")
    stems = {Path(name).stem for name in names}
    unet = load_unet(paths["unet_predictions"], stems)
    detector = load_detector(paths["detector_predictions"], args.detector_confidence, stems)
    sources = {"public_unet": unet, "yolo11s": detector}
    family_order = ["public_unet", "yolo11s"]

    voted: dict[str, list[dict]] = {}
    vote_histogram: dict[str, int] = {}
    for name in names:
        stem = Path(name).stem
        candidates = []
        for family in family_order:
            for item in sources[family][stem]:
                item["family"] = family
                candidates.append(item)
        other_rles = {
            family: [item["rle"] for item in sources[family][stem]]
            for family in family_order
        }
        for item in candidates:
            family = item["family"]
            support_ious = []
            for other in family_order:
                if other == family or not other_rles[other]:
                    continue
                overlaps = mask_utils.iou([item["rle"]], other_rles[other], [0] * len(other_rles[other]))
                best_iou = float(np.max(overlaps)) if len(overlaps) else 0.0
                if best_iou > 0.5:
                    support_ious.append(best_iou)
            item["votes"] = 1 + len(support_ious)
            item["support_ious"] = support_ious

        priority = {family: index for index, family in enumerate(family_order)}
        candidates.sort(key=lambda item: (priority[item["family"]], -item["score"], item["index"]))
        selected = []
        taken = np.zeros(SHAPE, dtype=bool)
        for item in candidates:
            if item["votes"] < args.minimum_family_votes:
                continue
            if (
                args.strict_support_iou is not None
                and item["votes"] == args.minimum_family_votes
                and (not item["support_ious"] or min(item["support_ious"]) < args.strict_support_iou)
            ):
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
            fallback = max(
                candidates,
                key=lambda item: (item["votes"], -priority[item["family"]], item["score"], -item["index"]),
            )
            selected = [fallback["rle"]]
            vote_histogram["fallback_images"] = vote_histogram.get("fallback_images", 0) + 1
        voted[stem] = selected
        for item in candidates:
            key = str(item["votes"])
            vote_histogram[key] = vote_histogram.get(key, 0) + 1

    predictions = {name: as_predictions(Path(name).stem, voted[Path(name).stem]) for name in names}
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    output_name = {
        "holdout": "validation_predictions.csv",
        "oof": "oof_predictions.csv",
        "test": "submission.csv",
    }[args.partition]
    predictions_path = paths["output_dir"] / output_name
    _write_predictions(predictions_path, predictions)

    observation_rows = []
    comparisons = {}
    overlays = []
    metrics = None
    if args.partition in {"holdout", "oof"}:
        metrics, observation_rows = score_validation(
            dataset, predictions, names, iou_threshold=0.5, diagnostic_overlap_threshold=0.0
        )
        if not args.skip_overlays:
            overlays = write_comparison_overlays(dataset, predictions, paths["output_dir"] / "overlays", maximum=12)
        if not args.skip_paired_bootstrap:
            candidate_rows = [
                SimpleNamespace(file_name=row.file_name, tp_iou_sum=row.tp_iou_sum, tp=row.tp, fp=row.fp, fn=row.fn)
                for row in observation_rows
            ]
        if not args.skip_paired_bootstrap and args.partition == "holdout":
            for label, comparison_path in (
                ("public_unet", paths["unet_observations"]),
                ("incumbent", paths["incumbent_observations"]),
            ):
                source = json.loads(comparison_path.read_text(encoding="utf-8")) if comparison_path.suffix == ".json" else None
                if source is not None:
                    baseline_rows = [
                        SimpleNamespace(file_name=row["file_name"], tp_iou_sum=row["tp_iou_sum"], tp=row["tp"], fp=row["fp"], fn=row["fn"])
                        for row in source["per_observation"]
                    ]
                else:
                    frame = pd.read_csv(comparison_path)
                    baseline_rows = [
                        SimpleNamespace(file_name=row.file_name, tp_iou_sum=row.tp_iou_sum, tp=row.tp_edges, fp=row.fp, fn=row.fn)
                        for row in frame.itertuples(index=False)
                    ]
                comparisons[label] = paired_bootstrap(baseline_rows, candidate_rows, resamples=20_000, seed=42)
        elif not args.skip_paired_bootstrap:
            for label in ("public_unet", "yolo11s"):
                baseline_predictions = {
                    name: as_predictions(
                        Path(name).stem,
                        [item["rle"] for item in sources[label][Path(name).stem]],
                    )
                    for name in names
                }
                _, baseline_observation_rows = score_validation(
                    dataset, baseline_predictions, names,
                    iou_threshold=0.5, diagnostic_overlap_threshold=0.0,
                )
                baseline_rows = [
                    SimpleNamespace(file_name=row.file_name, tp_iou_sum=row.tp_iou_sum, tp=row.tp, fp=row.fp, fn=row.fn)
                    for row in baseline_observation_rows
                ]
                comparisons[label] = paired_bootstrap(baseline_rows, candidate_rows, resamples=20_000, seed=42)

    per_observation = [
        {
            "file_name": row.file_name,
            "year": row.year,
            "site": row.site,
            "annotator_views": row.annotator_views,
            "gt_count": row.gt_count,
            "prediction_count": row.prediction_count,
            "tp": row.tp,
            "tp_iou_sum": row.tp_iou_sum,
            "fp": row.fp,
            "fn": row.fn,
            "pq": row.pq,
        }
        for row in observation_rows
    ]
    elapsed = time.perf_counter() - started
    metadata = {
        "run_name": paths["output_dir"].name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_families": family_order,
        "licenses": {
            "public_unet": "CC BY-NC-SA 4.0; attribution and ShareAlike terms apply",
            "yolo11s": "AGPL-3.0 code and weights",
        },
        "training_data_boundary": "The public U-Net uses only source-fold checkpoints that excluded each scored observation; YOLO models were trained on the outer training partition. No public MAGFiLO test labels were used.",
        "evaluation_partition": {
            "holdout": "frozen observation-grouped 20% holdout",
            "oof": "cross-fitted competition training observations",
            "test": "competition supplied test images",
        }[args.partition],
        "holdout_used_for_training": False,
        "holdout_used_for_model_or_threshold_selection": False,
        "test_ground_truth_used": False,
        "fusion_rule": {
            "minimum_family_votes": args.minimum_family_votes,
            "support_iou_strictly_greater_than": 0.5,
            "strict_support_iou": args.strict_support_iou,
            "overlap_clipping": "keep at least 50% of each candidate mask area after higher-priority masks",
            "priority": family_order,
            "empty_observation_fallback": "highest vote count, then public U-Net, detector confidence, stable index",
        },
        "confidence_thresholds": {"public_unet": "upstream nested OOF rule", "yolo11s": args.detector_confidence},
        "detector_inference": f"five grouped YOLO11s models, eight D4 transforms, duplicate masks suppressed at mask IoU 0.3 before confidence filtering at {args.detector_confidence:g}",
        "fixed_holdout_split_sha256": split.sha256,
        "evaluation_observations": len(names),
        "evaluation_annotator_views": sum(row.annotator_views for row in observation_rows),
        "metrics": metrics,
        "paired_bootstrap_vs": comparisons,
        "paired_bootstrap_skipped": args.skip_paired_bootstrap,
        "vote_histogram": vote_histogram,
        "per_observation": per_observation,
        "source_files": {
            label: {"path": str(path), "sha256": sha256(path)}
            for label, path in (
                ("public_unet_predictions", paths["unet_predictions"]),
                ("yolo11s_predictions", paths["detector_predictions"]),
                ("public_unet_observations", paths["unet_observations"]),
                ("incumbent_observations", paths["incumbent_observations"]),
            )
        },
        "output_predictions": {"path": str(predictions_path), "sha256": sha256(predictions_path)},
        "overlays": overlays,
        "hardware": hardware_info(),
        "package_versions": package_versions(),
        "runtime_seconds": elapsed,
        "source_commit": __import__("subprocess").run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip(),
        "source_script_sha256": sha256(Path(__file__).resolve()),
    }
    metadata_path = paths["output_dir"] / "run-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "event": "complete",
        "official_pq": metrics["global"]["pq"] if metrics is not None else None,
        "predicted_instances": sum(len(items) for items in predictions.values()),
        "observations_with_predictions": sum(bool(items) for items in predictions.values()),
        "paired_bootstrap_vs": comparisons,
        "metadata": str(metadata_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
