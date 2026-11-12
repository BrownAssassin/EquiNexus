# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Evaluate cross-fold medoid association with EquiNexus detector and U-Net outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from deduplicate_yolo_tta_oof import _deduplicate  # noqa: E402
from evaluate_component_selector_probability_blend_holdout import paired_bootstrap, per_observation_rows  # noqa: E402
from run_component_selector_oof import load_prediction_rows, load_probability_maps  # noqa: E402
from filament_baseline.data import load_dataset  # noqa: E402
from filament_baseline.detector_fusion import DetectorFusionConfig, fuse_detector_instances  # noqa: E402
from filament_baseline.predictor import PredictedInstance, encode_binary_mask  # noqa: E402
from filament_baseline.runner import _write_predictions  # noqa: E402
from filament_baseline.scorer import score_validation  # noqa: E402
from filament_baseline.split import create_observation_split  # noqa: E402


FROZEN_SPLIT_SHA256 = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"
DEFAULT_DATA_ROOT = REPO_ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026"
DEFAULT_HOLDOUT_FOLDS = REPO_ROOT / "artifacts/detector-experiments/yolo11s-seg-1280-oof-holdout-ensemble-tta8"
DEFAULT_PERANN_ROOT = REPO_ROOT / "artifacts/kaggle/fold-ensemble/remote-v3-output/EquiNexus/oof-fivefold-mean-v1/probabilities/holdout"
DEFAULT_CONSENSUS_ROOT = REPO_ROOT / "artifacts/model-experiments/oof-consensus512-foldmean-valid-holdout/probabilities"
DEFAULT_BASELINE = REPO_ROOT / "artifacts/model-experiments/component-selector-rf-blend-holdout-20261109-final/candidate-predictions.csv"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/model-experiments/crossfold-medoid-unet-fusion-20261109"
FOLD_ORDER = (1, 0, 3, 2, 4)
DEDUP_IOU_THRESHOLD = 0.30
ASSOCIATION_IOU_THRESHOLD = 0.15
MINIMUM_FOLD_SUPPORT = 3
MINIMUM_MEDOID_AREA = 96
MINIMUM_CLEAN_FRACTION = 0.65


@dataclass(frozen=True)
class FoldMask:
    fold: int
    score: float
    counts: str | bytes


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_rle(mask: FoldMask) -> dict[str, Any]:
    counts = mask.counts.encode("ascii") if isinstance(mask.counts, str) else mask.counts
    return {"size": [2048, 2048], "counts": counts}


def iou_matrix(left: list[FoldMask], right: list[FoldMask]) -> np.ndarray:
    if not left or not right:
        return np.zeros((len(left), len(right)), dtype=np.float64)
    return np.asarray(
        mask_utils.iou([as_rle(item) for item in left], [as_rle(item) for item in right], [0] * len(right)),
        dtype=np.float64,
    )


def medoid(cluster: list[FoldMask]) -> FoldMask:
    if not cluster:
        raise ValueError("Cannot select a medoid from an empty cluster")
    if len(cluster) == 1:
        return cluster[0]
    agreements = iou_matrix(cluster, cluster).mean(axis=1)
    best = max(range(len(cluster)), key=lambda index: (float(agreements[index]), cluster[index].score))
    return cluster[best]


def associate_folds(
    fold_detections: dict[int, list[dict[str, Any]]],
    *,
    match_iou: float = ASSOCIATION_IOU_THRESHOLD,
) -> tuple[list[list[FoldMask]], dict[int, int]]:
    """Cluster one fold at a time so each model contributes at most one vote."""
    clusters: list[list[FoldMask]] = []
    for fold in FOLD_ORDER:
        candidates = [
            FoldMask(fold=fold, score=float(item["score"]), counts=item["counts"])
            for item in sorted(fold_detections.get(fold, []), key=lambda item: -float(item["score"]))
        ]
        if not candidates:
            continue
        if not clusters:
            clusters = [[candidate] for candidate in candidates]
            continue
        representatives = [medoid(cluster) for cluster in clusters]
        overlaps = iou_matrix(representatives, candidates)
        matched: set[int] = set()
        if overlaps.size:
            cluster_indices, candidate_indices = linear_sum_assignment(-overlaps)
            for cluster_index, candidate_index in zip(cluster_indices, candidate_indices, strict=True):
                if overlaps[cluster_index, candidate_index] >= match_iou:
                    clusters[int(cluster_index)].append(candidates[int(candidate_index)])
                    matched.add(int(candidate_index))
        for index, candidate in enumerate(candidates):
            if index in matched:
                continue
            maximum_overlap = float(overlaps[:, index].max()) if overlaps.size else 0.0
            if maximum_overlap < match_iou:
                clusters.append([candidate])
    support_histogram: dict[int, int] = {}
    for cluster in clusters:
        support_histogram[len(cluster)] = support_histogram.get(len(cluster), 0) + 1
    return clusters, support_histogram


def consensus_detections(
    fold_detections: dict[int, list[dict[str, Any]]],
    *,
    match_iou: float = ASSOCIATION_IOU_THRESHOLD,
    minimum_support: int = MINIMUM_FOLD_SUPPORT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    clusters, support_histogram = associate_folds(fold_detections, match_iou=match_iou)
    accepted = [cluster for cluster in clusters if len(cluster) >= minimum_support]
    fallback = False
    if not accepted and clusters:
        accepted = [max(clusters, key=lambda cluster: (len(cluster), np.mean([item.score for item in cluster])))]
        fallback = True

    candidates: list[tuple[int, float, FoldMask]] = []
    for cluster in accepted:
        representative = medoid(cluster)
        average_score = float(np.mean([item.score for item in cluster]))
        candidates.append((len(cluster), average_score, representative))

    # Resolve overlaps by favoring support, then mean detector confidence.
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    claimed = np.zeros((2048, 2048), dtype=bool)
    output: list[dict[str, Any]] = []
    output_support: list[int] = []
    for support, average_score, representative in candidates:
        decoded = mask_utils.decode(as_rle(representative)).astype(bool)
        clean = decoded & ~claimed
        area = int(clean.sum())
        clean_fraction = area / max(int(decoded.sum()), 1)
        if area < MINIMUM_MEDOID_AREA or clean_fraction < MINIMUM_CLEAN_FRACTION:
            continue
        rle = encode_binary_mask(clean)
        output.append({
            "score": float(support) + average_score * 1e-3,
            "counts": rle["counts"],
            "fold_support": int(support),
            "mean_fold_score": average_score,
        })
        output_support.append(int(support))
        claimed |= clean

    metadata = {
        "cluster_count": len(clusters),
        "support_histogram": support_histogram,
        "accepted_cluster_count_before_overlap_cleanup": len(accepted),
        "retained_instance_count": len(output),
        "retained_support": output_support,
        "fallback_used": fallback,
    }
    return output, metadata


def predictions_from_labels(file_name: str, labels: np.ndarray) -> list[PredictedInstance]:
    stem = Path(file_name).stem
    identifiers = np.unique(labels)
    return [
        PredictedInstance(
            filament_id=f"{stem}_{int(instance_id)}",
            image_stem=stem,
            area=int(np.count_nonzero(labels == instance_id)),
            rle=encode_binary_mask(labels == instance_id),
        )
        for instance_id in identifiers
        if int(instance_id) > 0
    ]


def predictions_from_rows(
    names: list[str], rows_by_stem: dict[str, list[dict[str, Any]]]
) -> dict[str, list[PredictedInstance]]:
    output: dict[str, list[PredictedInstance]] = {}
    for file_name in names:
        stem = Path(file_name).stem
        output[file_name] = [
            PredictedInstance(
                filament_id=f"{stem}_{index}",
                image_stem=stem,
                area=int(mask_utils.area(rle)),
                rle=rle,
            )
            for index, rle in enumerate(rows_by_stem[stem], start=1)
        ]
    return output


def load_fold_files(root: Path, stems: set[str]) -> tuple[dict[int, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    metadata_path = root / "run-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    if metadata.get("status") != "complete" or metadata.get("partition") != "frozen_validation_holdout":
        raise ValueError("Detector predictions are not a complete frozen-holdout run")
    if metadata.get("frozen_holdout_sha256") != FROZEN_SPLIT_SHA256 or metadata.get("holdout_used_for_training") is not False:
        raise ValueError("Detector checkpoint lineage is not holdout-free")
    output: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for fold in range(5):
        path = root / f"fold-{fold}-predictions.json"
        rows = json.loads(path.read_text(encoding="utf-8-sig"))
        if set(rows) != stems:
            raise ValueError(f"Fold {fold} prediction coverage does not match the frozen holdout")
        output[fold] = {
            stem: _deduplicate(detections, DEDUP_IOU_THRESHOLD)
            for stem, detections in rows.items()
        }
    return output, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--fold-predictions", type=Path, default=DEFAULT_HOLDOUT_FOLDS)
    parser.add_argument("--perann-probabilities", type=Path, default=DEFAULT_PERANN_ROOT)
    parser.add_argument("--consensus-probabilities", type=Path, default=DEFAULT_CONSENSUS_ROOT)
    parser.add_argument("--baseline-predictions", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--match-iou", type=float, default=ASSOCIATION_IOU_THRESHOLD)
    parser.add_argument("--minimum-support", type=int, default=MINIMUM_FOLD_SUPPORT)
    args = parser.parse_args()

    data_root, fold_root = args.data_root.resolve(), args.fold_predictions.resolve()
    perann_root, consensus_root = args.perann_probabilities.resolve(), args.consensus_probabilities.resolve()
    baseline_path, output_dir = args.baseline_predictions.resolve(), args.output_dir.resolve()
    if not data_root.is_relative_to((REPO_ROOT / "data").resolve()):
        raise ValueError("Competition data must remain under data/")
    for path, label in (
        (fold_root, "fold predictions"), (perann_root, "per-annotator probabilities"),
        (consensus_root, "consensus probabilities"), (baseline_path, "incumbent predictions"),
    ):
        if not path.is_relative_to((REPO_ROOT / "artifacts").resolve()):
            raise ValueError(f"{label} must remain under artifacts/")
    if not output_dir.is_relative_to((REPO_ROOT / "artifacts/model-experiments").resolve()):
        raise ValueError("Outputs must remain under artifacts/model-experiments/")
    if not 0.0 <= args.match_iou <= 1.0 or args.minimum_support not in range(1, 6):
        raise ValueError("Invalid cross-fold association policy")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)

    started = time.perf_counter()
    dataset = load_dataset(data_root)
    split = create_observation_split(dataset, REPO_ROOT / "data/splits")
    if split.sha256 != FROZEN_SPLIT_SHA256:
        raise RuntimeError("Frozen grouped validation split changed")
    names = sorted(split.validation_names)
    stems = {Path(name).stem for name in names}
    fold_files, fold_metadata = load_fold_files(fold_root, stems)
    baseline_metrics_path = REPO_ROOT / "artifacts/model-experiments/component-selector-rf-blend-holdout-20261109-final/metrics.json"
    baseline_metrics = json.loads(baseline_metrics_path.read_text(encoding="utf-8-sig"))
    if baseline_metrics.get("split_sha256") != split.sha256 or baseline_metrics.get("test_images_or_labels_used") is not False:
        raise RuntimeError("Incumbent holdout reference does not match the frozen split")

    baseline_masks = load_prediction_rows(baseline_path, stems)
    baseline_predictions = predictions_from_rows(names, baseline_masks)
    consensus_predictions: dict[str, list[PredictedInstance]] = {}
    instance_diagnostics: dict[str, dict[str, Any]] = {}
    for index, file_name in enumerate(names, start=1):
        stem = Path(file_name).stem
        per_fold = {fold: fold_files[fold][stem] for fold in range(5)}
        detections, diagnostics = consensus_detections(
            per_fold, match_iou=args.match_iou, minimum_support=args.minimum_support
        )
        foreground, _ = load_probability_maps(
            file_name,
            fold=None,
            first_root=perann_root,
            second_root=consensus_root,
        )
        labels = fuse_detector_instances(
            foreground,
            detections,
            DetectorFusionConfig(
                foreground_threshold=0.50,
                detection_threshold=0.30,
                grow_radius=4,
                minimum_claimed_area=200,
                closing_radius=3,
                solar_radius=950,
            ),
        )
        consensus_predictions[file_name] = predictions_from_labels(file_name, labels)
        instance_diagnostics[stem] = {
            **diagnostics,
            "raw_fold_detection_count": [len(fold_files[fold][stem]) for fold in range(5)],
            "consensus_detection_count": len(detections),
            "fused_instance_count": len(consensus_predictions[file_name]),
        }
        if index % 20 == 0 or index == len(names):
            print(f"Fused cross-fold consensus {index}/{len(names)} holdout observations", flush=True)

    incumbent_metrics, incumbent_rows = score_validation(dataset, baseline_predictions, names)
    candidate_metrics, candidate_rows = score_validation(dataset, consensus_predictions, names)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_predictions(output_dir / "incumbent-predictions.csv", baseline_predictions)
    _write_predictions(output_dir / "candidate-predictions.csv", consensus_predictions)
    pd.DataFrame(per_observation_rows(incumbent_rows)).to_csv(output_dir / "incumbent-per-observation.csv", index=False)
    pd.DataFrame(per_observation_rows(candidate_rows)).to_csv(output_dir / "candidate-per-observation.csv", index=False)
    pd.DataFrame([
        {
            "file_name": Path(name).name,
            "incumbent_pq": base.pq,
            "candidate_pq": candidate.pq,
            "delta_pq": candidate.pq - base.pq,
            "candidate_predictions": len(consensus_predictions[name]),
            **instance_diagnostics[Path(name).stem],
        }
        for name, base, candidate in zip(names, incumbent_rows, candidate_rows, strict=True)
    ]).to_json(output_dir / "per-observation-diagnostics.jsonl", orient="records", lines=True)

    try:
        source_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        source_commit = "unknown"
    metadata = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "source_script_sha256": sha256(Path(__file__).resolve()),
        "split_sha256": split.sha256,
        "oof_manifest_sha256": fold_metadata.get("oof_manifest_sha256"),
        "holdout_observations": len(names),
        "holdout_used_for_training": False,
        "test_images_or_labels_used": False,
        "training_labels": "competition-provided training annotations only",
        "public_reference": {
            "name": "Filament V23 final reproducibility package",
            "reported_strict_oof_pq": 0.43520179088181965,
            "reported_public_score": 0.39,
            "method_adapted": "cross-fold Hungarian association, support threshold, and medoid mask selection",
            "reference_checkpoints_or_code_used": False,
            "reference_license_status": "public reference weights are eligible under the user's policy; retain upstream license, attribution, and ShareAlike terms, and do not reuse results without a valid fold-level evaluation",
        },
        "candidate_policy": {
            "selection": "fixed public-method policy; no holdout search",
            "within_fold_tta_dedup_iou": DEDUP_IOU_THRESHOLD,
            "fold_association_iou": args.match_iou,
            "minimum_fold_support": args.minimum_support,
            "medoid_tie_break": "mean fold confidence",
            "minimum_medoid_area": MINIMUM_MEDOID_AREA,
            "minimum_nonoverlap_fraction": MINIMUM_CLEAN_FRACTION,
            "unet_fusion": {
                "foreground_threshold": 0.50,
                "detector_threshold": 0.30,
                "grow_radius": 4,
                "minimum_claimed_area": 200,
                "closing_radius": 3,
                "solar_radius": 950,
                "foreground_blend": "25% five-fold per-annotator U-Net + 75% five-fold consensus U-Net",
            },
        },
        "incumbent": {
            "run": "component-selector-rf-blend-holdout-20261109-final",
            "global": incumbent_metrics["global"],
        },
        "candidate": {
            "global": candidate_metrics["global"],
            "paired_bootstrap_vs_incumbent": paired_bootstrap(
                incumbent_rows, candidate_rows, resamples=20_000, seed=42
            ),
        },
        "fold_consensus_summary": {
            "observation_count": len(instance_diagnostics),
            "mean_consensus_detection_count": float(np.mean([row["consensus_detection_count"] for row in instance_diagnostics.values()])),
            "mean_fused_instance_count": float(np.mean([row["fused_instance_count"] for row in instance_diagnostics.values()])),
            "fallback_observation_count": int(sum(bool(row["fallback_used"]) for row in instance_diagnostics.values())),
        },
        "runtime_seconds": time.perf_counter() - started,
        "hardware": {"device": "CPU for association/scoring; source detector predictions were generated on the RTX 5070 Laptop GPU"},
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "input_sha256": {
            "fold_run_metadata": sha256(fold_root / "run-metadata.json"),
            "fold_prediction_files": {str(fold): sha256(fold_root / f"fold-{fold}-predictions.json") for fold in range(5)},
            "perann_probability_manifest": sha256(perann_root.parent.parent / "metrics.json"),
            "consensus_probability_manifest": sha256(consensus_root.parent / "metrics.json"),
            "incumbent_predictions": sha256(baseline_path),
            "incumbent_metrics": sha256(baseline_metrics_path),
        },
        "artifacts": {
            "candidate_predictions": "candidate-predictions.csv",
            "incumbent_predictions": "incumbent-predictions.csv",
            "per_observation": "per-observation-diagnostics.jsonl",
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "incumbent_pq": incumbent_metrics["global"]["pq"],
        "candidate_pq": candidate_metrics["global"]["pq"],
        "candidate_vs_incumbent": metadata["candidate"]["paired_bootstrap_vs_incumbent"],
        "output_dir": str(output_dir),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
