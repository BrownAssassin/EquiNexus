from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
from pycocotools import mask as mask_utils

from .data import CompetitionDataset, ImageView
from .predictor import PredictedInstance


@dataclass
class ObservationCounts:
    file_name: str
    year: int
    site: str
    annotator_views: int = 0
    gt_count: int = 0
    prediction_count: int = 0
    tp_ious: list[float] = field(default_factory=list)
    fp: int = 0
    fn: int = 0

    @property
    def tp(self) -> int:
        return len(self.tp_ious)

    @property
    def tp_iou_sum(self) -> float:
        return float(sum(self.tp_ious))

    @property
    def pq(self) -> float:
        return pq_value(self.tp_ious, self.fp, self.fn)


@dataclass
class ScorerDiagnostics:
    positive_ious: list[float] = field(default_factory=list)
    positive_dices: list[float] = field(default_factory=list)
    gt_match_degrees: Counter[int] = field(default_factory=Counter)
    prediction_match_degrees: Counter[int] = field(default_factory=Counter)


def _coco_rle(rle: dict[str, Any]) -> dict[str, Any]:
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = counts.encode("ascii")
    return {"size": [int(rle["size"][0]), int(rle["size"][1])], "counts": counts}


def annotation_to_rle(annotation: dict[str, Any], height: int, width: int) -> dict[str, Any]:
    segmentation = annotation["segmentation"]
    if isinstance(segmentation, dict):
        return _coco_rle(segmentation)
    if not isinstance(segmentation, list) or not segmentation:
        raise ValueError(f"Unsupported or empty segmentation for annotation {annotation.get('id')}")
    rles = mask_utils.frPyObjects(segmentation, height, width)
    merged = mask_utils.merge(rles) if isinstance(rles, list) else rles
    return _coco_rle(merged)


def pairwise_iou(
    gt_rles: list[dict[str, Any]],
    prediction_rles: list[dict[str, Any]],
) -> np.ndarray:
    """Return (n_gt, n_pred) float32 IoUs, matching the organizer's tensor dtype."""
    if not gt_rles or not prediction_rles:
        return np.zeros((len(gt_rles), len(prediction_rles)), dtype=np.float32)
    # pycocotools returns (detections, ground truth) in float64. The organizer
    # decodes the same RLEs to float32 layers and computes float32 IoUs.
    matrix = mask_utils.iou(
        [_coco_rle(rle) for rle in prediction_rles],
        [_coco_rle(rle) for rle in gt_rles],
        [0] * len(gt_rles),
    )
    return np.asarray(matrix, dtype=np.float32).T


def pq_value(tp_ious: Iterable[float], fp: int, fn: int) -> float:
    tp_ious = list(tp_ious)
    denominator = len(tp_ious) + 0.5 * fp + 0.5 * fn
    return float(sum(tp_ious) / denominator) if denominator > 0 else 0.0


def _accumulate_view(
    counts: ObservationCounts,
    diagnostics: ScorerDiagnostics,
    gt_rles: list[dict[str, Any]],
    prediction_rles: list[dict[str, Any]],
    iou_threshold: float,
    diagnostic_overlap_threshold: float,
) -> None:
    counts.gt_count += len(gt_rles)
    counts.prediction_count += len(prediction_rles)
    if not gt_rles:
        counts.fp += len(prediction_rles)
        diagnostics.prediction_match_degrees.update({0: len(prediction_rles)})
        return
    if not prediction_rles:
        counts.fn += len(gt_rles)
        diagnostics.gt_match_degrees.update({0: len(gt_rles)})
        return

    ious = pairwise_iou(gt_rles, prediction_rles)
    hits = ious > iou_threshold

    # This intentionally mirrors the provided scorer: every qualifying edge is
    # a TP; it does not perform one-to-one assignment.
    counts.tp_ious.extend(float(value) for value in ious[hits])
    counts.fp += int(np.count_nonzero(np.sum(hits, axis=0) == 0))
    counts.fn += int(np.count_nonzero(np.sum(hits, axis=1) == 0))

    positive = ious[ious > 0]
    diagnostics.positive_ious.extend(float(value) for value in positive)
    diagnostics.positive_dices.extend(
        float(value) for value in (2.0 * positive / (1.0 + positive))
    )

    any_overlap = ious > diagnostic_overlap_threshold
    diagnostics.gt_match_degrees.update(int(value) for value in np.sum(any_overlap, axis=1))
    diagnostics.prediction_match_degrees.update(
        int(value) for value in np.sum(any_overlap, axis=0)
    )


def score_observation(
    dataset: CompetitionDataset,
    file_name: str,
    predictions: list[PredictedInstance],
    *,
    iou_threshold: float = 0.5,
    diagnostic_overlap_threshold: float = 0.0,
) -> tuple[ObservationCounts, ScorerDiagnostics]:
    views = dataset.views_by_observation[file_name]
    counts = ObservationCounts(
        file_name=file_name,
        year=views[0].year,
        site=views[0].site,
        annotator_views=len(views),
    )
    diagnostics = ScorerDiagnostics()
    prediction_rles = [prediction.rle for prediction in predictions]
    for view in views:
        annotations = dataset.annotations_by_image_id[view.image_id]
        gt_rles = [
            annotation_to_rle(annotation, view.height, view.width)
            for annotation in annotations
        ]
        _accumulate_view(
            counts,
            diagnostics,
            gt_rles,
            prediction_rles,
            iou_threshold,
            diagnostic_overlap_threshold,
        )
    return counts, diagnostics


def _aggregate_counts(rows: Iterable[ObservationCounts]) -> dict[str, float | int]:
    tp_ious: list[float] = []
    fp = 0
    fn = 0
    gt_count = 0
    prediction_count = 0
    annotator_views = 0
    for row in rows:
        tp_ious.extend(row.tp_ious)
        fp += row.fp
        fn += row.fn
        gt_count += row.gt_count
        prediction_count += row.prediction_count
        annotator_views += row.annotator_views
    return {
        "pq": pq_value(tp_ious, fp, fn),
        "tp": len(tp_ious),
        "tp_iou_sum": float(sum(tp_ious)),
        "fp": fp,
        "fn": fn,
        "gt_count_across_annotator_views": gt_count,
        "prediction_count_across_annotator_views": prediction_count,
        "annotator_view_count": annotator_views,
    }


def _histogram(values: list[float]) -> dict[str, Any]:
    bins = np.linspace(0.0, 1.0, 51, dtype=np.float64)
    counts, edges = np.histogram(values, bins=bins)
    total = int(counts.sum())
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)) if values else 0.0,
        "bins": [float(value) for value in edges],
        "counts": [int(value) for value in counts],
        "probabilities": [float(value / total) for value in counts] if total else [0.0] * 50,
    }


def _degree_distribution(counter: Counter[int]) -> dict[str, int]:
    return {str(degree): int(counter[degree]) for degree in sorted(counter)}


def score_validation(
    dataset: CompetitionDataset,
    predictions_by_observation: dict[str, list[PredictedInstance]],
    validation_names: Iterable[str],
    *,
    iou_threshold: float = 0.5,
    diagnostic_overlap_threshold: float = 0.0,
) -> tuple[dict[str, Any], list[ObservationCounts]]:
    rows: list[ObservationCounts] = []
    all_diagnostics = ScorerDiagnostics()
    for file_name in sorted(validation_names):
        counts, diagnostics = score_observation(
            dataset,
            file_name,
            predictions_by_observation.get(file_name, []),
            iou_threshold=iou_threshold,
            diagnostic_overlap_threshold=diagnostic_overlap_threshold,
        )
        rows.append(counts)
        all_diagnostics.positive_ious.extend(diagnostics.positive_ious)
        all_diagnostics.positive_dices.extend(diagnostics.positive_dices)
        all_diagnostics.gt_match_degrees.update(diagnostics.gt_match_degrees)
        all_diagnostics.prediction_match_degrees.update(
            diagnostics.prediction_match_degrees
        )

    grouped: dict[str, dict[str, list[ObservationCounts]]] = {
        "year": defaultdict(list),
        "site": defaultdict(list),
    }
    for row in rows:
        grouped["year"][str(row.year)].append(row)
        grouped["site"][row.site].append(row)

    result = {
        "global": _aggregate_counts(rows),
        "by_year": {
            label: _aggregate_counts(group_rows)
            for label, group_rows in sorted(grouped["year"].items())
        },
        "by_site": {
            label: _aggregate_counts(group_rows)
            for label, group_rows in sorted(grouped["site"].items())
        },
        "iou_positive_pairs": _histogram(all_diagnostics.positive_ious),
        "dice_positive_pairs": _histogram(all_diagnostics.positive_dices),
        "one_to_many_many_to_one": {
            "threshold": diagnostic_overlap_threshold,
            "gt_match_degree_counts": _degree_distribution(
                all_diagnostics.gt_match_degrees
            ),
            "prediction_match_degree_counts": _degree_distribution(
                all_diagnostics.prediction_match_degrees
            ),
        },
    }
    return result, rows


def views_for_observation(dataset: CompetitionDataset, file_name: str) -> tuple[ImageView, ...]:
    return dataset.views_by_observation[file_name]
