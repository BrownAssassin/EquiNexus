from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

from .data import CompetitionDataset
from .predictor import PredictedInstance, decode_rle, read_grayscale
from .scorer import annotation_to_rle


def write_comparison_overlays(
    dataset: CompetitionDataset,
    predictions: dict[str, list[PredictedInstance]],
    output_dir: str | Path,
    maximum: int = 12,
) -> list[str]:
    names = sorted(predictions)
    if not names or maximum <= 0:
        return []
    if len(names) <= maximum:
        selected = names
    elif maximum == 1:
        selected = [names[len(names) // 2]]
    else:
        indices = {
            round(index * (len(names) - 1) / (maximum - 1))
            for index in range(maximum)
        }
        selected = [names[index] for index in sorted(indices)]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for file_name in selected:
        gray = read_grayscale(dataset.train_image_dir / file_name)
        view = dataset.views_by_observation[file_name][0]
        ground_truth = np.zeros(gray.shape, dtype=bool)
        for annotation in dataset.annotations_by_image_id[view.image_id]:
            rle = annotation_to_rle(annotation, view.height, view.width)
            encoded = {
                "size": rle["size"],
                "counts": rle["counts"].encode("ascii")
                if isinstance(rle["counts"], str)
                else rle["counts"],
            }
            ground_truth |= mask_utils.decode(encoded).astype(bool)
        predicted = np.zeros(gray.shape, dtype=bool)
        for instance in predictions[file_name]:
            predicted |= decode_rle(instance.rle).astype(bool)

        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        overlay = canvas.astype(np.float32)
        gt_only = ground_truth & ~predicted
        pred_only = predicted & ~ground_truth
        overlap = ground_truth & predicted
        overlay[gt_only] = 0.55 * overlay[gt_only] + 0.45 * np.array([40, 255, 80])
        overlay[pred_only] = 0.55 * overlay[pred_only] + 0.45 * np.array([255, 50, 50])
        overlay[overlap] = 0.45 * overlay[overlap] + 0.55 * np.array([255, 230, 30])
        canvas = np.clip(overlay, 0, 255).astype(np.uint8)
        cv2.putText(
            canvas,
            f"GT union: {len(dataset.annotations_by_image_id[view.image_id])} instances | "
            f"Predicted: {len(predictions[file_name])} | green GT, red pred, yellow overlap",
            (24, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        destination = output_dir / f"{Path(file_name).stem}_comparison.png"
        if not cv2.imwrite(str(destination), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)):
            raise OSError(f"Failed to save validation comparison overlay: {destination}")
        saved.append(destination.name)
    return saved
