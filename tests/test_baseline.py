# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import sys
import tempfile
import unittest
import csv
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from filament_baseline.data import CompetitionDataset, ImageView  # noqa: E402
from filament_baseline.predictor import (  # noqa: E402
    PredictedInstance,
    decode_rle,
    encode_binary_mask,
    encode_label_components,
    predict_image,
)
from filament_baseline.scorer import (  # noqa: E402
    ObservationCounts,
    ScorerDiagnostics,
    _accumulate_view,
    annotation_to_rle,
    pairwise_iou,
    score_observation,
)
from filament_baseline.split import create_observation_split  # noqa: E402
from filament_baseline.runner import _write_predictions  # noqa: E402


def fake_dataset(observation_count: int = 40, views_per_observation: int = 2) -> CompetitionDataset:
    views = []
    by_observation: dict[str, list[ImageView]] = {}
    annotations_by_view = {}
    for index in range(observation_count):
        year = 2014 + index % 2
        # Keep observation identifiers unique while preserving the competition filename shape.
        file_name = f"{year}0101{index:06d}{'Bh' if index % 3 else 'Mh'}.jpeg"
        group = []
        for view_number in range(views_per_observation):
            image_id = f"rater{view_number}-{Path(file_name).stem}"
            view = ImageView(image_id, file_name, 2048, 2048, year, Path(file_name).stem[-2:])
            views.append(view)
            group.append(view)
            annotations_by_view[image_id] = (
                {
                    "id": index * views_per_observation + view_number + 1,
                    "image_id": image_id,
                    "segmentation": [[10, 10, 20, 10, 20, 20, 10, 20]],
                    "iscrowd": 0,
                },
            )
        by_observation[file_name] = group
    return CompetitionDataset(
        root=REPO_ROOT,
        train_image_dir=REPO_ROOT,
        test_image_dir=REPO_ROOT,
        annotation_path=REPO_ROOT / "annotations.json",
        image_views=tuple(views),
        annotations_by_image_id={
            key: tuple(value) for key, value in annotations_by_view.items()
        },
        views_by_observation={
            key: tuple(value) for key, value in by_observation.items()
        },
        categories=(),
    )


class SplitTests(unittest.TestCase):
    def test_split_is_deterministic_and_groups_views_by_observation(self) -> None:
        dataset = fake_dataset()
        with tempfile.TemporaryDirectory(dir=REPO_ROOT / "data" / "splits") as directory:
            first = create_observation_split(dataset, directory, seed=42, validation_fraction=0.2)
            first_bytes = first.manifest_path.read_bytes()
            second = create_observation_split(dataset, directory, seed=42, validation_fraction=0.2)
            self.assertEqual(first.sha256, second.sha256)
            self.assertEqual(first_bytes, second.manifest_path.read_bytes())
            self.assertEqual(len(first.validation_names), 8)
            self.assertEqual(len(first.training_names), 32)
            self.assertEqual(set(first.assignments), set(dataset.views_by_observation))
            for file_name, views in dataset.views_by_observation.items():
                self.assertEqual(first.assignments[file_name], first.assignments[views[0].file_name])
                self.assertTrue(all(view.file_name == file_name for view in views))


class ScorerTests(unittest.TestCase):
    @staticmethod
    def rle(points: list[tuple[int, int]]) -> dict[str, object]:
        mask = np.zeros((2048, 2048), dtype=np.uint8)
        for row, column in points:
            mask[row, column] = 1
        return encode_binary_mask(mask)

    def test_rle_round_trip_preserves_full_resolution_mask(self) -> None:
        mask = np.zeros((2048, 2048), dtype=np.uint8)
        mask[27:42, 100:180] = 1
        mask[1900, 17] = 1
        restored = decode_rle(encode_binary_mask(mask))
        np.testing.assert_array_equal(restored, mask)

    def test_label_map_encoding_matches_per_component_mask_encoding(self) -> None:
        labels = np.zeros((64, 80), dtype=np.int32)
        labels[0:3, 0:4] = 1
        labels[12:19, 50:55] = 2
        labels[63, 79] = 2
        encoded = encode_label_components(labels, [1, 2])
        self.assertEqual(set(encoded), {1, 2})
        for label in (1, 2):
            expected = encode_binary_mask((labels == label).astype(np.uint8))
            self.assertEqual(encoded[label]["counts"], expected["counts"])
            np.testing.assert_array_equal(decode_rle(encoded[label]), labels == label)

    def test_exactly_half_iou_is_not_a_hit(self) -> None:
        gt = self.rle([(30, 30), (30, 31)])
        pred = self.rle([(30, 30)])
        matrix = pairwise_iou([gt], [pred])
        self.assertEqual(float(matrix[0, 0]), 0.5)

        counts = ObservationCounts("x.jpeg", 2014, "Bh")
        _accumulate_view(counts, ScorerDiagnostics(), [gt], [pred], 0.5, 0.0)
        self.assertEqual(counts.tp, 0)
        self.assertEqual(counts.fp, 1)
        self.assertEqual(counts.fn, 1)
        self.assertEqual(counts.pq, 0.0)

    def test_iou_above_half_is_a_hit(self) -> None:
        gt = self.rle([(30, 30), (30, 31)])
        pred = self.rle([(30, 30), (30, 31), (30, 32)])
        counts = ObservationCounts("x.jpeg", 2014, "Bh")
        _accumulate_view(counts, ScorerDiagnostics(), [gt], [pred], 0.5, 0.0)
        self.assertEqual(counts.tp, 1)
        self.assertEqual(counts.fp, 0)
        self.assertEqual(counts.fn, 0)
        self.assertAlmostEqual(counts.pq, 2.0 / 3.0, places=6)

    def test_official_scorer_counts_all_duplicate_edges_without_assignment(self) -> None:
        gt = self.rle([(30, 30), (30, 31), (30, 32)])
        predictions = [self.rle([(30, 30), (30, 31), (30, 32)]) for _ in range(2)]
        matrix = pairwise_iou([gt], predictions)
        counts = ObservationCounts("x.jpeg", 2014, "Bh")
        diagnostics = ScorerDiagnostics()
        _accumulate_view(counts, diagnostics, [gt], predictions, 0.5, 0.0)
        self.assertEqual(matrix.shape, (1, 2))
        self.assertEqual(counts.tp, 2)
        self.assertEqual(counts.fp, 0)
        self.assertEqual(counts.fn, 0)
        self.assertEqual(diagnostics.gt_match_degrees[2], 1)
        self.assertEqual(diagnostics.prediction_match_degrees[1], 2)

    def test_one_prediction_can_count_against_multiple_ground_truth_edges(self) -> None:
        identical_gt = self.rle([(30, 30), (30, 31), (30, 32)])
        prediction = self.rle([(30, 30), (30, 31), (30, 32)])
        counts = ObservationCounts("x.jpeg", 2014, "Bh")
        diagnostics = ScorerDiagnostics()
        _accumulate_view(
            counts,
            diagnostics,
            [identical_gt, identical_gt],
            [prediction],
            0.5,
            0.0,
        )
        self.assertEqual(counts.tp, 2)
        self.assertEqual(counts.fp, 0)
        self.assertEqual(counts.fn, 0)
        self.assertEqual(diagnostics.prediction_match_degrees[2], 1)
        self.assertEqual(diagnostics.gt_match_degrees[1], 2)

    def test_empty_prediction_and_ground_truth_edges(self) -> None:
        gt = self.rle([(30, 30)])
        counts = ObservationCounts("x.jpeg", 2014, "Bh")
        diagnostics = ScorerDiagnostics()
        _accumulate_view(counts, diagnostics, [gt], [], 0.5, 0.0)
        self.assertEqual(counts.fn, 1)
        self.assertEqual(counts.fp, 0)
        self.assertEqual(counts.tp, 0)
        _accumulate_view(counts, diagnostics, [], [self.rle([(50, 50)])], 0.5, 0.0)
        self.assertEqual(counts.fp, 1)
        self.assertEqual(pairwise_iou([], [gt]).shape, (0, 1))
        self.assertEqual(pairwise_iou([gt], []).shape, (1, 0))

    def test_prediction_csv_uses_official_columns_and_counts_only(self) -> None:
        mask = np.zeros((2048, 2048), dtype=np.uint8)
        mask[10:13, 20:24] = 1
        rle = encode_binary_mask(mask)
        byte_rle = {**rle, "counts": rle["counts"].encode("ascii")}
        prediction = PredictedInstance("20140609195854Bh_1", "20140609195854Bh", 12, byte_rle)
        with tempfile.TemporaryDirectory(dir=REPO_ROOT / "data" / "splits") as directory:
            path = Path(directory) / "predictions.csv"
            _write_predictions(path, {prediction.image_stem: [prediction]})
            with path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.reader(stream))
        self.assertEqual(rows[0], ["filament_id", "segmentation_rle"])
        self.assertEqual(rows[1], [prediction.filament_id, rle["counts"]])
        decoded = decode_rle({"size": [2048, 2048], "counts": rows[1][1]})
        np.testing.assert_array_equal(decoded, mask)

    def test_predictions_are_reused_for_every_annotator_view(self) -> None:
        dataset = fake_dataset(observation_count=1, views_per_observation=2)
        file_name = dataset.observation_names[0]
        first_view = dataset.views_by_observation[file_name][0]
        gt_rle = annotation_to_rle(
            dataset.annotations_by_image_id[first_view.image_id][0], 2048, 2048
        )
        prediction = PredictedInstance(
            f"{Path(file_name).stem}_1",
            Path(file_name).stem,
            100,
            gt_rle,
        )
        counts, _ = score_observation(dataset, file_name, [prediction])
        self.assertEqual(counts.annotator_views, 2)
        self.assertEqual(counts.tp, 2)
        self.assertEqual(counts.gt_count, 2)
        self.assertEqual(counts.prediction_count, 2)
        self.assertEqual(counts.pq, 1.0)


class PredictorTests(unittest.TestCase):
    def test_local_contrast_predictor_is_deterministic_and_emits_coco_rle(self) -> None:
        import cv2

        image = np.full((2048, 2048), 150, dtype=np.uint8)
        cv2.line(image, (960, 1000), (1080, 1030), color=20, thickness=4)
        first = predict_image(image, "20140609195854Bh.jpeg")
        second = predict_image(image, "20140609195854Bh.jpeg")
        self.assertEqual(
            [(item.filament_id, item.area, item.rle) for item in first],
            [(item.filament_id, item.area, item.rle) for item in second],
        )
        self.assertTrue(first)
        self.assertTrue(all(item.filament_id.startswith("20140609195854Bh_") for item in first))
        self.assertTrue(all(item.rle["size"] == [2048, 2048] for item in first))


if __name__ == "__main__":
    unittest.main()
