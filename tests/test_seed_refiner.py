# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from filament_baseline.seed_refiner import (  # noqa: E402
    ObservationBatchSampler,
    SeedCropSample,
    SeedRefinerConfig,
    SeedRefinerDataset,
    _crop_bounds,
    _fill_segmentation_crop,
)


class SeedRefinerTests(unittest.TestCase):
    def test_crop_bounds_are_square_and_stay_inside_image(self) -> None:
        bounds = _crop_bounds((0.0, 0.0, 120.0, 80.0), (2048, 2048), 1.8)
        x0, y0, side = bounds
        self.assertEqual(side, 216)
        self.assertGreaterEqual(x0, 0)
        self.assertGreaterEqual(y0, 0)
        self.assertLessEqual(x0 + side, 2048)
        self.assertLessEqual(y0 + side, 2048)

    def test_polygon_target_crop_preserves_expected_foreground(self) -> None:
        # A polygon wholly inside the 64x64 crop should survive nearest rasterization.
        polygon = [[20, 20, 44, 20, 44, 44, 20, 44]]
        target = _fill_segmentation_crop(polygon, (0, 0, 64), 64)
        self.assertEqual(target.shape, (64, 64))
        self.assertGreater(int(target.sum()), 0)
        self.assertEqual(int(target[0, 0]), 0)

    def test_observation_sampler_never_mixes_source_observations(self) -> None:
        class DummyDataset:
            samples = (
                SeedCropSample("a.jpeg", "a", "", 1.0, (0, 0, 10, 10), None),
                SeedCropSample("a.jpeg", "a", "", 1.0, (0, 0, 10, 10), None),
                SeedCropSample("b.jpeg", "b", "", 1.0, (0, 0, 10, 10), None),
            )

            def __len__(self) -> int:
                return len(self.samples)

            def sample_file_name(self, index: int) -> str:
                return self.samples[index].file_name

        sampler = ObservationBatchSampler(DummyDataset(), batch_size=2, seed=42)
        for batch in sampler:
            self.assertEqual(len({DummyDataset().sample_file_name(index) for index in batch}), 1)

    def test_epoch_counter_is_shared_for_persistent_workers(self) -> None:
        class DummyCompetitionDataset:
            train_image_dir = REPO_ROOT / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026" / "train"

        sample = SeedCropSample(
            "20140609195854Bh.jpeg",
            "image_id",
            "",
            1.0,
            (900.0, 900.0, 200.0, 200.0),
            None,
        )
        dataset = SeedRefinerDataset(
            DummyCompetitionDataset(),
            (sample,),
            SeedRefinerConfig(crop_size=64),
            augment=True,
        )
        dataset.set_epoch(7)
        self.assertEqual(int(dataset._epoch.item()), 7)
        self.assertTrue(dataset._epoch.is_shared())


if __name__ == "__main__":
    unittest.main()
