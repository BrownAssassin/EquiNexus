# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import numpy as np

from filament_baseline.detector_fusion import (
    DetectorFusionConfig,
    fuse_detector_instances,
    solar_disk_mask_from_geometry,
)
from filament_baseline.predictor import encode_binary_mask


def _detection(mask: np.ndarray, score: float) -> dict[str, object]:
    rle = encode_binary_mask(mask)
    return {"score": score, "counts": rle["counts"]}


def test_detector_regions_claim_distinct_foreground_pixels() -> None:
    foreground = np.zeros((2048, 2048), dtype=np.float32)
    foreground[400:430, 400:430] = 1.0
    foreground[600:630, 600:630] = 1.0
    first = np.zeros((2048, 2048), dtype=np.uint8)
    second = np.zeros_like(first)
    first[398:432, 398:432] = 1
    second[598:632, 598:632] = 1

    labels = fuse_detector_instances(
        foreground,
        [_detection(second, 0.8), _detection(first, 0.9)],
        DetectorFusionConfig(
            foreground_threshold=0.5,
            detection_threshold=0.5,
            grow_radius=0,
            minimum_claimed_area=100,
            closing_radius=0,
        ),
    )

    assert set(np.unique(labels)) == {0, 1, 2}
    assert np.all(labels[400:430, 400:430] == 1)
    assert np.all(labels[600:630, 600:630] == 2)


def test_low_confidence_and_too_small_claims_are_dropped() -> None:
    foreground = np.zeros((2048, 2048), dtype=np.float32)
    foreground[400:410, 400:410] = 1.0
    mask = np.zeros((2048, 2048), dtype=np.uint8)
    mask[400:410, 400:410] = 1

    no_confident_detection = fuse_detector_instances(
        foreground,
        [_detection(mask, 0.1)],
        DetectorFusionConfig(detection_threshold=0.2, closing_radius=0),
    )
    too_small = fuse_detector_instances(
        foreground,
        [_detection(mask, 0.9)],
        DetectorFusionConfig(
            detection_threshold=0.2,
            minimum_claimed_area=200,
            closing_radius=0,
        ),
    )

    assert not np.any(no_confident_detection)
    assert not np.any(too_small)


def test_image_derived_disk_mask_clips_fusion_predictions() -> None:
    foreground = np.zeros((2048, 2048), dtype=np.float32)
    foreground[400:430, 400:430] = 1.0
    detection = np.zeros((2048, 2048), dtype=np.uint8)
    detection[400:430, 400:430] = 1
    disk = np.zeros((2048, 2048), dtype=bool)
    disk[400:415, 400:430] = True

    labels = fuse_detector_instances(
        foreground,
        [_detection(detection, 0.9)],
        DetectorFusionConfig(
            foreground_threshold=0.5,
            detection_threshold=0.5,
            grow_radius=0,
            minimum_claimed_area=1,
            closing_radius=0,
        ),
        disk_mask=disk,
    )

    assert np.array_equal(labels > 0, disk & (foreground >= 0.5))


def test_solar_disk_mask_uses_fitted_center_radius_and_shrink() -> None:
    disk = solar_disk_mask_from_geometry(
        (11, 11), {"cx": 5.0, "cy": 5.0, "radius": 3.0}, shrink=0.5
    )

    assert disk.sum() == 9
    assert disk[5, 6]
    assert not disk[5, 7]
