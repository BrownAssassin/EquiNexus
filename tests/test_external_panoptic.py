# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

import numpy as np
import torch
from pycocotools import mask as mask_utils

from filament_baseline.external_panoptic import (
    CropRefiner,
    build_detector,
    _overlap_groups,
    _safe_tta_mask,
    _split_with_detector_seeds,
    _square_crop,
    encode_instance,
)


def test_overlap_groups_join_transitive_intersections() -> None:
    masks = np.zeros((4, 16, 16), dtype=bool)
    masks[0, 2:6, 2:6] = True
    masks[1, 4:8, 4:8] = True
    masks[2, 6:10, 6:10] = True
    masks[3, 12:14, 12:14] = True

    assert _overlap_groups(masks) == [[0, 1, 2], [3]]


def test_square_crop_stays_inside_image_and_contains_active_region() -> None:
    mask = np.zeros((32, 48), dtype=bool)
    mask[0:5, 2:8] = True

    x0, y0, x1, y1 = _square_crop(mask, context=1.5, minimum=12)

    assert 0 <= x0 < x1 <= mask.shape[1]
    assert 0 <= y0 < y1 <= mask.shape[0]
    assert x1 - x0 == y1 - y0
    assert mask[y0:y1, x0:x1].sum() == mask.sum()


def test_seed_partition_is_disjoint_and_preserves_refined_union() -> None:
    refined = np.zeros((20, 40), dtype=bool)
    refined[7:13, 3:37] = True
    first = np.zeros_like(refined)
    first[6:14, 2:16] = True
    second = np.zeros_like(refined)
    second[6:14, 24:38] = True

    pieces = _split_with_detector_seeds(refined, [first, second], np.array([0.8, 0.7]))

    assert len(pieces) == 2
    assert not np.any(pieces[0] & pieces[1])
    assert np.array_equal(np.logical_or.reduce(pieces), refined)


def test_tta_area_guard_falls_back_to_single_view_probability() -> None:
    single = np.zeros((16, 16), dtype=np.float32)
    single[4:8, 4:8] = 1.0
    oversized = np.ones_like(single, dtype=np.float32)

    chosen = _safe_tta_mask(single, oversized)

    assert np.array_equal(chosen, single)


def test_crop_refiner_emits_native_crop_resolution() -> None:
    torch.manual_seed(3)
    model = CropRefiner().eval()
    with torch.inference_mode():
        output = model(torch.zeros((1, 3, 64, 64), dtype=torch.float32))

    assert output.shape == (1, 1, 64, 64)


def test_public_checkpoint_architecture_key_counts_match_model_card() -> None:
    detector = build_detector()
    refiner = CropRefiner()

    assert len(detector.state_dict()) == 432
    assert len(refiner.state_dict()) == 182


def test_instance_rle_round_trip_preserves_full_resolution_mask() -> None:
    mask = np.zeros((2048, 2048), dtype=bool)
    mask[125:139, 200:260] = True

    counts = encode_instance(mask)
    decoded = mask_utils.decode({"size": [2048, 2048], "counts": counts.encode("ascii")})

    assert decoded.shape == mask.shape
    assert np.array_equal(decoded.astype(bool), mask)
