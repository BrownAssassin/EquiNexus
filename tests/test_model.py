# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import sys
from unittest.mock import patch
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from filament_baseline.inference import (  # noqa: E402
    InstanceConfig,
    _inverse_transform,
    _transform,
    instances_from_probabilities,
)
from filament_baseline.features import prepare_input_channels  # noqa: E402
from filament_baseline.model import FilamentUNet, model_architecture  # noqa: E402
from filament_baseline.classical import ClassicalConfig, predictions_from_zmap  # noqa: E402
from filament_baseline.predictor import (  # noqa: E402
    decode_rle,
    encode_binary_mask,
    encode_cropped_binary_mask,
)
from filament_baseline.model_runner import _load_model  # noqa: E402
from filament_baseline.training import train_model  # noqa: E402


class ModelTests(unittest.TestCase):
    def test_model_architecture_metadata_identifies_each_model(self) -> None:
        self.assertEqual(
            model_architecture("unet"),
            "from_scratch_five_level_unet_foreground_boundary",
        )
        self.assertEqual(
            model_architecture("smp_unet_resnet34"),
            "smp_unet_resnet34_foreground_boundary",
        )
        self.assertEqual(
            model_architecture("dinov2_vits14"),
            "dinov2_vits14_frozen_backbone_cnn_decoder_foreground_boundary",
        )

    def test_raw_local_contrast_features_are_normalized_two_channel_inputs(self) -> None:
        gray = np.full((2048, 2048), 128, dtype=np.uint8)
        gray[990:996, 700:1300] = 90
        channels = prepare_input_channels(gray, "raw_local_contrast")
        self.assertEqual(tuple(channels.shape), (2, 2048, 2048))
        self.assertTrue(np.isfinite(channels).all())
        self.assertGreater(float(channels[1].max()), float(channels[1].min()))

    def test_radial_contrast_features_are_finite_three_channel_inputs(self) -> None:
        yy, xx = np.ogrid[:2048, :2048]
        radius = np.sqrt((yy - 1023) ** 2 + (xx - 1023) ** 2)
        gray = np.clip(180 - radius * 0.05, 0, 255).astype(np.uint8)
        gray[990:996, 700:1300] = 90
        channels = prepare_input_channels(gray, "raw_local_contrast_radial")
        self.assertEqual(tuple(channels.shape), (3, 2048, 2048))
        self.assertTrue(np.isfinite(channels).all())
        self.assertGreater(float(channels[2, 990:996, 700:1300].mean()), 0.0)

    def test_dihedral_tta_transforms_are_invertible_and_unique(self) -> None:
        source = torch.arange(35).reshape(1, 1, 5, 7)
        transformed = [_transform(source, operation) for operation in range(8)]
        for operation, value in enumerate(transformed):
            restored = _inverse_transform(value, operation)
            torch.testing.assert_close(restored, source)
        signatures = {tuple(value.flatten().tolist()) for value in transformed}
        self.assertEqual(len(signatures), 8)

    def test_final_fit_without_validation_saves_loadable_last_step(self) -> None:
        class TinyPatchDataset(torch.utils.data.Dataset):
            target_mode = "consensus"

            def __len__(self) -> int:
                return 2

            def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
                return torch.zeros(1, 128, 128), torch.zeros(2, 128, 128)

        checkpoint = REPO_ROOT / "artifacts" / "tests" / "final-fit-smoke.pt"
        with patch.object(torch.cuda, "is_available", return_value=False):
            result = train_model(
                train_dataset=TinyPatchDataset(),
                validation_dataset=None,
                output_path=checkpoint,
                steps=1,
                batch_size=1,
                validation_interval=1,
            )
            loaded = _load_model(checkpoint, torch.device("cpu"))
        self.assertEqual(result["selection_mode"], "final_step")
        self.assertEqual(result["best_step"], 1)
        self.assertIsNone(result["best_validation_patch_dice"])
        self.assertIsInstance(loaded, FilamentUNet)

    def test_cropped_rle_matches_full_canvas_encoding(self) -> None:
        full = np.zeros((128, 96), dtype=np.uint8)
        crop = np.zeros((23, 19), dtype=np.uint8)
        crop[3:17, 4:12] = 1
        x_offset, y_offset = 51, 37
        full[y_offset : y_offset + crop.shape[0], x_offset : x_offset + crop.shape[1]] = crop
        actual = encode_cropped_binary_mask(crop, x_offset, y_offset, full.shape)
        expected = encode_binary_mask(full)
        self.assertEqual(actual, expected)
        np.testing.assert_array_equal(decode_rle(actual), full)

    def test_two_head_unet_forward_backward(self) -> None:
        model = FilamentUNet(in_channels=2, base_channels=8)
        image = torch.randn(2, 2, 128, 128)
        target = torch.rand(2, 2, 128, 128)
        output = model(image)
        self.assertEqual(tuple(output.shape), (2, 2, 128, 128))
        output.square().mean().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

    def test_two_head_unet_accepts_non_power_of_two_stride_aligned_shape(self) -> None:
        model = FilamentUNet(base_channels=8)
        image = torch.randn(1, 1, 160, 160)
        output = model(image)
        self.assertEqual(tuple(output.shape), (1, 2, 160, 160))

    def test_boundary_watershed_emits_disjoint_roundtrip_rles(self) -> None:
        import cv2

        foreground = np.zeros((2048, 2048), dtype=np.float32)
        boundary = np.zeros_like(foreground)
        cv2.circle(foreground, (1008, 1024), 18, 0.9, thickness=-1)
        cv2.circle(foreground, (1038, 1024), 18, 0.9, thickness=-1)
        cv2.line(boundary, (1023, 990), (1023, 1058), 0.95, thickness=3)
        instances = instances_from_probabilities(
            foreground,
            boundary,
            "20140609195854Bh.jpeg",
            InstanceConfig(
                foreground_threshold=0.5,
                boundary_threshold=0.6,
                boundary_weight=2.0,
                seed_threshold=0.5,
                seed_min_area=5,
                min_instance_area=50,
            ),
        )
        self.assertEqual(len(instances), 2)
        masks = [decode_rle(instance.rle) for instance in instances]
        self.assertEqual(sum(mask.sum() for mask in masks), sum(i.area for i in instances))
        self.assertEqual(int(np.count_nonzero(masks[0] & masks[1])), 0)

    def test_classical_zmap_extractor_returns_decodable_instance(self) -> None:
        z_map = np.full((1024, 1024), -1.0, dtype=np.float32)
        disk = np.zeros((1024, 1024), dtype=bool)
        disk[100:924, 100:924] = True
        z_map[500:506, 480:540] = 5.0
        z_map[502:504, 480:540] = 7.0
        predictions = predictions_from_zmap(
            z_map,
            disk,
            "20140609195854Bh.jpeg",
            ClassicalConfig(z_threshold=2.5, bridge_size=3, min_instance_area=100),
        )
        self.assertEqual(len(predictions), 1)
        decoded = decode_rle(predictions[0].rle)
        self.assertEqual(decoded.shape, (2048, 2048))
        self.assertGreater(int(decoded.sum()), 100)


if __name__ == "__main__":
    unittest.main()
