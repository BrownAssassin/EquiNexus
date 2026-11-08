# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from filament_baseline.solar_ssl import (  # noqa: E402
    SolarContrastiveDataset,
    _distillation_loss,
)


def test_solar_contrastive_dataset_builds_two_normalized_views(tmp_path: Path) -> None:
    data_root = tmp_path / "gong"
    image_root = data_root / "images"
    image_root.mkdir(parents=True)
    image_path = image_root / "20261101000042Lh.jpg"
    pixels = np.arange(96 * 96, dtype=np.uint8).reshape(96, 96)
    Image.fromarray(pixels).save(image_path)
    records = [
        {
            "filename": image_path.name,
            "url": "https://example.invalid/image.jpg",
            "timestamp_utc": "2026-11-01T00:00:42+00:00",
            "site": "L",
            "bytes": image_path.stat().st_size,
            "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        }
    ]
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":"))
    manifest = {
        "files": records,
        "content_manifest_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }
    (data_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    dataset = SolarContrastiveDataset(data_root, size=28)
    first, second = dataset[0]

    assert len(dataset) == 1
    assert dataset.manifest_digest == manifest["content_manifest_sha256"]
    assert first.shape == second.shape == (3, 28, 28)
    assert torch.isfinite(first).all() and torch.isfinite(second).all()


def test_distillation_loss_centers_teacher_logits_and_is_finite() -> None:
    student_first = torch.randn(3, 8, requires_grad=True)
    student_second = torch.randn(3, 8, requires_grad=True)
    teacher_first = torch.randn(3, 8)
    teacher_second = torch.randn(3, 8)
    center = torch.randn(1, 8)

    loss, center_batch = _distillation_loss(
        student_first, student_second, teacher_first, teacher_second, center
    )
    loss.backward()

    expected_center = 0.5 * (teacher_first.mean(dim=0) + teacher_second.mean(dim=0))
    assert torch.isfinite(loss)
    assert torch.isfinite(student_first.grad).all()
    assert torch.isfinite(student_second.grad).all()
    assert torch.allclose(center_batch, expected_center)
