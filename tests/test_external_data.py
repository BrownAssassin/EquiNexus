# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from filament_baseline.external_data import load_chrotel_dataset


def _toy_chrotel_dataset(
    repository_root: Path, competition_names: tuple[str, ...] = ()
) -> tuple[Path, SimpleNamespace]:
    root = repository_root / "data" / "external" / "chrotel"
    image_dir = root / "images"
    mask_dir = root / "masks"
    test_image_dir = repository_root / "competition-test"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir()
    test_image_dir.mkdir()

    image = np.zeros((2048, 2048), dtype=np.uint8)
    cv2.circle(image, (1023, 1023), 900, color=180, thickness=-1)
    mask = np.zeros_like(image)
    mask[400:450, 500:550] = 255
    file_name = "chrotel_20120417_090150_ha.png"
    image_path = image_dir / file_name
    mask_path = mask_dir / file_name
    assert cv2.imwrite(str(image_path), image)
    assert cv2.imwrite(str(mask_path), mask)
    row = {
        "observation_id": "chrotel_20120417_090150_ha",
        "date": "20120417",
        "file_name": file_name,
        "source_timestamp": "2012-04-17T09:01:50.000+00:00",
        "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
    }
    selection_row = {
        key: row[key]
        for key in ("observation_id", "date", "file_name", "source_timestamp")
    }
    selection_digest = hashlib.sha256(
        json.dumps([selection_row], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "all_competition_dates_excluded": True,
                "competition_date_overlap": 0,
                "selection": {
                    "frozen_competition_split_sha256": (
                        "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"
                    )
                },
                "selection_manifest_sha256": selection_digest,
                "observations": [row],
            }
        ),
        encoding="utf-8",
    )
    competition = SimpleNamespace(
        observation_names=competition_names, test_image_dir=test_image_dir
    )
    return root, competition


def test_chrotel_loader_builds_single_semantic_target_and_checks_hashes(
    tmp_path: Path,
) -> None:
    root, competition = _toy_chrotel_dataset(tmp_path)

    dataset, metadata = load_chrotel_dataset(
        root, competition, repository_root=tmp_path
    )

    assert dataset.observation_names == ("chrotel_20120417_090150_ha.png",)
    assert len(dataset.image_views) == 1
    assert len(dataset.annotations_by_image_id[dataset.image_views[0].image_id]) == 1
    assert metadata["included_observations"] == 1
    assert metadata["competition_date_overlap"] == 0
    assert metadata["inference_uses_external_images"] is False


def test_chrotel_loader_rejects_competition_date_overlap(tmp_path: Path) -> None:
    root, competition = _toy_chrotel_dataset(
        tmp_path, ("20120417_000000_Bh.jpg",)
    )

    with pytest.raises(ValueError, match="date overlaps competition data"):
        load_chrotel_dataset(root, competition, repository_root=tmp_path)
