# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ANNOTATION_FILENAME = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
IMAGE_SIZE = 2048


@dataclass(frozen=True)
class ImageView:
    image_id: str
    file_name: str
    width: int
    height: int
    year: int
    site: str


@dataclass(frozen=True)
class CompetitionDataset:
    root: Path
    train_image_dir: Path
    test_image_dir: Path
    annotation_path: Path
    image_views: tuple[ImageView, ...]
    annotations_by_image_id: dict[str, tuple[dict[str, Any], ...]]
    views_by_observation: dict[str, tuple[ImageView, ...]]
    categories: tuple[dict[str, Any], ...]
    image_resize_to: tuple[int, int] | None = None

    @property
    def observation_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.views_by_observation))

    @property
    def test_image_count(self) -> int:
        return sum(path.is_file() for path in self.test_image_dir.iterdir())

    @property
    def annotation_count(self) -> int:
        return sum(len(items) for items in self.annotations_by_image_id.values())


def _looks_like_dataset(root: Path) -> bool:
    return (
        (root / "train" / ANNOTATION_FILENAME).is_file()
        and (root / "train" / "train_images").is_dir()
        and (root / "test" / "test_images").is_dir()
    )


def find_dataset_root(search_root: str | Path) -> Path:
    """Find the competition input folder without relying on a fixed Kaggle mount name."""
    search_root = Path(search_root).expanduser().resolve()
    if not search_root.exists():
        raise FileNotFoundError(f"Data search path does not exist: {search_root}")
    if _looks_like_dataset(search_root):
        return search_root

    matches = sorted(
        {path.parent.parent.resolve() for path in search_root.rglob(ANNOTATION_FILENAME)}
        if search_root.is_dir()
        else set()
    )
    matches = [path for path in matches if _looks_like_dataset(path)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"Could not find {ANNOTATION_FILENAME} with train/train_images and "
            f"test/test_images below {search_root}"
        )
    rendered = "\n".join(str(path) for path in matches)
    raise RuntimeError(
        "More than one competition dataset was found; pass an explicit dataset root:\n"
        f"{rendered}"
    )


def _year_and_site(file_name: str) -> tuple[int, str]:
    stem = Path(file_name).stem
    if len(stem) < 4 or not stem[:4].isdigit():
        raise ValueError(f"Cannot derive capture year from image filename: {file_name}")
    year = int(stem[:4])
    site_match = re.fullmatch(r"\d{14}([A-Za-z]{2})", stem)
    site = site_match.group(1) if site_match else "unknown"
    return year, site


def load_dataset(data_root: str | Path) -> CompetitionDataset:
    root = find_dataset_root(data_root)
    annotation_path = root / "train" / ANNOTATION_FILENAME
    train_image_dir = root / "train" / "train_images"
    test_image_dir = root / "test" / "test_images"
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))

    image_views: list[ImageView] = []
    image_ids: set[str] = set()
    views_by_observation: dict[str, list[ImageView]] = defaultdict(list)
    for record in payload.get("images", []):
        image_id = str(record["id"])
        file_name = str(record["file_name"])
        if Path(file_name).name != file_name:
            raise ValueError(f"Expected a basename in COCO file_name, got: {file_name}")
        if image_id in image_ids:
            raise ValueError(f"Duplicate COCO image id: {image_id}")
        image_ids.add(image_id)
        year, site = _year_and_site(file_name)
        view = ImageView(
            image_id=image_id,
            file_name=file_name,
            width=int(record["width"]),
            height=int(record["height"]),
            year=year,
            site=site,
        )
        if view.width != IMAGE_SIZE or view.height != IMAGE_SIZE:
            raise ValueError(
                f"Expected {IMAGE_SIZE}x{IMAGE_SIZE}, got {view.width}x{view.height} "
                f"for {file_name}"
            )
        image_views.append(view)
        views_by_observation[file_name].append(view)

    annotations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload.get("annotations", []):
        image_id = str(annotation["image_id"])
        if image_id not in image_ids:
            raise ValueError(f"Annotation references unknown image id: {image_id}")
        if int(annotation.get("iscrowd", 0)) != 0:
            raise ValueError("The competition training annotations unexpectedly contain crowd masks")
        annotations[image_id].append(annotation)

    for image_id in image_ids:
        annotations[image_id].sort(key=lambda item: str(item["id"]))
        if not annotations[image_id]:
            raise ValueError(f"Annotator view {image_id} has no filament annotations")

    for file_name, views in views_by_observation.items():
        expected_path = train_image_dir / file_name
        if not expected_path.is_file():
            raise FileNotFoundError(f"COCO image file is missing: {expected_path}")
        views.sort(key=lambda view: view.image_id)

    test_files = [path for path in test_image_dir.iterdir() if path.is_file()]
    if not test_files:
        raise ValueError(f"No test images found in {test_image_dir}")

    return CompetitionDataset(
        root=root,
        train_image_dir=train_image_dir,
        test_image_dir=test_image_dir,
        annotation_path=annotation_path,
        image_views=tuple(sorted(image_views, key=lambda view: (view.file_name, view.image_id))),
        annotations_by_image_id={
            image_id: tuple(items) for image_id, items in annotations.items()
        },
        views_by_observation={
            file_name: tuple(views)
            for file_name, views in sorted(views_by_observation.items())
        },
        categories=tuple(payload.get("categories", [])),
    )


def inventory(dataset: CompetitionDataset) -> dict[str, int]:
    return {
        "observation_count": len(dataset.views_by_observation),
        "annotator_view_count": len(dataset.image_views),
        "annotation_count": dataset.annotation_count,
        "train_image_file_count": sum(
            path.is_file() for path in dataset.train_image_dir.iterdir()
        ),
        "test_image_file_count": dataset.test_image_count,
    }
