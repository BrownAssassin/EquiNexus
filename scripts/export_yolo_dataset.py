# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from filament_baseline.data import CompetitionDataset, load_dataset
from filament_baseline.split import ObservationSplit, create_observation_split


def _safe_view_name(file_name: str, image_id: str) -> str:
    view_digest = hashlib.sha1(image_id.encode("utf-8")).hexdigest()[:12]
    return f"{Path(file_name).stem}__{view_digest}"


def _polygon_area(polygon: list[float]) -> float:
    points = list(zip(polygon[0::2], polygon[1::2], strict=True))
    return abs(
        sum(
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[index][1] * points[(index + 1) % len(points)][0]
            for index in range(len(points))
        )
    ) * 0.5


def _labels_for_view(dataset: CompetitionDataset, image_id: str) -> list[str]:
    view = next(view for view in dataset.image_views if view.image_id == image_id)
    output = []
    for annotation in dataset.annotations_by_image_id[image_id]:
        segmentation = annotation.get("segmentation")
        if not isinstance(segmentation, list):
            raise ValueError(f"Expected polygon annotations for YOLO export: {view.file_name}")
        polygons = [
            polygon
            for polygon in segmentation
            if isinstance(polygon, list) and len(polygon) >= 6 and len(polygon) % 2 == 0
        ]
        if not polygons:
            continue
        polygon = max(polygons, key=_polygon_area)
        values = []
        for x, y in zip(polygon[0::2], polygon[1::2], strict=True):
            values.extend(
                (
                    min(max(float(x) / view.width, 0.0), 1.0),
                    min(max(float(y) / view.height, 0.0), 1.0),
                )
            )
        output.append("0 " + " ".join(f"{value:.7f}" for value in values))
    return output


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def export_yolo_dataset(
    dataset: CompetitionDataset,
    split: ObservationSplit,
    output_root: str | Path,
    *,
    limit_per_partition: int | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root).expanduser().resolve()
    processed_root = (REPO_ROOT / "data" / "processed").resolve()
    if not _within(output_root, processed_root):
        raise ValueError(f"Exported detector data must remain under data/processed: {output_root}")
    if limit_per_partition is not None and limit_per_partition < 1:
        raise ValueError("limit_per_partition must be positive")
    for partition in ("train", "val"):
        for kind in ("images", "labels"):
            stale_partition = output_root / kind / partition
            if not _within(stale_partition, output_root):
                raise ValueError(f"Generated partition escaped the export root: {stale_partition}")
            if stale_partition.exists():
                shutil.rmtree(stale_partition)
    rows = []
    counts = {"train": {"views": 0, "instances": 0}, "val": {"views": 0, "instances": 0}}
    partitions = {"train": set(split.training_names), "val": set(split.validation_names)}
    for partition, observation_names in partitions.items():
        views = [view for view in dataset.image_views if view.file_name in observation_names]
        views.sort(key=lambda view: (view.file_name, view.image_id))
        if limit_per_partition is not None:
            views = views[:limit_per_partition]
        for view in views:
            sample = _safe_view_name(view.file_name, view.image_id)
            image_source = dataset.train_image_dir / view.file_name
            image_target = output_root / "images" / partition / f"{sample}.jpg"
            label_target = output_root / "labels" / partition / f"{sample}.txt"
            _link_or_copy(image_source, image_target)
            labels = _labels_for_view(dataset, view.image_id)
            label_target.parent.mkdir(parents=True, exist_ok=True)
            label_target.write_text("\n".join(labels) + "\n", encoding="utf-8")
            counts[partition]["views"] += 1
            counts[partition]["instances"] += len(labels)
            rows.append(
                {
                    "partition": partition,
                    "observation": view.file_name,
                    "view_id": view.image_id,
                    "sample": sample,
                    "instances": len(labels),
                }
            )

    output_root.mkdir(parents=True, exist_ok=True)
    data_yaml = (
        f"path: {output_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n  0: filament\n"
    )
    (output_root / "data.yaml").write_text(data_yaml, encoding="utf-8")
    view_manifest = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    result = {
        "split_sha256": split.sha256,
        "limit_per_partition": limit_per_partition,
        "grouping": "original observation; all annotator views stay together",
        "train_observations": len(split.training_names),
        "validation_observations": len(split.validation_names),
        "counts": counts,
        "view_manifest_sha256": hashlib.sha256(view_manifest).hexdigest(),
        "data_yaml": str(output_root / "data.yaml"),
        "views": rows,
    }
    (output_root / "export_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Export grouped COCO instance annotations to YOLO segmentation format.")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, default=REPO_ROOT / "data" / "splits")
    parser.add_argument("--limit-per-partition", type=int, default=None)
    args = parser.parse_args()
    data_root = (args.data_root or REPO_ROOT / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026").expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    split_dir = args.split_dir.expanduser().resolve()
    if not data_root.is_relative_to((REPO_ROOT / "data").resolve()):
        raise ValueError(f"Local competition data must remain under data/: {data_root}")
    if not output_root.is_relative_to((REPO_ROOT / "data" / "processed").resolve()):
        raise ValueError(f"Local exported data must remain under data/processed/: {output_root}")
    if not split_dir.is_relative_to((REPO_ROOT / "data" / "splits").resolve()):
        raise ValueError(f"Local split files must remain under data/splits/: {split_dir}")
    dataset = load_dataset(data_root)
    config = json.loads((REPO_ROOT / "configs" / "validation-baseline.json").read_text())
    split = create_observation_split(
        dataset,
        split_dir,
        seed=int(config["seed"]),
        validation_fraction=float(config["validation_fraction"]),
    )
    if split.sha256 != config["expected_split_sha256"]:
        raise RuntimeError(f"Frozen split changed: {split.sha256}")
    result = export_yolo_dataset(
        dataset,
        split,
        output_root,
        limit_per_partition=args.limit_per_partition,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "views"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
