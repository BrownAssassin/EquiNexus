# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import StratifiedShuffleSplit

from .data import CompetitionDataset


@dataclass(frozen=True)
class ObservationSplit:
    assignments: dict[str, str]
    manifest_path: Path
    sha256: str

    @property
    def validation_names(self) -> tuple[str, ...]:
        return tuple(
            sorted(name for name, partition in self.assignments.items() if partition == "validation")
        )

    @property
    def training_names(self) -> tuple[str, ...]:
        return tuple(
            sorted(name for name, partition in self.assignments.items() if partition == "train")
        )


def create_observation_split(
    dataset: CompetitionDataset,
    split_dir: str | Path,
    *,
    seed: int = 42,
    validation_fraction: float = 0.2,
) -> ObservationSplit:
    names = list(dataset.observation_names)
    years = [dataset.views_by_observation[name][0].year for name in names]
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=validation_fraction, random_state=seed
    )
    train_indices, validation_indices = next(splitter.split(names, years))
    assignments = {name: "train" for name in names}
    for index in validation_indices:
        assignments[names[int(index)]] = "validation"

    rows = []
    for name in names:
        view = dataset.views_by_observation[name][0]
        rows.append(
            {
                "file_name": name,
                "year": view.year,
                "site": view.site,
                "partition": assignments[name],
            }
        )

    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = split_dir / f"observation_split_seed{seed}.csv"
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=("file_name", "year", "site", "partition"),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    manifest_bytes = buffer.getvalue().encode("utf-8")
    digest = hashlib.sha256(manifest_bytes).hexdigest()

    if manifest_path.exists():
        existing = manifest_path.read_bytes()
        if existing != manifest_bytes:
            raise RuntimeError(
                f"Existing split manifest differs from the deterministic split: {manifest_path}"
            )
    else:
        manifest_path.write_bytes(manifest_bytes)

    if set(assignments.values()) != {"train", "validation"}:
        raise AssertionError("The split must contain both training and validation observations")
    if set(assignments) != set(names):
        raise AssertionError("The split manifest does not cover every original observation")
    if set(dataset.observation_names) != set(assignments):
        raise AssertionError("Observation split keys do not match COCO file_name groups")

    return ObservationSplit(assignments, manifest_path, digest)


def load_observation_split_manifest(
    dataset: CompetitionDataset,
    manifest_path: str | Path,
    *,
    allow_subset: bool = False,
) -> ObservationSplit:
    """Load an explicitly grouped train/validation manifest under data/splits."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    content = manifest_path.read_bytes()
    assignments: dict[str, str] = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["file_name", "year", "site", "partition"]:
            raise ValueError("Split manifest must contain file_name, year, site, partition columns")
        for row in reader:
            name = row["file_name"]
            if name in assignments:
                raise ValueError(f"Duplicate observation in split manifest: {name}")
            if name not in dataset.views_by_observation:
                raise ValueError(f"Unknown observation in split manifest: {name}")
            original_view = dataset.views_by_observation[name][0]
            if int(row["year"]) != original_view.year or row["site"] != original_view.site:
                raise ValueError(f"Year/site metadata changed for observation {name}")
            if row["partition"] not in {"train", "validation"}:
                raise ValueError(f"Invalid partition for observation {name}: {row['partition']}")
            assignments[name] = row["partition"]

    dataset_names = set(dataset.observation_names)
    manifest_names = set(assignments)
    if not manifest_names:
        raise ValueError("Split manifest contains no observations")
    if allow_subset:
        if not manifest_names.issubset(dataset_names):
            raise ValueError("Split manifest contains observations outside the competition dataset")
    elif manifest_names != dataset_names:
        raise ValueError("Split manifest must cover every competition training observation")
    if set(assignments.values()) != {"train", "validation"}:
        raise ValueError("Split manifest must contain both train and validation observations")

    return ObservationSplit(
        assignments=assignments,
        manifest_path=manifest_path,
        sha256=hashlib.sha256(content).hexdigest(),
    )
