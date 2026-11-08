# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from filament_baseline.data import load_dataset
from filament_baseline.external_data import load_swefil_dataset
from filament_baseline.split import create_observation_split

SOURCE_REVISION = "bc818c69a79266ad0fa91a539359485af57967cf"
SOURCE_ID = "mrinanksivakumar/equinexus-swefil-date-disjoint"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_dataset(
    *,
    repo_root: Path = REPO_ROOT,
    competition_root: Path | None = None,
    swefil_root: Path | None = None,
    output_root: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.expanduser().resolve()
    competition_root = (
        competition_root or repo_root / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026"
    ).expanduser().resolve()
    swefil_root = (swefil_root or repo_root / "data" / "external" / "SWEFil").expanduser().resolve()
    output_root = (
        output_root or repo_root / "data" / "processed" / "swefil-kaggle-date-disjoint"
    ).expanduser().resolve()
    if not swefil_root.is_relative_to(repo_root / "data" / "external"):
        raise ValueError(f"SWEFil source must remain under data/external/: {swefil_root}")
    if not output_root.is_relative_to(repo_root / "data" / "processed"):
        raise ValueError(f"Kaggle package must remain under data/processed/: {output_root}")
    if output_root == swefil_root or output_root.is_relative_to(swefil_root):
        raise ValueError("Package output must not overwrite or nest inside the source dataset")

    config = json.loads((repo_root / "configs" / "validation-baseline.json").read_text())
    competition = load_dataset(competition_root)
    frozen = create_observation_split(
        competition,
        repo_root / "data" / "splits",
        seed=int(config["seed"]),
        validation_fraction=float(config["validation_fraction"]),
    )
    if frozen.sha256 != config["expected_split_sha256"]:
        raise RuntimeError("Frozen competition split digest changed")
    external, provenance = load_swefil_dataset(
        swefil_root,
        competition,
        frozen,
        repository_root=repo_root,
        exclude_all_competition_dates=True,
    )
    audit_path = swefil_root / "equinexus_audit_all_competition_dates_excluded.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if provenance["included_manifest_sha256"] != audit["included_manifest_sha256"]:
        raise ValueError("Strict SWEFil subset differs from the audited source manifest")
    if provenance["included_observations"] != 229 or provenance["included_filament_annotations"] != 1504:
        raise ValueError("Unexpected strict SWEFil sample size")

    included = {view.file_name for view in external.image_views}
    if len(included) != len(external.image_views):
        raise ValueError("SWEFil image filenames must be unique in the filtered package")
    payloads: dict[str, dict[str, Any]] = {}
    included_record_names: list[str] = []
    expected_annotations = 0
    for filename in ("train.json", "test.json"):
        payload = json.loads((swefil_root / filename).read_text(encoding="utf-8"))
        images = [
            record
            for record in payload["images"]
            if Path(str(record["file_name"])).name in included
        ]
        image_ids = {int(record["id"]) for record in images}
        if len(image_ids) != len(images):
            raise ValueError(f"Duplicate image ids in {filename}")
        annotations = [
            record
            for record in payload["annotations"]
            if int(record["image_id"]) in image_ids
            and int(record["category_id"]) in {
                int(category["id"])
                for category in payload["categories"]
                if str(category["name"]) in {"QRF", "IRF", "ARF"}
            }
        ]
        expected_annotations += len(annotations)
        included_record_names.extend(Path(str(record["file_name"])).name for record in images)
        payloads[filename] = {
            key: payload[key]
            for key in ("info", "licenses", "categories")
            if key in payload
        }
        payloads[filename]["images"] = images
        payloads[filename]["annotations"] = annotations
    if len(included_record_names) != len(set(included_record_names)) or set(included_record_names) != included:
        raise ValueError("Filtered COCO records do not cover the audited images exactly")
    if expected_annotations != 1504:
        raise ValueError(f"Expected 1,504 filament annotations, found {expected_annotations}")

    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Package output is not empty; inspect it before rebuilding: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    image_output = output_root / "images"
    image_output.mkdir(parents=True, exist_ok=True)
    image_hash_rows = []
    copied_bytes = 0
    for view in external.image_views:
        source = swefil_root / "images" / view.file_name
        destination = image_output / view.file_name
        shutil.copy2(source, destination)
        size = destination.stat().st_size
        copied_bytes += size
        image_hash_rows.append(
            {"file_name": view.file_name, "bytes": size, "sha256": _sha256(destination)}
        )
    image_hash_digest = hashlib.sha256(
        json.dumps(image_hash_rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if copied_bytes != audit["included_image_bytes"]:
        raise ValueError("Copied SWEFil image bytes do not match the strict audit")
    if image_hash_digest != audit["included_image_hashes_sha256"]:
        raise ValueError("Copied SWEFil image hashes do not match the strict audit")

    for filename, payload in payloads.items():
        (output_root / filename).write_text(
            json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8"
        )
    shutil.copy2(swefil_root / "README.md", output_root / "README.md")
    shutil.copy2(audit_path, output_root / audit_path.name)
    (output_root / "equinexus_revision.txt").write_text(SOURCE_REVISION + "\n", encoding="utf-8")
    provenance.update(
        {
            "filtered_package": True,
            "included_images": len(included),
            "included_annotations": expected_annotations,
            "copied_image_bytes": copied_bytes,
            "copied_image_hashes_sha256": image_hash_digest,
            "competition_split_sha256": frozen.sha256,
            "package_scope": "date-disjoint SWEFil filament masks and matching H-alpha images only",
            "competition_or_test_images_in_package": False,
            "sunspot_labels_or_categories_in_package": False,
        }
    )
    (output_root / "package-provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    attribution = (
        "Filtered SWEFil subset for EquiNexus model training.\n\n"
        "Source: Reche, A. and Cid, C. (2024), Solar filament detection, classification, "
        "and tracking with deep learning, SPAICE2024, pp. 69-74.\n"
        "Dataset card: https://huggingface.co/datasets/antonio-reche/SWEFil\n"
        f"Source revision: {SOURCE_REVISION}\n"
        "License: Creative Commons Attribution 4.0 International (CC BY 4.0).\n"
        "Attribution: This work utilizes GONG data obtained by the NSO Integrated Synoptic "
        "Program, managed by NSO/AURA/NSF with contribution from NOAA.\n\n"
        "This package includes 229 date-disjoint processed H-alpha observations and their "
        "1,504 filament instance annotations. All calendar dates represented in the "
        "competition train or test set, exact competition observation names, raw image "
        "copies, sunspot labels, and filament class targets were excluded. No public MAGFiLO "
        "annotations outside the competition-provided training set are included."
    )
    (output_root / "ATTRIBUTION.txt").write_text(attribution + "\n", encoding="utf-8")
    metadata = {
        "title": "EquiNexus SWEFil Filament Pretraining Data",
        "id": SOURCE_ID,
        "licenses": [{"name": "CC-BY-4.0"}],
        "description": attribution,
        "userSpecifiedSources": "SWEFil, https://huggingface.co/datasets/antonio-reche/SWEFil",
        "expectedUpdateFrequency": "never",
        "keywords": ["computer vision", "solar", "image segmentation"],
    }
    (output_root / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    result = {
        "output_root": str(output_root),
        "source_revision": SOURCE_REVISION,
        "source_manifest_sha256": provenance["included_manifest_sha256"],
        "competition_split_sha256": frozen.sha256,
        "images": len(included),
        "annotations": expected_annotations,
        "image_bytes": copied_bytes,
        "total_bytes": sum(path.stat().st_size for path in output_root.rglob("*") if path.is_file()),
        "license": "CC-BY-4.0",
        "private_kaggle_dataset_id": SOURCE_ID,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Package only the audited, date-disjoint SWEFil labels for Kaggle input use."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--competition-data-root", type=Path, default=None)
    parser.add_argument("--swefil-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    package_dataset(
        repo_root=args.repo_root,
        competition_root=args.competition_data_root,
        swefil_root=args.swefil_root,
        output_root=args.output_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
