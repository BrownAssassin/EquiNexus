# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import hashlib
import gzip
import io
import json
import re
import struct
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .data import CompetitionDataset, ImageView
from .split import ObservationSplit


SWEFIL_REPOSITORY = "https://huggingface.co/datasets/antonio-reche/SWEFil"
SWEFIL_REVISION = "bc818c69a79266ad0fa91a539359485af57967cf"
SWEFIL_LICENSE = "CC-BY-4.0"
SWEFIL_CITATION = (
    "Reche, A. and Cid, C. (2024), Solar filament detection, classification, "
    "and tracking with deep learning, SPAICE2024, pp. 69-74."
)
HSOS_CITATION = (
    "Zhu, G., Lin, G., Wang, D. et al. Solar Filament Recognition Based on Deep Learning. "
    "Solar Physics 294, 117 (2019). https://doi.org/10.1007/s11207-019-1517-4"
)
MORDEN_DOI = "10.5281/zenodo.21582245"
MORDEN_ARCHIVE_SHA256 = "14fdd527594c3723442cc6a687eff3472a871ecb666dd7753ce8e37c565e5b3e"
MORDEN_ARCHIVE_MD5 = "498e00c9837cd01d9b3763b86e98d9d0"
MORDEN_IMAGE_PATTERN = re.compile(r"^RSM(\d{8})T\d{6}_\d{4}_HA\.png$", re.IGNORECASE)
CHROTEL_API = "https://archive.sdc.leibniz-kis.de/api/instruments/chrotel"
CHROTEL_LICENSE = "KIS SDC permits free scientific and educational use; cite and acknowledge ChroTel/KIS/HAO."
KSO_IMAGE_INDEX = "https://cesar.kso.ac.at/halpha3s/FITS/arch/2023/"
KSO_MASK_ROOT = "https://cesar.kso.ac.at/sniv/filaments/2023/"
KSO_LICENSE = "Non-profit scientific and educational use; profit use is prohibited."
KSO_ATTRIBUTION = (
    "H-alpha data / filament maps were provided by the Kanzelhoehe Observatory, "
    "University of Graz, Austria."
)
FROZEN_COMPETITION_SPLIT_SHA256 = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"


def _repo_revision(root: Path) -> str:
    revision_marker = root / "equinexus_revision.txt"
    if revision_marker.is_file():
        return revision_marker.read_text(encoding="utf-8").strip()
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"Could not read the SWEFil source revision under {root}") from exc


def _observation_stem(file_name: str) -> str:
    stem = Path(file_name).stem
    return stem.removesuffix("_raw")


def load_swefil_dataset(
    data_root: str | Path,
    competition: CompetitionDataset,
    split: ObservationSplit,
    *,
    repository_root: str | Path,
    exclude_all_competition_dates: bool = False,
    allow_readonly_external_root: bool = False,
) -> tuple[CompetitionDataset, dict[str, Any]]:
    """Load disjoint, labeled SWEFil H-alpha images for external training.

    Only the released processed image for each observation is used; its paired
    ``*_raw.jpg`` copy is excluded. Filament categories are retained and
    sunspots are discarded. Images sharing a UTC calendar date with a frozen
    validation observation or a competition test image are excluded by default.
    Cross-fit experiments can exclude every competition train and test date.
    No MAGFiLO annotations are read from SWEFil.
    """
    root = Path(data_root).expanduser().resolve()
    repository_root = Path(repository_root).expanduser().resolve()
    if not root.is_relative_to(repository_root) and not allow_readonly_external_root:
        raise ValueError(f"External data must stay inside the workspace: {root}")
    if not (root / "images").is_dir():
        raise FileNotFoundError(f"SWEFil image directory is missing: {root / 'images'}")
    readme_path = root / "README.md"
    if not readme_path.is_file() or "cc-by-4.0" not in readme_path.read_text(
        encoding="utf-8"
    ).lower():
        raise ValueError("SWEFil source card does not declare CC BY 4.0")

    validation_dates = {name[:8] for name in split.validation_names}
    competition_train_dates = {name[:8] for name in competition.observation_names}
    test_names = {path.name for path in competition.test_image_dir.iterdir() if path.is_file()}
    test_dates = {name[:8] for name in test_names}
    blocked_dates = (
        competition_train_dates | test_dates
        if exclude_all_competition_dates
        else validation_dates | test_dates
    )
    all_competition_stems = {
        Path(name).stem for name in competition.views_by_observation
    } | {Path(name).stem for name in test_names}

    payloads: list[dict[str, Any]] = []
    for split_name in ("train.json", "test.json"):
        path = root / split_name
        if not path.is_file():
            raise FileNotFoundError(f"SWEFil annotation file is missing: {path}")
        payloads.append(json.loads(path.read_text(encoding="utf-8")))

    kept_views: list[ImageView] = []
    annotations_by_id: dict[str, tuple[dict[str, Any], ...]] = {}
    views_by_observation: dict[str, tuple[ImageView, ...]] = {}
    category_counts: Counter[str] = Counter()
    source_image_records = 0
    raw_copy_count = 0
    blocked_date_count = 0
    exact_overlap_count = 0
    included_manifest: list[dict[str, Any]] = []

    for payload_index, payload in enumerate(payloads):
        categories = {int(item["id"]): str(item["name"]) for item in payload["categories"]}
        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in payload["annotations"]:
            annotations_by_image[int(annotation["image_id"])].append(annotation)

        for record in payload["images"]:
            source_image_records += 1
            basename = Path(str(record["file_name"])).name
            if Path(basename).stem.endswith("_raw"):
                raw_copy_count += 1
                continue
            observation = _observation_stem(basename)
            day = observation[:8]
            if observation in all_competition_stems:
                exact_overlap_count += 1
                continue
            if day in blocked_dates:
                blocked_date_count += 1
                continue
            image_path = root / "images" / basename
            if not image_path.is_file():
                raise FileNotFoundError(f"SWEFil image is missing: {image_path}")
            width, height = int(record["width"]), int(record["height"])
            if (width, height) != (2048, 2048):
                raise ValueError(f"Expected 2048x2048 SWEFil image {basename}, got {width}x{height}")

            view_id = f"swefil-{payload_index}-{record['id']}"
            view = ImageView(
                image_id=view_id,
                file_name=basename,
                width=width,
                height=height,
                year=int(day[:4]),
                site=observation[-2:],
            )
            filament_annotations = []
            for annotation in annotations_by_image.get(int(record["id"]), []):
                category = categories[int(annotation["category_id"])]
                if category not in {"QRF", "IRF", "ARF"}:
                    continue
                segmentation = annotation.get("segmentation")
                if not isinstance(segmentation, list):
                    raise ValueError(f"Expected polygon masks in SWEFil: {basename}")
                if not any(isinstance(polygon, list) and len(polygon) >= 6 for polygon in segmentation):
                    continue
                filament_annotations.append(annotation)
                category_counts[category] += 1
            if not filament_annotations:
                continue
            kept_views.append(view)
            annotations_by_id[view_id] = tuple(filament_annotations)
            views_by_observation[basename] = (view,)
            included_manifest.append(
                {
                    "annotation_source": "test.json" if payload_index else "train.json",
                    "file_name": basename,
                    "view_id": view_id,
                    "filament_annotations": len(filament_annotations),
                }
            )

    revision = _repo_revision(root)
    if revision != SWEFIL_REVISION:
        raise ValueError(f"Unexpected SWEFil revision: {revision} != {SWEFIL_REVISION}")
    if not kept_views:
        raise ValueError("No SWEFil observations remain after leakage filtering")
    canonical_manifest = json.dumps(
        included_manifest, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    metadata = {
        "name": "SWEFil",
        "repository": SWEFIL_REPOSITORY,
        "revision": revision,
        "license": SWEFIL_LICENSE,
        "citation": SWEFIL_CITATION,
        "acknowledgement": (
            "This work utilizes GONG data obtained by the NSO Integrated Synoptic Program, "
            "managed by NSO/AURA/NSF with contribution from NOAA."
        ),
        "source_image_records": source_image_records,
        "raw_copies_excluded": raw_copy_count,
        "exact_competition_observation_overlaps_excluded": exact_overlap_count,
        "same_validation_or_test_date_images_excluded": blocked_date_count,
        "included_observations": len(kept_views),
        "included_filament_annotations": sum(category_counts.values()),
        "included_by_category": dict(sorted(category_counts.items())),
        "excluded_calendar_dates": len(blocked_dates),
        "date_exclusion_policy": (
            "all competition training and test dates"
            if exclude_all_competition_dates
            else "frozen validation and competition test dates"
        ),
        "included_manifest_sha256": hashlib.sha256(canonical_manifest).hexdigest(),
        "inference_uses_external_images": False,
        "sunspot_annotations_used": False,
        "competition_ground_truth_used": False,
    }
    dataset = CompetitionDataset(
        root=root,
        train_image_dir=root / "images",
        test_image_dir=root / "images",
        annotation_path=root / "train.json",
        image_views=tuple(sorted(kept_views, key=lambda view: view.file_name)),
        annotations_by_image_id=annotations_by_id,
        views_by_observation=dict(sorted(views_by_observation.items())),
        categories=tuple(
            {"id": index, "name": name}
            for index, name in enumerate(("QRF", "IRF", "ARF"), start=1)
        ),
    )
    return dataset, metadata


def load_hsos_dataset(
    data_root: str | Path,
    competition: CompetitionDataset,
    split: ObservationSplit,
    *,
    repository_root: str | Path,
) -> tuple[CompetitionDataset, dict[str, Any]]:
    """Load date-disjoint binary filament masks from the public HSOS archive.

    The source masks are semantic binary annotations, so connected mask regions
    are represented as instance polygons only to reuse the patch-training path.
    This approximation is recorded in provenance and evaluated against the
    frozen competition scorer before any use in a submission model.
    """
    root = Path(data_root).expanduser().resolve()
    repository_root = Path(repository_root).expanduser().resolve()
    if not root.is_relative_to(repository_root):
        raise ValueError(f"External data must stay inside the workspace: {root}")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"HSOS download manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("split_sha256") != split.sha256:
        raise ValueError("HSOS manifest was prepared against a different frozen split")

    all_competition_names = set(competition.observation_names)
    all_competition_names.update(
        path.name for path in competition.test_image_dir.iterdir() if path.is_file()
    )
    blocked_dates = {name[:8] for name in all_competition_names}
    prepared_dir = root / "prepared_2048"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    views: list[ImageView] = []
    annotations_by_id: dict[str, tuple[dict[str, Any], ...]] = {}
    views_by_name: dict[str, tuple[ImageView, ...]] = {}
    accepted_rows: list[dict[str, Any]] = []
    disk = np.zeros((2048, 2048), dtype=np.uint8)
    cv2.circle(disk, (1023, 1023), 950, 1, thickness=-1)

    for row_index, record in enumerate(manifest.get("downloads", [])):
        file_name = str(record["image_file"])
        day = str(record["date"])
        if day in blocked_dates:
            raise ValueError(f"HSOS image shares a date with competition data: {file_name}")
        image_path = (root / record["image_path"]).resolve()
        mask_path = (root / record["mask_path"]).resolve()
        if not image_path.is_relative_to(root) or not mask_path.is_relative_to(root):
            raise ValueError("HSOS manifest contains a path outside its data directory")
        image_bytes = image_path.read_bytes()
        mask_bytes = mask_path.read_bytes()
        if hashlib.sha256(image_bytes).hexdigest() != record["image_sha256"]:
            raise ValueError(f"HSOS image checksum mismatch: {image_path}")
        if hashlib.sha256(mask_bytes).hexdigest() != record["mask_sha256"]:
            raise ValueError(f"HSOS mask checksum mismatch: {mask_path}")
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None or image.shape != mask.shape:
            raise ValueError(f"Invalid or misaligned HSOS image-mask pair: {file_name}")
        if (
            min(image.shape) < 1024
            or max(image.shape) > 2048
            or max(image.shape) / min(image.shape) > 1.02
        ):
            raise ValueError(f"Unexpected HSOS source dimensions: {file_name} {image.shape}")
        image = cv2.resize(image, (2048, 2048), interpolation=cv2.INTER_CUBIC)
        binary = cv2.resize(mask, (2048, 2048), interpolation=cv2.INTER_NEAREST)
        binary = ((binary > 127) & (disk > 0)).astype(np.uint8)
        prepared_path = prepared_dir / file_name
        if not cv2.imwrite(
            str(prepared_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]
        ):
            raise OSError(f"Could not write resized HSOS image: {prepared_path}")

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        annotations = []
        for contour in contours:
            if cv2.contourArea(contour) < 6.0:
                continue
            polygon = cv2.approxPolyDP(contour, 1.5, True).reshape(-1, 2)
            if len(polygon) < 3:
                continue
            annotations.append(
                {"segmentation": [polygon.astype(float).reshape(-1).tolist()]}
            )
        if not annotations:
            continue
        view_id = f"hsos-{row_index}"
        view = ImageView(
            image_id=view_id,
            file_name=file_name,
            width=2048,
            height=2048,
            year=int(day[:4]),
            site="BB",
        )
        views.append(view)
        annotations_by_id[view_id] = tuple(annotations)
        views_by_name[file_name] = (view,)
        accepted_rows.append(
            {
                "date": day,
                "file_name": file_name,
                "filament_components": len(annotations),
                "image_sha256": record["image_sha256"],
                "mask_sha256": record["mask_sha256"],
            }
        )

    if not views:
        raise ValueError("No usable HSOS masks remain after validation and conversion")
    canonical = json.dumps(accepted_rows, sort_keys=True, separators=(",", ":")).encode()
    metadata = {
        "name": "HSOS BBSO H-alpha filament-unet dataset",
        "source_image_index": "https://sun.bao.ac.cn/hsos_data/download/filaments-unet-dataset/img/",
        "source_mask_index": "https://sun.bao.ac.cn/hsos_data/download/filaments-unet-dataset/mask/",
        "source_code": "https://github.com/GF-Zhu/Filament-Unet",
        "citation": HSOS_CITATION,
        "license_status": "Public archive; no standalone dataset license is stated in the source listing.",
        "annotation_type": "binary filament masks; connected components approximate instances",
        "conversion": {
            "source_mask_threshold": 127,
            "solar_disk_center_xy": [1023, 1023],
            "solar_disk_radius_pixels": 950,
            "target_size": [2048, 2048],
            "image_resize": "OpenCV INTER_CUBIC",
            "mask_resize": "OpenCV INTER_NEAREST",
            "prepared_image_format": "JPEG quality 95",
            "contours": "RETR_EXTERNAL and CHAIN_APPROX_SIMPLE",
            "minimum_contour_area": 6.0,
            "polygon_approximation_epsilon": 1.5,
        },
        "source_selection_manifest_sha256": manifest["selected_source_manifest_sha256"],
        "included_manifest_sha256": hashlib.sha256(canonical).hexdigest(),
        "included_observations": len(views),
        "included_filament_components": sum(len(items) for items in annotations_by_id.values()),
        "blocked_competition_dates": len(blocked_dates),
        "all_competition_dates_excluded": True,
        "competition_ground_truth_used": False,
        "inference_uses_external_images": False,
    }
    converted_manifest = prepared_dir / "annotations_manifest.json"
    converted_manifest.write_text(
        json.dumps({"metadata": metadata, "observations": accepted_rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    dataset = CompetitionDataset(
        root=root,
        train_image_dir=prepared_dir,
        test_image_dir=prepared_dir,
        annotation_path=converted_manifest,
        image_views=tuple(sorted(views, key=lambda view: view.file_name)),
        annotations_by_image_id=annotations_by_id,
        views_by_observation=dict(sorted(views_by_name.items())),
        categories=({"id": 1, "name": "filament"},),
    )
    return dataset, metadata


def load_morden_ahas_dataset(
    data_root: str | Path,
    competition: CompetitionDataset,
    *,
    repository_root: str | Path,
) -> tuple[CompetitionDataset, dict[str, Any]]:
    """Load date-disjoint, CC-BY MORDEN/AHAS training masks only.

    The bundle's validation/test images are intentionally ignored. The 80
    anonymized MHAS samples mixed into AHAS/train are excluded because their
    individual observation dates cannot be independently audited.
    """
    root = Path(data_root).expanduser().resolve()
    repository_root = Path(repository_root).expanduser().resolve()
    if not root.is_relative_to(repository_root / "data" / "external"):
        raise ValueError(f"MORDEN data must stay under data/external/: {root}")
    manifest_path = root / "audit_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"MORDEN audit manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("archive", {}).get("sha256") != MORDEN_ARCHIVE_SHA256:
        raise ValueError("MORDEN archive SHA-256 does not match the audited Zenodo bundle")
    if manifest.get("archive", {}).get("md5") != MORDEN_ARCHIVE_MD5:
        raise ValueError("MORDEN archive MD5 does not match the published Zenodo checksum")
    if manifest.get("license") != "CC-BY-4.0":
        raise ValueError("MORDEN dataset record must declare CC-BY-4.0")
    audit = manifest.get("audit", {})
    if any(
        audit.get(key) != 0
        for key in (
            "same_calendar_date_overlaps",
            "exact_byte_identical_images",
            "exact_decoded_2048_pixel_overlaps",
        )
    ):
        raise ValueError("MORDEN audit manifest reports a competition data overlap")
    manifest_observations = manifest.get("observations", [])
    manifest_names = [str(row.get("file_name", "")) for row in manifest_observations]
    if len(manifest_names) != len(set(manifest_names)):
        raise ValueError("MORDEN audit manifest contains duplicate observation names")
    manifest_digest = hashlib.sha256(
        json.dumps(manifest_observations, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if manifest_digest != manifest.get("selection_manifest_sha256"):
        raise ValueError("MORDEN selection manifest digest does not match its contents")

    source_root = root / "extracted" / "AHAS" / "train"
    competition_names = set(competition.observation_names)
    competition_names.update(
        path.name for path in competition.test_image_dir.iterdir() if path.is_file()
    )
    competition_dates = {name[:8] for name in competition_names}
    views: list[ImageView] = []
    annotations_by_id: dict[str, tuple[dict[str, Any], ...]] = {}
    views_by_name: dict[str, tuple[ImageView, ...]] = {}
    included_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(manifest_observations):
        file_name = str(row["file_name"])
        match = MORDEN_IMAGE_PATTERN.fullmatch(file_name)
        if match is None:
            raise ValueError(f"MORDEN manifest contains an unaudited image name: {file_name}")
        date = match.group(1)
        if date[:4] not in {"2024", "2025"}:
            raise ValueError(f"Unexpected MORDEN/CHASE observation year: {file_name}")
        if date in competition_dates:
            raise ValueError(f"MORDEN image date overlaps competition data: {file_name}")
        image_path = (source_root / "images" / file_name).resolve()
        mask_path = (source_root / "masks" / file_name).resolve()
        if not image_path.is_relative_to(root) or not mask_path.is_relative_to(root):
            raise ValueError("MORDEN manifest path escaped its external data root")
        image_bytes = image_path.read_bytes()
        mask_bytes = mask_path.read_bytes()
        if hashlib.sha256(image_bytes).hexdigest() != row["image_sha256"]:
            raise ValueError(f"MORDEN image checksum mismatch: {file_name}")
        if hashlib.sha256(mask_bytes).hexdigest() != row["mask_sha256"]:
            raise ValueError(f"MORDEN mask checksum mismatch: {file_name}")
        mask = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != (2048, 2048):
            raise ValueError(f"Invalid MORDEN mask shape for {file_name}")
        binary = (mask > 127).astype(np.uint8)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        annotations = []
        for contour in contours:
            if cv2.contourArea(contour) < 6.0:
                continue
            polygon = cv2.approxPolyDP(contour, 1.5, True).reshape(-1, 2)
            if len(polygon) < 3:
                continue
            annotations.append(
                {"segmentation": [polygon.astype(float).reshape(-1).tolist()]}
            )
        if not annotations:
            continue
        view_id = f"morden-ahas-{row_index}"
        view = ImageView(
            image_id=view_id,
            file_name=file_name,
            width=2048,
            height=2048,
            year=int(date[:4]),
            site="CHASE",
        )
        views.append(view)
        annotations_by_id[view_id] = tuple(annotations)
        views_by_name[file_name] = (view,)
        included_rows.append(row)

    if not views:
        raise ValueError("No usable MORDEN AHAS training masks remain")
    if len(views) != 695 or len({view.file_name[:11] for view in views}) != 695:
        raise ValueError(f"Expected 695 date-unique AHAS observations, found {len(views)}")
    canonical = json.dumps(included_rows, sort_keys=True, separators=(",", ":")).encode()
    metadata = {
        "name": "MORDEN AHAS CHASE training data",
        "repository": "https://github.com/irisaltHu/MORDEN",
        "dataset_record": "https://zenodo.org/records/21582245",
        "doi": MORDEN_DOI,
        "license": "CC-BY-4.0",
        "citation": "Hu et al., A Modern ConvNet for Solar Filament Detection, arXiv:2607.24525 (2026).",
        "observation_source": "CHASE, 2024-2025, as documented by the dataset paper",
        "annotation_type": "automated semantic masks, split into connected-component polygons",
        "selected_partition": "AHAS/train only",
        "excluded_partitions": ["AHAS/val", "AHAS/test", "MHAS/train", "MHAS/val", "MHAS/test"],
        "anonymized_ahas_train_samples_excluded": 80,
        "included_observations": len(views),
        "included_filament_components": sum(len(items) for items in annotations_by_id.values()),
        "included_manifest_sha256": hashlib.sha256(canonical).hexdigest(),
        "competition_train_test_calendar_dates": len(competition_dates),
        "competition_calendar_date_overlaps": 0,
        "exact_competition_image_overlaps": 0,
        "competition_ground_truth_used": False,
        "inference_uses_external_images": False,
        "acknowledgement": "CHASE data courtesy of the Chinese H-alpha Solar Explorer mission.",
    }
    dataset = CompetitionDataset(
        root=root,
        train_image_dir=source_root / "images",
        test_image_dir=source_root / "images",
        annotation_path=manifest_path,
        image_views=tuple(sorted(views, key=lambda view: view.file_name)),
        annotations_by_image_id=annotations_by_id,
        views_by_observation=dict(sorted(views_by_name.items())),
        categories=({"id": 1, "name": "filament"},),
        image_resize_to=(2048, 2048),
    )
    return dataset, metadata


def load_chrotel_dataset(
    data_root: str | Path,
    competition: CompetitionDataset,
    *,
    repository_root: str | Path,
) -> tuple[CompetitionDataset, dict[str, Any]]:
    """Load KIS SDC ChroTel H-alpha masks after strict calendar-date exclusion.

    The public ChroTel mask products are semantic predictions from a published
    semi-supervised pipeline, not manual instance annotations. Each image's
    connected foreground contours are therefore represented as one noisy
    semantic target for pretraining; competition labels teach instance splits
    during the subsequent fine-tuning stage.
    """
    root = Path(data_root).expanduser().resolve()
    repository_root = Path(repository_root).expanduser().resolve()
    if not root.is_relative_to(repository_root / "data" / "external"):
        raise ValueError(f"ChroTel data must stay under data/external/: {root}")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"ChroTel manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("ChroTel external data preparation is incomplete")
    if manifest.get("all_competition_dates_excluded") is not True:
        raise ValueError("ChroTel manifest does not certify full date exclusion")
    if manifest.get("competition_date_overlap") != 0:
        raise ValueError("ChroTel manifest reports a competition date overlap")
    if (
        manifest.get("selection", {}).get("frozen_competition_split_sha256")
        != FROZEN_COMPETITION_SPLIT_SHA256
    ):
        raise ValueError("ChroTel selection was not audited against the frozen competition split")

    competition_names = set(competition.observation_names)
    competition_names.update(
        path.name for path in competition.test_image_dir.iterdir() if path.is_file()
    )
    blocked_dates = {name[:8] for name in competition_names}
    rows = manifest.get("observations", [])
    names = [str(row.get("file_name", "")) for row in rows]
    if not names or len(names) != len(set(names)):
        raise ValueError("ChroTel manifest must contain unique observation filenames")
    if any(not re.fullmatch(r"chrotel_\d{8}_\d{6}_ha\.png", name) for name in names):
        raise ValueError("ChroTel manifest contains an unexpected image filename")

    expected_manifest_hash = manifest.get("selection_manifest_sha256")
    canonical_selection = json.dumps(
        [
            {key: row[key] for key in ("observation_id", "date", "file_name", "source_timestamp")}
            for row in rows
        ],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if hashlib.sha256(canonical_selection).hexdigest() != expected_manifest_hash:
        raise ValueError("ChroTel selection manifest digest does not match its contents")

    views: list[ImageView] = []
    annotations_by_id: dict[str, tuple[dict[str, Any], ...]] = {}
    views_by_name: dict[str, tuple[ImageView, ...]] = {}
    included_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        date = str(row["date"])
        if date in blocked_dates:
            raise ValueError(f"ChroTel image date overlaps competition data: {date}")
        file_name = str(row["file_name"])
        image_path = (root / "images" / file_name).resolve()
        mask_path = (root / "masks" / file_name).resolve()
        if not image_path.is_relative_to(root) or not mask_path.is_relative_to(root):
            raise ValueError("ChroTel manifest path escaped its external data root")
        image_bytes = image_path.read_bytes()
        mask_bytes = mask_path.read_bytes()
        if hashlib.sha256(image_bytes).hexdigest() != row["image_sha256"]:
            raise ValueError(f"ChroTel image checksum mismatch: {file_name}")
        if hashlib.sha256(mask_bytes).hexdigest() != row["mask_sha256"]:
            raise ValueError(f"ChroTel mask checksum mismatch: {file_name}")
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None or image.shape != (2048, 2048) or mask.shape != image.shape:
            raise ValueError(f"Invalid ChroTel image or mask dimensions: {file_name}")
        binary = (mask > 127).astype(np.uint8)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        polygons = []
        for contour in contours:
            if cv2.contourArea(contour) < 20.0:
                continue
            polygon = cv2.approxPolyDP(contour, 1.5, True).reshape(-1, 2)
            if len(polygon) >= 3:
                polygons.append(polygon.astype(float).reshape(-1).tolist())
        if not polygons:
            continue
        view_id = f"chrotel-{index}"
        view = ImageView(
            image_id=view_id,
            file_name=file_name,
            width=2048,
            height=2048,
            year=int(date[:4]),
            site="CHROTEL",
        )
        annotations_by_id[view_id] = ({"segmentation": polygons},)
        views_by_name[file_name] = (view,)
        views.append(view)
        included_rows.append(
            {
                "date": date,
                "file_name": file_name,
                "image_sha256": row["image_sha256"],
                "mask_sha256": row["mask_sha256"],
                "mask_area_fraction": float(binary.mean()),
                "semantic_contours": len(polygons),
            }
        )
    if not views:
        raise ValueError("No usable ChroTel masks remain after validation")
    included_digest = hashlib.sha256(
        json.dumps(included_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    metadata = {
        "name": "ChroTel H-alpha filament segmentations",
        "source_api": CHROTEL_API,
        "source_policy": CHROTEL_LICENSE,
        "citation": "Diercke et al. (2024), A&A 686, A213, https://doi.org/10.1051/0004-6361/202348314",
        "acknowledgement": (
            "ChroTel is operated by the Leibniz Institute for Solar Physics (KIS), "
            "Freiburg, Germany, at the Spanish Observatorio del Teide, Tenerife, Canary Islands. "
            "The ChroTel filtergraph was developed by KIS with the High Altitude Observatory (HAO), Boulder, CO."
        ),
        "annotation_type": "noisy semantic masks generated by a semi-supervised method, not manual instance labels",
        "pretraining_representation": "all retained connected contours are one semantic pseudo-instance per image",
        "source_observations": len(rows),
        "included_observations": len(views),
        "included_semantic_contours": sum(row["semantic_contours"] for row in included_rows),
        "excluded_empty_masks": len(rows) - len(views),
        "blocked_competition_dates": len(blocked_dates),
        "competition_date_overlap": 0,
        "all_competition_dates_excluded": True,
        "selection_manifest_sha256": expected_manifest_hash,
        "included_manifest_sha256": included_digest,
        "competition_ground_truth_used": False,
        "inference_uses_external_images": False,
    }
    dataset = CompetitionDataset(
        root=root,
        train_image_dir=root / "images",
        test_image_dir=root / "images",
        annotation_path=manifest_path,
        image_views=tuple(sorted(views, key=lambda view: view.file_name)),
        annotations_by_image_id=annotations_by_id,
        views_by_observation=dict(sorted(views_by_name.items())),
        categories=({"id": 1, "name": "filament"},),
    )
    return dataset, metadata


def load_flat_unet_dataset(
    data_root: str | Path,
    competition: CompetitionDataset,
    *,
    repository_root: str | Path,
) -> tuple[CompetitionDataset, dict[str, Any]]:
    """Load date-disjoint binary Flat U-Net masks as weak semantic targets.

    The external masks do not distinguish individual filaments. Connected mask
    contours are represented as a single pseudo-instance for pretraining; the
    subsequent competition fine-tuning step supplies genuine instance labels.
    """
    root = Path(data_root).expanduser().resolve()
    repository_root = Path(repository_root).expanduser().resolve()
    if not root.is_relative_to(repository_root / "data" / "external"):
        raise ValueError(f"Flat U-Net data must stay under data/external/: {root}")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Flat U-Net manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Flat U-Net external data preparation is incomplete")
    if manifest.get("all_competition_dates_excluded") is not True:
        raise ValueError("Flat U-Net manifest does not certify full competition-date exclusion")
    if manifest.get("competition_date_overlap") != 0:
        raise ValueError("Flat U-Net manifest reports a competition-date overlap")
    if manifest.get("competition_split_sha256") != FROZEN_COMPETITION_SPLIT_SHA256:
        raise ValueError("Flat U-Net manifest was prepared against a different frozen split")

    competition_names = set(competition.observation_names)
    competition_names.update(
        path.name for path in competition.test_image_dir.iterdir() if path.is_file()
    )
    blocked_dates = {name[:8] for name in competition_names}
    rows = manifest.get("observations", [])
    names = [str(row.get("file_name", "")) for row in rows]
    if not names or len(names) != len(set(names)):
        raise ValueError("Flat U-Net manifest must contain unique observation filenames")
    if len(rows) != manifest.get("prepared_observations"):
        raise ValueError("Flat U-Net manifest observation count is inconsistent")
    canonical_rows = json.dumps(
        rows, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if hashlib.sha256(canonical_rows).hexdigest() != manifest.get("selection_manifest_sha256"):
        raise ValueError("Flat U-Net selection manifest digest does not match its contents")

    views: list[ImageView] = []
    annotations_by_id: dict[str, tuple[dict[str, Any], ...]] = {}
    views_by_name: dict[str, tuple[ImageView, ...]] = {}
    included_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        date = str(row.get("date", ""))
        file_name = str(row.get("image", {}).get("file_name", row.get("file_name", "")))
        mask_file_name = str(row.get("mask", {}).get("file_name", row.get("file_name", "")))
        if not re.fullmatch(r"\d{8}", date) or date in blocked_dates:
            raise ValueError(f"Flat U-Net data includes an invalid or blocked date: {date}")
        stem = Path(file_name).stem
        expected_match = re.fullmatch(
            r"bbso_halph_fl_(\d{8})_(\d{6})\.(?:png|jpe?g)", file_name, re.IGNORECASE
        )
        if (
            Path(file_name).name != file_name
            or Path(mask_file_name).name != mask_file_name
            or expected_match is None
            or expected_match.group(1) != date
            or stem != str(row.get("observation_id", ""))
        ):
            raise ValueError(f"Unexpected Flat U-Net observation filename: {file_name}")
        image_path = (root / "images" / file_name).resolve()
        mask_path = (root / "masks" / mask_file_name).resolve()
        if not image_path.is_relative_to(root) or not mask_path.is_relative_to(root):
            raise ValueError("Flat U-Net manifest path escaped its external data root")
        image_bytes = image_path.read_bytes()
        mask_bytes = mask_path.read_bytes()
        if hashlib.sha256(image_bytes).hexdigest() != row["image"]["processed_sha256"]:
            raise ValueError(f"Flat U-Net image checksum mismatch: {file_name}")
        if hashlib.sha256(mask_bytes).hexdigest() != row["mask"]["processed_sha256"]:
            raise ValueError(f"Flat U-Net mask checksum mismatch: {file_name}")
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None or image.shape != (2048, 2048) or mask.shape != image.shape:
            raise ValueError(f"Invalid Flat U-Net image or mask dimensions: {file_name}")
        binary = (mask > 127).astype(np.uint8)
        if not np.any(binary):
            raise ValueError(f"Flat U-Net mask is empty: {file_name}")
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        polygons = []
        for contour in contours:
            if cv2.contourArea(contour) < 20.0:
                continue
            polygon = cv2.approxPolyDP(contour, 1.5, True).reshape(-1, 2)
            if len(polygon) >= 3:
                polygons.append(polygon.astype(float).reshape(-1).tolist())
        if not polygons:
            continue
        view_id = f"flat-unet-{index}"
        view = ImageView(
            image_id=view_id,
            file_name=file_name,
            width=2048,
            height=2048,
            year=int(date[:4]),
            site="BBSO",
        )
        annotations_by_id[view_id] = ({"segmentation": polygons},)
        views_by_name[file_name] = (view,)
        views.append(view)
        included_rows.append(
            {
                "date": date,
                "file_name": file_name,
                "mask_file_name": mask_file_name,
                "image_sha256": row["image"]["processed_sha256"],
                "mask_sha256": row["mask"]["processed_sha256"],
                "mask_area_fraction": float(binary.mean()),
                "semantic_contours": len(polygons),
            }
        )

    if not views:
        raise ValueError("No usable Flat U-Net masks remain after validation")
    included_digest = hashlib.sha256(
        json.dumps(included_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    metadata = {
        "name": manifest["name"],
        "source_doi": manifest["source_doi"],
        "source_portal": manifest["source_portal"],
        "source_paper_citation": manifest["source_paper_citation"],
        "source_rights_note": manifest["source_rights_note"],
        "annotation_type": "binary semantic maps, not manually separated instance masks",
        "pretraining_representation": "connected contours are weak pseudo-instances; competition annotations provide instance-aware fine-tuning",
        "source_training_observations": manifest["source_training_fits_count"],
        "eligible_training_observations": manifest["eligible_training_observations"],
        "included_observations": len(views),
        "included_semantic_contours": sum(row["semantic_contours"] for row in included_rows),
        "competition_date_overlap": 0,
        "all_competition_dates_excluded": True,
        "selection_manifest_sha256": manifest["selection_manifest_sha256"],
        "included_manifest_sha256": included_digest,
        "competition_ground_truth_used": False,
        "source_test_gd_used": False,
        "inference_uses_external_images": False,
    }
    dataset = CompetitionDataset(
        root=root,
        train_image_dir=root / "images",
        test_image_dir=root / "images",
        annotation_path=manifest_path,
        image_views=tuple(sorted(views, key=lambda view: view.file_name)),
        annotations_by_image_id=annotations_by_id,
        views_by_observation=dict(sorted(views_by_name.items())),
        categories=({"id": 1, "name": "filament"},),
    )
    return dataset, metadata


def load_kso_weak_dataset(
    data_root: str | Path,
    competition: CompetitionDataset,
    split: ObservationSplit,
    *,
    repository_root: str | Path,
    allow_readonly_external_root: bool = False,
) -> tuple[CompetitionDataset, dict[str, Any]]:
    """Load date-disjoint KSO H-alpha images and automatically detected labels.

    The KSO binary products encode a distinct integer-valued label for each
    detected filament. These are weak automated labels, not manual ground
    truth. Only the exact 2023 synoptic image paired with each map is used.
    """
    root = Path(data_root).expanduser().resolve()
    repository_root = Path(repository_root).expanduser().resolve()
    kaggle_input_root = Path("/kaggle/input").resolve()
    inside_workspace = root.is_relative_to(repository_root / "data" / "external")
    inside_readonly_input = allow_readonly_external_root and root.is_relative_to(kaggle_input_root)
    if not inside_workspace and not inside_readonly_input:
        raise ValueError(f"KSO data must stay under data/external/: {root}")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"KSO data manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("KSO data manifest is not complete")
    if manifest.get("competition_split_sha256") != split.sha256:
        raise ValueError("KSO data were prepared against a different frozen split")
    if manifest.get("competition_train_test_dates_excluded") is not True:
        raise ValueError("KSO manifest does not certify competition-date exclusion")
    if manifest.get("competition_ground_truth_used") is not False:
        raise ValueError("KSO manifest must not use competition ground truth")

    competition_names = set(competition.observation_names)
    competition_names.update(
        path.name for path in competition.test_image_dir.iterdir() if path.is_file()
    )
    blocked_dates = {name[:8] for name in competition_names}
    rows = manifest.get("observations", [])
    if not rows or len(rows) != manifest.get("selected_observations"):
        raise ValueError("KSO manifest has no observations or an inconsistent count")
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != manifest.get("selection_manifest_sha256"):
        raise ValueError("KSO selection manifest digest does not match its contents")

    views: list[ImageView] = []
    annotations_by_id: dict[str, tuple[dict[str, Any], ...]] = {}
    views_by_name: dict[str, tuple[ImageView, ...]] = {}
    included_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        date = str(row.get("date", ""))
        stamp = str(row.get("timestamp", ""))
        file_name = str(row.get("file_name", ""))
        if not re.fullmatch(r"2023\d{4}", date) or date in blocked_dates:
            raise ValueError(f"KSO observation has an invalid or overlapping date: {date}")
        if not re.fullmatch(r"2023\d{4}_\d{6}", stamp):
            raise ValueError(f"Unexpected KSO timestamp: {stamp}")
        if file_name != f"{stamp}_KSO.jpg" or Path(file_name).name != file_name:
            raise ValueError(f"Unexpected processed KSO image name: {file_name}")
        image_path = (root / row["processed_image_path"]).resolve()
        source_image_path = (root / row["image_path"]).resolve()
        mask_path = (root / row["mask_path"]).resolve()
        if any(not path.is_relative_to(root) for path in (image_path, source_image_path, mask_path)):
            raise ValueError("KSO manifest contains a path outside its data root")
        processed_bytes = image_path.read_bytes()
        source_image_bytes = source_image_path.read_bytes()
        mask_bytes = mask_path.read_bytes()
        if hashlib.sha256(processed_bytes).hexdigest() != row["processed_image_sha256"]:
            raise ValueError(f"KSO processed image checksum mismatch: {file_name}")
        if hashlib.sha256(source_image_bytes).hexdigest() != row["source_image_sha256"]:
            raise ValueError(f"KSO source image checksum mismatch: {file_name}")
        if hashlib.sha256(mask_bytes).hexdigest() != row["source_mask_sha256"]:
            raise ValueError(f"KSO source mask checksum mismatch: {file_name}")
        image = cv2.imdecode(np.frombuffer(processed_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (2048, 2048):
            raise ValueError(f"Invalid processed KSO image: {file_name}")

        raw_mask = gzip.decompress(mask_bytes)
        if len(raw_mask) < 16:
            raise ValueError(f"KSO mask header is truncated: {file_name}")
        height, width, reserved_a, reserved_b = struct.unpack("<4I", raw_mask[:16])
        if (height, width, reserved_a, reserved_b) != (2048, 2048, 0, 0):
            raise ValueError(f"Unexpected KSO mask header for {file_name}")
        values = np.frombuffer(raw_mask, dtype="<f4", offset=16)
        if values.size != 2048 * 2048:
            raise ValueError(f"Unexpected KSO mask payload size for {file_name}")
        label_map = values.reshape(2048, 2048)
        if not np.isfinite(label_map).all() or np.any(label_map < 0):
            raise ValueError(f"KSO mask contains invalid labels: {file_name}")
        rounded = np.rint(label_map)
        if not np.allclose(label_map, rounded, rtol=0.0, atol=1e-5):
            raise ValueError(f"KSO mask is not integer-valued: {file_name}")
        label_ids = [int(value) for value in np.unique(rounded) if value > 0]
        if label_ids != [int(value) for value in row["label_ids"]]:
            raise ValueError(f"KSO label IDs differ from the manifest: {file_name}")

        annotations = []
        for label_id in label_ids:
            instance = (rounded == label_id).astype(np.uint8)
            contours, _ = cv2.findContours(
                instance, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            polygons = []
            for contour in contours:
                if cv2.contourArea(contour) < 6.0:
                    continue
                polygon = cv2.approxPolyDP(contour, 1.0, True).reshape(-1, 2)
                if len(polygon) >= 3:
                    polygons.append(polygon.astype(float).reshape(-1).tolist())
            if polygons:
                annotations.append({"segmentation": polygons})
        if len(annotations) != len(label_ids):
            raise ValueError(f"Some KSO filament IDs produced no trainable polygon: {file_name}")
        if not annotations:
            raise ValueError(f"KSO image has no filament annotations: {file_name}")

        view_id = f"kso-{date}-{stamp[-6:]}"
        view = ImageView(
            image_id=view_id,
            file_name=file_name,
            width=2048,
            height=2048,
            year=2023,
            site="KSO",
        )
        views.append(view)
        annotations_by_id[view_id] = tuple(annotations)
        views_by_name[file_name] = (view,)
        included_rows.append(
            {
                "date": date,
                "file_name": file_name,
                "source_image_sha256": row["source_image_sha256"],
                "source_mask_sha256": row["source_mask_sha256"],
                "label_count": len(label_ids),
                "foreground_pixels": int(np.count_nonzero(rounded)),
            }
        )

    included_digest = hashlib.sha256(
        json.dumps(included_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    metadata = {
        "name": manifest["name"],
        "source_image_index": KSO_IMAGE_INDEX,
        "source_mask_root": KSO_MASK_ROOT,
        "license": KSO_LICENSE,
        "attribution": KSO_ATTRIBUTION,
        "source_notification_requested": True,
        "annotation_type": "automatically detected instance label maps; weak supervision",
        "included_observations": len(views),
        "included_filament_instances": sum(len(items) for items in annotations_by_id.values()),
        "competition_train_test_dates_excluded": True,
        "blocked_competition_dates": len(blocked_dates),
        "competition_split_sha256": split.sha256,
        "competition_ground_truth_used": False,
        "public_MAGFiLO_test_annotations_used": False,
        "inference_uses_external_images": False,
        "selection_manifest_sha256": manifest["selection_manifest_sha256"],
        "included_manifest_sha256": included_digest,
    }
    dataset = CompetitionDataset(
        root=root,
        train_image_dir=root / "images",
        test_image_dir=root / "images",
        annotation_path=manifest_path,
        image_views=tuple(sorted(views, key=lambda view: view.file_name)),
        annotations_by_image_id=annotations_by_id,
        views_by_observation=dict(sorted(views_by_name.items())),
        categories=({"id": 1, "name": "filament"},),
    )
    return dataset, metadata
