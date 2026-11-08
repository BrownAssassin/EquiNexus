# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "data" / "external" / "flat-unet-101545" / "processed-256"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "external" / "flat-unet-101545" / "kaggle-256-q90"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def convert(source_root: Path, output_root: Path, repository_root: Path = REPO_ROOT) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if not source_root.is_relative_to(repository_root / "data" / "external"):
        raise ValueError(f"Flat U-Net source must stay under data/external/: {source_root}")
    if not output_root.is_relative_to(repository_root / "data" / "external"):
        raise ValueError(f"Kaggle input must stay under data/external/: {output_root}")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_root}")

    source_manifest = json.loads((source_root / "manifest.json").read_text(encoding="utf-8"))
    if (
        source_manifest.get("status") != "complete"
        or source_manifest.get("competition_date_overlap") != 0
        or source_manifest.get("prepared_observations") != 256
    ):
        raise ValueError("Source Flat U-Net manifest failed its completion, overlap, or count checks")
    output_root.mkdir(parents=True, exist_ok=True)
    image_root, mask_root = output_root / "images", output_root / "masks"
    image_root.mkdir(parents=True, exist_ok=True)
    mask_root.mkdir(parents=True, exist_ok=True)

    rows = []
    total_image_bytes = 0
    total_mask_bytes = 0
    image_mae = []
    for source_row in source_manifest["observations"]:
        source_name = str(source_row["file_name"])
        image_source = source_root / "images" / source_name
        mask_source = source_root / "masks" / str(source_row["mask"]["file_name"])
        source_image_bytes = image_source.read_bytes()
        source_mask_bytes = mask_source.read_bytes()
        if _sha256(source_image_bytes) != source_row["image"]["processed_sha256"]:
            raise ValueError(f"Source PNG hash mismatch: {source_name}")
        if _sha256(source_mask_bytes) != source_row["mask"]["processed_sha256"]:
            raise ValueError(f"Source mask hash mismatch: {source_name}")
        with Image.open(io.BytesIO(source_image_bytes)) as image:
            array = np.asarray(image.convert("L"))
        if array.shape != (2048, 2048):
            raise ValueError(f"Expected a 2048x2048 source image: {source_name}")
        encoded = io.BytesIO()
        Image.fromarray(array).save(
            encoded, format="JPEG", quality=90, optimize=True, progressive=True
        )
        jpeg_bytes = encoded.getvalue()
        with Image.open(io.BytesIO(jpeg_bytes)) as image:
            decoded = np.asarray(image.convert("L"))
        mae = float(np.mean(np.abs(decoded.astype(np.float32) - array.astype(np.float32))))
        image_mae.append(mae)

        stem = Path(source_name).stem
        image_name = f"{stem}.jpg"
        mask_name = f"{stem}.png"
        image_path = image_root / image_name
        mask_path = mask_root / mask_name
        image_path.write_bytes(jpeg_bytes)
        mask_path.write_bytes(source_mask_bytes)
        total_image_bytes += len(jpeg_bytes)
        total_mask_bytes += len(source_mask_bytes)

        row = dict(source_row)
        row["file_name"] = image_name
        row["image"] = dict(source_row["image"])
        row["image"].update(
            {
                "file_name": image_name,
                "source_processed_png_sha256": source_row["image"]["processed_sha256"],
                "processed_sha256": _sha256(jpeg_bytes),
                "processed_bytes": len(jpeg_bytes),
                "encoding": "JPEG",
                "quality": 90,
                "mean_absolute_error_vs_normalized_png": mae,
            }
        )
        row["mask"] = dict(source_row["mask"])
        row["mask"]["file_name"] = mask_name
        rows.append(row)

    selection_payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest = dict(source_manifest)
    manifest.update(
        {
            "observations": rows,
            "selection_manifest_sha256": hashlib.sha256(selection_payload).hexdigest(),
            "image_encoding": "8-bit grayscale JPEG quality 90; masks are original binary PNGs",
            "image_compression_mean_absolute_error": float(np.mean(image_mae)),
            "image_compression_max_mean_absolute_error": float(np.max(image_mae)),
            "prepared_image_bytes": total_image_bytes,
            "prepared_mask_bytes": total_mask_bytes,
            "prepared_data_bytes": total_image_bytes + total_mask_bytes,
            "derivative_of_selection_manifest_sha256": source_manifest["selection_manifest_sha256"],
        }
    )
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "SOURCE_NOTES.md").write_text(
        (source_root / "SOURCE_NOTES.md").read_text(encoding="utf-8")
        + "\nKaggle upload copy: normalized grayscale images were encoded as JPEG quality 90 "
        "to reduce transfer size. The mean absolute pixel error against the lossless PNG "
        f"images is {float(np.mean(image_mae)):.3f} intensity levels. The original PNG "
        "subset and FITS archive remain available locally under data/external/.\n",
        encoding="utf-8",
    )
    return {
        "output_root": str(output_root),
        "observations": len(rows),
        "selection_manifest_sha256": manifest["selection_manifest_sha256"],
        "derived_from_selection_manifest_sha256": source_manifest["selection_manifest_sha256"],
        "image_bytes": total_image_bytes,
        "mask_bytes": total_mask_bytes,
        "total_bytes": total_image_bytes + total_mask_bytes,
        "mean_absolute_error": float(np.mean(image_mae)),
        "max_image_mean_absolute_error": float(np.max(image_mae)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a compact, high-quality JPEG copy for private Kaggle upload."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    print(json.dumps(convert(args.source_root, args.output_root, args.repo_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
