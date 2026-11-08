# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Suppress repeated YOLO masks produced by dihedral test-time augmentation."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from pycocotools import mask as mask_utils

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SHAPE = [2048, 2048]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rle(detection: dict[str, Any]) -> dict[str, Any]:
    counts = detection["counts"]
    if isinstance(counts, str):
        counts = counts.encode("ascii")
    return {"size": IMAGE_SHAPE, "counts": counts}


def _deduplicate(
    detections: list[dict[str, Any]],
    threshold: float,
    *,
    merge_duplicates: bool = False,
) -> list[dict[str, Any]]:
    ordered = sorted(detections, key=lambda item: -float(item["score"]))
    clusters: list[list[dict[str, Any]]] = []
    cluster_leaders: list[dict[str, Any]] = []
    for candidate in ordered:
        rle = _rle(candidate)
        if cluster_leaders:
            overlaps = mask_utils.iou(
                [rle], cluster_leaders, [0] * len(cluster_leaders)
            ).reshape(-1)
            if overlaps.size and float(overlaps.max()) > threshold:
                if merge_duplicates:
                    cluster_index = int(overlaps.argmax())
                    clusters[cluster_index].append(candidate)
                continue
        clusters.append([candidate])
        cluster_leaders.append(rle)

    if not merge_duplicates:
        return [cluster[0] for cluster in clusters]

    merged = []
    for cluster in clusters:
        if len(cluster) == 1:
            merged.append(cluster[0])
            continue
        union = mask_utils.merge(
            [_rle(candidate) for candidate in cluster], intersect=False
        )
        counts = union["counts"]
        if isinstance(counts, bytes):
            counts = counts.decode("ascii")
        merged.append(
            {
                "score": float(cluster[0]["score"]),
                "counts": counts,
                "merged_members": len(cluster),
            }
        )
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply confidence-ordered mask NMS to duplicate TTA detections."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mask-iou-threshold", type=float, default=0.7)
    parser.add_argument(
        "--merge-duplicates",
        action="store_true",
        help="union duplicate masks into their highest-confidence cluster representative",
    )
    args = parser.parse_args()

    predictions_path = args.predictions.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not predictions_path.is_relative_to(REPO_ROOT / "artifacts"):
        raise ValueError("TTA predictions must be under artifacts/")
    if not output_dir.is_relative_to(REPO_ROOT / "artifacts"):
        raise ValueError("Deduplicated predictions must be written under artifacts/")
    if not 0.0 <= args.mask_iou_threshold <= 1.0:
        raise ValueError("mask-iou-threshold must be between 0 and 1")
    parent_metadata_path = predictions_path.parent / "run-metadata.json"
    if not parent_metadata_path.is_file():
        raise FileNotFoundError(f"Missing TTA run metadata: {parent_metadata_path}")

    source = json.loads(predictions_path.read_text(encoding="utf-8"))
    parent_metadata = json.loads(parent_metadata_path.read_text(encoding="utf-8"))
    partition = parent_metadata.get("partition", "oof")
    started = time.perf_counter()
    output: dict[str, list[dict[str, Any]]] = {}
    raw_count = kept_count = 0
    for index, (stem, detections) in enumerate(sorted(source.items()), start=1):
        raw_count += len(detections)
        retained = _deduplicate(
            detections,
            args.mask_iou_threshold,
            merge_duplicates=args.merge_duplicates,
        )
        output[stem] = retained
        kept_count += len(retained)
        if index % 50 == 0 or index == len(source):
            print(f"Deduplicated {index}/{len(source)} observations", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = (
        "test_detections_deduplicated.json"
        if partition == "competition_test_inference"
        else "oof_detections.json"
    )
    output_path = output_dir / output_name
    output_path.write_text(json.dumps(output, separators=(",", ":")), encoding="utf-8")
    metadata = {
        "status": "complete",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "partition": partition,
        "source_predictions": str(predictions_path.relative_to(REPO_ROOT)),
        "source_predictions_sha256": _sha256(predictions_path),
        "source_run_metadata_sha256": _sha256(parent_metadata_path),
        "oof_manifest_sha256": parent_metadata["oof_manifest_sha256"],
        "frozen_holdout_sha256": parent_metadata["frozen_holdout_sha256"],
        "observation_count": len(source),
        "oof_observations": len(source) if partition == "oof" else None,
        "mask_iou_suppression_threshold": args.mask_iou_threshold,
        "merge_duplicates": args.merge_duplicates,
        "merge_method": "union" if args.merge_duplicates else "keep_highest_confidence",
        "raw_detection_count": raw_count,
        "retained_detection_count": kept_count,
        "suppressed_detection_count": raw_count - kept_count,
        "prediction_sha256": _sha256(output_path),
        "predictions_file": output_path.name,
        "deduplication_seconds": round(time.perf_counter() - started, 3),
    }
    (output_dir / "run-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
