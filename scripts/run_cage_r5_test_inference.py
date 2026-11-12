# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Run the public CAGE r5 U-Net weights on the supplied competition test images.

The upstream inference implementation and checkpoints remain CC BY-NC 4.0.
This runner loads weights safely and does not read training annotations or the
frozen holdout. Its output is an exploratory test-only submission candidate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import cv2
from PIL import Image
from pycocotools import mask as mask_utils

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from filament_baseline.runtime import hardware_info, package_versions  # noqa: E402

DATASET_URL = "https://www.kaggle.com/datasets/jamesmills1975/cage-filament-r5-weights"
EXAMPLE_NOTEBOOK_URL = "https://www.kaggle.com/code/jamesmills1975/cage-filament-r5-weights-example"
SPLIT_SHA256 = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"
IMAGE_SIZE = (2048, 2048)
EXPECTED_HASHES = {
    "infer_r5.py": "590010a8b4ebd1f40f27c55b12474ed5b50bc9a80fdb867cfda033270c923f4b",
    "unet1024_ch0.pt": "3bc2bc63873e718f053842af79047168eafbd42d8b86cd9789d2ba4622c43fe4",
    "unet1024_ch1.pt": "36158922f3caace29d4d190100f2b8a736298f282ac141aff671eb93b10e72a2",
    "LICENSE.md": "cf48efe33a77c15d0ccebfc9fcf2a785422da54fda6c56f8c7b9d963e364001e",
    "MODEL_CARD.md": "a8d68a48359af55077256a6dfa6b0fa2850477b5d9a60c6e48e17056228a2174",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def require_within(path: Path, root: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"{description} must stay under {root}: {resolved}")
    return resolved


def verify_package(weights_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, expected in EXPECTED_HASHES.items():
        path = weights_root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {name}: expected {expected}, got {actual}")
        hashes[name] = actual

    sums_path = weights_root / "SHA256SUMS"
    if not sums_path.is_file():
        raise FileNotFoundError(sums_path)
    listed = {}
    for line in sums_path.read_text(encoding="ascii").splitlines():
        fields = line.split()
        if len(fields) == 2:
            listed[fields[1].lstrip("*")] = fields[0].lower()
    for name in ("infer_r5.py", "unet1024_ch0.pt", "unet1024_ch1.pt"):
        if listed.get(name) != hashes[name]:
            raise ValueError(f"Upstream SHA256SUMS does not match {name}")
    return hashes


def load_upstream_inference(inference_path: Path):
    spec = importlib.util.spec_from_file_location("cage_r5_infer", inference_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load upstream inference module: {inference_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model(weights_path: Path, device: torch.device) -> torch.nn.Module:
    import segmentation_models_pytorch as smp

    model = smp.Unet(
        encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=1
    )
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device)
    return model.eval()


def encode_mask(binary_mask: np.ndarray) -> str:
    encoded = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    return encoded["counts"].decode("ascii")


def save_overlay(image: np.ndarray, labels: np.ndarray, destination: Path) -> None:
    ids = np.unique(labels)
    ids = ids[ids > 0]
    colors = np.zeros((int(labels.max()) + 1, 3), dtype=np.uint8)
    for instance_id in ids:
        colors[instance_id] = (
            (int(instance_id) * 71 + 47) % 256,
            (int(instance_id) * 131 + 89) % 256,
            (int(instance_id) * 197 + 151) % 256,
        )
    base = np.repeat(image[:, :, None], 3, axis=2).astype(np.float32)
    color_map = colors[labels].astype(np.float32)
    foreground = labels > 0
    base[foreground] = base[foreground] * 0.58 + color_map[foreground] * 0.42
    Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB").save(destination)


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026",
    )
    parser.add_argument(
        "--weights-root",
        type=Path,
        default=REPO_ROOT / "data" / "reference" / "kaggle-datasets" / "cage-filament-r5-weights",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "submissions" / "cage-r5-test-20261114",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-images", type=int, help="limit inference for a local GPU smoke run")
    args = parser.parse_args()

    data_root = require_within(args.data_root, REPO_ROOT / "data", "Competition data")
    weights_root = require_within(
        args.weights_root, REPO_ROOT / "data" / "reference", "Public model package"
    )
    output_root = require_within(args.output_dir, REPO_ROOT / "artifacts", "Run output")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_root}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats(device)

    hashes = verify_package(weights_root)
    inference = load_upstream_inference(weights_root / "infer_r5.py")
    import segmentation_models_pytorch as smp

    mask_net = load_model(weights_root / "unet1024_ch0.pt", device)
    hull_net = load_model(weights_root / "unet1024_ch1.pt", device)
    hardware = hardware_info()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    test_dir = data_root / "test" / "test_images"
    test_paths = sorted(test_dir.glob("*.jpeg"))
    if len(test_paths) != 180:
        raise ValueError(f"Expected 180 supplied competition test images, found {len(test_paths)}")
    if len({path.stem for path in test_paths}) != len(test_paths):
        raise ValueError("Test image filenames are not unique")
    if args.max_images is not None:
        if args.max_images < 1:
            raise ValueError("--max-images must be positive")
        test_paths = test_paths[: args.max_images]

    output_root.mkdir(parents=True, exist_ok=True)
    overlay_root = output_root / "overlays"
    overlay_root.mkdir()
    overlay_indices = set(
        np.linspace(0, len(test_paths) - 1, min(6, len(test_paths)), dtype=int).tolist()
    )
    rows: list[tuple[str, str]] = []
    image_results: list[dict[str, Any]] = []
    started = time.perf_counter()

    for index, path in enumerate(test_paths):
        with Image.open(path) as source_image:
            if source_image.size != IMAGE_SIZE:
                raise ValueError(f"Expected {IMAGE_SIZE} image at {path}, got {source_image.size}")
            gray = np.asarray(source_image.convert("L"), dtype=np.uint8)
        labels, instance_count = inference.predict(gray, mask_net, hull_net, dev=str(device))
        if labels.shape != IMAGE_SIZE or labels.dtype != np.int32:
            raise ValueError(f"Unexpected instance map from CAGE r5 for {path.name}")
        ids = np.unique(labels)
        ids = ids[ids > 0]
        if not np.array_equal(ids, np.arange(1, instance_count + 1, dtype=ids.dtype)):
            raise ValueError(f"Non-contiguous instance IDs in {path.name}")

        for instance_index, instance_id in enumerate(ids, start=1):
            rows.append((f"{path.stem}_{instance_index}", encode_mask(labels == instance_id)))
        image_results.append(
            {"image": path.name, "instances": int(instance_count), "has_prediction": bool(instance_count)}
        )
        if index in overlay_indices:
            save_overlay(gray, labels, overlay_root / f"{path.stem}.png")
        if (index + 1) % 30 == 0 or index + 1 == len(test_paths):
            print(f"Processed {index + 1}/{len(test_paths)} supplied test images", flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    submission_path = output_root / "submission.csv"
    with submission_path.open("w", encoding="ascii", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("filament_id", "segmentation_rle"))
        writer.writerows(rows)

    elapsed = time.perf_counter() - started
    metadata = {
        "candidate_role": "test-only exploratory submission; no holdout evaluation",
        "source_dataset": DATASET_URL,
        "source_example_notebook": EXAMPLE_NOTEBOOK_URL,
        "model_author": "James Mills",
        "license": "CC-BY-NC-4.0",
        "attribution": "CAGE Filament r5; MAGFiLO (Ahmadzadeh et al., Scientific Data 2024, doi:10.1038/s41597-024-03876-y); NSO/GONG",
        "training_provenance": "Model card reports fine-tuning on the competition training split with ImageNet-pretrained ResNet-34 encoders and no other labeled data; exact observation split is not published. Upstream 100-image validation overlaps training through multi-annotator duplicates.",
        "holdout_evaluation": "not performed; external model training may overlap the frozen validation observations",
        "input_scope": "supplied competition test H-alpha images only; no competition labels loaded",
        "competition_test_images": len(test_paths),
        "test_images_with_predictions": sum(int(result["has_prediction"]) for result in image_results),
        "predicted_instances": len(rows),
        "image_results": image_results,
        "split_sha256": SPLIT_SHA256,
        "source_commit": git_value("rev-parse", "HEAD"),
        "working_tree": git_value("status", "--short"),
        "script_sha256": sha256(Path(__file__).resolve()),
        "upstream_files_sha256": hashes,
        "config": {"input_size": 1024, "mask_threshold": 0.5, "hull_threshold": 0.3,
                   "closing_fallback_px": 51, "minimum_instance_area_2048": 250,
                   "tta": "horizontal flip average", "normalization": "(pixel/255 - 0.5)/0.25"},
        "torch": torch.__version__,
        "segmentation_models_pytorch": smp.__version__,
        "opencv": cv2.__version__,
        "device": str(device),
        "hardware": hardware,
        "package_versions": package_versions(),
        "runtime_seconds": round(elapsed, 3),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "submission_path": str(submission_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "submission_sha256": sha256(submission_path),
        "overlay_paths": [str(path.relative_to(REPO_ROOT)).replace("\\", "/") for path in sorted(overlay_root.glob("*.png"))],
    }
    metadata_path = output_root / "run-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: metadata[key] for key in (
        "candidate_role", "competition_test_images", "test_images_with_predictions",
        "predicted_instances", "runtime_seconds", "peak_gpu_memory_bytes", "submission_sha256"
    )}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
