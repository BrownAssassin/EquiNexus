# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Run a provenance-checked public U-Net ensemble on the supplied test images.

The model weights remain under their upstream CC BY-NC-SA 4.0 license. This
runner is an independent implementation of the published inference procedure;
it does not load competition labels or evaluate the frozen holdout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from scipy import ndimage as ndi

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from filament_baseline.runtime import hardware_info, package_versions  # noqa: E402


SOURCE_DATASET = "https://www.kaggle.com/datasets/dariushafshar/solar-filament-unet-weights"
SOURCE_NOTEBOOK = "https://www.kaggle.com/code/dariushafshar/solar-filament-u-net-oof-0-45-vs-public-0-39"
EXPECTED_SPLIT_SHA256 = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"
IMAGE_SIZE = 2048
DISK_RADIUS = 950
KEEP_RULE = {"t_lo": 0.35, "t_hi": 0.99, "min_area": 200, "min_meanp": 0.85}
DIHEDRAL_VIEWS = ((0, False), (0, True), (2, False), (2, True),
                  (1, False), (1, True), (3, False), (3, True))


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


def read_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"Expected 2048x2048 image at {path}, got {image.size}")
        return np.asarray(image.convert("L"), dtype=np.uint8)


def disk_statistics(image: np.ndarray, disk: np.ndarray) -> tuple[float, float]:
    pixels = image[disk].astype(np.float32) / 255.0
    return float(pixels.mean()), max(float(pixels.std()), 1e-3)


def transform(tensor: torch.Tensor, rotation: int, flip: bool) -> torch.Tensor:
    if flip:
        tensor = torch.flip(tensor, dims=(3,))
    return torch.rot90(tensor, rotation, dims=(2, 3)) if rotation else tensor


def inverse_transform(tensor: torch.Tensor, rotation: int, flip: bool) -> torch.Tensor:
    if rotation:
        tensor = torch.rot90(tensor, -rotation, dims=(2, 3))
    return torch.flip(tensor, dims=(3,)) if flip else tensor


@torch.inference_mode()
def predict_map(
    model: torch.nn.Module,
    image: np.ndarray,
    mean: float,
    std: float,
    device: torch.device,
    *,
    tta_views: int,
    views_per_pass: int,
) -> np.ndarray:
    pixels = torch.from_numpy(image.copy()).to(device=device, dtype=torch.float32) / 255.0
    normalized = ((pixels - mean) / std)[None, None]
    accumulated = torch.zeros((IMAGE_SIZE, IMAGE_SIZE), device=device, dtype=torch.float32)
    selected_views = DIHEDRAL_VIEWS[:tta_views]
    use_amp = device.type == "cuda"

    for start in range(0, len(selected_views), views_per_pass):
        view_batch = selected_views[start : start + views_per_pass]
        batch = torch.cat(
            [transform(normalized, rotation, flip) for rotation, flip in view_batch], dim=0
        )
        if use_amp:
            batch = batch.contiguous(memory_format=torch.channels_last)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(batch)
        probabilities = logits.float().sigmoid()
        for index, (rotation, flip) in enumerate(view_batch):
            accumulated += inverse_transform(
                probabilities[index : index + 1], rotation, flip
            )[0, 0]

    quantized = (accumulated / len(selected_views) * 255.0).round().clamp_(0, 255)
    return quantized.to(dtype=torch.uint8, device="cpu").numpy()


def load_model(model_entry: dict[str, Any], weights_root: Path, device: torch.device):
    import segmentation_models_pytorch as smp

    checkpoint = weights_root / model_entry["file"]
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    expected_size = int(model_entry["bytes"])
    expected_hash = str(model_entry["sha256"])
    if checkpoint.stat().st_size != expected_size:
        raise ValueError(f"Checkpoint byte-count mismatch: {checkpoint.name}")
    if sha256(checkpoint) != expected_hash:
        raise ValueError(f"Checkpoint SHA-256 mismatch: {checkpoint.name}")

    model = smp.Unet(
        encoder_name=str(model_entry["encoder"]),
        encoder_weights=None,
        in_channels=1,
        classes=1,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device)
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)
    return model.eval()


def prediction_rows(probability_map: np.ndarray, stem: str) -> list[tuple[str, str]]:
    low = int(round(KEEP_RULE["t_lo"] * 255.0))
    high = int(round(KEEP_RULE["t_hi"] * 255.0))
    labels, component_count = ndi.label(
        probability_map > low, structure=np.ones((3, 3), dtype=bool)
    )
    if component_count == 0:
        return []

    flat_labels = labels.ravel()
    positions = np.flatnonzero(flat_labels)
    component_ids = flat_labels[positions]
    values = probability_map.ravel()[positions]
    areas = np.bincount(component_ids, minlength=component_count + 1)[1:]
    probability_sums = np.bincount(
        component_ids, weights=values, minlength=component_count + 1
    )[1:]
    peaks = np.zeros(component_count + 1, dtype=np.uint8)
    np.maximum.at(peaks, component_ids, values)
    means = probability_sums / np.maximum(areas, 1) / 255.0

    accepted = np.flatnonzero(
        (peaks[1:] > high)
        & (areas >= KEEP_RULE["min_area"])
        & (means >= KEEP_RULE["min_meanp"])
    ) + 1

    objects = ndi.find_objects(labels)
    output: list[tuple[str, str]] = []
    for sequence, component_id in enumerate(accepted, start=1):
        region = objects[int(component_id) - 1]
        component_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        component_mask[region] = labels[region] == component_id
        encoded = mask_utils.encode(np.asfortranarray(component_mask))["counts"].decode("ascii")
        output.append((f"{stem}_{sequence}", encoded))
    return output


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
        default=REPO_ROOT / "data" / "reference" / "dariushafshar-unet-weights",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "artifacts" / "submissions" / "public-unet-ensemble-20261112",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tta-views", type=int, choices=(1, 2, 4, 8), default=8)
    parser.add_argument("--views-per-pass", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--model-family", choices=("all", "resnet34", "efficientnetv2_s"), default="all")
    parser.add_argument("--max-images", type=int, default=None, help="limit inference for a local GPU smoke run")
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    weights_root = args.weights_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not data_root.is_relative_to(REPO_ROOT / "data"):
        raise ValueError(f"Competition input must stay inside the workspace data directory: {data_root}")
    if not weights_root.is_relative_to(REPO_ROOT / "data" / "reference"):
        raise ValueError(f"Public model data must stay inside data/reference: {weights_root}")
    if not output_root.is_relative_to(REPO_ROOT / "artifacts"):
        raise ValueError(f"Run outputs must stay under artifacts/: {output_root}")
    if args.tta_views not in (1, 2, 4, 8) or args.views_per_pass > args.tta_views:
        raise ValueError("Invalid TTA configuration")

    split_path = REPO_ROOT / "data" / "splits" / "observation_split_seed42.csv"
    split_sha = sha256(split_path)
    if split_sha != EXPECTED_SPLIT_SHA256:
        raise ValueError(f"Frozen split digest changed: {split_sha}")

    receipt_path = weights_root / "solar_unet_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
    models = list(receipt["ensemble"]["models"])
    if args.model_family != "all":
        models = [entry for entry in models if entry["family"] == args.model_family]
    if not models:
        raise ValueError(f"No checkpoints found for model family {args.model_family}")

    test_dir = data_root / "test" / "test_images"
    test_paths = sorted(test_dir.glob("*.jpeg"))
    if len(test_paths) != 180:
        raise ValueError(f"Expected all 180 supplied test images, found {len(test_paths)}")
    if args.max_images is not None:
        if args.max_images < 1:
            raise ValueError("--max-images must be positive")
        test_paths = test_paths[: args.max_images]

    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cudnn.benchmark = True
    hardware = hardware_info()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    yy, xx = np.ogrid[:IMAGE_SIZE, :IMAGE_SIZE]
    disk = ((yy - (IMAGE_SIZE - 1) / 2.0) ** 2 + (xx - (IMAGE_SIZE - 1) / 2.0) ** 2) <= DISK_RADIUS**2
    images = [read_gray(path) for path in test_paths]
    statistics = [disk_statistics(image, disk) for image in images]
    probability_sum = np.zeros((len(images), IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint16)
    model_records: list[dict[str, Any]] = []
    started = time.perf_counter()

    for model_index, entry in enumerate(models, start=1):
        model_started = time.perf_counter()
        model = load_model(entry, weights_root, device)
        for image_index, image in enumerate(images):
            probability = predict_map(
                model,
                image,
                *statistics[image_index],
                device,
                tta_views=args.tta_views,
                views_per_pass=args.views_per_pass,
            )
            probability_sum[image_index] += probability.astype(np.uint16)
            if (image_index + 1) % 30 == 0 or image_index + 1 == len(images):
                print(
                    f"{entry['file']}: {image_index + 1}/{len(images)} images",
                    flush=True,
                )
        model_records.append(
            {
                "file": entry["file"],
                "sha256": entry["sha256"],
                "fold": entry["fold"],
                "family": entry["family"],
                "seconds": round(time.perf_counter() - model_started, 3),
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"Completed {model_index}/{len(models)} models", flush=True)

    probability_maps = np.empty((len(images), IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    for image_index in range(len(images)):
        averaged = np.rint(probability_sum[image_index].astype(np.float32) / len(models))
        probability_maps[image_index] = np.clip(averaged, 0, 255).astype(np.uint8)
    del probability_sum

    rows: list[tuple[str, str]] = []
    images_with_predictions = 0
    for path, probability in zip(test_paths, probability_maps, strict=True):
        image_rows = prediction_rows(probability, path.stem)
        rows.extend(image_rows)
        images_with_predictions += int(bool(image_rows))

    submission_path = output_root / "submission.csv"
    with submission_path.open("w", encoding="ascii", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("filament_id", "segmentation_rle"))
        writer.writerows(rows)

    elapsed = time.perf_counter() - started
    try:
        smp_version = importlib.metadata.version("segmentation-models-pytorch")
    except importlib.metadata.PackageNotFoundError:
        smp_version = None
    try:
        timm_version = importlib.metadata.version("timm")
    except importlib.metadata.PackageNotFoundError:
        timm_version = None
    upstream_run = {"all": "v6", "resnet34": "v3", "efficientnetv2_s": "v5"}[
        args.model_family
    ]
    metadata = {
        "source_dataset": SOURCE_DATASET,
        "source_notebook": SOURCE_NOTEBOOK,
        "model_creator": "Dariush Afshar",
        "source_dataset_license": "CC-BY-NC-SA-4.0",
        "attribution_and_license_note": "Retain upstream attribution and comply with CC BY-NC-SA 4.0 terms for redistributed model material.",
        "upstream_receipt_sha256": sha256(receipt_path),
        "training_provenance": "published weights trained on competition-provided training annotations only; no public MAGFiLO test labels per upstream card",
        "upstream_run": upstream_run,
        "upstream_oof_reported": receipt["runs"][upstream_run]["oof"],
        "upstream_public_score": receipt["runs"][upstream_run]["public_lb"],
        "holdout_evaluation": "not performed: published folds trained on portions of the frozen holdout",
        "model_family": args.model_family,
        "models_used": model_records,
        "tta_views": args.tta_views,
        "views_per_pass": args.views_per_pass,
        "keep_rule": KEEP_RULE,
        "input_normalization": f"(pixel/255 - disk mean) / disk std; radius {DISK_RADIUS}",
        "competition_test_input_count": len(test_paths),
        "test_images_with_predictions": images_with_predictions,
        "predicted_instances": len(rows),
        "submission_sha256": sha256(submission_path),
        "submission_path": str(submission_path.relative_to(REPO_ROOT)),
        "split_sha256": split_sha,
        "source_commit": git_value("rev-parse", "HEAD"),
        "working_tree": git_value("status", "--short"),
        "script_sha256": sha256(Path(__file__).resolve()),
        "torch": torch.__version__,
        "segmentation_models_pytorch": smp_version,
        "timm": timm_version,
        "hardware": hardware,
        "package_versions": package_versions(),
        "runtime_seconds": round(elapsed, 3),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "full_180_image_submission": len(test_paths) == 180,
    }
    metadata_path = output_root / "run-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: metadata[k] for k in (
        "predicted_instances", "test_images_with_predictions", "runtime_seconds",
        "peak_gpu_memory_bytes", "submission_sha256"
    )}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
