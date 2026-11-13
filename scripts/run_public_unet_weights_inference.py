# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Run the public, training-only Dariushafshar U-Net ensemble on test images.

This is test inference only. The downloaded weights are CC BY-NC-SA 4.0 and
were trained on the competition training annotations; their fold models have
seen observations from the frozen holdout in other folds, so this runner must
not be used to report a frozen-holdout validation score.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from workspace_runtime import configure_workspace_runtime

configure_workspace_runtime(REPO_ROOT)

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import timm
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from scipy import ndimage as ndi

from filament_baseline.data import find_dataset_root
from filament_baseline.runtime import hardware_info, package_versions


DATASET_URL = "https://www.kaggle.com/datasets/dariushafshar/solar-filament-unet-weights"
DATASET_LICENSE = "CC-BY-NC-SA-4.0"
ATTRIBUTION = "Dariushafshar, Solar Filament U-Net Weights (PQ 0.45 OOF), Kaggle"
FULL_SIZE = 2048
DISK_RADIUS = 950
TTA_TRANSFORMS = (
    (0, False), (0, True), (2, False), (2, True),
    (1, False), (1, True), (3, False), (3, True),
)
BOX_KEEP_RULE = {"t_lo": 0.35, "t_hi": 0.99, "min_area": 200, "min_meanp": 0.85}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomically(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_commit() -> str | None:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def read_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.size != (FULL_SIZE, FULL_SIZE):
            raise ValueError(f"Expected 2048x2048 test image, got {image.size}: {path}")
        return np.asarray(image.convert("L"), dtype=np.uint8)


def disk_stats(image: np.ndarray, disk: np.ndarray) -> tuple[float, float]:
    values = image[disk].astype(np.float32) / 255.0
    return float(values.mean()), max(float(values.std()), 1e-3)


def transform(tensor: torch.Tensor, quarter_turns: int, flip: bool) -> torch.Tensor:
    if flip:
        tensor = torch.flip(tensor, dims=[3])
    return torch.rot90(tensor, quarter_turns, dims=[2, 3]) if quarter_turns else tensor


def inverse_transform(tensor: torch.Tensor, quarter_turns: int, flip: bool) -> torch.Tensor:
    if quarter_turns:
        tensor = torch.rot90(tensor, -quarter_turns, dims=[2, 3])
    return torch.flip(tensor, dims=[3]) if flip else tensor


@torch.inference_mode()
def predict_probability_u8(
    model: torch.nn.Module,
    image_u8: np.ndarray,
    mean: float,
    standard_deviation: float,
    device: torch.device,
    *,
    tta_views: int,
    views_per_pass: int,
) -> np.ndarray:
    image = torch.from_numpy(np.array(image_u8, copy=True)).to(device=device, dtype=torch.float32)
    image.div_(255.0)
    image = ((image - mean) / standard_deviation)[None, None]
    transforms = TTA_TRANSFORMS[:tta_views]
    accumulator = torch.zeros((FULL_SIZE, FULL_SIZE), dtype=torch.float32, device=device)
    use_amp = device.type == "cuda"

    for start in range(0, len(transforms), views_per_pass):
        group = transforms[start : start + views_per_pass]
        inputs = torch.cat([transform(image, k, flip) for k, flip in group], dim=0)
        if use_amp:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(inputs)
        probabilities = torch.sigmoid(logits.float())
        for index, (k, flip) in enumerate(group):
            accumulator += inverse_transform(probabilities[index : index + 1], k, flip)[0, 0]

    accumulator /= len(transforms)
    return accumulator.mul_(255).round_().clamp_(0, 255).to(torch.uint8).cpu().numpy()


def load_models(receipt: dict[str, Any], model_set: str) -> list[dict[str, Any]]:
    models = receipt["ensemble"]["models"]
    if model_set == "resnet34":
        models = [model for model in models if "resnet34" in model["encoder"].lower()]
    elif model_set == "efficientnetv2_s":
        models = [model for model in models if "efficientnetv2_s" in model["encoder"].lower()]
    if not models:
        raise ValueError(f"No weight files selected for model set {model_set!r}")
    return models


def load_model(model_info: dict[str, Any], weights_root: Path, device: torch.device) -> torch.nn.Module:
    weight_path = weights_root / model_info["file"]
    if not weight_path.is_file():
        raise FileNotFoundError(weight_path)
    observed_hash = sha256(weight_path)
    if observed_hash != model_info["sha256"]:
        raise ValueError(f"SHA-256 mismatch for public checkpoint {weight_path.name}")
    model = smp.Unet(
        encoder_name=model_info["encoder"],
        encoder_weights=None,
        in_channels=1,
        classes=1,
    )
    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    return model.eval()


def u8_threshold(value: float) -> int:
    return int(round(value * 255))


def extract_instances(probability_u8: np.ndarray, image_stem: str) -> list[tuple[str, np.ndarray]]:
    structure = np.ones((3, 3), dtype=bool)
    labels, count = ndi.label(probability_u8 > u8_threshold(BOX_KEEP_RULE["t_lo"]), structure=structure)
    if count == 0:
        return []
    labels_flat = labels.ravel()
    nonzero = np.flatnonzero(labels_flat)
    component_ids = labels_flat[nonzero]
    values = probability_u8.ravel()[nonzero]
    areas = np.bincount(component_ids, minlength=count + 1)[1:].astype(np.int64)
    probability_sums = np.bincount(component_ids, weights=values, minlength=count + 1)[1:]
    peaks = np.zeros(count + 1, dtype=np.uint8)
    np.maximum.at(peaks, component_ids, values)
    means = probability_sums / np.maximum(areas, 1) / 255.0
    keep = (
        (peaks[1:] > u8_threshold(BOX_KEEP_RULE["t_hi"]))
        & (areas >= BOX_KEEP_RULE["min_area"])
        & (means >= BOX_KEEP_RULE["min_meanp"])
    )
    kept_ids = np.flatnonzero(keep) + 1
    bounds = ndi.find_objects(labels)
    output: list[tuple[str, np.ndarray]] = []
    for sequence, component_id in enumerate(kept_ids, start=1):
        bounds_for_component = bounds[component_id - 1]
        mask = np.zeros((FULL_SIZE, FULL_SIZE), dtype=np.uint8)
        mask[bounds_for_component] = labels[bounds_for_component] == component_id
        output.append((f"{image_stem}_{sequence}", mask))
    return output


def encode_and_verify(mask: np.ndarray) -> str:
    encoded = mask_utils.encode(np.asfortranarray(mask))
    counts = encoded["counts"].decode("ascii")
    decoded = mask_utils.decode({"size": [FULL_SIZE, FULL_SIZE], "counts": counts.encode("ascii")})
    if not np.array_equal(decoded, mask):
        raise ValueError("COCO RLE did not round-trip exactly")
    return counts


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
        default=REPO_ROOT / "data" / "reference" / "public-datasets" / "dariushafshar-solar-filament-unet-weights",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "submissions" / "dariushafshar-unet-weights-20261114",
    )
    parser.add_argument("--model-set", choices=("all", "resnet34", "efficientnetv2_s"), default="all")
    parser.add_argument("--tta-views", choices=(1, 2, 4, 8), type=int, default=8)
    parser.add_argument("--views-per-pass", choices=(1, 2, 4, 8), type=int, default=1)
    parser.add_argument("--limit-images", type=int, default=None, help="partial inference only; never submit this output")
    parser.add_argument("--expected-test-images", type=int, default=180)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    weights_root = args.weights_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    kaggle_runtime = Path("/kaggle/input").is_dir()
    allowed_output_root = Path("/kaggle/working").resolve() if kaggle_runtime else (REPO_ROOT / "artifacts").resolve()
    if not output_dir.is_relative_to(allowed_output_root):
        raise ValueError(f"Outputs must remain under {allowed_output_root}")
    allowed_data_roots = [(REPO_ROOT / "data").resolve()]
    if kaggle_runtime:
        allowed_data_roots.append(Path("/kaggle/input").resolve())
    if not any(data_root.is_relative_to(root) for root in allowed_data_roots):
        raise ValueError("Competition data must be inside workspace data/ or Kaggle input")
    if not any(weights_root.is_relative_to(root) for root in allowed_data_roots):
        raise ValueError("Public weights must be inside workspace data/ or Kaggle input")
    if args.limit_images is not None and args.limit_images < 1:
        parser.error("--limit-images must be positive")

    competition_root = find_dataset_root(data_root)
    test_dir = competition_root / "test" / "test_images"
    test_files = sorted(
        path for path in test_dir.iterdir() if path.is_file() and path.suffix.lower() in {".jpeg", ".jpg"}
    )
    if len(test_files) != args.expected_test_images:
        raise ValueError(f"Expected {args.expected_test_images} test images, found {len(test_files)}")
    if args.limit_images is not None:
        test_files = test_files[: args.limit_images]

    receipt_path = weights_root / "solar_unet_receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    selected_models = load_models(receipt, args.model_set)
    disk_y, disk_x = np.ogrid[:FULL_SIZE, :FULL_SIZE]
    disk = (disk_y - (FULL_SIZE - 1) / 2.0) ** 2 + (disk_x - (FULL_SIZE - 1) / 2.0) ** 2 <= DISK_RADIUS**2

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
    hardware = hardware_info()

    output_dir.mkdir(parents=True, exist_ok=True)
    probability_dir = output_dir / "model-probabilities"
    probability_dir.mkdir(exist_ok=True)
    state_path = output_dir / "inference-state.json"
    submission_path = output_dir / "submission.csv"
    report_path = output_dir / "run-report.json"
    if submission_path.exists() or report_path.exists():
        raise FileExistsError(f"A completed inference output already exists in {output_dir}")

    model_specs = [
        {
            "file": model_info["file"],
            "sha256": model_info["sha256"],
            "encoder": model_info["encoder"],
            "fold": model_info["fold"],
        }
        for model_info in selected_models
    ]
    run_config = {
        "receipt_sha256": sha256(receipt_path),
        "model_set": args.model_set,
        "models": model_specs,
        "test_files": [path.name for path in test_files],
        "expected_test_images": args.expected_test_images,
        "tta_views": args.tta_views,
        "views_per_pass": args.views_per_pass,
        "keep_rule": BOX_KEEP_RULE,
    }
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("config") != run_config:
            raise ValueError("Saved inference state belongs to a different model, image set, or configuration")
        completed_models = list(state.get("completed_models", []))
        weight_records = list(state.get("model_records", []))
        if not set(completed_models).issubset({model["file"] for model in selected_models}):
            raise ValueError("Saved inference state names a model outside the selected ensemble")
        for model_info in selected_models:
            if model_info["file"] in completed_models:
                probability_path = probability_dir / f"{model_info['file']}.npy"
                if not probability_path.is_file():
                    raise FileNotFoundError(f"Completed probability map is missing: {probability_path}")
                probabilities = np.load(probability_path, mmap_mode="r")
                if probabilities.shape != (len(test_files), FULL_SIZE, FULL_SIZE) or probabilities.dtype != np.uint8:
                    raise ValueError(f"Completed probability map has an unexpected shape or type: {probability_path}")
                del probabilities
                record = next((row for row in weight_records if row.get("file") == model_info["file"]), None)
                if record is None or sha256(probability_path) != record.get("probability_map_sha256"):
                    raise ValueError(f"Completed probability map failed its saved SHA-256 check: {probability_path}")
        print(f"[resume] reusing {len(completed_models)}/{len(selected_models)} completed model maps", flush=True)
    else:
        existing = [path for path in output_dir.iterdir() if path.name != probability_dir.name]
        existing_maps = list(probability_dir.iterdir())
        if existing or existing_maps:
            raise FileExistsError(f"Refusing to overwrite unrelated files in {output_dir}")
        completed_models = []
        weight_records = []
        state = {"config": run_config, "completed_models": completed_models, "model_records": weight_records}
        write_json_atomically(state_path, state)

    started = time.perf_counter()
    expected_map_bytes = len(test_files) * FULL_SIZE * FULL_SIZE
    for model_index, model_info in enumerate(selected_models, start=1):
        if model_info["file"] in completed_models:
            continue
        model_started = time.perf_counter()
        model = load_model(model_info, weights_root, device)
        model_path = weights_root / model_info["file"]
        probability_path = probability_dir / f"{model_info['file']}.npy"
        partial_path = probability_dir / f"{model_info['file']}.partial.npy"
        if probability_path.exists():
            probability_path.unlink()
        if partial_path.exists():
            partial_path.unlink()
        model_probabilities = np.lib.format.open_memmap(
            partial_path,
            mode="w+",
            dtype=np.uint8,
            shape=(len(test_files), FULL_SIZE, FULL_SIZE),
        )
        for image_index, image_path in enumerate(test_files):
            image = read_gray(image_path)
            mean, std = disk_stats(image, disk)
            prediction = predict_probability_u8(
                model,
                image,
                mean,
                std,
                device,
                tta_views=args.tta_views,
                views_per_pass=args.views_per_pass,
            )
            model_probabilities[image_index] = prediction
            del image, prediction
            if (image_index + 1) % 20 == 0 or image_index + 1 == len(test_files):
                print(
                    json.dumps(
                        {
                            "event": "image_progress",
                            "model": model_info["file"],
                            "completed": image_index + 1,
                            "total": len(test_files),
                        }
                    ),
                    flush=True,
                )
        model_probabilities.flush()
        del model_probabilities
        if partial_path.stat().st_size < expected_map_bytes:
            raise ValueError(f"Incomplete probability map: {partial_path}")
        partial = np.load(partial_path, mmap_mode="r")
        if partial.shape != (len(test_files), FULL_SIZE, FULL_SIZE) or partial.dtype != np.uint8:
            raise ValueError(f"Invalid probability map: {partial_path}")
        del partial
        partial_path.replace(probability_path)
        record = {
            "file": model_info["file"],
            "encoder": model_info["encoder"],
            "fold": model_info["fold"],
            "sha256": sha256(model_path),
            "probability_map_sha256": sha256(probability_path),
            "probability_map_bytes": probability_path.stat().st_size,
            "seconds": round(time.perf_counter() - model_started, 3),
        }
        weight_records.append(record)
        completed_models.append(model_info["file"])
        state = {
            "config": run_config,
            "completed_models": completed_models,
            "model_records": weight_records,
        }
        write_json_atomically(state_path, state)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            json.dumps(
                {
                    "event": "model_complete",
                    "model": model_index,
                    "models": len(selected_models),
                    "file": model_info["file"],
                    "seconds": record["seconds"],
                }
            ),
            flush=True,
        )

    if len(completed_models) != len(selected_models) or set(completed_models) != {model["file"] for model in selected_models}:
        raise RuntimeError("Not every selected model completed inference")
    model_probability_maps = [
        np.load(probability_dir / f"{model_info['file']}.npy", mmap_mode="r")
        for model_info in selected_models
    ]
    rows: list[dict[str, str]] = []
    predicted_images: set[str] = set()
    for image_index, image_path in enumerate(test_files):
        total = np.zeros((FULL_SIZE, FULL_SIZE), dtype=np.uint16)
        for probabilities in model_probability_maps:
            total += probabilities[image_index].astype(np.uint16)
        average = np.rint(total.astype(np.float32) / len(selected_models)).clip(0, 255).astype(np.uint8)
        for filament_id, mask in extract_instances(average, image_path.stem):
            rows.append({"filament_id": filament_id, "segmentation_rle": encode_and_verify(mask)})
            predicted_images.add(image_path.stem)
        if (image_index + 1) % 20 == 0 or image_index + 1 == len(test_files):
            print(
                json.dumps(
                    {"event": "postprocess_progress", "completed": image_index + 1, "total": len(test_files)}
                ),
                flush=True,
            )

    frame = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    if (frame.empty and args.limit_images is None) or frame.isna().any().any() or not frame.filament_id.is_unique:
        raise ValueError("Generated submission is empty or has null/duplicate rows")
    covered_stems = {str(value).rsplit("_", maxsplit=1)[0] for value in frame.filament_id}
    if not covered_stems.issubset({path.stem for path in test_files}):
        raise ValueError("Generated submission contains a non-test image ID")
    if args.limit_images is None and len(test_files) != args.expected_test_images:
        raise ValueError("Full submission does not cover all expected test images")

    temporary_submission = output_dir / "submission.partial.csv"
    frame.to_csv(temporary_submission, index=False, quoting=csv.QUOTE_MINIMAL)
    temporary_submission.replace(submission_path)
    elapsed = time.perf_counter() - started
    source_run_name = {"resnet34": "v3", "efficientnetv2_s": "v5", "all": "v6"}[args.model_set]
    source_metrics: dict[str, Any] = {}
    if args.tta_views == 8:
        source_run = receipt["runs"][source_run_name]
        source_metrics = {
            "source_reported_nested_oof_pq": source_run["oof"]["nested_cv_pq"],
            "source_reported_public_score": source_run["public_lb"],
        }
    report = {
        "run_name": "dariushafshar-public-unet-weights-test-inference",
        "source_attribution": ATTRIBUTION,
        "source_dataset": DATASET_URL,
        "source_dataset_license": DATASET_LICENSE,
        **source_metrics,
        "source_receipt_sha256": sha256(receipt_path),
        "source_commit": source_commit(),
        "source_script_sha256": sha256(Path(__file__).resolve()),
        "competition_root": str(competition_root),
        "source_training_data_provenance": (
            "The public source declares mask-only training on the 707 competition training observations; "
            "its ten fold models collectively used the full training split. No test labels are included."
        ),
        "training_annotations_read_during_inference": False,
        "test_labels_read": False,
        "frozen_holdout_validation_performed": False,
        "test_images_processed": len(test_files),
        "expected_test_images": args.expected_test_images,
        "full_test_coverage": len(test_files) == args.expected_test_images and args.limit_images is None,
        "images_with_predictions": len(predicted_images),
        "images_without_predictions": len(test_files) - len(predicted_images),
        "model_set": args.model_set,
        "models": weight_records,
        "tta_views": args.tta_views,
        "views_per_pass": args.views_per_pass,
        "normalization": "per-image mean and standard deviation over centered radius-950 disk after pixel/255",
        "postprocess": {**BOX_KEEP_RULE, "connectivity": 8, "threshold_comparison": "strict greater-than"},
        "submission": {
            "path": str(submission_path),
            "rows": int(len(frame)),
            "unique_ids": bool(frame.filament_id.is_unique),
            "sha256": sha256(submission_path),
            "bytes": submission_path.stat().st_size,
        },
        "elapsed_seconds": elapsed,
        "model_runtime_seconds": round(sum(float(record["seconds"]) for record in weight_records), 3),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "device": str(device),
        "hardware": hardware,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "timm": timm.__version__,
        "segmentation_models_pytorch": importlib.metadata.version("segmentation-models-pytorch"),
        "package_versions": package_versions(),
    }
    write_json_atomically(report_path, report)
    print(json.dumps({key: value for key, value in report.items() if key != "models"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
