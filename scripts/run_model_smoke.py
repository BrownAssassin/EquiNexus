# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import json
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from filament_baseline.data import load_dataset  # noqa: E402
from filament_baseline.features import input_channel_count  # noqa: E402
from filament_baseline.inference import predict_probabilities  # noqa: E402
from filament_baseline.learning import FilamentPatchDataset  # noqa: E402
from filament_baseline.model import FilamentUNet  # noqa: E402
from filament_baseline.split import create_observation_split  # noqa: E402
from filament_baseline.training import training_loss  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test model training, crops, and inference.")
    parser.add_argument(
        "--input-mode", choices=("raw", "raw_local_contrast", "raw_local_contrast_radial"), default="raw"
    )
    args = parser.parse_args()
    started = time.perf_counter()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    in_channels = input_channel_count(args.input_mode)
    model = FilamentUNet(in_channels=in_channels).to(device)
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)
        torch.cuda.reset_peak_memory_stats(device)
    image = torch.randn(2, in_channels, 256, 256, device=device)
    target = torch.rand(2, 2, 256, 256, device=device)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        logits = model(image)
        loss = training_loss(logits, target)
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    dataset = load_dataset(REPO_ROOT / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026")
    split = create_observation_split(dataset, REPO_ROOT / "data" / "splits", seed=42, validation_fraction=0.2)
    sample_data = FilamentPatchDataset(
        dataset,
        split.training_names[:8],
        target_mode="consensus",
        patch_size=256,
        augment=True,
        input_mode=args.input_mode,
    )
    patch, targets = sample_data[0]
    if tuple(patch.shape) != (in_channels, 256, 256) or tuple(targets.shape) != (2, 256, 256):
        raise AssertionError("Training patch shape mismatch")

    full_image = np.full((2048, 2048), 150, dtype=np.uint8)
    import cv2

    cv2.line(full_image, (970, 1020), (1080, 1030), 25, 8)
    foreground, boundary = predict_probabilities(
        model.eval(), full_image, device=device, patch_size=512, stride=384, tta=1
    )
    if foreground.shape != (2048, 2048) or boundary.shape != foreground.shape:
        raise AssertionError("Sliding-window inference shape mismatch")

    result = {
        "status": "passed",
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "device_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory)
        if torch.cuda.is_available()
        else 0,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0,
        "patch_shape": list(patch.shape),
        "target_shape": list(targets.shape),
        "input_mode": args.input_mode,
        "inference_shape": list(foreground.shape),
        "split_sha256": split.sha256,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = (
        REPO_ROOT
        / "artifacts"
        / "model-experiments"
        / "smoke"
        / f"model_smoke_{args.input_mode}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
