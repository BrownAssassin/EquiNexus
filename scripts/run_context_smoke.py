from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from filament_baseline.model import FilamentUNet  # noqa: E402
from filament_baseline.training import training_loss  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure mixed-precision U-Net training memory for a large crop."
    )
    parser.add_argument("--patch-size", type=int, default=1888)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.patch_size < 128 or args.patch_size % 32 or args.batch_size < 1:
        parser.error("patch size must be >=128 and divisible by 32; batch size must be positive")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = FilamentUNet().to(device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        model.to(memory_format=torch.channels_last)
        torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    started = time.perf_counter()
    image = torch.randn(
        args.batch_size, 1, args.patch_size, args.patch_size, device=device
    )
    target = torch.rand(
        args.batch_size, 2, args.patch_size, args.patch_size, device=device
    )
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        logits = model(image)
        loss = training_loss(logits, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    result = {
        "status": "passed",
        "patch_size": args.patch_size,
        "batch_size": args.batch_size,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "device_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory)
        if device.type == "cuda"
        else 0,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = (
        REPO_ROOT
        / "artifacts"
        / "model-experiments"
        / f"context-smoke-{args.patch_size}-batch{args.batch_size}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
