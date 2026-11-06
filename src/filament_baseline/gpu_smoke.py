from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def run_gpu_smoke(output_path: str | Path | None = None) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch cannot see a CUDA GPU. Confirm the CUDA-enabled wheel and NVIDIA driver."
        )
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda:0")
    model = torch.nn.Conv2d(1, 8, kernel_size=3, padding=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    image = torch.randn(1, 1, 256, 256, device=device)
    target = torch.randn(1, 8, 256, 256, device=device)

    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        prediction = model(image)
        loss = torch.nn.functional.mse_loss(prediction, target)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    props = torch.cuda.get_device_properties(0)
    result = {
        "status": "passed",
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device_name": props.name,
        "device_memory_bytes": int(props.total_memory),
        "compute_capability": f"{props.major}.{props.minor}",
        "loss": float(loss.detach().cpu()),
        "elapsed_seconds": elapsed,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(0)),
        "shape": [1, 1, 256, 256],
        "amp": "float16",
    }
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Verify local CUDA forward/backward support.")
    parser.add_argument(
        "--output",
        default="artifacts/local-gpu-smoke/gpu_smoke.json",
        help="Output JSON path; keep it inside the repository.",
    )
    args = parser.parse_args()
    return 0 if run_gpu_smoke(args.output)["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
