from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


TRACKED_RUNTIME_PACKAGES = (
    "numpy",
    "opencv-python-headless",
    "pandas",
    "scikit-learn",
    "scipy",
    "matplotlib",
    "pycocotools",
    "torch",
)


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in TRACKED_RUNTIME_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def hardware_info() -> dict[str, Any]:
    result: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cuda_available": False,
        "torch_version": None,
        "torch_cuda_version": None,
        "gpu_devices": [],
        "nvidia_smi_summary": None,
    }
    try:
        result["nvidia_smi_summary"] = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().splitlines()
    except (OSError, subprocess.CalledProcessError):
        pass
    try:
        import torch
    except ImportError:
        return result

    result["torch_version"] = torch.__version__
    result["torch_cuda_version"] = torch.version.cuda
    result["cuda_available"] = bool(torch.cuda.is_available())
    if result["cuda_available"]:
        devices = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_bytes": int(props.total_memory),
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )
        result["gpu_devices"] = devices
        torch.cuda.reset_peak_memory_stats()
    return result


def git_info(repo_root: str | Path) -> dict[str, Any]:
    repo_root = Path(repo_root)
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {"commit": commit, "worktree_dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "worktree_dirty": None}
