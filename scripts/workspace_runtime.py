# SPDX-License-Identifier: AGPL-3.0-only
"""Keep model caches and temporary files inside the active workspace."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def configure_workspace_runtime(repo_root: str | Path) -> dict[str, str]:
    """Redirect common ML caches and temporary files before importing frameworks."""
    root = Path(repo_root).expanduser().resolve()
    cache = root / ".cache"
    temporary = root / ".tmp"
    paths = {
        "HF_HOME": cache / "huggingface",
        "HF_HUB_CACHE": cache / "huggingface" / "hub",
        "HUGGINGFACE_HUB_CACHE": cache / "huggingface" / "hub",
        "TRANSFORMERS_CACHE": cache / "huggingface" / "transformers",
        "TORCH_HOME": cache / "torch",
        "TORCH_EXTENSIONS_DIR": cache / "torch" / "extensions",
        "TORCHINDUCTOR_CACHE_DIR": cache / "torch" / "inductor",
        "TRITON_CACHE_DIR": cache / "triton",
        "XDG_CACHE_HOME": cache,
        "MPLCONFIGDIR": cache / "matplotlib",
        "PIP_CACHE_DIR": cache / "pip",
        "NUMBA_CACHE_DIR": cache / "numba",
        "YOLO_CONFIG_DIR": cache / "Ultralytics",
        "TEMP": temporary,
        "TMP": temporary,
        "TMPDIR": temporary,
    }
    for key, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    tempfile.tempdir = str(temporary)
    return {key: str(path) for key, path in paths.items()}
