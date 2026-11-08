# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from filament_baseline.model import DINO_V2_REPOSITORY  # noqa: E402
from filament_baseline.solar_ssl import adapt_dinov2_on_solar_images  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Self-supervised solar-domain adaptation of DINOv2.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--hub-dir", type=Path, default=None)
    parser.add_argument("--hub-repository", default=DINO_V2_REPOSITORY)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--projector-learning-rate", type=float, default=5e-4)
    parser.add_argument("--trainable-blocks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    result = adapt_dinov2_on_solar_images(
        data_root=args.data_root,
        output_path=args.output_path,
        steps=args.steps,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        projector_learning_rate=args.projector_learning_rate,
        trainable_blocks=args.trainable_blocks,
        seed=args.seed,
        device_name=args.device,
        hub_repository=args.hub_repository,
        hub_dir=args.hub_dir,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
