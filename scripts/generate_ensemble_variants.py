from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from filament_baseline.data import IMAGE_SIZE, load_dataset  # noqa: E402
from filament_baseline.inference import InstanceConfig, instances_from_probabilities  # noqa: E402
from filament_baseline.predictor import PredictedInstance  # noqa: E402
from filament_baseline.runner import _git_status, _write_predictions  # noqa: E402
from filament_baseline.runtime import hardware_info, package_versions  # noqa: E402
from filament_baseline.submission import _check_submission  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_evidence(repo_root: Path, relative_path: str) -> dict[str, Any]:
    path = (repo_root / relative_path).resolve()
    if not path.is_relative_to(repo_root) or not path.is_file():
        raise ValueError(f"OOF evidence must be a tracked local artifact path: {relative_path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    metrics = value.get("metrics", {}).get("global") or value.get("aggregate_oof", {})
    if "pq" not in metrics:
        raise ValueError(f"OOF evidence has no aggregate PQ: {path}")
    return {
        "path": str(path.relative_to(repo_root)),
        "sha256": _sha256(path),
        "pq": metrics["pq"],
        "tp": metrics.get("tp"),
        "fp": metrics.get("fp"),
        "fn": metrics.get("fn"),
        "observations": value.get("oof_observations")
        or value.get("splits", {}).get("oof_observations")
        or value.get("training_observations"),
    }


def generate_variants(
    *,
    repo_root: Path,
    data_root: Path,
    run_root: Path,
    output_dir: Path,
    config_path: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    data_root = data_root.resolve()
    run_root = run_root.resolve()
    output_dir = output_dir.resolve()
    config_path = config_path.resolve()
    for name, path in (("data", data_root), ("run", run_root), ("output", output_dir), ("config", config_path)):
        if not path.is_relative_to(repo_root):
            raise ValueError(f"Local {name} paths must remain inside the repository: {path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output folder already contains files: {output_dir}")

    source_metrics_path = run_root / "metrics.json"
    source_metrics = json.loads(source_metrics_path.read_text(encoding="utf-8"))
    if source_metrics.get("status") != "complete":
        raise ValueError("Source run must be complete")
    if source_metrics.get("data_boundary", {}).get("external_ground_truth") is not False:
        raise ValueError("Source run must certify no external ground-truth use")
    expected_modes = {
        "annotator1024": ("annotator", 1024, 8),
        "consensus512": ("consensus", 512, 4),
    }
    for member_name, (target_mode, patch_size, tta) in expected_modes.items():
        member = next((item for item in source_metrics.get("members", []) if item.get("name") == member_name), None)
        if member is None:
            raise ValueError(f"Source run is missing required member: {member_name}")
        config = member.get("configuration", {})
        if (config.get("target_mode"), config.get("patch_size"), config.get("tta")) != (target_mode, patch_size, tta):
            raise ValueError(f"Unexpected training configuration for member {member_name}")

    candidates = json.loads(config_path.read_text(encoding="utf-8")).get("candidates", [])
    if len(candidates) < 1:
        raise ValueError("At least one submission candidate must be configured")
    names = [candidate.get("name") for candidate in candidates]
    if len(set(names)) != len(names) or any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name) for name in names):
        raise ValueError("Candidate names must be unique safe identifiers")

    dataset = load_dataset(data_root)
    test_names = sorted(path.name for path in dataset.test_image_dir.iterdir() if path.is_file())
    if len(test_names) != 180 or len({Path(name).stem for name in test_names}) != 180:
        raise ValueError(f"Expected 180 uniquely named official test images, found {len(test_names)}")
    first_cache = run_root / "members" / "annotator1024" / "probabilities"
    second_cache = run_root / "members" / "consensus512" / "probabilities"
    for cache_dir in (first_cache, second_cache):
        present = {path.name for path in cache_dir.glob("*.npz")}
        expected = {f"{Path(name).stem}.npz" for name in test_names}
        if present != expected:
            raise ValueError(f"Probability cache inventory differs from official test set: {cache_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    git = _git_status(repo_root)
    result_candidates = []
    started = time.perf_counter()
    for candidate in candidates:
        candidate_started = time.perf_counter()
        weight = float(candidate["first_weight"])
        if not 0 <= weight <= 1:
            raise ValueError(f"Invalid first model weight for {candidate['name']}")
        instance_config = InstanceConfig(**candidate["instance"])
        predictions: dict[str, list[PredictedInstance]] = {}
        counts = []
        for index, file_name in enumerate(test_names, start=1):
            stem = Path(file_name).stem
            with np.load(first_cache / f"{stem}.npz") as first, np.load(second_cache / f"{stem}.npz") as second:
                arrays = [
                    first["foreground"].astype(np.float32),
                    first["boundary"].astype(np.float32),
                    second["foreground"].astype(np.float32),
                    second["boundary"].astype(np.float32),
                ]
            if any(array.shape != (IMAGE_SIZE, IMAGE_SIZE) for array in arrays):
                raise ValueError(f"Invalid probability shape for {file_name}")
            if any(not np.isfinite(array).all() for array in arrays):
                raise ValueError(f"Non-finite probability values for {file_name}")
            foreground = weight * arrays[0] + (1.0 - weight) * arrays[2]
            boundary = weight * arrays[1] + (1.0 - weight) * arrays[3]
            predictions[file_name] = instances_from_probabilities(foreground, boundary, file_name, instance_config)
            counts.append({"file_name": file_name, "instance_count": len(predictions[file_name])})
            if index % 30 == 0 or index == len(test_names):
                print(f"{candidate['name']}: postprocessed {index}/{len(test_names)}", flush=True)

        candidate_dir = output_dir / candidate["name"]
        candidate_dir.mkdir()
        submission_path = candidate_dir / "submission.csv"
        _write_predictions(submission_path, predictions)
        validation = _check_submission(submission_path, dataset.root, test_names, predictions)
        with (candidate_dir / "test_coverage.csv").open("w", encoding="utf-8", newline="") as stream:
            import csv

            writer = csv.DictWriter(stream, fieldnames=("file_name", "instance_count"), lineterminator="\n")
            writer.writeheader()
            writer.writerows(counts)
        evidence = _load_evidence(repo_root, candidate["oof_evidence"])
        row = {
            "name": candidate["name"],
            "probability_weights": {"annotator1024": weight, "consensus512": 1.0 - weight},
            "instance_config": instance_config.__dict__,
            "oof_evidence": evidence,
            "test_validation": validation,
            "submission_sha256": _sha256(submission_path),
            "elapsed_seconds": time.perf_counter() - candidate_started,
            "artifacts": {
                "submission": str(submission_path.relative_to(repo_root)),
                "coverage": str((candidate_dir / "test_coverage.csv").relative_to(repo_root)),
            },
        }
        result_candidates.append(row)
        print(json.dumps({key: row[key] for key in ("name", "probability_weights", "oof_evidence", "test_validation", "submission_sha256")}, indent=2), flush=True)

    source_commit = git.get("commit")
    result = {
        "status": "complete",
        "source_run": str(run_root.relative_to(repo_root)),
        "source_metrics_sha256": _sha256(source_metrics_path),
        "source_commit": source_commit,
        "source_worktree_dirty": git.get("worktree_dirty"),
        "candidate_config": str(config_path.relative_to(repo_root)),
        "candidate_config_sha256": _sha256(config_path),
        "data_boundary": {
            "training_labels": "competition-provided training annotations only",
            "inference_inputs": "official competition test H-alpha images only",
            "test_labels_read": False,
        },
        "dataset": {"test_images": len(test_names)},
        "runtime": {**hardware_info(), "package_versions": package_versions(), "elapsed_seconds": time.perf_counter() - started},
        "candidates": result_candidates,
    }
    (output_dir / "variants.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate validation-supported test submissions from a completed ensemble probability cache.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "submission-candidates-2026-11-06.json")
    args = parser.parse_args()
    output = generate_variants(repo_root=args.repo_root, data_root=args.data_root, run_root=args.run_root, output_dir=args.output_dir, config_path=args.config)
    print(json.dumps({"status": output["status"], "candidate_count": len(output["candidates"]), "elapsed_seconds": output["runtime"]["elapsed_seconds"]}, indent=2))
