# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Apply the cross-fitted component selector to full-fit test predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from pycocotools import mask as mask_utils

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_component_selector_oof import (  # noqa: E402
    extract_observation_features,
    fit_predict,
    FEATURES,
    load_detector_rows,
    load_prediction_rows,
    load_probability_maps,
    predictions_from_masks,
)
from filament_baseline.data import load_dataset  # noqa: E402
from filament_baseline.predictor import PredictedInstance  # noqa: E402
from filament_baseline.runner import _write_predictions  # noqa: E402
from filament_baseline.submission import _check_submission  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def keep_best_when_empty(
    names: list[str],
    masks_by_name: dict[str, list[dict[str, Any]]],
    probabilities_by_key: dict[tuple[str, int], float],
    threshold: float,
    predictions: dict[str, list[PredictedInstance]],
) -> list[str]:
    """Restore the top-ranked base component only when filtering empties an image."""
    fallback_names: list[str] = []
    for file_name in names:
        stem = Path(file_name).stem
        components = masks_by_name[stem]
        if predictions[file_name] or not components:
            continue
        best_index = max(
            range(1, len(components) + 1),
            key=lambda index: probabilities_by_key[(file_name, index)],
        )
        rle = components[best_index - 1]
        predictions[file_name] = [
            PredictedInstance(
                filament_id=f"{stem}_1",
                image_stem=stem,
                area=int(mask_utils.area(rle)),
                rle=rle,
            )
        ]
        fallback_names.append(file_name)
    return fallback_names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--selector-run", type=Path, default=REPO_ROOT / "artifacts/model-experiments/component-selector-official-pq-20261109-rle-fixed")
    parser.add_argument("--base-submission-run", type=Path, default=REPO_ROOT / "artifacts/submissions/final-yolo11s-fullfit-p25-20261108")
    parser.add_argument("--unet-run", type=Path, default=REPO_ROOT / "artifacts/submissions/final-unet-ensemble-fulltrain-20261108")
    parser.add_argument("--detector-run", type=Path, default=REPO_ROOT / "artifacts/detector-experiments/yolo11s-seg-1280-final-fulltrain-r1-20261108/test-inference-tta8/dedup-iou-030")
    parser.add_argument("--run-name", default="component-selector-fullfit-test-20261109")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional threshold from the saved OOF-only screen; defaults to its selected value.",
    )
    parser.add_argument(
        "--c-value",
        type=float,
        default=1.0,
        help="Logistic regularization selected from the grouped OOF model-family screen.",
    )
    parser.add_argument(
        "--keep-best-on-empty",
        action="store_true",
        help="Keep the highest-probability base component only when selector filtering empties an image.",
    )
    parser.add_argument(
        "--fallback-evaluation",
        type=Path,
        default=REPO_ROOT / "artifacts/verification/component-selector-empty-fallback-20261109/metrics.json",
        help="OOF and fixed-holdout evidence required for the optional empty-output fallback.",
    )
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    selector_run = args.selector_run.resolve()
    base_run = args.base_submission_run.resolve()
    unet_run = args.unet_run.resolve()
    detector_run = args.detector_run.resolve()
    for path, parent, label in (
        (data_root, REPO_ROOT / "data", "dataset"),
        (selector_run, REPO_ROOT / "artifacts", "selector run"),
        (base_run, REPO_ROOT / "artifacts", "base submission"),
        (unet_run, REPO_ROOT / "artifacts", "U-Net run"),
        (detector_run, REPO_ROOT / "artifacts", "detector run"),
    ):
        if not path.is_relative_to(parent.resolve()):
            raise ValueError(f"{label} must remain inside {parent}: {path}")
    output_dir = (args.output_dir or (REPO_ROOT / "artifacts/submissions" / args.run_name)).resolve()
    if not output_dir.is_relative_to((REPO_ROOT / "artifacts/submissions").resolve()):
        raise ValueError("Output directory must remain inside artifacts/submissions/")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)

    started = time.perf_counter()
    dataset = load_dataset(data_root)
    test_names = sorted(path.name for path in dataset.test_image_dir.iterdir() if path.is_file())
    if len(test_names) != 180:
        raise RuntimeError(f"Expected the 180 supplied test images, found {len(test_names)}")
    test_stems = {Path(name).stem for name in test_names}

    selector_metrics_path = selector_run / "metrics.json"
    selector_metrics = json.loads(selector_metrics_path.read_text(encoding="utf-8"))
    if selector_metrics.get("status") != "complete" or selector_metrics.get("split_sha256") != "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e":
        raise RuntimeError("Selector metrics do not identify the completed frozen-split experiment")
    threshold = (
        float(selector_metrics["fit"]["selected_threshold"])
        if args.threshold is None
        else float(args.threshold)
    )
    model_screen_path: Path | None = None
    oof_screen_pq: float | None = None
    if args.c_value == 1.0:
        screened_thresholds = {
            float(row["threshold"]): float(row["metrics"]["global"]["pq"])
            for row in selector_metrics["fit"]["thresholds"]
        }
        if threshold not in screened_thresholds:
            raise RuntimeError(f"Threshold {threshold} was not screened on cross-fitted OOF rows")
        oof_screen_pq = screened_thresholds[threshold]
    else:
        model_screen_path = REPO_ROOT / "artifacts/model-experiments/component-selector-model-screen-20261109/metrics.json"
        if not model_screen_path.is_file():
            raise FileNotFoundError("Missing grouped-OOF model-family screen for non-default C value")
        model_screen = json.loads(model_screen_path.read_text(encoding="utf-8"))
        if model_screen.get("split_sha256") != selector_metrics["split_sha256"]:
            raise RuntimeError("Model-family screen uses a different frozen split")
        expected_feature_sha = model_screen.get("input_sha256", {}).get("oof_features_targets")
        if expected_feature_sha != sha256(selector_run / "oof-component-features-targets.csv"):
            raise RuntimeError("Model-family screen used different OOF component features")
        model_key = f"logistic_c{args.c_value}"
        model_row = model_screen.get("models", {}).get(model_key)
        if model_row is None or float(model_row["best_threshold"]) != threshold:
            raise RuntimeError("Non-default model and threshold must match the saved OOF screen winner")
        oof_screen_pq = float(model_row["best_metrics"]["global"]["pq"])
    if args.keep_best_on_empty and (args.c_value != 1.0 or threshold != 0.40):
        raise RuntimeError("The empty-output fallback was validated only for C=1.0 at threshold 0.40")

    base_submission_path = base_run / "submission.csv"
    base_masks = load_prediction_rows(base_submission_path, test_stems)
    detector_path = detector_run / "test_detections_deduplicated.json"
    detector_rows = load_detector_rows(detector_path, test_stems)
    perann_root = unet_run / "members/annotator1024/probabilities"
    consensus_root = unet_run / "members/consensus512/probabilities"
    perann_maps = sorted(perann_root.glob("*.npz"))
    consensus_maps = sorted(consensus_root.glob("*.npz"))
    if len(perann_maps) != 180 or len(consensus_maps) != 180:
        raise RuntimeError("Full-fit U-Net probability caches do not cover all 180 test images")

    feature_rows: list[dict[str, Any]] = []
    for index, file_name in enumerate(test_names, start=1):
        foreground, boundary = load_probability_maps(
            file_name,
            fold=None,
            first_root=perann_root,
            second_root=consensus_root,
        )
        feature_rows.extend(
            extract_observation_features(
                dataset,
                file_name,
                base_masks[Path(file_name).stem],
                detector_rows[Path(file_name).stem],
                foreground,
                boundary,
                image_dir=dataset.test_image_dir,
            )
        )
        if index % 25 == 0 or index == len(test_names):
            print(f"Built test selector features {index}/{len(test_names)}", flush=True)

    test_features = pd.DataFrame(feature_rows)
    oof_feature_path = selector_run / "oof-component-features-targets.csv"
    oof_features = pd.read_csv(oof_feature_path)
    probabilities, fitted_model = fit_predict(oof_features, test_features, c_value=args.c_value)
    test_features["selector_probability"] = probabilities
    probability_by_key = {
        (str(row.file_name), int(row.component_id)): float(row.selector_probability)
        for row in test_features.itertuples()
    }
    predictions = predictions_from_masks(test_names, base_masks, probability_by_key, threshold)
    fallback_names: list[str] = []
    fallback_evaluation: dict[str, Any] | None = None
    if args.keep_best_on_empty:
        fallback_path = args.fallback_evaluation.resolve()
        if not fallback_path.is_relative_to((REPO_ROOT / "artifacts/verification").resolve()):
            raise ValueError("Fallback evidence must remain inside artifacts/verification/")
        fallback_evaluation = json.loads(fallback_path.read_text(encoding="utf-8"))
        if fallback_evaluation.get("status") != "complete":
            raise RuntimeError("Empty-output fallback has no completed validation record")
        if fallback_evaluation.get("split_sha256") != selector_metrics["split_sha256"]:
            raise RuntimeError("Fallback evidence uses a different frozen split")
        if float(fallback_evaluation.get("threshold", -1)) != threshold:
            raise RuntimeError("Fallback evidence uses a different selector threshold")
        fallback_names = keep_best_when_empty(
            test_names, base_masks, probability_by_key, threshold, predictions
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = output_dir / "submission.csv"
    _write_predictions(submission_path, predictions)
    validation = _check_submission(submission_path, data_root, test_names, predictions)
    test_features.to_csv(output_dir / "test-component-features.csv", index=False)
    with (output_dir / "test-coverage.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("file_name", "base_instances", "selected_instances"))
        for file_name in test_names:
            writer.writerow((file_name, len(base_masks[Path(file_name).stem]), len(predictions[file_name])))

    source_paths = {
        "base_submission": base_submission_path,
        "base_submission_metrics": base_run / "metrics.json",
        "detector_predictions": detector_path,
        "detector_metadata": detector_run / "run-metadata.json",
        "selector_metrics": selector_metrics_path,
        "oof_component_features_targets": oof_feature_path,
        "annotator_unet_metrics": unet_run / "members/annotator1024/metrics.json",
        "consensus_unet_metrics": unet_run / "members/consensus512/metrics.json",
    }
    if args.keep_best_on_empty and fallback_evaluation is not None:
        source_paths["fallback_validation_metrics"] = args.fallback_evaluation.resolve()
    if model_screen_path is not None:
        source_paths["model_family_screen_metrics"] = model_screen_path
    test_map_paths = perann_maps + consensus_maps
    result = {
        "status": "complete",
        "run_name": args.run_name,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_boundary": {
            "training_labels": "competition-provided training annotations only",
            "inference_inputs": "180 supplied competition H-alpha test images only",
            "test_labels_used": False,
            "test_images_used_for_training": False,
            "selector_fit_rows": "cross-fitted OOF training observations only",
        },
        "selection": {
            "threshold": threshold,
            "c_value": float(args.c_value),
            "method": "standardized L2 logistic regression refit on grouped OOF component targets",
            "feature_count": len(FEATURES),
            "component_count": int(len(test_features)),
            "base_prediction_count": int(sum(len(items) for items in base_masks.values())),
            "selected_prediction_count": int(sum(len(items) for items in predictions.values())),
            "oof_screen_pq": oof_screen_pq,
            "holdout_score_is_for_this_model_and_threshold": bool(
                args.c_value == 1.0 and threshold == float(selector_metrics["fit"]["selected_threshold"])
            ),
            "reference_holdout_pq_for_c1_threshold040": float(selector_metrics["fit"]["holdout_selected"]["global"]["pq"]),
            "validation_baseline_holdout_pq": float(selector_metrics["fit"]["holdout_baseline"]["global"]["pq"]),
            "fitted_model": fitted_model,
            "empty_output_fallback": {
                "enabled": bool(args.keep_best_on_empty),
                "policy": (
                    "When selector filtering removes every base component, keep the highest-probability component."
                    if args.keep_best_on_empty
                    else None
                ),
                "test_observations": fallback_names,
                "oof_pq": (
                    float(fallback_evaluation["oof"]["fallback_pq"])
                    if fallback_evaluation is not None
                    else None
                ),
                "holdout_pq": (
                    float(fallback_evaluation["holdout"]["fallback_pq"])
                    if fallback_evaluation is not None
                    else None
                ),
                "holdout_paired_delta_ci_95": (
                    fallback_evaluation["holdout"]["paired_bootstrap"]["paired_delta_ci_95"]
                    if fallback_evaluation is not None
                    else None
                ),
            },
        },
        "coverage_validation": validation,
        "split_sha256": selector_metrics["split_sha256"],
        "oof_manifest_sha256": selector_metrics["oof_manifest_sha256"],
        "source_hashes": {
            name: sha256(path) for name, path in source_paths.items() if path.is_file()
        },
        "probability_cache_digest": tree_digest(test_map_paths),
        "probability_cache_file_count": len(test_map_paths),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "source_worktree_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
            ).strip()
        ),
        "source_code_hashes": {
            "selector": sha256(REPO_ROOT / "scripts/run_component_selector_oof.py"),
            "candidate_builder": sha256(Path(__file__).resolve()),
            "scorer": sha256(REPO_ROOT / "src/filament_baseline/scorer.py"),
        },
        "submission_sha256": sha256(submission_path),
        "runtime_seconds": time.perf_counter() - started,
        "artifacts": {
            "submission": submission_path.name,
            "test_coverage": "test-coverage.csv",
            "test_component_features": "test-component-features.csv",
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "threshold": threshold,
        "base_instances": result["selection"]["base_prediction_count"],
        "selected_instances": result["selection"]["selected_prediction_count"],
        "oof_screen_pq": result["selection"]["oof_screen_pq"],
        "holdout_score_is_for_this_model_and_threshold": result["selection"]["holdout_score_is_for_this_model_and_threshold"],
        "coverage": validation,
        "output": str(output_dir),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
