# SPDX-License-Identifier: AGPL-3.0-only
"""Optionally final-fit the validated clDice model, then predict supplied test images."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))
from workspace_runtime import configure_workspace_runtime  # noqa: E402

configure_workspace_runtime(REPO_ROOT)

import cv2  # noqa: E402
import torch  # noqa: E402

import train_native_unet_grouped as native  # noqa: E402
from filament_baseline.runner import _write_predictions  # noqa: E402
from validate_submission_rles import validate_submission  # noqa: E402


SPLIT_SHA256 = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"
DEFAULT_RUN_DIR = REPO_ROOT / "artifacts/model-experiments/native-unet-cldice-holdout-20261112"
DEFAULT_PRELIMINARY_OUTPUT_DIR = REPO_ROOT / "artifacts/submissions/native-unet-cldice-train565-20261113"
DEFAULT_FINAL_OUTPUT_DIR = REPO_ROOT / "artifacts/submissions/native-unet-cldice-final-20261113"
DEFAULT_FINAL_FIT_DIR = REPO_ROOT / "artifacts/model-experiments/native-unet-cldice-final-707-20261113"
EXPECTED_TEST_IMAGES = 180


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_within(path: Path, root: Path, label: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must remain under {root}: {path}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fit-dir", type=Path, default=DEFAULT_FINAL_FIT_DIR)
    parser.add_argument(
        "--fit-all-training",
        action="store_true",
        help="retrain the validated configuration on all 707 competition training observations before inference",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    default_output = DEFAULT_FINAL_OUTPUT_DIR if args.fit_all_training else DEFAULT_PRELIMINARY_OUTPUT_DIR
    output_dir = (args.output_dir or default_output).expanduser().resolve()
    fit_dir = args.fit_dir.expanduser().resolve()
    require_within(data_root, REPO_ROOT / "data", "Competition data")
    require_within(run_dir, REPO_ROOT / "artifacts/model-experiments", "clDice run")
    require_within(output_dir, REPO_ROOT / "artifacts/submissions", "Submission output")
    require_within(fit_dir, REPO_ROOT / "artifacts/model-experiments", "Final-fit output")

    metadata_path = run_dir / "run-metadata.json"
    metrics_path = run_dir / "holdout-metrics.json"
    comparison_path = run_dir / "paired-bootstrap-vs-best-holdout.json"
    for path in (metadata_path, metrics_path, comparison_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required completed validation artifact is missing: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    baseline_path = REPO_ROOT / "artifacts/model-experiments/public-unet-yolo-family-vote-holdout-tta8-conf025-20261111/run-metadata.json"
    config = metadata.get("config", {})
    if config.get("split_sha256") != SPLIT_SHA256 or metrics.get("split_sha256") != SPLIT_SHA256:
        raise RuntimeError("clDice artifacts do not use the frozen observation split")
    if comparison.get("split_sha256") != SPLIT_SHA256 or comparison.get("observation_count") != 142:
        raise RuntimeError("Paired comparison does not cover the frozen 142-observation holdout")
    if comparison.get("resamples") != 20_000 or comparison.get("seed") != 42:
        raise RuntimeError("Paired comparison does not match the predeclared 20,000-resample seed-42 protocol")
    expected_baseline = "artifacts/model-experiments/public-unet-yolo-family-vote-holdout-tta8-conf025-20261111/run-metadata.json"
    if comparison.get("baseline", {}).get("path") != expected_baseline:
        raise RuntimeError("Paired comparison did not use the current best independent holdout incumbent")
    if not baseline_path.is_file() or comparison.get("baseline", {}).get("sha256") != sha256_file(baseline_path):
        raise RuntimeError("Paired comparison baseline hash does not match the recorded incumbent")
    baseline_metadata = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_pq = float(baseline_metadata["metrics"]["global"]["pq"])
    if (
        baseline_metadata.get("fixed_holdout_split_sha256") != SPLIT_SHA256
        or baseline_metadata.get("holdout_used_for_training") is not False
        or baseline_metadata.get("holdout_used_for_model_or_threshold_selection") is not False
        or abs(baseline_pq - float(comparison["baseline"]["pq"])) > 1e-12
    ):
        raise RuntimeError("Paired comparison baseline score differs from incumbent metadata")
    candidate_record = comparison.get("candidate", {})
    expected_candidate = "artifacts/model-experiments/native-unet-cldice-holdout-20261112/run-metadata.json"
    candidate_rows_path = run_dir / "holdout-per-observation.csv"
    if (
        candidate_record.get("path") != expected_candidate
        or candidate_record.get("sha256") != sha256_file(metadata_path)
        or candidate_record.get("metrics_sha256") != sha256_file(metrics_path)
        or candidate_record.get("per_observation_sha256") != sha256_file(candidate_rows_path)
    ):
        raise RuntimeError("Paired comparison does not match the current clDice run artifacts")
    bootstrap_record = comparison.get("bootstrap_implementation", {})
    bootstrap_path = REPO_ROOT / "scripts/compare_native_unet_cldice_holdout.py"
    if (
        bootstrap_record.get("path") != "scripts/compare_native_unet_cldice_holdout.py"
        or bootstrap_record.get("sha256") != sha256_file(bootstrap_path)
    ):
        raise RuntimeError("Paired comparison was not produced by the recorded bootstrap implementation")
    if comparison.get("candidate", {}).get("holdout_threshold_tuning") is not False:
        raise RuntimeError("Paired comparison does not certify fixed, untuned holdout inference")
    if config.get("training_observations") != 565 or config.get("holdout_observations") != 142:
        raise RuntimeError("The clDice screen must train on 565 groups and score 142 groups")
    if metrics.get("training_observations") != 565 or metrics.get("holdout_observations") != 142:
        raise RuntimeError("Completed holdout metrics report an unexpected partition size")
    if metrics.get("test_labels_read") is not False:
        raise RuntimeError("The holdout run does not certify test-label exclusion")
    pretrained = metadata.get("pretrained_encoder_provenance", {})
    if (
        metadata.get("test_labels_read") is not False
        or metadata.get("external_data_used") is not True
        or metadata.get("external_pretrained_weights_used") is not True
        or metadata.get("external_segmentation_labels_used") is not False
        or pretrained.get("license") != "Apache-2.0"
        or pretrained.get("weight_file_sha256") != "f5726851b5c55fab49f8302a79c3a898814e11bc20df76cc23cab28ae15c7328"
    ):
        raise RuntimeError("The clDice run does not certify its permitted pretrained-model and label data boundary")
    if config.get("holdout_threshold_tuning") is not False or config.get("cldice_weight") != 0.25:
        raise RuntimeError("Unexpected clDice holdout-selection policy or loss weight")
    if config.get("encoder") != native.ENCODER_NAME or int(config.get("tta_views", -1)) != 8:
        raise RuntimeError("Unexpected clDice encoder or test-time augmentation configuration")
    official_pq = float(metrics["official_scorer_metrics"]["global"]["pq"])
    paired_candidate_pq = float(comparison.get("candidate", {}).get("pq", -1.0))
    if abs(official_pq - paired_candidate_pq) > 1e-12:
        raise RuntimeError("Paired comparison candidate score differs from official holdout metrics")
    delta = comparison.get("paired_delta_candidate_minus_baseline", {})
    interval = delta.get("bootstrap_ci_95", [])
    if (
        len(interval) != 2
        or float(delta.get("point_estimate", 0.0)) <= 0.0
        or float(interval[0]) <= 0.0
        or float(delta.get("probability_candidate_better", 0.0)) < 0.975
    ):
        raise RuntimeError("clDice did not clear the predeclared paired holdout promotion gate")

    selected = config.get("postprocessing", {})
    selected_epoch = int(selected.get("epoch", 0))
    if selected_epoch != int(config.get("epochs", -1)):
        raise RuntimeError("Expected the fixed final-epoch clDice checkpoint")
    dataset, split = native.load_dataset_split(data_root, REPO_ROOT / "data/splits")
    if split.sha256 != SPLIT_SHA256 or len(dataset.observation_names) != 707:
        raise RuntimeError("Competition training inventory or frozen split changed")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True

    started = time.perf_counter()
    fit_metadata_path: Path | None = None
    fit_metadata: dict[str, Any] | None = None
    if args.fit_all_training:
        training_names = list(dataset.observation_names)
        if len(training_names) != 707:
            raise RuntimeError(f"Final fit requires all 707 competition training groups, found {len(training_names)}")
        fit_metadata_path = fit_dir / "final-fit-report.json"
        training_names_sha256 = hashlib.sha256("\n".join(training_names).encode("utf-8")).hexdigest()
        fit_config = {
            "epochs": selected_epoch,
            "batch_size": 1,
            "workers": 0,
            "encoder": native.ENCODER_NAME,
            "crop_size": native.CROP_SIZE,
            "cldice_weight": float(config["cldice_weight"]),
            "tta_views": int(config["tta_views"]),
            "split_sha256": split.sha256,
            "training_observations": 707,
            "training_observation_names_sha256": training_names_sha256,
            "postprocessing": selected,
        }
        if fit_metadata_path.is_file():
            existing_fit = json.loads(fit_metadata_path.read_text(encoding="utf-8"))
            if existing_fit.get("config") != fit_config:
                raise FileExistsError(f"Final-fit output already has a different configuration: {fit_metadata_path}")
        images, masks, stats = native.build_training_cache(dataset, training_names, fit_dir)
        cache_index_path = native.cache_paths(fit_dir)[3]
        cache_names = json.loads(cache_index_path.read_text(encoding="utf-8"))["names"]
        if set(cache_names) != set(training_names):
            raise RuntimeError("Final-fit cache does not cover all 707 competition training groups")
        fit_checkpoint = fit_dir / "final-train" / "last-checkpoint.pt"
        model, training_summary = native.train_model(
            images,
            masks,
            stats,
            cache_names,
            list(range(len(cache_names))),
            epochs=selected_epoch,
            batch_size=1,
            workers=0,
            device=device,
            checkpoint_path=fit_checkpoint,
            fold_label="final",
            cldice_weight=float(config["cldice_weight"]),
        )
        checkpoint_path = native.epoch_checkpoint_path(fit_checkpoint, selected_epoch)
        fit_metadata = {
            "fit_scope": "all 707 competition-provided training observations after frozen-holdout model selection",
            "training_observations": len(cache_names),
            "training_observation_names_sha256": training_names_sha256,
            "holdout_observations": len(split.validation_names),
            "frozen_holdout_labels_used_in_final_fit": True,
            "split_sha256": split.sha256,
            "config": fit_config,
            "training": training_summary,
            "checkpoint_path": str(checkpoint_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "source_commit": native.git_commit(),
            "source_script_sha256": sha256_file(Path(__file__).resolve()),
            "source": native.git_info(REPO_ROOT),
            "device": native.capture_device(device),
            "package_versions": {
                **native.package_versions(),
                "timm": native.importlib.metadata.version("timm"),
            },
        }
        fit_metadata_path.write_text(json.dumps(fit_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        checkpoint_path = run_dir / "frozen-train-full" / f"checkpoint-epoch-{selected_epoch:02d}.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Selected clDice checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if int(checkpoint.get("epoch", -1)) != selected_epoch:
        raise RuntimeError("Checkpoint epoch differs from the OOF-selected epoch")
    expected_fold = "final" if args.fit_all_training else "full"
    if checkpoint.get("fold") != expected_fold:
        raise RuntimeError("Checkpoint training partition label differs from the selected fit scope")
    checkpoint_config = checkpoint.get("config", {})
    if (
        checkpoint_config.get("encoder") != native.ENCODER_NAME
        or int(checkpoint_config.get("crop_size", -1)) != native.CROP_SIZE
        or int(checkpoint_config.get("epochs", -1)) != selected_epoch
        or int(checkpoint_config.get("batch_size", -1)) != 1
        or float(checkpoint_config.get("cldice_weight", 0.0)) != 0.25
        or int(checkpoint_config.get("cldice_patch_size", -1)) != native.CLDICE_PATCH_SIZE
        or int(checkpoint_config.get("cldice_iterations", -1)) != native.CLDICE_ITERATIONS
    ):
        raise RuntimeError("Checkpoint training configuration differs from the validated clDice run")
    checkpoint_names = set(checkpoint.get("training_observations", []))
    expected_training_names = set(dataset.observation_names) if args.fit_all_training else set(split.training_names)
    if checkpoint_names != expected_training_names:
        expected_count = 707 if args.fit_all_training else 565
        raise RuntimeError(f"Checkpoint must contain exactly the selected {expected_count}-group training partition")
    if len(checkpoint_names) != (707 if args.fit_all_training else 565):
        raise RuntimeError("Checkpoint training group count is inconsistent with its fit scope")
    model = native.NativeResolutionUNet(pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = model.to(memory_format=torch.channels_last).eval()

    test_paths = sorted(dataset.test_image_dir.glob("*.jpeg"))
    if len(test_paths) != EXPECTED_TEST_IMAGES:
        raise RuntimeError(f"Expected {EXPECTED_TEST_IMAGES} supplied H-alpha test images, found {len(test_paths)}")
    predictions: dict[str, list[Any]] = {}
    for index, path in enumerate(test_paths, start=1):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (native.FULL_SIZE, native.FULL_SIZE):
            raise ValueError(f"Could not decode a 2048x2048 H-alpha test image: {path.name}")
        crop = native.centre_crop(image)
        stats = native.intensity_stats(crop)
        probability = native.predict_crop(
            model, crop, stats, device=device, tta_views=int(config["tta_views"])
        )
        predictions[path.name] = native.components_to_instances(
            probability,
            path.name,
            threshold=float(selected["threshold"]),
            min_area=int(selected["min_area"]),
            closing_kernel=int(selected["closing_kernel"]),
        )
        if index % 10 == 0 or index == len(test_paths):
            print(f"[test] {index}/{len(test_paths)} supplied images", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = output_dir / "submission.csv"
    _write_predictions(submission_path, predictions)
    rle_report = validate_submission(
        submission_path,
        dataset.test_image_dir,
        expected_test_images=EXPECTED_TEST_IMAGES,
    )
    rle_report_path = output_dir / "rle-validation.json"
    rle_report_path.write_text(json.dumps(rle_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    test_manifest = "\n".join(path.name for path in test_paths).encode("utf-8")
    report = {
        "candidate_id": (
            "native-unet-cldice-final-fit-train707"
            if args.fit_all_training
            else "native-unet-cldice-holdout-supported-train565"
        ),
        "candidate_role": (
            "final fit on all 707 competition training observations after fixed-holdout model selection"
            if args.fit_all_training
            else "preliminary test submission from the independent-holdout-supported checkpoint"
        ),
        "fit_scope": "all 707 competition training observations" if args.fit_all_training else "frozen 565-observation training partition",
        "training_observations": len(checkpoint_names),
        "holdout_observations": 142,
        "frozen_holdout_labels_used_in_final_fit": bool(args.fit_all_training),
        "split_sha256": SPLIT_SHA256,
        "model_epoch": selected_epoch,
        "cldice_weight": float(config["cldice_weight"]),
        "postprocessing": selected,
        "tta_views": int(config["tta_views"]),
        "paired_holdout_comparison": comparison["paired_delta_candidate_minus_baseline"],
        "test_images": len(test_paths),
        "test_image_names_sha256": hashlib.sha256(test_manifest).hexdigest(),
        "test_labels_read": False,
        "submission_path": str(submission_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "submission_sha256": sha256_file(submission_path),
        "submission_rows": int(rle_report["rows"]),
        "rle_validation_path": str(rle_report_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "rle_validation": rle_report,
        "checkpoint_path": str(checkpoint_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "final_fit_report_path": (
            str(fit_metadata_path.relative_to(REPO_ROOT)).replace("\\", "/")
            if fit_metadata_path is not None else None
        ),
        "final_fit_report_sha256": (
            sha256_file(fit_metadata_path) if fit_metadata_path is not None else None
        ),
        "final_fit_training": fit_metadata["training"] if fit_metadata is not None else None,
        "run_metadata_sha256": sha256_file(metadata_path),
        "holdout_metrics_sha256": sha256_file(metrics_path),
        "paired_comparison_sha256": sha256_file(comparison_path),
        "source_commit": native.git_commit(),
        "source_script_sha256": sha256_file(Path(__file__).resolve()),
        "source": native.git_info(REPO_ROOT),
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "device": native.capture_device(device),
        "package_versions": {
            **native.package_versions(),
            "timm": native.importlib.metadata.version("timm"),
        },
    }
    report_path = output_dir / "run-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "native_cldice_test_candidate_complete", "report": str(report_path), "sha256": report["submission_sha256"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
