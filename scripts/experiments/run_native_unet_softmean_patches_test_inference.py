# SPDX-License-Identifier: AGPL-3.0-only
"""Final-fit a promoted patch U-Net on all training groups and make a test submission."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments"))
from workspace_runtime import configure_workspace_runtime  # noqa: E402

configure_workspace_runtime(REPO_ROOT)

import cv2  # noqa: E402
import torch  # noqa: E402

import train_native_unet_softmean_patches as native  # noqa: E402
from filament_baseline.runner import _write_predictions  # noqa: E402
from validate_submission_rles import validate_submission  # noqa: E402


SPLIT_SHA256 = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"
TRAINER_SHA256 = "2c89dffea15071af547106d42298360ad0adc92e2c92f486b0f12f28b560d8c6"
INCUMBENT_OOF_PQ = 0.459116
INCUMBENT_HOLDOUT_PQ = 0.4546683971185862
EXPECTED_TEST_IMAGES = 180
DEFAULT_RUN_DIR = REPO_ROOT / "artifacts/model-experiments/native-unet-softmean-patches512-oof-20261113"
DEFAULT_FINAL_FIT_DIR = REPO_ROOT / "artifacts/model-experiments/native-unet-softmean-patches512-final-707-20261113"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts/submissions/native-unet-softmean-patches512-final-20261113"


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


def load_promotion_artifacts(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    paths = {
        "oof": run_dir / "oof-summary.json",
        "metadata": run_dir / "run-metadata-oof.json",
        "holdout": run_dir / "holdout-metrics.json",
        "bootstrap": run_dir / "paired-bootstrap-vs-family-vote-incumbent.json",
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Promotion gate artifact is missing ({label}): {path}")
    loaded = {label: json.loads(path.read_text(encoding="utf-8")) for label, path in paths.items()}
    oof, metadata, holdout, bootstrap = (loaded[key] for key in ("oof", "metadata", "holdout", "bootstrap"))
    if oof.get("split_sha256") != SPLIT_SHA256 or metadata.get("config", {}).get("split_sha256") != SPLIT_SHA256:
        raise RuntimeError("OOF artifacts do not match the frozen split")
    if (
        int(oof.get("training_observations", -1)) != 565
        or int(oof.get("annotator_views", -1)) != 922
        or int(metadata.get("dataset", {}).get("training_observations", -1)) != 565
        or int(metadata.get("dataset", {}).get("training_annotator_views", -1)) != 922
        or int(metadata.get("config", {}).get("epochs", -1)) != 40
        or int(metadata.get("config", {}).get("training_patch_size", -1)) != 512
    ):
        raise RuntimeError("OOF artifacts do not match the five-fold 565-observation patch experiment")
    if (
        oof.get("script_sha256") != TRAINER_SHA256
        or metadata.get("source_script_sha256") != TRAINER_SHA256
        or sha256_file(Path(native.__file__).resolve()) != TRAINER_SHA256
    ):
        raise RuntimeError("OOF summary, run metadata, and final-fit trainer source hashes do not agree")
    weights = metadata.get("pretrained_encoder_weights", {})
    if (
        weights.get("model_id") != "timm/efficientvit_b2.r288_in1k"
        or weights.get("revision") != "891530d0c5a400dd2d4549c81d433661b192abe1"
        or weights.get("sha256") != "f5726851b5c55fab49f8302a79c3a898814e11bc20df76cc23cab28ae15c7328"
        or weights.get("license") != "Apache-2.0"
    ):
        raise RuntimeError("OOF run metadata does not match the archived Apache-2.0 encoder weights")
    if (
        holdout.get("split_sha256") != SPLIT_SHA256
        or int(holdout.get("training_observations", -1)) != 565
        or int(holdout.get("holdout_observations", -1)) != 142
        or int(holdout.get("holdout_annotator_views", -1)) != 232
        or bootstrap.get("observation_count") != 142
    ):
        raise RuntimeError("Independent holdout artifacts do not match the frozen 142-group split")
    if bootstrap.get("resamples") != 20_000 or bootstrap.get("seed") != 42:
        raise RuntimeError("Holdout comparison does not use the predeclared paired bootstrap")
    selected = oof.get("selected_oof_parameters") or {}
    if float(selected.get("pq", -1.0)) <= INCUMBENT_OOF_PQ:
        raise RuntimeError("Patch U-Net did not clear the frozen grouped-OOF promotion gate")
    if holdout.get("postprocessing_selected_from_oof") != selected:
        raise RuntimeError("Holdout inference configuration differs from the OOF-selected configuration")
    holdout_pq = float(holdout["official_scorer_metrics"]["global"]["pq"])
    if holdout_pq <= INCUMBENT_HOLDOUT_PQ:
        raise RuntimeError("Patch U-Net did not improve the independent holdout point estimate")
    delta = bootstrap.get("paired_delta_candidate_minus_baseline", {})
    interval = delta.get("bootstrap_ci_95", [])
    if (
        abs(float(bootstrap.get("baseline", {}).get("pq", -1.0)) - INCUMBENT_HOLDOUT_PQ) > 1e-12
        or abs(float(bootstrap.get("candidate", {}).get("pq", -1.0)) - holdout_pq) > 1e-12
        or abs(float(delta.get("point_estimate", 0.0)) - (holdout_pq - INCUMBENT_HOLDOUT_PQ)) > 1e-12
    ):
        raise RuntimeError("Paired bootstrap counts do not reproduce the frozen holdout comparison")
    if (
        len(interval) != 2
        or float(delta.get("point_estimate", 0.0)) <= 0.0
        or float(interval[0]) <= 0.0
        or float(delta.get("probability_candidate_better", 0.0)) < 0.975
    ):
        raise RuntimeError("Patch U-Net did not clear the paired observation-bootstrap promotion gate")
    if holdout.get("test_labels_read") is not False:
        raise RuntimeError("Holdout evaluation does not certify exclusion of test labels")
    return oof, metadata, holdout, bootstrap


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--split-dir", type=Path, default=REPO_ROOT / "data/splits")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--fit-dir", type=Path, default=DEFAULT_FINAL_FIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    split_dir = args.split_dir.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    fit_dir = args.fit_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    require_within(data_root, REPO_ROOT / "data", "Competition data")
    require_within(split_dir, REPO_ROOT / "data" / "splits", "Split manifests")
    require_within(run_dir, REPO_ROOT / "artifacts/model-experiments", "OOF and holdout run")
    require_within(fit_dir, REPO_ROOT / "artifacts/model-experiments", "Final-fit output")
    require_within(output_dir, REPO_ROOT / "artifacts/submissions", "Submission output")
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)

    oof, oof_metadata, holdout, bootstrap = load_promotion_artifacts(run_dir)
    config = oof_metadata["config"]
    selected = oof["selected_oof_parameters"]
    selected_epoch = int(selected["epoch"])
    if selected_epoch not in native.selected_epoch_checkpoints(int(config["epochs"])):
        raise RuntimeError("Selected epoch has no reproducible checkpoint schedule")
    if float(config.get("cldice_weight", -1.0)) != 0.0 or int(config.get("tta_views", 0)) != 8:
        raise RuntimeError("Final fit does not match the promoted 40-epoch eight-view OOF experiment")

    dataset, split = native.load_dataset_split(data_root, split_dir)
    if split.sha256 != SPLIT_SHA256 or len(dataset.observation_names) != 707:
        raise RuntimeError("Competition training inventory or frozen split changed")
    training_names = sorted(dataset.observation_names)
    if len(training_names) != 707 or set(split.training_names) | set(split.validation_names) != set(training_names):
        raise RuntimeError("Final fit must include exactly all 707 competition training observations")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
    cv2.setNumThreads(0)

    output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = output_dir / "submission.csv"
    report_path = output_dir / "run-report.json"
    if submission_path.exists() or report_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing final candidate under {output_dir}")

    started = time.perf_counter()
    images, masks, stats = native.build_training_cache(dataset, training_names, fit_dir)
    cache_index = json.loads(native.cache_paths(fit_dir)[3].read_text(encoding="utf-8"))
    cache_names = cache_index["names"]
    if cache_names != training_names or cache_index.get("split_sha256") != SPLIT_SHA256:
        raise RuntimeError("Final-fit cache does not cover the exact competition training inventory")
    checkpoint_path = fit_dir / "final-train" / "last-checkpoint.pt"
    model, training = native.train_model(
        images,
        masks,
        stats,
        cache_names,
        list(range(len(cache_names))),
        epochs=selected_epoch,
        batch_size=1,
        workers=0,
        device=device,
        checkpoint_path=checkpoint_path,
        fold_label="final",
        cldice_weight=float(config["cldice_weight"]),
    )
    selected_checkpoint = native.epoch_checkpoint_path(checkpoint_path, selected_epoch)
    if not selected_checkpoint.is_file():
        raise FileNotFoundError(f"Final-fit checkpoint is missing: {selected_checkpoint}")
    checkpoint = torch.load(selected_checkpoint, map_location="cpu", weights_only=True)
    if (
        int(checkpoint.get("epoch", -1)) != selected_epoch
        or checkpoint.get("fold") != "final"
        or set(checkpoint.get("training_observations", [])) != set(training_names)
    ):
        raise RuntimeError("Final-fit checkpoint does not cover the selected epoch and all 707 training observations")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    test_paths = sorted(dataset.test_image_dir.glob("*.jpeg"))
    if len(test_paths) != EXPECTED_TEST_IMAGES:
        raise RuntimeError(f"Expected {EXPECTED_TEST_IMAGES} supplied H-alpha test images, found {len(test_paths)}")
    predictions = {}
    for position, image_path in enumerate(test_paths, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (native.FULL_SIZE, native.FULL_SIZE):
            raise ValueError(f"Could not decode a 2048x2048 H-alpha test image: {image_path.name}")
        crop = native.centre_crop(image)
        image_stats = native.intensity_stats(crop)
        probability = native.predict_crop(
            model,
            crop,
            image_stats,
            device=device,
            tta_views=int(config["tta_views"]),
        )
        predictions[image_path.name] = native.components_to_instances(
            probability,
            image_path.name,
            threshold=float(selected["threshold"]),
            min_area=int(selected["min_area"]),
            closing_kernel=int(selected["closing_kernel"]),
        )
        if position % 10 == 0 or position == len(test_paths):
            print(f"[test] {position}/{len(test_paths)} supplied images", flush=True)

    _write_predictions(submission_path, predictions)
    rle_report = validate_submission(
        submission_path,
        dataset.test_image_dir,
        expected_test_images=EXPECTED_TEST_IMAGES,
    )
    if (
        rle_report.get("processed_test_images") != EXPECTED_TEST_IMAGES
        or rle_report.get("rle_exact_roundtrips") != rle_report.get("rows")
        or rle_report.get("instances_overlap_pixels") != 0
        or rle_report.get("unique_ids") is not True
        or rle_report.get("all_masks_nonempty") is not True
    ):
        raise RuntimeError("Final patch-U-Net submission failed full test-image/RLE validation")

    test_manifest = "\n".join(path.name for path in test_paths).encode("utf-8")
    final_fit_report = {
        "fit_scope": "all 707 competition training observations after fixed-holdout model selection",
        "training_observations": len(cache_names),
        "training_observation_names_sha256": hashlib.sha256("\n".join(cache_names).encode("utf-8")).hexdigest(),
        "frozen_holdout_labels_used_in_final_fit": True,
        "test_labels_read": False,
        "split_sha256": split.sha256,
        "selected_oof_parameters": selected,
        "training": training,
        "pretrained_encoder_weights": oof_metadata.get("pretrained_encoder_weights"),
        "checkpoint_path": str(selected_checkpoint.relative_to(REPO_ROOT)).replace("\\", "/"),
        "checkpoint_sha256": sha256_file(selected_checkpoint),
        "source_commit": native.git_commit(),
        "source_script_sha256": sha256_file(Path(__file__).resolve()),
        "device": native.capture_device(device),
        "package_versions": {**native.package_versions(), "timm": native.importlib.metadata.version("timm")},
    }
    fit_report_path = fit_dir / "final-fit-report.json"
    fit_report_path.write_text(json.dumps(final_fit_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = {
        "candidate_id": "native-unet-softmean-patches512-final-fit-train707",
        "candidate_role": "final fit after frozen grouped OOF and independent holdout promotion gates",
        "split_sha256": split.sha256,
        "oof_summary_sha256": sha256_file(run_dir / "oof-summary.json"),
        "oof_metadata_sha256": sha256_file(run_dir / "run-metadata-oof.json"),
        "holdout_metrics_sha256": sha256_file(run_dir / "holdout-metrics.json"),
        "paired_bootstrap_sha256": sha256_file(run_dir / "paired-bootstrap-vs-family-vote-incumbent.json"),
        "oof_pq": float(selected["pq"]),
        "holdout_pq": float(holdout["official_scorer_metrics"]["global"]["pq"]),
        "paired_holdout_delta": bootstrap["paired_delta_candidate_minus_baseline"],
        "fit_scope": final_fit_report["fit_scope"],
        "training_observations": len(cache_names),
        "frozen_holdout_labels_used_in_final_fit": True,
        "test_labels_read": False,
        "postprocessing": selected,
        "tta_views": int(config["tta_views"]),
        "test_images": len(test_paths),
        "test_image_names_sha256": hashlib.sha256(test_manifest).hexdigest(),
        "submission_path": str(submission_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "submission_sha256": sha256_file(submission_path),
        "submission_rows": int(rle_report["rows"]),
        "rle_validation_path": str((output_dir / "rle-validation.json").relative_to(REPO_ROOT)).replace("\\", "/"),
        "rle_validation": rle_report,
        "final_fit_report_path": str(fit_report_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "final_fit_report_sha256": sha256_file(fit_report_path),
        "source_commit": native.git_commit(),
        "source_script_sha256": sha256_file(Path(__file__).resolve()),
        "source": native.git_info(REPO_ROOT),
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "device": native.capture_device(device),
        "package_versions": {**native.package_versions(), "timm": native.importlib.metadata.version("timm")},
    }
    (output_dir / "rle-validation.json").write_text(json.dumps(rle_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "patch_unet_test_candidate_complete", "report": str(report_path), "sha256": report["submission_sha256"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
