# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Train RF-DETR-Seg on the frozen training partition and score the fixed holdout."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8", errors="replace")

from filament_baseline.data import load_dataset
from filament_baseline.predictor import PredictedInstance, encode_binary_mask
from filament_baseline.runtime import hardware_info, package_versions
from filament_baseline.scorer import pq_value, score_observation, score_validation
from filament_baseline.split import create_observation_split
from filament_baseline.visuals import write_comparison_overlays

EXPECTED_SPLIT_SHA256 = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"
SOURCE_NOTEBOOK = "https://www.kaggle.com/code/jirkaborovec/filamentsegm-eda-rf-detr-seg-ptl"
SOURCE_NOTEBOOK_SHA256 = "58039a063c2a021dcbdccdfd68d61097aaf74ac4eb793095f6bb54920f629a53"
MODEL_LICENSE = "Apache-2.0 (RF-DETR-Seg Small and rfdetr package)"
CONFIDENCE_GRID = tuple(round(float(value), 2) for value in np.arange(0.05, 0.55, 0.05))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def aggregate_observations(rows: list[Any]) -> dict[str, Any]:
    tp_ious: list[float] = []
    fp = 0
    fn = 0
    for row in rows:
        tp_ious.extend(row.tp_ious)
        fp += row.fp
        fn += row.fn
    return {"pq": pq_value(tp_ious, fp, fn), "tp": len(tp_ious), "tp_iou_sum": float(sum(tp_ious)), "fp": fp, "fn": fn}


def disjoint_predictions(
    masks: np.ndarray | None,
    scores: np.ndarray | None,
    file_name: str,
    confidence_threshold: float,
    image_size: int,
) -> list[PredictedInstance]:
    if masks is None or len(masks) == 0:
        return []
    masks = np.asarray(masks, dtype=bool)
    scores = np.ones(len(masks), dtype=np.float32) if scores is None else np.asarray(scores, dtype=np.float32)
    if masks.ndim != 3 or len(scores) != len(masks):
        raise ValueError(f"Unexpected RF-DETR outputs: masks={masks.shape}, scores={scores.shape}")
    order = np.argsort(-scores, kind="stable")
    occupied = np.zeros((image_size, image_size), dtype=bool)
    rows: list[PredictedInstance] = []
    stem = Path(file_name).stem
    instance_index = 0
    for index in order:
        confidence = float(scores[index])
        if confidence < confidence_threshold:
            continue
        mask = masks[index]
        if mask.shape != (image_size, image_size):
            resized = Image.fromarray(mask.astype(np.uint8)).resize(
                (image_size, image_size), resample=Image.Resampling.NEAREST
            )
            mask = np.asarray(resized, dtype=bool)
        exclusive = mask & ~occupied
        area = int(exclusive.sum())
        if area == 0:
            continue
        occupied |= exclusive
        instance_index += 1
        rows.append(
            PredictedInstance(
                filament_id=f"{stem}_{instance_index}",
                image_stem=stem,
                area=area,
                rle=encode_binary_mask(exclusive),
            )
        )
    return rows


def count_metadata(dataset: Any, names: tuple[str, ...] | list[str]) -> dict[str, int]:
    views = [view for name in names for view in dataset.views_by_observation[name]]
    return {
        "observations": len(names),
        "annotator_views": len(views),
        "annotations": sum(len(dataset.annotations_by_image_id[view.image_id]) for view in views),
    }


class RFDETRFilamentDataset(torch.utils.data.Dataset):
    """Map a source-image dataframe to RF-DETR samples, including on Windows workers."""

    def __init__(
        self,
        images: pd.DataFrame,
        annotations: pd.DataFrame,
        image_dir: Path,
        transforms: Any,
        polygon_converter: Any,
    ) -> None:
        self.images = images.reset_index(drop=True)
        self.annotations_by_id = dict(tuple(annotations.groupby("source_image_id")))
        self.image_dir = image_dir
        self.transforms = transforms
        self.polygon_converter = polygon_converter
        self.root = image_dir
        self.ids = self.images["source_image_id"].tolist()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
        record = self.images.iloc[index]
        with Image.open(self.image_dir / record.file_name) as raw_image:
            image = raw_image.convert("RGB")
        width, height = image.size
        image_annotations = self.annotations_by_id.get(record.source_image_id)
        if image_annotations is None or image_annotations.empty:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            masks = torch.zeros((0, height, width), dtype=torch.uint8)
            areas = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(np.stack(image_annotations["bbox"].to_numpy()), dtype=torch.float32)
            boxes[:, 2:] += boxes[:, :2]
            boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=width)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=height)
            masks = self.polygon_converter(
                image_annotations["segmentation"].tolist(), height, width
            )
            areas = torch.as_tensor(
                image_annotations["area"].to_numpy(copy=True), dtype=torch.float32
            )
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes, masks, areas = boxes[keep], masks[keep], areas[keep]
        count = boxes.shape[0]
        target = {
            "boxes": boxes,
            "labels": torch.zeros((count,), dtype=torch.int64),
            "image_id": torch.as_tensor([index]),
            "area": areas,
            "iscrowd": torch.zeros((count,), dtype=torch.int64),
            "masks": masks.bool(),
            "orig_size": torch.as_tensor([height, width]),
            "size": torch.as_tensor([height, width]),
        }
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target


def main() -> int:
    run_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--split-dir", type=Path, default=REPO_ROOT / "data" / "splits")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "model-experiments" / "rfdetr-fixedholdout-20261112")
    parser.add_argument("--checkpoint", type=Path, help="Score an existing trusted RF-DETR training checkpoint without fitting again.")
    parser.add_argument("--incumbent-observations", type=Path, default=REPO_ROOT / "artifacts" / "model-experiments" / "crossfold-medoid-unet-fusion-20261109" / "incumbent-per-observation.csv")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--mask-downsample-ratio", type=int, choices=(2, 4), default=4)
    parser.add_argument("--confidence-grid", default=",".join(map(str, CONFIDENCE_GRID)))
    parser.add_argument("--internal-validation-fraction", type=float, default=0.15)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    split_dir = args.split_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    allowed_roots = [REPO_ROOT.resolve()]
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.exists():
        allowed_roots.append(kaggle_working.resolve())
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        allowed_roots.append(kaggle_input.resolve())
    local_artifacts = (REPO_ROOT / "artifacts").resolve()
    remote_working = Path("/kaggle/working")
    checkpoint_input = args.checkpoint.expanduser().resolve() if args.checkpoint else None
    if checkpoint_input is not None:
        checkpoint_is_local = checkpoint_input.is_relative_to(local_artifacts)
        checkpoint_is_kaggle = remote_working.exists() and checkpoint_input.is_relative_to(remote_working.resolve())
        if not (checkpoint_is_local or checkpoint_is_kaggle) or not checkpoint_input.is_file():
            raise ValueError(f"Checkpoint must be an existing file under artifacts/ or /kaggle/working: {checkpoint_input}")
    if not output_dir.is_relative_to(local_artifacts) and not (
        remote_working.exists() and output_dir.is_relative_to(remote_working.resolve())
    ):
        raise ValueError(f"Run outputs must remain under the repository or /kaggle/working: {output_dir}")
    if not any(data_root.is_relative_to(root) for root in allowed_roots):
        raise ValueError(f"Competition data must remain under the repository or /kaggle/working: {data_root}")
    local_splits = (REPO_ROOT / "data" / "splits").resolve()
    if not split_dir.is_relative_to(local_splits) and not (
        remote_working.exists() and split_dir.is_relative_to(remote_working.resolve())
    ):
        raise ValueError(f"Split manifests must remain under data/splits or /kaggle/working: {split_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    if args.epochs < 1 or args.resolution < 384 or args.resolution % 24:
        raise ValueError("epochs must be positive and RF-DETR-Seg-Small resolution must be a multiple of 24 >= 384")
    thresholds = tuple(sorted({float(value) for value in args.confidence_grid.split(",")}))
    if not thresholds or any(value < 0 or value > 1 for value in thresholds):
        raise ValueError("Confidence thresholds must be in [0, 1]")

    output_dir.mkdir(parents=True, exist_ok=True)
    if remote_working.exists() and output_dir.is_relative_to(remote_working.resolve()):
        cache_root = remote_working.resolve() / ".cache" / "equinexus-rfdetr"
    else:
        cache_root = REPO_ROOT / "artifacts" / "cache" / "rfdetr"
    cache_roots = {
        "HF_HOME": cache_root / "huggingface",
        "MPLCONFIGDIR": cache_root / "matplotlib",
        "RF_HOME": cache_root / "roboflow",
        "TORCH_HOME": cache_root / "torch",
        "XDG_CACHE_HOME": cache_root / "xdg",
    }
    temporary_root = output_dir / ".tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    for name, path in cache_roots.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ["TEMP"] = str(temporary_root)
    os.environ["TMP"] = str(temporary_root)
    os.environ["TMPDIR"] = str(temporary_root)
    tempfile.tempdir = str(temporary_root)

    import cv2
    from pytorch_lightning.callbacks import LearningRateMonitor
    from rfdetr import RFDETRSegSmall
    from rfdetr.config import RFDETRSegSmallConfig, SegmentationTrainConfig
    from rfdetr.datasets.coco import convert_coco_poly_to_mask, make_coco_transforms_square_div_64
    from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer

    cv2.setNumThreads(0)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    dataset = load_dataset(data_root)
    split = create_observation_split(dataset, split_dir, seed=42, validation_fraction=0.2)
    if split.sha256 != EXPECTED_SPLIT_SHA256:
        raise RuntimeError(f"Frozen split changed: {split.sha256}")
    training_names = tuple(split.training_names)
    holdout_names = tuple(split.validation_names)
    if set(training_names).intersection(holdout_names):
        raise RuntimeError("Observation overlap in the frozen split")

    raw = json.loads(dataset.annotation_path.read_text(encoding="utf-8"))
    df_images = pd.DataFrame(raw["images"]).rename(columns={"id": "source_image_id"})
    df_annotations = pd.DataFrame(raw["annotations"])
    df_images["source_image_id"] = df_images["source_image_id"].astype(str)
    df_annotations["source_image_id"] = df_annotations["image_id"].astype(str)
    annotation_counts = df_annotations.groupby("source_image_id").size().rename("annotation_count")
    df_images = df_images.merge(annotation_counts, how="left", left_on="source_image_id", right_index=True)
    df_images["annotation_count"] = df_images["annotation_count"].fillna(0).astype(int)

    pool = df_images[df_images["file_name"].isin(training_names)].copy().reset_index(drop=True)
    if set(pool["file_name"].unique()) != set(training_names):
        raise RuntimeError("The competition-training pool does not cover every frozen training observation")
    inner_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=args.internal_validation_fraction,
        random_state=42,
    )
    inner_train_indices, inner_validation_indices = next(
        inner_splitter.split(pool, groups=pool["file_name"])
    )
    inner_train_views = pool.iloc[inner_train_indices].copy().reset_index(drop=True)
    inner_validation_views = pool.iloc[inner_validation_indices].copy().reset_index(drop=True)
    inner_train_names = tuple(sorted(inner_train_views["file_name"].unique()))
    inner_validation_names = tuple(sorted(inner_validation_views["file_name"].unique()))
    if set(inner_train_names).intersection(inner_validation_names) or set(inner_validation_names).intersection(holdout_names):
        raise RuntimeError("An observation crossed an inner or outer validation boundary")
    if set(inner_train_names) | set(inner_validation_names) != set(training_names):
        raise RuntimeError("The inner train/validation split does not cover the 565-observation training pool")

    class FixedInnerFoldDataModule(RFDETRDataModule):
        def __init__(self, model_config: Any, train_config: Any):
            super().__init__(model_config, train_config)
            # Use one deterministically chosen annotator view per training observation to avoid
            # identical pixels carrying conflicting instance targets. Keep every inner-val view.
            self._train_images = (
                inner_train_views.sort_values(
                    ["annotation_count", "source_image_id"], ascending=[False, True]
                )
                .drop_duplicates("file_name", keep="first")
                .sample(frac=1.0, random_state=42)
                .reset_index(drop=True)
            )
            self._validation_images = inner_validation_views.reset_index(drop=True)
            if set(self._train_images.file_name).intersection(self._validation_images.file_name):
                raise RuntimeError("Inner fit and validation observations overlap")
            self._image_dir = dataset.train_image_dir

        @property
        def class_names(self) -> list[str]:
            return ["filament"]

        @property
        def train_images(self) -> pd.DataFrame:
            return self._train_images

        @property
        def validation_images(self) -> pd.DataFrame:
            return self._validation_images

        def _build_dataset(self, image_set: str, images: pd.DataFrame):
            transforms = make_coco_transforms_square_div_64(
                image_set,
                self.model_config.resolution,
                multi_scale=self.train_config.multi_scale,
                expanded_scales=self.train_config.expanded_scales,
                skip_random_resize=not self.train_config.do_random_resize_via_padding,
                patch_size=self.model_config.patch_size,
                num_windows=self.model_config.num_windows,
                aug_config=self.train_config.aug_config,
                gpu_postprocess=False,
            )
            return RFDETRFilamentDataset(
                images,
                df_annotations,
                self._image_dir,
                transforms,
                convert_coco_poly_to_mask,
            )

        def setup(self, stage: str | None = None):
            if stage in (None, "fit"):
                if self._dataset_train is None:
                    self._dataset_train = self._build_dataset("train", self.train_images)
                if self._dataset_val is None:
                    self._dataset_val = self._build_dataset("val", self.validation_images)
            elif stage in {"validate", "predict"}:
                if self._dataset_val is None:
                    self._dataset_val = self._build_dataset("val", self.validation_images)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda:0")
        torch.cuda.set_device(device.index)
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type != "cpu":
        raise ValueError("This RF-DETR run supports CUDA or CPU")

    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_config = RFDETRSegSmallConfig(
        num_classes=1,
        device=str(device),
        resolution=args.resolution,
        pretrain_weights="rf-detr-seg-small.pt",
        mask_downsample_ratio=args.mask_downsample_ratio,
    )
    pretrained_path = Path(model_config.pretrain_weights)
    train_config = SegmentationTrainConfig(
        dataset_file="coco",
        dataset_dir=str(dataset.train_image_dir),
        output_dir=str(model_dir),
        class_names=["filament"],
        epochs=args.epochs,
        batch_size=1,
        grad_accum_steps=16,
        lr=1e-4,
        # This Dataset class is defined in main(); Windows spawn cannot pickle
        # local classes for worker processes. Linux Kaggle can use two workers.
        num_workers=2,
        seed=42,
        early_stopping=True,
        early_stopping_patience=max(2, args.epochs // 2),
        eval_interval=1,
        use_ema=True,
        run_test=False,
        tensorboard=True,
        progress_bar="tqdm",
        augmentation_backend="cpu",
        mask_dice_loss_coef=5.0,
    )
    datamodule = FixedInnerFoldDataModule(model_config, train_config)
    inner_training_views_after_deduplication = len(datamodule.train_images)
    if checkpoint_input is None:
        module = RFDETRModelModule(model_config, train_config)
        trainer = build_trainer(
            train_config,
            model_config,
            accelerator="gpu" if device.type == "cuda" else "cpu",
            fast_dev_run=False,
        )
        trainer.callbacks.append(LearningRateMonitor(logging_interval="step"))
        if device.type == "cuda":
            torch.cuda.empty_cache()
        training_started = time.perf_counter()
        trainer.fit(module, datamodule)
        training_seconds = time.perf_counter() - training_started
        checkpoint = model_dir / "checkpoint_best_total.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"RF-DETR did not write its best EMA/regular checkpoint: {checkpoint}")
    else:
        checkpoint = checkpoint_input
        training_seconds = 0.0
    del datamodule
    if checkpoint_input is None:
        del module, trainer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    checkpoint_hash = sha256(checkpoint)

    model = RFDETRSegSmall.from_checkpoint(str(checkpoint), device=str(device))
    if device.type == "cuda":
        model.optimize_for_inference(compile=False, dtype=torch.float16)

    def predict_raw(file_name: str) -> tuple[np.ndarray | None, np.ndarray | None]:
        with Image.open(dataset.train_image_dir / file_name) as raw_image:
            rgb_image = raw_image.convert("RGB")
        prediction = model.predict(rgb_image, threshold=min(thresholds))
        masks = None if prediction.mask is None else np.asarray(prediction.mask, dtype=bool)
        scores = None if prediction.confidence is None else np.asarray(prediction.confidence, dtype=np.float32)
        return masks, scores

    inner_started = time.perf_counter()
    threshold_counts: dict[float, list[Any]] = {threshold: [] for threshold in thresholds}
    for index, name in enumerate(inner_validation_names, start=1):
        masks, scores = predict_raw(name)
        for threshold in thresholds:
            predictions = disjoint_predictions(masks, scores, name, threshold, dataset.image_views[0].height)
            counts, _ = score_observation(dataset, name, predictions, iou_threshold=0.5, diagnostic_overlap_threshold=0.0)
            threshold_counts[threshold].append(counts)
        if index % 10 == 0 or index == len(inner_validation_names):
            print(f"inner validation inference: {index}/{len(inner_validation_names)} observations", flush=True)
    inner_seconds = time.perf_counter() - inner_started
    threshold_metrics = [
        {"confidence_threshold": threshold, **aggregate_observations(rows)}
        for threshold, rows in threshold_counts.items()
    ]
    selected = max(threshold_metrics, key=lambda row: (row["pq"], row["confidence_threshold"]))
    selected_threshold = float(selected["confidence_threshold"])
    (output_dir / "inner-threshold-sweep.json").write_text(
        json.dumps(threshold_metrics, indent=2) + "\n", encoding="utf-8"
    )
    del threshold_counts

    holdout_started = time.perf_counter()
    holdout_predictions: dict[str, list[PredictedInstance]] = {}
    serialized_predictions: dict[str, list[dict[str, Any]]] = {}
    for index, name in enumerate(sorted(holdout_names), start=1):
        masks, scores = predict_raw(name)
        predictions = disjoint_predictions(masks, scores, name, selected_threshold, dataset.image_views[0].height)
        holdout_predictions[name] = predictions
        serialized_predictions[name] = [
            {"filament_id": row.filament_id, "size": row.rle["size"], "counts": row.rle["counts"]}
            for row in predictions
        ]
        if index % 10 == 0 or index == len(holdout_names):
            print(f"fixed holdout inference: {index}/{len(holdout_names)} observations", flush=True)
    holdout_inference_seconds = time.perf_counter() - holdout_started

    metrics, observation_rows = score_validation(
        dataset,
        holdout_predictions,
        holdout_names,
        iou_threshold=0.5,
        diagnostic_overlap_threshold=0.0,
    )
    overlay_files = write_comparison_overlays(
        dataset,
        holdout_predictions,
        output_dir / "overlays",
        maximum=12,
    )
    observations = [
        {
            "file_name": row.file_name,
            "year": row.year,
            "site": row.site,
            "annotator_views": row.annotator_views,
            "gt_count": row.gt_count,
            "prediction_count": row.prediction_count,
            "tp": row.tp,
            "tp_iou_sum": row.tp_iou_sum,
            "fp": row.fp,
            "fn": row.fn,
            "pq": row.pq,
        }
        for row in observation_rows
    ]
    (output_dir / "holdout-predictions.json").write_text(
        json.dumps(serialized_predictions, separators=(",", ":")), encoding="utf-8"
    )
    pd.DataFrame(observations).to_csv(output_dir / "holdout-per-observation.csv", index=False)

    comparison = None
    incumbent_path = args.incumbent_observations.expanduser().resolve()
    if incumbent_path.is_file():
        from evaluate_component_selector_probability_blend_holdout import paired_bootstrap

        incumbent = pd.read_csv(incumbent_path)
        baseline_rows = [
            SimpleNamespace(file_name=row.file_name, tp_iou_sum=row.tp_iou_sum, tp=row.tp_edges, fp=row.fp, fn=row.fn)
            for row in incumbent.itertuples(index=False)
        ]
        candidate_rows = [
            SimpleNamespace(file_name=row["file_name"], tp_iou_sum=row["tp_iou_sum"], tp=row["tp"], fp=row["fp"], fn=row["fn"])
            for row in observations
        ]
        comparison = {
            "incumbent_per_observation_sha256": sha256(incumbent_path),
            "paired_bootstrap": paired_bootstrap(baseline_rows, candidate_rows, resamples=20_000, seed=42),
        }

    total_seconds = time.perf_counter() - run_started
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    metadata = {
        "run_name": output_dir.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_kernel": SOURCE_NOTEBOOK,
        "source_kernel_sha256": SOURCE_NOTEBOOK_SHA256,
        "model": "RF-DETR-Seg Small",
        "pretrained_initialization": "official RF-DETR-Seg Small COCO weights, downloaded from the public RF-DETR package",
        "model_and_package_license": MODEL_LICENSE,
        "source_training_method": "One most-annotated training view per observation; grouped internal validation; confidence chosen by exact official PQ on only the internal validation observations.",
        "training_data": "Competition-provided training annotations only",
        "test_ground_truth_used": False,
        "test_images_used": False,
        "fixed_holdout_used_for_training_or_threshold_selection": False,
        "fixed_holdout_split_sha256": split.sha256,
        "fixed_training_pool": count_metadata(dataset, training_names),
        "inner_training_observations": len(inner_train_names),
        "inner_training_views_before_annotator_deduplication": len(inner_train_views),
        "inner_training_views_after_annotator_deduplication": inner_training_views_after_deduplication,
        "inner_validation": count_metadata(dataset, inner_validation_names),
        "fixed_holdout": count_metadata(dataset, holdout_names),
        "inner_split_sha256": hashlib.sha256(
            json.dumps(
                {"train": inner_train_names, "validation": inner_validation_names},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "model_config": {
            "epochs_requested": args.epochs,
            "training_performed_in_this_run": checkpoint_input is None,
            "evaluation_checkpoint_source": str(checkpoint_input) if checkpoint_input else None,
            "resolution": model.model_config.resolution,
            "mask_downsample_ratio": model.model_config.mask_downsample_ratio,
            "batch_size": 1,
            "gradient_accumulation_steps": 16,
            "learning_rate": 1e-4,
            "use_ema": True,
            "confidence_grid": list(thresholds),
            "selected_confidence_threshold": selected_threshold,
            "selected_inner_validation_pq": selected["pq"],
            "prediction_masks_resolved_by_confidence_without_overlap": True,
        },
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_hash},
        "pretrained_initialization_checkpoint": {
            "path": str(pretrained_path),
            "sha256": sha256(pretrained_path),
        },
        "device": str(device),
        "hardware": hardware_info(),
        "package_versions": package_versions(),
        "training_seconds": training_seconds,
        "inner_validation_inference_and_selection_seconds": inner_seconds,
        "fixed_holdout_inference_seconds": holdout_inference_seconds,
        "total_seconds": total_seconds,
        "peak_gpu_memory_allocated_bytes": peak_memory,
        "workspace_cache_roots": {name: str(path) for name, path in cache_roots.items()},
        "workspace_temp_root": str(temporary_root),
        "inner_threshold_metrics": threshold_metrics,
        "metrics": metrics,
        "per_observation": observations,
        "fixed_holdout_overlay_files": overlay_files,
        "comparison_vs_incumbent": comparison,
        "runner_sha256": sha256(Path(__file__).resolve()),
        "source_commit": git_value("rev-parse", "HEAD"),
        "source_worktree_status": git_value("status", "--short"),
    }
    metadata_path = output_dir / "run-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "event": "complete",
        "fixed_holdout_pq": metrics["global"]["pq"],
        "tp": metrics["global"]["tp"],
        "fp": metrics["global"]["fp"],
        "fn": metrics["global"]["fn"],
        "selected_threshold": selected_threshold,
        "selected_inner_validation_pq": selected["pq"],
        "training_seconds": training_seconds,
        "fixed_holdout_inference_seconds": holdout_inference_seconds,
        "peak_gpu_memory_allocated_bytes": peak_memory,
        "comparison_vs_incumbent": comparison,
        "metadata": str(metadata_path),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
