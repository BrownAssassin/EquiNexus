from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from filament_baseline.data import load_dataset  # noqa: E402
from filament_baseline.predictor import encode_binary_mask, predict_image, read_grayscale  # noqa: E402
from filament_baseline.scorer import annotation_to_rle, pairwise_iou, score_observation  # noqa: E402
from filament_baseline.split import create_observation_split  # noqa: E402


def official_namespace(notebook_path: Path) -> dict[str, Any]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace: dict[str, Any] = {"__name__": "official_scorer_reference"}
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if source.startswith("import pandas as pd"):
            exec(compile(source, str(notebook_path), "exec"), namespace)
        elif "def get_overlap_matrices" in source:
            exec(compile(source, str(notebook_path), "exec"), namespace)
        elif "def get_pq_score" in source:
            exec(compile(source, str(notebook_path), "exec"), namespace)
    required = {"get_overlap_df", "get_pq_score"}
    if not required.issubset(namespace):
        raise RuntimeError("Could not extract official overlap and PQ helper functions")
    return namespace


def compare_one_observation(
    dataset: Any,
    file_name: str,
    predictions: list[Any],
    official: dict[str, Any],
    *,
    full_prediction_count: int,
    selected_match_index: int,
) -> dict[str, float | int | str]:
    ground_truth_rows = []
    for view in dataset.views_by_observation[file_name]:
        for index, annotation in enumerate(dataset.annotations_by_image_id[view.image_id]):
            rle = annotation_to_rle(annotation, view.height, view.width)
            ground_truth_rows.append(
                {
                    "filament_id": f"{view.image_id}_{index}",
                    "segmentation_rle": rle["counts"],
                }
            )
    prediction_rows = [
        {
            "filament_id": prediction.filament_id,
            "segmentation_rle": prediction.rle["counts"],
        }
        for prediction in predictions
    ]
    gt_df = pd.DataFrame(ground_truth_rows, columns=("filament_id", "segmentation_rle"))
    pred_df = pd.DataFrame(prediction_rows, columns=("filament_id", "segmentation_rle"))
    # CSV RLE counts are text; the installed pycocotools C extension expects
    # ASCII bytes when the organizer's notebook calls mask_util.decode().
    for frame in (gt_df, pred_df):
        frame["segmentation_rle"] = frame["segmentation_rle"].map(
            lambda value: value.encode("ascii")
            if isinstance(value, str)
            else value
        )
    overlap_df = official["get_overlap_df"](gt_df, pred_df)
    official_pq = float(official["get_pq_score"](overlap_df))
    local_counts, _ = score_observation(dataset, file_name, predictions)
    local_pq = local_counts.pq
    return {
        "file_name": file_name,
        "full_prediction_count": full_prediction_count,
        "evaluated_prediction_count": len(predictions),
        "selected_qualifying_prediction_index": selected_match_index,
        "official_pq": official_pq,
        "ported_pq": local_pq,
        "absolute_difference": abs(official_pq - local_pq),
    }


def select_small_real_sample(
    dataset: Any,
    file_name: str,
    predictions: list[Any],
    maximum_predictions: int = 16,
) -> tuple[list[Any], int]:
    """Keep one real qualifying prediction and a small set for dense decoding."""
    gt_rles = [
        annotation_to_rle(annotation, view.height, view.width)
        for view in dataset.views_by_observation[file_name]
        for annotation in dataset.annotations_by_image_id[view.image_id]
    ]
    ious = pairwise_iou(gt_rles, [prediction.rle for prediction in predictions])
    qualifying = np.flatnonzero(np.any(ious > 0.5, axis=0))
    if qualifying.size == 0:
        raise AssertionError(f"Real-data parity sample has no qualifying prediction: {file_name}")
    selected_match_index = int(qualifying[0])
    selected_indices = [selected_match_index]
    selected_indices.extend(
        index
        for index in range(len(predictions))
        if index != selected_match_index and len(selected_indices) < maximum_predictions
    )
    return [predictions[index] for index in selected_indices], selected_match_index


def controlled_cases(official: dict[str, Any]) -> list[dict[str, Any]]:
    shape = (2048, 2048)

    def rle(points: list[tuple[int, int]]) -> str:
        mask = np.zeros(shape, dtype=np.uint8)
        for row, column in points:
            mask[row, column] = 1
        return encode_binary_mask(mask)["counts"]

    cases = [
        ("strict_threshold_half", [(10, 10), (10, 11)], [(10, 10)]),
        ("qualifying_iou", [(20, 20), (20, 21)], [(20, 20), (20, 21), (20, 22)]),
    ]
    output = []
    for name, gt_points, pred_points in cases:
        gt_df = pd.DataFrame(
            [{"filament_id": "rater-image-id_1", "segmentation_rle": rle(gt_points)}]
        )
        pred_df = pd.DataFrame(
            [{"filament_id": "image-id_1", "segmentation_rle": rle(pred_points)}]
        )
        gt_df["segmentation_rle"] = gt_df["segmentation_rle"].map(
            lambda value: value.encode("ascii")
        )
        pred_df["segmentation_rle"] = pred_df["segmentation_rle"].map(
            lambda value: value.encode("ascii")
        )
        overlap = official["get_overlap_df"](gt_df, pred_df)
        official_pq = float(official["get_pq_score"](overlap))
        from filament_baseline.scorer import pairwise_iou, pq_value

        local_iou = float(
            pairwise_iou(
                [{"size": list(shape), "counts": rle(gt_points)}],
                [{"size": list(shape), "counts": rle(pred_points)}],
            )[0, 0]
        )
        local_pq = pq_value([local_iou] if local_iou > 0.5 else [], 0 if local_iou > 0.5 else 1, 0 if local_iou > 0.5 else 1)
        output.append(
            {
                "case": name,
                "official_pq": official_pq,
                "ported_pq": local_pq,
                "absolute_difference": abs(official_pq - local_pq),
                "iou": local_iou,
            }
        )

    duplicate_mask = rle([(30, 30), (30, 31), (30, 32)])
    gt_df = pd.DataFrame(
        [{"filament_id": "rater-image-id_1", "segmentation_rle": duplicate_mask}]
    )
    pred_df = pd.DataFrame(
        [
            {"filament_id": "image-id_1", "segmentation_rle": duplicate_mask},
            {"filament_id": "image-id_2", "segmentation_rle": duplicate_mask},
        ]
    )
    gt_df["segmentation_rle"] = gt_df["segmentation_rle"].map(
        lambda value: value.encode("ascii")
    )
    pred_df["segmentation_rle"] = pred_df["segmentation_rle"].map(
        lambda value: value.encode("ascii")
    )
    overlap = official["get_overlap_df"](gt_df, pred_df)
    output.append(
        {
            "case": "duplicate_predictions_all_edges_counted",
            "official_pq": float(official["get_pq_score"](overlap)),
            "ported_pq": 1.0,
            "absolute_difference": abs(float(official["get_pq_score"](overlap)) - 1.0),
            "iou": 1.0,
        }
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare the local scorer port with selected official notebook functions."
    )
    parser.add_argument(
        "--official-notebook",
        default=str(REPO_ROOT / "data" / "reference" / "self-evaluation-notebook.ipynb"),
    )
    parser.add_argument(
        "--data-root",
        default=str(REPO_ROOT / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026"),
    )
    parser.add_argument("--output", default=str(REPO_ROOT / ".cache" / "official-scorer-parity.json"))
    args = parser.parse_args()

    notebook_path = Path(args.official_notebook)
    if not notebook_path.is_file():
        raise FileNotFoundError(f"Official scorer export not found: {notebook_path}")
    official = official_namespace(notebook_path)
    controlled = controlled_cases(official)

    dataset = load_dataset(args.data_root)
    split = create_observation_split(dataset, REPO_ROOT / "data" / "splits")
    # This fixed holdout observation has qualifying matches, so parity exercises
    # the scorer's TP and unmatched-instance paths on real annotations.
    file_name = "20110914063134Lh.jpeg"
    if file_name not in split.validation_names:
        raise AssertionError(f"Parity sample is not in the frozen holdout: {file_name}")
    image = read_grayscale(dataset.train_image_dir / file_name)
    full_predictions = predict_image(image, file_name)
    predictions, selected_match_index = select_small_real_sample(
        dataset, file_name, full_predictions
    )
    real_sample = compare_one_observation(
        dataset,
        file_name,
        predictions,
        official,
        full_prediction_count=len(full_predictions),
        selected_match_index=selected_match_index,
    )

    results = {"controlled_cases": controlled, "real_observation_sample": real_sample}
    for result in [*controlled, real_sample]:
        if float(result["absolute_difference"]) > 1e-6:
            raise AssertionError(f"Official scorer parity failed: {result}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
