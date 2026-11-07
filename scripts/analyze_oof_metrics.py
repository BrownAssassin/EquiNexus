from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, int | float]:
    tp = sum(int(row["tp_edges"]) for row in rows)
    fp = sum(int(row["fp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    tp_iou_sum = sum(float(row["tp_iou_sum"]) for row in rows)
    denominator = tp + 0.5 * fp + 0.5 * fn
    return {
        "observation_count": len(rows),
        "annotator_view_count": sum(int(row["annotator_views"]) for row in rows),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tp_iou_sum": tp_iou_sum,
        "pq": tp_iou_sum / denominator if denominator else 0.0,
    }


def _merge_histograms(histograms: list[dict[str, Any]]) -> dict[str, Any]:
    if not histograms:
        return {"count": 0, "mean": 0.0, "bins": [], "counts": [], "probabilities": []}
    bins = histograms[0]["bins"]
    if any(item["bins"] != bins for item in histograms):
        raise ValueError("OOF folds use inconsistent histogram bins")
    counts = [
        sum(int(item["counts"][index]) for item in histograms)
        for index in range(len(histograms[0]["counts"]))
    ]
    total = sum(int(item["count"]) for item in histograms)
    mean = (
        sum(float(item["mean"]) * int(item["count"]) for item in histograms) / total
        if total
        else 0.0
    )
    in_range = sum(counts)
    return {
        "count": total,
        "mean": mean,
        "bins": bins,
        "counts": counts,
        "probabilities": [value / in_range for value in counts] if in_range else [0.0] * len(counts),
    }


def _merge_degree_counts(
    fold_metrics: list[dict[str, Any]], field_name: str
) -> dict[str, int]:
    aggregate: dict[str, int] = defaultdict(int)
    for metrics in fold_metrics:
        for degree, count in metrics["one_to_many_many_to_one"][field_name].items():
            aggregate[str(degree)] += int(count)
    return dict(sorted(aggregate.items(), key=lambda item: int(item[0])))


def analyze(run_root: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    metrics_path = run_root / "metrics.json"
    observations_path = run_root / "oof-per-observation.csv"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    with observations_path.open("r", encoding="utf-8", newline="") as stream:
        observations = list(csv.DictReader(stream))
    if len(observations) != int(metrics["splits"]["oof_observations"]):
        raise ValueError("OOF observation-score coverage does not match the run metrics")
    names = [row["file_name"] for row in observations]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate observation rows found in OOF scores")

    fold_metrics = [
        json.loads((run_root / f"fold-{index}" / "metrics.json").read_text(encoding="utf-8"))["metrics"]
        for index in range(int(metrics["splits"]["fold_count"]))
    ]
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        "by_year": defaultdict(list),
        "by_site": defaultdict(list),
    }
    for row in observations:
        for label, field in ((row["year"], "by_year"), (row["site"], "by_site")):
            grouped[field][label].append(row)

    result = {
        "status": "complete",
        "run_name": metrics["run_name"],
        "oof_observation_count": len(observations),
        "frozen_holdout_sha256": metrics["splits"]["frozen_holdout_sha256"],
        "oof_manifest_sha256": metrics["splits"]["oof_manifest_sha256"],
        "source_metrics_sha256": _sha256(metrics_path),
        "aggregate": {
            "global": _aggregate(observations),
            "by_year": {
                label: _aggregate(rows)
                for label, rows in sorted(grouped["by_year"].items())
            },
            "by_site": {
                label: _aggregate(rows)
                for label, rows in sorted(grouped["by_site"].items())
            },
            "iou_positive_pairs": _merge_histograms(
                [item["iou_positive_pairs"] for item in fold_metrics]
            ),
            "dice_positive_pairs": _merge_histograms(
                [item["dice_positive_pairs"] for item in fold_metrics]
            ),
            "one_to_many_many_to_one": {
                "threshold": fold_metrics[0]["one_to_many_many_to_one"]["threshold"],
                "gt_match_degree_counts": _merge_degree_counts(
                    fold_metrics, "gt_match_degree_counts"
                ),
                "prediction_match_degree_counts": _merge_degree_counts(
                    fold_metrics, "prediction_match_degree_counts"
                ),
            },
        },
    }
    output_path = run_root / "aggregate-analysis.json"
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate site, year, overlap, and match-degree metrics across OOF folds."
    )
    parser.add_argument("run_root", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(analyze(arguments.run_root), indent=2, sort_keys=True))
