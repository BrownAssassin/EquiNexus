# SPDX-License-Identifier: AGPL-3.0-only
"""Paired observation bootstrap for the clDice screen against the best holdout."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPLIT_SHA256 = "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"
RESAMPLES = 20_000
SEED = 42
BASELINE = ROOT / "artifacts/model-experiments/public-unet-yolo-family-vote-holdout-tta8-conf025-20261111/run-metadata.json"
CANDIDATE_RUN = ROOT / "artifacts/model-experiments/native-unet-cldice-holdout-20261112/run-metadata.json"
CANDIDATE_METRICS = ROOT / "artifacts/model-experiments/native-unet-cldice-holdout-20261112/holdout-metrics.json"
CANDIDATE_ROWS = ROOT / "artifacts/model-experiments/native-unet-cldice-holdout-20261112/holdout-per-observation.csv"
OUTPUT = ROOT / "artifacts/model-experiments/native-unet-cldice-holdout-20261112/paired-bootstrap-vs-best-holdout.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_candidate(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for row in rows:
        name = row["file_name"]
        if name in result:
            raise ValueError(f"Duplicate candidate observation: {name}")
        result[name] = {
            "tp_iou_sum": float(row["tp_iou_sum"]),
            "tp": float(row["tp"]),
            "fp": float(row["fp"]),
            "fn": float(row["fn"]),
        }
    return result


def pq(counts: np.ndarray) -> np.ndarray:
    totals = counts.sum(axis=-2)
    denominator = totals[..., 1] + 0.5 * totals[..., 2] + 0.5 * totals[..., 3]
    return np.divide(totals[..., 0], denominator, out=np.zeros_like(denominator), where=denominator > 0)


def main() -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    candidate = json.loads(CANDIDATE_RUN.read_text(encoding="utf-8"))
    candidate_metrics = json.loads(CANDIDATE_METRICS.read_text(encoding="utf-8"))
    if candidate.get("config", {}).get("split_sha256") != SPLIT_SHA256:
        raise ValueError("Candidate metadata does not use the frozen split")
    if candidate.get("config", {}).get("training_observations") != 565:
        raise ValueError("Candidate must train only on the frozen 565-observation training partition")
    if candidate.get("config", {}).get("holdout_observations") != 142:
        raise ValueError("Candidate metadata has the wrong fixed holdout size")
    if baseline.get("fixed_holdout_split_sha256") != SPLIT_SHA256:
        raise ValueError("Incumbent metadata does not use the frozen split")
    if baseline.get("holdout_used_for_training") is not False:
        raise ValueError("Incumbent holdout was used for training")
    if baseline.get("holdout_used_for_model_or_threshold_selection") is not False:
        raise ValueError("Incumbent holdout was used for selection")
    if candidate.get("config", {}).get("holdout_threshold_tuning") is not False:
        raise ValueError("Candidate tuned thresholds on the holdout")
    if (
        candidate_metrics.get("split_sha256") != SPLIT_SHA256
        or candidate_metrics.get("training_observations") != 565
        or candidate_metrics.get("holdout_observations") != 142
        or candidate_metrics.get("test_labels_read") is not False
    ):
        raise ValueError("Completed candidate metrics do not certify the fixed no-test-label protocol")

    base_rows = {row["file_name"]: row for row in baseline["per_observation"]}
    cand_rows = read_candidate(CANDIDATE_ROWS)
    if len(base_rows) != 142 or set(base_rows) != set(cand_rows):
        raise ValueError("Candidate and incumbent must contain the same 142 holdout observations")

    names = sorted(base_rows)
    base = np.asarray(
        [[base_rows[n][key] for key in ("tp_iou_sum", "tp", "fp", "fn")] for n in names],
        dtype=np.float64,
    )
    cand = np.asarray(
        [[cand_rows[n][key] for key in ("tp_iou_sum", "tp", "fp", "fn")] for n in names],
        dtype=np.float64,
    )
    indices = np.random.default_rng(SEED).integers(0, len(names), size=(RESAMPLES, len(names)))
    delta = pq(cand[indices]) - pq(base[indices])
    point_base = float(pq(base)[()])
    point_candidate = float(pq(cand)[()])
    official_base = float(baseline["metrics"]["global"]["pq"])
    official_candidate = float(candidate_metrics["official_scorer_metrics"]["global"]["pq"])
    if abs(point_base - official_base) > 1e-12:
        raise ValueError("Incumbent per-observation scores do not match its official PQ")
    if abs(point_candidate - official_candidate) > 1e-12:
        raise ValueError("Candidate per-observation scores do not match its official PQ")
    result = {
        "split_sha256": SPLIT_SHA256,
        "resampling_unit": "original observation; annotator-view scorer counts are grouped",
        "resamples": RESAMPLES,
        "seed": SEED,
        "observation_count": len(names),
        "baseline": {
            "path": str(BASELINE.relative_to(ROOT)),
            "sha256": sha256(BASELINE),
            "pq": point_base,
            "holdout_used_for_training": False,
            "holdout_used_for_model_or_threshold_selection": False,
        },
        "candidate": {
            "path": str(CANDIDATE_RUN.relative_to(ROOT)),
            "sha256": sha256(CANDIDATE_RUN),
            "metrics_path": str(CANDIDATE_METRICS.relative_to(ROOT)),
            "metrics_sha256": sha256(CANDIDATE_METRICS),
            "per_observation_path": str(CANDIDATE_ROWS.relative_to(ROOT)),
            "per_observation_sha256": sha256(CANDIDATE_ROWS),
            "pq": point_candidate,
            "holdout_threshold_tuning": False,
            "epochs": candidate["config"]["epochs"],
            "cldice_weight": candidate["config"]["cldice_weight"],
        },
        "paired_delta_candidate_minus_baseline": {
            "point_estimate": point_candidate - point_base,
            "bootstrap_ci_95": [float(value) for value in np.quantile(delta, [0.025, 0.975])],
            "probability_candidate_better": float(np.mean(delta > 0)),
        },
        "bootstrap_implementation": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(Path(__file__).resolve()),
        },
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["paired_delta_candidate_minus_baseline"], sort_keys=True))


if __name__ == "__main__":
    main()
