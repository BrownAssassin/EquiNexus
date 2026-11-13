from __future__ import annotations

from bisect import bisect_left
from datetime import datetime, timezone
import hashlib
import re
from pathlib import Path
from typing import Iterable


GONG_EXTERNAL_DATA_RULING = (
    "https://www.kaggle.com/competitions/filament-segmentation-2026/"
    "discussion/748045#3536644"
)
EXTERNAL_TARGET_POLICY_VERSION = "gong-proxy-targets-2026-11-12-v1"
EXTERNAL_TARGET_TIME_BUFFER_SECONDS = 3600
_OBSERVATION_ID = re.compile(r"^\d{14}[BCLMTU]h$")


def _normalize_ids(values: Iterable[str]) -> list[str]:
    normalized = sorted({Path(str(value)).stem for value in values})
    for value in normalized:
        if not _OBSERVATION_ID.fullmatch(value):
            raise ValueError(f"Invalid GONG observation identifier: {value!r}")
        datetime.strptime(value[:14], "%Y%m%d%H%M%S")
    return normalized


def _digest_ids(values: Iterable[str]) -> str:
    payload = "\n".join(sorted(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _timestamp(value: str) -> float:
    return datetime.strptime(value[:14], "%Y%m%d%H%M%S").replace(
        tzinfo=timezone.utc
    ).timestamp()


def _within_buffer(value: str, protected_timestamps: list[float]) -> bool:
    timestamp = _timestamp(value)
    index = bisect_left(
        protected_timestamps, timestamp - EXTERNAL_TARGET_TIME_BUFFER_SECONDS
    )
    return index < len(protected_timestamps) and protected_timestamps[index] <= (
        timestamp + EXTERNAL_TARGET_TIME_BUFFER_SECONDS
    )


def audit_external_target_ids(
    candidate_ids: Iterable[str],
    training_ids: Iterable[str],
    validation_ids: Iterable[str],
    test_ids: Iterable[str],
) -> dict[str, object]:
    """Exclude external GONG targets near test/development observations.

    The temporal buffers apply across all GONG sites. Development calendar
    dates are held out as well, matching the cited organizer ruling's
    interpretation in the public GONG-neighbour workflow.
    """
    candidates = _normalize_ids(candidate_ids)
    training = set(_normalize_ids(training_ids))
    validation = _normalize_ids(validation_ids)
    test = _normalize_ids(test_ids)
    if not training:
        raise ValueError("The frozen training observation set must be non-empty")
    if len(test) != 180:
        raise ValueError(f"Expected all 180 supplied test timestamps, found {len(test)}")

    validation_dates = {value[:8] for value in validation}
    validation_timestamps = sorted(_timestamp(value) for value in validation)
    test_timestamps = sorted(_timestamp(value) for value in test)
    reasons: dict[str, list[str]] = {
        "not_in_frozen_training_partition": [],
        "frozen_validation_calendar_date": [],
        "within_one_hour_of_test_across_sites": [],
        "within_one_hour_of_validation_across_sites": [],
    }

    eligible: list[str] = []
    for candidate in candidates:
        matched: list[str] = []
        if candidate not in training:
            matched.append("not_in_frozen_training_partition")
        if candidate[:8] in validation_dates:
            matched.append("frozen_validation_calendar_date")
        if _within_buffer(candidate, test_timestamps):
            matched.append("within_one_hour_of_test_across_sites")
        if _within_buffer(candidate, validation_timestamps):
            matched.append("within_one_hour_of_validation_across_sites")
        if matched:
            for reason in matched:
                reasons[reason].append(candidate)
        else:
            eligible.append(candidate)

    return {
        "authority_url": GONG_EXTERNAL_DATA_RULING,
        "policy_version": EXTERNAL_TARGET_POLICY_VERSION,
        "time_buffer_seconds": EXTERNAL_TARGET_TIME_BUFFER_SECONDS,
        "time_buffer_applies_across_sites": True,
        "development_calendar_dates_excluded": True,
        "candidate_target_ids": candidates,
        "eligible_training_target_ids": eligible,
        "excluded_by_rule": {
            reason: sorted(values) for reason, values in reasons.items()
        },
        "counts": {
            "candidate_target_ids": len(candidates),
            "eligible_training_target_ids": len(eligible),
            "excluded_target_ids_union": len(set(candidates) - set(eligible)),
            "training_observations": len(training),
            "validation_observations": len(validation),
            "validation_calendar_dates": len(validation_dates),
            "test_observations": len(test),
        },
        "digests": {
            "candidate_target_ids_sha256": _digest_ids(candidates),
            "training_ids_sha256": _digest_ids(training),
            "validation_ids_sha256": _digest_ids(validation),
            "validation_calendar_dates_sha256": _digest_ids(validation_dates),
            "test_ids_sha256": _digest_ids(test),
        },
    }
