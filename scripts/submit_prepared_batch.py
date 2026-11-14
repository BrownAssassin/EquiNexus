"""Submit a hash-locked, locally prepared Kaggle candidate batch.

Without ``--submit`` this command only validates the manifest and CSV files.
It deliberately refuses paths outside ``artifacts/submissions`` and caps each
batch at the competition's five-submission daily allowance.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPETITION = "filament-segmentation-2026"
MAX_BATCH_SIZE = 5
EXPECTED_COLUMNS = ["filament_id", "segmentation_rle"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def emit(event: dict[str, Any], log_path: Path | None) -> None:
    line = json.dumps(event, sort_keys=True)
    print(line, flush=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def validate_candidate(candidate: dict[str, Any]) -> Path:
    relative = Path(str(candidate["path"]))
    path = (REPO_ROOT / relative).resolve()
    submissions_root = (REPO_ROOT / "artifacts" / "submissions").resolve()
    try:
        path.relative_to(submissions_root)
    except ValueError as exc:
        raise ValueError(f"candidate path must stay under artifacts/submissions: {path}") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_hash = sha256(path)
    expected_hash = str(candidate["sha256"])
    if actual_hash != expected_hash:
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {expected_hash}, got {actual_hash}"
        )

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(f"unexpected submission columns in {path}: {reader.fieldnames}")
        seen_ids: set[str] = set()
        row_count = 0
        for row in reader:
            filament_id = str(row["filament_id"] or "")
            rle = str(row["segmentation_rle"] or "")
            if not filament_id or not rle:
                raise ValueError(f"empty ID/RLE in {path}, row {row_count + 2}")
            if filament_id in seen_ids:
                raise ValueError(f"duplicate filament_id {filament_id} in {path}")
            seen_ids.add(filament_id)
            row_count += 1
    if row_count != int(candidate["predicted_instances"]):
        raise ValueError(
            f"instance count mismatch for {path}: manifest={candidate['predicted_instances']} "
            f"CSV={row_count}"
        )
    if int(candidate["test_images"]) != 180:
        raise ValueError(f"expected complete 180-image coverage for {path}")
    return path


@contextmanager
def exclusive_submission_lock() -> Iterator[None]:
    """Prevent overlapping local processes from submitting the same batch."""
    lock_path = REPO_ROOT / "artifacts" / "automation" / "submit-prepared-batch.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)

        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(
                    "another submission process holds the local Kaggle submission lock"
                ) from exc
            unlock = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError(
                    "another submission process holds the local Kaggle submission lock"
                ) from exc
            unlock = lambda: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

        try:
            yield
        finally:
            stream.seek(0)
            unlock()


def kaggle_api() -> Any:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def existing_submission_descriptions(api: Any) -> set[str]:
    """Return recent team submission descriptions; fail closed if history is unavailable."""
    submissions = api.competition_submissions(COMPETITION, page_number=-1, page_size=100)
    if submissions is None:
        raise RuntimeError("Kaggle returned no submission-history response; refusing to upload")
    return {
        str(submission.description).strip()
        for submission in submissions
        if submission is not None and getattr(submission, "description", None)
    }


def verify_daily_submission_quota(api: Any, batch_size: int, log_path: Path | None) -> None:
    """Fail closed unless Kaggle currently permits the whole batch."""
    limits = api.competition_get_submission_limits(COMPETITION)
    allowed = int(limits.num_allowed_now)
    emit(
        {
            "event": "submission_quota_preflight",
            "num_today": int(limits.num_today),
            "num_total": int(limits.num_total),
            "num_allowed_now": allowed,
            "limited_by_total": bool(limits.limited_by_total),
            "batch_size": batch_size,
            "utc": datetime.now(timezone.utc).isoformat(),
        },
        log_path,
    )
    if allowed < batch_size:
        raise RuntimeError(
            f"Kaggle currently permits {allowed} more submissions, but this batch has {batch_size}; "
            "refusing to start a partial batch."
        )


def validate_submit_worktree(git_output: Any, current_commit: str) -> None:
    """Allow only the explicitly enumerated end-of-day archive while submitting."""
    if git_output("branch", "--show-current") != "main":
        raise ValueError("submission requires the committed main branch")
    status_rows = git_output(
        "status", "--porcelain=v1", "--untracked-files=all"
    ).splitlines()
    allowed_pending_paths = {
        "docs/experiment-log.md",
        "docs/model-sprint.md",
        "notebooks/README.md",
        "notebooks/kaggle/i079_topology_safe_refiner/kernel-metadata.json",
        "notebooks/kaggle/i079_topology_safe_refiner/lb-0-70-topology-safe-solar-filaments-handoff.ipynb",
        "scripts/experiments/train_native_unet_softmean_patches.py",
        "scripts/experiments/run_native_unet_softmean_patches_test_inference.py",
        "scripts/experiments/train_native_unet_swin_tiny.py",
        "scripts/experiments/build_rule_compliant_test_consensus.py",
        "scripts/submit_prepared_batch.py",
    }
    unexpected_changes = []
    for row in status_rows:
        path = row[3:].strip().strip('"').replace("\\", "/")
        if row[:2] not in {" M", "??"} or path not in allowed_pending_paths:
            unexpected_changes.append(path or row)
    if unexpected_changes:
        raise ValueError(
            "submission requires committed model and scoring source to remain unchanged; only "
            "the enumerated sprint documentation and archive paths may be pending, found: "
            + ", ".join(unexpected_changes)
        )
    remote_commit = git_output("rev-parse", "origin/main")
    if remote_commit != current_commit:
        raise ValueError("source commit has not been pushed to origin/main")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--submit", action="store_true", help="actually upload the candidates")
    parser.add_argument("--log", type=Path, help="append JSONL run events to this path")
    parser.add_argument(
        "--delay-seconds", type=float, default=10.0,
        help="pause between uploads (default: 10 seconds)",
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    try:
        manifest_path.relative_to(REPO_ROOT)
    except ValueError:
        parser.error("manifest must be inside the repository")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    def git_output(*git_args: str) -> str:
        result = subprocess.run(
            ["git", *git_args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            raise RuntimeError(f"git {' '.join(git_args)} failed: {result.stderr.strip()}")
        # Preserve leading spaces in porcelain status rows; strip only line endings.
        return result.stdout.rstrip("\r\n")

    source_commit = str(manifest.get("source_commit", ""))
    current_commit = git_output("rev-parse", "HEAD")
    if source_commit != current_commit:
        raise ValueError(
            f"manifest source commit {source_commit!r} does not match current HEAD {current_commit!r}"
        )
    if args.submit:
        validate_submit_worktree(git_output, current_commit)

    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= MAX_BATCH_SIZE:
        raise ValueError(f"batch must contain 1 to {MAX_BATCH_SIZE} candidates")
    if manifest.get("competition") != COMPETITION:
        raise ValueError("competition slug does not match this runner")
    if manifest.get("split_sha256") != (
        "30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e"
    ):
        raise ValueError("manifest does not use the frozen validation split")

    resolved: list[tuple[dict[str, Any], Path]] = []
    seen_hashes: set[str] = set()
    for candidate in candidates:
        path = validate_candidate(candidate)
        if candidate["sha256"] in seen_hashes:
            raise ValueError(f"duplicate candidate bytes: {path}")
        seen_hashes.add(candidate["sha256"])
        resolved.append((candidate, path))

    mode = "submit" if args.submit else "dry-run"
    emit(
        {
            "event": "batch_validated",
            "mode": mode,
            "utc": datetime.now(timezone.utc).isoformat(),
            "manifest": str(manifest_path.relative_to(REPO_ROOT)),
            "source_commit": manifest.get("source_commit"),
            "split_sha256": manifest["split_sha256"],
            "candidate_count": len(resolved),
        },
        args.log,
    )

    kaggle = REPO_ROOT / ".venv" / "Scripts" / "kaggle.exe"
    if args.submit:
        if not kaggle.is_file():
            raise FileNotFoundError(f"Kaggle CLI is not installed at {kaggle}")

        messages = [str(candidate.get("message", "")).strip() for candidate, _ in resolved]
        if any(not message for message in messages):
            raise ValueError("every candidate must have a non-empty Kaggle submission description")
        if len(messages) != len(set(messages)):
            raise ValueError("candidate descriptions must be unique within a batch")

        with exclusive_submission_lock():
            api = kaggle_api()
            existing = existing_submission_descriptions(api)
            for candidate, path in resolved:
                message = str(candidate["message"]).strip()
                if message in existing:
                    emit(
                        {
                            "event": "candidate_skipped_already_submitted",
                            "path": str(path.relative_to(REPO_ROOT)),
                            "sha256": candidate["sha256"],
                            "message": message,
                        },
                        args.log,
                    )
            pending = [
                pair for pair in resolved if str(pair[0]["message"]).strip() not in existing
            ]
            already_submitted = len(resolved) - len(pending)
            emit(
                {
                    "event": "submission_idempotency_preflight",
                    "candidate_count": len(resolved),
                    "already_submitted_count": already_submitted,
                    "pending_count": len(pending),
                    "utc": datetime.now(timezone.utc).isoformat(),
                },
                args.log,
            )
            if pending:
                verify_daily_submission_quota(api, len(pending), args.log)
            else:
                emit(
                    {"event": "batch_already_submitted", "candidate_count": len(resolved)},
                    args.log,
                )

            for index, (candidate, path) in enumerate(pending, start=1):
                event = {
                    "event": "candidate_ready",
                    "index": index,
                    "count": len(pending),
                    "path": str(path.relative_to(REPO_ROOT)),
                    "sha256": candidate["sha256"],
                    "predicted_instances": candidate["predicted_instances"],
                    "oof_pq": candidate.get("oof_pq"),
                    "holdout_pq": candidate.get("holdout_pq"),
                    "crossfit_pq": candidate.get("crossfit_pq"),
                    "validation_protocol": candidate.get("validation_protocol"),
                    "message": candidate["message"],
                }
                command = [
                    str(kaggle), "competitions", "submit", "-c", COMPETITION,
                    "-f", str(path), "-m", str(candidate["message"]),
                ]
                result = subprocess.run(
                    command, cwd=REPO_ROOT, capture_output=True, text=True, check=False
                )
                emit(
                    {
                        **event,
                        "event": "submission_result",
                        "returncode": result.returncode,
                        "stdout": result.stdout.strip(),
                        "stderr": result.stderr.strip(),
                        "utc": datetime.now(timezone.utc).isoformat(),
                    },
                    args.log,
                )
                if result.returncode != 0:
                    emit(
                        {"event": "batch_stopped", "reason": "kaggle_cli_failed", "index": index},
                        args.log,
                    )
                    return result.returncode
                if index < len(pending) and args.delay_seconds > 0:
                    time.sleep(args.delay_seconds)

            result = subprocess.run(
                [str(kaggle), "competitions", "submissions", "-c", COMPETITION],
                cwd=REPO_ROOT, capture_output=True, text=True, check=False,
            )
            emit(
                {
                    "event": "submission_status_refresh",
                    "returncode": result.returncode,
                    "stdout": result.stdout.strip(),
                    "stderr": result.stderr.strip(),
                    "utc": datetime.now(timezone.utc).isoformat(),
                },
                args.log,
            )
            return result.returncode

    for index, (candidate, path) in enumerate(resolved, start=1):
        event = {
            "event": "candidate_ready",
            "index": index,
            "count": len(resolved),
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": candidate["sha256"],
            "predicted_instances": candidate["predicted_instances"],
            "oof_pq": candidate.get("oof_pq"),
            "holdout_pq": candidate.get("holdout_pq"),
            "crossfit_pq": candidate.get("crossfit_pq"),
            "validation_protocol": candidate.get("validation_protocol"),
            "message": candidate["message"],
        }
        emit(event, args.log)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
