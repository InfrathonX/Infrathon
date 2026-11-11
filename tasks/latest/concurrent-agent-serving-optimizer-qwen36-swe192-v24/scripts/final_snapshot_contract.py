#!/usr/bin/env python3
"""Strict schema-v2 contract for an immutable stopped-container snapshot."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import stat
from typing import Any


class SnapshotContractError(RuntimeError):
    """The frozen snapshot metadata or archive is not authoritative."""


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SnapshotContractError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise SnapshotContractError(f"{label} must be finite")
    return number


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SnapshotContractError(f"{label} must be an integer >= {minimum}")
    return value


def _close(left: float, right: float, label: str, *, tolerance: float = 1e-6) -> None:
    if not math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance):
        raise SnapshotContractError(f"{label} does not match")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> tuple[Path, int, str]:
    if path.is_symlink() or not path.is_file():
        raise SnapshotContractError(f"{label} must be a regular non-symlink file")
    resolved = path.resolve(strict=True)
    info = resolved.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
        raise SnapshotContractError(f"{label} must be non-empty and regular")
    return resolved, info.st_size, _sha256(resolved)


def _strict_object(path: Path) -> dict[str, Any]:
    resolved, _, _ = _regular_file(path, "final snapshot metadata")

    def reject_constant(value: str) -> None:
        raise SnapshotContractError(
            f"final snapshot contains non-finite JSON constant {value}"
        )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SnapshotContractError(
                    f"final snapshot contains duplicate key {key}"
                )
            result[key] = value
        return result

    try:
        payload = json.loads(
            resolved.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotContractError("final snapshot is not strict UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise SnapshotContractError("final snapshot root must be an object")
    return payload


def validate_snapshot(
    snapshot_path: Path,
    archive_path: Path,
    expected_seconds: float,
) -> dict[str, Any]:
    """Validate and return the audit-relevant schema-v2 snapshot fields."""

    payload = _strict_object(snapshot_path)
    archive_resolved, archive_size, archive_sha256 = _regular_file(
        archive_path, "final archive"
    )
    if set(payload) != {
        "schema_version",
        "captured_at",
        "snapshot_elapsed_seconds",
        "snapshot_boundary",
        "archive_sha256",
        "deadline",
        "freeze",
        "capture",
        "archive",
    }:
        raise SnapshotContractError("final snapshot schema-v2 fields are not exact")
    if (
        isinstance(payload.get("schema_version"), bool)
        or payload.get("schema_version") != 2
        or payload.get("snapshot_boundary") != "whole_container_frozen"
    ):
        raise SnapshotContractError("final snapshot identity is invalid")

    deadline = payload.get("deadline")
    freeze = payload.get("freeze")
    capture = payload.get("capture")
    archive = payload.get("archive")
    if not isinstance(deadline, dict) or set(deadline) != {
        "at",
        "elapsed_seconds",
        "monotonic",
    }:
        raise SnapshotContractError("snapshot deadline schema is not exact")
    if not isinstance(freeze, dict) or set(freeze) != {
        "mechanism",
        "all_work_processes_frozen",
        "all_work_processes_stopped",
        "was_running",
        "freeze_requested_at",
        "freeze_requested_elapsed_seconds",
        "frozen_at",
        "frozen_elapsed_seconds",
        "container_status",
        "container_pid",
    }:
        raise SnapshotContractError("snapshot freeze schema is not exact")
    if not isinstance(capture, dict) or set(capture) != {
        "source",
        "transport",
        "work_container_running_during_capture",
        "started_at",
        "started_elapsed_seconds",
        "archive_completed_at",
        "archive_completed_elapsed_seconds",
    }:
        raise SnapshotContractError("snapshot capture schema is not exact")
    if not isinstance(archive, dict) or set(archive) != {
        "path",
        "format",
        "size_bytes",
        "sha256",
    }:
        raise SnapshotContractError("snapshot archive schema is not exact")

    captured_at = _finite(payload.get("captured_at"), "captured_at")
    snapshot_elapsed = _finite(
        payload.get("snapshot_elapsed_seconds"), "snapshot_elapsed_seconds"
    )
    deadline_at = _finite(deadline.get("at"), "deadline.at")
    deadline_elapsed = _finite(
        deadline.get("elapsed_seconds"), "deadline.elapsed_seconds"
    )
    deadline_monotonic = _finite(deadline.get("monotonic"), "deadline.monotonic")
    requested_at = _finite(freeze.get("freeze_requested_at"), "freeze.freeze_requested_at")
    requested_elapsed = _finite(
        freeze.get("freeze_requested_elapsed_seconds"),
        "freeze.freeze_requested_elapsed_seconds",
    )
    frozen_at = _finite(freeze.get("frozen_at"), "freeze.frozen_at")
    frozen_elapsed = _finite(
        freeze.get("frozen_elapsed_seconds"), "freeze.frozen_elapsed_seconds"
    )
    capture_started_at = _finite(capture.get("started_at"), "capture.started_at")
    capture_started_elapsed = _finite(
        capture.get("started_elapsed_seconds"), "capture.started_elapsed_seconds"
    )
    completed_at = _finite(
        capture.get("archive_completed_at"), "capture.archive_completed_at"
    )
    completed_elapsed = _finite(
        capture.get("archive_completed_elapsed_seconds"),
        "capture.archive_completed_elapsed_seconds",
    )

    expected = _finite(expected_seconds, "expected timeout")
    if expected <= 0.0:
        raise SnapshotContractError("expected timeout must be positive")
    _close(deadline_elapsed, expected, "snapshot deadline elapsed")
    if deadline_at <= 0.0 or deadline_monotonic <= 0.0:
        raise SnapshotContractError("snapshot deadline timestamps must be positive")
    if not deadline_elapsed <= requested_elapsed <= expected + 5.0:
        raise SnapshotContractError("freeze request is outside the deadline window")
    if not requested_elapsed <= frozen_elapsed <= requested_elapsed + 5.0:
        raise SnapshotContractError("container freeze latency exceeds 5 seconds")
    _close(snapshot_elapsed, frozen_elapsed, "snapshot/freeze boundary")
    if not frozen_elapsed <= capture_started_elapsed <= completed_elapsed:
        raise SnapshotContractError("capture elapsed timestamps are out of order")
    if not (
        deadline_at <= requested_at + 1e-6
        and requested_at <= frozen_at + 1.0
        and frozen_at <= capture_started_at + 1.0
        and capture_started_at <= completed_at + 1.0
    ):
        raise SnapshotContractError("snapshot wall-clock timestamps are out of order")
    _close(captured_at, completed_at, "captured/archive-completed timestamp")
    _close(
        requested_at - deadline_at,
        requested_elapsed - deadline_elapsed,
        "deadline/freeze-request clock delta",
        tolerance=0.01,
    )

    if (
        freeze.get("mechanism") != "docker_stop_timeout_0"
        or freeze.get("all_work_processes_frozen") is not True
        or freeze.get("all_work_processes_stopped") is not True
        or freeze.get("was_running") is not True
        or _integer(freeze.get("container_pid"), "freeze.container_pid") != 0
        or not isinstance(freeze.get("container_status"), str)
        or not freeze["container_status"]
        or freeze["container_status"] in {"running", "restarting", "paused"}
    ):
        raise SnapshotContractError("container freeze is not proven process-quiescent")
    if (
        capture.get("source") != "stopped_container_filesystem"
        or capture.get("transport") != "docker_get_archive_transport_tar"
        or capture.get("work_container_running_during_capture") is not False
    ):
        raise SnapshotContractError("snapshot capture did not remain stopped")

    archive_size_value = _integer(archive.get("size_bytes"), "archive.size_bytes", minimum=1)
    metadata_hash = archive.get("sha256")
    top_hash = payload.get("archive_sha256")
    if (
        archive.get("path") != "final_archive.tar.gz"
        or archive.get("format") != "tar+gzip"
        or archive_size_value != archive_size
        or not isinstance(metadata_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", metadata_hash) is None
        or top_hash != metadata_hash
        or metadata_hash != archive_sha256
    ):
        raise SnapshotContractError("snapshot archive commitment does not match bytes")

    return {
        "captured_at": captured_at,
        "elapsed_seconds": snapshot_elapsed,
        "boundary": payload["snapshot_boundary"],
        "deadline": deadline,
        "freeze": freeze,
        "capture": capture,
        "archive": {
            "path": str(archive_resolved),
            "size": archive_size,
            "sha256": archive_sha256,
        },
    }
