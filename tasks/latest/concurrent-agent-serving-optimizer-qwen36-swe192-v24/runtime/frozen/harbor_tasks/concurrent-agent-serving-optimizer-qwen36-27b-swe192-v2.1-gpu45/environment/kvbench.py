#!/usr/bin/env python3
"""Unprivileged client for the trusted benchmark controller."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import stat
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


CONTROLLER = os.environ.get("PUBLIC_BENCH_URL", "http://bench-controller:7000").rstrip("/")
SUBMISSION_ROOT = Path(os.environ.get("KVBENCH_SUBMISSION_ROOT", "/app/submission"))
EVAL_ROOT = Path(os.environ.get("KVBENCH_CANDIDATE_EVALS_DIR", "/app/public_evals"))
SNAPSHOT_EXCLUDED_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache"}
SNAPSHOT_PROFILE = (
    os.environ.get("KVBENCH_PUBLIC_PROFILE", "public-pressure-192").strip()
    or "public-pressure-192"
)


def capture_candidate_eval_enabled(profile: str) -> bool:
    """Capture only the performance-bearing public pressure evaluation."""

    return profile == SNAPSHOT_PROFILE and os.environ.get(
        "KVBENCH_CAPTURE_CANDIDATE_EVAL", "1"
    ).strip().lower() not in {"0", "false", "no", "off"}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def allocate_eval_id(root: Path | None = None) -> int:
    root = EVAL_ROOT if root is None else root
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".counter.lock"
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        raw = handle.read().strip()
        try:
            previous = int(raw or "0")
        except ValueError:
            previous = 0
        eval_id = previous + 1
        handle.seek(0)
        handle.truncate()
        handle.write(f"{eval_id}\n")
        handle.flush()
        os.fsync(handle.fileno())
        return eval_id


def snapshot_submission(source: Path, destination: Path) -> str:
    """Copy one symlink-free code version and return its content digest."""

    source_metadata = source.lstat()
    if source.is_symlink() or not stat.S_ISDIR(source_metadata.st_mode):
        raise RuntimeError("submission root is not a real directory")
    destination.mkdir(parents=False, exist_ok=False)
    digest = hashlib.sha256()
    for current, directory_names, file_names in os.walk(source, followlinks=False):
        current_path = Path(current)
        relative_dir = current_path.relative_to(source)
        directory_names[:] = sorted(
            name for name in directory_names if name not in SNAPSHOT_EXCLUDED_DIRS
        )
        target_dir = destination / relative_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        for directory_name in directory_names:
            child = current_path / directory_name
            metadata = child.lstat()
            if child.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
                raise RuntimeError(f"unsafe submission directory: {child.relative_to(source)}")
        for file_name in sorted(file_names):
            if file_name.endswith((".pyc", ".pyo")):
                continue
            child = current_path / file_name
            relative = child.relative_to(source)
            metadata = child.lstat()
            if child.is_symlink() or not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError(f"unsafe submission file: {relative}")
            payload = child.read_bytes()
            digest.update(str(relative).encode())
            digest.update(b"\0")
            digest.update(f"{stat.S_IMODE(metadata.st_mode):04o}".encode())
            digest.update(b"\0")
            digest.update(payload)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            target.chmod(stat.S_IMODE(metadata.st_mode))
    return digest.hexdigest()


def begin_candidate_eval(profile: str) -> tuple[Path, dict[str, Any]]:
    eval_id = allocate_eval_id()
    running = EVAL_ROOT / f"eval_{eval_id:06d}.RUNNING"
    running.mkdir(mode=0o755)
    try:
        tree_sha256 = snapshot_submission(SUBMISSION_ROOT, running / "code")
        with tarfile.open(running / "code_snapshot.tar.gz", "w:gz") as archive:
            archive.add(running / "code", arcname="submission", recursive=True)
        metadata: dict[str, Any] = {
            "schema_version": 1,
            "eval_id": eval_id,
            "version": f"v{eval_id:06d}",
            "profile": profile,
            "mode": "candidate",
            "status": "running",
            "started_unix": time.time(),
            "submission_tree_sha256": tree_sha256,
            "code_snapshot": "code_snapshot.tar.gz",
        }
        atomic_json(running / "metadata.json", metadata)
        return running, metadata
    except BaseException:
        shutil.rmtree(running, ignore_errors=True)
        raise


def finish_candidate_eval(
    running: Path,
    metadata: dict[str, Any],
    *,
    candidate_result: dict[str, Any] | None,
    comparison: dict[str, Any] | None,
    error: str | None,
    compare_error: str | None = None,
) -> Path:
    if candidate_result is not None:
        atomic_json(running / "candidate_result.json", candidate_result)
    if comparison is not None:
        atomic_json(running / "compare.json", comparison)
    completed = dict(metadata)
    completed.update(
        {
            "status": "failed" if error else "complete",
            "completed_unix": time.time(),
            "error": error,
            "compare_error": compare_error,
            "pair_id": (candidate_result or {}).get("pair_id"),
            "candidate_run_id": ((candidate_result or {}).get("summary") or {}).get(
                "run_id"
            ),
            "speedup": (comparison or {}).get("speedup"),
            "comparison_status": (comparison or {}).get("status"),
        }
    )
    atomic_json(running / "metadata.json", completed)
    final = running.with_name(running.name.removesuffix(".RUNNING") + ".COMPLETE")
    os.replace(running, final)
    atomic_json(EVAL_ROOT / "latest.json", {**completed, "artifact_dir": str(final)})
    with (EVAL_ROOT / "history.jsonl").open("a") as history:
        history.write(json.dumps({**completed, "artifact_dir": str(final)}, sort_keys=True) + "\n")
    return final


def request(method: str, path: str, payload: dict | None = None, timeout: int = 30) -> dict:
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{CONTROLLER}{path}",
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"kvbench: controller returned HTTP {exc.code}: {message}") from exc
    except OSError as exc:
        raise SystemExit(f"kvbench: benchmark controller is unavailable: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(prog="kvbench")
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor = subparsers.add_parser("doctor")
    doctor.add_argument(
        "--json", dest="json_output", action="store_true", help="print the full JSON response"
    )

    run = subparsers.add_parser("run")
    run.add_argument("--mode", choices=("baseline", "candidate"), required=True)
    run.add_argument("--profile", required=True)
    run.add_argument(
        "--json", dest="json_output", action="store_true", help="print the full JSON response"
    )

    compare = subparsers.add_parser("compare")
    compare.add_argument("--profile", required=True)
    compare.add_argument(
        "--json", dest="json_output", action="store_true", help="print the full JSON response"
    )

    args = parser.parse_args()
    if args.command == "doctor":
        result = request("GET", "/public/doctor")
    elif args.command == "run":
        evaluation: tuple[Path, dict[str, Any]] | None = None
        if args.mode == "candidate" and capture_candidate_eval_enabled(args.profile):
            try:
                evaluation = begin_candidate_eval(args.profile)
            except BaseException as exc:
                raise SystemExit(f"kvbench: could not snapshot candidate version: {exc}") from exc
        try:
            result = request(
                "POST",
                "/public/run",
                {"mode": args.mode, "profile": args.profile},
                timeout=21600,
            )
        except BaseException as exc:
            if evaluation is not None:
                finish_candidate_eval(
                    *evaluation,
                    candidate_result=None,
                    comparison=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
            raise
        if evaluation is not None:
            comparison = None
            compare_error = None
            try:
                comparison = request("GET", f"/public/compare?profile={args.profile}")
            except BaseException as exc:
                compare_error = f"{type(exc).__name__}: {exc}"
            artifact_dir = finish_candidate_eval(
                *evaluation,
                candidate_result=result,
                comparison=comparison,
                error=None,
                compare_error=compare_error,
            )
            result = {**result, "candidate_eval_artifact": str(artifact_dir)}
    else:
        result = request("GET", f"/public/compare?profile={args.profile}")

    if args.json_output:
        json.dump(result, sys.stdout, indent=2, sort_keys=True)
        print()
    elif "text" in result:
        identity = []
        for key in ("pair_id", "baseline_anchor_pair_id", "parent_pair_id"):
            if key in result:
                value = result[key]
                identity.append(f"{key}: {'none' if value is None else value}")
        if identity:
            print("\n".join(identity))
        print(result["text"])
    else:
        json.dump(result, sys.stdout, indent=2, sort_keys=True)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
