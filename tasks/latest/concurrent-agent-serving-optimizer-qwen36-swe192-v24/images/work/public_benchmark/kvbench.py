#!/usr/bin/env python3
"""Unprivileged client for the trusted MLSys Forge public evaluation plane."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import stat
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


PUBLIC_PROFILE = "public-swe192-distinct"
PUBLIC_WORKFLOWS = 192
PUBLIC_CONCURRENCY = 192
PUBLIC_MAX_STEPS = 100
PUBLIC_MEASUREMENT_WARMUP_SEC = 300
PUBLIC_MEASUREMENT_WINDOW_SEC = 600
PUBLIC_DIRECT_RATE_ENV = "KVBENCH_PUBLIC_DIRECT_RATE"

MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
DEFAULT_POLL_INTERVAL_SEC = 10.0
DEFAULT_EVALUATION_TIMEOUT_SEC = 2400.0
EXCLUDED_DIRECTORIES = {".git", "__pycache__", ".pytest_cache", ".mypy_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


class KvbenchError(RuntimeError):
    """A safe, user-facing public benchmark client error."""


@dataclass(frozen=True)
class ClientConfig:
    judge_url: str
    token: str
    submission_root: Path
    eval_root: Path
    poll_interval_sec: float
    evaluation_timeout_sec: float

    @classmethod
    def from_environment(cls) -> "ClientConfig":
        judge_url = os.environ.get("SFORGE_JUDGE_URL", "").strip().rstrip("/")
        token = os.environ.get("SFORGE_TOKEN", "").strip()
        if not judge_url:
            raise KvbenchError("SFORGE_JUDGE_URL is not set")
        parsed = urllib.parse.urlsplit(judge_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise KvbenchError("SFORGE_JUDGE_URL must be an HTTP(S) origin")
        if parsed.query or parsed.fragment or parsed.username or parsed.password:
            raise KvbenchError("SFORGE_JUDGE_URL must not contain credentials, a query, or a fragment")
        if not token:
            raise KvbenchError("SFORGE_TOKEN is not set")

        poll_interval = _positive_float_from_env(
            "KVBENCH_PUBLIC_POLL_INTERVAL_SEC", DEFAULT_POLL_INTERVAL_SEC
        )
        evaluation_timeout = _positive_float_from_env(
            "KVBENCH_PUBLIC_TIMEOUT_SEC", DEFAULT_EVALUATION_TIMEOUT_SEC
        )
        return cls(
            judge_url=judge_url,
            token=token,
            submission_root=Path(
                os.environ.get("KVBENCH_SUBMISSION_ROOT", "/app/submission")
            ),
            eval_root=Path(
                os.environ.get("KVBENCH_PUBLIC_EVALS_DIR", "/app/public_evals")
            ),
            poll_interval_sec=poll_interval,
            evaluation_timeout_sec=evaluation_timeout,
        )


def _positive_float_from_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise KvbenchError(f"{name} must be a positive number") from exc
    if not 0 < value <= 86_400:
        raise KvbenchError(f"{name} must be greater than zero and at most 86400")
    return value


def configured_direct_rate() -> float:
    """Return the non-secret Direct rate selected by the trusted run launcher."""

    raw = os.environ.get(PUBLIC_DIRECT_RATE_ENV, "").strip()
    if not raw:
        raise KvbenchError(
            f"{PUBLIC_DIRECT_RATE_ENV} is not provisioned by the trusted runner"
        )
    try:
        value = float(raw)
    except ValueError as exc:
        raise KvbenchError(f"{PUBLIC_DIRECT_RATE_ENV} must be a positive number") from exc
    if not 0 < value < float("inf"):
        raise KvbenchError(f"{PUBLIC_DIRECT_RATE_ENV} must be a finite positive number")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode_json_object(payload: bytes, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise KvbenchError(f"{description} was not valid JSON") from exc
    if not isinstance(value, dict):
        raise KvbenchError(f"{description} must be a JSON object")
    return value


def _read_bounded(response: Any) -> bytes:
    declared = response.headers.get("Content-Length")
    if declared:
        try:
            if int(declared) > MAX_RESPONSE_BYTES:
                raise KvbenchError("Judge response exceeded the public response limit")
        except ValueError:
            pass
    payload = response.read(MAX_RESPONSE_BYTES + 1)
    if len(payload) > MAX_RESPONSE_BYTES:
        raise KvbenchError("Judge response exceeded the public response limit")
    return payload


def request_json(
    config: ClientConfig,
    method: str,
    path: str,
    *,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    if not path.startswith("/"):
        raise KvbenchError("internal request path must be absolute")
    request = urllib.request.Request(
        f"{config.judge_url}{path}",
        data=body,
        method=method,
        headers=headers or {},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return _decode_json_object(
                _read_bounded(response), description="Judge response"
            )
    except urllib.error.HTTPError as exc:
        detail = exc.read(64 * 1024).decode("utf-8", errors="replace").strip()
        if detail:
            try:
                detail_object = _decode_json_object(
                    detail.encode("utf-8"), description="Judge error"
                )
                detail = str(detail_object.get("detail") or "request rejected")
            except KvbenchError:
                detail = "request rejected"
        else:
            detail = "request rejected"
        raise KvbenchError(f"Judge returned HTTP {exc.code}: {detail}") from exc
    except (OSError, TimeoutError) as exc:
        raise KvbenchError(f"public evaluation service is unavailable: {exc}") from exc


def build_submission_request(
    token: str,
    archive: bytes,
) -> tuple[bytes, dict[str, str]]:
    if len(archive) > MAX_ARCHIVE_BYTES:
        raise KvbenchError("candidate archive exceeds the 128 MiB upload limit")
    return archive, {
        "Content-Type": "application/gzip",
        "X-SForge-Token": token,
        "X-SForge-Submission-Kind": "public",
    }


def submit_public_archive(config: ClientConfig, archive_path: Path) -> dict[str, Any]:
    metadata = archive_path.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise KvbenchError("candidate archive is not a regular file")
    if metadata.st_size > MAX_ARCHIVE_BYTES:
        raise KvbenchError("candidate archive exceeds the 128 MiB upload limit")
    archive = archive_path.read_bytes()
    body, headers = build_submission_request(config.token, archive)
    result = request_json(
        config,
        "POST",
        "/api/v1/submit",
        body=body,
        headers=headers,
        timeout=120.0,
    )
    submission_id = result.get("submission_id")
    if not isinstance(submission_id, str) or not submission_id:
        raise KvbenchError("Judge accepted no public submission id")
    return result


def poll_public_result(
    config: ClientConfig,
    submission_id: str,
    *,
    monotonic: Any = time.monotonic,
    sleep: Any = time.sleep,
) -> dict[str, Any]:
    if not submission_id or "/" in submission_id or "?" in submission_id:
        raise KvbenchError("Judge returned an invalid public submission id")
    deadline = monotonic() + config.evaluation_timeout_sec
    while True:
        result = request_json(
            config,
            "GET",
            f"/api/v1/result/{urllib.parse.quote(submission_id, safe='')}",
            timeout=30.0,
        )
        status_value = result.get("status")
        if status_value in {"completed", "error"}:
            if status_value == "error" or result.get("error"):
                raise KvbenchError(
                    f"public evaluation failed: {result.get('error') or 'unknown Judge error'}"
                )
            report = result.get("report")
            if not isinstance(report, dict):
                raise KvbenchError("completed public evaluation has no report")
            return result
        if status_value not in {"queued", "running"}:
            raise KvbenchError("Judge returned an invalid public evaluation status")
        now = monotonic()
        if now >= deadline:
            raise KvbenchError("public evaluation did not finish before the client timeout")
        sleep(min(config.poll_interval_sec, max(0.0, deadline - now)))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        raise


def allocate_eval_id(root: Path) -> int:
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".counter.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
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


def _read_regular_file(path: Path) -> tuple[bytes, int]:
    metadata = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise KvbenchError(f"unsafe submission entry: {path.name}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
            metadata.st_dev,
            metadata.st_ino,
        ):
            raise KvbenchError(f"submission changed while snapshotting: {path.name}")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_ARCHIVE_BYTES:
                raise KvbenchError("submission tree exceeds the 128 MiB snapshot limit")
            chunks.append(chunk)
        return b"".join(chunks), stat.S_IMODE(opened.st_mode)
    finally:
        os.close(descriptor)


def snapshot_submission(source: Path, destination: Path) -> str:
    source_metadata = source.lstat()
    if source.is_symlink() or not stat.S_ISDIR(source_metadata.st_mode):
        raise KvbenchError("submission root is not a real directory")
    launch = source / "launch.sh"
    if not launch.is_file() or launch.is_symlink() or not os.access(launch, os.X_OK):
        raise KvbenchError("submission/launch.sh must be a regular executable file")

    destination.mkdir(mode=0o700, parents=False, exist_ok=False)
    digest = hashlib.sha256()
    total_size = 0
    for current, directory_names, file_names in os.walk(source, followlinks=False):
        current_path = Path(current)
        relative_directory = current_path.relative_to(source)
        directory_names[:] = sorted(
            name for name in directory_names if name not in EXCLUDED_DIRECTORIES
        )
        target_directory = destination / relative_directory
        target_directory.mkdir(mode=0o700, parents=True, exist_ok=True)

        for directory_name in directory_names:
            child = current_path / directory_name
            metadata = child.lstat()
            if child.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
                raise KvbenchError(
                    f"unsafe submission directory: {child.relative_to(source)}"
                )

        for file_name in sorted(file_names):
            child = current_path / file_name
            if child.suffix in EXCLUDED_SUFFIXES:
                continue
            relative = child.relative_to(source)
            payload, mode = _read_regular_file(child)
            total_size += len(payload)
            if total_size > MAX_ARCHIVE_BYTES:
                raise KvbenchError("submission tree exceeds the 128 MiB snapshot limit")
            digest.update(relative.as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(f"{mode:04o}".encode("ascii"))
            digest.update(b"\0")
            digest.update(payload)
            target = destination / relative
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            descriptor = os.open(
                target,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                mode,
            )
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(payload)
            except BaseException:
                with contextlib.suppress(FileNotFoundError):
                    target.unlink()
                raise
    return digest.hexdigest()


def create_snapshot_archive(code_directory: Path, archive_path: Path) -> None:
    with tarfile.open(archive_path, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        archive.add(code_directory, arcname="submission", recursive=True)
    if archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise KvbenchError("candidate archive exceeds the 128 MiB upload limit")


def begin_candidate_eval(config: ClientConfig) -> tuple[Path, dict[str, Any]]:
    eval_id = allocate_eval_id(config.eval_root)
    running = config.eval_root / f"eval_{eval_id:06d}.RUNNING"
    running.mkdir(mode=0o700, parents=False, exist_ok=False)
    try:
        tree_sha256 = snapshot_submission(config.submission_root, running / "code")
        create_snapshot_archive(running / "code", running / "code_snapshot.tar.gz")
        metadata: dict[str, Any] = {
            "schema_version": 1,
            "eval_id": eval_id,
            "version": f"v{eval_id:06d}",
            "mode": "candidate",
            "profile": PUBLIC_PROFILE,
            "status": "running",
            "started_unix": time.time(),
            "submission_tree_sha256": tree_sha256,
            "code_snapshot": "code_snapshot.tar.gz",
            "workload": public_workload_contract(),
        }
        atomic_json(running / "metadata.json", metadata)
        return running, metadata
    except BaseException:
        shutil.rmtree(running, ignore_errors=True)
        raise


def _freeze_tree(root: Path) -> None:
    for current, directory_names, file_names in os.walk(root, topdown=False):
        current_path = Path(current)
        for file_name in file_names:
            with contextlib.suppress(OSError):
                (current_path / file_name).chmod(0o444)
        for directory_name in directory_names:
            with contextlib.suppress(OSError):
                (current_path / directory_name).chmod(0o555)
    with contextlib.suppress(OSError):
        root.chmod(0o555)


def finish_candidate_eval(
    config: ClientConfig,
    running: Path,
    metadata: dict[str, Any],
    *,
    result: dict[str, Any] | None,
    error: str | None,
) -> Path:
    if result is not None:
        atomic_json(running / "candidate_result.json", result)
    completed = dict(metadata)
    completed.update(
        {
            "status": "failed" if error else "complete",
            "completed_unix": time.time(),
            "error": error,
            "submission_id": (result or {}).get("submission_id")
            or metadata.get("submission_id"),
        }
    )
    atomic_json(running / "metadata.json", completed)
    final = running.with_name(running.name.removesuffix(".RUNNING") + ".COMPLETE")
    os.replace(running, final)
    atomic_json(config.eval_root / "latest.json", {**completed, "artifact_dir": str(final)})
    history_path = config.eval_root / "history.jsonl"
    descriptor = os.open(
        history_path,
        os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "a", encoding="utf-8") as history:
        history.write(
            json.dumps({**completed, "artifact_dir": str(final)}, sort_keys=True) + "\n"
        )
        history.flush()
        os.fsync(history.fileno())
    _freeze_tree(final)
    return final


def public_workload_contract() -> dict[str, Any]:
    return {
        "profile": PUBLIC_PROFILE,
        "workflows": PUBLIC_WORKFLOWS,
        "concurrency": PUBLIC_CONCURRENCY,
        "max_steps": PUBLIC_MAX_STEPS,
        "measurement_warmup_sec": PUBLIC_MEASUREMENT_WARMUP_SEC,
        "measurement_window_sec": PUBLIC_MEASUREMENT_WINDOW_SEC,
    }


def frozen_baseline_result() -> dict[str, Any]:
    direct_rate = configured_direct_rate()
    return {
        "schema_version": 1,
        "mode": "baseline",
        "profile": PUBLIC_PROFILE,
        "status": "complete",
        "source": "frozen-direct",
        "valid": True,
        "score": direct_rate,
        "valid_steps_per_min": direct_rate,
        "workload": public_workload_contract(),
        "text": (
            "Frozen Direct public anchor: "
            f"{direct_rate:.3f} valid steps/min "
            f"({PUBLIC_WORKFLOWS} workflows, concurrency {PUBLIC_CONCURRENCY}, "
            f"{PUBLIC_MEASUREMENT_WARMUP_SEC}s unscored warmup followed by a "
            f"{PUBLIC_MEASUREMENT_WINDOW_SEC}s scored window)."
        ),
    }


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    if result < 0 or result != result or result in {float("inf"), float("-inf")}:
        return None
    return result


def _candidate_report(result: dict[str, Any]) -> dict[str, Any]:
    report = result.get("report")
    if not isinstance(report, dict):
        raise KvbenchError("saved candidate evaluation has no report")
    return report


def candidate_rate(result: dict[str, Any]) -> float:
    report = _candidate_report(result)
    candidates: list[Any] = [report.get("valid_steps_per_min"), report.get("score")]
    candidate = report.get("candidate")
    if isinstance(candidate, dict):
        candidates.append(candidate.get("valid_steps_per_min"))
    evaluation = report.get("evaluation")
    if isinstance(evaluation, dict):
        candidates.append(evaluation.get("candidate_valid_steps_per_min"))
    for value in candidates:
        parsed = _number(value)
        if parsed is not None:
            return parsed
    raise KvbenchError("saved candidate evaluation has no throughput score")


def candidate_valid(result: dict[str, Any]) -> bool:
    report = _candidate_report(result)
    value = report.get("valid")
    if isinstance(value, bool):
        return value
    evaluation = report.get("evaluation")
    if isinstance(evaluation, dict) and isinstance(evaluation.get("valid"), bool):
        return bool(evaluation["valid"])
    return True


def load_latest_candidate(config: ClientConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    latest_path = config.eval_root / "latest.json"
    if not latest_path.is_file() or latest_path.is_symlink():
        raise KvbenchError("no completed public candidate evaluation is available")
    latest = _decode_json_object(latest_path.read_bytes(), description="latest metadata")
    artifact_value = latest.get("artifact_dir")
    if not isinstance(artifact_value, str):
        raise KvbenchError("latest public evaluation metadata is incomplete")
    artifact = Path(artifact_value)
    try:
        artifact.relative_to(config.eval_root)
    except ValueError as exc:
        raise KvbenchError("latest public evaluation points outside its artifact root") from exc
    if artifact.name.endswith(".COMPLETE") is False or artifact.is_symlink():
        raise KvbenchError("latest public evaluation artifact is invalid")
    result_path = artifact / "candidate_result.json"
    if not result_path.is_file() or result_path.is_symlink():
        raise KvbenchError("latest public candidate evaluation did not complete successfully")
    result = _decode_json_object(result_path.read_bytes(), description="candidate result")
    return latest, result


def compare_result(config: ClientConfig) -> dict[str, Any]:
    latest, candidate_result = load_latest_candidate(config)
    baseline_rate = configured_direct_rate()
    measured_rate = candidate_rate(candidate_result)
    speedup = measured_rate / baseline_rate
    change_percent = (speedup - 1.0) * 100.0
    valid = candidate_valid(candidate_result)
    status = "valid" if valid else "invalid"
    return {
        "schema_version": 1,
        "profile": PUBLIC_PROFILE,
        "status": status,
        "valid": valid,
        "baseline_source": "frozen-direct",
        "baseline_valid_steps_per_min": baseline_rate,
        "candidate_valid_steps_per_min": measured_rate,
        "speedup": speedup,
        "change_percent": change_percent,
        "artifact_dir": latest["artifact_dir"],
        "submission_tree_sha256": latest.get("submission_tree_sha256"),
        "text": (
            f"Public candidate is {status}: {measured_rate:.3f} valid steps/min; "
            f"frozen Direct is {baseline_rate:.1f}; speedup {speedup:.3f}x "
            f"({change_percent:+.2f}%)."
        ),
    }


@contextlib.contextmanager
def public_run_lock(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".public-run.lock").open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield


def run_baseline(config: ClientConfig) -> dict[str, Any]:
    result = frozen_baseline_result()
    atomic_json(config.eval_root / "frozen_direct_baseline.json", result)
    return result


def run_candidate(config: ClientConfig) -> dict[str, Any]:
    with public_run_lock(config.eval_root):
        running, metadata = begin_candidate_eval(config)
        try:
            accepted = submit_public_archive(config, running / "code_snapshot.tar.gz")
            submission_id = str(accepted["submission_id"])
            metadata = {**metadata, "submission_id": submission_id}
            atomic_json(running / "metadata.json", metadata)
            result = poll_public_result(config, submission_id)
        except BaseException as exc:
            finish_candidate_eval(
                config,
                running,
                metadata,
                result=None,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        artifact = finish_candidate_eval(
            config, running, metadata, result=result, error=None
        )
    comparison = compare_result(config)
    return {
        **result,
        "candidate_eval_artifact": str(artifact),
        "comparison": comparison,
    }


def doctor(config: ClientConfig) -> dict[str, Any]:
    root_metadata = config.submission_root.lstat()
    if config.submission_root.is_symlink() or not stat.S_ISDIR(root_metadata.st_mode):
        raise KvbenchError("submission root is not a real directory")
    launch = config.submission_root / "launch.sh"
    if not launch.is_file() or launch.is_symlink() or not os.access(launch, os.X_OK):
        raise KvbenchError("submission/launch.sh must be a regular executable file")
    config.eval_root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=config.eval_root, prefix=".doctor-", delete=True):
        pass
    query = urllib.parse.urlencode({"token": config.token})
    service = request_json(
        config, "GET", f"/api/v1/public/history?{query}", timeout=30.0
    )
    return {
        "schema_version": 1,
        "status": "ok",
        "profile": PUBLIC_PROFILE,
        "workload": public_workload_contract(),
        "frozen_direct_valid_steps_per_min": configured_direct_rate(),
        "submission_root": str(config.submission_root),
        "public_evals_root": str(config.eval_root),
        "service": service,
        "text": (
            "Public evaluation is ready: "
            f"{PUBLIC_WORKFLOWS} workflows at concurrency {PUBLIC_CONCURRENCY}, "
            f"max {PUBLIC_MAX_STEPS} steps, fixed {PUBLIC_MEASUREMENT_WARMUP_SEC}s "
            f"unscored warmup + {PUBLIC_MEASUREMENT_WINDOW_SEC}s scored window."
        ),
    }


def _validate_profile(profile: str) -> None:
    if profile != PUBLIC_PROFILE:
        raise KvbenchError(
            f"unsupported public profile {profile!r}; expected {PUBLIC_PROFILE!r}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kvbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser(
        "doctor", help="check the public evaluation connection and local workspace"
    )
    doctor_parser.add_argument("--json", dest="json_output", action="store_true")

    run_parser = subparsers.add_parser("run", help="record Direct or evaluate Candidate")
    run_parser.add_argument("--mode", choices=("baseline", "candidate"), required=True)
    run_parser.add_argument("--profile", default=PUBLIC_PROFILE)
    run_parser.add_argument("--json", dest="json_output", action="store_true")

    compare_parser = subparsers.add_parser(
        "compare", help="compare the latest Candidate with frozen Direct"
    )
    compare_parser.add_argument("--profile", default=PUBLIC_PROFILE)
    compare_parser.add_argument("--json", dest="json_output", action="store_true")
    return parser


def _print_result(result: dict[str, Any], json_output: bool) -> None:
    if json_output:
        json.dump(result, sys.stdout, indent=2, sort_keys=True)
        print()
        return
    text = result.get("text")
    if not isinstance(text, str):
        comparison = result.get("comparison")
        if isinstance(comparison, dict):
            text = comparison.get("text")
    if isinstance(text, str):
        print(text)
    else:
        json.dump(result, sys.stdout, indent=2, sort_keys=True)
        print()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = ClientConfig.from_environment()
        if args.command == "doctor":
            result = doctor(config)
        else:
            _validate_profile(args.profile)
            if args.command == "run":
                result = run_baseline(config) if args.mode == "baseline" else run_candidate(config)
            else:
                result = compare_result(config)
        _print_result(result, args.json_output)
        return 0
    except (KvbenchError, OSError) as exc:
        print(f"kvbench: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
