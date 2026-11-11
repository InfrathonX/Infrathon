"""Agent execution with iterative evaluation via judge server.

The agent runs in a work container with:
- sforge-submit script for on-demand evaluation
- Stop hook to prevent premature exit (unconditional block)
- Auto-eval runs on the host side (invisible to agent)

Architecture:
    Host: sforge serve (judge HTTP server, started separately)
          + auto-eval thread (extracts code from container, submits to judge)
    Container: Agent + sforge-submit (curl-based)
    Communication: HTTP (container → host judge server)
"""

from __future__ import annotations

import copy
import fnmatch
import gzip
import hashlib
import io
import json
import math
import os
import signal
import shlex
import tarfile as _tarfile
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import requests

from sforge.harness.agent import Agent
from sforge.harness.agent_runtime import (
    AgentRuntimeError,
    load_agent_runtime,
    runtime_environment,
    runtime_probe_cmds,
)
from sforge.harness.backend import AcceleratorAllocation, ContainerBackend, ContainerHandle
from sforge.harness.config import (
    SForgeConfig,
    validate_single_gpu_handoff_config,
)
from sforge.harness.container_files import copy_container_file
from sforge.harness.constants import (
    ADMIN_SECRET,
    ADMIN_SECRET_HEADER,
    SESSION_TOKEN_HEADER,
    SUBMISSION_ARCHIVE_MEDIA_TYPE,
    SUBMISSION_KIND_HEADER,
)
from sforge.harness.cpu_affinity import pin_before_start
from sforge.harness.docker_build import (
    close_logger,
    setup_logger,
)
from sforge.harness.evolve_scripts import (
    generate_evolve_prompt,
    generate_game_prompt,
    generate_save_version_script,
    generate_submit_script,
)
from sforge.harness.task_spec import TaskSpec
from sforge.harness.resource_handoff import (
    handoff_annotations,
    lease_record_path,
    read_durable_lease,
)
from sforge.harness.resource_lease import FAILURE_STATES, LeaseState
from sforge.harness.version_store import StoredVersion, VersionStore

CODEX_PROCESS_GROUP_FILE = "/tmp/sforge-codex-process-group"
AUTO_EVAL_ADMISSION_FEEDBACK_PATH = PurePosixPath(
    "/workspace/benchmark/latest_auto_eval_admission_feedback.json"
)
AUTO_EVAL_SCORE_FEEDBACK_PATH = PurePosixPath(
    "/workspace/benchmark/latest_auto_eval_score_feedback.json"
)


def _portable_evaluation_contract_sha256(task_spec: TaskSpec) -> str:
    """Return the canonical portable digest without changing its domain."""

    if task_spec.portable is None:
        raise ValueError("portable evaluation contract is required")
    digest = task_spec.portable.evaluation_contract_sha256
    if (
        not isinstance(digest, str)
        or len(digest) != 71
        or not digest.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in digest[7:])
    ):
        raise ValueError("portable evaluation contract digest is not canonical")
    return digest


SAFE_DEBUG_WARNING_SUFFIXES = {".cu", ".cuh", ".h", ".hpp", ".cpp"}
MAX_SAFE_DEBUG_WARNINGS = 100


@dataclass
class RunResult:
    """Result of running an agent on a task."""

    archive: bytes = b""
    best_pass_rate: float = 0.0
    best_score: float | None = None
    best_round: str = ""
    total_rounds: int = 0
    agent_submissions: int = 0
    auto_submissions: int = 0
    agent_output: str = ""
    timed_out: bool = False
    runtime_seconds: float = 0.0
    resume_count: int = 0

    def to_dict(self) -> dict:
        d = {
            "best_pass_rate": self.best_pass_rate,
            "best_round": self.best_round,
            "total_rounds": self.total_rounds,
            "agent_submissions": self.agent_submissions,
            "auto_submissions": self.auto_submissions,
            "timed_out": self.timed_out,
            "runtime_seconds": self.runtime_seconds,
            "archive_size_bytes": len(self.archive),
            "resume_count": self.resume_count,
        }
        if self.best_score is not None:
            d["best_score"] = self.best_score
        return d


class JudgeDrainError(RuntimeError):
    """Raised when pending judge work cannot be proven quiescent."""


class FinalSnapshotError(RuntimeError):
    """Raised when an immutable final submission cannot be proven."""


class FinalVersionStoreError(RuntimeError):
    """Raised when a captured final submission cannot be made durable."""


class JudgeRegistrationError(RuntimeError):
    """Raised when the Judge session contract cannot be established."""


class HandoffFencedError(FinalSnapshotError):
    """Raised when the durable single-GPU lease entered a failure state.

    The coordinator fences a lease when it can no longer prove who owns the
    accelerator (for example a foreign process appeared on the card while
    Work was paused for a Judge cycle).  A fenced lease never resumes Work,
    so the run must stop immediately instead of idling the paused container
    until the fixed budget expires.
    """


class _FinalHandoffFence:
    """Idempotently persist never-resume before final/deadline container stop."""

    def __init__(self, host_judge_url: str, session_token: str) -> None:
        self._url = host_judge_url
        self._token = session_token
        self._lock = threading.Lock()
        self._payload: dict[str, object] | None = None

    def seal(self) -> dict[str, object]:
        with self._lock:
            if self._payload is not None:
                return dict(self._payload)
            response = requests.post(
                f"{self._url}/api/v1/handoff/seal",
                json={
                    "token": self._token,
                    "admin_secret": ADMIN_SECRET,
                },
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("recovery_policy") != "never_resume":
                raise FinalSnapshotError(
                    "Judge did not persist the final never-resume fence"
                )
            if not isinstance(payload.get("state"), str):
                raise FinalSnapshotError(
                    "Judge final fence response omitted the lease state"
                )
            self._payload = dict(payload)
            return dict(self._payload)


class _HandoffFenceWatchdog:
    """Abort the Work exec promptly once the durable lease fails closed.

    The Judge coordinator owns the lease state machine and persists every
    transition to ``resource_lease.json``.  Interactive handoffs (pause, idle
    attest, Judge, destroy, attest, resume) happen entirely inside the
    coordinator, so a fence raised there is invisible to run_agent, whose
    Docker exec simply blocks on the paused container.  This watchdog polls
    the durable record and, on any failure state, records the audited reason
    and releases the exec wait through ``exec_stop_event``.  An operator
    ``shutdown_event`` is mirrored into the same event so Ctrl+C keeps its
    existing meaning.
    """

    def __init__(
        self,
        lease_path: Path,
        *,
        shutdown_event: threading.Event | None = None,
        poll_interval: float = 5.0,
        logger=None,
    ) -> None:
        self._lease_path = lease_path
        self._shutdown_event = shutdown_event
        self._poll_interval = max(0.05, float(poll_interval))
        self._logger = logger
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.exec_stop_event = threading.Event()
        self.fenced_state: str | None = None
        self.fence_reason: str | None = None

    @property
    def fenced(self) -> bool:
        return self.fenced_state is not None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="sforge-handoff-fence-watchdog", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self._poll_interval + 1.0)

    def check(self) -> None:
        """Raise if the lease failed closed; otherwise return."""

        self.poll_once()
        if self.fenced:
            raise HandoffFencedError(
                "single GPU lease failed closed "
                f"({self.fenced_state}): {self.fence_reason}"
            )

    def poll_once(self) -> bool:
        """Read the durable lease once; return true when it is fenced."""

        if self.fenced:
            return True
        try:
            record = read_durable_lease(self._lease_path)
        except (OSError, ValueError, RuntimeError):
            # Missing or mid-replace records are not evidence of a fence.
            return False
        state = record.get("state")
        try:
            lease_state = LeaseState(state)
        except ValueError:
            return False
        if lease_state not in FAILURE_STATES:
            return False
        reason = "unrecorded"
        audit = record.get("audit")
        if isinstance(audit, list):
            for entry in reversed(audit):
                if not isinstance(entry, dict):
                    continue
                if entry.get("to_state") != lease_state.value:
                    continue
                entry_reason = entry.get("reason")
                if isinstance(entry_reason, str) and entry_reason.strip():
                    reason = entry_reason.strip()
                break
        with self._lock:
            if self.fenced_state is None:
                self.fenced_state = lease_state.value
                self.fence_reason = reason
        if self._logger is not None:
            self._logger.error(
                "Single GPU lease failed closed (%s): %s; aborting the Work "
                "exec without resuming the paused container",
                lease_state.value,
                reason,
            )
        self.exec_stop_event.set()
        return True

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._shutdown_event is not None and self._shutdown_event.is_set():
                self.exec_stop_event.set()
                return
            if self.poll_once():
                return
            self._stop.wait(self._poll_interval)


@dataclass(frozen=True)
class _DeadlineHandoffResult:
    """Final artifacts captured and Work retired inside a timeout callback."""

    archive: bytes
    snapshot: dict[str, object]


def _ensure_version_store_backend_supported(
    backend: ContainerBackend,
    *,
    version_saving_enabled: bool,
) -> None:
    """Require the backend that provides the immutable final-state boundary."""

    if version_saving_enabled and backend.backend_name != "docker":
        raise FinalSnapshotError(
            "Persistent submission versions currently require the Docker "
            "frozen-final-snapshot contract"
        )


# ---------------------------------------------------------------------------
# Environment and command helpers
# ---------------------------------------------------------------------------


def _build_agent_env(
    agent: Agent,
    model: str | None = None,
) -> dict[str, str]:
    """Build environment variables dict for an agent container."""
    config = agent._config
    env: dict[str, str] = {}

    if config.http_proxy:
        env["http_proxy"] = config.http_proxy
        env["HTTP_PROXY"] = config.http_proxy
    if config.https_proxy:
        env["https_proxy"] = config.https_proxy
        env["HTTPS_PROXY"] = config.https_proxy
    if config.no_proxy:
        env["no_proxy"] = config.no_proxy
        env["NO_PROXY"] = config.no_proxy

    if config.agent_api_key:
        env[agent.api_key_env] = config.agent_api_key

    if config.agent_api_base_url and agent.api_base_env:
        env[agent.api_base_env] = config.agent_api_base_url

    effective_model = model or config.agent_model or agent.default_model
    if effective_model and agent.model_env:
        env[agent.model_env] = effective_model

    if config.nodejs_mirror_url:
        env["SFORGE_NODEJS_MIRROR_URL"] = config.nodejs_mirror_url
    if config.npm_registry_url:
        env["npm_config_registry"] = config.npm_registry_url

    env.update(config.agent_extra_env)

    agent.augment_env(env, model)

    return env


def _configure_submission_service_env(
    env: dict[str, str],
    *,
    judge_url: str,
    session_token: str,
    max_submissions: int | None,
    version_saving_enabled: bool,
) -> None:
    """Expose the service when evaluation or free checkpoints are available."""
    if max_submissions == 0 and not version_saving_enabled:
        return
    env["SFORGE_JUDGE_URL"] = judge_url
    env["SFORGE_TOKEN"] = session_token
    judge_host = urlparse(judge_url).hostname or ""
    for key in ("NO_PROXY", "no_proxy"):
        existing = env.get(key, "")
        if judge_host and judge_host not in existing:
            env[key] = f"{existing},{judge_host}" if existing else judge_host


def _select_work_session_token(
    registration: dict[str, object],
    *,
    max_submissions: int | None,
    version_saving_enabled: bool,
) -> str:
    """Select the least-privileged token exposed to the Work container."""

    token_key = (
        "checkpoint_token"
        if max_submissions == 0 and version_saving_enabled
        else "token"
    )
    token = registration.get(token_key)
    if not isinstance(token, str) or not token:
        if token_key == "checkpoint_token":
            raise JudgeRegistrationError(
                "Judge registration omitted checkpoint-only token"
            )
        raise JudgeRegistrationError("Judge registration omitted session token")
    return token


def _validate_version_store_registration(
    registration: dict[str, object],
    version_store: VersionStore | None,
) -> None:
    """Fail closed when run and Judge processes name different stores."""

    if version_store is None:
        return
    if (
        registration.get("version_saving_enabled") is not True
        or registration.get("version_store_id") != version_store.store_id
    ):
        raise JudgeRegistrationError("Judge version store mismatch")


# ---------------------------------------------------------------------------
# Container setup helpers
# ---------------------------------------------------------------------------


def _install_tools(
    backend: ContainerBackend,
    handle: ContainerHandle,
    task_spec: TaskSpec,
    agent: Agent,
    log_dir: Path,
    logger,
    disable_stop_hook: bool = False,
    install_submit_tool: bool = True,
    install_save_version_tool: bool = False,
) -> None:
    """Install the enabled work-session tools and agent stop hooks."""

    # 1. Install sforge-submit script
    if install_submit_tool:
        submit_script = generate_submit_script()
        local_submit = log_dir / "_sforge-submit.sh"
        local_submit.write_text(submit_script)
        copy_container_file(
            backend,
            handle,
            local_submit,
            PurePosixPath("/usr/local/bin/sforge-submit"),
            owner="root:root",
            mode=0o755,
            description="sforge-submit",
        )
        logger.info("Installed sforge-submit script")
    else:
        logger.info("Submission tool omitted for fixed work session")

    if install_save_version_tool:
        save_script = generate_save_version_script()
        local_save = log_dir / "_sforge-save-version.sh"
        local_save.write_text(save_script)
        copy_container_file(
            backend,
            handle,
            local_save,
            PurePosixPath("/usr/local/bin/sforge-save-version"),
            owner="root:root",
            mode=0o755,
            description="sforge-save-version",
        )
        logger.info("Installed sforge-save-version script")

    # 2. Install agent-specific stop hook (unless disabled)
    if not disable_stop_hook:
        agent.install_stop_hook(backend, handle, log_dir, logger)
    else:
        logger.info("Stop hook disabled by flag")


def _extract_archive_from_container(
    backend: ContainerBackend,
    handle: ContainerHandle,
    task_spec: TaskSpec,
    agent: Agent,
) -> bytes:
    """Extract the current submission archive from a running work container."""
    patch_dir = task_spec.cwd
    submit_paths = " ".join(task_spec.submit_paths)
    excludes = " ".join(f"--exclude={e}" for e in task_spec.submit_exclude)
    tar_cmd = (
        f"cd {patch_dir} && tar czf /tmp/final.tar.gz "
        f"--exclude=.git {excludes} {submit_paths}"
    )
    result = backend.exec_run(
        handle,
        ["/bin/bash", "-c", f"rm -f /tmp/final.tar.gz && {tar_cmd}"],
    )
    if result.exit_code != 0:
        raise RuntimeError(
            f"submission archive creation failed: {result.output}"
        )
    raw = backend.copy_from_container(handle, PurePosixPath("/tmp/final.tar.gz"))
    outer = _tarfile.open(fileobj=io.BytesIO(raw))
    member = outer.getmembers()[0]
    archive = outer.extractfile(member).read()
    outer.close()
    archive = _normalize_live_submission_archive(archive)
    agent.validate_submission_archive(archive)
    return archive


def _safe_relative_path(raw_path: str, *, allow_dot: bool = False) -> PurePosixPath:
    """Normalize one task/archive path without allowing host traversal."""
    path = PurePosixPath(raw_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe relative path: {raw_path!r}")
    parts = tuple(part for part in path.parts if part not in ("", "."))
    if not parts:
        if allow_dot:
            return PurePosixPath(".")
        raise ValueError(f"empty relative path: {raw_path!r}")
    return PurePosixPath(*parts)


def _normalize_live_submission_archive(archive: bytes) -> bytes:
    """Canonicalize GNU tar's cwd-relative spellings before Judge upload.

    The running-container snapshot path uses ``tar ... .`` for tasks whose
    submit path is the whole working directory.  GNU tar then emits a wrapper
    member named ``./`` and prefixes descendants with ``./``.  Those names are
    safe but intentionally rejected by the Judge trust boundary, which only
    accepts already-canonical member names.  Remove only that conventional
    leading prefix here; absolute paths, traversal, internal dot components,
    duplicates, and malformed names remain hard failures.

    The immutable final-snapshot path already performs equivalent name
    remapping while copying from the stopped container.
    """
    raw_archive = io.BytesIO()
    emitted: set[str] = set()
    with _tarfile.open(fileobj=io.BytesIO(archive), mode="r:*") as source:
        with _tarfile.open(fileobj=raw_archive, mode="w") as output:
            for member in source:
                raw_name = member.name
                if not isinstance(raw_name, str) or not raw_name or "\x00" in raw_name:
                    raise ValueError("submission archive member name is invalid")
                if raw_name.startswith("/"):
                    raise ValueError("submission archive contains an absolute path")

                name = raw_name
                while name.startswith("./"):
                    name = name[2:]
                name = name[:-1] if name.endswith("/") else name
                if name in {"", "."}:
                    if member.isdir():
                        continue
                    raise ValueError("submission archive contains an empty path")
                components = name.split("/")
                if any(component in {"", ".", ".."} for component in components):
                    raise ValueError(
                        "submission archive member path is not canonical"
                    )
                canonical = str(PurePosixPath(*components))
                if canonical != name:
                    raise ValueError(
                        "submission archive member path changed during normalization"
                    )
                if canonical in emitted:
                    raise ValueError(
                        "submission archive contains a duplicate canonical path"
                    )

                info = copy.copy(member)
                info.name = canonical
                info.pax_headers = {
                    key: value
                    for key, value in member.pax_headers.items()
                    if key not in {"path", "linkpath"}
                }
                payload = source.extractfile(member) if member.isfile() else None
                if member.isfile() and payload is None:
                    raise ValueError(
                        f"unreadable submission archive member: {raw_name!r}"
                    )
                output.addfile(info, payload)
                emitted.add(canonical)

    return gzip.compress(raw_archive.getvalue(), compresslevel=9, mtime=0)


def _submission_member_is_excluded(
    member_name: str,
    submit_exclude: list[str],
) -> bool:
    """Apply the task blacklist plus the harness's unconditional .git rule."""
    path = _safe_relative_path(member_name)
    path_text = str(path)
    components = list(path.parts)
    for raw_pattern in [".git", *submit_exclude]:
        pattern = raw_pattern.rstrip("/")
        if not pattern:
            continue
        if path_text == pattern or path_text.startswith(pattern + "/"):
            return True
        if fnmatch.fnmatch(path_text, pattern):
            return True
        # GNU tar's unanchored basename patterns exclude matching directory
        # components too (not just the final basename).
        if any(fnmatch.fnmatch(component, pattern) for component in components):
            return True
    return False


def _safe_symlink_target(
    member_name: str,
    link_name: str,
    submission_root: tuple[str, ...],
) -> None:
    """Reject archive links that could escape the submitted tree."""
    link = PurePosixPath(link_name)
    if link.is_absolute():
        raise ValueError(
            f"absolute symlink in submission: {member_name!r} -> {link_name!r}"
        )
    resolved = list(PurePosixPath(member_name).parent.parts)
    for part in link.parts:
        if part in ("", "."):
            continue
        if part == "..":
            if len(resolved) <= len(submission_root):
                raise ValueError(
                    "symlink escapes submission tree: "
                    f"{member_name!r} -> {link_name!r}"
                )
            resolved.pop()
        else:
            resolved.append(part)
    if tuple(resolved[: len(submission_root)]) != submission_root:
        raise ValueError(
            "symlink escapes submission tree: "
            f"{member_name!r} -> {link_name!r}"
        )


def _mapped_transport_member_name(
    member_name: str,
    transport_root: str,
    submit_path: PurePosixPath,
) -> str | None:
    """Map Docker get_archive's basename-rooted tar into cwd-relative names."""
    transport_name = _safe_relative_path(member_name, allow_dot=True)
    transport_parts = tuple(
        part for part in transport_name.parts if part not in ("", ".")
    )
    if not transport_parts or transport_parts[0] != transport_root:
        raise ValueError(
            "container archive root mismatch: "
            f"expected {transport_root!r}, got {member_name!r}"
        )

    descendant_parts = transport_parts[1:]
    if str(submit_path) == ".":
        destination_parts = descendant_parts
    else:
        destination_parts = (*submit_path.parts, *descendant_parts)
    if not destination_parts:
        # The transport root for submit_path='.' is only a wrapper around cwd.
        return None
    return str(PurePosixPath(*destination_parts))


def _extract_archive_from_stopped_container(
    backend: ContainerBackend,
    handle: ContainerHandle,
    task_spec: TaskSpec,
    agent: Agent,
) -> bytes:
    """Build a submission tar locally from a verified stopped container.

    No command is executed in the Work container.  Each source path is read
    through the backend's filesystem-copy API, remapped to cwd-relative archive
    names, and validated before it is added to the normalized gzip stream.
    """
    cwd = PurePosixPath(task_spec.cwd)
    if not cwd.is_absolute() or ".." in cwd.parts or cwd.name in ("", ".", ".."):
        raise ValueError(f"task cwd must be a safe non-root absolute path: {cwd}")

    raw_archive = io.BytesIO()
    emitted: set[str] = set()
    with _tarfile.open(fileobj=raw_archive, mode="w") as output:
        for raw_submit_path in task_spec.submit_paths:
            submit_path = _safe_relative_path(
                raw_submit_path,
                allow_dot=True,
            )
            source = cwd if str(submit_path) == "." else cwd / submit_path
            transport_root = source.name

            backend.verify_container_frozen_for_snapshot(handle)
            transport_bytes = backend.copy_from_container(handle, source)
            backend.verify_container_frozen_for_snapshot(handle)
            with _tarfile.open(
                fileobj=io.BytesIO(transport_bytes),
                mode="r:*",
            ) as transport:
                members = transport.getmembers()
                if not members:
                    raise ValueError(
                        f"empty container archive for submit path {raw_submit_path!r}"
                    )
                for member in members:
                    destination_name = _mapped_transport_member_name(
                        member.name,
                        transport_root,
                        submit_path,
                    )
                    if destination_name is None:
                        continue
                    if _submission_member_is_excluded(
                        destination_name,
                        task_spec.submit_exclude,
                    ):
                        continue
                    if destination_name in emitted:
                        # Overlapping submit_paths commonly produce the same
                        # member.  The first stopped-filesystem read wins.
                        continue

                    info = copy.copy(member)
                    info.name = destination_name
                    info.pax_headers = {
                        key: value
                        for key, value in member.pax_headers.items()
                        if key not in {"path", "linkpath"}
                    }
                    payload = None
                    if member.isfile():
                        payload = transport.extractfile(member)
                        if payload is None:
                            raise ValueError(
                                f"unreadable archive member: {member.name!r}"
                            )
                    elif member.islnk():
                        # Materialize hard links so linkname remapping cannot
                        # point outside the locally built submission archive.
                        payload = transport.extractfile(member)
                        if payload is None:
                            raise ValueError(
                                f"unreadable hard link: {member.name!r}"
                            )
                        data = payload.read()
                        payload = io.BytesIO(data)
                        info.type = _tarfile.REGTYPE
                        info.linkname = ""
                        info.size = len(data)
                    elif member.issym():
                        _safe_symlink_target(
                            destination_name,
                            member.linkname,
                            (
                                ()
                                if str(submit_path) == "."
                                else tuple(submit_path.parts)
                            ),
                        )
                    elif not member.isdir():
                        raise ValueError(
                            "unsupported special file in submission: "
                            f"{member.name!r}"
                        )

                    output.addfile(info, payload)
                    emitted.add(destination_name)

    archive = gzip.compress(
        raw_archive.getvalue(),
        compresslevel=9,
        mtime=0,
    )
    agent.validate_submission_archive(archive)
    return archive


def _read_file_from_container_archive(
    backend: ContainerBackend,
    handle: ContainerHandle,
    source: PurePosixPath,
) -> bytes:
    """Read exactly one regular file from a backend transport tar."""
    transport = backend.copy_from_container(handle, source)
    with _tarfile.open(fileobj=io.BytesIO(transport), mode="r:*") as wrapper:
        members = wrapper.getmembers()
        if len(members) != 1 or not members[0].isfile():
            raise ValueError(
                f"container transport for {source} is not exactly one file"
            )
        if PurePosixPath(members[0].name).name != source.name:
            raise ValueError(f"container transport name mismatch for {source}")
        stream = wrapper.extractfile(members[0])
        if stream is None:
            raise ValueError(f"container transport is unreadable for {source}")
        payload = stream.read()
        if len(payload) != members[0].size:
            raise ValueError(f"container transport is truncated for {source}")
        return payload


def _scheduled_auto_eval_offsets(
    run_timeout: int,
    eval_interval: int,
) -> tuple[int, ...]:
    """Return fixed offsets strictly before the agent timeout boundary."""
    if run_timeout <= 0 or eval_interval <= 0:
        return ()
    return tuple(range(eval_interval, run_timeout, eval_interval))


def _remaining_run_timeout_seconds(
    run_deadline_monotonic: float,
    *,
    now_monotonic: float | None = None,
) -> float:
    """Return a segment timeout bounded by one absolute wall deadline."""
    if now_monotonic is None:
        now_monotonic = time.monotonic()
    remaining = run_deadline_monotonic - now_monotonic
    if remaining <= 0:
        return 0.0
    # Docker accepts fractional seconds here.  Rounding up would move the
    # whole-container freeze beyond the absolute candidate deadline.
    return remaining


def _max_resume_attempts(
    run_timeout: int,
    min_runtime_for_resume: float,
) -> int:
    """Place the resume safety cap beyond the deadline-bounded maximum."""
    if run_timeout <= 0 or min_runtime_for_resume <= 0:
        return 100
    return max(
        100,
        math.ceil(run_timeout / min_runtime_for_resume) + 1,
    )


def _agent_exec_command(agent: Agent, run_cmd: str) -> list[str]:
    """Run Codex in a dedicated session so Docker can stop its whole tree."""
    if agent.name == "codex":
        return [
            "setsid",
            "--wait",
            "/bin/bash",
            "-c",
            (
                'set -e; printf "%s\\n" "$$" '
                '>"$SFORGE_EXEC_PROCESS_GROUP_FILE"; '
                'exec /bin/bash -c "$1"'
            ),
            "sforge-codex-session",
            run_cmd,
        ]
    return ["/bin/bash", "-c", run_cmd]


def _stop_auto_eval_thread(
    stop_event: threading.Event | None,
    thread: threading.Thread | None,
    logger,
) -> None:
    """Stop and fully join the snapshot thread before touching final state."""
    if stop_event is None:
        return
    was_active = not stop_event.is_set() or (
        thread is not None and thread.is_alive()
    )
    stop_event.set()
    if thread is not None and thread is not threading.current_thread():
        thread.join()
    if was_active:
        logger.info("Auto-eval thread stopped")


def _join_threads(
    threads: list[threading.Thread],
) -> None:
    """Join tracked helper threads after their shared stop event is set."""
    current = threading.current_thread()
    for thread in list(threads):
        if thread is not current:
            thread.join()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Durably publish one artifact or leave the previous artifact intact."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _write_final_snapshot_metadata(
    log_dir: Path,
    run_started_monotonic: float,
    *,
    archive: bytes | None = None,
    freeze_metadata: dict[str, object] | None = None,
    run_deadline_monotonic: float | None = None,
    capture_started_at: float | None = None,
    capture_started_monotonic: float | None = None,
    seal_at: float | None = None,
    seal_monotonic: float | None = None,
) -> dict[str, object]:
    """Record when the immutable final archive finished being captured."""
    captured_monotonic = time.monotonic()
    captured_at = time.time()
    metadata: dict[str, object] = {
        "captured_at": captured_at,
        "snapshot_elapsed_seconds": max(
            0.0, captured_monotonic - run_started_monotonic
        ),
    }

    # Preserve the compact legacy payload when callers do not opt into the
    # stopped-container contract.  Existing third-party backends and tooling
    # can continue to use this helper unchanged.
    if archive is not None or freeze_metadata is not None:
        if archive is None or freeze_metadata is None:
            raise ValueError(
                "archive and freeze_metadata must be recorded together"
            )
        freeze_requested_monotonic = float(
            freeze_metadata["freeze_requested_monotonic"]
        )
        frozen_monotonic = float(freeze_metadata["frozen_at_monotonic"])
        capture_started_monotonic = (
            captured_monotonic
            if capture_started_monotonic is None
            else capture_started_monotonic
        )
        capture_started_at = (
            captured_at if capture_started_at is None else capture_started_at
        )
        if "archive_capture_started_monotonic" in freeze_metadata:
            capture_started_monotonic = float(
                freeze_metadata["archive_capture_started_monotonic"]
            )
            capture_started_at = float(
                freeze_metadata["archive_capture_started_at"]
            )
        deadline = (
            freeze_requested_monotonic
            if run_deadline_monotonic is None
            else run_deadline_monotonic
        )
        archive_sha256 = hashlib.sha256(archive).hexdigest()
        capture_contract: dict[str, object] = {
            "source": freeze_metadata.get(
                "archive_source", "stopped_container_filesystem"
            ),
            "transport": freeze_metadata.get(
                "archive_transport", "docker_get_archive_transport_tar"
            ),
            "work_container_running_during_capture": False,
            "started_at": capture_started_at,
            "started_elapsed_seconds": max(
                0.0,
                capture_started_monotonic - run_started_monotonic,
            ),
            "archive_completed_at": captured_at,
            "archive_completed_elapsed_seconds": max(
                0.0,
                captured_monotonic - run_started_monotonic,
            ),
        }
        if "archive_capture_work_paused" in freeze_metadata:
            source_completed_monotonic = float(
                freeze_metadata["archive_capture_completed_monotonic"]
            )
            capture_contract.update(
                {
                    "work_container_paused_during_capture": bool(
                        freeze_metadata["archive_capture_work_paused"]
                    ),
                    "source_transport_completed_at": float(
                        freeze_metadata["archive_capture_completed_at"]
                    ),
                    "source_transport_completed_elapsed_seconds": max(
                        0.0,
                        source_completed_monotonic - run_started_monotonic,
                    ),
                }
            )
        metadata.update(
            {
                "schema_version": 2,
                # For the strong contract this compatibility field denotes
                # the filesystem boundary, not later archive I/O completion.
                "snapshot_elapsed_seconds": max(
                    0.0, frozen_monotonic - run_started_monotonic
                ),
                "snapshot_boundary": "whole_container_frozen",
                "archive_sha256": archive_sha256,
                "deadline": {
                    "at": float(freeze_metadata["freeze_requested_at"])
                    - (freeze_requested_monotonic - deadline),
                    "elapsed_seconds": max(
                        0.0, deadline - run_started_monotonic
                    ),
                    "monotonic": deadline,
                },
                "freeze": {
                    "mechanism": freeze_metadata["mechanism"],
                    "all_work_processes_frozen": bool(
                        freeze_metadata.get(
                            "all_work_processes_frozen",
                            freeze_metadata["all_work_processes_stopped"],
                        )
                    ),
                    "all_work_processes_stopped": bool(
                        freeze_metadata["all_work_processes_stopped"]
                    ),
                    "was_running": bool(freeze_metadata["was_running"]),
                    "freeze_requested_at": float(
                        freeze_metadata["freeze_requested_at"]
                    ),
                    "freeze_requested_elapsed_seconds": max(
                        0.0,
                        freeze_requested_monotonic - run_started_monotonic,
                    ),
                    "frozen_at": float(freeze_metadata["frozen_at"]),
                    "frozen_elapsed_seconds": max(
                        0.0, frozen_monotonic - run_started_monotonic
                    ),
                    "container_status": freeze_metadata["container_status"],
                    "container_pid": freeze_metadata["container_pid"],
                    **seal_boundary_fields(
                        seal_at, seal_monotonic, run_started_monotonic
                    ),
                },
                "capture": capture_contract,
                "archive": {
                    "path": "final_archive.tar.gz",
                    "format": "tar+gzip",
                    "size_bytes": len(archive),
                    "sha256": archive_sha256,
                },
            }
        )

    _atomic_write_bytes(
        log_dir / "final_snapshot.json",
        (json.dumps(metadata, indent=2) + "\n").encode(),
    )
    return metadata


def _capture_frozen_final_snapshot(
    backend: ContainerBackend,
    handle: ContainerHandle,
    task_spec: TaskSpec,
    agent: Agent,
    log_dir: Path,
    run_started_monotonic: float,
    logger,
    *,
    run_deadline_monotonic: float | None,
    auto_eval_stop: threading.Event | None,
    auto_eval_thread: threading.Thread | None,
    feedback_threads: list[threading.Thread] | None = None,
    final_handoff_annotations: dict[str, str] | None = None,
    seal_at: float | None = None,
    seal_monotonic: float | None = None,
) -> tuple[bytes, dict[str, object]]:
    """Freeze all Work writers, then publish a verified final archive.

    The stop event is signalled before the backend call so no new host-side
    auto-eval work is admitted. Ordinarily the whole container is stopped.
    A single-GPU finalizer may instead adopt an already-paused exact Work
    container, but only after its durable ``never_resume`` fence was verified.
    """
    if auto_eval_stop is not None:
        auto_eval_stop.set()
    try:
        if final_handoff_annotations is None:
            freeze = backend.freeze_container_for_snapshot(
                handle,
                stop_without_inspect=True,
            )
        else:
            freeze = backend.freeze_container_for_final_handoff_snapshot(
                handle,
                expected_annotations=final_handoff_annotations,
                stop_without_inspect=True,
            )
        backend.verify_container_frozen_for_snapshot(handle)
    except Exception as exc:
        raise FinalSnapshotError(
            "Unable to prove whole-container freeze before final snapshot"
        ) from exc

    _stop_auto_eval_thread(auto_eval_stop, auto_eval_thread, logger)
    if feedback_threads:
        _join_threads(feedback_threads)

    capture_started_at = time.time()
    capture_started_monotonic = time.monotonic()
    try:
        backend.verify_container_frozen_for_snapshot(handle)
        archive = _extract_archive_from_stopped_container(
            backend,
            handle,
            task_spec,
            agent,
        )
        backend.verify_container_frozen_for_snapshot(handle)
        if freeze.get("container_paused") is True:
            freeze.update(
                {
                    "archive_capture_started_at": capture_started_at,
                    "archive_capture_started_monotonic": (
                        capture_started_monotonic
                    ),
                    "archive_capture_completed_at": time.time(),
                    "archive_capture_completed_monotonic": time.monotonic(),
                    "archive_capture_work_paused": True,
                }
            )
        _atomic_write_bytes(log_dir / "final_archive.tar.gz", archive)
        snapshot = _write_final_snapshot_metadata(
            log_dir,
            run_started_monotonic,
            archive=archive,
            freeze_metadata=freeze,
            run_deadline_monotonic=run_deadline_monotonic,
            capture_started_at=capture_started_at,
            capture_started_monotonic=capture_started_monotonic,
            seal_at=seal_at,
            seal_monotonic=seal_monotonic,
        )
    except FinalSnapshotError:
        raise
    except Exception as exc:
        raise FinalSnapshotError(
            "Stopped-container final snapshot capture failed"
        ) from exc

    return archive, snapshot


def _verify_post_snapshot_work_disposition(
    backend: ContainerBackend,
    handle: ContainerHandle,
    *,
    handoff_work_retired: bool,
) -> str:
    """Prove Work cannot write after capture without reopening retired Work."""

    if handoff_work_retired:
        # The coordinator already performed strict exact-ID destruction and a
        # post-destroy idle attestation.  Re-inspecting this stale handle would
        # incorrectly apply the pre-destroy paused-snapshot verifier.
        return "strictly retired"
    backend.verify_container_frozen_for_snapshot(handle)
    return "frozen"


def _persist_final_version(
    version_store: VersionStore | None,
    *,
    task_spec: TaskSpec,
    run_id: str,
    archive: bytes,
    agent_name: str,
    model: str | None,
    timed_out: bool,
    runtime_seconds: float,
    interrupted: bool,
) -> StoredVersion | None:
    """Persist the exact final snapshot bytes after local publication."""

    if version_store is None:
        return None
    try:
        return version_store.record_archive(
            task_id=task_spec.task_id,
            run_id=run_id,
            source="final",
            round_id="final",
            label=None,
            archive=archive,
            submit_paths=task_spec.submit_paths,
            submit_exclude=task_spec.submit_exclude,
            content_identity_exclude=task_spec.candidate_identity_exclude,
            metadata={
                "agent": agent_name,
                "interrupted": interrupted,
                "model": model,
                "runtime_seconds": runtime_seconds,
                "timed_out": timed_out,
            },
        )
    except Exception as exc:
        raise FinalVersionStoreError(
            "captured final version could not be persisted"
        ) from exc


def _judge_drain_timeout_seconds(
    eval_timeout: int,
    pending_count: int,
    *,
    margin: int = 60,
) -> int:
    """Allow every serialized pending evaluation one full timeout."""
    return max(900, max(1, pending_count) * eval_timeout + margin)


def _fetch_full_judge_history(
    host_judge_url: str,
    session_token: str,
) -> dict:
    """Fetch host-only history without putting the admin secret in the URL."""
    response = requests.get(
        f"{host_judge_url}/api/v1/history",
        params={"token": session_token},
        headers={ADMIN_SECRET_HEADER: ADMIN_SECRET},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def _fetch_required_final_judge_history(
    host_judge_url: str,
    session_token: str,
) -> dict:
    """Fetch the authoritative terminal history or fail the whole run."""
    try:
        return _fetch_full_judge_history(host_judge_url, session_token)
    except Exception as exc:
        raise JudgeDrainError(
            "Unable to fetch authoritative final Judge history"
        ) from exc


def _fetch_pending_judge_evaluations(
    host_judge_url: str,
    session_token: str,
) -> list[dict]:
    history = _fetch_full_judge_history(
        host_judge_url,
        session_token,
    )
    return [
        entry
        for entry in history.get("entries", [])
        if entry.get("status") in ("running", "queued")
    ]


def _drain_pending_judge_evaluations(
    host_judge_url: str,
    session_token: str,
    eval_timeout: int,
    logger,
    *,
    poll_interval: float = 15.0,
) -> None:
    """Wait for serialized judge work, failing if quiescence is uncertain."""
    try:
        pending = _fetch_pending_judge_evaluations(
            host_judge_url, session_token
        )
    except Exception as exc:
        raise JudgeDrainError(
            "Unable to inspect pending judge evaluations"
        ) from exc

    if not pending:
        return

    drain_timeout = _judge_drain_timeout_seconds(
        eval_timeout, len(pending)
    )
    drain_deadline = time.monotonic() + drain_timeout
    while pending:
        remaining = drain_deadline - time.monotonic()
        if remaining <= 0:
            raise JudgeDrainError(
                f"Judge drain timed out after {drain_timeout}s with "
                f"{len(pending)} evaluation(s) still pending"
            )
        logger.info(
            "Draining %d pending judge evals (deadline in %.0fs)...",
            len(pending),
            remaining,
        )
        time.sleep(min(poll_interval, remaining))
        try:
            pending = _fetch_pending_judge_evaluations(
                host_judge_url, session_token
            )
        except Exception as exc:
            logger.warning(
                "Failed to poll pending judge evaluations: %s", exc
            )


def _seal_drain_and_verify_final_handoff(
    fence: _FinalHandoffFence,
    *,
    host_judge_url: str,
    session_token: str,
    eval_timeout: int,
    lease_path: Path,
    expected_identity: dict[str, str],
    expected_allocation: AcceleratorAllocation,
    expected_contract_sha256: str,
    expected_annotations: dict[str, str],
    logger,
) -> tuple[float, float]:
    """Converge normal, timeout, and interrupt finalization before capture.

    Returns the wall-clock and monotonic time at which the seal (admission
    closed, Work paused, never-resume persisted) was requested.  That is the
    agent's true work boundary: a queued Judge may legitimately be drained
    afterwards, so the frozen snapshot can be later than the seal.
    """

    seal_at = time.time()
    seal_monotonic = time.monotonic()
    fence.seal()
    _drain_pending_judge_evaluations(
        host_judge_url,
        session_token,
        eval_timeout,
        logger,
    )
    durable_fence = read_durable_lease(lease_path)
    expected_allocation_value = {
        "vendor": expected_allocation.vendor,
        "device_ids": list(expected_allocation.device_ids),
    }
    if (
        durable_fence.get("recovery_policy") != "never_resume"
        or durable_fence.get("final_requested") is not True
        or durable_fence.get("state") != "FINALIZING"
        or durable_fence.get("identity") != expected_identity
        or durable_fence.get("allocation") != expected_allocation_value
        or durable_fence.get("evaluation_contract_sha256")
        != expected_contract_sha256
        or durable_fence.get("annotations") != expected_annotations
    ):
        raise FinalSnapshotError(
            "final Work freeze lacks an exact durable FINALIZING no-resume binding"
        )
    return seal_at, seal_monotonic


def seal_boundary_fields(
    seal_at: float | None,
    seal_monotonic: float | None,
    run_started_monotonic: float,
) -> dict[str, float]:
    """Snapshot ``freeze`` fields recording when the single-GPU seal happened."""
    if seal_at is None or seal_monotonic is None:
        return {}
    return {
        "seal_requested_at": float(seal_at),
        "seal_requested_elapsed_seconds": max(
            0.0, float(seal_monotonic) - float(run_started_monotonic)
        ),
    }


def _retire_final_handoff_work(
    *,
    host_judge_url: str,
    session_token: str,
    archive: bytes,
) -> None:
    """Ask the coordinator to destroy exact Work and prove the GPU idle."""

    if not archive:
        raise FinalSnapshotError(
            "single GPU final Work cannot retire without a durable archive"
        )
    response = requests.post(
        f"{host_judge_url}/api/v1/handoff/final-work-archive",
        json={
            "token": session_token,
            "admin_secret": ADMIN_SECRET,
            "archive_sha256": "sha256:" + hashlib.sha256(archive).hexdigest(),
            "archive_size": len(archive),
        },
        timeout=30,
    )
    response.raise_for_status()
    if response.json().get("state") != "FINAL_GPU_IDLE_ATTESTED":
        raise FinalSnapshotError(
            "Judge did not attest idle after strict final Work destruction"
        )


def _safe_debug_output_warnings(report: object) -> list[dict]:
    """Extract only non-secret, allowlisted source diagnostics from a report."""
    if not isinstance(report, dict):
        return []
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        return []
    raw = metrics.get("debug_output_warnings")
    if not isinstance(raw, list):
        return []

    safe: list[dict] = []
    for item in raw[:MAX_SAFE_DEBUG_WARNINGS]:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        line = item.get("line")
        category = item.get("category")
        if (
            not isinstance(path, str)
            or len(path) > 255
            or not path.startswith("kernels/")
            or "\n" in path
            or "\r" in path
        ):
            continue
        pure = PurePosixPath(path)
        if (
            pure.is_absolute()
            or ".." in pure.parts
            or pure.suffix not in SAFE_DEBUG_WARNING_SUFFIXES
        ):
            continue
        if (
            category != "debug_output"
            or isinstance(line, bool)
            or not isinstance(line, int)
            or not 1 <= line <= 10_000_000
        ):
            continue
        safe.append(
            {
                "category": "debug_output",
                "path": path,
                "line": line,
                "message": "Non-fatal debug output; hidden evaluation continued.",
            }
        )
    return safe


def _safe_auto_score_feedback(report: object) -> dict | None:
    """Extract an explicitly opted-in, bounded primary-score summary."""
    if not isinstance(report, dict):
        return None
    metrics = report.get("metrics")
    if (
        not isinstance(metrics, dict)
        or metrics.get("auto_feedback_safe") is not True
    ):
        return None

    primary_name = metrics.get("primary_score_name")
    primary_unit = metrics.get("primary_score_unit")
    primary_formula = metrics.get("primary_score_formula")
    score = report.get("score")
    if (
        not isinstance(primary_name, str)
        or not primary_name
        or len(primary_name) > 120
        or not isinstance(primary_unit, str)
        or len(primary_unit) > 120
        or not isinstance(primary_formula, str)
        or len(primary_formula) > 300
        or isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
    ):
        return None

    valid = report.get("valid") is True
    if valid:
        summary = (
            "CORRECTNESS BINARY GATE PASSED; the reported primary score is "
            "eligible for best-round comparison."
        )
    else:
        summary = (
            "CORRECTNESS BINARY GATE FAILED; the primary score is zero. "
            "Use a manual Judge submission for bounded phase-level diagnostics."
        )
    return {
        "valid": valid,
        "score": float(score),
        "primary_score_name": primary_name,
        "primary_score_unit": primary_unit,
        "primary_score_formula": primary_formula,
        "summary": summary,
        "secondary_metrics_omitted": True,
    }


def _forward_safe_auto_eval_feedback(
    backend: ContainerBackend,
    handle: ContainerHandle,
    host_judge_url: str,
    submission_id: str,
    round_id: str,
    stop_event: threading.Event,
    logger,
    log_dir: Path,
    *,
    timeout: float,
) -> None:
    """Copy only explicitly allowlisted auto-eval feedback into the workspace."""
    deadline = time.monotonic() + max(1.0, timeout)
    while not stop_event.is_set() and time.monotonic() < deadline:
        try:
            response = requests.get(
                f"{host_judge_url}/api/v1/result/{submission_id}",
                timeout=10,
            )
            response.raise_for_status()
            result = response.json()
            status = result.get("status")
            if status == "completed":
                report = result.get("report")
                if not isinstance(report, dict):
                    return
                metrics = report.get("metrics")
                if (
                    isinstance(metrics, dict)
                    and "debug_output_warnings" in metrics
                ):
                    payload = {
                        "schema_version": 1,
                        "round_id": round_id,
                        "debug_output_warnings": (
                            _safe_debug_output_warnings(report)
                        ),
                    }
                    local_feedback = (
                        log_dir / "latest_auto_eval_admission_feedback.json"
                    )
                    local_feedback.write_text(
                        json.dumps(
                            payload,
                            indent=2,
                            ensure_ascii=False,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    if stop_event.is_set():
                        return
                    copy_container_file(
                        backend,
                        handle,
                        local_feedback,
                        AUTO_EVAL_ADMISSION_FEEDBACK_PATH,
                        owner="agent:agent",
                        mode=0o600,
                        description="automatic evaluation admission feedback",
                    )

                score_feedback = _safe_auto_score_feedback(report)
                if score_feedback is not None:
                    score_feedback.update(
                        {
                            "schema_version": 1,
                            "round": round_id,
                        }
                    )
                    local_score_feedback = (
                        log_dir / "latest_auto_eval_score_feedback.json"
                    )
                    local_score_feedback.write_text(
                        json.dumps(
                            score_feedback,
                            indent=2,
                            ensure_ascii=False,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    if stop_event.is_set():
                        return
                    copy_container_file(
                        backend,
                        handle,
                        local_score_feedback,
                        AUTO_EVAL_SCORE_FEEDBACK_PATH,
                        owner="agent:agent",
                        mode=0o600,
                        description="automatic evaluation score feedback",
                    )
                return
            if status == "error":
                return
        except Exception as exc:
            logger.debug(
                "Unable to forward safe auto-eval feedback for %s: %s",
                submission_id,
                exc,
            )
        stop_event.wait(2.0)


def _auto_eval_loop(
    backend: ContainerBackend,
    handle: ContainerHandle,
    task_spec: TaskSpec,
    agent: Agent,
    host_judge_url: str,
    session_token: str,
    eval_interval: int,
    run_started_monotonic: float,
    run_timeout: int,
    stop_event: threading.Event,
    logger,
    log_dir: Path,
    feedback_threads: list[threading.Thread] | None = None,
) -> None:
    """Host-side auto-eval: periodically extract code and submit to judge.

    Runs as a daemon thread. Fire-and-forget — does not poll for results.
    Writes tick entries to auto_eval_ticks.log for post-mortem inspection.
    """
    ticks_log = log_dir / "auto_eval_ticks.log"
    run_deadline = run_started_monotonic + run_timeout
    for offset in _scheduled_auto_eval_offsets(run_timeout, eval_interval):
        tick_at = run_started_monotonic + offset
        if stop_event.wait(max(0.0, tick_at - time.monotonic())):
            break
        # The strict schedule above excludes an exact boundary tick. Keep this
        # guard as a fail-safe against delayed wakeups and monotonic clock jumps.
        if stop_event.is_set() or time.monotonic() >= run_deadline:
            break
        try:
            archive = _extract_archive_from_container(
                backend, handle, task_spec, agent
            )
            if stop_event.is_set() or time.monotonic() >= run_deadline:
                break
            resp = requests.post(
                f"{host_judge_url}/api/v1/submit",
                data=archive,
                headers={
                    "Content-Type": SUBMISSION_ARCHIVE_MEDIA_TYPE,
                    SESSION_TOKEN_HEADER: session_token,
                    SUBMISSION_KIND_HEADER: "auto",
                    ADMIN_SECRET_HEADER: ADMIN_SECRET,
                },
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            ts = time.strftime("%Y-%m-%dT%H:%M:%S")
            with open(ticks_log, "a") as f:
                f.write(f"[{ts}] submitted {len(archive)} bytes -> {data.get('submission_id', '?')} round={data.get('round_id', '?')}\n")
            submission_id = data.get("submission_id")
            round_id = data.get("round_id")
            judge_spec = getattr(task_spec, "judge", None)
            if (
                isinstance(submission_id, str)
                and isinstance(round_id, str)
                and judge_spec is not None
            ):
                feedback_thread = threading.Thread(
                    target=_forward_safe_auto_eval_feedback,
                    args=(
                        backend,
                        handle,
                        host_judge_url,
                        submission_id,
                        round_id,
                        stop_event,
                        logger,
                        log_dir,
                    ),
                    kwargs={
                        "timeout": min(
                            float(judge_spec.eval_timeout) + 60.0,
                            float(eval_interval),
                        )
                    },
                    daemon=True,
                )
                if feedback_threads is not None:
                    feedback_threads.append(feedback_thread)
                feedback_thread.start()
        except Exception as e:
            ts = time.strftime("%Y-%m-%dT%H:%M:%S")
            try:
                with open(ticks_log, "a") as f:
                    f.write(f"[{ts}] error: {e}\n")
            except Exception:
                pass
            logger.debug("Auto-eval tick failed: %s", e)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_agent(
    task_spec: TaskSpec,
    agent: Agent,
    config: SForgeConfig,
    backend: ContainerBackend,
    run_id: str,
    model: str | None = None,
    timeout: int | None = None,
    judge_url: str = "http://host.docker.internal:8080",
    eval_interval: int = 300,
    disable_stop_hook: bool = False,
    disable_auto_eval: bool = False,
    disable_auto_resume: bool = False,
    internet: bool = True,
    verbose: bool = False,
    shutdown_event: threading.Event | None = None,
    max_submissions: int | None = None,
    submission_cooldown: int | None = None,
) -> RunResult:
    """Run an agent on a task with iterative evaluation.

    Requires a running judge server (started via `sforge serve`).

    1. Build work + judge images
    2. Create work container (with judge URL in env)
    3. Install agent + tools (submit script, stop hook, auto-eval daemon)
    4. Run agent (high max-turns for iterative work)
    5. Read final state, extract archive
    """
    validate_single_gpu_handoff_config(config)
    handoff_enabled = config.single_gpu_handoff
    accelerator_allocation: AcceleratorAllocation | None = None
    trusted_lease_id: str | None = None
    evaluation_contract_sha256: str | None = None
    work_annotations: dict[str, str] | None = None
    if handoff_enabled:
        if backend.backend_name != "docker":
            raise ValueError("single GPU handoff requires the Docker backend")
        if task_spec.game_mode:
            raise ValueError("single GPU handoff does not support game tasks")
        if not disable_auto_eval:
            raise ValueError(
                "single GPU MVP requires auto-eval to be disabled"
            )
        if max_submissions == 0:
            raise ValueError(
                "single GPU MVP requires manual submission evaluation"
            )
        if task_spec.portable is None:
            raise ValueError(
                "single GPU handoff requires a portable evaluation contract"
            )
        accelerator_allocation = AcceleratorAllocation(
            "nvidia", (config.accelerator_device_uuid or "",)
        )
        trusted_lease_id = config.trusted_lease_id
        evaluation_contract_sha256 = _portable_evaluation_contract_sha256(
            task_spec
        )
        work_annotations = handoff_annotations(
            run_id=run_id,
            trusted_lease_id=trusted_lease_id,
            allocation_key=config.accelerator_allocation_key or "",
            evaluation_contract_sha256=evaluation_contract_sha256,
        )
    effective_timeout = timeout or config.agent_timeout or agent.timeout

    version_store = None
    if config.version_store_root is not None:
        configured_store = VersionStore.open(config.version_store_root)
        if configured_store.supports_task(task_spec.task_id):
            version_store = configured_store
    version_saving_enabled = version_store is not None
    _ensure_version_store_backend_supported(
        backend,
        version_saving_enabled=version_saving_enabled,
    )

    # Host-provided agent runtime: the task image stays agent-free and the
    # CLI (kind and version) is selected at launch, not at image build.
    agent_runtime = None
    if config.agent_runtime_dir is not None:
        agent_runtime = load_agent_runtime(config.agent_runtime_dir)
        if agent_runtime.agent != agent.name:
            raise AgentRuntimeError(
                f"agent runtime is for {agent_runtime.agent!r}, "
                f"but this run uses agent {agent.name!r}"
            )
        if backend.backend_name != "docker":
            raise AgentRuntimeError(
                "agent runtime mounts require the docker backend"
            )

    log_dir = config.log_dir / "runs" / run_id / task_spec.task_id
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(
        f"agent.{task_spec.task_id}.{run_id}",
        log_dir / "run_agent.log",
        verbose=verbose,
    )
    if version_saving_enabled:
        logger.info(
            "Persistent version saving enabled (store %s)",
            version_store.store_id,
        )
    elif config.version_store_root is not None:
        logger.warning(
            "Persistent version saving DISABLED: task %s is not registered "
            "in the version store at %s; this run will keep no replayable "
            "candidates",
            task_spec.task_id,
            config.version_store_root,
        )
    else:
        logger.warning(
            "Persistent version saving DISABLED: no version store is "
            "configured (set SFORGE_VERSION_STORE_ROOT); this run will keep "
            "no replayable candidates"
        )

    # judge_url is for container use; derive a host-local URL for registration/polling
    parsed_judge = urlparse(judge_url)
    if parsed_judge.hostname == "host.docker.internal":
        host_judge_url = judge_url.replace("host.docker.internal", "127.0.0.1")
    elif backend.backend_name == "k8s":
        # judge_url points to VPC IP (for pods); host talks to judge server locally
        host_judge_url = f"http://127.0.0.1:{parsed_judge.port or 8080}"
    else:
        host_judge_url = judge_url

    handle = None
    net_isolation = None
    api_proxy = None
    install_parts: list[str] = []
    auto_eval_stop: threading.Event | None = None
    auto_eval_thread: threading.Thread | None = None
    auto_eval_feedback_threads: list[threading.Thread] = []
    started_monotonic: float | None = None
    run_deadline_monotonic: float | None = None
    resume_count = 0
    session_token: str | None = None
    final_handoff_fence: _FinalHandoffFence | None = None
    fence_watchdog: _HandoffFenceWatchdog | None = None
    handoff_work_retired = False
    deadline_handoff_result: _DeadlineHandoffResult | None = None
    integrity_review_notice = False
    # Extra Judge-drain allowance: a queued submission may first spend up to
    # this long in the server's integrity review.
    review_allowance = 0

    try:
        # 0. Clean up stale iptables chains from previous runs that were killed
        if backend.backend_name == "docker":
            from sforge.harness.network_isolation import cleanup_stale_chains
            cleanup_stale_chains(logger)

        # 1. Check images exist (must run `sforge build` first)
        for image_key in (task_spec.work_image_key, task_spec.judge_image_key):
            if not backend.image_exists(image_key):
                raise RuntimeError(
                    f"Image '{image_key}' not found. Run `sforge pull --task {task_spec.task_id}` to fetch from registry, or `sforge build --task {task_spec.task_id}` to build locally."
                )
        logger.info("Images ready")

        # 1b. Register session with judge server
        reg_body: dict = {"task_id": task_spec.task_id, "run_id": run_id, "admin_secret": ADMIN_SECRET}
        if config.judge_cpu_limit is not None:
            reg_body["judge_cpu_limit"] = config.judge_cpu_limit
        if config.judge_mem_limit is not None:
            reg_body["judge_mem_limit"] = config.judge_mem_limit
        if max_submissions is not None:
            reg_body["max_agent_submissions"] = max_submissions
        if submission_cooldown is not None:
            reg_body["submission_cooldown"] = submission_cooldown
        if max_submissions == 0 and version_saving_enabled:
            reg_body["issue_checkpoint_token"] = True
        if version_store is not None:
            reg_body["require_version_saving"] = True
            reg_body["required_version_store_id"] = version_store.store_id
        if backend.backend_name != "docker":
            reg_body["backend"] = backend.backend_name
            reg_body["k8s_image_registry"] = config.k8s_image_registry
            reg_body["k8s_namespace"] = config.k8s_namespace
            if config.k8s_node_selector:
                reg_body["k8s_node_selector"] = config.k8s_node_selector
            if config.k8s_kubeconfig:
                reg_body["k8s_kubeconfig"] = config.k8s_kubeconfig
        if handoff_enabled:
            reg_body.update(
                {
                    "single_gpu_handoff": True,
                    "trusted_lease_id": trusted_lease_id,
                    "allocation_key": config.accelerator_allocation_key,
                    "accelerator_device_ids": list(
                        accelerator_allocation.device_ids
                    ),
                    "evaluation_contract_sha256": evaluation_contract_sha256,
                }
            )
        for _reg_attempt in range(1, 6):
            try:
                reg_resp = requests.post(
                    f"{host_judge_url}/api/v1/register",
                    json=reg_body,
                    timeout=10,
                )
                reg_resp.raise_for_status()
                break
            except requests.RequestException as exc:
                if _reg_attempt == 5:
                    raise JudgeRegistrationError(
                        "Judge session registration failed after 5 attempts"
                    ) from exc
                wait = 2 * _reg_attempt
                logger.warning("Register attempt %d/5 failed (%s), retrying in %ds...", _reg_attempt, exc, wait)
                time.sleep(wait)
        try:
            registration = reg_resp.json()
            if not isinstance(registration, dict):
                raise TypeError("registration response must be an object")
            _validate_version_store_registration(registration, version_store)
            session_token = registration.get("token")
            if not isinstance(session_token, str) or not session_token:
                raise JudgeRegistrationError(
                    "Judge registration omitted session token"
                )
            work_session_token = _select_work_session_token(
                registration,
                max_submissions=max_submissions,
                version_saving_enabled=version_saving_enabled,
            )
            if handoff_enabled and registration.get("single_gpu_handoff") is not True:
                raise JudgeRegistrationError(
                    "Judge did not accept the single GPU handoff contract"
                )
            integrity_review = registration.get("integrity_review")
            if isinstance(integrity_review, dict):
                integrity_review_notice = integrity_review.get("announce") is True
                raw_allowance = integrity_review.get("review_timeout_seconds")
                if isinstance(raw_allowance, int) and not isinstance(raw_allowance, bool):
                    review_allowance = min(max(raw_allowance, 0), 7200)
        except JudgeRegistrationError:
            raise
        except (TypeError, ValueError, KeyError) as exc:
            raise JudgeRegistrationError(
                "Judge registration returned an invalid response"
            ) from exc
        logger.info("Registered session with judge server")

        # 2. Create container
        container_name = f"sforge.run.{task_spec.task_id}.{run_id}"

        backend.remove_container_by_name(container_name)

        env = _build_agent_env(agent, model)
        if agent_runtime is not None:
            # BASH_ENV puts the mounted runtime's bin/ on PATH for every
            # `bash -c` exec (install probe, agent run, agent subshells).
            env.update(runtime_environment(agent_runtime))
        if agent.name == "codex":
            env["SFORGE_EXEC_PROCESS_GROUP_FILE"] = (
                CODEX_PROCESS_GROUP_FILE
            )
        _configure_submission_service_env(
            env,
            judge_url=judge_url,
            session_token=work_session_token,
            max_submissions=max_submissions,
            version_saving_enabled=version_saving_enabled,
        )
        env["SFORGE_PATCH_DIR"] = task_spec.cwd
        env["SFORGE_SUBMIT_PATHS"] = " ".join(task_spec.submit_paths)
        env["SFORGE_SUBMIT_EXCLUDE_FLAGS"] = " ".join(
            f"--exclude={e}" for e in task_spec.submit_exclude
        )

        if task_spec.game_mode:
            game_api_base = (
                judge_url.rstrip("/")
                + f"/api/v1/game/{run_id}/{task_spec.task_id}"
            )
            env["GAME_SERVER_URL"] = game_api_base

        # Network isolation: preflight + extra_hosts pre-resolve
        container_extra_hosts = {"host.docker.internal": "host-gateway"}
        container_cap_drop: list[str] = []

        if not internet:
            from sforge.harness.network_isolation import (
                check_iptables_permission,
                is_ip_address,
                resolve_hostname,
            )

            if config.http_proxy or config.https_proxy:
                # Proxy variables are useful while installing the agent runtime,
                # which happens before isolation is applied.  They are safe to
                # retain only when the runtime API itself is a separate local
                # reverse relay: the final iptables allowlist includes the
                # relay port and Judge port, but not the generic install proxy.
                api_url = config.agent_api_base_url or ""
                parsed_api = urlparse(api_url)
                if parsed_api.hostname != "host.docker.internal" or not parsed_api.port:
                    raise RuntimeError(
                        "Offline runs with an install proxy require a separate "
                        "SFORGE_AGENT_API_BASE_URL at "
                        "http://host.docker.internal:<relay-port>."
                    )
                api_endpoint = (parsed_api.hostname, parsed_api.port)
                for proxy_url in (config.http_proxy, config.https_proxy):
                    if not proxy_url:
                        continue
                    parsed_proxy = urlparse(proxy_url)
                    proxy_port = parsed_proxy.port or (
                        443 if parsed_proxy.scheme == "https" else 80
                    )
                    if (parsed_proxy.hostname, proxy_port) == api_endpoint:
                        raise RuntimeError(
                            "The generic install proxy and restricted model API "
                            "relay must use different endpoints."
                        )
                logger.info(
                    "Direct proxy is install-only; its endpoint is excluded "
                    "from the runtime network allowlist"
                )
            if not check_iptables_permission():
                raise RuntimeError(
                    "Network isolation requires sudo iptables access. "
                    "Ensure passwordless sudo is configured for iptables."
                )

            api_url = config.agent_api_base_url or agent.default_api_base_url
            if api_url:
                api_host = urlparse(api_url).hostname or ""
                if api_host and api_host != "host.docker.internal" and not is_ip_address(api_host):
                    resolved_ips = resolve_hostname(api_host, logger)
                    if resolved_ips:
                        container_extra_hosts[api_host] = resolved_ips[0]

            container_cap_drop = ["NET_RAW"]
            logger.info("Network isolation enabled: preflight passed")

        cpu = config.work_cpu_limit if config.work_cpu_limit is not None else task_spec.work.cpu_limit
        mem = config.work_mem_limit if config.work_mem_limit is not None else task_spec.work.mem_limit

        create_kwargs: dict = {}
        if agent_runtime is not None:
            # Keyword is passed only when configured so task-side backend
            # monkeypatches without the parameter keep working unchanged.
            create_kwargs["mounts"] = [
                (str(agent_runtime.root), agent_runtime.mount_target, "ro")
            ]
        if handoff_enabled:
            create_kwargs["annotations"] = work_annotations
            create_kwargs["accelerator_allocation"] = accelerator_allocation
        handle = backend.create_container(
            task_spec.work_image_key,
            container_name,
            environment=env,
            extra_hosts=container_extra_hosts,
            cap_drop=container_cap_drop or None,
            cpu_limit=cpu,
            mem_limit=mem,
            **create_kwargs,
        )
        # Opt-in CPU pinning of the created container; a no-op when off.  A
        # failure leaves the container unstarted for the cleanup below.
        pin_before_start(
            backend,
            handle,
            role="work",
            config=config,
            record_path=log_dir / "cpu_affinity_work.json",
            logger=logger,
        )
        backend.start_container(handle)
        if handoff_enabled:
            attach_response = requests.post(
                f"{host_judge_url}/api/v1/handoff/attach-work",
                json={
                    "token": session_token,
                    "admin_secret": ADMIN_SECRET,
                    "work_container_id": handle.id,
                },
                timeout=10,
            )
            attach_response.raise_for_status()
            attach_payload = attach_response.json()
            if attach_payload.get("state") != "WORK_ACTIVE":
                raise JudgeRegistrationError(
                    "Judge did not activate the exact Work container lease"
                )
            final_handoff_fence = _FinalHandoffFence(
                host_judge_url,
                session_token,
            )
        logger.info(f"Container started: {container_name} (judge_url={judge_url})")

        # 3. Install agent runtime
        if agent_runtime is not None:
            install_cmds = runtime_probe_cmds(agent_runtime)
            (log_dir / "agent_runtime.json").write_text(
                json.dumps(agent_runtime.describe(), indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            logger.info(
                "Agent runtime mounted read-only at %s (%s@%s); network "
                "install replaced by an offline probe",
                agent_runtime.mount_target,
                agent_runtime.agent,
                agent_runtime.version,
            )
        else:
            install_cmds = agent.install_cmds
        for i, cmd in enumerate(install_cmds):
            logger.info(
                f"Install step {i + 1}/{len(install_cmds)}: {cmd}"
            )
            result = backend.exec_run_with_exit_code(
                handle, f"/bin/bash -c {shlex.quote(cmd)}", timeout=600
            )
            install_parts.append(result.output)
            if result.timed_out:
                raise RuntimeError(f"Install command timed out: {cmd}")
            if result.exit_code != 0:
                raise RuntimeError(
                    f"Install command failed (exit {result.exit_code}): {cmd}\n{result.output}"
                )
            logger.info(f"Install step {i + 1} done ({result.elapsed_seconds:.1f}s)")

        (log_dir / "install_output.txt").write_text("\n".join(install_parts))
        logger.info("Agent installation complete")

        # Configure credential files and other agent-specific runtime state
        # only after package installation, and before any task prompt runs.
        agent.prepare_runtime(backend, handle, log_dir, logger)

        # 4. Install tools.
        #    Game-mode tasks skip sforge-submit (scoring is built into the
        #    game HTTP API and there is no archive to submit), but still need
        #    the stop hook — otherwise the agent exits naturally as soon as
        #    the model decides it's "done", losing the full timeout budget.
        effective_eval_interval = 0 if disable_auto_eval else eval_interval
        if not task_spec.game_mode:
            _install_tools(
                backend,
                handle,
                task_spec,
                agent,
                log_dir,
                logger,
                disable_stop_hook=disable_stop_hook,
                install_submit_tool=max_submissions != 0,
                install_save_version_tool=version_saving_enabled,
            )

            # Prepare the host-side auto-eval thread. It is started below from
            # the same monotonic origin as the agent timeout.
            if effective_eval_interval > 0:
                auto_eval_stop = threading.Event()
        elif not disable_stop_hook:
            agent.install_stop_hook(backend, handle, log_dir, logger)
        install_model_catalog = getattr(agent, "install_model_catalog", None)
        if install_model_catalog is not None:
            install_model_catalog(
                backend, handle, log_dir, logger,
                model or config.agent_model or agent.default_model,
            )

        # 4b. Apply network isolation (after install + tools, before agent)
        if not internet:
            from sforge.harness.network_isolation import (
                AllowedEndpoint,
                build_allowed_endpoints,
            )

            gateway_ip = backend.get_container_gateway_ip(handle) or ""
            if not gateway_ip and backend.backend_name == "docker":
                raise RuntimeError(
                    "Cannot determine gateway IP for network isolation"
                )

            effective_api_url = config.agent_api_base_url or agent.default_api_base_url
            endpoints = build_allowed_endpoints(
                judge_url, effective_api_url, gateway_ip, logger,
            )
            net_isolation = backend.create_network_isolation(handle, endpoints, logger)
            net_isolation.apply()

        # 5. Write prompt
        if task_spec.game_mode:
            prompt = generate_game_prompt(task_spec.work.agent_query, internet=internet)
        else:
            prompt = generate_evolve_prompt(
                task_spec.work.agent_query,
                submit_paths=task_spec.submit_paths,
                internet=internet,
                max_submissions=max_submissions,
                submission_cooldown=submission_cooldown,
                version_saving_enabled=version_saving_enabled,
                integrity_review=integrity_review_notice,
            )
        prompt_path = "/tmp/agent_prompt.md"
        local_prompt = log_dir / "agent_prompt.md"
        local_prompt.write_text(prompt)
        copy_container_file(
            backend,
            handle,
            local_prompt,
            PurePosixPath(prompt_path),
            owner="agent:agent",
            mode=0o600,
            description="agent prompt",
        )
        logger.info(f"Prompt written ({len(prompt)} bytes)")

        # 6. Run agent (with auto-resume on abnormal exit)
        can_resume = not disable_auto_resume and agent.resume_cmd is not None
        resume_count = 0
        total_runtime = 0.0
        agent_timed_out = False
        all_output_parts: list[str] = []
        agent_live_log = log_dir / "agent_output.txt"
        MIN_RUNTIME_FOR_RESUME = 1
        MAX_RESUMES = _max_resume_attempts(
            effective_timeout,
            MIN_RUNTIME_FOR_RESUME,
        )

        started_monotonic = time.monotonic()
        run_deadline_monotonic = started_monotonic + effective_timeout
        started_at = time.time()
        from datetime import datetime
        started_at_iso = datetime.fromtimestamp(started_at).strftime("%Y-%m-%dT%H:%M:%S")
        (log_dir / "started_at").write_text(f"{started_at_iso}\n{started_at}\n")

        deadline_finalize_lock = threading.Lock()

        def finalize_single_gpu_at_deadline() -> bool:
            """Capture and retire a possibly Judge-paused Work container.

            Docker invokes this on the trusted host thread before its ordinary
            timeout stop.  Returning true means this callback strictly removed
            the container, which closes the blocked exec stream without ever
            resuming a paused candidate process.
            """

            nonlocal deadline_handoff_result, handoff_work_retired
            if final_handoff_fence is None:
                return False
            assert session_token is not None
            assert work_annotations is not None
            with deadline_finalize_lock:
                if deadline_handoff_result is not None:
                    return True
                if fence_watchdog is not None:
                    # A fenced lease can never be sealed or resumed. Fail
                    # here, before any seal/freeze attempt, so the exec
                    # unblocks with the audited fence reason.
                    fence_watchdog.check()
                seal_at, seal_monotonic = _seal_drain_and_verify_final_handoff(
                    final_handoff_fence,
                    host_judge_url=host_judge_url,
                    session_token=session_token,
                    eval_timeout=task_spec.judge.eval_timeout + review_allowance,
                    lease_path=lease_record_path(
                        config.log_dir,
                        run_id,
                        task_spec.task_id,
                    ),
                    expected_identity={
                        "trusted_lease_id": trusted_lease_id,
                        "run_id": run_id,
                        "task_id": task_spec.task_id,
                        "allocation_key": (
                            config.accelerator_allocation_key or ""
                        ),
                        "work_container_id": handle.id,
                    },
                    expected_allocation=accelerator_allocation,
                    expected_contract_sha256=evaluation_contract_sha256,
                    expected_annotations=work_annotations,
                    logger=logger,
                )
                archive, final_snapshot = _capture_frozen_final_snapshot(
                    backend,
                    handle,
                    task_spec,
                    agent,
                    log_dir,
                    started_monotonic,
                    logger,
                    run_deadline_monotonic=run_deadline_monotonic,
                    auto_eval_stop=auto_eval_stop,
                    auto_eval_thread=auto_eval_thread,
                    feedback_threads=auto_eval_feedback_threads,
                    final_handoff_annotations=work_annotations,
                    seal_at=seal_at,
                    seal_monotonic=seal_monotonic,
                )
                _retire_final_handoff_work(
                    host_judge_url=host_judge_url,
                    session_token=session_token,
                    archive=archive,
                )
                deadline_handoff_result = _DeadlineHandoffResult(
                    archive=archive,
                    snapshot=final_snapshot,
                )
                handoff_work_retired = True
                return True

        if auto_eval_stop is not None:
            auto_eval_thread = threading.Thread(
                target=_auto_eval_loop,
                args=(
                    backend,
                    handle,
                    task_spec,
                    agent,
                    host_judge_url,
                    session_token,
                    effective_eval_interval,
                    started_monotonic,
                    effective_timeout,
                    auto_eval_stop,
                    logger,
                    log_dir,
                    auto_eval_feedback_threads,
                ),
                daemon=True,
            )
            auto_eval_thread.start()
            tick_count = len(
                _scheduled_auto_eval_offsets(
                    effective_timeout, effective_eval_interval
                )
            )
            logger.info(
                "Host-side auto-eval started "
                f"(interval={effective_eval_interval}s, ticks={tick_count})"
            )

        on_chunk_cb = None

        exec_shutdown_event = shutdown_event
        if final_handoff_fence is not None:
            fence_watchdog = _HandoffFenceWatchdog(
                lease_record_path(config.log_dir, run_id, task_spec.task_id),
                shutdown_event=shutdown_event,
                logger=logger,
            )
            fence_watchdog.start()
            exec_shutdown_event = fence_watchdog.exec_stop_event

        while True:
            if fence_watchdog is not None:
                fence_watchdog.check()
            remaining_timeout = _remaining_run_timeout_seconds(
                run_deadline_monotonic
            )
            if remaining_timeout <= 0:
                agent_timed_out = True
                break
            is_resume = resume_count > 0
            run_cmd = agent.format_run_cmd(
                prompt_path, model=model,
                internet=internet, resume=is_resume,
            )
            if is_resume:
                logger.info(
                    f"Resuming agent (attempt {resume_count}/{MAX_RESUMES}, "
                    f"{remaining_timeout:.0f}s left): {run_cmd[:200]}..."
                )
            else:
                logger.info(f"Running agent: {run_cmd[:200]}... (timeout={remaining_timeout:.0f}s)")

            exec_kwargs = {
                "log_file": agent_live_log,
                "workdir": task_spec.cwd,
                "environment": env,
                "stream_to_stdout": verbose,
                "shutdown_event": exec_shutdown_event,
                "log_append": is_resume,
                "on_chunk": on_chunk_cb,
            }
            if backend.backend_name == "docker":
                # Opt in only for the candidate's deadline-bearing exec.  Judge
                # and setup commands retain their ordinary timeout behavior.
                exec_kwargs["stop_container_on_timeout"] = True
                if final_handoff_fence is not None:
                    # The callback executes synchronously at the timeout.  It
                    # may capture a Judge-paused Work container under the
                    # durable final fence and strictly retire it itself.
                    exec_kwargs["before_stop_on_timeout"] = (
                        finalize_single_gpu_at_deadline
                    )
            try:
                seg_result = backend.exec_run_with_timeout(
                    handle,
                    _agent_exec_command(agent, run_cmd),
                    timeout=remaining_timeout,
                    **exec_kwargs,
                )
            except HandoffFencedError:
                raise
            except Exception as exc:
                if fence_watchdog is not None and fence_watchdog.fenced:
                    raise HandoffFencedError(
                        "single GPU lease failed closed "
                        f"({fence_watchdog.fenced_state}): "
                        f"{fence_watchdog.fence_reason}"
                    ) from exc
                if backend.backend_name == "docker":
                    raise FinalSnapshotError(
                        "Agent exec ended without a proven Docker snapshot "
                        "boundary"
                    ) from exc
                raise
            if fence_watchdog is not None:
                fence_watchdog.check()
            all_output_parts.append(seg_result.output)
            total_runtime += seg_result.elapsed_seconds

            if seg_result.timed_out:
                agent_timed_out = True
                break

            if not can_resume:
                break
            if seg_result.elapsed_seconds < MIN_RUNTIME_FOR_RESUME:
                logger.warning(
                    f"Agent exited after only {seg_result.elapsed_seconds:.1f}s "
                    f"(< {MIN_RUNTIME_FOR_RESUME}s), not resuming (likely systematic failure)"
                )
                break
            if resume_count >= MAX_RESUMES:
                logger.warning(f"Max resume attempts ({MAX_RESUMES}) reached")
                break

            resume_count += 1
            logger.info(f"Agent exited after {seg_result.elapsed_seconds:.1f}s, will resume")

        agent_output = "\n".join(all_output_parts)
        runtime = time.monotonic() - started_monotonic
        logger.info(
            f"Agent finished: runtime={runtime:.1f}s, timed_out={agent_timed_out}, "
            f"resumes={resume_count}, exec_runtime={total_runtime:.1f}s"
        )

        normal_seal: tuple[float, float] | None = None
        if final_handoff_fence is not None and deadline_handoff_result is None:
            # Close admission before the normal final freeze too. A deadline
            # path may already have done this through the pre-stop callback.
            assert session_token is not None
            normal_seal = _seal_drain_and_verify_final_handoff(
                final_handoff_fence,
                host_judge_url=host_judge_url,
                session_token=session_token,
                eval_timeout=task_spec.judge.eval_timeout + review_allowance,
                lease_path=lease_record_path(
                    config.log_dir,
                    run_id,
                    task_spec.task_id,
                ),
                expected_identity={
                    "trusted_lease_id": trusted_lease_id,
                    "run_id": run_id,
                    "task_id": task_spec.task_id,
                    "allocation_key": config.accelerator_allocation_key or "",
                    "work_container_id": handle.id,
                },
                expected_allocation=accelerator_allocation,
                expected_contract_sha256=evaluation_contract_sha256,
                expected_annotations=work_annotations,
                logger=logger,
            )

        # 7-8. Establish the final filesystem boundary, then capture it.  Docker
        # provides the strong contract; other backends retain the legacy
        # running-container behavior until they implement the optional API.
        assert started_monotonic is not None
        if deadline_handoff_result is not None:
            final_archive = deadline_handoff_result.archive
            snapshot = deadline_handoff_result.snapshot
        elif backend.backend_name == "docker":
            final_archive, snapshot = _capture_frozen_final_snapshot(
                backend,
                handle,
                task_spec,
                agent,
                log_dir,
                started_monotonic,
                logger,
                run_deadline_monotonic=run_deadline_monotonic,
                auto_eval_stop=auto_eval_stop,
                auto_eval_thread=auto_eval_thread,
                feedback_threads=auto_eval_feedback_threads,
                final_handoff_annotations=(
                    work_annotations if handoff_enabled else None
                ),
                seal_at=normal_seal[0] if normal_seal else None,
                seal_monotonic=normal_seal[1] if normal_seal else None,
            )
        else:
            _stop_auto_eval_thread(auto_eval_stop, auto_eval_thread, logger)
            try:
                final_archive = _extract_archive_from_container(
                    backend, handle, task_spec, agent
                )
                _atomic_write_bytes(
                    log_dir / "final_archive.tar.gz", final_archive
                )
                snapshot = _write_final_snapshot_metadata(
                    log_dir, started_monotonic
                )
            except Exception as exc:
                logger.warning(
                    "Failed to extract final archive "
                    f"(container may have stopped): {exc}"
                )
                final_archive = b""
                snapshot = None

        logger.info(f"Final archive: {len(final_archive)} bytes")
        if snapshot is not None:
            logger.info(
                "Final snapshot captured at %.1fs",
                snapshot["snapshot_elapsed_seconds"],
            )
        if handoff_enabled and not handoff_work_retired:
            assert session_token is not None
            _retire_final_handoff_work(
                host_judge_url=host_judge_url,
                session_token=session_token,
                archive=final_archive,
            )
            handoff_work_retired = True
        if not task_spec.game_mode:
            # No version event may be published after the final tombstone.
            # Work is already frozen and auto-eval stopped, so first drain all
            # submissions that reached the Judge before sealing the run.
            assert session_token is not None
            _drain_pending_judge_evaluations(
                host_judge_url,
                session_token,
                task_spec.judge.eval_timeout + review_allowance,
                logger,
            )
        if final_archive:
            final_version = _persist_final_version(
                version_store,
                task_spec=task_spec,
                run_id=run_id,
                archive=final_archive,
                agent_name=agent.name,
                model=model or config.agent_model or agent.default_model,
                timed_out=agent_timed_out,
                runtime_seconds=runtime,
                interrupted=False,
            )
            if final_version is not None:
                logger.info(
                    "Final version persisted: %s",
                    final_version.version_id,
                )

        # 8. Collect results
        if task_spec.game_mode:
            try:
                requests.post(
                    f"{host_judge_url}/api/v1/game/{run_id}/{task_spec.task_id}/close-all",
                    timeout=30,
                )
            except Exception:
                logger.warning("Failed to close active game sessions")

            try:
                history_resp = requests.get(
                    f"{host_judge_url}/api/v1/history?token={session_token}",
                    timeout=10,
                )
                history_resp.raise_for_status()
                history = history_resp.json()
            except Exception:
                logger.warning("Failed to fetch run history from judge server")
                history = {"run_id": run_id, "best_score": None, "entries": []}

            (log_dir / "game_history.json").write_text(
                json.dumps(history, indent=2, ensure_ascii=False)
            )

            game_entries = [
                e for e in history.get("entries", []) if e.get("type") == "game"
            ]
            best_score_raw = history.get("best_score")
            best_score = (
                float(best_score_raw) if best_score_raw is not None else None
            )
            best_pass_rate_raw = history.get("best_pass_rate", 0.0)
            best_pass_rate = float(best_pass_rate_raw) if best_pass_rate_raw else 0.0
            best_round = history.get("best_round", "")
            total_rounds = len(game_entries)

            result = RunResult(
                archive=final_archive,
                best_pass_rate=best_pass_rate,
                best_score=best_score,
                best_round=best_round,
                total_rounds=total_rounds,
                agent_submissions=total_rounds,
                auto_submissions=0,
                agent_output=agent_output,
                timed_out=agent_timed_out,
                runtime_seconds=runtime,
                resume_count=resume_count,
            )
            logger.info(
                f"Done (game): best_score={best_score} ({best_round}), "
                f"sessions={total_rounds}, resumes={resume_count}, runtime={runtime:.1f}s"
            )
        else:
            # Save debug artifacts from container (non-authoritative, best-effort)
            if backend.backend_name == "docker" and handoff_work_retired:
                logger.info(
                    "Skipping best-effort Work state read after strict retirement"
                )
            else:
                try:
                    if backend.backend_name == "docker":
                        backend.verify_container_frozen_for_snapshot(handle)
                        state_payload = _read_file_from_container_archive(
                            backend,
                            handle,
                            PurePosixPath("/tmp/sforge_state.json"),
                        )
                        backend.verify_container_frozen_for_snapshot(handle)
                        state = json.loads(state_payload.decode())
                    else:
                        state_result = backend.exec_run_with_timeout(
                            handle,
                            [
                                "/bin/bash",
                                "-c",
                                "cat /tmp/sforge_state.json 2>/dev/null || echo '{}'",
                            ],
                            timeout=10,
                        )
                        state = json.loads(state_result.output.strip())
                    (log_dir / "evolve_state.json").write_text(
                        json.dumps(state, indent=2, ensure_ascii=False)
                    )
                except Exception:
                    logger.warning("Failed to read state file from container")

            if backend.backend_name == "docker":
                # The final snapshot contract already proved PID 0 and captures
                # no data through a later in-container command. In handoff
                # mode, strict exact-ID destruction supersedes that frozen
                # state and stale handles must not be inspected again.
                try:
                    disposition = _verify_post_snapshot_work_disposition(
                        backend,
                        handle,
                        handoff_work_retired=handoff_work_retired,
                    )
                except Exception as exc:
                    raise FinalSnapshotError(
                        "Work container became runnable after final snapshot"
                    ) from exc
                logger.info(
                    "Work container is %s after final snapshot",
                    disposition,
                )
            else:
                try:
                    backend.exec_run(
                        handle,
                        "kill 1 2>/dev/null || true",
                        user="root",
                    )
                    logger.info("Work container processes stopped")
                except Exception:
                    pass

            # Query Judge Server for authoritative host-only results.  An
            # unavailable history must not be represented as a synthetic empty
            # history because formal final-only validation treats empty history
            # as proof that no Work-time evaluation ran.
            history = _fetch_required_final_judge_history(
                host_judge_url,
                session_token,
            )

            (log_dir / "run_history.json").write_text(
                json.dumps(history, indent=2, ensure_ascii=False)
            )

            best_pass_rate = history.get("best_pass_rate", 0.0)
            best_score_raw = history.get("best_score")
            best_score = (
                float(best_score_raw) if best_score_raw is not None else None
            )
            best_round = history.get("best_round", "")
            agent_subs = history.get("agent_submissions", 0)
            auto_subs = history.get("auto_submissions", 0)
            total_rounds = agent_subs + auto_subs

            result = RunResult(
                archive=final_archive,
                best_pass_rate=best_pass_rate,
                best_score=best_score,
                best_round=best_round,
                total_rounds=total_rounds,
                agent_submissions=agent_subs,
                auto_submissions=auto_subs,
                agent_output=agent_output,
                timed_out=agent_timed_out,
                runtime_seconds=runtime,
                resume_count=resume_count,
            )

            logger.info(
                f"Done: best={result.best_pass_rate:.2%} "
                f"(round {result.best_round!r}), "
                f"agent_subs={agent_subs}, auto_subs={auto_subs}, "
                f"resumes={resume_count}, runtime={runtime:.1f}s"
            )
        return result

    except KeyboardInterrupt:
        logger.info("Run interrupted by user (Ctrl+C)")
        interrupted_seal: tuple[float, float] | None = None
        if final_handoff_fence is not None:
            assert session_token is not None
            interrupted_seal = _seal_drain_and_verify_final_handoff(
                final_handoff_fence,
                host_judge_url=host_judge_url,
                session_token=session_token,
                eval_timeout=task_spec.judge.eval_timeout + review_allowance,
                lease_path=lease_record_path(
                    config.log_dir,
                    run_id,
                    task_spec.task_id,
                ),
                expected_identity={
                    "trusted_lease_id": trusted_lease_id,
                    "run_id": run_id,
                    "task_id": task_spec.task_id,
                    "allocation_key": config.accelerator_allocation_key or "",
                    "work_container_id": handle.id,
                },
                expected_allocation=accelerator_allocation,
                expected_contract_sha256=evaluation_contract_sha256,
                expected_annotations=work_annotations,
                logger=logger,
            )

        # Establish the same immutable boundary on interruption.  In the Docker
        # path a failure is fatal: returning a best-effort racing archive would
        # misrepresent which candidate state was judged.
        interrupted_archive = b""
        if handle is not None:
            if backend.backend_name == "docker":
                interrupted_started = (
                    started_monotonic
                    if started_monotonic is not None
                    else time.monotonic()
                )
                interrupted_archive, _ = _capture_frozen_final_snapshot(
                    backend,
                    handle,
                    task_spec,
                    agent,
                    log_dir,
                    interrupted_started,
                    logger,
                    run_deadline_monotonic=run_deadline_monotonic,
                    auto_eval_stop=auto_eval_stop,
                    auto_eval_thread=auto_eval_thread,
                    feedback_threads=auto_eval_feedback_threads,
                    final_handoff_annotations=(
                        work_annotations if handoff_enabled else None
                    ),
                    seal_at=interrupted_seal[0] if interrupted_seal else None,
                    seal_monotonic=(
                        interrupted_seal[1] if interrupted_seal else None
                    ),
                )
                logger.info(
                    "Final archive (interrupted): "
                    f"{len(interrupted_archive)} bytes"
                )
            else:
                _stop_auto_eval_thread(
                    auto_eval_stop, auto_eval_thread, logger
                )
                try:
                    interrupted_archive = _extract_archive_from_container(
                        backend, handle, task_spec, agent
                    )
                    _atomic_write_bytes(
                        log_dir / "final_archive.tar.gz",
                        interrupted_archive,
                    )
                    if started_monotonic is not None:
                        _write_final_snapshot_metadata(
                            log_dir, started_monotonic
                        )
                    logger.info(
                        "Final archive (interrupted): "
                        f"{len(interrupted_archive)} bytes"
                    )
                except Exception:
                    pass

        if handoff_enabled and interrupted_archive:
            assert session_token is not None
            _retire_final_handoff_work(
                host_judge_url=host_judge_url,
                session_token=session_token,
                archive=interrupted_archive,
            )
            handoff_work_retired = True

        if interrupted_archive:
            if not task_spec.game_mode and session_token is not None:
                _drain_pending_judge_evaluations(
                    host_judge_url,
                    session_token,
                    task_spec.judge.eval_timeout + review_allowance,
                    logger,
                )
            interrupted_runtime = (
                max(0.0, time.monotonic() - started_monotonic)
                if started_monotonic is not None
                else 0.0
            )
            interrupted_version = _persist_final_version(
                version_store,
                task_spec=task_spec,
                run_id=run_id,
                archive=interrupted_archive,
                agent_name=agent.name,
                model=model or config.agent_model or agent.default_model,
                timed_out=False,
                runtime_seconds=interrupted_runtime,
                interrupted=True,
            )
            if interrupted_version is not None:
                logger.info(
                    "Final interrupted version persisted: %s",
                    interrupted_version.version_id,
                )

        # Query judge server for results accumulated before interruption (full history)
        try:
            history = _fetch_full_judge_history(
                host_judge_url,
                session_token,
            )
        except Exception:
            history = {
                "run_id": run_id, "best_score": None,
                "best_pass_rate": 0.0, "best_round": "",
                "agent_submissions": 0, "auto_submissions": 0,
                "entries": [],
            }

        (log_dir / "run_history.json").write_text(
            json.dumps(history, indent=2, ensure_ascii=False)
        )

        best_pass_rate = history.get("best_pass_rate", 0.0)
        best_score_raw = history.get("best_score")
        best_score = float(best_score_raw) if best_score_raw is not None else None
        best_round = history.get("best_round", "")
        agent_subs = history.get("agent_submissions", 0)
        auto_subs = history.get("auto_submissions", 0)
        total_rounds = agent_subs + auto_subs

        logger.info(
            f"Interrupted results: best={best_pass_rate:.2%} "
            f"(round {best_round!r}), "
            f"agent_subs={agent_subs}, auto_subs={auto_subs}"
        )

        return RunResult(
            archive=interrupted_archive,
            best_pass_rate=best_pass_rate,
            best_score=best_score,
            best_round=best_round,
            total_rounds=total_rounds,
            agent_submissions=agent_subs,
            auto_submissions=auto_subs,
            agent_output="Stopped by user (Ctrl+C)",
            timed_out=False,
            resume_count=resume_count,
        )
    except JudgeDrainError:
        logger.error("Judge evaluations did not quiesce; aborting run")
        raise
    except FinalSnapshotError:
        logger.error("Immutable final snapshot could not be established")
        raise
    except FinalVersionStoreError:
        logger.error("Immutable final version could not be persisted")
        raise
    except JudgeRegistrationError:
        logger.error("Judge session registration could not be established")
        raise
    except Exception as e:
        logger.error(f"Error: {e}\n{traceback.format_exc()}")

        if handoff_enabled:
            # An opt-in single-GPU run must never return an ordinary RunResult
            # after an incomplete retirement.  The finally block persists the
            # final fence when attachment succeeded and deliberately leaves
            # the exact Work container quarantined for reconciliation.
            raise FinalSnapshotError(
                "single GPU run aborted before safe final Work retirement"
            ) from e

        try:
            history_resp = requests.get(
                f"{host_judge_url}/api/v1/history?token={session_token}",
                timeout=10,
            )
            history_resp.raise_for_status()
            history = history_resp.json()
            best_pass_rate = history.get("best_pass_rate", 0.0)
            best_score_raw = history.get("best_score")
            best_score = float(best_score_raw) if best_score_raw is not None else None
            best_round = history.get("best_round", "")
            agent_subs = history.get("agent_submissions", 0)
            auto_subs = history.get("auto_submissions", 0)
        except Exception:
            best_pass_rate = 0.0
            best_score = None
            best_round = ""
            agent_subs = 0
            auto_subs = 0

        return RunResult(
            best_pass_rate=best_pass_rate,
            best_score=best_score,
            best_round=best_round,
            total_rounds=agent_subs + auto_subs,
            agent_submissions=agent_subs,
            auto_submissions=auto_subs,
            agent_output=str(e),
            timed_out=False,
            resume_count=locals().get("resume_count", 0),
        )
    finally:
        if fence_watchdog is not None:
            fence_watchdog.stop()
        _stop_auto_eval_thread(auto_eval_stop, auto_eval_thread, logger)
        _join_threads(auto_eval_feedback_threads)
        if net_isolation is not None:
            try:
                net_isolation.cleanup()
            except Exception as exc:
                if logger:
                    logger.warning(f"Failed to cleanup network isolation: {exc}")
        handoff_attached = handoff_enabled and final_handoff_fence is not None
        if (
            handoff_attached
            and not handoff_work_retired
            and fence_watchdog is not None
            and fence_watchdog.fenced
        ):
            # The coordinator already persisted the failure state; a fenced
            # lease accepts no seal. Leave the exact Work container paused
            # and quarantined for operator reconciliation.
            logger.error(
                "Single GPU lease is %s (%s); Work container remains "
                "quarantined for operator reconciliation",
                fence_watchdog.fenced_state,
                fence_watchdog.fence_reason,
            )
        elif handoff_attached and not handoff_work_retired:
            try:
                final_handoff_fence.seal()
            except Exception:
                logger.error(
                    "Unable to persist handoff never-resume during cleanup; "
                    "leaving the exact Work container quarantined"
                )
            # Generic cleanup swallows Docker errors and has no post-idle
            # attestation. Never use it to release an attached handoff owner.
            logger.error(
                "Single GPU Work was not safely retired; container remains "
                "quarantined for operator reconciliation"
            )
        elif handoff_attached:
            # The Judge coordinator already performed strict exact-ID removal.
            pass
        elif threading.current_thread() is threading.main_thread():
            prev_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
            print("\nStopping container, please wait... (Ctrl+C disabled during cleanup)")
            backend.cleanup_container(handle, logger)
            signal.signal(signal.SIGINT, prev_handler)
        else:
            backend.cleanup_container(handle, logger)
        close_logger(logger)
