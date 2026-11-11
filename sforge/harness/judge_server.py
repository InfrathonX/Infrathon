"""Judge HTTP REST API server (FastAPI)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import secrets
import stat
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator

import requests
from fastapi import (
    BackgroundTasks,
    FastAPI,
    Header,
    HTTPException,
    Query,
    Request,
)
from pydantic import BaseModel

from sforge.harness.accelerator_observer import (
    AcceleratorBusyError,
    AcceleratorObserver,
    create_accelerator_observer,
)
from sforge.harness.backend import AcceleratorAllocation, ContainerBackend
from sforge.harness.config import (
    SForgeConfig,
    create_backend_from_config,
    get_container_env,
    load_config,
    validate_single_gpu_handoff_config,
)
from sforge.harness.constants import (
    ADMIN_SECRET,
    ADMIN_SECRET_HEADER,
    SESSION_TOKEN_HEADER,
    SUBMISSION_ARCHIVE_MEDIA_TYPE,
    SUBMISSION_KIND_HEADER,
    VERSION_LABEL_HEADER,
)
from sforge.harness.docker_build import BuildImageError, build_judge_image
from sforge.harness.llm_judge.agent_engine import AgentReviewEngine
from sforge.harness.llm_judge.bundle import (
    BaselineProvider,
    DockerWorkImageBaseline,
    NoBaseline,
)
from sforge.harness.llm_judge.egress import ReviewEgress
from sforge.harness.llm_judge.reviewer import (
    STATUS_ERROR,
    STATUS_VIOLATION,
    IntegrityReview,
    IntegrityReviewer,
    rejected_report,
    unavailable_report,
)
from sforge.harness.run_evaluation import (
    JudgeContainerLifecycle,
    PRIVATE_CONTEXT_ENV_PREFIX,
    PRIVATE_JUDGE_ARTIFACT_NAME,
    SUBMISSION_ARCHIVE_MAX_BYTES,
    _durable_atomic_write_bytes,
    _persist_report,
    judge_submission,
)
from sforge.harness.resource_handoff import (
    DurableResourceLease,
    ExclusiveProcessLock,
    HandoffBinding,
    ResourceHandoffError,
    handoff_annotations,
    judge_handoff_annotations,
    lease_record_path,
    read_durable_lease,
)
from sforge.harness.resource_lease import LeaseIdentity, LeaseState
from sforge.harness.score_rescale import rescale_score
from sforge.harness.selection import select_best
from sforge.harness.benchmark import load_benchmark
from sforge.harness.task_spec import TaskSpec, load_all_tasks
from sforge.harness.version_store import (
    StoredVersion,
    VersionStore,
    VersionStoreCandidateError,
    VersionStoreCollision,
    VersionStoreError,
    VersionStoreQuotaExceeded,
    VersionStoreWriteError,
    validate_version_label,
)

logger = logging.getLogger("sforge.judge_server")

GAME_SERVER_APP_PATH = Path(__file__).parent / "game_server_app.py"
_COARSE_FAMILY_NAMES = ("large-K", "large-N/high-D")
# Checkpoint saves are bounded by compressed bytes only; the per-run count is
# deliberately uncapped so long agent runs never lose late candidates.
CHECKPOINT_MAX_BYTES_PER_RUN = 1024 * 1024 * 1024
CHECKPOINT_MIN_INTERVAL_SECONDS = 1.0
ARCHIVE_UPLOAD_TIMEOUT_SECONDS = 120.0
_SAFE_FINAL_RUN_ID_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}",
    re.ASCII,
)
_CANONICAL_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}", re.ASCII)
# Round IDs reach task verifiers through the private evaluation context and
# name a submissions/ leaf in checkpoint records; keep them label-safe.
_SAFE_FINAL_ROUND_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", re.ASCII)
DEFAULT_FINAL_ROUND_ID = "manual-1"


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        os.chmod(temporary_name, 0o600)
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


def _safe_valid_family_bands(report: object) -> dict[str, dict[str, Any]] | None:
    """Extract only already-agent-visible categorical family bands."""

    if not isinstance(report, dict) or report.get("valid") is not True:
        return None
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        return None
    raw = metrics.get("deployment_family_feedback")
    if not isinstance(raw, dict) or set(raw) != set(_COARSE_FAMILY_NAMES):
        return None
    safe: dict[str, dict[str, Any]] = {}
    for family in _COARSE_FAMILY_NAMES:
        entry = raw.get(family)
        if not isinstance(entry, dict) or entry.get("status") != "passed":
            return None
        band = entry.get("warmed_latency_band_ms")
        if not isinstance(band, dict):
            return None
        lower = band.get("lower_ms")
        upper = band.get("upper_ms")
        if (
            isinstance(lower, bool)
            or isinstance(upper, bool)
            or not isinstance(lower, (int, float))
            or not isinstance(upper, (int, float))
        ):
            return None
        lower_value = float(lower)
        upper_value = float(upper)
        if (
            not math.isfinite(lower_value)
            or not math.isfinite(upper_value)
            or lower_value <= 0.0
            or not math.isclose(
                upper_value / lower_value,
                1.25,
                rel_tol=1e-5,
                abs_tol=1e-5,
            )
        ):
            return None
        safe[family] = {
            "status": "passed",
            "warmed_latency_band_ms": {
                "lower_ms": lower_value,
                "upper_ms": upper_value,
            },
        }
    return safe


def _add_family_band_trends(
    report: dict[str, Any],
    previous_entries: list[dict[str, Any]],
) -> None:
    """Compare current visible bands with earlier valid manual-band bests."""

    current = _safe_valid_family_bands(report)
    if current is None:
        return
    prior_lowers: dict[str, list[float]] = {
        family: [] for family in _COARSE_FAMILY_NAMES
    }
    for entry in previous_entries:
        if (
            entry.get("type") != "submission"
            or not str(entry.get("round") or "").startswith("agent-")
            or entry.get("status") != "completed"
            or entry.get("valid") is not True
        ):
            continue
        prior = entry.get("deployment_family_feedback")
        if not isinstance(prior, dict):
            continue
        for family in _COARSE_FAMILY_NAMES:
            family_entry = prior.get(family)
            if not isinstance(family_entry, dict):
                continue
            band = family_entry.get("warmed_latency_band_ms")
            if not isinstance(band, dict):
                continue
            lower = band.get("lower_ms")
            if (
                not isinstance(lower, bool)
                and isinstance(lower, (int, float))
                and math.isfinite(float(lower))
                and float(lower) > 0.0
            ):
                prior_lowers[family].append(float(lower))

    raw_feedback = report["metrics"]["deployment_family_feedback"]
    for family in _COARSE_FAMILY_NAMES:
        if not prior_lowers[family]:
            continue
        current_lower = current[family]["warmed_latency_band_ms"]["lower_ms"]
        best_lower = min(prior_lowers[family])
        if current_lower < best_lower:
            trend = "improved"
        elif current_lower > best_lower:
            trend = "regressed"
        else:
            trend = "unchanged"
        raw_feedback[family]["trend_vs_best_valid_band"] = trend


class SubmissionBudgetExceeded(Exception):
    pass


class SubmissionCooldownActive(Exception):
    pass


class AgentSubmissionBusy(Exception):
    pass


class PublicEvaluationBusy(Exception):
    pass


class TokenCapabilityDenied(Exception):
    pass


class CheckpointBusy(Exception):
    pass


class CheckpointRateLimited(Exception):
    pass


class ArchiveUploadBusy(Exception):
    pass


# --- Models ---


class SubmissionStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


class ResultResponse(BaseModel):
    submission_id: str
    status: SubmissionStatus
    report: dict | None = None
    error: str | None = None
    version_id: str | None = None
    candidate_id: str | None = None
    content_sha256: str | None = None


class GameNewRequest(BaseModel):
    pass


class GameStepRequest(BaseModel):
    action: str


class GameNewResponse(BaseModel):
    session_id: str
    observation: str
    score: int
    peak_score: int
    max_score: int
    done: bool
    moves: int


class GameStepResponse(BaseModel):
    session_id: str
    observation: str
    score: int
    peak_score: int
    max_score: int
    done: bool
    moves: int


class GameStatusResponse(BaseModel):
    session_id: str
    score: int
    peak_score: int
    max_score: int
    done: bool
    moves: int


class GameCloseResponse(BaseModel):
    session_id: str
    final_score: int
    peak_score: int
    max_score: int
    moves: int


class RegisterRequest(BaseModel):
    task_id: str
    run_id: str
    admin_secret: str = ""
    judge_cpu_limit: int | None = None
    judge_mem_limit: str | None = None
    backend: str | None = None
    k8s_image_registry: str | None = None
    k8s_namespace: str | None = None
    k8s_node_selector: dict[str, str] | None = None
    k8s_kubeconfig: str | None = None
    max_agent_submissions: int | None = None
    submission_cooldown: int | None = None
    issue_checkpoint_token: bool = False
    require_version_saving: bool = False
    required_version_store_id: str | None = None
    single_gpu_handoff: bool = False
    trusted_lease_id: str | None = None
    allocation_key: str | None = None
    accelerator_device_ids: list[str] | None = None
    evaluation_contract_sha256: str | None = None


class RegisterResponse(BaseModel):
    token: str
    checkpoint_token: str | None = None
    version_saving_enabled: bool
    version_store_id: str | None = None
    single_gpu_handoff: bool = False
    # Set when an integrity review covers this task's agent submissions:
    # {mode, announce, review_timeout_seconds}.
    integrity_review: dict[str, Any] | None = None


class AttachWorkRequest(BaseModel):
    token: str
    admin_secret: str = ""
    work_container_id: str


class HandoffControlRequest(BaseModel):
    token: str
    admin_secret: str = ""


class FinalWorkArchiveRequest(HandoffControlRequest):
    archive_sha256: str
    archive_size: int


class FinalJudgeRequest(BaseModel):
    admin_secret: str = ""
    source_run_id: str
    final_run_id: str
    task_id: str
    trusted_lease_id: str
    allocation_key: str
    evaluation_contract_sha256: str
    archive_sha256: str
    archive_size: int
    timeout: int | None = None
    round_id: str | None = None


class HandoffControlResponse(BaseModel):
    state: str
    epoch: int
    recovery_policy: str


class FinalJudgeResponse(HandoffControlResponse):
    report: dict[str, Any]


class SubmitResponse(BaseModel):
    submission_id: str
    round_id: str
    status: SubmissionStatus
    remaining_submissions: int | None = None
    version_id: str | None = None
    candidate_id: str | None = None
    content_sha256: str | None = None
    archive_sha256: str | None = None


class SaveVersionResponse(BaseModel):
    version_id: str
    round_id: str
    archive_sha256: str
    archive_size: int
    candidate_id: str
    content_sha256: str


# --- Game session state ---

GAME_SESSION_IDLE_TIMEOUT = 600  # 10 minutes
GAME_SESSION_MAX = 200
GAME_SESSION_PER_RUN_MAX = 3
GAME_REAPER_INTERVAL = 60
GAME_CONTAINER_READY_TIMEOUT = 30
GAME_CONTAINER_READY_POLL = 0.5
GAME_CONTAINER_PORT = 8000


@dataclass
class GameSessionState:
    session_id: str
    run_id: str
    task_id: str
    game_num: int
    container: object  # ContainerHandle
    container_url: str
    max_score: int = 0
    peak_score: int = 0
    current_score: int = 0
    moves: int = 0
    done: bool = False
    steps: list = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)


@dataclass
class HandoffSession:
    lease: DurableResourceLease
    work_handle: object
    allocation: AcceleratorAllocation
    annotations: dict[str, str]
    # Serializes the irreversible final fence with the complete
    # recovery-policy-check -> Work-unpause operation.  Without this lock a
    # deadline seal could persist ``never_resume`` immediately after a worker
    # checked the old policy but immediately before it resumed Work.
    control_lock: threading.RLock = field(default_factory=threading.RLock)


class HandoffResourceBusy(ResourceHandoffError):
    """A safely attributed busy observation left Work available for retry."""


class _HandoffJudgeLifecycle(JudgeContainerLifecycle):
    def __init__(
        self,
        state: "JudgeState",
        session: HandoffSession,
        submission_id: str,
        run_id: str,
        task_id: str,
        round_id: str | None,
    ) -> None:
        self.state = state
        self.session = session
        self.submission_id = submission_id
        self.run_id = run_id
        self.task_id = task_id
        self.round_id = round_id
        self._annotations = judge_handoff_annotations(
            session.lease.binding,
            attempt_id=submission_id,
        )

    def judge_annotations(self) -> dict[str, str]:
        return dict(self._annotations)

    def judge_started(self, handle) -> None:
        backend = self.state._get_backend(
            self.session.lease.binding.identity.run_id
        )
        exact = backend.get_container_handle(
            handle.id,
            expected_annotations=self._annotations,
        )
        if exact.id != handle.id:
            raise ResourceHandoffError(
                "Judge exact-ID lookup returned a different container"
            )
        self.session.lease.record_judge_container(
            container_id=handle.id,
            annotations=self._annotations,
            final=False,
        )
        self.session.lease.transition(LeaseState.JUDGE_RUNNING)
        submission = self.state.submissions[self.submission_id]
        submission["status"] = SubmissionStatus.RUNNING
        self.state._record_submission(
            self.run_id,
            self.submission_id,
            self.task_id,
            self.round_id,
            None,
            None,
            status="running",
        )

    def judge_cleaning(self, handle) -> None:
        self.session.lease.transition(LeaseState.JUDGE_CLEANING)

    def judge_destroyed(self, handle) -> None:
        self.session.lease.transition(LeaseState.JUDGE_CLEANED)


FINAL_ARCHIVE_PRIVATE_CONTEXT_KEY = PRIVATE_CONTEXT_ENV_PREFIX + "ARCHIVE_SHA256"


def _final_judge_private_context(
    judge_extra_env: dict[str, str],
    *,
    archive_sha256: str,
) -> dict[str, str]:
    """Judge environment for the standalone final: the server's role-bound
    environment plus the coordinator-authenticated final archive digest.

    Task verifiers that bind their final row commitments to the evaluated
    archive read it as the private context field ``ARCHIVE_SHA256`` (bare
    lowercase hex, exactly what ``sha256sum`` prints).  The value is derived
    from the archive the coordinator re-opened from its own immutable
    source-run record and checked against the retired Work lease, never from
    the caller.  A server started with a conflicting value for the same key
    is refused: the final Judge must not be told a digest that differs from
    the archive it is actually judging.
    """
    if not isinstance(archive_sha256, str) or re.fullmatch(
        r"sha256:[0-9a-f]{64}", archive_sha256
    ) is None:
        raise ResourceHandoffError("final Judge archive digest is invalid")
    archive_hex = archive_sha256.removeprefix("sha256:")
    existing = judge_extra_env.get(FINAL_ARCHIVE_PRIVATE_CONTEXT_KEY)
    if existing is not None and existing != archive_hex:
        raise ResourceHandoffError(
            "final Judge private archive context conflicts with the "
            "authenticated archive"
        )
    return {**judge_extra_env, FINAL_ARCHIVE_PRIVATE_CONTEXT_KEY: archive_hex}


class _FinalHandoffJudgeLifecycle(JudgeContainerLifecycle):
    """Durable final-Judge lifecycle owned solely by the coordinator."""

    def __init__(
        self,
        state: "JudgeState",
        session: HandoffSession,
        attempt_id: str,
    ) -> None:
        self.state = state
        self.session = session
        self._annotations = judge_handoff_annotations(
            session.lease.binding,
            attempt_id=attempt_id,
        )

    def judge_annotations(self) -> dict[str, str]:
        return dict(self._annotations)

    def judge_started(self, handle) -> None:
        backend = self.state._get_backend(
            self.session.lease.binding.identity.run_id
        )
        exact = backend.get_container_handle(
            handle.id,
            expected_annotations=self._annotations,
        )
        if exact.id != handle.id:
            raise ResourceHandoffError(
                "final Judge exact-ID lookup returned a different container"
            )
        self.session.lease.record_judge_container(
            container_id=handle.id,
            annotations=self._annotations,
            final=True,
        )
        self.session.lease.transition(LeaseState.FINAL_JUDGE_RUNNING)

    def judge_cleaning(self, handle) -> None:
        self.session.lease.transition(LeaseState.FINAL_JUDGE_CLEANING)

    def judge_destroyed(self, handle) -> None:
        self.session.lease.transition(LeaseState.FINAL_JUDGE_CLEANED)


# --- Server state ---


class JudgeState:
    def __init__(
        self,
        config: SForgeConfig,
        *,
        accelerator_observer: AcceleratorObserver | None = None,
        submission_evaluator=None,
    ):
        validate_single_gpu_handoff_config(config)
        self.config = config
        self.tasks: dict[str, TaskSpec] = {}
        self.submissions: dict[str, dict] = {}  # submission_id -> {status, report, ...}
        self.game_sessions: dict[str, GameSessionState] = {}
        self.run_history: dict[str, list[dict]] = {}  # run_id -> [entries]
        self.tokens: dict[str, dict] = {}  # token -> task/run counters and capabilities
        self.run_resource_limits: dict[str, dict] = {}  # run_id -> {judge_cpu_limit, judge_mem_limit}
        self.run_backends: dict[str, ContainerBackend] = {}  # run_id -> backend
        self._game_lock = threading.Lock()
        self._history_lock = threading.Lock()
        self._tokens_lock = threading.Lock()
        self._checkpoint_lock = threading.Lock()
        self._checkpoint_inflight: set[tuple[str, str]] = set()
        self._checkpoint_last_started: dict[tuple[str, str], float] = {}
        self._archive_upload_inflight: set[tuple[str, str]] = set()
        self._task_judge_locks_guard = threading.Lock()
        self._task_judge_locks: dict[str, threading.Lock] = {}
        self._handoff_sessions: dict[str, HandoffSession] = {}
        self._handoff_allocations: dict[str, str] = {}
        self._handoff_server_lock: ExclusiveProcessLock | None = None
        self.accelerator_observer = accelerator_observer
        if config.single_gpu_handoff:
            self.accelerator_observer = (
                accelerator_observer
                or create_accelerator_observer(
                    config.accelerator_observer,
                    idle_grace_seconds=(
                        config.accelerator_idle_grace_seconds
                    ),
                )
            )
            if self.accelerator_observer is None:
                raise ResourceHandoffError(
                    "single GPU handoff has no accelerator observer"
                )
            assert config.accelerator_allocation_key is not None
            allocation_digest = config.accelerator_allocation_key.removeprefix(
                "allocation-sha256:"
            )
            self._handoff_server_lock = ExclusiveProcessLock(
                config.log_dir
                / "resource_leases"
                / "allocations"
                / f"{allocation_digest}.lock",
                "single GPU Judge coordinator",
            )
        if submission_evaluator is not None and config.single_gpu_handoff:
            raise ValueError("A resident evaluator cannot use GPU container handoff")
        self.submission_evaluator = submission_evaluator
        self.backend = (
            create_backend_from_config(config) if submission_evaluator is None else None
        )
        self.version_store = (
            VersionStore.open(config.version_store_root)
            if config.version_store_root is not None
            else None
        )
        # Mandatory CPU-only LLM integrity review in front of the GPU Judge
        # (``off`` only by an explicit SFORGE_LLM_JUDGE_MODE, for debugging).
        self.integrity_reviewer: IntegrityReviewer | None = None
        self._review_egress: ReviewEgress | None = None
        if not config.llm_judge.enabled:
            logger.warning(
                "LLM integrity review is OFF: submissions reach the GPU Judge "
                "unreviewed (SFORGE_LLM_JUDGE_MODE=off is for debugging only)"
            )
        else:
            config.llm_judge.check_ready()
            engine = self._integrity_engine(config)
            if engine is not None or config.llm_judge.engine != "agent":
                self.integrity_reviewer = self._new_integrity_reviewer(engine)
            else:
                # A backend that is not local Docker (PyroMind) gets its
                # review sandboxes from the integrator: configure_integrity_
                # review().  Until then covered submissions fail closed.
                logger.info(
                    "LLM integrity review awaits a review engine for the %s backend",
                    getattr(self.backend, "backend_name", "configured"),
                )
        self._integrity_engine_obj = None
        self._baseline_providers: dict[int, BaselineProvider] = {}
        self._baseline_providers_lock = threading.Lock()
        self._run_baselines: dict[str, BaselineProvider] = {}

        self._reaper_stop = threading.Event()
        self._reaper_thread = threading.Thread(
            target=self._reaper_loop, daemon=True
        )
        self._reaper_thread.start()

    def _new_integrity_reviewer(self, engine) -> IntegrityReviewer:
        return IntegrityReviewer(
            self.config.llm_judge,
            tasks_dir=self.config.tasks_dir,
            cache_dir=self.config.log_dir / "integrity_review_cache",
            engine=engine,
        )

    def _integrity_engine(self, config: SForgeConfig):
        """Local Docker review sandboxes; None when the backend is not Docker
        (the integrator then supplies an engine) or for the api engine."""

        review = config.llm_judge
        if review.engine != "agent":
            return None
        if getattr(self.backend, "backend_name", None) != "docker":
            return None
        if review.network == "platform":
            raise ValueError(
                "SFORGE_LLM_JUDGE_NETWORK=platform needs an integrator-provided "
                "review sandbox; local Docker reviews use isolated or bridge"
            )
        self._review_egress = ReviewEgress(
            self.backend.client,
            mode=review.network,
            plan=AgentReviewEngine.egress_plan(review),
            sidecar_image_resolver=lambda image: image,
        )
        from sforge.harness.resource_profile import review_sandbox_limits

        return AgentReviewEngine(
            review, backend=self.backend, egress=self._review_egress,
            sandbox_limits=review_sandbox_limits(config),
        )

    def configure_integrity_review(self, engine) -> None:
        """Install the review engine for a backend that hosts its own review
        sandboxes (for example a PyroMind controller's CPU sandboxes)."""

        if not self.config.llm_judge.enabled:
            raise ValueError("the LLM integrity review is off")
        reviewer = self._new_integrity_reviewer(engine)
        if self.tasks:
            reviewer.validate_tasks(list(self.tasks.values()))
        self.integrity_reviewer = reviewer
        self._integrity_engine_obj = engine

    def set_run_baseline(self, run_id: str, provider: BaselineProvider) -> None:
        """Use ``provider`` for the starting files of every review in a run."""

        with self._baseline_providers_lock:
            self._run_baselines[run_id] = provider

    def close_integrity_review(self) -> None:
        if self._review_egress is not None:
            self._review_egress.close()
        engine = self._integrity_engine_obj
        if engine is not None and hasattr(engine, "close"):
            try:
                engine.close()
            except Exception:
                logger.warning("Could not close the integrity review engine", exc_info=True)

    def _reaper_loop(self) -> None:
        while not self._reaper_stop.is_set():
            self._reaper_stop.wait(GAME_REAPER_INTERVAL)
            self._cleanup_idle_sessions()

    def _cleanup_idle_sessions(self) -> None:
        now = time.time()
        expired: list[str] = []
        with self._game_lock:
            for sid, sess in self.game_sessions.items():
                if now - sess.last_active > GAME_SESSION_IDLE_TIMEOUT:
                    expired.append(sid)
        for sid in expired:
            self._destroy_game_session(sid)

    def _destroy_game_session(self, session_id: str) -> None:
        with self._game_lock:
            sess = self.game_sessions.pop(session_id, None)
        if sess is None:
            return
        self._archive_game_session(sess)
        try:
            backend = self._get_backend(sess.run_id)
            backend.cleanup_container(sess.container, logger)
        except Exception:
            logger.exception("Error cleaning up game container for session %s", session_id)

    def _game_log_dir(self, sess: GameSessionState) -> Path:
        return (
            self.config.log_dir / "runs" / sess.run_id
            / sess.task_id / "submissions" / f"game-{sess.game_num}"
        )

    def _archive_game_session(self, sess: GameSessionState) -> None:
        entry = {
            "type": "game",
            "round": f"game-{sess.game_num}",
            "session_id": sess.session_id,
            "task_id": sess.task_id,
            "max_score": sess.max_score,
            "peak_score": sess.peak_score,
            "final_score": sess.current_score,
            "score": sess.current_score,
            "moves": sess.moves,
            "steps": sess.steps,
        }
        if sess.max_score and sess.max_score > 0:
            entry["pass_rate"] = sess.current_score / sess.max_score
        else:
            entry["pass_rate"] = 0.0
        with self._history_lock:
            history_key = f"{sess.run_id}/{sess.task_id}"
            self.run_history.setdefault(history_key, []).append(entry)

        log_dir = self._game_log_dir(sess)
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "game_result.json").write_text(json.dumps(entry, indent=2, ensure_ascii=False))

    def _flush_game_step(self, sess: GameSessionState) -> None:
        log_dir = self._game_log_dir(sess)
        log_dir.mkdir(parents=True, exist_ok=True)
        step = sess.steps[-1]
        with open(log_dir / "steps.jsonl", "a") as f:
            f.write(json.dumps(step, ensure_ascii=False) + "\n")

    # --- Token-based session registration ---

    def _registration_judge_size(self, task_id, run_id, backend,
                                 judge_cpu_limit, judge_mem_limit):
        """Local Docker Judges take the task's deployment profile size
        (deploy/pyromind/RESOURCES.md) whatever the client registered."""
        from sforge.harness.resource_profile import judge_overlay, summary

        if self.config.resource_profile == "native":
            return judge_cpu_limit, judge_mem_limit, None
        backend_name = (backend or getattr(self.backend, "backend_name", None)
                        or self.config.backend)
        config = replace(self.config, backend=backend_name)
        cpu, memory, undersize, record = judge_overlay(
            config, self.tasks[task_id], judge_cpu_limit, judge_mem_limit
        )
        if record:
            directory = self.config.log_dir / "runs" / run_id / task_id
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "judge_resource_profile.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False) + "\n"
            )
            logger.info(summary(record))
        return cpu, memory, undersize

    def register_session(self, task_id: str, run_id: str,
                         judge_cpu_limit: int | None = None,
                         judge_mem_limit: str | None = None,
                         backend: str | None = None,
                         k8s_image_registry: str | None = None,
                         k8s_namespace: str | None = None,
                         k8s_node_selector: dict[str, str] | None = None,
                         k8s_kubeconfig: str | None = None,
                         max_agent_submissions: int | None = None,
                         submission_cooldown: int | None = None,
                         single_gpu_handoff: bool = False,
                         trusted_lease_id: str | None = None,
                         allocation_key: str | None = None,
                         accelerator_device_ids: list[str] | None = None,
                         evaluation_contract_sha256: str | None = None) -> str:
        if task_id not in self.tasks:
            raise ValueError(f"Unknown task: {task_id}")
        if self.config.single_gpu_handoff and not single_gpu_handoff:
            raise ValueError(
                "this Judge deployment requires explicit single GPU handoff registration"
            )
        handoff_declaration: dict[str, object] | None = None
        if single_gpu_handoff:
            if not self.config.single_gpu_handoff:
                raise ValueError(
                    "Judge server has not enabled single GPU handoff"
                )
            if not all(
                isinstance(value, str) and value
                for value in (
                    trusted_lease_id,
                    allocation_key,
                    evaluation_contract_sha256,
                )
            ):
                raise ValueError("incomplete single GPU handoff declaration")
            allocation = AcceleratorAllocation(
                "nvidia", tuple(accelerator_device_ids or ())
            )
            if allocation.device_ids != (
                self.config.accelerator_device_uuid,
            ) or allocation_key != self.config.accelerator_allocation_key:
                raise ValueError(
                    "registration allocation does not match Judge deployment"
                )
            task = self.tasks[task_id]
            expected_contract = (
                task.portable.evaluation_contract_sha256
                if task.portable is not None
                else None
            )
            if evaluation_contract_sha256 != expected_contract:
                raise ValueError(
                    "registration evaluation contract does not match Judge task"
                )
            annotations = handoff_annotations(
                run_id=run_id,
                trusted_lease_id=trusted_lease_id,
                allocation_key=allocation_key,
                evaluation_contract_sha256=evaluation_contract_sha256,
            )
            handoff_declaration = {
                "trusted_lease_id": trusted_lease_id,
                "allocation_key": allocation_key,
                "allocation": allocation,
                "evaluation_contract_sha256": evaluation_contract_sha256,
                "annotations": annotations,
            }
        judge_cpu_limit, judge_mem_limit, judge_cpuset_undersize = (
            self._registration_judge_size(
                task_id, run_id, backend, judge_cpu_limit, judge_mem_limit
            )
        )
        token = secrets.token_hex(16)
        with self._tokens_lock:
            self.tokens[token] = {
                "task_id": task_id,
                "run_id": run_id,
                "next_agent": 1,
                "next_auto": 1,
                "next_public": 1,
                "next_checkpoint": 1,
                "agent_inflight": False,
                "public_inflight": False,
                "judge_cpu_limit": judge_cpu_limit,
                "judge_mem_limit": judge_mem_limit,
                "judge_cpuset_undersize": judge_cpuset_undersize,
                "max_agent_submissions": max_agent_submissions,
                "submission_cooldown": submission_cooldown,
                "last_agent_submit_at": 0.0,
                "evaluation_enabled": True,
                "single_gpu_handoff": single_gpu_handoff,
                "handoff_declaration": handoff_declaration,
                "handoff_attached": False,
                "admission_closed": False,
            }
            self.run_resource_limits[run_id] = {
                "judge_cpu_limit": judge_cpu_limit,
                "judge_mem_limit": judge_mem_limit,
                "judge_cpuset_undersize": judge_cpuset_undersize,
            }
            if backend and backend != self.backend.backend_name:
                from sforge.harness.backend.factory import create_backend
                self.run_backends[run_id] = create_backend(
                    backend,
                    k8s_namespace=k8s_namespace or self.config.k8s_namespace,
                    k8s_node_selector=k8s_node_selector or self.config.k8s_node_selector,
                    k8s_image_registry=k8s_image_registry or self.config.k8s_image_registry,
                    k8s_kubeconfig=k8s_kubeconfig or self.config.k8s_kubeconfig,
                )
        return token

    def attach_work_container(self, token: str, work_container_id: str) -> HandoffSession:
        """Bind an exact labelled Work container before accepting submissions."""

        with self._tokens_lock:
            info = self.tokens.get(token)
            if info is None:
                raise KeyError("Invalid token")
            declaration = info.get("handoff_declaration")
            if info.get("single_gpu_handoff") is not True or not isinstance(
                declaration, dict
            ):
                raise ResourceHandoffError("session did not request GPU handoff")
            existing = self._handoff_sessions.get(token)
            if existing is not None:
                if existing.lease.binding.identity.work_container_id != work_container_id:
                    raise ResourceHandoffError(
                        "session is already bound to a different Work container"
                    )
                return existing
            allocation_key = str(declaration["allocation_key"])
            owner = self._handoff_allocations.get(allocation_key)
            if owner is not None and owner != token:
                raise ResourceHandoffError(
                    "accelerator allocation already has an active handoff lease"
                )

            requested_identity = {
                "trusted_lease_id": declaration["trusted_lease_id"],
                "run_id": info["run_id"],
                "task_id": info["task_id"],
                "allocation_key": allocation_key,
                "work_container_id": work_container_id,
            }

            annotations = dict(declaration["annotations"])
            backend = self._get_backend(info["run_id"])
            handle = backend.get_container_handle(
                work_container_id,
                expected_annotations=annotations,
            )
            identity = LeaseIdentity(
                trusted_lease_id=str(declaration["trusted_lease_id"]),
                run_id=info["run_id"],
                task_id=info["task_id"],
                allocation_key=allocation_key,
                work_container_id=work_container_id,
            )
            binding = HandoffBinding(
                identity=identity,
                allocation=declaration["allocation"],
                evaluation_contract_sha256=str(
                    declaration["evaluation_contract_sha256"]
                ),
                annotations=annotations,
            )
            # Different physical allocations retain independent lifetime
            # locks.  This short global admission lock only linearizes the
            # cross-allocation history scan with creation of the new durable
            # identity, preventing two processes from simultaneously reusing
            # one scheduler lease ID or run/task authority.
            admission_lock = ExclusiveProcessLock(
                self.config.log_dir
                / "resource_leases"
                / "identity-admission.lock",
                "resource lease identity admission",
                blocking=True,
            )
            try:
                # A process restart never reconstructs authority from JSON. It
                # instead fences any non-closed record until operator recovery.
                for record_path in self.config.log_dir.glob(
                    "runs/*/*/resource_lease.json"
                ):
                    record = read_durable_lease(record_path)
                    allocation_record = record.get("allocation")
                    record_identity = record.get("identity")
                    if not isinstance(record_identity, dict) or not isinstance(
                        allocation_record, dict
                    ):
                        raise ResourceHandoffError(
                            "durable lease record lacks a complete identity"
                        )
                    if (
                        record_identity.get("trusted_lease_id")
                        == requested_identity["trusted_lease_id"]
                        and record_identity != requested_identity
                    ):
                        raise ResourceHandoffError(
                            "trusted lease ID was already bound to another identity"
                        )
                    if (
                        (
                            record_identity.get("run_id"),
                            record_identity.get("task_id"),
                        )
                        == (
                            requested_identity["run_id"],
                            requested_identity["task_id"],
                        )
                        and record_identity != requested_identity
                    ):
                        raise ResourceHandoffError(
                            "run and task already have a different durable lease identity"
                        )
                    if (
                        record.get("state") != LeaseState.CLOSED.value
                        and record_identity != requested_identity
                        and record_identity.get("allocation_key")
                        == allocation_key
                    ):
                        raise ResourceHandoffError(
                            "accelerator has an unreconciled durable lease"
                        )

                lease = DurableResourceLease.create(
                    lease_record_path(
                        self.config.log_dir,
                        info["run_id"],
                        info["task_id"],
                    ),
                    binding,
                )
                lease.transition(LeaseState.WORK_ACTIVE)
            finally:
                admission_lock.close()
            session = HandoffSession(lease, handle, binding.allocation, annotations)
            self._handoff_sessions[token] = session
            self._handoff_allocations[allocation_key] = token
            info["handoff_attached"] = True
            return session

    def seal_handoff(self, token: str) -> HandoffSession:
        """Close admission and durably prohibit every later Work resume."""

        with self._tokens_lock:
            info = self.tokens.get(token)
            session = self._handoff_sessions.get(token)
            if info is None:
                raise KeyError("Invalid token")
            if session is None:
                raise ResourceHandoffError("handoff Work container is not attached")
        with session.control_lock:
            with self._tokens_lock:
                current_info = self.tokens.get(token)
                current_session = self._handoff_sessions.get(token)
                if current_info is not info or current_session is not session:
                    raise ResourceHandoffError(
                        "handoff identity changed while acquiring final fence"
                    )
                info["evaluation_enabled"] = False
                info["admission_closed"] = True
                agent_inflight = info.get("agent_inflight") is True
            # The durable no-resume write happens while holding the same lock
            # as the worker's policy check and Docker unpause operation.
            session.lease.request_final_no_resume()
            snapshot = session.lease.snapshot()
            if snapshot.state == LeaseState.WORK_ACTIVE:
                # The seal response itself is the final writer boundary. This
                # applies both to a queued accepted submission and to an idle
                # session; run_agent must not leave Work runnable while it
                # performs later HTTP/history/archive reconciliation. A queued
                # worker recognizes the pre-paused state and completes without
                # ever resuming Work.
                backend = self._get_backend(info["run_id"])
                try:
                    session.lease.transition(LeaseState.PAUSE_REQUESTED)
                    session.lease.transition(LeaseState.WORK_PAUSING)
                    backend.pause_container_for_handoff(session.work_handle)
                    backend.verify_container_paused_for_handoff(
                        session.work_handle
                    )
                    session.lease.transition(LeaseState.WORK_PAUSED)
                except Exception as exc:
                    try:
                        session.lease.fail_closed(
                            "deadline seal could not synchronously pause Work",
                            LeaseState.FENCED,
                        )
                    except Exception:
                        logger.exception(
                            "Failed to persist deadline pause failure fence"
                        )
                    raise ResourceHandoffError(
                        "deadline seal failed to freeze Work"
                    ) from exc
                snapshot = session.lease.snapshot()
            if not agent_inflight and snapshot.state in {
                LeaseState.WORK_PAUSED,
                LeaseState.WORK_GPU_IDLE_ATTESTED,
                LeaseState.JUDGE_GPU_IDLE_ATTESTED,
            }:
                session.lease.transition(LeaseState.FINALIZING)
        return session

    def finish_final_work(
        self,
        token: str,
        *,
        archive_sha256: str,
        archive_size: int,
    ) -> HandoffSession:
        """Destroy final Work and attest idle; final Judge remains separate."""

        session = self._handoff_sessions.get(token)
        if session is None:
            raise ResourceHandoffError("handoff Work container is not attached")
        with session.control_lock:
            lease = session.lease
            if (
                isinstance(archive_size, bool)
                or not isinstance(archive_size, int)
                or archive_size < 0
            ):
                raise ResourceHandoffError("invalid final archive size")
            if archive_size > SUBMISSION_ARCHIVE_MAX_BYTES:
                raise ResourceHandoffError(
                    "final archive exceeds the Judge submission size limit"
                )
            if (
                not isinstance(archive_sha256, str)
                or _CANONICAL_SHA256_RE.fullmatch(archive_sha256) is None
            ):
                raise ResourceHandoffError("invalid final archive digest")
            if lease.recovery_policy != "never_resume":
                raise ResourceHandoffError(
                    "final Work destruction requires durable never-resume"
                )
            state = lease.snapshot().state
            if state == LeaseState.FINAL_GPU_IDLE_ATTESTED:
                if lease.archive != {
                    "sha256": archive_sha256,
                    "size": archive_size,
                }:
                    raise ResourceHandoffError(
                        "retired final Work archive does not match this request"
                    )
                # The coordinator may have durably completed retirement before
                # its HTTP response was lost.  Exact request replay is safe and
                # must not repeat final-only freeze, destruction, or NVML work.
                return session
            if state != LeaseState.FINALIZING:
                raise ResourceHandoffError(
                    "final Work destruction requires FINALIZING state"
                )
            backend = self._get_backend(lease.binding.identity.run_id)
            backend.freeze_container_for_final_handoff_snapshot(
                session.work_handle,
                expected_annotations=session.annotations,
                stop_without_inspect=True,
            )
            backend.verify_container_frozen_for_snapshot(session.work_handle)
            archive_path = (
                self.config.log_dir
                / "runs"
                / lease.binding.identity.run_id
                / lease.binding.identity.task_id
                / "final_archive.tar.gz"
            )
            descriptor = os.open(archive_path, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                file_stat = os.fstat(descriptor)
                if not stat.S_ISREG(file_stat.st_mode):
                    raise ResourceHandoffError(
                        "final archive is not a regular file"
                    )
                if file_stat.st_size != archive_size:
                    raise ResourceHandoffError("final archive size does not match")
                digest = hashlib.sha256()
                remaining = archive_size
                while remaining:
                    chunk = os.read(
                        descriptor,
                        min(1024 * 1024, remaining),
                    )
                    if not chunk:
                        raise ResourceHandoffError("final archive is truncated")
                    digest.update(chunk)
                    remaining -= len(chunk)
                if os.read(descriptor, 1):
                    raise ResourceHandoffError(
                        "final archive grew while being verified"
                    )
            finally:
                os.close(descriptor)
            if archive_sha256 != "sha256:" + digest.hexdigest():
                raise ResourceHandoffError("final archive digest does not match")

            # Exact stopped state and exact persisted bytes are both verified
            # before irreversible Work destruction.
            lease.transition(LeaseState.FINAL_WORK_FROZEN)
            lease.record_archive(
                sha256=archive_sha256,
                size=archive_size,
            )
            lease.transition(LeaseState.FINAL_ARCHIVE_READY)
            lease.transition(LeaseState.FINAL_WORK_DESTROYING)
            backend.destroy_container_for_handoff(session.work_handle)
            lease.transition(LeaseState.FINAL_WORK_DESTROYED)
            if self.accelerator_observer is None:
                raise ResourceHandoffError(
                    "final Work release denied because no observer is configured"
                )
            idle = self.accelerator_observer.attest_idle(
                session.allocation,
                phase="after_final_work_destroy",
            )
            lease.transition(
                LeaseState.FINAL_GPU_IDLE_ATTESTED,
                attestation=idle,
            )
            # Keep the allocation and live coordinator session bound.  Only
            # the server-owned final Judge may advance FINAL_JUDGE_* and close
            # the lease; Work retirement alone is not allocation release.
        return session

    def _resolve_final_handoff_session(
        self,
        *,
        source_run_id: str,
        task_id: str,
        trusted_lease_id: str,
        allocation_key: str,
        evaluation_contract_sha256: str,
    ) -> HandoffSession:
        """Resolve one live lease by its complete scheduler-owned identity."""

        with self._tokens_lock:
            sessions = list(self._handoff_sessions.values())
        matches = []
        for session in sessions:
            binding = session.lease.binding
            identity = binding.identity
            if (
                identity.run_id == source_run_id
                and identity.task_id == task_id
                and identity.trusted_lease_id == trusted_lease_id
                and identity.allocation_key == allocation_key
                and binding.evaluation_contract_sha256
                == evaluation_contract_sha256
            ):
                matches.append(session)
        if len(matches) != 1:
            raise ResourceHandoffError(
                "final Judge source identity did not resolve exactly one "
                "live handoff lease"
            )
        return matches[0]

    def _read_authenticated_final_archive(
        self,
        session: HandoffSession,
        *,
        archive_sha256: str,
        archive_size: int,
    ) -> bytes:
        if archive_size > SUBMISSION_ARCHIVE_MAX_BYTES:
            raise ResourceHandoffError(
                "final archive exceeds the Judge submission size limit"
            )
        archive_path = (
            self.config.log_dir
            / "runs"
            / session.lease.binding.identity.run_id
            / session.lease.binding.identity.task_id
            / "final_archive.tar.gz"
        )
        descriptor = os.open(archive_path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            file_stat = os.fstat(descriptor)
            if not stat.S_ISREG(file_stat.st_mode):
                raise ResourceHandoffError("final archive is not a regular file")
            if file_stat.st_size != archive_size:
                raise ResourceHandoffError("final archive size does not match")
            payload = bytearray()
            digest = hashlib.sha256()
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                payload.extend(chunk)
                digest.update(chunk)
        finally:
            os.close(descriptor)
        if archive_sha256 != "sha256:" + digest.hexdigest():
            raise ResourceHandoffError("final archive digest does not match")
        return bytes(payload)

    def run_final_judge(
        self,
        *,
        source_run_id: str,
        final_run_id: str,
        task_id: str,
        trusted_lease_id: str,
        allocation_key: str,
        evaluation_contract_sha256: str,
        archive_sha256: str,
        archive_size: int,
        timeout: int | None,
        round_id: str | None = None,
    ) -> tuple[HandoffSession, dict[str, Any]]:
        """Judge only the server-authenticated final Work archive."""

        if _SAFE_FINAL_RUN_ID_RE.fullmatch(final_run_id) is None:
            raise ResourceHandoffError("invalid final run ID")
        if round_id is None:
            round_id = DEFAULT_FINAL_ROUND_ID
        if (
            not isinstance(round_id, str)
            or _SAFE_FINAL_ROUND_ID_RE.fullmatch(round_id) is None
        ):
            raise ResourceHandoffError("invalid final round ID")
        if (
            isinstance(archive_size, bool)
            or not isinstance(archive_size, int)
            or archive_size < 0
        ):
            raise ResourceHandoffError("invalid final archive size")
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, int)
            or timeout <= 0
        ):
            raise ResourceHandoffError("final Judge timeout must be positive")
        # The final Judge private context is derived before any lease
        # transition so a conflicting server environment fails closed without
        # consuming the retired lease's single final attempt.
        final_judge_extra_env = _final_judge_private_context(
            self.config.judge_extra_env,
            archive_sha256=archive_sha256,
        )
        session = self._resolve_final_handoff_session(
            source_run_id=source_run_id,
            task_id=task_id,
            trusted_lease_id=trusted_lease_id,
            allocation_key=allocation_key,
            evaluation_contract_sha256=evaluation_contract_sha256,
        )
        lease = session.lease
        task_spec = self.tasks.get(task_id)
        if task_spec is None:
            raise ResourceHandoffError("final Judge task is not loaded")
        if timeout is not None and timeout != task_spec.judge.eval_timeout:
            raise ResourceHandoffError(
                "final Judge timeout must equal the task evaluation contract"
            )
        effective_timeout = task_spec.judge.eval_timeout
        request_binding: dict[str, object] = {
            "schema_version": 1,
            "source_run_id": source_run_id,
            "final_run_id": final_run_id,
            "task_id": task_id,
            "trusted_lease_id": trusted_lease_id,
            "allocation_key": allocation_key,
            "evaluation_contract_sha256": evaluation_contract_sha256,
            "archive_sha256": archive_sha256,
            "archive_size": archive_size,
            "timeout": effective_timeout,
        }

        with session.control_lock:
            durable_final = lease.final_judge
            if lease.snapshot().state == LeaseState.CLOSED:
                if (
                    durable_final is None
                    or durable_final.get("request") != request_binding
                    or not isinstance(durable_final.get("report"), dict)
                    or not isinstance(durable_final.get("artifacts"), dict)
                ):
                    raise ResourceHandoffError(
                        "closed final Judge lease does not match this request"
                    )
                return session, dict(durable_final["report"])
            try:
                if lease.snapshot().state != LeaseState.FINAL_GPU_IDLE_ATTESTED:
                    raise ResourceHandoffError(
                        "final Judge requires retired Work and final GPU idle proof"
                    )
                if lease.archive != {
                    "sha256": archive_sha256,
                    "size": archive_size,
                }:
                    raise ResourceHandoffError(
                        "final Judge archive does not match retired Work binding"
                    )
                archive = self._read_authenticated_final_archive(
                    session,
                    archive_sha256=archive_sha256,
                    archive_size=archive_size,
                )
                lease.record_final_judge_request(request_binding)
                if self.accelerator_observer is None:
                    raise ResourceHandoffError(
                        "final Judge denied because no observer is configured"
                    )
                pre_idle = self.accelerator_observer.attest_idle(
                    session.allocation,
                    phase="before_final_judge",
                )
                lease.transition(
                    LeaseState.FINAL_JUDGE_STARTING,
                    attestation=pre_idle,
                )
                request_digest = hashlib.sha256(
                    json.dumps(
                        request_binding,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("ascii")
                ).hexdigest()
                lifecycle = _FinalHandoffJudgeLifecycle(
                    self,
                    session,
                    "final-" + request_digest[:24],
                )
                limits = self.run_resource_limits.get(source_run_id, {})
                overrides = {
                    field_name: field_value
                    for field_name, field_value in (
                        ("judge_cpu_limit", limits.get("judge_cpu_limit")),
                        ("judge_mem_limit", limits.get("judge_mem_limit")),
                        ("cpuset_undersize", limits.get("judge_cpuset_undersize")),
                    )
                    if field_value is not None
                }
                overrides["judge_extra_env"] = final_judge_extra_env
                final_config = replace(self.config, **overrides)
                report = (self.submission_evaluator or judge_submission)(
                    task_spec=task_spec,
                    archive=archive,
                    config=final_config,
                    backend=self._get_backend(source_run_id),
                    submission_id="final-" + request_digest[:12],
                    timeout=effective_timeout,
                    log_dir=(
                        self.config.log_dir
                        / "runs"
                        / final_run_id
                        / task_id
                        / "submissions"
                        / "manual-1"
                    ),
                    # Source identity authorizes the immutable archive only.
                    # Preserve the caller's formal-evaluation run identity in
                    # the private Judge context for cohort/seed semantics.
                    run_id=final_run_id,
                    round_id=round_id,
                    accelerator_allocation=session.allocation,
                    container_lifecycle=lifecycle,
                )
                if lease.snapshot().state != LeaseState.FINAL_JUDGE_CLEANED:
                    raise ResourceHandoffError(
                        "final Judge strict cleanup did not reach JUDGE_CLEANED"
                    )
                post_idle = self.accelerator_observer.attest_idle(
                    session.allocation,
                    phase="after_final_judge_destroy",
                )
                lease.transition(
                    LeaseState.FINAL_JUDGE_GPU_IDLE_ATTESTED,
                    attestation=post_idle,
                )
                report_dict = report.to_dict()
                defaults = task_spec.portable.defaults
                selected_ids = defaults.to_dict()
                trusted_contract_binding = {
                    "evaluation_contract_sha256": (
                        task_spec.portable.evaluation_contract_sha256
                    ),
                    "selected_ids": selected_ids,
                }
                if (
                    trusted_contract_binding["evaluation_contract_sha256"]
                    != evaluation_contract_sha256
                ):
                    raise ResourceHandoffError(
                        "final Judge task contract changed after registration"
                    )
                report_metrics = report_dict.get("metrics")
                if (
                    not isinstance(report_metrics, dict)
                    or report_metrics.get("portable_contract")
                    != trusted_contract_binding
                ):
                    raise ResourceHandoffError(
                        "final Judge report lacks the exact trusted contract binding"
                    )
                artifact_dir = (
                    self.config.log_dir
                    / "runs"
                    / final_run_id
                    / task_id
                    / "submissions"
                    / "manual-1"
                )
                persisted_submission = self._read_regular_artifact(
                    artifact_dir / "submission.tar.gz",
                    max_size=SUBMISSION_ARCHIVE_MAX_BYTES,
                )
                if (
                    len(persisted_submission) != archive_size
                    or "sha256:"
                    + hashlib.sha256(persisted_submission).hexdigest()
                    != archive_sha256
                ):
                    raise ResourceHandoffError(
                        "persisted final Judge submission does not match archive"
                    )
                persisted_report = self._read_regular_artifact(
                    artifact_dir / "report.json",
                    max_size=16 * 1024 * 1024,
                )
                try:
                    persisted_report_value = json.loads(persisted_report)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ResourceHandoffError(
                        "persisted final Judge report is malformed"
                    ) from exc
                if persisted_report_value != report_dict:
                    raise ResourceHandoffError(
                        "persisted final Judge report bytes disagree with result"
                    )
                final_artifacts: dict[str, object] = {
                    "submission": {
                        "sha256": archive_sha256,
                        "size": len(persisted_submission),
                    },
                    "report": {
                        "sha256": "sha256:"
                        + hashlib.sha256(persisted_report).hexdigest(),
                        "size": len(persisted_report),
                    },
                }
                if (task_spec.judge.private_artifact_path
                        or getattr(task_spec.judge, 'private_artifact_paths', None)):
                    persisted_private = self._read_regular_artifact(
                        artifact_dir / PRIVATE_JUDGE_ARTIFACT_NAME,
                        max_size=SUBMISSION_ARCHIVE_MAX_BYTES,
                    )
                    if not persisted_private:
                        raise ResourceHandoffError(
                            "persisted final Judge private artifact is empty"
                        )
                    final_artifacts["private_judge"] = {
                        "sha256": "sha256:"
                        + hashlib.sha256(persisted_private).hexdigest(),
                        "size": len(persisted_private),
                    }
                lease.record_final_judge_report(report_dict)
                lease.record_final_judge_artifacts(final_artifacts)
                if report.metrics.get("evaluation_status") == "infrastructure_error":
                    raise ResourceHandoffError(
                        "final Judge reported an infrastructure failure"
                    )
                lease.transition(LeaseState.CLOSED)
                with self._tokens_lock:
                    self._handoff_allocations.pop(allocation_key, None)
                return session, report_dict
            except Exception as exc:
                try:
                    if lease.snapshot().state not in {
                        LeaseState.CLOSED,
                        LeaseState.FENCED,
                        LeaseState.FAILED_STOPPED,
                        LeaseState.FAILED_WORK_ACTIVE,
                    }:
                        lease.fail_closed(
                            f"final Judge failed: {type(exc).__name__}",
                            LeaseState.FENCED,
                        )
                except Exception:
                    logger.exception("Failed to persist final Judge fence")
                raise

    @staticmethod
    def _read_regular_artifact(path: Path, *, max_size: int) -> bytes:
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        except OSError as exc:
            raise ResourceHandoffError(
                f"final Judge artifact is unavailable: {path.name}"
            ) from exc
        try:
            file_stat = os.fstat(descriptor)
            if not stat.S_ISREG(file_stat.st_mode):
                raise ResourceHandoffError(
                    f"final Judge artifact is not regular: {path.name}"
                )
            if file_stat.st_size > max_size:
                raise ResourceHandoffError(
                    f"final Judge artifact exceeds size limit: {path.name}"
                )
            chunks: list[bytes] = []
            remaining = file_stat.st_size
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    raise ResourceHandoffError(
                        f"final Judge artifact is truncated: {path.name}"
                    )
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise ResourceHandoffError(
                    f"final Judge artifact grew while being read: {path.name}"
                )
            return b"".join(chunks)
        finally:
            os.close(descriptor)

    def register_checkpoint_token(self, session_token: str) -> str:
        """Issue a run-scoped token that can only write version checkpoints."""

        checkpoint_token = secrets.token_hex(16)
        with self._tokens_lock:
            session = self.tokens.get(session_token)
            if session is None:
                raise KeyError("Invalid token")
            self.tokens[checkpoint_token] = {
                "task_id": session["task_id"],
                "run_id": session["run_id"],
                "next_checkpoint": 1,
                "evaluation_enabled": False,
            }
        return checkpoint_token

    def _get_backend(self, run_id: str | None = None) -> ContainerBackend:
        if run_id and run_id in self.run_backends:
            return self.run_backends[run_id]
        return self.backend
    def resolve_token(self, token: str) -> dict:
        with self._tokens_lock:
            info = self.tokens.get(token)
        if info is None:
            raise KeyError("Invalid token")
        return dict(info)

    def resolve_evaluation_token(self, token: str) -> dict:
        info = self.resolve_token(token)
        if info.get("evaluation_enabled", True) is not True:
            raise TokenCapabilityDenied("token is limited to version checkpoints")
        return info

    @staticmethod
    def _ensure_submission_available(info: dict, kind: str) -> None:
        """Check queue, budget, and cooldown without allocating a round."""

        if info.get("admission_closed") is True:
            raise TokenCapabilityDenied("submission admission is closed")
        if info.get("single_gpu_handoff") is True:
            if info.get("handoff_attached") is not True:
                raise TokenCapabilityDenied(
                    "single GPU Work container is not attached"
                )
            if kind != "agent":
                raise TokenCapabilityDenied(
                    "single GPU MVP accepts manual agent submissions only"
                )
        if kind == "agent":
            if info.get("agent_inflight") is True:
                raise AgentSubmissionBusy(
                    "an agent submission is already queued or running"
                )
            max_subs = info.get("max_agent_submissions")
            next_n = info.get("next_agent", 1)
            if max_subs is not None and next_n > max_subs:
                raise SubmissionBudgetExceeded(
                    f"Submission budget exhausted ({max_subs}/{max_subs} used)"
                )
            cooldown = info.get("submission_cooldown")
            last_submit = info.get("last_agent_submit_at", 0.0)
            if cooldown is not None and last_submit > 0:
                elapsed = time.monotonic() - last_submit
                if elapsed < cooldown:
                    remaining_wait = int(cooldown - elapsed) + 1
                    raise SubmissionCooldownActive(
                        "Submission cooldown active — wait "
                        f"{remaining_wait}s before next submission. Continue "
                        "working on improvements and submit again later."
                    )
        elif kind == "public":
            if info.get("public_inflight") is True:
                raise PublicEvaluationBusy(
                    "a public evaluation is already queued or running"
                )
        elif kind != "auto":
            raise ValueError("kind must be 'agent', 'auto', or 'public'")

    @contextmanager
    def submission_upload_admission(
        self,
        token: str,
        kind: str,
    ) -> Iterator[dict]:
        """Authenticate and reserve one run's bounded archive upload slot."""

        with self._tokens_lock:
            info = self.tokens.get(token)
            if info is None:
                raise KeyError("Invalid token")
            if info.get("evaluation_enabled") is not True:
                raise TokenCapabilityDenied(
                    "token is limited to version checkpoints"
                )
            self._ensure_submission_available(info, kind)
            upload_key = (info["task_id"], info["run_id"])
            with self._checkpoint_lock:
                if upload_key in self._archive_upload_inflight:
                    raise ArchiveUploadBusy(
                        "an archive upload is already in progress for this run"
                    )
                self._archive_upload_inflight.add(upload_key)
            admitted = dict(info)
        try:
            yield admitted
        finally:
            with self._checkpoint_lock:
                self._archive_upload_inflight.discard(upload_key)

    @contextmanager
    def submission_commit_admission(
        self,
        token: str,
        kind: str,
    ) -> Iterator[None]:
        """Linearize handoff round consumption and durable queue commit.

        Archive upload intentionally happens outside the session control lock.
        Once the bytes have been validated, however, a final seal must observe
        either closed admission or a fully durable queued submission--never a
        consumed/inflight round that has not yet reached the queue manifest.
        """

        with self._tokens_lock:
            info = self.tokens.get(token)
            if info is None:
                raise KeyError("Invalid token")
            handoff = info.get("single_gpu_handoff") is True
            session = self._handoff_sessions.get(token)
        if not handoff:
            yield
            return
        if session is None:
            raise ResourceHandoffError(
                "handoff submission lost its exact token/container binding"
            )
        with session.control_lock:
            with self._tokens_lock:
                current_info = self.tokens.get(token)
                current_session = self._handoff_sessions.get(token)
                if current_info is not info or current_session is not session:
                    raise ResourceHandoffError(
                        "handoff identity changed before submission commit"
                    )
                if info.get("evaluation_enabled") is not True:
                    raise TokenCapabilityDenied(
                        "handoff admission closed before submission commit"
                    )
                self._ensure_submission_available(info, kind)
            yield

    def consume_round(self, token: str, kind: str = "agent") -> tuple[str, str, str, int | None, str | None, int | None]:
        """Allocate the next round ID for a token. Returns (task_id, run_id, round_id, cpu, mem, remaining).

        For agent submissions, enforces one queued/running submission per
        session in addition to max_agent_submissions and submission_cooldown.
        Raises AgentSubmissionBusy, SubmissionBudgetExceeded, or
        SubmissionCooldownActive on violation.
        Auto submissions are never limited. Public submissions are unlimited
        and do not consume the Agent cooldown, but only one may be queued or
        running per session so an untrusted Work container cannot create an
        unbounded ten-minute evaluation backlog.
        """
        with self._tokens_lock:
            info = self.tokens.get(token)
            if info is None:
                raise KeyError("Invalid token")
            if info.get("evaluation_enabled") is not True:
                raise TokenCapabilityDenied(
                    "token is limited to version checkpoints"
                )

            self._ensure_submission_available(info, kind)
            if kind == "agent":
                info["last_agent_submit_at"] = time.monotonic()
                info["agent_inflight"] = True
            elif kind == "public":
                info["public_inflight"] = True

            counter_key = f"next_{kind}"
            n = info.get(counter_key, 1)
            round_id = f"{kind}-{n}"
            info[counter_key] = n + 1

            remaining = None
            if kind == "agent":
                max_subs = info.get("max_agent_submissions")
                if max_subs is not None:
                    remaining = max_subs - n

        return info["task_id"], info["run_id"], round_id, info.get("judge_cpu_limit"), info.get("judge_mem_limit"), remaining

    def _version_context(self, token: str) -> tuple[dict, TaskSpec]:
        info = self.resolve_token(token)
        task_id = info["task_id"]
        if self.version_store is None or not self.version_store.supports_task(task_id):
            raise LookupError("version saving is not enabled for this task")
        return info, self.tasks[task_id]

    def validate_version_archive(self, token: str, archive: bytes) -> None:
        """Validate candidate bytes before consuming a Judge round."""

        info, task = self._version_context(token)
        assert self.version_store is not None
        self.version_store.validate_archive(
            task_id=info["task_id"],
            archive=archive,
            submit_paths=task.submit_paths,
            submit_exclude=task.submit_exclude,
        )

    def resolve_saved_version(self, token: str, version_id: str) -> StoredVersion:
        """Resolve an immutable version inside this token's exact task and run.

        A caller supplies only an identifier, never a filesystem path, object
        prefix or another run ID. VersionStore performs no-follow manifest
        reads; ``read_archive`` separately verifies the bytes before admission.
        """
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,254}", version_id) is None:
            raise VersionStoreCandidateError("invalid version_id")
        info, _ = self._version_context(token)
        assert self.version_store is not None
        matches = self.version_store.list_versions(
            task=info["task_id"], run_id=info["run_id"],
        )
        for saved in matches:
            if saved.version_id == version_id:
                return saved
        raise LookupError("saved version not found in this task/run")

    def save_checkpoint(
        self,
        token: str,
        archive: bytes,
        label: str,
    ) -> StoredVersion:
        with self.checkpoint_upload_admission(token) as (info, task):
            return self._save_checkpoint_admitted(
                token,
                info,
                task,
                archive,
                label,
            )

    @contextmanager
    def checkpoint_upload_admission(
        self,
        token: str,
    ) -> Iterator[tuple[dict, TaskSpec]]:
        """Reserve checkpoint upload/storage work before reading its body."""

        info, task = self._version_context(token)
        checkpoint_key = (info["task_id"], info["run_id"])
        with self._checkpoint_lock:
            if (
                checkpoint_key in self._checkpoint_inflight
                or checkpoint_key in self._archive_upload_inflight
            ):
                raise CheckpointBusy("a checkpoint save is already in progress")
            now = time.monotonic()
            previous = self._checkpoint_last_started.get(checkpoint_key)
            if (
                previous is not None
                and now - previous < CHECKPOINT_MIN_INTERVAL_SECONDS
            ):
                raise CheckpointRateLimited("checkpoint rate limit is active")
            self._checkpoint_inflight.add(checkpoint_key)
            self._archive_upload_inflight.add(checkpoint_key)
            self._checkpoint_last_started[checkpoint_key] = now
        try:
            yield info, task
        finally:
            with self._checkpoint_lock:
                self._checkpoint_inflight.discard(checkpoint_key)
                self._archive_upload_inflight.discard(checkpoint_key)

    def _save_checkpoint_admitted(
        self,
        token: str,
        info: dict,
        task: TaskSpec,
        archive: bytes,
        label: str,
    ) -> StoredVersion:
        """Persist a checkpoint while ``checkpoint_upload_admission`` is held."""

        assert self.version_store is not None
        self.version_store.validate_archive(
            task_id=info["task_id"],
            archive=archive,
            submit_paths=task.submit_paths,
            submit_exclude=task.submit_exclude,
        )
        with self._tokens_lock:
            live = self.tokens.get(token)
            if live is None:
                raise KeyError("Invalid token")
            number = live.get("next_checkpoint", 1)
            round_id = f"checkpoint-{number}"
            live["next_checkpoint"] = number + 1
        return self.version_store.record_archive(
            task_id=info["task_id"],
            run_id=info["run_id"],
            source="checkpoint",
            round_id=round_id,
            label=label,
            archive=archive,
            submit_paths=task.submit_paths,
            submit_exclude=task.submit_exclude,
            max_bytes_for_source=CHECKPOINT_MAX_BYTES_PER_RUN,
            content_identity_exclude=task.candidate_identity_exclude,
        )

    def persist_submission_version(
        self,
        *,
        task_id: str,
        run_id: str,
        source: str,
        round_id: str,
        archive: bytes,
    ) -> StoredVersion | None:
        if self.version_store is None or not self.version_store.supports_task(task_id):
            return None
        task = self.tasks[task_id]
        return self.version_store.record_archive(
            task_id=task_id,
            run_id=run_id,
            source=source,
            round_id=round_id,
            label=None,
            archive=archive,
            submit_paths=task.submit_paths,
            submit_exclude=task.submit_exclude,
            content_identity_exclude=task.candidate_identity_exclude,
        )

    def release_agent_round(self, token: str) -> None:
        """Release one session's Agent queue slot after a terminal result."""
        with self._tokens_lock:
            info = self.tokens.get(token)
            if info is not None:
                info["agent_inflight"] = False

    def release_public_round(self, token: str) -> None:
        """Release one session's public queue slot after a terminal result."""
        with self._tokens_lock:
            info = self.tokens.get(token)
            if info is not None:
                info["public_inflight"] = False

    def _record_submission(self, run_id: str, submission_id: str, task_id: str, round_id: str | None, report_dict: dict | None, error: str | None, status: str = "completed") -> None:
        entry_type = (
            "public"
            if (round_id or "").startswith("public-")
            else "submission"
        )
        entry: dict[str, Any] = {
            "type": entry_type,
            "status": status,
            "submission_id": submission_id,
            "task_id": task_id,
            "round": round_id,
        }
        submission = self.submissions.get(submission_id, {})
        for key in ("version_id", "candidate_id", "content_sha256", "archive_sha256", "archive_size"):
            value = submission.get(key)
            if value is not None:
                entry[key] = value
        review = submission.get("integrity_review")
        if isinstance(review, IntegrityReview):
            # Host-side audit field; the agent-visible history strips it.
            entry["integrity_review"] = review.history_summary()
        if report_dict:
            entry["submitted_at"] = report_dict.get("submitted_at", 0.0)
            entry["pass_rate"] = report_dict.get("pass_rate", 0.0)
            entry["score"] = report_dict.get("score")
            entry["passed"] = report_dict.get("passed", 0)
            entry["failed"] = report_dict.get("failed", 0)
            entry["total_tests"] = report_dict.get("total_tests", 0)
            entry["valid"] = report_dict.get("valid", True)
            entry["summary"] = report_dict.get("summary")
            if (round_id or "").startswith("agent-"):
                family_feedback = _safe_valid_family_bands(report_dict)
                if family_feedback is not None:
                    entry["deployment_family_feedback"] = family_feedback
        if error:
            entry["error"] = error
        history_key = f"{run_id}/{task_id}"
        with self._history_lock:
            entries = self.run_history.setdefault(history_key, [])
            for i, e in enumerate(entries):
                if e.get("submission_id") == submission_id:
                    entries[i] = entry
                    return
            entries.append(entry)

    def get_run_history(self, run_id: str, task_id: str | None = None) -> dict:
        history_key = f"{run_id}/{task_id}" if task_id else run_id
        with self._history_lock:
            entries = list(self.run_history.get(history_key, []))

        # Resolve selection policy from task spec
        resolved_tid = task_id
        if not resolved_tid:
            for e in entries:
                if e.get("task_id"):
                    resolved_tid = e["task_id"]
                    break
        task_spec = self.tasks.get(resolved_tid) if resolved_tid else None
        policy = task_spec.judge.selection if task_spec else "pass_rate_first"
        direction = task_spec.judge.score_direction if task_spec else "maximize"

        # Public diagnostics use a different ten-minute workload and frozen
        # Direct anchor. They are history, never candidates for hidden-score
        # selection.
        selection_entries = [e for e in entries if e.get("type") != "public"]
        best = select_best(selection_entries, direction, policy)

        sub_entries = [e for e in entries if e.get("type") == "submission"]
        return {
            "run_id": run_id,
            "best_score": best["best_score"],
            "best_pass_rate": best["best_pass_rate"],
            "best_round": best["best_round"],
            "agent_submissions": sum(1 for e in sub_entries if (e.get("round") or "").startswith("agent-")),
            "auto_submissions": sum(1 for e in sub_entries if (e.get("round") or "").startswith("auto-")),
            "public_submissions": sum(1 for e in entries if e.get("type") == "public"),
            "entries": entries,
        }

    def load_tasks(self) -> None:
        benchmark = load_benchmark(self.config.tasks_dir)
        task_list = load_all_tasks(self.config.tasks_dir, benchmark)
        self.tasks = {t.task_id: t for t in task_list}
        if self.integrity_reviewer is not None:
            self.integrity_reviewer.validate_tasks(task_list)

    def integrity_review_notice(self, task_id: str) -> dict[str, Any] | None:
        """Registration hint for runs whose agent submissions are reviewed.

        ``announce`` tells the runner whether to describe the (enforcing)
        review in the agent prompt; ``review_timeout_seconds`` lets it extend
        its end-of-run Judge drain, since a queued submission may spend that
        long in review before its Judge even starts.
        """

        reviewer = self.integrity_reviewer
        task_spec = self.tasks.get(task_id)
        if reviewer is None or task_spec is None:
            return None
        if not reviewer.covers(task_spec, "agent"):
            return None
        return {
            "mode": reviewer.config.mode,
            "announce": reviewer.announced_to_agent(task_spec),
            "review_timeout_seconds": int(reviewer.config.review_timeout_seconds),
        }

    def _baseline_provider(self, run_id: str | None) -> BaselineProvider:
        if run_id is not None:
            with self._baseline_providers_lock:
                provided = self._run_baselines.get(run_id)
            if provided is not None:
                return provided
        backend = self._get_backend(run_id)
        if backend.backend_name != "docker":
            return NoBaseline(
                f"the {backend.backend_name} backend cannot read Work images"
            )
        with self._baseline_providers_lock:
            provider = self._baseline_providers.get(id(backend))
            if provider is None:
                provider = DockerWorkImageBaseline(backend.client)
                self._baseline_providers[id(backend)] = provider
            return provider

    def _run_integrity_review(
        self,
        submission_id: str,
        task_id: str,
        archive: bytes,
        run_id: str | None,
        round: str | None,
    ) -> None:
        """Review one queued submission before it may reach the GPU Judge.

        Runs without the task's Judge lock and before any single-GPU handoff:
        the archive is immutable and the review needs no accelerator, so Work
        keeps running and other Judges are not delayed.  The outcome is kept
        on the submission record for ``_grade_worker_serial``.
        """

        reviewer = self.integrity_reviewer
        task_spec = self.tasks.get(task_id)
        if task_spec is None:
            return
        kind = (round or "").split("-", 1)[0]
        config = self.config.llm_judge
        if reviewer is None:
            if config.enabled and config.covers(kind):
                # Mandatory but never configured for this backend.
                self.submissions[submission_id]["integrity_review"] = IntegrityReview(
                    review_id=submission_id,
                    task_id=task_id,
                    mode=config.mode,
                    status=STATUS_ERROR,
                    blocking=config.enforcing and config.on_error == "block",
                    reason="no integrity review engine is configured for this backend",
                )
            return
        if not reviewer.covers(task_spec, kind):
            return
        log_run_id = run_id or submission_id
        sub_log_dir = (
            self.config.log_dir / "runs" / log_run_id
            / task_id / "submissions" / (str(round) if round is not None else "1")
        )
        try:
            review = reviewer.review(
                task_spec=task_spec,
                archive=archive,
                baseline=self._baseline_provider(run_id),
                log_dir=sub_log_dir,
            )
        except Exception as exc:
            logger.exception("Integrity review of %s crashed", submission_id)
            config = reviewer.config
            review = IntegrityReview(
                review_id=submission_id,
                task_id=task_id,
                mode=config.mode,
                status=STATUS_ERROR,
                blocking=config.enforcing and config.on_error == "block",
                reason="the integrity review failed internally",
                error=f"{type(exc).__name__}: {exc}"[:500],
            )
        self.submissions[submission_id]["integrity_review"] = review

    def _integrity_terminal_report(
        self,
        task_spec: TaskSpec,
        submission_id: str,
        archive: bytes,
        review: IntegrityReview,
        log_dir: Path,
    ):
        """Terminal report for a submission the review stops before the Judge.

        Mirrors what ``judge_submission`` persists (exact archive bytes plus
        ``report.json``) so audit and replay tooling see an ordinary
        submission directory.
        """

        submitted_at = time.time()
        log_dir.mkdir(parents=True, exist_ok=True)
        if archive:
            _durable_atomic_write_bytes(log_dir / "submission.tar.gz", archive)
        if review.status == STATUS_VIOLATION:
            report = rejected_report(
                task_spec, submission_id, review, submitted_at=submitted_at
            )
        else:
            report = unavailable_report(
                task_spec, submission_id, review, submitted_at=submitted_at
            )
        report.score_0_100 = rescale_score(task_spec.judge.rescale, report.score)
        _persist_report(log_dir, report)
        return report

    def _get_task_judge_lock(self, task_id: str) -> threading.Lock:
        """Return the process-local serialization lock for one task."""
        with self._task_judge_locks_guard:
            return self._task_judge_locks.setdefault(
                task_id, threading.Lock()
            )

    def _persist_handoff_queue_entry(
        self,
        *,
        token: str | None,
        submission_id: str,
        task_id: str,
        run_id: str | None,
        round_id: str | None,
        archive: bytes,
    ) -> None:
        """Make accepted bytes and queued state durable before pause can run."""

        if token is None or token not in self._handoff_sessions:
            return
        if run_id is None or round_id is None:
            raise ResourceHandoffError(
                "handoff submission requires authoritative run and round IDs"
            )
        directory = (
            self.config.log_dir
            / "runs"
            / run_id
            / task_id
            / "submissions"
            / round_id
        )
        digest = hashlib.sha256(archive).hexdigest()
        _atomic_write(directory / "queued_submission.tar.gz", archive)
        _atomic_write(
            directory / "queued_submission.json",
            (
                json.dumps(
                    {
                        "schema_version": 1,
                        "submission_id": submission_id,
                        "task_id": task_id,
                        "run_id": run_id,
                        "round_id": round_id,
                        "status": "queued",
                        "archive_sha256": digest,
                        "archive_size": len(archive),
                        "persisted_at": time.time(),
                    },
                    sort_keys=True,
                    indent=2,
                )
                + "\n"
            ).encode(),
        )

    def submit(self, task_id: str, archive: bytes, run_id: str | None = None, round: str | None = None,
               judge_cpu_limit: int | None = None, judge_mem_limit: str | None = None,
               session_token: str | None = None,
               stored_version: StoredVersion | None = None,
               background_tasks: BackgroundTasks | None = None) -> str:
        """Submit and grade asynchronously. Returns submission_id for polling.

        Direct callers retain the historical immediate daemon-thread behavior.
        HTTP callers pass ``BackgroundTasks`` so grading cannot begin until
        FastAPI has sent the queued ``SubmitResponse``.
        """
        if task_id not in self.tasks:
            self._release_submission_round(session_token, round)
            raise ValueError(f"Unknown task: {task_id}")

        submission_id = uuid.uuid4().hex[:12]
        log_run_id = run_id or submission_id
        try:
            self.submissions[submission_id] = {
                "status": SubmissionStatus.QUEUED,
                "task_id": task_id,
                "run_id": run_id,
                "round": round,
                "report": None,
                "error": None,
                "version_id": (
                    stored_version.version_id if stored_version is not None else None
                ),
                "candidate_id": (
                    stored_version.candidate_id if stored_version is not None else None
                ),
                "content_sha256": (
                    stored_version.content_sha256 if stored_version is not None else None
                ),
                "archive_sha256": (
                    stored_version.archive_sha256 if stored_version is not None else None
                ),
                "archive_size": (
                    stored_version.archive_size if stored_version is not None else None
                ),
                "session_token": session_token,
            }
            self._persist_handoff_queue_entry(
                token=session_token,
                submission_id=submission_id,
                task_id=task_id,
                run_id=run_id,
                round_id=round,
                archive=archive,
            )
            self._record_submission(
                log_run_id,
                submission_id,
                task_id,
                round,
                None,
                None,
                status="queued",
            )

            worker_args = (
                submission_id, task_id, archive, run_id, round,
                judge_cpu_limit, judge_mem_limit, session_token,
            )
            if background_tasks is not None:
                background_tasks.add_task(self._grade_worker, *worker_args)
            else:
                thread = threading.Thread(
                    target=self._grade_worker,
                    args=worker_args,
                    daemon=True,
                )
                thread.start()
        except Exception as exc:
            submission = self.submissions.get(submission_id)
            if submission is not None:
                submission["status"] = SubmissionStatus.ERROR
                submission["error"] = str(exc)
                try:
                    self._record_submission(
                        log_run_id,
                        submission_id,
                        task_id,
                        round,
                        None,
                        str(exc),
                        status="error",
                    )
                except Exception:
                    logger.exception(
                        "Failed to record submission scheduling error %s",
                        submission_id,
                    )
            self._release_submission_round(session_token, round)
            raise
        return submission_id

    def _release_submission_round(
        self,
        session_token: str | None,
        round: str | None,
    ) -> None:
        if not session_token:
            return
        if (round or "").startswith("agent-"):
            self.release_agent_round(session_token)
        elif (round or "").startswith("public-"):
            self.release_public_round(session_token)

    def _grade_worker(self, submission_id: str, task_id: str, archive: bytes,
                      run_id: str | None = None, round: str | None = None,
                      judge_cpu_limit: int | None = None, judge_mem_limit: str | None = None,
                      session_token: str | None = None) -> None:
        try:
            self._run_integrity_review(
                submission_id, task_id, archive, run_id, round
            )
            with self._get_task_judge_lock(task_id):
                self._grade_worker_serial(
                    submission_id,
                    task_id,
                    archive,
                    run_id,
                    round,
                    judge_cpu_limit,
                    judge_mem_limit,
                    session_token,
                )
        finally:
            self._release_submission_round(session_token, round)
            self._settle_final_after_submission(session_token)

    def _settle_final_after_submission(
        self,
        session_token: str | None,
    ) -> None:
        """Finish a final fence that raced the tail of a Judge worker."""

        if session_token is None:
            return
        session = self._handoff_sessions.get(session_token)
        if session is None:
            return
        with session.control_lock:
            if session.lease.recovery_policy != "never_resume":
                return
            state = session.lease.snapshot().state
            if state in {
                LeaseState.WORK_ACTIVE,
                LeaseState.WORK_PAUSED,
                LeaseState.WORK_GPU_IDLE_ATTESTED,
                LeaseState.JUDGE_GPU_IDLE_ATTESTED,
            }:
                session.lease.transition(LeaseState.FINALIZING)

    def _grade_worker_serial(self, submission_id: str, task_id: str, archive: bytes,
                             run_id: str | None = None, round: str | None = None,
                             judge_cpu_limit: int | None = None,
                             judge_mem_limit: str | None = None,
                             session_token: str | None = None) -> None:
        log_run_id = run_id or submission_id
        recorded_token = self.submissions[submission_id].get("session_token")
        identity_error = recorded_token != session_token
        with self._tokens_lock:
            token_info = dict(self.tokens.get(session_token, {}))
        handoff_session = self._handoff_sessions.get(session_token)
        if (
            handoff_session is None
            and token_info.get("single_gpu_handoff") is not True
        ):
            self.submissions[submission_id]["status"] = SubmissionStatus.RUNNING
            self._record_submission(
                log_run_id,
                submission_id,
                task_id,
                round,
                None,
                None,
                status="running",
            )
        try:
            if identity_error:
                raise ResourceHandoffError(
                    "submission worker token does not match its queued identity"
                )
            if (
                token_info.get("single_gpu_handoff") is True
                and handoff_session is None
            ):
                raise ResourceHandoffError(
                    "handoff submission lost its exact token/container binding"
                )
            task_spec = self.tasks[task_id]
            is_public = (round or "").startswith("public-")
            if is_public and not task_spec.judge.public_eval_cmd:
                raise ValueError(f"Task {task_id} has no public evaluator")
            sub_num = str(round) if round is not None else "1"
            sub_log_dir = (
                self.config.log_dir / "runs" / log_run_id
                / task_id / "submissions" / sub_num
            )
            overrides: dict = {}
            if judge_cpu_limit is not None:
                overrides["judge_cpu_limit"] = judge_cpu_limit
            if judge_mem_limit is not None:
                overrides["judge_mem_limit"] = judge_mem_limit
            if token_info.get("judge_cpuset_undersize") is not None:
                overrides["cpuset_undersize"] = token_info["judge_cpuset_undersize"]
            config = replace(self.config, **overrides) if overrides else self.config
            judge_kwargs = {
                "task_spec": task_spec,
                "archive": archive,
                "config": config,
                "backend": self._get_backend(run_id),
                "submission_id": submission_id,
                "log_dir": sub_log_dir,
                "timeout": (
                    task_spec.judge.public_eval_timeout if is_public else None
                ),
                "eval_cmd": (
                    task_spec.judge.public_eval_cmd if is_public else None
                ),
                "private_artifact_path": (
                    task_spec.judge.public_private_artifact_path
                    if is_public
                    else task_spec.judge.private_artifact_path
                ),
                # Preserve the authoritative request context.  log_run_id may
                # fall back to the per-submission UUID for filesystem layout,
                # but private cohort allocators must fail closed when a real
                # run identifier was not supplied.
                "run_id": run_id,
                "round_id": round,
            }
            if is_public:
                # Public diagnostics must not inherit the formal Judge's
                # private evidence set, including per-evaluation nonces.
                public_artifact = task_spec.judge.public_private_artifact_path
                judge_kwargs['private_artifact_paths'] = (
                    [public_artifact] if public_artifact is not None else [])
            review = self.submissions[submission_id].get("integrity_review")
            if isinstance(review, IntegrityReview) and review.blocking:
                # Rejected (or unreviewable under on_error=block): no Judge
                # container, and in single-GPU mode Work is never paused.
                report = self._integrity_terminal_report(
                    task_spec, submission_id, archive, review, sub_log_dir
                )
                if handoff_session is not None:
                    # Like the Judge path, settle a deadline fence that
                    # arrived during the review *before* the terminal history
                    # becomes visible: the runner reads the lease as soon as
                    # nothing is pending and requires FINALIZING.
                    self._settle_final_after_submission(session_token)
            elif handoff_session is not None:
                report = self._judge_with_handoff(
                    handoff_session,
                    submission_id=submission_id,
                    task_id=task_id,
                    run_id=log_run_id,
                    round_id=round,
                    judge_kwargs=judge_kwargs,
                )
            else:
                report = (self.submission_evaluator or judge_submission)(**judge_kwargs)
            report_dict = report.to_dict()
            report_dict.pop("score_0_100", None)
            if (round or "").startswith("agent-"):
                history_key = f"{log_run_id}/{task_id}"
                with self._history_lock:
                    previous_entries = list(
                        self.run_history.get(history_key, [])
                    )
                _add_family_band_trends(report_dict, previous_entries)
            self.submissions[submission_id]["status"] = SubmissionStatus.COMPLETED
            self.submissions[submission_id]["report"] = report_dict
            self._record_submission(log_run_id, submission_id, task_id, round, report_dict, None)
        except Exception as e:
            self.submissions[submission_id]["status"] = SubmissionStatus.ERROR
            self.submissions[submission_id]["error"] = str(e)
            self._record_submission(log_run_id, submission_id, task_id, round, None, str(e))

    def _judge_with_handoff(
        self,
        session: HandoffSession,
        *,
        submission_id: str,
        task_id: str,
        run_id: str,
        round_id: str | None,
        judge_kwargs: dict[str, object],
    ):
        """Run one manual Judge only after pause plus real idle proof."""

        lease = session.lease
        backend = judge_kwargs["backend"]
        try:
            if self.accelerator_observer is None:
                raise ResourceHandoffError(
                    "Judge start denied because no accelerator observer is configured"
                )
            with session.control_lock:
                state = lease.snapshot().state
                if state == LeaseState.WORK_ACTIVE:
                    lease.transition(LeaseState.PAUSE_REQUESTED)
                    lease.transition(LeaseState.WORK_PAUSING)
                    backend.pause_container_for_handoff(session.work_handle)
                    backend.verify_container_paused_for_handoff(
                        session.work_handle
                    )
                    lease.transition(LeaseState.WORK_PAUSED)
                elif not (
                    state == LeaseState.WORK_PAUSED
                    and lease.recovery_policy == "never_resume"
                ):
                    raise ResourceHandoffError(
                        "Judge worker did not find active or deadline-paused Work"
                    )
            try:
                idle = self.accelerator_observer.attest_idle(
                    session.allocation,
                    phase="after_work_pause_before_judge",
                )
            except AcceleratorBusyError as exc:
                self._recover_attributed_busy(session, backend, exc)
            lease.transition(
                LeaseState.WORK_GPU_IDLE_ATTESTED,
                attestation=idle,
            )
            lease.transition(LeaseState.JUDGE_STARTING)
            lifecycle = _HandoffJudgeLifecycle(
                self,
                session,
                submission_id,
                run_id,
                task_id,
                round_id,
            )
            report = (self.submission_evaluator or judge_submission)(
                **judge_kwargs,
                accelerator_allocation=session.allocation,
                container_lifecycle=lifecycle,
            )
            # Strict Judge destruction is performed by judge_submission's
            # lifecycle-aware finally block before this second observation.
            try:
                post_idle = self.accelerator_observer.attest_idle(
                    session.allocation,
                    phase="after_judge_destroy_before_work_resume",
                )
            except AcceleratorBusyError as exc:
                self._recover_attributed_busy(session, backend, exc)
            lease.transition(
                LeaseState.JUDGE_GPU_IDLE_ATTESTED,
                attestation=post_idle,
            )
            with session.control_lock:
                if lease.recovery_policy == "never_resume":
                    lease.transition(LeaseState.FINALIZING)
                else:
                    lease.transition(LeaseState.WORK_UNPAUSING)
                    backend.unpause_container_after_handoff(session.work_handle)
                    backend.verify_container_running_after_handoff(
                        session.work_handle
                    )
                    lease.transition(LeaseState.WORK_ACTIVE)
            return report
        except HandoffResourceBusy:
            # A stronger observer proved that every busy context belongs to
            # the paused Work owner.  Recovery (or finalization) has already
            # been persisted, so this attempt may terminate without fencing.
            raise
        except Exception as exc:
            try:
                if lease.snapshot().state not in {
                    LeaseState.FENCED,
                    LeaseState.FAILED_STOPPED,
                    LeaseState.FAILED_WORK_ACTIVE,
                }:
                    lease.fail_closed(
                        f"single GPU handoff failed: {type(exc).__name__}",
                        LeaseState.FENCED,
                    )
            except Exception:
                logger.exception("Failed to persist handoff failure fence")
            raise

    def _recover_attributed_busy(
        self,
        session: HandoffSession,
        backend: ContainerBackend,
        busy: AcceleratorBusyError,
    ) -> None:
        """Recover only a busy result with trusted Work-owner attribution."""

        if not busy.recoverable:
            raise busy
        lease = session.lease
        with session.control_lock:
            state = lease.snapshot().state
            if state not in {LeaseState.WORK_PAUSED, LeaseState.JUDGE_CLEANED}:
                raise ResourceHandoffError(
                    "recoverable accelerator busy arrived in an unsafe lease state"
                ) from busy
            if lease.recovery_policy == "never_resume":
                if state == LeaseState.JUDGE_CLEANED:
                    lease.recover_attributed_busy(LeaseState.FINALIZING)
                else:
                    lease.transition(LeaseState.FINALIZING)
            else:
                if state == LeaseState.JUDGE_CLEANED:
                    lease.recover_attributed_busy(LeaseState.WORK_UNPAUSING)
                else:
                    lease.transition(LeaseState.WORK_UNPAUSING)
                backend.unpause_container_after_handoff(session.work_handle)
                backend.verify_container_running_after_handoff(
                    session.work_handle
                )
                lease.transition(LeaseState.WORK_ACTIVE)
        raise HandoffResourceBusy(
            "accelerator remained busy only with contexts attributed to Work; "
            "Judge was not started or was strictly destroyed, and Work was "
            "safely recovered for a later retry"
        ) from busy

    def get_result(self, submission_id: str) -> dict | None:
        return self.submissions.get(submission_id)

    # --- Game session management ---

    def create_game_session(self, run_id: str, task_id: str) -> tuple[GameSessionState, dict]:
        """Start a game container and create a new game session.

        Returns (session_state, response_from_container).
        """
        task_spec = self.tasks.get(task_id)
        if task_spec is None:
            raise ValueError(f"Unknown task: {task_id}")
        if not task_spec.game_mode:
            raise ValueError(f"Task {task_id} is not a game task")
        if not task_spec.judge.game_server_cmd:
            raise ValueError(f"Task {task_id} has no game_server_cmd")

        with self._game_lock:
            if len(self.game_sessions) >= GAME_SESSION_MAX:
                raise RuntimeError(f"Too many active game sessions ({GAME_SESSION_MAX})")

        # Evict oldest sessions for this run+task if at per-task cap
        with self._game_lock:
            task_sessions = sorted(
                [s for s in self.game_sessions.values() if s.run_id == run_id and s.task_id == task_id],
                key=lambda s: s.game_num,
            )
            to_evict = task_sessions[:len(task_sessions) - GAME_SESSION_PER_RUN_MAX + 1] if len(task_sessions) >= GAME_SESSION_PER_RUN_MAX else []
        for sess in to_evict:
            logger.info("Evicting oldest game session %s for %s/%s", sess.session_id, run_id, task_id)
            self._destroy_game_session(sess.session_id)

        backend = self._get_backend(run_id)

        if backend.backend_name == "docker":
            from sforge.harness.backend.docker_backend import DockerBackend
            assert isinstance(backend, DockerBackend)
            build_judge_image(task_spec, self.config, backend.client, force_rebuild=False)

        session_id = uuid.uuid4().hex[:12]
        container_name = f"sforge.game.{task_id}.{session_id}"
        port = GAME_CONTAINER_PORT
        env = get_container_env(self.config, include_judge_extra=True)

        rl = self.run_resource_limits.get(run_id, {})
        cpu = rl.get("judge_cpu_limit")
        mem = rl.get("judge_mem_limit")
        # fall back to task defaults
        if cpu is None:
            cpu = task_spec.judge.cpu_limit
        if mem is None:
            mem = task_spec.judge.mem_limit
        handle = backend.create_container(
            task_spec.judge_image_key,
            container_name,
            environment=env,
            cpu_limit=cpu,
            mem_limit=mem,
        )
        backend.start_container(handle)
        logger.info("Game container started: %s (%s)", container_name, handle.id[:12])

        backend.copy_to_container(handle, GAME_SERVER_APP_PATH, PurePosixPath("/tmp/game_server_app.py"))

        backend.exec_run(
            handle,
            f"/bin/bash -c '{task_spec.judge.game_server_cmd} &'",
            detach=True,
        )

        container_ip = backend.get_container_ip(handle)
        if not container_ip:
            backend.cleanup_container(handle, logger)
            raise RuntimeError("Cannot determine container IP address")

        container_url = f"http://{container_ip}:{port}"

        deadline = time.time() + GAME_CONTAINER_READY_TIMEOUT
        while time.time() < deadline:
            try:
                resp = requests.get(f"{container_url}/health", timeout=2)
                if resp.status_code == 200:
                    break
            except requests.ConnectionError:
                pass
            time.sleep(GAME_CONTAINER_READY_POLL)
        else:
            backend.cleanup_container(handle, logger)
            raise RuntimeError("Game container did not become ready in time")

        resp = requests.post(
            f"{container_url}/new",
            json={},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        with self._history_lock:
            history_key = f"{run_id}/{task_id}"
            archived = sum(1 for e in self.run_history.get(history_key, []) if e.get("type") == "game")
        with self._game_lock:
            active = sum(1 for s in self.game_sessions.values() if s.run_id == run_id)
        game_num = archived + active + 1

        sess = GameSessionState(
            session_id=session_id,
            run_id=run_id,
            task_id=task_id,
            game_num=game_num,
            container=handle,
            container_url=container_url,
            max_score=data.get("max_score", 0),
            peak_score=data.get("peak_score", 0),
            current_score=data.get("score", 0),
            moves=data.get("moves", 0),
            done=data.get("done", False),
        )
        with self._game_lock:
            self.game_sessions[session_id] = sess

        return sess, data

    def game_step(self, session_id: str, action: str) -> dict:
        with self._game_lock:
            sess = self.game_sessions.get(session_id)
        if sess is None:
            raise KeyError(f"Game session not found: {session_id}")

        resp = requests.post(
            f"{sess.container_url}/step",
            json={"action": action},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        sess.current_score = data.get("score", sess.current_score)
        sess.max_score = data.get("max_score", sess.max_score)
        sess.peak_score = data.get("peak_score", sess.peak_score)
        sess.moves = data.get("moves", sess.moves)
        sess.done = data.get("done", sess.done)
        sess.last_active = time.time()
        sess.steps.append({
            "move": sess.moves,
            "action": action,
            "observation": data.get("observation", ""),
            "score": data.get("score", 0),
            "peak_score": data.get("peak_score", 0),
            "max_score": data.get("max_score", 0),
            "done": data.get("done", False),
        })

        self._flush_game_step(sess)

        if sess.done:
            threading.Thread(
                target=self._auto_close_session,
                args=(session_id,),
                daemon=True,
            ).start()

        return data

    def _auto_close_session(self, session_id: str) -> None:
        with self._game_lock:
            sess = self.game_sessions.pop(session_id, None)
        if sess is None:
            return
        self._archive_game_session(sess)
        try:
            backend = self._get_backend(sess.run_id)
            backend.cleanup_container(sess.container, logger)
        except Exception:
            logger.exception("Error auto-closing game session %s", session_id)
        logger.info("Auto-closed done game session %s (max_score=%d)", session_id, sess.max_score)

    def game_status(self, session_id: str) -> dict:
        with self._game_lock:
            sess = self.game_sessions.get(session_id)
        if sess is None:
            raise KeyError(f"Game session not found: {session_id}")

        resp = requests.get(f"{sess.container_url}/status", timeout=5)
        resp.raise_for_status()
        data = resp.json()

        sess.current_score = data.get("score", sess.current_score)
        sess.max_score = data.get("max_score", sess.max_score)
        sess.peak_score = data.get("peak_score", sess.peak_score)
        sess.moves = data.get("moves", sess.moves)
        sess.done = data.get("done", sess.done)
        sess.last_active = time.time()

        return data

    def close_game_session(self, session_id: str) -> dict:
        with self._game_lock:
            sess = self.game_sessions.pop(session_id, None)
        if sess is None:
            raise KeyError(f"Game session not found: {session_id}")

        try:
            resp = requests.post(f"{sess.container_url}/close", timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            data = {
                "final_score": sess.current_score,
                "peak_score": sess.peak_score,
                "max_score": sess.max_score,
                "moves": sess.moves,
            }
        finally:
            self._archive_game_session(sess)
            try:
                backend = self._get_backend(sess.run_id)
                backend.cleanup_container(sess.container, logger)
            except Exception:
                logger.exception("Error cleaning up game container for session %s", session_id)

        return data

    def close_all_game_sessions(self, run_id: str, task_id: str) -> int:
        with self._game_lock:
            sessions = [
                (sid, sess) for sid, sess in self.game_sessions.items()
                if sess.run_id == run_id and sess.task_id == task_id
            ]
        # Phase 1: archive all sessions synchronously (fast — JSON write + dict
        # append). This is what later /history calls will read, so it must
        # complete before we return.
        containers_to_cleanup = []
        for sid, sess in sessions:
            with self._game_lock:
                self.game_sessions.pop(sid, None)
            self._archive_game_session(sess)
            containers_to_cleanup.append((sid, sess.container))
        # Phase 2: docker cleanup is slow (stop+rm per container). Run it in
        # a background thread so we can return to the caller immediately.
        backend = self._get_backend(run_id)
        def _bg_cleanup():
            for sid, container in containers_to_cleanup:
                try:
                    backend.cleanup_container(container, logger)
                except Exception:
                    logger.exception("Error cleaning up game container for session %s", sid)
        threading.Thread(target=_bg_cleanup, daemon=True).start()
        return len(sessions)



# --- FastAPI app factory ---


async def _read_archive_request(request: Request) -> bytes:
    """Read one raw gzip request body without exceeding the archive limit."""

    media_type = request.headers.get("content-type", "").split(";", 1)[0]
    if media_type.strip().lower() != SUBMISSION_ARCHIVE_MEDIA_TYPE:
        raise HTTPException(
            status_code=415,
            detail=f"Content-Type must be {SUBMISSION_ARCHIVE_MEDIA_TYPE}",
        )

    raw_length = request.headers.get("content-length")
    if raw_length is not None:
        try:
            content_length = int(raw_length, 10)
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid Content-Length")
        if content_length < 0:
            raise HTTPException(status_code=400, detail="invalid Content-Length")
        if content_length > SUBMISSION_ARCHIVE_MAX_BYTES:
            raise HTTPException(
                status_code=413,
                detail="submission archive is too large",
            )

    async def collect() -> bytes:
        archive = bytearray()
        async for chunk in request.stream():
            if len(chunk) > SUBMISSION_ARCHIVE_MAX_BYTES - len(archive):
                raise HTTPException(
                    status_code=413,
                    detail="submission archive is too large",
                )
            archive.extend(chunk)
        return bytes(archive)

    try:
        return await asyncio.wait_for(
            collect(),
            timeout=ARCHIVE_UPLOAD_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="archive upload timed out")


async def _read_version_reference_request(request: Request) -> tuple[str, str]:
    """Read a small, closed-schema reference after token admission."""
    maximum = 4096
    raw_length = request.headers.get("content-length")
    if raw_length is not None:
        try:
            length = int(raw_length, 10)
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid Content-Length")
        if length < 0:
            raise HTTPException(status_code=400, detail="invalid Content-Length")
        if length > maximum:
            raise HTTPException(status_code=413, detail="version reference is too large")

    async def collect() -> bytes:
        payload = bytearray()
        async for chunk in request.stream():
            if len(chunk) > maximum - len(payload):
                raise HTTPException(status_code=413, detail="version reference is too large")
            payload.extend(chunk)
        return bytes(payload)

    try:
        payload = await asyncio.wait_for(collect(), timeout=ARCHIVE_UPLOAD_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="version reference upload timed out")
    try:
        value = json.loads(payload)
    except (ValueError, UnicodeError):
        raise HTTPException(status_code=400, detail="invalid version reference JSON")
    if (
        not isinstance(value, dict)
        or set(value) != {"version_id", "archive_sha256"}
        or not isinstance(value["version_id"], str)
        or not isinstance(value["archive_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", value["archive_sha256"]) is None
    ):
        raise HTTPException(
            status_code=400,
            detail="version reference requires only version_id and archive_sha256",
        )
    return value["version_id"], value["archive_sha256"]


def create_app(
    config: SForgeConfig | None = None,
    *,
    submission_evaluator=None,
    version_persister: Callable[[StoredVersion], None] | None = None,
) -> FastAPI:
    if config is None:
        config = load_config()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            # Remove the integrity-review network and egress sidecar.
            state.close_integrity_review()

    app = FastAPI(title="SForge Judge", version="1.0.0", lifespan=lifespan)
    state = JudgeState(config, submission_evaluator=submission_evaluator)
    state.load_tasks()
    app.state.judge_state = state

    async def persist_before_accepting(saved: StoredVersion | None) -> None:
        """Finish optional external persistence before queueing or acknowledging.

        This runs inside archive admission, before submission background work is
        scheduled. A failure preserves the local version and follows the same
        503/slot-release path as a local store write failure. Allocated round IDs
        and cooldowns keep their existing semantics; they are not rolled back.
        """
        if version_persister is None:
            return
        try:
            if saved is None:
                raise VersionStoreWriteError("External persistence requires a version store")
            await asyncio.to_thread(version_persister, saved)
        except Exception as exc:
            # Transport errors can contain signed URLs or credentials. Retain
            # the local version identity, but expose only the exception type.
            logger.error(
                "External version persistence failed: version_id=%s round=%s source=%s error_type=%s",
                saved.version_id if saved is not None else None,
                saved.round_id if saved is not None else None,
                saved.source if saved is not None else None,
                type(exc).__name__,
            )
            raise VersionStoreWriteError("external version persistence failed") from None

    @app.get("/api/v1/result/{submission_id}")
    def get_result(submission_id: str) -> ResultResponse:
        """Get result of an async submission."""
        result = state.get_result(submission_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Submission not found")
        return ResultResponse(
            submission_id=submission_id,
            status=result["status"],
            report=result["report"],
            error=result["error"],
            version_id=result.get("version_id"),
            candidate_id=result.get("candidate_id"),
            content_sha256=result.get("content_sha256"),
        )

    @app.post("/api/v1/register")
    def register(req: RegisterRequest) -> RegisterResponse:
        """Register a session and get a token for submissions."""
        if req.admin_secret != ADMIN_SECRET:
            raise HTTPException(status_code=403, detail="Invalid admin secret")
        version_saving_enabled = (
            state.version_store is not None
            and state.version_store.supports_task(req.task_id)
        )
        version_store_id = (
            state.version_store.store_id
            if version_saving_enabled and state.version_store is not None
            else None
        )
        if req.require_version_saving and (
            not version_saving_enabled
            or req.required_version_store_id != version_store_id
        ):
            raise HTTPException(
                status_code=409,
                detail="required version store is unavailable",
            )
        try:
            token = state.register_session(
                req.task_id, req.run_id,
                judge_cpu_limit=req.judge_cpu_limit,
                judge_mem_limit=req.judge_mem_limit,
                backend=req.backend,
                k8s_image_registry=req.k8s_image_registry,
                k8s_namespace=req.k8s_namespace,
                k8s_node_selector=req.k8s_node_selector,
                k8s_kubeconfig=req.k8s_kubeconfig,
                max_agent_submissions=req.max_agent_submissions,
                submission_cooldown=req.submission_cooldown,
                single_gpu_handoff=req.single_gpu_handoff,
                trusted_lease_id=req.trusted_lease_id,
                allocation_key=req.allocation_key,
                accelerator_device_ids=req.accelerator_device_ids,
                evaluation_contract_sha256=req.evaluation_contract_sha256,
            )
            checkpoint_token = (
                state.register_checkpoint_token(token)
                if req.issue_checkpoint_token
                else None
            )
            return RegisterResponse(
                token=token,
                checkpoint_token=checkpoint_token,
                version_saving_enabled=version_saving_enabled,
                version_store_id=version_store_id,
                single_gpu_handoff=req.single_gpu_handoff,
                integrity_review=state.integrity_review_notice(req.task_id),
            )
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.post(
        "/api/v1/handoff/attach-work",
        response_model=HandoffControlResponse,
    )
    def attach_work(req: AttachWorkRequest) -> HandoffControlResponse:
        """Admin-only exact-ID bind after Work is created and labelled."""

        if req.admin_secret != ADMIN_SECRET:
            raise HTTPException(status_code=403, detail="Invalid admin secret")
        try:
            session = state.attach_work_container(
                req.token, req.work_container_id
            )
        except KeyError:
            raise HTTPException(status_code=401, detail="Invalid token")
        except (ResourceHandoffError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        snapshot = session.lease.snapshot()
        return HandoffControlResponse(
            state=snapshot.state.value,
            epoch=snapshot.epoch,
            recovery_policy=session.lease.recovery_policy,
        )

    @app.post(
        "/api/v1/handoff/seal",
        response_model=HandoffControlResponse,
    )
    def seal_handoff(req: HandoffControlRequest) -> HandoffControlResponse:
        """Persist never-resume and close admission before final/deadline."""

        if req.admin_secret != ADMIN_SECRET:
            raise HTTPException(status_code=403, detail="Invalid admin secret")
        try:
            session = state.seal_handoff(req.token)
        except KeyError:
            raise HTTPException(status_code=401, detail="Invalid token")
        except (ResourceHandoffError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        snapshot = session.lease.snapshot()
        return HandoffControlResponse(
            state=snapshot.state.value,
            epoch=snapshot.epoch,
            recovery_policy=session.lease.recovery_policy,
        )

    @app.post(
        "/api/v1/handoff/final-work-archive",
        response_model=HandoffControlResponse,
    )
    def final_work_archive(
        req: FinalWorkArchiveRequest,
    ) -> HandoffControlResponse:
        """Retire frozen Work after the trusted run persisted its archive."""

        if req.admin_secret != ADMIN_SECRET:
            raise HTTPException(status_code=403, detail="Invalid admin secret")
        try:
            session = state.finish_final_work(
                req.token,
                archive_sha256=req.archive_sha256,
                archive_size=req.archive_size,
            )
        except (ResourceHandoffError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        snapshot = session.lease.snapshot()
        return HandoffControlResponse(
            state=snapshot.state.value,
            epoch=snapshot.epoch,
            recovery_policy=session.lease.recovery_policy,
        )

    @app.post(
        "/api/v1/handoff/final-judge",
        response_model=FinalJudgeResponse,
    )
    def final_judge(req: FinalJudgeRequest) -> FinalJudgeResponse:
        """Run the final Judge from the server-authenticated Work archive."""

        if req.admin_secret != ADMIN_SECRET:
            raise HTTPException(status_code=403, detail="Invalid admin secret")
        try:
            session, report = state.run_final_judge(
                source_run_id=req.source_run_id,
                final_run_id=req.final_run_id,
                task_id=req.task_id,
                trusted_lease_id=req.trusted_lease_id,
                allocation_key=req.allocation_key,
                evaluation_contract_sha256=(
                    req.evaluation_contract_sha256
                ),
                archive_sha256=req.archive_sha256,
                archive_size=req.archive_size,
                timeout=req.timeout,
                round_id=req.round_id,
            )
        except (ResourceHandoffError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        snapshot = session.lease.snapshot()
        return FinalJudgeResponse(
            state=snapshot.state.value,
            epoch=snapshot.epoch,
            recovery_policy=session.lease.recovery_policy,
            report=report,
        )

    @app.post(
        "/api/v1/versions",
        response_model=SaveVersionResponse,
        status_code=201,
    )
    async def save_version(
        request: Request,
        token: str = Header(..., alias=SESSION_TOKEN_HEADER),
        label: str = Header(..., alias=VERSION_LABEL_HEADER),
    ) -> SaveVersionResponse:
        """Persist a named checkpoint without launching evaluation."""

        try:
            validated_label = validate_version_label(label)
            with state.checkpoint_upload_admission(token) as (info, task):
                archive_data = await _read_archive_request(request)
                saved = await asyncio.to_thread(
                    state._save_checkpoint_admitted,
                    token,
                    info,
                    task,
                    archive_data,
                    validated_label,
                )
                await persist_before_accepting(saved)
        except KeyError:
            raise HTTPException(status_code=401, detail="Invalid token")
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except CheckpointBusy as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except CheckpointRateLimited as exc:
            raise HTTPException(status_code=429, detail=str(exc))
        except VersionStoreQuotaExceeded as exc:
            raise HTTPException(status_code=429, detail=str(exc))
        except VersionStoreCollision as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except VersionStoreWriteError:
            logger.exception("Checkpoint version store write failed")
            raise HTTPException(
                status_code=503,
                detail="version store write failed",
            )
        except VersionStoreCandidateError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except VersionStoreError:
            logger.exception("Checkpoint version store integrity failure")
            raise HTTPException(
                status_code=503,
                detail="version store integrity failure",
            )
        assert saved.candidate_id is not None
        assert saved.content_sha256 is not None
        return SaveVersionResponse(
            version_id=saved.version_id,
            round_id=saved.round_id or "checkpoint",
            archive_sha256=saved.archive_sha256,
            archive_size=saved.archive_size,
            candidate_id=saved.candidate_id,
            content_sha256=saved.content_sha256,
        )

    @app.get("/api/v1/versions/{version_id}")
    def get_saved_version(
        version_id: str,
        token: str = Header(..., alias=SESSION_TOKEN_HEADER),
    ) -> dict[str, Any]:
        """Return a reference for this evaluation token's own saved version."""
        try:
            state.resolve_evaluation_token(token)
            saved = state.resolve_saved_version(token, version_id)
        except KeyError:
            raise HTTPException(status_code=401, detail="Invalid token")
        except TokenCapabilityDenied as exc:
            raise HTTPException(status_code=403, detail=str(exc))
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except VersionStoreCandidateError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except VersionStoreError:
            logger.exception("Saved version reference integrity failure")
            raise HTTPException(status_code=503, detail="version store integrity failure")
        return {
            "version_id": saved.version_id,
            "archive_sha256": saved.archive_sha256,
            "archive_size": saved.archive_size,
            "candidate_id": saved.candidate_id,
            "content_sha256": saved.content_sha256,
        }

    @app.post("/api/v1/submit")
    async def submit(
        request: Request,
        background_tasks: BackgroundTasks,
        token: str = Header(..., alias=SESSION_TOKEN_HEADER),
        kind: str = Header("agent", alias=SUBMISSION_KIND_HEADER),
        admin_secret: str = Header("", alias=ADMIN_SECRET_HEADER),
    ) -> SubmitResponse:
        """Submit raw gzip bytes or a JSON reference to an existing version.

        The server resolves task_id, run_id, and assigns a round_id.
        kind=auto requires a valid admin_secret (host-side only).
        """
        if kind not in ("agent", "auto", "public"):
            raise HTTPException(
                status_code=400,
                detail="kind must be 'agent', 'auto', or 'public'",
            )
        if kind == "auto" and admin_secret != ADMIN_SECRET:
            raise HTTPException(status_code=403, detail="admin_secret required for auto submissions")

        try:
            with state.submission_upload_admission(token, kind) as token_info:
                if (
                    kind == "public"
                    and not state.tasks[token_info["task_id"]].judge.public_eval_cmd
                ):
                    raise HTTPException(
                        status_code=404,
                        detail="public evaluation is not enabled for this task",
                    )
                referenced_version = None
                media_type = request.headers.get("content-type", "").split(";", 1)[0]
                if media_type.strip().lower() == "application/json":
                    version_id, expected_sha256 = await _read_version_reference_request(request)
                    referenced_version = await asyncio.to_thread(
                        state.resolve_saved_version, token, version_id,
                    )
                    if referenced_version.archive_sha256 != expected_sha256:
                        raise VersionStoreCandidateError("saved version archive SHA-256 mismatch")
                    assert state.version_store is not None
                    archive_data = await asyncio.to_thread(
                        state.version_store.read_archive, referenced_version,
                    )
                else:
                    archive_data = await _read_archive_request(request)
                if (
                    state.version_store is not None
                    and state.version_store.supports_task(token_info["task_id"])
                ):
                    await asyncio.to_thread(
                        state.validate_version_archive,
                        token,
                        archive_data,
                    )
                with state.submission_commit_admission(token, kind):
                    (
                        task_id,
                        run_id,
                        round_id,
                        judge_cpu_limit,
                        judge_mem_limit,
                        remaining,
                    ) = state.consume_round(token, kind)
                    try:
                        saved_version = referenced_version
                        if saved_version is None:
                            saved_version = await asyncio.to_thread(
                                state.persist_submission_version,
                                task_id=task_id,
                                run_id=run_id,
                                source=kind,
                                round_id=round_id,
                                archive=archive_data,
                            )
                        await persist_before_accepting(saved_version)
                        submission_id = state.submit(
                            task_id,
                            archive_data,
                            run_id,
                            round_id,
                            judge_cpu_limit=judge_cpu_limit,
                            judge_mem_limit=judge_mem_limit,
                            session_token=token,
                            stored_version=saved_version,
                            background_tasks=background_tasks,
                        )
                        return SubmitResponse(
                            submission_id=submission_id,
                            round_id=round_id,
                            status=SubmissionStatus.QUEUED,
                            remaining_submissions=remaining,
                            version_id=(
                                saved_version.version_id
                                if saved_version is not None
                                else None
                            ),
                            candidate_id=(
                                saved_version.candidate_id
                                if saved_version is not None
                                else None
                            ),
                            content_sha256=(
                                saved_version.content_sha256
                                if saved_version is not None
                                else None
                            ),
                            archive_sha256=(
                                saved_version.archive_sha256
                                if saved_version is not None
                                else None
                            ),
                        )
                    except VersionStoreCollision as exc:
                        state._release_submission_round(token, round_id)
                        raise HTTPException(status_code=409, detail=str(exc))
                    except VersionStoreWriteError:
                        state._release_submission_round(token, round_id)
                        logger.exception("Judge submission version store write failed")
                        raise HTTPException(
                            status_code=503,
                            detail="version store write failed",
                        )
                    except VersionStoreCandidateError as exc:
                        state._release_submission_round(token, round_id)
                        raise HTTPException(status_code=400, detail=str(exc))
                    except VersionStoreError:
                        state._release_submission_round(token, round_id)
                        logger.exception("Judge version store integrity failure")
                        raise HTTPException(
                            status_code=503,
                            detail="version store integrity failure",
                        )
                    except ValueError as exc:
                        state._release_submission_round(token, round_id)
                        raise HTTPException(status_code=404, detail=str(exc))
                    except Exception:
                        state._release_submission_round(token, round_id)
                        raise
        except KeyError:
            raise HTTPException(status_code=401, detail="Invalid token")
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except TokenCapabilityDenied as exc:
            raise HTTPException(status_code=403, detail=str(exc))
        except SubmissionBudgetExceeded as e:
            raise HTTPException(status_code=429, detail=str(e))
        except SubmissionCooldownActive as e:
            raise HTTPException(status_code=429, detail=str(e))
        except AgentSubmissionBusy as e:
            raise HTTPException(status_code=409, detail=str(e))
        except PublicEvaluationBusy as e:
            raise HTTPException(status_code=409, detail=str(e))
        except VersionStoreCandidateError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except VersionStoreError:
            logger.exception("Judge version store integrity failure")
            raise HTTPException(
                status_code=503,
                detail="version store integrity failure",
            )
        except ArchiveUploadBusy as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @app.get("/api/v1/history")
    def history(
        token: str = Query(...),
        admin_secret: str = Query("", deprecated=True),
        admin_secret_header: str | None = Header(
            default=None,
            alias=ADMIN_SECRET_HEADER,
        ),
    ) -> dict:
        """Get run history using a session token.

        Without an admin secret: returns only agent submissions (agent-visible
        view). With a valid admin header, or the deprecated query parameter,
        returns all entries including auto-eval (host-side view).
        """
        effective_admin_secret = (
            admin_secret_header
            if admin_secret_header is not None
            else admin_secret
        )
        try:
            if effective_admin_secret == ADMIN_SECRET:
                # The host-side audit view must stay readable after the final
                # seal closed evaluation admission: run_agent drains pending
                # Judge work through this endpoint before freezing Work.
                info = state.resolve_token(token)
            else:
                info = state.resolve_evaluation_token(token)
        except KeyError:
            raise HTTPException(status_code=401, detail="Invalid token")
        except TokenCapabilityDenied as exc:
            raise HTTPException(status_code=403, detail=str(exc))
        full = state.get_run_history(info["run_id"], info["task_id"])
        if effective_admin_secret == ADMIN_SECRET:
            return full
        # Filter to hidden Agent submissions only. Public diagnostics have a
        # separate history domain and must never affect hidden best selection.
        agent_entries = [
            {k: v for k, v in e.items() if k != "integrity_review"}
            for e in full.get("entries", [])
            if e.get("type") == "submission"
            and not (e.get("round") or "").startswith("auto-")
        ]
        resolved_tid = info["task_id"]
        task_spec = state.tasks.get(resolved_tid) if resolved_tid else None
        policy = task_spec.judge.selection if task_spec else "pass_rate_first"
        direction = task_spec.judge.score_direction if task_spec else "maximize"
        best = select_best(agent_entries, direction, policy)
        return {
            "run_id": info["run_id"],
            "best_score": best["best_score"],
            "best_pass_rate": best["best_pass_rate"],
            "best_round": best["best_round"],
            "agent_submissions": sum(1 for e in agent_entries if e.get("type") == "submission"),
            "auto_submissions": 0,
            "entries": agent_entries,
        }

    @app.get("/api/v1/public/history")
    def public_history(token: str = Query(...)) -> dict:
        """Return only public diagnostic runs for the authenticated session."""
        try:
            info = state.resolve_evaluation_token(token)
        except KeyError:
            raise HTTPException(status_code=401, detail="Invalid token")
        except TokenCapabilityDenied as exc:
            raise HTTPException(status_code=403, detail=str(exc))
        task_spec = state.tasks.get(info["task_id"])
        if task_spec is None or not task_spec.judge.public_eval_cmd:
            raise HTTPException(
                status_code=404,
                detail="public evaluation is not enabled for this task",
            )
        full = state.get_run_history(info["run_id"], info["task_id"])
        entries = [
            {k: v for k, v in entry.items() if k != "integrity_review"}
            for entry in full.get("entries", [])
            if entry.get("type") == "public"
        ]
        return {
            "run_id": info["run_id"],
            "enabled": True,
            "public_submissions": len(entries),
            "entries": entries,
        }

    # --- Game session routes ---

    @app.post("/api/v1/game/{run_id}/{task_id}/new")
    def game_new(run_id: str, task_id: str, req: GameNewRequest) -> GameNewResponse:
        """Start a new game session in a dedicated container."""
        try:
            sess, data = state.create_game_session(run_id, task_id)
        except ValueError as e:
            raise HTTPException(404, str(e))
        except (RuntimeError, BuildImageError) as e:
            raise HTTPException(503, str(e))
        return GameNewResponse(
            session_id=sess.session_id,
            observation=data.get("observation", ""),
            score=data.get("score", 0),
            peak_score=data.get("peak_score", 0),
            max_score=data.get("max_score", 0),
            done=data.get("done", False),
            moves=data.get("moves", 0),
        )

    @app.post("/api/v1/game/{run_id}/{task_id}/{session_id}/step")
    def game_step(run_id: str, task_id: str, session_id: str, req: GameStepRequest) -> GameStepResponse:
        try:
            data = state.game_step(session_id, req.action)
        except KeyError:
            raise HTTPException(404, f"Game session not found: {session_id}")
        except requests.HTTPError as e:
            raise HTTPException(400, str(e))
        return GameStepResponse(
            session_id=session_id,
            observation=data.get("observation", ""),
            score=data.get("score", 0),
            peak_score=data.get("peak_score", 0),
            max_score=data.get("max_score", 0),
            done=data.get("done", False),
            moves=data.get("moves", 0),
        )

    @app.get("/api/v1/game/{run_id}/{task_id}/{session_id}/status")
    def game_status(run_id: str, task_id: str, session_id: str) -> GameStatusResponse:
        try:
            data = state.game_status(session_id)
        except KeyError:
            raise HTTPException(404, f"Game session not found: {session_id}")
        return GameStatusResponse(
            session_id=session_id,
            score=data.get("score", 0),
            peak_score=data.get("peak_score", 0),
            max_score=data.get("max_score", 0),
            done=data.get("done", False),
            moves=data.get("moves", 0),
        )

    @app.post("/api/v1/game/{run_id}/{task_id}/{session_id}/close")
    def game_close(run_id: str, task_id: str, session_id: str) -> GameCloseResponse:
        try:
            data = state.close_game_session(session_id)
        except KeyError:
            raise HTTPException(404, f"Game session not found: {session_id}")
        return GameCloseResponse(
            session_id=session_id,
            final_score=data.get("final_score", 0),
            peak_score=data.get("peak_score", 0),
            max_score=data.get("max_score", 0),
            moves=data.get("moves", 0),
        )

    @app.post("/api/v1/game/{run_id}/{task_id}/close-all")
    def game_close_all(run_id: str, task_id: str) -> dict:
        count = state.close_all_game_sessions(run_id, task_id)
        return {"closed": count}

    return app
