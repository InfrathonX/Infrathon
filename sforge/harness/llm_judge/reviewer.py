"""One integrity review: bundle -> prompt -> verdict votes -> decision.

Every review leaves an audit trail in ``<submission log dir>/integrity_review``
(``bundle.json`` manifest, ``prompt.md``, raw ``response-*.json`` replies and
the final ``verdict.json``).  None of it is agent-visible.  Definitive
verdicts are cached by content, so resubmitting identical files reuses the
earlier decision instead of asking the model again.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from sforge.harness.grading import EvalDetail, EvalReport
from sforge.harness.llm_judge.bundle import (
    BaselineProvider,
    BundleError,
    NoBaseline,
    ReviewBundle,
    build_bundle,
)
from sforge.harness.llm_judge.client import ChatCompletionsClient, ChatResult
from sforge.harness.llm_judge.config import LLMJudgeConfig, LLMJudgeConfigError
from sforge.harness.llm_judge.policy import (
    PolicyError,
    TaskIntegrityPolicy,
    load_task_policy,
    policy_problems,
)
from sforge.harness.llm_judge.prompt import (
    FEEDBACK_MAX_CHARS,
    PROMPT_VERSION,
    ParsedVerdict,
    RenderedPrompt,
    VerdictError,
    parse_verdict,
    render_prompt,
    requote_message,
    verify_evidence,
)
from sforge.harness.task_spec import TaskSpec

logger = logging.getLogger("sforge.llm_judge")

REVIEW_DIRECTORY = "integrity_review"
EVALUATION_STATUS_REJECTED = "integrity_rejected"

STATUS_PASS = "pass"
STATUS_VIOLATION = "violation"
STATUS_FLAGGED = "flagged"
STATUS_ERROR = "error"
STATUS_SKIPPED = "skipped"
# Only settled outcomes are cached: a flagged (uncertain) verdict is asked
# again when the same content comes back.
_CACHEABLE = {STATUS_PASS, STATUS_VIOLATION}
# Bump when evidence verification or the decision rule changes meaning; it is
# part of every cache key, like PROMPT_VERSION.
DECISION_VERSION = 3


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


@dataclass
class IntegrityReview:
    review_id: str
    task_id: str
    mode: str
    status: str
    blocking: bool
    reason: str
    feedback_to_agent: str | None = None
    rules: list[str] = field(default_factory=list)
    bundle_digest: str | None = None
    cache_key: str | None = None
    cache_hit: bool = False
    model: str | None = None
    prompt_version: str = PROMPT_VERSION
    policy_source: str | None = None
    elapsed_seconds: float = 0.0
    votes: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    @property
    def rejected(self) -> bool:
        return self.blocking and self.status == STATUS_VIOLATION

    def agent_metrics(self) -> dict[str, Any]:
        """The part of a rejection the agent may see."""

        return {
            "verdict": "VIOLATION",
            "rules": self.rules,
            "feedback": self.feedback_to_agent or "",
            "review_id": self.review_id,
        }

    def history_summary(self) -> dict[str, Any]:
        return {
            "review_id": self.review_id,
            "mode": self.mode,
            "status": self.status,
            "blocking": self.blocking,
            "cache_hit": self.cache_hit,
            "rules": self.rules,
        }

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "review_id": self.review_id,
            "task_id": self.task_id,
            "mode": self.mode,
            "status": self.status,
            "blocking": self.blocking,
            "reason": self.reason,
            "feedback_to_agent": self.feedback_to_agent,
            "rules": self.rules,
            "bundle_digest": self.bundle_digest,
            "cache_key": self.cache_key,
            "cache_hit": self.cache_hit,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "policy_source": self.policy_source,
            "elapsed_seconds": self.elapsed_seconds,
            "votes": self.votes,
            "error": self.error,
        }


def _collect_secrets(config: LLMJudgeConfig) -> list[str]:
    """Credential strings that must never reach agent-visible text."""

    values: list[str] = []
    try:
        key = config.read_api_key()
    except LLMJudgeConfigError:
        key = None
    if key:
        values.append(key)
    if config.codex_auth_json is not None:
        try:
            auth = json.loads(config.codex_auth_json.expanduser().read_text())
        except (OSError, ValueError):
            auth = None

        def walk(value: Any) -> None:
            if isinstance(value, dict):
                for child in value.values():
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)
            elif isinstance(value, str) and len(value) >= 16:
                values.append(value)

        walk(auth)
    return values


def redact(text: str | None, secrets: Iterable[str]) -> str | None:
    if not text:
        return text
    for secret in secrets:
        if len(secret) >= 8 and secret in text:
            text = text.replace(secret, "[redacted]")
    return text


def redact_value(value: Any, secrets: list[str]) -> Any:
    """Redact credentials from every string inside a JSON-like value."""

    if isinstance(value, str):
        return redact(value, secrets)
    if isinstance(value, dict):
        return {key: redact_value(item, secrets) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_value(item, secrets) for item in value]
    return value


def _write_private_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(text)


class ApiReviewSession:
    """One verdict conversation over the chat-completions API."""

    def __init__(self, client: ChatCompletionsClient, prompt: RenderedPrompt) -> None:
        self.client = client
        self.messages = prompt.messages()
        self.last: ChatResult | None = None

    def ask(self, timeout: float) -> ChatResult:
        self.last = self.client.complete(self.messages, timeout=timeout)
        return self.last

    def repair(self, message: str, timeout: float) -> ChatResult:
        previous = self.last.content if self.last is not None else ""
        self.messages = self.messages + [
            {"role": "assistant", "content": previous[:20000]},
            {"role": "user", "content": message},
        ]
        return self.ask(timeout)

    def close(self) -> None:
        pass


class ApiReviewEngine:
    def __init__(self, client: ChatCompletionsClient) -> None:
        self.client = client

    def open(self, *, overview: RenderedPrompt, **_: Any) -> ApiReviewSession:
        return ApiReviewSession(self.client, overview)


def _synthesized_feedback(verdict: ParsedVerdict) -> str:
    parts = []
    for violation in verdict.violations[:3]:
        where = ", ".join(violation.get("files") or []) or "the submission"
        text = violation.get("explanation") or violation.get("rule") or ""
        parts.append(f"{where}: {text}".strip())
    return " ".join(parts)[:FEEDBACK_MAX_CHARS]


class IntegrityReviewer:
    """Thread-safe reviewer shared by every submission of one Judge server."""

    def __init__(
        self,
        config: LLMJudgeConfig,
        *,
        tasks_dir: Path | None,
        cache_dir: Path | None,
        engine: Any = None,
        client: ChatCompletionsClient | None = None,
    ) -> None:
        """``engine`` opens verdict sessions (``AgentReviewEngine`` for the
        agent engine, built by the caller because it needs Docker);
        the api engine is built here from ``client`` or the config."""

        self.config = config
        self.tasks_dir = tasks_dir
        self.cache_dir = cache_dir if config.cache else None
        if engine is None:
            if config.engine != "api":
                raise LLMJudgeConfigError(
                    "the agent review engine must be constructed by the caller"
                )
            engine = ApiReviewEngine(client or ChatCompletionsClient(config))
        self.engine = engine
        self._policies: dict[str, TaskIntegrityPolicy] = {}
        self._lock = threading.Lock()

    # --- task policy ---

    def policy_for(self, task_spec: TaskSpec) -> TaskIntegrityPolicy:
        with self._lock:
            cached = self._policies.get(task_spec.task_id)
        if cached is not None:
            return cached
        policy = load_task_policy(
            task_spec.task_id,
            tasks_dir=self.tasks_dir,
            policy_dir=self.config.policy_dir,
        )
        with self._lock:
            self._policies[task_spec.task_id] = policy
        return policy

    def validate_tasks(self, tasks: Iterable[TaskSpec]) -> None:
        """Fail at server start, not mid-run, on a malformed policy."""

        problems: list[str] = []
        for task_spec in tasks:
            try:
                policy = self.policy_for(task_spec)
            except PolicyError as exc:
                problems.append(f"{task_spec.task_id}: {exc}")
                continue
            problems.extend(
                f"{task_spec.task_id}: {problem}" for problem in policy_problems(policy)
            )
        if problems:
            raise PolicyError(
                "the mandatory integrity review cannot gate these tasks:\n  "
                + "\n  ".join(problems)
            )

    def covers(self, task_spec: TaskSpec, kind: str) -> bool:
        return self.config.covers(kind)

    def announced_to_agent(self, task_spec: TaskSpec) -> bool:
        return (
            self.config.enforcing
            and self.config.announce
            and self.covers(task_spec, "agent")
        )

    # --- cache ---

    def _cache_key(
        self,
        task_spec: TaskSpec,
        policy: TaskIntegrityPolicy,
        bundle: ReviewBundle,
        engine_identity: dict[str, Any],
    ) -> str:
        payload = {
            "prompt_version": PROMPT_VERSION,
            "decision_version": DECISION_VERSION,
            "engine": engine_identity,
            "config": self.config.identity(),
            "policy": policy.sha256,
            "bundle": bundle.digest(),
            "agent_query": hashlib.sha256(
                task_spec.work.agent_query.encode("utf-8")
            ).hexdigest(),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def _cache_path(self, task_id: str, key: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / task_id / f"{key}.json"

    def _cache_load(self, task_id: str, key: str) -> dict[str, Any] | None:
        path = self._cache_path(task_id, key)
        if path is None or not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(value, dict) or value.get("status") not in _CACHEABLE:
            return None
        return value

    def _cache_store(self, review: IntegrityReview) -> None:
        if review.cache_key is None or review.status not in _CACHEABLE:
            return
        path = self._cache_path(review.task_id, review.cache_key)
        if path is None:
            return
        try:
            _write_json(
                path,
                {
                    "status": review.status,
                    "reason": review.reason,
                    "feedback_to_agent": review.feedback_to_agent,
                    "rules": review.rules,
                    "model": review.model,
                    "prompt_version": review.prompt_version,
                    "source_review_id": review.review_id,
                    "created_at": time.time(),
                },
            )
        except OSError:
            logger.warning("Could not write integrity review cache entry", exc_info=True)

    # --- review ---

    def _blocking(self, status: str) -> bool:
        if not self.config.enforcing:
            return False
        if status == STATUS_VIOLATION:
            return True
        return status == STATUS_ERROR and self.config.on_error == "block"

    def review(
        self,
        *,
        task_spec: TaskSpec,
        archive: bytes,
        baseline: BaselineProvider,
        log_dir: Path,
        candidate_version: int | None = None,
    ) -> IntegrityReview:
        started = time.monotonic()
        deadline = started + self.config.review_timeout_seconds
        out_dir = log_dir / REVIEW_DIRECTORY
        out_dir.mkdir(parents=True, exist_ok=True)
        review = IntegrityReview(
            review_id=uuid.uuid4().hex[:12],
            task_id=task_spec.task_id,
            mode=self.config.mode,
            status=STATUS_ERROR,
            blocking=False,
            reason="",
            model=self.config.model,
        )
        try:
            self._review(
                review,
                task_spec=task_spec,
                archive=archive,
                baseline=baseline,
                out_dir=out_dir,
                deadline=deadline,
                candidate_version=candidate_version,
            )
        except Exception as exc:  # the review must never crash a submission
            logger.exception("Integrity review %s failed", review.review_id)
            review.status = STATUS_ERROR
            review.reason = "the integrity review failed internally"
            review.error = f"{type(exc).__name__}: {exc}"[:500]
        review.blocking = self._blocking(review.status)
        review.elapsed_seconds = round(time.monotonic() - started, 3)
        try:
            _write_json(out_dir / "verdict.json", review.to_record())
        except OSError:
            logger.warning("Could not persist integrity verdict", exc_info=True)
        logger.info(
            "Integrity review %s for %s: %s (blocking=%s, cache_hit=%s, %.1fs)",
            review.review_id,
            task_spec.task_id,
            review.status,
            review.blocking,
            review.cache_hit,
            review.elapsed_seconds,
        )
        return review

    def _review(
        self,
        review: IntegrityReview,
        *,
        task_spec: TaskSpec,
        archive: bytes,
        baseline: BaselineProvider,
        out_dir: Path,
        deadline: float,
        candidate_version: int | None,
    ) -> None:
        try:
            policy = self.policy_for(task_spec)
        except PolicyError as exc:
            review.reason = "the task integrity policy is invalid"
            review.error = str(exc)[:500]
            return
        review.policy_source = policy.source
        try:
            snapshot = baseline.snapshot(task_spec, policy)
        except Exception as exc:
            logger.warning("Baseline unavailable for %s: %s", task_spec.task_id, exc)
            snapshot = NoBaseline(f"{type(exc).__name__}: {exc}"[:200]).snapshot(
                task_spec, policy
            )
        try:
            bundle = build_bundle(
                task_spec,
                archive,
                policy=policy,
                baseline=snapshot,
                candidate_version=candidate_version,
            )
        except BundleError as exc:
            review.status = STATUS_SKIPPED
            review.reason = (
                "the archive cannot be reviewed; the Judge's own admission "
                "decides it"
            )
            review.error = str(exc)[:500]
            return
        review.bundle_digest = bundle.digest()
        _write_json(out_dir / "bundle.json", bundle.manifest())
        # The resolved review image and agent runtime are part of what a
        # verdict depends on; a rebuilt image or new runtime asks again.
        identify = getattr(self.engine, "identity", None)
        engine_identity = identify(task_spec) if identify is not None else {}
        review.cache_key = self._cache_key(task_spec, policy, bundle, engine_identity)
        cached = self._cache_load(task_spec.task_id, review.cache_key)
        if cached is not None:
            review.status = cached["status"]
            review.reason = str(cached.get("reason") or "")
            review.feedback_to_agent = cached.get("feedback_to_agent")
            review.rules = list(cached.get("rules") or [])
            review.model = cached.get("model") or review.model
            review.cache_hit = True
            return
        prompt = render_prompt(
            bundle,
            task_spec,
            policy,
            boundary=secrets.token_hex(8),
            char_budget=self.config.context_char_budget,
            file_char_cap=self.config.file_char_cap,
        )
        _write_private_text(
            out_dir / "prompt.md",
            "# system\n\n" + prompt.system + "\n\n# user\n\n" + prompt.user,
        )
        hidden = _collect_secrets(self.config)
        context = {
            "task_spec": task_spec,
            "archive": archive,
            "bundle": bundle,
            "overview": prompt,
            "policy": policy,
            "out_dir": out_dir,
            # Shared and extended by sessions (refreshed tokens), so every
            # redaction below sees the credentials a review actually used.
            "secrets": hidden,
            "image_ref": engine_identity.get("image_id"),
        }
        votes: list[dict[str, Any]] = []
        parsed: list[ParsedVerdict] = []
        for index in range(self.config.votes):
            outcome, verdict = self._vote(context, index, deadline)
            votes.append(outcome)
            if verdict is not None:
                parsed.append(verdict)
        review.votes = redact_value(votes, hidden)
        self._decide(review, parsed, votes)
        review.feedback_to_agent = redact(review.feedback_to_agent, hidden)
        review.rules = [redact(rule, hidden) or "" for rule in review.rules]
        review.reason = redact(review.reason, hidden) or ""
        review.error = redact(review.error, hidden)
        self._cache_store(review)

    def _vote(
        self,
        context: dict[str, Any],
        index: int,
        deadline: float,
    ) -> tuple[dict[str, Any], ParsedVerdict | None]:
        """One independent verdict: a fresh session, repaired on bad output.

        Infrastructure failures (endpoint errors, a review container that
        could not start) close the session and retry with a new one; a reply
        that is not a valid verdict is sent back to the same session.
        """

        bundle: ReviewBundle = context["bundle"]
        out_dir: Path = context["out_dir"]
        attempts: list[dict[str, Any]] = []
        session = None
        repair: str | None = None
        try:
            for attempt in range(1, self.config.max_attempts + 1):
                remaining = deadline - time.monotonic()
                if remaining < 5:
                    attempts.append({"attempt": attempt, "error": "review deadline reached"})
                    break
                timeout = min(self.config.request_timeout_seconds, remaining)
                try:
                    if session is None:
                        repair = None
                        session = self.engine.open(
                            **context,
                            session_label=f"vote{index + 1}-try{attempt}",
                            deadline=deadline,
                        )
                    result = (
                        session.ask(timeout)
                        if repair is None
                        else session.repair(repair, timeout)
                    )
                except LLMJudgeConfigError as exc:
                    attempts.append({"attempt": attempt, "error": str(exc)[:500]})
                    break
                except Exception as exc:
                    retryable = getattr(exc, "retryable", True)
                    attempts.append(
                        {"attempt": attempt, "error": f"{type(exc).__name__}: {exc}"[:500]}
                    )
                    if session is not None:
                        session.close()
                        session = None
                    if not retryable:
                        break
                    pause = min(5.0 * 2 ** (attempt - 1), 60.0)
                    if deadline - time.monotonic() - pause > 5:
                        time.sleep(pause)
                    continue
                _write_json(
                    out_dir / f"response-{index + 1}-{attempt}.json",
                    redact_value(result.record(), context["secrets"]),
                )
                try:
                    verdict = parse_verdict(result.content)
                except VerdictError as exc:
                    attempts.append(
                        {
                            "attempt": attempt,
                            "error": f"unusable reply: {exc}",
                            "finish_reason": result.finish_reason,
                        }
                    )
                    repair = (
                        f"That reply is not a valid verdict ({exc}). Reply again "
                        "with only the JSON verdict object described in your "
                        "instructions."
                    )
                    continue
                verify_evidence(verdict, bundle)
                attempts.append(
                    {
                        "attempt": attempt,
                        "ok": True,
                        "usage": result.usage,
                        "latency_seconds": round(result.latency_seconds, 3),
                        "finish_reason": result.finish_reason,
                    }
                )
                if verdict.is_violation and not verdict.verified:
                    verdict = self._requote(
                        session, verdict, context, index, attempt, deadline, attempts
                    )
                return (
                    {
                        "vote": index + 1,
                        "ok": True,
                        "verdict": verdict.to_dict(),
                        "attempts": attempts,
                    },
                    verdict,
                )
        finally:
            if session is not None:
                session.close()
        return (
            {
                "vote": index + 1,
                "ok": False,
                "error": attempts[-1]["error"] if attempts else "no attempt made",
                "attempts": attempts,
            },
            None,
        )

    def _requote(
        self,
        session,
        verdict: ParsedVerdict,
        context: dict[str, Any],
        index: int,
        attempt: int,
        deadline: float,
        attempts: list[dict[str, Any]],
    ) -> ParsedVerdict:
        """Ask once for verbatim evidence when a violation's quote failed.

        A sloppy quote (paraphrased, too short, wrong file) must not decide
        either way: the same session re-checks the code and either re-quotes
        it or withdraws the claim.  On any failure the first verdict stands.
        """

        remaining = deadline - time.monotonic()
        if remaining < 30:
            return verdict
        record: dict[str, Any] = {"attempt": attempt, "requote": True}
        try:
            result = session.repair(
                requote_message(verdict),
                min(self.config.request_timeout_seconds, remaining),
            )
            _write_json(
                context["out_dir"] / f"response-{index + 1}-{attempt}-requote.json",
                redact_value(result.record(), context["secrets"]),
            )
            retried = parse_verdict(result.content)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"[:500]
            attempts.append(record)
            return verdict
        verify_evidence(retried, context["bundle"])
        record.update(
            {
                "ok": True,
                "verdict": retried.verdict,
                "verified": retried.verified,
                "usage": result.usage,
                "latency_seconds": round(result.latency_seconds, 3),
            }
        )
        attempts.append(record)
        return retried

    def _decide(
        self,
        review: IntegrityReview,
        verdicts: list[ParsedVerdict],
        votes: list[dict[str, Any]],
    ) -> None:
        if not verdicts:
            review.status = STATUS_ERROR
            review.reason = "no usable verdict was obtained"
            failed = [vote.get("error") for vote in votes if not vote.get("ok")]
            review.error = str(failed[-1])[:500] if failed else None
            return
        threshold = self.config.min_confidence
        confident = [
            verdict
            for verdict in verdicts
            if verdict.is_violation and verdict.confidence >= threshold and verdict.verified
        ]
        claimed = [verdict for verdict in verdicts if verdict.is_violation]
        configured = max(self.config.votes, len(verdicts))
        failed = configured - len(verdicts)
        if 2 * len(confident) <= configured < 2 * (len(confident) + failed):
            # The missing votes could still decide either way: a failed vote
            # must not turn one dissenting review into a majority, or hide one.
            review.status = STATUS_ERROR
            review.reason = (
                f"only {len(verdicts)} of {configured} reviews produced a verdict, "
                "too few to decide"
            )
            return
        if 2 * len(confident) > configured:
            best = max(confident, key=lambda verdict: verdict.confidence)
            review.status = STATUS_VIOLATION
            review.reason = (
                f"{len(confident)} of {configured} review(s) found a verified "
                "violation"
            )
            review.feedback_to_agent = (
                best.feedback_to_agent or _synthesized_feedback(best)
            )
            rules: list[str] = []
            for violation, check in zip(best.violations, best.evidence_checks):
                rule = violation.get("rule") or "task scope"
                if check.get("verified") and rule not in rules:
                    rules.append(rule)
            review.rules = rules[:5]
            return
        if claimed:
            unverified = sum(1 for verdict in claimed if not verdict.verified)
            low = sum(1 for verdict in claimed if verdict.confidence < threshold)
            review.status = STATUS_FLAGGED
            review.reason = (
                f"{len(claimed)} of {len(verdicts)} review(s) claimed a violation "
                f"below the rejection bar ({unverified} with unverified evidence, "
                f"{low} below confidence {threshold})"
            )
            return
        review.status = STATUS_PASS
        review.reason = f"{len(verdicts)} review(s) found the submission within scope"


# --- terminal reports for submissions the review ends ---


def _bind_contract(task_spec: TaskSpec, report: EvalReport) -> None:
    from sforge.harness.run_evaluation import _bind_portable_contract_report

    _bind_portable_contract_report(task_spec, report)


def rejected_report(
    task_spec: TaskSpec,
    submission_id: str,
    review: IntegrityReview,
    *,
    submitted_at: float,
) -> EvalReport:
    """An invalid, unscored report for a submission the review rejected."""

    rules = "; ".join(review.rules) if review.rules else "task scope"
    report = EvalReport(
        task_id=task_spec.task_id,
        submission_id=submission_id,
        total_tests=1,
        failed=1,
        valid=False,
        score=None,
        summary=(
            "Integrity review rejected this submission before benchmarking "
            f"({rules}). It was not run and has no score. "
            + (review.feedback_to_agent or "")
        ).strip(),
        details=[
            EvalDetail(
                name="integrity_review",
                status="FAILED",
                message=review.feedback_to_agent or rules,
            )
        ],
        metrics={
            "evaluation_status": EVALUATION_STATUS_REJECTED,
            "correctness_gate_passed": False,
            "integrity_review": review.agent_metrics(),
        },
        submitted_at=submitted_at,
    )
    _bind_contract(task_spec, report)
    return report


def unavailable_report(
    task_spec: TaskSpec,
    submission_id: str,
    review: IntegrityReview,
    *,
    submitted_at: float,
) -> EvalReport:
    """Infrastructure-invalid report when ``on_error=block`` stops a submission."""

    report = EvalReport(
        task_id=task_spec.task_id,
        submission_id=submission_id,
        errors=1,
        valid=False,
        score=0.0,
        summary=(
            "Evaluation infrastructure could not complete the integrity review; "
            "the submission was not benchmarked. Submit again later."
        ),
        metrics={
            "evaluation_status": "infrastructure_error",
            "correctness_gate_passed": False,
            "primary_score_value": 0.0,
        },
        submitted_at=submitted_at,
    )
    _bind_contract(task_spec, report)
    return report
