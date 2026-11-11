"""Configuration of the LLM integrity review that gates submissions.

The review is mandatory: a configuration loaded from the environment
enforces it unless ``SFORGE_LLM_JUDGE_MODE`` explicitly says ``shadow`` or
``off`` (debugging only).  Every value is read from ``SFORGE_LLM_JUDGE_*``
environment variables (normally exported by the launch config) by the process
that hosts the Judge server, so a task launcher needs no new flags; the agent
runtime and Codex credentials default to the run's own
(``SFORGE_AGENT_RUNTIME_DIR``, ``CODEX_AUTH_JSON_PATH``).  Secrets are never
held in this object: only the paths or variable names they live in, which are
read when a review starts and never logged.

Two engines produce a verdict:

``agent`` (default)
    A Worker-like coding agent (Codex or Claude Code from a host agent
    runtime) runs in a CPU-only review container built from the task's Judge
    (or Work) image, inspects the submission with ordinary tools and answers
    with a JSON verdict.
``api``
    One chat-completions request carrying the rendered review bundle; cheap,
    but it cannot explore anything outside what the bundle shows.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

ENV_PREFIX = "SFORGE_LLM_JUDGE_"
MODES = ("off", "shadow", "enforce")
ON_ERROR_POLICIES = ("admit", "block")
SUBMISSION_KINDS = ("agent", "auto", "public")
ENGINES = ("agent", "api")
REVIEW_AGENTS = ("codex", "claude-code")
REVIEW_IMAGES = ("judge", "work")
# "platform": the sandbox backend owns the network (e.g. PyroMind); only
# for review engines whose sandboxes the integrator provides.
REVIEW_NETWORKS = ("isolated", "bridge", "platform")
# What a configuration loaded from the environment uses when unset.  The
# dataclass itself defaults to ``off`` so in-process test configs stay inert.
DEFAULT_MODE = "enforce"
# No default reviewer model: an enabled review needs SFORGE_LLM_JUDGE_MODEL
# (LLMJudgeConfig.check_ready says so).
DEFAULT_MODEL: str | None = None
DEFAULT_REASONING_EFFORT = "high"


class LLMJudgeConfigError(ValueError):
    """The integrity-review configuration is unusable."""


def _parse_bool(name: str, raw: str) -> bool:
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise LLMJudgeConfigError(f"{name} must be a boolean (1/0, true/false)")


def _parse_number(name: str, raw: str, kind: type) -> Any:
    try:
        value = kind(raw.strip())
    except ValueError as exc:
        raise LLMJudgeConfigError(f"{name} must be a {kind.__name__}") from exc
    if isinstance(value, float) and not math.isfinite(value):
        raise LLMJudgeConfigError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class LLMJudgeConfig:
    """Server-side settings of the integrity review.

    ``mode``
        ``off`` (dataclass default, for in-process configs; the environment
        loader defaults to ``enforce``): no review.  ``shadow``: every covered submission is
        reviewed and recorded host-side, but the agent sees nothing and the
        GPU Judge always runs.  ``enforce``: a verified violation ends the
        submission before any GPU work and its feedback goes to the agent.
    ``on_error``
        What happens when no verdict could be obtained (endpoint down, every
        reply unparseable): ``admit`` runs the GPU Judge as if the review
        passed and records the failure; ``block`` ends the submission as an
        evaluation infrastructure error.
    ``votes`` / ``min_confidence``
        Number of independent reviews and the confidence a VIOLATION needs.
        A submission is rejected only when a strict majority of the votes are
        confident violations whose cited evidence was found in the submission.
    """

    mode: str = "off"
    engine: str = "agent"
    agent: str = "codex"
    agent_runtime_dir: Path | None = None
    codex_auth_json: Path | None = None
    image: str = "judge"
    network: str = "isolated"
    egress_allow: tuple[str, ...] = ()
    cpu_limit: int = 4
    concurrency: int = 2
    mem_limit: str = "16g"
    output_schema: bool = True
    api_base_url: str | None = None
    api_key_file: Path | None = None
    api_key_env: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    extra_body: Mapping[str, Any] = field(default_factory=dict)
    temperature: float = 0.0
    max_output_tokens: int = 16384
    request_timeout_seconds: float = 1500.0
    review_timeout_seconds: float = 1800.0
    max_attempts: int = 3
    votes: int = 1
    min_confidence: float = 0.7
    on_error: str = "admit"
    kinds: tuple[str, ...] = ("agent", "auto")
    context_char_budget: int = 400_000
    file_char_cap: int = 60_000
    json_mode: bool = True
    announce: bool = True
    policy_dir: Path | None = None
    cache: bool = True

    def __post_init__(self) -> None:
        for name, value, choices in (
            ("MODE", self.mode, MODES),
            ("ENGINE", self.engine, ENGINES),
            ("AGENT", self.agent, REVIEW_AGENTS),
            ("IMAGE", self.image, REVIEW_IMAGES),
            ("NETWORK", self.network, REVIEW_NETWORKS),
        ):
            if value not in choices:
                raise LLMJudgeConfigError(
                    f"{ENV_PREFIX}{name} must be one of {', '.join(choices)}"
                )
        if self.on_error not in ON_ERROR_POLICIES:
            raise LLMJudgeConfigError(
                f"{ENV_PREFIX}ON_ERROR must be one of "
                f"{', '.join(ON_ERROR_POLICIES)}"
            )
        unknown = set(self.kinds) - set(SUBMISSION_KINDS)
        if unknown or not self.kinds:
            raise LLMJudgeConfigError(
                f"{ENV_PREFIX}KINDS must list submission kinds from "
                f"{', '.join(SUBMISSION_KINDS)}"
            )
        if not 0.0 <= self.temperature <= 2.0:
            raise LLMJudgeConfigError(f"{ENV_PREFIX}TEMPERATURE must be in [0, 2]")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise LLMJudgeConfigError(
                f"{ENV_PREFIX}MIN_CONFIDENCE must be in [0, 1]"
            )
        for name, value, minimum in (
            ("MAX_OUTPUT_TOKENS", self.max_output_tokens, 256),
            ("MAX_ATTEMPTS", self.max_attempts, 1),
            ("VOTES", self.votes, 1),
            ("CONTEXT_CHARS", self.context_char_budget, 20_000),
            ("FILE_CHARS", self.file_char_cap, 2_000),
            ("CPU_LIMIT", self.cpu_limit, 1),
            ("CONCURRENCY", self.concurrency, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise LLMJudgeConfigError(
                    f"{ENV_PREFIX}{name} must be an integer >= {minimum}"
                )
        for name, value in (
            ("REQUEST_TIMEOUT", self.request_timeout_seconds),
            ("REVIEW_TIMEOUT", self.review_timeout_seconds),
        ):
            if not math.isfinite(value) or value <= 0:
                raise LLMJudgeConfigError(f"{ENV_PREFIX}{name} must be positive")
        if not isinstance(self.extra_body, Mapping):
            raise LLMJudgeConfigError(f"{ENV_PREFIX}EXTRA_BODY must be a JSON object")

    def check_ready(self) -> None:
        """Raise unless an enabled review has everything it needs to run.

        Called by whatever is about to review (the Judge server at start,
        the offline CLI), not at load time: commands that never review must
        not fail on missing review settings.
        """

        if not self.enabled:
            return
        if not self.model:
            raise LLMJudgeConfigError(
                f"{ENV_PREFIX}MODEL is required when the review is on"
            )
        if self.api_key_file is not None and self.api_key_env is not None:
            raise LLMJudgeConfigError(
                f"set at most one of {ENV_PREFIX}API_KEY_FILE and "
                f"{ENV_PREFIX}API_KEY_ENV"
            )
        if self.engine == "api":
            if not self.api_base_url:
                raise LLMJudgeConfigError(
                    f"{ENV_PREFIX}API_BASE_URL is required by the api engine"
                )
            return
        if self.agent_runtime_dir is None:
            raise LLMJudgeConfigError(
                f"{ENV_PREFIX}AGENT_RUNTIME_DIR is required by the agent engine"
            )
        if self.agent == "codex":
            if self.codex_auth_json is None and not self.api_base_url:
                raise LLMJudgeConfigError(
                    f"the codex review agent needs {ENV_PREFIX}CODEX_AUTH_JSON "
                    f"or {ENV_PREFIX}API_BASE_URL"
                )
            if self.codex_auth_json is not None and self.api_base_url:
                raise LLMJudgeConfigError(
                    f"set only one of {ENV_PREFIX}CODEX_AUTH_JSON and "
                    f"{ENV_PREFIX}API_BASE_URL for the codex review agent"
                )
        elif self.api_key_file is None and self.api_key_env is None:
            raise LLMJudgeConfigError(
                f"the claude-code review agent needs {ENV_PREFIX}API_KEY_FILE "
                f"or {ENV_PREFIX}API_KEY_ENV"
            )

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def enforcing(self) -> bool:
        return self.mode == "enforce"

    def covers(self, kind: str) -> bool:
        return self.enabled and kind in self.kinds

    def read_api_key(self) -> str | None:
        """Return the API key, or None for an endpoint without auth."""

        if self.api_key_file is not None:
            try:
                key = self.api_key_file.expanduser().read_text(encoding="utf-8").strip()
            except OSError as exc:
                raise LLMJudgeConfigError(
                    f"cannot read {ENV_PREFIX}API_KEY_FILE"
                ) from exc
            if not key:
                raise LLMJudgeConfigError(f"{ENV_PREFIX}API_KEY_FILE is empty")
            return key
        if self.api_key_env is not None:
            key = os.environ.get(self.api_key_env, "").strip()
            if not key:
                raise LLMJudgeConfigError(
                    f"{ENV_PREFIX}API_KEY_ENV names an unset variable"
                )
            return key
        return None

    def identity(self) -> dict[str, Any]:
        """Secret-free settings that determine a verdict (cache key, logs)."""

        return {
            "engine": self.engine,
            "agent": self.agent if self.engine == "agent" else None,
            "image": self.image if self.engine == "agent" else None,
            "auth": "chatgpt" if self.codex_auth_json is not None else "api-key",
            "output_schema": self.output_schema,
            "api_base_url": self.api_base_url,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "extra_body": dict(self.extra_body),
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "votes": self.votes,
            "min_confidence": self.min_confidence,
            "context_char_budget": self.context_char_budget,
            "file_char_cap": self.file_char_cap,
        }


_SIMPLE_FIELDS: dict[str, tuple[str, type]] = {
    "MODE": ("mode", str),
    "ENGINE": ("engine", str),
    "AGENT": ("agent", str),
    "AGENT_RUNTIME_DIR": ("agent_runtime_dir", Path),
    "CODEX_AUTH_JSON": ("codex_auth_json", Path),
    "IMAGE": ("image", str),
    "NETWORK": ("network", str),
    "CPU_LIMIT": ("cpu_limit", int),
    "CONCURRENCY": ("concurrency", int),
    "MEM_LIMIT": ("mem_limit", str),
    "API_BASE_URL": ("api_base_url", str),
    "API_KEY_FILE": ("api_key_file", Path),
    "API_KEY_ENV": ("api_key_env", str),
    "MODEL": ("model", str),
    "REASONING_EFFORT": ("reasoning_effort", str),
    "TEMPERATURE": ("temperature", float),
    "MAX_OUTPUT_TOKENS": ("max_output_tokens", int),
    "REQUEST_TIMEOUT": ("request_timeout_seconds", float),
    "REVIEW_TIMEOUT": ("review_timeout_seconds", float),
    "MAX_ATTEMPTS": ("max_attempts", int),
    "VOTES": ("votes", int),
    "MIN_CONFIDENCE": ("min_confidence", float),
    "ON_ERROR": ("on_error", str),
    "CONTEXT_CHARS": ("context_char_budget", int),
    "FILE_CHARS": ("file_char_cap", int),
    "POLICY_DIR": ("policy_dir", Path),
}
_BOOL_FIELDS = {
    "OUTPUT_SCHEMA": "output_schema",
    "JSON_MODE": "json_mode",
    "ANNOUNCE": "announce",
    "CACHE": "cache",
}


def load_llm_judge_config(
    environ: Mapping[str, str] | None = None,
    *,
    fallbacks: Mapping[str, Any] | None = None,
    **overrides: Any,
) -> LLMJudgeConfig:
    """Build the config from ``SFORGE_LLM_JUDGE_*`` variables plus overrides.

    Empty variables count as unset.  Keyword overrides (from CLI flags) win
    over the environment; ``None`` overrides are ignored.  Unset, the mode is
    ``enforce`` and the model must be named.  ``fallbacks`` supplies the
    run's own ``agent_runtime_dir`` / ``codex_auth_json`` for settings neither
    the environment nor an override gives.
    """

    env = os.environ if environ is None else environ
    values: dict[str, Any] = {}
    for suffix, (name, kind) in _SIMPLE_FIELDS.items():
        raw = env.get(ENV_PREFIX + suffix, "")
        if not raw.strip():
            continue
        if kind is str:
            values[name] = raw.strip()
        elif kind is Path:
            values[name] = Path(raw.strip())
        else:
            values[name] = _parse_number(ENV_PREFIX + suffix, raw, kind)
    for suffix, name in _BOOL_FIELDS.items():
        raw = env.get(ENV_PREFIX + suffix, "")
        if raw.strip():
            values[name] = _parse_bool(ENV_PREFIX + suffix, raw)
    for suffix, name in (("KINDS", "kinds"), ("EGRESS_ALLOW", "egress_allow")):
        raw_list = env.get(ENV_PREFIX + suffix, "")
        if raw_list.strip():
            values[name] = tuple(
                dict.fromkeys(
                    item.strip() for item in raw_list.split(",") if item.strip()
                )
            )
    raw_extra = env.get(ENV_PREFIX + "EXTRA_BODY", "")
    if raw_extra.strip():
        try:
            extra = json.loads(raw_extra)
        except json.JSONDecodeError as exc:
            raise LLMJudgeConfigError(
                f"{ENV_PREFIX}EXTRA_BODY must be a JSON object"
            ) from exc
        if not isinstance(extra, dict):
            raise LLMJudgeConfigError(f"{ENV_PREFIX}EXTRA_BODY must be a JSON object")
        values["extra_body"] = extra
    for name, value in overrides.items():
        if value is None:
            continue
        if name not in LLMJudgeConfig.__dataclass_fields__:
            raise LLMJudgeConfigError(f"unknown integrity-review setting {name!r}")
        values[name] = value
    values.setdefault("mode", DEFAULT_MODE)
    if "model" not in values:
        values["model"] = DEFAULT_MODEL
    values.setdefault("reasoning_effort", DEFAULT_REASONING_EFFORT)
    fallbacks = fallbacks or {}
    runtime_dir = fallbacks.get("agent_runtime_dir") or env.get("SFORGE_AGENT_RUNTIME_DIR")
    auth_json = (
        fallbacks.get("codex_auth_json")
        or env.get("SFORGE_CODEX_AUTH_JSON_PATH")
        or env.get("CODEX_AUTH_JSON_PATH")
    )
    if values.get("agent_runtime_dir") is None and runtime_dir:
        values["agent_runtime_dir"] = Path(runtime_dir)
    if (
        values.get("codex_auth_json") is None
        and not values.get("api_base_url")
        and values.get("agent", "codex") == "codex"
        and auth_json
    ):
        values["codex_auth_json"] = Path(auth_json)
    return LLMJudgeConfig(**values)
