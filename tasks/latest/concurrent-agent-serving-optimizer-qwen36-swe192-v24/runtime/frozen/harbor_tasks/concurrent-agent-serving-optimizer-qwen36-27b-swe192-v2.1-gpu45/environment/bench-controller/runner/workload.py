from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from jinja2 import StrictUndefined, Template
from minisweagent.agents.default import (
    DefaultAgent,
    extract_bash_action,
)

from runner.corpus import TaskSpec, corpus_fingerprint, make_specs, repository_context
from runner.ledger import Ledger, LedgerClosed, MEASUREMENT_WINDOW_REASON
from runner.metrics import (
    assess_pressure,
    jain,
    metric_point,
    percentile,
    summarize_metric_delta,
    summarize_metric_series,
)
from runner.model import BenchmarkVllmModel, MeasurementWindowEnded, ModelConfig
from runner.sandbox import TimedSandbox


# Frozen from examples/inference/mini-swe-agent/src/minisweagent/config/extra/
# swebench.yaml at ThunderAgent commit a989f24c2d53096e243aae39775637b8ddd0e380.
# The benchmark must not depend on the parent checkout at runtime.
THUNDERAGENT_SWEBENCH_PROMPT_CONTRACT = (
    "thunderagent-a989f24-mini-swe-agent-swebench-yaml-v1"
)

# Warm-up is an infrastructure/protocol check, not part of the workload
# randomization.  Keep its real SWE instances and request seeds stable across
# profiles so changing a measured corpus seed cannot turn readiness into a
# different model-quality trial.  The measured leg is reset after warm-up and
# continues to use ``profile.seed``.
WARMUP_CORPUS_SEED = 410_148
WARMUP_MODEL_SEED_BASE = 1_310_148
THUNDERAGENT_SWEBENCH_SYSTEM_TEMPLATE = """You are a helpful assistant that can interact multiple times with a computer shell to solve programming tasks.
Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).

Include a THOUGHT section before your command where you explain your reasoning process.
Format your response as shown in <format_example>.

<format_example>
THOUGHT: Your reasoning and analysis here

```bash
your_command_here
```
</format_example>

Failure to follow these rules will cause your response to be rejected.
"""
THUNDERAGENT_SWEBENCH_INSTANCE_TEMPLATE = """<pr_description>
Consider the following PR description:
{{task}}
</pr_description>

<instructions>
# Task Instructions

## Overview
You're a software engineer interacting continuously with a computer by submitting commands.
You'll be helping implement necessary changes to meet requirements in the PR description.
Your task is specifically to make changes to non-test files in the current directory in order to fix the issue described in the PR description in a way that is general and consistent with the codebase.

IMPORTANT: This is an interactive process where you will think and issue ONE command, see its result, then think and issue your next command.

For each response:
1. Include a THOUGHT section explaining your reasoning and what you're trying to accomplish
2. Provide exactly ONE bash command to execute

## Important Boundaries
- MODIFY: Regular source code files in /testbed (this is the working directory for all your subsequent commands)
- DO NOT MODIFY: Tests, configuration files (pyproject.toml, setup.cfg, etc.)

## Recommended Workflow
1. Analyze the codebase by finding and reading relevant files
2. Create a script to reproduce the issue
3. Edit the source code to resolve the issue
4. Verify your fix works by running your script again
5. Test edge cases to ensure your fix is robust

## Command Execution Rules
You are operating in an environment where
1. You write a single command
2. The system executes that command in a subshell
3. You see the result
4. You write your next command

Each response should include:
1. A **THOUGHT** section where you explain your reasoning and plan
2. A single bash code block with your command

Format your responses like this:

<format_example>
THOUGHT: Here I explain my reasoning process, analysis of the current situation,
and what I'm trying to accomplish with the command below.

```bash
your_command_here
```
</format_example>

Commands must be specified in a single bash code block:

```bash
your_command_here
```

**CRITICAL REQUIREMENTS:**
- Your response SHOULD include a THOUGHT section explaining your reasoning
- Your response MUST include EXACTLY ONE bash code block
- This bash block MUST contain EXACTLY ONE command (or a set of commands connected with && or ||)
- If you include zero or multiple bash blocks, or no command at all, YOUR RESPONSE WILL FAIL
- Do NOT try to run multiple independent commands in separate blocks in one response
- Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
- However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment variables from files

Example of a CORRECT response:
<example_response>
THOUGHT: I need to understand the structure of the repository first. Let me check what files are in the current directory to get a better understanding of the codebase.

```bash
ls -la
```
</example_response>

Example of an INCORRECT response:
<example_response>
THOUGHT: I need to examine the codebase and then look at a specific file. I'll run multiple commands to do this.

```bash
ls -la
```

Now I'll read the file:

```bash
cat file.txt
```
</example_response>

If you need to run multiple commands, either:
1. Combine them in one block using && or ||
```bash
command1 && command2 || echo "Error occurred"
```

2. Wait for the first command to complete, see its output, then issue the next command in your following response.

## Environment Details
- You have a full Linux shell environment
- Always use non-interactive flags (-y, -f) for commands
- Avoid interactive tools like vi, nano, or any that require user input
- If a command isn't available, you can install it

## Useful Command Examples

### Create a new file:
```bash
cat <<'EOF' > newfile.py
import numpy as np
hello = "world"
print(hello)
EOF
```

### Edit files with sed:
```bash
# Replace all occurrences
sed -i 's/old_string/new_string/g' filename.py

# Replace only first occurrence
sed -i 's/old_string/new_string/' filename.py

# Replace first occurrence on line 1
sed -i '1s/old_string/new_string/' filename.py

# Replace all occurrences in lines 1-10
sed -i '1,10s/old_string/new_string/g' filename.py
```

### View file content:
```bash
# View specific lines with numbers
nl -ba filename.py | sed -n '10,20p'
```

### Any other command you want to run
```bash
anything
```

## Submission
When you've completed your work (reading, editing, testing), and cannot make further progress
issue exactly the following command:

```bash
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached
```

This command will submit your work.
You cannot continue working (reading, editing, testing) in any way on this task after submitting.
</instructions>
"""
THUNDERAGENT_SWEBENCH_FORMAT_ERROR_TEMPLATE = """Please always provide EXACTLY ONE action in triple backticks, found {{actions|length}} actions.

Please format your action in triple backticks as shown in <response_example>.

<response_example>
Here are some thoughts about why you want to perform the action.

```bash
<action>
```
</response_example>

If you have completed your assignment, please consult the first message about how to
submit your solution (you will not be able to continue working on this task after that).
"""


@dataclass(frozen=True)
class ModelAdapterContract:
    mode: str
    instruction_role: str
    instruction: str
    instance_template: str
    format_error_template: str
    temperature: float
    top_p: float

    def model_config(self, max_tokens: int) -> ModelConfig:
        return ModelConfig(
            max_tokens=self.effective_max_tokens(max_tokens),
            temperature=self.temperature,
            top_p=self.top_p,
        )

    def effective_max_tokens(self, configured_max_tokens: int) -> int:
        # Qualification and pressure profiles deliberately use different caps.
        # The adapter must report and honor the frozen profile value exactly.
        return configured_max_tokens

    def render_instance(self, task: str) -> str:
        return Template(self.instance_template, undefined=StrictUndefined).render(
            task=task
        )

    def render_format_error(self, actions: list[object]) -> str:
        return Template(
            self.format_error_template,
            undefined=StrictUndefined,
        ).render(actions=actions)

    def summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "instruction_role": self.instruction_role,
            "instruction_sha256": hashlib.sha256(self.instruction.encode()).hexdigest(),
            "instance_template_sha256": hashlib.sha256(
                self.instance_template.encode()
            ).hexdigest(),
            "format_error_sha256": hashlib.sha256(
                self.format_error_template.encode()
            ).hexdigest(),
            "prompt_contract": THUNDERAGENT_SWEBENCH_PROMPT_CONTRACT,
            "task_input": "raw-problem-statement",
            "offline_constraint": "sandbox-enforced-not-prompt-modified",
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens_policy": "profile_cap",
            "response_protocol": "single-bash-fence",
            "function_tools": False,
            "reasoning_override": "qwen-chat-template-thinking-disabled",
            "chat_template_kwargs": {"enable_thinking": False},
        }


def load_model_adapter_contract() -> ModelAdapterContract:
    mode = (
        os.environ.get("KV_MODEL_ADAPTER_MODE", "qwen36-text-bash-v1").strip()
        or "qwen36-text-bash-v1"
    )
    if mode == "qwen36-text-bash-v1":
        return ModelAdapterContract(
            mode=mode,
            instruction_role="system",
            instruction=THUNDERAGENT_SWEBENCH_SYSTEM_TEMPLATE,
            instance_template=THUNDERAGENT_SWEBENCH_INSTANCE_TEMPLATE,
            format_error_template=THUNDERAGENT_SWEBENCH_FORMAT_ERROR_TEMPLATE,
            temperature=0.0,
            top_p=1.0,
        )
    raise ValueError(f"unsupported KV_MODEL_ADAPTER_MODE: {mode}")


def workflow_cache_namespace(
    workload_nonce: str,
    leg_id: str,
    workflow_identity: str,
) -> str:
    """Return an opaque, stable namespace for one workflow within one leg."""
    return hashlib.sha256(
        f"{workload_nonce}:{leg_id}:{workflow_identity}".encode()
    ).hexdigest()


@dataclass(frozen=True)
class Profile:
    name: str
    enabled: bool
    workflows: int
    concurrency: int
    max_steps: int
    context_bytes: int
    max_tokens: int
    workflow_timeout_sec: int
    request_timeout_sec: int
    tool_delay_min_sec: float
    tool_delay_max_sec: float
    seed: int
    warmup_requests: int = 2
    warmup_context_bytes: int = 2048
    warmup_steps: int = 1
    workload_family: str = "synthetic-fixture"
    metrics_sample_interval_sec: float = 1.0
    metrics_request_timeout_sec: int = 15
    pressure_required: bool = False
    pressure_kv_threshold: float = 0.85
    pressure_kv_p95_min: float = 0.85
    pressure_time_fraction_min: float = 0.10
    pressure_min_samples: int = 5
    corpus_partition: str = "fixture"
    corpus_unique_instances_min: int = 1
    tool_timeout_sec: int = 60
    sandbox_prepare_concurrency: int = 16
    backend_stall_timeout_sec: int = 30
    candidate_admission_stall_timeout_sec: int = 60
    measurement_window_sec: int | None = None
    measurement_warmup_sec: int = 0

    def __post_init__(self) -> None:
        # Agent workers and vLLM sequence slots are deliberately separate
        # resources. Profiles may use the frozen 192-sequence Qwen ceiling so a
        # program-aware gateway has real admission choices; each profile-set
        # loader must opt into that ceiling explicitly.
        if self.workflows <= 0 or not 1 <= self.concurrency <= min(self.workflows, 192):
            raise ValueError(f"invalid workflow/concurrency shape for {self.name}")
        # Upstream ThunderAgent mini-SWE uses 135 steps. Development/scoring profiles
        # may choose a lower bounded budget, but the contract must not make a
        # paper-fidelity qualification profile impossible.
        if not 1 <= self.max_steps <= 135:
            raise ValueError(f"invalid max_steps for {self.name}")
        if not 0 <= self.context_bytes <= 512 * 1024:
            raise ValueError(f"invalid context_bytes for {self.name}")
        if not 1 <= self.max_tokens <= 4096:
            raise ValueError(f"invalid max_tokens for {self.name}")
        if (
            self.workflow_timeout_sec <= 0
            or self.request_timeout_sec <= 0
            or self.request_timeout_sec > self.workflow_timeout_sec
        ):
            raise ValueError(f"invalid timeout for {self.name}")
        if not 0 <= self.tool_delay_min_sec <= self.tool_delay_max_sec:
            raise ValueError(f"invalid tool delay range for {self.name}")
        if not 0 <= self.warmup_requests <= min(self.workflows, 16):
            raise ValueError(f"invalid warmup request count for {self.name}")
        if not 0 <= self.warmup_context_bytes <= 512 * 1024:
            raise ValueError(f"invalid warmup context size for {self.name}")
        if not 1 <= self.warmup_steps <= self.max_steps:
            raise ValueError(f"invalid warmup steps for {self.name}")
        if self.workload_family not in {"synthetic-fixture", "swebench-lite"}:
            raise ValueError(f"invalid workload family for {self.name}: {self.workload_family}")
        if self.corpus_partition not in {"fixture", "public", "calibration", "hidden"}:
            raise ValueError(f"invalid corpus partition for {self.name}: {self.corpus_partition}")
        if self.workload_family == "swebench-lite" and self.corpus_partition == "fixture":
            raise ValueError(f"SWE-bench profile has no corpus partition: {self.name}")
        if not 1 <= self.corpus_unique_instances_min <= self.workflows:
            raise ValueError(f"invalid unique corpus minimum for {self.name}")
        if not 0.1 <= self.metrics_sample_interval_sec <= 10:
            raise ValueError(f"invalid metrics sampling interval for {self.name}")
        if not 5 <= self.metrics_request_timeout_sec <= 60:
            raise ValueError(f"invalid metrics request timeout for {self.name}")
        for label, value in (
            ("pressure_kv_threshold", self.pressure_kv_threshold),
            ("pressure_kv_p95_min", self.pressure_kv_p95_min),
            ("pressure_time_fraction_min", self.pressure_time_fraction_min),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"invalid {label} for {self.name}")
        if not 1 <= self.pressure_min_samples <= 10_000:
            raise ValueError(f"invalid pressure sample count for {self.name}")
        if not 1 <= self.tool_timeout_sec <= 120:
            raise ValueError(f"invalid tool timeout for {self.name}")
        if not 1 <= self.sandbox_prepare_concurrency <= 32:
            raise ValueError(f"invalid sandbox preparation concurrency for {self.name}")
        if not 10 <= self.backend_stall_timeout_sec <= 300:
            raise ValueError(f"invalid backend stall timeout for {self.name}")
        if not 10 <= self.candidate_admission_stall_timeout_sec <= 300:
            raise ValueError(f"invalid candidate admission stall timeout for {self.name}")
        if (
            not isinstance(self.measurement_warmup_sec, int)
            or isinstance(self.measurement_warmup_sec, bool)
            or not 0 <= self.measurement_warmup_sec <= self.workflow_timeout_sec
        ):
            raise ValueError(f"invalid measurement warmup for {self.name}")
        if self.measurement_warmup_sec and self.measurement_window_sec is None:
            raise ValueError(
                f"measurement warmup requires a fixed measurement window: {self.name}"
            )
        if self.measurement_window_sec is not None:
            if (
                not isinstance(self.measurement_window_sec, int)
                or isinstance(self.measurement_window_sec, bool)
                or not 60 <= self.measurement_window_sec <= self.workflow_timeout_sec
            ):
                raise ValueError(f"invalid measurement window for {self.name}")
            if (
                self.measurement_warmup_sec + self.measurement_window_sec
                > self.workflow_timeout_sec
            ):
                raise ValueError(
                    f"measurement warmup + window exceeds workflow timeout: {self.name}"
                )
            if self.workflows != self.concurrency:
                raise ValueError(
                    f"fixed-window profile must use one live workflow wave: {self.name}"
                )


def load_profiles(
    path: Path,
    *,
    concurrency_limit: int = 128,
) -> dict[str, Profile]:
    if not 1 <= concurrency_limit <= 192:
        raise ValueError(f"invalid profile concurrency limit: {concurrency_limit}")
    raw = json.loads(path.read_text())
    profiles = {
        name: Profile(name=name, **config) for name, config in raw["profiles"].items()
    }
    over_limit = sorted(
        profile.name
        for profile in profiles.values()
        if profile.concurrency > concurrency_limit
    )
    if over_limit:
        raise ValueError(
            f"profiles exceed concurrency limit {concurrency_limit}: {over_limit}"
        )
    return profiles


def candidate_leg_wall_timeout_sec(profile: Profile) -> int:
    """Derive the measured candidate-leg wall from existing frozen budgets.

    Each logical wave may consume one per-workflow job timeout.  Keeping this
    derived avoids a second, independently drifting timeout in profile JSON,
    while an active timer makes the previously passive executor bound
    cancellable. The pool keeps a short cleanup grace outside this wall.
    """

    return math.ceil(profile.workflows / profile.concurrency) * profile.workflow_timeout_sec


def admin_json(
    method: str,
    base: str,
    path: str,
    token: str,
    timeout: int = 60,
    *,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = None
    headers = {"Authorization": f"Bearer {token}"}
    if payload is not None:
        data = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        f"{base.rstrip('/')}{path}",
        data=data,
        method=method,
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError(f"admin endpoint {path} returned {response.status}")
            raw = response.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"admin endpoint {path} returned {exc.code}: {exc.read()!r}") from exc


def release_gateway_program(
    endpoint: str,
    program_id: str,
    *,
    enabled: bool,
    timeout: float = 5.0,
) -> dict[str, Any]:
    """Send ThunderAgent's lifecycle termination hint to a candidate gateway.

    This control request never goes to the model backend and therefore is not a
    model step or part of the one-to-one data-plane audit.  Gateways that do not
    implement the optional endpoint may return 404/405 without failing a run.
    """

    result: dict[str, Any] = {
        "attempted": enabled,
        "acknowledged": False,
        "supported": None,
        "status": None,
        "error": None,
    }
    if not enabled:
        return result
    parsed = urllib.parse.urlsplit(endpoint)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    release_url = urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, f"{path}/programs/release", "", "")
    )
    body = json.dumps(
        {"program_id": program_id}, separators=(",", ":"), sort_keys=True
    ).encode()
    request = urllib.request.Request(
        release_url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Agent-Run-ID": program_id,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read(1024**2)
            result["status"] = int(response.status)
            result["supported"] = True
            result["acknowledged"] = 200 <= response.status < 300
    except urllib.error.HTTPError as exc:
        exc.read(1024**2)
        result["status"] = int(exc.code)
        result["supported"] = exc.code not in {404, 405}
        if exc.code not in {404, 405}:
            result["error"] = f"HTTP {exc.code}"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


class AdmissionStallDetector:
    """Detect an admission queue that cannot place even one backend request."""

    def __init__(self, timeout_sec: float) -> None:
        self.timeout_sec = timeout_sec
        self.stalled_since: float | None = None

    def observe(self, point: dict[str, float | None], now: float) -> str | None:
        running = point.get("requests_running")
        waiting = point.get("requests_waiting")
        kv_usage = point.get("kv_cache_usage")
        stalled = bool(
            isinstance(running, (int, float))
            and isinstance(waiting, (int, float))
            and isinstance(kv_usage, (int, float))
            and running == 0
            and waiting > 0
            and kv_usage <= 0.001
        )
        if not stalled:
            self.stalled_since = None
            return None
        if self.stalled_since is None:
            self.stalled_since = now
            return None
        duration = now - self.stalled_since
        if duration < self.timeout_sec:
            return None
        return (
            "backend_admission_stall:"
            f"running={running:.0f},waiting={waiting:.0f},kv={kv_usage:.4f},"
            f"duration_sec={duration:.1f}"
        )


class CandidateAdmissionStallDetector:
    """Bound a candidate gateway's zero-backend-admission interval.

    Deliberate scheduling queues are valid and can remain deep while backend
    work is active.  This detector therefore does not compare client and
    backend queue depths.  It fires only when measured client requests are in
    flight while the trusted model API has no active request and its completed
    audit count has made no progress for a continuous, frozen profile window.
    Both trusted values ride on the existing metrics poll; the watchdog never
    downloads the growing full audit ledger during measurement.
    """

    def __init__(self, timeout_sec: float) -> None:
        self.timeout_sec = timeout_sec
        self.stalled_since: float | None = None
        self.last_audit_count: int | None = None

    def observe(
        self,
        client_activity: dict[str, Any],
        trusted_activity: dict[str, Any],
        now: float,
    ) -> str | None:
        client_active = client_activity.get("active_requests")
        backend_active = trusted_activity.get("active_backend_requests")
        audit_count = trusted_activity.get("audit_count")
        if (
            not isinstance(client_active, int)
            or isinstance(client_active, bool)
            or client_active < 0
            or not isinstance(backend_active, int)
            or isinstance(backend_active, bool)
            or backend_active < 0
            or not isinstance(audit_count, int)
            or isinstance(audit_count, bool)
            or audit_count < 0
        ):
            self.stalled_since = None
            self.last_audit_count = None
            return None

        audit_progress = (
            self.last_audit_count is not None
            and audit_count > self.last_audit_count
        )
        audit_restarted = (
            self.last_audit_count is not None
            and audit_count < self.last_audit_count
        )
        self.last_audit_count = audit_count
        if client_active == 0 or backend_active > 0 or audit_progress or audit_restarted:
            self.stalled_since = None
            return None
        if self.stalled_since is None:
            self.stalled_since = now
            return None
        duration = now - self.stalled_since
        if duration < self.timeout_sec:
            return None
        oldest = client_activity.get("oldest_active_age_sec")
        oldest_text = (
            f"{float(oldest):.1f}"
            if isinstance(oldest, (int, float)) and not isinstance(oldest, bool)
            else "unknown"
        )
        return (
            "candidate_gateway_admission_stall:"
            f"client_active={client_active},backend_active={backend_active},"
            f"audited_completed={audit_count},oldest_client_sec={oldest_text},"
            f"duration_sec={duration:.1f}"
        )


class AgentExecutionTracker:
    """Freeze a finite leg when every expected ``agent.run`` has terminated.

    Agent execution and sandbox semantic verification intentionally have
    different timing contracts.  Workers report the former here immediately
    from the ``agent.run`` finally block; slow verification and cleanup remain
    visible in leg wall/teardown time without extending the throughput
    denominator.  Both clocks are retained because trusted metrics carry Unix
    timestamps while elapsed durations must be immune to wall-clock changes.
    """

    def __init__(self, expected_agent_runs: int) -> None:
        if (
            not isinstance(expected_agent_runs, int)
            or isinstance(expected_agent_runs, bool)
            or expected_agent_runs <= 0
        ):
            raise ValueError("expected_agent_runs must be a positive integer")
        self.expected_agent_runs = expected_agent_runs
        self.complete = threading.Event()
        self._lock = threading.Lock()
        self._ends: dict[int, tuple[float, float]] = {}
        self._errors: list[str] = []

    @staticmethod
    def _finite_number(value: object) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )

    def _reject(self, detail: str) -> None:
        with self._lock:
            self._errors.append(detail)
        raise RuntimeError(f"agent execution tracker rejected mark: {detail}")

    def mark(
        self,
        ordinal: int,
        *,
        ended_unix: float,
        ended_monotonic: float,
    ) -> None:
        """Record exactly one termination for one logical workflow ordinal."""

        if (
            not isinstance(ordinal, int)
            or isinstance(ordinal, bool)
            or not 0 <= ordinal < self.expected_agent_runs
        ):
            self._reject(f"ordinal_out_of_range:{ordinal!r}")
        if not self._finite_number(ended_unix):
            self._reject(f"invalid_ended_unix:{ended_unix!r}")
        if not self._finite_number(ended_monotonic):
            self._reject(f"invalid_ended_monotonic:{ended_monotonic!r}")

        with self._lock:
            if ordinal in self._ends:
                self._errors.append(f"duplicate_ordinal:{ordinal}")
                raise RuntimeError(
                    f"agent execution tracker rejected duplicate ordinal {ordinal}"
                )
            self._ends[ordinal] = (float(ended_unix), float(ended_monotonic))
            if len(self._ends) == self.expected_agent_runs and not self._errors:
                self.complete.set()

    def snapshot(
        self,
        *,
        started_unix: float,
        started_monotonic: float,
    ) -> dict[str, Any]:
        """Return the auditable complete interval, or fail closed."""

        if not self._finite_number(started_unix):
            raise RuntimeError("agent execution tracker received an invalid Unix start")
        if not self._finite_number(started_monotonic):
            raise RuntimeError("agent execution tracker received an invalid monotonic start")
        with self._lock:
            ends = dict(self._ends)
            errors = list(self._errors)
        if errors or len(ends) != self.expected_agent_runs or not self.complete.is_set():
            raise RuntimeError(
                "agent execution tracker incomplete: "
                f"observed={len(ends)}/{self.expected_agent_runs},"
                f"errors={errors[:10]}"
            )
        invalid_ordinals = [
            ordinal
            for ordinal, (ended_unix, ended_monotonic) in ends.items()
            if ended_unix < float(started_unix)
            or ended_monotonic < float(started_monotonic)
        ]
        if invalid_ordinals:
            raise RuntimeError(
                "agent execution tracker observed an end before the global start: "
                f"{sorted(invalid_ordinals)}"
            )
        _, (ended_unix, ended_monotonic) = max(
            ends.items(), key=lambda item: item[1][1]
        )
        elapsed = ended_monotonic - float(started_monotonic)
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError(
                f"agent execution tracker produced invalid elapsed time: {elapsed!r}"
            )
        return {
            "schema_version": 1,
            "start_event": "measured_leg_started",
            "end_event": "last_agent_run_termination",
            "duration_clock": "monotonic",
            "started_unix": float(started_unix),
            "ended_unix": ended_unix,
            "elapsed_sec": elapsed,
            "expected_agent_runs": self.expected_agent_runs,
            "observed_agent_runs": len(ends),
            "complete": True,
        }


def finite_metric_points_through(
    points: list[dict[str, float | None]],
    ended_unix: float,
) -> list[dict[str, float | None]]:
    """Keep trusted live samples created through a finite agent end."""

    if not AgentExecutionTracker._finite_number(ended_unix):
        raise ValueError("finite metric cutoff must be a finite Unix timestamp")
    return [
        point
        for point in points
        if isinstance(point.get("created_unix"), (int, float))
        and not isinstance(point.get("created_unix"), bool)
        and math.isfinite(float(point["created_unix"]))
        and float(point["created_unix"]) <= float(ended_unix)
    ]


def attach_agent_execution_interval(
    result: dict[str, Any],
    interval: dict[str, Any] | None,
) -> dict[str, Any]:
    """Attach the additive finite schema without changing fixed-window JSON."""

    if interval is not None:
        result["agent_execution_interval"] = interval
    return result


class LiveMetricsSampler:
    """Poll trusted vLLM gauges while the measured workload is active."""

    def __init__(
        self,
        model_api_base: str,
        token: str,
        interval_sec: float,
        request_timeout_sec: int,
        stall_timeout_sec: float,
        *,
        mode: str = "baseline",
        ledger: Ledger | None = None,
        candidate_stall_timeout_sec: float = 60,
        candidate_leg_wall_timeout_sec: float | None = None,
        expected_audit_epoch: int | None = None,
        agent_execution_complete_event: threading.Event | None = None,
    ) -> None:
        if mode not in {"baseline", "candidate"}:
            raise ValueError(f"invalid metrics sampler mode: {mode}")
        if mode == "candidate" and ledger is None:
            raise ValueError("candidate metrics sampler requires the measured client ledger")
        if mode == "candidate" and (
            not isinstance(expected_audit_epoch, int)
            or isinstance(expected_audit_epoch, bool)
            or expected_audit_epoch < 0
        ):
            raise ValueError("candidate metrics sampler requires the measured audit epoch")
        if candidate_leg_wall_timeout_sec is not None and (
            mode != "candidate"
            or isinstance(candidate_leg_wall_timeout_sec, bool)
            or not isinstance(candidate_leg_wall_timeout_sec, (int, float))
            or not math.isfinite(float(candidate_leg_wall_timeout_sec))
            or float(candidate_leg_wall_timeout_sec) <= 0
        ):
            raise ValueError(
                "candidate_leg_wall_timeout_sec requires candidate mode and a "
                "finite positive timeout"
            )
        self.model_api_base = model_api_base
        self.token = token
        self.interval_sec = interval_sec
        self.request_timeout_sec = request_timeout_sec
        self.mode = mode
        self.ledger = ledger
        self.expected_audit_epoch = expected_audit_epoch
        self.agent_execution_complete_event = agent_execution_complete_event
        self.candidate_leg_wall_timeout_sec = (
            float(candidate_leg_wall_timeout_sec)
            if candidate_leg_wall_timeout_sec is not None
            else None
        )
        self.points: list[dict[str, float | None]] = []
        self.errors: list[str] = []
        self.fatal_error: str | None = None
        self.fatal_type: str | None = None
        self.client_cancellation: dict[str, Any] | None = None
        self.abort_event = threading.Event()
        self._stall_detector = AdmissionStallDetector(stall_timeout_sec)
        self._candidate_stall_detector = CandidateAdmissionStallDetector(
            candidate_stall_timeout_sec
        )
        self._stop = threading.Event()
        self._abort_lock = threading.Lock()
        self._abort_started = False
        self._thread = threading.Thread(target=self._run, name="kvbench-metrics-sampler", daemon=True)
        self._deadline_timer: threading.Timer | None = None

    def _agent_execution_complete(self) -> bool:
        return (
            self.agent_execution_complete_event is not None
            and self.agent_execution_complete_event.is_set()
        )

    def start(self) -> None:
        if self.candidate_leg_wall_timeout_sec is not None:
            self._deadline_timer = threading.Timer(
                self.candidate_leg_wall_timeout_sec,
                self._candidate_leg_deadline_expired,
            )
            self._deadline_timer.name = "kvbench-candidate-leg-deadline"
            self._deadline_timer.daemon = True
            self._deadline_timer.start()
        self._thread.start()

    def _candidate_leg_deadline_expired(self) -> None:
        timeout = self.candidate_leg_wall_timeout_sec
        if timeout is None or self._stop.is_set() or self._agent_execution_complete():
            return
        client_active = (
            int(self.ledger.activity_snapshot()["active_requests"])
            if self.ledger is not None
            else 0
        )
        self._abort(
            "candidate_leg_wall_timeout",
            "candidate_leg_wall_timeout:"
            f"timeout_sec={timeout:.1f},"
            f"client_active={client_active}",
        )

    def _abort(self, fatal_type: str, detail: str) -> None:
        # The metrics poller and independent wall timer may notice failures at
        # the same instant. Exactly one cause owns cancellation and scoring.
        with self._abort_lock:
            # The final agent termination wins a simultaneous deadline race.
            # Semantic verification and cleanup are outside the finite
            # candidate execution deadline.
            if self._abort_started or self._agent_execution_complete():
                return
            self._abort_started = True
            self.fatal_type = fatal_type
            self.fatal_error = detail
        self.abort_event.set()
        if self.ledger is not None:
            self.client_cancellation = self.ledger.cancel_active(detail)
            detail += (
                ",client_cancelled_requests="
                f"{int(self.client_cancellation['newly_cancelled_requests'])}"
            )
        try:
            cancelled = admin_json(
                "POST",
                self.model_api_base,
                "/admin/requests/cancel",
                self.token,
                timeout=30,
            )
            detail += f",backend_cancelled_requests={int(cancelled.get('cancelled_requests', 0))}"
        except Exception as exc:
            detail += f",backend_cancel_error={type(exc).__name__}:{exc}"
        self.fatal_error = detail
        self.errors.append(detail)

    def _run(self) -> None:
        while not self._stop.is_set() and not self._agent_execution_complete():
            # BenchmarkVllmModel sets this shared event when it observes a
            # bounded-response violation. Escalate the client-side discovery
            # through the trusted abort path so an untrusted gateway cannot
            # leave its already-admitted upstream request running while the
            # controller waits for quiescence.
            if self.abort_event.is_set() and not self._abort_started:
                self._abort(
                    "model_response_limit_exceeded",
                    "model_response_limit_exceeded:client_parser_limit",
                )
                return
            started = time.monotonic()
            try:
                snapshot = admin_json(
                    "GET",
                    self.model_api_base,
                    "/admin/metrics/snapshot",
                    self.token,
                    timeout=self.request_timeout_sec,
                )
                point = metric_point(snapshot)
                self.points.append(point)
                stall = self._stall_detector.observe(point, time.monotonic())
                if stall is not None:
                    self._abort("backend_admission_stall", stall)
                    return
                if self.mode == "candidate":
                    if self.ledger is None:  # guarded by __init__
                        raise RuntimeError("candidate client ledger disappeared")
                    client_activity = self.ledger.activity_snapshot()
                    invalid_fields = [
                        name
                        for name, value in (
                            ("client_active_requests", client_activity.get("active_requests")),
                            ("active_backend_requests", snapshot.get("active_backend_requests")),
                            ("audit_count", snapshot.get("audit_count")),
                            ("audit_epoch", snapshot.get("audit_epoch")),
                        )
                        if not isinstance(value, int)
                        or isinstance(value, bool)
                        or value < 0
                    ]
                    if invalid_fields:
                        self._abort(
                            "candidate_watchdog_telemetry_invalid",
                            "candidate_watchdog_telemetry_invalid:fields="
                            + ",".join(invalid_fields),
                        )
                        return
                    if snapshot["audit_epoch"] != self.expected_audit_epoch:
                        self._abort(
                            "candidate_watchdog_audit_epoch_changed",
                            "candidate_watchdog_audit_epoch_changed:"
                            f"expected={self.expected_audit_epoch},"
                            f"observed={snapshot['audit_epoch']}",
                        )
                        return
                    candidate_stall = self._candidate_stall_detector.observe(
                        client_activity,
                        snapshot,
                        time.monotonic(),
                    )
                    if candidate_stall is not None:
                        self._abort(
                            "candidate_gateway_admission_stall",
                            candidate_stall,
                        )
                        return
            except Exception as exc:
                if not self._agent_execution_complete():
                    self.errors.append(f"{type(exc).__name__}: {exc}")
            remaining = max(0.0, self.interval_sec - (time.monotonic() - started))
            if self.agent_execution_complete_event is None:
                self._stop.wait(remaining)
            else:
                # A shared tracker event closes finite sampling promptly while
                # retaining the ordinary stop event for controller failures.
                deadline = time.monotonic() + remaining
                while (
                    not self._stop.is_set()
                    and not self._agent_execution_complete()
                    and time.monotonic() < deadline
                ):
                    wait_sec = max(0.0, min(0.1, deadline - time.monotonic()))
                    if wait_sec == 0.0:
                        break
                    self._stop.wait(wait_sec)

    def stop(self) -> None:
        # Close the race where the last workflow detects a response limit just
        # before run_benchmark stops the polling thread.
        if self.abort_event.is_set() and not self._abort_started:
            self._abort(
                "model_response_limit_exceeded",
                "model_response_limit_exceeded:client_parser_limit",
            )
        self._stop.set()
        if self._deadline_timer is not None:
            self._deadline_timer.cancel()
            self._deadline_timer.join(timeout=35.0)
        self._thread.join(timeout=max(10.0, self.request_timeout_sec + 2.0))
        if self._thread.is_alive():
            self.errors.append("metrics sampler did not stop before timeout")


class MeasurementWindowController:
    """Close a fixed-window leg at a trusted wall-clock boundary.

    This controller is intentionally independent from ``LiveMetricsSampler``'s
    abort event: reaching the measurement boundary is an expected
    censoring event, not a candidate or infrastructure failure.

    ``warmup_sec`` keeps the same 192 workflows and audit/cache epoch alive
    before measurement starts.  It must never reset the backend: the point of
    the warmup is to enter the measured interval with a populated KV cache.
    """

    def __init__(
        self,
        duration_sec: int,
        ledger: Ledger,
        model_api_base: str,
        token: str,
        audit_epoch: int,
        warmup_sec: int = 0,
    ) -> None:
        self.duration_sec = duration_sec
        self.warmup_sec = warmup_sec
        self.ledger = ledger
        self.model_api_base = model_api_base
        self.token = token
        self.audit_epoch = audit_epoch
        self.closed = threading.Event()
        self.workload_started_unix: float | None = None
        self.started_unix: float | None = None
        self.cutoff_unix: float | None = None
        self.triggered_unix: float | None = None
        self.cutoff_metrics: dict[str, Any] | None = None
        self.ledger_close: dict[str, Any] | None = None
        self.backend_cancellation: dict[str, Any] | None = None
        self.sealed_unix: float | None = None
        self.errors: list[str] = []
        self._lock = threading.Lock()
        self._stopped = False
        self._deadline_reached = False
        self._timer: threading.Timer | None = None

    def start(self, *, started_unix: float, started_monotonic: float) -> None:
        with self._lock:
            if self._timer is not None or self.started_unix is not None:
                raise RuntimeError("measurement window already started")
            self.workload_started_unix = float(started_unix)
            self.started_unix = float(started_unix) + self.warmup_sec
            self.cutoff_unix = self.started_unix + self.duration_sec
            delay = max(
                0.0,
                started_monotonic
                + self.warmup_sec
                + self.duration_sec
                - time.monotonic(),
            )
            self._timer = threading.Timer(delay, self._expire)
            self._timer.name = "kvbench-fixed-measurement-window"
            self._timer.daemon = True
            self._timer.start()

    def _expire(self) -> None:
        with self._lock:
            if self._stopped or self._deadline_reached:
                return
            if self.cutoff_unix is None:
                self.errors.append("measurement window expired before start")
                return
            self._deadline_reached = True
            self.triggered_unix = time.time()
            cutoff_unix = self.cutoff_unix
        # Signal worker-side checks first, then atomically close the ledger.
        # Ledger.begin/close ordering proves that every racing request is either
        # in the frozen cutoff set or rejected before reaching the gateway.
        self.closed.set()
        try:
            self.ledger_close = self.ledger.close_for_measurement(
                MEASUREMENT_WINDOW_REASON,
                cutoff_unix,
            )
        except Exception as exc:
            self.errors.append(f"ledger_close:{type(exc).__name__}: {exc}")
            return
        try:
            self.backend_cancellation = admin_json(
                "POST",
                self.model_api_base,
                "/admin/measurement/cutoff",
                self.token,
                timeout=30,
                payload={"audit_epoch": self.audit_epoch},
            )
        except Exception as exc:
            self.errors.append(f"backend_cutoff:{type(exc).__name__}: {exc}")
            return
        response = self.backend_cancellation
        if (
            response.get("sealed") is not True
            or response.get("audit_epoch") != self.audit_epoch
            or not isinstance(response.get("metrics"), dict)
            or not isinstance(response.get("sealed_unix"), (int, float))
        ):
            self.errors.append("backend_cutoff_contract_invalid")
            return
        self.sealed_unix = float(response["sealed_unix"])
        self.cutoff_metrics = dict(response["metrics"])
        trigger_lag = max(0.0, float(self.triggered_unix) - cutoff_unix)
        seal_lag = max(0.0, self.sealed_unix - cutoff_unix)
        if trigger_lag > 2.0:
            self.errors.append(f"measurement_trigger_lag_exceeded:{trigger_lag:.6f}>2.0")
        if seal_lag > 2.0:
            self.errors.append(f"backend_seal_lag_exceeded:{seal_lag:.6f}>2.0")

    @property
    def deadline_reached(self) -> bool:
        with self._lock:
            return self._deadline_reached

    def stop(self) -> None:
        with self._lock:
            self._stopped = True
            timer = self._timer
        if timer is not None:
            timer.cancel()
            timer.join(timeout=65.0)
            if timer.is_alive():
                self.errors.append("measurement window timer did not stop")

    def private_snapshot(self) -> dict[str, Any]:
        ledger_close = dict(self.ledger_close or {})
        cutoff_ids = list(ledger_close.pop("request_ids", []))
        backend = self.backend_cancellation or {}
        trigger_lag = (
            max(0.0, float(self.triggered_unix) - float(self.cutoff_unix))
            if self.triggered_unix is not None and self.cutoff_unix is not None
            else None
        )
        seal_lag = (
            max(0.0, float(self.sealed_unix) - float(self.cutoff_unix))
            if self.sealed_unix is not None and self.cutoff_unix is not None
            else None
        )
        return {
            "schema_version": 1,
            "configured": True,
            "duration_sec": self.duration_sec,
            "warmup_duration_sec": self.warmup_sec,
            "workload_started_unix": self.workload_started_unix,
            "started_unix": self.started_unix,
            "cutoff_unix": self.cutoff_unix,
            "triggered_unix": self.triggered_unix,
            "sealed_unix": self.sealed_unix,
            "trigger_lag_sec": trigger_lag,
            "backend_seal_lag_sec": seal_lag,
            "deadline_reached": self.deadline_reached,
            "termination": (
                "trusted_fixed_window" if self.deadline_reached else "workload_exhausted_early"
            ),
            "control_passed": self.deadline_reached and not self.errors,
            "ledger_close": ledger_close,
            "backend_cancelled_requests": int(backend.get("cancelled_requests", 0)),
            "errors": list(self.errors),
            "_trusted_cutoff_request_ids": cutoff_ids,
        }


def wait_for_quiescence(model_api_base: str, token: str, timeout: float = 60) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        quiescent = admin_json("GET", model_api_base, "/admin/quiescent", token)
        if quiescent.get("quiescent") is True:
            return
        time.sleep(0.1)
    raise RuntimeError("backend did not become quiescent")


def reset_measurement_state(model_api_base: str, token: str) -> int:
    reset = admin_json(
        "POST",
        model_api_base,
        "/admin/measurement/reset",
        token,
        timeout=60,
    )
    if reset.get("reset") is not True or reset.get("kv_cache_usage") != 0.0:
        raise RuntimeError(f"atomic measurement reset failed: {reset}")
    audit_epoch = reset.get("audit_epoch")
    if (
        not isinstance(audit_epoch, int)
        or isinstance(audit_epoch, bool)
        or audit_epoch < 0
    ):
        raise RuntimeError(f"atomic measurement reset returned invalid epoch: {reset}")
    return audit_epoch


def reset_audit(model_api_base: str, token: str) -> int:
    wait_for_quiescence(model_api_base, token)
    audit = admin_json("POST", model_api_base, "/admin/audit/reset", token)
    return int(audit["audit_epoch"])


def wait_for_audit_count(model_api_base: str, token: str, expected: int, timeout: float = 10) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    snapshot: dict[str, Any] = {}
    while time.monotonic() < deadline:
        snapshot = admin_json("GET", model_api_base, "/admin/audit/snapshot", token)
        if int(snapshot.get("count", 0)) >= expected:
            return snapshot
        time.sleep(0.1)
    return snapshot


def task_prompt(spec: TaskSpec, context_bytes: int, workload_nonce: str) -> str:
    # SWE-bench fidelity requires the dataset's problem_statement to occupy the
    # upstream {{task}} slot verbatim.  Network isolation is enforced by the
    # sandbox, not by rewriting or padding this candidate-visible prompt.
    if spec.workload_family == "swebench-lite":
        if context_bytes:
            raise ValueError("SWE-bench prompts do not permit synthetic context padding")
        return spec.issue

    context = repository_context(spec.task_id, context_bytes)
    test_guidance = f"Run `{spec.test_command}` (or a narrower relevant test while iterating)."
    prompt = (
        f"Repository issue: {spec.issue}\n\n"
        "Work in the provided offline repository. Inspect the code and make the smallest correct fix. "
        f"{test_guidance} Use one shell command per response. Only after the test passes, finish with a command "
        "whose first output line is `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`."
    )
    if context:
        prompt += f"\n\nPinned repository notes (fixture-only context):\n{context}"
    return prompt


def run_workflow(
    *,
    spec: TaskSpec,
    profile: Profile,
    endpoint: str,
    leg_id: str,
    profile_token: str,
    token: str,
    audit_epoch: int,
    ledger: Ledger,
    ordinal: int,
    workload_nonce: str,
    sandbox: TimedSandbox,
    model_adapter: ModelAdapterContract,
    cancel_event: threading.Event | None = None,
    measurement_window: MeasurementWindowController | None = None,
    agent_execution_tracker: AgentExecutionTracker | None = None,
    program_release_enabled: bool = False,
) -> dict[str, Any]:
    # Candidate-visible tracing values are deliberately opaque.  The trusted
    # result retains mode/profile in its measurement ID, but neither hidden
    # profile names nor corpus namespaces are encoded into gateway headers.
    run_id = hashlib.sha256(f"{leg_id}:{spec.task_id}".encode()).hexdigest()[:32]
    model = BenchmarkVllmModel(
        endpoint=endpoint,
        run_id=run_id,
        profile=profile_token,
        leg_id=leg_id,
        # Preserve prefix reuse across turns in one workflow while preventing
        # independent rollouts (including replicas of the same corpus item)
        # from receiving artificial cross-agent cache hits.
        cache_namespace=workflow_cache_namespace(
            workload_nonce,
            leg_id,
            spec.task_id,
        ),
        seed=profile.seed + ordinal * 1009,
        admin_token=token,
        audit_epoch=audit_epoch,
        ledger=ledger,
        config=model_adapter.model_config(profile.max_tokens),
        request_timeout_sec=profile.request_timeout_sec,
        cancel_event=cancel_event,
    )
    agent = DefaultAgent(
        model,
        sandbox,
        system_template=model_adapter.instruction,
        instance_template=model_adapter.instance_template,
        instruction_role=model_adapter.instruction_role,
        format_error_template=model_adapter.format_error_template,
        step_limit=profile.max_steps,
        cost_limit=0,
        job_timeout=profile.workflow_timeout_sec,
    )
    started = time.time()
    status = "RunnerError"
    message = ""
    error: str | None = None
    tests_passed = False
    test_output = ""
    semantic_evaluated = False
    measurement_truncated = False
    lifecycle_release: dict[str, Any] = {
        "attempted": False,
        "acknowledged": False,
        "supported": None,
        "status": None,
        "error": None,
    }
    release_sent = False
    try:
        if agent_execution_tracker is None:
            # Keep the trusted fixed-window path exactly as before. Its
            # controller, denominator, and public schema are independent from
            # the finite-agent interval introduced below.
            status, message = agent.run(
                task_prompt(spec, profile.context_bytes, workload_nonce)
            )
            agent_finished_unix = time.time()
        else:
            # Prompt construction is not an agent execution. If it fails, the
            # tracker remains incomplete and the finite leg fails closed.
            prompt = task_prompt(spec, profile.context_bytes, workload_nonce)
            try:
                status, message = agent.run(prompt)
            finally:
                agent_finished_monotonic = time.monotonic()
                agent_finished_unix = time.time()
                agent_execution_tracker.mark(
                    ordinal,
                    ended_unix=agent_finished_unix,
                    ended_monotonic=agent_finished_monotonic,
                )
        # The agent has no model request in flight once run() returns.  Release
        # its tracked KV working set before potentially slow sandbox tests.
        lifecycle_release = release_gateway_program(
            endpoint,
            run_id,
            enabled=program_release_enabled,
        )
        release_sent = True
        if measurement_window is None:
            tests_passed, test_output = sandbox.verify()
            semantic_evaluated = True
        elif (
            measurement_window.closed.is_set()
            or measurement_window.cutoff_unix is None
            or agent_finished_unix > measurement_window.cutoff_unix
        ):
            status = "MeasurementWindowEnded"
            message = "trusted public measurement window ended before semantic verification"
            measurement_truncated = True
    except (MeasurementWindowEnded, LedgerClosed) as exc:
        status = "MeasurementWindowEnded"
        message = str(exc)
        measurement_truncated = True
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        message = traceback.format_exc(limit=8)
    finally:
        if not release_sent:
            lifecycle_release = release_gateway_program(
                endpoint,
                run_id,
                enabled=program_release_enabled,
            )
        ended = time.time()
        sandbox.cleanup()

    measurement_start_unix = (
        measurement_window.started_unix if measurement_window is not None else None
    )
    cutoff_unix = measurement_window.cutoff_unix if measurement_window is not None else None

    def in_measurement_window(timestamp: object) -> bool:
        if not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool):
            return False
        observed = float(timestamp)
        return (
            (measurement_start_unix is None or observed >= measurement_start_unix)
            and (cutoff_unix is None or observed <= cutoff_unix)
        )

    def before_measurement(timestamp: object) -> bool:
        return (
            measurement_start_unix is not None
            and isinstance(timestamp, (int, float))
            and not isinstance(timestamp, bool)
            and float(timestamp) < measurement_start_unix
        )

    def after_cutoff(timestamp: object) -> bool:
        return (
            cutoff_unix is not None
            and isinstance(timestamp, (int, float))
            and not isinstance(timestamp, bool)
            and float(timestamp) > cutoff_unix
        )

    executed_timings = [
        timing
        for timing in agent.step_timings
        if "returncode" in timing.get("tool", {})
    ]
    valid_steps = sum(
        in_measurement_window(timing.get("tool", {}).get("end_ts"))
        for timing in executed_timings
    )
    pre_measurement_steps = sum(
        before_measurement(timing.get("tool", {}).get("end_ts"))
        for timing in executed_timings
    )
    post_cutoff_steps = sum(
        after_cutoff(timing.get("tool", {}).get("end_ts"))
        for timing in executed_timings
    )
    step_outcomes: dict[str, int] = {}
    format_error_details: dict[str, int] = {}
    boundary_query_cancellations = 0
    for timing in agent.step_timings:
        tool_timing = timing.get("tool", {})
        query_timing = timing.get("query", {})
        # The vendored loop allocates a timing row before query() checks the
        # step limit.  That terminal sentinel is not a model or action attempt.
        if query_timing.get("start_ts") is None and not tool_timing:
            continue
        # The trusted window closes an active query by disconnecting it.  This
        # terminal row is expected censoring and must not masquerade as an
        # invalid agent action or a candidate query failure.
        if (
            measurement_truncated
            and query_timing.get("start_ts") is not None
            and query_timing.get("last_token_ts") is None
            and not tool_timing
        ):
            boundary_query_cancellations += 1
            continue
        timing_end = tool_timing.get("end_ts", query_timing.get("last_token_ts"))
        if measurement_window is not None and not in_measurement_window(timing_end):
            continue
        if "returncode" in tool_timing:
            outcome = "executed"
        elif isinstance(tool_timing.get("exception"), str):
            outcome = str(tool_timing["exception"])
        elif query_timing.get("last_token_ts") is not None:
            outcome = "unclassified_after_query"
        else:
            outcome = "query_failure"
        step_outcomes[outcome] = step_outcomes.get(outcome, 0) + 1
        if outcome == "FormatError":
            detail = str(tool_timing.get("detail") or "unspecified")
            format_error_details[detail] = format_error_details.get(detail, 0) + 1
    query_latencies = [
        float(timing["query"]["total_s"])
        for timing in agent.step_timings
        if timing.get("query", {}).get("total_s") is not None
        and in_measurement_window(timing.get("query", {}).get("last_token_ts"))
    ]
    tool_latencies = [
        float(timing["tool"]["total_s"])
        for timing in agent.step_timings
        if timing.get("tool", {}).get("total_s") is not None
        and in_measurement_window(timing.get("tool", {}).get("end_ts"))
    ]
    step_latencies = []
    for timing in agent.step_timings:
        query_time = timing.get("query", {}).get("total_s")
        tool_time = timing.get("tool", {}).get("total_s")
        if (
            query_time is not None
            and tool_time is not None
            and in_measurement_window(timing.get("tool", {}).get("end_ts"))
        ):
            step_latencies.append(float(query_time) + float(tool_time))
    return {
        "task_id": spec.task_id,
        "run_id": run_id,
        "status": status,
        "completed": status == "Submitted" and tests_passed,
        "tests_passed": tests_passed,
        "valid_steps": valid_steps,
        "total_executed_steps": len(executed_timings),
        "step_outcomes": step_outcomes,
        "format_error_details": format_error_details,
        "model_calls": model.n_calls,
        "started_unix": started,
        "ended_unix": ended,
        "duration_sec": ended - started,
        "query_latencies_sec": query_latencies,
        "tool_latencies_sec": tool_latencies,
        "step_latencies_sec": step_latencies,
        "error": error,
        "program_lifecycle_release": lifecycle_release,
        "measurement": {
            "schema_version": 1,
            "configured": measurement_window is not None,
            "terminal_reason": (
                "trusted_fixed_window" if measurement_truncated else "natural"
            ),
            "measurement_truncated": measurement_truncated,
            "semantic_evaluated": semantic_evaluated,
            "valid_steps_in_window": valid_steps,
            "pre_measurement_steps": pre_measurement_steps,
            "post_cutoff_steps": post_cutoff_steps,
            "boundary_query_cancellations": boundary_query_cancellations,
        },
        "terminal_message": message[-1000:],
        "test_output": test_output,
    }


def prepare_sandboxes(specs: list[TaskSpec], profile: Profile) -> list[TimedSandbox]:
    """Create the requested clean runtimes with bounded preparation concurrency."""
    if not specs:
        return []
    prepared: list[TimedSandbox | None] = [None] * len(specs)

    def create(index: int) -> None:
        prepared[index] = TimedSandbox(
            specs[index],
            delay_seed=profile.seed,
            minimum_delay=profile.tool_delay_min_sec,
            maximum_delay=profile.tool_delay_max_sec,
            command_timeout_sec=profile.tool_timeout_sec,
        )

    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(profile.sandbox_prepare_concurrency, len(specs))
    ) as pool:
        futures = {pool.submit(create, index): index for index in range(len(specs))}
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                failures.append(f"{specs[futures[future]].task_id}: {type(exc).__name__}: {exc}")
    if failures:
        cleanup_failures = []
        for sandbox in prepared:
            if sandbox is not None:
                try:
                    sandbox.cleanup()
                except Exception as exc:
                    cleanup_failures.append(f"{type(exc).__name__}: {exc}")
        detail = "; ".join(failures[:10])
        if cleanup_failures:
            detail += "; cleanup=" + "; ".join(cleanup_failures[:10])
        raise RuntimeError(f"sandbox preparation failed: {detail}")
    return [sandbox for sandbox in prepared if sandbox is not None]


def cleanup_sandboxes(sandboxes: list[TimedSandbox]) -> None:
    failures = []
    for sandbox in sandboxes:
        try:
            sandbox.cleanup()
        except Exception as exc:
            failures.append(f"{sandbox.spec.task_id}: {type(exc).__name__}: {exc}")
    if failures:
        raise RuntimeError("sandbox cleanup failed: " + "; ".join(failures[:10]))


def _interval_union_seconds(intervals: list[tuple[float, float]]) -> float:
    """Return wall time during which at least one interval was active."""
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    total = 0.0
    current_start, current_end = ordered[0]
    for started, ended in ordered[1:]:
        if started <= current_end:
            current_end = max(current_end, ended)
            continue
        total += current_end - current_start
        current_start, current_end = started, ended
    return total + current_end - current_start


class QueuedSandboxPreparation:
    """Prepare replacement sandboxes without exceeding the configured limit."""

    def __init__(self, profile: Profile) -> None:
        self.profile = profile
        self._semaphore = threading.BoundedSemaphore(profile.sandbox_prepare_concurrency)
        self._lock = threading.Lock()
        self._attempts = 0
        self._prepared = 0
        self._failures = 0
        self._active = 0
        self._maximum_active = 0
        self._queue_waits: list[float] = []
        self._create_durations: list[float] = []
        self._create_intervals: list[tuple[float, float]] = []

    def create(self, spec: TaskSpec) -> TimedSandbox:
        wait_started = time.monotonic()
        with self._semaphore:
            create_started = time.monotonic()
            with self._lock:
                self._attempts += 1
                self._active += 1
                self._maximum_active = max(self._maximum_active, self._active)
                self._queue_waits.append(create_started - wait_started)
            succeeded = False
            try:
                sandbox = TimedSandbox(
                    spec,
                    delay_seed=self.profile.seed,
                    minimum_delay=self.profile.tool_delay_min_sec,
                    maximum_delay=self.profile.tool_delay_max_sec,
                    command_timeout_sec=self.profile.tool_timeout_sec,
                )
                succeeded = True
                return sandbox
            finally:
                create_ended = time.monotonic()
                with self._lock:
                    self._active -= 1
                    self._prepared += int(succeeded)
                    self._failures += int(not succeeded)
                    self._create_durations.append(create_ended - create_started)
                    self._create_intervals.append((create_started, create_ended))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            queue_waits = list(self._queue_waits)
            create_durations = list(self._create_durations)
            create_intervals = list(self._create_intervals)
            attempts = self._attempts
            prepared = self._prepared
            failures = self._failures
            maximum_active = self._maximum_active
        return {
            "attempts": attempts,
            "prepared": prepared,
            "failures": failures,
            "configured_create_concurrency": self.profile.sandbox_prepare_concurrency,
            "maximum_observed_create_concurrency": maximum_active,
            "queue_wait_cumulative_sec": sum(queue_waits),
            "queue_wait_p95_sec": percentile(queue_waits, 0.95),
            "create_cumulative_sec": sum(create_durations),
            "create_active_wall_sec": _interval_union_seconds(create_intervals),
            "create_p50_sec": percentile(create_durations, 0.50),
            "create_p95_sec": percentile(create_durations, 0.95),
            "create_max_sec": max(create_durations, default=None),
            "included_in_measured_interval": True,
        }


def cancelled_workflow_result(spec: TaskSpec, reason: str) -> dict[str, Any]:
    """Return the normal workflow schema without creating a queued sandbox."""

    observed = time.time()
    return {
        "task_id": spec.task_id,
        "run_id": None,
        "status": "RunnerError",
        "completed": False,
        "tests_passed": False,
        "valid_steps": 0,
        "step_outcomes": {"query_failure": 1},
        "format_error_details": {},
        "model_calls": 0,
        "started_unix": observed,
        "ended_unix": observed,
        "duration_sec": 0.0,
        "query_latencies_sec": [],
        "tool_latencies_sec": [],
        "step_latencies_sec": [],
        "error": f"RequestCancelled: {reason}",
        "terminal_message": "",
        "test_output": "",
    }


def window_ended_workflow_result(spec: TaskSpec) -> dict[str, Any]:
    """Return an expected, non-error result for a cutoff-before-start race."""

    observed = time.time()
    return {
        "task_id": spec.task_id,
        "run_id": None,
        "status": "MeasurementWindowEnded",
        "completed": False,
        "tests_passed": False,
        "valid_steps": 0,
        "total_executed_steps": 0,
        "step_outcomes": {},
        "format_error_details": {},
        "model_calls": 0,
        "started_unix": observed,
        "ended_unix": observed,
        "duration_sec": 0.0,
        "query_latencies_sec": [],
        "tool_latencies_sec": [],
        "step_latencies_sec": [],
        "error": None,
        "program_lifecycle_release": {
            "attempted": False,
            "acknowledged": False,
            "supported": None,
            "status": None,
            "error": None,
        },
        "measurement": {
            "schema_version": 1,
            "configured": True,
            "terminal_reason": "trusted_fixed_window",
            "measurement_truncated": True,
            "semantic_evaluated": False,
            "valid_steps_in_window": 0,
            "pre_measurement_steps": 0,
            "post_cutoff_steps": 0,
            "boundary_query_cancellations": 0,
        },
        "terminal_message": "trusted measurement window ended before workflow start",
        "test_output": "",
    }


def run_workflow_pool(
    *,
    specs: list[TaskSpec],
    profile: Profile,
    initial_sandboxes: list[TimedSandbox],
    endpoint: str,
    leg_id: str,
    profile_token: str,
    token: str,
    audit_epoch: int,
    ledger: Ledger,
    workload_nonce: str,
    model_adapter: ModelAdapterContract,
    cancel_event: threading.Event | None,
    measurement_window: MeasurementWindowController | None = None,
    agent_execution_tracker: AgentExecutionTracker | None = None,
    program_release_enabled: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run a fixed-width worker pool with at most one live sandbox per slot.

    The first worker wave receives sandboxes prepared before measurement.  A
    queued workflow prepares its sandbox inside the worker slot only after the
    preceding workflow has returned (and therefore cleaned its sandbox).
    """
    initial_count = min(len(specs), profile.concurrency)
    if len(initial_sandboxes) != initial_count:
        raise ValueError(
            "initial sandbox count must equal min(workflows, concurrency): "
            f"{len(initial_sandboxes)} != {initial_count}"
        )

    queued_preparation = QueuedSandboxPreparation(profile)
    created_sandboxes = list(initial_sandboxes)
    created_lock = threading.Lock()

    def one(index: int) -> dict[str, Any]:
        if measurement_window is not None and measurement_window.closed.is_set():
            return window_ended_workflow_result(specs[index])
        if cancel_event is not None and cancel_event.is_set():
            return cancelled_workflow_result(
                specs[index], "candidate leg cancelled before workflow start"
            )
        if index < initial_count:
            sandbox = initial_sandboxes[index]
        else:
            sandbox = queued_preparation.create(specs[index])
            with created_lock:
                created_sandboxes.append(sandbox)
        try:
            if measurement_window is not None and measurement_window.closed.is_set():
                return window_ended_workflow_result(specs[index])
            if cancel_event is not None and cancel_event.is_set():
                return cancelled_workflow_result(
                    specs[index], "candidate leg cancelled before model query"
                )
            return run_workflow(
                spec=specs[index],
                profile=profile,
                endpoint=endpoint,
                leg_id=leg_id,
                profile_token=profile_token,
                token=token,
                audit_epoch=audit_epoch,
                ledger=ledger,
                ordinal=index,
                workload_nonce=workload_nonce,
                sandbox=sandbox,
                model_adapter=model_adapter,
                cancel_event=cancel_event,
                measurement_window=measurement_window,
                agent_execution_tracker=agent_execution_tracker,
                program_release_enabled=program_release_enabled,
            )
        finally:
            # run_workflow normally owns this cleanup.  Retrying here also
            # covers setup/runner failures before its normal finally block.
            sandbox.cleanup()

    workflows: list[dict[str, Any]] = []
    try:
        # Submitting the full logical queue preserves workflow ordinals, seeds,
        # and cache namespaces.  Only `concurrency` functions can execute, so a
        # queued sandbox cannot be created before a worker slot is released.
        with concurrent.futures.ThreadPoolExecutor(max_workers=profile.concurrency) as pool:
            futures = [pool.submit(one, index) for index in range(len(specs))]
            for future in concurrent.futures.as_completed(
                futures,
                timeout=(
                    (
                        measurement_window.warmup_sec
                        + measurement_window.duration_sec
                        if measurement_window is not None
                        else math.ceil(profile.workflows / profile.concurrency)
                        * profile.workflow_timeout_sec
                    )
                    # The independent candidate wall timer has already
                    # cancelled model sockets at the wave budget. Keep only a
                    # bounded teardown allowance here for an in-progress
                    # 180-second sandbox create or tool/cleanup RPC; without
                    # it as_completed could turn an already classified
                    # candidate timeout into an unscored controller 500.
                    + 300
                ),
            ):
                workflows.append(future.result())
    finally:
        # Keep references to dynamically created sandboxes and retry every
        # cleanup after all workers stop.  TimedSandbox.cleanup is idempotent.
        cleanup_sandboxes(created_sandboxes)
    return workflows, queued_preparation.snapshot()


def verify_audit(client_records: list[dict[str, Any]], audit_records: list[dict[str, Any]]) -> dict[str, Any]:
    audit_by_id: dict[str, list[dict[str, Any]]] = {}
    for record in audit_records:
        audit_by_id.setdefault(str(record.get("request_id")), []).append(record)
    mismatches = []
    for client in client_records:
        matches = audit_by_id.get(client["request_id"], [])
        if len(matches) != 1:
            mismatches.append({"request_id": client["request_id"], "reason": f"audit_count={len(matches)}"})
            continue
        audit = matches[0]
        for key in (
            "audit_epoch",
            "run_id",
            "profile",
            "leg_id",
            "cache_namespace",
            "canonical_body_digest",
            "response_digest",
            "status",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "sse_done",
        ):
            if audit.get(key) != client.get(key):
                mismatches.append(
                    {"request_id": client["request_id"], "reason": key, "client": client.get(key), "audit": audit.get(key)}
                )
        if audit.get("benchmark_auth_valid") is not True:
            mismatches.append({"request_id": client["request_id"], "reason": "benchmark_auth_invalid"})
        if client.get("status") == 200 and (
            audit.get("client_disconnected") is not False
            or audit.get("error") is not None
            or audit.get("sse_done") is not True
        ):
            mismatches.append(
                {
                    "request_id": client["request_id"],
                    "reason": "backend_stream_not_complete",
                    "client_disconnected": audit.get("client_disconnected"),
                    "backend_error": audit.get("error"),
                    "sse_done": audit.get("sse_done"),
                }
            )
    extra = max(0, len(audit_records) - len(client_records))
    missing = max(0, len(client_records) - len(audit_records))
    waits = []
    for client in client_records:
        matches = audit_by_id.get(client["request_id"], [])
        if len(matches) == 1:
            waits.append(max(0.0, float(matches[0]["started_unix"]) - float(client["started_unix"])))
    passed = not mismatches and extra == 0 and missing == 0
    return {
        "passed": passed,
        "client_requests": len(client_records),
        "audited_backend_requests": len(audit_records),
        "extra_backend_requests": extra,
        "missing_backend_requests": missing,
        "p95_gateway_wait_sec": percentile(waits, 0.95),
        "p99_gateway_wait_sec": percentile(waits, 0.99),
        "max_gateway_wait_sec": max(waits) if waits else None,
        "mismatches": mismatches[:20],
    }


def verify_window_audit(
    client_records: list[dict[str, Any]],
    audit_records: list[dict[str, Any]],
    trusted_cutoff_request_ids: list[str],
) -> dict[str, Any]:
    """Verify strict completed traffic and trusted boundary-censored traffic.

    Only request IDs atomically frozen by ``Ledger.close_for_measurement`` may
    have no backend record (the request was still queued inside the candidate
    gateway) or a partial backend record (the trusted cutoff disconnected it).
    Every other request retains the full one-to-one response audit.
    """

    cutoff_ids = {str(value) for value in trusted_cutoff_request_ids}
    client_by_id = {str(record.get("request_id")): record for record in client_records}
    audit_by_id: dict[str, list[dict[str, Any]]] = {}
    for record in audit_records:
        audit_by_id.setdefault(str(record.get("request_id")), []).append(record)
    mismatches: list[dict[str, Any]] = []
    if len(client_by_id) != len(client_records):
        mismatches.append({"reason": "duplicate_client_request_id"})
    unknown_cutoff = sorted(cutoff_ids - set(client_by_id))
    if unknown_cutoff:
        mismatches.append(
            {"reason": "trusted_cutoff_id_missing_client_record", "count": len(unknown_cutoff)}
        )
    unknown_backend = sorted(set(audit_by_id) - set(client_by_id))
    if unknown_backend:
        mismatches.append(
            {"reason": "unknown_extra_backend_request", "count": len(unknown_backend)}
        )

    strict_keys = (
        "audit_epoch",
        "run_id",
        "profile",
        "leg_id",
        "cache_namespace",
        "canonical_body_digest",
        "response_digest",
        "status",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "sse_done",
    )
    cutoff_identity_keys = (
        "audit_epoch",
        "run_id",
        "profile",
        "leg_id",
        "cache_namespace",
        "canonical_body_digest",
    )
    waits: list[float] = []
    cutoff_admitted = 0
    cutoff_unadmitted = 0
    for request_id, client in client_by_id.items():
        matches = audit_by_id.get(request_id, [])
        if request_id in cutoff_ids:
            if len(matches) > 1:
                mismatches.append(
                    {"request_id": request_id, "reason": f"cutoff_audit_count={len(matches)}"}
                )
                continue
            if not matches:
                cutoff_unadmitted += 1
                continue
            cutoff_admitted += 1
            audit = matches[0]
            for key in cutoff_identity_keys:
                if audit.get(key) != client.get(key):
                    mismatches.append(
                        {
                            "request_id": request_id,
                            "reason": f"cutoff_{key}",
                            "client": client.get(key),
                            "audit": audit.get(key),
                        }
                    )
            if audit.get("benchmark_auth_valid") is not True:
                mismatches.append(
                    {"request_id": request_id, "reason": "cutoff_benchmark_auth_invalid"}
                )
            waits.append(
                max(0.0, float(audit["started_unix"]) - float(client["started_unix"]))
            )
            continue

        if len(matches) != 1:
            mismatches.append(
                {"request_id": request_id, "reason": f"audit_count={len(matches)}"}
            )
            continue
        audit = matches[0]
        for key in strict_keys:
            if audit.get(key) != client.get(key):
                mismatches.append(
                    {
                        "request_id": request_id,
                        "reason": key,
                        "client": client.get(key),
                        "audit": audit.get(key),
                    }
                )
        if audit.get("benchmark_auth_valid") is not True:
            mismatches.append({"request_id": request_id, "reason": "benchmark_auth_invalid"})
        if client.get("status") == 200 and (
            audit.get("client_disconnected") is not False
            or audit.get("error") is not None
            or audit.get("sse_done") is not True
        ):
            mismatches.append(
                {
                    "request_id": request_id,
                    "reason": "backend_stream_not_complete",
                    "client_disconnected": audit.get("client_disconnected"),
                    "backend_error": audit.get("error"),
                    "sse_done": audit.get("sse_done"),
                }
            )
        waits.append(
            max(0.0, float(audit["started_unix"]) - float(client["started_unix"]))
        )

    return {
        "passed": not mismatches,
        "window_censored": True,
        "client_requests": len(client_records),
        "audited_backend_requests": len(audit_records),
        "strict_client_requests": len(client_records) - len(cutoff_ids),
        "cutoff_client_requests": len(cutoff_ids),
        "cutoff_backend_admitted": cutoff_admitted,
        "cutoff_backend_unadmitted": cutoff_unadmitted,
        "extra_backend_requests": len(unknown_backend),
        "missing_backend_requests": cutoff_unadmitted,
        "p95_gateway_wait_sec": percentile(waits, 0.95),
        "p99_gateway_wait_sec": percentile(waits, 0.99),
        "max_gateway_wait_sec": max(waits) if waits else None,
        "mismatches": mismatches[:20],
    }


def request_token_distribution(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [
        int(record[key])
        for record in records
        if isinstance(record.get(key), (int, float)) and not isinstance(record.get(key), bool)
    ]
    return {
        "sample_count": len(values),
        "total": sum(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def request_depth_trajectory(
    records: list[dict[str, Any]],
    workflow_count: int,
) -> list[dict[str, Any]]:
    """Summarize real context growth and cohort survival at every agent turn."""

    by_index: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        index = record.get("call_index")
        if isinstance(index, int) and not isinstance(index, bool) and index >= 1:
            by_index.setdefault(index, []).append(record)
    return [
        {
            "call_index": index,
            "workflows_reached": len(rows),
            "survival_ratio": len(rows) / workflow_count if workflow_count else 0.0,
            "prompt_tokens": request_token_distribution(rows, "prompt_tokens"),
            "completion_tokens": request_token_distribution(rows, "completion_tokens"),
            "total_tokens": request_token_distribution(rows, "total_tokens"),
        }
        for index, rows in sorted(by_index.items())
    ]


def model_response_outcomes(records: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
    finish_reasons: dict[str, int] = {}
    tool_call_counts: dict[str, int] = {}
    with_tool_call = 0
    completion_budget_exhausted = 0
    no_tool_and_budget_exhausted = 0
    for record in records:
        reason = record.get("finish_reason")
        reason_key = reason if isinstance(reason, str) and reason else "missing"
        finish_reasons[reason_key] = finish_reasons.get(reason_key, 0) + 1
        raw_tool_count = record.get("tool_call_count")
        tool_count = int(raw_tool_count) if isinstance(raw_tool_count, int) else 0
        tool_call_counts[str(tool_count)] = tool_call_counts.get(str(tool_count), 0) + 1
        if tool_count > 0:
            with_tool_call += 1
        raw_completion_tokens = record.get("completion_tokens")
        exhausted = (
            isinstance(raw_completion_tokens, int)
            and not isinstance(raw_completion_tokens, bool)
            and raw_completion_tokens >= max_tokens
        )
        if exhausted:
            completion_budget_exhausted += 1
            if tool_count == 0:
                no_tool_and_budget_exhausted += 1
    return {
        "response_count": len(records),
        "finish_reasons": finish_reasons,
        "tool_call_count_distribution": tool_call_counts,
        "responses_with_tool_call": with_tool_call,
        "responses_without_tool_call": len(records) - with_tool_call,
        "completion_budget_exhausted": completion_budget_exhausted,
        "no_tool_and_budget_exhausted": no_tool_and_budget_exhausted,
        "content_chars_total": sum(int(record.get("content_chars") or 0) for record in records),
        "reasoning_content_chars_total": sum(
            int(record.get("reasoning_content_chars") or 0) for record in records
        ),
        "internal_reasoning_chars_total": sum(
            int(record.get("internal_reasoning_chars") or 0) for record in records
        ),
    }


def model_protocol_diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    tool_call_names: dict[str, int] = {}
    tool_argument_shapes: dict[str, int] = {}
    for record in records:
        for name in record.get("tool_call_names", []):
            key = str(name)
            tool_call_names[key] = tool_call_names.get(key, 0) + 1
        for shape in record.get("tool_argument_shapes", []):
            key = str(shape)
            tool_argument_shapes[key] = tool_argument_shapes.get(key, 0) + 1
    return {
        "tool_call_names": tool_call_names,
        "tool_argument_shapes": tool_argument_shapes,
    }


def model_response_limit_failure(
    records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    limited = [
        record for record in records if isinstance(record.get("response_limit"), dict)
    ]
    if not limited:
        return None
    kinds = sorted(
        {str(record["response_limit"].get("kind")) for record in limited}
    )
    return {
        "type": "model_response_limit_exceeded",
        "detail": (
            "model_response_limit_exceeded:"
            f"request_count={len(limited)},"
            f"kinds={','.join(kinds)}"
        ),
        "cancelled": True,
        "request_ids": [str(record.get("request_id")) for record in limited[:10]],
    }


def effective_scheduling_progress(workflow: dict[str, Any], max_steps: int) -> int:
    """Return progress for starvation/completion gates, not throughput.

    ``Submitted`` workflows have no remaining work to schedule, even when the
    agent finishes before consuming its whole step budget, so count them as
    fully scheduled.  Otherwise model-call attempts measure the scheduling
    opportunities a workflow received; this naturally puts ``LimitsExceeded``
    at the cap while preserving partial progress for abnormal or unfinished
    workflows.  Older records without ``model_calls`` fall back to valid action
    count.  Aggregate throughput continues to use unmodified ``valid_steps``.
    """

    if workflow.get("status") == "Submitted":
        return max_steps
    opportunities = workflow.get("model_calls")
    if opportunities is None:
        opportunities = workflow["valid_steps"]
    return max(0, min(int(opportunities), max_steps))


def append_warmup_response(
    messages: list[dict[str, Any]],
    response: dict[str, Any],
    observation: str,
    model_adapter: ModelAdapterContract,
) -> bool:
    """Append one warm-up response and validate the Qwen3.6 text-bash protocol."""
    messages.append({"role": "assistant", **response})
    try:
        extract_bash_action(response)
    except ValueError:
        messages.append(
            {"role": "user", "content": model_adapter.render_format_error([])}
        )
        return False
    messages.append({"role": "user", "content": observation})
    return True


def run_warmup(
    *,
    profile: Profile,
    endpoint: str,
    leg_id: str,
    profile_token: str,
    workload_nonce: str,
    token: str,
    audit_epoch: int,
    model_adapter: ModelAdapterContract,
    program_release_enabled: bool = False,
) -> tuple[Ledger, dict[str, int]]:
    ledger = Ledger()
    # Warm-up traffic is candidate-visible.  Never expose held-out problem
    # statements before a hidden leg; a fixed public slice exercises the same
    # Qwen3.6 mini-SWE text-bash protocol and is discarded before measurement.
    warmup_partition = (
        "public" if profile.workload_family == "swebench-lite" else profile.corpus_partition
    )
    specs = make_specs(
        f"warmup-{profile.name}",
        profile.warmup_requests,
        workload_family=profile.workload_family,
        corpus_partition=warmup_partition,
        seed=WARMUP_CORPUS_SEED,
    )
    def one(index: int) -> dict[str, int]:
        run_id = hashlib.sha256(f"{leg_id}:warmup:{index}".encode()).hexdigest()[:32]
        model = BenchmarkVllmModel(
            endpoint=endpoint,
            run_id=run_id,
            profile=profile_token,
            leg_id=leg_id,
            cache_namespace=workflow_cache_namespace(
                workload_nonce,
                leg_id,
                f"warmup:{index}",
            ),
            seed=WARMUP_MODEL_SEED_BASE + index,
            admin_token=token,
            audit_epoch=audit_epoch,
            ledger=ledger,
            config=model_adapter.model_config(profile.max_tokens),
            request_timeout_sec=profile.request_timeout_sec,
        )
        messages: list[dict[str, Any]] = [
            {"role": model_adapter.instruction_role, "content": model_adapter.instruction},
            {
                "role": "user",
                "content": model_adapter.render_instance(
                    task_prompt(
                        specs[index],
                        profile.warmup_context_bytes,
                        workload_nonce,
                    )
                ),
            },
        ]
        valid_steps = 0
        attempts = 0
        max_attempts = profile.warmup_steps + 3
        lifecycle_release: dict[str, Any]
        try:
            while valid_steps < profile.warmup_steps and attempts < max_attempts:
                response = model.query(messages)
                attempts += 1
                observation = (
                    f"Observation: warm-up tool step {valid_steps + 1} completed successfully."
                )
                if append_warmup_response(
                    messages,
                    response,
                    observation,
                    model_adapter,
                ):
                    valid_steps += 1
            if valid_steps != profile.warmup_steps:
                raise RuntimeError(
                    "warm-up protocol did not recover within its bounded correction budget: "
                    f"valid_steps={valid_steps}/{profile.warmup_steps}, attempts={attempts}/{max_attempts}"
                )
        finally:
            lifecycle_release = release_gateway_program(
                endpoint,
                run_id,
                enabled=program_release_enabled,
            )
        return {
            "attempts": attempts,
            "valid_steps": valid_steps,
            "format_corrections": attempts - valid_steps,
            "release_attempted": int(lifecycle_release["attempted"]),
            "release_acknowledged": int(lifecycle_release["acknowledged"]),
            "release_failed": int(
                lifecycle_release["attempted"]
                and lifecycle_release["supported"] is not False
                and not lifecycle_release["acknowledged"]
            ),
        }

    outcomes: list[dict[str, int]] = []
    if profile.warmup_requests:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(profile.warmup_requests, 4)) as pool:
            outcomes = list(pool.map(one, range(profile.warmup_requests)))
    return ledger, {
        "request_count": sum(item["attempts"] for item in outcomes),
        "required_valid_actions": profile.warmup_requests * profile.warmup_steps,
        "valid_actions": sum(item["valid_steps"] for item in outcomes),
        "format_corrections": sum(item["format_corrections"] for item in outcomes),
        "program_release_attempted": sum(
            item["release_attempted"] for item in outcomes
        ),
        "program_release_acknowledged": sum(
            item["release_acknowledged"] for item in outcomes
        ),
        "program_release_failed": sum(item["release_failed"] for item in outcomes),
        "maximum_attempts_per_warmup": max(
            (item["attempts"] for item in outcomes),
            default=0,
        ),
    }


def run_benchmark(
    *,
    profile: Profile,
    endpoint: str,
    mode: str,
    model_api_base: str,
    admin_token: str,
    corpus_namespace: str = "public",
    workload_nonce: str | None = None,
) -> dict[str, Any]:
    if not profile.enabled:
        raise RuntimeError(f"profile {profile.name} is not enabled")
    measurement_id = f"{mode}-{profile.name}-{uuid.uuid4().hex[:12]}"
    workload_nonce = workload_nonce or uuid.uuid4().hex
    leg_id = uuid.uuid4().hex
    profile_token = uuid.uuid4().hex[:24]
    model_adapter = load_model_adapter_contract()
    effective_max_tokens = model_adapter.effective_max_tokens(profile.max_tokens)
    warmup_audit_epoch = reset_measurement_state(model_api_base, admin_token)
    warmup_ledger, warmup_protocol = run_warmup(
        profile=profile,
        endpoint=endpoint,
        leg_id=leg_id,
        profile_token=profile_token,
        workload_nonce=workload_nonce,
        token=admin_token,
        audit_epoch=warmup_audit_epoch,
        model_adapter=model_adapter,
        program_release_enabled=mode == "candidate",
    )
    wait_for_quiescence(model_api_base, admin_token)
    warmup_records = warmup_ledger.snapshot()
    warmup_snapshot = wait_for_audit_count(model_api_base, admin_token, len(warmup_records))
    warmup_audit = verify_audit(warmup_records, warmup_snapshot.get("records", []))
    if not warmup_audit["passed"]:
        raise RuntimeError(f"warmup audit failed: {warmup_audit}")
    specs = make_specs(
        corpus_namespace,
        profile.workflows,
        workload_family=profile.workload_family,
        corpus_partition=profile.corpus_partition,
        seed=profile.seed,
    )
    initial_sandbox_count = min(profile.workflows, profile.concurrency)
    preparation_started = time.time()
    sandboxes = prepare_sandboxes(specs[:initial_sandbox_count], profile)
    preparation_ended = time.time()
    workflows: list[dict[str, Any]] = []
    queued_sandbox_preparation: dict[str, Any] | None = None
    sampler: LiveMetricsSampler | None = None
    measurement_window: MeasurementWindowController | None = None
    agent_execution_tracker: AgentExecutionTracker | None = None
    try:
        # Warm-up intentionally exercises the real Qwen3.6 protocol, but none of
        # its prefix blocks or audit records may leak into the measured leg.
        audit_epoch = reset_measurement_state(model_api_base, admin_token)
        metrics_before = admin_json("GET", model_api_base, "/admin/metrics/snapshot", admin_token)
        ledger = Ledger()
        if profile.measurement_window_sec is None:
            agent_execution_tracker = AgentExecutionTracker(profile.workflows)
        sampler = LiveMetricsSampler(
            model_api_base,
            admin_token,
            profile.metrics_sample_interval_sec,
            profile.metrics_request_timeout_sec,
            profile.backend_stall_timeout_sec,
            mode=mode,
            ledger=ledger,
            candidate_stall_timeout_sec=profile.candidate_admission_stall_timeout_sec,
            candidate_leg_wall_timeout_sec=(
                candidate_leg_wall_timeout_sec(profile)
                if mode == "candidate" and profile.measurement_window_sec is None
                else None
            ),
            expected_audit_epoch=audit_epoch,
            agent_execution_complete_event=(
                agent_execution_tracker.complete
                if agent_execution_tracker is not None
                else None
            ),
        )
        started = time.time()
        started_monotonic = time.monotonic()
        if profile.measurement_window_sec is not None:
            measurement_window = MeasurementWindowController(
                profile.measurement_window_sec,
                ledger,
                model_api_base,
                admin_token,
                audit_epoch,
                warmup_sec=profile.measurement_warmup_sec,
            )
        sampler.start()
        if measurement_window is not None:
            measurement_window.start(
                started_unix=started,
                started_monotonic=started_monotonic,
            )
        workflows, queued_sandbox_preparation = run_workflow_pool(
            specs=specs,
            profile=profile,
            initial_sandboxes=sandboxes,
            endpoint=endpoint,
            leg_id=leg_id,
            profile_token=profile_token,
            token=admin_token,
            audit_epoch=audit_epoch,
            ledger=ledger,
            workload_nonce=workload_nonce,
            model_adapter=model_adapter,
            cancel_event=sampler.abort_event,
            measurement_window=measurement_window,
            agent_execution_tracker=agent_execution_tracker,
            program_release_enabled=mode == "candidate",
        )
    finally:
        if measurement_window is not None:
            measurement_window.stop()
        if sampler is not None:
            sampler.stop()
        cleanup_sandboxes(sandboxes)
        workload_ended = time.time()
    if sampler is None:
        raise RuntimeError("metrics sampler was not initialized")
    if queued_sandbox_preparation is None:
        raise RuntimeError("queued sandbox preparation report was not initialized")
    agent_execution_interval = (
        agent_execution_tracker.snapshot(
            started_unix=started,
            started_monotonic=started_monotonic,
        )
        if agent_execution_tracker is not None
        else None
    )
    measurement_window_report = (
        measurement_window.private_snapshot()
        if measurement_window is not None
        else {
            "schema_version": 1,
            "configured": False,
            "deadline_reached": False,
            "termination": "finite_workload",
            "control_passed": True,
            "_trusted_cutoff_request_ids": [],
        }
    )
    workflows.sort(key=lambda item: item["task_id"])
    client_records = ledger.snapshot()
    wait_for_quiescence(model_api_base, admin_token)
    if measurement_window_report["deadline_reached"]:
        audit_snapshot = admin_json(
            "GET", model_api_base, "/admin/audit/snapshot", admin_token
        )
    else:
        audit_snapshot = wait_for_audit_count(
            model_api_base, admin_token, len(client_records)
        )
    if audit_snapshot.get("audit_epoch") != audit_epoch:
        raise RuntimeError("audit epoch changed during benchmark")
    trusted_cutoff_ids = measurement_window_report["_trusted_cutoff_request_ids"]
    if measurement_window_report["deadline_reached"]:
        audit = verify_window_audit(
            client_records,
            audit_snapshot.get("records", []),
            trusted_cutoff_ids,
        )
    else:
        audit = verify_audit(client_records, audit_snapshot.get("records", []))
    metrics_after = (
        measurement_window.cutoff_metrics
        if measurement_window is not None
        and measurement_window.deadline_reached
        and measurement_window.cutoff_metrics is not None
        else admin_json("GET", model_api_base, "/admin/metrics/snapshot", admin_token)
    )
    if measurement_window_report["deadline_reached"]:
        measurement_end_unix = float(measurement_window_report["cutoff_unix"])
        measurement_start_unix = float(measurement_window_report["started_unix"])
        # Fixed-window pressure is summarized only after the optional warmup,
        # through the inclusive trusted cutoff.
        metric_points = [
            point
            for point in sampler.points
            if (
                isinstance(point.get("created_unix"), (int, float))
                and not isinstance(point.get("created_unix"), bool)
                and float(point["created_unix"]) >= measurement_start_unix
                and float(point["created_unix"]) <= measurement_end_unix
            )
        ]
    elif measurement_window is not None:
        # Preserve the pre-existing invalid-public diagnostic when a fixed
        # workload exhausts before its trusted deadline. The window control
        # gate rejects it, but its private/public diagnostic schema remains
        # unchanged instead of being mistaken for a finite leg.
        measurement_end_unix = workload_ended
        measurement_start_unix = float(
            measurement_window_report.get("started_unix") or started
        )
        metric_points = list(sampler.points)
    else:
        if agent_execution_interval is None:
            raise RuntimeError("finite workload is missing its agent execution interval")
        measurement_end_unix = float(agent_execution_interval["ended_unix"])
        measurement_start_unix = started
        metric_points = finite_metric_points_through(
            sampler.points,
            measurement_end_unix,
        )
    live_metrics = summarize_metric_series(
        metric_points,
        pressure_threshold=profile.pressure_kv_threshold,
        expected_sample_interval_sec=profile.metrics_sample_interval_sec,
        sampler_errors=sampler.errors,
        request_intervals=audit_snapshot.get("records", []),
    )
    metric_delta = summarize_metric_delta(metrics_before, metrics_after)
    wall_ended = time.time()
    wall_ended_monotonic = (
        time.monotonic() if agent_execution_interval is not None else None
    )
    # Only direct-backend legs prove that the configured model/workload pair is
    # capable of creating pressure.  A successful candidate is expected to
    # reduce occupancy, so applying the same pressure gate to it would punish
    # the optimization being measured.
    pressure = assess_pressure(
        live_metrics,
        required=profile.pressure_required and mode == "baseline",
        minimum_samples=profile.pressure_min_samples,
        minimum_kv_p95=profile.pressure_kv_p95_min,
        minimum_time_fraction=profile.pressure_time_fraction_min,
        counter_diagnostics=metric_delta.get("derived", {}),
    )

    cutoff_id_set = {str(value) for value in trusted_cutoff_ids}
    measured_client_records = [
        record
        for record in client_records
        if str(record.get("request_id")) not in cutoff_id_set
        and (
            not measurement_window_report["configured"]
            or (
                isinstance(record.get("ended_unix"), (int, float))
                and not isinstance(record.get("ended_unix"), bool)
                and float(record["ended_unix"]) >= measurement_start_unix
            )
        )
    ]
    valid_steps = sum(item["valid_steps"] for item in workflows)
    total_executed_steps = sum(
        int(item.get("total_executed_steps", item["valid_steps"]))
        for item in workflows
    )
    completed = sum(bool(item["completed"]) for item in workflows)
    tests_passed = sum(bool(item["tests_passed"]) for item in workflows)
    semantic_evaluated = sum(
        bool((item.get("measurement") or {}).get("semantic_evaluated", True))
        for item in workflows
    )
    lifecycle_releases = [
        item.get("program_lifecycle_release") or {}
        for item in workflows
        if (item.get("program_lifecycle_release") or {}).get("attempted")
    ]
    if agent_execution_interval is None:
        elapsed = max(measurement_end_unix - measurement_start_unix, 1e-9)
        wall_elapsed = max(wall_ended - measurement_start_unix, 1e-9)
    else:
        elapsed = max(float(agent_execution_interval["elapsed_sec"]), 1e-9)
        if wall_ended_monotonic is None:  # guarded by the branch above
            raise RuntimeError("finite workload is missing its monotonic wall end")
        wall_elapsed = max(wall_ended_monotonic - started_monotonic, 1e-9)
    step_latencies = [value for item in workflows for value in item["step_latencies_sec"]]
    query_latencies = [value for item in workflows for value in item["query_latencies_sec"]]
    tool_latencies = [value for item in workflows for value in item["tool_latencies_sec"]]
    total_query_sec = sum(query_latencies)
    total_tool_sec = sum(tool_latencies)
    successful_requests = sum(
        record["status"] == 200 and record["error"] is None
        for record in measured_client_records
    )
    ttft = [
        float(record["first_token_unix"]) - float(record["started_unix"])
        for record in measured_client_records
        if record.get("first_token_unix") is not None
    ]
    prompt_tokens = sum(
        int(record.get("prompt_tokens") or 0) for record in measured_client_records
    )
    completion_tokens = sum(
        int(record.get("completion_tokens") or 0)
        for record in measured_client_records
    )
    token_distribution = {
        key: request_token_distribution(measured_client_records, key)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    workflow_model_calls = [
        int(item["model_calls"])
        for item in workflows
        if isinstance(item.get("model_calls"), int)
        and not isinstance(item.get("model_calls"), bool)
    ]
    infrastructure_failure = None
    candidate_failure = None
    if measurement_window_report["configured"] and not measurement_window_report["control_passed"]:
        infrastructure_failure = {
            "type": "measurement_window_control_failed",
            "detail": ",".join(measurement_window_report.get("errors", []))
            or str(measurement_window_report.get("termination")),
            "cancelled": bool(measurement_window_report["deadline_reached"]),
        }
    if sampler.fatal_error is not None:
        failure = {
            "type": sampler.fatal_type or "workload_watchdog_failure",
            "detail": sampler.fatal_error,
            "cancelled": True,
            "client_cancellation": sampler.client_cancellation,
        }
        if mode == "candidate" and sampler.fatal_type in {
            "candidate_gateway_admission_stall",
            "candidate_leg_wall_timeout",
            "model_response_limit_exceeded",
        }:
            candidate_failure = failure
        else:
            infrastructure_failure = infrastructure_failure or failure
    response_limit_failure = model_response_limit_failure(client_records)
    if response_limit_failure is not None:
        if mode == "candidate":
            candidate_failure = candidate_failure or response_limit_failure
        else:
            infrastructure_failure = infrastructure_failure or response_limit_failure
    scheduling_progress = (
        [int(item["valid_steps"]) for item in workflows]
        if measurement_window_report["configured"]
        else [effective_scheduling_progress(item, profile.max_steps) for item in workflows]
    )
    zero_progress_workflows = sum(progress == 0 for progress in scheduling_progress)
    agent_step_outcomes: dict[str, int] = {}
    format_error_details: dict[str, int] = {}
    for workflow in workflows:
        for outcome, count in workflow.get("step_outcomes", {}).items():
            agent_step_outcomes[outcome] = agent_step_outcomes.get(outcome, 0) + int(count)
        for detail, count in workflow.get("format_error_details", {}).items():
            format_error_details[detail] = format_error_details.get(detail, 0) + int(count)
    profile_config = asdict(profile)
    if effective_max_tokens != profile.max_tokens:
        profile_config["configured_max_tokens"] = profile.max_tokens
        profile_config["max_tokens"] = effective_max_tokens
    model_adapter_summary = model_adapter.summary()
    model_adapter_summary["effective_max_tokens"] = effective_max_tokens
    measurement_window_report.update(
        {
            "measurement_elapsed_sec": elapsed,
            "wall_elapsed_sec": wall_elapsed,
            "teardown_sec": max(0.0, wall_elapsed - elapsed),
            "total_executed_steps": total_executed_steps,
            "pre_measurement_steps": sum(
                int((item.get("measurement") or {}).get("pre_measurement_steps", 0))
                for item in workflows
            ),
            "post_cutoff_steps": sum(
                int((item.get("measurement") or {}).get("post_cutoff_steps", 0))
                for item in workflows
            ),
            "semantic_evaluated_workflows": semantic_evaluated,
            "truncated_workflows": sum(
                bool((item.get("measurement") or {}).get("measurement_truncated"))
                for item in workflows
            ),
            "zero_progress_workflows": zero_progress_workflows,
            "zero_progress_fraction": (
                zero_progress_workflows / len(workflows) if workflows else 1.0
            ),
            "audit_passed": audit["passed"],
            "backend_quiescent_after_cutoff": True,
            "passed": measurement_window_report["control_passed"] and audit["passed"],
        }
    )
    result = {
        "schema_version": 1,
        "run_id": measurement_id,
        "mode": mode,
        "profile": profile.name,
        "profile_config": profile_config,
        "model_adapter": model_adapter_summary,
        "corpus_namespace": corpus_namespace,
        "corpus_fingerprint": corpus_fingerprint(
            corpus_namespace,
            profile.workflows,
            workload_family=profile.workload_family,
            corpus_partition=profile.corpus_partition,
            seed=profile.seed,
        ),
        "workload_nonce_sha256": hashlib.sha256(workload_nonce.encode()).hexdigest(),
        "started_unix": measurement_start_unix,
        "workload_started_unix": started,
        "ended_unix": measurement_end_unix,
        "wall_ended_unix": wall_ended,
        # Backwards-compatible field: this remains preparation outside the
        # measured interval.  Detailed queued overhead is reported separately.
        "sandbox_preparation_sec": preparation_ended - preparation_started,
        "sandbox_preparation": {
            "live_sandbox_limit": profile.concurrency,
            "initial": {
                "count": initial_sandbox_count,
                "wall_sec": preparation_ended - preparation_started,
                "configured_create_concurrency": profile.sandbox_prepare_concurrency,
                "included_in_measured_interval": False,
            },
            "queued": {
                "count": profile.workflows - initial_sandbox_count,
                **queued_sandbox_preparation,
            },
        },
        "elapsed_sec": elapsed,
        "wall_elapsed_sec": wall_elapsed,
        "measurement_window": measurement_window_report,
        "candidate_leg_wall_timeout_sec": (
            candidate_leg_wall_timeout_sec(profile)
            if mode == "candidate" and profile.measurement_window_sec is None
            else None
        ),
        "valid_steps": valid_steps,
        "total_executed_steps": total_executed_steps,
        "valid_steps_per_min": valid_steps * 60 / elapsed,
        "completed_workflows": completed,
        "completed_workflows_per_min": completed * 60 / elapsed,
        "tests_passed_workflows": tests_passed,
        "tests_passed_workflows_per_min": tests_passed * 60 / elapsed,
        "program_lifecycle": {
            "enabled": mode == "candidate",
            "warmup": {
                key: warmup_protocol[key]
                for key in (
                    "program_release_attempted",
                    "program_release_acknowledged",
                    "program_release_failed",
                )
            },
            "measured_attempted": len(lifecycle_releases),
            "measured_acknowledged": sum(
                bool(item.get("acknowledged")) for item in lifecycle_releases
            ),
            "measured_unsupported": sum(
                item.get("supported") is False for item in lifecycle_releases
            ),
            "measured_failed": sum(
                item.get("supported") is not False
                and not item.get("acknowledged")
                for item in lifecycle_releases
            ),
        },
        "request_count": len(client_records),
        "measured_request_count": len(measured_client_records),
        "request_success_rate": (
            successful_requests / len(measured_client_records)
            if measured_client_records
            else 0.0
        ),
        "valid_step_ratio": (
            valid_steps / len(measured_client_records)
            if measured_client_records
            else 0.0
        ),
        "agent_step_outcomes": agent_step_outcomes,
        "format_error_details": format_error_details,
        "minimum_workflow_progress": min(scheduling_progress) if scheduling_progress else 0,
        "zero_progress_workflows": zero_progress_workflows,
        "p50_step_latency_sec": percentile(step_latencies, 0.50),
        "p95_step_latency_sec": percentile(step_latencies, 0.95),
        "p99_step_latency_sec": percentile(step_latencies, 0.99),
        "p95_query_latency_sec": percentile(query_latencies, 0.95),
        "p99_query_latency_sec": percentile(query_latencies, 0.99),
        "p50_tool_latency_sec": percentile(tool_latencies, 0.50),
        "p95_tool_latency_sec": percentile(tool_latencies, 0.95),
        "p99_tool_latency_sec": percentile(tool_latencies, 0.99),
        "aggregate_model_query_sec": total_query_sec,
        "aggregate_tool_sec": total_tool_sec,
        "model_query_time_fraction": (
            total_query_sec / (total_query_sec + total_tool_sec)
            if total_query_sec + total_tool_sec
            else None
        ),
        "p50_ttft_sec": percentile(ttft, 0.50),
        "p95_ttft_sec": percentile(ttft, 0.95),
        "p99_ttft_sec": percentile(ttft, 0.99),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "request_token_distribution": token_distribution,
        "agent_depth": {
            "sample_count": len(workflow_model_calls),
            "p25_model_calls": percentile(workflow_model_calls, 0.25),
            "p50_model_calls": percentile(workflow_model_calls, 0.50),
            "p95_model_calls": percentile(workflow_model_calls, 0.95),
            "max_model_calls": max(workflow_model_calls) if workflow_model_calls else None,
            "target_max_steps": profile.max_steps,
        },
        "request_depth_trajectory": request_depth_trajectory(
            measured_client_records,
            profile.workflows,
        ),
        "model_response_outcomes": model_response_outcomes(
            measured_client_records, effective_max_tokens
        ),
        "model_protocol_diagnostics": model_protocol_diagnostics(measured_client_records),
        "backend_tokens_per_valid_step": (prompt_tokens + completion_tokens) / valid_steps if valid_steps else None,
        "warmup_audit": warmup_audit,
        "warmup_protocol": warmup_protocol,
        "backend_audit": audit,
        "vllm_metrics": {
            **metric_delta,
            "live_series": live_metrics,
            # Trusted calibration/final evidence retains the actual sampled
            # trajectory so pressure knees can be audited after the run.
            # server.public_summary removes it from candidate-visible output.
            "live_samples": metric_points,
        },
        "pressure_validation": pressure,
        "infrastructure_failure": infrastructure_failure,
        "candidate_failure": candidate_failure,
        "workflows": workflows,
    }
    # Additive finite-only schema. Fixed-window public output is kept
    # byte-for-byte free of finite-agent timing fields.
    return attach_agent_execution_interval(result, agent_execution_interval)
