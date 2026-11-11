from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing_extensions import NotRequired, TypedDict

# --- Paths ---
# Resolved from env vars or cwd-relative defaults so sforge works as a
# pip-installed tool without depending on the source checkout location.
LOG_DIR = Path(os.environ.get("SFORGE_LOG_DIR", "logs")).resolve()
TASKS_DIR = Path(os.environ.get("SFORGE_TASKS_DIR", "tasks")).resolve()

BUILD_DIR = LOG_DIR / "build_images"
BASE_IMAGE_BUILD_DIR = BUILD_DIR / "base"
WORK_IMAGE_BUILD_DIR = BUILD_DIR / "work"
JUDGE_IMAGE_BUILD_DIR = BUILD_DIR / "judge"
RUNS_LOG_DIR = LOG_DIR / "runs"

# --- Benchmark registry (name → HuggingFace repo) ---
# Used only by `sforge fetch`, which downloads an upstream task catalogue from
# HuggingFace. This repository's own tasks live in tasks/latest/<task>/tasks/
# and need no fetch; the one entry here names the upstream ByteDance benchmark
# that the sforge/ runtime was snapshotted from.
BENCHMARK_REGISTRY: dict[str, str] = {
    "edgebench-upstream": "ByteDance-Seed/EdgeBench",
}
DEFAULT_BENCHMARK = "edgebench-upstream"

# --- Defaults ---
DEFAULT_EVAL_INTERVAL = 300  # seconds

# --- Admin secret for judge server registration ---
# Only known to host-side processes (run_agent.py, judge_server.py).
# Never injected into agent containers. Formal launchers export the same
# random value to both host processes. A process-local random fallback keeps
# read-only CLI commands usable without publishing a shared development key;
# separately launched serve/run processes must set SFORGE_ADMIN_SECRET.
ADMIN_SECRET_HEADER = "X-SForge-Admin-Secret"
SESSION_TOKEN_HEADER = "X-SForge-Token"
SUBMISSION_KIND_HEADER = "X-SForge-Submission-Kind"
VERSION_LABEL_HEADER = "X-SForge-Version-Label"
SUBMISSION_ARCHIVE_MEDIA_TYPE = "application/gzip"
_admin_secret_override = os.environ.get("SFORGE_ADMIN_SECRET")
if (
    _admin_secret_override is not None
    and not _admin_secret_override.strip()
):
    raise RuntimeError("SFORGE_ADMIN_SECRET must not be empty")
ADMIN_SECRET = (
    _admin_secret_override
    if _admin_secret_override is not None
    else secrets.token_urlsafe(48)
)

# --- Docker ---
DOCKER_USER = "root"
UTF8 = "utf-8"

# --- Log markers (reuse SWE-bench convention) ---
START_TEST_OUTPUT = ">>>>> Start Test Output"
END_TEST_OUTPUT = ">>>>> End Test Output"

# --- Test status ---
PASSED = "PASSED"
FAILED = "FAILED"
ERROR = "ERROR"
SKIPPED = "SKIPPED"


class WorkConfig(TypedDict):
    specs_dir: NotRequired[str]
    agent_query: NotRequired[str]
    setup_cmds: NotRequired[list[str]]
    image_tag: NotRequired[str]


class JudgeConfig(TypedDict):
    eval_cmd: str
    eval_timeout: int
    parser: str
    private_artifact_path: NotRequired[str]
    private_artifact_paths: NotRequired[list[str]]
    private_bootstrap: NotRequired[str]
    public_eval_cmd: NotRequired[str]
    public_eval_timeout: NotRequired[int]
    public_private_artifact_path: NotRequired[str]
    score_direction: NotRequired[str]
    selection: NotRequired[str]
    setup_cmds: NotRequired[list[str]]
    image_tag: NotRequired[str]


class SForgeTask(TypedDict):
    task_id: str
    name: str
    base_image: str
    platform: str
    cwd: str
    submit_paths: list[str]
    work: WorkConfig
    judge: JudgeConfig
    portable: NotRequired[dict]
    internet: NotRequired[bool]
