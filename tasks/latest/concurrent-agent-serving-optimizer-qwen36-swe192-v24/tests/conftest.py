from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest


TASK_ROOT = Path(__file__).resolve().parents[1]
MLSYSFORGE_ROOT = TASK_ROOT.parent
JUDGE_PATH = TASK_ROOT / "judge" / "run_judge.py"
RUNTIME_ENTRY_PATH = TASK_ROOT / "scripts" / "sforge_runtime_entry.py"


def load_module(name: str, path: Path):
    assert path.is_file(), f"required migration file is missing: {path}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def judge_module():
    return load_module("kvbench_judge_contract", JUDGE_PATH)


@pytest.fixture(scope="module")
def runtime_module():
    # The entry imports the in-tree SForge package before installing its
    # DockerBackend monkey patch.  Make that package importable in a checkout
    # without requiring an editable installation.
    sys.path.insert(0, str(MLSYSFORGE_ROOT))
    try:
        from sforge.harness import run_agent
        from sforge.harness.agent.claude_code import ClaudeCodeAgent
        from sforge.harness.agent.codex import CodexAgent
        from sforge.harness.backend.docker_backend import DockerBackend
        from sforge.harness.log_parsers import structured_json
        import uvicorn

        originals = {
            "create_container": DockerBackend.create_container,
            "cleanup_container": DockerBackend.cleanup_container,
            "install_cmds": CodexAgent.install_cmds,
            "claude_install_cmds": ClaudeCodeAgent.install_cmds,
            "format_run_cmd": CodexAgent.format_run_cmd,
            "generate_evolve_prompt": run_agent.generate_evolve_prompt,
            "runtime_probe_cmds": run_agent.runtime_probe_cmds,
            "uvicorn_run": uvicorn.run,
            "parse_structured_json": structured_json.parse_structured_json,
        }
        module = load_module(
            "kvbench_runtime_contract",
            RUNTIME_ENTRY_PATH,
        )
        yield module
    finally:
        if "originals" in locals():
            DockerBackend.create_container = originals["create_container"]
            DockerBackend.cleanup_container = originals["cleanup_container"]
            CodexAgent.install_cmds = originals["install_cmds"]
            ClaudeCodeAgent.install_cmds = originals["claude_install_cmds"]
            CodexAgent.format_run_cmd = originals["format_run_cmd"]
            run_agent.generate_evolve_prompt = originals["generate_evolve_prompt"]
            run_agent.runtime_probe_cmds = originals["runtime_probe_cmds"]
            uvicorn.run = originals["uvicorn_run"]
            structured_json.parse_structured_json = originals["parse_structured_json"]
        try:
            sys.path.remove(str(MLSYSFORGE_ROOT))
        except ValueError:
            pass


def fixed_window_audit() -> dict[str, object]:
    """A valid strict-window audit with one right-censored request."""

    return {
        "passed": True,
        "window_censored": True,
        "mismatches": [],
        "client_requests": 102,
        "audited_backend_requests": 101,
        "strict_client_requests": 100,
        "cutoff_client_requests": 2,
        "cutoff_backend_admitted": 1,
        "cutoff_backend_unadmitted": 1,
        "extra_backend_requests": 0,
        "missing_backend_requests": 1,
        "p99_gateway_wait_sec": 60.0,
    }


HIDDEN_WARMUP_SEC = 300
HIDDEN_MEASUREMENT_SEC = 1800


def fixed_window_profile_config(
    *, warmup_sec: int = HIDDEN_WARMUP_SEC, window_sec: int = HIDDEN_MEASUREMENT_SEC
) -> dict[str, object]:
    """The scored-window shape a trusted leg reports about itself.

    Window-relative gates read it, so the fixtures carry it exactly as the
    frozen runner does.
    """

    return {
        "workflows": 192,
        "concurrency": 192,
        "max_steps": 100,
        "measurement_warmup_sec": warmup_sec,
        "measurement_window_sec": window_sec,
    }


def fixed_window_report(
    *, warmup_sec: int = HIDDEN_WARMUP_SEC, window_sec: int = HIDDEN_MEASUREMENT_SEC
) -> dict[str, object]:
    return {
        "configured": True,
        "passed": True,
        "termination": "trusted_fixed_window",
        "duration_sec": window_sec,
        "warmup_duration_sec": warmup_sec,
    }


@pytest.fixture
def valid_baseline() -> dict[str, object]:
    return {
        "valid_steps_per_min": 83.73333333333333,
        "request_success_rate": 1.0,
        "valid_step_ratio": 0.99,
        "backend_tokens_per_valid_step": 1000.0,
        "profile_config": fixed_window_profile_config(),
        "measurement_window": fixed_window_report(),
        "backend_audit": fixed_window_audit(),
        "warmup_audit": {
            "passed": True,
            "mismatches": [],
            "extra_backend_requests": 0,
            "missing_backend_requests": 0,
        },
    }


@pytest.fixture
def valid_candidate(valid_baseline: dict[str, object]) -> dict[str, object]:
    candidate = copy.deepcopy(valid_baseline)
    candidate["valid_steps_per_min"] = 80.0
    candidate["request_success_rate"] = 0.99
    candidate["valid_step_ratio"] = 0.95
    return candidate


@pytest.fixture
def valid_evaluation() -> dict[str, object]:
    # Being slower than frozen Direct is a performance result, not a protocol
    # failure.  The old screening runner's qualification flag must therefore
    # not become an MLSys Forge validity gate.
    return {
        "valid": True,
        "qualification_passed": False,
        "strictly_better_than_v2_1_direct": False,
        "direct_valid_steps_per_min": 83.73333333333333,
        "candidate_valid_steps_per_min": 80.0,
        "speedup": 80.0 / 83.73333333333333,
        "change_percent": (80.0 / 83.73333333333333 - 1.0) * 100.0,
        "candidate_invalid_reasons": [],
        "pair_identity_invalid_reasons": [],
        "additional_identity_invalid_reasons": [],
        "all_invalid_reasons": [],
    }
