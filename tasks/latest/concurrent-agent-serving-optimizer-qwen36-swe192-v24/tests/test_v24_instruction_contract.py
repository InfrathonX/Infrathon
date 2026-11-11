from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from conftest import load_module


ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = ROOT / "images/work"
def _task() -> dict:
    path = ROOT / "tasks/concurrent_agent_serving_optimizer_qwen36_swe192_v24.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_public_runtime_contract_matches_frozen_sources() -> None:
    preflight = load_module(
        "kvbench_v24_runtime_currentness",
        ROOT / "scripts/preflight.py",
    )
    observed = preflight.validate_public_runtime_contract()

    assert observed == hashlib.sha256(
        (WORK_ROOT / "RUNTIME_CONTRACT.json").read_bytes()
    ).hexdigest()


def test_public_runtime_contract_mismatch_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preflight = load_module(
        "kvbench_v24_runtime_currentness_mutation",
        ROOT / "scripts/preflight.py",
    )
    contract = json.loads(
        (WORK_ROOT / "RUNTIME_CONTRACT.json").read_text(encoding="utf-8")
    )
    contract["hybrid_cache"]["allocatable_pages"] += 1
    mutated = tmp_path / "RUNTIME_CONTRACT.json"
    mutated.write_text(json.dumps(contract), encoding="utf-8")
    monkeypatch.setattr(preflight, "PUBLIC_RUNTIME_CONTRACT", mutated)

    with pytest.raises(RuntimeError, match="does not match frozen runtime facts"):
        preflight.validate_public_runtime_contract()


def test_runtime_contract_is_static_and_contains_no_policy_or_workload_state() -> None:
    contract = json.loads(
        (WORK_ROOT / "RUNTIME_CONTRACT.json").read_text(encoding="utf-8")
    )

    assert contract["scope"] == {
        "static_contract_only": True,
        "contains_live_backend_telemetry": False,
        "contains_workload_identity_or_order": False,
        "contains_reference_policy": False,
    }
    rendered = json.dumps(contract, sort_keys=True).lower()
    for forbidden in (
        "task_id",
        "nonce",
        "workload_order",
        "scheduler_interval",
        "watermark",
        "resume_quota",
        "fair_wait",
        "oracle",
        "thunder",
    ):
        assert forbidden not in rendered


def test_task_query_is_compact_architecture_first_and_budget_neutral() -> None:
    query = _task()["work"]["agent_query"]

    for marker in (
        "workflow-level serving architecture",
        "self-contained online monitoring and control plane",
        "/app/submission/ARCHITECTURE.md",
        "materially distinct, falsifiable architectural hypotheses",
        "harness-reported agent deadline",
    ):
        assert marker in query
    for forbidden in (
        "two-hour boundary",
        "twelve-hour boundary",
        "83.4",
        "83.73333333333333",
        "108.2",
        "sforge-submit",
        "ThunderAgent",
    ):
        assert forbidden not in query
    assert len(query) < 1_200


def test_agent_instruction_sets_the_architectural_bar_without_oracle_leaks() -> None:
    instruction = (WORK_ROOT / "AGENT_TASK.md").read_text(encoding="utf-8")
    normalized = " ".join(instruction.split())

    for marker in (
        "workflow-level serving-systems problem",
        "not merely an HTTP rate-limiting problem",
        "single sub-percent improvement",
        "Candidate-owned online monitoring",
        "NEW",
        "REASONING",
        "ACTING",
        "PAUSED",
        "RELEASED",
        "conservative capacity/admission invariant",
        "at least three materially different architecture families",
        "program-aware or adaptive architecture",
        "exact candidate checkpoint/archive/hash",
        "protected best",
    ):
        assert marker in normalized
    for forbidden in (
        "ThunderAgent",
        "THUNDER_",
        "reference_gateway",
        "101.7333",
        "108.2",
        "FAIR_WAIT_THRESHOLD_SEC",
        "AGED_RESUME_QUOTA_PER_TICK",
    ):
        assert forbidden not in instruction


def test_run_budget_is_injected_from_the_actual_formal_profile(
    runtime_module,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budget_path = tmp_path / "RUN_BUDGET.json"
    monkeypatch.setattr(runtime_module, "RUN_BUDGET_PATH", budget_path)
    monkeypatch.setenv("TASK_TIMEOUT", "28800")
    monkeypatch.setenv("SUBMISSION_COOLDOWN", "300")

    payload = runtime_module.run_budget_payload()
    assert payload["agent_timeout_seconds"] == 28800
    assert payload["submission_cooldown_seconds"] == 300
    assert payload["expected_hidden_runtime_seconds"] == 2400

    completed = subprocess.run(
        ["bash", "-c", runtime_module.run_budget_install_command()],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(budget_path.read_text(encoding="utf-8")) == payload
    assert budget_path.stat().st_mode & 0o777 == 0o644


@pytest.mark.parametrize("value", ["0", "-1", "not-an-integer"])
def test_invalid_formal_budget_fails_closed(
    runtime_module,
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("TASK_TIMEOUT", value)
    monkeypatch.setenv("SUBMISSION_COOLDOWN", "300")

    with pytest.raises(RuntimeError, match="TASK_TIMEOUT must be a positive integer"):
        runtime_module.run_budget_payload()


def test_work_image_is_agent_free_but_keeps_sudo() -> None:
    dockerfile = (WORK_ROOT / "Dockerfile").read_text(encoding="utf-8")

    for forbidden in (
        "CODEX_CLI_VERSION",
        "NODE_VERSION",
        "nodejs.org",
        "npm install",
        "codex --version",
    ):
        assert forbidden not in dockerfile
    assert "sudo" in dockerfile
    assert "/opt/agent-runtime" in dockerfile
    assert "SFORGE_AGENT_RUNTIME_DIR" in dockerfile


def test_agent_runtime_probe_retains_run_budget_injection(
    runtime_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TASK_TIMEOUT", "21600")
    monkeypatch.setenv("SUBMISSION_COOLDOWN", "300")
    runtime = SimpleNamespace(cli="codex", version="0.148.0")

    commands = runtime_module.configured_runtime_probe_cmds(runtime)

    assert len(commands) == 2
    assert "/opt/agent-runtime" in commands[0]
    assert "codex --version" in commands[0]
    assert str(runtime_module.RUN_BUDGET_PATH) in commands[1]
    assert "21600" in commands[1]


def test_work_image_contains_all_three_agent_contract_documents() -> None:
    dockerfile = (WORK_ROOT / "Dockerfile").read_text(encoding="utf-8")
    dockerignore = (WORK_ROOT / ".dockerignore").read_text(encoding="utf-8")

    for name in ("RUNTIME_CONTRACT.json", "RUN_BUDGET.json"):
        assert f"COPY {'--chown=agent:agent ' if name == 'RUN_BUDGET.json' else ''}{name}" in dockerfile
        assert f"!{name}" in dockerignore
    assert "COPY --chown=agent:agent submission/ /app/submission/" in dockerfile
    assert "/app/submission/ARCHITECTURE.md" in dockerfile
    assert "chmod 0444" in dockerfile and "/app/RUNTIME_CONTRACT.json" in dockerfile


def test_v24_judge_source_digest_is_well_formed() -> None:
    completed = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; judge_source_digest',
            "bash",
            str(ROOT / "scripts/lib.sh"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    observed = completed.stdout.strip()
    assert len(observed) == 64
    assert set(observed) <= set("0123456789abcdef")


def test_public_distinct_profile_keeps_pressure_required_with_stable_floor() -> None:
    profile_path = (
        ROOT
        / "runtime/frozen/harbor_tasks/"
        "concurrent-agent-serving-optimizer-qwen36-27b-swe192-v2.1-gpu45/"
        "environment/bench-controller/profiles/public.json"
    )
    profile = json.loads(profile_path.read_text(encoding="utf-8"))["profiles"][
        "public-swe192-distinct"
    ]

    assert profile["workflows"] == 192
    assert profile["concurrency"] == 192
    assert profile["measurement_warmup_sec"] == 300
    assert profile["measurement_window_sec"] == 600
    assert profile["pressure_required"] is True
    assert profile["pressure_kv_threshold"] == 0.8
    assert profile["pressure_kv_p95_min"] == 0.85
    assert profile["pressure_time_fraction_min"] == 0.25
    assert profile["pressure_min_samples"] == 60


def test_v24_builds_matching_work_judge_and_backend_images() -> None:
    build = (ROOT / "scripts/build_images.sh").read_text(encoding="utf-8")

    assert '--build-arg "WORK_IMAGE=${WORK_IMAGE}"' in build
    assert '--tag "${JUDGE_IMAGE}"' in build
    assert '--tag "${CONTROLLER_IMAGE}"' in build
    assert '--tag "${MODEL_API_IMAGE}"' in build
    assert 'docker tag "${V22_JUDGE_IMAGE}" "${JUDGE_IMAGE}"' not in build
