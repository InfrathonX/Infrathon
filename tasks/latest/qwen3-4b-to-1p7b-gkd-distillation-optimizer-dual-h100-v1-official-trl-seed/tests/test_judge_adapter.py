from __future__ import annotations

import importlib.util

import pytest

from conftest import TASK_ROOT


def load_adapter():
    path = TASK_ROOT / "judge/run_judge.py"
    spec = importlib.util.spec_from_file_location("gkd_judge_adapter", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_packaged_and_root_judge_adapters_are_byte_identical():
    assert (TASK_ROOT / "images/judge/run_judge.py").read_bytes() == (
        TASK_ROOT / "judge/run_judge.py"
    ).read_bytes()


def test_adapter_requires_formal_readiness():
    adapter = load_adapter()
    reward = {
        "reward": 123.0,
        "source_ok": 1.0,
        "correctness_ok": 1.0,
        "protocol_ok": 1.0,
        "e2e_schema_ok": 1.0,
        "e2e_execution_ok": 1.0,
        "loss_gate_ok": 1.0,
        "memory_gate_ok": 1.0,
        "correctness_evidence_ok": 1.0,
        "formal_ready": 0.0,
        "formal_score_enabled": 1.0,
    }
    result = adapter.build_result(reward, True, 0)
    assert result["valid"] is False
    assert result["score"] == 0.0


def test_adapter_accepts_only_all_gates_and_positive_finite_score():
    adapter = load_adapter()
    reward = {
        "reward": 123.0,
        "source_ok": 1.0,
        "correctness_ok": 1.0,
        "protocol_ok": 1.0,
        "e2e_schema_ok": 1.0,
        "e2e_execution_ok": 1.0,
        "loss_gate_ok": 1.0,
        "memory_gate_ok": 1.0,
        "correctness_evidence_ok": 1.0,
        "formal_ready": 1.0,
        "formal_score_enabled": 1.0,
    }
    result = adapter.build_result(reward, True, 0)
    assert result["valid"] is True
    assert result["score"] == 123.0

    del reward["correctness_evidence_ok"]
    assert adapter.build_result(reward, True, 0)["valid"] is False
    reward["correctness_evidence_ok"] = 0.0
    assert adapter.build_result(reward, True, 0)["valid"] is False


def test_adapter_rejects_legacy_reward_without_e2e_gates():
    adapter = load_adapter()
    reward = {
        "reward": 123.0,
        "source_ok": 1.0,
        "correctness_ok": 1.0,
        "protocol_ok": 1.0,
        "formal_ready": 1.0,
    }
    result = adapter.build_result(reward, True, 0)
    assert result["valid"] is False
    assert result["score"] == 0.0


def test_adapter_exposes_only_bounded_worker_failure_feedback():
    adapter = load_adapter()
    private = {
        "e2e_formal": {
            "failure_stage": "candidate_worker",
            "failure_class": "timeout",
            "failure_role": "candidate",
            "private_row": "must-not-forward",
            "stderr": "must-not-forward",
        }
    }
    result = adapter.build_result({}, True, 1, private)
    assert result["summary"] == (
        "Formal E2E execution did not complete: candidate_worker/timeout."
    )
    assert result["metrics"]["failure_stage"] == "candidate_worker"
    assert result["metrics"]["failure_class"] == "timeout"
    assert "failure_role" not in result["metrics"]
    assert "private_row" not in result["metrics"]
    assert "stderr" not in result["metrics"]


@pytest.mark.parametrize("role", ["reference", "candidate"])
def test_adapter_exposes_bounded_role_only_for_telemetry_validation(role):
    adapter = load_adapter()
    private = {
        "e2e_formal": {
            "failure_stage": "telemetry_validation",
            "failure_class": "invalid_telemetry",
            "failure_role": role,
            "private_row": "must-not-forward",
        }
    }

    result = adapter.build_result({}, True, 1, private)

    assert result["metrics"]["failure_role"] == role
    assert "private_row" not in result["metrics"]


@pytest.mark.parametrize("role", ["REFERENCE", "both", "", 1, None])
def test_adapter_rejects_unbounded_telemetry_failure_role(role):
    adapter = load_adapter()
    private = {
        "e2e_formal": {
            "failure_stage": "telemetry_validation",
            "failure_class": "invalid_telemetry",
            "failure_role": role,
        }
    }

    result = adapter.build_result({}, True, 1, private)

    assert "failure_role" not in result["metrics"]


def test_adapter_reports_completed_gate_groups_without_private_values():
    adapter = load_adapter()
    private = {
        "e2e_formal": {
            "failed_gate_groups": ["loss", "correctness_update"],
            "threshold": 0.123,
        }
    }
    result = adapter.build_result({}, True, 1, private)
    assert result["summary"] == (
        "Formal E2E execution completed; failed gate groups: "
        "loss, correctness_update."
    )
    assert result["metrics"]["failed_gate_groups"] == [
        "loss",
        "correctness_update",
    ]
    assert "threshold" not in result["metrics"]
