from __future__ import annotations

import pytest

from conftest import load_module


def test_host_and_image_judge_adapters_are_identical(task_root):
    assert (task_root / "judge/run_judge.py").read_bytes() == (
        task_root / "images/judge/run_judge.py"
    ).read_bytes()


def test_adapter_quality_is_non_blocking(task_root):
    module = load_module("judge_adapter", task_root / "judge/run_judge.py")
    reward = {
        "reward": 42.5,
        "source_ok": 1.0,
        "integrity_ok": 1.0,
        "quality_ok": 1.0,
        "performance_ok": 1.0,
    }
    result = module.build_result(reward, {"geomean_throughput_tok_s": 42.5}, True, 0)
    assert result["valid"] is True
    assert result["score"] == 42.5
    assert result["metrics"] == {"reward": 42.5}
    assert "private" not in result["summary"].lower()
    reward["quality_ok"] = 0.0
    result = module.build_result(reward, {}, True, 0)
    assert result["valid"] is True
    assert result["score"] == 42.5
    assert result["metrics"] == {"reward": 42.5}
    assert [item["name"] for item in result["details"]] == [
        "source",
        "integrity",
        "performance",
    ]
    assert "below the reference (non-blocking)" in result["summary"]


def test_adapter_still_rejects_a_real_gate_failure(task_root):
    module = load_module("judge_adapter_gate", task_root / "judge/run_judge.py")
    reward = {
        "reward": 42.5,
        "source_ok": 1.0,
        "integrity_ok": 0.0,
        "quality_ok": 1.0,
        "performance_ok": 1.0,
    }
    result = module.build_result(reward, {}, True, 0)
    assert result["valid"] is False
    assert result["score"] == 0.0


def test_adapter_enforces_and_reports_the_complete_workload_quality_gate(task_root):
    module = load_module("judge_adapter_complete_quality", task_root / "judge/run_judge.py")
    reward = {key: 1.0 for key in ("source_ok", "integrity_ok", "quality_ok", "performance_ok")}
    reward["reward"] = 42.5
    result = module.build_result(reward, {"quality_hard_gate": True}, True, 0)
    assert result["valid"] is True
    assert "quality gate passed (enforced)" in result["summary"]
    assert "non-blocking" not in result["summary"]
    reward["quality_ok"] = 0.0
    assert module.build_result(reward, {"quality_hard_gate": True}, True, 0)["valid"] is False


def test_adapter_returns_a_rejected_quality_control_classification(task_root):
    module = load_module("judge_adapter_quality_control", task_root / "judge/run_judge.py")
    reward = {key: 1.0 for key in ("source_ok", "integrity_ok", "performance_ok")}
    reward["quality_ok"] = 0.0
    reward["reward"] = 23.8
    detail = {"quality_hard_gate": True, "quality_control": True}
    result = module.build_result(reward, detail, True, 0)
    assert result["valid"] is True
    assert "Quality-control classification recorded" in result["summary"]


def test_adapter_rejects_nonzero_return(task_root):
    module = load_module("judge_adapter_return", task_root / "judge/run_judge.py")
    reward = {key: 1.0 for key in ("source_ok", "integrity_ok", "quality_ok", "performance_ok")}
    reward["reward"] = 10.0
    assert not module.build_result(reward, {}, True, 1)["valid"]


def test_adapter_uses_the_same_complete_final_timeout_for_all_rounds(
    task_root, monkeypatch
):
    module = load_module("judge_adapter_timeout", task_root / "judge/run_judge.py")
    monkeypatch.delenv("LLADA_JUDGE_TIMEOUT", raising=False)
    monkeypatch.setenv("LLADA_INTERACTIVE_JUDGE_TIMEOUT", "2345.5")
    monkeypatch.setenv("LLADA_FINAL_JUDGE_TIMEOUT", "45678")
    assert module.context_timeout("agent-17") == 45678.0
    assert module.context_timeout("manual-1") == 45678.0
    assert module.context_timeout("") == 45678.0


@pytest.mark.parametrize("value", [None, "", "0", "-1", "nan", "inf", "invalid"])
def test_adapter_fails_closed_without_valid_role_timeout(
    task_root, monkeypatch, value
):
    module = load_module("judge_adapter_timeout_closed", task_root / "judge/run_judge.py")
    monkeypatch.setenv("LLADA_JUDGE_TIMEOUT", "99999")
    if value is None:
        monkeypatch.delenv("LLADA_FINAL_JUDGE_TIMEOUT", raising=False)
    else:
        monkeypatch.setenv("LLADA_FINAL_JUDGE_TIMEOUT", value)
    with pytest.raises(ValueError, match="LLADA_FINAL_JUDGE_TIMEOUT"):
        module.context_timeout("manual-1")
