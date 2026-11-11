"""Redacting Judge-adapter contracts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

import pytest

from conftest import JUDGE, load_file

adapter = load_file("eagle3_run_judge_test", JUDGE / "run_judge.py")


def records(score: float = 10.0):
    """Build mutually consistent private reward/detail fixtures."""
    from test_f_adapter import records as complete_records
    reward, detail = complete_records()
    reward["reward"] = reward["score_tokens_per_s"] = score
    detail["reward"] = detail["score_tokens_per_s"] = score
    for name, cell in detail["cells"].items():
        rate = score or 100.0
        cell["tokens_per_s"] = detail["cell_tokens_per_s"][name] = rate
        for measurement in cell["measurements"]:
            measurement["wall_s"] = sum(measurement["generated_lengths"]) / rate
    detail["private_tokens"] = [1, 2, 3]
    return reward, detail


def test_adapter_accepts_consistent_finite_result() -> None:
    reward, detail = records()
    result = adapter.build_structured_result(
        reward,
        detail,
        verifier_completed=True,
        verifier_returncode=0,
    )
    assert result["valid"] is True
    assert result["score"] == pytest.approx(10.0)
    assert "private_tokens" not in json.dumps(result)


def test_adapter_rejects_record_disagreement() -> None:
    reward, detail = records()
    reward["score_tokens_per_s"] = 11.0
    result = adapter.build_structured_result(
        reward,
        detail,
        verifier_completed=True,
        verifier_returncode=0,
    )
    assert result["valid"] is False
    assert result["score"] == 0.0


def test_adapter_rejects_nonbinary_gate() -> None:
    reward, detail = records()
    reward[adapter.GATES[0]] = detail[adapter.GATES[0]] = 0.5
    result = adapter.build_structured_result(
        reward,
        detail,
        verifier_completed=True,
        verifier_returncode=0,
    )
    assert result["valid"] is False


@pytest.mark.parametrize("failed_gate", adapter.GATES)
def test_each_gate_failure_is_redacted_but_trusted(failed_gate: str) -> None:
    """Exercise the public-adapter failure branch for every private gate."""
    reward, detail = records(score=0.0)
    reward[failed_gate] = 0.0
    detail[failed_gate] = 0.0
    detail["status"] = "FAIL"
    result = adapter.build_structured_result(
        reward,
        detail,
        verifier_completed=True,
        verifier_returncode=1,
    )
    assert result["valid"] is False
    assert result["score"] == 0.0
    expected_rate = (len(adapter.GATES) - 1) / (len(adapter.GATES) + 1)
    assert result["pass_rate"] == pytest.approx(expected_rate)
    gate_detail = next(
        row for row in result["details"] if row["name"] == failed_gate
    )
    assert gate_detail["status"] == "FAILED"
    assert "private_tokens" not in json.dumps(result)


def test_pending_policy_never_leaks_score_or_gate_results() -> None:
    """Qualification-pending results are untrusted and therefore fully redacted."""
    reward, detail = records(score=10.0)
    detail["status"] = "PENDING_QUALIFICATION"
    result = adapter.build_structured_result(
        reward,
        detail,
        verifier_completed=True,
        verifier_returncode=0,
    )
    assert result["valid"] is False
    assert result["score"] == 0.0
    assert all(row["status"] == "ERROR" for row in result["details"][:-1])


def test_structured_markers_are_emitted_exactly_once(capsys) -> None:
    """Keep the harness marker framing stable and machine parseable."""
    adapter.emit_structured({"valid": False, "score": 0.0})
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == adapter.MARKER_START
    assert lines[-1] == adapter.MARKER_END
    assert lines.count(adapter.MARKER_START) == 1
    assert lines.count(adapter.MARKER_END) == 1
    assert json.loads(lines[1]) == {"score": 0.0, "valid": False}


def test_adapter_timeout_returns_redacted_failure(tmp_path) -> None:
    """A timed-out verifier session must not yield a partially trusted score."""
    verifier = tmp_path / "slow_verifier.py"
    verifier.write_text(
        "import time\ntime.sleep(60)\n",
        encoding="utf-8",
    )
    result = adapter.evaluate(
        verifier,
        tmp_path,
        tmp_path / "logs",
        0.05,
        environ={},
    )
    assert result["valid"] is False
    assert result["score"] == 0.0
    assert result["metrics"]["verifier_completed"] is False


def test_adapter_timeout_kills_nested_process_session(tmp_path) -> None:
    """A verifier timeout must also stop a worker in its own session."""
    child_pid_path = tmp_path / "child.pid"
    verifier = tmp_path / "nested_verifier.py"
    verifier.write_text(
        "import pathlib\n"
        "import subprocess\n"
        "import sys\n"
        "import time\n"
        "child = subprocess.Popen(\n"
        "    [sys.executable, '-c', 'import time; time.sleep(60)'],\n"
        "    start_new_session=True,\n"
        ")\n"
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid))\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    result = adapter.evaluate(
        verifier,
        tmp_path,
        tmp_path / "logs",
        0.25,
        environ=dict(os.environ),
    )
    assert result["valid"] is False
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        status_path = Path(f"/proc/{child_pid}/stat")
        if not status_path.exists():
            break
        fields = status_path.read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            break
        time.sleep(0.02)
    else:
        pytest.fail("nested verifier worker survived adapter timeout")
