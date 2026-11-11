from __future__ import annotations

import os
from pathlib import Path
import sys


def test_supervisor_allows_bounded_cooperative_cleanup(
    judge_module, monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("KVBENCH_RUNNER_TIMEOUT_SEC", "1")
    monkeypatch.setenv("KVBENCH_RUNNER_CLEANUP_GRACE_SEC", "3")
    runner_log = tmp_path / "runner.log"
    program = """
import signal
import sys
import time

def stop(_signum, _frame):
    print("cleanup-complete", flush=True)
    raise SystemExit(0)

signal.signal(signal.SIGTERM, stop)
print("measuring", flush=True)
time.sleep(30)
"""

    outcome = judge_module._run_supervised(
        [sys.executable, "-c", program],
        environment=dict(os.environ),
        runner_log=runner_log,
    )

    assert set(outcome) == {
        "returncode",
        "timed_out",
        "cancel_signal",
        "termination_sent",
        "forced_kill",
        "runtime_seconds",
        "force_reconcile",
    }
    assert outcome["timed_out"] is True
    assert outcome["termination_sent"] is True
    assert outcome["forced_kill"] is False
    assert outcome["force_reconcile"] is None
    assert outcome["returncode"] == 0
    assert "cleanup-complete" in runner_log.read_text(encoding="utf-8")
    assert runner_log.stat().st_mode & 0o777 == 0o600


def test_supervisor_kills_runner_that_ignores_cleanup_signal(
    judge_module, monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("KVBENCH_RUNNER_TIMEOUT_SEC", "1")
    monkeypatch.setenv("KVBENCH_RUNNER_CLEANUP_GRACE_SEC", "1")
    runner_log = tmp_path / "runner.log"
    program = """
import signal
import time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
time.sleep(30)
"""

    outcome = judge_module._run_supervised(
        [sys.executable, "-c", program],
        environment=dict(os.environ),
        runner_log=runner_log,
    )

    assert set(outcome) == {
        "returncode",
        "timed_out",
        "cancel_signal",
        "termination_sent",
        "forced_kill",
        "runtime_seconds",
        "force_reconcile",
    }
    assert outcome["timed_out"] is True
    assert outcome["termination_sent"] is True
    assert outcome["forced_kill"] is True
    assert outcome["force_reconcile"] is None
    assert outcome["returncode"] < 0
