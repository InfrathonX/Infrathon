"""The public sanitizer suite never hangs: attach failure and timeouts end the whole tree."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pytest

from conftest import TASK_ROOT, load_module

sys.path.insert(0, str(TASK_ROOT / "images/work/benchmark"))
runner = load_module("public_sanitize_runner", "images/work/benchmark/public_runner.py")


def _fake_sanitizer(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "compute-sanitizer"
    path.write_text("#!/bin/bash\n" + body + "\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(lab_root=tmp_path, candidate_version=1)


def test_attach_failure_stops_the_tree_and_reports_unavailable(monkeypatch, tmp_path, capsys) -> None:
    pid_file = tmp_path / "target.pid"
    sanitizer = _fake_sanitizer(tmp_path, f"""
echo "========= COMPUTE-SANITIZER"
sleep 600 &
echo $! > {pid_file}
echo "========= Error: No attachable process found. compute-sanitizer timed-out."
wait""")
    monkeypatch.setattr(runner, "SANITIZER", sanitizer)
    started = time.monotonic()
    assert runner._run_sanitize_outer(_args(tmp_path)) == runner.EXIT_SANITIZER_UNAVAILABLE
    assert time.monotonic() - started < 30
    assert "MEMCHECK                  UNAVAILABLE" in capsys.readouterr().out
    target = int(pid_file.read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(target, 0)
        except ProcessLookupError:
            break
        status = Path(f"/proc/{target}/stat")
        if status.exists() and status.read_text().rsplit(")", 1)[1].split()[0] == "Z":
            break
        time.sleep(0.1)
    else:
        pytest.fail("the hung target was left running")


def test_timeout_is_reported_and_bounded(monkeypatch, tmp_path, capsys) -> None:
    sanitizer = _fake_sanitizer(tmp_path, "sleep 600")
    monkeypatch.setattr(runner, "SANITIZER", sanitizer)
    monkeypatch.setattr(runner, "SANITIZE_TIMEOUT_SECONDS", 2)
    assert runner._run_sanitize_outer(_args(tmp_path)) == 1
    assert "MEMCHECK                  TIMEOUT" in capsys.readouterr().out


def test_clean_run_still_passes(monkeypatch, tmp_path, capsys) -> None:
    sanitizer = _fake_sanitizer(tmp_path, 'echo "========= ERROR SUMMARY: 0 errors"; echo "SANITIZE_CASE x exact_error=0.0"')
    monkeypatch.setattr(runner, "SANITIZER", sanitizer)
    assert runner._run_sanitize_outer(_args(tmp_path)) == 0
    output = capsys.readouterr().out
    assert "MEMCHECK                  PASS" in output and "SANITIZE_CASE x" in output
