from __future__ import annotations

import threading
from pathlib import Path

import pytest

from sforge.harness import judge_server
from sforge.harness.config import SForgeConfig
from sforge.harness.judge_server import JudgeState, SubmissionStatus


def _judge_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> JudgeState:
    monkeypatch.setattr(
        judge_server,
        "create_backend_from_config",
        lambda config: object(),
    )
    monkeypatch.setattr(
        JudgeState,
        "_reaper_loop",
        lambda self: None,
    )
    state = JudgeState(SForgeConfig(log_dir=tmp_path))
    state.tasks = {
        "task-a": object(),
        "task-b": object(),
    }
    return state


def test_same_task_submissions_are_serial_and_remain_queued(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = _judge_state(monkeypatch, tmp_path)
    first_entered = threading.Event()
    second_entered = threading.Event()
    release_first = threading.Event()
    calls_lock = threading.Lock()
    call_count = 0
    active = 0
    max_active = 0

    def fake_serial(
        submission_id,
        task_id,
        archive,
        run_id,
        round_id,
        judge_cpu_limit,
        judge_mem_limit,
        session_token,
    ) -> None:
        nonlocal call_count, active, max_active
        with calls_lock:
            index = call_count
            call_count += 1
            active += 1
            max_active = max(max_active, active)
        if index == 0:
            first_entered.set()
            assert release_first.wait(timeout=2)
        else:
            second_entered.set()
        state.submissions[submission_id]["status"] = (
            SubmissionStatus.COMPLETED
        )
        with calls_lock:
            active -= 1

    monkeypatch.setattr(state, "_grade_worker_serial", fake_serial)

    first_id = state.submit(
        "task-a", b"one", run_id="run", round="auto-1"
    )
    assert first_entered.wait(timeout=1)
    second_id = state.submit(
        "task-a", b"two", run_id="run", round="auto-2"
    )
    try:
        assert not second_entered.wait(timeout=0.05)
        assert (
            state.submissions[second_id]["status"]
            == SubmissionStatus.QUEUED
        )
        second_history = next(
            entry
            for entry in state.run_history["run/task-a"]
            if entry["submission_id"] == second_id
        )
        assert second_history["status"] == "queued"
    finally:
        release_first.set()

    assert second_entered.wait(timeout=1)
    assert state.submissions[first_id]["status"] == SubmissionStatus.COMPLETED
    assert state.submissions[second_id]["status"] == SubmissionStatus.COMPLETED
    assert max_active == 1


def test_different_tasks_can_be_judged_in_parallel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = _judge_state(monkeypatch, tmp_path)
    both_entered = threading.Event()
    release = threading.Event()
    lock = threading.Lock()
    active = 0

    def fake_serial(
        submission_id,
        task_id,
        archive,
        run_id,
        round_id,
        judge_cpu_limit,
        judge_mem_limit,
        session_token,
    ) -> None:
        nonlocal active
        with lock:
            active += 1
            if active == 2:
                both_entered.set()
        assert release.wait(timeout=2)
        state.submissions[submission_id]["status"] = (
            SubmissionStatus.COMPLETED
        )

    monkeypatch.setattr(state, "_grade_worker_serial", fake_serial)

    first_id = state.submit(
        "task-a", b"one", run_id="run-a", round="auto-1"
    )
    second_id = state.submit(
        "task-b", b"two", run_id="run-b", round="auto-1"
    )
    try:
        assert both_entered.wait(timeout=1)
    finally:
        release.set()

    for _ in range(100):
        if all(
            state.submissions[submission_id]["status"]
            == SubmissionStatus.COMPLETED
            for submission_id in (first_id, second_id)
        ):
            break
        threading.Event().wait(0.01)
    assert all(
        state.submissions[submission_id]["status"]
        == SubmissionStatus.COMPLETED
        for submission_id in (first_id, second_id)
    )
