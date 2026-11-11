from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from sforge.harness import judge_server
from sforge.harness.config import SForgeConfig
from sforge.harness.judge_server import (
    AgentSubmissionBusy,
    JudgeState,
    PublicEvaluationBusy,
    SubmissionCooldownActive,
    SubmissionStatus,
)


class _Report:
    def to_dict(self) -> dict:
        return {
            "valid": True,
            "score": 1.0,
            "pass_rate": 1.0,
            "total_tests": 1,
            "passed": 1,
            "failed": 0,
            "errors": 0,
            "score_0_100": 100.0,
        }


def _state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> JudgeState:
    monkeypatch.setattr(
        judge_server,
        "create_backend_from_config",
        lambda config: object(),
    )
    monkeypatch.setattr(JudgeState, "_reaper_loop", lambda self: None)
    state = JudgeState(SForgeConfig(log_dir=tmp_path))
    state.tasks = {
        "task": SimpleNamespace(
            judge=SimpleNamespace(
                selection="valid_then_score",
                score_direction="maximize",
                public_eval_cmd="public-eval",
                public_eval_timeout=1500,
                public_private_artifact_path=(
                    "/logs/verifier/public-private.json"
                ),
                private_artifact_path="/logs/verifier/hidden-private.json",
            )
        )
    }
    return state


def test_public_round_is_unlimited_but_one_may_be_inflight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = _state(monkeypatch, tmp_path)
    token = state.register_session(
        "task",
        "run",
        max_agent_submissions=None,
        submission_cooldown=300,
    )

    assert state.consume_round(token, "agent")[2] == "agent-1"
    assert state.consume_round(token, "public")[2] == "public-1"
    with pytest.raises(PublicEvaluationBusy):
        state.consume_round(token, "public")

    state.release_public_round(token)
    assert state.consume_round(token, "public")[2] == "public-2"
    state.release_public_round(token)

    # Public diagnostics neither consume nor reset the hidden Agent cooldown.
    state.release_agent_round(token)
    with pytest.raises(SubmissionCooldownActive):
        state.consume_round(token, "agent")


def test_agent_round_is_unlimited_sequentially_but_single_inflight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    now = [100.0]
    monkeypatch.setattr(judge_server.time, "monotonic", lambda: now[0])
    state = _state(monkeypatch, tmp_path)
    token = state.register_session(
        "task",
        "run",
        max_agent_submissions=None,
        submission_cooldown=300,
    )

    assert state.resolve_token(token)["agent_inflight"] is False
    assert state.consume_round(token, "agent")[2] == "agent-1"
    accepted = state.resolve_token(token)
    assert accepted["agent_inflight"] is True
    assert accepted["next_agent"] == 2
    assert accepted["last_agent_submit_at"] == 100.0

    # A concurrent attempt gets neither a round nor a new cooldown timestamp.
    with pytest.raises(AgentSubmissionBusy):
        state.consume_round(token, "agent")
    rejected_busy = state.resolve_token(token)
    assert rejected_busy["next_agent"] == 2
    assert rejected_busy["last_agent_submit_at"] == 100.0

    state.release_agent_round(token)
    with pytest.raises(SubmissionCooldownActive):
        state.consume_round(token, "agent")
    rejected_cooldown = state.resolve_token(token)
    assert rejected_cooldown["agent_inflight"] is False
    assert rejected_cooldown["next_agent"] == 2
    assert rejected_cooldown["last_agent_submit_at"] == 100.0

    now[0] = 400.0
    assert state.consume_round(token, "agent")[2] == "agent-2"
    state.release_agent_round(token)


def test_agent_slot_is_released_when_worker_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = _state(monkeypatch, tmp_path)
    token = state.register_session("task", "run")
    _, _, round_id, _, _, _ = state.consume_round(token, "agent")
    state.submissions["submission-id"] = {
        "status": SubmissionStatus.QUEUED,
        "report": None,
        "error": None,
    }

    def fail_worker(*args, **kwargs) -> None:
        raise RuntimeError("worker failed")

    monkeypatch.setattr(state, "_grade_worker_serial", fail_worker)
    with pytest.raises(RuntimeError, match="worker failed"):
        state._grade_worker(
            "submission-id",
            "task",
            b"archive",
            run_id="run",
            round=round_id,
            session_token=token,
        )

    assert state.resolve_token(token)["agent_inflight"] is False


def test_agent_slot_is_released_when_submit_cannot_schedule(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = _state(monkeypatch, tmp_path)
    token = state.register_session("task", "run")
    _, _, round_id, _, _, _ = state.consume_round(token, "agent")

    class BrokenBackgroundTasks:
        def add_task(self, *args, **kwargs) -> None:
            raise RuntimeError("scheduler failed")

    with pytest.raises(RuntimeError, match="scheduler failed"):
        state.submit(
            "task",
            b"archive",
            run_id="run",
            round=round_id,
            session_token=token,
            background_tasks=BrokenBackgroundTasks(),  # type: ignore[arg-type]
        )

    assert state.resolve_token(token)["agent_inflight"] is False
    submission = next(iter(state.submissions.values()))
    assert submission["status"] == SubmissionStatus.ERROR
    assert submission["error"] == "scheduler failed"


class _ObserveResponseBody:
    def __init__(self, app, response_complete: threading.Event) -> None:
        self.app = app
        self.response_complete = response_complete

    async def __call__(self, scope, receive, send) -> None:
        async def observed_send(message) -> None:
            await send(message)
            if (
                message["type"] == "http.response.body"
                and not message.get("more_body", False)
            ):
                self.response_complete.set()

        await self.app(scope, receive, observed_send)


def test_http_submit_starts_grading_after_queued_response_is_sent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, JudgeState] = {}
    response_complete = threading.Event()
    grade_started = threading.Event()

    monkeypatch.setattr(
        judge_server,
        "create_backend_from_config",
        lambda config: object(),
    )
    monkeypatch.setattr(JudgeState, "_reaper_loop", lambda self: None)

    def load_tasks(state: JudgeState) -> None:
        captured["state"] = state
        state.tasks = {
            "task": SimpleNamespace(
                judge=SimpleNamespace(public_eval_cmd="public-eval")
            )
        }

    monkeypatch.setattr(JudgeState, "load_tasks", load_tasks)

    def fake_grade_worker(
        state: JudgeState,
        submission_id: str,
        task_id: str,
        archive: bytes,
        run_id: str | None = None,
        round: str | None = None,
        judge_cpu_limit: int | None = None,
        judge_mem_limit: str | None = None,
        session_token: str | None = None,
    ) -> None:
        assert response_complete.is_set()
        grade_started.set()
        state._release_submission_round(session_token, round)

    monkeypatch.setattr(JudgeState, "_grade_worker", fake_grade_worker)
    app = judge_server.create_app(SForgeConfig(log_dir=tmp_path))
    state = captured["state"]
    token = state.register_session("task", "run")

    with TestClient(_ObserveResponseBody(app, response_complete)) as client:
        response = client.post(
            "/api/v1/submit",
            content=b"archive",
            headers={
                "Content-Type": "application/gzip",
                "X-SForge-Token": token,
                "X-SForge-Submission-Kind": "agent",
            },
        )
        _, _, second_round, _, _, _ = state.consume_round(token, "agent")
        busy_response = client.post(
            "/api/v1/submit",
            content=b"archive",
            headers={
                "Content-Type": "application/gzip",
                "X-SForge-Token": token,
                "X-SForge-Submission-Kind": "agent",
            },
        )

    assert response.status_code == 200
    assert response.json()["round_id"] == "agent-1"
    assert response.json()["status"] == "queued"
    assert grade_started.is_set()
    assert second_round == "agent-2"
    assert busy_response.status_code == 409
    assert "already queued or running" in busy_response.json()["detail"]
    assert state.resolve_token(token)["next_agent"] == 3
    state.release_agent_round(token)
    assert state.resolve_token(token)["agent_inflight"] is False


def test_public_results_never_participate_in_hidden_best_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = _state(monkeypatch, tmp_path)
    state._record_submission(
        "run",
        "hidden-id",
        "task",
        "agent-1",
        {"valid": True, "score": 10.0, "pass_rate": 1.0},
        None,
    )
    state._record_submission(
        "run",
        "public-id",
        "task",
        "public-1",
        {"valid": True, "score": 999.0, "pass_rate": 1.0},
        None,
    )

    history = state.get_run_history("run", "task")

    assert history["best_round"] == "agent-1"
    assert history["best_score"] == 10.0
    assert history["agent_submissions"] == 1
    assert history["public_submissions"] == 1
    public = [entry for entry in history["entries"] if entry["type"] == "public"]
    assert [entry["round"] for entry in public] == ["public-1"]


def test_releasing_unknown_public_session_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = _state(monkeypatch, tmp_path)
    state.release_public_round("not-a-token")


def test_public_round_uses_only_public_command_timeout_and_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = _state(monkeypatch, tmp_path)
    captured: dict = {}

    def fake_judge_submission(**kwargs):
        captured.update(kwargs)
        return _Report()

    monkeypatch.setattr(judge_server, "judge_submission", fake_judge_submission)
    state.submissions["public-id"] = {
        "status": "queued",
        "report": None,
        "error": None,
    }

    state._grade_worker_serial(
        "public-id",
        "task",
        b"archive",
        run_id="run",
        round="public-1",
    )

    assert captured["eval_cmd"] == "public-eval"
    assert captured["timeout"] == 1500
    assert (
        captured["private_artifact_path"]
        == "/logs/verifier/public-private.json"
    )
    assert captured["private_artifact_paths"] == ["/logs/verifier/public-private.json"]
    assert captured["run_id"] == "run"
    assert captured["round_id"] == "public-1"
    assert state.submissions["public-id"]["status"].value == "completed"


def test_public_without_private_artifact_does_not_inherit_formal_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = _state(monkeypatch, tmp_path)
    state.tasks['task'].judge.public_private_artifact_path = None
    state.tasks['task'].judge.private_artifact_paths = [
        '/logs/verifier/hidden-private.json', '/logs/verifier/evaluation_nonce.json']
    captured: dict = {}

    def fake_judge_submission(**kwargs):
        captured.update(kwargs)
        return _Report()

    monkeypatch.setattr(judge_server, 'judge_submission', fake_judge_submission)
    state.submissions['public-id'] = {
        'status': 'queued', 'report': None, 'error': None}
    state._grade_worker_serial(
        'public-id', 'task', b'archive', run_id='run', round='public-1')

    assert captured['private_artifact_path'] is None
    assert captured['private_artifact_paths'] == []


def test_missing_run_id_is_not_replaced_in_judge_container_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = _state(monkeypatch, tmp_path)
    captured: dict = {}

    def fake_judge_submission(**kwargs):
        captured.update(kwargs)
        return _Report()

    monkeypatch.setattr(judge_server, "judge_submission", fake_judge_submission)
    state.submissions["agent-id"] = {
        "status": "queued",
        "report": None,
        "error": None,
    }
    state._grade_worker_serial(
        "agent-id",
        "task",
        b"archive",
        run_id=None,
        round="agent-1",
    )

    assert captured["run_id"] is None
    assert captured["round_id"] == "agent-1"
