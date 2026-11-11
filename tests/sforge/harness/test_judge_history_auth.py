from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from httpx import Response

from sforge.harness import judge_server
from sforge.harness.config import SForgeConfig
from sforge.harness.constants import ADMIN_SECRET_HEADER
from sforge.harness.judge_server import JudgeState


TEST_ADMIN_SECRET = "unit-test-admin-secret"
TEST_TOKEN = "unit-test-session-token"


@pytest.fixture
def history_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> TestClient:
    full_history = {
        "run_id": "test-run",
        "best_score": 2.0,
        "best_pass_rate": 1.0,
        "best_round": "auto-1",
        "agent_submissions": 1,
        "auto_submissions": 1,
        "public_submissions": 1,
        "entries": [
            {
                "type": "submission",
                "round": "agent-1",
                "status": "completed",
                "pass_rate": 0.5,
                "score": 1.0,
            },
            {
                "type": "submission",
                "round": "auto-1",
                "status": "completed",
                "pass_rate": 1.0,
                "score": 2.0,
            },
            {
                "type": "public",
                "round": "public-1",
                "status": "completed",
                "pass_rate": 1.0,
                "score": 999.0,
            },
        ],
    }

    monkeypatch.setattr(judge_server, "ADMIN_SECRET", TEST_ADMIN_SECRET)
    monkeypatch.setattr(
        judge_server,
        "create_backend_from_config",
        lambda config: object(),
    )
    monkeypatch.setattr(JudgeState, "_reaper_loop", lambda self: None)
    monkeypatch.setattr(
        JudgeState,
        "load_tasks",
        lambda self: setattr(
            self,
            "tasks",
            {
                "test-task": SimpleNamespace(
                    judge=SimpleNamespace(
                        public_eval_cmd="public-eval",
                        selection="valid_then_score",
                        score_direction="maximize",
                    )
                )
            },
        ),
    )

    def resolve_token(_self, token: str) -> dict:
        if token != TEST_TOKEN:
            raise KeyError(token)
        return {
            "run_id": "test-run",
            "task_id": "test-task",
        }

    monkeypatch.setattr(
        JudgeState,
        "resolve_token",
        resolve_token,
    )
    monkeypatch.setattr(
        JudgeState,
        "get_run_history",
        lambda self, run_id, task_id: full_history,
    )

    return TestClient(
        judge_server.create_app(SForgeConfig(log_dir=tmp_path))
    )


def _rounds(response: Response) -> list[str]:
    assert response.status_code == 200
    return [entry["round"] for entry in response.json()["entries"]]


def test_history_accepts_admin_secret_header(
    history_client: TestClient,
) -> None:
    response = history_client.get(
        "/api/v1/history",
        params={"token": TEST_TOKEN},
        headers={ADMIN_SECRET_HEADER: TEST_ADMIN_SECRET},
    )

    assert _rounds(response) == ["agent-1", "auto-1", "public-1"]


def test_history_keeps_legacy_admin_secret_query_compatible(
    history_client: TestClient,
) -> None:
    response = history_client.get(
        "/api/v1/history",
        params={
            "token": TEST_TOKEN,
            "admin_secret": TEST_ADMIN_SECRET,
        },
    )

    assert _rounds(response) == ["agent-1", "auto-1", "public-1"]


def test_history_without_admin_secret_is_agent_only(
    history_client: TestClient,
) -> None:
    response = history_client.get(
        "/api/v1/history",
        params={"token": TEST_TOKEN},
    )

    assert _rounds(response) == ["agent-1"]


def test_history_header_takes_precedence_over_legacy_query(
    history_client: TestClient,
) -> None:
    response = history_client.get(
        "/api/v1/history",
        params={
            "token": TEST_TOKEN,
            "admin_secret": TEST_ADMIN_SECRET,
        },
        headers={ADMIN_SECRET_HEADER: "wrong-test-secret"},
    )

    assert _rounds(response) == ["agent-1"]


def test_valid_header_takes_precedence_over_wrong_legacy_query(
    history_client: TestClient,
) -> None:
    response = history_client.get(
        "/api/v1/history",
        params={
            "token": TEST_TOKEN,
            "admin_secret": "wrong-test-secret",
        },
        headers={ADMIN_SECRET_HEADER: TEST_ADMIN_SECRET},
    )

    assert _rounds(response) == ["agent-1", "auto-1", "public-1"]


def test_public_history_is_a_separate_authenticated_domain(
    history_client: TestClient,
) -> None:
    response = history_client.get(
        "/api/v1/public/history",
        params={"token": TEST_TOKEN},
    )

    assert response.status_code == 200
    assert response.json()["public_submissions"] == 1
    assert _rounds(response) == ["public-1"]


def test_public_history_rejects_an_invalid_session_token(
    history_client: TestClient,
) -> None:
    response = history_client.get(
        "/api/v1/public/history",
        params={"token": "wrong"},
    )

    assert response.status_code == 401


def test_host_history_view_survives_the_final_seal(
    history_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After seal_handoff closes evaluation admission, run_agent still drains
    pending Judge work through the host (admin) history view; only the
    agent-visible view is denied."""

    def sealed_resolve_token(_self, token: str) -> dict:
        if token != TEST_TOKEN:
            raise KeyError(token)
        return {
            "run_id": "test-run",
            "task_id": "test-task",
            "evaluation_enabled": False,
            "admission_closed": True,
        }

    monkeypatch.setattr(JudgeState, "resolve_token", sealed_resolve_token)
    host = history_client.get(
        "/api/v1/history",
        params={"token": TEST_TOKEN},
        headers={ADMIN_SECRET_HEADER: TEST_ADMIN_SECRET},
    )
    assert _rounds(host) == ["agent-1", "auto-1", "public-1"]
    agent = history_client.get("/api/v1/history", params={"token": TEST_TOKEN})
    assert agent.status_code == 403
    assert "version checkpoints" in agent.json()["detail"]
    unknown = history_client.get(
        "/api/v1/history",
        params={"token": "other"},
        headers={ADMIN_SECRET_HEADER: TEST_ADMIN_SECRET},
    )
    assert unknown.status_code == 401
