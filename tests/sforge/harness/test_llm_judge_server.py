from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from _llm_judge_fixtures import (
    SEED,
    FakeClient,
    make_archive,
    make_task,
    verdict_json,
    write_documents,
    write_tree,
)
from sforge.harness import judge_server
from sforge.harness.config import SForgeConfig
from sforge.harness.grading import EvalReport
from sforge.harness.judge_server import JudgeState, SubmissionStatus
from sforge.harness.llm_judge import reviewer as reviewer_module
from sforge.harness.llm_judge.bundle import HostDirectoryBaseline
from sforge.harness.llm_judge.config import LLMJudgeConfig

TASK = make_task()
CHEAT = {
    **SEED,
    "FINAL_VERSION": "2\n",
    "candidates/kernel_v002/submission.py": (
        "import torch\n\ndef kernel(q, k, v, out):\n"
        "    out.copy_(torch.nn.functional.scaled_dot_product_attention(q, k, v))\n"
    ),
}
VIOLATION = verdict_json(
    "VIOLATION",
    confidence=0.95,
    violations=[
        {
            "rule": "self-contained Triton only",
            "files": ["candidates/kernel_v002/submission.py"],
            "evidence": "scaled_dot_product_attention(q, k, v)",
            "explanation": "SDPA does the attention",
        }
    ],
    feedback="Implement the attention in Triton; SDPA is forbidden.",
)


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    replies: list[str] = []
    judged: list[str] = []

    monkeypatch.setattr(judge_server, "create_backend_from_config", lambda config: object())
    monkeypatch.setattr(JudgeState, "_reaper_loop", lambda self: None)
    monkeypatch.setattr(
        reviewer_module, "ChatCompletionsClient", lambda config: FakeClient(replies)
    )
    baseline = HostDirectoryBaseline(
        write_tree(tmp_path / "seed", SEED),
        documents_root=write_documents(tmp_path / "docs"),
    )
    monkeypatch.setattr(JudgeState, "_baseline_provider", lambda self, run_id: baseline)

    def fake_judge(**kwargs) -> EvalReport:
        judged.append(kwargs["submission_id"])
        return EvalReport(
            task_id=kwargs["task_spec"].task_id,
            submission_id=kwargs["submission_id"],
            total_tests=1,
            passed=1,
            pass_rate=1.0,
            score=42.0,
            valid=True,
            summary="measured",
        )

    monkeypatch.setattr(judge_server, "judge_submission", fake_judge)

    def make_state(mode: str = "enforce", **overrides) -> JudgeState:
        tasks_dir = tmp_path / "task" / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        review = LLMJudgeConfig(
            mode=mode,
            engine="api",
            api_base_url="http://127.0.0.1:9/v1",
            model="judge-model",
            cache=False,
            **overrides,
        )
        state = JudgeState(
            SForgeConfig(log_dir=tmp_path / "logs", tasks_dir=tasks_dir, llm_judge=review)
        )
        state.tasks = {TASK.task_id: TASK}
        return state

    return make_state, replies, judged, tmp_path


def _wait(state: JudgeState, submission_id: str) -> dict:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        record = state.submissions[submission_id]
        if record["status"] in (SubmissionStatus.COMPLETED, SubmissionStatus.ERROR):
            return record
        time.sleep(0.02)
    raise AssertionError("submission did not finish")


def test_enforced_violation_never_reaches_the_gpu_judge(harness) -> None:
    make_state, replies, judged, tmp_path = harness
    state = make_state()
    replies.append(VIOLATION)

    submission_id = state.submit(TASK.task_id, make_archive(CHEAT), run_id="run", round="agent-1")
    record = _wait(state, submission_id)

    assert judged == []
    assert record["status"] == SubmissionStatus.COMPLETED
    report = record["report"]
    assert report["valid"] is False and report["score"] is None
    assert report["metrics"]["evaluation_status"] == "integrity_rejected"
    assert "Triton" in report["metrics"]["integrity_review"]["feedback"]
    sub_dir = tmp_path / "logs" / "runs" / "run" / TASK.task_id / "submissions" / "agent-1"
    assert (sub_dir / "submission.tar.gz").read_bytes() == make_archive(CHEAT)
    assert json.loads((sub_dir / "report.json").read_text())["valid"] is False
    assert json.loads((sub_dir / "integrity_review" / "verdict.json").read_text())[
        "status"
    ] == "violation"
    history = state.get_run_history("run", TASK.task_id)
    entry = history["entries"][0]
    assert entry["integrity_review"]["status"] == "violation"
    assert entry["valid"] is False and history["best_score"] is None


def test_passing_review_continues_to_the_gpu_judge(harness) -> None:
    make_state, replies, judged, _ = harness
    state = make_state()
    replies.append(verdict_json("PASS"))

    submission_id = state.submit(TASK.task_id, make_archive(SEED), run_id="run", round="agent-1")
    record = _wait(state, submission_id)

    assert judged == [submission_id]
    assert record["report"]["score"] == 42.0
    assert "integrity_review" not in record["report"]["metrics"]
    entry = state.get_run_history("run", TASK.task_id)["entries"][0]
    assert entry["integrity_review"]["status"] == "pass"


def test_shadow_mode_records_but_never_blocks(harness) -> None:
    make_state, replies, judged, _ = harness
    state = make_state(mode="shadow")
    replies.append(VIOLATION)

    submission_id = state.submit(TASK.task_id, make_archive(CHEAT), run_id="run", round="agent-1")
    record = _wait(state, submission_id)

    assert judged == [submission_id]
    assert record["report"]["valid"] is True
    entry = state.get_run_history("run", TASK.task_id)["entries"][0]
    assert entry["integrity_review"] == {
        "review_id": entry["integrity_review"]["review_id"],
        "mode": "shadow",
        "status": "violation",
        "blocking": False,
        "cache_hit": False,
        "rules": ["self-contained Triton only"],
    }


def test_unavailable_reviewer_admits_or_blocks_per_policy(harness) -> None:
    make_state, replies, judged, _ = harness
    state = make_state(max_attempts=1)  # no scripted reply -> endpoint error

    admitted = state.submit(TASK.task_id, make_archive(SEED), run_id="run", round="agent-1")
    assert _wait(state, admitted)["report"]["score"] == 42.0

    blocking = make_state(max_attempts=1, on_error="block")
    blocked = blocking.submit(TASK.task_id, make_archive(SEED), run_id="run2", round="agent-1")
    report = _wait(blocking, blocked)["report"]
    assert report["metrics"]["evaluation_status"] == "infrastructure_error"
    assert judged == [admitted]


def test_public_submissions_are_not_reviewed_by_default(harness) -> None:
    make_state, replies, _, _ = harness
    state = make_state()
    state.submissions["s1"] = {"status": SubmissionStatus.QUEUED}

    state._run_integrity_review("s1", TASK.task_id, make_archive(CHEAT), "run", "public-1")
    assert "integrity_review" not in state.submissions["s1"]

    replies.append(verdict_json("PASS"))
    state._run_integrity_review("s1", TASK.task_id, make_archive(SEED), "run", "auto-1")
    assert state.submissions["s1"]["integrity_review"].status == "pass"


def test_registration_announces_and_agent_history_hides_audit_fields(
    harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_state, replies, _, tmp_path = harness
    monkeypatch.setattr(judge_server, "ADMIN_SECRET", "secret")
    monkeypatch.setattr(JudgeState, "load_tasks", lambda self: setattr(self, "tasks", {TASK.task_id: TASK}))
    tasks_dir = tmp_path / "task" / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    review = LLMJudgeConfig(
        mode="enforce",
        engine="api",
        api_base_url="http://127.0.0.1:9/v1",
        model="judge-model",
        cache=False,
    )
    app = judge_server.create_app(
        SForgeConfig(log_dir=tmp_path / "logs", tasks_dir=tasks_dir, llm_judge=review)
    )
    replies.append(VIOLATION)
    with TestClient(app) as client:
        registered = client.post(
            "/api/v1/register",
            json={"task_id": TASK.task_id, "run_id": "run", "admin_secret": "secret"},
        ).json()
        assert registered["integrity_review"] == {
            "mode": "enforce",
            "announce": True,
            "review_timeout_seconds": 1800,
        }
        token = registered["token"]
        submitted = client.post(
            "/api/v1/submit",
            content=make_archive(CHEAT),
            headers={"Content-Type": "application/gzip", "X-SForge-Token": token},
        ).json()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            result = client.get(f"/api/v1/result/{submitted['submission_id']}").json()
            if result["status"] == "completed":
                break
            time.sleep(0.05)
        assert result["report"]["metrics"]["evaluation_status"] == "integrity_rejected"
        agent_view = client.get(f"/api/v1/history?token={token}").json()
        assert "integrity_review" not in agent_view["entries"][0]
        admin_view = client.get(
            f"/api/v1/history?token={token}", headers={"X-SForge-Admin-Secret": "secret"}
        ).json()
        assert admin_view["entries"][0]["integrity_review"]["status"] == "violation"


def test_deadline_fence_is_settled_before_a_rejection_is_published(harness, monkeypatch) -> None:
    make_state, replies, judged, _ = harness
    state = make_state()
    replies.append(VIOLATION)
    seen: list[str] = []

    def settle(token) -> None:
        sid = next(iter(state.submissions))
        seen.append(str(state.submissions[sid]["status"]))

    monkeypatch.setattr(state, "_settle_final_after_submission", settle)
    state._handoff_sessions["tok"] = object()
    state.tokens["tok"] = {"task_id": TASK.task_id, "run_id": "run"}
    state._persist_handoff_queue_entry = lambda **kwargs: None

    submission_id = state.submit(
        TASK.task_id, make_archive(CHEAT), run_id="run", round="agent-1", session_token="tok"
    )
    record = _wait(state, submission_id)

    assert judged == []
    assert record["report"]["metrics"]["evaluation_status"] == "integrity_rejected"
    # First settle runs inside the rejection path, while history is not yet
    # terminal; the second is the usual one after the worker finishes.
    assert seen[0] != str(SubmissionStatus.COMPLETED)
    assert len(seen) == 2


# --- backends that host their own review sandboxes (PyroMind) ---


@pytest.fixture
def platform_harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A non-Docker backend: the integrator supplies the review engine."""

    import json as _json

    from sforge.harness.llm_judge.reviewer import ApiReviewEngine

    judged: list[str] = []
    monkeypatch.setattr(
        judge_server,
        "create_backend_from_config",
        lambda config: type("PlatformBackend", (), {"backend_name": "pyromind"})(),
    )
    monkeypatch.setattr(JudgeState, "_reaper_loop", lambda self: None)

    def fake_judge(**kwargs) -> EvalReport:
        judged.append(kwargs["submission_id"])
        return EvalReport(
            task_id=kwargs["task_spec"].task_id,
            submission_id=kwargs["submission_id"],
            total_tests=1,
            passed=1,
            pass_rate=1.0,
            score=7.0,
            valid=True,
            summary="measured",
        )

    monkeypatch.setattr(judge_server, "judge_submission", fake_judge)
    runtime = tmp_path / "runtime"
    (runtime / "bin").mkdir(parents=True)
    (runtime / "bin" / "codex").write_text("#!/bin/sh\n")
    (runtime / "env.sh").write_text("export PATH=/opt/agent-runtime/bin:$PATH\n")
    (runtime / ".sforge-agent-runtime.json").write_text(
        _json.dumps({"schema_version": 1, "kind": "sforge-agent-runtime", "agent": "codex",
                     "cli": "codex", "version": "9.9.9"})
    )

    def make_state(**overrides) -> JudgeState:
        tasks_dir = tmp_path / "task" / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        policy = tmp_path / "task" / "llm_judge" / "POLICY.md"
        policy.parent.mkdir(exist_ok=True)
        policy.write_text(
            "# Policy\n\n## What the task requires\n\nTriton.\n\n## Reject\n\nSDPA.\n\n"
            "## Allow -- never reject for these\n\nAnything else.\n"
        )
        review = LLMJudgeConfig(
            mode="enforce",
            engine="agent",
            agent_runtime_dir=runtime,
            api_base_url="http://model.internal:8000/v1",
            model="judge-model",
            network="platform",
            cache=False,
            **overrides,
        )
        state = JudgeState(
            SForgeConfig(log_dir=tmp_path / "logs", tasks_dir=tasks_dir, llm_judge=review)
        )
        state.tasks = {TASK.task_id: TASK}
        return state

    def engine(replies: list[str]):
        return ApiReviewEngine(FakeClient(replies))

    return make_state, engine, judged, tmp_path


@pytest.mark.parametrize(("on_error", "judged_count"), [("admit", 1), ("block", 0)])
def test_unconfigured_platform_review_fails_per_on_error(
    platform_harness, on_error: str, judged_count: int
) -> None:
    make_state, _, judged, _ = platform_harness
    state = make_state(on_error=on_error)
    assert state.integrity_reviewer is None

    submission_id = state.submit(TASK.task_id, make_archive(SEED), run_id="run", round="agent-1")
    record = _wait(state, submission_id)

    review = record["integrity_review"]
    assert review.status == "error" and "no integrity review engine" in review.reason
    assert len(judged) == judged_count


def test_configured_platform_review_uses_the_run_baseline(platform_harness) -> None:
    from sforge.harness.llm_judge.bundle import ArchiveBaseline

    make_state, engine, judged, tmp_path = platform_harness
    state = make_state()
    state.configure_integrity_review(engine([VIOLATION]))
    baseline = ArchiveBaseline(make_archive(SEED), {}, source="initial-checkpoint")
    state.set_run_baseline("run", baseline)
    assert state._baseline_provider("run") is baseline

    submission_id = state.submit(TASK.task_id, make_archive(CHEAT), run_id="run", round="agent-1")
    record = _wait(state, submission_id)

    assert judged == []
    assert record["report"]["metrics"]["evaluation_status"] == "integrity_rejected"
    assert record["integrity_review"].status == "violation"
