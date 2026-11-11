"""`sforge eval` reviews the archive before the GPU Judge, like `sforge serve`."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from sforge import cli
from sforge.harness.config import SForgeConfig
from sforge.harness.llm_judge.config import LLMJudgeConfig
from sforge.harness.llm_judge.reviewer import (
    STATUS_ERROR,
    STATUS_PASS,
    STATUS_VIOLATION,
    IntegrityReview,
)

TASK_ID = "task"


def _task():
    return SimpleNamespace(
        task_id=TASK_ID,
        portable=None,
        judge=SimpleNamespace(rescale=None),
    )


def _args(archive, **overrides):
    values = dict(
        archive=str(archive), run_id="ev", timeout=None, silent=True, json=False,
        standalone_final_handoff=False, handoff_source_run_id=None,
        final_round_id=None, skip_integrity_review=False,
    )
    return SimpleNamespace(**{**values, **overrides})


def _review(status, *, mode="enforce", on_error="admit"):
    blocking = (mode == "enforce") and (
        status == STATUS_VIOLATION or (status == STATUS_ERROR and on_error == "block")
    )
    return IntegrityReview(
        review_id="r1", task_id=TASK_ID, mode=mode, status=status,
        blocking=blocking, reason=f"{status} for the test",
        feedback_to_agent="restore the odds-ratio gradient" if status == STATUS_VIOLATION else None,
        rules=["complete objective"] if status == STATUS_VIOLATION else [],
    )


class _Reviewer:
    def __init__(self, review):
        self.review_result = review
        self.calls = []

    def validate_tasks(self, tasks):
        assert [task.task_id for task in tasks] == [TASK_ID]

    def review(self, *, task_spec, archive, baseline, log_dir):
        self.calls.append((task_spec.task_id, archive, log_dir))
        return self.review_result


class _Egress:
    closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def harness(monkeypatch, tmp_path):
    archive = tmp_path / "final.tar.gz"
    archive.write_bytes(b"final archive bytes")
    state = SimpleNamespace(judged=[], printed=[], reviewer=None, egress=_Egress())

    def setup(llm_judge: LLMJudgeConfig, review=None):
        config = SForgeConfig(log_dir=tmp_path, llm_judge=llm_judge)
        monkeypatch.setattr(cli, "_make_config", lambda args: config)
        monkeypatch.setattr(cli, "_resolve_task", lambda args, config: _task())
        monkeypatch.setattr(
            cli, "_apply_resource_profile", lambda config, task, roles: (config, {})
        )
        monkeypatch.setattr(
            cli,
            "create_backend_from_config",
            lambda config: SimpleNamespace(backend_name="docker", client=object()),
        )
        state.reviewer = _Reviewer(review)
        monkeypatch.setattr(
            cli,
            "_build_integrity_reviewer",
            lambda config, llm_config, backend, cache_dir: (state.reviewer, state.egress),
        )
        monkeypatch.setattr(
            cli, "judge_submission", lambda **kwargs: state.judged.append(kwargs) or "judged"
        )
        monkeypatch.setattr(
            cli, "_print_eval_report",
            lambda report, include_json: state.printed.append(report),
        )
        return archive

    state.setup = setup
    state.submission_dir = tmp_path / "runs/ev" / TASK_ID / "submissions/manual-1"
    return state


def _ready(**overrides):
    settings = dict(
        mode="enforce", model="reviewer", agent_runtime_dir="/runtime",
        codex_auth_json="/auth.json",
    )
    return LLMJudgeConfig(**{**settings, **overrides})


def test_violation_stops_the_eval_before_the_gpu_judge(harness):
    archive = harness.setup(_ready(), _review(STATUS_VIOLATION))
    cli.cmd_eval(_args(archive))
    assert harness.judged == []
    assert harness.egress.closed
    (task_id, reviewed, log_dir), = harness.reviewer.calls
    assert reviewed == archive.read_bytes() and log_dir == harness.submission_dir
    report = json.loads((harness.submission_dir / "report.json").read_text())
    assert report["valid"] is False and report["score"] is None
    assert report["metrics"]["evaluation_status"] == "integrity_rejected"
    assert (harness.submission_dir / "submission.tar.gz").read_bytes() == archive.read_bytes()


def test_pass_goes_on_to_the_gpu_judge(harness):
    archive = harness.setup(_ready(), _review(STATUS_PASS))
    cli.cmd_eval(_args(archive))
    assert len(harness.judged) == 1
    assert harness.judged[0]["archive"] == archive.read_bytes()
    assert harness.printed == ["judged"]


def test_shadow_violation_is_recorded_but_still_judged(harness):
    archive = harness.setup(_ready(mode="shadow"), _review(STATUS_VIOLATION, mode="shadow"))
    cli.cmd_eval(_args(archive))
    assert len(harness.judged) == 1


@pytest.mark.parametrize("on_error, judged", [("admit", 1), ("block", 0)])
def test_review_errors_follow_the_error_policy(harness, on_error, judged):
    archive = harness.setup(
        _ready(on_error=on_error), _review(STATUS_ERROR, on_error=on_error)
    )
    cli.cmd_eval(_args(archive))
    assert len(harness.judged) == judged
    if not judged:
        report = json.loads((harness.submission_dir / "report.json").read_text())
        assert report["metrics"]["evaluation_status"] == "infrastructure_error"


def test_review_off_warns_and_judges(harness, capsys):
    archive = harness.setup(LLMJudgeConfig(mode="off"))
    cli.cmd_eval(_args(archive))
    assert len(harness.judged) == 1
    assert harness.reviewer.calls == []
    assert "unreviewed" in capsys.readouterr().err


def test_agent_kind_not_covered_skips_the_review(harness):
    archive = harness.setup(_ready(kinds=("auto",)))
    cli.cmd_eval(_args(archive))
    assert len(harness.judged) == 1
    assert harness.reviewer.calls == []


def test_an_enabled_review_that_cannot_run_fails_closed(harness, capsys):
    archive = harness.setup(LLMJudgeConfig(mode="enforce", model="reviewer"))
    with pytest.raises(SystemExit) as exit_info:
        cli.cmd_eval(_args(archive))
    assert exit_info.value.code == 2
    assert harness.judged == []
    assert "--skip-integrity-review" in capsys.readouterr().err


def test_skip_flag_benchmarks_without_a_review(harness, capsys):
    archive = harness.setup(_ready(), _review(STATUS_VIOLATION))
    cli.cmd_eval(_args(archive, skip_integrity_review=True))
    assert len(harness.judged) == 1
    assert harness.reviewer.calls == []
    assert "unreviewed" in capsys.readouterr().err
