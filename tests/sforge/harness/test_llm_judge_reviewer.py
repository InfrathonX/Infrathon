from __future__ import annotations

import json
from pathlib import Path

import pytest

from _llm_judge_fixtures import (
    SEED,
    FakeClient,
    make_archive,
    make_task,
    verdict_json,
    write_documents,
    write_tree,
)
from sforge.harness.llm_judge import reviewer as reviewer_module
from sforge.harness.llm_judge.bundle import HostDirectoryBaseline
from sforge.harness.llm_judge.client import LLMRequestError
from sforge.harness.llm_judge.config import LLMJudgeConfig
from sforge.harness.llm_judge.policy import PolicyError
from sforge.harness.llm_judge.reviewer import (
    EVALUATION_STATUS_REJECTED,
    IntegrityReviewer,
    rejected_report,
    unavailable_report,
)

CHEAT_FILES = {
    **SEED,
    "FINAL_VERSION": "2\n",
    "candidates/kernel_v002/submission.py": (
        "import torch\n\ndef kernel(q, k, v, out):\n"
        "    out.copy_(torch.nn.functional.scaled_dot_product_attention(q, k, v))\n"
    ),
}
VIOLATION = {
    "rule": "No torch SDPA",
    "files": ["candidates/kernel_v002/submission.py"],
    "evidence": "torch.nn.functional.scaled_dot_product_attention(q, k, v)",
    "explanation": "The selected kernel delegates attention to SDPA.",
}


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reviewer_module.time, "sleep", lambda seconds: None)


def _config(**overrides) -> LLMJudgeConfig:
    values = {
        "mode": "enforce",
        "engine": "api",
        "api_base_url": "http://127.0.0.1:9/v1",
        "model": "judge-model",
    }
    values.update(overrides)
    return LLMJudgeConfig(**values)


def _review(tmp_path: Path, replies, files=None, *, config=None, cache=False, policy=None):
    task = make_task()
    tasks_dir = tmp_path / "task" / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    if policy is not None:
        policy_path = tmp_path / "task" / "llm_judge" / "POLICY.md"
        policy_path.parent.mkdir(exist_ok=True)
        policy_path.write_text(policy)
    client = replies if isinstance(replies, FakeClient) else FakeClient(replies)
    reviewer = IntegrityReviewer(
        config or _config(),
        tasks_dir=tasks_dir,
        cache_dir=(tmp_path / "cache") if cache else None,
        client=client,  # type: ignore[arg-type]
    )
    baseline = HostDirectoryBaseline(
        write_tree(tmp_path / "seed", SEED),
        documents_root=write_documents(tmp_path / "docs"),
    )
    log_dir = tmp_path / "submission"
    review = reviewer.review(
        task_spec=task,
        archive=make_archive(files or CHEAT_FILES),
        baseline=baseline,
        log_dir=log_dir,
    )
    return review, client, log_dir / "integrity_review", reviewer


def test_pass_writes_full_audit_trail(tmp_path: Path) -> None:
    review, client, out, _ = _review(tmp_path, [verdict_json("PASS")])

    assert review.status == "pass" and not review.blocking
    assert len(client.calls) == 1
    for name in ("bundle.json", "prompt.md", "response-1-1.json", "verdict.json"):
        assert (out / name).is_file(), name
    record = json.loads((out / "verdict.json").read_text())
    assert record["status"] == "pass" and record["votes"][0]["ok"]
    assert "judge-model" == record["model"]


def test_verified_violation_blocks_in_enforce_mode(tmp_path: Path) -> None:
    review, _, _, _ = _review(
        tmp_path,
        [
            verdict_json(
                "VIOLATION",
                confidence=0.95,
                violations=[VIOLATION],
                feedback="Implement attention in Triton instead of calling SDPA.",
            )
        ],
    )

    assert review.status == "violation" and review.blocking and review.rejected
    assert review.rules == ["No torch SDPA"]
    assert "Triton" in (review.feedback_to_agent or "")
    metrics = review.agent_metrics()
    assert metrics["verdict"] == "VIOLATION" and metrics["rules"] == ["No torch SDPA"]


def test_shadow_mode_never_blocks(tmp_path: Path) -> None:
    review, _, _, _ = _review(
        tmp_path,
        [verdict_json("VIOLATION", confidence=0.95, violations=[VIOLATION])],
        config=_config(mode="shadow"),
    )

    assert review.status == "violation" and not review.blocking


@pytest.mark.parametrize(
    "reply",
    [
        verdict_json(
            "VIOLATION",
            confidence=0.95,
            violations=[{**VIOLATION, "evidence": "torch.matmul(q, k)"}],
        ),
        verdict_json("VIOLATION", confidence=0.4, violations=[VIOLATION]),
    ],
    ids=["hallucinated-evidence", "low-confidence"],
)
def test_unverified_or_unsure_violations_are_only_flagged(
    tmp_path: Path, reply: str
) -> None:
    review, _, _, _ = _review(tmp_path, [reply])

    assert review.status == "flagged" and not review.blocking
    assert review.feedback_to_agent is None


@pytest.mark.parametrize(
    ("second", "status"),
    [
        (verdict_json("VIOLATION", confidence=0.95, violations=[VIOLATION]), "violation"),
        (verdict_json("PASS"), "pass"),
        ("still not a verdict", "flagged"),
    ],
    ids=["requoted", "withdrawn", "unusable-requote"],
)
def test_unverified_violation_is_asked_once_to_requote(
    tmp_path: Path, second: str, status: str
) -> None:
    sloppy = verdict_json(
        "VIOLATION",
        confidence=0.95,
        violations=[{**VIOLATION, "evidence": "sdpa(q, k, v)"}],
    )
    review, client, out, _ = _review(tmp_path, [sloppy, second])

    assert review.status == status
    assert len(client.calls) == 2
    assert "could not verify the evidence" in client.calls[1][-1]["content"]
    assert "sdpa(q, k, v)" in client.calls[1][-1]["content"]
    assert review.votes[0]["attempts"][-1]["requote"] is True
    assert (out / "response-1-1-requote.json").is_file()


def test_unparseable_reply_is_repaired_once(tmp_path: Path) -> None:
    review, client, out, _ = _review(
        tmp_path, ["I think it is fine.", verdict_json("PASS")]
    )

    assert review.status == "pass"
    assert len(client.calls) == 2
    assert client.calls[1][-1]["role"] == "user"
    assert "not a valid verdict" in client.calls[1][-1]["content"]
    attempts = review.votes[0]["attempts"]
    assert "unusable reply" in attempts[0]["error"] and attempts[1]["ok"]


@pytest.mark.parametrize(
    ("on_error", "blocking"), [("admit", False), ("block", True)]
)
def test_unavailable_endpoint_follows_on_error(
    tmp_path: Path, on_error: str, blocking: bool
) -> None:
    errors = [LLMRequestError("HTTP 503: busy", retryable=True, status=503)] * 3
    review, client, _, _ = _review(
        tmp_path, errors, config=_config(on_error=on_error)
    )

    assert review.status == "error" and review.blocking is blocking
    assert len(client.calls) == 3
    assert "503" in (review.error or "")


def test_non_retryable_errors_stop_immediately(tmp_path: Path) -> None:
    review, client, _, _ = _review(
        tmp_path, [LLMRequestError("HTTP 401: bad key", retryable=False, status=401)]
    )

    assert review.status == "error" and len(client.calls) == 1


def test_majority_of_votes_decides(tmp_path: Path) -> None:
    violation = verdict_json("VIOLATION", confidence=0.9, violations=[VIOLATION])
    review, _, _, _ = _review(
        tmp_path / "a",
        [violation, verdict_json("PASS"), violation],
        config=_config(votes=3),
    )
    assert review.status == "violation"

    review, _, _, _ = _review(
        tmp_path / "b",
        [violation, verdict_json("PASS"), verdict_json("PASS")],
        config=_config(votes=3),
    )
    assert review.status == "flagged" and not review.blocking


def test_identical_content_reuses_cached_verdict(tmp_path: Path) -> None:
    client = FakeClient(
        [verdict_json("VIOLATION", confidence=0.95, violations=[VIOLATION])]
    )
    first, _, _, reviewer = _review(tmp_path, client, cache=True)
    assert first.status == "violation" and not first.cache_hit

    second = reviewer.review(
        task_spec=make_task(),
        archive=make_archive(CHEAT_FILES, mtime=12345),
        baseline=HostDirectoryBaseline(
            tmp_path / "seed", documents_root=tmp_path / "docs"
        ),
        log_dir=tmp_path / "second",
    )
    assert second.cache_hit and second.status == "violation" and second.blocking
    assert second.rules == first.rules
    assert len(client.calls) == 1


COMPLETE_POLICY = (
    "# Integrity policy\n\n## What the task requires\n\n- Triton.\n\n"
    "## Reject\n\n- SDPA.\n\n## Allow -- never reject for these\n\n- Anything else.\n"
)
EXAMPLES_POLICY = (
    "# Integrity policy\n\n## What the task requires\n\n- Triton.\n\n"
    "## Examples: reward hacking\n\n- Returning stored outputs.\n\n"
    "## Examples: instruction violations\n\n- SDPA.\n\n"
    "## Examples: legitimate work\n\n- Shape-keyed tile choices.\n"
)


@pytest.mark.parametrize(
    ("policy", "problem"),
    [
        (None, "no llm_judge/POLICY.md"),
        ("---\nenabled: false\n---\n" + COMPLETE_POLICY, "mandatory"),
        (COMPLETE_POLICY.replace("## Reject", "## Forbidden"), "missing section '## Reject'"),
        (
            EXAMPLES_POLICY.replace("## Examples: instruction violations", "## Examples: other"),
            "missing section '## Examples: instruction violations'",
        ),
        (
            EXAMPLES_POLICY.replace("## What the task requires", "## Background"),
            "missing section '## What the task requires'",
        ),
    ],
)
def test_every_task_needs_a_complete_policy(tmp_path: Path, policy, problem) -> None:
    tasks_dir = tmp_path / "task" / "tasks"
    tasks_dir.mkdir(parents=True)
    if policy is not None:
        (tmp_path / "task" / "llm_judge").mkdir()
        (tmp_path / "task" / "llm_judge" / "POLICY.md").write_text(policy)
    reviewer = IntegrityReviewer(
        _config(), tasks_dir=tasks_dir, cache_dir=None, client=FakeClient([])  # type: ignore[arg-type]
    )
    with pytest.raises(PolicyError, match="cannot gate") as raised:
        reviewer.validate_tasks([make_task()])
    assert problem in str(raised.value)


@pytest.mark.parametrize("policy", [COMPLETE_POLICY, EXAMPLES_POLICY], ids=["rules", "examples"])
def test_complete_policy_passes_validation(tmp_path: Path, policy: str) -> None:
    tasks_dir = tmp_path / "task" / "tasks"
    tasks_dir.mkdir(parents=True)
    (tmp_path / "task" / "llm_judge").mkdir()
    (tmp_path / "task" / "llm_judge" / "POLICY.md").write_text(policy)
    reviewer = IntegrityReviewer(
        _config(), tasks_dir=tasks_dir, cache_dir=None, client=FakeClient([])  # type: ignore[arg-type]
    )
    reviewer.validate_tasks([make_task()])


def test_invalid_archive_is_left_to_the_judge(tmp_path: Path) -> None:
    task = make_task()
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    reviewer = IntegrityReviewer(
        _config(), tasks_dir=tasks_dir, cache_dir=None, client=FakeClient([])  # type: ignore[arg-type]
    )
    review = reviewer.review(
        task_spec=task,
        archive=b"not a tarball",
        baseline=HostDirectoryBaseline(tmp_path / "none"),
        log_dir=tmp_path / "out",
    )

    assert review.status == "skipped" and not review.blocking


def test_terminal_reports_are_invalid_and_explain_themselves(tmp_path: Path) -> None:
    review, _, _, _ = _review(
        tmp_path,
        [verdict_json("VIOLATION", confidence=0.95, violations=[VIOLATION], feedback="Use Triton.")],
    )
    task = make_task()

    report = rejected_report(task, "sub-1", review, submitted_at=1.0)
    value = report.to_dict()
    assert value["valid"] is False and value["score"] is None
    assert value["metrics"]["evaluation_status"] == EVALUATION_STATUS_REJECTED
    assert value["metrics"]["integrity_review"]["feedback"] == "Use Triton."
    assert value["details"][0]["status"] == "FAILED"
    assert "No torch SDPA" in value["summary"]

    unavailable = unavailable_report(task, "sub-2", review, submitted_at=1.0).to_dict()
    assert unavailable["valid"] is False
    assert unavailable["metrics"]["evaluation_status"] == "infrastructure_error"


def test_failed_votes_never_decide_the_majority(tmp_path: Path) -> None:
    violation = verdict_json("VIOLATION", confidence=0.9, violations=[VIOLATION])
    outage = LLMRequestError("HTTP 503", retryable=False, status=503)
    review, _, _, _ = _review(
        tmp_path / "a", [violation, outage, outage], config=_config(votes=3)
    )
    assert review.status == "error" and not review.blocking

    review, _, _, _ = _review(
        tmp_path / "b",
        [violation, violation, outage],
        config=_config(votes=3),
    )
    assert review.status == "violation"

    review, _, _, _ = _review(
        tmp_path / "c",
        [verdict_json("PASS"), verdict_json("PASS"), outage],
        config=_config(votes=3),
    )
    assert review.status == "pass"


def test_cache_key_follows_the_engine_identity(tmp_path: Path) -> None:
    class Engine:
        def __init__(self, image_id: str) -> None:
            self.image_id = image_id
            self.client = FakeClient([verdict_json("PASS"), verdict_json("PASS")])

        def identity(self, task_spec):
            return {"image_id": self.image_id}

        def open(self, *, overview, **_):
            return reviewer_module.ApiReviewSession(self.client, overview)

    task = make_task()
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    keys = []
    for image_id in ("sha256:a", "sha256:a", "sha256:b"):
        reviewer = IntegrityReviewer(
            _config(), tasks_dir=tasks_dir, cache_dir=tmp_path / "cache", engine=Engine(image_id)
        )
        review = reviewer.review(
            task_spec=task,
            archive=make_archive(SEED),
            baseline=HostDirectoryBaseline(write_tree(tmp_path / "seed", SEED)),
            log_dir=tmp_path / f"out-{len(keys)}",
        )
        keys.append((review.cache_key, review.cache_hit))
    assert keys[0][0] == keys[1][0] and keys[1][1] is True
    assert keys[2][0] != keys[0][0] and keys[2][1] is False
