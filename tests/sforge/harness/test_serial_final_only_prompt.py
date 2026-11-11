from pathlib import Path
from types import SimpleNamespace

import pytest

from sforge.harness import run_agent as runner
from sforge.harness.evolve_scripts import generate_evolve_prompt


def test_zero_submission_budget_uses_neutral_fixed_session_contract() -> None:
    prompt = generate_evolve_prompt(
        "Implement the task.",
        submit_paths=["kernels", "FINAL_VERSION"],
        internet=False,
        max_submissions=0,
        submission_cooldown=300,
    )

    assert "Fixed Work Session" in prompt
    assert "one immutable archive" in prompt
    assert "Only the captured files represent your final implementation" in prompt
    assert "`kernels`, `FINAL_VERSION`" in prompt
    assert "NO internet access" in prompt
    assert "sforge-submit" not in prompt
    assert "background process periodically" not in prompt
    lowered = prompt.lower()
    for forbidden in (
        "private",
        "judge",
        "hidden",
        "timeout boundary",
        "authoritative result",
        "reference",
        "tolerance",
    ):
        assert forbidden not in lowered
    assert "Implement the task." in prompt


def test_registered_fixed_session_defines_measurement_candidate_capture() -> None:
    prompt = generate_evolve_prompt(
        "Implement the task.",
        submit_paths=["submission"],
        max_submissions=0,
        version_saving_enabled=True,
    )

    assert "Candidate Version Capture" in prompt
    assert "sforge-save-version --label <short-hypothesis>" in prompt
    assert "Before the first score-producing public test or benchmark" in prompt
    assert "changing or restoring any submitted file" in prompt
    assert "before that implementation's first score-producing public measurement" in prompt
    assert "wait for the command to report a successful save" in prompt
    assert "do not edit submitted files between the save and test start" in prompt
    assert "If submitted files are unchanged since the last successful save" in prompt
    assert "Saving only records the submitted files" in prompt
    assert "does not run a test or produce a score" in prompt
    assert "Task-owned version files" in prompt
    assert "final filesystem state is saved automatically" in prompt
    assert "sforge-submit" not in prompt

    capture = prompt.split("### Candidate Version Capture", 1)[1].split(
        "\n### ", 1
    )[0]
    assert sum(line.startswith("- ") for line in capture.splitlines()) == 4
    for internal_term in (
        "normalized submitted tree",
        "chronological save event",
        "candidate/version identity",
        "does not create a new `vN`",
    ):
        assert internal_term not in capture


def test_nonzero_submission_budget_keeps_iterative_contract() -> None:
    prompt = generate_evolve_prompt(
        "Implement the task.",
        max_submissions=2,
        submission_cooldown=30,
    )

    assert "Iterative Evaluation Mode" in prompt
    assert "sforge-submit --details" in prompt
    assert "2 total" in prompt


def test_registered_iterative_prompt_records_candidate_revisits_as_attempts() -> None:
    prompt = generate_evolve_prompt(
        "Implement the task.",
        max_submissions=2,
        version_saving_enabled=True,
    )

    assert "sforge-save-version --label <short-hypothesis>" in prompt
    assert "changing or restoring any submitted file" in prompt
    assert "save again" in prompt
    assert "If submitted files are unchanged since the last successful save" in prompt
    assert "SForge saves accepted evaluation submissions automatically" in prompt
    assert "do not save solely for `sforge-submit`" in prompt
    assert "Save exactly once per distinct candidate" not in prompt

    capture = prompt.split("### Candidate Version Capture", 1)[1].split(
        "\n### ", 1
    )[0]
    assert sum(line.startswith("- ") for line in capture.splitlines()) == 5


@pytest.mark.parametrize(
    ("max_submissions", "expected_bullets"),
    [(0, 4), (2, 5)],
)
def test_candidate_capture_skips_unmeasured_notes_only_states(
    max_submissions: int, expected_bullets: int
) -> None:
    prompt = generate_evolve_prompt(
        "Implement the task.",
        max_submissions=max_submissions,
        version_saving_enabled=True,
    )

    capture = prompt.split("### Candidate Version Capture", 1)[1].split(
        "\n### ", 1
    )[0]
    assert "only update notes or evidence such as `SELFTEST.md`" in capture
    assert (
        "do not save that state unless you will run a score-producing public "
        "measurement on that exact state"
    ) in capture
    assert "final filesystem state is saved automatically" in capture
    assert sum(line.startswith("- ") for line in capture.splitlines()) == expected_bullets


def test_fixed_strategy_does_not_reintroduce_rollback_checkpoints() -> None:
    prompt = generate_evolve_prompt(
        "Before risky optimizations, make a restorable rollback checkpoint.",
        max_submissions=0,
        version_saving_enabled=True,
    )

    assert "before attempting risky optimizations" not in prompt
    rule = (
        "Use sforge-save-version only for the Work-local public-measurement cases "
        "above, not as a rollback checkpoint; this rule overrides conflicting "
        "task text."
    )
    assert rule in prompt
    assert prompt.index(rule) < prompt.index("Before risky optimizations")


def test_fixed_session_exposes_no_submission_service_credentials() -> None:
    env = {"NO_PROXY": "localhost", "no_proxy": "localhost"}
    runner._configure_submission_service_env(
        env,
        judge_url="http://judge.internal:8123",
        session_token="secret",
        max_submissions=0,
        version_saving_enabled=False,
    )
    assert env == {"NO_PROXY": "localhost", "no_proxy": "localhost"}


def test_interactive_session_keeps_submission_service_credentials() -> None:
    env: dict[str, str] = {}
    runner._configure_submission_service_env(
        env,
        judge_url="http://judge.internal:8123",
        session_token="secret",
        max_submissions=1,
        version_saving_enabled=False,
    )
    assert env["SFORGE_JUDGE_URL"] == "http://judge.internal:8123"
    assert env["SFORGE_TOKEN"] == "secret"
    assert env["NO_PROXY"] == "judge.internal"
    assert env["no_proxy"] == "judge.internal"


def test_registered_fixed_session_keeps_checkpoint_service_credentials() -> None:
    env: dict[str, str] = {}
    runner._configure_submission_service_env(
        env,
        judge_url="http://judge.internal:8123",
        session_token="secret",
        max_submissions=0,
        version_saving_enabled=True,
    )

    assert env["SFORGE_JUDGE_URL"] == "http://judge.internal:8123"
    assert env["SFORGE_TOKEN"] == "secret"


def test_fixed_session_selects_checkpoint_only_registration_token() -> None:
    registration = {"token": "evaluation", "checkpoint_token": "checkpoint"}

    assert (
        runner._select_work_session_token(
            registration,
            max_submissions=0,
            version_saving_enabled=True,
        )
        == "checkpoint"
    )
    assert (
        runner._select_work_session_token(
            registration,
            max_submissions=1,
            version_saving_enabled=True,
        )
        == "evaluation"
    )

    with pytest.raises(RuntimeError, match="checkpoint-only token"):
        runner._select_work_session_token(
            {"token": "evaluation"},
            max_submissions=0,
            version_saving_enabled=True,
        )


def test_runner_rejects_a_mismatched_judge_version_store() -> None:
    store = SimpleNamespace(store_id="expected-store")

    runner._validate_version_store_registration(
        {
            "version_saving_enabled": True,
            "version_store_id": "expected-store",
        },
        store,
    )

    with pytest.raises(RuntimeError, match="version store mismatch"):
        runner._validate_version_store_registration(
            {
                "version_saving_enabled": True,
                "version_store_id": "other-store",
            },
            store,
        )


def test_fixed_session_omits_submit_tool(monkeypatch, tmp_path: Path) -> None:
    copied: list[str] = []

    def fake_copy(*args, **kwargs):
        copied.append(kwargs.get("description", ""))

    class Agent:
        def install_stop_hook(self, *args, **kwargs):
            copied.append("stop-hook")

    monkeypatch.setattr(runner, "copy_container_file", fake_copy)
    runner._install_tools(
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        Agent(),
        tmp_path,
        SimpleNamespace(info=lambda *args, **kwargs: None),
        install_submit_tool=False,
        install_save_version_tool=False,
    )
    assert copied == ["stop-hook"]


def test_fixed_session_installs_only_save_version_tool(monkeypatch, tmp_path: Path) -> None:
    copied: list[str] = []

    def fake_copy(*args, **kwargs):
        copied.append(kwargs.get("description", ""))

    class Agent:
        def install_stop_hook(self, *args, **kwargs):
            copied.append("stop-hook")

    monkeypatch.setattr(runner, "copy_container_file", fake_copy)
    runner._install_tools(
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        Agent(),
        tmp_path,
        SimpleNamespace(info=lambda *args, **kwargs: None),
        install_submit_tool=False,
        install_save_version_tool=True,
    )

    assert copied == ["sforge-save-version", "stop-hook"]
