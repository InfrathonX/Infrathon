from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from _llm_judge_fixtures import (
    CWD,
    SEED,
    make_archive,
    make_task,
    write_documents,
    write_tree,
)
from sforge.harness.llm_judge.bundle import (
    ROLE_INACTIVE,
    ROLE_NOTES,
    ROLE_SELECTED,
    ROLE_SHARED,
    BundleError,
    HostDirectoryBaseline,
    NoBaseline,
    build_bundle,
)
from sforge.harness.llm_judge.config import (
    DEFAULT_MODEL,
    LLMJudgeConfig,
    LLMJudgeConfigError,
    load_llm_judge_config,
)
from sforge.harness.llm_judge.policy import (
    PolicyError,
    load_task_policy,
    parse_policy_text,
)


def _policy(task_id: str = "llm_judge_test_task", text: str = ""):
    return parse_policy_text(task_id, text, source=Path("POLICY.md"))


def _bundle(tmp_path: Path, files: dict, **kwargs):
    task = kwargs.pop("task", None) or make_task()
    policy = kwargs.pop("policy", None) or _policy()
    baseline = HostDirectoryBaseline(
        write_tree(tmp_path / "seed", SEED),
        documents_root=write_documents(tmp_path / "docs"),
    ).snapshot(task, policy)
    return build_bundle(task, make_archive(files), policy=policy, baseline=baseline, **kwargs)


def test_submission_side_is_exactly_what_the_judge_receives(tmp_path: Path) -> None:
    bundle = _bundle(
        tmp_path,
        {
            **SEED,
            "candidates/kernel_v002/submission.py": "x = 1\n",
            "tools/helper.sh": "echo hi\n",
            "unlisted.txt": "outside submit_paths\n",
        },
    )

    paths = {item.path for item in bundle.files}
    assert "tools/helper.sh" not in paths
    assert "unlisted.txt" not in paths
    assert bundle.dropped_paths == ["tools/helper.sh", "unlisted.txt"]


def test_statuses_and_roles_follow_baseline_and_selector(tmp_path: Path) -> None:
    files = dict(SEED)
    files["FINAL_VERSION"] = "2\n"
    files["notes.md"] = "# notes\nv2 is fastest\n"
    files["candidates/kernel_v002/submission.py"] = "import triton\n"
    del files["candidates/kernel_v001/submission.py"]

    bundle = _bundle(tmp_path, files)
    by_path = {item.path: item for item in bundle.files}

    assert by_path["candidates/kernel_v002/submission.py"].status == "added"
    assert by_path["candidates/kernel_v002/submission.py"].role == ROLE_SELECTED
    assert by_path["candidates/kernel_v001/submission.py"].status == "absent"
    assert by_path["candidates/kernel_v001/submission.py"].role == ROLE_INACTIVE
    assert by_path["FINAL_VERSION"].status == "modified"
    assert by_path["FINAL_VERSION"].role == ROLE_SHARED
    assert by_path["notes.md"].role == ROLE_NOTES
    assert bundle.selection is not None
    assert bundle.selection.value == 2
    assert bundle.selection.selected_prefix == "candidates/kernel_v002/"
    assert bundle.baseline_source.startswith("host-directory:")
    assert [path for path, _ in bundle.instruction_documents] == [
        f"{CWD}/AGENT_TASK.md"
    ]


def test_candidate_version_override_selects_another_implementation(
    tmp_path: Path,
) -> None:
    files = {**SEED, "candidates/kernel_v002/submission.py": "y = 2\n"}

    bundle = _bundle(tmp_path, files, candidate_version=2)

    assert bundle.selection is not None and bundle.selection.from_override
    roles = {item.path: item.role for item in bundle.files}
    assert roles["candidates/kernel_v002/submission.py"] == ROLE_SELECTED
    assert roles["candidates/kernel_v001/submission.py"] == ROLE_INACTIVE


def test_binary_files_are_hashed_not_decoded(tmp_path: Path) -> None:
    bundle = _bundle(
        tmp_path,
        {**SEED, "candidates/kernel_v001/table.bin": b"\x00\x01\x02binary"},
    )
    item = bundle.file("candidates/kernel_v001/table.bin")

    assert item is not None and item.submission is not None
    assert item.submission.binary and item.submission.text is None
    assert len(item.submission.sha256) == 64


def test_digest_ignores_tar_metadata(tmp_path: Path) -> None:
    task = make_task()
    policy = _policy()
    baseline = NoBaseline("test").snapshot(task, policy)
    first = build_bundle(
        task, make_archive(SEED, mtime=1), policy=policy, baseline=baseline
    )
    second = build_bundle(
        task, make_archive(SEED, mtime=99999), policy=policy, baseline=baseline
    )
    changed = build_bundle(
        task,
        make_archive({**SEED, "notes.md": "different\n"}),
        policy=policy,
        baseline=baseline,
    )

    assert first.digest() == second.digest()
    assert first.digest() != changed.digest()
    assert first.baseline_source == "unavailable:test"


def test_unsafe_archives_cannot_be_reviewed(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        info = tarfile.TarInfo("candidates/link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        bundle.addfile(info)
    task = make_task()
    policy = _policy()

    with pytest.raises(BundleError):
        build_bundle(
            task,
            buffer.getvalue(),
            policy=policy,
            baseline=NoBaseline("test").snapshot(task, policy),
        )


def test_policy_front_matter_and_review_exclude(tmp_path: Path) -> None:
    policy = _policy(
        text=(
            "---\n"
            "review_exclude: ['*.bin']\n"
            f"instruction_files: ['{CWD}/EXTRA.md']\n"
            "---\n"
            "## Required scope\nTriton only.\n"
        )
    )
    assert policy.instruction_files == (f"{CWD}/EXTRA.md",)
    assert policy.body.startswith("## Required scope")

    bundle = _bundle(
        tmp_path,
        {**SEED, "candidates/kernel_v001/table.bin": b"\x00"},
        policy=policy,
    )
    item = bundle.file("candidates/kernel_v001/table.bin")
    assert item is not None and item.review_excluded


@pytest.mark.parametrize(
    "text",
    [
        "---\nunknown_key: 1\n---\nbody\n",
        "---\ninstruction_files: ['relative/path.md']\n---\nbody\n",
        "---\nenabled: maybe\n---\nbody\n",
        "---\nnot terminated\n",
    ],
)
def test_malformed_policies_fail_loudly(text: str) -> None:
    with pytest.raises(PolicyError):
        _policy(text=text)


def test_policy_lookup_order(tmp_path: Path) -> None:
    tasks_dir = tmp_path / "task" / "tasks"
    tasks_dir.mkdir(parents=True)
    task_root_policy = tmp_path / "task" / "llm_judge" / "POLICY.md"
    task_root_policy.parent.mkdir()
    task_root_policy.write_text("generic task policy\n")

    assert load_task_policy("t", tasks_dir=tasks_dir).body == "generic task policy\n"

    (task_root_policy.parent / "t.md").write_text("task-id policy\n")
    assert load_task_policy("t", tasks_dir=tasks_dir).body == "task-id policy\n"

    override = tmp_path / "override"
    override.mkdir()
    (override / "t.md").write_text("override policy\n")
    assert (
        load_task_policy("t", tasks_dir=tasks_dir, policy_dir=override).body
        == "override policy\n"
    )

    empty = load_task_policy("other", tasks_dir=tmp_path / "missing")
    assert empty.source is None and not empty.has_task_rules


def test_config_from_environment(tmp_path: Path) -> None:
    key_file = tmp_path / "key"
    key_file.write_text("secret-value\n")
    config = load_llm_judge_config(
        {
            "SFORGE_LLM_JUDGE_MODE": "enforce",
            "SFORGE_LLM_JUDGE_ENGINE": "api",
            "SFORGE_LLM_JUDGE_API_BASE_URL": "http://127.0.0.1:9/v1",
            "SFORGE_LLM_JUDGE_API_KEY_FILE": str(key_file),
            "SFORGE_LLM_JUDGE_MODEL": "judge-model",
            "SFORGE_LLM_JUDGE_KINDS": "agent, public",
            "SFORGE_LLM_JUDGE_EXTRA_BODY": '{"enable_thinking": true}',
            "SFORGE_LLM_JUDGE_VOTES": "3",
            "SFORGE_LLM_JUDGE_ANNOUNCE": "false",
        }
    )

    assert config.enforcing and config.covers("public") and not config.covers("auto")
    assert config.extra_body == {"enable_thinking": True}
    assert config.votes == 3 and not config.announce
    assert config.read_api_key() == "secret-value"
    assert "secret-value" not in repr(config)
    assert "secret-value" not in repr(config.identity())


@pytest.mark.parametrize(
    "env",
    [
        {"SFORGE_LLM_JUDGE_MODE": "sometimes"},
        {"SFORGE_LLM_JUDGE_MODE": "enforce", "SFORGE_LLM_JUDGE_MODEL": "m"},
        {"SFORGE_LLM_JUDGE_MODE": "shadow", "SFORGE_LLM_JUDGE_ENGINE": "api"},
        {
            "SFORGE_LLM_JUDGE_MODE": "enforce",
            "SFORGE_LLM_JUDGE_MODEL": "m",
            "SFORGE_LLM_JUDGE_CODEX_AUTH_JSON": "/auth.json",
        },
        {
            "SFORGE_LLM_JUDGE_MODE": "enforce",
            "SFORGE_LLM_JUDGE_MODEL": "m",
            "SFORGE_LLM_JUDGE_AGENT_RUNTIME_DIR": "/runtime",
        },
        {"SFORGE_LLM_JUDGE_KINDS": "agent,bogus"},
        {"SFORGE_LLM_JUDGE_EXTRA_BODY": "[1, 2]"},
        {"SFORGE_LLM_JUDGE_MIN_CONFIDENCE": "1.5"},
    ],
)
def test_config_rejects_unusable_settings(env: dict[str, str]) -> None:
    with pytest.raises(LLMJudgeConfigError):
        load_llm_judge_config(env).check_ready()


def test_loaded_config_enforces_with_the_runs_codex_by_default() -> None:
    unnamed = load_llm_judge_config(
        {
            "SFORGE_AGENT_RUNTIME_DIR": "/runtimes/codex",
            "CODEX_AUTH_JSON_PATH": "/home/me/.codex/auth.json",
        }
    )
    assert DEFAULT_MODEL is None and unnamed.enforcing and unnamed.model is None
    with pytest.raises(LLMJudgeConfigError, match="SFORGE_LLM_JUDGE_MODEL is required"):
        unnamed.check_ready()

    config = load_llm_judge_config(
        {
            "SFORGE_AGENT_RUNTIME_DIR": "/runtimes/codex",
            "CODEX_AUTH_JSON_PATH": "/home/me/.codex/auth.json",
            "SFORGE_LLM_JUDGE_MODEL": "reviewer-model",
        }
    )
    config.check_ready()
    assert config.enforcing and config.covers("agent") and config.covers("auto")
    assert config.model == "reviewer-model"
    assert config.agent_runtime_dir == Path("/runtimes/codex")
    assert config.codex_auth_json == Path("/home/me/.codex/auth.json")

    # Explicit review settings win over the run's.
    own = load_llm_judge_config(
        {
            "SFORGE_AGENT_RUNTIME_DIR": "/runtimes/codex",
            "SFORGE_LLM_JUDGE_AGENT_RUNTIME_DIR": "/runtimes/review",
            "SFORGE_LLM_JUDGE_CODEX_AUTH_JSON": "/review/auth.json",
            "CODEX_AUTH_JSON_PATH": "/home/me/.codex/auth.json",
        },
        fallbacks={"agent_runtime_dir": Path("/runtimes/cli")},
    )
    assert own.agent_runtime_dir == Path("/runtimes/review")
    assert own.codex_auth_json == Path("/review/auth.json")


def test_loaded_config_without_a_runtime_fails_only_when_used() -> None:
    config = load_llm_judge_config({})
    assert config.enforcing
    with pytest.raises(LLMJudgeConfigError):
        config.check_ready()
    off = load_llm_judge_config({"SFORGE_LLM_JUDGE_MODE": "off"})
    off.check_ready()
    assert not off.enabled


def test_in_process_default_config_is_off() -> None:
    config = LLMJudgeConfig()
    assert not config.enabled and not config.covers("agent")
