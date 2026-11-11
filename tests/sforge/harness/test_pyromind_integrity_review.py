"""PyroMind wiring of the mandatory LLM integrity review (no remote calls)."""

from __future__ import annotations

import gzip
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sforge.harness.agent_runtime import load_agent_runtime
from sforge.harness.backend.pyromind_backend import PyroMindBackend
from sforge.harness.llm_judge.config import LLMJudgeConfigError
from sforge.pyromind import integrity_review as review

DIGEST = "registry.example/judge@sha256:" + "a" * 64


@pytest.fixture(autouse=True)
def _host_reviewer_model(monkeypatch):
    """The reviewer model a host names in launch/mlsysforge.env (no built-in default)."""
    monkeypatch.setenv("SFORGE_LLM_JUDGE_MODEL", "reviewer-model")


def _runtime(tmp_path: Path) -> Path:
    root = tmp_path / "codex-runtime"
    (root / "bin").mkdir(parents=True)
    (root / "lib").mkdir()
    (root / "lib" / "codex.js").write_text("console.log('codex')\n")
    (root / "bin" / "codex").symlink_to("../lib/codex.js")
    (root / "env.sh").write_text("export PATH=/opt/agent-runtime/bin:$PATH\n")
    (root / ".sforge-agent-runtime.json").write_text(json.dumps(
        {"schema_version": 1, "kind": "sforge-agent-runtime", "agent": "codex",
         "cli": "codex", "version": "9.9.9"}))
    return root


def _model(tmp_path: Path, mode: str) -> dict:
    model = {"MODEL": "worker-model", "REASONING_EFFORT": "medium",
             "SFORGE_AGENT_RUNTIME_DIR": str(_runtime(tmp_path)), "AUTH_MODE": mode}
    if mode == "chatgpt":
        model["CODEX_AUTH_JSON_PATH"] = str(tmp_path / "auth.json")
    else:
        model["MODEL_API_BASE_URL"] = "http://model.internal:8000/v1"
        model["MODEL_API_KEY_FILE"] = str(tmp_path / "model.key")
    return model


def test_worker_chatgpt_credentials_keep_the_hosts_reviewer(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SFORGE_LLM_JUDGE_MODEL", "reviewer-model")
    config = review.review_config(_model(tmp_path, "chatgpt"), mode="enforce", explicit=False,
                                  resources=review.DEFAULT_REVIEW_RESOURCES)
    assert config.enforcing and config.model == "reviewer-model" and config.reasoning_effort == "high"
    assert config.codex_auth_json == tmp_path / "auth.json" and config.api_base_url is None
    assert (config.network, config.cpu_limit, config.mem_limit) == ("platform", 2, "4Gi")
    assert config.concurrency == 1 and config.image == "judge"


def test_explicit_api_key_reviewer_keeps_its_model(tmp_path: Path) -> None:
    config = review.review_config(_model(tmp_path, "api-key"), mode="shadow", explicit=True,
                                  resources=review.DEFAULT_REVIEW_RESOURCES)
    assert config.mode == "shadow" and config.model == "worker-model"
    assert config.api_base_url == "http://model.internal:8000/v1"
    assert config.api_key_file == tmp_path / "model.key" and config.codex_auth_json is None


@pytest.mark.parametrize("auth", ["chatgpt", "api-key"])
def test_profile_reviewer_override_keeps_worker_credentials_and_runtime(tmp_path, auth):
    model = _model(tmp_path, auth)
    config = review.review_config(
        model, mode="enforce", explicit=False, resources=review.DEFAULT_REVIEW_RESOURCES,
        implicit_model="gpt-5.5", implicit_reasoning_effort="high")
    assert (config.model, config.reasoning_effort) == ("gpt-5.5", "high")
    assert config.agent_runtime_dir == Path(model["SFORGE_AGENT_RUNTIME_DIR"])
    if auth == "chatgpt":
        assert config.codex_auth_json == Path(model["CODEX_AUTH_JSON_PATH"])
    else:
        assert config.api_base_url == model["MODEL_API_BASE_URL"]
        assert config.api_key_file == Path(model["MODEL_API_KEY_FILE"])


@pytest.mark.parametrize("auth", ["chatgpt", "api-key"])
def test_explicit_reviewer_config_takes_precedence_over_profile(tmp_path, auth):
    model = _model(tmp_path, auth)
    config = review.review_config(
        model, mode="enforce", explicit=True, resources=review.DEFAULT_REVIEW_RESOURCES,
        implicit_model="gpt-5.5", implicit_reasoning_effort="high")
    assert (config.model, config.reasoning_effort) == ("worker-model", "medium")


@pytest.mark.parametrize("value", ["", True, "bad\nmodel"])
def test_profile_reviewer_override_rejects_malformed_values(value):
    with pytest.raises(ValueError, match="review_model"):
        review.review_config({}, mode="off", explicit=False, resources={}, implicit_model=value)


def test_deployment_can_block_review_errors_without_changing_the_default(tmp_path: Path) -> None:
    model = _model(tmp_path, "chatgpt")
    kwargs = {"mode": "enforce", "explicit": False,
              "resources": review.DEFAULT_REVIEW_RESOURCES}
    assert review.review_config(model, **kwargs).on_error == "admit"
    config = review.review_config(model, **kwargs, on_error="block")
    assert config.on_error == "block"
    record = review.describe(config, resources=review.DEFAULT_REVIEW_RESOURCES,
                             judge_image=DIGEST, staged_runtime=None,
                             explicit_model_config=False)
    assert record["on_error"] == "block"


@pytest.mark.parametrize("value", ["reject", "", None, True])
def test_invalid_deployment_review_error_policy_is_not_silently_ignored(value) -> None:
    with pytest.raises(ValueError, match="review_on_error"):
        review.review_config({}, mode="off", explicit=False, resources={}, on_error=value)


def test_unsupported_reviewer_credentials_fail_before_the_run(tmp_path: Path) -> None:
    model = {**_model(tmp_path, "api-key"), "AUTH_MODE": "claude-oauth"}
    with pytest.raises(LLMJudgeConfigError, match="review-model-config"):
        review.review_config(model, mode="enforce", explicit=False,
                             resources=review.DEFAULT_REVIEW_RESOURCES)
    assert not review.review_config({}, mode="off", explicit=False, resources={}).enabled


@pytest.mark.parametrize("override", [{"cpu": 0, "memory": "4Gi"}, {"cpu": 4, "memory": "16G"},
                                      {"cpu": True, "memory": "8Gi"}, {"gpu": 1}])
def test_review_resources_stay_cpu_only_and_whole_units(override) -> None:
    deployment = SimpleNamespace(config={"review_resources": override})
    with pytest.raises(ValueError):
        review.review_resources(deployment)
    for size in ({"cpu": 8, "memory": "16Gi"}, {"cpu": 2, "memory": "8Gi"}):
        deployment = SimpleNamespace(config={"review_resources": size})
        assert review.review_resources(deployment) == {**size, "gpu": 0}


def test_runtime_archive_is_deterministic_and_keeps_links(tmp_path: Path) -> None:
    runtime = load_agent_runtime(_runtime(tmp_path))
    first, digest, size = review.runtime_archive(runtime, tmp_path / "cache-a")
    second, digest_b, _ = review.runtime_archive(runtime, tmp_path / "cache-b")
    assert digest == digest_b and first.read_bytes() == second.read_bytes()
    assert size == first.stat().st_size
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(first.read_bytes()))) as bundle:
        members = {member.name: member for member in bundle}
    assert members["bin/codex"].issym() and members["bin/codex"].linkname == "../lib/codex.js"
    assert all(member.mtime == 0 and member.uid == 0 for member in members.values())
    assert "env.sh" in members


@pytest.mark.pyromind_sdk
def test_review_sandbox_request_is_cpu_only_with_the_runtime_mounted(tmp_path: Path) -> None:
    client = Mock()
    client.create.return_value = SimpleNamespace(id="sb-0123456789ab", name="sf-x-review-1")
    staged = {"data_prefix": "sforge-assets/v2/agent-runtime/codex-9.9.9/abc/files",
              "archive_sha256": "c" * 64}
    backend = PyroMindBackend(client, event_path=tmp_path / "events.jsonl", gpu_count=0,
                              volume_mounts=[review.runtime_mount(staged)])
    config = review.review_config(_model(tmp_path, "chatgpt"), mode="enforce", explicit=False,
                                  resources=review.DEFAULT_REVIEW_RESOURCES)
    engine = review.build_review_engine(config, backend, judge_image=DIGEST,
                                        name_prefix="review", staged_runtime=staged)
    task = SimpleNamespace(judge_image_key="judge:local", work_image_key="work:local", task_id="t")
    identity = engine.identity(task)
    assert identity["image_id"] == DIGEST and identity["runtime_archive_sha256"] == "c" * 64

    backend.create_container(DIGEST, "sf-x-review-1", cpu_limit=4, mem_limit="8Gi")
    request = client.create.call_args.args[0]
    resources = request.resources.model_dump()
    assert resources["cpu"] == "4" and resources["memory"] == "8Gi"
    assert not resources.get("gpu") and not resources.get("gpu_card")
    (mount,) = request.volume_mounts
    assert mount.mount_path == "/mnt/sforge-review-runtime" and mount.read_only
    assert mount.host_path == "/workspace/" + staged["data_prefix"]

    with pytest.raises(ValueError, match="digest"):
        review.build_review_engine(config, backend, judge_image="judge:latest",
                                   name_prefix="review", staged_runtime=staged)


def test_worker_visible_history_has_no_review_audit() -> None:
    history = {"run_id": "r", "entries": [
        {"round": "agent-1", "score": 1.0, "integrity_review": {"status": "flagged"}},
        {"round": "agent-2", "score": None}]}
    visible = review.worker_visible_history(history)
    assert all("integrity_review" not in entry for entry in visible["entries"])
    assert history["entries"][0]["integrity_review"] == {"status": "flagged"}


def test_baseline_is_the_initial_checkpoint_plus_briefs_read_before_the_agent(tmp_path) -> None:
    from _llm_judge_fixtures import CWD, SEED, make_archive, make_task
    from sforge.harness.llm_judge.policy import parse_policy_text

    task = make_task()
    policy = parse_policy_text(task.task_id, "Triton only.\n", source=Path("POLICY.md"))
    store = SimpleNamespace(read_archive=lambda version: make_archive(SEED))
    initial = SimpleNamespace(version_id="checkpoint-1")

    def read(path: str) -> bytes:
        if path == f"{CWD}/AGENT_TASK.md":
            return b"Implement it in Triton."
        raise FileNotFoundError(path)

    baseline = review.capture_baseline(task, policy, store, initial, read)
    snapshot = baseline.snapshot(task, policy)
    assert snapshot.source == "pyromind-initial-checkpoint:checkpoint-1"
    assert snapshot.documents == {f"{CWD}/AGENT_TASK.md": "Implement it in Triton."}
    assert snapshot.files


def test_review_allowance_extends_the_submit_wait() -> None:
    config = SimpleNamespace(enabled=True, review_timeout_seconds=1800.0)
    assert review.review_allowance_seconds(config, 300) == 1800 + 300 + 180
    assert review.review_allowance_seconds(SimpleNamespace(enabled=False), 300) == 0
