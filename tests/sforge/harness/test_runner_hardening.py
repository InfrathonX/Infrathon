from __future__ import annotations

import importlib
import io
import json
import logging
import os
import subprocess
import sys
import tarfile
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from sforge.harness.agent.codex import CodexAgent
from sforge.harness.config import SForgeConfig


runner = importlib.import_module("sforge.harness.run_agent")


class _Logger:
    def info(self, *args, **kwargs) -> None:
        pass

    def warning(self, *args, **kwargs) -> None:
        pass

    def debug(self, *args, **kwargs) -> None:
        pass


def test_admin_secret_override_is_shared_by_host_modules() -> None:
    env = os.environ.copy()
    env["SFORGE_ADMIN_SECRET"] = "test-only-admin-secret"
    code = """
from sforge.harness.constants import ADMIN_SECRET as constants_secret
from sforge.harness.judge_server import ADMIN_SECRET as judge_secret
from sforge.harness.run_agent import ADMIN_SECRET as runner_secret
assert constants_secret == judge_secret == runner_secret
print(constants_secret)
"""
    output = subprocess.check_output(
        [sys.executable, "-c", code],
        env=env,
        text=True,
    )
    assert output.strip() == env["SFORGE_ADMIN_SECRET"]


def test_empty_admin_secret_override_is_rejected() -> None:
    env = os.environ.copy()
    env["SFORGE_ADMIN_SECRET"] = ""
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "from sforge.harness.constants import ADMIN_SECRET",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert process.returncode != 0
    assert "SFORGE_ADMIN_SECRET must not be empty" in process.stderr


def test_full_history_sends_admin_secret_only_in_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_secret = "unit-test-admin-secret"
    captured: dict = {}

    class Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"entries": [{"round": "auto-1"}]}

    def fake_get(url: str, **kwargs) -> Response:
        captured["url"] = url
        captured.update(kwargs)
        return Response()

    monkeypatch.setattr(runner, "ADMIN_SECRET", test_secret)
    monkeypatch.setattr(runner.requests, "get", fake_get)

    history = runner._fetch_full_judge_history(
        "http://judge",
        "unit-test-session-token",
    )

    assert history == {"entries": [{"round": "auto-1"}]}
    assert captured["params"] == {"token": "unit-test-session-token"}
    assert captured["headers"] == {
        runner.ADMIN_SECRET_HEADER: test_secret,
    }
    prepared = runner.requests.Request(
        "GET",
        captured["url"],
        params=captured["params"],
        headers=captured["headers"],
    ).prepare()
    assert prepared.url is not None
    assert "admin_secret" not in prepared.url.lower()
    assert test_secret not in prepared.url


def test_required_final_history_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args):
        raise ConnectionError("judge unavailable")

    monkeypatch.setattr(runner, "_fetch_full_judge_history", fail)

    with pytest.raises(
        runner.JudgeDrainError,
        match="authoritative final Judge history",
    ):
        runner._fetch_required_final_judge_history(
            "http://judge",
            "unit-test-session-token",
        )


def _submission_archive(name: str, content: bytes) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        info = tarfile.TarInfo(name)
        info.size = len(content)
        bundle.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def _dot_prefixed_submission_archive() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        root = tarfile.TarInfo("./")
        root.type = tarfile.DIRTYPE
        bundle.addfile(root)
        nested = tarfile.TarInfo("./proofs/")
        nested.type = tarfile.DIRTYPE
        bundle.addfile(nested)
        payload = b"Lemma solved : True. Proof. exact I. Qed.\n"
        proof = tarfile.TarInfo("./proofs/task1.v")
        proof.size = len(payload)
        bundle.addfile(proof, io.BytesIO(payload))
    return buffer.getvalue()


def test_live_dot_submit_archive_is_canonicalized_before_judge_upload() -> None:
    normalized = runner._normalize_live_submission_archive(
        _dot_prefixed_submission_archive()
    )

    with tarfile.open(fileobj=io.BytesIO(normalized), mode="r:gz") as bundle:
        assert bundle.getnames() == ["proofs", "proofs/task1.v"]
        assert bundle.extractfile("proofs/task1.v").read().startswith(b"Lemma solved")

    # Exercise the same strict trust-boundary validator used by the Judge.
    from sforge.harness.run_evaluation import _validate_submission_archive

    assert _validate_submission_archive(normalized) == ["proofs/task1.v"]


def test_running_container_extraction_normalizes_dot_submit_path() -> None:
    inner = _dot_prefixed_submission_archive()
    transport = io.BytesIO()
    with tarfile.open(fileobj=transport, mode="w") as bundle:
        info = tarfile.TarInfo("final.tar.gz")
        info.size = len(inner)
        bundle.addfile(info, io.BytesIO(inner))

    class Backend:
        def exec_run(self, handle, command):
            return SimpleNamespace(exit_code=0, output="")

        def copy_from_container(self, handle, path):
            return transport.getvalue()

    class Agent:
        def validate_submission_archive(self, archive: bytes) -> None:
            from sforge.harness.run_evaluation import _validate_submission_archive

            assert _validate_submission_archive(archive) == ["proofs/task1.v"]

    archive = runner._extract_archive_from_container(
        Backend(),
        object(),
        SimpleNamespace(cwd="/workspace", submit_paths=["."], submit_exclude=[]),
        Agent(),
    )
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
        assert bundle.getnames() == ["proofs", "proofs/task1.v"]


@pytest.mark.parametrize(
    "name",
    ["./proofs/../escape.v", "./proofs/./task1.v", "/absolute/task1.v"],
)
def test_live_archive_normalization_rejects_unsafe_noncanonical_names(
    name: str,
) -> None:
    with pytest.raises(ValueError, match="absolute|canonical"):
        runner._normalize_live_submission_archive(
            _submission_archive(name, b"unsafe")
        )


def test_codex_archive_guard_rejects_renamed_credential_material() -> None:
    agent = object.__new__(CodexAgent)
    agent._submission_forbidden_blobs = (
        b'{"tokens":{"access_token":"secret-token-value"}}',
        b"secret-token-value",
    )

    with pytest.raises(RuntimeError, match="credential material"):
        agent.validate_submission_archive(
            _submission_archive(
                "kernels/innocent_name.cu",
                b"// secret-token-value",
            )
        )

    agent.validate_submission_archive(
        _submission_archive(
            "kernels/gemm_v001.cu",
            b"// ordinary CUDA source",
        )
    )


def test_codex_runs_in_a_dedicated_session() -> None:
    command = runner._agent_exec_command(
        SimpleNamespace(name="codex"),
        "codex exec resume --last",
    )
    assert command == [
        "setsid",
        "--wait",
        "/bin/bash",
        "-c",
        (
            'set -e; printf "%s\\n" "$$" '
            '>"$SFORGE_EXEC_PROCESS_GROUP_FILE"; '
            'exec /bin/bash -c "$1"'
        ),
        "sforge-codex-session",
        "codex exec resume --last",
    ]
    assert runner._agent_exec_command(
        SimpleNamespace(name="claude-code"), "claude"
    ) == ["/bin/bash", "-c", "claude"]


def test_codex_auth_file_uses_restricted_chatgpt_relay_without_web_search(
    tmp_path: Path,
) -> None:
    relay = "http://host.docker.internal:19090/"
    config = SForgeConfig(
        codex_auth_json_path=tmp_path / "auth.json",
        agent_api_base_url=relay,
        codex_reasoning_effort="max",
    )
    agent = CodexAgent(config)
    env = {"OPENAI_BASE_URL": relay}

    agent.augment_env(env, "example-model-mini")
    command = agent.format_run_cmd(
        "/tmp/prompt.md",
        model="example-model-mini",
        internet=False,
    )

    assert "OPENAI_BASE_URL" not in env
    assert 'web_search="disabled"' in command
    assert 'chatgpt_base_url="http://host.docker.internal:19090"' in command
    assert "example-model-mini" in command
    assert "model_reasoning_effort" in command


def test_codex_web_search_is_explicitly_live_when_internet_is_allowed(
    tmp_path: Path,
) -> None:
    # Every task allows internet (docs/INTERNET_AND_WEB_SEARCH.md); the flag is
    # explicit so a Work image's own codex-config.toml cannot disable search.
    config = SForgeConfig(codex_auth_json_path=tmp_path / "auth.json")
    command = CodexAgent(config).format_run_cmd("/tmp/prompt.md", internet=True)

    assert 'web_search="live"' in command
    assert 'web_search="disabled"' not in command


def test_codex_runs_one_agent_on_first_run_and_resume(tmp_path: Path) -> None:
    # Same as PyroMind: no sub-agent or code-mode tools for any model.
    config = SForgeConfig(codex_auth_json_path=tmp_path / "auth.json")
    agent = CodexAgent(config)
    for resume in (False, True):
        command = agent.format_run_cmd("/tmp/prompt.md", internet=True, resume=resume)
        for feature in ("multi_agent", "code_mode", "code_mode_only"):
            assert f"--disable {feature} " in command


def test_codex_uses_direct_tool_catalog_for_the_run_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Same catalog as PyroMind; installed with the model the run actually uses,
    # and referenced by the command only once it is in the Work container.
    import sforge.harness.agent.codex as codex_module

    seen: dict = {}
    monkeypatch.setattr(
        codex_module, "load_direct_tool_catalog",
        lambda runtime, model: seen.setdefault("model", model) and "{}",
    )
    monkeypatch.setattr(
        codex_module, "copy_container_file",
        lambda backend, handle, local, remote, **kw: seen.setdefault("path", str(remote)),
    )
    agent = CodexAgent(SForgeConfig(agent_runtime_dir=tmp_path))
    assert "model_catalog_json" not in agent.format_run_cmd("/tmp/p.md")
    agent.install_model_catalog(None, None, tmp_path, logging.getLogger("t"), None)
    assert "model" not in seen
    agent.install_model_catalog(None, None, tmp_path, logging.getLogger("t"), "example-model")
    assert seen == {"model": "example-model", "path": "/home/agent/.codex/sforge-model-catalog.json"}
    for resume in (False, True):
        command = agent.format_run_cmd("/tmp/p.md", resume=resume)
        assert 'model_catalog_json="/home/agent/.codex/sforge-model-catalog.json"' in command


def test_codex_can_use_a_host_allowlisted_connect_proxy(
    tmp_path: Path,
) -> None:
    restricted_proxy = "http://host.docker.internal:3128"
    config = SForgeConfig(
        codex_auth_json_path=tmp_path / "auth.json",
        agent_api_base_url=restricted_proxy,
        agent_extra_env={
            "SFORGE_CODEX_CHATGPT_BASE_URL":
                "https://chatgpt.com/backend-api/codex",
        },
    )
    command = CodexAgent(config).format_run_cmd(
        "/tmp/prompt.md", internet=False,
    )

    assert f"HTTPS_PROXY={restricted_proxy}" in command
    assert "env -u ALL_PROXY -u all_proxy" in command
    assert 'chatgpt_base_url="https://chatgpt.com/backend-api/codex"' in command
    assert 'web_search="disabled"' in command


def test_two_hour_schedule_excludes_timeout_boundary() -> None:
    assert runner._scheduled_auto_eval_offsets(7200, 1800) == (
        1800,
        3600,
        5400,
    )
    assert runner._scheduled_auto_eval_offsets(1800, 1800) == ()


def test_resume_gaps_do_not_extend_absolute_deadline() -> None:
    deadline = 7200.0
    assert runner._remaining_run_timeout_seconds(
        deadline, now_monotonic=0.0
    ) == 7200
    # A 1800s segment plus a 300s resume/setup gap leaves 5100s, not 5400s.
    assert runner._remaining_run_timeout_seconds(
        deadline, now_monotonic=2100.0
    ) == 5100
    assert runner._remaining_run_timeout_seconds(
        deadline, now_monotonic=7200.0
    ) == 0


def test_resume_safety_cap_cannot_preempt_fixed_run_deadline() -> None:
    cap = runner._max_resume_attempts(
        run_timeout=7200,
        min_runtime_for_resume=1,
    )

    assert cap == 7201
    assert cap * 1 > 7200


def test_auto_eval_loop_submits_only_preboundary_ticks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clock = SimpleNamespace(value=0.0)

    class AdvancingEvent:
        def __init__(self) -> None:
            self.waits: list[float] = []
            self._set = False

        def wait(self, seconds: float) -> bool:
            self.waits.append(seconds)
            clock.value += seconds
            return self._set

        def is_set(self) -> bool:
            return self._set

    class Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {
                "submission_id": "submission",
                "round_id": "auto",
            }

    submitted: list[dict] = []
    event = AdvancingEvent()
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock.value)
    monkeypatch.setattr(
        runner,
        "_extract_archive_from_container",
        lambda *args: b"archive",
    )
    monkeypatch.setattr(
        runner.requests,
        "post",
        lambda *args, **kwargs: submitted.append(kwargs) or Response(),
    )

    runner._auto_eval_loop(
        object(),
        object(),
        object(),
        object(),
        "http://judge",
        "token",
        1800,
        0.0,
        7200,
        event,
        _Logger(),
        tmp_path,
    )

    assert event.waits == [1800.0, 1800.0, 1800.0]
    assert len(submitted) == 3
    for request in submitted:
        assert request["data"] == b"archive"
        assert request["headers"] == {
            "Content-Type": "application/gzip",
            "X-SForge-Token": "token",
            "X-SForge-Submission-Kind": "auto",
            runner.ADMIN_SECRET_HEADER: runner.ADMIN_SECRET,
        }
        assert "files" not in request


def test_auto_eval_skips_a_tick_delayed_to_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clock = SimpleNamespace(value=0.0)

    class DelayedEvent:
        def wait(self, seconds: float) -> bool:
            clock.value += seconds + 7200
            return False

        def is_set(self) -> bool:
            return False

    submitted: list[object] = []
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock.value)
    monkeypatch.setattr(
        runner.requests,
        "post",
        lambda *args, **kwargs: submitted.append(kwargs),
    )

    runner._auto_eval_loop(
        object(),
        object(),
        object(),
        object(),
        "http://judge",
        "token",
        1800,
        0.0,
        7200,
        DelayedEvent(),
        _Logger(),
        tmp_path,
    )

    assert submitted == []


def test_safe_auto_eval_feedback_exposes_only_allowlisted_source_locations() -> None:
    report = {
        "valid": True,
        "score": 999.0,
        "metrics": {
            "hidden_seed": "secret",
            "debug_output_warnings": [
                {
                    "category": "debug_output",
                    "path": "kernels/gemm_v007_probe.cu",
                    "line": 42,
                    "message": "private wording",
                },
                {
                    "category": "debug_output",
                    "path": "../hidden/seed.txt",
                    "line": 1,
                },
                {
                    "category": "other",
                    "path": "kernels/gemm_v008_probe.cu",
                    "line": 2,
                },
            ],
        },
    }

    assert runner._safe_debug_output_warnings(report) == [
        {
            "category": "debug_output",
            "path": "kernels/gemm_v007_probe.cu",
            "line": 42,
            "message": "Non-fatal debug output; hidden evaluation continued.",
        }
    ]


def test_safe_auto_eval_feedback_copy_omits_hidden_result_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {
                "status": "completed",
                "report": {
                    "valid": True,
                    "score": 999.0,
                    "metrics": {
                        "hidden_seed": "secret",
                        "debug_output_warnings": [
                            {
                                "category": "debug_output",
                                "path": "kernels/gemm_v009_probe.cu",
                                "line": 7,
                            }
                        ],
                    },
                },
            }

    copied: dict = {}

    class Backend:
        def copy_to_container(
            self,
            handle: object,
            local_path: Path,
            remote_path: object,
        ) -> None:
            copied["handle"] = handle
            copied["remote_path"] = remote_path
            copied["payload"] = json.loads(local_path.read_text())

    monkeypatch.setattr(runner.requests, "get", lambda *args, **kwargs: Response())
    handle = object()
    runner._forward_safe_auto_eval_feedback(
        Backend(),
        handle,
        "http://judge",
        "submission-1",
        "auto-1",
        threading.Event(),
        _Logger(),
        tmp_path,
        timeout=1.0,
    )

    assert copied["handle"] is handle
    assert copied["remote_path"] == runner.AUTO_EVAL_ADMISSION_FEEDBACK_PATH
    assert copied["payload"] == {
        "schema_version": 1,
        "round_id": "auto-1",
        "debug_output_warnings": [
            {
                "category": "debug_output",
                "path": "kernels/gemm_v009_probe.cu",
                "line": 7,
                "message": (
                    "Non-fatal debug output; hidden evaluation continued."
                ),
            }
        ],
    }
    serialized = json.dumps(copied["payload"])
    assert "999" not in serialized
    assert "hidden_seed" not in serialized
    assert "secret" not in serialized


def test_auto_primary_score_feedback_requires_explicit_safe_opt_in() -> None:
    base_report = {
        "valid": True,
        "score": 73.25,
        "summary": " PRIMARY SCORE passed\nwith safe text ",
        "metrics": {
            "primary_score_name": "valid_payment_rate_per_sec",
            "primary_score_unit": "valid payment workflows/s",
            "primary_score_formula": "score = RPS after every gate passes",
        },
    }

    assert runner._safe_auto_score_feedback(base_report) is None

    opted_in = json.loads(json.dumps(base_report))
    opted_in["metrics"]["auto_feedback_safe"] = True
    assert runner._safe_auto_score_feedback(opted_in) == {
        "valid": True,
        "score": 73.25,
        "primary_score_name": "valid_payment_rate_per_sec",
        "primary_score_unit": "valid payment workflows/s",
        "primary_score_formula": "score = RPS after every gate passes",
        "summary": (
            "CORRECTNESS BINARY GATE PASSED; the reported primary score is "
            "eligible for best-round comparison."
        ),
        "secondary_metrics_omitted": True,
    }


def test_invalid_auto_primary_feedback_does_not_forward_hidden_phase_summary() -> None:
    report = {
        "valid": False,
        "score": 0.0,
        "summary": "hidden phase idempotency failed with private payload",
        "details": [{"message": "private payload"}],
        "metrics": {
            "auto_feedback_safe": True,
            "primary_score_name": "valid_payment_rate_per_sec",
            "primary_score_unit": "valid payment workflows/s",
            "primary_score_formula": "score = RPS after every gate passes",
        },
    }

    feedback = runner._safe_auto_score_feedback(report)

    assert feedback is not None
    assert feedback["valid"] is False
    assert feedback["score"] == 0.0
    assert feedback["summary"] == (
        "CORRECTNESS BINARY GATE FAILED; the primary score is zero. "
        "Use a manual Judge submission for bounded phase-level diagnostics."
    )
    serialized = json.dumps(feedback)
    assert "idempotency" not in serialized
    assert "private payload" not in serialized


@pytest.mark.parametrize("score", [float("nan"), float("inf"), True, "73.25"])
def test_auto_primary_score_feedback_rejects_unsafe_score(score: object) -> None:
    report = {
        "valid": True,
        "score": score,
        "summary": "safe",
        "metrics": {
            "auto_feedback_safe": True,
            "primary_score_name": "valid_payment_rate_per_sec",
            "primary_score_unit": "valid payment workflows/s",
            "primary_score_formula": "score = RPS",
        },
    }

    assert runner._safe_auto_score_feedback(report) is None


def test_auto_primary_score_feedback_copy_omits_submission_identifier_and_details(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {
                "status": "completed",
                "report": {
                    "valid": True,
                    "score": 88.5,
                    "summary": (
                        "PRIMARY SCORE (MAXIMIZE): "
                        "valid_payment_rate_per_sec=88.5"
                    ),
                    "details": [{"name": "hidden phase", "message": "secret"}],
                    "metrics": {
                        "auto_feedback_safe": True,
                        "primary_score_name": "valid_payment_rate_per_sec",
                        "primary_score_unit": "valid payment workflows/s",
                        "primary_score_formula": (
                            "score = valid payment RPS after every gate passes"
                        ),
                        "hidden_seed": "secret-seed",
                    },
                },
            }

    copied: dict = {}

    class Backend:
        def copy_to_container(
            self,
            handle: object,
            local_path: Path,
            remote_path: object,
        ) -> None:
            copied["remote_path"] = remote_path
            copied["payload"] = json.loads(local_path.read_text())

    monkeypatch.setattr(runner.requests, "get", lambda *args, **kwargs: Response())
    runner._forward_safe_auto_eval_feedback(
        Backend(),
        object(),
        "http://judge",
        "unguessable-submission-id",
        "auto-2",
        threading.Event(),
        _Logger(),
        tmp_path,
        timeout=1.0,
    )

    assert copied["remote_path"] == runner.AUTO_EVAL_SCORE_FEEDBACK_PATH
    assert copied["payload"]["round"] == "auto-2"
    assert "round_id" not in copied["payload"]
    assert copied["payload"]["valid"] is True
    assert copied["payload"]["score"] == 88.5
    serialized = json.dumps(copied["payload"])
    assert "submission" not in serialized
    assert "hidden phase" not in serialized
    assert "valid_payment_rate_per_sec=88.5" not in serialized
    assert "hidden_seed" not in serialized
    assert "secret" not in serialized


def test_stopping_auto_eval_joins_the_thread() -> None:
    stop = threading.Event()
    exited = threading.Event()

    def worker() -> None:
        stop.wait()
        exited.set()

    thread = threading.Thread(target=worker)
    thread.start()
    runner._stop_auto_eval_thread(stop, thread, _Logger())

    assert exited.is_set()
    assert not thread.is_alive()


def test_final_snapshot_metadata_records_epoch_and_elapsed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(runner.time, "monotonic", lambda: 7300.0)
    monkeypatch.setattr(runner.time, "time", lambda: 1_800_000_000.0)

    metadata = runner._write_final_snapshot_metadata(tmp_path, 100.0)
    persisted = json.loads((tmp_path / "final_snapshot.json").read_text())

    assert metadata == persisted
    assert persisted == {
        "captured_at": 1_800_000_000.0,
        "snapshot_elapsed_seconds": 7200.0,
    }


@pytest.mark.parametrize(
    ("pending_count", "expected"),
    [(0, 3660), (1, 3660), (2, 7260), (3, 10860)],
)
def test_serialized_judge_drain_budget(
    pending_count: int,
    expected: int,
) -> None:
    assert runner._judge_drain_timeout_seconds(
        3600, pending_count
    ) == expected


def test_judge_drain_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = SimpleNamespace(value=0.0)
    pending = [{"status": "running"}]

    monkeypatch.setattr(
        runner,
        "_fetch_pending_judge_evaluations",
        lambda *args: pending,
    )
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock.value)
    monkeypatch.setattr(
        runner.time,
        "sleep",
        lambda seconds: setattr(clock, "value", clock.value + seconds),
    )

    with pytest.raises(runner.JudgeDrainError, match="timed out"):
        runner._drain_pending_judge_evaluations(
            "http://judge",
            "token",
            1,
            _Logger(),
            poll_interval=1000,
        )


def test_judge_drain_rejects_unknown_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args):
        raise ConnectionError("judge unavailable")

    monkeypatch.setattr(
        runner, "_fetch_pending_judge_evaluations", fail
    )
    with pytest.raises(runner.JudgeDrainError, match="Unable to inspect"):
        runner._drain_pending_judge_evaluations(
            "http://judge",
            "token",
            3600,
            _Logger(),
        )
