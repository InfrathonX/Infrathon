from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from sforge import cli as sforge_cli
from sforge.harness import agent as agent_module
from sforge.harness import network_isolation
from sforge.harness import run_agent as runner
from sforge.harness.config import SForgeConfig
from sforge.harness.run_agent import (
    FinalSnapshotError,
    FinalVersionStoreError,
    _ensure_version_store_backend_supported,
    _persist_final_version,
)
from sforge.harness.version_store import VersionStore


def _archive() -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as bundle:
        payload = b"#!/bin/sh\necho final\n"
        member = tarfile.TarInfo("submission/run.sh")
        member.mode = 0o755
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))
    return output.getvalue()


def _store_root(root: Path) -> Path:
    root.mkdir()
    (root / ".sforge-version-store.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "sforge-version-store",
                "store_id": "final-test-store",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    task_root = root / "tasks" / "task-alpha"
    (task_root / "runs").mkdir(parents=True)
    (task_root / "task.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_directory": "task-alpha",
                "task_id": "task_alpha",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def _task_spec() -> SimpleNamespace:
    return SimpleNamespace(
        task_id="task_alpha",
        submit_paths=["submission"],
        submit_exclude=["*.tmp"],
        candidate_identity_exclude=[],
    )


def test_persist_final_version_is_a_noop_when_store_is_disabled() -> None:
    assert (
        _persist_final_version(
            None,
            task_spec=_task_spec(),
            run_id="run-001",
            archive=_archive(),
            agent_name="codex",
            model="example-model-max",
            timed_out=False,
            runtime_seconds=12.5,
            interrupted=False,
        )
        is None
    )


def test_version_store_requires_a_backend_with_frozen_final_snapshots() -> None:
    _ensure_version_store_backend_supported(
        SimpleNamespace(backend_name="docker"),
        version_saving_enabled=True,
    )
    _ensure_version_store_backend_supported(
        SimpleNamespace(backend_name="k8s"),
        version_saving_enabled=False,
    )
    with pytest.raises(FinalSnapshotError, match="Docker"):
        _ensure_version_store_backend_supported(
            SimpleNamespace(backend_name="k8s"),
            version_saving_enabled=True,
        )


def test_persist_final_version_records_exact_bytes_and_run_metadata(
    tmp_path: Path,
) -> None:
    archive = _archive()
    store = VersionStore.open(_store_root(tmp_path / "store"))

    first = _persist_final_version(
        store,
        task_spec=_task_spec(),
        run_id="run-001",
        archive=archive,
        agent_name="codex",
        model="example-model-max",
        timed_out=True,
        runtime_seconds=7200.25,
        interrupted=False,
    )
    repeated = _persist_final_version(
        store,
        task_spec=_task_spec(),
        run_id="run-001",
        archive=archive,
        agent_name="codex",
        model="example-model-max",
        timed_out=True,
        runtime_seconds=7200.25,
        interrupted=False,
    )

    assert first is not None
    assert repeated == first
    assert first.source == "final"
    assert first.round_id == "final"
    assert first.label is None
    assert first.archive_path.read_bytes() == archive
    assert first.archive_sha256 == hashlib.sha256(archive).hexdigest()
    assert first.metadata == {
        "agent": "codex",
        "interrupted": False,
        "model": "example-model-max",
        "runtime_seconds": 7200.25,
        "timed_out": True,
    }


def test_persist_final_version_failure_is_fatal_after_local_archive_exists(
    tmp_path: Path,
) -> None:
    archive = _archive()
    final_archive = tmp_path / "final_archive.tar.gz"
    final_archive.write_bytes(archive)

    class _FailingStore:
        def record_archive(self, **kwargs):
            assert final_archive.read_bytes() == kwargs["archive"]
            raise OSError("disk full")

    with pytest.raises(FinalVersionStoreError, match="final version"):
        _persist_final_version(
            _FailingStore(),
            task_spec=_task_spec(),
            run_id="run-001",
            archive=archive,
            agent_name="codex",
            model=None,
            timed_out=False,
            runtime_seconds=1.0,
            interrupted=True,
        )


@pytest.mark.parametrize(
    ("failure", "expected_message"),
    [
        ("store-mismatch", "version store mismatch"),
        ("http-unavailable", "session registration failed"),
        ("invalid-url", "session registration failed"),
    ],
)
def test_cli_run_propagates_required_store_registration_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    failure: str,
    expected_message: str,
) -> None:
    store_root = _store_root(tmp_path / "store")
    config = SForgeConfig(
        log_dir=tmp_path / "logs",
        version_store_root=store_root,
    )
    task_spec = SimpleNamespace(
        task_id="task_alpha",
        work_image_key="work-image",
        judge_image_key="judge-image",
        game_mode=False,
        internet=True,
        submit_paths=["submission"],
        submit_exclude=[],
        candidate_identity_exclude=[],
    )
    fake_agent = SimpleNamespace(
        name="codex",
        timeout=60,
        default_model="example-model-max",
        stop_hook=None,
        resume_cmd=None,
    )

    class Backend:
        backend_name = "docker"

        def image_exists(self, image_key: str) -> bool:
            return True

        def cleanup_container(self, handle, logger) -> None:
            pass

    class RegistrationResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {
                "token": "evaluation-token",
                "checkpoint_token": None,
                "version_saving_enabled": True,
                "version_store_id": "wrong-store",
            }

    monkeypatch.setattr(agent_module, "create_agent", lambda *args: fake_agent)
    monkeypatch.setattr(network_isolation, "cleanup_stale_chains", lambda logger: None)
    registration_attempts: list[object] = []
    if failure == "store-mismatch":
        monkeypatch.setattr(
            runner.requests,
            "post",
            lambda *args, **kwargs: RegistrationResponse(),
        )
    else:
        request_error = (
            runner.requests.ConnectionError("judge unavailable")
            if failure == "http-unavailable"
            else runner.requests.exceptions.MissingSchema("invalid judge URL")
        )

        def unavailable(*args, **kwargs):
            registration_attempts.append(object())
            raise request_error

        monkeypatch.setattr(runner.requests, "post", unavailable)
        monkeypatch.setattr(runner.time, "sleep", lambda seconds: None)
    args = SimpleNamespace(
        agent="codex",
        timeout=60,
        model=None,
        disable_stop_hook=False,
        disable_auto_eval=True,
        disable_auto_resume=True,
        eval_interval=300,
        disable_internet=False,
        enable_internet=False,
        max_submissions=1,
        submission_cooldown=None,
        judge_url="http://127.0.0.1:8080",
    )

    with pytest.raises(RuntimeError, match=expected_message):
        sforge_cli._run_single_task(
            task_spec,
            args,
            config,
            Backend(),
            "run-mismatch",
        )

    if failure != "store-mismatch":
        assert len(registration_attempts) == 5
    output = capsys.readouterr()
    assert "Agent completed" not in output.out
    assert not (
        tmp_path
        / "logs/runs/run-mismatch/task_alpha/final_result.json"
    ).exists()
