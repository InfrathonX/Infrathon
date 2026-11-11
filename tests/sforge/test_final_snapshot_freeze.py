from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
import stat
import tarfile
import threading
import time
import uuid
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import docker.errors
import pytest
from requests import Response

from sforge.harness.backend import docker_backend
from sforge.harness.backend.docker_backend import (
    DockerBackend,
    DockerContainerHandle,
)


runner = importlib.import_module("sforge.harness.run_agent")


def test_final_artifact_write_fsyncs_file_replace_and_directory_in_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    original_fsync = os.fsync
    original_replace = os.replace

    def recording_fsync(descriptor: int) -> None:
        kind = (
            "directory"
            if stat.S_ISDIR(os.fstat(descriptor).st_mode)
            else "file"
        )
        events.append(f"fsync:{kind}")
        original_fsync(descriptor)

    def recording_replace(source, destination) -> None:
        events.append("replace")
        original_replace(source, destination)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    monkeypatch.setattr(os, "replace", recording_replace)
    destination = tmp_path / "final_archive.tar.gz"

    runner._atomic_write_bytes(destination, b"durable-archive")

    assert destination.read_bytes() == b"durable-archive"
    assert events == ["fsync:file", "replace", "fsync:directory"]


def test_final_artifact_write_cleans_temp_on_replace_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "final_archive.tar.gz"
    destination.write_bytes(b"previous")

    def fail_replace(source, target) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        runner._atomic_write_bytes(destination, b"new")

    assert destination.read_bytes() == b"previous"
    assert list(tmp_path.glob(".final_archive.tar.gz.*")) == []


class _Logger:
    def info(self, *args, **kwargs) -> None:
        pass

    def error(self, *args, **kwargs) -> None:
        pass


class _RecordingLogger:
    def __init__(self) -> None:
        self.infos: list[str] = []
        self.errors: list[str] = []

    def info(self, message: str) -> None:
        self.infos.append(message)

    def error(self, message: str) -> None:
        self.errors.append(message)


class _Agent:
    def validate_submission_archive(self, archive: bytes) -> None:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
            names = bundle.getnames()
        assert all(".git" not in PurePosixPath(name).parts for name in names)


def _transport_archive(
    root: str,
    files: dict[str, tuple[bytes, int]],
) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as bundle:
        directory = tarfile.TarInfo(root)
        directory.type = tarfile.DIRTYPE
        directory.mode = 0o755
        bundle.addfile(directory)
        for relative_name, (payload, mode) in files.items():
            info = tarfile.TarInfo(f"{root}/{relative_name}")
            info.size = len(payload)
            info.mode = mode
            bundle.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


class _FrozenBackend:
    backend_name = "docker"

    def __init__(
        self,
        *,
        freeze_fails: bool = False,
        restart_after_copy: bool = False,
    ) -> None:
        self.frozen = False
        self.freeze_fails = freeze_fails
        self.restart_after_copy = restart_after_copy
        self.events: list[str] = []
        self.copies = 0

    def freeze_container_for_snapshot(
        self,
        handle,
        *,
        stop_without_inspect: bool = False,
    ) -> dict[str, object]:
        self.events.append("freeze")
        assert stop_without_inspect is True
        if self.freeze_fails:
            raise RuntimeError("stop failed")
        self.frozen = True
        return {
            "mechanism": "docker_stop_timeout_0",
            "all_work_processes_stopped": True,
            "was_running": True,
            "freeze_requested_at": 1_800_007_200.0,
            "freeze_requested_monotonic": 7300.0,
            "frozen_at": 1_800_007_200.01,
            "frozen_at_monotonic": 7300.01,
            "container_status": "exited",
            "container_pid": 0,
        }

    def verify_container_frozen_for_snapshot(self, handle) -> dict[str, object]:
        self.events.append("verify")
        if not self.frozen:
            raise RuntimeError("writer became runnable")
        return {
            "all_work_processes_stopped": True,
            "container_status": "exited",
            "container_pid": 0,
        }

    def copy_from_container(self, handle, source: PurePosixPath) -> bytes:
        self.events.append("copy")
        assert self.frozen, "copy must never race a runnable Work container"
        assert source == PurePosixPath("/workspace/submission")
        self.copies += 1
        archive = _transport_archive(
            "submission",
            {
                "run.sh": (b"#!/bin/sh\necho boundary\n", 0o755),
                "answer.txt": (b"at-deadline", 0o644),
                ".git/config": (b"private", 0o600),
                "discard.tmp": (b"excluded", 0o644),
            },
        )
        if self.restart_after_copy:
            self.frozen = False
        return archive


class _PausedFinalBackend(_FrozenBackend):
    def freeze_container_for_snapshot(self, handle, **kwargs):
        raise AssertionError("paused final capture must not use ordinary stop")

    def freeze_container_for_final_handoff_snapshot(
        self,
        handle,
        *,
        expected_annotations,
        stop_without_inspect=False,
    ) -> dict[str, object]:
        assert expected_annotations == {"binding": "exact"}
        assert stop_without_inspect is True
        self.events.append("adopt-paused-final")
        self.frozen = True
        return {
            "mechanism": "docker_pause_never_resume_final",
            "all_work_processes_frozen": True,
            "all_work_processes_stopped": False,
            "was_running": True,
            "freeze_requested_at": 1_800_007_200.0,
            "freeze_requested_monotonic": 7300.0,
            "frozen_at": 1_800_007_200.0,
            "frozen_at_monotonic": 7300.0,
            "container_status": "paused",
            "container_pid": 8123,
            "container_paused": True,
            "container_id": "a" * 64,
            "archive_source": "paused_work_container_filesystem",
            "archive_transport": (
                "docker_get_archive_transport_tar_while_paused"
            ),
        }


def _task_spec() -> SimpleNamespace:
    return SimpleNamespace(
        cwd="/workspace",
        submit_paths=["submission"],
        submit_exclude=["*.tmp"],
    )


def test_capture_freezes_before_copy_and_records_auditable_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = _FrozenBackend()
    stop = threading.Event()
    time_values = iter([1_800_007_200.02, 1_800_007_200.03])
    monotonic_values = iter([7300.02, 7300.03])
    monkeypatch.setattr(runner.time, "time", lambda: next(time_values))
    monkeypatch.setattr(
        runner.time, "monotonic", lambda: next(monotonic_values)
    )

    archive, metadata = runner._capture_frozen_final_snapshot(
        backend,
        object(),
        _task_spec(),
        _Agent(),
        tmp_path,
        100.0,
        _Logger(),
        run_deadline_monotonic=7300.0,
        auto_eval_stop=stop,
        auto_eval_thread=None,
    )

    assert stop.is_set()
    assert backend.events[0] == "freeze"
    assert backend.events.index("copy") > backend.events.index("verify")
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
        members = {member.name: member for member in bundle.getmembers()}
        assert bundle.extractfile("submission/answer.txt").read() == b"at-deadline"
    assert set(members) == {
        "submission",
        "submission/run.sh",
        "submission/answer.txt",
    }
    assert members["submission/run.sh"].mode & 0o111

    persisted = json.loads((tmp_path / "final_snapshot.json").read_text())
    assert persisted == metadata
    assert metadata["deadline"]["elapsed_seconds"] == 7200.0
    assert metadata["freeze"]["freeze_requested_elapsed_seconds"] == 7200.0
    assert metadata["freeze"]["frozen_elapsed_seconds"] == pytest.approx(7200.01)
    assert metadata["snapshot_elapsed_seconds"] == pytest.approx(7200.01)
    assert metadata["snapshot_boundary"] == "whole_container_frozen"
    assert metadata["capture"]["archive_completed_elapsed_seconds"] == pytest.approx(
        7200.03
    )
    assert metadata["capture"]["work_container_running_during_capture"] is False
    assert metadata["archive"]["sha256"] == hashlib.sha256(archive).hexdigest()
    assert metadata["archive_sha256"] == metadata["archive"]["sha256"]
    assert metadata["archive"]["size_bytes"] == len(archive)
    assert (tmp_path / "final_archive.tar.gz").read_bytes() == archive


def test_capture_can_use_durably_fenced_paused_handoff_without_unpause(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = _PausedFinalBackend()
    time_values = iter(
        [
            1_800_007_200.02,
            1_800_007_200.03,
            1_800_007_200.04,
            1_800_007_200.05,
        ]
    )
    monotonic_values = iter([7300.02, 7300.03, 7300.04, 7300.05])
    monkeypatch.setattr(runner.time, "time", lambda: next(time_values))
    monkeypatch.setattr(
        runner.time,
        "monotonic",
        lambda: next(monotonic_values),
    )

    archive, metadata = runner._capture_frozen_final_snapshot(
        backend,
        object(),
        _task_spec(),
        _Agent(),
        tmp_path,
        100.0,
        _Logger(),
        run_deadline_monotonic=7300.0,
        auto_eval_stop=threading.Event(),
        auto_eval_thread=None,
        final_handoff_annotations={"binding": "exact"},
    )

    assert archive
    assert backend.events[0] == "adopt-paused-final"
    assert "freeze" not in backend.events
    assert metadata["freeze"]["all_work_processes_frozen"] is True
    assert metadata["freeze"]["all_work_processes_stopped"] is False
    assert metadata["capture"]["work_container_paused_during_capture"] is True
    assert metadata["capture"]["source"] == "paused_work_container_filesystem"


def test_freeze_failure_is_fail_closed_before_any_copy(tmp_path: Path) -> None:
    backend = _FrozenBackend(freeze_fails=True)

    with pytest.raises(
        runner.FinalSnapshotError,
        match="whole-container freeze",
    ):
        runner._capture_frozen_final_snapshot(
            backend,
            object(),
            _task_spec(),
            _Agent(),
            tmp_path,
            100.0,
            _Logger(),
            run_deadline_monotonic=7300.0,
            auto_eval_stop=threading.Event(),
            auto_eval_thread=None,
        )

    assert backend.copies == 0
    assert not (tmp_path / "final_archive.tar.gz").exists()
    assert not (tmp_path / "final_snapshot.json").exists()


def test_restart_during_copy_is_fail_closed_without_publishing(
    tmp_path: Path,
) -> None:
    backend = _FrozenBackend(restart_after_copy=True)

    with pytest.raises(
        runner.FinalSnapshotError,
        match="capture failed",
    ):
        runner._capture_frozen_final_snapshot(
            backend,
            object(),
            _task_spec(),
            _Agent(),
            tmp_path,
            100.0,
            _Logger(),
            run_deadline_monotonic=7300.0,
            auto_eval_stop=threading.Event(),
            auto_eval_thread=None,
        )

    assert backend.copies == 1
    assert not (tmp_path / "final_archive.tar.gz").exists()
    assert not (tmp_path / "final_snapshot.json").exists()


class _ExecAPI:
    def __init__(self, container: "_TimeoutContainer") -> None:
        self.container = container

    def exec_create(self, container_id, command, **kwargs):
        return {"Id": "exec-id"}

    def exec_start(self, exec_id, stream=True):
        self.container.stream_done.wait(timeout=2)
        if False:
            yield b""

    def exec_inspect(self, exec_id):
        return {"Pid": 4242, "ExitCode": 0}


class _TimeoutContainer:
    def __init__(self) -> None:
        self.id = "container-id"
        self.name = "container"
        self.status = "running"
        self.attrs = {
            "State": {
                "Running": True,
                "Restarting": False,
                "Paused": False,
                "Pid": 99,
                "Status": "running",
            }
        }
        self.stream_done = threading.Event()
        self.stop_timeouts: list[int] = []
        self.remove_calls: list[bool] = []
        self.exec_commands: list[str] = []
        self.state_events: list[str] = []
        self.client = SimpleNamespace(api=_ExecAPI(self))

    def reload(self) -> None:
        self.state_events.append("reload")

    def stop(self, timeout: int) -> None:
        self.state_events.append("stop")
        self.stop_timeouts.append(timeout)
        self.status = "exited"
        self.attrs["State"].update(
            {"Running": False, "Pid": 0, "Status": "exited"}
        )
        self.stream_done.set()

    def exec_run(self, command, **kwargs):
        shell = command[-1] if isinstance(command, list) else str(command)
        self.exec_commands.append(shell)
        return SimpleNamespace(exit_code=0, output=b"")

    def remove(self, force: bool) -> None:
        self.remove_calls.append(force)


def test_agent_deadline_stops_whole_container_without_term_grace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(docker_backend, "EXEC_KILL_GRACE_SECONDS", 0.1)
    container = _TimeoutContainer()
    backend = DockerBackend(client=SimpleNamespace())

    result = backend.exec_run_with_timeout(
        DockerContainerHandle(container),
        ["/bin/sh", "-c", "detached-writer"],
        timeout=0.001,
        stop_container_on_timeout=True,
    )

    assert result.timed_out is True
    assert result.exit_code == -1
    assert container.stop_timeouts == [0]
    assert not any(
        command.startswith(("kill -TERM ", "kill -KILL "))
        for command in container.exec_commands
    )
    assert container.attrs["State"]["Pid"] == 0
    assert container.state_events[0] == "stop"
    assert "reload" in container.state_events[1:]


def test_agent_deadline_callback_can_strictly_retire_without_fallback_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(docker_backend, "EXEC_KILL_GRACE_SECONDS", 0.1)
    container = _TimeoutContainer()
    backend = DockerBackend(client=SimpleNamespace())
    events: list[str] = []

    def retire() -> bool:
        events.append("retire")
        container.stream_done.set()
        return True

    result = backend.exec_run_with_timeout(
        DockerContainerHandle(container),
        ["/bin/sh", "-c", "paused-writer"],
        timeout=0.001,
        stop_container_on_timeout=True,
        before_stop_on_timeout=retire,
    )

    assert result.timed_out is True
    assert result.exit_code == -1
    assert events == ["retire"]
    assert container.stop_timeouts == []


def test_agent_deadline_callback_true_requires_exec_stream_to_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(docker_backend, "EXEC_KILL_GRACE_SECONDS", 0.01)
    container = _TimeoutContainer()
    backend = DockerBackend(client=SimpleNamespace())

    try:
        with pytest.raises(RuntimeError, match="exec stream did not close"):
            backend.exec_run_with_timeout(
                DockerContainerHandle(container),
                ["/bin/sh", "-c", "still-paused-writer"],
                timeout=0.001,
                stop_container_on_timeout=True,
                before_stop_on_timeout=lambda: True,
            )
    finally:
        container.stream_done.set()

    assert container.stop_timeouts == []


def test_docker_freeze_rejects_container_that_survives_stop() -> None:
    container = _TimeoutContainer()
    container.stop = lambda timeout: container.stop_timeouts.append(timeout)
    backend = DockerBackend(client=SimpleNamespace())

    with pytest.raises(RuntimeError, match="not fully stopped"):
        backend.freeze_container_for_snapshot(DockerContainerHandle(container))

    assert container.stop_timeouts == [0]


def _missing_exit_event_error() -> docker.errors.APIError:
    response = Response()
    response.status_code = 500
    return docker.errors.APIError(
        "stop failed", response=response,
        explanation="tried to kill container, but did not receive an exit event",
    )


def test_docker_freeze_retries_missing_exit_event_before_snapshot() -> None:
    container = _TimeoutContainer()
    original_stop = container.stop

    def stop(timeout):
        if not container.stop_timeouts:
            container.stop_timeouts.append(timeout)
            raise _missing_exit_event_error()
        original_stop(timeout)

    container.stop = stop
    backend = DockerBackend(client=SimpleNamespace())
    handle = DockerContainerHandle(container)

    metadata = backend.freeze_container_for_snapshot(handle)

    assert container.stop_timeouts == [0, 0]
    assert container.remove_calls == []
    assert metadata["stop_attempts"] == 2
    assert metadata["all_work_processes_stopped"] is True
    assert metadata["container_pid"] == 0
    assert backend.freeze_container_for_snapshot(handle) == metadata
    assert container.stop_timeouts == [0, 0]


def test_docker_freeze_missing_exit_event_still_requires_stopped_state() -> None:
    container = _TimeoutContainer()

    def stop(timeout):
        container.stop_timeouts.append(timeout)
        raise _missing_exit_event_error()

    container.stop = stop
    handle = DockerContainerHandle(container)
    backend = DockerBackend(client=SimpleNamespace())

    with pytest.raises(RuntimeError, match="whole-container stop failed"):
        backend.freeze_container_for_snapshot(handle)

    assert container.stop_timeouts == [0] * docker_backend.FINAL_SNAPSHOT_STOP_ATTEMPTS
    assert container.remove_calls == []
    assert handle._snapshot_freeze_metadata is None


@pytest.mark.parametrize("state_field", ["Paused", "Restarting"])
def test_docker_freeze_does_not_retry_unsafe_handoff_state(state_field) -> None:
    container = _TimeoutContainer()

    def stop(timeout):
        container.stop_timeouts.append(timeout)
        container.attrs["State"][state_field] = True
        raise _missing_exit_event_error()

    container.stop = stop
    handle = DockerContainerHandle(container)
    backend = DockerBackend(client=SimpleNamespace())

    with pytest.raises(RuntimeError, match="whole-container stop failed"):
        backend.freeze_container_for_snapshot(handle)

    assert container.stop_timeouts == [0]
    assert handle._snapshot_freeze_metadata is None


def test_docker_freeze_does_not_retry_other_stop_errors() -> None:
    container = _TimeoutContainer()

    def stop(timeout):
        container.stop_timeouts.append(timeout)
        raise docker.errors.APIError("daemon unavailable")

    container.stop = stop
    handle = DockerContainerHandle(container)
    backend = DockerBackend(client=SimpleNamespace())

    with pytest.raises(RuntimeError, match="whole-container stop failed"):
        backend.freeze_container_for_snapshot(handle)

    assert container.stop_timeouts == [0]
    assert handle._snapshot_freeze_metadata is None


def test_docker_freeze_does_not_retry_ambiguous_inspect_failure() -> None:
    container = _TimeoutContainer()

    def stop(timeout):
        container.stop_timeouts.append(timeout)
        raise _missing_exit_event_error()

    container.stop = stop
    container.reload = lambda: (_ for _ in ()).throw(
        docker.errors.APIError("inspect failed")
    )
    handle = DockerContainerHandle(container)
    backend = DockerBackend(client=SimpleNamespace())

    with pytest.raises(RuntimeError, match="whole-container stop failed"):
        backend.freeze_container_for_snapshot(handle, stop_without_inspect=True)

    assert container.stop_timeouts == [0]
    assert handle._snapshot_freeze_metadata is None


def test_docker_freeze_accepts_verified_exit_despite_stop_error() -> None:
    container = _TimeoutContainer()
    original_stop = container.stop

    def stop(timeout):
        original_stop(timeout)
        raise _missing_exit_event_error()

    container.stop = stop
    backend = DockerBackend(client=SimpleNamespace())
    metadata = backend.freeze_container_for_snapshot(DockerContainerHandle(container))

    assert container.stop_timeouts == [0]
    assert metadata["mechanism"] == "docker_verified_stopped"
    assert metadata["stop_attempts"] == 1


def test_docker_freeze_verification_treats_removed_container_as_stopped() -> None:
    container = _TimeoutContainer()
    container.reload = lambda: (_ for _ in ()).throw(
        docker.errors.NotFound("container was removed")
    )
    backend = DockerBackend(client=SimpleNamespace())

    result = backend.verify_container_frozen_for_snapshot(
        DockerContainerHandle(container)
    )

    assert result == {
        "all_work_processes_stopped": True,
        "container_status": "removed",
        "container_pid": 0,
    }


def test_docker_freeze_verification_propagates_other_api_errors() -> None:
    container = _TimeoutContainer()
    container.reload = lambda: (_ for _ in ()).throw(
        docker.errors.APIError("daemon unavailable")
    )
    backend = DockerBackend(client=SimpleNamespace())

    with pytest.raises(docker.errors.APIError, match="daemon unavailable"):
        backend.verify_container_frozen_for_snapshot(
            DockerContainerHandle(container)
        )


def test_cleanup_is_idempotent_after_snapshot_stop() -> None:
    container = _TimeoutContainer()
    backend = DockerBackend(client=SimpleNamespace())
    handle = DockerContainerHandle(container)
    backend.freeze_container_for_snapshot(handle)

    backend.cleanup_container(handle, _Logger())

    assert container.stop_timeouts == [0]
    assert container.remove_calls == [True]


def test_cleanup_is_idempotent_after_container_was_removed() -> None:
    container = _TimeoutContainer()
    container.reload = lambda: (_ for _ in ()).throw(
        docker.errors.NotFound("container was removed")
    )
    backend = DockerBackend(client=SimpleNamespace())
    logger = _RecordingLogger()

    backend.cleanup_container(DockerContainerHandle(container), logger)

    assert container.stop_timeouts == []
    assert container.remove_calls == []
    assert logger.errors == []
    assert any("already removed" in message for message in logger.infos)


@pytest.mark.skipif(
    os.environ.get("SFORGE_RUN_DOCKER_FREEZE_TEST") != "1"
    or not os.environ.get("SFORGE_DOCKER_FREEZE_TEST_IMAGE"),
    reason=(
        "set SFORGE_RUN_DOCKER_FREEZE_TEST=1 and "
        "SFORGE_DOCKER_FREEZE_TEST_IMAGE to run the real Docker test"
    ),
)
@pytest.mark.parametrize("simulate_missing_exit_event", [False, True])
def test_real_docker_detached_writer_freeze_and_stopped_archive(
    tmp_path: Path,
    simulate_missing_exit_event: bool,
) -> None:
    image = os.environ["SFORGE_DOCKER_FREEZE_TEST_IMAGE"]
    backend = DockerBackend()
    name = f"sforge-final-freeze-it-{uuid.uuid4().hex[:12]}"
    handle = None
    evidence: dict[str, object] = {
        "container_name": name, "image": image,
        "simulate_missing_exit_event": simulate_missing_exit_event,
    }

    try:
        handle = backend.create_container(
            image,
            name,
            command="tail -f /dev/null",
        )
        backend.start_container(handle)
        if simulate_missing_exit_event:
            container = backend._raw(handle)
            container.reload()
            before_stop = dict(container.attrs["State"])
            original_stop = container.stop
            original_reload = container.reload
            stop_calls = 0
            stale_state_pending = False

            def stop_with_lost_event(timeout):
                nonlocal stop_calls, stale_state_pending
                stop_calls += 1
                original_stop(timeout=timeout)
                if stop_calls == 1:
                    stale_state_pending = True
                    raise _missing_exit_event_error()

            def reload_with_one_stale_state():
                nonlocal stale_state_pending
                original_reload()
                if stale_state_pending:
                    container.attrs["State"] = dict(before_stop)
                    stale_state_pending = False

            container.stop = stop_with_lost_event
            container.reload = reload_with_one_stale_state
        run_started = time.monotonic()
        timeout = 1.0
        command = [
            "/bin/bash",
            "-c",
            (
                "set -e; root=/tmp/sforge-freeze-it/submission; "
                "mkdir -p \"$root\"; "
                "setsid /bin/bash -c '"
                "trap \"\" TERM HUP; "
                "printf \"%s %s\\n\" \"$$\" "
                "\"$(ps -o pgid= -p $$ | xargs)\" "
                "> /tmp/sforge-freeze-it/submission/writer_identity; "
                "i=0; while :; do i=$((i+1)); "
                "printf \"%s %08d\\n\" \"$(date +%s%N)\" \"$i\" >> "
                "/tmp/sforge-freeze-it/submission/writer.log; "
                "sleep 0.01; done' "
                ">/tmp/sforge-freeze-it/writer.stdout 2>&1 "
                "</dev/null & "
                "printf '%s\\n' \"$!\" > \"$root/writer_pid\"; "
                "trap '' TERM; while :; do sleep 1; done"
            ),
        ]
        result = backend.exec_run_with_timeout(
            handle,
            command,
            timeout=timeout,
            stop_container_on_timeout=True,
        )
        assert result.timed_out is True
        assert result.exit_code == -1
        if simulate_missing_exit_event:
            assert handle._snapshot_freeze_metadata["stop_attempts"] == 2

        state = backend.verify_container_frozen_for_snapshot(handle)
        assert state["container_pid"] == 0
        assert state["all_work_processes_stopped"] is True

        writer_path = PurePosixPath(
            "/tmp/sforge-freeze-it/submission/writer.log"
        )
        first = runner._read_file_from_container_archive(
            backend, handle, writer_path
        )
        time.sleep(0.35)
        second = runner._read_file_from_container_archive(
            backend, handle, writer_path
        )
        assert first
        assert second == first

        archive, metadata = runner._capture_frozen_final_snapshot(
            backend,
            handle,
            SimpleNamespace(
                cwd="/tmp/sforge-freeze-it",
                submit_paths=["submission"],
                submit_exclude=[],
            ),
            _Agent(),
            tmp_path,
            run_started,
            _Logger(),
            run_deadline_monotonic=run_started + timeout,
            auto_eval_stop=threading.Event(),
            auto_eval_thread=None,
        )
        archive_sha = hashlib.sha256(archive).hexdigest()
        assert metadata["archive_sha256"] == archive_sha
        assert metadata["archive"]["sha256"] == archive_sha
        assert hashlib.sha256(
            (tmp_path / "final_archive.tar.gz").read_bytes()
        ).hexdigest() == archive_sha
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
            archived_writer = bundle.extractfile(
                "submission/writer.log"
            )
            assert archived_writer is not None
            assert archived_writer.read() == first

        last_write_at = int(first.splitlines()[-1].split()[0]) / 1e9
        stop_requested_at = float(
            metadata["freeze"]["freeze_requested_at"]
        )
        last_write_after_stop_request = last_write_at - stop_requested_at
        # The only allowance here is Docker/API scheduling jitter.  There is
        # no configurable TERM grace and the writer is in a different session.
        assert last_write_after_stop_request <= 0.1

        final_state = backend.verify_container_frozen_for_snapshot(handle)
        assert final_state["container_pid"] == 0
        evidence.update(
            {
                "exec_elapsed_seconds": result.elapsed_seconds,
                "writer_bytes": len(first),
                "writer_sha256": hashlib.sha256(first).hexdigest(),
                "writer_unchanged_after_seconds": 0.35,
                "last_write_after_stop_request_seconds": (
                    last_write_after_stop_request
                ),
                "container_pid_after_stop": final_state["container_pid"],
                "freeze": metadata["freeze"],
                "deadline": metadata["deadline"],
                "capture": metadata["capture"],
                "archive_bytes": len(archive),
                "archive_sha256": archive_sha,
                "persisted_metadata_sha256": hashlib.sha256(
                    (tmp_path / "final_snapshot.json").read_bytes()
                ).hexdigest(),
            }
        )
    finally:
        backend.cleanup_container(handle, _Logger())

    try:
        backend.client.containers.get(name)
    except docker.errors.NotFound:
        evidence["container_exists_after_cleanup"] = False
    else:
        evidence["container_exists_after_cleanup"] = True
        pytest.fail(f"integration container remained after cleanup: {name}")

    print("DOCKER_FREEZE_EVIDENCE=" + json.dumps(evidence, sort_keys=True))


def test_seal_boundary_fields_record_the_single_gpu_seal_time() -> None:
    """The seal (admission closed, Work paused) is the agent's work boundary;
    a drained Judge may legitimately delay the frozen snapshot after it."""
    from sforge.harness.run_agent import seal_boundary_fields

    assert seal_boundary_fields(None, None, 10.0) == {}
    assert seal_boundary_fields(1789.5, 910.25, 10.0) == {
        "seal_requested_at": 1789.5,
        "seal_requested_elapsed_seconds": 900.25,
    }
    assert seal_boundary_fields(1789.5, 5.0, 10.0)["seal_requested_elapsed_seconds"] == 0.0
