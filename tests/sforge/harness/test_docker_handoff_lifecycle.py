from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import docker.errors
import pytest

from sforge.harness.backend import (
    EVALUATION_CONTRACT_SHA256_ANNOTATION,
    HANDOFF_ALLOCATION_KEY_ANNOTATION,
    HANDOFF_DOCKER_PAUSE_PROTOCOL,
    HANDOFF_LEASE_ID_ANNOTATION,
    HANDOFF_PROTOCOL_ANNOTATION,
    HANDOFF_RUN_ID_ANNOTATION,
)
from sforge.harness.backend.docker_backend import (
    DockerBackend,
    DockerContainerHandle,
)
from sforge.harness.backend.k8s_backend import K8sBackend


_LOOKUP_CONTAINER_ID = "a" * 64


def _binding_annotations() -> dict[str, str]:
    return {
        HANDOFF_RUN_ID_ANNOTATION: "run-123",
        HANDOFF_LEASE_ID_ANNOTATION: "lease-456",
        HANDOFF_ALLOCATION_KEY_ANNOTATION: "allocation-sha256:" + "b" * 64,
        HANDOFF_PROTOCOL_ANNOTATION: HANDOFF_DOCKER_PAUSE_PROTOCOL,
        EVALUATION_CONTRACT_SHA256_ANNOTATION: "sha256:" + "c" * 64,
    }


def _state(
    *,
    running: bool,
    pid: int,
    status: str,
    paused: bool = False,
    restarting: bool = False,
    exit_code: int = 0,
    oom_killed: bool = False,
    dead: bool = False,
    error: str = "",
) -> dict[str, object]:
    return {
        "Running": running,
        "Restarting": restarting,
        "Paused": paused,
        "Pid": pid,
        "Status": status,
        "OOMKilled": oom_killed,
        "Dead": dead,
        "ExitCode": exit_code,
        "Error": error,
    }


class _Container:
    def __init__(
        self,
        state: dict[str, object],
        *,
        pause_transitions: bool = True,
        unpause_transitions: bool = True,
    ) -> None:
        self.id = "container-exact-id"
        self.name = "work"
        self.status = state["Status"]
        self.attrs = {
            "State": deepcopy(state),
            "Config": {"Labels": _binding_annotations()},
        }
        self.events: list[str] = []
        self.pause_transitions = pause_transitions
        self.unpause_transitions = unpause_transitions
        self.exec_session_id = "long-running-exec"
        self.exec_alive = True
        self.writable_layer = {"answer.txt": "preserved"}

    def reload(self) -> None:
        self.events.append("reload")

    def pause(self) -> None:
        self.events.append("pause")
        if self.pause_transitions:
            self.attrs["State"] = _state(
                running=True,
                paused=True,
                pid=8123,
                status="paused",
                oom_killed=self.attrs["State"]["OOMKilled"],
            )
            self.status = "paused"

    def unpause(self) -> None:
        self.events.append("unpause")
        if self.unpause_transitions:
            self.attrs["State"] = _state(
                running=True,
                pid=8123,
                status="running",
                oom_killed=self.attrs["State"]["OOMKilled"],
            )
            self.status = "running"

    def stop(self, *, timeout: int) -> None:
        self.events.append("stop")
        self.exec_alive = False
        self.attrs["State"] = _state(
            running=False,
            pid=0,
            status="exited",
            exit_code=143,
        )
        self.status = "exited"

    def start(self) -> None:
        self.events.append("start")
        self.exec_session_id = "replacement-exec"
        self.exec_alive = True
        self.attrs["State"] = _state(
            running=True,
            pid=9123,
            status="running",
        )
        self.status = "running"


def _backend_and_handle(
    container: _Container,
) -> tuple[DockerBackend, DockerContainerHandle]:
    backend = DockerBackend.__new__(DockerBackend)
    return backend, DockerContainerHandle(container)


def _pause_evidence(container_id: str = "container-exact-id") -> dict[str, object]:
    return {
        "container_id": container_id,
        "mechanism": "docker_pause",
        "pause_observed": True,
    }


def test_pause_then_unpause_preserves_the_same_live_exec_and_layer() -> None:
    container = _Container(_state(running=True, pid=8123, status="running"))
    backend, handle = _backend_and_handle(container)

    paused = backend.pause_container_for_handoff(handle)

    assert container.events == ["reload", "pause", "reload"]
    assert paused == {
        "mechanism": "docker_pause",
        "container_id": "container-exact-id",
        "container_paused": True,
        "container_status": "paused",
        "container_pid": 8123,
        "container_exit_code": 0,
        "container_oom_killed": False,
        "pause_observed": True,
        "was_paused": False,
    }
    assert "accelerator_released" not in paused
    assert handle._handoff_pause_metadata == paused
    assert handle._snapshot_freeze_metadata is None

    running = backend.unpause_container_after_handoff(handle)

    assert container.events == [
        "reload",
        "pause",
        "reload",
        "reload",
        "unpause",
        "reload",
    ]
    assert running["mechanism"] == "docker_unpause"
    assert running["container_running"] is True
    assert container.exec_session_id == "long-running-exec"
    assert container.exec_alive is True
    assert container.writable_layer == {"answer.txt": "preserved"}
    assert "stop" not in container.events
    assert "start" not in container.events


def test_child_oom_kill_does_not_fence_a_live_container_handoff() -> None:
    """State.OOMKilled is sticky after any OOM kill in the container cgroup.

    An agent process killed for exceeding the Work memory limit leaves init
    alive (running, exit code 0); pause and resume must still work, with the
    OOM kept as evidence instead of fencing the lease.
    """
    container = _Container(
        _state(running=True, pid=8123, status="running", oom_killed=True)
    )
    backend, handle = _backend_and_handle(container)

    paused = backend.pause_container_for_handoff(handle)
    assert paused["container_paused"] is True
    assert paused["container_oom_killed"] is True
    assert backend.verify_container_paused_for_handoff(handle)[
        "container_oom_killed"
    ] is True

    running = backend.unpause_container_after_handoff(handle)
    assert running["container_running"] is True
    assert running["container_oom_killed"] is True
    assert container.events == [
        "reload",
        "pause",
        "reload",
        "reload",
        "reload",
        "unpause",
        "reload",
    ]
    assert container.exec_alive is True


@pytest.mark.parametrize(
    "state, match",
    [
        (
            _state(running=False, pid=0, status="exited", exit_code=137, oom_killed=True),
            "ambiguous state",
        ),
        (
            _state(running=True, pid=8123, status="running", oom_killed=True, dead=True),
            "unhealthy state",
        ),
        (
            _state(running=True, pid=8123, status="running", oom_killed=True, error="runtime failure"),
            "unhealthy state",
        ),
    ],
)
def test_handoff_still_fails_closed_when_the_container_itself_died(
    state: dict[str, object], match: str
) -> None:
    container = _Container(state)
    backend, handle = _backend_and_handle(container)

    with pytest.raises(RuntimeError, match=match):
        backend.pause_container_for_handoff(handle)
    assert "pause" not in container.events


def test_pause_is_idempotent_only_with_same_handle_evidence() -> None:
    container = _Container(
        _state(running=True, paused=True, pid=8123, status="paused")
    )
    backend, handle = _backend_and_handle(container)

    with pytest.raises(RuntimeError, match="same-handle pause evidence"):
        backend.pause_container_for_handoff(handle)

    handle._handoff_pause_metadata = _pause_evidence()
    metadata = backend.pause_container_for_handoff(handle)

    assert metadata["mechanism"] == "docker_verified_paused"
    assert metadata["was_paused"] is True
    assert container.events == ["reload", "reload", "reload"]
    assert "pause" not in container.events


def test_pause_fails_when_docker_does_not_reach_paused_state() -> None:
    container = _Container(
        _state(running=True, pid=8123, status="running"),
        pause_transitions=False,
    )
    backend, handle = _backend_and_handle(container)

    with pytest.raises(RuntimeError, match="not paused"):
        backend.pause_container_for_handoff(handle)

    assert container.events == ["reload", "pause", "reload"]
    assert "stop" not in container.events


def test_unpause_requires_pause_evidence_and_rejects_stopped_state() -> None:
    stopped = _Container(_state(running=False, pid=0, status="exited"))
    backend, stopped_handle = _backend_and_handle(stopped)
    stopped_handle._handoff_pause_metadata = _pause_evidence()

    with pytest.raises(RuntimeError, match="ambiguous state"):
        backend.unpause_container_after_handoff(stopped_handle)

    running = _Container(_state(running=True, pid=8123, status="running"))
    backend, running_handle = _backend_and_handle(running)
    with pytest.raises(RuntimeError, match="same-handle pause evidence"):
        backend.unpause_container_after_handoff(running_handle)

    assert "start" not in stopped.events + running.events


@pytest.mark.parametrize(
    "state",
    [
        _state(running=False, paused=True, pid=8123, status="paused"),
        _state(
            running=True,
            paused=True,
            restarting=True,
            pid=8123,
            status="paused",
        ),
        _state(running=True, paused=True, pid=0, status="paused"),
        _state(running=True, paused=True, pid=8123, status="running"),
        _state(
            running=True,
            paused=True,
            pid=8123,
            status="paused",
            exit_code=137,
            oom_killed=True,
        ),
        _state(
            running=True,
            paused=True,
            pid=8123,
            status="paused",
            error="runtime failure",
        ),
    ],
)
def test_paused_verification_is_strict(state: dict[str, object]) -> None:
    container = _Container(state)
    backend, handle = _backend_and_handle(container)
    handle._handoff_pause_metadata = _pause_evidence()

    with pytest.raises(RuntimeError, match="not paused|unhealthy state"):
        backend.verify_container_paused_for_handoff(handle)


def test_final_freeze_refuses_to_implicitly_unpause_handoff() -> None:
    container = _Container(_state(running=True, pid=8123, status="running"))
    backend, handle = _backend_and_handle(container)
    backend.pause_container_for_handoff(handle)
    events_after_pause = list(container.events)

    with pytest.raises(RuntimeError, match="paused.*final snapshot"):
        backend.freeze_container_for_snapshot(handle)

    assert container.events == events_after_pause
    assert handle._snapshot_freeze_metadata is None
    assert container.exec_alive is True
    assert "unpause" not in container.events
    assert "stop" not in container.events


def test_deadline_final_freeze_also_rejects_paused_handoff() -> None:
    container = _Container(
        _state(running=True, paused=True, pid=8123, status="paused")
    )
    backend, handle = _backend_and_handle(container)
    handle._handoff_pause_metadata = _pause_evidence()

    with pytest.raises(RuntimeError, match="persistent no-resume tombstone"):
        backend.freeze_container_for_snapshot(
            handle,
            stop_without_inspect=True,
        )

    assert container.events == []
    assert container.exec_alive is True
    assert "unpause" not in container.events


def test_final_handoff_snapshot_adopts_exact_paused_container_without_resume() -> None:
    container = _Container(
        _state(running=True, paused=True, pid=8123, status="paused")
    )
    container.id = _LOOKUP_CONTAINER_ID
    registry = _DestroyContainers(container)
    backend = DockerBackend(client=SimpleNamespace(containers=registry))
    handle = DockerContainerHandle(container)

    metadata = backend.freeze_container_for_final_handoff_snapshot(
        handle,
        expected_annotations=_binding_annotations(),
        stop_without_inspect=True,
    )
    verified = backend.verify_container_frozen_for_snapshot(handle)

    assert metadata["mechanism"] == "docker_pause_never_resume_final"
    assert metadata["all_work_processes_frozen"] is True
    assert metadata["all_work_processes_stopped"] is False
    assert metadata["container_paused"] is True
    assert verified["container_paused"] is True
    assert verified["container_id"] == _LOOKUP_CONTAINER_ID
    assert registry.lookups == [_LOOKUP_CONTAINER_ID]
    assert "pause" not in container.events
    assert "unpause" not in container.events
    assert "stop" not in container.events


def test_final_handoff_snapshot_adopts_work_stopped_by_another_handle() -> None:
    container = _Container(_state(running=True, pid=8123, status="running"))
    container.id = _LOOKUP_CONTAINER_ID
    registry = _DestroyContainers(container)
    backend = DockerBackend(client=SimpleNamespace(containers=registry))
    freezing_handle = DockerContainerHandle(container)

    first_metadata = backend.freeze_container_for_snapshot(freezing_handle)
    assert first_metadata["mechanism"] == "docker_stop_timeout_0"
    assert container.attrs["State"]["Status"] == "exited"

    # The coordinator owns a distinct handle with no in-process freeze
    # evidence.  Exact ID and labels authenticate the stopped container, and
    # final adoption copies fresh verified evidence onto that handle.
    coordinator_handle = DockerContainerHandle(container)
    adopted = backend.freeze_container_for_final_handoff_snapshot(
        coordinator_handle,
        expected_annotations=_binding_annotations(),
        stop_without_inspect=True,
    )
    verified = backend.verify_container_frozen_for_snapshot(
        coordinator_handle
    )

    assert adopted["mechanism"] == "docker_verified_stopped"
    assert adopted["all_work_processes_stopped"] is True
    assert verified["all_work_processes_stopped"] is True
    assert registry.lookups == [_LOOKUP_CONTAINER_ID]
    assert container.events.count("stop") == 1
    assert "unpause" not in container.events


@pytest.mark.parametrize(
    "state",
    [
        _state(
            running=True,
            restarting=True,
            pid=8123,
            status="restarting",
        ),
        _state(running=False, pid=8123, status="exited"),
    ],
    ids=["restarting", "ambiguous-stopped-pid"],
)
def test_final_handoff_snapshot_rejects_ambiguous_nonpaused_state(
    state: dict[str, object],
) -> None:
    container = _Container(state)
    container.id = _LOOKUP_CONTAINER_ID
    registry = _DestroyContainers(container)
    backend = DockerBackend(client=SimpleNamespace(containers=registry))
    coordinator_handle = DockerContainerHandle(container)

    with pytest.raises(RuntimeError, match="ambiguous state"):
        backend.freeze_container_for_final_handoff_snapshot(
            coordinator_handle,
            expected_annotations=_binding_annotations(),
            stop_without_inspect=True,
        )

    assert registry.lookups == [_LOOKUP_CONTAINER_ID]
    assert coordinator_handle._snapshot_freeze_metadata is None
    assert "stop" not in container.events
    assert "unpause" not in container.events


def test_final_handoff_snapshot_rejects_binding_mismatch_before_adoption() -> None:
    container = _Container(
        _state(running=True, paused=True, pid=8123, status="paused")
    )
    container.id = _LOOKUP_CONTAINER_ID
    registry = _DestroyContainers(container)
    backend = DockerBackend(client=SimpleNamespace(containers=registry))
    handle = DockerContainerHandle(container)
    expected = _binding_annotations()
    expected[HANDOFF_LEASE_ID_ANNOTATION] = "different-lease"

    with pytest.raises(RuntimeError, match="annotation mismatch"):
        backend.freeze_container_for_final_handoff_snapshot(
            handle,
            expected_annotations=expected,
        )

    assert handle._snapshot_freeze_metadata is None
    assert "pause" not in container.events
    assert "unpause" not in container.events
    assert "stop" not in container.events


def test_final_snapshot_irreversibly_blocks_pause_unpause_and_start() -> None:
    container = _Container(_state(running=True, pid=8123, status="running"))
    backend, handle = _backend_and_handle(container)
    snapshot = backend.freeze_container_for_snapshot(handle)
    events_after_freeze = list(container.events)
    handle._handoff_pause_metadata = _pause_evidence()

    with pytest.raises(RuntimeError, match="irreversibly frozen.*final snapshot"):
        backend.pause_container_for_handoff(handle)
    with pytest.raises(RuntimeError, match="irreversibly frozen.*final snapshot"):
        backend.unpause_container_after_handoff(handle)
    with pytest.raises(RuntimeError, match="irreversibly frozen.*final snapshot"):
        backend.start_container(handle)

    assert container.events == events_after_freeze
    assert handle._snapshot_freeze_metadata == snapshot


class _DestroyContainers:
    def __init__(self, container: _Container) -> None:
        self.container = container
        self.removed = False
        self.lookups: list[str] = []

    def get(self, container_id: str) -> _Container:
        self.lookups.append(container_id)
        if self.removed:
            raise docker.errors.NotFound("removed")
        return self.container


def test_strict_handoff_destroy_stops_removes_then_verifies_absence() -> None:
    container = _Container(_state(running=True, pid=8123, status="running"))
    container.id = _LOOKUP_CONTAINER_ID
    registry = _DestroyContainers(container)

    def remove(*, force: bool) -> None:
        assert force is True
        container.events.append("remove")
        registry.removed = True

    container.remove = remove
    backend = DockerBackend(client=SimpleNamespace(containers=registry))
    handle = DockerContainerHandle(container)

    metadata = backend.destroy_container_for_handoff(handle)
    retry_metadata = backend.destroy_container_for_handoff(handle)

    assert container.events == ["reload", "stop", "remove"]
    assert registry.lookups == [_LOOKUP_CONTAINER_ID, _LOOKUP_CONTAINER_ID]
    assert metadata == {
        "mechanism": "docker_stop_remove_verified_absent",
        "container_id": _LOOKUP_CONTAINER_ID,
        "container_destroyed": True,
    }
    assert retry_metadata == metadata
    assert "accelerator_idle" not in metadata


def test_strict_handoff_destroy_propagates_remove_failure() -> None:
    container = _Container(_state(running=True, pid=8123, status="running"))
    container.id = _LOOKUP_CONTAINER_ID
    registry = _DestroyContainers(container)

    def remove(*, force: bool) -> None:
        raise RuntimeError("remove failed")

    container.remove = remove
    backend = DockerBackend(client=SimpleNamespace(containers=registry))

    with pytest.raises(RuntimeError, match="remove failed"):
        backend.destroy_container_for_handoff(DockerContainerHandle(container))

    # A remove error is not authoritative: strict destruction performs an
    # exact-ID lookup and reports the combined failure only when the ID still
    # resolves.
    assert registry.lookups == [_LOOKUP_CONTAINER_ID]


def test_strict_handoff_destroy_removes_created_or_stopped_container() -> None:
    container = _Container(_state(running=False, pid=0, status="created"))
    container.id = _LOOKUP_CONTAINER_ID
    registry = _DestroyContainers(container)

    def stop(*, timeout: int) -> None:
        assert timeout == 0
        container.events.append("stop")
        raise RuntimeError("container is not running")

    def remove(*, force: bool) -> None:
        assert force is True
        container.events.append("remove")
        registry.removed = True

    container.stop = stop
    container.remove = remove
    backend = DockerBackend(client=SimpleNamespace(containers=registry))

    metadata = backend.destroy_container_for_handoff(
        DockerContainerHandle(container)
    )

    assert metadata["container_destroyed"] is True
    assert container.events == ["reload", "stop", "remove"]
    assert registry.lookups == [_LOOKUP_CONTAINER_ID]


def test_strict_handoff_destroy_accepts_verified_already_absent() -> None:
    container = _Container(_state(running=False, pid=0, status="created"))
    container.id = _LOOKUP_CONTAINER_ID
    registry = _DestroyContainers(container)
    registry.removed = True

    def reload() -> None:
        container.events.append("reload")
        raise docker.errors.NotFound("removed")

    container.reload = reload
    backend = DockerBackend(client=SimpleNamespace(containers=registry))

    metadata = backend.destroy_container_for_handoff(
        DockerContainerHandle(container)
    )

    assert metadata == {
        "mechanism": "docker_already_absent_verified",
        "container_id": _LOOKUP_CONTAINER_ID,
        "container_destroyed": True,
    }
    assert registry.lookups == [_LOOKUP_CONTAINER_ID]


def test_strict_handoff_destroy_fails_if_exact_id_still_resolves() -> None:
    container = _Container(_state(running=True, pid=8123, status="running"))
    container.id = _LOOKUP_CONTAINER_ID
    registry = _DestroyContainers(container)

    def remove(*, force: bool) -> None:
        container.events.append("remove")

    container.remove = remove
    backend = DockerBackend(client=SimpleNamespace(containers=registry))

    with pytest.raises(RuntimeError, match="removal was not verified"):
        backend.destroy_container_for_handoff(DockerContainerHandle(container))


class _LookupContainers:
    def __init__(self, container: _Container) -> None:
        self.container = container
        self.lookups: list[str] = []

    def get(self, container_id: str) -> _Container:
        self.lookups.append(container_id)
        return self.container


def test_exact_lookup_verifies_reserved_bindings_but_not_pause_evidence() -> None:
    container = _Container(
        _state(running=True, paused=True, pid=8123, status="paused")
    )
    container.id = _LOOKUP_CONTAINER_ID
    containers = _LookupContainers(container)
    backend = DockerBackend(client=SimpleNamespace(containers=containers))

    handle = backend.get_container_handle(
        _LOOKUP_CONTAINER_ID,
        expected_annotations=_binding_annotations(),
    )

    assert handle.id == _LOOKUP_CONTAINER_ID
    assert handle._handoff_pause_metadata is None
    assert handle._snapshot_freeze_metadata is None
    with pytest.raises(RuntimeError, match="same-handle pause evidence"):
        backend.verify_container_paused_for_handoff(handle)
    with pytest.raises(RuntimeError, match="same-handle pause evidence"):
        backend.unpause_container_after_handoff(handle)
    with pytest.raises(RuntimeError, match="persistent lease tombstone"):
        backend.start_container(handle)
    assert "unpause" not in container.events
    assert "start" not in container.events


def test_exact_lookup_rejects_wrong_identity_or_binding() -> None:
    container = _Container(_state(running=True, pid=8123, status="running"))
    container.id = "b" * 64
    backend = DockerBackend(
        client=SimpleNamespace(containers=_LookupContainers(container))
    )
    with pytest.raises(RuntimeError, match="different identity"):
        backend.get_container_handle(
            _LOOKUP_CONTAINER_ID,
            expected_annotations=_binding_annotations(),
        )

    container.id = _LOOKUP_CONTAINER_ID
    wrong = _binding_annotations()
    wrong[HANDOFF_LEASE_ID_ANNOTATION] = "wrong-lease"
    with pytest.raises(RuntimeError, match="binding annotation mismatch"):
        backend.get_container_handle(
            _LOOKUP_CONTAINER_ID,
            expected_annotations=wrong,
        )


@pytest.mark.parametrize(
    "missing_key",
    sorted({
        HANDOFF_RUN_ID_ANNOTATION,
        HANDOFF_LEASE_ID_ANNOTATION,
        HANDOFF_ALLOCATION_KEY_ANNOTATION,
        HANDOFF_PROTOCOL_ANNOTATION,
        EVALUATION_CONTRACT_SHA256_ANNOTATION,
    }),
)
def test_exact_lookup_requires_every_reserved_binding(missing_key: str) -> None:
    expected = _binding_annotations()
    expected.pop(missing_key)
    container = _Container(_state(running=True, pid=8123, status="running"))
    backend = DockerBackend(
        client=SimpleNamespace(containers=_LookupContainers(container))
    )

    with pytest.raises(ValueError, match="missing trusted handoff"):
        backend.get_container_handle(
            _LOOKUP_CONTAINER_ID,
            expected_annotations=expected,
        )


@pytest.mark.parametrize(
    "container_id",
    ["work", "a" * 12, "A" * 64, "sha256:" + "a" * 64],
)
def test_exact_lookup_rejects_names_and_noncanonical_ids(
    container_id: str,
) -> None:
    container = _Container(_state(running=True, pid=8123, status="running"))
    containers = _LookupContainers(container)
    backend = DockerBackend(client=SimpleNamespace(containers=containers))

    with pytest.raises(ValueError, match="canonical 64-hex exact ID"):
        backend.get_container_handle(
            container_id,
            expected_annotations=_binding_annotations(),
        )

    assert containers.lookups == []


def test_k8s_handoff_lifecycle_is_explicitly_unsupported() -> None:
    backend = object.__new__(K8sBackend)

    with pytest.raises(NotImplementedError, match="transient container handoff"):
        backend.pause_container_for_handoff(object())
    with pytest.raises(NotImplementedError, match="exact container ID lookup"):
        backend.get_container_handle(
            "pod-id",
            expected_annotations=_binding_annotations(),
        )
    with pytest.raises(NotImplementedError, match="strict handoff destruction"):
        backend.destroy_container_for_handoff(object())
