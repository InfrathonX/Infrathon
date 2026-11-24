"""Reclaiming a GPU that paused Work still holds during a Judge handoff."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from sforge.harness import judge_server
from sforge.harness.accelerator_observer import AcceleratorBusyError
from sforge.harness.backend.docker_backend import DockerBackend
from sforge.harness.config import SForgeConfig, load_config
from sforge.harness.judge_server import SubmissionStatus
from sforge.harness.resource_lease import LeaseState

import test_single_gpu_handoff as handoff
from test_single_gpu_handoff import (
    GPU_UUID,
    WORK_ID,
    _allocation,
    _attestation,
    _Background,
    _Report,
    _SamplingObserver,
    _state,
)


OTHER_ID = "f" * 64


# --- observer -------------------------------------------------------------


def test_busy_error_reports_pids_from_every_allocated_device() -> None:
    second = "GPU-11111111-2222-3333-4444-555555555555"

    class TwoDeviceObserver(_SamplingObserver):
        pass

    observer = TwoDeviceObserver([((41,), ()), ((), (42, 41))])
    allocation = handoff.AcceleratorAllocation("nvidia", (GPU_UUID, second))

    with pytest.raises(AcceleratorBusyError) as caught:
        observer.attest_idle(allocation, phase="after-work-pause")

    assert caught.value.pids == (41, 42)
    assert caught.value.recoverable is False
    assert observer.seen == [GPU_UUID, second]


# --- config ---------------------------------------------------------------


def test_reclaim_is_on_by_default_and_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert SForgeConfig().accelerator_reclaim_paused_work is True
    monkeypatch.setenv("SFORGE_ACCELERATOR_RECLAIM_PAUSED_WORK", "0")
    assert load_config().accelerator_reclaim_paused_work is False
    monkeypatch.setenv("SFORGE_ACCELERATOR_RECLAIM_PAUSED_WORK", "maybe")
    with pytest.raises(ValueError, match="RECLAIM_PAUSED_WORK"):
        load_config()


# --- judge handoff --------------------------------------------------------


class _BusyOnceObserver:
    """Busy with the given PIDs on the first observation of one phase."""

    def __init__(self, events, phase, pids=(4242,)):
        self.events = events
        self.phase = phase
        self.pids = pids
        self.raised = False

    def attest_idle(self, allocation, *, phase):
        self.events.append(f"nvml:{phase}")
        if phase == self.phase and not self.raised:
            self.raised = True
            raise AcceleratorBusyError("work still holds the GPU", pids=self.pids)
        return _attestation(phase)


def _fake_judge(backend, events):
    def fake_judge_submission(**kwargs):
        lifecycle = kwargs["container_lifecycle"]
        judge = SimpleNamespace(id="b" * 64, name="judge")
        events.append("create-judge")
        lifecycle.judge_started(judge)
        lifecycle.judge_cleaning(judge)
        backend.destroy_container_for_handoff(judge)
        lifecycle.judge_destroyed(judge)
        return _Report()

    return fake_judge_submission


def _submit(state, token):
    _, _, round_id, cpu, mem, _ = state.consume_round(token, "agent")
    background = _Background()
    submission_id = state.submit(
        "task",
        b"archive",
        "run",
        round_id,
        judge_cpu_limit=cpu,
        judge_mem_limit=mem,
        session_token=token,
        background_tasks=background,
    )
    background.run()
    return submission_id


def test_pre_judge_busy_from_paused_work_is_reclaimed_and_judge_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state, backend, events, token, session = _state(monkeypatch, tmp_path)
    reclaimed = []

    def reclaim(handle, pids):
        assert handle.id == WORK_ID
        reclaimed.append(tuple(pids))
        events.append("reclaim-work-gpu")
        return {"container_id": handle.id, "killed_host_pids": list(pids)}

    backend.reclaim_accelerator_from_paused_handoff = reclaim
    state.accelerator_observer = _BusyOnceObserver(
        events, "after_work_pause_before_judge", pids=(4242, 4243)
    )
    monkeypatch.setattr(judge_server, "judge_submission", _fake_judge(backend, events))

    submission_id = _submit(state, token)

    assert reclaimed == [(4242, 4243)]
    assert state.submissions[submission_id]["status"] == SubmissionStatus.COMPLETED
    assert events[1:7] == [
        "pause-work",
        "verify-work-paused",
        "nvml:after_work_pause_before_judge",
        "reclaim-work-gpu",
        "nvml:after_work_pause_before_judge_after_reclaim",
        "create-judge",
    ]
    assert session.lease.snapshot().state == LeaseState.WORK_ACTIVE


def test_post_judge_busy_from_paused_work_is_reclaimed_before_resume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state, backend, events, token, session = _state(monkeypatch, tmp_path)

    def reclaim(handle, pids):
        events.append("reclaim-work-gpu")
        return {"container_id": handle.id, "killed_host_pids": list(pids)}

    backend.reclaim_accelerator_from_paused_handoff = reclaim
    state.accelerator_observer = _BusyOnceObserver(
        events, "after_judge_destroy_before_work_resume"
    )
    monkeypatch.setattr(judge_server, "judge_submission", _fake_judge(backend, events))

    submission_id = _submit(state, token)

    assert state.submissions[submission_id]["status"] == SubmissionStatus.COMPLETED
    destroy = events.index("strict-destroy:" + "b" * 64)
    reclaim_at = events.index("reclaim-work-gpu")
    unpause = events.index("unpause-work")
    assert destroy < reclaim_at < unpause
    assert session.lease.snapshot().state == LeaseState.WORK_ACTIVE


def test_refused_reclaim_still_fences_and_never_starts_judge(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state, backend, events, token, session = _state(monkeypatch, tmp_path)

    def reclaim(handle, pids):
        raise RuntimeError("host PID 4242 does not belong to paused Work")

    backend.reclaim_accelerator_from_paused_handoff = reclaim
    state.accelerator_observer = _BusyOnceObserver(events, "after_work_pause_before_judge")
    monkeypatch.setattr(judge_server, "judge_submission", _fake_judge(backend, events))

    _submit(state, token)

    assert "create-judge" not in events
    assert session.lease.snapshot().state == LeaseState.FENCED


def test_still_busy_after_reclaim_fences(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state, backend, events, token, session = _state(monkeypatch, tmp_path)
    backend.reclaim_accelerator_from_paused_handoff = (
        lambda handle, pids: {"container_id": handle.id}
    )

    class AlwaysBusy:
        def attest_idle(self, allocation, *, phase):
            events.append(f"nvml:{phase}")
            raise AcceleratorBusyError("busy", pids=(4242,))

    state.accelerator_observer = AlwaysBusy()
    monkeypatch.setattr(judge_server, "judge_submission", _fake_judge(backend, events))

    _submit(state, token)

    assert "nvml:after_work_pause_before_judge_after_reclaim" in events
    assert "create-judge" not in events
    assert session.lease.snapshot().state == LeaseState.FENCED


def test_disabled_reclaim_fences_without_touching_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state, backend, events, token, session = _state(monkeypatch, tmp_path)
    state.config.accelerator_reclaim_paused_work = False

    def reclaim(handle, pids):  # pragma: no cover - must not run
        raise AssertionError("reclaim must not run when disabled")

    backend.reclaim_accelerator_from_paused_handoff = reclaim
    state.accelerator_observer = _BusyOnceObserver(events, "after_work_pause_before_judge")
    monkeypatch.setattr(judge_server, "judge_submission", _fake_judge(backend, events))

    _submit(state, token)

    assert "create-judge" not in events
    assert session.lease.snapshot().state == LeaseState.FENCED


def test_backend_without_reclaim_support_keeps_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state, backend, events, token, session = _state(monkeypatch, tmp_path)

    def reclaim(handle, pids):
        raise NotImplementedError("k8s does not support accelerator reclaim")

    backend.reclaim_accelerator_from_paused_handoff = reclaim
    state.accelerator_observer = _BusyOnceObserver(events, "after_work_pause_before_judge")
    monkeypatch.setattr(judge_server, "judge_submission", _fake_judge(backend, events))

    _submit(state, token)

    assert session.lease.snapshot().state == LeaseState.FENCED


# --- docker backend attribution -------------------------------------------


def _proc(root: Path, pid: int, *, cgroup: str, nspid: str, state: str = "S") -> None:
    d = root / str(pid)
    d.mkdir(parents=True)
    (d / "cgroup").write_text(cgroup + "\n")
    (d / "status").write_text(f"Name:\tpython3\nNSpid:\t{nspid}\n")
    (d / "stat").write_text(f"{pid} (python3 (x)) {state} 1 2 3\n")


def _docker_backend(tmp_path: Path) -> DockerBackend:
    backend = DockerBackend.__new__(DockerBackend)
    backend._PROC_ROOT = tmp_path / "proc"
    return backend


def test_attribution_maps_owned_pids_to_container_namespace(tmp_path: Path) -> None:
    backend = _docker_backend(tmp_path)
    _proc(backend._PROC_ROOT, 4242, cgroup=f"0::/system.slice/docker-{WORK_ID}.scope", nspid="4242\t57")
    _proc(backend._PROC_ROOT, 4243, cgroup=f"12:memory:/docker/{WORK_ID}/sub", nspid="4243\t58")

    assert backend._paused_container_pid_mapping(WORK_ID, (4242, 4243)) == {4242: 57, 4243: 58}


@pytest.mark.parametrize(
    "cgroup, nspid",
    [
        (f"0::/system.slice/docker-{OTHER_ID}.scope", "4242\t57"),  # another container
        ("0::/user.slice/user-1001.slice/session-1.scope", "4242"),  # a host process
        (f"0::/system.slice/docker-{WORK_ID}.scope", "4242"),  # not in a nested PID ns
        (f"0::/system.slice/docker-{WORK_ID}.scope", "4242\t1"),  # the container's init
    ],
)
def test_attribution_refuses_anything_outside_the_work_container(
    tmp_path: Path, cgroup: str, nspid: str
) -> None:
    backend = _docker_backend(tmp_path)
    _proc(backend._PROC_ROOT, 4242, cgroup=cgroup, nspid=nspid)

    with pytest.raises(RuntimeError):
        backend._paused_container_pid_mapping(WORK_ID, (4242,))


def test_one_foreign_pid_refuses_the_whole_request(tmp_path: Path) -> None:
    backend = _docker_backend(tmp_path)
    _proc(backend._PROC_ROOT, 4242, cgroup=f"0::/system.slice/docker-{WORK_ID}.scope", nspid="4242\t57")
    _proc(backend._PROC_ROOT, 4300, cgroup=f"0::/system.slice/docker-{OTHER_ID}.scope", nspid="4300\t9")

    with pytest.raises(RuntimeError, match="4300"):
        backend._paused_container_pid_mapping(WORK_ID, (4242, 4300))


def test_missing_pid_and_low_pids_are_refused(tmp_path: Path) -> None:
    backend = _docker_backend(tmp_path)
    with pytest.raises(RuntimeError):
        backend._paused_container_pid_mapping(WORK_ID, (99999,))
    with pytest.raises(RuntimeError):
        backend._paused_container_pid_mapping(WORK_ID, (1,))


def test_exit_detection_treats_zombies_and_gone_pids_as_exited(tmp_path: Path) -> None:
    backend = _docker_backend(tmp_path)
    _proc(backend._PROC_ROOT, 10, cgroup="0::/", nspid="10\t2", state="S")
    _proc(backend._PROC_ROOT, 11, cgroup="0::/", nspid="11\t3", state="Z")

    assert backend._host_pid_has_exited(10) is False
    assert backend._host_pid_has_exited(11) is True
    assert backend._host_pid_has_exited(12) is True


def test_reclaim_signals_only_through_the_work_pid_namespace(tmp_path: Path) -> None:
    backend = _docker_backend(tmp_path)
    _proc(backend._PROC_ROOT, 4242, cgroup=f"0::/system.slice/docker-{WORK_ID}.scope", nspid="4242\t57")
    container = SimpleNamespace(attrs={"Image": "sha256:" + "e" * 64})
    handle = SimpleNamespace()
    runs = []

    class Containers:
        def run(self, image, **kwargs):
            runs.append((image, kwargs))
            # the helper's SIGKILL turns the frozen process into a zombie
            (backend._PROC_ROOT / "4242" / "stat").write_text("4242 (python3) Z 1\n")

    backend._client = SimpleNamespace(containers=Containers())
    backend._raw = lambda h: container
    backend._ensure_transient_handoff_allowed = lambda h: None
    backend._handoff_container_id = lambda c: WORK_ID
    backend._require_same_handle_pause_evidence = lambda h, cid: None
    paused_checks = []
    backend._verify_handoff_paused_state = lambda c, cid: paused_checks.append(cid)
    import sforge.harness.backend.docker_backend as module

    monkey_handle_type = module.DockerContainerHandle
    module.DockerContainerHandle = SimpleNamespace  # accept the fake handle
    try:
        evidence = backend.reclaim_accelerator_from_paused_handoff(handle, (4242,))
    finally:
        module.DockerContainerHandle = monkey_handle_type

    image, kwargs = runs[0]
    assert image == "sha256:" + "e" * 64
    assert kwargs["pid_mode"] == f"container:{WORK_ID}"
    assert kwargs["network_mode"] == "none"
    assert kwargs["cap_drop"] == ["ALL"] and kwargs["cap_add"] == ["KILL"]
    assert kwargs["command"] == ["57"]
    assert kwargs["environment"] == {"NVIDIA_VISIBLE_DEVICES": "void"}
    assert paused_checks == [WORK_ID, WORK_ID]
    assert evidence["killed_host_pids"] == [4242]
    assert evidence["killed_container_pids"] == [57]
