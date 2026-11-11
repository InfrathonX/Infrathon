from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from sforge.harness import judge_server
from sforge.harness.accelerator_observer import (
    AcceleratorAttestationError,
    AcceleratorBusyError,
    AcceleratorIdleAttestation,
    NvmlAcceleratorObserver,
)
from sforge.harness.backend import AcceleratorAllocation
from sforge.harness.config import SForgeConfig, validate_single_gpu_handoff_config
from sforge.harness.judge_server import JudgeState, SubmissionStatus
from sforge.harness.resource_handoff import (
    DurableResourceLease,
    ExclusiveProcessLock,
    HandoffBinding,
    ResourceHandoffError,
    canonical_allocation_key,
    handoff_annotations,
    read_durable_lease,
)
from sforge.harness.resource_lease import LeaseIdentity, LeaseState


GPU_UUID = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
WORK_ID = "a" * 64
CONTRACT = "sha256:" + "c" * 64


def _allocation() -> AcceleratorAllocation:
    return AcceleratorAllocation("nvidia", (GPU_UUID,))


def _config(tmp_path: Path) -> SForgeConfig:
    allocation = _allocation()
    return SForgeConfig(
        log_dir=tmp_path,
        backend="docker",
        single_gpu_handoff=True,
        accelerator_device_uuid=GPU_UUID,
        accelerator_allocation_key=canonical_allocation_key(allocation),
        accelerator_observer="nvml",
        trusted_lease_id="lease-test",
    )


def _attestation(phase: str) -> AcceleratorIdleAttestation:
    return AcceleratorIdleAttestation(
        observer="test-nvml",
        allocation_vendor="nvidia",
        device_ids=(GPU_UUID,),
        phase=phase,
        sample_count=2,
        observed_at=1.0,
        observed_monotonic=2.0,
    )


class _NvmlFunction:
    def __init__(self, result: int = 0) -> None:
        self.result = result
        self.calls = 0

    def __call__(self, *args):
        self.calls += 1
        return self.result


class _NvmlLibrary:
    def __init__(self) -> None:
        self.nvmlInit_v2 = _NvmlFunction()
        self.nvmlShutdown = _NvmlFunction()


class _SamplingObserver(NvmlAcceleratorObserver):
    def __init__(self, samples, *, idle_grace_seconds: float = 0):
        self.library = _NvmlLibrary()
        self.samples = list(samples)
        self.seen: list[str] = []
        super().__init__(
            sample_count=2,
            sample_interval_seconds=0,
            idle_grace_seconds=idle_grace_seconds,
            library=self.library,
        )

    def _sample_device(self, library, device_id):
        self.seen.append(device_id)
        return self.samples.pop(0)


def test_nvml_observer_requires_two_empty_samples() -> None:
    observer = _SamplingObserver([((), ()), ((), ())])

    evidence = observer.attest_idle(_allocation(), phase="after-work-pause")

    assert observer.seen == [GPU_UUID, GPU_UUID]
    assert evidence.sample_count == 2
    assert evidence.to_dict()["accelerator_idle"] is True


def test_nvml_observer_rejects_a_context_and_does_not_attest() -> None:
    observer = _SamplingObserver([((4242,), ())])

    with pytest.raises(
        AcceleratorAttestationError,
        match="Docker pause does not release CUDA contexts",
    ):
        observer.attest_idle(_allocation(), phase="after-work-pause")

    assert observer.library.nvmlShutdown.calls == 1


def test_nvml_observer_retries_transient_busy_until_two_empty_samples() -> None:
    observer = _SamplingObserver(
        [((4242,), ()), ((), ()), ((), ())],
        idle_grace_seconds=1,
    )

    evidence = observer.attest_idle(_allocation(), phase="after-work-pause")

    assert observer.seen == [GPU_UUID, GPU_UUID, GPU_UUID]
    assert evidence.sample_count == 2


def test_config_requires_canonical_device_set_digest(tmp_path: Path) -> None:
    config = _config(tmp_path)
    validate_single_gpu_handoff_config(config)

    config.accelerator_allocation_key = "allocation-sha256:" + "0" * 64
    with pytest.raises(ValueError, match="canonical physical device-set"):
        validate_single_gpu_handoff_config(config)


def test_judge_process_lock_is_scoped_to_canonical_allocation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        judge_server,
        "create_backend_from_config",
        lambda config: _Backend([]),
    )
    monkeypatch.setattr(JudgeState, "_reaper_loop", lambda self: None)
    first_config = _config(tmp_path)
    second_uuid = "GPU-bbbbbbbb-cccc-dddd-eeee-ffffffffffff"
    second_allocation = AcceleratorAllocation("nvidia", (second_uuid,))
    second_config = _config(tmp_path)
    second_config.accelerator_device_uuid = second_uuid
    second_config.accelerator_allocation_key = canonical_allocation_key(
        second_allocation
    )
    second_config.trusted_lease_id = "lease-second-gpu"

    first = JudgeState(first_config, accelerator_observer=_Observer([]))
    second = None
    try:
        with pytest.raises(ResourceHandoffError, match="active owner"):
            JudgeState(first_config, accelerator_observer=_Observer([]))
        second = JudgeState(
            second_config,
            accelerator_observer=_Observer([]),
        )
        assert len(list((tmp_path / "resource_leases/allocations").glob("*.lock"))) == 2
    finally:
        assert first._handoff_server_lock is not None
        first._handoff_server_lock.close()
        if second is not None:
            assert second._handoff_server_lock is not None
            second._handoff_server_lock.close()


def test_durable_lease_persists_never_resume_and_attestation(tmp_path: Path) -> None:
    allocation = _allocation()
    allocation_key = canonical_allocation_key(allocation)
    identity = LeaseIdentity(
        trusted_lease_id="lease-test",
        run_id="run-test",
        task_id="task-test",
        allocation_key=allocation_key,
        work_container_id=WORK_ID,
    )
    annotations = handoff_annotations(
        run_id=identity.run_id,
        trusted_lease_id=identity.trusted_lease_id,
        allocation_key=allocation_key,
        evaluation_contract_sha256=CONTRACT,
    )
    path = tmp_path / "resource_lease.json"
    lease = DurableResourceLease.create(
        path,
        HandoffBinding(identity, allocation, CONTRACT, annotations),
    )
    with pytest.raises(ResourceHandoffError, match="invalid final archive digest"):
        lease.record_archive(sha256="sha256:" + "z" * 64, size=1)
    lease.transition(LeaseState.WORK_ACTIVE)
    lease.transition(LeaseState.PAUSE_REQUESTED)
    lease.transition(LeaseState.WORK_PAUSING)
    lease.transition(LeaseState.WORK_PAUSED)
    with pytest.raises(Exception, match="requires real accelerator-idle"):
        lease.transition(LeaseState.WORK_GPU_IDLE_ATTESTED)
    lease.transition(
        LeaseState.WORK_GPU_IDLE_ATTESTED,
        attestation=_attestation("after-work-pause"),
    )
    lease.request_final_no_resume()
    lease.transition(LeaseState.FINALIZING)

    durable = read_durable_lease(path)
    assert durable["state"] == "FINALIZING"
    assert durable["recovery_policy"] == "never_resume"
    assert durable["attestations"][0]["sample_count"] == 2


class _Observer:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def attest_idle(self, allocation, *, phase):
        assert allocation == _allocation()
        self.events.append(f"nvml:{phase}")
        return _attestation(phase)


class _Backend:
    backend_name = "docker"

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.expected_annotations = None
        self.judge_annotations: list[dict[str, str]] = []

    def get_container_handle(self, container_id, *, expected_annotations):
        if container_id == WORK_ID:
            self.expected_annotations = expected_annotations
            self.events.append("exact-work-lookup")
            return SimpleNamespace(id=WORK_ID, name="work")
        assert len(container_id) == 64
        assert expected_annotations["io.sforge.handoff.role"] == "judge"
        self.judge_annotations.append(dict(expected_annotations))
        self.events.append("exact-judge-lookup")
        return SimpleNamespace(id=container_id, name="judge")

    def pause_container_for_handoff(self, handle):
        self.events.append("pause-work")
        return {"container_id": handle.id}

    def verify_container_paused_for_handoff(self, handle):
        self.events.append("verify-work-paused")
        return {"container_id": handle.id}

    def unpause_container_after_handoff(self, handle):
        self.events.append("unpause-work")
        return {"container_id": handle.id}

    def verify_container_running_after_handoff(self, handle):
        self.events.append("verify-work-running")
        return {"container_id": handle.id}

    def destroy_container_for_handoff(self, handle):
        self.events.append(f"strict-destroy:{handle.id}")
        return {"container_id": handle.id, "container_destroyed": True}

    def verify_container_frozen_for_snapshot(self, handle):
        self.events.append("verify-work-frozen")
        return {"all_work_processes_stopped": True}

    def freeze_container_for_final_handoff_snapshot(
        self,
        handle,
        *,
        expected_annotations,
        stop_without_inspect=False,
    ):
        assert expected_annotations["io.sforge.handoff.lease-id"] == "lease-test"
        assert stop_without_inspect is True
        self.events.append("adopt-final-work-freeze")
        return {"all_work_processes_stopped": True}


class _Background:
    def __init__(self) -> None:
        self.call = None

    def add_task(self, function, *args):
        self.call = (function, args)

    def run(self) -> None:
        assert self.call is not None
        function, args = self.call
        function(*args)


class _Report:
    metrics: dict[str, object] = {
        "portable_contract": {
            "evaluation_contract_sha256": CONTRACT,
            "selected_ids": {
                "evaluation_suite_id": "suite",
                "execution_profile_id": "profile",
                "metric_id": "metric",
            },
        }
    }

    def to_dict(self):
        return {
            "valid": True,
            "score": 1.0,
            "pass_rate": 1.0,
            "passed": 1,
            "failed": 0,
            "total_tests": 1,
            "metrics": self.metrics,
        }


def _unattached_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> JudgeState:
    events: list[str] = []
    backend = _Backend(events)
    monkeypatch.setattr(
        judge_server, "create_backend_from_config", lambda config: backend
    )
    monkeypatch.setattr(JudgeState, "_reaper_loop", lambda self: None)
    state = JudgeState(
        _config(tmp_path), accelerator_observer=_Observer(events)
    )
    task = SimpleNamespace(
        portable=SimpleNamespace(
            evaluation_contract_sha256=CONTRACT,
            defaults=SimpleNamespace(
                to_dict=lambda: {
                    "evaluation_suite_id": "suite",
                    "execution_profile_id": "profile",
                    "metric_id": "metric",
                }
            ),
        ),
        judge=SimpleNamespace(
            public_eval_cmd=None,
            public_eval_timeout=None,
            private_artifact_path=None,
            eval_timeout=60,
            selection="pass_rate_first",
            score_direction="maximize",
        ),
    )
    state.tasks = {"task": task}
    return state


def _register_handoff(
    state: JudgeState,
    *,
    run_id: str = "run",
    trusted_lease_id: str = "lease-test",
) -> str:
    return state.register_session(
        "task",
        run_id,
        single_gpu_handoff=True,
        trusted_lease_id=trusted_lease_id,
        allocation_key=canonical_allocation_key(_allocation()),
        accelerator_device_ids=[GPU_UUID],
        evaluation_contract_sha256=CONTRACT,
    )


def _state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    state = _unattached_state(monkeypatch, tmp_path)
    backend = state.backend
    assert isinstance(backend, _Backend)
    events = backend.events
    token = _register_handoff(state)
    session = state.attach_work_container(token, WORK_ID)
    return state, backend, events, token, session


def _write_historical_lease(
    tmp_path: Path,
    *,
    run_id: str,
    task_id: str,
    trusted_lease_id: str,
    work_container_id: str,
    state: str = "CLOSED",
) -> None:
    path = tmp_path / "runs" / run_id / task_id / "resource_lease.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state": state,
                "identity": {
                    "trusted_lease_id": trusted_lease_id,
                    "run_id": run_id,
                    "task_id": task_id,
                    "allocation_key": canonical_allocation_key(_allocation()),
                    "work_container_id": work_container_id,
                },
                "allocation": {
                    "vendor": "nvidia",
                    "device_ids": [GPU_UUID],
                },
            }
        ),
        encoding="utf-8",
    )


def test_closed_history_cannot_rebind_trusted_lease_id_to_another_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_historical_lease(
        tmp_path,
        run_id="old-run",
        task_id="old-task",
        trusted_lease_id="lease-test",
        work_container_id="b" * 64,
    )
    state = _unattached_state(monkeypatch, tmp_path)
    try:
        token = _register_handoff(state)
        with pytest.raises(
            ResourceHandoffError,
            match="trusted lease ID was already bound",
        ):
            state.attach_work_container(token, WORK_ID)
    finally:
        assert state._handoff_server_lock is not None
        state._handoff_server_lock.close()


def test_run_task_history_cannot_change_durable_lease_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_historical_lease(
        tmp_path,
        run_id="run",
        task_id="task",
        trusted_lease_id="old-lease",
        work_container_id="b" * 64,
    )
    state = _unattached_state(monkeypatch, tmp_path)
    try:
        token = _register_handoff(state)
        with pytest.raises(
            ResourceHandoffError,
            match="run and task already have a different durable lease identity",
        ):
            state.attach_work_container(token, WORK_ID)
    finally:
        assert state._handoff_server_lock is not None
        state._handoff_server_lock.close()


def test_closed_history_releases_allocation_for_a_new_lease_and_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_historical_lease(
        tmp_path,
        run_id="old-run",
        task_id="old-task",
        trusted_lease_id="old-lease",
        work_container_id="b" * 64,
    )
    state = _unattached_state(monkeypatch, tmp_path)
    try:
        token = _register_handoff(state)
        session = state.attach_work_container(token, WORK_ID)
        assert session.lease.binding.identity.trusted_lease_id == "lease-test"
        assert session.lease.snapshot().state == LeaseState.WORK_ACTIVE
    finally:
        assert state._handoff_server_lock is not None
        state._handoff_server_lock.close()


def test_cross_allocation_identity_admission_is_atomic_and_releases_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = _unattached_state(monkeypatch, tmp_path)
    second_uuid = "GPU-bbbbbbbb-cccc-dddd-eeee-ffffffffffff"
    second_allocation = AcceleratorAllocation("nvidia", (second_uuid,))
    second_config = _config(tmp_path)
    second_config.accelerator_device_uuid = second_uuid
    second_config.accelerator_allocation_key = canonical_allocation_key(
        second_allocation
    )
    second = JudgeState(
        second_config,
        accelerator_observer=_Observer([]),
    )
    second.tasks = first.tasks

    first_token = _register_handoff(
        first,
        run_id="concurrent-first",
        trusted_lease_id="shared-scheduler-lease",
    )
    second_token = second.register_session(
        "task",
        "concurrent-second",
        single_gpu_handoff=True,
        trusted_lease_id="shared-scheduler-lease",
        allocation_key=canonical_allocation_key(second_allocation),
        accelerator_device_ids=[second_uuid],
        evaluation_contract_sha256=CONTRACT,
    )

    original_create = DurableResourceLease.create.__func__
    first_create_entered = threading.Event()
    allow_first_create = threading.Event()
    create_identities: list[LeaseIdentity] = []
    create_guard = threading.Lock()

    def blocking_create(cls, path, binding):
        with create_guard:
            create_identities.append(binding.identity)
            is_first = len(create_identities) == 1
        if is_first:
            first_create_entered.set()
            if not allow_first_create.wait(5):
                raise AssertionError("timed out releasing durable create")
        return original_create(cls, path, binding)

    monkeypatch.setattr(
        DurableResourceLease,
        "create",
        classmethod(blocking_create),
    )
    successes: list[HandoffBinding] = []
    failures: list[Exception] = []

    def attach(state: JudgeState, token: str) -> None:
        try:
            successes.append(
                state.attach_work_container(token, WORK_ID).lease.binding
            )
        except Exception as exc:
            failures.append(exc)

    first_thread = threading.Thread(target=attach, args=(first, first_token))
    second_thread = threading.Thread(
        target=attach,
        args=(second, second_token),
    )
    try:
        first_thread.start()
        assert first_create_entered.wait(5)
        second_thread.start()
        # The second GPU has its own lifetime lock, but cannot pass the shared
        # identity scan/create admission domain while the first is inside it.
        second_thread.join(0.05)
        assert second_thread.is_alive()
        with create_guard:
            assert len(create_identities) == 1
    finally:
        allow_first_create.set()
        first_thread.join(5)
        second_thread.join(5)
        assert first._handoff_server_lock is not None
        first._handoff_server_lock.close()
        assert second._handoff_server_lock is not None
        second._handoff_server_lock.close()

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], ResourceHandoffError)
    assert "trusted lease ID was already bound" in str(failures[0])
    with create_guard:
        assert len(create_identities) == 1

    # The losing scan raises while owning the short lock; its finally block
    # must still release that global admission domain.
    admission = ExclusiveProcessLock(
        tmp_path / "resource_leases" / "identity-admission.lock",
        "test identity admission",
    )
    admission.close()


def test_manual_submit_persists_queue_before_pause_and_serially_reuses_gpu(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state, backend, events, token, session = _state(monkeypatch, tmp_path)

    def fake_judge_submission(**kwargs):
        assert kwargs["accelerator_allocation"] == _allocation()
        lifecycle = kwargs["container_lifecycle"]
        judge = SimpleNamespace(id="b" * 64, name="judge")
        events.append("create-judge")
        lifecycle.judge_started(judge)
        lifecycle.judge_cleaning(judge)
        backend.destroy_container_for_handoff(judge)
        lifecycle.judge_destroyed(judge)
        return _Report()

    monkeypatch.setattr(judge_server, "judge_submission", fake_judge_submission)
    _, _, round_id, cpu, mem, _ = state.consume_round(token, "agent")
    background = _Background()
    submission_id = state.submit(
        "task",
        b"durable-archive",
        "run",
        round_id,
        judge_cpu_limit=cpu,
        judge_mem_limit=mem,
        session_token=token,
        background_tasks=background,
    )

    queue_dir = tmp_path / "runs/run/task/submissions/agent-1"
    assert (queue_dir / "queued_submission.tar.gz").read_bytes() == b"durable-archive"
    queued = json.loads((queue_dir / "queued_submission.json").read_text())
    assert queued["status"] == "queued"
    assert events == ["exact-work-lookup"]

    background.run()

    assert state.submissions[submission_id]["status"] == SubmissionStatus.COMPLETED
    assert events == [
        "exact-work-lookup",
        "pause-work",
        "verify-work-paused",
        "nvml:after_work_pause_before_judge",
        "create-judge",
        "exact-judge-lookup",
        "strict-destroy:" + "b" * 64,
        "nvml:after_judge_destroy_before_work_resume",
        "unpause-work",
        "verify-work-running",
    ]
    assert session.lease.snapshot().state == LeaseState.WORK_ACTIVE
    assert backend.expected_annotations["io.sforge.handoff.lease-id"] == "lease-test"
    durable = read_durable_lease(session.lease.path)
    assert durable["judge_containers"][0]["container_id"] == "b" * 64
    assert (
        durable["judge_containers"][0]["annotations"][
            "io.sforge.handoff.attempt"
        ]
        == submission_id
    )


def test_registration_uses_portable_contract_digest_without_double_prefix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state, _, _, token, session = _state(monkeypatch, tmp_path)

    assert token
    assert session.lease.binding.evaluation_contract_sha256 == CONTRACT
    with pytest.raises(ValueError, match="does not match Judge task"):
        state.register_session(
            "task",
            "other-run",
            single_gpu_handoff=True,
            trusted_lease_id="other-lease",
            allocation_key=canonical_allocation_key(_allocation()),
            accelerator_device_ids=[GPU_UUID],
            evaluation_contract_sha256="sha256:" + CONTRACT,
        )


def test_idle_seal_pauses_work_before_returning_finalizing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state, _, events, token, session = _state(monkeypatch, tmp_path)

    returned = state.seal_handoff(token)

    assert returned is session
    assert session.lease.snapshot().state == LeaseState.FINALIZING
    assert events == [
        "exact-work-lookup",
        "pause-work",
        "verify-work-paused",
    ]
    assert "unpause-work" not in events
    durable = read_durable_lease(session.lease.path)
    assert durable["recovery_policy"] == "never_resume"
    assert [entry["to_state"] for entry in durable["audit"]][-4:] == [
        "PAUSE_REQUESTED",
        "WORK_PAUSING",
        "WORK_PAUSED",
        "FINALIZING",
    ]


@pytest.mark.parametrize("failure_phase", ["pause", "verify"])
def test_idle_seal_pause_failure_is_durably_fenced(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_phase: str,
) -> None:
    state, backend, events, token, session = _state(monkeypatch, tmp_path)

    if failure_phase == "pause":
        def fail_pause(handle):
            events.append("pause-work-failed")
            raise RuntimeError("pause failed")

        monkeypatch.setattr(backend, "pause_container_for_handoff", fail_pause)
    else:
        def fail_verify(handle):
            events.append("verify-work-paused-failed")
            raise RuntimeError("verify failed")

        monkeypatch.setattr(
            backend,
            "verify_container_paused_for_handoff",
            fail_verify,
        )

    with pytest.raises(ResourceHandoffError, match="failed to freeze Work"):
        state.seal_handoff(token)

    assert session.lease.snapshot().state == LeaseState.FENCED
    assert session.lease.recovery_policy == "never_resume"
    assert "unpause-work" not in events
    durable = read_durable_lease(session.lease.path)
    assert durable["state"] == "FENCED"
    assert durable["recovery_policy"] == "never_resume"


def test_final_seal_prevents_resume_and_strictly_retires_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state, _, events, token, session = _state(monkeypatch, tmp_path)
    state.tasks["task"].judge.private_artifact_path = (
        "/logs/verifier/final.json"
    )
    state.config.judge_extra_env["SFORGE_PRIVATE_CONTEXT_ROLE"] = "judge"

    state.seal_handoff(token)
    durable = read_durable_lease(session.lease.path)
    assert durable["recovery_policy"] == "never_resume"
    assert durable["state"] == "FINALIZING"
    assert events == [
        "exact-work-lookup",
        "pause-work",
        "verify-work-paused",
    ]

    payload = b"final"
    archive_path = tmp_path / "runs/run/task/final_archive.tar.gz"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_bytes(payload)
    archive_digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    state.finish_final_work(
        token,
        archive_sha256=archive_digest,
        archive_size=len(payload),
    )

    assert session.lease.snapshot().state == LeaseState.FINAL_GPU_IDLE_ATTESTED
    assert f"strict-destroy:{WORK_ID}" in events
    assert "nvml:after_final_work_destroy" in events
    assert "unpause-work" not in events
    retirement_epoch = session.lease.snapshot().epoch
    retirement_events = list(events)

    # Simulate a lost success response: exact request replay is side-effect
    # free, while either archive identity field changing is a hard conflict.
    assert state.finish_final_work(
        token,
        archive_sha256=archive_digest,
        archive_size=len(payload),
    ) is session
    assert session.lease.snapshot().epoch == retirement_epoch
    assert events == retirement_events
    with pytest.raises(ResourceHandoffError, match="does not match"):
        state.finish_final_work(
            token,
            archive_sha256="sha256:" + "d" * 64,
            archive_size=len(payload),
        )
    with pytest.raises(ResourceHandoffError, match="does not match"):
        state.finish_final_work(
            token,
            archive_sha256=archive_digest,
            archive_size=len(payload) + 1,
        )

    def fake_final_judge(**kwargs):
        assert kwargs["archive"] == payload
        assert kwargs["accelerator_allocation"] == _allocation()
        # The final Judge learns the coordinator-authenticated archive digest
        # through the private context channel; the server's own environment
        # is never mutated.
        final_env = kwargs["config"].judge_extra_env
        assert final_env["SFORGE_PRIVATE_CONTEXT_ARCHIVE_SHA256"] == (
            archive_digest.removeprefix("sha256:")
        )
        assert final_env["SFORGE_PRIVATE_CONTEXT_ROLE"] == "judge"
        assert "SFORGE_PRIVATE_CONTEXT_ARCHIVE_SHA256" not in (
            state.config.judge_extra_env
        )
        lifecycle = kwargs["container_lifecycle"]
        assert lifecycle.judge_annotations()["io.sforge.handoff.role"] == "judge"
        judge = SimpleNamespace(id="c" * 64, name="final-judge")
        lifecycle.judge_started(judge)
        lifecycle.judge_cleaning(judge)
        state.backend.destroy_container_for_handoff(judge)
        lifecycle.judge_destroyed(judge)
        artifact_dir = (
            tmp_path
            / "runs/final-run/task/submissions/manual-1"
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "submission.tar.gz").write_bytes(payload)
        (artifact_dir / "report.json").write_text(
            json.dumps(_Report().to_dict()),
            encoding="utf-8",
        )
        (artifact_dir / judge_server.PRIVATE_JUDGE_ARTIFACT_NAME).write_bytes(
            b"durable-private-evidence"
        )
        return _Report()

    monkeypatch.setattr(judge_server, "judge_submission", fake_final_judge)
    request = {
        "source_run_id": "run",
        "final_run_id": "final-run",
        "task_id": "task",
        "trusted_lease_id": "lease-test",
        "allocation_key": canonical_allocation_key(_allocation()),
        "evaluation_contract_sha256": CONTRACT,
        "archive_sha256": archive_digest,
        "archive_size": len(payload),
        "timeout": 60,
    }
    _, report = state.run_final_judge(**request)

    assert report["valid"] is True
    assert session.lease.snapshot().state == LeaseState.CLOSED
    assert "nvml:before_final_judge" in events
    assert "strict-destroy:" + "c" * 64 in events
    assert "nvml:after_final_judge_destroy" in events
    durable = read_durable_lease(session.lease.path)
    assert durable["final_judge"]["request"]["source_run_id"] == "run"
    assert durable["final_judge"]["report"]["valid"] is True
    assert durable["final_judge"]["report"]["metrics"][
        "portable_contract"
    ] == {
        "evaluation_contract_sha256": CONTRACT,
        "selected_ids": {
            "evaluation_suite_id": "suite",
            "execution_profile_id": "profile",
            "metric_id": "metric",
        },
    }
    assert durable["final_judge"]["artifacts"]["report"]["sha256"].startswith(
        "sha256:"
    )
    assert durable["final_judge"]["artifacts"]["private_judge"] == {
        "sha256": "sha256:"
        + hashlib.sha256(b"durable-private-evidence").hexdigest(),
        "size": len(b"durable-private-evidence"),
    }
    assert durable["judge_containers"][-1]["container_id"] == "c" * 64

    events_before_retry = list(events)
    _, retried_report = state.run_final_judge(**request)
    assert retried_report == report
    assert events == events_before_retry

    with pytest.raises(ResourceHandoffError, match="requires FINALIZING state"):
        state.finish_final_work(
            token,
            archive_sha256=archive_digest,
            archive_size=len(payload),
        )

    with pytest.raises(
        Exception,
        match="timeout must equal the task evaluation contract",
    ):
        state.run_final_judge(**{**request, "timeout": 59})


def test_missing_required_final_private_artifact_prevents_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state, backend, _, token, session = _state(monkeypatch, tmp_path)
    state.tasks["task"].judge.private_artifact_path = (
        "/logs/verifier/final.json"
    )
    state.seal_handoff(token)
    payload = b"final"
    archive_path = tmp_path / "runs/run/task/final_archive.tar.gz"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_bytes(payload)
    archive_digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    state.finish_final_work(
        token,
        archive_sha256=archive_digest,
        archive_size=len(payload),
    )

    def fake_final_judge(**kwargs):
        lifecycle = kwargs["container_lifecycle"]
        judge = SimpleNamespace(id="c" * 64, name="final-judge")
        lifecycle.judge_started(judge)
        lifecycle.judge_cleaning(judge)
        backend.destroy_container_for_handoff(judge)
        lifecycle.judge_destroyed(judge)
        artifact_dir = tmp_path / "runs/final-run/task/submissions/manual-1"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "submission.tar.gz").write_bytes(payload)
        (artifact_dir / "report.json").write_text(
            json.dumps(_Report().to_dict()),
            encoding="utf-8",
        )
        return _Report()

    monkeypatch.setattr(judge_server, "judge_submission", fake_final_judge)
    with pytest.raises(ResourceHandoffError, match="artifact is unavailable"):
        state.run_final_judge(
            source_run_id="run",
            final_run_id="final-run",
            task_id="task",
            trusted_lease_id="lease-test",
            allocation_key=canonical_allocation_key(_allocation()),
            evaluation_contract_sha256=CONTRACT,
            archive_sha256=archive_digest,
            archive_size=len(payload),
            timeout=60,
        )

    assert session.lease.snapshot().state == LeaseState.FENCED
    durable = read_durable_lease(session.lease.path)
    assert durable["state"] == "FENCED"
    assert durable["final_judge"]["report"] is None


def test_deadline_seal_during_judge_running_never_unpauses_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state, backend, events, token, session = _state(monkeypatch, tmp_path)

    def fake_judge_submission(**kwargs):
        lifecycle = kwargs["container_lifecycle"]
        judge = SimpleNamespace(id="b" * 64, name="judge")
        events.append("create-judge")
        lifecycle.judge_started(judge)
        state.seal_handoff(token)
        assert session.lease.snapshot().state == LeaseState.JUDGE_RUNNING
        assert session.lease.recovery_policy == "never_resume"
        lifecycle.judge_cleaning(judge)
        backend.destroy_container_for_handoff(judge)
        lifecycle.judge_destroyed(judge)
        return _Report()

    monkeypatch.setattr(judge_server, "judge_submission", fake_judge_submission)
    _, _, round_id, cpu, mem, _ = state.consume_round(token, "agent")
    background = _Background()
    state.submit(
        "task",
        b"deadline-archive",
        "run",
        round_id,
        judge_cpu_limit=cpu,
        judge_mem_limit=mem,
        session_token=token,
        background_tasks=background,
    )
    background.run()

    assert session.lease.snapshot().state == LeaseState.FINALIZING
    assert "unpause-work" not in events
    durable = read_durable_lease(session.lease.path)
    assert durable["recovery_policy"] == "never_resume"


def test_deadline_seal_synchronously_pauses_queued_work_before_response(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state, backend, events, token, session = _state(monkeypatch, tmp_path)

    def fake_judge_submission(**kwargs):
        lifecycle = kwargs["container_lifecycle"]
        judge = SimpleNamespace(id="b" * 64, name="judge")
        events.append("create-judge")
        lifecycle.judge_started(judge)
        lifecycle.judge_cleaning(judge)
        backend.destroy_container_for_handoff(judge)
        lifecycle.judge_destroyed(judge)
        return _Report()

    monkeypatch.setattr(judge_server, "judge_submission", fake_judge_submission)
    _, _, round_id, cpu, mem, _ = state.consume_round(token, "agent")
    background = _Background()
    state.submit(
        "task",
        b"queued-at-deadline",
        "run",
        round_id,
        judge_cpu_limit=cpu,
        judge_mem_limit=mem,
        session_token=token,
        background_tasks=background,
    )

    # The accepted background worker has not started.  seal_handoff must stop
    # Work writers synchronously rather than waiting behind the task lock.
    state.seal_handoff(token)

    assert session.lease.snapshot().state == LeaseState.WORK_PAUSED
    assert events == [
        "exact-work-lookup",
        "pause-work",
        "verify-work-paused",
    ]
    assert read_durable_lease(session.lease.path)["recovery_policy"] == "never_resume"

    background.run()

    assert session.lease.snapshot().state == LeaseState.FINALIZING
    assert events.count("pause-work") == 1
    assert "unpause-work" not in events


def test_submission_commit_and_seal_are_linearized_before_queue_visibility(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state, _, events, token, session = _state(monkeypatch, tmp_path)
    background = _Background()
    seal_finished = threading.Event()
    seal_errors: list[Exception] = []

    with state.submission_commit_admission(token, "agent"):
        _, _, round_id, cpu, mem, _ = state.consume_round(token, "agent")

        def seal() -> None:
            try:
                state.seal_handoff(token)
            except Exception as exc:
                seal_errors.append(exc)
            finally:
                seal_finished.set()

        sealer = threading.Thread(target=seal)
        sealer.start()
        assert not seal_finished.wait(0.05)
        state.submit(
            "task",
            b"commit-race",
            "run",
            round_id,
            judge_cpu_limit=cpu,
            judge_mem_limit=mem,
            session_token=token,
            background_tasks=background,
        )
        queue_path = (
            tmp_path
            / "runs/run/task/submissions/agent-1/queued_submission.tar.gz"
        )
        assert queue_path.read_bytes() == b"commit-race"
        assert events == ["exact-work-lookup"]

    sealer.join(5)
    assert not sealer.is_alive()
    assert seal_errors == []
    assert session.lease.snapshot().state == LeaseState.WORK_PAUSED
    assert events == [
        "exact-work-lookup",
        "pause-work",
        "verify-work-paused",
    ]


def test_missing_observer_fences_before_judge_creation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state, _, events, token, session = _state(monkeypatch, tmp_path)
    state.accelerator_observer = None
    called = False

    def unsafe_judge(**kwargs):
        nonlocal called
        called = True
        return _Report()

    monkeypatch.setattr(judge_server, "judge_submission", unsafe_judge)
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

    assert called is False
    assert events == ["exact-work-lookup"]
    assert state.submissions[submission_id]["status"] == SubmissionStatus.ERROR
    assert session.lease.snapshot().state == LeaseState.FENCED


def test_attributed_pre_judge_busy_recovers_work_without_fencing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state, _, events, token, session = _state(monkeypatch, tmp_path)

    class BusyObserver:
        def attest_idle(self, allocation, *, phase):
            events.append(f"busy:{phase}")
            raise AcceleratorBusyError(
                "contexts belong only to paused Work",
                recoverable=True,
            )

    state.accelerator_observer = BusyObserver()
    judge_created = False

    def unsafe_judge(**kwargs):
        nonlocal judge_created
        judge_created = True
        return _Report()

    monkeypatch.setattr(judge_server, "judge_submission", unsafe_judge)
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

    assert judge_created is False
    assert state.submissions[submission_id]["status"] == SubmissionStatus.ERROR
    assert session.lease.snapshot().state == LeaseState.WORK_ACTIVE
    assert events[-3:] == [
        "busy:after_work_pause_before_judge",
        "unpause-work",
        "verify-work-running",
    ]


def test_attributed_post_judge_busy_recovers_only_after_strict_destroy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state, backend, events, token, session = _state(monkeypatch, tmp_path)

    class BusyAfterJudgeObserver:
        def attest_idle(self, allocation, *, phase):
            events.append(f"observe:{phase}")
            if phase == "after_judge_destroy_before_work_resume":
                raise AcceleratorBusyError(
                    "remaining contexts belong only to paused Work",
                    recoverable=True,
                )
            return _attestation(phase)

    state.accelerator_observer = BusyAfterJudgeObserver()

    def fake_judge_submission(**kwargs):
        lifecycle = kwargs["container_lifecycle"]
        judge = SimpleNamespace(id="b" * 64, name="judge")
        lifecycle.judge_started(judge)
        lifecycle.judge_cleaning(judge)
        backend.destroy_container_for_handoff(judge)
        lifecycle.judge_destroyed(judge)
        return _Report()

    monkeypatch.setattr(judge_server, "judge_submission", fake_judge_submission)
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

    strict_destroy = events.index("strict-destroy:" + "b" * 64)
    post_observe = events.index(
        "observe:after_judge_destroy_before_work_resume"
    )
    unpause = events.index("unpause-work")
    assert strict_destroy < post_observe < unpause
    assert state.submissions[submission_id]["status"] == SubmissionStatus.ERROR
    assert session.lease.snapshot().state == LeaseState.WORK_ACTIVE


def test_unknown_busy_fences_and_never_starts_judge(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state, _, _, token, session = _state(monkeypatch, tmp_path)

    class UnknownBusyObserver:
        def attest_idle(self, allocation, *, phase):
            raise AcceleratorBusyError("unknown process ownership")

    state.accelerator_observer = UnknownBusyObserver()
    called = False

    def unsafe_judge(**kwargs):
        nonlocal called
        called = True
        return _Report()

    monkeypatch.setattr(judge_server, "judge_submission", unsafe_judge)
    _, _, round_id, cpu, mem, _ = state.consume_round(token, "agent")
    background = _Background()
    state.submit(
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

    assert called is False
    assert session.lease.snapshot().state == LeaseState.FENCED


def test_handoff_token_losing_exact_session_fails_without_legacy_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state, _, events, token, _ = _state(monkeypatch, tmp_path)
    del state._handoff_sessions[token]
    called = False

    def unsafe_legacy_judge(**kwargs):
        nonlocal called
        called = True
        return _Report()

    monkeypatch.setattr(judge_server, "judge_submission", unsafe_legacy_judge)
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

    assert called is False
    assert events == ["exact-work-lookup"]
    assert state.submissions[submission_id]["status"] == SubmissionStatus.ERROR
    assert "lost its exact token" in state.submissions[submission_id]["error"]


def test_seal_cannot_cross_policy_check_and_unpause(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state, original_backend, events, token, session = _state(
        monkeypatch,
        tmp_path,
    )
    unpause_entered = threading.Event()
    permit_unpause = threading.Event()
    seal_finished = threading.Event()

    class BlockingBackend(_Backend):
        def unpause_container_after_handoff(self, handle):
            self.events.append("unpause-work")
            unpause_entered.set()
            assert permit_unpause.wait(5)
            return {"container_id": handle.id}

    backend = BlockingBackend(events)
    backend.expected_annotations = original_backend.expected_annotations
    state.backend = backend

    original_seal = session.lease.request_final_no_resume

    def tracked_seal():
        result = original_seal()
        events.append("durable-seal")
        return result

    monkeypatch.setattr(session.lease, "request_final_no_resume", tracked_seal)

    def fake_judge_submission(**kwargs):
        lifecycle = kwargs["container_lifecycle"]
        judge = SimpleNamespace(id="b" * 64, name="judge")
        lifecycle.judge_started(judge)
        lifecycle.judge_cleaning(judge)
        backend.destroy_container_for_handoff(judge)
        lifecycle.judge_destroyed(judge)
        return _Report()

    monkeypatch.setattr(judge_server, "judge_submission", fake_judge_submission)
    _, _, round_id, cpu, mem, _ = state.consume_round(token, "agent")
    background = _Background()
    state.submit(
        "task",
        b"archive",
        "run",
        round_id,
        judge_cpu_limit=cpu,
        judge_mem_limit=mem,
        session_token=token,
        background_tasks=background,
    )

    worker = threading.Thread(target=background.run)
    worker.start()
    assert unpause_entered.wait(5)

    def seal() -> None:
        state.seal_handoff(token)
        seal_finished.set()

    sealer = threading.Thread(target=seal)
    sealer.start()
    assert not seal_finished.wait(0.05)
    permit_unpause.set()
    worker.join(5)
    sealer.join(5)

    assert not worker.is_alive()
    assert not sealer.is_alive()
    assert events.index("verify-work-running") < events.index("durable-seal")
    assert session.lease.snapshot().state == LeaseState.FINALIZING
    assert session.lease.recovery_policy == "never_resume"


def test_final_judge_private_context_carries_only_the_authenticated_archive() -> None:
    digest = "sha256:" + "e" * 64
    base = {"SFORGE_PRIVATE_CONTEXT_ROLE": "judge", "NVIDIA_VISIBLE_DEVICES": GPU_UUID}
    context = judge_server._final_judge_private_context(base, archive_sha256=digest)
    assert context == {
        **base,
        "SFORGE_PRIVATE_CONTEXT_ARCHIVE_SHA256": "e" * 64,
    }
    assert "SFORGE_PRIVATE_CONTEXT_ARCHIVE_SHA256" not in base
    # An identical pre-declared value is accepted; a different one is refused.
    agreeing = {**base, "SFORGE_PRIVATE_CONTEXT_ARCHIVE_SHA256": "e" * 64}
    assert judge_server._final_judge_private_context(
        agreeing, archive_sha256=digest
    ) == agreeing
    with pytest.raises(ResourceHandoffError, match="conflicts with the authenticated"):
        judge_server._final_judge_private_context(
            {**base, "SFORGE_PRIVATE_CONTEXT_ARCHIVE_SHA256": "f" * 64},
            archive_sha256=digest,
        )
    for bad in ("e" * 64, "sha256:" + "E" * 64, "sha256:" + "e" * 63, ""):
        with pytest.raises(ResourceHandoffError, match="archive digest is invalid"):
            judge_server._final_judge_private_context(base, archive_sha256=bad)


def test_final_judge_refuses_a_server_archive_context_that_disagrees(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state, _, events, token, session = _state(monkeypatch, tmp_path)
    state.config.judge_extra_env["SFORGE_PRIVATE_CONTEXT_ARCHIVE_SHA256"] = "f" * 64
    state.seal_handoff(token)
    payload = b"final"
    archive_path = tmp_path / "runs/run/task/final_archive.tar.gz"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_bytes(payload)
    archive_digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    state.finish_final_work(
        token,
        archive_sha256=archive_digest,
        archive_size=len(payload),
    )
    monkeypatch.setattr(
        judge_server,
        "judge_submission",
        lambda **kwargs: pytest.fail("a conflicting archive context must never reach Judge"),
    )
    with pytest.raises(ResourceHandoffError, match="conflicts with the authenticated"):
        state.run_final_judge(
            source_run_id="run",
            final_run_id="final-run",
            task_id="task",
            trusted_lease_id="lease-test",
            allocation_key=canonical_allocation_key(_allocation()),
            evaluation_contract_sha256=CONTRACT,
            archive_sha256=archive_digest,
            archive_size=len(payload),
            timeout=60,
        )
    assert session.lease.snapshot().state != LeaseState.CLOSED
    assert "nvml:before_final_judge" not in events


def _write_lease_record(path: Path, state: str, *, audit: list | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps({"schema_version": 1, "state": state, "audit": audit or []})
    )
    tmp.replace(path)


def test_fence_watchdog_ignores_live_states_and_missing_records(tmp_path: Path) -> None:
    from sforge.harness.run_agent import _HandoffFenceWatchdog

    lease_path = tmp_path / "runs/run/task/resource_lease.json"
    watchdog = _HandoffFenceWatchdog(lease_path, poll_interval=0.05)
    assert watchdog.poll_once() is False
    for state in ("WORK_ACTIVE", "WORK_PAUSED", "JUDGE_RUNNING", "CLOSED"):
        _write_lease_record(lease_path, state)
        assert watchdog.poll_once() is False, state
    watchdog.check()
    assert not watchdog.exec_stop_event.is_set()
    assert watchdog.fenced is False


def test_fence_watchdog_aborts_exec_with_the_audited_reason(tmp_path: Path) -> None:
    from sforge.harness.run_agent import HandoffFencedError, _HandoffFenceWatchdog

    lease_path = tmp_path / "runs/run/task/resource_lease.json"
    _write_lease_record(lease_path, "WORK_ACTIVE")
    watchdog = _HandoffFenceWatchdog(lease_path, poll_interval=0.05)
    watchdog.start()
    try:
        assert not watchdog.exec_stop_event.wait(0.3)
        _write_lease_record(
            lease_path,
            "FENCED",
            audit=[
                {"action": "transition", "to_state": "WORK_PAUSED", "reason": None},
                {
                    "action": "fail_closed",
                    "to_state": "FENCED",
                    "reason": "single GPU handoff failed: AcceleratorBusyError",
                },
            ],
        )
        assert watchdog.exec_stop_event.wait(2.0)
        assert watchdog.fenced_state == "FENCED"
        assert watchdog.fence_reason == "single GPU handoff failed: AcceleratorBusyError"
        with pytest.raises(HandoffFencedError, match="AcceleratorBusyError"):
            watchdog.check()
    finally:
        watchdog.stop()


def test_fence_watchdog_mirrors_operator_shutdown_without_fencing(tmp_path: Path) -> None:
    from sforge.harness.run_agent import _HandoffFenceWatchdog

    lease_path = tmp_path / "runs/run/task/resource_lease.json"
    _write_lease_record(lease_path, "WORK_ACTIVE")
    shutdown = threading.Event()
    watchdog = _HandoffFenceWatchdog(
        lease_path, shutdown_event=shutdown, poll_interval=0.05
    )
    watchdog.start()
    try:
        shutdown.set()
        assert watchdog.exec_stop_event.wait(2.0)
        assert watchdog.fenced is False
        watchdog.check()
    finally:
        watchdog.stop()


def test_fenced_error_is_a_final_snapshot_error() -> None:
    from sforge.harness.run_agent import FinalSnapshotError, HandoffFencedError

    assert issubclass(HandoffFencedError, FinalSnapshotError)


def test_final_judge_round_id_is_forwarded_and_validated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state, backend, _, token, session = _state(monkeypatch, tmp_path)
    state.seal_handoff(token)
    payload = b"final"
    archive_path = tmp_path / "runs/run/task/final_archive.tar.gz"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_bytes(payload)
    archive_digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    state.finish_final_work(
        token,
        archive_sha256=archive_digest,
        archive_size=len(payload),
    )
    common = dict(
        source_run_id="run",
        final_run_id="final-run",
        task_id="task",
        trusted_lease_id="lease-test",
        allocation_key=canonical_allocation_key(_allocation()),
        evaluation_contract_sha256=CONTRACT,
        archive_sha256=archive_digest,
        archive_size=len(payload),
        timeout=60,
    )
    # A malformed round ID is refused before any Judge or lease work.
    monkeypatch.setattr(
        judge_server,
        "judge_submission",
        lambda **kwargs: pytest.fail("no Judge may run for an invalid round ID"),
    )
    for bad in ("", "-final", "final run", "x" * 129):
        with pytest.raises(ResourceHandoffError, match="invalid final round ID"):
            state.run_final_judge(round_id=bad, **common)
    assert session.lease.snapshot().state != LeaseState.FENCED

    seen: list[str] = []

    def capture_round(**kwargs):
        seen.append(kwargs["round_id"])
        raise ResourceHandoffError("captured")

    monkeypatch.setattr(judge_server, "judge_submission", capture_round)
    with pytest.raises(ResourceHandoffError, match="captured"):
        state.run_final_judge(round_id="final-run-2026", **common)
    assert seen == ["final-run-2026"]
