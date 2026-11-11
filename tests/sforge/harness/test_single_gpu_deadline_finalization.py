from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from sforge.harness import run_agent as runner
from sforge.harness.accelerator_observer import AcceleratorIdleAttestation
from sforge.harness.backend import AcceleratorAllocation
from sforge.harness.resource_handoff import (
    DurableResourceLease,
    HandoffBinding,
    canonical_allocation_key,
    handoff_annotations,
)
from sforge.harness.resource_lease import LeaseIdentity, LeaseState


GPU_UUID = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
WORK_ID = "a" * 64
CONTRACT = "sha256:" + "c" * 64


def test_portable_contract_digest_keeps_its_single_domain_prefix() -> None:
    task = type(
        "Task",
        (),
        {
            "portable": type(
                "Portable",
                (),
                {"evaluation_contract_sha256": CONTRACT},
            )()
        },
    )()

    digest = runner._portable_evaluation_contract_sha256(task)

    assert digest == CONTRACT
    assert digest.count("sha256:") == 1


def test_post_snapshot_check_does_not_reinspect_strictly_retired_work() -> None:
    class Backend:
        def verify_container_frozen_for_snapshot(self, handle) -> None:
            raise AssertionError("retired Work handle must not be reopened")

    disposition = runner._verify_post_snapshot_work_disposition(
        Backend(),
        object(),
        handoff_work_retired=True,
    )

    assert disposition == "strictly retired"


def test_post_snapshot_check_reverifies_nonretired_frozen_work() -> None:
    calls = []

    class Backend:
        def verify_container_frozen_for_snapshot(self, handle) -> None:
            calls.append(handle)

    handle = object()
    disposition = runner._verify_post_snapshot_work_disposition(
        Backend(),
        handle,
        handoff_work_retired=False,
    )

    assert disposition == "frozen"
    assert calls == [handle]


class _Fence:
    def __init__(self) -> None:
        self.calls = 0

    def seal(self) -> dict[str, object]:
        self.calls += 1
        return {
            "state": "FINALIZING",
            "recovery_policy": "never_resume",
        }


class _Logger:
    def info(self, *args, **kwargs) -> None:
        pass

    def warning(self, *args, **kwargs) -> None:
        pass


def _lease(path: Path) -> tuple[DurableResourceLease, HandoffBinding]:
    allocation = AcceleratorAllocation("nvidia", (GPU_UUID,))
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
    binding = HandoffBinding(identity, allocation, CONTRACT, annotations)
    lease = DurableResourceLease.create(path, binding)
    lease.transition(LeaseState.WORK_ACTIVE)
    lease.request_final_no_resume()
    lease.transition(LeaseState.FINALIZING)
    return lease, binding


def test_final_handoff_verifies_every_durable_binding_before_capture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lease, binding = _lease(tmp_path / "resource_lease.json")
    drained: list[tuple[str, str, int]] = []
    monkeypatch.setattr(
        runner,
        "_drain_pending_judge_evaluations",
        lambda url, token, timeout, logger: drained.append(
            (url, token, timeout)
        ),
    )
    fence = _Fence()

    runner._seal_drain_and_verify_final_handoff(
        fence,
        host_judge_url="http://judge",
        session_token="token",
        eval_timeout=60,
        lease_path=lease.path,
        expected_identity=binding.identity.to_dict(),
        expected_allocation=binding.allocation,
        expected_contract_sha256=CONTRACT,
        expected_annotations=dict(binding.annotations),
        logger=_Logger(),
    )

    assert fence.calls == 1
    assert drained == [("http://judge", "token", 60)]


def test_final_handoff_rejects_exact_work_identity_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lease, binding = _lease(tmp_path / "resource_lease.json")
    monkeypatch.setattr(
        runner,
        "_drain_pending_judge_evaluations",
        lambda *args, **kwargs: None,
    )
    wrong_identity = binding.identity.to_dict()
    wrong_identity["work_container_id"] = "b" * 64

    with pytest.raises(
        runner.FinalSnapshotError,
        match="exact durable FINALIZING no-resume binding",
    ):
        runner._seal_drain_and_verify_final_handoff(
            _Fence(),
            host_judge_url="http://judge",
            session_token="token",
            eval_timeout=60,
            lease_path=lease.path,
            expected_identity=wrong_identity,
            expected_allocation=binding.allocation,
            expected_contract_sha256=CONTRACT,
            expected_annotations=dict(binding.annotations),
            logger=_Logger(),
        )


@pytest.mark.parametrize("returned_state", ["CLOSED", "FINALIZING"])
def test_work_retirement_accepts_only_post_destroy_idle_state(
    monkeypatch: pytest.MonkeyPatch,
    returned_state: str,
) -> None:
    archive = b"immutable-final-archive"
    seen: dict[str, object] = {}

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"state": returned_state}

    def post(url, *, json, timeout):
        seen.update({"url": url, "json": json, "timeout": timeout})
        return _Response()

    monkeypatch.setattr(runner.requests, "post", post)

    if returned_state == "FINAL_GPU_IDLE_ATTESTED":
        runner._retire_final_handoff_work(
            host_judge_url="http://judge",
            session_token="token",
            archive=archive,
        )
    else:
        with pytest.raises(runner.FinalSnapshotError, match="did not attest idle"):
            runner._retire_final_handoff_work(
                host_judge_url="http://judge",
                session_token="token",
                archive=archive,
            )

    assert seen["json"] == {
        "token": "token",
        "admin_secret": runner.ADMIN_SECRET,
        "archive_sha256": "sha256:" + hashlib.sha256(archive).hexdigest(),
        "archive_size": len(archive),
    }


def test_work_retirement_accepts_final_gpu_idle_attestation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = b"immutable-final-archive"

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"state": "FINAL_GPU_IDLE_ATTESTED"}

    monkeypatch.setattr(runner.requests, "post", lambda *args, **kwargs: _Response())

    runner._retire_final_handoff_work(
        host_judge_url="http://judge",
        session_token="token",
        archive=archive,
    )
