from __future__ import annotations

import json

import pytest

from sforge.harness.resource_lease import (
    FutureLeaseEpoch,
    IllegalLeaseTransition,
    LeaseIdentity,
    LeaseIdentityCollision,
    LeaseIdentityError,
    LeaseState,
    ResourceLease,
    ResourceLeaseRegistry,
    StaleLeaseEpoch,
)


ALLOCATION_KEY = "allocation-sha256:" + "a" * 64


@pytest.fixture
def identity() -> LeaseIdentity:
    return LeaseIdentity(
        trusted_lease_id="lease-authority-17",
        run_id="run-1",
        task_id="task-a",
        allocation_key=ALLOCATION_KEY,
        work_container_id="sha256:work-container-exact-id",
    )


def _advance(lease: ResourceLease, states: list[LeaseState]) -> None:
    for state in states:
        lease.transition(
            lease.identity,
            state,
            expected_epoch=lease.snapshot().epoch,
        )


def _checkpoint_close(lease: ResourceLease) -> None:
    _advance(
        lease,
        [
            LeaseState.WORK_ACTIVE,
            LeaseState.FINALIZING,
            LeaseState.FINAL_WORK_FROZEN,
            LeaseState.FINAL_ARCHIVE_READY,
            LeaseState.FINAL_WORK_DESTROYING,
            LeaseState.FINAL_WORK_DESTROYED,
            LeaseState.FINAL_GPU_IDLE_ATTESTED,
            LeaseState.CHECKPOINT_ONLY_CLOSING,
            LeaseState.CLOSED,
        ],
    )


@pytest.mark.parametrize(
    "states",
    [
        pytest.param(
            [
                LeaseState.WORK_ACTIVE,
                LeaseState.PAUSE_REQUESTED,
                LeaseState.WORK_PAUSING,
                LeaseState.WORK_PAUSED,
                LeaseState.WORK_GPU_IDLE_ATTESTED,
                LeaseState.JUDGE_STARTING,
                LeaseState.JUDGE_RUNNING,
                LeaseState.JUDGE_CLEANING,
                LeaseState.JUDGE_CLEANED,
                LeaseState.JUDGE_GPU_IDLE_ATTESTED,
                LeaseState.WORK_UNPAUSING,
                LeaseState.WORK_ACTIVE,
            ],
            id="interactive-judge",
        ),
        pytest.param(
            [
                LeaseState.WORK_ACTIVE,
                LeaseState.PAUSE_REQUESTED,
                LeaseState.WORK_PAUSING,
                LeaseState.WORK_PAUSED,
                LeaseState.WORK_GPU_IDLE_ATTESTED,
                LeaseState.JUDGE_STARTING,
                LeaseState.JUDGE_CLEANING,
                LeaseState.JUDGE_CLEANED,
                LeaseState.JUDGE_GPU_IDLE_ATTESTED,
                LeaseState.WORK_UNPAUSING,
                LeaseState.WORK_ACTIVE,
            ],
            id="judge-create-failure-cleanup",
        ),
        pytest.param(
            [
                LeaseState.WORK_ACTIVE,
                LeaseState.PAUSE_REQUESTED,
                LeaseState.WORK_PAUSING,
                LeaseState.WORK_PAUSED,
                LeaseState.WORK_UNPAUSING,
                LeaseState.WORK_ACTIVE,
            ],
            id="pre-judge-busy-retry",
        ),
    ],
)
def test_legal_interactive_workflows(
    identity: LeaseIdentity,
    states: list[LeaseState],
) -> None:
    lease = ResourceLease(identity)

    _advance(lease, states)

    assert lease.snapshot().state is LeaseState.WORK_ACTIVE
    assert lease.snapshot().epoch == len(states)
    assert [record.to_state for record in lease.audit_records()] == states


@pytest.mark.parametrize(
    "states",
    [
        pytest.param(
            [
                LeaseState.WORK_ACTIVE,
                LeaseState.FINALIZING,
                LeaseState.FINAL_WORK_FROZEN,
                LeaseState.FINAL_ARCHIVE_READY,
                LeaseState.FINAL_WORK_DESTROYING,
                LeaseState.FINAL_WORK_DESTROYED,
                LeaseState.FINAL_GPU_IDLE_ATTESTED,
                LeaseState.FINAL_JUDGE_STARTING,
                LeaseState.FINAL_JUDGE_RUNNING,
                LeaseState.FINAL_JUDGE_CLEANING,
                LeaseState.FINAL_JUDGE_CLEANED,
                LeaseState.FINAL_JUDGE_GPU_IDLE_ATTESTED,
                LeaseState.CLOSED,
            ],
            id="final-judge",
        ),
        pytest.param(
            [
                LeaseState.WORK_ACTIVE,
                LeaseState.FINALIZING,
                LeaseState.FINAL_WORK_FROZEN,
                LeaseState.FINAL_ARCHIVE_READY,
                LeaseState.FINAL_WORK_DESTROYING,
                LeaseState.FINAL_WORK_DESTROYED,
                LeaseState.FINAL_GPU_IDLE_ATTESTED,
                LeaseState.FINAL_JUDGE_STARTING,
                LeaseState.FINAL_JUDGE_CLEANING,
                LeaseState.FINAL_JUDGE_CLEANED,
                LeaseState.FINAL_JUDGE_GPU_IDLE_ATTESTED,
                LeaseState.CLOSED,
            ],
            id="final-judge-create-failure-cleanup",
        ),
        pytest.param(
            [
                LeaseState.WORK_ACTIVE,
                LeaseState.FINALIZING,
                LeaseState.FINAL_WORK_FROZEN,
                LeaseState.FINAL_ARCHIVE_READY,
                LeaseState.FINAL_WORK_DESTROYING,
                LeaseState.FINAL_WORK_DESTROYED,
                LeaseState.FINAL_GPU_IDLE_ATTESTED,
                LeaseState.CHECKPOINT_ONLY_CLOSING,
                LeaseState.CLOSED,
            ],
            id="explicit-checkpoint-only",
        ),
    ],
)
def test_legal_irreversible_final_workflows(
    identity: LeaseIdentity,
    states: list[LeaseState],
) -> None:
    lease = ResourceLease(identity)

    _advance(lease, states)

    assert lease.snapshot().state is LeaseState.CLOSED
    assert [record.to_state for record in lease.audit_records()] == states
    with pytest.raises(IllegalLeaseTransition, match="terminal state CLOSED"):
        lease.transition(
            identity,
            LeaseState.WORK_ACTIVE,
            expected_epoch=len(states),
        )


@pytest.mark.parametrize(
    "entry_states",
    [
        pytest.param([LeaseState.WORK_ACTIVE], id="active"),
        pytest.param(
            [
                LeaseState.WORK_ACTIVE,
                LeaseState.PAUSE_REQUESTED,
                LeaseState.WORK_PAUSING,
                LeaseState.WORK_PAUSED,
            ],
            id="paused",
        ),
        pytest.param(
            [
                LeaseState.WORK_ACTIVE,
                LeaseState.PAUSE_REQUESTED,
                LeaseState.WORK_PAUSING,
                LeaseState.WORK_PAUSED,
                LeaseState.WORK_GPU_IDLE_ATTESTED,
            ],
            id="work-gpu-idle",
        ),
        pytest.param(
            [
                LeaseState.WORK_ACTIVE,
                LeaseState.PAUSE_REQUESTED,
                LeaseState.WORK_PAUSING,
                LeaseState.WORK_PAUSED,
                LeaseState.WORK_GPU_IDLE_ATTESTED,
                LeaseState.JUDGE_STARTING,
                LeaseState.JUDGE_RUNNING,
                LeaseState.JUDGE_CLEANING,
                LeaseState.JUDGE_CLEANED,
                LeaseState.JUDGE_GPU_IDLE_ATTESTED,
            ],
            id="post-judge-gpu-idle",
        ),
    ],
)
def test_deadline_finalizes_without_resuming_work(
    identity: LeaseIdentity,
    entry_states: list[LeaseState],
) -> None:
    lease = ResourceLease(identity)
    final_states = [
        LeaseState.FINALIZING,
        LeaseState.FINAL_WORK_FROZEN,
        LeaseState.FINAL_ARCHIVE_READY,
        LeaseState.FINAL_WORK_DESTROYING,
        LeaseState.FINAL_WORK_DESTROYED,
        LeaseState.FINAL_GPU_IDLE_ATTESTED,
        LeaseState.FINAL_JUDGE_STARTING,
        LeaseState.FINAL_JUDGE_RUNNING,
        LeaseState.FINAL_JUDGE_CLEANING,
        LeaseState.FINAL_JUDGE_CLEANED,
        LeaseState.FINAL_JUDGE_GPU_IDLE_ATTESTED,
        LeaseState.CLOSED,
    ]

    _advance(lease, entry_states + final_states)

    audited_states = [record.to_state for record in lease.audit_records()]
    assert lease.snapshot().state is LeaseState.CLOSED
    assert LeaseState.WORK_UNPAUSING not in audited_states


@pytest.mark.parametrize(
    ("states", "illegal_target"),
    [
        pytest.param(
            [
                LeaseState.WORK_ACTIVE,
                LeaseState.PAUSE_REQUESTED,
                LeaseState.WORK_PAUSING,
                LeaseState.WORK_PAUSED,
            ],
            LeaseState.JUDGE_STARTING,
            id="judge-before-work-gpu-idle-attested",
        ),
        pytest.param(
            [
                LeaseState.WORK_ACTIVE,
                LeaseState.PAUSE_REQUESTED,
                LeaseState.WORK_PAUSING,
                LeaseState.WORK_PAUSED,
                LeaseState.WORK_GPU_IDLE_ATTESTED,
            ],
            LeaseState.WORK_UNPAUSING,
            id="busy-retry-after-work-idle-handoff",
        ),
        pytest.param(
            [
                LeaseState.WORK_ACTIVE,
                LeaseState.PAUSE_REQUESTED,
                LeaseState.WORK_PAUSING,
                LeaseState.WORK_PAUSED,
                LeaseState.WORK_GPU_IDLE_ATTESTED,
                LeaseState.JUDGE_STARTING,
            ],
            LeaseState.WORK_UNPAUSING,
            id="busy-retry-after-judge-start-attempt",
        ),
        pytest.param(
            [
                LeaseState.WORK_ACTIVE,
                LeaseState.PAUSE_REQUESTED,
                LeaseState.WORK_PAUSING,
                LeaseState.WORK_PAUSED,
                LeaseState.WORK_GPU_IDLE_ATTESTED,
                LeaseState.JUDGE_STARTING,
                LeaseState.JUDGE_CLEANING,
                LeaseState.JUDGE_CLEANED,
            ],
            LeaseState.WORK_UNPAUSING,
            id="unpause-before-judge-post-idle-attested",
        ),
        pytest.param(
            [
                LeaseState.WORK_ACTIVE,
                LeaseState.PAUSE_REQUESTED,
                LeaseState.WORK_PAUSING,
                LeaseState.WORK_PAUSED,
                LeaseState.FINALIZING,
            ],
            LeaseState.WORK_UNPAUSING,
            id="unpause-after-irreversible-final",
        ),
        pytest.param(
            [
                LeaseState.WORK_ACTIVE,
                LeaseState.FINALIZING,
                LeaseState.FINAL_WORK_FROZEN,
                LeaseState.FINAL_ARCHIVE_READY,
            ],
            LeaseState.CHECKPOINT_ONLY_CLOSING,
            id="checkpoint-before-work-destroy-and-idle",
        ),
        pytest.param(
            [
                LeaseState.WORK_ACTIVE,
                LeaseState.FINALIZING,
                LeaseState.FINAL_WORK_FROZEN,
                LeaseState.FINAL_ARCHIVE_READY,
            ],
            LeaseState.FINAL_JUDGE_STARTING,
            id="final-judge-before-work-destroy-and-idle",
        ),
        pytest.param(
            [
                LeaseState.WORK_ACTIVE,
                LeaseState.FINALIZING,
                LeaseState.FINAL_WORK_FROZEN,
                LeaseState.FINAL_ARCHIVE_READY,
                LeaseState.FINAL_WORK_DESTROYING,
            ],
            LeaseState.CLOSED,
            id="close-during-final-work-destroy",
        ),
        pytest.param(
            [
                LeaseState.WORK_ACTIVE,
                LeaseState.FINALIZING,
                LeaseState.FINAL_WORK_FROZEN,
                LeaseState.FINAL_ARCHIVE_READY,
                LeaseState.FINAL_WORK_DESTROYING,
                LeaseState.FINAL_WORK_DESTROYED,
                LeaseState.FINAL_GPU_IDLE_ATTESTED,
                LeaseState.FINAL_JUDGE_STARTING,
                LeaseState.FINAL_JUDGE_CLEANING,
                LeaseState.FINAL_JUDGE_CLEANED,
            ],
            LeaseState.CLOSED,
            id="final-close-before-judge-post-idle-attested",
        ),
    ],
)
def test_protocol_rejects_unsafe_shortcuts(
    identity: LeaseIdentity,
    states: list[LeaseState],
    illegal_target: LeaseState,
) -> None:
    lease = ResourceLease(identity)
    _advance(lease, states)

    with pytest.raises(IllegalLeaseTransition):
        lease.transition(
            identity,
            illegal_target,
            expected_epoch=len(states),
        )

    assert lease.snapshot().state is states[-1]
    assert lease.snapshot().epoch == len(states)


def test_baseline_cache_is_not_a_resource_ownership_state() -> None:
    assert not hasattr(LeaseState, "FINAL_CACHED_BASELINE_READY")


@pytest.mark.parametrize(
    ("expected_epoch", "error"),
    [(-1, StaleLeaseEpoch), (1, FutureLeaseEpoch)],
)
def test_epoch_fencing_rejects_stale_and_future_mutations(
    identity: LeaseIdentity,
    expected_epoch: int,
    error: type[Exception],
) -> None:
    lease = ResourceLease(identity)

    with pytest.raises(error):
        lease.transition(
            identity,
            LeaseState.WORK_ACTIVE,
            expected_epoch=expected_epoch,
        )

    assert lease.snapshot().epoch == 0
    assert lease.audit_records() == ()


def test_immediate_identical_retry_is_idempotent(identity: LeaseIdentity) -> None:
    lease = ResourceLease(identity)
    first = lease.transition(
        identity,
        LeaseState.WORK_ACTIVE,
        expected_epoch=0,
    )

    replay = lease.transition(
        identity,
        LeaseState.WORK_ACTIVE,
        expected_epoch=0,
    )

    assert replay == first
    assert replay.epoch == 1
    assert len(lease.audit_records()) == 1


def test_old_matching_transition_is_not_replayed_after_progress(
    identity: LeaseIdentity,
) -> None:
    lease = ResourceLease(identity)
    _advance(lease, [LeaseState.WORK_ACTIVE, LeaseState.PAUSE_REQUESTED])

    with pytest.raises(StaleLeaseEpoch):
        lease.transition(
            identity,
            LeaseState.WORK_ACTIVE,
            expected_epoch=0,
        )


def test_exact_identity_is_required_for_every_mutation(
    identity: LeaseIdentity,
) -> None:
    lease = ResourceLease(identity)
    wrong_container = LeaseIdentity(
        trusted_lease_id=identity.trusted_lease_id,
        run_id=identity.run_id,
        task_id=identity.task_id,
        allocation_key=identity.allocation_key,
        work_container_id="sha256:different-container",
    )

    with pytest.raises(LeaseIdentityError, match="exactly match"):
        lease.transition(
            wrong_container,
            LeaseState.WORK_ACTIVE,
            expected_epoch=0,
        )

    assert lease.snapshot().epoch == 0


@pytest.mark.parametrize(
    "replacement",
    [
        {"trusted_lease_id": "lease-authority-17"},
        {"run_id": "run-1", "task_id": "task-a"},
        {"allocation_key": ALLOCATION_KEY},
        {"work_container_id": "sha256:work-container-exact-id"},
    ],
)
def test_registry_rejects_identity_and_binding_collisions(
    identity: LeaseIdentity,
    replacement: dict[str, str],
) -> None:
    registry = ResourceLeaseRegistry()
    registry.create(identity)
    values = {
        "trusted_lease_id": "lease-other",
        "run_id": "run-other",
        "task_id": "task-other",
        "allocation_key": "allocation-sha256:" + "b" * 64,
        "work_container_id": "sha256:other-container",
    }
    values.update(replacement)

    with pytest.raises(LeaseIdentityCollision):
        registry.create(LeaseIdentity(**values))


@pytest.mark.parametrize(
    "failure_state",
    [
        LeaseState.FAILED_WORK_ACTIVE,
        LeaseState.FAILED_STOPPED,
        LeaseState.FENCED,
    ],
)
def test_all_failure_states_are_terminal_and_not_auto_releasable(
    identity: LeaseIdentity,
    failure_state: LeaseState,
) -> None:
    registry = ResourceLeaseRegistry()
    lease = registry.create(identity)
    failed = lease.fail_closed(
        identity,
        expected_epoch=0,
        reason="operator reconciliation required",
        failure_state=failure_state,
    )

    with pytest.raises(IllegalLeaseTransition, match="terminal state"):
        lease.transition(
            identity,
            LeaseState.WORK_ACTIVE,
            expected_epoch=1,
        )
    with pytest.raises(IllegalLeaseTransition, match="operator reconciliation"):
        registry.remove(identity, expected_epoch=1)

    assert failed.state is failure_state
    assert registry.get(identity) is lease


def test_fail_closed_retry_is_idempotent(identity: LeaseIdentity) -> None:
    lease = ResourceLease(identity)
    first = lease.fail_closed(
        identity,
        expected_epoch=0,
        reason="ownership unknown",
        failure_state=LeaseState.FENCED,
    )

    replay = lease.fail_closed(
        identity,
        expected_epoch=0,
        reason="ownership unknown",
        failure_state=LeaseState.FENCED,
    )

    assert replay == first
    assert len(lease.audit_records()) == 1


def test_registry_refuses_to_remove_nonterminal_lease(
    identity: LeaseIdentity,
) -> None:
    registry = ResourceLeaseRegistry()
    lease = registry.create(identity)

    with pytest.raises(IllegalLeaseTransition, match="nonterminal"):
        registry.remove(identity, expected_epoch=0)

    assert registry.get(identity) is lease


def test_clean_close_retires_authority_and_run_task_but_reuses_allocation(
    identity: LeaseIdentity,
) -> None:
    registry = ResourceLeaseRegistry()
    lease = registry.create(identity)
    _checkpoint_close(lease)

    removed = registry.remove(identity, expected_epoch=9)

    assert removed.state is LeaseState.CLOSED
    assert len(registry) == 0
    with pytest.raises(LeaseIdentityCollision, match="retired"):
        registry.create(identity)

    same_run_new_authority = LeaseIdentity(
        trusted_lease_id="lease-authority-18",
        run_id=identity.run_id,
        task_id=identity.task_id,
        allocation_key=identity.allocation_key,
        work_container_id="sha256:replacement-work-container",
    )
    with pytest.raises(LeaseIdentityCollision, match="run/task has been finalized"):
        registry.create(same_run_new_authority)
    with pytest.raises(LeaseIdentityCollision, match="run/task has been finalized"):
        registry.create_or_get(same_run_new_authority)

    later_identity = LeaseIdentity(
        trusted_lease_id="lease-authority-19",
        run_id="run-2",
        task_id=identity.task_id,
        allocation_key=identity.allocation_key,
        work_container_id=identity.work_container_id,
    )
    later = registry.create(later_identity)

    # Retired remove replay is harmless after every physical index is reused.
    assert registry.remove(identity, expected_epoch=9) == removed
    assert registry.get(later_identity) is later

    # Epoch zero from the retired authority cannot fence or mutate the new one.
    with pytest.raises(LeaseIdentityError, match="exactly match"):
        later.transition(
            identity,
            LeaseState.WORK_ACTIVE,
            expected_epoch=0,
        )
    assert later.snapshot().epoch == 0


def test_create_or_get_is_idempotent_only_for_exact_active_identity(
    identity: LeaseIdentity,
) -> None:
    registry = ResourceLeaseRegistry()

    first = registry.create_or_get(identity)
    replay = registry.create_or_get(identity)

    assert replay is first
    colliding = LeaseIdentity(
        trusted_lease_id=identity.trusted_lease_id,
        run_id=identity.run_id,
        task_id=identity.task_id,
        allocation_key=identity.allocation_key,
        work_container_id="sha256:colliding-container",
    )
    with pytest.raises(LeaseIdentityCollision):
        registry.create_or_get(colliding)


def test_remove_retry_is_exact_identity_and_epoch_fenced(
    identity: LeaseIdentity,
) -> None:
    registry = ResourceLeaseRegistry()
    lease = registry.create(identity)
    _checkpoint_close(lease)
    first = registry.remove(identity, expected_epoch=9)

    assert registry.remove(identity, expected_epoch=9) == first
    with pytest.raises(StaleLeaseEpoch):
        registry.remove(identity, expected_epoch=8)
    with pytest.raises(FutureLeaseEpoch):
        registry.remove(identity, expected_epoch=10)
    colliding = LeaseIdentity(
        trusted_lease_id=identity.trusted_lease_id,
        run_id=identity.run_id,
        task_id=identity.task_id,
        allocation_key=identity.allocation_key,
        work_container_id="sha256:colliding-container",
    )
    with pytest.raises(LeaseIdentityCollision):
        registry.remove(colliding, expected_epoch=9)


def test_allocation_key_requires_canonical_scheduler_digest() -> None:
    with pytest.raises(LeaseIdentityError, match="allocation-sha256"):
        LeaseIdentity(
            trusted_lease_id="lease-authority-raw-allocation",
            run_id="run-raw",
            task_id="task-raw",
            allocation_key="gpu-0",
            work_container_id="sha256:raw-allocation-container",
        )


def test_audit_is_deterministic_json(identity: LeaseIdentity) -> None:
    lease = ResourceLease(identity)
    _advance(lease, [LeaseState.WORK_ACTIVE, LeaseState.PAUSE_REQUESTED])

    encoded = lease.audit_json()
    decoded = json.loads(encoded)

    assert encoded == lease.audit_json()
    assert decoded[0] == {
        "action": "transition",
        "allocation_key": ALLOCATION_KEY,
        "epoch": 1,
        "from_state": "UNBOUND",
        "reason": None,
        "run_id": "run-1",
        "sequence": 1,
        "task_id": "task-a",
        "to_state": "WORK_ACTIVE",
        "trusted_lease_id": "lease-authority-17",
        "work_container_id": "sha256:work-container-exact-id",
    }
