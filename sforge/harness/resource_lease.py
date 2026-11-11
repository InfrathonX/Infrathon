"""Host-side coordinator state and fencing for a serial work/judge lease.

This module deliberately has no backend or device dependencies.  A deployment
owns the meaning of ``allocation_key`` and performs the actual stop/start
operations; this state machine only records which operation may happen next.
Transitions are trusted coordinator control logic, not a backend-attestation
security boundary.  States ending in ``_ATTESTED`` record evidence that the
coordinator has already verified through a backend-specific mechanism.
Entering ``FINALIZING`` requires the coordinator to persist a no-resume
tombstone.  ``FINAL_WORK_FROZEN`` additionally requires a stable filesystem
boundary, whether Work was stopped or safely paused; paused capture and the
durable tombstone are coordinator requirements, not backend features provided
by this module.
Baseline-cache lifecycle is durable metadata outside this ownership machine.
Registry tombstones are durable only for the lifetime of the registry process;
process-crash durability requires persisting them when runtime wiring is added.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping


_ALLOCATION_KEY_RE = re.compile(r"allocation-sha256:[0-9a-f]{64}")


class LeaseError(RuntimeError):
    """Base class for resource lease errors."""


class LeaseIdentityError(LeaseError):
    """The supplied identity is malformed or does not own the lease."""


class LeaseIdentityCollision(LeaseIdentityError):
    """A trusted identity or physical binding is already registered."""


class LeaseEpochError(LeaseError):
    """The mutation epoch is not the current lease epoch."""


class StaleLeaseEpoch(LeaseEpochError):
    """The mutation was issued against an epoch that has already passed."""


class FutureLeaseEpoch(LeaseEpochError):
    """The mutation was issued against an epoch not yet reached."""


class IllegalLeaseTransition(LeaseError):
    """The requested state transition is not in the transition graph."""


class LeaseState(str, Enum):
    """Observable states of one exclusive work/judge allocation."""

    UNBOUND = "UNBOUND"
    WORK_ACTIVE = "WORK_ACTIVE"
    PAUSE_REQUESTED = "PAUSE_REQUESTED"
    WORK_PAUSING = "WORK_PAUSING"
    WORK_PAUSED = "WORK_PAUSED"
    WORK_GPU_IDLE_ATTESTED = "WORK_GPU_IDLE_ATTESTED"
    JUDGE_STARTING = "JUDGE_STARTING"
    JUDGE_RUNNING = "JUDGE_RUNNING"
    JUDGE_CLEANING = "JUDGE_CLEANING"
    JUDGE_CLEANED = "JUDGE_CLEANED"
    JUDGE_GPU_IDLE_ATTESTED = "JUDGE_GPU_IDLE_ATTESTED"
    WORK_UNPAUSING = "WORK_UNPAUSING"
    FINALIZING = "FINALIZING"
    FINAL_WORK_FROZEN = "FINAL_WORK_FROZEN"
    FINAL_ARCHIVE_READY = "FINAL_ARCHIVE_READY"
    FINAL_WORK_DESTROYING = "FINAL_WORK_DESTROYING"
    FINAL_WORK_DESTROYED = "FINAL_WORK_DESTROYED"
    FINAL_GPU_IDLE_ATTESTED = "FINAL_GPU_IDLE_ATTESTED"
    FINAL_JUDGE_STARTING = "FINAL_JUDGE_STARTING"
    FINAL_JUDGE_RUNNING = "FINAL_JUDGE_RUNNING"
    FINAL_JUDGE_CLEANING = "FINAL_JUDGE_CLEANING"
    FINAL_JUDGE_CLEANED = "FINAL_JUDGE_CLEANED"
    FINAL_JUDGE_GPU_IDLE_ATTESTED = "FINAL_JUDGE_GPU_IDLE_ATTESTED"
    CHECKPOINT_ONLY_CLOSING = "CHECKPOINT_ONLY_CLOSING"
    CLOSED = "CLOSED"
    FAILED_WORK_ACTIVE = "FAILED_WORK_ACTIVE"
    FAILED_STOPPED = "FAILED_STOPPED"
    FENCED = "FENCED"


FAILURE_STATES = frozenset(
    {
        LeaseState.FAILED_WORK_ACTIVE,
        LeaseState.FAILED_STOPPED,
        LeaseState.FENCED,
    }
)
TERMINAL_STATES = frozenset(
    {
        LeaseState.CLOSED,
        *FAILURE_STATES,
    }
)
RELEASABLE_STATES = frozenset({LeaseState.CLOSED})

# Normal transitions are intentionally narrow: the allocation can belong to
# Work or Judge, never both.  WORK_PAUSED still has a known Work owner; only
# that pre-Judge state may take the normal busy-retry shortcut to
# WORK_UNPAUSING.  Post-Judge attributed-busy recovery uses the narrower
# ``recover_attributed_busy`` mutation rather than widening normal edges.
LEGAL_TRANSITIONS: Mapping[LeaseState, frozenset[LeaseState]] = MappingProxyType(
    {
        LeaseState.UNBOUND: frozenset({LeaseState.WORK_ACTIVE}),
        LeaseState.WORK_ACTIVE: frozenset(
            {LeaseState.PAUSE_REQUESTED, LeaseState.FINALIZING}
        ),
        LeaseState.PAUSE_REQUESTED: frozenset({LeaseState.WORK_PAUSING}),
        LeaseState.WORK_PAUSING: frozenset({LeaseState.WORK_PAUSED}),
        LeaseState.WORK_PAUSED: frozenset(
            {
                LeaseState.WORK_GPU_IDLE_ATTESTED,
                LeaseState.WORK_UNPAUSING,
                LeaseState.FINALIZING,
            }
        ),
        LeaseState.WORK_GPU_IDLE_ATTESTED: frozenset(
            {LeaseState.JUDGE_STARTING, LeaseState.FINALIZING}
        ),
        LeaseState.JUDGE_STARTING: frozenset(
            {LeaseState.JUDGE_RUNNING, LeaseState.JUDGE_CLEANING}
        ),
        LeaseState.JUDGE_RUNNING: frozenset({LeaseState.JUDGE_CLEANING}),
        LeaseState.JUDGE_CLEANING: frozenset({LeaseState.JUDGE_CLEANED}),
        LeaseState.JUDGE_CLEANED: frozenset(
            {LeaseState.JUDGE_GPU_IDLE_ATTESTED}
        ),
        LeaseState.JUDGE_GPU_IDLE_ATTESTED: frozenset(
            {LeaseState.WORK_UNPAUSING, LeaseState.FINALIZING}
        ),
        LeaseState.WORK_UNPAUSING: frozenset({LeaseState.WORK_ACTIVE}),
        LeaseState.FINALIZING: frozenset({LeaseState.FINAL_WORK_FROZEN}),
        LeaseState.FINAL_WORK_FROZEN: frozenset(
            {LeaseState.FINAL_ARCHIVE_READY}
        ),
        LeaseState.FINAL_ARCHIVE_READY: frozenset(
            {LeaseState.FINAL_WORK_DESTROYING}
        ),
        LeaseState.FINAL_WORK_DESTROYING: frozenset(
            {LeaseState.FINAL_WORK_DESTROYED}
        ),
        LeaseState.FINAL_WORK_DESTROYED: frozenset(
            {LeaseState.FINAL_GPU_IDLE_ATTESTED}
        ),
        LeaseState.FINAL_GPU_IDLE_ATTESTED: frozenset(
            {
                LeaseState.FINAL_JUDGE_STARTING,
                LeaseState.CHECKPOINT_ONLY_CLOSING,
            }
        ),
        LeaseState.FINAL_JUDGE_STARTING: frozenset(
            {
                LeaseState.FINAL_JUDGE_RUNNING,
                LeaseState.FINAL_JUDGE_CLEANING,
            }
        ),
        LeaseState.FINAL_JUDGE_RUNNING: frozenset(
            {LeaseState.FINAL_JUDGE_CLEANING}
        ),
        LeaseState.FINAL_JUDGE_CLEANING: frozenset(
            {LeaseState.FINAL_JUDGE_CLEANED}
        ),
        LeaseState.FINAL_JUDGE_CLEANED: frozenset(
            {LeaseState.FINAL_JUDGE_GPU_IDLE_ATTESTED}
        ),
        LeaseState.FINAL_JUDGE_GPU_IDLE_ATTESTED: frozenset(
            {LeaseState.CLOSED}
        ),
        LeaseState.CHECKPOINT_ONLY_CLOSING: frozenset({LeaseState.CLOSED}),
        LeaseState.CLOSED: frozenset(),
        LeaseState.FAILED_WORK_ACTIVE: frozenset(),
        LeaseState.FAILED_STOPPED: frozenset(),
        LeaseState.FENCED: frozenset(),
    }
)


def _identity_value(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise LeaseIdentityError(f"{name} must be a non-empty, trimmed string")
    if any(ord(character) < 32 for character in value):
        raise LeaseIdentityError(f"{name} must not contain control characters")
    return value


@dataclass(frozen=True)
class LeaseIdentity:
    """Immutable authority and exact deployment binding for one lease.

    ``trusted_lease_id`` is scheduler authority and must never be reused.
    ``allocation_key`` is a scheduler-issued digest of the canonical physical
    allocation, not a user-provided device index, UUID, or mutable alias.
    """

    trusted_lease_id: str
    run_id: str
    task_id: str
    allocation_key: str
    work_container_id: str

    def __post_init__(self) -> None:
        for name in (
            "trusted_lease_id",
            "run_id",
            "task_id",
            "work_container_id",
        ):
            _identity_value(name, getattr(self, name))
        allocation_key = _identity_value("allocation_key", self.allocation_key)
        if _ALLOCATION_KEY_RE.fullmatch(allocation_key) is None:
            raise LeaseIdentityError(
                "allocation_key must be allocation-sha256: followed by "
                "64 lowercase hexadecimal characters"
            )

    @property
    def run_task_key(self) -> tuple[str, str]:
        return (self.run_id, self.task_id)

    def to_dict(self) -> dict[str, str]:
        return {
            "trusted_lease_id": self.trusted_lease_id,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "allocation_key": self.allocation_key,
            "work_container_id": self.work_container_id,
        }


@dataclass(frozen=True)
class LeaseAuditRecord:
    """A deterministic, JSON-safe record of one accepted mutation."""

    sequence: int
    epoch: int
    trusted_lease_id: str
    run_id: str
    task_id: str
    allocation_key: str
    work_container_id: str
    from_state: LeaseState
    to_state: LeaseState
    action: str
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "epoch": self.epoch,
            "trusted_lease_id": self.trusted_lease_id,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "allocation_key": self.allocation_key,
            "work_container_id": self.work_container_id,
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "action": self.action,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class LeaseSnapshot:
    """Immutable view returned after a mutation or read."""

    identity: LeaseIdentity
    state: LeaseState
    epoch: int

    def to_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity.to_dict(),
            "state": self.state.value,
            "epoch": self.epoch,
        }


@dataclass(frozen=True)
class _LastMutation:
    expected_epoch: int
    from_state: LeaseState
    to_state: LeaseState
    action: str
    reason: str | None


@dataclass(frozen=True)
class _RetiredLeaseTombstone:
    identity: LeaseIdentity
    final_snapshot: LeaseSnapshot


class ResourceLease:
    """Thread-safe state machine for one exactly bound allocation.

    Every mutation requires the complete :class:`LeaseIdentity` and the epoch
    observed by the caller.  An immediate retry of the last identical mutation
    is idempotent and returns the current snapshot without incrementing the
    epoch or duplicating the audit record.
    """

    def __init__(self, identity: LeaseIdentity) -> None:
        if not isinstance(identity, LeaseIdentity):
            raise LeaseIdentityError("identity must be a LeaseIdentity")
        self._identity = identity
        self._state = LeaseState.UNBOUND
        self._epoch = 0
        self._audit: list[LeaseAuditRecord] = []
        self._last_mutation: _LastMutation | None = None
        self._lock = threading.RLock()

    @property
    def identity(self) -> LeaseIdentity:
        return self._identity

    def snapshot(self) -> LeaseSnapshot:
        with self._lock:
            return LeaseSnapshot(self._identity, self._state, self._epoch)

    def audit_records(self) -> tuple[LeaseAuditRecord, ...]:
        with self._lock:
            return tuple(self._audit)

    def audit_json(self) -> str:
        """Return stable JSON suitable for an append-only persistence layer."""

        with self._lock:
            values = [record.to_dict() for record in self._audit]
        return json.dumps(
            values,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    def transition(
        self,
        identity: LeaseIdentity,
        to_state: LeaseState,
        *,
        expected_epoch: int,
    ) -> LeaseSnapshot:
        """Apply one normal transition from :data:`LEGAL_TRANSITIONS`."""

        if not isinstance(to_state, LeaseState):
            raise IllegalLeaseTransition("to_state must be a LeaseState")
        if to_state in FAILURE_STATES:
            raise IllegalLeaseTransition(
                f"use fail_closed() to enter {to_state.value}"
            )
        return self._mutate(
            identity,
            to_state,
            expected_epoch=expected_epoch,
            action="transition",
            reason=None,
            require_normal_edge=True,
        )

    def fail_closed(
        self,
        identity: LeaseIdentity,
        *,
        expected_epoch: int,
        reason: str,
        failure_state: LeaseState,
    ) -> LeaseSnapshot:
        """Irreversibly record externally verified ownership after a failure.

        The caller must explicitly attest whether Work may still own the
        allocation (``FAILED_WORK_ACTIVE``), whether all use is confirmed
        stopped (``FAILED_STOPPED``), or whether ownership is ambiguous
        (``FENCED``).  Unknown ownership must never be reported as stopped.
        Every failure state remains quarantined until a future operator
        reconciliation API handles it; none is automatically releasable.
        """

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string")
        if (
            not isinstance(failure_state, LeaseState)
            or failure_state not in FAILURE_STATES
        ):
            allowed = ", ".join(sorted(state.value for state in FAILURE_STATES))
            raise IllegalLeaseTransition(
                f"failure_state must be one of: {allowed}"
            )
        return self._mutate(
            identity,
            failure_state,
            expected_epoch=expected_epoch,
            action="fail_closed",
            reason=reason,
            require_normal_edge=False,
        )

    def recover_attributed_busy(
        self,
        identity: LeaseIdentity,
        to_state: LeaseState,
        *,
        expected_epoch: int,
        reason: str,
    ) -> LeaseSnapshot:
        """Recover JUDGE_CLEANED only with trusted Work-owner attribution."""

        if self.snapshot().state != LeaseState.JUDGE_CLEANED:
            raise IllegalLeaseTransition(
                "attributed-busy recovery requires JUDGE_CLEANED"
            )
        if to_state not in {
            LeaseState.WORK_UNPAUSING,
            LeaseState.FINALIZING,
        }:
            raise IllegalLeaseTransition(
                "attributed-busy recovery may only resume or finalize Work"
            )
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string")
        return self._mutate(
            identity,
            to_state,
            expected_epoch=expected_epoch,
            action="recover_attributed_busy",
            reason=reason,
            require_normal_edge=False,
        )

    def _mutate(
        self,
        identity: LeaseIdentity,
        to_state: LeaseState,
        *,
        expected_epoch: int,
        action: str,
        reason: str | None,
        require_normal_edge: bool,
    ) -> LeaseSnapshot:
        with self._lock:
            self._check_identity(identity)
            if isinstance(expected_epoch, bool) or not isinstance(expected_epoch, int):
                raise LeaseEpochError("expected_epoch must be an integer")

            replay = _LastMutation(
                expected_epoch=expected_epoch,
                from_state=self._state if expected_epoch == self._epoch else (
                    self._last_mutation.from_state
                    if self._last_mutation is not None
                    else self._state
                ),
                to_state=to_state,
                action=action,
                reason=reason,
            )
            if expected_epoch < self._epoch:
                if self._last_mutation == replay:
                    return self._snapshot_unlocked()
                raise StaleLeaseEpoch(
                    f"expected epoch {expected_epoch}, current epoch is {self._epoch}"
                )
            if expected_epoch > self._epoch:
                raise FutureLeaseEpoch(
                    f"expected epoch {expected_epoch}, current epoch is {self._epoch}"
                )

            from_state = self._state
            if from_state in TERMINAL_STATES:
                raise IllegalLeaseTransition(
                    f"terminal state {from_state.value} cannot transition"
                )
            if require_normal_edge and to_state not in LEGAL_TRANSITIONS[from_state]:
                raise IllegalLeaseTransition(
                    f"illegal transition {from_state.value} -> {to_state.value}"
                )

            self._epoch += 1
            self._state = to_state
            self._last_mutation = _LastMutation(
                expected_epoch=expected_epoch,
                from_state=from_state,
                to_state=to_state,
                action=action,
                reason=reason,
            )
            self._audit.append(
                LeaseAuditRecord(
                    sequence=len(self._audit) + 1,
                    epoch=self._epoch,
                    **self._identity.to_dict(),
                    from_state=from_state,
                    to_state=to_state,
                    action=action,
                    reason=reason,
                )
            )
            return self._snapshot_unlocked()

    def _check_identity(self, identity: LeaseIdentity) -> None:
        if identity != self._identity:
            raise LeaseIdentityError(
                "mutation identity does not exactly match the lease binding"
            )

    def _snapshot_unlocked(self) -> LeaseSnapshot:
        return LeaseSnapshot(self._identity, self._state, self._epoch)


class ResourceLeaseRegistry:
    """In-memory ownership registry with non-reusable authority IDs.

    Safe removal releases allocation, run/task, and container indexes, but a
    retired trusted lease ID and its final run/task remain tombstoned for the
    registry lifetime.  The physical allocation may be reused by another run,
    but a finalized run/task can never acquire a new authority ID.
    """

    def __init__(self) -> None:
        self._leases: dict[str, ResourceLease] = {}
        self._run_tasks: dict[tuple[str, str], str] = {}
        self._allocations: dict[str, str] = {}
        self._work_containers: dict[str, str] = {}
        self._retired: dict[str, _RetiredLeaseTombstone] = {}
        self._retired_run_tasks: dict[
            tuple[str, str], _RetiredLeaseTombstone
        ] = {}
        self._lock = threading.RLock()

    def create(self, identity: LeaseIdentity) -> ResourceLease:
        if not isinstance(identity, LeaseIdentity):
            raise LeaseIdentityError("identity must be a LeaseIdentity")
        with self._lock:
            if identity.trusted_lease_id in self._retired:
                raise LeaseIdentityCollision(
                    "trusted lease identity has been retired and cannot be reused"
                )
            if identity.run_task_key in self._retired_run_tasks:
                raise LeaseIdentityCollision(
                    "run/task has been finalized and cannot acquire a new lease"
                )
            collisions = (
                ("trusted_lease_id", identity.trusted_lease_id, self._leases),
                ("run/task", identity.run_task_key, self._run_tasks),
                ("allocation_key", identity.allocation_key, self._allocations),
                (
                    "work_container_id",
                    identity.work_container_id,
                    self._work_containers,
                ),
            )
            for field, value, index in collisions:
                if value in index:
                    raise LeaseIdentityCollision(
                        f"{field} is already registered: {value!r}"
                    )

            lease = ResourceLease(identity)
            self._leases[identity.trusted_lease_id] = lease
            self._run_tasks[identity.run_task_key] = identity.trusted_lease_id
            self._allocations[identity.allocation_key] = identity.trusted_lease_id
            self._work_containers[identity.work_container_id] = (
                identity.trusted_lease_id
            )
            return lease

    def create_or_get(self, identity: LeaseIdentity) -> ResourceLease:
        """Create a lease or return the exact matching active registration."""

        if not isinstance(identity, LeaseIdentity):
            raise LeaseIdentityError("identity must be a LeaseIdentity")
        with self._lock:
            lease = self._leases.get(identity.trusted_lease_id)
            if lease is not None:
                if lease.identity != identity:
                    raise LeaseIdentityCollision(
                        "trusted lease identity collides with a different binding"
                    )
                return lease
            return self.create(identity)

    def get(self, identity: LeaseIdentity) -> ResourceLease:
        if not isinstance(identity, LeaseIdentity):
            raise LeaseIdentityError("identity must be a LeaseIdentity")
        with self._lock:
            lease = self._leases.get(identity.trusted_lease_id)
            if lease is None:
                raise LeaseIdentityError("trusted lease identity is not registered")
            if lease.identity != identity:
                raise LeaseIdentityCollision(
                    "trusted lease identity collides with a different binding"
                )
            return lease

    def remove(
        self,
        identity: LeaseIdentity,
        *,
        expected_epoch: int,
    ) -> LeaseSnapshot:
        """Remove an exactly identified clean CLOSED lease and release indexes.

        Failure terminals deliberately remain registered.  Releasing them
        requires an operator reconciliation API, which is not implemented by
        this host-side foundation.
        """

        with self._lock:
            if not isinstance(identity, LeaseIdentity):
                raise LeaseIdentityError("identity must be a LeaseIdentity")
            lease = self._leases.get(identity.trusted_lease_id)
            if lease is None:
                tombstone = self._retired.get(identity.trusted_lease_id)
                if tombstone is None:
                    raise LeaseIdentityError(
                        "trusted lease identity is not registered"
                    )
                if tombstone.identity != identity:
                    raise LeaseIdentityCollision(
                        "retired lease identity collides with a different binding"
                    )
                self._check_removal_epoch(
                    expected_epoch,
                    current_epoch=tombstone.final_snapshot.epoch,
                )
                return tombstone.final_snapshot
            if lease.identity != identity:
                raise LeaseIdentityCollision(
                    "trusted lease identity collides with a different binding"
                )
            snapshot = lease.snapshot()
            self._check_removal_epoch(
                expected_epoch,
                current_epoch=snapshot.epoch,
            )
            if snapshot.state not in TERMINAL_STATES:
                raise IllegalLeaseTransition(
                    f"cannot remove nonterminal lease in {snapshot.state.value}"
                )
            if snapshot.state not in RELEASABLE_STATES:
                raise IllegalLeaseTransition(
                    f"cannot automatically release allocation from "
                    f"{snapshot.state.value}; failure states require operator "
                    "reconciliation"
                )

            del self._leases[identity.trusted_lease_id]
            del self._run_tasks[identity.run_task_key]
            del self._allocations[identity.allocation_key]
            del self._work_containers[identity.work_container_id]
            tombstone = _RetiredLeaseTombstone(
                identity=identity,
                final_snapshot=snapshot,
            )
            self._retired[identity.trusted_lease_id] = tombstone
            self._retired_run_tasks[identity.run_task_key] = tombstone
            return snapshot

    @staticmethod
    def _check_removal_epoch(expected_epoch: int, *, current_epoch: int) -> None:
        if isinstance(expected_epoch, bool) or not isinstance(expected_epoch, int):
            raise LeaseEpochError("expected_epoch must be an integer")
        if expected_epoch < current_epoch:
            raise StaleLeaseEpoch(
                f"expected epoch {expected_epoch}, current epoch is "
                f"{current_epoch}"
            )
        if expected_epoch > current_epoch:
            raise FutureLeaseEpoch(
                f"expected epoch {expected_epoch}, current epoch is "
                f"{current_epoch}"
            )

    def __len__(self) -> int:
        with self._lock:
            return len(self._leases)
