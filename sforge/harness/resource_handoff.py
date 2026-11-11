"""Durable coordinator record for the opt-in single-GPU handoff path."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Mapping

from sforge.harness.accelerator_observer import AcceleratorIdleAttestation
from sforge.harness.backend import (
    AcceleratorAllocation,
    EVALUATION_CONTRACT_SHA256_ANNOTATION,
    HANDOFF_ALLOCATION_KEY_ANNOTATION,
    HANDOFF_DOCKER_PAUSE_PROTOCOL,
    HANDOFF_LEASE_ID_ANNOTATION,
    HANDOFF_PROTOCOL_ANNOTATION,
    HANDOFF_RUN_ID_ANNOTATION,
)
from sforge.harness.resource_lease import (
    LeaseIdentity,
    LeaseSnapshot,
    LeaseState,
    ResourceLease,
)


class ResourceHandoffError(RuntimeError):
    """The durable handoff contract could not be established."""


JUDGE_ROLE_ANNOTATION = "io.sforge.handoff.role"
JUDGE_ATTEMPT_ANNOTATION = "io.sforge.handoff.attempt"
_JUDGE_CONTAINER_ID_RE = re.compile(r"[0-9a-f]{64}", re.ASCII)
_SHA256_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}", re.ASCII)
_ANNOTATION_VALUE_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}",
    re.ASCII,
)


def canonical_allocation_key(allocation: AcceleratorAllocation) -> str:
    """Digest one canonical physical device set without accepting aliases."""

    payload = json.dumps(
        {
            "domain": "sforge-physical-accelerator-allocation-v1",
            "vendor": allocation.vendor,
            "device_ids": sorted(allocation.device_ids),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return "allocation-sha256:" + hashlib.sha256(payload).hexdigest()


def handoff_annotations(
    *,
    run_id: str,
    trusted_lease_id: str,
    allocation_key: str,
    evaluation_contract_sha256: str,
) -> dict[str, str]:
    """Return the complete reserved binding-label set."""

    return {
        HANDOFF_RUN_ID_ANNOTATION: run_id,
        HANDOFF_LEASE_ID_ANNOTATION: trusted_lease_id,
        HANDOFF_ALLOCATION_KEY_ANNOTATION: allocation_key,
        HANDOFF_PROTOCOL_ANNOTATION: HANDOFF_DOCKER_PAUSE_PROTOCOL,
        EVALUATION_CONTRACT_SHA256_ANNOTATION: evaluation_contract_sha256,
    }


def judge_handoff_annotations(
    binding: "HandoffBinding",
    *,
    attempt_id: str,
) -> dict[str, str]:
    """Bind one ephemeral Judge to the same lease and exact attempt."""

    if (
        not isinstance(attempt_id, str)
        or _ANNOTATION_VALUE_RE.fullmatch(attempt_id) is None
    ):
        raise ResourceHandoffError("invalid Judge handoff attempt ID")
    return {
        **dict(binding.annotations),
        JUDGE_ROLE_ANNOTATION: "judge",
        JUDGE_ATTEMPT_ANNOTATION: attempt_id,
    }


def lease_record_path(log_dir: Path, run_id: str, task_id: str) -> Path:
    return log_dir / "runs" / run_id / task_id / "resource_lease.json"


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


@dataclass(frozen=True)
class HandoffBinding:
    identity: LeaseIdentity
    allocation: AcceleratorAllocation
    evaluation_contract_sha256: str
    annotations: Mapping[str, str]


class DurableResourceLease:
    """Persist every accepted lease transition before exposing it.

    One Judge process owns this object.  The file is the cross-process fence
    consumed by ``run_agent``; it is also sufficient for crash diagnosis, but
    automatic crash recovery is intentionally unsupported.  A restarted Judge
    must not infer permission to unpause from this record.
    """

    SCHEMA_VERSION = 1

    def __init__(self, path: Path, binding: HandoffBinding) -> None:
        self.path = path
        self.binding = binding
        self._lease = ResourceLease(binding.identity)
        self._lock = threading.RLock()
        self._attestations: list[dict[str, object]] = []
        self._recovery_policy = "unpause_same_exec"
        self._final_requested = False
        self._archive: dict[str, object] | None = None
        self._judge_containers: list[dict[str, object]] = []
        self._final_judge: dict[str, object] | None = None

    @classmethod
    def create(cls, path: Path, binding: HandoffBinding) -> "DurableResourceLease":
        if path.exists():
            raise ResourceHandoffError(
                "resource lease record already exists; automatic crash "
                "recovery and lease-ID reuse are forbidden"
            )
        lease = cls(path, binding)
        lease._persist()
        return lease

    @property
    def recovery_policy(self) -> str:
        with self._lock:
            return self._recovery_policy

    @property
    def final_requested(self) -> bool:
        with self._lock:
            return self._final_requested

    def snapshot(self) -> LeaseSnapshot:
        return self._lease.snapshot()

    def transition(
        self,
        to_state: LeaseState,
        *,
        attestation: AcceleratorIdleAttestation | None = None,
    ) -> LeaseSnapshot:
        if to_state in {
            LeaseState.WORK_GPU_IDLE_ATTESTED,
            LeaseState.JUDGE_GPU_IDLE_ATTESTED,
            LeaseState.FINAL_GPU_IDLE_ATTESTED,
            LeaseState.FINAL_JUDGE_GPU_IDLE_ATTESTED,
        } and attestation is None:
            raise ResourceHandoffError(
                f"{to_state.value} requires real accelerator-idle attestation"
            )
        with self._lock:
            before = self._lease.snapshot()
            snapshot = self._lease.transition(
                self.binding.identity,
                to_state,
                expected_epoch=before.epoch,
            )
            if attestation is not None:
                evidence = attestation.to_dict()
                evidence["accepted_epoch"] = snapshot.epoch
                evidence["accepted_state"] = snapshot.state.value
                self._attestations.append(evidence)
            try:
                self._persist()
            except Exception as exc:
                # The in-memory process can no longer prove whether another
                # process observed this mutation.  Fence permanently.
                try:
                    current = self._lease.snapshot()
                    self._lease.fail_closed(
                        self.binding.identity,
                        expected_epoch=current.epoch,
                        reason="durable lease persistence failed",
                        failure_state=LeaseState.FENCED,
                    )
                    self._persist()
                except Exception:
                    pass
                raise ResourceHandoffError(
                    "durable lease transition could not be persisted"
                ) from exc
            return snapshot

    def request_final_no_resume(self) -> dict[str, object]:
        """Durably close admission before any final Work lifecycle action."""

        with self._lock:
            self._recovery_policy = "never_resume"
            self._final_requested = True
            self._persist()
            return self.to_dict()

    def record_archive(self, *, sha256: str, size: int) -> None:
        if (
            not isinstance(sha256, str)
            or _SHA256_DIGEST_RE.fullmatch(sha256) is None
        ):
            raise ResourceHandoffError("invalid final archive digest")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ResourceHandoffError("invalid final archive size")
        with self._lock:
            self._archive = {"sha256": sha256, "size": size}
            self._persist()

    @property
    def archive(self) -> dict[str, object] | None:
        with self._lock:
            return dict(self._archive) if self._archive is not None else None

    @property
    def final_judge(self) -> dict[str, object] | None:
        with self._lock:
            if self._final_judge is None:
                return None
            # JSON round-trip provides a defensive deep copy of the persisted
            # request/result without accepting custom object types.
            return json.loads(json.dumps(self._final_judge))

    def record_judge_container(
        self,
        *,
        container_id: str,
        annotations: Mapping[str, str],
        final: bool,
    ) -> None:
        """Persist an exact authenticated Judge owner before RUNNING."""

        if (
            not isinstance(container_id, str)
            or _JUDGE_CONTAINER_ID_RE.fullmatch(container_id) is None
        ):
            raise ResourceHandoffError(
                "Judge container ID must be canonical 64-hex"
            )
        expected_state = (
            LeaseState.FINAL_JUDGE_STARTING
            if final
            else LeaseState.JUDGE_STARTING
        )
        with self._lock:
            snapshot = self._lease.snapshot()
            if snapshot.state != expected_state:
                raise ResourceHandoffError(
                    "Judge container ownership was reported in an unsafe state"
                )
            record = {
                "container_id": container_id,
                "annotations": dict(annotations),
                "final": final,
                "recorded_epoch": snapshot.epoch,
                "recorded_at": time.time(),
            }
            for existing in self._judge_containers:
                if existing.get("annotations") == record["annotations"]:
                    if (
                        existing.get("container_id") != container_id
                        or existing.get("final") is not final
                    ):
                        raise ResourceHandoffError(
                            "Judge attempt is already bound to another container"
                        )
                    return
            self._judge_containers.append(record)
            self._persist()

    def record_final_judge_request(
        self,
        request: Mapping[str, object],
    ) -> None:
        """Bind the sole final Judge request before any new GPU owner."""

        request_copy = json.loads(json.dumps(dict(request), allow_nan=False))
        with self._lock:
            if self._lease.snapshot().state != LeaseState.FINAL_GPU_IDLE_ATTESTED:
                raise ResourceHandoffError(
                    "final Judge request requires FINAL_GPU_IDLE_ATTESTED"
                )
            if self._final_judge is not None:
                if self._final_judge.get("request") != request_copy:
                    raise ResourceHandoffError(
                        "final Judge is already bound to a different request"
                    )
                return
            self._final_judge = {
                "request": request_copy,
                "report": None,
                "requested_at": time.time(),
            }
            self._persist()

    def record_final_judge_report(self, report: Mapping[str, object]) -> None:
        report_copy = json.loads(json.dumps(dict(report), allow_nan=False))
        with self._lock:
            if self._final_judge is None:
                raise ResourceHandoffError("final Judge request is not recorded")
            existing = self._final_judge.get("report")
            if existing is not None and existing != report_copy:
                raise ResourceHandoffError(
                    "final Judge report conflicts with durable result"
                )
            self._final_judge["report"] = report_copy
            self._final_judge["reported_at"] = time.time()
            self._persist()

    def record_final_judge_artifacts(
        self,
        artifacts: Mapping[str, object],
    ) -> None:
        artifacts_copy = json.loads(
            json.dumps(dict(artifacts), allow_nan=False)
        )
        with self._lock:
            if self._final_judge is None:
                raise ResourceHandoffError("final Judge request is not recorded")
            existing = self._final_judge.get("artifacts")
            if existing is not None and existing != artifacts_copy:
                raise ResourceHandoffError(
                    "final Judge artifacts conflict with durable result"
                )
            self._final_judge["artifacts"] = artifacts_copy
            self._persist()

    def recover_attributed_busy(self, to_state: LeaseState) -> LeaseSnapshot:
        """Persist the narrow post-Judge Work-owner recovery mutation."""

        with self._lock:
            current = self._lease.snapshot()
            snapshot = self._lease.recover_attributed_busy(
                self.binding.identity,
                to_state,
                expected_epoch=current.epoch,
                reason="trusted observer attributed all busy contexts to Work",
            )
            try:
                self._persist()
            except Exception as exc:
                try:
                    fenced = self._lease.snapshot()
                    self._lease.fail_closed(
                        self.binding.identity,
                        expected_epoch=fenced.epoch,
                        reason="durable busy recovery persistence failed",
                        failure_state=LeaseState.FENCED,
                    )
                    self._persist()
                except Exception:
                    pass
                raise ResourceHandoffError(
                    "durable busy recovery could not be persisted"
                ) from exc
            return snapshot

    def fail_closed(self, reason: str, state: LeaseState) -> LeaseSnapshot:
        with self._lock:
            current = self._lease.snapshot()
            snapshot = self._lease.fail_closed(
                self.binding.identity,
                expected_epoch=current.epoch,
                reason=reason,
                failure_state=state,
            )
            self._recovery_policy = "never_resume"
            self._persist()
            return snapshot

    def to_dict(self) -> dict[str, object]:
        snapshot = self._lease.snapshot()
        return {
            "schema_version": self.SCHEMA_VERSION,
            "identity": self.binding.identity.to_dict(),
            "allocation": {
                "vendor": self.binding.allocation.vendor,
                "device_ids": list(self.binding.allocation.device_ids),
            },
            "evaluation_contract_sha256": (
                self.binding.evaluation_contract_sha256
            ),
            "annotations": dict(self.binding.annotations),
            "state": snapshot.state.value,
            "epoch": snapshot.epoch,
            "recovery_policy": self._recovery_policy,
            "final_requested": self._final_requested,
            "archive": self._archive,
            "judge_containers": list(self._judge_containers),
            "final_judge": self._final_judge,
            "attestations": list(self._attestations),
            "audit": [record.to_dict() for record in self._lease.audit_records()],
            "updated_at": time.time(),
        }

    def _persist(self) -> None:
        _atomic_json(self.path, self.to_dict())


def read_durable_lease(path: Path) -> dict[str, object]:
    """Read a complete durable fence without following a symlink."""

    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ResourceHandoffError("resource lease record is not regular")
        with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
            descriptor = -1
            value = json.load(stream)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ResourceHandoffError("resource lease record is malformed")
    return value


class ExclusiveProcessLock:
    """Hold an advisory cross-process fence for a trusted lock domain.

    Non-blocking instances protect a coordinator for its full lifetime;
    blocking instances serialize only short durable identity admissions.
    """

    def __init__(
        self,
        path: Path,
        description: str,
        *,
        blocking: bool = False,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream: IO[str] = path.open("a+", encoding="utf-8")
        try:
            operation = fcntl.LOCK_EX
            if not blocking:
                operation |= fcntl.LOCK_NB
            fcntl.flock(self._stream.fileno(), operation)
        except BlockingIOError as exc:
            self._stream.close()
            raise ResourceHandoffError(
                f"{description} already has an active owner; multi-process "
                "handoff coordination is unsupported"
            ) from exc
        os.fchmod(self._stream.fileno(), 0o600)

    def close(self) -> None:
        if not self._stream.closed:
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
            self._stream.close()
