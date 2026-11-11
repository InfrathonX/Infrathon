"""Abstract base classes for container backends."""

from __future__ import annotations

import abc
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from sforge.harness.cpu_affinity import CpuAffinityPolicy


@dataclass
class ExecResult:
    """Result of a simple (non-streaming) command execution."""

    output: str = ""
    exit_code: int = 0


@dataclass
class StreamingExecResult:
    """Result of a streaming command execution with timeout support."""

    output: str = ""
    exit_code: int = 0
    timed_out: bool = False
    elapsed_seconds: float = 0.0


@dataclass(frozen=True)
class ResolvedImage:
    """An immutable container image identity and its exact execution ref."""

    backend: str
    requested_ref: str
    execution_ref: str
    content_identity: str
    identity_kind: str


_ACCELERATOR_VENDOR_RE = re.compile(r"[a-z][a-z0-9._-]*", re.ASCII)
_ACCELERATOR_DEVICE_ID_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}",
    re.ASCII,
)
_NVIDIA_UUID_RE = r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}"
_NVIDIA_DEVICE_ID_RE = re.compile(
    rf"(?:GPU-{_NVIDIA_UUID_RE}|MIG-{_NVIDIA_UUID_RE}"
    rf"|MIG-GPU-{_NVIDIA_UUID_RE}/(?:0|[1-9][0-9]*)/(?:0|[1-9][0-9]*))",
    re.ASCII,
)

HANDOFF_RUN_ID_ANNOTATION = "io.sforge.handoff.run-id"
HANDOFF_LEASE_ID_ANNOTATION = "io.sforge.handoff.lease-id"
HANDOFF_ALLOCATION_KEY_ANNOTATION = "io.sforge.handoff.allocation-key"
HANDOFF_PROTOCOL_ANNOTATION = "io.sforge.handoff.protocol"
HANDOFF_DOCKER_PAUSE_PROTOCOL = "docker-pause-v1"
EVALUATION_CONTRACT_SHA256_ANNOTATION = (
    "io.sforge.evaluation-contract-sha256"
)
HANDOFF_BINDING_ANNOTATIONS = frozenset({
    HANDOFF_RUN_ID_ANNOTATION,
    HANDOFF_LEASE_ID_ANNOTATION,
    HANDOFF_ALLOCATION_KEY_ANNOTATION,
    HANDOFF_PROTOCOL_ANNOTATION,
    EVALUATION_CONTRACT_SHA256_ANNOTATION,
})


@dataclass(frozen=True)
class AcceleratorAllocation:
    """Physical devices assigned by a scheduler or deployment layer.

    This is runtime allocation output, not a task or evaluation hardware
    profile.  Passing the same allocation to multiple container creations lets
    those containers use the same devices without persisting host-specific
    device identifiers in task specifications.
    """

    vendor: str
    device_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.vendor, str):
            raise ValueError("accelerator vendor must be a string")
        vendor = self.vendor.strip().lower()
        if not _ACCELERATOR_VENDOR_RE.fullmatch(vendor):
            raise ValueError(f"invalid accelerator vendor: {self.vendor!r}")

        if isinstance(self.device_ids, (str, bytes)):
            raise ValueError("accelerator device_ids must be a sequence of IDs")
        try:
            device_ids = tuple(self.device_ids)
        except TypeError as exc:
            raise ValueError(
                "accelerator device_ids must be a sequence of IDs"
            ) from exc
        if not device_ids:
            raise ValueError("accelerator allocation must contain a device ID")

        seen: set[str] = set()
        for device_id in device_ids:
            if not isinstance(device_id, str):
                raise ValueError("accelerator device IDs must be strings")
            if (
                not _ACCELERATOR_DEVICE_ID_RE.fullmatch(device_id)
                or ".." in device_id
            ):
                raise ValueError(f"unsafe accelerator device ID: {device_id!r}")
            if (
                vendor == "nvidia"
                and not _NVIDIA_DEVICE_ID_RE.fullmatch(device_id)
            ):
                raise ValueError(
                    "NVIDIA accelerator device ID must be a canonical GPU "
                    f"UUID or MIG UUID, not an index or alias: {device_id!r}"
                )
            if device_id in seen:
                raise ValueError(
                    f"duplicate accelerator device ID: {device_id!r}"
                )
            seen.add(device_id)

        object.__setattr__(self, "vendor", vendor)
        object.__setattr__(self, "device_ids", device_ids)


class ContainerHandle(abc.ABC):
    """Opaque handle to a running container or pod."""

    @property
    @abc.abstractmethod
    def id(self) -> str:
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str:
        ...

    @property
    @abc.abstractmethod
    def ip_address(self) -> str | None:
        ...


class ContainerBackend(abc.ABC):
    """Abstract interface for container orchestration."""

    @property
    @abc.abstractmethod
    def backend_name(self) -> str:
        ...

    # --- Lifecycle ---

    @abc.abstractmethod
    def create_container(
        self,
        image: str,
        name: str,
        *,
        command: str = "tail -f /dev/null",
        environment: dict[str, str] | None = None,
        extra_hosts: dict[str, str] | None = None,
        cap_drop: list[str] | None = None,
        cpu_limit: int | None = None,
        mem_limit: str | None = None,
        network_mode: str | None = None,
        user: str | None = None,
        annotations: dict[str, str] | None = None,
        accelerator_allocation: AcceleratorAllocation | None = None,
    ) -> ContainerHandle:
        ...

    def get_container_handle(
        self,
        container_id: str,
        *,
        expected_annotations: dict[str, str],
    ) -> ContainerHandle:
        """Resolve an exact persisted container ID in another process.

        Backends verify scheduler/deployment binding annotations while
        resolving the handle.  These labels authenticate identity only; a
        persistent lease fence is still required across process boundaries to
        distinguish transient handoff from irreversible finalization.  The
        default is intentionally unsupported so a backend cannot silently
        resolve a mutable name instead of an exact ID.
        """
        raise NotImplementedError(
            f"{self.backend_name} does not support exact container ID lookup"
        )

    def authoritative_container_receipt(self, handle: ContainerHandle) -> dict:
        """Read actual live runtime identity from the trusted backend API.

        Task bootstraps may require a provider-specific receipt, including the
        running image and assigned devices. Requested create arguments and
        container environment are not runtime attestation. Unsupported
        backends fail closed rather than manufacturing such a receipt.
        """
        raise NotImplementedError(
            f"{self.backend_name} cannot attest actual container image/device identity"
        )

    @abc.abstractmethod
    def start_container(self, handle: ContainerHandle) -> None:
        ...

    def apply_cpu_binding(
        self,
        handle: ContainerHandle,
        policy: CpuAffinityPolicy,
    ) -> dict[str, object]:
        """Pin a created, not yet started container to host CPUs.

        The harness calls this between create and start only when an opt-in
        CPU affinity policy is configured (see sforge/harness/cpu_affinity.py)
        and returns the binding evidence for the host-side record.  The
        default fails closed so a configured policy never runs unpinned.
        """
        raise NotImplementedError(
            f"{self.backend_name} does not support CPU pinning"
        )

    @abc.abstractmethod
    def cleanup_container(
        self, handle: ContainerHandle | None, logger: logging.Logger | None = None,
    ) -> None:
        ...

    def freeze_container_for_snapshot(
        self,
        handle: ContainerHandle,
        *,
        stop_without_inspect: bool = False,
    ) -> dict[str, object]:
        """Stop all container processes before an immutable filesystem read.

        Backends that can provide this stronger final-snapshot contract should
        override this method.  It is intentionally not abstract so existing
        third-party backends keep working for ordinary SForge runs.
        """
        raise NotImplementedError(
            f"{self.backend_name} does not support stopped-container snapshots"
        )

    def verify_container_frozen_for_snapshot(
        self,
        handle: ContainerHandle,
    ) -> dict[str, object]:
        """Fail unless the container is still fully stopped."""
        raise NotImplementedError(
            f"{self.backend_name} does not support stopped-container snapshots"
        )

    def freeze_container_for_final_handoff_snapshot(
        self,
        handle: ContainerHandle,
        *,
        expected_annotations: dict[str, str],
        stop_without_inspect: bool = False,
    ) -> dict[str, object]:
        """Establish a final boundary for a durably no-resume handoff.

        Implementations may accept an already-paused container only on this
        explicit coordinator path.  Ordinary snapshot freezing must continue
        to reject a reversible handoff pause.
        """
        raise NotImplementedError(
            f"{self.backend_name} does not support final handoff snapshots"
        )

    def pause_container_for_handoff(
        self,
        handle: ContainerHandle,
    ) -> dict[str, object]:
        """Pause a container's processes while preserving live exec sessions.

        This lifecycle is deliberately separate from final snapshot freezing:
        a paused container is expected to be unpaused with the same identity,
        writable layer, and processes.  Container pause is not accelerator
        release attestation; a coordinator must perform that separately.
        """
        raise NotImplementedError(
            f"{self.backend_name} does not support transient container handoff"
        )

    def verify_container_paused_for_handoff(
        self,
        handle: ContainerHandle,
    ) -> dict[str, object]:
        """Fail unless the same-handle handoff container is strictly paused."""
        raise NotImplementedError(
            f"{self.backend_name} does not support transient container handoff"
        )

    def unpause_container_after_handoff(
        self,
        handle: ContainerHandle,
    ) -> dict[str, object]:
        """Unpause the exact same container and its existing processes."""
        raise NotImplementedError(
            f"{self.backend_name} does not support transient container handoff"
        )

    def verify_container_running_after_handoff(
        self,
        handle: ContainerHandle,
    ) -> dict[str, object]:
        """Fail unless a resumed handoff container is unambiguously running."""
        raise NotImplementedError(
            f"{self.backend_name} does not support transient container handoff"
        )

    def destroy_container_for_handoff(
        self,
        handle: ContainerHandle,
    ) -> dict[str, object]:
        """Strictly destroy an exact handoff container and verify absence.

        This is an irreversible container lifecycle gate, not accelerator-idle
        attestation.  Backends must propagate every stop, removal, or absence
        verification failure.  Same-handle retry evidence is process-local;
        cross-process reconciliation still requires a persistent lease fence.
        """
        raise NotImplementedError(
            f"{self.backend_name} does not support strict handoff destruction"
        )

    @abc.abstractmethod
    def container_exists(self, name: str) -> bool:
        ...

    @abc.abstractmethod
    def remove_container_by_name(self, name: str) -> None:
        ...

    # --- Image ---

    @abc.abstractmethod
    def image_exists(self, image_key: str) -> bool:
        ...

    def resolve_image_for_execution(self, image_key: str) -> ResolvedImage:
        """Resolve a mutable image name to an immutable execution reference.

        Replay provenance is only sound when the identity written to the plan
        is also the reference used to create every Judge container.  This is
        intentionally non-abstract for compatibility with third-party
        backends, but the default fails closed rather than claiming a tag is
        immutable.
        """

        raise NotImplementedError(
            f"{self.backend_name} cannot pin an immutable image for execution"
        )

    def verify_resolved_image_for_execution(
        self,
        resolved: ResolvedImage,
    ) -> None:
        """Fail unless a previously pinned immutable image is still executable."""

        raise NotImplementedError(
            f"{self.backend_name} cannot verify a pinned image for execution"
        )

    # --- File transfer ---

    @abc.abstractmethod
    def copy_to_container(
        self, handle: ContainerHandle, src: Path, dst: PurePosixPath,
    ) -> None:
        ...

    @abc.abstractmethod
    def copy_from_container(
        self, handle: ContainerHandle, src: PurePosixPath,
    ) -> bytes:
        ...

    @abc.abstractmethod
    def write_to_container(
        self, handle: ContainerHandle, data: str, dst: PurePosixPath,
    ) -> None:
        ...

    # --- Exec ---

    @abc.abstractmethod
    def exec_run(
        self,
        handle: ContainerHandle,
        cmd: str | list[str],
        *,
        user: str | None = None,
        workdir: str | None = None,
        environment: dict[str, str] | None = None,
        detach: bool = False,
    ) -> ExecResult:
        ...

    @abc.abstractmethod
    def exec_run_with_timeout(
        self,
        handle: ContainerHandle,
        cmd: str | list[str],
        timeout: int | None = 60,
        *,
        log_file: Path | None = None,
        user: str | None = None,
        workdir: str | None = None,
        environment: dict[str, str] | None = None,
        stream_to_stdout: bool = False,
        shutdown_event: threading.Event | None = None,
        log_append: bool = False,
        on_chunk: Callable[[bytes], None] | None = None,
        stop_container_on_timeout: bool = False,
        before_stop_on_timeout: Callable[[], bool | None] | None = None,
    ) -> StreamingExecResult:
        ...

    @abc.abstractmethod
    def exec_run_with_exit_code(
        self,
        handle: ContainerHandle,
        cmd: str | list[str],
        timeout: int | None = 60,
        *,
        user: str | None = None,
        workdir: str | None = None,
        environment: dict[str, str] | None = None,
    ) -> StreamingExecResult:
        ...

    # --- Inspection ---

    @abc.abstractmethod
    def get_container_ip(self, handle: ContainerHandle) -> str:
        ...

    @abc.abstractmethod
    def get_container_gateway_ip(self, handle: ContainerHandle) -> str | None:
        ...

    # --- Network isolation ---

    @abc.abstractmethod
    def create_network_isolation(
        self,
        handle: ContainerHandle,
        allowed_endpoints: list,
        logger: logging.Logger,
    ) -> NetworkIsolationStrategy:
        ...


class NetworkIsolationStrategy(abc.ABC):
    """Abstract network isolation that can be applied and cleaned up."""

    @abc.abstractmethod
    def apply(self) -> None:
        ...

    @abc.abstractmethod
    def cleanup(self) -> None:
        ...
