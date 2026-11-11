"""Abstract container backend interface for SForge.

Provides ContainerBackend ABC with Docker and Kubernetes implementations.
Select via --backend flag or SFORGE_BACKEND env var.
"""

from sforge.harness.backend.base import (
    AcceleratorAllocation,
    ContainerBackend,
    ContainerHandle,
    EVALUATION_CONTRACT_SHA256_ANNOTATION,
    ExecResult,
    HANDOFF_ALLOCATION_KEY_ANNOTATION,
    HANDOFF_BINDING_ANNOTATIONS,
    HANDOFF_DOCKER_PAUSE_PROTOCOL,
    HANDOFF_LEASE_ID_ANNOTATION,
    HANDOFF_PROTOCOL_ANNOTATION,
    HANDOFF_RUN_ID_ANNOTATION,
    NetworkIsolationStrategy,
    StreamingExecResult,
)
from sforge.harness.backend.factory import create_backend

__all__ = [
    "AcceleratorAllocation",
    "ContainerBackend",
    "ContainerHandle",
    "EVALUATION_CONTRACT_SHA256_ANNOTATION",
    "ExecResult",
    "HANDOFF_ALLOCATION_KEY_ANNOTATION",
    "HANDOFF_BINDING_ANNOTATIONS",
    "HANDOFF_DOCKER_PAUSE_PROTOCOL",
    "HANDOFF_LEASE_ID_ANNOTATION",
    "HANDOFF_PROTOCOL_ANNOTATION",
    "HANDOFF_RUN_ID_ANNOTATION",
    "NetworkIsolationStrategy",
    "StreamingExecResult",
    "create_backend",
]
