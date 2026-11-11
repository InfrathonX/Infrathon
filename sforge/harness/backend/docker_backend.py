"""Docker backend implementation wrapping docker-py SDK."""

from __future__ import annotations

import io
import copy
import logging
import os
import re
import signal
import shlex
import sys
import tarfile
import threading
import time
import traceback
from pathlib import Path, PurePosixPath
from typing import Callable

import docker
import docker.errors
from docker.models.containers import Container

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
    ResolvedImage,
    StreamingExecResult,
)
from sforge.harness.cpu_affinity import (
    ContainerCpuState,
    CpuAffinityPolicy,
    plan_cpu_binding,
)
from sforge.harness.host_topology import normalize_cpulist

_LOGGER = logging.getLogger(__name__)
HEREDOC_DELIMITER = "EOF_SFORGE_1399519320"
EXEC_TERM_GRACE_SECONDS = 10.0
EXEC_KILL_GRACE_SECONDS = 10.0
FINAL_SNAPSHOT_STOP_ATTEMPTS = 3
EXEC_PROCESS_GROUP_METADATA_TIMEOUT = 5.0
EXEC_PROCESS_GROUP_FILE_ENV = "SFORGE_EXEC_PROCESS_GROUP_FILE"
_CANONICAL_DOCKER_IMAGE_ID_RE = re.compile(r"sha256:[0-9a-f]{64}")
_CANONICAL_DOCKER_CONTAINER_ID_RE = re.compile(r"[0-9a-f]{64}", re.ASCII)
_DOCKER_LABEL_KEY_RE = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9_.:/-]{0,253}[A-Za-z0-9])?",
    re.ASCII,
)
_HANDOFF_BINDING_VALUE_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}",
    re.ASCII,
)
_HANDOFF_ALLOCATION_KEY_RE = re.compile(
    r"allocation-sha256:[0-9a-f]{64}",
    re.ASCII,
)
_EVALUATION_CONTRACT_SHA256_RE = re.compile(
    r"sha256:[0-9a-f]{64}",
    re.ASCII,
)


def _validated_docker_labels(
    annotations: dict[str, str],
) -> dict[str, str]:
    if not isinstance(annotations, dict):
        raise ValueError("Docker annotations must be a string mapping")
    labels: dict[str, str] = {}
    for key, value in annotations.items():
        if not isinstance(key, str) or not _DOCKER_LABEL_KEY_RE.fullmatch(key):
            raise ValueError(f"invalid Docker annotation key: {key!r}")
        if not isinstance(value, str):
            raise ValueError(
                f"Docker annotation {key!r} value must be a string"
            )
        if len(value) > 4096 or any(
            ord(character) < 32 or ord(character) == 127
            for character in value
        ):
            raise ValueError(f"unsafe Docker annotation value for {key!r}")
        labels[key] = value
    return labels


def _validated_handoff_binding_labels(
    annotations: dict[str, str],
) -> dict[str, str]:
    labels = _validated_docker_labels(annotations)
    missing = HANDOFF_BINDING_ANNOTATIONS.difference(labels)
    if missing:
        raise ValueError(
            "Docker exact container lookup is missing trusted handoff "
            f"annotations: {', '.join(sorted(missing))}"
        )
    for key in (
        HANDOFF_RUN_ID_ANNOTATION,
        HANDOFF_LEASE_ID_ANNOTATION,
    ):
        if not _HANDOFF_BINDING_VALUE_RE.fullmatch(labels[key]):
            raise ValueError(f"invalid trusted handoff annotation {key!r}")
    if not _HANDOFF_ALLOCATION_KEY_RE.fullmatch(
        labels[HANDOFF_ALLOCATION_KEY_ANNOTATION]
    ):
        raise ValueError(
            "Docker handoff allocation key must use "
            "'allocation-sha256:<64 lowercase hex>'"
        )
    if labels[HANDOFF_PROTOCOL_ANNOTATION] != HANDOFF_DOCKER_PAUSE_PROTOCOL:
        raise ValueError(
            "Docker handoff protocol annotation must be "
            f"{HANDOFF_DOCKER_PAUSE_PROTOCOL!r}"
        )
    if not _EVALUATION_CONTRACT_SHA256_RE.fullmatch(
        labels[EVALUATION_CONTRACT_SHA256_ANNOTATION]
    ):
        raise ValueError(
            "Docker evaluation contract annotation must use "
            "'sha256:<64 lowercase hex>'"
        )
    return labels


def _signal_exec_process_group(
    container: Container,
    process_group_id: int,
    signal_name: str,
) -> None:
    """Signal a setsid-backed Docker exec process group.

    Commands that are not process-group leaders retain the legacy single-PID
    fallback.  The formal Codex path is launched through ``setsid`` and
    therefore takes the process-group branch.
    """
    if process_group_id <= 0:
        return
    signal_name = signal_name.upper()
    if signal_name not in {"TERM", "KILL"}:
        raise ValueError(f"Unsupported signal: {signal_name}")
    container.exec_run(
        [
            "/bin/sh",
            "-c",
            (
                f"kill -{signal_name} -{process_group_id} 2>/dev/null "
                f"|| kill -{signal_name} {process_group_id} 2>/dev/null "
                "|| true"
            ),
        ],
    )


def _exec_process_group_alive(
    container: Container,
    process_group_id: int,
) -> bool:
    """Return whether any process still belongs to the exec's process group."""
    if process_group_id <= 0:
        return False
    result = container.exec_run(
        [
            "/bin/sh",
            "-c",
            (
                "if command -v ps >/dev/null 2>&1 "
                "&& command -v awk >/dev/null 2>&1; then "
                "ps -eo pgid=,stat= 2>/dev/null "
                f"| awk -v target={process_group_id} "
                "'$1 == target && $2 !~ /^Z/ { alive = 1 } "
                "END { exit(alive ? 0 : 1) }'; "
                "else "
                f"kill -0 -{process_group_id} 2>/dev/null; "
                "fi"
            ),
        ],
    )
    return result.exit_code == 0


def _wait_for_exec_process_group(
    container: Container,
    process_group_id: int,
    stream_thread: threading.Thread,
    timeout: float,
) -> bool:
    """Wait until both the process group and Docker exec stream are closed."""
    deadline = time.monotonic() + timeout
    while True:
        group_alive = _exec_process_group_alive(
            container, process_group_id
        )
        if not group_alive and not stream_thread.is_alive():
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        stream_thread.join(min(0.1, remaining))


def _validated_process_group_file(
    environment: dict[str, str] | None,
) -> str | None:
    if not environment:
        return None
    raw_path = environment.get(EXEC_PROCESS_GROUP_FILE_ENV)
    if not raw_path:
        return None
    path = PurePosixPath(raw_path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"{EXEC_PROCESS_GROUP_FILE_ENV} must be an absolute safe path"
        )
    return str(path)


def _read_process_group_id(
    container: Container,
    process_group_file: str | None,
    exec_pid: int,
) -> int:
    """Resolve the setsid child's PGID, falling back for legacy commands."""
    if process_group_file is None:
        return exec_pid

    quoted_path = shlex.quote(process_group_file)
    deadline = time.monotonic() + EXEC_PROCESS_GROUP_METADATA_TIMEOUT
    while True:
        result = container.exec_run(
            [
                "/bin/sh",
                "-c",
                f"cat {quoted_path} 2>/dev/null || true",
            ],
        )
        output = result.output
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        try:
            process_group_id = int((output or "").strip())
        except ValueError:
            process_group_id = 0
        if process_group_id > 1:
            return process_group_id
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "Timed out waiting for setsid process-group metadata"
            )
        time.sleep(0.05)


class DockerContainerHandle(ContainerHandle):
    """Wraps a docker Container object."""

    def __init__(self, container: Container) -> None:
        self._container = container
        self._snapshot_freeze_metadata: dict[str, object] | None = None
        self._handoff_pause_metadata: dict[str, object] | None = None
        self._handoff_destroy_metadata: dict[str, object] | None = None
        self._resolved_by_exact_lookup = False

    @property
    def id(self) -> str:
        return self._container.id

    @property
    def name(self) -> str:
        return self._container.name

    @property
    def ip_address(self) -> str | None:
        try:
            self._container.reload()
            networks = self._container.attrs.get("NetworkSettings", {}).get("Networks", {})
            for net_info in networks.values():
                ip = net_info.get("IPAddress", "")
                if ip:
                    return ip
        except Exception:
            pass
        return None

    @property
    def raw(self) -> Container:
        return self._container


class DockerNetworkIsolation(NetworkIsolationStrategy):
    """Delegates to the existing NetworkIsolation class in network_isolation.py."""

    def __init__(self, isolation) -> None:
        self._isolation = isolation

    def apply(self) -> None:
        self._isolation.apply()

    def cleanup(self) -> None:
        self._isolation.cleanup()


class DockerBackend(ContainerBackend):
    """Container backend using local Docker daemon."""

    def __init__(self, client: docker.DockerClient | None = None) -> None:
        self._client = client or docker.from_env()

    @property
    def backend_name(self) -> str:
        return "docker"

    @property
    def client(self) -> docker.DockerClient:
        return self._client

    def authoritative_container_receipt(self, handle: ContainerHandle) -> dict:
        """Return live Docker inspect data for this exact, already-started ID."""
        container = self._raw(handle)
        container.reload()
        receipt = copy.deepcopy(container.attrs)
        if (receipt.get("Id") != handle.id
                or _CANONICAL_DOCKER_CONTAINER_ID_RE.fullmatch(handle.id) is None
                or _CANONICAL_DOCKER_IMAGE_ID_RE.fullmatch(str(receipt.get("Image"))) is None
                or receipt.get("State", {}).get("Running") is not True):
            raise RuntimeError("Docker live identity receipt differs from the exact running handle")
        return receipt

    # --- Lifecycle ---

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
        mounts: list[tuple[str, str, str]] | None = None,
        accelerator_allocation: AcceleratorAllocation | None = None,
    ) -> DockerContainerHandle:
        kwargs: dict = {}
        if cpu_limit is not None:
            kwargs["nano_cpus"] = int(cpu_limit * 1e9)
        if mem_limit is not None:
            kwargs["mem_limit"] = mem_limit
        if cap_drop:
            kwargs["cap_drop"] = cap_drop
        if network_mode is not None:
            if network_mode != "none":
                raise ValueError("Docker Judge network mode must be 'none'")
            if extra_hosts:
                raise ValueError(
                    "Docker network mode none cannot include extra hosts"
                )
            kwargs["network_mode"] = "none"
        if extra_hosts:
            kwargs["extra_hosts"] = extra_hosts
        if user:
            kwargs["user"] = user
        if annotations is not None:
            labels = _validated_docker_labels(annotations)
            if HANDOFF_BINDING_ANNOTATIONS.intersection(labels):
                labels = _validated_handoff_binding_labels(labels)
            if labels:
                kwargs["labels"] = labels
        if mounts:
            # (host_path, container_path, mode) triples; mode "ro" or "rw".
            kwargs["volumes"] = {
                host_path: {"bind": container_path, "mode": mode}
                for host_path, container_path, mode in mounts
            }
        if accelerator_allocation is not None:
            if accelerator_allocation.vendor != "nvidia":
                raise ValueError(
                    "Docker backend supports only NVIDIA accelerator "
                    f"allocations, not {accelerator_allocation.vendor!r}"
                )
            kwargs["device_requests"] = [
                docker.types.DeviceRequest(
                    driver="nvidia",
                    device_ids=list(accelerator_allocation.device_ids),
                    capabilities=[["gpu"]],
                )
            ]

        container = self._client.containers.create(
            image,
            name=name,
            detach=True,
            command=command,
            environment=environment or {},
            **kwargs,
        )
        return DockerContainerHandle(container)

    def get_container_handle(
        self,
        container_id: str,
        *,
        expected_annotations: dict[str, str],
    ) -> DockerContainerHandle:
        """Resolve an exact ID and authenticate its immutable binding labels.

        A fresh handle intentionally has no handoff or finalization evidence.
        Callers must persistently fence the lease/final state across process
        boundaries; matching labels alone never authorize resuming an exited
        container.
        """
        if (
            not isinstance(container_id, str)
            or _CANONICAL_DOCKER_CONTAINER_ID_RE.fullmatch(container_id) is None
        ):
            raise ValueError(
                "Docker container_id must be a canonical 64-hex exact ID"
            )
        expected_labels = _validated_handoff_binding_labels(
            expected_annotations
        )
        try:
            container = self._client.containers.get(container_id)
        except docker.errors.NotFound as exc:
            raise RuntimeError(
                f"Docker handoff container {container_id} no longer exists"
            ) from exc

        actual_id = self._handoff_container_id(container)
        if actual_id != container_id:
            raise RuntimeError(
                "Docker exact container lookup returned a different identity "
                f"(requested={container_id}, actual={actual_id})"
            )

        self._inspect_handoff_container(container, container_id)
        attrs = getattr(container, "attrs", None)
        config = attrs.get("Config") if isinstance(attrs, dict) else None
        actual_labels = (
            config.get("Labels") if isinstance(config, dict) else None
        )
        if not isinstance(actual_labels, dict):
            actual_labels = {}
        mismatched = {
            key: {"expected": value, "actual": actual_labels.get(key)}
            for key, value in expected_labels.items()
            if actual_labels.get(key) != value
        }
        if mismatched:
            keys = ", ".join(sorted(mismatched))
            raise RuntimeError(
                "Docker container binding annotation mismatch for "
                f"{container_id}: {keys}"
            )
        handle = DockerContainerHandle(container)
        handle._resolved_by_exact_lookup = True
        return handle

    def start_container(self, handle: ContainerHandle) -> None:
        container = self._raw(handle)
        docker_handle = handle
        assert isinstance(docker_handle, DockerContainerHandle)
        if docker_handle._snapshot_freeze_metadata is not None:
            raise RuntimeError(
                "Docker container was irreversibly frozen for final snapshot; "
                "start is forbidden on this handle"
            )
        if docker_handle._resolved_by_exact_lookup:
            raise RuntimeError(
                "Docker exact re-lookup cannot authorize container start; "
                "a persistent lease tombstone/fence is required"
            )
        container.start()

    def apply_cpu_binding(
        self,
        handle: ContainerHandle,
        policy: CpuAffinityPolicy,
    ) -> dict[str, object]:
        """Write a cpuset into a created container's HostConfig before start.

        The GPUs, CPU quota and any task-owned cpuset are read back from the
        created container, so adapters that replace ``create_container`` are
        covered unchanged.  The update is re-read and must match exactly.
        """
        container = self._raw(handle)
        docker_handle = handle
        assert isinstance(docker_handle, DockerContainerHandle)
        if docker_handle._snapshot_freeze_metadata is not None:
            raise RuntimeError(
                "Docker container was frozen for final snapshot; CPU pinning "
                "is forbidden on this handle"
            )
        if docker_handle._resolved_by_exact_lookup:
            raise RuntimeError(
                "Docker exact re-lookup cannot authorize CPU pinning; pinning "
                "is applied only between create and start"
            )
        state = self._created_container_cpu_state(container)
        plan = plan_cpu_binding(policy, state)
        if plan.requires_update:
            update: dict[str, str] = {"cpuset_cpus": plan.cpuset_cpus or ""}
            if plan.cpuset_mems is not None:
                update["cpuset_mems"] = plan.cpuset_mems
            container.update(**update)
            applied = self._created_container_cpu_state(container)
            expected_mems = (
                plan.cpuset_mems
                if plan.cpuset_mems is not None
                else normalize_cpulist(state.cpuset_mems)
            )
            observed = (
                normalize_cpulist(applied.cpuset_cpus),
                normalize_cpulist(applied.cpuset_mems),
                applied.nano_cpus,
            )
            if observed != (plan.cpuset_cpus, expected_mems, state.nano_cpus):
                raise RuntimeError(
                    "Docker did not persist the requested CPU pinning on "
                    f"{container.id} (requested cpus={plan.cpuset_cpus} "
                    f"mems={expected_mems or '-'} nano_cpus={state.nano_cpus}; "
                    f"observed cpus={observed[0] or '-'} "
                    f"mems={observed[1] or '-'} nano_cpus={observed[2]})"
                )
        return plan.record()

    @staticmethod
    def _created_container_cpu_state(container: Container) -> ContainerCpuState:
        container.reload()
        attrs = getattr(container, "attrs", None)
        attrs = attrs if isinstance(attrs, dict) else {}
        state = attrs.get("State") or {}
        status = str(state.get("Status") or getattr(container, "status", ""))
        if (
            status != "created"
            or state.get("Running")
            or int(state.get("Pid") or 0) != 0
        ):
            raise RuntimeError(
                "CPU pinning is applied only to a created, not yet started "
                f"container; {container.id} is {status or 'unknown'}"
            )
        host_config = attrs.get("HostConfig") or {}
        request_ids: list[str] = []
        request_unspecified = False
        unrecognized: list[str] = []
        for request in host_config.get("DeviceRequests") or []:
            capabilities = request.get("Capabilities") or []
            device_ids = request.get("DeviceIDs") or []
            if request.get("Driver") != "nvidia" and not any(
                "gpu" in capability_set for capability_set in capabilities
            ):
                # Not recognizably a GPU request; one that still names
                # devices (e.g. a CDI nvidia.com/gpu=N) must not be skipped.
                if device_ids or int(request.get("Count") or 0) != 0:
                    unrecognized.append(
                        f"driver={request.get('Driver') or '-'} "
                        f"devices={','.join(map(str, device_ids)) or '-'} "
                        f"count={int(request.get('Count') or 0)}"
                    )
                continue
            if device_ids:
                request_ids.extend(str(device_id) for device_id in device_ids)
            elif int(request.get("Count") or 0) != 0:
                request_unspecified = True
        visible_devices = None
        for item in (attrs.get("Config") or {}).get("Env") or []:
            key, separator, value = str(item).partition("=")
            if separator and key == "NVIDIA_VISIBLE_DEVICES":
                visible_devices = value
        return ContainerCpuState(
            gpu_request_ids=tuple(request_ids),
            gpu_request_unspecified=request_unspecified,
            unrecognized_device_requests=tuple(unrecognized),
            visible_devices=visible_devices,
            nano_cpus=int(host_config.get("NanoCpus") or 0),
            cpu_quota=int(host_config.get("CpuQuota") or 0),
            cpu_period=int(host_config.get("CpuPeriod") or 0),
            cpuset_cpus=str(host_config.get("CpusetCpus") or ""),
            cpuset_mems=str(host_config.get("CpusetMems") or ""),
        )

    def cleanup_container(
        self, handle: ContainerHandle | None, logger: logging.Logger | None = None,
    ) -> None:
        if handle is None:
            return

        container = self._raw(handle)
        old_handler = None
        try:
            old_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        except ValueError:
            pass

        log_info = logger.info if logger else print
        log_error = logger.error if logger else print

        try:
            container.reload()
            state = container.attrs.get("State", {})
            still_running = bool(
                state.get("Running")
                or state.get("Restarting")
                or state.get("Paused")
                or int(state.get("Pid") or 0) > 0
            )
            if still_running:
                log_info(f"Stopping container {container.name}...")
                container.stop(timeout=15)
            else:
                log_info(f"Container {container.name} is already stopped.")
        except docker.errors.NotFound:
            log_info(f"Container {container.name} is already removed.")
            if old_handler is not None:
                try:
                    signal.signal(signal.SIGINT, old_handler)
                except ValueError:
                    pass
            return
        except Exception as e:
            log_error(f"Failed to stop container {container.name}: {e}")
            try:
                container_info = self._client.api.inspect_container(container.id)
                pid = container_info["State"].get("Pid", 0)
                if pid > 0:
                    log_info(f"Forcefully killing container PID {pid}...")
                    os.kill(pid, signal.SIGKILL)
            except Exception as e2:
                log_error(f"Failed to kill container {container.name}: {e2}")

        try:
            log_info(f"Removing container {container.name}...")
            container.remove(force=True)
            log_info(f"Container {container.name} removed.")
        except Exception as e:
            log_error(
                f"Failed to remove container {container.name}: {e}\n"
                f"{traceback.format_exc()}"
            )

        if old_handler is not None:
            try:
                signal.signal(signal.SIGINT, old_handler)
            except ValueError:
                pass

    def freeze_container_for_snapshot(
        self,
        handle: ContainerHandle,
        *,
        stop_without_inspect: bool = False,
    ) -> dict[str, object]:
        """Synchronously stop the whole Work container for final capture.

        ``timeout=0`` asks Docker to avoid a TERM grace period.  This matters
        at an evaluation deadline: a detached/forked writer must not receive
        extra time merely because it escaped the agent exec process group.

        A container reversibly paused for transient handoff is rejected.  The
        separate final-handoff API may adopt that pause only after a trusted
        coordinator has persisted its no-resume tombstone.
        """
        container = self._raw(handle)
        docker_handle = handle
        assert isinstance(docker_handle, DockerContainerHandle)

        existing = docker_handle._snapshot_freeze_metadata
        if existing is not None:
            self.verify_container_frozen_for_snapshot(handle)
            return dict(existing)
        if docker_handle._handoff_pause_metadata is not None:
            raise RuntimeError(
                "Docker container is paused for transient handoff; final "
                "snapshot capture requires a persistent no-resume tombstone "
                "and explicit retirement protocol"
            )

        requested_monotonic = time.monotonic()
        requested_at = time.time()
        if stop_without_inspect:
            # The candidate deadline path already knows its exec is live.  Do
            # not spend an inspect round trip while detached writers can run.
            was_running = True
        else:
            container.reload()
            state = container.attrs.get("State", {})
            if bool(state.get("Paused")):
                raise RuntimeError(
                    "Docker container is paused for transient handoff; final "
                    "snapshot freeze refuses to unpause it implicitly"
                )
            was_running = bool(
                state.get("Running")
                or state.get("Restarting")
                or state.get("Paused")
                or int(state.get("Pid") or 0) > 0
            )
        stop_attempts = 0
        if was_running:
            for stop_attempts in range(1, FINAL_SNAPSHOT_STOP_ATTEMPTS + 1):
                stop_error: Exception | None = None
                try:
                    container.stop(timeout=0)
                except Exception as exc:
                    # A concurrent natural exit can make stop report an error.
                    # The state verification below is authoritative.
                    stop_error = exc
                try:
                    verified = self.verify_container_frozen_for_snapshot(handle)
                except Exception as verify_error:
                    state = container.attrs.get("State", {})
                    # Docker can lose the first kill's exit event while its
                    # inspect state still reports a running container. Repeat
                    # only that force-stop failure, without a TERM grace or
                    # removal: the filesystem must survive for final capture.
                    # Paused/restarting and ambiguous inspect states still fail
                    # closed; an error alone never proves an immutable boundary.
                    if (
                        stop_attempts < FINAL_SNAPSHOT_STOP_ATTEMPTS
                        and isinstance(stop_error, docker.errors.APIError)
                        and stop_error.status_code == 500
                        and "did not receive an exit event" in str(stop_error)
                        and isinstance(verify_error, RuntimeError)
                        and str(verify_error).startswith(
                            "Work container is not fully stopped for final snapshot"
                        )
                        and not state.get("Paused")
                        and not state.get("Restarting")
                    ):
                        continue
                    if stop_error is not None:
                        raise RuntimeError(
                            "Docker whole-container stop failed before final snapshot"
                        ) from stop_error
                    raise
                if stop_error is not None:
                    # A force-stop racing a natural exit is still safe, but no
                    # running state was observed by the successful verification.
                    was_running = False
                break
        else:
            verified = self.verify_container_frozen_for_snapshot(handle)

        metadata: dict[str, object] = {
            "mechanism": (
                "docker_stop_timeout_0"
                if was_running
                else "docker_verified_stopped"
            ),
            "all_work_processes_stopped": True,
            "was_running": was_running,
            "stop_attempts": stop_attempts,
            "freeze_requested_at": requested_at,
            "freeze_requested_monotonic": requested_monotonic,
            "frozen_at": time.time(),
            "frozen_at_monotonic": time.monotonic(),
            "container_status": verified["container_status"],
            "container_pid": verified["container_pid"],
        }
        docker_handle._snapshot_freeze_metadata = dict(metadata)
        return metadata

    def freeze_container_for_final_handoff_snapshot(
        self,
        handle: ContainerHandle,
        *,
        expected_annotations: dict[str, str],
        stop_without_inspect: bool = False,
    ) -> dict[str, object]:
        """Adopt a durably fenced handoff pause as the final file boundary.

        This is deliberately separate from ``freeze_container_for_snapshot``:
        an ordinary transient handoff pause remains resumable and must never be
        accepted as a final boundary.  The trusted coordinator calls this only
        after persisting ``never_resume`` and ``FINALIZING``.  Docker still
        independently verifies the exact full container ID, all binding labels,
        and the paused state.  No unpause operation is issued.
        """

        exact = self.get_container_handle(
            handle.id,
            expected_annotations=expected_annotations,
        )
        if exact.id != handle.id:
            raise RuntimeError(
                "final handoff snapshot resolved a different Work container"
            )

        container = self._raw(exact)
        state = self._inspect_handoff_container(container, exact.id)
        if not state["paused"]:
            is_stopped = bool(
                not state["running"]
                and not state["restarting"]
                and state["container_pid"] == 0
                and state["container_status"] in {"created", "exited"}
            )
            if not self._handoff_state_is_running(state) and not is_stopped:
                raise self._ambiguous_handoff_state_error(
                    exact.id,
                    state,
                    "before final snapshot freeze",
                )

            # A final fence either won while Work was active, or a different
            # trusted handle already stopped it before archiving.  Operate on
            # the freshly exact-ID/label-authenticated handle.  A known stopped
            # state does not need another stop; a running state preserves the
            # caller's immediate-stop option.
            metadata = self.freeze_container_for_snapshot(
                exact,
                stop_without_inspect=(stop_without_inspect and not is_stopped),
            )
            docker_handle = handle
            assert isinstance(docker_handle, DockerContainerHandle)
            docker_handle._snapshot_freeze_metadata = dict(metadata)
            return metadata

        # Only the paused adoption branch applies transient-handoff health
        # requirements.  Restarting or otherwise inconsistent paused states
        # are rejected by the strict verifier and are never unpaused here.
        verified = self._verify_handoff_paused_state(container, exact.id)
        requested_at = time.time()
        requested_monotonic = time.monotonic()
        metadata: dict[str, object] = {
            "mechanism": "docker_pause_never_resume_final",
            "all_work_processes_frozen": True,
            "all_work_processes_stopped": False,
            "was_running": True,
            "freeze_requested_at": requested_at,
            "freeze_requested_monotonic": requested_monotonic,
            "frozen_at": requested_at,
            "frozen_at_monotonic": requested_monotonic,
            "container_status": verified["container_status"],
            "container_pid": verified["container_pid"],
            "container_paused": True,
            "container_id": exact.id,
            "archive_source": "paused_work_container_filesystem",
            "archive_transport": "docker_get_archive_transport_tar_while_paused",
        }
        docker_handle = handle
        assert isinstance(docker_handle, DockerContainerHandle)
        docker_handle._snapshot_freeze_metadata = dict(metadata)
        return metadata

    def verify_container_frozen_for_snapshot(
        self,
        handle: ContainerHandle,
    ) -> dict[str, object]:
        """Prove that no process can write the captured container layer."""
        container = self._raw(handle)
        docker_handle = handle
        assert isinstance(docker_handle, DockerContainerHandle)
        freeze_metadata = docker_handle._snapshot_freeze_metadata
        if (
            freeze_metadata is not None
            and freeze_metadata.get("mechanism")
            == "docker_pause_never_resume_final"
        ):
            container_id = self._handoff_container_id(container)
            if freeze_metadata.get("container_id") != container_id:
                raise RuntimeError(
                    "paused final-snapshot evidence has a different container ID"
                )
            verified = self._verify_handoff_paused_state(
                container,
                container_id,
            )
            return {
                "all_work_processes_frozen": True,
                "all_work_processes_stopped": False,
                "container_paused": True,
                "container_status": verified["container_status"],
                "container_pid": verified["container_pid"],
                "container_id": container_id,
            }
        try:
            container.reload()
        except docker.errors.NotFound:
            return {
                "all_work_processes_stopped": True,
                "container_status": "removed",
                "container_pid": 0,
            }
        state = container.attrs.get("State", {})
        running = bool(state.get("Running"))
        restarting = bool(state.get("Restarting"))
        paused = bool(state.get("Paused"))
        pid = int(state.get("Pid") or 0)
        status = str(
            state.get("Status") or getattr(container, "status", "unknown")
        )
        if running or restarting or paused or pid != 0:
            raise RuntimeError(
                "Work container is not fully stopped for final snapshot "
                f"(status={status}, running={running}, "
                f"restarting={restarting}, paused={paused}, pid={pid})"
            )
        return {
            "all_work_processes_stopped": True,
            "container_status": status,
            "container_pid": pid,
        }

    def pause_container_for_handoff(
        self,
        handle: ContainerHandle,
    ) -> dict[str, object]:
        """Pause all processes while preserving every live Docker exec.

        This reports Docker freezer state only.  It does not attest that an
        accelerator is idle or released; that belongs to an external observer.
        """
        container = self._raw(handle)
        docker_handle = handle
        assert isinstance(docker_handle, DockerContainerHandle)
        self._ensure_transient_handoff_allowed(docker_handle)
        container_id = self._handoff_container_id(container)
        before = self._inspect_handoff_container(container, container_id)
        self._verify_healthy_handoff_state(before, container_id)

        if self._handoff_state_is_paused(before):
            self._require_same_handle_pause_evidence(docker_handle, container_id)
            verified = self.verify_container_paused_for_handoff(handle)
            return {
                **verified,
                "mechanism": "docker_verified_paused",
                "was_paused": True,
            }
        if not self._handoff_state_is_running(before):
            raise self._ambiguous_handoff_state_error(
                container_id,
                before,
                "before handoff pause",
            )

        pause_error: Exception | None = None
        try:
            container.pause()
        except docker.errors.NotFound as exc:
            raise RuntimeError(
                f"Docker handoff container {container_id} no longer exists"
            ) from exc
        except Exception as exc:
            pause_error = exc

        try:
            verified = self._verify_handoff_paused_state(
                container,
                container_id,
            )
        except Exception:
            if pause_error is not None:
                raise RuntimeError(
                    "Docker container pause failed during transient handoff"
                ) from pause_error
            raise

        metadata = {
            **verified,
            "mechanism": "docker_pause",
            "pause_observed": True,
            "was_paused": False,
        }
        docker_handle._handoff_pause_metadata = dict(metadata)
        return metadata

    def verify_container_paused_for_handoff(
        self,
        handle: ContainerHandle,
    ) -> dict[str, object]:
        """Prove same-handle Docker freezer pause, not accelerator release."""
        container = self._raw(handle)
        docker_handle = handle
        assert isinstance(docker_handle, DockerContainerHandle)
        self._ensure_transient_handoff_allowed(docker_handle)
        container_id = self._handoff_container_id(container)
        self._require_same_handle_pause_evidence(docker_handle, container_id)
        return self._verify_handoff_paused_state(container, container_id)

    def unpause_container_after_handoff(
        self,
        handle: ContainerHandle,
    ) -> dict[str, object]:
        """Unpause the exact container so its existing execs continue."""
        container = self._raw(handle)
        docker_handle = handle
        assert isinstance(docker_handle, DockerContainerHandle)
        self._ensure_transient_handoff_allowed(docker_handle)
        container_id = self._handoff_container_id(container)
        self._require_same_handle_pause_evidence(docker_handle, container_id)
        before = self._inspect_handoff_container(container, container_id)
        self._verify_healthy_handoff_state(before, container_id)

        was_paused = self._handoff_state_is_paused(before)
        if not was_paused and not self._handoff_state_is_running(before):
            raise self._ambiguous_handoff_state_error(
                container_id,
                before,
                "before handoff unpause",
            )

        unpause_error: Exception | None = None
        if was_paused:
            try:
                container.unpause()
            except docker.errors.NotFound as exc:
                raise RuntimeError(
                    f"Docker handoff container {container_id} no longer exists"
                ) from exc
            except Exception as exc:
                unpause_error = exc

        try:
            verified = self.verify_container_running_after_handoff(handle)
            self._verify_handoff_container_id(container_id, verified)
        except Exception:
            if unpause_error is not None:
                raise RuntimeError(
                    "Docker container unpause failed after transient handoff"
                ) from unpause_error
            raise

        metadata = {
            **verified,
            "mechanism": (
                "docker_unpause"
                if was_paused
                else "docker_verified_running"
            ),
            "pause_observed": True,
            "was_paused": was_paused,
        }
        # Retain same-handle evidence so a separate explicit verification or
        # idempotent retry is possible.  Fresh exact-ID lookup never restores
        # this evidence.
        docker_handle._handoff_pause_metadata = dict(metadata)
        return metadata

    def verify_container_running_after_handoff(
        self,
        handle: ContainerHandle,
    ) -> dict[str, object]:
        """Prove the same handoff container is cleanly running."""
        container = self._raw(handle)
        docker_handle = handle
        assert isinstance(docker_handle, DockerContainerHandle)
        self._ensure_transient_handoff_allowed(docker_handle)
        self._require_same_handle_pause_evidence(
            docker_handle,
            self._handoff_container_id(container),
        )
        container_id = self._handoff_container_id(container)
        state = self._inspect_handoff_container(container, container_id)
        self._verify_healthy_handoff_state(state, container_id)
        if not self._handoff_state_is_running(state):
            raise RuntimeError(
                "Docker container is not running after transient handoff "
                f"(id={container_id}, status={state['container_status']}, "
                f"running={state['running']}, "
                f"restarting={state['restarting']}, "
                f"paused={state['paused']}, pid={state['container_pid']})"
            )
        return {
            "container_id": container_id,
            "container_running": True,
            "container_status": state["container_status"],
            "container_pid": state["container_pid"],
            "container_exit_code": state["container_exit_code"],
            "container_oom_killed": state["oom_killed"],
        }

    def destroy_container_for_handoff(
        self,
        handle: ContainerHandle,
    ) -> dict[str, object]:
        """Stop, remove, and prove absence of the exact Docker container.

        Successful absence does not attest that NVIDIA/NVML reports the
        accelerator idle; the coordinator must verify that independently.
        Idempotent evidence lives only on this handle.  A fresh process must
        reconcile destruction through its persistent lease tombstone.
        """
        container = self._raw(handle)
        docker_handle = handle
        assert isinstance(docker_handle, DockerContainerHandle)
        container_id = self._handoff_container_id(container)
        if _CANONICAL_DOCKER_CONTAINER_ID_RE.fullmatch(container_id) is None:
            raise RuntimeError(
                "Docker strict handoff destruction requires a canonical "
                "64-hex container ID"
            )
        existing = docker_handle._handoff_destroy_metadata
        if existing is not None:
            if existing.get("container_id") != container_id:
                raise RuntimeError(
                    "Docker handoff destruction evidence has a different "
                    "container identity"
                )
            self._verify_handoff_container_absent(container_id)
            return dict(existing)
        try:
            self._inspect_handoff_container(container, container_id)
        except Exception as inspect_error:
            # A create/start failure may race an external removal.  Exact-ID
            # absence is sufficient destruction evidence, but an ambiguous
            # inspect while the ID still resolves remains a hard failure.
            try:
                self._verify_handoff_container_absent(container_id)
            except Exception as absence_error:
                raise RuntimeError(
                    "Docker handoff container could not be inspected or "
                    f"proved absent (inspect={inspect_error}; "
                    f"absence={absence_error})"
                ) from absence_error
            metadata = {
                "mechanism": "docker_already_absent_verified",
                "container_id": container_id,
                "container_destroyed": True,
            }
            docker_handle._handoff_destroy_metadata = dict(metadata)
            return metadata

        # Deliberately do not reuse cleanup_container: handoff retirement must
        # never swallow an error or fall back to host PID signalling.
        errors: list[Exception] = []
        try:
            container.stop(timeout=0)
        except Exception as exc:
            # Docker may reject stop for a merely-created or already-stopped
            # container.  force-remove is still required and safe for the
            # exact handle, so a stop error must never skip it.
            errors.append(exc)
        identity_matches = self._handoff_container_id(container) == container_id
        if not identity_matches:
            errors.append(
                RuntimeError(
                    "Docker handoff container identity changed before removal"
                )
            )
        else:
            try:
                container.remove(force=True)
            except docker.errors.NotFound:
                pass
            except Exception as exc:
                errors.append(exc)
        try:
            self._verify_handoff_container_absent(container_id)
        except Exception as exc:
            errors.append(exc)
            details = "; ".join(
                f"{type(error).__name__}: {error}" for error in errors
            )
            raise RuntimeError(
                "Docker strict handoff destruction did not prove exact-ID "
                f"absence ({details})"
            ) from exc
        metadata = {
            "mechanism": "docker_stop_remove_verified_absent",
            "container_id": container_id,
            "container_destroyed": True,
        }
        docker_handle._handoff_destroy_metadata = dict(metadata)
        return metadata

    def container_exists(self, name: str) -> bool:
        try:
            self._client.containers.get(name)
            return True
        except docker.errors.NotFound:
            return False

    def remove_container_by_name(self, name: str) -> None:
        try:
            c = self._client.containers.get(name)
            c.remove(force=True)
        except docker.errors.NotFound:
            pass

    # --- Image ---

    def image_exists(self, image_key: str) -> bool:
        try:
            self._client.images.get(image_key)
            return True
        except docker.errors.ImageNotFound:
            return False

    def resolve_image_for_execution(self, image_key: str) -> ResolvedImage:
        """Pin a local tag/name to Docker's full content-addressed image ID."""

        try:
            image = self._client.images.get(image_key)
        except docker.errors.ImageNotFound as exc:
            raise RuntimeError(
                f"Judge image does not exist: {image_key}"
            ) from exc
        image_id = getattr(image, "id", None)
        if (
            not isinstance(image_id, str)
            or _CANONICAL_DOCKER_IMAGE_ID_RE.fullmatch(image_id) is None
        ):
            raise RuntimeError(
                "Docker did not return a canonical immutable Docker image ID"
            )
        return ResolvedImage(
            backend="docker",
            requested_ref=image_key,
            execution_ref=image_id,
            content_identity=image_id,
            identity_kind="docker-image-id",
        )

    def verify_resolved_image_for_execution(
        self,
        resolved: ResolvedImage,
    ) -> None:
        if (
            resolved.backend != "docker"
            or resolved.identity_kind != "docker-image-id"
            or _CANONICAL_DOCKER_IMAGE_ID_RE.fullmatch(
                resolved.execution_ref
            ) is None
            or resolved.content_identity != resolved.execution_ref
        ):
            raise RuntimeError("pinned Docker image identity is invalid")
        try:
            image = self._client.images.get(resolved.execution_ref)
        except docker.errors.ImageNotFound as exc:
            raise RuntimeError("pinned Docker image is no longer available") from exc
        observed_id = getattr(image, "id", None)
        if observed_id != resolved.content_identity:
            raise RuntimeError("pinned Docker image identity no longer matches")

    # --- File transfer ---

    def copy_to_container(
        self, handle: ContainerHandle, src: Path, dst: PurePosixPath,
    ) -> None:
        container = self._raw(handle)
        if str(dst.parent) == ".":
            raise ValueError(f"Destination parent directory cannot be empty: {dst}")

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            tar.add(src, arcname=dst.name)
        buf.seek(0)

        container.exec_run(f"mkdir -p {dst.parent}")
        container.put_archive(str(dst.parent), buf.read())

    def copy_from_container(
        self, handle: ContainerHandle, src: PurePosixPath,
    ) -> bytes:
        container = self._raw(handle)
        bits, _ = container.get_archive(str(src))
        return b"".join(bits)

    def write_to_container(
        self, handle: ContainerHandle, data: str, dst: PurePosixPath,
    ) -> None:
        container = self._raw(handle)
        command = f"cat <<'{HEREDOC_DELIMITER}' > {dst}\n{data}\n{HEREDOC_DELIMITER}"
        container.exec_run(command)

    # --- Exec ---

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
        container = self._raw(handle)
        kwargs: dict = {}
        if user:
            kwargs["user"] = user
        if workdir:
            kwargs["workdir"] = workdir
        if environment:
            kwargs["environment"] = environment
        result = container.exec_run(cmd, detach=detach, **kwargs)
        if detach:
            return ExecResult(output="", exit_code=0)
        output = result.output.decode(errors="replace") if isinstance(result.output, bytes) else (result.output or "")
        return ExecResult(output=output, exit_code=result.exit_code)

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
        result = self._exec_run_impl(
            handle, cmd, timeout,
            log_file=log_file, user=user, workdir=workdir,
            environment=environment, stream_to_stdout=stream_to_stdout,
            shutdown_event=shutdown_event, log_append=log_append,
            on_chunk=on_chunk,
            stop_container_on_timeout=stop_container_on_timeout,
            before_stop_on_timeout=before_stop_on_timeout,
        )
        return StreamingExecResult(
            output=result[0], exit_code=result[1],
            timed_out=result[2], elapsed_seconds=result[3],
        )

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
        result = self._exec_run_impl(
            handle, cmd, timeout,
            user=user, workdir=workdir, environment=environment,
        )
        return StreamingExecResult(
            output=result[0], exit_code=result[1],
            timed_out=result[2], elapsed_seconds=result[3],
        )

    def _exec_run_impl(
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
    ) -> tuple[str, int, bool, float]:
        container = self._raw(handle)
        process_group_file = _validated_process_group_file(environment)
        chunks: list[bytes] = []
        exec_id = None
        exception = None
        timed_out = False
        log_fh = None
        exec_created = threading.Event()
        process_group_id: int | None = None

        try:
            if log_file:
                log_file.parent.mkdir(parents=True, exist_ok=True)
                log_fh = open(log_file, "ab" if log_append else "wb")
            if process_group_file is not None:
                quoted_path = shlex.quote(process_group_file)
                remove_result = container.exec_run(
                    ["/bin/sh", "-c", f"rm -f {quoted_path}"]
                )
                if remove_result.exit_code != 0:
                    raise RuntimeError(
                        "Failed to clear stale setsid process-group metadata"
                    )

            def run_command():
                nonlocal exec_id, exception
                try:
                    kwargs = {}
                    if user:
                        kwargs["user"] = user
                    if workdir:
                        kwargs["workdir"] = workdir
                    if environment:
                        kwargs["environment"] = environment
                    exec_id = container.client.api.exec_create(container.id, cmd, **kwargs)["Id"]
                    exec_created.set()
                    exec_stream = container.client.api.exec_start(exec_id, stream=True)
                    for chunk in exec_stream:
                        chunks.append(chunk)
                        if log_fh:
                            log_fh.write(chunk)
                            log_fh.flush()
                        if stream_to_stdout:
                            sys.stdout.buffer.write(chunk)
                            sys.stdout.buffer.flush()
                        if on_chunk is not None:
                            try:
                                on_chunk(chunk)
                            except Exception:
                                pass
                except Exception as e:
                    exception = e
                finally:
                    exec_created.set()

            # The deadline-bearing path can deliberately quarantine a paused
            # Work container when its finalizer fails.  A blocked Docker exec
            # reader must not then prevent the trusted host process from
            # returning the failure to its launcher/operator.
            thread = threading.Thread(
                target=run_command,
                daemon=stop_container_on_timeout,
            )
            start_time = time.monotonic()
            thread.start()
            if process_group_file is not None:
                # Capture the PGID immediately. The agent cannot redirect
                # timeout cleanup by modifying the in-container metadata later.
                exec_created.wait(timeout=1.0)
                process_group_id = _read_process_group_id(
                    container, process_group_file, 0
                )

            deadline = start_time + timeout if timeout else None
            while thread.is_alive():
                remaining = (deadline - time.monotonic()) if deadline else 1.0
                if remaining <= 0:
                    break
                thread.join(min(remaining, 1.0))
                if shutdown_event and shutdown_event.is_set():
                    break

            if thread.is_alive():
                timed_out = True
                exit_code = -1
                if stop_container_on_timeout:
                    # This opt-in path is reserved for run_agent's final
                    # submission boundary.  Stop the whole container first;
                    # do not grant the candidate a process-group TERM grace.
                    retired_by_callback = False
                    if before_stop_on_timeout is not None:
                        retired_by_callback = before_stop_on_timeout() is True
                    if not retired_by_callback:
                        self.freeze_container_for_snapshot(
                            handle,
                            stop_without_inspect=True,
                        )
                    thread.join(EXEC_KILL_GRACE_SECONDS)
                    if thread.is_alive():
                        raise RuntimeError(
                            "Docker exec stream did not close after the whole "
                            "container was stopped"
                        )
                else:
                    # A very short timeout/shutdown can race exec_create().
                    exec_created.wait(timeout=1.0)
                    if exec_id is not None:
                        if process_group_id is None:
                            exec_pid = int(
                                container.client.api.exec_inspect(exec_id)["Pid"]
                            )
                            process_group_id = _read_process_group_id(
                                container, None, exec_pid
                            )
                        _signal_exec_process_group(
                            container, process_group_id, "TERM"
                        )
                        stopped = _wait_for_exec_process_group(
                            container,
                            process_group_id,
                            thread,
                            EXEC_TERM_GRACE_SECONDS,
                        )
                        if not stopped:
                            _signal_exec_process_group(
                                container, process_group_id, "KILL"
                            )
                            stopped = _wait_for_exec_process_group(
                                container,
                                process_group_id,
                                thread,
                                EXEC_KILL_GRACE_SECONDS,
                            )
                        if not stopped:
                            raise RuntimeError(
                                "Docker exec process group did not stop after "
                                "TERM/KILL"
                            )
                    elif thread.is_alive():
                        raise RuntimeError(
                            "Docker exec did not start before its timeout"
                        )
            else:
                exit_code = (
                    container.client.api.exec_inspect(exec_id)["ExitCode"]
                    if exec_id else -1
                )

            if exception is not None and not timed_out:
                raise exception
        finally:
            if log_fh:
                log_fh.close()

        end_time = time.monotonic()
        output = b"".join(chunks).decode(errors="replace")
        return output, exit_code, timed_out, end_time - start_time

    # --- Inspection ---

    def get_container_ip(self, handle: ContainerHandle) -> str:
        container = self._raw(handle)
        container.reload()
        networks = container.attrs.get("NetworkSettings", {}).get("Networks", {})
        for net_info in networks.values():
            ip = net_info.get("IPAddress", "")
            if ip:
                return ip
        raise RuntimeError(f"Cannot determine IP for container {container.name}")

    def get_container_gateway_ip(self, handle: ContainerHandle) -> str | None:
        container = self._raw(handle)
        container.reload()
        networks = container.attrs.get("NetworkSettings", {}).get("Networks", {})
        for net_info in networks.values():
            gw = net_info.get("Gateway", "")
            if gw:
                return gw
        return None

    # --- Network isolation ---

    def create_network_isolation(
        self,
        handle: ContainerHandle,
        allowed_endpoints: list,
        logger: logging.Logger,
    ) -> NetworkIsolationStrategy:
        from sforge.harness.network_isolation import NetworkIsolation

        container = self._raw(handle)
        return DockerNetworkIsolation(NetworkIsolation(container, allowed_endpoints, logger))

    # --- Helpers ---

    @staticmethod
    def _handoff_container_id(container: Container) -> str:
        container_id = getattr(container, "id", None)
        if not isinstance(container_id, str) or not container_id:
            raise RuntimeError("Docker handoff container has no stable ID")
        return container_id

    @classmethod
    def _inspect_handoff_container(
        cls,
        container: Container,
        expected_id: str,
    ) -> dict[str, object]:
        try:
            container.reload()
        except docker.errors.NotFound as exc:
            raise RuntimeError(
                f"Docker handoff container {expected_id} no longer exists"
            ) from exc

        actual_id = cls._handoff_container_id(container)
        if actual_id != expected_id:
            raise RuntimeError(
                "Docker handoff container identity changed "
                f"from {expected_id} to {actual_id}"
            )
        attrs = getattr(container, "attrs", None)
        state = attrs.get("State") if isinstance(attrs, dict) else None
        if not isinstance(state, dict):
            raise RuntimeError(
                f"Docker handoff container {expected_id} has no State"
            )

        values: dict[str, bool] = {}
        for source_key, result_key in (
            ("Running", "running"),
            ("Restarting", "restarting"),
            ("Paused", "paused"),
            ("OOMKilled", "oom_killed"),
            ("Dead", "dead"),
        ):
            value = state.get(source_key)
            if not isinstance(value, bool):
                raise RuntimeError(
                    "Docker handoff container has an ambiguous State "
                    f"field {source_key}={value!r}"
                )
            values[result_key] = value

        pid = state.get("Pid")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid < 0:
            raise RuntimeError(
                "Docker handoff container has an ambiguous State "
                f"field Pid={pid!r}"
            )
        exit_code = state.get("ExitCode")
        if (
            isinstance(exit_code, bool)
            or not isinstance(exit_code, int)
            or exit_code < 0
        ):
            raise RuntimeError(
                "Docker handoff container has an ambiguous State "
                f"field ExitCode={exit_code!r}"
            )
        state_error = state.get("Error")
        if not isinstance(state_error, str):
            raise RuntimeError(
                "Docker handoff container has an ambiguous State "
                f"field Error={state_error!r}"
            )
        status = state.get("Status")
        if not isinstance(status, str) or not status:
            status = getattr(container, "status", None)
        if not isinstance(status, str) or not status:
            raise RuntimeError(
                "Docker handoff container has an ambiguous status"
            )
        return {
            **values,
            "container_status": status,
            "container_pid": pid,
            "container_exit_code": exit_code,
            "container_error": state_error,
        }

    @staticmethod
    def _ensure_transient_handoff_allowed(
        handle: DockerContainerHandle,
    ) -> None:
        if handle._snapshot_freeze_metadata is not None:
            raise RuntimeError(
                "Docker container was irreversibly frozen for final snapshot; "
                "transient handoff is forbidden on this handle"
            )

    @staticmethod
    def _verify_healthy_handoff_state(
        state: dict[str, object],
        container_id: str,
    ) -> None:
        # State.OOMKilled is sticky: Docker sets it once the kernel OOM-kills
        # any process in the container's memory cgroup, and never clears it
        # while the container lives.  An agent experiment that exceeds the
        # Work memory limit is killed on its own while the container (init
        # alive, exit code 0) keeps running, so the flag alone says nothing
        # about whether the container can be paused and resumed.  If init
        # itself was killed the container is no longer running/paused with
        # exit code 0, and the running/paused predicates fail closed.  The
        # flag is kept as evidence (container_oom_killed), not as a fence.
        if state["dead"] or state["container_error"]:
            raise RuntimeError(
                "Docker container has an unhealthy state for transient handoff "
                f"(id={container_id}, oom_killed={state['oom_killed']}, "
                f"dead={state['dead']}, error={state['container_error']!r}, "
                f"exit_code={state['container_exit_code']})"
            )
        if state["oom_killed"]:
            _LOGGER.warning(
                "Docker container %s recorded an OOM kill inside its memory "
                "cgroup; init is alive, transient handoff continues",
                container_id,
            )

    @staticmethod
    def _handoff_state_is_running(state: dict[str, object]) -> bool:
        return bool(
            state["running"]
            and not state["restarting"]
            and not state["paused"]
            and state["container_pid"] > 0
            and state["container_status"] == "running"
            and state["container_exit_code"] == 0
        )

    @staticmethod
    def _handoff_state_is_paused(state: dict[str, object]) -> bool:
        return bool(
            state["running"]
            and not state["restarting"]
            and state["paused"]
            and state["container_pid"] > 0
            and state["container_status"] == "paused"
            and state["container_exit_code"] == 0
        )

    @staticmethod
    def _ambiguous_handoff_state_error(
        container_id: str,
        state: dict[str, object],
        phase: str,
    ) -> RuntimeError:
        return RuntimeError(
            f"Docker container has an ambiguous state {phase} "
            f"(id={container_id}, status={state['container_status']}, "
            f"running={state['running']}, restarting={state['restarting']}, "
            f"paused={state['paused']}, pid={state['container_pid']}, "
            f"exit_code={state['container_exit_code']})"
        )

    @staticmethod
    def _require_same_handle_pause_evidence(
        handle: DockerContainerHandle,
        container_id: str,
    ) -> None:
        evidence = handle._handoff_pause_metadata
        if (
            not isinstance(evidence, dict)
            or evidence.get("container_id") != container_id
            or evidence.get("pause_observed") is not True
        ):
            raise RuntimeError(
                "Docker container lacks same-handle pause evidence for "
                f"transient handoff (id={container_id}); a fresh lookup cannot "
                "recreate it"
            )

    @classmethod
    def _verify_handoff_paused_state(
        cls,
        container: Container,
        container_id: str,
    ) -> dict[str, object]:
        state = cls._inspect_handoff_container(container, container_id)
        cls._verify_healthy_handoff_state(state, container_id)
        if not cls._handoff_state_is_paused(state):
            raise RuntimeError(
                "Docker container is not paused for transient handoff "
                f"(id={container_id}, status={state['container_status']}, "
                f"running={state['running']}, "
                f"restarting={state['restarting']}, "
                f"paused={state['paused']}, pid={state['container_pid']}, "
                f"exit_code={state['container_exit_code']})"
            )
        return {
            "container_id": container_id,
            "container_paused": True,
            "container_status": state["container_status"],
            "container_pid": state["container_pid"],
            "container_exit_code": state["container_exit_code"],
            "container_oom_killed": state["oom_killed"],
        }

    @staticmethod
    def _verify_handoff_container_id(
        expected_id: str,
        metadata: dict[str, object],
    ) -> None:
        actual_id = metadata.get("container_id")
        if actual_id != expected_id:
            raise RuntimeError(
                "Docker handoff container identity changed "
                f"from {expected_id} to {actual_id}"
            )

    def _verify_handoff_container_absent(self, container_id: str) -> None:
        try:
            remaining = self._client.containers.get(container_id)
        except docker.errors.NotFound:
            return
        remaining_id = getattr(remaining, "id", None)
        raise RuntimeError(
            "Docker handoff container removal was not verified "
            f"(requested={container_id}, resolved={remaining_id!r})"
        )

    @staticmethod
    def _raw(handle: ContainerHandle) -> Container:
        if isinstance(handle, DockerContainerHandle):
            return handle.raw
        raise TypeError(f"Expected DockerContainerHandle, got {type(handle).__name__}")
