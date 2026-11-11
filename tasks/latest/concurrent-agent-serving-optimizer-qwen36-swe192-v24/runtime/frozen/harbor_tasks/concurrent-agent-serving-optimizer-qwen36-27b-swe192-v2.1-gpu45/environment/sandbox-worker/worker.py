#!/usr/bin/env python3
"""Trusted no-network executor for fixture workspaces and SWE-bench images."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import re
import secrets
import signal
import socket
import stat
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import docker
from docker.errors import APIError, ImageNotFound, NotFound
from docker.models.containers import Container
from docker.types import Ulimit


SOCKET_PATH = Path(os.environ.get("SANDBOX_SOCKET", "/run/kvbench-sandbox/control.sock"))
WORKSPACE_ROOT = Path(os.environ.get("SANDBOX_WORKSPACE_ROOT", "/run/kvbench-workspaces"))
ADMIN_TOKEN_FILE = Path(
    os.environ.get("ADMIN_TOKEN_FILE", "/run/kvbench-admin/token")
)
MAX_COMMAND_BYTES = 64 * 1024
MAX_OUTPUT_BYTES = 2 * 1024 * 1024
SANDBOX_ID = re.compile(r"^[0-9a-f]{32}$")
IMAGE_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
OWNER_VALUE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
CHALLENGE = re.compile(r"^[0-9a-f]{64}$")
CONTAINER_ID = re.compile(r"^[0-9a-f]{64}$")
COMPOSE_PROJECT = re.compile(r"^[a-z0-9][a-z0-9_-]{0,127}$")
MANAGED_LABEL = "org.harbor-kv-scheduler.swe-sandbox"
OWNER_LABEL = f"{MANAGED_LABEL}.owner"
CONTROL_SOCKET_BACKLOG = 512
CLEANUP_ATTEMPTS = 3
CLEANUP_BACKOFF_SEC = 0.2
MAIN_EGRESS_SEAL_ACTION = "seal_main_egress"
MAIN_EGRESS_SEAL_CONTEXT = b"kvbench-main-egress-seal-v2\n"
COMPOSE_PROJECT_LABEL = "com.docker.compose.project"
COMPOSE_SERVICE_LABEL = "com.docker.compose.service"
COMPOSE_ONEOFF_LABEL = "com.docker.compose.oneoff"
COMPOSE_NETWORK_LABEL = "com.docker.compose.network"
SANDBOX_OWNER = os.environ.get("SANDBOX_OWNER", "")
if SANDBOX_OWNER and not OWNER_VALUE.fullmatch(SANDBOX_OWNER):
    raise ValueError("SANDBOX_OWNER has an invalid format")


class ThreadLocalDockerClient:
    """Give each Docker dispatch thread an independent requests session.

    docker-py's UNIX adapter keys connection pools by the full request URL and
    retains only 25 pools.  Sharing one client across the worker thread pool
    can therefore evict and close a pool while another thread is still using
    it.  A client per executor thread keeps every session single-threaded and
    makes that eviction harmless.
    """

    def __init__(self) -> None:
        self._local = threading.local()
        self._clients: list[docker.DockerClient] = []
        self._clients_lock = threading.Lock()

    def _client(self) -> docker.DockerClient:
        client = getattr(self._local, "client", None)
        if client is None:
            client = docker.from_env(timeout=180, max_pool_size=8)
            self._local.client = client
            with self._clients_lock:
                self._clients.append(client)
        return client

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client(), name)

    def close_all(self) -> None:
        with self._clients_lock:
            clients = self._clients
            self._clients = []
        for client in clients:
            client.close()


docker_client = ThreadLocalDockerClient()
runtimes: dict[str, str] = {}
owned_containers: dict[str, str] = {}
runtimes_lock = threading.RLock()
main_egress_seal_lock = threading.Lock()


def _admin_token() -> bytes:
    """Read the root-only HMAC key provisioned by the trusted model API."""

    metadata = ADMIN_TOKEN_FILE.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != 0
        or stat.S_IMODE(metadata.st_mode) != 0o400
    ):
        raise RuntimeError("network-seal token is not a root-owned 0400 regular file")
    token = ADMIN_TOKEN_FILE.read_bytes().strip()
    if not re.fullmatch(rb"[0-9a-f]{64}", token):
        raise RuntimeError("network-seal token has an invalid format")
    return token


def _container_labels(container: Container) -> dict[str, str]:
    labels = (container.attrs.get("Config") or {}).get("Labels") or {}
    if not isinstance(labels, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in labels.items()
    ):
        raise RuntimeError("container has malformed Docker labels")
    return labels


def _compose_identity(
    container: Container, *, expected_service: str
) -> tuple[str, str, str]:
    container.reload()
    container_id = str(container.id).lower()
    if not CONTAINER_ID.fullmatch(container_id):
        raise RuntimeError("container has an invalid immutable ID")
    image_id = str(container.attrs.get("Image") or "").lower()
    if not IMAGE_DIGEST.fullmatch(image_id):
        raise RuntimeError("container has an invalid immutable image ID")
    labels = _container_labels(container)
    project = labels.get(COMPOSE_PROJECT_LABEL, "")
    if not COMPOSE_PROJECT.fullmatch(project):
        raise RuntimeError("container has an invalid Compose project label")
    if labels.get(COMPOSE_SERVICE_LABEL) != expected_service:
        raise RuntimeError(
            f"container is not the expected Compose service: {expected_service}"
        )
    if labels.get(COMPOSE_ONEOFF_LABEL, "False").lower() != "false":
        raise RuntimeError("one-off Compose containers cannot control main egress")
    return project, container_id, image_id


def _network_record(network_id: str, *, target_container_id: str) -> dict[str, Any]:
    if not CONTAINER_ID.fullmatch(network_id):
        raise RuntimeError("main has an invalid Docker network ID")
    network = docker_client.networks.get(network_id)
    attrs = network.attrs
    observed_id = str(attrs.get("Id") or getattr(network, "id", "")).lower()
    if observed_id != network_id:
        raise RuntimeError("Docker network identity changed during inspection")
    internal = attrs.get("Internal")
    if not isinstance(internal, bool):
        raise RuntimeError("Docker network Internal flag is malformed")
    labels = attrs.get("Labels") or {}
    if not isinstance(labels, dict):
        raise RuntimeError("Docker network labels are malformed")
    attached = attrs.get("Containers") or {}
    if not isinstance(attached, dict) or target_container_id not in attached:
        raise RuntimeError("Docker network inspection omitted the target main container")
    project = labels.get(COMPOSE_PROJECT_LABEL)
    compose_network = labels.get(COMPOSE_NETWORK_LABEL)
    if project is not None and not isinstance(project, str):
        raise RuntimeError("Docker network Compose project label is malformed")
    if compose_network is not None and not isinstance(compose_network, str):
        raise RuntimeError("Docker network Compose network label is malformed")
    return {
        "network_id": network_id,
        "internal": internal,
        "compose_project": project,
        "compose_network": compose_network,
    }


def _main_network_records(container: Container) -> list[dict[str, Any]]:
    container.reload()
    attrs = container.attrs
    state = attrs.get("State") or {}
    if state.get("Running") is not True:
        raise RuntimeError("target main container is not running")
    network_mode = str((attrs.get("HostConfig") or {}).get("NetworkMode") or "")
    if network_mode in {"host", "none"} or network_mode.startswith("container:"):
        raise RuntimeError(f"target main uses an undetachable network mode: {network_mode}")
    attachments = (attrs.get("NetworkSettings") or {}).get("Networks") or {}
    if not isinstance(attachments, dict) or not attachments:
        raise RuntimeError("target main has no inspectable Docker networks")
    network_ids: set[str] = set()
    for attachment in attachments.values():
        if not isinstance(attachment, dict):
            raise RuntimeError("target main has a malformed network attachment")
        network_id = str(attachment.get("NetworkID") or "").lower()
        if not CONTAINER_ID.fullmatch(network_id) or network_id in network_ids:
            raise RuntimeError("target main has duplicate or invalid network attachments")
        network_ids.add(network_id)
    return sorted(
        (
            _network_record(network_id, target_container_id=str(container.id).lower())
            for network_id in network_ids
        ),
        key=lambda record: record["network_id"],
    )


def _front_network(
    records: list[dict[str, Any]], *, compose_project: str
) -> dict[str, Any]:
    matches = [
        record
        for record in records
        if record["internal"] is True
        and record["compose_project"] == compose_project
        and record["compose_network"] == "front"
    ]
    if len(matches) != 1:
        raise RuntimeError("main must have exactly one internal same-project front network")
    return matches[0]


def _same_project_service(
    project: str, *, expected_service: str
) -> tuple[Container, str, str]:
    matches = docker_client.containers.list(
        all=True,
        filters={
            "label": [
                f"{COMPOSE_PROJECT_LABEL}={project}",
                f"{COMPOSE_SERVICE_LABEL}={expected_service}",
                f"{COMPOSE_ONEOFF_LABEL}=False",
            ]
        },
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Compose project must contain exactly one {expected_service} container"
        )
    container = matches[0]
    observed_project, container_id, image_id = _compose_identity(
        container, expected_service=expected_service
    )
    if observed_project != project:
        raise RuntimeError("same-project Compose service labels disagree")
    return container, container_id, image_id


def _same_project_main() -> tuple[
    Container,
    Container,
    Container,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
]:
    """Resolve the hidden topology from daemon-authenticated Compose labels."""

    worker = docker_client.containers.get(socket.gethostname())
    project, worker_id, worker_image_id = _compose_identity(
        worker, expected_service="sandbox-worker"
    )
    main, main_id, main_image_id = _same_project_service(
        project, expected_service="main"
    )
    controller, controller_id, controller_image_id = _same_project_service(
        project, expected_service="bench-controller"
    )
    if len({worker_id, main_id, controller_id}) != 3:
        raise RuntimeError("trusted Compose services must be distinct containers")
    return (
        worker,
        main,
        controller,
        project,
        worker_id,
        worker_image_id,
        main_id,
        main_image_id,
        controller_id,
        controller_image_id,
    )


def seal_main_egress(request: dict[str, Any]) -> dict[str, Any]:
    """Remove every non-internal network from this Compose project's main."""

    if set(request) != {"action", "schema_version", "challenge"}:
        raise ValueError("network-seal request has unsupported fields")
    if request.get("action") != MAIN_EGRESS_SEAL_ACTION:
        raise ValueError("invalid network-seal action")
    if request.get("schema_version") != 2 or isinstance(
        request.get("schema_version"), bool
    ):
        raise ValueError("invalid network-seal schema version")
    challenge = request.get("challenge")
    if not isinstance(challenge, str) or not CHALLENGE.fullmatch(challenge):
        raise ValueError("invalid network-seal challenge")

    with main_egress_seal_lock:
        (
            _worker,
            main,
            _controller,
            project,
            worker_id,
            worker_image_id,
            main_id,
            main_image_id,
            controller_id,
            controller_image_id,
        ) = _same_project_main()
        before = _main_network_records(main)
        front = _front_network(before, compose_project=project)
        external = [record for record in before if record["internal"] is False]
        detached_ids = [record["network_id"] for record in external]
        for network_id in detached_ids:
            docker_client.networks.get(network_id).disconnect(main_id, force=True)

        after = _main_network_records(main)
        after_front = _front_network(after, compose_project=project)
        after_external = [record for record in after if record["internal"] is False]
        expected_after_ids = {
            record["network_id"] for record in before if record["internal"] is True
        }
        observed_after_ids = {record["network_id"] for record in after}
        if after_external:
            raise RuntimeError("main retained a non-internal Docker network")
        if observed_after_ids != expected_after_ids:
            raise RuntimeError("main network topology changed unexpectedly during egress seal")
        if after_front["network_id"] != front["network_id"]:
            raise RuntimeError("main internal front network was not preserved")

        evidence: dict[str, Any] = {
            "schema_version": 2,
            "action": MAIN_EGRESS_SEAL_ACTION,
            "challenge": challenge,
            "attestation_nonce": secrets.token_hex(32),
            "attested_unix_ns": time.time_ns(),
            "worker": {
                "container_id": worker_id,
                "image_id": worker_image_id,
                "compose_project": project,
                "compose_service": "sandbox-worker",
            },
            "target": {
                "container_id": main_id,
                "image_id": main_image_id,
                "compose_project": project,
                "compose_service": "main",
            },
            "controller": {
                "container_id": controller_id,
                "image_id": controller_image_id,
                "compose_project": project,
                "compose_service": "bench-controller",
            },
            "topology": {
                "front_network_id": front["network_id"],
                "before": before,
                "detached_network_ids": detached_ids,
                "after": after,
                "non_internal_before": len(external),
                "non_internal_after": len(after_external),
                "front_preserved": True,
                "already_sealed": not external,
            },
        }
        canonical = json.dumps(
            evidence, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode()
        evidence["signature"] = hmac.new(
            _admin_token(), MAIN_EGRESS_SEAL_CONTEXT + canonical, hashlib.sha256
        ).hexdigest()
        return evidence


def _forget_container(sandbox_id: str, container_id: str) -> None:
    """Forget only the exact generation that was successfully removed."""

    with runtimes_lock:
        if runtimes.get(sandbox_id) == container_id:
            runtimes.pop(sandbox_id, None)
        if owned_containers.get(sandbox_id) == container_id:
            owned_containers.pop(sandbox_id, None)


def _remove_container(container_id: str) -> None:
    """Reach the idempotent cleanup target with bounded Docker API retries."""

    last_error: APIError | None = None
    for attempt in range(CLEANUP_ATTEMPTS):
        try:
            docker_client.containers.get(container_id).remove(force=True)
            return
        except NotFound:
            return
        except APIError as exc:
            last_error = exc
            if attempt + 1 < CLEANUP_ATTEMPTS:
                time.sleep(CLEANUP_BACKOFF_SEC * (attempt + 1))
    if last_error is not None:
        raise last_error
    raise RuntimeError("unreachable sandbox cleanup retry state")


def _bounded_output(raw: bytes) -> tuple[str, bool]:
    truncated = len(raw) > MAX_OUTPUT_BYTES
    if truncated:
        raw = raw[:MAX_OUTPUT_BYTES] + b"\n[output truncated by sandbox]\n"
    return raw.decode("utf-8", errors="replace"), truncated


def run_workspace_command(workspace_name: str, command: str, timeout_sec: int) -> dict[str, Any]:
    """Run a synthetic/local-corpus command inside bubblewrap."""
    if not workspace_name or "/" in workspace_name or workspace_name in {".", ".."}:
        raise ValueError("invalid workspace name")
    if len(command.encode()) > MAX_COMMAND_BYTES:
        raise ValueError("command is too large")
    workspace = (WORKSPACE_ROOT / workspace_name).resolve(strict=True)
    workspace.relative_to(WORKSPACE_ROOT.resolve(strict=True))
    if workspace.is_symlink() or workspace.stat().st_uid not in {0, os.getuid()}:
        raise ValueError("workspace owner is not trusted")
    timeout_sec = max(1, min(int(timeout_sec), 120))
    bwrap = [
        "bwrap",
        "--unshare-pid",
        "--unshare-net",
        "--unshare-ipc",
        "--unshare-uts",
        "--die-with-parent",
        "--ro-bind", "/usr", "/usr",
        "--ro-bind", "/bin", "/bin",
        "--ro-bind", "/lib", "/lib",
        "--ro-bind", "/lib64", "/lib64",
        "--ro-bind", "/etc", "/etc",
        "--tmpfs", "/tmp",
        "--proc", "/proc",
        "--dev", "/dev",
        "--bind", str(workspace), "/workspace",
        "--chdir", "/workspace",
        "--new-session",
        "--clearenv",
        "--setenv", "PATH", "/usr/local/bin:/usr/sbin:/usr/bin:/bin",
        "--setenv", "HOME", "/tmp",
        "--setenv", "LANG", "C.UTF-8",
        "--setenv", "GIT_CONFIG_COUNT", "1",
        "--setenv", "GIT_CONFIG_KEY_0", "safe.directory",
        "--setenv", "GIT_CONFIG_VALUE_0", "/workspace",
        "bash", "-c", command,
    ]
    argv = [
        "prlimit",
        "--core=0",
        "--fsize=67108864",
        "--nofile=256",
        "--nproc=512",
        "--as=4294967296",
        "--",
        *bwrap,
    ]
    process = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    timed_out = False
    try:
        output, _ = process.communicate(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate(timeout=5)
    output_text, truncated = _bounded_output(output)
    return {
        "output": output_text,
        "returncode": 124 if timed_out else process.returncode,
        "timed_out": timed_out,
        "output_truncated": truncated,
    }


def _validate_sandbox_id(value: object) -> str:
    sandbox_id = str(value or "")
    if not SANDBOX_ID.fullmatch(sandbox_id):
        raise ValueError("invalid sandbox id")
    return sandbox_id


def _container_for(sandbox_id: str) -> Container:
    with runtimes_lock:
        container_id = runtimes.get(sandbox_id)
    if container_id is None:
        raise ValueError("unknown sandbox id")
    try:
        return docker_client.containers.get(container_id)
    except NotFound as exc:
        _forget_container(sandbox_id, container_id)
        raise RuntimeError("sandbox container disappeared") from exc


def _exec_container(container: Container, command: str, timeout_sec: int) -> dict[str, Any]:
    if len(command.encode()) > MAX_COMMAND_BYTES:
        raise ValueError("command is too large")
    timeout_sec = max(1, min(int(timeout_sec), 120))
    wrapped = [
        "timeout",
        "--signal=KILL",
        "--kill-after=2s",
        f"{timeout_sec}s",
        "bash",
        "-lc",
        command,
    ]
    created = docker_client.api.exec_create(
        container.id,
        wrapped,
        stdout=True,
        stderr=True,
        stdin=False,
        tty=False,
        workdir="/testbed",
        environment={
            "HOME": "/root",
            "LANG": "C.UTF-8",
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "safe.directory",
            "GIT_CONFIG_VALUE_0": "/testbed",
        },
    )
    exec_id = str(created["Id"])
    chunks: list[bytes] = []
    retained = 0
    truncated = False
    for chunk in docker_client.api.exec_start(exec_id, stream=True, demux=False):
        if not isinstance(chunk, bytes):
            chunk = bytes(chunk)
        remaining = MAX_OUTPUT_BYTES - retained
        if remaining > 0:
            chunks.append(chunk[:remaining])
            retained += min(len(chunk), remaining)
        if len(chunk) > remaining:
            truncated = True
    inspection = docker_client.api.exec_inspect(exec_id)
    exit_code = inspection.get("ExitCode")
    if exit_code is None:
        raise RuntimeError("sandbox exec did not report an exit code")
    if truncated:
        chunks.append(b"\n[output truncated by sandbox]\n")
    return {
        "output": b"".join(chunks).decode("utf-8", errors="replace"),
        "returncode": int(exit_code),
        "timed_out": int(exit_code) == 124,
        "output_truncated": truncated,
    }


def create_runtime(request: dict[str, Any]) -> dict[str, Any]:
    sandbox_id = _validate_sandbox_id(request.get("sandbox_id"))
    image_name = str(request.get("image") or "")
    expected_digest = str(request.get("image_digest") or "").lower()
    expected_manifest = str(request.get("image_manifest_digest") or "").lower()
    base_commit = str(request.get("base_commit") or "").lower()
    runtime_head_commit = str(request.get("runtime_head_commit") or "").lower()
    if not image_name or len(image_name) > 512:
        raise ValueError("invalid image name")
    if not IMAGE_DIGEST.fullmatch(expected_digest):
        raise ValueError("invalid image digest")
    if expected_manifest and not IMAGE_DIGEST.fullmatch(expected_manifest):
        raise ValueError("invalid image manifest digest")
    if not COMMIT.fullmatch(base_commit):
        raise ValueError("invalid base commit")
    if not COMMIT.fullmatch(runtime_head_commit):
        raise ValueError("invalid runtime HEAD")
    with runtimes_lock:
        if sandbox_id in runtimes or sandbox_id in owned_containers:
            raise ValueError("sandbox id already exists")
    try:
        image = docker_client.images.get(image_name)
    except ImageNotFound as exc:
        raise RuntimeError(f"staged SWE-bench image is missing: {image_name}") from exc
    # The classic Docker image store reports the image config digest as the image ID; the containerd image
    # store reports the digest of the pulled manifest.  Both are content addresses of the same pinned image,
    # so either form of the corpus-recorded identity is accepted.
    if image.id.lower() != expected_digest and not (
        expected_manifest and image.id.lower() == expected_manifest
    ):
        raise RuntimeError(
            f"staged SWE-bench image digest mismatch: expected {expected_digest}, got {image.id.lower()}"
        )

    container: Container | None = None
    labels = {MANAGED_LABEL: "true", f"{MANAGED_LABEL}.id": sandbox_id}
    if SANDBOX_OWNER:
        labels[OWNER_LABEL] = SANDBOX_OWNER
    try:
        container = docker_client.containers.create(
            image=image.id,
            name=f"harbor-kv-swe-{sandbox_id}",
            # This host defaults Docker to the NVIDIA runtime.  Tool
            # sandboxes must never inherit GPU device plumbing; only the
            # separately attested vLLM container owns the eight GPUs.
            runtime="runc",
            entrypoint=["/bin/sleep"],
            command=["21600"],
            working_dir="/testbed",
            user="root",
            network_disabled=True,
            network_mode="none",
            auto_remove=True,
            init=True,
            cap_drop=["ALL"],
            security_opt=["no-new-privileges:true"],
            mem_limit="8g",
            memswap_limit="8g",
            pids_limit=1024,
            nano_cpus=2_000_000_000,
            ulimits=[
                Ulimit(name="core", soft=0, hard=0),
                Ulimit(name="fsize", soft=2 * 1024**3, hard=2 * 1024**3),
                Ulimit(name="nofile", soft=1024, hard=1024),
            ],
            environment={
                "HOME": "/root",
                "LANG": "C.UTF-8",
                "http_proxy": "",
                "https_proxy": "",
                "HTTP_PROXY": "",
                "HTTPS_PROXY": "",
                "ALL_PROXY": "",
                "NO_PROXY": "*",
            },
            labels=labels,
        )
        # Ownership begins as soon as Docker returns a container ID, before
        # start/baseline validation can fail.  Only ready containers enter the
        # narrower runtimes map used by execute/cleanup RPCs.
        with runtimes_lock:
            if sandbox_id in owned_containers:
                raise ValueError("sandbox id already exists")
            owned_containers[sandbox_id] = container.id
        container.start()
        baseline = _exec_container(
            container,
            "test -d /testbed && "
            "test \"$(git rev-parse HEAD)\" = " + runtime_head_commit + " && "
            "git merge-base --is-ancestor " + base_commit + " HEAD && "
            "git reset --hard HEAD >/dev/null 2>&1 && git clean -fd >/dev/null 2>&1 && "
            "test -z \"$(git status --porcelain=v1)\"",
            120,
        )
        if baseline["returncode"] != 0:
            raise RuntimeError(f"SWE-bench image baseline validation failed: {baseline['output'][-1000:]}")
        with runtimes_lock:
            if owned_containers.get(sandbox_id) != container.id:
                raise RuntimeError("sandbox ownership changed during creation")
            runtimes[sandbox_id] = container.id
        return {"created": True, "sandbox_id": sandbox_id, "image_digest": image.id.lower()}
    except Exception as original_error:
        if container is not None:
            try:
                _remove_container(container.id)
            except APIError as cleanup_error:
                # Keep the ownership record so shutdown cleanup and the
                # run-scoped wrapper fallback can still identify the child.
                raise RuntimeError(
                    "sandbox creation failed and compensating cleanup exhausted retries"
                ) from original_error
            else:
                _forget_container(sandbox_id, container.id)
        raise


def execute_runtime(request: dict[str, Any]) -> dict[str, Any]:
    sandbox_id = _validate_sandbox_id(request.get("sandbox_id"))
    return _exec_container(
        _container_for(sandbox_id),
        str(request.get("command", "")),
        int(request.get("timeout_sec", 30)),
    )


def cleanup_runtime(request: dict[str, Any]) -> dict[str, Any]:
    sandbox_id = _validate_sandbox_id(request.get("sandbox_id"))
    with runtimes_lock:
        container_id = runtimes.get(sandbox_id)
    if container_id is None:
        raise ValueError("unknown sandbox id")
    _remove_container(container_id)
    _forget_container(sandbox_id, container_id)
    return {"removed": True, "sandbox_id": sandbox_id}


def dispatch(request: dict[str, Any]) -> dict[str, Any]:
    action = str(request.get("action") or "execute")
    if action == MAIN_EGRESS_SEAL_ACTION:
        return seal_main_egress(request)
    if action == "create":
        return create_runtime(request)
    if action == "cleanup":
        return cleanup_runtime(request)
    if action != "execute":
        raise ValueError("unknown sandbox action")
    if request.get("sandbox_id") is not None:
        return execute_runtime(request)
    return run_workspace_command(
        str(request.get("workspace", "")),
        str(request.get("command", "")),
        int(request.get("timeout_sec", 30)),
    )


async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        raw = await asyncio.wait_for(reader.readline(), timeout=5)
        if not raw or len(raw) > 256 * 1024:
            raise ValueError("invalid request size")
        request = json.loads(raw)
        if not isinstance(request, dict):
            raise ValueError("request must be an object")
        result = await asyncio.to_thread(dispatch, request)
    except Exception as exc:
        result = {"protocol_error": f"{type(exc).__name__}: {exc}"}
    writer.write(json.dumps(result, separators=(",", ":")).encode())
    await writer.drain()
    writer.close()
    await writer.wait_closed()


def cleanup_all_runtimes() -> None:
    with runtimes_lock:
        owned = list(owned_containers.items())
    failures: list[str] = []
    for sandbox_id, container_id in owned:
        try:
            _remove_container(container_id)
        except APIError as exc:
            failures.append(f"{sandbox_id}:{container_id}:{type(exc).__name__}:{exc}")
            continue
        _forget_container(sandbox_id, container_id)
    if failures:
        raise RuntimeError("sandbox shutdown cleanup failed: " + ";".join(failures))


async def main() -> None:
    docker_client.ping()
    SOCKET_PATH.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    SOCKET_PATH.parent.chmod(0o700)
    SOCKET_PATH.unlink(missing_ok=True)
    # A full 192-agent wave can finish a step or lifecycle at nearly the same
    # instant.  The asyncio default backlog (100) can then reject otherwise
    # valid local control connections with EAGAIN before the event loop drains
    # the queue.  Admission remains bounded by the runner/worker pools; this
    # only makes that trusted control-plane burst lossless.
    active_handlers: set[asyncio.Task[None]] = set()

    async def tracked_handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("sandbox handler has no asyncio task")
        active_handlers.add(task)
        try:
            await handle(reader, writer)
        finally:
            active_handlers.discard(task)

    server = await asyncio.start_unix_server(
        tracked_handle,
        path=SOCKET_PATH,
        backlog=CONTROL_SOCKET_BACKLOG,
    )
    SOCKET_PATH.chmod(0o600)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop.set)
    try:
        async with server:
            await stop.wait()
    finally:
        server.close()
        await server.wait_closed()
        # start_unix_server stops accepting but does not wait for callback
        # tasks.  Drain every accepted create/execute/cleanup before taking the
        # final ownership snapshot so a late create cannot escape shutdown.
        while active_handlers:
            await asyncio.gather(*tuple(active_handlers), return_exceptions=True)
        try:
            await asyncio.to_thread(cleanup_all_runtimes)
        finally:
            docker_client.close_all()
        SOCKET_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())
