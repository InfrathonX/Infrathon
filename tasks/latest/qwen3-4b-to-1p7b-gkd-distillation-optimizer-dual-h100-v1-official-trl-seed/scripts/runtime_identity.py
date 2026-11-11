#!/usr/bin/env python3
"""Create and verify a non-secret host runtime identity for replay batches.

The identity intentionally excludes volatile utilization/current-clock values.
It pins the boot, Docker daemon/runtime, NVIDIA driver/device identity, and the
configured GPU policy values that can change benchmark performance.  GKD also
pins the complete ``nvidia-smi topo -m`` matrix.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
GPU_UUID_RE = re.compile(
    r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
GPU_QUERY = (
    ("index", "index"),
    ("uuid", "uuid"),
    ("name", "name"),
    ("driver_version", "driver_version"),
    ("memory_total_mib", "memory.total"),
    ("persistence_mode", "persistence_mode"),
    ("compute_mode", "compute_mode"),
    ("mig_mode_current", "mig.mode.current"),
    ("power_limit", "power.limit"),
    ("applications_graphics_clock", "clocks.applications.graphics"),
    ("applications_memory_clock", "clocks.applications.memory"),
    ("default_applications_graphics_clock", "clocks.default_applications.graphics"),
    ("default_applications_memory_clock", "clocks.default_applications.memory"),
    ("max_graphics_clock", "clocks.max.graphics"),
    ("max_sm_clock", "clocks.max.sm"),
    ("max_memory_clock", "clocks.max.memory"),
)
DOCKER_SERVER_KEYS = (
    "Version",
    "ApiVersion",
    "MinAPIVersion",
    "GitCommit",
    "GoVersion",
    "Os",
    "Arch",
    "KernelVersion",
    "BuildTime",
    "Experimental",
)
DOCKER_INFO_KEYS = (
    "ID",
    "ServerVersion",
    "Driver",
    "CgroupDriver",
    "CgroupVersion",
    "DefaultRuntime",
    "Runtimes",
    "SecurityOptions",
    "InitBinary",
    "DockerRootDir",
    "LiveRestoreEnabled",
    "Isolation",
    "OSType",
    "Architecture",
    "KernelVersion",
    "OperatingSystem",
    "OSVersion",
)


class RuntimeIdentityError(RuntimeError):
    """The host runtime cannot be attested without ambiguity."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def build_identity_document(observed: dict[str, Any]) -> dict[str, Any]:
    """Wrap observed non-secret fields in a self-authenticating document."""

    if not isinstance(observed, dict) or not observed:
        raise RuntimeIdentityError("host runtime identity is empty")
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "status": "verified",
        "identity": observed,
    }
    return {
        **unsigned,
        "identity_sha256": hashlib.sha256(_canonical_bytes(unsigned)).hexdigest(),
    }


def _validate_identity_document(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "status",
        "identity",
        "identity_sha256",
    }:
        raise RuntimeIdentityError("host runtime identity document is invalid")
    if document.get("schema_version") != SCHEMA_VERSION or document.get("status") != "verified":
        raise RuntimeIdentityError("host runtime identity document is unverified")
    observed = document.get("identity")
    claimed = document.get("identity_sha256")
    if not isinstance(observed, dict) or not observed or not isinstance(claimed, str):
        raise RuntimeIdentityError("host runtime identity document is invalid")
    expected = build_identity_document(observed)["identity_sha256"]
    if SHA256_RE.fullmatch(claimed) is None or claimed != expected:
        raise RuntimeIdentityError("host runtime identity digest is invalid")
    return document


def verify_same_identity(baseline: Any, current: Any) -> None:
    """Fail closed if any attested host field changed."""

    baseline_document = _validate_identity_document(baseline)
    current_document = _validate_identity_document(current)
    if baseline_document != current_document:
        raise RuntimeIdentityError("host runtime identity changed during replay batch")


def _stable_stat_tuple(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_stable_regular(
    path: Path,
    *,
    required_owner: int | None = None,
    private: bool = False,
) -> bytes:
    try:
        linked_before = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise RuntimeIdentityError("runtime identity input is unavailable") from exc
    if not stat.S_ISREG(linked_before.st_mode):
        raise RuntimeIdentityError("runtime identity input is not a regular file")
    if required_owner is not None and linked_before.st_uid != required_owner:
        raise RuntimeIdentityError("runtime identity input has an unexpected owner")
    if private and stat.S_IMODE(linked_before.st_mode) & 0o077:
        raise RuntimeIdentityError("runtime identity evidence is not private")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeIdentityError("runtime identity input could not be opened") from exc
    chunks: list[bytes] = []
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or (before.st_dev, before.st_ino)
            != (linked_before.st_dev, linked_before.st_ino)
        ):
            raise RuntimeIdentityError("runtime identity input changed before read")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        linked_after = os.stat(path, follow_symlinks=False)
        if (
            _stable_stat_tuple(before) != _stable_stat_tuple(after)
            or (after.st_dev, after.st_ino)
            != (linked_after.st_dev, linked_after.st_ino)
            or not stat.S_ISREG(linked_after.st_mode)
        ):
            raise RuntimeIdentityError("runtime identity input changed during read")
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _sha256_regular(path: Path) -> str:
    return hashlib.sha256(_read_stable_regular(path)).hexdigest()


def _trusted_executable(name: str) -> tuple[Path, dict[str, Any]]:
    located = shutil.which(name)
    if not located:
        raise RuntimeIdentityError(f"required runtime command is missing: {name}")
    try:
        resolved = Path(located).resolve(strict=True)
        info = os.stat(resolved, follow_symlinks=False)
    except OSError as exc:
        raise RuntimeIdentityError(f"runtime command is unavailable: {name}") from exc
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid not in {0, os.getuid()}
        or stat.S_IMODE(info.st_mode) & 0o022
    ):
        raise RuntimeIdentityError(f"runtime command is unsafe: {name}")
    content_sha256 = _sha256_regular(resolved)
    after = os.stat(resolved, follow_symlinks=False)
    if _stable_stat_tuple(info) != _stable_stat_tuple(after):
        raise RuntimeIdentityError(f"runtime command changed: {name}")
    return resolved, {
        "path": str(resolved),
        "owner_uid": info.st_uid,
        "mode": stat.S_IMODE(info.st_mode),
        "content_sha256": content_sha256,
    }


def _run(command: list[str], *, timeout: int = 30) -> str:
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
            env={**os.environ, "LANG": "C", "LC_ALL": "C"},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeIdentityError("host runtime identity query failed") from exc
    if completed.returncode != 0 or not completed.stdout.strip():
        raise RuntimeIdentityError("host runtime identity query failed")
    return completed.stdout.strip()


def _json_object(raw: str, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
        _canonical_bytes(value)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeIdentityError(f"{label} identity is invalid") from exc
    if not isinstance(value, dict):
        raise RuntimeIdentityError(f"{label} identity is not an object")
    return value


def _docker_socket_identity(docker: Path) -> dict[str, Any]:
    configured_host = os.environ.get("DOCKER_HOST", "").strip()
    context_name = _run([str(docker), "context", "show"])
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}", context_name) is None:
        raise RuntimeIdentityError("Docker context identity is invalid")
    context_host_raw = _run(
        [
            str(docker),
            "context",
            "inspect",
            "--format",
            "{{json .Endpoints.docker.Host}}",
            context_name,
        ]
    )
    try:
        context_host = json.loads(context_host_raw)
    except json.JSONDecodeError as exc:
        raise RuntimeIdentityError("Docker endpoint identity is invalid") from exc
    endpoint = configured_host or context_host
    if not isinstance(endpoint, str) or not endpoint.startswith("unix://"):
        raise RuntimeIdentityError("candidate replay requires a local Docker Unix socket")
    socket_path = Path(endpoint.removeprefix("unix://"))
    if not socket_path.is_absolute():
        raise RuntimeIdentityError("Docker Unix socket path is not absolute")
    try:
        before = os.stat(socket_path, follow_symlinks=False)
    except OSError as exc:
        raise RuntimeIdentityError("Docker Unix socket is unavailable") from exc
    if not stat.S_ISSOCK(before.st_mode):
        raise RuntimeIdentityError("Docker endpoint is not a Unix socket")
    return {
        "context": context_name,
        "endpoint": endpoint,
        "socket": {
            "device": before.st_dev,
            "inode": before.st_ino,
            "owner_uid": before.st_uid,
            "owner_gid": before.st_gid,
            "mode": stat.S_IMODE(before.st_mode),
        },
        "_socket_path": str(socket_path),
        "_socket_stat": _stable_stat_tuple(before),
    }


def _docker_identity(docker: Path, binary: dict[str, Any]) -> dict[str, Any]:
    endpoint = _docker_socket_identity(docker)
    socket_path = Path(endpoint.pop("_socket_path"))
    expected_socket_stat = endpoint.pop("_socket_stat")
    server = _json_object(
        _run([str(docker), "version", "--format", "{{json .Server}}"]),
        label="Docker server",
    )
    info = _json_object(
        _run([str(docker), "info", "--format", "{{json .}}"]),
        label="Docker daemon",
    )
    try:
        after = os.stat(socket_path, follow_symlinks=False)
    except OSError as exc:
        raise RuntimeIdentityError("Docker Unix socket changed") from exc
    if _stable_stat_tuple(after) != expected_socket_stat:
        raise RuntimeIdentityError("Docker Unix socket changed during attestation")
    if not isinstance(info.get("ID"), str) or not info["ID"].strip():
        raise RuntimeIdentityError("Docker daemon identity is empty")
    selected_server = {key: server.get(key) for key in DOCKER_SERVER_KEYS}
    selected_info = {key: info.get(key) for key in DOCKER_INFO_KEYS}
    _canonical_bytes(selected_server)
    _canonical_bytes(selected_info)
    return {
        "client_binary": binary,
        "endpoint": endpoint,
        "daemon_id": info["ID"],
        "server": selected_server,
        "daemon": selected_info,
    }


def _gpu_identity(
    nvidia_smi: Path,
    *,
    index: int,
    expected_uuid: str,
) -> dict[str, str]:
    query_fields = ",".join(field for _, field in GPU_QUERY)
    raw = _run(
        [
            str(nvidia_smi),
            f"--id={index}",
            f"--query-gpu={query_fields}",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = list(csv.reader(io.StringIO(raw), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != len(GPU_QUERY):
        raise RuntimeIdentityError("NVIDIA GPU identity query was ambiguous")
    values = [value.strip() for value in rows[0]]
    if any(not value for value in values):
        raise RuntimeIdentityError("NVIDIA GPU identity query returned an empty field")
    identity = {key: value for (key, _), value in zip(GPU_QUERY, values, strict=True)}
    if identity["index"] != str(index):
        raise RuntimeIdentityError("NVIDIA GPU index changed")
    if (
        GPU_UUID_RE.fullmatch(identity["uuid"]) is None
        or identity["uuid"].casefold() != expected_uuid.casefold()
    ):
        raise RuntimeIdentityError("NVIDIA GPU UUID changed")
    if identity["name"].casefold() in {"n/a", "unknown"}:
        raise RuntimeIdentityError("NVIDIA GPU name is unavailable")
    if identity["driver_version"].casefold() in {"n/a", "unknown"}:
        raise RuntimeIdentityError("NVIDIA driver version is unavailable")
    return identity


def _topology_identity(nvidia_smi: Path, indices: list[int]) -> dict[str, Any]:
    raw = _run([str(nvidia_smi), "topo", "-m"])
    lines = [" ".join(line.split()) for line in raw.splitlines() if line.strip()]
    normalized = "\n".join(lines) + "\n"
    gpu_labels = [f"GPU{index}" for index in indices]
    header: list[str] | None = None
    header_offset: int | None = None
    for line_offset, line in enumerate(lines):
        fields = line.split()
        if all(label in fields for label in gpu_labels):
            header = [field for field in fields if re.fullmatch(r"GPU[0-9]+", field)]
            header_offset = line_offset
            break
    if (
        header is None
        or header_offset is None
        or any(label not in header for label in gpu_labels)
    ):
        raise RuntimeIdentityError("NVIDIA topology header is incomplete")
    links: dict[str, str] = {}
    for source in gpu_labels:
        row = next(
            (
                line.split()
                for line in lines[header_offset + 1 :]
                if line.split() and line.split()[0] == source
            ),
            None,
        )
        if row is None:
            raise RuntimeIdentityError("NVIDIA topology row is incomplete")
        for target in gpu_labels:
            if source == target:
                continue
            offset = 1 + header.index(target)
            if offset >= len(row) or not row[offset]:
                raise RuntimeIdentityError("NVIDIA topology link is unavailable")
            links[f"{source}->{target}"] = row[offset]
    return {
        "matrix_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        "selected_links": links,
    }


def collect_host_identity(
    gpu_indices: list[int],
    expected_gpu_uuids: list[str],
    *,
    include_topology: bool,
) -> dict[str, Any]:
    if (
        not gpu_indices
        or len(gpu_indices) != len(expected_gpu_uuids)
        or len(set(gpu_indices)) != len(gpu_indices)
        or len({value.casefold() for value in expected_gpu_uuids}) != len(expected_gpu_uuids)
        or any(GPU_UUID_RE.fullmatch(value) is None for value in expected_gpu_uuids)
    ):
        raise RuntimeIdentityError("GPU identity selection is invalid")
    boot_id = _read_stable_regular(BOOT_ID_PATH).decode("ascii").strip().lower()
    if re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", boot_id) is None:
        raise RuntimeIdentityError("kernel boot identity is invalid")
    uname = os.uname()
    docker, docker_binary = _trusted_executable("docker")
    nvidia_smi, nvidia_binary = _trusted_executable("nvidia-smi")
    ordered = sorted(zip(gpu_indices, expected_gpu_uuids, strict=True))
    observed: dict[str, Any] = {
        "boot": {
            "boot_id": boot_id,
            "kernel": {
                "sysname": uname.sysname,
                "release": uname.release,
                "version": uname.version,
                "machine": uname.machine,
            },
        },
        "docker": _docker_identity(docker, docker_binary),
        "nvidia_smi_binary": nvidia_binary,
        "gpus": [
            _gpu_identity(nvidia_smi, index=index, expected_uuid=uuid)
            for index, uuid in ordered
        ],
    }
    if include_topology:
        observed["topology"] = _topology_identity(
            nvidia_smi, [index for index, _ in ordered]
        )
    return observed


def _read_identity(path: Path, *, expected_sha256: str | None = None) -> tuple[bytes, dict[str, Any]]:
    if not path.is_absolute():
        raise RuntimeIdentityError("runtime identity evidence path must be absolute")
    raw = _read_stable_regular(path, required_owner=os.getuid(), private=True)
    if expected_sha256 is not None:
        if SHA256_RE.fullmatch(expected_sha256) is None:
            raise RuntimeIdentityError("expected runtime identity file digest is invalid")
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise RuntimeIdentityError("runtime identity evidence changed")
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeIdentityError("runtime identity evidence is invalid") from exc
    return raw, _validate_identity_document(document)


def _write_new_private(path: Path, document: dict[str, Any]) -> None:
    if not path.is_absolute():
        raise RuntimeIdentityError("runtime identity output path must be absolute")
    try:
        parent = os.stat(path.parent, follow_symlinks=False)
    except OSError as exc:
        raise RuntimeIdentityError("runtime identity output parent is unavailable") from exc
    if (
        not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != os.getuid()
        or stat.S_IMODE(parent.st_mode) & 0o077
    ):
        raise RuntimeIdentityError("runtime identity output parent is unsafe")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    try:
        payload = _canonical_bytes(document) + b"\n"
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise RuntimeIdentityError("runtime identity output write failed")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-index", action="append", required=True)
    parser.add_argument("--expected-gpu-uuid", action="append", required=True)
    parser.add_argument("--include-topology", action="store_true")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path)
    action.add_argument("--verify-baseline", type=Path)
    parser.add_argument("--expected-baseline-sha256")
    args = parser.parse_args()
    try:
        if any(re.fullmatch(r"0|[1-9][0-9]*", value) is None for value in args.gpu_index):
            raise RuntimeIdentityError("GPU index must be a canonical nonnegative integer")
        gpu_indices = [int(value) for value in args.gpu_index]
        if args.output is not None and args.expected_baseline_sha256 is not None:
            parser.error("--expected-baseline-sha256 is valid only with --verify-baseline")
        if args.verify_baseline is not None and args.expected_baseline_sha256 is None:
            parser.error("--verify-baseline requires --expected-baseline-sha256")
        first = collect_host_identity(
            gpu_indices,
            args.expected_gpu_uuid,
            include_topology=args.include_topology,
        )
        second = collect_host_identity(
            gpu_indices,
            args.expected_gpu_uuid,
            include_topology=args.include_topology,
        )
        if first != second:
            raise RuntimeIdentityError("host runtime identity changed during attestation")
        current = build_identity_document(first)
        if args.output is not None:
            _write_new_private(args.output, current)
        else:
            assert args.verify_baseline is not None
            before_raw, baseline = _read_identity(
                args.verify_baseline,
                expected_sha256=args.expected_baseline_sha256,
            )
            verify_same_identity(baseline, current)
            after_raw, _ = _read_identity(
                args.verify_baseline,
                expected_sha256=args.expected_baseline_sha256,
            )
            if before_raw != after_raw:
                raise RuntimeIdentityError("runtime identity evidence changed")
    except RuntimeIdentityError as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
