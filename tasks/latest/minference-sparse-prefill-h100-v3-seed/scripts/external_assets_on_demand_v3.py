#!/usr/bin/env python3
"""Authenticate external RO assets and project only the bounded terminal QA files.

This root bootstrap authenticates metadata, membership and mount provenance. The
on-demand preparer and complete quality worker still authenticate all token and
model bytes. It accepts no QKV tree and copies no weights into an image.

Prepared E-only replacement: not installed or GPU-qualified. Expected content
descriptors are protected by the image epoch, never fabricated from an input
asset binding. Missing1950 descriptor is a hard failure, not an old300 fallback.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tempfile

PROTOCOL = "minference-external-assets-on-demand-v3"
CONTRACT_PROTOCOL = "minference-on-demand-asset-contract-v1"

ROOT = Path("/opt/minference_prefill_judge")
MODEL_REVISION = "d10aef7999a2b5ba950ab3974312feeedbfe0b77"
MOUNTS = {
    "private-performance": str(ROOT / "external-assets/kernel-prompts"),
    "model": str(ROOT / "model" / MODEL_REVISION),
}
MODEL_ASSET_ID = "4bbf2a8a1fe4f962a57a31052e924606c153a155fc67980ffcbb41f1fc9be174"
STORAGE_IDENTITY = {"endpoint": "https://storage.pyromind.ai", "bucket": "1000069766"}
CONTRACT_PATH = ROOT / "private/external-assets-contract.json"
WORK_MOUNTS = {"public-performance": "/workspace/benchmark/performance-prompts",
               "model": "/workspace/benchmark/model/" + MODEL_REVISION}
BINDINGS_PATH = ROOT / "private/external-assets-bindings.json"
RECEIPT_PATH = ROOT / "private/external-assets-receipt.json"
SHA = re.compile(r"[0-9a-f]{64}\Z")
IMAGE = re.compile(r"sha256:[0-9a-f]{64}\Z")
# A PyroMind Sandbox has no Docker inspect. The controller writes this receipt
# from the platform's own evidence (Sandbox id, pinned registry digest, the one
# GPU nvidia-smi reports inside the Judge, read-only volume mounts). Network
# isolation is not attested there, and the receipt says so.
PLATFORM_RECEIPT = "pyromind-sandbox-v1"
PLATFORM_NETWORK = "platform-default-not-attested"
SANDBOX = re.compile(r"sb-[0-9a-f]{12}\Z")
GPU_UUID = re.compile(r"GPU-[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}\Z")
PLATFORM_KEYS = {"platform", "sandbox_id", "image", "image_id", "user", "gpu_uuids",
                 "asset_mounts", "other_mounts", "network", "evidence"}
MAX_INPUT_BYTES = 4 * 1024**2
MAX_QA_FILE_BYTES = 1024 * 1024**2
MAX_KERNEL_FILE_BYTES = 64 * 1024**2


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def strict_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("non-finite JSON")))


def _absolute(value):
    if (not isinstance(value, str) or not value.startswith("/") or "\x00" in value
            or str(PurePosixPath(value)) != value or ".." in PurePosixPath(value).parts):
        raise ValueError("asset path must be canonical and absolute")
    return value


def _relative(value):
    if (not isinstance(value, str) or not value or "\x00" in value or value.startswith("/")
            or str(PurePosixPath(value)) != value or ".." in PurePosixPath(value).parts):
        raise ValueError("asset file path must be canonical and relative")
    return value


def _overlap(a, b):
    return a == b or a.startswith(b + "/") or b.startswith(a + "/")


def validate_contract(value, purpose="judge"):
    roles = set(MOUNTS) if purpose == "judge" else set(WORK_MOUNTS) if purpose == "public-diagnostic" else set()
    if (not roles or not isinstance(value, dict)
            or set(value) != {"schema_version", "protocol", "status", "purpose", "assets"}
            or type(value["schema_version"]) is not int or value["schema_version"] != 1
            or value["protocol"] != CONTRACT_PROTOCOL or value["status"] != "ready"
            or value["purpose"] != purpose or not isinstance(value["assets"], list)
            or len(value["assets"]) != len(roles)):
        raise ValueError("frozen ready asset descriptors are required; pending assets cannot run")
    seen = set()
    for row in value["assets"]:
        if not isinstance(row, dict) or set(row) != {"role", "asset_id", "files", "native_manifest_sha256"}:
            raise ValueError("invalid frozen asset descriptor fields")
        role = row["role"]
        if role not in roles or role in seen:
            raise ValueError("duplicate or unexpected frozen role")
        seen.add(role)
        files = row["files"]
        if not isinstance(files, list) or len(files) != (10 if role == "model" else 2):
            raise ValueError("exact model10 or two-file text asset required; retired QKV trees rejected")
        paths = []
        for entry in files:
            if not isinstance(entry, dict) or set(entry) != {"path", "size", "sha256"}:
                raise ValueError("invalid content manifest entry")
            paths.append(_relative(entry["path"]))
            if (type(entry["size"]) is not int or entry["size"] <= 0
                    or not isinstance(entry["sha256"], str) or not SHA.fullmatch(entry["sha256"])):
                raise ValueError("invalid content size or digest")
        if paths != sorted(set(paths)):
            raise ValueError("content files must be uniquely sorted")
        actual_id = digest(canonical({"schema_version": 1, "files": files}))
        if row["asset_id"] != actual_id:
            raise ValueError("frozen tree identity is not standard ASSETS content SHA")
        entries = {entry["path"]: entry for entry in files}
        if role == "model":
            if actual_id != MODEL_ASSET_ID or row["native_manifest_sha256"] is not None:
                raise ValueError("existing pinned model tree must be reused unchanged")
        elif (set(entries) != {"manifest.json", "prompts.jsonl"}
                or row["native_manifest_sha256"] != entries["manifest.json"]["sha256"]
                or any(e["size"] > (MAX_QA_FILE_BYTES if role == "terminal-quality" else MAX_KERNEL_FILE_BYTES)
                       for e in files)):
            raise ValueError("bounded native manifest plus real prompt JSONL required")
    return json.loads(canonical(value))


def load_contract():
    protected = _protected_map()
    raw = _read_file(CONTRACT_PATH, limit=MAX_INPUT_BYTES, root_private=True)
    if digest(raw) != protected.get(str(CONTRACT_PATH)):
        raise RuntimeError("expected asset contract is not protected by this image epoch")
    return validate_contract(strict_json(raw))


def validate_bindings(value, contract=None, *, purpose="judge"):
    # An explicit descriptor is for an already-authenticated trusted controller.
    # The installed Judge defaults to its root400, protected image contract.
    contract = validate_contract(load_contract() if contract is None else contract, purpose)
    expected = {row["role"]: row for row in contract["assets"]}
    mounts = MOUNTS if purpose == "judge" else WORK_MOUNTS
    if (not isinstance(value, dict) or set(value) != {"schema_version", "protocol", "assets"}
            or type(value["schema_version"]) is not int or value["schema_version"] != 1
            or value["protocol"] != PROTOCOL or not isinstance(value["assets"], list)
            or len(value["assets"]) != len(expected)):
        raise ValueError("all explicitly frozen on-demand external asset bindings are required")
    seen, sources = set(), []
    required = {"role", "mount_path", "source_directory", "asset_id", "files", "read_only",
                "visibility", "consumers", "native_manifest_sha256", "storage_identity"}
    for row in value["assets"]:
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError("unexpected asset binding fields")
        role = row["role"]
        if role not in expected or role in seen:
            raise ValueError("duplicate or unsupported asset role")
        seen.add(role); source = _absolute(row["source_directory"])
        if source == "/" or any(_overlap(source, previous) for previous in sources):
            raise ValueError("external source scopes must be distinct and non-overlapping")
        sources.append(source)
        public = purpose == "public-diagnostic" or role == "public-performance"
        consumers = ["worker"] if purpose == "public-diagnostic" else ["worker", "judge"] if public else ["judge"]
        if (row["storage_identity"] != STORAGE_IDENTITY or row["mount_path"] != mounts[role]
                or row["read_only"] is not True or row["visibility"] != ("public" if public else "private")
                or row["consumers"] != consumers):
            raise ValueError("asset role, destination, privacy, storage or read-only scope differs")
        if {k:row[k] for k in expected[role]} != expected[role]:
            raise ValueError("external tree differs from frozen expected descriptor")
    return json.loads(canonical(value))


def _mount_provenance(bindings):
    return [{"role": row["role"], "source_directory": row["source_directory"],
             "mount_path": row["mount_path"], "read_only": True,
             "asset_id": row["asset_id"], "storage_identity": row["storage_identity"]} for row in bindings["assets"]]


def validate_platform_receipt(receipt, bindings, image_id):
    """PyroMind branch: the same guarantees from platform evidence, never Docker fields."""
    if (not isinstance(receipt, dict) or set(receipt) != PLATFORM_KEYS
            or receipt["platform"] != PLATFORM_RECEIPT
            or not isinstance(receipt["sandbox_id"], str) or not SANDBOX.fullmatch(receipt["sandbox_id"])
            or not isinstance(image_id, str) or not IMAGE.fullmatch(image_id)
            or receipt["image_id"] != image_id
            or not isinstance(receipt["image"], str) or not receipt["image"].endswith("@" + image_id)
            or receipt["user"] != "root" or receipt["network"] != PLATFORM_NETWORK
            or not isinstance(receipt["evidence"], dict)):
        raise ValueError("actual PyroMind Sandbox/image/user receipt is required")
    gpus = receipt["gpu_uuids"]
    if not isinstance(gpus, list) or len(gpus) != 1 or not isinstance(gpus[0], str) or not GPU_UUID.fullmatch(gpus[0]):
        raise ValueError("exactly one physical GPU UUID is required")
    actual = {}
    for item in receipt["asset_mounts"] if isinstance(receipt["asset_mounts"], list) else [None]:
        if (not isinstance(item, dict) or set(item) != {"destination", "source", "read_only"}
                or item["read_only"] is not True or item["destination"] in actual):
            raise ValueError("asset mounts must be unique actual read-only mounts")
        actual[item["destination"]] = item["source"]
    expected = {row["mount_path"]: row["source_directory"] for row in bindings["assets"]}
    if actual != expected:
        raise ValueError("live mount sources/destinations differ from trusted content bindings")
    others = receipt["other_mounts"]
    if not isinstance(others, list):
        raise ValueError("malformed platform mount receipt")
    seen = set(actual)
    for item in others:
        if (not isinstance(item, dict) or set(item) != {"destination", "read_only"}
                or item["read_only"] is not True or not isinstance(item["destination"], str)
                or item["destination"] in seen
                or PurePosixPath(_absolute(item["destination"])).is_relative_to(ROOT)):
            raise ValueError("other mounts must be read-only and outside the trusted Judge root")
        seen.add(item["destination"])
    return {"container_id": receipt["sandbox_id"], "image_id": image_id, "mounts": _mount_provenance(bindings)}


def judge_gpu_uuid(inspection):
    """The one GPU the Judge was given, from either receipt kind."""
    if isinstance(inspection, list) and len(inspection) == 1:
        inspection = inspection[0]
    if isinstance(inspection, dict) and inspection.get("platform") == PLATFORM_RECEIPT:
        gpus = inspection.get("gpu_uuids")
        if not isinstance(gpus, list) or len(gpus) != 1:
            raise RuntimeError("kernel preparation lacks actual GPU binding")
        return gpus[0]
    devices = [device for request in (inspection.get("HostConfig", {}).get("DeviceRequests") or [])
               for device in request.get("DeviceIDs", [])]
    if not devices:
        raise RuntimeError("kernel preparation lacks actual GPU binding")
    return devices[0]


def validate_inspection(inspection, bindings, image_id, *, contract=None):
    bindings = validate_bindings(bindings, contract)
    if isinstance(inspection, list) and len(inspection) == 1:
        inspection = inspection[0]
    if isinstance(inspection, dict) and inspection.get("platform") == PLATFORM_RECEIPT:
        return validate_platform_receipt(inspection, bindings, image_id)
    if (not isinstance(image_id, str) or not IMAGE.fullmatch(image_id)
            or not isinstance(inspection, dict) or inspection.get("Image") != image_id
            or inspection.get("State", {}).get("Running") is not True
            or inspection.get("Config", {}).get("User") not in ("", "0", "root")
            or inspection.get("HostConfig", {}).get("NetworkMode") != "none"
            or inspection.get("HostConfig", {}).get("Privileged") is not False
            or inspection.get("HostConfig", {}).get("Devices") not in (None, [])):
        raise ValueError("actual running root container/image/network receipt is required")
    requests = inspection.get("HostConfig", {}).get("DeviceRequests") or []
    if not isinstance(requests, list):
        raise ValueError("malformed device request receipt")
    # Allocation and exact GPU UUID checks remain the controller/QC responsibility.
    mounts = inspection.get("Mounts")
    if not isinstance(mounts, list) or len(mounts) != 2:
        raise ValueError("exactly the two declared external mounts are allowed")
    actual = {}
    for item in mounts:
        if (not isinstance(item, dict) or item.get("Type") != "bind" or item.get("RW") is not False
                or item.get("Destination") in actual):
            raise ValueError("asset mounts must be unique actual read-only bind mounts")
        actual[item.get("Destination")] = item.get("Source")
    expected = {row["mount_path"]: row["source_directory"] for row in bindings["assets"]}
    if actual != expected:
        raise ValueError("live mount sources/destinations differ from trusted content bindings")
    cid = inspection.get("Id")
    if not isinstance(cid, str) or not SHA.fullmatch(cid):
        raise ValueError("actual container ID is required")
    return {"container_id": cid, "image_id": image_id, "mounts": _mount_provenance(bindings)}


def _identity(info):
    return tuple(getattr(info, name) for name in
        ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns"))


def _no_symlinks(path):
    path = Path(_absolute(str(path)))
    current = Path("/")
    for part in path.parts[1:]:
        current /= part
        if stat.S_ISLNK(current.lstat().st_mode):
            raise RuntimeError("asset path contains a symlink")
    return path


def _private_directory(path):
    path = _no_symlinks(path)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != 0 or stat.S_IMODE(info.st_mode) != 0o500:
        raise RuntimeError("private asset parent must be root-owned mode 0500")
    return _identity(info)


def _read_file(path, *, limit, expected=None, root_private=False):
    path = _no_symlinks(path)
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or not 0 < before.st_size <= limit
            or (root_private and (before.st_uid != 0 or stat.S_IMODE(before.st_mode) != 0o400))):
        raise RuntimeError("unsafe bounded asset metadata file")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        if _identity(os.fstat(fd)) != _identity(before):
            raise RuntimeError("asset file changed before read")
        chunks = []
        remaining = before.st_size
        while remaining:
            part = os.read(fd, min(remaining, 8 * 1024**2))
            if not part:
                raise RuntimeError("asset file truncated")
            chunks.append(part); remaining -= len(part)
        raw = b"".join(chunks)
        if _identity(os.fstat(fd)) != _identity(before) or _identity(path.lstat()) != _identity(before):
            raise RuntimeError("asset file changed during read")
        if expected is not None and (len(raw) != expected["size"] or digest(raw) != expected["sha256"]):
            raise RuntimeError("asset metadata size or digest differs")
        return raw
    finally:
        os.close(fd)


def _inventory(root, *, allow_hardlinks=False):
    root = _no_symlinks(root)
    if not stat.S_ISDIR(root.lstat().st_mode):
        raise RuntimeError("asset mount root must be a directory")
    found = {}
    for parent, directories, files in os.walk(root, followlinks=False):
        for name in directories:
            if stat.S_ISLNK((Path(parent) / name).lstat().st_mode):
                raise RuntimeError("asset directory symlink is forbidden")
        for name in files:
            path = Path(parent) / name
            info = path.lstat()
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink < 1
                    or (not allow_hardlinks and info.st_nlink != 1)):
                raise RuntimeError("asset files must be regular; non-model files must be single-link")
            found[path.relative_to(root).as_posix()] = info.st_size
    return found


def _protected_map():
    raw = _read_file(ROOT / "protected_hashes.txt", limit=16384, root_private=True)
    result = {}
    for line in raw.decode("ascii").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (/[^\x00]+)", line)
        if not match or match[2] in result:
            raise RuntimeError("malformed protected manifest")
        result[match[2]] = match[1]
    return result


def verify_sources(bindings):
    """Authenticate membership and native metadata, not model weights or QKV."""
    bindings = validate_bindings(bindings)
    for directory in (ROOT / "external-assets", ROOT / "model", ROOT / "private"):
        _private_directory(directory)
    protected = _protected_map(); summaries = []
    for row in bindings["assets"]:
        path = Path(row["mount_path"]); role = row["role"]
        if not os.statvfs(path).f_flag & os.ST_RDONLY:
            raise RuntimeError("actual external asset filesystem is not read-only")
        entries = {entry["path"]: entry for entry in row["files"]}
        if _inventory(path, allow_hardlinks=role == "model") != {n:e["size"] for n,e in entries.items()}:
            raise RuntimeError("asset membership or size differs from frozen tree")
        if role == "model":
            recipe_path = ROOT / "activations/recipe/RULER_V3_RECIPE.json"
            recipe_raw = _read_file(recipe_path, limit=4*1024**2, root_private=True)
            if digest(recipe_raw) != protected.get(str(recipe_path)):
                raise RuntimeError("model recipe is not protected")
            recipe = strict_json(recipe_raw)
            if ({n:e["sha256"] for n,e in entries.items()} != recipe["model"]["files"]
                    or recipe["model"]["revision"] != MODEL_REVISION):
                raise RuntimeError("model declaration differs from pinned recipe")
        else:
            native_raw = _read_file(path / "manifest.json", limit=4*1024**2, expected=entries["manifest.json"])
            native = strict_json(native_raw)
            if native.get("prompts_sha256") != entries["prompts.jsonl"]["sha256"]:
                raise RuntimeError("native JSONL digest differs from frozen two-file asset")
            if (native.get("protocol") != "ruler-derived-four-task-v1"
                    or native.get("partition") != ("public" if role == "public-performance" else "final")
                    or native.get("source_commit") != "a4eb395f949ea39e871f9bc586d683390692c6be"):
                raise RuntimeError("kernel native prompt recipe or partition differs")
        summaries.append({"role":role,"asset_id":row["asset_id"],"file_count":len(entries),
            "total_bytes":sum(e["size"] for e in entries.values()),
            "validation":"membership/size/native-manifest full SHA; input/model consumers verify their full bytes"})
    return summaries


PRIVACY_PROBE = r'''
import ctypes,json,os,sys
if os.getresuid()!=(1000,1000,1000) or os.getresgid()!=(1000,1000,1000) or os.getgroups():
    raise RuntimeError("probe did not permanently drop privileges")
libc=ctypes.CDLL(None,use_errno=True)
if libc.prctl(38,1,0,0,0)!=0 or libc.prctl(39,0,0,0,0)!=1:
    raise RuntimeError("probe no_new_privs failed")
value=json.loads(sys.stdin.buffer.read())
for path in value["directories"]:
    try: os.listdir(path)
    except PermissionError: pass
    else: raise RuntimeError("private directory readable after drop")
for path in value["files"]:
    try:
        fd=os.open(path,os.O_RDONLY)
    except PermissionError: pass
    else:
        os.close(fd);raise RuntimeError("private file readable after drop")
try: os.seteuid(0)
except PermissionError: pass
else: raise RuntimeError("probe recovered root")
if os.pread(value["retained_fd"],1,0)!=bytes.fromhex(value["expected_byte"]):
    raise RuntimeError("retained-FD filesystem read failed after drop")
print('{"direct_private_path_denied":true,"retained_fd_pread_after_permanent_drop":true,"uid":1000,"no_new_privs":true}')
'''


def probe_privacy(bindings):
    private = [row for row in bindings["assets"] if row["visibility"] == "private"]
    directories = [row["mount_path"] for row in private]
    files = [str(Path(row["mount_path"]) / entry["path"]) for row in private for entry in row["files"]]
    perf = next(row for row in private if row["role"] == "private-performance")
    payload = "prompts.jsonl"
    path = _no_symlinks(Path(perf["mount_path"]) / payload)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        expected = os.pread(fd, 1, 0)
        if len(expected) != 1:
            raise RuntimeError("private retained descriptor is empty")
        value = {"directories": directories, "files": files, "retained_fd": fd, "expected_byte": expected.hex()}
        result = subprocess.run([sys.executable, "-I", "-c", PRIVACY_PROBE], input=canonical(value),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=True,
            user=1000, group=1000, extra_groups=(), pass_fds=(fd,))
        return strict_json(result.stdout)
    finally:
        os.close(fd)


def _atomic_bytes(path, raw):
    parent = path.parent
    _private_directory(parent)
    fd, temporary = tempfile.mkstemp(prefix=".external-assets-", dir=parent)
    try:
        os.fchmod(fd, 0o400)
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        if os.path.lexists(temporary): os.unlink(temporary)


def _stream_projection(source, target, expected):
    """Bounded-memory full SHA copy and full readback, before any candidate import."""
    source = _no_symlinks(source); parent = target.parent; _private_directory(parent)
    fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    temporary = None
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or before.st_size != expected["size"] or not 0 < before.st_size <= MAX_QA_FILE_BYTES
                or _identity(source.lstat()) != _identity(before)):
            raise RuntimeError("unsafe bounded terminal source")
        outfd, temporary = tempfile.mkstemp(prefix=".external-quality-", dir=parent)
        h = hashlib.sha256(); copied = 0
        with os.fdopen(outfd, "wb") as out:
            os.fchmod(out.fileno(), 0o400)
            while copied < before.st_size:
                part = os.read(fd, min(8*1024**2, before.st_size-copied))
                if not part: raise RuntimeError("terminal source truncated")
                copied += len(part); h.update(part); out.write(part)
            out.flush(); os.fsync(out.fileno())
        if (h.hexdigest() != expected["sha256"] or _identity(os.fstat(fd)) != _identity(before)
                or _identity(source.lstat()) != _identity(before)):
            raise RuntimeError("terminal source digest/identity changed during projection")
        checkfd = os.open(temporary, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            check = os.fstat(checkfd); h = hashlib.sha256(); copied = 0
            if (check.st_uid != 0 or stat.S_IMODE(check.st_mode) != 0o400 or check.st_nlink != 1
                    or check.st_size != expected["size"]): raise RuntimeError("unsafe projected terminal file")
            while copied < check.st_size:
                part = os.read(checkfd, min(8*1024**2, check.st_size-copied))
                if not part: raise RuntimeError("projected terminal file truncated")
                copied += len(part); h.update(part)
            if h.hexdigest() != expected["sha256"] or _identity(check) != _identity(os.fstat(checkfd)):
                raise RuntimeError("projected terminal readback differs")
        finally: os.close(checkfd)
        os.replace(temporary, target); temporary = None
        directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        os.close(fd)
        if temporary is not None and os.path.lexists(temporary): os.unlink(temporary)


def _verify_stream_file(path, expected, *, root_private=False):
    path = _no_symlinks(path); fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or not 0 < before.st_size <= MAX_QA_FILE_BYTES or before.st_size != expected["size"]
                or (root_private and (before.st_uid != 0 or stat.S_IMODE(before.st_mode) != 0o400))):
            raise RuntimeError("unsafe bounded native input")
        h = hashlib.sha256(); count = 0
        while count < before.st_size:
            part = os.read(fd,min(8*1024**2,before.st_size-count))
            if not part: raise RuntimeError("native input truncated")
            h.update(part); count += len(part)
        if (h.hexdigest() != expected["sha256"] or _identity(before) != _identity(os.fstat(fd))
                or _identity(before) != _identity(path.lstat())):
            raise RuntimeError("native input digest or identity differs")
    finally: os.close(fd)


def install(bindings, inspection, *, image_id, bindings_sha256):
    if os.getuid() != 0 or os.geteuid() != 0:
        raise RuntimeError("external assets bootstrap must run as root before candidate import")
    bindings = validate_bindings(bindings)
    if not isinstance(bindings_sha256, str) or digest(canonical(bindings)) != bindings_sha256:
        raise ValueError("trusted external bindings digest differs")
    provenance = validate_inspection(inspection, bindings, image_id)
    summaries = verify_sources(bindings)
    privacy = probe_privacy(bindings)
    receipt = {"schema_version": 1, "protocol": PROTOCOL, "status": "ready",
        "bindings_sha256": bindings_sha256, "contract_sha256": digest(_read_file(CONTRACT_PATH, limit=MAX_INPUT_BYTES, root_private=True)), **provenance, "assets": summaries,
        "privacy": privacy,
        "source_validation_scope": "native-manifest/membership; terminal stream projected with full SHA; no persistent QKV, consumers authenticate model/kernel text"}
    _atomic_bytes(BINDINGS_PATH, canonical(bindings))
    _atomic_bytes(RECEIPT_PATH, canonical(receipt))
    return receipt


def verify_runtime_bindings(context):
    private = context.get("private")
    if not isinstance(private, dict):
        raise RuntimeError("trusted external assets context is missing")
    bindings_raw = _read_file(BINDINGS_PATH, limit=MAX_INPUT_BYTES, root_private=True)
    receipt_raw = _read_file(RECEIPT_PATH, limit=MAX_INPUT_BYTES, root_private=True)
    if (digest(bindings_raw) != private.get("EXTERNAL_ASSET_BINDINGS_SHA256")
            or digest(receipt_raw) != private.get("EXTERNAL_ASSET_RECEIPT_SHA256")):
        raise RuntimeError("trusted external asset binding/receipt digest differs")
    bindings = validate_bindings(strict_json(bindings_raw))
    receipt = strict_json(receipt_raw)
    expected_mounts = [{"role": row["role"], "source_directory": row["source_directory"],
                       "mount_path": row["mount_path"], "read_only": True,
                       "asset_id": row["asset_id"], "storage_identity": row["storage_identity"]} for row in bindings["assets"]]
    if (receipt.get("protocol") != PROTOCOL or receipt.get("schema_version") != 1
            or receipt.get("status") != "ready" or receipt.get("bindings_sha256") != digest(canonical(bindings))
            or receipt.get("contract_sha256") != digest(_read_file(CONTRACT_PATH, limit=MAX_INPUT_BYTES, root_private=True))
            or not IMAGE.fullmatch(str(receipt.get("image_id", "")))
            or not (SHA.fullmatch(str(receipt.get("container_id", "")))
                    or SANDBOX.fullmatch(str(receipt.get("container_id", ""))))
            or receipt.get("mounts") != expected_mounts
            or receipt.get("privacy") != {"direct_private_path_denied": True,
                "retained_fd_pread_after_permanent_drop": True, "uid": 1000, "no_new_privs": True}):
        raise RuntimeError("external asset bootstrap receipt is incomplete or mismatched")
    inspection = strict_json(_read_file(ROOT / "private/container-inspect.json", limit=MAX_INPUT_BYTES, root_private=True))
    provenance = validate_inspection(inspection, bindings, receipt["image_id"])
    if any(receipt.get(key) != value for key, value in provenance.items()):
        raise RuntimeError("external bootstrap belongs to a different inspected container")
    verify_sources(bindings)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["install"])
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--bindings-sha256", required=True)
    args = parser.parse_args()
    raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    if not raw or len(raw) > MAX_INPUT_BYTES:
        raise SystemExit("bounded trusted bindings and actual inspection are required")
    value = strict_json(raw)
    if not isinstance(value, dict) or set(value) != {"bindings", "inspection"}:
        raise SystemExit("stdin must contain exactly bindings and inspection")
    receipt = install(value["bindings"], value["inspection"], image_id=args.image_id, bindings_sha256=args.bindings_sha256)
    print(json.dumps({"status": "ready", "bindings_sha256": receipt["bindings_sha256"],
                      "receipt_sha256": digest(canonical(receipt))}, sort_keys=True))


if __name__ == "__main__":
    main()
