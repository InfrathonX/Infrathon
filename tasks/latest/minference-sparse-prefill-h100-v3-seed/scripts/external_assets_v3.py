#!/usr/bin/env python3
"""Authenticate external RO assets and project only the bounded terminal QA files.

This root bootstrap authenticates metadata, membership and mount provenance. The
retained-FD archive reader and quality worker still authenticate all payload and
model bytes. It never copies QKV payloads or model weights into an image.
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

PROTOCOL = "minference-external-assets-v3"
ROOT = Path("/opt/minference_prefill_judge")
MODEL_REVISION = "d10aef7999a2b5ba950ab3974312feeedbfe0b77"
MOUNTS = {
    "public-performance": "/workspace/benchmark/activations",
    "private-performance": str(ROOT / "external-assets/private-activations"),
    "model": str(ROOT / "model" / MODEL_REVISION),
    "terminal-quality": str(ROOT / "external-assets/quality-prompts"),
}
EXPECTED_ASSET_IDS = {
    "public-performance": "0a37803a0186055f1d6044c14a1413190e8450a8e0cf3cbfa4ba46cd6dd3a787",
    "private-performance": "54029ab60115e7e45941b718c2b6a5e860cac5e4e7f1d75baf112d212da78b0b",
    "model": "4bbf2a8a1fe4f962a57a31052e924606c153a155fc67980ffcbb41f1fc9be174",
    "terminal-quality": "773ec2587afee26137c640aa04589380215f03be013b85054ec7f9b764824d73",
}
EXPECTED_NATIVE_MANIFESTS = {
    "public-performance": "b0423abae3992c6654731a42f623c1c3bf5de86def154d80197732f7ce95a0a0",
    "private-performance": "ae90f621d1a02811d175ea0384fa39ec7e1c365bad816145d42de3e73df84b77",
}
STORAGE_IDENTITY = {"endpoint": "https://storage.pyromind.ai", "bucket": "1000069766"}
EXPECTED_COUNTS = {"public-performance": 183, "private-performance": 183, "model": 10, "terminal-quality": 2}
BINDINGS_PATH = ROOT / "private/external-assets-bindings.json"
RECEIPT_PATH = ROOT / "private/external-assets-receipt.json"
SHA = re.compile(r"[0-9a-f]{64}\Z")
IMAGE = re.compile(r"sha256:[0-9a-f]{64}\Z")
MAX_INPUT_BYTES = 4 * 1024**2
MAX_QA_FILE_BYTES = 256 * 1024**2


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


def validate_bindings(value):
    if (not isinstance(value, dict) or set(value) != {"schema_version", "protocol", "assets"}
            or type(value["schema_version"]) is not int or value["schema_version"] != 1
            or value["protocol"] != PROTOCOL or not isinstance(value["assets"], list)
            or len(value["assets"]) != 4):
        raise ValueError("four explicit v3 external asset bindings are required")
    seen, sources = set(), []
    required = {"role", "mount_path", "source_directory", "asset_id", "files", "read_only",
                "visibility", "consumers", "native_manifest_sha256", "storage_identity"}
    for row in value["assets"]:
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError("unexpected asset binding fields")
        role = row["role"]
        if role not in MOUNTS or role in seen:
            raise ValueError("duplicate or unsupported asset role")
        seen.add(role)
        source = _absolute(row["source_directory"])
        if source == "/" or any(_overlap(source, previous) for previous in sources):
            raise ValueError("external source scopes must be distinct and non-overlapping")
        sources.append(source)
        if row["storage_identity"] != STORAGE_IDENTITY:
            raise ValueError("external storage identity differs from trusted deployment")
        public = role == "public-performance"
        if (row["mount_path"] != MOUNTS[role] or row["read_only"] is not True
                or row["visibility"] != ("public" if public else "private")
                or row["consumers"] != (["worker", "judge"] if public else ["judge"])):
            raise ValueError("asset role, destination, privacy or read-only contract differs")
        files = row["files"]
        if not isinstance(files, list) or len(files) != EXPECTED_COUNTS[role]:
            raise ValueError("asset file count differs from frozen tree")
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
        if row["asset_id"] != actual_id or actual_id != EXPECTED_ASSET_IDS[role]:
            raise ValueError("external tree identity differs from frozen content")
        native = row["native_manifest_sha256"]
        if role == "model":
            if native is not None:
                raise ValueError("model has no native activation manifest")
        elif not isinstance(native, str) or not SHA.fullmatch(native):
            raise ValueError("native manifest digest is required")
        if role in EXPECTED_NATIVE_MANIFESTS and native != EXPECTED_NATIVE_MANIFESTS[role]:
            raise ValueError("native performance manifest differs from frozen capture")
        if role != "model":
            entries = {entry["path"]: entry for entry in files}
            if "manifest.json" not in entries or entries["manifest.json"]["sha256"] != native:
                raise ValueError("native manifest must be part of the content identity")
    return json.loads(canonical(value))


def validate_inspection(inspection, bindings, image_id):
    bindings = validate_bindings(bindings)
    if isinstance(inspection, list) and len(inspection) == 1:
        inspection = inspection[0]
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
    if not isinstance(mounts, list) or len(mounts) != 4:
        raise ValueError("exactly the four declared external mounts are allowed")
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
    return {"container_id": cid, "image_id": image_id,
            "mounts": [{"role": row["role"], "source_directory": row["source_directory"],
                        "mount_path": row["mount_path"], "read_only": True,
                        "asset_id": row["asset_id"], "storage_identity": row["storage_identity"]} for row in bindings["assets"]]}


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
    """Check only bounded metadata and membership, never the giant payload bytes."""
    bindings = validate_bindings(bindings)
    _private_directory(ROOT / "external-assets")
    _private_directory(ROOT / "model")
    _private_directory(ROOT / "private")
    _private_directory(ROOT / "private/quality-prompts")
    protected = _protected_map()
    summaries = []
    for row in bindings["assets"]:
        path = Path(row["mount_path"])
        if not os.statvfs(path).f_flag & os.ST_RDONLY:
            raise RuntimeError("actual external asset filesystem is not read-only")
        entries = {entry["path"]: entry for entry in row["files"]}
        if _inventory(path, allow_hardlinks=row["role"] == "model") != {name: entry["size"] for name, entry in entries.items()}:
            raise RuntimeError("asset membership or size differs from frozen tree")
        role = row["role"]
        if role.endswith("performance"):
            raw = _read_file(path / "manifest.json", limit=4 * 1024**2, expected=entries["manifest.json"])
            native = strict_json(raw)
            for name in ("capture-state.json", "performance-plan.json"):
                _read_file(path / name, limit=4 * 1024**2, expected=entries[name])
            spec_path = ROOT / "runtime/activation_archive_v3.py"
            if digest(_read_file(spec_path, limit=1024**2, root_private=True)) != protected.get(str(spec_path)):
                raise RuntimeError("archive reader differs from protected source")
            import importlib.util
            spec = importlib.util.spec_from_file_location("minference_external_archive_reader", spec_path)
            reader = importlib.util.module_from_spec(spec); sys.modules[spec.name] = reader
            spec.loader.exec_module(reader)
            reader.validate_manifest(native)
            if native["partition"] != ("public" if role == "public-performance" else "final"):
                raise RuntimeError("performance asset partition differs")
            fixtures = native["fixtures"]
            payload = {item["file"]: {"path": item["file"], "size": item["size"], "sha256": item["sha256"]}
                       for item in fixtures.values()}
            if ({name: entry for name, entry in entries.items() if name not in
                    {"manifest.json", "capture-state.json", "performance-plan.json"}} != payload):
                raise RuntimeError("content tree differs from native 180-fixture manifest")
            if role == "private-performance":
                expected = protected.get(str(ROOT / "private/final_manifest.json"))
                if digest(raw) != expected:
                    raise RuntimeError("private source manifest differs from protected final manifest")
        elif role == "model":
            recipe = strict_json(_read_file(ROOT / "activations/recipe/RECIPE.json", limit=1024**2, root_private=True))
            if ({name: entry["sha256"] for name, entry in entries.items()} != recipe["model"]["files"]
                    or recipe["model"]["revision"] != MODEL_REVISION):
                raise RuntimeError("model content declaration differs from protected recipe")
        else:
            if set(entries) != {"manifest.json", "prompts.jsonl"}:
                raise RuntimeError("only two terminal QA files may be projected")
            for name, entry in entries.items():
                target = ROOT / "private/quality-prompts" / name
                if entry["size"] > MAX_QA_FILE_BYTES or entry["sha256"] != protected.get(str(target)):
                    raise RuntimeError("terminal source files differ from protected terminal300")
        summaries.append({"role": role, "asset_id": row["asset_id"], "file_count": len(entries),
            "total_bytes": sum(entry["size"] for entry in entries.values()),
            "validation": "exact-membership-size-and-bounded-metadata; payload-SHA-deferred-to-native-consumer"})
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
    directories = [row["mount_path"] for row in private] + [str(ROOT / "private/quality-prompts")]
    files = [str(Path(row["mount_path"]) / entry["path"]) for row in private for entry in row["files"]]
    files += [str(ROOT / "private/quality-prompts" / name) for name in ("manifest.json", "prompts.jsonl")]
    perf = next(row for row in private if row["role"] == "private-performance")
    payload = next(entry["path"] for entry in perf["files"] if entry["path"].endswith(".safetensors"))
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


def project_quality(bindings):
    row = next(row for row in bindings["assets"] if row["role"] == "terminal-quality")
    result = []
    for entry in row["files"]:
        raw = _read_file(Path(row["mount_path"]) / entry["path"], limit=MAX_QA_FILE_BYTES, expected=entry)
        target = ROOT / "private/quality-prompts" / entry["path"]
        _atomic_bytes(target, raw)
        if _read_file(target, limit=MAX_QA_FILE_BYTES, expected=entry, root_private=True) != raw:
            raise RuntimeError("terminal QA projection readback differs")
        result.append(dict(entry))
    return result


def install(bindings, inspection, *, image_id, bindings_sha256):
    if os.getuid() != 0 or os.geteuid() != 0:
        raise RuntimeError("external assets bootstrap must run as root before candidate import")
    bindings = validate_bindings(bindings)
    if not isinstance(bindings_sha256, str) or digest(canonical(bindings)) != bindings_sha256:
        raise ValueError("trusted external bindings digest differs")
    provenance = validate_inspection(inspection, bindings, image_id)
    summaries = verify_sources(bindings)
    projection = project_quality(bindings)
    privacy = probe_privacy(bindings)
    receipt = {"schema_version": 1, "protocol": PROTOCOL, "status": "ready",
        "bindings_sha256": bindings_sha256, **provenance, "assets": summaries,
        "qa_projection": projection, "privacy": privacy,
        "source_validation_scope": "metadata/membership only; native consumer authenticates full QKV/model bytes"}
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
            or not IMAGE.fullmatch(str(receipt.get("image_id", "")))
            or not SHA.fullmatch(str(receipt.get("container_id", "")))
            or receipt.get("mounts") != expected_mounts
            or receipt.get("privacy") != {"direct_private_path_denied": True,
                "retained_fd_pread_after_permanent_drop": True, "uid": 1000, "no_new_privs": True}):
        raise RuntimeError("external asset bootstrap receipt is incomplete or mismatched")
    inspection = strict_json(_read_file(ROOT / "private/container-inspect.json", limit=MAX_INPUT_BYTES, root_private=True))
    provenance = validate_inspection(inspection, bindings, receipt["image_id"])
    if any(receipt.get(key) != value for key, value in provenance.items()):
        raise RuntimeError("external bootstrap belongs to a different inspected container")
    verify_sources(bindings)
    row = next(row for row in bindings["assets"] if row["role"] == "terminal-quality")
    if receipt.get("qa_projection") != row["files"]:
        raise RuntimeError("QA projection receipt differs from frozen tree")
    for entry in row["files"]:
        _read_file(ROOT / "private/quality-prompts" / entry["path"], limit=MAX_QA_FILE_BYTES,
                   expected=entry, root_private=True)
    return receipt


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
