#!/usr/bin/env python3
"""Record and optionally prune only a newly owned, uniquely named local source.

This author tool never removes images or applies a broad cache selector. Cache
cleanup is opt-in, follows independent immutable-image verification, and fails
closed if any cache ownership or builder identity is ambiguous.
"""
from __future__ import annotations

import argparse
import calendar
from datetime import datetime
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import time

TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_ROOT / "runtime"))
from activation_archive_v3 import read_json, validate_manifest
from built_image_policy import IMAGE_ID, validate_work_image

ALIAS = re.compile(r"minference_v3_(?:public_capture|private_chunk)_[a-z0-9_]{12,80}\Z")
CACHE_ID = re.compile(r"[a-z0-9]{10,128}\Z")
GO_TIME = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))? \+0000 UTC\Z")
BUILDER = re.compile(r"[a-zA-Z0-9_.-]{1,80}\Z")
SHA = re.compile(r"[0-9a-f]{64}\Z")


def run(command, *, timeout=600, input=None):
    result = subprocess.run(command, input=input, capture_output=True, text=True,
                            check=False, timeout=timeout)
    if result.returncode:
        raise ValueError("author Docker operation failed: " + result.stderr.strip())
    return result


def atomic_record(path, record):
    """Write a private author receipt; do not follow or replace foreign files."""
    path = Path(path)
    info = path.parent.lstat()
    if (not stat.S_ISDIR(info.st_mode) or path.parent.is_symlink()
            or info.st_uid not in (0, os.getuid())
            or stat.S_IMODE(info.st_mode) != 0o700):
        raise ValueError("receipt parent must be an author/root-owned 0700 directory")
    if path.exists() or path.is_symlink():
        info = path.lstat()
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or info.st_uid not in (0, os.getuid())
                or stat.S_IMODE(info.st_mode) != 0o600):
            raise ValueError("existing author receipt is unsafe")
    fd, temporary = tempfile.mkstemp(prefix=".cache-receipt-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try: os.fsync(parent_fd)
        finally: os.close(parent_fd)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def builder_identity(engine, builder):
    if BUILDER.fullmatch(builder) is None:
        raise ValueError("invalid builder name")
    result = run([engine, "buildx", "inspect", builder])
    fields = {}
    for line in result.stdout.splitlines():
        match = re.match(r"^(Name|Driver|Endpoint|BuildKit version):\s*(.+)$", line)
        if match: fields.setdefault(match[1], []).append(match[2].strip())
    # Prune visits every node. Restrict this author helper to one local Docker
    # node instead of treating an aggregate multi-node inventory as ownership.
    if (fields.get("Name", [None])[0] != builder or fields.get("Driver") != ["docker"]
            or len(fields.get("Endpoint", [])) != 1
            or len(fields.get("BuildKit version", [])) != 1):
        raise ValueError("owned cleanup requires an explicit, single-node Docker builder")
    version = run([engine, "buildx", "version"]).stdout.strip()
    if (not re.search(r"(?:^|\s)v0\.37\.0(?:\s|$)", version)
            or fields["BuildKit version"] != ["v0.33.0"]):
        raise ValueError("source cleanup filter semantics are qualified only for Buildx v0.37.0 / BuildKit v0.33.0")
    daemon_id = run([engine, "info", "--format", "{{.ID}}"]).stdout.strip()
    if re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", daemon_id) is None:
        raise ValueError("actual Docker daemon identity is missing")
    return {"builder": builder, "fields": fields,
            "inspect_stdout": result.stdout,
            "buildx_version": version, "docker_daemon_id": daemon_id}


def _same_builder(left, right):
    if (left["builder"] != right["builder"] or left["fields"] != right["fields"]
            or left["buildx_version"] != right["buildx_version"]
            or left.get("docker_daemon_id") != right.get("docker_daemon_id")):
        raise ValueError("builder identity changed since source inventory")


def inventory(engine, builder, filters=()):
    command = [engine, "buildx", "du", "--builder", builder, "--format=json"]
    for value in filters: command.extend(["--filter", value])
    result = run(command)
    records = []
    for line in result.stdout.splitlines():
        if line.strip():
            record = json.loads(line)
            if not isinstance(record, dict) or CACHE_ID.fullmatch(str(record.get("ID"))) is None:
                raise ValueError("malformed cache record identity")
            records.append(record)
    if len({record["ID"] for record in records}) != len(records):
        raise ValueError("cache inventory contains duplicate record IDs")
    return {"command": command, "stdout": result.stdout, "stderr": result.stderr,
            "records": records, "recorded_at_ns": time.time_ns()}


def created_at_ns(value):
    if not isinstance(value, str) or (match := GO_TIME.fullmatch(value)) is None:
        raise ValueError("cache CreatedAt is not the expected UTC Go timestamp")
    seconds = calendar.timegm(datetime.strptime(match[1], "%Y-%m-%d %H:%M:%S").timetuple())
    return seconds * 10**9 + int((match[2] or "").ljust(9, "0"))


def displayed_size_bytes(value):
    # v0.37 du JSON renders human-readable decimal units, with rounded digits.
    # Preserve the original record and use this only as a plausibility bound.
    if type(value) is int and value >= 0: return value
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(B|kB|MB|GB|TB)", str(value))
    if match is None: raise ValueError("cache Size is malformed")
    power = {"B": 0, "kB": 1, "MB": 2, "GB": 3, "TB": 4}[match[2]]
    return int(Decimal(match[1]) * 1000**power)


def start_source_capture(*, engine, builder, alias):
    if ALIAS.fullmatch(alias) is None:
        raise ValueError("source alias must include a unique author campaign identity")
    identity = builder_identity(engine, builder)
    before = inventory(engine, builder)
    description = "local source for " + alias
    if any(record.get("Description") == description for record in before["records"]):
        raise ValueError("source alias already has cached records; ownership is ambiguous")
    return {"schema_version": 1, "protocol": "minference-owned-source-cache-v3",
            "alias": alias, "description": description, "builder_identity": identity,
            "before": before, "build_started_at_ns": time.time_ns(),
            "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "author local source-cache ownership; not release qualification"}


def owned_records(record, after, *, expected_bytes):
    before_ids = {row["ID"] for row in record["before"]["records"]}
    selected = [row for row in after["records"]
                if row.get("Description") == record["description"]]
    if not selected: raise ValueError("unique source context did not produce observable cache records")
    for row in selected:
        if (row["ID"] in before_ids or row.get("Type") != "source.local"
                or row.get("Reclaimable") is not True or row.get("Shared") is not False
                or type(row.get("Mutable")) is not bool
                or not record["build_started_at_ns"] <= created_at_ns(row.get("CreatedAt")) <= after["recorded_at_ns"]):
            raise ValueError("source cache ownership, time, type or reclaimability is ambiguous")
        if displayed_size_bytes(row.get("Size")) > expected_bytes * 1.02 + 16 * 1024**2:
            raise ValueError("source cache exceeds the declared chunk payload size")
    return selected


def cache_filters(cache_id):
    if CACHE_ID.fullmatch(cache_id) is None: raise ValueError("invalid full cache ID")
    # buildx v0.37 maps id= to a regular expression; anchor the complete ID.
    return ("id=^" + cache_id + "$", "type=source.local")


def complete_source_capture(record, *, engine, expected_bytes, verification, receipt,
                            prune=False):
    """Call only after the caller has fully verified the retained image bytes."""
    if (verification.get("status") != "passed"
            or IMAGE_ID.fullmatch(str(verification.get("image_id"))) is None
            or SHA.fullmatch(str(verification.get("manifest_sha256"))) is None):
        raise ValueError("cache cleanup requires an authenticated immutable image receipt")
    _same_builder(record["builder_identity"], builder_identity(engine, record["builder_identity"]["builder"]))
    after = inventory(engine, record["builder_identity"]["builder"])
    record.update(after=after, expected_payload_bytes=expected_bytes,
                  retained_image_verification=verification)
    if not any(row.get("Description") == record["description"] for row in after["records"]):
        # An ordinary build may trigger daemon GC before this inspection.
        # Absence permits no cleanup request, never a broader selector.
        record.update(owned_records=[], prune_commands=[], prune_operations=[],
                      status="image-verified-source-cache-already-absent",
                      absence_boundary="No record with the new exact alias is observable; no cache deletion requested.")
        atomic_record(receipt, record)
        return record
    selected = owned_records(record, after, expected_bytes=expected_bytes)
    record["owned_records"] = selected
    record["prune_commands"] = [[engine, "buildx", "prune", "--builder", record["builder_identity"]["builder"],
        "--force", "--verbose", "--filter", cache_filters(row["ID"])[0],
        "--filter", cache_filters(row["ID"])[1]] for row in selected]
    record["status"] = "verified-cache-retained"
    record["prune_operations"] = []
    atomic_record(receipt, record)
    if not prune: return record
    return prune_owned_source_cache(record, engine=engine, receipt=receipt)


def prune_owned_source_cache(record, *, engine, receipt):
    builder = record["builder_identity"]["builder"]
    _same_builder(record["builder_identity"], builder_identity(engine, builder))
    if record.get("status") != "verified-cache-retained":
        raise ValueError("receipt is not a pending, image-verified source cleanup")
    selected = owned_records(record, record["after"], expected_bytes=record["expected_payload_bytes"])
    if selected != record.get("owned_records"):
        raise ValueError("receipt owned cache records differ from authenticated inventory")
    try:
        for original in selected:
            _same_builder(record["builder_identity"], builder_identity(engine, builder))
            immediate_before = inventory(engine, builder)
            current = inventory(engine, builder, cache_filters(original["ID"]))
            if len(current["records"]) != 1:
                raise ValueError("anchored ID preflight does not identify exactly one cache record")
            observed = current["records"][0]
            for key in ("ID", "CreatedAt", "Description", "Type", "Mutable"):
                if observed.get(key) != original.get(key):
                    raise ValueError("owned cache identity changed before cleanup")
            if observed.get("Reclaimable") is not True or observed.get("Shared") is not False:
                raise ValueError("owned cache is now in use or shared with an image")
            command = [engine, "buildx", "prune", "--builder", builder, "--force", "--verbose"]
            for value in cache_filters(original["ID"]): command.extend(["--filter", value])
            operation = {"cache_id": original["ID"], "preflight": current,
                         "immediate_before": immediate_before,
                         "command": command, "status": "pending"}
            record["prune_operations"].append(operation); atomic_record(receipt, record)
            result = run(command, timeout=1800)
            operation.update(stdout=result.stdout, stderr=result.stderr,
                             after=inventory(engine, builder, cache_filters(original["ID"])))
            if operation["after"]["records"]:
                raise ValueError("selected source record remains; no broader cleanup will be attempted")
            immediate_after = inventory(engine, builder)
            operation["immediate_after"] = immediate_after
            owned_ids = {row["ID"] for row in selected}
            before_foreign = {row["ID"] for row in immediate_before["records"]} - owned_ids
            after_foreign = {row["ID"] for row in immediate_after["records"]} - owned_ids
            operation["foreign_record_ids_removed_during_cleanup"] = sorted(before_foreign - after_foreign)
            operation["foreign_record_ids_added_during_cleanup"] = sorted(after_foreign - before_foreign)
            if before_foreign != after_foreign:
                raise ValueError("concurrent or daemon cache inventory changed during cleanup; stop further requests")
            operation["status"] = "selected-source-removed"; atomic_record(receipt, record)
        record["status"] = "owned-source-cleanup-complete"
        record["after_cleanup"] = inventory(engine, builder)
        before_ids = {row["ID"] for row in record["before"]["records"]}
        final_ids = {row["ID"] for row in record["after_cleanup"]["records"]}
        record["preexisting_record_ids_absent_after_cleanup"] = sorted(before_ids - final_ids)
        record["cleanup_boundary"] = ("Only the anchored selected source.local IDs are requested. "
            "BuildKit may also remove a hidden mutable/immutable companion and performs daemon garbage collection; "
            "concurrent or automatic inventory changes are not attributed to these selectors.")
        atomic_record(receipt, record)
        return record
    except BaseException:
        record["status"] = "owned-source-cleanup-incomplete"
        atomic_record(receipt, record)
        raise


def verify_public_image(*, engine, image_id, manifest_path):
    if IMAGE_ID.fullmatch(image_id) is None: raise ValueError("Work image must be immutable")
    manifest, digest = read_json(manifest_path); validate_manifest(manifest)
    if manifest.get("partition") != "public" or manifest.get("capture_state") != "complete":
        raise ValueError("Work source manifest is not complete public data")
    validate_work_image(engine=engine, image=image_id)
    code = "import hashlib;from pathlib import Path;print(hashlib.sha256(Path('/workspace/benchmark/activations/manifest.json').read_bytes()).hexdigest())"
    result = run([engine, "run", "--rm", "--pull", "never", "--runtime", "runc",
        "--network", "none", "--cpus", "2", "--memory", "1g", "--user", "agent",
        "--env", "NVIDIA_VISIBLE_DEVICES=void", "--env", "CUDA_VISIBLE_DEVICES=",
        "--entrypoint", "python3", image_id, "-I", "-c", code])
    if result.stdout.strip() != digest: raise ValueError("Work image and host public manifest differ")
    return {"status": "passed", "kind": "complete-public-work", "image_id": image_id,
            "manifest_sha256": digest, "fixture_count": len(manifest["fixtures"]),
            "verified_all_fixture_sha256_and_headers": True,
            "image_policy_helper_sha256": hashlib.sha256((TASK_ROOT / "scripts/built_image_policy.py").read_bytes()).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--work-image", required=True)
    parser.add_argument("--public-manifest", type=Path, required=True)
    parser.add_argument("--engine", default="docker")
    parser.add_argument("--prune-owned-source-cache", action="store_true")
    args = parser.parse_args(argv)
    info = args.receipt.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or info.st_uid not in (0, os.getuid()) or stat.S_IMODE(info.st_mode) != 0o600):
        raise ValueError("source receipt must be a private author/root-owned regular file")
    record = json.loads(args.receipt.read_bytes())
    verification = verify_public_image(engine=args.engine, image_id=args.work_image,
                                       manifest_path=args.public_manifest)
    if verification != record.get("retained_image_verification"):
        raise ValueError("reverified Work image differs from original source receipt")
    if args.prune_owned_source_cache and record.get('status') != 'image-verified-source-cache-already-absent':
        prune_owned_source_cache(record, engine=args.engine, receipt=args.receipt)
    print(json.dumps({"status": record["status"], "receipt": str(args.receipt),
                      "owned_ids": [row["ID"] for row in record["owned_records"]]}))


if __name__ == "__main__":
    main()
