#!/usr/bin/env python3
"""Precommit exact corrected/stress archives for one memory qualification cycle."""

from __future__ import annotations

import argparse
from pathlib import Path
import tarfile

from h100_policy_contract import (
    PROFILE,
    TASK_ID,
    object_at,
    sha256,
    validate_memory_policy,
    validate_workload_policy,
    write_object,
)


def candidate_source(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError("memory fixture archive is missing or unsafe")
    try:
        with tarfile.open(path, "r:gz") as bundle:
            members = bundle.getmembers()
            if any(
                member.issym()
                or member.islnk()
                or member.name.startswith("/")
                or ".." in Path(member.name).parts
                for member in members
            ):
                raise ValueError("memory fixture archive has unsafe members")
            candidates = [
                member
                for member in members
                if member.isfile()
                and member.name == "solutions/decoder_v001/candidate.py"
                and 0 < member.size <= 256 * 1024
            ]
            if len(candidates) != 1:
                raise ValueError("memory fixture archive has no unique candidate source")
            stream = bundle.extractfile(candidates[0])
            if stream is None:
                raise ValueError("memory fixture candidate source is unreadable")
            return stream.read()
    except (OSError, tarfile.TarError) as exc:
        raise ValueError("memory fixture archive is unreadable") from exc


def fixture(root: Path, path: Path, kind: str, target_mib: int | None) -> dict:
    resolved = path.resolve()
    if root not in resolved.parents:
        raise ValueError("memory fixture archive is outside task root")
    source = candidate_source(resolved)
    if kind == "corrected_reference":
        if b"STRESS_EXTRA_MIB" in source or source.count(b"THRESHOLD = 0.9") != 1:
            raise ValueError("corrected memory fixture source is not canonical")
    else:
        marker = f"STRESS_EXTRA_MIB = {target_mib}".encode()
        if source.count(marker) != 1 or b"stress.fill_(1)" not in source:
            raise ValueError("near-cap fixture does not encode/touch its exact target")
    return {
        "kind": kind,
        "archive": resolved.relative_to(root).as_posix(),
        "archive_sha256": sha256(resolved),
        "archive_size_bytes": resolved.stat().st_size,
        "stress_extra_mib": target_mib,
    }


def main() -> int:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-root", type=Path, default=default_root)
    parser.add_argument("--reference-archive", type=Path, required=True)
    parser.add_argument("--stress-archive", type=Path, required=True)
    parser.add_argument("--stress-extra-mib", type=int, required=True)
    parser.add_argument("--approval-id", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.task_root.resolve()
    if args.stress_extra_mib <= 0 or not args.approval_id:
        raise SystemExit("stress target and approval ID must be positive/nonempty")
    workload_path = root / "workload_policy.json"
    memory_path = root / "memory_policy.json"
    workload = object_at(workload_path, "workload policy")
    memory = object_at(memory_path, "memory policy")
    validate_workload_policy(workload)
    validate_memory_policy(memory)
    if memory.get("qualification_status") != "UNQUALIFIED":
        raise SystemExit("memory fixtures must be precommitted before qualification")
    payload = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "profile": PROFILE,
        "status": "PRECOMMITTED",
        "workload_policy_sha256": sha256(workload_path),
        "memory_policy_sha256": sha256(memory_path),
        "fixtures": {
            "corrected_reference": fixture(
                root, args.reference_archive, "corrected_reference", None
            ),
            "near_cap_stress": fixture(
                root, args.stress_archive, "near_cap_stress", args.stress_extra_mib
            ),
        },
        "approval_id": args.approval_id,
    }
    output = args.output.resolve() if args.output else root / "memory_qualification_fixtures.json"
    print(write_object(output, payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
