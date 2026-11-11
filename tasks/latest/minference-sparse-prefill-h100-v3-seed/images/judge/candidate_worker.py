#!/usr/bin/env python3
"""Process-isolated worker for one private MInference prefill suite."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import pwd
import re
import stat
import sys
from pathlib import Path
from typing import Any, Sequence


MAX_MANIFEST_BYTES = 16 * 1024
PROTOCOL = "minference-real-ruler-prefill-v3"
TRIAL_RANDOMNESS_PROTOCOL = "private-query-row-nonce-v3"
RUNTIME_ROOT = Path("/opt/minference_prefill_judge/runtime")
PR_SET_NO_NEW_PRIVS = 38
MAX_TRITON_CACHE_ENTRIES = 100_000


def _handoff_triton_cache(cache_root: Path, owner_uid: int, owner_gid: int) -> None:
    """Transfer only the freshly generated trusted Triton cache to ``agent``."""

    root = Path(cache_root)
    try:
        metadata = root.lstat()
    except FileNotFoundError:
        root.mkdir(mode=0o700)
        metadata = root.lstat()
    if root.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError("Triton cache root must be a real directory")
    entries = 1
    discovered: list[Path] = []
    for directory, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        for name in (*directory_names, *file_names):
            path = Path(directory) / name
            child = path.lstat()
            if path.is_symlink() or not (
                stat.S_ISDIR(child.st_mode) or stat.S_ISREG(child.st_mode)
            ):
                raise RuntimeError("Triton cache handoff found a symlink or non-regular entry")
            if stat.S_ISREG(child.st_mode) and child.st_nlink != 1:
                raise RuntimeError("Triton cache handoff found a hard-linked file")
            discovered.append(path)
            entries += 1
            if entries > MAX_TRITON_CACHE_ENTRIES:
                raise RuntimeError("Triton cache handoff exceeds the entry bound")
    for path in reversed(discovered):
        os.chown(path, owner_uid, owner_gid)
    os.chown(root, owner_uid, owner_gid)


def _candidate_triton_cache() -> Path:
    temporary_raw = os.environ.get("TMPDIR")
    triton_raw = os.environ.get("TRITON_CACHE_DIR")
    if not temporary_raw or not triton_raw:
        raise RuntimeError("candidate cache environment is incomplete")
    temporary = Path(os.path.abspath(temporary_raw))
    triton = Path(os.path.abspath(triton_raw))
    temporary_metadata = temporary.lstat()
    if temporary.is_symlink() or not stat.S_ISDIR(temporary_metadata.st_mode):
        raise RuntimeError("candidate temporary root must be a real directory")
    if triton != temporary / "triton":
        raise RuntimeError("candidate Triton cache must be the fixed temporary child")
    return triton


def parse_private_manifest_bytes(raw: bytes) -> dict[str, Any]:
    if not raw or len(raw) > MAX_MANIFEST_BYTES:
        raise ValueError("private manifest stdin is empty or oversized")
    try:
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate private JSON object key")
                result[key] = value
            return result
        parsed = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=unique,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("private manifest stdin is malformed") from exc
    if not isinstance(parsed, dict):
        raise ValueError("private manifest must be a JSON object")
    return parsed


def _stdin_manifest() -> dict[str, Any]:
    raw = sys.stdin.buffer.read(MAX_MANIFEST_BYTES + 1)
    sys.stdin.close()
    return parse_private_manifest_bytes(raw)


def _evaluation_nonce(manifest: dict[str, Any]) -> int:
    """Validate private transport metadata without disclosing supplied values."""

    nonce = manifest.get("evaluation_nonce")
    if type(nonce) is not int or not 1 <= nonce < (1 << 128):
        raise ValueError("private evaluation nonce is missing or malformed")
    if manifest.get("trial_randomness_protocol") != TRIAL_RANDOMNESS_PROTOCOL:
        raise ValueError("private trial randomness protocol differs")
    return nonce


def _transport_identity(manifest: dict[str, Any]) -> None:
    keys = {"schema_version", "protocol", "suite_type", "evaluation_nonce",
            "trial_randomness_protocol", "manifest_sha256",
            "numerical_policy_sha256", "candidate_sha256"}
    if (set(manifest) != keys or manifest.get("schema_version") != 3
            or manifest.get("protocol") != PROTOCOL
            or manifest.get("suite_type") != "final"):
        raise ValueError("private worker transport identity differs")
    for key in ("manifest_sha256", "numerical_policy_sha256", "candidate_sha256"):
        if not isinstance(manifest.get(key), str) or re.fullmatch(r"[0-9a-f]{64}", manifest[key]) is None:
            raise ValueError("private worker artifact digest is malformed")


def _contains_private_randomness(value: Any, *, depth: int = 0) -> bool:
    if depth > 64:
        raise RuntimeError("worker result nesting exceeds the bound")
    if isinstance(value, dict):
        return (any(key in value for key in ("nonce", "evaluation_nonce", "trial_nonce"))
                or any(_contains_private_randomness(child, depth=depth + 1)
                       for child in value.values()))
    if isinstance(value, (list, tuple)):
        return any(_contains_private_randomness(child, depth=depth + 1) for child in value)
    return False


def _load_runtime():
    sys.path.insert(0, str(RUNTIME_ROOT))
    try:
        import minference_prefill_runtime as runtime
    finally:
        sys.path.remove(str(RUNTIME_ROOT))
        os.environ.pop("PYTHONPATH", None)
    return runtime


def _drop_to_agent() -> None:
    if os.geteuid() != 0:
        return
    account = pwd.getpwnam("agent")
    _handoff_triton_cache(_candidate_triton_cache(), account.pw_uid, account.pw_gid)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS failed")
    os.setgroups([])
    os.setresgid(account.pw_gid, account.pw_gid, account.pw_gid)
    os.setresuid(account.pw_uid, account.pw_uid, account.pw_uid)
    os.umask(0o077)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--numerical-policy", type=Path, required=True)
    parser.add_argument("--activation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = _stdin_manifest()
    runtime = _load_runtime()
    try:
        try:
            nonce = _evaluation_nonce(manifest)
            _transport_identity(manifest)
        except ValueError as exc:
            raise runtime.RuntimeFailure(runtime.FailureCategory.CONTRACT, str(exc)) from exc
        if manifest.get("schema_version") != 3 or manifest.get("protocol") != PROTOCOL:
            raise runtime.RuntimeFailure(runtime.FailureCategory.CONTRACT,
                                         "private worker transport protocol differs")
        if manifest.get("suite_type") != "final":
            raise runtime.RuntimeFailure(runtime.FailureCategory.CONTRACT,
                                         "private worker suite type differs")
        result = runtime.run_suite(
            args.candidate.resolve(),
            manifest_path=args.manifest.resolve(),
            activation_root=args.activation_root.resolve(),
            numerical_policy_path=args.numerical_policy.resolve(),
            suite_type=manifest["suite_type"],
            nonce=nonce,
            before_candidate_import=_drop_to_agent,
        )
        if not isinstance(result, dict):
            raise RuntimeError("trusted runtime returned a non-object")
        if _contains_private_randomness(result):
            raise RuntimeError("trusted runtime returned private trial metadata")
        for field in ("manifest_sha256", "numerical_policy_sha256", "candidate_sha256"):
            if result.get(field) != manifest.get(field):
                raise RuntimeError("trusted runtime artifact identity differs")
        if result.get("schema_version") != 3 or result.get("protocol") != PROTOCOL:
            raise RuntimeError("trusted runtime protocol differs")
        _atomic_json(args.output.resolve(), result)
        return 0
    except runtime.RuntimeFailure as exc:
        _atomic_json(
            args.output.resolve(),
            {
                "schema_version": 3,
                "ok": False,
                "failure_category": exc.category.value,
                "case_id": exc.case_id,
                "message": str(exc),
            },
        )
        return 1
    except BaseException as exc:
        _atomic_json(
            args.output.resolve(),
            {
                "schema_version": 3,
                "ok": False,
                "failure_category": "crash",
                "message": f"{type(exc).__name__}: {exc}",
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
