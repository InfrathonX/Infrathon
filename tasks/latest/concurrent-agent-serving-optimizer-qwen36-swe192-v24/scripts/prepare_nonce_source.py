#!/usr/bin/env python3
"""Create and remove the formal run's minimal, private nonce source.

The formal Judge serves two independently committed workloads: hidden scoring
and the agent-visible public diagnostic.  Its dedicated source therefore holds
exactly those two values in a canonical order and nothing else.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import stat
import tempfile


HIDDEN_COMMITMENT = (
    "c525ae31a2bb1d3ed0232c345f9fc978fd52574d415aaa2777bead17f2fac20a"
)
PUBLIC_COMMITMENT = (
    "4902baae73726c1a2f372e35e9e104f6bb3db7dc7593822359548fba60e1b00c"
)
TOKEN_PATTERN = re.compile(r"(?<![0-9a-f])([0-9a-f]{64})(?![0-9a-f])")
TOKEN_LINE_PATTERN = re.compile(r"[0-9a-f]{64}")


class NonceSourceError(RuntimeError):
    """A nonce source failed a fail-closed ownership or content check."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_private_parent(parent: Path) -> Path:
    if parent.is_symlink():
        raise NonceSourceError("dedicated nonce parent must not be a symlink")
    resolved = parent.resolve(strict=True)
    info = resolved.stat()
    if not stat.S_ISDIR(info.st_mode):
        raise NonceSourceError("dedicated nonce parent must be a real directory")
    if info.st_uid != os.getuid():
        raise NonceSourceError("dedicated nonce parent is not owned by the current uid")
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise NonceSourceError("dedicated nonce parent is accessible by group or other")
    return resolved


def _commitments() -> tuple[str, str]:
    return HIDDEN_COMMITMENT, PUBLIC_COMMITMENT


def _matching_nonces(source: Path) -> tuple[str, str]:
    resolved = source.resolve(strict=True)
    info = resolved.stat()
    if not stat.S_ISREG(info.st_mode) or source.is_symlink():
        raise NonceSourceError("original nonce source must be a regular non-symlink file")
    commitments = _commitments()
    matches: dict[str, set[str]] = {commitment: set() for commitment in commitments}
    with resolved.open(encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            for token in TOKEN_PATTERN.findall(line):
                digest = _sha256_bytes(token.encode("ascii"))
                if digest in matches:
                    matches[digest].add(token)
    for label, commitment in zip(("hidden", "public"), commitments, strict=True):
        if len(matches[commitment]) != 1:
            raise NonceSourceError(
                f"original nonce source must contain exactly one distinct {label} "
                "committed value"
            )
    hidden = next(iter(matches[commitments[0]]))
    public = next(iter(matches[commitments[1]]))
    if hidden == public:
        raise NonceSourceError("hidden and public committed values must be distinct")
    return hidden, public


def _canonical_payload(tokens: tuple[str, str]) -> bytes:
    return (tokens[0] + "\n" + tokens[1] + "\n").encode("ascii")


def _validate_payload(raw: bytes) -> tuple[str, str]:
    if len(raw) != 130 or raw.count(b"\n") != 2 or not raw.endswith(b"\n"):
        raise NonceSourceError("dedicated nonce source is not the canonical two-line form")
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise NonceSourceError("dedicated nonce source is not ASCII") from exc
    if len(lines) != 2 or any(TOKEN_LINE_PATTERN.fullmatch(line) is None for line in lines):
        raise NonceSourceError("dedicated nonce source has an invalid token shape")
    for label, token, commitment in zip(
        ("hidden", "public"), lines, _commitments(), strict=True
    ):
        if _sha256_bytes(token.encode("ascii")) != commitment:
            raise NonceSourceError(
                f"dedicated nonce source {label} commitment mismatch"
            )
    if lines[0] == lines[1]:
        raise NonceSourceError("dedicated hidden and public values must be distinct")
    return lines[0], lines[1]


def _validate_descriptor(descriptor: int) -> tuple[os.stat_result, tuple[str, str]]:
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise NonceSourceError("dedicated nonce source must be a singly-linked regular file")
    if info.st_uid != os.getuid():
        raise NonceSourceError("dedicated nonce source is not owned by the current uid")
    if stat.S_IMODE(info.st_mode) != 0o600:
        raise NonceSourceError("dedicated nonce source mode must be exactly 0600")
    raw = os.pread(descriptor, 131, 0)
    return info, _validate_payload(raw)


def validate_dedicated(path: Path, *, expected_parent: Path) -> tuple[str, str]:
    parent = _require_private_parent(expected_parent)
    if path.parent.resolve(strict=True) != parent:
        raise NonceSourceError("dedicated nonce source escaped the formal run directory")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise NonceSourceError("dedicated nonce source must be a real non-symlink file") from exc
    try:
        _info, tokens = _validate_descriptor(descriptor)
        return tokens
    finally:
        os.close(descriptor)


def create(source: Path, output: Path, run_root: Path) -> None:
    root = _require_private_parent(run_root)
    if output.parent.resolve(strict=True) != root:
        raise NonceSourceError("dedicated nonce output must be directly under run root")
    if output.exists() or output.is_symlink():
        raise NonceSourceError("refusing to overwrite dedicated nonce source")
    tokens = _matching_nonces(source)
    payload = _canonical_payload(tokens)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=root
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        descriptor = -1
        os.replace(temporary, output)
        directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        validate_dedicated(output, expected_parent=root)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()


def delete(path: Path, run_root: Path) -> None:
    root = _require_private_parent(run_root)
    if path.parent.resolve(strict=True) != root:
        raise NonceSourceError("dedicated nonce source escaped the formal run directory")
    try:
        descriptor = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
    except OSError as exc:
        raise NonceSourceError("dedicated nonce source must be a real non-symlink file") from exc
    try:
        opened, _tokens = _validate_descriptor(descriptor)
        current = os.stat(path, follow_symlinks=False)
        if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
            raise NonceSourceError("dedicated nonce source changed before deletion")
        size = opened.st_size
        os.lseek(descriptor, 0, os.SEEK_SET)
        remaining = size
        zeroes = b"0" * min(size, 4096)
        while remaining:
            written = os.write(descriptor, zeroes[:remaining])
            if written <= 0:
                raise NonceSourceError("failed to overwrite dedicated nonce source")
            remaining -= written
        os.ftruncate(descriptor, size)
        os.fsync(descriptor)
        current = os.stat(path, follow_symlinks=False)
        if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
            raise NonceSourceError("dedicated nonce source changed during deletion")
        path.unlink()
    finally:
        os.close(descriptor)
    directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("--source", required=True, type=Path)
    create_parser.add_argument("--output", required=True, type=Path)
    create_parser.add_argument("--run-root", required=True, type=Path)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--path", required=True, type=Path)
    validate_parser.add_argument("--run-root", required=True, type=Path)
    delete_parser = subparsers.add_parser("delete")
    delete_parser.add_argument("--path", required=True, type=Path)
    delete_parser.add_argument("--run-root", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.action == "create":
        create(args.source, args.output, args.run_root)
    elif args.action == "validate":
        validate_dedicated(args.path, expected_parent=args.run_root)
    elif args.action == "delete":
        delete(args.path, args.run_root)
    else:  # pragma: no cover - argparse enforces this
        raise AssertionError(args.action)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
