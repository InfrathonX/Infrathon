#!/usr/bin/env python3
"""Materialize immutable admission bytes without reopening candidate paths."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SealedSubmission:
    path: Path
    sha256: str
    size: int


def seal_approved_submission(
    approved: Any,
    directory: Path,
    *,
    owner_uid: int | None = None,
    owner_gid: int | None = None,
    directory_mode: int = 0o700,
) -> SealedSubmission:
    """Write only ``ApprovedWorkspace.submission_bytes`` through ``O_EXCL``."""

    source = approved.submission_bytes
    claimed = approved.submission_sha256
    if not isinstance(source, bytes) or not source:
        raise ValueError("approved submission bytes must be nonempty bytes")
    actual = hashlib.sha256(source).hexdigest()
    if claimed != actual:
        raise ValueError("approved submission digest differs from approved bytes")
    root = Path(directory)
    root.mkdir(parents=True, mode=directory_mode, exist_ok=True)
    metadata = root.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or root.is_symlink():
        raise ValueError("sealed submission root must be a real directory")
    os.chmod(root, directory_mode)
    destination = root / f"submission.{actual}.py"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(destination, flags, 0o400)
    try:
        offset = 0
        while offset < len(source):
            written = os.write(descriptor, source[offset:])
            if written <= 0:
                raise OSError("short write while sealing approved submission")
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
        if owner_uid is not None or owner_gid is not None:
            os.fchown(
                descriptor,
                -1 if owner_uid is None else owner_uid,
                -1 if owner_gid is None else owner_gid,
            )
        sealed = os.fstat(descriptor)
        if (
            not stat.S_ISREG(sealed.st_mode)
            or sealed.st_nlink != 1
            or stat.S_IMODE(sealed.st_mode) != 0o400
            or sealed.st_size != len(source)
        ):
            raise RuntimeError("sealed submission file contract failed")
    except BaseException:
        os.close(descriptor)
        try:
            destination.unlink()
        except OSError:
            pass
        raise
    os.close(descriptor)
    return SealedSubmission(path=destination, sha256=actual, size=len(source))
