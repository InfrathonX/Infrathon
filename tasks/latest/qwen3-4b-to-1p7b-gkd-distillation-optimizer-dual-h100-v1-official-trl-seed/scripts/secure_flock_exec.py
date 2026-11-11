#!/usr/bin/env python3
"""Acquire one non-symlink host lock and exec a command with the FD held."""

from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import re
import stat


MARKER_RE = re.compile(r"[A-Z][A-Z0-9_]{0,127}")


def _validate_descriptor(
    *, parent_fd: int, name: str, descriptor: int, current_uid: int
) -> None:
    try:
        observed = os.fstat(descriptor)
        linked = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise SystemExit("inherited lock descriptor is unavailable") from exc
    if not stat.S_ISREG(observed.st_mode) or not stat.S_ISREG(linked.st_mode):
        raise SystemExit("lock path must be a regular non-symlink file")
    if observed.st_uid != current_uid:
        raise SystemExit("lock file is not owned by this user")
    if (observed.st_dev, observed.st_ino) != (linked.st_dev, linked.st_ino):
        raise SystemExit("lock descriptor does not match the requested path")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--fd", required=True, type=int)
    parser.add_argument("--marker", required=True)
    parser.add_argument("--verify-held", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if args.verify_held and args.command:
        parser.error("--verify-held does not accept a command")
    if not args.verify_held and not args.command:
        parser.error("a command is required after --")
    if not args.lock.is_absolute() or args.lock.name in {"", ".", ".."}:
        parser.error("--lock must be an absolute file path")
    if not 3 <= args.fd <= 255:
        parser.error("--fd must be between 3 and 255")
    if MARKER_RE.fullmatch(args.marker) is None:
        parser.error("--marker must be a safe environment variable name")

    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        parent_fd = os.open(args.lock.parent, directory_flags)
    except OSError as exc:
        raise SystemExit(f"unsafe or missing lock parent: {args.lock.parent}") from exc
    try:
        parent_stat = os.fstat(parent_fd)
        if not stat.S_ISDIR(parent_stat.st_mode):
            raise SystemExit("lock parent is not a directory")
        current_uid = os.getuid()
        parent_mode = stat.S_IMODE(parent_stat.st_mode)
        root_sticky = (
            parent_stat.st_uid == 0
            and bool(parent_mode & stat.S_IWOTH)
            and bool(parent_mode & stat.S_ISVTX)
        )
        user_owned_safe = (
            parent_stat.st_uid == current_uid
            and not bool(parent_mode & (stat.S_IWGRP | stat.S_IWOTH))
        )
        if not user_owned_safe and not root_sticky:
            raise SystemExit("lock parent is not private/current-owned or root-sticky")

        if args.verify_held:
            if (
                os.environ.get(args.marker) != "1"
                or os.environ.get("SFORGE_SECURE_LOCK_FD") != str(args.fd)
            ):
                raise SystemExit("inherited lock environment is inconsistent")
            _validate_descriptor(
                parent_fd=parent_fd,
                name=args.lock.name,
                descriptor=args.fd,
                current_uid=current_uid,
            )
            try:
                # Reaffirm an inherited open-file-description lease. If a
                # caller merely forged the environment, this either acquires
                # the lease on its inherited FD or fails against the owner.
                fcntl.flock(args.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise SystemExit(f"inherited lock is not held: {args.lock}") from exc
            return 0

        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(args.lock.name, flags, 0o600, dir_fd=parent_fd)
        try:
            _validate_descriptor(
                parent_fd=parent_fd,
                name=args.lock.name,
                descriptor=descriptor,
                current_uid=current_uid,
            )
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise SystemExit(f"lock is already held: {args.lock}") from exc

            if descriptor != args.fd:
                os.dup2(descriptor, args.fd, inheritable=True)
                os.close(descriptor)
                descriptor = args.fd
            else:
                os.set_inheritable(descriptor, True)
            environment = dict(os.environ)
            environment[args.marker] = "1"
            environment["SFORGE_SECURE_LOCK_FD"] = str(args.fd)
            os.execvpe(args.command[0], args.command, environment)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_fd)


if __name__ == "__main__":
    raise SystemExit(main())
