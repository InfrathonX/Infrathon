"""Secure host-to-container file injection helpers."""

from __future__ import annotations

import re
import shlex
from pathlib import Path, PurePosixPath

from sforge.harness.backend import ContainerBackend, ContainerHandle


_OWNER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*:[A-Za-z_][A-Za-z0-9_-]*\Z")


def copy_container_file(
    backend: ContainerBackend,
    handle: ContainerHandle,
    source: Path,
    destination: PurePosixPath,
    *,
    owner: str,
    mode: int,
    description: str,
) -> None:
    """Copy one file, then fail closed unless its exact owner/mode is set.

    Docker ``put_archive`` and Kubernetes tar extraction preserve host tar
    metadata.  That is unsafe to rely on: a harness launched with umask 077
    otherwise injects mode-0600 files owned by the host uid.  Perform the
    ownership/mode transition as root and verify the resulting metadata in the
    same checked container command.
    """
    if (
        not destination.is_absolute()
        or destination == PurePosixPath("/")
        or ".." in destination.parts
    ):
        raise ValueError(f"unsafe container destination: {destination}")
    if not _OWNER_RE.fullmatch(owner):
        raise ValueError(f"invalid container owner: {owner!r}")
    if not isinstance(mode, int) or isinstance(mode, bool) or not 0 <= mode <= 0o777:
        raise ValueError(f"invalid container mode: {mode!r}")

    backend.copy_to_container(handle, source, destination)

    quoted_path = shlex.quote(str(destination))
    quoted_owner = shlex.quote(owner)
    expected = shlex.quote(f"{owner}:{mode:o}")
    command = (
        "set -eu; "
        f"chown -- {quoted_owner} {quoted_path}; "
        f"chmod -- {mode:04o} {quoted_path}; "
        f"actual=$(stat -c '%U:%G:%a' -- {quoted_path}); "
        f"test \"$actual\" = {expected}"
    )
    result = backend.exec_run_with_exit_code(
        handle,
        ["/bin/sh", "-c", command],
        timeout=30,
        user="root",
    )
    if result.timed_out or result.exit_code != 0:
        state = "timed out" if result.timed_out else f"exit {result.exit_code}"
        raise RuntimeError(
            f"Failed to secure {description} ({destination}; {state}): "
            f"{result.output}"
        )
