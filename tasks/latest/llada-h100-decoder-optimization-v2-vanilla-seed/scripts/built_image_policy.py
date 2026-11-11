"""Trusted host checks for policy bytes in built Judge and Work images."""

from __future__ import annotations

import grp
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


SHA256 = re.compile(r"[0-9a-f]{64}\Z")
REEXEC_MARKER = "SFORGE_DOCKER_GROUP_REEXEC"


def _user_in_docker_group() -> bool:
    try:
        entry = grp.getgrnam("docker")
    except KeyError:
        return False
    try:
        login = os.getlogin()
    except OSError:
        login = ""
    names = {login} | {os.environ.get("USER", ""), os.environ.get("LOGNAME", "")}
    if entry.gr_gid == os.getgid() or entry.gr_gid in os.getgroups():
        return True
    return bool(names & set(entry.gr_mem))


def ensure_docker_group(engine: str = "docker") -> None:
    """Re-exec this validator through `sg docker` when the socket is unreachable.

    `ensure_docker_group` in `scripts/lib.sh` does this for every shell entry
    point: a user who was added to the `docker` group has no docker-group
    credential in an already-running login session. These validators inspect
    the built images over the same socket and need the same treatment, or they
    die with `permission denied while trying to connect to the docker API`
    while following the documented §6.2 commands verbatim.
    """
    if os.environ.get(REEXEC_MARKER) == "1":
        return
    probe = subprocess.run(
        [engine, "info"], capture_output=True, text=True, check=False
    )
    if probe.returncode == 0:
        return
    if not _user_in_docker_group() or shutil.which("sg") is None:
        # Nothing to gain from a re-exec; let the real inspection report why.
        return
    command = [sys.executable, str(Path(sys.argv[0]).resolve()), *sys.argv[1:]]
    quoted = " ".join(shlex.quote(part) for part in command)
    os.environ[REEXEC_MARKER] = "1"
    os.execvp("sg", ["sg", "docker", "-c", f"umask 022; {quoted}"])


def _run(command: list[str], label: str) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"{label} image inspection could not execute") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ValueError(f"{label} image inspection failed: {detail}")
    return completed


def validate_judge_policy_bytes(
    *,
    engine: str,
    image: str,
    container_path: str,
    expected_sha256: str,
    label: str = "Judge",
) -> None:
    if not engine or not image or not container_path.startswith("/"):
        raise ValueError("built Judge image arguments are invalid")
    if SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("tracked policy SHA256 is invalid")
    completed = _run(
        [
            engine,
            "run",
            "--rm",
            "--network",
            "none",
            "--user",
            "0:0",
            "--entrypoint",
            "sha256sum",
            image,
            container_path,
        ],
        label,
    )
    fields = completed.stdout.strip().split()
    if len(fields) != 2 or fields[0] != expected_sha256 or fields[1] != container_path:
        raise ValueError(f"built {label} file bytes differ from the tracked source")


def validate_work_policy_absent(
    *, engine: str, image: str, policy_filename: str
) -> None:
    if not engine or not image or re.fullmatch(r"[a-z_]+\.json", policy_filename) is None:
        raise ValueError("built Work image arguments are invalid")
    completed = _run(
        [
            engine,
            "run",
            "--rm",
            "--network",
            "none",
            "--user",
            "0:0",
            "--entrypoint",
            "find",
            image,
            "/",
            "-xdev",
            "-type",
            "f",
            "-name",
            policy_filename,
            "-print",
        ],
        "Work",
    )
    if completed.stdout.strip():
        raise ValueError(
            f"built Work image exposes forbidden {policy_filename}: "
            f"{completed.stdout.strip()}"
        )


def validate_launcher_memory_contract(
    *,
    launcher_lib: Path,
    policy: Path,
    expected_cap_gib: float,
    expected_headroom_mib: int,
    expected_admission_mib: int,
) -> None:
    if launcher_lib.is_symlink() or not launcher_lib.is_file():
        raise ValueError("launcher memory-policy loader is missing or unsafe")
    completed = _run(
        [
            "bash",
            "-c",
            'source "$1"; load_frozen_memory_policy "$2"; '
            'printf "%s\\t%s\\t%s\\n" "$QUALIFIED_MAX_PEAK_GIB" '
            '"$HEADROOM_MIB" "$GPU_ADMISSION_MIN_FREE_MIB"',
            "built-memory-policy",
            str(launcher_lib),
            str(policy),
        ],
        "launcher",
    )
    fields = completed.stdout.strip().split("\t")
    if len(fields) != 3:
        raise ValueError("launcher did not expose one memory-contract tuple")
    try:
        observed = (float(fields[0]), int(fields[1]), int(fields[2]))
    except ValueError as exc:
        raise ValueError("launcher memory-contract tuple is malformed") from exc
    if observed != (
        float(expected_cap_gib),
        expected_headroom_mib,
        expected_admission_mib,
    ):
        raise ValueError("launcher memory formula differs from Judge/Work policy")
