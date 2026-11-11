#!/usr/bin/env python3
"""Precisely reconcile Judge resources owned by one formal launcher run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
from typing import Any


TASK_ID = "concurrent_agent_serving_optimizer_qwen36_swe192_v24"
BENCHMARK = "kvbench_qwen36"
ROLE_LABEL = "mlsysforge.kvbench.role"
ROLE_VALUE = "trusted-judge"
NONCE_DESTINATION = "/run/kvbench-secrets/nonce-source.jsonl"
PROJECT = "kvbench-qwen36-swe192-v2-1-gpu67-validation"
SANDBOX_OWNER_LABEL = "org.harbor-kv-scheduler.swe-sandbox.owner"
SANDBOX_OWNER = "qwen36-swe192-v2-1-gpu67-validation"
STACK_RELATIVE = Path(
    "runtime/frozen/harbor_run_scripts/"
    "concurrent-agent-serving-optimizer-qwen36-27b-swe192-v2.1-gpu45/"
    "gpu67-v2.1-validation-stack.sh"
)
ORACLE_RELATIVE = Path("oracle/reference_gateway")
HIDDEN_RUNNER_BASENAME = "run-gpu67-v2.2-hidden-candidate-only.py"


class CleanupError(RuntimeError):
    """Cleanup could not prove ownership or runtime quiescence."""


def _run(
    command: list[str],
    *,
    timeout: float = 60.0,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )
    if check and completed.returncode != 0:
        raise CleanupError(
            f"command failed ({completed.returncode}): {command[0]} {command[1] if len(command) > 1 else ''}"
        )
    return completed


def _real_owned_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise CleanupError(f"{label} must be a regular non-symlink file")
    resolved = path.resolve(strict=True)
    info = resolved.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
        raise CleanupError(f"{label} must be owned by the current uid")
    if stat.S_IMODE(info.st_mode) != 0o600:
        raise CleanupError(f"{label} mode must be 0600")
    return resolved


def _real_directory(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise CleanupError(f"{label} must be a real directory")
    return path.resolve(strict=True)


def _mounts_by_destination(inspect: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mounts = inspect.get("Mounts")
    if not isinstance(mounts, list):
        raise CleanupError("Judge inspect Mounts is malformed")
    result: dict[str, dict[str, Any]] = {}
    for mount in mounts:
        if not isinstance(mount, dict) or not isinstance(mount.get("Destination"), str):
            raise CleanupError("Judge mount entry is malformed")
        destination = mount["Destination"]
        if destination in result:
            raise CleanupError("Judge has duplicate mount destinations")
        result[destination] = mount
    return result


def owned_judge(
    inspect: dict[str, Any],
    *,
    nonce_source: Path,
    migration_root: Path,
    eval_root: Path,
    judge_image: str,
) -> dict[str, str] | None:
    """Return validated ownership metadata, or None for an unrelated container."""
    labels = ((inspect.get("Config") or {}).get("Labels") or {})
    if labels.get(ROLE_LABEL) != ROLE_VALUE:
        return None
    name = inspect.get("Name")
    expected_prefix = f"/{BENCHMARK}.judge.{TASK_ID}."
    if not isinstance(name, str) or not name.startswith(expected_prefix):
        return None
    suffix = name.removeprefix(expected_prefix)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,191}", suffix):
        raise CleanupError("owned Judge name has an unsafe submission suffix")
    config = inspect.get("Config")
    if not isinstance(config, dict) or config.get("Image") != judge_image:
        raise CleanupError("owned Judge image does not match the pinned task image")
    identifier = inspect.get("Id")
    if not isinstance(identifier, str) or not re.fullmatch(r"[0-9a-f]{64}", identifier):
        raise CleanupError("owned Judge has an invalid full container ID")
    mounts = _mounts_by_destination(inspect)
    nonce_mount = mounts.get(NONCE_DESTINATION)
    if (
        not isinstance(nonce_mount, dict)
        or nonce_mount.get("Type") != "bind"
        or Path(str(nonce_mount.get("Source", ""))) != nonce_source
        or nonce_mount.get("RW") is not False
    ):
        return None
    migration_mount = mounts.get(str(migration_root))
    if (
        not isinstance(migration_mount, dict)
        or migration_mount.get("Type") != "bind"
        or Path(str(migration_mount.get("Source", ""))) != migration_root
        or migration_mount.get("RW") is not False
    ):
        raise CleanupError("owned Judge migration mount contract mismatch")
    scratch_candidates = []
    for destination, mount in mounts.items():
        source = Path(str(mount.get("Source", "")))
        if source.parent == eval_root and re.fullmatch(r"judge-[0-9a-f]{24}", source.name):
            scratch_candidates.append((destination, source, mount))
    if len(scratch_candidates) != 1:
        raise CleanupError("owned Judge must have exactly one task scratch mount")
    destination, scratch, scratch_mount = scratch_candidates[0]
    if (
        destination != str(scratch)
        or scratch_mount.get("Type") != "bind"
        or scratch_mount.get("RW") is not True
    ):
        raise CleanupError("owned Judge scratch mount contract mismatch")
    return {"id": identifier, "name": name, "scratch": str(scratch)}


class DockerRuntime:
    def __init__(
        self,
        *,
        nonce_source: Path,
        migration_root: Path,
        eval_root: Path,
        judge_image: str,
    ) -> None:
        self.nonce_source = _real_owned_file(nonce_source, "dedicated nonce source")
        self.migration_root = _real_directory(migration_root, "migration root")
        self.eval_root = _real_directory(eval_root, "Judge eval root")
        self.judge_image = judge_image

    def inspect(self, identifiers: list[str]) -> list[dict[str, Any]]:
        if not identifiers:
            return []
        completed = _run(["docker", "inspect", *identifiers], timeout=60)
        try:
            value = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise CleanupError("docker inspect returned malformed JSON") from exc
        if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
            raise CleanupError("docker inspect result is malformed")
        return value

    def owned(self) -> list[dict[str, str]]:
        listed = _run(
            [
                "docker",
                "ps",
                "--all",
                "--quiet",
                "--no-trunc",
                "--filter",
                f"label={ROLE_LABEL}={ROLE_VALUE}",
            ],
            timeout=60,
        ).stdout.split()
        result = []
        for inspect in self.inspect(listed):
            owned = owned_judge(
                inspect,
                nonce_source=self.nonce_source,
                migration_root=self.migration_root,
                eval_root=self.eval_root,
                judge_image=self.judge_image,
            )
            if owned is not None:
                result.append(owned)
        return sorted(result, key=lambda item: item["id"])

    def revalidate(self, identifier: str) -> dict[str, str]:
        inspected = self.inspect([identifier])
        if len(inspected) != 1:
            raise CleanupError("owned Judge disappeared before revalidation")
        result = owned_judge(
            inspected[0],
            nonce_source=self.nonce_source,
            migration_root=self.migration_root,
            eval_root=self.eval_root,
            judge_image=self.judge_image,
        )
        if result is None or result["id"] != identifier:
            raise CleanupError("Judge ownership changed before mutation")
        return result

    @staticmethod
    def _runner_action(identifier: str, action: str) -> int:
        source = r'''import os, pathlib, signal, sys
action = sys.argv[1]
needle = b"run-gpu67-v2.2-hidden-candidate-only.py"
matches = []
for entry in pathlib.Path("/proc").iterdir():
    if not entry.name.isdigit() or int(entry.name) == os.getpid():
        continue
    try:
        raw = (entry / "cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    if needle in raw:
        matches.append(int(entry.name))
if action != "count":
    selected = {"int": signal.SIGINT, "term": signal.SIGTERM, "kill": signal.SIGKILL}[action]
    for pid in matches:
        try:
            os.kill(pid, selected)
        except ProcessLookupError:
            pass
print(len(matches))
'''
        completed = _run(
            [
                "docker",
                "exec",
                "--user",
                "0",
                identifier,
                "python3",
                "-c",
                source,
                action,
            ],
            timeout=30,
            check=False,
        )
        if completed.returncode != 0:
            return 0
        try:
            return int(completed.stdout.strip().splitlines()[-1])
        except (IndexError, ValueError):
            raise CleanupError("could not determine hidden runner process count")

    def _wait_runner(self, identifier: str, seconds: float) -> bool:
        deadline = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < deadline:
            if self._runner_action(identifier, "count") == 0:
                return True
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
        return self._runner_action(identifier, "count") == 0

    def stop_hidden_runner(self, identifier: str, grace_seconds: float) -> None:
        if self._runner_action(identifier, "count") == 0:
            return
        self._runner_action(identifier, "int")
        if self._wait_runner(identifier, grace_seconds):
            return
        self._runner_action(identifier, "term")
        if self._wait_runner(identifier, 30.0):
            return
        self._runner_action(identifier, "kill")
        if not self._wait_runner(identifier, 10.0):
            raise CleanupError("hidden runner survived bounded INT/TERM/KILL")

    def stop_stack(self, identifier: str, scratch: Path, timeout: float) -> None:
        stack = self.migration_root / STACK_RELATIVE
        oracle = self.migration_root / ORACLE_RELATIVE
        if not stack.is_file() or stack.is_symlink() or not os.access(stack, os.X_OK):
            raise CleanupError("frozen stack script is unavailable")
        if not (oracle / "launch.sh").is_file() or (oracle / "launch.sh").is_symlink():
            raise CleanupError("frozen Oracle fallback is unavailable")
        candidate = scratch / "candidate"
        probe = _run(
            [
                "docker",
                "exec",
                "--user",
                "0",
                identifier,
                "sh",
                "-c",
                'test -d "$1" && test ! -L "$1" && test -x "$1/launch.sh"',
                "sh",
                str(candidate),
            ],
            timeout=30,
            check=False,
        )
        selected = candidate if probe.returncode == 0 else oracle
        completed = _run(
            [
                "docker",
                "exec",
                "--user",
                "0",
                "--env",
                f"KV_AUTHOR_REFERENCE_GATEWAY_DIR={selected}",
                "--env",
                "KV_GPU67_V21_OWNER_TOKEN=gpu67-v2-2-candidate-only",
                identifier,
                str(stack),
                "stop",
            ],
            timeout=timeout,
            check=False,
        )
        if completed.returncode != 0:
            raise CleanupError("exact frozen stack stop failed")

    def stop_stack_from_host(self, timeout: float) -> None:
        """Compensate after SForge already removed the owned outer Judge.

        The formal launcher calls this helper only after stopping its Judge
        server and while owning the task-wide formal lease/measurement handoff.
        The fixed project and sandbox owner are therefore scoped to this task;
        the persistent host-vLLM container is intentionally not referenced.
        """
        stack = self.migration_root / STACK_RELATIVE
        oracle = self.migration_root / ORACLE_RELATIVE
        if not stack.is_file() or stack.is_symlink() or not os.access(stack, os.X_OK):
            raise CleanupError("frozen host stack script is unavailable")
        if not (oracle / "launch.sh").is_file() or (oracle / "launch.sh").is_symlink():
            raise CleanupError("frozen Oracle host fallback is unavailable")
        environment = {
            **os.environ,
            "KV_AUTHOR_REFERENCE_GATEWAY_DIR": str(oracle),
            "KV_GPU67_V21_OWNER_TOKEN": "gpu67-v2-2-candidate-only",
        }
        completed = subprocess.run(
            [str(stack), "stop"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
            env=environment,
        )
        if completed.returncode != 0:
            raise CleanupError("exact host fallback stack stop failed")

    def clear_scratch(self, identifier: str, scratch: Path) -> None:
        source = r'''import pathlib, shutil, sys
root = pathlib.Path(sys.argv[1]).resolve(strict=True)
scratch = pathlib.Path(sys.argv[2]).resolve(strict=True)
if scratch.parent != root or not scratch.name.startswith("judge-") or scratch.is_symlink():
    raise SystemExit("unsafe scratch")
for child in scratch.iterdir():
    if child.is_symlink() or child.is_file():
        child.unlink()
    elif child.is_dir():
        shutil.rmtree(child)
    else:
        raise SystemExit("special scratch entry")
'''
        _run(
            [
                "docker",
                "exec",
                "--user",
                "0",
                identifier,
                "python3",
                "-c",
                source,
                str(self.eval_root),
                str(scratch),
            ],
            timeout=120,
        )

    @staticmethod
    def _running(identifier: str) -> bool:
        completed = _run(
            ["docker", "inspect", "--format", "{{.State.Running}}", identifier],
            timeout=30,
            check=False,
        )
        return completed.returncode == 0 and completed.stdout.strip() == "true"

    def stop_outer(self, identifier: str, stop_seconds: float) -> None:
        self.revalidate(identifier)
        if self._running(identifier):
            _run(
                ["docker", "kill", "--signal", "TERM", identifier],
                timeout=30,
                check=False,
            )
            deadline = time.monotonic() + stop_seconds
            while self._running(identifier) and time.monotonic() < deadline:
                time.sleep(0.5)
            if self._running(identifier):
                _run(
                    ["docker", "kill", "--signal", "KILL", identifier],
                    timeout=30,
                    check=False,
                )
                deadline = time.monotonic() + 10.0
                while self._running(identifier) and time.monotonic() < deadline:
                    time.sleep(0.25)
                if self._running(identifier):
                    raise CleanupError("owned Judge survived bounded TERM/KILL")
        self.revalidate(identifier)
        _run(["docker", "rm", identifier], timeout=60)

    @staticmethod
    def _container_ids(label: str) -> list[str]:
        return _run(
            ["docker", "ps", "--all", "--quiet", "--no-trunc", "--filter", label],
            timeout=60,
        ).stdout.split()

    def remove_owned_sandboxes(self) -> int:
        identifiers = self._container_ids(
            f"label={SANDBOX_OWNER_LABEL}={SANDBOX_OWNER}"
        )
        if not identifiers:
            return 0
        inspected = self.inspect(identifiers)
        validated: list[str] = []
        for item in inspected:
            identifier = item.get("Id")
            labels = ((item.get("Config") or {}).get("Labels") or {})
            if (
                not isinstance(identifier, str)
                or not re.fullmatch(r"[0-9a-f]{64}", identifier)
                or labels.get(SANDBOX_OWNER_LABEL) != SANDBOX_OWNER
            ):
                raise CleanupError("sandbox ownership changed before removal")
            validated.append(identifier)
        _run(["docker", "rm", "--force", *validated], timeout=180)
        return len(validated)

    def require_runtime_quiescent(self) -> None:
        compose = self._container_ids(f"label=com.docker.compose.project={PROJECT}")
        sandboxes = self._container_ids(
            f"label={SANDBOX_OWNER_LABEL}={SANDBOX_OWNER}"
        )
        if compose or sandboxes:
            raise CleanupError(
                f"validation runtime is not quiescent (compose={len(compose)}, sandboxes={len(sandboxes)})"
            )

    def runtime_is_quiescent(self) -> bool:
        try:
            self.require_runtime_quiescent()
        except CleanupError:
            return False
        return True

    def cleanup(self, *, runner_grace: float, stack_timeout: float, stop_timeout: float) -> dict[str, Any]:
        owned = self.owned()
        errors: list[str] = []
        warnings: list[str] = []
        # Never reconcile the shared fixed project while an owned hidden runner
        # remains alive. INT gives the frozen finally path its full stack grace;
        # TERM/KILL are bounded last resorts.
        for item in owned:
            identifier = item["id"]
            try:
                self.revalidate(identifier)
                self.stop_hidden_runner(identifier, runner_grace)
            except Exception as exc:
                errors.append(
                    f"{identifier[:12]}:runner:{type(exc).__name__}:{exc}"
                )
        if errors:
            remaining = self.owned()
            return {
                "schema_version": 1,
                "owned_judges": len(owned),
                "remaining_owned_judges": len(remaining),
                "status": "failed",
                "warnings": warnings,
                "errors": errors,
            }

        stack_failed = False
        for item in owned:
            identifier = item["id"]
            scratch = Path(item["scratch"])
            try:
                self.stop_stack(identifier, scratch, stack_timeout)
            except Exception as exc:
                stack_failed = True
                warnings.append(
                    f"{identifier[:12]}:inner_stack:{type(exc).__name__}:{exc}"
                )
        if stack_failed or not self.runtime_is_quiescent():
            try:
                self.stop_stack_from_host(stack_timeout)
            except Exception as exc:
                errors.append(f"host_stack:{type(exc).__name__}:{exc}")
        if not errors:
            try:
                removed = self.remove_owned_sandboxes()
                if removed:
                    warnings.append(f"owner_scoped_sandbox_fallback_removed:{removed}")
            except Exception as exc:
                errors.append(f"sandboxes:{type(exc).__name__}:{exc}")

        # Scratch and outer-container deletion happens only after the exact
        # stack compensation above, so deleting PID1 cannot strand teardown.
        for item in owned:
            identifier = item["id"]
            scratch = Path(item["scratch"])
            try:
                self.clear_scratch(identifier, scratch)
                self.stop_outer(identifier, stop_timeout)
                if scratch.exists():
                    scratch.rmdir()
            except Exception as exc:
                errors.append(f"{identifier[:12]}:outer:{type(exc).__name__}:{exc}")
        try:
            self.require_runtime_quiescent()
        except Exception as exc:
            errors.append(f"runtime:{type(exc).__name__}:{exc}")
        remaining = self.owned()
        if remaining:
            errors.append(f"owned_judges_remaining:{len(remaining)}")
        return {
            "schema_version": 1,
            "owned_judges": len(owned),
            "remaining_owned_judges": len(remaining),
            "status": "passed" if not errors else "failed",
            "warnings": warnings,
            "errors": errors,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("list", "cleanup", "verify-quiescent"))
    parser.add_argument("--nonce-source", required=True, type=Path)
    parser.add_argument("--migration-root", required=True, type=Path)
    parser.add_argument("--eval-root", required=True, type=Path)
    parser.add_argument("--judge-image", required=True)
    parser.add_argument("--runner-grace-seconds", type=float, default=330.0)
    parser.add_argument("--stack-timeout-seconds", type=float, default=360.0)
    parser.add_argument("--stop-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime = DockerRuntime(
        nonce_source=args.nonce_source,
        migration_root=args.migration_root,
        eval_root=args.eval_root,
        judge_image=args.judge_image,
    )
    if args.action == "list":
        payload: Any = runtime.owned()
        status = 0
    elif args.action == "verify-quiescent":
        runtime.require_runtime_quiescent()
        payload = {"schema_version": 1, "status": "passed"}
        status = 0
    else:
        payload = runtime.cleanup(
            runner_grace=args.runner_grace_seconds,
            stack_timeout=args.stack_timeout_seconds,
            stop_timeout=args.stop_timeout_seconds,
        )
        status = 0 if payload["status"] == "passed" else 1
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(encoded)
    else:
        args.output.write_text(encoded, encoding="utf-8")
        args.output.chmod(0o600)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
