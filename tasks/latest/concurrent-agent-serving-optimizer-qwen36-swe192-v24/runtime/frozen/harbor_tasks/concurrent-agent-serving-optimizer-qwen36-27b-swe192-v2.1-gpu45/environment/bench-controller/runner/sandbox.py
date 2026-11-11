from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from runner.corpus import TaskSpec


WORKSPACE_ROOT = Path(os.environ.get("SANDBOX_WORKSPACE_ROOT", "/run/kvbench-workspaces"))
SOCKET_PATH = Path(os.environ.get("SANDBOX_SOCKET", "/run/kvbench-sandbox/control.sock"))
MAX_AGENT_OUTPUT_CHARS = 10_000


def truncate_agent_output(output: object) -> tuple[str, bool]:
    text = str(output)
    if len(text) <= MAX_AGENT_OUTPUT_CHARS:
        return text, False
    head = MAX_AGENT_OUTPUT_CHARS // 2
    tail = MAX_AGENT_OUTPUT_CHARS - head
    return (
        text[:head]
        + "\n[... output elided by benchmark sandbox ...]\n"
        + text[-tail:],
        True,
    )


class TimedSandbox:
    """mini-SWE protocol backed by either bwrap or an immutable SWE image."""

    def __init__(
        self,
        spec: TaskSpec,
        *,
        delay_seed: int,
        minimum_delay: float,
        maximum_delay: float,
        command_timeout_sec: int = 60,
    ) -> None:
        self.spec = spec
        self.delay_seed = delay_seed
        self.minimum_delay = minimum_delay
        self.maximum_delay = maximum_delay
        self.command_timeout_sec = command_timeout_sec
        self.calls = 0
        self.workspace: Path | None = None
        self.runtime_id: str | None = None
        if spec.runtime_image:
            if spec.workspace_template or spec.files:
                raise ValueError("image-backed tasks cannot also provide a host workspace")
            if not spec.runtime_image_digest or not spec.runtime_base_commit or not spec.runtime_head_commit:
                raise ValueError("image-backed tasks require a digest, base commit, and runtime HEAD")
            self.runtime_id = secrets.token_hex(16)
            try:
                self._rpc_request(
                    {
                        "action": "create",
                        "sandbox_id": self.runtime_id,
                        "image": spec.runtime_image,
                        "image_digest": spec.runtime_image_digest,
                        "image_manifest_digest": spec.runtime_manifest_digest,
                        "base_commit": spec.runtime_base_commit,
                        "runtime_head_commit": spec.runtime_head_commit,
                    },
                    timeout=180,
                )
            except Exception:
                self.runtime_id = None
                raise
        else:
            WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
            self.workspace = Path(tempfile.mkdtemp(prefix=f"kvbench-{spec.task_id}-", dir=WORKSPACE_ROOT))
            try:
                self._populate()
            except Exception:
                shutil.rmtree(self.workspace, ignore_errors=True)
                raise

    def _populate(self) -> None:
        if self.workspace is None:
            raise RuntimeError("host workspace is not initialized")
        spec = self.spec
        if spec.workspace_template:
            source = Path(spec.workspace_template).resolve(strict=True)
            subprocess.run(
                ["cp", "-a", "--reflink=auto", f"{source}/.", str(self.workspace)],
                check=True,
            )
            if not (self.workspace / ".git").exists():
                raise RuntimeError(f"staged SWE-bench workspace has no git baseline: {spec.task_id}")
            dirty = subprocess.check_output(
                ["git", "status", "--porcelain=v1"],
                cwd=self.workspace,
                text=True,
            )
            if dirty:
                raise RuntimeError(f"staged SWE-bench workspace is dirty: {spec.task_id}")
        else:
            for relative, content in spec.files.items():
                destination = self.workspace / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(content)
            subprocess.run(["git", "init", "-q", "-b", "main"], cwd=self.workspace, check=True)
            subprocess.run(["git", "add", "--all"], cwd=self.workspace, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=KVBench",
                    "-c",
                    "user.email=kvbench@invalid",
                    "commit",
                    "-q",
                    "-m",
                    "offline baseline",
                ],
                cwd=self.workspace,
                check=True,
            )
        for directory, names, files in os.walk(self.workspace, followlinks=False):
            path = Path(directory)
            if not path.is_symlink():
                path.chmod(0o755)
            for name in [*names, *files]:
                child = path / name
                if child.is_symlink():
                    continue
                child.chmod(0o755 if child.is_dir() else 0o644)

    def _delay(self) -> float:
        if self.maximum_delay <= self.minimum_delay:
            return self.minimum_delay
        digest = hashlib.sha256(f"{self.delay_seed}:{self.spec.task_id}:{self.calls}".encode()).digest()
        fraction = int.from_bytes(digest[:8], "big") / (2**64 - 1)
        return self.minimum_delay + fraction * (self.maximum_delay - self.minimum_delay)

    def _rpc_request(self, request: dict[str, Any], *, timeout: int) -> dict[str, Any]:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(timeout + 10)
            client.connect(str(SOCKET_PATH))
            client.sendall(json.dumps(request, separators=(",", ":")).encode() + b"\n")
            client.shutdown(socket.SHUT_WR)
            chunks = []
            while chunk := client.recv(65536):
                chunks.append(chunk)
                if sum(len(item) for item in chunks) > 3 * 1024 * 1024:
                    raise RuntimeError("sandbox response exceeded the protocol limit")
        response = json.loads(b"".join(chunks))
        if "protocol_error" in response:
            raise RuntimeError(response["protocol_error"])
        return response

    def _rpc(self, command: str, timeout: int) -> dict[str, Any]:
        request: dict[str, Any] = {
            "action": "execute",
            "command": command,
            "timeout_sec": timeout,
        }
        if self.runtime_id is not None:
            request["sandbox_id"] = self.runtime_id
        elif self.workspace is not None:
            request["workspace"] = self.workspace.name
        else:
            raise RuntimeError("sandbox is not initialized")
        return self._rpc_request(request, timeout=timeout)

    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        if self.runtime_id is not None:
            if cwd and cwd not in {"/testbed", "/workspace"}:
                raise ValueError("image sandbox cwd must be /testbed")
        elif cwd and (self.workspace is None or Path(cwd).resolve() != self.workspace.resolve()):
            raise ValueError("sandbox cwd must be the assigned workspace")
        self.calls += 1
        result = self._rpc(command, timeout or self.command_timeout_sec)
        result["output"], result["agent_output_elided"] = truncate_agent_output(result.get("output", ""))
        delay = self._delay()
        if delay:
            time.sleep(delay)
        result["tool_delay_sec"] = delay
        return result

    def get_template_vars(self) -> dict[str, Any]:
        return {"workspace": "/testbed" if self.runtime_id is not None else "/workspace"}

    def verify(self) -> tuple[bool, str]:
        command = self.spec.verification_command or self.spec.test_command
        result = self._rpc(command, self.command_timeout_sec)
        output = result.get("output", "")
        passed = result.get("returncode") == 0
        return passed, output[-2000:]

    def cleanup(self) -> None:
        if self.runtime_id is not None:
            runtime_id = self.runtime_id
            self._rpc_request(
                {"action": "cleanup", "sandbox_id": runtime_id},
                timeout=60,
            )
            self.runtime_id = None
            return
        if self.workspace is not None:
            workspace = self.workspace
            shutil.rmtree(workspace, ignore_errors=True)
            if workspace.exists():
                raise RuntimeError(f"sandbox workspace cleanup failed: {workspace.name}")
            self.workspace = None
