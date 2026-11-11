"""Shared builders for the LLM integrity-review tests (not a test module)."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Any

from sforge.harness.llm_judge.client import ChatResult, LLMRequestError
from sforge.harness.task_spec import JudgeSpec, TaskSpec, WorkSpec

CWD = "/app/workdir/lab"
QUERY = (
    "Read AGENT_TASK.md. Implement paged attention in self-contained Triton "
    "in candidates/kernel_vNNN/submission.py and keep FINAL_VERSION on the "
    "best version. Do not call FlashAttention or torch SDPA."
)
SEED = {
    "candidates/kernel_v001/submission.py": "def kernel(q, k, v, out):\n    raise NotImplementedError\n",
    "FINAL_VERSION": "1\n",
    "notes.md": "# notes\n",
}


def make_task(
    *,
    task_id: str = "llm_judge_test_task",
    submit_paths: list[str] | None = None,
    submit_exclude: list[str] | None = None,
    version_enumeration: dict | None | str = "default",
    agent_query: str = QUERY,
) -> TaskSpec:
    if version_enumeration == "default":
        version_enumeration = {
            "root": "candidates",
            "directory_pattern": "kernel_v([0-9]{3})",
            "selector_file": "FINAL_VERSION",
        }
    return TaskSpec(
        task_id=task_id,
        name="LLM judge test task",
        base_image="base",
        platform="linux/amd64",
        cwd=CWD,
        submit_paths=submit_paths or ["candidates", "FINAL_VERSION", "notes.md"],
        submit_exclude=submit_exclude or [".git", "tools", "__pycache__"],
        candidate_identity_exclude=["notes.md"],
        version_enumeration=version_enumeration,  # type: ignore[arg-type]
        work=WorkSpec(specs_dir=CWD, agent_query=agent_query, image_tag="local"),
        judge=JudgeSpec(
            eval_cmd="true",
            eval_timeout=60,
            parser="structured_json",
            image_tag="local",
            selection="valid_then_score",
        ),
        benchmark_name="test",
    )


def make_archive(files: dict[str, str | bytes], *, mtime: int = 0) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        for name, content in sorted(files.items()):
            payload = content.encode() if isinstance(content, str) else content
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = mtime
            info.mode = 0o644
            bundle.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def write_tree(root: Path, files: dict[str, str | bytes]) -> Path:
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            path.write_text(content)
        else:
            path.write_bytes(content)
    return root


def write_documents(root: Path) -> Path:
    """A documents_root holding the agent brief at its container path."""

    brief = root / CWD.lstrip("/") / "AGENT_TASK.md"
    brief.parent.mkdir(parents=True, exist_ok=True)
    brief.write_text(
        "# Task\n\nWrite the whole computation in self-contained Triton. "
        "FlashAttention, torch SDPA and other attention libraries are "
        "forbidden. Do not special-case benchmark shapes.\n"
    )
    return root


def verdict_json(
    verdict: str = "PASS",
    *,
    confidence: float = 0.9,
    violations: list[dict[str, Any]] | None = None,
    feedback: str = "",
) -> str:
    return json.dumps(
        {
            "verdict": verdict,
            "confidence": confidence,
            "violations": violations or [],
            "feedback_to_agent": feedback,
            "notes": "",
        }
    )


class FakeClient:
    """Scripted stand-in for ChatCompletionsClient."""

    def __init__(self, replies: list[str | Exception]) -> None:
        # Shared, not copied: tests may script replies after construction.
        self.replies = replies
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages, *, timeout):
        self.calls.append(messages)
        if not self.replies:
            raise LLMRequestError("no scripted reply left", retryable=False)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return ChatResult(
            content=reply,
            reasoning="thinking...",
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            model="fake-model",
            latency_seconds=0.01,
        )
