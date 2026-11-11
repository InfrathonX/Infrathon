"""Review sandboxes on a generic ContainerBackend (the PyroMind path)."""

from __future__ import annotations

import io
import json
import re
import tarfile
import threading
import time
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from _llm_judge_fixtures import (
    CWD,
    SEED,
    make_archive,
    make_task,
    verdict_json,
)
from sforge.harness.llm_judge import reviewer as reviewer_module
from sforge.harness.llm_judge.agent_engine import (
    AgentReviewEngine,
    BackendSandboxProvider,
    extract_runtime_archive,
)
from sforge.harness.llm_judge.bundle import ArchiveBaseline
from sforge.harness.llm_judge.client import LLMRequestError
from sforge.harness.llm_judge.config import LLMJudgeConfig
from sforge.harness.llm_judge.policy import parse_policy_text
from sforge.harness.llm_judge.reviewer import IntegrityReviewer

SUBMITTED = "candidates/kernel_v002/submission.py"
CHEAT = {
    **SEED,
    "FINAL_VERSION": "2\n",
    SUBMITTED: (
        "import torch\n\ndef kernel(q, k, v, out):\n"
        "    out.copy_(torch.nn.functional.scaled_dot_product_attention(q, k, v))\n"
    ),
}
VIOLATION = {
    "rule": "self-contained Triton only",
    "files": [SUBMITTED],
    "evidence": "torch.nn.functional.scaled_dot_product_attention(q, k, v)",
    "explanation": "delegates attention to SDPA",
}
BRIEF = f"{CWD}/AGENT_TASK.md"


def _tar_member(name: str, payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as bundle:
        info = tarfile.TarInfo(name)
        info.size = len(payload)
        bundle.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


class FakeGenericBackend:
    """In-memory ContainerBackend: root exec only, files by absolute path."""

    backend_name = "fake-platform"

    def __init__(self, replies: list[str], *, agent_delay: float = 0.0) -> None:
        self.replies = list(replies)
        self.agent_delay = agent_delay
        self.created: list[SimpleNamespace] = []
        self.cleaned: list[str] = []
        self.commands: list[list[str]] = []
        self.files: dict[str, dict[str, bytes]] = {}
        self.live = 0
        self.peak = 0
        self._lock = threading.Lock()

    def create_container(self, image, name, *, cpu_limit=None, mem_limit=None, **extra):
        assert not extra, extra
        handle = SimpleNamespace(
            id=f"sb-{len(self.created)}", name=name, image=image, cpu=cpu_limit, mem=mem_limit
        )
        with self._lock:
            self.created.append(handle)
            self.files[handle.id] = {}
            self.live += 1
            self.peak = max(self.peak, self.live)
        return handle

    def start_container(self, handle) -> None:
        handle.started = True

    def copy_to_container(self, handle, src, dst) -> None:
        src = Path(src)
        files = self.files[handle.id]
        if src.is_file():
            files[str(dst)] = src.read_bytes()
            return
        for path in src.rglob("*"):
            if path.is_file():
                files[str(PurePosixPath(dst) / path.relative_to(src).as_posix())] = path.read_bytes()

    def exec_run_with_timeout(self, handle, cmd, timeout, **extra):
        assert not extra, "a generic backend gets no user/env/workdir arguments"
        self.commands.append(list(cmd))
        files = self.files[handle.id]
        script = cmd[-1]
        if cmd[0] == "runuser":
            if self.agent_delay:
                time.sleep(self.agent_delay)
            turn = re.search(r"last-(\d+)\.txt", script)
            assert turn, script
            files[f"/home/agent/.sforge-review/last-{turn.group(1)}.txt"] = self.replies.pop(0).encode()
            return SimpleNamespace(output="", exit_code=0, timed_out=False)
        extracted = re.match(r"tar -xf '?([^' ]+)'? -C / ", script)
        if extracted:
            with tarfile.open(fileobj=io.BytesIO(files.pop(extracted.group(1))), mode="r") as bundle:
                for member in bundle:
                    if member.isfile():
                        stream = bundle.extractfile(member)
                        assert stream is not None
                        files["/" + member.name] = stream.read()
            return SimpleNamespace(output="", exit_code=0, timed_out=False)
        output = "/home/agent" if "getent passwd" in script else ""
        return SimpleNamespace(output=output, exit_code=0, timed_out=False)

    def copy_from_container(self, handle, src) -> bytes:
        files = self.files[handle.id]
        if str(src) not in files:
            raise FileNotFoundError(str(src))
        return _tar_member(PurePosixPath(src).name, files[str(src)])

    def cleanup_container(self, handle) -> None:
        with self._lock:
            self.cleaned.append(handle.id)
            self.live -= 1


def _runtime(tmp_path: Path) -> Path:
    root = tmp_path / "runtime"
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "codex").write_text("#!/bin/sh\n")
    (root / "env.sh").write_text("export PATH=/opt/agent-runtime/bin:$PATH\n")
    (root / ".sforge-agent-runtime.json").write_text(
        json.dumps(
            {"schema_version": 1, "kind": "sforge-agent-runtime", "agent": "codex",
             "cli": "codex", "version": "9.9.9"}
        )
    )
    return root


def _config(tmp_path: Path, **overrides) -> LLMJudgeConfig:
    key = tmp_path / "model.key"
    key.write_text("sk-platform-review-0123456789\n")
    values = {
        "mode": "enforce",
        "engine": "agent",
        "agent_runtime_dir": _runtime(tmp_path),
        "api_base_url": "http://model.internal:8000/v1",
        "api_key_file": key,
        "model": "gpt-review",
        "network": "platform",
        "cpu_limit": 4,
        "mem_limit": "8Gi",
        "concurrency": 1,
    }
    values.update(overrides)
    return LLMJudgeConfig(**values)


def _engine(tmp_path: Path, backend: FakeGenericBackend, **overrides) -> AgentReviewEngine:
    provider = BackendSandboxProvider(
        backend,
        resolve_image=lambda task_spec, key: "registry/judge@sha256:" + "a" * 64,
        name_prefix="sf-run-review",
        runtime_installer=extract_runtime_archive("/mnt/sforge-review-runtime/runtime.tar.gz"),
        runtime_identity={"runtime_archive_sha256": "b" * 64},
    )
    return AgentReviewEngine(_config(tmp_path, **overrides), sandboxes=provider)


def _baseline(task) -> ArchiveBaseline:
    policy = parse_policy_text(task.task_id, "Triton only.\n", source=Path("POLICY.md"))
    return ArchiveBaseline.capture(
        task,
        policy,
        make_archive(SEED),
        lambda path: b"Write a self-contained Triton kernel.\n" if path == BRIEF else None,
        source="initial-checkpoint:checkpoint-1",
    )


def _review(tmp_path: Path, engine: AgentReviewEngine, label: str = "a"):
    task = make_task()
    tasks_dir = tmp_path / label / "task" / "tasks"
    tasks_dir.mkdir(parents=True)
    reviewer = IntegrityReviewer(engine.config, tasks_dir=tasks_dir, cache_dir=None, engine=engine)
    return reviewer.review(
        task_spec=task,
        archive=make_archive(CHEAT),
        baseline=_baseline(task),
        log_dir=tmp_path / label / "submission",
    )


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reviewer_module.time, "sleep", lambda seconds: None)


def test_platform_sandbox_reviews_on_the_cpu_judge_image_as_the_unprivileged_user(
    tmp_path: Path,
) -> None:
    backend = FakeGenericBackend(
        [verdict_json("VIOLATION", confidence=0.95, violations=[VIOLATION], feedback="Use Triton.")]
    )
    review = _review(tmp_path, _engine(tmp_path, backend))

    assert review.status == "violation" and review.blocking
    (handle,) = backend.created
    assert handle.image == "registry/judge@sha256:" + "a" * 64
    assert handle.name.startswith("sf-run-review-")
    assert (handle.cpu, handle.mem) == (4, "8Gi")
    assert backend.cleaned == [handle.id]
    roots = [cmd[-1] for cmd in backend.commands if cmd[0] != "runuser"]
    assert any("tar -xzf /mnt/sforge-review-runtime/runtime.tar.gz -C /opt/agent-runtime" in s for s in roots)
    files = backend.files[handle.id]
    assert files[f"/review/submission/{SUBMITTED}"] == CHEAT[SUBMITTED].encode()
    assert files[f"{CWD}/{SUBMITTED}"] == CHEAT[SUBMITTED].encode()
    assert "/review/baseline/candidates/kernel_v001/submission.py" in files
    assert b"self-contained Triton" in files[f"/review/task{BRIEF}"]
    config_toml = files["/home/agent/.codex/config.toml"].decode()
    assert 'base_url = "http://model.internal:8000/v1"' in config_toml
    # The agent runs unprivileged; its key comes from a private file, never argv.
    (agent,) = [cmd for cmd in backend.commands if cmd[0] == "runuser"]
    assert agent[:4] == ["runuser", "-u", "agent", "--"]
    assert "codex exec -C /review" in agent[-1]
    assert all("sk-platform-review" not in part for cmd in backend.commands for part in cmd)
    env_file = files["/home/agent/.sforge-review/exec.env"].decode()
    assert "SFORGE_REVIEW_API_KEY=sk-platform-review-0123456789" in env_file
    assert "NVIDIA_VISIBLE_DEVICES=void" in env_file


def test_identity_pins_the_platform_image_digest_and_runtime_archive(tmp_path: Path) -> None:
    engine = _engine(tmp_path, FakeGenericBackend([]))
    identity = engine.identity(make_task())
    assert identity["image_id"] == "registry/judge@sha256:" + "a" * 64
    assert identity["runtime_archive_sha256"] == "b" * 64
    assert engine.describe()["network"] == "platform"


def test_concurrency_caps_live_review_sandboxes(tmp_path: Path) -> None:
    backend = FakeGenericBackend([verdict_json("PASS"), verdict_json("PASS")], agent_delay=0.2)
    engine = _engine(tmp_path, backend)
    results = []
    threads = [
        threading.Thread(target=lambda label=label: results.append(_review(tmp_path, engine, label)))
        for label in ("a", "b")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert [review.status for review in results] == ["pass", "pass"]
    assert backend.peak == 1 and len(backend.cleaned) == 2


def test_a_session_that_cannot_get_a_slot_fails_without_a_sandbox(tmp_path: Path) -> None:
    backend = FakeGenericBackend([])
    engine = _engine(tmp_path, backend)
    assert engine.acquire_slot(0)
    try:
        with pytest.raises(LLMRequestError, match="slot"):
            engine.open(
                task_spec=make_task(), archive=b"", bundle=None, overview=None, policy=None,
                out_dir=tmp_path, secrets=[], deadline=time.monotonic() + 5.5,
            )
    finally:
        engine.release_slot()
    assert backend.created == []


def test_failed_setup_releases_the_sandbox_and_the_slot(tmp_path: Path) -> None:
    backend = FakeGenericBackend([])

    def broken(sandbox) -> None:
        raise RuntimeError("runtime archive missing")

    provider = BackendSandboxProvider(
        backend,
        resolve_image=lambda task_spec, key: "img",
        runtime_installer=broken,
    )
    engine = AgentReviewEngine(_config(tmp_path), sandboxes=provider)
    with pytest.raises(RuntimeError, match="runtime archive missing"):
        engine.open(
            task_spec=make_task(), archive=b"", bundle=None, overview=None, policy=None,
            out_dir=tmp_path, secrets=[], deadline=time.monotonic() + 60,
        )
    assert backend.cleaned == ["sb-0"]
    assert engine.acquire_slot(0)


def test_archive_baseline_uses_the_saved_starting_files_and_captured_briefs() -> None:
    task = make_task()
    policy = parse_policy_text(
        task.task_id,
        "---\ninstruction_files: ['/app/docs/INTERFACE.md']\n---\nTriton only.\n",
        source=Path("POLICY.md"),
    )
    reads = []

    def read(path: str):
        reads.append(path)
        return b"brief" if path == BRIEF else None

    baseline = ArchiveBaseline.capture(task, policy, make_archive(SEED), read, source="seed")
    snapshot = baseline.snapshot(task, policy)
    assert "/app/docs/INTERFACE.md" in reads and BRIEF in reads
    assert snapshot.files and set(snapshot.files) <= set(SEED)
    assert snapshot.documents == {BRIEF: "brief"}
    assert any("INTERFACE.md" in warning for warning in snapshot.warnings)
    assert snapshot.source == "seed"
