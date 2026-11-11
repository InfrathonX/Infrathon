#!/usr/bin/env python3
"""Separate-GPU entry for the dual-H100 ORPO task (host-only).

The default launch follows the paper's setup: Work runs on one GPU pair
(ORPO_GPU_UUIDS) and the Judge on a second pair (ORPO_JUDGE_GPU_UUIDS), and
Work keeps running while the Judge measures.  The launcher starts the Judge
server through this wrapper only in that separate mode; shared mode
(``--shared-gpus``), the archive Judge (eval_archive.sh), candidate replay and
the managed smoke keep calling sforge_dual_h100_entry.py directly.

sforge_dual_h100_entry.py stays byte-identical: its bytes feed the replay
evaluator fingerprint (replay_candidates.sh).  This wrapper loads it by path
and swaps two pieces:

* GPU pair per role: a Judge container gets the Judge pair, a Work container
  the Work pair.  Every other part of the container contract (mounts, data
  partitions, network isolation, CUDA devices ``0,1``) is the entry's own.
* Managed Judge: the per-pair managed-Judge lease is taken on the Judge pair
  and the idle wait looks at the Judge pair.  Work is never looked up, paused
  or resumed.
"""

from __future__ import annotations

import contextvars
import importlib.util
import os
import sys
import time
from contextlib import contextmanager
from fcntl import LOCK_EX, LOCK_UN, flock
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator


ENTRY_PATH = Path(__file__).resolve().with_name("sforge_dual_h100_entry.py")
ENTRY_MODULE_NAME = "orpo_sforge_dual_h100_entry"
JUDGE_GPU_UUID_ENV = "ORPO_JUDGE_GPU_UUIDS"

_role: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "orpo_separate_gpu_role", default=None
)


def load_entry(
    path: Path = ENTRY_PATH, module_name: str = ENTRY_MODULE_NAME
) -> ModuleType:
    """Import the unchanged entry by path (its ``__main__`` block does not run)."""

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load the task entry: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gpu_pairs(entry: ModuleType) -> tuple[list[str], list[str]]:
    """The Work pair and the Judge pair: two UUIDs each, all four distinct."""

    work = entry._original_gpu_uuids()
    raw = os.environ.get(JUDGE_GPU_UUID_ENV, "").strip()
    if not raw:
        raise RuntimeError(
            f"separate-GPU mode needs {JUDGE_GPU_UUID_ENV}: the two physical "
            "GPU UUIDs of the Judge pair, comma separated"
        )
    try:
        judge = entry._original_gpu_uuids(raw)
    except RuntimeError as exc:
        # The entry's message names the Work variable; this value is the Judge's.
        raise RuntimeError(
            str(exc).replace(entry.GPU_UUID_ENV, JUDGE_GPU_UUID_ENV)
        ) from exc
    if {uuid.casefold() for uuid in work} & {uuid.casefold() for uuid in judge}:
        raise RuntimeError(
            f"the Judge pair ({JUDGE_GPU_UUID_ENV}) overlaps the Work pair "
            f"({entry.GPU_UUID_ENV}); separate-GPU mode needs four distinct GPUs"
        )
    return work, judge


def install(entry: ModuleType) -> ModuleType:
    """Patch the loaded entry for separate Work and Judge pairs."""

    if getattr(entry, "_separate_gpu_installed", False):
        return entry
    if entry._replay_attestation_enabled():
        raise RuntimeError(
            "candidate replay runs on one attested pair through "
            "sforge_dual_h100_entry.py, not through the separate-GPU entry"
        )
    entry._original_gpu_uuids = entry._gpu_uuids
    gpu_pairs(entry)  # fail before serving anything

    def role_gpu_uuids(raw: str | None = None) -> list[str]:
        if raw is not None:
            return entry._original_gpu_uuids(raw)
        role = _role.get()
        if role is None:
            raise RuntimeError(
                "separate-GPU mode: a GPU pair was requested outside a Work "
                "or Judge container"
            )
        work, judge = gpu_pairs(entry)
        return judge if role == "judge" else work

    original_create_container = entry.dual_h100_create_container

    def separate_create_container(
        self: Any, image: str, name: str, **kwargs: Any
    ) -> Any:
        """Create one role container on its own pair (Judge pair or Work pair)."""

        token = _role.set("judge" if entry._is_judge(image, name) else "work")
        try:
            return original_create_container(self, image, name, **kwargs)
        finally:
            _role.reset(token)

    @contextmanager
    def separate_managed_judge(
        *, backend: Any, task_id: str, run_id: str
    ) -> Iterator[None]:
        """Lease the Judge pair and wait for it to go idle; Work keeps running."""

        if getattr(backend, "backend_name", None) != "docker":
            raise entry.ManagedJudgeLifecycleError("unsupported_backend")
        if getattr(backend, "client", None) is None:
            raise entry.ManagedJudgeLifecycleError("docker_client_unavailable")
        entry._work_container_name(task_id, run_id)  # run identity check only
        timeout = entry._managed_idle_timeout()
        _, judge = gpu_pairs(entry)
        lock_path = entry._managed_judge_lock_path(judge)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with entry._managed_judge_thread_lock, lock_path.open("a+") as lock_file:
            flock(lock_file.fileno(), LOCK_EX)
            try:
                # As in shared mode, the idle wait starts once the pair is
                # ours, so queueing behind another Judge is never gpu_busy.
                deadline = time.monotonic() + timeout
                while not entry._query_target_gpu_idle(judge):
                    if time.monotonic() >= deadline:
                        raise entry.ManagedJudgeLifecycleError("gpu_busy")
                    time.sleep(1.0)
                yield
            finally:
                flock(lock_file.fileno(), LOCK_UN)

    entry._gpu_uuids = role_gpu_uuids
    entry.dual_h100_create_container = separate_create_container
    entry.DockerBackend.create_container = separate_create_container
    entry._exclusive_managed_judge = separate_managed_judge
    entry._separate_gpu_installed = True
    return entry


def main() -> None:
    entry = install(load_entry())
    # The same hooks, in the same order, as the entry's own __main__ block.
    from sforge.harness.agent.claude_local import install_from_environment

    entry._install_managed_judge_wrapper()
    install_from_environment(entry._task_internet, sys.argv)
    entry.main()


if __name__ == "__main__":
    main()
