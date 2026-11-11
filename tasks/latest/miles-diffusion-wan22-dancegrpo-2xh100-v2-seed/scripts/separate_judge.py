#!/usr/bin/env python3
"""The Judge server of a miles_diffusion Wan2.2 DanceGRPO (v2) run whose Judge has its own GPU pair (host only).

scripts/run_task.py starts the Judge server (`serve`) through this file instead of runtime/sforge_entry.py when the run
uses a separate Judge pair (JUDGE_GPU_INDICES=<c>,<d> or --judge-gpus, the default setup); with --shared-gpus
(JUDGE_GPU_INDICES=shared) it runs runtime/sforge_entry.py directly, as before.

runtime/sforge_entry.py serializes every manual Judge with exclusive_judge(), which pauses Work for the whole Judge
because both roles share one GPU pair. With a separate pair Work keeps running: this wrapper loads the entry unchanged
(it is a source-digest input of the task images, images/release.json, so it stays byte-identical) and replaces only
exclusive_judge with judge_pair_only(). That keeps the entry's Judge serialization (one Judge at a time per process,
the per-run flock /tmp/mdwan-judge-<run id>.lock) and its wait for the Judge's GPUs to be idle, but never looks up, pauses or resumes Work.
The serve process's MDWAN_GPU_UUIDS names the Judge pair, so both the entry's gpu_idle() and the Judge containers
it creates see only that pair.
Everything else (container creation, the integrity review, the Judge scratch reclaim) is the entry's own.

  separate_judge.py serve --host 0.0.0.0 --port PORT      # the arguments of runtime/sforge_entry.py
"""
from __future__ import annotations

import fcntl
import importlib.util
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ENTRY = Path(__file__).resolve().parents[1] / "runtime/sforge_entry.py"
JUDGE_LOCK_PREFIX = "mdwan"  # runtime/sforge_entry.py exclusive_judge: /tmp/<prefix>-judge-<run id>.lock
IDLE_TIMEOUT_S = 900  # the entry's own wait for the GPUs to go idle


def load_entry():
    """runtime/sforge_entry.py as a module (its main() is not run on import)."""
    spec = importlib.util.spec_from_file_location("mdwan_sforge_entry", ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def judge_pair_only(entry):
    """The entry's exclusive_judge without the pause: Work runs on another GPU pair."""

    @contextmanager
    def exclusive_judge(backend):
        with entry.THREAD_LOCK, open(f"/tmp/{JUDGE_LOCK_PREFIX}-judge-{entry.run_id()}.lock", "a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                deadline = time.monotonic() + IDLE_TIMEOUT_S
                while not entry.gpu_idle():
                    entry.require(time.monotonic() < deadline,
                                  "the Judge GPU pair stayed busy: another process is using the Judge's GPUs")
                    time.sleep(2)
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    return exclusive_judge


def install(entry):
    """Swap the pause-free serialization in; install() inside entry.main() resolves exclusive_judge at call time."""
    entry.exclusive_judge = judge_pair_only(entry)
    return entry


def main():
    entry = install(load_entry())
    # Run the entry as if started directly: its directory first on sys.path, its path as argv[0].
    sys.path[0] = str(ENTRY.parent)
    sys.argv[0] = str(ENTRY)
    entry.main()


if __name__ == "__main__":
    main()
