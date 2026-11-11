#!/usr/bin/env python3
"""Forced-oracle candidate worker (LOCAL forced-oracle Judge image only).

Loads the Judge's own ``candidate_worker.py`` by path and runs its ``main``
unchanged, except that the trusted runtime's ``run_suite`` is called with
``stop_on_edge_failure=False``: after a failing edge case the scored grid is
still measured (with the identical protocol), so the private result carries
the scored-case numbers next to the failed edge gate.  The runtime still
returns ``status = "failed"`` with the edge gate as the failing gate, and it
still stops at once after a CUDA fault or a harness-integrity failure.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_WORKER = Path(__file__).resolve().parent / "candidate_worker.py"


def _load_worker():
    spec = importlib.util.spec_from_file_location("candidate_worker", _WORKER)
    if spec is None or spec.loader is None:
        raise RuntimeError("candidate_worker.py could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    worker = _load_worker()
    original_loader = worker._load_trusted_runtime

    def _load_trusted_runtime():
        runtime = original_loader()
        run_suite = runtime.run_suite

        def forced_run_suite(*args, **kwargs):
            kwargs["stop_on_edge_failure"] = False
            return run_suite(*args, **kwargs)

        runtime.run_suite = forced_run_suite
        return runtime

    worker._load_trusted_runtime = _load_trusted_runtime
    return int(worker.main())


if __name__ == "__main__":
    raise SystemExit(main())
