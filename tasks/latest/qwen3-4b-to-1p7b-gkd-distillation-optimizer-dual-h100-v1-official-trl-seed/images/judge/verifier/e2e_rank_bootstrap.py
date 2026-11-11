#!/usr/bin/env python3
"""Load the root-owned runtime package without adding the lab to sys.path."""

from __future__ import annotations

import importlib.util
import os
import runpy
import stat
import sys
from pathlib import Path


EXPECTED_RUNTIME_FILES = {
    "__init__.py",
    "_source_policy_impl.py",
    "api.py",
    "distributed_probe.py",
    "hf_reference.py",
    "loader.py",
    "source_policy.py",
    "train_step.py",
}


def _root_owned_read_only(path: Path, *, directory: bool) -> None:
    info = path.lstat()
    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    if stat.S_ISLNK(info.st_mode) or not expected_type(info.st_mode):
        raise RuntimeError("trusted runtime layout is invalid")
    if info.st_uid != 0 or info.st_mode & 0o022:
        raise RuntimeError("trusted runtime ownership or mode is invalid")


def _load_runtime(runtime_root: Path) -> None:
    runtime_root = runtime_root.absolute()
    _root_owned_read_only(runtime_root.parent, directory=True)
    _root_owned_read_only(runtime_root, directory=True)
    entries = {entry.name for entry in runtime_root.iterdir() if entry.is_file()}
    if entries != EXPECTED_RUNTIME_FILES:
        raise RuntimeError("trusted runtime file set is invalid")
    for name in EXPECTED_RUNTIME_FILES:
        _root_owned_read_only(runtime_root / name, directory=False)

    init_path = runtime_root / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "runtime",
        init_path,
        submodule_search_locations=[str(runtime_root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("trusted runtime package cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules["runtime"] = module
    spec.loader.exec_module(module)


def main() -> int:
    arguments = sys.argv[1:]
    if len(arguments) < 4 or arguments[0] != "--runtime-root" or arguments[2] != "--":
        raise RuntimeError("invalid rank bootstrap request")
    runtime_root = Path(arguments[1])
    worker_arguments = arguments[3:]
    if not worker_arguments:
        raise RuntimeError("rank bootstrap received no worker arguments")
    _load_runtime(runtime_root)
    probe_path = runtime_root / "distributed_probe.py"
    sys.argv = [str(probe_path), *worker_arguments]
    runpy.run_module("runtime.distributed_probe", run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        # Do not echo paths or import details into candidate-visible stderr.
        os.write(2, b"rank bootstrap failed\n")
        raise SystemExit(1)
