#!/usr/bin/env python3
"""Host-only SForge entry for a separate Judge GPU pair (StreamDiffusionV2 v2).

The task launcher runs this file instead of sforge_dual_h100_entry.py for
`serve` and `run` in separate mode only (SDV2_JUDGE_GPU_UUIDS or
SDV2_JUDGE_GPU_INDICES set). It loads that adapter by path and leaves it
byte-identical, so shared-mode runs, the final `eval` (eval_archive.sh) and
candidate replay (sforge_replay_entry.py) keep their loaded-evaluator identity.

On top of the adapter it changes two things:
- Judge containers get the Judge pair; Work keeps SDV2_GPU_UUIDS/INDICES.
- A managed Judge round keeps the adapter's thread lock, a flock keyed by the
  Judge pair and the same idle gate (now on the Judge pair, same timeout and
  gpu_busy error), but never looks up, pauses or unpauses Work. Judges drained
  after the deadline freeze therefore run without a live Work container.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from fcntl import LOCK_EX, LOCK_UN, flock
from pathlib import Path
from typing import Any, Iterator

from docker.types import DeviceRequest
from sforge.harness.backend.docker_backend import DockerBackend, DockerContainerHandle

BASE_PATH = Path(__file__).resolve().with_name("sforge_dual_h100_entry.py")
BASE_NAME = "sforge_dual_h100_entry"
JUDGE_GPU_UUID_ENV = "SDV2_JUDGE_GPU_UUIDS"
JUDGE_GPU_INDEX_ENV = "SDV2_JUDGE_GPU_INDICES"


def _load_base():
    loaded = sys.modules.get(BASE_NAME)
    if loaded is not None:
        if Path(getattr(loaded, "__file__", "") or "").resolve() != BASE_PATH:
            raise RuntimeError(f"{BASE_NAME} is already loaded from another path")
        return loaded
    spec = importlib.util.spec_from_file_location(BASE_NAME, BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[BASE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(BASE_NAME, None)
        raise
    return module


base = _load_base()


def _resolve_judge_uuids_from_indices(raw: str) -> str:
    values = [x.strip() for x in raw.split(",") if x.strip()]
    if len(values) != 2 or len(set(values)) != 2:
        raise RuntimeError(f"{JUDGE_GPU_INDEX_ENV} must name exactly two distinct physical GPU indices")
    try:
        indices = [int(x) for x in values]
    except ValueError as exc:
        raise RuntimeError(f"{JUDGE_GPU_INDEX_ENV} accepts only integer indices") from exc
    listing = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout
    by_index = {
        int(line.partition(",")[0].strip()): line.partition(",")[2].strip()
        for line in listing.splitlines() if line.strip()
    }
    if any(index not in by_index for index in indices):
        raise RuntimeError(f"{JUDGE_GPU_INDEX_ENV} names a missing GPU")
    return ",".join(by_index[index] for index in indices)


def separate_judge_pair() -> bool:
    return bool(
        os.environ.get(JUDGE_GPU_UUID_ENV, "").strip()
        or os.environ.get(JUDGE_GPU_INDEX_ENV, "").strip()
    )


def judge_gpu_uuids() -> list[str]:
    """The Judge pair: two valid UUIDs that share no GPU with the Work pair."""

    raw = os.environ.get(JUDGE_GPU_UUID_ENV, "").strip()
    if not raw:
        indices = os.environ.get(JUDGE_GPU_INDEX_ENV, "").strip()
        if not indices:
            raise RuntimeError(
                f"the separate-Judge entry needs {JUDGE_GPU_UUID_ENV} or {JUDGE_GPU_INDEX_ENV}"
            )
        raw = _resolve_judge_uuids_from_indices(indices)
    values = [x.strip() for x in raw.split(",") if x.strip()]
    if (
        len(values) != 2
        or any(base._GPU_UUID_RE.fullmatch(x) is None for x in values)
        or len({x.casefold() for x in values}) != 2
    ):
        raise RuntimeError(f"{JUDGE_GPU_UUID_ENV} must contain two distinct full GPU UUIDs")
    if {x.casefold() for x in values} & {x.casefold() for x in base._gpu_uuids()}:
        raise RuntimeError(
            f"{JUDGE_GPU_UUID_ENV} overlaps {base.GPU_UUID_ENV}; a separate Judge needs two other GPUs"
        )
    return values


@contextmanager
def separate_managed_judge(*, backend: Any, task_id: str, run_id: str) -> Iterator[None]:
    """Serialize and idle-gate a Judge round on the Judge pair; Work is untouched."""

    if getattr(backend, "backend_name", None) != "docker" or getattr(backend, "client", None) is None:
        raise base.ManagedJudgeLifecycleError("unsupported_backend")
    if task_id != base.TARGET_TASK_ID or base._RUN_ID_RE.fullmatch(run_id) is None:
        raise base.ManagedJudgeLifecycleError("invalid_run_identity")
    timeout = base._managed_idle_timeout()
    gpu_uuids = judge_gpu_uuids()
    with base._managed_judge_thread_lock, base._managed_judge_lock_path(gpu_uuids).open("a+") as lock:
        flock(lock.fileno(), LOCK_EX)
        # As in shared mode, the idle wait starts once the pair is ours.
        deadline = time.monotonic() + timeout
        try:
            while not base._query_target_gpu_idle(gpu_uuids):
                if time.monotonic() >= deadline:
                    raise base.ManagedJudgeLifecycleError("gpu_busy")
                time.sleep(1)
            yield
        finally:
            flock(lock.fileno(), LOCK_UN)


def separate_create_container(
    self: DockerBackend, image: str, name: str, *, command: str = "tail -f /dev/null",
    environment=None, extra_hosts=None, cap_drop=None, cpu_limit=None, mem_limit=None,
    user=None, annotations=None, mounts=None,
):
    """Work: the adapter's container on the Work pair. Judge: the same, on the Judge pair."""

    if not base._is_judge(image, name):
        return base.dual_h100_create_container(
            self, image, name, command=command, environment=environment,
            extra_hosts=extra_hosts, cap_drop=cap_drop, cpu_limit=cpu_limit,
            mem_limit=mem_limit, user=user, annotations=annotations, mounts=mounts,
        )
    del annotations
    if mounts:
        raise RuntimeError("Judge rejects external mounts")
    gpu_uuids = judge_gpu_uuids()
    container_environment = dict(environment or {})
    container_environment.update(CUDA_VISIBLE_DEVICES="0,1", NVIDIA_VISIBLE_DEVICES=",".join(gpu_uuids))
    kwargs = {
        "volumes": {},
        "device_requests": [DeviceRequest(driver="nvidia", device_ids=gpu_uuids, capabilities=[["gpu"]])],
        "security_opt": ["no-new-privileges:true"],
        "network_mode": "none",
    }
    if cpu_limit is not None:
        kwargs["nano_cpus"] = int(cpu_limit * 1e9)
    if mem_limit is not None:
        kwargs["mem_limit"] = mem_limit
    if cap_drop:
        kwargs["cap_drop"] = cap_drop
    if extra_hosts:
        kwargs["extra_hosts"] = extra_hosts
    if user:
        kwargs["user"] = user
    return DockerContainerHandle(
        self.client.containers.create(
            image, name=name, detach=True, command=command, environment=container_environment, **kwargs
        )
    )


def install() -> None:
    """Swap in the separate-pair container mapping and managed Judge round."""

    if not separate_judge_pair():
        raise RuntimeError(
            f"separate_judge_entry.py is for a separate Judge pair; set {JUDGE_GPU_INDEX_ENV}"
            " or run sforge_dual_h100_entry.py for a shared pair"
        )
    judge_gpu_uuids()  # fail before serving on a bad or overlapping pair
    DockerBackend.create_container = separate_create_container
    # The adapter's judge_submission wrapper looks this name up at call time.
    base._exclusive_managed_judge = separate_managed_judge


def main() -> None:
    from sforge.harness.agent.claude_local import install_from_environment

    install()
    # The adapter's own __main__ sequence, unchanged.
    base._install_managed_judge_wrapper()
    base._install_candidate_eval_round_wrapper()
    install_from_environment(base._task_internet, sys.argv)
    base._sforge_cli.main()


if __name__ == "__main__":
    main()
