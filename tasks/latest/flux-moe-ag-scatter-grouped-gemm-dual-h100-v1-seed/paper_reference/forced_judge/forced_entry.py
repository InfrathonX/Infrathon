#!/usr/bin/env python3
"""Run the task's own dual-H100 SForge entry, pinning the Judge to a cpuset.

scripts/sforge_dual_h100_entry.py only sets nano_cpus; when FORCED_JUDGE_CPUSET
is set the Judge container is additionally pinned to the CPU slices of its two
GPUs. Mounts, env, network none, the two-GPU DeviceRequest and the command are
the task entry's own.
"""
from __future__ import annotations

import os
import runpy
import sys

import docker.models.containers as containers

_original_create = containers.ContainerCollection.create


def _create(self, image=None, command=None, **kwargs):
    cpuset = os.environ.get("FORCED_JUDGE_CPUSET", "").strip()
    name = str(kwargs.get("name") or "")
    if cpuset and (".judge." in str(image) or ".judge." in name):
        kwargs["cpuset_cpus"] = cpuset
    return _original_create(self, image, command, **kwargs)


containers.ContainerCollection.create = _create

if __name__ == "__main__":
    entry = sys.argv[1]
    sys.argv = [entry, *sys.argv[2:]]
    runpy.run_path(entry, run_name="__main__")
