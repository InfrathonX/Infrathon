#!/usr/bin/env python3
"""Run the task's own SForge Docker entry, pinning Judge containers to a cpuset.

The task entry (scripts/sforge_h100_entry.py) only sets nano_cpus; the forced
runs additionally pin the Judge to the CPU slice of its GPU when
FORCED_JUDGE_CPUSET is set. Everything else (mounts, env, network none, GPU
DeviceRequest, labels, image-ID attestation) is the task entry's own code.
"""
from __future__ import annotations

import os
import runpy
import sys

import docker.models.containers as containers

_original_create = containers.ContainerCollection.create


def _create(self, image=None, command=None, **kwargs):
    cpuset = os.environ.get("FORCED_JUDGE_CPUSET", "").strip()
    labels = kwargs.get("labels") or {}
    if cpuset and labels.get("mlsysforge.gpu_role") == "judge":
        kwargs["cpuset_cpus"] = cpuset
    return _original_create(self, image, command, **kwargs)


containers.ContainerCollection.create = _create

if __name__ == "__main__":
    entry = sys.argv[1]
    sys.argv = [entry, *sys.argv[2:]]
    runpy.run_path(entry, run_name="__main__")
