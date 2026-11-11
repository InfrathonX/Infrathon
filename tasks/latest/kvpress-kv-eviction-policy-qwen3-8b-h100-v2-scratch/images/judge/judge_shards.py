#!/usr/bin/env python3
"""Root-owned multi-GPU sharding for the two-phase Judge.

Items are independent, so the verifier splits the suite across the GPUs the
Judge container sees.  Every shard runs the unchanged single-GPU protocol on
one device -- the unprivileged Phase-A worker, then the trusted Phase-B replay
-- bound to that device by index (``SFORGE_JUDGE_DEVICE``, PCI bus order;
the runtime makes it the current CUDA device and rejects candidate code that
touches any other GPU) and pinned to CPUs local to it.  The computation for an item does not depend on
which shard runs it or on how many shards there are, so the score is
independent of the GPU count; only the wall time changes.

This module never imports torch and never touches CUDA itself: device
inventory comes from ``nvidia-smi`` and CPU locality from sysfs.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

MAX_SHARDS = 8
NVIDIA_SMI_FIELDS = "index,uuid,name,pci.bus_id,memory.total,compute_cap"


@dataclass(frozen=True)
class Gpu:
    index: int
    uuid: str
    name: str
    pci_bus_id: str
    memory_total_mib: int
    compute_capability: tuple[int, int]

    def record(self) -> dict:
        value = asdict(self)
        value["compute_capability"] = list(self.compute_capability)
        return value


def _parse_gpu(line: str) -> Gpu:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 6:
        raise RuntimeError(f"unexpected nvidia-smi row: {line!r}")
    index, uuid, name, bus_id, memory, capability = parts
    major, _, minor = capability.partition(".")
    if not uuid.startswith("GPU-"):
        raise RuntimeError(f"GPU {index} has no whole-device UUID (MIG is not supported): {uuid!r}")
    return Gpu(int(index), uuid, name, bus_id, int(float(memory)), (int(major), int(minor or 0)))


def _visible_filter(gpus: Sequence[Gpu], environ: Mapping[str, str]) -> list[Gpu]:
    """Honour CUDA_VISIBLE_DEVICES (indices or UUIDs) when the runtime set it."""
    raw = environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None or raw.strip() in ("", "all"):
        return list(gpus)
    chosen = []
    for token in (x.strip() for x in raw.split(",") if x.strip()):
        match = [g for g in gpus if token == g.uuid or (token.isdigit() and int(token) == g.index)]
        if len(match) != 1:
            raise RuntimeError(f"CUDA_VISIBLE_DEVICES entry {token!r} does not name exactly one visible GPU")
        if match[0] not in chosen:
            chosen.append(match[0])
    return chosen


def visible_gpus(environ: Mapping[str, str] = os.environ) -> list[Gpu]:
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={NVIDIA_SMI_FIELDS}", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=60,
    ).stdout
    gpus = [_parse_gpu(line) for line in out.splitlines() if line.strip()]
    return _visible_filter(gpus, environ)


def validate_fleet(gpus: Sequence[Gpu], *, min_capability: tuple[int, int], min_memory_bytes: int) -> dict:
    """All shards must run on interchangeable devices that meet the envelope."""
    if not 1 <= len(gpus) <= MAX_SHARDS:
        raise RuntimeError(f"Judge needs 1..{MAX_SHARDS} GPUs, sees {len(gpus)}")
    if len({g.uuid for g in gpus}) != len(gpus):
        raise RuntimeError("duplicate GPU UUID")
    models = {(g.name, g.memory_total_mib, g.compute_capability) for g in gpus}
    if len(models) != 1:
        raise RuntimeError(f"Judge GPUs are not identical: {sorted(models)}")
    first = gpus[0]
    if first.compute_capability < tuple(min_capability):
        raise RuntimeError(f"GPU compute capability {first.compute_capability} is below {tuple(min_capability)}")
    if first.memory_total_mib * (1 << 20) < int(min_memory_bytes):
        raise RuntimeError(f"GPU memory {first.memory_total_mib} MiB is below {min_memory_bytes} bytes")
    return {"count": len(gpus), "name": first.name, "memory_total_mib": first.memory_total_mib,
            "compute_capability": list(first.compute_capability), "devices": [g.record() for g in gpus]}


def _parse_cpulist(text: str) -> set[int]:
    cpus: set[int] = set()
    for part in (x.strip() for x in text.strip().split(",") if x.strip()):
        lo, _, hi = part.partition("-")
        cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


def _local_cpus(gpu: Gpu, sysfs: Path) -> set[int]:
    domain, _, rest = gpu.pci_bus_id.lower().partition(":")
    # nvidia-smi prints an 8-digit domain (00000000:46:00.0); sysfs uses 4.
    path = sysfs / f"{domain[-4:]}:{rest}" / "local_cpulist"
    try:
        return _parse_cpulist(path.read_text())
    except (OSError, ValueError):
        return set()


def _core_groups(cpus: Sequence[int], cpu_sysfs: Path) -> list[list[int]]:
    """Group CPUs by physical core (SMT siblings together), ordered by lowest id."""
    wanted = set(cpus); groups: dict[tuple[int, ...], list[int]] = {}
    for cpu in sorted(wanted):
        try:
            siblings = _parse_cpulist((cpu_sysfs / f"cpu{cpu}/topology/thread_siblings_list").read_text())
        except (OSError, ValueError):
            siblings = {cpu}
        groups.setdefault(tuple(sorted(siblings & wanted)), []).append(cpu)
    return sorted((sorted(v) for v in groups.values()), key=lambda g: g[0])


def _split_even(cpus: Sequence[int], parts: int, cpu_sysfs: Path = Path("/sys/devices/system/cpu")) -> list[list[int]]:
    """Split into ``parts`` sets of whole physical cores (SMT siblings never straddle shards)."""
    cores = _core_groups(cpus, cpu_sysfs)
    if len(cores) < parts:
        cores = [[cpu] for cpu in sorted(cpus)]
    size, extra = divmod(len(cores), parts)
    out, start = [], 0
    for i in range(parts):
        end = start + size + (1 if i < extra else 0)
        out.append(sorted(cpu for core in cores[start:end] for cpu in core))
        start = end
    return out


def cpu_quota(cgroup: Path = Path("/sys/fs/cgroup")) -> float | None:
    """CPUs granted by the container's CFS quota (cgroup v2 ``cpu.max``, else v1), or None when unlimited.

    Docker ``--cpus`` and Kubernetes CPU limits are quotas: every host CPU
    stays visible to ``sched_getaffinity`` while only this much CPU time is
    granted, so the shard slices must not exceed it.
    """
    try:
        fields = (cgroup / "cpu.max").read_text().split()
        if fields[0] == "max": return None
        return int(fields[0]) / int(fields[1])
    except (OSError, ValueError, IndexError):
        pass
    try:
        quota = int((cgroup / "cpu" / "cpu.cfs_quota_us").read_text()); period = int((cgroup / "cpu" / "cpu.cfs_period_us").read_text())
        return quota / period if quota > 0 and period > 0 else None
    except (OSError, ValueError):
        return None


def cpu_slices(gpus: Sequence[Gpu], *, allowed: Sequence[int] | None = None,
               sysfs: Path = Path("/sys/bus/pci/devices"), quota: float | None | str = "auto") -> tuple[list[list[int]], str]:
    """Disjoint, equally sized CPU sets per GPU, NUMA-local when sysfs allows.

    GPUs that share a NUMA node share one ``local_cpulist``; that set is split
    evenly between them.  When locality is unavailable or leaves a shard
    without CPUs, the allowed set is split evenly instead.  Each slice is then
    capped at the CFS quota divided by the shard count (at least one CPU), so
    a shard's threads (``OMP_NUM_THREADS`` = slice size) never outnumber its
    share of CPU time.
    """
    granted = cpu_quota() if quota == "auto" else quota
    budget = None if granted is None else max(1, int(granted // len(gpus)))
    slices, mode = _locality_slices(gpus, allowed=allowed, sysfs=sysfs)
    if budget is not None and budget < min(len(x) for x in slices):
        return [x[:budget] for x in slices], f"{mode}+quota-{budget}"
    return slices, mode


def _locality_slices(gpus: Sequence[Gpu], *, allowed: Sequence[int] | None, sysfs: Path) -> tuple[list[list[int]], str]:
    allowed_set = sorted(set(os.sched_getaffinity(0) if allowed is None else allowed))
    if len(allowed_set) < len(gpus):
        raise RuntimeError(f"{len(gpus)} shards need at least {len(gpus)} CPUs, have {len(allowed_set)}")
    groups: dict[tuple[int, ...], list[int]] = {}
    for position, gpu in enumerate(gpus):
        local = tuple(sorted(_local_cpus(gpu, sysfs) & set(allowed_set)))
        groups.setdefault(local, []).append(position)
    slices: list[list[int]] = [[] for _ in gpus]
    local_ok = all(key for key in groups)
    if local_ok:
        keys = list(groups)
        if any(set(a) & set(b) for i, a in enumerate(keys) for b in keys[i + 1:]):
            local_ok = False
    if local_ok:
        for key, members in groups.items():
            for position, part in zip(members, _split_even(list(key), len(members))):
                slices[position] = part
        local_ok = all(slices)
    if local_ok:
        width = min(len(s) for s in slices)
        return [s[:width] for s in slices], "numa-local"
    return _split_even(allowed_set, len(gpus)), "even-split"


def plan_shards(item_ids: Sequence[str], costs: Mapping[str, float], n_shards: int) -> list[list[str]]:
    """Deterministic longest-processing-time partition.

    Items are placed heaviest first on the least-loaded shard (lowest index on
    ties); each shard then keeps the suite's original order.
    """
    if n_shards < 1:
        raise ValueError("n_shards must be positive")
    order = {item_id: i for i, item_id in enumerate(item_ids)}
    if len(order) != len(item_ids):
        raise ValueError("duplicate item id")
    loads = [0.0] * n_shards
    members: list[list[str]] = [[] for _ in range(n_shards)]
    for item_id in sorted(item_ids, key=lambda x: (-float(costs[x]), x)):
        target = min(range(n_shards), key=lambda i: (loads[i], i))
        loads[target] += float(costs[item_id])
        members[target].append(item_id)
    return [sorted(shard, key=order.__getitem__) for shard in members]


def shard_environment(base: Mapping[str, str], gpu: Gpu, cpus: Sequence[int]) -> dict[str, str]:
    env = dict(base)
    # Every allocated GPU stays visible and the shard selects its own with
    # set_device: PyroMind's HAMi memory accounting is correct only under the
    # identity device mapping.  With CUDA_VISIBLE_DEVICES=<UUID> it charged all
    # shards to device 0 (atoi); with a single index it mis-detects the process
    # (HAMi-core #225, #319).  With PCI_BUS_ID order the CUDA index equals the
    # nvidia-smi index; the verifier checks the UUID each shard ran on.
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["SFORGE_JUDGE_DEVICE"] = str(gpu.index)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    threads = str(max(1, len(cpus)))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[key] = threads
    return env


def pin_current_process(cpus: Sequence[int]) -> None:
    os.sched_setaffinity(0, set(cpus))
