#!/usr/bin/env python3
"""Root-owned multi-GPU sharding for the protected QVG controller.

Adapted from the kvpress v2 Judge (images/judge/judge_shards.py).  Cases are
independent, so the controller splits the suite across the GPUs the Judge
container sees.  Every slot runs the unchanged single-GPU protocol on one
device -- the isolated candidate stage, then the trusted metrics -- bound to
that device by index (``QVG_DEVICE_INDEX``, PCI bus order; the trusted
runners make it the current CUDA device and the controller rejects a profile
that left memory on any other GPU) and pinned to CPUs local to it.  The
computation for a case does not depend on which slot runs it or on how many
slots there are, so the score is independent of the GPU count; only the wall
time changes.

This module never imports torch and never touches CUDA itself: the device
inventory is measured by the controller's probe subprocess and CPU locality
comes from sysfs.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

MAX_SHARDS = 8
# A device whose allocator already holds more than this at the start of the
# run is busy (another process, a leaked context); the probe's own CUDA
# context is a few hundred MiB.
IDLE_SLACK_BYTES = 2 * 1024**3


@dataclass(frozen=True)
class Gpu:
    index: int
    uuid: str
    name: str
    pci_bus_id: str
    total_memory_bytes: int
    free_memory_bytes: int
    compute_capability: tuple[int, int]

    def record(self) -> dict:
        value = asdict(self)
        value["compute_capability"] = list(self.compute_capability)
        return value


def gpu_from_probe(row: Mapping) -> Gpu:
    """Parse one row of the controller's device probe (torch, PCI bus order)."""
    try:
        index = row["index"]
        capability = tuple(row["capability"])
        total = row["total_memory_bytes"]
        free = row["free_memory_bytes"]
        values = (index, total, free, *capability)
        if len(capability) != 2 or any(isinstance(x, bool) or not isinstance(x, int) for x in values):
            raise TypeError("non-integer device field")
        uuid = str(row["uuid"])
        if not uuid.startswith("GPU-"):
            # torch reports the UUID without nvidia-smi's "GPU-" prefix.
            uuid = "GPU-" + uuid
        if len(uuid) <= len("GPU-") or uuid.startswith(("GPU-MIG", "MIG-")):
            raise ValueError("no whole-device UUID (MIG is not supported)")
        return Gpu(int(index), uuid, str(row["name"]), str(row.get("pci_bus_id") or ""),
                   int(total), int(free), (int(capability[0]), int(capability[1])))
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"unexpected device probe row: {row!r}") from exc


def _visible_filter(gpus: Sequence[Gpu], environ: Mapping[str, str]) -> list[Gpu]:
    """Honour CUDA_VISIBLE_DEVICES (indices or UUIDs) when the runtime set it.

    The probe and every child run without it (identity mapping); a value set
    on the controller only narrows which of those devices become slots.
    """
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
    return sorted(chosen, key=lambda g: g.index)


def validate_fleet(gpus: Sequence[Gpu], *, min_count: int, max_count: int,
                   min_capability: tuple[int, int], min_memory_bytes: int,
                   idle_slack_bytes: int = IDLE_SLACK_BYTES) -> dict:
    """All slots must run on interchangeable, idle devices that meet the envelope."""
    if not 1 <= min_count <= max_count <= MAX_SHARDS:
        raise RuntimeError(f"invalid Judge GPU range {min_count}..{max_count}")
    if not min_count <= len(gpus) <= max_count:
        raise RuntimeError(f"Judge needs {min_count}..{max_count} GPUs, sees {len(gpus)}")
    if len({g.uuid for g in gpus}) != len(gpus):
        raise RuntimeError("duplicate GPU UUID")
    if len({g.index for g in gpus}) != len(gpus):
        raise RuntimeError("duplicate GPU index")
    models = {(g.name, g.total_memory_bytes, g.compute_capability) for g in gpus}
    if len(models) != 1:
        raise RuntimeError(f"Judge GPUs are not identical: {sorted(models)}")
    first = gpus[0]
    if first.compute_capability < tuple(min_capability):
        raise RuntimeError(f"GPU compute capability {first.compute_capability} is below {tuple(min_capability)}")
    if first.total_memory_bytes < int(min_memory_bytes):
        raise RuntimeError(f"GPU memory {first.total_memory_bytes} bytes is below {min_memory_bytes} bytes")
    busy = [g.index for g in gpus if g.total_memory_bytes - g.free_memory_bytes > idle_slack_bytes]
    if busy:
        raise RuntimeError(f"Judge GPUs {busy} are not idle at the start of the run")
    return {"count": len(gpus), "name": first.name, "total_memory_bytes": first.total_memory_bytes,
            "compute_capability": list(first.compute_capability), "devices": [g.record() for g in gpus]}


def _parse_cpulist(text: str) -> set[int]:
    cpus: set[int] = set()
    for part in (x.strip() for x in text.strip().split(",") if x.strip()):
        lo, _, hi = part.partition("-")
        cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


def _local_cpus(gpu: Gpu, sysfs: Path) -> set[int]:
    if not gpu.pci_bus_id:
        return set()
    domain, _, rest = gpu.pci_bus_id.lower().partition(":")
    # nvidia-smi and the probe print an 8-digit domain (00000000:46:00.0); sysfs uses 4.
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
    a shard's threads never outnumber its share of CPU time.
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


def slot_cpus(slices: Sequence[Sequence[int]], threads_per_slot: int) -> list[list[int]]:
    """Give every slot exactly ``threads_per_slot`` CPUs, whatever the GPU count.

    A case sees the same CPU budget on one GPU as on eight, so its wall time
    -- and the per-case wall gate -- does not depend on the fleet size.
    """
    if threads_per_slot < 1:
        raise RuntimeError("threads_per_slot must be positive")
    short = [i for i, cpus in enumerate(slices) if len(cpus) < threads_per_slot]
    if short:
        raise RuntimeError(f"slots {short} get fewer than {threads_per_slot} CPUs from this Judge allocation")
    return [sorted(cpus)[:threads_per_slot] for cpus in slices]


def plan_shards(item_ids: Sequence[str], costs: Mapping[str, float], n_shards: int,
                initial: Sequence[float] | None = None) -> list[list[str]]:
    """Deterministic longest-processing-time partition.

    Items are placed heaviest first on the shard that becomes free earliest
    (``initial`` start offset plus assigned cost; lowest index on ties); each
    shard then keeps the suite's original order.
    """
    if n_shards < 1:
        raise ValueError("n_shards must be positive")
    order = {item_id: i for i, item_id in enumerate(item_ids)}
    if len(order) != len(item_ids):
        raise ValueError("duplicate item id")
    loads = [0.0] * n_shards if initial is None else [float(x) for x in initial]
    if len(loads) != n_shards:
        raise ValueError("initial offsets do not match the shard count")
    members: list[list[str]] = [[] for _ in range(n_shards)]
    for item_id in sorted(item_ids, key=lambda x: (-float(costs[x]), order[x])):
        target = min(range(n_shards), key=lambda i: (loads[i], i))
        loads[target] += float(costs[item_id])
        members[target].append(item_id)
    return [sorted(shard, key=order.__getitem__) for shard in members]


def plan_makespan(plan: Sequence[Sequence[str]], costs: Mapping[str, float],
                  initial: Sequence[float] | None = None) -> float:
    """Finish time of the slowest shard when every item takes its full cost."""
    offsets = [0.0] * len(plan) if initial is None else [float(x) for x in initial]
    return max(offset + sum(float(costs[x]) for x in shard) for offset, shard in zip(offsets, plan))


def shard_environment(base: Mapping[str, str], gpu: Gpu, cpus: Sequence[int]) -> dict[str, str]:
    env = dict(base)
    # Every allocated GPU stays visible and the slot selects its own with
    # set_device: PyroMind's HAMi memory accounting is correct only under the
    # identity device mapping.  With CUDA_VISIBLE_DEVICES=<UUID> it charged all
    # shards to device 0 (atoi); with a single index it mis-detects the process
    # (HAMi-core #225, #319).  With PCI_BUS_ID order the CUDA index equals the
    # probe's index; the slot preflight checks the UUID each slot binds.
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["QVG_DEVICE_INDEX"] = str(gpu.index)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # PyroMind sets CUDA_DEVICE_SM_LIMIT=100 (the whole GPU), which starts
    # HAMi's per-process utilization watcher; with eight busy slots every
    # cudaMalloc then waits ~1 s on HAMi's shared lock and a 40 GB model load
    # takes ~21 min instead of ~80 s (measured 2026-10-06).  0 means "no SM
    # limit" -- no more than 100 grants -- and the memory limits are untouched.
    # Hosts without HAMi ignore it.
    env["CUDA_DEVICE_SM_LIMIT"] = "0"
    threads = str(max(1, len(cpus)))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = threads
    return env


def taskset_command(taskset: str, cpus: Sequence[int], command: Sequence[str]) -> list[str]:
    # Slots start children from several threads at once, so no preexec_fn:
    # the CPU slice is applied by taskset before the interpreter starts any
    # thread, and the privilege drop happens in setpriv after it.
    return [taskset, "--cpu-list", ",".join(str(c) for c in cpus), *command]
