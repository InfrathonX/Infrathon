"""Host CPU/GPU topology for opt-in GPU-local CPU pinning.

Algorithm ``gpu-local-l3-v1`` gives every NVIDIA GPU of the host a static,
disjoint slice of the CPUs of its own NUMA node:

* the node's online CPUs (minus an optional reserve) are grouped into whole
  cores (SMT siblings stay together) and the cores into L3 groups, ranked by
  their lowest CPU (never by cache ``id``, which is not in CPU order);
* the node's GPUs, sorted by PCI address, share the node's L3 groups: when the
  group count divides among them each GPU owns whole groups (one CCD per GPU
  on an AMD host with as many CCDs as GPUs), otherwise each GPU gets an equal
  contiguous chunk of cores;
* every GPU of a node gets the same core count; the surplus is node spare.

Everything is derived from sysfs at run time; there is no host constant.  A
container with several GPUs gets the union of their slices, trimmed to its
CPU budget whole cores first.

Stdlib only, so launcher shells can use it with the system python::

    PYTHONPATH=<repo> python3 -m sforge.harness.host_topology table
    PYTHONPATH=<repo> python3 -m sforge.harness.host_topology \\
        slice --gpu "$GPU_UUID" --budget 8 --format docker-args

``table``, ``policy-id`` and ``check`` read sysfs/procfs only.  GPU indices
(and UUIDs that procfs cannot map) are resolved through NVML in a child
process and then ``nvidia-smi``, each under a timeout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ALGORITHM_ID = "gpu-local-l3-v1"
RESOLVE_TIMEOUT_SECONDS = 20.0
# How long a timed-out inventory child may take to die once killed.
INVENTORY_REAP_SECONDS = 5.0

_CPULIST_RE = re.compile(r"\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*", re.ASCII)
_GPU_UUID_RE = re.compile(
    r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", re.ASCII
)
_PCI_RE = re.compile(
    r"(?:([0-9A-Fa-f]{4,8}):)?([0-9A-Fa-f]{2}):([0-9A-Fa-f]{2})\.([0-7])",
    re.ASCII,
)
_INDEX_RE = re.compile(r"\d{1,4}", re.ASCII)
_NVIDIA_VENDOR = "0x10de"
# VGA and 3D controllers; NVSwitch and bridges (0x0680) are not GPUs.
_GPU_CLASSES = (0x0300, 0x0302)


class TopologyError(RuntimeError):
    """The host topology cannot support the requested CPU pinning."""


# ---------------------------------------------------------------------------
# cpulist syntax
# ---------------------------------------------------------------------------


def parse_cpulist(text: str) -> frozenset[int]:
    """Parse kernel cpulist syntax (``0-3,32-35``); blank means empty."""

    text = text.strip()
    if not text:
        return frozenset()
    if not _CPULIST_RE.fullmatch(text):
        raise ValueError(f"malformed CPU list {text!r}")
    out: set[int] = set()
    for part in text.split(","):
        low, _, high = part.partition("-")
        first, last = int(low), int(high or low)
        if last < first:
            raise ValueError(f"descending range {part!r} in CPU list {text!r}")
        out.update(range(first, last + 1))
    return frozenset(out)


def format_cpulist(cpus) -> str:
    """Format a CPU (or node) set as a canonical kernel cpulist."""

    ordered = sorted(set(cpus))
    parts: list[str] = []
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1] == ordered[j] + 1:
            j += 1
        parts.append(
            str(ordered[i]) if i == j else f"{ordered[i]}-{ordered[j]}"
        )
        i = j + 1
    return ",".join(parts)


def normalize_cpulist(text: str | None) -> str:
    """Canonical form of a cpulist string, for comparing Docker readbacks."""

    return format_cpulist(parse_cpulist(text or ""))


def normalize_pci(bus_id: str) -> str:
    """``00000000:4E:00.0`` -> ``0000:4e:00.0`` (sysfs spelling)."""

    match = _PCI_RE.fullmatch(bus_id.strip())
    if match is None:
        raise ValueError(f"malformed PCI bus ID {bus_id!r}")
    domain = int(match.group(1) or "0", 16)
    if domain > 0xFFFF:
        raise ValueError(f"PCI domain out of range in {bus_id!r}")
    return (
        f"{domain:04x}:{match.group(2)}:{match.group(3)}.{match.group(4)}"
    ).lower()


# ---------------------------------------------------------------------------
# sysfs model
# ---------------------------------------------------------------------------


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="ascii", errors="replace").strip()
    except OSError:
        return None


@dataclass(frozen=True)
class Gpu:
    pci: str                     # 0000:46:00.0 (lowercase, 4-hex domain)
    node: int | None             # numa_node; None when -1 or missing
    local: frozenset[int]        # local_cpulist ∩ online


@dataclass(frozen=True)
class Topology:
    online: frozenset[int]
    nodes: dict[int, frozenset[int]]         # CPU-bearing nodes only
    all_nodes: frozenset[int]                # every nodeN directory
    core_of: dict[int, frozenset[int]]       # cpu -> SMT siblings
    l3_of: dict[int, frozenset[int] | None]  # cpu -> L3 sharers
    gpus: tuple[Gpu, ...]                    # sorted by PCI address

    def node_of(self, cpu: int) -> int | None:
        for node, cpus in self.nodes.items():
            if cpu in cpus:
                return node
        return None


def read_topology(sysfs: Path = Path("/sys")) -> Topology:
    """Read CPUs, NUMA nodes, cores, L3 groups and NVIDIA GPUs from sysfs."""

    sysfs = Path(sysfs)
    cpu_root = sysfs / "devices/system/cpu"
    try:
        online = parse_cpulist(_read(cpu_root / "online") or "")
    except ValueError as exc:
        raise TopologyError(f"unreadable online CPU list: {exc}") from exc
    # Never hand out CPUs the root cgroup cannot schedule.
    effective = _read(sysfs / "fs/cgroup/cpuset.cpus.effective")
    if effective:
        try:
            online &= parse_cpulist(effective)
        except ValueError:
            pass
    if not online:
        raise TopologyError(f"no online CPUs under {cpu_root}")

    nodes: dict[int, frozenset[int]] = {}
    all_nodes: set[int] = set()
    node_root = sysfs / "devices/system/node"
    if node_root.is_dir():
        for directory in sorted(node_root.glob("node[0-9]*")):
            if not directory.name[4:].isdigit():
                continue
            node = int(directory.name[4:])
            all_nodes.add(node)
            try:
                cpus = parse_cpulist(_read(directory / "cpulist") or "")
            except ValueError as exc:
                raise TopologyError(
                    f"unreadable CPU list of NUMA node {node}: {exc}"
                ) from exc
            cpus &= online
            if cpus:
                nodes[node] = cpus

    core_of: dict[int, frozenset[int]] = {}
    l3_of: dict[int, frozenset[int] | None] = {}
    for cpu in sorted(online):
        base = cpu_root / f"cpu{cpu}"
        siblings = _read(base / "topology/thread_siblings_list")
        try:
            core = parse_cpulist(siblings) & online if siblings else frozenset()
        except ValueError:
            core = frozenset()
        core_of[cpu] = core or frozenset({cpu})
        shared = None
        for index in sorted((base / "cache").glob("index[0-9]*")):
            if (
                _read(index / "level") == "3"
                and _read(index / "type") != "Instruction"
            ):
                shared = _read(index / "shared_cpu_list")
                break
        try:
            group = parse_cpulist(shared) & online if shared else frozenset()
        except ValueError:
            group = frozenset()
        l3_of[cpu] = group or None

    gpus: list[Gpu] = []
    pci_root = sysfs / "bus/pci/devices"
    for directory in sorted(pci_root.glob("*")) if pci_root.is_dir() else []:
        if (_read(directory / "vendor") or "").lower() != _NVIDIA_VENDOR:
            continue
        try:
            device_class = int(_read(directory / "class") or "0", 16)
        except ValueError:
            continue
        if (device_class >> 8) not in _GPU_CLASSES:
            continue
        try:
            pci = normalize_pci(directory.name)
        except ValueError:
            continue
        try:
            node = int(_read(directory / "numa_node") or "-1")
        except ValueError:
            node = -1
        try:
            local = parse_cpulist(_read(directory / "local_cpulist") or "")
        except ValueError:
            local = frozenset()
        gpus.append(Gpu(pci, None if node < 0 else node, local & online))

    return Topology(
        online=online,
        nodes=nodes,
        all_nodes=frozenset(all_nodes),
        core_of=core_of,
        l3_of=l3_of,
        gpus=tuple(sorted(gpus, key=lambda gpu: gpu.pci)),
    )


# ---------------------------------------------------------------------------
# Partition gpu-local-l3-v1
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GpuSlice:
    pci: str
    node: int | None
    cpus: frozenset[int]
    cores: int
    l3_groups: tuple[str, ...]
    l3_confined: bool             # no other GPU's slice shares its L3 groups
    node_spare: frozenset[int]    # domain cores no GPU of the domain owns


def _gpu_domain(topo: Topology, gpu: Gpu) -> tuple[int | None, frozenset[int]]:
    if gpu.node is not None:
        if gpu.node not in topo.nodes:
            raise TopologyError(
                f"GPU {gpu.pci} is on NUMA node {gpu.node}, which has no "
                "online CPUs"
            )
        return gpu.node, topo.nodes[gpu.node]
    if len(topo.nodes) <= 1:
        node = next(iter(topo.nodes), None)
        return node, (topo.nodes[node] if node is not None else topo.online)
    raise TopologyError(
        f"GPU {gpu.pci} reports no NUMA node on a {len(topo.nodes)}-node "
        "host; give an explicit CPULIST@NODES policy instead of auto"
    )


def partition(
    topo: Topology,
    reserve: frozenset[int] = frozenset(),
) -> dict[str, GpuSlice]:
    """Return the static per-GPU slices of the host, keyed by PCI address."""

    by_domain: dict[tuple[int | None, frozenset[int]], list[Gpu]] = {}
    for gpu in topo.gpus:
        by_domain.setdefault(_gpu_domain(topo, gpu), []).append(gpu)

    table: dict[str, GpuSlice] = {}
    for (node, domain), domain_gpus in by_domain.items():
        available = domain - reserve
        cores = {
            topo.core_of[cpu]
            for cpu in available
            if topo.core_of[cpu] <= available
        }

        def l3_rank(core: frozenset[int]) -> int:
            group = topo.l3_of[min(core)]
            return min(group) if group else min(domain)

        ordered = sorted(cores, key=lambda core: (l3_rank(core), min(core)))
        count = len(domain_gpus)
        grouped: dict[int, list[frozenset[int]]] = {}
        for core in ordered:
            grouped.setdefault(l3_rank(core), []).append(core)
        groups = [grouped[rank] for rank in sorted(grouped)]
        if len(groups) >= count and len(groups) % count == 0:
            # Whole-L3 mode: GPU i owns groups [i*m, (i+1)*m) and takes the
            # smallest owner's core count, so every GPU gets the same size.
            per_gpu_groups = len(groups) // count
            owned = [
                [
                    core
                    for group in groups[
                        i * per_gpu_groups:(i + 1) * per_gpu_groups
                    ]
                    for core in group
                ]
                for i in range(count)
            ]
            per = min(len(cores_owned) for cores_owned in owned)
            chunks = [cores_owned[:per] for cores_owned in owned]
        else:
            per = len(ordered) // count
            chunks = [ordered[i * per:(i + 1) * per] for i in range(count)]
        if per == 0:
            raise TopologyError(
                f"NUMA node {node}: {len(ordered)} free cores for "
                f"{count} GPUs"
            )
        used = frozenset(cpu for chunk in chunks for core in chunk for cpu in core)
        spare = frozenset(cpu for core in ordered for cpu in core) - used
        slices = []
        for gpu, chunk in zip(
            sorted(domain_gpus, key=lambda item: item.pci), chunks
        ):
            cpus = frozenset(cpu for core in chunk for cpu in core)
            l3_groups = frozenset(
                topo.l3_of[min(core)]
                for core in chunk
                if topo.l3_of[min(core)]
            )
            slices.append((gpu, cpus, len(chunk), l3_groups))
        for gpu, cpus, core_count, l3_groups in slices:
            others = frozenset(
                cpu
                for other, other_cpus, _, _ in slices
                if other.pci != gpu.pci
                for cpu in other_cpus
            )
            table[gpu.pci] = GpuSlice(
                pci=gpu.pci,
                node=node,
                cpus=cpus,
                cores=core_count,
                l3_groups=tuple(
                    sorted(
                        (format_cpulist(group) for group in l3_groups),
                        key=lambda text: min(parse_cpulist(text)),
                    )
                ),
                l3_confined=bool(l3_groups)
                and not any(group & others for group in l3_groups),
                node_spare=spare,
            )
    return table


def fingerprint(table: dict[str, GpuSlice]) -> str:
    """Host-specific digest of the partition (for records, never identity)."""

    canonical = {
        pci: {"node": item.node, "cpus": format_cpulist(item.cpus)}
        for pci, item in sorted(table.items())
    }
    encoded = json.dumps(canonical, sort_keys=True)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def slice_for(
    table: dict[str, GpuSlice],
    pcis,
) -> tuple[frozenset[int], frozenset[int]]:
    """Union of the static slices of ``pcis`` and the union of their nodes."""

    cpus: frozenset[int] = frozenset()
    mems: frozenset[int] = frozenset()
    for pci in pcis:
        if pci not in table:
            raise TopologyError(f"GPU {pci} is not an NVIDIA GPU in sysfs")
        cpus |= table[pci].cpus
        if table[pci].node is not None:
            mems |= {table[pci].node}
    return cpus, mems


def _ordered_cores(topo: Topology, cpus: frozenset[int]) -> list[frozenset[int]]:
    cores = {topo.core_of.get(cpu, frozenset({cpu})) & cpus for cpu in cpus}
    return sorted(
        cores,
        key=lambda core: (min(topo.l3_of.get(min(core)) or core), min(core)),
    )


def trim(topo: Topology, cpus: frozenset[int], budget: int) -> frozenset[int]:
    """Largest subset of ``cpus`` with at most ``budget`` CPUs.

    Whole cores are taken first in L3 order; an odd budget adds one thread of
    the next unused core.
    """

    if budget >= len(cpus):
        return cpus
    cores = _ordered_cores(topo, cpus)
    out: set[int] = set()
    for core in cores:
        if len(out) + len(core) <= budget:
            out |= core
    if len(out) < budget:
        for core in cores:
            if not core & out:
                out |= set(sorted(core)[: budget - len(out)])
                if len(out) >= budget:
                    break
    return frozenset(out)


def trim_slices(
    topo: Topology,
    table: dict[str, GpuSlice],
    pcis,
    budget: int,
) -> frozenset[int]:
    """Trim a multi-GPU union: ``budget // k`` per GPU, remainder to the first."""

    pcis = list(dict.fromkeys(pcis))
    union, _ = slice_for(table, pcis)
    if budget >= len(union):
        return union
    count = len(pcis)
    shares = [budget // count] * count
    shares[0] += budget - sum(shares)
    out: set[int] = set()
    carry = 0
    for pci, share in zip(pcis, shares):
        own = table[pci].cpus - out
        take = min(len(own), share + carry)
        carry = share + carry - take
        out |= trim(topo, own, take)
    return frozenset(out)


def policy_id(
    mode: str,
    *,
    cpu_count: int | None = None,
    node_count: int | None = None,
    mems_mode: str = "bind",
    reserve_count: int = 0,
    undersize: str = "refuse",
) -> str:
    """Normalized, host-independent policy identity (never raw CPU IDs).

    ``auto:gpu-local-l3-v1:mems=bind|none[:reserve=<n>]`` or
    ``explicit:cpus=<n>:nodes=<n>|local:mems=bind|none``, where ``local``
    means "the nodes of the listed CPUs" (no ``@NODELIST``).
    ``:undersize=allow`` is appended only when a CPU budget larger than the
    slice may run on it.
    """

    if mode == "auto":
        parts = ["auto", ALGORITHM_ID, f"mems={mems_mode}"]
        if reserve_count:
            parts.append(f"reserve={reserve_count}")
    elif mode == "explicit":
        parts = [
            "explicit",
            f"cpus={cpu_count}",
            f"nodes={'local' if node_count is None else node_count}",
            f"mems={mems_mode}",
        ]
    else:
        raise ValueError(f"unknown CPU pinning mode {mode!r}")
    if undersize == "allow":
        parts.append("undersize=allow")
    return ":".join(parts)


# ---------------------------------------------------------------------------
# Container bindings
# ---------------------------------------------------------------------------


def nodes_of(topo: Topology, cpus) -> frozenset[int]:
    return frozenset(
        node for node in (topo.node_of(cpu) for cpu in cpus) if node is not None
    )


def validate_explicit(
    topo: Topology,
    cpus: frozenset[int],
    nodes: frozenset[int] | None,
    mems_mode: str,
) -> frozenset[int] | None:
    """Check an explicit CPU list against the host; return the mems to set."""

    if not cpus:
        raise TopologyError("explicit CPU list is empty")
    outside = cpus - topo.online
    if outside:
        raise TopologyError(
            f"explicit CPUs {format_cpulist(outside)} are not online "
            f"(online: {format_cpulist(topo.online)})"
        )
    if nodes is not None:
        missing = nodes - topo.all_nodes
        if missing:
            raise TopologyError(
                f"explicit NUMA nodes {format_cpulist(missing)} do not exist"
            )
    if mems_mode != "bind":
        return None
    cpu_nodes = nodes_of(topo, cpus)
    if nodes is None:
        return cpu_nodes or None
    remote = cpu_nodes - nodes
    if remote:
        raise TopologyError(
            f"explicit CPUs on NUMA nodes {format_cpulist(remote)} are "
            f"remote to the bound memory nodes {format_cpulist(nodes)}"
        )
    return nodes


def parse_visible_devices(value: str) -> tuple[str, ...]:
    """Parse ``NVIDIA_VISIBLE_DEVICES``: ``()`` for no GPU, ``("all",)``.

    Entries are canonical GPU UUIDs or NVML indices.  MIG devices are refused:
    a MIG slice has no CPU locality of its own.
    """

    text = value.strip()
    if text.lower() in {"", "void", "none"}:
        return ()
    if text.lower() == "all":
        return ("all",)
    ids: list[str] = []
    for item in text.split(","):
        item = item.strip()
        if _INDEX_RE.fullmatch(item):
            item = str(int(item))
        elif item.startswith("MIG-"):
            raise TopologyError(
                f"MIG device {item!r} cannot be pinned automatically; give "
                "an explicit CPULIST@NODES policy"
            )
        elif not _GPU_UUID_RE.fullmatch(item):
            raise TopologyError(
                f"unrecognized NVIDIA device {item!r} in NVIDIA_VISIBLE_DEVICES"
            )
        if item not in ids:
            ids.append(item)
    return tuple(ids)


# ---------------------------------------------------------------------------
# GPU identity -> PCI address
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProcGpu:
    pci: str
    uuid: str | None     # None when the driver printed '?'
    minor: int | None    # /dev/nvidia<minor>


def read_procfs_gpus(procfs: Path = Path("/proc")) -> dict[str, ProcGpu]:
    """Parse ``/proc/driver/nvidia/gpus/<bus>/information`` for every GPU."""

    root = Path(procfs) / "driver/nvidia/gpus"
    gpus: dict[str, ProcGpu] = {}
    for directory in sorted(root.glob("*")) if root.is_dir() else []:
        text = _read(directory / "information")
        if text is None:
            continue
        fields: dict[str, str] = {}
        for line in text.splitlines():
            key, sep, value = line.partition(":")
            if sep:
                fields[key.strip()] = value.strip()
        try:
            pci = normalize_pci(fields.get("Bus Location") or directory.name)
        except ValueError:
            continue
        uuid = fields.get("GPU UUID", "")
        minor_text = fields.get("Device Minor", "")
        gpus[pci] = ProcGpu(
            pci=pci,
            uuid=uuid if _GPU_UUID_RE.fullmatch(uuid) else None,
            minor=int(minor_text) if minor_text.isdigit() else None,
        )
    return gpus


@dataclass(frozen=True)
class InventoryGpu:
    index: int
    uuid: str
    pci: str


_NVML_INVENTORY_SCRIPT = r"""
import ctypes, ctypes.util, json, sys
lib = ctypes.CDLL(ctypes.util.find_library("nvidia-ml") or "libnvidia-ml.so.1")
class PciInfo(ctypes.Structure):
    _fields_ = [("busIdLegacy", ctypes.c_char * 16), ("domain", ctypes.c_uint),
                ("bus", ctypes.c_uint), ("device", ctypes.c_uint),
                ("pciDeviceId", ctypes.c_uint), ("pciSubSystemId", ctypes.c_uint),
                ("busId", ctypes.c_char * 32)]
def check(status, name):
    if status != 0:
        sys.exit(f"{name} failed with NVML status {status}")
check(lib.nvmlInit_v2(), "nvmlInit_v2")
try:
    count = ctypes.c_uint(0)
    check(lib.nvmlDeviceGetCount_v2(ctypes.byref(count)), "nvmlDeviceGetCount_v2")
    rows = []
    for index in range(count.value):
        handle = ctypes.c_void_p()
        check(lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(index), ctypes.byref(handle)),
              "nvmlDeviceGetHandleByIndex_v2")
        uuid = ctypes.create_string_buffer(96)
        check(lib.nvmlDeviceGetUUID(handle, uuid, ctypes.c_uint(96)), "nvmlDeviceGetUUID")
        pci = PciInfo()
        check(lib.nvmlDeviceGetPciInfo_v3(handle, ctypes.byref(pci)), "nvmlDeviceGetPciInfo_v3")
        rows.append({"index": index, "uuid": uuid.value.decode("ascii"),
                     "pci": pci.busId.decode("ascii")})
finally:
    lib.nvmlShutdown()
json.dump(rows, sys.stdout)
"""


def _run_inventory_command(argv: list[str], timeout: float) -> str:
    """Run one inventory child and return its stdout, bounded in time.

    Not ``subprocess.run``: after a timeout it kills the child and then waits
    for it without a bound, and a child stuck in the driver (uninterruptible
    sleep) never exits.  Here the wait after the kill is bounded as well; a
    child that still has not exited is abandoned, not waited for.
    """

    try:
        process = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        raise TopologyError(f"{argv[0]} could not run: {exc}") from exc
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        detail = ""
        try:
            process.wait(timeout=INVENTORY_REAP_SECONDS)
        except subprocess.TimeoutExpired:
            detail = "; the killed child did not exit and was abandoned"
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()
        raise TopologyError(
            f"{argv[0]} timed out after {timeout:g}s{detail}"
        ) from None
    if process.returncode != 0:
        detail = (stderr or stdout or "").strip()
        raise TopologyError(
            f"{argv[0]} exited {process.returncode}: {detail[-300:]}"
        )
    return stdout


def nvml_inventory(timeout: float = RESOLVE_TIMEOUT_SECONDS) -> tuple[InventoryGpu, ...]:
    """Index/UUID/PCI of every GPU via NVML in a short-lived child process.

    The child isolates a driver stall from the calling Judge/run process:
    :func:`_run_inventory_command` gives up on it after the timeout.
    """

    output = _run_inventory_command(
        [sys.executable, "-I", "-c", _NVML_INVENTORY_SCRIPT], timeout
    )
    try:
        rows = json.loads(output)
        return tuple(
            InventoryGpu(int(row["index"]), str(row["uuid"]), normalize_pci(row["pci"]))
            for row in rows
        )
    except (TypeError, KeyError, ValueError) as exc:
        raise TopologyError(f"NVML inventory output is malformed: {exc}") from exc


def nvidia_smi_inventory(
    timeout: float = RESOLVE_TIMEOUT_SECONDS,
) -> tuple[InventoryGpu, ...]:
    """Index/UUID/PCI of every GPU from one ``nvidia-smi`` query."""

    output = _run_inventory_command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id",
            "--format=csv,noheader",
        ],
        timeout,
    )
    rows: list[InventoryGpu] = []
    try:
        for line in output.splitlines():
            if not line.strip():
                continue
            index, uuid, pci = (field.strip() for field in line.split(","))
            rows.append(InventoryGpu(int(index), uuid, normalize_pci(pci)))
    except ValueError as exc:
        raise TopologyError(f"nvidia-smi output is malformed: {exc}") from exc
    return tuple(rows)


# The GPU -> PCI mapping is static for the life of the process.  Lower-case
# names containing "cache": the replay evaluator identity
# (candidate_replay._loaded_evaluator_code_sha256) hashes upper-case module
# bindings by value but treats *cache* globals as ephemeral, and these hold
# host GPU ids that must never enter an identity.
_resolution_cache: dict[tuple[str, str], str] = {}
_inventory_cache: dict[str, tuple[InventoryGpu, ...]] = {}


def clear_resolution_cache() -> None:
    _resolution_cache.clear()
    _inventory_cache.clear()


def _inventory(source: str, timeout: float) -> tuple[InventoryGpu, ...]:
    if source not in _inventory_cache:
        loader = nvml_inventory if source == "nvml" else nvidia_smi_inventory
        _inventory_cache[source] = loader(timeout)
    return _inventory_cache[source]


def _resolve_one(
    gpu_id: str,
    procfs_map: dict[str, ProcGpu],
    timeout: float,
) -> str:
    failures: list[str] = []
    if _INDEX_RE.fullmatch(gpu_id):
        field, wanted = "index", int(gpu_id)
    elif _GPU_UUID_RE.fullmatch(gpu_id):
        for item in procfs_map.values():
            if item.uuid == gpu_id:
                return item.pci
        field, wanted = "uuid", gpu_id
        failures.append("procfs: no match")
    elif _PCI_RE.fullmatch(gpu_id):
        return normalize_pci(gpu_id)
    else:
        raise TopologyError(
            f"GPU {gpu_id!r} is not a GPU UUID, NVML index or PCI address"
        )
    for source in ("nvml", "nvidia-smi"):
        try:
            rows = _inventory(source, timeout)
        except TopologyError as exc:
            failures.append(f"{source}: {exc}")
            continue
        for row in rows:
            if getattr(row, field) == wanted:
                return row.pci
        failures.append(f"{source}: no match")
    raise TopologyError(
        f"cannot resolve GPU {gpu_id} to a PCI address "
        f"({'; '.join(failures)}); give an explicit CPULIST@NODES policy "
        "instead of auto"
    )


def resolve_gpu_pcis(
    gpu_ids,
    *,
    procfs: Path = Path("/proc"),
    timeout: float = RESOLVE_TIMEOUT_SECONDS,
) -> tuple[str, ...]:
    """Map GPU UUIDs, NVML indices or PCI addresses to sysfs PCI addresses.

    UUIDs go through procfs first.  Indices are NVML indices (procfs has only
    the device minor), so they and unmapped UUIDs fall through to NVML and
    then ``nvidia-smi``.
    """

    procfs_map: dict[str, ProcGpu] | None = None
    resolved: list[str] = []
    for gpu_id in gpu_ids:
        gpu_id = str(gpu_id).strip()
        key = (str(procfs), gpu_id)
        if key not in _resolution_cache:
            if procfs_map is None:
                procfs_map = read_procfs_gpus(procfs)
            _resolution_cache[key] = _resolve_one(gpu_id, procfs_map, timeout)
        resolved.append(_resolution_cache[key])
    return tuple(resolved)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _reserve_arg(text: str) -> frozenset[int]:
    try:
        return parse_cpulist(text)
    except ValueError as exc:
        raise TopologyError(f"--reserve: {exc}") from exc


def _cmd_table(args) -> int:
    topo = read_topology(Path(args.sysfs))
    reserve = _reserve_arg(args.reserve)
    table = partition(topo, reserve)
    labels = {
        item.pci: None if item.minor is None else str(item.minor)
        for item in read_procfs_gpus(Path(args.procfs)).values()
    }
    uuids: dict[str, str] = {}
    if args.resolve:
        for source in ("nvml", "nvidia-smi"):
            try:
                rows = _inventory(source, RESOLVE_TIMEOUT_SECONDS)
            except TopologyError as exc:
                print(f"warning: {source}: {exc}", file=sys.stderr)
                continue
            labels = {row.pci: str(row.index) for row in rows}
            uuids = {row.pci: row.uuid for row in rows}
            break
    if args.json:
        payload = {
            "algorithm": ALGORITHM_ID,
            "reserve": format_cpulist(reserve),
            "topology_fingerprint": fingerprint(table),
            "gpus": [
                {
                    "gpu": labels.get(pci),
                    "pci": pci,
                    **({"uuid": uuids[pci]} if pci in uuids else {}),
                    "node": item.node,
                    "cpus": format_cpulist(item.cpus),
                    "cpu_count": len(item.cpus),
                    "cores": item.cores,
                    "l3_groups": list(item.l3_groups),
                    "l3_confined": item.l3_confined,
                    "node_spare": format_cpulist(item.node_spare),
                }
                for pci, item in sorted(table.items())
            ],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    for pci, item in sorted(table.items()):
        label = labels.get(pci)
        node = "-" if item.node is None else item.node
        row = f"GPU{'?' if label is None else label} {pci} {format_cpulist(item.cpus)} node{node}"
        if pci in uuids:
            row += f" {uuids[pci]}"
        print(row)
    return 0


def _cmd_slice(args) -> int:
    topo = read_topology(Path(args.sysfs))
    table = partition(topo, _reserve_arg(args.reserve))
    pcis = list(dict.fromkeys(resolve_gpu_pcis(args.gpu, procfs=Path(args.procfs))))
    cpus, _ = slice_for(table, pcis)
    slice_cpus = cpus
    trimmed = False
    if args.budget is not None:
        if args.budget < 1:
            raise TopologyError("--budget must be a positive CPU count")
        if args.budget < len(cpus):
            cpus = trim_slices(topo, table, pcis, args.budget)
            trimmed = True
        elif args.budget > len(cpus) and args.undersize != "allow":
            raise TopologyError(
                f"CPU budget {args.budget} exceeds the {len(cpus)}-CPU slice of "
                f"{', '.join(pcis)}; set SFORGE_CPUSET_UNDERSIZE=allow to run "
                "on the smaller slice"
            )
    mems = nodes_of(topo, cpus) if args.mems == "bind" else frozenset()
    if args.format == "cpulist":
        print(format_cpulist(cpus))
    elif args.format == "docker-args":
        words = ["--cpuset-cpus", format_cpulist(cpus)]
        if mems:
            words += ["--cpuset-mems", format_cpulist(mems)]
        print(" ".join(words))
    else:
        print(
            json.dumps(
                {
                    "algorithm": ALGORITHM_ID,
                    "gpus": pcis,
                    "slice_cpus": format_cpulist(slice_cpus),
                    "cpuset_cpus": format_cpulist(cpus),
                    "cpuset_mems": format_cpulist(mems) or None,
                    "cpu_count": len(cpus),
                    "trimmed_to_quota": trimmed,
                    "topology_fingerprint": fingerprint(table),
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


def _cmd_policy_id(args) -> int:
    from sforge.harness.cpu_affinity import policy_from_env

    policy = policy_from_env(args.role)
    if policy is not None:
        print(policy.policy_id)
    return 0


def _cmd_check(args) -> int:
    from sforge.harness.cpu_affinity import policy_from_env

    policy = policy_from_env(
        args.role, sysfs_root=Path(args.sysfs), procfs_root=Path(args.procfs)
    )
    if policy is None:
        print("off")
        return 0
    topo = read_topology(policy.sysfs_root)
    if policy.mode == "auto":
        table = partition(topo, policy.reserve)
        detail = fingerprint(table)
        if args.gpu:
            # The GPUs' untrimmed slice; the container budget trims it later.
            cpus, mems = slice_for(
                table, resolve_gpu_pcis(args.gpu, procfs=policy.procfs_root)
            )
            detail += f" cpus={format_cpulist(cpus)}"
            if policy.mems_mode == "bind":
                detail += f" mems={format_cpulist(mems)}"
    else:
        mems = validate_explicit(topo, policy.cpus, policy.nodes, policy.mems_mode)
        detail = f"cpus={format_cpulist(policy.cpus)} mems={format_cpulist(mems or ())}"
    print(f"ok {policy.policy_id} {detail}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m sforge.harness.host_topology",
        description=f"GPU-local CPU slices ({ALGORITHM_ID}), read from sysfs.",
    )
    parser.add_argument("--sysfs", default="/sys", help=argparse.SUPPRESS)
    parser.add_argument("--procfs", default="/proc", help=argparse.SUPPRESS)
    commands = parser.add_subparsers(dest="command", required=True)

    table = commands.add_parser("table", help="per-GPU CPU slices of this host")
    table.add_argument("--json", action="store_true")
    table.add_argument(
        "--reserve",
        default=_env("SFORGE_CPUSET_RESERVE"),
        help="CPUs kept out of the partition (default: $SFORGE_CPUSET_RESERVE)",
    )
    table.add_argument(
        "--resolve",
        action="store_true",
        help="label rows with NVML indices and UUIDs (runs NVML/nvidia-smi)",
    )

    slice_parser = commands.add_parser(
        "slice", help="cpuset for one container's GPUs (for raw docker run)"
    )
    slice_parser.add_argument(
        "--gpu", action="append", required=True, help="UUID, PCI address or NVML index"
    )
    slice_parser.add_argument("--budget", type=int, help="container CPU budget")
    slice_parser.add_argument(
        "--format", choices=("docker-args", "cpulist", "json"), default="docker-args"
    )
    slice_parser.add_argument("--reserve", default=_env("SFORGE_CPUSET_RESERVE"))
    slice_parser.add_argument(
        "--mems", choices=("bind", "none"), default=_env("SFORGE_CPUSET_MEMS", "bind")
    )
    slice_parser.add_argument(
        "--undersize",
        choices=("refuse", "allow"),
        default=_env("SFORGE_CPUSET_UNDERSIZE", "refuse"),
    )

    policy = commands.add_parser(
        "policy-id", help="normalized policy identity from the environment"
    )
    policy.add_argument("--role", choices=("work", "judge"))

    check = commands.add_parser(
        "check", help="validate the environment policy against this host"
    )
    check.add_argument("--role", choices=("work", "judge"))
    check.add_argument("--gpu", action="append", default=[])

    args = parser.parse_args(argv)
    handlers = {
        "table": _cmd_table,
        "slice": _cmd_slice,
        "policy-id": _cmd_policy_id,
        "check": _cmd_check,
    }
    try:
        return handlers[args.command](args)
    except (TopologyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    # Run the canonical module so its classes match sforge.harness.cpu_affinity.
    from sforge.harness.host_topology import main as _main

    raise SystemExit(_main())
