"""Shared fixtures for the harness tests.

``fake_host`` builds a synthetic sysfs/procfs tree for the CPU pinning tests
(sforge/harness/host_topology.py), so no test reads the real host.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import pytest

from sforge.harness.host_topology import format_cpulist


def pytest_configure(config):
    config.addinivalue_line("markers", "pyromind_sdk: needs the optional pyromind-sdk package")


def pytest_collection_modifyitems(config, items):
    # pyromind-sdk is in the optional pyromind extra, not in `uv sync --dev`.
    if importlib.util.find_spec("pyromind_sdk") is not None:
        return
    skip = pytest.mark.skip(reason="needs the pyromind extra (pyromind-sdk)")
    for item in items:
        if "pyromind_sdk" in item.keywords:
            item.add_marker(skip)


# The shape of the reference H100 NVL host: 2 sockets x 16 cores, SMT2
# (sibling of CPU N is N+32), an L3 group per 4 cores, 4 GPUs per node.
H100_NODES = (16, 16)
H100_GPUS = (
    ("0000:46:00.0", 0),
    ("0000:49:00.0", 0),
    ("0000:4e:00.0", 0),
    ("0000:4f:00.0", 0),
    ("0000:c5:00.0", 1),
    ("0000:c6:00.0", 1),
    ("0000:c9:00.0", 1),
    ("0000:d0:00.0", 1),
)


def fake_uuid(index: int) -> str:
    return f"GPU-{index:08x}-1111-2222-3333-{index:012x}"


@dataclass(frozen=True)
class FakeHost:
    sysfs: Path
    procfs: Path


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="ascii")


def build_fake_host(
    root: Path,
    *,
    nodes=H100_NODES,
    smt: int = 2,
    l3_cores: int = 4,
    gpus=H100_GPUS,
    nonuma: bool = False,
    offline=(),
    cpuless_nodes=(),
    shuffle_l3_ids: bool = False,
    cgroup_effective: str | None = None,
    procfs_uuids: bool = True,
) -> FakeHost:
    """nodes: physical cores per node; l3_cores: cores per L3 group (0 = one
    group per node); gpus: (pci, node or -1) pairs, UUID/minor by position."""

    sysfs = root / "sys"
    procfs = root / "proc"
    total_cores = sum(nodes)
    ncpu = total_cores * smt
    cpu_root = sysfs / "devices/system/cpu"
    online = sorted(set(range(ncpu)) - set(offline))
    _write(cpu_root / "online", format_cpulist(online))
    if cgroup_effective is not None:
        _write(sysfs / "fs/cgroup/cpuset.cpus.effective", cgroup_effective)
    node_cpus: dict[int, list[int]] = {}
    core_base = 0
    for node, cores in enumerate(nodes):
        core_ids = list(range(core_base, core_base + cores))
        cpus = [c + t * total_cores for c in core_ids for t in range(smt)]
        node_cpus[node] = cpus
        if not nonuma:
            _write(
                sysfs / f"devices/system/node/node{node}/cpulist",
                format_cpulist(cpus),
            )
        group = l3_cores or cores
        for core in core_ids:
            siblings = [core + t * total_cores for t in range(smt)]
            first = core_base + ((core - core_base) // group) * group
            group_cores = range(first, min(first + group, core_base + cores))
            shared = [c + t * total_cores for c in group_cores for t in range(smt)]
            # Cache ids deliberately out of CPU order when asked to.
            cache_id = (first // group) ^ 1 if shuffle_l3_ids else first // group
            for cpu in siblings:
                base = cpu_root / f"cpu{cpu}"
                _write(
                    base / "topology/thread_siblings_list",
                    format_cpulist(siblings),
                )
                _write(base / "cache/index0/level", "1")
                _write(base / "cache/index0/type", "Data")
                _write(base / "cache/index0/shared_cpu_list", format_cpulist(siblings))
                _write(base / "cache/index3/level", "3")
                _write(base / "cache/index3/type", "Unified")
                _write(base / "cache/index3/id", str(cache_id))
                _write(base / "cache/index3/shared_cpu_list", format_cpulist(shared))
        core_base += cores
    for node in cpuless_nodes:
        _write(sysfs / f"devices/system/node/node{node}/cpulist", "")
    for index, (pci, node) in enumerate(gpus):
        device = sysfs / "bus/pci/devices" / pci
        _write(device / "vendor", "0x10de")
        _write(device / "class", "0x030200")
        _write(device / "numa_node", str(node))
        local = node_cpus.get(node, list(range(ncpu)))
        _write(device / "local_cpulist", format_cpulist(local))
        info = procfs / "driver/nvidia/gpus" / pci / "information"
        _write(
            info,
            "\n".join(
                (
                    "Model: \t\t NVIDIA H100 NVL",
                    f"GPU UUID: \t {fake_uuid(index) if procfs_uuids else '?'}",
                    f"Bus Location: \t {pci}",
                    f"Device Minor: \t {index}",
                )
            ),
        )
    # An NVIDIA bridge-class function (NVSwitch-like) that is not a GPU.
    bridge = sysfs / "bus/pci/devices/0001:ff:00.0"
    _write(bridge / "vendor", "0x10de")
    _write(bridge / "class", "0x068000")
    _write(bridge / "numa_node", "0")
    # A non-NVIDIA display controller.
    other = sysfs / "bus/pci/devices/0001:00:01.0"
    _write(other / "vendor", "0x1a03")
    _write(other / "class", "0x030000")
    return FakeHost(sysfs=sysfs, procfs=procfs)


@pytest.fixture
def fake_host(tmp_path: Path):
    """Factory: ``fake_host(**shape)`` -> FakeHost(sysfs, procfs)."""

    from sforge.harness import host_topology

    counter = iter(range(1_000_000))

    def build(**kwargs) -> FakeHost:
        return build_fake_host(tmp_path / f"host{next(counter)}", **kwargs)

    host_topology.clear_resolution_cache()
    yield build
    host_topology.clear_resolution_cache()
