"""Opt-in GPU-local CPU pinning of Work and Judge containers.

Pinning is off unless a policy is configured, normally exported by
``launch/<task>/run.sh --cpuset`` or the launch config.  When it is off,
:func:`pin_before_start` returns without touching the backend, so container
creation, records and digests are exactly what they are without this module.

When it is on, the harness calls :func:`pin_before_start` between
``create_container`` and ``start_container`` of every Work and Judge
container.  The backend derives the container's GPUs from the created
container itself (its device requests, else ``NVIDIA_VISIBLE_DEVICES``), so
one rule covers task adapters that replace ``create_container``, formal
single-GPU handoff, legacy dual-GPU mode, dual-GPU tasks and the operator
eval/replay paths.  Every failure propagates: a container is never started
unpinned while a policy is set.

Policy, per role (``SFORGE_<ROLE>_CPUSET`` overrides ``SFORGE_CPUSET``)::

    off | auto | CPULIST[@NODELIST]

``auto`` is the union of the ``gpu-local-l3-v1`` slices of the container's
GPUs (see :mod:`sforge.harness.host_topology`), memory on their nodes.  An
explicit list without ``@`` binds memory to the nodes of the listed CPUs.
Modifiers: ``SFORGE_CPUSET_MEMS=bind|none``, ``SFORGE_CPUSET_RESERVE=CPULIST``
(kept out of the automatic partition) and ``SFORGE_CPUSET_UNDERSIZE=
refuse|allow`` (a CPU budget larger than the slice).  The CPU quota is kept
and the cpuset is trimmed to at most the budget, so a thread pool sized from
the visible CPUs no longer oversubscribes the quota.  A slice kept saturated
still sees brief edge throttling at the quota (about 0.1% of its CPU time
when measured on the reference host).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from sforge.harness import host_topology
from sforge.harness.host_topology import (
    TopologyError,
    format_cpulist,
    normalize_cpulist,
    parse_cpulist,
)

if TYPE_CHECKING:
    from sforge.harness.backend.base import ContainerBackend, ContainerHandle
    from sforge.harness.config import SForgeConfig


ROLES = ("work", "judge")
CPUSET_ENV = "SFORGE_CPUSET"
ROLE_CPUSET_ENV = {
    "work": "SFORGE_WORK_CPUSET",
    "judge": "SFORGE_JUDGE_CPUSET",
}
MEMS_ENV = "SFORGE_CPUSET_MEMS"
RESERVE_ENV = "SFORGE_CPUSET_RESERVE"
UNDERSIZE_ENV = "SFORGE_CPUSET_UNDERSIZE"
MEMS_MODES = ("bind", "none")
UNDERSIZE_MODES = ("refuse", "allow")
RECORD_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Policy grammar and per-role resolution
# ---------------------------------------------------------------------------


def _clean(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value.strip() or None


def parse_policy_value(
    value: str | None,
    *,
    name: str = CPUSET_ENV,
) -> tuple[str, frozenset[int] | None, frozenset[int] | None]:
    """Parse ``off | auto | CPULIST[@NODELIST]`` into (mode, cpus, nodes).

    Unset or blank means ``off``.
    """

    text = _clean(value, name)
    if text is None or text == "off":
        return "off", None, None
    if text == "auto":
        return "auto", None, None
    cpus_text, separator, nodes_text = text.partition("@")
    problem = None
    if not cpus_text or (separator and not nodes_text):
        problem = "an empty CPU or node list"
    else:
        try:
            cpus = parse_cpulist(cpus_text)
            nodes = parse_cpulist(nodes_text) if separator else None
        except ValueError as exc:
            problem = str(exc)
    if problem is not None:
        raise ValueError(
            f"{name} must be off, auto or CPULIST[@NODELIST] "
            f"(e.g. 8-11,40-43@0), not {value!r}: {problem}"
        )
    return "explicit", cpus, nodes


def _enum(value: object, choices: tuple[str, ...], name: str) -> str:
    text = _clean(value, name)
    if text is None:
        return choices[0]
    if text not in choices:
        raise ValueError(
            f"{name} must be one of {'/'.join(choices)}, not {value!r}"
        )
    return text


def _reserve(value: object) -> frozenset[int]:
    text = _clean(value, RESERVE_ENV)
    try:
        return parse_cpulist(text or "")
    except ValueError as exc:
        raise ValueError(f"{RESERVE_ENV} must be a CPULIST: {exc}") from exc


@dataclass(frozen=True)
class CpuAffinityPolicy:
    """The resolved, enabled policy of one role."""

    role: str | None
    value: str                        # canonical: auto | CPULIST[@NODELIST]
    source_env: str
    mode: str                         # auto | explicit
    cpus: frozenset[int] | None       # explicit only
    nodes: frozenset[int] | None      # explicit with @NODELIST only
    mems_mode: str
    reserve: frozenset[int]
    undersize: str
    sysfs_root: Path = field(default=Path("/sys"), compare=False)
    procfs_root: Path = field(default=Path("/proc"), compare=False)

    @property
    def policy_id(self) -> str:
        return host_topology.policy_id(
            self.mode,
            cpu_count=None if self.cpus is None else len(self.cpus),
            node_count=None if self.nodes is None else len(self.nodes),
            mems_mode=self.mems_mode,
            reserve_count=len(self.reserve) if self.mode == "auto" else 0,
            undersize=self.undersize,
        )

    def describe(self) -> dict[str, object]:
        return {
            "value": self.value,
            "source_env": self.source_env,
            "mems_mode": self.mems_mode,
            "reserve": format_cpulist(self.reserve),
            "undersize": self.undersize,
        }


def resolve_policy(
    role: str | None,
    values: Mapping[str, object],
    *,
    sysfs_root: Path = Path("/sys"),
    procfs_root: Path = Path("/proc"),
) -> CpuAffinityPolicy | None:
    """Resolve one role's policy from env-named values; ``None`` when off.

    ``role=None`` resolves the global ``SFORGE_CPUSET`` alone.  Every value,
    including the modifiers, is validated even when the policy is off.
    """

    if role is not None and role not in ROLES:
        raise ValueError(f"unknown CPU pinning role {role!r}")
    source = CPUSET_ENV
    if role is not None:
        role_env = ROLE_CPUSET_ENV[role]
        if _clean(values.get(role_env), role_env) is not None:
            source = role_env
    mode, cpus, nodes = parse_policy_value(values.get(source), name=source)
    mems_mode = _enum(values.get(MEMS_ENV), MEMS_MODES, MEMS_ENV)
    reserve = _reserve(values.get(RESERVE_ENV))
    undersize = _enum(values.get(UNDERSIZE_ENV), UNDERSIZE_MODES, UNDERSIZE_ENV)
    if mode == "off":
        return None
    if mode == "auto":
        value = "auto"
    else:
        value = format_cpulist(cpus)
        if nodes is not None:
            value += "@" + format_cpulist(nodes)
            if mems_mode == "none":
                raise ValueError(
                    f"{source}={value} names memory nodes, but "
                    f"{MEMS_ENV}=none binds no memory; drop @NODELIST or "
                    "use bind"
                )
    return CpuAffinityPolicy(
        role=role,
        value=value,
        source_env=source,
        mode=mode,
        cpus=cpus,
        nodes=nodes,
        mems_mode=mems_mode,
        reserve=reserve,
        undersize=undersize,
        sysfs_root=Path(sysfs_root),
        procfs_root=Path(procfs_root),
    )


def config_values(config: "SForgeConfig") -> dict[str, object]:
    """The config's pinning fields, keyed by the env names they map from."""

    return {
        CPUSET_ENV: config.cpuset,
        ROLE_CPUSET_ENV["work"]: config.work_cpuset,
        ROLE_CPUSET_ENV["judge"]: config.judge_cpuset,
        MEMS_ENV: config.cpuset_mems_mode,
        RESERVE_ENV: config.cpuset_reserve,
        UNDERSIZE_ENV: config.cpuset_undersize,
    }


def policy_for_role(
    role: str,
    config: "SForgeConfig",
    *,
    sysfs_root: Path = Path("/sys"),
    procfs_root: Path = Path("/proc"),
) -> CpuAffinityPolicy | None:
    return resolve_policy(
        role,
        config_values(config),
        sysfs_root=sysfs_root,
        procfs_root=procfs_root,
    )


def policy_from_env(
    role: str | None = None,
    environ: Mapping[str, str] | None = None,
    *,
    sysfs_root: Path = Path("/sys"),
    procfs_root: Path = Path("/proc"),
) -> CpuAffinityPolicy | None:
    """Same resolution from the process environment (for shell scripts)."""

    environ = os.environ if environ is None else environ
    names = (
        CPUSET_ENV,
        *ROLE_CPUSET_ENV.values(),
        MEMS_ENV,
        RESERVE_ENV,
        UNDERSIZE_ENV,
    )
    return resolve_policy(
        role,
        {name: environ.get(name) for name in names},
        sysfs_root=sysfs_root,
        procfs_root=procfs_root,
    )


def validate_policy_values(
    values: Mapping[str, object],
    *,
    backend: str,
) -> None:
    """Grammar and backend checks at config load; never touches hardware."""

    # The global value is validated even when both roles override it.
    parse_policy_value(values.get(CPUSET_ENV), name=CPUSET_ENV)
    for role in ROLES:
        policy = resolve_policy(role, values)
        if policy is not None and backend != "docker":
            raise ValueError(
                f"CPU pinning ({policy.source_env}={policy.value}) requires "
                f"the Docker backend, not {backend!r}"
            )


def cpu_affinity_identity(config: "SForgeConfig") -> dict[str, str]:
    """``{role: policy_id}`` for every enabled role; empty when off."""

    identity: dict[str, str] = {}
    for role in ROLES:
        policy = policy_for_role(role, config)
        if policy is not None:
            identity[role] = policy.policy_id
    return identity


# ---------------------------------------------------------------------------
# Planning one container's binding (backend-neutral)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContainerCpuState:
    """What a backend observed on the created, not yet started container."""

    gpu_request_ids: tuple[str, ...] = ()
    gpu_request_unspecified: bool = False   # a GPU request by count / all
    # Device requests that name devices but are not recognizably NVIDIA GPU
    # requests (e.g. CDI); auto pinning refuses them rather than skip them.
    unrecognized_device_requests: tuple[str, ...] = ()
    visible_devices: str | None = None      # NVIDIA_VISIBLE_DEVICES
    nano_cpus: int = 0
    cpu_quota: int = 0
    cpu_period: int = 0
    cpuset_cpus: str = ""
    cpuset_mems: str = ""

    @property
    def cpu_budget(self) -> int | None:
        """Whole CPUs of the container's quota, or ``None`` without one."""

        if self.nano_cpus > 0:
            return max(1, self.nano_cpus // 1_000_000_000)
        if self.cpu_quota > 0:
            return max(1, self.cpu_quota // (self.cpu_period or 100_000))
        return None


@dataclass(frozen=True)
class CpuBindingPlan:
    outcome: str                  # pinned | task_owned | unpinned_no_gpu
    cpuset_cpus: str | None
    cpuset_mems: str | None
    nano_cpus: int
    cpu_budget: int | None
    gpu_source: str | None = None
    gpus: tuple[dict[str, object], ...] = ()
    slice_cpus: str | None = None
    trimmed_to_quota: bool = False
    undersized: bool = False
    cross_node: bool = False
    algorithm: str | None = None
    topology_fingerprint: str | None = None

    @property
    def requires_update(self) -> bool:
        return self.outcome == "pinned"

    def record(self) -> dict[str, object]:
        return {
            "outcome": self.outcome,
            "gpu_source": self.gpu_source,
            "gpus": [dict(gpu) for gpu in self.gpus],
            "slice_cpus": self.slice_cpus,
            "cpuset_cpus": self.cpuset_cpus,
            "cpuset_mems": self.cpuset_mems,
            "cpu_count": (
                len(parse_cpulist(self.cpuset_cpus))
                if self.cpuset_cpus
                else None
            ),
            "nano_cpus": self.nano_cpus,
            "cpu_budget": self.cpu_budget,
            "trimmed_to_quota": self.trimmed_to_quota,
            "undersized": self.undersized,
            "cross_node": self.cross_node,
            "algorithm": self.algorithm,
            "topology_fingerprint": self.topology_fingerprint,
        }


def container_gpu_ids(state: ContainerCpuState) -> tuple[tuple[str, ...], str]:
    """The container's GPU IDs and where they came from.

    Device requests are authoritative: the NVIDIA runtime appends their
    ``NVIDIA_VISIBLE_DEVICES`` after the image's own.  An environment value
    that names other specific devices is a conflict; an image default such as
    ``all`` or ``void`` is overridden by the request and ignored.
    """

    if state.unrecognized_device_requests:
        raise TopologyError(
            "the container has device requests auto pinning cannot map to "
            f"NVIDIA GPUs ({', '.join(state.unrecognized_device_requests)}); "
            "give an explicit CPULIST@NODES policy"
        )
    env_ids = (
        None
        if state.visible_devices is None
        else host_topology.parse_visible_devices(state.visible_devices)
    )
    if state.gpu_request_unspecified:
        raise TopologyError(
            "the container requests GPUs by count or 'all'; auto pinning "
            "needs exact device IDs (give an explicit CPULIST@NODES policy)"
        )
    if state.gpu_request_ids:
        request_ids = host_topology.parse_visible_devices(
            ",".join(state.gpu_request_ids)
        )
        if request_ids in ((), ("all",)):
            raise TopologyError(
                f"unusable GPU device request {list(state.gpu_request_ids)!r}"
            )
        return request_ids, "device_requests"
    if not env_ids:
        return (), "none"
    if env_ids == ("all",):
        raise TopologyError(
            "NVIDIA_VISIBLE_DEVICES=all; auto pinning needs exact device IDs "
            "(give an explicit CPULIST@NODES policy)"
        )
    return env_ids, "nvidia_visible_devices"


def plan_cpu_binding(
    policy: CpuAffinityPolicy,
    state: ContainerCpuState,
) -> CpuBindingPlan:
    """Compute the cpuset for one created container (reads sysfs/procfs)."""

    budget = state.cpu_budget
    try:
        preset_cpus = normalize_cpulist(state.cpuset_cpus)
        preset_mems = normalize_cpulist(state.cpuset_mems)
    except ValueError as exc:
        raise TopologyError(f"container has a malformed preset cpuset: {exc}") from exc
    if preset_cpus:
        # A task adapter pinned the container itself (Miles SD3.5/Wan).
        if policy.mode == "explicit" and parse_cpulist(preset_cpus) != policy.cpus:
            raise TopologyError(
                f"{policy.source_env}={policy.value} conflicts with the "
                f"task-owned cpuset {preset_cpus} of this container"
            )
        return CpuBindingPlan(
            outcome="task_owned",
            cpuset_cpus=preset_cpus,
            cpuset_mems=preset_mems or None,
            nano_cpus=state.nano_cpus,
            cpu_budget=budget,
        )
    if preset_mems:
        raise TopologyError(
            f"the container presets CpusetMems={preset_mems} without "
            "CpusetCpus; refusing to override it"
        )

    gpu_source = None
    gpus: tuple[dict[str, object], ...] = ()
    table: dict[str, host_topology.GpuSlice] = {}
    pcis: list[str] = []
    if policy.mode == "auto":
        gpu_ids, gpu_source = container_gpu_ids(state)
        if not gpu_ids:
            return CpuBindingPlan(
                outcome="unpinned_no_gpu",
                cpuset_cpus=None,
                cpuset_mems=None,
                nano_cpus=state.nano_cpus,
                cpu_budget=budget,
                gpu_source=gpu_source,
            )
        topo = host_topology.read_topology(policy.sysfs_root)
        table = host_topology.partition(topo, policy.reserve)
        resolved = host_topology.resolve_gpu_pcis(
            gpu_ids, procfs=policy.procfs_root
        )
        if gpu_source == "device_requests" and state.visible_devices is not None:
            env_ids = host_topology.parse_visible_devices(state.visible_devices)
            if env_ids not in ((), ("all",)) and set(env_ids) != set(gpu_ids):
                env_pcis = host_topology.resolve_gpu_pcis(
                    env_ids, procfs=policy.procfs_root
                )
                if set(env_pcis) != set(resolved):
                    raise TopologyError(
                        f"device request GPUs {list(gpu_ids)} disagree with "
                        f"NVIDIA_VISIBLE_DEVICES={state.visible_devices}"
                    )
        pcis = list(dict.fromkeys(resolved))
        slice_cpus, _ = host_topology.slice_for(table, pcis)
        gpus = tuple(
            {"id": gpu_id, "pci": pci, "numa_node": table[pci].node}
            for gpu_id, pci in zip(gpu_ids, resolved)
        )
    else:
        topo = host_topology.read_topology(policy.sysfs_root)
        assert policy.cpus is not None
        host_topology.validate_explicit(
            topo, policy.cpus, policy.nodes, policy.mems_mode
        )
        slice_cpus = policy.cpus

    cpus = slice_cpus
    trimmed = undersized = False
    if budget is not None and budget < len(cpus):
        cpus = (
            host_topology.trim_slices(topo, table, pcis, budget)
            if policy.mode == "auto"
            else host_topology.trim(topo, cpus, budget)
        )
        trimmed = True
    elif budget is not None and budget > len(cpus):
        if policy.undersize != "allow":
            raise TopologyError(
                f"the container's CPU budget of {budget} exceeds its "
                f"{len(cpus)}-CPU slice {format_cpulist(cpus)}; set "
                f"{UNDERSIZE_ENV}=allow to run it on the smaller slice, or "
                f"leave this task unpinned"
            )
        undersized = True
    cpu_nodes = host_topology.nodes_of(topo, cpus)
    mems: frozenset[int] | None = None
    if policy.mems_mode == "bind":
        mems = policy.nodes if policy.nodes is not None else cpu_nodes
    return CpuBindingPlan(
        outcome="pinned",
        cpuset_cpus=format_cpulist(cpus),
        cpuset_mems=format_cpulist(mems) if mems else None,
        nano_cpus=state.nano_cpus,
        cpu_budget=budget,
        gpu_source=gpu_source,
        gpus=gpus,
        slice_cpus=format_cpulist(slice_cpus),
        trimmed_to_quota=trimmed,
        undersized=undersized,
        cross_node=len(cpu_nodes) > 1,
        algorithm=host_topology.ALGORITHM_ID if policy.mode == "auto" else None,
        topology_fingerprint=(
            host_topology.fingerprint(table) if policy.mode == "auto" else None
        ),
    )


# ---------------------------------------------------------------------------
# Harness entry point
# ---------------------------------------------------------------------------


def write_cpu_affinity_record(path: Path, record: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def pin_before_start(
    backend: "ContainerBackend",
    handle: "ContainerHandle",
    *,
    role: str,
    config: "SForgeConfig",
    record_path: Path,
    logger: logging.Logger | None = None,
) -> dict[str, object] | None:
    """Pin a created container before it starts; ``None`` when pinning is off.

    With no policy for ``role`` the backend is not touched at all.  Otherwise
    the backend applies and verifies the binding and a host-side record is
    written; any failure raises, so the caller never starts the container.
    """

    policy = policy_for_role(role, config)
    if policy is None:
        return None
    apply_cpu_binding = getattr(backend, "apply_cpu_binding", None)
    if apply_cpu_binding is None:
        raise NotImplementedError(
            f"{getattr(backend, 'backend_name', type(backend).__name__)} "
            f"backend cannot apply CPU pinning ({policy.source_env}="
            f"{policy.value})"
        )
    binding = apply_cpu_binding(handle, policy)
    record: dict[str, object] = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "role": role,
        "container_id": handle.id,
        "container_name": handle.name,
        "policy": policy.describe(),
        "policy_id": policy.policy_id,
        **binding,
    }
    write_cpu_affinity_record(record_path, record)
    if logger is not None:
        logger.info(
            "CPU pinning (%s): %s cpus=%s mems=%s budget=%s policy=%s",
            role,
            record.get("outcome"),
            record.get("cpuset_cpus"),
            record.get("cpuset_mems"),
            record.get("cpu_budget"),
            policy.policy_id,
        )
    return record
