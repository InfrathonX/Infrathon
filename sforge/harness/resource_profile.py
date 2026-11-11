"""One CPU/memory size per task, shared by PyroMind Sandboxes and local Docker.

``deploy/pyromind/<deployment>/deployment.json`` is the only place a task's
Worker and Judge size is written down (``resources``, or the complete
``worker_resources``/``judge_resources`` overrides); its integrity-review
Sandbox uses ``review_resources`` or the 4 CPU / 8Gi default.  PyroMind
requests those sizes directly.  Local Docker entry points (``sforge run``,
``eval``, ``replay-*``, Judge-server registration and local review
containers) call into this module, so a task's local containers get the same
CPU quota and memory limit even though the task scripts pass their own
``--*-cpu-limit``/``--*-mem-limit`` flags.  Replaced values are recorded.
``SFORGE_RESOURCE_PROFILE=native`` keeps the task's own limits.

``sforge/harness/resource_profiles.json`` is the published copy of those sizes
(``python -m sforge.harness.resource_profile --write-table`` writes it from the
deployment profiles); a checkout without ``deploy/pyromind`` reads it instead.

See deploy/pyromind/RESOURCES.md.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

ENV = "SFORGE_RESOURCE_PROFILE"
MODES = ("native", "pyromind")
SCHEMA_VERSION = 1
REPO = Path(__file__).resolve().parents[2]
DEPLOYMENTS = Path("deploy/pyromind")
TABLE = Path("sforge/harness/resource_profiles.json")
ROLE_NAMES = {"work": "worker", "judge": "judge"}
UNDERSIZE_ENV = "SFORGE_CPUSET_UNDERSIZE"

# Deployments whose local runs keep the task's own launch sizes.
LOCAL_EXEMPT = {
    "flux-moe-ag": (
        "local Flux runs pin their own 32 CPU / 128g role profile; "
        "deploy/pyromind/flux-moe-ag/local_entry.py refuses any other size"
    ),
}

_WHOLE_GI = re.compile(r"([1-9][0-9]*)Gi")


class ResourceProfileError(ValueError):
    """The deployment profiles do not give a task one unambiguous size."""


def docker_memory(value: str) -> str:
    """PyroMind's ``16Gi`` in Docker's unit: ``16g`` is the same 16 * 1024**3 bytes."""

    match = _WHOLE_GI.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise ResourceProfileError(f"expected a whole number of Gi, not {value!r}")
    return match[1] + "g"


@dataclass(frozen=True)
class RoleSize:
    cpu: int
    memory: str  # Docker form, e.g. "16g"
    gpu: int

    def limits(self) -> dict[str, Any]:
        return {"cpu": self.cpu, "memory": self.memory}


@dataclass(frozen=True)
class TaskProfile:
    task_id: str
    deployment: str
    source: str
    sha256: str
    work: RoleSize
    judge: RoleSize
    review: RoleSize
    exempt: str | None = None

    def role(self, role: str) -> RoleSize:
        return {"work": self.work, "judge": self.judge, "review": self.review}[role]

    def identity(self) -> dict[str, str]:
        return {"deployment": self.deployment, "source": self.source, "sha256": self.sha256}


def _size(resources: Mapping[str, Any]) -> RoleSize:
    return RoleSize(int(resources["cpu"]), docker_memory(resources["memory"]),
                    int(resources.get("gpu", 0)))


@functools.lru_cache(maxsize=None)
def _index(repo: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted((repo / DEPLOYMENTS).glob("*/deployment.json")):
        try:
            task_id = json.loads(path.read_text())["task_id"]
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ResourceProfileError(f"unreadable deployment profile {path}: {error}") from error
        if not isinstance(task_id, str) or not task_id:
            raise ResourceProfileError(f"deployment profile {path} names no task")
        if task_id in index:
            raise ResourceProfileError(
                f"task {task_id} has two deployment profiles: {index[task_id]} and {path}")
        index[task_id] = path
    return index


def _deployment_profile(task_id: str, repo: Path) -> TaskProfile | None:
    from sforge.pyromind.deployment import Deployment
    from sforge.pyromind.integrity_review import review_resources

    path = _index(repo).get(task_id)
    if path is None:
        return None
    deployment = Deployment.load(path, repo)
    name = deployment.config["name"]
    return TaskProfile(
        task_id=task_id,
        deployment=name,
        source=str(path.resolve().relative_to(repo)),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        work=_size(deployment.resources_for_role("worker")),
        judge=_size(deployment.resources_for_role("judge")),
        review=_size(review_resources(deployment)),
        exempt=LOCAL_EXEMPT.get(name),
    )


def _table_profile(task_id: str, repo: Path) -> TaskProfile | None:
    path = repo / TABLE
    try:
        data = path.read_bytes()
        entry = json.loads(data)["tasks"].get(task_id)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise ResourceProfileError(f"unreadable resource profile table {path}: {error}") from error
    if entry is None:
        return None
    return TaskProfile(
        task_id=task_id,
        deployment=entry["deployment"],
        source=str(TABLE),
        sha256=hashlib.sha256(data).hexdigest(),
        work=_size(entry["work"]),
        judge=_size(entry["judge"]),
        review=_size(entry["review"]),
        exempt=LOCAL_EXEMPT.get(entry["deployment"]),
    )


@functools.lru_cache(maxsize=None)
def profile_for_task(task_id: str, repo: Path = REPO) -> TaskProfile | None:
    """The task's sizes, or None when no deployment profile names the task.

    The deployment profiles decide when the checkout has them; otherwise the
    published table (``TABLE``) does."""

    repo = Path(repo).resolve()
    if (repo / DEPLOYMENTS).is_dir():
        return _deployment_profile(task_id, repo)
    if (repo / TABLE).is_file():
        return _table_profile(task_id, repo)
    raise ResourceProfileError(
        f"{repo / DEPLOYMENTS} and {repo / TABLE} are missing; "
        f"set {ENV}=native to use the task's own limits")


def table(repo: Path = REPO) -> dict[str, Any]:
    """The published table: every deployment profile's role sizes, by task."""

    repo = Path(repo).resolve()
    tasks: dict[str, Any] = {}
    for task_id in sorted(_index(repo)):
        profile = _deployment_profile(task_id, repo)
        tasks[task_id] = {"deployment": profile.deployment, **{
            role: {"cpu": size.cpu, "memory": size.memory[:-1] + "Gi", "gpu": size.gpu}
            for role, size in (("work", profile.work), ("judge", profile.judge),
                               ("review", profile.review))}}
    return {"schema_version": SCHEMA_VERSION,
            "note": "Work, Judge and integrity-review sizes of local Docker runs in "
                    "SFORGE_RESOURCE_PROFILE=pyromind mode (the default); written by "
                    "python -m sforge.harness.resource_profile --write-table.",
            "tasks": tasks}


def mode_of(config) -> str:
    mode = getattr(config, "resource_profile", "native") or "native"
    if mode not in MODES:
        raise ResourceProfileError(f"{ENV} must be one of {'/'.join(MODES)}, not {mode!r}")
    return mode


def local_profile(config, task_spec) -> TaskProfile | None:
    """The profile a local Docker container of this task follows, if any.

    None in native mode, for other backends, for tasks without a deployment
    profile and for exempt deployments (see :func:`exemption`)."""

    if mode_of(config) != "pyromind" or (getattr(config, "backend", None) or "docker") != "docker":
        return None
    profile = profile_for_task(task_spec.task_id)
    if profile is None or profile.exempt:
        return None
    return profile


def exemption(config, task_spec) -> dict[str, Any] | None:
    """Record for an exempt deployment in pyromind mode (shown, not applied)."""

    if mode_of(config) != "pyromind" or (getattr(config, "backend", None) or "docker") != "docker":
        return None
    profile = profile_for_task(task_spec.task_id)
    if profile is None or not profile.exempt:
        return None
    return {"schema_version": SCHEMA_VERSION, "mode": "pyromind", "status": "exempt",
            "task_id": task_spec.task_id, "reason": profile.exempt, **profile.identity()}


def _gpu_local_slice(config, role: str, gpus: int) -> int | None:
    """CPUs of the smallest GPU-local slice times the role's GPU count.

    None unless the role is pinned with ``auto`` on a readable topology."""

    from sforge.harness import host_topology
    from sforge.harness.cpu_affinity import policy_for_role

    policy = policy_for_role(role, config)
    if policy is None or policy.mode != "auto" or gpus < 1:
        return None
    try:
        table = host_topology.partition(host_topology.read_topology(policy.sysfs_root),
                                        policy.reserve)
    except (host_topology.TopologyError, OSError):
        return None
    if not table:
        return None
    return min(len(item.cpus) for item in table.values()) * gpus


def undersize_decision(config, task_spec, profile: TaskProfile,
                       roles: tuple[str, ...]) -> dict[str, Any] | None:
    """Pin an over-slice quota to its GPU-local slice, never below native CPUs.

    A profile CPU count above the GPU-local slice (8 logical CPUs per H100 on
    the reference host) is refused by ``--cpuset auto`` unless
    ``SFORGE_CPUSET_UNDERSIZE=allow``.  When every such role's own task CPU
    count still fits its slice, pinning to the slice leaves the container at
    least the CPUs the task was designed for, so undersize is allowed;
    otherwise (qvg: 16 native CPUs on one GPU) pinning stays refused.  An
    explicit SFORGE_CPUSET_UNDERSIZE always wins."""

    if os.environ.get(UNDERSIZE_ENV, "").strip():
        return None
    roles_over: dict[str, dict[str, Any]] = {}
    for role in roles:
        size = profile.role(role)
        slice_cpus = _gpu_local_slice(config, role, size.gpu)
        if slice_cpus is None or size.cpu <= slice_cpus:
            continue
        roles_over[role] = {"profile_cpu": size.cpu, "slice_cpus": slice_cpus,
                            "native_cpu": getattr(task_spec, role).cpu_limit}
    if not roles_over:
        return None
    fits = all(item["native_cpu"] is not None and item["native_cpu"] <= item["slice_cpus"]
               for item in roles_over.values())
    return {"value": "allow" if fits else "refuse", "roles": roles_over,
            "reason": ("pinned to the GPU-local slice, which still holds the task's own CPU count"
                       if fits else
                       "a role's own CPU count exceeds its GPU-local slice; pinning stays refused")}


def apply_local_profile(config, task_spec, roles: tuple[str, ...] = ("work", "judge")):
    """Return ``(config, record)`` with the task's profile applied to ``roles``.

    The profile replaces any CLI/env/experiment/task-JSON limit for those
    roles.  ``record`` is None when nothing applies (native mode, another
    backend, or no deployment profile) so those run records stay unchanged."""

    profile = local_profile(config, task_spec)
    if profile is None:
        return config, exemption(config, task_spec)
    changes: dict[str, Any] = {}
    record_roles: dict[str, Any] = {}
    for role in roles:
        size = profile.role(role)
        spec = getattr(task_spec, role)
        current_cpu = getattr(config, role + "_cpu_limit")
        current_memory = getattr(config, role + "_mem_limit")
        record_roles[role] = {
            "applied": size.limits(),
            "replaced": {"cpu": spec.cpu_limit if current_cpu is None else current_cpu,
                         "memory": spec.mem_limit if current_memory is None else current_memory},
            "replaced_origin": "task" if current_cpu is None and current_memory is None else "config",
            "gpu": size.gpu,
        }
        changes[role + "_cpu_limit"] = size.cpu
        changes[role + "_mem_limit"] = size.memory
    record: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "mode": "pyromind",
                              "status": "applied", "task_id": task_spec.task_id,
                              **profile.identity(), "roles": record_roles}
    undersize = undersize_decision(config, task_spec, profile, roles)
    if undersize is not None:
        record["cpuset_undersize"] = undersize
        changes["cpuset_undersize"] = undersize["value"]
    return replace(config, **changes), record


def review_limits(config, task_spec) -> tuple[float, str, dict[str, Any]] | None:
    """The local review container's (cpu, memory, record), or None to keep defaults."""

    profile = local_profile(config, task_spec)
    if profile is None:
        return None
    size = profile.review
    return float(size.cpu), size.memory, {
        "schema_version": SCHEMA_VERSION, "mode": "pyromind", "status": "applied",
        "task_id": task_spec.task_id, **profile.identity(), "review": size.limits()}


def review_sandbox_limits(config):
    """``AgentReviewEngine(sandbox_limits=...)``: each task's review size."""

    def resolve(task_spec):
        sized = review_limits(config, task_spec)
        return None if sized is None else (sized[0], sized[1])

    return resolve


def judge_overlay(config, task_spec, cpu_limit, mem_limit):
    """A Judge-server registration's limits with the task's profile applied.

    Returns ``(cpu_limit, mem_limit, cpuset_undersize, record)``; the
    registered values pass through unchanged when no profile applies."""

    sized, record = apply_local_profile(
        replace(config, judge_cpu_limit=cpu_limit, judge_mem_limit=mem_limit),
        task_spec, ("judge",))
    if not record or record.get("status") != "applied":
        return cpu_limit, mem_limit, None, record
    undersize = record.get("cpuset_undersize", {}).get("value")
    return sized.judge_cpu_limit, sized.judge_mem_limit, undersize, record


def summary(record: Mapping[str, Any] | None) -> str | None:
    """One line for the console, e.g. ``work 8 CPU/32g -> 8/16g``."""

    if not record:
        return None
    if record.get("status") == "exempt":
        return f"resource profile: {record['deployment']} exempt ({record['reason']})"
    parts = []
    for role, item in record.get("roles", {}).items():
        old, new = item["replaced"], item["applied"]
        parts.append(f"{role} {old['cpu']} CPU/{old['memory']} -> {new['cpu']}/{new['memory']}")
    pinning = record.get("cpuset_undersize")
    if pinning:
        parts.append(f"cpuset undersize={pinning['value']}")
    return f"resource profile {record['deployment']} ({record['source']}): " + "; ".join(parts)


if __name__ == "__main__":
    import sys

    if sys.argv[1:] != ["--write-table"]:
        sys.exit("usage: python -m sforge.harness.resource_profile --write-table")
    (REPO / TABLE).write_text(json.dumps(table(), indent=2) + "\n")
    print(f"wrote {REPO / TABLE}")
