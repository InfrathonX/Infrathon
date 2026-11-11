#!/usr/bin/env python3
"""Opt-in CPU pinning for the Miles task family (docs/CPU_AFFINITY.md, "Miles family").

The policy is SForge's: SFORGE_CPUSET (off | auto | CPULIST[@NODELIST]), the per-role overrides
SFORGE_WORK_CPUSET / SFORGE_JUDGE_CPUSET and the modifiers SFORGE_CPUSET_MEMS, SFORGE_CPUSET_RESERVE and
SFORGE_CPUSET_UNDERSIZE. Unset or off, nothing here changes a container, an environment or a record.

The Miles tools need help in two places:

* their supervisors and qualification tools rebuild the SForge child environment from an allowlist, so
  ``resolve()["env"]`` is what they add to it; the harness then pins every Work and Judge container it
  creates to its own GPUs' slice and writes cpu_affinity_work.json / cpu_affinity.json;
* some tools start GPU containers themselves (``qualify.py public``, ``judge_archive.sh``), and the
  diffusion adapters preset their own cpuset; ``docker_binding()`` gives them the cpuset the harness would
  apply (sforge.harness.cpu_affinity.plan_cpu_binding on the same GPUs and CPU budget).

The OPCD, OPD and ALFWorld containers keep their 24-CPU quota, which is more than a GPU pair's slice on a
host with one L3 group per GPU (16 CPUs on the reference H100 host). SForge refuses such a container
unless SFORGE_CPUSET_UNDERSIZE=allow, so when the operator left that unset and the quota exceeds the pair's
slice, these tools set it to ``allow`` explicitly and record that they did: the quota stays, the threads
are confined to the pair's CPUs. An operator's own value (allow or refuse) is never overridden.

  cpu_pinning.py docker-args --gpu ID --gpu ID [--budget N] [--role work|judge] [--record PATH]
      the `docker run` arguments for a container on those GPUs (prints nothing when pinning is off)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))
PIN_KEYS = ("SFORGE_CPUSET", "SFORGE_WORK_CPUSET", "SFORGE_JUDGE_CPUSET", "SFORGE_CPUSET_MEMS",
            "SFORGE_CPUSET_RESERVE", "SFORGE_CPUSET_UNDERSIZE")
UNDERSIZE = "SFORGE_CPUSET_UNDERSIZE"
ROLES = ("work", "judge")
RECORD_SCHEMA_VERSION = 1


class PinningError(RuntimeError):
    """A policy is set but cannot be applied to these GPUs (the container must not start)."""


def _sforge(harness_root=None):
    root = str(Path(harness_root or REPO).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from sforge.harness import cpu_affinity, host_topology

    return cpu_affinity, host_topology


def policy_values(environ=None) -> dict[str, str]:
    """The non-blank SFORGE_CPUSET* values of ``environ``.

    Without ``environ``: the process environment, and for a key it does not set, the host config file
    (launch/mlsysforge.env or $MLSYSFORGE_CONFIG), per variable, as the other launchers read it."""
    if environ is not None:
        return {k: str(environ[k]).strip() for k in PIN_KEYS if str(environ.get(k) or "").strip()}
    from miles_common.host_config import _parse_env_file, host_config_file

    path = host_config_file()
    host = _parse_env_file(path) if path else {}
    values = {}
    for key in PIN_KEYS:
        value = str(os.environ.get(key) or "").strip() or str(host.get(key) or "").strip()
        if value:
            values[key] = value
    return values


def _plan(ca, policy, uuids, budget):
    # The policy carries the sysfs/procfs roots it was resolved with.
    state = ca.ContainerCpuState(gpu_request_ids=tuple(uuids), nano_cpus=int(budget or 0) * 1_000_000_000)
    return ca.plan_cpu_binding(policy, state)


def resolve(uuids, budget, environ=None, harness_root=None, sysfs_root=None, procfs_root=None) -> dict | None:
    """The pinning of containers on ``uuids`` with a ``budget``-CPU quota (None: no quota).

    Returns None when no role has a policy. Otherwise ``{"env": {...}, "record": {...}}``: the SFORGE_CPUSET*
    keys to pass to a harness child (including an explicit SFORGE_CPUSET_UNDERSIZE=allow when the quota
    exceeds the pair's slice and the operator did not choose), and the plan of each role as SForge will
    compute it. Raises PinningError when a set policy cannot be applied (grammar, topology, a refused
    undersized container), so a supervisor fails before it starts anything.
    """
    values = policy_values(environ)
    ca, ht = _sforge(harness_root)
    extra = {k: v for k, v in (("sysfs_root", sysfs_root), ("procfs_root", procfs_root)) if v is not None}
    uuids = [str(u) for u in uuids]
    try:
        policies = {role: ca.resolve_policy(role, values, **extra) for role in ROLES}
        if not any(policies.values()):
            return None
        undersize_note = None
        if budget and UNDERSIZE not in values:
            slices = {role: len(ht.parse_cpulist(_plan(ca, p, uuids, None).cpuset_cpus or ""))
                      for role, p in policies.items() if p is not None}
            over = {role: n for role, n in slices.items() if budget > n}
            if over:
                values[UNDERSIZE] = "allow"
                undersize_note = (f"set by the Miles tools: the {budget}-CPU quota exceeds the "
                                  f"{min(over.values())}-CPU slice of GPUs {','.join(uuids)}; the quota stays and the "
                                  "threads are confined to the slice")
                policies = {role: ca.resolve_policy(role, values, **extra) for role in ROLES}
        roles = {}
        for role, policy in policies.items():
            if policy is None:
                roles[role] = {"policy_id": None, "outcome": "off"}
                continue
            plan = _plan(ca, policy, uuids, budget)
            roles[role] = {"policy_id": policy.policy_id, "policy": policy.describe(), **plan.record()}
    except (ValueError, ht.TopologyError) as exc:
        raise PinningError(f"CPU pinning policy {values} cannot be applied to GPUs {','.join(uuids)}: {exc}") from exc
    record = {"schema_version": RECORD_SCHEMA_VERSION, "gpu_ids": uuids, "cpu_budget": budget,
              "policy_env": dict(values), "undersize_note": undersize_note, "roles": roles}
    return {"env": dict(values), "record": record}


def docker_binding(uuids, budget, role, environ=None, harness_root=None, **roots) -> tuple[list[str], dict | None]:
    """(`docker run` arguments, record) for a container of ``role`` that a Miles tool starts itself.

    ``([], None)`` when that role is unpinned. The arguments are --cpuset-cpus and, in bind mode,
    --cpuset-mems: the cpuset the harness would apply to a Work/Judge container on the same GPUs."""
    resolved = resolve(uuids, budget, environ=environ, harness_root=harness_root, **roots)
    if resolved is None or resolved["record"]["roles"][role]["policy_id"] is None:
        return [], None
    plan = resolved["record"]["roles"][role]
    args = ["--cpuset-cpus", plan["cpuset_cpus"]]
    if plan.get("cpuset_mems"):
        args += ["--cpuset-mems", plan["cpuset_mems"]]
    record = {**resolved["record"], "role": role, "outcome": plan["outcome"], "cpuset_cpus": plan["cpuset_cpus"],
              "cpuset_mems": plan.get("cpuset_mems"), "policy_id": plan["policy_id"]}
    return args, record


def shared_cpuset(uuids, environ=None, harness_root=None, **roots) -> dict | None:
    """The one cpuset of a task whose adapter pins Work and the Judge itself (SD3.5-M, Wan2.2), or None when off.

    Such a task sizes its CPU quota from its cpuset, so the plan has no budget: the pair's whole slice (auto)
    or the explicit list. Work and the Judge share it, so the roles' policies must agree. Returns
    ``{"cpuset_cpus", "cpuset_mems", "cpu_count", "policy_id", "record"}``."""
    resolved = resolve(uuids, None, environ=environ, harness_root=harness_root, **roots)
    if resolved is None:
        return None
    plans = {role: plan for role, plan in resolved["record"]["roles"].items() if plan["policy_id"] is not None}
    views = {(p["cpuset_cpus"], p.get("cpuset_mems")) for p in plans.values()}
    if len(views) != 1:
        raise PinningError(f"this task gives Work and the Judge one cpuset, but the pinning policy gives them "
                           f"{ {r: p['cpuset_cpus'] for r, p in plans.items()} }: set one policy for both roles")
    plan = plans.get("judge") or plans["work"]
    return {"cpuset_cpus": plan["cpuset_cpus"], "cpuset_mems": plan.get("cpuset_mems"), "cpu_count": plan["cpu_count"],
            "policy_id": plan["policy_id"], "record": resolved["record"]}


def write_record(path, record) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("docker-args")
    p.add_argument("--gpu", action="append", required=True, help="GPU UUID (repeat for each GPU of the container)")
    p.add_argument("--budget", type=int, default=None, help="the container's CPU quota (omit: none)")
    p.add_argument("--role", choices=ROLES, default="judge")
    p.add_argument("--record", default=None, help="write the binding record here (only when pinned)")
    p.add_argument("--harness-root", default=None)
    a = ap.parse_args(argv)
    try:
        args, record = docker_binding(a.gpu, a.budget, a.role, harness_root=a.harness_root)
    except PinningError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 78
    if record is not None and a.record:
        write_record(a.record, record)
    print(" ".join(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
