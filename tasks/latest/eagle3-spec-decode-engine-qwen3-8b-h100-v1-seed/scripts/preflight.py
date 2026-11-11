#!/usr/bin/env python3
"""Fail-closed identity, policy, image, and GPU-envelope preflight.

The checks are deliberately usable before Docker exists: image validation is
enabled when ``--image-manifest`` is supplied, while mount and policy checks
always run. GPU checks additionally require ``nvidia-smi`` and reject active
compute owners as well as excessive memory use.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
TASK_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from generate_protected_hashes import private_data_dir  # noqa: E402
from model_artifact_identity import compute  # noqa: E402

# Capability envelope. The Work/Judge split is a protocol convenience, not a
# device requirement: the engine's resident set is the frozen Qwen3-8B target
# plus the EAGLE-3 head and one tree KV cache, about 21 GiB measured on H100
# NVL, so 34 GiB per card is the honest floor.
MIN_COMPUTE_CAPABILITY = (8, 0)
MIN_DEVICE_MEMORY_MIB = 34 * 1024

REQUIRED_TIERS = (
    (0.0, 0.03284789994157488),
    (0.25, 0.001034126163391934),
    (0.5, 0.00031023784901758014),
    (0.75, 0.00020682523267838676),
    (1.0, 0.00020682523267838676),
    (1.5, 0.00015511892450879007),
    (2.0, 0.0001),
    (3.0, 0.0001),
)


def parse_capability(value: str) -> tuple[int, int]:
    """Parse an ``nvidia-smi`` compute capability such as ``9.0`` or ``10.0``."""
    major, _, minor = str(value).strip().partition(".")
    if not major.isdigit() or not minor.isdigit():
        raise ValueError(f"unparsable compute capability: {value!r}")
    return int(major), int(minor)


def command(*arguments: str) -> str:
    """Run a trusted host command and return its stdout."""
    return subprocess.check_output(arguments, text=True, stderr=subprocess.STDOUT)


def require_real_directory(path: Path, label: str) -> Path:
    """Resolve an existing non-symlink directory."""
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be an existing real directory: {path}")
    return path.resolve(strict=True)


def require_deprivileged_reachable(
    path: Path,
    label: str,
    uid: int = 1000,
) -> None:
    """Check that the worker uid can read or traverse a mounted artifact."""
    needed = os.X_OK if path.is_dir() else os.R_OK
    read_fd, write_fd = os.pipe()
    child = os.fork()
    if child == 0:  # pragma: no cover - exercised only on root launch hosts.
        os.close(read_fd)
        try:
            if os.geteuid() == 0:
                os.setgroups([])
                os.setgid(uid)
                os.setuid(uid)
            verdict = os.access(path, needed, effective_ids=True)
            os.write(write_fd, b"1" if verdict else b"0")
        except BaseException:
            os.write(write_fd, b"0")
        finally:
            os.close(write_fd)
            os._exit(0)
    os.close(write_fd)
    verdict = os.read(read_fd, 1)
    os.close(read_fd)
    _, status = os.waitpid(child, 0)
    if verdict != b"1" or status != 0:
        mode = stat.S_IMODE(path.stat().st_mode)
        raise ValueError(f"{label} is not reachable by uid {uid}: mode {mode:04o}")


def validate_policy(path: Path) -> dict[str, Any]:
    """Validate the calibrated tier values without claiming qualification."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"policy is missing or unsafe: {path}")
    policy = json.loads(path.read_text(encoding="utf-8"))
    if policy.get("status") not in {"PENDING_QUALIFICATION", "QUALIFIED"}:
        raise ValueError("policy status is not a recognized lifecycle state")
    if policy.get("attn_implementation") != "eager":
        raise ValueError("oracle attention implementation must remain eager")
    if policy.get("scope") != "global_run":
        raise ValueError("tier scope must be global_run")
    if float(policy.get("eps_hard", -1)) != 24.0:
        raise ValueError("frozen A2 hard cap differs")
    if policy.get("p_soft") != REQUIRED_TIERS[0][1]:
        raise ValueError("frozen A2 soft-rate cap differs")
    tiers = policy.get("tiers")
    if not isinstance(tiers, list):
        raise ValueError("frozen A2 tiers must be a list")
    observed_tiers = tuple(
        (row.get("margin"), row.get("max_rate"))
        for row in tiers
        if isinstance(row, dict)
    )
    if observed_tiers != REQUIRED_TIERS:
        raise ValueError("frozen A2 tier margins or max rates differ")
    required_caps = {
        "build_cap_s",
        "worker_timeout_s",
        "resource_cap_mode",
        "wall_cap_seed_multiple",
        "memory_cap_seed_multiple",
        "memory_cap_headroom_bytes",
        "oracle_rate_seed_multiple",
    }
    if required_caps - set(policy):
        raise ValueError("policy omits runtime caps")
    # No frozen per-device wall or peak may reappear as a gate.
    if {"wall_caps_s", "memory_caps_bytes"} & set(policy):
        raise ValueError("policy carries frozen per-device caps")
    if policy["resource_cap_mode"] != "seed_relative_same_run":
        raise ValueError("runtime caps must be derived from a same-run seed measurement")
    for name, minimum in (
        ("wall_cap_seed_multiple", 1.0),
        ("memory_cap_seed_multiple", 1.0),
        ("oracle_rate_seed_multiple", 1.0),
    ):
        value = policy[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < minimum:
            raise ValueError(f"{name} must be a number >= {minimum}")
    headroom = policy["memory_cap_headroom_bytes"]
    if isinstance(headroom, bool) or not isinstance(headroom, int) or headroom <= 0:
        raise ValueError("memory_cap_headroom_bytes must be a positive integer")
    return {
        "status": policy["status"],
        "scope": policy["scope"],
        "tier_count": len(tiers),
        "eps_hard": policy["eps_hard"],
    }


def validate_protected_hashes(task_root: Path) -> dict[str, Any]:
    """Run the same protected-file binding check used by release tooling."""
    script = task_root / "scripts/generate_protected_hashes.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--check"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(f"protected hash check failed: {completed.stderr.strip()}")
    manifest = task_root / "images/judge/protected_hashes.txt"
    return {"manifest": str(manifest), "record_count": len(manifest.read_text().splitlines())}


def validate_gpus(indices: list[int], max_used_mib: int) -> list[dict[str, Any]]:
    """Resolve UUIDs and require distinct idle devices inside the envelope."""
    if len(indices) != len(set(indices)):
        raise ValueError("Work and Judge GPU indices must be distinct")
    if not indices:
        return []
    raw = command(
        "nvidia-smi",
        "--query-gpu=index,uuid,name,compute_cap,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    )
    rows: dict[int, dict[str, Any]] = {}
    for row in csv.reader(io.StringIO(raw), skipinitialspace=True):
        rows[int(row[0])] = {
            "index": int(row[0]),
            "uuid": row[1],
            "name": row[2],
            "compute_capability": row[3],
            "memory_total_mib": int(row[4]),
            "memory_used_mib": int(row[5]),
            "min_compute_capability": ".".join(str(part) for part in MIN_COMPUTE_CAPABILITY),
            "min_device_memory_mib": MIN_DEVICE_MEMORY_MIB,
        }
    selected = []
    for index in indices:
        gpu = rows.get(index)
        if gpu is None:
            raise ValueError(f"host does not expose GPU {index}")
        # The model name and UUID are provenance. Admission is the envelope.
        if parse_capability(gpu["compute_capability"]) < MIN_COMPUTE_CAPABILITY:
            raise ValueError(
                f"GPU {index} compute capability {gpu['compute_capability']} is below "
                f"the task minimum {MIN_COMPUTE_CAPABILITY}"
            )
        if gpu["memory_total_mib"] < MIN_DEVICE_MEMORY_MIB:
            raise ValueError(
                f"GPU {index} has {gpu['memory_total_mib']} MiB total, below the task "
                f"minimum {MIN_DEVICE_MEMORY_MIB} MiB"
            )
        if gpu["memory_used_mib"] > max_used_mib:
            raise ValueError(f"GPU {index} is busy ({gpu['memory_used_mib']} MiB used)")
        selected.append(gpu)
    apps = command(
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    )
    selected_uuids = {gpu["uuid"] for gpu in selected}
    owners = [
        row
        for row in csv.reader(io.StringIO(apps), skipinitialspace=True)
        if row and row[0] in selected_uuids
    ]
    if owners:
        raise ValueError(f"requested GPU has active compute owners: {owners}")
    return selected


def validate_images(manifest_path: Path) -> dict[str, Any]:
    """Pin recorded immutable IDs to the currently tagged Work/Judge images."""
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("image manifest is missing or unsafe")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed: dict[str, str] = {}
    for role in ("work", "judge"):
        record = manifest.get(role)
        if not isinstance(record, dict):
            raise ValueError(f"image manifest lacks {role}")
        expected = record.get("id")
        name = record.get("name")
        if not isinstance(expected, str) or not expected.startswith("sha256:"):
            raise ValueError(f"invalid immutable {role} image ID")
        actual = command("docker", "image", "inspect", "--format", "{{.Id}}", str(name)).strip()
        if actual != expected:
            raise ValueError(f"{role} image tag moved")
        observed[role] = actual
    return {"manifest": str(manifest_path.resolve()), "ids": observed}


def default_policy_path() -> Path:
    """policy.json in the Judge-only private_data (a dataset asset)."""
    return private_data_dir(TASK_ROOT) / "policy.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse host preflight arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-repo", type=Path, required=True)
    parser.add_argument("--head-dir", type=Path, required=True)
    parser.add_argument("--work-gpu", type=int)
    parser.add_argument("--judge-gpu", type=int)
    parser.add_argument("--max-used-mib", type=int, default=1024)
    parser.add_argument("--image-manifest", type=Path)
    parser.add_argument(
        "--policy",
        type=Path,
        default=None,
        help="Judge-only policy (default: policy.json in the resolved private_data)",
    )
    parser.add_argument("--skip-protected-hashes", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Execute every selected preflight and print one machine-readable record."""
    args = parse_args(argv)
    target_repo = require_real_directory(args.target_repo, "target repository")
    head_dir = require_real_directory(args.head_dir, "draft head")
    require_deprivileged_reachable(target_repo, "target repository")
    require_deprivileged_reachable(head_dir, "draft head")
    indices = [value for value in (args.work_gpu, args.judge_gpu) if value is not None]
    result: dict[str, Any] = {
        "status": "PASS",
        "artifacts": compute(target_repo, head_dir),
        "policy": validate_policy(args.policy or default_policy_path()),
        "gpus": validate_gpus(indices, args.max_used_mib),
    }
    if not args.skip_protected_hashes:
        result["protected_hashes"] = validate_protected_hashes(TASK_ROOT)
    if args.image_manifest is not None:
        result["images"] = validate_images(args.image_manifest)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
