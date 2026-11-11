#!/usr/bin/env python3
"""Validate tracked and embedded H100 memory-policy bytes."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from built_image_policy import (
    ensure_docker_group,
    validate_launcher_memory_contract,
    validate_judge_policy_bytes,
    validate_work_policy_absent,
)
from h100_policy_contract import (
    object_at,
    public_memory_contract,
    sha256,
    validate_memory_policy,
    private_data_dir,
)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, default=root / "memory_policy.json")
    parser.add_argument(
        "--judge-copy",
        type=Path,
        default=private_data_dir(root) / "memory_policy.json",
    )
    parser.add_argument("--judge-image", required=True)
    parser.add_argument("--work-image", required=True)
    parser.add_argument("--container-engine", default="docker")
    parser.add_argument("--launcher-lib", type=Path, default=root / "scripts/lib.sh")
    parser.add_argument("--require-frozen", action="store_true")
    args = parser.parse_args()
    # Both checks inspect the built images over the Docker socket; re-exec
    # under the docker group first, exactly as scripts/lib.sh does.
    ensure_docker_group(args.container_engine)
    policy = object_at(args.policy, "tracked memory policy")
    validate_memory_policy(policy, require_frozen=args.require_frozen)
    expected = args.policy.read_bytes()
    for label, path in (("Judge", args.judge_copy),):
        if path.is_symlink() or not path.is_file() or path.read_bytes() != expected:
            raise SystemExit(f"{label} memory-policy bytes differ from tracked policy")
    public_contract_path = root / "images/work/public_tests/memory_contract.json"
    public_contract = object_at(public_contract_path, "public memory contract")
    if public_contract != public_memory_contract(policy):
        raise SystemExit("public memory contract differs from sanitized tracked policy")
    try:
        validate_judge_policy_bytes(
            engine=args.container_engine,
            image=args.judge_image,
            container_path="/opt/decoder_judge/private_data/memory_policy.json",
            expected_sha256=sha256(args.policy),
        )
        validate_work_policy_absent(
            engine=args.container_engine,
            image=args.work_image,
            policy_filename="memory_policy.json",
        )
        validate_judge_policy_bytes(
            engine=args.container_engine,
            image=args.work_image,
            container_path="/workspace/public_tests/memory_contract.json",
            expected_sha256=sha256(public_contract_path),
            label="Work public memory contract",
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if policy["qualification_status"] == "QUALIFIED":
        # The launcher admission floor is the capability envelope, not the
        # peak observed on the qualification device: the enforced cap is
        # derived from the visible device's own capacity at evaluation time.
        admission_mib = int(policy["min_device_memory_mib"])
        try:
            validate_launcher_memory_contract(
                launcher_lib=args.launcher_lib,
                policy=args.policy,
                expected_cap_gib=float(policy["qualified_max_peak_gib"]),
                expected_headroom_mib=int(policy["headroom_mib"]),
                expected_admission_mib=admission_mib,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"{sha256(args.policy)} admission_mib={admission_mib}")
    else:
        print(f"{sha256(args.policy)} UNQUALIFIED/PENDING")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
