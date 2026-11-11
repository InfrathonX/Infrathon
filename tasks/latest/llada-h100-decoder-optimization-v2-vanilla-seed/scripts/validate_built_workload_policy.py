#!/usr/bin/env python3
"""Validate tracked and Judge-embedded workload-policy bytes."""

from __future__ import annotations

import argparse
from pathlib import Path

from built_image_policy import (
    ensure_docker_group,
    validate_judge_policy_bytes,
    validate_work_policy_absent,
)

from h100_policy_contract import (
    canonical_hidden_sha256,
    object_at,
    sha256,
    validate_workload_policy,
    private_data_dir,
)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, default=root / "workload_policy.json")
    parser.add_argument(
        "--judge-copy",
        type=Path,
        default=private_data_dir(root) / "workload_policy.json",
    )
    parser.add_argument("--judge-image", required=True)
    parser.add_argument("--work-image", required=True)
    parser.add_argument("--container-engine", default="docker")
    parser.add_argument("--require-frozen", action="store_true")
    args = parser.parse_args()
    # Both checks inspect the built images over the Docker socket; re-exec
    # under the docker group first, exactly as scripts/lib.sh does.
    ensure_docker_group(args.container_engine)
    policy = object_at(args.policy, "tracked workload policy")
    validate_workload_policy(policy, require_frozen=args.require_frozen)
    expected = args.policy.read_bytes()
    for label, path in (("Judge", args.judge_copy),):
        if path.is_symlink() or not path.is_file() or path.read_bytes() != expected:
            raise SystemExit(f"{label} workload-policy bytes differ from tracked policy")
    manifest = policy["manifest"]
    dataset = root / "docs/dataset_manifest.json"
    if canonical_hidden_sha256(root) != manifest["hidden_cases_sha256"]:
        raise SystemExit("workload policy hidden manifest hash mismatch")
    if sha256(dataset) != manifest["dataset_manifest_sha256"]:
        raise SystemExit("workload policy dataset manifest hash mismatch")
    try:
        validate_judge_policy_bytes(
            engine=args.container_engine,
            image=args.judge_image,
            container_path="/opt/decoder_judge/private_data/workload_policy.json",
            expected_sha256=sha256(args.policy),
        )
        validate_work_policy_absent(
            engine=args.container_engine,
            image=args.work_image,
            policy_filename="workload_policy.json",
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(sha256(args.policy))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
