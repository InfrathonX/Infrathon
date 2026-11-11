#!/usr/bin/env python3
"""Freeze the reviewed peak-memory observation and launcher reserve."""

from __future__ import annotations

import argparse
from pathlib import Path

from h100_policy_contract import (
    PROFILE,
    QUALIFICATION_ROW_SENTINEL_GUARD_BYTES,
    ROW_SENTINEL_METRIC,
    ROW_SENTINEL_REPEAT_GUARD_BYTES,
    ROW_SENTINEL_REPEAT_METRIC,
    TASK_ID,
    object_at,
    public_memory_contract,
    sha256,
    validate_memory_policy,
    write_object,
)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, default=root / "memory_policy.json")
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument(
        "--fixture-manifest",
        type=Path,
        default=root / "memory_qualification_fixtures.json",
    )
    parser.add_argument("--approval-id", required=True)
    parser.add_argument("--output", type=Path, default=root / "memory_policy.json")
    args = parser.parse_args()
    policy = object_at(args.policy, "bootstrap memory policy")
    validate_memory_policy(policy)
    if policy["qualification_status"] != "UNQUALIFIED":
        raise SystemExit("only an unqualified memory policy may be frozen")
    qualification = object_at(args.qualification, "memory qualification")
    fixture_manifest = object_at(
        args.fixture_manifest, "memory qualification fixture manifest"
    )
    if (
        qualification.get("schema_version") != 1
        or qualification.get("status") != "PASS"
        or qualification.get("task_id") != TASK_ID
        or qualification.get("profile") != PROFILE
        or not isinstance(qualification.get("hardware"), str)
        or not qualification.get("hardware")
        or qualification.get("memory_policy_sha256") != sha256(args.policy)
        or qualification.get("workload_policy_sha256")
        != sha256(root / "workload_policy.json")
        or qualification.get("quality_controls_sha256")
        != sha256(root / "quality_controls.json")
        or qualification.get("fixture_manifest_sha256")
        != sha256(args.fixture_manifest)
        or fixture_manifest.get("status") != "PRECOMMITTED"
        or fixture_manifest.get("memory_policy_sha256") != sha256(args.policy)
        or fixture_manifest.get("workload_policy_sha256")
        != sha256(root / "workload_policy.json")
        or qualification.get("row_sentinel_repeat_growth_guard_bytes")
        != ROW_SENTINEL_REPEAT_GUARD_BYTES
    ):
        raise SystemExit("memory qualification is incomplete")
    policy.update(
        {
            "qualification_status": "QUALIFIED",
            "release_status": "FROZEN",
            "qualification_hardware": qualification["hardware"],
            "qualified_max_peak_gib": qualification["proposed_max_peak_gib"],
            "headroom_mib": qualification["proposed_headroom_mib"],
            "qualified_clean_idle_free_mib": qualification[
                "qualified_clean_idle_free_mib"
            ],
            "safety_margin_mib": qualification["safety_margin_mib"],
            "qualification_run_sha256": sha256(args.qualification),
            "approval_id": args.approval_id,
            "row_sentinel": {
                "metric": ROW_SENTINEL_METRIC,
                "repeat_metric": ROW_SENTINEL_REPEAT_METRIC,
                "status": "FROZEN",
                # Preserve the non-release observation ceiling for provenance.
                # Formal evaluation below uses max_growth_bytes instead.
                "qualification_guard_bytes": (
                    QUALIFICATION_ROW_SENTINEL_GUARD_BYTES
                ),
                "repeat_growth_guard_bytes": ROW_SENTINEL_REPEAT_GUARD_BYTES,
                "max_growth_bytes": qualification[
                    "proposed_row_sentinel_max_growth_bytes"
                ],
                "observed_max_growth_bytes": qualification[
                    "row_sentinel_observed_max_growth_bytes"
                ],
                "observed_max_repeat_growth_bytes": qualification[
                    "row_sentinel_observed_max_repeat_growth_bytes"
                ],
                "safety_margin_bytes": qualification[
                    "row_sentinel_safety_margin_bytes"
                ],
            },
            "near_cap_stress": {
                "status": "QUALIFIED",
                "minimum_cap_utilization": qualification[
                    "minimum_stress_cap_utilization"
                ],
                "maximum_gap_mib": qualification["max_stress_gap_mib"],
                "observed_cap_utilization": qualification[
                    "observed_stress_cap_utilization"
                ],
            },
        }
    )
    validate_memory_policy(policy, require_frozen=True)
    digest = write_object(args.output, policy)
    if args.output.resolve() == (root / "memory_policy.json").resolve():
        (root / "images/judge/private_data/memory_policy.json").write_bytes(
            args.output.read_bytes()
        )
        write_object(
            root / "images/work/public_tests/memory_contract.json",
            public_memory_contract(policy),
        )
        contract_path = root / "images/work/public_tests/memory_contract.json"
        protected_path = root / "images/judge/protected_hashes.txt"
        runtime_path = "/workspace/public_tests/memory_contract.json"
        lines = protected_path.read_text(encoding="utf-8").splitlines()
        matches = [index for index, line in enumerate(lines) if line.endswith(" " + runtime_path)]
        if len(matches) != 1:
            raise SystemExit("protected memory-contract hash entry is missing or duplicated")
        lines[matches[0]] = f"{sha256(contract_path)} {runtime_path}"
        protected_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
