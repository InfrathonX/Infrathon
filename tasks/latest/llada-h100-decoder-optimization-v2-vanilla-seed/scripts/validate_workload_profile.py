#!/usr/bin/env python3
"""Validate the exact public-fixture/Judge-only stratified-520 contract."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from h100_policy_contract import (
    CATEGORY_COUNTS,
    canonical_hidden_bytes,
    canonical_hidden_sha256,
    GENERATION_COUNT,
    PROFILE,
    REQUESTED_SLOTS,
    ROW_COUNTS,
    object_at,
    sha256,
    validate_workload_policy,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    root = args.task_root.resolve()
    policy = object_at(root / "workload_policy.json", "workload policy")
    validate_workload_policy(policy)
    manifest = object_at(root / "docs/dataset_manifest.json", "dataset manifest")
    hidden = json.loads(canonical_hidden_bytes(root))
    public = json.loads(
        (root / "images/work/public_tests/public_cases.json").read_text(encoding="utf-8")
    )
    if (
        manifest.get("schema_version") != 5
        or manifest.get("workload_profile") != PROFILE
        or manifest.get("generation_count") != GENERATION_COUNT
        or manifest.get("requested_slots") != REQUESTED_SLOTS
        or manifest.get("hidden_row_counts") != ROW_COUNTS
        or not isinstance(hidden, list)
        or len(hidden) != GENERATION_COUNT
        or sum(int(case["gen_length"]) for case in hidden) != REQUESTED_SLOTS
        or Counter(str(case["row_id"]) for case in hidden) != Counter(ROW_COUNTS)
    ):
        raise SystemExit("stratified-520 workload shape is invalid")
    rows: dict[str, set[str]] = defaultdict(set)
    dataset_records: dict[str, set[str]] = defaultdict(set)
    for case in hidden:
        row_id = str(case["row_id"])
        benchmark = str(case["benchmark"])
        record_id = str(case["record_id"])
        if record_id in rows[row_id]:
            raise SystemExit(f"row {row_id} repeats a record")
        if record_id in dataset_records[benchmark]:
            raise SystemExit(f"{benchmark} target repeats across score rows")
        rows[row_id].add(record_id)
        dataset_records[benchmark].add(record_id)
    if {key: len(value) for key, value in dataset_records.items()} != CATEGORY_COUNTS:
        raise SystemExit("hidden workload must have 130 unique targets per dataset")
    hidden_targets = {(case["benchmark"], case["record_id"]) for case in hidden}
    if any(case.get("source_split") == "test" for case in public):
        raise SystemExit("public fixtures consume an official test record")
    if hidden_targets & {(case["benchmark"], case["record_id"]) for case in public}:
        raise SystemExit("public and Judge target records overlap")
    excluded = {
        (entry["benchmark"], entry["record_id"])
        for entries in manifest["context_validation"]["excluded_candidates"].values()
        for entry in entries
    }
    if excluded != {("mbpp", "test:482/task:493")}:
        raise SystemExit("context-exclusion contract differs")
    if (
        canonical_hidden_sha256(root)
        != policy["manifest"]["hidden_cases_sha256"]
        or sha256(root / "docs/dataset_manifest.json")
        != policy["manifest"]["dataset_manifest_sha256"]
    ):
        raise SystemExit("workload policy hashes differ from generated data")
    print("PASS 520 generations / 216,320 requested slots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
