#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from final_snapshot_contract import SnapshotContractError, validate_snapshot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument("expected_seconds", type=float)
    args = parser.parse_args()
    try:
        result = validate_snapshot(args.snapshot, args.archive, args.expected_seconds)
    except SnapshotContractError as exc:
        raise SystemExit(f"final snapshot contract failed: {exc}") from exc
    print(
        "final snapshot boundary: PASS "
        f"({result['elapsed_seconds']:.3f}s, {result['boundary']}, "
        f"sha256={result['archive']['sha256']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
