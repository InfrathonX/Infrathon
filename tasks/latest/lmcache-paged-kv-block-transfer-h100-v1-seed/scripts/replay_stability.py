#!/usr/bin/env python3
"""Replay the public suite N times and report cross-replay stability.

Qualification instrument: answers "is the timing repeatable enough to score
with?" on one GPU, for one compiled library.  Reports per-case and aggregate
coefficients of variation across whole replays (not across trials inside one
replay — that is what the median already absorbs).

Usage:
  CUDA_VISIBLE_DEVICES=6 python3 scripts/replay_stability.py \
      --library build/tmp/seed.so [--replays 3] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TASK_ROOT / "runtime"))

import kv_transfer_runtime as runtime  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--replays", type=int, default=3)
    parser.add_argument("--json", type=Path)
    # Thresholds from the 2026-08-24 qualification measurement on GPU 6:
    # score CV 0.284%, every case under 1.5% except one host single-object
    # ct1024 store at 3.7% (a first-replay-in-process pinned/PCIe first-touch
    # effect: 45.0 then 47.8, 48.2 GB/s).  Scoring always compares fresh
    # processes running the same case order, so that bias is common-mode.
    parser.add_argument(
        "--max-case-cv", type=float, default=5.0,
        help="fail if any case's cross-replay CV exceeds this percentage",
    )
    parser.add_argument(
        "--max-score-cv", type=float, default=1.5,
        help="fail if the aggregate score's cross-replay CV exceeds this",
    )
    args = parser.parse_args(argv)
    if args.replays < 2:
        parser.error("--replays must be at least 2")

    import torch

    replays = []
    for index in range(args.replays):
        result = runtime.run_suite(
            torch, args.library, runtime.public_suite(),
            suite_id=runtime.PUBLIC_SUITE_ID,
        )
        replays.append(result)
        groups = {key: round(value, 1) for key, value in result["group_gbps"].items()}
        print(f"replay {index}: score={result['score_gbps']:.2f} groups={groups}",
              flush=True)

    print("\nper-case cross-replay stability")
    worst_case_cv = 0.0
    rows = []
    for position, case in enumerate(replays[0]["cases"]):
        values = [replay["cases"][position]["gbps"] for replay in replays]
        cv = statistics.stdev(values) / statistics.mean(values) * 100.0
        worst_case_cv = max(worst_case_cv, cv)
        rows.append({"case_id": case["case_id"], "gbps": values, "cv_percent": cv})
        print(f"  {case['case_id']:60s} "
              f"{[round(value, 1) for value in values]} CV={cv:.3f}%")

    scores = [replay["score_gbps"] for replay in replays]
    score_cv = statistics.stdev(scores) / statistics.mean(scores) * 100.0
    print(f"\nscore CV {score_cv:.3f}%   worst case CV {worst_case_cv:.3f}%")

    payload = {
        "library_sha256": runtime.sha256_file(args.library),
        "replays": args.replays,
        "scores": scores,
        "score_cv_percent": score_cv,
        "worst_case_cv_percent": worst_case_cv,
        "group_gbps": [replay["group_gbps"] for replay in replays],
        "cases": rows,
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    failures = []
    if score_cv > args.max_score_cv:
        failures.append(f"score CV {score_cv:.3f}% exceeds {args.max_score_cv}%")
    if worst_case_cv > args.max_case_cv:
        failures.append(
            f"worst case CV {worst_case_cv:.3f}% exceeds {args.max_case_cv}%"
        )
    for failure in failures:
        print(f"FAIL: {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
