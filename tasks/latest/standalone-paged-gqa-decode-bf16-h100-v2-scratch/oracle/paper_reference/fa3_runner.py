#!/usr/bin/env python3
"""Time any paged-GQA entrypoint file with the task's trusted runtime.

Host-only.  Drives paged_gqa_decode_runtime.run_suite (the exact module the
Judge image runs, mounted read-only at /opt/harness) over the frozen
final-holdout-v2 scored manifest (and optionally the 160-case edge manifest),
with the Judge's trial counts (1 warm-up, 7 measured trials, decoy pre-call
before every measured call, max(host wall, CUDA event)).  Not applied: source
admission and the compute-sanitizer pass.

  --mode scored : scored grid only (no edge cases) -> score_tflops
  --mode edge   : edge cases only, stop_on_edge_failure=False -> which
                  categories the implementation supports
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/opt/harness")

import paged_gqa_decode_runtime as rt  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--final-cases", required=True, type=Path)
    parser.add_argument("--edge-cases", type=Path)
    parser.add_argument("--mode", choices=("scored", "edge"), default="scored")
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    scored = json.loads(args.final_cases.read_text())["cases"]
    edges = []
    if args.mode == "edge":
        edges = json.loads(args.edge_cases.read_text())["cases"]
        scored = []
    result = rt.run_suite(
        args.candidate, scored, edges,
        trials=args.trials, warmups=args.warmups,
        require_hardware=True, stop_on_edge_failure=False,
    )
    summary = {
        "label": args.label,
        "mode": args.mode,
        "status": result.get("status"),
        "failed_gate": result.get("failed_gate"),
        "private_error": (result.get("private_error") or "")[:600],
        "score_tflops": result.get("primary_tflops"),
        "family_tflops": result.get("family_tflops"),
        "cases": {
            c["case_id"]: {
                "tflops": round(c["tflops"], 3),
                "median_us": round(c["median_us"], 1),
                "host_us": [round(x, 1) for x in c["trial_host_us"]],
                "event_us": [round(x, 1) for x in c["trial_event_us"]],
            }
            for c in result.get("cases", [])
        },
        "edge": {
            "passed": result.get("edge", {}).get("cases_passed"),
            "failed": result.get("edge", {}).get("cases_failed"),
            "failed_by_category": {},
        },
        "hardware": result.get("hardware", {}).get("name"),
    }
    for row in result.get("edge", {}).get("results", []):
        if not row.get("ok", True):
            cat = row.get("category", "?")
            summary["edge"]["failed_by_category"][cat] = summary["edge"]["failed_by_category"].get(cat, 0) + 1
    args.output.write_text(json.dumps({"summary": summary, "result": result}, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
