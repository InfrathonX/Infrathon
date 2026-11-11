#!/usr/bin/env python3
"""Time one host candidate on one private Judge workload without hard gates.

This is a host-only diagnostic.  It deliberately does not emit or alter an
MLSys Forge score.  The formal Judge remains authoritative for validity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


sys.path.insert(0, "/opt/flash_kmeans_benchmark")
sys.path.insert(0, "/opt/flash_kmeans_judge")

from hidden_cases import HIDDEN_DEPLOYMENT_CASES, HIDDEN_RUNTIME_CASES
from kernel_bench import (
    benchmark_case_to_spec,
    benchmark_stable_latency_ms,
    load_candidate,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--case", required=True)
    args = parser.parse_args()

    cases = {
        case.name: case
        for case in HIDDEN_RUNTIME_CASES + HIDDEN_DEPLOYMENT_CASES
    }
    case = cases[args.case]
    spec = benchmark_case_to_spec(case, seed=91_000)
    _module, assign_fn, update_fn = load_candidate(
        args.candidate,
        version=1,
    )
    measured = benchmark_stable_latency_ms(
        spec,
        assign_fn,
        update_fn,
        # Deployment manifests use warmup=0 because the formal protocol
        # records first-run and warmed latency separately.  This diagnostic
        # compares against candidate_ms (the warmed row), so prime it once.
        warmup=max(1, spec.warmup),
        iters=spec.iters,
        repeats=spec.repeats,
        validate_outputs=False,
        track_memory=True,
    )
    result = {
        "authoritative": False,
        "comparison_kind": "host_candidate_only_no_hard_gates",
        "case": case.name,
        "component": case.component,
        "shape": [case.b, case.n, case.k, case.d],
        "latency_ms": measured["latency_ms"],
        "trial_medians_ms": measured["trial_medians_ms"],
        "peak_bytes": measured["peak_bytes"],
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
