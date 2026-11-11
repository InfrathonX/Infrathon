#!/usr/bin/env python3
"""Authorized host smoke: the eleven disclosed edge fixtures plus three short cases.

Runs a candidate through the trusted runtime on one GPU, outside Docker, so the build can
be checked before an image is spent.  Defaults to the Oracle seed; `--candidate` points it
at any other admitted source (the host-only positive control, for instance).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Short self-attention, D = 64 and cross attention: a host check of the trusted loop, not a
# sample of the hidden suite.
SMOKE_CASES = (
    {"case_id": "smoke-wan", "family": "wan13b", "tier": "recipe", "h": 3, "d": 128,
     "s_q": 7800, "s_kv": 7800, "seed": 17011, "gated": True, "layer": None},
    {"case_id": "smoke-d64", "family": "kandinsky5", "tier": "recipe", "h": 8, "d": 64,
     "s_q": 10752, "s_kv": 10752, "seed": 27017, "gated": True, "layer": None},
    {"case_id": "smoke-cross", "family": "kandinsky5", "tier": "sweep", "h": 8, "d": 64,
     "s_q": 10752, "s_kv": 512, "seed": 37021, "gated": True, "layer": None},
)


def load_runtime():
    """Import the trusted runtime from the task tree under a private module name."""
    path = ROOT / "runtime/attn_qat_runtime.py"
    spec = importlib.util.spec_from_file_location("attn_qat_host_smoke_runtime", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def annotate_headroom(result, runtime):
    """Add each gated statistic's fraction of the limit it had to clear."""
    worst = 0.0
    for group in (result["edge_cases"], result["cases"]):
        for row in group:
            for name in runtime.GATE_TENSORS:
                limit = float(row[f"{name}_rel_l2_limit"])
                value = float(row[f"{name}_rel_l2"])
                fraction = value / limit if limit > 0 else 0.0
                row[f"{name}_error_over_limit"] = fraction
                worst = max(worst, fraction)
    result["worst_error_over_limit"] = worst


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path)
    p.add_argument("--candidate", type=Path,
                   default=ROOT / "oracle/candidates/kernel_v001/submission.py")
    p.add_argument("--trials", type=int, default=2)
    p.add_argument("--warmups", type=int, default=1)
    a = p.parse_args()
    runtime = load_runtime()
    os.environ["ATTN_QAT_FIXTURE_DIR"] = str(ROOT / "fixtures")
    cases = [dict(case) for case in SMOKE_CASES]
    started = time.perf_counter()
    result = runtime.run_suite(
        a.candidate, cases, trials=a.trials, warmups=a.warmups, require_hardware=True
    )
    result["wall_seconds"] = time.perf_counter() - started
    result["candidate_path"] = str(a.candidate)
    annotate_headroom(result, runtime)
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if a.output:
        a.output.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
