#!/usr/bin/env python3
"""Unprivileged Phase-A worker; validated scores are its only output channel.

It receives context-only rows (no questions, no answers) for one shard and
runs on the GPU named by SFORGE_JUDGE_DEVICE (the runtime rejects use of any other GPU).
"""
from __future__ import annotations
import argparse, json, sys, traceback
from pathlib import Path

sys.path.insert(0, "/opt/kvzip_kv_scoring_worker")
from kv_scoring_runtime import TrustedClock, TrustedOps, run_phase_a, validate_context_item

def main() -> int:
    import torch
    p = argparse.ArgumentParser(); p.add_argument("--candidate", type=Path, required=True); p.add_argument("--assets", type=Path); p.add_argument("--out-dir", type=Path, required=True); a = p.parse_args()
    try:
        request = json.load(sys.stdin); items = [validate_context_item(x) for x in request["items"]]
        result = run_phase_a(a.candidate, a.assets, items, a.out_dir, clock=TrustedClock.capture(torch), ops=TrustedOps.capture(torch))
        print(json.dumps({"candidate_sha256": result["candidate_sha256"], "phase_a_seconds": result["phase_a_seconds"], "rows": len(result["rows"])}, sort_keys=True)); return 0
    except Exception as exc:
        print(f"PHASE_A_FAILURE: {type(exc).__name__}: {exc}", file=sys.stderr); traceback.print_exc(); return 1

if __name__ == "__main__": raise SystemExit(main())
