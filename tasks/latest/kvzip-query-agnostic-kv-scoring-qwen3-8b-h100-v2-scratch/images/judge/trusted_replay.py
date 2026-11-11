#!/usr/bin/env python3
"""Trusted Phase-B process for one shard: never imports candidate code.

Runs as root on the GPU named by SFORGE_JUDGE_DEVICE, decodes the
shard's questions under full and masked caches and writes its rows to a
root-only result file.
"""
from __future__ import annotations
import argparse, json, os, sys, traceback
from pathlib import Path

sys.path.insert(0, "/opt/kvzip_kv_scoring_worker")
from kv_scoring_runtime import RATIOS, TrustedClock, TrustedOps, hardware_inventory, run_phase_b

def main() -> int:
    import torch
    p = argparse.ArgumentParser(); p.add_argument("--request", type=Path, required=True); p.add_argument("--scores", type=Path, required=True); p.add_argument("--out", type=Path, required=True); a = p.parse_args()
    try:
        items = json.loads(a.request.read_text(encoding="utf-8"))["items"]
        hardware = hardware_inventory(torch)
        result = run_phase_b(items, RATIOS, a.scores, clock=TrustedClock.capture(torch), ops=TrustedOps.capture(torch))
        tmp = a.out.with_name(f".{a.out.name}.tmp"); fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle: json.dump({"hardware": hardware, **result}, handle, sort_keys=True, allow_nan=False)
        os.replace(tmp, a.out); return 0
    except Exception as exc:
        print(f"PHASE_B_FAILURE: {type(exc).__name__}: {exc}", file=sys.stderr); traceback.print_exc(); return 1

if __name__ == "__main__": raise SystemExit(main())
