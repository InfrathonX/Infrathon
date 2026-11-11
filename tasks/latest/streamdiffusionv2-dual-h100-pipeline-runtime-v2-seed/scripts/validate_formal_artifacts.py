#!/usr/bin/env python3
"""Fail closed unless a structured result has the complete SDV2 gate set."""
from __future__ import annotations
import argparse, json
from pathlib import Path
# v2 contract: twelve gates (v1's eleven plus construction_ok).
GATES={
    "profile_ok", "hashes_ok", "admission_ok", "seed_reference_ok", "worker_ok",
    "completeness_ok", "parity_ok", "liveness_ok", "resources_ok",
    "ttff_ok", "p99_ok", "construction_ok",
}
def main():
    p=argparse.ArgumentParser(); p.add_argument("result",type=Path); a=p.parse_args(); value=json.loads(a.result.read_text())
    if not isinstance(value,dict) or set(value.get("gates",{}))!=GATES: raise SystemExit("formal result gate contract mismatch")
    if value.get("valid") is True and (not all(value["gates"].values()) or float(value.get("score",0))<=0): raise SystemExit("invalid valid-result claim")
    print("formal artifact validation passed"); return 0
if __name__=="__main__": raise SystemExit(main())
