#!/usr/bin/env python3
"""Describe the selected runtime source for immutable final replay."""
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path
def main():
    p=argparse.ArgumentParser(); p.add_argument("workspace",type=Path); p.add_argument("output",type=Path); a=p.parse_args()
    text=(a.workspace/"FINAL_VERSION").read_text().strip()
    if not text.isdigit() or not 1<=int(text)<=999: raise SystemExit("invalid FINAL_VERSION")
    candidate=a.workspace/"candidates"/f"runtime_v{int(text):03d}"
    if not candidate.is_dir() or candidate.is_symlink(): raise SystemExit("selected candidate missing")
    files=[]
    for path in sorted(candidate.rglob("*")):
        if path.is_file() and not path.is_symlink(): files.append({"path":path.relative_to(a.workspace).as_posix(),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"bytes":path.stat().st_size})
    payload={"schema_version":1,"task_id":"streamdiffusionv2_dual_h100_pipeline_runtime_seed_v2","final_version":int(text),"files":files}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    return 0
if __name__=="__main__": raise SystemExit(main())
