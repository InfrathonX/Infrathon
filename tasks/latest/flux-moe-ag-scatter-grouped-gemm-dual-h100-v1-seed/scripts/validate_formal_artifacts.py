#!/usr/bin/env python3
"""Validate build/evaluation artifacts without exposing private cases."""
from __future__ import annotations
import argparse,hashlib,json,stat
from pathlib import Path
def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--task-root",type=Path,default=Path(__file__).resolve().parents[1]); p.add_argument("--logs-dir",type=Path,default=None); a=p.parse_args(argv)
    logs=(a.logs_dir or a.task_root/"logs").expanduser()
    manifest=logs/"build/image-manifest.json"
    if not manifest.is_file():raise SystemExit(f"build manifest missing: {manifest} (pass --logs-dir for the LOGS_DIR the build used)")
    value=json.loads(manifest.read_text()); final=value.get("final_cases",{})
    recorded=str(final.get("sha256",""))
    if final.get("suite_id")!="final-holdout-v2" or final.get("case_count")!=986 or len(recorded)!=64:raise SystemExit("final manifest record differs")
    # Since 2026-09-13 the holdout is the frozen in-tree instance, which ships with
    # the task at whatever mode the checkout gives it; the 0600 guarantee belongs to
    # the staged build-context copy the Judge image was actually built from.
    staged=logs/"build/final-cases-context/final_cases.json"
    if not staged.is_file() or staged.is_symlink():raise SystemExit(f"staged final manifest missing or unsafe: {staged}")
    if stat.S_IMODE(staged.stat().st_mode)!=0o600:raise SystemExit("staged final manifest is not mode 0600")
    if hashlib.sha256(staged.read_bytes()).hexdigest()!=recorded:raise SystemExit("staged final manifest does not match the recorded sha256")
    print("formal artifacts: PASS"); return 0
if __name__=="__main__":raise SystemExit(main())
