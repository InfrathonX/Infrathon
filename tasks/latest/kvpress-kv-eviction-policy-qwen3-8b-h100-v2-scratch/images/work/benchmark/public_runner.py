#!/usr/bin/env python3
"""Candidate-only disclosed-suite controller."""
from __future__ import annotations
import argparse, json, sys, traceback
from pathlib import Path
from source_admission import validate_submission

sys.path.insert(0, "/workspace/runtime")
from kv_eviction_runtime import RATIOS, run_public_suite, suite_items

def main(argv=None) -> int:
    p = argparse.ArgumentParser(); p.add_argument("--lab-root", type=Path, default=Path("/app/workdir/kv_eviction_lab")); p.add_argument("--suite", choices=("one", "quick", "full"), default="full"); p.add_argument("--candidate-version", type=int, required=True); p.add_argument("--json-output", type=Path); a = p.parse_args(argv)
    try:
        admission = validate_submission(a.lab_root, a.candidate_version, workspace_mode=True)
        if not admission["ok"]: raise RuntimeError("public source admission failed: " + "; ".join(admission["errors"]))
        selected = Path(admission["selected_path"]); asset = selected.with_name("assets.pt"); result = run_public_suite(selected, asset if asset.exists() else None, suite_items(a.suite), RATIOS)
        payload = {**result, "suite": a.suite, "candidate_version": a.candidate_version, "num_candidate_versions": admission["num_candidate_versions"], "candidate_only": True, "final_holdout_run": False, "source_admission_ok": True}
        if a.json_output: a.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print("SOURCE_ADMISSION PASS\nCANDIDATE_ONLY YES\nFINAL_HOLDOUT NOT_RUN")
        for ratio, value in payload["ratio_score"].items(): print(f"RATIO_{ratio} {value:.8f}")
        print(f"PUBLIC_PRIMARY {payload['primary']:.8f}"); print(json.dumps(payload, sort_keys=True, allow_nan=False)); return 0 if all(payload["gates"].values()) else 1
    except Exception as exc:
        print(f"PUBLIC_FAILURE: {type(exc).__name__}: {exc}", file=sys.stderr); traceback.print_exc(); return 1

if __name__ == "__main__": raise SystemExit(main())

