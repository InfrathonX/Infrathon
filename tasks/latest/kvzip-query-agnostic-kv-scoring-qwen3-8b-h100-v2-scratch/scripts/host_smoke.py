#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))
sys.path.insert(0, str(ROOT / "images/judge"))
from kv_scoring_runtime import RATIOS, load_items_jsonl, run_public_suite, suite_items
from source_admission import validate_submission

def main() -> int:
    p = argparse.ArgumentParser(); source_group = p.add_mutually_exclusive_group(required=True); source_group.add_argument("--suite", choices=("one", "quick", "full")); source_group.add_argument("--items-jsonl", type=Path)
    p.add_argument("--candidate-version", type=int, default=1); p.add_argument("--candidate-path", type=Path); p.add_argument("--assets-path", type=Path); p.add_argument("--json-output", type=Path); a = p.parse_args()
    items = suite_items(a.suite) if a.suite else load_items_jsonl(a.items_jsonl)
    if a.candidate_path is None:
        lab = ROOT / "images/work/workdir/kv_scoring_lab"; admission = validate_submission(lab, a.candidate_version, workspace_mode=True)
        if not admission["ok"]: raise RuntimeError("candidate admission failed: " + "; ".join(admission["errors"]))
        source = Path(admission["selected_path"]); asset = admission.get("selected_assets_path")
    else:
        source = a.candidate_path.resolve()
        if not source.is_file() or source.is_symlink(): raise RuntimeError("candidate path is not a regular file")
        asset = str(a.assets_path.resolve()) if a.assets_path else (str(source.with_name("assets.pt")) if source.with_name("assets.pt").is_file() else None)
    started = time.perf_counter()
    result = run_public_suite(source, asset, items, RATIOS)
    payload = {**result, "suite": a.suite or "items-jsonl", "host_smoke": True, "total_wall_seconds": time.perf_counter() - started}
    if a.json_output: a.json_output.parent.mkdir(parents=True, exist_ok=True); a.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)); return 0 if all(result["gates"].values()) else 1
if __name__ == "__main__": raise SystemExit(main())
