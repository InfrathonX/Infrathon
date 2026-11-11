#!/usr/bin/env python3
"""Host-side check that a pilot result passes the Judge verifier's own result validation.

  python scripts/pilot_verify.py PILOT_RESULT.json MANIFEST.json [--only family,...]

Re-derives the score from the per-case rows exactly as verify_quant_attention.py does, so a
pilot run doubles as a check of the runtime/verifier contract.
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "images/judge"))
import verify_quant_attention as v  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("result")
p.add_argument("manifest")
p.add_argument("--only", default="")
p.add_argument("--exclude-model", default="")
p.add_argument("--limit", type=int, default=0)
a = p.parse_args()
doc = json.load(open(a.result))
result = doc.get("result", doc)
cases = json.load(open(a.manifest))["cases"]
if a.only:
    cases = [c for c in cases if c["family"] in a.only.split(",")]
if a.exclude_model:
    cases = [c for c in cases if c["model"] not in a.exclude_model.split(",")]
if a.limit:
    cases = cases[: a.limit]
partial = {f for f in v.FAMILIES if not any(c["family"] == f for c in cases)}
if partial:  # a partial pilot: score only the families it ran
    v.FAMILIES = tuple(f for f in v.FAMILIES if f not in partial)
score, families = v.validate_runtime_result(result, cases, expected_trials=3, expected_warmups=1)
print(json.dumps({"verified_score": score, "families": families, "cases": len(cases)}, indent=1))
