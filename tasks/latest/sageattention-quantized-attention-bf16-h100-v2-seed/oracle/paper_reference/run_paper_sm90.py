#!/usr/bin/env python3
"""Score a non-admissible candidate (upstream SM90 CUDA kernel) with the Judge's OWN runtime.

Runs inside the private Judge image (+ compiled SageAttention). It calls the trusted
quant_attention_runtime.run_suite exactly as candidate_worker.py does (same 8 edge cases,
same 62 real layers, same frozen final-holdout-v2 cases, 1 warm-up + 7 fresh-input trials with
decoy, max(host, CUDA event), median, family geomeans, same-card cuDNN BF16 reference).
Only source admission / LLM review / privilege drop are skipped (the candidate imports a
compiled extension, which admission forbids by design).

Usage (inside the container, as root so the 0400 private manifest is readable):
  python -I run_paper_sm90.py --candidate /harness/paper_sm90_candidate.py --output /out/x.json
Optional: --manifest PATH (default: the Judge's private final_cases.json),
          --skip-real (debug only; a scored run must include the real layers).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

RUNTIME_ROOT = Path("/opt/sageattention_quant_attention_worker")
FINAL_MANIFEST = Path("/opt/sageattention_quant_attention_judge/private/final_cases.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=FINAL_MANIFEST)
    ap.add_argument("--trials", type=int, default=7)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--skip-real", action="store_true")
    a = ap.parse_args()

    sys.path.insert(0, str(RUNTIME_ROOT))
    import quant_attention_runtime as rt  # trusted Judge runtime (protected hash in the image)

    manifest = json.loads(a.manifest.read_text())
    assert manifest["schema_version"] == 1 and manifest["suite_id"] == "final-holdout-v2", manifest.get("suite_id")
    cases = manifest["cases"]
    assert len(cases) == 5
    t0 = time.time()
    status = {"ok": False}
    try:
        result = rt.run_suite(
            a.candidate.resolve(), cases,
            trials=a.trials, warmups=a.warmups, require_hardware=True,
            real_layer_cases=() if a.skip_real else rt.REAL_LAYER_CASES,
            measure_bf16_reference=True,
        )
        fam = result["family_tflops"]
        status = {
            "ok": True,
            "valid_equivalent": not a.skip_real,
            "score_tflops": result["primary_tflops"],
            "dit_tflops": fam["dit"],
            "causal_tflops": fam["causal"],
            "cases": {c["case_id"]: {"l": c["l"], "tflops": c["tflops"], "median_us": c["median_us"]} for c in result["cases"]},
            "bf16_reference_tflops": result.get("bf16_reference", {}).get("primary_tflops"),
        }
        worst = []
        for c in list(result["edge_cases"]) + list(result["real_layers"]):
            r = max(c["rel_l1"] / c["rel_l1_limit"], c["worst_head_rel_l1"] / c["worst_head_rel_l1_limit"],
                    c["row_rel_l2_p999"] / c["row_rel_l2_p999_limit"], c["row_rel_l2_max"] / c["row_rel_l2_max_limit"])
            worst.append((round(r, 4), c["case_id"]))
        status["worst_gate_ratios"] = sorted(worst, reverse=True)[:8]
        status["full_result"] = result
    except BaseException as exc:  # a gate failure is a result, not a crash
        status = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:2000]}
    status["wall_seconds"] = time.time() - t0
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(status, indent=2, sort_keys=True, default=str, allow_nan=True) + "\n")
    print(json.dumps({k: v for k, v in status.items() if k != "full_result"}, indent=2, default=str))
    return 0 if status["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
