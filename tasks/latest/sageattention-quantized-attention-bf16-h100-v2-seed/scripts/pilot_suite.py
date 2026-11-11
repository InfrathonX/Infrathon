#!/usr/bin/env python3
"""Host-side pilot: run one candidate over a suite manifest through the trusted runtime.

Calibration tool, never part of an evaluation.  Gate violations are recorded instead of
aborting the run (``--strict`` restores the Judge behavior), so one pass yields every
case's time, TFLOPS and error.  Run inside the Judge image with the task mounted at /task:

  python /task/scripts/pilot_suite.py CANDIDATE.py MANIFEST.json OUT.json [--only real,llm] [--strict]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
import quant_attention_runtime as rt  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("candidate")
    p.add_argument("manifest")
    p.add_argument("output")
    p.add_argument("--only", default="", help="comma-separated families to run")
    p.add_argument("--limit", type=int, default=0, help="run at most this many cases")
    p.add_argument("--exclude-model", default="", help="comma-separated models (real sources) to skip")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--bf16-reference", action="store_true")
    a = p.parse_args()
    cases = json.load(open(a.manifest))["cases"]
    if a.only:
        keep = set(a.only.split(","))
        cases = [c for c in cases if c["family"] in keep]
    if a.exclude_model:
        drop = set(a.exclude_model.split(","))
        cases = [c for c in cases if c["model"] not in drop]
    if a.limit:
        cases = cases[: a.limit]
    violations: list[dict] = []
    if not a.strict:
        strict_check = rt.check_metric_bundle

        def lenient(candidate, reference, p999_mult=rt.P999_MULT):
            try:
                return strict_check(candidate, reference, p999_mult)
            except RuntimeError as exc:
                violations.append({"error": str(exc), "candidate": dict(candidate), "reference": dict(reference)})
                return rt.realised_limits(reference, p999_mult)

        rt.check_metric_bundle = lenient
    import torch

    started = time.time()
    result = rt.run_suite(rt.Path(a.candidate), cases, trials=3, warmups=1,
                          measure_bf16_reference=a.bf16_reference,
                          progress=lambda line: print(f"[{time.time() - started:7.1f}s] {line}", flush=True))
    wall = time.time() - started
    summary = {"wall_s": wall, "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
               "primary_tflops": result["primary_tflops"], "family_tflops": result["family_tflops"],
               "violations": len(violations)}
    json.dump({**summary, "violation_records": violations, "result": result}, open(a.output, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
