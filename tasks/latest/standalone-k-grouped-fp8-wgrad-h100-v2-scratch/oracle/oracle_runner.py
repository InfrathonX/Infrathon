#!/usr/bin/env python3
"""Measure an oracle implementation with the Judge's own runtime protocol.

Host-only.  This file lives outside the Work and Judge images and never runs inside a
scored evaluation.  It drives the trusted runtime (`runtime/k_grouped_wgrad_runtime.py`,
copied into the oracle image) over the same frozen scored manifest the Judge uses, so a
`--mode suite` number is directly comparable to a Judge score:

  * `--mode suite`   : `runtime.run_suite(candidate, scored_cases, edge_cases, trials=5,
                       warmups=1)` -- exactly what images/judge/candidate_worker.py calls
                       (guard bands, NaN-poisoned out, decoy pre-call on fresh tensors,
                       max(host wall, CUDA event), median of 5, two-family geomean,
                       four correctness layers, byte-exact input checks, sentinel,
                       plausibility bound).  The one Judge stage NOT applied is
                       source_admission (it forbids `import deep_gemm` by design) and the
                       compute-sanitizer pass.  Works for any candidate file: DeepGEMM
                       adapters and the baseline alike.
  * `--mode hoisted` : upper bound on the upstream kernel itself.  Same case/seed/ks
                       draws, same timer, but the K-major repack and the ks D2H copy are
                       done OUTSIDE the timer.  With `--inplace` also C->out is copied
                       outside the timer and DeepGEMM runs with d == c, which is exactly
                       upstream's own benchmark condition (tests/generators.py:446 `d = c`,
                       tests/test_fp8_fp4.py bench_kineto of the GEMM only).  Not a task
                       score: it does not pay the ABI.

Edge cases: pass `--edge-manifest` to run the 160 private edge cases first (with
stop_on_edge_failure=False so the scored grid still runs and every edge failure is
counted).  Run that as a SEPARATE invocation from the timed one.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

RUNTIME_ROOT = Path("/opt/oracle")
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

import k_grouped_wgrad_runtime as runtime  # noqa: E402


def _load(path: Path, key: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get(key) if isinstance(payload, dict) else None
    if not isinstance(cases, list):
        raise ValueError(f"{path}: no '{key}' list")
    return [dict(c) for c in cases]


# ---------------------------------------------------------------- hoisted mode
def _run_hoisted(cases: Sequence[Mapping[str, Any]], trials: int, warmups: int, inplace: bool) -> dict:
    import torch
    import deep_gemm
    from deep_gemm.utils import set_mk_alignment_for_contiguous_layout

    set_mk_alignment_for_contiguous_layout(128)
    recipe = (1, 1, 128)

    def to_k_major(t, ks_cpu, width):  # == upstream tests/generators.py:468-474
        packed = t.new_empty((t.numel(),))
        prefix = 0
        for k in ks_cpu:
            if k:
                packed[prefix * width:(prefix + k) * width] = t[prefix:prefix + k].T.flatten()
            prefix += k
        return packed

    def entry_for(prep):
        ks_cpu = [int(v) for v in prep.ks]
        pa = to_k_major(prep.a, ks_cpu, prep.m)
        pb = to_k_major(prep.b, ks_cpu, prep.n)
        sfa, sfb = prep.a_scale.T, prep.b_scale.T
        cd = None
        if inplace:
            prep.out.copy_(prep.c)
        torch.cuda.synchronize()

        def entry(a, b, a_scale, b_scale, ks, c, hint, out):
            deep_gemm.k_grouped_fp8_gemm_nt_contiguous(
                (pa, sfa), (pb, sfb), out, ks_cpu, ks, out if inplace else c, recipe=recipe)

        return {f: entry for f in runtime.FAMILIES}

    results = []
    for raw in cases:
        case = runtime.validate_scored_case(raw)
        g, m, n = int(case["g"]), int(case["m"]), int(case["n"])
        hint = runtime.expected_k_hint(int(case["expected_k"]))
        seed = int(case["seed"])
        warm = runtime.prepare_invocation(
            g, m, n, runtime.draw_ks(case, runtime.derive_seed(seed, "warmup-ks", 0)), hint,
            runtime.derive_seed(seed, "warmup", 0), references=True)
        for _ in range(warmups):
            runtime.poison_output(warm)
            runtime.invoke(entry_for(warm), warm)
        torch.cuda.synchronize()
        chk = runtime.check_output(warm, runtime.CorrectnessFailure)
        errs = [chk.max_element_ratio, chk.max_exact_error, chk.max_source_error]
        del warm
        torch.cuda.empty_cache()
        t_us, h_us, e_us, tf = [], [], [], []
        for trial in range(trials):
            ks = runtime.draw_ks(case, runtime.derive_seed(seed, "ks", trial))
            decoy = runtime.prepare_invocation(g, m, n, ks, hint, runtime.derive_seed(seed, "decoy", trial),
                                               references=False)
            runtime.invoke(entry_for(decoy), decoy)
            torch.cuda.synchronize()
            del decoy
            prep = runtime.prepare_invocation(g, m, n, ks, hint, runtime.derive_seed(seed, "measured", trial),
                                              references=True)
            eps = entry_for(prep)                       # repack (+ C->out) outside the timer
            measured, host, event = runtime._measure_once(eps, prep)
            runtime.check_memory(prep, "measured call")
            chk = runtime.check_output(prep, runtime.CorrectnessFailure)
            errs = [max(x, y) for x, y in zip(errs, [chk.max_element_ratio, chk.max_exact_error, chk.max_source_error])]
            t_us.append(measured * 1e6); h_us.append(host * 1e6); e_us.append(event * 1e6)
            tf.append(runtime.effective_tflops(ks, m, n, measured))
            del prep, eps
            torch.cuda.empty_cache()
        results.append({
            "case_id": case["case_id"], "family": case["family"], "band": case["band"],
            "g": g, "m": m, "n": n, "expected_k": int(case["expected_k"]),
            "trial_us": t_us, "trial_host_us": h_us, "trial_event_us": e_us, "trial_tflops": tf,
            "median_us": statistics.median(t_us), "tflops": statistics.median(tf),
            "max_element_ratio": errs[0], "max_exact_error": errs[1], "max_source_error": errs[2],
        })
    return {"status": "passed", "measurement": "kernel_only_inplace" if inplace else "repack_hoisted",
            "trials_per_case": trials, "warmups": warmups, **runtime.aggregate_case_results(results),
            "cases": results}


def _summary(label: str, r: Mapping[str, Any]) -> dict[str, Any]:
    fam = r.get("family_tflops", {}) or {}
    edge = r.get("edge", {}) or {}
    return {
        "label": label,
        "status": r.get("status"),
        "failed_gate": r.get("failed_gate"),
        "private_error": (r.get("private_error") or "")[:300],
        "score_tflops": r.get("primary_tflops"),
        "coarse_tflops": fam.get("coarse"),
        "fine_tflops": fam.get("fine"),
        "edge_passed": edge.get("cases_passed"),
        "edge_failed": edge.get("cases_failed"),
        "edge_failed_ids": [x["case_id"] + ": " + x.get("error", "")[:120]
                            for x in edge.get("results", []) if not x.get("ok")][:200],
        "cases": [{k: c.get(k) for k in ("case_id", "band", "g", "m", "n", "tflops", "median_us",
                                         "max_exact_error", "max_source_error")}
                  for c in r.get("cases", [])],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True)
    ap.add_argument("--manifest", type=Path, required=True, help="final_cases.json (scored grid + seeds)")
    ap.add_argument("--edge-manifest", type=Path, help="edge_cases.json; omit for a timing-only run")
    ap.add_argument("--candidate", type=Path)
    ap.add_argument("--mode", choices=("suite", "hoisted"), default="suite")
    ap.add_argument("--inplace", action="store_true")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args(argv)
    cases = _load(a.manifest, "cases")
    if a.mode == "suite":
        if a.candidate is None:
            raise SystemExit("--candidate is required in suite mode")
        edges = _load(a.edge_manifest, "cases") if a.edge_manifest else []
        result = runtime.run_suite(a.candidate, cases, edges, trials=a.trials, warmups=a.warmups,
                                   require_hardware=True, stop_on_edge_failure=False)
    else:
        result = _run_hoisted(cases, a.trials, a.warmups, a.inplace)
    summ = _summary(a.label, result)
    a.output.write_text(json.dumps({"summary": summ, "detail": result}, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(summ, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
