#!/usr/bin/env python3
"""Measure an implementation with the dynamic-grouped FP8 v3 Judge's own runtime.

Host-only harness (never part of the Work or Judge image).  It imports the
*trusted* task runtime (`grouped_fp8_runtime.py`, identical to the one baked into
the Judge image at /opt/standalone_grouped_fp8_worker) and the frozen
final-holdout-v3 / edge-holdout-v3 manifests, so

  --mode suite      == the Judge's worker step (`candidate_worker.py --mode suite`):
                       edge cases first (optional), then the 32 scored cases with
                       1 warm-up, decoy pre-call, 5 fresh-tensor trials, guard
                       bands, FP64 + calc_diff checks, sentinel, plausibility bound,
                       time = max(host wall, CUDA event).  Directly comparable to a
                       Judge score.  The stages it does NOT apply: source admission
                       (which rejects `import deep_gemm` by design), the
                       compute-sanitizer pass and the LLM integrity review.
  --mode hoisted    DeepGEMM only: the SM90 SFA transpose (K-major FP32 ->
                       MN-major TMA-aligned) is done outside the timer; everything
                       else as in suite mode.  Upper bound for "DeepGEMM fed the
                       layout DeepEP produces"; not a task score.
  --mode kernel     Any candidate: GPU time only (CUDA events around `--iters`
                       back-to-back calls on one prepared invocation of each case's
                       first trial draw, after warm-up).  Comparable to upstream's
                       `bench_kineto` kernel numbers; excludes host launch cost.

Output: one JSON file with per-case records and the task's aggregation
(family geomeans, band geomeans, sqrt(masked*psum)).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

RUNTIME_ROOT = Path("/opt/standalone_grouped_fp8_worker")
sys.path.insert(0, str(RUNTIME_ROOT))
import grouped_fp8_runtime as runtime  # noqa: E402  (captures trusted clocks first)

import torch  # noqa: E402


def _load(path: Path, key: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get(key) if isinstance(payload, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{path}: expected a nonempty '{key}' list")
    return [dict(c) for c in cases]


def _aggregate(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return runtime.aggregate_case_results(results)


# --------------------------------------------------------------------------
def run_suite(candidate: Path, cases, edges, trials, warmups) -> dict[str, Any]:
    return runtime.run_suite(
        candidate, cases, edges, trials=trials, warmups=warmups,
        require_hardware=True, stop_on_edge_failure=False,
    )


# --------------------------------------------------------------------------
def _deepgemm_hoisted_entry(prepared):
    import deep_gemm

    sfa = deep_gemm.get_mn_major_tma_aligned_tensor(prepared.a_scale)
    torch.cuda.synchronize()

    def masked(a, b, a_scale, b_scale, m_sizes, hint, out):
        deep_gemm.m_grouped_fp8_gemm_nt_masked((a, sfa), (b, b_scale), out, m_sizes, int(hint))

    def psum(a, b, a_scale, b_scale, ends, hint, out):
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (a, sfa), (b, b_scale), out, ends,
            use_psum_layout=True, expected_m_for_psum_layout=int(hint),
        )

    return {"masked": masked, "psum": psum}


def run_hoisted(cases, trials, warmups) -> dict[str, Any]:
    import deep_gemm  # noqa: F401  (import before the sentinel snapshot)

    sentinel = runtime.HarnessSentinel()
    results = []
    for raw in cases:
        case = runtime.validate_scored_case(raw)
        fam = str(case["family"])
        g, m_max, n, k = (int(case[x]) for x in ("g", "m_max", "n", "k"))
        hint = runtime.expected_m_hint(int(case["expected_m"]))
        seed = int(case["seed"])
        warm_seed = seed ^ runtime.WARMUP_SEED_SALT
        warm = runtime.prepare_invocation(fam, g, m_max, n, k, runtime.draw_m_sizes(case, warm_seed),
                                          hint, warm_seed, references=True, fingerprints=True)
        entry = _deepgemm_hoisted_entry(warm)
        for _ in range(warmups):
            runtime.poison_valid_output(warm)
            runtime.invoke(entry, warm)
        torch.cuda.synchronize()
        chk = runtime.check_output(warm, runtime.CorrectnessFailure)
        runtime.check_memory(warm, "warm-up")
        max_err = chk.max_exact_error
        del warm, entry
        torch.cuda.empty_cache()
        tf, us, host_us, ev_us, msz = [], [], [], [], []
        for trial in range(trials):
            ts = runtime.trial_seed(seed, trial)
            m_sizes = runtime.draw_m_sizes(case, ts)
            decoy = runtime.prepare_invocation(fam, g, m_max, n, k, m_sizes, hint,
                                               ts ^ runtime.DECOY_SEED_SALT, references=False)
            runtime.invoke(_deepgemm_hoisted_entry(decoy), decoy)
            torch.cuda.synchronize()
            del decoy
            prepared = runtime.prepare_invocation(fam, g, m_max, n, k, m_sizes, hint, ts,
                                                  references=True, fingerprints=True)
            entry = _deepgemm_hoisted_entry(prepared)   # transpose outside the timer
            measured, host, ev = runtime._measure_once(entry, prepared)
            sentinel.check("hoisted measured call")
            chk = runtime.check_output(prepared, runtime.CorrectnessFailure)
            runtime.check_memory(prepared, "hoisted measured call")
            max_err = max(max_err, chk.max_exact_error)
            tf.append(runtime.effective_tflops(m_sizes, n, k, measured))
            us.append(measured * 1e6); host_us.append(host * 1e6); ev_us.append(ev * 1e6)
            msz.append(list(m_sizes))
            del prepared, entry
        torch.cuda.empty_cache()
        results.append({
            "case_id": case["case_id"], "family": fam, "band": case["band"], "g": g,
            "expected_m": int(case["expected_m"]), "n": n, "k": k,
            "trial_us": us, "trial_host_us": host_us, "trial_event_us": ev_us,
            "trial_tflops": tf, "trial_m_sizes": msz,
            "median_us": statistics.median(us), "tflops": statistics.median(tf),
            "max_exact_error": max_err,
        })
    out = {"mode": "hoisted", "cases": results}
    out.update(_aggregate(results))
    return out


# --------------------------------------------------------------------------
def run_kernel(candidate: Path | None, cases, iters: int, hoisted_deepgemm: bool) -> dict[str, Any]:
    if candidate is not None:
        _module, entrypoints, sha = runtime.load_candidate(candidate)
    results = []
    for raw in cases:
        case = runtime.validate_scored_case(raw)
        fam = str(case["family"])
        g, m_max, n, k = (int(case[x]) for x in ("g", "m_max", "n", "k"))
        hint = runtime.expected_m_hint(int(case["expected_m"]))
        ts = runtime.trial_seed(int(case["seed"]), 0)
        m_sizes = runtime.draw_m_sizes(case, ts)
        prepared = runtime.prepare_invocation(fam, g, m_max, n, k, m_sizes, hint, ts,
                                              references=True, fingerprints=True)
        entry = _deepgemm_hoisted_entry(prepared) if hoisted_deepgemm else entrypoints
        for _ in range(3):
            runtime.invoke(entry, prepared)
        torch.cuda.synchronize()
        chk = runtime.check_output(prepared, runtime.CorrectnessFailure)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            runtime.invoke(entry, prepared)
        end.record()
        torch.cuda.synchronize()
        sec = start.elapsed_time(end) / 1e3 / iters
        results.append({
            "case_id": case["case_id"], "family": fam, "band": case["band"], "g": g,
            "expected_m": int(case["expected_m"]), "n": n, "k": k, "m_sizes": m_sizes,
            "kernel_us": sec * 1e6, "tflops": runtime.effective_tflops(m_sizes, n, k, sec),
            "max_exact_error": chk.max_exact_error,
        })
        del prepared, entry
        torch.cuda.empty_cache()
    out = {"mode": "kernel", "iters": iters, "cases": results}
    out.update(_aggregate(results))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("suite", "hoisted", "kernel"), required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--candidate", type=Path)
    p.add_argument("--final-cases", type=Path, required=True)
    p.add_argument("--edge-cases", type=Path)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--hoisted-deepgemm", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()

    cases = _load(a.final_cases, "cases")
    edges = _load(a.edge_cases, "cases") if a.edge_cases else []
    t0 = time.time()
    if a.mode == "suite":
        res = run_suite(a.candidate.resolve(), cases, edges, a.trials, a.warmups)
    elif a.mode == "hoisted":
        res = run_hoisted(cases, a.trials, a.warmups)
    else:
        res = run_kernel(a.candidate.resolve() if a.candidate else None, cases, a.iters, a.hoisted_deepgemm)
    res["label"] = a.label
    res["wall_seconds"] = time.time() - t0
    a.output.write_text(json.dumps(res, indent=2, default=str) + "\n", encoding="utf-8")
    fam = res.get("family_tflops", {})
    print(json.dumps({
        "label": a.label, "mode": a.mode, "status": res.get("status", "n/a"),
        "failed_gate": res.get("failed_gate"), "private_error": str(res.get("private_error", ""))[:300],
        "edge": {k: v for k, v in (res.get("edge") or {}).items() if k != "results"},
        "score_tflops": res.get("primary_tflops"), "masked": fam.get("masked"), "psum": fam.get("psum"),
        "bands": res.get("band_tflops"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
