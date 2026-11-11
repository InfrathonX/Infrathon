#!/usr/bin/env python3
"""Measure a candidate (incl. the FlashMLA paper wrapper) with the task's trusted runtime.

Host-only harness; never part of a scored evaluation.

--mode suite   : runtime.run_suite over the Judge's 13 scored cases (+ optional edge cases),
                 i.e. the same generator, guard bands, decoy pre-call, 256 MiB L2 flush,
                 max(host wall, CUDA event) timing and median-of-7 the Judge uses.  The only
                 Judge stage skipped is source admission (FlashMLA cannot pass it).
--mode peaked  : distribution-sensitivity probe (NOT a task score).  Same scored shapes and
                 protocol, but q is multiplied by ALPHA after generation, so the softmax
                 becomes peaked like real attention.  Reports TFLOPS and the fraction of
                 (row, head) pairs whose top key carries > 1/16 of the mass or whose max
                 logit leaves [-60, 64] log2 units (the rows the agent's fast pass hands to
                 its exact fallback kernel).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

RUNTIME_ROOT = Path("/opt/oracle")
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

import sparse_mla_prefill_runtime as runtime  # noqa: E402


def _cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [dict(case) for case in payload["cases"]]


def _suite(candidate: Path, scored, edges, trials: int, warmups: int) -> dict[str, Any]:
    result = runtime.run_suite(
        candidate, scored, edges, trials=trials, warmups=warmups,
        require_hardware=True, run_scored_after_edge_failure=True,
    )
    summary: dict[str, Any] = {
        "status": result.get("status"),
        "failed_gate": result.get("failed_gate"),
        "private_error": result.get("private_error"),
        "edge_passed": result.get("edge", {}).get("cases_passed"),
        "edge_failed": result.get("edge", {}).get("cases_failed"),
        "cases": [],
    }
    rows = [case for case in result.get("cases", []) if case.get("scored")]
    for case in rows:
        summary["cases"].append({
            "case_id": case["case_id"], "family": case["family"], "s_kv": case["s_kv"],
            "tflops": case.get("tflops"),
            "median_us": statistics.median(case["trial_us"]),
            "median_host_us": statistics.median(case["trial_host_us"]),
            "median_event_us": statistics.median(case["trial_event_us"]),
        })
    if rows and all(case.get("tflops") for case in rows):
        agg = runtime.aggregate_case_results(rows)
        summary["score_tflops"] = agg["primary_tflops"]
        summary["family_tflops"] = agg["family_tflops"]
    return {"summary": summary, "detail": result}


def _peaked(candidate: Path, scored, alphas, trials: int) -> dict[str, Any]:
    import torch

    runtime.use_exact_fp32()
    _module, entry, sha = runtime.load_candidate(candidate)
    flusher = runtime.L2Flusher()
    out: list[dict[str, Any]] = []
    for raw in scored:
        case = runtime.validate_scored_case(raw)
        spec = runtime.scored_spec(case)
        seed = int(case["seed"])
        for alpha in alphas:
            def prep(role: str, trial: int):
                inv = runtime.prepare_invocation(spec, runtime.derive_seed(seed, role, trial), pristine=False)
                inv.q.mul_(float(alpha))
                return inv

            warm = prep("warmup", 0)
            runtime.invoke(entry, warm)
            torch.cuda.synchronize()
            del warm
            times: list[float] = []
            flagged = window = blocks = 0.0
            correct = None
            for trial in range(trials):
                decoy = prep("decoy", trial)
                runtime.invoke(entry, decoy)
                torch.cuda.synchronize()
                del decoy
                inv = prep("measured", trial)
                flusher.flush()
                seconds, _host, _event = runtime._measure_once(entry, inv)
                times.append(seconds)
                ml, lse = inv.max_logits.float(), inv.lse.float()
                finite = torch.isfinite(ml) & torch.isfinite(lse)
                top_mass = torch.exp(ml - lse)
                flagged = float(((top_mass > 1.0 / 16.0) & finite).float().mean().item())
                log2_ml = ml * (1.0 / math.log(2.0))
                outside = ((log2_ml > 64.0) | (log2_ml < -60.0)) & finite
                window = float(outside.float().mean().item())
                # [audit] the agent's _smla_exact recomputes a whole (row, 64-head) block when ANY
                # of its 64 heads is flagged, so the block fraction is what predicts its slowdown.
                redo = ((top_mass > 1.0 / 16.0) & finite) | outside
                blocks = float(redo.reshape(redo.shape[0], -1, 64).any(dim=-1).float().mean().item())
                if trial == 0:
                    try:
                        runtime.check_outputs(inv, runtime.CorrectnessFailure)
                        correct = True
                    except Exception as exc:  # noqa: BLE001
                        correct = f"{type(exc).__name__}: {str(exc)[:160]}"
                del inv
                torch.cuda.empty_cache()
            med = statistics.median(times)
            positions = int(case["s_q"]) * int(case["topk"])
            row = {
                "case_id": case["case_id"], "family": case["family"], "s_kv": case["s_kv"],
                "alpha": alpha, "median_us": med * 1e6,
                "tflops": runtime.effective_tflops(positions, int(case["h_q"]), int(case["d_qk"]), med),
                "rows_top_key_gt_1_16": flagged, "rows_outside_window": window,
                "head_blocks_redone_by_agent_rule": blocks,
                "correct_trial0": correct,
            }
            print(json.dumps(row), flush=True)
            out.append(row)
    return {"candidate_sha256": sha, "rows": out}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--scored", type=Path, required=True)
    parser.add_argument("--edges", type=Path)
    parser.add_argument("--mode", choices=("suite", "peaked"), default="suite")
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--alphas", default="1,4,8,16,24")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    scored = _cases(args.scored)
    if args.mode == "suite":
        edges = _cases(args.edges) if args.edges else []
        result = _suite(args.candidate, scored, edges, args.trials, args.warmups)
        print(json.dumps({"label": args.label, **result["summary"]}, indent=2, default=str))
    else:
        alphas = [float(item) for item in args.alphas.split(",") if item]
        result = _peaked(args.candidate, scored, alphas, args.trials)
    args.output.write_text(json.dumps({"label": args.label, **result}, indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
