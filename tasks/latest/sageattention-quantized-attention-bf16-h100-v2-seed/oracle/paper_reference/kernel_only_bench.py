#!/usr/bin/env python3
"""Kernel-only timing of upstream's sm90 attention kernel (audit addition; NOT a Judge score).

Upstream's published H100 bars (assets/H100_hd*.png) exclude quantization. The Judge times the
whole call (K smoothing, Q/K/V quantization, V padding, attention, BF16 write) for every
submission, which is the right contract. This script reports both views for the paper kernel
on the 5 frozen holdout shapes, so the write-up can tell how much of the agent's lead comes from
the attention kernel itself and how much from cheaper, fused preprocessing:

  kernel_only : sm90_compile.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf alone, inputs
                pre-quantized outside the timed region
  end_to_end  : paper_sm90_candidate.quantized_attention_bf16 (all of upstream's host flow)

Both are timed with CUDA events: 3 warm-ups, then the median of 20 back-to-back calls on the same
inputs (warm L2; this is upstream's bench style, NOT the Judge's fresh-input/decoy protocol).
Inputs come from the Judge's own generator at the holdout seeds. Run inside the harness image as
root (the private manifest is 0400).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

RUNTIME_ROOT = Path("/opt/sageattention_quant_attention_worker")
FINAL_MANIFEST = Path("/opt/sageattention_quant_attention_judge/private/final_cases.json")


def _time(torch, fn, warm=3, iters=20):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) / 1000.0)
    return statistics.median(ts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=FINAL_MANIFEST)
    ap.add_argument("--qk-gran", choices=("per_thread", "per_warp"), default="per_thread")
    a = ap.parse_args()

    sys.path.insert(0, str(RUNTIME_ROOT))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    import quant_attention_runtime as rt
    import paper_sm90_candidate as cand
    from sageattention import sm90_compile
    from sageattention.quant import per_channel_fp8
    from sageattention.quant import per_warp_int8 as per_warp_int8_cuda
    from sageattention.triton.quant_per_thread import per_thread_int8 as per_thread_int8_triton

    cases = json.loads(a.manifest.read_text())["cases"]
    rows = {}
    for raw in cases:
        c = rt.validate_case(raw)
        t = rt._generate_inputs(torch, c, int(c["seed"]))
        q, k, v = t["q"], t["k"], t["v"]
        out = torch.empty_like(q)
        causal = bool(c["is_causal"])
        scale = float(c["softmax_scale"])

        km = k.mean(dim=1, keepdim=True)
        if a.qk_gran == "per_warp":
            q8, qs, k8, ks = per_warp_int8_cuda(q, k, km, tensor_layout="NHD", BLKQ=64, WARPQ=16, BLKK=128)
            gran = 2
        else:
            q8, qs, k8, ks = per_thread_int8_triton(q, k, km, tensor_layout="NHD", BLKQ=64, WARPQ=16,
                                                    BLKK=128, WARPK=128)
            gran = 3
        vp = v
        pad = (128 - v.size(1) % 128) % 128
        if pad:
            vp = torch.cat([v, torch.zeros(v.size(0), pad, v.size(2), v.size(3), dtype=v.dtype, device=v.device)], dim=1)
        v8, vs, _ = per_channel_fp8(vp, tensor_layout="NHD", smooth_v=False)

        def kern():
            sm90_compile.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(
                q8, k8, v8, out, qs, ks, vs, 0, 1 if causal else 0, gran, scale, 0)

        def e2e():
            cand.quantized_attention_bf16(q, k, v, causal, scale, out)

        sk = _time(torch, kern)
        se = _time(torch, e2e)
        f = lambda s: rt.effective_tflops(int(c["b"]), int(c["h_q"]), int(c["d"]), int(c["pairs"]), s)
        rows[c["case_id"]] = {"cell": c["cell"], "l": c["l"], "kernel_only_us": sk * 1e6,
                              "kernel_only_tflops": f(sk), "end_to_end_us": se * 1e6,
                              "end_to_end_tflops": f(se), "preprocessing_share": 1.0 - sk / se}
        del q, k, v, out, q8, k8, v8, vp
        torch.cuda.empty_cache()

    def fam(key):
        dit = rt.geometric_mean([r[key] for r in rows.values() if r["cell"].startswith("D")])
        cau = rt.geometric_mean([r[key] for r in rows.values() if r["cell"].startswith("L")])
        return {"dit": dit, "causal": cau, "primary": (dit * cau) ** 0.5}

    res = {"qk_gran": a.qk_gran, "gpu": torch.cuda.get_device_name(), "cases": rows,
           "kernel_only": fam("kernel_only_tflops"), "end_to_end_events": fam("end_to_end_tflops"),
           "note": "CUDA events, warm L2, same inputs; NOT the Judge protocol"}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
