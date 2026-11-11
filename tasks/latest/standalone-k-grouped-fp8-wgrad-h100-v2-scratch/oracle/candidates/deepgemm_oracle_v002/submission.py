"""Upstream DeepGEMM reference under this task's ABI, lean glue (host-side oracle only).

Same upstream call as `deepgemm_oracle_v001`: DeepGEMM 2.6.1
(559d79fb6994a58b8a15b4b93bf13ccc16edf247) `k_grouped_fp8_gemm_nt_contiguous`
with upstream's own SM90 heuristic, nothing overridden.  Only the ABI glue
differs:

* v001 repacks each group with upstream's test-generator recipe
  (tests/generators.py:468-474: two torch copies per group per operand) and
  allocates the packed buffers on every call;
* this file produces byte-identical per-group K-major [X, K_g] slabs with ONE
  Triton launch for both operands, launches the compiled repack directly
  (CompiledKernel.run, cached per shape/alignment; the Python JIT launcher
  costs ~25-60 us per call), and reuses one grow-only packed workspace.
  Repack num_warps=2 (measured 2.15 -> 2.42 TB/s on H100 NVL vs 4).

Costs that stay inside the timed call: the repack, `ks.tolist()` (the SM90
API needs host `ks`), DeepGEMM's own `d.copy_(c)` (its epilogue supports
C == D only) and DeepGEMM's host dispatch.  Nothing numeric is cached; every
call repacks the current a/b and runs the full upstream GEMM.  If the direct
launch handle is unavailable (TRITON_INTERPRET=1) the normal launcher runs.

Domain: the FP32 D tensor map needs N % 4 == 0, so this covers the scored
grid only (not the private edge cases).

This file never enters the Work or Judge image; `source_admission.py` forbids
`import deep_gemm` in an agent submission by design.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

import deep_gemm
from deep_gemm.utils import set_mk_alignment_for_contiguous_layout

_RECIPE = (1, 1, 128)
set_mk_alignment_for_contiguous_layout(128)

_TK = 128   # every group K is a multiple of 128, so a 128-row block never straddles groups
_TX = 64
_GP = 128   # max G

_ws = {"buf": None}   # one grow-only uint8 workspace for both packed operands
_views = {}           # (kt, m, n) -> (packed_a fp8 view, packed_b fp8 view, pa_ptr, pb_ptr)
_plans = {}           # (kt, m, n, g, align) -> (run, function, packed_metadata) or None


@triton.jit
def _repack_k_major_kernel(a_ptr, b_ptr, pa_ptr, pb_ptr, ks_ptr, KT, M, N, G,
                           TK: tl.constexpr, TX: tl.constexpr, GP: tl.constexpr):
    # Program (kb, x-tile) of operand A (first tiles_a programs) or B (the rest).
    # dst[g_start*X + x*K_g + (k - g_start)] = src[k, x]   (upstream per-group [X, K_g] slab)
    pid = tl.program_id(0)
    nk = KT // TK
    tiles_a = nk * tl.cdiv(M, TX)
    if pid < tiles_a:
        src = a_ptr
        dst = pa_ptr
        X = M
        t = pid
    else:
        src = b_ptr
        dst = pb_ptr
        X = N
        t = pid - tiles_a
    kb = t % nk
    xb = t // nk
    k0 = kb * TK
    gi = tl.arange(0, GP)
    ks_all = tl.load(ks_ptr + gi, mask=gi < G, other=0)
    ends = tl.cumsum(ks_all, 0)
    starts = ends - ks_all
    hit = (starts <= k0) & (k0 < ends)          # exactly one non-empty group
    g_start = tl.sum(tl.where(hit, starts, 0))
    g_len = tl.sum(tl.where(hit, ks_all, 0))
    offs_k = k0 + tl.arange(0, TK)
    offs_x = xb * TX + tl.arange(0, TX)
    mx = offs_x < X
    tile = tl.load(src + offs_k.to(tl.int64)[:, None] * X + offs_x[None, :], mask=mx[None, :], other=0)
    d = (dst + g_start.to(tl.int64) * X + offs_x.to(tl.int64)[:, None] * g_len
         + (offs_k - g_start)[None, :])
    tl.store(d, tl.trans(tile), mask=mx[:, None])


def _packed(kt, m, n, device):
    key = (kt, m, n)
    hit = _views.get(key)
    if hit is not None:
        return hit
    off = ((kt * m + 255) // 256) * 256
    need = off + kt * n
    buf = _ws["buf"]
    if buf is None or buf.numel() < need:
        _views.clear()                      # views of the old buffer must not be reused
        buf = torch.empty(max(need, 1 << 20), dtype=torch.uint8, device=device)
        _ws["buf"] = buf
    pa = buf[: kt * m]
    pb = buf[off: off + kt * n]
    hit = (pa.view(torch.float8_e4m3fn), pb.view(torch.float8_e4m3fn), pa, pb)
    _views[key] = hit
    return hit


def k_grouped_fp8_gemm_tn_contiguous(a, b, a_scale, b_scale, ks, c, expected_k_hint, out) -> None:
    kt, m = a.shape
    n = b.shape[1]
    g = out.shape[0]
    ks_cpu = ks.tolist()                              # host ks required by the SM90 API
    if kt == 0:
        packed_a, packed_b = a.reshape(-1), b.reshape(-1)
    else:
        packed_a, packed_b, pa_u8, pb_u8 = _packed(kt, m, n, a.device)
        ap, bp, ksp = a.data_ptr(), b.data_ptr(), ks.data_ptr()
        # Triton specialises pointers on 16-byte alignment; the workspace is always aligned.
        key = (kt, m, n, g, (ap & 15) == 0, (bp & 15) == 0, (ksp & 15) == 0)
        grid0 = (kt // _TK) * (triton.cdiv(m, _TX) + triton.cdiv(n, _TX))
        plan = _plans.get(key)
        if plan is None:
            ck = _repack_k_major_kernel[(grid0,)](a.view(torch.uint8), b.view(torch.uint8), pa_u8, pb_u8,
                                                  ks, kt, m, n, g, TK=_TK, TX=_TX, GP=_GP, num_warps=2)
            run = None if ck is None else ck.run
            _plans[key] = (run, ck.function, ck.packed_metadata) if run is not None else False
        elif plan is False:                           # no direct handle (interpreter): normal launcher
            _repack_k_major_kernel[(grid0,)](a.view(torch.uint8), b.view(torch.uint8), pa_u8, pb_u8,
                                             ks, kt, m, n, g, TK=_TK, TX=_TX, GP=_GP, num_warps=2)
        else:
            run, fn, md = plan
            stream = torch.cuda.current_stream(a.get_device()).cuda_stream
            run(grid0, 1, 1, stream, fn, md, None, None, None,
                ap, bp, pa_u8.data_ptr(), pb_u8.data_ptr(), ksp, kt, m, n, g, _TK, _TX, _GP)
    deep_gemm.k_grouped_fp8_gemm_nt_contiguous(
        (packed_a, a_scale.T), (packed_b, b_scale.T), out, ks_cpu, ks, c, recipe=_RECIPE,
    )
    return None
