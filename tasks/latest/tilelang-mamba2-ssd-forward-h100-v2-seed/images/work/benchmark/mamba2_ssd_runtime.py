#!/usr/bin/env python3
"""Public test runtime for the TileLang Mamba-2 SSD forward task, v3."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Mapping, Sequence

ENTRYPOINT = "ssd_forward"
CANDIDATE_MODULE = "tilelang_mamba2_ssd_candidate"

# --------------------------------------------------------------------------
# Workload model
# --------------------------------------------------------------------------
# A geometry is a Mamba-2 head layout (heads, head width, state width, groups).
# The public geometries are real deployed layouts or synthetic ones; none of
# them is a hidden-suite geometry.

GEOMETRIES: dict[str, dict[str, int]] = {
    "h48-p64-n256-g1": {"h": 48, "p": 64, "n": 256, "g": 1},
    "h96-p80-n128-g4": {"h": 96, "p": 80, "n": 128, "g": 4},
    "h40-p64-n64-g1": {"h": 40, "p": 64, "n": 64, "g": 1},
    "h256-p48-n128-g16": {"h": 256, "p": 48, "n": 128, "g": 16},
    "h48-p64-n128-g1": {"h": 48, "p": 64, "n": 128, "g": 1},
    "h64-p128-n256-g4": {"h": 64, "p": 128, "n": 256, "g": 4},
    "h24-p64-n128-g1": {"h": 24, "p": 64, "n": 128, "g": 1},
    "h80-p64-n64-g1": {"h": 80, "p": 64, "n": 64, "g": 1},
    "h24-p32-n64-g1": {"h": 24, "p": 32, "n": 64, "g": 1},
    "h48-p32-n128-g1": {"h": 48, "p": 32, "n": 128, "g": 1},
    # Correctness-only edge geometries.
    "E1": {"h": 6, "p": 16, "n": 16, "g": 1},
    "E2": {"h": 16, "p": 32, "n": 64, "g": 4},
    "E3": {"h": 12, "p": 48, "n": 32, "g": 3},
    "E4": {"h": 200, "p": 16, "n": 32, "g": 200},
    "E5": {"h": 7, "p": 256, "n": 64, "g": 1},
    "E6": {"h": 20, "p": 64, "n": 256, "g": 5},
}

# Workload families.  ``batch``: training and padded batches, batch >= 2.
# ``serving``: batch 1 at real request lengths.  ``long``: batch 1, long
# sequences.  ``real``: a layer captured from a real model on real text, timed
# once.
SCORE_FAMILIES: tuple[str, ...] = ("batch", "serving", "long", "real")
TIERS: tuple[str, ...] = ("recipe", "sweep", "real")
# Case count per family of the full public suite.
FINAL_FAMILY_COUNTS: dict[str, int] = {"batch": 6, "serving": 13, "long": 4, "real": 1}
FINAL_CASE_COUNT = sum(FINAL_FAMILY_COUNTS.values())
REAL_TRIALS = 1
MIN_LONG_LENGTH = 60000

# No tensor reaches 2**31 elements (the support domain says so).
MAX_TENSOR_ELEMENTS = 2**31 - 1
MAX_BATCH = 16

FAMILY_WEIGHTS: dict[str, float] = {family: 1.0 / len(SCORE_FAMILIES) for family in SCORE_FAMILIES}


# --------------------------------------------------------------------------
# Correctness gate
# --------------------------------------------------------------------------
# Qualified on the target H100 against the supplied seed and an emulated bf16
# tensor-core pipeline.  Honest implementations measured relative L2
# 1.62e-3 to 2.37e-3 across every family, band, decay regime, sequence length
# from 1 to 45056, chunk size from 64 to 512, and with the state carried in
# either FP32 or bf16.  Planted bugs -- dropped D residual, dropped inter-chunk
# state, wrong group mapping, causal mask off by one, missing intra-chunk decay
# -- measured 0.21 to 2.30, i.e. two to three orders of magnitude away.  The
# limit sits 2.5x above the worst honest value and 35x below the nearest bug.
OUT_REL_L2_TOL = 6.0e-3
# Absolute limit expressed as a fraction of the reference tensor's own maximum
# magnitude.  A fixed constant cannot work here: the honest floor *is* one bf16
# ulp of that magnitude, so a constant would be a shape-dependent lottery.
# Worst honest measurement 4.7e-3; nearest planted bug 0.44.
OUT_SCALED_ABS_TOL = 1.5e-2
OUT_REL_TOL = 3.01 / 128.0
# Captured layers are gated on relative L2 alone, against their own limit.  The
# seed, the oracle and r1's best kernel all measured at most 5.24e-3 on the 24
# Judge layers (H100 SXM, 2026-11-08); the limit is 2.5x that.  Their dt spans
# 1e-11 to 9, so long-memory heads carry the state across hundreds of chunks: an
# inter-chunk recurrence accumulated in bf16 instead of FP32 -- a contract
# violation that generated inputs do not expose -- measured 1.5e-2 and 3.4e-2 on
# two layers and fails here.  Planted bugs on the same layers: state dropped at
# chunk boundaries, B and C swapped and x shifted one token are caught on 24/24
# layers, a missing D residual and a decay error of A*1.1 on 23/24.
REAL_LAYER_REL_L2_TOL = 1.3e-2

# --------------------------------------------------------------------------
# Reference blocking.  Bounds the reference's peak memory independently of the
# case, so the Judge's own footprint never depends on the candidate.
# --------------------------------------------------------------------------
REF_CHUNK = 128
REF_HEAD_BLOCK = 8
REF_CHUNK_BLOCK = 32


def shape_in_domain(geometry: Mapping[str, int], batch: int, seqlen: int) -> bool:
    """The support domain of AGENT_TASK.md: batch 1-16 and no tensor at 2**31 elements."""
    return (
        1 <= int(batch) <= MAX_BATCH
        and int(seqlen) >= 1
        and int(batch) * int(seqlen) * geometry["h"] * geometry["p"] <= MAX_TENSOR_ELEMENTS
        and int(batch) * int(seqlen) * geometry["g"] * geometry["n"] <= MAX_TENSOR_ELEMENTS
    )


def case_io_bytes(geometry: Mapping[str, int], batch: int, seqlen: int) -> int:
    """x and out (bf16), dt (fp32), B and C (bf16); A and D are excluded."""
    return 4 * int(batch) * int(seqlen) * (
        geometry["h"] * geometry["p"] + geometry["h"] + geometry["g"] * geometry["n"]
    )


def effective_flops(batch: int, seqlen: int, heads: int, headdim: int, dstate: int) -> float:
    """Chunk-size-independent ideal work of the recurrence.

    Two ``P*N`` MAC counts per token per head: one for the state update, one for
    the output projection.  This is deliberately *not* the upstream chunk-scan
    benchmark's ``2*b*L*Q*H*P*0.5 + 2*b*L*H*P*N``, which depends on the chunk
    size; the two numbers must never be compared.
    """
    return 4.0 * float(batch) * float(seqlen) * float(heads) * float(headdim) * float(dstate)


def effective_tflops(
    batch: int, seqlen: int, heads: int, headdim: int, dstate: int, seconds: float
) -> float:
    return effective_flops(batch, seqlen, heads, headdim, dstate) / finite_positive(
        seconds, "measured seconds"
    ) / 1.0e12


def finite_positive(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise RuntimeError(f"{label} is not a finite positive number: {value!r}")
    return number


def geometric_mean(values: Iterable[float]) -> float:
    items = [finite_positive(value, "geometric mean term") for value in values]
    if not items:
        raise RuntimeError("geometric mean of an empty sequence")
    return math.exp(math.fsum(math.log(item) for item in items) / len(items))


# --------------------------------------------------------------------------
# Workload generation
# --------------------------------------------------------------------------
# Scored cases draw from one documented distribution.  ``A`` follows the
# reference Mamba-2 initialisation range, and ``dt`` is log-uniform over the
# range a trained model actually produces, so a single case contains both
# short-memory heads (per-step decay ~0.20) and long-memory heads (~0.999).
# That matters for correctness: with fast decay alone a candidate could drop the
# inter-chunk state term entirely and still pass.
A_LOG_RANGE = (0.0, math.log(16.0))          # A = -exp(u),  A in [-16, -1]
DT_LOG_RANGE = (math.log(1.0e-3), math.log(1.0e-1))

REGIMES = ("mixed", "slow_decay", "fast_decay", "unit_dt")


def _log_uniform(torch: Any, shape, low: float, high: float, generator: Any, device: Any):
    raw = torch.rand(shape, generator=generator, device=device, dtype=torch.float32)
    return torch.exp(raw * (high - low) + low)


def _regime_ranges(regime: str) -> tuple[tuple[float, float], tuple[float, float]]:
    if regime == "mixed":
        return A_LOG_RANGE, DT_LOG_RANGE
    if regime == "slow_decay":
        # exp(A*dt) close to 1: the inter-chunk state carries real signal and a
        # candidate that truncates the scan at a chunk boundary fails here.
        return (0.0, math.log(2.0)), (math.log(1.0e-4), math.log(1.0e-3))
    if regime == "fast_decay":
        # exp(A*dt) far below 1: the intra-chunk term dominates and any
        # mis-scaled decay shows up immediately.
        return (math.log(8.0), math.log(16.0)), (math.log(5.0e-2), math.log(2.0e-1))
    if regime == "unit_dt":
        return (math.log(1.0), math.log(1.0)), (0.0, 0.0)
    raise ValueError(f"unknown regime {regime!r}")


def derive_seed(case_seed: int, label: str) -> int:
    """A 63-bit seed for one tensor set of one case, derived with SHA-256.

    Warm-up, precall and every measured trial draw from their own derived seed, so
    knowing one tensor set says nothing usable about the next.
    """
    material = f"tilelang-ssd-v2:{int(case_seed)}:{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def generate_inputs(torch: Any, case: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """Deterministic workload for one case and seed, on the current device."""
    geometry = GEOMETRIES[str(case["geometry"])]
    heads = int(geometry["h"])
    headdim = int(geometry["p"])
    dstate = int(geometry["n"])
    groups = int(geometry["g"])
    batch = int(case["batch"])
    seqlen = int(case["seqlen"])
    regime = str(case.get("regime", "mixed"))
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) & 0x7FFF_FFFF_FFFF_FFFF)

    a_range, dt_range = _regime_ranges(regime)
    # B and C are unit-variance, exactly as a linear projection of a normalised
    # hidden state produces them.  Scaling them down by 1/sqrt(N) -- which looks
    # like sensible fan-in normalisation -- was measured to drive the state-space
    # term to 1% of the D residual in L2, at which point the numerical gate stops
    # testing the scan at all and a candidate that computes only D*x passes.
    # Unscaled, the two terms land within 11% of each other.
    bc_scale = 0.0 if str(case.get("bc_mode", "random")) == "zero" else 1.0

    x = torch.randn(
        (batch, seqlen, heads, headdim), generator=generator, device=device, dtype=torch.float32
    ).to(torch.bfloat16)
    b_proj = (
        torch.randn(
            (batch, seqlen, groups, dstate), generator=generator, device=device, dtype=torch.float32
        )
        * bc_scale
    ).to(torch.bfloat16)
    c_proj = (
        torch.randn(
            (batch, seqlen, groups, dstate), generator=generator, device=device, dtype=torch.float32
        )
        * bc_scale
    ).to(torch.bfloat16)
    a_log = _log_uniform(torch, (heads,), a_range[0], a_range[1], generator, device)
    a_param = -a_log
    dt = _log_uniform(torch, (batch, seqlen, heads), dt_range[0], dt_range[1], generator, device)
    d_param = torch.randn((heads,), generator=generator, device=device, dtype=torch.float32)
    if str(case.get("d_mode", "random")) == "zero":
        d_param = torch.zeros_like(d_param)

    return {
        "x": x.contiguous(),
        "dt": dt.contiguous(),
        "A": a_param.contiguous(),
        "B": b_proj.contiguous(),
        "C": c_proj.contiguous(),
        "D": d_param.contiguous(),
    }


# --------------------------------------------------------------------------
# Independent FP32 reference
# --------------------------------------------------------------------------
def _use_exact_fp32(torch: Any) -> None:
    """No TF32 anywhere.  The reference must be FP32, not 10-bit mantissa."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends.cuda, "matmul") and hasattr(
        torch.backends.cuda.matmul, "fp32_precision"
    ):
        torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.set_float32_matmul_precision("highest")


def reference_output(
    torch: Any,
    x: Any,
    dt: Any,
    a_param: Any,
    b_proj: Any,
    c_proj: Any,
    d_param: Any,
    *,
    chunk: int = REF_CHUNK,
    head_block: int = REF_HEAD_BLOCK,
    chunk_block: int = REF_CHUNK_BLOCK,
) -> Any:
    """Blocked FP32 chunked SSD.  Independent of any candidate or upstream code.

    Blocking is over ``(batch, head block, chunk block)`` so peak memory is set
    by the block sizes rather than by the case, and the inter-chunk scan stays
    an explicit sequential recurrence -- the one part of the algorithm that must
    not be reassociated, because dividing by a chunk's accumulated decay is how
    a fast closed form loses all its digits when the decay underflows.
    """
    _use_exact_fp32(torch)
    batch, seqlen, heads, headdim = x.shape
    groups, dstate = b_proj.shape[2], b_proj.shape[3]
    per_group = heads // groups
    device = x.device
    f32 = torch.float32

    pad = (-seqlen) % chunk
    total = seqlen + pad
    nchunks = total // chunk
    tril = torch.tril(torch.ones((chunk, chunk), dtype=torch.bool, device=device))

    # FP32 out: rounding the reference to bf16 first would inject a second ulp
    # of noise into a gate whose honest floor already *is* one bf16 ulp.
    out = torch.empty((batch, seqlen, heads, headdim), dtype=f32, device=device)
    a32 = a_param.to(f32)
    d32 = d_param.to(f32)

    for b in range(batch):
        for h0 in range(0, heads, head_block):
            h1 = min(h0 + head_block, heads)
            hb = h1 - h0
            head_ids = torch.arange(h0, h1, device=device)
            group_ids = torch.div(head_ids, per_group, rounding_mode="floor")

            # dA cumulative sum inside every chunk: [nchunks, chunk, hb]
            dt_bh = dt[b, :, h0:h1].to(f32)
            if pad:
                dt_bh = torch.cat(
                    [dt_bh, torch.zeros((pad, hb), dtype=f32, device=device)], dim=0
                )
            dt_bh = dt_bh.view(nchunks, chunk, hb)
            dA_cs = torch.cumsum(dt_bh * a32[h0:h1], dim=1)
            chunk_decay = torch.exp(dA_cs[:, -1, :])                     # [nc, hb]

            # ---- per-chunk states -----------------------------------------
            chunk_states = torch.empty((nchunks, hb, headdim, dstate), dtype=f32, device=device)
            for c0 in range(0, nchunks, chunk_block):
                c1 = min(c0 + chunk_block, nchunks)
                sl = slice(c0 * chunk, min(c1 * chunk, seqlen))
                span = c1 - c0
                x_blk = torch.zeros((span * chunk, hb, headdim), dtype=f32, device=device)
                b_blk = torch.zeros((span * chunk, hb, dstate), dtype=f32, device=device)
                take = sl.stop - sl.start
                x_blk[:take] = x[b, sl, h0:h1].to(f32)
                b_blk[:take] = b_proj[b, sl][:, group_ids].to(f32)
                x_blk = x_blk.view(span, chunk, hb, headdim)
                b_blk = b_blk.view(span, chunk, hb, dstate)
                decay_states = torch.exp(dA_cs[c0:c1, -1:, :] - dA_cs[c0:c1])
                weight = dt_bh[c0:c1] * decay_states                     # [span, chunk, hb]
                chunk_states[c0:c1] = torch.einsum(
                    "qshp,qsh,qshn->qhpn", x_blk, weight, b_blk
                )

            # ---- inter-chunk sequential scan ------------------------------
            states = torch.empty_like(chunk_states)
            carry = torch.zeros((hb, headdim, dstate), dtype=f32, device=device)
            for c in range(nchunks):
                states[c] = carry
                carry = chunk_decay[c][:, None, None] * carry + chunk_states[c]

            # ---- intra-chunk term plus state read-out ---------------------
            for c0 in range(0, nchunks, chunk_block):
                c1 = min(c0 + chunk_block, nchunks)
                sl = slice(c0 * chunk, min(c1 * chunk, seqlen))
                span = c1 - c0
                take = sl.stop - sl.start
                x_blk = torch.zeros((span * chunk, hb, headdim), dtype=f32, device=device)
                b_blk = torch.zeros((span * chunk, hb, dstate), dtype=f32, device=device)
                c_blk = torch.zeros((span * chunk, hb, dstate), dtype=f32, device=device)
                x_blk[:take] = x[b, sl, h0:h1].to(f32)
                b_blk[:take] = b_proj[b, sl][:, group_ids].to(f32)
                c_blk[:take] = c_proj[b, sl][:, group_ids].to(f32)
                x_blk = x_blk.view(span, chunk, hb, headdim)
                b_blk = b_blk.view(span, chunk, hb, dstate)
                c_blk = c_blk.view(span, chunk, hb, dstate)

                cb = torch.einsum("qlhn,qshn->qhls", c_blk, b_blk)
                seg = dA_cs[c0:c1].permute(0, 2, 1)                      # [span, hb, chunk]
                decay = torch.where(
                    tril,
                    torch.exp(seg[..., :, None] - seg[..., None, :]),
                    torch.zeros((), dtype=f32, device=device),
                )
                scores = cb * decay * dt_bh[c0:c1].permute(0, 2, 1)[..., None, :]
                y = torch.einsum("qhls,qshp->qlhp", scores, x_blk)
                y = y + torch.einsum(
                    "qlhn,qhpn->qlhp", c_blk, states[c0:c1]
                ) * torch.exp(dA_cs[c0:c1])[..., None]
                y = y.reshape(span * chunk, hb, headdim)[:take]
                y = y + d32[h0:h1][None, :, None] * x[b, sl, h0:h1].to(f32)
                out[b, sl, h0:h1] = y

    return out


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------
_COMPARE_CHUNK_ELEMENTS = 1 << 26


def output_is_finite(torch: Any, candidate: Any) -> bool:
    """Chunked finiteness check for a trial that is not compared to a reference."""
    flat = candidate.reshape(-1)
    return all(
        bool(torch.isfinite(flat[start:start + _COMPARE_CHUNK_ELEMENTS]).all().item())
        for start in range(0, int(flat.numel()), _COMPARE_CHUNK_ELEMENTS)
    )


@dataclass(frozen=True)
class ComparisonResult:
    ok: bool
    rel_l2: float
    max_abs: float
    max_abs_scaled: float
    max_rel: float
    violations: int
    detail: str


def compare_output(
    torch: Any, candidate: Any, reference: Any, *, real_layer: bool = False
) -> ComparisonResult:
    """Elementwise scaled-absolute OR relative, plus a global relative L2.

    The elementwise rule accepts on *either* test, so its two reported maxima
    normally come from different elements and neither one alone is a verdict.
    The verdict is the violation count, derived from the per-element
    disjunction, together with the relative L2 -- the only statistic measured to
    be stable across chunk size, state dtype, decay regime, and shape.
    """
    if tuple(candidate.shape) != tuple(reference.shape):
        return ComparisonResult(
            False, 0.0, 0.0, 0.0, 0.0, 0,
            f"shape {tuple(candidate.shape)} != {tuple(reference.shape)}",
        )
    # Chunked over the flattened tensors: a whole-tensor FP32 diff, its
    # denominator and its masks cost six copies of the output, which at the
    # largest cases is more device memory than the case itself.
    got_flat = candidate.reshape(-1)
    want_flat = reference.reshape(-1)
    count = int(want_flat.numel())
    step = _COMPARE_CHUNK_ELEMENTS
    scale = 0.0
    for start in range(0, count, step):
        want = want_flat[start:start + step].to(torch.float32)
        if not bool(torch.isfinite(want).all().item()):
            raise RuntimeError("reference contains a non-finite element -- Judge infrastructure error")
        if not bool(torch.isfinite(got_flat[start:start + step]).all().item()):
            return ComparisonResult(False, 0.0, 0.0, 0.0, 0.0, 0, "candidate output contains NaN or Inf")
        scale = max(scale, float(want.abs().max().item()))
    if not math.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("reference is identically zero -- Judge infrastructure error")

    max_abs = max_rel = 0.0
    diff_sq = want_sq = 0.0
    violations = 0
    for start in range(0, count, step):
        want = want_flat[start:start + step].to(torch.float32)
        diff = (got_flat[start:start + step].to(torch.float32) - want).abs()
        denominator = want.abs()
        max_abs = max(max_abs, float(diff.max().item()))
        max_rel = max(max_rel, float((diff / denominator.clamp_min(scale * 1.0e-6)).max().item()))
        diff_sq += float(diff.double().square().sum().item())
        want_sq += float(want.double().square().sum().item())
        accepted = (diff <= OUT_SCALED_ABS_TOL * scale) | (diff <= OUT_REL_TOL * denominator)
        violations += int((~accepted).sum().item())
        del want, diff, denominator, accepted
    rel_l2 = math.sqrt(diff_sq) / math.sqrt(want_sq)

    detail = ""
    if real_layer:
        if rel_l2 > REAL_LAYER_REL_L2_TOL:
            detail = f"relative L2 {rel_l2:.4g} on a captured layer exceeds {REAL_LAYER_REL_L2_TOL:g}"
    elif violations:
        detail = (
            f"{violations} element(s) outside both the scaled-absolute limit "
            f"({OUT_SCALED_ABS_TOL:g} * {scale:.4g}) and the relative limit ({OUT_REL_TOL:g})"
        )
    elif rel_l2 > OUT_REL_L2_TOL:
        detail = f"relative L2 {rel_l2:.4g} exceeds {OUT_REL_L2_TOL:g}"
    return ComparisonResult(
        not detail, rel_l2, max_abs, max_abs / scale, max_rel, violations, detail
    )


# --------------------------------------------------------------------------
# Trusted timing primitives
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TrustedClock:
    """Timing primitives captured *before* the candidate module is imported.

    The candidate runs in this interpreter, so anything reachable through
    ``torch.cuda`` or ``time`` after import is no longer trusted -- one line of
    ``torch.cuda.synchronize = lambda *a, **k: None`` would make every measured
    call return before its kernel finished.  Binding these here, while the
    candidate is still only a file on disk, closes that, and the
    ``max(host wall, CUDA event)`` rule keeps the score bounded from below even
    if the event class itself is replaced.
    """

    perf_counter: Callable[[], float]
    synchronize: Callable[..., Any]
    current_stream: Callable[..., Any]
    event: Callable[..., Any]
    reset_peak_memory_stats: Callable[..., Any]
    max_memory_allocated: Callable[..., int]
    memory_allocated: Callable[..., int]
    op_recorder: Callable[[], Any]

    @classmethod
    def capture(cls, torch: Any) -> "TrustedClock":
        return cls(
            perf_counter=time.perf_counter,
            synchronize=torch.cuda.synchronize,
            current_stream=torch.cuda.current_stream,
            event=torch.cuda.Event,
            reset_peak_memory_stats=torch.cuda.reset_peak_memory_stats,
            max_memory_allocated=torch.cuda.max_memory_allocated,
            memory_allocated=torch.cuda.memory_allocated,
            op_recorder=torch_op_recorder_class(),
        )


@dataclass
class PreparedInvocation:
    case_id: str
    family: str
    geometry: str
    batch: int
    seqlen: int
    heads: int
    headdim: int
    dstate: int
    groups: int
    x: Any
    dt: Any
    a_param: Any
    b_proj: Any
    c_proj: Any
    d_param: Any
    out: Any
    reference: Any
    input_ptrs: tuple[int, ...]
    output_ptr: int

    def io_bytes(self) -> int:
        return sum(
            int(tensor.numel()) * int(tensor.element_size())
            for tensor in (self.x, self.dt, self.b_proj, self.c_proj, self.out)
        )


def _case_geometry(case: Mapping[str, Any]) -> tuple[int, int, int, int]:
    geometry = GEOMETRIES[str(case["geometry"])]
    return int(geometry["h"]), int(geometry["p"]), int(geometry["n"]), int(geometry["g"])


def prepare_invocation(
    torch: Any,
    case: Mapping[str, Any],
    seed: int,
    *,
    with_reference: bool = True,
    tensors: Mapping[str, Any] | None = None,
    phases: dict[str, float] | None = None,
) -> PreparedInvocation:
    """One tensor set: generated from ``seed``, or copied to the device from ``tensors``.

    ``phases`` accumulates the seconds spent generating inputs and computing the
    reference, for the per-case phase record.
    """
    started = time.perf_counter()
    if tensors is None:
        tensors = generate_inputs(torch, case, seed)
    else:
        device = torch.device("cuda", torch.cuda.current_device())
        tensors = {name: value.to(device).contiguous() for name, value in tensors.items()}
    torch.cuda.synchronize()
    generated = time.perf_counter()
    heads, headdim, dstate, groups = _case_geometry(case)
    batch = int(case["batch"])
    seqlen = int(case["seqlen"])
    reference = (
        reference_output(
            torch, tensors["x"], tensors["dt"], tensors["A"],
            tensors["B"], tensors["C"], tensors["D"],
        )
        if with_reference
        else None
    )
    torch.cuda.synchronize()
    if phases is not None:
        phases["generate"] = phases.get("generate", 0.0) + (generated - started)
        phases["reference"] = phases.get("reference", 0.0) + (time.perf_counter() - generated)
    out = torch.full(
        (batch, seqlen, heads, headdim),
        float("nan"),
        dtype=torch.bfloat16,
        device=tensors["x"].device,
    )
    inputs = (tensors["x"], tensors["dt"], tensors["A"], tensors["B"], tensors["C"], tensors["D"])
    return PreparedInvocation(
        case_id=str(case["case_id"]),
        family=str(case["family"]),
        geometry=str(case["geometry"]),
        batch=batch,
        seqlen=seqlen,
        heads=heads,
        headdim=headdim,
        dstate=dstate,
        groups=groups,
        x=inputs[0], dt=inputs[1], a_param=inputs[2],
        b_proj=inputs[3], c_proj=inputs[4], d_param=inputs[5],
        out=out,
        reference=reference,
        input_ptrs=tuple(int(tensor.data_ptr()) for tensor in inputs),
        output_ptr=int(out.data_ptr()),
    )


def _invoke(entrypoint: Callable[..., Any], prepared: PreparedInvocation) -> None:
    returned = entrypoint(
        prepared.x, prepared.dt, prepared.a_param,
        prepared.b_proj, prepared.c_proj, prepared.d_param, prepared.out,
    )
    if returned is not None:
        raise RuntimeError(f"{ENTRYPOINT} must return None")


def _poison_output(prepared: PreparedInvocation) -> None:
    prepared.out.fill_(float("nan"))


# Fingerprints are taken chunk by chunk: an int64 copy of a whole activation
# would cost eight bytes per element of device memory for a check.
_CHECKSUM_CHUNK_ELEMENTS = 1 << 26


def _checksums(torch: Any, values: Any) -> tuple[int, int, int]:
    """Three integer checksums over one tensor's raw bit pattern.

    A single additive sum is blind to any permutation of the elements; adding
    the sums over every 7th and every 13th element makes a position-preserving
    forgery impractical while reading every element once.
    """
    flat = values.reshape(-1)
    count = int(flat.numel())
    total = every7 = every13 = 0
    for start in range(0, count, _CHECKSUM_CHUNK_ELEMENTS):
        part = flat[start:start + _CHECKSUM_CHUNK_ELEMENTS].to(torch.int64)
        total += int(part.sum().item())
        every7 += int(part[(-start) % 7::7].sum().item())
        every13 += int(part[(-start) % 13::13].sum().item())
    return total, every7, every13


def input_signature(torch: Any, prepared: PreparedInvocation) -> tuple[Any, ...]:
    return (
        _checksums(torch, prepared.x.view(torch.int16)),
        _checksums(torch, prepared.dt.view(torch.int32)),
        _checksums(torch, prepared.a_param.view(torch.int32)),
        _checksums(torch, prepared.b_proj.view(torch.int16)),
        _checksums(torch, prepared.c_proj.view(torch.int16)),
        _checksums(torch, prepared.d_param.view(torch.int32)),
    )


# --------------------------------------------------------------------------
# Runtime TileLang-only operator record
# --------------------------------------------------------------------------
# Submitted compute must be TileLang.  Source admission reads the text; this
# records what actually runs.  Every ATen operator the candidate dispatches
# during an untimed precall or edge-fixture call is recorded through a
# TorchDispatchMode that is built before the candidate module is imported.
# TileLang kernels launch through TileLang's own adapter and dispatch no ATen
# operator, so a TileLang-only candidate records allocations and views only.
# Anything else -- a GEMM, an elementwise or scan operator, a copy or dtype
# conversion, a host read of a device value -- is compute outside TileLang.
ALLOWED_TORCH_OPS: frozenset[str] = frozenset(
    {
        # allocation and constant initialisation of the candidate's own buffers
        "aten::empty.memory_format",
        "aten::empty_strided",
        "aten::empty_like",
        "aten::new_empty",
        "aten::new_empty_strided",
        "aten::zeros",
        "aten::zeros_like",
        "aten::new_zeros",
        "aten::zero_",
        "aten::full",
        "aten::full_like",
        "aten::new_full",
        "aten::fill_.Scalar",
        # metadata-only views
        "aten::alias",
        "aten::as_strided",
        "aten::detach",
        "aten::expand",
        "aten::narrow",
        "aten::permute",
        "aten::select.int",
        "aten::slice.Tensor",
        "aten::split.Tensor",
        "aten::split_with_sizes",
        "aten::squeeze",
        "aten::squeeze.dim",
        "aten::squeeze.dims",
        "aten::t",
        "aten::transpose.int",
        "aten::unbind.int",
        "aten::unsqueeze",
        "aten::view",
        "aten::view.dtype",
        "aten::_reshape_alias",
        "aten::_unsafe_view",
    }
)


class GateFailure(RuntimeError):
    """A failure attributed to one named Judge gate."""

    def __init__(self, gate: str, message: str) -> None:
        super().__init__(message)
        self.gate = gate


def torch_op_recorder_class() -> type:
    """Return a TorchDispatchMode subclass that records every dispatched op."""
    from torch.utils._python_dispatch import TorchDispatchMode  # noqa: PLC0415

    class _TorchOpRecorder(TorchDispatchMode):
        def __init__(self) -> None:
            super().__init__()
            self.ops: list[str] = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # noqa: ANN001
            self.ops.append(str(func.name()))
            return func(*args, **(kwargs or {}))

    return _TorchOpRecorder


def check_torch_ops(ops: Iterable[str], label: str) -> list[str]:
    """Return the sorted distinct ops, or fail the TileLang-only gate."""
    seen = sorted(set(ops))
    forbidden = [name for name in seen if name not in ALLOWED_TORCH_OPS]
    if forbidden:
        raise GateFailure(
            "tilelang_only_ok",
            f"{label}: compute outside TileLang -- the call dispatched host torch "
            f"operator(s) {forbidden}; torch may only allocate buffers and create views",
        )
    return seen


def _recorded_invoke(
    clock: TrustedClock, entrypoint: Callable[..., Any], prepared: PreparedInvocation
) -> list[str]:
    recorder = clock.op_recorder()
    with recorder:
        _invoke(entrypoint, prepared)
    return list(recorder.ops)


# --------------------------------------------------------------------------
# Candidate loading and hardware inventory
# --------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_candidate(path: Path) -> tuple[ModuleType, Callable[..., Any], str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"candidate submission not found: {path}")
    digest = sha256_file(path)
    spec = importlib.util.spec_from_file_location(CANDIDATE_MODULE, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load candidate module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    entrypoint = getattr(module, ENTRYPOINT, None)
    if entrypoint is None or not callable(entrypoint):
        raise AttributeError(f"candidate does not define a callable {ENTRYPOINT}")
    return module, entrypoint, digest


# Capability envelope: what the task contract actually needs, not the GPU model
# it was qualified on.  The GPU name and UUID are recorded as provenance only;
# they never gate admission.  Keep these two constants identical in the Judge
# verifier (tests assert the pair).
MIN_COMPUTE_CAPABILITY = (8, 0)
MIN_DEVICE_MEMORY_BYTES = 0


def hardware_inventory(torch: Any) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device is visible")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"exactly one CUDA device must be visible, found {torch.cuda.device_count()}"
        )
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    capability = (int(properties.major), int(properties.minor))
    total_memory_bytes = int(properties.total_memory)
    if capability < MIN_COMPUTE_CAPABILITY:
        raise RuntimeError(
            f"GPU compute capability {capability} is below the task minimum "
            f"{MIN_COMPUTE_CAPABILITY}: {properties.name}"
        )
    if total_memory_bytes < MIN_DEVICE_MEMORY_BYTES:
        raise RuntimeError(
            f"GPU memory {total_memory_bytes} bytes is below the task minimum "
            f"{MIN_DEVICE_MEMORY_BYTES} bytes: {properties.name}"
        )
    try:
        import tilelang  # noqa: PLC0415

        tilelang_version = str(getattr(tilelang, "__version__", "unknown"))
    except Exception as error:  # pragma: no cover - reported, never swallowed
        raise RuntimeError(f"tilelang is not importable in the evaluation image: {error}") from error
    return {
        "device_name": str(properties.name),
        "uuid": str(getattr(properties, "uuid", "")),
        "capability": f"sm_{capability[0]}{capability[1]}",
        "compute_capability": list(capability),
        "min_compute_capability": list(MIN_COMPUTE_CAPABILITY),
        "min_device_memory_bytes": MIN_DEVICE_MEMORY_BYTES,
        "total_memory_bytes": total_memory_bytes,
        "multi_processor_count": int(properties.multi_processor_count),
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "tilelang_version": tilelang_version,
    }


# --------------------------------------------------------------------------
# Case execution
# --------------------------------------------------------------------------
# TileLang compiles through nvcc on first call, so the warm-up window has to be
# generous; it is still a hard cap, because a submission that autotunes ninety
# configurations per case at evaluation time is not a submission, it is a search.
WARMUP_WALL_CAP_SECONDS = 300.0
# Transient device memory the candidate may hold, as a multiple of the case's
# own input plus output bytes, and never more than an absolute cap.  Staging
# every chunk state in FP32 fits inside this; an unbounded global scratch does
# not.  The cap keeps the largest long-context case inside one 80 GB device
# next to the Judge's own tensor sets and reference.
WORKSPACE_MULTIPLE = 4.0
WORKSPACE_CAP_BYTES = 24 * 1024**3


def workspace_limit_bytes(io_bytes: int) -> float:
    return min(WORKSPACE_MULTIPLE * float(io_bytes), float(WORKSPACE_CAP_BYTES))


def _measure_once(
    clock: TrustedClock, entrypoint: Callable[..., Any], prepared: PreparedInvocation
) -> tuple[float, float, float]:
    start_event = clock.event(enable_timing=True)
    end_event = clock.event(enable_timing=True)
    clock.synchronize()
    wall_start = clock.perf_counter()
    start_event.record(clock.current_stream())
    _invoke(entrypoint, prepared)
    end_event.record(clock.current_stream())
    clock.synchronize()
    host_wall = finite_positive(clock.perf_counter() - wall_start, "host wall time")
    event_seconds = finite_positive(
        float(start_event.elapsed_time(end_event)) / 1000.0, "CUDA event time"
    )
    return max(host_wall, event_seconds), host_wall, event_seconds


def _verify(
    torch: Any,
    prepared: PreparedInvocation,
    label: str,
    *,
    release: bool = False,
    real_layer: bool = False,
) -> dict[str, float]:
    """Compare one invocation against its own reference.

    ``release`` frees the reference afterwards.  Measured trials set it: the
    reference has served its purpose by then.  ``real_layer`` applies the
    captured-layer rule and reports its metrics under ``real_*`` keys.
    """
    result = compare_output(torch, prepared.out, prepared.reference, real_layer=real_layer)
    if release:
        prepared.reference = None
    if not result.ok:
        raise RuntimeError(f"{label}: {result.detail}")
    prefix = "real_" if real_layer else ""
    metrics = {
        f"{prefix}rel_l2": result.rel_l2,
        f"{prefix}max_abs": result.max_abs,
        f"{prefix}max_abs_scaled": result.max_abs_scaled,
        f"{prefix}max_rel": result.max_rel,
    }
    if real_layer:
        metrics["real_elementwise_violations"] = float(result.violations)
    return metrics


def run_edge_case(
    torch: Any,
    clock: TrustedClock,
    entrypoint: Callable[..., Any],
    case: Mapping[str, Any],
) -> dict[str, Any]:
    """Correctness only.  Never timed, never scored.

    The first call absorbs compilation; the second runs under the TileLang-only
    operator record on a freshly poisoned output and is verified again.
    """
    label = f"edge fixture {case['case_id']}"
    prepared = prepare_invocation(torch, case, int(case["seed"]))
    before = input_signature(torch, prepared)
    _invoke(entrypoint, prepared)
    clock.synchronize()
    _verify(torch, prepared, label)
    _poison_output(prepared)
    torch_ops = check_torch_ops(_recorded_invoke(clock, entrypoint, prepared), label)
    clock.synchronize()
    if input_signature(torch, prepared) != before:
        raise RuntimeError(f"{label}: candidate modified its inputs")
    metrics = _verify(torch, prepared, label, release=True)
    result = {
        "case_id": str(case["case_id"]),
        "family": str(case["family"]),
        "geometry": str(case["geometry"]),
        "scored": False,
        "batch": prepared.batch,
        "seqlen": prepared.seqlen,
        "regime": str(case.get("regime", "mixed")),
        "tilelang_only_ok": True,
        "torch_ops": torch_ops,
        **{key: float(value) for key, value in metrics.items()},
    }
    del prepared
    torch.cuda.empty_cache()
    return result


def run_scored_case(
    torch: Any,
    clock: TrustedClock,
    entrypoint: Callable[..., Any],
    case: Mapping[str, Any],
    *,
    trials: int,
    warmups: int,
    real_tensors: Mapping[str, Any] | None = None,
    perf_only: bool = False,
) -> dict[str, Any]:
    """Time one case on fresh tensors; verify the warm-up and one hidden trial.

    ``perf_only`` (public runner only, never the Judge) skips every reference
    and comparison and keeps the finiteness and input checks.

    ``warm`` absorbs compilation, proves correctness and measures the workspace
    before anything is timed; it is then released.  ``precall`` matches the case's
    shapes but has its own seed: it is invoked, untimed and under the TileLang-only operator
    record, immediately before every measured call, so a candidate that caches
    an answer keyed on pointers misses and one keyed on content returns the
    precall's answer.  Each trial's tensors are created only after that precall call
    and are invoked exactly once inside the timer.  A trial's tensors stay alive
    until the next trial has been allocated, so every trial's pointers differ
    from the precall's and the previous trial's, while device memory holds at most
    three tensor sets.

    Every trial is checked for complete, finite writes and unchanged inputs.
    Exactly one trial, named by the manifest's ``check_trial`` and unknown to the
    candidate, is also compared against its own FP32 reference; the warm-up is
    always compared.  A real case runs one trial on the captured layer, after a
    warm-up and precall on generated tensors of the same shape, and that trial is
    always compared.
    """
    if isinstance(trials, bool) or not isinstance(trials, int) or trials <= 0:
        raise ValueError("trials must be a positive integer")
    if isinstance(warmups, bool) or not isinstance(warmups, int) or warmups <= 0:
        raise ValueError("warmups must be a positive integer")
    case_id = str(case["case_id"])
    seed = int(case["seed"])
    is_real = str(case["family"]) == "real"
    if is_real and real_tensors is None:
        raise RuntimeError(f"case {case_id}: the captured layer {case.get('layer')!r} is not loaded")
    case_trials = REAL_TRIALS if is_real else trials
    check_trial = 0 if is_real else int(case["check_trial"])
    if not 0 <= check_trial < case_trials:
        raise ValueError(f"case {case_id}: check_trial {check_trial} is outside the trials")
    metrics: dict[str, float] = {}
    phases: dict[str, float] = {"generate": 0.0, "reference": 0.0, "compare": 0.0}

    def verify(prepared: PreparedInvocation, label: str, real_layer: bool = False) -> dict[str, float]:
        started = time.perf_counter()
        try:
            if perf_only:
                if not output_is_finite(torch, prepared.out):
                    raise RuntimeError(f"{label}: output is incomplete or contains NaN or Inf")
                return {}
            return _verify(torch, prepared, label, release=True, real_layer=real_layer)
        finally:
            phases["compare"] += time.perf_counter() - started

    warm = prepare_invocation(
        torch, case, derive_seed(seed, "warm"), with_reference=not perf_only, phases=phases
    )
    warm_signature = input_signature(torch, warm)
    io_bytes = warm.io_bytes()
    clock.reset_peak_memory_stats()
    memory_base = int(clock.memory_allocated())
    warm_start = clock.perf_counter()
    for _ in range(warmups):
        _poison_output(warm)
        _invoke(entrypoint, warm)
    clock.synchronize()
    warmup_wall = clock.perf_counter() - warm_start
    workspace_bytes = int(clock.max_memory_allocated()) - memory_base
    if warmup_wall > WARMUP_WALL_CAP_SECONDS:
        raise RuntimeError(
            f"case {case_id}: warm-up wall {warmup_wall:.1f}s exceeds the "
            f"{WARMUP_WALL_CAP_SECONDS:.0f}s cap"
        )
    if input_signature(torch, warm) != warm_signature:
        raise RuntimeError(f"case {case_id}: candidate modified its inputs")
    _merge_metrics(metrics, verify(warm, f"case {case_id} warm-up"))
    workspace_limit = workspace_limit_bytes(io_bytes)
    if workspace_bytes > workspace_limit:
        raise RuntimeError(
            f"case {case_id}: transient device memory {workspace_bytes / 2**30:.2f} GiB "
            f"exceeds the {workspace_limit / 2**30:.2f} GiB limit "
            f"(the smaller of {WORKSPACE_MULTIPLE:g}x the case's own {io_bytes / 2**30:.2f} GiB "
            f"and {WORKSPACE_CAP_BYTES / 2**30:.0f} GiB)"
        )
    del warm

    precall = prepare_invocation(
        torch, case, derive_seed(seed, "precall"), with_reference=False, phases=phases
    )
    torch_ops: set[str] = set()
    previous: PreparedInvocation | None = None
    fresh_inputs = 0
    fresh_outputs = 0
    seconds: list[float] = []
    host_walls: list[float] = []
    event_times: list[float] = []
    for trial in range(case_trials):
        _poison_output(precall)
        torch_ops.update(
            check_torch_ops(
                _recorded_invoke(clock, entrypoint, precall), f"case {case_id} precall call"
            )
        )
        clock.synchronize()

        prepared = prepare_invocation(
            torch,
            case,
            derive_seed(seed, f"trial-{trial}"),
            with_reference=trial == check_trial and not perf_only,
            tensors=real_tensors,
            phases=phases,
        )
        reserved_inputs = set(precall.input_ptrs)
        reserved_outputs = {precall.output_ptr}
        if previous is not None:
            reserved_inputs.update(previous.input_ptrs)
            reserved_outputs.add(previous.output_ptr)
        if prepared.output_ptr in reserved_outputs:
            raise RuntimeError(f"case {case_id}: fresh-output pointer protocol was not satisfied")
        if (
            len(set(prepared.input_ptrs)) != len(prepared.input_ptrs)
            or reserved_inputs.intersection(prepared.input_ptrs)
        ):
            raise RuntimeError(f"case {case_id}: fresh-input pointer protocol was not satisfied")
        fresh_inputs += 1
        fresh_outputs += 1

        _poison_output(prepared)
        signature = input_signature(torch, prepared)
        elapsed, host_wall, event = _measure_once(clock, entrypoint, prepared)
        if input_signature(torch, prepared) != signature:
            raise RuntimeError(f"case {case_id}: candidate modified its inputs")
        if int(prepared.out.data_ptr()) != prepared.output_ptr:
            raise RuntimeError(f"case {case_id}: output tensor was replaced")
        if trial == check_trial:
            _merge_metrics(
                metrics, verify(prepared, f"case {case_id} trial {trial}", real_layer=is_real)
            )
        elif not output_is_finite(torch, prepared.out):
            raise RuntimeError(
                f"case {case_id} trial {trial}: output is incomplete or contains NaN or Inf"
            )
        seconds.append(elapsed)
        host_walls.append(host_wall)
        event_times.append(event)
        previous = prepared
        del prepared

    ordered = sorted(seconds)
    median = ordered[len(ordered) // 2]
    heads, headdim, dstate, groups = _case_geometry(case)
    batch = int(case["batch"])
    seqlen = int(case["seqlen"])
    phases["warmup"] = warmup_wall
    result = {
        "case_id": case_id,
        "family": str(case["family"]),
        "geometry": str(case["geometry"]),
        "tier": str(case["tier"]),
        "scored": True,
        "batch": batch,
        "seqlen": seqlen,
        "heads": heads,
        "headdim": headdim,
        "dstate": dstate,
        "groups": groups,
        "regime": str(case.get("regime", "mixed")),
        "trials": case_trials,
        "checked_trial": check_trial,
        "trial_seconds": seconds,
        "median_seconds": median,
        "min_seconds": ordered[0],
        "max_seconds": ordered[-1],
        "host_wall_seconds": host_walls,
        "event_seconds": event_times,
        "warmup_wall_seconds": warmup_wall,
        "workspace_bytes": workspace_bytes,
        "io_bytes": int(io_bytes),
        "fresh_input_ok": fresh_inputs == case_trials,
        "fresh_output_ok": fresh_outputs == case_trials,
        "tilelang_only_ok": True,
        "torch_ops": sorted(torch_ops),
        "effective_flops": effective_flops(batch, seqlen, heads, headdim, dstate),
        "tflops": effective_tflops(batch, seqlen, heads, headdim, dstate, median),
        "phase_seconds": dict(phases),
        "perf_only": bool(perf_only),
        **{key: float(value) for key, value in metrics.items()},
    }
    del previous, precall
    torch.cuda.empty_cache()
    return result


def _merge_metrics(into: dict[str, float], new: Mapping[str, float]) -> None:
    """Keep the worst value seen for each metric across a case's invocations."""
    for key, value in new.items():
        current = into.get(key)
        if current is None or value > current:
            into[key] = value


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
def aggregate(
    results: Sequence[Mapping[str, Any]], *, require_full_suite: bool = True
) -> dict[str, Any]:
    """Family geometric means, then their equal-weight geometric mean.

    ``require_full_suite`` is always on for the full public suite: a score computed
    from part of it is not this task's score.  The public runner turns it off,
    and its aggregate covers whichever families the selected cases reach.
    """
    scored = [row for row in results if row.get("scored")]
    if not scored:
        raise RuntimeError("no scored case produced a result")
    per_family: dict[str, list[float]] = {}
    per_geometry: dict[str, list[float]] = {}
    for row in scored:
        per_family.setdefault(str(row["family"]), []).append(float(row["tflops"]))
        per_geometry.setdefault(str(row["geometry"]), []).append(float(row["tflops"]))
    unknown = sorted(set(per_family) - set(SCORE_FAMILIES))
    if unknown:
        raise RuntimeError(f"scored rows name unknown families {unknown}")
    if require_full_suite:
        counts = {name: len(values) for name, values in per_family.items()}
        if counts != FINAL_FAMILY_COUNTS:
            raise RuntimeError(f"scored cases do not match the full public suite's family counts: {counts}")
    family_means = {
        name: geometric_mean(per_family[name]) for name in SCORE_FAMILIES if name in per_family
    }
    return {
        "family_tflops": family_means,
        "geometry_tflops": {
            name: geometric_mean(values) for name, values in sorted(per_geometry.items())
        },
        "score_tflops": weighted_family_score(family_means),
        "scored_cases": len(scored),
    }


def weighted_family_score(family_means: Mapping[str, float]) -> float:
    """exp(sum_f w_f * log m_f / sum_f w_f) over the families present."""
    weights = {name: FAMILY_WEIGHTS[name] for name in family_means}
    total = math.fsum(weights.values())
    return math.exp(
        math.fsum(
            weight * math.log(finite_positive(family_means[name], "family mean"))
            for name, weight in weights.items()
        )
        / total
    )


# --------------------------------------------------------------------------
# Suites
# --------------------------------------------------------------------------
MANIFEST_SCHEMA_VERSION = 3
CASE_KEYS = frozenset(
    {"case_id", "geometry", "family", "tier", "batch", "seqlen", "regime", "seed", "check_trial", "layer"}
)
VALIDATED_CASE_EXTRAS = frozenset({"d_mode", "bc_mode", "scored"})
LAYER_NAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-")


def validate_case(raw: Mapping[str, Any], *, scored: bool) -> dict[str, Any]:
    """Check one case against the support domain and the workload-family rules."""
    geometry_name = str(raw["geometry"])
    if geometry_name not in GEOMETRIES:
        raise ValueError(f"unknown geometry {geometry_name!r}")
    geometry = GEOMETRIES[geometry_name]
    batch = int(raw["batch"])
    seqlen = int(raw["seqlen"])
    if not shape_in_domain(geometry, batch, seqlen):
        raise ValueError(f"({geometry_name}, {batch}, {seqlen}) is outside the support domain")
    regime = str(raw.get("regime", "mixed"))
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}")
    d_mode = str(raw.get("d_mode", "random"))
    if d_mode not in ("random", "zero"):
        raise ValueError(f"unknown d_mode {d_mode!r}")
    bc_mode = str(raw.get("bc_mode", "random"))
    if bc_mode not in ("random", "zero"):
        raise ValueError(f"unknown bc_mode {bc_mode!r}")
    seed = int(raw["seed"])
    if not 0 <= seed <= 0x7FFF_FFFF_FFFF_FFFF:
        raise ValueError("seed must be a non-negative 63-bit integer")
    case: dict[str, Any] = {
        "case_id": str(raw["case_id"]),
        "geometry": geometry_name,
        "batch": batch,
        "seqlen": seqlen,
        "regime": regime,
        "d_mode": d_mode,
        "bc_mode": bc_mode,
        "scored": bool(scored),
        "seed": seed,
    }
    if not scored:
        case.update({"family": "edge", "tier": "edge", "check_trial": 0, "layer": None})
        return case

    family = str(raw["family"])
    tier = str(raw["tier"])
    layer = raw.get("layer")
    check_trial = int(raw["check_trial"])
    if family not in SCORE_FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    if regime != "mixed" or d_mode != "random" or bc_mode != "random":
        raise ValueError("scored cases use the mixed decay regime with random D, B and C")
    if family == "batch" and batch < 2:
        raise ValueError("a batch case needs batch >= 2")
    if family in ("serving", "long", "real") and batch != 1:
        raise ValueError(f"a {family} case runs at batch 1")
    if family == "long" and seqlen < MIN_LONG_LENGTH:
        raise ValueError(f"a long case needs L >= {MIN_LONG_LENGTH}")
    if (family == "real") != (tier == "real"):
        raise ValueError("the real tier and the real family go together")
    if family == "real":
        if (
            not isinstance(layer, str)
            or not layer
            or not set(layer) <= LAYER_NAME_CHARS
            or check_trial != 0
        ):
            raise ValueError("a real case names its captured layer and checks its one trial")
    elif layer is not None:
        raise ValueError("only a real case names a captured layer")
    if check_trial < 0:
        raise ValueError("check_trial must be non-negative")
    case.update({"family": family, "tier": tier, "check_trial": check_trial, "layer": layer})
    return case


# Correctness-only fixtures, never timed or scored: lengths shorter than one
# chunk and straddling chunk boundaries, ragged long tails, slow and fast decay,
# dt pinned to one, D or B removed, a full batch and one head per group.
EDGE_FIXTURES: tuple[dict[str, Any], ...] = (
    {"case_id": "public-edge-len1", "geometry": "E1", "batch": 1, "seqlen": 1, "seed": 0xA001},
    {"case_id": "public-edge-len2", "geometry": "E3", "batch": 2, "seqlen": 2, "seed": 0xA002},
    {"case_id": "public-edge-len31", "geometry": "E2", "batch": 1, "seqlen": 31, "seed": 0xA003},
    {"case_id": "public-edge-len127", "geometry": "E1", "batch": 2, "seqlen": 127, "seed": 0xA004},
    {"case_id": "public-edge-len128", "geometry": "E3", "batch": 1, "seqlen": 128, "seed": 0xA005},
    {"case_id": "public-edge-len191", "geometry": "E2", "batch": 2, "seqlen": 191, "seed": 0xA006},
    {"case_id": "public-edge-len255", "geometry": "E1", "batch": 1, "seqlen": 255, "seed": 0xA007},
    {"case_id": "public-edge-len385", "geometry": "E2", "batch": 1, "seqlen": 385, "seed": 0xA008},
    {"case_id": "public-edge-len769", "geometry": "E5", "batch": 1, "seqlen": 769, "seed": 0xA009},
    {"case_id": "public-edge-slow-decay", "geometry": "E3", "batch": 2, "seqlen": 1500, "regime": "slow_decay", "seed": 0xA00A},
    {"case_id": "public-edge-slow-decay-g5", "geometry": "E6", "batch": 1, "seqlen": 2100, "regime": "slow_decay", "seed": 0xA00B},
    {"case_id": "public-edge-slow-decay-g4", "geometry": "E2", "batch": 1, "seqlen": 4000, "regime": "slow_decay", "seed": 0xA00C},
    {"case_id": "public-edge-fast-decay", "geometry": "E5", "batch": 2, "seqlen": 1000, "regime": "fast_decay", "seed": 0xA00D},
    {"case_id": "public-edge-unit-dt", "geometry": "E1", "batch": 1, "seqlen": 700, "regime": "unit_dt", "seed": 0xA00E},
    {"case_id": "public-edge-no-residual", "geometry": "E3", "batch": 1, "seqlen": 950, "d_mode": "zero", "seed": 0xA00F},
    {"case_id": "public-edge-residual-only", "geometry": "E6", "batch": 1, "seqlen": 1200, "bc_mode": "zero", "seed": 0xA010},
    {"case_id": "public-edge-full-batch", "geometry": "E1", "batch": 16, "seqlen": 300, "seed": 0xA011},
    {"case_id": "public-edge-one-head-per-group", "geometry": "E4", "batch": 1, "seqlen": 333, "seed": 0xA012},
    {"case_id": "public-edge-long-tail-g3", "geometry": "E3", "batch": 1, "seqlen": 7777, "seed": 0xA013},
    {"case_id": "public-edge-long-tail-g4", "geometry": "E2", "batch": 1, "seqlen": 13001, "seed": 0xA014},
    {"case_id": "public-edge-long-tail-g5", "geometry": "E6", "batch": 1, "seqlen": 24601, "seed": 0xA015},
)


def edge_fixtures() -> list[dict[str, Any]]:
    return [validate_case(raw, scored=False) for raw in EDGE_FIXTURES]


def validate_manifest_cases(
    raw_cases: Sequence[Mapping[str, Any]], *, require_full_suite: bool = True
) -> list[dict[str, Any]]:
    cases = []
    for raw in raw_cases:
        # A manifest row has exactly CASE_KEYS; an already-validated case also
        # carries the fields validate_case fills in.
        extra = set(raw) - CASE_KEYS
        if not CASE_KEYS <= set(raw) or not extra <= VALIDATED_CASE_EXTRAS:
            raise ValueError(f"case fields differ from schema v{MANIFEST_SCHEMA_VERSION}")
        cases.append(validate_case(raw, scored=True))
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("manifest contains duplicate case ids")
    if len({case["seed"] for case in cases}) != len(cases):
        raise ValueError("manifest contains duplicate seeds")
    synthetic = [
        (case["geometry"], case["batch"], case["seqlen"]) for case in cases if case["family"] != "real"
    ]
    if len(set(synthetic)) != len(synthetic):
        raise ValueError("manifest repeats a synthetic shape")
    layers = [case["layer"] for case in cases if case["family"] == "real"]
    if len(set(layers)) != len(layers):
        raise ValueError("manifest repeats a captured layer")
    if require_full_suite:
        counts: dict[str, int] = {}
        for case in cases:
            counts[case["family"]] = counts.get(case["family"], 0) + 1
        if counts != FINAL_FAMILY_COUNTS:
            raise ValueError(f"manifest family counts differ from the full public suite: {counts}")
    return cases


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("manifest schema version differs")
    suite_id = str(payload["suite_id"])
    return {"suite_id": suite_id, "cases": validate_manifest_cases(payload["cases"])}


# --------------------------------------------------------------------------
# Captured layers
# --------------------------------------------------------------------------
# A real case reads one layer captured from a real model on real text.  Every file is read and checked against the
# capture manifest before the candidate module is imported; the tensors stay
# on the host until their case runs.
REAL_LAYER_MANIFEST = "real_layers.manifest.json"
REAL_LAYER_TENSORS = ("x", "dt", "A", "B", "C", "D")


def load_real_layers(
    torch: Any, root: Path, cases: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """The captured layers the real cases name, checked against the capture manifest."""
    import io  # noqa: PLC0415

    wanted = [case for case in cases if case.get("family") == "real"]
    if not wanted:
        return {}
    root = Path(root)
    manifest = json.loads((root / REAL_LAYER_MANIFEST).read_text(encoding="utf-8"))
    files = manifest["files"]
    layers: dict[str, dict[str, Any]] = {}
    for case in wanted:
        name = str(case["layer"])
        entry = files.get(name)
        if not isinstance(entry, Mapping):
            raise RuntimeError(f"captured layer {name!r} is not in the capture manifest")
        raw = (root / f"{name}.pt").read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry.get("sha256"):
            raise RuntimeError(f"captured layer {name!r} differs from the capture manifest")
        tensors = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        if not isinstance(tensors, Mapping) or set(tensors) != set(REAL_LAYER_TENSORS):
            raise RuntimeError(f"captured layer {name!r} does not hold {REAL_LAYER_TENSORS}")
        heads, headdim, dstate, groups = _case_geometry(case)
        batch, seqlen = int(case["batch"]), int(case["seqlen"])
        expected = {
            "x": ((batch, seqlen, heads, headdim), torch.bfloat16),
            "dt": ((batch, seqlen, heads), torch.float32),
            "A": ((heads,), torch.float32),
            "B": ((batch, seqlen, groups, dstate), torch.bfloat16),
            "C": ((batch, seqlen, groups, dstate), torch.bfloat16),
            "D": ((heads,), torch.float32),
        }
        for key, (shape, dtype) in expected.items():
            value = tensors[key]
            if tuple(value.shape) != shape or value.dtype != dtype:
                raise RuntimeError(f"captured layer {name!r}: {key} is not {shape} {dtype}")
            if not bool(torch.isfinite(value.float()).all().item()):
                raise RuntimeError(f"captured layer {name!r}: {key} is not finite")
        if not bool((tensors["dt"] > 0).all().item()) or not bool((tensors["A"] < 0).all().item()):
            raise RuntimeError(f"captured layer {name!r} violates dt > 0 or A < 0")
        layers[name] = {key: tensors[key].contiguous() for key in REAL_LAYER_TENSORS}
    return layers


# --------------------------------------------------------------------------
# Suite driver
# --------------------------------------------------------------------------
def run_suite(
    torch: Any,
    submission_path: Path,
    cases: Sequence[Mapping[str, Any]],
    *,
    suite_id: str,
    trials: int,
    warmups: int,
    with_edge_fixtures: bool = True,
    require_full_suite: bool = True,
    real_layer_root: Path | None = None,
    perf_only: bool = False,
) -> dict[str, Any]:
    """Run one suite end to end and return a JSON-serialisable record.

    The clock and the operator recorder are captured, and the captured layers
    read and checked, before the candidate module is imported, and the hardware
    is inventoried before anything else, so a failure to meet the hardware
    contract is reported as such rather than as a slow kernel.
    """
    suite_started = time.perf_counter()
    _use_exact_fp32(torch)
    inventory = hardware_inventory(torch)
    clock = TrustedClock.capture(torch)
    cases = validate_manifest_cases(cases, require_full_suite=require_full_suite)
    real_layers = (
        load_real_layers(torch, Path(real_layer_root), cases) if real_layer_root is not None else {}
    )
    _module, entrypoint, digest = load_candidate(Path(submission_path))

    results: list[dict[str, Any]] = []
    edge_started = time.perf_counter()
    if with_edge_fixtures:
        for fixture in edge_fixtures():
            results.append(run_edge_case(torch, clock, entrypoint, fixture))
    edge_seconds = time.perf_counter() - edge_started
    for case in cases:
        results.append(
            run_scored_case(
                torch,
                clock,
                entrypoint,
                case,
                trials=trials,
                warmups=warmups,
                real_tensors=real_layers.get(str(case["layer"])) if case["layer"] else None,
                perf_only=perf_only,
            )
        )
    summary = aggregate(results, require_full_suite=require_full_suite)
    scored_rows = [row for row in results if row.get("scored")]
    phase_totals = {
        name: math.fsum(float(row["phase_seconds"].get(name, 0.0)) for row in scored_rows)
        for name in ("warmup", "generate", "reference", "compare")
    }
    phase_totals["timed_trials"] = math.fsum(math.fsum(row["trial_seconds"]) for row in scored_rows)
    phase_totals["edge_fixtures"] = edge_seconds
    phase_totals["suite"] = time.perf_counter() - suite_started
    return {
        "suite_id": suite_id,
        "submission_sha256": digest,
        "entrypoint": ENTRYPOINT,
        "trials": trials,
        "real_trials": REAL_TRIALS,
        "warmups": warmups,
        "hardware": inventory,
        "tolerances": {
            "rel_l2": OUT_REL_L2_TOL,
            "scaled_abs": OUT_SCALED_ABS_TOL,
            "rel": OUT_REL_TOL,
            "real_layer_rel_l2": REAL_LAYER_REL_L2_TOL,
        },
        "allowed_torch_ops": sorted(ALLOWED_TORCH_OPS),
        "edge_fixtures": [row for row in results if not row.get("scored")],
        "cases": scored_rows,
        "phase_seconds": phase_totals,
        **summary,
    }


# --------------------------------------------------------------------------
# Public suite
# --------------------------------------------------------------------------
PUBLIC_SUITE_ID = "tilelang-ssd-public-v3"
PUBLIC_SEED_BASE = 0x0066_0301
PUBLIC_LAYER_ROOT = Path(__file__).resolve().parent / "public_layer"

# (name, family, geometry, batch, L).  Serving lengths are real request lengths
# from public serving traces.
PUBLIC_CASES: tuple[tuple[str, str, str, int, int], ...] = (
    ("P1", "batch", "h48-p64-n256-g1", 8, 1280),
    ("P2", "batch", "h96-p80-n128-g4", 4, 7000),
    ("P3", "serving", "h40-p64-n64-g1", 1, 24000),
    ("P4", "batch", "h256-p48-n128-g16", 2, 5000),
    ("P5", "batch", "h48-p64-n128-g1", 8, 6144),
    ("P6", "batch", "h64-p128-n256-g4", 16, 1393),
    ("P7", "batch", "h24-p64-n128-g1", 2, 8222),
    ("P8", "serving", "h48-p64-n128-g1", 1, 6166),
    ("P9", "serving", "h48-p64-n128-g1", 1, 12535),
    ("P10", "serving", "h48-p64-n256-g1", 1, 3170),
    ("P11", "serving", "h48-p64-n256-g1", 1, 10128),
    ("P12", "serving", "h80-p64-n64-g1", 1, 3175),
    ("P13", "serving", "h80-p64-n64-g1", 1, 1632),
    ("P14", "serving", "h24-p64-n128-g1", 1, 6242),
    ("P15", "serving", "h24-p64-n128-g1", 1, 10318),
    ("P16", "serving", "h64-p128-n256-g4", 1, 5424),
    ("P17", "serving", "h64-p128-n256-g4", 1, 15919),
    ("P18", "serving", "h96-p80-n128-g4", 1, 6346),
    ("P19", "serving", "h96-p80-n128-g4", 1, 16890),
    ("P20", "long", "h48-p64-n128-g1", 1, 65500),
    ("P21", "long", "h48-p64-n256-g1", 1, 131000),
    ("P22", "long", "h24-p32-n64-g1", 1, 240000),
    ("P23", "long", "h64-p128-n256-g4", 1, 100003),
    ("P24", "real", "h48-p32-n128-g1", 1, 8123),
)
PUBLIC_LAYER = "public-real-layer16"


def public_suite() -> list[dict[str, Any]]:
    """The public cases with disclosed seeds; every trial's first one is compared."""
    return [
        validate_case(
            {
                "case_id": f"public-{name.lower()}",
                "geometry": geometry,
                "family": family,
                "tier": "real" if family == "real" else "sweep",
                "batch": batch,
                "seqlen": seqlen,
                "regime": "mixed",
                "seed": PUBLIC_SEED_BASE + index,
                "check_trial": 0,
                "layer": PUBLIC_LAYER if family == "real" else None,
            },
            scored=True,
        )
        for index, (name, family, geometry, batch, seqlen) in enumerate(PUBLIC_CASES)
    ]
