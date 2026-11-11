from __future__ import annotations

import triton
import triton.language as tl
import triton.runtime.errors


NEG_INF = tl.constexpr(float("-inf"))
POS_INF = tl.constexpr(float("inf"))


# s_q is grid axis 0 (up to 2**31 - 1 programs; axes 1 and 2 stop at 65535) and
# every row offset is computed from an int64 program id, so q and out may hold
# more than 2**31 elements (the v2 contract allows s_q up to 70000).
@triton.jit
def _sparse_mla_prefill_kernel(
    q_ptr,
    kv_ptr,
    idx_ptr,
    sink_ptr,
    len_ptr,
    out_ptr,
    max_logits_ptr,
    lse_ptr,
    sm_scale,
    s_kv,
    topk,
    H_Q: tl.constexpr,
    D_V: tl.constexpr,
    D_ROPE: tl.constexpr,
    HAS_SINK: tl.constexpr,
    HAS_LEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    D_QK: tl.constexpr = D_V + D_ROPE
    token = tl.program_id(axis=0).to(tl.int64)
    head_block = tl.program_id(axis=1)
    h_offsets = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    v_offsets = tl.arange(0, D_V)

    q_base = q_ptr + token * (H_Q * D_QK) + h_offsets[:, None] * D_QK
    q_nope = tl.load(q_base + v_offsets[None, :])
    if D_ROPE > 0:
        rope_offsets = tl.arange(0, D_ROPE)
        q_rope = tl.load(q_base + D_V + rope_offsets[None, :])

    # Slots at or beyond topk_length are invalid, exactly like an out-of-range
    # index; the kernel never reads them.
    length = topk
    if HAS_LEN:
        length = tl.minimum(tl.load(len_ptr + token), topk)

    running_max = tl.full([BLOCK_H], NEG_INF, tl.float32)
    running_sum = tl.zeros([BLOCK_H], tl.float32)
    accumulator = tl.zeros([BLOCK_H, D_V], tl.float32)

    index_row = idx_ptr + token * topk
    for start in range(0, topk, BLOCK_N):
        slots = start + tl.arange(0, BLOCK_N)
        in_range = slots < length
        indices = tl.load(index_row + slots, mask=in_range, other=-1)
        valid = in_range & (indices >= 0) & (indices < s_kv)
        kv_row = kv_ptr + tl.where(valid, indices, 0).to(tl.int64) * D_QK
        k_nope = tl.load(kv_row[:, None] + v_offsets[None, :], mask=valid[:, None], other=0.0)
        logits = tl.dot(q_nope, tl.trans(k_nope), out_dtype=tl.float32)
        if D_ROPE > 0:
            k_rope = tl.load(
                kv_row[:, None] + D_V + rope_offsets[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            logits += tl.dot(q_rope, tl.trans(k_rope), out_dtype=tl.float32)
        logits = logits * sm_scale

        masked = tl.where(valid[None, :], logits, NEG_INF)
        next_max = tl.maximum(running_max, tl.max(masked, axis=1))
        # A block whose maximum is still -inf has produced no valid key yet;
        # the shift is then irrelevant because every weight below is zero.
        shift = tl.where(next_max == NEG_INF, 0.0, next_max)
        rescale = tl.where(running_max == NEG_INF, 0.0, tl.exp(running_max - shift))
        weights = tl.where(valid[None, :], tl.exp(logits - shift[:, None]), 0.0)
        running_sum = running_sum * rescale + tl.sum(weights, axis=1)
        accumulator = accumulator * rescale[:, None] + tl.dot(
            weights.to(k_nope.dtype), k_nope, out_dtype=tl.float32
        )
        running_max = next_max

    lonely = running_max == NEG_INF
    raw_lse = tl.where(lonely, NEG_INF, running_max + tl.log(running_sum))

    normaliser = raw_lse
    if HAS_SINK:
        sink = tl.load(sink_ptr + h_offsets)
        peak = tl.maximum(raw_lse, sink)
        peak_shift = tl.where(peak == NEG_INF, 0.0, peak)
        merged = peak + tl.log(tl.exp(raw_lse - peak_shift) + tl.exp(sink - peak_shift))
        normaliser = tl.where(
            peak == POS_INF, POS_INF, tl.where(peak == NEG_INF, NEG_INF, merged)
        )
    normaliser = tl.where(normaliser == NEG_INF, POS_INF, normaliser)
    scale = tl.exp(running_max - normaliser)

    out_offsets = token * (H_Q * D_V) + h_offsets[:, None] * D_V + v_offsets[None, :]
    tl.store(out_ptr + out_offsets, (accumulator * scale[:, None]).to(out_ptr.dtype.element_ty))
    tl.store(max_logits_ptr + token * H_Q + h_offsets, running_max)
    tl.store(lse_ptr + token * H_Q + h_offsets, tl.where(lonely, POS_INF, raw_lse))


# Tile plans ``(BLOCK_H, BLOCK_N, num_stages)`` in preference order.  The
# first entry is the original fixed rule and is what an H100 (227 KiB opt-in
# shared memory) runs.  Those tiles need about 152 KiB, which sm_80/sm_86
# parts do not have (A100 164 KiB, RTX A6000 99 KiB), so the launch failed
# with ``OutOfResources`` before the kernel could run.  The footprint is
# dominated by the ``[BLOCK_H, 512]`` accumulator and its layout conversions,
# so halving BLOCK_H is what actually helps; shrinking BLOCK_N alone does
# not.  Instead of guessing the compiler's footprint, try the plans in order
# and keep the first one the device can launch; the kernel body is unchanged
# and the choice is cached per (device, H_Q, d_qk, d_v).
_TILE_PLANS = (
    (64, 64, 2),
    (64, 64, 1),
    (32, 64, 2),
    (32, 64, 1),
    (32, 32, 1),
    (16, 64, 1),
    (16, 32, 1),
)
_TILE_PLAN_CACHE: dict[tuple[int, int, int, int], tuple[int, int, int]] = {}


def sparse_mla_prefill(
    q, kv, indices, sm_scale, attn_sink, topk_length, out, max_logits, lse
) -> None:
    s_q, h_q, d_qk = (int(value) for value in q.shape)
    s_kv = int(kv.shape[0])
    topk = int(indices.shape[-1])
    d_v = int(out.shape[-1])
    cache_key = (int(q.device.index or 0), h_q, d_qk, d_v)
    cached = _TILE_PLAN_CACHE.get(cache_key)
    plans = (cached,) if cached is not None else _TILE_PLANS
    last_error = None
    for block_h, block_n, num_stages in plans:
        if h_q % block_h != 0:
            continue
        try:
            _sparse_mla_prefill_kernel[(s_q, h_q // block_h)](
                q,
                kv,
                indices,
                attn_sink,
                topk_length,
                out,
                max_logits,
                lse,
                float(sm_scale),
                s_kv,
                topk,
                H_Q=h_q,
                D_V=d_v,
                D_ROPE=d_qk - d_v,
                HAS_SINK=attn_sink is not None,
                HAS_LEN=topk_length is not None,
                BLOCK_H=block_h,
                BLOCK_N=block_n,
                num_warps=8,
                num_stages=num_stages,
            )
        except triton.runtime.errors.OutOfResources as exc:
            last_error = exc
            continue
        _TILE_PLAN_CACHE[cache_key] = (block_h, block_n, num_stages)
        return
    if last_error is None:
        raise ValueError(f"no tile plan divides H_Q={h_q}")
    raise last_error
