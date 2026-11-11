"""Oracle: a correct, deliberately straightforward Triton paged-GQA decode.

One program per (request, query head); BLOCK_M = 16 covers every S_q in 1..16;
per-lane logical-to-physical page translation with int64 addressing; online
softmax with a finite mask value.  A competent baseline, not a tuned entry:
it re-reads shared KV once per query head, so GQA packing, split-KV, and
page-aware scheduling all remain on the table.
"""

import torch
import triton
import triton.language as tl

NEG_INF = tl.constexpr(-1.0e30)


@triton.jit
def _paged_gqa_decode_kernel(
    q_ptr, k_ptr, v_ptr, pt_ptr, seqlens_ptr, out_ptr,
    scale,
    S_q, H_q, H_kv, PAGE_SIZE, TABLE_WIDTH,
    stride_qb, stride_qs, stride_qh,
    stride_ob, stride_os, stride_oh,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    h_q = tl.program_id(1)
    h_kv = h_q // (H_q // H_kv)

    L = tl.load(seqlens_ptr + b)

    rows = tl.arange(0, BLOCK_M)
    d = tl.arange(0, BLOCK_D)
    row_ok = rows < S_q

    q_ptrs = q_ptr + b * stride_qb + rows[:, None] * stride_qs \
        + h_q * stride_qh + d[None, :]
    q_tile = tl.load(q_ptrs, mask=row_ok[:, None], other=0.0)

    m_i = tl.full([BLOCK_M], NEG_INF, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    kv_row_stride = H_kv * BLOCK_D  # elements between consecutive cache slots
    limit = L - S_q + rows  # last visible logical key index per query row

    for start in range(0, L, BLOCK_N):
        j = start + tl.arange(0, BLOCK_N)
        j_ok = j < L
        page = tl.load(pt_ptr + b * TABLE_WIDTH + j // PAGE_SIZE,
                       mask=j_ok, other=0)
        lin = page.to(tl.int64) * PAGE_SIZE + (j % PAGE_SIZE).to(tl.int64)
        kv_off = lin[:, None] * kv_row_stride + h_kv * BLOCK_D + d[None, :]

        k_tile = tl.load(k_ptr + kv_off, mask=j_ok[:, None], other=0.0)
        qk = tl.dot(q_tile, tl.trans(k_tile)) * scale
        qk = tl.where(j[None, :] <= limit[:, None], qk, NEG_INF)

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_tile = tl.load(v_ptr + kv_off, mask=j_ok[:, None], other=0.0)
        acc += tl.dot(p.to(tl.bfloat16), v_tile)
        m_i = m_new

    o = acc / l_i[:, None]
    o_ptrs = out_ptr + b * stride_ob + rows[:, None] * stride_os \
        + h_q * stride_oh + d[None, :]
    tl.store(o_ptrs, o.to(tl.bfloat16), mask=row_ok[:, None])


def paged_gqa_decode_bf16(q, k_cache, v_cache, page_table, cache_seqlens,
                          softmax_scale: float, out) -> None:
    B, S_q, H_q, D = q.shape
    H_kv = k_cache.shape[2]
    page_size = k_cache.shape[1]
    table_width = page_table.shape[1]

    grid = (B, H_q)
    _paged_gqa_decode_kernel[grid](
        q, k_cache, v_cache, page_table, cache_seqlens, out,
        softmax_scale,
        S_q, H_q, H_kv, page_size, table_width,
        q.stride(0), q.stride(1), q.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_M=16, BLOCK_N=128, BLOCK_D=D,
        num_warps=4,
    )
    return None
