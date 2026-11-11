"""Seed kernel_v001: quantized attention in self-contained Triton."""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _k_mean_kernel(
    K, KM, L,
    stride_kb, stride_kl, stride_kh,
    stride_mb, stride_mh,
    HEAD_DIM: tl.constexpr, BLOCK_L: tl.constexpr,
):
    off_h = tl.program_id(0)
    off_b = tl.program_id(1)
    offs_l = tl.arange(0, BLOCK_L)
    offs_d = tl.arange(0, HEAD_DIM)
    base = K + off_b * stride_kb + off_h * stride_kh
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    for start in range(0, L, BLOCK_L):
        rows = start + offs_l
        x = tl.load(
            base + rows[:, None] * stride_kl + offs_d[None, :],
            mask=rows[:, None] < L, other=0.0,
        ).to(tl.float32)
        acc += tl.sum(x, 0)
    mean = acc / L.to(tl.float32)
    tl.store(KM + off_b * stride_mb + off_h * stride_mh + offs_d, mean)


@triton.jit
def _quant_q_per_block_kernel(
    Q, QI, QS, L,
    stride_qb, stride_ql, stride_qh,
    stride_ib, stride_il, stride_ih,
    stride_sb, stride_sh,
    scale_factor,
    HEAD_DIM: tl.constexpr, BLK: tl.constexpr,
):
    off_blk = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)
    offs_n = off_blk * BLK + tl.arange(0, BLK)
    offs_d = tl.arange(0, HEAD_DIM)
    row_mask = offs_n[:, None] < L
    x = tl.load(
        Q + off_b * stride_qb + off_h * stride_qh + offs_n[:, None] * stride_ql + offs_d[None, :],
        mask=row_mask, other=0.0,
    ).to(tl.float32)
    x = x * scale_factor
    scale = tl.max(tl.abs(x)) / 127.0 + 1e-7
    x_q = x / scale
    x_q += 0.5 * tl.where(x_q >= 0, 1.0, -1.0)
    tl.store(
        QI + off_b * stride_ib + off_h * stride_ih + offs_n[:, None] * stride_il + offs_d[None, :],
        x_q.to(tl.int8), mask=row_mask,
    )
    tl.store(QS + off_b * stride_sb + off_h * stride_sh + off_blk, scale)


@triton.jit
def _quant_k_per_block_kernel(
    K, KM, KI, KS, L,
    stride_kb, stride_kl, stride_kh,
    stride_mb, stride_mh,
    stride_ib, stride_il, stride_ih,
    stride_sb, stride_sh,
    HEAD_DIM: tl.constexpr, BLK: tl.constexpr,
):
    off_blk = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)
    offs_n = off_blk * BLK + tl.arange(0, BLK)
    offs_d = tl.arange(0, HEAD_DIM)
    row_mask = offs_n[:, None] < L
    x = tl.load(
        K + off_b * stride_kb + off_h * stride_kh + offs_n[:, None] * stride_kl + offs_d[None, :],
        mask=row_mask, other=0.0,
    ).to(tl.float32)
    km = tl.load(KM + off_b * stride_mb + off_h * stride_mh + offs_d)
    x = tl.where(row_mask, x - km[None, :], 0.0)
    scale = tl.max(tl.abs(x)) / 127.0 + 1e-7
    x_q = x / scale
    x_q += 0.5 * tl.where(x_q >= 0, 1.0, -1.0)
    tl.store(
        KI + off_b * stride_ib + off_h * stride_ih + offs_n[:, None] * stride_il + offs_d[None, :],
        x_q.to(tl.int8), mask=row_mask,
    )
    tl.store(KS + off_b * stride_sb + off_h * stride_sh + off_blk, scale)


@triton.jit
def _attn_inner(
    acc, l_i, m_i, q, q_scale, L,
    K_ptrs, K_scale_ptr, V_ptrs, stride_kl, stride_vl,
    start_m, offs_m, offs_n,
    lo, hi,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    DIAGONAL: tl.constexpr,
):
    K_ptrs += lo * stride_kl
    V_ptrs += lo * stride_vl
    K_scale_ptr += lo // BLOCK_N
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_mask = offs_n[None, :] < (L - start_n)
        k = tl.load(K_ptrs, mask=k_mask, other=0)
        k_scale = tl.load(K_scale_ptr)
        qk = tl.dot(q, k).to(tl.float32) * (q_scale * k_scale)
        mask = k_mask
        if DIAGONAL:
            mask = mask & (offs_m[:, None] >= (start_n + offs_n[None, :]))
        qk = qk + tl.where(mask, 0.0, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v = tl.load(V_ptrs, mask=offs_n[:, None] < (L - start_n), other=0.0).to(tl.float16)
        p = p.to(tl.float16)
        acc += tl.dot(p, v, out_dtype=tl.float16)
        m_i = m_ij
        K_ptrs += BLOCK_N * stride_kl
        K_scale_ptr += 1
        V_ptrs += BLOCK_N * stride_vl
    return acc, l_i, m_i


@triton.jit
def _attn_fwd_kernel(
    QI, KI, V, QS, KS, Out, L,
    stride_qb, stride_ql, stride_qh,
    stride_kb, stride_kl, stride_kh,
    stride_vb, stride_vl, stride_vh,
    stride_ob, stride_ol, stride_oh,
    stride_qsb, stride_qsh,
    stride_ksb, stride_ksh,
    NUM_KV_GROUPS: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_h = tl.program_id(1).to(tl.int64)
    off_b = tl.program_id(2).to(tl.int64)
    off_hk = off_h // NUM_KV_GROUPS

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    Q_ptrs = QI + off_b * stride_qb + off_h * stride_qh + offs_m[:, None] * stride_ql + offs_d[None, :]
    K_ptrs = KI + off_b * stride_kb + off_hk * stride_kh + offs_n[None, :] * stride_kl + offs_d[:, None]
    V_ptrs = V + off_b * stride_vb + off_hk * stride_vh + offs_n[:, None] * stride_vl + offs_d[None, :]
    O_ptrs = Out + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_ol + offs_d[None, :]
    q_scale = tl.load(QS + off_b * stride_qsb + off_h * stride_qsh + start_m)
    K_scale_ptr = KS + off_b * stride_ksb + off_hk * stride_ksh

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    q = tl.load(Q_ptrs, mask=offs_m[:, None] < L, other=0)
    if IS_CAUSAL:
        acc, l_i, m_i = _attn_inner(
            acc, l_i, m_i, q, q_scale, L,
            K_ptrs, K_scale_ptr, V_ptrs, stride_kl, stride_vl,
            start_m, offs_m, offs_n,
            0, start_m * BLOCK_M,
            BLOCK_M, BLOCK_N, HEAD_DIM, False,
        )
        acc, l_i, m_i = _attn_inner(
            acc, l_i, m_i, q, q_scale, L,
            K_ptrs, K_scale_ptr, V_ptrs, stride_kl, stride_vl,
            start_m, offs_m, offs_n,
            start_m * BLOCK_M, (start_m + 1) * BLOCK_M,
            BLOCK_M, BLOCK_N, HEAD_DIM, True,
        )
    else:
        acc, l_i, m_i = _attn_inner(
            acc, l_i, m_i, q, q_scale, L,
            K_ptrs, K_scale_ptr, V_ptrs, stride_kl, stride_vl,
            start_m, offs_m, offs_n,
            0, L,
            BLOCK_M, BLOCK_N, HEAD_DIM, False,
        )
    acc = acc / l_i[:, None]
    tl.store(O_ptrs, acc.to(Out.type.element_ty), mask=offs_m[:, None] < L)


def quantized_attention_bf16(q, k, v, is_causal, softmax_scale, out):
    B = q.shape[0]
    L = q.shape[1]
    H_Q = q.shape[2]
    D = q.shape[3]
    H_KV = k.shape[2]
    assert q.dtype == torch.bfloat16
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and out.is_contiguous()
    assert D == 64 or D == 128
    assert H_Q % H_KV == 0
    NUM_KV_GROUPS = H_Q // H_KV
    BLKQ = 128
    BLKK = 64
    NBLKQ = triton.cdiv(L, BLKQ)
    NBLKK = triton.cdiv(L, BLKK)

    km = torch.empty((B, H_KV, D), dtype=torch.float32, device=k.device)
    q_int8 = torch.empty((B, L, H_Q, D), dtype=torch.int8, device=q.device)
    k_int8 = torch.empty((B, L, H_KV, D), dtype=torch.int8, device=k.device)
    q_scale = torch.empty((B, H_Q, NBLKQ), dtype=torch.float32, device=q.device)
    k_scale = torch.empty((B, H_KV, NBLKK), dtype=torch.float32, device=k.device)

    _k_mean_kernel[(H_KV, B)](
        k, km, L,
        k.stride(0), k.stride(1), k.stride(2),
        km.stride(0), km.stride(1),
        HEAD_DIM=D, BLOCK_L=128, num_warps=8,
    )
    _quant_q_per_block_kernel[(NBLKQ, H_Q, B)](
        q, q_int8, q_scale, L,
        q.stride(0), q.stride(1), q.stride(2),
        q_int8.stride(0), q_int8.stride(1), q_int8.stride(2),
        q_scale.stride(0), q_scale.stride(1),
        float(softmax_scale) * 1.44269504,
        HEAD_DIM=D, BLK=BLKQ,
    )
    _quant_k_per_block_kernel[(NBLKK, H_KV, B)](
        k, km, k_int8, k_scale, L,
        k.stride(0), k.stride(1), k.stride(2),
        km.stride(0), km.stride(1),
        k_int8.stride(0), k_int8.stride(1), k_int8.stride(2),
        k_scale.stride(0), k_scale.stride(1),
        HEAD_DIM=D, BLK=BLKK,
    )
    if D == 64:
        NUM_WARPS = 4
        NUM_STAGES = 3
    else:
        NUM_WARPS = 8
        NUM_STAGES = 4
    if is_causal:
        CAUSAL = 1
    else:
        CAUSAL = 0
    _attn_fwd_kernel[(NBLKQ, H_Q, B)](
        q_int8, k_int8, v, q_scale, k_scale, out, L,
        q_int8.stride(0), q_int8.stride(1), q_int8.stride(2),
        k_int8.stride(0), k_int8.stride(1), k_int8.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        q_scale.stride(0), q_scale.stride(1),
        k_scale.stride(0), k_scale.stride(1),
        NUM_KV_GROUPS=NUM_KV_GROUPS,
        HEAD_DIM=D, BLOCK_M=BLKQ, BLOCK_N=BLKK,
        IS_CAUSAL=CAUSAL,
        num_warps=NUM_WARPS, num_stages=NUM_STAGES,
    )
