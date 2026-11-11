"""Paper control: upstream SageAttention-1 Triton path (sageattn_qk_int8_pv_fp16_triton).

Source: thu-ml/SageAttention @ d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5 (Apache-2.0)
  sageattention/triton/quant_per_block.py          (quant_per_block_int8_kernel, verbatim)
  sageattention/triton/attn_qk_int8_per_block.py   (_attn_fwd_inner/_attn_fwd, verbatim, renamed *_nc)
  sageattention/triton/attn_qk_int8_per_block_causal.py (_attn_fwd_inner/_attn_fwd, verbatim, renamed *_c)
  sageattention/core.py::sageattn_qk_int8_pv_fp16_triton (host flow)

This is the ONLY attention path the upstream repository ships in Triton.  Upstream's best
Hopper path (sageattn_qk_int8_pv_fp8_cuda_sm90, SageAttention2/2++) is a compiled CUDA
extension and cannot be admitted by this task's source policy; it is timed by the separate
harness in ../sm90_harness/.

Glue-only changes required by the task ABI / host policy (kernel math is upstream's):
  1. km = k.mean(dim=1) (host torch, forbidden) -> _glue_k_colsum_kernel (parallel column sums
     with atomics into a zeroed FP32 buffer); the division by L happens in the K quant kernel.
  2. k - km (host torch, forbidden) -> fused into a copy of upstream's quant kernel
     (_glue_quant_k_smooth_kernel: identical body plus the subtraction).
  3. v.to(torch.float16) (host torch copy, forbidden) -> _glue_cast_v_fp16_kernel (one pass,
     exactly the same data flow as upstream's host cast).
  4. Output written straight into the caller's BF16 `out` (upstream allocates o with the input
     dtype = BF16 and stores acc.to(Out.type.element_ty) -- same rounding).
  5. +1e-7 on the absmax scales (upstream divides by 0 on an all-zero block; seed does the same).
  6. RETURN_LSE=False with a 1-element device dummy instead of a CPU tensor.
Tile sizes, warps and stages are upstream's (BLOCK_M=128, BLOCK_N=64, BLKQ=128, BLKK=64,
num_warps 4/8 for D=64/128, num_stages 3/4 non-causal and 4 causal).
"""

import math

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------- glue kernels (ABI/policy)
@triton.jit
def _glue_k_colsum_kernel(K, KSUM, L,
                          stride_kb, stride_kl, stride_kh,
                          stride_sb, stride_sh,
                          HEAD_DIM: tl.constexpr, BLOCK_L: tl.constexpr):
    off_blk = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)
    rows = off_blk * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_d = tl.arange(0, HEAD_DIM)
    x = tl.load(K + off_b * stride_kb + off_h * stride_kh + rows[:, None] * stride_kl + offs_d[None, :],
                mask=rows[:, None] < L, other=0.0).to(tl.float32)
    tl.atomic_add(KSUM + off_b * stride_sb + off_h * stride_sh + offs_d, tl.sum(x, 0))


@triton.jit
def _glue_cast_v_fp16_kernel(V, VH, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    tl.store(VH + offs, tl.load(V + offs, mask=m).to(tl.float16), mask=m)


# ---------------------------------------------------------------- upstream quant_per_block.py
@triton.jit
def quant_per_block_int8_kernel(Input, Output, Scale, L,
                                stride_iz, stride_ih, stride_in,
                                stride_oz, stride_oh, stride_on,
                                stride_sz, stride_sh,
                                sm_scale,
                                C: tl.constexpr, BLK: tl.constexpr):
    off_blk = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    offs_n = off_blk * BLK + tl.arange(0, BLK)
    offs_k = tl.arange(0, C)

    input_ptrs = Input + off_b * stride_iz + off_h * stride_ih + offs_n[:, None] * stride_in + offs_k[None, :]
    output_ptrs = Output + off_b * stride_oz + off_h * stride_oh + offs_n[:, None] * stride_on + offs_k[None, :]
    scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_blk

    x = tl.load(input_ptrs, mask=offs_n[:, None] < L)
    x = x.to(tl.float32)
    x *= sm_scale
    scale = tl.max(tl.abs(x)) / 127. + 1e-7
    x_int8 = x / scale
    x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
    x_int8 = x_int8.to(tl.int8)
    tl.store(output_ptrs, x_int8, mask=offs_n[:, None] < L)
    tl.store(scale_ptrs, scale)


@triton.jit
def _glue_quant_k_smooth_kernel(Input, KSUM, Output, Scale, L,
                                stride_iz, stride_ih, stride_in,
                                stride_mz, stride_mh,
                                stride_oz, stride_oh, stride_on,
                                stride_sz, stride_sh,
                                C: tl.constexpr, BLK: tl.constexpr):
    # upstream quant_per_block_int8_kernel (sm_scale=1.0) applied to k - km, km = mean over tokens
    off_blk = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    offs_n = off_blk * BLK + tl.arange(0, BLK)
    offs_k = tl.arange(0, C)

    input_ptrs = Input + off_b * stride_iz + off_h * stride_ih + offs_n[:, None] * stride_in + offs_k[None, :]
    output_ptrs = Output + off_b * stride_oz + off_h * stride_oh + offs_n[:, None] * stride_on + offs_k[None, :]
    scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_blk

    km = tl.load(KSUM + off_b * stride_mz + off_h * stride_mh + offs_k) / L.to(tl.float32)
    x = tl.load(input_ptrs, mask=offs_n[:, None] < L)
    x = x.to(tl.float32)
    x = tl.where(offs_n[:, None] < L, x - km[None, :], 0.0)
    scale = tl.max(tl.abs(x)) / 127. + 1e-7
    x_int8 = x / scale
    x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
    x_int8 = x_int8.to(tl.int8)
    tl.store(output_ptrs, x_int8, mask=offs_n[:, None] < L)
    tl.store(scale_ptrs, scale)


# ---------------------------------------------------------------- upstream attn_qk_int8_per_block.py
@triton.jit
def _attn_fwd_inner_nc(acc, l_i, m_i, q, q_scale, qo_len, kv_len,
                       K_ptrs, K_scale_ptr, V_ptrs, stride_kn, stride_vn,
                       start_m, mask_ptrs, stride_maskn,
                       BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
                       STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
                       ):
    lo, hi = 0, kv_len
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_block = None
        skip = False
        if mask_ptrs is not None:
            if mask_ptrs.dtype.element_ty == tl.int1:
                mask_block = tl.load(mask_ptrs + start_n * stride_maskn, mask=(offs_m[:, None] < qo_len) & (offs_n[None, :] < kv_len - start_n), other=False)
                if tl.max(mask_block) == 0:
                    skip = True
            else:
                mask_block = tl.load(mask_ptrs + start_n * stride_maskn, mask=(offs_m[:, None] < qo_len) & (offs_n[None, :] < kv_len - start_n), other=-1.0e6)
        if not skip:
            k_mask = offs_n[None, :] < (kv_len - start_n)
            k = tl.load(K_ptrs, mask=k_mask)
            k_scale = tl.load(K_scale_ptr)

            qk = tl.dot(q, k).to(tl.float32) * (q_scale * k_scale)

            if mask_block is not None:
                if mask_block.dtype == tl.int1:
                    qk = qk + tl.where(mask_block, 0, -1.0e6)
                else:
                    qk = qk + mask_block
            else:
                qk += tl.where(k_mask, 0, -1.0e6)

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]
            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)

            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij

            acc = acc * alpha[:, None]

            v = tl.load(V_ptrs, mask=offs_n[:, None] < (kv_len - start_n))
            p = p.to(tl.float16)

            acc += tl.dot(p, v, out_dtype=tl.float16)
            m_i = m_ij
        K_ptrs += BLOCK_N * stride_kn
        K_scale_ptr += 1
        V_ptrs += BLOCK_N * stride_vn
    return acc, l_i, m_i


@triton.jit
def _attn_fwd_nc(Q, K, V, Q_scale, K_scale, Out, mask, Lse,
                 stride_qz, stride_qh, stride_qn,
                 stride_kz, stride_kh, stride_kn,
                 stride_vz, stride_vh, stride_vn,
                 stride_oz, stride_oh, stride_on,
                 stride_maskz, stride_maskh, stride_maskm, stride_maskn,
                 qo_len, kv_len, H: tl.constexpr, num_kv_groups: tl.constexpr,
                 HEAD_DIM: tl.constexpr,
                 BLOCK_M: tl.constexpr,
                 BLOCK_N: tl.constexpr,
                 STAGE: tl.constexpr,
                 RETURN_LSE: tl.constexpr,
                 ):
    start_m = tl.program_id(0)

    off_z = tl.program_id(2).to(tl.int64)
    off_h = tl.program_id(1).to(tl.int64)

    q_scale_offset = (off_z * H + off_h) * tl.cdiv(qo_len, BLOCK_M)
    k_scale_offset = (off_z * (H // num_kv_groups) + off_h // num_kv_groups) * tl.cdiv(kv_len, BLOCK_N)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    Q_ptrs = Q + (off_z * stride_qz + off_h * stride_qh) + offs_m[:, None] * stride_qn + offs_k[None, :]
    Q_scale_ptr = Q_scale + q_scale_offset + start_m
    K_ptrs = K + (off_z * stride_kz + (off_h // num_kv_groups) * stride_kh) + offs_n[None, :] * stride_kn + offs_k[:, None]
    K_scale_ptr = K_scale + k_scale_offset
    V_ptrs = V + (off_z * stride_vz + (off_h // num_kv_groups) * stride_vh) + offs_n[:, None] * stride_vn + offs_k[None, :]
    O_block_ptr = Out + (off_z * stride_oz + off_h * stride_oh) + offs_m[:, None] * stride_on + offs_k[None, :]
    if mask is None:
        mask_ptrs = None
    else:
        mask_ptrs = mask + (off_z * stride_maskz + off_h * stride_maskh) + offs_m[:, None] * stride_maskm + offs_n[None, :] * stride_maskn

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    q = tl.load(Q_ptrs, mask=offs_m[:, None] < qo_len)
    q_scale = tl.load(Q_scale_ptr)
    acc, l_i, m_i = _attn_fwd_inner_nc(acc, l_i, m_i, q, q_scale, qo_len, kv_len, K_ptrs, K_scale_ptr, V_ptrs, stride_kn, stride_vn,
                                       start_m, mask_ptrs, stride_maskn,
                                       BLOCK_M, HEAD_DIM, BLOCK_N,
                                       4 - STAGE, offs_m, offs_n
                                       )
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask=(offs_m[:, None] < qo_len))

    if RETURN_LSE:
        lse_ptrs = Lse + (off_z * qo_len * H + off_h * qo_len) + offs_m
        l_i = tl.log2(l_i) + m_i
        tl.store(lse_ptrs, l_i, mask=(offs_m < qo_len))


# ---------------------------------------------------------------- upstream attn_qk_int8_per_block_causal.py
@triton.jit
def _attn_fwd_inner_c(acc, l_i, m_i, q, q_scale, kv_len,
                      K_ptrs, K_scale_ptr, V_ptrs, stride_kn, stride_vn,
                      start_m,
                      BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
                      STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
                      ):
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
        K_scale_ptr += lo // BLOCK_N
        K_ptrs += stride_kn * lo
        V_ptrs += stride_vn * lo
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_mask = offs_n[None, :] < (kv_len - start_n)
        k = tl.load(K_ptrs, mask=k_mask)
        k_scale = tl.load(K_scale_ptr)
        qk = tl.dot(q, k).to(tl.float32) * (q_scale * k_scale)

        mask = k_mask
        if STAGE == 2:
            mask &= offs_m[:, None] >= (start_n + offs_n[None, :])
        qk += tl.where(mask, 0, float('-inf'))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        acc = acc * alpha[:, None]

        v = tl.load(V_ptrs, mask=offs_n[:, None] < (kv_len - start_n))
        p = p.to(tl.float16)

        acc += tl.dot(p, v, out_dtype=tl.float16)
        m_i = m_ij
        K_ptrs += BLOCK_N * stride_kn
        K_scale_ptr += 1
        V_ptrs += BLOCK_N * stride_vn
    return acc, l_i, m_i


@triton.jit
def _attn_fwd_c(Q, K, V, Q_scale, K_scale, Out, Lse,
                stride_qz, stride_qh, stride_qn,
                stride_kz, stride_kh, stride_kn,
                stride_vz, stride_vh, stride_vn,
                stride_oz, stride_oh, stride_on,
                qo_len, kv_len, H: tl.constexpr, num_kv_groups: tl.constexpr,
                HEAD_DIM: tl.constexpr,
                BLOCK_M: tl.constexpr,
                BLOCK_N: tl.constexpr,
                STAGE: tl.constexpr,
                RETURN_LSE: tl.constexpr,
                ):
    start_m = tl.program_id(0)

    off_z = tl.program_id(2).to(tl.int64)
    off_h = tl.program_id(1).to(tl.int64)

    q_scale_offset = (off_z * H + off_h) * tl.cdiv(qo_len, BLOCK_M)
    k_scale_offset = (off_z * (H // num_kv_groups) + off_h // num_kv_groups) * tl.cdiv(kv_len, BLOCK_N)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    Q_ptrs = Q + (off_z * stride_qz + off_h * stride_qh) + offs_m[:, None] * stride_qn + offs_k[None, :]
    Q_scale_ptr = Q_scale + q_scale_offset + start_m
    K_ptrs = K + (off_z * stride_kz + (off_h // num_kv_groups) * stride_kh) + offs_n[None, :] * stride_kn + offs_k[:, None]
    K_scale_ptr = K_scale + k_scale_offset
    V_ptrs = V + (off_z * stride_vz + (off_h // num_kv_groups) * stride_vh) + offs_n[:, None] * stride_vn + offs_k[None, :]
    O_block_ptr = Out + (off_z * stride_oz + off_h * stride_oh) + offs_m[:, None] * stride_on + offs_k[None, :]

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    q = tl.load(Q_ptrs, mask=offs_m[:, None] < qo_len)
    q_scale = tl.load(Q_scale_ptr)
    acc, l_i, m_i = _attn_fwd_inner_c(acc, l_i, m_i, q, q_scale, kv_len, K_ptrs, K_scale_ptr, V_ptrs, stride_kn, stride_vn,
                                      start_m,
                                      BLOCK_M, HEAD_DIM, BLOCK_N,
                                      4 - STAGE, offs_m, offs_n
                                      )

    acc, l_i, m_i = _attn_fwd_inner_c(acc, l_i, m_i, q, q_scale, kv_len, K_ptrs, K_scale_ptr, V_ptrs, stride_kn, stride_vn,
                                      start_m,
                                      BLOCK_M, HEAD_DIM, BLOCK_N,
                                      2, offs_m, offs_n
                                      )
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask=(offs_m[:, None] < qo_len))

    if RETURN_LSE:
        lse_ptrs = Lse + (off_z * qo_len * H + off_h * qo_len) + offs_m
        l_i = tl.log2(l_i) + m_i
        tl.store(lse_ptrs, l_i, mask=(offs_m < qo_len))


# ---------------------------------------------------------------- host flow of sageattn_qk_int8_pv_fp16_triton (NHD)
def quantized_attention_bf16(q, k, v, is_causal, softmax_scale, out):
    b = q.shape[0]
    qo_len = q.shape[1]
    h_qo = q.shape[2]
    head_dim = q.shape[3]
    kv_len = k.shape[1]
    h_kv = k.shape[2]
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and out.is_contiguous()
    assert head_dim == 64 or head_dim == 128
    assert h_qo % h_kv == 0
    num_kv_groups = h_qo // h_kv
    BLKQ = 128
    BLKK = 64
    dev = q.device

    # smooth_k: km = k.mean(dim=1)  (glue: parallel Triton column sums)
    ksum = torch.zeros((b, h_kv, head_dim), dtype=torch.float32, device=dev)
    _glue_k_colsum_kernel[(triton.cdiv(kv_len, 256), h_kv, b)](
        k, ksum, kv_len,
        k.stride(0), k.stride(1), k.stride(2),
        ksum.stride(0), ksum.stride(1),
        HEAD_DIM=head_dim, BLOCK_L=256, num_warps=8,
    )

    # v = v.to(torch.float16)  (glue: Triton cast)
    v_fp16 = torch.empty((b, kv_len, h_kv, head_dim), dtype=torch.float16, device=dev)
    nv = b * kv_len * h_kv * head_dim
    _glue_cast_v_fp16_kernel[(triton.cdiv(nv, 4096),)](v, v_fp16, nv, BLOCK=4096, num_warps=8)

    # per_block_int8 (NHD strides as upstream: stride_bz, stride_h, stride_seq)
    q_int8 = torch.empty((b, qo_len, h_qo, head_dim), dtype=torch.int8, device=dev)
    k_int8 = torch.empty((b, kv_len, h_kv, head_dim), dtype=torch.int8, device=dev)
    q_scale = torch.empty((b, h_qo, (qo_len + BLKQ - 1) // BLKQ), dtype=torch.float32, device=dev)
    k_scale = torch.empty((b, h_kv, (kv_len + BLKK - 1) // BLKK), dtype=torch.float32, device=dev)
    quant_per_block_int8_kernel[((qo_len + BLKQ - 1) // BLKQ, h_qo, b)](
        q, q_int8, q_scale, qo_len,
        q.stride(0), q.stride(2), q.stride(1),
        q_int8.stride(0), q_int8.stride(2), q_int8.stride(1),
        q_scale.stride(0), q_scale.stride(1),
        sm_scale=(float(softmax_scale) * 1.44269504),
        C=head_dim, BLK=BLKQ,
    )
    _glue_quant_k_smooth_kernel[((kv_len + BLKK - 1) // BLKK, h_kv, b)](
        k, ksum, k_int8, k_scale, kv_len,
        k.stride(0), k.stride(2), k.stride(1),
        ksum.stride(0), ksum.stride(1),
        k_int8.stride(0), k_int8.stride(2), k_int8.stride(1),
        k_scale.stride(0), k_scale.stride(1),
        C=head_dim, BLK=BLKK,
    )

    lse = torch.empty((1,), dtype=torch.float32, device=dev)
    BLOCK_M = 128
    BLOCK_N = 64
    num_warps = 8
    if head_dim == 64:
        num_warps = 4
    if is_causal:
        _attn_fwd_c[(triton.cdiv(qo_len, BLOCK_M), h_qo, b)](
            q_int8, k_int8, v_fp16, q_scale, k_scale, out, lse,
            q_int8.stride(0), q_int8.stride(2), q_int8.stride(1),
            k_int8.stride(0), k_int8.stride(2), k_int8.stride(1),
            v_fp16.stride(0), v_fp16.stride(2), v_fp16.stride(1),
            out.stride(0), out.stride(2), out.stride(1),
            qo_len, kv_len,
            h_qo, num_kv_groups,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=head_dim,
            STAGE=3,
            RETURN_LSE=False,
            num_warps=num_warps,
            num_stages=4,
        )
    else:
        num_stages = 4
        if head_dim == 64:
            num_stages = 3
        _attn_fwd_nc[(triton.cdiv(qo_len, BLOCK_M), h_qo, b)](
            q_int8, k_int8, v_fp16, q_scale, k_scale, out, None, lse,
            q_int8.stride(0), q_int8.stride(2), q_int8.stride(1),
            k_int8.stride(0), k_int8.stride(2), k_int8.stride(1),
            v_fp16.stride(0), v_fp16.stride(2), v_fp16.stride(1),
            out.stride(0), out.stride(2), out.stride(1),
            0, 0, 0, 0,
            qo_len, kv_len,
            h_qo, num_kv_groups,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=head_dim,
            STAGE=1,
            RETURN_LSE=False,
            num_warps=num_warps,
            num_stages=num_stages,
        )
