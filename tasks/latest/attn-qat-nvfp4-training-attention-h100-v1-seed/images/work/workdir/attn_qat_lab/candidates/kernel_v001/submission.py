"""Attn-QAT training attention: NVFP4 fake quantisation, forward + backward."""
import torch
import triton
import triton.language as tl

MXFP_BLOCK_SIZE: tl.constexpr = 16

# ---------------------------------------------------------------------------
# NVFP4 fake quantisation (software e2m1 encode/decode)
# ---------------------------------------------------------------------------


@triton.jit
def _compute_quant_and_scale(src_tensor, valid_src_mask):
    BLOCK_SIZE_OUT_DIM: tl.constexpr = src_tensor.shape[0]
    BLOCK_SIZE_QUANT_DIM: tl.constexpr = src_tensor.shape[1]
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = src_tensor.shape[1] // 16

    f32_tensor = src_tensor.to(tl.float32)
    abs_tensor = tl.abs(f32_tensor)
    abs_tensor = tl.where(valid_src_mask, abs_tensor, -1.0)

    abs_tensor = tl.reshape(abs_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 16])

    # No global scale factor: s_dec = s_enc = 1.0
    s_dec = 1.0
    s_enc = 1.0

    max_val = tl.max(abs_tensor, axis=2, keep_dims=True)
    s_dec_b = max_val / 6
    s_dec_b_e4m3 = (s_dec_b * s_enc).to(tl.float8e4nv)
    s_enc_b = 1 / (s_dec_b_e4m3.to(tl.float32) * s_dec)

    f32_tensor = tl.reshape(f32_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 16])
    quant_tensor = f32_tensor * s_enc_b

    quant_tensor = quant_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM])
    quant_tensor = tl.where(valid_src_mask, quant_tensor, 0.0)
    dequant_scale = s_dec_b_e4m3.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE])

    # software e2m1 round-to-nearest-even encode
    quant_tensor = quant_tensor.to(tl.uint32, bitcast=True)
    signs = quant_tensor & 0x80000000
    exponents = (quant_tensor >> 23) & 0xFF
    mantissas_orig = (quant_tensor & 0x7FFFFF)

    E8_BIAS: tl.constexpr = 127
    E2_BIAS: tl.constexpr = 1
    is_subnormal = exponents < E8_BIAS
    adjusted_exponents = tl.core.sub(E8_BIAS, exponents + 1, sanitize_overflow=False)
    mantissas_pre = (0x400000 | (mantissas_orig >> 1))
    mantissas = tl.where(is_subnormal, mantissas_pre >> adjusted_exponents, mantissas_orig)

    exponents = tl.maximum(exponents, E8_BIAS - E2_BIAS) - (E8_BIAS - E2_BIAS)

    m2bits = mantissas >> 21
    lsb_keep = (m2bits >> 1) & 0x1
    guard = m2bits & 0x1
    IS_SRC_FP32: tl.constexpr = src_tensor.dtype == tl.float32
    if IS_SRC_FP32:
        bit0_dropped = (mantissas_orig & 0x1) != 0
        mask = (1 << tl.minimum(adjusted_exponents, 31)) - 1
        dropped_post = (mantissas_pre & mask) != 0
        sticky = is_subnormal & (bit0_dropped | dropped_post)
        sticky |= ((mantissas & 0x1FFFFF) != 0).to(tl.uint32)
    else:
        sticky = ((mantissas & 0x1FFFFF) != 0).to(tl.uint32)
    round_inc = guard & (sticky | lsb_keep)
    e2m1_tmp = tl.minimum((((exponents << 2) | m2bits) + round_inc) >> 1, 0x7)
    e2m1_value = ((signs >> 28) | e2m1_tmp).to(tl.uint8)

    e2m1_value = tl.reshape(e2m1_value, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM // 2, 2])
    evens, odds = tl.split(e2m1_value)
    out_tensor = evens | (odds << 4)

    return out_tensor, dequant_scale, s_dec


@triton.jit
def _compute_dequant(mx_tensor, scale, s_dec, BLOCK_SIZE_OUT_DIM: tl.constexpr, BLOCK_SIZE_QUANT_DIM: tl.constexpr,
                     dst_dtype: tl.constexpr):
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = BLOCK_SIZE_QUANT_DIM // 16
    tl.static_assert(dst_dtype == tl.bfloat16)
    dst_scale = scale.to(tl.bfloat16)
    intermediate_dtype: tl.constexpr = tl.bfloat16

    dst_bias: tl.constexpr = 127
    dst_0p5: tl.constexpr = 16128
    dst_m_bits: tl.constexpr = 7
    em0 = mx_tensor & 0x07
    em1 = mx_tensor & 0x70
    x0 = (em0.to(tl.uint16) << (dst_m_bits - 1)) | ((mx_tensor & 0x08).to(tl.uint16) << 12)
    x1 = (em1.to(tl.uint16) << (dst_m_bits - 5)) | ((mx_tensor & 0x80).to(tl.uint16) << 8)
    x0 = tl.where((em0 & 0x06) != 0, x0 + ((dst_bias - 1) << dst_m_bits), x0)
    x1 = tl.where((em1 & 0x60) != 0, x1 + ((dst_bias - 1) << dst_m_bits), x1)
    x0 = tl.where(em0 == 0x01, dst_0p5 | (x0 & 0x8000), x0)
    x1 = tl.where(em1 == 0x10, dst_0p5 | (x1 & 0x8000), x1)
    dst_tensor = tl.interleave(x0, x1).to(intermediate_dtype, bitcast=True)

    dst_tensor = dst_tensor.to(dst_dtype)

    dst_tensor = dst_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 16])
    dst_scale = dst_scale.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 1])

    out_tensor = dst_tensor * dst_scale * s_dec
    max_fin = 3.3895313892515355e+38
    out_tensor = tl.clamp(out_tensor, min=-max_fin, max=max_fin)
    out_tensor = out_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM])
    out_tensor = out_tensor.to(dst_dtype)
    return out_tensor


@triton.jit
def _fake_quantize(src_tensor, valid_src_mask, BLOCK_SIZE_OUT_DIM: tl.constexpr, BLOCK_SIZE_QUANT_DIM: tl.constexpr,
                   dst_dtype: tl.constexpr):
    high_prec_src_tensor = src_tensor
    q_tensor, src_scale, src_s_dec = _compute_quant_and_scale(src_tensor=src_tensor, valid_src_mask=valid_src_mask)
    out = _compute_dequant(mx_tensor=q_tensor, scale=src_scale, s_dec=src_s_dec,
                           BLOCK_SIZE_OUT_DIM=BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM=BLOCK_SIZE_QUANT_DIM,
                           dst_dtype=dst_dtype)
    return out, high_prec_src_tensor.to(out.dtype)


@triton.jit
def _fake_quantize_rows(X, fake_X, stride_z, stride_h, stride_tok, stride_d, fake_stride_z, fake_stride_h,
                        fake_stride_tok, fake_stride_d, H, N_CTX, BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr):
    """FQ_D of one `[B, H, N, D]` tensor, one BLOCK_M-row tile of one head per program."""
    bhid = tl.program_id(1)
    X += (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    fake_X += (fake_stride_h * (bhid % H) + fake_stride_z * (bhid // H)).to(tl.int64)
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, HEAD_DIM)
    valid = offs_m < N_CTX
    x = tl.load(X + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d, mask=valid[:, None], other=0.0)
    x, _ = _fake_quantize(src_tensor=x, valid_src_mask=valid[:, None], BLOCK_SIZE_OUT_DIM=BLOCK_M,
                          BLOCK_SIZE_QUANT_DIM=HEAD_DIM, dst_dtype=tl.bfloat16)
    tl.store(fake_X + offs_m[:, None] * fake_stride_tok + offs_k[None, :] * fake_stride_d, x, mask=valid[:, None])


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------


@triton.jit
def _attn_fwd_exact_m(Q, K, M, L, sm_scale, H, H_KV, N_CTX_Q, N_CTX_KV, CAUSAL: tl.constexpr,
                      HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Legacy 32x32 softmax statistic: M = m + log2(l), l from bf16-rounded P~."""
    start_m = tl.program_id(0) * BLOCK_M
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_kv = off_z * H_KV + (off_hz % H) // (H // H_KV)
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    q_valid = offs_m < N_CTX_Q
    Qp = Q + off_hz.to(tl.int64) * N_CTX_Q * HEAD_DIM
    Kp = K + off_kv.to(tl.int64) * N_CTX_KV * HEAD_DIM
    q = tl.load(Qp + offs_m[:, None] * HEAD_DIM + offs_k[None, :], mask=q_valid[:, None], other=0.0)
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, tl.float32)
    qk_scale = sm_scale * 1.44269504
    hi = N_CTX_KV
    if CAUSAL:
        hi = tl.minimum(N_CTX_KV, start_m + BLOCK_M)
    for start_n in tl.range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        kv_valid = start_n + offs_n < N_CTX_KV
        keep = kv_valid[None, :]
        if CAUSAL:
            keep = keep & (offs_m[:, None] >= (start_n + offs_n)[None, :])
        k = tl.load(Kp + (start_n + offs_n)[:, None] * HEAD_DIM + offs_k[None, :], mask=kv_valid[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k))
        qk = tl.where(keep, qk, -1.0e6)
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1) * qk_scale)
        p = tl.math.exp2(qk * qk_scale - m_ij[:, None])
        l_ij = tl.sum(p.to(tl.bfloat16), axis=1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
    tl.store(M + off_hz * N_CTX_Q + offs_m, m_i, mask=q_valid)
    tl.store(L + off_hz * N_CTX_Q + offs_m, l_i, mask=q_valid)


@triton.jit
def _attn_fwd_final(sm_scale, M, L, MOUT, Z, H, H_KV, desc_q, desc_k, desc_v, desc_o, desc_high_prec_o, N_CTX_Q,
                    N_CTX_KV, CAUSAL: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr,
                    BLOCK_N: tl.constexpr, warp_specialize: tl.constexpr):
    dtype: tl.constexpr = tl.bfloat16
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_kv = off_z * H_KV + off_h // (H // H_KV)
    y_dim_q = Z * H * N_CTX_Q
    y_dim_kv = Z * H_KV * N_CTX_KV
    desc_q = tl.make_tensor_descriptor(desc_q, shape=[y_dim_q, HEAD_DIM], strides=[HEAD_DIM, 1],
                                       block_shape=[BLOCK_M, HEAD_DIM])
    desc_v = tl.make_tensor_descriptor(desc_v, shape=[y_dim_kv, HEAD_DIM], strides=[HEAD_DIM, 1],
                                       block_shape=[BLOCK_N, HEAD_DIM])
    desc_k = tl.make_tensor_descriptor(desc_k, shape=[y_dim_kv, HEAD_DIM], strides=[HEAD_DIM, 1],
                                       block_shape=[BLOCK_N, HEAD_DIM])
    desc_o = tl.make_tensor_descriptor(desc_o, shape=[Z * H, N_CTX_Q, HEAD_DIM],
                                       strides=[N_CTX_Q * HEAD_DIM, HEAD_DIM, 1], block_shape=[1, BLOCK_M, HEAD_DIM])
    desc_high_prec_o = tl.make_tensor_descriptor(desc_high_prec_o, shape=[Z * H, N_CTX_Q, HEAD_DIM],
                                                 strides=[N_CTX_Q * HEAD_DIM, HEAD_DIM, 1],
                                                 block_shape=[1, BLOCK_M, HEAD_DIM])
    offset_y_q = off_hz * N_CTX_Q
    offset_y_kv = off_kv * N_CTX_KV
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    q_valid = offs_m < N_CTX_Q
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    high_prec_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    q = desc_q.load([offset_y_q + start_m * BLOCK_M, 0])
    q = tl.where(q_valid[:, None], q, 0.0)
    m_row = tl.load(M + off_hz * N_CTX_Q + offs_m, mask=q_valid, other=0.0)
    m = m_row[:, None]
    l_i = tl.load(L + off_hz * N_CTX_Q + offs_m, mask=q_valid, other=1.0)
    hi = N_CTX_KV
    if CAUSAL:
        hi = tl.minimum(N_CTX_KV, (start_m + 1) * BLOCK_M)
    offsetk_y = offset_y_kv
    for start_n in tl.range(0, hi, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        kv_valid = (start_n + offs_n) < N_CTX_KV
        keep = kv_valid[None, :]
        if CAUSAL:
            keep = keep & (offs_m[:, None] >= (start_n + offs_n)[None, :])
        k = desc_k.load([offsetk_y, 0])
        k = tl.where(kv_valid[:, None], k, 0.0)
        qk = tl.dot(q, tl.trans(k))
        qk = tl.where(keep, qk, -1.0e6)
        p = tl.math.exp2(qk * qk_scale - m)
        p, high_prec_p = _fake_quantize(src_tensor=p,
                                        valid_src_mask=tl.full(shape=p.shape, value=1.0, dtype=p.dtype) == 1.0,
                                        BLOCK_SIZE_OUT_DIM=BLOCK_M, BLOCK_SIZE_QUANT_DIM=BLOCK_N, dst_dtype=dtype)
        v = desc_v.load([offsetk_y, 0])
        v = tl.where(kv_valid[:, None], v, 0.0)
        acc = tl.dot(p.to(dtype), v.to(dtype), acc)
        high_prec_acc = tl.dot(high_prec_p, v, high_prec_acc)
        offsetk_y += BLOCK_N
    acc = acc / l_i[:, None]
    high_prec_acc = high_prec_acc / l_i[:, None]
    tl.store(MOUT + off_hz * N_CTX_Q + offs_m, m_row + tl.math.log2(l_i), mask=q_valid)
    desc_o.store([off_hz, start_m * BLOCK_M, 0], acc.to(dtype)[None, :, :])
    desc_high_prec_o.store([off_hz, start_m * BLOCK_M, 0], high_prec_acc.to(dtype)[None, :, :])


# ---------------------------------------------------------------------------
# backward
# ---------------------------------------------------------------------------


@triton.jit
def _attn_bwd_preprocess(O, DO, Delta, Z, H, N_CTX, BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1).to(tl.int64)
    off_n = tl.arange(0, HEAD_DIM)
    valid = off_m < N_CTX
    o = tl.load(O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :], mask=valid[:, None],
                other=0.0)
    do = tl.load(DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :], mask=valid[:, None],
                 other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_hz * N_CTX + off_m, delta, mask=valid)


@triton.jit
def _attn_bwd_dq(Q, K, V, sm_scale, DO, DQ, M, D, stride_z_q, stride_z_kv, stride_h_q, stride_h_kv, stride_tok_q,
                 stride_tok_kv, stride_d_q, stride_d_kv, H, H_KV, N_CTX_Q, N_CTX_KV, CAUSAL: tl.constexpr,
                 BLOCK_M2: tl.constexpr, BLOCK_N2: tl.constexpr, HEAD_DIM: tl.constexpr):
    """dQ for one BLOCK_M2-row tile of one query head."""
    RCP_LN2: tl.constexpr = 1.4426950408889634
    qk_scale = sm_scale * RCP_LN2
    bhid = tl.program_id(2)
    off_z = bhid // H
    off_h = bhid % H
    off_kvh = off_h // (H // H_KV)
    off_chz = (bhid * N_CTX_Q).to(tl.int64)
    adj_q = (stride_h_q * off_h + stride_z_q * off_z).to(tl.int64)
    adj_kv = (stride_h_kv * off_kvh + stride_z_kv * off_z).to(tl.int64)
    Q += adj_q
    K += adj_kv
    V += adj_kv
    DO += adj_q
    DQ += adj_q
    M += off_chz
    D += off_chz

    start_m = tl.program_id(0) * BLOCK_M2
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_k = tl.arange(0, HEAD_DIM)
    q_valid = offs_m < N_CTX_Q
    q = tl.load(Q + offs_m[:, None] * stride_tok_q + offs_k[None, :] * stride_d_q, mask=q_valid[:, None], other=0.0)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok_q + offs_k[None, :] * stride_d_q, mask=q_valid[:, None],
                 other=0.0)
    m = tl.load(M + offs_m, mask=q_valid, other=0.0)[:, None]
    Di = tl.load(D + offs_m, mask=q_valid, other=0.0)
    hi = N_CTX_KV
    if CAUSAL:
        hi = tl.minimum(N_CTX_KV, start_m + BLOCK_M2)
    for start_n in range(0, hi, BLOCK_N2):
        offs_n = start_n + tl.arange(0, BLOCK_N2)
        kv_valid = offs_n < N_CTX_KV
        keep = q_valid[:, None] & kv_valid[None, :]
        if CAUSAL:
            keep = keep & (offs_m[:, None] >= offs_n[None, :])
        k = tl.load(K + offs_n[:, None] * stride_tok_kv + offs_k[None, :] * stride_d_kv, mask=kv_valid[:, None],
                    other=0.0)
        v = tl.load(V + offs_n[:, None] * stride_tok_kv + offs_k[None, :] * stride_d_kv, mask=kv_valid[:, None],
                    other=0.0)
        qk = tl.dot(q, tl.trans(k))
        qk = qk * qk_scale
        p = tl.where(keep, tl.math.exp2(qk - m), 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - Di[:, None])
        ds = ds.to(tl.bfloat16)
        dq += tl.dot(ds, k)
    dq *= sm_scale
    dq_ptrs = DQ + offs_m[:, None] * stride_tok_q + offs_k[None, :] * stride_d_q
    tl.store(dq_ptrs, dq, mask=q_valid[:, None])


@triton.jit
def _attn_bwd_dkdv(Q, K, V, sm_scale, DO, DK, DV, M, D, stride_z_q, stride_z_kv, stride_h_q, stride_h_kv,
                   stride_tok_q, stride_tok_kv, stride_d_q, stride_d_kv, H, H_KV, N_CTX_Q, N_CTX_KV,
                   CAUSAL: tl.constexpr, BLOCK_M1: tl.constexpr, BLOCK_N1: tl.constexpr, HEAD_DIM: tl.constexpr):
    """dK and dV for one BLOCK_N1-row tile of one key/value head, summed over its query heads."""
    RCP_LN2: tl.constexpr = 1.4426950408889634
    qk_scale = sm_scale * RCP_LN2
    bhid = tl.program_id(2)
    off_z = bhid // H_KV
    off_kvh = bhid % H_KV
    group = H // H_KV
    adj_kv = (stride_h_kv * off_kvh + stride_z_kv * off_z).to(tl.int64)
    K += adj_kv
    V += adj_kv
    DK += adj_kv
    DV += adj_kv

    start_n = tl.program_id(0) * BLOCK_N1
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    kv_valid = offs_n < N_CTX_KV
    k_block = tl.load(K + offs_n[:, None] * stride_tok_kv + offs_k[None, :] * stride_d_kv, mask=kv_valid[:, None],
                      other=0.0)
    v_block = tl.load(V + offs_n[:, None] * stride_tok_kv + offs_k[None, :] * stride_d_kv, mask=kv_valid[:, None],
                      other=0.0)
    lo = 0
    if CAUSAL:
        lo = (start_n // BLOCK_M1) * BLOCK_M1
    for g in range(group):
        off_h = off_kvh * group + g
        adj_q = (stride_h_q * off_h + stride_z_q * off_z).to(tl.int64)
        off_chz = ((off_z * H + off_h) * N_CTX_Q).to(tl.int64)
        for start_m in range(lo, N_CTX_Q, BLOCK_M1):
            offs_m = start_m + tl.arange(0, BLOCK_M1)
            q_valid = offs_m < N_CTX_Q
            keep = q_valid[:, None] & kv_valid[None, :]
            if CAUSAL:
                keep = keep & (offs_m[:, None] >= offs_n[None, :])
            q = tl.load(Q + adj_q + offs_m[:, None] * stride_tok_q + offs_k[None, :] * stride_d_q,
                        mask=q_valid[:, None], other=0.0)
            do = tl.load(DO + adj_q + offs_m[:, None] * stride_tok_q + offs_k[None, :] * stride_d_q,
                         mask=q_valid[:, None], other=0.0)
            m = tl.load(M + off_chz + offs_m, mask=q_valid, other=0.0)
            qk = tl.dot(q, tl.trans(k_block))
            qk = qk * qk_scale
            p = tl.where(keep, tl.math.exp2(qk - m[:, None]), 0.0)
            p_quant, _ = _fake_quantize(src_tensor=p,
                                        valid_src_mask=tl.full(shape=p.shape, value=1.0, dtype=p.dtype) == 1.0,
                                        BLOCK_SIZE_OUT_DIM=BLOCK_M1, BLOCK_SIZE_QUANT_DIM=BLOCK_N1,
                                        dst_dtype=tl.bfloat16)
            dv += tl.dot(tl.trans(p_quant.to(tl.bfloat16)), do)
            dp = tl.dot(do, tl.trans(v_block))
            Di = tl.load(D + off_chz + offs_m, mask=q_valid, other=0.0)
            ds = p * (dp - Di[:, None])
            ds = ds.to(tl.bfloat16)
            dk += tl.dot(tl.trans(ds), q)
    dv_ptrs = DV + offs_n[:, None] * stride_tok_kv + offs_k[None, :] * stride_d_kv
    tl.store(dv_ptrs, dv, mask=kv_valid[:, None])
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok_kv + offs_k[None, :] * stride_d_kv
    tl.store(dk_ptrs, dk, mask=kv_valid[:, None])


# ---------------------------------------------------------------------------
# host entry points
# ---------------------------------------------------------------------------

_QKV_BLOCK = 32
_FWD_BLOCK_M = 32
_FWD_BLOCK_N = 32
_FWD_STAGES = 2
_PRE_BLOCK = 128
_BWD_BLOCK = 32
_BWD_STAGES = 3
_WARP_SPECIALIZE = True
_STAT_BLOCK = 32
_STAT_WARPS = 8
_STAT_STAGES = 4


def _alloc_fn(size, alignment, stream):
    """Scratch allocator for the device-side TMA descriptors."""
    return torch.empty(size, dtype=torch.int8, device="cuda")


def attn_qat_fwd(q, k, v, softmax_scale, causal, o):
    """Overwrite `o`; return the opaque `saved` tuple consumed by `attn_qat_bwd`."""
    triton.set_allocator(_alloc_fn)
    Z, H, N_CTX_Q, HEAD_DIM = q.shape
    H_KV = k.shape[1]
    N_CTX_KV = k.shape[2]
    warps = 8 if HEAD_DIM == 256 else 4
    fake_q = torch.empty_like(q)
    fake_k = torch.empty_like(k)
    fake_v = torch.empty_like(v)
    high_prec_o = torch.empty_like(o)
    M = torch.empty((Z, H, N_CTX_Q), device=q.device, dtype=torch.float32)
    Mm = torch.empty_like(M)
    Ll = torch.empty_like(M)
    grid_q = (triton.cdiv(N_CTX_Q, _QKV_BLOCK), Z * H, 1)
    grid_kv = (triton.cdiv(N_CTX_KV, _QKV_BLOCK), Z * H_KV, 1)
    _fake_quantize_rows[grid_q](q, fake_q, q.stride(0), q.stride(1), q.stride(2), q.stride(3), fake_q.stride(0),
                                fake_q.stride(1), fake_q.stride(2), fake_q.stride(3), H, N_CTX_Q, BLOCK_M=_QKV_BLOCK,
                                HEAD_DIM=HEAD_DIM)
    _fake_quantize_rows[grid_kv](k, fake_k, k.stride(0), k.stride(1), k.stride(2), k.stride(3), fake_k.stride(0),
                                 fake_k.stride(1), fake_k.stride(2), fake_k.stride(3), H_KV, N_CTX_KV,
                                 BLOCK_M=_QKV_BLOCK, HEAD_DIM=HEAD_DIM)
    _fake_quantize_rows[grid_kv](v, fake_v, v.stride(0), v.stride(1), v.stride(2), v.stride(3), fake_v.stride(0),
                                 fake_v.stride(1), fake_v.stride(2), fake_v.stride(3), H_KV, N_CTX_KV,
                                 BLOCK_M=_QKV_BLOCK, HEAD_DIM=HEAD_DIM)
    gm = (triton.cdiv(N_CTX_Q, _STAT_BLOCK), Z * H)
    _attn_fwd_exact_m[gm](fake_q, fake_k, Mm, Ll, softmax_scale, H, H_KV, N_CTX_Q, N_CTX_KV, CAUSAL=causal,
                          HEAD_DIM=HEAD_DIM, BLOCK_M=_STAT_BLOCK, BLOCK_N=_STAT_BLOCK, num_warps=_STAT_WARPS,
                          num_stages=_STAT_STAGES)
    grid = (triton.cdiv(N_CTX_Q, _FWD_BLOCK_M), Z * H, 1)
    _attn_fwd_final[grid](softmax_scale, Mm, Ll, M, Z, H, H_KV, fake_q, fake_k, fake_v, o, high_prec_o,
                          N_CTX_Q=N_CTX_Q, N_CTX_KV=N_CTX_KV, CAUSAL=causal, HEAD_DIM=HEAD_DIM, BLOCK_M=_FWD_BLOCK_M,
                          BLOCK_N=_FWD_BLOCK_N, warp_specialize=_WARP_SPECIALIZE, num_warps=warps,
                          num_stages=_FWD_STAGES)
    return (fake_q, fake_k, fake_v, high_prec_o, M)


def attn_qat_bwd(q, k, v, softmax_scale, causal, o, saved, do, dq, dk, dv):
    """Overwrite `dq`, `dk`, `dv`.  `q, k, v, do, o` are read-only."""
    fake_q, fake_k, fake_v, high_prec_o, M = saved
    Z, H, N_CTX_Q, HEAD_DIM = q.shape
    H_KV = k.shape[1]
    N_CTX_KV = k.shape[2]
    warps = 8 if HEAD_DIM == 256 else 4
    delta = torch.empty_like(M)
    pre_grid = (triton.cdiv(N_CTX_Q, _PRE_BLOCK), Z * H)
    _attn_bwd_preprocess[pre_grid](high_prec_o, do, delta, Z, H, N_CTX_Q, BLOCK_M=_PRE_BLOCK, HEAD_DIM=HEAD_DIM)
    B = _BWD_BLOCK
    grid_dq = (triton.cdiv(N_CTX_Q, B), 1, Z * H)
    _attn_bwd_dq[grid_dq](fake_q, fake_k, fake_v, softmax_scale, do, dq, M, delta, fake_q.stride(0), fake_k.stride(0),
                          fake_q.stride(1), fake_k.stride(1), fake_q.stride(2), fake_k.stride(2), fake_q.stride(3),
                          fake_k.stride(3), H, H_KV, N_CTX_Q, N_CTX_KV, CAUSAL=causal, BLOCK_M2=B, BLOCK_N2=B,
                          HEAD_DIM=HEAD_DIM, num_warps=warps, num_stages=_BWD_STAGES)
    grid_dkdv = (triton.cdiv(N_CTX_KV, B), 1, Z * H_KV)
    _attn_bwd_dkdv[grid_dkdv](fake_q, fake_k, fake_v, softmax_scale, do, dk, dv, M, delta, fake_q.stride(0),
                              fake_k.stride(0), fake_q.stride(1), fake_k.stride(1), fake_q.stride(2), fake_k.stride(2),
                              fake_q.stride(3), fake_k.stride(3), H, H_KV, N_CTX_Q, N_CTX_KV, CAUSAL=causal,
                              BLOCK_M1=B, BLOCK_N1=B, HEAD_DIM=HEAD_DIM, num_warps=warps, num_stages=_BWD_STAGES)
