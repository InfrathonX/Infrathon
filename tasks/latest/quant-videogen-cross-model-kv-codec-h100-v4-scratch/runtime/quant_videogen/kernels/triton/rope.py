"""Triton rotary embedding used by the task-owned model integration."""

import torch
import triton
import triton.language as tl


def _configs():
    return [
        triton.Config(
            {"BLOCK_M": block}, num_stages=stages, num_warps=warps
        )
        for block in (64, 128, 256, 512)
        for warps in (4, 8)
        for stages in (2, 3, 4, 5)
    ]


@triton.autotune(configs=_configs(), key=["HEAD_DIM", "HEAD_DIM_HALF"])
@triton.jit
def _triton_rope(
    q_ptr,
    q_stride_0,
    q_stride_1,
    k_ptr,
    k_stride_0,
    k_stride_1,
    q_out_ptr,
    q_out_stride_0,
    q_out_stride_1,
    k_out_ptr,
    k_out_stride_0,
    k_out_stride_1,
    cos_ptr,
    cos_stride_0,
    sin_ptr,
    sin_stride_0,
    BLOCK_M: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_HALF: tl.constexpr,
):
    row_id = tl.program_id(0)
    seq_id = tl.program_id(1)
    q_ptr += row_id * q_stride_0
    k_ptr += row_id * k_stride_0
    q_out_ptr += row_id * q_out_stride_0
    k_out_ptr += row_id * k_out_stride_0

    sequence_offsets = seq_id * BLOCK_M + tl.arange(0, BLOCK_M)
    head_offsets = tl.arange(0, HEAD_DIM)
    q_ptr += sequence_offsets[:, None] * q_stride_1 + head_offsets[None, :]
    k_ptr += sequence_offsets[:, None] * k_stride_1 + head_offsets[None, :]
    cos_ptr += sequence_offsets[:, None] * cos_stride_0 + head_offsets[None, :]
    sin_ptr += sequence_offsets[:, None] * sin_stride_0 + head_offsets[None, :]
    q_out_ptr += sequence_offsets[:, None] * q_out_stride_1 + head_offsets[None, :]
    k_out_ptr += sequence_offsets[:, None] * k_out_stride_1 + head_offsets[None, :]
    mask = sequence_offsets[:, None] < SEQ_LEN

    q = tl.load(q_ptr, mask=mask, other=0.0).to(tl.float32)
    k = tl.load(k_ptr, mask=mask, other=0.0).to(tl.float32)
    cos = tl.load(cos_ptr, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr, mask=mask, other=0.0).to(tl.float32)

    cos = cos.reshape(BLOCK_M, HEAD_DIM_HALF, 2)
    sin = sin.reshape(BLOCK_M, HEAD_DIM_HALF, 2)
    cos_half, _ = tl.split(cos)
    sin_half, _ = tl.split(sin)
    q = q.reshape(BLOCK_M, HEAD_DIM_HALF, 2)
    k = k.reshape(BLOCK_M, HEAD_DIM_HALF, 2)
    q_first, q_second = tl.split(q)
    k_first, k_second = tl.split(k)

    q_out = tl.interleave(
        q_first * cos_half - q_second * sin_half,
        q_first * sin_half + q_second * cos_half,
    )
    k_out = tl.interleave(
        k_first * cos_half - k_second * sin_half,
        k_first * sin_half + k_second * cos_half,
    )
    q_out = tl.reshape(q_out, (BLOCK_M, HEAD_DIM))
    k_out = tl.reshape(k_out, (BLOCK_M, HEAD_DIM))
    tl.store(q_out_ptr, q_out, mask=mask)
    tl.store(k_out_ptr, k_out, mask=mask)


def rope_forward(q, k, cos, sin):
    batch_size, num_heads, sequence_length, head_dim = q.shape
    q = q.reshape(batch_size * num_heads, sequence_length, head_dim).contiguous()
    k = k.reshape(batch_size * num_heads, sequence_length, head_dim).contiguous()
    cos = cos.reshape(sequence_length, head_dim).contiguous()
    sin = sin.reshape(sequence_length, head_dim).contiguous()
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    grid = lambda meta: (
        batch_size * num_heads,
        triton.cdiv(sequence_length, meta["BLOCK_M"]),
    )
    _triton_rope[grid](
        q,
        q.stride(0),
        q.stride(1),
        k,
        k.stride(0),
        k.stride(1),
        q_out,
        q_out.stride(0),
        q_out.stride(1),
        k_out,
        k_out.stride(0),
        k_out.stride(1),
        cos,
        cos.stride(0),
        sin,
        sin.stride(0),
        SEQ_LEN=sequence_length,
        HEAD_DIM=head_dim,
        HEAD_DIM_HALF=head_dim // 2,
    )
    return (
        q_out.reshape(batch_size, num_heads, sequence_length, head_dim),
        k_out.reshape(batch_size, num_heads, sequence_length, head_dim),
    )


def rope_forward_seq_first(q, k, freqs_cis):
    q = q.permute(0, 2, 1, 3).contiguous()
    k = k.permute(0, 2, 1, 3).contiguous()
    q_out, k_out = rope_forward(q, k, *freqs_cis)
    return q_out.permute(0, 2, 1, 3), k_out.permute(0, 2, 1, 3)
