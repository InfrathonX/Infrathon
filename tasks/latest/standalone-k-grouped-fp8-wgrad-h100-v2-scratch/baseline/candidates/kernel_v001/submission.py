from __future__ import annotations

import torch
import triton
import triton.language as tl


_SCALE_BLOCK = 128


@triton.jit
def _k_grouped_wgrad_kernel(
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    starts_ptr,
    ks_ptr,
    c_ptr,
    out_ptr,
    M,
    N,
    SCALE_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tile_id = tl.program_id(axis=0)
    group = tl.program_id(axis=1)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    tile_m = tile_id // num_n_tiles
    tile_n = tile_id % num_n_tiles

    # Groups are concatenated along K without gaps; every group size is 0 or a
    # multiple of 128, so each K block lies inside one group and needs no K mask.
    start = tl.load(starts_ptr + group)
    group_k = tl.load(ks_ptr + group)

    offsets_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offsets_m < M
    mask_n = offsets_n < N
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, group_k, SCALE_BLOCK):
        rows = start + k_start + tl.arange(0, SCALE_BLOCK)
        a = tl.load(a_ptr + rows[:, None] * M + offsets_m[None, :], mask=mask_m[None, :], other=0.0)
        b = tl.load(b_ptr + rows[:, None] * N + offsets_n[None, :], mask=mask_n[None, :], other=0.0)
        # FP8 E4M3 values are exact in BF16, and the BF16 product accumulates in FP32.
        partial = tl.dot(
            tl.trans(a.to(tl.bfloat16)),
            b.to(tl.bfloat16),
            out_dtype=tl.float32,
        )
        scale_row = (start + k_start) // SCALE_BLOCK
        a_scale = tl.load(a_scale_ptr + scale_row * M + offsets_m, mask=mask_m, other=0.0)
        b_scale = tl.load(b_scale_ptr + scale_row * N + offsets_n, mask=mask_n, other=0.0)
        accumulator += partial * a_scale[:, None] * b_scale[None, :]

    output_offsets = group * M * N + offsets_m[:, None] * N + offsets_n[None, :]
    output_mask = mask_m[:, None] & mask_n[None, :]
    addend = tl.load(c_ptr + output_offsets, mask=output_mask, other=0.0)
    tl.store(out_ptr + output_offsets, accumulator + addend, mask=output_mask)


@triton.jit
def _copy_kernel(source_ptr, target_ptr, total, BLOCK: tl.constexpr):
    offsets = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    tl.store(target_ptr + offsets, tl.load(source_ptr + offsets, mask=mask), mask=mask)


def k_grouped_fp8_gemm_tn_contiguous(
    a, b, a_scale, b_scale, ks, c, expected_k_hint, out
) -> None:
    groups, m, n = (int(value) for value in out.shape)
    if int(a.shape[0]) == 0:
        # Every group is empty: the result is the addend.
        total = groups * m * n
        _copy_kernel[(triton.cdiv(total, 1024),)](c, out, total, BLOCK=1024)
        return
    starts = torch.cumsum(ks, dim=0, dtype=torch.int32) - ks
    block_m = 128 if m >= 1024 else 64
    block_n = 128 if n >= 1024 else 64
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n), groups)
    _k_grouped_wgrad_kernel[grid](
        a,
        b,
        a_scale,
        b_scale,
        starts,
        ks,
        c,
        out,
        m,
        n,
        SCALE_BLOCK=_SCALE_BLOCK,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=8,
        num_stages=3,
    )
