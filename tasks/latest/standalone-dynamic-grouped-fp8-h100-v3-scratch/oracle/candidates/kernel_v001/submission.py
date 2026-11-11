from __future__ import annotations

import triton
import triton.language as tl


_SCALE_BLOCK = 128
_PSUM_ALIGNMENT = 128


@triton.jit
def _grouped_fp8_kernel(
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    metadata_ptr,
    out_ptr,
    M_MAX: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    PSUM: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
    PSUM_ALIGNMENT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile_id = tl.program_id(axis=0)
    group = tl.program_id(axis=1)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    tile_m = tile_id // num_n_tiles
    tile_n = tile_id % num_n_tiles

    if PSUM:
        end = tl.load(metadata_ptr + group)
        previous_end = tl.load(metadata_ptr + group - 1, mask=group > 0, other=0)
        row_base = ((previous_end + PSUM_ALIGNMENT - 1) // PSUM_ALIGNMENT) * PSUM_ALIGNMENT
        valid_m = end - row_base
    else:
        row_base = group * M_MAX
        valid_m = tl.load(metadata_ptr + group)

    if tile_m * BLOCK_M < valid_m:
        offsets_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        b_group = group * N * K
        k_blocks = K // SCALE_BLOCK
        b_scale_group = group * (N // SCALE_BLOCK) * k_blocks
        for k_start in range(0, K, BLOCK_K):
            offsets_k = k_start + tl.arange(0, BLOCK_K)
            a = tl.load(
                a_ptr
                + (row_base + offsets_m[:, None]) * K
                + offsets_k[None, :],
                mask=offsets_m[:, None] < valid_m,
                other=0.0,
            )
            b = tl.load(
                b_ptr + b_group + offsets_n[None, :] * K + offsets_k[:, None],
                mask=offsets_n[None, :] < N,
                other=0.0,
            )
            partial = tl.dot(a, b, out_dtype=tl.float32)

            scale_index = k_start // SCALE_BLOCK
            a_scale = tl.load(
                a_scale_ptr
                + (row_base + offsets_m) * k_blocks
                + scale_index,
                mask=offsets_m < valid_m,
                other=0.0,
            ).to(tl.float32)
            b_scale = tl.load(
                b_scale_ptr
                + b_scale_group
                + (offsets_n // SCALE_BLOCK) * k_blocks
                + scale_index,
                mask=offsets_n < N,
                other=0.0,
            ).to(tl.float32)
            accumulator += partial * a_scale[:, None] * b_scale[None, :]

        tl.store(
            out_ptr
            + (row_base + offsets_m[:, None]) * N
            + offsets_n[None, :],
            accumulator,
            mask=(offsets_m[:, None] < valid_m) & (offsets_n[None, :] < N),
        )


def _launch(a, b, a_scale, b_scale, metadata, expected_m_hint, out, *, psum: bool) -> None:
    group_count = int(b.shape[0])
    n = int(b.shape[1])
    k = int(b.shape[2])
    m_max = int(a.shape[0] // group_count) if psum else int(a.shape[1])
    block_m = 64 if int(expected_m_hint) <= 768 else 128
    block_n = 128
    grid = (
        triton.cdiv(m_max, block_m) * triton.cdiv(n, block_n),
        group_count,
    )
    _grouped_fp8_kernel[grid](
        a,
        b,
        a_scale,
        b_scale,
        metadata,
        out,
        M_MAX=m_max,
        N=n,
        K=k,
        PSUM=psum,
        SCALE_BLOCK=_SCALE_BLOCK,
        PSUM_ALIGNMENT=_PSUM_ALIGNMENT,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=_SCALE_BLOCK,
        num_warps=4 if block_m == 64 else 8,
        num_stages=3,
    )


def grouped_fp8_gemm_nt_masked(
    a, b, a_scale, b_scale, m_sizes, expected_m_hint, out
) -> None:
    _launch(a, b, a_scale, b_scale, m_sizes, expected_m_hint, out, psum=False)


def grouped_fp8_gemm_nt_psum(
    a, b, a_scale, b_scale, group_ends, expected_m_hint, out
) -> None:
    _launch(a, b, a_scale, b_scale, group_ends, expected_m_hint, out, psum=True)
