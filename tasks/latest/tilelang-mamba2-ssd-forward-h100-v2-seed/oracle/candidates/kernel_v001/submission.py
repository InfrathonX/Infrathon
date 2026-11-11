"""Readable four-stage TileLang seed for Mamba-2 SSD forward.

The implementation deliberately fixes the chunk and tile sizes, materialises
the inter-stage FP32 tensors, and leaves scheduling and fusion work to benchmark
participants.  All four dense products use ``T.gemm`` with bfloat16 operands
and FP32 accumulators.
"""

from functools import lru_cache

import torch
import tilelang
import tilelang.language as T


CHUNK = 128
BLOCK_M = 64
BLOCK_N = 64
STATE_BLOCK_N = 128
BLOCK_K = 64
THREADS = 128
LOG2E = 1.4426950408889634

# Keep this baseline out of TileLang's automatic Hopper warp-specialised path.
_BASELINE_PASSES = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}


@tilelang.jit(pass_configs=_BASELINE_PASSES)
def _chunk_state(batch, seqlen, heads, headdim, groups, dstate):
    chunks = (seqlen + CHUNK - 1) // CHUNK
    state_block_m = 128 if headdim == 128 else BLOCK_M

    @T.prim_func
    def kernel(
        x: T.Tensor((batch, seqlen, heads, headdim), T.bfloat16),
        dt: T.Tensor((batch, seqlen, heads), T.float32),
        A: T.Tensor((heads,), T.float32),
        B: T.Tensor((batch, seqlen, groups, dstate), T.bfloat16),
        chunk_state: T.Tensor((batch, chunks, heads, headdim, dstate), T.bfloat16),
        chunk_decay: T.Tensor((batch, heads, chunks), T.float32),
    ):
        with T.Kernel(
            heads,
            T.ceildiv(headdim, state_block_m) * T.ceildiv(dstate, STATE_BLOCK_N),
            batch * chunks,
            threads=THREADS,
        ) as (head, tile, batch_chunk):
            x_shared = T.alloc_shared((BLOCK_K, state_block_m), T.bfloat16)
            x_local = T.alloc_fragment((BLOCK_K, state_block_m), T.bfloat16)
            scaled_x_t = T.alloc_fragment((state_block_m, BLOCK_K), T.bfloat16)
            b_shared = T.alloc_shared((BLOCK_K, STATE_BLOCK_N), T.bfloat16)
            dt_shared = T.alloc_shared((BLOCK_K,), T.float32)
            dt_local = T.alloc_fragment((BLOCK_K,), T.float32)
            values = T.alloc_fragment((CHUNK,), T.float32)
            prefix_all_shared = T.alloc_shared((CHUNK,), T.float32)
            state_acc = T.alloc_fragment((state_block_m, STATE_BLOCK_N), T.float32)
            state_shared = T.alloc_shared((state_block_m, STATE_BLOCK_N), T.float32)

            batch_idx = batch_chunk % batch
            chunk = batch_chunk // batch
            m_tile = tile // T.ceildiv(dstate, STATE_BLOCK_N)
            n_tile = tile % T.ceildiv(dstate, STATE_BLOCK_N)
            group = head // (heads // groups)
            for row in T.Parallel(CHUNK):
                token = chunk * CHUNK + row
                values[row] = T.if_then_else(
                    token < seqlen,
                    A[head] * dt[batch_idx, token, head],
                    0.0,
                )
            T.cumsum(values)
            T.copy(values, prefix_all_shared)
            chunk_decay[batch_idx, head, chunk] = T.exp2(
                prefix_all_shared[CHUNK - 1] * LOG2E
            )

            T.clear(state_acc)
            for k_tile in T.Pipelined(CHUNK // BLOCK_K, num_stages=2):
                token_start = chunk * CHUNK + k_tile * BLOCK_K
                T.copy(
                    x[
                        batch_idx,
                        token_start : token_start + BLOCK_K,
                        head,
                        m_tile * state_block_m : (m_tile + 1) * state_block_m,
                    ],
                    x_shared,
                )
                T.copy(
                    dt[batch_idx, token_start : token_start + BLOCK_K, head],
                    dt_shared,
                )
                T.copy(x_shared, x_local)
                T.copy(dt_shared, dt_local)
                for m, k in T.Parallel(state_block_m, BLOCK_K):
                    scaled_x_t[m, k] = (
                        x_local[k, m]
                        * T.exp2(
                            (
                                prefix_all_shared[CHUNK - 1]
                                - prefix_all_shared[k_tile * BLOCK_K + k]
                            )
                            * LOG2E
                        )
                        * dt_local[k]
                    )
                T.copy(
                    B[
                        batch_idx,
                        token_start : token_start + BLOCK_K,
                        group,
                        n_tile * STATE_BLOCK_N : (n_tile + 1) * STATE_BLOCK_N,
                    ],
                    b_shared,
                )
                T.gemm(scaled_x_t, b_shared, state_acc)

            T.copy(state_acc, state_shared)
            T.copy(
                state_shared,
                chunk_state[
                    batch_idx,
                    chunk,
                    head,
                    m_tile * state_block_m : (m_tile + 1) * state_block_m,
                    n_tile * STATE_BLOCK_N : (n_tile + 1) * STATE_BLOCK_N,
                ],
            )

    return kernel


@tilelang.jit(pass_configs=_BASELINE_PASSES)
def _state_passing(batch, seqlen, heads, headdim, dstate):
    chunks = (seqlen + CHUNK - 1) // CHUNK
    elements = batch * heads * headdim * dstate

    @T.prim_func
    def kernel(
        chunk_state: T.Tensor((batch, chunks, heads, headdim, dstate), T.bfloat16),
        chunk_decay: T.Tensor((batch, heads, chunks), T.float32),
        entering_state: T.Tensor((batch, chunks, heads, headdim, dstate), T.bfloat16),
    ):
        with T.Kernel(T.ceildiv(elements, 256), threads=256) as block:
            thread = T.get_thread_binding()
            linear = block * 256 + thread
            if linear < elements:
                state_col = linear % dstate
                state_row = (linear // dstate) % headdim
                head = (linear // (dstate * headdim)) % heads
                batch_idx = linear // (dstate * headdim * heads)
                carry = T.alloc_local((1,), T.float32)
                carry[0] = 0.0
                for chunk in T.serial(chunks):
                    entering_state[batch_idx, chunk, head, state_row, state_col] = carry[0]
                    carry[0] = (
                        chunk_decay[batch_idx, head, chunk] * carry[0]
                        + chunk_state[batch_idx, chunk, head, state_row, state_col]
                    )

    return kernel


@tilelang.jit(pass_configs=_BASELINE_PASSES)
def _chunk_scan(batch, seqlen, heads, headdim, groups, dstate):
    chunks = (seqlen + CHUNK - 1) // CHUNK

    @T.prim_func
    def kernel(
        x: T.Tensor((batch, seqlen, heads, headdim), T.bfloat16),
        dt: T.Tensor((batch, seqlen, heads), T.float32),
        A: T.Tensor((heads,), T.float32),
        B: T.Tensor((batch, seqlen, groups, dstate), T.bfloat16),
        C: T.Tensor((batch, seqlen, groups, dstate), T.bfloat16),
        D: T.Tensor((heads,), T.float32),
        entering_state: T.Tensor((batch, chunks, heads, headdim, dstate), T.bfloat16),
        output: T.Tensor((batch, seqlen, heads, headdim), T.bfloat16),
    ):
        with T.Kernel(
            heads,
            (CHUNK // BLOCK_M) * T.ceildiv(headdim, BLOCK_N),
            batch * chunks,
            threads=THREADS,
        ) as (head, tile, batch_chunk):
            c_shared = T.alloc_shared((BLOCK_M, dstate), T.bfloat16)
            b_shared = T.alloc_shared((BLOCK_K, dstate), T.bfloat16)
            state_shared = T.alloc_shared((BLOCK_N, dstate), T.bfloat16)
            x_shared = T.alloc_shared((BLOCK_K, BLOCK_N), T.bfloat16)
            x_residual_shared = T.alloc_shared((BLOCK_M, BLOCK_N), T.bfloat16)
            dt_shared = T.alloc_shared((BLOCK_K,), T.float32)

            dt_k = T.alloc_fragment((BLOCK_K,), T.float32)
            values = T.alloc_fragment((CHUNK,), T.float32)
            prefix_all_shared = T.alloc_shared((CHUNK,), T.float32)
            cb_acc = T.alloc_fragment((BLOCK_M, BLOCK_K), T.float32)
            scores = T.alloc_fragment((BLOCK_M, BLOCK_K), T.bfloat16)
            output_acc = T.alloc_fragment((BLOCK_M, BLOCK_N), T.float32)
            output_shared = T.alloc_shared((BLOCK_M, BLOCK_N), T.bfloat16)

            batch_idx = batch_chunk % batch
            chunk = batch_chunk // batch
            m_tile = tile // T.ceildiv(headdim, BLOCK_N)
            p_tile = tile % T.ceildiv(headdim, BLOCK_N)
            group = head // (heads // groups)
            query_start = chunk * CHUNK + m_tile * BLOCK_M

            for row in T.Parallel(CHUNK):
                token = chunk * CHUNK + row
                values[row] = T.if_then_else(
                    token < seqlen,
                    A[head] * dt[batch_idx, token, head],
                    0.0,
                )
            T.cumsum(values)
            T.copy(values, prefix_all_shared)
            T.clear(output_acc)

            # Keep the full C tile on chip for the state projection and all CB products.
            T.copy(
                C[batch_idx, query_start : query_start + BLOCK_M, group, 0:dstate],
                c_shared,
            )
            T.copy(
                entering_state[
                    batch_idx,
                    chunk,
                    head,
                    p_tile * BLOCK_N : (p_tile + 1) * BLOCK_N,
                    0:dstate,
                ],
                state_shared,
            )
            T.gemm(c_shared, state_shared, output_acc, transpose_B=True)

            for m, n in T.Parallel(BLOCK_M, BLOCK_N):
                output_acc[m, n] = output_acc[m, n] * T.exp2(
                    prefix_all_shared[m_tile * BLOCK_M + m] * LOG2E
                )

            # Form one C B^T tile, apply causal decay/dt, then multiply by x.
            for k_tile in T.Pipelined(CHUNK // BLOCK_K, num_stages=2):
                key_start = chunk * CHUNK + k_tile * BLOCK_K
                T.clear(cb_acc)
                T.copy(
                    B[batch_idx, key_start : key_start + BLOCK_K, group, 0:dstate],
                    b_shared,
                )
                T.gemm(c_shared, b_shared, cb_acc, transpose_B=True)

                T.copy(
                    dt[batch_idx, key_start : key_start + BLOCK_K, head],
                    dt_shared,
                )
                T.copy(dt_shared, dt_k)
                for m, k in T.Parallel(BLOCK_M, BLOCK_K):
                    scores[m, k] = T.if_then_else(
                        m_tile * BLOCK_M + m >= k_tile * BLOCK_K + k,
                        cb_acc[m, k]
                        * T.exp2(
                            (
                                prefix_all_shared[m_tile * BLOCK_M + m]
                                - prefix_all_shared[k_tile * BLOCK_K + k]
                            )
                            * LOG2E
                        )
                        * dt_k[k],
                        0.0,
                    )
                T.copy(
                    x[
                        batch_idx,
                        key_start : key_start + BLOCK_K,
                        head,
                        p_tile * BLOCK_N : (p_tile + 1) * BLOCK_N,
                    ],
                    x_shared,
                )
                T.gemm(scores, x_shared, output_acc)

            T.copy(
                x[
                    batch_idx,
                    query_start : query_start + BLOCK_M,
                    head,
                    p_tile * BLOCK_N : (p_tile + 1) * BLOCK_N,
                ],
                x_residual_shared,
            )
            for m, n in T.Parallel(BLOCK_M, BLOCK_N):
                output_acc[m, n] = output_acc[m, n] + D[head] * x_residual_shared[m, n]

            T.copy(output_acc, output_shared)
            T.copy(
                output_shared,
                output[
                    batch_idx,
                    query_start : query_start + BLOCK_M,
                    head,
                    p_tile * BLOCK_N : (p_tile + 1) * BLOCK_N,
                ],
            )

    return kernel


@lru_cache(maxsize=None)
def _compile(shape):
    batch, seqlen, heads, headdim, groups, dstate = shape
    return (
        _chunk_state.compile(
            batch=batch,
            seqlen=seqlen,
            heads=heads,
            headdim=headdim,
            groups=groups,
            dstate=dstate,
        ),
        _state_passing.compile(
            batch=batch,
            seqlen=seqlen,
            heads=heads,
            headdim=headdim,
            dstate=dstate,
        ),
        _chunk_scan.compile(
            batch=batch,
            seqlen=seqlen,
            heads=heads,
            headdim=headdim,
            groups=groups,
            dstate=dstate,
        ),
    )


def _shape_and_type_check(x, dt, A, B, C, D, out):
    if x.ndim != 4 or B.ndim != 4:
        raise ValueError("x and B must be rank four")
    batch, seqlen, heads, headdim = map(int, x.shape)
    groups, dstate = map(int, B.shape[2:])
    if seqlen <= 0:
        raise ValueError("seqlen must be positive")
    if tuple(dt.shape) != (batch, seqlen, heads):
        raise ValueError("dt shape mismatch")
    if tuple(A.shape) != (heads,) or tuple(D.shape) != (heads,):
        raise ValueError("A or D shape mismatch")
    if tuple(B.shape) != (batch, seqlen, groups, dstate) or tuple(C.shape) != tuple(B.shape):
        raise ValueError("B or C shape mismatch")
    if tuple(out.shape) != tuple(x.shape):
        raise ValueError("out shape mismatch")
    if heads % groups:
        raise ValueError("heads must be divisible by groups")
    if any(t.dtype != torch.bfloat16 for t in (x, B, C, out)):
        raise TypeError("x, B, C, and out must be bfloat16")
    if any(t.dtype != torch.float32 for t in (dt, A, D)):
        raise TypeError("dt, A, and D must be float32")
    tensors = (x, dt, A, B, C, D, out)
    if not all(t.is_cuda and t.is_contiguous() and t.device == x.device for t in tensors):
        raise ValueError("all tensors must be contiguous on the same CUDA device")
    return batch, seqlen, heads, headdim, groups, dstate


def ssd_forward(x, dt, A, B, C, D, out) -> None:
    """Write the complete SSD forward result into caller-allocated ``out``."""
    shape = _shape_and_type_check(x, dt, A, B, C, D, out)
    batch, seqlen, heads, headdim, groups, dstate = shape
    chunks = (seqlen + CHUNK - 1) // CHUNK
    device = x.device

    chunk_decay = torch.empty((batch, heads, chunks), device=device, dtype=torch.float32)
    chunk_state = torch.empty(
        (batch, chunks, heads, headdim, dstate), device=device, dtype=torch.bfloat16
    )
    entering_state = torch.empty(
        (batch, chunks, heads, headdim, dstate), device=device, dtype=torch.bfloat16
    )

    state_kernel, passing_kernel, scan_kernel = _compile(shape)
    state_kernel(x, dt, A, B, chunk_state, chunk_decay)
    passing_kernel(chunk_state, chunk_decay, entering_state)
    scan_kernel(x, dt, A, B, C, D, entering_state, out)


__all__ = ["ssd_forward"]
