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


CHUNK = 256
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 64
THREADS = 128

# Keep this baseline out of TileLang's automatic Hopper warp-specialised path.
_BASELINE_PASSES = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}


@tilelang.jit(pass_configs=_BASELINE_PASSES)
def _chunk_cumsum_decay(batch, seqlen, heads):
    chunks = (seqlen + CHUNK - 1) // CHUNK

    @T.prim_func
    def kernel(
        dt: T.Tensor((batch, seqlen, heads), T.float32),
        A: T.Tensor((heads,), T.float32),
        log_prefix: T.Tensor((batch, heads, chunks, CHUNK), T.float32),
        chunk_decay: T.Tensor((batch, heads, chunks), T.float32),
    ):
        with T.Kernel(chunks, heads, batch, threads=256) as (chunk, head, batch_idx):
            values = T.alloc_fragment((CHUNK,), T.float32)

            for row in T.Parallel(CHUNK):
                token = chunk * CHUNK + row
                values[row] = T.if_then_else(
                    token < seqlen,
                    A[head] * dt[batch_idx, token, head],
                    0.0,
                )

            T.cumsum(values)

            for row in T.Parallel(CHUNK):
                log_prefix[batch_idx, head, chunk, row] = values[row]
            chunk_decay[batch_idx, head, chunk] = T.exp(values[CHUNK - 1])

    return kernel


@tilelang.jit(pass_configs=_BASELINE_PASSES)
def _chunk_state(batch, seqlen, heads, headdim, groups, dstate):
    chunks = (seqlen + CHUNK - 1) // CHUNK

    @T.prim_func
    def kernel(
        x: T.Tensor((batch, seqlen, heads, headdim), T.bfloat16),
        dt: T.Tensor((batch, seqlen, heads), T.float32),
        B: T.Tensor((batch, seqlen, groups, dstate), T.bfloat16),
        log_prefix: T.Tensor((batch, heads, chunks, CHUNK), T.float32),
        chunk_state: T.Tensor((batch, chunks, heads, headdim, dstate), T.float32),
    ):
        with T.Kernel(
            heads,
            T.ceildiv(headdim, BLOCK_M) * T.ceildiv(dstate, BLOCK_N),
            batch * chunks,
            threads=THREADS,
        ) as (head, tile, batch_chunk):
            x_shared = T.alloc_shared((BLOCK_K, BLOCK_M), T.bfloat16)
            x_local = T.alloc_fragment((BLOCK_K, BLOCK_M), T.bfloat16)
            scaled_x_t = T.alloc_fragment((BLOCK_M, BLOCK_K), T.bfloat16)
            b_shared = T.alloc_shared((BLOCK_K, BLOCK_N), T.bfloat16)
            dt_shared = T.alloc_shared((BLOCK_K,), T.float32)
            prefix_shared = T.alloc_shared((BLOCK_K,), T.float32)
            dt_local = T.alloc_fragment((BLOCK_K,), T.float32)
            prefix_local = T.alloc_fragment((BLOCK_K,), T.float32)
            state_acc = T.alloc_fragment((BLOCK_M, BLOCK_N), T.float32)
            state_shared = T.alloc_shared((BLOCK_M, BLOCK_N), T.float32)

            batch_idx = batch_chunk % batch
            chunk = batch_chunk // batch
            m_tile = tile // T.ceildiv(dstate, BLOCK_N)
            n_tile = tile % T.ceildiv(dstate, BLOCK_N)
            group = head // (heads // groups)
            chunk_log = log_prefix[batch_idx, head, chunk, CHUNK - 1]

            T.clear(state_acc)
            for k_tile in T.Pipelined(CHUNK // BLOCK_K, num_stages=2):
                token_start = chunk * CHUNK + k_tile * BLOCK_K
                T.copy(
                    x[
                        batch_idx,
                        token_start : token_start + BLOCK_K,
                        head,
                        m_tile * BLOCK_M : (m_tile + 1) * BLOCK_M,
                    ],
                    x_shared,
                )
                T.copy(
                    dt[batch_idx, token_start : token_start + BLOCK_K, head],
                    dt_shared,
                )
                T.copy(
                    log_prefix[
                        batch_idx,
                        head,
                        chunk,
                        k_tile * BLOCK_K : (k_tile + 1) * BLOCK_K,
                    ],
                    prefix_shared,
                )
                T.copy(x_shared, x_local)
                T.copy(dt_shared, dt_local)
                T.copy(prefix_shared, prefix_local)
                for m, k in T.Parallel(BLOCK_M, BLOCK_K):
                    scaled_x_t[m, k] = (
                        x_local[k, m]
                        * T.exp(chunk_log - prefix_local[k])
                        * dt_local[k]
                    )
                T.copy(
                    B[
                        batch_idx,
                        token_start : token_start + BLOCK_K,
                        group,
                        n_tile * BLOCK_N : (n_tile + 1) * BLOCK_N,
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
                    m_tile * BLOCK_M : (m_tile + 1) * BLOCK_M,
                    n_tile * BLOCK_N : (n_tile + 1) * BLOCK_N,
                ],
            )

    return kernel


@tilelang.jit(pass_configs=_BASELINE_PASSES)
def _state_passing(batch, seqlen, heads, headdim, dstate):
    chunks = (seqlen + CHUNK - 1) // CHUNK
    elements = batch * heads * headdim * dstate

    @T.prim_func
    def kernel(
        chunk_state: T.Tensor((batch, chunks, heads, headdim, dstate), T.float32),
        chunk_decay: T.Tensor((batch, heads, chunks), T.float32),
        entering_state: T.Tensor((batch, chunks, heads, headdim, dstate), T.float32),
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
        B: T.Tensor((batch, seqlen, groups, dstate), T.bfloat16),
        C: T.Tensor((batch, seqlen, groups, dstate), T.bfloat16),
        D: T.Tensor((heads,), T.float32),
        log_prefix: T.Tensor((batch, heads, chunks, CHUNK), T.float32),
        entering_state: T.Tensor((batch, chunks, heads, headdim, dstate), T.float32),
        output: T.Tensor((batch, seqlen, heads, headdim), T.bfloat16),
    ):
        with T.Kernel(
            heads,
            (CHUNK // BLOCK_M) * T.ceildiv(headdim, BLOCK_N),
            batch * chunks,
            threads=THREADS,
        ) as (head, tile, batch_chunk):
            c_shared = T.alloc_shared((BLOCK_M, BLOCK_K), T.bfloat16)
            b_shared = T.alloc_shared((BLOCK_K, BLOCK_K), T.bfloat16)
            state_shared = T.alloc_shared((BLOCK_N, BLOCK_K), T.bfloat16)
            x_shared = T.alloc_shared((BLOCK_K, BLOCK_N), T.bfloat16)
            x_residual_shared = T.alloc_shared((BLOCK_M, BLOCK_N), T.bfloat16)
            prefix_m_shared = T.alloc_shared((BLOCK_M,), T.float32)
            prefix_k_shared = T.alloc_shared((BLOCK_K,), T.float32)
            dt_shared = T.alloc_shared((BLOCK_K,), T.float32)

            prefix_m = T.alloc_fragment((BLOCK_M,), T.float32)
            prefix_k = T.alloc_fragment((BLOCK_K,), T.float32)
            dt_k = T.alloc_fragment((BLOCK_K,), T.float32)
            cb_acc = T.alloc_fragment((BLOCK_M, BLOCK_K), T.float32)
            scores = T.alloc_fragment((BLOCK_M, BLOCK_K), T.bfloat16)
            output_acc = T.alloc_fragment((BLOCK_M, BLOCK_N), T.float32)
            output_shared = T.alloc_shared((BLOCK_M, BLOCK_N), T.bfloat16)
            residual = T.alloc_fragment((BLOCK_M, BLOCK_N), T.float32)

            batch_idx = batch_chunk % batch
            chunk = batch_chunk // batch
            m_tile = tile // T.ceildiv(headdim, BLOCK_N)
            p_tile = tile % T.ceildiv(headdim, BLOCK_N)
            group = head // (heads // groups)
            query_start = chunk * CHUNK + m_tile * BLOCK_M

            T.copy(
                log_prefix[
                    batch_idx,
                    head,
                    chunk,
                    m_tile * BLOCK_M : (m_tile + 1) * BLOCK_M,
                ],
                prefix_m_shared,
            )
            T.copy(prefix_m_shared, prefix_m)
            T.clear(output_acc)

            # Project the state entering this chunk into each output row.
            for state_tile in T.serial(T.ceildiv(dstate, BLOCK_K)):
                T.copy(
                    C[
                        batch_idx,
                        query_start : query_start + BLOCK_M,
                        group,
                        state_tile * BLOCK_K : (state_tile + 1) * BLOCK_K,
                    ],
                    c_shared,
                )
                T.copy(
                    entering_state[
                        batch_idx,
                        chunk,
                        head,
                        p_tile * BLOCK_N : (p_tile + 1) * BLOCK_N,
                        state_tile * BLOCK_K : (state_tile + 1) * BLOCK_K,
                    ],
                    state_shared,
                )
                T.gemm(c_shared, state_shared, output_acc, transpose_B=True)

            for m, n in T.Parallel(BLOCK_M, BLOCK_N):
                output_acc[m, n] = output_acc[m, n] * T.exp(prefix_m[m])

            # Form one C B^T tile, apply causal decay/dt, then multiply by x.
            for k_tile in T.serial(CHUNK // BLOCK_K):
                key_start = chunk * CHUNK + k_tile * BLOCK_K
                T.clear(cb_acc)
                for state_tile in T.serial(T.ceildiv(dstate, BLOCK_K)):
                    T.copy(
                        C[
                            batch_idx,
                            query_start : query_start + BLOCK_M,
                            group,
                            state_tile * BLOCK_K : (state_tile + 1) * BLOCK_K,
                        ],
                        c_shared,
                    )
                    T.copy(
                        B[
                            batch_idx,
                            key_start : key_start + BLOCK_K,
                            group,
                            state_tile * BLOCK_K : (state_tile + 1) * BLOCK_K,
                        ],
                        b_shared,
                    )
                    T.gemm(c_shared, b_shared, cb_acc, transpose_B=True)

                T.copy(
                    log_prefix[
                        batch_idx,
                        head,
                        chunk,
                        k_tile * BLOCK_K : (k_tile + 1) * BLOCK_K,
                    ],
                    prefix_k_shared,
                )
                T.copy(
                    dt[batch_idx, key_start : key_start + BLOCK_K, head],
                    dt_shared,
                )
                T.copy(prefix_k_shared, prefix_k)
                T.copy(dt_shared, dt_k)
                for m, k in T.Parallel(BLOCK_M, BLOCK_K):
                    scores[m, k] = T.if_then_else(
                        m_tile * BLOCK_M + m >= k_tile * BLOCK_K + k,
                        cb_acc[m, k]
                        * T.exp(prefix_m[m] - prefix_k[k])
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
            T.copy(x_residual_shared, residual)
            for m, n in T.Parallel(BLOCK_M, BLOCK_N):
                output_acc[m, n] = output_acc[m, n] + D[head] * residual[m, n]

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
        _chunk_cumsum_decay.compile(batch=batch, seqlen=seqlen, heads=heads),
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

    # Deliberately materialised baseline intermediates.
    log_prefix = torch.empty((batch, heads, chunks, CHUNK), device=device, dtype=torch.float32)
    chunk_decay = torch.empty((batch, heads, chunks), device=device, dtype=torch.float32)
    chunk_state = torch.empty(
        (batch, chunks, heads, headdim, dstate), device=device, dtype=torch.float32
    )
    entering_state = torch.empty(
        (batch, chunks, heads, headdim, dstate), device=device, dtype=torch.float32
    )

    cumsum_kernel, state_kernel, passing_kernel, scan_kernel = _compile(shape)
    cumsum_kernel(dt, A, log_prefix, chunk_decay)
    state_kernel(x, dt, B, log_prefix, chunk_state)
    passing_kernel(chunk_state, chunk_decay, entering_state)
    scan_kernel(x, dt, B, C, D, log_prefix, entering_state, out)


__all__ = ["ssd_forward"]
