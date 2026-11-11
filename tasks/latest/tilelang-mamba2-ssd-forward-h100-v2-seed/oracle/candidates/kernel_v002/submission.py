"""Paper-reference Mamba-2 SSD forward, v003: v001 + offline-tuned static configs.

Device code is v001's (upstream SSD decomposition of state-spaces/mamba @ e9594ce1
``_mamba_chunk_scan_combined_fwd``; chunk-state and chunk-scan bodies from
upstream TileLang @ f935b2a).  What v003 adds is only what upstream's own
benchmark does and v001 skipped: upstream ``benchmark/mamba2`` and
``example_mamba_chunk_state.py`` pick their tile configs with
``tilelang.autotune`` over a fixed search space.  Autotuning inside a submission
is forbidden, but POLICY.md allows "offline autotuning in Work scratch scripts
followed by bounded, static shape-based kernel choices", so the choices live in
the literal table ``TUNED`` below, written by ``tools/tune_paper_configs.py``
(run on the H100 in the Work image) and baked in by ``tools/bake_tuned.py``.

The tuned dimensions are upstream's own:
  * chunk_scan  block_M x block_N x block_K x num_stages (benchmark get_configs;
    block_Dstate = dstate, since upstream's fixed 128 would drop half of
    Falcon-H1's 256-wide state)
  * chunk_state block_M x block_N x block_K x num_stages (example get_configs)
plus two choices where the two upstreams disagree, so the faster is taken:
  * chunk size 256 (upstream TileLang benchmark, Mamba2 module default) or 128
    (Falcon-H1's shipped mamba_chunk_size)
  * CB dtype FP32 (mamba_ssm ``_bmm_chunk_fwd(output_dtype=float32)``) or BF16
    (upstream TileLang: CB in the same 16-bit dtype as x).  The score
    elementwise step is evaluated in FP32 either way.

With the default table (no tuning run) v003 computes exactly what v001 does.
"""

from functools import lru_cache

import torch
import tilelang
import tilelang.language as T


LOG2E = 1.44269504  # upstream's ``p``

# v001 / upstream non-autotuned defaults
DEFAULT = {
    "chunk": 256,
    "cb_bf16": False,
    "state": {"block_M": 64, "block_N": 128, "block_K": 64, "num_stages": 4, "threads": 128},
    "scan": {"block_M": 64, "block_N": 64, "block_K": 64, "num_stages": 2, "threads": 128},
}

# Offline-tuned choices keyed by "heads,headdim,dstate,groups" (geometry) and,
# optionally, "heads,headdim,dstate,groups,seqlen" (exact shape, takes
# precedence).  Missing keys fall back to DEFAULT.  BEGIN/END markers are
# rewritten by tools/bake_tuned.py.
# BEGIN TUNED
TUNED = {'80,64,128,1': {'chunk': 256, 'cb_bf16': True, 'state': {'block_M': 64, 'block_N': 128, 'block_K': 64, 'num_stages': 3, 'threads': 128}, 'scan': {'block_M': 64, 'block_N': 64, 'block_K': 64, 'num_stages': 2, 'threads': 128}}, '128,64,128,8': {'chunk': 256, 'cb_bf16': True, 'state': {'block_M': 64, 'block_N': 128, 'block_K': 64, 'num_stages': 3, 'threads': 128}, 'scan': {'block_M': 64, 'block_N': 64, 'block_K': 64, 'num_stages': 2, 'threads': 128}}, '32,128,256,2': {'chunk': 256, 'cb_bf16': False, 'state': {'block_M': 128, 'block_N': 128, 'block_K': 64, 'num_stages': 2, 'threads': 128}, 'scan': {'block_M': 128, 'block_N': 64, 'block_K': 64, 'num_stages': 2, 'threads': 128}}}
# END TUNED

_FAST_MATH = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}


def _choice(heads, headdim, dstate, groups, seqlen):
    geo = f"{heads},{headdim},{dstate},{groups}"
    pick = TUNED.get(f"{geo},{seqlen}")
    if pick is None:
        pick = TUNED.get(geo)
    if pick is None:
        pick = DEFAULT
    chunk = int(pick.get("chunk", DEFAULT["chunk"]))
    cb_bf16 = bool(pick.get("cb_bf16", DEFAULT["cb_bf16"]))
    state = dict(DEFAULT["state"])
    state.update(pick.get("state", {}))
    scan = dict(DEFAULT["scan"])
    scan.update(pick.get("scan", {}))
    return chunk, cb_bf16, state, scan


# ---------------------------------------------------------------- stage 1
def _cumsum_program(batch, seqlen, heads, chunk):
    nchunks = (seqlen + chunk - 1) // chunk

    @T.prim_func
    def main(
        dt: T.Tensor((batch, seqlen, heads), T.float32),
        A: T.Tensor((heads,), T.float32),
        dt_out: T.Tensor((batch, heads, nchunks, chunk), T.float32),
        dA_cumsum: T.Tensor((batch, heads, nchunks, chunk), T.float32),
    ):
        with T.Kernel(nchunks, heads, batch, threads=256) as (c, h, b):
            dt_f = T.alloc_fragment((chunk,), T.float32)
            dA_f = T.alloc_fragment((chunk,), T.float32)
            for i in T.Parallel(chunk):
                dt_f[i] = T.if_then_else(c * chunk + i < seqlen, dt[b, c * chunk + i, h], 0.0)
            for i in T.Parallel(chunk):
                dA_f[i] = dt_f[i] * A[h]
            T.cumsum(dA_f)
            for i in T.Parallel(chunk):
                dt_out[b, h, c, i] = dt_f[i]
                dA_cumsum[b, h, c, i] = dA_f[i]

    return main


# ---------------------------------------------------------------- stage 2 (upstream TileLang)
def _chunk_state_program(batch, seqlen, chunk_size, ngroups, nheads, headdim, dstate,
                         block_M=64, block_N=64, block_K=64, num_stages=2, threads=128):
    dtype = T.bfloat16
    accum_dtype = T.float32
    nchunks = T.ceildiv(seqlen, chunk_size)
    p = 1.44269504

    @T.prim_func
    def main(
        B: T.Tensor((batch, seqlen, ngroups, dstate), dtype),
        x: T.Tensor((batch, seqlen, nheads, headdim), dtype),
        dt: T.Tensor((batch, nheads, nchunks, chunk_size), accum_dtype),
        dA_cumsum: T.Tensor((batch, nheads, nchunks, chunk_size), accum_dtype),
        Output: T.Tensor((batch, nchunks, nheads, headdim, dstate), accum_dtype),
    ):
        with T.Kernel(nheads, T.ceildiv(headdim, block_M) * T.ceildiv(dstate, block_N), batch * nchunks, threads=threads) as (bz, bx, by):
            x_shared = T.alloc_shared((block_K, block_M), dtype)
            x_local = T.alloc_fragment((block_K, block_M), dtype)
            xt_local = T.alloc_fragment((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            dt_shared = T.alloc_shared((block_K), accum_dtype)
            dA_cumsum_shared = T.alloc_shared((block_K), accum_dtype)
            acc_o = T.alloc_fragment((block_M, block_N), accum_dtype)
            acc_o_shared = T.alloc_shared((block_M, block_N), accum_dtype)
            scale = T.alloc_fragment((block_K), accum_dtype)
            dA_cs_last = T.alloc_fragment((1), accum_dtype)
            dA_cumsum_local = T.alloc_fragment((block_K), accum_dtype)
            dt_local = T.alloc_fragment((block_K), accum_dtype)

            loop_range = T.ceildiv(chunk_size, block_K)

            batch_idx = by % batch
            chunk_idx = by // batch
            m_idx = bx // T.ceildiv(dstate, block_N)
            n_idx = bx % T.ceildiv(dstate, block_N)

            T.annotate_layout({x_shared: tilelang.layout.make_swizzled_layout(x_shared)})

            dA_cs_last[0] = dA_cumsum[batch_idx, bz, chunk_idx, chunk_size - 1]
            T.clear(acc_o)
            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(
                    x[
                        batch_idx,
                        chunk_idx * chunk_size + k * block_K : chunk_idx * chunk_size + (k + 1) * block_K,
                        bz,
                        m_idx * block_M : (m_idx + 1) * block_M,
                    ],
                    x_shared,
                )
                T.copy(dA_cumsum[batch_idx, bz, chunk_idx, k * block_K : (k + 1) * block_K], dA_cumsum_shared)
                T.copy(dt[batch_idx, bz, chunk_idx, k * block_K : (k + 1) * block_K], dt_shared)
                T.copy(dA_cumsum_shared, dA_cumsum_local)
                T.copy(dt_shared, dt_local)
                for i in T.Parallel(block_K):
                    scale[i] = T.exp2(dA_cs_last[0] * p - dA_cumsum_local[i] * p) * dt_local[i]
                T.copy(x_shared, x_local)
                for i, j in T.Parallel(block_M, block_K):
                    xt_local[i, j] = x_local[j, i] * scale[j]
                T.copy(
                    B[
                        batch_idx,
                        chunk_idx * chunk_size + k * block_K : chunk_idx * chunk_size + (k + 1) * block_K,
                        bz // (nheads // ngroups),
                        n_idx * block_N : (n_idx + 1) * block_N,
                    ],
                    B_shared,
                )
                T.gemm(xt_local, B_shared, acc_o)
            T.copy(acc_o, acc_o_shared)
            T.copy(
                acc_o_shared,
                Output[batch_idx, chunk_idx, bz, m_idx * block_M : (m_idx + 1) * block_M, n_idx * block_N : (n_idx + 1) * block_N],
            )

    return main


# ---------------------------------------------------------------- stage 3
def _passing_program(batch, nchunks, heads, headdim, dstate, chunk):
    rows = max(1, 2048 // dstate)  # 2048 state elements per CTA

    @T.prim_func
    def main(
        states: T.Tensor((batch, nchunks, heads, headdim, dstate), T.float32),
        dA_cumsum: T.Tensor((batch, heads, nchunks, chunk), T.float32),
        prev_states: T.Tensor((batch, nchunks, heads, headdim, dstate), T.bfloat16),
    ):
        with T.Kernel(T.ceildiv(headdim, rows), heads, batch, threads=128) as (pt, h, b):
            carry = T.alloc_fragment((rows, dstate), T.float32)
            decay = T.alloc_local((1,), T.float32)
            T.clear(carry)
            for c in T.serial(nchunks):
                decay[0] = T.exp(dA_cumsum[b, h, c, chunk - 1])
                for i, j in T.Parallel(rows, dstate):
                    prev_states[b, c, h, pt * rows + i, j] = carry[i, j]
                    carry[i, j] = decay[0] * carry[i, j] + states[b, c, h, pt * rows + i, j]

    return main


# ---------------------------------------------------------------- stage 4
def _bmm_program(batch, seqlen, groups, dstate, chunk, cb_bf16, block=64, block_k=64):
    nchunks = (seqlen + chunk - 1) // chunk
    tiles = chunk // block
    cb_dtype = T.bfloat16 if cb_bf16 else T.float32

    @T.prim_func
    def main(
        C: T.Tensor((batch, seqlen, groups, dstate), T.bfloat16),
        B: T.Tensor((batch, seqlen, groups, dstate), T.bfloat16),
        CB: T.Tensor((batch, nchunks, groups, chunk, chunk), cb_dtype),
    ):
        with T.Kernel(groups, tiles * tiles, batch * nchunks, threads=128) as (g, t, bc):
            c_shared = T.alloc_shared((block, block_k), T.bfloat16)
            b_shared = T.alloc_shared((block, block_k), T.bfloat16)
            acc = T.alloc_fragment((block, block), T.float32)
            acc_shared = T.alloc_shared((block, block), cb_dtype)
            b = bc % batch
            c = bc // batch
            mi = t // tiles
            ni = t % tiles
            T.clear(acc)
            for k in T.Pipelined(T.ceildiv(dstate, block_k), num_stages=2):
                T.copy(C[b, c * chunk + mi * block : c * chunk + (mi + 1) * block, g, k * block_k : (k + 1) * block_k], c_shared)
                T.copy(B[b, c * chunk + ni * block : c * chunk + (ni + 1) * block, g, k * block_k : (k + 1) * block_k], b_shared)
                T.gemm(c_shared, b_shared, acc, transpose_B=True)
            T.copy(acc, acc_shared)
            T.copy(acc_shared, CB[b, c, g, mi * block : (mi + 1) * block, ni * block : (ni + 1) * block])

    return main


# ---------------------------------------------------------------- stage 5 (upstream TileLang)
def _chunk_scan_program(batch, seqlen, chunk_size, ngroups, nheads, headdim, dstate,
                        block_M=64, block_N=64, block_K=64, block_Dstate=128, num_stages=2, threads=128,
                        cb_bf16=False):
    dtype = T.bfloat16
    accum_dtype = T.float32
    cb_dtype = T.bfloat16 if cb_bf16 else T.float32
    nchunks = T.ceildiv(seqlen, chunk_size)
    p = 1.44269504

    @T.prim_func
    def main(
        cb: T.Tensor((batch, nchunks, ngroups, chunk_size, chunk_size), cb_dtype),
        x: T.Tensor((batch, seqlen, nheads, headdim), dtype),
        dt: T.Tensor((batch, nheads, nchunks, chunk_size), accum_dtype),
        dA_cumsum: T.Tensor((batch, nheads, nchunks, chunk_size), accum_dtype),
        C: T.Tensor((batch, seqlen, ngroups, dstate), dtype),
        prev_states: T.Tensor((batch, nchunks, nheads, headdim, dstate), dtype),
        D: T.Tensor((nheads), accum_dtype),
        Output: T.Tensor((batch, seqlen, nheads, headdim), dtype),
    ):
        with T.Kernel(nheads, T.ceildiv(chunk_size, block_M) * T.ceildiv(headdim, block_N), batch * nchunks, threads=threads) as (
            bz,
            bx,
            by,
        ):
            acc_o = T.alloc_fragment((block_M, block_N), accum_dtype)
            acc_o_shared = T.alloc_shared((block_M, block_N), dtype)
            cb_shared = T.alloc_shared((block_M, block_K), cb_dtype)
            cb_local = T.alloc_fragment((block_M, block_K), dtype)
            dA_cs_k_shared = T.alloc_shared((block_K), accum_dtype)
            dA_cs_k_local = T.alloc_fragment((block_K), accum_dtype)
            dA_cs_m_local = T.alloc_fragment((block_M), accum_dtype)
            dt_shared = T.alloc_shared((block_K), accum_dtype)
            dt_local = T.alloc_fragment((block_K), accum_dtype)
            x_shared = T.alloc_shared((block_K, block_N), dtype)
            dA_cs_m_shared = T.alloc_shared((block_M), accum_dtype)
            scale_m_local = T.alloc_fragment((block_M), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_Dstate), dtype)
            prev_state_shared = T.alloc_shared((block_N, block_Dstate), dtype)
            D_local = T.alloc_fragment((1), accum_dtype)
            x_residual_shared = T.alloc_shared((block_M, block_N), dtype)
            x_residual_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            batch_idx = by % batch
            chunk_idx = by // batch
            # m: chunk_size
            # n : headdim
            m_idx = bx // T.ceildiv(headdim, block_N)
            n_idx = bx % T.ceildiv(headdim, block_N)

            T.annotate_layout(
                {
                    cb_shared: tilelang.layout.make_swizzled_layout(cb_shared),
                    x_residual_shared: tilelang.layout.make_swizzled_layout(x_residual_shared),
                }
            )

            T.no_set_max_nreg()

            T.copy(dA_cumsum[batch_idx, bz, chunk_idx, m_idx * block_M : (m_idx + 1) * block_M], dA_cs_m_shared)
            T.copy(dA_cs_m_shared, dA_cs_m_local)
            T.clear(acc_o)

            for i in T.Parallel(block_M):
                scale_m_local[i] = T.exp2(dA_cs_m_local[i] * p)
            T.copy(
                C[
                    batch_idx,
                    chunk_idx * chunk_size + m_idx * block_M : chunk_idx * chunk_size + (m_idx + 1) * block_M,
                    bz // (nheads // ngroups),
                    0:block_Dstate,
                ],
                C_shared,
            )
            T.copy(prev_states[batch_idx, chunk_idx, bz, n_idx * block_N : (n_idx + 1) * block_N, 0:block_Dstate], prev_state_shared)
            T.gemm(C_shared, prev_state_shared, acc_o, transpose_B=True)
            for i, j in T.Parallel(block_M, block_N):
                acc_o[i, j] *= scale_m_local[i]

            loop_range = T.ceildiv((m_idx + 1) * block_M, block_K)

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(
                    cb[
                        batch_idx,
                        chunk_idx,
                        bz // (nheads // ngroups),
                        m_idx * block_M : (m_idx + 1) * block_M,
                        k * block_K : (k + 1) * block_K,
                    ],
                    cb_shared,
                )
                T.copy(dA_cumsum[batch_idx, bz, chunk_idx, k * block_K : (k + 1) * block_K], dA_cs_k_shared)
                T.copy(dA_cs_k_shared, dA_cs_k_local)
                T.copy(dt[batch_idx, bz, chunk_idx, k * block_K : (k + 1) * block_K], dt_shared)
                T.copy(dt_shared, dt_local)
                # upstream: three 16-bit-fragment passes (decay, *dt, causal mask);
                # here one FP32 pass from the CB tile, rounded to BF16 once.
                for i, j in T.Parallel(block_M, block_K):
                    cb_local[i, j] = T.if_then_else(
                        m_idx * block_M + i >= k * block_K + j,
                        T.Cast(accum_dtype, cb_shared[i, j]) * T.exp2(dA_cs_m_local[i] * p - dA_cs_k_local[j] * p) * dt_local[j],
                        0,
                    )
                T.copy(
                    x[
                        batch_idx,
                        chunk_idx * chunk_size + k * block_K : chunk_idx * chunk_size + (k + 1) * block_K,
                        bz,
                        n_idx * block_N : (n_idx + 1) * block_N,
                    ],
                    x_shared,
                )
                T.gemm(cb_local, x_shared, acc_o)

            D_local[0] = D[bz]
            T.copy(
                x[
                    batch_idx,
                    chunk_idx * chunk_size + m_idx * block_M : chunk_idx * chunk_size + (m_idx + 1) * block_M,
                    bz,
                    n_idx * block_N : (n_idx + 1) * block_N,
                ],
                x_residual_shared,
            )
            T.copy(x_residual_shared, x_residual_local)
            for i, j in T.Parallel(block_M, block_N):
                acc_o[i, j] += x_residual_local[i, j] * D_local[0]

            T.copy(acc_o, acc_o_shared)
            T.copy(
                acc_o_shared,
                Output[
                    batch_idx,
                    chunk_idx * chunk_size + m_idx * block_M : chunk_idx * chunk_size + (m_idx + 1) * block_M,
                    bz,
                    n_idx * block_N : (n_idx + 1) * block_N,
                ],
            )

    return main


_cumsum_jit = tilelang.jit()(_cumsum_program)
_chunk_state_jit = tilelang.jit()(_chunk_state_program)
_passing_jit = tilelang.jit()(_passing_program)
_bmm_jit = tilelang.jit()(_bmm_program)
_chunk_scan_jit = tilelang.jit(pass_configs=_FAST_MATH)(_chunk_scan_program)


@lru_cache(maxsize=None)
def _compile(batch, seqlen, heads, headdim, groups, dstate):
    chunk, cb_bf16, state, scan = _choice(heads, headdim, dstate, groups, seqlen)
    nchunks = (seqlen + chunk - 1) // chunk
    kernels = (
        _cumsum_jit(batch, seqlen, heads, chunk),
        _chunk_state_jit(batch, seqlen, chunk, groups, heads, headdim, dstate, **state),
        _passing_jit(batch, nchunks, heads, headdim, dstate, chunk),
        _bmm_jit(batch, seqlen, groups, dstate, chunk, cb_bf16),
        _chunk_scan_jit(batch, seqlen, chunk, groups, heads, headdim, dstate, block_Dstate=dstate,
                        cb_bf16=cb_bf16, **scan),
    )
    return chunk, nchunks, cb_bf16, kernels


def ssd_forward(x, dt, A, B, C, D, out) -> None:
    """Write the complete SSD forward result into caller-allocated ``out``."""
    batch, seqlen, heads, headdim = x.shape
    groups, dstate = B.shape[2], B.shape[3]
    chunk, nchunks, cb_bf16, kernels = _compile(batch, seqlen, heads, headdim, groups, dstate)
    k_cumsum, k_state, k_pass, k_bmm, k_scan = kernels
    dev = x.device

    # intermediates allocated per call, as mamba_ssm does
    dt_c = torch.empty((batch, heads, nchunks, chunk), device=dev, dtype=torch.float32)
    dA_cs = torch.empty((batch, heads, nchunks, chunk), device=dev, dtype=torch.float32)
    states = torch.empty((batch, nchunks, heads, headdim, dstate), device=dev, dtype=torch.float32)
    prev_states = torch.empty((batch, nchunks, heads, headdim, dstate), device=dev, dtype=torch.bfloat16)
    cb = torch.empty((batch, nchunks, groups, chunk, chunk), device=dev,
                     dtype=torch.bfloat16 if cb_bf16 else torch.float32)

    k_cumsum(dt, A, dt_c, dA_cs)
    k_state(B, x, dt_c, dA_cs, states)
    k_pass(states, dA_cs, prev_states)
    k_bmm(C, B, cb)
    k_scan(cb, x, dt_c, dA_cs, C, prev_states, D, out)


__all__ = ["ssd_forward"]
