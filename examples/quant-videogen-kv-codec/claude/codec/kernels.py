"""Triton kernels: interleaved rANS coding of quantized PCA coefficients and the
fused reconstruction GEMM.

rANS layout: a segment is TSEG consecutive tokens of one (batch, head) row.
Its 32 lanes (one warp) each own the tokens congruent to the lane index; step
(t, j) codes component j of token ``base + t * 32 + lane``.  All lanes of a
segment share one 16-bit word stream: at every renormalization the lanes that
need a word read (decoder) or write (encoder) consecutive words in lane order.

The entropy-coded alphabet is {0, +1, -1, ..., +NBASE, -NBASE, ESC} with one
cumulative table per (row, component).  Escaped values (|q| > NBASE) are stored raw as int8 in a side
array, in decode order; each lane finds its byte with a warp ballot, so the
decoder has no data-dependent branch.
State: 32-bit, renormalized to [2^16, 2^32) with 16-bit words; 12-bit probabilities.
"""

import triton
import triton.language as tl



@triton.jit
def _ballot(p):
    return tl.inline_asm_elementwise(
        "{ .reg .pred q; setp.ne.u32 q, $1, 0; vote.sync.ballot.b32 $0, q, 0xffffffff; }",
        "=r,r", [p], dtype=tl.uint32, is_pure=False, pack=1)


@triton.jit
def _popc(v):
    return tl.inline_asm_elementwise(
        "popc.b32 $0, $1;", "=r,r", [v], dtype=tl.uint32, is_pure=True, pack=1)


@triton.jit
def _shfl(v, src):
    return tl.inline_asm_elementwise(
        "shfl.sync.idx.b32 $0, $1, $2, 0x1f, 0xffffffff;",
        "=r,r,r", [v, src], dtype=tl.uint32, is_pure=False, pack=1)


@triton.jit
def _enc_put(x, s, crow, valid, out_base, wptr, NS: tl.constexpr):
    """Encode symbol index ``s`` on lanes ``valid`` with the cumulative table at
    ``crow`` (interior boundaries only: cum[0] = 0 and the last cum = 4096 are
    implicit; ``NS`` is the escape symbol index); words are written below
    ``wptr``.  Returns the new (x, wptr)."""
    c = tl.load(crow + s - 1, mask=valid & (s > 0), other=0).to(tl.int64)
    cn = tl.load(crow + s, mask=valid & (s < NS), other=4096).to(tl.int64)
    f = tl.where(valid, cn - c, 1)
    emit = valid & (x >= (f << 20))
    e32 = emit.to(tl.int32)
    n = tl.sum(e32, axis=0)
    pos = wptr - n + tl.cumsum(e32, axis=0) - e32
    tl.store(out_base + pos, (x & 65535).to(tl.int16), mask=emit)
    x = tl.where(emit, x >> 16, x)
    qt = x // f
    x = tl.where(valid, (qt << 12) + (x - qt * f) + c, x)
    return x, wptr - n


@triton.jit
def recon_kernel(
    sym_ptr, ut_ptr, uscale_ptr, cen_ptr, cscale_ptr, mu_ptr, ids_ptr, out_ptr, step,
    S, D, NCL, ZERO,
    BT: tl.constexpr, BD: tl.constexpr, JT: tl.constexpr,
):
    row = tl.program_id(0)
    toks = tl.program_id(1) * BT + tl.arange(0, BT)
    tmask = toks < S
    dims = tl.arange(0, BD)
    dmask = dims < D
    rows64 = row.to(tl.int64) * S * D
    acc = tl.zeros([BT, BD], tl.float32)
    for j0 in range(0, D, JT):
        js = j0 + tl.arange(0, JT)
        jm = js < D
        sv = tl.load(sym_ptr + rows64 + js[:, None] * S + toks[None, :],
                     mask=jm[:, None] & tmask[None, :], other=ZERO)
        sv = (sv.to(tl.int32) - ZERO).to(tl.bfloat16)
        ut = tl.load(ut_ptr + (row * D + js[:, None]) * D + dims[None, :],
                     mask=jm[:, None] & dmask[None, :], other=0.0).to(tl.bfloat16)
        acc += tl.dot(tl.trans(sv), ut)
    ids = tl.load(ids_ptr + row * S + toks, mask=tmask, other=0).to(tl.int32)
    cen = tl.load(cen_ptr + (row * NCL + ids[:, None]) * D + dims[None, :],
                  mask=tmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
    mu = tl.load(mu_ptr + row * D + dims, mask=dmask, other=0.0)
    cs = tl.load(cscale_ptr + row * D + dims, mask=dmask, other=0.0)
    out = acc * (step * tl.load(uscale_ptr + row)) + cen * cs[None, :] + mu[None, :]
    tl.store(out_ptr + rows64 + toks[:, None] * D + dims[None, :], out.to(tl.bfloat16),
             mask=tmask[:, None] & dmask[None, :])


@triton.jit
def assign_kernel(x_ptr, c_ptr, cn_ptr, ids_ptr, S, D, NCL,
                  BT: tl.constexpr, BD: tl.constexpr, BC: tl.constexpr):
    """ids[row, s] = argmin_c |c|^2 - 2 x.c  (x, c in BF16; FP32 accumulation)."""
    row = tl.program_id(0)
    toks = tl.program_id(1) * BT + tl.arange(0, BT)
    tmask = toks < S
    dims = tl.arange(0, BD)
    dmask = dims < D
    x = tl.load(x_ptr + (row.to(tl.int64) * S + toks[:, None]) * D + dims[None, :],
                mask=tmask[:, None] & dmask[None, :], other=0.0)
    best = tl.full([BT], float("inf"), tl.float32)
    bidx = tl.zeros([BT], tl.int32)
    for c0 in range(0, NCL, BC):
        cs = c0 + tl.arange(0, BC)
        cmask = cs < NCL
        c = tl.load(c_ptr + (row * NCL + cs[:, None]) * D + dims[None, :],
                    mask=cmask[:, None] & dmask[None, :], other=0.0)
        cn = tl.load(cn_ptr + row * NCL + cs, mask=cmask, other=float("inf"))
        score = cn[None, :] - 2.0 * tl.dot(x, tl.trans(c))
        mn = tl.min(score, axis=1)
        am = tl.argmin(score, axis=1).to(tl.int32) + c0
        upd = mn < best
        best = tl.where(upd, mn, best)
        bidx = tl.where(upd, am, bidx)
    tl.store(ids_ptr + row.to(tl.int64) * S + toks, bidx, mask=tmask)


@triton.jit
def cluster_sum_kernel(x_ptr, ids_ptr, part_ptr, cnt_ptr, S, D, NCL, TCH,
                       BT: tl.constexpr, BD: tl.constexpr, BC: tl.constexpr):
    """Partial per-cluster sums and sizes over one token chunk (one-hot GEMM;
    no atomics, so the final reduction over chunks is deterministic)."""
    row = tl.program_id(0)
    cs = tl.program_id(1) * BC + tl.arange(0, BC)
    chunk = tl.program_id(2)
    dims = tl.arange(0, BD)
    dmask = dims < D
    acc = tl.zeros([BC, BD], tl.float32)
    cnt = tl.zeros([BC], tl.float32)
    for t0 in range(0, TCH, BT):
        toks = chunk * TCH + t0 + tl.arange(0, BT)
        tmask = toks < S
        ids = tl.load(ids_ptr + row.to(tl.int64) * S + toks, mask=tmask, other=-1)
        hit = cs[:, None] == ids[None, :].to(tl.int32)
        x = tl.load(x_ptr + (row.to(tl.int64) * S + toks[:, None]) * D + dims[None, :],
                    mask=tmask[:, None] & dmask[None, :], other=0.0)
        acc += tl.dot(hit.to(tl.bfloat16), x)
        cnt += tl.sum(hit.to(tl.float32), axis=1)
    nrows = tl.num_programs(0)
    cmask = cs < NCL
    base = (chunk * nrows + row) * NCL + cs
    tl.store(part_ptr + base[:, None] * D + dims[None, :], acc, mask=cmask[:, None] & dmask[None, :])
    tl.store(cnt_ptr + base, cnt, mask=cmask)


@triton.jit
def _tcq_level(y, k):
    """Nearest level n = 4m + k (units of the TCQ step) to y."""
    return 4.0 * tl.floor((y - k) * 0.25 + 0.5) + k


@triton.jit
def _err(y, k):
    d = y - _tcq_level(y, k)
    return d * d


@triton.jit
def tcq_kernel(coef_ptr, lvl_ptr, par_ptr, bits_ptr, S, D, inv_d, YMAX,
               BT: tl.constexpr):
    """Viterbi search of the 8-state TCQ trellis along each token's D coefficients
    (coef/levels laid out [R, D, S]).  Writes reconstruction levels n (multiples of
    the TCQ step) and the union-codebook parity of the state that coded them."""
    row = tl.program_id(0)
    toks = tl.program_id(1) * BT + tl.arange(0, BT)
    tmask = toks < S
    base = row.to(tl.int64) * D * S
    INF: tl.constexpr = 3.0e38
    c0 = tl.zeros([BT], tl.float32)
    c1 = tl.full([BT], INF, tl.float32)
    c2 = tl.full([BT], INF, tl.float32)
    c3 = tl.full([BT], INF, tl.float32)
    c4 = tl.full([BT], INF, tl.float32)
    c5 = tl.full([BT], INF, tl.float32)
    c6 = tl.full([BT], INF, tl.float32)
    c7 = tl.full([BT], INF, tl.float32)
    for j in range(D):
        y = tl.load(coef_ptr + base + j * S + toks, mask=tmask, other=0.0) * inv_d
        y = tl.minimum(tl.maximum(y, -YMAX), YMAX)
        e0 = _err(y, 0.0)
        e1 = _err(y, 1.0)
        e2 = _err(y, 2.0)
        e3 = _err(y, 3.0)
        a0 = c0 + e0
        b0 = c4 + e2
        a1 = c0 + e2
        b1 = c4 + e0
        a2 = c1 + e1
        b2 = c5 + e3
        a3 = c1 + e3
        b3 = c5 + e1
        a4 = c2 + e2
        b4 = c6 + e0
        a5 = c2 + e0
        b5 = c6 + e2
        a6 = c3 + e3
        b6 = c7 + e1
        a7 = c3 + e1
        b7 = c7 + e3
        packed = ((b0 < a0).to(tl.int32) | ((b1 < a1).to(tl.int32) << 1) | ((b2 < a2).to(tl.int32) << 2)
                  | ((b3 < a3).to(tl.int32) << 3) | ((b4 < a4).to(tl.int32) << 4) | ((b5 < a5).to(tl.int32) << 5)
                  | ((b6 < a6).to(tl.int32) << 6) | ((b7 < a7).to(tl.int32) << 7))
        tl.store(bits_ptr + base + j * S + toks, packed.to(tl.uint8), mask=tmask)
        c0 = tl.minimum(a0, b0)
        c1 = tl.minimum(a1, b1)
        c2 = tl.minimum(a2, b2)
        c3 = tl.minimum(a3, b3)
        c4 = tl.minimum(a4, b4)
        c5 = tl.minimum(a5, b5)
        c6 = tl.minimum(a6, b6)
        c7 = tl.minimum(a7, b7)
    # best final state
    best = c0
    st = tl.zeros([BT], tl.int32)
    best, st = tl.where(c1 < best, c1, best), tl.where(c1 < best, 1, st)
    best, st = tl.where(c2 < best, c2, best), tl.where(c2 < best, 2, st)
    best, st = tl.where(c3 < best, c3, best), tl.where(c3 < best, 3, st)
    best, st = tl.where(c4 < best, c4, best), tl.where(c4 < best, 4, st)
    best, st = tl.where(c5 < best, c5, best), tl.where(c5 < best, 5, st)
    best, st = tl.where(c6 < best, c6, best), tl.where(c6 < best, 6, st)
    best, st = tl.where(c7 < best, c7, best), tl.where(c7 < best, 7, st)
    tl.debug_barrier()
    for jj in range(D):
        j = D - 1 - jj
        packed = tl.load(bits_ptr + base + j * S + toks, mask=tmask, other=0).to(tl.int32)
        bit = (packed >> st) & 1
        pred = (st >> 1) + 4 * bit
        b = st & 1
        k = (pred & 3) ^ (2 * ((pred >> 2) ^ b))
        y = tl.load(coef_ptr + base + j * S + toks, mask=tmask, other=0.0) * inv_d
        y = tl.minimum(tl.maximum(y, -YMAX), YMAX)
        n = _tcq_level(y, k.to(tl.float32)).to(tl.int32)
        tl.store(lvl_ptr + base + j * S + toks, n.to(tl.int8), mask=tmask)
        tl.store(par_ptr + base + j * S + toks, (pred & 1).to(tl.int8), mask=tmask)
        st = pred


@triton.jit
def tcq_rans_encode_kernel(
    idx_ptr, par_ptr, cum_ptr, out_ptr, cnt_ptr, state_ptr,
    S, D, nseg_per_row, CAP,
    TSEG: tl.constexpr, TROW: tl.constexpr, NBASE: tl.constexpr,
):
    """rANS encoder for TCQ indices; the table of each symbol is selected by the
    union-codebook parity of the trellis state that produced it."""
    pid = tl.program_id(0)
    row = pid // nseg_per_row
    base = (pid % nseg_per_row) * TSEG
    lanes = tl.arange(0, 32)
    x = tl.full([32], 65536, tl.int64)
    out_base = out_ptr + pid.to(tl.int64) * CAP
    rbase = row.to(tl.int64) * D * S
    wptr = CAP
    NT: tl.constexpr = TSEG // 32
    nsteps = NT * D
    for i in range(nsteps):
        step = nsteps - 1 - i
        t = step // D
        j = step - t * D
        tok = base + t * 32 + lanes
        valid = tok < S
        q = tl.load(idx_ptr + rbase + j * S + tok, mask=valid, other=0).to(tl.int32)
        p = tl.load(par_ptr + rbase + j * S + tok, mask=valid, other=0).to(tl.int32)
        bs = tl.where(q > 0, 2 * q - 1, -2 * q)
        bs = tl.where((q > NBASE) | (q < -NBASE), 2 * NBASE + 1, bs)
        crow = cum_ptr + ((row * D + j) * 2 + p) * TROW
        x, wptr = _enc_put(x, bs, crow, valid, out_base, wptr, 2 * NBASE + 1)
    tl.store(state_ptr + pid * 32 + lanes, x.to(tl.int32))
    tl.store(cnt_ptr + pid, CAP - wptr)


@triton.jit
def tcq_rans_decode_kernel(
    words_ptr, off_ptr, state_ptr, cum_ptr, esc_ptr, eoff_ptr, sym_ptr,
    S, D, nseg_per_row,
    TSEG: tl.constexpr, TROW: tl.constexpr, NBASE: tl.constexpr,
):
    """Decodes TCQ indices, follows each lane's trellis (reset per token) and
    writes reconstruction levels n = 2 * idx + parity (int8, [R, D, S])."""
    pid = tl.program_id(0)
    row = pid // nseg_per_row
    base = (pid % nseg_per_row) * TSEG
    lanes = tl.arange(0, 32)
    ltmask = (tl.full([32], 1, tl.uint32) << lanes.to(tl.uint32)) - 1
    x = tl.load(state_ptr + pid * 32 + lanes).to(tl.uint32)
    rptr = tl.load(off_ptr + pid) + tl.zeros([32], tl.int32)
    eptr = tl.load(eoff_ptr + pid) + tl.zeros([32], tl.int32)
    sym_row = sym_ptr + row.to(tl.int64) * D * S
    NT: tl.constexpr = TSEG // 32
    ESC: tl.constexpr = 2 * NBASE + 1
    for t in range(NT):
        tok = base + t * 32 + lanes
        valid = tok < S
        st = tl.zeros([32], tl.int32)
        for j in range(D):
            win = tl.load(words_ptr + rptr + lanes).to(tl.uint16).to(tl.uint32)
            par = st & 1
            crow0 = cum_ptr + (row * D + j) * 2 * TROW
            slot = (x & 4095).to(tl.int32)
            s = tl.zeros([32], tl.int32)
            for kk in tl.static_range(0, ESC):
                th = tl.where(par == 1, tl.load(crow0 + TROW + kk).to(tl.int32), tl.load(crow0 + kk).to(tl.int32))
                s += (slot >= th).to(tl.int32)
            crow = crow0 + par * TROW
            c = tl.load(crow + s - 1, mask=s > 0, other=0).to(tl.int32)
            f = tl.load(crow + s, mask=s < ESC, other=4096).to(tl.int32) - c
            xn = f.to(tl.uint32) * (x >> 12) + (slot - c).to(tl.uint32)
            xn = tl.where(valid, xn, x)
            need = (valid & (xn < 65536)).to(tl.uint32)
            bal = _ballot(need)
            w = _shfl(win, _popc(bal & ltmask))
            x = tl.where(need != 0, (xn << 16) | w, xn)
            rptr += _popc(bal).to(tl.int32)
            esc = valid & (s == ESC)
            ebal = _ballot(esc.to(tl.uint32))
            ev = tl.load(esc_ptr + eptr + _popc(ebal & ltmask).to(tl.int32), mask=esc, other=0)
            eptr += _popc(ebal).to(tl.int32)
            q = tl.where(esc, ev.to(tl.int32), tl.where((s & 1) == 1, (s + 1) >> 1, -(s >> 1)))
            n = 2 * q + par
            k = n & 3
            sub1 = (st & 3) ^ (2 * ((st >> 2) ^ 1))
            st = 2 * (st & 3) + (k == sub1).to(tl.int32)
            tl.store(sym_row + j * S + tok, n.to(tl.int8), mask=valid)
