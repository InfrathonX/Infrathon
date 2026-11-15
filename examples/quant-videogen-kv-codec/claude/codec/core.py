"""Encoder/decoder for one BHSD tensor.

Encode: per (batch, head) row, k-means centroids model the token clusters; the
residual is rotated into its PCA basis and uniformly quantized with plain
rounding (no deadzone: shrinkage would bias softmax mass of decoded keys).  The
integer coefficients are entropy coded with interleaved rANS using one
frequency table per (row, component).
"""

import torch
import triton

from codec import kernels
from codec import pack

TSEG = 512
NBASE = 4
TROW = 2 * NBASE + 1
QMAX = 127
PROB_SCALE = 4096
JOINT_MAX_DIM = 256
JOINT_K_WEIGHT = 1.3
TCQ_YMAX = 120.0
MAX_REL_ERR = 0.6
KMEANS_ITERS = 10
MAX_CLUSTERS = 256
TOKENS_PER_CLUSTER = 128


def _num_clusters(S):
    return max(1, min(MAX_CLUSTERS, S // TOKENS_PER_CLUSTER))


def _assign(xb, cent):
    """Nearest centroid per token (fused distance + argmin kernel)."""
    R, S, D = xb.shape
    ncl = cent.shape[1]
    ids = torch.empty(R, S, device=xb.device, dtype=torch.int32)
    bd = max(16, triton.next_power_of_2(D))
    bt = 64 if bd <= 128 else 32
    kernels.assign_kernel[(R, triton.cdiv(S, bt))](
        xb, cent.to(torch.bfloat16), (cent * cent).sum(-1), ids, S, D, ncl,
        BT=bt, BD=bd, BC=64, num_warps=4,
    )
    return ids.to(torch.int64)


def _cluster_means(xb, ids, ncl, prev):
    """Deterministic cluster means; empty clusters keep ``prev``."""
    R, S, D = xb.shape
    bd = max(16, triton.next_power_of_2(D))
    bc = 128 if bd <= 128 else 64
    tch = 4096
    nch = triton.cdiv(S, tch)
    part = torch.empty(nch, R, ncl, D, device=xb.device, dtype=torch.float32)
    cnt = torch.empty(nch, R, ncl, device=xb.device, dtype=torch.float32)
    kernels.cluster_sum_kernel[(R, triton.cdiv(ncl, bc), nch)](
        xb, ids, part, cnt, S, D, ncl, tch, BT=64, BD=bd, BC=bc, num_warps=8,
    )
    sums = part.sum(0)
    cnt = cnt.sum(0).unsqueeze(-1)
    return torch.where(cnt > 0, sums / cnt.clamp_min(1.0), prev)


def _kmeans(xf, ncl):
    S = xf.shape[1]
    init = (torch.arange(ncl, device=xf.device) * S) // ncl
    cent = xf[:, init, :].clone()
    xb = xf.to(torch.bfloat16)
    for _ in range(KMEANS_ITERS):
        cent = _cluster_means(xb, _assign(xb, cent), ncl, cent)
    return _assign(xb, cent), cent


def _normalize(counts):
    """Scale symbol counts [..., n] to integer frequencies summing to PROB_SCALE,
    keeping every occurring symbol codable; returns the cumulative table [..., n + 1]."""
    total = counts.sum(-1, keepdim=True)
    freq = torch.floor(counts * PROB_SCALE / total.clamp_min(1.0))
    freq = torch.where(counts > 0, freq.clamp_min(1.0), freq)
    deficit = PROB_SCALE - freq.sum(-1, keepdim=True)
    freq.scatter_add_(-1, counts.argmax(-1, keepdim=True), deficit)
    zero = torch.zeros(freq.shape[:-1] + (1,), device=freq.device, dtype=freq.dtype)
    return torch.cat([zero, torch.cumsum(freq, -1)], -1)


def _freq_tables(q, par):
    """q: [R, D, S] int8 TCQ indices, par: [R, D, S] int8 codebook parity ->
    interior cumulative boundaries [R, D, 2, TROW] int16 of the base alphabet
    {0, +1, -1, ..., +NBASE, -NBASE, ESC}, one table per union codebook."""
    R, D, S = q.shape
    qi = q.to(torch.int32)
    bs = torch.where(qi > 0, 2 * qi - 1, -2 * qi)
    bs = torch.where(qi.abs() > NBASE, 2 * NBASE + 1, bs)
    key = (par.to(torch.int64) * (2 * NBASE + 2) + bs.to(torch.int64))
    counts = torch.zeros(R, D, 2 * (2 * NBASE + 2), device=q.device, dtype=torch.float32)
    counts.scatter_add_(2, key, torch.ones(key.shape, device=q.device, dtype=torch.float32))
    counts = counts.reshape(R, D, 2, 2 * NBASE + 2)
    # Only the interior boundaries cum[1..TROW] are stored (cum[0] = 0 and
    # cum[TROW + 1] = PROB_SCALE are implicit).
    return _normalize(counts)[..., 1:TROW + 1].to(torch.int16).contiguous()


def _escapes(q, nseg_row):
    """Raw values of |q| > NBASE in decoder order (segment, token round, component,
    lane) plus each segment's start offset."""
    R, D, S = q.shape
    pad = nseg_row * TSEG - S
    qp = torch.nn.functional.pad(q, (0, pad)) if pad else q
    order = qp.reshape(R, D, nseg_row, TSEG // 32, 32).permute(0, 2, 3, 1, 4)
    flat = order.reshape(R * nseg_row, -1)
    mask = flat.abs() > NBASE
    cnt = mask.sum(-1).to(torch.int32)
    vals = flat[mask]
    return torch.cat([vals, torch.zeros(32, device=q.device, dtype=torch.int8)]), torch.cumsum(cnt, 0) - cnt


def encode_pair(k, v, step_k, step_v, shape_gamma_k=0.0, shape_gamma_v=0.0, row_alpha_k=0.0, row_alpha_v=0.0):
    """Keys and values share one token clustering (ids computed on the joint,
    per-row normalized [K, V] features); both payloads reference the same
    packed ids, whose storage is counted once."""
    B, H, S, D = k.shape
    if k.numel() == 0 or k.shape != v.shape or 2 * D > JOINT_MAX_DIM:
        return encode_tensor(k, step_k, shape_gamma_k), encode_tensor(v, step_v, shape_gamma_v)
    R = B * H
    ncl = _num_clusters(S)
    kc = k.reshape(R, S, D).float()
    kc = kc - kc.mean(1, keepdim=True)
    vc = v.reshape(R, S, D).float()
    vc = vc - vc.mean(1, keepdim=True)
    kn = kc.pow(2).mean((1, 2), keepdim=True).sqrt().clamp_min(1e-30)
    vn = vc.pow(2).mean((1, 2), keepdim=True).sqrt().clamp_min(1e-30)
    ids, _ = _kmeans(torch.cat([JOINT_K_WEIGHT * kc / kn, vc / vn], -1), ncl)
    shared = pack.pack_ids(ids.to(torch.uint8)) if ncl <= 256 else ids.to(torch.int16)
    pk = encode_tensor(k, step_k, shape_gamma_k, clusters=(ids, shared), row_alpha=row_alpha_k)
    pv = encode_tensor(v, step_v, shape_gamma_v, clusters=(ids, shared), row_alpha=row_alpha_v)
    return pk, pv


def encode_tensor(x, step_rel, shape_gamma=0.0, clusters=None, row_alpha=0.0):
    B, H, S, D = x.shape
    R = B * H
    dev = x.device
    if x.numel() == 0:
        return {"s": (B, H, S, D), "z": torch.empty(0, device=dev, dtype=torch.int8)}
    xf = x.reshape(R, S, D).float()
    mu = xf.mean(1, keepdim=True)
    xc = xf - mu
    ncl = _num_clusters(S)
    if clusters is None:
        ids, cent = _kmeans(xc, ncl)
        ids_store = pack.pack_ids(ids.to(torch.uint8)) if ncl <= 256 else ids.to(torch.int16)
    else:
        ids, ids_store = clusters
        cent = _cluster_means(xc.to(torch.bfloat16), ids, ncl, torch.zeros(R, ncl, D, device=dev))
    # Centroids are stored as int8 relative to the row mean, one scale per (row, dim).
    cscale = cent.abs().amax(1, keepdim=True).clamp_min(1e-30) / 127.0
    cent8 = torch.round(cent / cscale).clamp(-127, 127).to(torch.int8)
    resid = xc - torch.gather(cent8.float() * cscale, 1, ids.unsqueeze(-1).expand(-1, -1, D))
    cov = torch.bmm(resid.transpose(1, 2), resid) / S
    # PCA basis of the PSD covariance: its left singular vectors, by decreasing
    # variance (batched gesvda is far faster than eigh for many small matrices).
    try:
        vecs, lam, _ = torch.linalg.svd(cov, driver="gesvda")
    except RuntimeError:  # degenerate spectra: fall back to the exact solver
        lam, vecs = torch.linalg.eigh(cov)
        lam, vecs = lam.flip(-1), vecs.flip(-1)
    # int8 basis (rows = components) with one scale per row; coefficients use its
    # exact inverse so basis rounding adds no reconstruction error.  Optional
    # error shaping scales component j's effective step by (lam_j / mean)^-gamma
    # (gamma > 0: finer steps where the residual variance is large; gamma < 0:
    # the reverse) by scaling basis rows.
    basis = vecs.transpose(1, 2)
    if shape_gamma != 0:
        rel_var = lam.clamp_min(0) / lam.clamp_min(0).mean(-1, keepdim=True).clamp_min(1e-30)
        mult = rel_var.clamp(1e-4, 1e4).pow(-shape_gamma)
        mult = mult / mult.pow(2).mean(-1, keepdim=True).sqrt()
        basis = basis * mult.unsqueeze(-1)
    bscale = basis.abs().amax((1, 2), keepdim=True).clamp_min(1e-30) / 127.0
    basis8 = torch.round(basis / bscale).clamp(-127, 127).to(torch.int8)
    coef = torch.bmm(resid, torch.linalg.inv(basis8.float() * bscale))  # [R, S, comp]
    # Optional per-row (head) step adaptation: row r's step is scaled by
    # (rms_r / rms)^alpha, folded into its basis scale.
    if row_alpha != 0.0:
        rrow = coef.pow(2).mean((1, 2), keepdim=True).sqrt()
        frow = (rrow / rrow.pow(2).mean().sqrt().clamp_min(1e-30)).clamp(1e-3, 1e3).pow(row_alpha)
        coef = coef / frow
        bscale = bscale * frow
    rms = float(coef.pow(2).mean().sqrt())
    step = max(step_rel * rms, 1e-12)
    # Quality floor: on weakly structured inputs (e.g. white noise) shrink the step
    # until the quantization error is at most MAX_REL_ERR of the input energy.
    budget = MAX_REL_ERR * MAX_REL_ERR * float(xf.pow(2).sum())
    for _ in range(12):
        err = float((coef - torch.round(coef / step).clamp(-QMAX, QMAX) * step).pow(2).sum())
        if err <= budget:
            break
        step = step * 0.85
    # Trellis-coded quantization along each token's coefficients: the TCQ step is
    # half the scalar step (same rate, lower distortion); plain nearest-level
    # Viterbi search (no rate term, which would shrink values toward zero).
    tstep = 0.5 * step
    coef_t = coef.transpose(1, 2).contiguous()
    lvl = torch.empty(R, D, S, device=dev, dtype=torch.int8)
    par = torch.empty_like(lvl)
    tbits = torch.empty(R, D, S, device=dev, dtype=torch.uint8)
    kernels.tcq_kernel[(R, triton.cdiv(S, 64))](
        coef_t, lvl, par, tbits, S, D, 1.0 / tstep, TCQ_YMAX, BT=64, num_warps=4,
    )
    q = ((lvl.to(torch.int16) - par.to(torch.int16)) // 2).to(torch.int8)
    cum = _freq_tables(q, par)

    nseg_row = triton.cdiv(S, TSEG)
    nseg = R * nseg_row
    cap = TSEG * D
    tmp = torch.empty(nseg, cap, device=dev, dtype=torch.int16)
    cnt = torch.empty(nseg, device=dev, dtype=torch.int32)
    states = torch.empty(nseg * 32, device=dev, dtype=torch.int32)
    kernels.tcq_rans_encode_kernel[(nseg,)](
        q, par, cum, tmp, cnt, states, S, D, nseg_row, cap,
        TSEG=TSEG, TROW=TROW, NBASE=NBASE, num_warps=1,
    )
    keep = torch.arange(cap, device=dev).unsqueeze(0) >= (cap - cnt).unsqueeze(1)
    # 32 trailing words keep the decoder's prefetch window in bounds.
    words = torch.cat([tmp[keep], torch.zeros(32, device=dev, dtype=torch.int16)])
    offsets = torch.cumsum(cnt, 0) - cnt
    esc_vals, esc_off = _escapes(q, nseg_row)
    return {
        "w": words,
        "o": offsets.to(torch.int32),
        "x": states,
        "t": cum,
        "e": esc_vals,
        "f": esc_off.to(torch.int32),
        "c": pack.pack_int8(cent8),
        "k": cscale.reshape(R, D),
        "m": mu.reshape(R, D),
        "b": bscale.reshape(R),
        "i": ids_store,
        "u": pack.pack_int8(basis8),
        "q": tstep,
        "s": (B, H, S, D),
    }


def decode_tensor(p):
    B, H, S, D = p["s"]
    R = B * H
    if B * H * S * D == 0:
        return torch.empty(B, H, S, D, device=p["z"].device, dtype=torch.bfloat16)
    dev = p["w"].device
    nseg_row = triton.cdiv(S, TSEG)
    nseg = R * nseg_row
    sym = torch.empty(R, D, S, device=dev, dtype=torch.int8)
    kernels.tcq_rans_decode_kernel[(nseg,)](
        p["w"], p["o"], p["x"], p["t"], p["e"], p["f"], sym, S, D, nseg_row,
        TSEG=TSEG, TROW=TROW, NBASE=NBASE, num_warps=1,
    )
    out = torch.empty(B, H, S, D, device=dev, dtype=torch.bfloat16)
    ncl = _num_clusters(S)
    cent8 = pack.unpack_int8(p["c"], R * ncl * D).reshape(R, ncl, D)
    basis8 = pack.unpack_int8(p["u"], R * D * D).reshape(R, D, D)
    ids = p["i"] if isinstance(p["i"], torch.Tensor) else pack.unpack_ids(p["i"], R, S)
    bd = max(16, triton.next_power_of_2(D))
    bt = max(16, min(64, 8192 // bd))
    kernels.recon_kernel[(R, triton.cdiv(S, bt))](
        sym, basis8, p["b"], cent8, p["k"], p["m"], ids, out, p["q"], S, D, ncl, 0,
        BT=bt, BD=bd, JT=32, num_warps=4,
    )
    return out
