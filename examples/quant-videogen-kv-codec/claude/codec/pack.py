"""Lossless packing helpers for codec side information (torch only)."""

import torch


def _bitplanes_pack(u, b):
    """u: [N] uint values < 2^b (N multiple of 8) -> [N // 8 * b] uint8 (bit planes)."""
    g = u.reshape(-1, 8).to(torch.int32)
    shifts = torch.arange(b, device=u.device, dtype=torch.int32)
    bits = (g.unsqueeze(-1) >> shifts) & 1                      # [G, 8, b]
    weights = (1 << torch.arange(8, device=u.device, dtype=torch.int32)).view(1, 8, 1)
    return (bits * weights).sum(1).to(torch.uint8).reshape(-1)  # [G * b]


def _bitplanes_unpack(packed, b, n):
    g = packed.reshape(-1, b).to(torch.int32)                   # [G, b]
    lane = torch.arange(8, device=packed.device, dtype=torch.int32).view(1, 8, 1)
    bits = (g.unsqueeze(1) >> lane) & 1                         # [G, 8, b]
    weights = (1 << torch.arange(b, device=packed.device, dtype=torch.int32)).view(1, 1, b)
    return (bits * weights).sum(-1).reshape(-1)[:n]


def pack_int8(x):
    """Lossless b-bit packing of an int8 tensor with escapes for out-of-range
    values; b is chosen to minimize the packed size."""
    flat = x.reshape(-1).to(torch.int32)
    n = flat.numel()
    pad = (-n) % 8
    best = None
    for b in (4, 5, 6, 7):
        lim = (1 << (b - 1)) - 1
        esc = (flat > lim) | (flat < -lim)
        size = (n + pad) // 8 * b + int(esc.sum())
        if best is None or size < best[0]:
            best = (size, b)
    if best is None or best[0] >= n:
        return {"r": x.reshape(-1).clone()}
    b = best[1]
    lim = (1 << (b - 1)) - 1
    esc = (flat > lim) | (flat < -lim)
    code = torch.where(esc, torch.zeros_like(flat), flat + lim + 1)  # 0 = escape
    if pad:
        code = torch.cat([code, torch.zeros(pad, device=x.device, dtype=torch.int32)])
    return {"p": _bitplanes_pack(code, b), "e": flat[esc].to(torch.int8), "n": b}


def unpack_int8(d, n):
    if "r" in d:
        return d["r"]
    b = d["n"]
    lim = (1 << (b - 1)) - 1
    code = _bitplanes_unpack(d["p"], b, n)
    esc = code == 0
    idx = (torch.cumsum(esc.to(torch.int32), 0) - 1).clamp_min(0)
    raw = d["e"].to(torch.int32)
    vals = code - lim - 1
    if raw.numel():
        vals = torch.where(esc, raw[idx.clamp_max(raw.numel() - 1)], vals)
    return vals.to(torch.int8)


def pack_ids(ids):
    """ids: [R, S] uint8 cluster ids -> repeat flags (same as previous token) and
    the ids of non-repeating tokens."""
    R, S = ids.shape
    rep = torch.zeros(R, S, device=ids.device, dtype=torch.bool)
    rep[:, 1:] = ids[:, 1:] == ids[:, :-1]
    flat = rep.reshape(-1).to(torch.int32)
    pad = (-flat.numel()) % 8
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, device=ids.device, dtype=torch.int32)])
    return {"g": _bitplanes_pack(flat, 1), "v": ids[~rep].contiguous()}


def unpack_ids(d, R, S):
    rep = _bitplanes_unpack(d["g"], 1, R * S).reshape(R, S).to(torch.bool)
    pos = torch.cumsum((~rep).to(torch.int32).reshape(-1), 0) - 1
    return d["v"][pos.long()].reshape(R, S)
