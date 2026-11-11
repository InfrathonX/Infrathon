"""Paper baseline RTN (QVG eq. 1-2): symmetric per-group round-to-nearest.

Groups of 16 along the head dimension; one FP8-E4M3 scale per group relative
to a BF16 per-(batch, head) super-scale, so INT2 costs 2 + 8/16 bits per value
(6.40x, the paper's RTN/KIVI ratio) and INT4 costs 4 + 8/16 bits (3.56x).
"""
import torch

GROUP = 16
FP8_MAX = 448.0


def encode_one(x, bits):
    B, H, L, D = x.shape
    qmax = 2 ** (bits - 1) - 1
    g = x.float().reshape(B, H, L, D // GROUP, GROUP)
    scale = g.abs().amax(dim=-1) / qmax                       # [B,H,L,D/G]
    sup = (scale.amax(dim=(2, 3), keepdim=True) / FP8_MAX).clamp_min(1e-30)
    rel = (scale / sup).clamp(max=FP8_MAX).to(torch.float8_e4m3fn)
    eff = rel.float() * sup
    q = torch.round(g / torch.where(eff > 0, eff, torch.ones_like(eff)).unsqueeze(-1))
    q = q.clamp(-qmax, qmax).to(torch.int16) + (qmax + 1)      # non-negative codes
    q = q.reshape(B, H, L, D).to(torch.uint8)
    per = 8 // bits
    q = q.reshape(B, H, L, D // per, per)
    packed = torch.zeros(B, H, L, D // per, dtype=torch.uint8, device=x.device)
    for i in range(per):
        packed |= q[..., i] << (bits * i)
    return {"q": packed, "s": rel, "ss": sup.to(torch.bfloat16), "bits": bits, "shape": (B, H, L, D)}


def decode_one(p):
    B, H, L, D = p["shape"]
    bits = p["bits"]
    qmax = 2 ** (bits - 1) - 1
    per = 8 // bits
    mask = (1 << bits) - 1
    q = torch.stack([(p["q"] >> (bits * i)) & mask for i in range(per)], dim=-1)
    q = q.reshape(B, H, L, D // GROUP, GROUP).float() - (qmax + 1)
    eff = p["s"].float() * p["ss"].float()
    return (q * eff.unsqueeze(-1)).reshape(B, H, L, D).to(torch.bfloat16)

