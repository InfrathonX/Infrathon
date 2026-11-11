"""CPU sanity of the fp32 training oracle on a tiny random WanTransformer3DModel (same class/config family)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import oracle_dance_fp32 as O  # noqa: E402
import wan_dance_contract as C  # noqa: E402

diffusers = pytest.importorskip("diffusers")


def _tiny():
    torch.manual_seed(0)
    m = diffusers.WanTransformer3DModel(
        patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=16, in_channels=48, out_channels=48,
        text_dim=32, freq_dim=32, ffn_dim=64, num_layers=2, cross_attn_norm=True, qk_norm="rms_norm_across_heads",
        rope_max_seq_len=64,
    )
    return m.float().eval()


def _pair(adv, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(48, 3, 4, 6, generator=g).bfloat16()
    return O.Pair(x=x, x_next=(x.float() + 0.1 * torch.randn(48, 3, 4, 6, generator=g)).bfloat16(),
                  sigma=0.9757429957389832, sigma_next=0.9462729692459106,
                  cond_pos=torch.randn(7, 32, generator=g), cond_neg=torch.randn(7, 32, generator=g), adv=adv)


def test_pair_logprob_consistent():
    m = _tiny()
    p = _pair(1.0, 1)
    r = O.pair_logprob(m, p, device="cpu")
    mu, s = C.dance_mean_std(p.x[None], r["v"][None], p.sigma, p.sigma_next)
    assert abs(float(C.dance_log_prob(p.x_next[None], mu, s)[0]) - r["lp"]) < 1e-6 * max(1, abs(r["lp"]))
    assert abs(r["s"] - C.ETA * (p.sigma - p.sigma_next) ** 0.5) < 1e-12


def test_gradient_matches_finite_difference():
    m = _tiny().double()
    for mod in m.modules():
        for name, buf in list(mod.named_buffers(recurse=False)):
            mod.register_buffer(name, buf.double() if buf.is_floating_point() else buf)
    pairs = [_pair(1.5, 2), _pair(-0.7, 3)]

    # double-precision model to make the finite difference meaningful
    def vel(model, x, sigma, cp, cn, device):
        xb = x.to(torch.float64)[None]
        t = torch.tensor([1000.0 * sigma], dtype=torch.float64)
        vp = model(hidden_states=xb, timestep=t, encoder_hidden_states=cp.double()[None], return_dict=False)[0]
        vn = model(hidden_states=xb, timestep=t, encoder_hidden_states=cn.double()[None], return_dict=False)[0]
        return vn + C.GUIDANCE_SCALE * (vp - vn)

    O_vel = O.velocity
    O.velocity = vel
    try:
        _, grads = O.loss_and_grad(m, pairs, device="cpu")
        name = "proj_out.weight"
        q = dict(m.named_parameters())[name]
        idx = (3, 5)
        eps = 1e-6
        with torch.no_grad():
            q[idx] += eps
            lp_plus = sum(-(p.adv / 2) * O.pair_logprob(m, p, device="cpu")["lp"] for p in pairs)
            q[idx] -= 2 * eps
            lp_minus = sum(-(p.adv / 2) * O.pair_logprob(m, p, device="cpu")["lp"] for p in pairs)
            q[idx] += eps
        fd = (lp_plus - lp_minus) / (2 * eps)
        assert abs(float(grads[name][idx]) - fd) <= 1e-4 * max(1.0, abs(fd))
    finally:
        O.velocity = O_vel
