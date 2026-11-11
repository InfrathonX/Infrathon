"""Author fp32 oracle for the training math (J1 probe, R3 pi_old, R6' honest-gradient seeds).

Independent of miles_diffusion: plain diffusers ``WanTransformer3DModel`` (pinned diffusers f53d5520), fp32 weights,
TF32 off, eager attention path chosen by diffusers. For each pair (x_k, x_{k+1}, sigma_k, sigma_{k+1}, cond pos/neg):
  v = v_neg + 5 * (v_pos - v_neg)                      (fp32, both CFG branches, timestep = 1000*sigma_k)
  mu, s = Dance(x_k, v, sigma_k, sigma_{k+1}, eta=1.2)   (fp64, wan_dance_contract)
  lp = mean_elems[-(x_{k+1} - mu)^2 / (2 s^2)]          (fp64)
Gradient mode: L = -(1/N) sum_i A_i * lp_i  (the PPO loss at ratio 1, its gradient equals the clipped loss's),
backward in fp32 -> dL/dtheta.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wan_dance_contract as C  # noqa: E402


@dataclass
class Pair:
    x: torch.Tensor  # [48, F, H, W] (as stored by the engine, usually bf16)
    x_next: torch.Tensor
    sigma: float
    sigma_next: float
    cond_pos: torch.Tensor  # [L, 4096]
    cond_neg: torch.Tensor
    adv: float = 0.0


def load_transformer(path: str, device="cuda"):
    from diffusers import WanTransformer3DModel

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    m = WanTransformer3DModel.from_pretrained(path, torch_dtype=torch.float32)
    return m.to(device).eval()


def velocity(model, x: torch.Tensor, sigma: float, cond_pos: torch.Tensor, cond_neg: torch.Tensor, device) -> torch.Tensor:
    xb = x.to(device=device, dtype=torch.float32)[None]
    t = torch.tensor([1000.0 * sigma], device=device, dtype=torch.float32)
    vp = model(hidden_states=xb, timestep=t, encoder_hidden_states=cond_pos.to(device, torch.float32)[None],
               return_dict=False)[0]
    vn = model(hidden_states=xb, timestep=t, encoder_hidden_states=cond_neg.to(device, torch.float32)[None],
               return_dict=False)[0]
    return vn + C.GUIDANCE_SCALE * (vp - vn)


@torch.no_grad()
def pair_logprob(model, p: Pair, device="cuda"):
    v = velocity(model, p.x, p.sigma, p.cond_pos, p.cond_neg, device)
    mu, s = C.dance_mean_std(p.x.to(device)[None], v, p.sigma, p.sigma_next)
    lp = C.dance_log_prob(p.x_next.to(device)[None], mu, s)
    return {"lp": float(lp[0]), "mu": mu[0].float().cpu(), "s": s, "v": v[0].float().cpu()}


def loss_and_grad(model, pairs: list[Pair], device="cuda"):
    """Returns (loss float, {name: grad fp32 cpu}) of L = -(1/N) sum A_i lp_i at theta (ratio == 1)."""
    model.zero_grad(set_to_none=True)
    n = len(pairs)
    total = 0.0
    for p in pairs:
        v = velocity(model, p.x, p.sigma, p.cond_pos, p.cond_neg, device)
        mu, s = C.dance_mean_std(p.x.to(device)[None], v, p.sigma, p.sigma_next)
        lp = C.dance_log_prob(p.x_next.to(device)[None], mu, s)  # fp64 graph back to fp32 v
        loss = -(p.adv / n) * lp[0]
        loss.backward()
        total += float(loss.detach())
    grads = {k: q.grad.detach().float().cpu() for k, q in model.named_parameters() if q.grad is not None}
    return total, grads
