"""Canonical math for the Wan2.2-TI2V-5B DanceGRPO task (Judge/oracle side, implementation-independent).

Everything here is the *contract* definition that candidate code, the reference and the Judge all have to agree on:

- canonical RNG keys: ``H(s)`` and the group/sample key strings;
- seed group: one initial noise x_T per prompt group, independent SDE noise and window per sample;
- the Dance-SDE transition (DanceGRPO / verl-omni ``dance_sde``) and its log-prob without constants;
- the GRPO advantage (group mean, global unbiased std of raw rewards + 1e-4, clip +-5);
- the PPO-clip loss (clip 1e-4, mean over pairs, no KL).

Sources (Apache-2.0 / MIT):
  verl-omni b8b67c9a  verl_omni/pipelines/schedulers/flow_match_sde.py (dance_sde branch)
                      verl_omni/trainer/diffusion/diffusion_algos.py (advantage, FlowGRPOLoss)
  DanceGRPO 15cc71d5  fastvideo/train_grpo_flux.py flux_step
  Flow-Factory 1bcb5293 scheduler/flow_match_euler_discrete.py Dance-SDE branch
  UniRL b5079be1      unirl/sde/kernels.py DanceSDEStrategy
The fp64 functions below are the oracle. They take torch tensors of any float dtype and compute in float64.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch

ETA = 1.2
N_PER_GROUP = 8
WINDOW_SIZE = 2
WINDOW_START_RANGE = 4  # start w in [0, 4): VO sde_window_range=[0,5] with randint(lo, hi - window_size + 1)
NUM_STEPS = 10
GUIDANCE_SCALE = 5.0
ADV_EPS = 1e-4
ADV_CLIP = 5.0
PPO_CLIP = 1e-4
REWARD_SCALE = 0.1
REWARD_FRAMES = (0, 4, 8)
LATENT_SHAPE = (1, 48, 3, 44, 80)  # 704x1280x9 frames, TI2V VAE 16x16 spatial / 4x temporal (S1 confirms)

_MOD = 1 << 63


def H(s: str) -> int:
    """Canonical 63-bit key: first 8 bytes of sha256(utf-8), big-endian, mod 2**63."""
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "big") % _MOD


@dataclass(frozen=True)
class GroupKey:
    run_seed: int
    round: int
    prompt_row: int

    def _base(self) -> str:
        return f"{self.run_seed}|{self.round}|{self.prompt_row}"

    def init_key(self) -> int:
        return H(f"init|{self._base()}")

    def window_key(self, j: int) -> int:
        return H(f"win|{self._base()}|{j}")

    def sde_key(self, j: int) -> int:
        return H(f"sde|{self._base()}|{j}")

    def window_start(self, j: int) -> int:
        return self.window_key(j) % WINDOW_START_RANGE

    def window(self, j: int) -> tuple[int, int]:
        w = self.window_start(j)
        return (w, w + 1)

    def pair_ids(self) -> list[tuple[int, int, int, int, int]]:
        """Canonical pair ids (run_seed, round, prompt_row, j, k) with k the absolute step index."""
        return [
            (self.run_seed, self.round, self.prompt_row, j, k)
            for j in range(N_PER_GROUP)
            for k in self.window(j)
        ]


def canonical_init_noise(key: GroupKey, shape=LATENT_SHAPE) -> torch.Tensor:
    """x_T shared by the 8 samples of a group. CPU Philox-free generator, fp32, bitwise portable within one torch build."""
    g = torch.Generator(device="cpu").manual_seed(key.init_key())
    return torch.randn(shape, generator=g, dtype=torch.float32, device="cpu")


def canonical_sde_noise(key: GroupKey, j: int, shape=LATENT_SHAPE) -> list[torch.Tensor]:
    """SDE noise of sample j: one draw per window step, in step order, from one per-sample generator."""
    g = torch.Generator(device="cpu").manual_seed(key.sde_key(j))
    return [torch.randn(shape, generator=g, dtype=torch.float32, device="cpu") for _ in range(WINDOW_SIZE)]


# ---------------------------------------------------------------------------------------------------------------
# Dance-SDE transition (fp64 oracle)
# ---------------------------------------------------------------------------------------------------------------


def dance_score_direction(x: torch.Tensor, v: torch.Tensor, sigma: float, sigma_next: float) -> torch.Tensor:
    """The score term of the Dance mean per unit eta^2: 0.5*(x - x0_hat*(1-sigma))/sigma^2 * dsigma (fp64).
    mu = x + dsigma*v + eta^2 * S, so an engine using eta_e^2 leaves eps_hat - eps = (eta_e^2 - eta^2) * S / s."""
    x = x.double()
    v = v.double()
    sigma, sigma_next = float(sigma), float(sigma_next)
    x0_hat = x - sigma * v
    return 0.5 * (x - x0_hat * (1.0 - sigma)) / sigma**2 * (sigma_next - sigma)


def dance_mean_std(x: torch.Tensor, v: torch.Tensor, sigma: float, sigma_next: float, eta: float = ETA):
    """mu = x + dsigma*v + 0.5*eta^2*(x - x0_hat*(1-sigma))/sigma^2 * dsigma, s = eta*sqrt(sigma - sigma_next).

    x0_hat = x - sigma*v.  Valid at sigma == 1 (no sigma/(1-sigma) term)."""
    x = x.double()
    v = v.double()
    sigma = float(sigma)
    sigma_next = float(sigma_next)
    if not (0.0 < sigma <= 1.0 and 0.0 <= sigma_next < sigma):
        raise ValueError(f"bad sigma pair {sigma}, {sigma_next}")
    dsigma = sigma_next - sigma
    x0_hat = x - sigma * v
    mu = x + dsigma * v + 0.5 * eta**2 * (x - x0_hat * (1.0 - sigma)) / sigma**2 * dsigma
    std = eta * math.sqrt(sigma - sigma_next)
    return mu, std


def dance_log_prob(x_next: torch.Tensor, mu: torch.Tensor, std: float) -> torch.Tensor:
    """Per-sample mean over elements of -(x' - mu)^2 / (2 s^2); constants dropped. Returns shape [B]."""
    d = x_next.double() - mu.double()
    lp = -(d**2) / (2.0 * std**2)
    return lp.flatten(1).mean(dim=1)


def euler_step(x: torch.Tensor, v: torch.Tensor, sigma: float, sigma_next: float) -> torch.Tensor:
    return x.double() + (float(sigma_next) - float(sigma)) * v.double()


def cfg_combine(v_cond: torch.Tensor, v_uncond: torch.Tensor, scale: float = GUIDANCE_SCALE) -> torch.Tensor:
    """v = v_uncond + scale*(v_cond - v_uncond) (Wan/diffusers convention)."""
    return v_uncond.double() + scale * (v_cond.double() - v_uncond.double())


# ---------------------------------------------------------------------------------------------------------------
# Reward -> advantage -> loss
# ---------------------------------------------------------------------------------------------------------------


def video_reward(frame_mu: torch.Tensor) -> torch.Tensor:
    """frame_mu: [..., 3] HPSv3 mu of frames (0, 4, 8). reward = 0.1 * mean."""
    return REWARD_SCALE * frame_mu.double().mean(dim=-1)


def grpo_advantages(rewards: torch.Tensor, group_ids: list) -> torch.Tensor:
    """rewards: [N] raw per-video rewards of one round. A = clip((r - mean_group) / (std_unbiased(all r) + 1e-4), +-5).

    Note: verl-omni computes torch.std over a [B, T] reward tensor with the reward repeated over T; here the std is
    over one reward per video (miles --globalize-reward-std semantics). Declared deviation (factor ~0.996 at N=64)."""
    r = rewards.double()
    std = torch.std(r, unbiased=True)
    out = torch.empty_like(r)
    groups: dict = {}
    for i, g in enumerate(group_ids):
        groups.setdefault(g, []).append(i)
    for idx in groups.values():
        m = r[idx].mean()
        out[idx] = (r[idx] - m) / (std + ADV_EPS)
    return out.clamp(-ADV_CLIP, ADV_CLIP)


def ppo_clip_terms(lp_new: torch.Tensor, lp_old: torch.Tensor, adv: torch.Tensor, clip: float = PPO_CLIP):
    """Per-pair PPO-clip loss max(-A r, -A clip(r)); returns (per_pair_loss, ratio, clipped_mask) in fp64."""
    ratio = torch.exp(lp_new.double() - lp_old.double())
    a = adv.double()
    unclipped = -a * ratio
    clipped = -a * ratio.clamp(1.0 - clip, 1.0 + clip)
    per = torch.maximum(unclipped, clipped)
    mask = (ratio - 1.0).abs() > clip
    return per, ratio, mask


def ppo_clip_loss(lp_new, lp_old, adv, clip: float = PPO_CLIP) -> torch.Tensor:
    per, _, _ = ppo_clip_terms(lp_new, lp_old, adv, clip)
    return per.mean()
