"""CPU tests of the canonical contract math against independent upstream implementations (oracle check O3).

Run:  <MDWAN_ASSETS>/venv-cpu/bin/python -m pytest -q scripts/miles_diffusion_wan/tests
Upstream sources are loaded straight from the pinned checkouts under $WAN_REFS (on the authoring layout
<MDWAN_ASSETS>/refs): verl-omni, DanceGRPO, Flow-Factory, UniRL-Tencent. Without WAN_REFS the tests that
compare against them are skipped.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import os
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import wan_dance_contract as C  # noqa: E402

REFS = Path(os.environ["WAN_REFS"]) if os.environ.get("WAN_REFS") else None
needs_refs = pytest.mark.skipif(REFS is None, reason="set WAN_REFS to the pinned upstream checkouts (verl-omni, "
                                "DanceGRPO, Flow-Factory, UniRL-Tencent; usually <MDWAN_ASSETS>/refs)")
VO = REFS / "verl-omni" if REFS else None
DG = REFS / "DanceGRPO" if REFS else None
FF = REFS / "Flow-Factory" if REFS else None
UR = REFS / "UniRL-Tencent" if REFS else None

# sigma grid used in the kernel tests: sd3_time_shift(5, linspace(1, 0, 11)) (VO's grid); the engine grid is
# recorded in S4 and only changes the numbers, not the formulas.
SIGMAS = [5 * t / (1 + 4 * t) for t in np.linspace(1.0, 0.0, 11).tolist()]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _extract_function(path: Path, name: str, glb: dict, cls: Optional[str] = None):
    """Exec one top-level function (or a method of `cls`) from an upstream file without importing the module."""
    tree = ast.parse(path.read_text())
    body = tree.body
    if cls is not None:
        body = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    fn = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    mod = ast.Module(body=[fn], type_ignores=[])
    ns = dict(glb)
    exec(compile(mod, str(path), "exec"), ns)
    return ns[name]


def _rand(shape=(2, 48, 3, 4, 6), seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g, dtype=torch.float64)


# ------------------------------------------------------------------------------------------------ canonical keys


def test_H_vectors_frozen():
    # Frozen test vectors: if these change the contract changed.
    assert C.H("init|42|0|0") == C.H("init|42|0|0")
    assert 0 <= C.H("x") < 2**63
    frozen = {s: C.H(s) for s in ["init|42|0|0", "win|42|0|0|0", "sde|42|0|0|7", ""]}
    assert frozen[""] == int("e3b0c44298fc1c14", 16) % 2**63  # sha256("") prefix
    assert len(set(frozen.values())) == 4


def test_window_and_pairs():
    starts = []
    for row in range(200):
        k = C.GroupKey(42, 0, row)
        for j in range(C.N_PER_GROUP):
            w0, w1 = k.window(j)
            assert 0 <= w0 < 4 and w1 == w0 + 1
            starts.append(w0)
        ids = k.pair_ids()
        assert len(ids) == 16 and len(set(ids)) == 16
    counts = np.bincount(starts, minlength=4)
    assert counts.min() > 0.2 * len(starts)  # roughly uniform over {0,1,2,3}


def test_seed_group_noise():
    shape = (1, 4, 3, 5, 7)
    k = C.GroupKey(42, 1, 17)
    a = C.canonical_init_noise(k, shape)
    b = C.canonical_init_noise(k, shape)
    assert torch.equal(a, b) and a.dtype == torch.float32
    assert not torch.equal(a, C.canonical_init_noise(C.GroupKey(42, 1, 18), shape))
    assert not torch.equal(a, C.canonical_init_noise(C.GroupKey(42, 2, 17), shape))
    eps = [C.canonical_sde_noise(k, j, shape) for j in range(8)]
    for j in range(8):
        assert len(eps[j]) == 2 and not torch.equal(eps[j][0], eps[j][1])
        assert torch.equal(eps[j][0], C.canonical_sde_noise(k, j, shape)[0])
        for i in range(j):
            assert not torch.equal(eps[i][0], eps[j][0])
        assert not torch.equal(eps[j][0], a)


# ------------------------------------------------------------------------------------------------ Dance kernel


def _vo_step():
    mod = _load_module("vo_flow_match_sde", VO / "verl_omni/pipelines/schedulers/flow_match_sde.py")
    sch = mod.FlowMatchSDEDiscreteScheduler(num_train_timesteps=1000, shift=1.0)
    return sch


@needs_refs
@pytest.mark.parametrize("k", [0, 1, 2, 3, 4])
def test_dance_matches_verl_omni(k):
    sch = _vo_step()
    sigmas = torch.tensor(SIGMAS, dtype=torch.float32)
    sch.sigmas = sigmas
    sch.timesteps = sigmas[:-1] * 1000
    sch._step_index = k
    x = _rand(seed=k).float()
    v = _rand(seed=100 + k).float()
    xn = _rand(seed=200 + k).float()
    prev, lp, mean, std = sch.step(
        v, None, x, prev_sample=xn, sde_type="dance_sde", noise_level=C.ETA, include_logprob_normalizer=False,
        return_dict=False,
    )[:4]
    mu, s = C.dance_mean_std(x, v, float(sigmas[k]), float(sigmas[k + 1]))
    assert torch.allclose(mean.double(), mu, rtol=1e-5, atol=1e-5)
    assert abs(float(std) - s) / s < 1e-6
    lp_o = C.dance_log_prob(xn, mu, s)
    assert torch.allclose(lp.double(), lp_o, rtol=1e-4)


@needs_refs
def test_dance_matches_dancegrpo_flux_step():
    flux_step = _extract_function(DG / "fastvideo/train_grpo_flux.py", "flux_step", {"torch": torch, "math": math})
    sig = torch.tensor(SIGMAS, dtype=torch.float64)
    for k in range(5):
        x, v, xn = _rand(seed=k), _rand(seed=10 + k), _rand(seed=20 + k)
        _, _, lp = flux_step(v, x, C.ETA, sig, k, xn, True, True)
        mu, s = C.dance_mean_std(x, v, SIGMAS[k], SIGMAS[k + 1])
        # DG includes the normalizer (-log s - log sqrt(2 pi)) and computes the log-prob in fp32
        const = -math.log(s) - math.log(math.sqrt(2 * math.pi))
        assert torch.allclose(lp.double() - const, C.dance_log_prob(xn, mu, s), rtol=1e-5, atol=1e-5)
        # DG mean path (grpo=False returns prev_sample_mean)
        mean, _ = flux_step(v, x, C.ETA, sig, k, None, False, True)
        assert torch.allclose(mean, mu, rtol=1e-12, atol=1e-12)


@needs_refs
def test_dance_matches_unirl():
    mod = _load_module("ur_kernels", UR / "unirl/sde/kernels.py")
    strat = mod.DanceSDEStrategy()
    for k in range(5):
        x, v = _rand(seed=k), _rand(seed=30 + k)
        sig = torch.tensor(SIGMAS[k], dtype=torch.float64)
        sig_n = torch.tensor(SIGMAS[k + 1], dtype=torch.float64)
        xn = _rand(seed=40 + k)
        prev, mean, std = strat.step(v, x, sig, sig_n, eta=C.ETA, prev_sample=xn)
        mu, s = C.dance_mean_std(x, v, SIGMAS[k], SIGMAS[k + 1])
        assert torch.allclose(mean, mu, rtol=1e-12, atol=1e-12)
        assert abs(float(std) - s) / s < 1e-12


@needs_refs
def test_dance_matches_flow_factory_formula():
    # Flow-Factory's Dance-SDE branch lives inside a large method with package-relative imports; check the
    # transcription of its three lines (flow_match_euler_discrete.py Dance-SDE branch) against the oracle.
    src = (FF / "src/flow_factory/scheduler/flow_match_euler_discrete.py").read_text()
    assert "0.5 * noise_level**2 * (latents - pred_original_sample * (1 - sigma)) / sigma**2" in src
    assert "next_latents_mean = latents + (velocity + log_term) * dt" in src
    for k in range(5):
        x, v = _rand(seed=k), _rand(seed=50 + k)
        sigma, sigma_prev = SIGMAS[k], SIGMAS[k + 1]
        dt = sigma_prev - sigma
        pred = x - sigma * v
        log_term = 0.5 * C.ETA**2 * (x - pred * (1 - sigma)) / sigma**2
        mean = x + (v + log_term) * dt
        std = C.ETA * math.sqrt(-dt)
        mu, s = C.dance_mean_std(x, v, sigma, sigma_prev)
        assert torch.allclose(mean, mu, rtol=1e-12, atol=1e-12) and abs(std - s) < 1e-15


def test_dance_sigma_one_finite_and_flow_sde_differs():
    x, v = _rand(seed=1), _rand(seed=2)
    mu, s = C.dance_mean_std(x, v, 1.0, SIGMAS[1])
    assert torch.isfinite(mu).all() and math.isfinite(s)
    # Flow-SDE ("sde") with the engine's sigma_max guard at sigma=1: planted bug P1 must be far away.
    sigma, sn, eta = 1.0, SIGMAS[1], C.ETA
    std_sde = math.sqrt(sigma / (1 - SIGMAS[1])) * eta
    dt = sn - sigma
    mean_sde = x * (1 + std_sde**2 / (2 * sigma) * dt) + v * (1 + std_sde**2 * (1 - sigma) / (2 * sigma)) * dt
    s_sde = std_sde * math.sqrt(-dt)
    assert abs(s_sde - s) / s > 0.5
    d = (mean_sde - mu).norm() / (s * math.sqrt(mu.numel()))
    assert d > 0.1


# ------------------------------------------------------------------------------------------------ advantage / loss


@needs_refs
def test_advantage_matches_verl_omni():
    fn = _extract_function(
        VO / "verl_omni/trainer/diffusion/diffusion_algos.py",
        "compute_flow_grpo_outcome_advantage",
        {"torch": torch, "np": np, "defaultdict": defaultdict, "Optional": Optional, "DictConfig": object},
    )
    g = torch.Generator().manual_seed(3)
    r = torch.randn(64, generator=g, dtype=torch.float64) * 0.3 + 1.0
    r[5] = 40.0  # force a clipped advantage
    idx = np.array([i // 8 for i in range(64)])
    vo, _ = fn(r[:, None].clone(), idx, epsilon=C.ADV_EPS, global_std=True)
    vo = vo[:, 0].clamp(-C.ADV_CLIP, C.ADV_CLIP)  # VO clamps inside the loss (adv_clip_max=5)
    ours = C.grpo_advantages(r, idx.tolist())
    assert torch.allclose(vo, ours, rtol=1e-12, atol=1e-12)
    assert ours.abs().max() == C.ADV_CLIP


@needs_refs
def test_ppo_loss_matches_verl_omni():
    fn = _extract_function(
        VO / "verl_omni/trainer/diffusion/diffusion_algos.py",
        "compute_loss",
        {"torch": torch, "Optional": Optional, "Any": object, "DiffusionActorConfig": object},
        cls="FlowGRPOLoss",
    )
    cfg = types.SimpleNamespace(diffusion_loss=types.SimpleNamespace(adv_clip_max=5.0, clip_ratio=C.PPO_CLIP))
    g = torch.Generator().manual_seed(4)
    old = torch.randn(256, generator=g, dtype=torch.float64)
    new = old + 3e-4 * torch.randn(256, generator=g, dtype=torch.float64)
    adv = torch.randn(256, generator=g, dtype=torch.float64)
    loss_vo, _ = fn(None, old_log_prob=old, log_prob=new, advantages=adv, config=cfg)
    per, ratio, mask = C.ppo_clip_terms(new, old, adv)
    assert torch.allclose(loss_vo.double(), per.mean(), rtol=1e-12)
    assert 0 < mask.float().mean() < 1


def test_video_reward():
    mu = torch.tensor([[1.0, 2.0, 3.0], [4.0, 4.0, 4.0]])
    assert torch.allclose(C.video_reward(mu), torch.tensor([0.2, 0.4], dtype=torch.float64))
