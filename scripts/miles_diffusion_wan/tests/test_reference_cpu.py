"""CPU tests: the reference MD pieces (DanceSdeStepBackend, canonical_rng) agree with the contract oracle.

Loads single files from the reference MD copy ($WAN_MD_REF: a miles_diffusion 29fde6ee export with
tasks/archive/miles-diffusion-wan22-dancegrpo-2xh100-v1-seed/reference/miles_diffusion_wan.patch applied) without
importing the miles package (no ray/sglang needed). Skipped when WAN_MD_REF is not set.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import wan_dance_contract as C  # noqa: E402

if not os.environ.get("WAN_MD_REF"):
    pytest.skip("set WAN_MD_REF to the reference miles_diffusion tree (29fde6ee + reference/miles_diffusion_wan.patch)",
                allow_module_level=True)
MD = Path(os.environ["WAN_MD_REF"])
SG_SIGMAS = [1.0, 0.97574, 0.94627, 0.90971, 0.86313, 0.80179, 0.71732, 0.59362, 0.39510, 0.02441, 0.0]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, MD / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_canonical_rng_matches_contract():
    crng = _load("md_canonical_rng", "miles/rollout/canonical_rng.py")
    for run_seed, rnd, row in [(42, 0, 0), (42, 2, 17), (7, 1, 399), (123456789, 5, 1000)]:
        k = C.GroupKey(run_seed, rnd, row)
        base = crng.group_base(run_seed, rnd, row)
        assert crng.init_seed(base) == k.init_key()
        for j in range(8):
            assert crng.sde_noise_seed(base, j) == k.sde_key(j)
            assert crng.window(base, j, 2, 0, 5) == list(k.window(j))


def test_dance_backend_matches_contract():
    sb = _load("md_sde_step_backend", "miles/backends/fsdp_utils/sde_step_backend.py")
    sched = types.SimpleNamespace(sigmas=torch.tensor(SG_SIGMAS, dtype=torch.float32))
    sched.index_for_timestep = lambda t: int(np.argmin(np.abs(np.array(SG_SIGMAS[:-1]) * 1000 - float(t))))
    be = sb.DanceSdeStepBackend(sched)
    g = torch.Generator().manual_seed(0)
    for k in range(5):
        x = torch.randn(2, 48, 3, 4, 6, generator=g)
        v = torch.randn(2, 48, 3, 4, 6, generator=g)
        xn = torch.randn(2, 48, 3, 4, 6, generator=g)
        t = torch.tensor([SG_SIGMAS[k] * 1000] * 2)
        tn = torch.tensor([SG_SIGMAS[k + 1] * 1000] * 2)
        _, lp, mean, std = be.sde_step_logprob(v, t, tn, x, prev_sample=xn, noise_level=C.ETA)
        mu, s = C.dance_mean_std(x, v, SG_SIGMAS[k], SG_SIGMAS[k + 1])
        assert torch.allclose(mean.double(), mu, rtol=1e-5, atol=1e-5)
        assert torch.allclose(std.double().flatten(), torch.full((2,), s, dtype=torch.float64), rtol=1e-6)
        assert torch.allclose(lp.double(), C.dance_log_prob(xn, mu, s), rtol=1e-4)


def test_flow_sde_backend_is_not_dance():
    """Planted bug P1 at component level: the HEAD flow-SDE backend gives a different std/mean."""
    sb = _load("md_sde_step_backend2", "miles/backends/fsdp_utils/sde_step_backend.py")
    sched = types.SimpleNamespace(sigmas=torch.tensor(SG_SIGMAS, dtype=torch.float32))
    sched.index_for_timestep = lambda t: int(np.argmin(np.abs(np.array(SG_SIGMAS[:-1]) * 1000 - float(t))))
    x = torch.randn(1, 48, 3, 4, 6)
    v = torch.randn(1, 48, 3, 4, 6)
    for k in (0, 2):
        t = torch.tensor([SG_SIGMAS[k] * 1000])
        tn = torch.tensor([SG_SIGMAS[k + 1] * 1000])
        _, _, m_sde, std_sde = sb.DiffusersSdeStepBackend(sched).sde_step_logprob(v, t, tn, x, prev_sample=x,
                                                                                    noise_level=C.ETA)
        _, _, m_d, std_d = sb.DanceSdeStepBackend(sched).sde_step_logprob(v, t, tn, x, prev_sample=x,
                                                                          noise_level=C.ETA)
        noise_sde = std_sde * torch.sqrt(torch.tensor(SG_SIGMAS[k] - SG_SIGMAS[k + 1]))
        assert (noise_sde.flatten() / std_d.flatten() - 1).abs().max() > 0.5
