"""Engine-overlay unit tests (run inside the pinned image with the overlay first on PYTHONPATH):

  PYTHONPATH=/work/sglang_wan/python:/work/wan python -m pytest -q /work/wan/tests/test_engine_overlay.py

Checks the reference Dance branch + canonical SDE-noise key of the sglang-d rollout mixin against the fp64 contract
oracle, and that the legacy sde/cps/ode branches are bitwise unchanged versus the pristine installed engine (J0).
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import wan_dance_contract as C  # noqa: E402

# engine rollout sigma grid for 10 steps, shift 5 (codemap C; S4 re-measures it from a real response)
SG_SIGMAS = [1.0, 0.97574, 0.94627, 0.90971, 0.86313, 0.80179, 0.71732, 0.59362, 0.39510, 0.02441, 0.0]
SHAPE = (1, 4, 3, 5, 6)


class _PC:
    def shard_latents_for_sp(self, batch, latents):
        return latents, None


def _mixin_module():
    m = importlib.import_module("sglang.multimodal_gen.runtime.post_training.scheduler_rl_mixin")
    m.get_sp_world_size = lambda: 1
    return m


def _scheduler(m):
    class _S(m.SchedulerRLMixin):
        def __init__(self):
            self.sigmas = torch.tensor(SG_SIGMAS, dtype=torch.float32)

    return _S()


def _batch(sde_type, window, noise_seed, eta=C.ETA):
    return types.SimpleNamespace(
        rollout_log_prob_no_const=False,
        rollout_noise_level=eta,
        rollout_sde_type=sde_type,
        rollout_debug_mode=False,
        rollout_sde_step_indices=list(window),
        rollout_sde_noise_seed=noise_seed,
        latents=torch.zeros(SHAPE),
        _rollout_session_data=None,
    )


def test_overlay_is_the_imported_engine():
    m = _mixin_module()
    assert "/work/sglang_wan/" in m.__file__ or os.environ.get("ALLOW_ANY_ENGINE") == "1", m.__file__


@pytest.mark.parametrize("window", [(0, 1), (1, 2), (3, 4)])
def test_dance_trajectory_matches_contract(window):
    m = _mixin_module()
    sch = _scheduler(m)
    key = C.GroupKey(7, 0, 3)
    j = 5
    batch = _batch("dance", window, key.sde_key(j))
    sch.prepare_rollout(batch, _PC())
    eps = C.canonical_sde_noise(key, j, SHAPE)
    g = torch.Generator().manual_seed(123)
    x = torch.randn(SHAPE, generator=g)
    used = 0
    for k in range(6):
        batch._rollout_loop_step_index = k
        v = torch.randn(SHAPE, generator=g)
        s, sn = SG_SIGMAS[k], SG_SIGMAS[k + 1]
        x_next = sch.flow_sde_sampling(
            batch, v, x, torch.tensor(s), torch.tensor(sn), generator=torch.Generator().manual_seed(999)
        )
        if k in window:
            mu, std = C.dance_mean_std(x, v, s, sn)
            ref = mu + std * eps[used].double()
            used += 1
            tol = 1e-5
        else:
            ref = C.euler_step(x, v, s, sn)
            tol = 1e-6
        assert torch.allclose(x_next.double(), ref, rtol=0, atol=tol), (k, (x_next.double() - ref).abs().max())
        x = x_next
    assert used == 2
    lp, _ = sch.consume_local_rollout_log_probs(batch)
    assert lp.shape == (1, 6)


def test_sde_noise_independent_of_init_generator():
    m = _mixin_module()
    outs = []
    for init_seed in (1, 2):
        sch = _scheduler(m)
        batch = _batch("dance", (1, 2), 424242)
        sch.prepare_rollout(batch, _PC())
        batch._rollout_loop_step_index = 1
        x = torch.zeros(SHAPE)
        v = torch.zeros(SHAPE)
        outs.append(sch.flow_sde_sampling(batch, v, x, torch.tensor(0.9757), torch.tensor(0.9463),
                                          generator=torch.Generator().manual_seed(init_seed)))
    assert torch.equal(outs[0], outs[1])


def test_legacy_branches_unchanged_vs_pristine_engine():
    """J0: sde/cps/ode outputs of the overlay equal the pristine engine bitwise (fixed tensors and generator)."""
    pristine = os.environ.get("PRISTINE_SGLANG_PYTHON", "/sgl-workspace/sglang/python")
    if not Path(pristine).exists():
        pytest.skip("pristine engine path not found")
    import subprocess

    code = r'''
import sys, types, torch, pickle
import sglang.multimodal_gen.runtime.post_training.scheduler_rl_mixin as m
m.get_sp_world_size = lambda: 1
class S(m.SchedulerRLMixin):
    def __init__(self): self.sigmas = torch.tensor(%r, dtype=torch.float32)
class PC:
    def shard_latents_for_sp(self, batch, latents): return latents, None
res = {}
for t in ("sde", "cps", "ode"):
    sch = S()
    b = types.SimpleNamespace(rollout_log_prob_no_const=(t == "ode"), rollout_noise_level=0.7, rollout_sde_type=t,
        rollout_debug_mode=False, rollout_sde_step_indices=[1, 2], latents=torch.zeros(2, 4, 3, 5, 6),
        _rollout_session_data=None)
    sch.prepare_rollout(b, PC())
    g = torch.Generator().manual_seed(0); gens = [torch.Generator().manual_seed(11), torch.Generator().manual_seed(12)]
    x = torch.randn(2, 4, 3, 5, 6, generator=g); outs = []
    for k in range(4):
        b._rollout_loop_step_index = k
        v = torch.randn(2, 4, 3, 5, 6, generator=g).to(torch.bfloat16)
        x = sch.flow_sde_sampling(b, v, x, sch.sigmas[k], sch.sigmas[k + 1], gens); outs.append(x)
    res[t] = (torch.stack(outs), sch.consume_local_rollout_log_probs(b)[0])
pickle.dump(res, open(sys.argv[1], "wb"))
''' % (SG_SIGMAS,)
    import tempfile, pickle

    results = []
    overlay = os.environ.get("OVERLAY_SGLANG_PYTHON", "/work/sglang_wan/python")
    for pp in (pristine, overlay):
        with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
            env = dict(os.environ, PYTHONPATH=pp)
            subprocess.run([sys.executable, "-c", code, f.name], check=True, env=env)
            results.append(pickle.load(open(f.name, "rb")))
    for t in ("sde", "cps", "ode"):
        assert torch.equal(results[0][t][0], results[1][t][0]), t
        assert torch.equal(results[0][t][1], results[1][t][1]), t
