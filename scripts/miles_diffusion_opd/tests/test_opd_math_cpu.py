"""CPU check: the reference OPD loss (miles_diffusion_opd worktree) equals the independent fp64 oracle
transcription of the sglang-miles-h3 CPS kernel, and the pre-fix SD3 divisor (1.0) does not."""

import math
import os
import sys
import types

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "oracle"))
WORKTREE = os.environ.get("MD_OPD_WORKTREE")
if not WORKTREE:
    pytest.skip("set MD_OPD_WORKTREE to a miles_diffusion 29fde6ee tree with "
                "tasks/latest/miles-diffusion-sd35m-mopd-2xh100-v2-seed/reference/md-opd-reference.patch applied",
                allow_module_level=True)
sys.path.insert(0, WORKTREE)

from sd3_oracle import cps_mean, kl_over_d  # noqa: E402

from miles.backends.fsdp_utils.loss_hub.opd import opd_loss_formula  # noqa: E402
from miles.backends.fsdp_utils.loss_hub.types import DiffusionLossContext, PreparedBatch  # noqa: E402
from miles.backends.fsdp_utils.metrics import new_metric_buffer  # noqa: E402
from miles.backends.fsdp_utils.sde_step_backend import CpsSdeStepBackend  # noqa: E402

# FlowMatchEuler shift=3, 10 steps (sigmas[i] = 3 s / (1 + 2 s), s = linspace(1, 0.1, 10) then terminal 0)
S = [1.0 - i * 0.1 for i in range(10)]
SIGMAS = [3 * s / (1 + 2 * s) for s in S] + [0.0]


def _run(divisor: float, x, v_s, v_t, idx):
    ctx = DiffusionLossContext(
        models={"transformer": None}, train_pipeline_config=None,
        sde_backend=CpsSdeStepBackend(None, sde_timestep_divisor=divisor), scheduler=None,
        args=types.SimpleNamespace(diffusion_noise_level=0.8, opd_dump_means=False), forward_dtype=torch.float32,
        device=torch.device("cpu"),
    )
    ts = torch.tensor([SIGMAS[i] * 1000 for i in idx])
    nts = torch.tensor([SIGMAS[i + 1] * 1000 for i in idx])
    prepared = PreparedBatch(latents=x, timesteps=ts, timesteps_for_model=ts, model=None, component_name="transformer",
                             guidance_scale=1.0, use_cfg=False, cfg_batching=False, true_cfg_scale=None, pos_cond=None,
                             neg_cond=None, joint_cond=None, advantage=torch.zeros(len(idx)),
                             extras={"next_timesteps": nts, "routes": ["ocr"] * len(idx)})
    batch = [{} for _ in idx]
    metrics = new_metric_buffer(None, torch.device("cpu"), ["transformer"])
    loss_sum = opd_loss_formula(ctx, batch, prepared, new_pred=v_s, ref_pred=v_t, metrics=metrics)
    return loss_sum, [p["opd_kl"] for p in batch]


def test_reference_loss_matches_oracle():
    torch.manual_seed(0)
    idx = [0, 1, 2, 3, 4]
    x = torch.randn(len(idx), 16, 64, 64)
    v_t = torch.randn_like(x)
    v_s = v_t + 0.05 * torch.randn_like(x)
    loss_sum, kls = _run(1000.0, x, v_s, v_t, idx)
    for j, i in enumerate(idx):
        s, sn = SIGMAS[i], SIGMAS[i + 1]
        want = kl_over_d(cps_mean(x[j:j+1], v_s[j:j+1], s, sn, 0.8), cps_mean(x[j:j+1], v_t[j:j+1], s, sn, 0.8), sn, 0.8)
        assert math.isclose(kls[j], want.item(), rel_tol=1e-4), (i, kls[j], want.item())
    assert math.isclose(loss_sum.item(), sum(kls), rel_tol=1e-6)
    # the pre-fix SD3 divisor (1.0) resolves sigma ~ 1000x too large: not even the same order of magnitude
    _, kls_bad = _run(1.0, x, v_s, v_t, idx)
    assert all(abs(b - g) / g > 1.0 or not math.isfinite(b) for b, g in zip(kls_bad, kls)), kls_bad


def test_gradient_flows_only_through_student():
    torch.manual_seed(1)
    x = torch.randn(2, 16, 8, 8)
    v_s = torch.randn_like(x, requires_grad=True)
    v_t = torch.randn_like(x, requires_grad=True)
    loss_sum, _ = _run(1000.0, x, v_s, v_t, [1, 3])
    loss_sum.backward()
    assert v_s.grad is not None and v_s.grad.abs().sum() > 0
    assert v_t.grad is None


if __name__ == "__main__":
    test_reference_loss_matches_oracle()
    test_gradient_flows_only_through_student()
    print("ok")
