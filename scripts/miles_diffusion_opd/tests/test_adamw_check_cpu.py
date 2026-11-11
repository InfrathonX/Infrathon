"""CPU test: adamw_check accepts real torch AdamW trajectories (foreach and fused-off variants) and flags a skipped
step, a double step and an injected foreign LoRA."""

import os
import sys
import tempfile

import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "judge"))
from adamw_check import check_run  # noqa: E402

KEYS = ["base_model.model.a.lora_A.weight", "base_model.model.a.lora_B.weight"]


def simulate(d, steps=6, foreach=True, skip_at=None, double_at=None, inject_at=None):
    torch.manual_seed(0)
    ps = [torch.nn.Parameter(torch.randn(32, 64) / 32), torch.nn.Parameter(torch.zeros(64, 32))]
    init = {k: p.detach().clone() for k, p in zip(KEYS, ps)}
    save_file(init, os.path.join(d, "init.safetensors"))
    opt = torch.optim.AdamW(ps, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4, foreach=foreach)
    for k in range(steps):
        if inject_at == k:
            with torch.no_grad():
                ps[1].add_(0.01)
        pre = {kk: p.detach().clone() for kk, p in zip(KEYS, ps)}
        for p in ps:
            p.grad = torch.randn_like(p) * 1e-3
        gn = torch.nn.utils.clip_grad_norm_(ps, 1.0).item()
        if skip_at != k:
            opt.step()
            if double_at == k:
                opt.step()
        st = [opt.state[p] for p in ps]
        rec = {"lora_pre": pre, "exp_avg": {kk: s["exp_avg"].clone() for kk, s in zip(KEYS, st)},
               "exp_avg_sq": {kk: s["exp_avg_sq"].clone() for kk, s in zip(KEYS, st)},
               "adam_step": int(st[0]["step"]), "grad_norm_preclip": gn, "lr": 1e-4, "betas": (0.9, 0.999),
               "eps": 1e-8, "weight_decay": 1e-4, "clip_grad": 1.0}
        torch.save(rec, os.path.join(d, f"step_{k:05d}.pt"))


def run(**kw):
    with tempfile.TemporaryDirectory() as d:
        simulate(d, **kw)
        return check_run(d, os.path.join(d, "init.safetensors"))


def test_honest_passes():
    for foreach in (True, False):
        r = run(foreach=foreach)
        assert not r["violations"], r["violations"]
        assert r["max_v_rel"] < 1e-5 and r["max_theta_rel_to_update"] < 1e-3 and r["max_clip_rel"] < 1e-5, r


def test_planted_fail():
    r = run(skip_at=2)  # step 0 exists, so the optimizer state is there; skipping leaves adam_step behind
    assert r["violations"], "skipped step not caught"
    r = run(double_at=3)
    assert r["violations"] or r["max_theta_rel_to_update"] > 0.1, r
    r = run(inject_at=4)
    assert r["max_theta_rel_to_update"] > 0.1, r


if __name__ == "__main__":
    test_honest_passes()
    test_planted_fail()
    print("ok")
