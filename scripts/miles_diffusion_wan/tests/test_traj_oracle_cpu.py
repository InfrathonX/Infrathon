"""CPU check of the R6 trajectory oracle (precompute_train_oracle --mode traj) on a tiny random WanTransformer3DModel:
the deduplicated 4-step fp32 replay equals a plain per-pair reimplementation, step 0 is unclipped (ratio 1), and
traj_delta gathers the same Δθ from the saved checkpoint layout."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "judge"))
import oracle_dance_fp32 as O  # noqa: E402
import wan_dance_contract as C  # noqa: E402

diffusers = pytest.importorskip("diffusers")
from safetensors.torch import save_file  # noqa: E402

SIG = [1.0, 0.97574, 0.94627, 0.90971, 0.86313, 0.80179, 0.71732, 0.59362, 0.39510, 0.02441, 0.0]
P, N, SHAPE = 4, 3, (48, 3, 4, 6)


def _tiny():
    torch.manual_seed(0)
    return diffusers.WanTransformer3DModel(
        patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=16, in_channels=48, out_channels=48,
        text_dim=32, freq_dim=32, ffn_dim=64, num_layers=2, cross_attn_norm=True, qk_norm="rms_norm_across_heads",
        rope_max_seq_len=64,
    ).float()


def _fixture(tmp: Path):
    model_dir = tmp / "transformer"
    _tiny().save_pretrained(model_dir, safe_serialization=True)
    g = torch.Generator().manual_seed(7)
    # samples j=0 and j=1 of each prompt share x_w (same window start): exercises the forward dedup
    windows = [[1, 1, 3], [0, 0, 2], [2, 2, 4], [3, 3, 0]]
    lat = torch.empty(P, N, 3, *SHAPE)
    for p in range(P):
        for j in range(N):
            x0 = torch.randn(SHAPE, generator=g) if j != 1 else lat[p, 0, 0].clone()
            lat[p, j, 0] = x0
            lat[p, j, 1] = x0 + 0.1 * torch.randn(SHAPE, generator=g)
            lat[p, j, 2] = lat[p, j, 1] + 0.1 * torch.randn(SHAPE, generator=g)
    dump = tmp / "dump"
    dump.mkdir()
    save_file({"x": lat.bfloat16().contiguous()}, str(dump / "latents.safetensors"))
    save_file({"pos": torch.randn(P, 7, 32, generator=g), "neg": torch.randn(P, 7, 32, generator=g)},
              str(dump / "cond.safetensors"))
    (dump / "meta.json").write_text(json.dumps({"rows": list(range(P)), "n": N, "windows": windows, "sigmas": SIG,
                                                "run_seed": 5, "round": 0}))
    rew = tmp / "hpsv3_oracle.json"
    rew.write_text(json.dumps({"frame_mu": torch.randn(P, N, 3, generator=g).tolist()}))
    contract = json.loads((HERE / "judge" / "contract_template.json").read_text())
    contract["optim"]["lr"] = 1e-3  # visible updates on a tiny model; same code path
    (tmp / "contract.json").write_text(json.dumps(contract))
    return model_dir, dump, rew, tmp / "contract.json", contract


def _run(*args):
    subprocess.run([sys.executable, str(HERE / "judge" / "precompute_train_oracle.py"), *map(str, args),
                    "--device", "cpu"], check=True, capture_output=True, text=True)


def test_traj_matches_plain_reimplementation(tmp_path):
    model_dir, dump, rew, cpath, contract = _fixture(tmp_path)
    _run("--dump", dump, "--mode", "pi_old", "--model", model_dir, "--out", tmp_path / "o")
    _run("--dump", dump, "--mode", "traj", "--model", model_dir, "--out", tmp_path / "o", "--rewards", rew,
         "--pi-old", tmp_path / "o" / "train_oracle_pi_old.json", "--contract", cpath)
    summary = json.loads((tmp_path / "o" / "train_oracle_traj.json").read_text())
    got = torch.load(tmp_path / "o" / "train_oracle_traj.pt")
    assert summary["steps"][0]["clipfrac"] == 0.0
    assert all(s["forwards"] < s["pairs"] for s in summary["steps"])  # dedup happened

    # plain reimplementation: every pair its own forward, no grouping
    import precompute_train_oracle as PTO

    meta, pairs = PTO.dump_pairs(dump)
    model = O.load_transformer(str(model_dir), "cpu").train()
    lp_old = json.loads((tmp_path / "o" / "train_oracle_pi_old.json").read_text())["pi_old"]
    adv = PTO.oracle_advantages(str(rew))
    o = contract["optim"]
    names = sorted(dict(model.named_parameters()))
    params = dict(model.named_parameters())
    theta0 = {k: v.detach().clone() for k, v in params.items()}
    opt = torch.optim.AdamW([params[k] for k in names], lr=o["lr"], betas=tuple(o["betas"]), eps=o["eps"],
                            weight_decay=o["weight_decay"])
    steps = contract["replay"]["optimizer_steps"]
    for t in range(steps):
        sel = [(pid, g, p) for pid, g, p in pairs if t * (P // steps) <= g < (t + 1) * (P // steps)]
        opt.zero_grad(set_to_none=True)
        for pid, g, p in sel:
            v = O.velocity(model, p.x, p.sigma, p.cond_pos, p.cond_neg, "cpu")
            mu, s = C.dance_mean_std(p.x[None], v, p.sigma, p.sigma_next)
            lp = C.dance_log_prob(p.x_next[None], mu, s)[0]
            per, _, _ = C.ppo_clip_terms(lp, torch.tensor(lp_old["|".join(map(str, pid))], dtype=torch.float64),
                                         torch.tensor(float(adv[g, pid[1]]), dtype=torch.float64), o["clip_range"])
            (per / len(sel)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), o["grad_clip"])
        opt.step()
    import traj_delta as T

    ref = T.gather_params(params, names, [params[k].numel() for k in names], got["U"]) - \
        T.gather_params(theta0, names, [theta0[k].numel() for k in names], got["U"])
    assert ref.abs().max() > 0
    # summation order differs (shared forwards): Adam amplifies it on near-cancelling coordinates, so compare the
    # distribution (observed: median 0, q99 2e-8, max 1.4e-6 against updates of ~1e-3); a weighting bug moves the median
    d = (got["delta_U"] - ref).abs()
    scale = float(ref.abs().max())
    assert float(torch.quantile(d.float(), 0.5)) <= 1e-7 * scale
    assert float(torch.quantile(d.float(), 0.99)) <= 1e-4 * scale
    assert float(d.max()) <= 1e-2 * scale
    assert float(torch.corrcoef(torch.stack([got["delta_U"], ref]))[0, 1]) > 0.999999
    # per-step cumulative updates: last row is the final update; first row is a single AdamW step
    assert got["delta_U_steps"].shape[0] == contract["replay"]["optimizer_steps"]
    assert torch.allclose(got["delta_U_steps"][-1].double(), got["delta_U"], atol=1e-7)
    import traj_delta as T2
    dec = T2.step1_decomposition(got["delta_U"], got["delta_U_steps"][0].double())
    assert 0.0 < dec["step1_resid"] < 1.0
    assert T2.step1_decomposition(3 * got["delta_U_steps"][0].double(), got["delta_U_steps"][0].double())["step1_resid"] < 1e-6
    # theta0 on U gathered from the checkpoint files equals the in-memory theta0
    assert torch.equal(got["theta0_U"], T.gather_params(theta0, names, [theta0[k].numel() for k in names], got["U"]))
