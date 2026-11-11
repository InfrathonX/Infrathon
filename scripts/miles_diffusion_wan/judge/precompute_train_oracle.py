"""Precompute the training-math oracles from a replay/probe dump with the author fp32 oracle (oracle_dance_fp32).

  --mode pi_old : R3 — pi_old log-prob (Dance, constants dropped, fp64 on fp32 CFG velocity) of every pair of the
                  dump at theta_0. Pairs sharing x_k bitwise (same prompt, same first-window step) share the forward.
  --mode probe  : J1 — per pair mu (fp32), s, lp, and the oracle gradient of L = -(1/N) sum A_i lp_i at theta_0 with
                  A from the official-HPSv3 oracle rewards (``--rewards hpsv3_oracle.json``); stores the flattened
                  (sorted parameter names) indices + signs of the top-q |g| coordinates and a fixed random subset.
  --mode traj   : R6 — the full replay in fp32: 4 optimizer steps (contract mini-batch order: step t trains prompts
                  [tP/S, (t+1)P/S)), per-pair PPO-clip (clip 1e-4) against the fp32 pi_old oracle (``--pi-old``), A from
                  the official-HPSv3 oracle rewards, loss = mean over the step's pairs, grad-norm clip 1.0, torch AdamW
                  (contract lr/betas/eps/wd, constant lr). Pairs sharing x_k share one forward/backward. Stores
                  Δθ_o = θ_4 − θ_0 on the fixed coordinate sample U of traj_delta, the cumulative update after every
                  step (`delta_U_steps`, [steps, |U|]; row 0 = the first step, used by the R6 step-1 decomposition),
                  and per-step loss/grad-norm/clipfrac.
Outputs ``<out>/train_oracle_<mode>.pt`` (+ .json summary). Needs one GPU with ~80 GB (traj: ~110 GB).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # oracle_dance_fp32, wan_dance_contract
sys.path.insert(0, str(Path(__file__).resolve().parent))
import oracle_dance_fp32 as O  # noqa: E402
import wan_dance_contract as C  # noqa: E402


def dump_pairs(dump: Path):
    meta = json.load(open(dump / "meta.json"))
    lat = load_file(str(dump / "latents.safetensors"))["x"]
    cond = load_file(str(dump / "cond.safetensors"))
    sig = meta["sigmas"]
    pairs = []
    for g, row in enumerate(meta["rows"]):
        base = f"{meta['run_seed']}|{meta['round']}|{row}"
        for j in range(int(meta["n"])):
            w = int(meta["windows"][g][j])
            for i, k in enumerate((w, w + 1)):
                p = O.Pair(x=lat[g, j, i], x_next=lat[g, j, i + 1], sigma=sig[k], sigma_next=sig[k + 1],
                           cond_pos=cond["pos"][g], cond_neg=cond["neg"][g])
                pairs.append(((base, j, k), g, p))
    return meta, pairs


def oracle_advantages(rewards_json: str):
    orc = json.load(open(rewards_json))
    mu_fr = torch.tensor(orc["frame_mu"], dtype=torch.float64)  # [P][n][3]
    rew = (C.REWARD_SCALE * mu_fr.mean(-1)).flatten()
    gids = [g for g in range(mu_fr.shape[0]) for _ in range(mu_fr.shape[1])]
    return C.grpo_advantages(rew, gids).view(mu_fr.shape[0], mu_fr.shape[1])


def traj(model, meta, pairs, a):
    import traj_delta as T

    dev = a.device
    contract = json.load(open(a.contract))
    o, steps = contract["optim"], int(contract["replay"]["optimizer_steps"])
    adv = oracle_advantages(a.rewards)
    lp_old = json.load(open(a.pi_old))["pi_old"]
    P = len(meta["rows"])
    per = P // steps
    layout = T.safetensors_layout(T.st_files(Path(a.model)))
    names, numels, U = T.coord_sample(layout)
    theta0_U = T.gather(T.st_files(Path(a.model)), names, numels, U)
    model.train()
    params = dict(model.named_parameters())
    assert sorted(params) == names, "diffusers parameter names differ from the checkpoint layout"
    opt = torch.optim.AdamW([params[n] for n in names], lr=o["lr"], betas=tuple(o["betas"]), eps=o["eps"],
                            weight_decay=o["weight_decay"])
    log, steps_U = [], []
    for t in range(steps):
        sel = [(pid, g, p) for pid, g, p in pairs if t * per <= g < (t + 1) * per]
        n = len(sel)
        groups: dict = {}
        for pid, g, p in sel:
            h = (g, pid[2], hashlib.sha1(p.x.float().numpy().tobytes()).hexdigest())
            groups.setdefault(h, []).append((pid, g, p))
        opt.zero_grad(set_to_none=True)
        tot, clipped = 0.0, 0
        for members in groups.values():
            p0 = members[0][2]
            v = O.velocity(model, p0.x, p0.sigma, p0.cond_pos, p0.cond_neg, dev)
            loss = 0.0
            for pid, g, p in members:
                mu, s = C.dance_mean_std(p.x.to(dev)[None], v, p.sigma, p.sigma_next)
                lp = C.dance_log_prob(p.x_next.to(dev)[None], mu, s)[0]
                old = torch.tensor(lp_old["|".join(map(str, pid))], dtype=torch.float64, device=dev)
                a_ = torch.tensor(float(adv[g, pid[1]]), dtype=torch.float64, device=dev)
                per_pair, ratio, mask = C.ppo_clip_terms(lp, old, a_, o["clip_range"])
                loss = loss + per_pair / n
                clipped += int(mask)
            loss.backward()
            tot += float(loss.detach())
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), o["grad_clip"]))
        opt.step()
        with torch.no_grad():  # cumulative update after this step, on U (R6 step-1 decomposition)
            steps_U.append((T.gather_params(params, names, numels, U) - theta0_U).float())
        log.append({"step": t, "pairs": n, "forwards": len(groups), "loss": tot, "grad_norm": gn,
                    "clipfrac": clipped / n})
        print(json.dumps(log[-1]), flush=True)
    with torch.no_grad():
        flat_idx_theta = T.gather_params(params, names, numels, U)
    delta_U = flat_idx_theta - theta0_U
    torch.save({"U_seed": T.U_SEED, "U_size": T.U_SIZE, "U": U, "delta_U": delta_U, "theta0_U": theta0_U,
                "delta_U_steps": torch.stack(steps_U), "names": names}, Path(a.out) / "train_oracle_traj.pt")
    return {"steps": log, "delta_U_absmean": float(delta_U.abs().mean()), "moved_frac": float((delta_U != 0).double().mean()),
            "U": int(U.numel())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--mode", choices=["pi_old", "probe", "traj"], required=True)
    ap.add_argument("--rewards", default=None)
    ap.add_argument("--pi-old", default=None, help="traj: train_oracle_pi_old.json of the same dump")
    ap.add_argument("--contract", default="/work/wan/judge/contract_template.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="/models/Wan2.2-TI2V-5B-Diffusers/transformer")
    ap.add_argument("--top-q", type=float, default=1e-3)
    ap.add_argument("--subset", type=int, default=10_000_000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    meta, pairs = dump_pairs(Path(a.dump))
    model = O.load_transformer(a.model, a.device)
    res = {"n_pairs": len(pairs)}
    if a.mode == "pi_old":
        lp, cache, n_fwd = {}, {}, 0
        for pid, g, p in pairs:
            h = (g, pid[2], hashlib.sha1(p.x.float().numpy().tobytes()).hexdigest())
            if h not in cache:
                with torch.no_grad():
                    cache[h] = O.velocity(model, p.x, p.sigma, p.cond_pos, p.cond_neg, a.device).cpu()
                n_fwd += 1
            v = cache[h].to(a.device)
            mu, s = C.dance_mean_std(p.x.to(a.device)[None], v[None], p.sigma, p.sigma_next)
            lp["|".join(map(str, pid))] = float(C.dance_log_prob(p.x_next.to(a.device)[None], mu, s)[0])
        res.update(n_forwards=n_fwd, pi_old=lp)
    elif a.mode == "traj":
        res.update(traj(model, meta, pairs, a))
    else:
        orc = json.load(open(a.rewards))
        mu_fr = torch.tensor(orc["frame_mu"], dtype=torch.float64)  # [P][n][3]
        rew = (C.REWARD_SCALE * mu_fr.mean(-1)).flatten()
        gids = [g for g in range(mu_fr.shape[0]) for _ in range(mu_fr.shape[1])]
        adv = C.grpo_advantages(rew, gids).view(mu_fr.shape[0], mu_fr.shape[1])
        per_pair = {}
        for pid, g, p in pairs:
            p.adv = float(adv[g, pid[1]])
            r = O.pair_logprob(model, p, device=a.device)
            per_pair["|".join(map(str, pid))] = {"lp": r["lp"], "s": r["s"], "adv": p.adv}
            torch.save(r["mu"], out / f"mu_{'_'.join(map(str, pid)).replace('|', '-')}.pt")
        loss, grads = O.loss_and_grad(model, [p for _, _, p in pairs], device=a.device)
        names = sorted(grads)
        flat = torch.cat([grads[n].flatten() for n in names])
        gen = torch.Generator().manual_seed(12345)
        U = torch.randint(0, flat.numel(), (a.subset,), generator=gen)
        thr = torch.quantile(flat[U].abs().float(), 1 - a.top_q).item()
        top = torch.nonzero(flat.abs() >= thr).flatten()
        res.update(loss=loss, per_pair=per_pair, names=names, numels=[grads[n].numel() for n in names],
                   top_threshold=thr, n_top=int(top.numel()))
        torch.save({"top_idx": top, "top_sign": torch.sign(flat[top]).to(torch.int8), "top_g": flat[top],
                    "U": U, "U_g": flat[U], "names": names}, out / "train_oracle_probe.pt")
    res["seconds"] = time.time() - t0
    (out / f"train_oracle_{a.mode}.json").write_text(json.dumps(res))
    print(json.dumps({k: v for k, v in res.items() if k not in ("pi_old", "per_pair", "names", "numels")}))


if __name__ == "__main__":
    main()
