"""J3 (real-RL audit) and J4 (transition / seed-group recomputation) from the Judge hook records of a `run`.

Inputs: the run contract, the hook directory (``audit-*.jsonl`` + ``traj/<request_id>/kNN.pt``), and the candidate's
pair-log directory. Everything compared exactly except the float transition checks (provisional tolerances, to be
calibrated from two honest engine implementations: tau_eps, euler_rel).
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import wan_dance_contract as C  # noqa: E402

EXPECT = {"width": 1280, "height": 704, "num_frames": 9, "num_inference_steps": 10, "guidance_scale": 5.0,
          "true_cfg_scale": 5.0, "negative_prompt": " ", "rollout": True, "rollout_sde_type": "dance",
          "rollout_noise_level": 1.2, "generator_device": "cpu", "num_outputs_per_prompt": 1}
TAU_EPS_MEAN = 0.02      # provisional (S4 honest: <= 0.006)
EPS_CORR_MIN = 0.999     # provisional (S4 honest: >= 0.99996)
EULER_ULPS = 2.0         # out-of-window residual in bf16 ulps of |x_{k+1}| (S4: <= 1 ulp)
# Projection of the transition residual eps_hat - eps onto the score direction S/s (coefficient = the engine's
# eta^2 error) and onto the velocity direction dsigma*v/s (coefficient = its velocity-step error). Provisional; S6d
# calibrates. The mean-abs eps test above only just caught eta instead of eta^2 (0.023 vs 0.02).
COEF_MAX = 0.01         # honest 0.001 (ref_v1_1), eta-instead-of-eta^2 plant 0.240 (S6d)


def records(audit: Path) -> list[dict]:
    out = []
    for f in sorted(audit.glob("audit-*.jsonl")):
        out += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return out


def _bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    a = x.abs().clamp_min(2.0 ** -20)
    return torch.pow(2.0, torch.floor(torch.log2(a)) - 7)


def check_run(contract: dict, audit: Path, pair_dir: Path, j4_samples: int | None = None) -> dict:
    res: dict = {}
    run = contract["run"]
    R, P, n, seed = int(run["rounds"]), int(run["prompts_per_round"]), int(run["n"]), int(run["run_seed"])
    rows = [json.loads(l) for l in open(run["prompts"])][: R * P]
    by_prompt = {r["prompt"]: (i // P, r["row"]) for i, r in enumerate(rows)}
    recs = records(audit)
    reqs = [r for r in recs if r["kind"] == "request"]

    # ---- J3a: request parameters
    bad = [(r["request_id"], k) for r in reqs for k, v in EXPECT.items() if r.get(k) != v]
    bad += [(r["request_id"], "sigmas") for r in reqs if r.get("sigmas") not in (None, [])]
    bad += [(r["request_id"], "cache") for r in reqs if r.get("enable_teacache") or r.get("enable_cache_dit")]
    res["J3_params"] = {"ok": not bad and len(reqs) == R * P * n, "n_requests": len(reqs), "violations": bad[:10]}

    # ---- J3b: canonical keys → (round, row, j), each exactly once, window canonical
    ident = {}
    viol = []
    for r in reqs:
        if r["prompt"] not in by_prompt:
            viol.append((r["request_id"], "unknown prompt"))
            continue
        rnd, row = by_prompt[r["prompt"]]
        key = C.GroupKey(seed, rnd, row)
        if r["seed"] != key.init_key():
            viol.append((r["request_id"], "init seed"))
        js = [j for j in range(n) if key.sde_key(j) == r.get("rollout_sde_noise_seed")]
        if len(js) != 1:
            viol.append((r["request_id"], "sde key"))
            continue
        j = js[0]
        if list(r.get("rollout_sde_step_indices") or []) != list(key.window(j)):
            viol.append((r["request_id"], "window"))
        ident[r["request_id"]] = (rnd, row, j, key)
    cnt = Counter((v[0], v[1], v[2]) for v in ident.values())
    res["J3_keys"] = {"ok": not viol and len(cnt) == R * P * n and all(c == 1 for c in cnt.values()),
                      "violations": viol[:10], "distinct": len(cnt)}

    # ---- J3c: publications and lag 0 (requests of round r strictly between publication r and r+1)
    pubs = [r for r in recs if r["kind"] == "publication_complete"]
    per_pid = defaultdict(list)
    for p in pubs:
        per_pid[p["pid"]].append(p)
    n_engines = len(per_pid)
    counts = sorted(len(v) for v in per_pid.values())
    res["J3_publications"] = {"ok": n_engines == 2 and all(c == R + 1 for c in counts), "per_engine": counts}
    if res["J3_publications"]["ok"]:
        done = [max(sorted(v, key=lambda x: x["t_complete"])[i]["t_complete"] for v in per_pid.values())
                for i in range(R + 1)]
        start = [min(sorted(v, key=lambda x: x["t_complete"])[i]["t_first_bucket"] for v in per_pid.values())
                 for i in range(R + 1)]
        lag = [rid for rid, (rnd, *_rest) in ident.items()
               if not (done[rnd] <= next(q["t"] for q in reqs if q["request_id"] == rid) <= start[rnd + 1])]
        res["J3_lag0"] = {"ok": not lag, "violations": len(lag)}
        res["final_publication_checksums"] = [sorted(v, key=lambda x: x["t_complete"])[-1].get("checksum")
                                              for v in per_pid.values()]

    # ---- J3d: trained pairs == generated pairs (each once, in its own round)
    ev = []
    for f in sorted(pair_dir.glob("rollout*_rank*.jsonl")):
        ev += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    steps = [e for e in ev if e["kind"] == "step"]
    trained = Counter((e["rollout"], e["group"], e["j"], e["k"]) for e in steps)
    expected = set()
    for rnd, row, j, key in ident.values():
        for k in key.window(j):
            expected.add((rnd, f"{seed}|{rnd}|{row}", j, k))
    res["J3_pairs"] = {"ok": set(trained) == expected and all(c == 1 for c in trained.values()),
                       "trained": len(trained), "expected": len(expected)}

    # ---- J4: transitions from the hook's tensors
    worst = {"x_T_mismatch": 0, "eps_corr_min": 1.0, "eps_mean_max": 0.0, "euler_ulps_max": 0.0, "missing": 0,
             "eta2_coef_absmax": 0.0, "vel_coef_absmax": 0.0}
    rids = list(ident)[: j4_samples] if j4_samples else list(ident)
    for rid in rids:
        rnd, row, j, key = ident[rid]
        d = audit / "traj" / str(rid)
        if not d.exists():
            worst["missing"] += 1
            continue
        eps = C.canonical_sde_noise(key, j)
        win = list(key.window(j))
        used = 0
        for k in range(10):
            t = torch.load(d / f"k{k:02d}.pt")
            x, v, xn = t["x"].double(), t["v"].double(), t["x_next"].double()
            if k == 0 and not torch.equal(t["x"].float(), C.canonical_init_noise(key).float()):
                worst["x_T_mismatch"] += 1
            if k in win:
                mu, s = C.dance_mean_std(x, v, t["sigma"], t["sigma_next"])
                eh = ((xn - mu) / s).flatten()
                e = eps[used].double().flatten()
                used += 1
                worst["eps_corr_min"] = min(worst["eps_corr_min"], float(torch.corrcoef(torch.stack([eh, e]))[0, 1]))
                worst["eps_mean_max"] = max(worst["eps_mean_max"], float((eh - e).abs().mean()))
                r = eh - e
                for key, direction in (("eta2_coef_absmax", C.dance_score_direction(x, v, t["sigma"], t["sigma_next"])),
                                       ("vel_coef_absmax", (t["sigma_next"] - t["sigma"]) * v)):
                    u = (direction / s).flatten()
                    worst[key] = max(worst[key], abs(float(r @ u / (u @ u))))
            else:
                pred = C.euler_step(x, v, t["sigma"], t["sigma_next"])
                # the engine forms dt*v in bf16 (0-dim promotion) then adds in fp32 and rounds to bf16: bound the
                # residual by the operands' ulps, not the (possibly cancelled) result's
                step = (t["sigma_next"] - t["sigma"]) * v
                ulps = ((xn - pred).abs() / (_bf16_ulp(x) + _bf16_ulp(step) + _bf16_ulp(pred))).max()
                worst["euler_ulps_max"] = max(worst["euler_ulps_max"], float(ulps))
    res["J4"] = {"ok": worst["missing"] == 0 and worst["x_T_mismatch"] == 0 and worst["eps_corr_min"] >= EPS_CORR_MIN
                 and worst["eps_mean_max"] <= TAU_EPS_MEAN and worst["euler_ulps_max"] <= EULER_ULPS
                 and worst["eta2_coef_absmax"] <= COEF_MAX and worst["vel_coef_absmax"] <= COEF_MAX,
                 "checked": len(rids), **worst}
    return res


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", required=True)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--j4-samples", type=int, default=None)
    a = ap.parse_args()
    print(json.dumps(check_run(json.load(open(a.contract)), Path(a.audit), Path(a.pairs), a.j4_samples), indent=1))
