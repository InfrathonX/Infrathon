"""Replay-output gates computable on CPU from the candidate's per-pair log (R2, R4, R5 structure).

Inputs: the dump's ``meta.json`` (what the Judge released), the candidate's pair-log directory for this replay
(``rollout*_rank*.jsonl``: ``pi_old`` and ``step`` events), and the Judge engines' hook records.
R1 (per-frame mu vs the official HPSv3 oracle) and R3 (pi_old vs the fp32 oracle) need precomputed oracles and are
separate. Returns a dict of named checks {name: {"ok": bool, ...}}.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import wan_dance_contract as C  # noqa: E402


def load_events(pair_dir: Path) -> list[dict]:
    ev = []
    for f in sorted(pair_dir.glob("rollout*_rank*.jsonl")):
        ev += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return ev


def expected_pairs(meta: dict) -> dict[tuple, int]:
    """(group, j, k) -> prompt index g, from the canonical keys the dump was built with."""
    out = {}
    for g, row in enumerate(meta["rows"]):
        key = C.GroupKey(int(meta["run_seed"]), int(meta["round"]), int(row))
        base = f"{key.run_seed}|{key.round}|{key.prompt_row}"
        for j in range(int(meta["n"])):
            w = int(meta["windows"][g][j])
            for k in (w, w + 1):
                out[(base, j, k)] = g
    return out


def check_replay(meta: dict, pair_dir: Path, optimizer_steps: int, n: int = 8, clip_range: float = C.PPO_CLIP,
                 rel_tol: float = 1e-5) -> dict:
    res = {}
    ev = load_events(pair_dir)
    steps = [e for e in ev if e["kind"] == "step"]
    pi_old = [e for e in ev if e["kind"] == "pi_old"]
    exp = expected_pairs(meta)

    # ---- coverage: every expected pair trained exactly once, nothing else
    got = Counter((e["group"], e["j"], e["k"]) for e in steps)
    res["pairs_exactly_once"] = {
        "ok": set(got) == set(exp) and all(v == 1 for v in got.values()),
        "expected": len(exp), "got": len(got), "dup": sum(1 for v in got.values() if v > 1),
        "missing": len(set(exp) - set(got)), "extra": len(set(got) - set(exp)),
    }

    # ---- mini-batch k = prompts [P/steps*k, P/steps*(k+1))
    P = len(meta["rows"])
    per = P // optimizer_steps
    bad = [e for e in steps if (e["group"], e["j"], e["k"]) in exp and exp[(e["group"], e["j"], e["k"])] // per != e["optim_step"]]
    res["minibatch_prompt_order"] = {"ok": not bad, "violations": len(bad)}

    # ---- R2: advantages recomputed from the candidate's own rewards (fp64)
    reward = {}
    for e in steps:
        reward[(e["group"], e["j"])] = e["raw_reward"]
    groups = defaultdict(list)
    for (base, j), r in reward.items():
        groups[base].append((j, r))
    keys = sorted(reward)
    r_all = torch.tensor([reward[k] for k in keys], dtype=torch.float64)
    adv_ref = C.grpo_advantages(r_all, [k[0] for k in keys])
    adv_map = {k: float(a) for k, a in zip(keys, adv_ref)}
    errs = [abs(e["adv"] - adv_map[(e["group"], e["j"])]) / max(1.0, abs(adv_map[(e["group"], e["j"])])) for e in steps]
    res["R2_advantage"] = {"ok": bool(errs) and max(errs) <= rel_tol, "max_rel": max(errs) if errs else None,
                           "n_groups": len(groups), "complete_groups": all(len(v) == n for v in groups.values())}

    # ---- R4: per-pair fp64 self-consistency of the logged PPO terms
    worst = 0.0
    viol = 0
    for e in steps:
        r = math.exp(e["lp_new"] - e["lp_old"])
        a = e["adv"]
        loss = max(-a * r, -a * min(max(r, 1 - clip_range), 1 + clip_range))
        err_r = abs(r - e["ratio"]) / max(1.0, abs(r))
        err_l = abs(loss - e["loss"]) / max(1e-6, abs(loss))
        worst = max(worst, err_r, min(err_l, abs(loss - e["loss"])))
        if (abs(r - 1) > clip_range) != bool(e["clipped"]) and abs(abs(r - 1) - clip_range) > 1e-9:
            viol += 1
    res["R4_self_consistency"] = {"ok": worst <= 1e-4 and viol == 0, "worst": worst, "clip_mask_violations": viol}

    # ---- step 0 reuses its forward as pi_old (ratio == 1); later steps' pi_old come from pi_old events
    s0 = [e for e in steps if e["optim_step"] == 0]
    res["step0_ratio_one"] = {"ok": all(abs(e["ratio"] - 1.0) < 1e-9 for e in s0), "n": len(s0)}
    po = {(e["group"], e["j"], e["k"]): e["lp_old"] for e in pi_old}
    later = [e for e in steps if e["optim_step"] != 0]
    mism = [e for e in later if (e["group"], e["j"], e["k"]) not in po
            or abs(po[(e["group"], e["j"], e["k"])] - e["lp_old"]) > 1e-9 * max(1.0, abs(e["lp_old"]))]
    res["pi_old_logged_and_used"] = {"ok": not mism, "violations": len(mism), "n_pi_old_events": len(pi_old)}
    res["pi_old_all"] = {  # all 256 pi_old values for R3 (oracle comparison happens elsewhere)
        "n": len(set(po) | {(e["group"], e["j"], e["k"]) for e in s0})}
    return res


def check_publications(hook_pubs: list[dict], n_engines: int, t_go: float, t_log_marker: float | None) -> dict:
    new = [p for p in hook_pubs if p["t_complete"] >= t_go]
    ok = len(new) == n_engines
    if t_log_marker is not None:
        ok = ok and all(p["t_first_bucket"] >= t_log_marker for p in new)
    return {"R5_publication": {"ok": ok, "n": len(new),
                               "first_bucket_after_log": None if t_log_marker is None else
                               all(p["t_first_bucket"] >= t_log_marker for p in new)}}


def log_digest(pair_dir: Path) -> dict:
    """sha256 of every per-pair log file of one replay (the files load_events reads); None for an unreadable one."""
    files = {}
    for f in sorted(pair_dir.glob("rollout*_rank*.jsonl")):
        try:
            files[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
        except OSError:
            files[f.name] = None
    return {"files": files}


def check_log_final(log_at_log_complete: dict, log_final: dict) -> dict:
    """R5: T ends at the later of the engines' publication records and `log_complete`. The records are written in a
    directory the candidate's uid can also write, so the Judge's own evidence is `log_complete` (the Judge hashes the
    log itself when the event arrives): the per-pair log that R1-R4 check must already be complete, byte for byte, when
    it arrives, and T never ends before the logged work."""
    files = log_at_log_complete.get("files") or {}
    same = bool(files) and None not in files.values() and log_at_log_complete == log_final
    return {"R5_log_final": {"ok": same, "files": sorted(files)}}

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--steps", type=int, default=4)
    a = ap.parse_args()
    print(json.dumps(check_replay(json.load(open(a.meta)), Path(a.pairs), a.steps), indent=1))
