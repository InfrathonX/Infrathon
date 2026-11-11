"""CPU tests for judge/checks_replay.py on a synthetic honest pair log and component-level planted bugs."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "judge"))
import checks_replay as R  # noqa: E402
import wan_dance_contract as C  # noqa: E402


def _meta(P=16, n=8):
    rows = list(range(100, 100 + P))
    windows = [[C.GroupKey(7, 0, r).window_start(j) for j in range(n)] for r in rows]
    return {"run_seed": 7, "round": 0, "rows": rows, "n": n, "windows": windows}


def _honest_events(meta, steps=4, adv_fn=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    P, n = len(meta["rows"]), meta["n"]
    keys, rewards = [], []
    for gi, row in enumerate(meta["rows"]):
        base = f"7|0|{row}"
        for j in range(n):
            keys.append((base, j, gi))
            rewards.append(float(torch.rand(1, generator=g)) * 0.7)
    r = torch.tensor(rewards, dtype=torch.float64)
    adv = (adv_fn or C.grpo_advantages)(r, [k[0] for k in keys])
    per = P // steps
    ev = []
    for (base, j, gi), rew, a in zip(keys, rewards, adv.tolist()):
        w = meta["windows"][gi][j]
        step = gi // per
        for k in (w, w + 1):
            lp_old = float(torch.randn(1, generator=g)) * 0.2
            lp_new = lp_old if step == 0 else lp_old + float(torch.randn(1, generator=g)) * 3e-4
            ratio = math.exp(lp_new - lp_old)
            loss = max(-a * ratio, -a * min(max(ratio, 1 - C.PPO_CLIP), 1 + C.PPO_CLIP))
            if step != 0:
                ev.append({"kind": "pi_old", "group": base, "j": j, "k": k, "lp_old": lp_old})
            ev.append({"kind": "step", "optim_step": step, "group": base, "j": j, "k": k, "adv": a, "raw_reward": rew,
                       "lp_old": lp_old, "lp_new": lp_new, "ratio": ratio, "clipped": abs(ratio - 1) > C.PPO_CLIP,
                       "loss": loss})
    return ev


def _write(tmp: Path, ev):
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "rollout000_rank0.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ev))
    return tmp


def test_honest_passes(tmp_path):
    meta = _meta()
    res = R.check_replay(meta, _write(tmp_path / "h", _honest_events(meta)), 4)
    assert all(v["ok"] for k, v in res.items() if "ok" in v), res


def test_planted_missing_pair(tmp_path):
    meta = _meta()
    ev = [e for i, e in enumerate(_honest_events(meta)) if not (e["kind"] == "step" and i == 5)]
    assert not R.check_replay(meta, _write(tmp_path / "m", ev), 4)["pairs_exactly_once"]["ok"]


def test_planted_per_group_std(tmp_path):
    meta = _meta()

    def per_group(r, ids):
        out = torch.empty_like(r)
        for gid in set(ids):
            idx = [i for i, x in enumerate(ids) if x == gid]
            out[idx] = (r[idx] - r[idx].mean()) / (r[idx].std() + 1e-4)
        return out.clamp(-5, 5)

    res = R.check_replay(meta, _write(tmp_path / "s", _honest_events(meta, adv_fn=per_group)), 4)
    assert not res["R2_advantage"]["ok"]


def test_planted_wrong_loss_and_order(tmp_path):
    meta = _meta()
    ev = _honest_events(meta)
    for e in ev:
        if e["kind"] == "step" and e["optim_step"] == 2:
            e["loss"] = -e["adv"] * e["ratio"] * 1.1
    assert not R.check_replay(meta, _write(tmp_path / "l", ev), 4)["R4_self_consistency"]["ok"]
    ev = _honest_events(meta)
    for e in ev:
        if e["kind"] == "step":
            e["optim_step"] = (e["optim_step"] + 1) % 4
    assert not R.check_replay(meta, _write(tmp_path / "o", ev), 4)["minibatch_prompt_order"]["ok"]
