"""Publication integrity of a session from the in-engine audit (GPU; the fingerprint's random probes are CUDA draws).

For every engine the audit hook records each weight update (`forward` UpdateWeightFromTensorReqInput, with t_start /
t_end) and the fingerprint of the LoRA tensors it received (`lora_publish`: sum over tensors of <W, R_name>, R_name a
CUDA uniform(-0.5, 0.5) draw seeded from the tensor name). This check requires, for rounds r = 0..R-1:

  P1 order    every engine holds exactly R + 1 publications (initial + one per round), all ok, and the replay server's
              `published` call for round r comes after publication r + 1 ended on EVERY engine (replay sessions only);
  P2 content  publication 0 carries the dumped student before step 0 and publication r + 1 the student after the
              round's last optimizer step (the next step's lora_pre, or the run's last lora_post): the Judge recomputes
              the fingerprint from the dumps on its own GPU and compares it with the engines' in units of the
              fingerprint's own spread (||W||_2 / sqrt(12)). With the hook's `fingerprint16` (W rounded to fp16, the
              engine's precision: fp32 and fp16 publications of the student agree bit for bit) the match must be
              exact up to summation order (1e-9 units); the published version must also be closer to its own
              dumped version than to the neighbouring ones.

Without P1 a candidate could call `published` before its engines had the new weights (overlapping the next round in the
timed replay while behaving in the real run); without P2 it could publish anything that merely changes.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from collections import defaultdict

import torch

PEFT = "base_model.model."


def fingerprint(named: dict, device: str = "cuda", fp16: bool = False) -> float:
    """Same probe as the hook (opd_audit_hook._fingerprint), on the dumped fp32 tensors under the engine's names;
    fp16=True is the hook's `fingerprint16` (tensors rounded to fp16 first)."""
    total = 0.0
    for name, t in named.items():
        seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little") & 0x7FFFFFFFFFFFFFFF
        g = torch.Generator(device=device).manual_seed(seed)
        t = t.to(device)
        if fp16:
            t = t.to(torch.float16)
        r = torch.rand(t.shape, generator=g, device=device, dtype=torch.float32) - 0.5
        total += float((t.float() * r).double().sum().item())
    return total


def fp_unit(named: dict, device: str = "cuda") -> float:
    """||W||_2 / sqrt(12): the standard deviation of <W, R> over uniform(-0.5, 0.5) probes — the natural unit of a
    fingerprint difference (a weight change dW moves the fingerprint by ~||dW||_2 / sqrt(12))."""
    sq = sum(float(t.to(device).float().double().pow(2).sum().item()) for t in named.values())
    return (sq ** 0.5) / 12 ** 0.5


def engine_names(lora: dict) -> dict:
    """PEFT dump keys (base_model.model.<module>.lora_A.weight) -> engine names (<module>.lora_A)."""
    out = {}
    for k, v in lora.items():
        name = k[len(PEFT):] if k.startswith(PEFT) else k
        out[name[: -len(".weight")] if name.endswith(".weight") else name] = v
    return out


def published_versions(dump_dir: str, rounds: int, steps_per_round: int) -> list[dict]:
    files = sorted(glob.glob(os.path.join(dump_dir, "step_*.pt")))
    if len(files) != rounds * steps_per_round:
        raise ValueError(f"{len(files)} dumps, want {rounds * steps_per_round}")
    load = lambda i: torch.load(files[i], weights_only=True, map_location="cpu", mmap=True)
    versions = [engine_names(load(0)["lora_pre"])]
    for r in range(rounds):
        nxt = (r + 1) * steps_per_round
        rec = load(nxt) if nxt < len(files) else load(len(files) - 1)
        lora = rec["lora_pre"] if nxt < len(files) else rec.get("lora_post")
        if lora is None:
            raise ValueError("last dump has no lora_post")
        versions.append(engine_names(lora))
    return versions


def check(audit_dir: str, dump_dir: str, rounds: int, steps_per_round: int, replay_log: str | None,
          rel_tol: float, exact_tol: float = 1e-9) -> dict:
    ev = []
    for f in sorted(glob.glob(os.path.join(audit_dir, "*.jsonl"))):
        for line in open(f):
            try:
                ev.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    out = {"violations": []}
    V = out["violations"].append
    ups = defaultdict(list)
    for e in ev:
        if e["kind"] == "forward" and e.get("batch") == "UpdateWeightFromTensorReqInput":
            ups[e["port"]].append(e)
    fps = defaultdict(list)
    for e in ev:
        if e["kind"] == "lora_publish" and "fingerprint" in e:
            fps[e["pid"]].append(e)
        elif e["kind"] == "lora_publish":
            V(f"P2 engine pid {e.get('pid')}: publication without fingerprint ({e.get('error')})")
    # fp16-rounded fingerprints when the hook records them: exact up to float64 summation order
    exact = bool(fps) and all("fingerprint16" in p for ps in fps.values() for p in ps)
    key = "fingerprint16" if exact else "fingerprint"
    out["fingerprint_kind"] = key
    out["engines"] = len(ups)
    if len(ups) < 1:
        V("P1 no engine publications recorded")
        return out
    for port, es in ups.items():
        es.sort(key=lambda e: e["t_start"])
        if len(es) != rounds + 1 or not all(e["ok"] for e in es):
            V(f"P1 engine {port}: {len(es)} publications (ok {sum(e['ok'] for e in es)}), want {rounds + 1}")
    if replay_log:
        pub = {e["round"]: e["t_wall"] for e in map(json.loads, open(replay_log)) if e["kind"] == "published"}
        lead = []
        for r in range(rounds):
            ends = [es[r + 1]["t_end"] for es in ups.values() if len(es) > r + 1]
            if r not in pub or len(ends) != len(ups):
                V(f"P1 round {r}: published call or engine publication missing")
                continue
            lead.append(pub[r] - max(ends))
            if pub[r] < max(ends):
                V(f"P1 round {r}: published {max(ends) - pub[r]:.3f} s before the last engine applied the update")
        out["published_minus_last_update_s"] = {"min": min(lead, default=None), "max": max(lead, default=None)}
    versions = published_versions(dump_dir, rounds, steps_per_round)
    want = [fingerprint(v, fp16=exact) for v in versions]
    unit = [fp_unit(v) for v in versions]
    worst = 0.0
    for pid, ps in fps.items():
        ps.sort(key=lambda e: e["t"])
        got = [p[key] for p in ps]
        if len(got) != len(want):
            V(f"P2 engine pid {pid}: {len(got)} fingerprints, want {len(want)}")
            continue
        for k, g in enumerate(got):
            rel = abs(g - want[k]) / max(unit[k], 1e-30)
            worst = max(worst, rel)
            nearest = min(range(max(0, k - 1), min(len(want), k + 2)), key=lambda j: abs(g - want[j]))
            if rel > (exact_tol if exact else rel_tol) or nearest != k:
                V(f"P2 engine pid {pid} publication {k}: fingerprint {g:.6f} vs dumped version {k} {want[k]:.6f} "
                  f"(rel {rel:.2e}; nearest dumped version {nearest})")
    out["max_fingerprint_rel"] = worst
    # consecutive published versions apart, in the same unit: the tolerance must sit well below this
    out["min_version_step_rel"] = min((abs(a - b) / max(u, 1e-30) for a, b, u in zip(want, want[1:], unit[1:])),
                                      default=None)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--steps-per-round", type=int, default=2)
    ap.add_argument("--replay-log")
    ap.add_argument("--rel-tol", type=float, default=2e-3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = check(a.audit, a.dump, a.rounds, a.steps_per_round, a.replay_log, a.rel_tol)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1)[:3000])


if __name__ == "__main__":
    main()
