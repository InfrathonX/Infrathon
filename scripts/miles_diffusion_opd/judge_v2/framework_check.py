"""The run went through miles_diffusion's own training framework (CPU; from the in-engine / in-process audit).

The task asks for DiffusionOPD implemented in miles_diffusion: the MD driver (`train_diffusion.py`) with its Ray
RolloutManager (rollout path `miles.rollout.base_types.call_rollout_fn`) and FSDP training actor
(`FSDPTrainRayActor.train` / `.update_weights`). Candidates may change any of these, but they must stay the path that
actually runs. The audit hook records windows of those calls; this check ties the events the candidate cannot fake —
the engines' own weight updates and generation requests (or, in replay sessions, the replay server's request log) —
to them, on the same host clock:

  F1 driver       a process ran `train_diffusion.py`
  F2 rollout      every generation request of the session (engine `Req`, or replay-server `request` in replay
                  sessions) lies inside a training-rollout window (`md_rollout`, evaluation False)
  F3 training     `FSDPTrainRayActor.train` ran for every rollout id 0..R-1
  F4 publication  every engine weight update lies inside an `FSDPTrainRayActor.update_weights` window
A standalone trainer that drives the engines itself fails F1-F4.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

SLACK = 0.05  # s; all events come from time.time() on one host


def check(audit_dir: str, rounds: int, replay_log: str | None) -> dict:
    ev = []
    for f in sorted(glob.glob(os.path.join(audit_dir, "*.jsonl"))):
        for line in open(f):
            try:
                ev.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    out = {"violations": []}
    V = out["violations"].append

    drivers = [e for e in ev if e["kind"] == "proc" and any(os.path.basename(a) == "train_diffusion.py"
                                                              for a in (e.get("argv") or []) + (e.get("orig_argv") or []))]
    out["driver_pids"] = sorted({e["pid"] for e in drivers})
    if not drivers:
        V("F1 no process ran miles_diffusion's train_diffusion.py")

    def windows(kind, pred=lambda e: True):
        return sorted((e["t_start"], e["t_end"]) for e in ev if e["kind"] == kind and e.get("ok") and pred(e))

    def inside(t, wins):
        return any(a - SLACK <= t <= b + SLACK for a, b in wins)

    roll = windows("md_rollout", lambda e: not e.get("evaluation"))
    out["md_rollout_calls"] = len(roll)
    if replay_log:
        reqs = [e["t_wall"] for e in map(json.loads, open(replay_log)) if e["kind"] == "request"]
        src = "replay-server request"
    else:
        reqs = [e["t_start"] for e in ev if e["kind"] == "forward" and e.get("batch") == "Req"]
        src = "engine request"
    outside = [t for t in reqs if not inside(t, roll)]
    out["generation_requests"] = len(reqs)
    if not reqs:
        V(f"F2 no {src} recorded")
    elif outside:
        V(f"F2 {len(outside)} of {len(reqs)} {src}s outside miles_diffusion's rollout path (call_rollout_fn)")

    trained = {e.get("rollout_id") for e in ev if e["kind"] == "md_train" and e.get("ok")}
    missing = [r for r in range(rounds) if r not in trained]
    out["md_train_rollout_ids"] = len(trained)
    if missing:
        V(f"F3 FSDPTrainRayActor.train did not run for rollout ids {missing[:8]}{'...' if len(missing) > 8 else ''}")

    upd = windows("md_update_weights")
    pubs = [e["t_start"] for e in ev if e["kind"] == "forward" and e.get("batch") == "UpdateWeightFromTensorReqInput"]
    stray = [t for t in pubs if not inside(t, upd)]
    out["md_update_weights_calls"] = len(upd)
    out["engine_publications"] = len(pubs)
    if not pubs:
        V("F4 no engine weight update recorded")
    elif stray:
        V(f"F4 {len(stray)} of {len(pubs)} engine weight updates outside FSDPTrainRayActor.update_weights")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", required=True)
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--replay-log")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = check(a.audit, a.rounds, a.replay_log)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
