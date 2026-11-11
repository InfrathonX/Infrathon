#!/usr/bin/env python3
"""Structural check of a probe dump against its batch (the fp32 oracle and provenance run separately).

P1 exactly one step dump; every batch episode trained exactly once, nothing else
P2 response region, loss mask and engine log-probs are the batch's
P3 advantages are GRPO of the batch rewards per row group
P4 the logged loss equals the fp64 recomputation from the dump (alfworld_checks)
"""

import argparse
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit"))
import alfworld_checks as oc  # noqa: E402
from alfworld_audit_check import grpo_advantages  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--out-dir", required=True, help="the probe's OUT directory")
    ap.add_argument("--contract", required=True)
    a = ap.parse_args()
    import torch

    c = json.load(open(a.contract))
    batch = json.load(open(a.batch))
    issues = []
    files = sorted(glob.glob(oc.step_files(a.out_dir)))
    if not files or any("step_000_" not in os.path.basename(f) for f in files):
        issues.append(f"expected only step_000 dumps, found {[os.path.basename(f) for f in files]}")
    groups = collections.defaultdict(list)
    for i, e in enumerate(batch["episodes"]):
        groups[e["row_id"]].append(i)
    adv = {}
    for idxs in groups.values():
        for i, v in zip(idxs, grpo_advantages([batch["episodes"][i]["reward"] for i in idxs])):
            adv[i] = v
    # Twin episodes share token sequences but carry distinct recorded log-probs: assign trained
    # sequences to batch episodes by bipartite matching on (tokens, mask, log-probs).
    want = collections.defaultdict(list)
    for i, e in enumerate(batch["episodes"]):
        want[tuple(e["tokens"])].append(i)
    trained = []
    for f in files:
        r = torch.load(f, map_location="cpu", weights_only=False)
        for tok, R, m_, elp, ad in zip(r["tokens"], r["response_lengths"], r["loss_masks"], r["engine_logprobs"], r["advantages"]):
            trained.append((tuple(int(x) for x in tok.tolist()), int(R), [int(x) for x in m_.tolist()],
                            [float(x) for x in elp.tolist()], float(ad)))

    def fits(t, i):
        e = batch["episodes"][i]
        _, R, mask, elp, _ = t
        return (R == len(e["loss_mask"]) and mask == [int(x) for x in e["loss_mask"]]
                and max((abs(x - float(y)) for x, y, mm in zip(elp, e["engine_logprobs"], mask) if mm), default=0) <= 1e-5)

    owner = {}

    def augment(ti, seen):
        for i in want.get(trained[ti][0], []):
            if i in seen or not fits(trained[ti], i):
                continue
            seen.add(i)
            if i not in owner or augment(owner[i], seen):
                owner[i] = ti
                return True
        return False

    for ti in range(len(trained)):
        if not augment(ti, set()):
            issues.append("a trained sequence matches no unused batch episode (tokens, mask and engine log-probs)")
    seen = set(owner)
    for i, ti in owner.items():
        if abs(trained[ti][4] - adv[i]) > 1e-4:
            issues.append(f"episode {i}: advantage {trained[ti][4]:.5f} != GRPO {adv[i]:.5f}")
    if len(seen) != len(batch["episodes"]):
        issues.append(f"{len(batch['episodes']) - len(seen)} batch episodes were not trained")
    steps = oc.steps_self_consistency(a.out_dir, 1, c["loss"]["eps_clip"], c["loss"]["eps_clip_high"])
    issues += steps["issues"]
    rep = {"passed": not issues, "issues": issues[:20], "episodes": len(batch["episodes"]), "trained": len(seen),
           "loss": steps["per_step"].get(0)}
    print(json.dumps(rep, indent=2))
    raise SystemExit(0 if rep["passed"] else 1)


if __name__ == "__main__":
    main()
