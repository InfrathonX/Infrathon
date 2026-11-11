#!/usr/bin/env python3
"""Build a probe batch from replayed episodes (alfworld_audit_check.py --episodes-out).

Takes whole groups (all samples of a row) of the chosen rows, keeps groups with mixed
rewards first so the step has a non-zero gradient, and writes
{"lr": LR, "episodes": [{row_id, tokens, loss_mask, engine_logprobs, reward}]} where
loss_mask / engine_logprobs cover the response region (everything after the first prompt).
"""

import argparse
import collections
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--rows", required=True, help="training jsonl the episodes were played from")
    ap.add_argument("--groups", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.rows)]
    by_row = collections.defaultdict(list)
    for e in json.load(open(a.episodes)):
        by_row[e["row"]].append(e)
    mixed = [r for r, eps in sorted(by_row.items()) if 0 < sum(e["won"] for e in eps) < len(eps)]
    flat = [r for r, eps in sorted(by_row.items()) if r not in mixed]
    chosen = (mixed + flat)[: a.groups]
    out = []
    for r in chosen:
        for e in by_row[r]:
            p = e["prompt_len"]
            out.append({"row_id": rows[r]["row_id"], "tokens": e["tokens"], "loss_mask": e["mask"][p:],
                        "engine_logprobs": e["logprobs"][p:], "reward": 1.0 if e["won"] else 0.0})
    json.dump({"lr": a.lr, "episodes": out}, open(a.out, "w"))
    print(json.dumps({"groups": len(chosen), "mixed_groups": sum(r in mixed for r in chosen), "episodes": len(out)}))


if __name__ == "__main__":
    main()
