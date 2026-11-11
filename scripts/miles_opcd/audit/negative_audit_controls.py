#!/usr/bin/env python3
"""Offline negative controls for opcd_audit_check.

Starts from a run that passes, rewrites its audit log / training dumps to look
like a specific rule violation, runs the checker on each variant, and reports
which check fired. Every variant must fail, and on the intended check.
"""

import argparse
import copy
import glob
import json
import os
import shutil
import subprocess
import sys

import torch


def load_audit(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        out[os.path.basename(f)] = [json.loads(l) for l in open(f) if l.strip()]
    return out


def write_audit(recs_by_file, d):
    os.makedirs(d, exist_ok=True)
    for name, recs in recs_by_file.items():
        with open(os.path.join(d, name), "w") as fh:
            for r in recs:
                fh.write(json.dumps(r, separators=(",", ":")) + "\n")


def link_dumps(src, dst, replace=None):
    os.makedirs(dst, exist_ok=True)
    replace = replace or {}
    for f in sorted(glob.glob(os.path.join(src, "rollout_*_rank_*.pt"))):
        name = os.path.basename(f)
        if name in replace:
            torch.save(replace[name], os.path.join(dst, name))
        elif replace.get("__drop__") and name in replace["__drop__"]:
            continue
        else:
            os.symlink(f, os.path.join(dst, name))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit-dir", required=True)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--checker-args", required=True, help="JSON list of the remaining checker CLI args")
    args = ap.parse_args()

    base = load_audit(args.audit_dir)
    main_file = max(base, key=lambda k: len(base[k]))
    recs = base[main_file]
    gen_idx = [i for i, r in enumerate(recs) if r["kind"] == "dispatch" and not str(r.get("rid", "")).startswith("HEALTH")]
    fin_by_rid = {r["rid"]: i for i, r in enumerate(recs) if r["kind"] == "final"}
    d0 = torch.load(os.path.join(args.dump_dir, "rollout_0_rank_0.pt"), map_location="cpu", weights_only=False)
    ctx = None

    variants = {}

    def v(name, expect, audit=None, dumps=None):
        variants[name] = (expect, audit, dumps)

    # R1: top-k taken from the engine
    a = copy.deepcopy(base); a[main_file][gen_idx[5]]["top_logprobs_num"] = 256
    v("engine top-k (top_logprobs_num=256)", "W2", audit=a)
    # sampling not T=1
    a = copy.deepcopy(base)
    for i in gen_idx[:512]:
        a[main_file][i]["sampling"]["temperature"] = 0.8
    v("sampling at T=0.8", "W2", audit=a)
    # oversampling: 64 extra completed generations that were never trained on
    a = copy.deepcopy(base)
    extra = []
    for k, i in enumerate(gen_idx[:64]):
        dd = copy.deepcopy(recs[i]); dd["rid"] = f"extra{k}"
        ff = copy.deepcopy(recs[fin_by_rid[recs[i]["rid"]]]); ff["rid"] = f"extra{k}"
        extra += [dd, ff]
    a[main_file] = a[main_file] + extra
    v("oversample 64 and discard", "W3", audit=a)
    # reuse: step 1 trains on step 0's samples again
    v("reuse step 0 data at step 1", "W3", dumps={"rollout_1_rank_0.pt": d0})
    # drop one trained sample from step 3
    d3 = torch.load(os.path.join(args.dump_dir, "rollout_3_rank_0.pt"), map_location="cpu", weights_only=False)
    d3m = copy.deepcopy(d3)
    for key, val in d3m["rollout_data"].items():
        if isinstance(val, list) and len(val) == len(d3["rollout_data"]["tokens"]):
            d3m["rollout_data"][key] = val[:-1]
    v("drop one sample at step 3", "W3", dumps={"rollout_3_rank_0.pt": d3m})
    # stale by 2: publish only every other step (weight calls of odd publications removed)
    a = copy.deepcopy(base)
    w_idx = [i for i, r in enumerate(recs) if r["kind"] == "weights"]
    groups, cur, last_t = [], [], None
    for i in w_idx:
        if cur and recs[i]["t_start"] - recs[cur[-1]]["t_end"] > 1.0:
            groups.append(cur); cur = []
        cur.append(i)
    groups.append(cur)
    drop = set(j for g in groups[2::2] for j in g)
    a[main_file] = [r for i, r in enumerate(recs) if i not in drop]
    # the remaining calls must be renumbered as the hook would have counted them
    cnt = 0
    for r in a[main_file]:
        if r["kind"] == "weights" and r.get("success"):
            cnt += 1; r["weight_calls"] = cnt
    # dispatch/final counts shift accordingly
    cnt = 0
    for r in a[main_file]:
        if r["kind"] == "weights" and r.get("success"):
            cnt += 1
        elif r["kind"] in ("dispatch", "final"):
            r["weight_calls"] = cnt
    v("publish every other step (stale by 2)", "W4/W5", audit=a)
    # split every publication into two halves with a pause in between (fake freshness)
    a = copy.deepcopy(base)
    out = []
    for i, r in enumerate(recs):
        out.append(r)
        if r["kind"] == "weights" and any(i == g[len(g) // 2 - 1] for g in groups[1:]):
            out.append({"kind": "pause", "t": r["t_end"] + 1e-4, "model_path": r["model_path"], "weight_calls": r["weight_calls"]})
    a[main_file] = out
    v("split each publication in two", "W5", audit=a)
    # context leak: first sample's prompt carries a fake prefix (prompt then matches no row)
    a = copy.deepcopy(base); a[main_file][gen_idx[0]]["input_ids"] = [151644, 8948, 198] + a[main_file][gen_idx[0]]["input_ids"]
    v("rewritten student prompt", "W1", audit=a)

    rows = []
    for name, (expect, audit, dumps) in variants.items():
        wd = os.path.join(args.work, name.split(" (")[0].replace(" ", "_").replace("/", "_"))
        shutil.rmtree(wd, ignore_errors=True)
        ad, dd = os.path.join(wd, "audit"), os.path.join(wd, "dumps")
        if audit is not None:
            write_audit(audit, ad)
        else:
            os.makedirs(wd, exist_ok=True); os.symlink(args.audit_dir, ad)
        link_dumps(args.dump_dir, dd, replace=dumps)
        cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "opcd_audit_check.py"),
               "--audit-dir", ad, "--dumps", os.path.join(dd, "rollout_*_rank_*.pt"), *json.loads(args.checker_args)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        try:
            rep = json.loads(res.stdout[res.stdout.index("{"):])
        except Exception:
            rep = {"passed": None, "issue_counts": {"crash": res.stderr[-300:]}}
        fired = sorted(rep.get("issue_counts", {}))
        ok = rep.get("passed") is False and any(e in fired for e in expect.split("/"))
        rows.append((name, expect, ",".join(fired) or "-", "REJECTED" if ok else "!! NOT CAUGHT"))
        shutil.rmtree(wd, ignore_errors=True)
    w = max(len(r[0]) for r in rows)
    for r in rows:
        print(f"{r[0]:<{w}}  expect {r[1]:<6} fired {r[2]:<10} {r[3]}")
    sys.exit(0 if all(r[3] == "REJECTED" for r in rows) else 1)


if __name__ == "__main__":
    main()
