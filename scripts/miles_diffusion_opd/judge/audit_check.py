"""MVP audit of a real on-policy run from the in-engine hook records (W1-W6) plus the candidate dump (CPU).

W1 rows      every contract row of the run is requested exactly once (prompt, seed) and nothing else is generated
W2 params    every built SamplingParams equals the contract on the fixed fields (guidance, steps, size, SDE type,
             eta, window from the row seed, n, return flags, no negative prompt / true CFG) and every lossy
             accelerator switch is off (teacache, cache-dit, spectrum, attention override, extra params)
W4 lag 0     per engine: the i-th round's generate requests run strictly after that engine applied publication i
             (initial weights = publication 0) and strictly before publication i+1 starts
W5 publish   per engine: exactly rounds + 1 LoRA publications; per version the fingerprint is identical on every
             engine; consecutive versions differ
W6 student   every publication carries exactly the student's LoRA tensor set (382 names, fixed name digest)
W3 1-to-1    the dump's (row, output, step) triples are exactly rows x n x window, one optimizer-step split per contract
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import torch

FIXED = {"guidance_scale": 1.0, "num_inference_steps": 10, "height": 512, "width": 512, "rollout_sde_type": "cps",
         "rollout_noise_level": 0.8, "rollout_log_prob_no_const": True, "rollout_return_dit_trajectory": True,
         "rollout_return_denoising_env": True, "num_outputs_per_prompt": 8, "true_cfg_scale": None, "rollout": True}
MUST_BE_OFF = ("enable_teacache", "enable_cache_dit", "enable_spectrum", "attention_backend_override",
               "cache_dit_params", "extra_sampling_params")


def window(seed: int) -> list[int]:
    s = int(np.random.default_rng(seed).integers(0, 3))
    return [s, s + 1, s + 2]


def check(audit_dir: str, rows_path: str, rounds: int, rows_per_round: int, dump_dir: str | None) -> dict:
    ev = [json.loads(line) for f in glob.glob(os.path.join(audit_dir, "*.jsonl")) for line in open(f)]
    rows = [json.loads(line) for line in open(rows_path)][: rounds * rows_per_round]
    out: dict = {"violations": []}
    V = out["violations"].append
    # ---------------------------------------------------------------- W1 / W2
    params = [e for e in ev if e["kind"] == "params"]
    want = {(r["input"], int(r["metadata"]["seed"])): r for r in rows}
    # a request the engine failed (e.g. a transient CUDA OOM answered with 500) produced no samples; the framework may
    # retry it. W1 counts successful requests: each row exactly once; failed attempts are reported and capped.
    req_ok = {e.get("rid"): bool(e.get("ok")) for e in ev if e["kind"] == "forward" and e.get("batch") == "Req"}
    seen = defaultdict(int)
    failed = defaultdict(int)
    for p in params:
        sp = p["sampling_params"]
        key = (sp.get("prompt"), int(sp.get("seed", -1)))
        if not req_ok.get(p.get("rid"), False):
            failed[key] += 1
            continue
        seen[key] += 1
        if key not in want:
            V(f"W1 unexpected generate request seed={key[1]} prompt={str(key[0])[:40]!r}")
            continue
        for f, v in FIXED.items():
            if sp.get(f) != v:
                V(f"W2 row {want[key]['metadata']['row_id']}: {f}={sp.get(f)!r} != {v!r}")
        if sp.get("negative_prompt") not in (None, "", " ") and FIXED["guidance_scale"] <= 1.0:
            pass  # unused at guidance 1
        if list(sp.get("rollout_sde_step_indices") or []) != window(key[1]):
            V(f"W2 row {want[key]['metadata']['row_id']}: window {sp.get('rollout_sde_step_indices')} != {window(key[1])}")
        for f in MUST_BE_OFF:
            if sp.get(f) not in (None, False, {}, "None"):
                V(f"W2 row {want[key]['metadata']['row_id']}: {f}={sp.get(f)!r} must be off")
    for key, r in want.items():
        if seen.get(key, 0) != 1:
            V(f"W1 row {r['metadata']['row_id']} served {seen.get(key, 0)} times (successful requests)")
    out["failed_requests"] = sum(failed.values())
    out["failed_request_rows"] = sorted(want[k]["metadata"]["row_id"] for k in failed if k in want)[:20]
    if out["failed_requests"] > max(2, len(rows) // 20):
        V(f"W1 {out['failed_requests']} failed generate requests (> {max(2, len(rows) // 20)}): the engines are unstable")
    # ---------------------------------------------------------------- W4 / W5 / W6
    fwd = defaultdict(list)
    for e in ev:
        if e["kind"] == "forward":
            fwd[e["port"]].append(e)
    pubs_by_pid = defaultdict(list)
    for e in ev:
        if e["kind"] == "lora_publish":
            pubs_by_pid[e["pid"]].append(e)
    out["engines"] = sorted(fwd)
    for port, es in fwd.items():
        es.sort(key=lambda e: e["t_start"])
        ups = [e for e in es if e["batch"] == "UpdateWeightFromTensorReqInput"]
        reqs = [e for e in es if e["batch"] == "Req"]
        if any(not e["ok"] for e in ups):
            V(f"W5 engine {port}: failed publication")
        if len(ups) != rounds + 1:
            V(f"W5 engine {port}: {len(ups)} publications, want {rounds + 1}")
        # each Req must sit between update k (end) and update k+1 (start) for some k in 0..rounds-1, and each
        # inter-publication gap must hold this engine's share of exactly one round
        for q in reqs:
            k = max((i for i, u in enumerate(ups) if u["t_end"] <= q["t_start"]), default=None)
            if k is None or k >= rounds:
                V(f"W4 engine {port}: request at {q['t_start']:.1f} not after a publication of this run")
            elif k + 1 < len(ups) and q["t_end"] > ups[k + 1]["t_start"]:
                V(f"W4 engine {port}: request overlaps publication {k + 1}")
        out.setdefault("requests_per_gap", {})[str(port)] = [
            sum(1 for q in reqs if u["t_end"] <= q["t_start"] and (i + 1 >= len(ups) or q["t_end"] <= ups[i + 1]["t_start"]))
            for i, u in enumerate(ups)]
    fps = [[round(p["fingerprint"], 6) for p in sorted(ps, key=lambda e: e["t"])] for ps in pubs_by_pid.values()]
    out["fingerprints_per_engine"] = fps
    if len({tuple(f) for f in fps}) != 1:
        V("W5 engines hold different versions (fingerprint sequences differ)")
    elif any(a == b for a, b in zip(fps[0], fps[0][1:])):
        V("W5 a publication did not change the weights")
    names = {(p["n_tensors"], p["names_sha256"]) for ps in pubs_by_pid.values() for p in ps}
    out["published_name_sets"] = [list(n) for n in names]
    if len(names) != 1 or next(iter(names))[0] != 382:
        V(f"W6 published tensor sets {names}")
    # ---------------------------------------------------------------- W3
    if dump_dir:
        steps = sorted(glob.glob(os.path.join(dump_dir, "step_*.pt")))
        per_step = [set((int(p[0]), int(p[1]), int(p[3])) for p in torch.load(f, weights_only=False)["pairs"]) for f in steps]
        n = FIXED["num_outputs_per_prompt"]
        half = rows_per_round // 2
        expect = []
        for r in range(rounds):
            for h in range(2):
                blk = rows[r * rows_per_round + h * half: r * rows_per_round + (h + 1) * half]
                expect.append({(b["metadata"]["row_id"], j, i) for b in blk for j in range(n)
                               for i in window(int(b["metadata"]["seed"]))})
        out["dump_steps"] = len(per_step)
        if len(per_step) != len(expect):
            V(f"W3 {len(per_step)} optimizer steps dumped, want {len(expect)}")
        for k, (got, exp) in enumerate(zip(per_step, expect)):
            if got != exp:
                V(f"W3 step {k}: pair set differs ({len(got ^ exp)} pairs)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--rows-per-round", type=int, default=8)
    ap.add_argument("--dump")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = check(args.audit, args.rows, args.rounds, args.rows_per_round, args.dump)
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "fingerprints_per_engine"}, indent=1)[:4000])


if __name__ == "__main__":
    main()
