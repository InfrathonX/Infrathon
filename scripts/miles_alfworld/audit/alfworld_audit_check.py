#!/usr/bin/env python3
"""Judge-side check of an ALFWorld run against the SGLang audit log.

Inputs: the audit JSONL written by ``alfworld_audit_hook`` inside the SGLang processes,
the per-step training dumps, the training rows (line order fixes the row->round
assignment), the game data and the pinned tokenizer. No candidate code is imported;
missing evidence fails closed.

Checks
  E1 episodes  requests chain into episodes by exact token prefix; the first request of
               every episode is the spec rendering of an assigned game's initial
               observation; every assistant output, parsed with the spec rule and
               replayed in the Judge's own ALFWorld, yields exactly the next request's
               appended tokens; episodes end exactly on a win or at the step cap
  E2 sampling  temperature 1, top_p 1, no top-k/min-p truncation, no top-logprob or
               token-id-logprob options, no sessions, at most max_turn_tokens per turn,
               no other generation requests (health probes excepted)
  E3 work      every training row of round r is played exactly samples_per_game times,
               only in round r; every episode is trained exactly once, at step r
  E4 staleness every turn of an episode trained at step s was generated entirely under
               version s-1 or s (no publication inside a turn)
  E5 publish   one publication per step after the first training generation
  E6 dumps     each trained sequence == the episode's final tokens; its loss mask == the
               assistant-output spans; engine log-probs == the engine's; advantage ==
               GRPO of the replayed rewards
  T            first training generation dispatch -> end of the last publication, minus
               evaluator-owned intervals (--exclude-intervals); must be positive
"""

import argparse
import bisect
import collections
import glob
import json
import math
import os
import re
import sys
from multiprocessing import get_context

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_check import IM_START, Game, Renderer, _h, parse_action  # noqa: E402


def load_audit(audit_dir):
    recs = []
    files = sorted(glob.glob(os.path.join(audit_dir, "*.jsonl")))
    if not files:
        raise AssertionError(f"no audit files in {audit_dir}")
    for f in files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    return recs


def is_health_probe(d):
    s = d.get("sampling") or {}
    return list(d.get("input_ids") or []) == [0] and s.get("max_new_tokens") == 1


_BASE = np.uint64(1000003)
_POW = np.ones(1, dtype=np.uint64)


def prefix_hashes(ids):
    """H[n-1] = sum_{j<n} ids[j] * B^j (mod 2^64) for every prefix length n (vectorised)."""
    global _POW
    arr = np.asarray(ids, dtype=np.uint64)
    if len(arr) > len(_POW):
        n = max(len(arr), 2 * len(_POW))
        with np.errstate(over="ignore"):
            _POW = np.cumprod(np.concatenate([[np.uint64(1)], np.full(n - 1, _BASE, dtype=np.uint64)]))
    with np.errstate(over="ignore"):
        return np.cumsum(arr * _POW[: len(arr)])


def seq_key(ids, n=None, h=None):
    """Key of the prefix of length n (default: the whole sequence)."""
    if n is None:
        n = len(ids)
    if n == 0:
        return (0, 0)
    if h is None:
        h = prefix_hashes(ids)
    return (n, int(h[n - 1]))


def build_chains(reqs, im_start_id):
    """Chain requests by exact token-prefix continuation (dispatch-time order).

    A request continues an open episode when that episode's last prompt+output is a prefix of
    the request's prompt ending just before an <|im_start|> (the rendered turn boundary). Keys
    are 64-bit polynomial prefix hashes; a hit is confirmed by exact comparison. Each request
    also gets r["key"] = key(prompt+output) and r["in_len"] for content matching later."""
    open_ends = collections.defaultdict(list)  # key -> [chain index]
    tails = {}  # chain index -> np array of its current prompt+output
    chains = []
    for r in reqs:
        ids = np.asarray(r["input_ids"], dtype=np.int64)
        h = prefix_hashes(ids)
        parent = None
        for p in np.flatnonzero(ids == im_start_id)[::-1]:
            for cut in (int(p), int(p) - 1, int(p) - 2):
                if cut <= 0:
                    continue
                for ci in open_ends.get(seq_key(ids, cut, h), []):
                    t = tails[ci]
                    if len(t) == cut and np.array_equal(t, ids[:cut]):
                        parent = ci
                        break
                if parent is not None:
                    break
            if parent is not None:
                break
        full = np.concatenate([ids, np.asarray(r["output_ids"], dtype=np.int64)])
        r["in_len"] = len(ids)
        r["key"] = seq_key(full)
        if parent is None:
            chains.append([r])
            parent = len(chains) - 1
        else:
            old = seq_key(tails[parent])
            open_ends[old].remove(parent)
            chains[parent].append(r)
        tails[parent] = full
        open_ends[r["key"]].append(parent)
    return chains


_W = {}


def _init_worker(tokenizer_path, system_prompt):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    _W.update(tok=tok, renderer=Renderer(tok, system_prompt), im_end=tok.convert_tokens_to_ids("<|im_end|>"))


def _initial_render(job):
    gamefile, max_steps = job
    g = Game(gamefile, max_steps)
    try:
        return _W["renderer"].initial(g.first)
    finally:
        g.close()


def _replay(job):
    """Replay one episode. job = (gamefile, max_steps, outputs, suffixes, ends_im_end): the model
    outputs per turn, the tokens each next request appended after the previous prompt+output,
    and whether that prompt+output ended in <|im_end|>."""
    gamefile, max_steps, outputs, suffixes, ends_im_end = job
    tok, renderer = _W["tok"], _W["renderer"]
    g = Game(gamefile, max_steps)
    issue, done, won = None, False, False
    try:
        for k, out in enumerate(outputs):
            if done:
                issue = "request after the episode ended"
                break
            obs, done, won = g.step(parse_action(tok.decode(out, skip_special_tokens=True)))
            done = done or won or (k + 1) >= max_steps
            if k + 1 < len(outputs):
                prev_tail = [_W["im_end"]] if ends_im_end[k] else [0]
                if suffixes[k] != renderer.continuation(prev_tail, obs):
                    issue = f"turn {k + 1}: next prompt is not previous tokens + spec observation"
                    break
        if issue is None and not done:
            issue = f"stopped after {len(outputs)} turns without a win"
    finally:
        g.close()
    return issue, bool(won)


def grpo_advantages(rewards):
    import torch

    r = torch.tensor(rewards, dtype=torch.float)
    a = r - r.mean()
    if len(rewards) > 1:
        std = r.std()
        if std > 0:
            a = a / (std + 1e-6)
    return a.tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit-dir", required=True)
    ap.add_argument("--dumps", required=True, help="glob of step_{k}_rank_{r}.pt")
    ap.add_argument("--contract", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--exclude-intervals", default=None)
    ap.add_argument("--no-dumps", action="store_true", help="authoring only: skip E6 and the trained-once part of E3")
    ap.add_argument("--episodes-out", default=None, help="write the replayed episodes (tokens, masks, logprobs, reward)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from transformers import AutoTokenizer

    c = json.load(open(a.contract))
    R, G, S = c["rounds"], c["games_per_round"], c["samples_per_game"]
    max_turn = c["max_turn_tokens"]
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    renderer = Renderer(tok, c["system_prompt"])
    im_start_id = tok.convert_tokens_to_ids(IM_START)
    issues = collections.defaultdict(list)

    rows = [json.loads(l) for l in open(c["train_data"])][: R * G]
    if len(rows) < R * G:
        raise AssertionError(f"training file has {len(rows)} rows, quota needs {R * G}")
    pool = get_context("fork").Pool(min(64, os.cpu_count() or 8), initializer=_init_worker,
                                    initargs=(a.tokenizer, c["system_prompt"]))
    initial = pool.map(_initial_render, [(os.path.join(c["alfworld_data"], r["game"]), c["max_steps"]) for r in rows])
    first_to_rows = collections.defaultdict(list)
    for idx, ids in enumerate(initial):
        first_to_rows[seq_key(ids)].append(idx)

    recs = load_audit(a.audit_dir)
    engine = lambda r: r.get("model_path") == c["policy_hf"]
    dispatches = [r for r in recs if r["kind"] == "dispatch" and engine(r) and r.get("rid") and not is_health_probe(r)]
    finals = {r["rid"]: r for r in recs if r["kind"] == "final" and engine(r) and r.get("rid")}
    t_first = min((r["t"] for r in dispatches), default=float("inf"))

    def any_between(times, lo, hi):
        return bisect.bisect_right(times, lo) < bisect.bisect_left(times, hi)

    # ---- E5 publications, per engine ----
    # The hook counts weight-update calls per SGLang tokenizer-manager process (one per
    # engine); versions and publications are therefore computed engine by engine. A
    # publication is a run of successful calls with no generation dispatch, pause/continue
    # or version-label change in between; calls before the engine's first training
    # generation are its initial push (version 0).
    engines = sorted({r["pid"] for r in dispatches})
    versions, pubs_by_engine = {}, {}
    for pid in engines:
        wc = sorted([r for r in recs if r["kind"] == "weights" and engine(r) and r.get("success") and r["pid"] == pid],
                    key=lambda r: r["t_start"])
        dt = sorted(r["t"] for r in dispatches if r["pid"] == pid)
        mk = sorted(r["t"] for r in recs if r["kind"] in ("pause", "continue") and engine(r) and r["pid"] == pid)
        groups, prev = [], None
        for idx, w in enumerate(wc):
            if prev is None or any_between(dt, prev["t_end"], w["t_start"]) or any_between(mk, prev["t_end"], w["t_start"]) \
                    or prev.get("server_weight_version") != w.get("server_weight_version"):
                groups.append([])
            groups[-1].append((idx + 1, w))
            prev = w
        first = dt[0] if dt else float("inf")
        pre = [g for g in groups if g[-1][1]["t_end"] <= first]
        pubs_p = [g for g in groups if g[-1][1]["t_end"] > first]
        n_pre_p = sum(len(g) for g in pre)
        bounds_p = [(g[0][0], g[-1][0]) for g in pubs_p]
        versions[pid] = (bounds_p, n_pre_p)
        pubs_by_engine[pid] = pubs_p
        if len(pubs_p) != R:
            issues["E5"].append(f"engine {pid}: {len(pubs_p)} publications after its first training generation; need exactly {R}")

    def version(pid, count):
        if count is None or pid not in versions:
            return None, True
        bounds_p, n_pre_p = versions[pid]
        done = sum(1 for lo, hi in bounds_p if count >= hi)
        inside = any(lo <= count < hi for lo, hi in bounds_p) or 0 < count < n_pre_p
        return done, inside

    pubs = max(pubs_by_engine.values(), key=len, default=[])
    n_pre = sum(v[1] for v in versions.values())
    last_pub_end = max((g[-1][1]["t_end"] for ps in pubs_by_engine.values() for g in ps), default=None)

    # ---- E2 sampling ----
    reqs = []
    for d in dispatches:
        s = d.get("sampling") or {}
        if d.get("top_logprobs_num") or d.get("token_ids_logprob"):
            issues["E2"].append(f"rid {d['rid']}: top-logprob / token-id-logprob request")
        if d.get("session"):
            issues["E2"].append(f"rid {d['rid']}: session request")
        top_k_off = s.get("top_k") in (-1, None, 0) or (s.get("top_k") or 0) >= (1 << 30)
        if s.get("temperature") != 1.0 or s.get("top_p") != 1.0 or not top_k_off or s.get("min_p") not in (0, 0.0, None):
            issues["E2"].append(f"rid {d['rid']}: sampling {s}")
        if not isinstance(s.get("max_new_tokens"), int) or not 0 < s["max_new_tokens"] <= max_turn:
            issues["E2"].append(f"rid {d['rid']}: max_new_tokens {s.get('max_new_tokens')}")
        f = finals.get(d["rid"])
        aborted = isinstance(f, dict) and isinstance(f.get("finish_reason"), dict) and f["finish_reason"].get("type") == "abort"
        if f is None or f.get("output_ids") is None or aborted or f.get("error") not in (None, "GeneratorExit"):
            issues["E1"].append(f"rid {d['rid']}: dispatched but no completed output")
            continue
        if f.get("output_logprobs") is None or len(f["output_logprobs"]) != len(f["output_ids"]):
            issues["E2"].append(f"rid {d['rid']}: engine returned no log-prob per output token")
        reqs.append({"rid": d["rid"], "t": d["t"], "input_ids": list(d["input_ids"]), "output_ids": list(f["output_ids"]),
                     "logprobs": f.get("output_logprobs") or [], "v0": version(d["pid"], d.get("weight_calls")),
                     "v1": version(f.get("pid"), f.get("weight_calls"))})
    reqs.sort(key=lambda r: r["t"])

    # ---- E1 episodes: chain + replay ----
    chains = build_chains(reqs, im_start_id)
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")

    # Different game files can share an initial observation (trials of one ALFWorld task
    # directory), so an episode's row is resolved by (a) the weight version its first turn
    # ran under (round r is generated under version r-1 or r), (b) replaying it against each
    # remaining candidate game, and (c) a capacity-S matching of episodes to rows.
    def replay_job(chain, row):
        outputs = [r["output_ids"] for r in chain]
        suffixes, ends = [], []
        for k in range(len(chain) - 1):
            n = chain[k]["in_len"] + len(chain[k]["output_ids"])
            suffixes.append(chain[k + 1]["input_ids"][n:])
            ends.append(bool(chain[k]["output_ids"]) and chain[k]["output_ids"][-1] == im_end_id)
        return (os.path.join(c["alfworld_data"], rows[row]["game"]), c["max_steps"], outputs, suffixes, ends)

    cands = {}
    for ci, chain in enumerate(chains):
        rs = [i for i in first_to_rows.get(seq_key(chain[0]["input_ids"]), []) if initial[i] == chain[0]["input_ids"]]
        if not rs:
            issues["E1"].append(f"episode {ci}: first request is not the initial rendering of any assigned game")
            continue
        v0 = chain[0]["v0"][0]
        in_window = [i for i in rs if v0 is None or (i // G) in (v0, v0 + 1)]
        cands[ci] = in_window or rs
    pairs = [(ci, row) for ci, rs in cands.items() for row in rs]
    results = dict(zip(pairs, pool.map(_replay, [replay_job(chains[ci], row) for ci, row in pairs], chunksize=4)))
    pool.close()

    slots = collections.defaultdict(list)  # row -> [chain]
    assign = {}

    def place(ci, allowed, seen):
        for row in allowed(ci):
            if row in seen:
                continue
            seen.add(row)
            if len(slots[row]) < S:
                slots[row].append(ci)
                assign[ci] = row
                return True
            for other in list(slots[row]):
                if place(other, allowed, seen | {row}):
                    slots[row].remove(other)
                    slots[row].append(ci)
                    assign[ci] = row
                    return True
        return False

    ok_rows = lambda ci: [r for r in cands[ci] if results[(ci, r)][0] is None]
    for ci in sorted(cands):
        place(ci, ok_rows, set())
    for ci in sorted(cands):
        if ci not in assign:
            place(ci, lambda ci: cands[ci], set())
    row_count = collections.Counter({row: len(v) for row, v in slots.items()})
    episodes = []
    for ci in sorted(cands):
        chain = chains[ci]
        if ci not in assign:
            issues["E3"].append(f"episode {ci}: every candidate row already has {S} episodes")
            continue
        row = assign[ci]
        issue, won = results[(ci, row)]
        if issue:
            issues["E1"].append(f"episode {ci} (row {row}): {issue}")
        last = chain[-1]
        full = last["input_ids"] + last["output_ids"]
        mask = [0] * len(full)
        lps = [0.0] * len(full)
        for r in chain:
            s0 = r["in_len"]
            for j, lp in enumerate(r["logprobs"][: len(r["output_ids"])]):
                mask[s0 + j] = 1
                lps[s0 + j] = float(lp) if lp is not None else float("nan")
        episodes.append({"row": row, "round": row // G, "won": won, "turns": len(chain), "tokens": full,
                         "mask": mask, "logprobs": lps, "prompt_len": chain[0]["in_len"], "ok": issue is None})

    # ---- E3 rows ----
    for idx in range(R * G):
        if row_count[idx] != S:
            issues["E3"].append(f"row {idx} (round {idx // G}): {row_count[idx]} episodes, need {S}")

    # advantages per row group
    by_row = collections.defaultdict(list)
    for e in episodes:
        by_row[e["row"]].append(e)
    for row, eps in by_row.items():
        for e, adv in zip(eps, grpo_advantages([1.0 if e["won"] else 0.0 for e in eps])):
            e["advantage"] = adv

    # ---- E3/E4/E6 against the dumps ----
    lags = []
    if not a.no_dumps:
        import torch

        by_tokens = collections.defaultdict(list)
        for i, e in enumerate(episodes):
            by_tokens[seq_key(e["tokens"])].append(i)
        # Requests keyed by (prompt length, prompt + output): twin episodes of a group can share
        # identical token prefixes, so turns are matched to requests by content, each request once.
        req_by_key = collections.defaultdict(list)
        for ri, r in enumerate(reqs):
            req_by_key[(r["in_len"], r["key"])].append(ri)
        used, used_req = set(), set()
        demands = []
        steps = collections.defaultdict(list)
        for path in sorted(glob.glob(a.dumps)):
            m = re.search(r"step_(\d+)_rank_(\d+)\.pt$", path)
            if m:
                steps[int(m.group(1))].append(torch.load(path, map_location="cpu", weights_only=False))
        if sorted(steps) != list(range(R)):
            issues["E3"].append(f"step dumps for {sorted(steps)}, expected 0..{R - 1}")
        for s in range(R):
            n = 0
            for rec in steps.get(s, []):
                for key in ("tokens", "response_lengths", "loss_masks", "engine_logprobs", "train_logprobs", "advantages"):
                    if key not in rec:
                        issues["E6"].append(f"step {s}: dump misses {key}")
                        break
                else:
                    for tokens, Rl, m_, elp, adv in zip(rec["tokens"], rec["response_lengths"], rec["loss_masks"],
                                                        rec["engine_logprobs"], rec["advantages"]):
                        n += 1
                        seq = [int(x) for x in (tokens.tolist() if hasattr(tokens, "tolist") else tokens)]
                        cands = [i for i in by_tokens.get(seq_key(seq), []) if i not in used]
                        if not cands:
                            issues["E3"].append(f"step {s}: a trained sequence matches no unused replayed episode")
                            continue
                        i = cands[0]
                        used.add(i)
                        e = episodes[i]
                        if e["round"] != s:
                            issues["E3"].append(f"step {s}: episode of row {e['row']} (round {e['round']}) trained here")
                        Rl = int(Rl)
                        off = len(seq) - Rl
                        if Rl < 0 or any(e["mask"][:off]):
                            issues["E6"].append(f"step {s} row {e['row']}: response region misses assistant tokens")
                            continue
                        mask = [int(x) for x in (m_.tolist() if hasattr(m_, "tolist") else m_)]
                        if mask != e["mask"][off:]:
                            issues["E6"].append(f"step {s} row {e['row']}: loss mask != assistant-output spans")
                            continue
                        elp = [float(x) for x in (elp.tolist() if hasattr(elp, "tolist") else elp)]
                        if len(elp) != Rl:
                            issues["E6"].append(f"step {s} row {e['row']}: {len(elp)} engine log-probs for {Rl} tokens")
                            continue
                        if abs(float(adv) - e["advantage"]) > 1e-4:
                            issues["E6"].append(f"step {s} row {e['row']}: advantage {float(adv):.5f} != GRPO {e['advantage']:.5f}")
                        hs = prefix_hashes(seq)
                        spans, j = [], 0
                        full_mask = e["mask"]
                        while j < len(seq):
                            if full_mask[j]:
                                k = j
                                while k < len(seq) and full_mask[k]:
                                    k += 1
                                spans.append((j, k))
                                j = k
                            else:
                                j += 1
                        for t, (lo, hi) in enumerate(spans):
                            demands.append({"step": s, "row": e["row"], "turn": t, "key": (lo, seq_key(seq, hi, hs)),
                                            "want": elp[lo - off:hi - off]})
            if n != G * S:
                issues["E3"].append(f"step {s}: {n} trained sequences, need {G * S}")
        # Each dumped turn must be one engine request with the same prompt+output and the same
        # log-probs; twins share keys, so solve the assignment per key (bipartite matching).
        def fits(d, ri):
            got = reqs[ri]["logprobs"][: len(d["want"])]
            return len(got) == len(d["want"]) and all(abs(x - float(y)) <= 1e-5 for x, y in zip(d["want"], got))

        # Twin rows (trials of one ALFWorld task directory) in different rounds can produce the same turn
        # (prompt + output) with the same log-probs (every token ~certain under both weight versions), so
        # a key can fit requests of both rounds; the matching first uses only requests whose weight
        # version is allowed for the demand's step, and only a demand left without one is matched to
        # any fitting request (and then reported under E4).
        def in_window(d, ri):
            (v0, _), (v1, _) = reqs[ri]["v0"], reqs[ri]["v1"]
            return (v0 is not None and v1 is not None and d["step"] - c["max_staleness"] <= v0 and v1 <= d["step"]
                    and fits(d, ri))

        by_key = collections.defaultdict(list)
        for di, d in enumerate(demands):
            by_key[d["key"]].append(di)
        for key, dis in by_key.items():
            rids = req_by_key.get(key, [])
            owner = {}

            def augment(di, seen, accept, movable):
                for ri in rids:
                    if ri in seen or not accept(demands[di], ri) or (ri in owner and owner[ri] not in movable):
                        continue
                    seen.add(ri)
                    if ri not in owner or augment(owner[ri], seen, accept, movable):
                        owner[ri] = di
                        return True
                return False

            every = set(dis)
            for di in dis:
                augment(di, set(), in_window, every)
            rest = [di for di in dis if di not in set(owner.values())]
            for di in rest:
                augment(di, set(), fits, set(rest))
            got = {di: ri for ri, di in owner.items()}
            for di in dis:
                d = demands[di]
                ri = got.get(di)
                if ri is None:
                    issues["E6"].append(f"step {d['step']} row {d['row']} turn {d['turn']}: engine log-probs match no engine request")
                    continue
                used_req.add(ri)
                s = d["step"]
                (v0, in0), (v1, in1) = reqs[ri]["v0"], reqs[ri]["v1"]
                if in0 or in1 or v0 != v1:
                    issues["E4"].append(f"step {s} row {d['row']} turn {d['turn']}: a publication happened inside the turn")
                lags.append(s - v0)
                if v0 < s - c["max_staleness"] or v1 > s:
                    issues["E4"].append(f"step {s} row {d['row']} turn {d['turn']}: version {v0}->{v1} outside [{s - c['max_staleness']}, {s}]")
        if len(used_req) != len(reqs):
            issues["E3"].append(f"{len(reqs) - len(used_req)} generation requests are not part of any trained sequence")
        untrained = len(episodes) - len(used)
        if untrained:
            issues["E3"].append(f"{untrained} played episodes were never trained on")

    t0 = min((r["t"] for r in reqs), default=None)
    t1 = last_pub_end
    excluded = 0.0
    if a.exclude_intervals and t0 and t1:
        end = t0
        for lo, hi in sorted(json.load(open(a.exclude_intervals))):
            lo, hi = max(lo, end, t0), min(hi, t1)
            if hi > lo:
                excluded += hi - lo
                end = hi
    wall = (t1 - t0 - excluded) if (t0 and t1) else None
    if wall is None or not wall > 0:
        issues["T"].append(f"no positive timed window: first request {t0}, end of the last publication {t1}, "
                           f"excluded {excluded:.3f} s")
    report = {
        "passed": not any(issues.values()),
        "issues": {k: v[:20] + ([f"... {len(v) - 20} more"] if len(v) > 20 else []) for k, v in issues.items()},
        "issue_counts": {k: len(v) for k, v in issues.items()},
        "requests": len(reqs),
        "episodes": len(episodes),
        "success_rate": sum(e["won"] for e in episodes) / max(1, len(episodes)),
        "success_by_round": [round(sum(e["won"] for e in episodes if e["round"] == r) / max(1, sum(e["round"] == r for e in episodes)), 4) for r in range(R)],
        "mean_turns": sum(e["turns"] for e in episodes) / max(1, len(episodes)),
        "initial_push_calls": n_pre,
        "publications": len(pubs),
        "engines": len(engines),
        "lag_histogram": {str(k): lags.count(k) for k in sorted(set(lags))},
        "wall_clock_seconds": wall,
        "wall_clock_raw_seconds": (t1 - t0) if (t0 and t1) else None,
        "excluded_seconds": excluded,
        "t_start": t0,
        "t_end": t1,
    }
    if a.episodes_out:
        with open(a.episodes_out, "w") as fh:
            json.dump([{k: e[k] for k in ("row", "round", "won", "turns", "tokens", "mask", "logprobs", "prompt_len")}
                       for e in episodes], fh)
    print(json.dumps(report, indent=2))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(report, fh, indent=2)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
