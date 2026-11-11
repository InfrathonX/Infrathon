"""Judge-side ALFWorld episode reconstruction and replay (W1/W2 of the ALFWorld task).

Input: the audit hook's jsonl files (``dispatch`` + ``final`` records of the
rollout engine). The checker never reads candidate-side logs:

1. join dispatch/final by rid and keep sampled (T=1) requests;
2. chain requests into episodes: request B continues request A when
   ``A.input_ids + A.output_ids`` is a prefix of ``B.input_ids``;
3. the first request of an episode must be the spec rendering of the initial
   observation of one of the games assigned to that round;
4. every assistant output is decoded, parsed with the spec rule and replayed in
   the official ALFWorld engine; the next request must extend the previous one
   with exactly the spec rendering of the replayed observation, and the episode
   must stop exactly when the game is won or the step cap is hit.

Returns per-episode records (game, actions, won, steps) and a list of failures.
"""

import argparse
import collections
import glob
import hashlib
import json
import re
import sys

MAX_STEPS = 30
IM_START = "<|im_start|>"
ACTION_RE = re.compile(r"<action>(.*?)</action>", re.S)

SYSTEM_PROMPT = (
    "You are an expert agent operating in the ALFRED embodied household environment. "
    "Each turn you receive an observation and the list of admissible actions. "
    "Reply with at most two short sentences of reasoning, then the chosen action inside <action></action>, "
    "for example <action>go to shelf 1</action>. The action must be exactly one of the admissible actions."
)


def format_observation(obs, admissible):
    admissible = [a for a in admissible if a not in ("look", "inventory")] + ["look", "inventory"]
    return obs.strip() + "\n\nAdmissible actions: " + ", ".join(admissible)


def parse_action(text):
    m = ACTION_RE.search(text)
    return m.group(1).strip() if m else ""


class Renderer:
    """Spec token rendering (Qwen3 chat template, enable_thinking=False)."""

    def __init__(self, tokenizer, system_prompt=SYSTEM_PROMPT):
        self.tok = tokenizer
        self.system_prompt = system_prompt

    def initial(self, first_user):
        msgs = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": first_user}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return self.tok.encode(text, add_special_tokens=False)

    def continuation(self, prev_full, user):
        """Tokens appended after ``prev_full`` (= previous input + output) for a new user turn.

        The previous output ends with <|im_end|> when it stopped naturally; the
        template then continues with "\\n<|im_start|>user\\n...". A length-truncated
        output has no <|im_end|>, and the template closes the turn first.
        """
        im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        head = "\n" if prev_full and prev_full[-1] == im_end else "<|im_end|>\n"
        text = f"{head}{IM_START}user\n{user}<|im_end|>\n{IM_START}assistant\n<think>\n\n</think>\n\n"
        return self.tok.encode(text, add_special_tokens=False)


class Game:
    def __init__(self, gamefile, max_steps=MAX_STEPS):
        import textworld
        import textworld.gym
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

        infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
        env_id = textworld.gym.register_games(
            [gamefile], infos, batch_size=1, asynchronous=False, max_episode_steps=max_steps,
            wrappers=[AlfredDemangler(shuffle=False), AlfredInfos],
        )
        self.env = textworld.gym.make(env_id)
        obs, info = self.env.reset()
        self.first = format_observation(obs[0], info["admissible_commands"][0])

    def step(self, action):
        obs, _, done, info = self.env.step([action])
        return format_observation(obs[0], info["admissible_commands"][0]), bool(done[0]), bool(info["won"][0])

    def close(self):
        self.env.close()


def load_requests(audit_glob, temperature=1.0):
    dispatch, final = {}, {}
    for path in sorted(glob.glob(audit_glob)):
        with open(path) as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("kind") == "dispatch" and r.get("rid"):
                    dispatch[r["rid"]] = r
                elif r.get("kind") == "final" and r.get("rid"):
                    final[r["rid"]] = r
    reqs = []
    for rid, d in dispatch.items():
        f = final.get(rid)
        if f is None or f.get("error") or not d.get("input_ids"):
            continue
        sampling = d.get("sampling") or {}
        if sampling.get("temperature") != temperature:
            continue
        if sampling.get("max_new_tokens") == 1 and len(d["input_ids"]) == 1:
            continue  # Miles engine health probe (input [0], one new token)
        reqs.append({"rid": rid, "t": d["t"], "input_ids": d["input_ids"], "output_ids": f["output_ids"] or [],
                     "output_logprobs": f.get("output_logprobs"), "finish_reason": f.get("finish_reason"),
                     "weight_calls": d.get("weight_calls"), "weight_calls_final": f.get("weight_calls"),
                     "sampling": d.get("sampling")})
    reqs.sort(key=lambda r: r["t"])
    return reqs


def _h(ids):
    return hashlib.sha1(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def build_chains(reqs, im_start_id):
    """Chain requests by exact token-prefix continuation (dispatch-time order)."""
    open_ends = collections.defaultdict(list)  # hash(input+output) -> [chain idx]
    chains = []
    for r in reqs:
        ids = r["input_ids"]
        parent = None
        for p in range(len(ids) - 1, 0, -1):
            if ids[p] != im_start_id:
                continue
            for cut in (p, p - 1, p - 2):
                lst = open_ends.get(_h(ids[:cut]))
                if lst:
                    parent = lst.pop()
                    break
            if parent is not None:
                break
        if parent is None:
            chains.append([r])
            parent = len(chains) - 1
        else:
            chains[parent].append(r)
        open_ends[_h(r["input_ids"] + r["output_ids"])].append(parent)
    return chains


def check_chain(chain, games_by_first, renderer, tok):
    """Replay one chain; return (episode record, failures)."""
    fails = []
    root = chain[0]["input_ids"]
    game = games_by_first.get(_h(root))
    if game is None:
        return None, ["first request does not render the initial observation of any assigned game"]
    g = Game(game)
    actions, won, done = [], False, False
    try:
        if renderer.initial(g.first) != root:
            fails.append("initial render mismatch")
        for k, r in enumerate(chain):
            if done:
                fails.append(f"request after the episode ended (turn {k})")
                break
            text = tok.decode(r["output_ids"], skip_special_tokens=True)
            action = parse_action(text)
            actions.append(action)
            obs, done, won = g.step(action)
            done = done or won or len(actions) >= MAX_STEPS
            if k + 1 < len(chain):
                full = r["input_ids"] + r["output_ids"]
                expected = full + renderer.continuation(full, obs)
                if chain[k + 1]["input_ids"] != expected:
                    fails.append(f"turn {k + 1}: next prompt is not previous tokens + spec observation")
                    break
        if not done:
            fails.append(f"episode stopped early after {len(actions)} steps without win")
    finally:
        g.close()
    return {"game": game, "actions": actions, "steps": len(actions), "won": won,
            "rids": [r["rid"] for r in chain]}, fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-glob", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--rows", required=True, help="train jsonl (fixed order)")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--num-rows", type=int, required=True, help="rows used by the run (rounds x games/round)")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    renderer = Renderer(tok)
    rows = [json.loads(l) for l in open(a.rows)][: a.num_rows]
    games_by_first, row_of = {}, {}
    for row in rows:
        gf = f"{a.data_root}/{row['metadata']['game']}"
        g = Game(gf)
        key = _h(renderer.initial(g.first))
        g.close()
        if key in games_by_first:
            print(f"warning: two assigned games share an initial prompt: {gf}", file=sys.stderr)
        games_by_first[key] = gf
        row_of[gf] = row["metadata"]["row"]

    reqs = load_requests(a.audit_glob)
    chains = build_chains(reqs, tok.convert_tokens_to_ids(IM_START))
    episodes, failures = [], []
    for i, chain in enumerate(chains):
        ep, fails = check_chain(chain, games_by_first, renderer, tok)
        if ep is not None:
            ep["row"] = row_of[ep["game"]]
            episodes.append(ep)
        failures += [f"chain {i}: {f}" for f in fails]
    per_row = collections.Counter(ep["row"] for ep in episodes)
    for row in rows:
        n = per_row.get(row["metadata"]["row"], 0)
        if n != a.samples:
            failures.append(f"row {row['metadata']['row']}: {n} episodes (expected {a.samples})")
    summary = {"requests": len(reqs), "chains": len(chains), "episodes": len(episodes),
               "success": sum(e["won"] for e in episodes) / max(1, len(episodes)),
               "mean_steps": sum(e["steps"] for e in episodes) / max(1, len(episodes)),
               "failures": len(failures)}
    json.dump({"summary": summary, "failures": failures, "episodes": episodes}, open(a.out, "w"))
    print(json.dumps(summary, indent=1))
    for f in failures[:20]:
        print("FAIL", f)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
