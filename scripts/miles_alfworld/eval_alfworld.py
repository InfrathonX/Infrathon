#!/usr/bin/env python3
"""Judge-owned greedy ALFWorld evaluation of an HF checkpoint.

Starts one SGLang server per visible GPU (data parallel), plays every game of the
evaluation file with the task's rendering and action rule at temperature 0, and repeats
the whole set ``--repeats`` times. Prints {"success": mean over repeats, "per_repeat": [...]}.
"""

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from multiprocessing import Pool

import requests

ACTION_RE = re.compile(r"<action>(.*?)</action>", re.S)


def format_observation(obs, admissible):
    admissible = [x for x in admissible if x not in ("look", "inventory")] + ["look", "inventory"]
    return obs.strip() + "\n\nAdmissible actions: " + ", ".join(admissible)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def episode(job):
    gamefile, url, system_prompt, max_steps, max_tokens, template_kwargs = job
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

    infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
    env = textworld.gym.make(textworld.gym.register_games(
        [gamefile], infos, batch_size=1, asynchronous=False, max_episode_steps=max_steps,
        wrappers=[AlfredDemangler(shuffle=False), AlfredInfos]))
    obs, info = env.reset()
    msgs = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": format_observation(obs[0], info["admissible_commands"][0])}]
    won, steps = False, 0
    try:
        for steps in range(1, max_steps + 1):
            for attempt in range(5):
                try:
                    r = requests.post(url, json={"model": "m", "messages": msgs, "temperature": 0.0, "top_p": 1.0,
                                                 "max_tokens": max_tokens, "chat_template_kwargs": template_kwargs},
                                      timeout=600)
                    r.raise_for_status()
                    text = r.json()["choices"][0]["message"]["content"] or ""
                    break
                except requests.RequestException:
                    if attempt == 4:
                        raise
                    time.sleep(2)
            msgs.append({"role": "assistant", "content": text})
            m = ACTION_RE.search(text)
            obs, _, done, info = env.step([m.group(1).strip() if m else ""])
            won = bool(info["won"][0])
            if done[0] or won:
                break
            msgs.append({"role": "user", "content": format_observation(obs[0], info["admissible_commands"][0])})
    finally:
        env.close()
    return {"game": gamefile, "won": won, "steps": steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval", required=True, help="jsonl rows {row_id, game}")
    ap.add_argument("--games-root", required=True)
    ap.add_argument("--contract", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--procs", type=int, default=96)
    ap.add_argument("--limit", type=int, default=None, help="authoring only")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    c = json.load(open(a.contract))
    rows = [json.loads(l) for l in open(a.eval)][: a.limit]
    gpus = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if g != ""]
    servers, urls = [], []
    try:
        for g in gpus:
            port = free_port()
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=g)
            log = open(f"/tmp/eval_sglang_{g}.log", "w")
            servers.append(subprocess.Popen([sys.executable, "-m", "sglang.launch_server", "--model-path", a.model,
                                             "--port", str(port), "--host", "127.0.0.1", "--mem-fraction-static", "0.85",
                                             "--context-length", "40960", "--served-model-name", "m",
                                             "--random-seed", "0"],
                                            env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
            urls.append(f"http://127.0.0.1:{port}")
        deadline = time.time() + 900
        for u in urls:
            while True:
                try:
                    if requests.get(u + "/health", timeout=5).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                if time.time() > deadline:
                    raise RuntimeError("SGLang did not become healthy")
                time.sleep(5)
        per_repeat = []
        for rep in range(a.repeats):
            jobs = [(os.path.join(a.games_root, r["game"]), urls[i % len(urls)] + "/v1/chat/completions",
                     c["system_prompt"], c["max_steps"], c["max_turn_tokens"], c["template_kwargs"])
                    for i, r in enumerate(rows)]
            with Pool(a.procs) as pool:
                res = pool.map(episode, jobs, chunksize=1)
            per_repeat.append(sum(r["won"] for r in res) / len(res))
            print(f"repeat {rep}: {per_repeat[-1]:.4f}", file=sys.stderr, flush=True)
    finally:
        for p in servers:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    out = {"success": sum(per_repeat) / len(per_repeat), "per_repeat": per_repeat, "games": len(rows)}
    print(json.dumps(out))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
