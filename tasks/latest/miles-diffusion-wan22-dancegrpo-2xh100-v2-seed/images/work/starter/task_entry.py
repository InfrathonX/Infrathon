#!/usr/bin/env python3
"""Task entry points (docs/INTERFACE.md): `score`, `run`, `serve`. The Judge runs them as `agent` with cwd = repo/.

Starter: the three commands parse their arguments and the contract, then stop. miles_diffusion cannot run the recipe yet
(no Dance-SDE transition, no seed-group RNG in the engine, no HPSv3 reward): see AGENT_TASK.md. Useful pieces of this
checkout: train_diffusion.py / miles.utils.arguments (the MD launcher and options), miles/backends/fsdp_utils (FSDP2
trainer, sde_step_backend.py, loss_hub/flow_grpo.py), miles/rollout (rollout functions, rm_hub), and
miles/backends/sglang_utils (engines, weight publication).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent


def cmd_score(a) -> None:
    raise NotImplementedError("HPSv3 video reward (official semantics, AGENT_TASK.md §3) -> O/scores.json")


def cmd_run(a) -> None:
    c = json.load(open(a.contract))
    _ = (c["sampling"], c["optim"], c["run"])
    raise NotImplementedError("audited DanceGRPO run: 3 rounds x 8 prompts x 8 samples (docs/INTERFACE.md)")


def cmd_serve(a) -> None:
    c = json.load(open(a.contract))
    _ = c["replay"]["engine_urls"]
    raise NotImplementedError("fixed-replay server: ready / probe / replay / dump_master / reset / stop (docs/INTERFACE.md)")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--videos", required=True)
    s.add_argument("--prompts", required=True)
    s.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--contract", required=True)
    r.add_argument("--out", required=True)
    v = sub.add_parser("serve")
    v.add_argument("--contract", required=True)
    v.add_argument("--ctl", required=True)
    v.add_argument("--events", required=True)
    v.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(REPO))
    {"score": cmd_score, "run": cmd_run, "serve": cmd_serve}[a.cmd](a)


if __name__ == "__main__":
    main()
