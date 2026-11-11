#!/usr/bin/env python3
"""Public checks for the Wan2.2 DanceGRPO task: the Judge's own code (run_judge.py --public), run as you, on public data.

  python /opt/wan_public/run_public.py --suite {cpu,score,probe,small,perf,full} [--repo /app/workdir/miles_lab/repo]

  cpu    J0: engine-overlay whitelist, AST policy, legacy sde/cps/ode bitwise (CPU, ~1 min)
  score  J2: `task_entry.py score` on the public score set vs the official HPSv3 oracle (~2 min)
  small  J3/J4: `task_entry.py run` (3 rounds, public prompts) under the in-engine audit (~15 min)
  probe  J1: `serve` + the public probe vs the fp32 oracle (~5 min)
  perf   probe + one timed replay of the public dump pub0 (8 prompts x 8 videos, 2 optimizer steps): T, R1-R5 (~5 min)
  full   all of the above

`serve` must take the replay size from its contract (`replay.prompts_per_dump`, `replay.optimizer_steps`):
/data/public/contract.json gives pub0's, and the Judge passes its own. The public T is not the Judge's T, so no score
is reported. R6' runs on the public replay; the R6 trajectory check does not (the public data has no trajectory
oracle). The public dump and probe use prompt rows 0-399; the Judge's use other prompts. Stop your own GPU processes
first: the checks start two engines and a `serve` process of their own. Results:
/scratch/public/<suite>-<time>/wan_judge.json and logs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quality_eval  # noqa: E402

SUITES = {"cpu": ("J0", 0), "score": ("J2", 0), "small": ("J0,RUN", 0), "probe": ("SERVE", 0),
          "perf": ("SERVE", 1), "full": ("J0,J2,RUN,SERVE", 1)}
# J6 (`full`) copies the run's ~20 GB final master here (run_judge.stage_master, in EVAL_DIR) for the QUALITY phase,
# which no suite runs: removed afterwards so it does not stay on this container's /dev/shm and memory limit
STAGED_MASTER = Path("/dev/shm/judge_eval/master_fp32.safetensors")
NOT_CHECKED = {"R6_traj_agg": "the public data has no R6 trajectory oracle"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=sorted(SUITES), required=True)
    ap.add_argument("--repo", type=Path, default=Path("/app/workdir/miles_lab/repo"))
    ap.add_argument("--data", type=Path, default=Path("/data/public"))
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    phases, replays = SUITES[a.suite]
    out = a.out or Path("/scratch/public") / f"{a.suite}-{time.strftime('%Y%m%dT%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    judge = Path(__file__).resolve().parent / "run_judge.py"
    cmd = [sys.executable, str(judge), "--public", "--submission", str(a.repo), "--private", str(a.data),
           "--out", str(out), "--phases", phases, "--max-replays", str(replays)]
    print("running:", " ".join(cmd), flush=True)
    try:
        rc = subprocess.call(cmd)
    finally:
        try:
            STAGED_MASTER.unlink(missing_ok=True)
        except OSError as e:
            print(f"could not remove {STAGED_MASTER}: {e}", flush=True)
    res = json.loads((out / "wan_judge.json").read_text()) if (out / "wan_judge.json").exists() else {}
    gates = {k: v for k, v in (res.get("gates") or {}).items() if not (replays and k in NOT_CHECKED)}
    summary = {"suite": a.suite, "rc": rc, "out": str(out), "error": res.get("error"), "timings": res.get("timings"),
               "gates": {k: _flat(v) for k, v in gates.items()},
               "checks_pass": bool(gates) and not res.get("error") and all(_ok(v) for v in gates.values())}
    if replays:
        summary["not_checked"] = NOT_CHECKED
        Ts = (res.get("timings") or {}).get("T") or []
        r = json.loads((a.data / "contract.json").read_text())["replay"]
        summary["performance"] = {"T_per_replay": Ts, "T_seconds": quality_eval.median_T(Ts),
                                  "replay": f"{r['prompts_per_dump']} prompts x {r['n']} videos, "
                                            f"{r['optimizer_steps']} optimizer steps"}
    print(json.dumps(summary, indent=1, default=str))


def _ok(v):  # run_judge.all_pass
    if isinstance(v, dict):
        return bool(v.get("ok", True)) and all(_ok(x) for x in v.values() if isinstance(x, dict))
    return True


def _flat(v):
    if isinstance(v, dict) and "ok" in v and not any(isinstance(x, dict) for x in v.values()):
        return v
    if isinstance(v, dict):
        return {k: (x.get("ok") if isinstance(x, dict) else x) for k, x in v.items()}
    return v


if __name__ == "__main__":
    main()
