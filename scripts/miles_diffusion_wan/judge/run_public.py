#!/usr/bin/env python3
"""Public checks for the Wan2.2 DanceGRPO task: the Judge's own code (run_judge.py --public), run as you, on public data.

  python /opt/wan_public/run_public.py --suite {cpu,score,probe,small,perf,full} [--repo /app/workdir/miles_lab/repo]

  cpu    J0: engine-overlay whitelist, AST policy, legacy sde/cps/ode bitwise (CPU, ~1 min)
  score  J2: `task_entry.py score` on the public score set vs the official HPSv3 oracle (~2 min)
  small  J3/J4: `task_entry.py run` (3 rounds, public prompts) under the in-engine audit (~15 min)
  probe  J1: `serve` + the public probe vs the fp32 oracle (~5 min)
  perf   probe + one timed replay of the public dump: T, R1-R6 on it, and a provisional score 3600 / T (~8 min)
  full   all of the above

Public thresholds are the Judge's, except R6: with one dump the medians of R6 are that one dump's values. The public dump
and probe use prompt rows 0-399; the Judge's use other rows. Stop your own GPU processes first: the checks start two
engines and a `serve` process of their own. Results: /scratch/public/<suite>-<time>/wan_judge.json and logs.

The provisional score uses the Judge's formula (metric v2: 3600 / median(T), 0 when a check fails) on the one public
replay. It assumes the Judge's held-out quality gate passes: no suite runs that eval (~9-10 min on both GPUs after a full
`run`; the Judge's QUALITY phase, quality_eval.py, on the final master of its own `run`).
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
    summary = {"suite": a.suite, "rc": rc, "out": str(out), "error": res.get("error"), "timings": res.get("timings"),
               "gates": {k: _flat(v) for k, v in (res.get("gates") or {}).items()}}
    if replays:
        summary["performance"] = performance(res)
    print(json.dumps(summary, indent=1, default=str))


def performance(res: dict) -> dict:
    """T of the public replay(s) and the provisional score with the Judge's formula (0 unless every check passed; the
    quality gate is not evaluated here)."""
    Ts = (res.get("timings") or {}).get("T") or []
    T = quality_eval.median_T(Ts)
    correct = bool(res.get("gates")) and not res.get("error") and all(_ok(v) for v in res["gates"].values())
    metric = res.get("metrics") or {}
    return {"T_per_replay": Ts, "T_seconds": T,
            "provisional_score": (quality_eval.SCORE_SCALE_SECONDS / T if correct else 0.0) if T else None,
            "provisional_note": "3600 / T of the public replay; provisional: public dump, and the Judge also requires "
                                f"its held-out quality gate (quality_score >= {metric.get('quality_gate_min')} "
                                f"{metric.get('quality_unit')} on the final master of its own run), which no public "
                                "suite evaluates"}


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
