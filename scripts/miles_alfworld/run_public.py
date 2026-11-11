#!/usr/bin/env python3
"""Public checks for the Miles ALFWorld task (run inside Work, as the agent user).

  python /opt/miles_alfworld_public/run_public.py --suite cpu   --output /scratch/public/cpu-001
  python /opt/miles_alfworld_public/run_public.py --suite probe --output /scratch/public/probe-001
  python /opt/miles_alfworld_public/run_public.py --suite small --output /scratch/public/small-001 [--probe-output ...]
  python /opt/miles_alfworld_public/run_public.py --suite perf  --output /scratch/public/perf-001  [--probe-output ...]

Every GPU suite runs `task_entry.py` with the evaluator's instrumentation switched on (the
ALFWORLD_AUDIT_* environment variables must reach every process your command starts).

cpu    the replay checker on a recorded public reference run: rendering, action parsing
       and the observation format (no GPU, about a minute)
probe  one GRPO step on public fixed episodes: the dump against the batch, advantages,
       the logged loss, the fp32 oracle (trainer log-probs, loss) and provenance (one Megatron-Core step inside Miles, contract AdamW)
small  3 rounds x 4 public games x 8 samples through `task_entry.py run`: provenance, the
       audit (episodes replayed in ALFWorld, sampling, every row played 8 times and every
       episode trained once at its round, staleness <= 1 with no publication inside a
       turn, one publication per step, dumps == engine records), every step's loss, the
       fp32 oracle at step 0 and at one served-weight step, and with --probe-output the
       same Megatron numerics as the probe
perf   the public contract (4 rounds x 8 games x 8 samples): the same checks, the audited timed window T,
       and greedy success of the final policy on the 8 public evaluation games in 3 repeats.
These suites only get a pipeline working: their sizes are not the scored run's, and neither T
nor the success predicts a score, so no score is reported here.
Exit code 0 means every check passed. The Judge runs the same checks on hidden rows and
games and draws its own random step; passing here does not guarantee a Judge score.

When a GPU suite ends (pass, fail or error) it deletes the bulky state under --output: the
contract scratch (scratch/), final_hf/, Megatron checkpoints (checkpoints/, ckpt/) and the
served-weight snapshots (audit/snapshots/). Reports, logs,
step dumps and provenance records stay. Pass --keep-artifacts to keep everything.
"""

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import alfworld_checks as oc  # noqa: E402
import alfworld_pipeline as pl  # noqa: E402

BULKY_NAMES = ("final_hf", "checkpoints", "ckpt")  # anywhere under --output


def clean_artifacts(out):
    """Delete the bulky directories a GPU suite leaves under `out`; return what was removed or left."""
    targets = [out / "scratch", out / "audit" / "snapshots"]
    for root, dirs, _ in os.walk(out):
        for name in list(dirs):
            path = Path(root) / name
            if path in targets or name in BULKY_NAMES:
                dirs.remove(name)  # do not descend into what is about to be deleted
                if name in BULKY_NAMES:
                    targets.append(path)
    removed, left = [], []
    for path in targets:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
            (left if path.exists() else removed).append(str(path.relative_to(out)))
    return {"removed": removed, "could_not_remove": left}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=["cpu", "probe", "small", "perf"], required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--app", type=Path, default=Path("/app/workdir/miles_lab"))
    ap.add_argument("--contract", type=Path, default=HERE / "contract.json")
    ap.add_argument("--probe-output", type=Path, default=None)
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--keep-artifacts", action="store_true",
                    help="keep scratch/, final_hf/, checkpoints and audit snapshots under --output")
    a = ap.parse_args()
    out = a.output
    out.mkdir(parents=True, exist_ok=False)
    report = {"suite": a.suite, "checks": {}}
    try:
        run_suite(a, out, report)
    finally:
        if a.suite != "cpu":
            report["artifacts"] = {"kept": True} if a.keep_artifacts else clean_artifacts(out)
    report["passed"] = bool(report["checks"]) and all(v.get("passed") for v in report["checks"].values())
    oc.write_json(out / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "checks"} | {
        "checks": {k: {kk: vv for kk, vv in v.items() if kk in ("passed", "issues", "reasons", "error", "code", "seconds")}
                   for k, v in report["checks"].items()}}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


def run_suite(a, out, report):
    base = oc.load_contract(a.contract, scratch=str(out / "scratch"))

    if a.suite == "cpu":
        code, res = oc.run_check([oc.PY, oc.tool("audit/alfworld_audit_check.py"), "--audit-dir", "/data/public/sample_run/audit",
                                  "--dumps", "/data/public/sample_run/steps/step_*_rank_*.pt",
                                  "--contract", "/data/public/sample_run/contract.json", "--tokenizer", oc.POLICY],
                                 out / "cpu.log")
        report["checks"]["replay_sample_run"] = res or {"passed": False, "error": "see cpu.log"}
    else:
        contract = base if a.suite in ("probe", "perf") else dict(base, rounds=3, games_per_round=4)
        audit = out / "audit"
        audit.mkdir()
        oc.write_json(out / "contract.json", contract)
        env = dict(os.environ, ALFWORLD_AUDIT_DIR=str(audit), ALFWORLD_AUDIT_ROUNDS=str(1 if a.suite == "probe" else contract["rounds"]))

        def entry(cmd):
            code, secs = oc.run_logged([oc.PY, "task_entry.py", *cmd, "--contract", str(out / "contract.json")],
                                       out / "task_entry.log", cwd=a.app, env=env, timeout=a.timeout)
            report["checks"]["task_entry_exit"] = {"passed": code == 0, "code": code, "seconds": round(secs, 1)}
            return code == 0

        if a.suite == "probe":
            batch = "/data/public/probe_batch.json"
            if entry(["probe", "--batch", batch, "--out", str(out / "probe")]):
                checks, _ = pl.check_probe(out / "probe", batch, out / "contract.json", audit, out)
                report["checks"].update(checks)
        elif entry(["run", "--out", str(out / "run")]):
            probe_prov = None
            if a.probe_output and (a.probe_output / "provenance_probe.json").is_file():
                probe_prov = json.loads((a.probe_output / "provenance_probe.json").read_text())
            checks, aud = pl.check_run(out / "run", out / "contract.json", audit, out, random.SystemRandom(),
                                       probe_prov=probe_prov, keep_snapshots=a.keep_artifacts)
            report["checks"].update(checks)
            if a.suite == "perf":
                code, ev = oc.run_check([oc.PY, oc.tool("eval_alfworld.py"), "--model", str(out / "run/final_hf"),
                                         "--eval", "/data/public/eval_public.jsonl", "--games-root", "/data/public/games",
                                         "--contract", str(out / "contract.json"), "--repeats", "3"],
                                        out / "eval.log")
                report["performance"] = performance(aud, ev)


def performance(aud, ev):
    """The perf report: T, the public success and the training records."""
    aud, ev = aud or {}, ev or {}
    return {
        "wall_clock_seconds": aud.get("wall_clock_seconds"),
        "excluded_seconds": aud.get("excluded_seconds"),
        "public_eval_success": ev.get("success"),
        "public_eval_per_repeat": ev.get("per_repeat"),
        "train_success_by_round": aud.get("success_by_round"),
        "mean_turns": aud.get("mean_turns"),
        "note": "for getting the pipeline working: this run is much smaller than the scored one, so neither T nor the "
                "success predicts a score; only sforge-submit measures it",
    }


if __name__ == "__main__":
    main()
