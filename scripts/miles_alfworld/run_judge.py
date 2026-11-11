#!/usr/bin/env python3
"""Judge for the Miles ALFWorld task.

Runs as root in the Judge container; candidate code runs as the unprivileged `agent` user
and is never imported here. Both candidate commands run under the SGLang audit hook and
the Megatron/SGLang provenance hook.

  1 probe  one GRPO step on hidden recorded episodes: the dump against the batch, the fp32
           oracle (trainer log-probs, loss), provenance
  2 run    the full contract on hidden training rows (their game files are copied in for
           the run only): provenance, audit E1-E6 (episodes replayed in the Judge's
           ALFWorld, sampling, quota, staleness, publication, dumps == engine records),
           per-step loss, the probe's Megatron numerics, fp32 oracle at step 0 and at one
           random served-weight step (loss, trainer log-probs); the audited timed window must
           lie inside the run as the Judge's own clock saw it, with at most
           EXCLUDED_MAX_SECONDS of evaluator intervals taken out
  3 eval   greedy success of the final policy on 1134 hidden games, 3 repeats (Judge's own
           SGLang and agent loop)
  score = quality_score * 3600 / T   if every check passes and quality_score >= the quality gate, else 0
          quality_score = mean of the 3 repeats - the original model's frozen 3-repeat mean, in success
          points; T = the audited timed window (alfworld_checks.metric_v2, qualification.json "metric")
A candidate failure (a check or the quality gate) is valid=true, score=0; valid=false only for
Judge/infrastructure errors.
"""

import argparse
import json
import os
import pwd
import random
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "public"))
import alfworld_checks as oc  # noqa: E402
import alfworld_pipeline as pl  # noqa: E402

QUAL = json.loads((HERE / "qualification.json").read_text())
METRIC = QUAL["metric"]
PRIVATE = HERE / "private"
WORK = Path("/scratch/judge")
AGENT = pwd.getpwnam("agent")
RNG = random.SystemRandom()
DETAIL_KEYS = ("passed", "issues", "issue_counts", "reasons", "error", "code", "seconds")
# the evaluator's intervals taken out of T (two snapshot copies per engine and the final digests, each at most
# alfworld_provenance_check.EXCLUDE_MAX_SECONDS) add up to at most this; the reference's two copies took ~18 s
EXCLUDED_MAX_SECONDS = 300.0


def as_agent():
    os.setgid(AGENT.pw_gid)
    os.setgroups([])
    os.setuid(AGENT.pw_uid)


def agent_env(extra=None):
    env = {"PATH": "/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
           "HOME": "/home/agent", "USER": "agent", "LANG": "C.UTF-8",
           "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"),
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
           "WANDB_MODE": "disabled", "RAY_USAGE_STATS_ENABLED": "0", "PYTHONUNBUFFERED": "1",
           "PYTHONDONTWRITEBYTECODE": "1"}
    env.update(extra or {})
    return env


def reap_agent():
    subprocess.run(["pkill", "-KILL", "-u", "agent"], check=False)
    subprocess.run(["bash", "-c", "sleep 3"], check=False)


def private_run_data(d, rounds, games):
    """Copy the hidden training rows used by this run, and only their game files, into `d`."""
    rows = [json.loads(l) for l in open(PRIVATE / "train_private.jsonl")][: rounds * games]
    games_dir = d / "games"
    for r in rows:
        dst = games_dir / r["game"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PRIVATE / "games" / r["game"], dst)
    data = d / "train.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(data), str(games_dir)


def candidate(app, name, args, timeout, rounds, overrides):
    """Run task_entry.py as agent on an agent-owned copy of the submission.

    The contract and the hidden data it names (`overrides`) are written twice: the candidate's
    copy under WORK/<name> (chowned to agent with the rest of that directory), and a root-owned,
    root-only copy under WORK/<name>-judge that the checkers read (its contract names the Judge's
    own copies of the rows and games). Returns the Judge's contract path."""
    d = WORK / name
    judge = WORK / f"{name}-judge"
    shutil.copytree(app, d / "miles_lab", symlinks=True)
    for sub in ("out", "scratch", "audit"):
        (d / sub).mkdir()
    judge.mkdir(mode=0o700)
    contract = oc.load_contract(HERE / "contract.json", scratch=str(d / "scratch"))
    judge_contract = dict(contract)
    if overrides:
        contract.update(overrides(d, contract))
        judge_contract.update(overrides(judge, judge_contract))
    contract_path = d / "contract.json"
    oc.write_json(contract_path, contract)
    oc.write_json(judge / "contract.json", judge_contract)
    subprocess.run(["chown", "-R", "agent:agent", str(d)], check=True)
    env = agent_env({"ALFWORLD_AUDIT_DIR": str(d / "audit"), "ALFWORLD_AUDIT_ROUNDS": str(rounds)})
    try:
        code, secs = oc.run_logged([oc.PY, "task_entry.py", *args(d), "--contract", str(contract_path)],
                                   d / "task_entry.log", cwd=d / "miles_lab", env=env, timeout=timeout, preexec=as_agent)
    finally:
        reap_agent()
    return d, judge / "contract.json", code, secs


def judge_clock_issues(aud, started, ended):
    """Why the audited timed window is not inside the candidate run [started, ended] as the Judge's clock
    saw it, or takes out more than EXCLUDED_MAX_SECONDS of evaluator intervals; [] when it is fine (the
    audit and provenance records are written in a directory the candidate can write)."""
    t0, t1, ex = aud.get("t_start"), aud.get("t_end"), aud.get("excluded_seconds")
    issues = []
    if not (all(isinstance(v, (int, float)) for v in (t0, t1)) and started <= t0 <= t1 <= ended):
        issues.append(f"timed window [{t0}, {t1}] is not inside the run [{started:.3f}, {ended:.3f}] (Judge clock)")
    if not (isinstance(ex, (int, float)) and 0 <= ex <= EXCLUDED_MAX_SECONDS):
        issues.append(f"{ex} s of evaluator intervals taken out of T; at most {EXCLUDED_MAX_SECONDS:g} s")
    return issues


def compact(check):
    return {k: v for k, v in check.items() if k in DETAIL_KEYS or not isinstance(v, (dict, list))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", type=Path, default=Path("/app/workdir/miles_lab"))
    ap.add_argument("--log-dir", type=Path, default=Path("/logs/verifier"))
    a = ap.parse_args()
    a.log_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True)
    os.chmod(WORK, 0o755)
    checks, metrics, private = {}, {"score_formula": METRIC["formula"]}, {}
    result = {"valid": True, "score": 0.0}
    summary = None
    qual = {"oracle_limits": QUAL.get("oracle_limits"), "grad_limits": QUAL.get("grad_limits")}
    try:
        if not (a.app / "task_entry.py").is_file() or not (a.app / "repo/train.py").is_file():
            checks["submission"] = {"passed": False, "error": "task_entry.py or repo/ missing"}
            raise StopIteration

        # ---- 1 probe on hidden recorded episodes ----
        batch = WORK / "probe_batch.json"
        shutil.copy(PRIVATE / "probe_batch.json", batch)
        os.chmod(batch, 0o644)
        d, cpath, code, secs = candidate(a.app, "probe", lambda d: ["probe", "--batch", str(batch), "--out", str(d / "out/probe")],
                                         QUAL["timeouts"]["probe"], 1, None)
        checks["probe_exit"] = {"passed": code == 0, "code": code, "seconds": round(secs, 1)}
        if code != 0:
            raise StopIteration
        pc, probe_prov = pl.check_probe(d / "out/probe", batch, cpath, d / "audit", WORK, qual)
        private["probe"] = pc
        checks.update({k: compact(v) for k, v in pc.items()})
        if not all(v.get("passed") for v in pc.values()):
            raise StopIteration

        # ---- 2 full run on hidden rows ----
        def hidden(d, c):
            data, games = private_run_data(d, c["rounds"], c["games_per_round"])
            return {"train_data": data, "alfworld_data": games}

        c0 = json.loads((HERE / "contract.json").read_text())
        started = time.time()
        d, cpath, code, secs = candidate(a.app, "run", lambda d: ["run", "--out", str(d / "out/run")],
                                         QUAL["timeouts"]["run"], c0["rounds"], hidden)
        ended = time.time()
        checks["run_exit"] = {"passed": code == 0, "code": code, "seconds": round(secs, 1)}
        if code != 0:
            raise StopIteration
        rc, aud = pl.check_run(d / "out/run", cpath, d / "audit", WORK, RNG, qual, probe_prov=probe_prov, n_served=1)
        why = judge_clock_issues(aud, started, ended) if aud.get("passed") else []
        if why:
            aud["passed"] = False
            aud.setdefault("issues", {}).setdefault("T", []).extend(why)
            aud.setdefault("issue_counts", {})["T"] = len(aud["issues"]["T"])
        private["run"] = rc
        checks.update({k: compact(v) for k, v in rc.items()})
        if not all(v.get("passed") for v in rc.values()):
            raise StopIteration

        # ---- 3 evaluation of the final policy ----
        final = d / "out/run/final_hf"
        model_dir = WORK / "scored_model"
        model_dir.mkdir()
        shards = sorted(final.glob("*.safetensors"))
        if not shards:
            checks["final_weights"] = {"passed": False, "error": "no safetensors in final_hf"}
            raise StopIteration
        for f in Path(oc.POLICY).iterdir():
            if not f.name.endswith(".safetensors") and f.name != "model.safetensors.index.json":
                shutil.copy(f.resolve(), model_dir / f.name)
        for f in shards:
            shutil.copy(f, model_dir / f.name)
        idx = final / "model.safetensors.index.json"
        if idx.is_file():
            shutil.copy(idx, model_dir / idx.name)
        _, ev = oc.run_check([oc.PY, oc.tool("eval_alfworld.py"), "--model", str(model_dir),
                              "--eval", str(PRIVATE / "eval_hidden.jsonl"), "--games-root", str(PRIVATE / "games"),
                              "--contract", str(HERE / "contract.json"), "--repeats", str(METRIC["quality_repeats"])],
                             WORK / "eval.log")
        if not ev or "success" not in ev or not ev.get("per_repeat"):
            checks["final_weights"] = {"passed": False, "error": "final weights could not be evaluated"}
            raise StopIteration
        checks["final_weights"] = {"passed": True}
        if len(ev["per_repeat"]) != METRIC["quality_repeats"]:
            raise RuntimeError(f"evaluation returned {len(ev['per_repeat'])} repeats, the metric needs {METRIC['quality_repeats']}")
        m = oc.metric_v2([100.0 * float(s) for s in ev["per_repeat"]], METRIC["base_quality_per_repeat"],
                         aud.get("wall_clock_seconds"), METRIC["quality_gate_min"], METRIC["score_scale_seconds"],
                         METRIC["quality_unit"])
        checks["quality_gate"] = {"passed": m["quality_gate_passed"], "quality_score": m["quality_score"],
                                  "quality_gate_min": m["quality_gate_min"], "unit": m["quality_unit"],
                                  "final_quality_per_repeat": m["final_quality_per_repeat"],
                                  "base_quality_per_repeat": m["base_quality_per_repeat"]}
        metrics.update({k: v for k, v in m.items() if k not in ("score", "summary")},
                       final_success=float(ev["success"]), eval_per_repeat=ev["per_repeat"],
                       wall_clock_seconds=aud.get("wall_clock_seconds"), excluded_seconds=aud.get("excluded_seconds"),
                       train_success_by_round=aud.get("success_by_round"), mean_turns=aud.get("mean_turns"))
        if m["score"] is None:
            raise RuntimeError(f"the audit passed but reported no usable timed window: {m['summary']}")
        result["score"] = m["score"]
        summary = m["summary"]
    except StopIteration:
        pass
    except Exception:
        result["valid"] = False
        metrics["judge_error"] = traceback.format_exc()[-4000:]
    finally:
        reap_agent()
    correct = bool(checks) and all(v.get("passed") for k, v in checks.items() if k != "quality_gate")
    result["metrics"] = {**metrics, "correctness_gate_passed": correct,
                         "quality_gate_passed": bool(checks.get("quality_gate", {}).get("passed"))}
    result["details"] = [{"name": k, "status": "PASSED" if v.get("passed") else "FAILED",
                          "message": json.dumps(v)[:2000]} for k, v in checks.items()]
    if not result["valid"]:
        result["summary"] = "evaluator error: not scored"
    elif not correct:
        result["summary"] = "correctness checks failed: score 0"
    else:
        result["summary"] = summary or f"score {result['score']:.3f}"
    oc.write_json(a.log_dir / "alfworld_private.json", {**result, "checks": checks, "private": private})
    print(">>>>> Start Structured Result")
    print(json.dumps(result, allow_nan=False))
    print(">>>>> End Structured Result")


if __name__ == "__main__":
    main()
