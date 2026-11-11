#!/usr/bin/env python3
"""Judge for the Miles OPCD task, v3 (the implementation must stay inside Miles; the teacher is the frozen
initial Qwen3-1.7B snapshot conditioned on the context, the OPCD paper's same-size setting).

Runs as root in the Judge container; candidate code runs as the unprivileged `agent`
user and is never imported here. Both candidate commands run under the SGLang audit hook
and the provenance hook.

  1 probe  one OPCD step on hidden fixed trajectories -> opcd_probe_check (P1-P4) and
           provenance: one Megatron-Core step driven by Miles' training actor, AdamW with
           the batch's lr, final weights = that step's result
  2 run    the full contract -> provenance M1-M6 (Megatron-Core inside Miles, AdamW
           settings, the weights the engine served and the final checkpoint are exactly
           Megatron's, each step starts from the weights and optimizer state the previous
           one returned, a Ray-managed engine; M6: every final_hf tensor equals the full
           digests of the trainer after its last step and of the engine after its last
           publication), audit W1-W5 (the audited timed window must lie inside the run as
           the Judge's own clock saw it, with at most EXCLUDED_MAX_SECONDS of evaluator
           intervals taken out), every step's logged loss, probe and run use the same
           Megatron numerics, step 0 vs the fp32 oracle, two random steps recomputed from
           copies of the weights the engine served (loss, top-k set, teacher log-probs, and
           the gradient Megatron applied), and the teacher log-probs of two more random steps
  3 eval   accuracy of the final student (no context) on the 2436 hidden questions, 3 seeded
           sampled repeats (protocol S1: opcd_checks.QUALITY_PROTOCOL); in parallel on the second GPU,
           report-only, the paper-protocol diagnostic (opcd_checks.DIAG_PROTOCOL) on the 500-question test set
  score = quality_score * 3600 / T   if every check passes and quality_score >= the quality gate, else 0
          quality_score = mean of the 3 repeats - the original model's frozen 3-repeat mean, in accuracy
          points; T = the audited timed window (opcd_checks.metric_v2, qualification.json "metric")
T excludes the time the evaluator's own weight copies and digests took inside the timed window.
A candidate failure (a check or the quality gate) is valid=true, score=0; valid=false only for
Judge/infrastructure errors. Every record the candidate writes is read with torch.load(weights_only=True).
Reported, never scored: the answer-tag rate and the accuracy of tagged answers (final and original student),
the In-Context reference (the original student with the context, the teacher itself) and the paper-protocol
diagnostic, so a gain from the answer format is told apart from a gain in the answers.
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
import opcd_checks as oc  # noqa: E402

QUAL = json.loads((HERE / "qualification.json").read_text())
METRIC = QUAL["metric"]
PRIVATE = HERE / "private"
WORK = Path("/scratch/judge")
AGENT = pwd.getpwnam("agent")
RNG = random.SystemRandom()
N_RANDOM_ORACLE = 4  # every served-weight copy the hook takes (opcd_provenance_hook.N_SNAPSHOTS)
N_TEACHER_SPOT, TEACHER_SPOT_ROWS = 2, 32
DETAIL_KEYS = ("passed", "issues", "issue_counts", "reasons", "error", "code", "seconds")
# the evaluator's intervals taken out of T (two snapshot copies, the trainer's digest and the engine's digests
# after the publications that can be its last, each credited at most opcd_provenance_check.CREDIT_MAX_SECONDS)
# add up to at most this; the H100 reference's two copies took 7.6 s
EXCLUDED_MAX_SECONDS = 60.0


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


def candidate(app, name, args, timeout, rounds):
    """Fresh agent-owned copy of the submission; run task_entry.py under both hooks; kill what it left."""
    d = WORK / name
    shutil.copytree(app, d / "miles_lab", symlinks=True)
    for sub in ("out", "scratch", "audit"):
        (d / sub).mkdir()
    subprocess.run(["chown", "-R", "agent:agent", str(d)], check=True)
    contract = oc.load_contract(HERE / "contract.json", scratch=str(d / "scratch"))
    contract_path = d / "contract.json"
    oc.write_json(contract_path, contract)
    os.chmod(contract_path, 0o644)
    env = agent_env({"OPCD_AUDIT_DIR": str(d / "audit"), "OPCD_AUDIT_ROUNDS": str(rounds)})
    try:
        code, secs = oc.run_logged([oc.PY, "task_entry.py", *args(d), "--contract", str(contract_path)],
                                   d / "task_entry.log", cwd=d / "miles_lab", env=env,
                                   timeout=timeout, preexec=as_agent)
    finally:
        reap_agent()
    return d, contract, code, secs


def provenance(mode, d, steps, final_hf, lr=None):
    out = WORK / f"provenance_{mode}.json"
    cmd = [oc.PY, oc.tool("audit/opcd_provenance_check.py"), "--audit-dir", str(d / "audit"), "--mode", mode,
           "--steps", str(steps), "--contract", str(HERE / "contract.json"), "--initial-hf", oc.STUDENT,
           "--final-hf", str(final_hf), "--out", str(out)]
    if lr is not None:
        cmd += ["--lr", repr(float(lr))]
    code, res = oc.run_check(cmd, WORK / f"provenance_{mode}.log")
    return res or {"passed": False, "error": f"provenance checker produced no report (exit {code})"}


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


def oracle_cmd(c, dump, out, **kw):
    cmd = [oc.PY, oc.tool("opcd_oracle.py"), "--dump", dump, "--student", oc.STUDENT, "--teacher", oc.TEACHER,
           "--context-file", oc.CONTEXT, "--top-k", str(c["top_k"]), "--out", str(out)]
    if kw.get("snapshot"):
        cmd += ["--student-snapshot", kw["snapshot"]]
    if kw.get("grads"):
        cmd += ["--grad-records", ",".join(kw["grads"])]
        if kw.get("grad_norm") is not None:
            cmd += ["--grad-norm", repr(float(kw["grad_norm"]))]
    if kw.get("teacher_only"):
        cmd += ["--teacher-only", "--sample-rows", str(kw["rows"]), "--row-seed", str(kw["seed"])]
    return cmd


def report_only(ev, diag):
    """Never scored: the format/content split and the references the OPCD paper reports next to OPCD."""
    ref = METRIC.get("report_only_references") or {}
    return {"final_answer_tag_rate": ev.get("answer_tag_rate"), "final_tagged_accuracy": ev.get("tagged_accuracy"),
            "paper_protocol_final_accuracy": diag.get("accuracy"),
            "paper_protocol_final_answer_tag_rate": diag.get("answer_tag_rate"),
            "paper_protocol_error": None if "accuracy" in diag else "the diagnostic produced no report",
            **ref}


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
    grad_limits = QUAL.get("grad_limits")
    try:
        if not (a.app / "task_entry.py").is_file() or not (a.app / "repo/train.py").is_file():
            checks["submission"] = {"passed": False, "error": "task_entry.py or repo/ missing"}
            raise StopIteration

        # ---- 1 probe on hidden trajectories ----
        batch = WORK / "probe_batch.json"
        shutil.copy(PRIVATE / "probe_batch.json", batch)
        os.chmod(batch, 0o644)
        probe_lr = json.loads(batch.read_text())["lr"]
        d, c, code, secs = candidate(a.app, "probe", lambda d: ["probe", "--batch", str(batch), "--out", str(d / "out/probe")],
                                     timeout=QUAL["timeouts"]["probe"], rounds=1)
        checks["probe_exit"] = {"passed": code == 0, "code": code, "seconds": round(secs, 1)}
        if code != 0:
            raise StopIteration
        _, res = oc.run_check([oc.PY, oc.tool("opcd_probe_check.py"), "--batch", str(batch),
                               "--dumps", oc.step_files(d / "out/probe"), "--final-hf", str(d / "out/probe/final_hf"),
                               "--student", oc.STUDENT, "--teacher", oc.TEACHER, "--context-file", oc.CONTEXT,
                               "--top-k", str(c["top_k"])], WORK / "probe_check.log")
        checks["probe"] = res or {"passed": False, "error": "probe checker produced no report"}
        prov_probe = provenance("probe", d, 1, d / "out/probe/final_hf", lr=probe_lr)
        private["provenance_probe"] = prov_probe
        checks["probe_provenance"] = compact(prov_probe)
        if not (checks["probe"].get("passed") and prov_probe.get("passed")):
            raise StopIteration

        # ---- 2 full run under the audit and provenance hooks ----
        started = time.time()
        d, c, code, secs = candidate(a.app, "run", lambda d: ["run", "--out", str(d / "out/run")],
                                     timeout=QUAL["timeouts"]["run"], rounds=c["rounds"])
        ended = time.time()
        checks["run_exit"] = {"passed": code == 0, "code": code, "seconds": round(secs, 1)}
        if code != 0:
            raise StopIteration
        R = c["rounds"]
        prov = provenance("run", d, R, d / "out/run/final_hf")
        private["provenance_run"] = prov
        checks["provenance"] = compact(prov)
        excl = WORK / "exclude_intervals.json"
        oc.write_json(excl, prov.get("exclude_intervals") or [])
        _, res = oc.run_check([oc.PY, oc.tool("audit/opcd_audit_check.py"), "--audit-dir", str(d / "audit"),
                               "--dumps", oc.step_files(d / "out/run"), "--data", c["train_data"], "--tokenizer", oc.STUDENT,
                               "--context-file", oc.CONTEXT, "--student-model-path", c["student_hf"],
                               "--rounds", str(R), "--per-round", str(c["prompts_per_round"]),
                               "--max-lag", str(c["max_staleness"]), "--exclude-intervals", str(excl)],
                              WORK / "audit_check.log")
        checks["audit"] = aud = res or {"passed": False, "error": "audit checker produced no report"}
        why = judge_clock_issues(aud, started, ended) if aud.get("passed") else []
        if why:
            aud["passed"] = False
            aud.setdefault("issues", {}).setdefault("T", []).extend(why)
            aud.setdefault("issue_counts", {})["T"] = len(aud["issues"]["T"])
        checks["steps"] = oc.steps_self_consistency(d / "out/run", R, c["top_k"])
        want, got = prov_probe.get("model_config") or {}, prov.get("model_config") or {}
        diff = {k: [want.get(k), got.get(k)] for k in sorted(set(want) | set(got)) if want.get(k) != got.get(k)}
        checks["same_path"] = {"passed": bool(want) and not diff, "differences_probe_vs_run": diff}
        if not all(checks[k].get("passed") for k in ("provenance", "audit", "steps", "same_path")):
            raise StopIteration

        # ---- step 0, two random served-weight steps (with the gradient), two random teacher steps ----
        steps_dir = d / "out/run/steps"
        dump = lambda k: str(steps_dir / f"step_{k:03d}_rank_*.pt")
        by_step = {}
        for s in prov.get("snapshots") or []:
            if "step" in s:
                by_step.setdefault(s["step"], s["path"])
        chosen = sorted(RNG.sample(sorted(by_step), min(N_RANDOM_ORACLE, len(by_step))))
        rest = [k for k in range(1, R) if k not in chosen]
        teacher_steps = sorted(RNG.sample(rest, min(N_TEACHER_SPOT, len(rest))))
        jobs = [("step0", oracle_cmd(c, dump(0), WORK / "oracle_step0.json"), WORK / "oracle_step0.log")]
        for k in chosen:
            jobs.append((f"served{k}", oracle_cmd(c, dump(k), WORK / f"oracle_step{k}.json", snapshot=by_step[k],
                                                  grads=(prov.get("grad_records") or {}).get(str(k)),
                                                  grad_norm=(prov.get("grad_norms") or {}).get(str(k))),
                         WORK / f"oracle_step{k}.log"))
        for k in teacher_steps:
            jobs.append((f"teacher{k}", oracle_cmd(c, dump(k), WORK / f"teacher_step{k}.json", teacher_only=True,
                                                   rows=TEACHER_SPOT_ROWS, seed=RNG.randrange(1 << 30)),
                         WORK / f"teacher_step{k}.log"))
        out = oc.run_parallel(jobs)
        strip = lambda r: {k: v for k, v in (r or {}).items() if k != "per_sample"}
        ok, why = oc.oracle_verdict(out["step0"][1])
        checks["oracle_step0"] = {"passed": ok and out["step0"][0] == 0, "reasons": why, **strip(out["step0"][1])}
        served = {}
        for k in chosen:
            code, res = out[f"served{k}"]
            ok, why = oc.oracle_verdict(res, grad=grad_limits or {})
            served[str(k)] = {"passed": ok and code == 0, "reasons": why, **strip(res)}
        checks["oracle_served_steps"] = {"passed": bool(chosen) and oc.served_steps_passed(served),
                                         "steps": chosen, "reasons": {k: v["reasons"] for k, v in served.items() if v["reasons"]}}
        private["oracle_served_steps"] = served
        teach = {}
        for k in teacher_steps:
            code, res = out[f"teacher{k}"]
            ok, why = oc.oracle_verdict(res, teacher_only=True)
            teach[str(k)] = {"passed": ok and code == 0, "reasons": why, **strip(res)}
        checks["teacher_random_steps"] = {"passed": all(v["passed"] for v in teach.values()), "steps": teacher_steps,
                                          "reasons": {k: v["reasons"] for k, v in teach.items() if v["reasons"]}}
        private["teacher_random_steps"] = teach
        if not all(checks[k].get("passed") for k in ("oracle_step0", "oracle_served_steps", "teacher_random_steps")):
            raise StopIteration

        # ---- 3 evaluation of the final student (Judge-rendered, pinned tokenizer/config) ----
        final = d / "out/run/final_hf"
        model_dir = WORK / "scored_model"
        model_dir.mkdir()
        shards = sorted(final.glob("*.safetensors"))
        if not shards:
            checks["final_weights"] = {"passed": False, "error": "no safetensors in final_hf"}
            raise StopIteration
        for f in Path(oc.STUDENT).iterdir():
            if not f.name.endswith(".safetensors") and f.name != "model.safetensors.index.json":
                shutil.copy(f.resolve(), model_dir / f.name)
        for f in shards:
            shutil.copy(f, model_dir / f.name)
        idx = final / "model.safetensors.index.json"
        if idx.is_file():
            shutil.copy(idx, model_dir / idx.name)
        out = oc.run_parallel([("quality", oc.quality_eval_cmd(model_dir, PRIVATE / "eval.jsonl", METRIC["quality_repeats"]),
                                WORK / "eval.log"),
                               ("diagnostic", oc.diagnostic_eval_cmd(model_dir, PRIVATE / "paper_test.jsonl"),
                                WORK / "eval_paper_protocol.log")])
        ev = out["quality"][1]
        diag = out["diagnostic"][1] or {}
        if not ev or "accuracy" not in ev or not ev.get("per_repeat"):
            checks["final_weights"] = {"passed": False, "error": "final weights could not be evaluated"}
            raise StopIteration
        checks["final_weights"] = {"passed": True}
        why = oc.eval_protocol_issue(ev, METRIC["quality_repeats"])
        if why:
            raise RuntimeError(why)
        m = oc.metric_v2([100.0 * float(r["accuracy"]) for r in ev["per_repeat"]], METRIC["base_quality_per_repeat"],
                         checks["audit"].get("wall_clock_seconds"), METRIC["quality_gate_min"],
                         METRIC["score_scale_seconds"], METRIC["quality_unit"])
        checks["quality_gate"] = {"passed": m["quality_gate_passed"], "quality_score": m["quality_score"],
                                  "quality_gate_min": m["quality_gate_min"], "unit": m["quality_unit"],
                                  "final_quality_per_repeat": m["final_quality_per_repeat"],
                                  "base_quality_per_repeat": m["base_quality_per_repeat"]}
        metrics.update({k: v for k, v in m.items() if k not in ("score", "summary")},
                       accuracy=float(ev["accuracy"]), accuracy_per_repeat=[r["accuracy"] for r in ev["per_repeat"]],
                       answer_tag_rate=ev.get("answer_tag_rate"), tagged_accuracy=ev.get("tagged_accuracy"),
                       report_only=report_only(ev, diag),
                       wall_clock_seconds=checks["audit"].get("wall_clock_seconds"),
                       excluded_seconds=checks["audit"].get("excluded_seconds"))
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
                          "message": json.dumps({kk: vv for kk, vv in v.items() if kk != "per_step"})[:2000]}
                         for k, v in checks.items()]
    if not result["valid"]:
        result["summary"] = "evaluator error: not scored"
    elif not correct:
        result["summary"] = "correctness checks failed: score 0"
    else:
        result["summary"] = summary or f"score {result['score']:.3f}"
    try:  # diagnostics only: a checker report may carry NaN (e.g. a gradient cosine), and must not cost the result
        oc.write_json(a.log_dir / "opcd_private.json", {**result, "checks": checks, "private": private}, allow_nan=True)
    except (TypeError, ValueError, OSError) as exc:
        print(f"opcd_private.json not written: {exc!r}", file=sys.stderr)
    print(">>>>> Start Structured Result")
    print(json.dumps(result, allow_nan=False))
    print(">>>>> End Structured Result")


if __name__ == "__main__":
    main()
