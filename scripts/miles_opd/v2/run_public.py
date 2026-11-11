#!/usr/bin/env python3
"""Public checks for the Miles OPD task (run inside Work, as the agent user).

  python /opt/miles_opd_public/run_public.py --suite probe --output /scratch/public/probe-001
  python /opt/miles_opd_public/run_public.py --suite small --output /scratch/public/small-001 [--probe-output /scratch/public/probe-001]
  python /opt/miles_opd_public/run_public.py --suite perf  --output /scratch/public/perf-001  [--probe-output /scratch/public/probe-001]

Every suite runs `task_entry.py` with the evaluator's instrumentation switched on (the
OPD_AUDIT_* environment variables must reach every process your command starts).

probe  one OPD step on 8 public fixed trajectories; checks the dump contract, the
       advantages and logged loss against the dumped tensors, the student and teacher
       log-probabilities against an independent fp32 computation, that the weight
       change follows the OPD gradient (the Judge also compares it with the reference
       implementation's change on its hidden batch), and provenance: one Megatron-Core
       optimizer step driven by Miles' training actor with the contract's AdamW
       settings, whose result is final_hf.
small  4 rounds x 16 public prompts through `task_entry.py run`: provenance (every step
       is a Megatron-Core step inside Miles; the weights the student SGLang engine served
       and the final checkpoint are exactly Megatron's; the engine was started by a Ray
       actor), the audit rules (engines, prompts, sampling, one training use per row,
       staleness <= 1, one publication per step), every step's advantages and loss, step
       0 against the fp32 oracle, the steps whose served weights were copied recomputed in
       fp32 (old and teacher log-probs, loss, and the gradient Megatron applied), and the
       teacher log-probs of random rows of two more steps; final_hf equal, tensor by
       tensor, to Megatron's weights after its last step and to what each student engine
       served after its last publication (provenance M6). With --probe-output, also that
       the probe used the same Megatron numerics as this run.
perf   the public contract (8 rounds x 32 prompts): the same checks, the audited timed
       window T (minus the evaluator's weight copies and digests), and the final student's
       accuracy and answer-format rates on the 256 public test problems with the Judge's
       protocol (3 seeded sampled repeats), next to the original student's accuracy
       (published_reference.json).
These suites only get a pipeline working: their sizes are not the scored run's, and neither T
nor the accuracy predicts a score, so no score is reported here.
Exit code 0 means every check of the suite passed. The Judge runs its own copy of these
checks on hidden data and picks its own random steps; passing here does not guarantee a
Judge score. The weight copies (about 3.4 GB each) are deleted after the checks unless
--keep-snapshots is given.
"""

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import opd_checks as oc  # noqa: E402

CONTRACT = str(Path(__file__).resolve().parent / "contract.json")
PUBLISHED = Path(__file__).resolve().parent / "published_reference.json"
RNG = random.SystemRandom()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=["probe", "small", "perf"], required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--app", type=Path, default=Path("/app/workdir/miles_lab"))
    ap.add_argument("--contract", type=Path, default=Path(CONTRACT))
    ap.add_argument("--probe-output", type=Path, default=None, help="a passed probe suite's --output, to compare numerics")
    ap.add_argument("--keep-snapshots", action="store_true")
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args()
    out = a.output
    out.mkdir(parents=True, exist_ok=False)
    report = {"suite": a.suite, "checks": {}}
    base = oc.load_contract(a.contract, scratch=str(out / "scratch"))
    contract = base if a.suite in ("probe", "perf") else dict(base, rounds=4, prompts_per_round=16)
    audit = out / "audit"
    audit.mkdir()
    env = dict(os.environ, OPD_AUDIT_DIR=str(audit), OPD_AUDIT_ROUNDS=str(1 if a.suite == "probe" else contract["rounds"]))

    def entry(cmd):
        oc.write_json(out / "contract.json", contract)
        code, secs = oc.run_logged([oc.PY, "task_entry.py", *cmd, "--contract", str(out / "contract.json")],
                                   out / "task_entry.log", cwd=a.app, env=env, timeout=a.timeout)
        report["checks"]["task_entry_exit"] = {"passed": code == 0, "code": code, "seconds": round(secs, 1)}
        return code == 0

    def provenance(mode, steps, final_hf, lr=None):
        cmd = [oc.PY, oc.tool("audit/opd_provenance_check.py"), "--audit-dir", str(audit), "--mode", mode,
               "--steps", str(steps), "--contract", str(out / "contract.json"), "--initial-hf", contract["student_hf"],
               "--final-hf", str(final_hf), "--out", str(out / f"provenance_{mode}.json")]
        if lr is not None:
            cmd += ["--lr", repr(float(lr))]
        code, res = oc.run_check(cmd, out / f"provenance_{mode}.log")
        return res or {"passed": False, "error": f"see provenance_{mode}.log (exit {code})"}

    if a.suite == "probe":
        batch = "/data/public/probe_batch.json"
        if entry(["probe", "--batch", batch, "--out", str(out / "probe")]):
            code, res = oc.run_check([oc.PY, oc.tool("opd_probe_check.py"), "--batch", batch,
                                      "--dumps", oc.step_files(out / "probe"), "--final-hf", str(out / "probe/final_hf"),
                                      "--student", contract["student_hf"], "--teacher", contract["teacher_hf"],
                                      "--kl-coef", str(contract["kl_coef"]),
                                      "--weight-decay", str(contract["optimizer"]["weight_decay"])], out / "probe_check.log")
            report["checks"]["probe"] = res or {"passed": False, "error": "see probe_check.log"}
            report["checks"]["provenance"] = provenance("probe", 1, out / "probe/final_hf",
                                                        lr=json.loads(Path(batch).read_text())["lr"])
    else:
        R = contract["rounds"]
        if entry(["run", "--out", str(out / "run")]):
            prov = provenance("run", R, out / "run/final_hf")
            report["checks"]["provenance"] = prov
            excl = out / "exclude_intervals.json"
            oc.write_json(excl, prov.get("exclude_intervals") or [])
            code, res = oc.run_check([oc.PY, oc.tool("audit/opd_audit_check.py"), "--audit-dir", str(audit),
                                      "--dumps", oc.step_files(out / "run"), "--data", contract["train_data"],
                                      "--tokenizer", contract["student_hf"],
                                      "--student-model-path", contract["student_hf"],
                                      "--teacher-model-path", contract["teacher_hf"],
                                      "--rounds", str(R), "--per-round", str(contract["prompts_per_round"]),
                                      "--max-new-tokens", str(contract["max_response_len"]),
                                      "--max-lag", str(contract["max_staleness"]),
                                      "--temperature", str(contract["sampling"]["temperature"]),
                                      "--top-p", str(contract["sampling"]["top_p"]),
                                      "--template-kwargs", json.dumps(contract["template_kwargs"]),
                                      "--exclude-intervals", str(excl)], out / "audit_check.log")
            report["checks"]["audit"] = res or {"passed": False, "error": "see audit_check.log"}
            report["checks"]["steps"] = oc.steps_self_consistency(out / "run", R, contract["kl_coef"])
            if a.probe_output:
                want = (json.loads((a.probe_output / "provenance_probe.json").read_text()).get("model_config")
                        if (a.probe_output / "provenance_probe.json").is_file() else None)
                got = prov.get("model_config") or {}
                diff = {k: [(want or {}).get(k), got.get(k)] for k in sorted(set(want or {}) | set(got))
                        if (want or {}).get(k) != got.get(k)}
                report["checks"]["same_path"] = {"passed": bool(want) and not diff, "differences_probe_vs_run": diff}

            # step 0 from the initial student, the steps whose served weights were copied (with the
            # gradient), and random rows of two more steps (teacher only)
            dump = lambda k: str(out / "run/steps" / f"step_{k:03d}_rank_*.pt")
            by_step = {}
            for s in prov.get("snapshots") or []:
                if "step" in s:
                    by_step.setdefault(s["step"], s["path"])
            chosen = sorted(by_step)[:2]
            rest = [k for k in range(1, R) if k not in chosen]
            teacher_steps = sorted(RNG.sample(rest, min(2, len(rest))))

            def cmd(k, dest, **kw):
                c = [oc.PY, oc.tool("opd_step_oracle.py"), "--dumps", dump(k), "--student", contract["student_hf"],
                     "--teacher", contract["teacher_hf"], "--kl-coef", str(contract["kl_coef"]), "--out", str(dest)]
                if kw.get("snapshot"):
                    c += ["--student-snapshot", kw["snapshot"], "--grad-records",
                          ",".join((prov.get("grad_records") or {}).get(str(k)) or [])]
                    if (prov.get("grad_norms") or {}).get(str(k)) is not None:
                        c += ["--grad-norm", repr(float(prov["grad_norms"][str(k)]))]
                if kw.get("teacher_only"):
                    c += ["--teacher-only", "--sample-rows", "64", "--row-seed", str(RNG.randrange(1 << 30))]
                return c

            jobs = [("step0", cmd(0, out / "oracle_step0.json"), out / "oracle_step0.log")]
            jobs += [(f"served{k}", cmd(k, out / f"oracle_step{k}.json", snapshot=by_step[k]), out / f"oracle_step{k}.log")
                     for k in chosen]
            jobs += [(f"teacher{k}", cmd(k, out / f"teacher_step{k}.json", teacher_only=True), out / f"teacher_step{k}.log")
                     for k in teacher_steps]
            res = oc.run_parallel(jobs)
            ok, why = oc.step_oracle_verdict(res["step0"][1], oc.LIMITS)
            report["checks"]["oracle_step0"] = {**(res["step0"][1] or {}), "passed": ok and res["step0"][0] == 0,
                                                "reasons": why}
            served = {}
            for k in chosen:
                ok, why = oc.step_oracle_verdict(res[f"served{k}"][1], oc.LIMITS, grad=True)
                served[str(k)] = {**(res[f"served{k}"][1] or {}), "passed": ok and res[f"served{k}"][0] == 0, "reasons": why}
            report["checks"]["oracle_served_steps"] = {"passed": bool(chosen) and oc.served_steps_passed(served),
                                                       "steps": served}
            teach = {}
            for k in teacher_steps:
                ok, why = oc.step_oracle_verdict(res[f"teacher{k}"][1], oc.LIMITS, teacher_only=True)
                teach[str(k)] = {**(res[f"teacher{k}"][1] or {}), "passed": ok and res[f"teacher{k}"][0] == 0,
                                 "reasons": why}
            report["checks"]["teacher_random_steps"] = {"passed": all(v["passed"] for v in teach.values()), "steps": teach}
            if not a.keep_snapshots:
                shutil.rmtree(audit / "snapshots", ignore_errors=True)
            if a.suite == "perf":
                published = json.loads(PUBLISHED.read_text())
                base = published.get("public_base_accuracy_per_repeat")
                oc.run_logged(oc.eval_math_cmd(out / "run/final_hf", contract["student_hf"],
                                               {"public": "/data/public/test_public.jsonl"}, contract["max_response_len"],
                                               len(base) if base else oc.EVAL_REPEATS) + ["--out", str(out / "eval.json")],
                              out / "eval.log")
                try:
                    ev = json.loads((out / "eval.json").read_text())["sets"]["public"]
                except Exception:
                    ev = None
                report["performance"] = performance(report["checks"]["audit"], ev, published)
    report["passed"] = all(v.get("passed") for v in report["checks"].values()) and bool(report["checks"])
    oc.write_json(out / "report.json", report)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


def performance(aud, ev, published):
    """The perf report: T, the public accuracy and answer-format rates, and the original student's public accuracy."""
    aud, ev = aud or {}, ev or {}
    return {
        "wall_clock_seconds": aud.get("wall_clock_seconds"),
        "excluded_seconds": aud.get("excluded_seconds"),
        "public_test_accuracy": ev.get("accuracy"),
        "public_test_accuracy_per_repeat": list(ev.get("accuracy_per_repeat") or []),
        "public_test_boxed_rate": ev.get("boxed_rate"),
        "public_test_answer_line_rate": ev.get("answer_line_rate"),
        "public_test_truncated_rate": ev.get("truncated_rate"),
        "public_base_accuracy_per_repeat": published.get("public_base_accuracy_per_repeat"),
        "note": "for getting the pipeline working: this run is much smaller than the scored one, so neither T nor the "
                "accuracy predicts a score; only sforge-submit measures it",
    }


if __name__ == "__main__":
    main()
