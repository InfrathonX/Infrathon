#!/usr/bin/env python3
"""Public checks for the Miles OPCD task, v3 (run inside Work, as the agent user). The teacher is the frozen
initial Qwen3-1.7B snapshot with the context (the contract's teacher_hf).

  python /opt/miles_opcd_public/run_public.py --suite probe --output /scratch/public/probe-001
  python /opt/miles_opcd_public/run_public.py --suite small --output /scratch/public/small-001 [--probe-output /scratch/public/probe-001]
  python /opt/miles_opcd_public/run_public.py --suite perf  --output /scratch/public/perf-001  [--probe-output /scratch/public/probe-001]

Every suite runs `task_entry.py` with the evaluator's instrumentation switched on (the
OPCD_AUDIT_* environment variables must reach every process your command starts).

probe  one OPCD step on 8 public fixed trajectories: the dump contract, the logged loss
       against the dumped tensors and an independent fp32 model, the top-256 set, the
       teacher's context-conditioned log-probs, that the weight change follows the OPCD
       gradient, and provenance: one Megatron-Core optimizer step driven by Miles'
       training actor with the contract's AdamW settings, whose result is final_hf.
small  4 rounds x 16 public prompts through `task_entry.py run`: provenance (every step
       is a Megatron-Core step inside Miles; the weights the SGLang engine served and the
       final checkpoint are exactly Megatron's, the final checkpoint tensor by tensor (M6);
       the engine was started by a Ray actor),
       the audit rules (prompts, sampling, one training use per row, staleness <= 1, one
       publication per step), every step's logged loss, step 0 against the fp32 oracle,
       the steps whose served weights were copied recomputed in fp32 (loss, top-k set,
       teacher log-probs and the gradient Megatron applied), and the teacher log-probs
       of random rows of two more steps. With --probe-output, also that the probe used
       the same Megatron numerics as this run.
perf   the public contract (8 rounds x 64 prompts): the same checks, the audited timed
       window T (minus the evaluator's weight copies and digests), and the final student's
       accuracy and answer-tag rate on the 64 public test questions with the Judge's protocol
       (3 seeded sampled repeats; the model the Judge scores: final_hf's safetensors and index
       with the pinned snapshot's config and tokenizer, final_hf's other files ignored), next
       to the original model's accuracy (published_reference.json).
These suites only get a pipeline working: their sizes are not the scored run's, and neither T
nor the accuracy predicts a score, so no score is reported here.
Exit code 0 means every check of the suite passed. The Judge runs its own copy of these
checks on hidden data and picks its own random steps; passing here does not guarantee a
Judge score. The weight copies (about 3.4 GB each) and perf's scored_model are deleted after
use unless --keep-snapshots is given.
"""

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import opcd_checks as oc  # noqa: E402

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
    env = dict(os.environ, OPCD_AUDIT_DIR=str(audit), OPCD_AUDIT_ROUNDS=str(1 if a.suite == "probe" else contract["rounds"]))

    def entry(cmd):
        oc.write_json(out / "contract.json", contract)
        code, secs = oc.run_logged([oc.PY, "task_entry.py", *cmd, "--contract", str(out / "contract.json")],
                                   out / "task_entry.log", cwd=a.app, env=env, timeout=a.timeout)
        report["checks"]["task_entry_exit"] = {"passed": code == 0, "code": code, "seconds": round(secs, 1)}
        return code == 0

    def provenance(mode, steps, final_hf, lr=None):
        cmd = [oc.PY, oc.tool("audit/opcd_provenance_check.py"), "--audit-dir", str(audit), "--mode", mode,
               "--steps", str(steps), "--contract", str(out / "contract.json"), "--initial-hf", oc.STUDENT,
               "--final-hf", str(final_hf), "--out", str(out / f"provenance_{mode}.json")]
        if lr is not None:
            cmd += ["--lr", repr(float(lr))]
        code, res = oc.run_check(cmd, out / f"provenance_{mode}.log")
        return res or {"passed": False, "error": f"see provenance_{mode}.log (exit {code})"}

    if a.suite == "probe":
        batch = "/data/public/probe_batch.json"
        if entry(["probe", "--batch", batch, "--out", str(out / "probe")]):
            code, res = oc.run_check([oc.PY, oc.tool("opcd_probe_check.py"), "--batch", batch,
                                      "--dumps", oc.step_files(out / "probe"), "--final-hf", str(out / "probe/final_hf"),
                                      "--student", oc.STUDENT, "--teacher", oc.TEACHER, "--context-file", oc.CONTEXT,
                                      "--top-k", str(base["top_k"])], out / "probe_check.log")
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
            code, res = oc.run_check([oc.PY, oc.tool("audit/opcd_audit_check.py"), "--audit-dir", str(audit),
                                      "--dumps", oc.step_files(out / "run"), "--data", contract["train_data"],
                                      "--tokenizer", oc.STUDENT, "--context-file", oc.CONTEXT,
                                      "--student-model-path", contract["student_hf"], "--rounds", str(R),
                                      "--per-round", str(contract["prompts_per_round"]),
                                      "--max-lag", str(contract["max_staleness"]), "--exclude-intervals", str(excl)],
                                     out / "audit_check.log")
            report["checks"]["audit"] = res or {"passed": False, "error": "see audit_check.log"}
            report["checks"]["steps"] = oc.steps_self_consistency(out / "run", R, contract["top_k"])
            if a.probe_output:
                want = (json.loads((a.probe_output / "provenance_probe.json").read_text()).get("model_config")
                        if (a.probe_output / "provenance_probe.json").is_file() else None)
                got = prov.get("model_config") or {}
                diff = {k: [(want or {}).get(k), got.get(k)] for k in sorted(set(want or {}) | set(got))
                        if (want or {}).get(k) != got.get(k)}
                report["checks"]["same_path"] = {"passed": bool(want) and not diff, "differences_probe_vs_run": diff}

            dump = lambda k: str(out / "run/steps" / f"step_{k:03d}_rank_*.pt")
            by_step = {}
            for s in prov.get("snapshots") or []:
                if "step" in s:
                    by_step.setdefault(s["step"], s["path"])
            chosen = sorted(by_step)[:2]
            rest = [k for k in range(1, R) if k not in chosen]
            teacher_steps = sorted(RNG.sample(rest, min(2, len(rest))))

            def cmd(k, dest, **kw):
                c = [oc.PY, oc.tool("opcd_oracle.py"), "--dump", dump(k), "--student", oc.STUDENT, "--teacher", oc.TEACHER,
                     "--context-file", oc.CONTEXT, "--top-k", str(contract["top_k"]), "--out", str(dest)]
                if kw.get("snapshot"):
                    c += ["--student-snapshot", kw["snapshot"], "--grad-records",
                          ",".join((prov.get("grad_records") or {}).get(str(k)) or [])]
                    if (prov.get("grad_norms") or {}).get(str(k)) is not None:
                        c += ["--grad-norm", repr(float(prov["grad_norms"][str(k)]))]
                if kw.get("teacher_only"):
                    c += ["--teacher-only", "--sample-rows", "32", "--row-seed", str(RNG.randrange(1 << 30))]
                return c

            jobs = [("step0", cmd(0, out / "oracle_step0.json"), out / "oracle_step0.log")]
            jobs += [(f"served{k}", cmd(k, out / f"oracle_step{k}.json", snapshot=by_step[k]), out / f"oracle_step{k}.log")
                     for k in chosen]
            jobs += [(f"teacher{k}", cmd(k, out / f"teacher_step{k}.json", teacher_only=True), out / f"teacher_step{k}.log")
                     for k in teacher_steps]
            res = oc.run_parallel(jobs)
            strip = lambda r: {k: v for k, v in (r or {}).items() if k != "per_sample"}
            ok, why = oc.oracle_verdict(res["step0"][1])
            report["checks"]["oracle_step0"] = {"passed": ok and res["step0"][0] == 0, "reasons": why, **strip(res["step0"][1])}
            served = {}
            for k in chosen:
                ok, why = oc.oracle_verdict(res[f"served{k}"][1], grad={})
                served[str(k)] = {"passed": ok and res[f"served{k}"][0] == 0, "reasons": why, **strip(res[f"served{k}"][1])}
            report["checks"]["oracle_served_steps"] = {"passed": bool(chosen) and oc.served_steps_passed(served),
                                                       "steps": served}
            teach = {}
            for k in teacher_steps:
                ok, why = oc.oracle_verdict(res[f"teacher{k}"][1], teacher_only=True)
                teach[str(k)] = {"passed": ok and res[f"teacher{k}"][0] == 0, "reasons": why, **strip(res[f"teacher{k}"][1])}
            report["checks"]["teacher_random_steps"] = {"passed": all(v["passed"] for v in teach.values()), "steps": teach}
            if not a.keep_snapshots:
                shutil.rmtree(audit / "snapshots", ignore_errors=True)
            if a.suite == "perf":
                published = json.loads(PUBLISHED.read_text())
                public_base = published.get("public_base_accuracy_per_repeat")
                ev = None
                if scored_model(out / "run/final_hf", out / "scored_model"):
                    code, ev = oc.run_check(oc.quality_eval_cmd(out / "scored_model", "/data/public/test_public.jsonl",
                                                                len(public_base) if public_base else 1), out / "eval.log")
                if not a.keep_snapshots:
                    shutil.rmtree(out / "scored_model", ignore_errors=True)
                report["performance"] = performance(report["checks"]["audit"], ev, published)
    report["passed"] = all(v.get("passed") for v in report["checks"].values()) and bool(report["checks"])
    oc.write_json(out / "report.json", report)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


def scored_model(final_hf, model_dir):
    """Assemble in model_dir the model the Judge scores (run_judge_v3.py, '3 evaluation'): the pinned student
    snapshot's non-weight files (config, tokenizer, generation config) with final_hf's safetensors and its
    index; final_hf's other files are ignored. Returns the shards copied ([]: nothing to evaluate)."""
    final_hf, model_dir = Path(final_hf), Path(model_dir)
    shards = sorted(final_hf.glob("*.safetensors"))
    if not shards:
        return []
    model_dir.mkdir()
    for f in Path(oc.STUDENT).iterdir():
        if not f.name.endswith(".safetensors") and f.name != "model.safetensors.index.json":
            shutil.copy(f.resolve(), model_dir / f.name)
    for f in shards:
        shutil.copy(f, model_dir / f.name)
    idx = final_hf / "model.safetensors.index.json"
    if idx.is_file():
        shutil.copy(idx, model_dir / idx.name)
    return shards


def performance(aud, ev, published):
    """The perf report: T, the public accuracy and answer-tag rate, and the original model's public accuracy."""
    aud, ev = aud or {}, ev or {}
    return {
        "wall_clock_seconds": aud.get("wall_clock_seconds"),
        "excluded_seconds": aud.get("excluded_seconds"),
        "public_test_accuracy": ev.get("accuracy"),
        "public_test_accuracy_per_repeat": [r.get("accuracy") for r in ev.get("per_repeat") or []],
        "public_test_answer_tag_rate": ev.get("answer_tag_rate"),
        "public_base_accuracy_per_repeat": published.get("public_base_accuracy_per_repeat"),
        "note": "for getting the pipeline working: this run is much smaller than the scored one, so neither T nor the "
                "accuracy predicts a score; only sforge-submit measures it",
    }


if __name__ == "__main__":
    main()
