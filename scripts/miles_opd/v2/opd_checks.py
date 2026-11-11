#!/usr/bin/env python3
"""Per-step checks and shared plumbing for the Miles OPD public runner and Judge.
No candidate or Miles code is imported.

As a module: process helpers; `steps_self_consistency` (every step dump: keys and
shapes, advantages == -beta (old - teacher) on loss tokens, logged loss == fp64
recomputation (1/N) sum_i masked-mean(old - teacher) over all ranks);
`step_oracle_verdict`, the limits applied to an `opd_step_oracle.py` report (step 0, the
served-weight steps with the gradient, the teacher-only spot steps); `run_parallel`,
which runs those oracles one per GPU; `eval_math_cmd`, the quality evaluation's command
(metric-v2 protocol, math grading); and `metric_v2`, the task's scoring arithmetic.

As a command, the per-step self-consistency of a run output directory:

  python opd_checks.py steps --out-dir OUT --rounds R [--kl-coef 1.0]
"""

import argparse
import glob
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
# metric-v2 quality protocol: the training sampling contract (temperature 1, top_p 1, no top-k, max_new_tokens =
# the contract's max_response_len) with per-request seeds in SGLang's deterministic-inference mode, repeat r seeded
# r, one engine for every set; graded by math_grader.grade. The frozen base
# accuracies (qualification.json metric, published_reference.json) were measured with exactly these flags.
EVAL_REPEATS = 3
EVAL_SAMPLING = ("--temperature", "1.0", "--top-p", "1.0", "--mem-fraction", "0.7", "--seed", "0")
KEYS = ("tokens", "response_lengths", "loss_masks", "old_logprobs", "teacher_logprobs", "advantages")
ADV_ABS, LOSS_REL = 1e-4, 1e-3
# step-oracle limits for the public runner; the Judge reads its own from qualification.json
# (step_oracle_limits, grad_limits); keep the two equal. The log-prob and loss limits are the probe
# check's P3 limits; the gradient limits come from the reference's public runs (see qualification.json).
STEP_ORACLE_LIMITS = {"student_lp_err": 0.05, "teacher_lp_err": 0.05, "loss_abs": 0.02}
GRAD_LIMITS = {"cos_min": 0.93, "slice_ratio": [0.75, 1.35], "norm_ratio": [0.75, 1.35]}  # = runtime/qualification.json
LIMITS = {**STEP_ORACLE_LIMITS, **GRAD_LIMITS}


def load_contract(path, **overrides):
    c = json.loads(Path(path).read_text())
    c.update(overrides)
    return c


def write_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")


def json_safe(x):
    """`x` with non-finite floats (e.g. from a report of a diverged run) replaced by strings"""
    if isinstance(x, dict):
        return {k: json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    return _num(x)


def run_logged(cmd, log, *, cwd=None, env=None, timeout=None, preexec=None):
    """Run a command in its own session, logging to `log`; kill the whole group on timeout."""
    t0 = time.time()
    with open(log, "w") as fh:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                start_new_session=True, preexec_fn=preexec)
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            code = "timeout"
    return code, time.time() - t0


def tool(name):
    return str(HERE / name)


def step_files(out_dir):
    return str(Path(out_dir) / "steps" / "step_*_rank_*.pt")


def json_tail(text):
    i = text.find("{")
    if i < 0:
        return None
    try:
        return json.loads(text[i:text.rindex("}") + 1])
    except ValueError:
        return None


def run_check(cmd, log):
    code, _ = run_logged(cmd, log)
    return code, json_tail(Path(log).read_text())


def _num(x):
    """JSON-safe number (non-finite values become strings)"""
    return x if not isinstance(x, float) or math.isfinite(x) else repr(x)


def steps_self_consistency(out_dir, rounds, kl_coef):
    """Per step: dump keys/shapes, advantages vs the dumped log-probs, logged loss vs fp64 recomputation."""
    import torch

    issues, per_step = [], {}
    by_step = {}
    for f in sorted(glob.glob(step_files(out_dir))):
        m = re.search(r"step_(\d+)_rank_(\d+)\.pt$", f)
        if m:
            by_step.setdefault(int(m.group(1)), []).append(f)
    if sorted(by_step) != list(range(rounds)):
        issues.append(f"step dumps present for {sorted(by_step)}, expected 0..{rounds - 1}")
    for s, files in sorted(by_step.items()):
        per_seq, logged, adv_err, bad = [], [], 0.0, False
        for f in files:
            r = torch.load(f, map_location="cpu", weights_only=True)
            miss = [k for k in (*KEYS, "loss", "step") if k not in r]
            if miss:
                issues.append(f"step {s}: {Path(f).name} misses {miss}")
                bad = True
                continue
            if r["step"] != s:
                issues.append(f"step {s}: {Path(f).name} records step {r['step']}")
            logged.append(r["loss"] if isinstance(r["loss"], (int, float)) else repr(r["loss"])[:80])
            if len({len(r[k]) for k in KEYS}) != 1:
                issues.append(f"step {s}: {Path(f).name} per-sequence lists differ in length")
                bad = True
                continue
            for tok, R, m_, lp, lq, a in zip(*(r[k] for k in KEYS)):
                R = int(R)
                if any(tuple(x.shape) != (R,) for x in (m_, lp, lq, a)) or not 0 < R < len(tok):
                    issues.append(f"step {s}: tensor shapes do not match response length {R}")
                    bad = True
                    break
                m_ = m_.double() > 0
                lp, lq, a = lp.double(), lq.double(), a.double()
                if m_.any():
                    err = float((a + kl_coef * (lp - lq))[m_].abs().max())
                    adv_err = err if not math.isfinite(err) else max(adv_err, err)
                    per_seq.append(float((lp - lq)[m_].mean()))
                else:
                    per_seq.append(0.0)
        ref = sum(per_seq) / max(len(per_seq), 1)
        per_step[s] = {"logged": _num(logged[0]) if logged else None, "recomputed": _num(ref),
                       "sequences": len(per_seq), "advantage_max_err": _num(adv_err)}
        if bad:
            continue
        if not adv_err <= ADV_ABS:
            issues.append(f"step {s}: advantages differ from -beta*(old - teacher) by up to {adv_err:.3g}")
        tol = LOSS_REL * max(abs(ref), 1e-3)
        if not all(isinstance(x, (int, float)) and math.isfinite(x) and abs(x - ref) <= tol for x in logged):
            issues.append(f"step {s}: logged loss {[_num(x) for x in logged[:4]]} != {ref:.6f} recomputed from the dumped tensors")
    return {"passed": not issues, "issues": issues[:20], "per_step": per_step}



def served_steps_passed(served):
    """The served-step recomputations pass unless more than half of the recomputed steps fail. One step can miss the
    gradient limits through bf16 noise alone (the honest reference did once in ten steps on the H100 SXM pair); a
    solution that computes a different loss misses them on every step."""
    failed = sum(not v["passed"] for v in served.values())
    return bool(served) and 2 * failed <= len(served)


def step_oracle_verdict(report, limits, grad=False, teacher_only=False):
    """(passed, reasons) for an opd_step_oracle.py report. `limits` holds student_lp_err,
    teacher_lp_err, loss_abs and, with `grad`, cos_min, slice_ratio and norm_ratio. A missing,
    None or non-finite metric fails."""
    if not report:
        return False, ["step oracle produced no report"]
    why = []

    def value(key):
        v = report.get(key)
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None

    def at_most(key, limit, what):
        v = value(key)
        if v is None or v > limit:
            why.append(f"{what}: {key} {report.get(key)} (limit {limit})")

    def within(key, bounds, what):
        v = value(key)
        if v is None or not bounds[0] <= v <= bounds[1]:
            why.append(f"{what}: {key} {report.get(key)} (limits {bounds})")

    at_most("teacher_lp_mean_err", limits["teacher_lp_err"], "teacher log-probs differ from the fp32 teacher")
    if teacher_only:
        return not why, why
    at_most("student_lp_mean_err", limits["student_lp_err"],
            "old log-probs differ from the fp32 student at the weights the step started from")
    at_most("logged_vs_oracle_abs_err", limits["loss_abs"], "logged loss differs from the fp32 loss")
    if grad:
        cos = value("grad_slice_cosine")
        if cos is None or cos < limits["cos_min"]:
            why.append(f"gradient direction: cosine {report.get('grad_slice_cosine')} with the recomputed OPD gradient "
                       f"(limit {limits['cos_min']})")
        within("grad_slice_norm_ratio", limits["slice_ratio"], "gradient magnitude: embedding-row gradient ratio")
        within("grad_norm_ratio", limits["norm_ratio"],
               f"gradient norm (reported {report.get('candidate_grad_norm')}, recomputed {report.get('oracle_grad_norm')})")
    return not why, why


EVAL_DP = 2  # one engine per GPU of the pair, rows split by index (same responses as one engine, see eval_math.py)


def eval_math_cmd(model, tokenizer, tests, max_new_tokens, repeats=EVAL_REPEATS, dp=EVAL_DP):
    """eval_math.py on `tests` ({name: jsonl of math rows}) with the metric-v2 protocol: one JSON whose sets[name]
    holds the mean `accuracy` and `accuracy_per_repeat` (one accuracy per seeded pass)."""
    sets = [x for name, path in tests.items() for x in ("--test", f"{name}={path}")]
    return [PY, tool("eval_math.py"), "--model", str(model), "--tokenizer", str(tokenizer), *sets,
            "--max-new-tokens", str(int(max_new_tokens)), *EVAL_SAMPLING, "--repeats", str(repeats), "--dp", str(dp)]


def metric_v2(final_per_repeat, base_per_repeat, T, gate_min, scale_seconds=3600, unit="accuracy points"):
    """Score of a candidate whose correctness checks all passed (metric v2; the Judge and perf use it).

    `final_per_repeat` / `base_per_repeat`: the final and the original student's accuracy per
    repeat, in points (x100), under the same evaluation protocol. quality_score = mean(final) -
    mean(base); the quality gate passes iff quality_score >= gate_min, and then score =
    quality_score * scale_seconds / T (points gained per hour of timed training), else 0 whatever T
    is. `score` is None when the gate passes but T is not a positive finite number of seconds."""
    final = [float(x) for x in final_per_repeat]
    base = [float(x) for x in base_per_repeat]
    if not final or not base or not all(map(math.isfinite, final + base)):
        raise ValueError(f"quality values must be finite and non-empty: final {final}, base {base}")
    final_q, base_q = sum(final) / len(final), sum(base) / len(base)
    q = final_q - base_q
    passed = q >= gate_min
    T = float(T) if isinstance(T, (int, float)) and not isinstance(T, bool) else None
    usable = T is not None and math.isfinite(T) and T > 0
    score = (q * scale_seconds / T if usable else None) if passed else 0.0
    if not passed:
        summary = f"quality gate failed: quality {q:+.3f} {unit} < {gate_min:g} {unit}: score 0"
    elif not usable:
        summary = f"no usable timed window T ({T}): not scored"
    else:
        summary = f"score {score:.3f} = quality {q:+.3f} {unit} x {scale_seconds:g} / T {T:.1f} s"
    return {"quality_repeats": len(final), "final_quality_per_repeat": final, "final_quality": final_q,
            "base_quality_per_repeat": base, "base_quality": base_q, "quality_score": q, "quality_unit": unit,
            "quality_gate_min": gate_min, "quality_gate_passed": passed, "T_seconds": T, "score": score,
            "score_formula": f"quality_score * {scale_seconds:g} / T if correct and quality_score >= {float(gate_min)!r} else 0",
            "summary": summary}


def run_parallel(jobs, devices=("0", "1")):
    """Run [(name, cmd, log)] on the GPUs, one job per GPU at a time; {name: (code, json)}."""
    pending, running, out = list(jobs), {}, {}
    while pending or running:
        for dev in devices:
            if dev not in running and pending:
                name, cmd, log = pending.pop(0)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=dev)
                fh = open(log, "w")
                running[dev] = (name, subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env), fh, log)
        for dev, (name, proc, fh, log) in list(running.items()):
            if proc.poll() is not None:
                fh.close()
                out[name] = (proc.returncode, json_tail(Path(log).read_text()))
                del running[dev]
        time.sleep(2)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("steps")
    s.add_argument("--out-dir", required=True, help="run output directory (with steps/)")
    s.add_argument("--rounds", type=int, required=True)
    s.add_argument("--kl-coef", type=float, default=1.0)
    a = ap.parse_args()
    res = steps_self_consistency(a.out_dir, a.rounds, a.kl_coef)
    print(json.dumps(res, indent=2))
    raise SystemExit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
