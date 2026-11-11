"""Shared plumbing for the Miles OPCD v3 public runner and Judge (no candidate code is imported), and the
task's scoring arithmetic and quality-evaluation protocol (metric_v2, quality_eval_cmd).

v3: the teacher is the frozen initial student snapshot (Qwen3-1.7B) conditioned on the context, as in the
OPCD paper's system-prompt distillation (same-size frozen teacher); every record the candidate
writes is read with torch.load(weights_only=True); the paper-protocol diagnostic (DIAG_PROTOCOL) is
reported, never scored."""

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
STUDENT = "/models/qwen3-1.7b/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
# v3: the teacher is the frozen initial student itself (same snapshot), conditioned on the context
TEACHER = STUDENT
CONTEXT = "/data/public/context.txt"
PY = sys.executable


def load_contract(path, **overrides):
    c = json.loads(Path(path).read_text())
    c.update(overrides)
    return c


def write_json(path, value, allow_nan=False):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=allow_nan) + "\n")


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


def steps_self_consistency(out_dir, rounds, top_k):
    """Per step: dumped logged loss == fp64 recomputation from the dumped tensors; K and spans."""
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
        num = den = 0.0
        logged = None
        for f in files:
            r = torch.load(f, map_location="cpu", weights_only=True)
            logged = r.get("loss") if logged is None else logged
            for key in ("tokens", "response_lengths", "loss_masks", "opcd_topk_ids", "opcd_old_topk_logprobs",
                        "opcd_teacher_topk_logprobs"):
                if key not in r:
                    issues.append(f"step {s}: dump misses {key}")
                    break
            else:
                for ids, lp, lq, m_, R in zip(r["opcd_topk_ids"], r["opcd_old_topk_logprobs"],
                                              r["opcd_teacher_topk_logprobs"], r["loss_masks"], r["response_lengths"]):
                    if tuple(ids.shape) != (R, top_k) or lp.shape != ids.shape or lq.shape != ids.shape or m_.numel() != R:
                        issues.append(f"step {s}: tensor shapes do not match response length {R} and K={top_k}")
                        break
                    lp, lq, m_ = lp.double(), lq.double(), m_.double()
                    num += float(((lp.exp() * (lp - lq)).sum(-1) * m_).sum())
                    den += max(float(m_.sum()), 1.0)
        ref = num / max(den, 1.0)
        ok = isinstance(logged, (int, float)) and math.isfinite(logged) and abs(logged - ref) <= 1e-3 * max(abs(ref), 1e-12)
        per_step[s] = {"logged": logged, "recomputed": ref}
        if not ok:
            issues.append(f"step {s}: logged loss {logged} != {ref:.6f} recomputed from the dumped tensors")
    return {"passed": not issues, "issues": issues[:20], "per_step": per_step}


# ---- verdicts, shared by run_public_v3.py and run_judge_v3.py ----------------------
# step-0 thresholds also apply to the random steps recomputed from the served-weight copies; the gradient
# thresholds are set from the reference (qualification.json). v3 tightens the teacher error from v2's 0.1 to
# 0.05 (student-probability-weighted |log q| error against the fp32 frozen teacher with c): the bf16 reference
# teacher measures 0.015-0.019 at every step, a teacher that follows the student 0.056 at step 1 and >= 0.13 from
# step 2, and FP8 (e4m3, per-tensor) teacher weights 0.068-0.082 (qualification.json teacher_error_evidence)
# The logged loss must match the fp32 oracle within max(loss_rel x oracle loss, loss_abs): v3's loss is small (about
# 0.19 at step 0, 0.015 by the last steps, against v2's 1.08 -> 0.73) while bf16 noise stays absolute (honest
# |logged - oracle| 0.0002-0.0058 over ~70 v2 oracle runs, 0.0040 on the v3 probe), so v2's pure 1% relative
# tolerance would reject the reference.
ORACLE_LIMITS = {"loss_rel": 0.01, "loss_abs": 0.015, "coverage": 0.999, "teacher_pw_err": 0.05}
# v3: the gradient of a small KL (loss about 0.015 by the last steps) carries relatively more bf16 noise: the reference's
# embedding-slice cosine with the fp32 oracle is 0.997 at step 14 but 0.925 at step 27 (an HF bf16 oracle agrees with
# the fp32 one at only 0.968 there), so v2's 0.95 / [0.8, 1.25] slice bounds reject honest runs; the total-norm bound,
# which the top-1-only control breaks at every step (1.33 / 2.11), stays (qualification.json grad_limits_source)
GRAD_LIMITS = {"cos_min": 0.80, "slice_ratio": [0.7, 1.4], "norm_ratio": [0.8, 1.25]}



def served_steps_passed(served):
    """The served-step recomputations pass unless more than half of the recomputed steps fail. One step can miss the
    gradient limits through bf16 noise alone (the honest reference did once in ten steps on the H100 SXM pair); a
    solution that computes a different loss misses them on every step."""
    failed = sum(not v["passed"] for v in served.values())
    return bool(served) and 2 * failed <= len(served)


def oracle_verdict(res, grad=None, teacher_only=False):
    """(passed, reasons) for an opcd_oracle.py report."""
    if not res:
        return False, ["oracle produced no report"]
    why = []
    if res.get("student_prompts_with_context_leak") != 0:
        why.append("student prompts carry the context")
    if (res.get("candidate_teacher_logq_p_weighted_abs_err") or 1) > ORACLE_LIMITS["teacher_pw_err"]:
        why.append(f"teacher log-probs off by {res.get('candidate_teacher_logq_p_weighted_abs_err')} (p-weighted)")
    if teacher_only:
        return not why, why
    if res.get("logged_vs_oracle_set_abs_err", 1) > max(ORACLE_LIMITS["loss_rel"] * res.get("loss_oracle_set_token_mean", 1),
                                                         ORACLE_LIMITS["loss_abs"]):
        why.append(f"logged loss {res.get('logged_loss')} vs oracle {res.get('loss_oracle_set_token_mean')}")
    if (res.get("candidate_mass_coverage") or 0) < ORACLE_LIMITS["coverage"]:
        why.append(f"top-k set covers {res.get('candidate_mass_coverage')} of the student mass")
    if grad is not None:
        g = {**GRAD_LIMITS, **grad}
        cos, sr, nr = res.get("grad_slice_cosine"), res.get("grad_slice_norm_ratio"), res.get("grad_norm_ratio")
        if cos is None or cos < g["cos_min"]:
            why.append(f"gradient direction: cosine {cos} with the recomputed OPCD gradient")
        if sr is None or not g["slice_ratio"][0] <= sr <= g["slice_ratio"][1]:
            why.append(f"gradient magnitude: embedding-row gradient ratio {sr}")
        if nr is None or not g["norm_ratio"][0] <= nr <= g["norm_ratio"][1]:
            why.append(f"gradient norm ratio {nr} (reported {res.get('candidate_grad_norm')}, recomputed {res.get('oracle_grad_norm')})")
    return not why, why


# ---- metric v2, shared by run_public_v3.py and run_judge_v3.py ----------------------
# the quality protocol (S1; qualification.json metric.quality_protocol records the same values): the training
# sampling contract with per-request seeds and deterministic inference, repeat r seeded with r, one engine
QUALITY_PROTOCOL = {"temperature": 1.0, "top_p": 1.0, "max_new_tokens": 512, "mem_fraction_static": 0.6,
                    "seed_mode": "request", "seeds": [0, 1, 2], "deterministic_inference": True}


def quality_eval_cmd(model, test, repeats):
    """eval_medmcqa.py under QUALITY_PROTOCOL, `repeats` passes (seeds 0..repeats-1) printing one JSON report."""
    p = QUALITY_PROTOCOL
    return [PY, tool("eval_medmcqa.py"), "--model", str(model), "--tokenizer", STUDENT, "--test", str(test),
            "--temperature", repr(p["temperature"]), "--top-p", repr(p["top_p"]),
            "--max-new-tokens", str(p["max_new_tokens"]), "--mem-fraction", repr(p["mem_fraction_static"]),
            "--repeats", str(repeats), "--seed", str(p["seeds"][0]), "--seed-mode", p["seed_mode"]]


# report-only diagnostic under the OPCD paper's evaluation protocol (microsoft/LMOps opcd/scripts/sys_eval.sh:
# val_kwargs temperature 1.0, top_p 0.95, top_k 20, n 1, 512 response tokens) on the paper's 500-question test set
DIAG_PROTOCOL = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "max_new_tokens": 512, "mem_fraction_static": 0.6,
                 "seed_mode": "request", "seeds": [0], "deterministic_inference": True}


def diagnostic_eval_cmd(model, test):
    """eval_medmcqa.py under DIAG_PROTOCOL: one seeded pass, never scored."""
    p = DIAG_PROTOCOL
    return [PY, tool("eval_medmcqa.py"), "--model", str(model), "--tokenizer", STUDENT, "--test", str(test),
            "--temperature", repr(p["temperature"]), "--top-p", repr(p["top_p"]), "--top-k", str(p["top_k"]),
            "--max-new-tokens", str(p["max_new_tokens"]), "--mem-fraction", repr(p["mem_fraction_static"]),
            "--repeats", str(len(p["seeds"])), "--seed", str(p["seeds"][0]), "--seed-mode", p["seed_mode"]]


def eval_protocol_issue(ev, repeats):
    """Why eval_medmcqa.py's report `ev` is not `repeats` passes under QUALITY_PROTOCOL; None when it is."""
    p = QUALITY_PROTOCOL
    per = ev.get("per_repeat") or []
    got = {"repeats": len(per), "seeds": [r.get("seed") for r in per], "seed_mode": ev.get("seed_mode"),
           "deterministic_inference": ev.get("deterministic_inference"), "temperature": ev.get("temperature"),
           "top_p": ev.get("top_p"), "top_k": ev.get("top_k", -1), "max_new_tokens": ev.get("max_new_tokens")}
    want = {"repeats": repeats, "seeds": [p["seeds"][0] + i for i in range(repeats)], "top_k": -1,
            **{k: p[k] for k in ("seed_mode", "deterministic_inference", "temperature", "top_p", "max_new_tokens")}}
    diff = {k: [got[k], want[k]] for k in want if got[k] != want[k]}
    return f"evaluation protocol differs (got, want): {diff}" if diff else None


def metric_v2(final_per_repeat, base_per_repeat, T, gate_min, scale_seconds=3600, unit="accuracy points"):
    """Score of a candidate whose correctness checks all passed (metric v2; the Judge and perf use it).

    `final_per_repeat` / `base_per_repeat`: the final and the original student's accuracy per repeat,
    in points (x100), under the same evaluation protocol. quality_score = mean(final) - mean(base);
    the quality gate passes iff quality_score >= gate_min, and then score = quality_score *
    scale_seconds / T (points gained per hour of timed training), else 0 whatever T is. `score` is
    None when the gate passes but T is not a positive finite number of seconds."""
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
    out = json_tail(Path(log).read_text())
    return code, out
