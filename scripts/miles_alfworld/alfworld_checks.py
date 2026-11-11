"""Shared plumbing for the Miles ALFWorld public runner and Judge (no candidate code is imported), and the
task's scoring arithmetic (metric_v2)."""

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
POLICY = "/models/qwen3-4b/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
PY = sys.executable


def load_contract(path, **overrides):
    c = json.loads(Path(path).read_text())
    c.update(overrides)
    return c


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


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


def step_glob(out_dir, k):
    return str(Path(out_dir) / "steps" / f"step_{k:03d}_rank_*.pt")


def steps_self_consistency(out_dir, rounds, eps_clip, eps_clip_high):
    """Per step: logged loss == fp64 GRPO loss recomputed from the dumped tensors."""
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
        num = den = scale = 0.0
        logged = None
        for f in files:
            r = torch.load(f, map_location="cpu", weights_only=False)
            logged = r.get("loss") if logged is None else logged
            keys = ("tokens", "response_lengths", "loss_masks", "engine_logprobs", "train_logprobs", "advantages")
            if any(k not in r for k in keys):
                issues.append(f"step {s}: dump misses {[k for k in keys if k not in r]}")
                continue
            for tok, R, m_, old, lp, adv in zip(*(r[k] for k in keys)):
                R = int(R)
                if m_.numel() != R or old.numel() != R or lp.numel() != R or tok.numel() < R:
                    issues.append(f"step {s}: tensor lengths do not match response length {R}")
                    break
                m_, old, lp, adv = m_.double(), old.double(), lp.double(), float(adv)
                ratio = torch.exp(lp - old)
                pg = torch.maximum(-adv * ratio, -adv * ratio.clamp(1 - eps_clip, 1 + eps_clip_high))
                num += float((pg * m_).sum())
                den += float(m_.sum())
                scale += abs(adv) * float(m_.sum())
        ref = num / max(den, 1.0)
        tol = 1e-3 * max(scale / max(den, 1.0), 1e-3)
        ok = isinstance(logged, (int, float)) and math.isfinite(logged) and abs(logged - ref) <= tol
        per_step[s] = {"logged": logged, "recomputed": ref, "tolerance": tol}
        if not ok:
            issues.append(f"step {s}: logged loss {logged} != {ref:.6g} recomputed from the dumped tensors (tol {tol:.2g})")
    return {"passed": not issues, "issues": issues[:20], "per_step": per_step}


# Thresholds for the fp32 oracle; set from the reference (qualification.json may override).
ORACLE_LIMITS = {"logprob_mean_abs_err": 0.02, "loss_abs_err_rel_adv": 0.02}
GRAD_LIMITS = {"cos_min": 0.9, "slice_ratio": [0.75, 1.33], "norm_ratio": [0.75, 1.33]}


def oracle_verdict(res, limits=None, grad=None):
    """(passed, reasons) for an alfworld_oracle.py report."""
    if not res:
        return False, ["oracle produced no report"]
    lim = {**ORACLE_LIMITS, **(limits or {})}
    why = []
    if (res.get("train_logprob_mean_abs_err") or 1) > lim["logprob_mean_abs_err"]:
        why.append(f"trainer log-probs off by {res.get('train_logprob_mean_abs_err')} (mean abs) vs the fp32 policy")
    tol = lim["loss_abs_err_rel_adv"] * max(res.get("mean_abs_advantage") or 0.0, 1e-3)
    if res.get("logged_vs_oracle_abs_err") is None or res["logged_vs_oracle_abs_err"] > tol:
        why.append(f"logged loss {res.get('logged_loss')} vs fp32 oracle {res.get('loss_oracle')} (tol {tol:.3g})")
    if grad is not None:
        g = {**GRAD_LIMITS, **grad}
        cos, sr, nr = res.get("grad_slice_cosine"), res.get("grad_slice_norm_ratio"), res.get("grad_norm_ratio")
        if cos is None or cos < g["cos_min"]:
            why.append(f"gradient direction: cosine {cos} with the recomputed GRPO gradient")
        if sr is None or not g["slice_ratio"][0] <= sr <= g["slice_ratio"][1]:
            why.append(f"gradient magnitude: embedding-row gradient ratio {sr}")
        if nr is None or not g["norm_ratio"][0] <= nr <= g["norm_ratio"][1]:
            why.append(f"gradient norm ratio {nr} (reported {res.get('candidate_grad_norm')}, recomputed {res.get('oracle_grad_norm')})")
    return not why, why


def metric_v2(final_per_repeat, base_per_repeat, T, gate_min, scale_seconds=3600, unit="success-rate points"):
    """Score of a candidate whose correctness checks all passed (metric v2; the Judge and perf use it).

    `final_per_repeat` / `base_per_repeat`: the final and the original policy's greedy success per
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
    T = float(T) if isinstance(T, (int, float)) else None
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
    return code, json_tail(Path(log).read_text())
