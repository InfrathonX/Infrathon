"""AdamW algebra over every dumped optimizer step (G7-R1 for the real run, G8 for replay). CPU only.

For consecutive dumps k (lora_pre = theta_k, exp_avg/exp_avg_sq = m_k, v_k after the step, adam_step = t):
  g_k   = (m_k - beta1 m_{k-1}) / (1 - beta1)                           (m_{-1} = v_{-1} = 0 at a run's first step)
  v-rec : v_k == beta2 v_{k-1} + (1 - beta2) g_k^2
  update: theta_{k+1} == theta_k (1 - lr wd) - lr / (1 - beta1^t) * m_k / (sqrt(v_k) / sqrt(1 - beta2^t) + eps)
          (torch AdamW, decoupled weight decay; theta_{k+1} is the next dump's lora_pre, or lora_post if present)
  clip  : ||g_k|| == min(1, clip / (grad_norm_preclip + 1e-6)) * grad_norm_preclip
  count : adam_step increases by exactly 1 per dump (no skipped or extra optimizer steps); step 0 starts at t = 1
  init  : theta_0 equals the contract's student_init bytes
  optim : with --optimizer (the contract's optimizer block; the Judge always passes it) every record's lr, betas, eps,
          weight_decay and clip_grad equal the contract's (rel 1e-6): the algebra above uses the settings a record
          reports, so without this a record could claim, e.g., a larger lr for an update the contract does not allow
Errors are reported relative to the size of the quantity being checked (update size, ||v_k||, ||g_k||) so the
tolerances are precision-independent; fp32 rounding of honest fused / foreach AdamW sets them.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os

import torch
from safetensors.torch import load_file


def flat(d, keys):
    return torch.cat([d[k].double().flatten() for k in keys])


def _load(path):
    # mmap: only the tensors a step's check touches are read from disk
    return torch.load(path, weights_only=True, map_location="cpu", mmap=True)


OPTIMIZER_KEYS = ("lr", "betas", "eps", "weight_decay", "clip_grad")


def optimizer_mismatches(rec: dict, optimizer: dict) -> list[str]:
    """The record's optimizer settings that differ from the contract's `optimizer` block (as floats, rel 1e-6)."""
    bad = []
    for name in OPTIMIZER_KEYS:
        want, got = optimizer[name], rec.get(name)
        try:
            if name == "betas":
                ok = len(got) == len(want) and all(math.isclose(float(g), float(w), rel_tol=1e-6) for g, w in zip(got, want))
            else:
                ok = math.isclose(float(got), float(want), rel_tol=1e-6)
        except (TypeError, ValueError):
            ok = False
        if not ok:
            bad.append(f"{name} {got!r} != contract {want!r}"[:200])
    return bad


def _check_step(args):
    """Step k needs m_{k-1}, v_{k-1} (previous dump), its own dump, and theta_{k+1} (next dump or its lora_post)."""
    paths, k, optimizer = args
    torch.set_num_threads(2)
    rec = _load(paths[k])
    keys = sorted(rec["lora_pre"])
    mismatches = optimizer_mismatches(rec, optimizer) if optimizer is not None else []
    b1, b2 = rec["betas"]
    lr, wd, eps, t = rec["lr"], rec["weight_decay"], rec["eps"], rec["adam_step"]
    th, m, v = flat(rec["lora_pre"], keys), flat(rec["exp_avg"], keys), flat(rec["exp_avg_sq"], keys)
    if k == 0:
        m_prev, v_prev = torch.zeros_like(m), torch.zeros_like(v)
    else:
        prev = _load(paths[k - 1])
        m_prev, v_prev = flat(prev["exp_avg"], keys), flat(prev["exp_avg_sq"], keys)
        del prev
    g = (m - b1 * m_prev) / (1 - b1)
    v_pred = b2 * v_prev + (1 - b2) * g * g
    del m_prev, v_prev
    row = {"step": k, "v_rel": ((v_pred - v).norm() / v.norm().clamp_min(1e-300)).item()}
    if mismatches:
        row["optimizer_mismatch"] = mismatches
    gn = rec["grad_norm_preclip"]
    coef = min(1.0, rec["clip_grad"] / (gn + 1e-6))
    row["clip_rel"] = abs(g.norm().item() - coef * gn) / max(coef * gn, 1e-300)
    nxt = _load(paths[k + 1])["lora_pre"] if k + 1 < len(paths) else rec.get("lora_post")
    if nxt is not None:
        th_next = flat(nxt, keys)
        step_size = lr / (1 - b1 ** t)
        denom = v.sqrt() / math.sqrt(1 - b2 ** t) + eps
        th_pred = th * (1 - lr * wd) - step_size * m / denom
        upd = (th_next - th).norm().clamp_min(1e-300)
        row["theta_rel_to_update"] = ((th_pred - th_next).norm() / upd).item()
        row["update_norm"] = upd.item()
    return row, t, [(key, tuple(rec["lora_pre"][key].shape)) for key in keys]


def check_run(dump_dir: str, student_init: str | None, workers: int = 0, optimizer: dict | None = None) -> dict:
    files = sorted(f for f in os.listdir(dump_dir) if f.startswith("step_") and f.endswith(".pt"))
    paths = [os.path.join(dump_dir, f) for f in files]
    out = {"steps": len(paths), "per_step": [], "violations": []}
    if not paths:
        return out
    workers = workers or min(16, max(1, (os.cpu_count() or 2) // 4), len(paths))
    jobs = [(paths, k, optimizer) for k in range(len(paths))]
    if workers > 1:
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            results = pool.map(_check_step, jobs, chunksize=1)
    else:
        results = [_check_step(j) for j in jobs]
    shapes = results[0][2]
    keys = [k for k, _ in shapes]
    if student_init is not None:
        init = load_file(student_init)
        first = _load(paths[0])["lora_pre"]
        same = set(init) == set(keys) and all(torch.equal(init[k].float(), first[k].float()) for k in keys)
        out["init_matches_student_init"] = same
        if not same:
            out["violations"].append("theta_0 != student_init")
    for k, (row, t, ks) in enumerate(results):
        if ks != shapes:
            out["violations"].append(f"step {k}: LoRA keys or shapes differ from step 0")
        if t != k + 1:
            out["violations"].append(f"step {k}: adam_step {t} != {k + 1}")
        out["violations"] += [f"step {k}: {m}" for m in row.get("optimizer_mismatch", [])]
        out["per_step"].append(row)
    if optimizer is not None:
        out["optimizer_matches_contract"] = not any("optimizer_mismatch" in r for r in out["per_step"])
    for name in ("v_rel", "clip_rel", "theta_rel_to_update"):
        # NaN compares false, so max() would silently skip a step whose error is NaN
        vals = [r[name] for r in out["per_step"] if name in r]
        bad = [r["step"] for r in out["per_step"] if name in r and not math.isfinite(r[name])]
        if bad:
            out["violations"].append(f"steps {bad[:10]}: {name} is not finite")
        finite = [v for v in vals if math.isfinite(v)]
        out[f"max_{name}"] = max(finite) if finite else None
    return out


def final_lora_issues(final: str, dump_dir: str) -> list[str]:
    """The run's final_lora.safetensors must be the student after its last optimizer step bit for bit: the last dump's
    `lora_post` (the theta check_run verifies that step's AdamW update against), the same keys, fp32. A LoRA trained or
    edited after the last step fails. That lora_post must hold exactly the trained tensors, the keys and shapes of the
    same record's lora_pre (check_run ties those to every step and to student_init; it only reads lora_post at these
    keys, so an extra adapter tensor there would reach the quality evaluation unchecked). Only this record is read,
    with torch.load(weights_only=True). [] when it holds."""
    files = sorted(f for f in os.listdir(dump_dir) if f.startswith("step_") and f.endswith(".pt")) \
        if os.path.isdir(dump_dir) else []
    if not files:
        return ["no step records"]
    if os.path.islink(final) or not os.path.isfile(final):
        return ["final_lora.safetensors is not a regular file"]
    try:
        rec = torch.load(os.path.join(dump_dir, files[-1]), weights_only=True, map_location="cpu", mmap=True)
    except Exception as e:  # noqa: BLE001 - any unreadable record is a finding, not a Judge error
        return [f"{files[-1]}: not readable as plain tensors ({type(e).__name__})"]
    post = rec.get("lora_post") if isinstance(rec, dict) else None
    pre = rec.get("lora_pre") if isinstance(rec, dict) else None
    if not isinstance(post, dict) or not post:
        return [f"{files[-1]} (the last step) has no lora_post"]
    if not isinstance(pre, dict) or not all(isinstance(k, str) and isinstance(v, torch.Tensor)
                                            for d in (pre, post) for k, v in d.items()):
        return [f"{files[-1]}: lora_pre / lora_post are not maps of names to tensors"]
    issues = [f"{k}: in the last lora_post, not a trained tensor (not in that step's lora_pre)" for k in sorted(set(post) - set(pre))]
    issues += [f"{k}: in the last step's lora_pre, missing from its lora_post" for k in sorted(set(pre) - set(post))]
    issues += [f"{k}: lora_post shape {tuple(post[k].shape)} != lora_pre shape {tuple(pre[k].shape)}"
               for k in sorted(set(pre) & set(post)) if post[k].shape != pre[k].shape]
    if issues:
        return issues[:20]
    try:
        lora = load_file(final)
    except Exception as e:  # noqa: BLE001
        return [f"final_lora.safetensors: not readable ({type(e).__name__})"]
    issues = [f"{k}: in the last lora_post, missing from final_lora" for k in sorted(set(post) - set(lora))]
    issues += [f"{k}: in final_lora, not in the last lora_post" for k in sorted(set(lora) - set(post))]
    for k in sorted(set(post) & set(lora)):
        a, b = lora[k], post[k]
        if a.dtype != torch.float32:
            issues.append(f"{k}: final_lora dtype {a.dtype}, not float32")
        elif a.shape != b.shape or not torch.equal(
                a.contiguous().view(torch.int32), b.to(torch.float32).contiguous().view(torch.int32)):
            issues.append(f"{k}: differs from the last lora_post")
    return issues[:20]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--student-init")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=0, help="0 = min(16, cpus / 4)")
    ap.add_argument("--optimizer", type=json.loads, help="the contract's optimizer block (JSON): every record must match it")
    args = ap.parse_args()
    res = check_run(args.dump, args.student_init, args.workers, args.optimizer)
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_step"}, indent=1))


if __name__ == "__main__":
    main()
