#!/usr/bin/env python3
"""Independent recomputation of one OPD training step with plain HF fp32 models.

Reads the candidate's step dump for one step (all ranks: tokens, response_lengths,
loss_masks, old_logprobs, teacher_logprobs, advantages, loss) and re-derives, with no
Miles code:

* ``log p_old``: the sampled tokens' log-probabilities under the weights the step started
  from, i.e. the initial student for step 0 or, with ``--student-snapshot``, a copy of the
  weights the student SGLang engine served before the publication that started the step
  (``opd_provenance_hook``). Compared with the dumped ``old_logprobs`` this checks that the
  candidate scored the tokens with the pre-update weights of this step;
* ``log q``: the frozen teacher's log-probabilities of the same tokens, compared with the
  dumped ``teacher_logprobs``;
* with ``--grad-records``: the gradient of the specified objective at those weights,
  ``(1/N) sum_i (1/|R_i|) sum_t (-A_t) grad log p_t`` with the candidate's own advantages
  (ratio 1 at the start of the step, so the clip is inactive), over every sequence of the
  step. Its first ``FP_ROWS`` rows of the input embedding are compared with the reduced
  gradient Megatron held just before its optimizer step, and its total norm with the grad
  norm Megatron reported.

``--sample-rows`` recomputes only random rows (log-prob checks); ``--teacher-only`` skips
the student. Metrics only; the Judge applies the limits (``opd_checks.step_oracle_verdict``).
"""

import argparse
import glob
import json
import math
import random
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent / "audit"))
from opd_provenance_hook import FP_ROWS  # noqa: E402  (constants only; the hook is inert here)

KEYS = ("tokens", "response_lengths", "loss_masks", "old_logprobs", "teacher_logprobs", "advantages")


def load_dump(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise AssertionError(f"no dump matches {pattern}")
    recs = [torch.load(f, map_location="cpu", weights_only=True) for f in files]
    missing = sorted({k for r in recs for k in KEYS if k not in r})
    if missing:
        raise AssertionError(f"step dump is missing required keys {missing}")
    d = {k: [x for r in recs for x in r[k]] for k in KEYS}
    d["loss"] = recs[0].get("loss")
    d["steps"] = sorted({r.get("step") for r in recs})
    return d


def load_sglang_snapshot(model, path):
    """Load SGLang's served Qwen3 weights (fused qkv_proj / gate_up_proj) into an HF model."""
    from safetensors.torch import load_file

    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    q, kv = cfg.num_attention_heads * head_dim, cfg.num_key_value_heads * head_dim
    sd = {}
    for name, t in load_file(path).items():
        t = t.to(torch.float32)
        if name.endswith(".qkv_proj.weight"):
            base = name[: -len("qkv_proj.weight")]
            sd[base + "q_proj.weight"], sd[base + "k_proj.weight"], sd[base + "v_proj.weight"] = t.split([q, kv, kv], 0)
        elif name.endswith(".gate_up_proj.weight"):
            base = name[: -len("gate_up_proj.weight")]
            sd[base + "gate_proj.weight"], sd[base + "up_proj.weight"] = t.chunk(2, 0)
        else:
            sd[name] = t
    want = model.state_dict()
    tied = {"lm_head.weight"} if getattr(cfg, "tie_word_embeddings", False) else set()
    missing = sorted(set(want) - set(sd) - tied)
    unexpected = sorted(set(sd) - set(want))
    bad_shape = sorted(k for k in set(sd) & set(want) if tuple(sd[k].shape) != tuple(want[k].shape))
    if missing or unexpected or bad_shape:
        raise AssertionError(f"snapshot does not map onto the pinned model: missing {missing[:4]}, "
                             f"unexpected {unexpected[:4]}, shape {bad_shape[:4]}")
    model.load_state_dict(sd, strict=False)
    return model


def load_grad_records(paths, rows, width):
    """Stitch the per-rank gradient records into [rows*width] values plus a validity mask."""
    vals = torch.zeros(rows * width, dtype=torch.float64)
    valid = torch.zeros(rows * width, dtype=torch.bool)
    for path in paths:
        r = torch.load(path, map_location="cpu", weights_only=True)
        flat = r["rows"].to(torch.float64).reshape(-1)[: rows * width]
        lo, hi = (0, flat.numel()) if r.get("owned") is None else (max(0, r["owned"][0]), min(flat.numel(), r["owned"][1]))
        if hi > lo:
            fresh = ~valid[lo:hi]
            vals[lo:hi][fresh] = flat[lo:hi][fresh]
            valid[lo:hi] = True
    return vals, valid


def sampled_logprobs(model, seq, P, R, device):
    """fp32 log-probabilities of seq[P:P+R] given their prefixes (grad follows the model's mode)."""
    x = seq.unsqueeze(0).to(device)
    logits = model(input_ids=x, use_cache=False).logits[0, P - 1 : P + R - 1].to(torch.float32)
    return torch.log_softmax(logits, -1).gather(-1, x[0, P : P + R].unsqueeze(-1)).squeeze(-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dumps", required=True, help="glob of one step's dump files (all ranks)")
    ap.add_argument("--student", required=True, help="HF dir of the pinned student (config; weights unless --student-snapshot)")
    ap.add_argument("--student-snapshot", default=None, help="served-weight copy the step started from")
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--kl-coef", type=float, default=1.0)
    ap.add_argument("--grad-records", default=None, help="comma-separated prov-grad-*.pt files of this step (all ranks)")
    ap.add_argument("--grad-norm", type=float, default=None, help="grad norm Megatron reported for this step")
    ap.add_argument("--teacher-only", action="store_true")
    ap.add_argument("--sample-rows", type=int, default=None, help="recompute only this many random rows")
    ap.add_argument("--row-seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    grad_mode = a.grad_records is not None
    if grad_mode and (a.sample_rows or a.teacher_only):
        raise AssertionError("the gradient check needs every row of the step and the student model")
    dev = torch.device(a.device)

    d = load_dump(a.dumps)
    n = len(d["tokens"])
    rows = list(range(n))
    if a.sample_rows:
        rows = sorted(random.Random(a.row_seed).sample(rows, min(a.sample_rows, n)))

    student = None
    if not a.teacher_only:
        student = AutoModelForCausalLM.from_pretrained(a.student, dtype=torch.float32)
        if a.student_snapshot:
            load_sglang_snapshot(student, a.student_snapshot)
        student = student.to(dev).eval()
        student.requires_grad_(grad_mode)
        if grad_mode:
            # v2: responses reach 4096 tokens; an fp32 backward that keeps every layer's activations does not fit a
            # 94 GB card past ~4.4k tokens. Recompute each decoder layer in the backward instead (same fp32 ops;
            # HF only checkpoints in train mode, which changes nothing else here: Qwen3 has no dropout).
            student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            student.train()
    teacher = AutoModelForCausalLM.from_pretrained(a.teacher, dtype=torch.float32).to(dev).eval()

    s_err = t_err = s_max = t_max = n_tok = 0.0
    oracle_sum = 0.0
    for i in rows:
        seq = d["tokens"][i].to(torch.long)
        R = int(d["response_lengths"][i])
        P = seq.numel() - R
        m = d["loss_masks"][i].to(dev, torch.float64)
        den = m.sum().clamp_min(1)
        with torch.no_grad():
            lq = sampled_logprobs(teacher, seq, P, R, dev).double()
        e = (d["teacher_logprobs"][i].to(dev, torch.float64) - lq).abs() * m
        t_err += float(e.sum()); t_max = max(t_max, float(e.max()) if R else 0.0)
        n_tok += float(m.sum())
        if student is None:
            continue
        if grad_mode:
            lp_g = sampled_logprobs(student, seq, P, R, dev)
            adv = d["advantages"][i].to(dev, torch.float64)
            ((-adv * lp_g.double() * m).sum() / den / n).float().backward()
            lp = lp_g.detach().double()
            del lp_g
        else:
            with torch.no_grad():
                lp = sampled_logprobs(student, seq, P, R, dev).double()
        e = (d["old_logprobs"][i].to(dev, torch.float64) - lp).abs() * m
        s_err += float(e.sum()); s_max = max(s_max, float(e.max()) if R else 0.0)
        oracle_sum += float(((lp - lq) * m).sum() / den)

    report = {"dumps": a.dumps, "steps": d["steps"], "rows": len(rows), "sequences": n, "valid_tokens": n_tok,
              "student_weights": a.student_snapshot or a.student, "teacher_only": a.teacher_only,
              "teacher_lp_mean_err": t_err / max(n_tok, 1), "teacher_lp_max_err": t_max,
              "logged_loss": d["loss"]}
    if student is not None:
        report.update(student_lp_mean_err=s_err / max(n_tok, 1), student_lp_max_err=s_max,
                      oracle_loss=oracle_sum / len(rows) if rows else None)
        if d["loss"] is not None and not a.sample_rows:
            report["logged_vs_oracle_abs_err"] = abs(d["loss"] - report["oracle_loss"])
    if grad_mode:
        emb = student.get_input_embeddings().weight
        width = emb.shape[1]
        oracle = emb.grad.detach()[:FP_ROWS].to(torch.float64).reshape(-1).cpu()
        total = math.sqrt(sum(float(p.grad.detach().double().pow(2).sum()) for p in student.parameters() if p.grad is not None))
        vals, valid = load_grad_records([x for x in a.grad_records.split(",") if x], FP_ROWS, width)
        x, y = vals[valid], oracle[valid]
        report.update(
            grad_valid_elements=int(valid.sum()),
            grad_slice_cosine=float((x @ y) / max(float(x.norm() * y.norm()), 1e-30)) if x.numel() else None,
            grad_slice_norm_ratio=float(x.norm() / max(float(y.norm()), 1e-30)) if x.numel() else None,
            oracle_grad_norm=total, candidate_grad_norm=a.grad_norm,
            grad_norm_ratio=(a.grad_norm / total) if (a.grad_norm is not None and total > 0) else None)
    text = json.dumps(report, indent=2)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
