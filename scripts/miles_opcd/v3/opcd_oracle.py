#!/usr/bin/env python3
"""Independent OPCD oracle: recompute the top-k reverse-KL loss with plain HF models.

Reads a Miles ``--save-debug-train-data`` dump (tokens, response_lengths,
loss_masks and the candidate's ``opcd_topk_ids`` / ``opcd_teacher_topk_logprobs``),
re-derives everything from scratch with Hugging Face fp32 forwards:

* student prompt must not contain the context prefix; teacher input is
  ``context_prefix_ids + tokens`` where the prefix is derived from the chat template;
* canonical top-k set from the pre-update student's fp32 log-softmax;
* teacher log-probs on that set from the fp32 teacher forward;
* per-token kl_t and the global valid-token mean.

It reports the loss under (a) the oracle's own set and (b) the candidate's
reported set, the overlap between the two sets, the max abs error of the
candidate's teacher log-probs on the candidate's set, and compares (b) with the
loss the candidate logged (``--logged-loss``). No Miles code is imported.

Extra modes (the task's v2 Judge):

* ``--student-snapshot``: the pre-update student is a copy of the weights the SGLang
  engine served (``opcd_provenance_hook``), converted from SGLang's fused Qwen3 layout;
* ``--grad-records`` / ``--grad-norm``: also back-propagate the specified loss (the
  candidate's own top-k ids and teacher log-probs, fp32 student) and compare the
  gradient of the first embedding rows with the reduced gradient Megatron held just
  before its optimizer step, and the full-model gradient norm with the one Megatron
  reported;
* ``--teacher-only``: no student model; recompute the teacher log-probs of a random
  subset of rows (``--sample-rows``, ``--row-seed``) and weight their error by the
  candidate's own student probabilities.
"""

import argparse
import json
import math
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

FP_ROWS = 128  # rows of the input embedding the provenance hook fingerprints and whose gradient it records


def context_prefix_ids(tok, context: str, template_kwargs: dict) -> list[int]:
    kw = dict(tokenize=False, add_generation_prompt=True, **template_kwargs)
    probe = [{"role": "user", "content": "probe"}]
    s = tok.apply_chat_template(probe, **kw)
    t = tok.apply_chat_template([{"role": "system", "content": context}, *probe], **kw)
    assert t.endswith(s), "context is not a string prefix"
    s_ids, t_ids = tok.encode(s, add_special_tokens=False), tok.encode(t, add_special_tokens=False)
    cut = len(t_ids) - len(s_ids)
    assert cut > 0 and t_ids[cut:] == s_ids, "context is not a token prefix"
    return t_ids[:cut]


def response_logprobs_grad(model, seq: torch.Tensor, start: int, end: int, device) -> torch.Tensor:
    """fp32 log-softmax rows predicting seq[start+1 : end+1] (i.e. logits at [start, end))."""
    logits = model(input_ids=seq.unsqueeze(0).to(device), use_cache=False).logits[0, start:end].to(torch.float32)
    return torch.log_softmax(logits, dim=-1)


response_logprobs = torch.no_grad()(response_logprobs_grad)


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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", required=True, help="rollout_{id}_rank_{r}.pt written by --save-debug-train-data")
    ap.add_argument("--student", required=True, help="HF dir of the *pre-update* student for this rollout")
    ap.add_argument("--teacher", required=True, help="HF dir of the frozen teacher")
    ap.add_argument("--context-file", required=True)
    ap.add_argument("--template-kwargs", default='{"enable_thinking": false}')
    ap.add_argument("--top-k", type=int, default=256)
    ap.add_argument("--renorm-topk", action="store_true")
    ap.add_argument("--logged-loss", type=float, default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-models", action="store_true", help="skip the HF forwards: only the fp64 self-consistency and shape checks (CPU, seconds)")
    ap.add_argument("--allow-missing", action="store_true", help="debugging only: do not fail on an incomplete dump")
    ap.add_argument("--student-snapshot", default=None, help="SGLang served-weight copy of the pre-update student; "
                    "--student then only supplies config and tokenizer")
    ap.add_argument("--grad-records", default=None, help="comma-separated prov-grad-*.pt files of this step (all ranks)")
    ap.add_argument("--grad-norm", type=float, default=None, help="grad norm Megatron reported for this step")
    ap.add_argument("--teacher-only", action="store_true", help="skip the student; check teacher log-probs only")
    ap.add_argument("--sample-rows", type=int, default=None, help="with --teacher-only: rows drawn at random")
    ap.add_argument("--row-seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cpu" if args.no_models else args.device)
    import glob as _glob

    files = sorted(_glob.glob(args.dump))
    if not files:
        raise AssertionError(f"no dump matches {args.dump}")
    recs = [torch.load(f, map_location="cpu", weights_only=True) for f in files]
    if all("rollout_data" in r for r in recs):
        recs = [r["rollout_data"] for r in recs]
    rd = {}
    for r in recs:
        for k, v in r.items():
            if isinstance(v, list):
                rd.setdefault(k, []).extend(v)
    if args.logged_loss is None and isinstance(recs[0].get("loss"), (int, float)):
        args.logged_loss = float(recs[0]["loss"])
    tokens = [t.to(torch.long) for t in rd["tokens"]]
    response_lengths = list(rd["response_lengths"])
    loss_masks = [m.to(torch.float32) for m in rd["loss_masks"]]
    cand_ids = rd.get("opcd_topk_ids")
    cand_logq = rd.get("opcd_teacher_topk_logprobs")
    cand_old = rd.get("opcd_old_topk_logprobs")
    cand_loss = rd.get("opcd_logged_loss")
    if not args.allow_missing:
        # Fail closed: the dump contract is part of the task, so a renamed or absent
        # key is a rule violation, not a reason to silently skip every teacher gate.
        missing = [k for k, v in (("opcd_topk_ids", cand_ids), ("opcd_teacher_topk_logprobs", cand_logq)) if v is None]
        if missing:
            raise AssertionError(f"dump is missing required OPCD keys: {missing}; keys present: {sorted(rd)}")
    n = len(tokens) if args.max_samples is None else min(args.max_samples, len(tokens))
    order = list(range(n))
    if args.teacher_only and args.sample_rows:
        import random

        order = sorted(random.Random(args.row_seed).sample(range(len(tokens)), min(args.sample_rows, len(tokens))))
    grad_mode = args.grad_records is not None
    if grad_mode and (args.max_samples is not None or args.teacher_only):
        raise AssertionError("the gradient check needs every row of the step and the student model")
    if args.teacher_only and cand_old is None:
        raise AssertionError("--teacher-only weights errors by the dumped opcd_old_topk_logprobs, which are missing")

    tok = AutoTokenizer.from_pretrained(args.student)
    context = open(args.context_file, encoding="utf-8").read().strip()
    prefix = torch.tensor(context_prefix_ids(tok, context, json.loads(args.template_kwargs)), dtype=torch.long)
    C = prefix.numel()

    student = teacher = None
    if not args.no_models:
        if not args.teacher_only:
            student = AutoModelForCausalLM.from_pretrained(args.student, dtype=torch.float32)
            if args.student_snapshot:
                load_sglang_snapshot(student, args.student_snapshot)
            student = student.to(device).eval()
            student.requires_grad_(grad_mode)
        teacher = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=torch.float32).to(device).eval()
    n_valid = float(sum(float(loss_masks[i].sum()) for i in order))

    tot_oracle = tot_cand_set = tot_tokens = 0.0
    leak = 0
    overlap_sum = overlap_n = 0
    max_logq_err = 0.0
    weighted_err_sum = weighted_err_norm = 0.0
    plausible_max_err = 0.0
    coverage_sum = self_sum = 0.0
    shape_violations = []
    per_sample = []
    for i in order:
        seq, R = tokens[i], int(response_lengths[i])
        T = seq.numel()
        P = T - R
        if seq[:C].tolist() == prefix.tolist():
            leak += 1  # student prompt already carries the context

        mask = loss_masks[i].to(device)
        if args.no_models:
            tot_tokens += max(float(mask.sum()), 1.0)
            if cand_ids is not None and cand_ids[i].shape != (R, args.top_k):
                shape_violations.append({"sample": i, "got": list(cand_ids[i].shape), "expected": [R, args.top_k]})
            if cand_old is not None and cand_logq is not None:
                lp = cand_old[i].to(device=device, dtype=torch.float64)
                lq = cand_logq[i].to(device=device, dtype=torch.float64)
                self_sum += float((((lp.exp() * (lp - lq)).sum(-1)) * mask.to(torch.float64)).sum())
            continue
        tseq = torch.cat([prefix, seq])
        logq_full = response_logprobs(teacher, tseq, C + P - 1, C + T - 1, device)  # [R, V]
        if args.teacher_only:
            ids = cand_ids[i].to(device=device, dtype=torch.long)
            ref = logq_full.gather(-1, ids)
            err = (cand_logq[i].to(device=device, dtype=torch.float32) - ref).abs()
            w = cand_old[i].to(device=device, dtype=torch.float32).exp()
            wsum_i, wnorm_i = float((w * err).sum()), float(w.sum())
            plausible = ref >= math.log(1e-4)
            max_logq_err = max(max_logq_err, float(err.max()))
            weighted_err_sum += wsum_i
            weighted_err_norm += wnorm_i
            plausible_max_err = max(plausible_max_err, float(err[plausible].max()) if plausible.any() else 0.0)
            tot_tokens += max(float(mask.sum()), 1.0)
            per_sample.append({"sample": i, "R": R, "p_weighted_teacher_logq_abs_err": wsum_i / max(wnorm_i, 1e-12)})
            continue
        if grad_mode:
            logp_g = response_logprobs_grad(student, seq, P - 1, T - 1, device)  # [R, V] with grad
            ids_g = cand_ids[i].to(device=device, dtype=torch.long)
            lp_g = logp_g.gather(-1, ids_g)
            lq_g = cand_logq[i].to(device=device, dtype=torch.float32)
            contrib = ((lp_g.exp() * (lp_g - lq_g)).sum(-1) * mask).sum() / max(n_valid, 1.0)
            contrib.backward()
            logp = logp_g.detach()
            del logp_g, lp_g, contrib
        else:
            logp = response_logprobs(student, seq, P - 1, T - 1, device)  # [R, V]

        def kl_on(ids):
            lp = logp.gather(-1, ids)
            lq = logq_full.gather(-1, ids)
            if args.renorm_topk:
                lp = lp - torch.logsumexp(lp, -1, keepdim=True)
                lq = lq - torch.logsumexp(lq, -1, keepdim=True)
            return (lp.exp() * (lp - lq)).sum(-1)

        oracle_ids = torch.topk(logp, args.top_k, dim=-1).indices
        kl_oracle = kl_on(oracle_ids)
        tot_oracle += float((kl_oracle * mask).sum())
        tot_tokens += max(float(mask.sum()), 1.0)
        row = {"sample": i, "R": R, "kl_oracle_mean": float((kl_oracle * mask).sum() / mask.sum().clamp_min(1))}
        if cand_ids is not None:
            ids = cand_ids[i].to(device=device, dtype=torch.long)
            if ids.shape != (R, args.top_k):
                shape_violations.append({"sample": i, "got": list(ids.shape), "expected": [R, args.top_k]})
            kl_cand = kl_on(ids)
            tot_cand_set += float((kl_cand * mask).sum())
            inter = torch.tensor([len(set(a.tolist()) & set(b.tolist())) for a, b in zip(ids, oracle_ids)], dtype=torch.float32)
            overlap_sum += float(inter.sum())
            # denominator is the number of ids the candidate SHOULD have reported, so a
            # shrunken set scores low instead of trivially scoring 1.0
            overlap_n += R * args.top_k
            # what the loss actually depends on: how much oracle student mass the set covers
            cov = logp.gather(-1, ids).exp().sum(-1)
            coverage_sum += float((cov * mask).sum())
            row["kl_candidate_set_mean"] = float((kl_cand * mask).sum() / mask.sum().clamp_min(1))
            row["set_overlap"] = float(inter.mean() / args.top_k)
            row["mass_coverage"] = float((cov * mask).sum() / mask.sum().clamp_min(1))
            if cand_logq is not None:
                cand = cand_logq[i].to(device=device, dtype=torch.float32)
                ref = logq_full.gather(-1, ids)
                err = (cand - ref).abs()
                max_logq_err = max(max_logq_err, float(err.max()))
                row["max_teacher_logq_abs_err"] = float(err.max())
                # what the loss actually sees: student-probability-weighted error, and the
                # error restricted to teacher-plausible tokens (q >= 1e-4)
                w = logp.gather(-1, ids).exp()
                wsum_i = float((w * err).sum()); wnorm_i = float(w.sum())
                plausible = ref >= math.log(1e-4)
                row["p_weighted_teacher_logq_abs_err"] = wsum_i / max(wnorm_i, 1e-12)
                row["max_teacher_logq_abs_err_q_ge_1e-4"] = float(err[plausible].max()) if plausible.any() else 0.0
                weighted_err_sum += wsum_i; weighted_err_norm += wnorm_i
                plausible_max_err = max(plausible_max_err, row["max_teacher_logq_abs_err_q_ge_1e-4"])
        if cand_old is not None and cand_ids is not None and cand_logq is not None:
            # fp64 identity on the candidate's own numbers: no model, no dtype confound.
            # Reproduces the specified reduction exactly, so a shrunken K, a renormalised
            # KL or a subsampled span shows up here even when the fp32 model comparison cannot see it.
            lp = cand_old[i].to(device=device, dtype=torch.float64)
            lq = cand_logq[i].to(device=device, dtype=torch.float64)
            self_kl = (lp.exp() * (lp - lq)).sum(-1)
            self_sum += float((self_kl * mask.to(torch.float64)).sum())
        per_sample.append(row)

    grad_report = {}
    if grad_mode and student is not None:
        emb = student.get_input_embeddings().weight
        rows, width = FP_ROWS, emb.shape[1]
        oracle = emb.grad.detach()[:rows].to(torch.float64).reshape(-1).cpu()
        total = math.sqrt(sum(float(p.grad.detach().double().pow(2).sum()) for p in student.parameters() if p.grad is not None))
        vals, valid = load_grad_records([x for x in args.grad_records.split(",") if x], rows, width)
        a_, b_ = vals[valid], oracle[valid]
        cos = float((a_ @ b_) / max(float(a_.norm() * b_.norm()), 1e-30)) if a_.numel() else None
        grad_report = {
            "grad_valid_elements": int(valid.sum()),
            "grad_slice_cosine": cos,
            "grad_slice_norm_ratio": float(a_.norm() / max(float(b_.norm()), 1e-30)) if a_.numel() else None,
            "oracle_grad_norm": total,
            "candidate_grad_norm": args.grad_norm,
            "grad_norm_ratio": (args.grad_norm / total) if (args.grad_norm is not None and total > 0) else None,
        }
    report = {
        "dump": args.dump,
        "samples": len(order),
        "valid_tokens": tot_tokens,
        "context_prefix_tokens": C,
        "student_prompts_with_context_leak": leak,
        "loss_oracle_set_token_mean": tot_oracle / max(tot_tokens, 1),
        "loss_candidate_set_token_mean": (tot_cand_set / max(tot_tokens, 1)) if cand_ids is not None else None,
        "candidate_set_overlap_mean": (overlap_sum / overlap_n) if overlap_n else None,
        "candidate_mass_coverage": (coverage_sum / max(tot_tokens, 1)) if cand_ids is not None else None,
        "candidate_topk_shape_violations": shape_violations,
        "loss_self_consistent_token_mean": (self_sum / max(tot_tokens, 1)) if cand_old is not None else None,
        "candidate_teacher_logq_max_abs_err": max_logq_err if cand_logq is not None else None,
        "candidate_teacher_logq_p_weighted_abs_err": (weighted_err_sum / max(weighted_err_norm, 1e-12)) if cand_logq is not None else None,
        "candidate_teacher_logq_max_abs_err_q_ge_1e-4": plausible_max_err if cand_logq is not None else None,
        "logged_loss": args.logged_loss,
        "student_weights": args.student_snapshot or args.student,
        "teacher_only": bool(args.teacher_only),
        **grad_report,
        "per_sample": per_sample,
    }
    if args.logged_loss is not None and not args.teacher_only:
        # always comparable: the oracle-set loss needs no candidate key at all
        report["logged_vs_oracle_set_abs_err"] = abs(args.logged_loss - report["loss_oracle_set_token_mean"])
        if cand_ids is not None:
            report["logged_vs_candidate_set_abs_err"] = abs(args.logged_loss - report["loss_candidate_set_token_mean"])
        if report["loss_self_consistent_token_mean"] is not None:
            ref = report["loss_self_consistent_token_mean"]
            report["logged_vs_self_consistent_rel_err"] = abs(args.logged_loss - ref) / max(abs(ref), 1e-12)
    text = json.dumps({k: v for k, v in report.items() if k != "per_sample"}, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
