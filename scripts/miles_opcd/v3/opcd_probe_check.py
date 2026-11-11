#!/usr/bin/env python3
"""Judge check of the one-step OPCD probe (no candidate code is imported).

The candidate trained exactly one optimizer step, from the pinned initial
weights, on fixed trajectories. Given its step dump and resulting HF weights:

  P1 contract   one step; the trained sequences are exactly the batch; K == top_k
  P2 self       logged loss == fp64 recomputation from the dumped tensors
  P3 oracle     independent fp32 HF student/teacher: canonical top-k mass covered by the
                candidate's ids, loss vs the oracle loss, teacher log-probs on the ids
  P4 update     sign test: with Adam's first step, delta = -lr * g / (|g| + eps) - lr * wd * w,
                so every weight that moved must move against the oracle OPCD gradient.
                An estimator that is not the analytic top-k KL gradient (e.g. a REINFORCE
                advantage) agrees only about half the time.
"""

import argparse
import glob
import json
import math

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer


def context_prefix(tok, context, tk):
    kw = dict(tokenize=False, add_generation_prompt=True, **tk)
    probe = [{"role": "user", "content": "probe"}]
    s = tok.apply_chat_template(probe, **kw)
    t = tok.apply_chat_template([{"role": "system", "content": context}, *probe], **kw)
    si, ti = tok.encode(s, add_special_tokens=False), tok.encode(t, add_special_tokens=False)
    assert t.endswith(s) and ti[len(ti) - len(si):] == si
    return ti[: len(ti) - len(si)]


def load_dump(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise AssertionError(f"no step dump matches {pattern}")
    recs = [torch.load(f, map_location="cpu", weights_only=True) for f in files]
    steps = {r.get("step") for r in recs}
    merged = {"steps": steps, "loss": recs[0].get("loss")}
    for key in ("tokens", "response_lengths", "loss_masks", "opcd_topk_ids", "opcd_old_topk_logprobs", "opcd_teacher_topk_logprobs"):
        merged[key] = [x for r in recs for x in r[key]]
    return merged


def hf_tensors(path):
    out = {}
    for f in sorted(glob.glob(f"{path}/*.safetensors")):
        with safe_open(f, "pt") as h:
            for k in h.keys():
                out[k] = (f, k)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", required=True)
    ap.add_argument("--dumps", required=True, help="glob of the probe's step dump files")
    ap.add_argument("--final-hf", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--context-file", required=True)
    ap.add_argument("--top-k", type=int, default=256)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--template-kwargs", default='{"enable_thinking": false}')
    ap.add_argument("--thresholds", default='{"self_rel": 1e-3, "oracle_rel": 0.01, "oracle_abs": 0.015, "coverage": 0.999, "teacher_pw_err": 0.05, "sign_agree": 0.9, "moved_frac": 0.05}')
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    th = json.loads(args.thresholds)
    dev = torch.device("cuda")
    batch = json.load(open(args.batch))
    lr = float(batch["lr"])
    issues = []
    report = {}

    # ---- P1 contract ----
    d = load_dump(args.dumps)
    if d["steps"] != {0}:
        issues.append(f"P1: expected exactly one step 0, got {sorted(d['steps'])}")
    want = sorted(tuple(s["prompt_ids"]) + tuple(s["response_ids"]) for s in batch["samples"])
    got_seqs = [tuple(int(x) for x in t.tolist()) for t in d["tokens"]]
    if sorted(got_seqs) != want:
        issues.append("P1: trained sequences differ from the probe batch")
    for i, (ids, R) in enumerate(zip(d["opcd_topk_ids"], d["response_lengths"])):
        if tuple(ids.shape) != (R, args.top_k):
            issues.append(f"P1: sample {i} top-k shape {tuple(ids.shape)} != ({R}, {args.top_k})")

    # ---- P2 self-consistency ----
    num = den = 0.0
    for lp, lq, m in zip(d["opcd_old_topk_logprobs"], d["opcd_teacher_topk_logprobs"], d["loss_masks"]):
        lp, lq, m = lp.double(), lq.double(), m.double()
        num += float(((lp.exp() * (lp - lq)).sum(-1) * m).sum())
        den += max(float(m.sum()), 1.0)
    self_loss = num / den
    logged = d["loss"]
    report["logged_loss"], report["self_consistent_loss"] = logged, self_loss
    if logged is None or not math.isfinite(logged) or abs(logged - self_loss) / max(abs(self_loss), 1e-12) > th["self_rel"]:
        issues.append(f"P2: logged loss {logged} vs recomputed {self_loss}")

    # ---- P3 oracle + gradient ----
    tok = AutoTokenizer.from_pretrained(args.student)
    ctx = torch.tensor(context_prefix(tok, open(args.context_file).read().strip(), json.loads(args.template_kwargs)))
    student = AutoModelForCausalLM.from_pretrained(args.student, dtype=torch.float32).to(dev)
    student.train()
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=torch.float32).to(dev).eval()
    by_seq = {s: i for i, s in enumerate(got_seqs)}
    total_tokens = sum(max(float(m.sum()), 1.0) for m in d["loss_masks"])
    oracle_loss = cover = pw_err = pw_norm = 0.0
    student.zero_grad(set_to_none=True)
    for s in batch["samples"]:
        seq = tuple(s["prompt_ids"]) + tuple(s["response_ids"])
        j = by_seq.get(seq)
        if j is None:
            continue
        P, R = len(s["prompt_ids"]), len(s["response_ids"])
        ids_c = d["opcd_topk_ids"][j].to(dev).long()
        mask = d["loss_masks"][j].to(dev).double()
        x = torch.tensor(seq, device=dev).unsqueeze(0)
        logits = student(input_ids=x, use_cache=False).logits[0, P - 1 : P + R - 1].float()
        logp_full = torch.log_softmax(logits, -1)
        with torch.no_grad():
            tl = teacher(input_ids=torch.cat([ctx.to(dev), x[0]]).unsqueeze(0), use_cache=False).logits[0]
            C = ctx.numel()
            logq_full = torch.log_softmax(tl[C + P - 1 : C + P + R - 1].float(), -1)
            canon = torch.topk(logp_full.detach(), args.top_k, -1).indices
            cover += float((logp_full.detach().gather(-1, ids_c).exp().sum(-1).double() * mask).sum())
            lq_c = logq_full.gather(-1, ids_c)
            err = (d["opcd_teacher_topk_logprobs"][j].to(dev).float() - lq_c).abs()
            w = logp_full.detach().gather(-1, ids_c).exp()
            pw_err += float((w * err).sum()); pw_norm += float(w.sum())
        lp = logp_full.gather(-1, ids_c)
        kl = (lp.exp() * (lp - lq_c)).sum(-1)
        contrib = (kl.double() * mask).sum() / total_tokens
        contrib.float().backward()
        oracle_loss += float(contrib)
    report.update(oracle_loss=oracle_loss, mass_coverage=cover / total_tokens,
                  teacher_p_weighted_err=pw_err / max(pw_norm, 1e-12))
    if abs(logged - oracle_loss) > max(th["oracle_rel"] * oracle_loss, th.get("oracle_abs", 0.0)):
        issues.append(f"P3: logged loss {logged} vs oracle {oracle_loss}")
    if report["mass_coverage"] < th["coverage"]:
        issues.append(f"P3: top-k mass coverage {report['mass_coverage']:.6f} < {th['coverage']}")
    if report["teacher_p_weighted_err"] > th["teacher_pw_err"]:
        issues.append(f"P3: teacher log-prob error {report['teacher_p_weighted_err']:.4f} (context missing or wrong teacher?)")

    # ---- P4 update direction ----
    grads = {n: p.grad for n, p in student.named_parameters() if p.grad is not None}
    init, final = hf_tensors(args.student), hf_tensors(args.final_hf)
    agree = moved = considered = total = 0
    missing = [n for n in grads if n not in final]
    if missing:
        issues.append(f"P4: final weights miss {len(missing)} tensors, e.g. {missing[:3]}")
    for n, g in grads.items():
        if n not in final or n not in init:
            continue
        with safe_open(init[n][0], "pt") as h0, safe_open(final[n][0], "pt") as h1:
            w0, w1 = h0.get_tensor(n).to(dev).float(), h1.get_tensor(n).to(dev).float()
        if w0.shape != w1.shape:
            issues.append(f"P4: {n} shape {tuple(w1.shape)} != {tuple(w0.shape)}")
            continue
        delta = (w1 - w0) + lr * args.weight_decay * w0
        total += w0.numel()
        ch = (w1 != w0) & (g.abs() > 0)
        moved += int(ch.sum())
        considered += int(ch.sum())
        agree += int(((torch.sign(delta) == -torch.sign(g)) & ch).sum())
    report.update(sign_agreement=agree / max(considered, 1), moved_fraction=moved / max(total, 1))
    if report["sign_agreement"] < th["sign_agree"]:
        issues.append(f"P4: weight change agrees with the OPCD gradient sign on {report['sign_agreement']:.3f} of moved weights")
    if report["moved_fraction"] < th["moved_frac"]:
        issues.append(f"P4: only {report['moved_fraction']:.4f} of weights changed")

    report["issues"] = issues
    report["passed"] = not issues
    print(json.dumps(report, indent=2))
    if args.out:
        json.dump(report, open(args.out, "w"), indent=2)
    raise SystemExit(0 if not issues else 1)


if __name__ == "__main__":
    main()
