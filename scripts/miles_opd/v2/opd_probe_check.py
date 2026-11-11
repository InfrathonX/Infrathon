#!/usr/bin/env python3
"""Judge check of the one-step OPD probe (no candidate code is imported).

The candidate trained exactly one optimizer step, from the pinned initial weights, on
fixed trajectories. Given its step dump and resulting HF weights:

  P1 contract   one step; the trained sequences are exactly the batch; per-token shapes [R]
  P2 self       advantages == -beta * (old - teacher); logged loss == fp64 recomputation
  P3 oracle     independent fp32 HF student/teacher: the candidate's old and teacher
                log-probs of the sampled tokens, and the loss, agree with fp32 values
  P4 update     sign test: with Adam's first step, delta = -lr * g / (|g| + eps) - lr * wd * w,
                so every weight that moved must move against the fp32 OPD gradient
                g = (1/N) sum_i mean_t (log p_t - log q_t) * grad log p_t
  P5 reference  (with --reference-hf) the weight change agrees with the change the task's
                reference implementation made on the same batch, on the
                weights whose fp32 gradient is in the top TOP_FRAC of each tensor by magnitude:
                sign agreement and cosine of the two deltas there. Adam's first step moves
                every weight by about lr * sign(g), so on near-zero gradients the direction is
                bf16 noise in any implementation; the strong-gradient weights are where two
                correct implementations must agree. Whole-model figures are reported too.
"""

import argparse
import glob
import json
import math

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM

KEYS = ("tokens", "response_lengths", "loss_masks", "old_logprobs", "teacher_logprobs", "advantages")
DEFAULT_THRESHOLDS = {
    "adv_abs": 1e-4,         # P2: |A - (-beta (old - teacher))| per token
    "self_rel": 1e-3,        # P2: logged loss vs fp64 recomputation from the dump
    "student_lp_err": 0.05,  # P3: token-mean |old - fp32 student|
    "teacher_lp_err": 0.05,  # P3: token-mean |teacher - fp32 teacher|
    "oracle_abs": 0.02,      # P3: |logged loss - fp32 loss| (the loss is a mean reverse KL, O(1))
    "sign_agree": 0.9,       # P4
    "moved_frac": 0.05,      # P4
    "ref_cos_top": 0.9,         # P5, on the top-gradient weights
    "ref_sign_agree_top": 0.95,  # P5, on the top-gradient weights
}
TOP_FRAC = 0.1


def load_dump(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise AssertionError(f"no step dump matches {pattern}")
    recs = [torch.load(f, map_location="cpu", weights_only=True) for f in files]
    merged = {"steps": {r.get("step") for r in recs}, "loss": recs[0].get("loss")}
    for key in KEYS:
        merged[key] = [x for r in recs for x in r[key]]
    return merged


def hf_tensors(path):
    out = {}
    for f in sorted(glob.glob(f"{path}/*.safetensors")):
        with safe_open(f, "pt") as h:
            for k in h.keys():
                out[k] = f
    return out


def get(index, name, dev):
    with safe_open(index[name], "pt") as h:
        return h.get_tensor(name).to(dev).float()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", required=True)
    ap.add_argument("--dumps", required=True, help="glob of the probe's step dump files")
    ap.add_argument("--final-hf", required=True)
    ap.add_argument("--reference-hf", default=None, help="final_hf of the reference probe on the same batch")
    ap.add_argument("--student", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--kl-coef", type=float, default=1.0)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--thresholds", default="{}", help="JSON overrides of DEFAULT_THRESHOLDS")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    th = {**DEFAULT_THRESHOLDS, **json.loads(args.thresholds)}
    dev = torch.device("cuda")
    batch = json.load(open(args.batch))
    lr = float(batch["lr"])
    beta = args.kl_coef
    issues, report = [], {"thresholds": th}

    # ---- P1 contract ----
    d = load_dump(args.dumps)
    if d["steps"] != {0}:
        issues.append(f"P1: expected exactly one step 0, got {sorted(d['steps'])}")
    want = sorted(tuple(s["prompt_ids"]) + tuple(s["response_ids"]) for s in batch["samples"])
    got_seqs = [tuple(int(x) for x in t.tolist()) for t in d["tokens"]]
    if sorted(got_seqs) != want:
        issues.append("P1: trained sequences differ from the probe batch")
    for i, R in enumerate(d["response_lengths"]):
        for key in ("loss_masks", "old_logprobs", "teacher_logprobs", "advantages"):
            if tuple(d[key][i].shape) != (R,):
                issues.append(f"P1: sample {i} {key} shape {tuple(d[key][i].shape)} != ({R},)")

    # ---- P2 self-consistency ----
    adv_err, per_seq = 0.0, []
    for lp, lq, a, m in zip(d["old_logprobs"], d["teacher_logprobs"], d["advantages"], d["loss_masks"]):
        lp, lq, a, m = lp.double(), lq.double(), a.double(), m.double()
        adv_err = max(adv_err, float(((a + beta * (lp - lq)).abs() * m).max()) if m.sum() > 0 else 0.0)
        per_seq.append(float(((lp - lq) * m).sum() / m.sum().clamp_min(1)))
    self_loss = sum(per_seq) / max(len(per_seq), 1)
    logged = d["loss"]
    report.update(logged_loss=logged, self_consistent_loss=self_loss, advantage_max_err=adv_err)
    if adv_err > th["adv_abs"]:
        issues.append(f"P2: advantages differ from -beta*(old - teacher) by up to {adv_err:.3g}")
    if logged is None or not math.isfinite(logged) or abs(logged - self_loss) / max(abs(self_loss), 1e-12) > th["self_rel"]:
        issues.append(f"P2: logged loss {logged} vs recomputed {self_loss}")

    # ---- P3 oracle + fp32 gradient ----
    student = AutoModelForCausalLM.from_pretrained(args.student, dtype=torch.float32).to(dev)
    student.train()
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=torch.float32).to(dev).eval()
    by_seq = {s: i for i, s in enumerate(got_seqs)}
    n_seq = len(batch["samples"])
    s_err = t_err = n_tok = 0.0
    oracle_loss = 0.0
    student.zero_grad(set_to_none=True)
    for s in batch["samples"]:
        seq = tuple(s["prompt_ids"]) + tuple(s["response_ids"])
        j = by_seq.get(seq)
        if j is None:
            continue
        P, R = len(s["prompt_ids"]), len(s["response_ids"])
        x = torch.tensor(seq, device=dev).unsqueeze(0)
        y = x[0, P : P + R]
        mask = d["loss_masks"][j].to(dev).double()
        lp = torch.log_softmax(student(input_ids=x, use_cache=False).logits[0, P - 1 : P + R - 1].float(), -1)
        lp = lp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            lq = torch.log_softmax(teacher(input_ids=x, use_cache=False).logits[0, P - 1 : P + R - 1].float(), -1)
            lq = lq.gather(-1, y.unsqueeze(-1)).squeeze(-1)
            s_err += float(((d["old_logprobs"][j].to(dev).double() - lp.detach().double()).abs() * mask).sum())
            t_err += float(((d["teacher_logprobs"][j].to(dev).double() - lq.double()).abs() * mask).sum())
            n_tok += float(mask.sum())
            coef = (lp.detach() - lq).double()  # = -A_t / beta in fp32
        denom = mask.sum().clamp_min(1)
        oracle_loss += float((coef * mask).sum() / denom) / n_seq
        # surrogate whose gradient is the OPD policy gradient at ratio 1
        (((beta * coef) * lp.double() * mask).sum() / denom / n_seq).float().backward()
    report.update(oracle_loss=oracle_loss, student_lp_mean_err=s_err / max(n_tok, 1),
                  teacher_lp_mean_err=t_err / max(n_tok, 1))
    if report["student_lp_mean_err"] > th["student_lp_err"]:
        issues.append(f"P3: old log-probs differ from the fp32 student by {report['student_lp_mean_err']:.4f} per token")
    if report["teacher_lp_mean_err"] > th["teacher_lp_err"]:
        issues.append(f"P3: teacher log-probs differ from the fp32 teacher by {report['teacher_lp_mean_err']:.4f} per token")
    if logged is None or abs(logged - oracle_loss) > th["oracle_abs"]:
        issues.append(f"P3: logged loss {logged} vs fp32 oracle {oracle_loss}")

    # ---- P4 update direction / P5 reference agreement ----
    grads = {n: p.grad for n, p in student.named_parameters() if p.grad is not None}
    init, final = hf_tensors(args.student), hf_tensors(args.final_hf)
    ref = hf_tensors(args.reference_hf) if args.reference_hf else None
    missing = [n for n in grads if n not in final]
    if missing:
        issues.append(f"P4: final weights miss {len(missing)} tensors, e.g. {missing[:3]}")
    agree = moved = total = 0
    dot = nc = nr = 0.0
    ref_agree = ref_moved = 0
    top_agree = top_n = 0
    tdot = tnc = tnr = 0.0
    tref_agree = tref_n = 0
    for n, g in grads.items():
        if n not in final or n not in init:
            continue
        w0, w1 = get(init, n, dev), get(final, n, dev)
        if w0.shape != w1.shape:
            issues.append(f"P4: {n} shape {tuple(w1.shape)} != {tuple(w0.shape)}")
            continue
        delta = (w1 - w0) + lr * args.weight_decay * w0
        total += w0.numel()
        ch = (w1 != w0) & (g.abs() > 0)
        moved += int(ch.sum())
        agree += int(((torch.sign(delta) == -torch.sign(g)) & ch).sum())
        a_ = g.abs().reshape(-1)
        k = max(1, int(a_.numel() * (1 - TOP_FRAC)))
        top = (g.abs() >= a_.kthvalue(k).values) & (g.abs() > 0)
        del a_
        top_moved = top & ch  # a bf16 checkpoint cannot record a step on large weights (e.g. norms)
        top_n += int(top_moved.sum())
        top_agree += int(((torch.sign(delta) == -torch.sign(g)) & top_moved).sum())
        if ref is not None and n in ref:
            dr = get(ref, n, dev) - w0
            dc = w1 - w0
            dot += float((dc.double() * dr.double()).sum())
            nc += float((dc.double() ** 2).sum())
            nr += float((dr.double() ** 2).sum())
            both = (dc != 0) | (dr != 0)
            ref_moved += int(both.sum())
            ref_agree += int(((torch.sign(dc) == torch.sign(dr)) & both).sum())
            top_either = top & both
            tc, tr = dc[top_either].double(), dr[top_either].double()
            tdot += float((tc * tr).sum()); tnc += float((tc ** 2).sum()); tnr += float((tr ** 2).sum())
            tref_n += int(top_either.sum())
            tref_agree += int(((torch.sign(dc) == torch.sign(dr)) & top_either).sum())
    report.update(sign_agreement=agree / max(moved, 1), moved_fraction=moved / max(total, 1),
                  sign_agreement_top=top_agree / max(top_n, 1))
    if report["sign_agreement"] < th["sign_agree"]:
        issues.append(f"P4: weight change agrees with the OPD gradient sign on {report['sign_agreement']:.3f} of moved weights")
    if report["moved_fraction"] < th["moved_frac"]:
        issues.append(f"P4: only {report['moved_fraction']:.4f} of weights changed")
    if ref is not None:
        report.update(reference_delta_cos=dot / max(math.sqrt(nc * nr), 1e-30),
                      reference_sign_agreement=ref_agree / max(ref_moved, 1),
                      reference_delta_cos_top=tdot / max(math.sqrt(tnc * tnr), 1e-30),
                      reference_sign_agreement_top=tref_agree / max(tref_n, 1))
        if report["reference_delta_cos_top"] < th["ref_cos_top"]:
            issues.append(f"P5: on the top-gradient weights the weight change vs the reference has cosine "
                          f"{report['reference_delta_cos_top']:.4f} < {th['ref_cos_top']}")
        if report["reference_sign_agreement_top"] < th["ref_sign_agree_top"]:
            issues.append(f"P5: on the top-gradient weights the weight change agrees in sign with the reference on "
                          f"{report['reference_sign_agreement_top']:.3f} < {th['ref_sign_agree_top']}")

    report["issues"] = issues
    report["passed"] = not issues
    print(json.dumps(report, indent=2))
    if args.out:
        json.dump(report, open(args.out, "w"), indent=2)
    raise SystemExit(0 if not issues else 1)


if __name__ == "__main__":
    main()
