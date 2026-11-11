#!/usr/bin/env python3
"""Independent fp32 oracle for one GRPO step of the ALFWorld task (plain HF, no Miles code).

Reads the step dump(s) of one optimizer step (tokens, response_lengths, loss_masks,
engine_logprobs, train_logprobs, advantages, loss) and recomputes with a Hugging Face
fp32 forward of the pre-update policy:

* the policy log-probs of every response token and their error against the candidate's
  ``train_logprobs`` (mask-weighted mean / max abs error);
* the specified loss  L = sum_t m_t max(-A rho_t, -A clip(rho_t, 1-e_lo, 1+e_hi)) / sum_t m_t,
  rho_t = exp(log pi_theta - log pi_engine), with the dumped engine log-probs and
  advantages, and its error against the logged loss;
* optionally (``--grad-records``) the gradient of L w.r.t. the first input-embedding rows
  and the full gradient norm, compared with what Megatron held before its optimizer step
  (provenance hook records) and the norm it reported.

``--policy-snapshot`` loads a copy of the weights the SGLang engine served (fused layout).
"""

import argparse
import glob
import json
import math
import os
import sys

import torch
from transformers import AutoModelForCausalLM

FP_ROWS = 128
CHUNK = 2048
Q_CHUNK = 1024


def chunked_causal_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    """Exact fp32 causal attention, computed in query blocks so ~20k-token episodes fit.

    query [B, H, T, D], key/value [B, KV, T, D] (grouped-query heads are expanded here)."""
    b, h, t, d = query.shape
    rep = h // key.shape[1]
    k = key.repeat_interleave(rep, dim=1)
    v = value.repeat_interleave(rep, dim=1)
    scale = scaling if scaling is not None else d ** -0.5
    outs = []
    pos = torch.arange(t, device=query.device)
    for q0 in range(0, t, Q_CHUNK):
        q = query[:, :, q0:q0 + Q_CHUNK]
        kv_end = min(t, q0 + q.shape[2])
        scores = torch.matmul(q, k[:, :, :kv_end].transpose(-1, -2)) * scale
        causal = pos[q0:q0 + q.shape[2], None] >= pos[None, :kv_end]
        scores = scores.masked_fill(~causal, float("-inf"))
        outs.append(torch.matmul(torch.softmax(scores, dim=-1), v[:, :, :kv_end]))
    return torch.cat(outs, dim=2).transpose(1, 2).contiguous(), None


def _register_attention():
    from transformers import AttentionInterface

    AttentionInterface.register("chunked_causal_fp32", chunked_causal_attention)


def load_sglang_snapshot(model, path):
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
    if missing or unexpected:
        raise AssertionError(f"snapshot does not map onto the pinned model: missing {missing[:4]}, unexpected {unexpected[:4]}")
    model.load_state_dict(sd, strict=False)
    return model


def load_grad_records(paths, rows, width):
    vals = torch.zeros(rows * width, dtype=torch.float64)
    valid = torch.zeros(rows * width, dtype=torch.bool)
    for path in paths:
        r = torch.load(path, map_location="cpu", weights_only=False)
        flat = r["rows"].to(torch.float64).reshape(-1)[: rows * width]
        lo, hi = (0, flat.numel()) if r.get("owned") is None else (max(0, r["owned"][0]), min(flat.numel(), r["owned"][1]))
        if hi > lo:
            fresh = ~valid[lo:hi]
            vals[lo:hi][fresh] = flat[lo:hi][fresh]
            valid[lo:hi] = True
    return vals, valid


def load_step(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise AssertionError(f"no dump matches {pattern}")
    recs = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
    rd = {}
    for r in recs:
        for k, v in r.items():
            if isinstance(v, list):
                rd.setdefault(k, []).extend(v)
    need = ("tokens", "response_lengths", "loss_masks", "engine_logprobs", "train_logprobs", "advantages")
    missing = [k for k in need if k not in rd]
    if missing:
        raise AssertionError(f"dump misses {missing}")
    return rd, recs[0].get("loss")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--policy", required=True, help="HF dir of the pinned policy (config/tokenizer; weights unless snapshot)")
    ap.add_argument("--policy-snapshot", default=None)
    ap.add_argument("--eps-clip", type=float, default=0.2)
    ap.add_argument("--eps-clip-high", type=float, default=0.28)
    ap.add_argument("--grad-records", default=None)
    ap.add_argument("--grad-norm", type=float, default=None)
    ap.add_argument("--max-samples", type=int, default=None, help="authoring only")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", help="authoring diagnostics only")
    ap.add_argument("--save-grad", default=None, help="authoring: save the oracle's embedding-row gradient")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rd, logged = load_step(a.dump)
    n = len(rd["tokens"]) if a.max_samples is None else min(a.max_samples, len(rd["tokens"]))
    grad_mode = a.grad_records is not None
    dev = torch.device(a.device)
    _register_attention()
    model = AutoModelForCausalLM.from_pretrained(a.policy, dtype=torch.float32, attn_implementation="chunked_causal_fp32").to(getattr(torch, a.dtype))
    if a.policy_snapshot:
        load_sglang_snapshot(model, a.policy_snapshot)
    model = model.to(dev).eval()
    model.requires_grad_(grad_mode)
    if grad_mode:
        # HF only checkpoints layers in train mode; Qwen3 has no dropout, so train() changes nothing else
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
        model.train()

    den = sum(float(rd["loss_masks"][i].double().sum()) for i in range(n))
    num = self_num = 0.0
    lp_err_sum = lp_err_max = 0.0
    abs_adv = 0.0
    for i in range(n):
        seq = rd["tokens"][i].long()
        R = int(rd["response_lengths"][i])
        T = seq.numel()
        mask = rd["loss_masks"][i].to(dev, torch.float32)
        old = rd["engine_logprobs"][i].to(dev, torch.float32)
        adv = float(rd["advantages"][i])
        ctx = torch.enable_grad() if grad_mode else torch.no_grad()
        with ctx:
            hidden = model.model(input_ids=seq.unsqueeze(0).to(dev), use_cache=False).last_hidden_state[0, T - R - 1:T - 1]
            targets = seq[T - R:].to(dev)
            parts = []
            for c0 in range(0, R, CHUNK):
                logits = model.lm_head(hidden[c0:c0 + CHUNK]).float()
                parts.append(torch.log_softmax(logits, -1).gather(-1, targets[c0:c0 + CHUNK].unsqueeze(-1)).squeeze(-1))
            lp = torch.cat(parts)
            ratio = torch.exp(lp - old)
            pg = torch.maximum(-adv * ratio, -adv * ratio.clamp(1 - a.eps_clip, 1 + a.eps_clip_high))
            contrib = (pg * mask).sum() / max(den, 1.0)
            if grad_mode:
                contrib.backward()
        num += float(contrib) * max(den, 1.0)
        cand = rd["train_logprobs"][i].to(dev, torch.float32)
        err = (cand - lp.detach()).abs() * mask
        lp_err_sum += float(err.sum())
        lp_err_max = max(lp_err_max, float(err.max()) if err.numel() else 0.0)
        abs_adv += abs(adv) * float(mask.sum())
        # fp64 identity on the candidate's own numbers
        c64, o64, m64 = rd["train_logprobs"][i].double(), rd["engine_logprobs"][i].double(), rd["loss_masks"][i].double()
        r64 = torch.exp(c64 - o64)
        self_num += float((torch.maximum(-adv * r64, -adv * r64.clamp(1 - a.eps_clip, 1 + a.eps_clip_high)) * m64).sum())
        del hidden, parts, logits, lp, ratio, pg, contrib

    report = {
        "samples": n,
        "valid_tokens": den,
        "loss_oracle": num / max(den, 1.0),
        "loss_self_consistent": self_num / max(den, 1.0),
        "logged_loss": logged,
        "mean_abs_advantage": abs_adv / max(den, 1.0),
        "train_logprob_mean_abs_err": lp_err_sum / max(den, 1.0),
        "train_logprob_max_abs_err": lp_err_max,
        "policy_weights": a.policy_snapshot or a.policy,
    }
    if isinstance(logged, (int, float)):
        report["logged_vs_oracle_abs_err"] = abs(logged - report["loss_oracle"])
        report["logged_vs_self_abs_err"] = abs(logged - report["loss_self_consistent"])
    if grad_mode:
        emb = model.get_input_embeddings().weight
        rows, width = FP_ROWS, emb.shape[1]
        oracle = emb.grad.detach()[:rows].to(torch.float64).reshape(-1).cpu()
        if a.save_grad:
            named = dict(model.named_parameters())
            keep = ["model.norm.weight", "model.layers.18.mlp.down_proj.weight", "model.layers.0.self_attn.q_proj.weight",
                    "model.layers.35.mlp.gate_proj.weight", "model.layers.18.input_layernorm.weight"]
            torch.save({"emb_rows": oracle, **{k: named[k].grad.detach().double().cpu() for k in keep}}, a.save_grad)
        total = math.sqrt(sum(float(p.grad.detach().double().pow(2).sum()) for p in model.parameters() if p.grad is not None))
        vals, valid = load_grad_records([x for x in a.grad_records.split(",") if x], rows, width)
        x, y = vals[valid], oracle[valid]
        top = y.abs() >= torch.quantile(y.abs().float(), 0.95).double() if y.numel() else y.bool()
        xt, yt = x[top], y[top]
        report.update({
            "grad_top5_cosine": float((xt @ yt) / max(float(xt.norm() * yt.norm()), 1e-30)) if xt.numel() else None,
            "grad_top5_sign_agree": float(((xt > 0) == (yt > 0)).double().mean()) if xt.numel() else None,
            "grad_top5_norm_ratio": float(xt.norm() / max(float(yt.norm()), 1e-30)) if xt.numel() else None,
            "grad_valid_elements": int(valid.sum()),
            "grad_slice_cosine": float((x @ y) / max(float(x.norm() * y.norm()), 1e-30)) if x.numel() else None,
            "grad_slice_norm_ratio": float(x.norm() / max(float(y.norm()), 1e-30)) if x.numel() else None,
            "oracle_grad_norm": total,
            "candidate_grad_norm": a.grad_norm,
            "grad_norm_ratio": (a.grad_norm / total) if (a.grad_norm is not None and total > 0) else None,
        })
    print(json.dumps(report, indent=2))
    if a.out:
        json.dump(report, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
