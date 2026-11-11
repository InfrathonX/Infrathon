#!/usr/bin/env python3
"""Judge-side provenance check: was the student trained by Megatron-Core inside Miles, and
are the weights the engine served and the final checkpoint exactly what those optimizer
steps produced?

Reads only the evidence written by ``opcd_provenance_hook`` (``prov-*.jsonl``,
``prov-grad-*.pt``, ``snapshots/``) and ``opcd_audit_hook`` (``audit-*.jsonl``), plus
the pinned initial student and the candidate's final checkpoint. Imports no candidate
code and fails closed on missing evidence.

  M1 framework  exactly `--steps` successful outermost Megatron-Core optimizer steps per
                training process, each called from Miles' MegatronTrainRayActor, with the
                optimizer and megatron.core loaded from the pinned Megatron-LM tree
  M2 optimizer  AdamW (decoupled weight decay) with the contract's lr, betas, eps, weight
                decay (Megatron's no-decay group may use 0) and clip value
  M3 lineage    Megatron starts from the pinned initial student; every weight state the
                student engine served before its first training generation is that
                initial state; its k-th publication after that is Megatron's weights
                after step k-1; the final checkpoint is the weights after the last step;
                each optimizer step starts from exactly the student weights and optimizer
                state (main params, moments) the previous step returned with (the hook's
                state fingerprints, sampled at fresh random positions per step)
  M4 snapshots  (run) each served-weight copy taken before a random publication b lies in
                its own student engine's publication timeline (after the engine's previous
                publication completed, before b completed), which gives the version it
                copied, and matches that version's fingerprint; at least one lies on a step
                in 1..steps-1; at most N_SNAPSHOTS copies per engine, each lasting at most
                EXCLUDE_MAX_SECONDS
  M5 engine     (run) every student engine process was started by a Ray actor
  M6 final      (run) the final checkpoint is, tensor by tensor, both the weights Megatron held
                right after its last optimizer step (full SHA-256 digest in Megatron's names, layout
                and dtype; trainer at tensor- and pipeline-parallel size 1) and the weights each
                student engine served after its last publication (full digest in SGLang's fused
                layout and served dtype), with nothing else in the checkpoint. With M3's per-step
                fingerprints and state chain this binds the scored model to the training steps
Reports the student's Megatron model config (the Judge compares probe and run), the
per-step gradient records and grad norms (for the random-step gradient oracle), and the
evaluator's intervals taken out of the timed window: the snapshot copies and the digests,
each where the hook takes it (a copy in its publication timeline as in M4, a digest starting
at most DIGEST_START_SLACK after the record it follows) and lasting at most
EXCLUDE_MAX_SECONDS. Of each, T loses at most CREDIT_MAX_SECONDS and nothing while the
trainer was in a training step (from its first grad-enabled forward to the optimizer step's
return), so a slowed or overlapped copy or digest cannot take training time out of T.
"""

import argparse
import contextlib
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from opcd_provenance_hook import FP_ROWS, N_SNAPSHOTS, tensor_digest  # noqa: E402  (the hook is inert here)

HEALTH = "HEALTH_CHECK"
# model-config fields that change numerics: probe and run must agree on all of them
NUMERIC_CONFIG = ("bf16", "fp16", "params_dtype", "fp8", "fp4", "attention_backend", "attention_softmax_in_fp32",
                  "fp32_residual_connection", "apply_rope_fusion", "masked_softmax_fusion", "bias_dropout_fusion",
                  "gradient_accumulation_fusion", "calculate_per_token_loss", "deterministic_mode",
                  "tensor_model_parallel_size", "sequence_parallel", "context_parallel_size")
# an interval the evaluator takes out of the timed window (a snapshot copy, a digest) lasts at most this long
# (the H100 reference's 3.4 GB snapshot copies took 3.8-4.2 s), and a digest starts at most DIGEST_START_SLACK
# after the record it follows (both hooks start it at once)
EXCLUDE_MAX_SECONDS = 60.0
DIGEST_START_SLACK = 5.0
# T loses at most this much of each interval: the copies took 3.75 / 3.81 s in h100pin-ref-01; hashing the 3.44 GB
# student with the hook's 8 threads took 1.8 s warm / 4.6 s with fresh host memory on the qualification host's
# CPUs, before the device-to-host copy. The copies and digests run in candidate processes, so their length is
# credited only up to this calibrated bound
CREDIT_MAX_SECONDS = 8.0


def load_records(audit_dir):
    recs = []
    for f in sorted(glob.glob(os.path.join(audit_dir, "*.jsonl"))):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        recs.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # a line cut by a killed process
    return recs


def hf_fingerprint(model_dir):
    """Fingerprint (same definition as the hook) of an HF checkpoint's input embedding."""
    import hashlib

    import torch
    from safetensors import safe_open

    d = Path(model_dir)
    idx = d / "model.safetensors.index.json"
    name = "model.embed_tokens.weight"
    if idx.is_file():
        wmap = json.loads(idx.read_text())["weight_map"]
        if name not in wmap:
            name = "lm_head.weight" if "lm_head.weight" in wmap else name
        files = [d / wmap[name]] if name in wmap else []
    else:
        files = sorted(d.glob("*.safetensors"))
    for f in files:
        with safe_open(str(f), framework="pt") as fh:
            if name in fh.keys():
                w = fh.get_slice(name)[:FP_ROWS]
                if w.dtype != torch.bfloat16:
                    w = w.to(torch.bfloat16)
                return hashlib.sha256(w.contiguous().view(torch.int16).numpy().tobytes()).hexdigest()
    return None


# SGLang serves these HF projections as one tensor, concatenated along dim 0 (the split in
# opcd_oracle.load_sglang_snapshot is the inverse)
FUSED = {"qkv_proj.weight": ("q_proj.weight", "k_proj.weight", "v_proj.weight"),
         "gate_up_proj.weight": ("gate_proj.weight", "up_proj.weight")}
# Megatron-Core GPT at tensor- and pipeline-parallel size 1 (transformer-engine or local layer spec) -> the HF
# tensors each parameter holds, as Miles converts them: linear_qkv interleaves q/k/v per query group
# (MEGATRON_QKV), linear_fc1 is [gate; up], and the TE layers carry the preceding RMSNorm weight
MEGATRON_TOP = {"embedding.word_embeddings.weight": ("model.embed_tokens.weight",),
                "decoder.final_layernorm.weight": ("model.norm.weight",),
                "output_layer.weight": ("lm_head.weight",)}
MEGATRON_LAYER = {
    "self_attention.linear_qkv.weight": ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"),
    "self_attention.linear_qkv.layer_norm_weight": ("input_layernorm.weight",),
    "input_layernorm.weight": ("input_layernorm.weight",),
    "self_attention.q_layernorm.weight": ("self_attn.q_norm.weight",),
    "self_attention.k_layernorm.weight": ("self_attn.k_norm.weight",),
    "self_attention.linear_proj.weight": ("self_attn.o_proj.weight",),
    "mlp.linear_fc1.layer_norm_weight": ("post_attention_layernorm.weight",),
    "pre_mlp_layernorm.weight": ("post_attention_layernorm.weight",),
    "mlp.linear_fc1.weight": ("mlp.gate_proj.weight", "mlp.up_proj.weight"),
    "mlp.linear_fc2.weight": ("mlp.down_proj.weight",),
}
MEGATRON_QKV = "self_attention.linear_qkv.weight"


def megatron_parts(name):
    """The HF tensor names Megatron parameter `name` is made of, or None."""
    if name in MEGATRON_TOP:
        return MEGATRON_TOP[name]
    m = re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", name)
    if m and m.group(2) in MEGATRON_LAYER:
        return tuple(f"model.layers.{m.group(1)}.{p}" for p in MEGATRON_LAYER[m.group(2)])
    return None


def attention_heads(model_dir):
    """(query heads, key/value heads, head dim) from an HF config.json."""
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    return (cfg["num_attention_heads"], cfg["num_key_value_heads"],
            cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"])


@contextlib.contextmanager
def checkpoint_tensors(model_dir):
    """({name: open safetensors file} over every *.safetensors in `model_dir`, [tensors stored twice])."""
    from safetensors import safe_open

    where, twice = {}, []
    with contextlib.ExitStack() as stack:
        for f in sorted(Path(model_dir).glob("*.safetensors")):
            fh = stack.enter_context(safe_open(str(f), framework="pt"))
            for k in fh.keys():
                if k in where:
                    twice.append(f"tensor {k} is stored twice (again in {f.name})")
                where[k] = fh
        yield where, twice


def digest_mismatch(t, rec, side):
    """Why tensor `t`, converted to the recorded dtype, is not the digest record `rec`; None when it is."""
    import torch

    dtype = getattr(torch, str(rec.get("dtype", "")).removeprefix("torch."), None)
    if not isinstance(dtype, torch.dtype):
        return f"unknown {side} dtype {rec.get('dtype')}"
    if list(t.shape) != list(rec.get("shape") or []):
        return f"shape {list(t.shape)} != {side} {rec.get('shape')}"
    return None if tensor_digest(t.to(dtype)) == rec.get("sha256") else f"bytes differ from the {side} weights"


def final_weight_issues(model_dir, served):
    """Why the final checkpoint is not exactly the served weights `served` ({served name: {dtype,
    shape, sha256}} in SGLang's layout, from an ``sglang_digest`` record); [] when it is.

    Every tensor of every *.safetensors file in `model_dir` (the Judge evaluates all of them) must
    map onto one served tensor, fused as SGLang fuses it, and have its SHA-256 after conversion to
    the served dtype; a tied ``lm_head.weight`` must equal the served input embedding."""
    import torch

    used = set()
    with checkpoint_tensors(model_dir) as (where, issues):
        def differs(parts, rec):
            ts = [where[p].get_tensor(p) for p in parts]
            return digest_mismatch(torch.cat(ts, 0) if len(ts) > 1 else ts[0], rec, "served")

        for name, rec in sorted(served.items()):
            suffix = next((x for x in FUSED if name.endswith(x)), None)
            parts = [name[: -len(suffix)] + x for x in FUSED[suffix]] if suffix else [name]
            used.update(parts)
            missing = [x for x in parts if x not in where]
            if missing:
                issues.append(f"served tensor {name}: {missing} not in the final checkpoint")
                continue
            why = differs(parts, rec)
            if why:
                issues.append(f"served tensor {name}: {why}")
        extra = sorted(set(where) - used)
        emb = served.get("model.embed_tokens.weight")
        if "lm_head.weight" in extra and "lm_head.weight" not in served and emb:
            extra.remove("lm_head.weight")
            why = differs(["lm_head.weight"], emb)
            if why:
                issues.append(f"lm_head.weight (tied to the served embedding): {why}")
        issues += [f"final checkpoint tensor {k} is not a served weight" for k in extra]
    return issues


def trainer_weight_issues(model_dir, trainer, heads):
    """Why the final checkpoint is not exactly Megatron's weights `trainer` ({Megatron name: {dtype,
    shape, sha256}}, from a ``megatron_digest`` record); [] when it is.

    Every trainer tensor is rebuilt from the checkpoint's HF tensors in Megatron's layout (MEGATRON_*;
    `heads` = (query heads, key/value heads, head dim) orders linear_qkv) and must have its SHA-256 after
    conversion to the trainer's dtype, and every checkpoint tensor must be part of one (a tied
    ``lm_head.weight`` is left to final_weight_issues, which binds it to the embedding)."""
    import torch

    n_q, n_kv, d = heads
    issues, used = [], set()
    with checkpoint_tensors(model_dir) as (where, _):
        for name, rec in sorted(trainer.items()):
            parts = megatron_parts(name)
            if parts is None:
                issues.append(f"trainer tensor {name} is not a Qwen3 Megatron weight at tensor- and "
                              "pipeline-parallel size 1")
                continue
            used.update(parts)
            missing = [x for x in parts if x not in where]
            if missing:
                issues.append(f"trainer tensor {name}: {missing} not in the final checkpoint")
                continue
            ts = [where[p].get_tensor(p) for p in parts]
            try:
                if name.endswith(MEGATRON_QKV):
                    q, k, v = ts
                    h = q.shape[-1]
                    t = torch.cat([q.reshape(n_kv, n_q // n_kv * d, h), k.reshape(n_kv, d, h), v.reshape(n_kv, d, h)], 1)
                    t = t.reshape(-1, h)
                else:
                    t = torch.cat(ts, 0) if len(ts) > 1 else ts[0]
            except RuntimeError as exc:
                issues.append(f"trainer tensor {name}: the checkpoint's {list(parts)} do not form it ({str(exc)[:120]})")
                continue
            why = digest_mismatch(t, rec, "trainer")
            if why:
                issues.append(f"trainer tensor {name}: {why}")
        extra = set(where) - used
        if "output_layer.weight" not in trainer:
            extra.discard("lm_head.weight")
        issues += [f"final checkpoint tensor {k} is not a trainer weight" for k in sorted(extra)]
    return issues


def interval_issue(x, anchor, before=None):
    """Why record `x`'s [t_start, t_end] is not where its hook takes it: starting at most
    DIGEST_START_SLACK after `anchor` (the time of the record it follows), lasting at most
    EXCLUDE_MAX_SECONDS, and ending by `before` when given; None when it is."""
    t0, t1 = x.get("t_start"), x.get("t_end")
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (t0, t1, anchor)) or t1 < t0:
        return f"[{t0}, {t1}] after {anchor} is not a time interval"
    if not anchor <= t0 <= anchor + DIGEST_START_SLACK:
        return f"starts {t0 - anchor:+.3f} s after the record it follows (the hook starts it at once)"
    if t1 - t0 > EXCLUDE_MAX_SECONDS:
        return f"lasts {t1 - t0:.1f} s, more than {EXCLUDE_MAX_SECONDS:g} s"
    if before is not None and t1 > before:
        return "ends after the next publication completed"
    return None


def close(a, b, rel=1e-6):
    try:
        return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=1e-15)
    except (TypeError, ValueError):
        return False


def snapshot_version(snapshot, publications, first_dispatch):
    """The weight version the engine served during a snapshot copy, from that engine's own publication
    timeline (as in alfworld_provenance_check and opd_provenance_check); None when the copy is not where
    the hook takes it.

    The hook copies synchronously right before publication b begins, and each ``sglang_publish`` record
    is written when its publication completes. So the copy must come from an engine with exactly one
    completed publication b, start after the engine's previous publication completed and end before b
    completed. Only completed publications after the first training generation advance the version."""
    pid = snapshot.get("pid")
    start, end = snapshot.get("t_start"), snapshot.get("t_end")
    if pid is None or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (start, end)) or start > end:
        return None
    ordered = sorted((p for p in publications if p.get("pid") == pid and p.get("success") is not False),
                     key=lambda p: p["t"])
    target = [p for p in ordered if p.get("b") == snapshot.get("b")]
    if len(target) != 1:
        return None
    completed = [p for p in ordered if p["t"] <= start]
    pending = ordered[len(completed):]
    if not pending or pending[0] is not target[0] or end > target[0]["t"]:
        return None
    return sum(p["t"] > first_dispatch for p in completed)


def credited(t0, t1, busy):
    """The parts of an evaluator interval [t0, t1] taken out of T: none of it while the trainer was in a
    training step (`busy`: [start, end] pairs) and at most CREDIT_MAX_SECONDS in all, from its start."""
    free, cur = [], t0
    for b0, b1 in sorted(busy):
        if b1 <= cur or b0 >= t1:
            continue
        if b0 > cur:
            free.append([cur, b0])
        cur = b1
        if cur >= t1:
            break
    if cur < t1:
        free.append([cur, t1])
    out, left = [], CREDIT_MAX_SECONDS
    for a, b in free:
        if left <= 0:
            break
        out.append([a, min(b, a + left)])
        left -= out[-1][1] - a
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit-dir", required=True)
    ap.add_argument("--mode", choices=["run", "probe"], required=True)
    ap.add_argument("--steps", type=int, required=True, help="optimizer steps the command must take")
    ap.add_argument("--contract", required=True)
    ap.add_argument("--lr", type=float, default=None, help="override the contract lr (probe batches carry their own)")
    ap.add_argument("--initial-hf", required=True, help="the pinned initial student")
    ap.add_argument("--final-hf", required=True)
    ap.add_argument("--student-model-path", default=None, help="model_path of the student engine (default: --initial-hf)")
    ap.add_argument("--pinned-megatron", default="/root/Megatron-LM")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    c = json.loads(Path(a.contract).read_text())
    o = c["optimizer"]
    lr = a.lr if a.lr is not None else o["lr"]
    N = a.steps
    student_path = a.student_model_path or a.initial_hf
    pinned = str(Path(a.pinned_megatron).resolve()) + "/"
    recs = load_records(a.audit_dir)
    issues = defaultdict(list)

    def issue(code, msg):
        issues[code].append(msg)

    # ---------------------------------------------------------------- M1 framework
    attempts = sorted((r for r in recs if r.get("kind") == "megatron_step"), key=lambda r: r["t_start"])
    by_pid = defaultdict(list)
    for r in attempts:
        by_pid[r["pid"]].append(r)
    if not attempts:
        issue("M1", "no Megatron-Core optimizer step was recorded: the student must be trained by Megatron-Core inside "
                    "Miles' MegatronTrainRayActor, and the OPCD_AUDIT_* environment must reach every process")
    for pid, rs in by_pid.items():
        bad = [r["attempt"] for r in rs if not r.get("success")]
        if bad:
            issue("M1", f"training process {pid}: optimizer step attempts {bad[:5]} did not update (skipped step)")
        if len(rs) != N:
            issue("M1", f"training process {pid}: {len(rs)} optimizer steps, need exactly {N}")
    no_actor = [r["attempt"] for r in attempts
                if not any(q.startswith("miles.") and q.endswith(".MegatronTrainRayActor") for q in r.get("miles_stack") or [])]
    if no_actor:
        issue("M1", f"{len(no_actor)} optimizer steps were not driven by Miles' MegatronTrainRayActor "
                    f"(Miles classes on the stack: {attempts[0].get('miles_stack') if attempts else None})")
    unpinned = set()
    for r in attempts:
        for f in [r.get("optimizer_file"), r.get("megatron_core_file")] + [i.get("file") for i in r.get("inner") or []]:
            if not f or not str(Path(f).resolve()).startswith(pinned):
                unpinned.add(str(f))
    if unpinned:
        issue("M1", f"Megatron-Core not loaded from the pinned tree {pinned}: {sorted(unpinned)[:3]}")

    # ---------------------------------------------------------------- M2 optimizer
    m2 = set()
    for r in attempts:
        for i in r.get("inner") or []:
            cfg = i.get("config") or {}
            if cfg.get("optimizer") != "adam" or cfg.get("decoupled_weight_decay") is False or i.get("adam_w_mode") is False:
                m2.add(f"optimizer {cfg.get('optimizer')} decoupled_weight_decay={cfg.get('decoupled_weight_decay')} "
                       f"adam_w_mode={i.get('adam_w_mode')}: need AdamW")
            if not close(cfg.get("clip_grad"), o["clip_grad"]):
                m2.add(f"clip_grad {cfg.get('clip_grad')} != {o['clip_grad']}")
            for key, want in (("adam_beta1", o["betas"][0]), ("adam_beta2", o["betas"][1]), ("adam_eps", o["eps"])):
                if not close(cfg.get(key), want):
                    m2.add(f"{key} {cfg.get(key)} != {want}")
        groups = r.get("param_groups")
        if not groups:
            m2.add("optimizer param groups not recorded")
            continue
        for g in groups:
            if not close(g.get("lr"), lr):
                m2.add(f"lr {g.get('lr')} != {lr}")
            if g.get("betas") is not None and not (close(g["betas"][0], o["betas"][0]) and close(g["betas"][1], o["betas"][1])):
                m2.add(f"betas {g.get('betas')} != {o['betas']}")
            if g.get("eps") is not None and not close(g["eps"], o["eps"]):
                m2.add(f"eps {g.get('eps')} != {o['eps']}")
            if not (close(g.get("weight_decay"), o["weight_decay"]) or g.get("weight_decay") == 0.0):
                m2.add(f"weight_decay {g.get('weight_decay')} not in {{0, {o['weight_decay']}}}")
        if not any(close(g.get("weight_decay"), o["weight_decay"]) for g in groups):
            m2.add(f"no parameter group uses weight decay {o['weight_decay']}")
    for m in sorted(m2):
        issue("M2", m)
    if any(r.get("optimizer_error") for r in attempts):
        issue("M2", f"optimizer evidence unreadable: {next(r['optimizer_error'] for r in attempts if r.get('optimizer_error'))}")

    # ---------------------------------------------------------------- M3 lineage
    # the process with the input embedding (all data-parallel replicas agree; take the lowest rank)
    emb_pids = sorted({r["pid"] for r in attempts if r.get("fp_after")},
                      key=lambda p: min(x.get("world_rank", 0) for x in by_pid[p]))
    V = defaultdict(set)
    primary = emb_pids[0] if emb_pids else None
    if primary is None:
        issue("M3", "no weight fingerprint recorded on the training side")
    else:
        ok_steps = [r for r in by_pid[primary] if r.get("success")]
        for k, r in enumerate(ok_steps):
            V[k + 1].add(r["fp_after"])
        for r in recs:
            if r.get("kind") == "megatron_forward_fp" and r["pid"] == primary and r.get("fp"):
                V[int(r["after_steps"])].add(r["fp"])
        for pid in emb_pids[1:]:
            other = [r["fp_after"] for r in by_pid[pid] if r.get("success")]
            mine = [r["fp_after"] for r in ok_steps]
            if other != mine:
                issue("M3", f"data-parallel replicas disagree on the weights after some step (pids {primary}, {pid})")
    init_fp = hf_fingerprint(a.initial_hf)
    if primary is not None:
        if not V.get(0):
            issue("M3", "no fingerprint of the weights the first training forward used")
        elif init_fp not in V[0] or len(V[0]) != 1:
            issue("M3", "Megatron did not start from the pinned initial student weights")
    final_fp = hf_fingerprint(a.final_hf) if Path(a.final_hf).is_dir() else None
    if final_fp is None:
        issue("M3", "final checkpoint has no readable input embedding")
    elif primary is not None and final_fp not in V.get(N, set()):
        issue("M3", f"the final checkpoint is not Megatron's weights after step {N - 1}")
    # nothing between two optimizer steps may change the student's weights or the optimizer's state: each step
    # enters with exactly the state fingerprint the previous one returned with (same random positions)
    for pid, rs in by_pid.items():
        for prev, cur in zip(rs, rs[1:]):
            left, found = prev.get("state_exit"), cur.get("state_entry")
            if not (isinstance(left, str) and isinstance(found, str)):
                why = prev.get("state_error") or cur.get("state_error")
                issue("M3", f"training process {pid}: no state fingerprint between optimizer steps {prev['attempt']} "
                            f"and {cur['attempt']}" + (f" ({why})" if why else ""))
            elif left != found:
                issue("M3", f"training process {pid}: the student's weights or the optimizer's state changed between "
                            f"optimizer steps {prev['attempt']} and {cur['attempt']}")

    # when the trainer was in a training step: from the student's first grad-enabled forward after the previous step
    # (else the optimizer step's own start) to the step's return. No evaluator interval is taken out of T there
    fwd_at = defaultdict(list)
    for r in recs:
        if r.get("kind") == "megatron_forward_fp" and r.get("pid") == primary and isinstance(r.get("t"), (int, float)):
            fwd_at[r.get("after_steps")].append(r["t"])
    busy = []
    for s, r in enumerate(x for x in by_pid.get(primary, []) if x.get("success")):
        t0 = min((v for v in fwd_at[s] + [r.get("t_start")] if isinstance(v, (int, float))), default=None)
        t1 = r.get("t_end")
        if all(isinstance(v, (int, float)) and math.isfinite(v) for v in (t0, t1)):
            busy.append([t0, t1])

    report = {"mode": a.mode, "steps_recorded": {str(p): len(v) for p, v in by_pid.items()},
              "training_processes": len(by_pid)}

    student_engine = lambda r: r.get("model_path") == student_path
    if a.mode == "run":
        disp = [r for r in recs if r.get("kind") == "dispatch" and student_engine(r)
                and not str(r.get("rid", "")).startswith(HEALTH) and (r.get("sampling") or {}).get("max_new_tokens") != 0]
        t_first = min((r["t"] for r in disp), default=float("inf"))
        engine_pids = sorted({r["pid"] for r in disp})
        pubs = defaultdict(list)
        for r in recs:
            if r.get("kind") == "sglang_publish" and student_engine(r):
                pubs[r["pid"]].append(r)
        if not pubs:
            issue("M3", "no publication was recorded inside the student engine")
        checked, after_by_pid = 0, {}
        for pid, rs in sorted(pubs.items()):
            rs.sort(key=lambda r: r["t"])
            failed = [r["b"] for r in rs if r.get("success") is False]
            if failed:
                issue("M3", f"engine {pid}: publications {failed[:5]} failed inside the engine")
            ok = [r for r in rs if r.get("success") is not False]
            before = [r for r in ok if r["t"] <= t_first]
            after = after_by_pid[pid] = [r for r in ok if r["t"] > t_first]
            if primary is not None and any(r.get("fp") not in V.get(0, set()) for r in before):
                issue("M3", f"engine {pid}: served weights before the first training generation are not the initial student")
            if len(after) != N:
                issue("M3", f"engine {pid}: {len(after)} publications after the first training generation, need {N}")
            for i, r in enumerate(after[:N], 1):
                checked += 1
                if primary is not None and r.get("fp") not in V.get(i, set()):
                    issue("M3", f"engine {pid}: publication {i} served weights that are not Megatron's weights after step {i - 1}")
        report["publications_checked"] = checked

        # ------------------------------------------------------------ M4 snapshots
        snaps, excl, per_engine, timing = [], [], defaultdict(int), []

        def take_out(what, x):
            """Take evaluator interval record `x` (already placed and bounded) out of T, as credited."""
            parts = credited(x["t_start"], x["t_end"], busy)
            excl.extend(parts)
            timing.append({"what": what, "t_start": x["t_start"], "seconds": round(x["t_end"] - x["t_start"], 3),
                           "credited_seconds": round(sum(hi - lo for lo, hi in parts), 3)})

        for r in recs:
            if r.get("kind") != "sglang_snapshot" or not student_engine(r):
                continue
            per_engine[r.get("pid")] += 1
            if per_engine[r.get("pid")] == N_SNAPSHOTS + 1:
                issue("M4", f"engine {r.get('pid')}: more than the hook's {N_SNAPSHOTS} snapshots")
            # anchored to its own engine's publications, before one the hook copies at (publications 2..steps-1): a
            # record for a process that published nothing, or placed anywhere else in the window, is not a copy the
            # hook took and is never taken out of T
            version = snapshot_version(r, pubs.get(r.get("pid"), []), t_first)
            if not (isinstance(r.get("b"), int) and 2 <= r["b"] <= N - 1):
                version = None
            lasts = version is not None and r["t_end"] - r["t_start"] <= EXCLUDE_MAX_SECONDS
            if version is not None and not lasts:
                issue("M4", f"snapshot before publication {r['b']}: the copy [{r['t_start']}, {r['t_end']}] does not "
                            f"last 0-{EXCLUDE_MAX_SECONDS:g} s")
            ks = sorted(k for k, fps in V.items() if r.get("fp") in fps)
            if lasts and version in ks:
                take_out(f"engine {r['pid']} copy before publication {r['b']}", r)
            path = r.get("path")
            present = isinstance(path, str) and os.path.isfile(path) and os.path.getsize(path) > 0
            ent = {"b": r.get("b"), "path": path, "present": present, "versions": ks, "served_version": version,
                   "seconds": round(r["t_end"] - r["t_start"], 3) if version is not None else None}
            if version is None:
                issue("M4", f"engine {r.get('pid')}: snapshot before publication {r.get('b')} is not a copy the hook "
                            f"takes (before a publication in 2..{N - 1} of that engine, after its previous one)")
            elif version not in ks:
                issue("M4", f"snapshot before publication {r['b']} does not match the version {version} its engine "
                            "served then")
            elif lasts and 1 <= version <= N - 1 and ks == list(range(ks[0], ks[-1] + 1)) and present:
                ent["step"] = version
            snaps.append(ent)
        usable = sorted({s["step"] for s in snaps if "step" in s})
        skipped = [r.get("reason") for r in recs if r.get("kind") == "sglang_snapshot_skipped" and student_engine(r)]
        if not usable:
            issue("M4", "no usable copy of the served weights at a random step"
                        + (f" ({skipped[0]}; student engines must use tensor parallel size 1)" if skipped else ""))
        report["snapshots"] = snaps
        report["snapshot_steps"] = usable

        # ------------------------------------------------------------ M6 final weights
        def m6(who, problems):
            for msg in problems[:10] + ([f"... {len(problems) - 10} more"] if len(problems) > 10 else []):
                issue("M6", f"{who}: {msg}")

        have_final = Path(a.final_hf).is_dir()
        # the hook takes one digest per training process and one per engine publication
        seen = defaultdict(int)
        for r in recs:
            if r.get("kind") in ("megatron_digest", "sglang_digest"):
                seen[r["kind"], r.get("pid"), r.get("after_steps"), r.get("b")] += 1
        for (kind, pid, steps, b), n in sorted(seen.items(), key=str):
            if n > 1:
                issue("M6", f"process {pid}: {n} {kind} records for " + (f"step {steps}" if b is None else f"publication {b}"))
        # the trainer's weights right after its last step (the hook digests them before the step returns)
        tdig = [r for r in recs
                if r.get("kind") == "megatron_digest" and r.get("pid") in by_pid and r.get("after_steps") == N]
        for x in tdig:
            ok_steps = [r for r in by_pid[x["pid"]] if r.get("success")]
            why = (interval_issue(x, ok_steps[-1].get("t_end")) if len(ok_steps) == N
                   else "comes from a process that did not take the run's steps")
            if why:
                issue("M6", f"training process {x['pid']}: digest after step {N - 1} {why}")
            else:
                take_out(f"training process {x['pid']} digest after step {N - 1}", x)
        mine = [x for x in tdig if x["pid"] == primary]
        if primary is not None:
            if not mine or not mine[-1].get("tensors"):
                why = [r.get("reason") for r in recs if r.get("kind") == "megatron_digest_skipped" and r.get("pid") == primary]
                issue("M6", f"training process {primary}: no full digest of Megatron's weights after step {N - 1}"
                            + (f" ({why[-1]})" if why else ""))
            else:
                m6(f"training process {primary}",
                   trainer_weight_issues(a.final_hf, mine[-1]["tensors"], attention_heads(a.initial_hf))
                   if have_final else ["no final checkpoint"])
        report["trainer_digest_checked"] = bool(mine)

        # the weights each student engine served after its last publication
        digests = {(r["pid"], r.get("b")): r for r in recs if r.get("kind") == "sglang_digest" and student_engine(r)}
        final_checked = 0
        for pid, after in sorted(after_by_pid.items()):
            if len(after) != N:
                continue  # M3 reports the count
            # the hook digests after every publication b >= rounds - 1: the last one, and those before it that can be
            # the last (one more per initial push). Those inside the timed window are the evaluator's, like the copies;
            # each ends before the engine's next publication completes (the hook digests before returning)
            for i, r in enumerate(after):
                x = digests.get((pid, r.get("b")))
                if x is None or not isinstance(r.get("b"), int) or r["b"] < N - 1:
                    continue
                why = interval_issue(x, r["t"], after[i + 1]["t"] if i + 1 < len(after) else None)
                if why:
                    issue("M6", f"engine {pid}: digest after publication {r['b']} {why}")
                else:
                    take_out(f"engine {pid} digest after publication {r['b']}", x)
            d = digests.get((pid, after[-1]["b"]))
            if not d or not d.get("tensors"):
                why = [r.get("reason") for r in recs if r.get("kind") == "sglang_digest_skipped" and r.get("pid") == pid]
                issue("M6", f"engine {pid}: no full digest of the weights served after its last publication "
                            f"{after[-1]['b']}" + (f" ({why[-1]})" if why else ""))
                continue
            m6(f"engine {pid}", final_weight_issues(a.final_hf, d["tensors"]) if have_final else ["no final checkpoint"])
            final_checked += 1
        report["final_digest_engines"] = final_checked
        report["exclude_intervals"] = sorted(excl)
        report["evaluator_intervals"] = sorted(timing, key=lambda x: x["t_start"])

        # ------------------------------------------------------------ M5 engine parent
        parents = {}
        for r in recs:
            if r.get("kind") == "hook_installed" and r.get("module") and r["pid"] in engine_pids:
                parents[r["pid"]] = r.get("parent_title")
        for pid in engine_pids:
            t = parents.get(pid)
            if t is None:
                issue("M5", f"engine process {pid}: no record of the process that started it")
            elif not t.startswith("ray::"):
                issue("M5", f"engine process {pid} was started by {t[:80]!r}, not by a Ray actor (Miles' SGLangEngine)")
        report["engine_parents"] = {str(k): v for k, v in parents.items()}

    # ---------------------------------------------------------------- per-step evidence for the oracles
    grads, norms = defaultdict(list), {}
    for pid, rs in by_pid.items():
        for k, r in enumerate([x for x in rs if x.get("success")]):
            if r.get("grad"):
                grads[k].append(r["grad"])
            if pid == primary and r.get("grad_norm") is not None:
                norms[k] = r["grad_norm"]
    if attempts and not grads:
        issue("M1", "no gradient record on the training side")
    report["grad_records"] = {str(k): v for k, v in sorted(grads.items())}
    report["grad_norms"] = {str(k): v for k, v in sorted(norms.items())}
    models = [r for r in recs if r.get("kind") == "megatron_model" and r["pid"] in by_pid and r.get("pre_process") is not False]
    cfg = (models[-1].get("config") or {}) if models else {}
    report["model_config"] = {k: cfg.get(k) for k in NUMERIC_CONFIG}
    report["model_config_all"] = cfg
    if attempts and not models:
        issue("M1", "no Megatron-Core GPTModel was built in the training process")

    report["issues"] = {k: v[:20] + ([f"... {len(v) - 20} more"] if len(v) > 20 else []) for k, v in issues.items()}
    report["issue_counts"] = {k: len(v) for k, v in issues.items()}
    report["passed"] = not issues
    text = json.dumps(report, indent=2)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
