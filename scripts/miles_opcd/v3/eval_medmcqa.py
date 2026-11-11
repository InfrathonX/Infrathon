#!/usr/bin/env python3
"""Standalone multiple-choice accuracy eval with an SGLang offline engine (no Miles code), N repeats in one engine.

Renders each test row with the Qwen chat template (``enable_thinking=False``), optionally with a
system-message context, and scores the last ``<answer>X</answer>`` tag case-insensitively (a
missing tag is wrong). The Judge and the public perf suite score the final student with it
(``--temperature 1 --top-p 1 --repeats 3 --seed 0 --seed-mode request``,
opcd_checks.quality_eval_cmd); it also bounded the OPCD diagnostics (student/teacher with and
without the context).

Repeats: one engine is started once and ``generate`` is called ``--repeats`` times over all
prompts. The radix (prefix) cache is flushed between passes so every pass starts from the
same cache state. Repeat ``i`` uses seed ``--seed + i``.

Seeding (``--seed-mode``):
  none     (default) no seed and no engine ``random_seed`` (greedy needs none).
  engine   ``sgl.Engine(random_seed=seed)`` (repeat i continues the same RNG stream); sampled
           tokens come from the global torch RNG, so they depend on batch composition and are
           NOT guaranteed reproducible.
  request  per-request ``sampling_seed = seed * 1_000_003 + row_index`` plus
           ``enable_deterministic_inference=True`` (batch-invariant kernels, pytorch sampler).
           SGLang only honours ``sampling_seed`` in deterministic mode; this is the
           reproducible sampled mode.
  auto     none if temperature == 0, else request.

Also: ``--top-k`` (default -1, SGLang's "no top-k") for the report-only paper-protocol diagnostic, and
``tagged_accuracy`` (correct / answers carrying a tag) next to ``answer_tag_rate``, so a gain from emitting
the tag is told apart from a gain in the tagged answers themselves.

Output: exactly ONE JSON object on stdout (the Judge's json_tail parses the first '{' to the
last '}'), progress lines go to stderr. Top-level ``accuracy`` is the mean over repeats, so
callers that read ``report["accuracy"]`` keep working; ``per_repeat`` holds each pass. Without
the repeat and seed options the command and its engine are the pre-v2 ones; the report keeps
every pre-v2 field and adds the repeat fields.
"""

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time

# Eager decode at >256 running requests grows the caching allocator to the card limit on 93.6 GiB cards.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
SEED_STRIDE = 1_000_003  # per-request seed = seed * SEED_STRIDE + row_index (row_index < SEED_STRIDE)


def parse_answer(text):
    m = _ANSWER_RE.findall(text or "")
    return m[-1].strip().strip("()[]").strip().upper() if m else ""


def resolve_seed_mode(seed_mode, temperature):
    if seed_mode == "auto":
        return "none" if temperature == 0 else "request"
    return seed_mode


def build_prompts(tok, rows, context=None):
    prompts = []
    for r in rows:
        msgs = list(r["content"])
        if context:
            msgs = [{"role": "system", "content": context}, *msgs]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompts.append(tok.encode(text, add_special_tokens=False))
    return prompts


def sampling_params_for(n, temperature, top_p, max_new_tokens, seed, seed_mode, top_k=-1):
    base = {"temperature": temperature, "top_p": top_p, "max_new_tokens": max_new_tokens}
    if top_k != -1:
        base["top_k"] = top_k
    if seed_mode != "request":
        return base
    return [{**base, "sampling_seed": seed * SEED_STRIDE + i} for i in range(n)]


def run_pass(engine, prompts, rows, temperature, top_p, max_new_tokens, seed, seed_mode, flush, top_k=-1):
    """One generate over all prompts; returns (per-repeat summary, predictions)."""
    if flush:
        engine.flush_cache()
    t0 = time.time()
    outs = engine.generate(
        input_ids=prompts,
        sampling_params=sampling_params_for(len(prompts), temperature, top_p, max_new_tokens, seed, seed_mode, top_k),
    )
    elapsed = time.time() - t0
    preds = [parse_answer(o["text"]) for o in outs]
    correct = sum(p == str(r["answer"]).strip().upper() for p, r in zip(preds, rows))
    lengths = [o["meta_info"]["completion_tokens"] for o in outs]
    summary = {
        "seed": seed if seed_mode != "none" else None,
        "accuracy": correct / len(rows),
        "correct": correct,
        "answer_tag_rate": sum(bool(p) for p in preds) / len(rows),
        "tagged_accuracy": correct / max(sum(bool(p) for p in preds), 1),
        "mean_completion_tokens": sum(lengths) / len(lengths),
        "truncated_rate": sum(n >= max_new_tokens for n in lengths) / len(lengths),
        "generation_seconds": elapsed,
        "predictions_sha256": hashlib.sha256(json.dumps(preds).encode()).hexdigest(),
    }
    return summary, preds


def aggregate(per_repeat, preds_all):
    accs = [r["accuracy"] for r in per_repeat]
    mean = lambda k: sum(r[k] for r in per_repeat) / len(per_repeat)
    diffs = [sum(a != b for a, b in zip(preds_all[0], p)) for p in preds_all[1:]]
    return {
        "accuracy": sum(accs) / len(accs),
        "accuracy_sd": statistics.stdev(accs) if len(accs) > 1 else None,
        "accuracy_min": min(accs),
        "accuracy_max": max(accs),
        "answer_tag_rate": mean("answer_tag_rate"),
        "tagged_accuracy": mean("tagged_accuracy"),
        "mean_completion_tokens": mean("mean_completion_tokens"),
        "truncated_rate": mean("truncated_rate"),
        "generation_seconds": sum(r["generation_seconds"] for r in per_repeat),
        "prediction_diffs_vs_repeat0": diffs,
        "identical_predictions_across_repeats": all(d == 0 for d in diffs),
    }


def make_engine(model, mem_fraction, seed, deterministic):
    """The engine; `seed` None starts it without a random_seed (the pre-v2 engine)."""
    import sglang as sgl

    kw = dict(model_path=model, mem_fraction_static=mem_fraction, tp_size=1, log_level="warning")
    if seed is not None:
        kw["random_seed"] = seed
    if deterministic:
        kw["enable_deterministic_inference"] = True
    return sgl.Engine(**kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None,
                    help="pinned tokenizer used to render prompts (default: --model). The Judge passes the pinned "
                         "snapshot so nothing in the scored checkpoint's tokenizer files can change the prompts")
    ap.add_argument("--test", required=True, help="jsonl with content (messages) and answer")
    ap.add_argument("--context-file", default=None, help="prepend this text as the system message")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=-1, help="-1 (default): no top-k truncation")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--mem-fraction", type=float, default=0.6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=1, help="generate passes over the whole set in one engine")
    ap.add_argument("--seed", type=int, default=0, help="repeat i uses seed + i")
    ap.add_argument("--seed-mode", choices=["auto", "none", "engine", "request"], default="none")
    ap.add_argument("--deterministic-inference", action="store_true",
                    help="force enable_deterministic_inference (implied by --seed-mode request)")
    ap.add_argument("--no-flush", action="store_true", help="keep the radix cache between passes")
    ap.add_argument("--out", default=None, help="also write the report plus the predictions (per repeat) here")
    args = ap.parse_args()
    if args.repeats < 1:
        ap.error("--repeats must be >= 1")

    from transformers import AutoTokenizer

    t_start = time.time()
    seed_mode = resolve_seed_mode(args.seed_mode, args.temperature)
    deterministic = args.deterministic_inference or seed_mode == "request"
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    context = open(args.context_file, encoding="utf-8").read().strip() if args.context_file else None
    rows = [json.loads(l) for l in open(args.test, encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    prompts = build_prompts(tok, rows, context)

    t_engine = time.time()
    engine = make_engine(args.model, args.mem_fraction, None if seed_mode == "none" else args.seed, deterministic)
    engine_start_seconds = time.time() - t_engine
    per_repeat, preds_all = [], []
    for i in range(args.repeats):
        s, p = run_pass(engine, prompts, rows, args.temperature, args.top_p, args.max_new_tokens,
                        args.seed + i, seed_mode, flush=(i > 0 and not args.no_flush), top_k=args.top_k)
        s["repeat"] = i
        if seed_mode == "engine":  # one engine RNG stream: pass i continues it, so record the engine seed
            s["seed"] = args.seed
        per_repeat.append(s)
        preds_all.append(p)
        print(f"eval repeat {i + 1}/{args.repeats} seed={s['seed']} acc={s['accuracy']:.5f} "
              f"gen={s['generation_seconds']:.1f}s", file=sys.stderr, flush=True)
    engine.shutdown()

    report = {
        "model": args.model,
        "context": bool(context),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "n": len(rows),
        "repeats": args.repeats,
        "seed": args.seed,
        "seed_mode": seed_mode,
        "deterministic_inference": deterministic,
        "mem_fraction": args.mem_fraction,
        **aggregate(per_repeat, preds_all),
        "engine_start_seconds": engine_start_seconds,
        "wall_seconds": time.time() - t_start,
        "per_repeat": per_repeat,
    }
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w") as f:  # one repeat: the pre-v2 flat list
            json.dump({**report, "predictions": preds_all if args.repeats > 1 else preds_all[0]}, f, indent=2)


if __name__ == "__main__":
    main()
