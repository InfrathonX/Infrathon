#!/usr/bin/env python3
"""Math accuracy eval for OPD v2 with an SGLang offline engine (no Miles code), N repeats in one engine.

The sampling and seeding protocol:
chat template with ``enable_thinking=False``, one sample per prompt per repeat, per-request
``sampling_seed = (seed + r) * 1_000_003 + row_index`` with deterministic inference, radix
cache flushed between repeats. Grading is ``math_grader.grade``
(last ``\\boxed{}`` OR last ``Answer:`` line).

``--view-caps`` additionally grades each response truncated to its first K tokens (K below
``--max-new-tokens``), which is the response a K-token cap would have produced when generation
is prefix-consistent (deterministic inference, position-seeded sampling); check that with a real
K-token run (``--save-ids`` on both, compare).

``--dp N`` splits every set over N engines, one per visible GPU (row i goes to engine i mod N), each in
its own process; the per-request seed still uses the global row index, so with deterministic inference the
responses are the same as one engine's. Grading
and the report are done once, in the parent.

Output: one JSON object on stdout; ``--out`` adds per-repeat per-item results.
"""

import argparse
import gzip
import hashlib
import json
import os
import statistics
import subprocess
import tempfile
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from math_grader.grade import grade  # noqa: E402

SEED_STRIDE = 1_000_003


def build_prompts(tok, rows):
    return [
        tok.encode(
            tok.apply_chat_template(list(r["content"]), tokenize=False, add_generation_prompt=True, enable_thinking=False),
            add_special_tokens=False,
        )
        for r in rows
    ]


def score(texts, rows, n_tokens, cap):
    graded = [grade(t, r["answer"]) for t, r in zip(texts, rows)]
    correct = [c for c, _ in graded]
    return {
        "accuracy": sum(correct) / len(rows),
        "correct": sum(correct),
        "grade_timeouts": sum(t for _, t in graded),
        "boxed_rate": sum("\\boxed{" in t for t in texts) / len(rows),
        "answer_line_rate": sum("answer:" in t.lower() for t in texts) / len(rows),
        "mean_completion_tokens": sum(n_tokens) / len(n_tokens),
        "median_completion_tokens": statistics.median(n_tokens),
        "truncated_rate": sum(n >= cap for n in n_tokens) / len(n_tokens),
        "correct_sha256": hashlib.sha256(bytes(correct)).hexdigest(),
    }, correct


def generate(args, sets, shard=0, nshards=1):
    """Responses of this shard's rows (row i with i % nshards == shard), every set and repeat, in one engine:
    {name: [{"rows": [i...], "texts": [...], "tokens": [...], "ids": [...], "seconds": s} per repeat]}."""
    import sglang as sgl

    t0 = time.time()
    engine = sgl.Engine(model_path=args.model, tokenizer_path=args.tokenizer or args.model,
                        mem_fraction_static=args.mem_fraction, tp_size=1, log_level="warning",
                        random_seed=args.seed, enable_deterministic_inference=True)
    out = {"engine_start_seconds": time.time() - t0, "sets": {}}
    first = True
    for name, (rows, prompts) in sets.items():
        idx = [i for i in range(len(prompts)) if i % nshards == shard]
        reps = []
        for r in range(args.repeats):
            if not first:
                engine.flush_cache()
            first = False
            seed = args.seed + r
            sp = [{"temperature": args.temperature, "top_p": args.top_p, "max_new_tokens": args.max_new_tokens,
                   "sampling_seed": seed * SEED_STRIDE + i} for i in idx]
            t1 = time.time()
            outs = engine.generate(input_ids=[prompts[i] for i in idx], sampling_params=sp) if idx else []
            reps.append({"rows": idx, "texts": [o["text"] for o in outs],
                         "tokens": [o["meta_info"]["completion_tokens"] for o in outs],
                         "ids": [list(o["output_ids"]) for o in outs] if (args.save_ids or args.view_caps) else None,
                         "seconds": time.time() - t1})
            print(f"[shard {shard}/{nshards}] {name} repeat {r} rows={len(idx)} gen={reps[-1]['seconds']:.0f}s",
                  file=sys.stderr, flush=True)
        out["sets"][name] = reps
    engine.shutdown()
    return out


def generate_sharded(args, sets):
    """--dp N: one worker process per GPU (the parent's CUDA_VISIBLE_DEVICES, else 0..N-1)."""
    devices = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()] or [str(i) for i in range(args.dp)]
    if len(devices) < args.dp:
        raise SystemExit(f"--dp {args.dp} needs {args.dp} visible GPUs, have {devices}")
    with tempfile.TemporaryDirectory() as tmp:
        procs = []
        for k in range(args.dp):
            cmd = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker", str(k),
                   "--worker-out", f"{tmp}/{k}.json"]
            procs.append(subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=devices[k])))
        codes = [p.wait() for p in procs]
        if any(codes):
            raise SystemExit(f"eval workers exited {codes}")
        parts = [json.loads(Path(f"{tmp}/{k}.json").read_text()) for k in range(args.dp)]
    merged = {"engine_start_seconds": max(p["engine_start_seconds"] for p in parts), "sets": {}}
    for name, (rows, _) in sets.items():
        reps = []
        for r in range(args.repeats):
            texts, tokens, ids = [None] * len(rows), [None] * len(rows), [None] * len(rows)
            for part in parts:
                rep = part["sets"][name][r]
                for j, i in enumerate(rep["rows"]):
                    texts[i], tokens[i] = rep["texts"][j], rep["tokens"][j]
                    ids[i] = rep["ids"][j] if rep["ids"] is not None else None
            assert all(t is not None for t in texts), f"{name} repeat {r}: rows missing from the workers"
            reps.append({"rows": list(range(len(rows))), "texts": texts, "tokens": tokens,
                         "ids": ids if ids and ids[0] is not None else None,
                         "seconds": max(part["sets"][name][r]["seconds"] for part in parts)})
        merged["sets"][name] = reps
    return merged


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--test", action="append", required=True, help="NAME=path.jsonl (repeatable)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, required=True)
    ap.add_argument("--view-caps", type=int, nargs="*", default=[])
    ap.add_argument("--mem-fraction", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dp", type=int, default=1, help="engines (one per visible GPU); rows split by index")
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-ids", default=None, help="gzip json of output token ids per set and repeat")
    ap.add_argument("--worker", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--worker-out", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    t_start = time.time()
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    sets = {}
    for spec in args.test:
        name, path = spec.split("=", 1)
        rows = [json.loads(line) for line in open(path, encoding="utf-8")][: args.limit]
        sets[name] = (rows, build_prompts(tok, rows))
    if args.worker is not None:
        Path(args.worker_out).write_text(json.dumps(generate(args, sets, args.worker, args.dp)))
        return
    gen = generate_sharded(args, sets) if args.dp > 1 else generate(args, sets)
    report = {"model": args.model, "max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
              "top_p": args.top_p, "repeats": args.repeats, "seed": args.seed, "dp": args.dp,
              "engine_start_seconds": gen["engine_start_seconds"], "sets": {}}
    detail, ids_dump = {}, {}
    for name, (rows, _) in sets.items():
        per_rep, views = [], {k: [] for k in args.view_caps}
        detail[name], ids_dump[name] = [], []
        for r, g in enumerate(gen["sets"][name]):
            assert g["rows"] == list(range(len(rows)))
            texts, n_tok = g["texts"], g["tokens"]
            s, correct = score(texts, rows, n_tok, args.max_new_tokens)
            s.update(repeat=r, seed=args.seed + r, generation_seconds=g["seconds"],
                     responses_sha256=hashlib.sha256(json.dumps(texts).encode()).hexdigest())
            per_rep.append(s)
            detail[name].append({"seed": args.seed + r, "correct": correct, "tokens": n_tok})
            if args.save_ids:
                ids_dump[name].append(g["ids"])
            for k in args.view_caps:
                vt = [tok.decode(x[:k], skip_special_tokens=True) for x in g["ids"]]
                vs, _ = score(vt, rows, [min(n, k) for n in n_tok], k)
                views[k].append(vs)
            print(f"{name} repeat {r} acc={s['accuracy']:.4f} tok={s['mean_completion_tokens']:.0f} "
                  f"trunc={s['truncated_rate']:.3f} gen={g['seconds']:.0f}s", file=sys.stderr, flush=True)
        agg = lambda reps, k: sum(x[k] for x in reps) / len(reps)
        entry = {"n": len(rows), "accuracy": agg(per_rep, "accuracy"),
                 "accuracy_per_repeat": [x["accuracy"] for x in per_rep],
                 "mean_completion_tokens": agg(per_rep, "mean_completion_tokens"),
                 "truncated_rate": agg(per_rep, "truncated_rate"), "boxed_rate": agg(per_rep, "boxed_rate"),
                 "answer_line_rate": agg(per_rep, "answer_line_rate"),
                 "generation_seconds": sum(x["generation_seconds"] for x in per_rep), "per_repeat": per_rep,
                 "views": {str(k): {"accuracy": agg(v, "accuracy"), "accuracy_per_repeat": [x["accuracy"] for x in v],
                                    "truncated_rate": agg(v, "truncated_rate"),
                                    "mean_completion_tokens": agg(v, "mean_completion_tokens"), "per_repeat": v}
                           for k, v in views.items()}}
        report["sets"][name] = entry
    report["wall_seconds"] = time.time() - t_start
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps({**report, "detail": detail}) + "\n")
    if args.save_ids:
        with gzip.open(args.save_ids, "wt") as f:
            json.dump(ids_dump, f)


if __name__ == "__main__":
    main()
