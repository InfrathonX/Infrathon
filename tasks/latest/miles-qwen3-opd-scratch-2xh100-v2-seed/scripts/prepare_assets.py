#!/usr/bin/env python3
"""Build the public and private data assets of the Miles OPD v2 (math) task (run in the task image).

Input: the split directory of scripts/miles_opd/v2/make_split.py (the upstream Miles DAPO-Math-17k
prompt file rebuilt from the Apache-2.0 BytedTsinghua-SIA/DAPO-Math-17k parquet, deduplicated on
prompt text, permuted with upstream's split seed 42; the last 2,000 rows held out, a superset of
upstream's 512-row eval) and MATH-500 (openai/prm800k, MIT).

public/   the data in the Judge image's /data/public; scripts/make_work_data.py derives the Work image's data from it
  train.jsonl          rounds x prompts_per_round training rows: the first R*N train_pool rows that do not
                       repeat a hidden or earlier row up to case and whitespace; round r uses lines [N r, N r + N)
  train_public.jsonl   train_pool lines [R*N, R*N + 64) (public small run: 4 rounds x 16)
  test_public.jsonl    the next --public-test lines (indicative accuracy only)
  probe_batch.json     8 fixed trajectories from train_public rows for the public one-step probe
private/  Judge only
  eval.jsonl           the 2,000 held-out DAPO rows (scored)
  math500.jsonl        MATH-500 with the Qwen math instruction (reported, not scored)
  probe_batch.json     8 fixed trajectories from train_pool rows no Work file contains, for the Judge probe
Rows: {row_id, content: [{role: user, content}], answer}. Every directory gets manifest.json with the row
ranges and the SHA-256 of every file.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

PUBLIC_SMALL = 64
PROBE_N = 8


def read_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(r):
    return " ".join(r["content"][-1]["content"].split()).lower()


def probe_batch(model, rows, max_new_tokens, seed):
    import sglang as sgl
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    prompts = [tok.encode(tok.apply_chat_template(r["content"], tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False), add_special_tokens=False) for r in rows]
    engine = sgl.Engine(model_path=model, mem_fraction_static=0.5, tp_size=1, log_level="warning", random_seed=seed)
    outs = engine.generate(input_ids=prompts, sampling_params={"temperature": 1.0, "top_p": 1.0, "max_new_tokens": max_new_tokens})
    engine.shutdown()
    return {"lr": 1e-3, "samples": [{"row_id": r["row_id"], "prompt_ids": p, "response_ids": list(o["output_ids"])}
                                    for r, p, o in zip(rows, prompts, outs)]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", type=Path, required=True, help="make_split.py output (train_pool, hidden, math500)")
    ap.add_argument("--student", required=True)
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--per-round", type=int, required=True)
    ap.add_argument("--public-test", type=int, default=256)
    ap.add_argument("--probe-max-new-tokens", type=int, default=1024)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    pub, priv = args.out / "public", args.out / "private"
    pub.mkdir(parents=True, exist_ok=False)
    priv.mkdir(parents=True, exist_ok=False)

    raw_pool = read_jsonl(args.split / "train_pool.jsonl")
    hidden = read_jsonl(args.split / "hidden.jsonl")
    # upstream deduplicates on the exact prompt text only; 52 pool rows differ from a hidden row in case or
    # whitespace alone. Skip them (and such repeats inside the pool) so no Work file holds a hidden problem.
    seen, pool, skipped = {key(r) for r in hidden}, [], []
    for r in raw_pool:
        (skipped if key(r) in seen else pool).append(r)
        seen.add(key(r))
    n_train = args.rounds * args.per_round
    cut = [0, n_train, n_train + PUBLIC_SMALL, n_train + PUBLIC_SMALL + args.public_test]
    train, small, test = (pool[cut[i]:cut[i + 1]] for i in range(3))
    unshipped = pool[cut[3]:]
    assert len(test) == args.public_test and len(unshipped) >= PROBE_N
    visible = {key(r) for r in train + small + test}
    assert visible.isdisjoint(key(r) for r in hidden), "a hidden row is visible in Work"
    write_jsonl(pub / "train.jsonl", train)
    write_jsonl(pub / "train_public.jsonl", small)
    write_jsonl(pub / "test_public.jsonl", test)
    write_jsonl(priv / "eval.jsonl", hidden)
    (priv / "math500.jsonl").write_bytes((args.split / "math500.jsonl").read_bytes())

    (pub / "probe_batch.json").write_text(json.dumps(probe_batch(args.student, small[:PROBE_N], args.probe_max_new_tokens, 11), indent=1))
    (priv / "probe_batch.json").write_text(json.dumps(probe_batch(args.student, unshipped[:PROBE_N], args.probe_max_new_tokens, 23), indent=1))

    # ranges index the filtered pool (train_pool minus the skipped near-duplicates, in order)
    ranges = {"train.jsonl": [cut[0], cut[1]], "train_public.jsonl": [cut[1], cut[2]], "test_public.jsonl": [cut[2], cut[3]],
              "private_probe_rows": [cut[3], cut[3] + PROBE_N]}
    split_manifest = json.loads((args.split / "manifest.json").read_text())
    for d in (pub, priv):
        files = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(d.iterdir()) if p.name != "manifest.json"}
        (d / "manifest.json").write_text(json.dumps({
            "schema_version": 2, "files": files, "train_pool_line_ranges": ranges,
            "split": {k: split_manifest[k] for k in ("rows_dedup", "train_pool", "hidden", "split_seed",
                                                      "upstream_eval_rows_in_hidden", "files", "sources")},
            "probe_max_new_tokens": args.probe_max_new_tokens,
            "train_pool_rows_skipped_as_near_duplicates": [r["row_id"] for r in skipped]}, indent=2) + "\n")
    print(json.dumps({"public": sorted(os.listdir(pub)), "private": sorted(os.listdir(priv)), "ranges": ranges,
                      "unshipped_pool_rows": len(unshipped)}, indent=2))


if __name__ == "__main__":
    main()
