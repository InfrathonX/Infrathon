#!/usr/bin/env python3
"""Rebuild the upstream Miles DAPO-Math-17k prompt file and split it for the OPD v2 task.

Upstream Miles trains and evaluates on ``zhuzilin/dapo-math-17k`` (``dapo-math-17k.jsonl``,
17,398 rows, no license declared). That file is the Apache-2.0
``BytedTsinghua-SIA/DAPO-Math-17k`` parquet (1,791,700 rows: each problem about 100 times)
deduplicated on prompt text (first occurrence kept) with one instruction sentence changed
(``Answer: $Answer (without quotes)`` -> ``Answer: \\boxed{$Answer}``). This script rebuilds it
from the Apache-2.0 parquet, checks it against the zhuzilin file when given, and then applies
the split of Miles' self-distillation example (``make_split.py``: dedup on prompt text, then
``random.Random(42).shuffle`` and the LAST ``eval_n`` rows held out). Holding out the last 2,000
rows of the same permutation makes the hidden set a superset of upstream's 512-row eval.

Output rows: {"row_id": "dapo:<dedup index>", "content": [{"role": "user", ...}], "answer": label}.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

OLD = "The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem."
NEW = "The last line of your response should be of the form Answer: \\boxed{$Answer} where $Answer is the answer to the problem."
UPSTREAM_SPLIT_SEED = 42
UPSTREAM_EVAL_N = 512


def dedup_rows(parquet):
    import pyarrow.parquet as pq

    t = pq.read_table(parquet, columns=["prompt", "reward_model"]).to_pydict()
    seen, rows, labels = set(), [], {}
    for prompt, rm in zip(t["prompt"], t["reward_model"]):
        assert len(prompt) == 1 and prompt[0]["role"] == "user"
        c = prompt[0]["content"]
        labels.setdefault(c, set()).add(rm["ground_truth"])
        if c in seen:
            continue
        seen.add(c)
        assert c.count(OLD) == 1
        rows.append({"content": c.replace(OLD, NEW), "answer": rm["ground_truth"]})
    ambiguous = {c.replace(OLD, NEW) for c, s in labels.items() if len(s) > 1}
    return rows, ambiguous


def check_zhuzilin(rows, path):
    z = [json.loads(line) for line in open(path, encoding="utf-8")]
    assert len(z) == len(rows), (len(z), len(rows))
    for r, zr in zip(rows, z):
        assert len(zr["prompt"]) == 1 and zr["prompt"][0]["content"] == r["content"] and zr["label"] == r["answer"]


def write_jsonl(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for i, r in items:
            f.write(json.dumps({"row_id": f"dapo:{i}", "content": [{"role": "user", "content": r["content"]}],
                                "answer": r["answer"]}, ensure_ascii=False) + "\n")


def split(rows, hidden_n):
    order = list(range(len(rows)))
    random.Random(UPSTREAM_SPLIT_SEED).shuffle(order)  # the permutation upstream applies to its rows
    return order[:-hidden_n], order[-hidden_n:]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--zhuzilin", type=Path, default=None, help="zhuzilin/dapo-math-17k jsonl to check against")
    ap.add_argument("--math500", type=Path, required=True, help="prm800k math_splits/test.jsonl")
    ap.add_argument("--hidden-n", type=int, default=2000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    rows, ambiguous = dedup_rows(args.parquet)
    if args.zhuzilin:
        check_zhuzilin(rows, args.zhuzilin)
    train_idx, hidden_idx = split(rows, args.hidden_n)
    upstream_eval = set(hidden_idx[-UPSTREAM_EVAL_N:])
    assert {rows[i]["content"] for i in train_idx}.isdisjoint({rows[i]["content"] for i in hidden_idx})
    write_jsonl(args.out / "train_pool.jsonl", [(i, rows[i]) for i in train_idx])
    write_jsonl(args.out / "hidden.jsonl", [(i, rows[i]) for i in hidden_idx])

    m500 = [json.loads(line) for line in open(args.math500, encoding="utf-8")]
    with open(args.out / "math500.jsonl", "w", encoding="utf-8") as f:
        for r in m500:  # the Qwen math-eval instruction, so MATH-500 can be compared with the Qwen3 report
            q = r["problem"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."
            f.write(json.dumps({"row_id": f"math500:{r['unique_id']}", "content": [{"role": "user", "content": q}],
                                "answer": r["answer"], "subject": r["subject"], "level": r["level"]},
                               ensure_ascii=False) + "\n")

    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = {
        "rows_dedup": len(rows), "train_pool": len(train_idx), "hidden": len(hidden_idx), "math500": len(m500),
        "split_seed": UPSTREAM_SPLIT_SEED, "upstream_eval_rows_in_hidden": len(upstream_eval),
        "ambiguous_label_prompts": len(ambiguous),
        "ambiguous_in_hidden": sum(rows[i]["content"] in ambiguous for i in hidden_idx),
        "zhuzilin_checked": bool(args.zhuzilin),
        "files": {p.name: sha(p) for p in sorted(args.out.glob("*.jsonl"))},
        "sources": {"parquet": sha(args.parquet), "math500": sha(args.math500)},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
