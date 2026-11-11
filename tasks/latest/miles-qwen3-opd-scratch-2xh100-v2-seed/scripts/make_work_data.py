#!/usr/bin/env python3
"""Build the data of the Work image from the task's assets (host tool, no GPU).

The public suites only get a pipeline working, so Work holds 256 training rows instead of the Judge's 4096: the 64
rows of train_public.jsonl (the small suite) and 192 more, drawn with a fixed seed from the train pool rows no asset
uses, skipping any problem that repeats a problem of the assets or an earlier draw (without case, punctuation or
layout: this also keeps near-duplicates out). The public test problems that repeat a scored problem in that sense are
replaced in place, the same way. The Judge's data does not change.

  OUT/train_public.jsonl   the 256 rows; the perf suite uses them all (8 rounds x 32), the small suite the first 64
  OUT/test_public.jsonl    the 256 public test problems, with those replacements
  OUT/probe_batch.json     copied from the public assets
  OUT/manifest.json        the SHA-256 of these files, and nothing about the split or the hidden files

  python make_work_data.py --split SPLIT_DIR --assets ASSET_DIR --out OUT

SPLIT_DIR holds the make_split.py train_pool.jsonl; ASSET_DIR holds the public/ and private/ of prepare_assets.py.
"""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

SMALL, ADDED, SEED = 64, 192, 20261106


def read_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def stem(row):
    """A row's problem without case, punctuation or layout: near-duplicates share it."""
    return " ".join(re.sub(r"[\W_]+", " ", row["content"][0]["content"].lower()).split())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--assets", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    pub, priv = a.assets / "public", a.assets / "private"
    manifest = json.loads((pub / "manifest.json").read_text())
    pool_file = a.split / "train_pool.jsonl"
    assert sha(pool_file) == manifest["split"]["files"]["train_pool.jsonl"], "train_pool.jsonl is not the task's split"
    pool = read_jsonl(pool_file)
    train, small, test = (read_jsonl(pub / f"{n}.jsonl") for n in ("train", "train_public", "test_public"))
    hidden, math500 = read_jsonl(priv / "eval.jsonl"), read_jsonl(priv / "math500.jsonl")
    assert len(small) == SMALL
    by_id = {r["row_id"]: r for r in pool}
    probes = {name: [by_id[s["row_id"]] for s in json.loads((d / "probe_batch.json").read_text())["samples"]]
              for name, d in (("public", pub), ("private", priv))}

    groups = (train, small, test, hidden, math500, probes["public"], probes["private"])
    used = {r["row_id"] for g in groups for r in g}
    seen = {stem(r) for g in groups for r in g}
    candidates = [r for r in pool if r["row_id"] not in used]
    random.Random(SEED).shuffle(candidates)
    added = []
    for r in candidates:
        if stem(r) not in seen:
            seen.add(stem(r))
            added.append(r)
            if len(added) == ADDED:
                break
    assert len(added) == ADDED, "not enough unused problems"
    hidden_stems = {stem(r) for r in hidden}
    leaked = [i for i, r in enumerate(test) if stem(r) in hidden_stems]
    replacements = []
    for r in candidates:
        if len(replacements) == len(leaked):
            break
        if stem(r) not in seen:
            seen.add(stem(r))
            replacements.append(r)
    assert len(replacements) == len(leaked), "not enough unused problems"
    work_test = list(test)
    for i, r in zip(leaked, replacements):
        work_test[i] = r

    a.out.mkdir(parents=True, exist_ok=False)
    with open(a.out / "train_public.jsonl", "w", encoding="utf-8") as f:
        for r in small + added:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(a.out / "test_public.jsonl", "w", encoding="utf-8") as f:
        for r in work_test:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (a.out / "probe_batch.json").write_bytes((pub / "probe_batch.json").read_bytes())
    files = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(a.out.iterdir())}
    (a.out / "manifest.json").write_text(json.dumps({"schema_version": 1, "files": files}, indent=2) + "\n")

    stems = lambda rows: {stem(r) for r in rows}
    visible = {"small": small, "test": work_test, "public_probe": probes["public"]}
    judge = {"train": train, "eval": hidden, "private_probe": probes["private"]}
    print(json.dumps({"pool": len(pool), "unused": len(candidates), "added": len(added), "test_replaced_lines": leaked,
                      "same_stem": {f"{v}_in_{j}": len(stems(visible[v]) & stems(judge[j])) for v in visible for j in judge}},
                     indent=1))


if __name__ == "__main__":
    main()
