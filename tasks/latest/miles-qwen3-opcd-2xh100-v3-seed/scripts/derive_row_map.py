#!/usr/bin/env python3
"""One-time derivation of the OPCD v3 row map (host tool, run once; its output is committed).

The v2 assets were cut from ytz20/sys_medmcqa_{train,test} (no declared license). Every row of those
two tables is one MedMCQA training row (openlifescienceai/medmcqa, Apache-2.0) rendered with the LMOps
prompt template (TEMPLATE in prepare_assets.py): measured 2026-10-07, 182,322 + 500 rows, each matching
exactly one MedMCQA row on (rendered prompt, answer letter), none ambiguous, none missing.

This script records, for every row the v2 assets use, the index of its MedMCQA row in the pinned parquet
(revision 91c6572c, data/train-00000-of-00001.parquet), so v3 rebuilds the same
questions, in the same order and split, from MedMCQA alone. It needs the ytz20 tables and the v2 assets
only here, once.

  python derive_row_map.py --medmcqa DIR --ytz20 DIR --v2-assets DIR --out row_map.json
"""

import argparse
import collections
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from prepare_assets import MEDMCQA_REVISION, TRAIN_PARQUET, render


def ytz_rows(path):
    for row in pq.read_table(path).to_pylist():
        content = row["content"]
        content = json.loads(content) if isinstance(content, str) else content
        yield content[-1]["content"], str(row["answer"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--medmcqa", type=Path, required=True, help=f"snapshot dir of openlifescienceai/medmcqa @ {MEDMCQA_REVISION}")
    ap.add_argument("--ytz20", type=Path, required=True, help="dir with sys_medmcqa_train/ and sys_medmcqa_test/")
    ap.add_argument("--v2-assets", type=Path, required=True, help="the v2 (= v1) asset dir with public/ and private/")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    parquet = a.medmcqa / TRAIN_PARQUET
    mm = pq.read_table(parquet).to_pylist()
    index = collections.defaultdict(list)
    for i, r in enumerate(mm):
        index[render(r)].append(i)
    tables = {}
    for split in ("train", "test"):
        rows = list(ytz_rows(a.ytz20 / f"sys_medmcqa_{split}/data/train-00000-of-00001.parquet"))
        mapped = []
        for prompt, answer in rows:
            hits = [i for i in index.get((prompt, answer), [])]
            assert len(hits) == 1, f"{split}: {len(hits)} MedMCQA rows match {prompt[:80]!r}"
            mapped.append(hits[0])
        tables[split] = mapped

    def ids(path):
        out = []
        for line in open(path, encoding="utf-8"):
            src, i = json.loads(line)["row_id"].split(":")
            out.append(tables[{"medmcqa_train": "train", "medmcqa_test": "test"}[src]][int(i)])
        return out

    pub, priv = a.v2_assets / "public", a.v2_assets / "private"
    probe = lambda p: [tables["train"][s["row_id"]] for s in json.loads(p.read_text())["samples"]]
    row_map = {
        "schema_version": 1,
        "source": f"openlifescienceai/medmcqa @ {MEDMCQA_REVISION}, {TRAIN_PARQUET}",
        "source_sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
        "note": "row positions in that parquet; see derive_row_map.py for how they were derived",
        "train": ids(pub / "train.jsonl"),
        "train_public": ids(pub / "train_public.jsonl"),
        "test_public": ids(pub / "test_public.jsonl"),
        "eval": ids(priv / "eval.jsonl"),
        "paper_test": tables["test"],
        "probe_public": probe(pub / "probe_batch.json"),
        "probe_private": probe(priv / "probe_batch.json"),
    }
    assert row_map["test_public"] == tables["test"][:64]
    a.out.write_text(json.dumps(row_map) + "\n")
    print(json.dumps({k: len(v) if isinstance(v, list) else v for k, v in row_map.items()}, indent=1))


if __name__ == "__main__":
    main()
