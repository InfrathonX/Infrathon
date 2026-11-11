#!/usr/bin/env python3
"""One-time choice of the Work image's extra training rows (host tool, run once; its output is committed).

The public suites only get a pipeline working, so the Work image holds 512 training rows instead of the Judge's
16384: the 64 rows of train_public (the small suite) and 448 more, drawn with a fixed seed from the MedMCQA rows that
medmcqa_row_map.json does not use, skipping any question whose stem repeats one of the map's questions or an earlier
draw (without case, punctuation or layout, whatever its options: this also keeps near-duplicates out). They go to their own file, so the row map and the Judge's data stay as they were.

  python extend_public_rows.py --medmcqa DIR
"""

import argparse
import json
import random
from pathlib import Path

import pyarrow.parquet as pq

from prepare_assets import TRAIN_PARQUET, TRAIN_PARQUET_SHA256, sha, stem_key

ADDED, SEED = 448, 20261106
ROW_KEYS = ("train", "train_public", "test_public", "eval", "paper_test", "probe_public", "probe_private")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--medmcqa", type=Path, required=True)
    ap.add_argument("--row-map", type=Path, default=Path(__file__).with_name("medmcqa_row_map.json"))
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("medmcqa_work_rows.json"))
    a = ap.parse_args()
    parquet = a.medmcqa / TRAIN_PARQUET
    assert sha(parquet) == TRAIN_PARQUET_SHA256, f"{parquet} is not the pinned MedMCQA revision"
    rows = pq.read_table(parquet).to_pylist()
    row_map = json.loads(a.row_map.read_text())
    used = {i for k in ROW_KEYS for i in row_map[k]}
    seen = {stem_key(rows[i]) for i in used}
    candidates = [i for i in range(len(rows)) if i not in used]
    random.Random(SEED).shuffle(candidates)
    added = []
    for i in candidates:
        key = stem_key(rows[i])
        if key not in seen:
            seen.add(key)
            added.append(i)
            if len(added) == ADDED:
                break
    assert len(added) == ADDED, "not enough unused questions"
    a.out.write_text(json.dumps({"schema_version": 1, "source": row_map["source"], "source_sha256": row_map["source_sha256"],
                                 "note": "row positions in that parquet, after train_public; see extend_public_rows.py",
                                 "added": added}) + "\n")
    print(json.dumps({"rows": len(rows), "unused": len(candidates), "added": len(added)}))


if __name__ == "__main__":
    main()
