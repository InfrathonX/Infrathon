"""Assemble a frozen replay workload from a recorded reference run.

Input: a record dir (row_{id}.json/.msgpack written by opd_author.record_generate) and the contract rows.jsonl the
run used. Output: <out>/{manifest.json, rows.jsonl, row_*.{json,msgpack}} with ``rounds`` = consecutive blocks of
``rows_per_round`` rows starting at ``first_row`` and ``segments`` = consecutive blocks of ``rounds_per_segment``
rounds. Row files are hard-linked when possible (same filesystem) to avoid copying ~75 MB per row.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--rows", required=True, help="contract rows.jsonl used by the recorded run")
    ap.add_argument("--first-row", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=18)
    ap.add_argument("--rows-per-round", type=int, default=8)
    ap.add_argument("--rounds-per-segment", type=int, default=6)
    ap.add_argument("--warmup-rounds", type=int, default=0,
                    help="untimed leading rounds (the Judge's session P has its probe round there: first publication, "
                         "first step and allocator warm-up stay out of the timed segments)")
    ap.add_argument("--student-init", help="contract student_init.safetensors to place next to the rows")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    all_rows = [json.loads(line) for line in open(args.rows)]
    by_id = {r["metadata"]["row_id"]: r for r in all_rows}
    rounds = [[args.first_row + r * args.rows_per_round + k for k in range(args.rows_per_round)]
              for r in range(args.warmup_rounds + args.rounds)]
    with open(os.path.join(args.out, "rows.jsonl"), "w") as f:
        for rows in rounds:
            for row in rows:
                f.write(json.dumps(by_id[row]) + "\n")
                for ext in ("json", "msgpack"):
                    src = os.path.join(args.record, f"row_{row:06d}.{ext}")
                    dst = os.path.join(args.out, f"row_{row:06d}.{ext}")
                    if not os.path.exists(dst):
                        try:
                            os.link(src, dst)
                        except OSError:
                            shutil.copy2(src, dst)
    if args.student_init:
        shutil.copy2(args.student_init, os.path.join(args.out, "student_init.safetensors"))
    w = args.warmup_rounds
    segs = [[w + s * args.rounds_per_segment, w + (s + 1) * args.rounds_per_segment - 1]
            for s in range(args.rounds // args.rounds_per_segment)]
    json.dump({"rounds": rounds, "segments": segs, "source_record": os.path.abspath(args.record)},
              open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    print(f"{len(rounds)} rounds, {len(segs)} segments -> {args.out}")


if __name__ == "__main__":
    main()
