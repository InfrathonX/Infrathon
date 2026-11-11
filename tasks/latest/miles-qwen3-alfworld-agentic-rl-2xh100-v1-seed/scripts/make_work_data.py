#!/usr/bin/env python3
"""Build the data of the Work image from the task's public data (host tool, no GPU).

The public suites only get a pipeline working, so Work holds 32 training rows instead of the 1276 public ones (the
perf suite uses them all, 4 rounds x 8, and the small suite and the recorded sample run the first ones) and the first
8 of the 140 public evaluation games. The Judge's data does not change.

  OUT/train_public.jsonl   the first 32 rows of the public training file
  OUT/eval_public.jsonl    the 8 evaluation games
  OUT/games/...            the game files of those rows and games
  OUT/probe_batch.json     copied from the public data
  OUT/sample_run/...       copied from the public data
  OUT/manifest.json        the SHA-256 of these files

  python make_work_data.py --public PUBLIC_DIR --out OUT
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

TRAIN_ROWS, EVAL_GAMES = 32, 8


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--public", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    train = open(a.public / "train_public.jsonl", encoding="utf-8").readlines()[:TRAIN_ROWS]
    evals = open(a.public / "eval_public.jsonl", encoding="utf-8").readlines()[:EVAL_GAMES]
    sample = json.loads((a.public / "sample_run/contract.json").read_text())
    assert sample["rounds"] * sample["games_per_round"] <= TRAIN_ROWS, "the sample run needs more rows"

    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / "train_public.jsonl").write_text("".join(train), encoding="utf-8")
    (a.out / "eval_public.jsonl").write_text("".join(evals), encoding="utf-8")
    for line in train + evals:
        game = json.loads(line)["game"]
        (a.out / "games" / game).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(a.public / "games" / game, a.out / "games" / game)
    shutil.copy2(a.public / "probe_batch.json", a.out / "probe_batch.json")
    shutil.copytree(a.public / "sample_run", a.out / "sample_run")
    files = {p.relative_to(a.out).as_posix(): {"bytes": p.stat().st_size, "sha256": sha(p)}
             for p in sorted(a.out.rglob("*")) if p.is_file()}
    (a.out / "manifest.json").write_text(json.dumps({"schema_version": 1, "files": files}, indent=2) + "\n")
    print(json.dumps({"train_rows": len(train), "eval_games": len(evals), "game_files": len(train) + len(evals), "files": len(files) + 1}))


if __name__ == "__main__":
    main()
