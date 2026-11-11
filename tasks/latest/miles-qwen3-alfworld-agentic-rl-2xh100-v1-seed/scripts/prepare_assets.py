#!/usr/bin/env python3
"""Build the public and private data of the Miles ALFWorld task.

Solvable ALFWorld games (json_2.1.1, movable/Sliced removed) of the train split are
shuffled with seed 0 and cut into public_train (rows 0-1275), private_train (1276-2552)
and heldout (2553-3552). Hidden evaluation = heldout + valid_unseen; public evaluation =
valid_seen. Private training rows are re-ordered with a secret seed.

  OUT/public/train_public.jsonl   {row_id, game}
  OUT/public/eval_public.jsonl    {row_id, game}
  OUT/public/games/...            game.tw-pddl of every public game (paths as in `game`)
  OUT/private/train_private.jsonl {row_id, game}
  OUT/private/eval_hidden.jsonl   {row_id, game}
  OUT/private/games/...           game.tw-pddl of every private game
  OUT/manifest.json               counts and sha256 of every file
Probe batches are added later by make_probe_batch.py (they need recorded episodes).
"""

import argparse
import glob
import hashlib
import json
import os
import random
import secrets
import shutil
from pathlib import Path

PUBLIC_TRAIN, PRIVATE_TRAIN = (0, 1276), (1276, 2553)


def solvable(root, split):
    out = []
    for gf in sorted(glob.glob(f"{root}/json_2.1.1/{split}/*/*/game.tw-pddl")):
        d = os.path.dirname(gf)
        if "movable" in d or "Sliced" in d:
            continue
        if json.load(open(gf)).get("solvable"):
            out.append(os.path.relpath(gf, root))
    return out


def write_rows(path, games, start=0):
    with open(path, "w") as fh:
        for i, g in enumerate(games):
            fh.write(json.dumps({"row_id": start + i, "game": g}) + "\n")


def copy_games(root, games, dst):
    for g in games:
        (dst / g).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(root) / g, dst / g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alfworld-data", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--private-seed", type=int, default=None, help="secret; drawn at random when omitted")
    a = ap.parse_args()
    train = solvable(a.alfworld_data, "train")
    assert len(train) == 3553, len(train)
    random.Random(0).shuffle(train)
    pub, priv, held = train[slice(*PUBLIC_TRAIN)], train[slice(*PRIVATE_TRAIN)], train[PRIVATE_TRAIN[1]:]
    seen, unseen = solvable(a.alfworld_data, "valid_seen"), solvable(a.alfworld_data, "valid_unseen")
    seed = a.private_seed if a.private_seed is not None else secrets.randbits(31)
    priv = list(priv)
    random.Random(seed).shuffle(priv)

    for sub in ("public", "private"):
        (a.out / sub / "games").mkdir(parents=True, exist_ok=False)
    write_rows(a.out / "public/train_public.jsonl", pub)
    write_rows(a.out / "public/eval_public.jsonl", seen)
    write_rows(a.out / "private/train_private.jsonl", priv, start=100000)
    write_rows(a.out / "private/eval_hidden.jsonl", held + unseen, start=200000)
    copy_games(a.alfworld_data, pub + seen, a.out / "public/games")
    copy_games(a.alfworld_data, priv + held + unseen, a.out / "private/games")
    (a.out / "private/private_seed.txt").write_text(f"{seed}\n")

    files = {}
    for f in sorted(a.out.rglob("*.jsonl")):
        files[str(f.relative_to(a.out))] = hashlib.sha256(f.read_bytes()).hexdigest()
    manifest = {"counts": {"public_train": len(pub), "private_train": len(priv), "heldout": len(held),
                           "valid_seen": len(seen), "valid_unseen": len(unseen)}, "files": files}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["counts"]))


if __name__ == "__main__":
    main()
