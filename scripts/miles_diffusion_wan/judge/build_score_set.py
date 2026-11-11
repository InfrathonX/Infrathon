"""Build the J2 score set (CPU): natural rollout videos + spliced videos + edge cases, as a videos npz + prompts jsonl.

- natural: the first N videos of one or more dumps (engine uint8 output);
- spliced: frames 0/4/8 taken from three different natural videos (catches "score frame 0 only / copy");
- edge: all-black, near-white, low-texture (flat colour + faint gradient) videos.
Frames other than 0/4/8 are filled from the same source video so every video is a valid [9,H,W,3] uint8 array.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True, help="comma list of dump dirs")
    ap.add_argument("--natural", type=int, default=48)
    ap.add_argument("--spliced", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    vids, prompts, kinds = [], [], []
    for d in a.dumps.split(","):
        meta = json.load(open(Path(d) / "meta.json"))
        fr = np.load(Path(d) / "frames.npy", mmap_mode="r")
        for g in range(fr.shape[0]):
            for j in range(fr.shape[1]):
                if len(vids) >= a.natural:
                    break
                vids.append(np.ascontiguousarray(fr[g, j]))
                prompts.append(meta["prompts"][g])
                kinds.append("natural")
    nat = len(vids)
    for _ in range(a.spliced):
        i0, i1, i2 = rng.choice(nat, size=3, replace=False)
        v = vids[i0].copy()
        v[4:8] = vids[i1][4:8]
        v[8] = vids[i2][8]
        vids.append(v)
        prompts.append(prompts[i0])
        kinds.append(f"spliced:{i0},{i1},{i2}")
    T, H, W, _ = vids[0].shape
    black = np.zeros((T, H, W, 3), np.uint8)
    white = np.full((T, H, W, 3), 250, np.uint8)
    ramp = np.linspace(90, 110, W, dtype=np.float32)[None, None, :, None]
    flat = np.broadcast_to(np.clip(ramp + np.array([20, 60, 40], np.float32), 0, 255), (T, H, W, 3)).astype(np.uint8)
    for v, k in [(black, "edge:black"), (white, "edge:white"), (flat, "edge:flat")]:
        for p in prompts[:2]:
            vids.append(v.copy())
            prompts.append(p)
            kinds.append(k)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "videos.npz", videos=np.stack(vids), prompts=np.array(prompts))
    (out / "prompts.jsonl").write_text("".join(json.dumps({"prompt": p}) + "\n" for p in prompts))
    (out / "kinds.json").write_text(json.dumps(kinds))
    print(json.dumps({"n": len(vids), "natural": nat, "spliced": a.spliced, "edge": len(vids) - nat - a.spliced}))


if __name__ == "__main__":
    main()
