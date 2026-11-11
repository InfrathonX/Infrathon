"""Precompute the J2/R1 reward oracle: official MizzenAI HPSv3 (github @bd0c5fcb) under its pinned transformers 4.45.2.

Run with PYTHONPATH=/official_env:/work/refs/HPSv3-official:/work/wan (see spikes/s3c_hpsv3_official.py for the
environment). Scores frames 0/4/8 of every video of a replay dump (raw decoded frames; official preprocessing) or of
a video npz. Writes ``<out>/hpsv3_oracle.json``: {"frame_mu": [[g][j][3]] or [[N][3]], "reward": ..., "facts": ...}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

FRAMES = (0, 4, 8)


def build_inferencer():
    import transformers
    from hpsv3.inference import HPSv3RewardInferencer

    cfg = yaml.safe_load(open("/work/refs/HPSv3-official/hpsv3/config/HPSv3_7B.yaml"))
    cfg["model_name_or_path"] = "/models/Qwen2-VL-7B-Instruct-full"
    cfg["disable_flash_attn2"] = True
    Path("/tmp/hpsv3cfg").mkdir(exist_ok=True)
    yaml.safe_dump(cfg, open("/tmp/hpsv3cfg/cfg.yaml", "w"))
    inf = HPSv3RewardInferencer(config_path="/tmp/hpsv3cfg/cfg.yaml",
                                checkpoint_path="/models/HPSv3/HPSv3.safetensors", device="cuda")
    return inf, {"transformers": transformers.__version__, "torch": torch.__version__,
                 "attn": inf.model.config._attn_implementation}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=None, help="replay dump dir (frames.npy + meta.json)")
    ap.add_argument("--videos", default=None, help="npz with videos [N,F,H,W,3] and prompts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()
    inf, facts = build_inferencer()
    if a.dump:
        meta = json.load(open(Path(a.dump) / "meta.json"))
        fr = np.load(Path(a.dump) / "frames.npy", mmap_mode="r")
        items = [(fr[g, j], meta["prompts"][g]) for g in range(fr.shape[0]) for j in range(fr.shape[1])]
        shape = (fr.shape[0], fr.shape[1])
    else:
        z = np.load(a.videos)
        items = list(zip(z["videos"], [str(p) for p in z["prompts"]]))
        shape = (len(items),)
    imgs, prompts = [], []
    for v, p in items:
        for f in FRAMES:
            imgs.append(Image.fromarray(np.ascontiguousarray(v[f])))
            prompts.append(p)
    mus = []
    with torch.inference_mode():
        for s in range(0, len(imgs), a.batch):
            mus += inf.reward(prompts[s:s + a.batch], imgs[s:s + a.batch])[:, 0].float().cpu().tolist()
    mu = torch.tensor(mus, dtype=torch.float64).view(*shape, len(FRAMES))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "hpsv3_oracle.json").write_text(json.dumps(
        {"frame_mu": mu.tolist(), "reward": (0.1 * mu.mean(-1)).tolist(), "facts": facts, "batch": a.batch}))
    print(json.dumps({"n_frames": len(mus), "mu_mean": float(mu.mean()), **facts}))


if __name__ == "__main__":
    main()
