"""Build fixed-replay dumps (Judge/author side) with the reference engine overlay and canonical RNG keys.

Runs inside the pinned image with 2 GPUs and PYTHONPATH=<reference merged engine>:/work/wan. For each dump:
P prompts x n=8 samples, one request per sample (seed=H(init|g), CPU generator, rollout_sde_type=dance, eta 1.2,
window = canonical, rollout_sde_noise_seed = H(sde|g|j)), requesting latents at [w, w+1, w+2]. Writes the layout
read by ``miles.rollout.replay_rollout`` (meta.json, frames.npy, latents.safetensors, cond.safetensors) plus
``requests.jsonl`` (the exact request bodies) for provenance.

  python build_replay_dump.py --out /out/dumps/public0 --prompts /models/video_prompts/train.txt \
      --rows 0-15 --run-seed 1701 --round 0 [--probe]
``--probe`` picks the first single row (from --rows) whose group has >= 4 samples with window start 0 (sigma=1).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, "/work/wan")
sys.path.insert(0, "/work/wan/spikes")
import wan_dance_contract as C  # noqa: E402
from s1_engine_rollout import decode, generate, launch, wait_healthy  # noqa: E402

MODEL = "/models/Wan2.2-TI2V-5B-Diffusers"


def request(prompt: str, key: C.GroupKey, j: int, window_start: int | None = None) -> dict:
    w = list(key.window(j)) if window_start is None else [window_start, window_start + 1]
    return {
        "prompt": prompt, "negative_prompt": " ", "seed": key.init_key(), "generator_device": "cpu",
        "width": 1280, "height": 704, "num_frames": 9, "num_inference_steps": 10,
        "guidance_scale": 5.0, "true_cfg_scale": 5.0, "num_outputs_per_prompt": 1,
        "rollout": True, "rollout_sde_type": "dance", "rollout_noise_level": C.ETA,
        "rollout_log_prob_no_const": False, "rollout_debug_mode": False, "rollout_video_dtype": "uint8",
        "rollout_return_denoising_env": True, "rollout_return_dit_trajectory": True,
        "rollout_sde_step_indices": w, "rollout_return_step_indices": [w[0], w[0] + 1, w[0] + 2],
        "extra_sampling_params": {"rollout_sde_noise_seed": key.sde_key(j)},
    }


def _one(v):
    return decode(v[0] if isinstance(v, list) else v)


def parse_rows(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", default="/models/video_prompts/train.txt")
    ap.add_argument("--rows", required=True)
    ap.add_argument("--run-seed", type=int, required=True)
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--ports", default="30400,30410")
    ap.add_argument("--windows", default=None,
                    help="probe only: comma list of per-sample window starts overriding the canonical ones "
                         "(e.g. 0,0,0,0,6,6,7,7 puts pairs at small sigma where the Dance score term is large)")
    a = ap.parse_args()
    lines = [l.strip() for l in open(a.prompts, encoding="utf-8") if l.strip()]
    rows = parse_rows(a.rows)
    if a.probe:
        for r in rows:
            k = C.GroupKey(a.run_seed, a.round, r)
            if sum(k.window_start(j) == 0 for j in range(a.n)) >= 4:
                rows = [r]
                break
        else:
            raise SystemExit("no probe row with >= 4 sigma=1 windows in --rows")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ports = [int(p) for p in a.ports.split(",")]
    procs = [launch(g, p, MODEL, out / f"engine{g}.log", ["--dit-cpu-offload", "false", "--text-encoder-cpu-offload",
                                                          "false", "--vae-cpu-offload", "false"])
             for g, p in enumerate(ports)]
    t0 = time.time()
    try:
        for p, pr in zip(ports, procs):
            wait_healthy(p, pr)
        jobs = [(gi, j, r) for gi, r in enumerate(rows) for j in range(a.n)]
        results = {}

        def work(engine_idx):
            port = ports[engine_idx]
            for gi, j, r in jobs[engine_idx::len(ports)]:
                key = C.GroupKey(a.run_seed, a.round, r)
                ws = None if a.windows is None else int(a.windows.split(",")[j])
                body = request(lines[r], key, j, ws)
                b, *_ = generate(port, body)
                results[(gi, j)] = (body, b[0])

        with cf.ThreadPoolExecutor(len(ports)) as ex:
            list(ex.map(work, range(len(ports))))
        P, n = len(rows), a.n
        frames = None
        lat = None
        pos, neg = [], []
        windows = [[0] * n for _ in range(P)]
        sig = ts = None
        with open(out / "requests.jsonl", "w") as f:
            for gi in range(P):
                for j in range(n):
                    body, resp = results[(gi, j)]
                    f.write(json.dumps({"g": gi, "j": j, "row": rows[gi], "body": body}) + "\n")
                    go = decode(resp["generated_output"])  # [3, F, H, W] uint8
                    fr = go.permute(1, 2, 3, 0).contiguous().numpy()
                    traj = resp["dit_trajectory"]
                    xs = decode(traj["latents"])
                    idx = decode(traj["latent_step_indices"]).tolist()
                    w = body["rollout_sde_step_indices"][0]
                    assert idx == [w, w + 1, w + 2], idx
                    if frames is None:
                        frames = np.lib.format.open_memmap(out / "frames.npy", mode="w+", dtype=np.uint8,
                                                           shape=(P, n) + fr.shape)
                        # fp32 storage: x_T (w=0) is fp32 in the engine, later latents are bf16 (exact in fp32)
                        lat = torch.empty((P, n) + tuple(xs.shape), dtype=torch.float32)
                    frames[gi, j] = fr
                    lat[gi, j] = xs.float()
                    windows[gi][j] = w
                    if sig is None:
                        sig = decode(traj["sigmas"]).tolist()
                        ts = decode(traj["timesteps"]).tolist()
                    if j == 0:
                        env = resp["denoising_env"]
                        pos.append(_one(env["pos_cond_kwargs"]["encoder_hidden_states"]))
                        neg.append(_one(env["neg_cond_kwargs"]["encoder_hidden_states"]))
        frames.flush()
        save_file({"x": lat.contiguous()}, str(out / "latents.safetensors"))
        save_file({"pos": torch.stack(pos).contiguous(), "neg": torch.stack(neg).contiguous()},
                  str(out / "cond.safetensors"))
        meta = {"run_seed": a.run_seed, "round": a.round, "rows": rows, "prompts": [lines[r] for r in rows], "n": n,
                "windows": windows, "timesteps": ts, "sigmas": sig, "latent_dtype": str(lat.dtype),
                "build_seconds": time.time() - t0}
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        print(json.dumps({k: v for k, v in meta.items() if k != "prompts"}))
    finally:
        for p in procs:
            try:
                os.killpg(p.pid, 9)
            except Exception:
                pass


if __name__ == "__main__":
    main()
