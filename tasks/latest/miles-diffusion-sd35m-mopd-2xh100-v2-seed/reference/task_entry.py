#!/usr/bin/env python3
"""Reference entry for the SD3.5-M DiffusionOPD task (host-only; never shipped in the Work image).

  python task_entry.py run --contract CONTRACT.json --out OUT

Maps the contract onto miles_diffusion's native recipe scripts/run_diffusion_opd_sd3_mopd.py (FSDP2 DP=2 student
LoRA + 2 colocated sglang-diffusion engines, frozen merged teacher copies per rank) and writes OUT/steps/*.pt,
OUT/final_lora.safetensors and OUT/result.json.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "repo"
RECIPE_CONSTANTS = {  # what scripts/run_diffusion_opd_sd3_mopd.py hard-codes; the contract must agree
    "samples_per_row": 8,
    "sampling": {"num_inference_steps": 10, "height": 512, "width": 512, "guidance_scale": 1.0, "sde_type": "cps",
                 "noise_level": 0.8, "window": 3, "window_start_range": [0, 3], "log_prob_no_const": True},
    "optimizer": {"lr": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 1e-4, "clip_grad": 1.0},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--contract", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    c = json.load(open(a.contract))
    for k, v in RECIPE_CONSTANTS.items():
        if c[k] != v:
            raise SystemExit(f"contract field {k} = {c[k]} differs from what this entry implements ({v})")
    out = Path(a.out)
    (out / "steps").mkdir(parents=True, exist_ok=False)
    scratch = Path(c["scratch"])
    scratch.mkdir(parents=True, exist_ok=True)
    (label_a, path_a), (label_b, path_b) = list(c["teachers"].items())
    cmd = [sys.executable, str(REPO / "scripts/run_diffusion_opd_sd3_mopd.py"),
           "--model", c["model"], "--prompt-data", c["rows"], "--student-init", c["student_init"],
           "--ocr-teacher", path_a, "--ocr-label", label_a, "--aes-teacher", path_b, "--aes-label", label_b,
           "--num-rollout", str(c["rounds"]), "--rows-per-round", str(c["rows_per_round"]),
           "--steps-per-round", str(c["steps_per_round"]), "--rollout-seed", str(c["rollout_seed"]),
           "--probe-steps", str(c.get("probe_steps", 0)), "--opd-dump-dir", str(out / "steps"),
           "--train-dtype", "fp16", "--output-dir", str(scratch / "logs")]
    if c.get("generate_url"):
        cmd += ["--rollout-generate-url", c["generate_url"]]
    env = dict(os.environ, PYTHONPATH=str(REPO))
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO, env=env)

    import torch
    from safetensors.torch import save_file

    steps = sorted(glob.glob(str(out / "steps/step_*.pt")))
    rec = torch.load(steps[-1], weights_only=False, map_location="cpu")
    save_file({k: v.float().contiguous() for k, v in rec["lora_post"].items()}, str(out / "final_lora.safetensors"))
    json.dump({"status": "completed", "steps": len(steps)}, open(out / "result.json", "w"))


if __name__ == "__main__":
    main()
