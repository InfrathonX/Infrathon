#!/usr/bin/env python3
"""Starter entry for the SD3.5-M on-policy distillation task. Edit freely; the command line and outputs are the contract.

  python task_entry.py run --contract CONTRACT.json --out OUT

`run` must train `rounds` rounds of multi-teacher on-policy distillation and write OUT/steps/step_XXXXX.pt,
OUT/final_lora.safetensors and OUT/result.json (docs/INTERFACE.md).

What is here: the contract mapped onto miles_diffusion's existing options for the SD3.5-M student (sampling, SDE
window, LoRA, optimizer, 2-GPU colocate). What is missing is the task itself: miles_diffusion has no teacher path and
no distillation loss, so this starter stops before launching and scores 0. See AGENT_TASK.md.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "repo"


def miles_args(c: dict, out: Path) -> list[str]:
    s, o, lora = c["sampling"], c["optimizer"], c["lora"]
    return [
        "--hf-checkpoint", c["model"],
        "--rollout-function-path", "miles.rollout.sglang_diffusion_rollout.generate_rollout",
        "--prompt-data", c["rows"], "--input-key", "input", "--metadata-key", "metadata",
        "--rollout-batch-size", str(c["rows_per_round"]), "--n-samples-per-prompt", str(c["samples_per_row"]),
        "--rollout-microgroup-size", str(c["samples_per_row"]), "--num-rollout", str(c["rounds"]),
        "--rollout-seed", str(c["rollout_seed"]), "--num-steps-per-rollout", str(c["steps_per_round"]),
        "--diffusion-num-steps", str(s["num_inference_steps"]), "--diffusion-guidance-scale", str(s["guidance_scale"]),
        "--diffusion-height", str(s["height"]), "--diffusion-width", str(s["width"]),
        "--diffusion-sde-type", s["sde_type"], "--diffusion-noise-level", str(s["noise_level"]),
        "--diffusion-log-prob-no-const",
        "--diffusion-step-strategy-path", "miles.rollout.step_strategy_hub.sde_window",
        "--diffusion-num-sde-steps", str(s["window"]),
        "--diffusion-sde-window-range", f"{s['window_start_range'][0]},{s['window_start_range'][1] + s['window'] - 1}",
        "--lr", str(o["lr"]), "--adam-beta1", str(o["betas"][0]), "--adam-beta2", str(o["betas"][1]),
        "--adam-eps", str(o["eps"]), "--weight-decay", str(o["weight_decay"]), "--clip-grad", str(o["clip_grad"]),
        "--use-lora", "--lora-ipc-weight-sync", "--lora-rank", str(lora["rank"]), "--lora-alpha", str(lora["alpha"]),
        "--use-miles-router", "--sglang-server-concurrency", "8", "--sglang-dit-precision", "fp16",
        "--train-backend", "fsdp", "--diffusion-forward-dtype", "fp16", "--gradient-checkpointing",
        "--actor-num-gpus-per-node", "2", "--rollout-num-gpus", "2", "--rollout-num-gpus-per-engine", "1",
        "--num-gpus-per-node", "2", "--colocate",
        # TODO(on-policy distillation): route each row to its teacher (c["teachers"], c["teacher_key"]), start the student from
        # c["student_init"], send generate requests to c["generate_url"] when set (and report publications), use the
        # distillation loss, and write the per-step records under out / "steps".
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--contract", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    c = json.load(open(a.contract))
    out = Path(a.out)
    (out / "steps").mkdir(parents=True, exist_ok=False)
    cmd = [sys.executable, str(REPO / "train_diffusion.py"), *miles_args(c, out)]
    print("would run:", shlex.join(cmd), flush=True)
    raise NotImplementedError("on-policy distillation is not implemented: miles_diffusion has no teacher path or distillation loss")
    # miles_diffusion recipes launch through Ray (see repo/miles/utils/external_utils/command_utils.py execute_train);
    # subprocess.run(..., check=True, cwd=REPO, env=dict(os.environ, PYTHONPATH=str(REPO)))


if __name__ == "__main__":
    main()
