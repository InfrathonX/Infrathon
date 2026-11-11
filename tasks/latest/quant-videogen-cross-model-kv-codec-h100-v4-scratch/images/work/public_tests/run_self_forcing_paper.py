#!/usr/bin/env python3
"""Run the protected Self-Forcing paper workload with an editable codec."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


QVG_ROOT = Path("/opt/qvg-src")
MODEL_ROOT = QVG_ROOT / "ckpts/Self-Forcing"
PROMPTS = Path("/opt/mlsysforge/assets/public_prompts.txt")
OUTPUT_ROOT = Path("/workspace/public-results/self-forcing")
GOLDEN_ROOT = Path("/eval/public/self-forcing")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("candidate", "bf16"), default="candidate")
    parser.add_argument("--prefixes", type=int, nargs="+", default=[350, 700])
    parser.add_argument(
        "--vbench",
        action="store_true",
        help="Also run the paper's four pinned VBench dimensions for each prefix",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run 27 latent frames so the first real 24-latent compression event executes; diagnostic only",
    )
    args = parser.parse_args()
    if any(value not in (350, 700) for value in args.prefixes):
        raise SystemExit("paper public prefixes are exactly 350 and 700")

    checkpoint = MODEL_ROOT / "Self-Forcing/checkpoints/self_forcing_dmd.pt"
    base = MODEL_ROOT / "Wan2.1-T2V-1.3B"
    for required in (checkpoint, base, PROMPTS):
        if not required.exists():
            raise SystemExit(f"missing protected asset: {required}")

    latent_frames = 27 if args.smoke else 180
    output = OUTPUT_ROOT / args.mode
    output.mkdir(parents=True, exist_ok=True)
    pythonpath = [str(QVG_ROOT / "experiments/Self-Forcing"), str(QVG_ROOT)]
    quant_type = "none" if args.mode == "bf16" else "triton-nstages-kmeans-int2"
    if args.mode == "candidate":
        pythonpath.insert(0, "/workspace/submission")
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": os.pathsep.join(pythonpath),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "DIFFUSERS_OFFLINE": "1",
        "DUMP_KV_LEVEL": "0",
    })
    command = [
        sys.executable,
        str(QVG_ROOT / "experiments/Self-Forcing/inference.py"),
        "--config_path", str(QVG_ROOT / "experiments/Self-Forcing/configs/self_forcing_dmd.yaml"),
        "--checkpoint_path", str(checkpoint),
        "--data_path", str(PROMPTS),
        "--output_folder", str(output),
        "--num_samples", "1",
        "--num_output_frames", str(latent_frames),
        "--local_attn_size", str(latent_frames),
        "--seed", "0",
        "--use_ema",
        "--save_with_index",
        "--save_raw_frames",
        "--profile",
        "--quant_type", quant_type,
        "--cache_num_k_centroids", "256",
        "--cache_num_v_centroids", "256",
        "--kmeans_max_iters", "2",
        "--quant_block_size", "64",
        "--num_prq_stages", "1",
    ]
    subprocess.run(command, cwd=QVG_ROOT, env=env, check=True)
    if args.smoke or args.mode != "candidate":
        return
    metric = Path("/opt/mlsysforge/metrics/compare_frames.py")
    prompts = PROMPTS.read_text(encoding="utf-8").splitlines()
    for prompt_index in range(2):
        candidate = output / f"{prompt_index}-0_ema.frames.pt"
        reference = GOLDEN_ROOT / "bf16" / f"{prompt_index}-0_ema.frames.pt"
        for prefix in args.prefixes:
            subprocess.run([
                sys.executable, str(metric), "--reference", str(reference),
                "--candidate", str(candidate), "--prefix", str(prefix),
                "--output", str(output / f"metrics-p{prompt_index}-{prefix}.json"),
            ], check=True)
            if args.vbench:
                subprocess.run([
                    sys.executable,
                    "/opt/mlsysforge/metrics/run_vbench.py",
                    "--frames", str(candidate),
                    "--prompt", prompts[prompt_index],
                    "--fps", "16",
                    "--prefix", str(prefix),
                    "--output-dir", str(output / f"vbench-p{prompt_index}-{prefix}"),
                    "--name", f"candidate-p{prompt_index}-{prefix}",
                ], check=True)


if __name__ == "__main__":
    main()
