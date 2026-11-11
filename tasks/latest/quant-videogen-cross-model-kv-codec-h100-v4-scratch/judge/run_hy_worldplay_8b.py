#!/usr/bin/env python3
"""Candidate-readable model adapter for the protected image workload.

The root controller stages the only permitted initialization image and passes
the selected prompt after dropping to its GPU slot's candidate UID.  This adapter
never opens the release, golden frames, gates, or scorer.

The sharded Judge runs several cases at once with every GPU visible: each case
names its own GPU (``--device-index``, PCI bus order) and torchrun port
(``--master-port``); the upstream entrypoint binds that GPU from
``QVG_DEVICE_INDEX``.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path


QVG_ROOT = Path("/opt/qvg-src")
BASE_MODEL = Path("/models/hidden/hy-worldplay/base")
ACTION_CKPT = Path("/models/hidden/hy-worldplay/action/model.safetensors")
PAPER_POSE_SPANS = [("w", 8), ("s", 8), ("a", 8), ("d", 8), ("up", 8),
                    ("down", 8), ("w", 8), ("s", 8), ("a", 7)]
# hyvideo/generate.py parse_pose_string actions; spans count latents.
POSE_COMMAND = re.compile(r"(w|s|a|d|up|down|left|right)-([1-9][0-9]*)")


def latent_steps(video_length: int) -> int:
    if video_length < 5 or (video_length - 1) % 4:
        raise SystemExit("HY video length must be 4*n+1 and contain at least two latents")
    return (video_length - 1) // 4


def pose_for_video_length(video_length: int) -> str:
    remaining = latent_steps(video_length)
    commands: list[str] = []
    for action, span in PAPER_POSE_SPANS:
        if remaining <= 0:
            break
        take = min(span, remaining)
        commands.append(f"{action}-{take}")
        remaining -= take
    if remaining:
        raise SystemExit("video length exceeds the frozen 285-frame paper trajectory")
    return ",".join(commands)


def checked_pose(pose: str, video_length: int) -> str:
    """Accept only an inline pose string whose spans cover every latent step.

    Upstream also reads ``*.json`` pose files and fractional spans; the case
    table never needs either, so both are refused here.
    """
    total = 0
    for command in pose.split(","):
        match = POSE_COMMAND.fullmatch(command)
        if match is None:
            raise SystemExit(f"invalid HY pose command: {command!r}")
        total += int(match.group(2))
    if total != latent_steps(video_length):
        raise SystemExit(f"HY pose covers {total} latent steps, video length needs {latent_steps(video_length)}")
    return pose


def master_port(text: str) -> int:
    value = int(text)
    if not 1024 <= value <= 65535:
        raise argparse.ArgumentTypeError("--master-port must be in 1024..65535")
    return value


def non_negative(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("--device-index must be >= 0")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("candidate", "qvg-reference", "bf16"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video-length", type=int, default=285)
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--master-port", type=master_port, required=True)
    parser.add_argument("--device-index", type=non_negative, required=True,
                        help="CUDA device index in PCI bus order; every GPU stays visible")
    parser.add_argument("--pose", help="Camera trajectory; default: the paper trajectory for --video-length")
    args = parser.parse_args(argv)
    args.pose = (pose_for_video_length(args.video_length) if args.pose is None
                 else checked_pose(args.pose, args.video_length))
    return args


def check_inputs(args: argparse.Namespace) -> None:
    for path in (BASE_MODEL / "transformer/480p_i2v", ACTION_CKPT, args.image_path):
        if not path.exists():
            raise SystemExit(f"missing protected input: {path}")


def build_env(args: argparse.Namespace, base: Mapping[str, str]) -> dict[str, str]:
    pythonpath = [str(QVG_ROOT / "experiments/HY-WorldPlay")]
    if args.mode == "candidate":
        pythonpath[0:0] = ["/workspace/submission", "/opt/qvg-src"]
    else:
        # The bf16 / qvg-reference modes only serve golden generation
        # (scripts/stage_eval_inputs.py).  The Judge image carries the
        # upstream package at /opt/qvg-src/quant_videogen, whose pipeline
        # imports the task ABI module ``mlsysforge_codec`` from the codec
        # directory named by QVG_GOLDEN_CODEC_DIR (the in-tree oracle/ tree,
        # bind-mounted by the staging script); the historical
        # /opt/qvg-reference copy no longer exists in any image.
        pythonpath[0:0] = [base.get("QVG_GOLDEN_CODEC_DIR", "/opt/qvg-golden-codec"), "/opt/qvg-src"]
    # Inherit the controller's environment, including the per-slot
    # OMP_NUM_THREADS / MKL_NUM_THREADS.  CUDA_VISIBLE_DEVICES is never set
    # here (see run_longcat.py).  PYTORCH_CUDA_ALLOC_CONF is left alone:
    # hyvideo/generate.py already defaults it, exactly as for the golden.
    env = dict(base)
    env.update({
        "PYTHONPATH": os.pathsep.join(pythonpath),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "DIFFUSERS_OFFLINE": "1",
        "QVG_DEVICE_INDEX": str(args.device_index),
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    })
    if base.get("QVG_CUDA_RESERVE") == "1":
        # PyroMind: one up-front reservation (hyvideo/generate.qvg_reserve_device_memory)
        # instead of expandable-segment growth, which HAMi serializes across slots.
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
    return env


def build_command(args: argparse.Namespace) -> list[str]:
    return [
        "torchrun", "--nproc_per_node=1", "--master_addr=127.0.0.1", f"--master_port={args.master_port}",
        str(QVG_ROOT / "experiments/HY-WorldPlay/hyvideo/generate.py"),
        "--prompt", args.prompt,
        "--resolution", "480p",
        "--model_path", str(BASE_MODEL),
        "--action_ckpt", str(ACTION_CKPT),
        "--video_length", str(args.video_length),
        "--num_inference_steps", "4",
        "--few_step", "true",
        "--sr", "false",
        "--rewrite", "false",
        "--offloading", "false",
        "--group_offloading", "false",
        "--model_type", "ar",
        "--height", "480", "--width", "832",
        "--image_path", str(args.image_path),
        "--pose", args.pose,
        "--seed", str(args.seed),
        "--output_path", str(args.output),
        "--qvg_quantize_kv", "false" if args.mode == "bf16" else "true",
        "--save_raw_frames", "true",
        "--profile", "true",
    ]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    check_inputs(args)
    args.output.mkdir(parents=True, exist_ok=True)
    subprocess.run(build_command(args), cwd=QVG_ROOT, env=build_env(args, os.environ), check=True)


if __name__ == "__main__":
    main()
