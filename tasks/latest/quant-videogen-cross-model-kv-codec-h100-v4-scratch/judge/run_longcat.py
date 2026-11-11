#!/usr/bin/env python3
"""Candidate-readable model adapter for the protected continuation workload.

The root controller selects and stages the only prompt and initialization
video visible to this process before dropping to its GPU slot's candidate UID.
This adapter never opens the release, golden frames, gates, or scorer.

The sharded Judge runs several cases at once with every GPU visible: each case
names its own GPU (``--device-index``, PCI bus order) and torchrun port
(``--master-port``); the upstream entrypoint binds that GPU from
``QVG_DEVICE_INDEX``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path


QVG_ROOT = Path("/opt/qvg-src")
CHECKPOINT_DIR = Path("/models/hidden/longcat")


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


def positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("--segments must be >= 1")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("candidate", "qvg-reference", "bf16"), required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--prompt-index", type=int, required=True, help="One-based official prompt record")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--master-port", type=master_port, required=True)
    parser.add_argument("--device-index", type=non_negative, required=True,
                        help="CUDA device index in PCI bus order; every GPU stays visible")
    parser.add_argument("--init-video", type=Path)
    parser.add_argument("--make-init", action="store_true")
    # Upstream scripts/LongCat/run_qvg.sh continues for 10 segments (273 frames).
    parser.add_argument("--segments", type=positive, default=10)
    return parser.parse_args(argv)


def check_inputs(args: argparse.Namespace) -> None:
    for path in (CHECKPOINT_DIR, args.prompt_file):
        if not path.exists():
            raise SystemExit(f"missing protected input: {path}")
    if not args.make_init and (args.init_video is None or not args.init_video.is_file()):
        raise SystemExit("--init-video is required for continuation")


def build_env(args: argparse.Namespace, base: Mapping[str, str]) -> dict[str, str]:
    pythonpath = [str(QVG_ROOT / "experiments/LongCat")]
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
    # here: PyroMind's HAMi accounting is only correct when every GPU stays
    # visible and the process selects its own (kvpress v2 judge_shards.py).
    env = dict(base)
    env.update({
        "PYTHONPATH": os.pathsep.join(pythonpath),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "DIFFUSERS_OFFLINE": "1",
        "DUMP_KV_LEVEL": "0",
        "QVG_DEVICE_INDEX": str(args.device_index),
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        # Every mode, goldens included: without it the BF16 10-segment
        # rollout OOMs on fragmentation at segment 2 (12.6 GiB reserved but
        # unallocated on a 93 GiB H100 NVL).
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    if base.get("QVG_CUDA_RESERVE") == "1":
        # PyroMind: one up-front reservation (run_long_t2v.qvg_reserve_device_memory)
        # instead of page-by-page growth, which HAMi serializes across slots.
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
    return env


def build_command(args: argparse.Namespace) -> list[str]:
    command = [
        "torchrun", "--nproc_per_node=1", "--master_addr=127.0.0.1", f"--master_port={args.master_port}",
        str(QVG_ROOT / "experiments/LongCat/run_long_t2v.py"),
        "--checkpoint_dir", str(CHECKPOINT_DIR),
        "--output_dir", str(args.output),
        "--seed", str(args.seed),
        "--prompt_source", "text_to_video_from_file",
        "--prompt", str(args.prompt_file),
        "--prompt_idx", str(args.prompt_index),
        "--save_raw_frames", "--profile", "--no_offload_kv_cache",
    ]
    if args.make_init:
        command += ["--workload", "480p_init", "--num_frames", "73"]
    else:
        command += [
            "--workload", "480p_long_gen",
            "--init_video_path", str(args.init_video),
            "--num_segments", str(args.segments),
            "--num_frames", "93",
            "--num_cond_frames", "73",
            "--quant_type", "none" if args.mode == "bf16" else "triton-nstages-kmeans-int2",
            "--quant_block_size", "64",
            "--cache_num_k_centroids", "256",
            "--cache_num_v_centroids", "256",
            "--kmeans_max_iters", "100",
            "--num_prq_stages", "1",
        ]
    return command


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    check_inputs(args)
    args.output.mkdir(parents=True, exist_ok=True)
    subprocess.run(build_command(args), cwd=QVG_ROOT, env=build_env(args, os.environ), check=True)


if __name__ == "__main__":
    main()
