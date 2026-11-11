#!/usr/bin/env python3
"""Honest calibration variants (spec §J1/R3/R6 tolerances): copies of the reference repo that differ only in
`variant_md_args.json` (extra MD flags appended last). Different kernels/precision/reduction, same math.
  h2_efficient   trainer attention = PyTorch mem-efficient SDPA (reference: diffusers default dispatch)
  h3_fp32fwd     trainer forward in fp32 (reference: bf16)
  h4_bf16reduce  gradient reduce in bf16 + activation checkpointing (reference: fp32 reduce, no checkpointing)
"""
import argparse, json, shutil
from pathlib import Path

VARIANTS = {"h2_efficient": ["--fsdp-attention-backend", "_native_efficient"],
            "h3_fp32fwd": ["--diffusion-forward-dtype", "fp32"],
            "h4_bf16reduce": ["--fsdp-reduce-dtype", "bf16", "--gradient-checkpointing"]}

ap = argparse.ArgumentParser()
ap.add_argument("--src", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
for name, extra in VARIANTS.items():
    dst = a.out / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(a.src, dst, symlinks=True, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    (dst / "variant_md_args.json").write_text(json.dumps(extra))
    (dst / "HONEST").write_text(name + "\n")
    print("honest", name, extra)
