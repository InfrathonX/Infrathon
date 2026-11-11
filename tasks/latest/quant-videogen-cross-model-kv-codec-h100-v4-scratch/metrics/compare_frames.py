#!/usr/bin/env python3
"""Paper fidelity metrics on canonical lossless RGB uint8 frame archives."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F


def load_frames(path: Path, prefix: int | None = None) -> torch.Tensor:
    frames = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(frames, torch.Tensor):
        raise TypeError(f"{path} does not contain a tensor")
    if frames.dtype != torch.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"{path} must be RGB uint8 [T,H,W,C], got {frames.dtype} {tuple(frames.shape)}")
    if prefix is not None:
        if frames.shape[0] < prefix:
            raise ValueError(f"{path} has {frames.shape[0]} frames, prefix {prefix} requested")
        frames = frames[:prefix]
    return frames.contiguous()


@torch.inference_mode()
def fidelity(reference: torch.Tensor, candidate: torch.Tensor, device: torch.device) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError(f"frame shape mismatch: {tuple(reference.shape)} vs {tuple(candidate.shape)}")
    mse_sum = 0.0
    psnr_sum = 0.0
    ssim_sum = 0.0
    frame_count = 0
    lpips_sum = 0.0
    lpips_model = None
    try:
        import lpips

        # This matches experiments/LongCat/longcat_video/utils/metric.py from
        # the released QVG repository, including its VGG backbone and [0,1]
        # input convention.
        lpips_model = lpips.LPIPS(net="vgg", version="0.1").to(device).eval()
    except Exception as exc:
        raise RuntimeError("the released QVG VGG-LPIPS evaluator is required") from exc

    # The released evaluator intentionally starts at frame 1.
    for start in range(1, reference.shape[0], 8):
        ref = reference[start : start + 8].permute(0, 3, 1, 2).to(device=device, dtype=torch.float32).div_(255)
        cand = candidate[start : start + 8].permute(0, 3, 1, 2).to(device=device, dtype=torch.float32).div_(255)
        diff = cand - ref
        mse_per_frame = (diff * diff).mean(dim=(1, 2, 3))
        mse_sum += float(mse_per_frame.sum().item())
        # A frame identical to the reference scores 100 dB instead of +inf, so a
        # lossless codec yields a finite, JSON-safe reading.
        psnr_sum += float((-10.0 * torch.log10(mse_per_frame.clamp_min(1e-10))).sum().item())

        mu_x = F.avg_pool2d(ref, kernel_size=11, stride=1, padding=5)
        mu_y = F.avg_pool2d(cand, kernel_size=11, stride=1, padding=5)
        sigma_x = F.avg_pool2d(ref * ref, kernel_size=11, stride=1, padding=5) - mu_x * mu_x
        sigma_y = F.avg_pool2d(cand * cand, kernel_size=11, stride=1, padding=5) - mu_y * mu_y
        sigma_xy = F.avg_pool2d(ref * cand, kernel_size=11, stride=1, padding=5) - mu_x * mu_y
        c1 = 0.01**2
        c2 = 0.03**2
        score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
            (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
        )
        ssim_sum += float(score.mean(dim=(1, 2, 3)).sum().item())
        frame_count += ref.shape[0]

        lpips_values = lpips_model(ref.mul(2).sub(1), cand.mul(2).sub(1))
        lpips_sum += float(lpips_values.flatten().sum().item())

    if frame_count == 0:
        raise ValueError("at least two frames are required by the released evaluator")
    return {
        "mse": mse_sum / frame_count,
        "psnr": psnr_sum / frame_count,
        "ssim": ssim_sum / frame_count,
        "lpips": lpips_sum / frame_count,
        "metric_source": "experiments/LongCat/longcat_video/utils/metric.py",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--prefix", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--include-identical-history",
        action="store_true",
        help="Include copied pre-divergence history (usually makes released per-frame PSNR infinite)",
    )
    parser.add_argument(
        "--frame-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Score frames [START, END) only, e.g. the first chunk generated from compressed KV",
    )
    args = parser.parse_args()
    os.environ.setdefault("TORCH_HOME", "/opt/vbench-cache/torch")
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is not None:
        # The sharded Judge keeps every GPU visible; bind the slot's device so
        # nothing (LPIPS weights, library .cuda() calls) lands on GPU 0.
        torch.cuda.set_device(device)
    reference = load_frames(args.reference, args.prefix)
    candidate = load_frames(args.candidate, args.prefix)
    if reference.shape != candidate.shape:
        raise ValueError(f"frame shape mismatch: {tuple(reference.shape)} vs {tuple(candidate.shape)}")
    changed = (reference != candidate).flatten(1).any(dim=1)
    changed_indices = changed.nonzero(as_tuple=False).flatten()
    first_divergent = int(changed_indices[0]) if len(changed_indices) else None
    evaluation_start = 0
    evaluation_end = reference.shape[0]
    if args.frame_range is not None:
        start, end = args.frame_range
        if not 1 <= start < end <= reference.shape[0]:
            raise ValueError(f"frame range {start}..{end} is outside 1..{reference.shape[0]}")
        # The released evaluator skips the first frame it is given.
        evaluation_start, evaluation_end = start - 1, end
    elif first_divergent is not None and not args.include_identical_history:
        # Keep one history frame because the released QVG evaluator itself
        # starts at index 1. This evaluates every divergent frame without
        # averaging +inf PSNR from copied context into the result.
        evaluation_start = max(0, first_divergent - 1)
    result = fidelity(
        reference[evaluation_start:evaluation_end],
        candidate[evaluation_start:evaluation_end],
        device,
    )
    result["prefix_frames"] = args.prefix
    result["first_divergent_frame"] = first_divergent
    result["evaluation_start_frame"] = evaluation_start
    result["evaluation_end_frame"] = evaluation_end
    result["frame_range"] = args.frame_range
    result["identical_history_excluded"] = not args.include_identical_history
    encoded = json.dumps(result, sort_keys=True, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
