#!/usr/bin/env python3
"""Run the four VBench dimensions reported by the QVG paper."""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

import torch
from torchvision.io import write_video


DIMENSIONS = [
    "background_consistency",
    "imaging_quality",
    "subject_consistency",
    "aesthetic_quality",
]


def install_minimal_pyiqa_namespace() -> None:
    """Load upstream MUSIQ without pyiqa's unrelated eager dataset imports."""

    package_root = Path("/opt/conda/lib/python3.11/site-packages/pyiqa")
    if not package_root.is_dir():
        raise SystemExit(f"missing pinned pyiqa package: {package_root}")
    package = types.ModuleType("pyiqa")
    package.__path__ = [str(package_root)]
    package.__package__ = "pyiqa"
    archs = types.ModuleType("pyiqa.archs")
    archs.__path__ = [str(package_root / "archs")]
    archs.__package__ = "pyiqa.archs"
    data = types.ModuleType("pyiqa.data")
    data.__path__ = [str(package_root / "data")]
    data.__package__ = "pyiqa.data"
    sys.modules.setdefault("pyiqa", package)
    sys.modules.setdefault("pyiqa.archs", archs)
    sys.modules.setdefault("pyiqa.data", data)


def load_frames(path: Path, prefix: int | None) -> torch.Tensor:
    frames = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(frames, torch.Tensor):
        raise TypeError("frame archive must contain one tensor")
    if frames.dtype != torch.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected RGB uint8 [T,H,W,C], got {frames.dtype} {tuple(frames.shape)}")
    if prefix is not None:
        if frames.shape[0] < prefix:
            raise ValueError(f"requested prefix {prefix}, archive has {frames.shape[0]} frames")
        frames = frames[:prefix]
    return frames.contiguous()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--prefix", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default="qvg-paper")
    parser.add_argument("--vbench-root", type=Path, default=Path("/opt/vbench"))
    parser.add_argument("--vbench-cache", type=Path, default=Path("/opt/vbench-cache"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is not None:
        # The sharded Judge keeps every GPU visible; bind the slot's device so
        # VBench's own .cuda() / "cuda" placements stay on it, not on GPU 0.
        torch.cuda.set_device(device)

    install_minimal_pyiqa_namespace()
    sys.path.insert(0, str(args.vbench_root))
    os.environ["VBENCH_CACHE_DIR"] = str(args.vbench_cache)
    os.environ.setdefault("TORCH_HOME", str(args.vbench_cache / "torch"))
    from vbench import VBench

    frames = load_frames(args.frames, args.prefix)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.output_dir / f"{args.name}.mp4"
    # VBench's pinned loader accepts MP4/GIF. CRF 0 is frozen here so both
    # candidate and reference traverse the same evaluator input path.
    write_video(
        str(video_path),
        frames,
        fps=args.fps,
        video_codec="libx264",
        options={"crf": "0", "preset": "veryslow"},
    )

    evaluator = VBench(
        args.device,
        str(args.vbench_root / "vbench/VBench_full_info.json"),
        str(args.output_dir),
    )
    evaluator.evaluate(
        videos_path=str(video_path),
        name=args.name,
        prompt_list=[args.prompt],
        dimension_list=DIMENSIONS,
        local=True,
        mode="custom_input",
        imaging_quality_preprocessing_mode="longer",
    )
    raw = json.loads((args.output_dir / f"{args.name}_eval_results.json").read_text())
    scores = {dimension: float(raw[dimension][0]) * 100.0 for dimension in DIMENSIONS}
    result = {
        "vbench_commit": "45e79ec14e69a2187202c675d2dbce1a71843d53",
        "dimensions": {
            "background_consistency": scores["background_consistency"],
            "image_quality": scores["imaging_quality"],
            "subject_consistency": scores["subject_consistency"],
            "aesthetic_quality": scores["aesthetic_quality"],
        },
        "prefix_frames": args.prefix,
        "fps": args.fps,
        "evaluator_input_codec": "libx264-crf0-preset-veryslow",
    }
    output_path = args.output_dir / f"{args.name}-paper-metrics.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
