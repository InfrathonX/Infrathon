#!/usr/bin/env python3
"""Deterministic NumPy-only video fixtures for the SDV2 runtime task.

The public ``generate`` function deliberately has no image/video dependency.  The
small CLI at the bottom streams its returned RGB array to ffmpeg only for human
preview files; benchmark cases consume the uint8 arrays directly.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np


def _texture(rng: np.random.Generator, height: int, width: int) -> np.ndarray:
    """Return a softly correlated, colourful uint8 background."""

    # Nearest-neighbour expansion followed by a short separable box blur gives
    # stable Perlin-ish structure without scipy, Pillow, or OpenCV.
    cell = max(12, min(height, width) // 24)
    gh = height // cell + 3
    gw = width // cell + 3
    coarse = rng.uniform(0.0, 1.0, size=(gh, gw, 3)).astype(np.float32)
    field = np.repeat(np.repeat(coarse, cell, axis=0), cell, axis=1)
    field = field[:height, :width]
    for radius in (1, 2, 4, 8):
        field = (
            field
            + np.roll(field, radius, axis=0)
            + np.roll(field, -radius, axis=0)
            + np.roll(field, radius, axis=1)
            + np.roll(field, -radius, axis=1)
        ) / 5.0
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    waves = 0.08 * np.sin(xx / 19.0 + yy / 37.0)[..., None]
    tint = rng.uniform(0.65, 1.15, size=(1, 1, 3)).astype(np.float32)
    return np.clip((0.14 + 0.68 * field + waves) * tint * 255.0, 0, 255).astype(np.uint8)


def _scene(rng: np.random.Generator, height: int, width: int) -> tuple[np.ndarray, list[dict]]:
    background = _texture(rng, height, width)
    palette = rng.integers(35, 236, size=(5, 3), dtype=np.uint8)
    shapes: list[dict] = []
    for index in range(5):
        size = float(rng.uniform(0.055, 0.14) * min(height, width))
        shapes.append(
            {
                "kind": index % 3,
                "cx": float(rng.uniform(0.12, 0.88) * width),
                "cy": float(rng.uniform(0.12, 0.88) * height),
                "vx": float(rng.uniform(-1.0, 1.0)),
                "vy": float(rng.uniform(-1.0, 1.0)),
                "size": size,
                "phase": float(rng.uniform(0, 2 * np.pi)),
                "spin": float(rng.uniform(-0.055, 0.055)),
                "colour": palette[index].astype(np.float32),
                "stripe": float(rng.uniform(0.12, 0.30)),
            }
        )
    return background, shapes


def generate(
    seed: int,
    num_frames: int,
    height: int,
    width: int,
    *,
    cut_at: int | None = None,
) -> np.ndarray:
    """Generate RGB frames as ``uint8 [T,H,W,3]``.

    The motion amplitude and tempo are seed-dependent.  ``cut_at`` replaces the
    complete scene from that frame onward, making the discontinuity exact and
    suitable for motion-aware input-controller tests.
    """

    if num_frames < 1 or height < 16 or width < 16:
        raise ValueError("num_frames >= 1 and height,width >= 16 are required")
    if cut_at is not None and not 0 < cut_at < num_frames:
        raise ValueError("cut_at must be strictly inside the clip")

    rng = np.random.default_rng(np.uint64(seed))
    scenes = [_scene(rng, height, width), _scene(rng, height, width)]
    frames = np.empty((num_frames, height, width, 3), dtype=np.uint8)
    yy, xx = np.mgrid[:height, :width].astype(np.float32)

    # A deliberately wide seed-dependent range ensures the frozen motion
    # controller observes meaningfully different MAD/current-step regimes.
    motion = 0.55 + (int(seed) % 17) / 8.0
    tempo = 0.55 + ((int(seed) // 17) % 13) / 14.0
    pan_x = float(rng.uniform(-0.55, 0.55) * motion)
    pan_y = float(rng.uniform(-0.35, 0.35) * motion)

    for frame_index in range(num_frames):
        scene_index = int(cut_at is not None and frame_index >= cut_at)
        local_t = frame_index - (cut_at if scene_index else 0)
        background, shapes = scenes[scene_index]
        pan_phase = local_t * tempo
        frame = np.roll(
            background,
            (int(round(pan_y * pan_phase)), int(round(pan_x * pan_phase))),
            axis=(0, 1),
        ).astype(np.float32)

        for item in shapes:
            t = float(local_t) * tempo
            cx = item["cx"] + item["vx"] * motion * t
            cy = item["cy"] + item["vy"] * motion * t
            # Reflect motion at the canvas edges without branchy per-pixel work.
            cx = abs((cx + width) % (2 * width) - width)
            cy = abs((cy + height) % (2 * height) - height)
            scale = 1.0 + 0.18 * np.sin(0.055 * t + item["phase"])
            angle = item["spin"] * t + item["phase"]
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            dx, dy = xx - cx, yy - cy
            rx = (cos_a * dx + sin_a * dy) / (item["size"] * scale)
            ry = (-sin_a * dx + cos_a * dy) / (item["size"] * scale)
            if item["kind"] == 0:
                mask = rx * rx + ry * ry <= 1.0
            elif item["kind"] == 1:
                mask = np.maximum(np.abs(rx), np.abs(ry)) <= 1.0
            else:
                mask = (ry >= -1.0) & (ry <= 1.0) & (np.abs(rx) <= (1.0 - ry) * 0.72)
            stripe = 0.76 + 0.24 * np.sin(
                (rx * 7.0 + ry * 5.0 + t * 0.08) / item["stripe"]
            )
            coloured = item["colour"][None, None, :] * stripe[..., None]
            edge_alpha = np.clip(2.6 * (1.0 - np.sqrt(np.minimum(rx * rx + ry * ry, 1.0))), 0.42, 0.90)
            alpha = (edge_alpha * mask)[..., None]
            frame = frame * (1.0 - alpha) + coloured * alpha

        brightness = 0.94 + 0.055 * np.sin(frame_index * 0.035 + (seed % 31))
        frames[frame_index] = np.clip(frame * brightness, 0, 255).astype(np.uint8)

    return frames


def _write_mp4(frames: np.ndarray, destination: Path, fps: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1:3]
    command = [
        "/usr/bin/ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
        "-framerate", str(fps), "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", str(destination),
    ]
    subprocess.run(command, input=frames.tobytes(), check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--frames", type=int, default=129)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--cut-at", type=int)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()
    data = generate(args.seed, args.frames, args.height, args.width, cut_at=args.cut_at)
    _write_mp4(data, args.output, args.fps)
    delta = np.abs(data[1:].astype(np.int16) - data[:-1].astype(np.int16)).mean(axis=(1, 2, 3))
    print(
        f"{args.output}: shape={list(data.shape)} dtype={data.dtype} "
        f"frame_mad_min={delta.min():.6f} mean={delta.mean():.6f} max={delta.max():.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
