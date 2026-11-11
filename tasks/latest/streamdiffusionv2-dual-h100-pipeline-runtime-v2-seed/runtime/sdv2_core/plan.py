# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frozen input chunking, motion schedule, cache windows, and indexed noise.

The motion calculation is copied from ``streamv2v/inference.py`` at the pinned
upstream commit.  Random tensors are addressed rather than consumed from a
process-global RNG, so scheduling cannot alter an output.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final, Iterator

import torch


_MASK64: Final[int] = (1 << 64) - 1
_INPUT_KIND: Final[int] = 0x243F_6A88_85A3_08D3
_STEP_KIND: Final[int] = 0x1319_8A2E_0370_7344
_INDEX_MUL: Final[int] = 0x9E37_79B9_7F4A_7C15
_STEP_MUL: Final[int] = 0xD1B5_4A32_D192_ED03


def _splitmix64(value: int) -> int:
    """Return the SplitMix64 permutation of ``value``."""
    value = (value + _INDEX_MUL) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58_476D_1CE4_E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D0_49BB_1331_11EB) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def mix_seed(case_seed: int, kind: int, chunk_idx: int, step_idx: int = 0) -> int:
    """Mix a case seed and a logical noise address into a torch seed.

    Exact formula::

        splitmix64((case_seed & (2**64-1)) ^ kind ^
                   ((chunk_idx * 0x9e3779b97f4a7c15) & (2**64-1)) ^
                   ((step_idx * 0xd1b54a32d192ed03) & (2**64-1))) & (2**63-1)

    The final mask keeps the seed in torch's portable non-negative range.
    """
    if chunk_idx < 0:
        raise ValueError(f"chunk_idx must be non-negative, got {chunk_idx}")
    if step_idx < 0:
        raise ValueError(f"step_idx must be non-negative, got {step_idx}")
    mixed_input = (
        (case_seed & _MASK64)
        ^ kind
        ^ ((chunk_idx * _INDEX_MUL) & _MASK64)
        ^ ((step_idx * _STEP_MUL) & _MASK64)
    )
    return _splitmix64(mixed_input) & ((1 << 63) - 1)


class NoiseProvider:
    """Produce deterministic BF16 latent noise for an indexed stream.

    Input noise is addressed by chunk. Re-noise is addressed by the chunk and
    denoising step it belongs to, so changing the interleaving of chunks cannot
    change their stochastic inputs.
    """

    def __init__(
        self,
        *,
        case_seed: int,
        steps: int,
        latent_height: int,
        latent_width: int,
        device: torch.device | str = "cpu",
    ) -> None:
        if not 1 <= steps <= 4:
            raise ValueError(f"steps must be in [1, 4], got {steps}")
        if latent_height <= 0 or latent_width <= 0:
            raise ValueError("latent dimensions must be positive")
        self.case_seed = int(case_seed)
        self.steps = int(steps)
        self.latent_height = int(latent_height)
        self.latent_width = int(latent_width)
        self.device = torch.device(device)

    def on(self, device: torch.device | str) -> "NoiseProvider":
        """Return an equivalent provider whose outputs live on ``device``."""
        return NoiseProvider(
            case_seed=self.case_seed,
            steps=self.steps,
            latent_height=self.latent_height,
            latent_width=self.latent_width,
            device=device,
        )

    def _draw(
        self,
        shape: tuple[int, ...],
        *,
        kind: int,
        chunk_idx: int,
        step_idx: int = 0,
    ) -> torch.Tensor:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(mix_seed(self.case_seed, kind, chunk_idx, step_idx))
        return torch.randn(
            shape,
            generator=generator,
            device=self.device,
            dtype=torch.bfloat16,
        )

    def input(self, chunk_idx: int, *, latent_frames: int = 1) -> torch.Tensor:
        if latent_frames <= 0:
            raise ValueError("latent_frames must be positive")
        return self._draw(
            (1, latent_frames, 16, self.latent_height, self.latent_width),
            kind=_INPUT_KIND,
            chunk_idx=int(chunk_idx),
        )

    def step(self, chunk_idx: int, step_idx: int, *, latent_frames: int = 1) -> torch.Tensor:
        """Return the re-noise after ``step_idx`` of ``chunk_idx``."""
        if chunk_idx < 0:
            raise ValueError("chunk_idx must be non-negative")
        if not 0 <= step_idx < self.steps - 1:
            raise ValueError(
                f"step_idx must be in [0, {self.steps - 1}), got {step_idx}"
            )
        if latent_frames <= 0:
            raise ValueError("latent_frames must be positive")
        return self._draw(
            (1, latent_frames, 16, self.latent_height, self.latent_width),
            kind=_STEP_KIND,
            chunk_idx=int(chunk_idx),
            step_idx=int(step_idx),
        )


@dataclass(frozen=True)
class StreamChunk:
    """One real input chunk and its frozen protocol metadata."""

    index: int
    frame_start: int
    frame_end: int
    noise_scale: float
    current_step: int
    current_start: int
    current_end: int

    @property
    def frame_count(self) -> int:
        return self.frame_end - self.frame_start


@dataclass(frozen=True)
class StreamPlanMeta:
    """Everything a runtime may know about a case before its frames arrive.

    This is the plan without the input video: chunk descriptors (including the
    motion-derived first step, one scalar per chunk), indexed noise, and cache
    geometry. Frames reach the runtime only through ``start`` and ``push``, so
    no input work can be done ahead of the chunk being offered.
    """

    steps: int
    case_seed: int
    chunks: tuple[StreamChunk, ...]
    noise: NoiseProvider
    frame_seq_length: int
    kv_cache_length: int
    t_refresh: int = 50
    denoising_steps: tuple[int, ...] = ()

    def on(self, device: torch.device | str) -> "StreamPlanMeta":
        """Return a shallow copy whose indexed noise is drawn on ``device``."""
        return replace(self, noise=self.noise.on(torch.device(device)))

    def __iter__(self) -> Iterator[StreamChunk]:
        return iter(self.chunks)

    def __len__(self) -> int:
        return len(self.chunks)


@dataclass(frozen=True)
class StreamPlan:
    """Complete deterministic input plan for one V2V case."""

    frames: torch.Tensor
    steps: int
    case_seed: int
    chunks: tuple[StreamChunk, ...]
    noise: NoiseProvider
    frame_seq_length: int
    kv_cache_length: int
    t_refresh: int = 50
    denoising_steps: tuple[int, ...] = ()

    @classmethod
    def from_video(
        cls,
        frames: torch.Tensor,
        steps: int,
        noise_scale: float = 0.8,
        case_seed: int = 0,
    ) -> "StreamPlan":
        """Build the upstream 5-then-4-frame streaming protocol.

        ``frames`` must be ``[3,T,H,W]`` floating point in ``[-1,1]`` and
        ``T == 4*n + 1``.  The first five frames produce two latent frames;
        each subsequent four-frame chunk produces one.
        """
        if frames.ndim != 4 or frames.shape[0] != 3:
            raise ValueError(f"frames must have shape [3,T,H,W], got {tuple(frames.shape)}")
        if not frames.is_floating_point():
            raise TypeError(f"frames must be floating point, got {frames.dtype}")
        if frames.device.type != "cpu":
            raise ValueError("StreamPlan.from_video requires CPU frames; use plan.on(device) afterwards")
        if frames.numel() and (frames.min().item() < -1.0 or frames.max().item() > 1.0):
            raise ValueError("frames values must be in [-1, 1]")
        if not 1 <= int(steps) <= 4:
            raise ValueError(f"steps must be in [1, 4], got {steps}")
        _, total_frames, height, width = frames.shape
        if total_frames < 5 or (total_frames - 1) % 4 != 0:
            raise ValueError(f"frame count must be 4*n+1 and at least 5, got {total_frames}")
        if height % 16 or width % 16:
            raise ValueError(f"height and width must be divisible by 16, got {height}x{width}")
        if not 0.0 <= noise_scale <= 1.0:
            raise ValueError(f"noise_scale must be in [0, 1], got {noise_scale}")

        frame_seq_length = (height // 16) * (width // 16)
        kv_cache_length = 6 * frame_seq_length
        denoising_steps = (700, 500, 400, 200)[: int(steps)]
        # The upstream entrypoint converts the decoded video to BF16 before
        # calling compute_noise_scale_and_step. Keep that dtype path even when
        # the public plan input is float32.
        motion_frames = frames.to(dtype=torch.bfloat16)
        chunks: list[StreamChunk] = [
            StreamChunk(
                index=0,
                frame_start=0,
                frame_end=5,
                noise_scale=float(noise_scale),
                current_step=denoising_steps[0],
                current_start=0,
                current_end=2 * frame_seq_length,
            )
        ]

        running_noise_scale = float(noise_scale)
        init_noise_scale = float(noise_scale)
        current_start = 2 * frame_seq_length
        current_end = 3 * frame_seq_length
        for index, frame_start in enumerate(range(5, total_frames, 4), start=1):
            frame_end = frame_start + 4
            # Line-for-line arithmetic from upstream compute_noise_scale_and_step,
            # expressed on the previous frame plus this four-frame chunk.
            window = motion_frames[:, frame_start - 1:frame_end].unsqueeze(0)
            l2_dist = (window[:, :, 1:] - window[:, :, :-1]) ** 2
            l2_dist = (torch.sqrt(l2_dist.mean(dim=(0, 1, 3, 4))).max() / 0.2).clamp(0, 1)
            running_noise_scale = (
                (init_noise_scale - 0.1 * l2_dist.item()) * 0.9
                + running_noise_scale * 0.1
            )
            current_step = int(1000 * running_noise_scale) - 100

            if current_start // frame_seq_length >= 50:
                current_start = kv_cache_length - frame_seq_length
                current_end = current_start + frame_seq_length
            chunks.append(
                StreamChunk(
                    index=index,
                    frame_start=frame_start,
                    frame_end=frame_end,
                    noise_scale=running_noise_scale,
                    current_step=current_step,
                    current_start=current_start,
                    current_end=current_end,
                )
            )
            current_start = current_end
            current_end += frame_seq_length

        noise = NoiseProvider(
            case_seed=case_seed,
            steps=int(steps),
            latent_height=height // 8,
            latent_width=width // 8,
            device=frames.device,
        )
        return cls(
            frames=frames,
            steps=int(steps),
            case_seed=int(case_seed),
            chunks=tuple(chunks),
            noise=noise,
            frame_seq_length=frame_seq_length,
            kv_cache_length=kv_cache_length,
            denoising_steps=denoising_steps,
        )

    def meta(self) -> StreamPlanMeta:
        """The frames-free view handed to a candidate runtime."""
        return StreamPlanMeta(
            steps=self.steps,
            case_seed=self.case_seed,
            chunks=self.chunks,
            noise=self.noise,
            frame_seq_length=self.frame_seq_length,
            kv_cache_length=self.kv_cache_length,
            t_refresh=self.t_refresh,
            denoising_steps=self.denoising_steps,
        )

    def on(self, device: torch.device | str) -> "StreamPlan":
        """Return a shallow plan copy with frames and noise on ``device``."""
        target = torch.device(device)
        return replace(self, frames=self.frames.to(target), noise=self.noise.on(target))

    def frame_chunk(self, chunk: int | StreamChunk) -> torch.Tensor:
        descriptor = self.chunks[chunk] if isinstance(chunk, int) else chunk
        return self.frames[:, descriptor.frame_start:descriptor.frame_end]

    def __iter__(self) -> Iterator[StreamChunk]:
        return iter(self.chunks)

    def __len__(self) -> int:
        return len(self.chunks)
