# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frozen causal Wan DiT math, factored at the rolling-KV policy seam."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, MutableMapping, Sequence

import torch

from .kv_policy import KVPolicy, KVSlot


CrossAttentionSlot = MutableMapping[str, Any]


@dataclass(frozen=True)
class PatchMetadata:
    shape: torch.Tensor
    grid_sizes: torch.Tensor
    seq_lens: torch.Tensor


@dataclass(frozen=True)
class DITContext:
    e: torch.Tensor
    e0: torch.Tensor
    e_head: torch.Tensor
    context: torch.Tensor
    context_lens: None
    grid_sizes: torch.Tensor
    seq_lens: torch.Tensor
    causal_rope_cache: list[tuple[int, torch.Tensor]]
    current_start: torch.Tensor
    current_end: torch.Tensor

    def lane(self, index: int) -> "DITContext":
        return DITContext(
            e=self.e[index:index + 1],
            e0=self.e0[index:index + 1],
            e_head=self.e_head[index:index + 1],
            context=self.context[index:index + 1],
            context_lens=None,
            grid_sizes=self.grid_sizes[index:index + 1],
            seq_lens=self.seq_lens[index:index + 1],
            causal_rope_cache=self.causal_rope_cache[index:index + 1],
            current_start=self.current_start[index:index + 1],
            current_end=self.current_end[index:index + 1],
        )


class DITCore:
    """Thin pure-math facade over a loaded upstream ``CausalWanModel``."""

    def __init__(self, generator_wrapper: torch.nn.Module) -> None:
        self.wrapper = generator_wrapper
        self.model = generator_wrapper.model
        self.scheduler = generator_wrapper.scheduler
        self.num_blocks = len(self.model.blocks)
        self.dim = int(self.model.dim)
        self.num_heads = int(self.model.num_heads)
        self.head_dim = self.dim // self.num_heads
        if self.num_blocks != 30 or self.dim != 1536 or self.num_heads != 12 or self.head_dim != 128:
            raise ValueError(
                "unexpected Wan model geometry: "
                f"blocks={self.num_blocks}, dim={self.dim}, heads={self.num_heads}, head_dim={self.head_dim}"
            )
        # These bookkeeping fields are forbidden on the frozen math module.
        for block in self.model.blocks:
            for name in ("evict_idx", "sink_size", "adapt_sink_thr"):
                if hasattr(block.self_attn, name):
                    delattr(block.self_attn, name)

    @property
    def device(self) -> torch.device:
        return self.model.patch_embedding.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.model.patch_embedding.weight.dtype

    def patchify(self, latents_bf16: torch.Tensor) -> tuple[torch.Tensor, PatchMetadata]:
        if latents_bf16.ndim != 5:
            raise ValueError(
                f"latents must have shape [B,F,C,H,W], got {tuple(latents_bf16.shape)}"
            )
        if latents_bf16.dtype != torch.bfloat16:
            raise TypeError(f"latents must be bfloat16, got {latents_bf16.dtype}")
        if latents_bf16.device != self.device:
            raise ValueError(f"latents are on {latents_bf16.device}, model is on {self.device}")
        inputs = latents_bf16.permute(0, 2, 1, 3, 4)
        patched = [self.model.patch_embedding(sample.unsqueeze(0)) for sample in inputs]
        bsz, channels, frames, height, width = patched[0].shape
        shape = torch.tensor(
            [bsz, channels, frames, height, width],
            dtype=torch.int64,
            device=self.device,
        )
        # Match upstream: grid_sizes and seq_lens are CPU tensors.
        grid_sizes = torch.stack([torch.tensor(item.shape[2:], dtype=torch.long) for item in patched])
        flattened = [item.flatten(2).transpose(1, 2) for item in patched]
        seq_lens = torch.tensor([item.size(1) for item in flattened], dtype=torch.long)
        x = torch.cat(flattened)
        return x, PatchMetadata(shape=shape, grid_sizes=grid_sizes, seq_lens=seq_lens)

    def metadata_from_shape(self, shape: torch.Tensor, batch_size: int) -> PatchMetadata:
        if shape.numel() != 5:
            raise ValueError("patched shape must contain five integers")
        if shape.device.type != "cpu":
            raise ValueError("patched shape must be a CPU tensor")
        bsz, channels, frames, height, width = (int(value) for value in shape.tolist())
        if bsz != 1:
            raise ValueError(f"per-sample patched batch dimension must be one, got {bsz}")
        grid_sizes = torch.tensor([[frames, height, width]] * batch_size, dtype=torch.long)
        seq_lens = torch.full((batch_size,), frames * height * width, dtype=torch.long)
        return PatchMetadata(shape=shape, grid_sizes=grid_sizes, seq_lens=seq_lens)

    def make_context(
        self,
        prompt_embeds: torch.Tensor,
        timestep: torch.Tensor,
        grid_sizes: torch.Tensor,
        start_frame: int | torch.Tensor,
        *,
        current_start: torch.Tensor | None = None,
        current_end: torch.Tensor | None = None,
    ) -> DITContext:
        from models.wan.causal_model import _prepare_causal_rope_cache
        from models.wan.wan_base.modules.model import sinusoidal_embedding_1d

        if self.model.freqs.device != self.device:
            self.model.freqs = self.model.freqs.to(self.device)
        if timestep.ndim == 1:
            timestep = timestep.unsqueeze(1)
        timestep = timestep.to(device=self.device)
        batch_size = int(timestep.shape[0])
        if prompt_embeds.ndim != 3:
            raise ValueError("prompt_embeds must have shape [B,L,C]")
        prompt_embeds = prompt_embeds.to(device=self.device, dtype=self.dtype)
        if prompt_embeds.shape[0] == 1 and batch_size > 1:
            prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)
        if prompt_embeds.shape[0] != batch_size:
            raise ValueError("prompt batch size does not match timestep batch size")
        if prompt_embeds.shape[1] > self.model.text_len:
            raise ValueError("prompt length exceeds model text_len")

        e = self.model.time_embedding(
            sinusoidal_embedding_1d(self.model.freq_dim, timestep.flatten()).type_as(prompt_embeds)
        )
        e0 = self.model.time_projection(e).unflatten(1, (6, self.dim)).unflatten(0, timestep.shape)
        context = self.model.text_embedding(
            torch.stack(
                [
                    torch.cat(
                        [sample, sample.new_zeros(self.model.text_len - sample.size(0), sample.size(1))]
                    )
                    for sample in prompt_embeds
                ]
            )
        )
        if isinstance(start_frame, int):
            start_frames = torch.full((batch_size,), start_frame, dtype=torch.long)
        else:
            if start_frame.device.type != "cpu":
                raise ValueError("start_frame must be a Python int or CPU tensor")
            start_frames = start_frame.detach().to(dtype=torch.long).reshape(-1)
        if start_frames.numel() != batch_size:
            raise ValueError("start_frame must have one value per batch lane")
        causal_rope_cache = _prepare_causal_rope_cache(
            grid_sizes,
            self.model.freqs,
            start_frame=start_frames,
        )
        frame_seqlen = int(grid_sizes[0, 1:].prod().item())
        if current_start is None:
            current_start = start_frames.to(self.device) * frame_seqlen
        else:
            current_start = current_start.to(device=self.device, dtype=torch.long).reshape(-1)
        if current_end is None:
            current_end = current_start + int(grid_sizes[0, 0].item()) * frame_seqlen
        else:
            current_end = current_end.to(device=self.device, dtype=torch.long).reshape(-1)
        if current_start.numel() != batch_size or current_end.numel() != batch_size:
            raise ValueError("current_start/current_end must have one value per batch lane")
        return DITContext(
            e=e,
            e0=e0,
            e_head=e.unflatten(0, timestep.shape).unsqueeze(2),
            context=context,
            context_lens=None,
            grid_sizes=grid_sizes,
            seq_lens=torch.tensor(
                [int(math.prod(item)) for item in grid_sizes.tolist()], dtype=torch.long
            ),
            causal_rope_cache=causal_rope_cache,
            current_start=current_start,
            current_end=current_end,
        )

    def _self_attention(
        self,
        block_index: int,
        x: torch.Tensor,
        ctx: DITContext,
        kv_slot: KVSlot,
        kv_policy: KVPolicy,
        lane_ids: Sequence[int] | None,
    ) -> torch.Tensor:
        from models.wan.causal_model import causal_rope_apply, flash_attn_interface, FLASH_ATTN_AVAILABLE

        attention = self.model.blocks[block_index].self_attn
        batch_size, sequence_length = x.shape[:2]
        query = attention.norm_q(attention.q(x)).view(
            batch_size, sequence_length, attention.num_heads, attention.head_dim
        )
        key = attention.norm_k(attention.k(x)).view(
            batch_size, sequence_length, attention.num_heads, attention.head_dim
        )
        value = attention.v(x).view(
            batch_size, sequence_length, attention.num_heads, attention.head_dim
        )
        frame_seqlen = int(math.prod(ctx.grid_sizes[0][1:]))
        start_frame = ctx.current_start // frame_seqlen
        roped_query = causal_rope_apply(
            query,
            ctx.grid_sizes,
            self.model.freqs,
            start_frame=start_frame,
            rope_cache=ctx.causal_rope_cache,
        ).type_as(value)
        roped_key = causal_rope_apply(
            key,
            ctx.grid_sizes,
            self.model.freqs,
            start_frame=start_frame,
            rope_cache=ctx.causal_rope_cache,
        ).type_as(value)
        update = kv_policy.update(
            block_index,
            kv_slot,
            roped_key,
            value,
            ctx.current_start,
            ctx.current_end,
            frame_seqlen,
            sink_size=3,
            adapt_thr=0.2,
            lane_ids=lane_ids,
        )
        if update.read_len <= 0 or update.read_len > kv_slot["k"].shape[1]:
            raise ValueError(f"policy returned invalid read_len={update.read_len}")
        if not FLASH_ATTN_AVAILABLE or flash_attn_interface is None:
            raise RuntimeError("flash-attn 2 flash_attn_with_kvcache is required by the frozen protocol")
        with torch.cuda.device(roped_query.device):
            output = flash_attn_interface.flash_attn_with_kvcache(
                q=roped_query,
                k_cache=kv_slot["k"][:, :update.read_len],
                v_cache=kv_slot["v"][:, :update.read_len],
                cache_seqlens=update.seq_lens,
            )
        return attention.o(output.flatten(2))

    def run_block(
        self,
        block_index: int,
        x: torch.Tensor,
        ctx: DITContext,
        kv_slot: KVSlot,
        kv_policy: KVPolicy,
        crossattn_slot: CrossAttentionSlot,
        *,
        lane_ids: Sequence[int] | None = None,
    ) -> torch.Tensor:
        if not 0 <= block_index < self.num_blocks:
            raise IndexError(f"block index {block_index} is outside [0, {self.num_blocks})")
        block = self.model.blocks[block_index]
        num_frames = int(ctx.e0.shape[1])
        frame_seqlen = x.shape[1] // num_frames
        modulation = (block.modulation.unsqueeze(1) + ctx.e0).chunk(6, dim=2)
        normalized = (
            block.norm1(x).unflatten(1, (num_frames, frame_seqlen))
            * (1 + modulation[1])
            + modulation[0]
        ).flatten(1, 2)
        y = self._self_attention(block_index, normalized, ctx, kv_slot, kv_policy, lane_ids)
        x = x + (y.unflatten(1, (num_frames, frame_seqlen)) * modulation[2]).flatten(1, 2)
        x = x + block.cross_attn(
            block.norm3(x),
            ctx.context,
            ctx.context_lens,
            crossattn_cache=crossattn_slot,
        )
        y = block.ffn(
            (
                block.norm2(x).unflatten(1, (num_frames, frame_seqlen))
                * (1 + modulation[4])
                + modulation[3]
            ).flatten(1, 2)
        )
        return x + (y.unflatten(1, (num_frames, frame_seqlen)) * modulation[5]).flatten(1, 2)

    def run_blocks(
        self,
        lo: int,
        hi: int,
        x: torch.Tensor,
        ctx: DITContext,
        kv_slots: Sequence[KVSlot],
        kv_policy: KVPolicy,
        crossattn_slots: Sequence[CrossAttentionSlot],
        *,
        lane_ids: Sequence[int] | None = None,
    ) -> torch.Tensor:
        if not 0 <= lo <= hi <= self.num_blocks:
            raise ValueError(f"invalid block interval [{lo}, {hi})")
        for block_index in range(lo, hi):
            x = self.run_block(
                block_index,
                x,
                ctx,
                kv_slots[block_index],
                kv_policy,
                crossattn_slots[block_index],
                lane_ids=lane_ids,
            )
        return x

    def head(self, x: torch.Tensor, ctx: DITContext, patched_shape: PatchMetadata) -> torch.Tensor:
        output = self.model.head(x, ctx.e_head)
        return torch.stack(self.model.unpatchify(output, patched_shape.grid_sizes))

    def flow_to_x0(
        self,
        flow: torch.Tensor,
        xt: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        original_dtype = flow.dtype
        flow_flat = flow.permute(0, 2, 1, 3, 4).flatten(0, 1)
        xt_flat = xt.flatten(0, 1)
        timestep_flat = timestep.reshape(-1)
        flow_double, xt_double, sigmas, timesteps = map(
            lambda value: value.double().to(flow.device),
            [flow_flat, xt_flat, self.scheduler.sigmas, self.scheduler.timesteps],
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep_flat.unsqueeze(1)).abs(), dim=1
        )
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0 = (xt_double - sigma_t * flow_double).to(original_dtype)
        return x0.unflatten(0, xt.shape[:2])

    def add_noise(
        self,
        x0: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        return self.scheduler.add_noise(x0, noise, timestep)

    def allocate_kv_cache(
        self,
        batch_size: int,
        frame_seq_length: int,
    ) -> list[dict[str, torch.Tensor]]:
        return [
            {
                "k": torch.zeros(
                    (batch_size, 6 * frame_seq_length, self.num_heads, self.head_dim),
                    dtype=self.dtype,
                    device=self.device,
                ),
                "v": torch.zeros(
                    (batch_size, 6 * frame_seq_length, self.num_heads, self.head_dim),
                    dtype=self.dtype,
                    device=self.device,
                ),
                "global_end_index": torch.zeros(batch_size, dtype=torch.long, device=self.device),
                "local_end_index": torch.zeros(batch_size, dtype=torch.long, device=self.device),
            }
            for _ in range(self.num_blocks)
        ]

    def allocate_crossattn_cache(self, batch_size: int) -> list[dict[str, Any]]:
        return [
            {
                "k": torch.zeros(
                    (batch_size, 512, self.num_heads, self.head_dim),
                    dtype=self.dtype,
                    device=self.device,
                ),
                "v": torch.zeros(
                    (batch_size, 512, self.num_heads, self.head_dim),
                    dtype=self.dtype,
                    device=self.device,
                ),
                "is_init": False,
            }
            for _ in range(self.num_blocks)
        ]

    def expand_prepared_caches(
        self,
        kv_slots: Sequence[KVSlot],
        crossattn_slots: Sequence[CrossAttentionSlot],
        batch_size: int,
    ) -> None:
        for kv_slot, cross_slot in zip(kv_slots, crossattn_slots):
            kv_slot["k"] = kv_slot["k"].repeat(batch_size, 1, 1, 1)
            kv_slot["v"] = kv_slot["v"].repeat(batch_size, 1, 1, 1)
            kv_slot["global_end_index"] = kv_slot["global_end_index"].repeat(batch_size)
            kv_slot["local_end_index"] = kv_slot["local_end_index"].repeat(batch_size)
            cross_slot["k"] = cross_slot["k"].expand(batch_size, -1, -1, -1)
            cross_slot["v"] = cross_slot["v"].expand(batch_size, -1, -1, -1)

    def blocks_state_dict_sha256(self) -> str:
        digest = hashlib.sha256()
        for name, tensor in self.model.blocks.state_dict().items():
            contiguous = tensor.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(contiguous.dtype).encode("ascii"))
            digest.update(str(tuple(contiguous.shape)).encode("ascii"))
            digest.update(contiguous.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()
