# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-process Stream-Batch reference expressed only through ``sdv2_core``."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Sequence

import torch

from .dit import DITContext, DITCore, PatchMetadata
from .kv_policy import KVSlot, UpstreamKVPolicy
from .loader import FrozenModel, load_model
from .plan import StreamPlan
from .sink import OutputSink, RunLedger


@dataclass
class StreamState:
    hidden_states: torch.Tensor
    chunk_indices: torch.Tensor
    starts: torch.Tensor
    ends: torch.Tensor
    start_frames: list[int]
    timesteps: torch.Tensor
    kv_slots: list[dict[str, torch.Tensor]]
    crossattn_slots: list[dict[str, Any]]
    policy: UpstreamKVPolicy


def _slot_lane(slot: KVSlot, lane: int) -> dict[str, torch.Tensor]:
    return {
        "k": slot["k"][lane:lane + 1],
        "v": slot["v"][lane:lane + 1],
        "global_end_index": slot["global_end_index"][lane:lane + 1],
        "local_end_index": slot["local_end_index"][lane:lane + 1],
    }


def _cross_lane(slot: dict[str, Any], lane: int) -> dict[str, Any]:
    return {
        "k": slot["k"][lane:lane + 1],
        "v": slot["v"][lane:lane + 1],
        "is_init": bool(slot["is_init"]),
    }


def forward_dit(
    dit: DITCore,
    hidden_states: torch.Tensor,
    prompt_embeds: torch.Tensor,
    timesteps: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    kv_slots: Sequence[KVSlot],
    crossattn_slots: Sequence[dict[str, Any]],
    policy: UpstreamKVPolicy,
    *,
    start_frames: Sequence[int] | torch.Tensor | None = None,
    block_range: tuple[int, int] = (0, 30),
    per_lane: bool = False,
    patched_x: torch.Tensor | None = None,
    patched_shape: torch.Tensor | None = None,
) -> tuple[torch.Tensor, PatchMetadata, DITContext]:
    """Run a contiguous DiT block interval and optionally finish with the head.

    If ``patched_x`` is supplied, patch embedding is skipped.  The returned
    tensor is patched activations unless ``block_range[1] == dit.num_blocks``;
    callers that need x0 should use :func:`finish_dit` after the final stage.
    """
    lo, hi = block_range
    if patched_x is None:
        x, metadata = dit.patchify(hidden_states)
    else:
        if patched_shape is None:
            raise ValueError("patched_shape is required with patched_x")
        x = patched_x
        metadata = dit.metadata_from_shape(patched_shape, int(x.shape[0]))
    frame_seqlen = int(metadata.grid_sizes[0, 1:].prod().item())
    if start_frames is None:
        if starts.device.type != "cpu":
            raise ValueError("start_frames must be supplied when starts is a CUDA tensor")
        start_frames = starts // frame_seqlen
    if not isinstance(start_frames, torch.Tensor):
        start_frames = torch.tensor(list(start_frames), dtype=torch.long)
    context = dit.make_context(
        prompt_embeds,
        timesteps,
        metadata.grid_sizes,
        start_frames,
        current_start=starts,
        current_end=ends,
    )
    if not per_lane:
        x = dit.run_blocks(lo, hi, x, context, kv_slots, policy, crossattn_slots)
        return x, metadata, context

    lane_outputs: list[torch.Tensor] = []
    for lane in range(x.shape[0]):
        lane_x = x[lane:lane + 1]
        lane_kv = [_slot_lane(slot, lane) for slot in kv_slots]
        lane_cross = [_cross_lane(slot, lane) for slot in crossattn_slots]
        lane_x = dit.run_blocks(
            lo,
            hi,
            lane_x,
            context.lane(lane),
            lane_kv,
            policy,
            lane_cross,
            lane_ids=[lane],
        )
        lane_outputs.append(lane_x)
    return torch.cat(lane_outputs, dim=0), metadata, context


def finish_dit(
    dit: DITCore,
    patched_x: torch.Tensor,
    metadata: PatchMetadata,
    context: DITContext,
    hidden_states: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    flow = dit.head(patched_x, context, metadata)
    return dit.flow_to_x0(flow, hidden_states, timesteps)


def prepare_state(
    core: FrozenModel,
    plan: StreamPlan,
    prompt_embeds: torch.Tensor,
    initial_noisy_latents: torch.Tensor,
    *,
    block_range: tuple[int, int] = (0, 30),
) -> tuple[torch.Tensor, StreamState]:
    """Run upstream's batch-one preparation and expand caches to Stream-Batch."""
    dit = core.dit
    steps = plan.steps
    kv_slots = dit.allocate_kv_cache(1, plan.frame_seq_length)
    cross_slots = dit.allocate_crossattn_cache(1)
    policy = UpstreamKVPolicy(steps)
    starts = torch.zeros(1, dtype=torch.long, device=dit.device)
    ends = torch.full(
        (1,), 2 * plan.frame_seq_length, dtype=torch.long, device=dit.device
    )
    noisy = initial_noisy_latents
    denoised = noisy
    for index, timestep_value in enumerate(plan.denoising_steps):
        timestep = torch.full(
            (1, noisy.shape[1]), timestep_value, dtype=torch.long, device=dit.device
        )
        patched, metadata, context = forward_dit(
            dit,
            noisy,
            prompt_embeds,
            timestep,
            starts,
            ends,
            kv_slots,
            cross_slots,
            policy,
            start_frames=[0],
            block_range=block_range,
        )
        if block_range != (0, dit.num_blocks):
            raise ValueError("preparation must run all blocks before cache expansion")
        denoised = finish_dit(dit, patched, metadata, context, noisy, timestep)
        if index < steps - 1:
            next_timestep = torch.full(
                (1,),
                plan.denoising_steps[index + 1],
                dtype=torch.long,
                device=dit.device,
            )
            noise = plan.noise.step(0, index, latent_frames=denoised.shape[1])
            noisy = dit.add_noise(
                denoised.flatten(0, 1),
                noise.flatten(0, 1),
                next_timestep,
            ).unflatten(0, denoised.shape[:2])

    dit.expand_prepared_caches(kv_slots, cross_slots, steps)
    for block_index in range(dit.num_blocks):
        policy.expand_lanes(block_index, steps)
    latent_shape = (steps, 1, *initial_noisy_latents.shape[2:])
    state = StreamState(
        hidden_states=torch.zeros(latent_shape, dtype=torch.bfloat16, device=dit.device),
        chunk_indices=torch.zeros(steps, dtype=torch.long, device=dit.device),
        starts=torch.full((steps,), 2 * plan.frame_seq_length, dtype=torch.long, device=dit.device),
        ends=torch.full((steps,), 3 * plan.frame_seq_length, dtype=torch.long, device=dit.device),
        start_frames=[2] * steps,
        timesteps=torch.tensor(plan.denoising_steps, dtype=torch.long, device=dit.device),
        kv_slots=kv_slots,
        crossattn_slots=cross_slots,
        policy=policy,
    )
    return denoised, state


def stream_step(
    core: FrozenModel,
    plan: StreamPlan,
    prompt_embeds: torch.Tensor,
    state: StreamState,
    injected: torch.Tensor,
    *,
    chunk_idx: int,
    iteration: int,
    current_start: int,
    current_end: int,
    current_step: int | None,
    per_lane: bool = False,
) -> torch.Tensor:
    state.hidden_states[1:] = state.hidden_states[:-1].clone()
    state.hidden_states[0] = injected[0]
    state.chunk_indices[1:] = state.chunk_indices[:-1].clone()
    state.chunk_indices[0] = chunk_idx
    state.starts[1:] = state.starts[:-1].clone()
    state.starts[0] = current_start
    state.ends[1:] = state.ends[:-1].clone()
    state.ends[0] = current_end
    state.start_frames[1:] = state.start_frames[:-1]
    state.start_frames[0] = current_start // plan.frame_seq_length
    if current_step is not None:
        state.timesteps[0] = current_step
    timestep_matrix = state.timesteps.unsqueeze(1).expand(-1, state.hidden_states.shape[1])
    patched, metadata, context = forward_dit(
        core.dit,
        state.hidden_states,
        prompt_embeds,
        timestep_matrix,
        state.starts,
        state.ends,
        state.kv_slots,
        state.crossattn_slots,
        state.policy,
        start_frames=state.start_frames,
        per_lane=per_lane,
    )
    denoised = finish_dit(
        core.dit,
        patched,
        metadata,
        context,
        state.hidden_states,
        timestep_matrix,
    )
    for lane in range(plan.steps - 1):
        noise = plan.noise.step(int(state.chunk_indices[lane].item()), lane)
        next_timestep = torch.full(
            (1,),
            plan.denoising_steps[lane + 1],
            dtype=torch.long,
            device=core.dit.device,
        )
        denoised[[lane]] = core.dit.add_noise(denoised[[lane]], noise, next_timestep)
    state.hidden_states = denoised
    return denoised


@torch.inference_mode()
def run_case(
    plan: StreamPlan,
    prompt_embeds: torch.Tensor,
    device: torch.device | str,
    *,
    core: FrozenModel | None = None,
    sink: OutputSink | None = None,
    per_lane: bool = False,
    trace: dict[str, Any] | None = None,
    push_log: dict[int, int] | None = None,
) -> RunLedger:
    """Run the exact upstream single-GPU Stream-Batch loop with indexed noise.

    ``push_log``, when given, receives the monotonic time at which each real
    chunk entered the loop, so the caller can time this trusted reference with
    the same push-anchored definitions the Judge uses.
    """
    target = torch.device(device)
    local_plan = plan.on(target)
    model = core if core is not None else load_model(target, load_text_encoder=False)
    if model.dit.device != target:
        raise ValueError(f"provided model is on {model.dit.device}, requested {target}")
    output = sink if sink is not None else OutputSink()
    model.vae.reset()

    if push_log is not None:
        push_log[0] = time.monotonic_ns()
    first_frames = local_plan.frame_chunk(0).unsqueeze(0).to(dtype=torch.bfloat16)
    encoded = model.vae.stream_encode(first_frames).transpose(2, 1).contiguous().to(torch.bfloat16)
    initial_noise = local_plan.noise.input(0, latent_frames=encoded.shape[1])
    noisy = initial_noise * local_plan.chunks[0].noise_scale + encoded * (
        1 - local_plan.chunks[0].noise_scale
    )
    initial_x0, state = prepare_state(model, local_plan, prompt_embeds, noisy)
    # The prepare call needs two temporal latents to decode five frames, while
    # the frozen output contract exposes one finished-lane x0 per chunk.
    output.emit_latent(0, initial_x0[:, [-1]])
    initial_pixels = model.vae.stream_decode(initial_x0)
    output.emit_frames(0, initial_pixels[0].permute(0, 2, 3, 1).contiguous().float())

    total_iterations = (len(local_plan) - 1) + (local_plan.steps - 1)
    next_start = local_plan.chunks[-1].current_end
    next_end = next_start + local_plan.frame_seq_length
    for iteration in range(1, total_iterations + 1):
        if iteration < len(local_plan):
            chunk = local_plan.chunks[iteration]
            if push_log is not None:
                push_log[chunk.index] = time.monotonic_ns()
            frames = local_plan.frame_chunk(chunk).unsqueeze(0).to(dtype=torch.bfloat16)
            encoded = model.vae.stream_encode(frames).transpose(2, 1).contiguous().to(torch.bfloat16)
            injected = (
                local_plan.noise.input(chunk.index) * chunk.noise_scale
                + encoded * (1 - chunk.noise_scale)
            )
            current_start = chunk.current_start
            current_end = chunk.current_end
            current_step: int | None = chunk.current_step
            next_start = current_end
            next_end = next_start + local_plan.frame_seq_length
        else:
            if next_start // local_plan.frame_seq_length >= local_plan.t_refresh:
                next_start = local_plan.kv_cache_length - local_plan.frame_seq_length
                next_end = next_start + local_plan.frame_seq_length
            injected = local_plan.noise.input(iteration)
            current_start = next_start
            current_end = next_end
            current_step = None
            next_start = next_end
            next_end += local_plan.frame_seq_length

        denoised = stream_step(
            model,
            local_plan,
            prompt_embeds,
            state,
            injected,
            chunk_idx=iteration,
            iteration=iteration,
            current_start=current_start,
            current_end=current_end,
            current_step=current_step,
            per_lane=per_lane,
        )
        if iteration >= local_plan.steps:
            output_chunk = iteration - local_plan.steps + 1
            finished = denoised[[-1]]
            output.emit_latent(output_chunk, finished)
            pixels = model.vae.stream_decode(finished)
            output.emit_frames(
                output_chunk,
                pixels[0].permute(0, 2, 3, 1).contiguous().float(),
            )

    ledger = output.ledger()
    expected = set(range(len(local_plan)))
    if set(ledger.frames) != expected or set(ledger.latents) != expected:
        raise RuntimeError(
            f"incomplete output: frames={sorted(ledger.frames)}, latents={sorted(ledger.latents)}, "
            f"expected={sorted(expected)}"
        )
    if trace is not None:
        trace.update(
            {
                "kv_wrap_count": state.policy.wrap_count,
                "sink_refresh_count": state.policy.sink_refresh_count,
                "kv_events": list(state.policy.events),
            }
        )
    return ledger
