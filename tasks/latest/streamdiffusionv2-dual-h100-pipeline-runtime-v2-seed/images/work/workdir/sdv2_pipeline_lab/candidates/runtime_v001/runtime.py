# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Upstream lag-1 NCCL ring with chunk/step-keyed stochastic replay."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import os
import socket
from pathlib import Path
import time
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sdv2_core.loader import FrozenModel, load_model
from sdv2_core.plan import NoiseProvider, StreamPlan, StreamPlanMeta
from sdv2_core.reference import StreamState, finish_dit, forward_dit, prepare_state
from sdv2_core.sink import OutputSink, RunLedger


LATENT_HDR = 11001
LATENT_PAY = 11002
PATCHED_X_SHAPE = 11004
LATENT_ORIGIN_HDR = 11005
LATENT_ORIGIN_PAY = 11006
LANE_METADATA = 11007


def _trace(rank: int, message: str) -> None:
    if os.environ.get("SDV2_RING_TRACE") == "1":
        print(f"[lag-ring rank{rank}] {message}", flush=True)


@dataclass(frozen=True)
class _PlanMeta:
    steps: int
    case_seed: int
    frame_seq_length: int
    kv_cache_length: int
    t_refresh: int
    denoising_steps: tuple[int, ...]
    num_chunks: int
    latent_height: int
    latent_width: int

    @classmethod
    def from_plan(cls, plan: StreamPlanMeta) -> "_PlanMeta":
        return cls(
            steps=plan.steps,
            case_seed=plan.case_seed,
            frame_seq_length=plan.frame_seq_length,
            kv_cache_length=plan.kv_cache_length,
            t_refresh=plan.t_refresh,
            denoising_steps=plan.denoising_steps,
            num_chunks=len(plan),
            latent_height=plan.noise.latent_height,
            latent_width=plan.noise.latent_width,
        )

    def on(self, device: torch.device | str) -> "_ChildPlan":
        return _ChildPlan(self, device)


class _ChildPlan:
    """Small duck-typed subset consumed by preparation and re-noising."""

    def __init__(self, meta: _PlanMeta, device: torch.device | str) -> None:
        self.steps = meta.steps
        self.case_seed = meta.case_seed
        self.frame_seq_length = meta.frame_seq_length
        self.kv_cache_length = meta.kv_cache_length
        self.t_refresh = meta.t_refresh
        self.denoising_steps = meta.denoising_steps
        self.noise = NoiseProvider(
            case_seed=meta.case_seed,
            steps=meta.steps,
            latent_height=meta.latent_height,
            latent_width=meta.latent_width,
            device=device,
        )


@dataclass
class _Payload:
    batch_idx: int
    activations: torch.Tensor
    original_latents: torch.Tensor
    patched_shape: torch.Tensor
    chunk_indices: torch.Tensor
    starts: torch.Tensor
    ends: torch.Tensor
    timesteps: torch.Tensor
    start_frames: torch.Tensor


@dataclass
class _LaneSnapshot:
    hidden_states: torch.Tensor
    chunk_indices: torch.Tensor
    starts: torch.Tensor
    ends: torch.Tensor
    timesteps: torch.Tensor
    start_frames: list[int]


@dataclass
class _CacheLaneSnapshot:
    block: int
    tensors: dict[str, torch.Tensor]
    policy_lanes: dict[int, dict[str, Any]]


@dataclass
class _Outstanding:
    work: list[Any]
    keepalive: list[torch.Tensor]

    def wait(self) -> None:
        for item in self.work:
            item.wait()
        self.keepalive.clear()


def _header(batch_idx: int, tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    return torch.tensor([batch_idx, *tensor.shape], dtype=torch.int64, device=device)


def _send_payload(payload: _Payload, dst: int, device: torch.device) -> _Outstanding:
    latent_header = _header(payload.batch_idx, payload.activations, device)
    origin_header = _header(payload.batch_idx, payload.original_latents, device)
    lane_metadata = torch.cat(
        [
            payload.chunk_indices,
            payload.starts,
            payload.ends,
            payload.timesteps,
            payload.start_frames,
        ]
    )
    tensors = [
        latent_header,
        payload.activations,
        origin_header,
        payload.original_latents,
        payload.patched_shape,
        lane_metadata,
    ]
    tags = [
        LATENT_HDR,
        LATENT_PAY,
        LATENT_ORIGIN_HDR,
        LATENT_ORIGIN_PAY,
        PATCHED_X_SHAPE,
        LANE_METADATA,
    ]
    work = [dist.isend(tensor, dst=dst, tag=tag) for tensor, tag in zip(tensors, tags)]
    return _Outstanding(work=work, keepalive=tensors)


def _receive_header(src: int, length: int, tag: int, device: torch.device) -> tuple[int, tuple[int, ...]]:
    header = torch.empty(length, dtype=torch.int64, device=device)
    dist.recv(header, src=src, tag=tag)
    values = header.tolist()
    return int(values[0]), tuple(int(value) for value in values[1:])


def _receive_payload(src: int, steps: int, device: torch.device) -> _Payload:
    batch_idx, activation_shape = _receive_header(src, 4, LATENT_HDR, device)
    activations = torch.empty(activation_shape, dtype=torch.bfloat16, device=device)
    dist.recv(activations, src=src, tag=LATENT_PAY)
    origin_chunk, origin_shape = _receive_header(src, 6, LATENT_ORIGIN_HDR, device)
    if origin_chunk != batch_idx:
        raise RuntimeError(f"payload header mismatch: {batch_idx} != {origin_chunk}")
    original = torch.empty(origin_shape, dtype=torch.bfloat16, device=device)
    dist.recv(original, src=src, tag=LATENT_ORIGIN_PAY)
    patched_shape = torch.empty(5, dtype=torch.int64, device=device)
    dist.recv(patched_shape, src=src, tag=PATCHED_X_SHAPE)
    lane_metadata = torch.empty(5 * steps, dtype=torch.int64, device=device)
    dist.recv(lane_metadata, src=src, tag=LANE_METADATA)
    return _Payload(
        batch_idx=batch_idx,
        activations=activations,
        original_latents=original,
        patched_shape=patched_shape,
        chunk_indices=lane_metadata[:steps].clone(),
        starts=lane_metadata[steps:2 * steps].clone(),
        ends=lane_metadata[2 * steps:3 * steps].clone(),
        timesteps=lane_metadata[3 * steps:4 * steps].clone(),
        start_frames=lane_metadata[4 * steps:].clone(),
    )


def _broadcast_prepared_state(state: StreamState, src: int) -> None:
    for kv_slot, cross_slot in zip(state.kv_slots, state.crossattn_slots):
        for name in ("k", "v", "global_end_index", "local_end_index"):
            dist.broadcast(kv_slot[name], src=src)
        # Cross-attention was expanded with stride zero exactly like upstream;
        # broadcast the one physical prompt cache, then restore the batch view.
        for name in ("k", "v"):
            base = cross_slot[name][0:1].contiguous()
            dist.broadcast(base, src=src)
            cross_slot[name] = base.expand(state.hidden_states.shape[0], -1, -1, -1)


def _offload_unowned(state: StreamState, owned: tuple[int, int]) -> None:
    lo, hi = owned
    for block_index, kv_slot in enumerate(state.kv_slots):
        if lo <= block_index < hi:
            continue
        for name in ("k", "v", "global_end_index", "local_end_index"):
            kv_slot[name] = kv_slot[name].cpu()
    # Prompt K/V is immutable, small relative to rolling self-attention K/V,
    # and remains resident on both ranks so the optional scheduler can change
    # block ownership without another state transfer.


def _snapshot_cache_lanes(
    state: StreamState,
    owned: tuple[int, int],
    lanes: list[int],
) -> list[_CacheLaneSnapshot]:
    """Save speculative warm-up lanes so duplicate batch-2 writes can roll back."""
    if not lanes:
        return []
    lo, hi = owned
    snapshots: list[_CacheLaneSnapshot] = []
    for block in range(lo, hi):
        slot = state.kv_slots[block]
        policy_state = state.policy.export_state(block)
        snapshots.append(
            _CacheLaneSnapshot(
                block=block,
                tensors={name: slot[name][lanes].clone() for name in (
                    "k", "v", "global_end_index", "local_end_index"
                )},
                policy_lanes={
                    lane: deepcopy(policy_state["lanes"][lane]) for lane in lanes
                },
            )
        )
    return snapshots


def _restore_cache_lanes(
    state: StreamState,
    lanes: list[int],
    snapshots: list[_CacheLaneSnapshot],
) -> None:
    for snapshot in snapshots:
        slot = state.kv_slots[snapshot.block]
        for name, value in snapshot.tensors.items():
            slot[name][lanes] = value
        policy_state = state.policy.export_state(snapshot.block)
        for lane, lane_state in snapshot.policy_lanes.items():
            policy_state["lanes"][lane] = lane_state
        state.policy.import_state(snapshot.block, policy_state)


def _child_main(
    core_loader: Callable[..., FrozenModel],
    device_index: int,
    init_method: str,
    split: int,
    meta: _PlanMeta,
    output_dir: str,
) -> None:
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        rank=1,
        world_size=2,
    )
    try:
        core = core_loader(device, load_text_encoder=False)
        plan = meta.on(device)
        # __init__ returns only after both ranks initialized NCCL and loaded
        # the model. The child then blocks here until start() provides prompt.
        dist.barrier()
        prompt_ndim = torch.empty(1, dtype=torch.int64, device=device)
        dist.broadcast(prompt_ndim, src=0)
        prompt_shape = torch.empty(int(prompt_ndim.item()), dtype=torch.int64, device=device)
        dist.broadcast(prompt_shape, src=0)
        prompt_embeds = torch.empty(
            tuple(int(value) for value in prompt_shape.tolist()),
            dtype=torch.bfloat16,
            device=device,
        )
        dist.broadcast(prompt_embeds, src=0)
        initial_noisy = torch.empty(
            (1, 2, 16, meta.latent_height, meta.latent_width),
            dtype=torch.bfloat16,
            device=device,
        )
        dist.broadcast(initial_noisy, src=0)
        initial_x0, state = prepare_state(core, plan, prompt_embeds, initial_noisy)
        dist.broadcast(initial_x0, src=0)
        _broadcast_prepared_state(state, src=0)
        _offload_unowned(state, (split, core.dit.num_blocks))

        sink = OutputSink()
        core.vae.reset()
        initial_pixels = core.vae.stream_decode(initial_x0)
        sink.emit_latent(0, initial_x0[:, [-1]])
        sink.emit_frames(0, initial_pixels[0].permute(0, 2, 3, 1).contiguous().float())
        dist.barrier()

        # One denoising lane advances every two ring batches. Real pushes plus
        # exactly 2*(S-1) fillers finish the last chunk without an orphan send.
        message_count = meta.num_chunks - 1 + 2 * (meta.steps - 1)
        outstanding: _Outstanding | None = None
        next_output = 1
        for processed in range(1, message_count + 1):
            _trace(1, f"recv-start batch={processed}")
            payload = _receive_payload(0, meta.steps, device)
            _trace(1, f"recv-done batch={processed}")
            if payload.batch_idx != processed:
                raise RuntimeError(
                    f"rank 1 expected batch {processed}, got {payload.batch_idx}"
                )
            if outstanding is not None:
                outstanding.wait()
                outstanding = None
            state.hidden_states.copy_(payload.original_latents)
            state.chunk_indices.copy_(payload.chunk_indices)
            state.starts.copy_(payload.starts)
            state.ends.copy_(payload.ends)
            state.timesteps.copy_(payload.timesteps)
            state.start_frames = [int(value) for value in payload.start_frames.tolist()]
            rollback_lanes = (
                [
                    lane
                    for lane, chunk_idx in enumerate(payload.chunk_indices.tolist())
                    if int(chunk_idx) == 0
                ]
                if processed % 2 == 0
                else []
            )
            cache_snapshot = _snapshot_cache_lanes(
                state, (split, core.dit.num_blocks), rollback_lanes
            )
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
                block_range=(split, core.dit.num_blocks),
                patched_x=payload.activations,
                patched_shape=payload.patched_shape.cpu(),
            )
            _restore_cache_lanes(state, rollback_lanes, cache_snapshot)
            _trace(1, f"dit-done batch={processed}")
            denoised = finish_dit(
                core.dit,
                patched,
                metadata,
                context,
                state.hidden_states,
                timestep_matrix,
            )
            for lane in range(meta.steps - 1):
                chunk_idx = int(state.chunk_indices[lane].item())
                noise = plan.noise.step(chunk_idx, lane)
                next_timestep = torch.full(
                    (1,),
                    meta.denoising_steps[lane + 1],
                    dtype=torch.long,
                    device=device,
                )
                denoised[[lane]] = core.dit.add_noise(denoised[[lane]], noise, next_timestep)
            state.hidden_states = denoised

            outstanding = _send_payload(
                _Payload(
                    batch_idx=payload.batch_idx,
                    activations=payload.activations,
                    original_latents=denoised,
                    patched_shape=payload.patched_shape,
                    chunk_indices=state.chunk_indices,
                    starts=state.starts,
                    ends=state.ends,
                    timesteps=state.timesteps,
                    start_frames=torch.tensor(
                        state.start_frames, dtype=torch.long, device=device
                    ),
                ),
                dst=0,
                device=device,
            )
            _trace(1, f"send-issued batch={processed}")
            if processed >= 2 * meta.steps - 1:
                output_chunk = int(state.chunk_indices[-1].item())
                if output_chunk != next_output or output_chunk >= meta.num_chunks:
                    raise RuntimeError(
                        f"rank 1 expected output chunk {next_output}, got "
                        f"{output_chunk} from batch {processed}"
                    )
                finished = denoised[[-1]]
                sink.emit_latent(output_chunk, finished)
                pixels = core.vae.stream_decode(finished)
                sink.emit_frames(
                    output_chunk,
                    pixels[0].permute(0, 2, 3, 1).contiguous().float(),
                )
                next_output += 1
                _trace(1, f"emit chunk={output_chunk} batch={processed}")
        if outstanding is not None:
            outstanding.wait()
        if next_output != meta.num_chunks:
            raise RuntimeError(
                f"rank 1 emitted through chunk {next_output - 1}, "
                f"expected {meta.num_chunks - 1}"
            )
        sink.save(output_dir)
        sink.close()
        dist.barrier()
    finally:
        dist.destroy_process_group()


class Runtime:
    """Upstream lag-1 two-rank ring with a static configurable block split.

    Rank 1 writes its sink ledger and tensor archives under ``lab/build``.
    ``close`` merges those files into the driver-owned sink, preserving child
    monotonic timestamps; no frame payload is copied through a multiprocessing
    pipe. Rank 0 completes its DiT for batch t before receiving batch t-1 and
    forwarding t, keeping the two DiT stages overlapped while preserving
    per-chunk stochastic identity.
    """

    def __init__(
        self,
        core_loader: Callable[..., FrozenModel],
        devices: list[torch.device | str],
        plan_meta: StreamPlanMeta,
        sink: OutputSink,
        *,
        split: int = 15,
        master_port: int | None = None,
    ) -> None:
        if len(devices) != 2:
            raise ValueError("runtime_v001 requires exactly two devices")
        if not 1 <= split <= 29:
            raise ValueError("split must be in [1, 29]")
        if master_port is None:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                listener.bind(("127.0.0.1", 0))
                master_port = listener.getsockname()[1]
        if isinstance(master_port, bool) or not isinstance(master_port, int) or not 1 <= master_port <= 65535:
            raise ValueError("master_port must be an integer in [1, 65535]")
        self.core_loader = core_loader
        self.devices = [torch.device(value) for value in devices]
        if any(device.type != "cuda" for device in self.devices):
            raise ValueError("runtime_v001 requires CUDA devices")
        self.plan = plan_meta
        self.sink = sink
        self.split = int(split)
        self.master_port = int(master_port)
        self._ctx = mp.get_context("spawn")
        self._process: mp.Process | None = None
        self._core: FrozenModel | None = None
        self._state: StreamState | None = None
        self._prompt_embeds: torch.Tensor | None = None
        self._outstanding: _Outstanding | None = None
        self._bootstrap: _LaneSnapshot | None = None
        self._iterations = 0
        self._next_real_chunk = 1
        self._flushed = False
        self._closed = False
        self._started = False
        build_root = Path(__file__).resolve().parents[2]
        self._child_output = build_root / "results" / "runtime_v001_runs" / (
            f"pid{os.getpid()}_{time.monotonic_ns()}_split{split}"
        )
        self._child_output.mkdir(parents=True, exist_ok=False)

        init_method = f"tcp://127.0.0.1:{self.master_port}"
        meta = _PlanMeta.from_plan(self.plan)
        self._process = self._ctx.Process(
            target=_child_main,
            args=(
                self.core_loader,
                int(self.devices[1].index or 0),
                init_method,
                self.split,
                meta,
                str(self._child_output),
            ),
            daemon=False,
        )
        self._process.start()
        torch.cuda.set_device(self._rank0_device)
        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            rank=0,
            world_size=2,
        )
        self._core = self.core_loader(self._rank0_device, load_text_encoder=False)
        dist.barrier()

    @property
    def _rank0_device(self) -> torch.device:
        return self.devices[0]

    def start(self, prompt_embeds: torch.Tensor, first_chunk_frames: torch.Tensor) -> None:
        if self._started:
            raise RuntimeError("runtime has already started")
        if self._closed:
            raise RuntimeError("runtime has already closed")
        prompt_cpu = prompt_embeds.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        torch.cuda.set_device(self._rank0_device)
        self._prompt_embeds = prompt_cpu.to(self._rank0_device)
        prompt_ndim = torch.tensor(
            [self._prompt_embeds.ndim], dtype=torch.int64, device=self._rank0_device
        )
        prompt_shape = torch.tensor(
            self._prompt_embeds.shape, dtype=torch.int64, device=self._rank0_device
        )
        dist.broadcast(prompt_ndim, src=0)
        dist.broadcast(prompt_shape, src=0)
        dist.broadcast(self._prompt_embeds, src=0)
        local_plan = self.plan.on(self._rank0_device)
        frames = first_chunk_frames
        if frames.ndim == 4:
            frames = frames.unsqueeze(0)
        frames = frames.to(device=self._rank0_device, dtype=torch.bfloat16)
        encoded = self._core.vae.stream_encode(frames).transpose(2, 1).contiguous().to(torch.bfloat16)
        noise = local_plan.noise.input(0, latent_frames=encoded.shape[1])
        initial_noisy = noise * local_plan.chunks[0].noise_scale + encoded * (
            1 - local_plan.chunks[0].noise_scale
        )
        dist.broadcast(initial_noisy, src=0)
        initial_x0, self._state = prepare_state(
            self._core,
            local_plan,
            self._prompt_embeds,
            initial_noisy,
        )
        self._bootstrap = _LaneSnapshot(
            hidden_states=self._state.hidden_states.clone(),
            chunk_indices=self._state.chunk_indices.clone(),
            starts=self._state.starts.clone(),
            ends=self._state.ends.clone(),
            timesteps=self._state.timesteps.clone(),
            start_frames=list(self._state.start_frames),
        )
        dist.broadcast(initial_x0, src=0)
        _broadcast_prepared_state(self._state, src=0)
        _offload_unowned(self._state, (0, self.split))
        dist.barrier()
        self._started = True

    def _require_started(self) -> tuple[FrozenModel, StreamState, torch.Tensor]:
        if self._core is None or self._state is None or self._prompt_embeds is None:
            raise RuntimeError("start must be called before streaming")
        return self._core, self._state, self._prompt_embeds

    def _iteration(
        self,
        injected: torch.Tensor,
        *,
        chunk_idx: int,
        current_start: int,
        current_end: int,
        current_step: int | None,
    ) -> None:
        core, state, prompt_embeds = self._require_started()
        # Release the previous send's in-place source before shifting lanes.
        # Rank 1 can still be computing that batch while rank 0 starts here.
        if self._outstanding is not None:
            self._outstanding.wait()
            self._outstanding = None
        # Batch 2 is the second in-flight stream and starts from the same
        # post-prepare state as batch 1. Thereafter return t-1 is installed at
        # the end of batch t, so each parity stream advances every two batches.
        if self._iterations == 1:
            if self._bootstrap is None:
                raise RuntimeError("missing second-batch bootstrap state")
            state.hidden_states.copy_(self._bootstrap.hidden_states)
            state.chunk_indices.copy_(self._bootstrap.chunk_indices)
            state.starts.copy_(self._bootstrap.starts)
            state.ends.copy_(self._bootstrap.ends)
            state.timesteps.copy_(self._bootstrap.timesteps)
            state.start_frames = list(self._bootstrap.start_frames)
        self._iterations += 1
        _trace(0, f"dit-start batch={self._iterations}")
        state.hidden_states[1:] = state.hidden_states[:-1].clone()
        state.hidden_states[0] = injected[0]
        state.chunk_indices[1:] = state.chunk_indices[:-1].clone()
        state.chunk_indices[0] = chunk_idx
        state.starts[1:] = state.starts[:-1].clone()
        state.starts[0] = current_start
        state.ends[1:] = state.ends[:-1].clone()
        state.ends[0] = current_end
        state.start_frames[1:] = state.start_frames[:-1]
        state.start_frames[0] = current_start // self.plan.frame_seq_length
        if current_step is not None:
            state.timesteps[0] = current_step
        timestep_matrix = state.timesteps.unsqueeze(1).expand(-1, state.hidden_states.shape[1])
        rollback_lanes = (
            [
                lane
                for lane, candidate_chunk in enumerate(state.chunk_indices.tolist())
                if int(candidate_chunk) == 0
            ]
            if self._iterations % 2 == 0
            else []
        )
        cache_snapshot = _snapshot_cache_lanes(
            state, (0, self.split), rollback_lanes
        )
        patched, metadata, _ = forward_dit(
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
            block_range=(0, self.split),
        )
        _restore_cache_lanes(state, rollback_lanes, cache_snapshot)
        _trace(0, f"dit-done batch={self._iterations}")
        received: _Payload | None = None
        # Match inference_pipe.run_rank_0_loop's NCCL ordering: after issuing
        # rank-0 DiT(t), receive return(t-1), then send batch(t). The returned
        # state is installed only after batch(t)'s source is released.
        if self._iterations >= 2:
            _trace(0, f"recv-start batch={self._iterations - 1}")
            received = _receive_payload(1, self.plan.steps, self._rank0_device)
            _trace(0, f"recv-done batch={received.batch_idx}")
            if received.batch_idx != self._iterations - 1:
                raise RuntimeError(
                    f"rank 0 expected batch {self._iterations - 1}, "
                    f"got {received.batch_idx}"
                )
        self._outstanding = _send_payload(
            _Payload(
                batch_idx=self._iterations,
                activations=patched,
                original_latents=state.hidden_states,
                patched_shape=metadata.shape,
                chunk_indices=state.chunk_indices,
                starts=state.starts,
                ends=state.ends,
                timesteps=state.timesteps,
                start_frames=torch.tensor(
                    state.start_frames, dtype=torch.long, device=self._rank0_device
                ),
            ),
            dst=1,
            device=self._rank0_device,
        )
        _trace(0, f"send-issued batch={self._iterations}")
        if received is not None:
            self._outstanding.wait()
            self._outstanding = None
            self._install_return(state, received)

    def _install_return(self, state: StreamState, received: _Payload) -> None:
        state.hidden_states.copy_(received.original_latents)
        state.chunk_indices.copy_(received.chunk_indices)
        state.starts.copy_(received.starts)
        state.ends.copy_(received.ends)
        state.timesteps.copy_(received.timesteps)
        state.start_frames = [int(value) for value in received.start_frames.tolist()]

    def push(self, chunk_idx: int, frames: torch.Tensor) -> None:
        core, _, _ = self._require_started()
        if self._flushed:
            raise RuntimeError("cannot push after flush")
        if chunk_idx != self._next_real_chunk:
            raise ValueError(f"expected chunk {self._next_real_chunk}, got {chunk_idx}")
        if chunk_idx >= len(self.plan):
            raise IndexError(f"chunk {chunk_idx} is outside this plan")
        descriptor = self.plan.chunks[chunk_idx]
        values = frames.unsqueeze(0) if frames.ndim == 4 else frames
        values = values.to(device=self._rank0_device, dtype=torch.bfloat16)
        encoded = core.vae.stream_encode(values).transpose(2, 1).contiguous().to(torch.bfloat16)
        noise = self.plan.noise.on(self._rank0_device).input(chunk_idx)
        injected = noise * descriptor.noise_scale + encoded * (1 - descriptor.noise_scale)
        self._iteration(
            injected,
            chunk_idx=chunk_idx,
            current_start=descriptor.current_start,
            current_end=descriptor.current_end,
            current_step=descriptor.current_step,
        )
        self._next_real_chunk += 1

    def flush(self) -> None:
        if self._flushed:
            return
        self._require_started()
        if self._next_real_chunk != len(self.plan):
            raise RuntimeError(
                f"cannot flush before all real chunks are pushed: next={self._next_real_chunk}, total={len(self.plan)}"
            )
        filler_count = 2 * (self.plan.steps - 1)
        next_start = self.plan.chunks[-1].current_end
        next_end = next_start + self.plan.frame_seq_length
        noise = self.plan.noise.on(self._rank0_device)
        for filler in range(filler_count):
            if next_start // self.plan.frame_seq_length >= self.plan.t_refresh:
                next_start = self.plan.kv_cache_length - self.plan.frame_seq_length
                next_end = next_start + self.plan.frame_seq_length
            chunk_idx = len(self.plan) + filler
            self._iteration(
                noise.input(chunk_idx),
                chunk_idx=chunk_idx,
                current_start=next_start,
                current_end=next_end,
                current_step=None,
            )
            next_start = next_end
            next_end += self.plan.frame_seq_length
        # Drain the final return without issuing the unmatched speculative tail
        # present in upstream. Every send has exactly one receive.
        if self._iterations:
            _trace(0, f"final-recv-start batch={self._iterations}")
            if self._outstanding is not None:
                self._outstanding.wait()
                self._outstanding = None
            received = _receive_payload(1, self.plan.steps, self._rank0_device)
            _trace(0, f"final-recv-done batch={received.batch_idx}")
            if received.batch_idx != self._iterations:
                raise RuntimeError(
                    f"rank 0 expected final batch {self._iterations}, "
                    f"got {received.batch_idx}"
                )
            _, state, _ = self._require_started()
            self._install_return(state, received)
        dist.barrier()
        self._flushed = True

    def close(self) -> RunLedger:
        if self._closed:
            return self.sink.ledger()
        if not self._flushed:
            self.flush()
        dist.destroy_process_group()
        assert self._process is not None
        self._process.join(timeout=600)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=30)
            raise TimeoutError("rank-1 child did not exit after flush")
        if self._process.exitcode != 0:
            raise RuntimeError(f"rank-1 child exited with code {self._process.exitcode}")
        self.sink.merge_saved(self._child_output)
        self._closed = True
        return self.sink.ledger()


@torch.inference_mode()
def run_case(
    plan: StreamPlan,
    prompt_embeds: torch.Tensor,
    *,
    devices: list[torch.device | str] | None = None,
    split: int = 15,
    master_port: int | None = None,
) -> RunLedger:
    """Convenience adapter used by qualification tests."""
    selected = devices or ["cuda:0", "cuda:1"]
    sink = OutputSink()
    runtime = Runtime(
        load_model,
        selected,
        plan.meta(),
        sink,
        split=split,
        master_port=master_port,
    )
    runtime.start(prompt_embeds, plan.frame_chunk(0))
    for chunk in plan.chunks[1:]:
        runtime.push(chunk.index, plan.frame_chunk(chunk))
    runtime.flush()
    return runtime.close()
