# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rolling KV-cache seam and the bit-exact upstream bookkeeping policy."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, MutableMapping, Protocol, Sequence

import torch
import torch.nn.functional as F


KVSlot = MutableMapping[str, torch.Tensor]


@dataclass(frozen=True)
class KVUpdate:
    """Policy decision consumed by frozen attention math.

    ``read_len`` is a Python integer and must be greater than or equal to every
    entry of ``seq_lens``. An optimized policy may return the fixed cache
    capacity to keep the frozen path free of device-to-host reads.
    """

    seq_lens: torch.Tensor
    read_len: int


class KVPolicy(Protocol):
    def update(
        self,
        block: int,
        kv_slot: KVSlot,
        roped_key: torch.Tensor,
        value: torch.Tensor,
        current_start: torch.Tensor,
        current_end: torch.Tensor,
        frame_seqlen: int,
        *,
        sink_size: int,
        adapt_thr: float,
        lane_ids: Sequence[int] | None = None,
    ) -> KVUpdate: ...

    def export_state(self, block: int) -> dict[str, Any]: ...

    def import_state(self, block: int, state: dict[str, Any]) -> None: ...


@dataclass
class LaneState:
    evict_idx: list[int]
    current_step: int
    total_steps: int


class UpstreamKVPolicy:
    """Line-for-line semantic port of the upstream KV branch.

    Python state formerly stored on ``CausalWanSelfAttention`` and scalar
    ``current_step``/``total_steps`` fields formerly stored in each cache dict
    live here per block and lane.  Tensor writes and branch ordering match the
    pinned source.
    """

    def __init__(self, total_steps: int) -> None:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        self.total_steps = int(total_steps)
        self._states: dict[int, list[LaneState]] = {}
        self.wrap_count = 0
        self.sink_refresh_count = 0
        self.events: list[dict[str, int | str]] = []

    def _ensure_lanes(self, block: int, lane_count: int) -> list[LaneState]:
        states = self._states.get(block)
        if states is None:
            states = [LaneState([], self.total_steps, self.total_steps)]
            self._states[block] = states
        while len(states) < lane_count:
            states.append(deepcopy(states[0]))
        return states

    def expand_lanes(self, block: int, lane_count: int) -> None:
        self._ensure_lanes(block, lane_count)

    def update(
        self,
        block: int,
        kv_slot: KVSlot,
        roped_key: torch.Tensor,
        value: torch.Tensor,
        current_start: torch.Tensor,
        current_end: torch.Tensor,
        frame_seqlen: int,
        *,
        sink_size: int,
        adapt_thr: float,
        lane_ids: Sequence[int] | None = None,
    ) -> KVUpdate:
        del current_end  # Upstream recomputes this from current_start and query length.
        required = {"k", "v", "global_end_index", "local_end_index"}
        missing = required.difference(kv_slot)
        if missing:
            raise KeyError(f"KV slot is missing fields: {sorted(missing)}")
        if roped_key.shape != value.shape:
            raise ValueError("key and value shapes must match")
        cache_bs = int(kv_slot["k"].shape[0])
        if roped_key.shape[0] != cache_bs:
            raise ValueError("cache, key, and value batch sizes must match")
        if current_start.ndim == 0:
            current_start = current_start.expand(cache_bs)
        if current_start.numel() != cache_bs:
            raise ValueError("current_start must have one entry per cache batch lane")
        if lane_ids is None:
            lane_ids = tuple(range(cache_bs))
        if len(lane_ids) != cache_bs:
            raise ValueError("lane_ids must have one entry per cache batch lane")

        state_count = max(lane_ids, default=-1) + 1
        states = self._ensure_lanes(block, state_count)
        seq_lens: list[int] = []
        kv_cache_size = int(kv_slot["k"].shape[1])
        num_new_tokens = int(roped_key.shape[1])
        sink_tokens = int(sink_size) * int(frame_seqlen)

        for local_lane, (c_start, policy_lane) in enumerate(zip(current_start, lane_ids)):
            state = states[policy_lane]
            c_end = c_start + num_new_tokens

            if sink_tokens > 0 and adapt_thr > -1 and value.shape[1] <= frame_seqlen:
                k_sink_mean = (
                    kv_slot["k"][local_lane:local_lane + 1, :sink_tokens]
                    .reshape(sink_size, frame_seqlen, -1)
                    .mean(1)
                )
                k_new_mean = roped_key[local_lane:local_lane + 1].reshape(1, frame_seqlen, -1).mean(1)
                k_cos_sim = F.cosine_similarity(k_sink_mean, k_new_mean, dim=-1)

                v_sink_mean = (
                    kv_slot["v"][local_lane:local_lane + 1, :sink_tokens]
                    .reshape(sink_size, frame_seqlen, -1)
                    .mean(1)
                )
                v_new_mean = value[local_lane:local_lane + 1].reshape(1, frame_seqlen, -1).mean(1)
                v_cos_sim = F.cosine_similarity(v_sink_mean, v_new_mean, dim=-1)

                avg_cos_sim = (k_cos_sim + v_cos_sim) / 2
                if avg_cos_sim.min() < adapt_thr:
                    idx = torch.argmin(avg_cos_sim).item()
                    temp_evict_idx = (idx + 1) * frame_seqlen
                    state.evict_idx.insert(0, temp_evict_idx)
                    self.sink_refresh_count += 1
                    self.events.append(
                        {"kind": "sink_refresh", "block": block, "lane": policy_lane, "target": temp_evict_idx}
                    )

            if c_end > kv_cache_size or kv_slot["local_end_index"][local_lane] >= kv_cache_size:
                kv_slot["global_end_index"][local_lane].fill_(c_start)
                kv_slot["local_end_index"][local_lane].fill_(kv_cache_size)
                if not state.evict_idx:
                    raise RuntimeError(
                        f"upstream eviction queue is empty for block={block}, lane={policy_lane}"
                    )
                target_end = state.evict_idx[0]

                # ``len(states)`` is the logical cache batch.  It remains B
                # when a numerical-envelope probe presents one lane view at a
                # time, while preparation genuinely has one logical lane.
                if len(states) == 1 and state.current_step > 1:
                    state.current_step -= 1
                else:
                    evict_idx = state.evict_idx.pop(0)
                    if evict_idx > sink_tokens:
                        state.evict_idx.append(evict_idx)
                    state.current_step = state.total_steps

                kv_slot["k"][local_lane:local_lane + 1, target_end - num_new_tokens:target_end] = roped_key[
                    local_lane:local_lane + 1
                ]
                kv_slot["v"][local_lane:local_lane + 1, target_end - num_new_tokens:target_end] = value[
                    local_lane:local_lane + 1
                ]
                local_end_index = kv_slot["local_end_index"][local_lane].item()
                self.wrap_count += 1
                self.events.append(
                    {"kind": "wrap", "block": block, "lane": policy_lane, "target": target_end}
                )
            else:
                local_end_index = (
                    kv_slot["local_end_index"][local_lane].item()
                    + c_end
                    - kv_slot["global_end_index"][local_lane].item()
                )
                rolling_end = (c_end + num_new_tokens).item()
                if (
                    rolling_end > sink_size * frame_seqlen
                    and rolling_end <= kv_cache_size
                    and (not state.evict_idx or state.evict_idx[-1] != rolling_end)
                ):
                    state.evict_idx.append(rolling_end)

                local_start_index = local_end_index - num_new_tokens
                kv_slot["k"][local_lane:local_lane + 1, local_start_index:local_end_index] = roped_key[
                    local_lane:local_lane + 1
                ]
                kv_slot["v"][local_lane:local_lane + 1, local_start_index:local_end_index] = value[
                    local_lane:local_lane + 1
                ]

            seq_lens.append(int(local_end_index))
            kv_slot["global_end_index"][local_lane].fill_(c_end)
            kv_slot["local_end_index"][local_lane].fill_(local_end_index)

        return KVUpdate(
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=roped_key.device),
            read_len=max(seq_lens),
        )

    def export_state(self, block: int) -> dict[str, Any]:
        states = self._states.get(block, [])
        return {
            "block": int(block),
            "lanes": [asdict(state) for state in states],
        }

    def import_state(self, block: int, state: dict[str, Any]) -> None:
        if int(state.get("block", block)) != block:
            raise ValueError(f"policy state belongs to block {state.get('block')}, not {block}")
        lanes = state.get("lanes")
        if not isinstance(lanes, list):
            raise TypeError("policy state lanes must be a list")
        imported: list[LaneState] = []
        for lane in lanes:
            imported.append(
                LaneState(
                    evict_idx=[int(value) for value in lane["evict_idx"]],
                    current_step=int(lane["current_step"]),
                    total_steps=int(lane["total_steps"]),
                )
            )
        self._states[block] = imported
