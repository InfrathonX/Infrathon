#!/usr/bin/env python3
"""Task-compatible extraction of ThunderAgent's program scheduler.

This module is derived from the MIT-licensed implementation in the sibling
``ThunderAgent`` package, principally:

* ``ThunderAgent/program/state.py``
* ``ThunderAgent/backend/state.py``
* ``ThunderAgent/scheduler/router.py``

The original project supports multiple inference backends and owns its HTTP
proxy.  The Harbor task has one fixed backend and a stricter byte-transparency
contract, so only the program lifecycle, token-capacity accounting, and
pause/restore policy live here.  HTTP forwarding remains in ``gateway.py``.

Copyright (c) 2026 Hao Kang.  Licensed under the MIT License; see
``THIRD_PARTY_LICENSE.md`` in this directory.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ProgramStatus(str, Enum):
    """The phase of one agentic program."""

    REASONING = "reasoning"
    ACTING = "acting"


class ProgramLifecycle(str, Enum):
    """Whether a program is admitted to the tracked KV working set."""

    ACTIVE = "active"
    PAUSED = "paused"
    TERMINATED = "terminated"


@dataclass
class Program:
    program_id: str
    status: ProgramStatus = ProgramStatus.ACTING
    lifecycle: ProgramLifecycle = ProgramLifecycle.ACTIVE
    total_tokens: int = 0
    context_chars: int = 0
    step_count: int = 0
    acting_since: float | None = None
    marked_for_pause: bool = False
    request_pending: bool = False
    inflight_requests: int = 0
    release_requested: bool = False
    # This clock starts only when an agent-facing request is actually blocked
    # on admission.  It intentionally does not reuse ``PausedInfo.paused_at``:
    # an idle ACTING program can be evicted long before its next request arrives.
    request_wait_started_monotonic: float | None = None
    # True only while this request is paused by the finite-cohort service-lag
    # adapter.  Keeping this as explicit bounded state makes a real run able to
    # distinguish fairness admission from ordinary KV-capacity admission.
    service_lag_blocked: bool = False
    waiting_event: asyncio.Event | None = field(default=None, repr=False)
    request_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)


@dataclass(frozen=True)
class PausedInfo:
    program_id: str
    total_tokens: int
    paused_at: float
    step_count: int


class ProgramReleased(RuntimeError):
    """A lifecycle release raced a request that was waiting for admission."""


class ProgramAdmissionCapacityTimeout(RuntimeError):
    """A timed-out request still could not be admitted without overflow."""


@dataclass(frozen=True)
class HybridPageAccounting:
    """Translate one hybrid-model program's token demand into KV pages.

    vLLM exposes ``kv_cache_size_tokens`` for model-level reporting, but that
    scalar is not additive across short hybrid-model sequences.  Each live
    sequence also needs a fixed number of resident recurrent-state pages and
    its full attention allocation is rounded up independently.  vLLM's Mamba
    ``align`` manager uses one resident state page per Mamba group.  A request
    can briefly hold a second page per group while copying the previous state
    into the next state.  A brand-new uncached request initially allocates only
    the resident page, but one OpenAI request normally spans multiple internal
    vLLM scheduler iterations and can reach the two-page peak before its
    response completes.  Agent ``step_count`` therefore cannot safely predict
    that peak.  Schema v3 reserves the request-lifetime peak for every active
    REASONING/request-pending program and only the resident pages for ACTING.

    This optional adapter keeps that model/runtime-specific translation
    outside the generic ThunderAgent scheduling policy.  Its page totals are
    conservative admission reservations, not claims about an instantaneous
    allocator snapshot.
    """

    raw_gpu_blocks: int
    reserved_null_blocks: int
    attention_page_tokens: int
    mamba_group_count: int
    mamba_resident_pages_per_group: int
    mamba_peak_pages_per_group: int
    reported_logical_capacity_tokens: int
    accounting_mode: str = "qwen36-vllm024-hybrid-state-aware-pages"
    accounting_schema_version: int = 3

    def __post_init__(self) -> None:
        positive = {
            "raw_gpu_blocks": self.raw_gpu_blocks,
            "attention_page_tokens": self.attention_page_tokens,
            "mamba_group_count": self.mamba_group_count,
            "mamba_resident_pages_per_group": (
                self.mamba_resident_pages_per_group
            ),
            "mamba_peak_pages_per_group": self.mamba_peak_pages_per_group,
            "reported_logical_capacity_tokens": (
                self.reported_logical_capacity_tokens
            ),
            "accounting_schema_version": self.accounting_schema_version,
        }
        for name, value in positive.items():
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        if (
            not isinstance(self.reserved_null_blocks, int)
            or isinstance(self.reserved_null_blocks, bool)
            or not 0 <= self.reserved_null_blocks < self.raw_gpu_blocks
        ):
            raise ValueError(
                "reserved_null_blocks must be a non-negative integer below "
                "raw_gpu_blocks"
            )
        if not self.accounting_mode:
            raise ValueError("accounting_mode must not be empty")
        if self.mamba_resident_pages_per_group > self.mamba_peak_pages_per_group:
            raise ValueError(
                "mamba_resident_pages_per_group must not exceed "
                "mamba_peak_pages_per_group"
            )

    @property
    def capacity_pages(self) -> int:
        return self.raw_gpu_blocks - self.reserved_null_blocks

    @property
    def fixed_mamba_pages_per_program(self) -> int:
        return self.mamba_group_count * self.mamba_resident_pages_per_group

    @property
    def transient_mamba_pages_per_program(self) -> int:
        return self.mamba_group_count * (
            self.mamba_peak_pages_per_group
            - self.mamba_resident_pages_per_group
        )

    @property
    def peak_mamba_pages_per_reasoning_program(self) -> int:
        return (
            self.fixed_mamba_pages_per_program
            + self.transient_mamba_pages_per_program
        )

    def attention_pages(
        self,
        total_tokens: int,
        *,
        decode_headroom_tokens: int,
        token_weight: float = 1.0,
    ) -> int:
        if decode_headroom_tokens < 0:
            raise ValueError("decode_headroom_tokens must not be negative")
        if not math.isfinite(token_weight) or not 0.0 <= token_weight <= 1.0:
            raise ValueError("token_weight must be finite and in [0, 1]")
        demand = max(0, total_tokens) * token_weight + decode_headroom_tokens
        return math.ceil(demand / self.attention_page_tokens)

    def program_pages(
        self,
        total_tokens: int,
        *,
        decode_headroom_tokens: int,
        reserve_transient_mamba: bool,
        token_weight: float = 1.0,
    ) -> int:
        if not isinstance(reserve_transient_mamba, bool):
            raise ValueError("reserve_transient_mamba must be a boolean")
        return (
            self.fixed_mamba_pages_per_program
            + (
                self.transient_mamba_pages_per_program
                if reserve_transient_mamba
                else 0
            )
            + self.attention_pages(
                total_tokens,
                decode_headroom_tokens=decode_headroom_tokens,
                token_weight=token_weight,
            )
        )

    def contract(self) -> dict[str, Any]:
        return {
            "accounting_schema_version": self.accounting_schema_version,
            "accounting_mode": self.accounting_mode,
            "decision_unit": "kv_pages",
            "raw_gpu_blocks": self.raw_gpu_blocks,
            "reserved_null_blocks": self.reserved_null_blocks,
            "capacity_pages": self.capacity_pages,
            "attention_page_tokens": self.attention_page_tokens,
            "mamba_group_count": self.mamba_group_count,
            "mamba_resident_pages_per_group": (
                self.mamba_resident_pages_per_group
            ),
            "mamba_peak_pages_per_group": self.mamba_peak_pages_per_group,
            "fixed_mamba_pages_per_program": (
                self.fixed_mamba_pages_per_program
            ),
            "transient_mamba_pages_per_program": (
                self.transient_mamba_pages_per_program
            ),
            "peak_mamba_pages_per_reasoning_program": (
                self.peak_mamba_pages_per_reasoning_program
            ),
            "mamba_initial_uncached_pages_per_program": (
                self.fixed_mamba_pages_per_program
            ),
            "mamba_reservation_scope": "request_lifetime_peak",
            "mamba_reservation_phase": "reasoning_or_request_pending",
            "mamba_reservation_uses_agent_step_count": False,
            "decision_pages_are_reservations": True,
            "acting_attention_decay_scope": "restore_only",
            "reported_logical_capacity_tokens": (
                self.reported_logical_capacity_tokens
            ),
            "reported_logical_capacity_is_informational": True,
        }


@dataclass(frozen=True)
class PageComponents:
    """One self-consistent state-aware hybrid-page reservation snapshot."""

    total_pages: int
    attention_pages: int
    resident_mamba_pages: int
    reserved_transient_mamba_pages: int
    reasoning_peak_reserved_programs: int


class ThunderScheduler:
    """Single-backend ThunderAgent scheduler for the frozen Harbor runtime.

    The KV accounting, pressure selection, and shortest-context tie-breaks
    follow the upstream implementation and paper:

    * account for the full context of REASONING programs;
    * account for ACTING programs with configurable exponential decay when
      considering restoration;
    * pause ACTING before REASONING, shortest context first;
    * restore the first twelve requests by lowest request ordinal, then shortest
      context, before deeper shortest-context requests and idle ACTING state;
    * bound an existing workflow's request-ordinal lead through request twelve,
      then restore upstream-style total-token SJF for deeper requests;
    * reserve a small decode buffer for every active program.

    Harbor's finite-cohort service-lag and bounded-age rules are explicit
    adapters, not claims about upstream ThunderAgent.  Also unlike upstream,
    marked REASONING programs are subtracted from the projected post-response
    working set.  Upstream records ``future_paused_tokens`` but its pause loop
    does not consume that value, which can mark every reasoning program during
    one scheduler tick.
    """

    def __init__(
        self,
        *,
        capacity_tokens: int,
        scheduler_interval_sec: float = 5.0,
        acting_decay_base: float = 2.0,
        buffer_per_program: int = 100,
        hybrid_page_accounting: HybridPageAccounting | None = None,
        high_watermark: float = 1.0,
        low_watermark: float = 1.0,
        force_resume_timeout_sec: float = 1800.0,
        fair_wait_threshold_sec: float = 220.0,
        aged_resume_quota_per_tick: int = 4,
        fair_service_lag_steps: int = 1,
        fair_service_until_step: int = 12,
    ) -> None:
        if capacity_tokens <= 0:
            raise ValueError("capacity_tokens must be positive")
        if scheduler_interval_sec <= 0:
            raise ValueError("scheduler_interval_sec must be positive")
        if acting_decay_base <= 1:
            raise ValueError("acting_decay_base must be greater than one")
        if buffer_per_program < 0:
            raise ValueError("buffer_per_program must not be negative")
        if not 0 < low_watermark <= high_watermark <= 1:
            raise ValueError("watermarks must satisfy 0 < low <= high <= 1")
        if force_resume_timeout_sec <= 0:
            raise ValueError("force_resume_timeout_sec must be positive")
        if (
            not math.isfinite(fair_wait_threshold_sec)
            or fair_wait_threshold_sec <= 0
        ):
            raise ValueError("fair_wait_threshold_sec must be finite and positive")
        if (
            not isinstance(aged_resume_quota_per_tick, int)
            or isinstance(aged_resume_quota_per_tick, bool)
            or aged_resume_quota_per_tick <= 0
        ):
            raise ValueError("aged_resume_quota_per_tick must be a positive integer")
        if (
            not isinstance(fair_service_lag_steps, int)
            or isinstance(fair_service_lag_steps, bool)
            or fair_service_lag_steps < 0
        ):
            raise ValueError("fair_service_lag_steps must be a non-negative integer")
        if (
            not isinstance(fair_service_until_step, int)
            or isinstance(fair_service_until_step, bool)
            or fair_service_until_step <= 0
        ):
            raise ValueError("fair_service_until_step must be a positive integer")

        self.capacity_tokens = capacity_tokens
        self.scheduler_interval_sec = scheduler_interval_sec
        self.acting_decay_base = acting_decay_base
        self.buffer_per_program = buffer_per_program
        self.hybrid_page_accounting = hybrid_page_accounting
        self.high_watermark = high_watermark
        self.low_watermark = low_watermark
        self.force_resume_timeout_sec = force_resume_timeout_sec
        self.fair_wait_threshold_sec = fair_wait_threshold_sec
        self.aged_resume_quota_per_tick = aged_resume_quota_per_tick
        self.fair_service_lag_steps = fair_service_lag_steps
        self.fair_service_until_step = fair_service_until_step

        self.programs: dict[str, Program] = {}
        self.programs_peak = 0
        self.waiting: OrderedDict[str, PausedInfo] = OrderedDict()
        self.char_to_token_ratio = 5.0
        self._ratio_initialized = False
        self._lock = asyncio.Lock()
        self._scheduler_task: asyncio.Task[None] | None = None
        self._stopping = False
        self._last_scheduler_error: str | None = None
        self._created_monotonic = time.monotonic()
        self._first_pause_elapsed_sec: float | None = None
        self._first_pause_reason: str | None = None
        self._first_pause_tracked_full: float | None = None
        self._first_pause_tracked_pages_full: int | None = None
        self._first_pause_attention_pages_full: int | None = None
        self._first_pause_fixed_mamba_pages: int | None = None
        self._first_pause_reserved_transient_mamba_pages: int | None = None
        self._first_pause_reasoning_peak_reserved_programs: int | None = None
        self._first_pause_active: int | None = None
        self._first_resume_elapsed_sec: float | None = None
        self._active_peak = 0
        self._paused_peak = 0
        self._marked_for_pause_peak = 0
        self._tracked_tokens_full_peak = 0.0
        self._tracked_tokens_with_decay_peak = 0.0
        self._tracked_pages_full_peak = 0
        self._tracked_pages_with_decay_peak = 0
        self._tracked_attention_pages_full_peak = 0
        self._tracked_attention_pages_with_decay_peak = 0
        self._tracked_fixed_mamba_pages_peak = 0
        self._tracked_reserved_transient_mamba_pages_peak = 0
        self._reasoning_peak_reserved_programs_peak = 0
        self._pause_wait_count = 0
        self._pause_wait_total_sec = 0.0
        self._pause_wait_max_sec = 0.0
        self._scheduler_ticks = 0
        self._pending_request_waiters_peak = 0
        self._protected_pending_request_waiters_peak = 0
        self._deep_pending_request_waiters_peak = 0
        self._aged_request_waiters_peak = 0
        self._oldest_pending_request_wait_sec_peak = 0.0
        self._service_lag_blocked_peak = 0
        self._protected_pause_wait_count = 0
        self._protected_pause_wait_total_sec = 0.0
        self._protected_pause_wait_max_sec = 0.0
        self._deep_pause_wait_count = 0
        self._deep_pause_wait_total_sec = 0.0
        self._deep_pause_wait_max_sec = 0.0
        # One oldest aged request may reserve exact full-capacity decision
        # units.  The owner remains PAUSED until real ACTIVE capacity has been
        # released; in-flight REASONING is never treated as already free.
        self._aged_reservation_program_id: str | None = None
        self._aged_reservation_started_monotonic: float | None = None
        self._aged_reservation_oversize_program_ids: set[str] = set()
        self._aged_reservation_active_peak = 0
        self._aged_reservation_required_capacity_peak = 0
        self._aged_reservation_gap_capacity_peak = 0
        self._aged_reservation_age_sec_peak = 0.0
        self._aged_rescue_post_commit_overflow_peak = 0
        self._counters = {
            "pause_admission": 0,
            "pause_admission_new": 0,
            "pause_admission_reentry": 0,
            "pause_admission_aged_reservation": 0,
            "pause_admission_service_lag": 0,
            "protected_request_offers": 0,
            "deep_request_offers": 0,
            "deep_service_lag_bypass_decisions": 0,
            "pause_acting": 0,
            "pause_acting_pressure": 0,
            "pause_acting_aged_reservation": 0,
            "mark_reasoning": 0,
            "resume_pending": 0,
            "resume_new": 0,
            "resume_acting": 0,
            "resume_aged_pending": 0,
            "resume_aged_new": 0,
            "resume_aged_reservation_pending": 0,
            "resume_aged_reservation_new": 0,
            "resume_least_service_request": 0,
            "resume_protected_request": 0,
            "resume_deep_request": 0,
            "least_service_priority_ticks": 0,
            "deep_sjf_priority_ticks": 0,
            "mixed_request_tier_ticks": 0,
            "aged_priority_ticks": 0,
            "aged_capacity_skip_count": 0,
            "aged_no_fit_ticks": 0,
            "aged_quota_exhausted_ticks": 0,
            "aged_reservation_started": 0,
            "aged_reservation_satisfied": 0,
            "aged_reservation_cancelled": 0,
            "aged_reservation_oversize": 0,
            "aged_reservation_oversize_owner": 0,
            "aged_reservation_partial_ticks": 0,
            "aged_reservation_no_safe_victim_ticks": 0,
            "aged_reservation_blocked_reentry": 0,
            "aged_reservation_victims_paused": 0,
            "aged_reservation_capacity_freed": 0,
            "aged_reservation_post_commit_capacity_violations": 0,
            "force_resume": 0,
            "blocked_capacity_timeout": 0,
            "release": 0,
            "release_deferred": 0,
            "scheduler_errors": 0,
        }

    @property
    def high_capacity(self) -> float:
        return self.decision_capacity * self.high_watermark

    @property
    def low_capacity(self) -> float:
        return self.decision_capacity * self.low_watermark

    @property
    def decision_capacity(self) -> int:
        if self.hybrid_page_accounting is not None:
            return self.hybrid_page_accounting.capacity_pages
        return self.capacity_tokens

    async def start(self) -> None:
        if self._scheduler_task is not None:
            return
        self._stopping = False
        self._scheduler_task = asyncio.create_task(
            self._scheduler_loop(), name="thunderagent-program-scheduler"
        )

    async def stop(self) -> None:
        self._stopping = True
        if self._scheduler_task is not None:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
            self._scheduler_task = None
        async with self._lock:
            for program in self.programs.values():
                if program.waiting_event is not None:
                    program.waiting_event.set()

    async def get_or_create_program(self, program_id: str) -> Program:
        async with self._lock:
            program = self.programs.get(program_id)
            if program is None:
                program = Program(program_id=program_id)
                self.programs[program_id] = program
                self.programs_peak = max(self.programs_peak, len(self.programs))
                self._observe_peaks_locked()
            return program

    async def begin_request(self, program: Program, context_chars: int) -> None:
        """Transition a program to REASONING and wait if it is paused."""

        wait_event: asyncio.Event | None = None
        async with self._lock:
            current = self.programs.get(program.program_id)
            if current is not program:
                raise ProgramReleased(program.program_id)
            is_new = program.step_count == 0
            program.step_count += 1
            offer_counter = (
                "protected_request_offers"
                if self._is_protected_request(program)
                else "deep_request_offers"
            )
            self._counters[offer_counter] += 1
            program.context_chars = max(0, context_chars)
            program.total_tokens = max(
                1, int(program.context_chars / max(self.char_to_token_ratio, 1e-9))
            )
            program.status = ProgramStatus.REASONING
            program.acting_since = None
            program.request_pending = True
            # A previously evicted ACTING program is guaranteed to block on
            # its existing admission event.  Start the request clock before
            # observing the offered working set so telemetry never confuses
            # its old eviction time with this new agent-facing wait.
            if program.lifecycle == ProgramLifecycle.PAUSED:
                self._start_request_wait_locked(program)
            # Capture the offered working set before admission can move this
            # program out of the active set.
            self._observe_peaks_locked()

            if program.lifecycle == ProgramLifecycle.PAUSED:
                if program.waiting_event is None:
                    program.waiting_event = asyncio.Event()
                self._refresh_paused_info_locked(program)
                wait_event = program.waiting_event
            else:
                # A brand-new program never bypasses an existing queue.  V5
                # preserves V4's lag bound through ``fair_service_until_step``
                # and lets deeper requests bypass only that lag decision.
                # Capacity remains independently checked on every
                # ACTING->REASONING transition because that transition adds
                # request-lifetime Mamba peak headroom and its context can grow.
                queue_blocks_new = is_new and bool(self.waiting)
                capacity_overflow = self._full_used_locked() > self.high_capacity
                reservation_blocks_reentry = (
                    not is_new
                    and self._aged_reservation_blocks_program_locked(program)
                )
                service_lag_would_block_reentry = (
                    not is_new
                    and self._service_lag_would_block_reentry_locked(program)
                )
                deep_service_lag_bypass = (
                    service_lag_would_block_reentry
                    and not self._is_protected_request(program)
                )
                if deep_service_lag_bypass:
                    self._counters[
                        "deep_service_lag_bypass_decisions"
                    ] += 1
                service_lag_blocks_reentry = (
                    service_lag_would_block_reentry
                    and not deep_service_lag_bypass
                )
                if (
                    queue_blocks_new
                    or capacity_overflow
                    or reservation_blocks_reentry
                    or service_lag_blocks_reentry
                ):
                    # ``program`` is already present in the active set, so
                    # _full_used_locked() already includes its context, decode
                    # buffer, resident Mamba pages, and (now that it is
                    # REASONING) transient peak reservation.  Adding
                    # _required(program) again would double-count it.
                    program.service_lag_blocked = service_lag_blocks_reentry
                    pause_reason = (
                        "admission_service_lag"
                        if service_lag_blocks_reentry
                        else "admission_capacity"
                    )
                    self._pause_locked(program, reason=pause_reason)
                    self._counters["pause_admission"] += 1
                    split_counter = (
                        "pause_admission_new"
                        if is_new
                        else "pause_admission_reentry"
                    )
                    self._counters[split_counter] += 1
                    if reservation_blocks_reentry:
                        self._counters[
                            "pause_admission_aged_reservation"
                        ] += 1
                        self._counters[
                            "aged_reservation_blocked_reentry"
                        ] += 1
                    if service_lag_blocks_reentry:
                        self._counters["pause_admission_service_lag"] += 1
                    wait_event = program.waiting_event

            if wait_event is None:
                program.inflight_requests += 1

        if wait_event is None:
            return
        wait_started_monotonic = time.monotonic()
        try:
            await asyncio.wait_for(
                wait_event.wait(), timeout=self.force_resume_timeout_sec
            )
        except asyncio.TimeoutError:
            async with self._lock:
                if self._handle_admission_timeout_locked(program):
                    self._record_pause_wait_locked(
                        program,
                        time.monotonic() - wait_started_monotonic
                    )
                    raise ProgramAdmissionCapacityTimeout(program.program_id)
        except asyncio.CancelledError:
            # Retain the partial agent-visible admission delay without
            # swallowing or postponing request cancellation.
            async with self._lock:
                self._record_pause_wait_locked(
                    program,
                    time.monotonic() - wait_started_monotonic
                )
            raise
        async with self._lock:
            self._record_pause_wait_locked(
                program,
                time.monotonic() - wait_started_monotonic
            )
            if self.programs.get(program.program_id) is not program:
                raise ProgramReleased(program.program_id)
            if program.lifecycle != ProgramLifecycle.ACTIVE:
                raise RuntimeError(
                    f"program {program.program_id} woke without active admission"
                )
            program.inflight_requests += 1

    async def cancel_waiting_request(self, program: Program) -> None:
        """Undo pending state when a client cancels before backend admission."""

        async with self._lock:
            if self.programs.get(program.program_id) is not program:
                return
            was_reservation_owner = (
                self._aged_reservation_program_id == program.program_id
            )
            self._clear_aged_reservation_locked(program, outcome="cancelled")
            self._aged_reservation_oversize_program_ids.discard(
                program.program_id
            )
            program.request_pending = False
            program.request_wait_started_monotonic = None
            program.service_lag_blocked = False
            if program.step_count <= 1 and program.lifecycle == ProgramLifecycle.PAUSED:
                self.waiting.pop(program.program_id, None)
                program.lifecycle = ProgramLifecycle.TERMINATED
                if program.waiting_event is not None:
                    program.waiting_event.set()
                self.programs.pop(program.program_id, None)
                if was_reservation_owner:
                    self._elect_oldest_aged_reservation_locked(time.monotonic())
                self._observe_peaks_locked()
                return
            program.status = ProgramStatus.ACTING
            program.acting_since = time.time()
            self._refresh_paused_info_locked(program)
            if was_reservation_owner:
                self._elect_oldest_aged_reservation_locked(time.monotonic())
            self._observe_peaks_locked()

    async def finish_request(
        self,
        program: Program,
        *,
        total_tokens: int | None,
        prompt_tokens: int | None,
    ) -> None:
        """Record backend usage and transition a completed request to ACTING."""

        async with self._lock:
            if self.programs.get(program.program_id) is not program:
                return
            if (
                isinstance(prompt_tokens, int)
                and prompt_tokens > 0
                and program.context_chars > 0
            ):
                sample = program.context_chars / prompt_tokens
                if math.isfinite(sample) and sample > 0:
                    if not self._ratio_initialized:
                        self.char_to_token_ratio = sample
                        self._ratio_initialized = True
                    else:
                        self.char_to_token_ratio = (
                            0.2 * sample + 0.8 * self.char_to_token_ratio
                        )
            if isinstance(total_tokens, int) and total_tokens > 0:
                program.total_tokens = total_tokens
            self._observe_peaks_locked()
            if program.inflight_requests <= 0:
                raise RuntimeError(
                    f"program {program.program_id} completed without an in-flight request"
                )
            program.inflight_requests -= 1
            program.request_pending = False
            if program.release_requested and program.inflight_requests == 0:
                self._terminate_locked(program)
                return
            program.status = ProgramStatus.ACTING
            program.acting_since = time.time()
            if program.marked_for_pause:
                program.marked_for_pause = False
                self._pause_locked(program, reason="marked_reasoning_pressure")
            else:
                self._observe_peaks_locked()

    async def update_streaming_tokens(self, program: Program, delta_tokens: int) -> None:
        """Track bounded in-flight decode growth between final usage reports."""

        if delta_tokens <= 0:
            return
        async with self._lock:
            if self.programs.get(program.program_id) is not program:
                return
            if (
                program.inflight_requests > 0
                and program.status == ProgramStatus.REASONING
            ):
                program.total_tokens += delta_tokens
                self._observe_peaks_locked()

    async def release_program(self, program_id: str) -> tuple[bool, bool]:
        """Apply ThunderAgent's explicit program termination signal.

        Return ``(found, released_now)``.  A lifecycle request that races an
        active backend call is acknowledged but deferred until that call's
        finally path.  This preserves capacity accounting while the request is
        still physically consuming vLLM resources.
        """

        async with self._lock:
            program = self.programs.get(program_id)
            if program is None:
                return False, False
            if program.inflight_requests > 0:
                program.release_requested = True
                self._counters["release_deferred"] += 1
                return True, False
            self._terminate_locked(program)
            return True, True

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            now = time.time()
            now_monotonic = time.monotonic()
            self._observe_peaks_locked(now, now_monotonic=now_monotonic)
            (
                pending_request_waiters,
                aged_request_waiters,
                oldest_pending_request_wait_sec,
            ) = self._request_wait_telemetry_locked(now_monotonic)
            (
                protected_pending_request_waiters,
                deep_pending_request_waiters,
            ) = self._request_tier_waiter_telemetry_locked()
            (
                aged_reservation_active,
                aged_reservation_required_capacity,
                aged_reservation_gap_capacity,
                aged_reservation_age_sec,
            ) = self._aged_reservation_telemetry_locked(now_monotonic)
            active = [
                program
                for program in self.programs.values()
                if program.lifecycle == ProgramLifecycle.ACTIVE
            ]
            tracked_tokens_full = self._full_token_used_locked()
            tracked_tokens_with_decay = self._decayed_token_used_locked(now)
            page_components_full = self._page_components_locked(now, decay=False)
            page_components_with_decay = self._page_components_locked(
                now, decay=True
            )
            if self.hybrid_page_accounting is None:
                accounting_contract: dict[str, Any] = {
                    "accounting_schema_version": 1,
                    "accounting_mode": "thunderagent-token-linear",
                    "decision_unit": "tokens",
                }
            else:
                accounting_contract = self.hybrid_page_accounting.contract()
            accounting_contract.update(
                {
                    "decision_capacity": self.decision_capacity,
                    "decode_headroom_tokens": self.buffer_per_program,
                }
            )
            tracked_pages_full = (
                page_components_full.total_pages
                if page_components_full is not None
                else None
            )
            tracked_pages_with_decay = (
                page_components_with_decay.total_pages
                if page_components_with_decay is not None
                else None
            )
            page_overflow = (
                max(0, tracked_pages_full - self.decision_capacity)
                if tracked_pages_full is not None
                else None
            )
            first_pause_page_overflow = (
                max(
                    0,
                    self._first_pause_tracked_pages_full
                    - self.decision_capacity,
                )
                if self._first_pause_tracked_pages_full is not None
                else None
            )
            return {
                "algorithm": "thunderagent-program-aware-single-backend",
                "scheduler_policy_schema_version": 4,
                "queue_policy": (
                    "thunderagent-tiered-sjf-service-lag-age-reservation-swap-v5"
                ),
                "fair_service_lag_steps": self.fair_service_lag_steps,
                "fair_service_until_step": self.fair_service_until_step,
                "deep_request_first_step": self.fair_service_until_step + 1,
                "protected_request_policy": (
                    f"v4-exact-through-step-{self.fair_service_until_step}"
                ),
                "deep_request_policy": (
                    "total_tokens-sjf-with-service-lag-bypass-from-step-"
                    f"{self.fair_service_until_step + 1}"
                ),
                "service_lag_scope": (
                    "protected-reentry-vs-feasible-blocked-request-ordinal"
                ),
                "service_lag_comparison": (
                    "protected_reentry_ordinal>least_waiting_ordinal+lag"
                ),
                "normal_restore_order": (
                    "aged-then-protected-request-ordinal-then-deep-total-tokens-"
                    "then-idle-acting"
                ),
                "deep_restore_order": (
                    "total_tokens-then-request-wait-age-then-program-id"
                ),
                "deep_reentry_bypasses_service_lag": True,
                "fair_wait_threshold_sec": self.fair_wait_threshold_sec,
                "aged_resume_quota_per_tick": (
                    self.aged_resume_quota_per_tick
                ),
                "request_age_scope": "blocked_gateway_request_only",
                "aged_priority_scope": (
                    "single-oldest-reservation-then-aged-quota-then-tiered-sjf"
                ),
                "aged_reservation_precedes_service_lag_order": True,
                "aged_reservation_owner_limit": 1,
                "aged_reservation_capacity_basis": (
                    "full_phase_aware_decision_units"
                ),
                "aged_reservation_victim_policy": (
                    "active-idle-acting-shortest-context-first"
                ),
                "aged_reservation_blocks_younger_restore": True,
                "aged_reservation_blocks_conflicting_reentry": True,
                "aged_reservation_inflight_reasoning_policy": "never_pause",
                "capacity_tokens": self.capacity_tokens,
                **accounting_contract,
                "high_watermark": self.high_watermark,
                "low_watermark": self.low_watermark,
                "scheduler_interval_sec": self.scheduler_interval_sec,
                "acting_decay_base": self.acting_decay_base,
                "buffer_per_program": self.buffer_per_program,
                "programs": len(self.programs),
                "programs_peak": self.programs_peak,
                "scheduler_task_alive": (
                    self._scheduler_task is not None
                    and not self._scheduler_task.done()
                ),
                "last_scheduler_error": self._last_scheduler_error,
                "active": len(active),
                "reasoning": sum(
                    program.status == ProgramStatus.REASONING for program in active
                ),
                "acting": sum(
                    program.status == ProgramStatus.ACTING for program in active
                ),
                "paused": len(self.waiting),
                "service_lag_blocked": sum(
                    program.lifecycle == ProgramLifecycle.PAUSED
                    and program.request_pending
                    and program.service_lag_blocked
                    for program in self.programs.values()
                ),
                "pending_request_waiters": pending_request_waiters,
                "protected_pending_request_waiters": (
                    protected_pending_request_waiters
                ),
                "deep_pending_request_waiters": deep_pending_request_waiters,
                "aged_request_waiters": aged_request_waiters,
                "oldest_pending_request_wait_sec": round(
                    oldest_pending_request_wait_sec, 6
                ),
                "aged_reservation_active": aged_reservation_active,
                "aged_reservation_required_capacity": (
                    aged_reservation_required_capacity
                ),
                "aged_reservation_gap_capacity": (
                    aged_reservation_gap_capacity
                ),
                "aged_reservation_age_sec": round(
                    aged_reservation_age_sec, 6
                ),
                "marked_for_pause": sum(
                    program.marked_for_pause for program in active
                ),
                "terminating": sum(
                    program.release_requested for program in self.programs.values()
                ),
                "tracked_tokens_full": round(tracked_tokens_full, 3),
                "tracked_tokens_with_decay": round(
                    tracked_tokens_with_decay, 3
                ),
                "tracked_pages_full": tracked_pages_full,
                "tracked_pages_with_decay": tracked_pages_with_decay,
                "tracked_attention_pages_full": (
                    page_components_full.attention_pages
                    if page_components_full is not None
                    else None
                ),
                "tracked_attention_pages_with_decay": (
                    page_components_with_decay.attention_pages
                    if page_components_with_decay is not None
                    else None
                ),
                "tracked_fixed_mamba_pages": (
                    page_components_full.resident_mamba_pages
                    if page_components_full is not None
                    else None
                ),
                "tracked_reserved_transient_mamba_pages": (
                    page_components_full.reserved_transient_mamba_pages
                    if page_components_full is not None
                    else None
                ),
                "reasoning_peak_reserved_programs": (
                    page_components_full.reasoning_peak_reserved_programs
                    if page_components_full is not None
                    else None
                ),
                "capacity_overflow_pages": page_overflow,
                "first_pause_elapsed_sec": self._rounded_optional(
                    self._first_pause_elapsed_sec
                ),
                "first_pause_reason": self._first_pause_reason,
                "first_pause_tracked_full": self._rounded_optional(
                    self._first_pause_tracked_full, digits=3
                ),
                "first_pause_tracked_pages_full": (
                    self._first_pause_tracked_pages_full
                ),
                "first_pause_overflow_pages": first_pause_page_overflow,
                "first_pause_attention_pages_full": (
                    self._first_pause_attention_pages_full
                ),
                "first_pause_fixed_mamba_pages": (
                    self._first_pause_fixed_mamba_pages
                ),
                "first_pause_reserved_transient_mamba_pages": (
                    self._first_pause_reserved_transient_mamba_pages
                ),
                "first_pause_reasoning_peak_reserved_programs": (
                    self._first_pause_reasoning_peak_reserved_programs
                ),
                "first_pause_active": self._first_pause_active,
                "first_resume_elapsed_sec": self._rounded_optional(
                    self._first_resume_elapsed_sec
                ),
                "active_peak": self._active_peak,
                "paused_peak": self._paused_peak,
                "service_lag_blocked_peak": self._service_lag_blocked_peak,
                "marked_for_pause_peak": self._marked_for_pause_peak,
                "pending_request_waiters_peak": (
                    self._pending_request_waiters_peak
                ),
                "protected_pending_request_waiters_peak": (
                    self._protected_pending_request_waiters_peak
                ),
                "deep_pending_request_waiters_peak": (
                    self._deep_pending_request_waiters_peak
                ),
                "aged_request_waiters_peak": self._aged_request_waiters_peak,
                "oldest_pending_request_wait_sec_peak": round(
                    self._oldest_pending_request_wait_sec_peak, 6
                ),
                "aged_reservation_active_peak": (
                    self._aged_reservation_active_peak
                ),
                "aged_reservation_required_capacity_peak": (
                    self._aged_reservation_required_capacity_peak
                ),
                "aged_reservation_gap_capacity_peak": (
                    self._aged_reservation_gap_capacity_peak
                ),
                "aged_reservation_age_sec_peak": round(
                    self._aged_reservation_age_sec_peak, 6
                ),
                "aged_rescue_post_commit_overflow_peak": (
                    self._aged_rescue_post_commit_overflow_peak
                ),
                "tracked_tokens_full_peak": round(
                    self._tracked_tokens_full_peak, 3
                ),
                "tracked_tokens_with_decay_peak": round(
                    self._tracked_tokens_with_decay_peak, 3
                ),
                "tracked_pages_full_peak": self._tracked_pages_full_peak,
                "tracked_pages_with_decay_peak": (
                    self._tracked_pages_with_decay_peak
                ),
                "tracked_attention_pages_full_peak": (
                    self._tracked_attention_pages_full_peak
                ),
                "tracked_attention_pages_with_decay_peak": (
                    self._tracked_attention_pages_with_decay_peak
                ),
                "tracked_fixed_mamba_pages_peak": (
                    self._tracked_fixed_mamba_pages_peak
                ),
                "tracked_reserved_transient_mamba_pages_peak": (
                    self._tracked_reserved_transient_mamba_pages_peak
                ),
                "reasoning_peak_reserved_programs_peak": (
                    self._reasoning_peak_reserved_programs_peak
                ),
                "pause_wait_count": self._pause_wait_count,
                "pause_wait_total_sec": round(self._pause_wait_total_sec, 6),
                "pause_wait_max_sec": round(self._pause_wait_max_sec, 6),
                "protected_pause_wait_count": (
                    self._protected_pause_wait_count
                ),
                "protected_pause_wait_total_sec": round(
                    self._protected_pause_wait_total_sec, 6
                ),
                "protected_pause_wait_max_sec": round(
                    self._protected_pause_wait_max_sec, 6
                ),
                "deep_pause_wait_count": self._deep_pause_wait_count,
                "deep_pause_wait_total_sec": round(
                    self._deep_pause_wait_total_sec, 6
                ),
                "deep_pause_wait_max_sec": round(
                    self._deep_pause_wait_max_sec, 6
                ),
                "scheduler_ticks": self._scheduler_ticks,
                "char_to_token_ratio": round(self.char_to_token_ratio, 6),
                "counters": dict(self._counters),
            }

    async def schedule_once(self) -> None:
        """Run one deterministic scheduler tick (also used by unit tests)."""

        async with self._lock:
            self._scheduler_ticks += 1
            now = time.time()
            now_monotonic = time.monotonic()
            self._restore_waiting_locked(now, now_monotonic)
            self._pause_until_safe_locked()
            self._observe_peaks_locked(now, now_monotonic=now_monotonic)

    @staticmethod
    def _reserves_transient_mamba(program: Program) -> bool:
        """Fail closed if phase and request-pending bookkeeping ever disagree."""

        return (
            program.status == ProgramStatus.REASONING
            or program.request_pending
        )

    def _is_protected_request(self, program: Program) -> bool:
        """Whether this request retains the finite-cohort v4 fairness tier."""

        return program.step_count <= self.fair_service_until_step

    def _required(self, program: Program) -> int:
        if self.hybrid_page_accounting is not None:
            return self.hybrid_page_accounting.program_pages(
                program.total_tokens,
                decode_headroom_tokens=self.buffer_per_program,
                reserve_transient_mamba=self._reserves_transient_mamba(program),
            )
        return max(0, program.total_tokens) + self.buffer_per_program

    def _minimum_required_capacity(self) -> int:
        """Smallest cost of a real request in the current decision unit."""

        if self.hybrid_page_accounting is not None:
            return self.hybrid_page_accounting.program_pages(
                1,
                decode_headroom_tokens=self.buffer_per_program,
                # This is a lower-bound fast gate.  The selection loop below
                # still commits each candidate's exact phase-aware cost.
                reserve_transient_mamba=False,
            )
        # ``begin_request`` always records at least one context token.  Keeping
        # this as buffer+1 also preserves the generic scheduler's historical
        # strict ``available > buffer`` restoration condition.
        return self.buffer_per_program + 1

    def _handle_admission_timeout_locked(self, program: Program) -> bool:
        """Handle one wait timeout without weakening the capacity invariant.

        Return ``True`` only when the request remains paused and cannot be
        restored at its full request-lifetime cost.  A concurrent normal resume
        is deliberately a no-op here and must not be counted as a force resume.
        """

        if self.programs.get(program.program_id) is not program:
            return False
        if program.lifecycle != ProgramLifecycle.PAUSED:
            return False
        projected = self._full_used_locked() + self._required(program)
        if projected <= self.high_capacity:
            self._resume_locked(program)
            self._counters["force_resume"] += 1
            return False
        self._counters["blocked_capacity_timeout"] += 1
        return True

    def _terminate_locked(self, program: Program) -> None:
        was_reservation_owner = (
            self._aged_reservation_program_id == program.program_id
        )
        self._clear_aged_reservation_locked(program, outcome="cancelled")
        self._aged_reservation_oversize_program_ids.discard(program.program_id)
        self.programs.pop(program.program_id, None)
        self.waiting.pop(program.program_id, None)
        program.lifecycle = ProgramLifecycle.TERMINATED
        program.request_pending = False
        program.request_wait_started_monotonic = None
        program.service_lag_blocked = False
        program.release_requested = False
        if program.waiting_event is not None:
            program.waiting_event.set()
        self._counters["release"] += 1
        if was_reservation_owner:
            self._elect_oldest_aged_reservation_locked(time.monotonic())
        self._observe_peaks_locked()

    def _active_programs_locked(self) -> list[Program]:
        return [
            program
            for program in self.programs.values()
            if program.lifecycle == ProgramLifecycle.ACTIVE
        ]

    def _full_used_locked(self) -> float:
        return float(
            sum(
                self._required(program)
                for program in self._active_programs_locked()
            )
        )

    def _decayed_used_locked(self, now: float) -> float:
        if self.hybrid_page_accounting is None:
            return self._decayed_token_used_locked(now)
        used = 0.0
        for program in self._active_programs_locked():
            token_weight = 1.0
            if (
                program.status == ProgramStatus.ACTING
                and not program.request_pending
                and program.acting_since is not None
            ):
                elapsed = max(0.0, now - program.acting_since)
                token_weight = self.acting_decay_base ** (-elapsed)
            used += self.hybrid_page_accounting.program_pages(
                program.total_tokens,
                decode_headroom_tokens=self.buffer_per_program,
                reserve_transient_mamba=self._reserves_transient_mamba(program),
                token_weight=token_weight,
            )
        return used

    def _full_token_used_locked(self) -> float:
        active = self._active_programs_locked()
        return float(
            sum(max(0, program.total_tokens) for program in active)
            + len(active) * self.buffer_per_program
        )

    def _decayed_token_used_locked(self, now: float) -> float:
        used = 0.0
        active = self._active_programs_locked()
        for program in active:
            tokens = float(max(0, program.total_tokens))
            if (
                program.status == ProgramStatus.ACTING
                and program.acting_since is not None
            ):
                elapsed = max(0.0, now - program.acting_since)
                tokens *= self.acting_decay_base ** (-elapsed)
            used += tokens
        return used + len(active) * self.buffer_per_program

    def _page_components_locked(
        self, now: float, *, decay: bool
    ) -> PageComponents | None:
        accounting = self.hybrid_page_accounting
        if accounting is None:
            return None
        active = self._active_programs_locked()
        attention_pages = 0
        reasoning_peak_reserved_programs = 0
        for program in active:
            reserve_transient = self._reserves_transient_mamba(program)
            reasoning_peak_reserved_programs += int(reserve_transient)
            token_weight = 1.0
            if (
                decay
                and program.status == ProgramStatus.ACTING
                and not program.request_pending
                and program.acting_since is not None
            ):
                elapsed = max(0.0, now - program.acting_since)
                token_weight = self.acting_decay_base ** (-elapsed)
            attention_pages += accounting.attention_pages(
                program.total_tokens,
                decode_headroom_tokens=self.buffer_per_program,
                token_weight=token_weight,
            )
        fixed_pages = len(active) * accounting.fixed_mamba_pages_per_program
        transient_pages = (
            reasoning_peak_reserved_programs
            * accounting.transient_mamba_pages_per_program
        )
        return PageComponents(
            total_pages=fixed_pages + transient_pages + attention_pages,
            attention_pages=attention_pages,
            resident_mamba_pages=fixed_pages,
            reserved_transient_mamba_pages=transient_pages,
            reasoning_peak_reserved_programs=reasoning_peak_reserved_programs,
        )

    def _projected_full_used_locked(self) -> float:
        used = self._full_used_locked()
        for program in self._active_programs_locked():
            if program.marked_for_pause:
                used -= self._required(program)
        return max(0.0, used)

    def _refresh_paused_info_locked(self, program: Program) -> None:
        if program.lifecycle != ProgramLifecycle.PAUSED:
            return
        previous = self.waiting.get(program.program_id)
        self.waiting[program.program_id] = PausedInfo(
            program_id=program.program_id,
            total_tokens=program.total_tokens,
            paused_at=previous.paused_at if previous else time.time(),
            step_count=program.step_count,
        )

    @staticmethod
    def _start_request_wait_locked(program: Program) -> None:
        """Start the fairness clock only for a genuinely blocked request."""

        if not program.request_pending:
            raise RuntimeError("cannot age a program without a pending request")
        if program.request_wait_started_monotonic is None:
            program.request_wait_started_monotonic = time.monotonic()

    def _pause_locked(self, program: Program, *, reason: str = "unspecified") -> None:
        if program.lifecycle == ProgramLifecycle.PAUSED:
            self._refresh_paused_info_locked(program)
            return
        self._record_first_pause_locked(reason)
        program.lifecycle = ProgramLifecycle.PAUSED
        program.waiting_event = asyncio.Event()
        if program.request_pending:
            self._start_request_wait_locked(program)
        self._refresh_paused_info_locked(program)
        self._observe_peaks_locked()

    def _resume_locked(self, program: Program) -> None:
        if program.lifecycle != ProgramLifecycle.PAUSED:
            return
        self._clear_aged_reservation_locked(program, outcome="satisfied")
        self._aged_reservation_oversize_program_ids.discard(program.program_id)
        now_monotonic = time.monotonic()
        if self._first_resume_elapsed_sec is None:
            self._first_resume_elapsed_sec = max(
                0.0, now_monotonic - self._created_monotonic
            )
        self.waiting.pop(program.program_id, None)
        program.lifecycle = ProgramLifecycle.ACTIVE
        program.request_wait_started_monotonic = None
        program.service_lag_blocked = False
        event = program.waiting_event
        program.waiting_event = None
        if event is not None:
            event.set()
        self._observe_peaks_locked()

    @staticmethod
    def _rounded_optional(
        value: float | None, *, digits: int = 6
    ) -> float | None:
        return None if value is None else round(value, digits)

    def _record_first_pause_locked(self, reason: str) -> None:
        if self._first_pause_elapsed_sec is not None:
            return
        self._first_pause_elapsed_sec = max(
            0.0, time.monotonic() - self._created_monotonic
        )
        self._first_pause_reason = reason
        # Preserve this legacy field's token-denominated meaning even when the
        # admission decision itself uses hybrid KV pages.  Page-denominated
        # first-pause evidence is exposed separately below.
        self._first_pause_tracked_full = self._full_token_used_locked()
        self._first_pause_active = len(self._active_programs_locked())
        page_components = self._page_components_locked(time.time(), decay=False)
        if page_components is not None:
            self._first_pause_tracked_pages_full = page_components.total_pages
            self._first_pause_attention_pages_full = (
                page_components.attention_pages
            )
            self._first_pause_fixed_mamba_pages = (
                page_components.resident_mamba_pages
            )
            self._first_pause_reserved_transient_mamba_pages = (
                page_components.reserved_transient_mamba_pages
            )
            self._first_pause_reasoning_peak_reserved_programs = (
                page_components.reasoning_peak_reserved_programs
            )

    def _record_pause_wait_locked(
        self, program: Program, duration: float
    ) -> None:
        """Record one completed or cancelled agent-facing admission wait."""

        duration = max(0.0, duration)
        self._pause_wait_count += 1
        self._pause_wait_total_sec += duration
        self._pause_wait_max_sec = max(self._pause_wait_max_sec, duration)
        if self._is_protected_request(program):
            self._protected_pause_wait_count += 1
            self._protected_pause_wait_total_sec += duration
            self._protected_pause_wait_max_sec = max(
                self._protected_pause_wait_max_sec, duration
            )
        else:
            self._deep_pause_wait_count += 1
            self._deep_pause_wait_total_sec += duration
            self._deep_pause_wait_max_sec = max(
                self._deep_pause_wait_max_sec, duration
            )

    def _request_tier_waiter_telemetry_locked(self) -> tuple[int, int]:
        """Return current protected and deep pending-request waiters."""

        protected = 0
        deep = 0
        for program_id in self.waiting:
            program = self.programs.get(program_id)
            if (
                program is None
                or program.lifecycle != ProgramLifecycle.PAUSED
                or not program.request_pending
            ):
                continue
            if self._is_protected_request(program):
                protected += 1
            else:
                deep += 1
        return protected, deep

    def _record_request_resume_tier_locked(self, program: Program) -> None:
        """Partition scheduler-selected request resumes by request tier."""

        if not program.request_pending:
            raise RuntimeError("cannot record a request resume without a request")
        counter = (
            "resume_protected_request"
            if self._is_protected_request(program)
            else "resume_deep_request"
        )
        self._counters[counter] += 1

    def _request_wait_telemetry_locked(
        self, now_monotonic: float
    ) -> tuple[int, int, float]:
        """Return current blocked-request count, aged count, and oldest age."""

        pending_count = 0
        aged_count = 0
        oldest_age = 0.0
        for program_id in self.waiting:
            program = self.programs.get(program_id)
            if (
                program is None
                or program.lifecycle != ProgramLifecycle.PAUSED
                or not program.request_pending
            ):
                continue
            started = program.request_wait_started_monotonic
            if started is None:
                raise RuntimeError(
                    f"pending waiter {program.program_id} has no request wait clock"
                )
            age = max(0.0, now_monotonic - started)
            pending_count += 1
            aged_count += int(age >= self.fair_wait_threshold_sec)
            oldest_age = max(oldest_age, age)
        return pending_count, aged_count, oldest_age

    def _aged_reservation_owner_locked(self) -> Program | None:
        program_id = self._aged_reservation_program_id
        if program_id is None:
            return None
        program = self.programs.get(program_id)
        if (
            program is None
            or program.lifecycle != ProgramLifecycle.PAUSED
            or not program.request_pending
            or program.request_wait_started_monotonic is None
        ):
            raise RuntimeError("aged reservation owner lifecycle is inconsistent")
        return program

    def _start_aged_reservation_locked(
        self, program: Program, now_monotonic: float
    ) -> None:
        if self._aged_reservation_program_id is not None:
            if self._aged_reservation_program_id != program.program_id:
                raise RuntimeError("multiple aged reservation owners")
            return
        if (
            program.lifecycle != ProgramLifecycle.PAUSED
            or not program.request_pending
            or program.request_wait_started_monotonic is None
        ):
            raise RuntimeError("cannot reserve for a non-waiting request")
        if self._required(program) > self.high_capacity:
            raise RuntimeError("cannot reserve an oversized request")
        self._aged_reservation_program_id = program.program_id
        self._aged_reservation_started_monotonic = now_monotonic
        self._counters["aged_reservation_started"] += 1

    def _clear_aged_reservation_locked(
        self, program: Program, *, outcome: str
    ) -> None:
        if self._aged_reservation_program_id != program.program_id:
            return
        if outcome not in {"satisfied", "cancelled", "oversize"}:
            raise ValueError("invalid aged reservation outcome")
        self._aged_reservation_program_id = None
        self._aged_reservation_started_monotonic = None
        if outcome == "satisfied":
            self._counters["aged_reservation_satisfied"] += 1
        elif outcome == "cancelled":
            self._counters["aged_reservation_cancelled"] += 1
        else:
            self._counters["aged_reservation_oversize_owner"] += 1

    def _aged_reservation_telemetry_locked(
        self, now_monotonic: float
    ) -> tuple[int, int, int, float]:
        owner = self._aged_reservation_owner_locked()
        if owner is None:
            return 0, 0, 0, 0.0
        required = int(self._required(owner))
        gap = int(
            math.ceil(
                max(
                    0.0,
                    self._full_used_locked()
                    + required
                    - self.high_capacity,
                )
            )
        )
        started = owner.request_wait_started_monotonic
        if started is None:
            raise RuntimeError("aged reservation owner has no request clock")
        age = max(0.0, now_monotonic - started)
        return 1, required, gap, age

    def _observe_aged_reservation_peaks_locked(
        self, now_monotonic: float
    ) -> None:
        active, required, gap, age = self._aged_reservation_telemetry_locked(
            now_monotonic
        )
        self._aged_reservation_active_peak = max(
            self._aged_reservation_active_peak, active
        )
        self._aged_reservation_required_capacity_peak = max(
            self._aged_reservation_required_capacity_peak, required
        )
        self._aged_reservation_gap_capacity_peak = max(
            self._aged_reservation_gap_capacity_peak, gap
        )
        self._aged_reservation_age_sec_peak = max(
            self._aged_reservation_age_sec_peak, age
        )

    def _aged_reservation_blocks_program_locked(self, program: Program) -> bool:
        """Keep a pending oldest request's exact full-capacity reservation."""

        owner = self._aged_reservation_owner_locked()
        if owner is None or owner is program:
            return False
        return (
            self._full_used_locked() + self._required(owner)
            > self.high_capacity
        )

    def _service_lag_would_block_reentry_locked(self, program: Program) -> bool:
        """Return the exact v4 lag decision before the v5 deep-tier bypass.

        ``begin_request`` increments ``program.step_count`` before this check,
        so both sides are compared as request ordinals.  An individually
        oversized waiter is ignored: a request that can never fit must not stop
        every healthy workflow.  The separate bounded-age reservation still
        owns the long-wait tail and takes precedence during restoration.
        Callers apply the result only through ``fair_service_until_step``;
        deeper requests retain this comparison as telemetry but bypass it.
        """

        least_waiting_ordinal: int | None = None
        for program_id in self.waiting:
            waiter = self.programs.get(program_id)
            if (
                waiter is None
                or waiter.lifecycle != ProgramLifecycle.PAUSED
                or not waiter.request_pending
                or self._required(waiter) > self.high_capacity
            ):
                continue
            least_waiting_ordinal = (
                waiter.step_count
                if least_waiting_ordinal is None
                else min(least_waiting_ordinal, waiter.step_count)
            )
        return (
            least_waiting_ordinal is not None
            and program.step_count
            > least_waiting_ordinal + self.fair_service_lag_steps
        )

    def _safe_aged_reservation_victims_locked(self) -> list[Program]:
        """Return deterministic, upstream-compatible idle ACTING victims."""

        return sorted(
            (
                program
                for program in self._active_programs_locked()
                if program.status == ProgramStatus.ACTING
                and not program.request_pending
                and program.inflight_requests == 0
                and not program.release_requested
                and not program.marked_for_pause
            ),
            key=lambda program: (program.total_tokens, program.program_id),
        )

    def _oldest_aged_waiters_locked(
        self, now_monotonic: float
    ) -> list[Program]:
        aged: list[Program] = []
        for program_id in self.waiting:
            program = self.programs.get(program_id)
            if (
                program is None
                or program.lifecycle != ProgramLifecycle.PAUSED
                or not program.request_pending
            ):
                continue
            started = program.request_wait_started_monotonic
            if started is None:
                raise RuntimeError(
                    f"pending waiter {program.program_id} has no request clock"
                )
            if now_monotonic - started >= self.fair_wait_threshold_sec:
                aged.append(program)
        return sorted(
            aged,
            key=lambda program: (
                program.request_wait_started_monotonic,
                program.step_count == 1,
                program.total_tokens,
                program.program_id,
            ),
        )

    def _elect_oldest_aged_reservation_locked(
        self,
        now_monotonic: float,
        aged: list[Program] | None = None,
    ) -> Program | None:
        """Elect one feasible oldest owner; skip impossible requests once."""

        owner = self._aged_reservation_owner_locked()
        if owner is not None:
            return owner
        candidates = (
            self._oldest_aged_waiters_locked(now_monotonic)
            if aged is None
            else aged
        )
        for candidate in candidates:
            required = self._required(candidate)
            if required > self.high_capacity:
                if (
                    candidate.program_id
                    not in self._aged_reservation_oversize_program_ids
                ):
                    self._aged_reservation_oversize_program_ids.add(
                        candidate.program_id
                    )
                    self._counters["aged_reservation_oversize"] += 1
                continue
            self._aged_reservation_oversize_program_ids.discard(
                candidate.program_id
            )
            self._start_aged_reservation_locked(candidate, now_monotonic)
            return candidate
        return None

    def _service_aged_reservation_locked(
        self,
        aged: list[Program],
        now_monotonic: float,
    ) -> tuple[Program | None, set[str], bool]:
        """Reserve real full capacity for one oldest aged request.

        Return ``(resumed_owner, victim_ids, blocked)``.  ``blocked`` means a
        valid owner remains waiting, so younger/unaged restoration must stop.
        Only idle ACTING working sets are paused.  Active REASONING requests
        continue untouched and may become safe victims only after completion.
        """

        owner = self._aged_reservation_owner_locked()
        if owner is not None and self._required(owner) > self.high_capacity:
            self._clear_aged_reservation_locked(owner, outcome="oversize")
            if owner.program_id not in self._aged_reservation_oversize_program_ids:
                self._aged_reservation_oversize_program_ids.add(owner.program_id)
                self._counters["aged_reservation_oversize"] += 1
            owner = None

        if owner is None:
            owner = self._elect_oldest_aged_reservation_locked(
                now_monotonic, aged
            )

        if owner is None:
            return None, set(), False

        self._observe_aged_reservation_peaks_locked(now_monotonic)
        required = self._required(owner)
        projected_full = self._full_used_locked()
        victim_plan: list[tuple[Program, int]] = []
        for victim in self._safe_aged_reservation_victims_locked():
            if projected_full + required <= self.high_capacity:
                break
            victim_required = int(self._required(victim))
            victim_plan.append((victim, victim_required))
            projected_full -= victim_required

        # The plan and commit are one non-awaiting critical section.  A partial
        # plan is intentional when in-flight REASONING still owns the remaining
        # pages: those real requests finish normally while this reservation
        # prevents younger work from consuming the pages already accumulated.
        victim_ids: set[str] = set()
        freed = 0
        for victim, victim_required in victim_plan:
            victim_ids.add(victim.program_id)
            self._pause_locked(victim, reason="aged_reservation_swap")
            self._counters["pause_acting"] += 1
            self._counters["pause_acting_aged_reservation"] += 1
            self._counters["aged_reservation_victims_paused"] += 1
            self._counters["aged_reservation_capacity_freed"] += (
                victim_required
            )
            freed += victim_required

        if self._full_used_locked() + required <= self.high_capacity:
            was_new = owner.step_count == 1
            self._record_request_resume_tier_locked(owner)
            self._resume_locked(owner)
            if was_new:
                self._counters["resume_new"] += 1
                self._counters["resume_aged_new"] += 1
                self._counters["resume_aged_reservation_new"] += 1
            else:
                self._counters["resume_pending"] += 1
                self._counters["resume_aged_pending"] += 1
                self._counters["resume_aged_reservation_pending"] += 1
            overflow = int(
                math.ceil(
                    max(0.0, self._full_used_locked() - self.high_capacity)
                )
            )
            self._aged_rescue_post_commit_overflow_peak = max(
                self._aged_rescue_post_commit_overflow_peak, overflow
            )
            if overflow:
                self._counters[
                    "aged_reservation_post_commit_capacity_violations"
                ] += 1
            return owner, victim_ids, False

        self._counters["aged_no_fit_ticks"] += 1
        self._counters["aged_capacity_skip_count"] += 1
        if freed:
            self._counters["aged_reservation_partial_ticks"] += 1
        self._counters["aged_reservation_no_safe_victim_ticks"] += 1
        self._observe_aged_reservation_peaks_locked(now_monotonic)
        return None, victim_ids, True

    def _observe_peaks_locked(
        self,
        now: float | None = None,
        *,
        now_monotonic: float | None = None,
    ) -> None:
        """Update persistent, bounded diagnostics while the scheduler lock is held."""

        if now is None:
            now = time.time()
        if now_monotonic is None:
            now_monotonic = time.monotonic()
        active_count = 0
        marked_for_pause = 0
        tracked_full = 0.0
        tracked_with_decay = 0.0
        for program in self.programs.values():
            if program.lifecycle != ProgramLifecycle.ACTIVE:
                continue
            active_count += 1
            marked_for_pause += int(program.marked_for_pause)
            tokens = float(max(0, program.total_tokens))
            tracked_full += tokens + self.buffer_per_program
            if (
                program.status == ProgramStatus.ACTING
                and program.acting_since is not None
            ):
                elapsed = max(0.0, now - program.acting_since)
                tokens *= self.acting_decay_base ** (-elapsed)
            tracked_with_decay += tokens + self.buffer_per_program
        self._active_peak = max(self._active_peak, active_count)
        self._paused_peak = max(self._paused_peak, len(self.waiting))
        service_lag_blocked = sum(
            program.lifecycle == ProgramLifecycle.PAUSED
            and program.request_pending
            and program.service_lag_blocked
            for program in self.programs.values()
        )
        self._service_lag_blocked_peak = max(
            self._service_lag_blocked_peak, service_lag_blocked
        )
        self._marked_for_pause_peak = max(
            self._marked_for_pause_peak, marked_for_pause
        )
        (
            pending_request_waiters,
            aged_request_waiters,
            oldest_pending_request_wait_sec,
        ) = self._request_wait_telemetry_locked(now_monotonic)
        (
            protected_pending_request_waiters,
            deep_pending_request_waiters,
        ) = self._request_tier_waiter_telemetry_locked()
        self._pending_request_waiters_peak = max(
            self._pending_request_waiters_peak, pending_request_waiters
        )
        self._protected_pending_request_waiters_peak = max(
            self._protected_pending_request_waiters_peak,
            protected_pending_request_waiters,
        )
        self._deep_pending_request_waiters_peak = max(
            self._deep_pending_request_waiters_peak,
            deep_pending_request_waiters,
        )
        self._aged_request_waiters_peak = max(
            self._aged_request_waiters_peak, aged_request_waiters
        )
        self._oldest_pending_request_wait_sec_peak = max(
            self._oldest_pending_request_wait_sec_peak,
            oldest_pending_request_wait_sec,
        )
        self._observe_aged_reservation_peaks_locked(now_monotonic)
        self._tracked_tokens_full_peak = max(
            self._tracked_tokens_full_peak, tracked_full
        )
        self._tracked_tokens_with_decay_peak = max(
            self._tracked_tokens_with_decay_peak, tracked_with_decay
        )
        page_components_full = self._page_components_locked(now, decay=False)
        page_components_with_decay = self._page_components_locked(
            now, decay=True
        )
        if (
            page_components_full is not None
            and page_components_with_decay is not None
        ):
            self._tracked_pages_full_peak = max(
                self._tracked_pages_full_peak, page_components_full.total_pages
            )
            self._tracked_pages_with_decay_peak = max(
                self._tracked_pages_with_decay_peak,
                page_components_with_decay.total_pages,
            )
            self._tracked_attention_pages_full_peak = max(
                self._tracked_attention_pages_full_peak,
                page_components_full.attention_pages,
            )
            self._tracked_attention_pages_with_decay_peak = max(
                self._tracked_attention_pages_with_decay_peak,
                page_components_with_decay.attention_pages,
            )
            self._tracked_fixed_mamba_pages_peak = max(
                self._tracked_fixed_mamba_pages_peak,
                page_components_full.resident_mamba_pages,
            )
            self._tracked_reserved_transient_mamba_pages_peak = max(
                self._tracked_reserved_transient_mamba_pages_peak,
                page_components_full.reserved_transient_mamba_pages,
            )
            self._reasoning_peak_reserved_programs_peak = max(
                self._reasoning_peak_reserved_programs_peak,
                page_components_full.reasoning_peak_reserved_programs,
            )

    async def _scheduler_loop(self) -> None:
        while not self._stopping:
            try:
                await asyncio.sleep(self.scheduler_interval_sec)
                await self.schedule_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # One malformed lifecycle state must not silently kill the
                # background scheduler for the rest of a multi-hour run.
                async with self._lock:
                    self._counters["scheduler_errors"] += 1
                    self._last_scheduler_error = type(exc).__name__

    def _restore_waiting_locked(
        self, now: float, now_monotonic: float
    ) -> None:
        if not self.waiting:
            return

        def paused_programs() -> list[Program]:
            return [
                self.programs[program_id]
                for program_id in self.waiting
                if program_id in self.programs
                and self.programs[program_id].lifecycle
                == ProgramLifecycle.PAUSED
            ]

        paused = paused_programs()
        aged = self._oldest_aged_waiters_locked(now_monotonic)
        # Capture the queue before any selected event is woken.  Otherwise a
        # single overdue waiter could be resumed in this tick without ever
        # appearing in the persistent age peaks sampled after restoration.
        (
            pending_request_waiters,
            aged_request_waiters,
            oldest_pending_request_wait_sec,
        ) = self._request_wait_telemetry_locked(now_monotonic)
        self._pending_request_waiters_peak = max(
            self._pending_request_waiters_peak, pending_request_waiters
        )
        self._aged_request_waiters_peak = max(
            self._aged_request_waiters_peak, aged_request_waiters
        )
        self._oldest_pending_request_wait_sec_peak = max(
            self._oldest_pending_request_wait_sec_peak,
            oldest_pending_request_wait_sec,
        )
        if aged:
            self._counters["aged_priority_ticks"] += 1

        resumed_owner: Program | None = None
        reservation_victim_ids: set[str] = set()
        if aged or self._aged_reservation_program_id is not None:
            (
                resumed_owner,
                reservation_victim_ids,
                reservation_blocked,
            ) = self._service_aged_reservation_locked(aged, now_monotonic)
            if reservation_blocked:
                # The reservation owns every capacity unit it has accumulated.
                # Younger aged, unaged, NEW, and idle-ACTING programs must not
                # consume it while the oldest feasible request still waits.
                return

        # Reservation may have resumed its owner and paused idle ACTING
        # victims.  Rebuild the queue before applying the v5 protected/deep
        # tiers to remaining capacity.
        paused = paused_programs()
        aged = self._oldest_aged_waiters_locked(now_monotonic)
        reservation_resumes = int(resumed_owner is not None)

        used = self._decayed_used_locked(now)
        if used >= self.low_capacity:
            if aged:
                self._counters["aged_no_fit_ticks"] += 1
                self._counters["aged_capacity_skip_count"] += min(
                    len(aged), self.aged_resume_quota_per_tick
                )
            return
        available = self.high_capacity - used
        if self.hybrid_page_accounting is None:
            # Preserve the generic token scheduler's exact historical gate,
            # including fractional available capacity after ACTING decay.
            if available <= self.buffer_per_program:
                if aged:
                    self._counters["aged_no_fit_ticks"] += 1
                    self._counters["aged_capacity_skip_count"] += min(
                        len(aged), self.aged_resume_quota_per_tick
                    )
                return
        elif available < self._minimum_required_capacity():
            if aged:
                self._counters["aged_no_fit_ticks"] += 1
                self._counters["aged_capacity_skip_count"] += min(
                    len(aged), self.aged_resume_quota_per_tick
                )
            return

        aged_ids = {program.program_id for program in aged}
        protected_requesting = sorted(
            (
                program
                for program in paused
                if program.request_pending
                and program.program_id not in aged_ids
                and self._is_protected_request(program)
            ),
            key=lambda program: (
                program.step_count,
                program.total_tokens,
                program.request_wait_started_monotonic,
                program.program_id,
            ),
        )
        deep_requesting = sorted(
            (
                program
                for program in paused
                if program.request_pending
                and program.program_id not in aged_ids
                and not self._is_protected_request(program)
            ),
            key=lambda program: (
                program.total_tokens,
                program.request_wait_started_monotonic,
                program.program_id,
            ),
        )
        acting = sorted(
            (program for program in paused if not program.request_pending),
            key=lambda program: program.total_tokens,
        )

        selected: list[Program] = []
        selected_ids: set[str] = set()
        aged_selected_ids: set[str] = set()
        least_service_selected_ids: set[str] = set()
        request_ordinals = {
            program.step_count for program in protected_requesting
        }
        least_request_ordinal = (
            min(request_ordinals) if request_ordinals else None
        )
        service_tier_competition = len(request_ordinals) > 1
        if service_tier_competition:
            self._counters["least_service_priority_ticks"] += 1
        if deep_requesting:
            self._counters["deep_sjf_priority_ticks"] += 1
        if protected_requesting and deep_requesting:
            self._counters["mixed_request_tier_ticks"] += 1
        committed = 0.0
        for program in aged:
            if (
                reservation_resumes + len(aged_selected_ids)
                >= self.aged_resume_quota_per_tick
            ):
                break
            required = self._required(program)
            if committed + required <= available:
                selected.append(program)
                selected_ids.add(program.program_id)
                aged_selected_ids.add(program.program_id)
                committed += required
            else:
                self._counters["aged_capacity_skip_count"] += 1
        if aged and not aged_selected_ids:
            self._counters["aged_no_fit_ticks"] += 1
        if (
            reservation_resumes + len(aged_selected_ids)
            >= self.aged_resume_quota_per_tick
            and len(aged) > len(aged_selected_ids)
        ):
            self._counters["aged_quota_exhausted_ticks"] += 1

        for program in [
            *protected_requesting,
            *deep_requesting,
            *acting,
        ]:
            if program.program_id in selected_ids:
                continue
            if program.program_id in reservation_victim_ids:
                # Avoid pausing and immediately restoring the same idle
                # working set in one tick.  It may rejoin normal SJF next tick.
                continue
            required = self._required(program)
            if committed + required <= available:
                selected.append(program)
                selected_ids.add(program.program_id)
                if (
                    service_tier_competition
                    and program.request_pending
                    and program.step_count == least_request_ordinal
                ):
                    least_service_selected_ids.add(program.program_id)
                committed += required

        # Preserve the upstream largest-first placement phase.  With one fixed
        # backend this does not change the selected set, but wakes the larger
        # admitted contexts first after shortest-first set selection.
        for program in sorted(selected, key=lambda item: item.total_tokens, reverse=True):
            was_pending = program.request_pending
            was_new = program.step_count == 1
            was_aged = program.program_id in aged_selected_ids
            was_least_service = (
                program.program_id in least_service_selected_ids
            )
            if was_pending:
                self._record_request_resume_tier_locked(program)
            self._resume_locked(program)
            if was_least_service:
                self._counters["resume_least_service_request"] += 1
            if was_new:
                self._counters["resume_new"] += 1
                if was_aged:
                    self._counters["resume_aged_new"] += 1
            elif was_pending:
                self._counters["resume_pending"] += 1
                if was_aged:
                    self._counters["resume_aged_pending"] += 1
            else:
                self._counters["resume_acting"] += 1

    def _pause_until_safe_locked(self) -> None:
        while self._projected_full_used_locked() > self.high_capacity:
            acting = sorted(
                (
                    program
                    for program in self._active_programs_locked()
                    if program.status == ProgramStatus.ACTING
                    and not program.marked_for_pause
                ),
                key=lambda program: program.total_tokens,
            )
            if acting:
                self._pause_locked(acting[0], reason="acting_pressure")
                self._counters["pause_acting"] += 1
                self._counters["pause_acting_pressure"] += 1
                continue

            reasoning = sorted(
                (
                    program
                    for program in self._active_programs_locked()
                    if program.status == ProgramStatus.REASONING
                    and not program.marked_for_pause
                ),
                key=lambda program: program.total_tokens,
            )
            if not reasoning:
                break
            reasoning[0].marked_for_pause = True
            self._counters["mark_reasoning"] += 1
            self._observe_peaks_locked()
