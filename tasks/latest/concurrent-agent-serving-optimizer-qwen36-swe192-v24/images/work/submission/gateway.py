#!/usr/bin/env python3
"""Transparent, program-aware admission gateway for a hybrid-cache model.

The transport path is deliberately boring: every admitted data-plane request
is opened once, and response bytes are copied in arrival order.  The control
plane only decides *when* a chat request may be opened.  It uses bounded state
derived from the request/response pair and the optional program release hint.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import aiohttp.http_writer as _aiohttp_http_writer
from aiohttp import (
    ClientError,
    ClientSession,
    ClientTimeout,
    DummyCookieJar,
    TCPConnector,
    web,
)
from multidict import CIMultiDict


BACKEND_URL = os.environ["BACKEND_URL"].rstrip("/")
PORT = int(os.environ.get("PORT", "9000"))
MAX_REQUEST_BYTES = 64 * 1024 * 1024

# Fixed public runtime geometry.  The 64-page guard covers estimation error,
# an output tail beyond the learned reservation, and allocator fragmentation.
ALLOCATABLE_PAGES = 1252
PAGE_GUARD = 64
WORKING_PAGE_BUDGET = ALLOCATABLE_PAGES - PAGE_GUARD
ATTENTION_TOKENS_PER_PAGE = 1568
RESIDENT_RECURRENT_PAGES = 3
PENDING_RECURRENT_PAGES = 6
MAX_BACKEND_CHATS = 192

# Bounded policy/state constants.
INITIAL_OUTPUT_RESERVE = ATTENTION_TOKENS_PER_PAGE
MAX_ADAPTIVE_OUTPUT_RESERVE = 64 * ATTENTION_TOKENS_PER_PAGE
# A stable owner cohort is accounted in resident-equivalent pages: a pending
# request's recurrent peak is three pages higher than the state it can leave
# resident.  The request ledger below always retains the full peak claim.
PINNED_ENVELOPE_BUDGET = 896
MAX_PINNED_PROGRAMS = 128
BORROWER_BASE_COUNT = 32
BORROWER_BASE_PAGE_BUDGET = 292
BORROWER_EXPANDED_COUNT = 64
BORROWER_EXPANDED_PAGE_BUDGET = 448
BORROWER_UNDERFILL_INFLIGHT = 96
# A finite first-wave circulation rule may use the already-accounted expanded
# borrower envelope even while owner work is ready.  Only an ordinal-one FIFO
# head can create this debt, and at most one modest request may bank hard-page
# runway at a time.
FIRST_SERVICE_AGE_SECONDS = 30.0
FIRST_SERVICE_RUNWAY_MAX_PAGES = 32
PIN_LEASE_COMPLETIONS = 2
PIN_LEASE_SECONDS = 120.0
# A full-cohort membership exchange is permitted only after a recurring
# request-scoped borrower has completed successfully.  It may retain that
# freshly materialized context only by releasing one boundary-safe owner that
# has been idle for a full tool-gap interval.
HANDOFF_MIN_TICKET_ORDINAL = 2
HANDOFF_ENTRY_MAX_ORDINAL = 3
HANDOFF_STALE_OWNER_SECONDS = 30.0
ROTATION_WINDOW_SECONDS = 60.0
ROTATION_MAX_PROGRAMS = 32
ROTATION_MAX_PAGES = 224
STARVATION_AGE_SECONDS = 45.0
# A recurring outsider still blocked after a full turnover window may exchange
# one boundary-safe completed owner, preferring a stale victim.  This changes
# membership one-for-one instead of widening the cold borrower plane.
LATE_OWNER_EXCHANGE_AGE_SECONDS = 60.0
STATE_TTL_SECONDS = 3600.0
MAX_PROGRAMS = 2048
MAX_USAGE_SAMPLES = 256
MAX_OBSERVER_BYTES = 256 * 1024

# RFC 9110/9112 hop-by-hop fields plus the common non-standard extension.
HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)


def _serialize_headers_octet_safe(
    status_line: str, headers: CIMultiDict[str]
) -> bytes:
    """Serialize aiohttp's surrogate-escaped raw header view losslessly.

    aiohttp's optional C serializer silently drops surrogate-escaped octets in
    field values.  Both its HTTP parser and this gateway use UTF-8 with
    surrogateescape for raw fields, so matching that codec on output preserves
    valid UTF-8 and arbitrary RFC obs-text bytes one-for-one.
    """

    lines = []
    for name, value in headers.items():
        if "\r" in name or "\n" in name or "\r" in value or "\n" in value:
            raise ValueError("newline in HTTP header")
        lines.append(f"{name}: {value}")
    block = status_line + "\r\n" + "\r\n".join(lines) + "\r\n\r\n"
    return block.encode("utf-8", "surrogateescape")


# StreamWriter resolves this module global at call time.  Use one codec for
# client-side upstream requests and server-side downstream responses.
_aiohttp_http_writer._serialize_headers = _serialize_headers_octet_safe


def _ceil_pages(tokens: int) -> int:
    if tokens <= 0:
        return 0
    return (tokens + ATTENTION_TOKENS_PER_PAGE - 1) // ATTENTION_TOKENS_PER_PAGE


def _ewma(old: float, sample: float, weight: float = 0.2) -> float:
    if old <= 0:
        return sample
    return old * (1.0 - weight) + sample * weight


class ProgramPhase(str, Enum):
    NEW = "NEW"
    REASONING = "REASONING"
    ACTING = "ACTING"
    PAUSED = "PAUSED"
    RELEASED = "RELEASED"


class TicketPhase(str, Enum):
    QUEUED = "QUEUED"
    ADMITTED = "ADMITTED"
    FINISHED = "FINISHED"
    CANCELLED = "CANCELLED"


class AdmissionLane(str, Enum):
    PINNED = "PINNED"
    BORROWER = "BORROWER"
    DRAIN = "DRAIN"


@dataclass(slots=True)
class RequestShape:
    message_bytes: int
    message_count: int
    prompt_tokens: int
    output_reserve: int
    declared_output_cap: int | None


@dataclass(slots=True)
class UsageRecord:
    prompt_tokens: int
    completion_tokens: int


@dataclass(slots=True)
class Program:
    key: str
    phase: ProgramPhase = ProgramPhase.NEW
    active: bool = False
    ephemeral: bool = False
    release_pending: bool = False
    pinned: bool = False
    pin_envelope_pages: int = 0
    pin_started_at: float = 0.0
    pin_completed_steps: int = 0
    pin_generation: int = 0
    ordinal: int = 0
    activation_steps: int = 0
    resident_tokens: int = 0
    last_prompt_tokens: int = 0
    last_message_bytes: int = 0
    completion_ewma: float = 0.0
    last_seen: float = field(default_factory=time.monotonic)
    last_admitted: float = 0.0
    last_completed: float = 0.0
    last_paused: float = 0.0
    queue_wait_ewma: float = 0.0
    service_ewma: float = 0.0
    ttft_ewma: float = 0.0
    queued: list[int] = field(default_factory=list)
    inflight: set[int] = field(default_factory=set)


@dataclass(slots=True)
class Ticket:
    ident: int
    program: Program
    ordinal: int
    enqueued_at: float
    shape: RequestShape
    gate: asyncio.Future[None]
    phase: TicketPhase = TicketPhase.QUEUED
    admitted_at: float = 0.0
    prior_active: bool = False
    prior_resident_tokens: int = 0
    prior_activation_steps: int = 0
    admission_lane: AdmissionLane | None = None
    granted_pin: bool = False

    @property
    def estimated_pages(self) -> int:
        attention = _ceil_pages(
            self.shape.prompt_tokens + self.shape.output_reserve
        )
        return PENDING_RECURRENT_PAGES + attention

    @property
    def reserved_pages(self) -> int:
        # A lone request cannot be assigned more than the physical ledger.
        # Saturating its claim at the hard ceiling makes it drain-and-run-alone
        # while preserving both one-to-one forwarding and R <= 1188.
        return min(WORKING_PAGE_BUDGET, self.estimated_pages)


class ResponseObserver:
    """Bounded, fail-open usage and timing observation of copied bytes."""

    __slots__ = (
        "is_stream",
        "first_at",
        "total_bytes",
        "_buffer",
        "_line_buffer",
        "_truncated",
        "usage",
    )

    def __init__(self, content_type: str | None) -> None:
        self.is_stream = bool(
            content_type and "text/event-stream" in content_type.lower()
        )
        self.first_at = 0.0
        self.total_bytes = 0
        self._buffer = bytearray()
        self._line_buffer = bytearray()
        self._truncated = False
        self.usage: UsageRecord | None = None

    @staticmethod
    def _usage_from_object(value: Any) -> UsageRecord | None:
        if not isinstance(value, dict):
            return None
        usage = value.get("usage")
        if not isinstance(usage, dict):
            return None
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        if isinstance(prompt, int) and isinstance(completion, int):
            if prompt >= 0 and completion >= 0:
                return UsageRecord(prompt, completion)
        return None

    def feed(self, chunk: bytes) -> None:
        try:
            if not self.first_at:
                self.first_at = time.monotonic()
            self.total_bytes += len(chunk)
            if self.is_stream:
                self._feed_sse(chunk)
            elif not self._truncated:
                room = MAX_OBSERVER_BYTES - len(self._buffer)
                if len(chunk) <= room:
                    self._buffer.extend(chunk)
                else:
                    self._buffer.extend(chunk[: max(0, room)])
                    self._truncated = True
        except Exception:
            # Monitoring must never influence the byte-transparent data path.
            self._truncated = True
            self._buffer.clear()
            self._line_buffer.clear()

    def _feed_sse(self, chunk: bytes) -> None:
        if len(self._line_buffer) + len(chunk) > MAX_OBSERVER_BYTES:
            # Drop only the observational copy of an implausibly large event.
            self._line_buffer.clear()
            self._truncated = True
            return
        self._line_buffer.extend(chunk)
        while True:
            newline = self._line_buffer.find(b"\n")
            if newline < 0:
                return
            line = bytes(self._line_buffer[:newline]).rstrip(b"\r")
            del self._line_buffer[: newline + 1]
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if not data or data == b"[DONE]":
                continue
            try:
                observed = self._usage_from_object(json.loads(data))
            except (UnicodeDecodeError, json.JSONDecodeError):
                observed = None
            if observed is not None:
                self.usage = observed

    def finish(self) -> UsageRecord | None:
        if self.usage is not None or self.is_stream or self._truncated:
            return self.usage
        try:
            self.usage = self._usage_from_object(json.loads(self._buffer))
        except Exception:
            pass
        return self.usage


class ProgramScheduler:
    """Atomic hybrid-page ledger with a partially rotating owner cohort."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._programs: dict[str, Program] = {}
        self._tickets: dict[int, Ticket] = {}
        self._next_ticket = 1
        self._next_ephemeral = 1
        self._completion_samples: deque[int] = deque(maxlen=MAX_USAGE_SAMPLES)
        self._rotation_timer: asyncio.TimerHandle | None = None
        self._rotation_timer_deadline = 0.0
        self._pin_generation = 0
        self._rotation_events: deque[tuple[float, int]] = deque()
        # Conservative high-water estimate of prompt tokens per message byte.
        self._tokens_per_byte = 0.36
        self._counters: dict[str, int] = {
            "offered": 0,
            "admitted": 0,
            "completed": 0,
            "cancelled_queued": 0,
            "cancelled_admitted": 0,
            "backend_errors": 0,
            "pauses": 0,
            "restorations": 0,
            "aged_admissions": 0,
            "pin_promotions": 0,
            "pin_demotions": 0,
            "pin_rotations": 0,
            "forced_rotations": 0,
            "handoff_attempts": 0,
            "handoff_completions": 0,
            "handoff_direct_promotions": 0,
            "handoff_no_compatible_victim": 0,
            "handoff_ineligible_target": 0,
            "late_owner_exchange_attempts": 0,
            "late_owner_exchange_completions": 0,
            "late_owner_exchange_direct_promotions": 0,
            "late_owner_exchange_no_victim": 0,
            "late_owner_exchange_handoff_deferrals": 0,
            "aged_borrower_admissions": 0,
            "aged_borrower_capacity_waits": 0,
            "aged_borrower_hard_waits": 0,
            "aged_borrower_owner_bypasses": 0,
            "first_service_admissions": 0,
            "first_service_expanded_admissions": 0,
            "first_service_lane_skips": 0,
            "first_service_large_skips": 0,
            "first_service_runway_waits": 0,
            "first_service_owner_bypasses": 0,
            "recurring_service_admissions": 0,
            "recurring_service_expanded_admissions": 0,
            "recurring_service_runway_waits": 0,
            "recurring_service_owner_bypasses": 0,
            "pin_envelope_expansions": 0,
            "pin_growth_demotions": 0,
            "rotation_budget_denials": 0,
            "borrower_admissions": 0,
            "borrower_yields": 0,
            "borrower_expanded_admissions": 0,
            "drain_admissions": 0,
            "output_overshoots": 0,
            "releases": 0,
        }

    def new_ephemeral_key(self) -> str:
        value = self._next_ephemeral
        self._next_ephemeral += 1
        return f"__anonymous_request_{value}"

    def _completion_p95(self) -> int:
        if not self._completion_samples:
            return 0
        ordered = sorted(self._completion_samples)
        index = max(0, math.ceil(0.95 * len(ordered)) - 1)
        return ordered[index]

    def _output_reserve(
        self, program: Program, declared_cap: int | None
    ) -> int:
        learned = max(
            INITIAL_OUTPUT_RESERVE,
            math.ceil(1.5 * self._completion_p95()),
            math.ceil(1.25 * program.completion_ewma),
        )
        learned = _ceil_pages(learned) * ATTENTION_TOKENS_PER_PAGE
        learned = min(MAX_ADAPTIVE_OUTPUT_RESERVE, learned)
        if declared_cap is not None:
            learned = min(learned, declared_cap)
        return max(0, learned)

    def _estimate_prompt(
        self,
        program: Program,
        message_bytes: int,
        message_count: int,
    ) -> int:
        direct = math.ceil(message_bytes * self._tokens_per_byte)
        direct += 64 + 12 * message_count
        if program.last_prompt_tokens > 0:
            growth_bytes = max(0, message_bytes - program.last_message_bytes)
            grown_from_request = program.last_prompt_tokens
            grown_from_request += math.ceil(
                growth_bytes * max(self._tokens_per_byte, 1 / 2.7)
            )
            grown_from_request += 64
            # The byte delta already includes the preceding assistant output;
            # adding it to resident_tokens would count that completion twice.
            direct = max(direct, grown_from_request, program.resident_tokens + 64)
        # An uncalibrated program carries a one-page estimate guard.
        if program.last_prompt_tokens <= 0:
            direct += ATTENTION_TOKENS_PER_PAGE
        return max(1, direct)

    @staticmethod
    def _parse_request(body: bytes) -> tuple[int, int, int | None]:
        """Return observational message bytes/count and declared output cap."""

        try:
            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise TypeError
            messages = payload.get("messages")
            if not isinstance(messages, list):
                raise TypeError
            observed_context: dict[str, Any] = {"messages": messages}
            tools = payload.get("tools")
            if isinstance(tools, list):
                observed_context["tools"] = tools
            encoded = json.dumps(
                observed_context, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            cap = payload.get("max_completion_tokens")
            if cap is None:
                cap = payload.get("max_tokens")
            if not isinstance(cap, int) or isinstance(cap, bool) or cap < 0:
                cap = None
            return len(encoded), len(messages), cap
        except Exception:
            # Invalid/compressed JSON still forwards exactly; only its estimate
            # falls back to a deliberately high byte-derived value.
            return max(1, len(body)), 1, None

    def _shape_for(self, program: Program, body: bytes) -> RequestShape:
        try:
            message_bytes, count, cap = self._parse_request(body)
            prompt = self._estimate_prompt(program, message_bytes, count)
            return RequestShape(
                message_bytes=message_bytes,
                message_count=count,
                prompt_tokens=prompt,
                output_reserve=self._output_reserve(program, cap),
                declared_output_cap=cap,
            )
        except Exception:
            # Observational parsing/accounting must fail open.  This fallback
            # is bounded and intentionally expensive, but never alters body.
            message_bytes = max(1, len(body))
            return RequestShape(
                message_bytes=message_bytes,
                message_count=1,
                prompt_tokens=message_bytes + ATTENTION_TOKENS_PER_PAGE,
                output_reserve=INITIAL_OUTPUT_RESERVE,
                declared_output_cap=None,
            )

    def _resident_pages(self, program: Program) -> int:
        return RESIDENT_RECURRENT_PAGES + _ceil_pages(program.resident_tokens)

    def _program_pages(self, program: Program) -> int:
        if program.inflight:
            return sum(self._tickets[ticket_id].reserved_pages for ticket_id in program.inflight)
        if program.active:
            return self._resident_pages(program)
        return 0

    def _used_pages(self) -> int:
        return sum(self._program_pages(program) for program in self._programs.values())

    def _inflight_count(self) -> int:
        return sum(len(program.inflight) for program in self._programs.values())

    def _queued_count(self) -> int:
        return sum(len(program.queued) for program in self._programs.values())

    def _candidate_increment(self, ticket: Ticket) -> int:
        before = self._program_pages(ticket.program)
        # Requests for one program are serialized, so an admitted request
        # replaces its resident claim rather than adding another fixed state.
        return ticket.reserved_pages - before

    def _pin_envelope_used(self) -> int:
        return sum(
            program.pin_envelope_pages
            for program in self._programs.values()
            if program.pinned
        )

    def _pinned_count(self) -> int:
        return sum(int(program.pinned) for program in self._programs.values())

    def _member_resident_equivalent(self, program: Program) -> int:
        """State a member owns, excluding the request-only recurrent peak."""

        if program.inflight:
            ticket_id = next(iter(program.inflight))
            request_pages = self._tickets[ticket_id].reserved_pages
            recurrent_delta = PENDING_RECURRENT_PAGES - RESIDENT_RECURRENT_PAGES
            return max(RESIDENT_RECURRENT_PAGES, request_pages - recurrent_delta)
        if program.active:
            return self._resident_pages(program)
        return 0

    def _pinned_actual_pages(self) -> int:
        return sum(
            self._program_pages(program)
            for program in self._programs.values()
            if program.pinned
        )

    def _pinned_resident_equivalent_pages(self) -> int:
        return sum(
            self._member_resident_equivalent(program)
            for program in self._programs.values()
            if program.pinned
        )

    def _borrower_tickets(self) -> list[Ticket]:
        return [
            ticket
            for ticket in self._tickets.values()
            if ticket.phase is TicketPhase.ADMITTED
            and ticket.admission_lane is AdmissionLane.BORROWER
        ]

    def _borrower_inflight_count(self) -> int:
        return len(self._borrower_tickets())

    def _borrower_actual_pages(self) -> int:
        return sum(ticket.reserved_pages for ticket in self._borrower_tickets())

    def _head_for_program(self, ticket: Ticket) -> bool:
        queued = ticket.program.queued
        return bool(queued and queued[0] == ticket.ident)

    def _desired_pin_envelope(self, ticket: Ticket) -> int:
        recurrent_delta = PENDING_RECURRENT_PAGES - RESIDENT_RECURRENT_PAGES
        prospective = max(
            RESIDENT_RECURRENT_PAGES,
            ticket.reserved_pages - recurrent_delta,
        )
        if ticket.program.active and not ticket.program.inflight:
            prospective = max(prospective, self._resident_pages(ticket.program))
        if prospective > PINNED_ENVELOPE_BUDGET:
            return 0
        return prospective

    def _pin_expired(self, program: Program, now: float) -> bool:
        return (
            program.pin_completed_steps >= PIN_LEASE_COMPLETIONS
            and program.pin_started_at > 0
            and now - program.pin_started_at >= PIN_LEASE_SECONDS
        )

    def _prune_rotation_events(self, now: float) -> None:
        cutoff = now - ROTATION_WINDOW_SECONDS
        while self._rotation_events and self._rotation_events[0][0] <= cutoff:
            self._rotation_events.popleft()

    def _rotation_budget_allows(self, pages: int, now: float) -> bool:
        self._prune_rotation_events(now)
        allowed = (
            len(self._rotation_events) < ROTATION_MAX_PROGRAMS
            and sum(event_pages for _, event_pages in self._rotation_events) + pages
            <= ROTATION_MAX_PAGES
        )
        if not allowed:
            self._counters["rotation_budget_denials"] += 1
        return allowed

    def _record_rotation(self, pages: int, now: float) -> None:
        self._prune_rotation_events(now)
        self._rotation_events.append((now, pages))
        assert len(self._rotation_events) <= ROTATION_MAX_PROGRAMS
        assert sum(value for _, value in self._rotation_events) <= ROTATION_MAX_PAGES

    def _safe_pin_victims(
        self,
        candidate: Program,
        now: float,
        *,
        forced: bool,
    ) -> list[Program]:
        victims = [
            program
            for program in self._programs.values()
            if program is not candidate
            and program.pinned
            and not program.inflight
            and not program.queued
            and (forced or self._pin_expired(program, now))
        ]
        victims.sort(
            key=lambda program: (
                0.0 if self._pin_expired(program, now) else 1.0,
                program.last_completed,
                -float(program.pin_envelope_pages),
                program.pin_generation,
                program.key,
            )
        )
        return victims

    def _demote_pin(
        self,
        program: Program,
        now: float,
        *,
        rotation: bool = False,
    ) -> bool:
        if not program.pinned or program.inflight:
            return False
        released = program.pin_envelope_pages
        if rotation and not self._rotation_budget_allows(released, now):
            return False
        program.pinned = False
        program.pin_envelope_pages = 0
        program.pin_started_at = 0.0
        program.pin_completed_steps = 0
        self._pause(program, now)
        if rotation:
            self._record_rotation(released, now)
        self._counters["pin_demotions"] += 1
        return True

    def _promote_pin(
        self,
        ticket: Ticket,
        desired: int,
        now: float,
    ) -> None:
        program = ticket.program
        assert not program.pinned and not program.inflight
        assert desired > 0
        assert self._pinned_count() < MAX_PINNED_PROGRAMS
        assert self._pin_envelope_used() + desired <= PINNED_ENVELOPE_BUDGET
        self._pin_generation += 1
        program.pinned = True
        program.pin_envelope_pages = desired
        program.pin_started_at = now
        program.pin_completed_steps = 0
        program.pin_generation = self._pin_generation
        ticket.granted_pin = True
        self._counters["pin_promotions"] += 1

    def _promote_completed_borrower(
        self,
        program: Program,
        desired: int,
        now: float,
    ) -> None:
        """Retain one successful borrower's already-materialized context."""

        assert not program.pinned and not program.inflight
        assert program.active and not program.release_pending
        assert desired == self._resident_pages(program)
        assert desired > 0
        assert self._pinned_count() < MAX_PINNED_PROGRAMS
        assert self._pin_envelope_used() + desired <= PINNED_ENVELOPE_BUDGET
        assert self._used_pages() <= WORKING_PAGE_BUDGET
        self._pin_generation += 1
        program.pinned = True
        program.pin_envelope_pages = desired
        program.pin_started_at = now
        # This envelope was earned by the response that just completed.  Count
        # that real turn once; do not give a completion-time promotion an
        # artificial extra lease turn.
        program.pin_completed_steps = 1
        program.pin_generation = self._pin_generation
        self._counters["pin_promotions"] += 1

    def _expand_pin(
        self,
        program: Program,
        desired: int,
        now: float,
    ) -> bool:
        """Grow one member, demoting at most one lease-eligible owner."""

        if not program.pinned or desired <= program.pin_envelope_pages:
            return True
        if desired > PINNED_ENVELOPE_BUDGET:
            return False
        extra = desired - program.pin_envelope_pages
        free = PINNED_ENVELOPE_BUDGET - self._pin_envelope_used()
        if free < extra:
            needed = extra - free
            victim = next(
                (
                    item
                    for item in self._safe_pin_victims(
                        program, now, forced=False
                    )
                    if item.pin_envelope_pages >= needed
                    and self._rotation_budget_allows(
                        item.pin_envelope_pages, now
                    )
                ),
                None,
            )
            if victim is None or not self._demote_pin(
                victim, now, rotation=True
            ):
                return False
        if self._pin_envelope_used() + extra > PINNED_ENVELOPE_BUDGET:
            return False
        program.pin_envelope_pages = desired
        self._counters["pin_envelope_expansions"] += 1
        return True

    def _reconcile_pin_completion(self, program: Program, now: float) -> None:
        if not program.pinned or program.inflight or not program.active:
            return
        desired = self._resident_pages(program)
        if desired <= program.pin_envelope_pages:
            return
        if not self._expand_pin(program, desired, now):
            # Completion is the safe boundary: never report an owner envelope
            # smaller than the state that would remain resident.
            self._demote_pin(program, now)
            self._counters["pin_growth_demotions"] += 1

    def _prepare_pinned_request(self, ticket: Ticket, now: float) -> bool:
        program = ticket.program
        if not program.pinned:
            return False
        desired = self._desired_pin_envelope(ticket)
        if desired and self._expand_pin(program, desired, now):
            return True
        # If its next resident-equivalent claim cannot be represented, this
        # request uses the borrower lane and yields at completion.
        self._demote_pin(program, now)
        return False

    def _rotate_one(
        self,
        now: float,
        candidates: list[Ticket],
    ) -> bool:
        """Fill genuinely free membership; never proactively evict an owner.

        Full-envelope exchanges are response-causal and happen only in
        ``_borrower_completion_handoff``.  This keeps a timer or an offer from
        evicting an owner in the absence of a safe, observed service boundary.
        """

        outsiders = [
            ticket
            for ticket in candidates
            if not ticket.program.pinned
            and self._desired_pin_envelope(ticket) > 0
        ]
        if not outsiders:
            return False
        outsiders.sort(key=lambda ticket: (ticket.enqueued_at, ticket.ident))
        target = outsiders[0]
        desired = self._desired_pin_envelope(target)
        free = PINNED_ENVELOPE_BUDGET - self._pin_envelope_used()
        count_full = self._pinned_count() >= MAX_PINNED_PROGRAMS
        if free >= desired and not count_full:
            self._promote_pin(target, desired, now)
            return True
        return False

    def _late_owner_exchange_candidate(
        self,
        candidates: list[Ticket],
        now: float,
    ) -> Ticket | None:
        """Return the sole chronological head needing a late owner slot."""

        if not candidates:
            return None
        target = min(
            candidates,
            key=lambda ticket: (ticket.enqueued_at, ticket.ident),
        )
        if (
            target.phase is not TicketPhase.QUEUED
            or target.gate.cancelled()
            or target.program.pinned
            or target.program.active
            or target.program.ephemeral
            or target.program.inflight
            or target.program.release_pending
            or not self._head_for_program(target)
            or target.ordinal <= HANDOFF_ENTRY_MAX_ORDINAL
            or now - target.enqueued_at
            < LATE_OWNER_EXCHANGE_AGE_SECONDS
            or self._desired_pin_envelope(target) <= 0
        ):
            return None
        lane_fits, _ = self._borrower_fits(
            target, allow_expanded=False
        )
        if lane_fits and self._hard_fits(target):
            return None
        return target

    def _has_late_owner_exchange_debt(self, now: float) -> bool:
        return self._late_owner_exchange_candidate(
            self._ranked_tickets(now), now
        ) is not None

    def _late_owner_exchange(
        self,
        now: float,
        candidates: list[Ticket],
    ) -> bool:
        """Exchange one completed idle owner for one truly blocked late head.

        Unlike expanded recurring debt, this never increases cold borrower
        concurrency.  The oldest global head must already have failed to fit
        the ordinary 32/292 borrower plane for a full turnover window.  Both
        membership and physical-page effects are proved before the lock-held
        one-for-one transition.
        """

        target = self._late_owner_exchange_candidate(candidates, now)
        if target is None:
            return False

        desired = self._desired_pin_envelope(target)
        self._counters["late_owner_exchange_attempts"] += 1
        free = PINNED_ENVELOPE_BUDGET - self._pin_envelope_used()
        count_full = self._pinned_count() >= MAX_PINNED_PROGRAMS
        if free >= desired and not count_full and self._hard_fits(target):
            self._promote_pin(target, desired, now)
            self._counters[
                "late_owner_exchange_direct_promotions"
            ] += 1
            return True
        used = self._used_pages()
        envelope_used = self._pin_envelope_used()
        target_increment = self._candidate_increment(target)
        victims = [
            program
            for program in self._programs.values()
            if program is not target.program
            and program.pinned
            and program.active
            and not program.inflight
            and not program.queued
            and not program.release_pending
            and program.pin_completed_steps >= 1
            and program.last_completed > 0
        ]
        victims.sort(
            key=lambda program: (
                0.0
                if now - program.last_completed
                >= HANDOFF_STALE_OWNER_SECONDS
                else 1.0,
                program.last_completed,
                -float(program.pin_envelope_pages),
                program.pin_generation,
                program.key,
            )
        )

        victim = next(
            (
                program
                for program in victims
                if envelope_used
                - program.pin_envelope_pages
                + desired
                <= PINNED_ENVELOPE_BUDGET
                and used
                - self._program_pages(program)
                + target_increment
                <= WORKING_PAGE_BUDGET
                and self._rotation_budget_allows(
                    program.pin_envelope_pages, now
                )
            ),
            None,
        )
        if victim is None:
            self._counters["late_owner_exchange_no_victim"] += 1
            return False

        if not self._demote_pin(victim, now, rotation=True):
            raise AssertionError("late owner exchange victim changed")
        if (
            self._pinned_count() >= MAX_PINNED_PROGRAMS
            or self._pin_envelope_used() + desired
            > PINNED_ENVELOPE_BUDGET
        ):
            raise AssertionError("late owner exchange envelope diverged")
        self._promote_pin(target, desired, now)
        if not self._hard_fits(target):
            raise AssertionError("late owner exchange hard fit diverged")
        self._counters["pin_rotations"] += 1
        self._counters["forced_rotations"] += 1
        self._counters["late_owner_exchange_completions"] += 1
        return True

    def _borrower_completion_handoff(
        self,
        ticket: Ticket,
        program: Program,
        now: float,
    ) -> bool:
        """Keep a recurring successful borrower by swapping one stale owner.

        The completed request has already been removed from ``program.inflight``
        and response observation has reconciled ``resident_tokens``.  Thus the
        target is the freshest materialized cache boundary available.  A timer
        or queued request can never call this transition.
        """

        self._counters["handoff_attempts"] += 1
        if (
            ticket.admission_lane is not AdmissionLane.BORROWER
            or ticket.ordinal < HANDOFF_MIN_TICKET_ORDINAL
            or program.pinned
            or program.inflight
            or program.release_pending
            or not program.active
        ):
            self._counters["handoff_ineligible_target"] += 1
            return False

        desired = self._resident_pages(program)
        if not (0 < desired <= PINNED_ENVELOPE_BUDGET):
            self._counters["handoff_ineligible_target"] += 1
            return False

        used = self._used_pages()
        envelope_used = self._pin_envelope_used()
        count_full = self._pinned_count() >= MAX_PINNED_PROGRAMS

        # Genuinely unused cohort capacity needs no victim and is not turnover.
        if (
            not count_full
            and envelope_used + desired <= PINNED_ENVELOPE_BUDGET
            and used <= WORKING_PAGE_BUDGET
        ):
            self._promote_completed_borrower(program, desired, now)
            self._counters["handoff_direct_promotions"] += 1
            self._counters["handoff_completions"] += 1
            return True

        # A global late head has already waited through the ordinary borrower
        # plane.  Preserve victim-consuming turnover for its pre-service owner
        # exchange; genuinely free promotion above remains work-conserving.
        if self._has_late_owner_exchange_debt(now):
            self._counters[
                "late_owner_exchange_handoff_deferrals"
            ] += 1
            return False

        victims = [
            candidate
            for candidate in self._programs.values()
            if candidate is not program
            and candidate.pinned
            and candidate.active
            and not candidate.inflight
            and not candidate.queued
            and not candidate.release_pending
            and candidate.last_completed > 0
            and now - candidate.last_completed >= HANDOFF_STALE_OWNER_SECONDS
        ]
        victims.sort(
            key=lambda candidate: (
                candidate.last_completed,
                -float(candidate.pin_envelope_pages),
                candidate.pin_generation,
                candidate.key,
            )
        )

        victim: Program | None = None
        for candidate in victims:
            if (
                envelope_used - candidate.pin_envelope_pages + desired
                > PINNED_ENVELOPE_BUDGET
            ):
                continue
            if used - self._program_pages(candidate) > WORKING_PAGE_BUDGET:
                continue
            if not self._rotation_budget_allows(
                candidate.pin_envelope_pages, now
            ):
                continue
            victim = candidate
            break
        if victim is None:
            self._counters["handoff_no_compatible_victim"] += 1
            return False

        # The helpers repeat all mutable preconditions.  No await or fallible
        # operation occurs inside this lock-held two-state transition.
        if not self._demote_pin(victim, now, rotation=True):
            return False
        if (
            self._pinned_count() >= MAX_PINNED_PROGRAMS
            or self._pin_envelope_used() + desired > PINNED_ENVELOPE_BUDGET
            or self._used_pages() > WORKING_PAGE_BUDGET
        ):
            raise AssertionError("atomic borrower handoff preflight diverged")
        self._promote_completed_borrower(program, desired, now)
        self._counters["pin_rotations"] += 1
        self._counters["forced_rotations"] += 1
        self._counters["handoff_completions"] += 1
        return True

    def _ranked_tickets(self, now: float) -> list[Ticket]:
        candidates = [
            ticket
            for ticket in self._tickets.values()
            if ticket.phase is TicketPhase.QUEUED
            and not ticket.program.inflight
            and self._head_for_program(ticket)
        ]
        # A ticket can wait while an earlier request for the same program
        # completes and while global calibration changes.  Refresh from its
        # bounded parsed fields before comparing or installing the claim.
        for ticket in candidates:
            ticket.shape.prompt_tokens = self._estimate_prompt(
                ticket.program,
                ticket.shape.message_bytes,
                ticket.shape.message_count,
            )
            ticket.shape.output_reserve = self._output_reserve(
                ticket.program, ticket.shape.declared_output_cap
            )

        def key(ticket: Ticket) -> tuple[float, ...]:
            wait = max(0.0, now - ticket.enqueued_at)
            program = ticket.program
            if wait >= STARVATION_AGE_SECONDS:
                return (0.0, ticket.enqueued_at, float(ticket.ident))
            if program.pinned:
                return (1.0, ticket.enqueued_at, float(ticket.ident))
            # Borrowers fill otherwise-unused capacity.  Aging continuously
            # offsets their marginal page cost before hard promotion.
            marginal_pages = max(0, self._candidate_increment(ticket))
            effective_pages = max(0.0, marginal_pages - wait / 8.0)
            return (
                2.0,
                effective_pages,
                ticket.enqueued_at,
                float(ticket.ident),
            )

        candidates.sort(key=key)
        return candidates

    def _pause(self, program: Program, now: float) -> None:
        if not program.active or program.inflight:
            return
        program.active = False
        program.phase = ProgramPhase.PAUSED
        program.activation_steps = 0
        program.last_paused = now
        self._counters["pauses"] += 1

    def _hard_fits(self, ticket: Ticket) -> bool:
        used = self._used_pages()
        increment = self._candidate_increment(ticket)
        if used + increment <= WORKING_PAGE_BUDGET:
            return True
        return (
            ticket.reserved_pages == WORKING_PAGE_BUDGET
            and self._inflight_count() == 0
        )

    def _make_hard_room(self, ticket: Ticket, now: float) -> bool:
        if not self._hard_fits(ticket):
            return False
        used = self._used_pages()
        increment = self._candidate_increment(ticket)
        if used + increment <= WORKING_PAGE_BUDGET:
            return True
        # A saturated estimate drains and runs alone.  Only safe ACTING state
        # is paused; no in-flight request is ever evicted or duplicated.
        for program in self._programs.values():
            if program is not ticket.program:
                self._pause(program, now)
        return (
            self._used_pages() + self._candidate_increment(ticket)
            <= WORKING_PAGE_BUDGET
        )

    def _make_aged_drain_room(self, ticket: Ticket, now: float) -> bool:
        """At hard age, shed only idle resident state and run the ticket."""

        if self._inflight_count() != 0:
            return False
        while (
            self._used_pages() + self._candidate_increment(ticket)
            > WORKING_PAGE_BUDGET
        ):
            victims = [
                program
                for program in self._programs.values()
                if program is not ticket.program
                and program.active
                and not program.inflight
            ]
            if not victims:
                return False
            victims.sort(
                key=lambda program: (
                    0.0 if not program.pinned else 1.0,
                    0.0 if self._pin_expired(program, now) else 1.0,
                    -float(self._program_pages(program)),
                    program.last_completed,
                )
            )
            self._pause(victims[0], now)
        return True

    def _borrower_fits(
        self,
        ticket: Ticket,
        *,
        allow_expanded: bool,
    ) -> tuple[bool, bool]:
        next_count = self._borrower_inflight_count() + 1
        next_pages = self._borrower_actual_pages() + ticket.reserved_pages
        if (
            next_count <= BORROWER_BASE_COUNT
            and next_pages <= BORROWER_BASE_PAGE_BUDGET
        ):
            return True, False
        if (
            allow_expanded
            and self._inflight_count() < BORROWER_UNDERFILL_INFLIGHT
            and next_count <= BORROWER_EXPANDED_COUNT
            and next_pages <= BORROWER_EXPANDED_PAGE_BUDGET
        ):
            return True, True
        return False, False

    def _first_service_lane_fits(
        self,
        ticket: Ticket,
    ) -> tuple[bool, bool]:
        """Use the existing 64/448 total without the ordinary underfill gate.

        The boolean result is ``(fits, uses_expanded_part)``.  This creates no
        additional physical lane: every admitted ticket remains a BORROWER and
        is included by the existing shared invariants.
        """

        next_count = self._borrower_inflight_count() + 1
        next_pages = self._borrower_actual_pages() + ticket.reserved_pages
        if (
            next_count > BORROWER_EXPANDED_COUNT
            or next_pages > BORROWER_EXPANDED_PAGE_BUDGET
        ):
            return False, False
        ordinary = (
            next_count <= BORROWER_BASE_COUNT
            and next_pages <= BORROWER_BASE_PAGE_BUDGET
        )
        return True, not ordinary

    def _first_service_choices(
        self,
        candidates: list[Ticket],
        now: float,
    ) -> tuple[tuple[Ticket, bool] | None, Ticket | None]:
        """Return a direct-fit service debt and one bounded runway target.

        First requests become eligible at 30 seconds.  At 45 seconds, only an
        ordinal-two or ordinal-three chronological head gets a bounded
        handoff-entry opportunity.  Later turns retain the ordinary aged
        contract; a separate owner-exchange state may act before this plane.
        Lane-incompatible and large non-fitting tickets never become younger
        global barriers.  A directly hard-fitting large ticket is still useful
        work and may be returned as the direct choice.
        """

        debts = [
            ticket
            for ticket in candidates
            if not ticket.program.pinned
            and not ticket.program.inflight
            and self._head_for_program(ticket)
            and (
                (
                    ticket.ordinal == 1
                    and now - ticket.enqueued_at
                    >= FIRST_SERVICE_AGE_SECONDS
                )
                or (
                    HANDOFF_MIN_TICKET_ORDINAL
                    <= ticket.ordinal
                    <= HANDOFF_ENTRY_MAX_ORDINAL
                    and now - ticket.enqueued_at
                    >= STARVATION_AGE_SECONDS
                )
            )
        ]
        debts.sort(key=lambda ticket: (ticket.enqueued_at, ticket.ident))
        runway: Ticket | None = None
        for ticket in debts:
            lane_fits, expanded = self._first_service_lane_fits(ticket)
            if not lane_fits:
                self._counters["first_service_lane_skips"] += 1
                continue
            if self._hard_fits(ticket):
                return (ticket, expanded), runway
            if ticket.reserved_pages > FIRST_SERVICE_RUNWAY_MAX_PAGES:
                self._counters["first_service_large_skips"] += 1
                continue
            if runway is None:
                runway = ticket
        return None, runway

    def _owner_preserves_first_service_runway(
        self,
        ticket: Ticket,
        runway_pages: int,
    ) -> bool:
        """Whether one owner return leaves the sole debt increment intact."""

        increment = self._candidate_increment(ticket)
        return (
            increment <= 0
            or self._used_pages() + increment
            <= WORKING_PAGE_BUDGET - runway_pages
        )

    def _membership_invariants(self, now: float | None = None) -> None:
        check_now = time.monotonic() if now is None else now
        self._prune_rotation_events(check_now)
        assert self._used_pages() <= WORKING_PAGE_BUDGET
        assert self._pinned_count() <= MAX_PINNED_PROGRAMS
        assert self._pin_envelope_used() <= PINNED_ENVELOPE_BUDGET
        assert all(
            self._member_resident_equivalent(program)
            <= program.pin_envelope_pages
            for program in self._programs.values()
            if program.pinned
        )
        assert self._borrower_inflight_count() <= BORROWER_EXPANDED_COUNT
        assert self._borrower_actual_pages() <= BORROWER_EXPANDED_PAGE_BUDGET
        assert len(self._rotation_events) <= ROTATION_MAX_PROGRAMS
        assert sum(pages for _, pages in self._rotation_events) <= ROTATION_MAX_PAGES
        assert self._inflight_count() <= MAX_BACKEND_CHATS

    def _admit(
        self,
        ticket: Ticket,
        now: float,
        *,
        borrower_expanded: bool = False,
        force_drain: bool = False,
    ) -> None:
        program = ticket.program
        was_active = program.active
        ticket.prior_active = program.active
        ticket.prior_resident_tokens = program.resident_tokens
        ticket.prior_activation_steps = program.activation_steps
        ticket.phase = TicketPhase.ADMITTED
        ticket.admitted_at = now
        if force_drain or ticket.reserved_pages == WORKING_PAGE_BUDGET:
            ticket.admission_lane = AdmissionLane.DRAIN
            self._counters["drain_admissions"] += 1
        elif program.pinned:
            ticket.admission_lane = AdmissionLane.PINNED
        else:
            ticket.admission_lane = AdmissionLane.BORROWER
            self._counters["borrower_admissions"] += 1
            if borrower_expanded:
                self._counters["borrower_expanded_admissions"] += 1
        self._tickets[ticket.ident] = ticket
        if ticket.ident in program.queued:
            program.queued.remove(ticket.ident)
        program.inflight.add(ticket.ident)
        program.active = True
        program.phase = ProgramPhase.REASONING
        program.last_admitted = now
        program.queue_wait_ewma = _ewma(
            program.queue_wait_ewma, now - ticket.enqueued_at
        )
        if not was_active:
            self._counters["restorations"] += 1
        if now - ticket.enqueued_at >= STARVATION_AGE_SECONDS:
            self._counters["aged_admissions"] += 1
        self._counters["admitted"] += 1
        self._membership_invariants(now)
        if not ticket.gate.done():
            ticket.gate.set_result(None)

    def _enforce_budget(self, now: float) -> None:
        # Usage can reveal a larger resident context than was estimated.  Shed
        # only boundary-safe ACTING reservations until the invariant recovers.
        while self._used_pages() > WORKING_PAGE_BUDGET:
            victims = [
                program
                for program in self._programs.values()
                if program.active and not program.inflight
            ]
            if not victims:
                raise AssertionError("hard page ledger has no safe recovery")
            victims.sort(
                key=lambda program: (
                    0.0 if not program.pinned else 1.0,
                    0.0 if self._pin_expired(program, now) else 1.0,
                    -float(self._program_pages(program)),
                    program.last_completed,
                )
            )
            self._pause(victims[0], now)
        assert self._used_pages() <= WORKING_PAGE_BUDGET

    def _dispatch(self) -> None:
        now = time.monotonic()
        self._enforce_budget(now)
        ranked = self._ranked_tickets(now)

        # Reconcile at most one queued member before the admission loop.  A
        # reconciliation itself can demote at most one owner.
        demotions_before = self._counters["pin_demotions"]
        pinned_to_prepare = next(
            (
                ticket
                for ticket in ranked
                if ticket.program.pinned
                and (
                    self._desired_pin_envelope(ticket) == 0
                    or self._desired_pin_envelope(ticket)
                    > ticket.program.pin_envelope_pages
                )
            ),
            None,
        )
        if pinned_to_prepare is not None:
            self._prepare_pinned_request(pinned_to_prepare, now)
        ranked = self._ranked_tickets(now)
        if self._counters["pin_demotions"] == demotions_before:
            late_debt = self._late_owner_exchange_candidate(
                ranked, now
            )
            if late_debt is not None:
                self._late_owner_exchange(now, ranked)
            else:
                self._rotate_one(now, ranked)

        while self._inflight_count() < MAX_BACKEND_CHATS:
            ranked = self._ranked_tickets(now)
            for ticket in list(ranked):
                if ticket.gate.cancelled():
                    self._cancel_queued(ticket, now)
            ranked = [
                ticket
                for ticket in ranked
                if ticket.phase is TicketPhase.QUEUED
            ]
            if not ranked:
                return

            # Hard age remains the older service contract.  A younger 30 s
            # first-wave debt may not supersede any chronological 45 s head;
            # this expanded plane remains limited to handoff-entry ordinals.
            chronological = min(
                ranked, key=lambda ticket: (ticket.enqueued_at, ticket.ident)
            )
            chronological_hard_aged = (
                now - chronological.enqueued_at >= STARVATION_AGE_SECONDS
            )
            first_service_scope = (
                [chronological] if chronological_hard_aged else ranked
            )

            # Service debt is finite and chronological: an ordinal-one request
            # gets the existing expanded 64/448 borrower total after 30 s; at
            # 45 s the sole global chronological head gets the same opportunity
            # at ordinal two/three.  Later blocked turns use ordinary service
            # unless the earlier one-for-one owner exchange succeeded.  Direct
            # fit substitutes for a positive-
            # growth owner return.  If hard pages alone block it, one at-most-
            # 32-page target retains its increment as completion runway.
            direct_first, first_runway = self._first_service_choices(
                first_service_scope, now
            )
            if direct_first is not None:
                target, expanded = direct_first
                if not self._make_hard_room(target, now):
                    raise AssertionError("direct first-service fit diverged")
                self._admit(target, now, borrower_expanded=expanded)
                if target.ordinal == 1:
                    self._counters["first_service_admissions"] += 1
                    if expanded:
                        self._counters[
                            "first_service_expanded_admissions"
                        ] += 1
                else:
                    self._counters["recurring_service_admissions"] += 1
                    if expanded:
                        self._counters[
                            "recurring_service_expanded_admissions"
                        ] += 1
                now = time.monotonic()
                continue

            first_runway_pages = 0
            if first_runway is not None:
                first_runway_pages = max(
                    0, self._candidate_increment(first_runway)
                )
                if not (
                    0 < first_runway_pages
                    <= FIRST_SERVICE_RUNWAY_MAX_PAGES
                ):
                    raise AssertionError("invalid first-service runway")
                if first_runway.ordinal == 1:
                    self._counters["first_service_runway_waits"] += 1
                else:
                    self._counters["recurring_service_runway_waits"] += 1

            oldest = ranked[0]
            oldest_aged = now - oldest.enqueued_at >= STARVATION_AGE_SECONDS
            # Ordinary aged requests remain work-conserving: rank escalation,
            # borrower FIFO, and completion handoffs serve them without
            # stopping unrelated model work.  Only a request too large to fit
            # the ordinary borrower page lane may invoke exclusive drain.
            if (
                first_runway is None
                and oldest_aged
                and not oldest.program.pinned
                and oldest.reserved_pages > BORROWER_BASE_PAGE_BUDGET
            ):
                if (
                    self._make_hard_room(oldest, now)
                    or self._make_aged_drain_room(oldest, now)
                ) and self._inflight_count() == 0:
                    self._admit(oldest, now, force_drain=True)
                    now = time.monotonic()
                    continue
                return

            aged_borrower_capacity_blocked = False
            aged_borrower_hard_blocked = False
            if (
                first_runway is None
                and oldest_aged
                and not oldest.program.pinned
                and oldest.reserved_pages <= BORROWER_BASE_PAGE_BUDGET
            ):
                # A representable aged outsider owns the next borrower service
                # opportunity, not the entire backend.  While the 32/292 lane
                # is full, fitting owner returns remain work-conserving because
                # a real borrower completion is the required wake.  Once lane
                # capacity exists, only non-positive-marginal owner returns may
                # bypass a hard-page block, so completion-freed pages converge
                # on this exact claim without cancelling or draining inflight
                # work.
                lane_fits, _ = self._borrower_fits(
                    oldest, allow_expanded=False
                )
                room = False
                if lane_fits:
                    if self._hard_fits(oldest):
                        room = self._make_hard_room(oldest, now)
                    elif self._inflight_count() == 0:
                        room = self._make_aged_drain_room(oldest, now)
                if room:
                    self._admit(oldest, now)
                    self._counters["aged_borrower_admissions"] += 1
                    now = time.monotonic()
                    continue
                if lane_fits:
                    aged_borrower_hard_blocked = True
                    self._counters["aged_borrower_hard_waits"] += 1
                else:
                    aged_borrower_capacity_blocked = True
                    self._counters["aged_borrower_capacity_waits"] += 1

            pinned = [ticket for ticket in ranked if ticket.program.pinned]
            pinned_fit = [
                ticket
                for ticket in pinned
                if 0 < self._desired_pin_envelope(ticket)
                <= ticket.program.pin_envelope_pages
                and self._hard_fits(ticket)
                and (
                    not aged_borrower_hard_blocked
                    or self._candidate_increment(ticket) <= 0
                )
                and (
                    first_runway is None
                    or self._owner_preserves_first_service_runway(
                        ticket, first_runway_pages
                    )
                )
            ]
            if pinned_fit:
                target = pinned_fit[0]
                if self._make_hard_room(target, now):
                    if first_runway is not None:
                        if first_runway.ordinal == 1:
                            self._counters[
                                "first_service_owner_bypasses"
                            ] += 1
                        else:
                            self._counters[
                                "recurring_service_owner_bypasses"
                            ] += 1
                    if aged_borrower_capacity_blocked:
                        self._counters[
                            "aged_borrower_owner_bypasses"
                        ] += 1
                    self._admit(target, now)
                    now = time.monotonic()
                    continue
            pinned_blocked = bool(pinned)

            if first_runway is not None:
                # No unrelated positive-growth request may consume this one
                # finite chronological service opportunity.  A real completion,
                # cancellation, or release re-enters dispatch; no timer spins
                # and no owner/inflight state is evicted.
                return

            if aged_borrower_capacity_blocked or aged_borrower_hard_blocked:
                # No younger positive-growth request consumes the aged
                # borrower's next lane/page opportunity.  A real admitted
                # completion, cancellation, or release re-enters dispatch.
                return

            # No ready pinned request fits.  Outsiders can now use the bounded
            # borrower lane.  Its wider cap is an underfill escape only when no
            # owner is waiting on pages or envelope space.
            allow_expanded = (
                not pinned_blocked
                and self._inflight_count() < BORROWER_UNDERFILL_INFLIGHT
            )
            admitted = False
            for ticket in ranked:
                if ticket.program.pinned:
                    continue
                if not self._hard_fits(ticket):
                    continue
                if (
                    ticket.reserved_pages == WORKING_PAGE_BUDGET
                    and self._inflight_count() == 0
                    and self._make_hard_room(ticket, now)
                ):
                    self._admit(ticket, now, force_drain=True)
                    admitted = True
                    break
                fits, expanded = self._borrower_fits(
                    ticket, allow_expanded=allow_expanded
                )
                if not fits:
                    continue
                if self._make_hard_room(ticket, now):
                    self._admit(
                        ticket, now, borrower_expanded=expanded
                    )
                    admitted = True
                    break
            if not admitted:
                return
            now = time.monotonic()

    def _arm_rotation_timer(self) -> None:
        now = time.monotonic()
        first_service_deadlines = [
            ticket.enqueued_at + FIRST_SERVICE_AGE_SECONDS
            for ticket in self._tickets.values()
            if ticket.phase is TicketPhase.QUEUED
            and ticket.ordinal == 1
            and not ticket.program.inflight
            and self._head_for_program(ticket)
        ]
        age_deadlines = [
            ticket.enqueued_at + STARVATION_AGE_SECONDS
            for ticket in self._tickets.values()
            if ticket.phase is TicketPhase.QUEUED
            and not ticket.program.inflight
            and self._head_for_program(ticket)
        ]
        late_exchange_deadlines = [
            ticket.enqueued_at + LATE_OWNER_EXCHANGE_AGE_SECONDS
            for ticket in self._tickets.values()
            if ticket.phase is TicketPhase.QUEUED
            and ticket.ordinal > HANDOFF_ENTRY_MAX_ORDINAL
            and not ticket.program.pinned
            and not ticket.program.inflight
            and self._head_for_program(ticket)
        ]
        future_deadlines = [
            deadline
            for deadline in (
                first_service_deadlines
                + age_deadlines
                + late_exchange_deadlines
            )
            if deadline > now
        ]
        self._prune_rotation_events(now)
        if self._rotation_events:
            future_deadlines.append(
                self._rotation_events[0][0] + ROTATION_WINDOW_SECONDS
            )
        deadline = min(future_deadlines) if future_deadlines else None
        if (
            self._rotation_timer is not None
            and not self._rotation_timer.cancelled()
            and self._rotation_timer_deadline > 0.0
            and self._rotation_timer_deadline <= now + 1e-6
        ):
            # Dispatch may sample just before a deadline and rearm just after
            # it.  Preserve the already-due callback for one fresh dispatch
            # instead of swallowing the temporal state transition.
            return
        if self._rotation_timer is not None:
            self._rotation_timer.cancel()
            self._rotation_timer = None
        self._rotation_timer_deadline = deadline or 0.0
        if deadline is None:
            # A due request with no changing time predicate will be retried on
            # the next real completion/cancellation.  Avoid a zero-delay spin.
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._rotation_timer = loop.call_later(
            max(0.001, deadline - now),
            self._rotation_timer_fired,
            deadline,
        )

    def _rotation_timer_fired(self, expected_deadline: float) -> None:
        self._rotation_timer = None
        asyncio.create_task(self._expire_rotation(expected_deadline))

    async def _expire_rotation(self, expected_deadline: float) -> None:
        async with self._lock:
            if abs(self._rotation_timer_deadline - expected_deadline) >= 1e-6:
                return
            self._dispatch()
            self._arm_rotation_timer()

    def _prune(self, now: float) -> None:
        removable = [
            program
            for program in self._programs.values()
            if not program.queued
            and not program.inflight
            and (
                program.release_pending
                or program.ephemeral
                or now - program.last_seen >= STATE_TTL_SECONDS
            )
        ]
        for program in removable:
            self._programs.pop(program.key, None)

        if len(self._programs) < MAX_PROGRAMS:
            return
        idle = [
            program
            for program in self._programs.values()
            if not program.queued and not program.inflight
        ]
        idle.sort(key=lambda program: program.last_seen)
        target = max(0, len(self._programs) - MAX_PROGRAMS + 1)
        for program in idle[:target]:
            self._programs.pop(program.key, None)

    async def offer(self, key: str, body: bytes, *, ephemeral: bool) -> Ticket:
        loop = asyncio.get_running_loop()
        async with self._lock:
            now = time.monotonic()
            self._prune(now)
            program = self._programs.get(key)
            if program is None:
                program = Program(key=key, ephemeral=ephemeral)
                self._programs[key] = program
            program.last_seen = now
            program.ordinal += 1
            ticket_id = self._next_ticket
            self._next_ticket += 1
            ticket = Ticket(
                ident=ticket_id,
                program=program,
                ordinal=program.ordinal,
                enqueued_at=now,
                shape=self._shape_for(program, body),
                gate=loop.create_future(),
            )
            self._tickets[ticket_id] = ticket
            program.queued.append(ticket_id)
            if program.active:
                program.phase = ProgramPhase.REASONING
            elif program.ordinal == 1:
                program.phase = ProgramPhase.NEW
            else:
                program.phase = ProgramPhase.PAUSED
            self._counters["offered"] += 1
            self._dispatch()
            self._arm_rotation_timer()
            return ticket

    def _cancel_queued(self, ticket: Ticket, now: float) -> None:
        program = ticket.program
        if ticket.phase is not TicketPhase.QUEUED:
            return
        ticket.phase = TicketPhase.CANCELLED
        if ticket.ident in program.queued:
            program.queued.remove(ticket.ident)
        self._tickets.pop(ticket.ident, None)
        if not ticket.gate.done():
            ticket.gate.cancel()
        if (
            ticket.granted_pin
            and program.pinned
            and not program.inflight
        ):
            # A speculative owner grant belongs to this exact queued ticket.
            # If it cancels before admission, a same-program follower cannot
            # inherit an envelope for context that was never materialized.
            self._demote_pin(program, now)
        program.last_seen = now
        program.phase = ProgramPhase.ACTING if program.active else ProgramPhase.PAUSED
        self._counters["cancelled_queued"] += 1

    async def finish(
        self,
        ticket: Ticket,
        *,
        usage: UsageRecord | None,
        response_bytes: int,
        first_response_at: float,
        completed: bool,
        response_status: int | None,
    ) -> None:
        async with self._lock:
            now = time.monotonic()
            program = ticket.program
            if ticket.phase is TicketPhase.QUEUED:
                self._cancel_queued(ticket, now)
            elif ticket.phase is TicketPhase.ADMITTED:
                successful = (
                    completed
                    and response_status is not None
                    and 200 <= response_status < 300
                )
                ticket.phase = TicketPhase.FINISHED if completed else TicketPhase.CANCELLED
                program.inflight.discard(ticket.ident)
                self._tickets.pop(ticket.ident, None)
                program.last_seen = now
                if successful:
                    self._counters["completed"] += 1
                    program.activation_steps += 1
                elif completed:
                    self._counters["backend_errors"] += 1
                    program.active = ticket.prior_active
                    program.resident_tokens = ticket.prior_resident_tokens
                    program.activation_steps = ticket.prior_activation_steps
                else:
                    self._counters["cancelled_admitted"] += 1
                    program.active = ticket.prior_active
                    program.resident_tokens = ticket.prior_resident_tokens
                    program.activation_steps = ticket.prior_activation_steps

                service = max(0.0, now - ticket.admitted_at)
                program.service_ewma = _ewma(program.service_ewma, service)
                if first_response_at:
                    program.ttft_ewma = _ewma(
                        program.ttft_ewma,
                        max(0.0, first_response_at - ticket.admitted_at),
                    )

                if successful and usage is not None:
                    program.last_prompt_tokens = usage.prompt_tokens
                    program.last_message_bytes = ticket.shape.message_bytes
                    program.resident_tokens = max(
                        program.resident_tokens,
                        usage.prompt_tokens + usage.completion_tokens,
                    )
                    program.completion_ewma = _ewma(
                        program.completion_ewma, float(usage.completion_tokens)
                    )
                    self._completion_samples.append(usage.completion_tokens)
                    if usage.completion_tokens > ticket.shape.output_reserve:
                        self._counters["output_overshoots"] += 1
                    if ticket.shape.message_bytes > 0:
                        observed_ratio = usage.prompt_tokens / ticket.shape.message_bytes
                        observed_ratio = min(1.25, max(0.08, observed_ratio * 1.10))
                        self._tokens_per_byte = max(
                            self._tokens_per_byte * 0.997, observed_ratio
                        )
                elif successful:
                    # Preserve monotonicity when usage was not requested.  The
                    # bounded response-byte proxy deliberately errs high.
                    generated = min(
                        ticket.shape.output_reserve,
                        max(0, math.ceil(response_bytes / 3)),
                    )
                    program.resident_tokens = max(
                        program.resident_tokens,
                        ticket.shape.prompt_tokens + generated,
                    )
                    program.last_message_bytes = ticket.shape.message_bytes

                if successful and program.pinned:
                    program.pin_completed_steps += 1
                    self._reconcile_pin_completion(program, now)
                elif successful:
                    retained = self._borrower_completion_handoff(
                        ticket, program, now
                    )
                    if not retained:
                        # One-shot, failed-eligibility, and capacity-blocked
                        # borrowing remains request-scoped exactly as in v3.
                        program.active = False
                        program.activation_steps = 0
                        program.last_paused = now
                        self._counters["borrower_yields"] += 1
                elif ticket.granted_pin and program.pinned:
                    # A cancelled/erroring admission must not strand a newly
                    # granted envelope that produced no valid owner turn.
                    self._demote_pin(program, now)

                if program.release_pending and not program.queued and not program.inflight:
                    program.active = False
                    program.phase = ProgramPhase.RELEASED
                    self._programs.pop(program.key, None)
                elif program.queued:
                    program.phase = ProgramPhase.REASONING if program.active else ProgramPhase.PAUSED
                elif program.active:
                    program.phase = ProgramPhase.ACTING
                    program.last_completed = now
                else:
                    program.phase = ProgramPhase.PAUSED

                # Completion may reveal a larger resident context than the
                # request estimate.  Reconciliation above either expands its
                # owner envelope or demotes it at this safe boundary.
                self._enforce_budget(now)
                self._membership_invariants(now)

            self._prune(now)
            self._dispatch()
            self._arm_rotation_timer()

    async def release(self, key: str) -> None:
        async with self._lock:
            now = time.monotonic()
            program = self._programs.get(key)
            self._counters["releases"] += 1
            if program is None:
                return
            program.release_pending = True
            program.last_seen = now
            if not program.queued and not program.inflight:
                program.active = False
                program.phase = ProgramPhase.RELEASED
                self._programs.pop(key, None)
            self._dispatch()
            self._arm_rotation_timer()

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            self._prune_rotation_events(time.monotonic())
            phases = {phase.value: 0 for phase in ProgramPhase}
            for program in self._programs.values():
                phases[program.phase.value] += 1
            return {
                "mode": "stable-owner-cohort-v3.11-late-owner-exchange",
                "page_budget": WORKING_PAGE_BUDGET,
                "used_pages": self._used_pages(),
                "pinned_envelope_budget": PINNED_ENVELOPE_BUDGET,
                "pinned_envelope_pages": self._pin_envelope_used(),
                "pinned_actual_pages": self._pinned_actual_pages(),
                "pinned_resident_equivalent_pages": (
                    self._pinned_resident_equivalent_pages()
                ),
                "pinned_program_limit": MAX_PINNED_PROGRAMS,
                "borrower_base_page_budget": BORROWER_BASE_PAGE_BUDGET,
                "borrower_base_count": BORROWER_BASE_COUNT,
                "borrower_expanded_page_budget": (
                    BORROWER_EXPANDED_PAGE_BUDGET
                ),
                "borrower_expanded_count": BORROWER_EXPANDED_COUNT,
                "borrower_actual_pages": self._borrower_actual_pages(),
                "borrower_inflight": self._borrower_inflight_count(),
                "first_service_age_seconds": FIRST_SERVICE_AGE_SECONDS,
                "first_service_runway_max_pages": (
                    FIRST_SERVICE_RUNWAY_MAX_PAGES
                ),
                "pinned_programs": self._pinned_count(),
                "rotation_window_programs": len(self._rotation_events),
                "rotation_window_pages": sum(
                    pages for _, pages in self._rotation_events
                ),
                "programs": len(self._programs),
                "inflight": self._inflight_count(),
                "queued": self._queued_count(),
                "phases": phases,
                "counters": dict(self._counters),
            }


def _text_header_pair(pair: tuple[bytes, bytes]) -> tuple[str, str]:
    name, value = pair
    # aiohttp exposes raw octets and serializes surrogate-escaped strings back
    # to those octets.  Latin-1 decoding would turn a byte such as 0xe9 into a
    # Unicode code point and later change it to UTF-8 bytes c3 a9.
    return (
        name.decode("utf-8", "surrogateescape"),
        value.decode("utf-8", "surrogateescape"),
    )


def _forward_headers(
    raw_headers: Iterable[tuple[bytes, bytes]],
    *,
    request_side: bool,
) -> CIMultiDict[str]:
    """Retain duplicate end-to-end headers and remove hop-by-hop fields."""

    pairs = [_text_header_pair(pair) for pair in raw_headers]
    connection_tokens = {
        token.strip().lower()
        for name, value in pairs
        if name.lower() == "connection"
        for token in value.split(",")
        if token.strip()
    }
    blocked = set(HOP_BY_HOP) | connection_tokens
    if request_side:
        blocked.add("host")

    forwarded: CIMultiDict[str] = CIMultiDict()
    for name, value in pairs:
        if name.lower() not in blocked:
            forwarded.add(name, value)
    return forwarded


def _backend_target(request: web.Request) -> str:
    suffix = request.path.removeprefix("/v1")
    target = f"{BACKEND_URL}{suffix}"
    raw_query = request.rel_url.raw_query_string
    if raw_query:
        target = f"{target}?{raw_query}"
    return target


async def _open_upstream(
    session: ClientSession,
    request: web.Request,
    body: bytes,
):
    return await session.request(
        request.method,
        _backend_target(request),
        data=body if body else None,
        headers=_forward_headers(request.raw_headers, request_side=True),
        allow_redirects=False,
        auto_decompress=False,
        skip_auto_headers={
            "Accept",
            "Accept-Encoding",
            "Content-Type",
            "User-Agent",
        },
    )


async def proxy(request: web.Request) -> web.StreamResponse:
    """Forward one request exactly once after optional chat admission."""

    body = await request.read()
    session: ClientSession = request.app["upstream_session"]
    scheduler: ProgramScheduler = request.app["scheduler"]
    ticket: Ticket | None = None
    upstream = None
    downstream = None
    observer: ResponseObserver | None = None
    completed = False

    if request.path == "/v1/chat/completions":
        run_id = request.headers.get("X-Agent-Run-ID")
        ephemeral = run_id is None
        key = scheduler.new_ephemeral_key() if ephemeral else run_id
        ticket = await scheduler.offer(key, body, ephemeral=ephemeral)

    try:
        if ticket is not None:
            # Directly awaited in this handler: cancellation while queued can
            # never leave an orphan task that later opens an upstream request.
            await ticket.gate

        upstream = await _open_upstream(session, request, body)
        observer = ResponseObserver(upstream.headers.get("Content-Type"))
        response_headers = _forward_headers(upstream.raw_headers, request_side=False)
        downstream = web.StreamResponse(
            status=upstream.status,
            reason=upstream.reason,
            headers=response_headers,
        )
        await downstream.prepare(request)

        async for chunk in upstream.content.iter_any():
            observer.feed(chunk)
            await downstream.write(chunk)
        await downstream.write_eof()
        completed = True
        return downstream
    except asyncio.CancelledError:
        raise
    except (ConnectionResetError, BrokenPipeError):
        raise
    except ClientError as exc:
        if downstream is not None:
            raise ConnectionResetError("backend response stream ended early") from exc
        return web.json_response(
            {"error": {"type": "backend_unavailable", "message": str(exc)}},
            status=502,
        )
    finally:
        # Closing upstream precedes releasing the claim, so a replacement can
        # never overlap model work from a disconnected client.
        if upstream is not None:
            upstream.close()
        if ticket is not None:
            try:
                usage = observer.finish() if observer is not None else None
            except Exception:
                usage = None
            await scheduler.finish(
                ticket,
                usage=usage,
                response_bytes=observer.total_bytes if observer is not None else 0,
                first_response_at=observer.first_at if observer is not None else 0.0,
                completed=completed,
                response_status=upstream.status if upstream is not None else None,
            )


async def health(request: web.Request) -> web.Response:
    scheduler: ProgramScheduler = request.app["scheduler"]
    snapshot = await scheduler.snapshot()
    return web.json_response({"status": "ok", **snapshot})


async def release_program(request: web.Request) -> web.Response:
    body = await request.read()
    run_id = request.headers.get("X-Agent-Run-ID")
    if run_id is None:
        try:
            payload = json.loads(body)
            value = payload.get("program_id") if isinstance(payload, dict) else None
            if isinstance(value, str):
                run_id = value
        except Exception:
            pass
    if run_id is not None:
        scheduler: ProgramScheduler = request.app["scheduler"]
        await scheduler.release(run_id)
    return web.Response(status=204)


async def upstream_session(app: web.Application):
    connector = TCPConnector(limit=0)
    timeout = ClientTimeout(total=None, connect=30, sock_connect=30, sock_read=None)
    session = ClientSession(
        connector=connector,
        timeout=timeout,
        cookie_jar=DummyCookieJar(),
        auto_decompress=False,
        skip_auto_headers={"Accept", "Accept-Encoding", "User-Agent"},
    )
    app["upstream_session"] = session
    try:
        yield
    finally:
        await session.close()


def make_app() -> web.Application:
    app = web.Application(
        client_max_size=MAX_REQUEST_BYTES,
        handler_args={
            "auto_decompress": False,
            "handler_cancellation": True,
        },
    )
    app["scheduler"] = ProgramScheduler()
    app.cleanup_ctx.append(upstream_session)
    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", proxy)
    app.router.add_post("/v1/chat/completions", proxy)
    app.router.add_post("/programs/release", release_program)
    return app


if __name__ == "__main__":
    web.run_app(make_app(), host="0.0.0.0", port=PORT, access_log=None)
