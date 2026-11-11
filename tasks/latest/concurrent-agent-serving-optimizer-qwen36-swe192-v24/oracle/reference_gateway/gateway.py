#!/usr/bin/env python3
"""Byte-transparent Harbor Oracle using ThunderAgent program scheduling.

The HTTP data plane intentionally remains the task's small aiohttp proxy: each
accepted chat request forwards the original body and opaque headers exactly
once, and response chunks are relayed in order without decompression.  The
program state machine and admission decisions are provided by the extracted
ThunderAgent scheduler in ``scheduler.py``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import signal
from pathlib import Path
from typing import Any

from aiohttp import ClientConnectionError, ClientSession, ClientTimeout, TCPConnector, web

from scheduler import (
    HybridPageAccounting,
    ProgramAdmissionCapacityTimeout,
    ProgramReleased,
    ThunderScheduler,
)


BACKEND_URL = os.environ["BACKEND_URL"].rstrip("/")
PORT = int(os.environ.get("PORT", "9000"))

# vLLM's reported Qwen3.6-27B TP2 logical KV capacity under the frozen
# 30-GiB/GPU cache profile.  It is useful diagnostics but is not the additive
# admission capacity for this hybrid attention/Mamba model.
KV_CAPACITY_TOKENS = int(
    os.environ.get("THUNDER_KV_CAPACITY_TOKENS", "1887738")
)
# The runtime reports 1,253 GPU KV blocks.  Block zero is vLLM's reserved null
# block, leaving 1,252 allocatable page indices.  Every live sequence consumes
# three resident recurrent-state pages (one in each Mamba group), plus
# independently rounded 1,568-token full-attention pages.  In ``align`` mode
# vLLM may briefly hold a second Mamba page per group while copying state at an
# aligned boundary.  Although an uncached request initially allocates only one
# page per group, one OpenAI request spans internal engine iterations and can
# reach the two-page peak before returning.  The Oracle therefore reserves the
# peak for every REASONING/request-pending program and drops back to the three
# resident pages in ACTING; agent step_count is not an allocator-state signal.
KV_RAW_GPU_BLOCKS = int(os.environ.get("THUNDER_KV_RAW_GPU_BLOCKS", "1253"))
KV_RESERVED_NULL_BLOCKS = int(
    os.environ.get("THUNDER_KV_RESERVED_NULL_BLOCKS", "1")
)
ATTENTION_PAGE_TOKENS = int(
    os.environ.get("THUNDER_ATTENTION_PAGE_TOKENS", "1568")
)
MAMBA_GROUP_COUNT = int(os.environ.get("THUNDER_MAMBA_GROUP_COUNT", "3"))
MAMBA_RESIDENT_PAGES_PER_GROUP = int(
    os.environ.get("THUNDER_MAMBA_RESIDENT_PAGES_PER_GROUP", "1")
)
MAMBA_PEAK_PAGES_PER_GROUP = int(
    os.environ.get("THUNDER_MAMBA_PEAK_PAGES_PER_GROUP", "2")
)
MAX_INFLIGHT = int(os.environ.get("MAX_INFLIGHT", "192"))
SCHEDULER_INTERVAL_SEC = float(
    os.environ.get("THUNDER_SCHEDULER_INTERVAL_SEC", "5.0")
)
ACTING_DECAY_BASE = float(os.environ.get("THUNDER_ACTING_DECAY_BASE", "2.0"))
# ThunderAgent's 100-token per-program decode headroom remains token-denominated
# and is rounded together with each program's attention-token demand.
BUFFER_PER_PROGRAM = int(os.environ.get("THUNDER_BUFFER_PER_PROGRAM", "100"))
HIGH_WATERMARK = float(os.environ.get("THUNDER_HIGH_WATERMARK", "1.0"))
LOW_WATERMARK = float(os.environ.get("THUNDER_LOW_WATERMARK", "1.0"))
FORCE_RESUME_TIMEOUT_SEC = float(
    os.environ.get("THUNDER_FORCE_RESUME_TIMEOUT_SEC", "1800")
)
# Harbor's frozen finite cohort has an explicit p99 admission-wait bound that
# upstream throughput-oriented shortest-first selection does not provide.  A
# genuinely blocked request receives a bounded age boost after 220 seconds.
# V5 retains the single-oldest full-page reservation from V4, pausing only idle
# ACTING working sets and preserving in-flight REASONING.  The first twelve
# request ordinals retain V4's finite-cohort service-lag order; deeper requests
# restore upstream-style total-token SJF after the aged quota is serviced.
FAIR_WAIT_THRESHOLD_SEC = float(
    os.environ.get("THUNDER_FAIR_WAIT_THRESHOLD_SEC", "220")
)
AGED_RESUME_QUOTA_PER_TICK = int(
    os.environ.get("THUNDER_AGED_RESUME_QUOTA_PER_TICK", "4")
)
# Keep one request ordinal of look-ahead through the public-compatible tier.
FAIR_SERVICE_LAG_STEPS = int(
    os.environ.get("THUNDER_FAIR_SERVICE_LAG_STEPS", "1")
)
FAIR_SERVICE_UNTIL_STEP = int(
    os.environ.get("THUNDER_FAIR_SERVICE_UNTIL_STEP", "12")
)
THUNDERAGENT_SOURCE_COMMIT = "7ddc8610270e56d3b109eed8796b3a4360fc67c9"


def reference_gateway_tree_sha256(root: Path = Path(__file__).resolve().parent) -> str:
    records: list[bytes] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts:
            continue
        if path.is_symlink() or (not path.is_dir() and not path.is_file()):
            raise RuntimeError(f"invalid Oracle tree entry: {relative.as_posix()}")
        if path.is_dir():
            continue
        payload = path.read_bytes()
        records.append(
            json.dumps(
                {
                    "path": relative.as_posix(),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size": len(payload),
                },
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        )
    if not records:
        raise RuntimeError("Oracle reference tree is empty")
    digest = hashlib.sha256(b"reference-gateway-tree-v1\n")
    for record in records:
        digest.update(record)
        digest.update(b"\n")
    return digest.hexdigest()


REFERENCE_GATEWAY_TREE_SHA256_AT_START = reference_gateway_tree_sha256()

if not 1 <= MAX_INFLIGHT <= 192:
    raise ValueError("MAX_INFLIGHT must be between 1 and the frozen max-num-seqs value 192")

HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "content-length",
    "host",
}
OBSERVER_LIMIT_BYTES = 16 * 1024**2


def filtered_headers(headers: Any) -> list[tuple[str, str]]:
    """Remove hop-by-hop fields while retaining order and duplicate values."""

    connection_values = (
        headers.getall("Connection", [])
        if hasattr(headers, "getall")
        else [headers.get("Connection", "")]
    )
    dynamic_hop_headers = {
        token.strip().lower()
        for value in connection_values
        for token in str(value).split(",")
        if token.strip()
    }
    excluded = HOP_HEADERS | dynamic_hop_headers
    return [
        (name, value)
        for name, value in headers.items()
        if name.lower() not in excluded
    ]


def upstream_url(suffix: str, request: web.Request) -> str:
    target = f"{BACKEND_URL}/{suffix.lstrip('/')}"
    # raw_path preserves the client's query octets rather than parsing and
    # re-encoding them through a dict.
    _, separator, raw_query = request.raw_path.partition("?")
    return f"{target}?{raw_query}" if separator else target


def program_id(request: web.Request) -> str:
    """Use the task's opaque workflow ID as the ThunderAgent program key."""

    return str(
        request.headers.get("X-Agent-Run-ID")
        or request.headers.get("X-Session-ID")
        or "__anonymous__"
    )


def request_context_chars(body: bytes) -> int:
    """Match upstream JSON-character estimation without changing body bytes."""

    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return len(body)
    try:
        return len(json.dumps(payload, ensure_ascii=False))
    except (TypeError, ValueError):
        return len(body)


def request_is_streaming(body: bytes) -> bool:
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("stream") is True


def extract_usage(payload: Any) -> tuple[int | None, int | None]:
    if not isinstance(payload, dict) or not isinstance(payload.get("usage"), dict):
        return None, None
    usage = payload["usage"]

    def positive_integer(name: str) -> int | None:
        value = usage.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        integer = int(value)
        return integer if integer > 0 else None

    return positive_integer("total_tokens"), positive_integer("prompt_tokens")


class UsageObserver:
    """Bounded, read-only response observer; it never changes relayed bytes."""

    def __init__(self, streaming: bool) -> None:
        self.streaming = streaming
        self.line_buffer = bytearray()
        self.body_buffer = bytearray()
        self.total_tokens: int | None = None
        self.prompt_tokens: int | None = None
        self.stream_events_pending = 0

    def _record(self, payload: Any) -> None:
        total, prompt = extract_usage(payload)
        if total is not None:
            self.total_tokens = total
        if prompt is not None:
            self.prompt_tokens = prompt

    def feed(self, chunk: bytes) -> int:
        streamed_delta = 0
        if self.streaming:
            if len(self.line_buffer) + len(chunk) > OBSERVER_LIMIT_BYTES:
                self.line_buffer.clear()
                return 0
            self.line_buffer.extend(chunk)
            while b"\n" in self.line_buffer:
                raw, _, remainder = self.line_buffer.partition(b"\n")
                self.line_buffer[:] = remainder
                line = raw.rstrip(b"\r")
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if not data or data == b"[DONE]":
                    continue
                try:
                    self._record(json.loads(data))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                self.stream_events_pending += 1
                if self.stream_events_pending >= 20:
                    streamed_delta += self.stream_events_pending
                    self.stream_events_pending = 0
            return streamed_delta
        if len(self.body_buffer) + len(chunk) <= OBSERVER_LIMIT_BYTES:
            self.body_buffer.extend(chunk)
        return 0

    def finish(self) -> None:
        if self.streaming or not self.body_buffer:
            return
        try:
            self._record(json.loads(self.body_buffer))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return


async def health(request: web.Request) -> web.Response:
    scheduler: ThunderScheduler = request.app["scheduler"]
    snapshot = await scheduler.snapshot()
    return web.json_response(
        {
            "status": "ok",
            "oracle": "thunderagent-extracted",
            "thunderagent_source_commit": THUNDERAGENT_SOURCE_COMMIT,
            "reference_gateway_tree_sha256_at_start": (
                REFERENCE_GATEWAY_TREE_SHA256_AT_START
            ),
            "transport": {
                "max_inflight": MAX_INFLIGHT,
                "inbound_auto_decompress": False,
                "handler_cancellation": True,
            },
            "scheduler": snapshot,
        }
    )


async def models(request: web.Request) -> web.Response:
    try:
        async with request.app["control_client"].get(
            upstream_url("models", request),
            headers=filtered_headers(request.headers),
        ) as upstream:
            return web.Response(
                status=upstream.status,
                reason=upstream.reason,
                body=await upstream.read(),
                headers=filtered_headers(upstream.headers),
            )
    except (OSError, ClientConnectionError, asyncio.TimeoutError):
        return web.json_response(
            {"error": {"message": "backend unavailable", "type": "backend_error"}},
            status=502,
        )


async def programs(request: web.Request) -> web.Response:
    scheduler: ThunderScheduler = request.app["scheduler"]
    return web.json_response(await scheduler.snapshot())


async def release_program(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return web.json_response({"error": "invalid_json"}, status=400)
    value = payload.get("program_id") if isinstance(payload, dict) else None
    if not isinstance(value, str) or not value or len(value) > 256:
        return web.json_response({"error": "invalid_program_id"}, status=400)
    scheduler: ThunderScheduler = request.app["scheduler"]
    found, released = await scheduler.release_program(value)
    return web.json_response(
        {
            "program_id": value,
            "accepted": found,
            "released": released,
            "deferred": found and not released,
        }
    )


async def completions(request: web.Request) -> web.StreamResponse:
    body = await request.read()
    scheduler: ThunderScheduler = request.app["scheduler"]
    program = await scheduler.get_or_create_program(program_id(request))
    observer = UsageObserver(request_is_streaming(body))
    admitted = False
    response_started = False

    async with program.request_lock:
        try:
            await scheduler.begin_request(program, request_context_chars(body))
            admitted = True
        except asyncio.CancelledError:
            await scheduler.cancel_waiting_request(program)
            raise
        except ProgramAdmissionCapacityTimeout:
            await scheduler.cancel_waiting_request(program)
            return web.json_response(
                {
                    "error": {
                        "message": (
                            "request admission timed out while KV capacity "
                            "remained unavailable"
                        ),
                        "type": "admission_capacity_timeout",
                        "code": "admission_capacity_timeout",
                    }
                },
                status=503,
            )
        except ProgramReleased:
            return web.json_response(
                {"error": {"message": "program lifecycle ended", "type": "cancelled"}},
                status=409,
            )

        try:
            async with request.app["backend_slots"]:
                async with request.app["data_client"].post(
                    upstream_url("chat/completions", request),
                    data=body,
                    headers=filtered_headers(request.headers),
                ) as upstream:
                    downstream = web.StreamResponse(
                        status=upstream.status,
                        reason=upstream.reason,
                        headers=filtered_headers(upstream.headers),
                    )
                    await downstream.prepare(request)
                    response_started = True
                    async for chunk in upstream.content.iter_any():
                        stream_delta = observer.feed(chunk)
                        if stream_delta:
                            await scheduler.update_streaming_tokens(
                                program, stream_delta
                            )
                        await downstream.write(chunk)
                    observer.finish()
                    await downstream.write_eof()
                    return downstream
        except (ConnectionResetError, asyncio.CancelledError):
            raise
        except (OSError, ClientConnectionError, asyncio.TimeoutError):
            if response_started:
                raise
            return web.json_response(
                {
                    "error": {
                        "message": "backend unavailable",
                        "type": "backend_error",
                    }
                },
                status=502,
            )
        finally:
            if admitted:
                await scheduler.finish_request(
                    program,
                    total_tokens=observer.total_tokens,
                    prompt_tokens=observer.prompt_tokens,
                )


async def on_startup(app: web.Application) -> None:
    # aiohttp otherwise imposes a hidden 100-connection queue.  ThunderAgent's
    # token scheduler owns admission; this connector only preserves the frozen
    # vLLM max-num-seqs transport ceiling.
    app["data_client"] = ClientSession(
        connector=TCPConnector(limit=MAX_INFLIGHT, limit_per_host=0),
        timeout=ClientTimeout(total=None, connect=30, sock_read=None),
        auto_decompress=False,
    )
    app["control_client"] = ClientSession(
        connector=TCPConnector(limit=8, limit_per_host=0),
        timeout=ClientTimeout(total=30, connect=10, sock_read=30),
        auto_decompress=False,
    )
    await app["scheduler"].start()


async def on_cleanup(app: web.Application) -> None:
    await app["scheduler"].stop()
    await app["data_client"].close()
    await app["control_client"].close()


def make_app() -> web.Application:
    hybrid_page_accounting = HybridPageAccounting(
        raw_gpu_blocks=KV_RAW_GPU_BLOCKS,
        reserved_null_blocks=KV_RESERVED_NULL_BLOCKS,
        attention_page_tokens=ATTENTION_PAGE_TOKENS,
        mamba_group_count=MAMBA_GROUP_COUNT,
        mamba_resident_pages_per_group=MAMBA_RESIDENT_PAGES_PER_GROUP,
        mamba_peak_pages_per_group=MAMBA_PEAK_PAGES_PER_GROUP,
        reported_logical_capacity_tokens=KV_CAPACITY_TOKENS,
    )
    scheduler = ThunderScheduler(
        capacity_tokens=KV_CAPACITY_TOKENS,
        scheduler_interval_sec=SCHEDULER_INTERVAL_SEC,
        acting_decay_base=ACTING_DECAY_BASE,
        buffer_per_program=BUFFER_PER_PROGRAM,
        hybrid_page_accounting=hybrid_page_accounting,
        high_watermark=HIGH_WATERMARK,
        low_watermark=LOW_WATERMARK,
        force_resume_timeout_sec=FORCE_RESUME_TIMEOUT_SEC,
        fair_wait_threshold_sec=FAIR_WAIT_THRESHOLD_SEC,
        aged_resume_quota_per_tick=AGED_RESUME_QUOTA_PER_TICK,
        fair_service_lag_steps=FAIR_SERVICE_LAG_STEPS,
        fair_service_until_step=FAIR_SERVICE_UNTIL_STEP,
    )
    app = web.Application(client_max_size=128 * 1024**2)
    app["scheduler"] = scheduler
    app["backend_slots"] = asyncio.Semaphore(MAX_INFLIGHT)
    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/chat/completions", completions)
    app.router.add_get("/programs", programs)
    app.router.add_post("/programs/release", release_program)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


async def serve() -> None:
    """Run aiohttp without inbound decompression and cancel on disconnect."""

    runner = web.AppRunner(
        make_app(),
        access_log=None,
        auto_decompress=False,
        handler_cancellation=True,
    )
    await runner.setup()
    site = web.TCPSite(runner, host="0.0.0.0", port=PORT)
    await site.start()
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stopped.set)
        except NotImplementedError:
            pass
    try:
        await stopped.wait()
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(serve())
