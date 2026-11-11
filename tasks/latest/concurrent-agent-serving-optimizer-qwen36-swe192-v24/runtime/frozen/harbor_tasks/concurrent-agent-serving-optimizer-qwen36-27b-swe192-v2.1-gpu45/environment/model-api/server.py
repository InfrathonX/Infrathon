#!/usr/bin/env python3
"""Narrow, audited OpenAI-compatible proxy in front of the fixed vLLM."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hashlib
import hmac
import json
import os
import re
import secrets
import time
from pathlib import Path
from typing import Any

from aiohttp import ClientConnectionError, ClientSession, ClientTimeout, TCPConnector, web

from transport import configured_base_url, upstream_url


UPSTREAM_BASE_URL = configured_base_url()
TOKEN_FILE = Path(os.environ.get("ADMIN_TOKEN_FILE", "/run/kvbench-admin/token"))
AUDIT_FILE = Path(os.environ.get("AUDIT_FILE", "/run/kvbench-audit/requests.jsonl"))
BACKEND_CONTRACT_FILE = Path(
    os.environ.get(
        "BACKEND_CONTRACT_FILE", "/opt/model-api/backend-serving-contract.json"
    )
)
EXPECTED_SERVED_MODEL_NAME = "target-model"
# The digest of ``backend-serving-contract.json`` as shipped in this image.
# Revision v2.5 states the backend requirement as a capability predicate
# instead of an "NVIDIA H100 NVL" name whitelist; the previous revision is
# kept beside it as ``backend-serving-contract-v1.json`` so historical Direct
# evidence stays attributable.
BACKEND_SERVING_CONTRACT_SHA256 = (
    "1cf6d2e6ff7af8448d6b7735590621d083d6a5071252bc60b40c9756ffe328c4"
)
LEGACY_PROFILE_PROVENANCE = {
    "profile_id": "qwen3.6-27b-6a9e13bd-h100-nvl-tp2-kv30g-v2",
    "profile_sha256": (
        "7289ed315324b9b5165d4d680efb93004c59e0d34f0f98ef9d85406ed5a9153a"
    ),
}
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
PRIVATE_FORWARD_HEADERS = {
    "x-kvbench-request-id",
    "x-kvbench-auth",
    "x-kvbench-audit-epoch",
    "x-kvbench-body-sha256",
    "x-agent-run-id",
    "x-kvbench-profile",
    "x-kvbench-leg",
    "x-kvbench-cache-namespace",
    "x-kvbench-transport-auth",
}
TRANSPORT_AUTH_HEADER = "X-KVBench-Transport-Auth"
TRANSPORT_AUTH_CONTEXT = b"kvbench-model-api-transport-v1"
BENCHMARK_AUTH_CONTEXT = b"kvbench-data-request-v2"
METRIC_NAME_PREFIXES = (
    "vllm:",
    "vllm_",
    "process_resident_memory_bytes",
    "process_cpu_seconds_total",
)
PROMETHEUS_SAMPLE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>.*)\})?\s+(?P<value>[-+0-9.eE]+|NaN|Inf|-Inf)$"
)


def token() -> bytes:
    return TOKEN_FILE.read_text().strip().encode()


def load_backend_serving_contract() -> dict[str, Any]:
    try:
        payload = json.loads(BACKEND_CONTRACT_FILE.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("backend serving contract is unavailable or invalid") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("backend serving contract must be a JSON object")
    return payload


def backend_serving_contract_sha256(contract: dict[str, Any]) -> str:
    canonical = json.dumps(
        contract,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def backend_contract_valid(contract: dict[str, Any]) -> bool:
    return secrets.compare_digest(
        backend_serving_contract_sha256(contract), BACKEND_SERVING_CONTRACT_SHA256
    )


def backend_contract_payload(contract: dict[str, Any]) -> dict[str, Any]:
    """Return stable serving provenance without exposing the raw backend URL."""

    model = contract["model"]
    runtime = contract["runtime"]
    return {
        "schema_version": 1,
        "state": "ready",
        "kind": "backend-serving-contract",
        # These two fields are retained for the v2.1 controller response
        # schema.  The clean serving-contract digest deliberately does not
        # depend on the legacy profile file or its host GPU placement.
        "profile_id": LEGACY_PROFILE_PROVENANCE["profile_id"],
        "profile_sha256": LEGACY_PROFILE_PROVENANCE["profile_sha256"],
        "backend_serving_contract_sha256": BACKEND_SERVING_CONTRACT_SHA256,
        "model": {
            "revision": model["revision"],
            "served_model_name": EXPECTED_SERVED_MODEL_NAME,
        },
        "runtime": {
            "runtime_image_id": runtime["runtime_image_id"],
            "tensor_parallel_size": runtime["tensor_parallel_size"],
            "kv_cache_dtype": runtime["kv_cache_dtype"],
        },
    }


def derive_transport_credential(admin_token: bytes) -> str:
    if not admin_token:
        raise ValueError("admin token is empty")
    return hmac.new(
        admin_token,
        TRANSPORT_AUTH_CONTEXT,
        hashlib.sha256,
    ).hexdigest()


def admin_allowed(request: web.Request) -> bool:
    supplied = request.headers.get("Authorization", "")
    expected = f"Bearer {token().decode()}"
    return secrets.compare_digest(supplied, expected)


def transport_allowed(request: web.Request) -> bool:
    supplied = request.headers.get(TRANSPORT_AUTH_HEADER, "")
    return secrets.compare_digest(supplied, request.app["transport_credential"])


class DataPlaneGate:
    """Concurrent-reader/exclusive-reset gate for the trusted data plane."""

    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0

    async def enter_request(self) -> None:
        async with self._condition:
            while self._writer or self._waiting_writers:
                await self._condition.wait()
            self._readers += 1

    async def exit_request(self) -> None:
        async with self._condition:
            if self._readers <= 0:
                raise RuntimeError("data-plane gate reader underflow")
            self._readers -= 1
            if self._readers == 0:
                self._condition.notify_all()

    @asynccontextmanager
    async def exclusive(self):
        async with self._condition:
            self._waiting_writers += 1
            try:
                while self._writer or self._readers:
                    await self._condition.wait()
                self._writer = True
            finally:
                self._waiting_writers -= 1
        try:
            yield
        finally:
            async with self._condition:
                self._writer = False
                self._condition.notify_all()


def benchmark_auth_message(
    *,
    audit_epoch: int,
    method: str,
    path: str,
    request_id: str,
    run_id: str,
    profile: str,
    leg_id: str,
    cache_namespace: str,
    body_digest: str,
) -> bytes:
    value = {
        "audit_epoch": audit_epoch,
        "body_digest": body_digest,
        "cache_namespace": cache_namespace,
        "leg_id": leg_id,
        "method": method.upper(),
        "path": path,
        "profile": profile,
        "request_id": request_id,
        "run_id": run_id,
    }
    canonical = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return BENCHMARK_AUTH_CONTEXT + b"\n" + canonical


def benchmark_auth_fields(request: web.Request, body_digest: str) -> dict[str, Any] | None:
    raw_epoch = request.headers.get("X-KVBench-Audit-Epoch", "")
    try:
        audit_epoch = int(raw_epoch)
    except ValueError:
        return None
    if str(audit_epoch) != raw_epoch or audit_epoch < 0:
        return None
    fields: dict[str, Any] = {
        "audit_epoch": audit_epoch,
        "method": request.method,
        "path": request.path,
        "request_id": request.headers.get("X-KVBench-Request-ID", ""),
        "run_id": request.headers.get("X-Agent-Run-ID", ""),
        "profile": request.headers.get("X-KVBench-Profile", ""),
        "leg_id": request.headers.get("X-KVBench-Leg", ""),
        "cache_namespace": request.headers.get("X-KVBench-Cache-Namespace", ""),
        "body_digest": body_digest,
    }
    if not all(
        isinstance(fields[name], str) and 0 < len(fields[name]) <= 256
        for name in ("request_id", "run_id", "profile", "leg_id", "cache_namespace")
    ):
        return None
    if not re.fullmatch(r"[0-9a-f]{64}", body_digest):
        return None
    return fields


@web.middleware
async def require_data_plane_transport(
    request: web.Request,
    handler: Any,
) -> web.StreamResponse:
    # Authenticate before a data-plane handler can read a request body, touch
    # the upstream transport, increment active state, or append an audit record.
    if request.path == "/v1/metrics":
        # ThunderAgent's VLLMMetricsClient needs the backend's live Prometheus
        # cache view.  The loopback bridge still proves that this request came
        # through the task-local transport, but metrics reads are not model
        # generations and therefore do not consume one-time benchmark HMACs.
        if not transport_allowed(request):
            rejection = web.HTTPNotFound()
            rejection.force_close()
            raise rejection
        return await handler(request)
    if request.path.startswith("/v1/"):
        if not transport_allowed(request):
            rejection = web.HTTPNotFound()
            rejection.force_close()
            raise rejection
        # The UID-900 loopback bridge deliberately supplies transport auth for
        # candidate traffic, so transport auth alone is not authorization to
        # mutate vLLM prefix-cache state.  Require the controller's per-request
        # body HMAC before the handler can touch upstream/active/audit state.
        # The claimed digest is itself signed, which lets us reject arbitrary
        # candidate traffic before buffering its body. Only a valid, unique
        # controller request is read, then its actual bytes must match.
        claimed_body_digest = request.headers.get("X-KVBench-Body-SHA256", "")
        fields = benchmark_auth_fields(request, claimed_body_digest)
        gate = request.app["data_plane_gate"]
        await gate.enter_request()
        try:
            async with request.app["authorization_lock"]:
                request_id = fields.get("request_id") if fields is not None else None
                valid = (
                    fields is not None
                    and fields["audit_epoch"] == request.app["audit_epoch"]
                    and request.app["audit_epoch_sealed"] is False
                    and benchmark_auth_valid(
                        fields,
                        request.headers.get("X-KVBench-Auth"),
                    )
                    and request_id not in request.app["seen_request_ids"]
                )
                if valid:
                    request.app["seen_request_ids"].add(request_id)
            if not valid:
                rejection = web.HTTPNotFound()
                rejection.force_close()
                raise rejection
            if request.method == "GET":
                actual_body_digest = hashlib.sha256(b"").hexdigest()
            else:
                actual_body_digest = hashlib.sha256(await request.read()).hexdigest()
            if not secrets.compare_digest(actual_body_digest, claimed_body_digest):
                rejection = web.HTTPNotFound()
                rejection.force_close()
                raise rejection
            admitted_task = asyncio.current_task()
            async with request.app["authorization_lock"]:
                # The public cutoff seals this epoch under the same lock.  A
                # request that passed the header check but was still buffering
                # its body at the boundary must not reach the upstream later.
                if (
                    request.app["audit_epoch_sealed"] is True
                    or fields["audit_epoch"] != request.app["audit_epoch"]
                ):
                    rejection = web.HTTPNotFound()
                    rejection.force_close()
                    raise rejection
                request.app["admitted_request_count"] += 1
                if admitted_task is not None:
                    request.app["admitted_tasks"].add(admitted_task)
            request["benchmark_auth_valid"] = True
            try:
                return await handler(request)
            finally:
                if admitted_task is not None:
                    async with request.app["authorization_lock"]:
                        request.app["admitted_tasks"].discard(admitted_task)
        finally:
            await gate.exit_request()
    return await handler(request)


def filtered_headers(headers: Any, *, request_headers: bool = False) -> dict[str, str]:
    result = {}
    for name, value in headers.items():
        lowered = name.lower()
        if lowered in HOP_HEADERS:
            continue
        if request_headers and lowered in PRIVATE_FORWARD_HEADERS:
            continue
        result[name] = value
    return result


def benchmark_auth_valid(fields: dict[str, Any], auth: str | None) -> bool:
    if not auth:
        return False
    expected = hmac.new(
        token(),
        benchmark_auth_message(**fields),
        hashlib.sha256,
    ).hexdigest()
    return secrets.compare_digest(auth, expected)


async def append_audit(app: web.Application, record: dict[str, Any]) -> None:
    line = json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
    async with app["audit_lock"]:
        with AUDIT_FILE.open("a") as handle:
            handle.write(line)
        # Keep a lightweight trusted completion counter under the exact same
        # lock/commit boundary as the full append-only ledger.  The measured
        # candidate watchdog reads this through the existing metrics poll and
        # therefore never downloads a growing JSONL audit every second.
        app["audit_count"] += 1


def observe_sse(buffer: bytearray, chunk: bytes, state: dict[str, Any]) -> None:
    """Extract timing/usage diagnostics without changing forwarded bytes."""
    buffer.extend(chunk)
    while b"\n" in buffer:
        raw_line, _, remainder = buffer.partition(b"\n")
        buffer[:] = remainder
        line = raw_line.rstrip(b"\r")
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if not data:
            continue
        if data == b"[DONE]":
            state["sse_done"] = True
            continue
        try:
            event = json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        choices = event.get("choices") or []
        if choices:
            delta = choices[0].get("delta") or {}
            if state["first_token_unix"] is None and (delta.get("content") or delta.get("reasoning_content")):
                state["first_token_unix"] = time.time()
        usage = event.get("usage")
        if isinstance(usage, dict):
            state["usage"] = {
                key: int(value)
                for key, value in usage.items()
                if key in {"prompt_tokens", "completion_tokens", "total_tokens"} and isinstance(value, (int, float))
            }


def parse_metric_samples(raw: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for line in raw.splitlines():
        if not line or line.startswith("#"):
            continue
        match = PROMETHEUS_SAMPLE.match(line)
        if not match or not match.group("name").startswith(METRIC_NAME_PREFIXES):
            continue
        value_text = match.group("value")
        try:
            value = float(value_text)
        except ValueError:
            continue
        samples.append(
            {
                "name": match.group("name"),
                "labels": match.group("labels") or "",
                "value": value,
            }
        )
    samples.sort(key=lambda item: (item["name"], item["labels"]))
    return samples


async def upstream_models_ready(app: web.Application) -> bool:
    """Require the configured origin to advertise the frozen model name."""

    try:
        async with app["control_client"].get(
            upstream_url("/v1/models", base_url=UPSTREAM_BASE_URL),
            allow_redirects=False,
        ) as response:
            raw = await response.read()
            if response.status != 200 or len(raw) > 1024 * 1024:
                return False
        payload = json.loads(raw)
    except (
        OSError,
        UnicodeDecodeError,
        asyncio.TimeoutError,
        ClientConnectionError,
        json.JSONDecodeError,
    ):
        return False
    if not isinstance(payload, dict) or payload.get("object") != "list":
        return False
    models = payload.get("data")
    return isinstance(models, list) and any(
        isinstance(model, dict) and model.get("id") == EXPECTED_SERVED_MODEL_NAME
        for model in models
    )


async def upstream_metrics(app: web.Application) -> tuple[bool, list[dict[str, Any]]]:
    try:
        async with app["control_client"].get(
            upstream_url("/metrics", base_url=UPSTREAM_BASE_URL),
            allow_redirects=False,
        ) as response:
            raw = await response.read()
            if response.status != 200:
                return False, []
    except (OSError, asyncio.TimeoutError, ClientConnectionError):
        return False, []
    return True, parse_metric_samples(raw.decode("utf-8", errors="replace"))


async def scheduler_metrics(request: web.Request) -> web.StreamResponse:
    """Expose the fixed vLLM's raw, read-only metrics to online schedulers."""

    try:
        async with request.app["control_client"].get(
            upstream_url("/metrics", base_url=UPSTREAM_BASE_URL),
            allow_redirects=False,
        ) as upstream:
            body = await upstream.read()
            return web.Response(
                status=upstream.status,
                body=body,
                headers=filtered_headers(upstream.headers),
            )
    except (OSError, asyncio.TimeoutError, ClientConnectionError):
        return web.json_response(
            {"error": "upstream_metrics_unavailable"},
            status=503,
        )


def scheduler_gauge(samples: list[dict[str, Any]], *suffixes: str) -> float | None:
    values = []
    for sample in samples:
        name = str(sample.get("name", ""))
        normalized = name.removeprefix("vllm:").removeprefix("vllm_")
        value = sample.get("value")
        if normalized in suffixes and isinstance(value, (int, float)):
            values.append(float(value))
    return max(values) if values else None


async def health(request: web.Request) -> web.Response:
    try:
        async with request.app["control_client"].get(
            upstream_url("/health", base_url=UPSTREAM_BASE_URL),
            allow_redirects=False,
        ) as response:
            upstream_ok = response.status == 200
            await response.read()
    except (OSError, asyncio.TimeoutError, ClientConnectionError):
        upstream_ok = False
    model_ok = await upstream_models_ready(request.app)
    contract_ok = backend_contract_valid(request.app["backend_serving_contract"])
    status = 200 if upstream_ok and model_ok and contract_ok else 503
    return web.json_response(
        {
            "status": "ok" if status == 200 else "starting",
            "upstream": upstream_ok,
            "served_model": model_ok,
            "backend_contract": contract_ok,
        },
        status=status,
    )


async def proxy_get(request: web.Request) -> web.StreamResponse:
    started = time.time()
    empty_digest = hashlib.sha256(b"").hexdigest()
    request_id = request.headers.get("X-KVBench-Request-ID")
    response_digest = hashlib.sha256()
    status = 599
    body = b""
    error: str | None = None
    app = request.app
    async with app["active_lock"]:
        app["active"] += 1
    try:
        async with app["data_client"].get(
            upstream_url(request.path_qs, base_url=UPSTREAM_BASE_URL),
            headers=filtered_headers(request.headers, request_headers=True),
            allow_redirects=False,
        ) as upstream:
            status = upstream.status
            body = await upstream.read()
            response_digest.update(body)
            return web.Response(status=status, body=body, headers=filtered_headers(upstream.headers))
    except (OSError, asyncio.TimeoutError, ClientConnectionError) as exc:
        error = type(exc).__name__
        status = 502
        body = json.dumps(
            {"error": {"message": "fixed backend unavailable", "type": "backend_error"}},
            separators=(",", ":"),
        ).encode()
        response_digest.update(body)
        return web.Response(
            status=status,
            body=body,
            content_type="application/json",
        )
    finally:
        ended = time.time()
        try:
            await append_audit(
                app,
                {
                    "schema_version": 1,
                    "audit_epoch": app["audit_epoch"],
                    "method": "GET",
                    "request_id": request_id,
                    "run_id": request.headers.get("X-Agent-Run-ID"),
                    "benchmark_auth_valid": request.get("benchmark_auth_valid") is True,
                    "canonical_body_digest": empty_digest,
                    "response_digest": response_digest.hexdigest(),
                    "response_bytes": len(body),
                    "status": status,
                    "started_unix": started,
                    "ended_unix": ended,
                    "duration_sec": ended - started,
                    "profile": request.headers.get("X-KVBench-Profile"),
                    "leg_id": request.headers.get("X-KVBench-Leg"),
                    "cache_namespace": request.headers.get("X-KVBench-Cache-Namespace"),
                    "client_disconnected": False,
                    "error": error,
                },
            )
        finally:
            async with app["active_lock"]:
                app["active"] -= 1


async def chat_completions(request: web.Request) -> web.StreamResponse:
    started = time.time()
    body = await request.read()
    body_digest = hashlib.sha256(body).hexdigest()
    request_id = request.headers.get("X-KVBench-Request-ID")
    run_id = request.headers.get("X-Agent-Run-ID")
    auth_valid = request.get("benchmark_auth_valid") is True
    response_digest = hashlib.sha256()
    response_bytes = 0
    status = 599
    disconnected = False
    error: str | None = None
    upstream_headers_unix: float | None = None
    diagnostic: dict[str, Any] = {"first_token_unix": None, "usage": {}, "sse_done": False}
    sse_buffer = bytearray()
    response_buffer = bytearray()
    try:
        request_payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        request_payload = {}
    app = request.app
    handler_task = asyncio.current_task()
    async with app["active_lock"]:
        app["active"] += 1
        if handler_task is not None:
            app["active_tasks"].add(handler_task)
    try:
        async with app["data_client"].post(
            upstream_url("/v1/chat/completions", base_url=UPSTREAM_BASE_URL),
            data=body,
            headers=filtered_headers(request.headers, request_headers=True),
            allow_redirects=False,
        ) as upstream:
            upstream_headers_unix = time.time()
            status = upstream.status
            downstream = web.StreamResponse(status=status, headers=filtered_headers(upstream.headers))
            await downstream.prepare(request)
            try:
                async for chunk in upstream.content.iter_any():
                    response_digest.update(chunk)
                    response_bytes += len(chunk)
                    observe_sse(sse_buffer, chunk, diagnostic)
                    if request_payload.get("stream") is False and len(response_buffer) < 16 * 1024**2:
                        response_buffer.extend(chunk)
                    await downstream.write(chunk)
                if request_payload.get("stream") is False:
                    try:
                        response_payload = json.loads(response_buffer)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        response_payload = {}
                    usage = response_payload.get("usage")
                    if isinstance(usage, dict):
                        diagnostic["usage"] = {
                            key: int(value)
                            for key, value in usage.items()
                            if key in {"prompt_tokens", "completion_tokens", "total_tokens"}
                            and isinstance(value, (int, float))
                        }
                    if response_payload.get("choices"):
                        diagnostic["first_token_unix"] = upstream_headers_unix
                await downstream.write_eof()
            except (ConnectionResetError, asyncio.CancelledError):
                disconnected = True
                raise
            return downstream
    except asyncio.CancelledError:
        error = "client_cancelled"
        raise
    except (OSError, asyncio.TimeoutError, ClientConnectionError) as exc:
        error = type(exc).__name__
        if not request.transport or request.transport.is_closing():
            disconnected = True
            raise
        status = 502
        error_body = json.dumps(
            {"error": {"message": "fixed backend unavailable", "type": "backend_error"}},
            separators=(",", ":"),
        ).encode()
        response_digest.update(error_body)
        response_bytes += len(error_body)
        return web.Response(status=status, body=error_body, content_type="application/json")
    finally:
        ended = time.time()
        # Quiescence is a transaction boundary used before every audit/cache
        # reset.  Keep the request active until its durable audit record is
        # appended so a reset can never race between active-- and append.
        try:
            await append_audit(
                app,
                {
                    "schema_version": 1,
                    "audit_epoch": app["audit_epoch"],
                    "method": "POST",
                    "request_id": request_id,
                    "run_id": run_id,
                    "benchmark_auth_valid": auth_valid,
                    "canonical_body_digest": body_digest,
                    "response_digest": response_digest.hexdigest(),
                    "response_bytes": response_bytes,
                    "status": status,
                    "started_unix": started,
                    "ended_unix": ended,
                    "duration_sec": ended - started,
                    "upstream_headers_unix": upstream_headers_unix,
                    "first_token_unix": diagnostic["first_token_unix"],
                    "prompt_tokens": diagnostic["usage"].get("prompt_tokens"),
                    "completion_tokens": diagnostic["usage"].get("completion_tokens"),
                    "total_tokens": diagnostic["usage"].get("total_tokens"),
                    "stream": request_payload.get("stream"),
                    "sse_done": diagnostic["sse_done"],
                    "requested_max_tokens": request_payload.get(
                        "max_tokens", request_payload.get("max_completion_tokens")
                    ),
                    "profile": request.headers.get("X-KVBench-Profile"),
                    "leg_id": request.headers.get("X-KVBench-Leg"),
                    "cache_namespace": request.headers.get("X-KVBench-Cache-Namespace"),
                    "client_disconnected": disconnected,
                    "error": error,
                },
            )
        finally:
            async with app["active_lock"]:
                app["active"] -= 1
                if handler_task is not None:
                    app["active_tasks"].discard(handler_task)


async def require_admin(request: web.Request) -> None:
    if not admin_allowed(request):
        raise web.HTTPNotFound()


async def admin_backend_contract(request: web.Request) -> web.Response:
    await require_admin(request)
    try:
        contract = request.app["backend_serving_contract"]
        if not backend_contract_valid(contract):
            raise RuntimeError("backend serving contract digest mismatch")
        return web.json_response(backend_contract_payload(contract))
    except (KeyError, RuntimeError, TypeError) as exc:
        return web.json_response({"error": str(exc)}, status=503)


async def admin_quiescent(request: web.Request) -> web.Response:
    await require_admin(request)
    metrics_ok, samples = await upstream_metrics(request.app)
    running = scheduler_gauge(samples, "num_requests_running", "requests_running")
    waiting = scheduler_gauge(samples, "num_requests_waiting", "requests_waiting")
    async with request.app["active_lock"]:
        active = int(request.app["active"])
    quiescent = (
        metrics_ok
        and active == 0
        and running == 0.0
        and waiting == 0.0
    )
    return web.json_response(
        {
            "quiescent": quiescent,
            "active_backend_requests": active,
            "upstream_metrics_ok": metrics_ok,
            "upstream_requests_running": running,
            "upstream_requests_waiting": waiting,
        },
        status=200 if metrics_ok else 503,
    )


async def reset_audit_state(app: web.Application) -> int:
    """Advance the signed-request epoch and clear its replay/audit state."""

    async with app["authorization_lock"]:
        async with app["audit_lock"]:
            app["audit_epoch"] += 1
            AUDIT_FILE.write_text("")
            app["audit_count"] = 0
            app["admitted_request_count"] = 0
            app["seen_request_ids"].clear()
            app["audit_epoch_sealed"] = False
            return int(app["audit_epoch"])


async def admin_audit_reset(request: web.Request) -> web.Response:
    await require_admin(request)
    async with request.app["data_plane_gate"].exclusive():
        audit_epoch = await reset_audit_state(request.app)
    return web.json_response({"audit_epoch": audit_epoch})


async def admin_audit_snapshot(request: web.Request) -> web.Response:
    await require_admin(request)
    async with request.app["audit_lock"]:
        lines = AUDIT_FILE.read_text().splitlines() if AUDIT_FILE.exists() else []
        expected_count = int(request.app["audit_count"])
        audit_epoch = int(request.app["audit_epoch"])
    records = [json.loads(line) for line in lines if line]
    if len(records) != expected_count:
        return web.json_response(
            {
                "error": "audit_counter_mismatch",
                "counter": expected_count,
                "records": len(records),
            },
            status=503,
        )
    return web.json_response(
        {
            "audit_epoch": audit_epoch,
            "count": expected_count,
            "records": records,
        }
    )


async def admin_cache_reset(request: web.Request) -> web.Response:
    await require_admin(request)
    async with request.app["data_plane_gate"].exclusive():
        async with request.app["control_client"].post(
            upstream_url("/reset_prefix_cache", base_url=UPSTREAM_BASE_URL),
            allow_redirects=False,
        ) as response:
            body = await response.read()
            if response.status != 200:
                return web.Response(
                    status=response.status,
                    body=body,
                    headers=filtered_headers(response.headers),
                )
    # vLLM's development cache endpoint intentionally returns an empty 200.
    return web.json_response({"reset": True})


async def admin_measurement_reset(request: web.Request) -> web.Response:
    """Atomically close one request epoch and open a cache-clean next epoch."""

    await require_admin(request)
    app = request.app
    async with app["data_plane_gate"].exclusive():
        metrics_ok, samples = await upstream_metrics(app)
        running = scheduler_gauge(samples, "num_requests_running", "requests_running")
        waiting = scheduler_gauge(samples, "num_requests_waiting", "requests_waiting")
        if not metrics_ok or running != 0.0 or waiting != 0.0:
            return web.json_response(
                {
                    "error": "backend_not_quiescent",
                    "upstream_metrics_ok": metrics_ok,
                    "upstream_requests_running": running,
                    "upstream_requests_waiting": waiting,
                },
                status=409 if metrics_ok else 503,
            )
        async with app["control_client"].post(
            upstream_url("/reset_prefix_cache", base_url=UPSTREAM_BASE_URL),
            allow_redirects=False,
        ) as response:
            body = await response.read()
            if response.status != 200:
                return web.Response(
                    status=response.status,
                    body=body,
                    headers=filtered_headers(response.headers),
                )

        deadline = time.monotonic() + 30.0
        last_usage: float | None = None
        while time.monotonic() < deadline:
            metrics_ok, samples = await upstream_metrics(app)
            last_usage = scheduler_gauge(
                samples,
                "kv_cache_usage_perc",
                "gpu_cache_usage_perc",
            )
            if metrics_ok and last_usage == 0.0:
                break
            await asyncio.sleep(0.1)
        else:
            return web.json_response(
                {
                    "error": "prefix_cache_did_not_clear",
                    "upstream_metrics_ok": metrics_ok,
                    "kv_cache_usage": last_usage,
                },
                status=503,
            )

        audit_epoch = await reset_audit_state(app)
        return web.json_response(
            {
                "reset": True,
                "audit_epoch": audit_epoch,
                "kv_cache_usage": last_usage,
            }
        )


async def admin_cancel_requests(request: web.Request) -> web.Response:
    """Cancel trusted in-flight benchmark requests after an infrastructure stall."""
    await require_admin(request)
    async with request.app["authorization_lock"]:
        tasks = list(request.app["admitted_tasks"])
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    return web.json_response({"cancelled_requests": len(tasks)})


async def admin_measurement_cutoff(request: web.Request) -> web.Response:
    """Seal the current epoch, cancel every admitted request, then snapshot metrics.

    Sealing and copying ``admitted_tasks`` share the authorization lock with
    final data-plane admission.  Thus each racing request is either included
    in the cancellation set or rejected before it can touch the upstream.
    """

    await require_admin(request)
    app = request.app
    try:
        payload = await request.json()
    except (json.JSONDecodeError, ValueError):
        return web.json_response({"error": "invalid_json"}, status=400)
    expected_epoch = payload.get("audit_epoch")
    if (
        not isinstance(expected_epoch, int)
        or isinstance(expected_epoch, bool)
        or expected_epoch < 0
    ):
        return web.json_response({"error": "invalid_audit_epoch"}, status=400)
    async with app["authorization_lock"]:
        if expected_epoch != app["audit_epoch"]:
            return web.json_response(
                {
                    "error": "audit_epoch_mismatch",
                    "expected": expected_epoch,
                    "observed": app["audit_epoch"],
                },
                status=409,
            )
        if app["audit_epoch_sealed"] is True:
            return web.json_response({"error": "audit_epoch_already_sealed"}, status=409)
        app["audit_epoch_sealed"] = True
        sealed_unix = time.time()
        admitted_request_count = int(app["admitted_request_count"])
        tasks = list(app["admitted_tasks"])
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    async with app["control_client"].get(
        upstream_url("/metrics", base_url=UPSTREAM_BASE_URL),
        allow_redirects=False,
    ) as response:
        raw = await response.read()
        if response.status != 200:
            return web.json_response(
                {"error": "upstream_metrics_unavailable", "status": response.status},
                status=503,
            )
    decoded = raw.decode("utf-8", errors="replace")
    async with app["active_lock"]:
        active_backend_requests = int(app["active"])
    async with app["audit_lock"]:
        audit_count = int(app["audit_count"])
    metrics = {
        "schema_version": 1,
        "created_unix": time.time(),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "samples": parse_metric_samples(decoded),
        "active_backend_requests": active_backend_requests,
        "audit_epoch": expected_epoch,
        "audit_count": audit_count,
        "admitted_request_count": admitted_request_count,
    }
    return web.json_response(
        {
            "schema_version": 1,
            "sealed": True,
            "audit_epoch": expected_epoch,
            "sealed_unix": sealed_unix,
            "cancelled_requests": len(tasks),
            "metrics": metrics,
        }
    )


async def admin_metrics_snapshot(request: web.Request) -> web.Response:
    await require_admin(request)
    async with request.app["control_client"].get(
        upstream_url("/metrics", base_url=UPSTREAM_BASE_URL),
        allow_redirects=False,
    ) as response:
        raw = await response.read()
        if response.status != 200:
            return web.json_response(
                {"error": "upstream_metrics_unavailable", "status": response.status},
                status=503,
            )
    decoded = raw.decode("utf-8", errors="replace")
    async with request.app["active_lock"]:
        active_backend_requests = int(request.app["active"])
    async with request.app["audit_lock"]:
        audit_epoch = int(request.app["audit_epoch"])
        audit_count = int(request.app["audit_count"])
    async with request.app["authorization_lock"]:
        admitted_request_count = int(request.app["admitted_request_count"])
    return web.json_response(
        {
            "schema_version": 1,
            "created_unix": time.time(),
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "samples": parse_metric_samples(decoded),
            "active_backend_requests": active_backend_requests,
            "audit_epoch": audit_epoch,
            "audit_count": audit_count,
            "admitted_request_count": admitted_request_count,
        }
    )


async def on_startup(app: web.Application) -> None:
    contract = load_backend_serving_contract()
    if not backend_contract_valid(contract):
        raise RuntimeError("backend serving contract digest mismatch")
    app["backend_serving_contract"] = contract
    app["transport_credential"] = derive_transport_credential(token())
    AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_FILE.touch(exist_ok=True)
    # Startup is single-threaded. Mirror retained IDs as well as the audit
    # count so a restarted proxy cannot re-admit a request from the current
    # epoch before the controller performs its mandatory atomic reset.
    retained_records = [
        json.loads(line)
        for line in AUDIT_FILE.read_text().splitlines()
        if line
    ]
    retained_ids = [record.get("request_id") for record in retained_records]
    if not all(isinstance(value, str) and value for value in retained_ids):
        raise RuntimeError("retained audit contains an invalid request ID")
    if len(set(retained_ids)) != len(retained_ids):
        raise RuntimeError("retained audit contains duplicate request IDs")
    app["audit_count"] = len(retained_records)
    app["admitted_request_count"] = len(retained_records)
    app["seen_request_ids"] = set(retained_ids)
    # The data plane must expose the engine's frozen 192-sequence ceiling;
    # aiohttp's implicit connector default of 100 would silently benchmark a
    # different scheduler.  A separate small control-plane pool prevents
    # health/metrics/cache-reset calls from queueing behind 192 long streams.
    app["data_client"] = ClientSession(
        # Streaming generations can outlive the upstream HTTP keep-alive
        # window.  Never return those transports to a reusable pool: an
        # otherwise healthy long sweep can race the server-side close and turn
        # the next request into an immediate ClientOSError/502.  Opening a new
        # upstream connection preserves the one-client-request/one-backend-
        # request contract without adding retries.
        connector=TCPConnector(
            limit=192,
            limit_per_host=0,
            force_close=True,
        ),
        timeout=ClientTimeout(total=None, connect=30, sock_read=None),
        auto_decompress=False,
    )
    app["control_client"] = ClientSession(
        connector=TCPConnector(limit=8, limit_per_host=0),
        timeout=ClientTimeout(total=30, connect=10, sock_read=30),
        auto_decompress=False,
    )


async def on_cleanup(app: web.Application) -> None:
    await app["data_client"].close()
    await app["control_client"].close()


def make_app() -> web.Application:
    app = web.Application(
        client_max_size=16 * 1024**2,
        # A closed loopback-bridge connection must cancel the corresponding
        # upstream request instead of letting an orphaned generation consume
        # serving capacity.
        handler_args={"handler_cancellation": True},
        middlewares=[require_data_plane_transport],
    )
    app["audit_lock"] = asyncio.Lock()
    app["authorization_lock"] = asyncio.Lock()
    app["active_lock"] = asyncio.Lock()
    app["data_plane_gate"] = DataPlaneGate()
    app["active"] = 0
    app["active_tasks"] = set()
    app["audit_epoch"] = 0
    app["audit_epoch_sealed"] = False
    app["audit_count"] = 0
    app["admitted_request_count"] = 0
    app["admitted_tasks"] = set()
    app["seen_request_ids"] = set()
    app.router.add_get("/health", health)
    app.router.add_get("/v1/metrics", scheduler_metrics)
    app.router.add_get("/v1/models", proxy_get)
    app.router.add_post("/v1/chat/completions", chat_completions)
    app.router.add_get("/admin/backend-contract", admin_backend_contract)
    app.router.add_get("/admin/quiescent", admin_quiescent)
    app.router.add_post("/admin/audit/reset", admin_audit_reset)
    app.router.add_get("/admin/audit/snapshot", admin_audit_snapshot)
    app.router.add_post("/admin/cache/reset", admin_cache_reset)
    app.router.add_post("/admin/measurement/reset", admin_measurement_reset)
    app.router.add_post("/admin/measurement/cutoff", admin_measurement_cutoff)
    app.router.add_post("/admin/requests/cancel", admin_cancel_requests)
    app.router.add_get("/admin/metrics/snapshot", admin_metrics_snapshot)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


if __name__ == "__main__":
    web.run_app(make_app(), host="0.0.0.0", port=8100, access_log=None)
