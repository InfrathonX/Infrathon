from __future__ import annotations

import hashlib
import hmac
import http.client
import json
import os
import socket
import threading
import time
import urllib.parse
import uuid
from dataclasses import dataclass
from typing import Any

from runner.ledger import Ledger, LedgerClosed, MEASUREMENT_WINDOW_REASON


TRANSPORT_AUTH_HEADER = "X-KVBench-Transport-Auth"
TRANSPORT_AUTH_CONTEXT = b"kvbench-model-api-transport-v1"
BENCHMARK_AUTH_CONTEXT = b"kvbench-data-request-v2"
MAX_SSE_LINE_BYTES = 1024 * 1024
MAX_MODEL_RESPONSE_BYTES = 8 * 1024 * 1024


class RequestCancelled(RuntimeError):
    """A trusted benchmark watchdog cancelled an active HTTP request."""


class MeasurementWindowEnded(RuntimeError):
    """The trusted fixed-duration public measurement reached its boundary."""

    def __init__(self, cutoff_unix: float) -> None:
        self.cutoff_unix = cutoff_unix
        super().__init__(f"measurement window ended at {cutoff_unix:.6f}")


class ResponseLimitExceeded(RuntimeError):
    """An untrusted response crossed a frozen client-side byte limit."""

    def __init__(self, kind: str, observed_bytes: int, limit_bytes: int) -> None:
        if kind not in {"sse_line_bytes", "response_bytes"}:
            raise ValueError(f"unsupported response limit kind: {kind}")
        self.kind = kind
        self.observed_bytes = observed_bytes
        self.limit_bytes = limit_bytes
        super().__init__(
            f"{kind} exceeded: observed_bytes={observed_bytes},"
            f"limit_bytes={limit_bytes}"
        )


class CancellableHTTPConnection(http.client.HTTPConnection):
    """One-shot HTTP connection that another thread can reliably interrupt."""

    def __init__(self, host: str, port: int, timeout: int) -> None:
        super().__init__(host, port, timeout=timeout)
        self._cancel_lock = threading.Lock()
        self._cancel_reason: str | None = None

    @property
    def cancel_reason(self) -> str | None:
        with self._cancel_lock:
            return self._cancel_reason

    def connect(self) -> None:
        with self._cancel_lock:
            reason = self._cancel_reason
        if reason is not None:
            raise RequestCancelled(reason)
        # Both trusted endpoints are on the local Compose network.  Keep the
        # connect phase independently bounded so a gateway that stops accepting
        # sockets after its health check cannot strand worker threads for the
        # much larger generation/read timeout.
        io_timeout = self.timeout
        self.timeout = min(float(io_timeout), 5.0)
        try:
            super().connect()
        finally:
            self.timeout = io_timeout
        if self.sock is not None:
            try:
                self.sock.settimeout(io_timeout)
            except OSError:
                pass
        with self._cancel_lock:
            reason = self._cancel_reason
            connected_socket = self.sock
        if reason is not None:
            if connected_socket is not None:
                try:
                    connected_socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            super().close()
            raise RequestCancelled(reason)

    def cancel(self, reason: str) -> None:
        with self._cancel_lock:
            if self._cancel_reason is None:
                self._cancel_reason = reason
            connected_socket = self.sock
        # close() alone does not portably wake a recv() blocked in another
        # thread.  shutdown() does, including while waiting for HTTP headers or
        # while a malicious gateway trickles an incomplete SSE response.
        if connected_socket is not None:
            try:
                connected_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        super().close()


def cancellable_connection(
    endpoint: str,
    timeout_sec: int,
) -> tuple[CancellableHTTPConnection, str]:
    parsed = urllib.parse.urlsplit(endpoint)
    if (
        parsed.scheme != "http"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"unsupported benchmark model endpoint: {endpoint!r}")
    port = parsed.port or 80
    base_path = parsed.path.rstrip("/")
    path = f"{base_path}/chat/completions" or "/chat/completions"
    return CancellableHTTPConnection(parsed.hostname, port, timeout_sec), path


def derive_transport_credential(admin_token: str) -> str:
    if not admin_token:
        raise ValueError("admin token is empty")
    return hmac.new(
        admin_token.encode(),
        TRANSPORT_AUTH_CONTEXT,
        hashlib.sha256,
    ).hexdigest()


def is_direct_model_api_endpoint(endpoint: str) -> bool:
    """Select only the controller's exact trusted baseline data endpoint."""
    trusted = os.environ.get(
        "TRUSTED_MODEL_API_DATA_URL",
        "http://model-api:8100/v1",
    )
    return endpoint.rstrip("/") == trusted.rstrip("/")


def transport_headers_for_endpoint(endpoint: str, admin_token: str) -> dict[str, str]:
    if not is_direct_model_api_endpoint(endpoint):
        return {}
    return {TRANSPORT_AUTH_HEADER: derive_transport_credential(admin_token)}


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
    """Mirror the trusted model-api's canonical data-request v2 contract."""

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


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = "target-model"
    max_tokens: int = 384
    temperature: float = 1.0
    top_p: float = 1.0
    enable_thinking: bool = False


def classify_tool_arguments(raw_arguments: object) -> str:
    if not isinstance(raw_arguments, str):
        return "non_string"
    stripped = raw_arguments.rstrip()
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError:
        if not stripped:
            return "invalid_empty"
        if stripped.endswith('"]}'):
            return "invalid_trailing_bracket"
        if stripped.startswith('{"cmd"'):
            return "invalid_cmd_json_other"
        return "invalid_json_other"
    if not isinstance(arguments, dict):
        return "valid_json_non_object"
    if (
        set(arguments) == {"cmd"}
        and isinstance(arguments.get("cmd"), str)
        and bool(arguments["cmd"].strip())
    ):
        return "valid_cmd_only"
    return "valid_json_bad_schema"


class BenchmarkVllmModel:
    """Synchronous streaming adapter used by one mini-SWE workflow thread."""

    def __init__(
        self,
        *,
        endpoint: str,
        run_id: str,
        profile: str,
        leg_id: str,
        cache_namespace: str,
        seed: int,
        admin_token: str,
        audit_epoch: int,
        ledger: Ledger,
        config: ModelConfig,
        request_timeout_sec: int,
        cancel_event: threading.Event | None = None,
        max_sse_line_bytes: int = MAX_SSE_LINE_BYTES,
        max_response_bytes: int = MAX_MODEL_RESPONSE_BYTES,
    ) -> None:
        if (
            not isinstance(audit_epoch, int)
            or isinstance(audit_epoch, bool)
            or audit_epoch < 0
        ):
            raise ValueError("audit_epoch must be a non-negative integer")
        if (
            not isinstance(max_sse_line_bytes, int)
            or isinstance(max_sse_line_bytes, bool)
            or max_sse_line_bytes <= 0
        ):
            raise ValueError("max_sse_line_bytes must be a positive integer")
        if (
            not isinstance(max_response_bytes, int)
            or isinstance(max_response_bytes, bool)
            or max_response_bytes < max_sse_line_bytes
        ):
            raise ValueError(
                "max_response_bytes must be an integer at least max_sse_line_bytes"
            )
        self.endpoint = endpoint.rstrip("/")
        self.run_id = run_id
        self.profile = profile
        self.leg_id = leg_id
        self.cache_namespace = cache_namespace
        self.seed = seed
        self.admin_token = admin_token
        self.audit_epoch = audit_epoch
        self.ledger = ledger
        self.config = config
        self.request_timeout_sec = request_timeout_sec
        self.cancel_event = cancel_event
        self.max_sse_line_bytes = max_sse_line_bytes
        self.max_response_bytes = max_response_bytes
        self.cost = 0.0
        self.n_calls = 0

    def get_template_vars(self) -> dict[str, Any]:
        return {"model_name": self.config.model_name, "n_model_calls": self.n_calls, "model_cost": 0.0}

    def query(self, messages: list[dict[str, Any]], **_: Any) -> dict[str, Any]:
        response, _timing = self.query_stream_timed(messages)
        return response

    def query_stream_timed(self, messages: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, float]]:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise RuntimeError("benchmark cancelled by the infrastructure watchdog")
        call_index = self.n_calls + 1
        request_id = str(uuid.uuid4())
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "seed": self.seed + call_index,
            "max_tokens": self.config.max_tokens,
            # Qwen3.6 can spend the entire action budget in an unterminated
            # think block.  This benchmark freezes the model's supported
            # non-thinking chat-template mode so real SWE action trajectories
            # reach the tool loop under the paper's 2048-token cap.
            "chat_template_kwargs": {
                "enable_thinking": self.config.enable_thinking,
            },
            # vLLM applies cache_salt only to prefix-cache hashing.  A stable
            # per-workflow value preserves reuse across an agent's turns while
            # preventing duplicated SWE-bench rollouts from sharing KV blocks.
            "cache_salt": self.cache_namespace,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        body_digest = hashlib.sha256(body).hexdigest()
        connection, request_path = cancellable_connection(
            self.endpoint,
            self.request_timeout_sec,
        )
        auth = hmac.new(
            self.admin_token.encode(),
            benchmark_auth_message(
                audit_epoch=self.audit_epoch,
                method="POST",
                path=request_path,
                request_id=request_id,
                run_id=self.run_id,
                profile=self.profile,
                leg_id=self.leg_id,
                cache_namespace=self.cache_namespace,
                body_digest=body_digest,
            ),
            hashlib.sha256,
        ).hexdigest()
        headers = {
            "Content-Type": "application/json",
            "X-Agent-Run-ID": self.run_id,
            # Upstream ThunderAgent recognizes X-Session-ID as a program ID.
            # Mirror the same opaque value; X-Agent-Run-ID remains the public
            # task identity and the trusted audit identity.
            "X-Session-ID": self.run_id,
            "X-KVBench-Request-ID": request_id,
            "X-KVBench-Auth": auth,
            "X-KVBench-Audit-Epoch": str(self.audit_epoch),
            "X-KVBench-Body-SHA256": body_digest,
            "X-KVBench-Profile": self.profile,
            "X-KVBench-Leg": self.leg_id,
            "X-KVBench-Cache-Namespace": self.cache_namespace,
        }
        # Candidate legs terminate at main:9000 and must never receive the
        # model-api transport credential.  Only an exact match for the frozen
        # direct baseline URL gets this derived header.
        headers.update(
            transport_headers_for_endpoint(self.endpoint, self.admin_token)
        )
        started = time.time()
        first_token: float | None = None
        ended = started
        response_digest = hashlib.sha256()
        response_bytes = 0
        status = 599
        content: list[str] = []
        reasoning: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        internal_reasoning_chars = 0
        finish_reason: str | None = None
        usage: dict[str, Any] | None = None
        done_seen = False
        stream_error: str | None = None
        error: str | None = None
        response_limit: dict[str, int | str] | None = None
        try:
            self.ledger.begin(
                {
                    "request_id": request_id,
                    "run_id": self.run_id,
                    "profile": self.profile,
                    "leg_id": self.leg_id,
                    "cache_namespace": self.cache_namespace,
                    "call_index": call_index,
                    "audit_epoch": self.audit_epoch,
                    "canonical_body_digest": body_digest,
                    "started_unix": started,
                }
            )
        except LedgerClosed as exc:
            connection.close()
            if exc.reason == MEASUREMENT_WINDOW_REASON:
                raise MeasurementWindowEnded(exc.cutoff_unix) from exc
            raise
        self.ledger.register_canceller(request_id, connection.cancel)
        try:
            connection.request("POST", request_path, body=body, headers=headers)
            with connection.getresponse() as response:
                status = response.status
                if status != 200:
                    raw = response.read(self.max_response_bytes + 1)
                    response_digest.update(raw)
                    response_bytes += len(raw)
                    if response_bytes > self.max_response_bytes:
                        raise ResponseLimitExceeded(
                            "response_bytes",
                            response_bytes,
                            self.max_response_bytes,
                        )
                    ended = time.time()
                    error = f"HTTP {status}: {raw[-500:]!r}"
                    raise RuntimeError(error)
                while True:
                    # HTTPResponse iteration calls an unbounded readline(). A
                    # candidate gateway controls this byte stream and could
                    # otherwise force a workflow thread to buffer one
                    # arbitrarily large unterminated SSE line. Reading one
                    # byte past the frozen line limit makes the violation
                    # observable without ever allocating the whole line.
                    line = response.readline(self.max_sse_line_bytes + 1)
                    if not line:
                        break
                    response_digest.update(line)
                    response_bytes += len(line)
                    if len(line) > self.max_sse_line_bytes:
                        raise ResponseLimitExceeded(
                            "sse_line_bytes",
                            len(line),
                            self.max_sse_line_bytes,
                        )
                    if response_bytes > self.max_response_bytes:
                        raise ResponseLimitExceeded(
                            "response_bytes",
                            response_bytes,
                            self.max_response_bytes,
                        )
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if not data:
                        continue
                    if data == b"[DONE]":
                        done_seen = True
                        continue
                    event = json.loads(data)
                    raw_stream_error = event.get("error")
                    if raw_stream_error is not None:
                        if isinstance(raw_stream_error, dict):
                            selected = {
                                key: raw_stream_error.get(key)
                                for key in ("type", "code", "message")
                                if raw_stream_error.get(key) is not None
                            }
                            stream_error = json.dumps(
                                selected,
                                ensure_ascii=False,
                                separators=(",", ":"),
                                sort_keys=True,
                            )[:1000]
                        else:
                            stream_error = str(raw_stream_error)[:1000]
                        continue
                    choices = event.get("choices") or []
                    if choices:
                        choice = choices[0]
                        if isinstance(choice.get("finish_reason"), str):
                            finish_reason = choice["finish_reason"]
                        delta = choice.get("delta") or {}
                        text = delta.get("content")
                        thought = delta.get("reasoning_content")
                        internal_reasoning = delta.get("reasoning")
                        delta_tool_calls = delta.get("tool_calls") or []
                        if text:
                            content.append(text)
                        if thought:
                            reasoning.append(thought)
                        if isinstance(internal_reasoning, str):
                            internal_reasoning_chars += len(internal_reasoning)
                        for position, chunk in enumerate(delta_tool_calls):
                            if not isinstance(chunk, dict):
                                continue
                            raw_index = chunk.get("index", position)
                            if not isinstance(raw_index, int) or isinstance(raw_index, bool) or raw_index < 0:
                                continue
                            assembled = tool_calls.setdefault(
                                raw_index,
                                {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                            )
                            if isinstance(chunk.get("id"), str):
                                assembled["id"] += chunk["id"]
                            if isinstance(chunk.get("type"), str):
                                assembled["type"] = chunk["type"]
                            function = chunk.get("function")
                            if isinstance(function, dict):
                                if isinstance(function.get("name"), str):
                                    assembled["function"]["name"] += function["name"]
                                if isinstance(function.get("arguments"), str):
                                    assembled["function"]["arguments"] += function["arguments"]
                        if (text or thought or internal_reasoning or delta_tool_calls) and first_token is None:
                            first_token = time.time()
                    if isinstance(event.get("usage"), dict):
                        usage = event["usage"]
                if status == 200 and not done_seen:
                    raise RuntimeError("stream ended without the [DONE] sentinel")
                if status == 200 and stream_error is not None:
                    raise RuntimeError(f"model stream error: {stream_error}")
                if status == 200 and not isinstance(usage, dict):
                    raise RuntimeError("stream ended without a usage event")
                ended = time.time()
        except Exception as exc:
            ended = time.time()
            cancel_reason = self.ledger.cancellation_reason(request_id)
            if cancel_reason is not None:
                error = f"RequestCancelled: {cancel_reason}"
                if cancel_reason == MEASUREMENT_WINDOW_REASON:
                    cutoff = self.ledger.measurement_state().get("cutoff_unix")
                    if isinstance(cutoff, (int, float)) and not isinstance(cutoff, bool):
                        raise MeasurementWindowEnded(float(cutoff)) from exc
                raise RequestCancelled(cancel_reason) from exc
            if isinstance(exc, ResponseLimitExceeded):
                response_limit = {
                    "kind": exc.kind,
                    "observed_bytes": exc.observed_bytes,
                    "limit_bytes": exc.limit_bytes,
                }
            if error is None:
                error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            connection.close()
            self.n_calls += 1
            ordered_tool_calls = [
                tool_calls[index] for index in sorted(tool_calls)
            ]
            self.ledger.complete(
                {
                    "request_id": request_id,
                    "run_id": self.run_id,
                    "profile": self.profile,
                    "leg_id": self.leg_id,
                    "cache_namespace": self.cache_namespace,
                    "call_index": call_index,
                    "audit_epoch": self.audit_epoch,
                    "canonical_body_digest": body_digest,
                    "response_digest": response_digest.hexdigest(),
                    "response_bytes": response_bytes,
                    "status": status,
                    "started_unix": started,
                    "first_token_unix": first_token,
                    "ended_unix": ended,
                    "duration_sec": ended - started,
                    "prompt_tokens": usage.get("prompt_tokens") if usage else None,
                    "completion_tokens": usage.get("completion_tokens") if usage else None,
                    "total_tokens": usage.get("total_tokens") if usage else None,
                    "finish_reason": finish_reason,
                    "content_chars": sum(len(item) for item in content),
                    "reasoning_content_chars": sum(len(item) for item in reasoning),
                    "internal_reasoning_chars": internal_reasoning_chars,
                    "tool_call_count": len(tool_calls),
                    "tool_call_names": [
                        str((call.get("function") or {}).get("name"))[:160]
                        for call in ordered_tool_calls
                    ],
                    "tool_argument_shapes": [
                        classify_tool_arguments(
                            (call.get("function") or {}).get("arguments")
                        )
                        for call in ordered_tool_calls
                    ],
                    "sse_done": done_seen,
                    "sse_error": stream_error,
                    "response_limit": response_limit,
                    "error": error,
                }
            )
            if response_limit is not None:
                # Complete this record first so peer cancellation cannot turn
                # the originating limit violation into a generic
                # RequestCancelled record. The shared event stops queued
                # workflows, while shutdown() wakes every active peer socket.
                detail = (
                    "model_response_limit_exceeded:"
                    f"kind={response_limit['kind']},"
                    f"observed_bytes={response_limit['observed_bytes']},"
                    f"limit_bytes={response_limit['limit_bytes']}"
                )
                if self.cancel_event is not None:
                    self.cancel_event.set()
                self.ledger.cancel_active(detail)

        if status != 200:
            raise RuntimeError(f"model endpoint returned HTTP {status}")
        first_token = first_token or ended
        message: dict[str, Any] = {"content": "".join(content)}
        if reasoning:
            message["reasoning_content"] = "".join(reasoning)
        if tool_calls:
            message["tool_calls"] = ordered_tool_calls
        return message, {"first_token_ts": first_token, "last_token_ts": ended}
