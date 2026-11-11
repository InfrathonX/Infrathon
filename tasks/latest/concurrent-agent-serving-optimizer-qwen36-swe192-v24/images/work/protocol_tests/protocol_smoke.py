#!/usr/bin/env python3
"""Public loopback smoke test for an OpenAI-compatible scheduling gateway.

This is intentionally a protocol-only test.  It uses deterministic synthetic
requests and an in-process fake backend; it contains no benchmark workload,
model access, capacity constants, private nonce, scoring controller, or oracle.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import http.client
import json
import os
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


SUBMISSION_DIR = Path(os.environ.get("SUBMISSION_DIR", "/app/submission"))
LAUNCH = SUBMISSION_DIR / "launch.sh"
NONSTREAM_BODY = json.dumps(
    {
        "id": "local-nonstream",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hello 世界"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    },
    ensure_ascii=False,
    separators=(",", ":"),
).encode()
MODELS_BODY = b'{"object":"list","data":[{"id":"target-model","object":"model"}]}'
STREAM_CHUNKS = (
    b'data: {"id":"local-stream","choices":[{"index":0,"delta":{"content":"alpha"}}]}\n\n',
    b'data: {"id":"local-stream","choices":[{"index":0,"delta":{"content":" beta"},"finish_reason":"stop"}]}\n\n',
    b'data: {"id":"local-stream","choices":[],"usage":{"prompt_tokens":6,"completion_tokens":2,"total_tokens":8}}\n\n',
    b"data: [DONE]\n\n",
)


def reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class Ledger:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: list[dict[str, Any]] = []

    def append(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._records.append(record)

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(record) for record in self._records]


LEDGER = Ledger()


class FakeBackendHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "MLSys ForgeLocalFake/1"

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _send(
        self,
        status: int,
        body: bytes,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _record(self, body: bytes) -> None:
        LEDGER.append(
            {
                "method": self.command,
                "path": self.path,
                "body_sha256": hashlib.sha256(body).hexdigest(),
                "run_id": self.headers.get("X-Agent-Run-ID"),
                "trace": self.headers.get("X-Local-Trace"),
            }
        )

    def do_GET(self) -> None:  # noqa: N802
        self._record(b"")
        if self.path == "/v1/models":
            self._send(
                200,
                MODELS_BODY,
                {"Content-Type": "application/json", "X-Local-Backend": "models"},
            )
        else:
            self._send(404, b'{"error":"not_found"}')

    def do_POST(self) -> None:  # noqa: N802
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self._record(body)
        if self.path != "/v1/chat/completions":
            self._send(404, b'{"error":"not_found"}')
            return

        try:
            payload = json.loads(body)
            content = str(payload["messages"][-1]["content"])
        except (KeyError, IndexError, TypeError, UnicodeDecodeError, json.JSONDecodeError):
            self._send(
                400,
                b'{"error":{"type":"invalid_json","message":"malformed"}}',
                {"Content-Type": "application/json"},
            )
            return

        if content == "local-force-429":
            self._send(
                429,
                b'{"error":{"type":"rate_limit","message":"local synthetic 429"}}',
                {"Content-Type": "application/json", "Retry-After": "2"},
            )
            return

        if bool(payload.get("stream")):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            for chunk in STREAM_CHUNKS:
                self.wfile.write(chunk)
                self.wfile.flush()
                time.sleep(0.005)
            return

        self._send(
            200,
            NONSTREAM_BODY,
            {"Content-Type": "application/json", "X-Local-Backend": "nonstream"},
        )


class LocalServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 64


def chat_body(content: str, *, stream: bool = False) -> bytes:
    return json.dumps(
        {
            "model": "target-model",
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": 16,
            "stream": stream,
        },
        separators=(",", ":"),
    ).encode()


def request(
    port: int,
    method: str,
    path: str,
    body: bytes | None,
    run_id: str,
) -> tuple[int, bytes, list[tuple[str, str]]]:
    headers = {
        "X-Agent-Run-ID": run_id,
        "X-Local-Trace": f"trace-{run_id}",
        "Authorization": "Bearer local-opaque-value",
    }
    if body is not None:
        headers["Content-Type"] = "application/json"
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        return response.status, response.read(), response.getheaders()
    finally:
        connection.close()


def require_response(
    response: tuple[int, bytes, list[tuple[str, str]]],
    status: int,
    body: bytes,
) -> None:
    if response[0] != status or response[1] != body:
        raise RuntimeError(
            "response changed: "
            f"status={response[0]}/{status}, "
            f"body_sha256={hashlib.sha256(response[1]).hexdigest()}/"
            f"{hashlib.sha256(body).hexdigest()}"
        )


def wait_for_health(port: int, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 20
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = process.stdout.read().decode(errors="replace") if process.stdout else ""
            raise RuntimeError(
                f"gateway exited during startup ({process.returncode}): {output[-2000:]}"
            )
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.05)
    raise RuntimeError("gateway did not become healthy")


def main() -> int:
    if not LAUNCH.is_file() or not os.access(LAUNCH, os.X_OK):
        raise RuntimeError(f"missing executable launch script: {LAUNCH}")

    backend = LocalServer(("127.0.0.1", 0), FakeBackendHandler)
    backend_port = int(backend.server_address[1])
    backend_thread = threading.Thread(target=backend.serve_forever, daemon=True)
    backend_thread.start()

    gateway_port = reserve_port()
    environment = os.environ.copy()
    environment.update(
        {
            "BACKEND_URL": f"http://127.0.0.1:{backend_port}/v1",
            "PORT": str(gateway_port),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    process = subprocess.Popen(
        [str(LAUNCH)],
        cwd=SUBMISSION_DIR,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    checks: list[str] = []
    expected: list[tuple[bytes, str]] = []
    try:
        wait_for_health(gateway_port, process)
        if LEDGER.snapshot():
            raise RuntimeError("gateway /health was incorrectly forwarded")
        checks.append("local_health")

        require_response(
            request(gateway_port, "GET", "/v1/models", None, "models"),
            200,
            MODELS_BODY,
        )
        checks.append("models_forwarding")

        normal = chat_body("local-nonstream")
        expected.append((normal, "normal"))
        require_response(
            request(gateway_port, "POST", "/v1/chat/completions", normal, "normal"),
            200,
            NONSTREAM_BODY,
        )
        checks.append("nonstream_bytes")

        stream = chat_body("local-stream", stream=True)
        expected.append((stream, "stream"))
        require_response(
            request(gateway_port, "POST", "/v1/chat/completions", stream, "stream"),
            200,
            b"".join(STREAM_CHUNKS),
        )
        checks.append("stream_order_and_done")

        throttled = chat_body("local-force-429")
        expected.append((throttled, "error"))
        require_response(
            request(gateway_port, "POST", "/v1/chat/completions", throttled, "error"),
            429,
            b'{"error":{"type":"rate_limit","message":"local synthetic 429"}}',
        )
        checks.append("error_passthrough")

        records_before_release = len(LEDGER.snapshot())
        release_body = b'{"program_id":"release-local"}'
        release = request(
            gateway_port,
            "POST",
            "/programs/release",
            release_body,
            "release-local",
        )
        if not (200 <= release[0] < 300 or release[0] in {404, 405}):
            raise RuntimeError(f"unexpected lifecycle status: {release[0]}")
        if len(LEDGER.snapshot()) != records_before_release:
            raise RuntimeError("lifecycle notification reached the model backend")
        checks.append("lifecycle_not_forwarded")

        concurrent_cases: list[tuple[bytes, str]] = []
        for index in range(12):
            run_id = f"concurrent-{index}"
            body = chat_body(f"local-{run_id}")
            concurrent_cases.append((body, run_id))
            expected.append((body, run_id))
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            futures = [
                pool.submit(
                    request,
                    gateway_port,
                    "POST",
                    "/v1/chat/completions",
                    body,
                    run_id,
                )
                for body, run_id in concurrent_cases
            ]
            for future in futures:
                require_response(future.result(), 200, NONSTREAM_BODY)
        checks.append("concurrent_12")

        records = LEDGER.snapshot()
        model_records = [
            record
            for record in records
            if record["method"] == "GET" and record["path"] == "/v1/models"
        ]
        if len(model_records) != 1:
            raise RuntimeError(f"models request reached backend {len(model_records)} times")
        if model_records[0]["run_id"] != "models":
            raise RuntimeError("models opaque run id changed")

        for body, run_id in expected:
            digest = hashlib.sha256(body).hexdigest()
            matching = [record for record in records if record["body_sha256"] == digest]
            if len(matching) != 1:
                raise RuntimeError(
                    f"request {run_id} reached backend {len(matching)} times"
                )
            if matching[0]["run_id"] != run_id:
                raise RuntimeError(f"opaque run id changed for {run_id}")
            if matching[0]["trace"] != f"trace-{run_id}":
                raise RuntimeError(f"opaque trace header changed for {run_id}")
        checks.append("one_to_one_body_and_headers")

        report = {
            "schema_version": 1,
            "kind": "local-transparent-gateway-protocol-smoke",
            "passed": True,
            "checks": checks,
            "accepted_model_requests": len(expected) + 1,
            "observed_backend_requests": len(records),
            "note": "synthetic loopback protocol test only; not a performance score",
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        backend.shutdown()
        backend.server_close()
        backend_thread.join(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
