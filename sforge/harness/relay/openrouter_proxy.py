"""Host-side Responses relay with a private upstream key and pinned routing.

Workers receive a per-run relay token. Request bodies and credentials are never
logged; usage receipts contain only routing, timings, status, and numeric usage.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import signal
import threading
import time
import uuid
import zlib

import requests

from sforge.harness.agent_config import read_private_file

DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash"
MAX_REQUEST_BYTES = 16 * 1024 * 1024
MAX_EVENT_BYTES = 1024 * 1024


def load_openrouter_key(path: Path) -> str:
    """Accept a private bare-key file or a conventional dotenv key file."""
    data = read_private_file(path).decode("utf-8").strip()
    if data.startswith("sk-or-") and "\n" not in data:
        return data
    from dotenv import dotenv_values
    import io
    values = dotenv_values(stream=io.StringIO(data), interpolate=False)
    key = values.get("OPENROUTER_API_KEY")
    if not isinstance(key, str) or not key.strip():
        raise ValueError("OpenRouter credential file has no usable API key")
    return key.strip()


def load_or_create_relay_token(path: Path) -> str:
    """The short-lived bearer is the only credential copied into the Worker."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return read_private_file(path).decode("utf-8").strip()
    token = secrets.token_urlsafe(48)
    with os.fdopen(fd, "w") as stream:
        stream.write(token + "\n")
    return token


def pin_request(document: dict, model: str, provider: str, effort: str) -> dict:
    """Preserve Codex input/tools while enforcing the requested experiment."""
    if not isinstance(document, dict):
        raise ValueError("Responses request must be a JSON object")
    result = dict(document)
    result["model"] = model
    result.pop("models", None)
    result["provider"] = {"only": [provider], "allow_fallbacks": False}
    reasoning = result.get("reasoning")
    reasoning = dict(reasoning) if isinstance(reasoning, dict) else {}
    reasoning.pop("max_tokens", None)
    reasoning.update(effort=effort, enabled=True, exclude=False)
    result["reasoning"] = reasoning
    return result


def numeric_usage(document: dict) -> dict:
    """Allow numeric usage only; provider errors may contain sensitive text."""
    usage = document.get("usage") if isinstance(document, dict) else None
    if not isinstance(usage, dict):
        return {}
    def numbers(value):
        return {str(key): numbers(item) if isinstance(item, dict) else item
                for key, item in value.items()
                if isinstance(item, dict) or isinstance(item, (int, float))
                and not isinstance(item, bool)}
    return numbers(usage)


class UsageObserver:
    """Observe complete SSE events without altering streamed response bytes."""
    def __init__(self):
        self.pending = bytearray()
        self.usage: dict = {}
        self.response_status: str | None = None

    def feed(self, chunk: bytes):
        self.pending.extend(chunk)
        if len(self.pending) > MAX_EVENT_BYTES:
            self.pending.clear()
            return
        while b"\n" in self.pending:
            line, _, remainder = self.pending.partition(b"\n")
            self.pending = bytearray(remainder)
            if not line.startswith(b"data:"):
                continue
            try:
                event = json.loads(line[5:].strip())
            except (ValueError, UnicodeDecodeError):
                continue
            if not isinstance(event, dict):
                continue
            response = event.get("response")
            if isinstance(response, dict):
                self.usage = numeric_usage(response) or self.usage
                status = response.get("status")
                if status in {"completed", "failed", "incomplete", "cancelled"}:
                    self.response_status = status


class OpenRouterResponsesProxy:
    def __init__(self, *, api_key: str, relay_token: str, log_file: Path,
                 model: str = DEFAULT_MODEL, provider: str = "alibaba",
                 effort: str = "max", host: str = "127.0.0.1", port: int = 19413,
                 upstream_base: str = "https://openrouter.ai/api/v1"):
        if not api_key or not relay_token:
            raise ValueError("Both private upstream and relay credentials are required")
        self.api_key, self.relay_token = api_key, relay_token
        self.model, self.provider, self.effort = model, provider, effort
        self.upstream_base = upstream_base.rstrip("/")
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.log_file, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
        os.fchmod(fd, 0o600)
        self.log = os.fdopen(fd, "a")
        self.log_lock = threading.Lock()
        self.server = ThreadingHTTPServer((host, port), self._handler())
        self.thread: threading.Thread | None = None

    def receipt(self, value: dict):
        with self.log_lock:
            self.log.write(json.dumps(value, separators=(",", ":")) + "\n")
            self.log.flush()

    def _handler(self):
        owner = self
        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *args):
                pass

            def send_json(self, status, document):
                payload = json.dumps(document).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def authorized(self):
                actual = self.headers.get("Authorization", "")
                return hmac.compare_digest(actual, "Bearer " + owner.relay_token)

            def do_GET(self):
                if self.path == "/health":
                    self.send_json(200, {"status": "ready"})
                elif not self.authorized():
                    self.send_json(401, {"error": {"message": "Invalid relay token"}})
                elif self.path == "/v1/models":
                    self.send_json(200, {"object": "list", "data": [{"id": owner.model, "object": "model", "owned_by": "openrouter"}]})
                else:
                    self.send_json(404, {"error": {"message": "Unsupported relay endpoint"}})

            def do_POST(self):
                if not self.authorized():
                    self.send_json(401, {"error": {"message": "Invalid relay token"}})
                    return
                if self.path != "/v1/responses":
                    self.send_json(404, {"error": {"message": "Unsupported relay endpoint"}})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= MAX_REQUEST_BYTES or self.headers.get("Transfer-Encoding"):
                        raise ValueError("Invalid request size")
                    self.connection.settimeout(30)
                    raw = self.rfile.read(length)
                    if len(raw) != length:
                        raise ValueError("Incomplete request")
                    encoding = self.headers.get("Content-Encoding", "identity").lower()
                    if encoding == "gzip":
                        decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
                        raw = decoder.decompress(raw, MAX_REQUEST_BYTES + 1)
                        if len(raw) > MAX_REQUEST_BYTES or decoder.unconsumed_tail or not decoder.eof:
                            raise ValueError("Invalid compressed request size")
                    elif encoding != "identity":
                        raise ValueError("Unsupported request encoding")
                    document = pin_request(json.loads(raw), owner.model, owner.provider, owner.effort)
                except (ValueError, OSError, UnicodeDecodeError, zlib.error):
                    self.send_json(400, {"error": {"message": "Invalid bounded Responses request"}})
                    return
                started = time.monotonic()
                receipt = {"timestamp": datetime.now(timezone.utc).isoformat(),
                           "request_id": uuid.uuid4().hex, "model": owner.model,
                           "provider": owner.provider, "reasoning_effort": owner.effort,
                           "stream": bool(document.get("stream")), "status_code": None}
                observer = UsageObserver()
                response_started = False
                try:
                    with requests.Session() as session:
                        session.trust_env = False
                        with session.post(owner.upstream_base + "/responses",
                                          headers={"Authorization": "Bearer " + owner.api_key,
                                                   "Content-Type": "application/json"},
                                          json=document, stream=True, timeout=(20, 600)) as response:
                            receipt["status_code"] = response.status_code
                            self.send_response(response.status_code)
                            for header in ("Content-Type", "Retry-After", "X-Request-ID"):
                                if header in response.headers:
                                    self.send_header(header, response.headers[header])
                            self.end_headers()
                            response_started = True
                            nonstream = bytearray()
                            for chunk in response.iter_content(chunk_size=None):
                                if not chunk:
                                    continue
                                if document.get("stream"):
                                    observer.feed(chunk)
                                elif len(nonstream) < MAX_EVENT_BYTES:
                                    nonstream.extend(chunk)
                                self.wfile.write(chunk)
                                self.wfile.flush()
                            if not document.get("stream"):
                                try:
                                    result = json.loads(nonstream)
                                    observer.usage = numeric_usage(result)
                                    if result.get("status") in {"completed", "failed", "incomplete", "cancelled"}:
                                        observer.response_status = result["status"]
                                except (ValueError, UnicodeDecodeError, AttributeError):
                                    pass
                except (requests.RequestException, OSError):
                    receipt["transport_error"] = True
                    if not response_started:
                        self.send_json(502, {"error": {"message": "OpenRouter transport failed"}})
                finally:
                    receipt.update(elapsed_seconds=round(time.monotonic() - started, 3),
                                   response_status=observer.response_status, usage=observer.usage)
                    owner.receipt(receipt)
        return Handler

    @property
    def port(self):
        return self.server.server_address[1]

    def start(self):
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        if self.thread:
            self.thread.join(timeout=5)
        with self.log_lock:
            self.log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--relay-token-file", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19413)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--provider", default="alibaba")
    parser.add_argument("--effort", default="max", choices=["max", "xhigh", "high", "medium", "low"])
    args = parser.parse_args()
    proxy = OpenRouterResponsesProxy(api_key=load_openrouter_key(args.key_file),
        relay_token=load_or_create_relay_token(args.relay_token_file), log_file=args.log_file,
        model=args.model, provider=args.provider, effort=args.effort, host=args.host, port=args.port)
    stopped = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopped.set())
    proxy.start()
    print(json.dumps({"ready": True, "port": proxy.port, "model": args.model,
                      "provider": args.provider, "reasoning_effort": args.effort}), flush=True)
    try:
        stopped.wait()
    finally:
        proxy.stop()


if __name__ == "__main__":
    main()
