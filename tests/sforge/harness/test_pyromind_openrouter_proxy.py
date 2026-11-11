import json
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading

import pytest
import requests

from sforge.harness.relay.openrouter_proxy import (
    OpenRouterResponsesProxy, UsageObserver, load_openrouter_key,
    load_or_create_relay_token, numeric_usage, pin_request,
)


def test_key_file_bare_and_env(tmp_path):
    path = tmp_path / "key"
    path.write_text("sk-or-dummy\n")
    path.chmod(0o600)
    assert load_openrouter_key(path) == "sk-or-dummy"
    path.write_text("OPENROUTER_API_KEY='sk-or-dummy'\n")
    assert load_openrouter_key(path) == "sk-or-dummy"
    path.chmod(0o644)
    with pytest.raises(ValueError, match="0600"):
        load_openrouter_key(path)


def test_relay_token_private_idempotent(tmp_path):
    path = tmp_path / "private" / "token"
    token = load_or_create_relay_token(path)
    assert len(token) >= 48
    assert path.stat().st_mode & 0o777 == 0o600
    assert load_or_create_relay_token(path) == token


def test_pin_preserves_history_and_tools_enforces_routing():
    document = {"model": "wrong", "models": ["other"], "input": [{"type": "reasoning", "id": "opaque"}],
                "tools": [{"type": "function", "name": "exec"}], "stream": True,
                "provider": {"only": ["wrong"], "allow_fallbacks": True},
                "reasoning": {"effort": "low", "max_tokens": 200, "summary": "auto"}}
    result = pin_request(document, "selected", "alibaba", "max")
    assert result["model"] == "selected" and "models" not in result
    assert result["provider"] == {"only": ["alibaba"], "allow_fallbacks": False}
    assert result["reasoning"] == {"effort": "max", "enabled": True, "exclude": False, "summary": "auto"}
    assert result["input"] == document["input"] and result["tools"] == document["tools"]
    assert document["reasoning"]["effort"] == "low"


def test_usage_receipt_excludes_text_and_booleans():
    assert numeric_usage({"usage": {"input_tokens": 2, "secret": "dummy", "is_byok": False,
                                   "output_tokens_details": {"reasoning_tokens": 3, "text": "secret"}}}) == {
        "input_tokens": 2, "output_tokens_details": {"reasoning_tokens": 3}}


def test_stream_observer_split_lines_and_content_ignored():
    observer = UsageObserver()
    event = b'data: {"type":"response.completed","response":{"status":"completed","output":"private text","usage":{"input_tokens":3}}}\n\n'
    for chunk in (event[:19], event[19:71], event[71:]):
        observer.feed(chunk)
    assert observer.usage == {"input_tokens": 3}
    assert observer.response_status == "completed"


@pytest.fixture
def relay(tmp_path):
    records = []
    class Upstream(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"
        def log_message(self, *_):
            pass
        def do_POST(self):
            document = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            records.append({"headers": dict(self.headers), "body": document})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream" if document.get("stream") else "application/json")
            self.end_headers()
            response = {"object": "response", "status": "completed", "output": [{"private": "text"}],
                        "usage": {"input_tokens": 3, "output_tokens": 5, "secret": "upstream-key"}}
            payload = ("data: " + json.dumps({"type": "response.completed", "response": response}) + "\n\n").encode() if document.get("stream") else json.dumps(response).encode()
            self.wfile.write(payload)
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    proxy = OpenRouterResponsesProxy(api_key="upstream-key", relay_token="worker-token", log_file=tmp_path / "usage.jsonl",
                                    port=0, upstream_base=f"http://127.0.0.1:{upstream.server_address[1]}")
    proxy.start()
    yield proxy, records
    proxy.stop()
    upstream.shutdown()
    upstream.server_close()


@pytest.mark.parametrize("stream", [False, True])
def test_real_http_transport_auth_routing_streaming_and_receipt(relay, stream):
    proxy, records = relay
    url = f"http://127.0.0.1:{proxy.port}/v1/responses"
    assert requests.post(url, json={"input": "hi"}, timeout=3).status_code == 401
    response = requests.post(url, headers={"Authorization": "Bearer worker-token", "X-Secret": "do-not-forward"},
                             json={"model": "wrong", "input": "private prompt", "stream": stream}, timeout=3)
    assert response.status_code == 200
    assert records[0]["headers"]["Authorization"] == "Bearer upstream-key"
    assert "X-Secret" not in records[0]["headers"]
    assert records[0]["body"]["model"] == "deepseek/deepseek-v4.1-flash"
    assert records[0]["body"]["reasoning"]["effort"] == "max"
    proxy.log.flush()
    receipts = [json.loads(line) for line in proxy.log_file.read_text().splitlines()]
    assert receipts[0]["usage"] == {"input_tokens": 3, "output_tokens": 5}
    for secret in ("upstream-key", "worker-token", "private prompt", "private text"):
        assert secret not in proxy.log_file.read_text()
    assert receipts[0]["response_status"] == "completed"


def test_unknown_endpoint_not_forwarded(relay):
    proxy, records = relay
    url = f"http://127.0.0.1:{proxy.port}/v1/responses/compact"
    response = requests.post(url, headers={"Authorization": "Bearer worker-token"}, json={"input": "hi"}, timeout=3)
    assert response.status_code == 404 and records == []


def test_codex_gzip_request_preserves_tools(relay):
    proxy, records = relay
    payload = {"input": "private prompt", "tools": [{"name": "exec_command", "type": "function"}]}
    response = requests.post(f"http://127.0.0.1:{proxy.port}/v1/responses",
        headers={"Authorization": "Bearer worker-token", "Content-Encoding": "gzip"},
        data=gzip.compress(json.dumps(payload).encode()), timeout=3)
    assert response.status_code == 200
    assert records[0]["body"]["tools"] == payload["tools"]
