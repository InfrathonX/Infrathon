"""Controller-side, model-pinned inference relay for an experiment matrix.

Only a revocable relay bearer reaches Workers. Upstream credentials and request
bodies stay out of receipts. Native Responses and Chat conversion share one
bounded HTTP service; they do not change a task's images or scoring contract.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import io
import json
import os
from pathlib import Path
import re
import signal
import threading
import time
import uuid
import zlib

import requests

from sforge.harness.agent_config import read_private_file
from sforge.harness.relay.openrouter_proxy import (
    MAX_REQUEST_BYTES, UsageObserver, load_or_create_relay_token, numeric_usage,
)

# requests' first timeout bounds both the connect and sending the request body.
# Long agent conversations post multi-MB bodies, and the route to the provider
# can upload at only ~10-50 KB/s, so 20 s aborted them mid-write.
UPSTREAM_TIMEOUT = (300, 600)


def load_provider_key(path: Path) -> str:
    """Accept a bare token or recognized dotenv field, without printing either."""
    data = read_private_file(path).decode().strip()
    if data and '\n' not in data and not re.match(r'[A-Za-z_][A-Za-z0-9_]*=', data):
        return data
    from dotenv import dotenv_values
    values = dotenv_values(stream=io.StringIO(data), interpolate=False)
    matches = [values[k].strip() for k in (
        'OPENROUTER_API_KEY', 'DASHSCOPE_API_KEY', 'ALICLOUD_API_KEY',
        'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'API_KEY',
    ) if isinstance(values.get(k), str) and values[k].strip()]
    if len(set(matches)) != 1:
        raise ValueError('Credential file must contain exactly one recognized provider key')
    return matches[0]


def pin_native_request(document: dict, model: str, effort: str, vendor: str,
                       provider: str | None = None) -> dict:
    if not isinstance(document, dict):
        raise ValueError('Responses request must be an object')
    result = dict(document)
    result.update(model=model, store=False)
    result.pop('models', None)
    reasoning = dict(result.get('reasoning') or {})
    reasoning.pop('max_tokens', None)
    reasoning['effort'] = effort
    if vendor == 'openrouter':
        reasoning.update(enabled=True, exclude=False)
        result.pop('provider', None)
        if provider:
            # One upstream for the whole experiment; no silent fallback.
            result['provider'] = {'only': [provider], 'allow_fallbacks': False}
    result['reasoning'] = reasoning
    # This controller keeps full Responses input history; upstream state must
    # never make one experiment depend on another's conversation.
    result.pop('previous_response_id', None)
    return result


class ModelResponsesProxy:
    def __init__(self, *, api_key: str, relay_token: str, model: str,
                 upstream_base: str, log_file: Path, vendor: str,
                 effort: str = 'max', wire: str = 'responses',
                 host: str = '127.0.0.1', port: int = 0, provider: str | None = None):
        if not all((api_key, relay_token, model)) or wire not in ('responses', 'chat'):
            raise ValueError('Explicit model, credentials, and a supported wire are required')
        if provider is not None and vendor != 'openrouter':
            raise ValueError('Provider pinning applies only to OpenRouter')
        self.api_key, self.relay_token, self.model = api_key, relay_token, model
        self.upstream_base = upstream_base.rstrip('/')
        self.vendor, self.effort, self.wire = vendor, effort, wire
        self.provider = provider
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.log_file, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
        os.fchmod(descriptor, 0o600)
        self.log = os.fdopen(descriptor, 'a')
        self.log_lock = threading.Lock()
        self.active_requests = 0
        self.closing = False
        self.server = ThreadingHTTPServer((host, port), self._handler())
        self.server.daemon_threads = True
        self.thread = None

    def receipt(self, value):
        with self.log_lock:
            self.log.write(json.dumps(value, separators=(',', ':')) + '\n')
            self.log.flush()

    def _handler(self):
        owner = self
        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.0'

            def log_message(self, *_):
                pass

            def send_json(self, status, document):
                payload = json.dumps(document).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def authorized(self):
                return hmac.compare_digest(self.headers.get('Authorization', ''),
                                           'Bearer ' + owner.relay_token)

            def do_GET(self):
                if self.path == '/health':
                    self.send_json(200, {'status': 'ready', 'model': owner.model, 'wire': owner.wire})
                elif not self.authorized():
                    self.send_json(401, {'error': {'message': 'Invalid relay bearer'}})
                elif self.path == '/v1/models':
                    self.send_json(200, {'object': 'list', 'data': [{'id': owner.model, 'object': 'model'}]})
                else:
                    self.send_json(404, {'error': {'message': 'Unsupported relay endpoint'}})

            def do_POST(self):
                if not self.authorized():
                    self.send_json(401, {'error': {'message': 'Invalid relay bearer'}})
                    return
                if self.path != '/v1/responses':
                    self.send_json(404, {'error': {'message': 'Unsupported relay endpoint'}})
                    return
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                    if not 0 < length <= MAX_REQUEST_BYTES or self.headers.get('Transfer-Encoding'):
                        raise ValueError('Invalid bounded request')
                    self.connection.settimeout(30)
                    raw = self.rfile.read(length)
                    if len(raw) != length:
                        raise ValueError('Incomplete request')
                    encoding = self.headers.get('Content-Encoding', 'identity').lower()
                    if encoding == 'gzip':
                        decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
                        raw = decoder.decompress(raw, MAX_REQUEST_BYTES + 1)
                        if len(raw) > MAX_REQUEST_BYTES or decoder.unconsumed_tail or not decoder.eof:
                            raise ValueError('Invalid compressed request')
                    elif encoding != 'identity':
                        raise ValueError('Unsupported encoding')
                    request = pin_native_request(json.loads(raw), owner.model, owner.effort, owner.vendor,
                                                 owner.provider)
                except (ValueError, OSError, UnicodeDecodeError, zlib.error):
                    self.send_json(400, {'error': {'message': 'Invalid bounded Responses request'}})
                    return
                with owner.log_lock:
                    if owner.closing:
                        self.send_json(503, {'error': {'message': 'Relay is stopping'}})
                        return
                    owner.active_requests += 1
                started = time.monotonic()
                receipt = {'timestamp': datetime.now(timezone.utc).isoformat(),
                           'request_id': uuid.uuid4().hex, 'model': owner.model,
                           'vendor': owner.vendor, 'provider': owner.provider, 'wire': owner.wire,
                           'reasoning_effort': owner.effort, 'status_code': None}
                observer = UsageObserver()
                response_started = False
                try:
                    with requests.Session() as session:
                        session.trust_env = False
                        headers = {'Authorization': 'Bearer ' + owner.api_key, 'Content-Type': 'application/json'}
                        if owner.wire == 'chat':
                            from sforge.harness.relay.chat_responses_bridge import (
                                request_to_chat, chat_to_response, response_sse_events,
                            )
                            body = request_to_chat(request, owner.model, owner.effort)
                            with ThreadPoolExecutor(max_workers=1) as executor:
                                pending = executor.submit(session.post, owner.upstream_base + '/chat/completions',
                                                          headers=headers, json=body, timeout=UPSTREAM_TIMEOUT)
                                if request.get('stream'):
                                    self.send_response(200)
                                    self.send_header('Content-Type', 'text/event-stream')
                                    self.end_headers()
                                    response_started = True
                                while True:
                                    try:
                                        response = pending.result(timeout=10)
                                        break
                                    except FutureTimeout:
                                        if response_started:
                                            self.wfile.write(b': inference pending\n\n')
                                            self.wfile.flush()
                                receipt['status_code'] = response.status_code
                                response.raise_for_status()
                                document = chat_to_response(response.json(), request, owner.model)
                            observer.usage = numeric_usage(document)
                            observer.response_status = document.get('status')
                            if request.get('stream'):
                                for event in response_sse_events(document):
                                    self.wfile.write(('event: ' + event['type'] + '\ndata: ' + json.dumps(event) + '\n\n').encode())
                                self.wfile.flush()
                            else:
                                self.send_json(200, document)
                        else:
                            with session.post(owner.upstream_base + '/responses', headers=headers,
                                              json=request, stream=True, timeout=UPSTREAM_TIMEOUT) as response:
                                receipt['status_code'] = response.status_code
                                if response.status_code >= 400:
                                    self.send_json(response.status_code, {'error': {'message': 'Pinned provider rejected request',
                                                                                  'type': 'provider_error'}})
                                    return
                                self.send_response(response.status_code)
                                self.send_header('Content-Type', response.headers.get('Content-Type', 'application/json'))
                                self.end_headers()
                                response_started = True
                                buffer = bytearray()
                                for chunk in response.iter_content(chunk_size=None):
                                    if not chunk:
                                        continue
                                    if request.get('stream'):
                                        observer.feed(chunk)
                                    elif len(buffer) + len(chunk) <= MAX_REQUEST_BYTES:
                                        buffer.extend(chunk)
                                    self.wfile.write(chunk)
                                    self.wfile.flush()
                                if not request.get('stream'):
                                    document = json.loads(buffer)
                                    observer.usage = numeric_usage(document)
                                    observer.response_status = document.get('status')
                except (requests.RequestException, OSError, ValueError, KeyError) as exc:
                    receipt['transport_error'] = True
                    receipt['error_type'] = type(exc).__name__
                    error = {'error': {'message': 'Pinned provider request failed', 'type': 'provider_error'}}
                    if not response_started:
                        self.send_json(502, error)
                    else:
                        try:
                            self.wfile.write(('event: error\ndata: ' + json.dumps({'type': 'error', **error}) + '\n\n').encode())
                            self.wfile.flush()
                        except OSError:
                            pass
                finally:
                    receipt.update(elapsed_seconds=round(time.monotonic() - started, 3),
                                   usage=observer.usage, response_status=observer.response_status)
                    owner.receipt(receipt)
                    with owner.log_lock:
                        owner.active_requests -= 1
                        if owner.closing and owner.active_requests == 0:
                            owner.log.close()
        return Handler

    @property
    def port(self):
        return self.server.server_address[1]

    def start(self):
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        with self.log_lock:
            self.closing = True
        self.server.shutdown()
        self.server.server_close()
        if self.thread:
            self.thread.join(timeout=5)
        with self.log_lock:
            # In-flight handlers close this descriptor after their final usage
            # receipt. Shutdown remains bounded rather than waiting600seconds.
            if self.active_requests == 0:
                self.log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True, help='Private JSON; credential file paths only')
    args = parser.parse_args()
    config = json.loads(read_private_file(args.config))
    proxy = ModelResponsesProxy(api_key=load_provider_key(Path(config.pop('key_file'))),
        relay_token=load_or_create_relay_token(Path(config.pop('relay_token_file'))), **config)
    stopped = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopped.set())
    proxy.start()
    print(json.dumps({'ready': True, 'port': proxy.port, 'model': proxy.model, 'wire': proxy.wire}), flush=True)
    try:
        stopped.wait()
    finally:
        proxy.stop()


if __name__ == '__main__':
    main()
