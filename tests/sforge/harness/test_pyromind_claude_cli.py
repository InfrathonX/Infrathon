"""Exercise the pinned real Claude CLI against a local Anthropic stand-in.

This checks CLI/hook scheduling, not provider credentials or a task GPU score.
No request reaches an external model service and the only credential is a fake
test string. The test is skipped on CI hosts without the explicitly pinned CLI.
"""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sforge.harness.agent import claude_agent as CLAUDE
from sforge.harness.agent_config import AgentCredentials
from sforge.harness import submit_wait


# the sibling agent-runtime store of this checkout (launch/build-agent-runtime.sh), unless named
RUNTIME = Path(os.environ.get("SFORGE_TEST_CLAUDE_RUNTIME",
    Path(__file__).resolve().parents[3].parent / "agent_runtimes/claude-code-2.1.287"))


@pytest.mark.skipif(not (RUNTIME / "bin/claude").exists(), reason="Pinned Claude 2.1.287 runtime unavailable")
def test_real_probe_stream_is_parsed_by_worker_preflight_with_exact_model(tmp_path, monkeypatch):
    """Exercise probe=True, which differs from a normal CLI tool round trip.

    --verbose --output-format json produces a single JSON array on this CLI;
    the production Worker parser needs stream-json records instead. Validate
    real serialization using that exact production parser, with synthetic auth.
    """
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *ignored):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if "count_tokens" in self.path:
                payload = json.dumps({"input_tokens": 20}).encode()
                content_type = "application/json"
            elif "/messages" in self.path:
                calls.append(body)
                events = [
                    ("message_start", {"type": "message_start", "message": {
                        "id": "msg_probe", "type": "message", "role": "assistant",
                        "model": body["model"], "content": [], "stop_reason": None,
                        "stop_sequence": None, "usage": {"input_tokens": 20, "output_tokens": 0}}}),
                    ("content_block_start", {"type": "content_block_start", "index": 0,
                        "content_block": {"type": "text", "text": ""}}),
                    ("content_block_delta", {"type": "content_block_delta", "index": 0,
                        "delta": {"type": "text_delta", "text": "READY"}}),
                    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                    ("message_delta", {"type": "message_delta", "delta": {
                        "stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 5}}),
                    ("message_stop", {"type": "message_stop"}),
                ]
                payload = "".join("event: " + name + "\ndata: " + json.dumps(value) + "\n\n"
                                  for name, value in events).encode()
                content_type = "text/event-stream"
            else:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    home = tmp_path / "home"
    home.mkdir()
    settings = tmp_path / "settings.json"
    monkeypatch.setattr(CLAUDE, "CLAUDE_SETTINGS_PATH", str(settings))
    model = {"AUTH_MODE": "anthropic-api-key", "MODEL": "claude-opus-5-5",
             "MODEL_API_BASE_URL": "http://127.0.0.1:" + str(server.server_port),
             "REASONING_EFFORT": "max"}
    settings.write_text(json.dumps(CLAUDE.claude_settings(model)))
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("CLAUDE", "ANTHROPIC"))}
    environment.update(HOME=str(home), ANTHROPIC_API_KEY="fake-local-probe-key",
        ANTHROPIC_BASE_URL=model["MODEL_API_BASE_URL"], CLAUDE_CODE_EFFORT_LEVEL="max",
        CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1", DISABLE_AUTOUPDATER="1",
        DISABLE_TELEMETRY="1", DISABLE_ERROR_REPORTING="1",
        PATH=str(RUNTIME / "bin") + os.pathsep + environment["PATH"])
    try:
        command = CLAUDE.claude_command(model, "Reply with exactly READY. Do not use tools or change files.", probe=True)
        command[0] = str(RUNTIME / "bin/claude")
        result = subprocess.run(command, cwd=tmp_path, env=environment, text=True,
                                capture_output=True, timeout=90)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert result.returncode == 0, result.stderr[-1500:]
    assert len(calls) == 1 and calls[0]["model"] == model["MODEL"]
    assert calls[0].get("tools", []) == []

    # Obtain the actual generated Worker preflight from the installer. Execute
    # only its result-validation nodes with the real CLI's serialization; use
    # the OAuth branch to also enforce exact modelUsage without real OAuth.
    commands = []
    CLAUDE.install_and_validate_agent(Mock(), object(), model,
        SimpleNamespace(agent="claude-code", root=RUNTIME),
        AgentCredentials("anthropic-api-key", b"fake-local-probe-key", frozenset({"fake-local-probe-key"})),
        lambda _backend, _worker, command, **_kwargs: commands.append(command) or "OK")
    generated = next(command[2] for command in commands if isinstance(command, list)
                     and command[:2] == ["python3", "-c"])
    tree = ast.parse(generated)
    start = next(index for index, node in enumerate(tree.body)
                 if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                    and target.id == "records" for target in node.targets))
    validator = compile(ast.Module(body=tree.body[start:], type_ignores=[]), "<worker-preflight-result>", "exec")
    namespace = {"result": result, "json": json,
                 "sys": SimpleNamespace(argv=["-c", "claude-oauth", "", "max", "unused", model["MODEL"]])}
    exec(validator, namespace)
    final = namespace["final"][0]
    assert final["result"].strip() == "READY" and set(final["modelUsage"]) == {model["MODEL"]}
    for change in ({"subtype": "error_during_execution"}, {"is_error": True},
                   {"result": "NOT READY"}, {"modelUsage": {"different-model": {}}}):
        with pytest.raises(AssertionError):
            exec(validator, {**namespace, "result": SimpleNamespace(returncode=0,
                stdout=json.dumps({**final, **change}))})


@pytest.mark.skipif(not (RUNTIME / "bin/claude").exists(), reason="Pinned Claude 2.1.287 runtime unavailable")
def test_real_cli_failed_tool_then_synchronous_submit(tmp_path, monkeypatch):
    engine, adapter = tmp_path / "engine.py", tmp_path / "claude-hook.py"
    engine.write_text(submit_wait.generate_submit_wait_hook_script())
    monkeypatch.setattr(submit_wait, "SUBMIT_WAIT_HOOK_PATH", str(engine))
    adapter.write_text(CLAUDE.generate_claude_hook_script())
    monkeypatch.setattr(CLAUDE, "CLAUDE_HOOK_PATH", str(adapter))
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(CLAUDE, "CLAUDE_SETTINGS_PATH", str(settings_path))
    binary = tmp_path / "bin"
    binary.mkdir()
    marker, completed = tmp_path / "submit-started", tmp_path / "submit-completed"
    submit = binary / "sforge-submit"
    submit.write_text("#!" + sys.executable + "\nimport pathlib,time\n"
        + "pathlib.Path(" + repr(str(marker)) + ").write_text('started')\n"
        + "time.sleep(0.7)\n"
        + "pathlib.Path(" + repr(str(completed)) + ").write_text('completed')\n"
        + "print('Score: 180.0')\n")
    submit.chmod(0o755)
    calls, violations = [], []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *ignored):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if "count_tokens" in self.path:
                payload = json.dumps({"input_tokens": 20}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if "/messages" not in self.path:
                self.send_response(404)
                self.end_headers()
                return
            index = len(calls)
            calls.append({"model": request["model"], "time": time.monotonic(),
                          "messages": request["messages"]})
            if index >= 2 and not completed.exists():
                violations.append("Model advanced before native submission completed")
            tool = index < 2
            content = {"type": "tool_use", "id": "toolu_standin_" + str(index), "name": "Bash", "input": {}} if tool else {"type": "text", "text": ""}
            delta = {"type": "input_json_delta", "partial_json": json.dumps({"command": "false" if index == 0 else "sforge-submit", "description": "SForge scheduling diagnostic"})} if tool else {"type": "text_delta", "text": "READY"}
            events = [("message_start", {"type": "message_start", "message": {
                "id": "msg_standin_" + str(index), "type": "message", "role": "assistant",
                "model": request["model"], "content": [], "stop_reason": None,
                "stop_sequence": None, "usage": {"input_tokens": 20, "output_tokens": 0}}}),
                ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": content}),
                ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": delta}),
                ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                ("message_delta", {"type": "message_delta", "delta": {
                    "stop_reason": "tool_use" if tool else "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 10}}),
                ("message_stop", {"type": "message_stop"})]
            payload = "".join("event: " + name + "\ndata: " + json.dumps(value) + "\n\n" for name, value in events).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    model = {"MODEL": "claude-opus-5-5", "REASONING_EFFORT": "max",
             "MODEL_API_BASE_URL": "http://127.0.0.1:" + str(server.server_port)}
    settings_path.write_text(json.dumps(CLAUDE.claude_settings(model, submit_wait_timeout_seconds=180)))
    home = tmp_path / "home"
    home.mkdir()
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("CLAUDE", "ANTHROPIC"))}
    env.update(HOME=str(home), ANTHROPIC_API_KEY="fake-local-test-key",
        ANTHROPIC_BASE_URL=model["MODEL_API_BASE_URL"], CLAUDE_CODE_EFFORT_LEVEL="max",
        CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1", DISABLE_AUTOUPDATER="1",
        DISABLE_TELEMETRY="1", DISABLE_ERROR_REPORTING="1",
        SFORGE_SUBMIT_WAIT_STATE_DIR=str(tmp_path / "gate"),
        PATH=str(binary) + os.pathsep + str(RUNTIME / "bin") + os.pathsep + env["PATH"])
    for key in ("SFORGE_SUBMIT_TOOL_TIMEOUT_MS", "SFORGE_AGENT_DEADLINE_AT",
                "SFORGE_AGENT_DEADLINE_MONOTONIC", "SFORGE_REMAINING_SECONDS"):
        env.pop(key, None)
    try:
        command = CLAUDE.claude_command(model, "Run false, then sforge-submit, then reply READY.")
        command[0] = str(RUNTIME / "bin/claude")
        result = subprocess.run(command, cwd=tmp_path, env=env, text=True, capture_output=True, timeout=90)
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)
    assert result.returncode == 0, result.stderr[-1500:]
    records = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    final = [record for record in records if record.get("type") == "result"]
    assert final and final[-1].get("result", "").strip() == "READY", result.stdout[-2000:]
    assert not violations
    assert len(calls) == 3 and all(call["model"] == "claude-opus-5-5" for call in calls)
    assert completed.exists()
    gate = json.loads((tmp_path / "gate/state.json").read_text())
    assert gate["active"] is None and gate["active_tools"] == {}
    assert gate["last_completion"]["outcome"] == "completed"
    submit_result = [block for message in calls[-1]["messages"] if isinstance(message.get("content"), list)
                     for block in message["content"] if block.get("type") == "tool_result"]
    assert any("Score: 180.0" in str(block.get("content")) for block in submit_result)
