from __future__ import annotations

import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from sforge.harness.evolve_scripts import generate_submit_script


def _executable(path, text):
    path.write_text(text)
    path.chmod(0o755)
    return path


@pytest.mark.parametrize("pin_sha", [False, True])
def test_saved_version_cli_submits_only_reference_and_polls_result(tmp_path, pin_sha):
    receipt = {"version_id": "saved-checkpoint-1", "archive_sha256": "a" * 64}
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, body):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def do_GET(self):
            requests.append(("GET", self.path, dict(self.headers)))
            if self.path.startswith("/api/v1/versions/"):
                self.respond(receipt)
            elif self.path.startswith("/api/v1/history"):
                self.respond({"entries": [{"round": "agent-1", "status": "completed"}], "best_score": 42})
            else:
                self.respond({"status": "completed", "report": {
                    "passed": 1, "failed": 0, "total_tests": 1,
                    "pass_rate": 1, "score": 42, "valid": True, "summary": "saved bytes evaluated",
                }})

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            requests.append(("POST", self.path, dict(self.headers), json.loads(body)))
            self.respond({"submission_id": "submission-id", "round_id": "agent-1"})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    touched = tmp_path / "unexpected-workspace-access"
    _executable(fake_bin / "sleep", "#!/bin/sh\nexit 0\n")
    _executable(fake_bin / "tar", f"#!/bin/sh\ntouch '{touched}'\nexit 99\n")
    hook = _executable(tmp_path / "hook", f"#!/bin/sh\ntouch '{touched}'\nexit 98\n")
    script = generate_submit_script().replace("/tmp/sforge_state.json", str(tmp_path / "state.json"))
    script = script.replace("/tmp/_submit_resp.json", str(tmp_path / "response.json"))
    command = _executable(tmp_path / "sforge-submit", script)
    args = ["--version", receipt["version_id"]]
    if pin_sha:
        args += ["--archive-sha256", receipt["archive_sha256"]]
    try:
        result = subprocess.run([str(command), *args], text=True, capture_output=True, timeout=15, env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "SFORGE_JUDGE_URL": f"http://127.0.0.1:{server.server_port}",
            "SFORGE_TOKEN": "scoped-token",
            "SFORGE_PATCH_DIR": str(tmp_path / "does-not-exist"),
            "SFORGE_PRE_SUBMIT_HOOK": str(hook),
        })
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "saved bytes evaluated" in result.stdout
    assert not touched.exists()
    posts = [item for item in requests if item[0] == "POST"]
    assert len(posts) == 1
    assert posts[0][1] == "/api/v1/submit"
    assert posts[0][2]["Content-Type"] == "application/json"
    assert posts[0][2]["X-SForge-Token"] == "scoped-token"
    assert posts[0][3] == receipt
    lookups = [item for item in requests if item[1].startswith("/api/v1/versions/")]
    assert len(lookups) == (0 if pin_sha else 1)
    if lookups:
        assert lookups[0][2]["X-SForge-Token"] == "scoped-token"
    assert any(item[1] == "/api/v1/result/submission-id" for item in requests)


@pytest.mark.parametrize("args", [
    ["--version"], ["--version", ""], ["--version", "../other"],
    ["--version", "valid", "--list"], ["--archive-sha256", "a" * 64],
    ["--version", "valid", "--archive-sha256", "bad"],
])
def test_saved_version_cli_rejects_invalid_reference_before_any_io(tmp_path, args):
    script = _executable(tmp_path / "submit", generate_submit_script())
    result = subprocess.run([str(script), *args], text=True, capture_output=True, timeout=5, env={
        "PATH": os.environ["PATH"], "SFORGE_JUDGE_URL": "", "SFORGE_TOKEN": "",
    })
    assert result.returncode == 2
    assert "ERROR:" in result.stderr
