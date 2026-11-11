import io
import tarfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sforge.harness.config import SForgeConfig
from sforge.harness.judge_server import create_app
from sforge.harness.version_store import VersionStore
from sforge.pyromind.bootstrap import DEMO, initialize_store
from sforge.pyromind.resident_judge import evaluate_addition


def archive(code, name="solution.py", link=False):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as bundle:
        member = tarfile.TarInfo(name)
        member.size = len(code)
        if link:
            member.type = tarfile.SYMTYPE
            member.linkname = "/etc/passwd"
        bundle.addfile(member, io.BytesIO(code))
    return data.getvalue()


def test_native_http_submit_uses_resident_evaluator_and_shared_store(tmp_path, monkeypatch):
    from sforge.harness import judge_server
    monkeypatch.setattr(judge_server, "create_backend_from_config",
                        lambda _: pytest.fail("Resident Judge must not initialize Docker or platform SDK"))
    root = tmp_path / "shared" / "version-store"
    initialize_store(root, "resident-test")
    app = create_app(SForgeConfig(tasks_dir=DEMO, version_store_root=root,
        log_dir=tmp_path / "logs"), submission_evaluator=evaluate_addition)
    state = app.state.judge_state
    try:
        token = state.register_session("addition", "test", max_agent_submissions=5,
                                       submission_cooldown=0)
        with TestClient(app) as client:
            headers = {"X-SForge-Token": token, "Content-Type": "application/gzip"}
            first = archive(b"def add(a,b): return a-b\n")
            checkpoint = client.post("/api/v1/versions", content=first,
                headers={**headers, "X-SForge-Version-Label": "v1"})
            assert checkpoint.status_code == 201, checkpoint.text
            for code, score in ((first, .25), (archive(b"def add(a,b): return a+b\n"), 1.), (first, .25)):
                submission = client.post("/api/v1/submit", content=code, headers=headers)
                assert submission.status_code == 200, submission.text
                result = client.get("/api/v1/result/" + submission.json()["submission_id"]).json()
                assert result["status"] == "completed", result
                assert result["report"]["score"] == score, result
            assert len(VersionStore.open(root).list_versions()) == 4
            assert state.backend is None
    finally:
        state._reaper_stop.set()


@pytest.mark.parametrize("payload", [archive(b"x", "../solution.py"), archive(b"x", link=True)])
def test_resident_rejects_unsafe_archives(tmp_path, payload):
    from types import SimpleNamespace
    with pytest.raises(ValueError):
        evaluate_addition(task_spec=SimpleNamespace(task_id="addition"), archive=payload,
                          submission_id="bad", log_dir=tmp_path)


def test_service_bootstrap_honors_configured_registration_credential(tmp_path):
    import json
    import os
    import secrets
    import socket
    import subprocess
    import sys
    import time
    import requests

    credential = secrets.token_urlsafe(32)
    config = tmp_path / "admin.json"
    config.write_text(json.dumps({"secret": credential}))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {k: v for k, v in os.environ.items()
           if k not in {"PYROMIND_API_KEY", "SFORGE_ADMIN_SECRET"}}
    with (tmp_path / "server.log").open("w") as log:
        process = subprocess.Popen([sys.executable, "-m", "sforge.pyromind.resident_judge",
            "--run-id", "startup-test", "--shared-dir", str(tmp_path / "shared"),
            "--admin-file", str(config), "--port", str(port)], env=env,
            stdout=log, stderr=subprocess.STDOUT)
        session = requests.Session()
        session.trust_env = False
        url = f"http://127.0.0.1:{port}"
        try:
            for _ in range(100):
                if process.poll() is not None:
                    pytest.fail((tmp_path / "server.log").read_text())
                try:
                    if session.get(url + "/openapi.json", timeout=.2).status_code == 200:
                        break
                except requests.RequestException:
                    time.sleep(.05)
            body = {"task_id": "addition", "run_id": "startup-test", "admin_secret": "wrong"}
            assert session.post(url + "/api/v1/register", json=body, timeout=2).status_code == 403
            body["admin_secret"] = credential
            response = session.post(url + "/api/v1/register", json=body, timeout=2)
            assert response.status_code == 200, response.text
            assert response.json()["token"]
        finally:
            session.close()
            process.terminate()
            process.wait(timeout=5)
