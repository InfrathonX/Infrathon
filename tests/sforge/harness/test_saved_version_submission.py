from __future__ import annotations

import json

import pytest

from sforge.harness import judge_server
from sforge.harness.version_store import VersionStore

from test_version_store_api import (
    _TOKEN_HEADER, _archive, _client, _post_checkpoint, _task,
)


def _submit_reference(client, token, receipt, *, kind="agent", **extra):
    headers = {_TOKEN_HEADER: token, "X-SForge-Submission-Kind": kind}
    if kind == "auto":
        headers[judge_server.ADMIN_SECRET_HEADER] = judge_server.ADMIN_SECRET
    return client.post(
        "/api/v1/submit",
        json={
            "version_id": receipt["version_id"],
            "archive_sha256": receipt["archive_sha256"],
            **extra,
        },
        headers=headers,
    )


@pytest.mark.parametrize("kind", ["agent", "public", "auto"])
def test_saved_submission_reuses_exact_archive_and_original_checkpoint(monkeypatch, tmp_path, kind):
    persisted = []
    client, state = _client(monkeypatch, tmp_path, version_persister=persisted.append)
    token = state.register_session("task_alpha", "saved-run")
    archive = _archive(("submission/main.py", b"the saved code\n"))
    receipt = _post_checkpoint(client, token, "original", archive).json()
    original = state.version_store.list_versions()[0]
    original_manifest = (original.path / "manifest.json").read_bytes()
    evaluated = []

    def grade(submission_id, task_id, received, run_id, round_id, *args):
        evaluated.append(received)
        assert persisted == [original, original]
        state.submissions[submission_id]["status"] = judge_server.SubmissionStatus.COMPLETED
        state._record_submission(run_id, submission_id, task_id, round_id, {"score": 42}, None)

    monkeypatch.setattr(state, "_grade_worker_serial", grade)
    response = _submit_reference(client, token, receipt, kind=kind)
    assert response.status_code == 200, response.text
    assert response.json()["version_id"] == original.version_id
    assert response.json()["archive_sha256"] == original.archive_sha256
    assert response.json()["round_id"] == f"{kind}-1"
    assert evaluated == [archive]
    assert state.version_store.list_versions() == [original]
    assert (original.path / "manifest.json").read_bytes() == original_manifest
    history = state.get_run_history("saved-run", "task_alpha")["entries"][0]
    assert history["version_id"] == original.version_id
    assert history["archive_sha256"] == original.archive_sha256
    assert history["archive_size"] == len(archive)
    assert history["round"] == f"{kind}-1"
    assert state.resolve_token(token)["agent_inflight"] is False
    assert state.resolve_token(token)["public_inflight"] is False


def test_reference_lookup_has_no_side_effects_and_hides_paths(monkeypatch, tmp_path):
    client, state = _client(monkeypatch, tmp_path)
    token = state.register_session("task_alpha", "saved-run", max_agent_submissions=1)
    receipt = _post_checkpoint(client, token, "original", _archive(("submission/a", b"a"))).json()
    before = dict(state.resolve_token(token))
    response = client.get(f"/api/v1/versions/{receipt['version_id']}", headers={_TOKEN_HEADER: token})
    assert response.status_code == 200
    assert response.json() == {key: receipt[key] for key in (
        "version_id", "archive_sha256", "archive_size", "candidate_id", "content_sha256",
    )}
    assert state.resolve_token(token) == before
    assert state.submissions == {}
    assert str(tmp_path) not in response.text


@pytest.mark.parametrize("foreign", ["run", "task", "checkpoint-token", "unknown-token"])
def test_reference_never_crosses_token_scope(monkeypatch, tmp_path, foreign):
    client, state = _client(monkeypatch, tmp_path)
    token = state.register_session("task_alpha", "saved-run")
    receipt = _post_checkpoint(client, token, "original", _archive(("submission/a", b"a"))).json()
    if foreign == "run":
        stranger = state.register_session("task_alpha", "another-run")
        expected = 404
    elif foreign == "task":
        directory = tmp_path / "store/tasks/task-beta"
        (directory / "runs").mkdir(parents=True)
        (directory / "task.json").write_text(json.dumps({
            "schema_version": 1, "task_directory": "task-beta", "task_id": "task_beta",
        }))
        state.version_store = VersionStore.open(tmp_path / "store")
        state.tasks["task_beta"] = _task()
        stranger = state.register_session("task_beta", "saved-run")
        expected = 404
    elif foreign == "checkpoint-token":
        stranger = state.register_checkpoint_token(token)
        expected = 403
    else:
        stranger = "not-a-valid-token"
        expected = 401
    reference = client.get(f"/api/v1/versions/{receipt['version_id']}", headers={_TOKEN_HEADER: stranger})
    submitted = _submit_reference(client, stranger, receipt)
    assert reference.status_code == expected, reference.text
    assert submitted.status_code == expected, submitted.text
    assert state.submissions == {}


@pytest.mark.parametrize("mutation", ["sha", "archive", "symlink", "contract"])
def test_reference_integrity_failure_precedes_round_consumption(monkeypatch, tmp_path, mutation):
    client, state = _client(monkeypatch, tmp_path)
    token = state.register_session("task_alpha", "saved-run")
    receipt = _post_checkpoint(client, token, "original", _archive(("submission/a", b"a"))).json()
    saved = state.version_store.list_versions()[0]
    if mutation == "sha":
        receipt["archive_sha256"] = "f" * 64
        expected = 400
    elif mutation == "archive":
        saved.archive_path.write_bytes(b"corrupted")
        expected = 503
    elif mutation == "symlink":
        real = tmp_path / "untrusted.tar.gz"
        saved.archive_path.rename(real)
        saved.archive_path.symlink_to(real)
        expected = 503
    else:
        state.tasks["task_alpha"].submit_paths = ["different-path"]
        expected = 400
    response = _submit_reference(client, token, receipt)
    assert response.status_code == expected, response.text
    assert state.resolve_token(token)["next_agent"] == 1
    assert state.resolve_token(token)["agent_inflight"] is False
    assert state.submissions == {}
    assert state._archive_upload_inflight == set()


def test_reference_storage_failure_does_not_queue_or_duplicate_saved_archive(monkeypatch, tmp_path):
    unavailable = [False]

    def persist(saved):
        if unavailable[0]:
            raise ConnectionError("private-storage-details")

    client, state = _client(monkeypatch, tmp_path, version_persister=persist)
    token = state.register_session("task_alpha", "saved-run")
    receipt = _post_checkpoint(client, token, "original", _archive(("submission/a", b"a"))).json()
    unavailable[0] = True
    response = _submit_reference(client, token, receipt)
    assert response.status_code == 503
    assert "private-storage-details" not in response.text
    assert state.submissions == {}
    assert state.resolve_token(token)["agent_inflight"] is False
    assert state.resolve_token(token)["next_agent"] == 2
    assert len(state.version_store.list_versions()) == 1
    unavailable[0] = False

    def grade(submission_id, *args):
        state.submissions[submission_id]["status"] = judge_server.SubmissionStatus.COMPLETED

    monkeypatch.setattr(state, "_grade_worker_serial", grade)
    retried = _submit_reference(client, token, receipt)
    assert retried.status_code == 200
    assert retried.json()["round_id"] == "agent-2"
    assert retried.json()["version_id"] == receipt["version_id"]
    assert len(state.submissions) == 1
    assert len(state.version_store.list_versions()) == 1


@pytest.mark.parametrize("value", [
    {}, [], None,
    {"version_id": "v1", "archive_sha256": "a" * 64, "host_path": "/other/run"},
    {"version_id": "../private", "archive_sha256": "a" * 64},
    {"version_id": "v1", "archive_sha256": "bad"},
])
def test_reference_json_has_strict_schema(monkeypatch, tmp_path, value):
    client, state = _client(monkeypatch, tmp_path)
    token = state.register_session("task_alpha", "saved-run")
    response = client.post("/api/v1/submit", content=json.dumps(value), headers={
        "Content-Type": "application/json", _TOKEN_HEADER: token,
    })
    assert response.status_code == 400, response.text
    assert state.resolve_token(token)["next_agent"] == 1
    assert state.submissions == {}


@pytest.mark.parametrize("admission", ["invalid-token", "busy", "budget", "cooldown", "oversized"])
def test_reference_admission_rejects_before_reading_body(monkeypatch, tmp_path, admission):
    client, state = _client(monkeypatch, tmp_path)
    token = state.register_session("task_alpha", "saved-run", max_agent_submissions=1, submission_cooldown=60)
    info = state.tokens[token]
    headers = {"Content-Type": "application/json", _TOKEN_HEADER: token}
    if admission == "invalid-token":
        headers[_TOKEN_HEADER] = "invalid"
        expected = 401
    elif admission == "busy":
        info["agent_inflight"] = True
        expected = 409
    elif admission == "budget":
        info["next_agent"] = 2
        expected = 429
    elif admission == "cooldown":
        info["last_agent_submit_at"] = judge_server.time.monotonic()
        expected = 429
    else:
        headers["Content-Length"] = "4097"
        expected = 413
    reads = []

    def body():
        reads.append(True)
        yield b"{}"

    response = client.post("/api/v1/submit", content=body(), headers=headers)
    assert response.status_code == expected, response.text
    assert reads == []
    assert state.submissions == {}
