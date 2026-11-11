from __future__ import annotations

import io
import json
import tarfile
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks
from fastapi.testclient import TestClient

from sforge.harness import judge_server
from sforge.harness.config import SForgeConfig
from sforge.harness.judge_server import JudgeState
from sforge.harness.version_store import VersionStore, VersionStoreWriteError


_TOKEN_HEADER = "X-SForge-Token"
_VERSION_LABEL_HEADER = "X-SForge-Version-Label"
_SUBMISSION_KIND_HEADER = "X-SForge-Submission-Kind"
_CONTENT_TYPE = "application/gzip"


def _archive(*members: tuple[str, bytes]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as bundle:
        for name, payload in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            bundle.addfile(info, io.BytesIO(payload))
    return output.getvalue()


def _store(root: Path, *, task_id: str = "task_alpha") -> Path:
    root.mkdir()
    (root / ".sforge-version-store.json").write_text(
        '{"schema_version":1,"kind":"sforge-version-store","store_id":"api-test-store"}\n',
        encoding="utf-8",
    )
    task_root = root / "tasks/task-alpha"
    (task_root / "runs").mkdir(parents=True)
    (task_root / "task.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_directory": "task-alpha",
                "task_id": task_id,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def _task() -> SimpleNamespace:
    return SimpleNamespace(
        submit_paths=["submission"],
        submit_exclude=["auth.json"],
        candidate_identity_exclude=[],
        judge=SimpleNamespace(
            public_eval_cmd="public-eval",
            selection="valid_then_score",
            score_direction="maximize",
        ),
    )


def _client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    store_task_id: str | None = "task_alpha",
    store_directory: str = "store",
    version_persister=None,
) -> tuple[TestClient, JudgeState]:
    captured: dict[str, JudgeState] = {}
    monkeypatch.setattr(
        judge_server,
        "create_backend_from_config",
        lambda config: object(),
    )
    monkeypatch.setattr(JudgeState, "_reaper_loop", lambda self: None)

    def load_tasks(state: JudgeState) -> None:
        captured["state"] = state
        state.tasks = {"task_alpha": _task()}

    monkeypatch.setattr(JudgeState, "load_tasks", load_tasks)
    store_root = None
    if store_task_id is not None:
        store_root = _store(tmp_path / store_directory, task_id=store_task_id)
    app = judge_server.create_app(
        SForgeConfig(log_dir=tmp_path / "logs", version_store_root=store_root),
        version_persister=version_persister,
    )
    return TestClient(app), captured["state"]


def _post_checkpoint(
    client: TestClient,
    token: str,
    label: str,
    archive: bytes,
):
    return client.post(
        "/api/v1/versions",
        content=archive,
        headers={
            "Content-Type": _CONTENT_TYPE,
            _TOKEN_HEADER: token,
            _VERSION_LABEL_HEADER: label,
        },
    )


def _post_submission(
    client: TestClient,
    token: str,
    archive: bytes,
    *,
    kind: str = "agent",
):
    headers = {
        "Content-Type": _CONTENT_TYPE,
        _TOKEN_HEADER: token,
        _SUBMISSION_KIND_HEADER: kind,
    }
    if kind == "auto":
        headers[judge_server.ADMIN_SECRET_HEADER] = judge_server.ADMIN_SECRET
    return client.post("/api/v1/submit", content=archive, headers=headers)


@pytest.mark.parametrize("kind", ["agent", "public", "auto"])
def test_external_persistence_finishes_before_ack_and_one_background_judge(
    monkeypatch, tmp_path, kind,
):
    events = []
    remote = {}
    archive = _archive(("submission/main.py", b"persisted before grading\n"))

    def persist(version):
        assert state.submissions == {}
        assert version.archive_path.read_bytes() == archive
        remote[version.version_id] = version.archive_path.read_bytes()
        events.append("persisted")

    client, state = _client(monkeypatch, tmp_path, version_persister=persist)
    token = state.register_session("task_alpha", "run-durable")

    def grade(submission_id, *args):
        assert len(remote) == 1
        assert events == ["persisted", "ack"]
        events.append("judge")
        state.submissions[submission_id]["status"] = judge_server.SubmissionStatus.COMPLETED

    monkeypatch.setattr(state, "_grade_worker_serial", grade)
    original_app = client.app

    async def observe_ack(scope, receive, send):
        async def observed_send(message):
            if message["type"] == "http.response.start":
                assert message["status"] == 200
                events.append("ack")
            await send(message)
        await original_app(scope, receive, observed_send)

    response = _post_submission(TestClient(observe_ack), token, archive, kind=kind)
    assert response.status_code == 200
    assert events == ["persisted", "ack", "judge"]
    assert len(state.submissions) == 1
    assert state.resolve_token(token)["agent_inflight"] is False
    assert state.resolve_token(token)["public_inflight"] is False


def test_external_storage_failure_releases_slot_keeps_archive_and_allows_retry(
    monkeypatch, tmp_path, caplog,
):
    calls = []
    now = [100.0]
    monkeypatch.setattr(judge_server.time, "monotonic", lambda: now[0])

    def persist(version):
        calls.append(version)
        if len(calls) == 1:
            raise ConnectionError("signed-url-secret-must-not-escape")

    client, state = _client(monkeypatch, tmp_path, version_persister=persist)
    token = state.register_session(
        "task_alpha", "run-storage-outage", max_agent_submissions=3,
        submission_cooldown=30,
    )
    archive = _archive(("submission/main.py", b"first\n"))
    judges = []

    def grade(submission_id, *args):
        judges.append(submission_id)
        state.submissions[submission_id]["status"] = judge_server.SubmissionStatus.COMPLETED

    monkeypatch.setattr(state, "_grade_worker_serial", grade)
    failed = _post_submission(client, token, archive)
    assert failed.status_code == 503
    assert failed.json()["detail"] == "version store write failed"
    assert state.submissions == {}
    assert judges == []
    assert state._archive_upload_inflight == set()
    token_info = state.resolve_token(token)
    assert token_info["agent_inflight"] is False
    # Existing admission policy consumes a round/cooldown before store writes.
    assert token_info["next_agent"] == 2
    assert token_info["last_agent_submit_at"] == 100.0
    stored = state.version_store.list_versions()
    assert len(stored) == 1
    assert stored[0].archive_path.read_bytes() == archive
    assert stored[0].version_id in caplog.text
    assert "round=agent-1" in caplog.text
    assert "ConnectionError" in caplog.text
    assert "signed-url-secret-must-not-escape" not in caplog.text
    assert _post_submission(client, token, archive).status_code == 429
    now[0] += 31
    second_archive = _archive(("submission/main.py", b"second\n"))
    accepted = _post_submission(client, token, second_archive)
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["round_id"] == "agent-2"
    assert accepted.json()["remaining_submissions"] == 1
    assert len(judges) == 1
    assert len(state.submissions) == 1
    assert state.resolve_token(token)["agent_inflight"] is False
    assert [v.round_id for v in state.version_store.list_versions()] == ["agent-1", "agent-2"]


def test_checkpoint_external_storage_failure_has_no_judge_and_can_save_again(
    monkeypatch, tmp_path,
):
    calls = []

    def persist(version):
        calls.append(version)
        if len(calls) == 1:
            raise TimeoutError("storage temporarily unreachable")

    client, state = _client(monkeypatch, tmp_path, version_persister=persist)
    monkeypatch.setattr(judge_server, "CHECKPOINT_MIN_INTERVAL_SECONDS", 0)
    token = state.register_session("task_alpha", "run-checkpoint-outage")
    archive = _archive(("submission/main.py", b"checkpoint\n"))
    failed = _post_checkpoint(client, token, "first", archive)
    assert failed.status_code == 503
    assert state._checkpoint_inflight == state._archive_upload_inflight == set()
    assert state.submissions == {}
    assert state.resolve_token(token)["next_agent"] == 1
    assert state.resolve_token(token)["agent_inflight"] is False
    assert calls[0].archive_path.read_bytes() == archive
    accepted = _post_checkpoint(client, token, "retry", archive)
    assert accepted.status_code == 201
    assert accepted.json()["round_id"] == "checkpoint-2"
    assert len(state.version_store.list_versions()) == 2
    assert state.submissions == {}


def test_artifact_mirror_network_fault_never_commits_a_judge_queue(monkeypatch, tmp_path):
    from sforge.pyromind.runner import ArtifactMirror

    class Storage:
        fail_once = True

        def __init__(self):
            self.objects = {}

        def put_object(self, bucket, key, stream, size):
            if self.fail_once and key.endswith("submission.tar.gz"):
                self.fail_once = False
                raise ConnectionError("storage put failed")
            data = stream.read()
            assert len(data) == size
            self.objects[key] = data

    storage = Storage()
    mirror = ArtifactMirror(tmp_path, "run-artifacts", client=storage, bucket="test")
    client, state = _client(
        monkeypatch, tmp_path, store_directory="version-store",
        version_persister=lambda version: mirror.sync(),
    )
    token = state.register_session("task_alpha", "run-mirror-outage")
    first_archive = _archive(("submission/main.py", b"first\n"))
    failed = _post_submission(client, token, first_archive)
    assert failed.status_code == 503
    assert state.submissions == {}
    assert state.resolve_token(token)["agent_inflight"] is False
    assert not any(key.endswith("submission.tar.gz") for key in storage.objects)
    assert not any(key.endswith("submission.tar.gz") for key in mirror.seen)
    versions = state.version_store.list_versions()
    assert len(versions) == 1
    assert versions[0].archive_path.read_bytes() == first_archive
    judges = []

    def grade(submission_id, *args):
        for version in state.version_store.list_versions():
            key = "run-artifacts/" + version.archive_path.relative_to(tmp_path).as_posix()
            assert storage.objects[key] == version.archive_path.read_bytes()
        judges.append(submission_id)
        state.submissions[submission_id]["status"] = judge_server.SubmissionStatus.COMPLETED

    monkeypatch.setattr(state, "_grade_worker_serial", grade)
    accepted = _post_submission(
        client, token, _archive(("submission/main.py", b"second\n")),
    )
    assert accepted.status_code == 200
    assert accepted.json()["round_id"] == "agent-2"
    assert judges == [accepted.json()["submission_id"]]
    assert state.resolve_token(token)["agent_inflight"] is False


def test_external_persistence_fails_closed_without_a_version_store(monkeypatch, tmp_path):
    calls = []
    client, state = _client(
        monkeypatch, tmp_path, store_task_id=None,
        version_persister=lambda version: calls.append(version),
    )
    token = state.register_session("task_alpha", "run-no-store")
    response = _post_submission(client, token, _archive(("submission/main.py", b"pass\n")))
    assert response.status_code == 503
    assert calls == []
    assert state.submissions == {}
    assert state.resolve_token(token)["agent_inflight"] is False


def test_accepted_submission_keeps_background_evaluation_error_visible(monkeypatch, tmp_path):
    persisted = []
    client, state = _client(
        monkeypatch, tmp_path, version_persister=lambda version: persisted.append(version),
    )
    state.tasks["task_alpha"].judge.private_artifact_path = None
    calls = []

    def evaluate(**kwargs):
        assert len(persisted) == 1
        calls.append(kwargs["submission_id"])
        raise RuntimeError("Judge allocation failed")

    state.submission_evaluator = evaluate
    token = state.register_session("task_alpha", "run-judge-error")
    response = _post_submission(client, token, _archive(("submission/main.py", b"pass\n")))
    assert response.status_code == 200, response.text
    submission_id = response.json()["submission_id"]
    result = client.get(f"/api/v1/result/{submission_id}").json()
    assert calls == [submission_id]
    assert result["status"] == "error"
    assert result["error"] == "Judge allocation failed"
    assert state.resolve_token(token)["agent_inflight"] is False


def test_checkpoint_endpoint_saves_without_consuming_judge_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, state = _client(monkeypatch, tmp_path)
    token = state.register_session(
        "task_alpha",
        "run-001",
        max_agent_submissions=1,
        submission_cooldown=300,
    )
    archive = _archive(("submission/main.py", b"print('ok')\n"))

    response = _post_checkpoint(client, token, "known-good", archive)

    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["round_id"] == "checkpoint-1"
    assert payload["version_id"].endswith(f"__checkpoint-1__{payload['archive_sha256'][:12]}")
    assert payload["candidate_id"] == f"candidate-{payload['content_sha256']}"
    assert payload["archive_size"] == len(archive)
    token_state = state.resolve_token(token)
    assert token_state["next_agent"] == 1
    assert token_state["last_agent_submit_at"] == 0.0
    assert token_state["next_checkpoint"] == 2
    assert state.submissions == {}

    stored = VersionStore.open(tmp_path / "store").list_versions()
    assert len(stored) == 1
    assert stored[0].source == "checkpoint"
    assert stored[0].label == "known-good"
    assert stored[0].archive_path.read_bytes() == archive


def test_registration_fails_closed_on_version_store_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, _ = _client(monkeypatch, tmp_path)

    mismatch = client.post(
        "/api/v1/register",
        json={
            "task_id": "task_alpha",
            "run_id": "run-mismatch",
            "admin_secret": judge_server.ADMIN_SECRET,
            "require_version_saving": True,
            "required_version_store_id": "different-store",
        },
    )
    matched = client.post(
        "/api/v1/register",
        json={
            "task_id": "task_alpha",
            "run_id": "run-matched",
            "admin_secret": judge_server.ADMIN_SECRET,
            "require_version_saving": True,
            "required_version_store_id": "api-test-store",
        },
    )

    assert mismatch.status_code == 409
    assert mismatch.json()["detail"] == "required version store is unavailable"
    assert matched.status_code == 200
    assert matched.json()["version_saving_enabled"] is True
    assert matched.json()["version_store_id"] == "api-test-store"


def test_checkpoint_endpoint_enforces_rate_and_storage_quotas(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, state = _client(monkeypatch, tmp_path)
    token = state.register_session("task_alpha", "run-quota")
    archive = _archive(("submission/main.py", b"pass\n"))
    monkeypatch.setattr(judge_server, "CHECKPOINT_MIN_INTERVAL_SECONDS", 60.0)

    first = _post_checkpoint(client, token, "first", archive)
    rate_limited = _post_checkpoint(client, token, "too-fast", archive)

    assert first.status_code == 201
    assert rate_limited.status_code == 429
    assert "rate limit" in rate_limited.json()["detail"]

    monkeypatch.setattr(judge_server, "CHECKPOINT_MIN_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(judge_server, "CHECKPOINT_MAX_BYTES_PER_RUN", len(archive))
    quota_limited = _post_checkpoint(
        client, token, "over-quota", _archive(("submission/main.py", b"pass2\n"))
    )

    assert quota_limited.status_code == 429
    assert "quota" in quota_limited.json()["detail"]
    assert len(VersionStore.open(tmp_path / "store").list_versions()) == 1


def test_checkpoint_endpoint_allows_only_one_inflight_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, state = _client(monkeypatch, tmp_path)
    token = state.register_session("task_alpha", "run-single-flight")
    archive = _archive(("submission/main.py", b"pass\n"))
    assert state.version_store is not None
    original_record = state.version_store.record_archive
    entered = threading.Event()
    release = threading.Event()
    failures: list[BaseException] = []

    def slow_record(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original_record(*args, **kwargs)

    monkeypatch.setattr(state.version_store, "record_archive", slow_record)

    def save_first() -> None:
        try:
            state.save_checkpoint(token, archive, "first")
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    worker = threading.Thread(target=save_first)
    worker.start()
    assert entered.wait(5)
    try:
        with pytest.raises(judge_server.CheckpointBusy, match="in progress"):
            state.save_checkpoint(token, archive, "second")
    finally:
        release.set()
        worker.join(5)

    assert worker.is_alive() is False
    assert failures == []


@pytest.mark.parametrize("kind", ["agent", "public", "auto"])
def test_checkpoint_only_token_cannot_submit_for_evaluation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    kind: str,
) -> None:
    client, state = _client(monkeypatch, tmp_path)
    registration = client.post(
        "/api/v1/register",
        json={
            "task_id": "task_alpha",
            "run_id": f"fixed-{kind}",
            "admin_secret": judge_server.ADMIN_SECRET,
            "max_agent_submissions": 0,
            "issue_checkpoint_token": True,
        },
    )
    assert registration.status_code == 200, registration.text
    checkpoint_token = registration.json()["checkpoint_token"]
    assert checkpoint_token

    archive = _archive(("submission/main.py", kind.encode()))
    saved = _post_checkpoint(client, checkpoint_token, "fixed-session", archive)
    assert saved.status_code == 201, saved.text

    scheduled: list[object] = []
    monkeypatch.setattr(
        state,
        "submit",
        lambda *args, **kwargs: scheduled.append(args) or "submission-id",
    )
    submitted = _post_submission(
        client,
        checkpoint_token,
        archive,
        kind=kind,
    )

    assert submitted.status_code == 403
    assert submitted.json()["detail"] == "token is limited to version checkpoints"
    assert scheduled == []
    assert (
        client.get("/api/v1/history", params={"token": checkpoint_token}).status_code
        == 403
    )
    assert (
        client.get(
            "/api/v1/public/history",
            params={"token": checkpoint_token},
        ).status_code
        == 403
    )
    assert state.resolve_token(checkpoint_token)["next_checkpoint"] == 2
    assert VersionStore.open(tmp_path / "store").list_versions(source=kind) == []


def test_checkpoint_endpoint_rejects_invalid_token_and_unregistered_task(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, state = _client(monkeypatch, tmp_path, store_task_id="other_task")
    archive = _archive(("submission/main.py", b"pass\n"))

    invalid = _post_checkpoint(client, "invalid", "known-good", archive)
    token = state.register_session("task_alpha", "run-001")
    unsupported = _post_checkpoint(client, token, "known-good", archive)

    assert invalid.status_code == 401
    assert unsupported.status_code == 404
    assert state.resolve_token(token)["next_checkpoint"] == 1


@pytest.mark.parametrize(
    ("path", "headers", "expected_status"),
    [
        (
            "/api/v1/versions",
            {
                _TOKEN_HEADER: "invalid",
                _VERSION_LABEL_HEADER: "known-good",
            },
            401,
        ),
        (
            "/api/v1/submit",
            {
                _TOKEN_HEADER: "checkpoint-token",
                _SUBMISSION_KIND_HEADER: "agent",
            },
            403,
        ),
    ],
)
def test_unauthorized_archive_upload_is_rejected_before_body_is_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    path: str,
    headers: dict[str, str],
    expected_status: int,
) -> None:
    client, state = _client(monkeypatch, tmp_path)
    if path.endswith("submit"):
        session_token = state.register_session("task_alpha", "run-fixed")
        checkpoint_token = state.register_checkpoint_token(session_token)
        headers = dict(headers)
        headers[_TOKEN_HEADER] = checkpoint_token
    body_reads = 0

    def body():
        nonlocal body_reads
        body_reads += 1
        yield b"must-not-be-read"

    response = client.post(
        path,
        content=body(),
        headers={"Content-Type": _CONTENT_TYPE, **headers},
    )

    assert response.status_code == expected_status
    assert body_reads == 0


@pytest.mark.parametrize("path", ["/api/v1/versions", "/api/v1/submit"])
def test_archive_upload_rejects_oversized_content_length_before_reading_body(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    path: str,
) -> None:
    client, state = _client(monkeypatch, tmp_path)
    token = state.register_session("task_alpha", f"run-size-{path.rsplit('/', 1)[-1]}")
    body_reads = 0

    def body():
        nonlocal body_reads
        body_reads += 1
        yield b"must-not-be-read"

    headers = {
        "Content-Type": _CONTENT_TYPE,
        "Content-Length": str(judge_server.SUBMISSION_ARCHIVE_MAX_BYTES + 1),
        _TOKEN_HEADER: token,
    }
    if path.endswith("versions"):
        headers[_VERSION_LABEL_HEADER] = "oversized"
    else:
        headers[_SUBMISSION_KIND_HEADER] = "agent"

    response = client.post(path, content=body(), headers=headers)

    assert response.status_code == 413
    assert body_reads == 0


@pytest.mark.parametrize("path", ["/api/v1/versions", "/api/v1/submit"])
def test_authenticated_multipart_upload_is_rejected_without_spooling_body(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    path: str,
) -> None:
    client, state = _client(monkeypatch, tmp_path)
    token = state.register_session(
        "task_alpha",
        f"run-multipart-{path.rsplit('/', 1)[-1]}",
    )
    body_reads = 0

    def body():
        nonlocal body_reads
        body_reads += 1
        yield b"--attack\r\nlarge multipart payload"

    headers = {
        "Content-Type": "multipart/form-data; boundary=attack",
        _TOKEN_HEADER: token,
    }
    if path.endswith("versions"):
        headers[_VERSION_LABEL_HEADER] = "no-multipart"
    else:
        headers[_SUBMISSION_KIND_HEADER] = "agent"

    response = client.post(path, content=body(), headers=headers)

    assert response.status_code == 415
    assert body_reads == 0


def test_chunked_archive_upload_has_a_hard_total_byte_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, state = _client(monkeypatch, tmp_path)
    token = state.register_session("task_alpha", "run-chunked-limit")
    monkeypatch.setattr(judge_server, "SUBMISSION_ARCHIVE_MAX_BYTES", 10)

    response = client.post(
        "/api/v1/submit",
        content=(chunk for chunk in (b"123456", b"789012")),
        headers={
            "Content-Type": _CONTENT_TYPE,
            _TOKEN_HEADER: token,
            _SUBMISSION_KIND_HEADER: "agent",
        },
    )

    assert response.status_code == 413
    token_state = state.resolve_token(token)
    assert token_state["next_agent"] == 1
    assert token_state["agent_inflight"] is False


def test_one_run_accepts_only_one_archive_upload_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, state = _client(monkeypatch, tmp_path)
    token = state.register_session("task_alpha", "run-upload-single-flight")

    with state.submission_upload_admission(token, "agent"):
        with pytest.raises(judge_server.ArchiveUploadBusy, match="in progress"):
            with state.submission_upload_admission(token, "public"):
                pass


def test_checkpoint_rejects_forbidden_member_before_allocating_round(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, state = _client(monkeypatch, tmp_path)
    token = state.register_session("task_alpha", "run-001")
    archive = _archive(
        ("submission/main.py", b"pass\n"),
        ("auth.json", b"secret\n"),
    )

    response = _post_checkpoint(client, token, "unsafe", archive)

    assert response.status_code == 400
    assert "outside the submit contract" in response.json()["detail"]
    assert state.resolve_token(token)["next_checkpoint"] == 1
    assert VersionStore.open(tmp_path / "store").list_versions() == []


@pytest.mark.parametrize("kind", ["agent", "public", "auto"])
def test_submit_persists_exact_evaluated_bytes_before_queuing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    kind: str,
) -> None:
    client, state = _client(monkeypatch, tmp_path)
    token = state.register_session("task_alpha", f"run-{kind}")
    archive = _archive(("submission/main.py", kind.encode()))
    scheduled: list[tuple] = []

    def fake_submit(*args, **kwargs) -> str:
        scheduled.append((args, kwargs))
        return f"{kind}-submission-id"

    monkeypatch.setattr(state, "submit", fake_submit)
    response = _post_submission(client, token, archive, kind=kind)

    assert response.status_code == 200, response.text
    assert len(scheduled) == 1
    stored = VersionStore.open(tmp_path / "store").list_versions(source=kind)
    assert len(stored) == 1
    assert stored[0].round_id == f"{kind}-1"
    assert stored[0].archive_path.read_bytes() == archive
    payload = response.json()
    assert payload["version_id"] == stored[0].version_id
    assert payload["candidate_id"] == stored[0].candidate_id
    assert payload["content_sha256"] == stored[0].content_sha256
    assert scheduled[0][1]["stored_version"] == stored[0]


def test_checkpoint_and_judge_event_share_one_logical_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, state = _client(monkeypatch, tmp_path)
    token = state.register_session("task_alpha", "run-candidate")
    checkpoint_archive = _archive(("submission/main.py", b"same\n"))
    evaluated_archive = _archive(("submission/main.py", b"same\n"))
    checkpoint = _post_checkpoint(
        client,
        token,
        "before-public",
        checkpoint_archive,
    )
    monkeypatch.setattr(state, "submit", lambda *args, **kwargs: "submission-id")
    submitted = _post_submission(client, token, evaluated_archive)

    assert checkpoint.status_code == 201
    assert submitted.status_code == 200
    assert checkpoint.json()["candidate_id"] == submitted.json()["candidate_id"]
    candidates = VersionStore.open(tmp_path / "store").list_candidates(
        run_id="run-candidate"
    )
    assert len(candidates) == 1
    assert [event.round_id for event in candidates[0].events] == [
        "checkpoint-1",
        "agent-1",
    ]


def test_judge_history_keeps_candidate_identity_with_the_score(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, state = _client(monkeypatch, tmp_path)
    archive = _archive(("submission/main.py", b"scored\n"))
    stored = state.persist_submission_version(
        task_id="task_alpha",
        run_id="run-scored",
        source="agent",
        round_id="agent-1",
        archive=archive,
    )
    assert stored is not None
    submission_id = state.submit(
        "task_alpha",
        archive,
        "run-scored",
        "agent-1",
        stored_version=stored,
        background_tasks=BackgroundTasks(),
    )
    state._record_submission(
        "run-scored",
        submission_id,
        "task_alpha",
        "agent-1",
        {
            "submitted_at": 1.0,
            "pass_rate": 1.0,
            "score": 42.5,
            "passed": 1,
            "failed": 0,
            "total_tests": 1,
            "valid": True,
            "summary": "valid",
        },
        None,
    )

    result = state.get_result(submission_id)
    assert result is not None
    assert result["candidate_id"] == stored.candidate_id
    entry = state.get_run_history("run-scored", "task_alpha")["entries"][0]
    assert entry["candidate_id"] == stored.candidate_id
    assert entry["content_sha256"] == stored.content_sha256
    assert entry["version_id"] == stored.version_id
    assert entry["score"] == 42.5


def test_submit_rejects_forbidden_archive_without_consuming_agent_round(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, state = _client(monkeypatch, tmp_path)
    token = state.register_session("task_alpha", "run-agent")
    scheduled: list[object] = []
    monkeypatch.setattr(
        state,
        "submit",
        lambda *args, **kwargs: scheduled.append(args) or "submission-id",
    )
    archive = _archive(("auth.json", b"secret\n"))

    response = _post_submission(client, token, archive)

    assert response.status_code == 400
    assert scheduled == []
    token_state = state.resolve_token(token)
    assert token_state["next_agent"] == 1
    assert token_state["agent_inflight"] is False


def test_submit_store_failure_is_503_and_releases_inflight_slot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, state = _client(monkeypatch, tmp_path)
    token = state.register_session("task_alpha", "run-agent")
    archive = _archive(("submission/main.py", b"pass\n"))
    scheduled: list[object] = []
    monkeypatch.setattr(
        state,
        "submit",
        lambda *args, **kwargs: scheduled.append(args) or "submission-id",
    )

    def fail_write(*args, **kwargs):
        raise VersionStoreWriteError("disk unavailable")

    assert state.version_store is not None
    monkeypatch.setattr(state.version_store, "record_archive", fail_write)

    response = _post_submission(client, token, archive)

    assert response.status_code == 503
    assert response.json()["detail"] == "version store write failed"
    assert scheduled == []
    assert state.resolve_token(token)["agent_inflight"] is False


def test_store_integrity_error_is_sanitized_at_checkpoint_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, state = _client(monkeypatch, tmp_path)
    token = state.register_session("task_alpha", "run-integrity")
    corrupt = (
        tmp_path
        / "store/tasks/task-alpha/runs/run-integrity/versions/published/manifest.json"
    )
    corrupt.parent.mkdir(parents=True)
    corrupt.write_text("{", encoding="utf-8")
    archive = _archive(("submission/main.py", b"pass\n"))

    response = _post_checkpoint(client, token, "safe", archive)

    assert response.status_code == 503
    assert response.json()["detail"] == "version store integrity failure"
    assert str(tmp_path) not in response.text


def test_checkpoint_endpoint_is_disabled_without_a_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, state = _client(monkeypatch, tmp_path, store_task_id=None)
    token = state.register_session("task_alpha", "run-001")
    archive = _archive(("submission/main.py", b"pass\n"))

    response = _post_checkpoint(client, token, "known-good", archive)

    assert response.status_code == 404
