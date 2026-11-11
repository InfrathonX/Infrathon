from __future__ import annotations

import importlib.util
import io
import json
import os
import stat
import sys
import tarfile
from pathlib import Path

import pytest


TASK_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = TASK_ROOT / "images" / "work"
CLIENT_PATH = WORK_ROOT / "public_benchmark" / "kvbench.py"
TEST_PUBLIC_DIRECT_RATE = 137.25


def _load_client():
    name = "kvbench_public_work_client_contract"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, CLIENT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("KVBENCH_PUBLIC_DIRECT_RATE", str(TEST_PUBLIC_DIRECT_RATE))
    return _load_client()


def _config(client, tmp_path: Path, token: str = "test-session-token"):
    submission = tmp_path / "submission"
    submission.mkdir()
    launch = submission / "launch.sh"
    launch.write_text("#!/bin/sh\nexec python3 gateway.py\n", encoding="utf-8")
    launch.chmod(0o755)
    (submission / "gateway.py").write_text("print('candidate')\n", encoding="utf-8")
    return client.ClientConfig(
        judge_url="http://judge.internal:8080",
        token=token,
        submission_root=submission,
        eval_root=tmp_path / "public_evals",
        poll_interval_sec=0.01,
        evaluation_timeout_sec=30.0,
    )


def test_public_profile_matches_original_fixed_window_contract(client) -> None:
    assert client.public_workload_contract() == {
        "profile": "public-swe192-distinct",
        "workflows": 192,
        "concurrency": 192,
        "max_steps": 100,
        "measurement_warmup_sec": 300,
        "measurement_window_sec": 600,
    }
    assert client.configured_direct_rate() == TEST_PUBLIC_DIRECT_RATE


@pytest.mark.parametrize("raw", ["", "0", "-1", "nan", "inf", "not-a-number"])
def test_public_direct_rate_projection_fails_closed(
    client, monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv(client.PUBLIC_DIRECT_RATE_ENV, raw)
    with pytest.raises(client.KvbenchError, match="KVBENCH_PUBLIC_DIRECT_RATE"):
        client.configured_direct_rate()


def test_work_image_installs_unprivileged_public_client() -> None:
    dockerfile = (WORK_ROOT / "Dockerfile").read_text(encoding="utf-8")
    dockerignore = (WORK_ROOT / ".dockerignore").read_text(encoding="utf-8")
    wrapper = WORK_ROOT / "public_benchmark" / "kvbench"

    assert "COPY public_benchmark/ /opt/kvbench-public/" in dockerfile
    assert "/usr/local/bin/kvbench" in dockerfile
    assert "/app/public_evals" in dockerfile
    assert "!public_benchmark/**" in dockerignore
    assert "python3 /opt/kvbench-public/kvbench.py" in wrapper.read_text(encoding="utf-8")


def test_agent_instructions_expose_real_public_workflow() -> None:
    instructions = (WORK_ROOT / "AGENT_TASK.md").read_text(encoding="utf-8")
    for required in (
        "kvbench doctor",
        "kvbench run --mode baseline --profile public-swe192-distinct",
        "kvbench run --mode candidate --profile public-swe192-distinct",
        "kvbench compare",
        "192 distinct SWE-Bench Lite",
        "concurrency 192",
        "100 agent steps",
        "600-second scored window",
        "KVBENCH_PUBLIC_DIRECT_RATE",
    ):
        assert required in instructions


def test_public_client_contains_no_private_evidence_identifiers() -> None:
    public_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((WORK_ROOT / "public_benchmark").iterdir())
        if path.is_file()
    )
    forbidden = (
        "4821d102",
        "7a2c89fb",
        "3852ca88",
        "110a9f3a",
        "rollout-2026-07-31",
        "/tmp/harbor-kv-qwen36",
        "workload_nonce",
        "83.4",
        "83.73333333333333",
    )
    for value in forbidden:
        assert value not in public_source


def test_raw_submission_uses_existing_token_and_public_kind_headers(client) -> None:
    body, headers = client.build_submission_request(
        "opaque-token",
        b"archive-bytes",
    )

    assert body == b"archive-bytes"
    assert headers == {
        "Content-Type": "application/gzip",
        "X-SForge-Token": "opaque-token",
        "X-SForge-Submission-Kind": "public",
    }


def test_candidate_run_snapshots_polls_and_saves_complete_artifact(
    client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(client, tmp_path, token="must-not-be-persisted")
    calls: list[tuple[str, str]] = []

    def fake_request(config_arg, method, path, **kwargs):
        assert config_arg is config
        calls.append((method, path))
        if path == "/api/v1/submit":
            assert kwargs["body"].startswith(b"\x1f\x8b")
            assert kwargs["headers"] == {
                "Content-Type": "application/gzip",
                "X-SForge-Token": "must-not-be-persisted",
                "X-SForge-Submission-Kind": "public",
            }
            return {
                "submission_id": "public-001",
                "round_id": "public-1",
                "status": "queued",
            }
        assert path == "/api/v1/result/public-001"
        return {
            "submission_id": "public-001",
            "status": "completed",
            "report": {
                "valid": True,
                    "candidate": {"valid_steps_per_min": 113.63},
            },
            "error": None,
        }

    monkeypatch.setattr(client, "request_json", fake_request)
    result = client.run_candidate(config)

    assert calls == [
        ("POST", "/api/v1/submit"),
        ("GET", "/api/v1/result/public-001"),
    ]
    assert result["comparison"]["speedup"] == pytest.approx(
        113.63 / TEST_PUBLIC_DIRECT_RATE
    )
    complete = config.eval_root / "eval_000001.COMPLETE"
    assert complete.is_dir()
    assert not (config.eval_root / "eval_000001.RUNNING").exists()
    assert (complete / "code" / "launch.sh").is_file()
    assert stat.S_IMODE((complete / "code" / "launch.sh").stat().st_mode) == 0o444
    assert json.loads((complete / "candidate_result.json").read_text())["report"][
        "valid"
    ] is True
    metadata_text = (complete / "metadata.json").read_text(encoding="utf-8")
    assert "must-not-be-persisted" not in metadata_text
    assert "judge.internal" not in metadata_text
    with tarfile.open(complete / "code_snapshot.tar.gz", "r:gz") as archive:
        names = set(archive.getnames())
        assert "submission/launch.sh" in names
        assert "submission/gateway.py" in names
        assert archive.getmember("submission/launch.sh").mode & 0o111


def test_failed_candidate_is_finalized_without_result_leak(
    client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(client, tmp_path)

    def reject(*args, **kwargs):
        raise client.KvbenchError("public service busy")

    monkeypatch.setattr(client, "submit_public_archive", reject)
    with pytest.raises(client.KvbenchError, match="busy"):
        client.run_candidate(config)

    complete = config.eval_root / "eval_000001.COMPLETE"
    metadata = json.loads((complete / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "failed"
    assert not (complete / "candidate_result.json").exists()


def test_poll_failure_retains_accepted_submission_id(
    client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(client, tmp_path)
    monkeypatch.setattr(
        client,
        "submit_public_archive",
        lambda *args, **kwargs: {"submission_id": "public-trace-7", "status": "queued"},
    )

    def poll_failure(*args, **kwargs):
        raise client.KvbenchError("result connection ended")

    monkeypatch.setattr(client, "poll_public_result", poll_failure)
    with pytest.raises(client.KvbenchError, match="connection ended"):
        client.run_candidate(config)

    metadata = json.loads(
        (config.eval_root / "eval_000001.COMPLETE" / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["submission_id"] == "public-trace-7"


def test_snapshot_rejects_symlinks(client, tmp_path: Path) -> None:
    config = _config(client, tmp_path)
    (config.submission_root / "outside.py").symlink_to("/etc/passwd")

    with pytest.raises(client.KvbenchError, match="unsafe submission entry"):
        client.begin_candidate_eval(config)
    assert not list(config.eval_root.glob("*.RUNNING"))


def test_baseline_is_frozen_and_uses_no_http(
    client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(client, tmp_path)

    def unexpected(*args, **kwargs):
        raise AssertionError("frozen baseline must not use the live service")

    monkeypatch.setattr(client, "request_json", unexpected)
    baseline = client.run_baseline(config)

    assert baseline["source"] == "frozen-direct"
    assert baseline["valid_steps_per_min"] == TEST_PUBLIC_DIRECT_RATE
    saved = json.loads(
        (config.eval_root / "frozen_direct_baseline.json").read_text(encoding="utf-8")
    )
    assert saved == baseline


def test_doctor_checks_task_scoped_public_endpoint_without_printing_token(
    client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(client, tmp_path, token="doctor-secret")
    observed: dict[str, str] = {}

    def fake_request(config_arg, method, path, **kwargs):
        observed.update(method=method, path=path)
        return {"status": "ready", "entries": []}

    monkeypatch.setattr(client, "request_json", fake_request)
    result = client.doctor(config)

    assert observed == {
        "method": "GET",
        "path": "/api/v1/public/history?token=doctor-secret",
    }
    encoded = json.dumps(result, sort_keys=True)
    assert "doctor-secret" not in encoded
    assert result["workload"]["measurement_window_sec"] == 600
    assert result["frozen_direct_valid_steps_per_min"] == TEST_PUBLIC_DIRECT_RATE


def test_poll_rejects_invalid_or_error_results(
    client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(client, tmp_path)
    replies = iter(
        [
            {"submission_id": "p1", "status": "queued", "report": None, "error": None},
            {"submission_id": "p1", "status": "error", "report": None, "error": "runner stopped"},
        ]
    )
    monkeypatch.setattr(client, "request_json", lambda *args, **kwargs: next(replies))

    with pytest.raises(client.KvbenchError, match="runner stopped"):
        client.poll_public_result(
            config,
            "p1",
            monotonic=iter([0.0, 0.0]).__next__,
            sleep=lambda _: None,
        )


def test_compare_command_profile_defaults_to_public_contract(client) -> None:
    args = client.build_parser().parse_args(["compare", "--json"])
    assert args.command == "compare"
    assert args.profile == "public-swe192-distinct"
    assert args.json_output is True
