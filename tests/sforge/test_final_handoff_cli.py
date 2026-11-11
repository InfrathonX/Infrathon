from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from sforge import cli
from sforge.harness.config import SForgeConfig


CONTRACT = "sha256:" + "c" * 64
ALLOCATION = "allocation-sha256:" + "a" * 64


def _args(archive, **overrides):
    values = {
        "archive": str(archive),
        "run_id": "final-run",
        "timeout": 60,
        "silent": True,
        "json": False,
        "standalone_final_handoff": True,
        "handoff_source_run_id": "source-run",
        "judge_url": "http://127.0.0.1:8123/",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _config(tmp_path) -> SForgeConfig:
    return SForgeConfig(
        log_dir=tmp_path,
        backend="docker",
        single_gpu_handoff=True,
        trusted_lease_id="lease-1",
        accelerator_allocation_key=ALLOCATION,
    )


def _task():
    return SimpleNamespace(
        task_id="task",
        portable=SimpleNamespace(
            evaluation_contract_sha256=CONTRACT,
            defaults=SimpleNamespace(
                to_dict=lambda: {
                    "evaluation_suite_id": "suite",
                    "execution_profile_id": "profile",
                    "metric_id": "metric",
                }
            ),
        ),
        judge=SimpleNamespace(eval_timeout=60),
    )


def _report() -> dict:
    return {
        "task_id": "task",
        "submission_id": "final-submission",
        "total_tests": 1,
        "passed": 1,
        "failed": 0,
        "errors": 0,
        "pass_rate": 1.0,
        "score_0_100": 100.0,
        "runtime_seconds": 2.5,
        "timed_out": False,
        "valid": True,
        "metrics": {
            "portable_contract": {
                "evaluation_contract_sha256": CONTRACT,
                "selected_ids": {
                    "evaluation_suite_id": "suite",
                    "execution_profile_id": "profile",
                    "metric_id": "metric",
                },
            }
        },
    }


class _Response:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self.body


def _wire(monkeypatch, tmp_path, response_body):
    monkeypatch.setattr(cli, "_make_config", lambda args: _config(tmp_path))
    monkeypatch.setattr(cli, "_resolve_task", lambda args, config: _task())
    monkeypatch.setattr(
        cli,
        "create_backend_from_config",
        lambda config: pytest.fail("final handoff must not create a local backend"),
    )
    monkeypatch.setattr(
        cli,
        "judge_submission",
        lambda **kwargs: pytest.fail("final handoff must not run a local Judge"),
    )
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return _Response(response_body)

    monkeypatch.setattr(cli.requests, "post", post)
    return calls


def test_standalone_final_handoff_is_coordinator_only_and_exactly_bound(
    monkeypatch, tmp_path
):
    archive = tmp_path / "final.tar.gz"
    archive.write_bytes(b"immutable final archive")
    calls = _wire(
        monkeypatch,
        tmp_path,
        {
            "state": "CLOSED",
            "epoch": 14,
            "recovery_policy": "never_resume",
            "report": _report(),
        },
    )

    cli.cmd_eval(_args(archive))

    assert len(calls) == 1
    url, request = calls[0]
    assert url == "http://127.0.0.1:8123/api/v1/handoff/final-judge"
    assert request["timeout"] == (10, 180)
    payload = request["json"]
    assert payload == {
        "admin_secret": cli.ADMIN_SECRET,
        "source_run_id": "source-run",
        "final_run_id": "final-run",
        "task_id": "task",
        "trusted_lease_id": "lease-1",
        "allocation_key": ALLOCATION,
        "evaluation_contract_sha256": CONTRACT,
        "archive_sha256": "sha256:"
        + hashlib.sha256(archive.read_bytes()).hexdigest(),
        "archive_size": archive.stat().st_size,
        "timeout": 60,
    }
    assert "archive" not in payload
    assert "archive_path" not in payload
    assert payload["evaluation_contract_sha256"].count("sha256:") == 1


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"handoff_source_run_id": None}, "--handoff-source-run-id"),
        ({"judge_url": None}, "--judge-url"),
    ],
)
def test_standalone_final_handoff_requires_explicit_source_and_coordinator(
    monkeypatch, tmp_path, overrides, message
):
    archive = tmp_path / "final.tar.gz"
    archive.write_bytes(b"archive")
    calls = _wire(monkeypatch, tmp_path, {})

    with pytest.raises(ValueError, match=message):
        cli.cmd_eval(_args(archive, **overrides))

    assert calls == []


def test_standalone_final_handoff_never_sends_admin_secret_off_host(
    monkeypatch, tmp_path
):
    archive = tmp_path / "final.tar.gz"
    archive.write_bytes(b"archive")
    calls = _wire(monkeypatch, tmp_path, {})

    with pytest.raises(ValueError, match="loopback HTTP origin"):
        cli.cmd_eval(_args(archive, judge_url="https://judge.example.test"))

    assert calls == []


def test_standalone_final_handoff_rejects_unclosed_response(
    monkeypatch, tmp_path
):
    archive = tmp_path / "final.tar.gz"
    archive.write_bytes(b"archive")
    _wire(
        monkeypatch,
        tmp_path,
        {
            "state": "FINAL_JUDGE_CLEANED",
            "epoch": 12,
            "recovery_policy": "never_resume",
            "report": _report(),
        },
    )

    with pytest.raises(RuntimeError, match="did not close"):
        cli.cmd_eval(_args(archive))


def test_standalone_final_handoff_rejects_wrong_portable_contract(
    monkeypatch, tmp_path
):
    archive = tmp_path / "final.tar.gz"
    archive.write_bytes(b"archive")
    report = _report()
    report["metrics"]["portable_contract"][
        "evaluation_contract_sha256"
    ] = "sha256:" + "d" * 64
    _wire(
        monkeypatch,
        tmp_path,
        {
            "state": "CLOSED",
            "epoch": 14,
            "recovery_policy": "never_resume",
            "report": report,
        },
    )

    with pytest.raises(RuntimeError, match="wrong portable contract"):
        cli.cmd_eval(_args(archive))


def test_source_identity_cannot_be_used_with_ordinary_local_eval(
    monkeypatch, tmp_path
):
    archive = tmp_path / "final.tar.gz"
    archive.write_bytes(b"archive")
    config = _config(tmp_path)
    config.single_gpu_handoff = False
    monkeypatch.setattr(cli, "_make_config", lambda args: config)
    monkeypatch.setattr(cli, "_resolve_task", lambda args, config: _task())
    monkeypatch.setattr(
        cli,
        "create_backend_from_config",
        lambda config: pytest.fail("invalid invocation must fail before backend"),
    )

    with pytest.raises(ValueError, match="requires --standalone-final-handoff"):
        cli.cmd_eval(
            _args(
                archive,
                standalone_final_handoff=False,
                handoff_source_run_id="source-run",
            )
        )


def test_standalone_final_handoff_forwards_an_explicit_round_id(monkeypatch, tmp_path):
    archive = tmp_path / "final.tar.gz"
    archive.write_bytes(b"immutable final archive")
    calls = _wire(
        monkeypatch,
        tmp_path,
        {
            "state": "CLOSED",
            "epoch": 14,
            "recovery_policy": "never_resume",
            "report": _report(),
        },
    )

    cli.cmd_eval(_args(archive, final_round_id="final-source-run"))

    assert len(calls) == 1
    payload = calls[0][1]["json"]
    assert payload["round_id"] == "final-source-run"
    # Without the flag the payload stays exactly the historical shape.
    calls.clear()
    cli.cmd_eval(_args(archive))
    assert "round_id" not in calls[0][1]["json"]


def test_final_round_id_requires_standalone_final_handoff(monkeypatch, tmp_path):
    archive = tmp_path / "final.tar.gz"
    archive.write_bytes(b"immutable final archive")
    _wire(monkeypatch, tmp_path, {})
    with pytest.raises(ValueError, match="--final-round-id requires"):
        cli.cmd_eval(
            _args(
                archive,
                standalone_final_handoff=False,
                handoff_source_run_id=None,
                final_round_id="final-source-run",
            )
        )
