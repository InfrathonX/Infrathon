from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_public_candidate.py"


def _load_runner(name: str):
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configure(
    monkeypatch,
    tmp_path: Path,
    *,
    evaluation: dict[str, object] | None = None,
    candidate_error: BaseException | None = None,
    stop_fails: bool = False,
):
    runner = _load_runner(
        f"kvbench_public_runner_{tmp_path.name}_{stop_fails}_{id(evaluation)}"
    )
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    launch = candidate_dir / "launch.sh"
    launch.write_text("#!/bin/sh\n", encoding="utf-8")
    launch.chmod(0o755)
    (candidate_dir / "gateway.py").write_text("# candidate\n", encoding="utf-8")
    expected_tree = runner.V22.V21.tree_sha256(candidate_dir)

    raw_nonce = "a" * 64
    nonce_source = tmp_path / "public-nonce-source.log"
    nonce_source.write_text(f"trusted audit value={raw_nonce}\n", encoding="utf-8")
    output = tmp_path / "private" / "public-evidence.json"
    baseline_sha256 = "b" * 64
    direct_rate = 103.3
    monkeypatch.setattr(
        runner,
        "parse_args",
        lambda: SimpleNamespace(
            candidate_dir=candidate_dir,
            expected_candidate_tree_sha256=expected_tree,
            public_workload_nonce_source=nonce_source,
            output=output,
        ),
    )
    monkeypatch.setattr(
        runner,
        "_validate_frozen_public_spec",
        lambda: {
            "profile": runner.PUBLIC_PROFILE,
            "workload_nonce_sha256": runner.PUBLIC_NONCE_SHA256,
            "request_timeout_sec": 3600,
        },
    )

    def validate_nonce(value: str, *, expected_sha256: str) -> str:
        if value != raw_nonce or expected_sha256 != runner.PUBLIC_NONCE_SHA256:
            raise ValueError("not the synthetic public nonce")
        return value

    monkeypatch.setattr(runner.V22, "validate_workload_nonce", validate_nonce)
    monkeypatch.setattr(
        runner.V22,
        "load_and_validate_baseline",
        lambda suite: {
            "suite": suite,
            "profile": runner.PUBLIC_PROFILE,
            "path": "/private/public-direct.json",
            "file_sha256": baseline_sha256,
            "workload_nonce_sha256": runner.PUBLIC_NONCE_SHA256,
            "direct_valid_steps_per_min": direct_rate,
            "backend_serving_contract_sha256": (
                runner.V22.BACKEND_SERVING_CONTRACT_SHA256
            ),
            "leg": {"private": "baseline"},
        },
    )
    monkeypatch.setattr(
        runner.V22, "MEASUREMENT_LOCK_PATH", tmp_path / "measurement.lock"
    )
    monkeypatch.setattr(runner, "reconcile_runtime", lambda path: False)

    calls: list[tuple[str, int]] = []

    def run_stack(action: str, *, candidate_dir: Path, timeout: int):
        calls.append((action, timeout))
        if action == "stop" and stop_fails:
            raise RuntimeError("synthetic stop failure")
        return "ok"

    monkeypatch.setattr(runner.V22, "run_stack", run_stack)

    leg_calls: list[dict[str, object]] = []
    private_task_id = "PRIVATE-TASK-ID-MUST-NOT-REACH-STDOUT"

    def run_candidate_leg(**kwargs):
        leg_calls.append(kwargs)
        if candidate_error is not None:
            raise candidate_error
        return {
            "valid_steps_per_min": 140.0,
            "workflows": [{"task_id": private_task_id}],
        }

    monkeypatch.setattr(runner.V22, "run_candidate_leg", run_candidate_leg)
    default_evaluation: dict[str, object] = {
        "valid": True,
        "qualification_passed": False,
        "strictly_better_than_v2_1_direct": False,
        "direct_valid_steps_per_min": direct_rate,
        "candidate_valid_steps_per_min": 140.0,
        "speedup": 140.0 / direct_rate,
        "change_percent": (
            140.0 / direct_rate - 1.0
        ) * 100.0,
        "all_invalid_reasons": [],
    }
    monkeypatch.setattr(
        runner.V22,
        "candidate_evaluation",
        lambda **kwargs: evaluation or default_evaluation,
    )
    empty_residue = {
        "compose_container_ids": [],
        "sandbox_container_ids": [],
    }
    monkeypatch.setattr(
        runner,
        "wait_runtime_quiescent",
        lambda timeout=runner.QUIESCENCE_TIMEOUT_SEC: (
            {
                "compose_container_ids": ["dirty"] if stop_fails else [],
                "sandbox_container_ids": [],
            }
        ),
    )
    return SimpleNamespace(
        runner=runner,
        candidate_dir=candidate_dir,
        expected_tree=expected_tree,
        raw_nonce=raw_nonce,
        nonce_source=nonce_source,
        output=output,
        calls=calls,
        leg_calls=leg_calls,
        private_task_id=private_task_id,
        empty_residue=empty_residue,
        baseline_sha256=baseline_sha256,
        direct_rate=direct_rate,
    )


def test_frozen_public_contract_is_exact() -> None:
    runner = _load_runner("kvbench_public_runner_contract")

    spec = runner._validate_frozen_public_spec()

    assert spec["profile"] == "public-swe192-distinct"
    assert spec["workload_nonce_sha256"] == (
        "4902baae73726c1a2f372e35e9e104f6bb3db7dc7593822359548fba60e1b00c"
    )
    assert spec["request_timeout_sec"] == 3600
    assert "path" not in spec
    assert "file_sha256" not in spec
    assert "direct_valid_steps_per_min" not in spec
    assert runner.V22.V21.EXPECTED_WINDOWS["public"] == {
        "warmup_sec": 300,
        "measurement_sec": 600,
    }


def test_valid_slower_candidate_completes_and_keeps_full_leg_private(
    monkeypatch, tmp_path, capsys
) -> None:
    configured = _configure(monkeypatch, tmp_path)

    assert configured.runner.main() == 0
    captured = capsys.readouterr()
    evidence = json.loads(configured.output.read_text(encoding="utf-8"))

    assert configured.calls == [
        ("start", configured.runner.STACK_TIMEOUT_SEC),
        ("stop", configured.runner.STACK_TIMEOUT_SEC),
    ]
    assert configured.leg_calls == [
        {
            "suite": "public",
            "profile": "public-swe192-distinct",
            "workload_nonce": configured.raw_nonce,
            "request_timeout_sec": 3600,
        }
    ]
    assert evidence["status"] == "complete"
    assert evidence["phase"] == "quiescent"
    assert evidence["evaluation"]["valid"] is True
    assert evidence["evaluation"]["qualification_passed"] is False
    assert evidence["workload_contract"] == {
        "suite": "public",
        "profile": "public-swe192-distinct",
        "workflows": 192,
        "concurrency": 192,
        "max_steps": 100,
        "measurement_warmup_sec": 300,
        "measurement_window_sec": 600,
        "workload_nonce_sha256": configured.runner.PUBLIC_NONCE_SHA256,
    }
    assert evidence["baseline_evidence"] == {
        "path": "/private/public-direct.json",
        "file_sha256": configured.baseline_sha256,
        "workload_nonce_sha256": configured.runner.PUBLIC_NONCE_SHA256,
        "direct_valid_steps_per_min": configured.direct_rate,
        "backend_serving_contract_sha256": (
            configured.runner.V22.BACKEND_SERVING_CONTRACT_SHA256
        ),
    }
    assert evidence["preflight_backend_serving_contract_sha256"] == (
        configured.runner.V22.BACKEND_SERVING_CONTRACT_SHA256
    )
    assert evidence["leg"]["workflows"][0]["task_id"] == configured.private_task_id
    assert os.stat(configured.output).st_mode & 0o777 == 0o600
    assert configured.raw_nonce not in configured.output.read_text(encoding="utf-8")
    assert configured.raw_nonce not in captured.out
    assert configured.private_task_id not in captured.out
    assert captured.err == ""


def test_protocol_invalid_candidate_is_exit_three(monkeypatch, tmp_path) -> None:
    invalid = {
        "valid": False,
        "qualification_passed": False,
        "strictly_better_than_v2_1_direct": False,
        "direct_valid_steps_per_min": 103.3,
        "candidate_valid_steps_per_min": 0.0,
        "speedup": 0.0,
        "change_percent": -100.0,
        "all_invalid_reasons": ["profile_shape_mismatch"],
    }
    configured = _configure(monkeypatch, tmp_path, evaluation=invalid)

    assert configured.runner.main() == 3
    evidence = json.loads(configured.output.read_text(encoding="utf-8"))

    assert evidence["status"] == "candidate_invalid"
    assert evidence["phase"] == "quiescent"
    assert evidence["evaluation"]["all_invalid_reasons"] == [
        "profile_shape_mismatch"
    ]
    assert evidence["cleanup"]["stack_stop_succeeded"] is True


def test_cancellation_is_cleaned_before_exit_130(monkeypatch, tmp_path) -> None:
    configured = _configure(monkeypatch, tmp_path)

    def cancel(**kwargs):
        raise configured.runner.EvaluationCancelled("synthetic TERM")

    monkeypatch.setattr(configured.runner.V22, "run_candidate_leg", cancel)

    assert configured.runner.main() == 130
    evidence = json.loads(configured.output.read_text(encoding="utf-8"))

    assert configured.calls == [
        ("start", configured.runner.STACK_TIMEOUT_SEC),
        ("stop", configured.runner.STACK_TIMEOUT_SEC),
    ]
    assert evidence["status"] == "cancelled_clean"
    assert evidence["cleanup"]["quiescent"] is True


def test_cleanup_failure_overrides_a_valid_measurement(monkeypatch, tmp_path) -> None:
    configured = _configure(monkeypatch, tmp_path, stop_fails=True)

    assert configured.runner.main() == 5
    evidence = json.loads(configured.output.read_text(encoding="utf-8"))

    assert evidence["status"] == "infrastructure_error_dirty"
    assert evidence["cleanup"]["quiescent"] is False
    assert any("cleanup" in error for error in evidence["errors"])


def test_resource_busy_never_touches_the_shared_stack(monkeypatch, tmp_path) -> None:
    configured = _configure(monkeypatch, tmp_path)

    def busy(*args, **kwargs):
        raise BlockingIOError("synthetic held lock")

    monkeypatch.setattr(configured.runner.fcntl, "flock", busy)

    assert configured.runner.main() == 4
    evidence = json.loads(configured.output.read_text(encoding="utf-8"))

    assert configured.calls == []
    assert evidence["status"] == "resource_busy"
    assert evidence["cleanup"]["attempted"] is False


def test_infrastructure_failure_is_cleanly_torn_down_and_redacted(
    monkeypatch, tmp_path, capsys
) -> None:
    secret_error = "PRIVATE-TASK-ID-IN-TRUSTED-ERROR"
    configured = _configure(
        monkeypatch,
        tmp_path,
        candidate_error=RuntimeError(secret_error),
    )

    assert configured.runner.main() == 5
    captured = capsys.readouterr()
    evidence = json.loads(configured.output.read_text(encoding="utf-8"))

    assert evidence["status"] == "infrastructure_error_clean"
    assert secret_error in evidence["errors"][0]
    assert secret_error not in captured.err
    assert configured.raw_nonce not in captured.err
    assert configured.calls[-1] == (
        "stop",
        configured.runner.STACK_TIMEOUT_SEC,
    )


def test_frozen_baseline_failure_is_exit_five_before_evidence_creation(
    monkeypatch, tmp_path
) -> None:
    configured = _configure(monkeypatch, tmp_path)

    def unavailable(_suite: str):
        raise RuntimeError("synthetic frozen evidence failure")

    monkeypatch.setattr(
        configured.runner.V22, "load_and_validate_baseline", unavailable
    )

    assert configured.runner.main() == 5
    assert not configured.output.exists()
    assert configured.calls == []


def test_nonce_source_requires_exactly_one_matching_value(
    monkeypatch, tmp_path
) -> None:
    runner = _load_runner("kvbench_public_runner_nonce_source")
    source = tmp_path / "nonce-source.log"
    first = "1" * 64
    second = "2" * 64
    source.write_text(f"{first}\n{second}\n", encoding="utf-8")
    monkeypatch.setattr(
        runner.V22,
        "validate_workload_nonce",
        lambda raw, *, expected_sha256: raw,
    )

    with pytest.raises(ValueError, match="exactly one"):
        runner._extract_public_nonce(source, expected_sha256="f" * 64)


def test_candidate_and_nonce_leaf_symlinks_are_rejected(monkeypatch, tmp_path) -> None:
    runner = _load_runner("kvbench_public_runner_symlinks")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    launch = candidate / "launch.sh"
    launch.write_text("#!/bin/sh\n", encoding="utf-8")
    launch.chmod(0o755)
    candidate_link = tmp_path / "candidate-link"
    candidate_link.symlink_to(candidate, target_is_directory=True)
    expected_tree = runner.V22.V21.tree_sha256(candidate)

    with pytest.raises(ValueError, match="real directory"):
        runner._validate_candidate(candidate_link, expected_tree)

    nonce = tmp_path / "nonce"
    nonce.write_text("a" * 64, encoding="utf-8")
    nonce_link = tmp_path / "nonce-link"
    nonce_link.symlink_to(nonce)
    monkeypatch.setattr(
        runner.V22,
        "validate_workload_nonce",
        lambda raw, *, expected_sha256: raw,
    )
    with pytest.raises(ValueError, match="real file"):
        runner._extract_public_nonce(nonce_link, expected_sha256="f" * 64)


def test_candidate_tree_change_is_candidate_invalid(monkeypatch, tmp_path) -> None:
    configured = _configure(monkeypatch, tmp_path)
    real_tree = configured.runner.V22.V21.tree_sha256
    calls = 0

    def changing_tree(path: Path) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_tree(path)
        return "e" * 64

    monkeypatch.setattr(configured.runner.V22.V21, "tree_sha256", changing_tree)

    assert configured.runner.main() == 3
    evidence = json.loads(configured.output.read_text(encoding="utf-8"))

    assert evidence["status"] == "candidate_invalid"
    assert configured.leg_calls == []
    assert evidence["cleanup"]["quiescent"] is True
