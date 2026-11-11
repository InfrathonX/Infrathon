from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "runtime/frozen/harbor_run_scripts/"
    "concurrent-agent-serving-optimizer-qwen36-27b-swe192-v2.2-gpu45/"
    "run-gpu67-v2.2-hidden-candidate-only.py"
)


def _load_runner(name: str):
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configure(monkeypatch, tmp_path: Path, *, stop_fails: bool = False):
    runner = _load_runner(f"kvbench_hidden_lifecycle_{tmp_path.name}_{stop_fails}")
    monkeypatch.setattr(
        runner.V22, "MEASUREMENT_LOCK_PATH", tmp_path / "measurement.lock"
    )
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    launch = candidate / "launch.sh"
    launch.write_text("#!/bin/sh\n", encoding="utf-8")
    launch.chmod(0o755)
    (candidate / "gateway.py").write_text("# candidate\n", encoding="utf-8")
    output = tmp_path / "evidence.json"
    expected_tree = runner.V22.V21.tree_sha256(candidate)
    monkeypatch.setattr(
        runner,
        "parse_args",
        lambda: SimpleNamespace(
            candidate_dir=candidate,
            expected_candidate_tree_sha256=expected_tree,
            hidden_workload_nonce="synthetic",
            hidden_workload_nonce_source=None,
            output=output,
        ),
    )
    monkeypatch.setattr(
        runner.V22,
        "validate_workload_nonce",
        lambda value, expected_sha256: "synthetic",
    )
    monkeypatch.setattr(
        runner.V22,
        "load_and_validate_baseline",
        lambda suite: {
            "path": "/private/baseline",
            "file_sha256": "b" * 64,
            "workload_nonce_sha256": "c" * 64,
            "direct_valid_steps_per_min": 83.73333333333333,
            "backend_serving_contract_sha256": (
                runner.V22.BACKEND_SERVING_CONTRACT_SHA256
            ),
        },
    )
    monkeypatch.setattr(
        runner.V22,
        "run_candidate_leg",
        lambda **kwargs: {"valid_steps_per_min": 90.0},
    )
    monkeypatch.setattr(
        runner.V22,
        "candidate_evaluation",
        lambda **kwargs: {"valid": True, "qualification_passed": True},
    )
    monkeypatch.setattr(runner, "reconcile_runtime", lambda path: False)

    calls: list[str] = []

    def run_stack(action: str, *, candidate_dir: Path):
        calls.append(action)
        if action == "stop" and stop_fails:
            raise RuntimeError("synthetic stop failure")
        return "ok"

    monkeypatch.setattr(runner.V22, "run_stack", run_stack)
    monkeypatch.setattr(
        runner,
        "wait_runtime_quiescent",
        lambda timeout=30.0: {
            "compose_container_ids": ["dirty"] if stop_fails else [],
            "sandbox_container_ids": [],
        },
    )
    return runner, output, calls


def test_complete_is_written_only_after_successful_teardown(monkeypatch, tmp_path) -> None:
    runner, output, calls = _configure(monkeypatch, tmp_path)

    assert runner.main() == 0
    evidence = json.loads(output.read_text(encoding="utf-8"))

    assert calls == ["start", "stop"]
    assert evidence["status"] == "complete"
    assert evidence["phase"] == "quiescent"
    assert evidence["errors"] == []
    assert evidence["cleanup"] == {
        "attempted": True,
        "stack_stop_succeeded": True,
        "quiescent": True,
        "reconciled_before_start": False,
        "residue": {"compose_container_ids": [], "sandbox_container_ids": []},
    }


def test_teardown_failure_cannot_leave_complete_evidence(monkeypatch, tmp_path) -> None:
    runner, output, calls = _configure(monkeypatch, tmp_path, stop_fails=True)

    assert runner.main() == 5
    evidence = json.loads(output.read_text(encoding="utf-8"))

    assert calls == ["start", "stop"]
    assert evidence["status"] == "failed_dirty"
    assert evidence["phase"] == "terminal"
    assert evidence["cleanup"]["quiescent"] is False
    assert any("cleanup" in error for error in evidence["errors"])


def test_cooperative_cancellation_still_cleans_before_terminal_evidence(
    monkeypatch, tmp_path
) -> None:
    runner, output, calls = _configure(monkeypatch, tmp_path)

    def cancel(**kwargs):
        raise runner.EvaluationCancelled("synthetic TERM")

    monkeypatch.setattr(runner.V22, "run_candidate_leg", cancel)
    assert runner.main() == 130
    evidence = json.loads(output.read_text(encoding="utf-8"))

    assert calls == ["start", "stop"]
    assert evidence["status"] == "cancelled_clean"
    assert evidence["cleanup"]["quiescent"] is True


CURRENT_URL_CONTROLLER_SHA256 = (
    "a287364dbcb513e3eb73ecaae0621479b6810b41ee6befde544ff3a52d651da6"
)


def _stable_model_provenance(runner) -> dict[str, object]:
    return {
        "profile_id": "qwen36-swe192",
        "profile_sha256": "1" * 64,
        "model_revision": "2" * 40,
        "runtime_image_id": "sha256:" + "3" * 64,
        "tensor_parallel_size": 2,
        "kv_cache_dtype": "fp8",
        "backend_serving_contract_sha256": (
            runner.V22.BACKEND_SERVING_CONTRACT_SHA256
        ),
    }


def _direct_leg(runner, controller_config_sha256) -> dict[str, object]:
    spec = runner.V22.BASELINE_SPECS["hidden"]
    return {
        "profile": spec["profile"],
        "mode": "baseline",
        "workload_nonce_sha256": spec["workload_nonce_sha256"],
        "controller_config_sha256": controller_config_sha256,
        "release_id": runner.V22.EXPECTED_RELEASE_ID,
        "valid_steps_per_min": 84.96666666666667,
        "model_provenance": _stable_model_provenance(runner),
        "workflows": [
            {"task_id": f"hidden-task-{index:03d}"} for index in range(192)
        ],
    }


def _write_hidden_baseline(path: Path, runner, controller_config_sha256) -> None:
    spec = runner.V22.BASELINE_SPECS["hidden"]
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "suite": "hidden",
                "profile": spec["profile"],
                "workload_nonce_sha256": spec["workload_nonce_sha256"],
                "legs": {
                    "baseline": _direct_leg(
                        runner, controller_config_sha256
                    )
                },
            }
        ),
        encoding="utf-8",
    )


def _load_synthetic_hidden_baseline(
    monkeypatch, tmp_path: Path, controller_config_sha256
):
    runner = _load_runner(f"kvbench_hidden_baseline_{tmp_path.name}")
    baseline_path = tmp_path / "hidden-direct.json"
    _write_hidden_baseline(baseline_path, runner, controller_config_sha256)
    monkeypatch.setenv("KVBENCH_HIDDEN_BASELINE_PATH", str(baseline_path))
    # The fixture isolates the v2.2 selection contract from the frozen v2.1
    # fixed-window accounting checks, which have their own direct tests.
    monkeypatch.setattr(
        runner.V22.V21, "leg_validation_reasons", lambda *args, **kwargs: []
    )
    return runner, runner.V22.load_and_validate_baseline("hidden")


def test_current_url_controller_baseline_is_selected_without_historical_pin(
    monkeypatch, tmp_path
) -> None:
    runner, baseline = _load_synthetic_hidden_baseline(
        monkeypatch, tmp_path, CURRENT_URL_CONTROLLER_SHA256
    )

    assert baseline["leg"]["controller_config_sha256"] == (
        CURRENT_URL_CONTROLLER_SHA256
    )
    assert all(
        "controller_config_sha256" not in spec
        for spec in runner.V22.BASELINE_SPECS.values()
    )
    assert not hasattr(runner.V22, "EXPECTED_CONTROLLER_CONFIG_SHA256")


@pytest.mark.parametrize(
    "malformed",
    [None, 7, "a" * 63, "A" * 64, "g" * 64],
)
def test_selected_baseline_rejects_malformed_controller_commitment(
    monkeypatch, tmp_path, malformed
) -> None:
    with pytest.raises(
        RuntimeError, match="Direct controller-config commitment is invalid"
    ):
        _load_synthetic_hidden_baseline(monkeypatch, tmp_path, malformed)


def test_candidate_controller_must_match_selected_baseline(monkeypatch) -> None:
    runner = _load_runner("kvbench_hidden_controller_pair_mismatch")
    baseline = _direct_leg(runner, CURRENT_URL_CONTROLLER_SHA256)
    candidate = dict(baseline)
    candidate["mode"] = "candidate"
    candidate["controller_config_sha256"] = "b" * 64
    candidate["valid_steps_per_min"] = 90.0
    monkeypatch.setattr(
        runner.V22.V21, "leg_validation_reasons", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(
        runner.V22.V21, "pair_validation_reasons", lambda *args, **kwargs: []
    )

    evaluation = runner.V22.candidate_evaluation(
        baseline_bundle={
            "suite": "hidden",
            "profile": runner.V22.BASELINE_SPECS["hidden"]["profile"],
            "leg": baseline,
            "direct_valid_steps_per_min": baseline["valid_steps_per_min"],
        },
        candidate=candidate,
        workload_nonce="synthetic",
    )

    assert evaluation["valid"] is False
    assert evaluation["qualification_passed"] is False
    assert evaluation["additional_identity_invalid_reasons"] == [
        "controller_config_mismatch"
    ]
    assert "controller_config_mismatch" in evaluation["all_invalid_reasons"]
