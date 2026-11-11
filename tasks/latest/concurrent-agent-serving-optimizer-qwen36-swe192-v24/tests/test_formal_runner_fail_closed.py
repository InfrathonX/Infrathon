from __future__ import annotations

import copy
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tarfile

import pytest

from conftest import load_module


TASK_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MLSYSFORGE_ROOT = str(TASK_ROOT.parents[2])
SCRIPTS = TASK_ROOT / "scripts"
VALIDATOR_PATH = SCRIPTS / "validate_final_evaluation.py"
PREFLIGHT_SCRIPT_PATH = SCRIPTS / "preflight.py"
NONCE_PATH = SCRIPTS / "prepare_nonce_source.py"
CLEANUP_PATH = SCRIPTS / "cleanup_formal_runtime.py"
COMPLETION_PATH = SCRIPTS / "write_formal_completion.py"
SELECTOR_PATH = SCRIPTS / "select_best.py"
SNAPSHOT_CHECK_PATH = SCRIPTS / "check_final_snapshot.py"
LAUNCHER_PATH = SCRIPTS / "run_fixed_budget.sh"
TASK_ID = "concurrent_agent_serving_optimizer_qwen36_swe192_v24"
SUBMISSION_ID = "formal-eval-a1"
BACKEND_SERVING_CONTRACT_SHA256 = (
    "18d27d97c736f90e1d74822e80ce1f212685e26627b31a08e47569a1457f52c6"
)
HIDDEN_NONCE_SHA256 = (
    "c525ae31a2bb1d3ed0232c345f9fc978fd52574d415aaa2777bead17f2fac20a"
)
REANCHORED_HIDDEN_DIRECT = 83.73333333333333
REANCHORED_HIDDEN_ORACLE = 108.2
HIDDEN_BASELINE_PAYLOAD = {
    "schema_version": 1,
    "status": "complete",
    "suite": "hidden",
    "profile": "hidden-pressure-192",
    "workload_nonce_sha256": HIDDEN_NONCE_SHA256,
    "legs": {
        "baseline": {
            "schema_version": 1,
            "mode": "baseline",
            "profile": "hidden-pressure-192",
            "workload_nonce_sha256": HIDDEN_NONCE_SHA256,
            "valid_steps_per_min": REANCHORED_HIDDEN_DIRECT,
            "model_provenance": {
                "profile_id": "qwen3.6-27b-6a9e13bd-h100-nvl-tp2-kv30g-v2",
                "profile_sha256": (
                    "7289ed315324b9b5165d4d680efb93004c59e0d34f0f98ef9d85406ed5a9153a"
                ),
                "model_revision": (
                    "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
                ),
                "runtime_image_id": (
                    "sha256:1d8369b9e3c91aa8cba5c7aeeef26e3c4afd455e5d2ce9aebe6f30f6dd9344bd"
                ),
                "tensor_parallel_size": 2,
                "kv_cache_dtype": "fp8",
                "backend_serving_contract_sha256": (
                    BACKEND_SERVING_CONTRACT_SHA256
                ),
            },
        }
    },
}
HIDDEN_BASELINE_BYTES = (
    json.dumps(HIDDEN_BASELINE_PAYLOAD, indent=2, sort_keys=True) + "\n"
).encode()
REANCHORED_HIDDEN_BASELINE_SHA256 = hashlib.sha256(
    HIDDEN_BASELINE_BYTES
).hexdigest()
HIDDEN_BASELINE_IDENTITY = {
    "file_sha256": REANCHORED_HIDDEN_BASELINE_SHA256,
    "valid_steps_per_min": REANCHORED_HIDDEN_DIRECT,
    "backend_serving_contract_sha256": BACKEND_SERVING_CONTRACT_SHA256,
    "workload_nonce_sha256": HIDDEN_NONCE_SHA256,
}
GATES = (
    "protocol_identity_fixed_window_lifecycle",
    "backend_one_to_one_audit",
    "paired_request_success",
    "paired_valid_step_ratio",
    "candidate_p99_gateway_wait",
    "paired_backend_work",
    "throughput_accounting",
)


@pytest.fixture(scope="module")
def validator():
    return load_module("kvbench_formal_validator", VALIDATOR_PATH)


@pytest.fixture(scope="module")
def nonce_helper():
    return load_module("kvbench_formal_nonce", NONCE_PATH)


@pytest.fixture(scope="module")
def cleanup_helper():
    return load_module("kvbench_formal_cleanup", CLEANUP_PATH)


@pytest.fixture(scope="module")
def preflight_helper():
    return load_module("kvbench_formal_preflight", PREFLIGHT_SCRIPT_PATH)


@pytest.fixture(scope="module")
def completion_helper():
    sys.path.insert(0, str(SCRIPTS))
    try:
        return load_module("kvbench_formal_completion", COMPLETION_PATH)
    finally:
        sys.path.remove(str(SCRIPTS))


def test_formal_helpers_bind_the_repository_root(
    preflight_helper, completion_helper
) -> None:
    repository_root = TASK_ROOT.parents[2]
    assert (repository_root / "pyproject.toml").is_file()
    assert (repository_root / "sforge").is_dir()
    assert preflight_helper.EXPECTED_MLSYSFORGE_ROOT == repository_root
    assert completion_helper.EXPECTED_MLSYSFORGE_ROOT == str(repository_root)


def test_validator_loads_selected_baseline_and_projects_legacy_provenance(
    validator, monkeypatch, tmp_path: Path
) -> None:
    baseline_path = tmp_path / "hidden-baseline.json"
    baseline_path.write_bytes(HIDDEN_BASELINE_BYTES)
    monkeypatch.setenv("KVBENCH_HIDDEN_BASELINE_PATH", str(baseline_path))

    validator.configure_baseline_contract()
    assert validator.direct_rate() == REANCHORED_HIDDEN_DIRECT
    assert validator.baseline_sha256() == hashlib.sha256(
        HIDDEN_BASELINE_BYTES
    ).hexdigest()

    legacy = copy.deepcopy(HIDDEN_BASELINE_PAYLOAD)
    provenance = legacy["legs"]["baseline"]["model_provenance"]
    provenance.pop("backend_serving_contract_sha256")
    provenance["runtime_boot_identity_sha256"] = "f" * 64
    _write_json(baseline_path, legacy)
    validator.configure_baseline_contract()

    provenance["profile_id"] = "unknown-legacy-profile"
    _write_json(baseline_path, legacy)
    with pytest.raises(validator.ValidationError, match="legacy baseline"):
        validator.configure_baseline_contract()


def test_display_rescale_uses_task_spec_not_run_selected_direct(
    validator, monkeypatch, tmp_path: Path
) -> None:
    selected_direct = 71.25
    assert selected_direct != REANCHORED_HIDDEN_DIRECT
    selected_baseline = copy.deepcopy(HIDDEN_BASELINE_PAYLOAD)
    selected_baseline["legs"]["baseline"]["valid_steps_per_min"] = selected_direct
    baseline_path = tmp_path / "run-selected-hidden-baseline.json"
    _write_json(baseline_path, selected_baseline)
    monkeypatch.setenv("KVBENCH_HIDDEN_BASELINE_PATH", str(baseline_path))
    monkeypatch.setattr(
        validator, "EXPECTED_BASELINE_SHA256", validator.EXPECTED_BASELINE_SHA256
    )
    monkeypatch.setattr(validator, "DIRECT_RATE", validator.DIRECT_RATE)
    monkeypatch.setattr(
        validator, "DISPLAY_RESCALE_LOWER", validator.DISPLAY_RESCALE_LOWER
    )
    monkeypatch.setattr(
        validator, "DISPLAY_RESCALE_UPPER", validator.DISPLAY_RESCALE_UPPER
    )

    validator.configure_display_rescale_contract()
    validator.configure_baseline_contract()
    task_spec = json.loads(validator.TASK_SPEC_PATH.read_text(encoding="utf-8"))
    display = task_spec["judge"]["rescale"]
    assert validator.direct_rate() == selected_direct
    assert validator.DISPLAY_RESCALE_LOWER == display["lower"]
    assert validator.DISPLAY_RESCALE_UPPER == display["upper"]

    report = _public_report(score=96.0)
    validator._validate_public_shape(
        report,
        expected_submission_id=SUBMISSION_ID,
        max_runtime=3060.0,
    )
    dynamic_baseline_rescale = max(
        0.0,
        min(
            100.0,
            100.0
            * (report["score"] - selected_direct)
            / (display["upper"] - selected_direct),
        ),
    )
    assert not math.isclose(
        report["score_0_100"], dynamic_baseline_rescale, abs_tol=1e-9
    )


def test_display_rescale_contract_rejects_task_spec_drift(
    validator, monkeypatch, tmp_path: Path
) -> None:
    task_spec = json.loads(validator.TASK_SPEC_PATH.read_text(encoding="utf-8"))
    task_spec["judge"]["rescale"]["lower"] = task_spec["judge"]["rescale"][
        "upper"
    ]
    drifted_path = tmp_path / "drifted-task.json"
    _write_json(drifted_path, task_spec)
    monkeypatch.setattr(validator, "TASK_SPEC_PATH", drifted_path)

    with pytest.raises(validator.ValidationError, match="rescale bounds"):
        validator.configure_display_rescale_contract()


def _public_report(*, score: float = 96.0) -> dict:
    relation = "above" if score > REANCHORED_HIDDEN_DIRECT else "at or below"
    details = [
        {
            "name": f"hard_gate:{name}",
            "status": "PASSED",
            "message": "required hard gate passed",
            "score": 1.0,
            "weight": 1.0,
        }
        for name in GATES
    ]
    return {
        "task_id": TASK_ID,
        "submission_id": SUBMISSION_ID,
        "total_tests": 7,
        "passed": 7,
        "failed": 0,
        "errors": 0,
        "pass_rate": 1.0,
        "score": score,
        "score_0_100": max(
            0.0,
            min(
                100.0,
                100.0
                * (score - REANCHORED_HIDDEN_DIRECT)
                / (REANCHORED_HIDDEN_ORACLE - REANCHORED_HIDDEN_DIRECT),
            ),
        ),
        "timed_out": False,
        "runtime_seconds": 2300.0,
        "test_details": [],
        "valid": True,
        "summary": (
            f"VALID: primary score valid_steps_per_min={score:.6f} (maximize), "
            f"{relation} frozen Direct={REANCHORED_HIDDEN_DIRECT:.1f}; "
            "all 7 hard gates passed. "
            "A below-Direct result remains valid and is ranked by raw throughput."
        ),
        "details": details,
        "metrics": {
            "source": "kvbench_candidate_only_vs_serving_contract_direct",
            "evaluation_contract": "candidate_only_same_nonce_same_serving_contract",
            "primary_score_name": "valid_steps_per_min",
            "primary_score_value": score,
            "primary_score_unit": "valid executable agent steps/min",
            "primary_score_direction": "maximize",
            "correctness_gate_mode": "binary",
            "correctness_gate_passed": True,
            "frozen_direct_valid_steps_per_min": REANCHORED_HIDDEN_DIRECT,
            "speedup_over_frozen_direct": score / REANCHORED_HIDDEN_DIRECT,
            "change_percent": (
                score / REANCHORED_HIDDEN_DIRECT - 1.0
            )
            * 100.0,
            "strictly_better_than_frozen_direct": (
                score > REANCHORED_HIDDEN_DIRECT
            ),
            "hard_gates_passed": 7,
            "hard_gates_total": 7,
            "hard_gate_failures": [],
            "evaluation_status": "valid",
            "auto_feedback_safe": True,
        },
        "submitted_at": 1_700_000_000.0,
    }


def _private_evidence(*, score: float = 96.0) -> dict:
    gate_metrics = {
        "protocol_identity_fixed_window_lifecycle": {
            "runner_invalid_reasons": []
        },
        "backend_one_to_one_audit": {
            "baseline_right_censor_reconciled": True,
            "candidate_right_censor_reconciled": True,
        },
        "paired_request_success": {
            "baseline": 0.99,
            "candidate": 0.99,
            "floor": 0.98,
        },
        "paired_valid_step_ratio": {
            "baseline": 0.96,
            "candidate": 0.96,
            "floor": 0.90,
        },
        "candidate_p99_gateway_wait": {
            "candidate_sec": 1.0,
            "maximum_sec": 720.0,
        },
        "paired_backend_work": {
            "baseline_tokens_per_valid_step": 100.0,
            "candidate_tokens_per_valid_step": 100.0,
            "ratio": 1.0,
            "minimum": 0.90,
            "maximum": None,
        },
        "throughput_accounting": {"candidate_valid_steps_per_min": score},
    }
    checks = [
        {
            "name": name,
            "passed": True,
            "message": "canonical hard gate passed",
            "metrics": gate_metrics[name],
        }
        for name in GATES
    ]
    evidence = {
        "schema_version": 1,
        "kind": "gpu67-v2.2-hidden-candidate-only-vs-frozen-v2.1-direct",
        "status": "complete",
        "phase": "quiescent",
        "started_at_utc": "2026-08-01T00:00:00Z",
        "completed_at_utc": "2026-08-01T00:40:00Z",
        "direct_leg_executed": False,
        "candidate_dir": "/private/candidate",
        "candidate_tree_sha256": "a" * 64,
        "preflight_backend_serving_contract_sha256": (
            BACKEND_SERVING_CONTRACT_SHA256
        ),
        "baseline_evidence": {
            "path": "/private/frozen-direct.json",
            "file_sha256": REANCHORED_HIDDEN_BASELINE_SHA256,
            "workload_nonce_sha256": HIDDEN_NONCE_SHA256,
            "direct_valid_steps_per_min": REANCHORED_HIDDEN_DIRECT,
            "backend_serving_contract_sha256": (
                BACKEND_SERVING_CONTRACT_SHA256
            ),
        },
        "leg": {
            "schema_version": 1,
            "mode": "candidate",
            "profile": "hidden-pressure-192",
            "workload_nonce_sha256": HIDDEN_NONCE_SHA256,
            "model_provenance": {
                "backend_serving_contract_sha256": (
                    BACKEND_SERVING_CONTRACT_SHA256
                )
            },
            "valid_steps_per_min": score,
        },
        "evaluation": {
            "suite": "hidden",
            "profile": "hidden-pressure-192",
            "backend_serving_contract_sha256": (
                BACKEND_SERVING_CONTRACT_SHA256
            ),
            "direct_valid_steps_per_min": REANCHORED_HIDDEN_DIRECT,
            "candidate_valid_steps_per_min": score,
            "valid": True,
            "qualification_passed": score > REANCHORED_HIDDEN_DIRECT,
        },
        "errors": [],
        "cleanup": {
            "attempted": True,
            "stack_stop_succeeded": True,
            "quiescent": True,
            "reconciled_before_start": False,
            "residue": {
                "compose_container_ids": [],
                "sandbox_container_ids": [],
            },
        },
        "limitations": [
            "Direct is frozen prior evidence and was not rerun.",
            "The original hidden workload nonce and stable backend serving contract are reused.",
            "Prefix-cache state and wall-clock time differ from the earlier Direct leg.",
        ],
        "mlsysforge_evaluation": {
            "schema_version": 1,
            "valid": True,
            "score": score,
            "primary_score_name": "valid_steps_per_min",
            "direct_valid_steps_per_min": REANCHORED_HIDDEN_DIRECT,
            "speedup_over_frozen_direct": score / REANCHORED_HIDDEN_DIRECT,
            "change_percent": (
                score / REANCHORED_HIDDEN_DIRECT - 1.0
            )
            * 100.0,
            "strictly_better_than_frozen_direct": (
                score > REANCHORED_HIDDEN_DIRECT
            ),
            "reasons": [],
            "checks": checks,
        },
        "mlsysforge_adapter": {
            "schema_version": 1,
            "candidate_only": True,
            "fresh_paired_ab": False,
            "canonical_additional_hard_gates_applied": True,
            "primary_score": "candidate_valid_steps_per_min",
        },
    }
    return _private_envelope(
        _public_report(score=score),
        classification="valid",
        candidate_tree_sha256="a" * 64,
        runner_evidence=evidence,
        runner={
            "returncode": 0 if score > REANCHORED_HIDDEN_DIRECT else 3,
            "timed_out": False,
            "cancel_signal": None,
            "termination_sent": False,
            "forced_kill": False,
            "runtime_seconds": 2300.0,
            "force_reconcile": None,
        },
        private_errors=[],
    )


def _structured_projection(report: dict) -> dict:
    return {
        key: report[key]
        for key in (
            "valid",
            "score",
            "pass_rate",
            "total_tests",
            "passed",
            "failed",
            "errors",
            "summary",
            "details",
            "metrics",
        )
    }


def _canonical_json_sha256(value: dict) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _private_envelope(
    report: dict,
    *,
    classification: str,
    candidate_tree_sha256: str | None,
    runner_evidence: dict | None,
    runner: dict,
    private_errors: list[str],
) -> dict:
    public_result = _structured_projection(report)
    return {
        "schema_version": 2,
        "kind": "kvbench-mlsysforge-private-evaluation",
        "status": "terminal",
        "classification": classification,
        "started_at_utc": "2026-08-01T00:00:00Z",
        "completed_at_utc": "2026-08-01T00:40:01Z",
        "candidate_tree_sha256": candidate_tree_sha256,
        "commitments": {
            "hidden_baseline_identity": HIDDEN_BASELINE_IDENTITY,
        },
        "runner": runner,
        "runner_evidence": runner_evidence,
        "cleanup": (
            runner_evidence.get("cleanup")
            if isinstance(runner_evidence, dict)
            else None
        ),
        "mlsysforge_evaluation": (
            runner_evidence.get("mlsysforge_evaluation")
            if isinstance(runner_evidence, dict)
            else None
        ),
        "public_result": public_result,
        "public_result_sha256": _canonical_json_sha256(public_result),
        "private_errors": private_errors,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_private(path: Path, payload: dict) -> None:
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo("kvbench-private-evidence.json")
        member.mode = 0o600
        member.size = len(raw)
        archive.addfile(member, BytesIO(raw))


def _snapshot_payload(archive_bytes: bytes) -> dict:
    archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()
    return {
        "schema_version": 2,
        "captured_at": 1_700_007_200.03,
        "snapshot_elapsed_seconds": 21600.01,
        "snapshot_boundary": "whole_container_frozen",
        "archive_sha256": archive_sha256,
        "deadline": {
            "at": 1_700_007_200.0,
            "elapsed_seconds": 21600.0,
            "monotonic": 7300.0,
        },
        "freeze": {
            "mechanism": "docker_stop_timeout_0",
            "all_work_processes_frozen": True,
            "all_work_processes_stopped": True,
            "was_running": True,
            "freeze_requested_at": 1_700_007_200.0,
            "freeze_requested_elapsed_seconds": 21600.0,
            "frozen_at": 1_700_007_200.01,
            "frozen_elapsed_seconds": 21600.01,
            "container_status": "exited",
            "container_pid": 0,
        },
        "capture": {
            "source": "stopped_container_filesystem",
            "transport": "docker_get_archive_transport_tar",
            "work_container_running_during_capture": False,
            "started_at": 1_700_007_200.02,
            "started_elapsed_seconds": 21600.02,
            "archive_completed_at": 1_700_007_200.03,
            "archive_completed_elapsed_seconds": 21600.03,
        },
        "archive": {
            "path": "final_archive.tar.gz",
            "format": "tar+gzip",
            "size_bytes": len(archive_bytes),
            "sha256": archive_sha256,
        },
    }


def _run_validator(
    tmp_path: Path,
    report: dict,
    private: dict | None,
    *,
    eval_rc: int = 0,
    evaluated_archive_bytes: bytes = b"evaluated-submission",
    expected_archive_bytes: bytes | None = None,
    preflight_overrides: dict[str, object] | None = None,
) -> tuple[subprocess.CompletedProcess[str], dict, Path, Path]:
    report_path = tmp_path / "report.json"
    artifact_path = tmp_path / "private.tar.gz"
    output_path = tmp_path / "validation.json"
    evaluated_archive_path = tmp_path / "submission.tar.gz"
    expected_archive_path = tmp_path / "expected-submission.tar.gz"
    baseline_path = tmp_path / "hidden-baseline.json"
    preflight_path = tmp_path / "preflight.json"
    baseline_path.write_bytes(HIDDEN_BASELINE_BYTES)
    preflight_payload = {
        "schema_version": 1,
        "status": "passed",
        "backend_serving_contract_sha256": BACKEND_SERVING_CONTRACT_SHA256,
        "baseline_file_sha256": REANCHORED_HIDDEN_BASELINE_SHA256,
        "baseline_valid_steps_per_min": REANCHORED_HIDDEN_DIRECT,
        "nonce_commitment_sha256": HIDDEN_NONCE_SHA256,
    }
    preflight_payload.update(preflight_overrides or {})
    _write_json(preflight_path, preflight_payload)
    _write_json(report_path, report)
    if private is not None:
        _write_private(artifact_path, private)
    evaluated_archive_path.write_bytes(evaluated_archive_bytes)
    expected_archive_path.write_bytes(
        evaluated_archive_bytes
        if expected_archive_bytes is None
        else expected_archive_bytes
    )
    completed = subprocess.run(
        [
            "python3",
            str(VALIDATOR_PATH),
            "--report",
            str(report_path),
            "--private-artifact",
            str(artifact_path),
            "--expected-submission-id",
            SUBMISSION_ID,
            "--eval-rc",
            str(eval_rc),
            "--evaluated-archive",
            str(evaluated_archive_path),
            "--expected-archive",
            str(expected_archive_path),
            "--expected-preflight",
            str(preflight_path),
            "--output",
            str(output_path),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            **os.environ,
            "KVBENCH_HIDDEN_BASELINE_PATH": str(baseline_path),
        },
        check=False,
    )
    return completed, json.loads(output_path.read_text()), report_path, artifact_path


def test_valid_final_report_and_private_evidence_are_authoritative(tmp_path: Path) -> None:
    completed, result, _, _ = _run_validator(
        tmp_path, _public_report(), _private_evidence()
    )
    assert completed.returncode == 0
    assert result["classification"] == "candidate_valid"
    assert result["authoritative"] is True
    assert result["valid"] is True
    assert result["score"] == 96.0
    assert len(result["report"]["sha256"]) == 64
    assert len(result["private_artifact"]["sha256"]) == 64
    assert result["hidden_baseline_identity"] == HIDDEN_BASELINE_IDENTITY


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("baseline_file_sha256", "f" * 64),
        ("baseline_valid_steps_per_min", REANCHORED_HIDDEN_DIRECT + 1.0),
        ("backend_serving_contract_sha256", "e" * 64),
        ("nonce_commitment_sha256", "d" * 64),
    ],
)
def test_validator_fails_closed_when_preflight_baseline_identity_differs(
    tmp_path: Path, field: str, value: object
) -> None:
    completed, result, _, _ = _run_validator(
        tmp_path,
        _public_report(),
        _private_evidence(),
        preflight_overrides={field: value},
    )

    assert completed.returncode == 22
    assert result["classification"] == "malformed"
    assert result["authoritative"] is False
    assert set(result["hidden_baseline_identity"]) == set(HIDDEN_BASELINE_IDENTITY)


def test_real_success_supervisor_envelope_is_authoritative(
    tmp_path: Path, judge_module, monkeypatch
) -> None:
    runner_log = tmp_path / "producer-runner.log"
    runner = judge_module._run_supervised(
        [sys.executable, "-c", "raise SystemExit(0)"],
        environment=dict(os.environ),
        runner_log=runner_log,
    )
    report = _public_report()
    evidence = copy.deepcopy(_private_evidence()["runner_evidence"])
    baseline_path = tmp_path / "producer-hidden-baseline.json"
    baseline_path.write_bytes(HIDDEN_BASELINE_BYTES)
    monkeypatch.setattr(judge_module, "BASELINE_PATH", baseline_path)
    private = judge_module._write_private_envelope(
        tmp_path / "producer-private.json",
        started_at_utc="2026-08-01T00:00:00Z",
        classification="valid",
        candidate_tree_sha256="a" * 64,
        runner=runner,
        runner_evidence=evidence,
        public_result=_structured_projection(report),
        private_errors=[],
        baseline_bundle=judge_module._load_frozen_baseline_bundle(),
    )

    completed, result, _, _ = _run_validator(tmp_path, report, private)

    assert completed.returncode == 0
    assert result["classification"] == "candidate_valid"
    assert result["authoritative"] is True
    assert result["valid"] is True


def test_runner_rc3_can_still_be_canonically_valid_but_below_direct(
    tmp_path: Path,
) -> None:
    completed, result, _, _ = _run_validator(
        tmp_path, _public_report(score=80.0), _private_evidence(score=80.0)
    )
    assert completed.returncode == 0
    assert result["classification"] == "candidate_valid"
    assert result["valid"] is True
    assert result["score"] == 80.0


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_gate",
        "duplicate_gate",
        "valid_conflict",
        "score_mismatch",
        "non_finite_score",
        "teardown_error",
        "missing_force_reconcile",
    ],
)
def test_final_validator_rejects_fail_open_shapes(tmp_path: Path, mutation: str) -> None:
    report = _public_report()
    private = _private_evidence()
    evidence = private["runner_evidence"]
    if mutation == "missing_gate":
        report["details"].pop()
        report["total_tests"] = 6
        report["passed"] = 6
        evidence["mlsysforge_evaluation"]["checks"].pop()
    elif mutation == "duplicate_gate":
        evidence["mlsysforge_evaluation"]["checks"][-1]["name"] = GATES[0]
    elif mutation == "valid_conflict":
        evidence["mlsysforge_evaluation"]["checks"][0]["passed"] = False
    elif mutation == "score_mismatch":
        evidence["leg"]["valid_steps_per_min"] = 95.0
    elif mutation == "non_finite_score":
        report["score"] = float("nan")
    elif mutation == "teardown_error":
        evidence["errors"] = ["cleanup: stack stop failed"]
    elif mutation == "missing_force_reconcile":
        private["runner"].pop("force_reconcile")
    completed, result, _, _ = _run_validator(tmp_path, report, private)
    assert completed.returncode == 22
    assert result["classification"] == "malformed"
    assert result["authoritative"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "classification_list",
        "cancel_signal_list",
        "detail_status_list",
        "gate_name_list",
        "runner_schema_bool",
        "projection_bool_int_confusion",
    ],
)
def test_malicious_json_types_always_produce_structured_malformed_result(
    tmp_path: Path, mutation: str
) -> None:
    report = _public_report()
    private = _private_evidence()
    if mutation == "classification_list":
        private["classification"] = []
    elif mutation == "cancel_signal_list":
        private["runner"]["cancel_signal"] = []
    elif mutation == "detail_status_list":
        report["details"][0]["status"] = []
    elif mutation == "gate_name_list":
        private["runner_evidence"]["mlsysforge_evaluation"]["checks"][0]["name"] = []
    elif mutation == "runner_schema_bool":
        private["runner_evidence"]["schema_version"] = True
    elif mutation == "projection_bool_int_confusion":
        private["public_result"]["valid"] = 1
        private["public_result_sha256"] = _canonical_json_sha256(
            private["public_result"]
        )
    completed, result, _, _ = _run_validator(tmp_path, report, private)
    assert completed.returncode == 22
    assert result["classification"] == "malformed"
    assert result["authoritative"] is False
    assert result["errors"]


def test_missing_private_artifact_is_never_authoritative(tmp_path: Path) -> None:
    completed, result, _, _ = _run_validator(tmp_path, _public_report(), None)
    assert completed.returncode == 23
    assert result["classification"] == "artifact_missing"


def test_timeout_is_distinct_even_when_sforge_report_is_incomplete(tmp_path: Path) -> None:
    report = {
        "task_id": TASK_ID,
        "submission_id": SUBMISSION_ID,
        "timed_out": True,
        "valid": True,
    }
    completed, result, _, _ = _run_validator(tmp_path, report, None)
    assert completed.returncode == 21
    assert result["classification"] == "timeout"
    assert result["authoritative"] is False


def test_nonzero_evaluator_exit_overrides_apparently_valid_report(tmp_path: Path) -> None:
    completed, result, _, _ = _run_validator(
        tmp_path, _public_report(), _private_evidence(), eval_rc=9
    )
    assert completed.returncode == 24
    assert result["classification"] == "evaluation_process_error"
    assert result["authoritative"] is False


def test_evaluated_archive_must_match_expected_candidate_bytes(tmp_path: Path) -> None:
    completed, result, _, _ = _run_validator(
        tmp_path,
        _public_report(),
        _private_evidence(),
        expected_archive_bytes=b"tampered-expected-archive",
    )
    assert completed.returncode == 22
    assert result["classification"] == "malformed"
    assert result["authoritative"] is False
    assert any("differs from expected archive" in item for item in result["errors"])


def test_schema2_snapshot_requires_active_whole_container_freeze_and_archive_hash(
    tmp_path: Path,
) -> None:
    archive_bytes = b"frozen-final-archive"
    archive = tmp_path / "final_archive.tar.gz"
    snapshot = tmp_path / "final_snapshot.json"
    archive.write_bytes(archive_bytes)
    payload = _snapshot_payload(archive_bytes)
    _write_json(snapshot, payload)

    command = [
        "python3",
        str(SNAPSHOT_CHECK_PATH),
        str(snapshot),
        str(archive),
        "21600",
    ]
    accepted = subprocess.run(command, text=True, capture_output=True, check=False)
    assert accepted.returncode == 0, accepted.stderr

    payload["freeze"]["container_pid"] = 17
    _write_json(snapshot, payload)
    live_pid = subprocess.run(command, text=True, capture_output=True, check=False)
    assert live_pid.returncode != 0
    assert "process-quiescent" in live_pid.stderr

    payload["freeze"]["container_pid"] = 0
    payload["freeze"]["freeze_requested_elapsed_seconds"] = 7205.01
    _write_json(snapshot, payload)
    late_freeze = subprocess.run(command, text=True, capture_output=True, check=False)
    assert late_freeze.returncode != 0
    assert "outside the deadline window" in late_freeze.stderr

    payload["freeze"]["freeze_requested_elapsed_seconds"] = 21600.0
    _write_json(snapshot, payload)
    archive.write_bytes(b"archive-mutated-after-snapshot")
    hash_mismatch = subprocess.run(command, text=True, capture_output=True, check=False)
    assert hash_mismatch.returncode != 0
    assert "commitment does not match" in hash_mismatch.stderr


def _failure_report(kind: str) -> tuple[dict, dict]:
    if kind == "candidate":
        detail = {
            "name": "hard_gate:submission_contract",
            "status": "FAILED",
            "message": "launch_not_executable",
            "score": 0.0,
            "weight": 1.0,
        }
        metrics = {
            "source": "kvbench_mlsysforge_adapter",
            "primary_score_name": "valid_steps_per_min",
            "primary_score_value": 0.0,
            "primary_score_direction": "maximize",
            "correctness_gate_passed": False,
            "evaluation_status": "candidate_invalid",
            "submission_failure_code": "launch_not_executable",
            "auto_feedback_safe": True,
        }
        counts = (0, 1, 0)
        summary = "INVALID SUBMISSION: packaging gate failed (launch_not_executable)"
    else:
        public_message = (
            "trusted evaluation infrastructure did not produce a complete result; "
            "the private artifact contains the operator diagnostic"
        )
        detail = {
            "name": "judge:infrastructure",
            "status": "ERROR",
            "message": public_message,
            "score": 0.0,
            "weight": 1.0,
        }
        metrics = {
            "source": "kvbench_mlsysforge_adapter",
            "primary_score_name": "valid_steps_per_min",
            "primary_score_value": 0.0,
            "primary_score_direction": "maximize",
            "correctness_gate_passed": False,
            "evaluation_status": "resource_busy",
            "auto_feedback_safe": False,
        }
        counts = (0, 0, 1)
        summary = "JUDGE INFRASTRUCTURE ERROR: " + public_message
    report = {
        "task_id": TASK_ID,
        "submission_id": SUBMISSION_ID,
        "total_tests": 1,
        "passed": counts[0],
        "failed": counts[1],
        "errors": counts[2],
        "pass_rate": 0.0,
        "score": 0.0,
        "score_0_100": 0.0,
        "timed_out": False,
        "runtime_seconds": 1.0,
        "test_details": [],
        "valid": False,
        "summary": summary,
        "details": [detail],
        "metrics": metrics,
        "submitted_at": 1_700_000_000.0,
    }
    if kind == "candidate":
        private = _private_envelope(
            report,
            classification="candidate_invalid",
            candidate_tree_sha256=None,
            runner_evidence=None,
            runner={
                "returncode": None,
                "timed_out": False,
                "cancel_signal": None,
                "termination_sent": False,
                "forced_kill": False,
                "runtime_seconds": 0.0,
                "force_reconcile": None,
            },
            private_errors=["SubmissionError: launch_not_executable"],
        )
    else:
        cleanup = {
            "attempted": False,
            "stack_stop_succeeded": False,
            "quiescent": True,
            "reconciled_before_start": False,
            "residue": {
                "compose_container_ids": [],
                "sandbox_container_ids": [],
            },
        }
        evidence = {
            "schema_version": 1,
            "kind": "gpu67-v2.2-hidden-candidate-only-vs-frozen-v2.1-direct",
            "status": "resource_busy",
            "phase": "terminal",
            "started_at_utc": "2026-08-01T00:00:00Z",
            "completed_at_utc": "2026-08-01T00:00:01Z",
            "direct_leg_executed": False,
            "candidate_dir": "/private/candidate",
            "candidate_tree_sha256": "a" * 64,
            "preflight_backend_serving_contract_sha256": None,
            "baseline_evidence": {
                "path": "/private/frozen-direct.json",
                "file_sha256": REANCHORED_HIDDEN_BASELINE_SHA256,
                "workload_nonce_sha256": HIDDEN_NONCE_SHA256,
                "direct_valid_steps_per_min": REANCHORED_HIDDEN_DIRECT,
                "backend_serving_contract_sha256": (
                    BACKEND_SERVING_CONTRACT_SHA256
                ),
            },
            "leg": None,
            "evaluation": None,
            "errors": ["BlockingIOError: measurement lock is already held"],
            "cleanup": cleanup,
            "limitations": ["Direct is frozen prior evidence and was not rerun."],
        }
        private = _private_envelope(
            report,
            classification="resource_busy",
            candidate_tree_sha256="a" * 64,
            runner_evidence=evidence,
            runner={
                "returncode": 4,
                "timed_out": False,
                "cancel_signal": None,
                "termination_sent": False,
                "forced_kill": False,
                "runtime_seconds": 0.1,
                "force_reconcile": None,
            },
            private_errors=[
                "RuntimeError: trusted runner returned a non-terminal exit"
            ],
        )
    return report, private


def test_authoritative_candidate_invalid_is_distinct_from_infrastructure(tmp_path: Path) -> None:
    report, private = _failure_report("candidate")
    completed, result, _, _ = _run_validator(tmp_path, report, private)
    assert completed.returncode == 0
    assert result["classification"] == "candidate_invalid"
    assert result["authoritative"] is True
    assert result["valid"] is False


def test_only_exact_lock_nb_signature_is_retryable(tmp_path: Path) -> None:
    report, private = _failure_report("infrastructure")
    completed, result, _, _ = _run_validator(tmp_path, report, private)
    assert completed.returncode == 75
    assert result["classification"] == "measurement_lock_contention"
    assert result["retryable"] is True
    assert result["authoritative"] is False


def test_external_lock_holder_residue_is_not_a_retryable_self_handoff(
    tmp_path: Path,
) -> None:
    report, private = _failure_report("infrastructure")
    cleanup = private["runner_evidence"]["cleanup"]
    cleanup["quiescent"] = False
    cleanup["residue"]["compose_container_ids"] = ["d" * 64]
    private["cleanup"] = copy.deepcopy(cleanup)
    completed, result, _, _ = _run_validator(tmp_path, report, private)
    assert completed.returncode == 22
    assert result["classification"] == "malformed"
    assert result["retryable"] is False


def test_dedicated_nonce_is_minimal_private_and_original_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nonce_helper
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(mode=0o700)
    source = tmp_path / "source.log"
    hidden = "a" * 64
    public = "b" * 64
    source.write_text(
        f"unrelated\nhidden={hidden}\npublic={public}\n"
        f"duplicate-hidden={hidden}\nduplicate-public={public}\n"
    )
    source.chmod(0o644)
    original = source.read_bytes()
    original_mode = stat.S_IMODE(source.stat().st_mode)
    monkeypatch.setattr(
        nonce_helper, "HIDDEN_COMMITMENT", nonce_helper._sha256_bytes(hidden.encode())
    )
    monkeypatch.setattr(
        nonce_helper, "PUBLIC_COMMITMENT", nonce_helper._sha256_bytes(public.encode())
    )
    dedicated = run_root / "hidden-nonce-source.txt"

    nonce_helper.create(source, dedicated, run_root)
    assert dedicated.read_bytes() == (hidden + "\n" + public + "\n").encode()
    assert dedicated.stat().st_size == 130
    assert dedicated.stat().st_uid == os.getuid()
    assert stat.S_IMODE(dedicated.stat().st_mode) == 0o600
    assert nonce_helper.validate_dedicated(dedicated, expected_parent=run_root) == (
        hidden,
        public,
    )
    assert source.read_bytes() == original
    assert stat.S_IMODE(source.stat().st_mode) == original_mode

    nonce_helper.delete(dedicated, run_root)
    assert not dedicated.exists()
    assert source.read_bytes() == original
    assert stat.S_IMODE(source.stat().st_mode) == original_mode


@pytest.mark.parametrize("present", ["hidden", "public"])
def test_dedicated_nonce_create_requires_both_commitments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nonce_helper,
    present: str,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(mode=0o700)
    hidden = "c" * 64
    public = "d" * 64
    monkeypatch.setattr(
        nonce_helper, "HIDDEN_COMMITMENT", nonce_helper._sha256_bytes(hidden.encode())
    )
    monkeypatch.setattr(
        nonce_helper, "PUBLIC_COMMITMENT", nonce_helper._sha256_bytes(public.encode())
    )
    source = tmp_path / "source.log"
    source.write_text((hidden if present == "hidden" else public) + "\n")

    with pytest.raises(nonce_helper.NonceSourceError, match="committed value"):
        nonce_helper.create(source, run_root / "nonce-source", run_root)
    assert not (run_root / "nonce-source").exists()


@pytest.mark.parametrize(
    "variant",
    [
        "reversed",
        "duplicate",
        "third_line",
        "crlf",
        "missing_final_lf",
        "uppercase",
        "wrong_public",
    ],
)
def test_dedicated_nonce_rejects_noncanonical_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nonce_helper,
    variant: str,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(mode=0o700)
    hidden = "1" * 64
    public = "e" * 64
    monkeypatch.setattr(
        nonce_helper, "HIDDEN_COMMITMENT", nonce_helper._sha256_bytes(hidden.encode())
    )
    monkeypatch.setattr(
        nonce_helper, "PUBLIC_COMMITMENT", nonce_helper._sha256_bytes(public.encode())
    )
    payloads = {
        "reversed": f"{public}\n{hidden}\n",
        "duplicate": f"{hidden}\n{hidden}\n",
        "third_line": f"{hidden}\n{public}\n{'f' * 64}\n",
        "crlf": f"{hidden}\r\n{public}\r\n",
        "missing_final_lf": f"{hidden}\n{public}",
        "uppercase": f"{hidden}\n{public.upper()}\n",
        "wrong_public": f"{hidden}\n{'0' * 64}\n",
    }
    dedicated = run_root / "nonce-source"
    dedicated.write_bytes(payloads[variant].encode())
    dedicated.chmod(0o600)

    with pytest.raises(nonce_helper.NonceSourceError):
        nonce_helper.validate_dedicated(dedicated, expected_parent=run_root)


def test_dedicated_nonce_rejects_symlink_and_hardlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nonce_helper
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(mode=0o700)
    hidden = "2" * 64
    public = "3" * 64
    monkeypatch.setattr(
        nonce_helper, "HIDDEN_COMMITMENT", nonce_helper._sha256_bytes(hidden.encode())
    )
    monkeypatch.setattr(
        nonce_helper, "PUBLIC_COMMITMENT", nonce_helper._sha256_bytes(public.encode())
    )
    real = run_root / "real"
    real.write_text(hidden + "\n" + public + "\n")
    real.chmod(0o600)
    symlink = run_root / "symlink"
    symlink.symlink_to(real)
    with pytest.raises(nonce_helper.NonceSourceError):
        nonce_helper.validate_dedicated(symlink, expected_parent=run_root)

    hardlink = run_root / "hardlink"
    os.link(real, hardlink)
    with pytest.raises(nonce_helper.NonceSourceError, match="singly-linked"):
        nonce_helper.validate_dedicated(real, expected_parent=run_root)


def test_trusted_source_attestation_binds_content_mode_and_rejects_symlink(
    tmp_path: Path, preflight_helper
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    first = runtime / "runner.py"
    second = runtime / "stack.sh"
    first.write_text("print('frozen')\n")
    second.write_text("#!/bin/sh\nexit 0\n")
    second.chmod(0o755)

    initial = preflight_helper.attest_tree(runtime)
    assert initial["file_count"] == 2
    assert initial["total_bytes"] == first.stat().st_size + second.stat().st_size
    assert len(initial["tree_sha256"]) == 64

    first.write_text("print('mutated')\n")
    mutated = preflight_helper.attest_tree(runtime)
    assert mutated["tree_sha256"] != initial["tree_sha256"]
    second.chmod(0o700)
    mode_mutated = preflight_helper.attest_tree(runtime)
    assert mode_mutated["tree_sha256"] != mutated["tree_sha256"]

    (runtime / "unsafe-link").symlink_to(first)
    with pytest.raises(RuntimeError, match="contains a symlink"):
        preflight_helper.attest_tree(runtime)


def test_harness_attestation_is_fixed_ignores_generated_and_rejects_unsafe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preflight_helper,
) -> None:
    mlsysforge = tmp_path / "MLSys Forge"
    harness = mlsysforge / "sforge" / "harness"
    harness.mkdir(parents=True)
    source = harness / "run_agent.py"
    source.write_text("DEADLINE = 21600\n")
    root_pyc = mlsysforge / "sforge" / "generated.pyc"
    root_pyc.write_bytes(b"generated-one")
    cache = harness / "__pycache__"
    cache.mkdir()
    cached = cache / "run_agent.cpython-312.pyc"
    cached.write_bytes(b"cached-one")
    monkeypatch.setattr(preflight_helper, "EXPECTED_MLSYSFORGE_ROOT", mlsysforge)

    initial = preflight_helper.attest_mlsysforge_harness(mlsysforge)
    root_pyc.write_bytes(b"generated-two")
    cached.write_bytes(b"cached-two")
    assert preflight_helper.attest_mlsysforge_harness(mlsysforge) == initial

    source.write_text("DEADLINE = 1\n")
    assert preflight_helper.attest_mlsysforge_harness(mlsysforge) != initial

    unsafe_link = harness / "docker_backend.py"
    unsafe_link.symlink_to(source)
    with pytest.raises(RuntimeError, match="contains a symlink"):
        preflight_helper.attest_mlsysforge_harness(mlsysforge)
    unsafe_link.unlink()

    unsafe_fifo = harness / "judge_server.py"
    os.mkfifo(unsafe_fifo)
    with pytest.raises(RuntimeError, match="contains a special file"):
        preflight_helper.attest_mlsysforge_harness(mlsysforge)


def test_harness_attestation_rejects_alternate_or_symlinked_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preflight_helper,
) -> None:
    fixed = tmp_path / "fixed" / "MLSys Forge"
    (fixed / "sforge").mkdir(parents=True)
    (fixed / "sforge" / "run.py").write_text("pass\n")
    alternate = tmp_path / "alternate" / "MLSys Forge"
    (alternate / "sforge").mkdir(parents=True)
    (alternate / "sforge" / "run.py").write_text("pass\n")
    monkeypatch.setattr(preflight_helper, "EXPECTED_MLSYSFORGE_ROOT", fixed)

    with pytest.raises(RuntimeError, match="fixed formal path"):
        preflight_helper.attest_mlsysforge_harness(alternate)

    alias = tmp_path / "mlsysforge-alias"
    alias.symlink_to(fixed, target_is_directory=True)
    monkeypatch.setattr(preflight_helper, "EXPECTED_MLSYSFORGE_ROOT", alias)
    with pytest.raises(RuntimeError, match="symlink component"):
        preflight_helper.attest_mlsysforge_harness(alias)


def test_final_trusted_source_verification_rejects_two_hour_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preflight_helper
) -> None:
    task_root = tmp_path / "task"
    runtime = task_root / "runtime" / "frozen"
    runtime.mkdir(parents=True)
    (runtime / "runner.py").write_text("VALUE = 1\n")
    mlsysforge = tmp_path / "MLSys Forge"
    mlsysforge_source = mlsysforge / "sforge" / "harness" / "run_agent.py"
    mlsysforge_source.parent.mkdir(parents=True)
    mlsysforge_source.write_text("DEADLINE = 21600\n")
    for root_name in preflight_helper.TRUSTED_EXECUTION_ROOTS:
        source_root = task_root / root_name
        source_root.mkdir()
        (source_root / "source.txt").write_text(f"{root_name}\n")
    monkeypatch.setattr(preflight_helper, "ROOT", task_root)
    monkeypatch.setattr(preflight_helper, "FROZEN_RUNTIME_SOURCES", runtime)
    monkeypatch.setattr(preflight_helper, "EXPECTED_MLSYSFORGE_ROOT", mlsysforge)
    monkeypatch.setattr(
        preflight_helper,
        "validate_mlsysforge_import_origins",
        lambda _root: dict(preflight_helper.MLSYSFORGE_REQUIRED_IMPORTS),
    )
    initial = {
        "schema_version": 1,
        "status": "passed",
        "trusted_runtime_sources": preflight_helper.attest_tree(runtime),
        "trusted_execution_sources": preflight_helper.attest_execution_sources(
            task_root
        ),
        "mlsysforge_root": str(mlsysforge),
        "mlsysforge_generated_exclusions": list(
            preflight_helper.MLSYSFORGE_GENERATED_EXCLUSIONS
        ),
        "mlsysforge_import_origins": dict(
            preflight_helper.MLSYSFORGE_REQUIRED_IMPORTS
        ),
        "mlsysforge_harness_sources": preflight_helper.attest_mlsysforge_harness(
            mlsysforge
        ),
    }
    preflight = tmp_path / "preflight.json"
    _write_json(preflight, initial)
    assert preflight_helper.verify_trusted_sources(preflight, mlsysforge)["status"] == "passed"

    (task_root / "scripts" / "source.txt").write_text("changed during run\n")
    with pytest.raises(RuntimeError, match="execution sources changed"):
        preflight_helper.verify_trusted_sources(preflight, mlsysforge)

    (task_root / "scripts" / "source.txt").write_text("scripts\n")
    mlsysforge_source.write_text("DEADLINE = 1\n")
    with pytest.raises(RuntimeError, match="MLSys Forge harness sources changed"):
        preflight_helper.verify_trusted_sources(preflight, mlsysforge)


def test_completion_provenance_parser_rejects_missing_or_mutated_harness_attestation(
    tmp_path: Path,
    completion_helper,
) -> None:
    attestation = {
        "algorithm": "sha256-canonical-json-path-mode-size-content-v1",
        "tree_sha256": "d" * 64,
        "file_count": 5,
        "total_bytes": 100,
    }
    preflight_payload = {
        "schema_version": 1,
        "status": "passed",
        "backend_serving_contract_sha256": (
            BACKEND_SERVING_CONTRACT_SHA256
        ),
        "baseline_file_sha256": "b" * 64,
        "baseline_valid_steps_per_min": REANCHORED_HIDDEN_DIRECT,
        "nonce_commitment_sha256": "c" * 64,
        "matching_nonce_count": 1,
        "public_baseline_file_sha256": "f" * 64,
        "public_baseline_valid_steps_per_min": 103.3,
        "public_nonce_commitment_sha256": "e" * 64,
        "public_matching_nonce_count": 1,
        "trusted_runtime_sources": attestation,
        "trusted_execution_sources": attestation,
        "mlsysforge_root": completion_helper.EXPECTED_MLSYSFORGE_ROOT,
        "mlsysforge_generated_exclusions": (
            completion_helper.EXPECTED_MLSYSFORGE_GENERATED_EXCLUSIONS
        ),
        "mlsysforge_import_origins": (
            completion_helper.EXPECTED_MLSYSFORGE_IMPORT_ORIGINS
        ),
        "mlsysforge_harness_sources": attestation,
    }
    preflight = tmp_path / "preflight.json"
    _write_json(preflight, preflight_payload)
    parsed, errors = completion_helper._preflight(preflight)
    assert errors == []

    missing = dict(preflight_payload)
    missing.pop("mlsysforge_harness_sources")
    _write_json(tmp_path / "missing.json", missing)
    _, errors = completion_helper._preflight(tmp_path / "missing.json")
    assert "preflight:mlsysforge_harness_sources_schema" in errors

    verification_payload = {
        "schema_version": 1,
        "status": "passed",
        "attested_preflight_sha256": hashlib.sha256(
            preflight.read_bytes()
        ).hexdigest(),
        "trusted_runtime_sources": attestation,
        "trusted_execution_sources": attestation,
        "mlsysforge_root": completion_helper.EXPECTED_MLSYSFORGE_ROOT,
        "mlsysforge_generated_exclusions": (
            completion_helper.EXPECTED_MLSYSFORGE_GENERATED_EXCLUSIONS
        ),
        "mlsysforge_import_origins": (
            completion_helper.EXPECTED_MLSYSFORGE_IMPORT_ORIGINS
        ),
        "mlsysforge_harness_sources": {**attestation, "tree_sha256": "e" * 64},
    }
    verification = tmp_path / "verification.json"
    _write_json(verification, verification_payload)
    _, errors = completion_helper._trusted_source_verification(
        verification,
        parsed,
    )
    assert "trusted_source_verification:mlsysforge_harness_sources_mismatch" in errors


def _inspect_fixture(tmp_path: Path) -> tuple[dict, dict]:
    nonce = tmp_path / "run" / "hidden-nonce-source.txt"
    migration = tmp_path / "migration"
    eval_root = tmp_path / "eval"
    scratch = eval_root / ("judge-" + "a" * 24)
    fixture = {
        "Id": "b" * 64,
        "Name": f"/kvbench_qwen36.judge.{TASK_ID}.formal-eval-a1",
        "Config": {
            "Image": f"kvbench_qwen36.judge.{TASK_ID}:local",
            "Labels": {"mlsysforge.kvbench.role": "trusted-judge"},
        },
        "Mounts": [
            {
                "Type": "bind",
                "Source": str(nonce),
                "Destination": "/run/kvbench-secrets/nonce-source.jsonl",
                "RW": False,
            },
            {
                "Type": "bind",
                "Source": str(migration),
                "Destination": str(migration),
                "RW": False,
            },
            {
                "Type": "bind",
                "Source": str(scratch),
                "Destination": str(scratch),
                "RW": True,
            },
        ],
    }
    args = {
        "nonce_source": nonce,
        "migration_root": migration,
        "eval_root": eval_root,
        "judge_image": f"kvbench_qwen36.judge.{TASK_ID}:local",
    }
    return fixture, args


def test_cleanup_ownership_guard_requires_name_label_image_and_unique_nonce_mount(
    tmp_path: Path, cleanup_helper
) -> None:
    fixture, args = _inspect_fixture(tmp_path)
    owned = cleanup_helper.owned_judge(fixture, **args)
    assert owned == {
        "id": "b" * 64,
        "name": fixture["Name"],
        "scratch": fixture["Mounts"][2]["Source"],
    }

    unrelated = copy.deepcopy(fixture)
    unrelated["Mounts"][0]["Source"] = str(tmp_path / "another-run" / "nonce")
    assert cleanup_helper.owned_judge(unrelated, **args) is None

    wrong_migration = copy.deepcopy(fixture)
    wrong_migration["Mounts"][1]["RW"] = True
    with pytest.raises(cleanup_helper.CleanupError):
        cleanup_helper.owned_judge(wrong_migration, **args)


def test_launcher_has_fail_closed_signal_lock_and_nonce_ordering() -> None:
    source = LAUNCHER_PATH.read_text()
    assert "umask 077" in source
    assert "setsid --wait" in source
    assert "kill -INT -- \"-${active_pgid}\"" in source
    assert "kill -TERM -- \"-${active_pgid}\"" in source
    assert "kill -KILL -- \"-${active_pgid}\"" in source
    assert source.index("stop_judge_server\nexec 7>") < source.index(
        "start_logged \"${TASK_RUN_ROOT}/final-120m-eval"
    )
    final_section = source[source.index('stage="final_evaluation"') :]
    assert final_section.index(
        "start_logged \"${TASK_RUN_ROOT}/final-120m-eval"
    ) < final_section.index(
        "flock -u 7"
    )
    assert source.index("--nonce-source \"${ORIGINAL_NONCE_SOURCE}\"") < source.index(
        "export SFORGE_KVBENCH_NONCE_SOURCE=\"${DEDICATED_NONCE_SOURCE}\""
    )
    assert source.count('--expected-preflight "${PREFLIGHT_PATH}"') == 3
    assert '--experiment-manifest "${EXPERIMENT_MANIFEST_PATH}"' in source
    assert 'install -m 0400 -- "${HIDDEN_BASELINE_PATH}"' in source
    assert 'install -m 0400 -- "${PUBLIC_BASELINE_PATH}"' in source
    assert 'export KVBENCH_HIDDEN_BASELINE_PATH="${HIDDEN_BASELINE_PATH}"' in source
    assert 'export KVBENCH_PUBLIC_BASELINE_PATH="${PUBLIC_BASELINE_PATH}"' in source
    assert "hidden_baseline_identity: $hidden_baseline_identity" in source
    assert "preflight_sha256: $preflight_sha256" in source
    assert ".preserved_report" in source
    assert ".preserved_private_artifact" in source
    assert ".preserved_archive" in source
    assert 'NONCE_RUNTIME_PARENT="/run/user/$(id -u)"' in source
    assert 'NONCE_RUNTIME_DIR="$(mktemp -d -- ' in source
    manifest_section = source[source.index("jq -n ") : source.index('stage="agent_run"')]
    assert "dedicated_nonce_file_sha256" not in manifest_section
    for secret_or_endpoint in (
        "VLLM_BASE_URL",
        "KVBENCH_VLLM_BASE_URL",
        "CODEX_AUTH_JSON_PATH",
        "DEDICATED_NONCE_SOURCE",
        "ORIGINAL_NONCE_SOURCE",
    ):
        assert secret_or_endpoint not in manifest_section
    assert "trusted_runtime_sources" in manifest_section
    assert "trusted_execution_sources" in manifest_section
    assert "mlsysforge_harness_sources" in manifest_section
    assert "mlsysforge_generated_exclusions" in manifest_section
    assert "mlsysforge_import_origins" in manifest_section
    assert 'EXPECTED_MLSYSFORGE_ROOT="${REPOSITORY_ROOT}"' in source
    assert '[[ "${MLSYSFORGE_ROOT}" == "${EXPECTED_MLSYSFORGE_ROOT}" ]]' in source
    host_proxy_unset = (
        "unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy "
    )
    assert host_proxy_unset in source
    assert source.index(host_proxy_unset) < source.index('stage="agent_run"')
    assert '--allowed-peer "${proxy_bridge_ip}"' in source
    assert '--upstream-proxy-host "${docker_bridge_gateway}"' in source
    assert 'https://api.openai.com/v1/models' in source
    assert source.index("--verify-trusted-sources") < source.index(
        'completion_requested="passed"'
    )
    cleanup_source = CLEANUP_PATH.read_text()
    assert "qwen36-swe192-v2-1-gpu67-validation" in cleanup_source
    assert "harbor-kv-qwen36-27b-host" not in cleanup_source


def test_launcher_requires_marked_agent_runtime_and_records_identity() -> None:
    source = LAUNCHER_PATH.read_text()

    assert 'AGENT_RUNTIME_DIR="${SFORGE_AGENT_RUNTIME_DIR:-}"' in source
    assert ".sforge-agent-runtime.json" in source
    assert '"${runtime_kind}" == "sforge-agent-runtime"' in source
    assert '"${runtime_agent}" == "codex"' in source
    assert '"${runtime_version}" == "${CODEX_CLI_VERSION}"' in source
    # The guard runs during static validation, before images are trusted and
    # long before the agent stage.
    assert source.index(
        'AGENT_RUNTIME_DIR="${SFORGE_AGENT_RUNTIME_DIR:-}"'
    ) < source.index("assert_built_images_current")
    manifest_section = source[
        source.index("jq -n ") : source.index('stage="agent_run"')
    ]
    assert "agent_runtime: {" in manifest_section
    for field in (
        "dir: $runtime_dir",
        "agent: $runtime_agent",
        "version: $runtime_version",
        "tree_sha256: $runtime_tree_sha256",
    ):
        assert field in manifest_section


def test_completion_writer_passes_only_with_all_hash_bound_provenance(
    tmp_path: Path,
) -> None:
    report = _public_report()
    private = _private_evidence()
    _, validation, report_path, artifact_path = _run_validator(
        tmp_path, report, private
    )
    validation_path = tmp_path / "validation.json"
    assert validation["classification"] == "candidate_valid"
    assert artifact_path.is_file()

    preflight = tmp_path / "preflight.json"
    runtime_attestation = {
        "algorithm": "sha256-canonical-json-path-mode-size-content-v1",
        "tree_sha256": "d" * 64,
        "file_count": 52,
        "total_bytes": 760_000,
    }
    execution_attestation = {
        "algorithm": "sha256-canonical-json-path-mode-size-content-v1",
        "tree_sha256": "e" * 64,
        "file_count": 24,
        "total_bytes": 180_000,
    }
    mlsysforge_attestation = {
        "algorithm": "sha256-canonical-json-path-mode-size-content-v1",
        "tree_sha256": "f" * 64,
        "file_count": 41,
        "total_bytes": 310_000,
    }
    _write_json(
        preflight,
        {
            "schema_version": 1,
            "status": "passed",
            "backend_serving_contract_sha256": (
                BACKEND_SERVING_CONTRACT_SHA256
            ),
                "baseline_file_sha256": REANCHORED_HIDDEN_BASELINE_SHA256,
                "baseline_valid_steps_per_min": REANCHORED_HIDDEN_DIRECT,
                "nonce_commitment_sha256": HIDDEN_NONCE_SHA256,
                "matching_nonce_count": 1,
                "public_baseline_file_sha256": "f" * 64,
                "public_baseline_valid_steps_per_min": 103.3,
                "public_nonce_commitment_sha256": "e" * 64,
                "public_matching_nonce_count": 1,
            "trusted_runtime_sources": runtime_attestation,
            "trusted_execution_sources": execution_attestation,
            "mlsysforge_root": EXPECTED_MLSYSFORGE_ROOT,
            "mlsysforge_generated_exclusions": [
                "directory-component:__pycache__",
                "file-suffix:.pyc",
            ],
            "mlsysforge_import_origins": {
                "sforge": "sforge/__init__.py",
                "sforge.cli": "sforge/cli.py",
                "sforge.harness.run_agent": "sforge/harness/run_agent.py",
                "sforge.harness.backend.docker_backend": (
                    "sforge/harness/backend/docker_backend.py"
                ),
                "sforge.harness.judge_server": "sforge/harness/judge_server.py",
                "sforge.harness.run_evaluation": (
                    "sforge/harness/run_evaluation.py"
                ),
                "sforge.harness.task_spec": "sforge/harness/task_spec.py",
            },
            "mlsysforge_harness_sources": mlsysforge_attestation,
        },
    )
    source_verification = tmp_path / "trusted-source-verification.json"
    _write_json(
        source_verification,
        {
            "schema_version": 1,
            "status": "passed",
            "attested_preflight_sha256": hashlib.sha256(
                preflight.read_bytes()
            ).hexdigest(),
            "trusted_runtime_sources": runtime_attestation,
            "trusted_execution_sources": execution_attestation,
            "mlsysforge_root": EXPECTED_MLSYSFORGE_ROOT,
            "mlsysforge_generated_exclusions": [
                "directory-component:__pycache__",
                "file-suffix:.pyc",
            ],
            "mlsysforge_import_origins": {
                "sforge": "sforge/__init__.py",
                "sforge.cli": "sforge/cli.py",
                "sforge.harness.run_agent": "sforge/harness/run_agent.py",
                "sforge.harness.backend.docker_backend": (
                    "sforge/harness/backend/docker_backend.py"
                ),
                "sforge.harness.judge_server": "sforge/harness/judge_server.py",
                "sforge.harness.run_evaluation": (
                    "sforge/harness/run_evaluation.py"
                ),
                "sforge.harness.task_spec": "sforge/harness/task_spec.py",
            },
            "mlsysforge_harness_sources": mlsysforge_attestation,
        },
    )
    formal_archive_bytes = b"evaluated-submission"
    final_archive = tmp_path / "final_archive.tar.gz"
    final_archive.write_bytes(formal_archive_bytes)
    snapshot = tmp_path / "final_snapshot.json"
    _write_json(snapshot, _snapshot_payload(formal_archive_bytes))
    digest = lambda path: __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    preflight_sha256 = digest(preflight)
    experiment_manifest = tmp_path / "experiment-manifest.json"
    _write_json(
        experiment_manifest,
        {
            "schema_version": 2,
            "preflight_sha256": preflight_sha256,
            "hidden_baseline_identity": HIDDEN_BASELINE_IDENTITY,
        },
    )
    source_submission = tmp_path / "source/submissions/manual-1"
    source_submission.mkdir(parents=True)
    shutil.copy2(report_path, source_submission / "report.json")
    shutil.copy2(artifact_path, source_submission / "private_judge_artifacts.tar.gz")
    shutil.copy2(final_archive, source_submission / "submission.tar.gz")
    selection_root = tmp_path / "selection"
    selector_completed = subprocess.run(
        [
            "python3",
            str(SELECTOR_PATH),
            "--task-id",
            TASK_ID,
            "--source-root",
            str(tmp_path / "source"),
            "--output-dir",
            str(selection_root),
            "--validator",
            str(VALIDATOR_PATH),
            "--expected-preflight",
            str(preflight),
        ],
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "KVBENCH_HIDDEN_BASELINE_PATH": str(tmp_path / "hidden-baseline.json"),
        },
        check=False,
    )
    assert selector_completed.returncode == 0, selector_completed.stderr
    selection = selection_root / "best_selection.json"
    selection_payload = json.loads(selection.read_text())
    selected_validation_path = tmp_path / "selected-validation.json"
    selected_report = Path(selection_payload["preserved_report"])
    selected_private = Path(selection_payload["preserved_private_artifact"])
    selected_archive = Path(selection_payload["preserved_archive"])
    selected_completed = subprocess.run(
        [
            "python3",
            str(VALIDATOR_PATH),
            "--report",
            str(selected_report),
            "--private-artifact",
            str(selected_private),
            "--expected-submission-id",
            SUBMISSION_ID,
            "--eval-rc",
            "0",
            "--evaluated-archive",
            str(selected_archive),
            "--expected-archive",
            str(selected_archive),
            "--expected-preflight",
            str(preflight),
            "--output",
            str(selected_validation_path),
        ],
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "KVBENCH_HIDDEN_BASELINE_PATH": str(tmp_path / "hidden-baseline.json"),
        },
        check=False,
    )
    assert selected_completed.returncode == 0, selected_completed.stderr
    launcher = tmp_path / "launcher.sh"
    task_spec = tmp_path / "task.json"
    launcher.write_text("#!/bin/sh\n")
    task_spec.write_text("{}\n")
    completion = tmp_path / "completion.json"
    dedicated_nonce = tmp_path / "deleted-nonce.txt"
    completion_command = [
        "python3",
        str(COMPLETION_PATH),
        "--output",
        str(completion),
            "--requested-status",
            "passed",
            "--run-id",
            "formal-run",
            "--task-id",
            TASK_ID,
            "--started-at",
            "2026-08-01T00:00:00Z",
            "--finished-at",
            "2026-08-01T03:00:00Z",
            "--model",
            "gpt-example-model",
            "--reasoning-effort",
            "max",
            "--codex-cli-version",
            "0.147.0",
            "--task-timeout",
            "21600",
            "--submission-cooldown",
            "300",
            "--stage",
            "complete",
            "--exit-code",
            "0",
            "--run-rc",
            "0",
            "--final-rc",
            "0",
            "--final-attempt",
            "1",
            "--final-eval-id",
            SUBMISSION_ID,
            "--cleanup-status",
            "passed",
            "--launcher",
            str(launcher),
            "--task-spec",
            str(task_spec),
            "--preflight",
            str(preflight),
            "--experiment-manifest",
            str(experiment_manifest),
            "--trusted-source-verification",
            str(source_verification),
            "--final-archive",
            str(final_archive),
            "--final-snapshot",
            str(snapshot),
            "--final-validation",
            str(validation_path),
            "--selection",
            str(selection),
            "--selected-validation",
            str(selected_validation_path),
            "--dedicated-nonce-source",
        str(dedicated_nonce),
    ]
    completed = subprocess.run(
        completion_command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completion.read_text())
    assert result["status"] == "passed"
    assert (
        result["provenance"]["preflight"]["trusted_runtime_sources"]
        == runtime_attestation
    )
    assert (
        result["provenance"]["trusted_source_verification"][
            "trusted_execution_sources"
        ]
        == execution_attestation
    )
    assert (
        result["provenance"]["trusted_source_verification"][
            "mlsysforge_harness_sources"
        ]
        == mlsysforge_attestation
    )
    assert result["final_snapshot"]["deadline"]["elapsed_seconds"] == 21600.0
    assert result["final_snapshot"]["elapsed_seconds"] == pytest.approx(21600.01)
    assert result["final_snapshot"]["freeze"]["frozen_elapsed_seconds"] == pytest.approx(
        21600.01
    )
    assert len(result["final_snapshot"]["sha256"]) == 64
    assert result["run"]["rc"] == 0
    assert result["final_evaluation"]["rc"] == 0
    assert result["final_evaluation"]["report"]["sha256"] == digest(report_path)
    assert result["final_evaluation"]["private_artifact"]["sha256"] == digest(
        artifact_path
    )
    assert result["final_evaluation"]["evaluated_archive"]["sha256"] == digest(
        final_archive
    )
    assert result["final_evaluation"]["expected_archive"]["sha256"] == digest(
        final_archive
    )
    assert result["selected"]["round"] == "manual-1"
    assert result["selected"]["valid"] is True
    assert result["selected"]["score"] == 96.0
    assert result["hidden_baseline_identity"] == HIDDEN_BASELINE_IDENTITY
    assert stat.S_IMODE(completion.stat().st_mode) == 0o600

    # Any one of the five independently persisted identities drifting must
    # downgrade a requested success, even when every other artifact is intact.
    identity_documents = (
        (preflight, "preflight"),
        (experiment_manifest, "manifest"),
        (validation_path, "final"),
        (selected_validation_path, "selected"),
        (selection, "selection"),
    )
    for index, (path, label) in enumerate(identity_documents):
        original = path.read_bytes()
        mutated = json.loads(original)
        if label == "preflight":
            mutated["baseline_valid_steps_per_min"] += 0.5
        else:
            mutated["hidden_baseline_identity"]["valid_steps_per_min"] += 0.5
        _write_json(path, mutated)
        downgraded_output = tmp_path / f"completion-{label}-mismatch.json"
        downgraded_command = list(completion_command)
        downgraded_command[3] = str(downgraded_output)
        downgraded = subprocess.run(
            downgraded_command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert downgraded.returncode == 2, (label, downgraded.stderr)
        downgraded_payload = json.loads(downgraded_output.read_text())
        assert downgraded_payload["status"] == "failed"
        assert downgraded_payload["failure"]["strict_errors"]
        path.write_bytes(original)

    # Zero-valid run: every hidden submission was rejected, the
    # selector produced nothing, but quiescence and trusted-source verification
    # ran. That is a terminal outcome with its own status and the selector's
    # message as the reason, never "failed / still in progress".
    zero_valid_reason = (
        "no authoritatively eligible hidden submission was found (2 discovered, 2 rejected)"
    )

    def zero_valid_command(output: Path, **overrides: str) -> list[str]:
        command = list(completion_command)
        command[3] = str(output)
        command[command.index("--requested-status") + 1] = "completed_no_valid_submission"
        command[command.index("--selection") + 1] = overrides.get(
            "selection", str(tmp_path / "absent-selection.json")
        )
        command[command.index("--selected-validation") + 1] = overrides.get(
            "selected_validation", str(tmp_path / "absent-selected-validation.json")
        )
        if "trusted_source_verification" in overrides:
            command[command.index("--trusted-source-verification") + 1] = overrides[
                "trusted_source_verification"
            ]
        command += ["--reason", overrides.get("reason", zero_valid_reason)]
        return command

    zero_valid_output = tmp_path / "completion-zero-valid.json"
    zero_valid = subprocess.run(
        zero_valid_command(zero_valid_output),
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    assert zero_valid.returncode == 0, zero_valid.stderr
    zero_valid_payload = json.loads(zero_valid_output.read_text())
    assert zero_valid_payload["status"] == "completed_no_valid_submission"
    assert zero_valid_payload["failure"] is None
    assert zero_valid_payload["no_valid_submission"] == {"reason": zero_valid_reason}
    assert zero_valid_payload["selected"]["selection"]["exists"] is False
    assert zero_valid_payload["provenance"]["trusted_source_verification"]["exists"] is True

    for label, overrides in (
        ("selection_evidence_present", {"selection": str(selection)}),
        ("selected_validation_present", {"selected_validation": str(selected_validation_path)}),
        ("missing_reason", {"reason": ""}),
        ("missing_trusted_source_verification", {"trusted_source_verification": str(tmp_path / "absent-verification.json")}),
    ):
        downgraded_output = tmp_path / f"completion-zero-valid-{label}.json"
        downgraded = subprocess.run(
            zero_valid_command(downgraded_output, **overrides),
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        assert downgraded.returncode == 2, (label, downgraded.stderr)
        downgraded_payload = json.loads(downgraded_output.read_text())
        assert downgraded_payload["status"] == "failed", label
        assert downgraded_payload["no_valid_submission"] is None
        assert downgraded_payload["failure"]["strict_errors"], label


def test_launcher_records_zero_valid_selection_as_a_terminal_outcome() -> None:
    source = LAUNCHER_PATH.read_text()
    selection_section = source[source.index('stage="selection"') : source.index('stage="final_quiescence"')]
    assert '== "no authoritatively eligible hidden submission was found"*' in selection_section
    assert 'no_valid_submission_reason="${selection_message}"' in selection_section
    assert '[[ ! -e "${SELECTION_PATH}" && ! -e "${SELECTED_VALIDATION}" ]]' in selection_section
    assert 'completion_requested="completed_no_valid_submission"' in source
    assert source.index("--verify-trusted-sources") < source.index(
        'completion_requested="completed_no_valid_submission"'
    )
    assert "passed|completed_no_valid_submission)" in source
    assert 'if [[ "${requested}" != "failed" && "${completion_rc}" == "0" ]]; then' in source


def test_completion_writer_downgrades_missing_success_evidence(tmp_path: Path) -> None:
    launcher = tmp_path / "launcher.sh"
    task_spec = tmp_path / "task.json"
    launcher.write_text("#!/bin/sh\n")
    task_spec.write_text("{}\n")
    output = tmp_path / "completion.json"
    completed = subprocess.run(
        [
            "python3",
            str(COMPLETION_PATH),
            "--output",
            str(output),
            "--requested-status",
            "passed",
            "--run-id",
            "formal-run",
            "--task-id",
            TASK_ID,
            "--started-at",
            "2026-08-01T00:00:00Z",
            "--finished-at",
            "2026-08-01T00:01:00Z",
            "--model",
            "gpt-example-model",
            "--reasoning-effort",
            "max",
            "--codex-cli-version",
            "0.147.0",
            "--task-timeout",
            "21600",
            "--submission-cooldown",
            "300",
            "--stage",
            "complete",
            "--exit-code",
            "0",
            "--run-rc",
            "0",
            "--final-rc",
            "0",
            "--cleanup-status",
            "passed",
            "--launcher",
            str(launcher),
            "--task-spec",
            str(task_spec),
            "--dedicated-nonce-source",
            str(tmp_path / "absent"),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 2
    payload = json.loads(output.read_text())
    assert payload["status"] == "failed"
    assert payload["failure"]["strict_errors"]
