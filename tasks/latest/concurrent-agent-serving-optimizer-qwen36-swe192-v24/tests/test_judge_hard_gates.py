from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest


def frozen_baseline_payload(judge_module, *, include_contract: bool = True) -> dict:
    provenance = {
        **judge_module.LEGACY_MODEL_PROVENANCE,
    }
    if include_contract:
        # Evidence measured under the current (v3) revision reports the
        # recipe-bound image identity, not the qualification host's image ID.
        provenance["runtime_image_id"] = judge_module.EXPECTED_RUNTIME_IMAGE_ID
        provenance["backend_serving_contract_sha256"] = (
            judge_module.EXPECTED_BACKEND_SERVING_CONTRACT_SHA256
        )
    return {
        "schema_version": 1,
        "status": "complete",
        "suite": "hidden",
        "profile": "hidden-pressure-192",
        "workload_nonce_sha256": judge_module.EXPECTED_NONCE_COMMITMENT_SHA256,
        "legs": {
            "baseline": {
                "valid_steps_per_min": 83.73333333333333,
                "model_provenance": provenance,
            }
        },
    }


def gate(judge_module, baseline, candidate, evaluation):
    result = judge_module.canonical_hard_gate_result(
        baseline,
        candidate,
        evaluation,
    )
    assert set(result) >= {"valid", "reasons", "checks", "score"}
    assert isinstance(result["valid"], bool)
    assert isinstance(result["reasons"], list)
    assert isinstance(result["checks"], list)
    return result


def assert_reason(result: dict, fragment: str) -> None:
    assert any(fragment in str(reason) for reason in result["reasons"]), result


def check(result: dict, name: str) -> dict:
    return next(item for item in result["checks"] if item["name"] == name)


def test_slower_candidate_remains_valid_and_uses_raw_throughput_score(
    judge_module,
    valid_baseline,
    valid_candidate,
    valid_evaluation,
) -> None:
    hard_gate = gate(
        judge_module,
        valid_baseline,
        valid_candidate,
        valid_evaluation,
    )
    assert hard_gate["valid"] is True
    assert hard_gate["score"] == 80.0
    assert hard_gate["strictly_better_than_frozen_direct"] is False

    outer_report = {
        "status": "complete",
        "baseline_evidence": {
            "path": "/private/frozen-direct.json",
            "workload_nonce_sha256": "d" * 64,
        },
        "leg": valid_candidate,
        "evaluation": valid_evaluation,
        "mlsysforge_evaluation": hard_gate,
    }
    result = judge_module.structured_result(outer_report)

    assert result["valid"] is True
    assert result["score"] == 80.0
    assert result["metrics"]["primary_score_value"] == 80.0
    # Exact gate aggregates stay in the private artifact; public feedback
    # carries only the raw primary score and gate pass/fail names.
    assert all(not key.startswith("gate_") for key in result["metrics"])


@pytest.mark.parametrize("candidate_success", [0.0, 0.979999])
def test_paired_request_success_floor_is_a_hard_gate(
    judge_module,
    valid_baseline,
    valid_candidate,
    valid_evaluation,
    candidate_success,
) -> None:
    valid_candidate["request_success_rate"] = candidate_success
    result = gate(judge_module, valid_baseline, valid_candidate, valid_evaluation)

    assert result["valid"] is False
    assert_reason(result, "request_success")
    assert check(result, "paired_request_success")["metrics"]["floor"] == pytest.approx(
        0.98
    )


def test_request_success_floor_boundary_passes(
    judge_module,
    valid_baseline,
    valid_candidate,
    valid_evaluation,
) -> None:
    valid_candidate["request_success_rate"] = 0.98
    result = gate(judge_module, valid_baseline, valid_candidate, valid_evaluation)

    assert result["valid"] is True
    assert check(result, "paired_request_success")["metrics"]["floor"] == pytest.approx(
        0.98
    )


def test_paired_valid_ratio_floor_is_preserved(
    judge_module,
    valid_baseline,
    valid_candidate,
    valid_evaluation,
) -> None:
    # max(0.90, 0.99 - 0.06) == 0.93
    valid_candidate["valid_step_ratio"] = 0.929999
    failed = gate(judge_module, valid_baseline, valid_candidate, valid_evaluation)
    assert failed["valid"] is False
    assert_reason(failed, "valid_step_ratio")
    assert check(failed, "paired_valid_step_ratio")["metrics"]["floor"] == pytest.approx(
        0.93
    )

    valid_candidate["valid_step_ratio"] = 0.93
    passed = gate(judge_module, valid_baseline, valid_candidate, valid_evaluation)
    assert passed["valid"] is True


@pytest.mark.parametrize(
    ("ratio", "expected_valid"),
    [
        (0.899999, False),
        (0.90, True),
        (1.15, True),
        (1.159, True),  # the frozen Oracle on H100 SXM (2026-10-16)
        (2.0, True),
    ],
)
def test_backend_work_ratio_bounds_are_inclusive_and_fail_closed(
    judge_module,
    valid_baseline,
    valid_candidate,
    valid_evaluation,
    ratio,
    expected_valid,
) -> None:
    valid_candidate["backend_tokens_per_valid_step"] = 1000.0 * ratio
    result = gate(judge_module, valid_baseline, valid_candidate, valid_evaluation)

    assert result["valid"] is expected_valid
    assert check(result, "paired_backend_work")["metrics"]["ratio"] == pytest.approx(
        ratio
    )
    if not expected_valid:
        assert_reason(result, "paired_backend_work")


def test_p99_gateway_wait_ceiling_is_a_fraction_of_the_scored_window(
    judge_module,
    valid_baseline,
    valid_candidate,
    valid_evaluation,
) -> None:
    # 0.4 x the leg's own 1800 s scored window reproduces the historical
    # 720 s ceiling without freezing a wall-clock number measured on one
    # backend.
    assert judge_module.P99_GATEWAY_WAIT_WINDOW_FRACTION == 0.4

    valid_candidate["backend_audit"]["p99_gateway_wait_sec"] = 720.000001
    failed = gate(judge_module, valid_baseline, valid_candidate, valid_evaluation)

    assert failed["valid"] is False
    assert_reason(failed, "p99_gateway_wait")

    valid_candidate["backend_audit"]["p99_gateway_wait_sec"] = 720.0
    passed = gate(judge_module, valid_baseline, valid_candidate, valid_evaluation)
    assert passed["valid"] is True
    metrics = check(passed, "candidate_p99_gateway_wait")["metrics"]
    assert metrics["measurement_window_sec"] == pytest.approx(1800.0)
    assert metrics["maximum_sec"] == pytest.approx(720.0)


def test_p99_gateway_wait_ceiling_scales_with_a_shorter_window(
    judge_module,
    valid_baseline,
    valid_candidate,
    valid_evaluation,
) -> None:
    # The same rule on the 600 s public-shaped window is 240 s.  A wall-clock
    # constant would instead have admitted a candidate that spent its entire
    # window queued.
    valid_candidate["profile_config"]["measurement_window_sec"] = 600
    valid_candidate["measurement_window"]["duration_sec"] = 600
    valid_candidate["backend_audit"]["p99_gateway_wait_sec"] = 240.000001
    failed = gate(judge_module, valid_baseline, valid_candidate, valid_evaluation)

    assert failed["valid"] is False
    assert_reason(failed, "p99_gateway_wait")
    assert check(failed, "candidate_p99_gateway_wait")["metrics"][
        "maximum_sec"
    ] == pytest.approx(240.0)

    valid_candidate["backend_audit"]["p99_gateway_wait_sec"] = 240.0
    passed = gate(judge_module, valid_baseline, valid_candidate, valid_evaluation)
    assert passed["valid"] is True


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_profile_config",
        "missing_window_sec",
        "non_positive_window",
        "window_disagrees_with_report",
    ],
)
def test_p99_gateway_wait_fails_closed_without_a_declared_window(
    judge_module,
    valid_baseline,
    valid_candidate,
    valid_evaluation,
    mutation,
) -> None:
    if mutation == "missing_profile_config":
        valid_candidate.pop("profile_config")
    elif mutation == "missing_window_sec":
        valid_candidate["profile_config"].pop("measurement_window_sec")
    elif mutation == "non_positive_window":
        valid_candidate["profile_config"]["measurement_window_sec"] = 0
    else:
        valid_candidate["measurement_window"]["duration_sec"] = 900

    valid_candidate["backend_audit"]["p99_gateway_wait_sec"] = 1.0
    result = gate(judge_module, valid_baseline, valid_candidate, valid_evaluation)

    assert result["valid"] is False
    assert_reason(result, "p99_gateway_wait")
    assert check(result, "candidate_p99_gateway_wait")["metrics"]["maximum_sec"] is None


def test_trusted_fixed_window_right_censoring_is_accepted(
    judge_module,
    valid_baseline,
    valid_candidate,
    valid_evaluation,
) -> None:
    passed = gate(judge_module, valid_baseline, valid_candidate, valid_evaluation)
    assert passed["valid"] is True
    assert check(passed, "backend_one_to_one_audit")["metrics"][
        "candidate_right_censor_reconciled"
    ] is True

    broken = copy.deepcopy(valid_candidate)
    broken["backend_audit"]["missing_backend_requests"] = 0
    failed = gate(judge_module, valid_baseline, broken, valid_evaluation)
    assert failed["valid"] is False
    assert_reason(failed, "backend_one_to_one_audit")
    assert check(failed, "backend_one_to_one_audit")["metrics"][
        "candidate_right_censor_reconciled"
    ] is False


def test_runner_identity_or_protocol_failure_cannot_be_scored(
    judge_module,
    valid_baseline,
    valid_candidate,
    valid_evaluation,
) -> None:
    valid_evaluation["valid"] = False
    valid_evaluation["all_invalid_reasons"] = ["model_provenance_mismatch"]
    result = gate(judge_module, valid_baseline, valid_candidate, valid_evaluation)

    assert result["valid"] is False
    assert_reason(result, "protocol_identity_fixed_window_lifecycle")


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_gate",
        "duplicate_gate",
        "unknown_gate",
        "validity_conflict",
        "negative_score",
        "nan_score",
    ],
)
def test_structured_result_fails_closed_on_malformed_gate_contract(
    judge_module,
    valid_baseline,
    valid_candidate,
    valid_evaluation,
    mutation,
) -> None:
    edge = judge_module.canonical_hard_gate_result(
        valid_baseline, valid_candidate, valid_evaluation
    )
    if mutation == "missing_gate":
        edge["checks"].pop()
    elif mutation == "duplicate_gate":
        edge["checks"][-1]["name"] = edge["checks"][0]["name"]
    elif mutation == "unknown_gate":
        edge["checks"][-1]["name"] = "unknown_hidden_gate"
    elif mutation == "validity_conflict":
        edge["valid"] = False
    elif mutation == "negative_score":
        edge["score"] = -1.0
    elif mutation == "nan_score":
        edge["score"] = float("nan")

    result = judge_module.structured_result({"mlsysforge_evaluation": edge})

    assert result["valid"] is False
    assert result["score"] == 0.0
    assert result["errors"] == 1
    assert result["metrics"]["evaluation_status"] == "infrastructure_error"


def test_complete_runner_evidence_requires_successful_quiescent_cleanup(
    judge_module,
) -> None:
    candidate_sha = "a" * 64
    report = {
        "schema_version": 1,
        "kind": judge_module.EXPECTED_RUNNER_KIND,
        "status": "complete",
        "phase": "quiescent",
        "direct_leg_executed": False,
        "candidate_tree_sha256": candidate_sha,
        "preflight_backend_serving_contract_sha256": (
            judge_module.EXPECTED_BACKEND_SERVING_CONTRACT_SHA256
        ),
        "errors": [],
        "completed_at_utc": "2026-08-01T00:00:00Z",
        "cleanup": {
            "attempted": True,
            "stack_stop_succeeded": True,
            "quiescent": True,
            "residue": {
                "compose_container_ids": [],
                "sandbox_container_ids": [],
            },
        },
        "baseline_evidence": {
            "path": "/run/kvbench-baselines/hidden.json",
            "file_sha256": "b" * 64,
            "workload_nonce_sha256": (
                judge_module.EXPECTED_NONCE_COMMITMENT_SHA256
            ),
            "direct_valid_steps_per_min": 83.73333333333333,
            "backend_serving_contract_sha256": (
                judge_module.EXPECTED_BACKEND_SERVING_CONTRACT_SHA256
            ),
        },
        "leg": {"valid_steps_per_min": 90.0},
        "evaluation": {"candidate_valid_steps_per_min": 90.0},
    }

    baseline_bundle = {
        "path": "/run/kvbench-baselines/hidden.json",
        "hidden_baseline_identity": {
            "file_sha256": "b" * 64,
            "valid_steps_per_min": 83.73333333333333,
            "backend_serving_contract_sha256": (
                judge_module.EXPECTED_BACKEND_SERVING_CONTRACT_SHA256
            ),
            "workload_nonce_sha256": (
                judge_module.EXPECTED_NONCE_COMMITMENT_SHA256
            ),
        },
    }
    candidate, evaluation = judge_module._validate_complete_runner_report(
        report, candidate_sha, baseline_bundle
    )
    assert candidate["valid_steps_per_min"] == 90.0
    assert evaluation["candidate_valid_steps_per_min"] == 90.0

    report["cleanup"]["quiescent"] = False
    with pytest.raises(RuntimeError):
        judge_module._validate_complete_runner_report(
            report, candidate_sha, baseline_bundle
        )

    report["cleanup"]["quiescent"] = True
    mutations = {
        "path": "/different/hidden.json",
        "file_sha256": "c" * 64,
        "workload_nonce_sha256": "d" * 64,
        "direct_valid_steps_per_min": 83.7333333333334,
        "backend_serving_contract_sha256": "e" * 64,
    }
    for field, value in mutations.items():
        broken = copy.deepcopy(report)
        broken["baseline_evidence"][field] = value
        with pytest.raises(RuntimeError):
            judge_module._validate_complete_runner_report(
                broken, candidate_sha, baseline_bundle
            )

    broken = copy.deepcopy(report)
    broken["baseline_evidence"]["unexpected"] = True
    with pytest.raises(RuntimeError):
        judge_module._validate_complete_runner_report(
            broken, candidate_sha, baseline_bundle
        )


def test_frozen_baseline_load_hashes_and_parses_one_byte_snapshot(
    judge_module, monkeypatch, tmp_path: Path
) -> None:
    baseline_path = tmp_path / "hidden.json"
    raw = (
        json.dumps(frozen_baseline_payload(judge_module), indent=2, sort_keys=True)
        + "\n"
    ).encode()
    baseline_path.write_bytes(raw)
    monkeypatch.setattr(judge_module, "BASELINE_PATH", baseline_path)
    monkeypatch.setattr(
        judge_module,
        "file_sha256",
        lambda _path: pytest.fail("loader must hash its already-read bytes"),
    )
    monkeypatch.setattr(
        judge_module,
        "_strict_json_object",
        lambda _path: pytest.fail("loader must parse its already-read bytes"),
    )

    bundle = judge_module._load_frozen_baseline_bundle()

    assert bundle["path"] == str(baseline_path)
    assert bundle["hidden_baseline_identity"] == {
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "valid_steps_per_min": 83.73333333333333,
        "backend_serving_contract_sha256": (
            judge_module.EXPECTED_BACKEND_SERVING_CONTRACT_SHA256
        ),
        "workload_nonce_sha256": judge_module.EXPECTED_NONCE_COMMITMENT_SHA256,
    }


def test_frozen_baseline_requires_actual_or_exact_legacy_contract(
    judge_module, monkeypatch, tmp_path: Path
) -> None:
    baseline_path = tmp_path / "hidden.json"
    monkeypatch.setattr(judge_module, "BASELINE_PATH", baseline_path)

    wrong_contract = frozen_baseline_payload(judge_module)
    wrong_contract["legs"]["baseline"]["model_provenance"][
        "backend_serving_contract_sha256"
    ] = "0" * 64
    baseline_path.write_text(json.dumps(wrong_contract), encoding="utf-8")
    with pytest.raises(RuntimeError, match="backend serving contract"):
        judge_module._load_frozen_baseline_bundle()

    # Evidence recorded before the digest existed can only be the v1 boot, so
    # it is attributed to the retained v1 revision rather than restamped with
    # whichever revision this checkout ships.
    legacy = frozen_baseline_payload(judge_module, include_contract=False)
    baseline_path.write_text(json.dumps(legacy), encoding="utf-8")
    assert (
        judge_module._load_frozen_baseline_bundle()["hidden_baseline_identity"][
            "backend_serving_contract_sha256"
        ]
        == judge_module.LEGACY_BACKEND_SERVING_CONTRACT_SHA256
    )

    # An anchor measured under v1 still loads and keeps its own revision; it
    # carries the qualification host's image ID, which is what v1 bound.
    v1 = frozen_baseline_payload(judge_module)
    v1["legs"]["baseline"]["model_provenance"][
        "backend_serving_contract_sha256"
    ] = judge_module.LEGACY_BACKEND_SERVING_CONTRACT_SHA256
    v1["legs"]["baseline"]["model_provenance"][
        "runtime_image_id"
    ] = judge_module.LEGACY_RUNTIME_IMAGE_ID
    baseline_path.write_text(json.dumps(v1), encoding="utf-8")
    assert (
        judge_module._load_frozen_baseline_bundle()["hidden_baseline_identity"][
            "backend_serving_contract_sha256"
        ]
        == judge_module.LEGACY_BACKEND_SERVING_CONTRACT_SHA256
    )

    # v3 evidence that reports the old host image ID, or v1 evidence that
    # reports the recipe digest, mixes the two bindings and is refused.
    mixed = frozen_baseline_payload(judge_module)
    mixed["legs"]["baseline"]["model_provenance"][
        "runtime_image_id"
    ] = judge_module.LEGACY_RUNTIME_IMAGE_ID
    baseline_path.write_text(json.dumps(mixed), encoding="utf-8")
    with pytest.raises(RuntimeError, match="runtime image identity"):
        judge_module._load_frozen_baseline_bundle()

    legacy["legs"]["baseline"]["model_provenance"]["profile_id"] = (
        "unknown-legacy-profile"
    )
    baseline_path.write_text(json.dumps(legacy), encoding="utf-8")
    with pytest.raises(RuntimeError, match="legacy serving provenance"):
        judge_module._load_frozen_baseline_bundle()


def test_private_envelope_commits_loaded_bundle_without_rereading_path(
    judge_module, monkeypatch, tmp_path: Path
) -> None:
    identity = {
        "file_sha256": "b" * 64,
        "valid_steps_per_min": 83.73333333333333,
        "backend_serving_contract_sha256": (
            judge_module.EXPECTED_BACKEND_SERVING_CONTRACT_SHA256
        ),
        "workload_nonce_sha256": judge_module.EXPECTED_NONCE_COMMITMENT_SHA256,
    }
    bundle = {
        "path": str(tmp_path / "hidden.json"),
        "hidden_baseline_identity": identity,
        "leg": {},
    }
    monkeypatch.setattr(
        judge_module,
        "file_sha256",
        lambda _path: pytest.fail("envelope must not reread the baseline path"),
    )
    output = tmp_path / "private.json"

    envelope = judge_module._write_private_envelope(
        output,
        started_at_utc="2026-08-01T00:00:00Z",
        classification="infrastructure_error",
        candidate_tree_sha256=None,
        runner={},
        runner_evidence=None,
        public_result=judge_module.infrastructure_failure("test"),
        private_errors=[],
        baseline_bundle=bundle,
    )

    assert envelope["commitments"] == {
        "hidden_baseline_identity": identity,
    }
    assert json.loads(output.read_text(encoding="utf-8"))["commitments"] == {
        "hidden_baseline_identity": identity,
    }


def test_end_guard_detects_baseline_byte_drift_without_changing_commitment(
    judge_module, tmp_path: Path
) -> None:
    baseline_path = tmp_path / "hidden.json"
    baseline_path.write_bytes(b"original frozen bytes")
    committed_sha = hashlib.sha256(b"original frozen bytes").hexdigest()
    identity = {
        "file_sha256": committed_sha,
        "valid_steps_per_min": 83.73333333333333,
        "backend_serving_contract_sha256": (
            judge_module.EXPECTED_BACKEND_SERVING_CONTRACT_SHA256
        ),
        "workload_nonce_sha256": judge_module.EXPECTED_NONCE_COMMITMENT_SHA256,
    }
    bundle = {
        "path": str(baseline_path),
        "hidden_baseline_identity": identity,
        "leg": {},
    }
    judge_module._assert_frozen_baseline_snapshot_current(bundle)

    baseline_path.write_bytes(b"replacement frozen bytes")
    with pytest.raises(RuntimeError, match="bytes drifted"):
        judge_module._assert_frozen_baseline_snapshot_current(bundle)
    assert bundle["hidden_baseline_identity"]["file_sha256"] == committed_sha
