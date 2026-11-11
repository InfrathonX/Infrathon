"""Complete final protocol for interactive and terminal evaluation, without GPUs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from conftest import TASK_ROOT, load_module, judge_only_file

verifier = load_module("shared_final_verifier", "images/judge/verify_moe_ag.py")
verifier.RUNTIME_ROOT = TASK_ROOT / "runtime"
adapter = load_module("shared_final_adapter", "images/judge/run_judge.py")
POLICY = "full-final-shared-v3"
PROTOCOL = "flux-moe-ag-shared-final-v3"


def context_file(tmp_path, round_id, override=None):
    path = tmp_path / "context.json"
    payload = {"round_id": round_id}
    if override is not None:
        payload["candidate_version"] = override
    path.write_text(json.dumps(payload))
    path.chmod(0o600)
    return path


@pytest.mark.parametrize("round_id", ["agent-1", "auto-2", "manual-1", "candidate-v3-attempt-1"])
def test_shared_protocol_preserves_round_identity(tmp_path, round_id):
    path = context_file(tmp_path, round_id)
    assert verifier.load_evaluation_profile(path, expected_uid=os.getuid()) == (
        "final", "final-holdout-v2", round_id, None
    )


@pytest.mark.parametrize("round_id", ["agent-1", "auto-2"])
def test_full_final_does_not_grant_interactive_override(tmp_path, round_id):
    path = context_file(tmp_path, round_id, 3)
    with pytest.raises(RuntimeError):
        verifier.load_evaluation_profile(path, expected_uid=os.getuid())


@pytest.mark.parametrize("round_id", ["manual-1", "candidate-v3-attempt-1"])
def test_trusted_host_replay_keeps_override(tmp_path, round_id):
    path = context_file(tmp_path, round_id, 3)
    assert verifier.load_evaluation_profile(path, expected_uid=os.getuid()) == (
        "final", "final-holdout-v2", round_id, 3
    )


@pytest.mark.parametrize("round_id", ["agent-1", "auto-2", "manual-1", "candidate-v3-attempt-1"])
def test_real_evaluation_dispatch_passes_complete_frozen_suite(tmp_path, monkeypatch, round_id):
    context = context_file(tmp_path, round_id)
    load_profile = verifier.load_evaluation_profile
    monkeypatch.setattr(verifier, "load_evaluation_profile", lambda: load_profile(context, expected_uid=os.getuid()))
    frozen = tmp_path / "final_cases.json"
    frozen.write_bytes(judge_only_file("judge/private/final_cases.json").read_bytes())
    frozen.chmod(0o600)
    load_final = verifier.load_final_manifest
    expected_cases, digest = load_final(frozen, expected_uid=os.getuid())
    monkeypatch.setattr(verifier, "load_final_manifest", lambda: load_final(frozen, expected_uid=os.getuid()))
    monkeypatch.setattr(verifier, "verify_protected_hashes", lambda _path: [])
    if hasattr(verifier, "_trusted_hardware"):
        monkeypatch.setattr(verifier, "_trusted_hardware", lambda: {})
    source = tmp_path / "submission.py"
    source.write_text("# CPU dispatcher fixture only\n")
    monkeypatch.setattr(verifier, "locate_submission", lambda _path: tmp_path)
    monkeypatch.setattr(verifier, "validate_submission", lambda _root, _override: {
        "ok": True, "final_version": 1, "num_candidate_versions": 1,
        "selected_path": source, "selected_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    })
    captured = {}
    def capture_worker(_source, _root, cases, suite, **timing):
        captured.update(cases=cases, suite=suite, timing=timing)
        raise RuntimeError("CPU dispatcher capture; no GPU execution")
    monkeypatch.setattr(verifier, "run_worker", capture_worker)
    args = verifier.parse_args(["--app-dir", str(tmp_path)])
    _rc, detail = verifier.evaluate(args)
    assert captured["cases"] == expected_cases
    assert len(captured["cases"]) == 986
    assert captured["suite"] == "final-holdout-v2"
    assert captured["timing"]["trials"] == 7
    assert captured["timing"]["warmups"] == 1
    assert detail["manifest_sha256"] == digest
    assert detail["round_id"] == round_id
    assert detail["evaluation_policy"] == POLICY
    assert detail["selection_protocol_id"] == PROTOCOL


def records():
    reward = {key: 1.0 for key in adapter.SAFE_KEYS}
    reward.update(schema_version=2.0, case_count=967.0, edge_case_count=19.0,
                  profile_code=2.0, score_tflops=100.0, reward=100.0)
    reward.update({key: 100.0 for key in adapter.FAMILY_KEYS})
    detail = dict(reward, evaluation_profile="final", suite_id="final-holdout-v2",
                  evaluation_policy=POLICY, selection_protocol_id=PROTOCOL,
                  manifest_sha256="a" * 64)
    return reward, detail


def test_sanitized_result_records_shared_policy():
    reward, detail = records()
    result = adapter.build_structured_result(reward, detail, verifier_completed=True, verifier_returncode=0)
    assert result["valid"] is True
    assert result["metrics"]["evaluation_policy"] == POLICY
    assert result["metrics"]["selection_protocol_id"] == PROTOCOL
    assert "manifest_sha256" not in result["metrics"]


@pytest.mark.parametrize("mutation", ["missing_policy", "old_policy", "wrong_protocol", "old_anchor"])
def test_legacy_or_mismatched_protocol_cannot_be_current_result(mutation):
    reward, detail = records()
    if mutation == "missing_policy":
        detail.pop("evaluation_policy")
    elif mutation == "old_policy":
        detail["evaluation_policy"] = "anchor-v1"
    elif mutation == "wrong_protocol":
        detail["selection_protocol_id"] = "another-task-shared-final-v2"
    else:
        reward["profile_code"] = detail["profile_code"] = 1.0
        detail["evaluation_profile"] = "auto"
        detail["suite_id"] = "flux-moe-ag-anchor-v1"
    result = adapter.build_structured_result(reward, detail, verifier_completed=True, verifier_returncode=0)
    assert result["valid"] is False and result["score"] == 0.0
