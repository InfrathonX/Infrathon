"""CPU regressions for complete frozen-suite routing and private result binding."""
from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile

import pytest
import conftest

from conftest import judge_only_file

ROOT = conftest.TASK_ROOT
load_module = getattr(conftest, "load_module", None) or conftest._load
verifier = load_module("shared_final_verifier", "images/judge/verify_attn_qat.py")
adapter = load_module("shared_final_adapter", "images/judge/run_judge.py")
formal = load_module("shared_final_formal", "scripts/validate_formal_artifacts.py")


def context_file(tmp_path, round_id, override=None):
    path = tmp_path / "context.json"
    payload = {"round_id": round_id}
    if override is not None:
        payload["candidate_version"] = override
    path.write_text(json.dumps(payload))
    path.chmod(0o600)
    return path


@pytest.mark.parametrize("round_id", ["agent-1", "auto-2", "manual-1", "candidate-v3-attempt-1"])
def test_native_dispatch_loads_complete_frozen_cases_for_each_round(tmp_path, monkeypatch, round_id):
    context = context_file(tmp_path, round_id)
    router = verifier.load_evaluation_profile
    monkeypatch.setattr(verifier, "load_evaluation_profile", lambda: router(context, expected_uid=os.getuid()))
    frozen = tmp_path / "final_cases.json"
    frozen.write_bytes(judge_only_file("judge/private/final_cases.json").read_bytes())
    frozen.chmod(0o600)
    loader = verifier.load_final_manifest
    expected_cases, digest = loader(frozen, expected_uid=os.getuid())
    monkeypatch.setattr(verifier, "load_final_manifest", lambda: loader(frozen, expected_uid=os.getuid()))
    monkeypatch.setattr(verifier, "verify_protected_hashes", lambda _path: [])
    if hasattr(verifier, "_trusted_hardware"):
        monkeypatch.setattr(verifier, "_trusted_hardware", lambda: {})
    source = tmp_path / "submission.py"
    source.write_text("# Native routing fixture; GPU boundary is mocked.\n")
    monkeypatch.setattr(verifier, "locate_submission", lambda _path: tmp_path)
    monkeypatch.setattr(verifier, "validate_submission", lambda _root, _override: {
        "ok": True, "final_version": 1, "num_candidate_versions": 1,
        "selected_path": source, "selected_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    })
    captured = {}
    def worker(_source, _root, cases, suite, **timing):
        captured.update(cases=cases, suite=suite, timing=timing)
        raise RuntimeError("CPU worker capture; no GPU execution")
    monkeypatch.setattr(verifier, "run_worker", worker)
    args = verifier.parse_args(["--app-dir", str(tmp_path)])
    rc, detail = verifier.evaluate(args)
    assert rc == 1 and "CPU worker capture" in detail["private_error"]
    assert captured["cases"] == expected_cases
    assert len(captured["cases"]) == 287
    assert captured["suite"] == verifier.FINAL_SUITE_ID
    assert captured["timing"]["trials"] == 3
    assert captured["timing"]["warmups"] == 1
    assert detail["manifest_sha256"] == digest
    assert detail["round_id"] == round_id
    assert detail["evaluation_profile"] == "final" and detail["profile_code"] == 2.0
    assert detail["evaluation_policy"] == verifier.EVALUATION_POLICY
    assert detail["selection_protocol_id"] == verifier.SELECTION_PROTOCOL_ID


@pytest.mark.parametrize("round_id", ["agent-1", "auto-2"])
def test_shared_final_does_not_grant_interactive_override(tmp_path, round_id):
    with pytest.raises(RuntimeError):
        verifier.load_evaluation_profile(context_file(tmp_path, round_id, 3), expected_uid=os.getuid())


@pytest.mark.parametrize("round_id", ["manual-1", "candidate-v3-attempt-1"])
def test_host_override_keeps_original_round_identity(tmp_path, round_id):
    assert verifier.load_evaluation_profile(context_file(tmp_path, round_id, 3), expected_uid=os.getuid()) == (
        "final", verifier.FINAL_SUITE_ID, round_id, 3,
    )


@pytest.mark.parametrize("round_id", ["agent-", "auto-", "manual-", "candidate-", "preview-1"])
def test_bare_or_unknown_round_cannot_select_a_suite(tmp_path, round_id):
    with pytest.raises(RuntimeError):
        verifier.load_evaluation_profile(context_file(tmp_path, round_id), expected_uid=os.getuid())


def test_shared_suite_context_remains_private(tmp_path):
    path = context_file(tmp_path, "agent-1")
    path.chmod(0o644)
    with pytest.raises(RuntimeError):
        verifier.load_evaluation_profile(path, expected_uid=os.getuid())


def records():
    reward = {key: 1.0 for key in adapter.SAFE_KEYS}
    reward.update(schema_version=1.0, case_count=float(287), edge_case_count=float(15),
                  profile_code=2.0, score_tflops=100.0, reward=100.0)
    reward.update({key: 100.0 for key in adapter.FAMILY_KEYS})
    detail = dict(reward, **{key: 100.0 for key in formal.FAMILY_KEYS}, evaluation_profile="final", suite_id=verifier.FINAL_SUITE_ID,
                  evaluation_policy=verifier.EVALUATION_POLICY, selection_protocol_id=verifier.SELECTION_PROTOCOL_ID,
                  manifest_sha256="a" * 64)
    return reward, detail


@pytest.mark.parametrize("mutation", [None, "missing_policy", "old_policy", "wrong_protocol", "old_anchor"])
def test_adapter_rejects_old_protocol_including_same_count_anchor(mutation):
    reward, detail = records()
    if mutation == "missing_policy":
        detail.pop("evaluation_policy")
    elif mutation == "old_policy":
        detail["evaluation_policy"] = "anchor-v1"
    elif mutation == "wrong_protocol":
        detail["selection_protocol_id"] = "another-task-shared-final-v2"
    elif mutation == "old_anchor":
        reward["profile_code"] = detail["profile_code"] = 1.0
        detail["evaluation_profile"] = "auto"
        detail["suite_id"] = "attn-qat-anchor-v1"
    result = adapter.build_structured_result(reward, detail, verifier_completed=True, verifier_returncode=0)
    assert result["valid"] is (mutation is None)
    if mutation is None:
        assert result["metrics"]["evaluation_policy"] == verifier.EVALUATION_POLICY
        assert result["metrics"]["selection_protocol_id"] == verifier.SELECTION_PROTOCOL_ID
    else:
        assert result["score"] == 0.0


@pytest.mark.parametrize("mutation", [None, "old_policy", "wrong_protocol", "wrong_profile", "wrong_manifest", "short_suite"])
def test_formal_private_archive_requires_full_frozen_identity(tmp_path, mutation):
    _, detail = records()
    if mutation == "old_policy":
        detail["evaluation_policy"] = "anchor-v1"
    elif mutation == "wrong_protocol":
        detail["selection_protocol_id"] = "another-task-shared-final-v2"
    elif mutation == "wrong_profile":
        detail["evaluation_profile"] = "auto"
    elif mutation == "wrong_manifest":
        detail["manifest_sha256"] = "b" * 64
    elif mutation == "short_suite":
        detail["case_count"] -= 1
    raw = json.dumps(detail).encode()
    archive = tmp_path / formal.PRIVATE_ARTIFACT
    with tarfile.open(archive, "w:gz") as bundle:
        member = tarfile.TarInfo(formal.PRIVATE_MEMBER)
        member.size = len(raw)
        bundle.addfile(member, io.BytesIO(raw))
    archive.chmod(0o600)
    kwargs = dict(root=tmp_path, expected_score=100.0, expected_suite_id=verifier.FINAL_SUITE_ID,
                  expected_manifest_sha256="a" * 64, expected_case_count=287)
    if mutation is None:
        proof = formal._validate_private_artifact(archive, **kwargs)
        assert proof["manifest_sha256"] == "a" * 64 and proof["case_count"] == 287
    else:
        with pytest.raises(formal.ArtifactValidationError):
            formal._validate_private_artifact(archive, **kwargs)
