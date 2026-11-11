"""Every admitted round runs the complete final suite and the private edge suite, without GPUs."""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tarfile
import types
from pathlib import Path
from typing import Any

import pytest
from conftest import load_module
from suite_fixtures import build_edge_manifest, build_final_manifest

verifier = load_module("shared_final_verifier", "images/judge/verify_sparse_mla.py")
adapter = load_module("shared_final_adapter", "images/judge/run_judge.py")
worker = load_module("shared_final_worker", "images/judge/candidate_worker.py")
POLICY = "full-final-shared-v3"
PROTOCOL = "sparse-mla-shared-final-v3"


def context_file(tmp_path, round_id, override=None):
    path = tmp_path / "context.json"
    payload = {"round_id": round_id}
    if override is not None:
        payload["candidate_version"] = override
    path.write_text(json.dumps(payload))
    path.chmod(0o600)
    return path


def private_file(tmp_path: Path, name: str, payload: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    path.chmod(0o600)
    return path


def test_identities_are_the_v3_shared_final_protocol():
    assert verifier.FINAL_SUITE_ID == "final-holdout-v4"
    assert verifier.EDGE_SUITE_ID == "edge-holdout-v3"
    assert verifier.FINAL_SCHEMA_VERSION == 5 and verifier.SCHEMA_VERSION == 3
    assert verifier.EVALUATION_POLICY == adapter.EVALUATION_POLICY == POLICY
    assert verifier.SELECTION_PROTOCOL_ID == adapter.SELECTION_PROTOCOL_ID == PROTOCOL
    assert verifier.RESULT_NAME == adapter.DETAIL_NAME == "standalone_sparse_mla_prefill_v2_results.json"
    assert worker.FINAL_SUITE_ID == verifier.FINAL_SUITE_ID
    assert worker.EDGE_SUITE_ID == verifier.EDGE_SUITE_ID
    assert worker.SCORED_KEYS == set(verifier.SCORED_KEYS)
    assert worker.SCORED_GROUP_KEYS == {group: set(keys) for group, keys in verifier.SCORED_GROUP_KEYS.items()}
    assert worker.EXPECTED_CASE_COUNT == verifier.FINAL_CASE_COUNT
    assert adapter.PROFILE_IDENTITIES == {2.0: ("final", verifier.FINAL_SUITE_ID)}


@pytest.mark.parametrize("round_id", ["agent-1", "auto-2", "manual-1", "candidate-v3-attempt-1"])
def test_shared_protocol_preserves_round_identity(tmp_path, round_id):
    path = context_file(tmp_path, round_id)
    assert verifier.load_evaluation_profile(path, expected_uid=os.getuid()) == (
        "final", "final-holdout-v4", round_id, None
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
        "final", "final-holdout-v4", round_id, 3
    )


@pytest.mark.parametrize("round_id", ["auto-", "agent-", "manual-", "candidate-", "preview-1"])
def test_unknown_or_bare_rounds_fail_closed(tmp_path, round_id):
    with pytest.raises(RuntimeError):
        verifier.load_evaluation_profile(context_file(tmp_path, round_id), expected_uid=os.getuid())


@pytest.mark.parametrize("override", [0, 1000, True, "3"])
def test_invalid_host_override_fails_closed(tmp_path, override):
    with pytest.raises(RuntimeError):
        verifier.load_evaluation_profile(context_file(tmp_path, "manual-1", override), expected_uid=os.getuid())


def test_context_permissions_still_fail_closed(tmp_path):
    context = context_file(tmp_path, "agent-1")
    context.chmod(0o644)
    with pytest.raises(RuntimeError):
        verifier.load_evaluation_profile(context, expected_uid=os.getuid())


@pytest.mark.parametrize("loader, builder", [
    ("load_final_manifest", build_final_manifest),
    ("load_edge_manifest", build_edge_manifest),
])
@pytest.mark.parametrize("damage", [None, "group_readable", "symlink", "hard_link", "other_owner", "empty"])
def test_private_manifests_are_root_only_regular_files(tmp_path, loader, builder, damage):
    path = private_file(tmp_path, "manifest.json", builder())
    load = getattr(verifier, loader)
    uid = os.getuid()
    if damage == "group_readable":
        path.chmod(0o640)
    elif damage == "symlink":
        link = tmp_path / "link.json"
        link.symlink_to(path)
        path = link
    elif damage == "hard_link":
        os.link(path, tmp_path / "second.json")
    elif damage == "other_owner":
        uid += 1
    elif damage == "empty":
        path.write_text("")
    if damage is None:
        cases, digest = load(path, expected_uid=uid)
        assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
        assert len(cases) in (verifier.FINAL_CASE_COUNT, verifier.EDGE_CASE_COUNT)
    else:
        with pytest.raises(RuntimeError):
            load(path, expected_uid=uid)


@pytest.mark.parametrize("loader, builder, cap", [
    ("load_final_manifest", build_final_manifest, 1024 * 1024),
    ("load_edge_manifest", build_edge_manifest, 2 * 1024 * 1024),
])
def test_private_manifest_size_caps_fit_the_frozen_design(tmp_path, loader, builder, cap):
    """The cap must hold a 997-case manifest with its stream call lists (the frozen one
    is about 600 KB) and still bound what the Judge reads."""
    load = getattr(verifier, loader)
    encoded = json.dumps(builder()).encode()
    fits = tmp_path / "fits.json"
    fits.write_bytes(encoded + b" " * (cap - len(encoded)))
    fits.chmod(0o600)
    cases, _digest = load(fits, expected_uid=os.getuid())
    assert len(cases) in (verifier.FINAL_CASE_COUNT, verifier.EDGE_CASE_COUNT)
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(encoded + b" " * (cap + 1 - len(encoded)))
    oversized.chmod(0o600)
    with pytest.raises(RuntimeError, match="root-only file contract"):
        load(oversized, expected_uid=os.getuid())


def _dispatch(tmp_path, monkeypatch, round_id, *, args=()):
    """Run evaluate() through the real root-only loaders up to the native worker."""
    context = context_file(tmp_path, round_id)
    load_profile = verifier.load_evaluation_profile
    monkeypatch.setattr(verifier, "load_evaluation_profile", lambda: load_profile(context, expected_uid=os.getuid()))
    final = private_file(tmp_path, "final_cases.json", build_final_manifest())
    edges = private_file(tmp_path, "edge_cases.json", build_edge_manifest())
    load_final, load_edges = verifier.load_final_manifest, verifier.load_edge_manifest
    monkeypatch.setattr(verifier, "load_final_manifest", lambda: load_final(final, expected_uid=os.getuid()))
    monkeypatch.setattr(verifier, "load_edge_manifest", lambda: load_edges(edges, expected_uid=os.getuid()))
    monkeypatch.setattr(verifier, "verify_protected_hashes", lambda _path: [])
    monkeypatch.setattr(verifier, "_trusted_hardware", lambda: {})
    source = tmp_path / "submission.py"
    source.write_text("# CPU dispatcher fixture only\n")
    monkeypatch.setattr(verifier, "locate_submission", lambda _path: tmp_path)
    monkeypatch.setattr(verifier, "validate_submission", lambda _root, _override: {
        "ok": True, "final_version": 1, "num_candidate_versions": 1,
        "selected_path": source, "selected_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    })
    run_root = tmp_path / "run"
    run_root.mkdir()
    monkeypatch.setattr(verifier.tempfile, "mkdtemp", lambda **kwargs: str(run_root))
    order: list[str] = []
    captured: dict[str, Any] = {}

    def sanitizer(_source, _root, cases, **kwargs):
        order.append("sanitizer")
        captured["sanitize"] = tuple(cases)
        return {"status": "passed"}

    def capture_worker(_source, _root, cases, edge_cases, **timing):
        order.append("worker")
        captured.update(cases=tuple(cases), edges=tuple(edge_cases), timing=timing)
        raise RuntimeError("CPU dispatcher capture; no GPU execution")

    monkeypatch.setattr(verifier, "run_sanitizer", sanitizer)
    monkeypatch.setattr(verifier, "run_worker", capture_worker)
    parsed = verifier.parse_args(["--app-dir", str(tmp_path), *args])
    rc, detail = verifier.evaluate(parsed)
    expected = (load_final(final, expected_uid=os.getuid()), load_edges(edges, expected_uid=os.getuid()))
    return rc, detail, order, captured, expected


@pytest.mark.parametrize("round_id", ["agent-1", "auto-2", "manual-1", "candidate-v3-attempt-1"])
def test_real_evaluation_dispatch_passes_complete_frozen_suites(tmp_path, monkeypatch, round_id):
    rc, detail, order, captured, expected = _dispatch(tmp_path, monkeypatch, round_id)
    (final_cases, final_digest), (edge_cases, edge_digest) = expected
    assert rc == 1 and "CPU dispatcher capture" in detail["private_error"]
    assert order == ["sanitizer", "worker"]
    assert captured["cases"] == final_cases and len(final_cases) == verifier.FINAL_CASE_COUNT == 997
    assert all(
        set(case) == verifier.SCORED_KEYS | verifier.SCORED_GROUP_KEYS[case["group"]]
        for case in captured["cases"]
    )
    assert captured["edges"] == edge_cases and len(edge_cases) == verifier.EDGE_CASE_COUNT
    assert captured["sanitize"] == tuple(case for case in edge_cases if case["sanitize"])
    assert captured["timing"]["trials"] == 3
    assert captured["timing"]["warmups"] == 1
    assert detail["manifest_sha256"] == final_digest
    assert detail["edge_manifest_sha256"] == edge_digest
    assert detail["case_count"] == 997.0
    assert detail["edge_cases_total"] == float(verifier.EDGE_CASE_COUNT)
    assert detail["round_id"] == round_id
    assert detail["evaluation_profile"] == "final" and detail["profile_code"] == 2.0
    assert detail["suite_id"] == "final-holdout-v4"
    assert detail["evaluation_policy"] == POLICY
    assert detail["selection_protocol_id"] == PROTOCOL
    assert detail["suite_ok"] == 1.0 and detail["sanitizer_ran"] == 1.0
    assert detail["sanitizer_inconclusive"] == 0.0
    assert captured["timing"]["promoted"] == []
    assert detail["failed_gate"] == "correctness_ok"


@pytest.mark.parametrize("flag", [("--trials", "7"), ("--trials", "5"), ("--warmups", "2")])
def test_judge_runs_exactly_three_trials_and_one_warmup(tmp_path, monkeypatch, flag):
    rc, detail, order, _captured, _expected = _dispatch(tmp_path, monkeypatch, "agent-1", args=flag)
    assert rc == 1 and order == []
    assert "exactly 3 trials and 1 warmup" in detail["private_error"]
    assert detail["score_tflops"] == 0.0


def test_overlapping_scored_and_edge_ids_fail_the_suite_gate(tmp_path, monkeypatch):
    edges = build_edge_manifest()
    edges["cases"][0]["case_id"] = build_final_manifest()["cases"][0]["case_id"]
    path = private_file(tmp_path, "edge.json", edges)
    load_edges = verifier.load_edge_manifest
    monkeypatch.setattr(verifier, "load_edge_manifest", lambda: load_edges(path, expected_uid=os.getuid()))
    final = private_file(tmp_path, "final.json", build_final_manifest())
    load_final = verifier.load_final_manifest
    monkeypatch.setattr(verifier, "load_final_manifest", lambda: load_final(final, expected_uid=os.getuid()))
    context = context_file(tmp_path, "agent-1")
    load_profile = verifier.load_evaluation_profile
    monkeypatch.setattr(verifier, "load_evaluation_profile", lambda: load_profile(context, expected_uid=os.getuid()))
    monkeypatch.setattr(verifier, "verify_protected_hashes", lambda _path: [])
    rc, detail = verifier.evaluate(verifier.parse_args(["--app-dir", str(tmp_path)]))
    assert rc == 1 and detail["failed_gate"] == "suite_ok" and "overlap" in detail["private_error"]
    assert detail["profile_ok"] == 1.0 and detail["suite_ok"] == 0.0


def test_run_worker_sends_both_suites_to_the_native_worker(tmp_path, monkeypatch):
    cases = verifier.validate_final_manifest(build_final_manifest())
    edges = verifier.validate_edge_manifest(build_edge_manifest())
    account = types.SimpleNamespace(pw_uid=os.getuid(), pw_gid=os.getgid())
    result_dir = tmp_path / "worker_result"
    result_dir.mkdir()
    monkeypatch.setattr(verifier, "_prepare_run_dirs", lambda root, name: (account, result_dir, tmp_path))
    seen = {}

    def fake_run(command, *, stdin_payload, run_root, log_name, env, timeout):
        seen.update(command=command, payload=json.loads(stdin_payload), env=env, log_name=log_name)
        path = result_dir / "result.json"
        path.write_text(json.dumps({"status": "passed"}))
        path.chmod(0o600)
        return 0

    monkeypatch.setattr(verifier, "_run_unprivileged", fake_run)
    result = verifier.run_worker(tmp_path / "c.py", tmp_path, cases, edges, trials=3, warmups=1, timeout=60)
    assert result == {"status": "passed"}
    command = seen["command"]
    assert command[command.index("--mode") + 1] == "suite"
    assert command[command.index("--trials") + 1] == "3"
    assert command[command.index("--warmups") + 1] == "1"
    assert seen["payload"] == {
        "schema_version": 3, "mode": "suite", "suite_id": "final-holdout-v4", "cases": list(cases),
        "edge_suite_id": "edge-holdout-v3", "edge_cases": list(edges), "promoted_sanitize_only": [],
    }
    assert "PYTORCH_NO_CUDA_MEMORY_CACHING" not in seen["env"]
    assert seen["log_name"] == "candidate_worker.log"


def test_candidate_environment_is_an_allowlist(monkeypatch, tmp_path):
    monkeypatch.setenv("LD_PRELOAD", "/tmp/evil.so")
    monkeypatch.setenv("PYTHONPATH", "/app/workdir")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/vgpu")
    env = verifier._candidate_environment(tmp_path)
    assert "LD_PRELOAD" not in env and "PYTHONPATH" not in env
    # GPU virtualization (PyroMind HAMi) needs the visible-device and loader settings.
    assert env["CUDA_VISIBLE_DEVICES"] == "0" and env["LD_LIBRARY_PATH"] == "/usr/local/vgpu"
    assert env["HOME"] == "/home/agent" and env["TRITON_CACHE_DIR"] == str(tmp_path / "triton")


# --------------------------------------------------------------------------
# candidate_worker.py --mode suite.
# --------------------------------------------------------------------------
def _suite_manifest() -> dict[str, Any]:
    return {
        "schema_version": 3, "mode": "suite", "suite_id": "final-holdout-v4",
        "cases": list(verifier.validate_final_manifest(build_final_manifest())),
        "edge_suite_id": "edge-holdout-v3",
        "edge_cases": list(verifier.validate_edge_manifest(build_edge_manifest())),
    }


def _run_worker_main(monkeypatch, tmp_path, manifest, argv):
    calls = []
    fake_runtime = types.SimpleNamespace(
        run_suite=lambda candidate, cases, edges, **kwargs: calls.append((candidate, cases, edges, kwargs))
        or {"status": "passed"},
    )
    monkeypatch.setattr(worker, "_load_trusted_runtime", lambda: fake_runtime, raising=False)
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(json.dumps(manifest).encode())))
    output = tmp_path / "result.json"
    rc = worker.main([*argv, "--candidate", str(tmp_path / "c.py"), "--output", str(output)])
    return rc, calls, output


def test_worker_suite_mode_runs_edges_then_the_grid(monkeypatch, tmp_path):
    manifest = _suite_manifest()
    rc, calls, output = _run_worker_main(monkeypatch, tmp_path, manifest, ["--trials", "3", "--warmups", "1"])
    assert rc == 0 and len(calls) == 1
    _candidate, cases, edges, kwargs = calls[0]
    assert cases == manifest["cases"] and edges == manifest["edge_cases"]
    assert kwargs == {
        "trials": 3, "warmups": 1, "require_hardware": True, "promoted_sanitize_only": [],
        "run_scored_after_edge_failure": False,
    }
    assert json.loads(output.read_text()) == {"status": "passed"}


def test_worker_suite_mode_forwards_promoted_sanitizer_only_cases(monkeypatch, tmp_path):
    manifest = _suite_manifest()
    large = next(case["case_id"] for case in manifest["edge_cases"] if not case["native"])
    manifest["promoted_sanitize_only"] = [large]
    rc, calls, _output = _run_worker_main(monkeypatch, tmp_path, manifest, ["--trials", "3", "--warmups", "1"])
    assert rc == 0 and calls[0][3]["promoted_sanitize_only"] == [large]
    manifest["promoted_sanitize_only"] = "all"
    with pytest.raises(ValueError):
        _run_worker_main(monkeypatch, tmp_path, manifest, ["--trials", "3", "--warmups", "1"])


@pytest.mark.parametrize("damage", ["v1_suite", "short", "v1_case_fields", "edge_suite", "no_trials"])
def test_worker_suite_mode_refuses_mismatched_manifests(monkeypatch, tmp_path, damage):
    manifest = _suite_manifest()
    argv = ["--trials", "3", "--warmups", "1"]
    if damage == "v1_suite":
        manifest["suite_id"] = "final-holdout-v1"
    elif damage == "short":
        manifest["cases"].pop()
    elif damage == "v1_case_fields":
        manifest["cases"][0]["b"] = 256
    elif damage == "edge_suite":
        manifest["edge_suite_id"] = "edge-public-v3"
    else:
        argv = []
    with pytest.raises(ValueError):
        _run_worker_main(monkeypatch, tmp_path, manifest, argv)


# --------------------------------------------------------------------------
# Adapter and formal artifacts bind the current identity.
# --------------------------------------------------------------------------
def records():
    detail = verifier.failure_detail("", profile="final", suite_id=verifier.FINAL_SUITE_ID)
    detail.update(final_version=1.0, num_candidate_versions=1.0, manifest_sha256="a" * 64,
                  edge_manifest_sha256="e" * 64, case_count=997.0,
                  edge_cases_total=float(verifier.EDGE_CASE_COUNT), edge_cases_failed=0.0,
                  failed_gate_code=0.0, sanitizer_ran=1.0, score_tflops=100.0, reward=100.0)
    for gate, _label in adapter.GATES:
        detail[gate] = 1.0
    for key in adapter.FAMILY_KEYS:
        detail[key] = 100.0
    reward = {key: value for key, value in detail.items() if isinstance(value, (int, float))}
    return reward, detail


def test_sanitized_result_records_shared_policy():
    reward, detail = records()
    result = adapter.build_structured_result(reward, detail, verifier_completed=True, verifier_returncode=0)
    assert result["valid"] is True
    assert result["metrics"]["evaluation_policy"] == POLICY
    assert result["metrics"]["selection_protocol_id"] == PROTOCOL
    assert "manifest_sha256" not in result["metrics"]
    assert "edge_manifest_sha256" not in result["metrics"]


@pytest.mark.parametrize(
    "mutation",
    ["missing_policy", "v2_policy", "v2_protocol", "old_anchor", "v1_suite", "missing_edge_digest"],
)
def test_legacy_or_mismatched_protocol_cannot_be_current_result(mutation):
    reward, detail = records()
    if mutation == "missing_policy":
        detail.pop("evaluation_policy")
    elif mutation == "v2_policy":
        detail["evaluation_policy"] = "full-final-shared-v2"
    elif mutation == "v2_protocol":
        detail["selection_protocol_id"] = "sparse-mla-shared-final-v2"
    elif mutation == "old_anchor":
        reward["profile_code"] = detail["profile_code"] = 1.0
        detail["evaluation_profile"] = "auto"
        detail["suite_id"] = "flashmla-sparse-prefill-anchor-v1"
    elif mutation == "v1_suite":
        detail["suite_id"] = "final-holdout-v1"
    else:
        detail.pop("edge_manifest_sha256")
    result = adapter.build_structured_result(reward, detail, verifier_completed=True, verifier_returncode=0)
    assert result["valid"] is False and result["score"] == 0.0


formal = load_module("shared_final_formal", "scripts/validate_formal_artifacts.py")


def test_formal_validator_binds_the_judge_identity():
    assert formal.PRIVATE_MEMBER == verifier.RESULT_NAME
    assert formal.FINAL_SUITE_ID == verifier.FINAL_SUITE_ID
    assert formal.EDGE_SUITE_ID == verifier.EDGE_SUITE_ID
    assert formal.EDGE_CASE_COUNT == verifier.EDGE_CASE_COUNT
    assert formal.EVALUATION_POLICY == POLICY
    assert formal.SELECTION_PROTOCOL_ID == PROTOCOL


@pytest.mark.parametrize(
    "mutation",
    [None, "old_policy", "v2_protocol", "v1_suite", "wrong_manifest", "short_suite", "edge_failed",
     "edge_short", "no_edge_digest", "wrong_edge_digest"],
)
def test_formal_artifact_validator_requires_shared_final_identity(tmp_path, mutation):
    _reward, detail = records()
    if mutation == "old_policy":
        detail["evaluation_policy"] = "full-final-shared-v2"
    elif mutation == "v2_protocol":
        detail["selection_protocol_id"] = "sparse-mla-shared-final-v2"
    elif mutation == "v1_suite":
        detail["suite_id"] = "final-holdout-v1"
    elif mutation == "wrong_manifest":
        detail["manifest_sha256"] = "b" * 64
    elif mutation == "short_suite":
        detail["case_count"] = 180.0
    elif mutation == "edge_failed":
        detail["edge_cases_failed"] = 1.0
    elif mutation == "edge_short":
        detail["edge_cases_total"] -= 1.0
    elif mutation == "no_edge_digest":
        detail.pop("edge_manifest_sha256")
    elif mutation == "wrong_edge_digest":
        detail["edge_manifest_sha256"] = "f" * 64
    raw = json.dumps(detail).encode()
    artifact = tmp_path / formal.PRIVATE_ARTIFACT
    with tarfile.open(artifact, "w:gz") as bundle:
        member = tarfile.TarInfo(formal.PRIVATE_MEMBER)
        member.size = len(raw)
        bundle.addfile(member, io.BytesIO(raw))
    artifact.chmod(0o600)
    kwargs = dict(root=tmp_path, expected_score=100.0, expected_suite_id="final-holdout-v4",
                  expected_manifest_sha256="a" * 64, expected_case_count=997,
                  expected_edge_manifest_sha256="e" * 64)
    if mutation is None:
        accepted = formal._validate_private_artifact(artifact, **kwargs)
        assert accepted["member"] == verifier.RESULT_NAME
        assert accepted["suite_id"] == "final-holdout-v4" and accepted["case_count"] == 997
        assert accepted["manifest_sha256"] == "a" * 64
        assert accepted["edge_manifest_sha256"] == "e" * 64
    else:
        with pytest.raises(formal.ArtifactValidationError):
            formal._validate_private_artifact(artifact, **kwargs)
