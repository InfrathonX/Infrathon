"""CPU regressions for shared full-final routing and private result identity."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import tarfile
from pathlib import Path

import pytest

from conftest import judge_only_file

ROOT = Path(__file__).resolve().parents[1]


def load(relative):
    spec = importlib.util.spec_from_file_location("shared_final_" + ROOT.name + relative.replace("/", "_"), ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verifier = load("images/judge/verify_quant_attention.py")
formal = load("scripts/validate_formal_artifacts.py")


@pytest.mark.parametrize("round_id", ["auto-1", "agent-2", "manual-1", "candidate-v3"])
def test_every_admitted_round_sends_the_complete_frozen_suite_to_worker(tmp_path, monkeypatch, round_id):
    context = tmp_path / "context.json"
    context.write_text(json.dumps({"round_id": round_id}))
    context.chmod(0o600)
    original_profile = verifier.load_evaluation_profile
    monkeypatch.setattr(verifier, "load_evaluation_profile", lambda: original_profile(context, expected_uid=os.getuid()))
    raw = judge_only_file("judge/private/final_cases.json").read_bytes()
    frozen_cases = tuple(json.loads(raw)["cases"])
    frozen_digest = hashlib.sha256(raw).hexdigest()
    loads = []

    def frozen():
        loads.append(True)
        return frozen_cases, frozen_digest

    monkeypatch.setattr(verifier, "load_final_manifest", frozen)
    monkeypatch.setattr(verifier, "verify_protected_hashes", lambda path: [])
    monkeypatch.setattr(verifier, "locate_submission", lambda path: tmp_path)
    candidate = tmp_path / "candidate"
    candidate.write_text("// CPU fixture")
    monkeypatch.setattr(verifier, "validate_submission", lambda root, override: {
        "ok": True, "final_version": 3, "num_candidate_versions": 3,
        "selected_path": str(candidate), "selected_sha256": "c" * 64,
    })
    if hasattr(verifier, "_trusted_hardware"):
        monkeypatch.setattr(verifier, "_trusted_hardware", lambda: {})
    run_root = tmp_path / "worker-run"
    run_root.mkdir()
    monkeypatch.setattr(verifier.tempfile, "mkdtemp", lambda **kwargs: str(run_root))
    calls = []

    def worker(source, run_root, cases, suite_id, **kwargs):
        calls.append((tuple(cases), suite_id, kwargs))
        raise RuntimeError("CPU worker capture")

    monkeypatch.setattr(verifier, "run_worker", worker)
    args = verifier.parse_args(["--app-dir", str(tmp_path)])
    rc, detail = verifier.evaluate(args)
    assert rc == 1 and "CPU worker capture" in detail["private_error"]
    assert loads == [True] and len(calls) == 1
    assert calls[0][0] == frozen_cases
    assert calls[0][1] == verifier.FINAL_SUITE_ID
    assert detail["round_id"] == round_id
    assert detail["evaluation_profile"] == "final" and detail["profile_code"] == 2.0
    assert detail["manifest_sha256"] == frozen_digest
    assert detail["case_count"] == len(frozen_cases)
    assert detail["evaluation_policy"] == "full-final-shared-v3"
    assert detail["selection_protocol_id"] == verifier.SELECTION_PROTOCOL_ID
    if hasattr(args, "trials"):
        assert calls[0][2]["trials"] == args.trials
        assert calls[0][2]["warmups"] == args.warmups


@pytest.mark.parametrize("round_id", ["auto-1", "agent-1"])
def test_shared_suite_does_not_grant_interactive_candidate_override(tmp_path, round_id):
    context = tmp_path / "context.json"
    context.write_text(json.dumps({"round_id": round_id, "candidate_version": 3}))
    context.chmod(0o600)
    with pytest.raises(RuntimeError):
        verifier.load_evaluation_profile(context, expected_uid=os.getuid())


@pytest.mark.parametrize("round_id", ["manual-1", "candidate-v3"])
def test_host_replay_keeps_override_and_original_identity(tmp_path, round_id):
    context = tmp_path / "context.json"
    context.write_text(json.dumps({"round_id": round_id, "candidate_version": 3}))
    context.chmod(0o600)
    assert verifier.load_evaluation_profile(context, expected_uid=os.getuid()) == (
        "final", verifier.FINAL_SUITE_ID, round_id, 3,
    )


@pytest.mark.parametrize("round_id", ["auto-", "agent-", "manual-", "candidate-", "preview-1"])
def test_unknown_or_bare_rounds_fail_closed(tmp_path, round_id):
    context = tmp_path / "context.json"
    context.write_text(json.dumps({"round_id": round_id}))
    context.chmod(0o600)
    with pytest.raises(RuntimeError):
        verifier.load_evaluation_profile(context, expected_uid=os.getuid())


@pytest.mark.parametrize("mutation", [None, "legacy_policy", "anchor_suite", "wrong_digest", "short_suite"])
def test_formal_private_result_binds_policy_and_complete_frozen_identity(tmp_path, mutation):
    detail = {
        "evaluation_policy": verifier.EVALUATION_POLICY,
        "selection_protocol_id": verifier.SELECTION_PROTOCOL_ID,
        "suite_id": verifier.FINAL_SUITE_ID,
        "manifest_sha256": "a" * 64,
        "case_count": 20 if "lmcache" in ROOT.name else (774 if "sageattention" in ROOT.name else 8),
        "score_gbps" if "lmcache" in ROOT.name else "score_tflops": 1.0,
    }
    expected_count = detail["case_count"]
    if mutation == "legacy_policy":
        detail["selection_protocol_id"] = "legacy-anchor-v1"
    elif mutation == "anchor_suite":
        detail["suite_id"] = getattr(verifier, "ANCHOR_SUITE_ID", "final-holdout-v2")
    elif mutation == "wrong_digest":
        detail["manifest_sha256"] = "b" * 64
    elif mutation == "short_suite":
        detail["case_count"] -= 1
    payload = json.dumps(detail).encode()
    archive = tmp_path / "private.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        member = tarfile.TarInfo(formal.PRIVATE_MEMBER)
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))
    archive.chmod(0o600)
    kwargs = dict(root=tmp_path, expected_score=1.0, expected_suite_id=verifier.FINAL_SUITE_ID,
                  expected_manifest_sha256="a" * 64, expected_case_count=expected_count)
    if mutation is None:
        assert formal._validate_private_artifact(archive, **kwargs)["case_count"] == expected_count
    else:
        with pytest.raises(formal.ArtifactValidationError):
            formal._validate_private_artifact(archive, **kwargs)


@pytest.mark.parametrize("mutation", [None, "missing_policy", "old_protocol", "old_profile"])
def test_adapter_accepts_only_current_full_final_identity(mutation):
    adapter = load("images/judge/run_judge.py")
    detail = verifier.failure_detail("", profile="final", suite_id=verifier.FINAL_SUITE_ID)
    detail.update(reward=1.0, final_version=1.0, num_candidate_versions=1.0,
                  manifest_sha256="a" * 64,
                  case_count=20.0 if "lmcache" in ROOT.name else (774.0 if "sageattention" in ROOT.name else 8.0))
    for gate, _label in adapter.GATES:
        detail[gate] = 1.0
    if hasattr(adapter, "GROUP_KEYS"):
        detail.update(score_gbps=1.0, edge_case_count=14.0)
        families = adapter.GROUP_KEYS
    else:
        detail["score_tflops"] = 1.0
        families = adapter.FAMILY_KEYS
        if "sageattention" in ROOT.name:
            detail["edge_case_count"] = 8.0
            detail["real_layer_case_count"] = 139.0
    for key in families:
        detail[key] = 1.0
    if mutation == "missing_policy":
        detail.pop("evaluation_policy")
    elif mutation == "old_protocol":
        detail["selection_protocol_id"] = "legacy-anchor-v1"
    elif mutation == "old_profile":
        detail.update(evaluation_profile="auto", profile_code=1.0, suite_id=getattr(verifier, "ANCHOR_SUITE_ID", "final-holdout-v2"))
    numeric = {k: v for k, v in detail.items() if isinstance(v, (int, float))}
    result = adapter.build_structured_result(numeric, detail, verifier_completed=True, verifier_returncode=0)
    assert result["valid"] is (mutation is None)
    if mutation is None:
        assert result["metrics"]["evaluation_policy"] == verifier.EVALUATION_POLICY
        assert result["metrics"]["selection_protocol_id"] == verifier.SELECTION_PROTOCOL_ID
    else:
        assert result["score"] == 0.0


def test_context_permissions_still_fail_closed(tmp_path):
    context = tmp_path / "context.json"
    context.write_text(json.dumps({"round_id": "agent-1"}))
    context.chmod(0o644)
    with pytest.raises(RuntimeError):
        verifier.load_evaluation_profile(context, expected_uid=os.getuid())


@pytest.mark.parametrize("mutation", [None, "wrong_digest", "source_symlink", "destination_symlink"])
def test_real_launcher_stages_digest_bound_manifest_inside_task_root(tmp_path, mutation):
    import subprocess
    import sys

    # Match the real logs/runs/RUN_ID versus logs/runs/RUN_ID/TASK_ID topology.
    run_root = tmp_path / "logs/runs/run-1"
    task_root = run_root / "task-1"
    task_root.mkdir(parents=True)
    source = run_root / "experiment-manifest.json"
    payload = json.dumps({
        "evaluation_policy": formal.EVALUATION_POLICY,
        "selection_protocol_id": formal.SELECTION_PROTOCOL_ID,
        "final_holdout": {"suite_id": formal.FINAL_SUITE_ID, "manifest_sha256": "a" * 64, "case_count": 8},
    }).encode()
    source.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    destination = task_root / "experiment-manifest.json"
    if mutation == "wrong_digest":
        digest = "b" * 64
    elif mutation == "source_symlink":
        original = run_root / "original.json"
        source.rename(original)
        source.symlink_to(original)
    elif mutation == "destination_symlink":
        destination.symlink_to(source)
    runner = (ROOT / "scripts/run_fixed_budget.sh").read_text()
    start = runner.index('"${PYTHON_BIN}" "${MIGRATION_ROOT}/scripts/validate_formal_artifacts.py" stage')
    end = runner.index('formal_precheck="${TASK_RUN_ROOT}/formal-precheck.json"', start)
    assert start < runner.index('"${MIGRATION_ROOT}/scripts/validate_formal_artifacts.py" pre')
    # Execute the actual launcher staging block without its container/GPU body.
    env = dict(os.environ, PYTHON_BIN=sys.executable, MIGRATION_ROOT=str(ROOT),
               TASK_RUN_ROOT=str(task_root), experiment_manifest=str(source),
               experiment_manifest_sha256="sha256:" + digest)
    result = subprocess.run(["bash", "-eu", "-c", runner[start:end]], env=env,
                            capture_output=True, text=True)
    if mutation is None:
        assert result.returncode == 0, result.stderr
        assert destination.read_bytes() == payload
        assert destination.stat().st_mode & 0o777 == 0o600
        recorded, _path, actual_digest = formal._json_file(destination, "experiment manifest", root=task_root)
        assert recorded["selection_protocol_id"] == formal.SELECTION_PROTOCOL_ID
        assert actual_digest == digest
    else:
        assert result.returncode != 0
        assert source.read_bytes() == payload
        assert not destination.exists() or destination.is_symlink()
