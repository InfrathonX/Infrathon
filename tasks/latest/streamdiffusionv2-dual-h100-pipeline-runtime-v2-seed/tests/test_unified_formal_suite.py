"""CPU checks execute formal routing without starting GPU or candidate processes."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def verifier(monkeypatch, final_cases_file):
    # Isolate the GPU-dependent comparator imports. The real verifier and its
    # manifest, selection, security, and score control flow run in these tests.
    dependencies = {
        "evaluation": {"compare_saved_ledger": None, "digest_saved_ledger": None},
        "hardware": {"MIN_COMPUTE_CAPABILITY": (8, 0), "MIN_DEVICE_MEMORY_BYTES": 26 * 1024**3,
                     "MIN_HOST_MEMORY_BYTES": 64 * 1024**3},
        "source_admission": {"AdmissionError": ValueError, "admit_candidate": lambda path: {}},
    }
    for name, values in dependencies.items():
        module = types.ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
    scoring_spec = importlib.util.spec_from_file_location("scoring", ROOT / "runtime/scoring.py")
    scoring = importlib.util.module_from_spec(scoring_spec)
    scoring_spec.loader.exec_module(scoring)
    monkeypatch.setitem(sys.modules, "scoring", scoring)
    spec = importlib.util.spec_from_file_location("sdv2_unified_verifier", ROOT / "images/judge/verify_sdv2_pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "PRIVATE_ROOT", final_cases_file.parent if final_cases_file else ROOT / "build/private")
    monkeypatch.setattr(module, "WORKER_RUNTIME", ROOT / "runtime")
    monkeypatch.delenv("SDV2_TOTAL_TIMEOUT", raising=False)
    return module


@pytest.mark.parametrize("round_id", ["agent-1", "auto-7", "manual-1", "candidate-v003-a1", "final"])
def test_formal_rounds_run_identical_complete_suite(verifier, tmp_path, monkeypatch, round_id, final_cases_path):
    workspace = tmp_path / "app/workdir/sdv2_pipeline_lab"
    candidate = workspace / "candidates/runtime_v002"
    candidate.mkdir(parents=True)
    (workspace / "FINAL_VERSION").write_text("2\n")
    context = tmp_path / "context.json"
    context.write_text(json.dumps({"round_id": round_id}))
    monkeypatch.setenv("SDV2_PROFILE_FILE", str(context))
    monkeypatch.setattr(verifier, "verify_protected_hashes", lambda: {})
    calls = []

    def seed(manifest, log_dir):
        calls.append(("seed", manifest))
        seed_job = tmp_path / "seed-job"
        seed_job.mkdir()
        reference_root = tmp_path / "reference"
        reference_root.mkdir()
        return {"hardware": {}, "cases": {}}, reference_root, seed_job

    def spawn(selected, manifest, log_dir):
        assert selected == candidate
        calls.append(("candidate", manifest))
        job = tmp_path / "candidate-job"
        job.mkdir()
        return {"cases": [{"id": c["id"], "steady_fps": 40., "push_to_emit_p99_seconds": 1.,
                           "seed_push_to_emit_p99_seconds": 1.} for c in manifest["cases"]]}, job

    monkeypatch.setattr(verifier, "_seed_reference", seed)
    monkeypatch.setattr(verifier, "_spawn_worker", spawn)
    monkeypatch.setattr(verifier, "_compare_outputs", lambda *args: {})
    for name in ["_validate_hardware_record", "_validate_completeness", "_validate_parity",
                 "_validate_liveness", "_validate_resources", "_validate_p99", "_validate_construction"]:
        monkeypatch.setattr(verifier, name, lambda *args: None)
    monkeypatch.setattr(verifier, "_decorate_and_validate_ttff", lambda worker, *args: worker["cases"])
    result = verifier.verify(tmp_path / "app", tmp_path / "logs")
    expected = json.loads(final_cases_path.read_text())
    assert calls == [("seed", expected), ("candidate", expected)]
    assert len(result["rows"]) == result["formal_case_count"] == 12
    assert result["valid"] and all(result["gates"].values()), result["failure_detail"]
    assert result["suite_id"] == expected["suite"]
    assert result["evaluation_protocol_id"] == "sdv2-pipeline-v2-full-final"
    assert result["score"] == pytest.approx(40.)
    assert result["suite_manifest_sha256"] == hashlib.sha256(final_cases_path.read_bytes()).hexdigest()


@pytest.mark.parametrize("round_id", ["agent-0", "agent-01", "agent-١", "agent-one", "agent-1/x", "auto-", "public-1", "", None])
def test_invalid_round_identity_is_rejected(verifier, round_id):
    with pytest.raises(verifier.VerificationError):
        verifier._suite_for_round(round_id)


@pytest.mark.parametrize("round_id", ["agent-1", "auto-2"])
@pytest.mark.parametrize("override", [None, 1, True, "1"])
def test_interactive_override_cannot_select_another_candidate(verifier, tmp_path, monkeypatch, round_id, override):
    context = tmp_path / "context.json"
    context.write_text(json.dumps({"round_id": round_id, "candidate_version": override}))
    monkeypatch.setenv("SDV2_PROFILE_FILE", str(context))
    with pytest.raises(verifier.VerificationError, match="without an override"):
        verifier.load_profile(tmp_path)


def test_host_replay_override_preserves_the_shared_suite(verifier, tmp_path, monkeypatch):
    context = tmp_path / "context.json"
    context.write_text(json.dumps({"round_id": "candidate-v003-a1", "candidate_version": 3}))
    monkeypatch.setenv("SDV2_PROFILE_FILE", str(context))
    profile = verifier.load_profile(tmp_path)
    assert verifier._candidate_dir(tmp_path, profile) == tmp_path / "workdir/sdv2_pipeline_lab/candidates/runtime_v003"
    assert verifier._suite_for_round(profile["round_id"]) == "final"


def test_formal_manifest_tampering_is_rejected(verifier, tmp_path, monkeypatch, final_cases_path):
    private = tmp_path / "private"
    private.mkdir()
    source = final_cases_path
    (private / "final_cases.json").write_bytes(source.read_bytes() + b"\n")
    monkeypatch.setattr(verifier, "PRIVATE_ROOT", private)
    with pytest.raises(verifier.VerificationError, match="hash mismatch"):
        verifier._formal_suite()


def test_formal_protocol_timeout_cannot_silently_change(verifier, monkeypatch):
    monkeypatch.setenv("SDV2_TOTAL_TIMEOUT", "9999")
    with pytest.raises(verifier.VerificationError, match="protocol mismatch"):
        verifier._formal_suite()


def test_missing_build_assets_cannot_remove_protected_hashes(tmp_path):
    task = tmp_path / "task"
    output = task / "images/judge/protected_hashes/h100-80gb-700w.txt"
    output.parent.mkdir(parents=True)
    output.write_text("existing protected inputs\n")
    completed = subprocess.run([
        sys.executable, str(ROOT / "scripts/generate_protected_hashes.py"),
        "--task", str(task), "--assets", str(tmp_path / "missing-assets"),
        "--hardware", "h100-80gb-700w",
    ], capture_output=True, text=True)
    assert completed.returncode != 0
    assert "missing protected input directory" in completed.stderr
    assert output.read_text() == "existing protected inputs\n"


def test_protocol_files_and_protected_runtime_bind_one_suite(final_cases_path):
    spec = json.loads((ROOT / "tasks/streamdiffusionv2_dual_h100_pipeline_runtime_seed_v2.json").read_text())
    portable = spec["portable"]
    # Envelope form: the harness parses it like the other dual-GPU tasks.
    assert "evaluation_suites" not in portable and "task_core" not in portable
    formal = json.loads((ROOT / "portable/formal-suite-manifest.json").read_text())
    selection = json.loads((ROOT / "portable/selection-protocol.json").read_text())
    policy = json.loads((ROOT / "runtime/evaluation_protocol.json").read_text())
    private = hashlib.sha256(final_cases_path.read_bytes()).hexdigest()
    assert policy["selected_manifest_sha256"] == formal["selected_manifest_sha256"] == private
    assert selection["final"]["selected_manifest_sha256"] == selection["interactive"]["selected_manifest_sha256"] == private
    assert policy["timeouts"] == formal["timeouts"] == selection["timeouts"]
    assert policy["timeouts"]["outer_judge_seconds"] == spec["judge"]["eval_timeout"]
    assert "cases" not in policy and "cases" not in formal
    expected = hashlib.sha256((ROOT / "runtime/evaluation_protocol.json").read_bytes()).hexdigest()
    for hardware in ("h100-94gb-400w", "h100-80gb-700w"):
        protected = (ROOT / f"images/judge/protected_hashes/{hardware}.txt").read_text()
        assert expected + " worker:evaluation_protocol.json" in protected
