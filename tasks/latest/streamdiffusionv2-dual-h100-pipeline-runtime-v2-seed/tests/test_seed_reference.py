"""Every seed-relative limit comes from the in-run seed pass, not a frozen file."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

TASK_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = TASK_ROOT / "runtime"
JUDGE_DIR = TASK_ROOT / "images" / "judge"


def _stub_torch() -> bool:
    """Return True when a real torch is active, False when a stub stands in.

    Another test module in the same session may already have installed the
    stub, so probe for a real API rather than trusting a bare import.
    """
    try:
        import torch  # noqa: F401
    except Exception:
        torch = types.ModuleType("torch")
        torch.__version__ = "0.0-test-stub"
        torch.bfloat16 = object()
        torch.uint16 = object()
        torch.Tensor = type("Tensor", (), {})
        torch.cuda = types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
        torch.version = types.SimpleNamespace(cuda="0.0-test-stub")
        sys.modules["torch"] = torch
    if "sdv2_core.sink" not in sys.modules:
        try:
            import sdv2_core.sink  # noqa: F401
        except Exception:
            package = types.ModuleType("sdv2_core")
            package.__path__ = []
            sink = types.ModuleType("sdv2_core.sink")
            sink.RunLedger = type("RunLedger", (), {})
            package.sink = sink
            sys.modules["sdv2_core"] = package
            sys.modules["sdv2_core.sink"] = sink
    return hasattr(sys.modules["torch"], "from_numpy")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(RUNTIME_DIR))
sys.path.insert(0, str(JUDGE_DIR))
REAL_TORCH = _stub_torch()
scoring = _load(RUNTIME_DIR / "scoring.py", "sdv2_scoring_seedref")
evaluation = _load(RUNTIME_DIR / "evaluation.py", "sdv2_evaluation_seedref")
verifier = _load(JUDGE_DIR / "verify_sdv2_pipeline.py", "sdv2_verifier_seedref")

CHUNKS = 3
CASES = ("case_alpha", "case_beta")
MANIFEST = {"cases": [{"id": name, "chunks_expected": CHUNKS} for name in CASES]}
HARDWARE = {
    "count": 2,
    "host_memory_bytes": 96 * 1024**3,
    "devices": [
        {
            "index": index,
            "name": "NVIDIA RTX A6000",
            "uuid": f"GPU-fake-{index}",
            "capability": [8, 6],
            "total_memory_bytes": 48 * 1024**3,
            "peer_access_to_other": True,
        }
        for index in range(2)
    ],
}
LIVENESS = {
    "clean_exit": True,
    "terminated_session_pids": [],
    "leftover_session_pids": [],
    "active_gpu_pids": [],
    "gpus_idle": True,
}


def _write_artifacts(root: Path, *, offset: float = 0.0) -> None:
    """Write one case's saved-ledger directory the way OutputSink.save does."""
    root.mkdir(parents=True, exist_ok=True)
    frames, latents, records, pushes = {}, {}, [], {}
    for index in range(CHUNKS):
        key = f"chunk_{index:05d}"
        frames[key] = np.full((4, 8, 8, 3), float(index) + offset, dtype=np.float32)
        latents[key] = np.full((1, 1, 4, 2, 2), 16000 + index, dtype=np.uint16)
        records.append({"chunk_idx": index, "kind": "frames", "timestamp_ns": 10 + index, "shape": [4, 8, 8, 3], "dtype": "torch.float32"})
        records.append({"chunk_idx": index, "kind": "latent", "timestamp_ns": 10 + index, "shape": [1, 1, 4, 2, 2], "dtype": "torch.bfloat16"})
        pushes[str(index)] = index
    np.savez(root / "frames.npz", **frames)
    np.savez(root / "latents.npz", **latents)
    (root / "ledger.json").write_text(json.dumps(records), encoding="utf-8")
    (root / "push_timestamps.json").write_text(json.dumps(pushes), encoding="utf-8")


def _seed_worker(job: Path, *, fps=(30.0, 40.0), peaks=(10000, 12000)) -> dict:
    for index, case_id in enumerate(CASES):
        _write_artifacts(job / "artifacts" / case_id)
    return {
        "ok": True,
        "hardware": HARDWARE,
        "liveness": dict(LIVENESS),
        "resources": {
            "baseline_memory_mib_per_rank": {"0": 3, "1": 4},
            "post_close_memory_mib_per_rank": {"0": 5, "1": 6},
            "gpu_uuid_per_rank": {"0": "GPU-fake-0", "1": "GPU-fake-1"},
            "monitor_errors": [],
        },
        "cases": [
            {
                "id": case_id,
                "chunks_expected": CHUNKS,
                "frame_chunks": CHUNKS,
                "latent_chunks": CHUNKS,
                "steady_fps": fps[index],
                "ttff_seconds": 2.0 + index,
                "push_to_emit_p99_seconds": 1.0 + index,
                "construction_seconds": 20.0 + index,
                "peak_memory_mib_per_rank": {"0": peaks[0] + index, "1": peaks[1] + index},
                "host_rss_growth_mib": 1000.0,
            }
            for index, case_id in enumerate(CASES)
        ],
    }


@pytest.fixture
def reference(tmp_path, monkeypatch):
    job = tmp_path / "seed-job"
    worker = _seed_worker(job)
    monkeypatch.setattr(verifier, "admit_candidate", lambda path: {"version": 1})
    monkeypatch.setattr(verifier, "_spawn_worker", lambda *args, **kwargs: (worker, job))
    real_mkdtemp = verifier.tempfile.mkdtemp

    def mkdtemp(*args, **kwargs):
        if kwargs.get("prefix") == "sdv2-reference-":
            (tmp_path / "reference").mkdir(exist_ok=True)
            return str(tmp_path / "reference")
        return real_mkdtemp(*args, **kwargs)

    monkeypatch.setattr(verifier.tempfile, "mkdtemp", mkdtemp)
    return verifier._seed_reference(MANIFEST, tmp_path)


def test_reference_is_measured_in_run_and_keyed_by_rank(reference):
    payload, root, job = reference
    assert payload["runtime"] == "runtime_v001"
    assert "re-measured on the evaluation devices" in payload["source"]
    assert set(payload["cases"]) == set(CASES)
    for case_id in CASES:
        row = payload["cases"][case_id]
        assert set(row["peak_memory_mib_per_rank"]) == {"0", "1"}
        assert row["steady_fps"] > 0 and row["ttff_seconds"] > 0 and row["push_to_emit_p99_seconds"] > 0
        manifest = json.loads((root / case_id / "manifest.json").read_text())
        assert manifest["origin"] == "in_run_seed_reference"
        assert manifest["chunks"] == CHUNKS
        assert set(manifest["files"]) == {"frame_hashes.json", "latents.npz"}
    assert payload["hardware"] == HARDWARE


def test_seed_failures_are_trusted_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(verifier, "admit_candidate", lambda path: {"version": 1})

    def explode(*args, **kwargs):
        raise verifier.VerificationError("seed worker exited 1")

    monkeypatch.setattr(verifier, "_spawn_worker", explode)
    with pytest.raises(verifier.SeedReferenceError):
        verifier._seed_reference(MANIFEST, tmp_path)


def test_verify_reports_a_seed_failure_as_a_trusted_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(verifier, "load_profile", lambda app_dir: {"round_id": "auto-1"})
    monkeypatch.setattr(verifier, "verify_protected_hashes", lambda: {})
    monkeypatch.setattr(verifier, "_candidate_dir", lambda app_dir, profile: tmp_path / "runtime_v001")
    monkeypatch.setattr(verifier, "admit_candidate", lambda path: {"version": 1})
    monkeypatch.setattr(verifier, "_read_json", lambda path, limit: MANIFEST)
    monkeypatch.setattr(verifier, "_formal_suite", lambda: (MANIFEST, {}))

    def explode(*args, **kwargs):
        raise verifier.SeedReferenceError("seed reference pass failed: boom")

    monkeypatch.setattr(verifier, "_seed_reference", explode)
    result = verifier.verify(tmp_path, tmp_path)
    assert result["valid"] is False and result["score"] == 0.0
    assert result["trusted_failure"] is True
    assert result["gates"]["seed_reference_ok"] is False
    assert result["gates"]["admission_ok"] is True


def test_memory_gate_is_rank_keyed_seed_relative_and_symmetric():
    seed = {"peak_memory_mib_per_rank": {"0": 10000, "1": 12000}}
    # Both ranks share the cap set by the seed's larger peak.
    assert scoring.memory_ok({"0": 20192, "1": 20192}, seed)
    assert not scoring.memory_ok({"0": 20193, "1": 0}, seed)
    assert not scoring.memory_ok({"0": 10000}, seed)
    # A UUID-keyed baseline is exactly what this change removed.
    assert not scoring.memory_ok(
        {"GPU-72590098": 1, "GPU-9a81af24": 1},
        {"peak_memory_mib_per_rank": {"GPU-72590098": 10000, "GPU-9a81af24": 12000}},
    )
    assert not scoring.memory_ok({"0": 1, "1": 1}, {"peak_memory_mib_per_gpu": {"0": 10000, "1": 12000}})


def test_post_close_memory_is_relative_to_this_run_baseline():
    baseline = {"0": 400, "1": 900}
    margin = scoring.POST_CLOSE_MEMORY_MARGIN_MIB
    assert scoring.post_close_memory_ok({"0": 400 + margin, "1": 900}, baseline)
    assert not scoring.post_close_memory_ok({"0": 400 + margin + 1, "1": 900}, baseline)
    # A busy machine whose idle occupancy already exceeds the old absolute cap
    # is admitted, because the gate now measures what this run left behind.
    assert scoring.post_close_memory_ok({"0": 9000, "1": 9000}, {"0": 9000, "1": 9000})
    assert not scoring.post_close_memory_ok({"0": 100, "1": 100}, {"uuid-a": 100, "uuid-b": 100})


def test_resource_gate_uses_the_in_run_reference(reference):
    payload, _, _ = reference
    worker = {
        "resources": {
            "baseline_memory_mib_per_rank": {"0": 10, "1": 10},
            "post_close_memory_mib_per_rank": {"0": 20, "1": 20},
            "monitor_errors": [],
        },
        "cases": [
            {
                "id": case_id,
                "peak_memory_mib_per_rank": {
                    rank: scoring.memory_cap_mib(payload["cases"][case_id]["peak_memory_mib_per_rank"])
                    for rank in ("0", "1")
                },
                "host_rss_growth_mib": 1000.0,
            }
            for case_id in CASES
        ],
    }
    verifier._validate_resources(worker, payload)
    worker["cases"][0]["peak_memory_mib_per_rank"]["0"] += 1
    with pytest.raises(verifier.VerificationError, match="GPU memory cap"):
        verifier._validate_resources(worker, payload)
    worker["cases"][0]["peak_memory_mib_per_rank"]["0"] -= 1
    worker["cases"][1]["host_rss_growth_mib"] = scoring.HOST_RSS_GROWTH_CAP_MIB + 1
    with pytest.raises(verifier.VerificationError, match="host RSS"):
        verifier._validate_resources(worker, payload)


def test_timing_limits_come_from_the_reference_pass(reference):
    payload, _, _ = reference
    seed = payload["cases"][CASES[0]]
    row = {
        "id": CASES[0],
        "steady_fps": 2 * seed["steady_fps"],
        "push_to_emit_p99_seconds": seed["push_to_emit_p99_seconds"],
        "ttff_seconds": seed["ttff_seconds"],
        "construction_seconds": seed["construction_seconds"],
    }
    worker = {"cases": [row]}
    comparisons = {CASES[0]: {"parity_ok": True}}
    rows = verifier._decorate_and_validate_ttff(worker, comparisons, payload)
    assert rows[0]["seed_steady_fps"] == seed["steady_fps"]
    assert rows[0]["p99_limit_seconds"] == 3 * seed["push_to_emit_p99_seconds"]
    assert rows[0]["ttff_limit_seconds"] == max(
        2.5 * seed["ttff_seconds"], seed["ttff_seconds"] + 4
    )
    assert rows[0]["construction_limit_seconds"] == max(
        2 * seed["construction_seconds"], seed["construction_seconds"] + 60
    )
    verifier._validate_construction(rows)
    over = dict(rows[0], construction_seconds=rows[0]["construction_limit_seconds"] + 1e-6)
    with pytest.raises(verifier.VerificationError, match="construction exceeded"):
        verifier._validate_construction([over])
    row["ttff_seconds"] = rows[0]["ttff_limit_seconds"] + 1e-6
    with pytest.raises(verifier.VerificationError, match="TTFF exceeded"):
        verifier._decorate_and_validate_ttff(worker, comparisons, payload)
    missing = {"cases": {}}
    with pytest.raises(verifier.VerificationError, match="seed reference is missing"):
        verifier._decorate_and_validate_ttff(worker, comparisons, missing)


def test_no_frozen_device_table_and_the_seed_is_protected():
    source = (JUDGE_DIR / "verify_sdv2_pipeline.py").read_text()
    assert "calibration" not in source.replace("calibrated", "")
    assert not (RUNTIME_DIR / "calibration_v1.json").exists()
    generator = (TASK_ROOT / "scripts/generate_protected_hashes.py").read_text()
    assert 'f"seed:{path.relative_to(seed).as_posix()}"' in generator, "the in-run reference seed must be protected"
    assert 'f"worker:{path.relative_to(runtime).as_posix()}"' in generator


@pytest.mark.skipif(not REAL_TORCH, reason="bit-exact ledger comparison needs the pinned torch")
def test_candidate_is_compared_against_the_in_run_seed_outputs(tmp_path):
    seed_dir = tmp_path / "seed" / CASES[0]
    _write_artifacts(seed_dir)
    evaluation.digest_saved_ledger(seed_dir, tmp_path / "reference" / CASES[0], expected_chunks=CHUNKS)
    same = tmp_path / "same"
    _write_artifacts(same)
    verdict = evaluation.compare_saved_ledger(same, tmp_path / "reference" / CASES[0])
    assert verdict["parity_ok"] and verdict["frame_exact"] and verdict["latent_exact"]
    drifted = tmp_path / "drifted"
    _write_artifacts(drifted, offset=1e-6)
    verdict = evaluation.compare_saved_ledger(drifted, tmp_path / "reference" / CASES[0])
    assert verdict["parity_ok"] is False and verdict["frame_exact"] is False


def test_reference_digest_rejects_an_incomplete_reference(tmp_path):
    source = tmp_path / "short"
    _write_artifacts(source)
    with pytest.raises(ValueError, match="incomplete"):
        evaluation.digest_saved_ledger(source, tmp_path / "out", expected_chunks=CHUNKS + 1)


def test_gate_set_is_consistent_across_verifier_adapter_and_validators():
    adapter = _load(TASK_ROOT / "judge/run_judge.py", "sdv2_adapter_seedref")
    assert set(adapter.GATES) == set(verifier.GATE_NAMES)
    assert "seed_reference_ok" in adapter.GATES
    formal = (TASK_ROOT / "scripts/validate_formal_artifacts.py").read_text()
    layout = (TASK_ROOT / "scripts/validate_layout.py").read_text()
    for gate in verifier.GATE_NAMES:
        assert gate in formal and gate in layout, gate


@pytest.mark.skipif(not REAL_TORCH, reason="end-to-end verify() needs the pinned torch")
def test_verify_end_to_end_scores_against_the_in_run_seed(tmp_path, monkeypatch):
    seed_job = tmp_path / "seed-job"
    candidate_job = tmp_path / "candidate-job"
    seed = _seed_worker(seed_job)
    candidate = _seed_worker(candidate_job, fps=(60.0, 80.0), peaks=(10000, 12000))
    calls = []

    def spawn(runtime, manifest, log_dir, label="candidate"):
        calls.append(label)
        return (seed, seed_job) if label == "seed" else (candidate, candidate_job)

    monkeypatch.setattr(verifier, "load_profile", lambda app_dir: {"round_id": "auto-1"})
    monkeypatch.setattr(verifier, "verify_protected_hashes", lambda: {})
    monkeypatch.setattr(verifier, "_candidate_dir", lambda app_dir, profile: tmp_path / "runtime_v002")
    monkeypatch.setattr(verifier, "admit_candidate", lambda path: {"version": 2})
    monkeypatch.setattr(verifier, "_read_json", lambda path, limit: MANIFEST)
    monkeypatch.setattr(verifier, "_formal_suite", lambda: (MANIFEST, {}))
    monkeypatch.setattr(verifier, "_spawn_worker", spawn)
    result = verifier.verify(tmp_path, tmp_path)
    assert result["failure_detail"] is None, result["failure_detail"]
    assert calls == ["seed", "candidate"], "the seed reference must be measured before the candidate"
    assert result["valid"] is True and all(result["gates"].values())
    assert result["trusted_failure"] is False and result["calibrated"] is True
    assert result["seed_reference"]["cases"][CASES[0]]["steady_fps"] == 30.0
    # Identical p99 leaves the latency factor at 1, so the score is the raw FPS geomean.
    assert result["score"] == pytest.approx((60.0 * 80.0) ** 0.5)
    assert result["rows"][0]["parity"]["reference"] == "in_run_seed_reference"
    assert result["gates"]["construction_ok"] is True
    assert result["rows"][0]["construction_limit_seconds"] == pytest.approx(80.0)
    # The seed job and the reference digest are removed once the run finishes.
    assert not seed_job.exists() and not candidate_job.exists()
