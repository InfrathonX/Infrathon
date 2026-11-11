from __future__ import annotations
import hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
GATES={"profile_ok","hashes_ok","admission_ok","seed_reference_ok","worker_ok","completeness_ok","parity_ok","liveness_ok","resources_ok","ttff_ok","p99_ok","construction_ok"}
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def test_judge_port_is_byte_identical(): assert sha(ROOT/"judge/run_judge.py")==sha(ROOT/"images/judge/run_judge.py")
def test_verifier_is_fail_closed_and_wired():
    text=(ROOT/"images/judge/verify_sdv2_pipeline.py").read_text()
    assert all(gate in text for gate in GATES)
    assert '"failure_detail": None' in text and '"valid": False' in text and '"score": 0.0' in text
    assert "compare_saved_ledger" in text and "calibration_v1.json" not in text
    assert "auto-" in text and "candidate-" in text
def test_seed_reference_pass_is_the_only_baseline_source():
    text=(ROOT/"images/judge/verify_sdv2_pipeline.py").read_text()
    # The pristine seed is re-measured on the evaluation devices in every run;
    # v2 ships no frozen device table at all.
    assert "SEED_RUNTIME" in text and "_seed_reference(" in text and "digest_saved_ledger" in text
    assert 'calibration["cases"]' not in text and "resource_caps" not in text
    assert "qualification_provenance_only" not in text
    assert "peak_memory_mib_per_gpu" not in text and "--query-gpu=index,uuid,memory.used" in text
def test_process_model_is_explicit():
    worker=(ROOT/"images/judge/candidate_worker.py").read_text(); verifier=(ROOT/"images/judge/verify_sdv2_pipeline.py").read_text()
    assert "sys.stdin.buffer.read" in worker and "os.replace" in worker
    assert '"CUDA_VISIBLE_DEVICES": "0,1"' in verifier
    assert "start_new_session=True" in verifier and "os.killpg" in verifier
    assert "leftover_session_pids" in verifier and "active_gpu_pids" in verifier
def test_controller_owns_the_push_stamps_and_the_plan_has_no_frames():
    worker=(ROOT/"images/judge/candidate_worker.py").read_text()
    sink=(ROOT/"runtime/sdv2_core/sink.py").read_text()
    assert "plan.meta()" in worker and "pushed_ns[chunk.index] = time.monotonic_ns()" in worker
    assert '"push_timestamps.json"' in worker
    assert "def mark_push" not in sink and "push_timestamps" not in sink
    plan=(ROOT/"runtime/sdv2_core/plan.py").read_text()
    meta=plan[plan.index("class StreamPlanMeta"):plan.index("class StreamPlan:")]
    assert "frames" not in meta.split('"""')[2], "StreamPlanMeta must not carry the input video"
