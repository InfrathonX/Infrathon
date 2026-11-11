from __future__ import annotations

import json
import subprocess
import sys

from conftest import private_data_dir


def test_calibration_is_fail_closed_on_missing_producer_inputs(task_root, tmp_path):
    script = task_root / "scripts/calibrate_reference.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            str(tmp_path / "missing-submission"),
            "--producer-image-manifest", str(tmp_path / "missing-image.json"),
            "--candidate-source", str(tmp_path / "missing-candidate.py"),
            "--model-validation", str(tmp_path / "missing-model.json"),
            "--evaluation-attestation", str(tmp_path / "missing-attestation.json"),
            "--output", str(tmp_path / "reference.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    # Whichever gate trips first, nothing is written and nothing is inferred.
    assert completed.stderr.strip() in {
        "producer calibration requires provisional qualified workload policy",
        "trusted submission root is missing or unsafe",
    }
    assert not (tmp_path / "reference.json").exists()
    assert not (tmp_path / "reference_cache").exists()


def test_calibrator_emits_stratified520_identity_and_parent_policy_provenance(task_root):
    source = (task_root / "scripts/calibrate_reference.py").read_text()
    for literal in (
        '"schema_version": 4',
        '"status": "CALIBRATED_STRATIFIED520"',
        # The device is taken from the producer report, never asserted.
        '"hardware": hardware_name',
        '"measurement_profile": measurement_profile',
        '"measurement_profile_key": profile_key(measurement_profile)',
        '"device_profile_key": profile_key(device_profile)',
        '"generation_count": GENERATION_COUNT',
        '"requested_slots": REQUESTED_SLOTS',
        '"workload_policy_sha256": sha256(workload_policy_path)',
        '"memory_policy_sha256": sha256(memory_policy_path)',
        '"canonical_candidate_archive_sha256"',
        '"evaluation_attestation_sha256"',
        '"runtime_provenance_sha256"',
        '"sforge_runtime": runtime_provenance["sforge_runtime"]',
        '"judge_invariant_core": runtime_provenance["judge_invariant_core"]',
    ):
        assert literal in source
    assert '"NVIDIA H100 NVL"' not in source
    # The producer files the reference under its own device profile so a later
    # run on that profile reuses it instead of re-measuring.
    assert 'REFERENCE_CACHE_DIRNAME = "reference_cache"' in source
    assert 'key: measurement_profile[key] for key in ("compute_capability", "gpu_name")' in source
    assert 'cache_dir / f"{profile_key(reference_profile)}.json"' in source
    shipped = json.loads(
        (private_data_dir() / "reference_quality.json").read_text()
    )
    if shipped["status"] == "PENDING_STRATIFIED520_CALIBRATION":
        assert shipped["calibrated"] is False
        assert shipped["calibration_provenance"] is None
        assert shipped["rows"] == {}
        assert shipped["case_records"] == {}
        for key in ("measurement_profile", "measurement_profile_key", "device_profile_key"):
            assert key not in shipped
    else:
        assert shipped["calibrated"] is True
        assert shipped["status"] == "CALIBRATED_STRATIFIED520"
        assert shipped["calibration_provenance"] is not None


def test_profile_selection_step_is_fail_closed_and_never_reuses_another_device(
    task_root, tmp_path
):
    """The launcher-side selection refuses to run without a matching profile."""
    source = (task_root / "scripts/select_reference_profile.py").read_text()
    assert "--check" in source
    assert "no quality reference is calibrated for device profile" in source
    assert "the built reference does not match device profile" in source
    # The launcher checks the profile before Work starts, and the build step
    # refuses a Judge GPU without a reference before the image is built.
    runner = (task_root / "scripts/run_fixed_budget.sh").read_text()
    assert "select_reference_profile.py" in runner and "--check" in runner
    build = (task_root / "scripts/build_images.sh").read_text()
    assert "select_reference_profile.py" in build
    assert build.index("select_reference_profile.py") < build.index("docker build")
