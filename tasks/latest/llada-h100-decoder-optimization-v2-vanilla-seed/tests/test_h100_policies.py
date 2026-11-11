from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from conftest import private_data_dir


def load_contract(task_root: Path):
    path = task_root / "scripts/h100_policy_contract.py"
    spec = importlib.util.spec_from_file_location("h100_policy_contract_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def provisional_policy(task_root: Path):
    value = json.loads((task_root / "workload_policy.json").read_text())
    value["qualification_status"] = "QUALIFIED"
    value["release_status"] = "PROVISIONAL"
    value["interactive"].update(
        {
            "status": "QUALIFIED",
            "case_timeout_seconds": 600.0,
            "judge_timeout_seconds": 6_000.0,
            "cooldown_seconds": 60.0,
        }
    )
    value["final"].update(
        {
            "case_timeout_seconds": 600.0,
            "row_timeout_seconds": 3_000.0,
            "inner_timeout_seconds": 6_000.0,
            "outer_timeout_seconds": 7_000.0,
            "wrapper_margin_seconds": 1_000.0,
        }
    )
    value["qualification"].update(
        {
            "observations_sha256": "a" * 64,
            "approval_id": "workload-review",
            "max_observed_case_seconds": 100.0,
            "max_observed_row_seconds": 2_000.0,
            "max_observed_elapsed_seconds": 4_500.0,
        }
    )
    return value


def fullscale_identity(task_root: Path, run_id: str, archive_digit: str = "a"):
    workload = json.loads((task_root / "workload_policy.json").read_text())
    runtime_files = ["sforge/harness/runtime.py"]
    file_set_sha256 = hashlib.sha256(
        ("\n".join(runtime_files) + "\n").encode()
    ).hexdigest()
    core = {
        "verify_decoder_sha256": "1" * 64,
        "candidate_worker_sha256": "2" * 64,
        "cohort_sha256": "3" * 64,
        "quality_sha256": "4" * 64,
        "source_admission_sha256": "5" * 64,
    }
    return {
        "schema_version": 1,
        "task_id": workload["task_id"],
        "profile": workload["profile"],
        "generation_count": 520,
        "requested_slots": 216_320,
        "row_counts": workload["manifest"]["row_counts"],
        "run_id": run_id,
        "archive_sha256": archive_digit * 64,
        "hidden_cases_sha256": workload["manifest"]["hidden_cases_sha256"],
        "dataset_manifest_sha256": workload["manifest"]["dataset_manifest_sha256"],
        "workload_policy_sha256": hashlib.sha256(
            (task_root / "workload_policy.json").read_bytes()
        ).hexdigest(),
        "memory_policy_sha256": hashlib.sha256(
            (task_root / "memory_policy.json").read_bytes()
        ).hexdigest(),
        "quality_controls_sha256": hashlib.sha256(
            (task_root / "quality_controls.json").read_bytes()
        ).hexdigest(),
        "model_validation_sha256": "b" * 64,
        "runtime_entry_sha256": "c" * 64,
        "source_digest": "d" * 64,
        "validator_sha256": "0" * 64,
        "runtime_provenance_sha256": "6" * 64,
        "sforge_runtime_sha256": "7" * 64,
        "sforge_runtime_file_set_sha256": file_set_sha256,
        "sforge_runtime": {
            "scope": "all_sforge_python",
            "sha256": "7" * 64,
            "files": runtime_files,
            "file_count": len(runtime_files),
            "file_set_sha256": file_set_sha256,
        },
        "judge_invariant_core": core,
        **core,
        "archive_size_bytes": 1234,
        "work_image_id": "sha256:" + "e" * 64,
        "judge_image_id": "sha256:" + "f" * 64,
        "judge_gpu_uuid": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    }


def test_tracked_policies_validate_and_release_gating_matches_their_state(task_root):
    """The tracked policies validate, and release gating follows their status.

    A pending requalification must reject formal release. Frozen policies
    must retain complete qualification and pass the same release gate; the
    memory policy carries a capability envelope and labelled provenance.
    """
    module = load_contract(task_root)
    workload = json.loads((task_root / "workload_policy.json").read_text())
    memory = json.loads((task_root / "memory_policy.json").read_text())
    module.validate_workload_policy(workload)
    module.validate_memory_policy(memory)
    if workload["release_status"] == "PENDING":
        assert workload["qualification_status"] == "UNQUALIFIED"
        assert workload["quality_policy"]["status"] == "UNQUALIFIED"
        with pytest.raises(ValueError, match="not qualified and frozen"):
            module.validate_workload_policy(workload, require_frozen=True)
    else:
        assert workload["release_status"] == "FROZEN"
        module.validate_workload_policy(workload, require_frozen=True)
        assert workload["quality_policy"]["status"] == "QUALIFIED"
        provisional = dict(workload, release_status="PROVISIONAL")
        with pytest.raises(ValueError, match="not frozen"):
            module.validate_workload_policy(provisional, require_frozen=True)
    if memory["release_status"] == "PENDING":
        assert memory["qualification_status"] == "UNQUALIFIED"
        assert memory["qualified_max_peak_gib"] is None
        with pytest.raises(ValueError, match="not qualified and frozen"):
            module.validate_memory_policy(memory, require_frozen=True)
    else:
        assert memory["release_status"] == "FROZEN"
        module.validate_memory_policy(memory, require_frozen=True)
        assert memory["qualified_max_peak_gib"] == 82.0
    assert memory["min_compute_capability"] == module.MIN_COMPUTE_CAPABILITY
    assert memory["min_device_memory_mib"] == module.MIN_DEVICE_MEMORY_MIB
    assert memory["peak_cap_policy"].startswith("device_relative_v1:")
    # The qualification numbers are provenance about one device, not a gate.
    assert memory["qualification_hardware"] == "NVIDIA H100 NVL"
    assert memory["historical_a6000_policy"]["applicability"] == "NONE"
    assert module.QUALIFICATION_ROW_SENTINEL_GUARD_BYTES == 12 * 2**30
    assert memory["row_sentinel"]["qualification_guard_bytes"] == 12 * 2**30
    assert module.ROW_SENTINEL_REPEAT_GUARD_BYTES == 16 * 2**20
    assert memory["row_sentinel"]["repeat_growth_guard_bytes"] == 16 * 2**20
    assert (
        private_data_dir() / "memory_policy.json"
    ).read_bytes() == (task_root / "memory_policy.json").read_bytes()
    public = module.public_memory_contract(memory)
    assert "row_sentinel" not in public
    # The public contract publishes the envelope and a device-relative formula.
    assert public["min_compute_capability"] == module.MIN_COMPUTE_CAPABILITY
    assert public["min_device_memory_mib"] == module.MIN_DEVICE_MEMORY_MIB
    assert public["release_status"] == memory["release_status"]
    if memory["release_status"] == "PENDING":
        assert memory["headroom_mib"] is None
        assert memory["safety_margin_mib"] is None
        assert public["memory_reserve_mib"] is None
        assert public["qualified_max_peak_gib"] is None
    else:
        assert public["memory_reserve_mib"] == (
            memory["headroom_mib"] + memory["safety_margin_mib"]
        )
    assert "total_device_memory_mib" in public["peak_cap_formula"]
    assert "admission_mib" not in public
    interactive = workload["interactive"]
    assert interactive["category_definition"]["kind"] == "dataset"
    assert interactive["selection"] == "all_frozen_final_v1"
    assert interactive["total_cases"] == 520
    assert interactive["category_base_quota"] == 130
    assert interactive["bonus_categories"] == 0
    assert interactive["minimum_per_row"] == 1
    assert "per_category_quotas" not in interactive
    assert "per_row_quotas" not in interactive


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("category_definition", {"kind": "row", "field": "row_id"}),
        ("selection", "per_row_hmac_circular_window_v1"),
        ("total_cases", 49),
        ("category_base_quota", 11),
        ("bonus_categories", 1),
        ("minimum_per_row", 0),
        ("row_atomic", False),
        ("timeout_policy", "interactive_short"),
        ("per_row_quotas", {"table1_gsm8k_5shot_g256": 1}),
    ),
)
def test_interactive_policy_rejects_category_or_quota_drift(
    task_root, field, replacement
):
    module = load_contract(task_root)
    policy = json.loads((task_root / "workload_policy.json").read_text())
    policy["interactive"][field] = replacement
    with pytest.raises(ValueError, match="complete 520-case"):
        module.validate_workload_policy(policy)


def test_deadline_order_and_task_ceiling_are_enforced(task_root):
    module = load_contract(task_root)
    policy = provisional_policy(task_root)
    module.validate_workload_policy(policy)
    wrong = copy.deepcopy(policy)
    wrong["final"]["inner_timeout_seconds"] = 2_500.0
    with pytest.raises(ValueError, match="coherently nested"):
        module.validate_workload_policy(wrong)
    above_task = copy.deepcopy(policy)
    above_task["final"]["outer_timeout_seconds"] = 7_201.0
    with pytest.raises(ValueError, match="coherently nested"):
        module.validate_workload_policy(above_task)


def qualified_memory_policy(task_root):
    """Synthetic memory evidence, independent of tracked qualification state."""
    value = json.loads((task_root / "memory_policy.json").read_text())
    value.update(
        qualification_status="QUALIFIED",
        release_status="FROZEN",
        qualified_max_peak_gib=82.0,
        headroom_mib=6144,
        qualified_clean_idle_free_mib=95320,
        safety_margin_mib=1024,
        qualification_run_sha256="a" * 64,
        approval_id="synthetic-memory-review",
    )
    value["row_sentinel"].update(
        status="FROZEN",
        max_growth_bytes=64 * 2**20,
        observed_max_growth_bytes=32 * 2**20,
        observed_max_repeat_growth_bytes=0,
        safety_margin_bytes=16 * 2**20,
    )
    value["near_cap_stress"].update(
        status="QUALIFIED",
        minimum_cap_utilization=0.95,
        maximum_gap_mib=1024,
        observed_cap_utilization=0.97,
    )
    return value


@pytest.mark.parametrize("require_frozen", [False, True])
def test_qualified_memory_validation_is_independent_of_workload_deadlines(
    task_root, require_frozen
):
    module = load_contract(task_root)
    policy = qualified_memory_policy(task_root)
    assert "interactive" not in policy and "final" not in policy
    module.validate_memory_policy(policy, require_frozen=require_frozen)

    unsafe = copy.deepcopy(policy)
    unsafe["row_sentinel"]["observed_max_repeat_growth_bytes"] = (
        module.ROW_SENTINEL_REPEAT_GUARD_BYTES + 1
    )
    with pytest.raises(ValueError, match="growth bound is not evidence-backed"):
        module.validate_memory_policy(unsafe, require_frozen=require_frozen)


@pytest.mark.parametrize("field", ["case_timeout_seconds", "judge_timeout_seconds"])
def test_workload_still_rejects_different_complete_workload_deadlines(task_root, field):
    module = load_contract(task_root)
    policy = provisional_policy(task_root)
    policy["interactive"][field] -= 1
    with pytest.raises(ValueError, match="complete-workload deadlines must match"):
        module.validate_workload_policy(policy)


def memory_report(
    task_root,
    kind: str,
    *,
    allocated_mib: int,
    reserved_mib: int,
    used_mib: int,
    clean_mib: int = 95_300,
    row_growth_mib: int = 32,
    row_repeat_growth_mib: int = 0,
    hardware: str = "NVIDIA H100 NVL",
):
    mib = 2**20
    return {
        **fullscale_identity(
            task_root,
            f"memory-{kind}",
            "1" if kind == "corrected_reference" else "2",
        ),
        "status": "PASS",
        "mode": "qualification",
        "release_evidence": False,
        "reference_replay": False,
        "hardware": hardware,
        "visible_device_count": 1,
        "fixture_kind": kind,
        "torch_peak_allocated_bytes": allocated_mib * mib,
        "torch_peak_reserved_bytes": reserved_mib * mib,
        "nvml_peak_device_used_bytes": used_mib * mib,
        "nvml_total_bytes": 95_830 * mib,
        "nvml_clean_idle_free_bytes": clean_mib * mib,
        "torch_device_total_bytes": 95_319 * mib,
        "completed_without_oom": True,
        "row_sentinel_observed_max_growth_bytes": row_growth_mib * mib,
        "row_sentinel_observed_max_repeat_growth_bytes": (
            row_repeat_growth_mib * mib
        ),
        "row_sentinel_all_stable": True,
        "row_sentinel_repeat_all_stable": True,
    }


def run_memory_qualifier(
    task_root,
    tmp_path,
    *,
    headroom=4096,
    clean_mib=95_300,
    low_load=False,
    stress_snapshot_sha256=None,
    stress_approval_id=None,
    hardware="NVIDIA H100 NVL",
    stress_hardware=None,
):
    reference = tmp_path / "reference.json"
    stress = tmp_path / "stress.json"
    fixtures = tmp_path / "memory-fixtures.json"
    output = tmp_path / f"qualification-{headroom}-{clean_mib}.json"
    fixtures.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_id": "llada_h100_decoder_optimization_v2",
                "profile": "fast_dllm_v1_llada_12row_stratified520_batch1",
                "status": "PRECOMMITTED",
                "workload_policy_sha256": hashlib.sha256(
                    (task_root / "workload_policy.json").read_bytes()
                ).hexdigest(),
                "memory_policy_sha256": hashlib.sha256(
                    (task_root / "memory_policy.json").read_bytes()
                ).hexdigest(),
                "fixtures": {
                    "corrected_reference": {
                        "kind": "corrected_reference",
                        "archive_sha256": "1" * 64,
                        "archive_size_bytes": 1234,
                        "stress_extra_mib": None,
                    },
                    "near_cap_stress": {
                        "kind": "near_cap_stress",
                        "archive_sha256": "2" * 64,
                        "archive_size_bytes": 1234,
                        "stress_extra_mib": 4096,
                    },
                },
                "approval_id": "memory-fixtures-review",
            }
        )
    )
    fixture_sha256 = hashlib.sha256(fixtures.read_bytes()).hexdigest()
    approval_id = "memory-fixtures-review"
    reference_payload = memory_report(
        task_root,
        "corrected_reference",
        allocated_mib=(60 if low_load else 80) * 1024,
        reserved_mib=(61 if low_load else 82) * 1024,
        used_mib=(62 if low_load else 83) * 1024,
        clean_mib=clean_mib,
        hardware=hardware,
    )
    stress_payload = memory_report(
        task_root,
        "near_cap_stress",
        allocated_mib=(70 * 1024 if low_load else 80 * 1024 + 512),
        reserved_mib=(71 if low_load else 83) * 1024,
        used_mib=(72 if low_load else 84) * 1024,
        clean_mib=clean_mib,
        hardware=stress_hardware or hardware,
    )
    for kind, payload in (
        ("corrected_reference", reference_payload),
        ("near_cap_stress", stress_payload),
    ):
        manifest_sha = (
            stress_snapshot_sha256
            if kind == "near_cap_stress" and stress_snapshot_sha256 is not None
            else fixture_sha256
        )
        report_approval = (
            stress_approval_id
            if kind == "near_cap_stress" and stress_approval_id is not None
            else approval_id
        )
        payload.update(
            {
                "memory_fixture_manifest_sha256": manifest_sha,
                "memory_fixture_approval_id": report_approval,
                "memory_fixture": {
                    "schema_version": 1,
                    "manifest_sha256": manifest_sha,
                    "approval_id": report_approval,
                    "kind": kind,
                    "archive_sha256": payload["archive_sha256"],
                    "archive_size_bytes": payload["archive_size_bytes"],
                },
            }
        )
    reference.write_text(json.dumps(reference_payload))
    stress.write_text(json.dumps(stress_payload))
    completed = subprocess.run(
        [
            sys.executable,
            str(task_root / "scripts/qualify_h100_memory.py"),
            "--reference-report", str(reference),
            "--stress-report", str(stress),
            "--fixture-manifest", str(fixtures),
            "--proposed-max-peak-gib", "81",
            "--proposed-headroom-mib", str(headroom),
            "--safety-margin-mib", "512",
            "--proposed-row-sentinel-max-growth-mib", "64",
            "--row-sentinel-safety-margin-mib", "16",
            "--minimum-stress-cap-utilization", "0.95",
            "--max-stress-gap-mib", "1024",
            "--output", str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed, output


def test_memory_cap_bounds_allocated_while_headroom_covers_nvml_peak(task_root, tmp_path):
    completed, output = run_memory_qualifier(task_root, tmp_path)
    assert completed.returncode == 0, completed.stderr
    qualified = json.loads(output.read_text())
    assert qualified["proposed_max_peak_gib"] == 81
    assert qualified["max_measured_reserved_or_nvml_bytes"] == 84 * 1024 * 2**20
    assert qualified["proposed_headroom_mib"] == 4096
    assert qualified["row_sentinel_observed_max_repeat_growth_bytes"] == 0
    assert qualified["row_sentinel_repeat_growth_guard_bytes"] == 16 * 2**20
    # The device is recorded from the evidence, not asserted as a model name.
    assert qualified["hardware"] == "NVIDIA H100 NVL"


def test_memory_qualification_records_whatever_device_produced_the_evidence(
    task_root, tmp_path
):
    """A qualification cycle on any admitted device is accepted and recorded."""
    completed, output = run_memory_qualifier(
        task_root, tmp_path, hardware="NVIDIA A100-SXM4-80GB"
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_text())["hardware"] == "NVIDIA A100-SXM4-80GB"


def test_memory_qualification_rejects_evidence_from_two_devices(task_root, tmp_path):
    """Reference and near-cap stress must come from one device."""
    completed, _ = run_memory_qualifier(
        task_root,
        tmp_path,
        hardware="NVIDIA H100 NVL",
        stress_hardware="NVIDIA A100-SXM4-80GB",
    )
    assert completed.returncode != 0
    assert "different devices" in completed.stderr


def test_memory_qualification_rejects_insufficient_headroom_and_unattainable_floor(task_root, tmp_path):
    insufficient, _ = run_memory_qualifier(task_root, tmp_path, headroom=1024)
    assert insufficient.returncode != 0
    assert "headroom" in insufficient.stderr
    impossible, _ = run_memory_qualifier(task_root, tmp_path, clean_mib=85_000)
    assert impossible.returncode != 0
    assert "clean idle" in impossible.stderr


def test_memory_qualification_rejects_low_load_labeled_as_near_cap(task_root, tmp_path):
    completed, _ = run_memory_qualifier(task_root, tmp_path, low_load=True)
    assert completed.returncode != 0
    assert "not sufficiently close" in completed.stderr


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    (
        ({"stress_snapshot_sha256": "f" * 64}, "precommitted archive"),
        ({"stress_approval_id": "different-review"}, "precommitted archive"),
    ),
)
def test_memory_qualification_rejects_fixture_snapshot_or_approval_drift(
    task_root, tmp_path, kwargs, expected
):
    completed, _ = run_memory_qualifier(task_root, tmp_path, **kwargs)
    assert completed.returncode != 0
    assert expected in completed.stderr



def _rewind_policies_to_bootstrap(root) -> None:
    """Rewrite copied policies into the UNQUALIFIED/PENDING bootstrap state."""
    memory_path = root / "memory_policy.json"
    memory = json.loads(memory_path.read_text())
    memory.update(
        {
            "qualification_status": "UNQUALIFIED",
            "release_status": "PENDING",
            "qualified_max_peak_gib": None,
            "headroom_mib": None,
            "qualified_clean_idle_free_mib": None,
            "safety_margin_mib": None,
            "qualification_run_sha256": None,
            "approval_id": None,
        }
    )
    memory["row_sentinel"].update(
        {
            "status": "PENDING",
            "max_growth_bytes": None,
            "observed_max_growth_bytes": None,
            "observed_max_repeat_growth_bytes": None,
            "safety_margin_bytes": None,
        }
    )
    memory["near_cap_stress"] = {
        "status": "PENDING",
        "minimum_cap_utilization": None,
        "maximum_gap_mib": None,
        "observed_cap_utilization": None,
    }
    memory_path.write_text(json.dumps(memory, sort_keys=True, indent=2) + "\n")


def test_task_owned_memory_stress_archive_is_deterministic_and_precommitted(
    task_root, tmp_path
):
    for relative in (
        "images/work/workdir/decoder_lab/solutions/decoder_v001/candidate.py",
        "host_reference/corrected_decoder/candidate.py",
        "workload_policy.json",
        "memory_policy.json",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((task_root / relative).read_bytes())
    # Fixtures are precommitted before a qualification cycle, so rewind the
    # copied policies to the bootstrap state that step is defined against.
    _rewind_policies_to_bootstrap(tmp_path)
    command = [
        sys.executable,
        str(task_root / "scripts/make_reference_archives.py"),
        "--task-root", str(tmp_path),
        "--stress-extra-mib", "4096",
    ]
    first = subprocess.run(command, capture_output=True, text=True, check=False)
    assert first.returncode == 0, first.stderr
    stress = tmp_path / "artifacts/near-cap-stress-reference.tar.gz"
    assert stress.read_bytes()[9] == 3
    first_sha = hashlib.sha256(stress.read_bytes()).hexdigest()
    second = subprocess.run(command, capture_output=True, text=True, check=False)
    assert second.returncode == 0, second.stderr
    assert hashlib.sha256(stress.read_bytes()).hexdigest() == first_sha
    with tarfile.open(stress, "r:gz") as bundle:
        source = bundle.extractfile("solutions/decoder_v001/candidate.py").read()
    assert b"STRESS_EXTRA_MIB = 4096" in source
    assert b"stress.fill_(1)" in source
    assert b"self.inner.generate" in source

    fixture_manifest = tmp_path / "memory_qualification_fixtures.json"
    precommit = subprocess.run(
        [
            sys.executable,
            str(task_root / "scripts/precommit_h100_memory_fixtures.py"),
            "--task-root", str(tmp_path),
            "--reference-archive",
            str(tmp_path / "artifacts/corrected-decoder-reference.tar.gz"),
            "--stress-archive", str(stress),
            "--stress-extra-mib", "4096",
            "--approval-id", "memory-fixture-review",
            "--output", str(fixture_manifest),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert precommit.returncode == 0, precommit.stderr
    manifest = json.loads(fixture_manifest.read_text())
    assert manifest["status"] == "PRECOMMITTED"
    assert manifest["fixtures"]["near_cap_stress"]["stress_extra_mib"] == 4096


@pytest.mark.parametrize(
    ("field", "value"),
    (("mode", "final"), ("release_evidence", True), ("reference_replay", True)),
)
def test_workload_qualifier_rejects_wrong_evidence_class(
    task_root, tmp_path, field, value
):
    policy = json.loads((task_root / "workload_policy.json").read_text())
    reports = []
    for index in range(3):
        payload = {
            **fullscale_identity(task_root, f"qualification-{index}", str(index + 3)),
            "status": "PASS",
            "mode": "qualification",
            "release_evidence": False,
            "reference_replay": False,
            "hardware": "NVIDIA H100 NVL",
            "visible_device_count": 1,
            "elapsed_seconds": 1000.0 + index,
            "row_seconds": {
                row_id: 10.0 + index for row_id in policy["manifest"]["row_counts"]
            },
        }
        if index == 0:
            payload[field] = value
        path = tmp_path / f"timing-{index}.json"
        path.write_text(json.dumps(payload))
        reports.extend(("--report", str(path)))
    completed = subprocess.run(
        [
            sys.executable,
            str(task_root / "scripts/qualify_h100_workload.py"),
            *reports,
            "--output",
            str(tmp_path / "timing-qualification.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "not bound qualification evidence" in completed.stderr


def test_quality_control_precommit_binds_exact_archives_and_parent_policy(task_root, tmp_path):
    policy = tmp_path / "workload_policy.json"
    policy.write_text(json.dumps(provisional_policy(task_root), sort_keys=True, indent=2) + "\n")
    positive = tmp_path / "positive.tar.gz"
    negative = tmp_path / "negative.tar.gz"
    calibration = tmp_path / "calibration.tar.gz"
    positive.write_bytes(b"positive-control")
    negative.write_bytes(b"negative-control")
    calibration.write_bytes(b"calibration-producer")
    output = tmp_path / "quality_controls.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(task_root / "scripts/precommit_h100_quality_controls.py"),
            "--task-root", str(tmp_path),
            "--policy", str(policy),
            "--positive", f"corrected={positive}",
            "--negative", f"degraded={negative}",
            "--calibration-archive", str(calibration),
            "--approval-id", "controls-review",
            "--output", str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    manifest = json.loads(output.read_text())
    assert manifest["status"] == "PRECOMMITTED"
    assert manifest["workload_policy_sha256"] == __import__("hashlib").sha256(policy.read_bytes()).hexdigest()
    assert manifest["positive_controls"][0]["expected_classification"] == "ACCEPT"
    assert manifest["negative_controls"][0]["expected_classification"] == "REJECT"
    assert manifest["producer_calibration"]["archive_sha256"] == hashlib.sha256(
        calibration.read_bytes()
    ).hexdigest()


def test_control_qualifier_to_finalizer_preserves_provisional_parent_chain(task_root, tmp_path):
    policy_path = tmp_path / "provisional.json"
    policy_path.write_text(
        json.dumps(provisional_policy(task_root), sort_keys=True, indent=2) + "\n"
    )
    parent_sha = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    policy = json.loads(policy_path.read_text())
    hidden_sha = policy["manifest"]["hidden_cases_sha256"]
    runtime_source = fullscale_identity(task_root, "runtime-template")
    runtime_fields = {
        field: runtime_source[field]
        for field in (
            "runtime_provenance_sha256",
            "sforge_runtime_sha256",
            "sforge_runtime_file_set_sha256",
            "verify_decoder_sha256",
            "candidate_worker_sha256",
            "cohort_sha256",
            "quality_sha256",
            "source_admission_sha256",
            "sforge_runtime",
            "judge_invariant_core",
        )
    }
    control_manifest_path = tmp_path / "quality_controls.json"
    control_manifest = {
        "schema_version": 1,
        "status": "PRECOMMITTED",
        "task_id": policy["task_id"],
        "profile": policy["profile"],
        "workload_policy_sha256": parent_sha,
        "hidden_cases_sha256": hidden_sha,
        "methodology": {
            "formula": "producer_relative_min_retention_ppm_v1",
            "metric": "producer_relative_min_retention_ppm_v1",
            "margin_selection": "producer_derived_after_precommitted_controls",
            "threshold_derivation": "midpoint_min_positive_max_negative_v1",
            "minimum_separation_ppm": 10_000,
            "control_measurement_floor_ppm": 1,
        },
        "producer_calibration": {
            "archive": "producer.tar.gz",
            "archive_sha256": "a" * 64,
            "expected_mode": "calibration",
        },
        "approval_id": "precommit",
        "historical_a6000_controls_applicability": "NONE",
        "positive_controls": [
            {
                "id": "positive",
                "archive": "positive.tar.gz",
                "archive_sha256": "6" * 64,
                "expected_classification": "ACCEPT",
                "acceptance_target": {},
            }
        ],
        "negative_controls": [
            {
                "id": "negative",
                "archive": "negative.tar.gz",
                "archive_sha256": "7" * 64,
                "expected_classification": "REJECT",
                "acceptance_target": {},
            }
        ],
    }
    control_manifest_path.write_text(json.dumps(control_manifest))
    controls_sha = hashlib.sha256(control_manifest_path.read_bytes()).hexdigest()
    producer = tmp_path / "producer.json"
    producer_payload = {
        "schema_version": 1,
        "status": "PASS",
        "mode": "calibration",
        "release_evidence": False,
        "reference_replay": False,
        "task_id": policy["task_id"],
        "hardware": "NVIDIA H100 NVL",
        "visible_device_count": 1,
        "profile": policy["profile"],
        "generation_count": 520,
        "requested_slots": 216_320,
        "row_counts": policy["manifest"]["row_counts"],
        "row_seconds": {
            row_id: 2_500.0 for row_id in policy["manifest"]["row_counts"]
        },
        "max_case_seconds": 500.0,
        "elapsed_seconds": 5_000.0,
        "archive_sha256": "a" * 64,
        "archive_size_bytes": 1_234,
        "hidden_cases_sha256": hidden_sha,
        "dataset_manifest_sha256": policy["manifest"]["dataset_manifest_sha256"],
        "workload_policy_sha256": parent_sha,
        "memory_policy_sha256": "2" * 64,
        "quality_controls_sha256": controls_sha,
        "model_validation_sha256": "3" * 64,
        "runtime_entry_sha256": "4" * 64,
        "source_digest": "a" * 64,
        "validator_sha256": "b" * 64,
        "work_image_id": "sha256:" + "c" * 64,
        "judge_image_id": "sha256:" + "d" * 64,
        "judge_gpu_uuid": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        **runtime_fields,
    }
    producer.write_text(json.dumps(producer_payload))
    reference = tmp_path / "reference_quality.json"
    reference.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "calibrated": True,
                "status": "CALIBRATED_STRATIFIED520",
                "calibration_provenance": {
                    "evaluation_attestation_sha256": hashlib.sha256(
                        producer.read_bytes()
                    ).hexdigest(),
                    "quality_controls_sha256": controls_sha,
                    "workload_policy_sha256": parent_sha,
                    "runtime_provenance_sha256": runtime_fields[
                        "runtime_provenance_sha256"
                    ],
                    **{
                        field: runtime_fields[field]
                        for field in (
                            "sforge_runtime_sha256",
                            "sforge_runtime_file_set_sha256",
                            "verify_decoder_sha256",
                            "candidate_worker_sha256",
                            "cohort_sha256",
                            "quality_sha256",
                            "source_admission_sha256",
                        )
                    },
                    "sforge_runtime": runtime_fields["sforge_runtime"],
                    "judge_invariant_core": runtime_fields[
                        "judge_invariant_core"
                    ],
                },
            }
        )
    )
    reference_sha = hashlib.sha256(reference.read_bytes()).hexdigest()

    def measurement(score):
        base = {
            "schema_version": 1,
            "metric": "producer_relative_min_retention_ppm_v1",
            "scale_ppm": 1_000_000,
            "score_ppm": score,
            "components_ppm": {"aggregate:correct": score},
        }
        base["measurement_sha256"] = hashlib.sha256(
            (json.dumps(base, sort_keys=True, separators=(",", ":")) + "\n").encode()
        ).hexdigest()
        return base

    reports = []
    for control_id, expected, archive_sha, quality_score in (
        ("positive", "ACCEPT", "6" * 64, 900_000),
        ("negative", "REJECT", "7" * 64, 100_000),
    ):
        path = tmp_path / f"{control_id}-report.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "PASS",
                    "mode": "quality_control",
                    "release_evidence": False,
                    "reference_replay": False,
                    "task_id": policy["task_id"],
                    "profile": policy["profile"],
                    "hardware": "NVIDIA H100 NVL",
                    "visible_device_count": 1,
                    "generation_count": 520,
                    "requested_slots": 216_320,
                    "row_counts": policy["manifest"]["row_counts"],
                    "control_id": control_id,
                    "expected_classification": expected,
                    "observed_classification": expected,
                    "archive_sha256": archive_sha,
                    "archive_size_bytes": 2_345,
                    "hidden_cases_sha256": hidden_sha,
                    "dataset_manifest_sha256": policy["manifest"]["dataset_manifest_sha256"],
                    "workload_policy_sha256": parent_sha,
                    "memory_policy_sha256": "2" * 64,
                    "quality_controls_sha256": controls_sha,
                    "model_validation_sha256": "3" * 64,
                    "runtime_entry_sha256": "4" * 64,
                    "source_digest": "5" * 64,
                    "validator_sha256": "b" * 64,
                    "reference_manifest_sha256": reference_sha,
                    "quality_measurement": measurement(quality_score),
                    "quality_score_ppm": quality_score,
                    "work_image_id": "sha256:" + "8" * 64,
                    "judge_image_id": "sha256:" + "9" * 64,
                    "judge_gpu_uuid": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                    **runtime_fields,
                }
            )
        )
        reports.append(path)
    controls_attestation = tmp_path / "controls-attestation.json"
    qualify = subprocess.run(
        [
            sys.executable,
            str(task_root / "scripts/qualify_h100_quality_controls.py"),
            "--manifest", str(control_manifest_path),
            "--report", str(reports[0]),
            "--report", str(reports[1]),
            "--output", str(controls_attestation),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert qualify.returncode == 0, qualify.stderr
    frozen_path = tmp_path / "frozen.json"
    finalize = subprocess.run(
        [
            sys.executable,
            str(task_root / "scripts/finalize_h100_release_policy.py"),
            "--policy", str(policy_path),
            "--producer-attestation", str(producer),
            "--quality-control-attestation", str(controls_attestation),
            "--quality-controls", str(control_manifest_path),
            "--reference-quality", str(reference),
            "--approval-id", "release-review",
            "--output", str(frozen_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert finalize.returncode == 0, finalize.stderr
    frozen = json.loads(frozen_path.read_text())
    assert frozen["release_status"] == "FROZEN"
    assert frozen["provisional_parent_sha256"] == parent_sha
    assert frozen["quality_policy"]["provisional_parent_sha256"] == parent_sha
    assert frozen["quality_policy"]["thresholds"]["minimum_score_ppm"] == 500_000
    load_contract(task_root).validate_workload_policy(frozen, require_frozen=True)

    slow_producer = json.loads(producer.read_text())
    slow_producer["elapsed_seconds"] = 5_100.0
    producer.write_text(json.dumps(slow_producer))
    deadline_reject = subprocess.run(
        [
            sys.executable,
            str(task_root / "scripts/finalize_h100_release_policy.py"),
            "--policy", str(policy_path),
            "--producer-attestation", str(producer),
            "--quality-control-attestation", str(controls_attestation),
            "--quality-controls", str(control_manifest_path),
            "--reference-quality", str(reference),
            "--approval-id", "release-review",
            "--output", str(tmp_path / "deadline-reject.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert deadline_reject.returncode != 0
    assert "deadlines do not cover" in deadline_reject.stderr
    producer.write_text(json.dumps(producer_payload))

    close_negative = json.loads(reports[1].read_text())
    close_negative["quality_measurement"] = measurement(895_000)
    close_negative["quality_score_ppm"] = 895_000
    reports[1].write_text(json.dumps(close_negative))
    insufficient_margin = subprocess.run(
        [
            sys.executable,
            str(task_root / "scripts/qualify_h100_quality_controls.py"),
            "--manifest", str(control_manifest_path),
            "--report", str(reports[0]),
            "--report", str(reports[1]),
            "--output", str(tmp_path / "insufficient-margin.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert insufficient_margin.returncode != 0
    assert "score separation" in insufficient_margin.stderr
    close_negative["quality_measurement"] = measurement(100_000)
    close_negative["quality_score_ppm"] = 100_000
    reports[1].write_text(json.dumps(close_negative))

    runtime_drift = json.loads(reports[0].read_text())
    runtime_drift["judge_invariant_core"]["quality_sha256"] = "e" * 64
    reports[0].write_text(json.dumps(runtime_drift))
    drift_qualify = subprocess.run(
        [
            sys.executable,
            str(task_root / "scripts/qualify_h100_quality_controls.py"),
            "--manifest", str(control_manifest_path),
            "--report", str(reports[0]),
            "--report", str(reports[1]),
            "--output", str(tmp_path / "runtime-drift-controls.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert drift_qualify.returncode != 0
    assert "runtime provenance" in drift_qualify.stderr
    runtime_drift["judge_invariant_core"]["quality_sha256"] = runtime_fields[
        "quality_sha256"
    ]
    reports[0].write_text(json.dumps(runtime_drift))

    controls_drift = json.loads(controls_attestation.read_text())
    controls_drift["sforge_runtime"]["files"] = ["sforge/drift.py"]
    drift_attestation = tmp_path / "controls-drift.json"
    drift_attestation.write_text(json.dumps(controls_drift))
    drift_finalize = subprocess.run(
        [
            sys.executable,
            str(task_root / "scripts/finalize_h100_release_policy.py"),
            "--policy", str(policy_path),
            "--producer-attestation", str(producer),
            "--quality-control-attestation", str(drift_attestation),
            "--quality-controls", str(control_manifest_path),
            "--reference-quality", str(reference),
            "--approval-id", "release-review",
            "--output", str(tmp_path / "runtime-drift-frozen.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert drift_finalize.returncode != 0
    assert "SForge/Judge provenance" in drift_finalize.stderr

    missing_producer_runtime = json.loads(producer.read_text())
    missing_producer_runtime.pop("sforge_runtime")
    producer.write_text(json.dumps(missing_producer_runtime))
    missing_finalize = subprocess.run(
        [
            sys.executable,
            str(task_root / "scripts/finalize_h100_release_policy.py"),
            "--policy", str(policy_path),
            "--producer-attestation", str(producer),
            "--quality-control-attestation", str(controls_attestation),
            "--quality-controls", str(control_manifest_path),
            "--reference-quality", str(reference),
            "--approval-id", "release-review",
            "--output", str(tmp_path / "missing-runtime-frozen.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing_finalize.returncode != 0
    assert "complete SForge/Judge provenance" in missing_finalize.stderr
    producer.write_text(json.dumps(producer_payload))

    wrong_control = json.loads(reports[0].read_text())
    wrong_control["mode"] = "final"
    reports[0].write_text(json.dumps(wrong_control))
    wrong_qualify = subprocess.run(
        [
            sys.executable,
            str(task_root / "scripts/qualify_h100_quality_controls.py"),
            "--manifest", str(control_manifest_path),
            "--report", str(reports[0]),
            "--report", str(reports[1]),
            "--output", str(tmp_path / "wrong-controls.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert wrong_qualify.returncode != 0
    assert "full identity-bound evidence" in wrong_qualify.stderr

    wrong_control["mode"] = "quality_control"
    wrong_control["expected_classification"] = "REJECT"
    reports[0].write_text(json.dumps(wrong_control))
    mismatch_qualify = subprocess.run(
        [
            sys.executable,
            str(task_root / "scripts/qualify_h100_quality_controls.py"),
            "--manifest", str(control_manifest_path),
            "--report", str(reports[0]),
            "--report", str(reports[1]),
            "--output", str(tmp_path / "mismatch-controls.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert mismatch_qualify.returncode != 0

    wrong_producer = json.loads(producer.read_text())
    wrong_producer["mode"] = "final"
    producer.write_text(json.dumps(wrong_producer))
    wrong_finalize = subprocess.run(
        [
            sys.executable,
            str(task_root / "scripts/finalize_h100_release_policy.py"),
            "--policy", str(policy_path),
            "--producer-attestation", str(producer),
            "--quality-control-attestation", str(controls_attestation),
            "--quality-controls", str(control_manifest_path),
            "--reference-quality", str(reference),
            "--approval-id", "release-review",
            "--output", str(tmp_path / "wrong-frozen.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert wrong_finalize.returncode != 0
    assert "producer is not accepted" in wrong_finalize.stderr
