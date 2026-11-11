from __future__ import annotations

import pytest

import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

from conftest import private_data_dir


RUN_ID = "unit-qualification"
TASK_ID = "llada_h100_decoder_optimization_v2"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def runtime_snapshot(task_root: Path, run_root: Path) -> tuple[Path, dict]:
    repo = task_root.parents[2]
    paths = sorted(
        (path.resolve() for path in (repo / "sforge").rglob("*.py")),
        key=lambda path: str(path),
    )
    files = [path.relative_to(repo).as_posix() for path in paths]
    sforge = {
        "scope": "all_sforge_python",
        "sha256": hashlib.sha256(
            "".join(
                f"{digest(path)}  {name}\n"
                for path, name in zip(paths, files, strict=True)
            ).encode()
        ).hexdigest(),
        "files": files,
        "file_count": len(files),
        "file_set_sha256": hashlib.sha256(
            ("\n".join(files) + "\n").encode()
        ).hexdigest(),
    }
    core = {
        name: digest(task_root / relative)
        for name, relative in {
            "verify_decoder_sha256": "images/judge/verifier/verify_decoder.py",
            "candidate_worker_sha256": "images/judge/verifier/candidate_worker.py",
            "cohort_sha256": "images/judge/verifier/cohort.py",
            "quality_sha256": "images/judge/verifier/quality.py",
            "source_admission_sha256": "images/judge/verifier/source_admission.py",
        }.items()
    }
    value = {
        "schema_version": 1,
        "sforge_runtime": sforge,
        "judge_invariant_core": core,
    }
    path = run_root / "runtime-provenance.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    path.chmod(0o444)
    return path, value


def build_fixture(task_root: Path, tmp_path: Path, *, memory_fixture: bool = False):
    workload = json.loads((task_root / "workload_policy.json").read_text(encoding="utf-8"))
    if workload.get("qualification_status") != "UNQUALIFIED":
        # The validator accepts qualification-mode reports only while the
        # tracked policies are in their initial pending state.
        pytest.skip("qualification-mode fixtures need the initial pending policies")
    script = task_root / "scripts/validate_eval_artifacts.py"
    submission = tmp_path / RUN_ID / TASK_ID / "submissions/manual-1"
    submission.mkdir(parents=True)
    input_archive = tmp_path / "input.tar.gz"
    input_archive.write_bytes(b"fixed qualification archive bytes")
    (submission / "submission.tar.gz").write_bytes(input_archive.read_bytes())

    hidden_bytes = gzip.decompress(
        (private_data_dir() / "hidden_cases.json.gz").read_bytes()
    )
    hidden = json.loads(hidden_bytes)
    hidden_sha = hashlib.sha256(hidden_bytes).hexdigest()
    reference_path = private_data_dir() / "reference_quality.json"
    quality_path = task_root / "images/judge/verifier/quality.py"
    workload_path = task_root / "workload_policy.json"
    memory_path = task_root / "memory_policy.json"
    image_manifest = submission.parents[2] / "evaluation-image-manifest.json"
    image_value = {
        "source_digest": "a" * 64,
        "work": {"name": "work:test", "id": "sha256:" + "b" * 64},
        "judge": {"name": "judge:test", "id": "sha256:" + "c" * 64},
    }
    image_manifest.write_text(json.dumps(image_value), encoding="utf-8")
    model_validation = submission.parents[2] / "full-model-validation.json"
    model_validation.write_text(
        json.dumps(
            {
                "status": "PASS",
                "full": True,
                "hashes_checked": 16,
                "logical_size_bytes": 16037449215,
                "revision": "6059b30c9531767ce47e0ca6a8161a6bb0c67693",
            }
        ),
        encoding="utf-8",
    )
    runtime_path, runtime_value = runtime_snapshot(task_root, submission.parents[2])
    runtime_sha = digest(runtime_path)
    row_order = list(json.loads(workload_path.read_text())["manifest"]["row_order"])
    identity = {
        "run_id": RUN_ID,
        "archive_sha256": digest(input_archive),
        "work_image_id": image_value["work"]["id"],
        "judge_image_id": image_value["judge"]["id"],
        "model_validation_sha256": digest(model_validation),
        "runtime_entry_sha256": digest(task_root / "scripts/sforge_h100_entry.py"),
        "source_digest": image_value["source_digest"],
        "workload_policy_sha256": digest(workload_path),
        "memory_policy_sha256": digest(memory_path),
        "quality_controls_sha256": digest(task_root / "quality_controls.json"),
        "hidden_manifest_sha256": hidden_sha,
        "row_order_sha256": hashlib.sha256(("\n".join(row_order) + "\n").encode()).hexdigest(),
        "runtime_provenance_sha256": runtime_sha,
        "sforge_runtime_sha256": runtime_value["sforge_runtime"]["sha256"],
        "sforge_runtime_file_set_sha256": runtime_value["sforge_runtime"][
            "file_set_sha256"
        ],
        **runtime_value["judge_invariant_core"],
    }
    records = [
        {
            "row_id": case["row_id"],
            "case_sha256": hashlib.sha256(case["case_id"].encode()).hexdigest(),
            "correct": False,
            "parsed": True,
            "useful_tokens": 1,
            "seconds": 1.0,
            "torch_peak_allocated_bytes": 100,
            "torch_peak_reserved_bytes": 200,
            "nvml_peak_device_used_bytes": 300,
        }
        for case in hidden
    ]
    row_counts = {row_id: 0 for row_id in row_order}
    for record in records:
        row_counts[record["row_id"]] += 1
    rows = {
        row_id: {
            "cases": count,
            "correct": 0,
            "parsed": count,
            "useful_tokens": count,
            "seconds": float(count),
            "throughput_tok_s": 1.0,
        }
        for row_id, count in row_counts.items()
    }
    commitments = []
    deadline_payload = {
        "schema_version": 1,
        "identity": identity,
        "started_at_unix_seconds": 1_000_000.0,
        "outer_deadline_unix_seconds": 1_007_000.0,
        "outer_timeout_seconds": 7_000.0,
    }
    deadline_payload["commitment_sha256"] = hashlib.sha256(
        canonical(deadline_payload)
    ).hexdigest()
    for row_index, row_id in enumerate(row_order):
        members = [record for record in records if record["row_id"] == row_id]
        hashes = [record["case_sha256"] for record in members]
        full = {
            "schema_version": 2,
            "status": "COMPLETED",
            "identity": identity,
            "row_id": row_id,
            "row_index": row_index,
            "attempt": 0,
            "record_count": len(members),
            "case_set_sha256": hashlib.sha256(("\n".join(sorted(hashes)) + "\n").encode()).hexdigest(),
            "records_sha256": hashlib.sha256(
                json.dumps(members, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "records": members,
            "elapsed_seconds": float(len(members)) + 1.0,
            "deadline_commitment_sha256": deadline_payload["commitment_sha256"],
            "row_memory_sentinel": {
                "schema_version": 2,
                "case": "gsm8k-public-fixed-shape-32",
                "start": {
                    "torch_peak_allocated_bytes": 100,
                    "torch_peak_reserved_bytes": 200,
                    "nvml_peak_device_used_bytes": 300,
                },
                "end": {
                    "torch_peak_allocated_bytes": 104,
                    "torch_peak_reserved_bytes": 204,
                    "nvml_peak_device_used_bytes": 304,
                },
                "repeat": {
                    "torch_peak_allocated_bytes": 104,
                    "torch_peak_reserved_bytes": 204,
                    "nvml_peak_device_used_bytes": 304,
                },
                "deltas_bytes": {
                    "torch_peak_allocated_bytes": 4,
                    "torch_peak_reserved_bytes": 4,
                    "nvml_peak_device_used_bytes": 4,
                },
                "repeat_deltas_bytes": {
                    "torch_peak_allocated_bytes": 0,
                    "torch_peak_reserved_bytes": 0,
                    "nvml_peak_device_used_bytes": 0,
                },
                "observed_max_positive_growth_bytes": 4,
                "observed_max_positive_repeat_growth_bytes": 0,
                "max_material_growth_bytes": 12884901888,
                "max_repeat_growth_bytes": 16777216,
                "repeat_stable": True,
                "stable": True,
                "start_output_sha256": "d" * 64,
                "end_output_sha256": "d" * 64,
                "repeat_output_sha256": "d" * 64,
            },
        }
        full["commitment_sha256"] = hashlib.sha256(canonical(full)).hexdigest()
        commitments.append({key: value for key, value in full.items() if key != "records"})
    all_hashes = [record["case_sha256"] for record in records]
    private = {
        "schema_version": 3,
        "calibration": False,
        "quality_control": False,
        "qualification": True,
        "release_evidence": False,
        "isolation": "root-controller/gold-free-agent-worker",
        "source_ok": True,
        "integrity_ok": True,
        "quality_ok": True,
        "quality_hard_gate": False,
        "performance_ok": True,
        "evaluation_mode": "final",
        "workload_profile": "fast_dllm_v1_llada_12row_stratified520_batch1",
        "generation_count": 520,
        "requested_slots": 216_320,
        "cohort_case_count": 520,
        "cohort_row_counts": row_counts,
        "cohort_category_definition": {
            "kind": "dataset",
            "field": "benchmark",
            "categories": ["gsm8k", "math", "humaneval", "mbpp"],
        },
        "cohort_category_counts": {
            "gsm8k": 130,
            "math": 130,
            "humaneval": 130,
            "mbpp": 130,
        },
        "cohort_bonus_categories": [],
        "submission_number": None,
        "cohort_cycle": None,
        "cohort_slot": None,
        "dataset_manifest_sha256": hidden_sha,
        "reference_manifest_sha256": digest(reference_path),
        "quality_runtime_sha256": digest(quality_path),
        "workload_policy_sha256": digest(workload_path),
        "memory_policy_sha256": digest(memory_path),
        "quality_controls_sha256": digest(task_root / "quality_controls.json"),
        "workload_policy_status": "PENDING",
        "memory_policy_status": "PENDING",
        "visible_device_count": 1,
        "gpu_name": "NVIDIA H100 NVL",
        "cohort_case_set_sha256": hashlib.sha256(
            "\n".join(sorted(all_hashes)).encode()
        ).hexdigest(),
        "rows": rows,
        "records": records,
        "row_commitments": commitments,
        "run_deadline": deadline_payload,
        "reward": 1.0,
        "geomean_throughput_tok_s": 1.0,
        "runtime_provenance": {
            "schema_version": 1,
            "runtime_provenance_sha256": runtime_sha,
            "sforge_runtime": {
                key: runtime_value["sforge_runtime"][key]
                for key in ("scope", "sha256", "file_set_sha256", "file_count")
            },
            "judge_invariant_core": runtime_value["judge_invariant_core"],
        },
        "memory_fixture": None,
    }
    memory_manifest = None
    if memory_fixture:
        memory_manifest = submission.parents[2] / "memory-qualification-fixtures.json"
        manifest_value = {
            "schema_version": 1,
            "status": "PRECOMMITTED",
            "task_id": TASK_ID,
            "profile": "fast_dllm_v1_llada_12row_stratified520_batch1",
            "workload_policy_sha256": digest(workload_path),
            "memory_policy_sha256": digest(memory_path),
            "approval_id": "memory-review",
            "fixtures": {
                "corrected_reference": {
                    "kind": "corrected_reference",
                    "archive": str(input_archive),
                    "archive_sha256": digest(input_archive),
                    "archive_size_bytes": input_archive.stat().st_size,
                    "stress_extra_mib": None,
                },
                "near_cap_stress": {
                    "kind": "near_cap_stress",
                    "archive": "stress.tar.gz",
                    "archive_sha256": "e" * 64,
                    "archive_size_bytes": 123,
                    "stress_extra_mib": 4096,
                },
            },
        }
        memory_manifest.write_text(json.dumps(manifest_value), encoding="utf-8")
        memory_manifest.chmod(0o444)
        private["memory_fixture"] = {
            "schema_version": 1,
            "manifest_sha256": digest(memory_manifest),
            "approval_id": "memory-review",
            "kind": "corrected_reference",
            "archive_sha256": digest(input_archive),
            "archive_size_bytes": input_archive.stat().st_size,
        }
    payload = json.dumps(private).encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        member = tarfile.TarInfo("decoder_private.json")
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))
    (submission / "private_judge_artifacts.tar.gz").write_bytes(buffer.getvalue())
    (submission / "report.json").write_text(
        json.dumps(
            {
                "task_id": TASK_ID,
                "submission_id": RUN_ID,
                "valid": True,
                "timed_out": False,
                "errors": 0,
                "score": 1.0,
                "runtime_seconds": sum(
                    commitment["elapsed_seconds"] for commitment in commitments
                ) + 100.0,
                "metrics": {"reward": 1.0},
            }
        ),
        encoding="utf-8",
    )
    output = submission.parents[2] / "evaluation-attestation.json"
    command = [
        sys.executable,
        str(script),
        "--submission-root", str(submission),
        "--input-archive", str(input_archive),
        "--expected-archive-sha256", digest(input_archive),
        "--expected-archive-size", str(input_archive.stat().st_size),
        "--run-id", RUN_ID,
        "--mode", "qualification",
        "--image-manifest", str(image_manifest),
        "--model-validation", str(model_validation),
        "--runtime-provenance", str(runtime_path),
        "--judge-gpu-uuid", "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "--output", str(output),
        "--task-root", str(task_root),
    ]
    if memory_fixture:
        command.extend(
            (
                "--memory-fixture-kind", "corrected_reference",
                "--memory-fixture-manifest", str(memory_manifest),
                "--nvml-total-bytes", "1000",
                "--nvml-clean-idle-free-bytes", "900",
                "--torch-device-total-bytes", "800",
            )
        )
    return command, output, submission


def rewrite_private_payload(submission: Path, mutate) -> None:
    artifact = submission / "private_judge_artifacts.tar.gz"
    with tarfile.open(artifact, "r:gz") as bundle:
        member = bundle.getmembers()[0]
        stream = bundle.extractfile(member)
        assert stream is not None
        private = json.loads(stream.read())
    mutate(private)
    payload = json.dumps(private).encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        member = tarfile.TarInfo("decoder_private.json")
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))
    artifact.write_bytes(buffer.getvalue())


def test_eval_artifact_validator_produces_identity_bound_qualification_report(task_root, tmp_path):
    command, output, _submission = build_fixture(task_root, tmp_path)
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    attestation = json.loads(output.read_text())
    assert attestation["status"] == "PASS"
    assert attestation["mode"] == "qualification"
    assert attestation["release_evidence"] is False
    assert attestation["generation_count"] == 520
    assert attestation["requested_slots"] == 216_320
    assert attestation["visible_device_count"] == 1
    assert attestation["hardware"] == "NVIDIA H100 NVL"
    assert attestation["row_sentinel_observed_max_growth_bytes"] == 4
    assert attestation["row_sentinel_observed_max_repeat_growth_bytes"] == 0
    assert attestation["row_sentinel_repeat_all_stable"] is True
    assert set(attestation["row_seconds"]) == set(attestation["row_counts"])
    for field in (
        "archive_sha256",
        "hidden_cases_sha256",
        "dataset_manifest_sha256",
        "workload_policy_sha256",
        "memory_policy_sha256",
        "model_validation_sha256",
        "runtime_entry_sha256",
        "source_digest",
        "work_image_id",
        "judge_image_id",
        "judge_gpu_uuid",
    ):
        assert attestation[field]


def test_eval_artifact_validator_rejects_archive_drift(task_root, tmp_path):
    command, _output, submission = build_fixture(task_root, tmp_path)
    (submission / "submission.tar.gz").write_bytes(b"changed")
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode != 0
    assert "differs" in completed.stderr or "changed" in completed.stderr


def test_eval_artifact_validator_binds_memory_fixture_snapshot(task_root, tmp_path):
    command, output, _submission = build_fixture(
        task_root, tmp_path, memory_fixture=True
    )
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    attestation = json.loads(output.read_text())
    assert attestation["memory_fixture_approval_id"] == "memory-review"
    assert attestation["memory_fixture"]["kind"] == "corrected_reference"
    assert attestation["memory_fixture_manifest_sha256"] == attestation[
        "memory_fixture"
    ]["manifest_sha256"]


def test_eval_artifact_validator_rejects_runtime_snapshot_drift(task_root, tmp_path):
    command, _output, submission = build_fixture(task_root, tmp_path)
    snapshot = submission.parents[2] / "runtime-provenance.json"
    snapshot.chmod(0o644)
    value = json.loads(snapshot.read_text())
    value["judge_invariant_core"]["quality_sha256"] = "0" * 64
    snapshot.write_text(json.dumps(value))
    snapshot.chmod(0o444)
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode != 0
    assert "Judge invariant core differs" in completed.stderr


def test_eval_artifact_validator_surfaces_private_failure_before_success_provenance(
    task_root, tmp_path
):
    command, _output, submission = build_fixture(task_root, tmp_path)
    failure = "CandidateFailure: row-long fixed-shape memory/output sentinel changed materially"

    def reject(private):
        private.pop("runtime_provenance")
        private.update(
            {
                "error": failure,
                "integrity_ok": False,
                "performance_ok": False,
                "quality_ok": False,
                "reward": 0.0,
            }
        )

    rewrite_private_payload(submission, reject)
    report_path = submission / "report.json"
    report = json.loads(report_path.read_text())
    report.update({"valid": False, "score": 0.0, "metrics": {"reward": 0.0}})
    report_path.write_text(json.dumps(report), encoding="utf-8")

    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode != 0
    assert f"Judge rejected result: {failure}" in completed.stderr
    assert "private runtime provenance differs" not in completed.stderr


def test_eval_artifact_validator_still_requires_success_runtime_provenance(
    task_root, tmp_path
):
    command, _output, submission = build_fixture(task_root, tmp_path)
    rewrite_private_payload(
        submission, lambda private: private.pop("runtime_provenance")
    )

    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode != 0
    assert "private runtime provenance differs from the host snapshot" in completed.stderr
