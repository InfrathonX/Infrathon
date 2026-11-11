from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import pytest


def load_lifecycle(task_root: Path):
    path = task_root / "images/judge/verifier/row_lifecycle.py"
    name = "row_lifecycle_under_test"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def test_interactive_checkpoints_preserve_identity_and_do_not_reuse_rounds(
    task_root, tmp_path,
):
    module = load_lifecycle(task_root)
    root = tmp_path / "checkpoints"
    root.mkdir(mode=0o700)
    first_metadata = {"round_id": "agent-1", "submission_id": "first", "run_id": "run-1"}
    original = dict(first_metadata)
    first = module.evaluation_checkpoint_root(root, first_metadata, expected_uid=os.getuid())
    second = module.evaluation_checkpoint_root(root, {**first_metadata, "round_id": "agent-2"}, expected_uid=os.getuid())
    retried_request = module.evaluation_checkpoint_root(root, {**first_metadata, "submission_id": "second"}, expected_uid=os.getuid())
    assert first_metadata == original
    assert first == root / "agent-1/first"
    assert len({first, second, retried_request}) == 3
    assert module.evaluation_checkpoint_root(root, {"round_id": "manual-1"}, expected_uid=os.getuid()) == root
    # The same archive identity is legal in independent submissions. A prior
    # deadline/row checkpoint must not change the later request's time budget.
    old_deadline = module.load_or_create_absolute_deadline(first, identity=identity(module), outer_timeout_seconds=60)
    new_deadline = module.load_or_create_absolute_deadline(second, identity=identity(module), outer_timeout_seconds=60)
    assert (first / "run-deadline.json").is_file()
    assert (second / "run-deadline.json").is_file()
    assert new_deadline.started_at_unix_seconds >= old_deadline.started_at_unix_seconds


@pytest.mark.parametrize("metadata", [
    {"round_id": "agent-0", "submission_id": "safe"},
    {"round_id": "agent-1", "submission_id": "../escape"},
    {"round_id": "agent-1"},
])
def test_interactive_checkpoint_identity_fails_closed(task_root, tmp_path, metadata):
    module = load_lifecycle(task_root)
    tmp_path.chmod(0o700)
    with pytest.raises(module.RowLifecycleError, match="identity"):
        module.evaluation_checkpoint_root(tmp_path, metadata, expected_uid=os.getuid())


def test_interactive_checkpoint_directory_cannot_be_a_symlink(task_root, tmp_path):
    module = load_lifecycle(task_root)
    tmp_path.chmod(0o700)
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    (tmp_path / "agent-1").symlink_to(outside, target_is_directory=True)
    with pytest.raises(module.RowLifecycleError, match="ownership/type"):
        module.evaluation_checkpoint_root(tmp_path, {"round_id": "agent-1", "submission_id": "safe"}, expected_uid=os.getuid())


def identity(module):
    return module.EvaluationIdentity(
        run_id="run-1",
        archive_sha256="a" * 64,
        work_image_id="sha256:" + "b" * 64,
        judge_image_id="sha256:" + "c" * 64,
        model_validation_sha256="d" * 64,
        runtime_entry_sha256="e" * 64,
        source_digest="f" * 64,
        workload_policy_sha256="1" * 64,
        memory_policy_sha256="2" * 64,
        quality_controls_sha256="6" * 64,
        runtime_provenance_sha256="7" * 64,
        sforge_runtime_sha256="8" * 64,
        sforge_runtime_file_set_sha256="9" * 64,
        verify_decoder_sha256="a" * 64,
        candidate_worker_sha256="b" * 64,
        cohort_sha256="c" * 64,
        quality_sha256="d" * 64,
        source_admission_sha256="e" * 64,
        hidden_manifest_sha256="3" * 64,
        row_order_sha256="4" * 64,
    )


def record(case: str = "a"):
    return {
        "case_sha256": case * 64,
        "row_id": "row_a",
        "seconds": 1.0,
    }


def sentinel(
    growth: int = 4,
    bound: int = 16,
    *,
    repeat_growth: int = 0,
):
    start = {
        "torch_peak_allocated_bytes": 100,
        "torch_peak_reserved_bytes": 200,
        "nvml_peak_device_used_bytes": 300,
    }
    end = {key: value + growth for key, value in start.items()}
    repeat = {key: value + repeat_growth for key, value in end.items()}
    return {
        "schema_version": 2,
        "case": "gsm8k-public-fixed-shape-32",
        "start": start,
        "end": end,
        "repeat": repeat,
        "deltas_bytes": {key: growth for key in start},
        "repeat_deltas_bytes": {key: repeat_growth for key in start},
        "observed_max_positive_growth_bytes": growth,
        "observed_max_positive_repeat_growth_bytes": repeat_growth,
        "max_material_growth_bytes": bound,
        "max_repeat_growth_bytes": 16 * 1024 * 1024,
        "repeat_stable": True,
        "stable": True,
        "start_output_sha256": "5" * 64,
        "end_output_sha256": "5" * 64,
        "repeat_output_sha256": "5" * 64,
    }


def deadline(module, root, timeout=60.0):
    return module.load_or_create_absolute_deadline(
        root, identity=identity(module), outer_timeout_seconds=timeout
    )


def test_infrastructure_retry_commits_full_payload_and_restart_resumes(task_root, tmp_path):
    module = load_lifecycle(task_root)
    calls = []

    def run(attempt):
        calls.append(attempt)
        if attempt == 0:
            raise module.InfrastructureFailure("controller transport")
        return [record()], 1.25, sentinel()

    run_deadline = deadline(module, tmp_path)

    records, summary = module.run_row_with_retry(
        artifact_root=tmp_path,
        identity=identity(module),
        row_id="row_a",
        row_index=0,
        run_attempt=run,
        deadline=run_deadline,
    )
    assert calls == [0, 1]
    assert records == [record()]
    assert "records" not in summary
    completed = module._read_owned(tmp_path / "completed_rows/row_a.json")
    assert completed["records"] == records
    assert (tmp_path / "row_attempts/row_a/attempt-0-started.json").is_file()
    assert (tmp_path / "row_attempts/row_a/attempt-0-outcome.json").is_file()
    assert (tmp_path / "row_attempts/row_a/attempt-1-started.json").is_file()
    assert (tmp_path / "row_attempts/row_a/attempt-1-outcome.json").is_file()

    def must_not_run(_attempt):
        raise AssertionError("completed row was re-executed")

    resumed_records, resumed_summary = module.run_row_with_retry(
        artifact_root=tmp_path,
        identity=identity(module),
        row_id="row_a",
        row_index=0,
        run_attempt=must_not_run,
        deadline=run_deadline,
    )
    assert resumed_records == records
    assert resumed_summary == summary


def test_candidate_failure_is_never_retried_even_after_restart(task_root, tmp_path):
    module = load_lifecycle(task_root)
    calls = []

    def fail(_attempt):
        calls.append(1)
        raise module.CandidateFailure("candidate timeout")

    with pytest.raises(module.CandidateFailure):
        run_deadline = deadline(module, tmp_path)
        module.run_row_with_retry(
            artifact_root=tmp_path,
            identity=identity(module),
            row_id="row_a",
            row_index=0,
            run_attempt=fail,
            deadline=run_deadline,
        )
    assert len(calls) == 1
    with pytest.raises(module.RowLifecycleError, match="non-retriable"):
        module.run_row_with_retry(
            artifact_root=tmp_path,
            identity=identity(module),
            row_id="row_a",
            row_index=0,
            run_attempt=fail,
            deadline=run_deadline,
        )
    assert len(calls) == 1


def test_completed_row_identity_or_payload_drift_fails_closed(task_root, tmp_path):
    module = load_lifecycle(task_root)
    run_deadline = deadline(module, tmp_path)
    module.run_row_with_retry(
        artifact_root=tmp_path,
        identity=identity(module),
        row_id="row_a",
        row_index=0,
        run_attempt=lambda _attempt: ([record()], 1.0, sentinel()),
        deadline=run_deadline,
    )
    different = identity(module)
    different = module.EvaluationIdentity(
        **{**different.as_dict(), "archive_sha256": "9" * 64}
    )
    with pytest.raises(module.RowLifecycleError, match="identity"):
        module.load_completed_row(
            tmp_path,
            identity=different,
            row_id="row_a",
            row_index=0,
            deadline_commitment_sha256=run_deadline.commitment_sha256,
        )


def test_absolute_deadline_is_identity_bound_and_resume_does_not_reset_budget(
    task_root, tmp_path, monkeypatch
):
    module = load_lifecycle(task_root)
    clock = [1_000_000.0]
    monkeypatch.setattr(module.time, "time", lambda: clock[0])
    first = deadline(module, tmp_path, timeout=120.0)
    assert first.outer_deadline_unix_seconds == 1_000_120.0
    clock[0] += 30.0
    resumed = deadline(module, tmp_path, timeout=120.0)
    assert resumed == first
    assert resumed.remaining_seconds() == 90.0
    clock[0] = 1_000_121.0
    with pytest.raises(module.RowLifecycleError, match="expired"):
        deadline(module, tmp_path, timeout=120.0)


def test_row_sentinel_growth_or_output_drift_cannot_be_committed(task_root, tmp_path):
    module = load_lifecycle(task_root)
    run_deadline = deadline(module, tmp_path)
    leaked = sentinel(growth=17, bound=16)
    with pytest.raises(module.RowLifecycleError, match="internally inconsistent"):
        module.run_row_with_retry(
            artifact_root=tmp_path,
            identity=identity(module),
            row_id="row_a",
            row_index=0,
            run_attempt=lambda _attempt: ([record()], 1.0, leaked),
            deadline=run_deadline,
        )

    repeated_growth = sentinel(repeat_growth=16 * 1024 * 1024 + 1)
    with pytest.raises(module.RowLifecycleError, match="internally inconsistent"):
        module.run_row_with_retry(
            artifact_root=tmp_path / "repeat-growth",
            identity=identity(module),
            row_id="row_a",
            row_index=0,
            run_attempt=lambda _attempt: ([record()], 1.0, repeated_growth),
            deadline=deadline(module, tmp_path / "repeat-growth"),
        )


def test_consistent_unstable_sentinel_is_preserved_as_diagnostic(task_root, tmp_path):
    module = load_lifecycle(task_root)
    diagnostic = sentinel(growth=17, bound=16)
    diagnostic["stable"] = False
    records, summary = module.run_row_with_retry(
        artifact_root=tmp_path,
        identity=identity(module),
        row_id="row_a",
        row_index=0,
        run_attempt=lambda _attempt: ([record()], 1.0, diagnostic),
        deadline=deadline(module, tmp_path),
    )
    assert records == [record()]
    assert summary["row_memory_sentinel"]["stable"] is False


def test_orphan_started_attempt_is_preserved_and_consumes_retry(
    task_root, tmp_path, monkeypatch
):
    module = load_lifecycle(task_root)
    clock = [1_000_000.0]
    monkeypatch.setattr(module.time, "time", lambda: clock[0])
    run_deadline = deadline(module, tmp_path)
    module.record_attempt_started(
        tmp_path,
        identity=identity(module),
        row_id="row_a",
        row_index=0,
        attempt=0,
        deadline_commitment_sha256=run_deadline.commitment_sha256,
    )
    started_path = tmp_path / "row_attempts/row_a/attempt-0-started.json"
    original = started_path.read_bytes()
    clock[0] += 7.0
    calls = []

    records, summary = module.run_row_with_retry(
        artifact_root=tmp_path,
        identity=identity(module),
        row_id="row_a",
        row_index=0,
        run_attempt=lambda attempt: (
            calls.append(attempt) or [record()],
            1.0,
            sentinel(),
        ),
        deadline=run_deadline,
    )
    assert calls == [1]
    assert records == [record()]
    assert summary["attempt"] == 1
    assert started_path.read_bytes() == original
    outcome = module._read_owned(
        tmp_path / "row_attempts/row_a/attempt-0-outcome.json"
    )
    assert outcome["status"] == "FAILED_INFRASTRUCTURE"
    assert outcome["error_class"] == "ControllerInterrupted"
    assert outcome["elapsed_seconds"] == 7.0


def test_second_orphan_started_attempt_exhausts_retry_without_execution(
    task_root, tmp_path
):
    module = load_lifecycle(task_root)
    run_deadline = deadline(module, tmp_path)
    for attempt in (0, 1):
        module.record_attempt_started(
            tmp_path,
            identity=identity(module),
            row_id="row_a",
            row_index=0,
            attempt=attempt,
            deadline_commitment_sha256=run_deadline.commitment_sha256,
        )

    with pytest.raises(module.RowLifecycleError, match="exhausted"):
        module.run_row_with_retry(
            artifact_root=tmp_path,
            identity=identity(module),
            row_id="row_a",
            row_index=0,
            run_attempt=lambda _attempt: (_ for _ in ()).throw(
                AssertionError("must not execute")
            ),
            deadline=run_deadline,
        )
