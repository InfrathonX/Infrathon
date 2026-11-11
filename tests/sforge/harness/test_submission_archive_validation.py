from __future__ import annotations

import io
import hashlib
import json
import os
import stat
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import sforge.harness.run_evaluation as evaluation
from sforge.harness.backend import AcceleratorAllocation, ExecResult, StreamingExecResult
from sforge.harness.config import SForgeConfig
from sforge.harness.constants import END_TEST_OUTPUT, START_TEST_OUTPUT
from sforge.harness.run_evaluation import (
    _canonical_submission_member_name,
    _durable_atomic_write_bytes,
    _filter_archive_files,
    _validate_submission_archive,
    judge_submission,
)
from sforge.harness.task_spec import JudgeSpec, TaskSpec, WorkSpec
from sforge.harness.resource_handoff import canonical_allocation_key


def test_evaluation_artifact_write_fsyncs_file_replace_and_directory_in_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    original_fsync = os.fsync
    original_replace = os.replace

    def recording_fsync(descriptor: int) -> None:
        kind = (
            "directory"
            if stat.S_ISDIR(os.fstat(descriptor).st_mode)
            else "file"
        )
        events.append(f"fsync:{kind}")
        original_fsync(descriptor)

    def recording_replace(source, destination) -> None:
        events.append("replace")
        original_replace(source, destination)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    monkeypatch.setattr(os, "replace", recording_replace)
    destination = tmp_path / "artifacts" / "report.json"

    _durable_atomic_write_bytes(destination, b"durable-report\n")

    assert destination.read_bytes() == b"durable-report\n"
    assert destination.stat().st_mode & 0o777 == 0o600
    assert events == ["fsync:file", "replace", "fsync:directory"]


def test_evaluation_artifact_write_cleans_temp_on_replace_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "report.json"
    destination.write_bytes(b"previous")

    def fail_replace(source, target) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        _durable_atomic_write_bytes(destination, b"new")

    assert destination.read_bytes() == b"previous"
    assert list(tmp_path.glob(".report.json.*")) == []


def test_private_judge_artifact_uses_durable_writer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = b'{"trusted":"evidence"}\n'
    transport = _tar(("final.json", payload, None, None))

    class Backend:
        def copy_from_container(self, handle, path) -> bytes:
            assert str(path) == "/logs/verifier/final.json"
            return transport

    calls: list[tuple[Path, bytes]] = []
    original_write = evaluation._durable_atomic_write_bytes

    def recording_write(path: Path, value: bytes) -> None:
        calls.append((path, value))
        original_write(path, value)

    monkeypatch.setattr(
        evaluation,
        "_durable_atomic_write_bytes",
        recording_write,
    )

    destination = evaluation._persist_private_judge_artifact(
        Backend(),
        object(),
        "/logs/verifier/final.json",
        tmp_path,
    )

    assert destination == tmp_path / evaluation.PRIVATE_JUDGE_ARTIFACT_NAME
    assert calls == [(destination, destination.read_bytes())]
    assert destination.stat().st_mode & 0o777 == 0o600
    with tarfile.open(destination, mode="r:gz") as archive:
        member = archive.getmember("final.json")
        stream = archive.extractfile(member)
        assert stream is not None
        assert stream.read() == payload


def _tar(*members: tuple[str, bytes | None, bytes | None, str | None]) -> bytes:
    """Build a tar from (name, payload, type, linkname) test entries."""

    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, payload, member_type, linkname in members:
            info = tarfile.TarInfo(name)
            info.type = member_type or tarfile.REGTYPE
            if linkname is not None:
                info.linkname = linkname
            if info.type in {tarfile.REGTYPE, tarfile.AREGTYPE}:
                data = payload or b""
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
            else:
                info.size = 0
                archive.addfile(info)
    return output.getvalue()


def test_accepts_canonical_regular_files_and_directories() -> None:
    archive = _tar(
        ("submission/", None, tarfile.DIRTYPE, None),
        ("submission/gateway.py", b"print('ok')\n", None, None),
        ("submission/launch.sh", b"#!/bin/sh\n", None, None),
    )

    assert _validate_submission_archive(archive) == [
        "submission/gateway.py",
        "submission/launch.sh",
    ]


def test_validated_names_extract_with_nul_verbatim_file_list(tmp_path: Path) -> None:
    tricky_name = "submission/diagnostic\n-C"
    archive = _tar(
        ("submission/launch.sh", b"#!/bin/sh\n", None, None),
        (tricky_name, b"safe\n", None, None),
    )
    names = _validate_submission_archive(archive)
    archive_path = tmp_path / "submission.tar.gz"
    archive_path.write_bytes(archive)
    names_path = tmp_path / "members"
    names_path.write_bytes(
        b"\0".join(name.encode("utf-8") for name in names) + b"\0"
    )
    destination = tmp_path / "destination"
    destination.mkdir()

    completed = subprocess.run(
        [
            "tar",
            "xzf",
            str(archive_path),
            "--null",
            "--verbatim-files-from",
            "-T",
            str(names_path),
        ],
        cwd=destination,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (destination / "submission" / "launch.sh").read_bytes() == b"#!/bin/sh\n"
    assert (destination / tricky_name).read_bytes() == b"safe\n"


@pytest.mark.parametrize(
    "name",
    [
        "/opt/kvbench-mlsysforge/run_judge.py",
        "submission/../../opt/run_judge.py",
        "./submission/launch.sh",
        "submission/./launch.sh",
        "submission//launch.sh",
    ],
)
def test_rejects_noncanonical_or_traversing_member_names(name: str) -> None:
    with pytest.raises(ValueError, match="absolute|canonical"):
        _validate_submission_archive(_tar((name, b"owned", None, None)))


def test_rejects_nul_member_name() -> None:
    with pytest.raises(ValueError, match="NUL"):
        _canonical_submission_member_name("submission/launch.sh\x00shadow")


@pytest.mark.parametrize(
    "member_type",
    [
        tarfile.LNKTYPE,
        tarfile.CHRTYPE,
        tarfile.BLKTYPE,
        tarfile.FIFOTYPE,
    ],
)
def test_rejects_hardlinks_devices_fifos_and_other_special_files(
    member_type: bytes,
) -> None:
    with pytest.raises(ValueError, match="link or special"):
        _validate_submission_archive(
            _tar(
                (
                    "submission/payload",
                    None,
                    member_type,
                    "submission/launch.sh" if member_type == tarfile.LNKTYPE else None,
                )
            )
        )


def test_rejects_symlink_pivot_before_following_file() -> None:
    archive = _tar(
        ("submission/pivot", None, tarfile.SYMTYPE, "../../../opt"),
        ("submission/pivot/kvbench-mlsysforge/run_judge.py", b"owned", None, None),
    )

    with pytest.raises(ValueError, match="link or special"):
        _validate_submission_archive(archive)


def test_rejects_duplicate_member_names() -> None:
    archive = _tar(
        ("submission/launch.sh", b"first", None, None),
        ("submission/launch.sh", b"second", None, None),
    )

    with pytest.raises(ValueError, match="duplicate"):
        _validate_submission_archive(archive)


def test_rejects_compressed_uncompressed_and_member_count_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sforge.harness import run_evaluation

    one_file = _tar(("submission/data", b"1234", None, None))
    monkeypatch.setattr(run_evaluation, "SUBMISSION_ARCHIVE_MAX_BYTES", len(one_file) - 1)
    with pytest.raises(ValueError, match="compressed size"):
        _validate_submission_archive(one_file)

    monkeypatch.setattr(run_evaluation, "SUBMISSION_ARCHIVE_MAX_BYTES", len(one_file))
    monkeypatch.setattr(run_evaluation, "SUBMISSION_ARCHIVE_MAX_UNCOMPRESSED_BYTES", 3)
    with pytest.raises(ValueError, match="uncompressed size"):
        _validate_submission_archive(one_file)

    two_files = _tar(
        ("submission/a", b"a", None, None),
        ("submission/b", b"b", None, None),
    )
    monkeypatch.setattr(run_evaluation, "SUBMISSION_ARCHIVE_MAX_BYTES", len(two_files))
    monkeypatch.setattr(
        run_evaluation,
        "SUBMISSION_ARCHIVE_MAX_UNCOMPRESSED_BYTES",
        len(two_files),
    )
    monkeypatch.setattr(run_evaluation, "SUBMISSION_ARCHIVE_MAX_MEMBERS", 1)
    with pytest.raises(ValueError, match="too many members"):
        _validate_submission_archive(two_files)


def test_whitelist_rechecks_canonical_names_and_excludes_nested_components() -> None:
    files = [
        "submission/gateway.py",
        "submission/pkg/__pycache__/gateway.cpython-313.pyc",
        "outside/ignored.py",
    ]
    assert _filter_archive_files(
        files,
        submit_paths=["submission"],
        submit_exclude=["__pycache__", "*.pyc"],
    ) == ["submission/gateway.py"]

    with pytest.raises(ValueError, match="canonical"):
        _filter_archive_files(
            ["submission/../../opt/run_judge.py"],
            submit_paths=["submission"],
            submit_exclude=[],
        )


def _task() -> TaskSpec:
    return TaskSpec(
        task_id="archive-contract",
        name="archive contract",
        base_image="base",
        platform="linux/amd64",
        cwd="/app",
        submit_paths=["submission"],
        submit_exclude=[],
        work=WorkSpec(specs_dir="/tmp/specs", agent_query="", image_tag="prebuilt"),
        judge=JudgeSpec(
            eval_cmd="true",
            eval_timeout=60,
            parser="structured_json",
            image_tag="prebuilt",
        ),
        benchmark_name="test",
    )


def _portable_task() -> TaskSpec:
    task = _task()
    task.portable = SimpleNamespace(
        evaluation_contract_sha256="sha256:" + "c" * 64,
        defaults=SimpleNamespace(
            to_dict=lambda: {
                "evaluation_suite_id": "suite",
                "execution_profile_id": "profile",
                "metric_id": "metric",
            }
        ),
    )
    return task


def _portable_binding() -> dict[str, object]:
    return {
        "evaluation_contract_sha256": "sha256:" + "c" * 64,
        "selected_ids": {
            "evaluation_suite_id": "suite",
            "execution_profile_id": "profile",
            "metric_id": "metric",
        },
    }


class _NeverCreateBackend:
    backend_name = "k8s"

    def __init__(self) -> None:
        self.created = False

    def create_container(self, *args, **kwargs):
        self.created = True
        raise AssertionError("unsafe archive reached Judge creation")

    def cleanup_container(self, handle, logger=None) -> None:
        pass


def test_malformed_archive_is_rejected_before_judge_creation(tmp_path: Path) -> None:
    backend = _NeverCreateBackend()
    malicious = _tar(
        ("submission/pivot", None, tarfile.SYMTYPE, "../../../opt"),
        ("submission/pivot/run_judge.py", b"owned", None, None),
    )

    report = judge_submission(
        _task(),
        archive=malicious,
        config=SForgeConfig(log_dir=tmp_path),
        backend=backend,
        submission_id="malformed-archive",
        log_dir=tmp_path / "evaluation",
    )

    assert backend.created is False
    assert report.valid is False
    assert report.score is None
    assert (tmp_path / "evaluation/submission.tar.gz").read_bytes() == malicious
    persisted = json.loads(
        (tmp_path / "evaluation/report.json").read_text(encoding="utf-8")
    )
    assert persisted["submission_id"] == "malformed-archive"
    assert persisted["valid"] is False
    assert persisted["metrics"]["evaluation_status"] == "infrastructure_error"


def test_early_failure_report_binds_trusted_portable_contract(
    tmp_path: Path,
) -> None:
    backend = _NeverCreateBackend()
    report = judge_submission(
        _portable_task(),
        archive=_tar(("submission/pivot", None, tarfile.SYMTYPE, "../x")),
        config=SForgeConfig(log_dir=tmp_path),
        backend=backend,
        submission_id="portable-early-failure",
        log_dir=tmp_path / "evaluation",
    )

    assert report.metrics["portable_contract"] == _portable_binding()
    persisted = json.loads(
        (tmp_path / "evaluation/report.json").read_text(encoding="utf-8")
    )
    assert persisted["metrics"]["portable_contract"] == _portable_binding()


class _ExtractionBackend:
    backend_name = "k8s"

    def __init__(self) -> None:
        self.copied: dict[str, bytes] = {}
        self.commands: list[str | list[str]] = []
        self.environment: dict[str, str] = {}
        self.created_image: str | None = None
        self.create_kwargs: dict[str, object] = {}

    def create_container(self, *args, **kwargs):
        self.created_image = args[0]
        self.create_kwargs = dict(kwargs)
        self.environment = dict(kwargs.get("environment", {}))
        return type("Handle", (), {"id": "judge-id", "name": "judge-name"})()

    def start_container(self, handle) -> None:
        pass

    def copy_to_container(self, handle, src: Path, dst) -> None:
        self.copied[str(dst)] = src.read_bytes()

    def exec_run(self, handle, cmd, **kwargs):
        self.commands.append(cmd)
        return ExecResult(output="", exit_code=0)

    def exec_run_with_timeout(self, handle, cmd, timeout):
        output = "\n".join(
            (
                START_TEST_OUTPUT,
                ">>>>> Start Structured Result",
                '{"valid":true,"score":1.0,"passed":1,"total_tests":1}',
                ">>>>> End Structured Result",
                END_TEST_OUTPUT,
                "Exit code: 0",
            )
        )
        return StreamingExecResult(
            output=output,
            exit_code=0,
            timed_out=False,
            elapsed_seconds=0.01,
        )

    def cleanup_container(self, handle, logger=None) -> None:
        pass


def test_valid_archive_flow_uses_validated_verbatim_names(tmp_path: Path) -> None:
    archive = _tar(
        ("submission/", None, tarfile.DIRTYPE, None),
        ("submission/gateway.py", b"print('ok')\n", None, None),
        ("submission/launch.sh", b"#!/bin/sh\n", None, None),
    )
    backend = _ExtractionBackend()
    config = SForgeConfig(log_dir=tmp_path)
    config.judge_extra_env["SFORGE_PRIVATE_CONTEXT_COHORT_SECRET"] = "secret-value"

    report = judge_submission(
        _task(),
        archive=archive,
        config=config,
        backend=backend,
        submission_id="valid-archive",
        log_dir=tmp_path / "evaluation",
        run_id="run-42",
        round_id="agent-7",
    )

    assert report.valid is True
    assert report.score == 1.0
    assert backend.copied["/tmp/submission.tar.gz"] == archive
    assert backend.copied["/tmp/allowed_files.txt"] == (
        b"submission/gateway.py\0submission/launch.sh\0"
    )
    assert len(backend.commands) == 3
    extraction = next(
        command
        for command in backend.commands
        if isinstance(command, str) and "tar xzf" in command
    )
    assert "--null --verbatim-files-from" in extraction
    assert "--no-same-owner" in extraction
    assert "--touch" not in extraction
    metadata_command = next(
        command
        for command in backend.commands
        if isinstance(command, list) and "touch -h -d @0" in command[-1]
    )
    assert metadata_command[:2] == ["/bin/bash", "-c"]
    assert "read -r -d '' path" in metadata_command[-1]
    assert "tar tzf" not in extraction
    assert not any(key.startswith("SFORGE_EVAL_") for key in backend.environment)
    assert not any(
        key.startswith("SFORGE_PRIVATE_CONTEXT_") for key in backend.environment
    )
    context = json.loads(
        backend.copied["/tmp/.sforge-evaluation-context.json"].decode("utf-8")
    )
    assert context == {
        "schema_version": 1,
        "submission_id": "valid-archive",
        "run_id": "run-42",
        "round_id": "agent-7",
        "private": {
            "COHORT_SECRET": "secret-value",
            "ARCHIVE_SHA256": hashlib.sha256(archive).hexdigest(),
            "EVALUATION_PHASE": "interactive",
        },
    }


def test_success_report_binds_trusted_portable_contract(tmp_path: Path) -> None:
    report = judge_submission(
        _portable_task(),
        archive=_tar(("submission/gateway.py", b"ok\n", None, None)),
        config=SForgeConfig(log_dir=tmp_path),
        backend=_ExtractionBackend(),
        submission_id="portable-success",
        log_dir=tmp_path / "evaluation",
    )

    assert report.valid is True
    assert report.metrics["portable_contract"] == _portable_binding()
    persisted = json.loads(
        (tmp_path / "evaluation/report.json").read_text(encoding="utf-8")
    )
    assert persisted["metrics"]["portable_contract"] == _portable_binding()


def test_explicit_judge_image_ref_is_the_exact_execution_image(
    tmp_path: Path,
) -> None:
    backend = _ExtractionBackend()
    immutable_ref = "sha256:" + "a" * 64

    report = judge_submission(
        _task(),
        archive=_tar(("submission/gateway.py", b"ok\n", None, None)),
        config=SForgeConfig(log_dir=tmp_path),
        backend=backend,
        submission_id="pinned-image",
        log_dir=tmp_path / "evaluation",
        judge_image_ref=immutable_ref,
    )

    assert report.valid is True
    assert backend.created_image == immutable_ref


def test_judge_network_none_reaches_create_boundary_without_host_gateway(
    tmp_path: Path,
) -> None:
    backend = _ExtractionBackend()
    backend.backend_name = "docker"
    report = judge_submission(
        _task(),
        archive=_tar(("submission/gateway.py", b"ok\n", None, None)),
        config=SForgeConfig(log_dir=tmp_path, judge_network_mode="none"),
        backend=backend,
        submission_id="network-none",
        log_dir=tmp_path / "evaluation",
        judge_image_ref="sha256:" + "b" * 64,
    )

    assert report.valid is True
    assert backend.create_kwargs["network_mode"] == "none"
    assert "extra_hosts" not in backend.create_kwargs


def test_judge_network_none_fails_before_an_unsupported_backend_create(
    tmp_path: Path,
) -> None:
    backend = _ExtractionBackend()
    report = judge_submission(
        _task(),
        archive=_tar(("submission/gateway.py", b"ok\n", None, None)),
        config=SForgeConfig(log_dir=tmp_path, judge_network_mode="none"),
        backend=backend,
        submission_id="network-none-unsupported",
        log_dir=tmp_path / "evaluation",
        judge_image_ref="sha256:" + "c" * 64,
    )

    assert report.valid is False
    assert report.metrics["evaluation_status"] == "infrastructure_error"
    assert backend.created_image is None


def test_cleanup_or_post_attestation_failure_overrides_a_valid_report(
    tmp_path: Path,
) -> None:
    class _CleanupFailureBackend(_ExtractionBackend):
        def cleanup_container(self, handle, logger=None) -> None:
            raise RuntimeError("private post-attestation details")

    backend = _CleanupFailureBackend()
    log_dir = tmp_path / "evaluation"

    report = judge_submission(
        _portable_task(),
        archive=_tar(("submission/gateway.py", b"ok\n", None, None)),
        config=SForgeConfig(log_dir=tmp_path),
        backend=backend,
        submission_id="cleanup-failed",
        log_dir=log_dir,
    )

    assert report.valid is False
    assert report.score == 0.0
    assert report.errors >= 1
    assert report.metrics["evaluation_status"] == "infrastructure_error"
    assert report.metrics["portable_contract"] == _portable_binding()
    assert "private post-attestation details" not in report.raw_output
    persisted = json.loads((log_dir / "report.json").read_text(encoding="utf-8"))
    assert persisted["valid"] is False
    assert persisted["metrics"]["evaluation_status"] == "infrastructure_error"
    assert persisted["metrics"]["portable_contract"] == _portable_binding()


def test_lifecycle_cleaning_failure_cannot_skip_strict_destroy(
    tmp_path: Path,
) -> None:
    events: list[str] = []

    class Backend(_ExtractionBackend):
        def destroy_container_for_handoff(self, handle) -> None:
            events.append("strict-destroy")

    class Lifecycle:
        def judge_annotations(self) -> dict[str, str]:
            return {"test.lifecycle": "cleaning-failure"}

        def judge_started(self, handle) -> None:
            events.append("started")

        def judge_cleaning(self, handle) -> None:
            events.append("cleaning-error")
            raise RuntimeError("cleaning callback failed")

        def judge_destroyed(self, handle) -> None:
            events.append("destroyed-callback")

    backend = Backend()
    report = judge_submission(
        _task(),
        archive=_tar(("submission/gateway.py", b"ok\n", None, None)),
        config=SForgeConfig(log_dir=tmp_path),
        backend=backend,
        submission_id="cleaning-callback-failed",
        log_dir=tmp_path / "evaluation",
        judge_image_ref="sha256:" + "e" * 64,
        accelerator_allocation=AcceleratorAllocation(
            "nvidia",
            ("GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",),
        ),
        container_lifecycle=Lifecycle(),
    )

    assert report.valid is False
    assert events == [
        "started",
        "cleaning-error",
        "strict-destroy",
        "destroyed-callback",
    ]
    assert backend.create_kwargs["annotations"] == {
        "test.lifecycle": "cleaning-failure"
    }


def test_start_failure_still_strictly_destroys_created_judge(
    tmp_path: Path,
) -> None:
    events: list[str] = []

    class Backend(_ExtractionBackend):
        def start_container(self, handle) -> None:
            events.append("start-error")
            raise RuntimeError("start failed")

        def destroy_container_for_handoff(self, handle) -> None:
            events.append("strict-destroy")

    class Lifecycle:
        def judge_annotations(self) -> dict[str, str]:
            return {"test.lifecycle": "start-failure"}

        def judge_started(self, handle) -> None:
            raise AssertionError("start failure cannot enter JUDGE_RUNNING")

        def judge_cleaning(self, handle) -> None:
            events.append("cleaning")

        def judge_destroyed(self, handle) -> None:
            events.append("destroyed-callback")

    report = judge_submission(
        _task(),
        archive=_tar(("submission/gateway.py", b"ok\n", None, None)),
        config=SForgeConfig(log_dir=tmp_path),
        backend=Backend(),
        submission_id="start-failed",
        log_dir=tmp_path / "evaluation",
        judge_image_ref="sha256:" + "f" * 64,
        accelerator_allocation=AcceleratorAllocation(
            "nvidia",
            ("GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",),
        ),
        container_lifecycle=Lifecycle(),
    )

    assert report.valid is False
    assert events == [
        "start-error",
        "cleaning",
        "strict-destroy",
        "destroyed-callback",
    ]


def test_direct_single_gpu_eval_requires_live_coordinator(
    tmp_path: Path,
) -> None:
    gpu_uuid = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    allocation = AcceleratorAllocation("nvidia", (gpu_uuid,))
    config = SForgeConfig(
        log_dir=tmp_path,
        backend="docker",
        single_gpu_handoff=True,
        accelerator_device_uuid=gpu_uuid,
        accelerator_allocation_key=canonical_allocation_key(allocation),
        accelerator_observer="nvml",
        trusted_lease_id="lease-coordinator-required",
    )

    with pytest.raises(ValueError, match="live coordinator lifecycle"):
        judge_submission(
            _task(),
            archive=_tar(("submission/gateway.py", b"ok\n", None, None)),
            config=config,
            backend=_ExtractionBackend(),
            submission_id="unsafe-direct-final",
            log_dir=tmp_path / "evaluation",
            accelerator_allocation=allocation,
        )


def test_cpu_pinning_off_leaves_judge_creation_byte_identical(
    tmp_path: Path,
) -> None:
    # _ExtractionBackend has no apply_cpu_binding: the off path never looks.
    backend = _ExtractionBackend()
    log_dir = tmp_path / "evaluation"

    report = judge_submission(
        _task(),
        archive=_tar(("submission/gateway.py", b"ok\n", None, None)),
        config=SForgeConfig(log_dir=tmp_path, cpuset="off"),
        backend=backend,
        submission_id="unpinned",
        log_dir=log_dir,
    )

    assert report.valid is True
    assert backend.create_kwargs == {
        "environment": {"SFORGE_JUDGE_URL": "http://host.docker.internal:8080"},
        "cpu_limit": None,
        "mem_limit": None,
        "user": "root",
        "extra_hosts": {"host.docker.internal": "host-gateway"},
    }
    assert not (log_dir / "cpu_affinity.json").exists()


def test_cpu_pinning_runs_between_create_and_start_and_is_recorded(
    tmp_path: Path,
) -> None:
    events: list[str] = []

    class Backend(_ExtractionBackend):
        backend_name = "docker"

        def create_container(self, *args, **kwargs):
            events.append("create")
            return super().create_container(*args, **kwargs)

        def apply_cpu_binding(self, handle, policy):
            events.append(f"pin:{policy.role}:{policy.value}")
            return {"outcome": "pinned", "cpuset_cpus": "8-11,40-43", "cpuset_mems": "0"}

        def start_container(self, handle) -> None:
            events.append("start")

    backend = Backend()
    log_dir = tmp_path / "evaluation"
    report = judge_submission(
        _task(),
        archive=_tar(("submission/gateway.py", b"ok\n", None, None)),
        config=SForgeConfig(log_dir=tmp_path, cpuset="auto", work_cpuset="off"),
        backend=backend,
        submission_id="pinned",
        log_dir=log_dir,
        judge_image_ref="sha256:" + "f" * 64,
    )

    assert report.valid is True
    assert events == ["create", "pin:judge:auto", "start"]
    # Pinning never changes what the Judge container is created with.
    assert "cpuset_cpus" not in backend.create_kwargs
    record = json.loads((log_dir / "cpu_affinity.json").read_text(encoding="utf-8"))
    assert record["role"] == "judge"
    assert record["container_id"] == "judge-id"
    assert record["policy_id"] == "auto:gpu-local-l3-v1:mems=bind"
    assert record["cpuset_cpus"] == "8-11,40-43"
    # The record is host-side only; the Judge sees no pinning context.
    assert not any("cpu" in key.lower() for key in backend.environment)
