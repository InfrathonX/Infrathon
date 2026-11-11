from __future__ import annotations

import subprocess
import hashlib
import io
import json
import tarfile
from pathlib import Path

from sforge.harness.backend import StreamingExecResult
from sforge.harness.backend import ExecResult
from sforge.harness.config import SForgeConfig
from sforge.harness.constants import END_TEST_OUTPUT, START_TEST_OUTPUT
from sforge.harness.run_evaluation import judge_submission
from sforge.harness.task_spec import JudgeSpec, TaskSpec, WorkSpec


def _task(*, eval_cmd: str = "bash -c 'exit 17'") -> TaskSpec:
    return TaskSpec(
        task_id="nonzero-verifier",
        name="nonzero verifier",
        base_image="base",
        platform="linux/amd64",
        cwd="/workspace",
        submit_paths=["."],
        submit_exclude=[],
        work=WorkSpec(
            specs_dir="/tmp/specs",
            agent_query="",
            image_tag="prebuilt",
        ),
        judge=JudgeSpec(
            eval_cmd=eval_cmd,
            eval_timeout=60,
            parser="structured_json",
            image_tag="prebuilt",
        ),
        benchmark_name="test",
    )


def test_generated_eval_script_propagates_verifier_exit_code() -> None:
    completed = subprocess.run(
        ["/bin/bash"],
        input=_task().eval_script,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 17
    assert START_TEST_OUTPUT in completed.stdout
    assert END_TEST_OUTPUT in completed.stdout
    assert "Exit code: 17" in completed.stdout


class _Handle:
    id = "judge-id"
    name = "judge-name"


class _NonzeroBackend:
    backend_name = "k8s"

    def __init__(self, output: str) -> None:
        self.output = output
        self.injected_eval_script = ""
        self.cleaned = False
        self.context = None
        self.environment = None

    def create_container(self, *args, **kwargs):
        self.environment = kwargs.get("environment")
        return _Handle()

    def start_container(self, handle) -> None:
        pass

    def copy_to_container(self, handle, src: Path, dst) -> None:
        if str(dst) == "/tmp/.sforge-evaluation-context.json":
            self.context = json.loads(src.read_text())
        if str(dst) == "/tmp/eval.sh":
            self.injected_eval_script = src.read_text(encoding="utf-8")

    def exec_run(self, handle, cmd, **kwargs):
        return ExecResult(output="", exit_code=0)

    def exec_run_with_timeout(self, handle, cmd, timeout):
        return StreamingExecResult(
            output=self.output,
            exit_code=17,
            timed_out=False,
            elapsed_seconds=0.01,
        )

    def cleanup_container(self, handle, logger=None) -> None:
        self.cleaned = True


def test_nonzero_structured_verifier_cannot_publish_valid_score(
    tmp_path: Path,
) -> None:
    output = "\n".join(
        (
            START_TEST_OUTPUT,
            ">>>>> Start Structured Result",
            '{"valid":true,"score":999.0,"passed":1,"total_tests":1}',
            ">>>>> End Structured Result",
            END_TEST_OUTPUT,
            "Exit code: 17",
        )
    )
    backend = _NonzeroBackend(output)

    report = judge_submission(
        _task(),
        archive=b"",
        config=SForgeConfig(log_dir=tmp_path),
        backend=backend,
        submission_id="explicit-nonzero",
        log_dir=tmp_path / "evaluation",
    )

    assert backend.cleaned is True
    assert 'exit "$EXIT_CODE"' in backend.injected_eval_script
    assert report.valid is False
    assert report.score == 0.0
    assert report.errors >= 1
    assert report.metrics["evaluation_status"] == "infrastructure_error"


def test_interactive_context_binds_exact_archive_without_exposing_it_in_env(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        member = tarfile.TarInfo("candidate.py")
        content = b"pass\n"
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))
    candidate = buffer.getvalue()
    backend = _NonzeroBackend("")
    judge_submission(
        _task(), archive=candidate, config=SForgeConfig(log_dir=tmp_path),
        backend=backend, submission_id="archive-bound", round_id="agent-1",
        log_dir=tmp_path / "evaluation",
    )
    assert backend.context["round_id"] == "agent-1"
    assert backend.context["private"]["ARCHIVE_SHA256"] == hashlib.sha256(candidate).hexdigest()
    assert all("ARCHIVE_SHA256" not in key for key in backend.environment)


def test_mismatched_host_archive_claim_never_creates_judge(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz"):
        pass
    backend = _NonzeroBackend("")
    report = judge_submission(
        _task(), archive=buffer.getvalue(),
        config=SForgeConfig(log_dir=tmp_path, judge_extra_env={"SFORGE_PRIVATE_CONTEXT_ARCHIVE_SHA256": "f" * 64}),
        backend=backend, submission_id="wrong-archive", round_id="agent-1",
        log_dir=tmp_path / "evaluation",
    )
    assert not report.valid
    assert backend.context is None
    assert backend.environment is None


class _PinFailureBackend:
    """A Docker-named backend whose opt-in CPU pin step fails."""

    backend_name = "docker"

    def __init__(self) -> None:
        self.events: list[object] = []

    def create_container(self, *args, **kwargs):
        self.events.append("create")
        return _Handle()

    def apply_cpu_binding(self, handle, policy):
        self.events.append(("pin", policy.role, policy.value))
        raise RuntimeError("cpuset update was not persisted")

    def start_container(self, handle) -> None:
        self.events.append("start")

    def cleanup_container(self, handle, logger=None) -> None:
        self.events.append("cleanup")

    def destroy_container_for_handoff(self, handle) -> dict[str, object]:
        self.events.append("strict-destroy")
        return {"container_destroyed": True}


def test_judge_cpu_pin_failure_never_starts_and_invalidates(tmp_path: Path) -> None:
    backend = _PinFailureBackend()
    log_dir = tmp_path / "evaluation"

    report = judge_submission(
        _task(),
        archive=b"",
        config=SForgeConfig(log_dir=tmp_path, judge_cpuset="auto"),
        backend=backend,
        submission_id="pin-failed",
        log_dir=log_dir,
        judge_image_ref="sha256:" + "d" * 64,
    )

    assert backend.events == ["create", ("pin", "judge", "auto"), "cleanup"]
    assert report.valid is False
    assert report.metrics["evaluation_status"] == "infrastructure_error"
    assert "cpuset update was not persisted" in report.raw_output
    assert not (log_dir / "cpu_affinity.json").exists()


def test_judge_cpu_pin_failure_strictly_destroys_a_coordinated_judge(
    tmp_path: Path,
) -> None:
    from sforge.harness.backend import AcceleratorAllocation

    backend = _PinFailureBackend()

    class Lifecycle:
        def judge_annotations(self) -> dict[str, str]:
            return {"test.lifecycle": "pin-failure"}

        def judge_started(self, handle) -> None:
            raise AssertionError("a pin failure cannot enter JUDGE_RUNNING")

        def judge_cleaning(self, handle) -> None:
            backend.events.append("cleaning")

        def judge_destroyed(self, handle) -> None:
            backend.events.append("destroyed")

    report = judge_submission(
        _task(),
        archive=b"",
        config=SForgeConfig(log_dir=tmp_path, cpuset="auto"),
        backend=backend,
        submission_id="pin-failed-coordinated",
        log_dir=tmp_path / "evaluation",
        judge_image_ref="sha256:" + "d" * 64,
        accelerator_allocation=AcceleratorAllocation(
            "nvidia", ("GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",)
        ),
        container_lifecycle=Lifecycle(),
    )

    assert report.valid is False
    assert report.metrics["evaluation_status"] == "infrastructure_error"
    assert backend.events == [
        "create",
        ("pin", "judge", "auto"),
        "cleaning",
        "strict-destroy",
        "destroyed",
    ]


def test_judge_cpu_pin_on_a_backend_without_support_fails_closed(
    tmp_path: Path,
) -> None:
    backend = _NonzeroBackend("")
    started: list[object] = []
    backend.start_container = started.append  # type: ignore[method-assign]

    report = judge_submission(
        _task(),
        archive=b"",
        config=SForgeConfig(log_dir=tmp_path, judge_cpuset="8-11"),
        backend=backend,
        submission_id="pin-unsupported",
        log_dir=tmp_path / "evaluation",
    )

    assert started == []
    assert backend.cleaned is True
    assert report.valid is False
    assert "cannot apply CPU pinning" in report.raw_output
