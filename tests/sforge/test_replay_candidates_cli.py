from __future__ import annotations

import hashlib
import io
import json
import re
import sys
import tarfile
import threading
import types
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pytest

from sforge import cli
from sforge.harness import candidate_replay
from sforge.harness.backend.base import ResolvedImage
from sforge.harness.config import SForgeConfig
from sforge.harness.grading import EvalReport
from sforge.harness.score_rescale import RescaleSpec
from sforge.harness.task_spec import JudgeSpec, TaskSpec, WorkSpec
from sforge.harness.version_store import VersionStore, VersionStoreCollision


def _archive(payload: bytes) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as bundle:
        member = tarfile.TarInfo("submission/main.py")
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))
    return output.getvalue()


def _store_root(root: Path) -> Path:
    root.mkdir()
    (root / ".sforge-version-store.json").write_text(
        '{"schema_version": 1, "kind": "sforge-version-store", '
        '"store_id": "replay-test-store"}\n',
        encoding="utf-8",
    )
    task_root = root / "tasks" / "task-alpha"
    (task_root / "runs").mkdir(parents=True)
    (task_root / "task.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_directory": "task-alpha",
                "task_id": "task_alpha",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def _record_a_b_a(root: Path) -> list[bytes]:
    store = VersionStore.open(root)
    archives = [_archive(b"A\n"), _archive(b"B\n"), _archive(b"A\n")]
    for index, archive in enumerate(archives, start=1):
        store.record_archive(
            task_id="task_alpha",
            run_id="origin-run",
            source="checkpoint",
            round_id=f"checkpoint-{index}",
            label=("first-a", "try-b", "restore-a")[index - 1],
            archive=archive,
            submit_paths=["submission"],
            submit_exclude=[],
            created_at=datetime(
                2026, 8, 20, 12, index, tzinfo=timezone.utc
            ),
        )
    return archives


def test_replay_input_stays_on_first_seen_event_when_checkpoint_arrives_later(
    tmp_path: Path,
) -> None:
    root = _store_root(tmp_path / "store")
    store = VersionStore.open(root)
    archive = _archive(b"A\n")
    first = store.record_archive(
        task_id="task_alpha",
        run_id="origin-run",
        source="public",
        round_id="public-1",
        label="first-seen-by-judge",
        archive=archive,
        submit_paths=["submission"],
        submit_exclude=[],
        created_at=datetime(2026, 8, 20, 12, 1, tzinfo=timezone.utc),
    )
    store.record_archive(
        task_id="task_alpha",
        run_id="origin-run",
        source="checkpoint",
        round_id="checkpoint-1",
        label="late-checkpoint",
        archive=archive,
        submit_paths=["submission"],
        submit_exclude=[],
        created_at=datetime(2026, 8, 20, 12, 2, tzinfo=timezone.utc),
    )

    candidate = store.list_candidates(
        task="task_alpha", run_id="origin-run"
    )[0]

    assert candidate_replay.input_version(candidate).version_id == first.version_id
    assert candidate.archive_path == first.archive_path


def _task_spec(*, score_direction: str = "maximize") -> TaskSpec:
    return TaskSpec(
        task_id="task_alpha",
        name="Task alpha",
        base_image="python310",
        platform="linux/amd64",
        cwd="/workspace",
        submit_paths=["submission"],
        submit_exclude=[],
        work=WorkSpec(specs_dir="", agent_query="", image_tag="work-image"),
        judge=JudgeSpec(
            eval_cmd="pytest -q",
            eval_timeout=60,
            parser="pytest",
            image_tag="judge-image",
            score_direction=score_direction,
        ),
        benchmark_name="mlsysforge",
        base_image_spec={"image": "python:3.10"},
    )


def _run_cli(monkeypatch: pytest.MonkeyPatch, *arguments: str) -> None:
    monkeypatch.setattr(sys, "argv", ["sforge", *arguments])
    cli.main()


def _patch_config_and_task(
    monkeypatch: pytest.MonkeyPatch,
    *,
    root: Path,
    output: Path,
    score_direction: str = "maximize",
    allow_open_origin: bool = True,
) -> None:
    config = SForgeConfig(
        log_dir=output,
        version_store_root=root,
    )
    monkeypatch.setattr(cli, "load_config", lambda overrides=None: config)
    monkeypatch.setattr(
        cli,
        "_resolve_task",
        lambda args, resolved_config: _task_spec(
            score_direction=score_direction
        ),
    )
    monkeypatch.setattr(
        cli,
        "_resolve_replay_judge_image",
        lambda backend, task_spec, resolved_config: _resolved_test_image(
            task_spec
        ),
    )
    monkeypatch.setattr(
        cli,
        "_reuse_replay_judge_image",
        lambda backend, plan, task_spec, resolved_config: _resolved_test_image(
            task_spec
        ),
        raising=False,
    )
    if allow_open_origin:
        monkeypatch.setattr(
            cli,
            "_validate_replay_origin_closed",
            lambda candidates: None,
            raising=False,
        )
    monkeypatch.setattr(
        candidate_replay,
        "_loaded_sforge_module_names",
        lambda: ["sforge.cli", "sforge.harness.candidate_replay"],
    )
    monkeypatch.setattr(
        candidate_replay,
        "_loaded_evaluator_code_sha256",
        lambda modules=None: "sha256:" + "c" * 64,
    )
    monkeypatch.setattr(
        candidate_replay,
        "_loaded_evaluator_bytecode_sha256",
        lambda modules=None: "sha256:" + "d" * 64,
    )


def _resolved_test_image(task_spec: TaskSpec | None = None) -> ResolvedImage:
    task_spec = task_spec or _task_spec()
    image_id = "sha256:" + "a" * 64
    return ResolvedImage(
        backend="docker",
        requested_ref=task_spec.judge_image_key,
        execution_ref=image_id,
        content_identity=image_id,
        identity_kind="docker-image-id",
    )


def _record_final(root: Path, archive: bytes) -> None:
    VersionStore.open(root).record_archive(
        task_id="task_alpha",
        run_id="origin-run",
        source="final",
        round_id="final",
        label="final",
        archive=archive,
        submit_paths=["submission"],
        submit_exclude=[],
        created_at=datetime(2026, 8, 20, 12, 59, tzinfo=timezone.utc),
    )


def test_replay_candidates_requires_one_last_final_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _store_root(tmp_path / "store")
    archives = _record_a_b_a(root)
    _patch_config_and_task(
        monkeypatch,
        root=root,
        output=tmp_path / "output",
        allow_open_origin=False,
    )
    arguments = (
        "replay-candidates",
        "--task",
        "task_alpha",
        "--run-id",
        "origin-run",
        "--measurement-set",
        "hidden",
        "--dry-run",
    )

    with pytest.raises(SystemExit) as raised:
        _run_cli(monkeypatch, *arguments)
    assert raised.value.code == 2
    assert "exactly one final event" in capsys.readouterr().err

    _record_final(root, archives[0])
    _run_cli(monkeypatch, *arguments)
    capsys.readouterr()

    with pytest.raises(VersionStoreCollision, match="closed by its final event"):
        VersionStore.open(root).record_archive(
            task_id="task_alpha",
            run_id="origin-run",
            source="checkpoint",
            round_id="checkpoint-after-final",
            label="late-write",
            archive=archives[0],
            submit_paths=["submission"],
            submit_exclude=[],
            created_at=datetime(2026, 8, 20, 13, 0, tzinfo=timezone.utc),
        )


def test_replay_candidates_dry_run_plans_unique_code_states_without_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _store_root(tmp_path / "store")
    _record_a_b_a(root)
    _patch_config_and_task(
        monkeypatch,
        root=root,
        output=tmp_path / "output",
    )
    monkeypatch.setattr(
        cli,
        "create_backend_from_config",
        lambda config: pytest.fail("dry-run must not create a backend"),
    )

    _run_cli(
        monkeypatch,
        "replay-candidates",
        "--task",
        "task_alpha",
        "--run-id",
        "origin-run",
        "--measurement-set",
        "hidden-20260820",
        "--dry-run",
        "--json",
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["origin_run_id"] == "origin-run"
    assert payload["measurement_set"] == "hidden-20260820"
    assert payload["candidate_count"] == 2
    assert [point["version"] for point in payload["points"]] == ["v1", "v2"]
    assert [point["event_count"] for point in payload["points"]] == [2, 1]
    assert all(point["status"] == "planned" for point in payload["points"])
    assert all(point["score"] is None for point in payload["points"])
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("timeout", ["0", "-1"])
def test_replay_candidates_rejects_nonpositive_timeout_before_output_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    timeout: str,
) -> None:
    root = _store_root(tmp_path / "store")
    _record_a_b_a(root)
    output = tmp_path / "output"
    _patch_config_and_task(monkeypatch, root=root, output=output)
    monkeypatch.setattr(
        cli,
        "create_backend_from_config",
        lambda config: pytest.fail("invalid timeout must not create a backend"),
    )

    with pytest.raises(SystemExit) as raised:
        _run_cli(
            monkeypatch,
            "replay-candidates",
            "--task",
            "task_alpha",
            "--run-id",
            "origin-run",
            "--measurement-set",
            "hidden",
            "--timeout",
            timeout,
        )

    assert raised.value.code == 2
    assert "positive integer" in capsys.readouterr().err
    assert not output.exists()


def test_replay_candidates_json_reports_backend_setup_failure_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _store_root(tmp_path / "store")
    _record_a_b_a(root)
    _patch_config_and_task(
        monkeypatch,
        root=root,
        output=tmp_path / "output",
    )
    monkeypatch.setattr(
        cli,
        "create_backend_from_config",
        lambda config: (_ for _ in ()).throw(
            RuntimeError("private backend details")
        ),
    )

    with pytest.raises(SystemExit) as raised:
        _run_cli(
            monkeypatch,
            "replay-candidates",
            "--task",
            "task_alpha",
            "--run-id",
            "origin-run",
            "--measurement-set",
            "hidden",
            "--json",
        )

    assert raised.value.code == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload == {
        "kind": "sforge-candidate-replay-error",
        "status": "error",
        "error": "failed to initialize replay backend",
    }
    assert "private backend details" not in captured.out
    assert "Traceback" not in captured.err


def test_replay_candidates_json_reports_missing_version_store_as_one_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda overrides=None: SForgeConfig(
            log_dir=tmp_path / "output",
            version_store_root=None,
        ),
    )

    with pytest.raises(SystemExit) as raised:
        _run_cli(
            monkeypatch,
            "replay-candidates",
            "--task",
            "task_alpha",
            "--run-id",
            "origin-run",
            "--measurement-set",
            "hidden",
            "--json",
        )

    assert raised.value.code == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["kind"] == "sforge-candidate-replay-error"
    assert payload["status"] == "error"
    assert "version store" in payload["error"]
    assert captured.out.count("\n{") == 0


def test_replay_candidates_json_normalizes_config_and_task_resolution_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    arguments = (
        "replay-candidates",
        "--task",
        "task_alpha",
        "--run-id",
        "origin-run",
        "--measurement-set",
        "hidden",
        "--json",
    )
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda overrides=None: (_ for _ in ()).throw(
            ValueError("private config parser detail")
        ),
    )
    with pytest.raises(SystemExit) as raised:
        _run_cli(monkeypatch, *arguments)
    assert raised.value.code == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out)["error"] == "failed to load replay configuration"
    assert "private config parser detail" not in captured.out + captured.err

    root = _store_root(tmp_path / "store")
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda overrides=None: SForgeConfig(
            log_dir=tmp_path / "output",
            version_store_root=root,
        ),
    )
    monkeypatch.setattr(
        cli,
        "_resolve_task",
        lambda args, config: (_ for _ in ()).throw(SystemExit(1)),
    )
    with pytest.raises(SystemExit) as raised:
        _run_cli(monkeypatch, *arguments)
    assert raised.value.code == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out)["error"] == "failed to resolve replay task"


def test_replay_candidates_persists_candidate_measurements_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _store_root(tmp_path / "store")
    _record_a_b_a(root)
    output = tmp_path / "output"
    _patch_config_and_task(monkeypatch, root=root, output=output)
    backend = object()
    monkeypatch.setattr(cli, "create_backend_from_config", lambda config: backend)
    calls: list[dict] = []

    def fake_judge_submission(**kwargs):
        calls.append(kwargs)
        assert str(kwargs["log_dir"]).startswith("/proc/self/fd/")
        (kwargs["log_dir"] / "judge-evidence.txt").write_text(
            "pinned\n", encoding="utf-8"
        )
        score = 10.0 if len(calls) == 1 else 15.0
        return EvalReport(
            task_id="task_alpha",
            submission_id=kwargs["submission_id"],
            total_tests=1,
            passed=1,
            pass_rate=1.0,
            score=score,
            valid=True,
            runtime_seconds=1.25,
            submitted_at=1.0,
        )

    monkeypatch.setattr(cli, "judge_submission", fake_judge_submission)
    arguments = (
        "replay-candidates",
        "--task",
        "task_alpha",
        "--run-id",
        "origin-run",
        "--measurement-set",
        "hidden-20260820",
        "--replay-run-id",
        "replay-run",
        "--json",
    )

    _run_cli(monkeypatch, *arguments)

    payload = json.loads(capsys.readouterr().out)
    assert len(calls) == 2
    assert [point["score"] for point in payload["points"]] == [10.0, 15.0]
    assert [point["history_best"] for point in payload["points"]] == [10.0, 15.0]
    assert [point["history_best_version"] for point in payload["points"]] == [
        "v1",
        "v2",
    ]
    assert all(point["status"] == "completed" for point in payload["points"])
    assert payload["completed_count"] == 2
    assert len(list(output.glob("**/judge-evidence.txt"))) == 2

    measurement_paths = sorted(
        path
        for path in output.glob("**/measurement.json")
        if path.parent.name.startswith("candidate-")
    )
    assert len(measurement_paths) == 2
    first = json.loads(measurement_paths[0].read_text(encoding="utf-8"))
    assert first["kind"] == "sforge-candidate-measurement"
    assert first["origin_run_id"] == "origin-run"
    assert first["replay_run_id"] == "replay-run"
    assert first["candidate"]["candidate_id"].startswith("candidate-")
    assert first["candidate"]["sequence"] == 1
    assert first["input_version"]["version_id"]
    assert len(first["input_version"]["archive_sha256"]) == 64
    assert first["evaluator"]["fingerprint"].startswith("sha256:")
    assert first["report"]["score"] == 10.0

    calls.clear()
    _run_cli(monkeypatch, *arguments, "--resume")
    resumed = json.loads(capsys.readouterr().out)
    assert calls == []
    assert resumed["completed_count"] == 2
    assert all(point["status"] == "completed" for point in resumed["points"])


def test_resume_reuses_the_plan_image_without_resolving_a_moved_tag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _store_root(tmp_path / "store")
    _record_a_b_a(root)
    output = tmp_path / "output"
    _patch_config_and_task(monkeypatch, root=root, output=output)
    monkeypatch.setattr(cli, "create_backend_from_config", lambda config: object())
    monkeypatch.setattr(
        cli,
        "judge_submission",
        lambda **kwargs: EvalReport(
            task_id="task_alpha",
            submission_id=kwargs["submission_id"],
            score=1.0,
            pass_rate=1.0,
            valid=True,
        ),
    )
    arguments = (
        "replay-candidates",
        "--task",
        "task_alpha",
        "--run-id",
        "origin-run",
        "--measurement-set",
        "hidden",
        "--replay-run-id",
        "resume-image-run",
        "--json",
    )
    _run_cli(monkeypatch, *arguments)
    capsys.readouterr()

    reused: list[dict] = []
    monkeypatch.setattr(
        cli,
        "_resolve_replay_judge_image",
        lambda *args: pytest.fail("resume must not resolve the mutable tag"),
    )
    monkeypatch.setattr(
        cli,
        "_reuse_replay_judge_image",
        lambda backend, plan, task_spec, config: (
            reused.append(plan) or _resolved_test_image(task_spec)
        ),
        raising=False,
    )

    _run_cli(monkeypatch, *arguments, "--resume")

    payload = json.loads(capsys.readouterr().out)
    assert payload["completed_count"] == 2
    assert len(reused) == 1
    assert reused[0]["kind"] == candidate_replay.PLAN_KIND


def test_reuse_replay_image_fails_closed_when_pinned_content_is_missing() -> None:
    resolved = _resolved_test_image()
    plan = {
        "schema_version": candidate_replay.MEASUREMENT_SCHEMA_VERSION,
        "kind": candidate_replay.PLAN_KIND,
        "evaluator": {
            "resolved_judge_image": {
                "backend": resolved.backend,
                "requested_ref": resolved.requested_ref,
                "execution_ref": resolved.execution_ref,
                "content_identity": resolved.content_identity,
                "identity_kind": resolved.identity_kind,
            }
        },
    }

    class _MissingBackend:
        backend_name = "docker"

        def verify_resolved_image_for_execution(self, image: ResolvedImage) -> None:
            assert image == resolved
            raise RuntimeError("private image-store detail")

    with pytest.raises(
        candidate_replay.CandidateReplayError,
        match="pinned Judge image is no longer executable",
    ) as raised:
        cli._reuse_replay_judge_image(
            _MissingBackend(),
            plan,
            _task_spec(),
            SForgeConfig(),
        )

    assert "private image-store detail" not in str(raised.value)


def test_replay_plan_and_every_judge_use_the_same_resolved_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _store_root(tmp_path / "store")
    _record_a_b_a(root)
    output = tmp_path / "output"
    _patch_config_and_task(monkeypatch, root=root, output=output)
    resolved = _resolved_test_image()
    monkeypatch.setattr(cli, "create_backend_from_config", lambda config: object())
    monkeypatch.setattr(
        cli,
        "_resolve_replay_judge_image",
        lambda backend, task_spec, config: resolved,
        raising=False,
    )
    executed_refs: list[str] = []

    def fake_judge_submission(**kwargs):
        executed_refs.append(kwargs["judge_image_ref"])
        return EvalReport(
            task_id="task_alpha",
            submission_id=kwargs["submission_id"],
            score=1.0,
            pass_rate=1.0,
            valid=True,
        )

    monkeypatch.setattr(cli, "judge_submission", fake_judge_submission)

    _run_cli(
        monkeypatch,
        "replay-candidates",
        "--task",
        "task_alpha",
        "--run-id",
        "origin-run",
        "--measurement-set",
        "hidden",
        "--replay-run-id",
        "pinned-image-run",
        "--json",
    )
    capsys.readouterr()

    assert executed_refs == [resolved.execution_ref, resolved.execution_ref]
    plan_path = next(output.glob("**/replay_plan.json"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["evaluator"]["resolved_judge_image"] == {
        "backend": resolved.backend,
        "requested_ref": resolved.requested_ref,
        "execution_ref": resolved.execution_ref,
        "content_identity": resolved.content_identity,
        "identity_kind": resolved.identity_kind,
    }


def test_replay_submission_ids_are_namespaced_by_curve_identity() -> None:
    common = {
        "candidate_id": "candidate-" + "a" * 64,
        "attempt": 1,
    }

    first = candidate_replay.replay_submission_id(
        replay_run_id="replay-a",
        measurement_set="hidden-a",
        **common,
    )
    different_run = candidate_replay.replay_submission_id(
        replay_run_id="replay-b",
        measurement_set="hidden-a",
        **common,
    )
    different_set = candidate_replay.replay_submission_id(
        replay_run_id="replay-a",
        measurement_set="hidden-b",
        **common,
    )

    assert len({first, different_run, different_set}) == 3
    assert all(value.startswith("replay-") for value in (first, different_run, different_set))
    assert all(len(value) < 64 for value in (first, different_run, different_set))


def test_replay_plan_rejects_a_later_same_candidate_timeline_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _store_root(tmp_path / "store")
    archives = _record_a_b_a(root)
    output = tmp_path / "output"
    _patch_config_and_task(monkeypatch, root=root, output=output)
    monkeypatch.setattr(cli, "create_backend_from_config", lambda config: object())
    monkeypatch.setattr(
        cli,
        "judge_submission",
        lambda **kwargs: EvalReport(
            task_id="task_alpha",
            submission_id=kwargs["submission_id"],
            score=1.0,
            pass_rate=1.0,
            valid=True,
        ),
    )
    arguments = (
        "replay-candidates",
        "--task",
        "task_alpha",
        "--run-id",
        "origin-run",
        "--measurement-set",
        "hidden",
        "--replay-run-id",
        "replay-run",
        "--json",
    )
    _run_cli(monkeypatch, *arguments)
    capsys.readouterr()

    VersionStore.open(root).record_archive(
        task_id="task_alpha",
        run_id="origin-run",
        source="checkpoint",
        round_id="checkpoint-4",
        label="same-a-after-plan",
        archive=archives[0],
        submit_paths=["submission"],
        submit_exclude=[],
        created_at=datetime(2026, 8, 20, 12, 4, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(
        cli,
        "judge_submission",
        lambda **kwargs: pytest.fail("changed timeline must fail before Judge"),
    )

    with pytest.raises(SystemExit) as raised:
        _run_cli(monkeypatch, *arguments, "--resume")

    assert raised.value.code == 2
    assert "existing replay plan changed" in capsys.readouterr().err


def test_replay_marks_curve_stale_if_origin_changes_during_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _store_root(tmp_path / "store")
    archives = _record_a_b_a(root)
    _record_final(root, archives[0])
    output = tmp_path / "output"
    _patch_config_and_task(
        monkeypatch,
        root=root,
        output=output,
        allow_open_origin=False,
    )
    monkeypatch.setattr(cli, "create_backend_from_config", lambda config: object())
    calls = 0
    origin_changed = False
    real_list_candidates = VersionStore.list_candidates

    def list_candidates(self, *, task=None, run_id=None):
        if origin_changed and run_id == "origin-run":
            return []
        return real_list_candidates(self, task=task, run_id=run_id)

    monkeypatch.setattr(VersionStore, "list_candidates", list_candidates)

    def fake_judge_submission(**kwargs):
        nonlocal calls, origin_changed
        calls += 1
        origin_changed = True
        return EvalReport(
            task_id="task_alpha",
            submission_id=kwargs["submission_id"],
            score=1.0,
            pass_rate=1.0,
            valid=True,
        )

    monkeypatch.setattr(cli, "judge_submission", fake_judge_submission)

    with pytest.raises(SystemExit) as raised:
        _run_cli(
            monkeypatch,
            "replay-candidates",
            "--task",
            "task_alpha",
            "--run-id",
            "origin-run",
            "--measurement-set",
            "hidden",
            "--replay-run-id",
            "source-race-run",
            "--json",
        )

    assert raised.value.code == 1
    assert calls == 1
    captured = capsys.readouterr()
    curve = json.loads(captured.out)
    assert curve["terminal_error"]["kind"] == "source_run_changed"
    assert curve["completed_count"] == 1
    assert "origin run changed" in captured.err


def test_replay_candidates_rejects_existing_measurements_without_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _store_root(tmp_path / "store")
    _record_a_b_a(root)
    output = tmp_path / "output"
    _patch_config_and_task(monkeypatch, root=root, output=output)
    monkeypatch.setattr(cli, "create_backend_from_config", lambda config: object())
    monkeypatch.setattr(
        cli,
        "judge_submission",
        lambda **kwargs: EvalReport(
            task_id="task_alpha",
            submission_id=kwargs["submission_id"],
            score=1.0,
            valid=True,
        ),
    )
    arguments = (
        "replay-candidates",
        "--task",
        "task_alpha",
        "--run-id",
        "origin-run",
        "--measurement-set",
        "hidden",
        "--replay-run-id",
        "replay-run",
    )
    _run_cli(monkeypatch, *arguments)
    capsys.readouterr()

    with pytest.raises(SystemExit) as raised:
        _run_cli(monkeypatch, *arguments)

    assert raised.value.code == 2
    assert "--resume" in capsys.readouterr().err


def test_existing_measurement_validation_rejects_incomplete_or_mismatched_state(
    tmp_path: Path,
) -> None:
    root = _store_root(tmp_path / "store")
    _record_a_b_a(root)
    candidate = VersionStore.open(root).list_candidates(
        task="task_alpha", run_id="origin-run"
    )[0]
    version = candidate_replay.input_version(candidate)
    evaluator = candidate_replay.evaluator_descriptor(
        _task_spec(), SForgeConfig(), timeout=None
    )
    completed = candidate_replay.measurement_record(
        candidate=candidate,
        version=version,
        origin_run_id="origin-run",
        replay_run_id="replay-run",
        measurement_set="hidden",
        evaluator=evaluator,
        attempt=1,
        status="completed",
        started_at="2026-08-20T12:00:00Z",
        completed_at="2026-08-20T12:01:00Z",
        report={
            "task_id": "task_alpha",
            "submission_id": candidate_replay.replay_submission_id(
                replay_run_id="replay-run",
                measurement_set="hidden",
                candidate_id=candidate.candidate_id,
                attempt=1,
            ),
            "score": 1.0,
            "pass_rate": 1.0,
            "valid": True,
        },
    )
    invalid_records = []
    for mutation in (
        "missing-report",
        "missing-task",
        "missing-submission",
        "zero-attempt",
        "wrong-archive",
        "fake-evaluator",
    ):
        value = deepcopy(completed)
        if mutation == "missing-report":
            value["report"] = None
        elif mutation == "missing-task":
            value["report"].pop("task_id")
        elif mutation == "missing-submission":
            value["report"].pop("submission_id")
        elif mutation == "zero-attempt":
            value["attempt"] = 0
        elif mutation == "wrong-archive":
            value["input_version"]["archive_sha256"] = "0" * 64
        else:
            value["evaluator"]["judge_image"] = "different"
        invalid_records.append(value)

    for value in invalid_records:
        with pytest.raises(candidate_replay.CandidateReplayError):
            candidate_replay.validate_existing_measurement(
                value,
                candidate=candidate,
                origin_run_id="origin-run",
                replay_run_id="replay-run",
                measurement_set="hidden",
                evaluator=evaluator,
            )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("selection", "unknown-policy", "selection policy"),
        ("selection", ["score_first"], "selection policy"),
        ("score_direction", "sideways", "score direction"),
        ("score_direction", {"maximize": True}, "score direction"),
        ("eval_timeout", 0, "timeout"),
    ],
)
def test_evaluator_descriptor_rejects_invalid_curve_contract(
    field: str,
    value: object,
    message: str,
) -> None:
    task = _task_spec()
    setattr(task.judge, field, value)

    with pytest.raises(candidate_replay.CandidateReplayError, match=message):
        candidate_replay.evaluator_descriptor(
            task,
            SForgeConfig(),
            timeout=None,
        )


def test_replay_candidates_rejects_kubernetes_before_backend_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _store_root(tmp_path / "store")
    output = tmp_path / "output"
    config = SForgeConfig(
        backend="k8s",
        version_store_root=root,
        log_dir=output,
    )
    monkeypatch.setattr(cli, "load_config", lambda overrides=None: config)
    monkeypatch.setattr(
        cli,
        "_resolve_task",
        lambda args, resolved_config: _task_spec(),
    )
    monkeypatch.setattr(
        cli,
        "create_backend_from_config",
        lambda config: pytest.fail("unsupported K8s replay must not create backend"),
    )

    with pytest.raises(SystemExit) as raised:
        _run_cli(
            monkeypatch,
            "replay-candidates",
            "--task",
            "task_alpha",
            "--run-id",
            "origin-run",
            "--measurement-set",
            "hidden",
            "--dry-run",
            "--json",
        )

    assert raised.value.code == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out)["status"] == "error"
    assert "requires the Docker backend" in captured.err
    assert not output.exists()


def test_resume_rejects_nonfinite_measurement_before_backend_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _store_root(tmp_path / "store")
    _record_a_b_a(root)
    output = tmp_path / "output"
    _patch_config_and_task(monkeypatch, root=root, output=output)
    monkeypatch.setattr(cli, "create_backend_from_config", lambda config: object())
    monkeypatch.setattr(
        cli,
        "judge_submission",
        lambda **kwargs: EvalReport(
            task_id="task_alpha",
            submission_id=kwargs["submission_id"],
            score=1.0,
            score_0_100=50.0,
            pass_rate=1.0,
            valid=True,
        ),
    )
    arguments = (
        "replay-candidates",
        "--task",
        "task_alpha",
        "--run-id",
        "origin-run",
        "--measurement-set",
        "hidden",
        "--replay-run-id",
        "nonfinite-run",
        "--candidate",
        "v1",
        "--json",
    )
    _run_cli(monkeypatch, *arguments)
    capsys.readouterr()
    top = next(
        path
        for path in output.glob("**/measurement.json")
        if path.parent.name.startswith("candidate-")
    )
    payload = json.loads(top.read_text(encoding="utf-8"))
    payload["report"]["score_0_100"] = float("nan")
    top.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        cli,
        "create_backend_from_config",
        lambda config: pytest.fail("non-finite JSON must fail before backend"),
    )

    with pytest.raises(SystemExit) as raised:
        _run_cli(monkeypatch, *arguments, "--resume")

    assert raised.value.code == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out)["status"] == "error"
    assert "unreadable" in captured.err or "non-finite" in captured.err


def test_replay_json_depth_failures_are_normalized(
    tmp_path: Path,
) -> None:
    nested: dict[str, object] = {}
    cursor = nested
    for _ in range(1_500):
        child: dict[str, object] = {}
        cursor["nested"] = child
        cursor = child

    with pytest.raises(
        candidate_replay.CandidateReplayError,
        match="failed to encode replay JSON",
    ):
        candidate_replay._encode_replay_json(nested)

    lease = candidate_replay.acquire_replay_lock(tmp_path / "replay")
    try:
        stored = lease.root / "deep.json"
        stored.write_text(
            '{"nested":' * 1_500 + "null" + "}" * 1_500,
            encoding="utf-8",
        )
        with pytest.raises(
            candidate_replay.CandidateReplayError,
            match="unreadable",
        ):
            candidate_replay._read_json(
                stored,
                label="deep replay JSON",
                missing_ok=False,
            )
    finally:
        lease.close()


@pytest.mark.parametrize(
    "invalid_kind",
    [
        "fields",
        "nested-nan",
        "nested-object",
        "oversized",
        "pretty-overflow",
    ],
)
def test_first_invalid_terminal_report_is_persisted_as_retryable_infrastructure(
    invalid_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _store_root(tmp_path / "store")
    _record_a_b_a(root)
    output = tmp_path / "output"
    _patch_config_and_task(monkeypatch, root=root, output=output)
    monkeypatch.setattr(cli, "create_backend_from_config", lambda config: object())
    if invalid_kind in {"oversized", "pretty-overflow"}:
        monkeypatch.setattr(candidate_replay, "_MAX_REPLAY_JSON_BYTES", 64 * 1024)

    def invalid_report(**kwargs: object) -> EvalReport:
        report = EvalReport(
            task_id="task_alpha",
            submission_id=kwargs["submission_id"],
            score=1.0,
            score_0_100=50.0,
            pass_rate=1.0,
            valid=True,
        )
        if invalid_kind == "fields":
            report.score = True
            report.score_0_100 = 101.0
            report.pass_rate = 2.0
        elif invalid_kind == "nested-nan":
            report.metrics = {"nested": {"value": float("nan")}}
        elif invalid_kind == "nested-object":
            report.metrics = {"nested": {"value": object()}}
        elif invalid_kind == "oversized":
            report.metrics = {"oversized": "x" * (128 * 1024)}
        else:
            report.metrics = {"items": ["x"] * 10_000}
        return report

    monkeypatch.setattr(cli, "judge_submission", invalid_report)

    with pytest.raises(SystemExit) as raised:
        _run_cli(
            monkeypatch,
            "replay-candidates",
            "--task",
            "task_alpha",
            "--run-id",
            "origin-run",
            "--measurement-set",
            "hidden",
            "--replay-run-id",
            "invalid-report-run",
            "--candidate",
            "v1",
            "--json",
        )

    assert raised.value.code == 1
    curve = json.loads(capsys.readouterr().out)
    assert curve["terminal_error"]["kind"] == "infrastructure_failure"
    point = curve["points"][0]
    assert point["status"] == "failed"
    assert point["score"] is None
    top = next(
        path
        for path in output.glob("**/measurement.json")
        if path.parent.name.startswith("candidate-")
    )
    measurement = json.loads(top.read_text(encoding="utf-8"))
    assert measurement["status"] == "failed"
    assert measurement["report"]["metrics"] == {
        "evaluation_status": "infrastructure_error",
        "infrastructure_reason": "invalid_terminal_report",
    }
    assert measurement["report"]["score"] == 0.0
    assert measurement["report"]["score_0_100"] is None
    assert measurement["report"]["pass_rate"] == 0.0


@pytest.mark.parametrize("failure_stage", ["judge", "to_dict", "non-mapping"])
def test_evaluator_exception_is_persisted_as_retryable_infrastructure(
    failure_stage: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _store_root(tmp_path / "store")
    _record_a_b_a(root)
    output = tmp_path / "output"
    _patch_config_and_task(monkeypatch, root=root, output=output)
    monkeypatch.setattr(cli, "create_backend_from_config", lambda config: object())

    class BrokenReport:
        def to_dict(self) -> object:
            if failure_stage == "to_dict":
                raise RuntimeError("private evaluator detail")
            return []

    def broken_judge(**kwargs: object) -> object:
        if failure_stage == "judge":
            raise RuntimeError("private evaluator detail")
        return BrokenReport()

    monkeypatch.setattr(cli, "judge_submission", broken_judge)

    with pytest.raises(SystemExit) as raised:
        _run_cli(
            monkeypatch,
            "replay-candidates",
            "--task",
            "task_alpha",
            "--run-id",
            "origin-run",
            "--measurement-set",
            "hidden",
            "--replay-run-id",
            f"{failure_stage}-exception-run",
            "--candidate",
            "v1",
            "--json",
        )

    assert raised.value.code == 1
    captured = capsys.readouterr()
    assert "private evaluator detail" not in captured.out
    assert "private evaluator detail" not in captured.err
    curve = json.loads(captured.out)
    assert curve["points"][0]["status"] == "failed"
    top = next(
        path
        for path in output.glob("**/measurement.json")
        if path.parent.name.startswith("candidate-")
    )
    measurement = json.loads(top.read_text(encoding="utf-8"))
    assert measurement["report"]["metrics"]["infrastructure_reason"] == (
        "evaluator_exception"
    )


def test_replay_candidates_retries_infrastructure_failure_as_a_new_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _store_root(tmp_path / "store")
    _record_a_b_a(root)
    output = tmp_path / "output"
    _patch_config_and_task(monkeypatch, root=root, output=output)
    monkeypatch.setattr(cli, "create_backend_from_config", lambda config: object())
    calls: list[dict] = []

    def fake_judge_submission(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return EvalReport(
                task_id="task_alpha",
                submission_id=kwargs["submission_id"],
                score=0.0,
                valid=False,
                errors=1,
                metrics={"evaluation_status": "infrastructure_error"},
            )
        return EvalReport(
            task_id="task_alpha",
            submission_id=kwargs["submission_id"],
            total_tests=1,
            passed=1,
            pass_rate=1.0,
            score=float(len(calls)),
            valid=True,
        )

    monkeypatch.setattr(cli, "judge_submission", fake_judge_submission)
    arguments = (
        "replay-candidates",
        "--task",
        "task_alpha",
        "--run-id",
        "origin-run",
        "--measurement-set",
        "hidden",
        "--replay-run-id",
        "replay-run",
        "--json",
    )

    with pytest.raises(SystemExit) as raised:
        _run_cli(monkeypatch, *arguments)
    assert raised.value.code == 1
    failed_output = capsys.readouterr()
    assert "infrastructure failed" in failed_output.err
    failed_curve = json.loads(failed_output.out)
    assert failed_curve["terminal_error"] == {
        "kind": "infrastructure_failure",
        "candidate": "v1",
        "message": "replay infrastructure failed; resume after fixing the evaluator",
    }
    assert [point["status"] for point in failed_curve["points"]] == [
        "failed",
        "planned",
    ]

    _run_cli(monkeypatch, *arguments, "--resume")
    payload = json.loads(capsys.readouterr().out)
    assert len(calls) == 3
    assert payload["completed_count"] == 2
    first_path = next(
        path
        for path in output.glob("**/measurement.json")
        if path.parent.name.startswith("candidate-")
        and json.loads(path.read_text(encoding="utf-8"))["candidate"]["sequence"]
        == 1
    )
    current = json.loads(first_path.read_text(encoding="utf-8"))
    assert current["attempt"] == 2
    assert current["status"] == "completed"
    assert (first_path.parent / "attempt-001" / "measurement.json").exists()
    assert (first_path.parent / "attempt-002" / "measurement.json").exists()


def test_resume_promotes_a_durable_terminal_attempt_without_rejudging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _store_root(tmp_path / "store")
    _record_a_b_a(root)
    output = tmp_path / "output"
    _patch_config_and_task(monkeypatch, root=root, output=output)
    monkeypatch.setattr(cli, "create_backend_from_config", lambda config: object())
    judge_calls = 0

    def fake_judge_submission(**kwargs):
        nonlocal judge_calls
        judge_calls += 1
        return EvalReport(
            task_id="task_alpha",
            submission_id=kwargs["submission_id"],
            score=17.0,
            pass_rate=1.0,
            valid=True,
        )

    monkeypatch.setattr(cli, "judge_submission", fake_judge_submission)
    real_write = candidate_replay.write_measurement
    crash = True

    def crash_before_top_level_promotion(path: Path, payload: dict) -> None:
        nonlocal crash
        if (
            crash
            and path.parent.name.startswith("candidate-")
            and payload.get("status") == "completed"
        ):
            crash = False
            raise RuntimeError("simulated process loss after durable attempt")
        real_write(path, payload)

    monkeypatch.setattr(
        candidate_replay,
        "write_measurement",
        crash_before_top_level_promotion,
    )
    arguments = (
        "replay-candidates",
        "--task",
        "task_alpha",
        "--run-id",
        "origin-run",
        "--measurement-set",
        "hidden",
        "--replay-run-id",
        "promotion-run",
        "--candidate",
        "v1",
        "--json",
    )

    with pytest.raises(RuntimeError, match="simulated process loss"):
        _run_cli(monkeypatch, *arguments)
    assert judge_calls == 1
    capsys.readouterr()
    monkeypatch.setattr(candidate_replay, "write_measurement", real_write)
    monkeypatch.setattr(
        cli,
        "judge_submission",
        lambda **kwargs: pytest.fail("durable terminal attempt must be promoted"),
    )

    _run_cli(monkeypatch, *arguments, "--resume")

    curve = json.loads(capsys.readouterr().out)
    assert judge_calls == 1
    assert curve["completed_count"] == 1
    point = curve["points"][0]
    assert point["status"] == "completed"
    assert point["score"] == 17.0
    top = next(
        path
        for path in output.glob("**/measurement.json")
        if path.parent.name.startswith("candidate-")
    )
    assert json.loads(top.read_text(encoding="utf-8"))["status"] == "completed"


def test_replay_candidates_verifies_archives_before_creating_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _store_root(tmp_path / "store")
    _record_a_b_a(root)
    first = VersionStore.open(root).list_versions()[0]
    first.archive_path.write_bytes(b"corrupt")
    _patch_config_and_task(
        monkeypatch,
        root=root,
        output=tmp_path / "output",
    )
    monkeypatch.setattr(
        cli,
        "create_backend_from_config",
        lambda config: pytest.fail("corrupt archives must fail before backend creation"),
    )

    with pytest.raises(SystemExit) as raised:
        _run_cli(
            monkeypatch,
            "replay-candidates",
            "--task",
            "task_alpha",
            "--run-id",
            "origin-run",
            "--measurement-set",
            "hidden",
        )

    assert raised.value.code == 2
    assert "integrity" in capsys.readouterr().err


def test_candidate_replay_lock_rejects_concurrent_runner(tmp_path: Path) -> None:
    root = tmp_path / "replay"
    first = candidate_replay.acquire_replay_lock(root)
    try:
        with pytest.raises(candidate_replay.CandidateReplayError, match="another process"):
            candidate_replay.acquire_replay_lock(root)
    finally:
        first.close()

    replacement = candidate_replay.acquire_replay_lock(root)
    replacement.close()


def test_candidate_replay_root_rejects_symlinked_output_component(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    outside = tmp_path / "outside"
    output.mkdir()
    outside.mkdir()
    (output / "runs").symlink_to(outside, target_is_directory=True)
    root = output / "runs" / "replay-run" / "task" / "candidate-replays"

    with pytest.raises(
        candidate_replay.CandidateReplayError,
        match="non-symlink directory",
    ):
        candidate_replay.acquire_replay_lock(root, base=output)

    assert list(outside.iterdir()) == []


def test_candidate_replay_uses_pinned_root_after_path_retarget(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    outside = tmp_path / "outside"
    outside.mkdir()
    root = output / "replay"
    lease = candidate_replay.acquire_replay_lock(root, base=output)
    try:
        moved = output / "moved"
        root.rename(moved)
        root.symlink_to(outside, target_is_directory=True)
        candidate_replay.write_measurement(
            lease.root / "curve.json",
            {"kind": "pinned-root"},
        )
    finally:
        lease.close()

    assert json.loads((moved / "curve.json").read_text()) == {
        "kind": "pinned-root"
    }
    assert list(outside.iterdir()) == []


def test_candidate_replay_root_rename_cannot_split_one_logical_lease(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    root = output / "replay"
    first = candidate_replay.acquire_replay_lock(root, base=output)
    try:
        root.rename(output / "moved-replay")
        root.mkdir()
        with pytest.raises(
            candidate_replay.CandidateReplayError,
            match="another process",
        ):
            candidate_replay.acquire_replay_lock(root, base=output)
    finally:
        first.close()

    replacement = candidate_replay.acquire_replay_lock(root, base=output)
    replacement.close()


def test_candidate_replay_refuses_symlinked_plan_file(tmp_path: Path) -> None:
    output = tmp_path / "output"
    root = output / "replay"
    target = tmp_path / "target.json"
    target.write_text('{"do_not_touch": true}\n', encoding="utf-8")
    lease = candidate_replay.acquire_replay_lock(root, base=output)
    try:
        (root / "replay_plan.json").symlink_to(target)
        with pytest.raises(
            candidate_replay.CandidateReplayError,
            match="regular non-symlink",
        ):
            candidate_replay.ensure_replay_plan(
                lease.root,
                {"kind": candidate_replay.PLAN_KIND},
            )
    finally:
        lease.close()

    assert json.loads(target.read_text()) == {"do_not_touch": True}


def test_candidate_replay_attempt_directory_must_be_new_and_empty(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    lease = candidate_replay.acquire_replay_lock(
        output / "replay",
        base=output,
    )
    try:
        attempt = lease.root / "candidate-a" / "attempt-001"
        with candidate_replay.pin_replay_directory(
            attempt,
            must_create=True,
        ):
            pass
        with pytest.raises(
            candidate_replay.CandidateReplayError,
            match="already exists",
        ):
            candidate_replay.pin_replay_directory(
                attempt,
                must_create=True,
            )
    finally:
        lease.close()


def test_replay_rejects_preplanted_attempt_children_before_judge_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store_root = _store_root(tmp_path / "store")
    _record_a_b_a(store_root)
    store = VersionStore.open(store_root)
    candidate = store.list_candidates(
        task="task_alpha",
        run_id="origin-run",
    )[0]
    output = tmp_path / "output"
    _patch_config_and_task(
        monkeypatch,
        root=store_root,
        output=output,
    )
    replay_root = candidate_replay.replay_root(
        output,
        replay_run_id="preplant-run",
        task_id="task_alpha",
        origin_run_id="origin-run",
        measurement_set="hidden",
    )
    attempt = replay_root / candidate.candidate_id / "attempt-001"
    attempt.mkdir(parents=True)
    victim = tmp_path / "victim"
    victim.write_text("keep\n", encoding="utf-8")
    (attempt / "submission.tar.gz").symlink_to(victim)
    monkeypatch.setattr(cli, "create_backend_from_config", lambda config: object())
    monkeypatch.setattr(
        cli,
        "judge_submission",
        lambda **kwargs: pytest.fail("preplanted attempt must block Judge"),
    )

    with pytest.raises(SystemExit) as raised:
        _run_cli(
            monkeypatch,
            "replay-candidates",
            "--task",
            "task_alpha",
            "--run-id",
            "origin-run",
            "--measurement-set",
            "hidden",
            "--replay-run-id",
            "preplant-run",
            "--candidate",
            "v1",
            "--json",
        )

    assert raised.value.code == 2
    assert "already exists" in capsys.readouterr().err
    assert victim.read_text(encoding="utf-8") == "keep\n"


def test_replay_candidates_resume_rejects_changed_evaluator_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _store_root(tmp_path / "store")
    _record_a_b_a(root)
    output = tmp_path / "output"
    _patch_config_and_task(monkeypatch, root=root, output=output)
    monkeypatch.setattr(cli, "create_backend_from_config", lambda config: object())
    monkeypatch.setattr(
        cli,
        "judge_submission",
        lambda **kwargs: EvalReport(
            task_id="task_alpha",
            submission_id=kwargs["submission_id"],
            score=1.0,
            pass_rate=1.0,
            valid=True,
        ),
    )
    arguments = (
        "replay-candidates",
        "--task",
        "task_alpha",
        "--source-run-id",
        "origin-run",
        "--measurement-set",
        "hidden",
        "--replay-run-id",
        "replay-run",
    )
    _run_cli(monkeypatch, *arguments)
    capsys.readouterr()

    monkeypatch.setattr(
        cli,
        "_resolve_task",
        lambda args, config: _task_spec(score_direction="minimize"),
    )
    monkeypatch.setattr(
        cli,
        "create_backend_from_config",
        lambda config: object(),
    )
    monkeypatch.setattr(
        cli,
        "judge_submission",
        lambda **kwargs: pytest.fail(
            "fingerprint mismatch must fail before Judge execution"
        ),
    )
    with pytest.raises(SystemExit) as raised:
        _run_cli(monkeypatch, *arguments, "--resume")

    assert raised.value.code == 2
    assert "existing replay plan changed" in capsys.readouterr().err


def test_evaluator_context_is_canonical_and_changes_fingerprint() -> None:
    config = SForgeConfig()
    plain = candidate_replay.evaluator_descriptor(
        _task_spec(), config, timeout=None
    )
    contextual = candidate_replay.evaluator_descriptor(
        _task_spec(),
        config,
        timeout=None,
        runtime_context={"z_asset": "sha256:z", "a_adapter": "sha256:a"},
    )

    assert "runtime_context" not in plain
    assert contextual["runtime_context"] == {
        "a_adapter": "sha256:a",
        "z_asset": "sha256:z",
    }
    assert contextual["fingerprint"] != plain["fingerprint"]


def test_live_evaluator_pins_core_contract_and_execution_image() -> None:
    task_spec = _task_spec()
    resolved = ResolvedImage(
        backend="docker",
        requested_ref=task_spec.judge_image_key,
        execution_ref="sha256:" + "a" * 64,
        content_identity="sha256:" + "a" * 64,
        identity_kind="docker-image-id",
    )

    descriptor = candidate_replay.evaluator_descriptor(
        task_spec,
        SForgeConfig(),
        timeout=None,
        resolved_image=resolved,
    )

    assert descriptor["contract_schema_version"] == 3
    assert descriptor["task_contract_sha256"].startswith("sha256:")
    assert descriptor["backend_contract_sha256"].startswith("sha256:")
    assert descriptor["judge_environment_sha256"].startswith("sha256:")
    assert descriptor["evaluator_source_sha256"].startswith("sha256:")
    assert descriptor["resolved_judge_image"] == {
        "backend": "docker",
        "requested_ref": task_spec.judge_image_key,
        "execution_ref": "sha256:" + "a" * 64,
        "content_identity": "sha256:" + "a" * 64,
        "identity_kind": "docker-image-id",
    }


def test_evaluator_source_identity_covers_the_full_sforge_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = candidate_replay._evaluator_source_sha256()
    real_read_bytes = Path.read_bytes

    def changed_runtime(path: Path) -> bytes:
        payload = real_read_bytes(path)
        if path.name == "constants.py" and path.parent.name == "harness":
            return payload + b"\n# simulated evaluator contract change\n"
        return payload

    monkeypatch.setattr(Path, "read_bytes", changed_runtime)

    assert candidate_replay._evaluator_source_sha256() != baseline


def test_evaluator_descriptor_rejects_source_that_changes_while_pinning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identities = iter(("sha256:" + "a" * 64, "sha256:" + "b" * 64))
    monkeypatch.setattr(
        candidate_replay,
        "_evaluator_source_sha256",
        lambda: next(identities),
    )

    with pytest.raises(
        candidate_replay.CandidateReplayError,
        match="changed while its identity was being pinned",
    ):
        candidate_replay.evaluator_descriptor(
            _task_spec(),
            SForgeConfig(),
            timeout=None,
        )


def test_evaluator_fingerprint_commits_already_loaded_code_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_identity = "sha256:" + "a" * 64
    monkeypatch.setattr(
        candidate_replay,
        "_evaluator_source_sha256",
        lambda: "sha256:" + "f" * 64,
    )
    monkeypatch.setattr(
        candidate_replay,
        "_loaded_sforge_module_names",
        lambda: ["sforge.cli", "sforge.harness.run_evaluation"],
    )
    monkeypatch.setattr(
        candidate_replay,
        "_loaded_evaluator_code_sha256",
        lambda modules=None: loaded_identity,
    )

    first = candidate_replay.evaluator_descriptor(
        _task_spec(), SForgeConfig(), timeout=None
    )
    loaded_identity = "sha256:" + "b" * 64
    second = candidate_replay.evaluator_descriptor(
        _task_spec(), SForgeConfig(), timeout=None
    )

    assert first["evaluator_source_sha256"] == second["evaluator_source_sha256"]
    assert first["loaded_evaluator_code_sha256"] != second[
        "loaded_evaluator_code_sha256"
    ]
    assert first["fingerprint"] != second["fingerprint"]


def test_loaded_code_identity_commits_function_runtime_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "sforge._candidate_replay_identity_test"
    module = types.ModuleType(module_name)
    module.__dict__["__name__"] = module_name
    monkeypatch.setitem(sys.modules, module_name, module)

    def default_template(value=1):
        return value

    module.function = types.FunctionType(
        default_template.__code__, module.__dict__, "function", (1,)
    )
    default_one = candidate_replay._loaded_evaluator_code_sha256([module_name])
    module.function = types.FunctionType(
        default_template.__code__, module.__dict__, "function", (2,)
    )
    default_two = candidate_replay._loaded_evaluator_code_sha256([module_name])
    assert default_one != default_two

    module.function.__kwdefaults__ = {"mode": "a"}
    kw_a = candidate_replay._loaded_evaluator_code_sha256([module_name])
    module.function.__kwdefaults__ = {"mode": "b"}
    kw_b = candidate_replay._loaded_evaluator_code_sha256([module_name])
    assert kw_a != kw_b

    def closure_factory(bound: int):
        def closure():
            return bound

        return closure

    module.function = closure_factory(1)
    closure_one = candidate_replay._loaded_evaluator_code_sha256([module_name])
    module.function = closure_factory(2)
    closure_two = candidate_replay._loaded_evaluator_code_sha256([module_name])
    assert closure_one != closure_two

    exec("policy = 1\ndef function():\n    return policy\n", module.__dict__)
    global_one = candidate_replay._loaded_evaluator_code_sha256([module_name])
    module.policy = 2
    global_two = candidate_replay._loaded_evaluator_code_sha256([module_name])
    assert global_one != global_two

    exec(
        "policy = {'threshold': 1}\n"
        "def function(value):\n"
        "    return value >= policy['threshold']\n",
        module.__dict__,
    )
    mapping_one = candidate_replay._loaded_evaluator_code_sha256([module_name])
    module.policy = {"threshold": 2}
    mapping_two = candidate_replay._loaded_evaluator_code_sha256([module_name])
    assert mapping_one != mapping_two


def test_code_identity_supports_python_310_code_fields() -> None:
    code = types.SimpleNamespace(
        co_argcount=0,
        co_posonlyargcount=0,
        co_kwonlyargcount=0,
        co_nlocals=0,
        co_stacksize=1,
        co_flags=0,
        co_code=b"",
        co_consts=(),
        co_names=(),
        co_varnames=(),
        co_freevars=(),
        co_cellvars=(),
        co_name="function",
        co_firstlineno=1,
        co_linetable=b"",
    )

    value = candidate_replay._canonical_code_value(code)

    assert value["qualname"] == "function"
    assert value["exceptiontable"] == ""


def test_loaded_code_identity_commits_regex_path_nested_and_class_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "sforge._candidate_replay_object_identity_test"
    module = types.ModuleType(module_name)
    module.__dict__.update({"__name__": module_name, "re": re, "Path": Path})
    monkeypatch.setitem(sys.modules, module_name, module)

    exec(
        "PATTERN = re.compile('a')\n"
        "ROOT = Path('/first')\n"
        "def accepts(value):\n"
        "    return bool(PATTERN.fullmatch(value)) and ROOT.is_absolute()\n",
        module.__dict__,
    )
    regex_a = candidate_replay._loaded_evaluator_code_sha256([module_name])
    module.PATTERN = re.compile("b")
    regex_b = candidate_replay._loaded_evaluator_code_sha256([module_name])
    assert regex_a != regex_b

    path_first = candidate_replay._loaded_evaluator_code_sha256([module_name])
    module.ROOT = tmp_path / "second"
    path_second = candidate_replay._loaded_evaluator_code_sha256([module_name])
    assert path_first != path_second

    exec(
        "policy = 1\n"
        "def outer():\n"
        "    def inner():\n"
        "        return policy\n"
        "    return inner()\n",
        module.__dict__,
    )
    nested_one = candidate_replay._loaded_evaluator_code_sha256([module_name])
    module.policy = 2
    nested_two = candidate_replay._loaded_evaluator_code_sha256([module_name])
    assert nested_one != nested_two

    exec(
        "class Policy:\n"
        "    threshold = 1\n"
        "    def accepts(self, value):\n"
        "        return value >= self.threshold\n",
        module.__dict__,
    )
    class_one = candidate_replay._loaded_evaluator_code_sha256([module_name])
    module.Policy.threshold = 2
    class_two = candidate_replay._loaded_evaluator_code_sha256([module_name])
    assert class_one != class_two


def test_runtime_adapter_snapshot_commits_sensitive_named_helper_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = tmp_path / "runtime_entry.py"
    source = (
        "def _configured_nonce_source():\n"
        "    return 'first'\n"
        "def verify():\n"
        "    return _configured_nonce_source()\n"
    )
    adapter.write_text(source, encoding="utf-8")
    module = types.ModuleType("__main__")
    module.__dict__.update({"__name__": "__main__", "__file__": str(adapter)})
    exec(source, module.__dict__)
    monkeypatch.setitem(sys.modules, "__main__", module)
    expected_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()

    first = candidate_replay.loaded_evaluator_snapshot(
        ["__main__"],
        expected_main_sha256=expected_sha256,
    )
    exec(
        "def _configured_nonce_source():\n"
        "    return 'second'\n",
        module.__dict__,
    )
    second = candidate_replay.loaded_evaluator_snapshot(
        ["__main__"],
        expected_main_sha256=expected_sha256,
    )

    assert first["runtime_sha256"] != second["runtime_sha256"]
    assert first["bytecode_sha256"] != second["bytecode_sha256"]


def test_loaded_snapshot_precedes_mutable_build_cache_and_resumes_fresh() -> None:
    from sforge.harness import docker_build

    original_locks = dict(docker_build._base_image_locks)
    original_rebuilt = set(docker_build._base_image_rebuilt)
    original_pushed = set(docker_build._pushed_base_refs)
    startup = candidate_replay.loaded_evaluator_snapshot()
    try:
        docker_build._base_image_locks["loaded-identity-probe"] = threading.Lock()
        docker_build._base_image_rebuilt.add("loaded-identity-probe")
        docker_build._pushed_base_refs.add("loaded-identity-probe")
        mutated = candidate_replay.loaded_evaluator_snapshot()

        assert startup["runtime_sha256"] != mutated["runtime_sha256"]
        assert startup["bytecode_sha256"] == mutated["bytecode_sha256"]
        descriptor = candidate_replay.evaluator_descriptor(
            _task_spec(),
            SForgeConfig(),
            timeout=None,
            loaded_snapshot=startup,
        )
        assert descriptor["loaded_evaluator_code_sha256"] == startup[
            "runtime_sha256"
        ]
    finally:
        docker_build._base_image_locks.clear()
        docker_build._base_image_locks.update(original_locks)
        docker_build._base_image_rebuilt.clear()
        docker_build._base_image_rebuilt.update(original_rebuilt)
        docker_build._pushed_base_refs.clear()
        docker_build._pushed_base_refs.update(original_pushed)

    assert candidate_replay.loaded_evaluator_snapshot() == startup


def test_replay_stops_if_evaluator_source_changes_during_a_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _store_root(tmp_path / "store")
    _record_a_b_a(root)
    _patch_config_and_task(
        monkeypatch,
        root=root,
        output=tmp_path / "output",
    )
    monkeypatch.setattr(cli, "create_backend_from_config", lambda config: object())
    source_identity = "sha256:" + "a" * 64
    source_changed = False
    monkeypatch.setattr(
        candidate_replay,
        "_evaluator_source_sha256",
        lambda: "sha256:" + ("b" if source_changed else "a") * 64,
    )
    calls = 0

    def fake_judge_submission(**kwargs):
        nonlocal calls, source_changed
        calls += 1
        if calls == 1:
            source_changed = True
        return EvalReport(
            task_id="task_alpha",
            submission_id=kwargs["submission_id"],
            score=1.0,
            pass_rate=1.0,
            valid=True,
        )

    monkeypatch.setattr(cli, "judge_submission", fake_judge_submission)

    with pytest.raises(SystemExit) as raised:
        _run_cli(
            monkeypatch,
            "replay-candidates",
            "--task",
            "task_alpha",
            "--run-id",
            "origin-run",
            "--measurement-set",
            "hidden",
            "--replay-run-id",
            "source-change-run",
            "--candidate",
            "v1",
            "--json",
        )

    assert raised.value.code == 1
    assert calls == 1
    captured = capsys.readouterr()
    curve = json.loads(captured.out)
    assert curve["evaluator"]["evaluator_source_sha256"] == source_identity
    assert curve["terminal_error"]["kind"] == "evaluator_source_changed"
    assert curve["points"][0]["status"] == "failed"
    assert "evaluator source changed" in captured.err

    source_changed = False
    _run_cli(
        monkeypatch,
        "replay-candidates",
        "--task",
        "task_alpha",
        "--run-id",
        "origin-run",
        "--measurement-set",
        "hidden",
        "--replay-run-id",
        "source-change-run",
        "--candidate",
        "v1",
        "--resume",
        "--json",
    )
    resumed = json.loads(capsys.readouterr().out)
    assert calls == 2
    assert resumed["points"][0]["status"] == "completed"
    assert resumed["points"][0]["attempt_count"] == 2


@pytest.mark.parametrize("malformed_field", ["metrics", "errors"])
def test_source_change_with_malformed_report_is_sanitized_and_retried(
    malformed_field: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _store_root(tmp_path / "store")
    _record_a_b_a(root)
    output = tmp_path / "output"
    _patch_config_and_task(monkeypatch, root=root, output=output)
    monkeypatch.setattr(cli, "create_backend_from_config", lambda config: object())
    source_changed = False
    monkeypatch.setattr(
        candidate_replay,
        "_evaluator_source_sha256",
        lambda: "sha256:" + ("b" if source_changed else "a") * 64,
    )
    calls = 0

    class MalformedReport:
        def __init__(self, submission_id: str) -> None:
            self.submission_id = submission_id

        def to_dict(self) -> dict[str, object]:
            value = EvalReport(
                task_id="task_alpha",
                submission_id=self.submission_id,
                score=1.0,
                pass_rate=1.0,
                valid=True,
            ).to_dict()
            value[malformed_field] = object()
            return value

    def fake_judge_submission(**kwargs: object) -> object:
        nonlocal calls, source_changed
        calls += 1
        if calls == 1:
            source_changed = True
            return MalformedReport(str(kwargs["submission_id"]))
        return EvalReport(
            task_id="task_alpha",
            submission_id=str(kwargs["submission_id"]),
            score=2.0,
            pass_rate=1.0,
            valid=True,
        )

    monkeypatch.setattr(cli, "judge_submission", fake_judge_submission)
    arguments = (
        "replay-candidates",
        "--task",
        "task_alpha",
        "--run-id",
        "origin-run",
        "--measurement-set",
        "hidden",
        "--replay-run-id",
        f"source-malformed-{malformed_field}",
        "--candidate",
        "v1",
        "--json",
    )

    with pytest.raises(SystemExit) as raised:
        _run_cli(monkeypatch, *arguments)

    assert raised.value.code == 1
    first = json.loads(capsys.readouterr().out)
    assert first["terminal_error"]["kind"] == "evaluator_source_changed"
    assert first["points"][0]["status"] == "failed"
    top = next(
        path
        for path in output.glob("**/measurement.json")
        if path.parent.name.startswith("candidate-")
    )
    measurement = json.loads(top.read_text(encoding="utf-8"))
    assert measurement["report"]["metrics"]["infrastructure_reason"] == (
        "evaluator_source_changed"
    )

    source_changed = False
    _run_cli(monkeypatch, *arguments, "--resume")
    resumed = json.loads(capsys.readouterr().out)
    assert calls == 2
    assert resumed["points"][0]["attempt_count"] == 2


def test_live_evaluator_rejects_a_wrapper_image_claim_mismatch() -> None:
    task_spec = _task_spec()

    with pytest.raises(
        candidate_replay.CandidateReplayError,
        match="wrapper Judge image identity",
    ):
        candidate_replay.evaluator_descriptor(
            task_spec,
            SForgeConfig(),
            timeout=None,
            runtime_context={"judge_image_id": "sha256:" + "b" * 64},
            resolved_image=_resolved_test_image(task_spec),
        )


@pytest.mark.parametrize("changed_field", ["parser", "rescale", "cwd", "platform", "base"])
def test_contextual_evaluator_pins_complete_task_contract(
    changed_field: str,
) -> None:
    original = _task_spec()
    changed = _task_spec()
    if changed_field == "parser":
        changed.judge.parser = "structured_json"
    elif changed_field == "rescale":
        changed.judge.rescale = RescaleSpec(kind="linear", lower=0, upper=10)
    elif changed_field == "cwd":
        changed.cwd = "/different"
    elif changed_field == "platform":
        changed.platform = "linux/arm64"
    else:
        changed.base_image_spec = {"image": "python:3.11"}

    context = {"judge_image_id": "sha256:image-a"}
    before = candidate_replay.evaluator_descriptor(
        original, SForgeConfig(), timeout=None, runtime_context=context
    )
    after = candidate_replay.evaluator_descriptor(
        changed, SForgeConfig(), timeout=None, runtime_context=context
    )

    assert before["contract_schema_version"] == 3
    assert before["task_contract_sha256"].startswith("sha256:")
    assert before["evaluator_source_sha256"].startswith("sha256:")
    assert before["fingerprint"] != after["fingerprint"]


def test_contextual_evaluator_commits_effective_judge_environment() -> None:
    context = {"judge_image_id": "sha256:image-a"}
    first = candidate_replay.evaluator_descriptor(
        _task_spec(),
        SForgeConfig(judge_extra_env={"FIXTURE_MODE": "a"}),
        timeout=None,
        runtime_context=context,
    )
    second = candidate_replay.evaluator_descriptor(
        _task_spec(),
        SForgeConfig(judge_extra_env={"FIXTURE_MODE": "b"}),
        timeout=None,
        runtime_context=context,
    )

    assert first["judge_environment_sha256"].startswith("sha256:")
    assert "FIXTURE_MODE" not in json.dumps(first)
    assert first["fingerprint"] != second["fingerprint"]


def test_contextual_evaluator_commits_judge_network_isolation() -> None:
    context = {"judge_image_id": "sha256:image-a"}
    connected = candidate_replay.evaluator_descriptor(
        _task_spec(),
        SForgeConfig(),
        timeout=None,
        runtime_context=context,
    )
    isolated = candidate_replay.evaluator_descriptor(
        _task_spec(),
        SForgeConfig(judge_network_mode="none"),
        timeout=None,
        runtime_context=context,
    )

    assert connected["backend_contract_sha256"] != isolated[
        "backend_contract_sha256"
    ]
    assert connected["fingerprint"] != isolated["fingerprint"]


# The descriptor keys as they were before opt-in CPU pinning existed.  With no
# Judge policy the descriptor must keep exactly these keys (and so its digest).
UNPINNED_DESCRIPTOR_KEYS = {
    "task_id",
    "judge_image",
    "eval_cmd_sha256",
    "eval_timeout",
    "private_artifact_path",
    "submit_paths",
    "submit_exclude",
    "backend",
    "judge_cpu_limit",
    "judge_mem_limit",
    "score_direction",
    "selection",
    "contract_schema_version",
    "task_contract_sha256",
    "backend_contract_sha256",
    "judge_environment_sha256",
    "evaluator_source_sha256",
    "loaded_evaluator_code_sha256",
    "loaded_evaluator_bytecode_sha256",
    "loaded_evaluator_modules",
    "resolved_judge_image",
    "runtime_context",
    "fingerprint",
}


def _contextual_descriptor(config: SForgeConfig) -> dict:
    return candidate_replay.evaluator_descriptor(
        _task_spec(),
        config,
        timeout=None,
        runtime_context={"judge_image_id": "sha256:image-a"},
    )


def test_unpinned_evaluator_descriptor_is_unchanged_by_cpu_pinning() -> None:
    plain = _contextual_descriptor(SForgeConfig())

    assert set(plain) == UNPINNED_DESCRIPTOR_KEYS
    # An explicit off, or a Work-only policy, is not a Judge measurement input.
    for config in (
        SForgeConfig(cpuset="off"),
        SForgeConfig(cpuset="auto", judge_cpuset="off"),
        SForgeConfig(work_cpuset="auto"),
    ):
        assert _contextual_descriptor(config) == plain


def test_pinned_judge_policy_changes_the_evaluator_fingerprint() -> None:
    plain = _contextual_descriptor(SForgeConfig())
    pinned = _contextual_descriptor(SForgeConfig(cpuset="auto"))

    assert pinned["judge_cpu_affinity"] == "auto:gpu-local-l3-v1:mems=bind"
    assert pinned["fingerprint"] != plain["fingerprint"]
    # The policy id is the only difference: without it the digest is the
    # unpinned one, byte for byte.
    stripped = {
        key: value
        for key, value in pinned.items()
        if key not in {"judge_cpu_affinity", "fingerprint"}
    }
    canonical = json.dumps(
        stripped, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    assert f"sha256:{hashlib.sha256(canonical).hexdigest()}" == plain["fingerprint"]


def test_judge_cpu_affinity_identity_is_host_independent() -> None:
    first = _contextual_descriptor(SForgeConfig(judge_cpuset="8-11,40-43@0"))
    second = _contextual_descriptor(SForgeConfig(judge_cpuset="24-27,56-59@1"))
    auto = _contextual_descriptor(SForgeConfig(judge_cpuset="auto"))

    assert first["judge_cpu_affinity"] == "explicit:cpus=8:nodes=1:mems=bind"
    assert first["fingerprint"] == second["fingerprint"]
    assert first["fingerprint"] != auto["fingerprint"]
    assert "40-43" not in json.dumps(first)


def _run_config(config: SForgeConfig) -> dict:
    args = types.SimpleNamespace(
        agent="claude-code",
        model=None,
        timeout=None,
        eval_interval=None,
        judge_url="http://host.docker.internal:8080",
    )
    return cli._effective_config_dict(_task_spec(), args, config)


def test_run_config_records_cpu_affinity_only_when_set() -> None:
    plain = _run_config(SForgeConfig())
    assert "cpu_affinity" not in plain
    assert _run_config(SForgeConfig(cpuset="off")) == plain

    pinned = _run_config(SForgeConfig(cpuset="auto", judge_cpuset="8-11,40-43@0"))
    assert pinned["cpu_affinity"] == {
        "work": "auto:gpu-local-l3-v1:mems=bind",
        "judge": "explicit:cpus=8:nodes=1:mems=bind",
    }
    assert {key: value for key, value in pinned.items() if key != "cpu_affinity"} == plain

    judge_only = _run_config(SForgeConfig(judge_cpuset="auto"))
    assert judge_only["cpu_affinity"] == {"judge": "auto:gpu-local-l3-v1:mems=bind"}


def test_replay_candidates_rejects_ambiguous_evaluator_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _store_root(tmp_path / "store")
    _record_a_b_a(root)
    _patch_config_and_task(
        monkeypatch,
        root=root,
        output=tmp_path / "output",
    )

    with pytest.raises(SystemExit) as raised:
        _run_cli(
            monkeypatch,
            "replay-candidates",
            "--task",
            "task_alpha",
            "--run-id",
            "origin-run",
            "--measurement-set",
            "hidden",
            "--evaluator-context",
            "asset=sha256:a",
            "--evaluator-context",
            "asset=sha256:b",
            "--dry-run",
        )

    assert raised.value.code == 2
    assert "duplicate evaluator context key" in capsys.readouterr().err


@pytest.mark.parametrize(
    "value",
    ["missing-separator", "=empty-key", "bad key=value", "key=bad\nvalue"],
)
def test_evaluator_context_rejects_unsafe_fields(value: str) -> None:
    with pytest.raises(candidate_replay.CandidateReplayError):
        candidate_replay.parse_runtime_context([value])


def test_candidate_curve_history_best_honors_minimize_policy(tmp_path: Path) -> None:
    root = _store_root(tmp_path / "store")
    _record_a_b_a(root)
    candidates = VersionStore.open(root).list_candidates(
        task="task_alpha", run_id="origin-run"
    )
    measurements = {
        candidate.candidate_id: {
            "status": "completed",
            "attempt": 1,
            "report": {
                "score": score,
                "pass_rate": 1.0,
                "valid": True,
            },
        }
        for candidate, score in zip(candidates, (10.0, 5.0), strict=True)
    }

    curve = candidate_replay.build_curve(
        candidates=candidates,
        measurements=measurements,
        origin_run_id="origin-run",
        replay_run_id="replay-run",
        measurement_set="latency",
        evaluator={
            "task_id": "task_alpha",
            "score_direction": "minimize",
            "selection": "score_first",
        },
        curve_root=tmp_path / "curve",
    )

    assert [point["history_best"] for point in curve["points"]] == [10.0, 5.0]
    assert [point["history_best_version"] for point in curve["points"]] == [
        "v1",
        "v2",
    ]
    assert [point["version"] for point in curve["timeline_points"]] == [
        "v1",
        "v2",
        "v1",
    ]
    assert [point["relation"] for point in curve["timeline_points"]] == [
        "first_seen",
        "first_seen",
        "revisit",
    ]
    assert [point["linked_score"] for point in curve["timeline_points"]] == [
        10.0,
        5.0,
        10.0,
    ]
    assert [point["history_best"] for point in curve["timeline_points"]] == [
        10.0,
        5.0,
        5.0,
    ]
    assert [
        point["history_best_version"] for point in curve["timeline_points"]
    ] == ["v1", "v2", "v2"]


def test_replay_candidates_can_fill_one_pinned_curve_in_selected_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _store_root(tmp_path / "store")
    _record_a_b_a(root)
    _patch_config_and_task(
        monkeypatch,
        root=root,
        output=tmp_path / "output",
    )
    monkeypatch.setattr(cli, "create_backend_from_config", lambda config: object())
    calls: list[str] = []

    def fake_judge_submission(**kwargs):
        calls.append(kwargs["round_id"])
        return EvalReport(
            task_id="task_alpha",
            submission_id=kwargs["submission_id"],
            score=float(len(calls)),
            pass_rate=1.0,
            valid=True,
        )

    monkeypatch.setattr(cli, "judge_submission", fake_judge_submission)
    base = (
        "replay-candidates",
        "--task",
        "task_alpha",
        "--run-id",
        "origin-run",
        "--measurement-set",
        "hidden",
        "--replay-run-id",
        "replay-run",
        "--json",
    )

    _run_cli(monkeypatch, *base, "--candidate", "v1")
    first = json.loads(capsys.readouterr().out)
    assert first["candidate_count"] == 2
    assert first["completed_count"] == 1
    assert [point["status"] for point in first["points"]] == [
        "completed",
        "planned",
    ]

    _run_cli(monkeypatch, *base, "--candidate", "v2", "--resume")
    second = json.loads(capsys.readouterr().out)
    assert second["candidate_count"] == 2
    assert second["completed_count"] == 2
    assert calls == ["candidate-v1-attempt-1", "candidate-v2-attempt-1"]


@pytest.mark.parametrize(
    "report",
    [
        {"timed_out": True},
        {"metrics": {"evaluation_status": "infrastructure_error"}},
        {"metrics": {"evaluation_status": "resource_busy"}},
        {"metrics": {"evaluation_status": "cancelled"}},
        {"metrics": {"infrastructure_invalid": True}},
        {"metrics": {"failure_stage": "reference_worker", "failure_class": "timeout"}},
        {"metrics": {"failure_stage": "candidate_worker", "failure_class": "gpu_busy"}},
        {
            "metrics": {
                "failure_stage": "telemetry_validation",
                "failure_class": "invalid_telemetry",
                "failure_role": "reference",
            }
        },
    ],
)
def test_candidate_replay_classifies_cross_task_infrastructure_for_retry(
    report: dict,
) -> None:
    assert candidate_replay.report_is_infrastructure_failure(report) is True


@pytest.mark.parametrize(
    "report",
    [
        {"valid": False, "score": 0.0, "metrics": {"evaluation_status": "candidate_invalid"}},
        {"valid": False, "score": 0.0, "metrics": {"candidate_invalid": True}},
        {"valid": False, "score": 0.0, "metrics": {"failure_stage": "telemetry_validation", "failure_class": "invalid_telemetry"}},
        {
            "valid": False,
            "score": 0.0,
            "metrics": {
                "failure_stage": "telemetry_validation",
                "failure_class": "invalid_telemetry",
                "failure_role": "candidate",
            },
        },
    ],
)
def test_candidate_replay_keeps_candidate_failures_as_completed_points(
    report: dict,
) -> None:
    assert candidate_replay.report_is_infrastructure_failure(report) is False


def _stride_candidate(sequence: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        sequence=sequence,
        candidate_id=f"candidate-{sequence:03d}",
    )


def test_selected_replay_candidates_every_samples_and_keeps_last() -> None:
    candidates = [_stride_candidate(sequence) for sequence in range(1, 11)]

    sampled = cli._selected_replay_candidates(candidates, [], None, None, 2)
    assert [candidate.sequence for candidate in sampled] == [1, 3, 5, 7, 9, 10]

    sampled = cli._selected_replay_candidates(candidates, [], None, None, 3)
    assert [candidate.sequence for candidate in sampled] == [1, 4, 7, 10]

    # A stride that lands exactly on the last candidate does not duplicate it.
    sampled = cli._selected_replay_candidates(candidates, [], None, None, 9)
    assert [candidate.sequence for candidate in sampled] == [1, 10]


def test_selected_replay_candidates_every_composes_with_range() -> None:
    candidates = [_stride_candidate(sequence) for sequence in range(1, 11)]

    sampled = cli._selected_replay_candidates(candidates, [], 3, 8, 2)
    assert [candidate.sequence for candidate in sampled] == [3, 5, 7, 8]

    assert cli._selected_replay_candidates(candidates, [], None, None, 1) == candidates
    assert cli._selected_replay_candidates(candidates, [], None, None, 99) == [
        candidates[0],
        candidates[-1],
    ]


def test_selected_replay_candidates_every_rejects_bad_combinations() -> None:
    candidates = [_stride_candidate(1)]

    with pytest.raises(candidate_replay.CandidateReplayError):
        cli._selected_replay_candidates(candidates, ["v1"], None, None, 2)
    with pytest.raises(candidate_replay.CandidateReplayError):
        cli._selected_replay_candidates(candidates, [], None, None, 0)


def _replicate_measurement(scores, statuses=None):
    statuses = statuses or ["completed"] * len(scores)
    return {
        "status": "completed",
        "replicates": [
            {"attempt": i + 1, "status": st, "score": sc}
            for i, (sc, st) in enumerate(zip(scores, statuses))
        ],
    }


def test_replicate_statistics_needs_two_completed_samples() -> None:
    assert candidate_replay.replicate_statistics(None) is None
    assert candidate_replay.replicate_statistics({"status": "completed"}) is None
    assert candidate_replay.replicate_statistics(_replicate_measurement([1.0])) is None
    # Failed replicates do not count toward the sample set.
    assert (
        candidate_replay.replicate_statistics(
            _replicate_measurement([1.0, 2.0], ["completed", "failed"])
        )
        is None
    )
    # Non-finite scores are excluded.
    assert (
        candidate_replay.replicate_statistics(_replicate_measurement([1.0, None]))
        is None
    )


def test_replicate_statistics_reports_median_and_spread() -> None:
    stats = candidate_replay.replicate_statistics(
        _replicate_measurement([158.45, 140.38, 150.92, 158.32, 146.67])
    )

    assert stats["score_sample_count"] == 5
    assert stats["score_samples"] == [140.38, 146.67, 150.92, 158.32, 158.45]
    assert stats["score_median"] == 150.92
    assert stats["score_min"] == 140.38
    assert stats["score_max"] == 158.45
    assert stats["score_stdev"] > 0


def test_replicate_statistics_ignores_incomplete_measurements() -> None:
    running = _replicate_measurement([1.0, 2.0])
    running["status"] = "running"
    # build_curve only consults statistics for completed measurements, but the
    # helper itself stays permissive about the parent status.
    assert candidate_replay.replicate_statistics(running) is not None
    assert candidate_replay.replicate_statistics({"replicates": "nope"}) is None


def _versioned_archive(sources: dict[int, bytes], selected: int) -> bytes:
    """A submitted tree that retains one file per implementation."""

    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as bundle:
        for number in sorted(sources):
            member = tarfile.TarInfo(f"kernels/impl_v{number:03d}.py")
            member.size = len(sources[number])
            member.mode = 0o644
            bundle.addfile(member, io.BytesIO(sources[number]))
        selector = f"{selected}\n".encode()
        member = tarfile.TarInfo("FINAL_VERSION")
        member.size = len(selector)
        member.mode = 0o644
        bundle.addfile(member, io.BytesIO(selector))
    return output.getvalue()


def _version_task_spec() -> TaskSpec:
    spec = _task_spec()
    spec.submit_paths = ["kernels", "FINAL_VERSION"]
    spec.version_enumeration = {
        "root": "kernels",
        "file_pattern": r"impl_v([0-9]{3})\.py",
        "selector_file": "FINAL_VERSION",
    }
    return spec


def test_replay_versions_repeat_reports_the_median_of_every_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Repeated passes must reach every implementation, median not last score.

    Neighbouring implementations can sit inside the benchmark's own noise, so
    a single measurement per version can order them wrongly. --repeat measures
    each one again in a later pass and the curve reports the median.
    """

    root = _store_root(tmp_path / "store")
    store = VersionStore.open(root)
    common = {
        "task_id": "task_alpha",
        "run_id": "origin-run",
        "submit_paths": ["kernels", "FINAL_VERSION"],
        "submit_exclude": [],
    }
    # v002 is never selected; only per-version replay can reach it.
    store.record_archive(
        source="checkpoint",
        round_id="checkpoint-1",
        label="first",
        archive=_versioned_archive({1: b"one\n"}, selected=1),
        created_at=datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc),
        **common,
    )
    store.record_archive(
        source="final",
        round_id="final",
        label="final",
        archive=_versioned_archive({1: b"one\n", 2: b"two\n"}, selected=1),
        created_at=datetime(2026, 8, 20, 12, 30, tzinfo=timezone.utc),
        **common,
    )

    output = tmp_path / "output"
    _patch_config_and_task(monkeypatch, root=root, output=output)
    monkeypatch.setattr(
        cli, "_resolve_task", lambda args, resolved_config: _version_task_spec()
    )
    monkeypatch.setattr(cli, "create_backend_from_config", lambda config: object())

    # Pass one is noisy on v001; the median across passes must ignore it.
    scores = {("1", 1): 90.0, ("1", 2): 10.0, ("1", 3): 12.0,
              ("2", 1): 20.0, ("2", 2): 21.0, ("2", 3): 22.0}
    seen: list[int] = []

    def fake_judge_submission(**kwargs):
        version = kwargs["candidate_version"]
        seen.append(version)
        attempt = sum(1 for item in seen if item == version)
        return EvalReport(
            task_id="task_alpha",
            submission_id=kwargs["submission_id"],
            total_tests=1,
            passed=1,
            pass_rate=1.0,
            score=scores[(str(version), attempt)],
            valid=True,
            runtime_seconds=1.0,
            submitted_at=1.0,
        )

    monkeypatch.setattr(cli, "judge_submission", fake_judge_submission)
    _run_cli(
        monkeypatch,
        "replay-versions",
        "--task", "task_alpha",
        "--run-id", "origin-run",
        "--measurement-set", "hidden-20260820",
        "--replay-run-id", "replay-run",
        "--repeat", "2",
        "--json",
    )
    curve = json.loads(capsys.readouterr().out)

    # Every implementation is measured once per pass, passes interleaved so
    # drift over the batch hits both versions alike.
    assert seen == [1, 2, 1, 2, 1, 2]
    points = {point["version_number"]: point for point in curve["points"]}
    assert set(points) == {1, 2}
    assert points[1]["score"] == 12.0        # median(90, 10, 12), not 12 by luck
    assert points[1]["statistics"]["score_samples"] == [10.0, 12.0, 90.0]
    assert points[2]["score"] == 21.0
    # The noisy first pass must not survive as the run's best.
    assert points[2]["history_best"] == 21.0


def test_replay_versions_repeat_must_not_be_negative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _store_root(tmp_path / "store")
    store = VersionStore.open(root)
    store.record_archive(
        task_id="task_alpha",
        run_id="origin-run",
        source="final",
        round_id="final",
        label="final",
        archive=_versioned_archive({1: b"one\n"}, selected=1),
        submit_paths=["kernels", "FINAL_VERSION"],
        submit_exclude=[],
        created_at=datetime(2026, 8, 20, 12, 30, tzinfo=timezone.utc),
    )
    _patch_config_and_task(monkeypatch, root=root, output=tmp_path / "output")
    monkeypatch.setattr(
        cli, "_resolve_task", lambda args, resolved_config: _version_task_spec()
    )
    with pytest.raises(SystemExit):
        _run_cli(
            monkeypatch,
            "replay-versions",
            "--task", "task_alpha",
            "--run-id", "origin-run",
            "--measurement-set", "hidden-20260820",
            "--repeat", "-1",
            "--json",
        )
    assert "repeat" in capsys.readouterr().err


def test_rewrite_selector_archive_changes_only_the_selector() -> None:
    """Judges score what the selector names, so replay must move it.

    The deployment children of flash-kmeans re-derive the selection from the
    tree and refuse a candidate that differs from the selected source, so a
    side-channel override alone cannot reach implementations the agent had
    not yet selected — the tree itself has to name the measured version.
    """

    members = {
        "kernels/impl_v001.py": b"one\n",
        "kernels/impl_v020.py": b"twenty\n",
        "FINAL_VERSION": b"19\n",
        "notes.md": b"n\n",
    }
    original = _versioned_archive({1: b"one\n"}, selected=19)
    # Build directly so member order and modes are realistic.
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as bundle:
        for name in sorted(members):
            info = tarfile.TarInfo(name)
            info.size = len(members[name])
            info.mode = 0o644
            bundle.addfile(info, io.BytesIO(members[name]))
    original = output.getvalue()

    rewritten = candidate_replay.rewrite_selector_archive(
        original, selector_file="FINAL_VERSION", version_number=20
    )
    with tarfile.open(fileobj=io.BytesIO(rewritten), mode="r:gz") as bundle:
        extracted = {
            member.name: bundle.extractfile(member).read()
            for member in bundle.getmembers()
            if member.isfile()
        }
    assert extracted.pop("FINAL_VERSION") == b"20\n"
    assert extracted == {
        name: data for name, data in members.items() if name != "FINAL_VERSION"
    }

    with pytest.raises(candidate_replay.CandidateReplayError, match="selector"):
        candidate_replay.rewrite_selector_archive(
            original, selector_file="MISSING", version_number=20
        )
    with pytest.raises(candidate_replay.CandidateReplayError, match="1..999"):
        candidate_replay.rewrite_selector_archive(
            original, selector_file="FINAL_VERSION", version_number=0
        )


def test_replay_versions_rewrites_the_selector_for_each_implementation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every judged archive must select the implementation being measured."""

    root = _store_root(tmp_path / "store")
    store = VersionStore.open(root)
    common = {
        "task_id": "task_alpha",
        "run_id": "origin-run",
        "submit_paths": ["kernels", "FINAL_VERSION"],
        "submit_exclude": [],
    }
    # v002 exists from the first event but the selector never reaches it.
    store.record_archive(
        source="checkpoint",
        round_id="checkpoint-1",
        label="first",
        archive=_versioned_archive({1: b"one\n", 2: b"two\n"}, selected=1),
        created_at=datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc),
        **common,
    )
    store.record_archive(
        source="final",
        round_id="final",
        label="final",
        archive=_versioned_archive({1: b"one\n", 2: b"two\n"}, selected=1),
        created_at=datetime(2026, 8, 20, 12, 30, tzinfo=timezone.utc),
        **common,
    )
    _patch_config_and_task(monkeypatch, root=root, output=tmp_path / "output")
    monkeypatch.setattr(
        cli, "_resolve_task", lambda args, resolved_config: _version_task_spec()
    )
    monkeypatch.setattr(cli, "create_backend_from_config", lambda config: object())
    seen_selectors: list[tuple[int, bytes]] = []

    def fake_judge_submission(**kwargs):
        with tarfile.open(
            fileobj=io.BytesIO(kwargs["archive"]), mode="r:gz"
        ) as bundle:
            selector = bundle.extractfile("FINAL_VERSION").read()
        seen_selectors.append((kwargs["candidate_version"], selector))
        return EvalReport(
            task_id="task_alpha",
            submission_id=kwargs["submission_id"],
            total_tests=1,
            passed=1,
            pass_rate=1.0,
            score=1.0,
            valid=True,
            runtime_seconds=1.0,
            submitted_at=1.0,
        )

    monkeypatch.setattr(cli, "judge_submission", fake_judge_submission)
    _run_cli(
        monkeypatch,
        "replay-versions",
        "--task", "task_alpha",
        "--run-id", "origin-run",
        "--measurement-set", "hidden-20260820",
        "--replay-run-id", "replay-run",
        "--json",
    )
    json.loads(capsys.readouterr().out)
    # The stored selector said "1" in both events; the judged archives must
    # each select the implementation being measured instead.
    assert seen_selectors == [(1, b"1\n"), (2, b"2\n")]
