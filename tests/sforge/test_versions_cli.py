from __future__ import annotations

import io
import json
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from sforge import cli
from sforge.harness.config import SForgeConfig
from sforge.harness.version_store import VersionStore


def _archive(name: str, payload: bytes) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as bundle:
        member = tarfile.TarInfo(f"submission/{name}")
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))
    return output.getvalue()


def _store_root(root: Path) -> Path:
    root.mkdir()
    (root / ".sforge-version-store.json").write_text(
        '{"schema_version": 1, "kind": "sforge-version-store", "store_id": "cli-test-store"}\n',
        encoding="utf-8",
    )
    for task_directory, task_id in (
        ("task-alpha", "task_alpha"),
        ("task-beta", "task_beta"),
    ):
        task_root = root / "tasks" / task_directory
        (task_root / "runs").mkdir(parents=True)
        (task_root / "task.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "task_directory": task_directory,
                    "task_id": task_id,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    return root


def _record_versions(root: Path):
    store = VersionStore.open(root)
    saved = []
    for task_id, run_id, source, round_id, minute in (
        ("task_alpha", "run-002", "final", "final", 2),
        ("task_beta", "run-001", "agent", "agent-1", 0),
        ("task_alpha", "run-001", "checkpoint", "checkpoint-1", 1),
    ):
        saved.append(
            store.record_archive(
                task_id=task_id,
                run_id=run_id,
                source=source,
                round_id=round_id,
                label="known-good" if source == "checkpoint" else None,
                archive=_archive(f"{round_id}.txt", round_id.encode()),
                submit_paths=["submission"],
                submit_exclude=[],
                metadata={"agent": "codex"},
                created_at=datetime(2026, 8, 18, 12, minute, tzinfo=timezone.utc),
            )
        )
    return saved


def _run_cli(monkeypatch: pytest.MonkeyPatch, *arguments: str) -> None:
    monkeypatch.setattr(sys, "argv", ["sforge", *arguments])
    cli.main()


def test_versions_table_is_stably_ordered_and_contains_replay_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _store_root(tmp_path / "store")
    versions = _record_versions(root)

    _run_cli(monkeypatch, "versions", "--version-store-root", str(root))

    output = capsys.readouterr().out
    assert all(column in output.splitlines()[0] for column in (
        "CREATED", "SOURCE", "ROUND", "TASK", "RUN", "SHA256", "ARCHIVE"
    ))
    ordered = sorted(
        versions,
        key=lambda version: (
            version.task_id,
            version.run_id,
            version.event_sequence,
        ),
    )
    positions = [output.index(version.version_id) for version in ordered]
    assert positions == sorted(positions)
    for version in versions:
        assert str(version.archive_path.resolve()) in output


@pytest.mark.parametrize("task_filter", ["task_alpha", "task-alpha"])
def test_versions_json_filters_by_task_id_or_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    task_filter: str,
) -> None:
    root = _store_root(tmp_path / "store")
    _record_versions(root)

    _run_cli(
        monkeypatch,
        "versions",
        "--version-store-root",
        str(root),
        "--task",
        task_filter,
        "--run-id",
        "run-001",
        "--source",
        "checkpoint",
        "--json",
        "--verify",
    )

    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 1
    assert payload[0]["task_id"] == "task_alpha"
    assert payload[0]["task_directory"] == "task-alpha"
    assert payload[0]["run_id"] == "run-001"
    assert payload[0]["source"] == "checkpoint"
    assert payload[0]["event_sequence"] == 1
    assert payload[0]["candidate_sequence"] == 1
    assert payload[0]["candidate_id"].startswith("candidate-")
    assert len(payload[0]["content_sha256"]) == 64
    assert payload[0]["label"] == "known-good"
    assert payload[0]["metadata"] == {"agent": "codex"}
    assert payload[0]["archive_path"] == str(
        Path(payload[0]["archive_path"]).resolve()
    )


def test_versions_verify_exits_nonzero_when_archive_is_corrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _store_root(tmp_path / "store")
    versions = _record_versions(root)
    versions[0].archive_path.write_bytes(b"corrupt")

    with pytest.raises(SystemExit) as raised:
        _run_cli(
            monkeypatch,
            "versions",
            "--version-store-root",
            str(root),
            "--verify",
        )

    captured = capsys.readouterr()
    assert raised.value.code == 1
    assert "size mismatch" in captured.err or "SHA-256 mismatch" in captured.err


def test_versions_candidates_groups_archive_events_by_content_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _store_root(tmp_path / "store")
    store = VersionStore.open(root)
    archive = _archive("main.py", b"same candidate\n")
    for source, round_id, minute in (
        ("checkpoint", "checkpoint-1", 0),
        ("agent", "agent-1", 1),
    ):
        store.record_archive(
            task_id="task_alpha",
            run_id="run-candidate",
            source=source,
            round_id=round_id,
            label="before-public" if source == "checkpoint" else None,
            archive=archive,
            submit_paths=["submission"],
            submit_exclude=[],
            created_at=datetime(2026, 8, 18, 12, minute, tzinfo=timezone.utc),
        )

    _run_cli(
        monkeypatch,
        "versions",
        "--version-store-root",
        str(root),
        "--run-id",
        "run-candidate",
        "--candidates",
        "--json",
        "--verify",
    )

    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 1
    candidate = payload[0]
    assert candidate["sequence"] == 1
    assert candidate["candidate_id"] == f"candidate-{candidate['content_sha256']}"
    assert candidate["label"] == "before-public"
    assert [event["round_id"] for event in candidate["events"]] == [
        "checkpoint-1",
        "agent-1",
    ]


def test_versions_timeline_distinguishes_first_seen_revisit_and_same_previous(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _store_root(tmp_path / "store")
    store = VersionStore.open(root)
    archive_a = _archive("main.py", b"candidate A\n")
    archive_b = _archive("main.py", b"candidate B\n")
    for source, round_id, label, archive, minute in (
        ("checkpoint", "checkpoint-1", "candidate-a", archive_a, 0),
        ("checkpoint", "checkpoint-2", "candidate-b", archive_b, 1),
        ("checkpoint", "checkpoint-3", "rollback-a", archive_a, 2),
        ("agent", "agent-1", None, archive_a, 3),
    ):
        store.record_archive(
            task_id="task_alpha",
            run_id="run-timeline",
            source=source,
            round_id=round_id,
            label=label,
            archive=archive,
            submit_paths=["submission"],
            submit_exclude=[],
            created_at=datetime(2026, 8, 18, 12, minute, tzinfo=timezone.utc),
        )

    _run_cli(
        monkeypatch,
        "versions",
        "--version-store-root",
        str(root),
        "--run-id",
        "run-timeline",
        "--timeline",
        "--json",
        "--verify",
    )

    payload = json.loads(capsys.readouterr().out)
    assert [item["event_sequence"] for item in payload] == [1, 2, 3, 4]
    assert [item["candidate_sequence"] for item in payload] == [1, 2, 1, 1]
    assert [item["relation"] for item in payload] == [
        "first_seen",
        "first_seen",
        "revisit",
        "same_as_previous",
    ]
    candidate_a = payload[0]["candidate_id"]
    candidate_b = payload[1]["candidate_id"]
    assert [item["previous_candidate_id"] for item in payload] == [
        None,
        candidate_a,
        candidate_b,
        candidate_a,
    ]
    assert [item["first_seen_event_sequence"] for item in payload] == [1, 2, 1, 1]
    assert [item["round_id"] for item in payload] == [
        "checkpoint-1",
        "checkpoint-2",
        "checkpoint-3",
        "agent-1",
    ]
    assert payload[2]["label"] == "rollback-a"

    _run_cli(
        monkeypatch,
        "versions",
        "--version-store-root",
        str(root),
        "--run-id",
        "run-timeline",
        "--timeline",
    )
    table = capsys.readouterr().out
    assert "TASK" in table and "RUN" in table
    assert "task-alpha" in table
    assert "run-timeline" in table


def test_versions_timeline_and_candidates_are_mutually_exclusive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _store_root(tmp_path / "store")

    with pytest.raises(SystemExit) as raised:
        _run_cli(
            monkeypatch,
            "versions",
            "--version-store-root",
            str(root),
            "--timeline",
            "--candidates",
        )

    assert raised.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


def test_versions_timeline_verify_checks_only_listed_source_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _store_root(tmp_path / "store")
    store = VersionStore.open(root)
    archive = _archive("main.py", b"candidate A\n")
    checkpoint = store.record_archive(
        task_id="task_alpha",
        run_id="run-filtered-timeline",
        source="checkpoint",
        round_id="checkpoint-1",
        label="before-public",
        archive=archive,
        submit_paths=["submission"],
        submit_exclude=[],
        created_at=datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc),
    )
    agent = store.record_archive(
        task_id="task_alpha",
        run_id="run-filtered-timeline",
        source="agent",
        round_id="agent-1",
        label=None,
        archive=archive,
        submit_paths=["submission"],
        submit_exclude=[],
        created_at=datetime(2026, 8, 18, 12, 1, tzinfo=timezone.utc),
    )
    agent.archive_path.write_bytes(b"corrupt unlisted event")

    _run_cli(
        monkeypatch,
        "versions",
        "--version-store-root",
        str(root),
        "--run-id",
        "run-filtered-timeline",
        "--source",
        "checkpoint",
        "--timeline",
        "--json",
        "--verify",
    )

    payload = json.loads(capsys.readouterr().out)
    assert [item["version_id"] for item in payload] == [checkpoint.version_id]


def test_versions_requires_a_configured_store(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "load_config", lambda overrides=None: SForgeConfig())

    with pytest.raises(SystemExit) as raised:
        _run_cli(monkeypatch, "versions")

    assert raised.value.code == 2
    assert "version store" in capsys.readouterr().err.lower()
