from __future__ import annotations

import concurrent.futures
import hashlib
import io
import json
import os
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from sforge.harness.config import load_config
from sforge.harness.version_store import (
    VersionStore,
    VersionStoreCollision,
    VersionStoreError,
    VersionStoreWriteError,
    discover_sibling_version_store,
)


def _archive(*members: tuple[str, bytes]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as bundle:
        for name, payload in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o755 if name.endswith(".sh") else 0o644
            bundle.addfile(info, io.BytesIO(payload))
    return output.getvalue()


def _archive_with_metadata(
    *members: tuple[str, bytes, int, int],
) -> bytes:
    """Build equivalent trees with deliberately different tar metadata."""

    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as bundle:
        for name, payload, mode, mtime in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = mode
            info.mtime = mtime
            bundle.addfile(info, io.BytesIO(payload))
    return output.getvalue()


def _plain_tar(*members: tuple[str, bytes]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as bundle:
        for name, payload in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            bundle.addfile(info, io.BytesIO(payload))
    return output.getvalue()


def _write_store(
    root: Path,
    *,
    registrations: tuple[tuple[str, str], ...] = (("task-alpha", "task_alpha"),),
) -> Path:
    root.mkdir()
    (root / ".sforge-version-store.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "sforge-version-store",
                "store_id": "test-store",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    for task_directory, task_id in registrations:
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


def test_open_loads_registered_tasks_and_rejects_duplicate_task_ids(
    tmp_path: Path,
) -> None:
    root = _write_store(
        tmp_path / "store",
        registrations=(("task-alpha", "task_alpha"), ("task-beta", "task_beta")),
    )

    store = VersionStore.open(root)

    assert store.store_id == "test-store"
    assert store.supports_task("task_alpha") is True
    assert store.supports_task("task_beta") is True
    assert store.supports_task("task_gamma") is False

    (root / "tasks/task-beta/task.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_directory": "task-beta",
                "task_id": "task_alpha",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(VersionStoreError, match="duplicate task_id"):
        VersionStore.open(root)


@pytest.mark.parametrize(
    "marker",
    [
        {},
        {"schema_version": 2, "kind": "sforge-version-store"},
        {"schema_version": 1, "kind": "not-a-version-store"},
        {"schema_version": 1, "kind": "sforge-version-store"},
    ],
)
def test_open_rejects_invalid_store_marker(tmp_path: Path, marker: dict) -> None:
    root = _write_store(tmp_path / "store")
    (root / ".sforge-version-store.json").write_text(
        json.dumps(marker), encoding="utf-8"
    )

    with pytest.raises(VersionStoreError, match="marker"):
        VersionStore.open(root)


def test_open_rejects_symlinked_task_directory(tmp_path: Path) -> None:
    root = _write_store(tmp_path / "store", registrations=())
    outside = tmp_path / "outside"
    (outside / "runs").mkdir(parents=True)
    (outside / "task.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_directory": "task-alpha",
                "task_id": "task_alpha",
            }
        ),
        encoding="utf-8",
    )
    (root / "tasks").mkdir(exist_ok=True)
    (root / "tasks/task-alpha").symlink_to(outside, target_is_directory=True)

    with pytest.raises(VersionStoreError, match="symlink|outside"):
        VersionStore.open(root)


def test_config_reads_explicit_version_store_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _write_store(tmp_path / "store")
    monkeypatch.setenv("SFORGE_VERSION_STORE_ROOT", str(root))

    config = load_config()

    assert config.version_store_root == root
    assert isinstance(config.version_store_root, Path)


def test_sibling_discovery_requires_a_valid_marker(tmp_path: Path) -> None:
    mlsysforge = tmp_path / "mlsysforge_format_task"
    mlsysforge.mkdir()
    sibling = tmp_path / "agent_task_version_submit"
    sibling.mkdir()

    assert discover_sibling_version_store(mlsysforge) is None

    (sibling / ".sforge-version-store.json").write_text(
        '{"schema_version": 1, "kind": "sforge-version-store", "store_id": "sibling-test"}\n',
        encoding="utf-8",
    )

    assert discover_sibling_version_store(mlsysforge) == sibling


def test_record_preserves_exact_bytes_and_writes_complete_manifest(
    tmp_path: Path,
) -> None:
    root = _write_store(tmp_path / "store")
    archive = _archive(
        ("submission/gateway.py", b"print('ok')\n"),
        ("submission/launch.sh", b"#!/bin/sh\n"),
    )
    created = datetime(2026, 8, 18, 12, 34, 56, 123456, tzinfo=timezone.utc)

    saved = VersionStore.open(root).record_archive(
        task_id="task_alpha",
        run_id="run-001",
        source="checkpoint",
        round_id="checkpoint-1",
        label="known-good",
        archive=archive,
        submit_paths=["submission"],
        submit_exclude=["auth.json"],
        metadata={"agent": "codex", "model": "example-model"},
        created_at=created,
    )

    assert (saved.path / "submission.tar.gz").read_bytes() == archive
    manifest = json.loads((saved.path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == {
        "archive_sha256": hashlib.sha256(archive).hexdigest(),
        "archive_size": len(archive),
        "candidate_id": saved.candidate_id,
        "candidate_sequence": 1,
        "content_identity_exclude": [],
        "content_sha256": saved.content_sha256,
        "created_at": "2026-08-18T12:34:56.123456Z",
        "event_sequence": 1,
        "label": "known-good",
        "member_count": 2,
        "metadata": {"agent": "codex", "model": "example-model"},
        "round_id": "checkpoint-1",
        "run_id": "run-001",
        "schema_version": 1,
        "source": "checkpoint",
        "task_directory": "task-alpha",
        "task_id": "task_alpha",
        "version_id": saved.version_id,
    }
    assert saved.version_id.startswith(
        "20260818T123456.123456Z__checkpoint-1__"
    )
    assert not list(saved.path.parent.glob(".tmp-*"))


def test_sequences_follow_save_order_not_candidate_timestamps_and_survive_reopen(
    tmp_path: Path,
) -> None:
    store = VersionStore.open(_write_store(tmp_path / "store"))
    archive_a = _archive(("submission/main.py", b"candidate A\n"))
    archive_b = _archive(("submission/main.py", b"candidate B\n"))
    common = {
        "task_id": "task_alpha",
        "run_id": "run-sequences",
        "source": "checkpoint",
        "label": None,
        "submit_paths": ["submission"],
        "submit_exclude": [],
    }

    first_a = store.record_archive(
        round_id="checkpoint-1",
        archive=archive_a,
        created_at=datetime(2026, 8, 18, 12, 10, tzinfo=timezone.utc),
        **common,
    )
    second_b = store.record_archive(
        round_id="checkpoint-2",
        archive=archive_b,
        created_at=datetime(2026, 8, 18, 12, 5, tzinfo=timezone.utc),
        **common,
    )
    revisit_a = store.record_archive(
        round_id="checkpoint-3",
        archive=archive_a,
        created_at=datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc),
        **common,
    )

    assert [
        (item.event_sequence, item.candidate_sequence)
        for item in (first_a, second_b, revisit_a)
    ] == [(1, 1), (2, 2), (3, 1)]
    for stored, expected in (
        (first_a, (1, 1)),
        (second_b, (2, 2)),
        (revisit_a, (3, 1)),
    ):
        manifest = json.loads(stored.path.joinpath("manifest.json").read_text())
        assert (manifest["event_sequence"], manifest["candidate_sequence"]) == expected

    reopened = VersionStore.open(store.root)
    assert [item.round_id for item in reopened.list_versions(run_id="run-sequences")] == [
        "checkpoint-1",
        "checkpoint-2",
        "checkpoint-3",
    ]
    assert [
        (item.sequence, [event.event_sequence for event in item.events])
        for item in reopened.list_candidates(run_id="run-sequences")
    ] == [(1, [1, 3]), (2, [2])]
    timeline = reopened.list_timeline(run_id="run-sequences")
    assert [item.event_sequence for item in timeline] == [1, 2, 3]
    assert [item.candidate_sequence for item in timeline] == [1, 2, 1]
    assert [item.relation for item in timeline] == [
        "first_seen",
        "first_seen",
        "revisit",
    ]


def test_legacy_effective_sequences_remain_stable_after_a_backdated_new_event(
    tmp_path: Path,
) -> None:
    store = VersionStore.open(_write_store(tmp_path / "store"))
    common = {
        "task_id": "task_alpha",
        "run_id": "run-legacy-sequences",
        "source": "checkpoint",
        "label": None,
        "submit_paths": ["submission"],
        "submit_exclude": [],
    }
    legacy = [
        store.record_archive(
            round_id=f"checkpoint-{index}",
            archive=_archive(("submission/main.py", payload)),
            created_at=datetime(2026, 8, 18, 12, minute, tzinfo=timezone.utc),
            **common,
        )
        for index, minute, payload in (
            (1, 10, b"legacy A\n"),
            (2, 20, b"legacy B\n"),
        )
    ]
    for stored in legacy:
        manifest_path = stored.path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.pop("event_sequence")
        manifest.pop("candidate_sequence")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    reopened = VersionStore.open(store.root)
    assert [item.event_sequence for item in reopened.list_versions()] == [1, 2]
    assert [item.sequence for item in reopened.list_candidates()] == [1, 2]

    newest = reopened.record_archive(
        round_id="checkpoint-3",
        archive=_archive(("submission/main.py", b"new but backdated\n")),
        created_at=datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc),
        **common,
    )

    assert (newest.event_sequence, newest.candidate_sequence) == (3, 3)
    reopened_again = VersionStore.open(store.root)
    assert [item.round_id for item in reopened_again.list_versions()] == [
        "checkpoint-1",
        "checkpoint-2",
        "checkpoint-3",
    ]
    assert [item.sequence for item in reopened_again.list_candidates()] == [1, 2, 3]


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"event_sequence": 0}, "event sequence"),
        ({"event_sequence": True}, "event sequence"),
        ({"candidate_sequence": -1}, "candidate sequence"),
        ({"candidate_sequence": None}, "candidate sequence"),
    ],
)
def test_manifest_rejects_invalid_persistent_sequences(
    tmp_path: Path,
    updates: dict[str, object],
    message: str,
) -> None:
    store = VersionStore.open(_write_store(tmp_path / "store"))
    saved = store.record_archive(
        task_id="task_alpha",
        run_id="run-invalid-sequence",
        source="checkpoint",
        round_id="checkpoint-1",
        label=None,
        archive=_archive(("submission/main.py", b"candidate\n")),
        submit_paths=["submission"],
        submit_exclude=[],
    )
    manifest_path = saved.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(updates)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(VersionStoreError, match=message):
        VersionStore.open(store.root).list_versions()


def test_manifest_rejects_only_one_persistent_sequence_field(
    tmp_path: Path,
) -> None:
    store = VersionStore.open(_write_store(tmp_path / "store"))
    saved = store.record_archive(
        task_id="task_alpha",
        run_id="run-incomplete-sequence",
        source="checkpoint",
        round_id="checkpoint-1",
        label=None,
        archive=_archive(("submission/main.py", b"candidate\n")),
        submit_paths=["submission"],
        submit_exclude=[],
    )
    manifest_path = saved.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("candidate_sequence")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(VersionStoreError, match="sequence fields"):
        VersionStore.open(store.root).list_versions()


def test_listing_rejects_conflicting_persistent_sequence_assignments(
    tmp_path: Path,
) -> None:
    store = VersionStore.open(_write_store(tmp_path / "store"))
    common = {
        "task_id": "task_alpha",
        "run_id": "run-conflict",
        "source": "checkpoint",
        "label": None,
        "submit_paths": ["submission"],
        "submit_exclude": [],
    }
    first = store.record_archive(
        round_id="checkpoint-1",
        archive=_archive(("submission/main.py", b"candidate A\n")),
        **common,
    )
    second = store.record_archive(
        round_id="checkpoint-2",
        archive=_archive(("submission/main.py", b"candidate B\n")),
        **common,
    )
    manifest_path = second.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["event_sequence"] = first.event_sequence
    manifest["candidate_sequence"] = first.candidate_sequence
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(VersionStoreError, match="sequence conflict"):
        VersionStore.open(store.root).list_versions()


def test_record_rejects_every_member_outside_submit_contract(tmp_path: Path) -> None:
    store = VersionStore.open(_write_store(tmp_path / "store"))
    archive = _archive(
        ("submission/gateway.py", b"safe\n"),
        ("auth.json", b'{"token":"secret"}\n'),
    )

    with pytest.raises(VersionStoreError, match="outside the submit contract"):
        store.record_archive(
            task_id="task_alpha",
            run_id="run-001",
            source="checkpoint",
            round_id="checkpoint-1",
            label="unsafe",
            archive=archive,
            submit_paths=["submission"],
            submit_exclude=["auth.json"],
        )

    assert not (tmp_path / "store/tasks/task-alpha/runs/run-001").exists()


@pytest.mark.parametrize(
    "archive",
    [
        _plain_tar(("submission/main.py", b"pass\n")),
        _archive(("submission/main.py", b"pass\n")) + b"trailing-data",
    ],
)
def test_store_requires_one_complete_gzip_stream(
    tmp_path: Path,
    archive: bytes,
) -> None:
    store = VersionStore.open(_write_store(tmp_path / "store"))

    with pytest.raises(VersionStoreError, match="gzip"):
        store.validate_archive(
            task_id="task_alpha",
            archive=archive,
            submit_paths=["submission"],
            submit_exclude=[],
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("run_id", "../escape", "run_id"),
        ("source", "unknown", "source"),
        ("round_id", "../round", "round_id"),
        ("label", "bad\nlabel", "label"),
        ("label", "x" * 81, "label"),
    ],
)
def test_record_rejects_unsafe_identity_fields(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    store = VersionStore.open(_write_store(tmp_path / "store"))
    kwargs = {
        "task_id": "task_alpha",
        "run_id": "run-001",
        "source": "checkpoint",
        "round_id": "checkpoint-1",
        "label": "safe",
        "archive": _archive(("submission/main.py", b"pass\n")),
        "submit_paths": ["submission"],
        "submit_exclude": [],
    }
    kwargs[field] = value

    with pytest.raises(VersionStoreError, match=message):
        store.record_archive(**kwargs)


def test_same_event_is_idempotent_but_different_bytes_collide(tmp_path: Path) -> None:
    store = VersionStore.open(_write_store(tmp_path / "store"))
    first_archive = _archive(("submission/main.py", b"first\n"))
    second_archive = _archive(("submission/main.py", b"second\n"))
    kwargs = {
        "task_id": "task_alpha",
        "run_id": "run-001",
        "source": "agent",
        "round_id": "agent-1",
        "label": None,
        "submit_paths": ["submission"],
        "submit_exclude": [],
    }

    first = store.record_archive(archive=first_archive, **kwargs)
    repeated = store.record_archive(archive=first_archive, **kwargs)

    assert repeated == first
    assert len(list(first.path.parent.iterdir())) == 1

    with pytest.raises(VersionStoreCollision, match="agent-1"):
        store.record_archive(archive=second_archive, **kwargs)


def test_archive_metadata_does_not_create_a_second_candidate(
    tmp_path: Path,
) -> None:
    store = VersionStore.open(_write_store(tmp_path / "store"))
    checkpoint_archive = _archive_with_metadata(
        ("submission/run.sh", b"#!/bin/sh\n", 0o755, 1),
        ("submission/main.py", b"print('same')\n", 0o644, 2),
    )
    evaluated_archive = _archive_with_metadata(
        ("submission/main.py", b"print('same')\n", 0o644, 200),
        ("submission/run.sh", b"#!/bin/sh\n", 0o755, 100),
    )
    assert hashlib.sha256(checkpoint_archive).digest() != hashlib.sha256(
        evaluated_archive
    ).digest()

    checkpoint = store.record_archive(
        task_id="task_alpha",
        run_id="run-001",
        source="checkpoint",
        round_id="checkpoint-1",
        label="before-public",
        archive=checkpoint_archive,
        submit_paths=["submission"],
        submit_exclude=[],
    )
    evaluated = store.record_archive(
        task_id="task_alpha",
        run_id="run-001",
        source="agent",
        round_id="agent-1",
        label=None,
        archive=evaluated_archive,
        submit_paths=["submission"],
        submit_exclude=[],
    )

    assert checkpoint.archive_sha256 != evaluated.archive_sha256
    assert checkpoint.content_sha256 == evaluated.content_sha256
    assert checkpoint.candidate_id == evaluated.candidate_id
    assert checkpoint.archive_path.read_bytes() == checkpoint_archive
    assert evaluated.archive_path.read_bytes() == evaluated_archive

    candidates = store.list_candidates(task="task_alpha", run_id="run-001")
    assert len(candidates) == 1
    assert candidates[0].candidate_id == checkpoint.candidate_id
    assert candidates[0].content_sha256 == checkpoint.content_sha256
    assert [event.round_id for event in candidates[0].events] == [
        "checkpoint-1",
        "agent-1",
    ]


def test_executable_mode_is_part_of_candidate_identity(tmp_path: Path) -> None:
    store = VersionStore.open(_write_store(tmp_path / "store"))
    common = {
        "task_id": "task_alpha",
        "run_id": "run-001",
        "label": None,
        "submit_paths": ["submission"],
        "submit_exclude": [],
    }
    regular = store.record_archive(
        source="agent",
        round_id="agent-1",
        archive=_archive_with_metadata(
            ("submission/run.sh", b"#!/bin/sh\n", 0o644, 1),
        ),
        **common,
    )
    executable = store.record_archive(
        source="agent",
        round_id="agent-2",
        archive=_archive_with_metadata(
            ("submission/run.sh", b"#!/bin/sh\n", 0o755, 1),
        ),
        **common,
    )

    assert regular.content_sha256 != executable.content_sha256
    assert regular.candidate_id != executable.candidate_id
    assert len(store.list_candidates(run_id="run-001")) == 2


def test_candidate_sequence_is_stable_when_events_are_filtered(
    tmp_path: Path,
) -> None:
    store = VersionStore.open(_write_store(tmp_path / "store"))
    common = {
        "task_id": "task_alpha",
        "run_id": "run-001",
        "label": None,
        "submit_paths": ["submission"],
        "submit_exclude": [],
    }
    store.record_archive(
        source="checkpoint",
        round_id="checkpoint-1",
        archive=_archive(("submission/main.py", b"first\n")),
        created_at=datetime(2026, 8, 18, 1, tzinfo=timezone.utc),
        **common,
    )
    second = store.record_archive(
        source="agent",
        round_id="agent-1",
        archive=_archive(("submission/main.py", b"second\n")),
        created_at=datetime(2026, 8, 18, 2, tzinfo=timezone.utc),
        **common,
    )

    filtered = store.list_candidates(run_id="run-001", source="agent")

    assert len(filtered) == 1
    assert filtered[0].candidate_id == second.candidate_id
    assert filtered[0].sequence == 2


def test_legacy_event_manifests_are_grouped_without_migration(
    tmp_path: Path,
) -> None:
    store = VersionStore.open(_write_store(tmp_path / "store"))
    archive = _archive(("submission/main.py", b"legacy\n"))
    for source, round_id, minute in (
        ("checkpoint", "checkpoint-1", 1),
        ("agent", "agent-1", 2),
    ):
        stored = store.record_archive(
            task_id="task_alpha",
            run_id="run-legacy",
            source=source,
            round_id=round_id,
            label="legacy" if source == "checkpoint" else None,
            archive=archive,
            submit_paths=["submission"],
            submit_exclude=[],
            created_at=datetime(2026, 8, 18, minute, tzinfo=timezone.utc),
        )
        manifest = json.loads(stored.path.joinpath("manifest.json").read_text())
        manifest.pop("candidate_id")
        manifest.pop("candidate_sequence")
        manifest.pop("content_sha256")
        manifest.pop("event_sequence")
        stored.path.joinpath("manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )

    candidates = VersionStore.open(store.root).list_candidates(
        run_id="run-legacy"
    )

    assert len(candidates) == 1
    assert candidates[0].sequence == 1
    assert [event.round_id for event in candidates[0].events] == [
        "checkpoint-1",
        "agent-1",
    ]
    assert all(event.content_sha256 for event in candidates[0].events)

    repeated = VersionStore.open(store.root).record_archive(
        task_id="task_alpha",
        run_id="run-legacy",
        source="agent",
        round_id="agent-1",
        label=None,
        archive=archive,
        submit_paths=["submission"],
        submit_exclude=[],
    )
    assert repeated.content_sha256 == candidates[0].content_sha256
    assert repeated.candidate_id == candidates[0].candidate_id


def test_idempotent_event_rejects_a_missing_or_corrupt_existing_archive(
    tmp_path: Path,
) -> None:
    store = VersionStore.open(_write_store(tmp_path / "store"))
    archive = _archive(("submission/main.py", b"first\n"))
    kwargs = {
        "task_id": "task_alpha",
        "run_id": "run-001",
        "source": "agent",
        "round_id": "agent-1",
        "label": None,
        "submit_paths": ["submission"],
        "submit_exclude": [],
    }
    first = store.record_archive(archive=archive, **kwargs)
    first.archive_path.write_bytes(b"corrupt")

    with pytest.raises(VersionStoreWriteError, match="integrity"):
        store.record_archive(archive=archive, **kwargs)


def test_verify_rejects_a_symlinked_archive_even_with_matching_bytes(
    tmp_path: Path,
) -> None:
    store = VersionStore.open(_write_store(tmp_path / "store"))
    archive = _archive(("submission/main.py", b"first\n"))
    saved = store.record_archive(
        task_id="task_alpha",
        run_id="run-001",
        source="agent",
        round_id="agent-1",
        label=None,
        archive=archive,
        submit_paths=["submission"],
        submit_exclude=[],
    )
    external = tmp_path / "external.tar.gz"
    external.write_bytes(archive)
    saved.archive_path.unlink()
    saved.archive_path.symlink_to(external)

    errors = store.verify([saved])

    assert len(errors) == 1
    assert "non-symlink" in errors[0]


def test_verify_rejects_a_version_directory_replaced_by_a_symlink(
    tmp_path: Path,
) -> None:
    store = VersionStore.open(_write_store(tmp_path / "store"))
    archive = _archive(("submission/main.py", b"first\n"))
    saved = store.record_archive(
        task_id="task_alpha",
        run_id="run-001",
        source="agent",
        round_id="agent-1",
        label=None,
        archive=archive,
        submit_paths=["submission"],
        submit_exclude=[],
    )
    displaced = tmp_path / "displaced-version"
    saved.path.rename(displaced)
    external = tmp_path / "external-version"
    external.mkdir()
    (external / "submission.tar.gz").write_bytes(archive)
    saved.path.symlink_to(external, target_is_directory=True)

    errors = store.verify([saved])

    assert len(errors) == 1
    assert "real directory" in errors[0] or "symlink" in errors[0]


def test_listing_ignores_unpublished_temporary_directories(tmp_path: Path) -> None:
    root = _write_store(tmp_path / "store")
    temporary = root / "tasks/task-alpha/runs/run-001/versions/.tmp-interrupted"
    temporary.mkdir(parents=True)
    (temporary / "manifest.json").write_text("{", encoding="utf-8")

    assert VersionStore.open(root).list_versions() == []


@pytest.mark.parametrize("component", ["runs", "run", "versions", "version"])
def test_runtime_symlinks_are_rejected_without_external_mutation(
    tmp_path: Path,
    component: str,
) -> None:
    root = _write_store(tmp_path / "store")
    task_root = root / "tasks/task-alpha"
    outside = tmp_path / f"outside-{component}"
    outside.mkdir()

    if component == "runs":
        (task_root / "runs").rmdir()
        (task_root / "runs").symlink_to(outside, target_is_directory=True)
    elif component == "run":
        (task_root / "runs/run-001").symlink_to(outside, target_is_directory=True)
    elif component == "versions":
        run_root = task_root / "runs/run-001"
        run_root.mkdir()
        (run_root / "versions").symlink_to(outside, target_is_directory=True)
    else:
        versions = task_root / "runs/run-001/versions"
        versions.mkdir(parents=True)
        (versions / "external-version").symlink_to(
            outside,
            target_is_directory=True,
        )

    store = VersionStore.open(root)
    with pytest.raises(VersionStoreError, match="symlink|real directory"):
        store.list_versions()

    if component != "version":
        with pytest.raises(VersionStoreError, match="symlink|real directory"):
            store.record_archive(
                task_id="task_alpha",
                run_id="run-001",
                source="checkpoint",
                round_id="checkpoint-1",
                label="safe",
                archive=_archive(("submission/main.py", b"pass\n")),
                submit_paths=["submission"],
                submit_exclude=[],
            )
    assert list(outside.iterdir()) == []


def test_store_lock_may_not_be_a_symlink(tmp_path: Path) -> None:
    root = _write_store(tmp_path / "store")
    external_lock = tmp_path / "external.lock"
    external_lock.write_bytes(b"external")
    (root / ".sforge-version-store.lock").symlink_to(external_lock)

    with pytest.raises(VersionStoreError, match="lock"):
        VersionStore.open(root).record_archive(
            task_id="task_alpha",
            run_id="run-001",
            source="checkpoint",
            round_id="checkpoint-1",
            label="safe",
            archive=_archive(("submission/main.py", b"pass\n")),
            submit_paths=["submission"],
            submit_exclude=[],
        )

    assert external_lock.read_bytes() == b"external"


def test_atomic_write_failure_removes_temporary_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sforge.harness import version_store as version_store_module

    root = _write_store(tmp_path / "store")
    store = VersionStore.open(root)
    archive = _archive(("submission/main.py", b"pass\n"))

    def fail_replace(*args, **kwargs) -> None:
        raise OSError("simulated rename failure")

    monkeypatch.setattr(version_store_module.os, "replace", fail_replace)

    with pytest.raises(VersionStoreError, match="write version"):
        store.record_archive(
            task_id="task_alpha",
            run_id="run-001",
            source="checkpoint",
            round_id="checkpoint-1",
            label="known-good",
            archive=archive,
            submit_paths=["submission"],
            submit_exclude=[],
        )

    versions_root = root / "tasks/task-alpha/runs/run-001/versions"
    assert not list(versions_root.glob(".tmp-*"))
    assert not list(versions_root.glob("*/manifest.json"))


def test_list_filters_versions_and_verify_detects_corruption(tmp_path: Path) -> None:
    store = VersionStore.open(
        _write_store(
            tmp_path / "store",
            registrations=(("task-alpha", "task_alpha"), ("task-beta", "task_beta")),
        )
    )
    archive = _archive(("submission/main.py", b"pass\n"))
    alpha = store.record_archive(
        task_id="task_alpha",
        run_id="run-a",
        source="final",
        round_id="final",
        label=None,
        archive=archive,
        submit_paths=["submission"],
        submit_exclude=[],
        created_at=datetime(2026, 8, 18, 1, tzinfo=timezone.utc),
    )
    store.record_archive(
        task_id="task_beta",
        run_id="run-b",
        source="checkpoint",
        round_id="checkpoint-1",
        label="beta",
        archive=archive,
        submit_paths=["submission"],
        submit_exclude=[],
        created_at=datetime(2026, 8, 18, 2, tzinfo=timezone.utc),
    )

    assert store.list_versions(task="task-alpha") == [alpha]
    assert store.list_versions(task="task_alpha", run_id="run-a", source="final") == [
        alpha
    ]
    assert store.list_versions(source="public") == []
    assert store.verify() == []

    (alpha.path / "submission.tar.gz").write_bytes(b"corrupt")

    errors = store.verify()
    assert len(errors) == 1
    assert alpha.version_id in errors[0]
    assert "SHA-256" in errors[0] or "size" in errors[0]


def test_final_event_seals_run_but_idempotent_retries_still_work(
    tmp_path: Path,
) -> None:
    store = VersionStore.open(_write_store(tmp_path / "store"))
    checkpoint_archive = _archive(("submission/main.py", b"checkpoint\n"))
    final_archive = _archive(("submission/main.py", b"final\n"))
    checkpoint = store.record_archive(
        task_id="task_alpha",
        run_id="sealed-run",
        source="checkpoint",
        round_id="checkpoint-1",
        label="candidate",
        archive=checkpoint_archive,
        submit_paths=["submission"],
        submit_exclude=[],
    )
    final = store.record_archive(
        task_id="task_alpha",
        run_id="sealed-run",
        source="final",
        round_id="final",
        label="final",
        archive=final_archive,
        submit_paths=["submission"],
        submit_exclude=[],
    )

    assert store.record_archive(
        task_id="task_alpha",
        run_id="sealed-run",
        source="checkpoint",
        round_id="checkpoint-1",
        label="retry-metadata-is-ignored",
        archive=checkpoint_archive,
        submit_paths=["submission"],
        submit_exclude=[],
    ) == checkpoint
    assert store.record_archive(
        task_id="task_alpha",
        run_id="sealed-run",
        source="final",
        round_id="final",
        label="retry-metadata-is-ignored",
        archive=final_archive,
        submit_paths=["submission"],
        submit_exclude=[],
    ) == final

    with pytest.raises(VersionStoreCollision, match="closed by its final event"):
        store.record_archive(
            task_id="task_alpha",
            run_id="sealed-run",
            source="checkpoint",
            round_id="checkpoint-2",
            label="too-late",
            archive=checkpoint_archive,
            submit_paths=["submission"],
            submit_exclude=[],
        )


def test_final_checkpoint_race_never_publishes_an_event_after_final(
    tmp_path: Path,
) -> None:
    store = VersionStore.open(_write_store(tmp_path / "store"))
    final_archive = _archive(("submission/main.py", b"final\n"))
    checkpoint_archive = _archive(("submission/main.py", b"late\n"))

    def save(source: str) -> str:
        try:
            store.record_archive(
                task_id="task_alpha",
                run_id="racing-run",
                source=source,
                round_id="final" if source == "final" else "checkpoint-1",
                label=source,
                archive=final_archive if source == "final" else checkpoint_archive,
                submit_paths=["submission"],
                submit_exclude=[],
            )
            return "saved"
        except VersionStoreCollision:
            return "closed"

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(save, ("final", "checkpoint")))

    versions = store.list_versions(task="task_alpha", run_id="racing-run")
    assert versions[-1].source == "final"
    assert sum(version.source == "final" for version in versions) == 1
    assert outcomes.count("saved") == len(versions)
    assert [version.source for version in versions] in [
        ["final"],
        ["checkpoint", "final"],
    ]


def test_verify_detects_a_false_normalized_content_identity(tmp_path: Path) -> None:
    root = _write_store(tmp_path / "store")
    saved = VersionStore.open(root).record_archive(
        task_id="task_alpha",
        run_id="run-a",
        source="checkpoint",
        round_id="checkpoint-1",
        label="candidate",
        archive=_archive(("submission/main.py", b"pass\n")),
        submit_paths=["submission"],
        submit_exclude=[],
    )
    manifest = json.loads(saved.path.joinpath("manifest.json").read_text())
    manifest["content_sha256"] = "0" * 64
    manifest["candidate_id"] = f"candidate-{'0' * 64}"
    saved.path.joinpath("manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    reopened = VersionStore.open(root)
    errors = reopened.verify()

    assert len(errors) == 1
    assert "content SHA-256 mismatch" in errors[0]


def test_identity_exclude_merges_notes_only_changes(tmp_path: Path) -> None:
    root = _write_store(tmp_path / "store")
    store = VersionStore.open(root)
    common = {
        "task_id": "task_alpha",
        "run_id": "run-identity",
        "source": "checkpoint",
        "label": None,
        "submit_paths": ["."],
        "submit_exclude": [],
        "content_identity_exclude": ["notes.md"],
    }
    code_a_notes_1 = _archive(
        ("submission/main.py", b"print('a')\n"),
        ("notes.md", b"hypothesis one\n"),
    )
    code_a_notes_2 = _archive(
        ("submission/main.py", b"print('a')\n"),
        ("notes.md", b"hypothesis two, code unchanged\n"),
    )
    code_b_notes_2 = _archive(
        ("submission/main.py", b"print('b')\n"),
        ("notes.md", b"hypothesis two, code unchanged\n"),
    )

    first = store.record_archive(
        round_id="checkpoint-1",
        archive=code_a_notes_1,
        created_at=datetime(2026, 8, 21, 10, 0, tzinfo=timezone.utc),
        **common,
    )
    notes_only = store.record_archive(
        round_id="checkpoint-2",
        archive=code_a_notes_2,
        created_at=datetime(2026, 8, 21, 10, 5, tzinfo=timezone.utc),
        **common,
    )
    code_change = store.record_archive(
        round_id="checkpoint-3",
        archive=code_b_notes_2,
        created_at=datetime(2026, 8, 21, 10, 10, tzinfo=timezone.utc),
        **common,
    )

    # A notes-only edit maps back to the existing logical candidate; a real
    # code edit still creates a new one.
    assert first.candidate_id == notes_only.candidate_id
    assert first.candidate_sequence == notes_only.candidate_sequence == 1
    assert code_change.candidate_sequence == 2
    assert code_change.candidate_id != first.candidate_id

    # The excluded file is still archived verbatim.
    assert (notes_only.path / "submission.tar.gz").read_bytes() == code_a_notes_2

    # The exclusions used are persisted per event and survive reopen + verify.
    manifest = json.loads((first.path / "manifest.json").read_text())
    assert manifest["content_identity_exclude"] == ["notes.md"]
    reopened = VersionStore.open(root)
    assert reopened.verify() == []
    candidates = reopened.list_candidates(task="task_alpha", run_id="run-identity")
    assert [len(candidate.events) for candidate in candidates] == [2, 1]


def test_identity_exclude_absent_keeps_legacy_identity(tmp_path: Path) -> None:
    root = _write_store(tmp_path / "store")
    store = VersionStore.open(root)
    common = {
        "task_id": "task_alpha",
        "run_id": "run-legacy",
        "source": "checkpoint",
        "label": None,
        "submit_paths": ["."],
        "submit_exclude": [],
    }
    with_notes_1 = _archive(
        ("submission/main.py", b"print('a')\n"),
        ("notes.md", b"one\n"),
    )
    with_notes_2 = _archive(
        ("submission/main.py", b"print('a')\n"),
        ("notes.md", b"two\n"),
    )

    first = store.record_archive(
        round_id="checkpoint-1",
        archive=with_notes_1,
        created_at=datetime(2026, 8, 21, 11, 0, tzinfo=timezone.utc),
        **common,
    )
    second = store.record_archive(
        round_id="checkpoint-2",
        archive=with_notes_2,
        created_at=datetime(2026, 8, 21, 11, 5, tzinfo=timezone.utc),
        **common,
    )

    assert first.candidate_id != second.candidate_id
    manifest = json.loads((first.path / "manifest.json").read_text())
    assert manifest["content_identity_exclude"] == []
    assert VersionStore.open(root).verify() == []


def test_identity_exclude_rejects_invalid_patterns(tmp_path: Path) -> None:
    root = _write_store(tmp_path / "store")
    store = VersionStore.open(root)

    with pytest.raises(VersionStoreError, match="identity exclusions"):
        store.record_archive(
            task_id="task_alpha",
            run_id="run-bad",
            source="checkpoint",
            round_id="checkpoint-1",
            label=None,
            archive=_archive(("submission/main.py", b"pass\n")),
            submit_paths=["."],
            submit_exclude=[],
            content_identity_exclude=["/"],
        )
