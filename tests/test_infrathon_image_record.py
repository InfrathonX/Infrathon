"""scripts/infrathon/image_record.py: source digests and build records for tasks without their own."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("infrathon_image_record", REPOSITORY_ROOT / "scripts/infrathon/image_record.py")
assert SPEC is not None and SPEC.loader is not None
record = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(record)
TASKS = sorted(p.parent.parent.name for p in (REPOSITORY_ROOT / "tasks/latest").glob("*/images/release.json"))


def test_release_specs_cover_their_builds_and_name_local_tags():
    assert len(TASKS) == 6
    for task in TASKS:
        spec = record.release_spec(task)
        files = {p.relative_to(REPOSITORY_ROOT).as_posix() for p in record.source_files(spec)}
        assert f"tasks/latest/{task}/scripts/build_images.sh" in files
        assert any(f.startswith(f"tasks/latest/{task}/images/") for f in files)
        assert "work" in spec["local_tags"] and "judge" in spec["local_tags"]
        if "holdout" in spec:
            assert "judge_public" in spec["local_tags"]
            assert (REPOSITORY_ROOT / "tasks/latest" / task / "images/judge/Dockerfile.private").is_file()


def test_published_record_is_not_a_source_file():
    """images/published.json is written after publishing, so it never feeds the source digest."""
    for task in TASKS:
        files = {p.relative_to(REPOSITORY_ROOT).as_posix() for p in record.source_files(record.release_spec(task))}
        assert f"tasks/latest/{task}/images/published.json" not in files


def test_check_refuses_a_stale_record(tmp_path, monkeypatch):
    task = TASKS[0]
    monkeypatch.setattr(record, "record_path", lambda t: tmp_path / "image-manifest.json")
    (tmp_path / "image-manifest.json").write_text(json.dumps({"source_digest": "0" * 64}))
    with pytest.raises(SystemExit, match="source changed"):
        record.check(task)


def test_excluded_files_do_not_change_the_source_digest():
    task = "minference-sparse-prefill-h100-v3-seed"
    spec = record.release_spec(task)
    assert spec["exclude"]
    files = {p.relative_to(REPOSITORY_ROOT).as_posix() for p in record.source_files(spec)}
    assert not files.intersection(spec["exclude"])
    assert f"tasks/latest/{task}/scripts/build_images.sh" in files
