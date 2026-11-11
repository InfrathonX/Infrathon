from __future__ import annotations

import hashlib
from pathlib import Path


def expected_tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink() or "__pycache__" in path.parts:
            continue
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def test_tree_sha256_matches_frozen_v21_contract(judge_module, tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b.txt").write_bytes(b"bravo\n")
    (tmp_path / "a.txt").write_bytes(b"alpha\n")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "ignored.pyc").write_bytes(b"noise")
    (tmp_path / "alias").symlink_to(tmp_path / "a.txt")

    observed = judge_module.tree_sha256(tmp_path)

    assert observed == expected_tree_hash(tmp_path)
    (tmp_path / "nested" / "b.txt").write_bytes(b"changed\n")
    assert judge_module.tree_sha256(tmp_path) != observed


def test_tree_sha256_does_not_follow_external_symlinks(
    judge_module, tmp_path: Path
) -> None:
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "launch.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    outside = tmp_path / "hidden-nonce"
    outside.write_text("not part of the submission", encoding="utf-8")
    (submission / "escape").symlink_to(outside)

    before = judge_module.tree_sha256(submission)
    outside.write_text("changed secret", encoding="utf-8")

    assert judge_module.tree_sha256(submission) == before
