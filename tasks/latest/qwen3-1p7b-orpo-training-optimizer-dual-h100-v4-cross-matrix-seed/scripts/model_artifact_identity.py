#!/usr/bin/env python3
"""Hash the exact Qwen3 files mounted into the ORPO Judge."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat


MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
REQUIRED_MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "merges.txt",
    "vocab.json",
)


def _stable_stat_tuple(
    value: os.stat_result,
) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def stable_file_identity(
    directory_fd: int,
    name: str,
    *,
    display_path: str,
    expected: os.stat_result,
) -> tuple[int, str]:
    """Hash one non-symlink file and reject a concurrent replacement/write."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise SystemExit(
            f"model file is unavailable or unsafe: {display_path}"
        ) from exc
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SystemExit(
                f"model artifact is not a regular file: {display_path}"
            )
        if (
            before.st_dev,
            before.st_ino,
        ) != (expected.st_dev, expected.st_ino):
            raise SystemExit(
                f"model artifact changed during traversal: {display_path}"
            )
        for chunk in iter(lambda: os.read(descriptor, 8 * 1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(descriptor)
        try:
            linked = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as exc:
            raise SystemExit(f"model artifact disappeared: {display_path}") from exc
        if (
            _stable_stat_tuple(before) != _stable_stat_tuple(after)
            or (after.st_dev, after.st_ino) != (linked.st_dev, linked.st_ino)
            or not stat.S_ISREG(linked.st_mode)
        ):
            raise SystemExit(
                f"model artifact changed while hashing: {display_path}"
            )
        return after.st_size, digest.hexdigest()
    finally:
        os.close(descriptor)


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def mounted_tree_manifest(
    root: Path,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    """Describe every path visible through the model-cache bind mount."""

    rows: list[dict[str, object]] = []
    regular_by_path: dict[str, dict[str, object]] = {}
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    directory_flags |= getattr(os, "O_CLOEXEC", 0)

    def visit(directory_fd: int, prefix: str) -> None:
        before_directory = os.fstat(directory_fd)
        if not stat.S_ISDIR(before_directory.st_mode):
            raise SystemExit(f"model tree directory became unsafe: {prefix or '.'}")
        try:
            with os.scandir(directory_fd) as iterator:
                entries = sorted(iterator, key=lambda item: item.name)
        except OSError as exc:
            raise SystemExit("cannot traverse mounted model cache") from exc
        for entry in entries:
            relative = f"{prefix}/{entry.name}" if prefix else entry.name
            try:
                observed = os.stat(
                    entry.name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise SystemExit(f"model tree entry disappeared: {relative}") from exc
            mode = stat.S_IMODE(observed.st_mode)
            if stat.S_ISDIR(observed.st_mode):
                try:
                    child_fd = os.open(
                        entry.name,
                        directory_flags,
                        dir_fd=directory_fd,
                    )
                except OSError as exc:
                    raise SystemExit(
                        f"model directory is unavailable or unsafe: {relative}"
                    ) from exc
                opened = os.fstat(child_fd)
                if (
                    not stat.S_ISDIR(opened.st_mode)
                    or (opened.st_dev, opened.st_ino)
                    != (observed.st_dev, observed.st_ino)
                ):
                    os.close(child_fd)
                    raise SystemExit(
                        f"model directory changed during traversal: {relative}"
                    )
                rows.append({"path": relative, "type": "directory", "mode": mode})
                try:
                    visit(child_fd, relative)
                    linked_after = os.stat(
                        entry.name,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                    opened_after = os.fstat(child_fd)
                except OSError as exc:
                    raise SystemExit(
                        f"model directory disappeared during traversal: {relative}"
                    ) from exc
                finally:
                    os.close(child_fd)
                if (
                    _stable_stat_tuple(opened) != _stable_stat_tuple(opened_after)
                    or (opened_after.st_dev, opened_after.st_ino)
                    != (linked_after.st_dev, linked_after.st_ino)
                    or not stat.S_ISDIR(linked_after.st_mode)
                ):
                    raise SystemExit(
                        f"model directory changed during traversal: {relative}"
                    )
            elif stat.S_ISREG(observed.st_mode):
                size, digest = stable_file_identity(
                    directory_fd,
                    entry.name,
                    display_path=relative,
                    expected=observed,
                )
                row = {
                    "path": relative,
                    "type": "file",
                    "mode": mode,
                    "size": size,
                    "sha256": digest,
                }
                rows.append(row)
                regular_by_path[relative] = row
            elif stat.S_ISLNK(observed.st_mode):
                try:
                    target = os.readlink(entry.name, dir_fd=directory_fd)
                    resolved = (root / relative).resolve(strict=True)
                    resolved.relative_to(root)
                    linked_after = os.stat(
                        entry.name,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                except (OSError, ValueError) as exc:
                    raise SystemExit(
                        f"model symlink is broken or escapes cache: {relative}"
                    ) from exc
                if _stable_stat_tuple(observed) != _stable_stat_tuple(linked_after):
                    raise SystemExit(f"model symlink changed during traversal: {relative}")
                rows.append(
                    {
                        "path": relative,
                        "type": "symlink",
                        "mode": mode,
                        "target": target,
                    }
                )
            else:
                raise SystemExit(f"unsupported model tree entry: {relative}")

        after_directory = os.fstat(directory_fd)
        if _stable_stat_tuple(before_directory) != _stable_stat_tuple(after_directory):
            raise SystemExit(
                f"model directory changed during traversal: {prefix or '.'}"
            )

    try:
        root_fd = os.open(root, directory_flags)
    except OSError as exc:
        raise SystemExit("mounted model cache root is unavailable or unsafe") from exc
    try:
        opened_root = os.fstat(root_fd)
        visit(root_fd, "")
        linked_root = os.stat(root, follow_symlinks=False)
        opened_root_after = os.fstat(root_fd)
        if (
            _stable_stat_tuple(opened_root) != _stable_stat_tuple(opened_root_after)
            or (opened_root_after.st_dev, opened_root_after.st_ino)
            != (linked_root.st_dev, linked_root.st_ino)
            or not stat.S_ISDIR(linked_root.st_mode)
        ):
            raise SystemExit("mounted model cache root changed during traversal")
    finally:
        os.close(root_fd)
    rows.sort(key=lambda row: str(row["path"]))
    return rows, regular_by_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-repo", required=True, type=Path)
    parser.add_argument("--private-data-root", type=Path)
    args = parser.parse_args()
    if args.model_repo.is_symlink():
        raise SystemExit("model repo must not be a symlink")
    root = args.model_repo.resolve(strict=True)
    if not root.is_dir() or root.name != "models--Qwen--Qwen3-1.7B":
        raise SystemExit("model repo is not the frozen Qwen3-1.7B cache root")
    snapshot = (root / "snapshots" / MODEL_REVISION).resolve(strict=True)
    tree_rows, regular_by_path = mounted_tree_manifest(root)
    private_data_rows: list[dict[str, object]] | None = None
    if args.private_data_root is not None:
        if args.private_data_root.is_symlink():
            raise SystemExit("private data root must not be a symlink")
        private_data_root = args.private_data_root.resolve(strict=True)
        if not private_data_root.is_dir():
            raise SystemExit("private data root is not a directory")
        private_data_rows, _private_regular_by_path = mounted_tree_manifest(
            private_data_root
        )
    rows: list[dict[str, object]] = []
    for relative in REQUIRED_MODEL_FILES:
        requested = snapshot / relative
        resolved = requested.resolve(strict=True)
        try:
            resolved_relative = resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise SystemExit(f"model artifact escapes cache root: {relative}") from exc
        if not resolved.is_file():
            raise SystemExit(f"model artifact is not a file: {relative}")
        resolved_row = regular_by_path.get(resolved_relative)
        if resolved_row is None:
            raise SystemExit(f"resolved model artifact was not attested: {relative}")
        rows.append(
            {
                "path": relative,
                "resolved": resolved_relative,
                "size": resolved_row["size"],
                "sha256": resolved_row["sha256"],
            }
        )
    blob_rows = [
        {"path": row["path"], "resolved": row["resolved"]}
        for row in rows
    ]
    payload = {
        "model_files_content_sha256": canonical_sha256(rows),
        "model_mount_tree_sha256": canonical_sha256(tree_rows),
        "model_resolved_blobs_sha256": canonical_sha256(blob_rows),
    }
    if private_data_rows is not None:
        payload["private_data_mount_tree_sha256"] = canonical_sha256(
            private_data_rows
        )
    print(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
