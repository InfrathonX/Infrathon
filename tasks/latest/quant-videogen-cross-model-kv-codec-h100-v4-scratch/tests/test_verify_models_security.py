from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
VERIFY_MODELS = ROOT / "scripts/verify_models.py"


def _load_verifier() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "qvg_verify_models_security_test", VERIFY_MODELS
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tree_manifest_records_directory_file_mode_and_content(tmp_path: Path) -> None:
    verifier = _load_verifier()
    root = tmp_path / "bundle"
    nested = root / "nested"
    nested.mkdir(parents=True)
    nested.chmod(0o750)
    artifact = nested / "weights.bin"
    artifact.write_bytes(b"weights")
    artifact.chmod(0o640)

    result = verifier.verify_tree_files(root, ["nested/weights.bin"])

    entries = {row["path"]: row for row in result["entries"]}
    assert entries["nested"] == {
        "path": "nested",
        "type": "directory",
        "mode": 0o750,
    }
    assert entries["nested/weights.bin"]["type"] == "file"
    assert entries["nested/weights.bin"]["mode"] == 0o640
    assert entries["nested/weights.bin"]["sha256"] == verifier.sha256_bytes(
        b"weights"
    )


def test_tree_manifest_rejects_symlink_to_directory_instead_of_skipping_it(
    tmp_path: Path,
) -> None:
    verifier = _load_verifier()
    root = tmp_path / "bundle"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "hidden.bin").write_bytes(b"outside")
    (root / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symlink"):
        verifier.verify_tree_files(root, [])


def test_tree_manifest_rejects_file_replaced_by_symlink_before_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    verifier = _load_verifier()
    root = tmp_path / "bundle"
    root.mkdir()
    victim = root / "victim.bin"
    victim.write_bytes(b"trusted")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"hostile")
    original_open = verifier.os.open
    replaced = False

    def racing_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replaced
        if path == "victim.bin" and dir_fd is not None and not replaced:
            replaced = True
            victim.rename(root / "trusted.bin")
            victim.symlink_to(outside)
        if dir_fd is None:
            return original_open(path, flags, mode)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(verifier.os, "open", racing_open)
    with pytest.raises(RuntimeError, match="unsafe|changed|symlink"):
        verifier.verify_tree_files(root, [])
    assert replaced is True


def test_tree_manifest_rejects_file_mutation_while_hashing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    verifier = _load_verifier()
    root = tmp_path / "bundle"
    root.mkdir()
    victim = root / "victim.bin"
    victim.write_bytes(b"a" * (1024 * 1024 + 1))
    original_read = verifier.os.read
    mutated = False

    def racing_read(descriptor: int, count: int) -> bytes:
        nonlocal mutated
        chunk = original_read(descriptor, count)
        if chunk and not mutated:
            mutated = True
            with victim.open("ab") as stream:
                stream.write(b"changed")
        return chunk

    monkeypatch.setattr(verifier.os, "read", racing_read)
    with pytest.raises(RuntimeError, match="changed"):
        verifier.verify_tree_files(root, [])
    assert mutated is True


class _FrozenScandir:
    def __init__(self, entries: list[os.DirEntry[str]]) -> None:
        self._entries = iter(entries)

    def __enter__(self) -> "_FrozenScandir":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def __iter__(self) -> "_FrozenScandir":
        return self

    def __next__(self) -> os.DirEntry[str]:
        return next(self._entries)

    def close(self) -> None:
        return None


def test_tree_manifest_rejects_directory_mutation_during_enumeration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    verifier = _load_verifier()
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "initial.bin").write_bytes(b"initial")
    original_scandir = verifier.os.scandir
    mutated = False

    def racing_scandir(path: object) -> object:
        nonlocal mutated
        iterator = original_scandir(path)
        if not mutated:
            with iterator:
                entries = list(iterator)
            (root / "late.bin").write_bytes(b"late")
            mutated = True
            return _FrozenScandir(entries)
        return iterator

    monkeypatch.setattr(verifier.os, "scandir", racing_scandir)
    with pytest.raises(RuntimeError, match="directory changed"):
        verifier.verify_tree_files(root, [])
    assert mutated is True


def test_mounted_tree_lock_covers_hidden_inputs_and_complete_metric_cache(
    tmp_path: Path,
) -> None:
    verifier = _load_verifier()
    artifact_root = tmp_path / "artifacts"
    hidden = artifact_root / "hidden"
    cache = artifact_root / "staging/vbench-cache"
    hidden.mkdir(parents=True)
    cache.mkdir(parents=True)
    (hidden / "golden.mp4").write_bytes(b"golden-a")
    (cache / "extra-visible-cache.bin").write_bytes(b"cache-a")

    first = verifier.verify_mounted_trees(
        artifact_root,
        {
            "hidden": "hidden",
            "vbench_cache": "staging/vbench-cache",
        },
    )
    (hidden / "golden.mp4").write_bytes(b"golden-b")
    second = verifier.verify_mounted_trees(
        artifact_root,
        {
            "hidden": "hidden",
            "vbench_cache": "staging/vbench-cache",
        },
    )

    assert set(first) == {"hidden", "vbench_cache"}
    assert first["hidden"]["tree_sha256"] != second["hidden"]["tree_sha256"]
    assert (
        first["vbench_cache"]["tree_sha256"]
        == second["vbench_cache"]["tree_sha256"]
    )


def test_model_lock_writer_refuses_symlink_without_touching_target(
    tmp_path: Path,
) -> None:
    verifier = _load_verifier()
    target = tmp_path / "target.json"
    target.write_text("keep", encoding="utf-8")
    output = tmp_path / "model-lock.json"
    output.symlink_to(target)

    with pytest.raises(RuntimeError, match="overwrite"):
        verifier.atomic_write_text(output, "hostile\n")

    assert target.read_text(encoding="utf-8") == "keep"
