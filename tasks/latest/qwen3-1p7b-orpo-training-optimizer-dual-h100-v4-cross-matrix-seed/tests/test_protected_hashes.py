from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from conftest import TASK_ROOT, judge_only_file


def _load_verifier():
    path = TASK_ROOT / "images/judge/verifier/verify_orpo.py"
    name = f"orpo_verify_protected_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest(entries: list[tuple[str, str]]) -> str:
    lines = ["# mlsysforge-protected-files-v1", "# sha256  relative-path"]
    lines.extend(f"{digest}  {relative}" for digest, relative in entries)
    return "\n".join(lines) + "\n"


def _write_manifest(root: Path, content: str) -> Path:
    manifest = root / "opt/orpo_judge/protected_hashes.txt"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(content, encoding="utf-8")
    return manifest


def test_canonical_task_manifest_is_current():
    judge_only_file()  # the manifest covers the Judge-only hidden cases
    tool = TASK_ROOT / "scripts/generate_protected_hashes.py"
    result = subprocess.run(
        [sys.executable, str(tool), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_verifier_accepts_matching_regular_file(tmp_path):
    verifier = _load_verifier()
    target = tmp_path / "app/workdir/orpo_lab/runtime/api.py"
    target.parent.mkdir(parents=True)
    payload = b"trusted runtime\n"
    target.write_bytes(payload)
    manifest = _write_manifest(
        tmp_path, _manifest([(_digest(payload), "app/workdir/orpo_lab/runtime/api.py")])
    )
    verifier.verify_protected_files(manifest, container_root=tmp_path)


def test_verifier_rejects_tampered_protected_content(tmp_path):
    verifier = _load_verifier()
    target = tmp_path / "app/workdir/orpo_lab/runtime/api.py"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"modified\n")
    manifest = _write_manifest(
        tmp_path, _manifest([(_digest(b"original\n"), "app/workdir/orpo_lab/runtime/api.py")])
    )
    with pytest.raises(verifier.ProtectedIntegrityError):
        verifier.verify_protected_files(manifest, container_root=tmp_path)


@pytest.mark.parametrize(
    "manifest_text",
    [
        "PENDING_FORMAL_CALIBRATION\n",
        _manifest(
            [
                (_digest(b"one"), "app/workdir/orpo_lab/runtime/api.py"),
                (_digest(b"two"), "app/workdir/orpo_lab/runtime/api.py"),
            ]
        ),
        _manifest([(_digest(b"x"), "app/workdir/orpo_lab/runtime/../api.py")]),
    ],
)
def test_verifier_rejects_pending_duplicate_and_traversal_manifests(tmp_path, manifest_text):
    verifier = _load_verifier()
    manifest = _write_manifest(tmp_path, manifest_text)
    with pytest.raises(verifier.ProtectedIntegrityError):
        verifier.verify_protected_files(manifest, container_root=tmp_path)
