from __future__ import annotations

import fcntl
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/replay_candidates.sh"
LOCK_HELPER = ROOT / "scripts/secure_flock_exec.py"
PREFLIGHT = ROOT / "scripts/preflight.py"
RUNTIME_ENTRY = ROOT / "scripts/sforge_runtime_entry.py"
README = ROOT / "README.md"


def _capture_launcher(tmp_path: Path, *arguments: str) -> list[str]:
    capture = tmp_path / "argv.txt"
    python = tmp_path / "python"
    python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$@\" >\"$REPLAY_TEST_CAPTURE\"\n",
        encoding="utf-8",
    )
    python.chmod(0o755)
    version_store = tmp_path / "versions"
    version_store.mkdir()
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--origin-run",
            "origin-run",
            "--measurement-set",
            "hidden-a",
            "--replay-run-id",
            "replay-run",
            "--version-store-root",
            str(version_store),
            "--output-dir",
            str(tmp_path / "output"),
            "--tasks-dir",
            str(ROOT / "tasks"),
            *arguments,
        ],
        text=True,
        capture_output=True,
        check=False,
        env={
            **os.environ,
            "REPOSITORY_ROOT": str(ROOT.parents[2]),
            "PYTHON_BIN": str(python),
            "REPLAY_TEST_CAPTURE": str(capture),
        },
    )
    assert result.returncode == 0, result.stderr
    return capture.read_text(encoding="utf-8").splitlines()


def test_replay_launcher_is_valid_shell_and_has_self_contained_help() -> None:
    assert SCRIPT.is_file()
    syntax = subprocess.run(
        ["bash", "-n", str(SCRIPT)], text=True, capture_output=True, check=False
    )
    assert syntax.returncode == 0, syntax.stderr
    help_result = subprocess.run(
        ["bash", str(SCRIPT), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "--origin-run" in help_result.stdout
    assert "--measurement-set" in help_result.stdout
    assert "--version-store-root" in help_result.stdout


def test_task_readme_documents_only_the_supported_candidate_batch_entry() -> None:
    if not README.is_file():
        pytest.skip("operator README not present")
    text = README.read_text(encoding="utf-8")
    if "scripts/replay_candidates.sh" not in text and not any(
        "\u4e00" <= char <= "\u9fff" for char in text
    ):
        # A released tree replaces the operator README with a public overview.
        pytest.skip("operator README not present (public overview README)")
    assert "scripts/replay_candidates.sh" in text
    assert "SFORGE_KVBENCH_NONCE_SOURCE" in text
    assert "--dry-run" in text and "--resume" in text
    assert "sforge_runtime_entry.py` 不是 batch launcher" in text
    assert "不会创建不可变快照" in text


def test_parser_fails_before_runtime_when_required_arguments_are_missing() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "--origin-run", "only-one-field"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--measurement-set is required" in result.stderr


def test_preflight_pins_the_actual_monorepo_root() -> None:
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.util,sys;"
                "spec=importlib.util.spec_from_file_location('kv_preflight_probe',sys.argv[1]);"
                "module=importlib.util.module_from_spec(spec);"
                "spec.loader.exec_module(module);"
                "print(module.EXPECTED_MLSYSFORGE_ROOT)"
            ),
            str(PREFLIGHT),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr
    assert Path(probe.stdout.strip()) == ROOT.parents[2]


def test_dry_run_uses_mount_hardened_adapter_and_forwards_range_flags(
    tmp_path: Path,
) -> None:
    argv = _capture_launcher(
        tmp_path,
        "--from-v",
        "3",
        "--to-v",
        "7",
        "--resume",
        "--dry-run",
        "--json",
    )
    assert argv[0] == str(ROOT / "scripts/sforge_runtime_entry.py")
    assert "replay-candidates" in argv
    assert "--from-v" in argv and "3" in argv
    assert "--to-v" in argv and "7" in argv
    assert "--resume" in argv
    assert "--dry-run" in argv
    assert any(value.startswith("task_runtime_adapter_sha256=") for value in argv)
    assert any(value.startswith("launcher_lib_sha256=") for value in argv)
    assert f"launcher_lib_sha256={hashlib.sha256((ROOT / 'scripts/lib.sh').read_bytes()).hexdigest()}" in argv
    assert any(value.startswith("lock_helper_sha256=") for value in argv)
    assert any(value.startswith("nonce_commitment_sha256=") for value in argv)
    assert any(value.startswith("oracle_tree_sha256=") for value in argv)
    assert any(value.startswith("mlsysforge_root=") for value in argv)


def test_full_replay_uses_formal_lease_preflight_and_trusted_mount_environment() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert source.index('[[ -f "${LAUNCHER_LIB}"') < source.index(
        'source "${SCRIPT_DIR}/lib.sh"'
    )
    assert '"judge_image_id=' not in source
    assert "/tmp/harbor-kv-qwen36-27b-1002/mlsysforge-v24-formal-run.lock" in source
    assert "secure_flock_exec.py" in source
    assert "--fd 8 --marker KV_REPLAY_LOCKED" in source
    assert "--verify-held" in source
    assert 'exec 8>' not in source
    assert 'scripts/preflight.py' in source
    assert 'assert_built_images_current' in source
    assert 'SFORGE_KVBENCH_NONCE_SOURCE' in source
    assert 'SFORGE_KVBENCH_EVAL_ROOT' in source
    assert 'scripts/sforge_runtime_entry.py' in source
    assert "mktemp -d -t sforge-kv-replay-preflight" in source
    assert "candidate-replay-preflight/${TASK_ID}" not in source


def test_batch_does_not_hold_the_nested_runner_measurement_lock() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "benchmark-measurement.lock" not in source
    assert "Each Judge attempt acquires the nested measurement lock" in source


def test_runtime_revalidates_full_preflight_around_each_judge_attempt() -> None:
    launcher = SCRIPT.read_text(encoding="utf-8")
    runtime = RUNTIME_ENTRY.read_text(encoding="utf-8")
    assert "SFORGE_KVBENCH_REPLAY_PREFLIGHT" in launcher
    assert "SFORGE_KVBENCH_REPLAY_PREFLIGHT_SHA256" in launcher
    assert "def _verify_replay_live_artifacts" in runtime
    assert "_verify_replay_live_artifacts()" in runtime
    assert 'exec "${command[@]}"' not in launcher


def test_secure_lock_helper_rejects_symlink_without_touching_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    target.write_text("keep", encoding="utf-8")
    lock = tmp_path / "lock"
    lock.symlink_to(target)
    result = subprocess.run(
        [
            sys.executable,
            str(LOCK_HELPER),
            "--lock",
            str(lock),
            "--fd",
            "8",
            "--marker",
            "KV_TEST_LOCKED",
            "--",
            "/bin/true",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert target.read_text(encoding="utf-8") == "keep"


def test_secure_lock_helper_verifies_and_adopts_inherited_lease(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "lock"
    descriptor = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(LOCK_HELPER),
                "--lock",
                str(lock),
                "--fd",
                str(descriptor),
                "--marker",
                "KV_TEST_LOCKED",
                "--verify-held",
            ],
            text=True,
            capture_output=True,
            check=False,
            pass_fds=(descriptor,),
            env={
                **os.environ,
                "KV_TEST_LOCKED": "1",
                "SFORGE_SECURE_LOCK_FD": str(descriptor),
            },
        )
        assert result.returncode == 0, result.stderr
        contender = os.open(lock, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(contender)
    finally:
        os.close(descriptor)


def test_secure_lock_helper_rejects_user_owned_world_writable_parent(
    tmp_path: Path,
) -> None:
    unsafe_parent = tmp_path / "unsafe-lock-parent"
    unsafe_parent.mkdir(mode=0o777)
    unsafe_parent.chmod(0o777)
    lock = unsafe_parent / "lock"
    result = subprocess.run(
        [
            sys.executable,
            str(LOCK_HELPER),
            "--lock",
            str(lock),
            "--fd",
            "8",
            "--marker",
            "KV_TEST_LOCKED",
            "--",
            "/bin/true",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert not lock.exists()


def test_version_store_symlink_is_rejected_before_runtime(tmp_path: Path) -> None:
    real_store = tmp_path / "real-store"
    real_store.mkdir()
    linked_store = tmp_path / "linked-store"
    linked_store.symlink_to(real_store, target_is_directory=True)
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--origin-run",
            "origin",
            "--measurement-set",
            "preview",
            "--version-store-root",
            str(linked_store),
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "version store is missing or unsafe" in result.stderr
