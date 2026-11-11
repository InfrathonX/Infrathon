from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/replay_candidates.sh"
LOCK_HELPER = ROOT / "scripts/secure_flock_exec.py"
MODEL_IDENTITY = ROOT / "scripts/model_artifact_identity.py"
RUNTIME_IDENTITY = ROOT / "scripts/runtime_identity.py"
RUNTIME_ENTRY = ROOT / "scripts/sforge_dual_h100_entry.py"
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
            "SFORGE_PYTHON": str(python),
            "REPLAY_TEST_CAPTURE": str(capture),
            # P4: the launcher has no default pair, so even a dry run has to
            # name one.  Placeholder identifiers: the dry run takes no lease.
            "ORPO_GPU_UUIDS": (
                "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa,"
                "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
            ),
            "ORPO_GPU_INDICES": "0,1",
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
    text = README.read_text(encoding="utf-8")
    if "is not a batch launcher" not in text:
        pytest.skip("the maintainer README is not shipped")
    assert "scripts/replay_candidates.sh" in text
    assert "ORPO_MODEL_HOST_PATH" in text
    assert "ORPO_PRIVATE_DATA_HOST_ROOT" in text
    assert "--dry-run" in text and "--resume" in text
    assert "sforge_dual_h100_entry.py` is not a batch launcher" in text
    assert "does not snapshot" in text
    assert "kernel boot" in text and "power/clock policy" in text


def test_parser_fails_before_runtime_when_required_arguments_are_missing() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "--origin-run", "only-one-field"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--measurement-set is required" in result.stderr


def test_dry_run_uses_dual_h100_adapter_and_forwards_selection_flags(
    tmp_path: Path,
) -> None:
    argv = _capture_launcher(
        tmp_path,
        "--candidate",
        "v2",
        "--candidate",
        "candidate-abcd",
        "--resume",
        "--dry-run",
        "--json",
    )
    assert argv[0] == str(ROOT / "scripts/sforge_dual_h100_entry.py")
    assert argv[1:7] == [
        "--output-dir",
        str(tmp_path / "output"),
        "--tasks-dir",
        str(ROOT / "tasks"),
        "replay-candidates",
        "--backend",
    ]
    assert "--candidate" in argv
    assert argv.count("--candidate") == 2
    assert "--resume" in argv
    assert "--dry-run" in argv
    assert "--evaluator-context" in argv
    assert any(value.startswith("task_runtime_adapter_sha256=") for value in argv)
    assert any(value.startswith("launcher_lib_sha256=") for value in argv)
    assert f"launcher_lib_sha256={hashlib.sha256((ROOT / 'scripts/lib.sh').read_bytes()).hexdigest()}" in argv
    assert any(value.startswith("lock_helper_sha256=") for value in argv)
    assert any(value.startswith("runtime_identity_helper_sha256=") for value in argv)
    assert any(value.startswith("host_runtime_identity_sha256=") for value in argv)
    assert any(value.startswith("public_manifest_sha256=") for value in argv)
    assert any(value.startswith("private_cross_1024_manifest_sha256=") for value in argv)


def test_full_replay_holds_managed_gpu_lease_and_runs_formal_preflight() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert source.index('[[ -f "${LAUNCHER_LIB}"') < source.index('source "${SCRIPT_DIR}/lib.sh"')
    assert '"judge_image_id=' not in source
    assert "/tmp/qwen3-orpo-dual-h100-managed-judge-${replay_pair}.lock" in source
    assert "secure_flock_exec.py" in source
    assert "--fd 9 --marker ORPO_REPLAY_LOCKED" in source
    assert "--verify-held" in source
    assert 'exec 9>' not in source
    assert 'scripts/preflight.py' in source
    assert 'scripts/sforge_dual_h100_entry.py' in source
    assert 'SFORGE_TASKS_DIR' in source
    assert 'SFORGE_LOG_DIR' in source
    assert "mktemp -d -t sforge-orpo-replay-preflight" in source
    assert "model_files_content_sha256" in source
    assert "model_resolved_blobs_sha256" in source
    assert "model_mount_tree_sha256" in source
    assert "runtime_identity.py" in source
    assert "host_runtime_identity_sha256" in source
    assert "umask 077" in source


def test_runtime_revalidates_live_assets_and_host_identity_around_each_judge() -> None:
    launcher = SCRIPT.read_text(encoding="utf-8")
    runtime = RUNTIME_ENTRY.read_text(encoding="utf-8")
    assert "ORPO_REPLAY_EXPECTED_MODEL_MOUNT_SHA256" in launcher
    assert "ORPO_REPLAY_EXPECTED_PRIVATE_DATA_MOUNT_SHA256" in launcher
    assert "ORPO_REPLAY_MODEL_IDENTITY_HELPER_SHA256" in launcher
    assert "ORPO_REPLAY_RUNTIME_IDENTITY_PATH" in launcher
    assert "ORPO_REPLAY_RUNTIME_IDENTITY_SHA256" in launcher
    assert "ORPO_REPLAY_RUNTIME_IDENTITY_HELPER_SHA256" in launcher
    assert "private_data_mount_tree_sha256" in launcher
    assert "def _verify_replay_live_artifacts" in runtime
    assert "_verify_replay_live_artifacts()" in runtime
    assert "DockerBackend.cleanup_container" in runtime
    assert "runtime_identity.py" in runtime and "--verify-baseline" in runtime


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("boot", "boot_id", "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        ("docker", "daemon_id", "NEW-DAEMON"),
        ("gpus", "driver_version", "999.99"),
        ("gpus", "power_limit", "650.00"),
        ("topology", "matrix_sha256", "f" * 64),
    ],
)
def test_runtime_identity_rejects_changed_host_contract(
    tmp_path: Path, section: str, field: str, replacement: str
) -> None:
    spec = importlib.util.spec_from_file_location(
        "orpo_runtime_identity_test", RUNTIME_IDENTITY
    )
    assert spec is not None and spec.loader is not None
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    observed = {
        "boot": {
            "boot_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "kernel": {"release": "6.8.0", "version": "one", "machine": "x86_64"},
        },
        "docker": {"daemon_id": "DAEMON", "default_runtime": "runc"},
        "gpus": [
            {
                "index": "6",
                "uuid": "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "name": "NVIDIA H100 NVL",
                "driver_version": "570.00",
                "power_limit": "600.00",
            }
        ],
        "topology": {"matrix_sha256": "e" * 64},
    }
    baseline = helper.build_identity_document(observed)
    changed = json.loads(json.dumps(observed))
    if section == "gpus":
        changed[section][0][field] = replacement
    else:
        changed[section][field] = replacement
    current = helper.build_identity_document(changed)

    assert baseline["identity_sha256"] != current["identity_sha256"]
    with pytest.raises(RuntimeError, match="host runtime identity changed"):
        helper.verify_same_identity(baseline, current)


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
            "9",
            "--marker",
            "ORPO_TEST_LOCKED",
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
                "ORPO_TEST_LOCKED",
                "--verify-held",
            ],
            text=True,
            capture_output=True,
            check=False,
            pass_fds=(descriptor,),
            env={
                **os.environ,
                "ORPO_TEST_LOCKED": "1",
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
            "9",
            "--marker",
            "ORPO_TEST_LOCKED",
            "--",
            "/bin/true",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert not lock.exists()


def test_model_identity_changes_when_resolved_model_content_changes(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "models--Qwen--Qwen3-1.7B"
    blobs = repo / "blobs"
    snapshot = repo / "snapshots" / "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    names = (
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
    for index, name in enumerate(names):
        blob = blobs / f"{index:064x}"
        blob.write_bytes(f"payload-{index}".encode())
        (snapshot / name).symlink_to(blob)

    def identity() -> dict[str, str]:
        result = subprocess.run(
            [sys.executable, str(MODEL_IDENTITY), "--model-repo", str(repo)],
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    before = identity()
    (blobs / f"{3:064x}").write_bytes(b"mutated")
    after = identity()
    assert before["model_files_content_sha256"] != after["model_files_content_sha256"]
    assert before["model_resolved_blobs_sha256"] == after["model_resolved_blobs_sha256"]


def test_model_identity_covers_extra_files_visible_in_the_mounted_cache(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "models--Qwen--Qwen3-1.7B"
    blobs = repo / "blobs"
    snapshot = repo / "snapshots" / "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    refs = repo / "refs"
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    refs.mkdir()
    names = (
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
    for index, name in enumerate(names):
        blob = blobs / f"{index:064x}"
        blob.write_bytes(f"payload-{index}".encode())
        (snapshot / name).symlink_to(blob)
    extra = refs / "main"
    extra.write_text("first", encoding="utf-8")

    def identity() -> dict[str, str]:
        result = subprocess.run(
            [sys.executable, str(MODEL_IDENTITY), "--model-repo", str(repo)],
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    before = identity()
    extra.write_text("second", encoding="utf-8")
    after = identity()
    assert before["model_files_content_sha256"] == after["model_files_content_sha256"]
    assert before["model_mount_tree_sha256"] != after["model_mount_tree_sha256"]


def test_model_identity_refuses_directory_replaced_by_symlink_during_walk(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = tmp_path / "models--Qwen--Qwen3-1.7B"
    mutable = repo / "mutable"
    outside = tmp_path / "outside"
    mutable.mkdir(parents=True)
    outside.mkdir()
    (mutable / "inside").write_text("inside", encoding="utf-8")
    (outside / "outside").write_text("outside", encoding="utf-8")

    spec = importlib.util.spec_from_file_location("orpo_model_identity_test", MODEL_IDENTITY)
    assert spec is not None and spec.loader is not None
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    real_open = helper.os.open
    replaced = False

    def racing_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replaced
        if (
            path == "mutable"
            and dir_fd is not None
            and flags & getattr(helper.os, "O_DIRECTORY", 0)
            and not replaced
        ):
            replaced = True
            mutable.rename(repo / "mutable-before-race")
            mutable.symlink_to(outside, target_is_directory=True)
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(helper.os, "open", racing_open)
    with pytest.raises(SystemExit):
        helper.mounted_tree_manifest(repo.resolve())
    assert replaced is True


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
