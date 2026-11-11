from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from conftest import release_file


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/replay_candidates.sh"
RUNTIME_ENTRY = ROOT / "scripts/sforge_runtime_entry.py"
README = ROOT / "README.md"


def _launcher_env(**overrides: str) -> dict[str, str]:
    env = dict(os.environ)
    for name in ("JUDGE_GPUS", "ORACLE_TIMEOUT", "JUDGE_CPU_LIMIT", "JUDGE_MEM_LIMIT"):
        env.pop(name, None)
    env.update(overrides)
    return env


def _capture_launcher(tmp_path: Path, *arguments: str) -> list[str]:
    release_file()  # the launcher reads the Judge-only release config
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
        env=_launcher_env(
            REPOSITORY_ROOT=str(ROOT.parents[2]),
            SFORGE_PYTHON=str(python),
            REPLAY_TEST_CAPTURE=str(capture),
        ),
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
    assert "JUDGE_GPUS" in help_result.stdout
    assert "QVG_JUDGE_GPU=" not in help_result.stdout


def test_task_readme_documents_only_the_supported_candidate_batch_entry() -> None:
    text = README.read_text(encoding="utf-8")
    if "## Candidate batch replay" not in text:
        pytest.skip("this README does not carry the maintainer replay section")
    assert "scripts/replay_candidates.sh" in text and "scripts/eval_archive.sh" in text
    assert "QVG_ARTIFACT_ROOT" in text and "JUDGE_GPUS" in text
    assert "QVG_JUDGE_GPU=" not in text
    assert "--dry-run" in text and "--resume" in text
    assert "sforge_runtime_entry.py` is not a batch launcher" in text
    assert "does not snapshot" in text
    assert "kernel boot" in text and "power/clock-policy" in text


def test_parser_fails_before_runtime_when_required_arguments_are_missing() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "--origin-run", "only-one-field"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--measurement-set is required" in result.stderr


def test_dry_run_uses_qvg_mount_adapter_and_forwards_candidate(
    tmp_path: Path,
) -> None:
    argv = _capture_launcher(
        tmp_path,
        "--candidate",
        "v4",
        "--dry-run",
        "--json",
    )
    assert argv[0] == str(RUNTIME_ENTRY)
    assert "replay-candidates" in argv
    assert "--candidate" in argv and "v4" in argv
    assert "--dry-run" in argv
    assert any(value.startswith("task_runtime_adapter_sha256=") for value in argv)
    assert any(value.startswith("launcher_lib_sha256=") for value in argv)
    assert f"launcher_lib_sha256={hashlib.sha256((ROOT / 'scripts/lib.sh').read_bytes()).hexdigest()}" in argv
    assert not any(value.startswith("lock_helper_sha256=") for value in argv)
    assert any(value.startswith("runtime_identity_helper_sha256=") for value in argv)
    assert "judge_gpu_uuids=unverified-dry-run" in argv
    assert any(value.startswith("host_runtime_identity_sha256=") for value in argv)
    assert any(value.startswith("release_sha256=") for value in argv)
    assert any(value.startswith("paper_protocol_sha256=") for value in argv)
    # Without JUDGE_GPUS a preview does not guess the Judge size.
    assert "--timeout" not in argv and "--judge-cpu-limit" not in argv


def test_dry_run_forwards_an_explicit_timeout(tmp_path: Path) -> None:
    argv = _capture_launcher(tmp_path, "--dry-run", "--timeout", "9000")
    assert argv[argv.index("--timeout") + 1] == "9000"


def test_full_replay_reuses_eval_archive_locks_preflight_and_gpu_list() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert source.index('[[ -f "${LAUNCHER_LIB}"') < source.index('source "${SCRIPT_DIR}/lib.sh"')
    assert '"judge_image_id=' not in source
    assert 'judge_gpu_uuids "${JUDGE_GPUS:-}"' in source
    # Both host lock names of every Judge GPU, held by this shell for the batch.
    assert 'lock_gpu "${uuid}" judge' in source
    assert source.index('lock_gpu "${uuid}" judge') < source.index('assert_gpu_idle "${uuid}" judge')
    assert 'assert_no_compute_process "${uuid}" judge' in source
    assert "assert_identical_gpus judge" in source
    assert "assert_built_images_current" in source
    assert "secure_flock_exec.py" not in source and "exec 9>" not in source
    assert 'judge_plan "${#judge_uuids[@]}"' in source
    assert 'scripts/verify_models.py' in source
    assert "unset SFORGE_JUDGE_EXTRA_ENV" in source
    assert "CUDA_VISIBLE_DEVICES=0" not in source
    assert 'export QVG_JUDGE_GPU_UUIDS="${judge_uuids[*]}"' in source
    assert 'export QVG_REPLAY_GPU_INDICES="${judge_indices[*]}"' in source
    assert 'export QVG_REPLAY_GPU_UUIDS="${judge_uuids[*]}"' in source
    assert "scripts/sforge_runtime_entry.py" in source
    assert "mktemp -d -t sforge-qvg-replay-preflight" in source
    assert "runtime_identity.py" in source
    assert "host_runtime_identity_sha256" in source
    assert "candidate-replay-preflight/${TASK_ID}" not in source


def test_runtime_revalidates_models_and_host_identity_around_each_judge_attempt() -> None:
    launcher = SCRIPT.read_text(encoding="utf-8")
    runtime = RUNTIME_ENTRY.read_text(encoding="utf-8")
    assert "QVG_REPLAY_MODEL_LOCK_PATH" in launcher
    assert "QVG_REPLAY_MODEL_LOCK_SHA256" in launcher
    assert "QVG_REPLAY_RUNTIME_IDENTITY_PATH" in launcher
    assert "QVG_REPLAY_RUNTIME_IDENTITY_SHA256" in launcher
    assert "QVG_REPLAY_RUNTIME_IDENTITY_HELPER_SHA256" in launcher
    assert "def _verify_replay_live_artifacts" in runtime
    assert "_verify_replay_live_artifacts()" in runtime
    assert "runtime_identity.py" in runtime and "--verify-baseline" in runtime
    assert 'exec "${command[@]}"' not in launcher


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
