from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


TASK_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = TASK_ROOT / "scripts" / "validate_layout.py"


def _copy_task(tmp_path: Path) -> Path:
    destination = tmp_path / "relocated-task"
    shutil.copytree(
        TASK_ROOT,
        destination,
        ignore=shutil.ignore_patterns("logs", "artifacts", "__pycache__", "*.pyc"),
    )
    return destination


def _run(root: Path, output: Path | None = None) -> tuple[subprocess.CompletedProcess[str], dict]:
    command = [sys.executable, str(VALIDATOR), str(root)]
    if output is not None:
        command.extend(["--output", str(output)])
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return result, json.loads(result.stdout)


def test_current_layout_is_valid_and_machine_readable(tmp_path: Path) -> None:
    output = tmp_path / "layout-validation.json"
    result, report = _run(TASK_ROOT, output)

    assert result.returncode == 0, result.stdout + result.stderr
    assert report["valid"] is True
    assert report["error_count"] == 0
    assert report["expected_images"]["work"].startswith("kvbench_qwen36.work.")
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_missing_required_file_fails_closed(tmp_path: Path) -> None:
    root = _copy_task(tmp_path)
    (root / "oracle/reference_gateway/scheduler.py").unlink()

    result, report = _run(root)

    assert result.returncode == 1
    assert report["valid"] is False
    assert any("missing required file" in error for error in report["errors"])


def test_required_symlink_fails_closed(tmp_path: Path) -> None:
    root = _copy_task(tmp_path)
    launch = root / "images/work/submission/launch.sh"
    replacement = root / "images/work/submission/real-launch.sh"
    launch.rename(replacement)
    launch.symlink_to(replacement.name)

    result, report = _run(root)

    assert result.returncode == 1
    assert report["valid"] is False
    assert any("must not be a symlink" in error for error in report["errors"])


def test_image_tag_mismatch_fails_closed(tmp_path: Path) -> None:
    root = _copy_task(tmp_path)
    lib_path = root / "scripts/lib.sh"
    text = lib_path.read_text(encoding="utf-8")
    lib_path.write_text(
        text.replace(
            'JUDGE_IMAGE="${BENCHMARK_NAME}.judge.${TASK_ID}:local"',
            'JUDGE_IMAGE="wrong.judge.${TASK_ID}:local"',
        ),
        encoding="utf-8",
    )

    result, report = _run(root)

    assert result.returncode == 1
    assert report["valid"] is False
    assert any("JUDGE_IMAGE must be" in error for error in report["errors"])


def test_duplicate_json_key_fails_closed(tmp_path: Path) -> None:
    root = _copy_task(tmp_path)
    task_path = root / "tasks/concurrent_agent_serving_optimizer_qwen36_swe192_v24.json"
    text = task_path.read_text(encoding="utf-8")
    task_path.write_text(
        text.replace(
            '"image_tag": "local",',
            '"image_tag": "local",\n    "image_tag": "local",',
            1,
        ),
        encoding="utf-8",
    )

    result, report = _run(root)

    assert result.returncode == 1
    assert report["valid"] is False
    assert any("duplicate object key 'image_tag'" in error for error in report["errors"])


def test_build_runs_layout_validation_before_docker(tmp_path: Path) -> None:
    root = _copy_task(tmp_path)
    build = (root / "scripts/build_images.sh").read_text(encoding="utf-8")

    assert build.index('"${MIGRATION_ROOT}/scripts/validate_layout.py"') < build.index(
        "docker build"
    )
