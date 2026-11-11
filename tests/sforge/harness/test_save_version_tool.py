from __future__ import annotations

import json
import os
import subprocess
import tarfile
from pathlib import Path

import pytest

from sforge.harness.evolve_scripts import generate_save_version_script


def _write_executable(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)
    return path


def _fake_curl(bin_dir: Path) -> Path:
    return _write_executable(
        bin_dir / "curl",
        r"""#!/bin/bash
set -eu
output=""
archive=""
label=""
token=""
content_type=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        -o)
            output="$2"
            shift 2
            ;;
        -H)
            case "$2" in
                "Content-Type: "*) content_type="${2#Content-Type: }" ;;
                "X-SForge-Token: "*) token="${2#X-SForge-Token: }" ;;
                "X-SForge-Version-Label: "*) label="${2#X-SForge-Version-Label: }" ;;
            esac
            shift 2
            ;;
        --data-binary)
            archive="${2#@}"
            shift 2
            ;;
        *) shift ;;
    esac
done
cp -- "$archive" "$CAPTURE_ARCHIVE"
printf '%s' "$label" > "$CAPTURE_LABEL"
printf '%s' "$token" > "$CAPTURE_TOKEN"
printf '%s' "$content_type" > "$CAPTURE_CONTENT_TYPE"
printf '%s\n' '{"version_id":"20260818T120000.000000Z__checkpoint-1__0123456789ab","round_id":"checkpoint-1","archive_sha256":"0123456789abcdef","archive_size":123,"candidate_id":"candidate-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","content_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}' > "$output"
printf '201'
""",
    )


@pytest.mark.parametrize("json_mode", [False, True])
def test_generated_tool_packages_submit_contract_and_reports_saved_version(
    tmp_path: Path, json_mode: bool,
) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "submission").mkdir(parents=True)
    (workspace / "submission/gateway.py").write_text("print('ok')\n", encoding="utf-8")
    (workspace / "submission/auth.json").write_text("secret\n", encoding="utf-8")
    (workspace / "outside.txt").write_text("outside\n", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake_curl(bin_dir)
    tool = _write_executable(tmp_path / "sforge-save-version", generate_save_version_script())
    captured_archive = tmp_path / "captured.tar.gz"
    captured_label = tmp_path / "captured-label"
    captured_token = tmp_path / "captured-token"
    captured_content_type = tmp_path / "captured-content-type"
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "TMPDIR": str(temp_dir),
            "CAPTURE_ARCHIVE": str(captured_archive),
            "CAPTURE_LABEL": str(captured_label),
            "CAPTURE_TOKEN": str(captured_token),
            "CAPTURE_CONTENT_TYPE": str(captured_content_type),
            "SFORGE_JUDGE_URL": "http://judge.internal:8080",
            "SFORGE_TOKEN": "token-value",
            "SFORGE_PATCH_DIR": str(workspace),
            "SFORGE_SUBMIT_PATHS": "submission",
            "SFORGE_SUBMIT_EXCLUDE_FLAGS": "--exclude=auth.json",
        }
    )

    completed = subprocess.run(
        [str(tool), "--label", "known-good", *(["--json"] if json_mode else [])],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    if json_mode:
        receipt = json.loads(completed.stdout)
        assert receipt["version_id"] == "20260818T120000.000000Z__checkpoint-1__0123456789ab"
        assert receipt["archive_sha256"] == "0123456789abcdef"
    else:
        assert completed.stdout.strip() == (
            "Saved candidate for replay: known-good (content aaaaaaaaaaaa)"
        )
        for internal_term in ("checkpoint-1", "Event:", "Archive SHA:", "Candidate:"):
            assert internal_term not in completed.stdout
    assert captured_label.read_text(encoding="utf-8") == "known-good"
    assert captured_token.read_text(encoding="utf-8") == "token-value"
    assert captured_content_type.read_text(encoding="utf-8") == "application/gzip"
    with tarfile.open(captured_archive, mode="r:gz") as archive:
        assert [member.name for member in archive.getmembers()] == [
            "submission",
            "submission/gateway.py",
        ]
    assert list(temp_dir.iterdir()) == []


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--label", ""],
        ["--label", "bad\nlabel"],
        ["--label", "x" * 81],
        ["--unknown"],
        ["--label", "one", "extra"],
    ],
)
def test_generated_tool_rejects_invalid_arguments_before_network(
    tmp_path: Path, arguments: list[str]
) -> None:
    tool = _write_executable(tmp_path / "sforge-save-version", generate_save_version_script())
    env = os.environ.copy()
    env.update(
        {
            "PATH": "/usr/bin:/bin",
            "SFORGE_JUDGE_URL": "http://judge.invalid",
            "SFORGE_TOKEN": "token-value",
        }
    )

    completed = subprocess.run(
        [str(tool), *arguments],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "Usage:" in completed.stderr
    assert "Save the current candidate version" in completed.stderr
    assert "replayable checkpoint" not in completed.stderr


def test_generated_tool_reports_http_error_and_removes_temporary_files(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(
        bin_dir / "curl",
        r"""#!/bin/bash
set -eu
output=""
while [ "$#" -gt 0 ]; do
    if [ "$1" = -o ]; then output="$2"; shift 2; else shift; fi
done
printf '%s\n' '{"detail":"version store unavailable"}' > "$output"
printf '503'
""",
    )
    workspace = tmp_path / "workspace"
    (workspace / "submission").mkdir(parents=True)
    (workspace / "submission/main.py").write_text("pass\n", encoding="utf-8")
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    tool = _write_executable(tmp_path / "sforge-save-version", generate_save_version_script())
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "TMPDIR": str(temp_dir),
            "SFORGE_JUDGE_URL": "http://judge.internal:8080",
            "SFORGE_TOKEN": "token-value",
            "SFORGE_PATCH_DIR": str(workspace),
            "SFORGE_SUBMIT_PATHS": "submission",
        }
    )

    completed = subprocess.run(
        [str(tool), "--label", "known-good"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "version store unavailable" in completed.stderr
    assert list(temp_dir.iterdir()) == []
