"""SForge context, profile, and candidate-version tests."""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path
import sys
import time

import pytest

from conftest import JUDGE, load_file

verify = load_file("verify_eagle3_test", JUDGE / "verify_eagle3.py")


def write_context(path, value) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    path.chmod(0o600)


def test_host_round_honors_candidate_version(tmp_path) -> None:
    path = tmp_path / "context.json"
    write_context(path, {"round_id": "candidate-7", "candidate_version": 3})
    assert verify.load_evaluation_request(path, expected_uid=os.getuid()) == (
        "final",
        "candidate-7",
        3,
    )


def test_interactive_round_rejects_override(tmp_path) -> None:
    path = tmp_path / "context.json"
    write_context(path, {"round_id": "agent-1", "candidate_version": 2})
    with pytest.raises(RuntimeError, match="host-side only"):
        verify.load_evaluation_request(path, expected_uid=os.getuid())


@pytest.mark.parametrize("value", [True, 0, 1000, "2"])
def test_invalid_candidate_versions_fail_closed(tmp_path, value) -> None:
    path = tmp_path / "context.json"
    write_context(path, {"round_id": "manual-1", "candidate_version": value})
    with pytest.raises(RuntimeError, match="candidate_version"):
        verify.load_evaluation_request(path, expected_uid=os.getuid())


def test_private_reward_records_are_mode_0600(tmp_path) -> None:
    """Root-only result records must retain their mode when SForge archives them."""
    detail = {
        "gates": {name: True for name in verify.GATES},
        "cell_tokens_per_s": {"only": 1.0},
    }
    verify.write_reward(tmp_path, detail, 1.0)
    for name in ("reward.json", "reward.txt", verify.DETAIL_NAME):
        mode = stat.S_IMODE((tmp_path / name).stat().st_mode)
        assert mode == 0o600


def test_controller_timeout_kills_its_nested_worker(tmp_path) -> None:
    """The verifier's inner timeout must stop the separately sessioned worker."""
    child_pid_path = tmp_path / "child.pid"
    controller = tmp_path / "fake_controller.py"
    controller.write_text(
        "import pathlib\n"
        "import subprocess\n"
        "import sys\n"
        "import time\n"
        "child = subprocess.Popen(\n"
        "    [sys.executable, '-c', 'import time; time.sleep(60)'],\n"
        "    start_new_session=True,\n"
        ")\n"
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid))\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    with pytest.raises(subprocess.TimeoutExpired):
        verify.run_controller_command(
            [sys.executable, str(controller)],
            timeout=0.25,
        )
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        status_path = Path(f"/proc/{child_pid}/stat")
        if not status_path.exists():
            break
        fields = status_path.read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            break
        time.sleep(0.02)
    else:
        pytest.fail("nested controller worker survived verifier timeout")
