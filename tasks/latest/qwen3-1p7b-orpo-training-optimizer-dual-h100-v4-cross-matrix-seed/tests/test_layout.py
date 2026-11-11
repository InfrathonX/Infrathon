from __future__ import annotations

import subprocess
import sys

from conftest import TASK_ROOT


def test_static_layout_validator_passes():
    process = subprocess.run(
        [sys.executable, str(TASK_ROOT / "scripts/validate_layout.py"), str(TASK_ROOT)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
