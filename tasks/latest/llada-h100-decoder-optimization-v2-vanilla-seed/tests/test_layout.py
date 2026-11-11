from __future__ import annotations

import subprocess
import sys


def test_static_layout_validator(task_root):
    subprocess.run(
        [sys.executable, str(task_root / "scripts/validate_layout.py"), str(task_root)],
        check=True,
    )
