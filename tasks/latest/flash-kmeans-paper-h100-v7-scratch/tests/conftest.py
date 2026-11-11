"""Make this repository's ``sforge`` importable from the task test directory.

The portable-contract and single-GPU migration tests import the harness that
lives at the repository root.  When pytest is started from the task directory
the interpreter may resolve ``sforge`` to an unrelated installed copy, so the
repository root is placed first on ``sys.path`` here.
"""

from __future__ import annotations

import sys
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = next(
    parent for parent in TASK_ROOT.parents if (parent / "sforge").is_dir()
)
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
