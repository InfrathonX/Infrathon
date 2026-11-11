"""Every shipped task carries a complete integrity policy.

The LLM integrity review is mandatory, so a task without
``llm_judge/POLICY.md`` cannot run.  This catches a newly added task at test
time instead of at the start of its first run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sforge.harness.llm_judge.policy import load_task_policy, policy_problems

REPO_ROOT = Path(__file__).resolve().parents[3]
TASK_DIRS = sorted(
    path.parent.parent
    for path in (REPO_ROOT / "tasks" / "latest").glob("*/tasks/*.json")
)


def test_task_directories_were_found() -> None:
    released = json.loads((REPO_ROOT / "scripts/infrathon/tasks.json").read_text())["tasks"]
    names = {path.name for path in TASK_DIRS}
    for task, entry in released.items():
        if any(status != "unsupported" for status in entry.get("hardware", {}).values()):
            assert task in names, task
    assert len(TASK_DIRS) >= 20


@pytest.mark.parametrize("task_dir", TASK_DIRS, ids=lambda path: path.name)
def test_task_ships_a_complete_integrity_policy(task_dir: Path) -> None:
    for task_json in sorted((task_dir / "tasks").glob("*.json")):
        task_id = json.loads(task_json.read_text(encoding="utf-8"))["task_id"]
        policy = load_task_policy(task_id, tasks_dir=task_dir / "tasks")
        assert policy_problems(policy) == []
