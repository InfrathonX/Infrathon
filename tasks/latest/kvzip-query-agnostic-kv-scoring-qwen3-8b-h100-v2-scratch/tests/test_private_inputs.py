from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from conftest import TASK_ROOT, judge_input, load_module

layout = load_module("kv_scoring_v2_validate_layout", "scripts/validate_layout.py")


@pytest.fixture(scope="module")
def NEEDLES() -> list:
    return json.loads(judge_input("judge/private/needle_private.json").read_text())


def _copy_task(tmp_path: Path) -> Path:
    root = tmp_path / "task"
    shutil.copytree(TASK_ROOT, root, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "assets.pt"))
    return root


def test_layout_passes_on_the_committed_tree() -> None:
    assert layout.validate_private_inputs(TASK_ROOT) == []
    assert layout.validate_multi_gpu_judge(TASK_ROOT) == []


def test_three_distinct_private_needles(NEEDLES: list) -> None:
    assert len(NEEDLES) == 3
    words = [set(x["needle"].lower().split()) - {"the", "is", "in", "of", "a", "its"} for x in NEEDLES]
    assert not (words[0] & words[1] or words[0] & words[2] or words[1] & words[2])
    assert all("san francisco" not in x["needle"].lower() for x in NEEDLES)


def test_private_needle_in_a_work_file_is_detected(tmp_path: Path, NEEDLES: list) -> None:
    root = _copy_task(tmp_path)
    readme = root / "images/work/workdir/kv_scoring_lab/README_AGENT.md"
    readme.write_text(readme.read_text() + "\n" + NEEDLES[0]["needle"].strip() + "\n")
    assert any("contains a private needle" in e for e in layout.validate_private_inputs(root))


def test_manifest_planted_in_a_work_context_is_detected(tmp_path: Path) -> None:
    manifest = judge_input("judge/private/final_manifest.json")
    root = _copy_task(tmp_path)
    shutil.copyfile(manifest, root / "corpus/public/extra.json")
    assert any("byte-identical to the Judge-only" in e for e in layout.validate_private_inputs(root))
