from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from conftest import TASK_ROOT, final_manifest_path, load_module

layout = load_module("kv_eviction_v2_validate_layout", "scripts/validate_layout.py")


def _copy_task(tmp_path: Path) -> Path:
    root = tmp_path / "task"
    shutil.copytree(TASK_ROOT, root, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
    return root


def test_layout_passes_on_the_committed_tree() -> None:
    assert layout.validate_private_inputs(TASK_ROOT) == []
    assert layout.validate_multi_gpu_judge(TASK_ROOT) == []


def test_manifest_planted_in_a_work_context_is_detected(tmp_path: Path) -> None:
    root = _copy_task(tmp_path)
    shutil.copyfile(final_manifest_path(), root / "images/work/benchmark/leaked.json")
    errors = layout.validate_private_inputs(root)
    assert any("byte-identical to the Judge-only" in e for e in errors)


def test_holdout_task_name_in_work_files_is_detected(tmp_path: Path) -> None:
    final_manifest_path()  # the holdout task names come from it
    root = _copy_task(tmp_path)
    brief = root / "images/work/AGENT_TASK.md"
    brief.write_text(brief.read_text() + "\nHint: the private suite contains musique.\n")
    assert any("names holdout task 'musique'" in e for e in layout.validate_private_inputs(root))


def test_work_dockerfile_must_not_reference_private_inputs(tmp_path: Path) -> None:
    root = _copy_task(tmp_path)
    dockerfile = root / "images/work/Dockerfile"
    dockerfile.write_text(dockerfile.read_text() + "\n# COPY judge/private/final_manifest.json /tmp/\n")
    assert any("must not reference" in e for e in layout.validate_private_inputs(root))


def test_work_brief_never_names_holdout_sources() -> None:
    manifest = json.loads(final_manifest_path().read_text())
    names = {e["task"].split(".", 1)[1] for e in manifest["items"] if not e["task"].startswith("ruler.")} - layout.COMMON_WORDS
    text = (TASK_ROOT / "images/work/AGENT_TASK.md").read_text().lower()
    for name in names:
        assert name.lower() not in text, name
    for corpus in ("loogle", "zero-scrolls", "zero_scrolls", "zeroscrolls"):
        assert corpus not in text


def test_build_stages_items_only_into_the_judge_context() -> None:
    build = (TASK_ROOT / "scripts/build_images.sh").read_text()
    work_build = build[build.index("Building ${WORK_IMAGE}"):build.index("Building ${JUDGE_IMAGE}")]
    assert "final_items" not in work_build and "FINAL_ITEMS_CONTEXT" not in work_build
    assert '--build-context "final_items=${FINAL_ITEMS_CONTEXT}"' in build[build.index("Building ${JUDGE_IMAGE}"):]
