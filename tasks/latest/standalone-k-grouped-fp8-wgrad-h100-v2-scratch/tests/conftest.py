from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType


TASK_ROOT = Path(__file__).resolve().parents[1]
# The portable-contract and single-GPU migration tests import this
# repository's harness; keep it ahead of any installed ``sforge`` copy.
REPOSITORY_ROOT = next(
    parent for parent in TASK_ROOT.parents if (parent / "sforge").is_dir()
)
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def load_module(name: str, relative: str) -> ModuleType:
    path = TASK_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve their annotations through sys.modules.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Judge-only holdouts (assets/manifest.json, source "infrathon-dataset"): a development
# checkout carries them in the repository; the public tree does not, and
# assets/fetch.sh places them under <INFRATHON_ASSETS_ROOT>/<task directory>/<asset path>.
JUDGE_ONLY_ASSETS = {
    "judge/private/final_cases.json": "judge/final_cases.json",
    "judge/private/edge_cases.json": "judge/edge_cases.json",
}


def judge_only_file(relative: str) -> Path:
    """The repository copy of a Judge-only holdout, else the fetched asset, else skip."""
    path = TASK_ROOT / relative
    if path.is_file():
        return path
    assets_root = os.environ.get("INFRATHON_ASSETS_ROOT") or "~/.cache/infrathon/assets"
    fetched = Path(assets_root).expanduser() / TASK_ROOT.name / JUDGE_ONLY_ASSETS[relative]
    if fetched.is_file():
        return fetched
    import pytest

    pytest.skip("Judge-only data: run assets/fetch.sh", allow_module_level=True)
