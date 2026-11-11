from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest


TASK_ROOT = Path(__file__).resolve().parents[1]
ROOT = TASK_ROOT
# The portable-contract and single-GPU migration tests import this
# repository's harness; keep it ahead of any installed ``sforge`` copy.
REPOSITORY_ROOT = next(parent for parent in ROOT.parents if (parent / "sforge").is_dir())
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_file(name: str, path: Path) -> ModuleType:
    """Import a module from an explicit task-local path and register it."""
    module = load_module(name, path)
    sys.modules[name] = module
    return module


@pytest.fixture
def task_root() -> Path:
    return TASK_ROOT


def private_data_dir() -> Path:
    """Judge-only images/judge/private_data/: the in-repository copy, else the
    fetched dataset asset under the asset root; skip the calling test when
    neither is present (scripts/h100_policy_contract.py resolves the same)."""
    contract = load_module(
        "conftest_private_data_contract", TASK_ROOT / "scripts/h100_policy_contract.py"
    )
    path = contract.private_data_dir(TASK_ROOT)
    if not (path / "hidden_cases.json.gz").is_file():
        pytest.skip("Judge-only data: run assets/fetch.sh")
    return path


@pytest.fixture
def private_data() -> Path:
    return private_data_dir()
