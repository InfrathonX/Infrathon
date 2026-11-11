"""Locate this task's Judge-only dataset assets for the tests.

The public repository does not carry the in-repository copies of the dataset assets
(``assets/manifest.json`` entries with ``staged_from``); ``assets/fetch.sh`` places them
under ``$INFRATHON_ASSETS_ROOT/<task directory>/<asset path>``.  A test that needs one resolves
it in the repository first, then in the asset root, and otherwise skips.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import ModuleType

TASK_ROOT = Path(__file__).resolve().parents[1]
SKIP_REASON = "Judge-only data: run assets/fetch.sh"
HIDDEN_CASES = "images/judge/hidden_cases.py"


# Repository path of each dataset asset -> its path in the asset store.  The public
# assets/manifest.json does not carry staged_from, so the mapping is spelled out here.
STAGED_ASSETS = {HIDDEN_CASES: "judge/hidden_cases.py"}


def assets_root() -> Path:
    configured = os.environ.get("INFRATHON_ASSETS_ROOT")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache/infrathon/assets"


def resolve(relative: str) -> Path | None:
    """Return the file for a task-relative path, or None when it is not available."""
    local = TASK_ROOT / relative
    if local.is_file():
        return local
    asset = STAGED_ASSETS.get(relative)
    if asset is not None:
        fetched = assets_root() / TASK_ROOT.name / asset
        if fetched.is_file():
            return fetched
    return None


def is_dataset_copy(relative: str) -> bool:
    return relative in STAGED_ASSETS


def require(test: unittest.TestCase, relative: str) -> Path:
    """The file for ``relative``; skips only for a dataset asset that is not available."""
    path = resolve(relative)
    if path is None:
        if not is_dataset_copy(relative):
            return TASK_ROOT / relative
        test.skipTest(SKIP_REASON)
    return path


def load_hidden_cases(*, required: bool = False) -> ModuleType | None:
    """Import ``hidden_cases`` (the caller has put images/judge/benchmark on sys.path).

    With ``required`` the calling test module is skipped when the asset is missing.
    """
    if "hidden_cases" in sys.modules:
        return sys.modules["hidden_cases"]
    path = resolve(HIDDEN_CASES)
    if path is None:
        if required:
            import pytest

            pytest.skip(SKIP_REASON, allow_module_level=True)
        return None
    spec = importlib.util.spec_from_file_location("hidden_cases", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["hidden_cases"] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop("hidden_cases", None)
        raise
    return module


requires_hidden_cases = unittest.skipIf(resolve(HIDDEN_CASES) is None, SKIP_REASON)
