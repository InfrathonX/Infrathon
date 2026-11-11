"""Shared paths and import helper for CPU-only task tests."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT
# The portable-contract and single-GPU migration tests import this
# repository's harness; keep it ahead of any installed ``sforge`` copy.
REPOSITORY_ROOT = next(parent for parent in ROOT.parents if (parent / "sforge").is_dir())
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
SPEC = ROOT / "images/work/workdir/spec_lab"
VERIFIER = ROOT / "images/judge/verifier"
JUDGE = ROOT / "images/judge"
SCRIPTS = ROOT / "scripts"

for path in (SPEC, VERIFIER, JUDGE, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def load_file(name: str, path: Path) -> ModuleType:
    """Import a module from an explicit task-local path."""
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


# Judge-only holdout data (images/judge/private_data/) is a dataset asset: the
# development tree carries a copy, a public checkout fetches it with
# assets/fetch.sh into <INFRATHON_ASSETS_ROOT>/<task directory>/judge/private_data/
# (asset paths in assets/manifest.json).
PRIVATE_DATA = "images/judge/private_data"
PRIVATE_DATA_ASSET = "judge/private_data"
FETCH_HINT = "Judge-only data: run assets/fetch.sh"


def assets_root() -> Path:
    """The Infrathon asset root (assets/fetch.sh destination)."""
    configured = os.environ.get("INFRATHON_ASSETS_ROOT")
    return Path(configured) if configured else Path.home() / ".cache/infrathon/assets"


def dataset_path(relative: str) -> Path | None:
    """A task file that may be a dataset asset: the repository copy, else the
    fetched asset, else None."""
    relative = relative.rstrip("/")
    local = ROOT / relative
    if local.exists():
        return local
    if relative == PRIVATE_DATA or relative.startswith(PRIVATE_DATA + "/"):
        fetched = assets_root() / ROOT.name / (PRIVATE_DATA_ASSET + relative[len(PRIVATE_DATA):])
        if fetched.exists():
            return fetched
    return None


def require_dataset(relative: str) -> Path:
    """dataset_path(), or skip the calling test when the asset is not fetched."""
    path = dataset_path(relative)
    if path is None:
        pytest.skip(FETCH_HINT, allow_module_level=True)
    return path


def private_data() -> Path:
    """The Judge-only private_data directory, or skip."""
    return require_dataset(PRIVATE_DATA)


def maintainer_doc(relative: str) -> str:
    """Text of a maintainer document that the public tree does not ship, or skip."""
    path = ROOT / relative
    if not path.is_file():
        pytest.skip(f"maintainer document not shipped: {relative}")
    return path.read_text(encoding="utf-8")
