from __future__ import annotations
import os
import sys
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT
# The portable-contract and single-GPU migration tests import this
# repository's harness; keep it ahead of any installed ``sforge`` copy.
REPOSITORY_ROOT = next(parent for parent in TASK_ROOT.parents if (parent / "sforge").is_dir())
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(ROOT / "runtime"))
sys.path.insert(0, str(ROOT / "images" / "judge"))

def load_module(name: str, relative: str) -> ModuleType:
    path = TASK_ROOT / relative; spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec); previous = sys.modules.get(name); sys.modules[name] = module; parent = str(path.parent); sys.path.insert(0, parent)
    try: spec.loader.exec_module(module)
    except Exception:
        if previous is None: sys.modules.pop(name, None)
        else: sys.modules[name] = previous
        raise
    finally: sys.path.remove(parent)
    return module


def judge_only(repository_path: str, asset_path: str) -> Path:
    """A Judge-only dataset asset: the in-repository copy when present, else the fetched
    copy at <INFRATHON_ASSETS_ROOT>/<task directory>/<asset path> (assets/manifest.json), else skip."""
    path = TASK_ROOT / repository_path
    if path.is_file():
        return path
    assets = Path(os.environ.get("INFRATHON_ASSETS_ROOT") or Path.home() / ".cache/infrathon/assets")
    path = assets / TASK_ROOT.name / asset_path
    if path.is_file():
        return path
    pytest.skip("Judge-only data: run assets/fetch.sh")


def final_manifest_path() -> Path:
    return judge_only("judge/private/final_manifest.json", "judge/final_manifest.json")
