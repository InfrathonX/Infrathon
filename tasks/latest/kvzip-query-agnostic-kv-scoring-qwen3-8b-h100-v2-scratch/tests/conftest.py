from __future__ import annotations
import sys
import importlib.util
from pathlib import Path
from types import ModuleType
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


def judge_input(relative: str) -> Path:
    """A Judge-only input by its in-repository path (judge/private/...): the
    in-repository copy, else the dataset asset fetched into the asset root;
    skip when neither is present (public checkout without assets)."""
    import pytest
    layout = load_module("kv_scoring_v2_validate_layout", "scripts/validate_layout.py")
    path = layout.judge_input(TASK_ROOT, relative)
    if path is None:
        pytest.skip("Judge-only data: run assets/fetch.sh")
    return path
