from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType


TASK_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str) -> ModuleType:
    path = TASK_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register path-loaded modules while they execute.  The runtime contains
    # dataclasses (which consult sys.modules), and the Judge files use local
    # absolute imports just as they do inside their images.
    previous = sys.modules.get(name)
    sys.modules[name] = module
    parent = str(path.parent)
    sys.path.insert(0, parent)
    try:
        spec.loader.exec_module(module)
    except Exception:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise
    finally:
        sys.path.remove(parent)
    return module


# Judge-only holdouts (assets/manifest.json, source "infrathon-dataset"): a development
# checkout carries them in the repository; the public tree does not, and
# assets/fetch.sh places them under <INFRATHON_ASSETS_ROOT>/<task directory>/<asset path>.
JUDGE_ONLY_ASSETS = {
    "judge/private/final_cases.json": "judge/final_cases.json",
    **{
        f"judge/private/routing/{name}.{ext}": f"judge/routing/{name}.{ext}"
        for name in ("gpt-oss-20b", "phi-3.5-moe", "mixtral-8x7b", "granite-3.1-3b-a800m")
        for ext in ("npy", "json")
    },
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
