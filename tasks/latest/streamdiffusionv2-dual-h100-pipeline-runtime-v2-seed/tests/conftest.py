from __future__ import annotations
import importlib.util
import os
import sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]


def judge_asset(build_relative, asset_relative):
    """A Judge-only input: the in-repository build/ copy first, else the dataset copy at
    $INFRATHON_ASSETS_ROOT/<task>/<asset path> (assets/manifest.json), else None."""
    local=ROOT/build_relative
    if local.exists(): return local
    root=Path(os.environ.get("INFRATHON_ASSETS_ROOT") or Path.home()/".cache/infrathon/assets")
    fetched=root/ROOT.name/asset_relative
    return fetched if fetched.exists() else None


@pytest.fixture
def final_cases_file():
    """The formal suite manifest, or None when neither build/ nor the dataset holds it."""
    return judge_asset("build/private/final_cases.json", "judge/final_cases.json")


@pytest.fixture
def final_cases_path(final_cases_file):
    if final_cases_file is None: pytest.skip("Judge-only data: run assets/fetch.sh")
    return final_cases_file

@pytest.fixture
def load_module():
    def load(relative):
        path=ROOT/relative; name="sdv2_test_"+path.stem+str(abs(hash(path)))
        spec=importlib.util.spec_from_file_location(name,path); assert spec and spec.loader
        module=importlib.util.module_from_spec(spec); sys.modules[name]=module; spec.loader.exec_module(module); return module
    return load
