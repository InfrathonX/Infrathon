from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


TASK_ROOT = Path(__file__).resolve().parents[1]
LAB_ROOT = TASK_ROOT / "images/work/workdir/gkd_lab"
sys.path.insert(0, str(LAB_ROOT))
# Present only in the maintainer tree; the public Infrathon export drops docs/
# and replaces README.md, so tests of their wording skip there.
MAINTAINER_DOCS = (TASK_ROOT / "docs/DESIGN.md").is_file()


def judge_asset(asset_path: str, repo_path: str) -> Path:
    """Locate a Judge-only data file: the in-repository copy, else
    ``$INFRATHON_ASSETS_ROOT/<task directory>/<asset path>`` (assets/manifest.json),
    else skip the calling test."""

    local = TASK_ROOT / repo_path
    if local.is_file():
        return local
    root = os.environ.get("INFRATHON_ASSETS_ROOT") or str(Path.home() / ".cache/infrathon/assets")
    fetched = Path(root).expanduser() / TASK_ROOT.name / asset_path
    if fetched.is_file():
        return fetched
    pytest.skip("Judge-only data: run assets/fetch.sh")


def hidden_cases_path() -> Path:
    return judge_asset("judge/private_data/hidden_cases.json", "images/judge/private_data/hidden_cases.json")


@pytest.fixture(autouse=True)
def runtime_on_path(monkeypatch):
    monkeypatch.syspath_prepend(str(LAB_ROOT))
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    yield


sys.dont_write_bytecode = True
