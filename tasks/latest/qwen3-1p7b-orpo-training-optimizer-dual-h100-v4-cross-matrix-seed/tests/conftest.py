from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


TASK_ROOT = Path(__file__).resolve().parents[1]
LAB_ROOT = TASK_ROOT / "images/work/workdir/orpo_lab"
sys.path.insert(0, str(LAB_ROOT))
HIDDEN_CASES = "images/judge/private_data/hidden_cases.json"
HIDDEN_CASES_ASSET = "judge/private_data/hidden_cases.json"


def judge_only_file(repository_path: str = HIDDEN_CASES, asset_path: str = HIDDEN_CASES_ASSET) -> Path:
    """A Judge-only data file: the repository copy, else the fetched asset, else skip.

    The public tree does not carry the Judge-only holdouts; ``assets/fetch.sh``
    installs them under ``$INFRATHON_ASSETS_ROOT/<task directory>/<asset path>``.
    """

    local = TASK_ROOT / repository_path
    if local.is_file():
        return local
    root = Path(os.environ.get("INFRATHON_ASSETS_ROOT") or "~/.cache/infrathon/assets").expanduser()
    fetched = root / TASK_ROOT.name / asset_path
    if fetched.is_file():
        return fetched
    pytest.skip("Judge-only data: run assets/fetch.sh")


@pytest.fixture(autouse=True)
def runtime_on_path(monkeypatch):
    monkeypatch.syspath_prepend(str(LAB_ROOT))
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    yield


sys.dont_write_bytecode = True
