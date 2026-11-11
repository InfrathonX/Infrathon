"""Shared paths for the CPU-only task tests.

The existing tests define their own ROOT; this file only makes the repository
harness importable ahead of any installed ``sforge`` copy for the portable
contract and sharded-Judge launcher tests.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT
REPOSITORY_ROOT = next(parent for parent in ROOT.parents if (parent / "sforge").is_dir())
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

# The release config is Judge-only data: the in-repository copy when present,
# else the asset-root copy written by assets/fetch.sh.
RELEASE_RELATIVE = "configs/h100_scratch_v4_release.json"
RELEASE_ASSET = "judge/h100_scratch_v4_release.json"


def _asset_task_root() -> Path:
    assets = os.environ.get("INFRATHON_ASSETS_ROOT") or str(
        Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "infrathon" / "assets")
    return Path(assets).expanduser() / ROOT.name


def _release_location() -> Path:
    local = ROOT / RELEASE_RELATIVE
    if local.is_file():
        return local
    return _asset_task_root() / RELEASE_ASSET


RELEASE_PATH = _release_location()


def release_file() -> Path:
    """The release config, or skip the calling test when it is not available."""
    if not RELEASE_PATH.is_file():
        pytest.skip("Judge-only data: run assets/fetch.sh")
    return RELEASE_PATH


# Per device class: the quality reference and the oracle's score band
# (hw/<type>/judge/, Judge-only like the release).
HARDWARE_TYPES = ("h100-94gb-400w", "h100-80gb-700w")
QUALITY_ASSET = "hw/{hardware}/judge/h100_scratch_v4_quality.json"


def quality_file(hardware: str) -> Path:
    """One device class's quality file, or skip the calling test when it is not available."""
    path = _asset_task_root() / QUALITY_ASSET.format(hardware=hardware)
    if not path.is_file():
        pytest.skip(f"Judge-only data for {hardware}: run assets/fetch.sh --hardware {hardware}")
    return path


def task_file(relative: str) -> Path:
    """A task file by its manifest path, resolving the Judge-only release config."""
    return release_file() if relative == RELEASE_RELATIVE else ROOT / relative
