"""scripts/infrathon/images.sh: published images are pulled by digest from a configurable repository."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
IMAGES_SH = REPOSITORY_ROOT / "scripts/infrathon/images.sh"
DEFAULT = json.loads((REPOSITORY_ROOT / "scripts/infrathon/registry.json").read_text())["repository"]
WORK = "sha256:" + "a" * 64
JUDGE = "sha256:" + "b" * 64


def run(script: str, tmp_path: Path, **env: str) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    docker = bin_dir / "docker"
    docker.write_text(f'#!/bin/sh\necho "$*" >> "{tmp_path}/docker.log"\n')
    docker.chmod(0o755)
    environment = {k: v for k, v in os.environ.items() if k != "INFRATHON_REGISTRY_REPOSITORY"}
    environment.update(PATH=f"{bin_dir}:{os.environ['PATH']}", INFRATHON_HARDWARE="h100-80gb-700w", **env)
    return subprocess.run(["bash", "-c", f'die() {{ echo "ERROR: $*" >&2; exit 1; }}; source "{IMAGES_SH}"; {script}'],
                          capture_output=True, text=True, env=environment)


def task_root(tmp_path: Path) -> Path:
    root = tmp_path / "task"
    (root / "images").mkdir(parents=True)
    (root / "images/published.json").write_text(json.dumps({
        "source_digest": "src",
        "hardware": {"h100-80gb-700w": {"work": WORK, "judge_public": JUDGE}},
    }))
    return root


def test_no_published_record_names_a_repository():
    records = sorted((REPOSITORY_ROOT / "tasks/latest").glob("*/images/published.json"))
    assert records
    for path in records:
        assert "repository" not in json.loads(path.read_text()), path


@pytest.mark.parametrize("override", [None, "ghcr.io/example/infrathon", "registry.local:5000/mirror/sforge"])
def test_pull_uses_the_configured_repository(tmp_path, override):
    root = task_root(tmp_path)
    env = {"INFRATHON_PULL": "1"}
    if override:
        env["INFRATHON_REGISTRY_REPOSITORY"] = override
    result = run(f'infrathon_pull_published_images "{root}" work:local judge:local src', tmp_path, **env)
    assert result.returncode == 0, result.stderr
    repository = override or DEFAULT
    assert (tmp_path / "docker.log").read_text().splitlines() == [
        f"pull {repository}@{WORK}", f"pull {repository}@{JUDGE}",
        f"tag {repository}@{WORK} work:local", f"tag {repository}@{JUDGE} judge:local",
    ]


@pytest.mark.parametrize("bad", ["sforge", "Example/Repo", "example/repo:tag", "example/repo@sha256:x", "a//b"])
def test_malformed_repository_is_refused(tmp_path, bad):
    result = run("infrathon_registry_repository", tmp_path, INFRATHON_REGISTRY_REPOSITORY=bad)
    assert result.returncode != 0
    assert "must look like" in result.stderr
