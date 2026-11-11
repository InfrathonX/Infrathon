"""The published resource-profile table: what a checkout without deploy/pyromind sizes from."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sforge.harness import resource_profile as rp

REPO = Path(__file__).resolve().parents[3]


@pytest.mark.skipif(not (REPO / rp.DEPLOYMENTS).is_dir(), reason="no deploy/pyromind in this checkout")
def test_the_table_is_written_from_the_deployment_profiles():
    assert json.loads((REPO / rp.TABLE).read_text()) == rp.table(REPO), (
        "run python -m sforge.harness.resource_profile --write-table")


def test_every_released_task_with_a_table_entry_reads_it_without_deployments(tmp_path):
    (tmp_path / rp.TABLE).parent.mkdir(parents=True)
    (tmp_path / rp.TABLE).write_bytes((REPO / rp.TABLE).read_bytes())
    tasks = json.loads((REPO / rp.TABLE).read_text())["tasks"]
    assert tasks
    for task_id, entry in tasks.items():
        profile = rp.profile_for_task(task_id, tmp_path)
        assert profile.deployment == entry["deployment"]
        assert profile.source == str(rp.TABLE)
        for role in ("work", "judge", "review"):
            size = profile.role(role)
            assert (size.cpu, size.memory, size.gpu) == (
                entry[role]["cpu"], rp.docker_memory(entry[role]["memory"]), entry[role]["gpu"])
        assert profile.exempt == rp.LOCAL_EXEMPT.get(entry["deployment"])
    assert rp.profile_for_task("no_such_task", tmp_path) is None


def test_the_deployment_profiles_win_over_the_table(tmp_path):
    (tmp_path / rp.TABLE).parent.mkdir(parents=True)
    (tmp_path / rp.TABLE).write_text(json.dumps({"schema_version": 1, "tasks": {
        "t": {"deployment": "d", **{r: {"cpu": 1, "memory": "1Gi", "gpu": 0}
                                    for r in ("work", "judge", "review")}}}}))
    (tmp_path / rp.DEPLOYMENTS).mkdir(parents=True)
    assert rp.profile_for_task("t", tmp_path) is None


def test_neither_deployments_nor_table_is_an_error(tmp_path):
    with pytest.raises(rp.ResourceProfileError, match="SFORGE_RESOURCE_PROFILE=native"):
        rp.profile_for_task("t", tmp_path)
