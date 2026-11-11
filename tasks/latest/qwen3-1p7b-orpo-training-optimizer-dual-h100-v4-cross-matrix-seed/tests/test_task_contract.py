from __future__ import annotations

import json

import pytest

from conftest import TASK_ROOT


def test_task_metadata_is_narrow_and_offline():
    task = json.loads(
        (TASK_ROOT / "tasks/qwen3_1p7b_orpo_training_optimizer_dual_h100_v4.json").read_text()
    )
    assert task["task_id"] == "qwen3_1p7b_orpo_training_optimizer_dual_h100_v4"
    assert "Training Optimization" in task["name"]
    assert "build_model_patch" in task["work"]["agent_query"]
    assert task["internet"] is True
    assert task["cwd"] == "/app/workdir/orpo_lab"
    assert set(task["submit_paths"]) == {"solutions", "FINAL_VERSION", "notes.md"}
    assert task["judge"]["selection"] == "valid_then_score"
    assert task["judge"]["parser"] == "structured_json"


def test_model_identity_is_exact():
    manifest = json.loads((TASK_ROOT / "docs/model_manifest.json").read_text())
    assert manifest["repo_id"] == "Qwen/Qwen3-1.7B"
    assert manifest["revision"] == "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    assert manifest["parameter_count_unique"] == 1_720_574_976
    assert manifest["config"]["tie_word_embeddings"] is True


def test_dataset_identity_and_token_accounting_are_frozen():
    path = TASK_ROOT / "docs/dataset_manifest.json"
    if not path.is_file():
        pytest.skip("maintainer dataset manifest is not shipped")
    manifest = json.loads(path.read_text())
    assert manifest["source"]["revision"] == "7ae482b332979f8ce40d2a6b5f78f9eddf137efc"
    assert manifest["root_manifest"]["offline_reproduction_identical"] is True
    assert manifest["public"]["tensors_sha256"] == (
        "e1218b35260ea9e7afd048962b53c25257d7f298e2a787628d8f5b640cfc7e79"
    )
    runtime = (
        TASK_ROOT
        / "images/work/workdir/orpo_lab/runtime/distributed_probe.py"
    ).read_text()
    assert '"primary": "labels[:, 1:] != -100"' in runtime


def test_seed_does_not_export_an_imperative_model_hook():
    seed = (
        TASK_ROOT
        / "images/work/workdir/orpo_lab/solutions/objective_v001/candidate.py"
    ).read_text()
    assert "def optimize_model(" not in seed
