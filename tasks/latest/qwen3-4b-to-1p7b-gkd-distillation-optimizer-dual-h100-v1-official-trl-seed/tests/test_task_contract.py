from __future__ import annotations

import json

import pytest

from conftest import TASK_ROOT


def test_task_metadata_is_narrow_and_offline():
    task = json.loads(
        (
            TASK_ROOT
            / "tasks/qwen3_4b_to_1p7b_gkd_distillation_optimizer_dual_h100_v1.json"
        ).read_text()
    )
    assert task["task_id"] == "qwen3_4b_to_1p7b_gkd_distillation_optimizer_dual_h100_v1"
    assert "GKD Distillation Optimization" in task["name"]
    assert "build_objective" in task["work"]["agent_query"]
    assert "build_teacher_patch" in task["work"]["agent_query"]
    assert task["internet"] is True
    assert task["cwd"] == "/app/workdir/gkd_lab"
    assert set(task["submit_paths"]) == {"solutions", "FINAL_VERSION", "notes.md"}
    assert task["judge"]["selection"] == "valid_then_score"
    assert task["judge"]["parser"] == "structured_json"


def test_model_identity_is_exact():
    manifest = json.loads((TASK_ROOT / "docs/model_manifest.json").read_text())
    assert manifest["student"]["repo_id"] == "Qwen/Qwen3-1.7B"
    assert manifest["student"]["revision"] == "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    assert manifest["student"]["total_size"] == 4_063_479_808
    assert manifest["teacher"]["repo_id"] == "Qwen/Qwen3-4B"
    assert manifest["teacher"]["revision"] == "1cfa9a7208912126459214e8b04321603b3df60c"
    assert manifest["teacher"]["total_size"] == 8_044_936_192
    runtime = (
        TASK_ROOT / "images/work/workdir/gkd_lab/runtime/train_step.py"
    ).read_text()
    assert "STUDENT_MODEL_UNIQUE_PARAMETERS = 1_720_574_976" in runtime
    assert "TEACHER_MODEL_UNIQUE_PARAMETERS = 4_022_468_096" in runtime
    assert runtime.count('"tie_word_embeddings": True,') == 2


def test_dataset_identity_and_token_accounting_are_frozen():
    # The frozen tensors are deployment artifacts, not repository content:
    # `.gitignore` excludes `artifacts/` and the deployed copy lives outside the
    # checkout. Assert on them when a maintainer has them locally, and skip on a
    # plain clone rather than reporting a missing deployment as a contract break.
    freeze_manifest = TASK_ROOT / "artifacts/tldr_gkd_v1/freeze_manifest.json"
    if not freeze_manifest.is_file():
        pytest.skip("frozen dataset artifacts are not present in this checkout")
    manifest = json.loads(freeze_manifest.read_text())
    assert manifest["dataset_repo_id"] == "trl-lib/tldr-preference"
    assert manifest["dataset_revision"] == "7ae482b332979f8ce40d2a6b5f78f9eddf137efc"
    assert manifest["status"] == "frozen"
    assert manifest["partitions"]["public"]["tensors"]["sha256"] == (
        "9a92460dad2938181bfc51f9cebefc81b6f9b59ac21c2144f4cba66739a953b6"
    )
    runtime = (
        TASK_ROOT
        / "images/work/workdir/gkd_lab/runtime/distributed_probe.py"
    ).read_text()
    assert '"primary": "labels[:, 1:] != -100"' in runtime


def test_seed_does_not_export_an_imperative_model_hook():
    seed = (
        TASK_ROOT
        / "images/work/workdir/gkd_lab/solutions/objective_v001/candidate.py"
    ).read_text()
    assert "def optimize_model(" not in seed
