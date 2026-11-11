from __future__ import annotations

from conftest import TASK_ROOT


TASK_JSON = TASK_ROOT / "tasks/qwen3_4b_to_1p7b_gkd_distillation_optimizer_dual_h100_v1.json"

PRIVATE_GEOMETRY_MARKERS = (
    "batch_per_rank 4",
    "batch_per_rank 8",
    "[2, 1023, H]",
    "[2, 2047, H]",
    "[4, 511, H]",
    "[8, 511, H]",
    "gkd_v1_row_seq512_batch4",
    "gkd_v1_row_seq512_batch8",
    "four fresh paired",
    "five warm-up steps",
    "ten timed steps",
    "RC, CR, CR, RC",
)


def test_work_image_does_not_copy_host_reference_or_private_data():
    dockerfile = (TASK_ROOT / "images/work/Dockerfile").read_text()
    assert "host_reference" not in dockerfile
    assert "private_data" not in dockerfile
    assert "liger" not in dockerfile.lower()
    assert "oracle" not in dockerfile.lower()


def test_submission_excludes_runtime_and_tests():
    task = TASK_JSON.read_text()
    assert '"runtime/"' in task
    assert '"tests/"' in task


def test_participant_facing_contract_exposes_only_declarative_model_patching():
    instruction = (TASK_ROOT / "images/work/AGENT_TASK.md").read_text()
    workspace = (TASK_ROOT / "images/work/workdir/gkd_lab/README_AGENT.md").read_text()
    task = TASK_JSON.read_text()
    visible = "\n".join((instruction, workspace, task))
    for required in (
        "build_objective",
        "build_model_patch",
        "build_teacher_patch",
        "Qwen3ModelPatch",
    ):
        assert required in visible
    normalized_instruction = " ".join(instruction.lower().split())
    assert "do not export `optimize_model`" in normalized_instruction
    assert "Liger" not in visible
    assert "1.02" not in visible
    for private_geometry in PRIVATE_GEOMETRY_MARKERS:
        assert private_geometry not in visible


def test_agent_instruction_is_concise_and_does_not_supply_a_solution_recipe():
    instruction = (TASK_ROOT / "images/work/AGENT_TASK.md").read_text()
    # 170 + the 2026-10-05 precision rule (compute in BF16 or FP32).
    assert len(instruction.splitlines()) <= 175
    for unnecessary_detail in (
        "avoiding full-vocabulary-logit materialization",
        "custom autograd",
        "ordered nine-element",
        "generalized_jsd =",
        "1,720,574,976",
        "4,022,468,096",
        "151,936",
        "2,048",
        "fresh, paired processes",
        "centered near score 100",
        "complete gradient/optimizer evidence",
    ):
        assert unnecessary_detail not in instruction


def test_complete_work_build_context_does_not_disclose_private_geometry():
    work_root = TASK_ROOT / "images/work"
    readable_suffixes = {".md", ".py", ".json", ".sh", ".toml"}
    chunks: list[str] = []
    for path in work_root.rglob("*"):
        if not path.is_file():
            continue
        if path.name == "Dockerfile" or path.suffix in readable_suffixes:
            chunks.append(path.read_text(encoding="utf-8"))
    work_text = "\n".join(chunks)
    for private_geometry in PRIVATE_GEOMETRY_MARKERS:
        assert private_geometry not in work_text
