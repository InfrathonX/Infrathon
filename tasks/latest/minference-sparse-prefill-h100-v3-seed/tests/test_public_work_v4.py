"""The Work image's public tests stay separate from the Judge."""
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import generate_public_prompts as public  # noqa: E402
import v3_contract as contract  # noqa: E402

WORK = ROOT / "images/work"
LAB = WORK / "workdir/minference_prefill_lab"


def test_public_grid_shares_no_task_budget_or_layer_with_the_judge():
    assert public.TASKS == contract.PUBLIC_TASKS and public.BUDGETS == contract.PUBLIC_BUDGETS
    assert not {"niah_multikey_2", "cwe"} & set(contract.TASKS)
    assert not set(contract.PUBLIC_BUDGETS) & set(contract.BUDGETS)
    assert not set(contract.PUBLIC_LAYERS) & {*contract.LAYERS, 2}
    assert public.SUITE["warmup_layer"] not in {*contract.LAYERS, 2}
    assert public.SUITE["timed_layers"] == list(contract.PUBLIC_LAYERS)


def test_public_boundary_lengths_differ_from_the_judge_except_the_contract_minimum():
    sys.path.insert(0, str(ROOT / "runtime"))
    from kernel_fidelity_v3 import BOUNDARY_LENGTHS
    assert set(public.SUITE["boundary_lengths"]) & set(BOUNDARY_LENGTHS) == {64}


def test_agent_visible_files_say_nothing_about_the_judge_suite():
    texts = {path: path.read_text() for path in (*WORK.rglob("*.md"), *WORK.rglob("*.sh"), *(WORK / "benchmark").glob("public_*.py"))}
    for path, text in texts.items():
        for token in ("niah_single_2", "niah_multikey_1", "131072 groups", "Llama-3.1", "final-performance", "kernel-prompts",
                      "48 group", "all48", "PyroMind", "HAMi", "exactly the same", "terminal300"):
            assert token not in text, (path.name, token)


def test_work_image_context_has_no_shared_runtime_or_recipe():
    dockerfile = (WORK / "Dockerfile").read_text()
    assert "public_metadata" not in dockerfile
    assert not (WORK / "benchmark/external_assets_v3.py").exists()
    stage = (ROOT / "scripts/stage_thin_images_v3.py").read_text()
    assert "'public/" not in stage


def test_brief_and_prompt_state_the_length_range_and_final_score_rule():
    assert (WORK / "AGENT_TASK.md").read_bytes() == (LAB / "README_AGENT.md").read_bytes()
    brief = (WORK / "AGENT_TASK.md").read_text()
    assert "64 <= N <= 131072" in brief and "best valid score" in brief
    assert re.search(r"Single-GPU|shared device|on one H100", brief) is None
    query = json.loads((ROOT / "tasks/minference_sparse_prefill_h100_seed_v3.json").read_text())["work"]["agent_query"]
    assert "64 to 131072" in query and "best valid score" in query and "single-GPU" not in query


def test_assets_route_each_model_to_one_side():
    assets = {row["path"]: row for row in json.loads((ROOT / "assets/manifest.json").read_text())["assets"]}
    assert assets["public/models/Llama-3-8B-Instruct-262k/"]["consumer"] == "work"
    assert assets["public/models/Meta-Llama-3.1-8B-Instruct/"]["consumer"] == "judge"
    assert assets["public/public-prompts/"]["consumer"] == "work"
    assert assets["judge/kernel-prompts/"]["consumer"] == "judge"
