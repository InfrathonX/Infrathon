"""CPU-only checks that Work holds a small public workload and the scored workload stays with the Judge."""
import json
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
REPO = TASK.parents[2]


def contract(name):
    return json.loads((TASK / "runtime" / name).read_text())


def test_public_contract_is_small():
    public, scored = contract("public_contract.json"), contract("contract.json")
    assert (public["rounds"], public["games_per_round"]) == (4, 8)
    assert (scored["rounds"], scored["games_per_round"]) == (12, 16)
    # Everything but the size is the same training recipe.
    differ = {"rounds", "games_per_round"}
    assert {k: v for k, v in public.items() if k not in differ} == {k: v for k, v in scored.items() if k not in differ}


def test_work_and_judge_build_from_their_own_data():
    build = (TASK / "scripts/build_images.sh").read_text()
    assert 'cp -a "${work_source}/." "${STAGE}/work/public-data/"' in build
    assert 'cp -a "${public_source}/." "${STAGE}/judge/public-data/"' in build
    assert 'cp "${TASK}/runtime/public_contract.json" "${STAGE}/work/public/contract.json"' in build
    assert 'wc -l < /data/public/train_public.jsonl)" = 32' in (TASK / "images/work/Dockerfile").read_text()
    assets = {a["path"]: a for a in json.loads((TASK / "assets/manifest.json").read_text())["assets"]}
    assert assets["public/work-task-data/"]["files"] == 51


def test_public_runner_reports_no_score():
    public = (REPO / "scripts/miles_alfworld/run_public.py").read_text()
    assert "provisional" not in public and "metric_v2" not in public and "1134" not in public


def test_work_gets_no_reference_results():
    assert not (TASK / "runtime/published_reference.json").exists()


def test_agent_text_does_not_state_the_scored_workload():
    prompt = json.loads((TASK / "tasks/miles_qwen3_alfworld_agentic_rl_2xh100_v1.json").read_text())["work"]["agent_query"]
    texts = {"AGENT_TASK.md": (TASK / "images/work/AGENT_TASK.md").read_text(),
             "INTERFACE.md": (TASK / "docs/INTERFACE.md").read_text(), "agent_query": prompt}
    for name, text in texts.items():
        for phrase in ("12, 16", "(12)", "16r", "128 ", "1134", "140 ", "provisional", "fixed-size", "same two GPUs", "same assigned pair"):
            assert phrase not in text, (name, phrase)
    assert "only to get the pipeline working" in prompt
    assert "only for getting your pipeline working" in texts["AGENT_TASK.md"]
    assert texts["AGENT_TASK.md"].index("## Score") < texts["AGENT_TASK.md"].index("## Submission evaluation policy")
