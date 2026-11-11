"""CPU-only checks that Work holds a small public workload and the scored workload stays with the Judge."""
import json
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
REPO = TASK.parents[2]


def contract(name):
    return json.loads((TASK / "runtime" / name).read_text())


def test_public_contract_is_small_and_names_public_rows():
    public, scored = contract("public_contract.json"), contract("contract.json")
    assert (public["rounds"], public["prompts_per_round"]) == (8, 32)
    assert (scored["rounds"], scored["prompts_per_round"]) == (32, 128)
    assert public["train_data"] == "/data/public/train_public.jsonl"
    assert scored["train_data"] == "/data/public/train.jsonl"
    # Everything but the data and the size is the same training recipe.
    differ = {"train_data", "rounds", "prompts_per_round"}
    assert {k: v for k, v in public.items() if k not in differ} == {k: v for k, v in scored.items() if k not in differ}


def test_work_and_judge_build_from_their_own_data():
    build = (TASK / "scripts/build_images.sh").read_text()
    assert 'cp "${work_source}/"* "${STAGE}/work/public-data/"' in build
    assert 'cp "${public_source}/"* "${STAGE}/judge/public-data/"' in build
    assert 'cp "${TASK}/runtime/public_contract.json" "${STAGE}/work/public/contract.json"' in build
    assert "test ! -e /data/public/train.jsonl" in (TASK / "images/work/Dockerfile").read_text()
    assets = {a["path"]: a for a in json.loads((TASK / "assets/manifest.json").read_text())["assets"]}
    assert assets["public/work-task-data/"]["files"] == 4


def test_public_runner_reports_no_score():
    public = (REPO / "scripts/miles_opd/v2/run_public.py").read_text()
    assert "provisional_score" not in public and "metric_v2" not in public
    assert "2436" not in public and "32 rounds x 512" not in public


def test_published_reference_holds_no_reference_results():
    published = json.loads((TASK / "runtime/published_reference.json").read_text())
    assert set(published) == {"public_base_accuracy_per_repeat", "note"}


def test_agent_text_does_not_state_the_scored_workload():
    prompt = json.loads((TASK / "tasks/miles_qwen3_opd_scratch_2xh100_v2.json").read_text())["work"]["agent_query"]
    texts = {"AGENT_TASK.md": (TASK / "images/work/AGENT_TASK.md").read_text(),
             "INTERFACE.md": (TASK / "docs/INTERFACE.md").read_text(), "agent_query": prompt}
    for name, text in texts.items():
        for phrase in ("32 rounds", "128r", "2,000", "train.jsonl", "provisional", "same two GPUs"):
            assert phrase not in text, (name, phrase)
    assert "only to get the pipeline working" in prompt and "sforge-submit" in prompt
    assert "only for\ngetting your pipeline working" in texts["AGENT_TASK.md"]
    assert "`run` after 5400 s" in texts["AGENT_TASK.md"]
    brief = texts["AGENT_TASK.md"]
    assert brief.index("## Score") < brief.index("## Submission evaluation policy") < brief.index("## Submission workflow")
