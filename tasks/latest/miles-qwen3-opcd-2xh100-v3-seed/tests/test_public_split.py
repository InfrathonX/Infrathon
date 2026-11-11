"""CPU-only checks that Work holds a small public workload and the scored workload stays with the Judge."""
import json
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
REPO = TASK.parents[2]


def contract(name):
    return json.loads((TASK / "runtime" / name).read_text())


def test_public_contract_is_small_and_names_public_rows():
    public, scored = contract("public_contract.json"), contract("contract.json")
    assert (public["rounds"], public["prompts_per_round"]) == (8, 64)
    assert (scored["rounds"], scored["prompts_per_round"]) == (32, 512)
    assert public["train_data"] == "/data/public/train_public.jsonl"
    assert scored["train_data"] == "/data/public/train.jsonl"
    # Everything but the data and the size is the same training recipe.
    differ = {"train_data", "rounds", "prompts_per_round"}
    assert {k: v for k, v in public.items() if k not in differ} == {k: v for k, v in scored.items() if k not in differ}


def test_extra_work_rows_are_new():
    rows = json.loads((TASK / "scripts/medmcqa_row_map.json").read_text())
    added = json.loads((TASK / "scripts/medmcqa_work_rows.json").read_text())["added"]
    assert len(rows["train_public"]) == 64 and len(added) == len(set(added)) == 448
    for key in ("train", "train_public", "test_public", "eval", "paper_test", "probe_public", "probe_private"):
        assert not set(added) & set(rows[key]), key


def test_work_and_judge_build_from_their_own_data():
    build = (TASK / "scripts/build_images.sh").read_text()
    assert 'cp "${work_source}/"* "${STAGE}/work/public-data/"' in build
    assert 'cp "${public_source}/"* "${STAGE}/judge/public-data/"' in build
    assert 'cp "${TASK}/runtime/public_contract.json" "${STAGE}/work/public/contract.json"' in build
    assert "test ! -e /data/public/train.jsonl" in (TASK / "images/work/Dockerfile").read_text()
    assets = {a["path"]: a for a in json.loads((TASK / "assets/manifest.json").read_text())["assets"]}
    assert assets["public/work-task-data/"]["files"] == 5


def test_public_runner_reports_no_score():
    public = (REPO / "scripts/miles_opcd/v3/run_public_v3.py").read_text()
    assert "provisional_score" not in public and "metric_v2" not in public


def test_published_reference_holds_no_reference_results():
    published = json.loads((TASK / "runtime/published_reference.json").read_text())
    assert set(published) == {"public_base_accuracy_per_repeat", "note"}


def test_agent_text_does_not_state_the_scored_workload():
    prompt = json.loads((TASK / "tasks/miles_qwen3_opcd_2xh100_v3.json").read_text())["work"]["agent_query"]
    texts = {"AGENT_TASK.md": (TASK / "images/work/AGENT_TASK.md").read_text(),
             "INTERFACE.md": (TASK / "docs/INTERFACE.md").read_text(), "agent_query": prompt}
    for name, text in texts.items():
        for phrase in ("32 rounds", "16384", "2436", "train.jsonl", "provisional", "same two GPUs", "paper's", "OPCD paper"):
            assert phrase not in text, (name, phrase)
    assert "only" in prompt and "sforge-submit" in prompt
    assert "only for\ngetting your pipeline working" in texts["AGENT_TASK.md"]
