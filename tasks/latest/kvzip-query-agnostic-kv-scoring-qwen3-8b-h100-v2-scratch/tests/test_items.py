from __future__ import annotations

import json
from collections import Counter

import pytest

from conftest import judge_input
import kv_scoring_runtime as rt

HOLDOUT_COUNTS = {"gsm": 200, "needle": 96, "ruler_niah": 192, "ruler_qa": 152, "ruler_vt": 234, "scbench_kv": 150, "scbench_many_shot": 54,
                  "scbench_mf": 250, "scbench_prefix_suffix": 150, "scbench_repoqa": 22, "scbench_summary": 62, "squad": 250}


def test_public_suite_shape() -> None:
    items = rt.PUBLIC_ITEMS
    assert len(items) == 63
    assert Counter(x["task"] for x in items) == {"needle": 11, "squad": 10, "gsm": 10, "scbench_kv": 8, "scbench_prefix_suffix": 8,
                                                  "scbench_mf": 8, "ruler_niah": 4, "ruler_qa": 2, "ruler_vt": 2}
    assert all(rt.MIN_CTX_TOKENS <= x["meta"]["context_tokens"] <= rt.MAX_CTX_TOKENS for x in items)


def test_quick_takes_two_rows_per_task() -> None:
    quick = rt.suite_items("quick")
    counts = Counter(x["task"] for x in quick)
    assert all(v == 2 for v in counts.values()) and set(counts) == {x["task"] for x in rt.PUBLIC_ITEMS}
    with pytest.raises(ValueError):
        rt.suite_items("anchor")


def test_task_registry_groups() -> None:
    groups = Counter(spec.group for spec in rt.TASKS.values())
    assert groups == {"retrieval": 5, "contextual_qa": 3, "redundancy": 4}
    assert rt.TASKS["scbench_repoqa"].metric == "repoqa" and rt.TASKS["ruler_qa"].metric == "ruler_part"
    assert rt.MAX_CTX_TOKENS == 32000 and rt.MODEL_WINDOW_TOKENS == 40960


def test_phase_a_sees_context_only() -> None:
    item = next(x for x in rt.PUBLIC_ITEMS if x["task"] == "gsm")
    seen = rt.context_only(item)
    assert seen == {"item_id": item["item_id"], "prompt": "reasoning", "context": item["context"]}
    with pytest.raises(ValueError):
        rt.validate_context_item({**seen, "questions": item["questions"]})


def test_answers_may_be_value_lists_and_prefix_is_text() -> None:
    row = next(x for x in rt.PUBLIC_ITEMS if x["task"] == "ruler_niah")
    assert isinstance(row["answers"][0], list) and isinstance(row["meta"]["answer_prefix"], str)
    with pytest.raises(ValueError):
        rt.validate_item({**row, "answers": [[]]})
    with pytest.raises(ValueError):
        rt.validate_item({**row, "meta": {**row["meta"], "answer_prefix": 7}})


@pytest.fixture(scope="module")
def MANIFEST() -> dict:
    return json.loads(judge_input("judge/private/final_manifest.json").read_text(encoding="utf-8"))


def test_final_manifest_structure(MANIFEST: dict) -> None:
    assert MANIFEST["suite_id"] == rt.FINAL_SUITE_ID == "final-holdout-v2"
    entries = MANIFEST["items"]
    assert len(entries) == MANIFEST["row_count"] == 1812
    assert Counter(e["task"] for e in entries) == HOLDOUT_COUNTS == MANIFEST["counts_per_task"]
    assert [e["item_id"] for e in entries] == [f"f{i:04d}" for i in range(1812)]
    assert len({e["sha256"] for e in entries}) == 1812
    assert not {rt.item_sha256(x) for x in rt.PUBLIC_ITEMS} & {e["sha256"] for e in entries}


def test_holdout_follows_the_upstream_protocol_rows(MANIFEST: dict) -> None:
    upstream = [e for e in MANIFEST["items"] if e["task"] in ("scbench_kv", "scbench_prefix_suffix", "scbench_mf") and "repo" in e["origin"]]
    assert Counter(e["task"] for e in upstream) == {"scbench_kv": 100, "scbench_prefix_suffix": 100, "scbench_mf": 100}
    squad = sorted(e["origin"]["row"] for e in MANIFEST["items"] if e["task"] == "squad")
    assert squad[0] == 0 and all(e["origin"]["file"].endswith("train-00000-of-00001.parquet") for e in MANIFEST["items"] if e["task"] == "squad")
    needles = Counter(e["origin"]["niah"]["needle"] for e in MANIFEST["items"] if e["task"] == "needle")
    assert set(needles) == {0, 1, 2}


def test_validate_final_items_rejects_text_drift(MANIFEST: dict) -> None:
    entry = MANIFEST["items"][0]
    row = {"item_id": entry["item_id"], "task": entry["task"], "context": "x" * 400, "questions": ["q"], "answers": ["a"], "meta": {}}
    manifest = {**MANIFEST, "items": [entry], "counts_per_task": {entry["task"]: 1}}
    with pytest.raises(ValueError, match="text differs"):
        rt.validate_final_items([row], manifest)
