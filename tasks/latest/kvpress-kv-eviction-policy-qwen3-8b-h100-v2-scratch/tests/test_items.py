from __future__ import annotations

import json
from collections import Counter

import pytest

from conftest import final_manifest_path
import kv_eviction_runtime as rt



@pytest.fixture(scope="module")
def MANIFEST() -> dict:
    return json.loads(final_manifest_path().read_text(encoding="utf-8"))


def test_public_suite_shape() -> None:
    items = rt.PUBLIC_ITEMS
    assert len(items) == 85
    assert {x["family"] for x in items} == set(rt.FAMILIES)
    tasks = Counter(x["task"].split(".")[0] for x in items)
    assert tasks == {"ruler": 65, "longbench": 20}
    real = {x["task"].removesuffix("_e") for x in items if x["task"].startswith("longbench.")}
    assert real == {"longbench.hotpotqa", "longbench.triviaqa", "longbench.qmsum", "longbench.passage_count"}
    assert len({x["item_id"] for x in items}) == 85


def test_public_subsets() -> None:
    quick = rt.suite_items("quick")
    assert len(quick) == len({(x["family"], x["task"]) for x in rt.PUBLIC_ITEMS})
    assert rt.suite_items("one") == rt.PUBLIC_ITEMS[:1]
    assert rt.suite_items("full") == rt.PUBLIC_ITEMS
    with pytest.raises(ValueError):
        rt.suite_items("anchor")


def test_item_validation_rules() -> None:
    base = {"item_id": "x-1", "family": "qa", "task": "longbench.hotpotqa", "context": "ctx", "question": "q?", "answer_prefix": "Answer:"}
    assert rt.validate_item(base)["answer_prefix"] == "Answer:"
    # summarisation items legitimately have an empty question
    assert rt.validate_item({**base, "question": ""})["question"] == ""
    for bad in ({"family": "needle"}, {"context": "  "}, {"item_id": "../x"}, {"answer_prefix": 3}, {"task": "Bad Task"}):
        with pytest.raises(ValueError):
            rt.validate_item({**base, **bad})


def test_phase_a_sees_context_only() -> None:
    item = rt.PUBLIC_ITEMS[0]
    seen = rt.context_only(item)
    assert set(seen) == {"item_id", "context"} and seen["context"] == item["context"]
    assert rt.validate_context_item(seen) == seen
    with pytest.raises(ValueError):
        rt.validate_context_item({**seen, "question": item["question"]})


def test_item_identity_covers_question_and_prefix() -> None:
    item = dict(rt.PUBLIC_ITEMS[0])
    digest = rt.item_sha256(item)
    assert rt.item_sha256({**item, "question": item["question"] + "?"}) != digest
    assert rt.item_sha256({**item, "answer_prefix": item["answer_prefix"] + " "}) != digest


def test_final_manifest_structure(MANIFEST: dict) -> None:
    assert MANIFEST["suite_id"] == rt.FINAL_SUITE_ID == "final-holdout-v2"
    entries = MANIFEST["items"]
    assert len(entries) == MANIFEST["item_count"] == 5000
    assert Counter(e["family"] for e in entries) == MANIFEST["counts_per_family"]
    assert set(MANIFEST["counts_per_family"]) == set(rt.FAMILIES)
    # opaque ids: Phase A sees them, so they must not name the task or source
    assert [e["item_id"] for e in entries] == [f"f{i:05d}" for i in range(5000)]
    assert len({e["sha256"] for e in entries}) == 5000
    lo, hi = MANIFEST["context_tokens"]
    assert all(lo <= e["context_tokens"] <= hi for e in entries)
    assert "context" not in json.dumps(entries[0]) or "context_tokens" in entries[0]
    public = {rt.item_sha256(x) for x in rt.PUBLIC_ITEMS}
    assert not public & {e["sha256"] for e in entries}


def test_final_manifest_source_split(MANIFEST: dict) -> None:
    holdout_corpora = {e["task"] for e in MANIFEST["items"] if not e["task"].startswith("ruler.")}
    for public_corpus in ("hotpotqa", "triviaqa", "qmsum", "passage_count"):
        assert not any(task.split(".", 1)[1].startswith(public_corpus) for task in holdout_corpora), public_corpus
    excluded = ("samsum", "repobench", "multi_news", "book_sum_sort", "lcc", "trec", "multifieldqa", "summ_screen", "space_digest", "shortdep_cloze")
    assert not any(any(name in task for name in excluded) for task in holdout_corpora)
    ruler_public = {(x["meta"]["source"]["file"], x["meta"]["source"]["row"]) for x in rt.PUBLIC_ITEMS if x["task"].startswith("ruler.")}
    ruler_final = {(e["source"]["file"], e["source"]["row"]) for e in MANIFEST["items"] if e["task"].startswith("ruler.")}
    assert ruler_public and ruler_final and not ruler_public & ruler_final


def test_validate_final_items_checks_every_item(MANIFEST: dict) -> None:
    entries = MANIFEST["items"][:2]
    rows = [{"item_id": e["item_id"], "family": e["family"], "task": e["task"], "context": "c", "question": "q", "answer_prefix": ""} for e in entries]
    manifest = {**MANIFEST, "items": entries, "counts_per_family": dict(Counter(e["family"] for e in entries))}
    with pytest.raises(ValueError, match="text differs"):
        rt.validate_final_items(rows, manifest)
