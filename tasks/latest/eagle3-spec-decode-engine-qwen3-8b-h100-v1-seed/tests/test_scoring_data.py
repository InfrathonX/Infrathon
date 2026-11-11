"""Prompt split sizes, commitments, and source disjointness."""

from __future__ import annotations

import hashlib
import json
import math

from conftest import ROOT, private_data
from scoring import cell_throughput, geometric_mean

PUBLIC = ROOT / "images/work/workdir/spec_lab/public_prompts"


def rows(path):
    """Load nonempty JSONL records."""
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_scoring_math() -> None:
    measurements = [{"generated_lengths": [2, 3], "wall_s": 0.5}]
    assert cell_throughput(measurements) == 10
    assert math.isclose(geometric_mean([4, 9]), 6)


def test_split_sizes_and_commitments() -> None:
    PRIVATE = private_data()
    commitments = json.loads((PRIVATE / "commitments.json").read_text())
    public_counts = {"chat_bs2": 4, "math_bs4": 8, "code_bs3": 6, "longctx_bs2": 4, "think_bs2": 4, "mixed_bs6": 12}
    for cell, count in public_counts.items():
        assert len(rows(PUBLIC / f"{cell}.jsonl")) == count
    for cell in ("chat_bs1", "math_bs1", "code_bs1", "longctx_bs1", "think_bs1"):
        assert len(rows(PRIVATE / f"anchor/{cell}.jsonl")) == 8
        assert len(rows(PRIVATE / f"final/{cell}.jsonl")) == 20
    assert len(rows(PRIVATE / "anchor/mixed_bs8.jsonl")) == 16
    assert len(rows(PRIVATE / "final/mixed_bs8.jsonl")) == 40
    for key, expected in commitments.items():
        split, cell = key.split(":")
        if split == "public":
            # Retired public-v1 digests kept in the unchanged Judge data; the
            # Judge never reads them. public-v2 is checked by its own manifest.
            continue
        path = PRIVATE / split / f"{cell}.jsonl"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_c2_c6_questions_disjoint() -> None:
    PRIVATE = private_data()

    def source_ids(*paths):
        return {row["source_id"] for path in paths for row in rows(path)}

    def hidden(cell):
        return PRIVATE / f"anchor/{cell}.jsonl", PRIVATE / f"final/{cell}.jsonl"

    assert source_ids(*hidden("math_bs1")).isdisjoint(source_ids(*hidden("think_bs1")))
    assert source_ids(PUBLIC / "math_bs4.jsonl").isdisjoint(source_ids(PUBLIC / "think_bs2.jsonl"))


def test_public_shares_no_source_or_prompt_with_judge() -> None:
    """public-v2 draws from splits the Judge never uses and repeats no Judge message."""
    PRIVATE = private_data()
    public = [row for path in sorted(PUBLIC.rglob("*.jsonl")) for row in rows(path)]
    judge = [row for path in sorted(PRIVATE.rglob("*.jsonl")) for row in rows(path)]
    assert {row["source"] for row in public} == {
        "ultrachat_200k_test_sft", "gsm8k_train", "mbpp_sanitized_test",
    }
    assert not {row["source"] for row in public} & {row["source"] for row in judge}
    judge_text = {msg["content"].strip() for row in judge for msg in row["messages"]}
    assert not [row["prompt_id"] for row in public
                if any(msg["content"].strip() in judge_text for msg in row["messages"])]
