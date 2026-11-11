"""The frozen hidden suite and the public suite partition a serving workload.

Every hidden case is a frozen decode call (or sequence of calls) of one
tensor-parallel rank of a serving engine; every producer (runtime, Judge
verifier, manifest converter, public runner) must agree on the four families
and the schema.  The public suite covers the same families with disjoint head
configurations, seeds below 2**32 and different contents, and the hidden suite
keeps exercising the properties that make it a serving workload: very uneven
lengths in one batch, shared prefix pages, long contexts, speculative query
lengths that are not powers of two, several page sizes and sequences whose
batch changes on every step.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from conftest import TASK_ROOT, load_module, judge_only_file


verifier = load_module("paged_gqa_v4_verifier_partition", "images/judge/verify_paged_gqa.py")
runtime = load_module("paged_gqa_v4_runtime_partition", "runtime/paged_gqa_decode_runtime.py")
generator = load_module("paged_gqa_v4_manifest_generator", "scripts/generate_final_manifest.py")
public_runner = load_module("paged_gqa_v4_public_runner", "images/work/benchmark/public_runner.py")
public_runtime = load_module("paged_gqa_v4_public_runtime", "images/work/benchmark/paged_gqa_decode_runtime.py")

PUBLIC_CASES = public_runner.PUBLIC_CASES
FROZEN = json.loads(judge_only_file("judge/private/final_cases.json").read_text(encoding="utf-8"))
HIDDEN = [runtime.validate_scored_case(case) for case in FROZEN["cases"]]


def test_families_schema_and_numerical_contract_match_every_producer() -> None:
    assert runtime.DECODE_ENTRYPOINT == verifier.DECODE_ENTRYPOINT == "paged_gqa_decode_bf16"
    assert runtime.HEAD_DIM == verifier.HEAD_DIM == 128
    assert runtime.FAMILIES == verifier.FAMILIES == generator.FAMILIES == public_runner.FAMILIES == (
        "decode", "spec", "long", "stream")
    assert runtime.CALL_KEYS == verifier.CALL_KEYS == set(generator.CALL_KEYS) == public_runtime.CALL_KEYS
    assert runtime.STREAM_KEYS == verifier.STREAM_KEYS == set(generator.STREAM_KEYS) == public_runtime.STREAM_KEYS
    assert verifier.FINAL_CASE_COUNT == generator.FINAL_CASE_COUNT == len(FROZEN["cases"]) == 1114
    assert generator.PUBLIC_CASE_COUNT == len(PUBLIC_CASES) == 127
    assert verifier.FINAL_SEED_MINIMUM == generator.FINAL_SEED_MINIMUM == 1 << 32
    assert runtime.MAX_L_MAX == verifier.MAX_L_MAX == public_runtime.MAX_L_MAX == 1 << 20
    assert runtime.BLOCK_SIZES == verifier.BLOCK_SIZES == (16, 32, 64, 128, 256, 512)
    assert runtime.SCORED_VALUES == verifier.SCORED_VALUES == ("randn", "outlier")


def test_the_frozen_hidden_suite_validates_and_counts_every_family() -> None:
    judged = verifier.validate_final_manifest(FROZEN)
    assert Counter(case["family"] for case in judged) == {"decode": 440, "spec": 374, "long": 220, "stream": 80}
    for checked, expanded in zip(judged, HIDDEN):
        assert checked["total_pairs"] == expanded["total_pairs"]
        assert checked["b"] == expanded["b"] and checked["l_max"] == expanded["l_max"]
    assert len({case["seed"] for case in FROZEN["cases"]}) == 1114
    assert len({case["case_id"] for case in FROZEN["cases"]}) == 1114


def test_public_suite_is_disjoint_from_the_hidden_suite() -> None:
    public = [public_runtime.validate_scored_case(case) for case in PUBLIC_CASES]
    heads_public = {(case["h_q"], case["h_kv"]) for case in public}
    heads_hidden = {(case["h_q"], case["h_kv"]) for case in HIDDEN}
    assert heads_public.isdisjoint(heads_hidden)
    assert all(case["seed"] < verifier.FINAL_SEED_MINIMUM for case in PUBLIC_CASES)
    assert {case["seed"] for case in PUBLIC_CASES}.isdisjoint({case["seed"] for case in FROZEN["cases"]})
    assert {case["family"] for case in public} == set(runtime.FAMILIES)
    work = lambda cases: sum(case["total_pairs"] * case["h_q"] for case in cases)  # noqa: E731
    assert work(HIDDEN) >= 5 * work(public)
    assert max(case["l_max"] for case in HIDDEN) > max(case["l_max"] for case in public)


def test_public_suite_holds_calls_at_the_scale_of_the_busier_hidden_calls() -> None:
    by_id = {case["case_id"]: case for case in PUBLIC_CASES}
    kv = lambda case: sum(case["lengths"]) * case["h_kv"] * 128 * 2 * 2  # noqa: E731
    for case_id, family in (("public-0125-decode", "decode"), ("public-0126-spec", "spec")):
        case = by_id[case_id]
        assert case["family"] == family
        hidden = sorted(kv(other) for other in FROZEN["cases"] if other["family"] == family)
        assert kv(case) >= hidden[3 * len(hidden) // 4]
        assert len(case["lengths"]) >= sorted(len(o["lengths"]) for o in FROZEN["cases"]
                                              if o["family"] == family)[3 * len(hidden) // 4]


def test_the_hidden_suite_is_a_serving_workload() -> None:
    calls = [case for case in HIDDEN if case["family"] != "stream"]
    decode = [case for case in calls if case["family"] == "decode"]
    uneven = [case for case in decode
              if case["b"] > 1 and max(case["lengths"]) >= 2 * sum(case["lengths"]) / case["b"]]
    assert len(uneven) >= len(decode) // 2
    shared = [case for case in calls
              if len({block for ids in case["blocks"] for block in ids}) < sum(map(len, case["blocks"]))]
    assert len(shared) >= 20
    assert max(case["l_max"] for case in calls) > (1 << 19)
    assert sum(case["l_max"] >= 131072 for case in calls) >= 20
    spec_lengths = {case["s_q"] for case in calls if case["family"] == "spec"}
    assert {3, 5, 6} <= spec_lengths and max(spec_lengths) == 16
    assert {1, 16, 64, 128} <= {case["page_size"] for case in HIDDEN}
    assert any(case["fill"] == "nan" for case in calls)
    assert {case["h_q"] // case["h_kv"] for case in HIDDEN} >= {4, 7, 8, 12, 16}
    # PagedAttention sharing workloads: groups of requests behind the same pages.
    heavy = [case for case in calls if case["block_tokens"] == 16]
    assert len(heavy) >= 16
    for case in heavy:
        refs = sum(map(len, case["blocks"]))
        assert refs >= 2 * len({block for ids in case["blocks"] for block in ids})
    # FlashAttention-3 outlier values on a quarter of the single calls.
    assert sum(case["values"] == "outlier" for case in calls) >= len(calls) // 5
    streams = [case for case in HIDDEN if case["family"] == "stream"]
    assert len(streams) == 80 and {case["s_q"] for case in streams} >= {1, 2, 3, 4, 5}
    assert {(case["h_q"], case["h_kv"]) for case in streams} >= {(case["h_q"], case["h_kv"]) for case in calls
                                                                if case["family"] == "decode"}
    for case in streams:
        plan = runtime.stream_plan(case)
        sizes = [len(now) for now in plan.active]
        # The batch changes size, or (a server saturated at its sequence cap)
        # stays full while many requests leave and others take their place.
        leaving = sum(end < 512 for _, end, _, _ in case["requests"])
        assert len(plan.active) == 512 and (len(set(sizes)) >= 3 or leaving >= 64)
        assert any(plan.starts[index] > 0 for index in range(len(plan.starts)))
    assert sum(len({len(now) for now in runtime.stream_plan(case).active}) >= 3 for case in streams) >= 70


def test_root_only_context_routes_every_round_to_the_shared_final(tmp_path: Path) -> None:
    context = tmp_path / "context.json"

    def write(payload: dict[str, Any]) -> None:
        context.write_text(json.dumps(payload), encoding="utf-8")
        context.chmod(0o600)

    for round_id in ("auto-123", "agent-2", "manual-final", "candidate-v3-attempt-1"):
        write({"round_id": round_id})
        assert verifier.load_evaluation_profile(context, expected_uid=os.getuid())[:2] == (
            "final",
            "final-holdout-v4",
        )
    for bare_prefix in ("auto-", "agent-", "manual-", "candidate-"):
        write({"round_id": bare_prefix})
        with pytest.raises(RuntimeError, match="unknown profile"):
            verifier.load_evaluation_profile(context, expected_uid=os.getuid())
    write({"round_id": "candidate-impl-v3", "candidate_version": 3})
    assert verifier.load_evaluation_profile(context, expected_uid=os.getuid())[3] == 3
    write({"round_id": "agent-2", "candidate_version": 3})
    with pytest.raises(RuntimeError, match="host-side only"):
        verifier.load_evaluation_profile(context, expected_uid=os.getuid())
    for bad in (0, 1000, True, "3"):
        write({"round_id": "candidate-x", "candidate_version": bad})
        with pytest.raises(RuntimeError, match="candidate_version"):
            verifier.load_evaluation_profile(context, expected_uid=os.getuid())
    write({"round_id": "preview-1"})
    with pytest.raises(RuntimeError, match="unknown profile"):
        verifier.load_evaluation_profile(context, expected_uid=os.getuid())
    context.chmod(0o644)
    with pytest.raises(RuntimeError, match="root-only"):
        verifier.load_evaluation_profile(context, expected_uid=os.getuid())
