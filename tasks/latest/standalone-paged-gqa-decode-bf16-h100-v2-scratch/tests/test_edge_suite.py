"""The edge suites cover the declared support domain, not just the scored grid.

Both frozen instances must be exactly what the seeded generator produces, the
Judge's private instance must be the one the verifier and build expect, and
together the cases must exercise every value the upstream tests list: batch
sizes such as 1, 3, 7 and 255; lengths from S_q and 1 up to 32768 including
page boundaries; every S_q from the upstream tests; non-power-of-two GQA
groups; page sizes 1-256; wide tables, oversized pools with repeated ids and
shared prefix pages; NaN/Inf hidden slots and extreme logits; the v1 contract
subset with the two historical cross combinations; and a sanitizer subset
that ends with the one sanitizer-only large-offset case.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter

import pytest

from conftest import TASK_ROOT, load_module, judge_only_file


runtime = load_module("edge_suite_runtime", "runtime/paged_gqa_decode_runtime.py")
verifier = load_module("edge_suite_verifier", "images/judge/verify_paged_gqa.py")
generator = load_module("edge_suite_generator", "scripts/generate_edge_manifest.py")
PRIVATE = json.loads(judge_only_file("judge/private/edge_cases.json").read_text(encoding="utf-8"))
PUBLIC = json.loads(
    (TASK_ROOT / "images/work/benchmark/public_edge_cases.json").read_text(encoding="utf-8")
)


def _kv(case: dict) -> int:
    return runtime.kv_cache_bytes(case["pool_pages"], case["page_size"], case["h_kv"])


def _needed(case: dict) -> int:
    return runtime.blocks_for(case["l_max"], case["page_size"])


def _group(case: dict) -> int:
    return case["h_q"] // case["h_kv"]


def test_frozen_instances_are_the_seeded_generator_output() -> None:
    completed = subprocess.run(
        [sys.executable, "-B", str(TASK_ROOT / "scripts/generate_edge_manifest.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "edge-holdout-v4: 168 cases, 167 native, 12 sanitize (1 sanitizer-only)" in completed.stdout


def test_identities_counts_and_categories() -> None:
    assert PRIVATE["schema_version"] == PUBLIC["schema_version"] == 4
    assert PRIVATE["suite_id"] == generator.PRIVATE_SUITE_ID == verifier.EDGE_SUITE_ID
    assert PUBLIC["suite_id"] == generator.PUBLIC_SUITE_ID == "edge-public-v4"
    expected_private = sum(count for _name, count, _public in generator.CATEGORY_COUNTS)
    expected_public = sum(public for _name, _count, public in generator.CATEGORY_COUNTS)
    assert len(PRIVATE["cases"]) == expected_private == verifier.EDGE_CASE_COUNT == 168
    assert len(PUBLIC["cases"]) == expected_public == 44
    assert sum(case["sanitize"] for case in PRIVATE["cases"]) == verifier.EDGE_SANITIZE_COUNT == 12
    assert sum(not case["native"] for case in PRIVATE["cases"]) == verifier.EDGE_SANITIZE_ONLY_COUNT == 1
    assert sum(case["sanitize"] for case in PUBLIC["cases"]) == len(generator.SANITIZE_PUBLIC) == 4
    assert all(case["native"] for case in PUBLIC["cases"])
    for payload, column in ((PRIVATE, 1), (PUBLIC, 2)):
        counts = Counter(case["category"] for case in payload["cases"])
        assert counts == {row[0]: row[column] for row in generator.CATEGORY_COUNTS}
    # The sanitizer flag is exactly the sanitize category.
    for case in (*PRIVATE["cases"], *PUBLIC["cases"]):
        assert case["sanitize"] == (case["category"] == "sanitize")


def test_every_case_is_inside_the_declared_domain() -> None:
    for case in (*PRIVATE["cases"], *PUBLIC["cases"]):
        assert set(case) == runtime.EDGE_KEYS
        runtime.validate_edge_case(case)
        verifier.validate_edge_case(case)
    assert len(verifier.validate_edge_manifest(PRIVATE)) == verifier.EDGE_CASE_COUNT


def test_size_caps_keep_the_native_suite_fast() -> None:
    for payload in (PRIVATE, PUBLIC):
        native = [case for case in payload["cases"] if case["native"]]
        assert generator.CASE_KV_CAP == 64 << 20 and generator.LONG_CASE_KV_CAP == 1 << 30
        assert all(
            _kv(case) <= (generator.LONG_CASE_KV_CAP if case["category"] == "long_context" else generator.CASE_KV_CAP)
            for case in native
        )
        assert sum(map(_kv, native)) <= generator.NATIVE_KV_TOTAL_CAP
        assert all(
            case["b"] * case["s_q"] * case["h_q"] * runtime.HEAD_DIM <= runtime.MAX_OUTPUT_ELEMENTS
            for case in payload["cases"]
        )
    # The pinned tables are what the runtime and the generator share.
    assert generator.PAGE_SIZES == runtime.PAGE_SIZES
    assert set(generator.S_Q_VALUES) <= set(range(1, runtime.MAX_S_Q + 1))
    assert max(generator.BATCHES) == runtime.MAX_B and max(generator.UPSTREAM_LENGTHS) == 32768
    # The long-context category reaches the top of the v4 length domain.
    assert max(case["l_max"] for case in PRIVATE["cases"]) == runtime.MAX_L_MAX


def test_private_and_public_draws_do_not_overlap() -> None:
    private_ids = {case["case_id"] for case in PRIVATE["cases"]}
    public_ids = {case["case_id"] for case in PUBLIC["cases"]}
    assert private_ids.isdisjoint(public_ids)
    assert {case["seed"] for case in PRIVATE["cases"]}.isdisjoint(
        {case["seed"] for case in PUBLIC["cases"]}
    )
    identity = lambda case: json.dumps({k: v for k, v in case.items() if k not in ("case_id", "seed")}, sort_keys=True)  # noqa: E731
    assert {identity(case) for case in PRIVATE["cases"]}.isdisjoint(
        {identity(case) for case in PUBLIC["cases"]}
    )


def test_private_suite_covers_every_upstream_value_list() -> None:
    cases = PRIVATE["cases"]
    assert set(generator.BATCHES) <= {case["b"] for case in cases}
    assert set(generator.S_Q_VALUES) <= {case["s_q"] for case in cases}
    assert set(generator.HEAD_CONFIGS) <= {(case["h_q"], case["h_kv"]) for case in cases}
    assert set(generator.PAGE_SIZES) <= {case["page_size"] for case in cases}
    lengths = {value for case in cases for value in case["lengths"]}
    assert set(generator.UPSTREAM_LENGTHS) <= lengths
    assert {case["table_mode"] for case in cases} == set(runtime.TABLE_MODES)
    assert {case["fill"] for case in cases} == set(runtime.FILLS)
    assert {case["values"] for case in cases} == set(runtime.VALUES)
    # Non-power-of-two groups and query lengths, and the 2^k +/- 1 batches.
    assert {3, 6, 7, 12} <= {_group(case) for case in cases}
    assert sum(_group(case) & (_group(case) - 1) != 0 for case in cases) >= 30
    assert {3, 5, 7, 9, 15} <= {case["s_q"] for case in cases}
    # Contract corners: L_b = S_q, L_b = S_q + 1, every length 1, L_max = 32768.
    assert sum(case["s_q"] > 1 and case["s_q"] in case["lengths"] for case in cases) >= 20
    assert sum(case["s_q"] + 1 in case["lengths"] for case in cases) >= 20
    assert any(case["s_q"] == 1 and set(case["lengths"]) == {1} and case["b"] > 1 for case in cases)
    assert any(32768 in case["lengths"] for case in cases)
    # Page boundaries k*ps - 1 / k*ps / k*ps + 1 for every page size above 1.
    for page_size in generator.PAGE_SIZES[1:]:
        boundary = [
            case for case in cases
            if case["page_size"] == page_size
            and {value % page_size for value in case["lengths"]} >= {page_size - 1, 0, 1}
        ]
        assert boundary, page_size
    # Layouts: FA3's 3x table, the 4x table, vLLM's oversized random pool,
    # pages at the top of a larger pool, shared prefix pages.
    widths = Counter(case["table_width"] // _needed(case) for case in cases if case["table_width"] % _needed(case) == 0)
    assert widths[3] >= 10 and widths[4] >= 5
    assert sum(case["table_mode"] == "random" and case["pool_pages"] >= 2 * case["b"] * case["table_width"] for case in cases) >= 10
    assert sum(case["table_mode"] == "tail_perm" and case["pool_pages"] > case["b"] * case["table_width"] for case in cases) >= 10
    assert any(
        case["table_mode"] == "shared"
        and any(min(a, b) // 2 >= case["page_size"] for a, b in zip(case["lengths"][::2], case["lengths"][1::2]))
        for case in cases
    )
    # Hazards across many shapes, not one fixture each.
    assert sum(case["fill"] == "nan" for case in cases) >= 15
    assert sum(case["fill"] == "inf" for case in cases) >= 5
    assert sum(case["values"] == "negative_logits" for case in cases) >= 5
    assert sum(case["values"] == "large_logits" for case in cases) >= 5


def test_v1_contract_subset_stays_inside_v1() -> None:
    v1 = [case for case in PRIVATE["cases"] if case["category"] == "v1_contract"]
    assert len(v1) == 30
    for case in v1:
        assert (case["h_q"], case["h_kv"]) in generator.V1_HEADS
        assert case["page_size"] in generator.V1_PAGE_SIZES and case["s_q"] in generator.V1_S_Q
        assert case["fill"] == "randn" and case["values"] in ("randn", "large_logits")
        assert case["table_mode"] in ("perm", "identity", "reversed")
        assert case["table_width"] == _needed(case) and case["pool_pages"] == case["b"] * case["table_width"]
    # The two cross combinations that broke historical kernels come first,
    # with multi-page contexts, and every v1 head x page x S_q cross appears.
    first = [(case["h_q"], case["h_kv"], case["page_size"], case["s_q"]) for case in v1[:2]]
    assert first == [(64, 8, 16, 16), (32, 8, 64, 16)]
    assert all(_needed(case) >= 2 for case in v1[:2])
    crosses = {(case["h_q"], case["h_kv"], case["page_size"], case["s_q"]) for case in v1}
    assert len(crosses) == 18


def test_sanitizer_subset_covers_the_riskiest_layouts() -> None:
    sanitize = [case for case in PRIVATE["cases"] if case["sanitize"]]
    assert len(sanitize) == 12
    native = [case for case in sanitize if case["native"]]
    assert any(case["b"] == 1 for case in native)
    assert any(case["table_mode"] == "random" and case["page_size"] == 1 and case["pool_pages"] >= 16384 for case in native)
    assert any(
        (case["h_q"], case["h_kv"], case["page_size"], case["s_q"]) == (6, 3, 4, 3)
        and case["table_width"] == 3 * _needed(case)
        for case in native
    )
    assert any(_group(case) == 7 and case["s_q"] == 16 for case in native)
    assert any(_group(case) == 12 and case["s_q"] == 5 for case in native)
    assert any(case["page_size"] == 256 and max(case["lengths"]) > 256 for case in native)
    assert any(case["table_width"] == 4 * _needed(case) for case in native)
    assert any(case["fill"] == "nan" for case in native)
    assert any(case["values"] == "negative_logits" for case in native)
    assert any(case["table_mode"] == "shared" for case in native)
    assert any((case["h_q"], case["h_kv"], case["page_size"], case["s_q"]) == (64, 8, 16, 16) for case in native)
    # The large-offset case: sanitizer only, every used page beyond 2**31
    # K elements, within the runtime's sanitizer-only bound.
    (large,) = [case for case in sanitize if not case["native"]]
    page_elements = large["page_size"] * large["h_kv"] * runtime.HEAD_DIM
    assert large["table_mode"] == "tail_perm"
    assert large["pool_pages"] * page_elements > 1 << 31
    assert (large["pool_pages"] - large["b"] * large["table_width"]) * page_elements >= 1 << 31
    assert runtime.MAX_CASE_KV_BYTES < _kv(large) <= runtime.MAX_SANITIZE_ONLY_KV_BYTES
    assert large is sanitize[-1]


def test_generator_is_deterministic_and_seed_sensitive() -> None:
    first = generator.generate(generator.PUBLIC_SUITE_ID, public=True)
    assert generator.encode(first) == generator.encode(
        generator.generate(generator.PUBLIC_SUITE_ID, public=True)
    )
    assert generator.encode(first) == (TASK_ROOT / "images/work/benchmark/public_edge_cases.json").read_bytes()
    other = generator.generate(generator.PUBLIC_SUITE_ID, public=True, seed=20261006)
    assert other["cases"] != first["cases"]
    assert generator.default_seed(generator.PRIVATE_SUITE_ID) != generator.default_seed(generator.PUBLIC_SUITE_ID)
    source = (TASK_ROOT / "scripts/generate_edge_manifest.py").read_text(encoding="utf-8")
    assert "SystemRandom" not in source and "import secrets" not in source


@pytest.mark.parametrize(
    "mutation",
    ["page_size", "group", "lengths", "width", "pool", "native_only", "suite", "count", "sanitize_count", "extra_field", "coverage"],
)
def test_edge_manifest_mutations_fail_closed(mutation: str) -> None:
    payload = json.loads(json.dumps(PRIVATE))
    case = payload["cases"][0]
    if mutation == "page_size":
        case["page_size"] = 48
    elif mutation == "group":
        case["h_kv"] = 5
    elif mutation == "lengths":
        case["lengths"][0] = case["s_q"] - 1
    elif mutation == "width":
        case["table_width"] = 4 * _needed(case) + 1
    elif mutation == "pool":
        case["pool_pages"] = case["b"] * case["table_width"] - 1
    elif mutation == "native_only":
        case["native"] = False
    elif mutation == "suite":
        payload["suite_id"] = "edge-public-v2"
    elif mutation == "count":
        payload["cases"].pop(0)
    elif mutation == "sanitize_count":
        case["sanitize"] = True
    elif mutation == "extra_field":
        case["logit_scale"] = "1.0"
    elif mutation == "coverage":
        for item in payload["cases"]:
            if item["fill"] == "nan":
                item["fill"] = "randn"
    with pytest.raises(RuntimeError):
        verifier.validate_edge_manifest(payload)
