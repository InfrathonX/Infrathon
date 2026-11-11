"""The edge suites cover the declared support domain, not just the scored cases.

Both frozen instances must be exactly what the seeded generator produces, the
Judge's private instance must be the one the verifier and build expect, and
together the cases must exercise every axis the infrastructure review asked
about: sizes of one, odd numbers of 128-blocks, non-power-of-two group counts,
empty / single-row / full groups, tile boundaries, misleading hints and tight
capacities, plus the v4 axes: prefix-sum row offsets past 2^31 bytes, M_max past
4096, single groups, one hot group, all-empty calls and extreme hints.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from conftest import TASK_ROOT, load_module, judge_only_file


runtime = load_module("edge_suite_runtime", "runtime/grouped_fp8_runtime.py")
verifier = load_module("edge_suite_verifier", "images/judge/verify_grouped.py")
generator = load_module("edge_suite_generator", "scripts/generate_edge_manifest.py")
PRIVATE = json.loads(judge_only_file("judge/private/edge_cases.json").read_text(encoding="utf-8"))
PUBLIC = json.loads(
    (TASK_ROOT / "images/work/benchmark/public_edge_cases.json").read_text(encoding="utf-8")
)


def test_frozen_instances_are_the_seeded_generator_output() -> None:
    completed = subprocess.run(
        [sys.executable, "-B", str(TASK_ROOT / "scripts/generate_edge_manifest.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_identities_counts_and_categories() -> None:
    assert PRIVATE["suite_id"] == generator.PRIVATE_SUITE_ID == verifier.EDGE_SUITE_ID
    assert PUBLIC["suite_id"] == generator.PUBLIC_SUITE_ID
    def layouts(name: str) -> int:
        return 1 if name in generator.PSUM_ONLY else 2
    expected_private = sum(layouts(name) * count for name, count, _public in generator.CATEGORY_COUNTS)
    expected_public = sum(layouts(name) * public for name, _count, public in generator.CATEGORY_COUNTS)
    assert len(PRIVATE["cases"]) == expected_private == verifier.EDGE_CASE_COUNT
    assert len(PUBLIC["cases"]) == expected_public
    assert sum(case["sanitize"] for case in PRIVATE["cases"]) == verifier.EDGE_SANITIZE_COUNT
    assert sum(case["sanitize"] for case in PUBLIC["cases"]) >= 2
    categories = {name for name, _count, _public in generator.CATEGORY_COUNTS}
    assert {case["category"] for case in PRIVATE["cases"]} == categories
    assert {case["category"] for case in PUBLIC["cases"]} == {generator.PUBLIC_LABEL, generator.PUBLIC_SANITIZE_LABEL}
    for payload in (PRIVATE, PUBLIC):
        assert {case["layout"] for case in payload["cases"]} == {"masked", "psum"}


def test_every_case_is_inside_the_declared_domain() -> None:
    for case in (*PRIVATE["cases"], *PUBLIC["cases"]):
        runtime.validate_edge_case(case)
    assert len(verifier.validate_edge_manifest(PRIVATE)) == verifier.EDGE_CASE_COUNT


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


def _odd_blocks(value: int) -> bool:
    return (value // 128) % 2 == 1


def test_private_suite_covers_the_review_axes() -> None:
    cases = PRIVATE["cases"]
    groups = [(case, size) for case in cases for size in case["m_sizes"]]
    assert {1, 64} <= {case["g"] for case in cases}
    assert {g for g in (case["g"] for case in cases) if g & (g - 1)}  # non-powers of two
    assert 128 in {case["n"] for case in cases} and 128 in {case["k"] for case in cases}
    assert 8192 in {case["n"] for case in cases} and 8192 in {case["k"] for case in cases}
    assert sum(_odd_blocks(case["n"]) for case in cases) >= 40
    assert sum(_odd_blocks(case["k"]) for case in cases) >= 40
    assert {128, 4096} <= {case["m_max"] for case in cases}
    assert sum(size == 0 for _case, size in groups) >= 50
    assert sum(size == 1 for _case, size in groups) >= 50
    assert sum(size == case["m_max"] for case, size in groups) >= 50
    for boundary in (63, 64, 65, 127, 128, 129, 255, 256, 257):
        assert any(size == boundary for _case, size in groups), boundary
    assert any(case["hint"] == 1 and max(case["m_sizes"]) > 1 for case in cases)
    assert any(case["hint"] >= 4 * max(1, max(case["m_sizes"])) for case in cases)
    assert any(case["m_sizes"][-1] == case["m_max"] and case["g"] == 1 for case in cases)
    for layout in ("masked", "psum"):
        assert sum(case["layout"] == layout and case["sanitize"] for case in cases) >= 2


def _psum_last_start(case) -> int:
    return runtime.group_starts(case["m_sizes"])[-1]


def test_v4_axes_are_covered_in_both_instances() -> None:
    private, public = PRIVATE["cases"], PUBLIC["cases"]
    for cases in (private, public):
        # A prefix-sum row offset past 2^31 bytes of A: 64-bit addressing is required.
        assert any(case["layout"] == "psum" and _psum_last_start(case) * case["k"] > 2**31 for case in cases)
        assert any(case["layout"] == "psum" and case["m_max"] > 4096 for case in cases)
        assert any(case["g"] == 1 for case in cases)
        assert any(sum(case["m_sizes"]) == 0 for case in cases)
    assert {case["m_max"] for case in private} >= {131072}
    assert any(case["hint"] == runtime.MAX_HINT for case in private)
    assert any(
        case["g"] >= 8 and max(case["m_sizes"]) > 100 * max(1, sorted(case["m_sizes"])[-2]) for case in private
    )
    assert not any(case["sanitize"] and case["m_max"] > 4096 for case in (*private, *public))


@pytest.mark.parametrize(
    "mutation",
    ["oversized_group", "wrong_length", "unaligned_n", "zero_groups", "hint_zero", "suite", "count"],
)
def test_edge_manifest_mutations_fail_closed(mutation: str) -> None:
    payload = json.loads(json.dumps(PRIVATE))
    case = payload["cases"][0]
    if mutation == "oversized_group":
        case["m_sizes"][0] = case["m_max"] + 1
    elif mutation == "wrong_length":
        case["m_sizes"].append(1)
    elif mutation == "unaligned_n":
        case["n"] = 1000
    elif mutation == "zero_groups":
        case["g"] = 0
    elif mutation == "hint_zero":
        case["hint"] = 0
    elif mutation == "suite":
        payload["suite_id"] = "edge-public-v4"
    elif mutation == "count":
        payload["cases"].pop()
    with pytest.raises(RuntimeError):
        verifier.validate_edge_manifest(payload)


def test_public_edges_cover_the_paths_without_reusing_private_shapes() -> None:
    cases = PUBLIC["cases"]
    assert any(case["g"] == 64 for case in cases)
    assert any(case["m_max"] == 4096 for case in cases)
    assert any(0 in case["m_sizes"] for case in cases)
    assert sum(case["sanitize"] for case in cases) == 4
    private = generator.generate(generator.PRIVATE_SUITE_ID, public=False)["cases"]
    private_shapes = {(case["g"], case["n"], case["k"]) for case in private}
    assert private_shapes.isdisjoint({(case["g"], case["n"], case["k"]) for case in cases})
    final = json.loads(judge_only_file("judge/private/final_cases.json").read_text(encoding="utf-8"))
    scored_nk = {(case["n"], case["k"]) for case in final["cases"]}
    assert scored_nk.isdisjoint({(case["n"], case["k"]) for case in cases})
