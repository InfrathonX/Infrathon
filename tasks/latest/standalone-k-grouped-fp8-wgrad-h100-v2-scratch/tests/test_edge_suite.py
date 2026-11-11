"""The edge suites cover the declared support domain, not just the scored grid.

Both frozen instances must be exactly what the seeded generator produces, the
Judge's private instance must be the one the verifier and build expect, and
together the cases must exercise every axis the infrastructure review asked
about: sizes of one and other tiny sizes, sizes that are not multiples of 8,
odd numbers of 128-blocks, non-power-of-two group counts, empty groups and an
empty problem, tile boundaries, misleading hints and tails on the last group.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from conftest import TASK_ROOT, load_module, judge_only_file


runtime = load_module("edge_suite_runtime", "runtime/k_grouped_wgrad_runtime.py")
verifier = load_module("edge_suite_verifier", "images/judge/verify_wgrad.py")
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
    expected_private = sum(count for _name, count, _public in generator.CATEGORY_COUNTS)
    expected_public = sum(public for _name, _count, public in generator.CATEGORY_COUNTS)
    assert len(PRIVATE["cases"]) == expected_private == verifier.EDGE_CASE_COUNT
    assert len(PUBLIC["cases"]) == expected_public
    assert sum(case["sanitize"] for case in PRIVATE["cases"]) == verifier.EDGE_SANITIZE_COUNT
    assert sum(case["sanitize"] for case in PUBLIC["cases"]) >= 2
    categories = {name for name, _count, _public in generator.CATEGORY_COUNTS}
    assert {case["category"] for case in PRIVATE["cases"]} == categories
    assert {case["category"] for case in PUBLIC["cases"]} == {
        generator.PUBLIC_CATEGORY_NAMES.get(name, name) for name in categories
    }


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
    return value % 128 == 0 and (value // 128) % 2 == 1


def test_private_suite_covers_the_review_axes() -> None:
    cases = PRIVATE["cases"]
    dims = [value for case in cases for value in (case["m"], case["n"])]
    for tiny in (1, 3, 7, 9):
        assert tiny in dims, tiny
    assert sum(1 for case in cases if case["m"] % 8 or case["n"] % 8) >= 100
    assert sum(1 for case in cases if case["m"] % 128 or case["n"] % 128) >= 120
    assert sum(_odd_blocks(case["m"]) or _odd_blocks(case["n"]) for case in cases) >= 30
    for boundary in (63, 64, 65, 127, 128, 129, 255, 256, 257):
        assert boundary in dims, boundary
    assert {8191, 8192} <= set(dims)
    assert {1, 2, 3, 127, 128} <= {case["g"] for case in cases}
    assert {g for g in (case["g"] for case in cases) if g & (g - 1)}  # non-powers of two
    assert sum(0 in case["ks"] for case in cases) >= 20
    assert sum(sum(case["ks"]) == 0 for case in cases) >= 3
    assert any(case["ks"] == [49_152] for case in cases)
    assert any(10_752 in case["ks"] and 128 in case["ks"] for case in cases)
    assert sum(any(_odd_blocks(k) and k > 128 for k in case["ks"]) for case in cases) >= 20
    assert any(case["hint"] == 1 and max(case["ks"]) >= 1024 for case in cases)
    assert any(case["hint"] == 1 << 20 and 0 < max(case["ks"]) <= 256 for case in cases)
    assert sum(case["g"] == 1 and (case["m"] % 64 or case["n"] % 64) for case in cases) >= 5
    # The v1-contract subset: 128-multiples, no empty group, 2 <= G <= 32, M >= 512, N >= 1024.
    v1 = [case for case in cases if case["category"] == "v1_contract"]
    assert len(v1) == 30
    for case in v1:
        assert case["m"] % 128 == 0 and case["n"] % 128 == 0 and case["m"] >= 512 and case["n"] >= 1024
        assert 0 not in case["ks"] and 2 <= case["g"] <= 32
    assert sum(_odd_blocks(case["m"]) or _odd_blocks(case["n"]) for case in v1) >= 20
    sanitize = [case for case in cases if case["sanitize"]]
    assert any(0 in case["ks"] for case in sanitize) and any(sum(case["ks"]) == 0 for case in sanitize)
    assert any(case["m"] == 1 and case["n"] == 1 for case in sanitize)


@pytest.mark.parametrize(
    "mutation",
    ["unaligned_k", "wrong_length", "zero_m", "too_many_groups", "hint_zero", "too_large", "suite", "count", "no_empty"],
)
def test_edge_manifest_mutations_fail_closed(mutation: str) -> None:
    payload = json.loads(json.dumps(PRIVATE))
    case = payload["cases"][0]
    if mutation == "unaligned_k":
        case["ks"][0] = 100
    elif mutation == "wrong_length":
        case["ks"].append(128)
    elif mutation == "zero_m":
        case["m"] = 0
    elif mutation == "too_many_groups":
        case["g"] = 129
    elif mutation == "hint_zero":
        case["hint"] = 0
    elif mutation == "too_large":
        case.update(g=2, m=8192, n=8192, ks=[128, 128])
    elif mutation == "suite":
        payload["suite_id"] = "edge-public-v2"
    elif mutation == "count":
        payload["cases"].pop()
    elif mutation == "no_empty":
        for item in payload["cases"]:
            item["ks"] = [value or 128 for value in item["ks"]]
    with pytest.raises(RuntimeError):
        verifier.validate_edge_manifest(payload)


def test_public_edges_cover_the_paths_without_reusing_private_shapes() -> None:
    cases = PUBLIC["cases"]
    assert any(case["g"] > 32 for case in cases)
    assert any(sum(case["ks"]) == 0 for case in cases)
    assert max(max(case["ks"] or [0]) for case in cases) > 32_768
    assert any(generator.PUBLIC_UNBALANCED_K in case["ks"] for case in cases)
    assert sum(case["sanitize"] for case in cases) == 4
    private = generator.generate(generator.PRIVATE_SUITE_ID, public=False)["cases"]
    private_shapes = {(case["g"], case["m"], case["n"]) for case in private}
    assert private_shapes.isdisjoint({(case["g"], case["m"], case["n"]) for case in cases})
