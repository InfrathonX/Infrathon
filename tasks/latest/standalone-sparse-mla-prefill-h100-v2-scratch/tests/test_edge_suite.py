"""The edge suites cover the declared support domain, not just the scored shapes.

Both frozen instances must be exactly what the seeded generator produces.  The
private instance must hold the upstream FlashMLA ``ba89a34`` correctness,
with-features and corner lists inside the three DeepSeek geometries (283
cases, the s_q = 70000 corner included), every anchor shape with the edge
semantics and with zero-bias precision probes, the odd-size sweep over the four
families, the v1 fixtures, causal indices in every family, keys at each
family's context limit and tensor-parallel head counts (470 in all); the
sanitizer subset (17 private, 7 public) runs the sanitizer-only large-s_q case
first.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter

import pytest

from conftest import TASK_ROOT, load_module, judge_only_file


runtime = load_module("edge_suite_runtime", "runtime/sparse_mla_prefill_runtime.py")
verifier = load_module("edge_suite_verifier", "images/judge/verify_sparse_mla.py")
generator = load_module("edge_suite_generator", "scripts/generate_edge_manifest.py")
public_runtime = load_module("edge_suite_public_runtime", "images/work/benchmark/sparse_mla_prefill_runtime.py")
PRIVATE = json.loads(judge_only_file("judge/private/edge_cases.json").read_text(encoding="utf-8"))
PUBLIC = json.loads(
    (TASK_ROOT / "images/work/benchmark/public_edge_cases.json").read_text(encoding="utf-8")
)
GEOMETRY_ORDER = ("v4_pro", "v4_flash", "v32")  # upstream loop order: (512,128), (512,64), (576,128)


def _shape(case: dict) -> tuple:
    return (case["family"], case["s_q"], case["s_kv"], case["topk"])


def _category(payload: dict, name: str) -> list[dict]:
    return [case for case in payload["cases"] if case["category"] == name]


def test_frozen_instances_are_the_seeded_generator_output() -> None:
    completed = subprocess.run(
        [sys.executable, "-B", str(TASK_ROOT / "scripts/generate_edge_manifest.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "edge-holdout-v3: 470 cases, 469 native, 17 sanitize (1 sanitizer-only)" in completed.stdout
    assert "edge-public-v3: 73 cases, 73 native, 7 sanitize (0 sanitizer-only)" in completed.stdout


def test_identities_counts_and_categories() -> None:
    assert PRIVATE["schema_version"] == PUBLIC["schema_version"] == 3
    assert PRIVATE["suite_id"] == generator.PRIVATE_SUITE_ID == verifier.EDGE_SUITE_ID
    assert PRIVATE["suite_id"] == "edge-holdout-v3"
    assert PUBLIC["suite_id"] == generator.PUBLIC_SUITE_ID == "edge-public-v3"
    assert len(PRIVATE["cases"]) == verifier.EDGE_CASE_COUNT == generator.PRIVATE_CASE_COUNT == 470
    assert len(PUBLIC["cases"]) == generator.PUBLIC_CASE_COUNT == 73
    private_counts = Counter(case["category"] for case in PRIVATE["cases"])
    public_counts = Counter(case["category"] for case in PUBLIC["cases"])
    assert dict(private_counts) == verifier.EDGE_CATEGORY_COUNTS == {
        name: count for name, count in generator.CATEGORY_COUNTS
    }
    assert dict(public_counts) == dict(generator.PUBLIC_CATEGORY_COUNTS)
    assert sum(case["sanitize"] for case in PRIVATE["cases"]) == verifier.EDGE_SANITIZE_COUNT == 17
    assert sum(case["sanitize"] for case in PUBLIC["cases"]) == generator.PUBLIC_SANITIZE_COUNT == 7
    assert sum(not case["native"] for case in PRIVATE["cases"]) == verifier.EDGE_SANITIZE_ONLY_COUNT == 1
    assert all(case["native"] for case in PUBLIC["cases"])
    # The plan's headline numbers: 50 private cases at s_q = 1, about 420 at
    # an s_q that is not a multiple of 8.
    assert sum(case["s_q"] == 1 for case in PRIVATE["cases"]) == 50
    assert sum(case["s_q"] % 8 != 0 for case in PRIVATE["cases"]) >= 400


def test_every_case_is_inside_the_declared_domain() -> None:
    for case in PRIVATE["cases"]:
        runtime.validate_edge_case(case)
        verifier.validate_edge_case(case)
    for case in PUBLIC["cases"]:
        public_runtime.validate_edge_case(case)
    verifier.validate_edge_manifest(PRIVATE)


def test_upstream_lists_are_the_pinned_test_file_restricted_to_the_deployed_geometries() -> None:
    # tests/test_flash_mla_sparse_prefill.py at ba89a34.
    pairs10 = [(128, 128), (256, 256), (512, 512), (592, 128), (1840, 256), (1592, 384), (1521, 512),
               (95, 128), (153, 256), (114, 384)]
    correctness = [
        (family, s_q, s_kv, topk)
        for family in GEOMETRY_ORDER for s_kv, topk in pairs10 for s_q in (1, 62, 213)
    ]
    features = [
        (family, s_q, s_kv, topk, sink, length)
        for family in GEOMETRY_ORDER for s_kv, topk in pairs10[3:] for s_q in (62, 213)
        for sink in (False, True) for length in (False, True)
    ]
    all_invalid = [
        (family, s_q, s_kv, topk)
        for family in GEOMETRY_ORDER
        for s_q, s_kv, topk in ((1, 128, 128), (1, 256, 256), (1234, 4321, 2048), (4096, 2048, 2048))
    ]
    empty = [
        (family, s_q, s_kv, topk)
        for family in GEOMETRY_ORDER for s_kv, topk in ((32, 2048), (64, 2048)) for s_q in (1, 1024)
    ]
    for payload in (PRIVATE,):
        assert [_shape(c) for c in _category(payload, "upstream_correctness")] == correctness
        assert [
            (*_shape(c), c["sink_mode"] == "upstream", c["length_mode"] == "upstream")
            for c in _category(payload, "upstream_features")
        ] == features
        assert [_shape(c) for c in _category(payload, "upstream_all_invalid")] == all_invalid
        assert [_shape(c) for c in _category(payload, "upstream_empty_blocks")] == empty
        for case in _category(payload, "upstream_correctness"):
            assert (case["sink_mode"], case["length_mode"], case["invalid_rows"]) == ("none", "none", "none")
        for case in _category(payload, "upstream_all_invalid"):
            assert case["invalid_rows"] == "upstream"
            assert (case["sink_mode"], case["length_mode"]) == ("upstream", "upstream")
        for case in _category(payload, "upstream_empty_blocks"):
            assert (case["sink_mode"], case["length_mode"]) == ("upstream", "upstream")
        for case in payload["cases"]:
            if case["category"].startswith("upstream_"):
                assert case["sm_scale"] == 0.5 and case["bias"] == "upstream"
                assert case["index_mode"] == "upstream"
    # Same upstream shapes publicly and privately, different seeds.
    private_seeds = {case["seed"] for case in PRIVATE["cases"]}
    assert private_seeds.isdisjoint({case["seed"] for case in PUBLIC["cases"]})
    large = _category(PRIVATE, "upstream_large_s_q")
    assert [_shape(c) for c in large] == [("v4_flash", 70000, 256, 256)]
    assert large[0]["sanitize"] is True and large[0]["native"] is False
    assert not _category(PUBLIC, "upstream_large_s_q")


def test_the_large_s_q_corner_needs_64_bit_offsets_and_grid_axis_0() -> None:
    (large,) = _category(PRIVATE, "upstream_large_s_q")
    d_qk, h_q = runtime.FAMILY_GEOMETRY[large["family"]]
    assert large["s_q"] > 65535  # Triton grid axes 1 and 2 stop at 65535
    assert large["s_q"] * h_q * d_qk > 2**31  # q offsets overflow int32
    assert large["s_q"] * h_q * runtime.D_V > 2**31  # so do out offsets
    assert large["s_q"] * h_q * d_qk <= runtime.MAX_Q_ELEMENTS
    assert large["s_q"] * large["topk"] <= runtime.MAX_TOPK_POSITIONS


def test_scored_shapes_carry_the_edge_semantics_and_the_precision_probes() -> None:
    grid = list(runtime.anchor_grid())
    for payload, precision_shapes in ((PRIVATE, grid),):
        semantics = _category(payload, "scored_semantics")
        assert [(c["family"], c["s_kv"]) for c in semantics] == grid
        for case in semantics:
            d_qk, _h_q = runtime.FAMILY_GEOMETRY[case["family"]]
            assert case["s_q"] == runtime.ANCHOR_S_Q and case["topk"] == runtime.FAMILY_TOPK[case["family"]]
            assert "h_q" not in case
            assert case["sm_scale"] == pytest.approx(d_qk ** -0.55)
            assert (case["index_mode"], case["invalid_rows"], case["sink_mode"], case["length_mode"]) == (
                "sprinkle", "sparse", "extreme", "edge"
            )
            assert case["duplicates"] and case["scramble"]
        precision = _category(payload, "scored_precision")
        assert [(c["family"], c["s_kv"]) for c in precision] == precision_shapes
        for case in precision:
            assert case["bias"] == "zero" and case["sm_scale"] == runtime.ANCHOR_SM_SCALE
            assert {key: case[key] for key in runtime.SCORED_CONTENT if key != "bias"} == {
                key: value for key, value in runtime.SCORED_CONTENT.items() if key != "bias"
            }


def test_odd_sizes_cover_every_topk_regime_and_odd_s_kv() -> None:
    odd = _category(PRIVATE, "odd_sizes")
    assert len(odd) == 112 == 28 * len(runtime.FAMILIES)
    assert {case["s_q"] for case in odd} == set(generator.PRIVATE_ODD_S_Q)
    assert {case["topk"] for case in odd} == set(generator.ODD_TOPK)
    assert {case["family"] for case in odd} == set(runtime.FAMILIES)
    assert {1, 7, 95, 129, 131071} <= {case["s_kv"] for case in odd}
    assert any(case["s_kv"] < case["topk"] for case in odd)
    assert {case["length_mode"] for case in odd} == {"none", "edge"}
    assert {case["bias"] for case in odd} == {"upstream", "zero"}
    public_odd = _category(PUBLIC, "odd_sizes")
    assert {case["s_q"] for case in public_odd} == set(generator.PUBLIC_ODD_S_Q)
    assert {case["s_q"] for case in public_odd}.isdisjoint({case["s_q"] for case in odd})


def test_v1_fixtures_keep_their_v1_shapes() -> None:
    # v1 is archived (2026-10-07) under tasks/archive/.
    v1 = load_module("edge_suite_v1_runtime", "../../archive/standalone-sparse-mla-prefill-h100-v1-scratch/runtime/sparse_mla_prefill_runtime.py") \
        if (TASK_ROOT.parents[1] / "archive/standalone-sparse-mla-prefill-h100-v1-scratch").is_dir() else None
    fixtures = _category(PRIVATE, "v1_fixture")
    assert [case["case_id"].split("-", 2)[2] for case in fixtures] == [f["name"] for f in generator.V1_FIXTURES]
    if v1 is None:
        pytest.skip("the v1 task directory is not present")
    rename = {"v32": "v32", "m1_64": "v4_flash", "m1_128": "v4_pro"}
    assert [(rename[c["family"]], c["s_q"], c["s_kv"], c["topk"]) for c in v1.EDGE_CASES] == [
        _shape(case) for case in fixtures
    ]
    for old, new in zip(v1.EDGE_CASES, fixtures):
        assert new["sm_scale"] == pytest.approx(old["sm_scale"])
        assert new["invalid_rows"] == old["invalid_rows"]
        assert new["scramble"] == old["scramble"] and new["duplicates"] == old["duplicates"]
        assert new["index_mode"] == ("valid_pad" if old["all_valid"] else "upstream")
        assert new["sink_mode"] == (old["sink_mode"] if old["sink"] else "none")
        expected_length = "none" if not old["length"] else ("edge" if old["edge_lengths"] else "half")
        assert new["length_mode"] == expected_length


def test_sanitizer_subset_is_the_planned_one_and_runs_the_large_case_first() -> None:
    order = runtime.sanitize_order(PRIVATE["cases"])
    assert [case["case_id"] for case in order] == [case["case_id"] for case in verifier.sanitize_order(PRIVATE["cases"])]
    assert order[0]["category"] == "upstream_large_s_q"
    flagged = Counter(case["category"] for case in order)
    assert flagged == {
        "upstream_large_s_q": 1, "upstream_correctness": 2, "upstream_features": 1,
        "upstream_all_invalid": 2, "upstream_empty_blocks": 2, "scored_semantics": 3,
        "odd_sizes": 2, "v1_fixture": 1, "causal": 1, "large_kv": 1, "tp_heads": 1,
    }
    shapes = {_shape(case) for case in order}
    assert {("v32", 7, 1, case["topk"]) for case in order if case["s_q"] == 7} <= shapes
    assert any(case["s_q"] == 129 and case["s_kv"] == 131071 for case in order)
    # An anchor shape of every DeepSeek family runs under memcheck.
    assert {case["family"] for case in order if case["category"] == "scored_semantics"} == {
        "v32", "v4_flash", "v4_pro"
    }
    assert all(case["s_kv"] == 8192 for case in order if case["category"] == "scored_semantics")
    assert any(case["case_id"].endswith("edge-wide-index") for case in order)
    assert any(case["category"] == "causal" and case["family"] == "glm5" for case in order)
    assert any(case["s_kv"] >= runtime.MAX_S_KV - 1 for case in order)
    assert any(case.get("h_q") == 16 for case in order)
    public_order = runtime.sanitize_order(PUBLIC["cases"])
    assert len(public_order) == 7 and all(case["native"] for case in public_order)


def test_serving_semantics_reach_every_family() -> None:
    causal = _category(PRIVATE, "causal")
    assert len(causal) == 20 and {case["family"] for case in causal} == set(runtime.FAMILIES)
    for case in causal:
        assert case["index_mode"] == "causal" and case["topk"] == runtime.FAMILY_TOPK[case["family"]]
        assert case["sm_scale"] == runtime.FAMILY_SM_SCALE[case["family"]]
        assert case["sink_mode"] in ("none", "finite") and case["s_q"] <= case["s_kv"]
    # One-row and shorter-than-topk first chunks, and a short chunk deep in a long context.
    assert {case["s_q"] for case in causal if case["s_q"] == case["s_kv"]} >= {1, 63}
    assert any(case["s_kv"] > 131072 for case in causal)
    large = _category(PRIVATE, "large_kv")
    assert sorted((case["family"], case["s_kv"]) for case in large) == sorted(
        (family, runtime.FAMILY_MAX_CONTEXT[family] - 1) for family in runtime.FAMILIES
    )
    tp = _category(PRIVATE, "tp_heads")
    assert {case["family"] for case in tp} == set(runtime.FAMILIES)
    for case in tp:
        heads = runtime.FAMILY_GEOMETRY[case["family"]][1]
        assert case["h_q"] % runtime.H_Q_MULTIPLE == 0 and runtime.MIN_H_Q <= case["h_q"] < heads
    assert {case["h_q"] for case in tp} >= {16, 32}
    assert {case["category"] for case in PUBLIC["cases"]} >= {"causal", "large_kv", "tp_heads"}


def test_generator_is_deterministic_and_seed_sensitive() -> None:
    first = generator.generate(generator.PRIVATE_SUITE_ID, public=False)
    second = generator.generate(generator.PRIVATE_SUITE_ID, public=False)
    other = generator.generate(generator.PRIVATE_SUITE_ID, public=False, seed=1)
    assert first == second
    assert [c["seed"] for c in first["cases"]] != [c["seed"] for c in other["cases"]]
    assert [_shape(c) for c in first["cases"]] == [_shape(c) for c in other["cases"]]
    assert generator.encode(first) == judge_only_file("judge/private/edge_cases.json").read_bytes()


def test_public_instance_shares_no_shape_with_the_private_one() -> None:
    private = {_shape(case) for case in PRIVATE["cases"]}
    public = {_shape(case) for case in PUBLIC["cases"]}
    assert not private & public
    scored = {(family, runtime.ANCHOR_S_Q, s_kv, runtime.FAMILY_TOPK[family])
              for family, s_kv in runtime.anchor_grid()}
    assert not scored & public
    assert not any(category.startswith(("upstream_", "scored_", "v1_"))
                   for category in {case["category"] for case in PUBLIC["cases"]})
