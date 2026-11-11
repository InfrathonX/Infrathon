"""Public and final suites: no shared geometry, shape, length or seed."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from conftest import _load, judge_only_file

runtime = _load("mamba2_ssd_runtime", "runtime/mamba2_ssd_runtime.py")
generator = _load("ssd_final_manifest", "scripts/generate_final_manifest.py")
work = _load("mamba2_ssd_work_runtime", "images/work/benchmark/mamba2_ssd_runtime.py")
WORK_ROOT = Path(__file__).resolve().parents[1] / "images/work"


def shape(case) -> tuple:
    return (case["geometry"], case["batch"], case["seqlen"])


def synthetic_plan() -> dict:
    """A plan that satisfies every family rule; the real plan is Judge-only."""
    shapes = {
        "batch": ("mamba2_2p7b", 2, 1000),
        "serving": ("codestral_7b", 1, 2000),
        "long": ("falcon_h1_34b", 1, 60000),
    }
    judge = []
    for family, count in generator.FAMILY_COUNTS.items():
        for ordinal in range(count):
            if family == "real":
                geometry = "granite_h_micro" if ordinal < 12 else "granite_h_small"
                judge.append({"family": "real", "tier": "real", "geometry": geometry, "b": 1,
                              "L": 16381 if ordinal < 12 else 32749})
            else:
                geometry, batch, seqlen = shapes[family]
                judge.append({"family": family, "tier": "sweep", "geometry": geometry, "b": batch,
                              "L": seqlen + ordinal})
    return {"judge": judge}


RECIPE = json.loads(generator.RECIPE.read_text())


def test_generator_identity_is_the_runtime_identity() -> None:
    assert generator.FAMILY_COUNTS == runtime.FINAL_FAMILY_COUNTS
    assert generator.PUBLIC_SEED_BASE == work.PUBLIC_SEED_BASE
    assert generator.PUBLIC_CASE_COUNT == len(work.PUBLIC_CASES)
    assert generator.SUITE_ID == runtime.FINAL_SUITE_ID
    assert generator.SCHEMA_VERSION == runtime.MANIFEST_SCHEMA_VERSION
    for geometry, source in generator.REAL_SOURCES.items():
        assert RECIPE["sources"][source]["geometry"] == geometry
        assert RECIPE["sources"][source]["consumer"] == "judge"


@pytest.mark.parametrize("trial", range(8))
def test_generated_final_manifest_is_valid_with_private_seeds(trial: int) -> None:
    manifest = generator.generate_manifest(synthetic_plan(), RECIPE, random.Random(trial))
    cases = runtime.validate_manifest_cases(manifest["cases"])
    assert manifest["suite_id"] == runtime.FINAL_SUITE_ID
    public_seeds = {c["seed"] for c in work.public_suite()} | {c["seed"] for c in runtime.edge_fixtures()}
    assert not {c["seed"] for c in cases} & public_seeds
    assert all(0 < c["seed"] <= 0x7FFF_FFFF_FFFF_FFFF for c in cases)
    assert {c["check_trial"] for c in cases if c["family"] != "real"} <= {0, 1, 2}
    layers = [c["layer"] for c in cases if c["family"] == "real"]
    assert layers == generator.real_layer_names(RECIPE, "granite_h_micro") + generator.real_layer_names(
        RECIPE, "bamba_9b_v2"
    )


def test_default_draw_is_reproducible_and_seeds_differ_between_draws() -> None:
    first = generator.generate_manifest(synthetic_plan(), RECIPE)
    assert first == generator.generate_manifest(synthetic_plan(), RECIPE)
    other = generator.generate_manifest(synthetic_plan(), RECIPE, random.Random(99))
    assert [c["seed"] for c in other["cases"]] != [c["seed"] for c in first["cases"]]


def test_derived_trial_seeds_are_distinct() -> None:
    labels = ["warm", "decoy", *(f"trial-{i}" for i in range(7))]
    seeds = [runtime.derive_seed(123456789, label) for label in labels]
    assert len(set(seeds)) == len(seeds)
    assert all(0 <= seed <= 0x7FFF_FFFF_FFFF_FFFF for seed in seeds)
    assert runtime.derive_seed(1, "warm") != runtime.derive_seed(2, "warm")


def test_edge_fixtures_are_unscored_and_cover_the_hard_cases() -> None:
    fixtures = runtime.edge_fixtures()
    assert all(not f["scored"] and f["family"] == "edge" for f in fixtures)
    assert len({f["case_id"] for f in fixtures}) == len(fixtures) == 24
    lengths = {f["seqlen"] for f in fixtures}
    assert 1 in lengths, "no sub-chunk length"
    assert lengths & {63, 65, 129, 257, 513}, "no length straddling a chunk boundary"
    long_tails = [f for f in fixtures if f["seqlen"] > 8192]
    assert all(f["seqlen"] % 64 for f in long_tails)
    assert max(lengths) == 262143
    assert {"slow_decay", "fast_decay", "unit_dt"} <= {f["regime"] for f in fixtures}
    assert any(f["d_mode"] == "zero" for f in fixtures), "no state-space-only fixture"
    assert any(f["bc_mode"] == "zero" for f in fixtures), "no residual-only fixture"
    assert {"nemotron_h_47b", "zamba2_7b", "granite_h_small"} <= {f["geometry"] for f in fixtures}


def test_scored_cases_reject_family_and_domain_violations() -> None:
    base = {"case_id": "x", "tier": "sweep", "regime": "mixed", "seed": 1, "check_trial": 0, "layer": None}
    for raw in (
        {"geometry": "codestral_7b", "family": "serving", "batch": 2, "seqlen": 6400},
        {"geometry": "codestral_7b", "family": "long", "batch": 1, "seqlen": 32768},
        {"geometry": "codestral_7b", "family": "batch", "batch": 8, "seqlen": 32768},
        {"geometry": "mamba2_2p7b", "family": "batch", "batch": 8, "seqlen": 3000, "regime": "slow_decay"},
        {"geometry": "not_a_model", "family": "batch", "batch": 8, "seqlen": 3000},
    ):
        with pytest.raises(ValueError):
            runtime.validate_case({**base, **raw}, scored=True)


def test_manifest_with_a_missing_case_is_rejected() -> None:
    cases = generator.generate_manifest(synthetic_plan(), RECIPE, random.Random(7))["cases"]
    with pytest.raises(ValueError):
        runtime.validate_manifest_cases(cases[:-1])


def test_frozen_instance_follows_the_rules() -> None:
    path = judge_only_file("judge/private/final_cases.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = runtime.validate_manifest_cases(payload["cases"])
    assert payload["suite_id"] == runtime.FINAL_SUITE_ID
    assert sum(c["batch"] == 1 for c in cases) > len(cases) // 2
    assert sum(c["seqlen"] % 64 != 0 for c in cases) > len(cases) // 2
    assert max(c["seqlen"] for c in cases) == 262144


def _geometry(geometries, name) -> tuple:
    g = geometries[name]
    return (g["h"], g["p"], g["n"], g["g"])


def test_public_suite_shares_no_geometry_length_or_seed_with_the_final_suite() -> None:
    judge_geometries = {_geometry(runtime.GEOMETRIES, name) for name in runtime.GEOMETRIES}
    public_geometries = {_geometry(work.GEOMETRIES, name) for name in work.GEOMETRIES}
    assert not judge_geometries & public_geometries
    public_cases = work.public_suite() + work.edge_fixtures()
    edge_lengths = {f["seqlen"] for f in runtime.edge_fixtures()}
    # L = 1 is the one shared length: it is the bottom of the support domain.
    assert {c["seqlen"] for c in public_cases} & edge_lengths == {1}
    final_seeds = {c["seed"] for c in runtime.edge_fixtures()}
    assert not final_seeds & {c["seed"] for c in public_cases}
    assert not {c["case_id"] for c in runtime.edge_fixtures()} & {c["case_id"] for c in public_cases}
    path = judge_only_file("judge/private/final_cases.json")
    frozen = json.loads(path.read_text(encoding="utf-8"))["cases"]
    assert not {c["seqlen"] for c in frozen} & {c["seqlen"] for c in work.public_suite()}
    assert not {c["seed"] for c in frozen} & {c["seed"] for c in public_cases}


def test_public_suite_covers_every_family_inside_the_domain() -> None:
    cases = work.public_suite()
    assert len(cases) == work.FINAL_CASE_COUNT == 24
    counts = {family: sum(c["family"] == family for c in cases) for family in work.SCORE_FAMILIES}
    assert counts == work.FINAL_FAMILY_COUNTS
    for geometry in work.GEOMETRIES.values():
        assert 1 <= geometry["h"] <= 256 and geometry["h"] % geometry["g"] == 0
        assert geometry["p"] % 16 == 0 and 16 <= geometry["p"] <= 256
        assert geometry["n"] in (16, 32, 64, 128, 256)
    for case in (*cases, *work.edge_fixtures()):
        assert 1 <= case["batch"] <= 16


def test_public_runtime_is_generated_from_the_trusted_runtime() -> None:
    maker = _load("make_public_runtime", "scripts/make_public_runtime.py")
    assert (WORK_ROOT / "benchmark/mamba2_ssd_runtime.py").read_text(encoding="utf-8") == maker.build()


def test_agent_visible_files_say_nothing_about_the_scored_suite() -> None:
    tokens = (
        "mamba2_2p7b", "codestral", "falcon_h1", "Falcon", "Codestral", "Mamba-2-2.7B", "scored_grid",
        "SCORED_", "tilelang-ssd-final", "eighteen", "73 GiB", "73 *", "decoy", "Decoy",
        "PyroMind", "HAMi", "terminal Judge", "deadline Judge", "same shapes", "Nemotron",
        "nemotron", "Granite", "granite", "Bamba", "Zamba", "zamba", "Mooncake",
        "16381", "32749", "262144", "256000",
        # The hidden suite's distribution: geometry ranges, batch mix, length range,
        # captured-layer statistics and case count are not disclosed.
        "mostly batch 1", "most cases are batch", "260,000", "60,000", "about 140",
        "16 to 256 heads", "width 64 or 128", "1 to 8 groups", "not a multiple of 64",
        "spans many orders of magnitude", "serving at batch 1", "138 ", "Why this kernel matters",
        "hundreds of thousands of tokens",
    )
    for path in (*WORK_ROOT.rglob("*.md"), *WORK_ROOT.rglob("*.py"), *WORK_ROOT.rglob("*.sh")):
        text = path.read_text(encoding="utf-8")
        for token in tokens:
            assert token not in text, (path.name, token)
    query = json.loads(
        (WORK_ROOT.parents[1] / "tasks/tilelang_mamba2_ssd_forward_h100_seed_v2.json").read_text(encoding="utf-8")
    )["work"]["agent_query"]
    for token in tokens:
        assert token not in query, token


def test_public_suite_rows_pass_the_driver_validation() -> None:
    """run_suite revalidates its cases; the public runner hands it public_suite() rows."""
    cases = work.validate_manifest_cases(work.public_suite(), require_full_suite=True)
    assert len(cases) == len(work.PUBLIC_CASES)
    with pytest.raises(ValueError):
        work.validate_manifest_cases([dict(work.public_suite()[0], unexpected=1)], require_full_suite=False)
