"""Partition honesty: coverage on both tiers, disjointness of the holdout."""

from __future__ import annotations

import json

from conftest import TASK_ROOT, _load, judge_only_file

runtime = _load("kv_runtime_p", "runtime/kv_transfer_runtime.py")
verifier = _load("kv_verifier_p", "images/judge/verify_kv_transfer.py")

PUBLIC = runtime.load_case_table(TASK_ROOT / "images/judge/public_cases.json")["cases"]


def _final():
    return runtime.load_manifest(judge_only_file("judge/private/final_cases.json"))["cases"]


def _coverage(cases) -> dict[str, set]:
    families, layouts, directions, heads = set(), set(), set(), set()
    for case in cases:
        families.add(case["family"])
        for wave in case["waves"]:
            layouts.add(wave["layout"])
            directions.add((case["arm"], wave["direction"]))
            heads.add(wave["hs"])
    return {"families": families, "layouts": layouts, "directions": directions, "heads": heads}


def test_public_and_final_cover_everything() -> None:
    for suite in (PUBLIC, _final()):
        cover = _coverage(suite)
        assert cover["families"] == set(runtime.FAMILIES)
        assert cover["layouts"] == {"split", "fused"}
        assert cover["heads"] == set(runtime.HEAD_SIZES)
        for arm in ("device", "host", "overlap", "duplex"):
            assert {(arm, "store"), (arm, "retrieve")} <= cover["directions"], arm


def test_public_and_final_share_no_shape_seed_or_id() -> None:
    final = _final()
    def shapes(cases):
        return {verifier.shape_identity(w) for c in cases for w in c["waves"]}

    assert not shapes(PUBLIC) & shapes(final)
    assert not {c["seed"] for c in PUBLIC} & {c["seed"] for c in final}
    assert not {c["case_id"] for c in PUBLIC} & {c["case_id"] for c in final}


def test_final_manifest_passes_the_verifier() -> None:
    payload = json.loads(judge_only_file("judge/private/final_cases.json").read_text())
    excluded_shapes = frozenset(verifier.shape_identity(w) for c in PUBLIC for w in c["waves"])
    excluded_seeds = frozenset(c["seed"] for c in PUBLIC)
    cases = verifier.validate_final_manifest(payload, excluded_shapes, excluded_seeds)
    assert len(cases) == verifier.EXPECTED_FINAL_COUNT


def test_every_case_sits_above_the_timer_floor() -> None:
    """Every case moves at least 24 MiB per trial, well above CUDA-event resolution."""
    for case in (*PUBLIC, *_final()):
        assert runtime.case_logical_bytes(case) >= 24 * 1024 * 1024, case["case_id"]


def test_strata_cover_both_tiers_in_every_family() -> None:
    final = _final()
    for family in runtime.FAMILIES:
        tiers = {c["tier"] for c in final if c["family"] == family}
        strata = {c["stratum"] for c in final if c["family"] == family}
        assert tiers == {"paper", "other"}, family
        assert len(strata) >= 40, (family, len(strata))


def test_cases_fit_the_judge() -> None:
    for case in (*PUBLIC, *_final()):
        assert runtime.pool_bytes(case["waves"][0]) <= runtime.MAX_POOL_BYTES
        assert runtime.case_payload_bytes(case) <= 2.5 * 1024**3 + 1, case["case_id"]


def test_public_is_smaller_than_the_judge() -> None:
    public = sum(runtime.case_logical_bytes(c) for c in PUBLIC)
    final = sum(runtime.case_logical_bytes(c) for c in _final())
    assert public < final
    assert max(runtime.pool_bytes(c["waves"][0]) for c in PUBLIC) < 13e9
    assert max(runtime.pool_bytes(c["waves"][0]) for c in _final()) > 30e9


def test_block_maps_carry_engine_run_structure() -> None:
    """Replayed maps have runs of adjacent blocks; a uniform permutation has almost none."""
    adjacent = total = 0
    for case in _final():
        for wave in case["waves"]:
            ids = wave["ids"]
            total += len(ids) - 1
            adjacent += sum(abs(b - a) == 1 for a, b in zip(ids, ids[1:]))
    assert adjacent / total > 0.5
