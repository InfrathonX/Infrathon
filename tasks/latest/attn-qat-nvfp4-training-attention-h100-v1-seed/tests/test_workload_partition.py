from __future__ import annotations

import random

import pytest
from conftest import load_module

runtime = load_module("attn_qat_runtime_partition", "runtime/attn_qat_runtime.py")
verifier = load_module("attn_qat_verifier_partition", "images/judge/verify_attn_qat.py")
generator = load_module("attn_qat_manifest_partition", "scripts/generate_final_manifest.py")
public = load_module("attn_qat_public_partition", "images/work/benchmark/public_runner.py")


def manifest(seed: int = 7) -> dict:
    return generator.generate_manifest(random.Random(seed))


def row(**changes) -> dict:
    base = {
        "case_id": "a", "family": "kandinsky5", "tier": "recipe", "b": 1, "h": 32, "h_kv": 32,
        "d": 64, "s_q": 47616, "s_kv": 47616, "causal": False, "seed": 1, "gated": True,
        "layer": None,
    }
    return {**base, **changes}


def llm_row(**changes) -> dict:
    base = {
        "case_id": "l", "family": "llm_pretrain", "tier": "sweep", "b": 4, "h": 32, "h_kv": 8,
        "d": 128, "s_q": 4096, "s_kv": 4096, "causal": True, "seed": 1, "gated": True,
        "layer": None,
    }
    return {**base, **changes}


def test_suite_shape() -> None:
    assert runtime.FAMILIES == (
        "wan13b", "wan14b", "ltx2", "kandinsky5", "cogvideox", "hunyuan", "real_video",
        "llm_pretrain", "llm_long", "llm_sft", "llm_d256", "real_llm",
    )
    assert runtime.FAMILIES == verifier.FAMILIES
    assert runtime.FAMILY_DOMAIN == verifier.FAMILY_DOMAIN
    assert runtime.ENTRYPOINTS == ("attn_qat_fwd", "attn_qat_bwd")
    rows = generator.grid_cases()
    counts: dict = {}
    for item in rows:
        counts.setdefault(item["family"], {}).setdefault(item["tier"], 0)
        counts[item["family"]][item["tier"]] += 1
    assert counts == verifier.EXPECTED_TIER_COUNTS
    assert len(rows) == verifier.EXPECTED_CASE_COUNT == generator.EXPECTED_CASE_COUNT == 287
    domains = {runtime.FAMILY_DOMAIN[r["family"]] for r in rows}
    assert domains == {"video", "llm"}
    recipe_rows = {
        (r["family"], r["h"], r["s_q"], r["s_kv"])
        for r in rows if r["tier"] == "recipe" and runtime.FAMILY_DOMAIN[r["family"]] == "video"
    }
    assert recipe_rows == generator.VIDEO_RECIPES
    llm_recipes = [r for r in rows if r["tier"] == "recipe" and r["family"] == "llm_pretrain"]
    assert [(r["b"], r["h"], r["h_kv"], r["s_q"]) for r in llm_recipes] == [(1, 40, 8, 8192)]


def test_worker_accepts_exactly_the_runtime_case_schema() -> None:
    # The worker re-checks every case's keys before importing the runtime; a stale set made
    # the first end-to-end v3 evaluation fail with "worker failed" before any case ran.
    worker = load_module("attn_qat_worker_partition", "images/judge/candidate_worker.py")
    assert worker.CASE_KEYS == set(runtime.CASE_KEYS) == verifier.CASE_KEYS
    assert worker.EXPECTED_CASE_COUNT == verifier.EXPECTED_CASE_COUNT
    assert worker.SUITE_IDS == {verifier.FINAL_SUITE_ID}
    payload = manifest()
    assert all(set(case) == worker.CASE_KEYS for case in payload["cases"])


def test_lengths_sit_on_their_lattices() -> None:
    assert (runtime.LATTICE_WAN, runtime.LATTICE_WAN_720P, runtime.LATTICE_LTX2, runtime.LATTICE_KANDINSKY5) == (
        1560, 3600, 390, 1536,
    )
    assert (runtime.LATTICE_COGVIDEOX, runtime.COGVIDEOX_TEXT_TOKENS) == (1350, 226)
    assert (runtime.LATTICE_HUNYUAN, runtime.HUNYUAN_TEXT_TOKENS) == (2040, 256)
    for item in generator.grid_cases():
        if runtime.FAMILY_DOMAIN[item["family"]] == "video":
            assert runtime._on_lattice(item["family"], item["s_q"])
            assert not item["causal"] and item["h_kv"] == item["h"] and item["b"] == 1
        else:
            assert item["s_q"] % runtime.LLM_LENGTH_QUANTUM == 0
            assert item["causal"] and item["s_kv"] == item["s_q"] and item["h"] % item["h_kv"] == 0


def test_public_cases_are_off_the_scored_identities() -> None:
    rows = generator.grid_cases()
    scored_geometry = {(r["h"], r["h_kv"], r["d"]) for r in rows}
    scored_lengths = {r["s_q"] for r in rows}
    for _family, _b, h, h_kv, d, s_q, _s_kv, _causal in public.GEOMETRY.values():
        assert (h, h_kv, d) not in scored_geometry
        assert (h, h_kv, d) in runtime.PUBLIC_GEOMETRIES
        assert s_q not in scored_lengths
        assert s_q in runtime.PUBLIC_LENGTHS
        assert s_q & (s_q - 1) != 0
    assert set(public.GEOMETRY) == set(load_module(
        "attn_qat_work_runtime_partition", "images/work/benchmark/attn_qat_runtime.py"
    ).PUBLIC_GRID)


def test_scale_separates_public_from_hidden() -> None:
    rows = generator.grid_cases()
    assert max(r["s_q"] for r in rows) > max(g[5] for g in public.GEOMETRY.values())
    largest_public = max(
        b * h * runtime.attention_pairs(q, k, c) * d for _f, b, h, _hk, d, q, k, c in public.GEOMETRY.values()
    )
    largest_hidden = max(
        r["b"] * r["h"] * runtime.attention_pairs(r["s_q"], r["s_kv"], r["causal"]) * r["d"] for r in rows
    )
    assert largest_hidden > 4 * largest_public


def test_case_derives_geometry_and_rejects_mismatch() -> None:
    case = runtime.validate_case(row())
    assert (case["b"], case["h"], case["d"], case["s_q"], case["s_kv"]) == (1, 32, 64, 47616, 47616)
    assert case["softmax_scale"] == pytest.approx(64.0**-0.5)
    assert case["effective_flops"] == 26 * 32 * 47616 * 47616 * 64
    assert case["domain"] == "video"
    for bad, match in (
        ({"family": "wan13b"}, "head dimension"),
        ({"s_q": 12345, "s_kv": 12345}, "lattice"),
        ({"s_kv": 511}, "cross attention"),
        ({"h": 16, "h_kv": 16}, "public identity"),
        ({"tier": "real"}, "mismatch"),
        ({"gated": False}, "gated"),
        ({"layer": "wan13b-b00-s010"}, "real layer"),
        ({"causal": True}, "non-causal"),
        ({"h_kv": 8}, "non-causal"),
        ({"b": 2}, "one clip"),
        ({"h_kv": 5}, "multiple of h_kv"),
        ({"causal": 1}, "causal flag"),
    ):
        with pytest.raises(ValueError, match=match):
            runtime.validate_case(row(**bad))
    with pytest.raises(ValueError, match="schema"):
        runtime.validate_case({"case_id": "x", "family": "ltx2"})


def test_llm_case_counts_only_the_causal_triangle() -> None:
    case = runtime.validate_case(llm_row())
    assert case["domain"] == "llm"
    assert case["effective_flops"] == 26 * 4 * 32 * (4096 * 4097 // 2) * 128
    assert case["input_bytes"] == 2 * 4 * 128 * (2 * 32 * 4096 + 2 * 8 * 4096)
    for bad, match in (
        ({"causal": False}, "causal self attention"),
        ({"s_kv": 2048}, "causal self attention"),
        ({"s_q": 4100, "s_kv": 4100}, "length quantum"),
        ({"h": 28, "h_kv": 4, "s_q": 3584, "s_kv": 3584}, "public identity"),
        ({"d": 64}, "head dimension"),
    ):
        with pytest.raises(ValueError, match=match):
            runtime.validate_case(llm_row(**bad))


def test_cross_attention_is_only_the_text_length() -> None:
    case = runtime.validate_case(row(s_kv=runtime.KANDINSKY5_TEXT_TOKENS))
    assert case["s_kv"] == 512 and case["s_q"] == 47616


def test_real_case_requires_a_layer_name() -> None:
    real = {
        "case_id": "r", "family": "real_video", "tier": "real", "b": 1, "h": 12, "h_kv": 12,
        "d": 128, "s_q": 32760, "s_kv": 32760, "causal": False, "seed": 1, "gated": True,
    }
    assert runtime.validate_case({**real, "layer": "wan13b-b00-s010"})["real_layer"] == "wan13b-b00-s010"
    with pytest.raises(ValueError, match="real layer"):
        runtime.validate_case({**real, "layer": None})
    with pytest.raises(ValueError, match="real layer"):
        runtime.validate_case({**real, "layer": "../secret"})
    llm = {**real, "family": "real_llm", "h": 40, "h_kv": 8, "s_q": 8192, "s_kv": 8192, "causal": True}
    assert runtime.validate_case({**llm, "layer": "qwen3-14b-l03-x01"})["domain"] == "llm"
    assert runtime._real_layer_source("qwen3-14b-l03-x01") == "qwen3-14b"
    assert runtime._real_layer_source("cogvideox2b-b06-s010") == "cogvideox2b"
    assert runtime._real_layer_source("wan13b-b00-s010") == "wan13b"
    for layer in ("qwen3-14b-l03-x01", "qwen35-4b-l03-x01", "cogvideox2b-b00-s010", "wan14b-b39-s050"):
        assert runtime.REAL_LAYER_RECIPES[runtime._real_layer_source(layer)]


def test_edge_fixture_set_is_frozen() -> None:
    assert len(runtime.EDGE_CASES) == 15
    assert runtime.EXPECTED_EDGE_CASE_COUNT == verifier.EXPECTED_EDGE_CASE_COUNT
    assert runtime.EDGE_CASE_IDS == verifier.EDGE_CASE_IDS
    by_id = {case["case_id"]: case for case in runtime.EDGE_CASES}
    assert by_id[runtime.FIXTURE_CASE_ID]["fixture"] is True
    assert by_id["edge-e3-cross-attention"]["s_q"] != by_id["edge-e3-cross-attention"]["s_kv"]
    assert by_id["edge-e4-unaligned-cross"]["s_kv"] % 16 != 0
    assert by_id["edge-e5-zero-key-zero-v-groups"]["k_zero"] is True
    assert by_id["edge-e5-zero-key-zero-v-groups"]["v_zero_groups"] > 0.0
    assert by_id["edge-e6-rtne-ties"]["tie_set"] is True
    assert by_id["edge-e8-forced-sink"]["overrides"]["maxp_targets"] == [1.0] * 12
    assert by_id["edge-e9-tiny-do-scale"]["do_scale"] == 1.0e-6
    assert by_id["edge-e12-causal-gqa"]["causal"] is True
    assert by_id["edge-e12-causal-gqa"]["s_q"] % 16 != 0
    assert by_id["edge-e13-causal-sink"]["h_kv"] == 1
    assert {case["d"] for case in runtime.EDGE_CASES} == {64, 128, 256}


def test_edge_case_validation_derives_geometry() -> None:
    case = runtime.validate_edge_case(dict(runtime.EDGE_CASES[3]))
    assert case["scored"] is False and case["tier"] == "edge"
    assert case["effective_flops"] == runtime.effective_flops(2, 2, 1000, 515, 128)
    causal = runtime.validate_edge_case(dict(runtime.EDGE_CASES[11]))
    assert causal["effective_flops"] == runtime.effective_flops(2, 8, 1003, 1003, 128, True)
    with pytest.raises(ValueError, match="head dimension"):
        runtime.validate_edge_case({"b": 1, "s_q": 8, "s_kv": 8, "h": 1, "d": 32})
    with pytest.raises(ValueError, match="geometry"):
        runtime.validate_edge_case({"b": 1, "s_q": 8, "s_kv": 4, "h": 1, "d": 64, "causal": True})


def test_manifest_and_verifier_agree() -> None:
    payload = manifest()
    cases = verifier.validate_final_manifest(payload)
    assert len(cases) == 287
    for case in cases:
        runtime.validate_case(dict(case))
    for item in payload["cases"]:
        assert item["gated"] or item["tier"] == "sweep"
    for family, tiers in verifier.EXPECTED_TIER_COUNTS.items():
        if "sweep" in tiers:
            gated = sum(
                1 for c in payload["cases"]
                if c["family"] == family and c["tier"] == "sweep" and c["gated"]
            )
            assert gated == (tiers["sweep"] + 1) // 2


def test_fine_tuning_draw_changes_with_the_seed_but_keeps_its_shape() -> None:
    first = [c for c in manifest(7)["cases"] if c["family"] == "llm_sft"]
    second = [c for c in manifest(8)["cases"] if c["family"] == "llm_sft"]
    assert len(first) == len(second) == generator.SFT_CASES
    assert [(c["h"], c["s_q"]) for c in first] != [(c["h"], c["s_q"]) for c in second]
    for case in first + second:
        assert case["b"] * case["s_q"] <= 32768
        assert case["s_q"] not in runtime.PUBLIC_LENGTHS


def test_manifest_rejects_a_duplicate_a_missing_case_and_a_public_identity() -> None:
    payload = manifest(11)
    payload["cases"][1] = {**payload["cases"][0], "case_id": "dup", "seed": 99}
    with pytest.raises(RuntimeError, match="not unique"):
        verifier.validate_final_manifest(payload)
    payload = manifest(11)
    payload["cases"].pop()
    with pytest.raises(RuntimeError, match="count"):
        verifier.validate_final_manifest(payload)
    payload = manifest(13)
    payload["cases"][0]["h"] = 24
    payload["cases"][0]["h_kv"] = 24
    with pytest.raises(RuntimeError, match="public identity"):
        verifier.validate_final_manifest(payload)


def test_manifest_rejects_too_few_gated_sweep_cases() -> None:
    payload = manifest(17)
    for item in payload["cases"]:
        if item["tier"] == "sweep":
            item["gated"] = False
    with pytest.raises(RuntimeError, match="too few"):
        verifier.validate_final_manifest(payload)
