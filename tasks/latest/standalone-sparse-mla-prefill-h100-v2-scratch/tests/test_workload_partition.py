"""The scored suite, the frozen holdout and the public suite partition the workload.

The Judge scores four deployed geometries (DeepSeek-V3.2, GLM-5,
DeepSeek-V4-Flash, DeepSeek-V4-Pro) in eight groups; its anchors are FlashMLA's
sparse-prefill performance suite at ``ba89a34`` (s_q = 4096, each template's own
s_kv list) plus the same template at the GLM-5 geometry.  The Judge runtime,
verifier and holdout generator carry the same tables.  The public suite has its
own 108 cases in the Work runtime, at top-k values and head counts no hidden
case uses, and shares no call shape with the hidden suite.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import pytest

from conftest import TASK_ROOT, load_module, judge_only_file
from suite_fixtures import fake_plan


verifier = load_module("sparse_mla_v2_verifier_partition", "images/judge/verify_sparse_mla.py")
runtime = load_module("sparse_mla_v2_runtime_partition", "runtime/sparse_mla_prefill_runtime.py")
generator = load_module("sparse_mla_v2_manifest_generator", "scripts/generate_final_manifest.py")
public_runner = load_module("sparse_mla_v2_public_runner", "images/work/benchmark/public_runner.py")
public_runtime = load_module("sparse_mla_v2_public_runtime", "images/work/benchmark/sparse_mla_prefill_runtime.py")


# Independent design fixture, not a projection of any producer: FlashMLA
# ba89a34 tests/test_flash_mla_sparse_prefill.py performance_case_templates
# (d_qk, h_q, topk, s_kv list), the same template at the GLM-5 geometry, and the
# HF config.json of each model (max_position_embeddings, qk_head_dim).
UPSTREAM_TEMPLATES = (
    ("v32", "DeepSeek-V3.2", 576, 128, 2048, (8192, 32768, 65536, 98304, 131072), 163_840, 192),
    ("glm5", "GLM-5", 576, 64, 2048, (8192, 32768, 65536, 131072), 202_752, 256),
    ("v4_flash", "DeepSeek-V4-Flash", 512, 64, 512, (8192, 32768, 49152, 65536), 1 << 20, 512),
    ("v4_pro", "DeepSeek-V4-Pro", 512, 128, 1024, (8192, 32768, 49152, 65536), 1 << 20, 512),
)
UPSTREAM_SHAPES = tuple(
    (family, s_kv) for family, _model, _d, _h, _k, values, _ctx, _qk in UPSTREAM_TEMPLATES for s_kv in values
)
FROZEN = json.loads(judge_only_file("judge/private/final_cases.json").read_text(encoding="utf-8"))


def _plan_of(instance: dict) -> list[dict]:
    rows = []
    for case in instance["cases"]:
        row = {key: case[key] for key in ("group", "family", "h_q", "s_q", "s_kv")}
        if case["group"] == "length":
            row["length"] = f"topk/{case['length_divisor']}"
            row["ragged"] = case["ragged"]
        elif case["group"] == "pattern":
            row["index_pattern"] = case["index_pattern"]
        elif case["group"] == "multi":
            row["request_lengths"] = case["request_lengths"]
        elif case["group"] in ("trace", "replay"):
            row["call_list"] = case["calls"]
        rows.append(row)
    return rows


def _call_shapes(case: dict) -> set[tuple[int, int]]:
    if "calls" in case:
        return {(int(s_q), int(s_kv)) for s_q, s_kv in case["calls"]}
    return {(int(case["s_q"]), int(case["s_kv"]))}


def test_frozen_suite_and_numerical_contract_match_every_producer() -> None:
    assert runtime.FAMILIES == verifier.FAMILIES == generator.FAMILIES == public_runtime.FAMILIES
    assert runtime.FAMILIES == tuple(template[0] for template in UPSTREAM_TEMPLATES)
    for family, model, d_qk, h_q, topk, s_kv, context, qk_head_dim in UPSTREAM_TEMPLATES:
        for module in (runtime, verifier):
            assert module.FAMILY_GEOMETRY[family] == (d_qk, h_q)
            assert module.FAMILY_TOPK[family] == topk
            assert module.FAMILY_MODEL[family] == model
            assert module.ANCHOR_S_KV[family] == s_kv
            assert module.FAMILY_MAX_CONTEXT[family] == context
            assert module.FAMILY_SM_SCALE[family] == qk_head_dim ** -0.5
        assert generator.FAMILY_HEADS[family] == h_q
        assert public_runtime.FAMILY_GEOMETRY[family] == (d_qk, h_q)
    assert runtime.anchor_grid() == UPSTREAM_SHAPES
    assert runtime.ANCHOR_S_Q == verifier.ANCHOR_S_Q == 4096
    assert runtime.ANCHOR_SM_SCALE == verifier.ANCHOR_SM_SCALE == 0.5
    assert runtime.SCORED_CONTENT == {
        "bias": "upstream", "index_mode": "upstream", "invalid_rows": "none", "duplicates": False,
        "scramble": False, "sink_mode": "upstream", "length_mode": "none",
    }
    groups = {case["group"] for case in FROZEN["cases"]}
    assert groups == set(runtime.GROUPS)
    anchors = sorted((case["family"], case["s_kv"]) for case in FROZEN["cases"] if case["group"] == "anchor")
    assert anchors == sorted(UPSTREAM_SHAPES)


def test_public_suite_shares_no_shape_with_the_scored_grid() -> None:
    cases = [public_runtime.validate_scored_case(case) for case in public_runner.public_cases()]
    topk = {family: value for family, _m, _d, _h, value, _s, _c, _q in UPSTREAM_TEMPLATES}
    anchors = {(family, 4096, topk[family], s_kv) for family, s_kv in UPSTREAM_SHAPES}
    assert not {(c["family"], c["s_q"], c["topk"], c["s_kv"]) for c in cases} & anchors
    # Public cases never run a hidden case's top-k, so no public call has a hidden call's shape.
    assert all(case["topk"] != topk[case["family"]] for case in cases)
    assert all(case["seed"] < verifier.FINAL_SEED_MINIMUM for case in cases)
    assert {c["inputs"] for c in cases if c["group"] == "basic"} == {"A", "B"}
    hidden_heads = {(case["family"], case["h_q"]) for case in FROZEN["cases"] if case["group"] == "heads"}
    public_heads = {(case["family"], case["h_q"]) for case in cases if case["group"] == "heads"}
    assert public_heads and not public_heads & hidden_heads
    # Whole call identities (geometry, heads, top-k and sizes); failures report only a count,
    # never a hidden shape.
    hidden_calls = {
        (case["family"], case["h_q"], topk[case["family"]], s_q, s_kv)
        for case in FROZEN["cases"] for s_q, s_kv in _call_shapes(case)
    }
    public_calls = {
        (case["family"], case["h_q"], case["topk"], s_q, s_kv)
        for case in cases for s_q, s_kv in _call_shapes(case)
    }
    overlap = len(hidden_calls & public_calls)
    assert overlap == 0, f"{overlap} public call shapes are hidden call shapes"
    for name in ("ANCHOR_S_KV", "ANCHOR_S_Q", "anchor_grid", "FAMILY_TOPK", "FAMILY_MAX_CONTEXT",
                 "FAMILY_S_KV", "scored_grid"):
        assert not hasattr(public_runtime, name), name


def test_many_generated_holdouts_validate_and_all_partitions_are_disjoint() -> None:
    public_seeds = {case["seed"] for case in public_runner.public_cases()}
    plans = (_plan_of(FROZEN), fake_plan())
    drawn = [generator.generate_manifest(plan, random.Random(seed)) for plan in plans for seed in range(16)]
    for manifest in (FROZEN, *drawn):
        assert set(manifest) == {"schema_version", "suite_id", "cases"}
        cases = verifier.validate_final_manifest(manifest)
        anchors = [(case["family"], case["s_kv"]) for case in cases if case["group"] == "anchor"]
        assert sorted(anchors) == sorted(UPSTREAM_SHAPES)
        assert {case["seed"] for case in cases}.isdisjoint(public_seeds)
        for raw_case in cases:
            expanded = runtime.validate_scored_case(raw_case)
            if raw_case["group"] == "anchor":
                assert expanded["s_q"] == 4096 and expanded["sm_scale"] == 0.5
            else:
                assert expanded["sm_scale"] == runtime.FAMILY_SM_SCALE[raw_case["family"]]
            assert verifier.FINAL_SEED_MINIMUM <= raw_case["seed"] <= verifier.MAX_SEED


def test_per_call_seeds_are_63_bit_and_role_separated() -> None:
    seed = FROZEN["cases"][0]["seed"]
    roles = {runtime.derive_seed(seed, role, trial) for role in ("warmup", "decoy", "measured", "call")
             for trial in range(runtime.STREAM_WARM_CALLS)}
    assert len(roles) == 4 * runtime.STREAM_WARM_CALLS
    assert all(0 <= value <= runtime.MAX_SEED for value in roles)
    assert runtime.derive_seed(seed, "decoy", 0) == runtime.derive_seed(seed, "decoy", 0)


@pytest.mark.parametrize(
    "mutation",
    ["off_grid", "repeat_shape", "low_seed", "high_seed", "repeat_seed", "extra_field", "short",
     "old_suite", "old_schema"],
)
def test_final_manifest_mutations_fail_closed(mutation: str) -> None:
    manifest = generator.generate_manifest(fake_plan(), random.Random(20261006))
    verifier.validate_final_manifest(json.loads(json.dumps(manifest)))
    cases = manifest["cases"]
    assert cases[0]["group"] == cases[1]["group"] == "anchor" and cases[0]["family"] == cases[1]["family"]
    if mutation == "off_grid":
        cases[0]["s_kv"] = 12288
    elif mutation == "repeat_shape":
        cases[1] = dict(cases[0], case_id="dup", seed=cases[1]["seed"])
    elif mutation == "low_seed":
        cases[0]["seed"] = public_runner.public_cases()[0]["seed"]
    elif mutation == "high_seed":
        cases[0]["seed"] = 1 << 63
    elif mutation == "repeat_seed":
        cases[1]["seed"] = cases[0]["seed"]
    elif mutation == "extra_field":
        cases[0]["topk"] = 2048
    elif mutation == "short":
        cases.pop()
    elif mutation == "old_suite":
        manifest["suite_id"] = "final-holdout-v3"
    elif mutation == "old_schema":
        manifest["schema_version"] = 4
    with pytest.raises(RuntimeError):
        verifier.validate_final_manifest(manifest)


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
