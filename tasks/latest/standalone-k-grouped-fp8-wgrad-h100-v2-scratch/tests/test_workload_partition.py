from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from conftest import TASK_ROOT, load_module, judge_only_file


verifier = load_module("standalone_v3_verifier_partition", "images/judge/verify_wgrad.py")
generator = load_module("standalone_v3_manifest_generator", "scripts/generate_final_manifest.py")
sys.path.insert(0, str(TASK_ROOT / "images/work/benchmark"))
public_runner = load_module("standalone_v3_public_runner", "images/work/benchmark/public_runner.py")


def _frozen() -> dict:
    return json.loads(judge_only_file("judge/private/final_cases.json").read_text(encoding="utf-8"))


def _shapes(cases) -> set[tuple[int, int]]:
    pairs = {(c["m"], c["n"]) for c in cases}
    pairs |= {(c["m2"], c["n2"]) for c in cases if c.get("m2")}
    return pairs


def test_the_frozen_final_is_four_families_of_training_cases() -> None:
    cases = verifier.validate_final_manifest(_frozen())
    assert len(cases) == verifier.FINAL_CASE_COUNT == generator.FINAL_CASE_COUNT == 1082
    counts = {family: sum(c["family"] == family for c in cases) for family in verifier.FAMILIES}
    assert counts == verifier.FAMILY_CASE_COUNTS
    assert all(case["seed"] >= verifier.FINAL_SEED_MINIMUM for case in cases)
    assert {c["dist"] for c in cases} == {"dirichlet"}
    assert sum(c["hint_scale"] != 1.0 for c in cases) == 96
    assert max(c["total_k"] for c in cases) > 4 * 49_152
    assert max(c["g"] for c in cases if c["family"] == "many_groups") == 128
    assert all(c["total_k"] // c["g"] >= 1024 for c in cases if c["family"] == "large")
    assert all(c["total_k"] // c["g"] < 1024 for c in cases if c["family"] == "small_batch")
    assert len(_shapes(cases)) >= 20
    assert len({c["model"] for c in cases}) == 12


def test_public_cases_share_no_shape_or_seed_with_the_hidden_suite_and_are_smaller() -> None:
    frozen = _frozen()["cases"]
    public = public_runner.PUBLIC_CASES
    assert len(public) == generator.PUBLIC_CASE_COUNT == 123
    assert _shapes(public).isdisjoint(_shapes(frozen))
    public_seeds = {case["seed"] for case in public}
    assert max(public_seeds) < verifier.FINAL_SEED_MINIMUM
    assert public_seeds.isdisjoint({case["seed"] for case in frozen})
    assert max(c["total_k"] for c in frozen) > 3 * max(c["total_k"] for c in public)
    assert max(c["g"] for c in frozen) > max(c["g"] for c in public)
    assert {c["family"] for c in public} == set(verifier.FAMILIES)
    assert any(c["hint_scale"] != 1.0 for c in public)
    quick = public_runner.cases_for_suite("quick")
    assert len(quick) == 2 and {case["family"] for case in quick} == {"small_batch", "large"}
    assert len(public_runner.cases_for_suite("one")) == 1
    for family in verifier.FAMILIES:
        assert {c["family"] for c in public_runner.cases_for_suite("family", family)} == {family}
    with pytest.raises(ValueError):
        public_runner.cases_for_suite("family", "coarse")


def test_agent_visible_files_say_nothing_about_hidden_cases() -> None:
    work = TASK_ROOT / "images/work"
    for path in (*work.rglob("*.md"), *work.rglob("*.py"), *work.rglob("*.sh")):
        if path.name == "source_admission.py":
            continue  # byte identical to the Judge's copy, with its import denylist
        text = path.read_text(encoding="utf-8")
        for token in ("UPSTREAM_GRID", "final-holdout", "edge-holdout", "generate_final_manifest",
                      "final_cases", "FAMILY_CASE_COUNTS", "1082", "166 private",
                      "enumerate_k_grouped", "scored grid", "scored-grid", "7-shape", "160 private",
                      "ep64_", "g8_k128", "HarnessSentinel", "plausibility_bound", "decoy", "Decoy",
                      "PyroMind", "HAMi", "terminal Judge", "DeepGEMM", "deep_gemm",
                      "upstream", "Upstream", "559d79f", "RUN_RESULTS", "iteration plan", "Oracle",
                      "calibrated", "split-K", "stream-K", "TMA descriptors", "v1_contract",
                      "provenance", "Forge", "DESIGN", "Kimi", "GLM", "ERNIE", "Ling-1T", "Hunyuan",
                      "dots.llm", "Qwen3-Coder", "Qwen3-Next", "Qwen3.5", "V4-Pro", "V3/R1"):
            assert token not in text, (path.name, token)


def test_public_and_trusted_runtimes_draw_identical_group_sizes() -> None:
    work = load_module("work_runtime_partition", "images/work/benchmark/k_grouped_wgrad_runtime.py")
    judge = load_module("judge_runtime_partition", "runtime/k_grouped_wgrad_runtime.py")
    for case in public_runner.PUBLIC_CASES:
        work.validate_scored_case(case)
        for trial in range(3):
            seed = work.derive_seed(case["seed"], "ks", trial)
            ks = work.draw_ks(case, seed)
            assert ks == judge.draw_ks(case, seed)
            assert all(k >= 0 and k % 128 == 0 for k in ks)
            assert sum(ks) <= work.MAX_TOTAL_K
        if case["family"] == "sequence":
            assert work.sequence_schedule(case, 3) == judge.sequence_schedule(case, 3)
    for name in ("MAX_TOTAL_K", "MAX_OUTPUT_ELEMENTS", "MAX_HINT", "FAMILIES", "SEQUENCE_LAYERS",
                 "SEQUENCE_OUT_POOL", "MIN_DEVICE_MEMORY_BYTES", "EXACT_ERROR_LIMIT", "SOURCE_ERROR_LIMIT",
                 "ELEMENT_RATIO_LIMIT", "CHECK_CHUNK_ROWS"):
        assert getattr(work, name) == getattr(judge, name), name


def test_dirichlet_draws_keep_k_total_and_allow_empty_groups() -> None:
    judge = load_module("judge_runtime_dirichlet", "runtime/k_grouped_wgrad_runtime.py")
    cases = [c for c in _frozen()["cases"] if c["family"] != "sequence"][:200]
    empty = 0
    for case in cases:
        for trial in range(2):
            ks = judge.draw_ks(case, judge.derive_seed(case["seed"], "ks", trial))
            assert sum(ks) == case["total_k"] and len(ks) == case["g"]
            assert ks == verifier.draw_ks(case, verifier.derive_seed(case["seed"], "ks", trial))
            empty += ks.count(0)
    assert empty > 0


@pytest.mark.parametrize(
    "mutation",
    ["off_domain", "repeat_config", "low_seed", "repeat_seed", "extra_field", "short", "old_suite",
     "family_counts", "dist_fields", "uniform", "deep_k"],
)
def test_final_manifest_mutations_fail_closed(mutation: str) -> None:
    manifest = json.loads(json.dumps(_frozen()))
    cases = manifest["cases"]
    if mutation == "off_domain":
        cases[0]["n"] = 8193
    elif mutation == "repeat_config":
        cases[1] = dict(cases[0], case_id=cases[1]["case_id"], seed=cases[1]["seed"])
    elif mutation == "low_seed":
        cases[0]["seed"] = 1000
    elif mutation == "repeat_seed":
        cases[1]["seed"] = cases[0]["seed"]
    elif mutation == "extra_field":
        cases[0]["ks"] = [128]
    elif mutation == "short":
        cases.pop()
    elif mutation == "old_suite":
        manifest["suite_id"] = "final-holdout-v2"
    elif mutation == "family_counts":
        next(case for case in cases if case["family"] == "small_batch")["family"] = "large"
    elif mutation == "dist_fields":
        cases[0]["alpha"] = None
    elif mutation == "uniform":
        cases[0]["dist"] = "uniform"
    elif mutation == "deep_k":
        cases[0]["total_k"] = 262_144 + 128
    with pytest.raises(RuntimeError):
        verifier.validate_final_manifest(manifest)


def test_root_only_context_routes_every_round_to_the_shared_final(tmp_path: Path) -> None:
    context = tmp_path / "context.json"

    def write(payload) -> None:
        context.write_text(json.dumps(payload), encoding="utf-8")
        context.chmod(0o600)

    for round_id in ("auto-123", "manual-final", "agent-2", "candidate-v3-attempt-1"):
        write({"round_id": round_id})
        assert verifier.load_evaluation_profile(context, expected_uid=os.getuid())[:2] == (
            "final",
            "final-holdout-v3",
        )
    for bare_prefix in ("auto-", "agent-", "manual-", "candidate-"):
        write({"round_id": bare_prefix})
        with pytest.raises(RuntimeError, match="unknown profile"):
            verifier.load_evaluation_profile(context, expected_uid=os.getuid())
    write({"round_id": "candidate-impl-v3", "candidate_version": 3})
    profile = verifier.load_evaluation_profile(context, expected_uid=os.getuid())
    assert profile[:2] == ("final", "final-holdout-v3") and profile[3] == 3
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
