from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from conftest import TASK_ROOT, load_module, judge_only_file


verifier = load_module("standalone_v4_verifier_partition", "images/judge/verify_grouped.py")
generator = load_module("standalone_v4_manifest_generator", "scripts/generate_final_manifest.py")
sys.path.insert(0, str(TASK_ROOT / "images/work/benchmark"))
public_runner = load_module("standalone_v4_public_runner", "images/work/benchmark/public_runner.py")


def _frozen() -> dict:
    return json.loads(judge_only_file("judge/private/final_cases.json").read_text(encoding="utf-8"))


def _shapes(cases) -> set[tuple[int, int]]:
    pairs = {(c["n"], c["k"]) for c in cases}
    pairs |= {(c["n2"], c["k2"]) for c in cases if c.get("n2")}
    return pairs


def test_the_frozen_final_is_four_families_of_serving_cases() -> None:
    cases = verifier.validate_final_manifest(_frozen())
    assert len(cases) == verifier.FINAL_CASE_COUNT == generator.FINAL_CASE_COUNT == 1000
    counts = {family: sum(c["family"] == family for c in cases) for family in verifier.FAMILIES}
    assert counts == verifier.FAMILY_CASE_COUNTS
    assert all(case["seed"] >= verifier.FINAL_SEED_MINIMUM for case in cases)
    assert {c["layout"] for c in cases if c["family"] == "prefill"} == {"psum"}
    for family in ("decode", "latency", "sequence"):
        assert {c["layout"] for c in cases if c["family"] == family} == {"masked", "psum"}
    assert any(c["hint_scale"] != 1.0 for c in cases if c["family"] == "decode")
    assert max(c["m_max"] for c in cases) > 4096
    assert len(_shapes(cases)) >= 20


def test_public_cases_share_no_shape_or_seed_with_the_hidden_suite_and_are_smaller() -> None:
    frozen = _frozen()["cases"]
    public = public_runner.PUBLIC_CASES
    assert len(public) == generator.PUBLIC_CASE_COUNT == 60
    assert _shapes(public).isdisjoint(_shapes(frozen))
    public_seeds = {case["seed"] for case in public}
    assert max(public_seeds) < verifier.FINAL_SEED_MINIMUM
    assert public_seeds.isdisjoint({case["seed"] for case in frozen})
    # Hidden cases are larger: bigger single calls and wider weight blocks.
    def flops(case):
        return 2 * case["g"] * case["expected_m"] * case["n"] * case["k"]
    assert max(flops(c) for c in frozen) > 3 * max(flops(c) for c in public)
    assert max(c["n"] * c["k"] for c in frozen) > max(c["n"] * c["k"] for c in public)
    assert {c["family"] for c in public} == set(verifier.FAMILIES)
    quick = public_runner.cases_for_suite("quick")
    assert len(quick) == 2 and {case["layout"] for case in quick} == {"masked", "psum"}
    assert len(public_runner.cases_for_suite("one")) == 1
    for family in verifier.FAMILIES:
        assert {c["family"] for c in public_runner.cases_for_suite("family", family)} == {family}
    with pytest.raises(ValueError):
        public_runner.cases_for_suite("family", "masked")


def test_agent_visible_files_say_nothing_about_hidden_cases() -> None:
    work = TASK_ROOT / "images/work"
    for path in (*work.rglob("*.md"), *work.rglob("*.py"), *work.rglob("*.sh")):
        if path.name == "source_admission.py":
            continue  # byte identical to the Judge's copy, with its import denylist
        text = path.read_text(encoding="utf-8")
        for token in ("M_REGIMES", "NK_REGIMES", "final-holdout", "edge-holdout", "generate_final_manifest",
                      "final_cases", "FAMILY_CASE_COUNTS", "1000 hidden", "188 private",
                      "enumerate_m_grouped", "scored grid", "scored-grid", "32-shape", "158 private",
                      "g32_m", "g6_m", "HarnessSentinel", "plausibility_bound", "decoy", "Decoy",
                      "PyroMind", "HAMi", "terminal Judge", "DeepGEMM", "deep_gemm",
                      "upstream", "Upstream", "559d79f", "Oracle", "calc_diff", "CUTLASS", "CuTe",
                      "split-K", "stream-K", "TMA descriptors", "provenance", "Forge", "DESIGN"):
            assert token not in text, (path.name, token)


def test_public_and_trusted_runtimes_draw_identical_rows() -> None:
    work = load_module("work_runtime_partition", "images/work/benchmark/grouped_fp8_runtime.py")
    judge = load_module("judge_runtime_partition", "runtime/grouped_fp8_runtime.py")
    for case in public_runner.PUBLIC_CASES:
        work.validate_scored_case(case)
        for trial in range(3):
            seed = work.trial_seed(case["seed"], trial)
            assert work.draw_m_sizes(case, seed) == judge.draw_m_sizes(case, seed)
        if case["family"] == "sequence":
            assert work.sequence_schedule(case, 3) == judge.sequence_schedule(case, 3)
    for name in ("MAX_M_MAX", "MAX_CAPACITY_BYTES", "MAX_HINT", "FAMILIES", "LAYOUTS", "SEQUENCE_LAYERS",
                 "MIN_DEVICE_MEMORY_BYTES", "EXACT_ERROR_LIMIT", "SOURCE_ERROR_LIMIT"):
        assert getattr(work, name) == getattr(judge, name), name


@pytest.mark.parametrize(
    "mutation",
    ["off_domain", "repeat_config", "low_seed", "repeat_seed", "extra_field", "short", "old_suite",
     "prefill_masked", "family_counts", "dist_fields"],
)
def test_final_manifest_mutations_fail_closed(mutation: str) -> None:
    manifest = json.loads(json.dumps(_frozen()))
    cases = manifest["cases"]
    if mutation == "off_domain":
        cases[0]["n"] = 1000
    elif mutation == "repeat_config":
        cases[1] = dict(cases[0], case_id=cases[1]["case_id"], seed=cases[1]["seed"])
    elif mutation == "low_seed":
        cases[0]["seed"] = 1000
    elif mutation == "repeat_seed":
        cases[1]["seed"] = cases[0]["seed"]
    elif mutation == "extra_field":
        cases[0]["m_sizes"] = [1]
    elif mutation == "short":
        cases.pop()
    elif mutation == "old_suite":
        manifest["suite_id"] = "final-holdout-v3"
    elif mutation == "prefill_masked":
        next(case for case in cases if case["family"] == "prefill")["layout"] = "masked"
    elif mutation == "family_counts":
        next(case for case in cases if case["family"] == "latency")["family"] = "decode"
    elif mutation == "dist_fields":
        next(case for case in cases if case["dist"] == "dirichlet")["alpha"] = None
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
            "final-holdout-v4",
        )
    for bare_prefix in ("auto-", "agent-", "manual-", "candidate-"):
        write({"round_id": bare_prefix})
        with pytest.raises(RuntimeError, match="unknown profile"):
            verifier.load_evaluation_profile(context, expected_uid=os.getuid())
    write({"round_id": "candidate-impl-v3", "candidate_version": 3})
    profile = verifier.load_evaluation_profile(context, expected_uid=os.getuid())
    assert profile[:2] == ("final", "final-holdout-v4") and profile[3] == 3
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
