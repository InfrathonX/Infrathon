"""Controller/worker trust-phase ordering tests."""

from __future__ import annotations

from conftest import ROOT


def test_controller_protects_and_admits_before_worker() -> None:
    text = (ROOT / "images/judge/verifier/controller.py").read_text(encoding="utf-8")
    protected = text.index("verify_protected(args.judge_root")
    admitted = text.index("validate_submission(args.candidate_root")
    binding = text.index("from runtime.oracle import judge_outputs")
    worker = text.index("completed = run_worker(worker_command")
    reference_load = text.index("reference_model = bound_load_target")
    assert protected < admitted < binding < worker < reference_load


def test_controller_uses_global_aggregate_for_frozen_tiers() -> None:
    text = (ROOT / "images/judge/verifier/controller.py").read_text(encoding="utf-8")
    assert "all_records.extend(records)" in text
    assert "aggregate = bound_judge_outputs(" in text
    assert "all_records," in text
    assert 'tiers_ok=aggregate["tiers_passed"]' in text


def test_public_evaluator_applies_tiers_only_to_global_run() -> None:
    """The quick suite must not turn one rare event into a per-cell 33% rate."""
    public_eval = (
        ROOT / "images/work/public_tests/public_eval.py"
    ).read_text(encoding="utf-8")
    assert "all_records.extend(oracle_records)" in public_eval
    assert "aggregate = judge_outputs(" in public_eval
    assert 'policy.get("tiers"),' in public_eval


def test_protected_hashes_cover_profile_entry() -> None:
    manifest = (ROOT / "images/judge/protected_hashes.txt").read_text(encoding="utf-8")
    assert "judge:verify_eagle3.py" in manifest
    assert "judge:verifier/controller.py" in manifest
