"""RED/GREEN contracts for the C2 lifecycle and Judge ports."""

from __future__ import annotations

import json

import pytest

from conftest import ROOT, load_file, maintainer_doc, private_data

preflight = load_file("preflight_policy_test", ROOT / "scripts/preflight.py")


def test_declared_versions_reach_replay_and_judge() -> None:
    """Per-version replay must select an unselected engine end to end."""
    replay = (ROOT / "scripts/replay_candidates.sh").read_text(encoding="utf-8")
    verifier = ROOT / "images/judge/verify_eagle3.py"
    assert "--per-version" in replay
    assert "replay-versions" in replay
    assert verifier.is_file()
    assert "candidate_version" in verifier.read_text(encoding="utf-8")


def test_frozen_tier_values_survive_qualification() -> None:
    """D's resource freeze must not alter A2's oracle numbers."""
    policy = json.loads(
        (private_data() / "policy.json").read_text(encoding="utf-8")
    )
    assert policy["status"] in {"QUALIFIED", "PENDING_QUALIFICATION"}
    assert policy["eps_hard"] == 24.0
    assert policy["scope"] == "global_run"
    assert [row["margin"] for row in policy["tiers"]] == [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
        1.5,
        2.0,
        3.0,
    ]
    assert [row["max_rate"] for row in policy["tiers"]] == [
        0.03284789994157488,
        0.001034126163391934,
        0.00031023784901758014,
        0.00020682523267838676,
        0.00020682523267838676,
        0.00015511892450879007,
        0.0001,
        0.0001,
    ]


def test_preflight_rejects_a_mutated_frozen_tier(tmp_path) -> None:
    """A rate inside (0, 1) is still invalid when it differs from A2."""
    policy = json.loads(
        (private_data() / "policy.json").read_text(encoding="utf-8")
    )
    policy["tiers"][3]["max_rate"] = 0.0002
    mutated = tmp_path / "policy.json"
    mutated.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(ValueError, match="tier"):
        preflight.validate_policy(mutated)


def test_lifecycle_scripts_name_required_guards() -> None:
    """The launch path retains every important LLaDA lifecycle boundary."""
    run = (ROOT / "scripts/run_fixed_budget.sh").read_text(encoding="utf-8")
    for token in (
        "acquire_dual_gpu_locks",
        "remove_task_gpu_containers",
        "assert_built_images_current",
        "assert_distinct_gpu_pair",
        "restricted_proxy",
        "formal-profile",
        "runtime-provenance",
        "final_archive",
        "final_task_candidates",
        "validate_formal_artifacts.py",
    ):
        assert token in run
    library = (ROOT / "scripts/lib.sh").read_text(encoding="utf-8")
    assert '--gpus "device=${expected_uuid}"' in library


def test_judge_server_inherits_gpu_and_immutable_image_environment() -> None:
    """Interactive Judge creation must see role inputs parsed at startup."""
    run = (ROOT / "scripts/run_fixed_budget.sh").read_text(encoding="utf-8")
    server_start = run.index('serve --host "${bridge_gateway}"')
    for export in (
        'export EAGLE3_JUDGE_GPU_UUID="${judge_uuid}"',
        'export SFORGE_EXPECTED_JUDGE_IMAGE_ID="${recorded_judge_image_id}"',
        'export SFORGE_JUDGE_EXTRA_ENV="NVIDIA_VISIBLE_DEVICES=${judge_uuid}',
    ):
        assert run.index(export) < server_start


def test_replay_adapter_accepts_only_expected_immutable_image_ids() -> None:
    """Replay may pass a resolved SHA, but arbitrary SHAs remain rejected."""
    entry = (ROOT / "scripts/sforge_h100_entry.py").read_text(encoding="utf-8")
    assert 'image == os.environ.get(WORK_IMAGE_ID_ENV)' in entry
    assert 'image == os.environ.get(JUDGE_IMAGE_ID_ENV)' in entry


def test_port_ledger_covers_each_requested_port() -> None:
    """Reviewers get an explicit kept/adapted/dropped decision per port."""
    ledger = maintainer_doc("docs/PORT_LEDGER.md")
    for name in (
        "run_fixed_budget.sh",
        "eval_archive.sh",
        "replay_candidates.sh",
        "preflight.py",
        "run_judge.py",
        "verify_eagle3.py",
    ):
        assert name in ledger
    assert "dropped" in ledger.lower()
