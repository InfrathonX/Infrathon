"""Protected-hash and hidden-manifest generation tests."""

from __future__ import annotations

import json

import pytest

from conftest import ROOT, load_file, private_data

hashes = load_file("protected_hashes_test", ROOT / "scripts/generate_protected_hashes.py")
hidden = load_file("hidden_manifest_test", ROOT / "scripts/generate_hidden_manifest.py")
calibration = load_file("calibration_test", ROOT / "scripts/calibrate_oracle.py")


def test_rendered_protected_hashes_equal_committed_manifest() -> None:
    expected = hashes.render(ROOT, private_data())
    committed = (ROOT / "images/judge/protected_hashes.txt").read_text()
    assert expected == committed


def test_qualification_cannot_override_a2_tiers_or_the_cap_policy() -> None:
    """Image qualification owns effort budgets and identities, nothing else."""
    template = json.loads((private_data() / "policy.json").read_text())
    frozen = {
        "build_cap_s": 600,
        "worker_timeout_s": 4800,
        "eps_hard": 1.0,
        "wall_cap_seed_multiple": 1.0e6,
        "memory_cap_seed_multiple": 1.0e6,
        "memory_cap_headroom_bytes": 1,
        "oracle_rate_seed_multiple": 1.0e6,
        "resource_cap_mode": "frozen_h100",
    }
    result = hidden.qualified_policy(template, frozen)
    assert result["eps_hard"] == 24.0
    assert result["tiers"] == template["tiers"]
    assert result["resource_cap_mode"] == "seed_relative_same_run"
    assert result["wall_cap_seed_multiple"] == 8.0
    assert result["memory_cap_seed_multiple"] == 1.1
    assert result["memory_cap_headroom_bytes"] == 8 * 1024**3
    assert result["oracle_rate_seed_multiple"] == 3.0


def test_frozen_per_device_caps_cannot_reenter_the_policy() -> None:
    """A wall or peak measured on one GPU may never become a gate again."""
    policy = json.loads((private_data() / "policy.json").read_text())
    policy["wall_caps_s"] = {"chat_bs1": 37.518}
    with pytest.raises(ValueError, match="frozen per-device caps"):
        hidden.validate_frozen_oracle(policy)
    policy = json.loads((private_data() / "policy.json").read_text())
    policy["memory_caps_bytes"] = {"chat_bs1": 29825357927}
    with pytest.raises(ValueError, match="frozen per-device caps"):
        hidden.validate_frozen_oracle(policy)


def test_relative_cap_policy_is_pinned() -> None:
    """The multiples that turn a seed measurement into a cap are not tunable."""
    policy = json.loads((private_data() / "policy.json").read_text())
    policy["wall_cap_seed_multiple"] = 1000.0
    with pytest.raises(ValueError, match="resource-cap policy"):
        hidden.validate_frozen_oracle(policy)


def test_malformed_tier_policy_is_rejected() -> None:
    policy = json.loads((private_data() / "policy.json").read_text())
    policy["tiers"] = policy["tiers"][:-1]
    with pytest.raises(ValueError, match="tier"):
        hidden.validate_frozen_oracle(policy)


def test_soft_rate_cannot_diverge_from_frozen_first_tier() -> None:
    """Qualification cannot independently loosen the aggregate O2 rate."""
    policy = json.loads((private_data() / "policy.json").read_text())
    policy["p_soft"] = 0.04
    with pytest.raises(ValueError, match="soft-rate"):
        hidden.validate_frozen_oracle(policy)


def test_compact_calibration_retains_tier_evidence() -> None:
    """The summary must use the oracle's actual field names."""
    report = {
        "status": "PASS",
        "oracle": {
            "positions": 100,
            "soft_divergences": 2,
            "soft_rate": 0.02,
            "max_margin": 0.5,
            "tier_results": [
                {"margin": 0.25, "count": 1, "rate": 0.01, "passed": True}
            ],
        },
    }
    compact = calibration.compact_report(report)
    assert compact["oracle"]["soft_divergences"] == 2
    assert compact["oracle"]["tier_results"][0]["margin"] == 0.25
