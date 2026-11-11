"""The hardware gate is a capability envelope and every cap is same-run relative.

No GPU model name, GPU UUID, or constant measured on one GPU may act as a hard
gate or as a score input. The launcher and the host preflight admit any CUDA
device inside the envelope; the Judge derives its wall, peak-memory and oracle
limits from the seed engine measured on the device it is actually running on.
"""

from __future__ import annotations

import json
import math

import pytest

from conftest import ROOT, VERIFIER, load_file, private_data

preflight = load_file("capability_preflight", ROOT / "scripts/preflight.py")
controller = load_file("capability_controller", VERIFIER / "controller.py")


@pytest.fixture(name="POLICY", scope="module")
def _policy() -> dict:
    """The Judge-only policy (skips when the asset is not fetched)."""
    return json.loads((private_data() / "policy.json").read_text(encoding="utf-8"))


LIBRARY = (ROOT / "scripts/lib.sh").read_text(encoding="utf-8")

# Every card the fleet may run this task on, with its real compute capability.
FLEET = (
    ("NVIDIA RTX A6000", "8.6", 48 * 1024),
    ("NVIDIA A100-SXM4-40GB", "8.0", 40 * 1024),
    ("NVIDIA A100-SXM4-80GB", "8.0", 80 * 1024),
    ("NVIDIA H20", "9.0", 96 * 1024),
    ("NVIDIA H100 NVL", "9.0", 94 * 1024),
    ("NVIDIA H200", "9.0", 141 * 1024),
    ("NVIDIA B200", "10.0", 180 * 1024),
    ("Some Future Accelerator", "12.0", 64 * 1024),
)


def fake_command(rows, apps=""):
    """Return a stub for preflight.command that serves one nvidia-smi table."""

    def command(*arguments: str) -> str:
        if any(argument.startswith("--query-compute-apps") for argument in arguments):
            return apps
        return rows

    return command


def gpu_row(index, name, capability, total_mib, used_mib=0):
    """Render one nvidia-smi CSV row in the exact preflight query order."""
    return f"{index}, GPU-{index:08d}, {name}, {capability}, {total_mib}, {used_mib}\n"


def test_envelope_is_declared_and_is_not_h100_specific() -> None:
    assert preflight.MIN_COMPUTE_CAPABILITY == (8, 0)
    assert preflight.MIN_DEVICE_MEMORY_MIB == 34 * 1024


def test_capability_parsing_handles_two_digit_majors() -> None:
    assert preflight.parse_capability("8.0") == (8, 0)
    assert preflight.parse_capability("10.0") == (10, 0)
    assert preflight.parse_capability("10.0") > preflight.MIN_COMPUTE_CAPABILITY
    with pytest.raises(ValueError, match="compute capability"):
        preflight.parse_capability("sm90")


@pytest.mark.parametrize("name, capability, total_mib", FLEET)
def test_any_gpu_meeting_the_envelope_is_admitted(
    name: str, capability: str, total_mib: int, monkeypatch
) -> None:
    if total_mib < preflight.MIN_DEVICE_MEMORY_MIB:
        pytest.skip("below this task's honest memory floor")
    monkeypatch.setattr(
        preflight,
        "command",
        fake_command(gpu_row(0, name, capability, total_mib) + gpu_row(1, name, capability, total_mib)),
    )
    selected = preflight.validate_gpus([0, 1], 1024)
    assert [gpu["name"] for gpu in selected] == [name, name]
    assert selected[0]["compute_capability"] == capability
    assert selected[0]["min_compute_capability"] == "8.0"


def test_below_capability_floor_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        preflight, "command", fake_command(gpu_row(0, "NVIDIA V100-SXM2-32GB", "7.0", 32768))
    )
    with pytest.raises(ValueError, match="compute capability"):
        preflight.validate_gpus([0], 1024)


def test_below_memory_floor_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        preflight,
        "command",
        fake_command(gpu_row(0, "NVIDIA RTX A6000", "8.6", preflight.MIN_DEVICE_MEMORY_MIB - 1)),
    )
    with pytest.raises(ValueError, match="below the task"):
        preflight.validate_gpus([0], 1024)


def test_busy_and_shared_devices_are_still_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        preflight, "command", fake_command(gpu_row(0, "NVIDIA H200", "9.0", 141 * 1024, 4096))
    )
    with pytest.raises(ValueError, match="busy"):
        preflight.validate_gpus([0], 1024)
    monkeypatch.setattr(
        preflight,
        "command",
        fake_command(
            gpu_row(0, "NVIDIA H200", "9.0", 141 * 1024),
            apps="GPU-00000000, 4242, python3, 512\n",
        ),
    )
    with pytest.raises(ValueError, match="active compute owners"):
        preflight.validate_gpus([0], 1024)


def test_no_gpu_model_name_gates_any_trusted_entry_point() -> None:
    for relative in (
        "scripts/lib.sh",
        "scripts/preflight.py",
        "scripts/smoke_submission_plane.sh",
        "scripts/sforge_h100_entry.py",
        "images/judge/verifier/controller.py",
        "images/judge/verifier/candidate_worker.py",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "*H100*" not in text, relative
        assert '"H100" not in' not in text, relative
        assert "== (9, 0)" not in text, relative
        assert 'capability == "9.0"' not in text, relative


def test_launcher_floor_is_the_envelope_not_one_card() -> None:
    assert 'GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-${H100_MIN_FREE_MIB:-34816}}"' in LIBRARY
    assert 'GPU_MIN_COMPUTE_CAPABILITY="${GPU_MIN_COMPUTE_CAPABILITY:-8.0}"' in LIBRARY
    assert "assert_gpu_available()" in LIBRARY
    assert 34816 >= preflight.MIN_DEVICE_MEMORY_MIB


def test_policy_carries_no_constant_measured_on_one_gpu(POLICY) -> None:
    assert "wall_caps_s" not in POLICY
    assert "memory_caps_bytes" not in POLICY
    assert POLICY["resource_cap_mode"] == "seed_relative_same_run"
    for key in (
        "wall_cap_seed_multiple",
        "memory_cap_seed_multiple",
        "memory_cap_headroom_bytes",
        "oracle_rate_seed_multiple",
    ):
        assert POLICY[key] > 0
    preflight.validate_policy(private_data() / "policy.json")


def test_preflight_rejects_a_policy_that_reintroduces_frozen_caps(tmp_path, POLICY) -> None:
    mutated = dict(POLICY)
    mutated["wall_caps_s"] = {"chat_bs1": 37.518}
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen per-device caps"):
        preflight.validate_policy(path)
    mutated = dict(POLICY)
    mutated["resource_cap_mode"] = "frozen_h100"
    path.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(ValueError, match="same-run seed measurement"):
        preflight.validate_policy(path)


def test_seed_reference_uses_committed_warmup_rows_only() -> None:
    cells = [
        {"cell": "chat_bs1", "batch_size": 1, "max_new_tokens": 512,
         "enable_thinking": False, "rows": [{"prompt_id": "scored"}],
         "warmup": [{"prompt_id": "warm"}]},
        {"cell": "mixed_bs8", "batch_size": 8, "max_new_tokens": 512,
         "enable_thinking": False, "rows": [{"prompt_id": f"s{i}"} for i in range(16)],
         "warmup": [{"prompt_id": f"w{i}"} for i in range(8)]},
    ]
    reference = controller.seed_reference_cells(cells)
    assert [cell["cell"] for cell in reference] == ["chat_bs1", "mixed_bs8"]
    assert reference[0]["rows"] == [{"prompt_id": "warm"}]
    assert len(reference[1]["rows"]) == 8
    assert all(
        row["prompt_id"].startswith("w") for cell in reference for row in cell["rows"]
    )


def test_seed_reference_requires_one_complete_batch() -> None:
    with pytest.raises(RuntimeError, match="complete warm-up batch"):
        controller.seed_reference_cells(
            [{"cell": "mixed_bs8", "batch_size": 8, "max_new_tokens": 512,
              "rows": [], "warmup": [{"prompt_id": "w"}]}]
        )


def test_bootstrap_policy_is_bounded_by_the_worker_timeout_only() -> None:
    cells = [{"cell": "chat_bs1"}, {"cell": "mixed_bs8"}]
    bootstrap = controller.bootstrap_policy(
        {"worker_timeout_s": 4800, "build_cap_s": 600}, cells
    )
    assert bootstrap == {
        "build_cap_s": 600,
        "wall_caps_s": {"chat_bs1": 4800.0, "mixed_bs8": 4800.0},
    }


def test_caps_scale_with_the_device_the_judge_is_running_on(POLICY) -> None:
    """The same candidate behaviour must pass on a card half as fast."""
    cells = [
        {"cell": "chat_bs1", "batch_size": 1, "max_new_tokens": 512},
        {"cell": "mixed_bs8", "batch_size": 8, "max_new_tokens": 512},
    ]

    def seed_result(scale: float) -> dict:
        return {
            "cells": {
                "chat_bs1": {
                    "measurements": [
                        {"wall_s": 4.0 * scale, "generated_lengths": [256],
                         "peak_memory_bytes": 24 * 1024**3}
                    ]
                },
                "mixed_bs8": {
                    "measurements": [
                        {"wall_s": 32.0 * scale, "generated_lengths": [512] * 8,
                         "peak_memory_bytes": 26 * 1024**3}
                    ]
                },
            }
        }

    fast_wall, fast_memory, record = controller.derive_resource_caps(
        POLICY, cells, seed_result(1.0)
    )
    slow_wall, slow_memory, _ = controller.derive_resource_caps(
        POLICY, cells, seed_result(2.0)
    )
    # 4 s for 256 tokens extrapolates to 8 s for the full 512-token budget.
    assert fast_wall["chat_bs1"] == pytest.approx(8.0 * 8.0)
    assert fast_wall["mixed_bs8"] == pytest.approx(8.0 * 32.0)
    for name in ("chat_bs1", "mixed_bs8"):
        assert slow_wall[name] == pytest.approx(2.0 * fast_wall[name])
        assert slow_memory[name] == fast_memory[name]
    assert fast_memory["chat_bs1"] == math.ceil(1.1 * 24 * 1024**3) + 8 * 1024**3
    assert record["mode"] == "seed_relative_same_run"
    assert record["cells"]["chat_bs1"]["seed_peak_memory_bytes"] == 24 * 1024**3


def test_unusable_seed_measurements_fail_closed(POLICY) -> None:
    cells = [{"cell": "chat_bs1", "batch_size": 1, "max_new_tokens": 512}]
    with pytest.raises(RuntimeError, match="no measurement"):
        controller.derive_resource_caps(
            POLICY, cells, {"cells": {"chat_bs1": {"measurements": []}}}
        )
    with pytest.raises(RuntimeError, match="usable measurement"):
        controller.derive_resource_caps(
            POLICY,
            cells,
            {"cells": {"chat_bs1": {"measurements": [
                {"wall_s": 4.0, "generated_lengths": [256], "peak_memory_bytes": 0}
            ]}}},
        )
    with pytest.raises(RuntimeError, match="same-run relative caps"):
        controller.derive_resource_caps(
            {**POLICY, "resource_cap_mode": "frozen_h100"}, cells, {"cells": {}}
        )


def test_oracle_limits_are_a_floor_the_seed_can_only_relax(POLICY) -> None:
    quiet = {
        "soft_rate": 0.0,
        "tier_results": [
            {"margin": tier["margin"], "rate": 0.0} for tier in POLICY["tiers"]
        ],
    }
    p_soft, tiers = controller.seed_relative_oracle_limits(POLICY, quiet)
    assert p_soft == POLICY["p_soft"]
    assert [tier["max_rate"] for tier in tiers] == [
        tier["max_rate"] for tier in POLICY["tiers"]
    ]
    # A device on which the trusted seed itself disagrees more may not fail a
    # candidate for that alone: the limit follows the same-run seed measurement.
    noisy = {
        "soft_rate": 0.05,
        "tier_results": [
            {"margin": tier["margin"], "rate": 0.01} for tier in POLICY["tiers"]
        ],
    }
    p_soft, tiers = controller.seed_relative_oracle_limits(POLICY, noisy)
    assert p_soft == pytest.approx(0.15)
    assert [tier["max_rate"] for tier in tiers] == [
        pytest.approx(max(tier["max_rate"], 0.03)) for tier in POLICY["tiers"]
    ]
    assert any(tier["max_rate"] > 0.03 for tier in tiers)
    assert [tier["margin"] for tier in tiers] == [
        tier["margin"] for tier in POLICY["tiers"]
    ]


def test_oracle_limits_stay_probabilities(POLICY) -> None:
    saturated = {
        "soft_rate": 0.9,
        "tier_results": [
            {"margin": tier["margin"], "rate": 0.9} for tier in POLICY["tiers"]
        ],
    }
    p_soft, tiers = controller.seed_relative_oracle_limits(POLICY, saturated)
    assert p_soft == 1.0
    assert all(tier["max_rate"] == 1.0 for tier in tiers)


def test_seed_reference_identity_is_bound_by_the_protected_manifest() -> None:
    manifest = (ROOT / "images/judge/protected_hashes.txt").read_text(encoding="utf-8")
    assert "seed:solutions/__init__.py" in manifest
    assert "seed:solutions/engine_v001/candidate.py" in manifest
    dockerfile = (ROOT / "images/judge/Dockerfile").read_text(encoding="utf-8")
    assert "/opt/eagle3_judge/seed_reference" in dockerfile
    text = (VERIFIER / "controller.py").read_text(encoding="utf-8")
    assert '"seed": root / "seed_reference"' in text


def test_task_declares_a_portable_execution_profile() -> None:
    task = json.loads(
        (ROOT / "tasks/eagle3_spec_decode_engine_qwen3_8b_h100_v1.json").read_text(
            encoding="utf-8"
        )
    )
    portable = task["portable"]
    profile_id = portable["defaults"]["execution_profile_id"]
    profile = portable["execution_profiles"][profile_id]
    assert profile["count"] == 1
    assert profile["min_compute_capability"] == "8.0"
    assert profile["min_memory_bytes"] == preflight.MIN_DEVICE_MEMORY_MIB * 1024**2
    assert portable["metrics"]["primary"]["comparability"] == "same_measurement_profile"
    entry = (ROOT / "scripts/sforge_h100_entry.py").read_text(encoding="utf-8")
    assert f'EXECUTION_PROFILE = "{profile_id}"' in entry
