"""Core API termination and oracle tier tests."""

from __future__ import annotations

import torch

from runtime.api import (
    EngineContract,
    GenerationResult,
    GenerationSpec,
    truncate_generation,
    validate_result,
)
from runtime.oracle import OracleRecord, judge_outputs


def contract() -> EngineContract:
    """Build a small CPU contract fixture."""
    return EngineContract(
        "target",
        "target-revision",
        "draft",
        "draft-revision",
        device="cpu",
        vocab_size=20,
        eos_token_ids=(7, 9),
    )


def record(margins: list[float]) -> OracleRecord:
    """Build an oracle record with divergences wherever margin is nonzero."""
    count = len(margins)
    return OracleRecord(
        torch.ones(count, dtype=torch.long),
        torch.tensor([2 if margin else 1 for margin in margins]),
        torch.tensor(margins),
        torch.ones(count),
    )


def test_truncate_first_stop_and_budget() -> None:
    tokens = torch.tensor([1, 2, 3, 7, 4])
    assert truncate_generation(tokens, 2, 3, (7, 9)).tolist() == [1, 2, 3, 7]


def test_shape_and_termination_pass() -> None:
    prompt = torch.tensor([1, 2])
    result = GenerationResult([torch.tensor([1, 2, 3, 7])])
    assert validate_result([prompt], result, GenerationSpec(3), contract()) == [2]


def test_tier_pass_and_fail() -> None:
    tiers = [{"margin": 0.3, "max_rate": 0.02}]
    passed = judge_outputs([record([0.0] * 98 + [0.4] * 2)], 1.0, 0.1, tiers)
    failed = judge_outputs([record([0.0] * 97 + [0.4] * 3)], 1.0, 0.1, tiers)
    assert passed["passed"] and passed["tiers_passed"]
    assert not failed["passed"] and not failed["tiers_passed"]


def test_o1_and_o2_remain_additional() -> None:
    assert not judge_outputs([record([2.0])], 1.5, 1.0, [])["o1_passed"]
    assert not judge_outputs([record([0.1] * 2 + [0.0] * 8)], 1.5, 0.1, [
    ])["o2_passed"]
