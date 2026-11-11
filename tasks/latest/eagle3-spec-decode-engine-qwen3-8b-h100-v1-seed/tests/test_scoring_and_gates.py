"""Score aggregation and frozen tier contracts."""

from __future__ import annotations

import json
import math

import pytest

from conftest import ROOT, private_data
from scoring import cell_throughput, geometric_mean


def test_cell_throughput_uses_total_tokens_over_total_wall() -> None:
    rows = [
        {"generated_lengths": [10, 20], "wall_s": 2.0},
        {"generated_lengths": [30], "wall_s": 1.0},
    ]
    assert cell_throughput(rows) == pytest.approx(20.0)


def test_geometric_mean_is_uncapped() -> None:
    assert geometric_mean([100.0, 400.0]) == pytest.approx(200.0)
    assert geometric_mean([1e3, 1e3]) == pytest.approx(1e3)


def test_frozen_tiers_are_monotone_in_rate() -> None:
    policy = json.loads((private_data() / "policy.json").read_text())
    rates = [row["max_rate"] for row in policy["tiers"]]
    margins = [row["margin"] for row in policy["tiers"]]
    assert margins == sorted(margins)
    assert all(math.isfinite(rate) and 0 < rate < 1 for rate in rates)
    assert rates == sorted(rates, reverse=True)
