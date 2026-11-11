"""Trusted cell and aggregate throughput calculations."""

from __future__ import annotations
import math
from typing import Mapping


def cell_throughput(measurements: list[Mapping]) -> float:
    """Compute total generated tokens divided by total synchronized wall."""
    tokens = sum(sum(int(n) for n in row["generated_lengths"]) for row in measurements)
    wall = math.fsum(float(row["wall_s"]) for row in measurements)
    if tokens <= 0 or not math.isfinite(wall) or wall <= 0:
        raise ValueError("invalid token/wall totals")
    return tokens / wall


def geometric_mean(values) -> float:
    """Compute the finite positive geometric mean."""
    values = [float(value) for value in values]
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("geometric mean requires finite positive values")
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))
