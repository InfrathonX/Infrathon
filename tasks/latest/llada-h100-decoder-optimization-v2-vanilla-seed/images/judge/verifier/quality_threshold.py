"""Deterministic producer-relative quality measurement and release threshold.

The release threshold is not chosen in this module.  It is derived later from
the already-precommitted positive and negative control observations.  This
module only defines the frozen scalar measurement shared by Judge and host
qualification tools.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


METRIC = "producer_relative_min_retention_ppm_v1"
SCALE_PPM = 1_000_000
FIELDS = ("correct", "parsed", "useful_tokens")
# Fixed upper bound on candidate/producer useful-token retention, per row and
# in aggregate.  HumanEval counts every non-EOS slot of the fixed buffer, so
# rewriting or suppressing EOS would otherwise raise the throughput numerator
# without any extra work or quality change.  The calibrated reference replays
# at exactly 1.0x and quality-preserving decoders stayed within 1.2x; the
# precommitted positive control must stay below this bound to qualify.
MAX_USEFUL_TOKENS_RETENTION_PPM = 1_500_000
# Fixed retention floors on top of the control-derived per-component minimum.
# Rows of 13 generations are too small for a tight floor (one answer is about
# 8-12% of a row), so the strict floors apply where the counts are large: the
# whole 520-generation cohort and each benchmark's 130 generations.  Fast-dLLM
# reports accuracy within 1-2 points of the baseline decoder.
AGGREGATE_FLOOR_PPM = 900_000
BENCHMARK_FLOOR_PPM = 750_000
BENCHMARKS = ("gsm8k", "math", "humaneval", "mbpp")
QUALITY_GATE = "control_min_retention_fixed_floors_token_ceiling_v1"


def canonical_sha256(value: Mapping[str, Any]) -> str:
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return hashlib.sha256(raw).hexdigest()


def measure_quality_retention(
    candidate_rows: Mapping[str, Mapping[str, Any]],
    producer_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the minimum exact integer retention across all supported views.

    Zero producer denominators are excluded rather than inventing a ratio.
    At least one component must remain, and candidate values must be
    nonnegative integers.  Ratios above one million are retained so an
    improvement is not silently clipped.
    """

    if set(candidate_rows) != set(producer_rows) or not candidate_rows:
        raise ValueError("candidate/producer quality row sets differ")
    components: dict[str, int] = {}
    candidate_totals = {field: 0 for field in FIELDS}
    producer_totals = {field: 0 for field in FIELDS}
    for row_id in sorted(producer_rows):
        candidate = candidate_rows[row_id]
        producer = producer_rows[row_id]
        for field in FIELDS:
            numerator = candidate.get(field)
            denominator = producer.get(field)
            if (
                isinstance(numerator, bool)
                or not isinstance(numerator, int)
                or numerator < 0
                or isinstance(denominator, bool)
                or not isinstance(denominator, int)
                or denominator < 0
            ):
                raise ValueError("quality retention inputs must be nonnegative integers")
            candidate_totals[field] += numerator
            producer_totals[field] += denominator
            if denominator > 0:
                components[f"row:{row_id}:{field}"] = (
                    numerator * SCALE_PPM // denominator
                )
    for field in FIELDS:
        if producer_totals[field] > 0:
            components[f"aggregate:{field}"] = (
                candidate_totals[field] * SCALE_PPM // producer_totals[field]
            )
    if not components:
        raise ValueError("producer quality baseline has no positive denominator")
    measurement: dict[str, Any] = {
        "schema_version": 1,
        "metric": METRIC,
        "scale_ppm": SCALE_PPM,
        "score_ppm": min(components.values()),
        "components_ppm": components,
    }
    measurement["measurement_sha256"] = canonical_sha256(measurement)
    return measurement


def useful_tokens_above_ceiling(measurement: Mapping[str, Any]) -> list[str]:
    """Return the useful-token components that exceed the fixed ceiling."""

    return sorted(
        key
        for key, value in measurement["components_ppm"].items()
        if key.endswith(":useful_tokens") and value > MAX_USEFUL_TOKENS_RETENTION_PPM
    )


def benchmark_of_row(row_id: str) -> str:
    """Return the benchmark of a frozen row id such as ``table1_gsm8k_5shot_g256``."""

    parts = row_id.split("_")
    if len(parts) < 2 or parts[1] not in BENCHMARKS:
        raise ValueError(f"quality row id has no known benchmark: {row_id}")
    return parts[1]


def benchmark_retention(
    candidate_rows: Mapping[str, Mapping[str, Any]],
    producer_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    """Return candidate/producer retention per benchmark, pooling its rows."""

    if set(candidate_rows) != set(producer_rows) or not candidate_rows:
        raise ValueError("candidate/producer quality row sets differ")
    totals: dict[tuple[str, str], list[int]] = {}
    for row_id in producer_rows:
        benchmark = benchmark_of_row(row_id)
        for field in FIELDS:
            pair = totals.setdefault((benchmark, field), [0, 0])
            pair[0] += int(candidate_rows[row_id][field])
            pair[1] += int(producer_rows[row_id][field])
    return {
        f"benchmark:{benchmark}:{field}": numerator * SCALE_PPM // denominator
        for (benchmark, field), (numerator, denominator) in sorted(totals.items())
        if denominator > 0
    }


def fixed_gate_failures(
    measurement: Mapping[str, Any],
    benchmark_ppm: Mapping[str, int],
) -> list[str]:
    """Return the fixed-floor and token-ceiling failures of one measurement."""

    failures = []
    if any(
        value < AGGREGATE_FLOOR_PPM
        for key, value in measurement["components_ppm"].items()
        if key.startswith("aggregate:")
    ):
        failures.append("producer_relative_aggregate_below_floor")
    if any(value < BENCHMARK_FLOOR_PPM for value in benchmark_ppm.values()):
        failures.append("producer_relative_benchmark_below_floor")
    if useful_tokens_above_ceiling(measurement) or any(
        key.endswith(":useful_tokens") and value > MAX_USEFUL_TOKENS_RETENTION_PPM
        for key, value in benchmark_ppm.items()
    ):
        failures.append("producer_relative_useful_tokens_above_ceiling")
    return failures


def validate_measurement(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("quality measurement is missing")
    frozen = dict(value)
    commitment = frozen.pop("measurement_sha256", None)
    components = value.get("components_ppm")
    if (
        value.get("schema_version") != 1
        or value.get("metric") != METRIC
        or value.get("scale_ppm") != SCALE_PPM
        or not isinstance(components, dict)
        or not components
        or any(
            not isinstance(key, str)
            or not key
            or isinstance(item, bool)
            or not isinstance(item, int)
            or item < 0
            for key, item in components.items()
        )
        or value.get("score_ppm") != min(components.values())
        or not isinstance(commitment, str)
        or commitment != canonical_sha256(frozen)
    ):
        raise ValueError("quality measurement commitment is invalid")
    return value
