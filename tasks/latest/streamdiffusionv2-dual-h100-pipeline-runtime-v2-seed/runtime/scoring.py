"""Frozen, CPU-only latency-weighted FPS formula and seed-relative gates.

Every seed-relative quantity (`seed_steady_fps`, `seed_ttff_seconds`,
`seed_push_to_emit_p99_seconds`, `seed_construction_seconds`,
`peak_memory_mib_per_rank`) is supplied by the pristine seed runtime measured on
the *same* devices in the *same* evaluation run.  Nothing in this module reads a
frozen per-device measurement.

The GPU memory cap does not depend on where the seed placed its work: every rank
may use up to the seed's *larger* per-rank peak plus a fixed margin, so moving
stages between the two cards (or using only one) is judged on speed, not on
which card the seed happened to load more onto.
"""
from __future__ import annotations
import math

PRIMARY_SCORE_NAME = "latency_weighted_fps"
PRIMARY_SCORE_UNIT = "fps"
LATENCY_EXPONENT = 0.25
# Rank index of each worker GPU; the memory baseline is keyed by these, so it
# transfers to any qualifying pair instead of pinning two physical GPU UUIDs.
RANK_KEYS = ("0", "1")
# Headroom a candidate may add on top of the in-run seed's larger per-rank peak.
GPU_MEMORY_MARGIN_MIB = 8192
# Construction (model loads, spawn, process groups) may take at most
# max(CONSTRUCTION_FACTOR * seed, seed + CONSTRUCTION_ALLOWANCE_SECONDS).
CONSTRUCTION_FACTOR = 2.0
CONSTRUCTION_ALLOWANCE_SECONDS = 60.0
# Host-memory policy, not a device measurement: the capability envelope
# declares >= 64 GiB of usable host RAM and this is that number.
HOST_RSS_GROWTH_CAP_MIB = 65536
# Memory a worker may leave behind on a rank, measured against that rank's own
# pre-run occupancy in the same run.
POST_CLOSE_MEMORY_MARGIN_MIB = 1024


def positive(value):
    if isinstance(value, bool):
        raise ValueError("boolean metric")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("metric must be finite and positive")
    return value


def nonnegative(value):
    if isinstance(value, bool):
        raise ValueError("boolean metric")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("metric must be finite and non-negative")
    return value


def geomean(values):
    values = [positive(value) for value in values]
    if not values:
        raise ValueError("empty score suite")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def decorate(row, seed):
    row = dict(row)
    fps = positive(row["steady_fps"])
    p99 = positive(row["push_to_emit_p99_seconds"])
    seed_p99 = positive(seed["push_to_emit_p99_seconds"])
    seed_ttff = positive(seed["ttff_seconds"])
    seed_construction = positive(seed["construction_seconds"])
    row.update(
        seed_steady_fps=positive(seed["steady_fps"]),
        steady_fps_ratio=fps / positive(seed["steady_fps"]),
        seed_push_to_emit_p99_seconds=seed_p99,
        p99_latency_ratio=p99 / seed_p99,
        seed_ttff_seconds=seed_ttff,
        ttff_limit_seconds=max(2.5 * seed_ttff, seed_ttff + 4),
        p99_limit_seconds=3 * seed_p99,
        seed_construction_seconds=seed_construction,
        construction_limit_seconds=max(
            CONSTRUCTION_FACTOR * seed_construction,
            seed_construction + CONSTRUCTION_ALLOWANCE_SECONDS,
        ),
        case_score=fps * (seed_p99 / p99) ** LATENCY_EXPONENT,
    )
    return row


def timing_gates(row):
    return {
        "ttff_ok": positive(row["ttff_seconds"]) <= positive(row["ttff_limit_seconds"]),
        "p99_ok": positive(row["push_to_emit_p99_seconds"]) <= positive(row["p99_limit_seconds"]),
        "construction_ok": positive(row["construction_seconds"]) <= positive(row["construction_limit_seconds"]),
    }


def memory_cap_mib(seed_peaks):
    """Per-rank cap, identical for every rank: the seed's larger peak + margin."""
    if not isinstance(seed_peaks, dict) or set(seed_peaks) != set(RANK_KEYS):
        raise ValueError("seed peak memory must be keyed by every rank")
    return max(positive(seed_peaks[rank]) for rank in RANK_KEYS) + GPU_MEMORY_MARGIN_MIB


def memory_ok(peaks, seed):
    baseline = seed.get("peak_memory_mib_per_rank")
    if not isinstance(peaks, dict) or not isinstance(baseline, dict):
        return False
    if set(peaks) != set(RANK_KEYS) or set(baseline) != set(RANK_KEYS):
        return False
    cap = memory_cap_mib(baseline)
    # A rank the candidate never touches may read 0 MiB; that is legal.
    return all(nonnegative(peaks[rank]) <= cap for rank in RANK_KEYS)


def post_close_memory_ok(post_close, baseline):
    if not isinstance(post_close, dict) or not isinstance(baseline, dict):
        return False
    if set(post_close) != set(RANK_KEYS) or set(baseline) != set(RANK_KEYS):
        return False
    for rank in RANK_KEYS:
        try:
            used = float(post_close[rank])
            floor = float(baseline[rank])
        except (TypeError, ValueError):
            return False
        if not math.isfinite(used) or not math.isfinite(floor) or used < 0 or floor < 0:
            return False
        if used > floor + POST_CLOSE_MEMORY_MARGIN_MIB:
            return False
    return True


def suite_metrics(rows):
    # Derive from measurements; never trust a supplied case_score.
    scores = [positive(row["steady_fps"]) * (positive(row["seed_push_to_emit_p99_seconds"]) / positive(row["push_to_emit_p99_seconds"])) ** LATENCY_EXPONENT for row in rows]
    return {
        "primary_score_name": PRIMARY_SCORE_NAME,
        "primary_score_unit": PRIMARY_SCORE_UNIT,
        "score": geomean(scores),
        "fps_geomean": geomean(row["steady_fps"] for row in rows),
        "p99_geomean_ratio": geomean(positive(row["push_to_emit_p99_seconds"]) / positive(row["seed_push_to_emit_p99_seconds"]) for row in rows),
    }
