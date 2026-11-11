from __future__ import annotations

import math
import re
import statistics
from collections.abc import Mapping
from typing import Any, Iterable


def percentile(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def jain(values: Iterable[float]) -> float:
    numbers = list(values)
    if not numbers or not any(numbers):
        return 0.0
    numerator = sum(numbers) ** 2
    denominator = len(numbers) * sum(value * value for value in numbers)
    return numerator / denominator if denominator else 0.0


def geometric_mean(values: Iterable[float]) -> float:
    numbers = [float(value) for value in values]
    if not numbers or any(value <= 0 for value in numbers):
        return 0.0
    return math.exp(sum(math.log(value) for value in numbers) / len(numbers))


def coefficient_of_variation(values: Iterable[float]) -> float:
    numbers = [float(value) for value in values]
    if len(numbers) < 2:
        return 0.0
    mean = statistics.fmean(numbers)
    return statistics.pstdev(numbers) / mean if mean else math.inf


def _metric_suffix(name: str) -> str:
    """Normalize both current ``vllm:`` and legacy ``vllm_`` names."""
    if name.startswith("vllm:"):
        return name.removeprefix("vllm:")
    if name.startswith("vllm_"):
        return name.removeprefix("vllm_")
    return name


def _positive_integer_label(labels: str, name: str) -> int | None:
    """Extract one unambiguous positive integer Prometheus label value."""

    matches = re.findall(rf'(?:^|,){re.escape(name)}="([0-9]+)"(?:,|$)', labels)
    if len(matches) != 1:
        return None
    value = int(matches[0])
    return value if value > 0 else None


def metric_point(snapshot: Mapping[str, object]) -> dict[str, float | None]:
    """Extract the small set of live gauges needed for pressure validation.

    A complete Prometheus snapshot is intentionally not retained by the
    workload runner.  Apart from making long runs unnecessarily large, raw
    labels can disclose deployment details in public feedback.  Multiple
    engine-labelled KV gauges are combined with ``max``; request gauges are
    also represented by their maximum because some vLLM deployments export a
    replica of the scheduler gauge for every worker.
    """

    values: dict[str, list[float]] = {
        "kv_cache_usage": [],
        "requests_running": [],
        "requests_waiting": [],
    }
    token_capacities: set[int] = set()
    for sample in snapshot.get("samples", []):  # type: ignore[union-attr]
        if not isinstance(sample, dict):
            continue
        name = sample.get("name")
        value = sample.get("value")
        if not isinstance(name, str) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if not math.isfinite(number):
            continue
        suffix = _metric_suffix(name)
        if suffix in {"kv_cache_usage_perc", "gpu_cache_usage_perc"}:
            values["kv_cache_usage"].append(number)
        elif suffix in {"num_requests_running", "requests_running"}:
            values["requests_running"].append(number)
        elif suffix in {"num_requests_waiting", "requests_waiting"}:
            values["requests_waiting"].append(number)
        elif suffix == "cache_config_info":
            labels = sample.get("labels")
            if not isinstance(labels, str):
                continue
            # Hybrid models expose an alignment page size as ``block_size``;
            # multiplying it by num_gpu_blocks overstates logical token
            # capacity. vLLM 0.24 publishes the authoritative scheduler value.
            capacity = _positive_integer_label(labels, "kv_cache_size_tokens")
            if capacity is not None:
                token_capacities.add(capacity)
    created = snapshot.get("created_unix")
    return {
        "created_unix": float(created) if isinstance(created, (int, float)) else None,
        **{key: max(numbers) if numbers else None for key, numbers in values.items()},
        # Capacity is in logical scheduler tokens. Equal per-worker labels
        # collapse; disagreement or missing labels remain fail-closed.
        "kv_token_capacity": (
            float(next(iter(token_capacities))) if len(token_capacities) == 1 else None
        ),
    }


def summarize_metric_series(
    points: Iterable[Mapping[str, float | None]],
    *,
    pressure_threshold: float,
    expected_sample_interval_sec: float,
    sampler_errors: Iterable[str] = (),
    request_intervals: Iterable[Mapping[str, object]] = (),
) -> dict[str, Any]:
    """Summarize live metrics captured while workflows are active.

    Prefix-cached blocks remain allocated while requests execute shell
    commands. Those blocks may be evictable, but sustained high occupancy is
    the state this benchmark is designed to measure. Active-working-set,
    waiting, churn, and preemption signals characterize that state without
    gating it. Finite callers seal this series at the last ``agent.run``
    termination; repository semantic verification and cleanup are teardown,
    not live agent metrics. Pressure percentiles and durations are consequently
    conditioned on the backend having at least one running or waiting request.
    The unconditioned high-water marks are kept as ``retained_*`` diagnostics
    so calibration can still audit cache residency.
    """

    if (
        not isinstance(expected_sample_interval_sec, (int, float))
        or not math.isfinite(float(expected_sample_interval_sec))
        or float(expected_sample_interval_sec) <= 0
    ):
        raise ValueError("expected_sample_interval_sec must be finite and positive")
    expected_interval = float(expected_sample_interval_sec)
    # Three missed polling periods is an evidence gap, independent of the
    # HTTP request timeout.  A request may legally wait much longer for I/O,
    # but its preceding gauge value must not be projected across that wait.
    sampling_gap_limit = expected_interval * 3.0

    samples = list(points)
    retained_kv = [
        float(point["kv_cache_usage"])
        for point in samples
        if point.get("kv_cache_usage") is not None
    ]

    def backend_active(point: Mapping[str, float | None]) -> bool:
        return any(
            isinstance(point.get(key), (int, float)) and float(point[key]) > 0
            for key in ("requests_running", "requests_waiting")
        )

    active_kv = [
        float(point["kv_cache_usage"])
        for point in samples
        if point.get("kv_cache_usage") is not None and backend_active(point)
    ]
    timed_points = [
        point
        for point in samples
        if isinstance(point.get("created_unix"), (int, float))
        and math.isfinite(float(point["created_unix"]))
    ]

    # The trusted model-api audit supplies exact tokenizer counts after each
    # request completes.  Matching its lifetime to a sample reconstructs the
    # prompt tokens offered at that instant. Prompt tokens are a conservative
    # lower bound: tokens generated after the sample are intentionally not
    # counted. ``total_tokens`` is retained only as an upper-bound diagnostic.
    intervals = [
        row
        for row in request_intervals
        if isinstance(row.get("started_unix"), (int, float))
        and not isinstance(row.get("started_unix"), bool)
        and math.isfinite(float(row["started_unix"]))
        and isinstance(row.get("ended_unix"), (int, float))
        and not isinstance(row.get("ended_unix"), bool)
        and math.isfinite(float(row["ended_unix"]))
        and float(row["ended_unix"]) >= float(row["started_unix"])
    ]

    def audited_tokens(
        rows: list[Mapping[str, object]], key: str
    ) -> float | None:
        if not rows:
            return None
        values = [row.get(key) for row in rows]
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) >= 0
            for value in values
        ):
            return None
        return sum(float(value) for value in values)  # type: ignore[arg-type]

    sample_evidence: list[dict[str, float | int | None]] = []
    for point in samples:
        created = point.get("created_unix")
        in_flight: list[Mapping[str, object]] = []
        if (
            backend_active(point)
            and isinstance(created, (int, float))
            and math.isfinite(float(created))
        ):
            in_flight = [
                row
                for row in intervals
                if float(row["started_unix"])
                <= float(created)
                <= float(row["ended_unix"])
            ]
        prompt = audited_tokens(in_flight, "prompt_tokens")
        total = audited_tokens(in_flight, "total_tokens")
        capacity = point.get("kv_token_capacity")
        ratio = (
            prompt / float(capacity)
            if prompt is not None
            and isinstance(capacity, (int, float))
            and math.isfinite(float(capacity))
            and float(capacity) > 0
            else None
        )
        sample_evidence.append(
            {
                "request_count": len(in_flight),
                "prompt_tokens": prompt,
                "total_tokens": total,
                "prompt_tokens_to_capacity": ratio,
            }
        )

    timestamp_error_count = len(samples) - len(timed_points)
    sample_intervals: list[float] = []
    sampling_gaps: list[float] = []
    sampler_span = 0.0
    sampler_duration = 0.0
    active_duration = 0.0
    pressurized_duration = 0.0
    waiting_duration = 0.0
    active_working_set_pressure_duration = 0.0
    oversubscribed_working_set_duration = 0.0
    oversubscribed_working_set_waiting_duration = 0.0
    pressurized_waiting_duration = 0.0
    for index, (point, next_point) in enumerate(zip(samples, samples[1:])):
        created = point.get("created_unix")
        next_created = next_point.get("created_unix")
        if not (
            isinstance(created, (int, float))
            and math.isfinite(float(created))
            and isinstance(next_created, (int, float))
            and math.isfinite(float(next_created))
        ):
            # Never bridge over a sample whose timestamp cannot anchor the
            # state transition. The malformed point is counted above and
            # makes required pressure evidence invalid.
            continue
        duration = float(next_created) - float(created)
        if not math.isfinite(duration) or duration <= 0:
            timestamp_error_count += 1
            continue
        sample_intervals.append(duration)
        sampler_span += duration
        if duration > sampling_gap_limit:
            sampling_gaps.append(duration)
            # Missing time has no trustworthy gauge value. Excluding the
            # whole interval is deliberately conservative.
            continue
        sampler_duration += duration
        if not backend_active(point):
            continue
        active_duration += duration
        waiting_value = point.get("requests_waiting")
        waiting_now = (
            isinstance(waiting_value, (int, float)) and float(waiting_value) > 0
        )
        if waiting_now:
            waiting_duration += duration
        # This is offered logical demand, not retained physical occupancy.
        # Once eviction starts, vLLM's occupancy gauge may fall even while the
        # set of concurrently live prompts cannot fit in the cache. Keep this
        # predicate independent of the occupancy gauge so that such thrashing
        # remains observable from the trusted per-request token audit.
        ratio = sample_evidence[index]["prompt_tokens_to_capacity"]
        oversubscribed = isinstance(ratio, (int, float)) and ratio >= 1.0
        if oversubscribed:
            oversubscribed_working_set_duration += duration
            if waiting_now:
                oversubscribed_working_set_waiting_duration += duration
        kv_value = point.get("kv_cache_usage")
        high_kv = (
            isinstance(kv_value, (int, float))
            and float(kv_value) >= pressure_threshold
        )
        if not high_kv:
            continue
        pressurized_duration += duration
        if waiting_now:
            # This is a synchronized predicate, not an overlap inferred from
            # independent waiting and occupancy maxima.
            pressurized_waiting_duration += duration
        if isinstance(ratio, (int, float)) and ratio >= pressure_threshold:
            active_working_set_pressure_duration += duration

    offered_prompt_tokens: list[float] = []
    offered_total_tokens: list[float] = []
    offered_request_counts: list[float] = []
    offered_prompt_token_ratios: list[float] = []
    token_demand_incomplete_samples = 0
    capacity_incomplete_samples = 0
    high_kv_samples = 0
    active_working_set_pressure_samples = 0
    oversubscribed_working_set_samples = 0
    oversubscribed_working_set_waiting_samples = 0
    pressurized_waiting_samples = 0
    for point, evidence in zip(samples, sample_evidence):
        if not backend_active(point):
            continue
        request_count = int(evidence["request_count"] or 0)
        prompt = evidence["prompt_tokens"]
        total = evidence["total_tokens"]
        ratio = evidence["prompt_tokens_to_capacity"]
        if request_count:
            offered_request_counts.append(float(request_count))
        if isinstance(prompt, (int, float)):
            offered_prompt_tokens.append(float(prompt))
        else:
            token_demand_incomplete_samples += 1
        if isinstance(total, (int, float)):
            offered_total_tokens.append(float(total))
        if isinstance(ratio, (int, float)):
            offered_prompt_token_ratios.append(float(ratio))
        capacity = point.get("kv_token_capacity")
        if not (
            isinstance(capacity, (int, float))
            and math.isfinite(float(capacity))
            and float(capacity) > 0
        ):
            capacity_incomplete_samples += 1
        if isinstance(ratio, (int, float)) and ratio >= 1.0:
            oversubscribed_working_set_samples += 1
            waiting_value = point.get("requests_waiting")
            if (
                isinstance(waiting_value, (int, float))
                and float(waiting_value) > 0
            ):
                oversubscribed_working_set_waiting_samples += 1
        kv_value = point.get("kv_cache_usage")
        if not (
            isinstance(kv_value, (int, float))
            and float(kv_value) >= pressure_threshold
        ):
            continue
        high_kv_samples += 1
        if isinstance(ratio, (int, float)) and ratio >= pressure_threshold:
            active_working_set_pressure_samples += 1
        waiting_value = point.get("requests_waiting")
        if isinstance(waiting_value, (int, float)) and float(waiting_value) > 0:
            pressurized_waiting_samples += 1

    running = [
        float(point["requests_running"])
        for point in samples
        if point.get("requests_running") is not None
    ]
    waiting = [
        float(point["requests_waiting"])
        for point in samples
        if point.get("requests_waiting") is not None
    ]
    capacities = {
        int(float(point["kv_token_capacity"]))
        for point in samples
        if isinstance(point.get("kv_token_capacity"), (int, float))
        and math.isfinite(float(point["kv_token_capacity"]))
        and float(point["kv_token_capacity"]) > 0
    }
    errors = list(sampler_errors)
    return {
        "sample_count": len(samples),
        "timed_sample_count": len(timed_points),
        "kv_sample_count": len(active_kv),
        "retained_kv_sample_count": len(retained_kv),
        "backend_active_sample_count": sum(backend_active(point) for point in samples),
        "kv_cache_usage_p50": percentile(active_kv, 0.50),
        "kv_cache_usage_p95": percentile(active_kv, 0.95),
        "kv_cache_usage_max": max(active_kv) if active_kv else None,
        "retained_kv_cache_usage_p50": percentile(retained_kv, 0.50),
        "retained_kv_cache_usage_p95": percentile(retained_kv, 0.95),
        "retained_kv_cache_usage_max": max(retained_kv) if retained_kv else None,
        "pressure_threshold": pressure_threshold,
        "expected_sample_interval_sec": expected_interval,
        "sampling_gap_limit_sec": sampling_gap_limit,
        "sample_interval_sec_p95": percentile(sample_intervals, 0.95),
        "sample_interval_sec_max": max(sample_intervals) if sample_intervals else None,
        "sampling_gap_count": len(sampling_gaps),
        "sampling_gap_max_sec": max(sampling_gaps) if sampling_gaps else None,
        "sampling_gap_total_sec": sum(sampling_gaps),
        "sampling_timestamp_error_count": timestamp_error_count,
        "sample_fraction_at_or_above_pressure_threshold": (
            sum(value >= pressure_threshold for value in active_kv) / len(active_kv)
            if active_kv
            else None
        ),
        "sampler_span_sec": sampler_span or None,
        "sampler_duration_sec": sampler_duration or None,
        "observed_metric_duration_sec": active_duration or None,
        "backend_active_duration_sec": active_duration or None,
        "backend_active_time_fraction": (
            active_duration / sampler_duration if sampler_duration else None
        ),
        "pressurized_duration_sec": pressurized_duration if active_duration else None,
        "fraction_at_or_above_pressure_threshold": (
            pressurized_duration / active_duration if active_duration else None
        ),
        "waiting_duration_sec": waiting_duration if active_duration else None,
        "fraction_backend_active_time_with_waiting": (
            waiting_duration / active_duration if active_duration else None
        ),
        "kv_token_capacity": next(iter(capacities)) if len(capacities) == 1 else None,
        "kv_token_capacity_unit": "logical_tokens",
        "kv_token_capacity_source": "vllm:cache_config_info.kv_cache_size_tokens",
        "kv_token_capacity_sample_count": sum(
            isinstance(point.get("kv_token_capacity"), (int, float))
            and float(point["kv_token_capacity"]) > 0
            for point in samples
        ),
        "kv_token_capacity_inconsistent": len(capacities) > 1,
        "kv_token_capacity_incomplete_active_sample_count": capacity_incomplete_samples,
        "offered_token_demand_sample_count": len(offered_prompt_tokens),
        "offered_token_demand_incomplete_sample_count": token_demand_incomplete_samples,
        "offered_request_count_p50": percentile(offered_request_counts, 0.50),
        "offered_request_count_p95": percentile(offered_request_counts, 0.95),
        "offered_request_count_max": max(offered_request_counts) if offered_request_counts else None,
        "offered_prompt_tokens_p50": percentile(offered_prompt_tokens, 0.50),
        "offered_prompt_tokens_p95": percentile(offered_prompt_tokens, 0.95),
        "offered_prompt_tokens_max": max(offered_prompt_tokens) if offered_prompt_tokens else None,
        "offered_total_tokens_p50": percentile(offered_total_tokens, 0.50),
        "offered_total_tokens_p95": percentile(offered_total_tokens, 0.95),
        "offered_total_tokens_max": max(offered_total_tokens) if offered_total_tokens else None,
        "offered_prompt_tokens_to_capacity_p50": percentile(offered_prompt_token_ratios, 0.50),
        "offered_prompt_tokens_to_capacity_p95": percentile(offered_prompt_token_ratios, 0.95),
        "offered_prompt_tokens_to_capacity_max": (
            max(offered_prompt_token_ratios) if offered_prompt_token_ratios else None
        ),
        "high_kv_sample_count": high_kv_samples,
        "active_working_set_pressure_sample_count": active_working_set_pressure_samples,
        "active_working_set_pressure_duration_sec": (
            active_working_set_pressure_duration if active_duration else None
        ),
        "fraction_backend_active_time_with_active_working_set_pressure": (
            active_working_set_pressure_duration / active_duration
            if active_duration
            else None
        ),
        "oversubscribed_working_set_sample_count": (
            oversubscribed_working_set_samples
        ),
        "oversubscribed_working_set_duration_sec": (
            oversubscribed_working_set_duration if active_duration else None
        ),
        "fraction_backend_active_time_with_oversubscribed_working_set": (
            oversubscribed_working_set_duration / active_duration
            if active_duration
            else None
        ),
        "oversubscribed_working_set_waiting_sample_count": (
            oversubscribed_working_set_waiting_samples
        ),
        "oversubscribed_working_set_waiting_duration_sec": (
            oversubscribed_working_set_waiting_duration
            if active_duration
            else None
        ),
        "fraction_backend_active_time_with_oversubscribed_working_set_waiting": (
            oversubscribed_working_set_waiting_duration / active_duration
            if active_duration
            else None
        ),
        "pressurized_waiting_sample_count": pressurized_waiting_samples,
        "pressurized_waiting_duration_sec": (
            pressurized_waiting_duration if active_duration else None
        ),
        "fraction_backend_active_time_pressurized_with_waiting": (
            pressurized_waiting_duration / active_duration if active_duration else None
        ),
        "requests_running_p50": percentile(running, 0.50),
        "requests_running_p95": percentile(running, 0.95),
        "requests_running_max": max(running) if running else None,
        "requests_waiting_p50": percentile(waiting, 0.50),
        "requests_waiting_p95": percentile(waiting, 0.95),
        "requests_waiting_max": max(waiting) if waiting else None,
        "sampler_error_count": len(errors),
        "sampler_errors": errors[:5],
    }


def assess_pressure(
    series: Mapping[str, object],
    *,
    required: bool,
    minimum_samples: int,
    minimum_kv_p95: float,
    minimum_time_fraction: float,
    counter_diagnostics: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Return a sampling-completeness plus high-KV occupancy verdict.

    A required baseline qualifies when its live metric series is complete, has
    enough samples, and demonstrates sustained high KV occupancy. A sustained
    audited oversubscribed working set corroborated by synchronized waiting or
    substantial leg-local cache churn remains an eviction-safe alternative to
    the physical occupancy gauge. Active-working-set, waiting, preemption, and
    contention signals are retained as diagnostics only; none is independently
    required for, or sufficient to establish, pressure validity.
    """

    reasons: list[str] = []
    sample_count = int(series.get("kv_sample_count") or 0)
    kv_p95 = series.get("kv_cache_usage_p95")
    time_fraction = series.get("fraction_at_or_above_pressure_threshold")
    evidence_minimum_samples = max(
        1, math.ceil(minimum_samples * minimum_time_fraction)
    )

    def duration_path(
        *,
        sample_key: str,
        fraction_key: str,
        label: str,
        extra_reasons: Iterable[str] = (),
    ) -> dict[str, Any]:
        path_reasons = list(extra_reasons)
        evidence_samples = int(series.get(sample_key) or 0)
        fraction = series.get(fraction_key)
        if evidence_samples < evidence_minimum_samples:
            path_reasons.append(
                f"insufficient_{label}_samples:"
                f"{evidence_samples}<{evidence_minimum_samples}"
            )
        if (
            not isinstance(fraction, (int, float))
            or float(fraction) < minimum_time_fraction
        ):
            rendered = "missing" if fraction is None else f"{float(fraction):.4f}"
            path_reasons.append(
                f"{label}_duration_below_floor:"
                f"{rendered}<{minimum_time_fraction:.4f}"
            )
        return {
            "valid": not path_reasons,
            "sample_count": evidence_samples,
            "minimum_samples": evidence_minimum_samples,
            "backend_active_time_fraction": fraction,
            "minimum_backend_active_time_fraction": minimum_time_fraction,
            "reasons": path_reasons,
        }

    capacity = series.get("kv_token_capacity")
    working_set_extra: list[str] = []
    if not isinstance(capacity, (int, float)) or float(capacity) <= 0:
        working_set_extra.append("missing_or_ambiguous_kv_token_capacity")
    if bool(series.get("kv_token_capacity_inconsistent")):
        working_set_extra.append("inconsistent_kv_token_capacity")
    if int(series.get("kv_token_capacity_incomplete_active_sample_count") or 0):
        working_set_extra.append("incomplete_active_capacity_samples")
    active_working_set = duration_path(
        sample_key="active_working_set_pressure_sample_count",
        fraction_key="fraction_backend_active_time_with_active_working_set_pressure",
        label="active_working_set_pressure",
        extra_reasons=working_set_extra,
    )
    active_working_set.update(
        {
            "demand_basis": "audited_in_flight_prompt_tokens_lower_bound",
            "capacity": capacity,
            "capacity_unit": series.get("kv_token_capacity_unit"),
            "capacity_source": series.get("kv_token_capacity_source"),
            "demand_ratio_threshold": series.get("pressure_threshold"),
            "demand_ratio_p95": series.get("offered_prompt_tokens_to_capacity_p95"),
            "demand_ratio_max": series.get("offered_prompt_tokens_to_capacity_max"),
            "demand_incomplete_sample_count": series.get(
                "offered_token_demand_incomplete_sample_count"
            ),
            "limitation": (
                "generation growth at each sample is unavailable; prompt tokens are "
                "a conservative lower bound and final total tokens are not used"
            ),
        }
    )
    oversubscribed_base = duration_path(
        sample_key="oversubscribed_working_set_sample_count",
        fraction_key=(
            "fraction_backend_active_time_with_oversubscribed_working_set"
        ),
        label="oversubscribed_working_set",
        extra_reasons=working_set_extra,
    )
    oversubscribed_waiting = duration_path(
        sample_key="oversubscribed_working_set_waiting_sample_count",
        fraction_key=(
            "fraction_backend_active_time_with_oversubscribed_working_set_waiting"
        ),
        label="oversubscribed_working_set_waiting",
    )
    oversubscribed_waiting["predicate"] = (
        "audited_in_flight_prompt_tokens>=logical_kv_capacity AND "
        "requests_waiting>0 at the same sample"
    )

    diagnostics = counter_diagnostics or {}

    def finite_nonnegative(name: str) -> float | None:
        value = diagnostics.get(name)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) >= 0
        ):
            return float(value)
        return None

    # These are after-before deltas for this one workload leg. Do not use the
    # hit *ratio* as proof: a cold cache or unrelated short traffic can have a
    # low ratio. Instead require observed reuse, at least one full capacity of
    # prompt misses, and at least two capacities of total KV writes while the
    # independently audited live prompt set remains oversubscribed.
    prefix_hits = finite_nonnegative("prefix_cache_hits")
    prefix_queries = finite_nonnegative("prefix_cache_queries")
    generation_tokens = finite_nonnegative("generation_tokens")
    cache_churn_reasons: list[str] = []
    cache_miss_tokens: float | None = None
    estimated_kv_write_tokens: float | None = None
    if prefix_hits is None or prefix_queries is None:
        cache_churn_reasons.append("missing_leg_local_prefix_cache_counters")
    elif prefix_hits > prefix_queries:
        cache_churn_reasons.append("invalid_prefix_cache_counter_delta")
    else:
        cache_miss_tokens = prefix_queries - prefix_hits
    if generation_tokens is None:
        cache_churn_reasons.append("missing_leg_local_generation_token_counter")
    elif cache_miss_tokens is not None:
        estimated_kv_write_tokens = cache_miss_tokens + generation_tokens
    if prefix_hits is not None and prefix_hits <= 0:
        cache_churn_reasons.append("no_leg_local_cache_reuse_observed")
    minimum_cache_miss_tokens = (
        float(capacity)
        if isinstance(capacity, (int, float)) and float(capacity) > 0
        else None
    )
    minimum_kv_write_tokens = (
        2.0 * float(capacity)
        if isinstance(capacity, (int, float)) and float(capacity) > 0
        else None
    )
    if minimum_cache_miss_tokens is None:
        cache_churn_reasons.append("missing_or_ambiguous_kv_token_capacity")
    elif (
        cache_miss_tokens is None
        or cache_miss_tokens < minimum_cache_miss_tokens
    ):
        rendered = (
            "missing" if cache_miss_tokens is None else f"{cache_miss_tokens:.0f}"
        )
        cache_churn_reasons.append(
            "leg_local_cache_miss_tokens_below_floor:"
            f"{rendered}<{minimum_cache_miss_tokens:.0f}"
        )
    if minimum_kv_write_tokens is not None and (
        estimated_kv_write_tokens is None
        or estimated_kv_write_tokens < minimum_kv_write_tokens
    ):
        rendered = (
            "missing"
            if estimated_kv_write_tokens is None
            else f"{estimated_kv_write_tokens:.0f}"
        )
        cache_churn_reasons.append(
            "leg_local_kv_write_tokens_below_floor:"
            f"{rendered}<{minimum_kv_write_tokens:.0f}"
        )
    cache_churn = {
        "valid": not cache_churn_reasons,
        "prefix_cache_hits_delta": prefix_hits,
        "prefix_cache_queries_delta": prefix_queries,
        "cache_miss_prompt_tokens_delta": cache_miss_tokens,
        "generation_tokens_delta": generation_tokens,
        "estimated_kv_write_tokens_delta": estimated_kv_write_tokens,
        "minimum_cache_miss_prompt_tokens": minimum_cache_miss_tokens,
        "minimum_estimated_kv_write_tokens": minimum_kv_write_tokens,
        "requires_nonzero_prefix_cache_hits": True,
        "source": "vllm monotonic counter after-before deltas for this leg",
        "reasons": cache_churn_reasons,
    }
    oversubscription_corroboration_paths = [
        label
        for label, path in (
            ("synchronized_waiting", oversubscribed_waiting),
            ("leg_local_cache_churn", cache_churn),
        )
        if path["valid"]
    ]
    oversubscribed_reasons = list(oversubscribed_base["reasons"])
    if not oversubscription_corroboration_paths:
        oversubscribed_reasons.append(
            "missing_synchronized_waiting_or_leg_local_cache_churn"
        )
    oversubscribed_working_set = {
        **oversubscribed_base,
        "valid": (
            oversubscribed_base["valid"]
            and bool(oversubscription_corroboration_paths)
        ),
        "reasons": oversubscribed_reasons,
        "demand_basis": "audited_in_flight_prompt_tokens_lower_bound",
        "demand_ratio_threshold": 1.0,
        "demand_ratio_p95": series.get("offered_prompt_tokens_to_capacity_p95"),
        "demand_ratio_max": series.get("offered_prompt_tokens_to_capacity_max"),
        "capacity": capacity,
        "capacity_unit": series.get("kv_token_capacity_unit"),
        "capacity_source": series.get("kv_token_capacity_source"),
        "occupancy_independent": True,
        "corroboration_paths": oversubscription_corroboration_paths,
        "synchronized_waiting": oversubscribed_waiting,
        "leg_local_cache_churn": cache_churn,
        "limitation": (
            "generation growth at each sample is unavailable; prompt tokens "
            "are a conservative lower bound"
        ),
    }
    waiting_at_high_kv = duration_path(
        sample_key="pressurized_waiting_sample_count",
        fraction_key="fraction_backend_active_time_pressurized_with_waiting",
        label="pressurized_waiting",
    )
    waiting_at_high_kv["predicate"] = (
        "kv_cache_usage>=pressure_threshold AND requests_waiting>0 at same sample"
    )
    waiting_at_high_kv["qualification_role"] = (
        "diagnostic_only; waiting may reflect max-num-seqs or prefill admission"
    )
    preemptions = diagnostics.get("preemptions")
    preemption = {
        "valid": (
            isinstance(preemptions, (int, float))
            and not isinstance(preemptions, bool)
            and math.isfinite(float(preemptions))
            and float(preemptions) > 0
        ),
        "preemptions_delta": preemptions,
        "source": "vllm:num_preemptions_total after-before for this leg",
    }
    diagnostic_paths = [
        label
        for label, path in (
            ("audited_active_working_set", active_working_set),
            ("audited_oversubscribed_working_set", oversubscribed_working_set),
            ("vllm_preemption", preemption),
        )
        if path["valid"]
    ]
    contention_evidence = {
        "required": False,
        "valid": True,
        "status": "diagnostic_only",
        "qualification_role": "diagnostic_only",
        "signal_observed": bool(diagnostic_paths),
        "diagnostic_paths": diagnostic_paths,
        "active_working_set": active_working_set,
        "oversubscribed_working_set": oversubscribed_working_set,
        "waiting_at_high_kv": waiting_at_high_kv,
        "preemption": preemption,
        # Cache churn is reported only as corroboration for an independently
        # audited, sustained oversubscribed live set. It is not called
        # recompute, which still needs a dedicated monotonic counter.
        "not_inferred": ["recompute", "cross_profile_contention_knee"],
    }
    observed_occupancy_reasons: list[str] = []
    if not isinstance(kv_p95, (int, float)) or float(kv_p95) < minimum_kv_p95:
        rendered = "missing" if kv_p95 is None else f"{float(kv_p95):.4f}"
        observed_occupancy_reasons.append(
            f"kv_p95_below_floor:{rendered}<{minimum_kv_p95:.4f}"
        )
    if (
        not isinstance(time_fraction, (int, float))
        or float(time_fraction) < minimum_time_fraction
    ):
        rendered = (
            "missing" if time_fraction is None else f"{float(time_fraction):.4f}"
        )
        observed_occupancy_reasons.append(
            "pressure_duration_below_floor:"
            f"{rendered}<{minimum_time_fraction:.4f}"
        )
    observed_occupancy_valid = not observed_occupancy_reasons
    occupancy_evidence = {
        "valid": (
            observed_occupancy_valid or oversubscribed_working_set["valid"]
        ),
        "qualification": (
            "observed_kv_occupancy"
            if observed_occupancy_valid
            else (
                "audited_oversubscription_with_contention"
                if oversubscribed_working_set["valid"]
                else "not_qualified"
            )
        ),
        "observed_kv_occupancy": {
            "valid": observed_occupancy_valid,
            "kv_p95": kv_p95,
            "minimum_kv_p95": minimum_kv_p95,
            "time_fraction": time_fraction,
            "minimum_time_fraction": minimum_time_fraction,
            "reasons": observed_occupancy_reasons,
        },
        "eviction_safe_alternative": "audited_oversubscribed_working_set",
    }
    if required:
        if int(series.get("sampler_error_count") or 0):
            reasons.append("metrics_sampler_error")
        if int(series.get("sampling_gap_count") or 0):
            reasons.append("metrics_sampling_gap")
        if int(series.get("sampling_timestamp_error_count") or 0):
            reasons.append("metrics_sampling_timestamp_error")
        if sample_count < minimum_samples:
            reasons.append(f"insufficient_kv_samples:{sample_count}<{minimum_samples}")
        if not occupancy_evidence["valid"]:
            reasons.extend(observed_occupancy_reasons)
    return {
        "qualification_schema_version": 2,
        "gate_basis": "sampling_completeness_and_high_kv_occupancy",
        "required": required,
        "valid": not reasons,
        "status": "pressurized" if required and not reasons else ("profile_not_pressurized" if reasons else "not_required"),
        "minimum_samples": minimum_samples,
        "minimum_kv_p95": minimum_kv_p95,
        "minimum_time_fraction": minimum_time_fraction,
        "occupancy_evidence": occupancy_evidence,
        "contention_evidence": contention_evidence,
        "reasons": reasons,
    }


def summarize_metric_delta(
    before: Mapping[str, object],
    after: Mapping[str, object],
) -> dict[str, object]:
    def sample_map(snapshot: Mapping[str, object]) -> dict[tuple[str, str], float]:
        result: dict[tuple[str, str], float] = {}
        for sample in snapshot.get("samples", []):  # type: ignore[union-attr]
            if not isinstance(sample, dict):
                continue
            name = sample.get("name")
            labels = sample.get("labels", "")
            value = sample.get("value")
            if isinstance(name, str) and isinstance(labels, str) and isinstance(value, (int, float)):
                result[(name, labels)] = float(value)
        return result

    start = sample_map(before)
    end = sample_map(after)
    counters = []
    gauges = []
    for (name, labels), value in sorted(end.items()):
        record = {"name": name, "labels": labels}
        if name.endswith(("_total", "_count", "_sum")) and (name, labels) in start:
            delta = value - start[(name, labels)]
            if math.isfinite(delta):
                counters.append(record | {"delta": delta})
        elif math.isfinite(value):
            gauges.append(record | {"value": value})
    def counter_value(*suffixes: str) -> float | None:
        values = [
            float(sample["delta"])
            for sample in counters
            if _metric_suffix(str(sample["name"])) in suffixes
        ]
        return sum(values) if values else None

    prefix_hits = counter_value("prefix_cache_hits_total")
    prefix_queries = counter_value("prefix_cache_queries_total")
    prompt_tokens = counter_value("prompt_tokens_total")
    generation_tokens = counter_value("generation_tokens_total")
    computed_prompt_tokens = (
        max(0.0, prefix_queries - prefix_hits)
        if prefix_hits is not None and prefix_queries is not None
        else None
    )
    estimated_kv_write_tokens = (
        computed_prompt_tokens + generation_tokens
        if computed_prompt_tokens is not None and generation_tokens is not None
        else None
    )
    return {
        "before_raw_sha256": before.get("raw_sha256"),
        "after_raw_sha256": after.get("raw_sha256"),
        "counter_deltas": counters,
        "gauge_end": gauges,
        "derived": {
            "prefix_cache_hits": prefix_hits,
            "prefix_cache_queries": prefix_queries,
            "prefix_cache_hit_ratio": (
                prefix_hits / prefix_queries
                if prefix_hits is not None and prefix_queries is not None and prefix_queries > 0
                else None
            ),
            # Each cache-miss prompt token and generated token requires a KV
            # write.  This is a calibration demand estimate, not live
            # occupancy: evictions can make the same logical prefix miss more
            # than once.
            "computed_prompt_tokens": computed_prompt_tokens,
            "estimated_kv_write_tokens": estimated_kv_write_tokens,
            "preemptions": counter_value("num_preemptions_total"),
            "prompt_tokens": prompt_tokens,
            "generation_tokens": generation_tokens,
        },
    }
