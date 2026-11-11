"""Judge-only benchmark primitives for correctness, memory, and timing."""

from __future__ import annotations

import importlib.util
import math
import statistics
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import torch

from benchmark_cases import (
    ASSIGN_WEIGHT,
    FULL_WEIGHT,
    RUNTIME_WEIGHT,
    UPDATE_WEIGHT,
    BenchmarkCase,
)


MIB = 1024 * 1024
STAGE_WEIGHTS = {
    "assign": ASSIGN_WEIGHT,
    "update": UPDATE_WEIGHT,
    "full": FULL_WEIGHT,
}
REFERENCE_BLOCK_N = 2048
REFERENCE_BLOCK_K = 128
# Capture trusted timer/synchronization callables before candidate import.
DEVICE_SYNCHRONIZE = torch.cuda.synchronize
WALL_CLOCK = time.perf_counter
MEMORY_ALLOCATED = torch.cuda.memory_allocated
RESET_PEAK_MEMORY = torch.cuda.reset_peak_memory_stats
MAX_MEMORY_ALLOCATED = torch.cuda.max_memory_allocated
EMPTY_CACHE = torch.cuda.empty_cache
RUNTIME_TIMING_MIN_TRIALS = 2
RUNTIME_TIMING_MAX_TRIALS = 3
RUNTIME_TIMING_MAX_SPREAD_RATIO = 1.25
TIMING_TRIAL_SEED_STRIDE = 1_000_000
COMPONENT_BY_STAGE = {
    "assign": "assignment",
    "update": "centroid_update",
    "full": "combined_iteration",
}
ALLOWED_COMPONENTS = frozenset(
    {*COMPONENT_BY_STAGE.values(), "deployment_bundle"}
)
ALLOWED_MEASUREMENT_MODES = frozenset({"steady_state", "fresh_process"})


@dataclass(frozen=True)
class CaseSpec:
    name: str
    stage: str
    category: str
    b: int
    n: int
    k: int
    d: int
    seed: int
    distribution: str = "uniform"
    tie: bool = False
    weight: float = 0.0
    warmup: int = 1
    iters: int = 1
    repeats: int = 1
    role: str = "performance"
    component: str = ""
    verification: str = "full_exact"
    visibility: str = "unspecified"
    measurement_mode: str = "steady_state"
    score_group: str = ""

    def __post_init__(self) -> None:
        if self.stage not in COMPONENT_BY_STAGE:
            raise ValueError(f"invalid stage: {self.stage}")
        component = self.component or COMPONENT_BY_STAGE[self.stage]
        object.__setattr__(self, "component", component)
        if component not in ALLOWED_COMPONENTS:
            raise ValueError(f"invalid component: {component}")
        if self.measurement_mode not in ALLOWED_MEASUREMENT_MODES:
            raise ValueError(
                f"invalid measurement mode: {self.measurement_mode}"
            )
        if component == "deployment_bundle":
            if self.stage != "full" or self.measurement_mode != "fresh_process":
                raise ValueError(
                    "deployment_bundle requires stage='full' and "
                    "measurement_mode='fresh_process'"
                )
        elif (
            component != COMPONENT_BY_STAGE[self.stage]
            or self.measurement_mode != "steady_state"
        ):
            raise ValueError(
                "steady-state component does not match its execution stage"
            )

    def safe_public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "stage": self.stage,
            "category": self.category,
            "shape": [self.b, self.n, self.k, self.d],
            "distribution": self.distribution,
            "tie": self.tie,
            "weight": self.weight,
            "warmup": self.warmup,
            "iters": self.iters,
            "repeats": self.repeats,
            "role": self.role,
            "component": self.component,
            "verification": self.verification,
            "visibility": self.visibility,
            "measurement_mode": self.measurement_mode,
            "score_group": self.score_group,
        }


_DATA_FAMILY_MAP: dict[str, tuple[str, str, bool]] = {
    "gaussian_high_margin": ("online", "uniform", False),
    "balanced_shuffled": ("online", "balanced_shuffled", False),
    "multiway_tie_shuffled": ("online_tie", "uniform", True),
    "empty_clusters_shuffled": ("online", "hot99_empty", False),
    "close_centroids_secret_permutation": ("close_centroid", "uniform", False),
    "hot99_secret_permutation": ("online", "hot99_empty", False),
    "zipf_secret_permutation": ("online", "zipf", False),
}


def benchmark_case_to_spec(case: BenchmarkCase, seed: int) -> CaseSpec:
    """Adapt one immutable benchmark case to its runtime specification.

    Execution and measurement remain independent: a deployment bundle executes
    the ``full`` stage in a fresh process, while ordinary performance and
    correctness cases use steady-state measurement.
    """

    try:
        category, distribution, tie = _DATA_FAMILY_MAP[case.data_family]
    except KeyError as exc:
        raise ValueError(
            f"unsupported benchmark data family: {case.data_family!r}"
        ) from exc
    return CaseSpec(
        name=case.name,
        stage=case.stage,
        category=category,
        b=case.b,
        n=case.n,
        k=case.k,
        d=case.d,
        seed=seed,
        distribution=distribution,
        tie=tie,
        weight=case.weight,
        warmup=case.warmup,
        iters=case.iters,
        repeats=case.repeats,
        role=case.role,
        component=case.component,
        verification=case.verification,
        visibility=case.visibility,
        measurement_mode=case.measurement_mode,
        score_group=case.score_group,
    )


def exact_tie_pairs(n_clusters: int) -> tuple[tuple[int, int], ...]:
    """Return distinct low/high ID pairs crossing common K-tile boundaries."""

    if n_clusters < 2:
        return ()
    candidates = (
        (0, 1),
        (n_clusters // 2 - 1, n_clusters // 2),
        (n_clusters - 2, n_clusters - 1),
    )
    pairs: list[tuple[int, int]] = []
    for lower, upper in candidates:
        if 0 <= lower < upper < n_clusters and (lower, upper) not in pairs:
            pairs.append((lower, upper))
    return tuple(pairs)


def normalize_stage_weights(specs: list[CaseSpec]) -> list[CaseSpec]:
    counts = {
        stage: sum(1 for spec in specs if spec.stage == stage)
        for stage in STAGE_WEIGHTS
    }
    normalized: list[CaseSpec] = []
    for spec in specs:
        if spec.stage not in counts or counts[spec.stage] <= 0:
            raise ValueError(f"invalid or missing stage: {spec.stage}")
        normalized.append(
            replace(
                spec,
                weight=STAGE_WEIGHTS[spec.stage] / counts[spec.stage],
            )
        )
    total = sum(spec.weight for spec in normalized)
    if not math.isclose(total, RUNTIME_WEIGHT, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"runtime case weights sum to {total}, expected {RUNTIME_WEIGHT}"
        )
    return normalized


def make_inputs(spec: CaseSpec, device: torch.device | str = "cuda") -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(spec.seed)
    x = torch.randn(
        (spec.b, spec.n, spec.d),
        generator=generator,
        device=device,
        dtype=torch.float16,
    )
    centroids = torch.randn(
        (spec.b, spec.k, spec.d),
        generator=generator,
        device=device,
        dtype=torch.float16,
    )
    if spec.category == "close_centroid" and spec.k >= 2:
        pair_count = spec.k // 2
        base = torch.randn(
            (spec.b, pair_count, spec.d),
            generator=generator,
            device=device,
            dtype=torch.float16,
        ) * 4.0
        near = base.clone()
        close_dims = min(8, spec.d)
        # torch 2.3.1 has no CUDA-FP16 nextafter. Reinterpret finite random
        # half values as signed 16-bit IEEE payloads and move one bit-pattern
        # step toward +inf. Negative payloads decrement; nonnegative payloads
        # increment; -0 needs the explicit smallest-positive-subnormal code.
        base_head = base[:, :, :close_dims].contiguous()
        bits = base_head.view(torch.int16)
        successor_bits = torch.where(bits >= 0, bits + 1, bits - 1)
        successor_bits = torch.where(
            bits == torch.iinfo(torch.int16).min,
            torch.ones_like(bits),
            successor_bits,
        )
        near[:, :, :close_dims] = successor_bits.view(torch.float16)
        centroids[:, 0 : 2 * pair_count : 2] = base
        centroids[:, 1 : 2 * pair_count : 2] = near
        point_pairs = torch.arange(spec.n, device=device) % pair_count
        for batch in range(spec.b):
            midpoint = (
                base[batch, point_pairs].float()
                + near[batch, point_pairs].float()
            ) * 0.5
            x[batch] = midpoint.to(torch.float16)
    if spec.tie and spec.k >= 2:
        tie_pairs = exact_tie_pairs(spec.k)
        for lower, upper in tie_pairs:
            centroids[:, upper].copy_(centroids[:, lower])
        tie_points = min(32, spec.n)
        for point_id in range(tie_points):
            lower, _upper = tie_pairs[point_id % len(tie_pairs)]
            x[:, point_id].copy_(centroids[:, lower])
    if spec.stage == "update":
        # x_sq is not an update input (run_stage, clone_guards and check_output
        # never read it for update), so build it row-chunked: the full-shape FP32
        # copy of x plus its square is a 4x-input transient, 33 GiB at the
        # N=33M/D=128 update cases, and set the Judge's device-memory peak.
        # Assign/full keep the unchunked expression so their inputs stay
        # bit-identical; x_sq consumes no generator state either way.
        x_sq = torch.empty((spec.b, spec.n), device=device, dtype=torch.float32)
        point_chunk = max(1024, min(65536, 64 * 1024 * 1024 // max(4, spec.b * spec.d * 4)))
        for start in range(0, spec.n, point_chunk):
            end = min(start + point_chunk, spec.n)
            x_sq[:, start:end] = x[:, start:end].float().square().sum(dim=-1)
    else:
        x_sq = x.float().square().sum(dim=-1)
    old_centroids = torch.randn(
        (spec.b, spec.k, spec.d),
        generator=generator,
        device=device,
        dtype=torch.float16,
    )
    labels = make_labels(spec, generator, device)
    return {
        "x": x.contiguous(),
        "centroids": centroids.contiguous(),
        "x_sq": x_sq.contiguous(),
        "labels": labels.contiguous(),
        "old_centroids": old_centroids.contiguous(),
    }


def make_labels(
    spec: CaseSpec,
    generator: torch.Generator,
    device: torch.device | str,
) -> torch.Tensor:
    if spec.distribution == "uniform":
        return torch.randint(
            0,
            spec.k,
            (spec.b, spec.n),
            generator=generator,
            device=device,
            dtype=torch.int64,
        ).to(torch.int32)
    if spec.distribution == "balanced_shuffled":
        # Every cluster receives either floor(N/K) or ceil(N/K) points.  A
        # generator-backed permutation removes contiguous runs while keeping
        # the count difference exactly bounded by one for every batch.
        balanced = torch.arange(
            spec.n,
            device=device,
            dtype=torch.int64,
        ).remainder(spec.k).to(torch.int32)
        labels = torch.empty(
            (spec.b, spec.n),
            device=device,
            dtype=torch.int32,
        )
        for batch in range(spec.b):
            order = torch.randperm(
                spec.n,
                generator=generator,
                device=device,
            )
            labels[batch] = balanced[order]
        return labels
    if spec.distribution == "zipf":
        uniform = torch.rand(
            (spec.b, spec.n),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        # A deterministic power-law-like distribution with a long tail.
        return torch.clamp((uniform.pow(3.5) * spec.k).floor(), max=spec.k - 1).to(torch.int32)
    if spec.distribution == "hot99_empty":
        active = max(2, spec.k // 8)
        labels = torch.randint(
            1,
            active,
            (spec.b, spec.n),
            generator=generator,
            device=device,
            dtype=torch.int64,
        ).to(torch.int32)
        hot = int(spec.n * 0.99)
        labels[:, :hot] = 0
        return labels
    if spec.distribution == "batched_skew":
        labels = torch.empty((spec.b, spec.n), device=device, dtype=torch.int32)
        for batch in range(spec.b):
            active = max(2, spec.k // (2 + batch % 5))
            labels[batch] = torch.randint(
                0,
                active,
                (spec.n,),
                generator=generator,
                device=device,
                dtype=torch.int64,
            ).to(torch.int32)
            labels[batch, : int(spec.n * (0.5 + 0.08 * (batch % 5)))] = batch % active
        return labels
    raise ValueError(f"unknown label distribution: {spec.distribution}")


def run_stage(
    stage: str,
    assign_fn: Callable[..., Any],
    update_fn: Callable[..., Any],
    data: dict[str, torch.Tensor],
) -> Any:
    if stage == "assign":
        return assign_fn(data["x"], data["centroids"], data["x_sq"])
    if stage == "update":
        return update_fn(data["x"], data["labels"], data["old_centroids"])
    if stage == "full":
        labels = assign_fn(data["x"], data["centroids"], data["x_sq"])
        counts, centroids = update_fn(data["x"], labels, data["centroids"])
        return labels, counts, centroids
    raise ValueError(f"unknown stage: {stage}")


def benchmark_latency_ms(
    stage: str,
    assign_fn: Callable[..., Any],
    update_fn: Callable[..., Any],
    data: dict[str, torch.Tensor],
    *,
    warmup: int,
    iters: int,
    repeats: int,
) -> float:
    for _ in range(max(1, warmup)):
        output = run_stage(stage, assign_fn, update_fn, data)
    DEVICE_SYNCHRONIZE()
    del output
    samples: list[float] = []
    for _ in range(max(1, repeats)):
        # Device-wide synchronized wall time closes the private-stream/event
        # loophole and includes Python launch overhead in this end-to-end task.
        DEVICE_SYNCHRONIZE()
        start = WALL_CLOCK()
        for _ in range(max(1, iters)):
            output = run_stage(stage, assign_fn, update_fn, data)
        DEVICE_SYNCHRONIZE()
        samples.append((WALL_CLOCK() - start) * 1000.0 / max(1, iters))
        del output
    latency = float(statistics.median(samples))
    if not math.isfinite(latency) or latency <= 0.0:
        raise RuntimeError("benchmark produced a non-positive/non-finite latency")
    return latency


def benchmark_fresh_latency_ms(
    spec: CaseSpec,
    assign_fn: Callable[..., Any],
    update_fn: Callable[..., Any],
    *,
    warmup: int,
    iters: int,
    repeats: int,
    seed_offset: int = 10_000_000,
    validate_outputs: bool = False,
    track_memory: bool = False,
) -> float | tuple[float, int]:
    """Median synchronized wall latency with one fresh tensor set per call.

    Tensor generation is outside the timed interval. Keeping the full pool
    alive makes every timed call use a distinct data pointer, which prevents a
    candidate from winning by caching an output for the correctness tensor or
    for the previous benchmark iteration.
    """
    for index in range(max(1, warmup)):
        data = make_inputs(replace(spec, seed=spec.seed + seed_offset + index))
        output = run_stage(spec.stage, assign_fn, update_fn, data)
        DEVICE_SYNCHRONIZE()
        del output
        del data
    samples: list[float] = []
    maximum_peak_delta = 0
    for repeat_index in range(max(1, repeats)):
        pool = [
            make_inputs(
                replace(
                    spec,
                    seed=(
                        spec.seed
                        + seed_offset
                        + 1000
                        + repeat_index * max(1, iters)
                        + call_index
                    ),
                )
            )
            for call_index in range(max(1, iters))
        ]
        guards = (
            [clone_guards(data, spec.stage) for data in pool]
            if validate_outputs
            else []
        )
        outputs: list[Any] = []
        DEVICE_SYNCHRONIZE()
        if track_memory:
            RESET_PEAK_MEMORY()
            allocated_before_calls = int(MEMORY_ALLOCATED())
        start = WALL_CLOCK()
        for data in pool:
            outputs.append(run_stage(spec.stage, assign_fn, update_fn, data))
        DEVICE_SYNCHRONIZE()
        if track_memory:
            retained_output_bytes = sum(_output_tensor_bytes(output) for output in outputs)
            maximum_peak_delta = max(
                maximum_peak_delta,
                max(
                    0,
                    int(MAX_MEMORY_ALLOCATED())
                    - allocated_before_calls
                    - retained_output_bytes,
                ),
            )
        samples.append((WALL_CLOCK() - start) * 1000.0 / max(1, iters))
        if validate_outputs:
            for data, guard, output in zip(pool, guards, outputs):
                unchanged, mutation_message = inputs_unchanged(data, guard)
                if not unchanged:
                    raise RuntimeError(
                        f"timed-call input integrity failed: {mutation_message}"
                    )
                correct, correctness_message = check_output(spec, output, data)
                if not correct:
                    raise RuntimeError(
                        f"timed-call correctness failed: {correctness_message}"
                    )
        del outputs
        del guards
        del pool
        EMPTY_CACHE()
    latency = float(statistics.median(samples))
    if not math.isfinite(latency) or latency <= 0.0:
        raise RuntimeError("benchmark produced a non-positive/non-finite latency")
    if track_memory:
        return latency, maximum_peak_delta
    return latency


def benchmark_stable_latency_ms(
    spec: CaseSpec,
    assign_fn: Callable[..., Any],
    update_fn: Callable[..., Any],
    *,
    warmup: int,
    iters: int,
    repeats: int,
    seed_offset: int = 10_000_000,
    validate_outputs: bool = False,
    track_memory: bool = False,
) -> dict[str, Any]:
    """Measure one scored runtime case with an adaptive outlier guard.

    A complete synchronized timing window can occasionally suffer host or
    device scheduling interference, independent of the kernel stage.  One such
    window must not become a misleading optimization signal. Every
    positive-weight assignment, update, and full-iteration case
    therefore uses two independent median measurements and adds a third when
    their spread exceeds 25%.  Non-runtime diagnostics retain one manifest
    measurement.  Trial medians are returned for auditability.
    """

    guarded_runtime = (
        spec.measurement_mode == "steady_state"
        and spec.stage in STAGE_WEIGHTS
        and spec.weight > 0.0
    )
    minimum_trials = RUNTIME_TIMING_MIN_TRIALS if guarded_runtime else 1
    maximum_trials = RUNTIME_TIMING_MAX_TRIALS if guarded_runtime else 1
    trial_ms: list[float] = []
    peak_bytes: list[int] = []

    for trial_index in range(maximum_trials):
        measured = benchmark_fresh_latency_ms(
            spec,
            assign_fn,
            update_fn,
            warmup=warmup,
            iters=iters,
            repeats=repeats,
            seed_offset=seed_offset + trial_index * TIMING_TRIAL_SEED_STRIDE,
            validate_outputs=validate_outputs,
            track_memory=track_memory,
        )
        if track_memory:
            latency, peak = measured
            peak_bytes.append(int(peak))
        else:
            latency = measured
        trial_ms.append(float(latency))

        if len(trial_ms) < minimum_trials:
            continue
        spread_ratio = max(trial_ms) / min(trial_ms)
        if spread_ratio <= RUNTIME_TIMING_MAX_SPREAD_RATIO:
            break

    latency_ms = float(statistics.median(trial_ms))
    spread_ratio = max(trial_ms) / min(trial_ms)
    inlier_count = sum(
        max(value, latency_ms) / min(value, latency_ms)
        <= RUNTIME_TIMING_MAX_SPREAD_RATIO
        for value in trial_ms
    )
    stable = not guarded_runtime or inlier_count >= 2
    return {
        "latency_ms": latency_ms,
        "trial_medians_ms": trial_ms,
        "trial_count": len(trial_ms),
        "spread_ratio": float(spread_ratio),
        "adaptive_retry": len(trial_ms) > minimum_trials,
        "stable": stable,
        "inlier_count": inlier_count,
        "peak_bytes": max(peak_bytes, default=0),
        "policy": (
            "adaptive_median_2_to_3"
            if guarded_runtime
            else "manifest_median"
        ),
    }


def _output_tensor_bytes(value: Any) -> int:
    """Logical bytes of trusted-shape outputs retained by the harness."""
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, (tuple, list)):
        return sum(_output_tensor_bytes(item) for item in value)
    return 0


def memory_limit_bytes(spec: CaseSpec) -> int:
    """Permit O(BN + BKD) algorithms while rejecting O(BNK).

    A sort-based centroid update necessarily owns sorted labels, an int64
    permutation, an int32 gather permutation, radix-sort workspace, and FP32 centroid
    accumulators.  PyTorch/CUB's transient sort workspace is material at the
    large N=33M and B=32 workloads, so the coefficients include explicit
    allocator headroom calibrated against the unmodified repository port.
    The bounds still scale only in points and centroid state, never in the
    complete point-by-centroid distance matrix.
    """
    base = 128 * MIB
    point_state = spec.b * spec.n
    centroid_state = spec.b * spec.k * spec.d
    if spec.stage == "assign":
        return int(base + 8 * point_state + 8 * centroid_state)
    if spec.stage == "update":
        return int(base + 40 * point_state + 8 * centroid_state)
    if spec.stage == "full":
        return int(base + 48 * point_state + 8 * centroid_state)
    raise ValueError(f"unknown stage: {spec.stage}")


MEMORY_COMFORT_RATIO = 0.85


def memory_diagnostics(peak_bytes: int, limit_bytes: int) -> dict[str, Any]:
    """Return continuous diagnostics for the hard streaming-memory contract.

    The limit remains a feasibility gate.  The ratio and signed headroom make
    near misses visible to an optimizer without turning excess memory into a
    latency tradeoff that could reward materializing the complete B*N*K
    distance tensor.
    """

    peak = max(0, int(peak_bytes))
    limit = int(limit_bytes)
    if limit <= 0:
        raise ValueError("memory limit must be positive")
    ratio = peak / limit
    signed_headroom = limit - peak
    if ratio > 1.0:
        pressure = "over_budget"
    elif ratio > MEMORY_COMFORT_RATIO:
        pressure = "tight"
    else:
        pressure = "comfortable"
    return {
        "memory_ratio": float(ratio),
        "memory_headroom_bytes": int(signed_headroom),
        "memory_headroom_ratio": float(1.0 - ratio),
        "memory_overage_bytes": int(max(0, -signed_headroom)),
        "memory_overage_ratio": float(max(0.0, ratio - 1.0)),
        "memory_pressure": pressure,
        "memory_comfort_ratio": MEMORY_COMFORT_RATIO,
    }


def peak_output(
    spec: CaseSpec,
    assign_fn: Callable[..., Any],
    update_fn: Callable[..., Any],
    data: dict[str, torch.Tensor],
) -> tuple[Any, int]:
    DEVICE_SYNCHRONIZE()
    RESET_PEAK_MEMORY()
    start_allocated = int(MEMORY_ALLOCATED())
    output = run_stage(spec.stage, assign_fn, update_fn, data)
    DEVICE_SYNCHRONIZE()
    peak = max(0, int(MAX_MEMORY_ALLOCATED()) - start_allocated)
    return output, peak


@torch.no_grad()
def reference_assign(
    x: torch.Tensor,
    centroids: torch.Tensor,
    x_sq: torch.Tensor,
    *,
    return_best_distance: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Independent chunked FP32 reference; never allocates full B*N*K."""
    bsz, n_points, _ = x.shape
    n_clusters = centroids.shape[1]
    labels = torch.empty((bsz, n_points), device=x.device, dtype=torch.int64)
    best_distances = torch.empty(
        (bsz, n_points),
        device=x.device,
        dtype=torch.float32,
    )
    centroid_f32 = centroids.float()
    centroid_sq = centroid_f32.square().sum(dim=-1)
    for n_start in range(0, n_points, REFERENCE_BLOCK_N):
        n_end = min(n_start + REFERENCE_BLOCK_N, n_points)
        x_tile = x[:, n_start:n_end].float()
        best = torch.full(
            (bsz, n_end - n_start),
            float("inf"),
            device=x.device,
            dtype=torch.float32,
        )
        best_id = torch.zeros(
            (bsz, n_end - n_start),
            device=x.device,
            dtype=torch.int64,
        )
        for k_start in range(0, n_clusters, REFERENCE_BLOCK_K):
            k_end = min(k_start + REFERENCE_BLOCK_K, n_clusters)
            cross = torch.bmm(
                x_tile,
                centroid_f32[:, k_start:k_end].transpose(1, 2),
            )
            distance = (
                x_sq[:, n_start:n_end, None]
                + centroid_sq[:, None, k_start:k_end]
                - 2.0 * cross
            )
            local, local_id = distance.min(dim=-1)
            replace = local < best
            best = torch.where(replace, local, best)
            best_id = torch.where(replace, local_id + k_start, best_id)
        labels[:, n_start:n_end] = best_id
        best_distances[:, n_start:n_end] = best
    if return_best_distance:
        return labels, best_distances
    return labels


@torch.no_grad()
def reference_update(
    x: torch.Tensor,
    labels: torch.Tensor,
    old_centroids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Chunked FP32 sums plus a count/conditioning-aware error envelope."""
    bsz, n_points, dim = x.shape
    n_clusters = old_centroids.shape[1]
    counts = torch.zeros((bsz, n_clusters), device=x.device, dtype=torch.int64)
    sums = torch.zeros((bsz, n_clusters, dim), device=x.device, dtype=torch.float32)
    chunk = 65536
    for batch in range(bsz):
        for start in range(0, n_points, chunk):
            end = min(start + chunk, n_points)
            ids = labels[batch, start:end].to(torch.int64)
            counts[batch] += torch.bincount(ids, minlength=n_clusters)
            sums[batch].index_add_(0, ids, x[batch, start:end].float())
    means = sums / counts.clamp_min(1).float().unsqueeze(-1)
    means = torch.where(
        (counts == 0).unsqueeze(-1),
        old_centroids.float(),
        means,
    )
    mean_abs_input = x.float().abs().mean().clamp_min(1e-6)
    rounding = (
        64.0
        * torch.finfo(torch.float32).eps
        * counts.float().sqrt().unsqueeze(-1)
        * mean_abs_input
    )
    tolerance = 0.003 + 0.003 * means.abs() + rounding
    return counts, means.to(x.dtype), tolerance


def clone_guards(data: dict[str, torch.Tensor], stage: str) -> dict[str, torch.Tensor]:
    names = {
        "assign": ("x", "centroids", "x_sq"),
        "update": ("x", "labels", "old_centroids"),
        "full": ("x", "centroids", "x_sq"),
    }[stage]
    return {name: data[name].clone() for name in names}


def inputs_unchanged(
    data: dict[str, torch.Tensor],
    guards: dict[str, torch.Tensor],
) -> tuple[bool, str]:
    for name, guard in guards.items():
        if not torch.equal(data[name], guard):
            return False, f"input tensor was mutated: {name}"
    return True, ""


def _valid_label_tensor(value: Any, spec: CaseSpec) -> tuple[bool, str]:
    if not isinstance(value, torch.Tensor):
        return False, "labels is not a tensor"
    if value.shape != (spec.b, spec.n):
        return False, "labels has the wrong shape"
    if not value.is_cuda:
        return False, "labels must be on CUDA"
    if value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        return False, "labels must have an integer dtype"
    if value.numel() and (int(value.min()) < 0 or int(value.max()) >= spec.k):
        return False, "labels contains an out-of-range cluster id"
    return True, ""


def _check_update(
    value: Any,
    spec: CaseSpec,
    x: torch.Tensor,
    labels: torch.Tensor,
    old_centroids: torch.Tensor,
    *,
    sampled: bool = False,
) -> tuple[bool, str]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return False, "update must return (counts, new_centroids)"
    counts, centroids = value
    if not isinstance(counts, torch.Tensor) or not isinstance(centroids, torch.Tensor):
        return False, "update outputs must be tensors"
    if counts.shape != (spec.b, spec.k) or centroids.shape != (spec.b, spec.k, spec.d):
        return False, "update output shape mismatch"
    if not counts.is_cuda or not centroids.is_cuda:
        return False, "update outputs must remain on CUDA"
    if counts.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        return False, "counts must have an integer dtype"
    if centroids.dtype != torch.float16:
        return False, "new_centroids must be FP16"
    if sampled:
        return _check_update_sampled(
            counts,
            centroids,
            spec,
            x,
            labels,
            old_centroids,
        )
    ref_counts, ref_centroids, tolerance = reference_update(x, labels, old_centroids)
    if not torch.equal(counts.to(torch.int64), ref_counts):
        return False, "cluster counts are not exact"
    empty = ref_counts == 0
    if empty.any() and not torch.equal(centroids[empty], old_centroids[empty]):
        return False, "an empty cluster did not retain its old centroid exactly"
    error = (centroids.float() - ref_centroids.float()).abs()
    if not bool(torch.all(error <= tolerance)):
        return False, "centroid error exceeds the conditioning-aware FP32 tolerance"
    return True, ""


def _deterministic_sample_ids(
    size: int,
    count: int,
    seed: int,
) -> list[int]:
    """Return stable, unique, spread-out IDs without allocating ``O(size)``."""
    count = max(1, min(size, count))
    offset = seed % max(1, size // count)
    values = {
        min(size - 1, (index * size) // count + offset)
        for index in range(count)
    }
    values.update((0, size - 1))
    return sorted(values)


def _assignment_sample_indices(spec: CaseSpec) -> torch.Tensor:
    """Bound sampled-reference FLOPs across all benchmark shapes and batches."""
    target_fma = 128_000_000
    work_per_point = max(1, spec.b * spec.k * spec.d)
    count = max(4, min(128, target_fma // work_per_point, spec.n))
    ids = _deterministic_sample_ids(spec.n, count, spec.seed + 17_171)
    if spec.tie:
        ids = sorted(set(ids).union(range(min(32, spec.n))))
    return torch.tensor(ids, device="cuda", dtype=torch.int64)


def _check_assignment_sampled(
    spec: CaseSpec,
    labels: torch.Tensor,
    data: dict[str, torch.Tensor],
) -> tuple[bool, str]:
    indices = _assignment_sample_indices(spec)
    sampled_spec = replace(spec, n=int(indices.numel()))
    sampled_data = {
        "x": data["x"][:, indices],
        "centroids": data["centroids"],
        "x_sq": data["x_sq"][:, indices],
    }
    sampled_labels = labels[:, indices]
    reference = reference_assign(
        sampled_data["x"],
        sampled_data["centroids"],
        sampled_data["x_sq"],
    )
    return _check_assignment_distance(
        sampled_spec,
        sampled_labels,
        reference,
        sampled_data,
    )


def _check_update_sampled(
    counts: torch.Tensor,
    centroids: torch.Tensor,
    spec: CaseSpec,
    x: torch.Tensor,
    labels: torch.Tensor,
    old_centroids: torch.Tensor,
) -> tuple[bool, str]:
    """Check all counts and deterministic centroid samples in ``O(B*N)``.

    Full FP32 centroid reconstruction is retained for the task-only correctness
    gates.  Large performance cases instead scan all labels for exact
    counts, then reduce only protected sampled clusters.  This keeps the
    verifier bounded for the 8--33M-point cases without weakening the ABI or
    allowing an incorrect count vector.
    """
    ref_counts = torch.stack(
        [
            torch.bincount(labels[batch].to(torch.int64), minlength=spec.k)
            for batch in range(spec.b)
        ],
        dim=0,
    )
    if not torch.equal(counts.to(torch.int64), ref_counts):
        return False, "cluster counts are not exact"

    cluster_ids_list = _deterministic_sample_ids(
        spec.k,
        min(16, spec.k),
        spec.seed + 29_297,
    )
    cluster_ids = torch.tensor(cluster_ids_list, device=x.device, dtype=torch.int64)
    lookup = torch.full((spec.k,), -1, device=x.device, dtype=torch.int64)
    lookup[cluster_ids] = torch.arange(
        cluster_ids.numel(),
        device=x.device,
        dtype=torch.int64,
    )
    for batch in range(spec.b):
        sample_sums = torch.zeros(
            (cluster_ids.numel(), spec.d),
            device=x.device,
            dtype=torch.float32,
        )
        chunk = 262_144
        for start in range(0, spec.n, chunk):
            end = min(start + chunk, spec.n)
            local_labels = labels[batch, start:end].to(torch.int64)
            slots = lookup[local_labels]
            selected = slots >= 0
            if bool(selected.any()):
                sample_sums.index_add_(
                    0,
                    slots[selected],
                    x[batch, start:end][selected].float(),
                )
        sample_counts = ref_counts[batch, cluster_ids]
        expected = sample_sums / sample_counts.clamp_min(1).float().unsqueeze(-1)
        old_sample = old_centroids[batch, cluster_ids].float()
        expected = torch.where(
            (sample_counts == 0).unsqueeze(-1),
            old_sample,
            expected,
        )
        candidate = centroids[batch, cluster_ids].float()
        empty = sample_counts == 0
        if empty.any() and not torch.equal(
            centroids[batch, cluster_ids[empty]],
            old_centroids[batch, cluster_ids[empty]],
        ):
            return False, "a sampled empty cluster did not retain its old centroid"
        rounding = (
            128.0
            * torch.finfo(torch.float32).eps
            * sample_counts.float().sqrt().unsqueeze(-1)
        )
        tolerance = 0.003 + 0.003 * expected.abs() + rounding
        if not bool(torch.all((candidate - expected).abs() <= tolerance)):
            return False, "sampled centroid error exceeds the FP32 tolerance"
    return True, ""


def _uses_sampled_verification(spec: CaseSpec) -> bool:
    return "sampled" in spec.verification or spec.verification.startswith("first_run")


def check_output(
    spec: CaseSpec,
    output: Any,
    data: dict[str, torch.Tensor],
) -> tuple[bool, str]:
    if spec.stage == "assign":
        ok, message = _valid_label_tensor(output, spec)
        if not ok:
            return ok, message
        if _uses_sampled_verification(spec):
            return _check_assignment_sampled(spec, output, data)
        reference = reference_assign(data["x"], data["centroids"], data["x_sq"])
        return _check_assignment_distance(spec, output, reference, data)
    if spec.stage == "update":
        return _check_update(
            output,
            spec,
            data["x"],
            data["labels"],
            data["old_centroids"],
            sampled=_uses_sampled_verification(spec),
        )
    if spec.stage == "full":
        if not isinstance(output, (tuple, list)) or len(output) != 3:
            return False, "full stage produced an invalid internal output"
        labels, counts, centroids = output
        ok, message = _valid_label_tensor(labels, spec)
        if not ok:
            return ok, message
        if _uses_sampled_verification(spec):
            label_ok, label_message = _check_assignment_sampled(spec, labels, data)
        else:
            reference = reference_assign(data["x"], data["centroids"], data["x_sq"])
            label_ok, label_message = _check_assignment_distance(
                spec,
                labels,
                reference,
                data,
            )
        if not label_ok:
            return False, f"full-stage {label_message}"
        return _check_update(
            (counts, centroids),
            spec,
            data["x"],
            labels,
            data["centroids"],
            sampled=_uses_sampled_verification(spec),
        )
    return False, "unknown stage"


def _check_assignment_distance(
    spec: CaseSpec,
    labels: torch.Tensor,
    reference: torch.Tensor,
    data: dict[str, torch.Tensor],
) -> tuple[bool, str]:
    """Gap-aware exactness using one direct arithmetic path on mismatches."""
    labels_i64 = labels.to(torch.int64)
    mismatch = labels_i64 != reference
    if mismatch.any():
        batch_index, point_index = torch.where(mismatch)
        points = data["x"][batch_index, point_index].float()
        candidate_centroids = data["centroids"][
            batch_index,
            labels_i64[batch_index, point_index],
        ].float()
        reference_centroids = data["centroids"][
            batch_index,
            reference[batch_index, point_index],
        ].float()
        candidate_distance = (points - candidate_centroids).square().sum(dim=-1)
        reference_distance = (points - reference_centroids).square().sum(dim=-1)
        # The assignment contract evaluates ||x||² + ||c||² - 2<x,c>. For
        # near points those large terms cancel, so an error bound based only on the tiny
        # final distance is unsound. A standard forward-error scale for this
        # identity is the sum of absolute terms in the expression.
        point_energy = points.square().sum(dim=-1)
        candidate_energy = candidate_centroids.square().sum(dim=-1)
        reference_energy = reference_centroids.square().sum(dim=-1)
        candidate_cross_abs = (points.abs() * candidate_centroids.abs()).sum(dim=-1)
        reference_cross_abs = (points.abs() * reference_centroids.abs()).sum(dim=-1)
        cancellation_scale = (
            point_energy
            + torch.maximum(candidate_energy, reference_energy)
            + 2.0 * torch.maximum(candidate_cross_abs, reference_cross_abs)
        )
        forward_error_envelope = (
            2.0
            * torch.finfo(torch.float32).eps
            * cancellation_scale
            + spec.d * torch.finfo(torch.float32).eps
        )
        # The forward bound is deliberately conservative, but must not turn
        # into a broad approximate-nearest-neighbour allowance at large input
        # magnitudes.  The fixed FP16 task families need at most 1.49e-4 for
        # equivalent one-ULP close-centroid choices across public/hidden fresh
        # seeds on H100.  A 2.5e-4 cap leaves reproducibility headroom while
        # rejecting an exactly representable 9.77e-4 farther-centroid gap.
        envelope = torch.clamp(forward_error_envelope, max=2.5e-4)
        if not bool(torch.all(candidate_distance <= reference_distance + envelope)):
            return False, "assignment selected a meaningfully farther centroid"
    if spec.tie:
        # The protected prefix contains byte-identical centroid pairs spanning
        # common K-tile boundaries. Compute the contract labels directly:
        # equivalent centroids evaluated by different GEMM tile shapes can
        # receive slightly different cancellation error in the chunked
        # arithmetic reference, but an exact byte-identical tie must still
        # choose the lower ID.
        tie_points = min(32, spec.n)
        pairs = exact_tie_pairs(spec.k)
        expected = torch.tensor(
            [pairs[index % len(pairs)][0] for index in range(tie_points)],
            device=labels.device,
            dtype=torch.int64,
        ).unsqueeze(0).expand(spec.b, -1)
        if not bool(
            torch.equal(
                labels_i64[:, :tie_points],
                expected,
            )
        ):
            return False, "exact tie did not choose the smallest cluster id"
    return True, ""


def load_candidate(path: Path, version: int) -> tuple[ModuleType, Callable[..., Any], Callable[..., Any]]:
    module_name = f"_flash_kmeans_candidate_v{version:03d}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError("could not construct candidate module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assign_fn = getattr(module, "assign", None)
    update_fn = getattr(module, "update", None)
    if not callable(assign_fn) or not callable(update_fn):
        raise TypeError("candidate must export callable assign and update")
    return module, assign_fn, update_fn
