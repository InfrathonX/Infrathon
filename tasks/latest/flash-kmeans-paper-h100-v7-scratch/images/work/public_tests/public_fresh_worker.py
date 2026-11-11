#!/usr/bin/env python3
"""One public, candidate-only fresh-process scaling probe.

This file is agent-visible by design.  It contains only public proxy shapes and
sampled validation logic; it does not contain a formal Judge workload or an
optimized K-means implementation.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import math
import os
import signal
import statistics
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Final

import torch


MIB: Final[int] = 1024 * 1024
INTEGER_DTYPES: Final[tuple[torch.dtype, ...]] = (
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
)
PUBLIC_CASES: Final[dict[str, dict[str, Any]]] = {
    "control": {
        "family": "control",
        "rung": "control",
        "full": (1, 1_048_576, 8_192, 512),
        "quick": (1, 131_072, 4_096, 256),
        "seed": 71_101,
    },
    "large_k_small": {
        "family": "large-K",
        "rung": "small",
        "full": (1, 393_216, 49_152, 160),
        "quick": (1, 98_304, 24_576, 160),
        "seed": 71_102,
    },
    "large_k_large": {
        "family": "large-K",
        "rung": "large",
        "full": (1, 1_048_576, 49_152, 160),
        "quick": (1, 98_304, 24_576, 160),
        "seed": 71_103,
    },
    "large_n_high_d_small": {
        "family": "large-N/high-D",
        "rung": "small",
        "full": (1, 3_145_728, 6_144, 384),
        "quick": (1, 524_288, 3_072, 384),
        "seed": 71_104,
    },
    "large_n_high_d_large": {
        "family": "large-N/high-D",
        "rung": "large",
        "full": (1, 6_291_456, 6_144, 384),
        "quick": (1, 524_288, 3_072, 384),
        "seed": 71_105,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--case-id", choices=tuple(PUBLIC_CASES), required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def arm_parent_death_signal() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        value = ctypes.get_errno()
        raise OSError(value, "prctl(PR_SET_PDEATHSIG) failed")
    if os.getppid() == 1:
        os.kill(os.getpid(), signal.SIGKILL)


def validate_empty_cache_root(cache_root: Path) -> None:
    root = cache_root.resolve()
    expected = {
        "TRITON_CACHE_DIR": root / "triton",
        "CUDA_CACHE_PATH": root / "cuda",
        "TORCHINDUCTOR_CACHE_DIR": root / "torchinductor",
        "XDG_CACHE_HOME": root / "xdg",
        "TMPDIR": root / "tmp",
        "TORCH_EXTENSIONS_DIR": root / "torch_extensions",
    }
    for name, path in expected.items():
        actual = Path(os.environ.get(name, "")).resolve()
        if actual != path or not actual.is_dir() or any(actual.iterdir()):
            raise RuntimeError(f"{name} is not the requested empty cache")


def load_candidate(
    source: Path,
) -> tuple[ModuleType, Callable[..., Any], Callable[..., Any]]:
    spec = importlib.util.spec_from_file_location(
        "_flash_kmeans_public_fresh_candidate", source
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not construct candidate module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assign = getattr(module, "assign", None)
    update = getattr(module, "update", None)
    if not callable(assign) or not callable(update):
        raise RuntimeError("candidate must export callable assign and update")
    return module, assign, update


def x_squared(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty(x.shape[:2], device=x.device, dtype=torch.float32)
    for start in range(0, x.shape[1], 262_144):
        end = min(start + 262_144, x.shape[1])
        output[:, start:end] = x[:, start:end].float().square().sum(-1)
    return output


def make_inputs(
    shape: tuple[int, int, int, int], seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b, n, k, d = shape
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    x = torch.randn(
        (b, n, d), device="cuda", dtype=torch.float16, generator=generator
    )
    centroids = torch.randn(
        (b, k, d), device="cuda", dtype=torch.float16, generator=generator
    )
    return x, centroids, x_squared(x)


def guard_indices(length: int, seed: int, count: int = 257) -> torch.Tensor:
    fixed = [0, min(1, length - 1), length // 2, max(0, length - 2), length - 1]
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    random = torch.randint(
        0,
        length,
        (max(0, count - len(fixed)),),
        device="cuda",
        generator=generator,
    )
    return torch.unique(
        torch.cat(
            [torch.tensor(fixed, device="cuda", dtype=torch.int64), random]
        ),
        sorted=True,
    )


def capture_guards(
    x: torch.Tensor, centroids: torch.Tensor, squared: torch.Tensor, seed: int
) -> dict[str, Any]:
    x_ids = guard_indices(x.shape[1], seed)
    c_ids = guard_indices(centroids.shape[1], seed + 1, count=33)
    return {
        "x_shape": tuple(x.shape),
        "x_stride": tuple(x.stride()),
        "c_shape": tuple(centroids.shape),
        "c_stride": tuple(centroids.stride()),
        "s_shape": tuple(squared.shape),
        "s_stride": tuple(squared.stride()),
        "x_ids": x_ids,
        "c_ids": c_ids,
        "x": x.index_select(1, x_ids).clone(),
        "c": centroids.index_select(1, c_ids).clone(),
        "s": squared.index_select(1, x_ids).clone(),
    }


def inputs_unchanged(
    x: torch.Tensor,
    centroids: torch.Tensor,
    squared: torch.Tensor,
    guards: dict[str, Any],
) -> bool:
    return bool(
        tuple(x.shape) == guards["x_shape"]
        and tuple(x.stride()) == guards["x_stride"]
        and tuple(centroids.shape) == guards["c_shape"]
        and tuple(centroids.stride()) == guards["c_stride"]
        and tuple(squared.shape) == guards["s_shape"]
        and tuple(squared.stride()) == guards["s_stride"]
        and torch.equal(x.index_select(1, guards["x_ids"]), guards["x"])
        and torch.equal(
            centroids.index_select(1, guards["c_ids"]), guards["c"]
        )
        and torch.equal(
            squared.index_select(1, guards["x_ids"]), guards["s"]
        )
    )


def validate_outputs(
    labels: Any,
    output: Any,
    *,
    shape: tuple[int, int, int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b, n, k, d = shape
    if (
        not isinstance(labels, torch.Tensor)
        or not labels.is_cuda
        or labels.shape != (b, n)
        or labels.dtype not in INTEGER_DTYPES
    ):
        raise AssertionError("assign output violates CUDA integer [B,N] ABI")
    if labels.numel() and (
        int(labels.min().item()) < 0 or int(labels.max().item()) >= k
    ):
        raise AssertionError("assign output contains an id outside [0,K)")
    if not isinstance(output, (tuple, list)) or len(output) != 2:
        raise AssertionError("update must return (counts, new_centroids)")
    counts, new_centroids = output
    if (
        not isinstance(counts, torch.Tensor)
        or not counts.is_cuda
        or counts.shape != (b, k)
        or counts.dtype not in INTEGER_DTYPES
    ):
        raise AssertionError("counts violate CUDA integer [B,K] ABI")
    if (
        not isinstance(new_centroids, torch.Tensor)
        or not new_centroids.is_cuda
        or new_centroids.shape != (b, k, d)
        or new_centroids.dtype != torch.float16
    ):
        raise AssertionError("new_centroids violate CUDA FP16 [B,K,D] ABI")
    if not bool(torch.isfinite(new_centroids).all().item()):
        raise AssertionError("new_centroids contains non-finite values")
    return labels, counts, new_centroids


def sampled_assignment_reference(
    x: torch.Tensor,
    centroids: torch.Tensor,
    squared: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
) -> bool:
    for batch in range(x.shape[0]):
        sample_x = x[batch].index_select(0, indices).float()
        sample_sq = squared[batch].index_select(0, indices)
        best_distance = torch.full(
            (indices.numel(),), float("inf"), device="cuda", dtype=torch.float32
        )
        best_index = torch.zeros(
            (indices.numel(),), device="cuda", dtype=torch.int64
        )
        for start in range(0, centroids.shape[1], 1024):
            end = min(start + 1024, centroids.shape[1])
            chunk = centroids[batch, start:end].float()
            distances = (
                sample_sq[:, None]
                + chunk.square().sum(-1)[None, :]
                - 2.0 * (sample_x @ chunk.transpose(0, 1))
            )
            values, local = distances.min(dim=1)
            better = values < best_distance
            best_distance = torch.where(better, values, best_distance)
            best_index = torch.where(better, local.to(torch.int64) + start, best_index)
        actual = labels[batch].index_select(0, indices).to(torch.int64)
        if not bool(torch.equal(actual, best_index)):
            return False
    return True


def sampled_update_reference(
    x: torch.Tensor,
    old: torch.Tensor,
    labels: torch.Tensor,
    counts: torch.Tensor,
    new_centroids: torch.Tensor,
    seed: int,
) -> tuple[bool, int]:
    checked = 0
    for batch in range(x.shape[0]):
        ids = labels[batch].to(torch.int64)
        expected_counts = torch.bincount(ids, minlength=old.shape[1])
        if not torch.equal(counts[batch].to(torch.int64), expected_counts):
            return False, checked
        candidates = guard_indices(old.shape[1], seed + batch, count=33)
        nonempty = torch.nonzero(expected_counts > 0, as_tuple=False).flatten()
        empty = torch.nonzero(expected_counts == 0, as_tuple=False).flatten()
        selected = torch.unique(
            torch.cat([candidates, nonempty[:8], empty[:8]]), sorted=True
        )
        mapping = torch.full(
            (old.shape[1],), -1, device="cuda", dtype=torch.int64
        )
        mapping[selected] = torch.arange(selected.numel(), device="cuda")
        sums = torch.zeros(
            (selected.numel(), x.shape[2]), device="cuda", dtype=torch.float32
        )
        absolute_sum = torch.zeros((), device="cuda", dtype=torch.float32)
        for start in range(0, x.shape[1], 65_536):
            end = min(start + 65_536, x.shape[1])
            chunk_ids = ids[start:end]
            local = mapping[chunk_ids]
            mask = local >= 0
            if bool(mask.any().item()):
                values = x[batch, start:end][mask].float()
                sums.index_add_(0, local[mask], values)
                absolute_sum += values.abs().sum()
        selected_counts = expected_counts.index_select(0, selected)
        reference = sums / selected_counts.clamp_min(1).float().unsqueeze(-1)
        old_selected = old[batch].index_select(0, selected).float()
        reference = torch.where(
            (selected_counts == 0).unsqueeze(-1), old_selected, reference
        )
        actual = new_centroids[batch].index_select(0, selected).float()
        empty_selected = selected_counts == 0
        if bool(empty_selected.any().item()) and not torch.equal(
            new_centroids[batch].index_select(0, selected)[empty_selected],
            old[batch].index_select(0, selected)[empty_selected],
        ):
            return False, checked
        mean_abs = (
            absolute_sum / max(1, int(selected_counts.sum().item()) * x.shape[2])
        ).clamp_min(1e-6)
        rounding = (
            64.0
            * torch.finfo(torch.float32).eps
            * selected_counts.float().sqrt().unsqueeze(-1)
            * mean_abs
        )
        tolerance = 0.003 + 0.003 * reference.abs() + rounding
        if not bool(torch.all(torch.abs(actual - reference) <= tolerance).item()):
            return False, checked
        checked += int(selected.numel())
    return True, checked


def run_iteration(
    assign: Callable[..., Any],
    update: Callable[..., Any],
    x: torch.Tensor,
    old: torch.Tensor,
    squared: torch.Tensor,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    labels = assign(x, old, squared)
    output = update(x, labels, old)
    return labels, output


def verify_iteration(
    labels: Any,
    output: Any,
    x: torch.Tensor,
    old: torch.Tensor,
    squared: torch.Tensor,
    guards: dict[str, Any],
    shape: tuple[int, int, int, int],
    seed: int,
) -> tuple[bool, bool, int, int]:
    labels, counts, new_centroids = validate_outputs(labels, output, shape=shape)
    assignment_indices = guard_indices(shape[1], seed + 9_001)
    assignment_ok = sampled_assignment_reference(
        x, old, squared, labels, assignment_indices
    )
    update_ok, centroid_count = sampled_update_reference(
        x, old, labels, counts, new_centroids, seed + 9_002
    )
    integrity_ok = inputs_unchanged(x, old, squared, guards)
    return (
        bool(assignment_ok and update_ok),
        integrity_ok,
        int(assignment_indices.numel()) * shape[0],
        centroid_count,
    )


def memory_limit(shape: tuple[int, int, int, int]) -> int:
    b, n, k, d = shape
    return 128 * MIB + 48 * b * n + 8 * b * k * d


def effective_assignment_tflops(
    shape: tuple[int, int, int, int], latency_ms: float
) -> float:
    b, n, k, d = shape
    return 2.0 * b * n * k * d / (latency_ms / 1000.0) / 1.0e12


def execute(args: argparse.Namespace) -> dict[str, Any]:
    case = PUBLIC_CASES[args.case_id]
    shape = tuple(case["quick" if args.quick else "full"])
    assert len(shape) == 4
    source = args.source.resolve()
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    validate_empty_cache_root(args.cache_root)
    result: dict[str, Any] = {
        "case_id": args.case_id,
        "family": case["family"],
        "rung": case["rung"],
        "shape": list(shape),
        "measurement_mode": "fresh_process",
        "pid": os.getpid(),
        "cache_was_empty": True,
        "status": "failed",
        "correctness": {"status": "not_reached"},
    }
    warmed_repeats = 1 if args.quick else 2
    first_data = make_inputs(shape, int(case["seed"]))
    first_guards = capture_guards(*first_data, int(case["seed"]) + 1_000)
    torch.cuda.synchronize()
    allocated_before = int(torch.cuda.memory_allocated())
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    _module, assign, update = load_candidate(source)
    first_labels, first_output = run_iteration(assign, update, *first_data)
    torch.cuda.synchronize()
    first_run_ms = (time.perf_counter() - started) * 1000.0
    peak_extra = max(0, int(torch.cuda.max_memory_allocated()) - allocated_before)
    correct, integrity, assignment_checked, centroid_checked = verify_iteration(
        first_labels,
        first_output,
        *first_data,
        first_guards,
        shape,
        int(case["seed"]),
    )
    if not correct or not integrity:
        raise AssertionError("first-run sampled correctness or input integrity failed")
    del first_labels, first_output, first_guards, first_data
    torch.cuda.empty_cache()

    warmed_samples: list[float] = []
    for repeat in range(warmed_repeats):
        seed = int(case["seed"]) + 1_000_000 + repeat
        data = make_inputs(shape, seed)
        guards = capture_guards(*data, seed + 1_000)
        torch.cuda.synchronize()
        before = int(torch.cuda.memory_allocated())
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        labels, output = run_iteration(assign, update, *data)
        torch.cuda.synchronize()
        warmed_samples.append((time.perf_counter() - started) * 1000.0)
        peak_extra = max(
            peak_extra,
            max(0, int(torch.cuda.max_memory_allocated()) - before),
        )
        correct, integrity, a_count, c_count = verify_iteration(
            labels, output, *data, guards, shape, seed
        )
        assignment_checked = min(assignment_checked, a_count)
        centroid_checked = min(centroid_checked, c_count)
        if not correct or not integrity:
            raise AssertionError(
                "warmed sampled correctness or input integrity failed"
            )
        del labels, output, guards, data
        torch.cuda.empty_cache()

    if hashlib.sha256(source.read_bytes()).hexdigest() != source_digest:
        raise AssertionError("candidate source changed during the public probe")
    warmed_ms = float(statistics.median(warmed_samples))
    limit = memory_limit(shape)
    tflops = effective_assignment_tflops(shape, warmed_ms)
    memory_ok = peak_extra <= limit
    result.update(
        {
            "status": "passed" if memory_ok else "failed",
            "first_run_ms": first_run_ms,
            "warmed_iteration_ms": warmed_ms,
            "warmed_iteration_samples_ms": warmed_samples,
            "peak_extra_bytes": peak_extra,
            "memory_limit_bytes": limit,
            "memory_ratio": peak_extra / limit,
            "memory_ok": memory_ok,
            "input_integrity_ok": True,
            "output_contract_ok": True,
            "effective_assignment_tflops": tflops,
            "correctness": {
                "status": "passed",
                "assignment_points_checked": assignment_checked,
                "counts_checked": "all",
                "centroid_clusters_checked": centroid_checked,
                "first_and_every_warmed_checked": True,
            },
        }
    )
    if not memory_ok:
        result["error"] = "candidate exceeded the public streaming-memory budget"
    return result


def main() -> int:
    args = parse_args()
    arm_parent_death_signal()
    result: dict[str, Any]
    code = 0
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        result = execute(args)
        code = 0 if result.get("status") == "passed" else 1
    except Exception as exc:
        code = 1
        case = PUBLIC_CASES.get(args.case_id, {})
        mode = "quick" if args.quick else "full"
        result = {
            "case_id": args.case_id,
            "family": case.get("family", "unavailable"),
            "rung": case.get("rung", "unavailable"),
            "shape": list(case.get(mode, ())),
            "measurement_mode": "fresh_process",
            "pid": os.getpid(),
            "status": "failed",
            "correctness": {"status": "failed"},
            "error": f"{type(exc).__name__}: {exc}",
        }
    args.result.write_text(
        json.dumps(result, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
