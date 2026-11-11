#!/usr/bin/env python3
"""Agent-visible correctness, steady-state, and fresh-process scale tests.

Tiny correctness uses literal golden fixtures. Large public proxies run in
independent processes with empty compilation caches and sampled correctness.
They intentionally differ from every private workload. The formal Judge owns
authoritative correctness, score aggregation, and private performance cases.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import torch


MIB = 1024 * 1024
INTEGER_DTYPES = {torch.int16, torch.int32, torch.int64}
VERSION_PATTERN = re.compile(r"flash_kmeans_v([0-9]{3})\.py")
RESULTS_DIR = Path("/workspace/public_tests/results")
FRESH_WORKER = Path(__file__).with_name("public_fresh_worker.py")
FRESH_CASES_FULL = (
    "control",
    "large_k_small",
    "large_k_large",
    "large_n_high_d_small",
    "large_n_high_d_large",
)
FRESH_CASES_QUICK = (
    "control",
    "large_k_small",
    "large_n_high_d_small",
)
FRESH_CASE_TIMEOUT_SECONDS = 180


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--candidate-version", type=int)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def select_source(root: Path, requested: int | None) -> tuple[int, Path]:
    if requested is None:
        selector = root / "FINAL_VERSION"
        try:
            raw = selector.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise RuntimeError(
                "FINAL_VERSION is missing; create a candidate and select it"
            ) from exc
        if not re.fullmatch(r"[0-9]{1,3}", raw):
            raise RuntimeError("FINAL_VERSION must contain one integer")
        version = int(raw)
    else:
        version = int(requested)
    if not 1 <= version <= 999:
        raise RuntimeError("candidate version must be between 1 and 999")
    source = root / "kernels" / f"flash_kmeans_v{version:03d}.py"
    if not source.is_file() or source.is_symlink():
        raise RuntimeError(
            f"missing kernels/flash_kmeans_v{version:03d}.py"
        )
    for path in (root / "kernels").glob("*.py"):
        if not VERSION_PATTERN.fullmatch(path.name):
            raise RuntimeError(
                f"unexpected Python file under kernels/: {path.name}"
            )
    return version, source


def load_candidate(
    source: Path,
) -> tuple[ModuleType, Callable[..., Any], Callable[..., Any]]:
    spec = importlib.util.spec_from_file_location(
        "_flash_kmeans_public_candidate",
        source,
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
    """Construct the documented input tensor without a full FP32 x copy."""

    output = torch.empty(x.shape[:2], device=x.device, dtype=torch.float32)
    chunk_points = 262_144
    for start in range(0, x.shape[1], chunk_points):
        end = min(start + chunk_points, x.shape[1])
        chunk = x[:, start:end].float()
        output[:, start:end] = chunk.square().sum(-1)
    return output


def assert_unchanged(
    values: tuple[torch.Tensor, ...],
    copies: tuple[torch.Tensor, ...],
    label: str,
) -> None:
    if not all(torch.equal(value, copy) for value, copy in zip(values, copies)):
        raise AssertionError(f"{label} mutated an input")


def validate_labels(
    labels: Any,
    *,
    b: int,
    n: int,
    k: int,
) -> torch.Tensor:
    if not isinstance(labels, torch.Tensor):
        raise AssertionError("assign must return a tensor")
    if labels.device.type != "cuda":
        raise AssertionError("assign output must be on CUDA")
    if labels.shape != (b, n):
        raise AssertionError(
            f"assign output shape must be {(b, n)}, got {tuple(labels.shape)}"
        )
    if labels.dtype not in INTEGER_DTYPES:
        raise AssertionError("assign output must have an integer dtype")
    if labels.numel() and (
        int(labels.min().item()) < 0 or int(labels.max().item()) >= k
    ):
        raise AssertionError("assign returned a cluster id outside [0,K)")
    return labels


def validate_update(
    output: Any,
    *,
    b: int,
    n: int,
    k: int,
    d: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(output, tuple) or len(output) != 2:
        raise AssertionError("update must return (counts, new_centroids)")
    counts, centroids = output
    if not isinstance(counts, torch.Tensor) or not isinstance(
        centroids, torch.Tensor
    ):
        raise AssertionError("update outputs must be tensors")
    if counts.device.type != "cuda" or centroids.device.type != "cuda":
        raise AssertionError("update outputs must be on CUDA")
    if counts.shape != (b, k) or counts.dtype not in INTEGER_DTYPES:
        raise AssertionError("counts must be a CUDA integer tensor [B,K]")
    if centroids.shape != (b, k, d) or centroids.dtype != torch.float16:
        raise AssertionError("new_centroids must be CUDA FP16 [B,K,D]")
    if int(counts.to(torch.int64).sum().item()) != b * n:
        raise AssertionError("counts must sum to B*N")
    if not bool(torch.isfinite(centroids).all().item()):
        raise AssertionError("new_centroids contains a non-finite value")
    return counts, centroids


def golden_assignment(
    assign: Callable[..., Any],
) -> dict[str, Any]:
    x = torch.tensor(
        [
            [[0, 0], [1, 0], [0, 1], [1, 1], [2, 0]],
            [[-1, 0], [1, 0], [0, -1], [0, 1], [3, 3]],
        ],
        device="cuda",
        dtype=torch.float16,
    )
    centroids = torch.tensor(
        [
            [[0, 0], [2, 0], [0, 2]],
            [[-2, 0], [2, 0], [0, 2]],
        ],
        device="cuda",
        dtype=torch.float16,
    )
    squared = x_squared(x)
    expected = torch.tensor(
        [[0, 0, 0, 0, 1], [0, 1, 0, 2, 1]],
        device="cuda",
        dtype=torch.int64,
    )
    originals = (x.clone(), centroids.clone(), squared.clone())
    actual = validate_labels(assign(x, centroids, squared), b=2, n=5, k=3)
    torch.cuda.synchronize()
    assert_unchanged((x, centroids, squared), originals, "assign")
    if not torch.equal(actual.to(torch.int64), expected):
        raise AssertionError(
            "golden assignment mismatch, including smallest-id tie behavior"
        )
    return {"name": "golden_assignment", "status": "passed"}


def golden_update(
    update: Callable[..., Any],
) -> dict[str, Any]:
    x = torch.tensor(
        [
            [[0, 0], [2, 0], [0, 2], [2, 2], [4, 0], [1, 3]],
            [[-2, 0], [0, 0], [2, 0], [0, 2], [2, 2], [4, 4]],
        ],
        device="cuda",
        dtype=torch.float16,
    )
    labels = torch.tensor(
        [[0, 0, 1, 1, 1, 2], [3, 3, 0, 0, 2, 2]],
        device="cuda",
        dtype=torch.int32,
    )
    old = torch.tensor(
        [
            [[7, 7], [8, 8], [6, 6], [9, 9]],
            [[7, 7], [-9, -9], [8, 8], [9, 9]],
        ],
        device="cuda",
        dtype=torch.float16,
    )
    expected_counts = torch.tensor(
        [[2, 3, 1, 0], [2, 0, 2, 2]],
        device="cuda",
        dtype=torch.int64,
    )
    expected_centroids = torch.tensor(
        [
            [[1, 0], [2, 1.3333333333333333], [1, 3], [9, 9]],
            [[1, 1], [-9, -9], [3, 3], [-1, 0]],
        ],
        device="cuda",
        dtype=torch.float16,
    )
    originals = (x.clone(), labels.clone(), old.clone())
    counts, centroids = validate_update(
        update(x, labels, old),
        b=2,
        n=6,
        k=4,
        d=2,
    )
    torch.cuda.synchronize()
    assert_unchanged((x, labels, old), originals, "update")
    if not torch.equal(counts.to(torch.int64), expected_counts):
        raise AssertionError("golden update counts mismatch")
    torch.testing.assert_close(
        centroids,
        expected_centroids,
        rtol=0.0,
        atol=0.0,
    )
    return {"name": "golden_update", "status": "passed"}


def golden_combined(
    assign: Callable[..., Any],
    update: Callable[..., Any],
) -> dict[str, Any]:
    x = torch.tensor(
        [[[0, 0], [1, 0], [0, 1], [1, 1], [2, 0]]],
        device="cuda",
        dtype=torch.float16,
    )
    old = torch.tensor(
        [[[0, 0], [2, 0], [0, 2]]],
        device="cuda",
        dtype=torch.float16,
    )
    squared = x_squared(x)
    expected_counts = torch.tensor(
        [[4, 1, 0]],
        device="cuda",
        dtype=torch.int64,
    )
    expected_centroids = torch.tensor(
        [[[0.5, 0.5], [2, 0], [0, 2]]],
        device="cuda",
        dtype=torch.float16,
    )
    originals = (x.clone(), old.clone(), squared.clone())
    labels = validate_labels(assign(x, old, squared), b=1, n=5, k=3)
    counts, centroids = validate_update(
        update(x, labels, old),
        b=1,
        n=5,
        k=3,
        d=2,
    )
    torch.cuda.synchronize()
    assert_unchanged((x, old, squared), originals, "combined iteration")
    if not torch.equal(counts.to(torch.int64), expected_counts):
        raise AssertionError("golden combined counts mismatch")
    torch.testing.assert_close(
        centroids,
        expected_centroids,
        rtol=0.0,
        atol=0.0,
    )
    return {"name": "golden_combined", "status": "passed"}


def memory_limit(stage: str, b: int, n: int, k: int, d: int) -> int:
    point_factor = {
        "assign": 8,
        "update": 40,
        "full": 48,
    }[stage]
    return 128 * MIB + point_factor * b * n + 8 * b * k * d


def make_probe(
    stage: str,
    *,
    quick: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if quick:
        shapes = {
            "assign": (1, 32_768, 1_024, 128),
            "update": (2, 65_536, 1_024, 128),
            "full": (1, 32_768, 1_024, 128),
        }
    else:
        shapes = {
            "assign": (1, 327_680, 3_072, 128),
            "update": (4, 262_144, 1_024, 128),
            "full": (1, 262_144, 2_048, 128),
        }
    b, n, k, d = shapes[stage]
    generator = torch.Generator(device="cuda")
    generator.manual_seed(
        {
            "assign": 4101,
            "update": 4102,
            "full": 4103,
        }[stage]
    )
    x = torch.randn(
        (b, n, d),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    old = torch.randn(
        (b, k, d),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    squared = x_squared(x)
    labels = (
        torch.arange(n, device="cuda", dtype=torch.int32)
        .remainder(k)
        .unsqueeze(0)
        .expand(b, -1)
        .contiguous()
    )
    return x, old, squared, labels


def call_stage(
    stage: str,
    assign: Callable[..., Any],
    update: Callable[..., Any],
    x: torch.Tensor,
    old: torch.Tensor,
    squared: torch.Tensor,
    labels: torch.Tensor,
) -> Any:
    if stage == "assign":
        return assign(x, old, squared)
    if stage == "update":
        return update(x, labels, old)
    assert stage == "full"
    assigned = assign(x, old, squared)
    return update(x, assigned, old)


def performance_probe(
    stage: str,
    assign: Callable[..., Any],
    update: Callable[..., Any],
    *,
    quick: bool,
) -> dict[str, Any]:
    x, old, squared, labels = make_probe(stage, quick=quick)
    b, n, d = x.shape
    k = old.shape[1]
    originals = (x.clone(), old.clone(), squared.clone(), labels.clone())

    allocated_before_call = int(torch.cuda.memory_allocated())
    torch.cuda.reset_peak_memory_stats()
    first_call_started = time.perf_counter()
    warm = call_stage(stage, assign, update, x, old, squared, labels)
    torch.cuda.synchronize()
    first_call_latency = (time.perf_counter() - first_call_started) * 1000.0
    del warm
    torch.cuda.empty_cache()

    measured_started = time.perf_counter()
    output = call_stage(stage, assign, update, x, old, squared, labels)
    torch.cuda.synchronize()
    measured_latency = (time.perf_counter() - measured_started) * 1000.0
    peak_extra = max(
        0,
        int(torch.cuda.max_memory_allocated()) - allocated_before_call,
    )
    if stage == "assign":
        validate_labels(output, b=b, n=n, k=k)
    else:
        validate_update(output, b=b, n=n, k=k, d=d)
    del output
    assert_unchanged(
        (x, old, squared, labels),
        originals,
        f"{stage} performance probe",
    )

    samples: list[float] = [measured_latency]
    for _ in range(3 if not quick else 2):
        started = time.perf_counter()
        output = call_stage(stage, assign, update, x, old, squared, labels)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
        del output
    latency = float(statistics.median(samples))
    limit = memory_limit(stage, b, n, k, d)
    return {
        "name": f"{stage}_candidate_only_probe",
        "status": "passed" if peak_extra <= limit else "failed",
        "stage": stage,
        "shape": [b, n, k, d],
        "first_call_latency_ms": first_call_latency,
        "latency_ms": latency,
        "latency_samples_ms": samples,
        "peak_extra_bytes": peak_extra,
        "memory_limit_bytes": limit,
        "memory_ratio": peak_extra / limit,
        "note": (
            "candidate-only diagnostic; no baseline or reference was executed; "
            "this steady-state row is not a formal Judge case"
        ),
    }


def fresh_cache_environment(cache_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    for name, child in (
        ("TRITON_CACHE_DIR", "triton"),
        ("CUDA_CACHE_PATH", "cuda"),
        ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
        ("XDG_CACHE_HOME", "xdg"),
        ("TMPDIR", "tmp"),
        ("TORCH_EXTENSIONS_DIR", "torch_extensions"),
    ):
        path = cache_root / child
        path.mkdir(mode=0o700)
        env[name] = str(path)
    return env


def fresh_process_probe(
    source: Path,
    case_id: str,
    *,
    quick: bool,
) -> dict[str, Any]:
    temp_root = Path(tempfile.mkdtemp(prefix=f"flash-kmeans-public-{case_id}-"))
    cache_root = temp_root / "cache"
    cache_root.mkdir(mode=0o700)
    output = temp_root / "result.json"
    command = [
        sys.executable,
        str(FRESH_WORKER),
        "--source",
        str(source),
        "--case-id",
        case_id,
        "--result",
        str(output),
        "--cache-root",
        str(cache_root),
    ]
    if quick:
        command.append("--quick")
    proc: subprocess.Popen[bytes] | None = None
    try:
        proc = subprocess.Popen(
            command,
            env=fresh_cache_environment(cache_root),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            returncode = proc.wait(timeout=FRESH_CASE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)
            return {
                "case_id": case_id,
                "measurement_mode": "fresh_process",
                "status": "failed",
                "correctness": {"status": "not_reached"},
                "error": "fresh-process public proxy timed out",
            }
        try:
            row = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            return {
                "case_id": case_id,
                "measurement_mode": "fresh_process",
                "status": "failed",
                "correctness": {"status": "not_reached"},
                "error": f"worker result unavailable: {type(exc).__name__}",
            }
        if not isinstance(row, dict):
            raise RuntimeError("fresh-process worker returned a non-object")
        if returncode != 0:
            row["status"] = "failed"
        return row
    finally:
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=10)
        shutil.rmtree(temp_root, ignore_errors=True)


def scale_ladders(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for family in ("large-K", "large-N/high-D"):
        family_rows = {
            str(row.get("rung")): row
            for row in rows
            if row.get("family") == family
        }
        small = family_rows.get("small")
        large = family_rows.get("large")
        result: dict[str, Any] = {
            "family": family,
            "status": "unavailable",
            "scaling_band": "unavailable",
            "note": (
                "relative public trend only; compare warmed throughput across "
                "visible rungs rather than hard-coding either tuple"
            ),
        }
        if (
            isinstance(small, dict)
            and isinstance(large, dict)
            and small.get("status") == "passed"
            and large.get("status") == "passed"
        ):
            small_shape = small.get("shape")
            large_shape = large.get("shape")
            small_ms = small.get("warmed_iteration_ms")
            large_ms = large.get("warmed_iteration_ms")
            if (
                isinstance(small_shape, list)
                and isinstance(large_shape, list)
                and isinstance(small_ms, (int, float))
                and isinstance(large_ms, (int, float))
                and small_ms > 0.0
                and large_ms > 0.0
            ):
                small_work = math.prod(int(value) for value in small_shape)
                large_work = math.prod(int(value) for value in large_shape)
                work_ratio = large_work / small_work
                latency_ratio = float(large_ms) / float(small_ms)
                retention = work_ratio / latency_ratio
                if retention >= 0.80:
                    band = "healthy"
                elif retention >= 0.50:
                    band = "moderate"
                elif retention >= 0.25:
                    band = "poor"
                else:
                    band = "pathological"
                result.update(
                    {
                        "status": "available",
                        "work_multiplier": work_ratio,
                        "latency_multiplier": latency_ratio,
                        "throughput_retention": retention,
                        "scaling_band": band,
                    }
                )
        output.append(result)
    return output


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    result: dict[str, Any] = {
        "schema_version": 3,
        "authoritative": False,
        "mode": "quick" if args.quick else "full",
        "deployment_protocol": "public_fresh_process_v1",
        "message": (
            "Public tests use literal golden fixtures, sampled large-proxy "
            "correctness, and candidate-only timings. Judge is authoritative."
        ),
        "tests": [],
        "performance": [],
        "deployment_cases": [],
        "scale_ladders": [],
    }
    exit_code = 0
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        version, source = select_source(root, args.candidate_version)
        result["tested_version"] = version
        result["tested_source"] = source.name
        _module, assign, update = load_candidate(source)
        for test in (
            lambda: golden_assignment(assign),
            lambda: golden_update(update),
            lambda: golden_combined(assign, update),
        ):
            result["tests"].append(test())
        for stage in ("assign", "update", "full"):
            row = performance_probe(
                stage,
                assign,
                update,
                quick=args.quick,
            )
            result["performance"].append(row)
            if row["status"] != "passed":
                exit_code = 1
        case_ids = FRESH_CASES_QUICK if args.quick else FRESH_CASES_FULL
        for case_id in case_ids:
            row = fresh_process_probe(source, case_id, quick=args.quick)
            result["deployment_cases"].append(row)
            if row.get("status") != "passed":
                exit_code = 1
        result["scale_ladders"] = scale_ladders(result["deployment_cases"])
        result["all_public_gates_ok"] = exit_code == 0
    except Exception as exc:
        exit_code = 1
        result["all_public_gates_ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / "public_results.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if exit_code == 0:
        print("PUBLIC TESTS PASSED (diagnostic only; submit for Judge score)")
    else:
        print("PUBLIC TESTS FAILED")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
