#!/usr/bin/env python3
"""Streaming real-activation sparse-prefill measurement and absolute screening.

Quality is evaluated separately by a frozen paired downstream policy. Numerical
screening here never claims to imply preservation of downstream answer quality.
"""
from __future__ import annotations
import hashlib
import importlib.util
import io
import json
import math
import secrets
import statistics
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence
from activation_archive_v3 import (PROTOCOL, ArchiveError, validate_manifest,
    read_json, open_handle, read_tensors)
ENTRYPOINT = "minference_sparse_prefill_bf16"
QUERY_HEADS, KV_HEADS, HEAD_DIM = 32, 8, 128
GQA_GROUP = QUERY_HEADS // KV_HEADS
SOFTMAX_SCALE = HEAD_DIM ** -0.5
TRIALS_PER_SCORED_CASE = 3
TRIAL_RANDOMNESS_PROTOCOL = "private-query-row-nonce-v3"
MIN_COMPUTE_CAPABILITY = (9, 0)
MIN_DEVICE_MEMORY_BYTES = 80_000_000_000
BAD_ROW_THRESHOLD = 0.5
TAIL_QUANTILE = 0.999
ROW_NORM_EPSILON = 1e-6
REFERENCE_Q_CHUNK = 2048
REFERENCE_K_CHUNK = 8192
PROBE_ROWS = 512
TAIL_ROWS = 64


class FailureCategory(str, Enum):
    DEPENDENCY = "dependency"
    IMPORT = "candidate_import"
    CANDIDATE_EXCEPTION = "candidate_exception"
    CONTRACT = "contract"
    MEASUREMENT_ISOLATION = "measurement_isolation"
    TIMING = "timing"
    INPUT_MUTATION = "input_mutation"
    INCOMPLETE_OUTPUT = "incomplete_output"
    NONDETERMINISM = "nondeterminism"
    CORRECTNESS = "correctness"
    STALE_OUTPUT = "stale_output"
    TIMEOUT = "timeout"
    CRASH = "crash"


class RuntimeFailure(RuntimeError):
    """Fail-closed runtime error with a stable machine-readable category."""

    def __init__(self, category: FailureCategory, message: str, *, case_id: str | None = None):
        super().__init__(message)
        self.category = category
        self.case_id = case_id

    def to_record(self) -> dict[str, Any]:
        return {
            "ok": False,
            "failure_category": self.category.value,
            "case_id": self.case_id,
            "detail": str(self),
        }


@dataclass(frozen=True)
class TrustedClock:
    """CUDA timing primitives bound before candidate code is imported."""

    perf_counter: Any
    synchronize: Any
    current_stream: Any
    new_stream: Any
    stream_context: Any
    event: Any
    caller_stream: Any

    @classmethod
    def capture(cls, torch: Any) -> "TrustedClock":
        return cls(
            perf_counter=time.perf_counter,
            synchronize=torch.cuda.synchronize,
            current_stream=torch.cuda.current_stream,
            new_stream=torch.cuda.Stream,
            stream_context=torch.cuda.stream,
            event=torch.cuda.Event,
            caller_stream=torch.cuda.Stream(),
        )


@dataclass(frozen=True)
class Measurement:
    seconds: float
    host_seconds: float
    event_seconds: float
    stream: Any


def measure_call(clock: TrustedClock, invoke: Any, snapshot: Any) -> Measurement:
    """Measure one ABI call plus its immediate same-stream output snapshot."""

    caller_stream = clock.caller_stream
    start_event = clock.event(enable_timing=True)
    end_event = clock.event(enable_timing=True)
    clock.synchronize()
    wall_start = clock.perf_counter()
    try:
        with clock.stream_context(caller_stream):
            start_event.record(caller_stream)
            invoke()
            snapshot()
            end_event.record(caller_stream)
        clock.synchronize()
    except RuntimeFailure:
        raise
    except BaseException as exc:
        raise RuntimeFailure(FailureCategory.TIMING, f"measured call failed: {exc}") from exc
    host_seconds = _finite_positive(clock.perf_counter() - wall_start, "host wall time")
    event_seconds = _finite_positive(
        float(start_event.elapsed_time(end_event)) / 1000.0, "CUDA event time"
    )
    return Measurement(
        seconds=max(host_seconds, event_seconds),
        host_seconds=host_seconds,
        event_seconds=event_seconds,
        stream=caller_stream,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_candidate(
    path: Path, *, module_role: str = "candidate"
) -> tuple[ModuleType, Any, str]:
    candidate = Path(path).resolve()
    if not candidate.is_file() or candidate.is_symlink():
        raise RuntimeFailure(FailureCategory.IMPORT, f"candidate is not a regular file: {candidate}")
    source_sha256 = _sha256(candidate)
    safe_role = "".join(character for character in module_role if character.isalnum() or character == "_")
    module_name = f"minference_prefill_{safe_role}_{source_sha256[:24]}"
    spec = importlib.util.spec_from_file_location(module_name, candidate)
    if spec is None or spec.loader is None:
        raise RuntimeFailure(FailureCategory.IMPORT, "candidate module could not be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            spec.loader.exec_module(module)
    except BaseException as exc:
        sys.modules.pop(module_name, None)
        raise RuntimeFailure(FailureCategory.IMPORT, f"candidate import failed: {exc}") from exc
    entrypoint = getattr(module, ENTRYPOINT, None)
    if not callable(entrypoint):
        raise RuntimeFailure(FailureCategory.CONTRACT, f"candidate must define callable {ENTRYPOINT}")
    return module, entrypoint, source_sha256


def invoke_candidate(entrypoint: Any, arguments: Sequence[Any]) -> None:
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            returned = entrypoint(*arguments)
    except BaseException as exc:
        raise RuntimeFailure(
            FailureCategory.CANDIDATE_EXCEPTION, f"candidate ABI call failed: {exc}"
        ) from exc
    if returned is not None:
        raise RuntimeFailure(FailureCategory.CONTRACT, f"{ENTRYPOINT} must return None")


def _finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise RuntimeError(f"{label} must be finite and positive")
    return number


def geometric_mean(values: Iterable[float]) -> float:
    numbers = [_finite_positive(value, "geometric-mean input") for value in values]
    if not numbers:
        raise ValueError("geometric mean requires at least one value")
    logs = [math.log(number) for number in numbers]
    center = statistics.fmean(logs)
    return _finite_positive(math.exp(center), "geometric mean")


def dense_flops(n: int) -> int:
    """FLOPs of dense causal attention: 4 * Hq * D * n(n+1)/2 = 8192 * n * (n+1)."""

    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise RuntimeError("n must be a positive integer")
    return 2 * QUERY_HEADS * HEAD_DIM * n * (n + 1)


def effective_tflops(n: int, seconds: float) -> float:
    measured = _finite_positive(seconds, "measured seconds")
    return _finite_positive(dense_flops(n) / measured / 1.0e12, "dense-equivalent TFLOPS")


@dataclass(frozen=True)
class ErrorStats:
    finite: bool
    rows: int
    mean: float
    head_max_mean: float
    tail: float
    max: float
    bad: float

    def as_dict(self, prefix: str = "") -> dict[str, Any]:
        return {
            f"{prefix}finite": self.finite,
            f"{prefix}rows": self.rows,
            f"{prefix}mean": self.mean,
            f"{prefix}head_max_mean": self.head_max_mean,
            f"{prefix}tail": self.tail,
            f"{prefix}max": self.max,
            f"{prefix}bad": self.bad,
        }


def _use_exact_fp32(torch: Any) -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def trusted_dense_reference(torch: Any, q: Any, k: Any, v: Any, softmax_scale: float) -> Any:
    """fp32 dense causal attention with an online softmax (no fused operator).

    ``q`` is ``[1, 32, N, 128]`` and ``k``/``v`` are ``[1, 8, N, 128]``.  The
    four query heads of one GQA group are folded into the row dimension so the
    KV tensors are never expanded.
    """

    if q.ndim != 4 or tuple(q.shape[:2]) != (1, QUERY_HEADS) or int(q.shape[3]) != HEAD_DIM:
        raise RuntimeError("reference q must have shape [1, 32, N, 128]")
    n = int(q.shape[2])
    if tuple(k.shape) != (1, KV_HEADS, n, HEAD_DIM) or tuple(v.shape) != tuple(k.shape):
        raise RuntimeError("reference k/v must have shape [1, 8, N, 128]")
    _use_exact_fp32(torch)
    device = q.device
    out = torch.empty((1, QUERY_HEADS, n, HEAD_DIM), dtype=torch.float32, device=device)
    k32 = k[0].float()
    v32 = v[0].float()
    neg_inf = -float("inf")
    for q_start in range(0, n, REFERENCE_Q_CHUNK):
        q_end = min(n, q_start + REFERENCE_Q_CHUNK)
        rows = q_end - q_start
        # [8, 4*rows, D]: sub-head major, row minor.
        q32 = (
            q[0, :, q_start:q_end].float().reshape(KV_HEADS, GQA_GROUP * rows, HEAD_DIM)
            * softmax_scale
        )
        m = torch.full((KV_HEADS, GQA_GROUP * rows, 1), neg_inf, dtype=torch.float32, device=device)
        l = torch.zeros((KV_HEADS, GQA_GROUP * rows, 1), dtype=torch.float32, device=device)
        acc = torch.zeros((KV_HEADS, GQA_GROUP * rows, HEAD_DIM), dtype=torch.float32, device=device)
        row_pos = torch.arange(q_start, q_end, device=device).repeat(GQA_GROUP)[:, None]
        for k_start in range(0, q_end, REFERENCE_K_CHUNK):
            k_end = min(q_end, k_start + REFERENCE_K_CHUNK)
            scores = torch.matmul(q32, k32[:, k_start:k_end].transpose(-1, -2))
            if k_end > q_start:
                col_pos = torch.arange(k_start, k_end, device=device)[None, :]
                scores = scores.masked_fill(col_pos > row_pos, neg_inf)
            new_m = torch.maximum(m, scores.amax(dim=-1, keepdim=True))
            probs = torch.exp(scores - new_m)
            alpha = torch.exp(m - new_m)
            l = l * alpha + probs.sum(dim=-1, keepdim=True)
            acc = acc * alpha + torch.matmul(probs, v32[:, k_start:k_end])
            m = new_m
            del scores, probs
        out[0, :, q_start:q_end] = (acc / l).reshape(QUERY_HEADS, rows, HEAD_DIM)
    return out


def probe_dense_reference(
    torch: Any, q: Any, k: Any, v: Any, softmax_scale: float, rows: Any
) -> Any:
    """``trusted_dense_reference`` restricted to the query rows ``rows``.

    Attention output row *i* depends on query row *i* and on the whole causal
    key/value prefix. The selected rows evaluate the same mathematical dense
    attention, subject to FP32 reduction rounding. Computing only those rows
    makes a fresh reference affordable for every trial.
    """

    if q.ndim != 4 or tuple(q.shape[:2]) != (1, QUERY_HEADS) or int(q.shape[3]) != HEAD_DIM:
        raise RuntimeError("probe reference q must have shape [1, 32, N, 128]")
    n = int(q.shape[2])
    if tuple(k.shape) != (1, KV_HEADS, n, HEAD_DIM) or tuple(v.shape) != tuple(k.shape):
        raise RuntimeError("probe reference k/v must have shape [1, 8, N, 128]")
    if rows.ndim != 1 or int(rows.numel()) == 0:
        raise RuntimeError("probe reference needs a nonempty one-dimensional row index")
    _use_exact_fp32(torch)
    device = q.device
    probes = int(rows.numel())
    k32 = k[0].float()
    v32 = v[0].float()
    neg_inf = -float("inf")
    # [8, 4*probes, D]: sub-head major, probe-row minor, exactly as the full
    # reference folds the four query heads of a GQA group into the row axis.
    q32 = q[0].index_select(1, rows).float().reshape(KV_HEADS, GQA_GROUP * probes, HEAD_DIM)
    q32 = q32 * softmax_scale
    row_pos = rows.repeat(GQA_GROUP)[:, None]
    m = torch.full((KV_HEADS, GQA_GROUP * probes, 1), neg_inf, dtype=torch.float32, device=device)
    l = torch.zeros((KV_HEADS, GQA_GROUP * probes, 1), dtype=torch.float32, device=device)
    acc = torch.zeros(
        (KV_HEADS, GQA_GROUP * probes, HEAD_DIM), dtype=torch.float32, device=device
    )
    key_limit = int(rows.max().item()) + 1
    for k_start in range(0, key_limit, REFERENCE_K_CHUNK):
        k_end = min(key_limit, k_start + REFERENCE_K_CHUNK)
        scores = torch.matmul(q32, k32[:, k_start:k_end].transpose(-1, -2))
        col_pos = torch.arange(k_start, k_end, device=device)[None, :]
        scores = scores.masked_fill(col_pos > row_pos, neg_inf)
        new_m = torch.maximum(m, scores.amax(dim=-1, keepdim=True))
        probs = torch.exp(scores - new_m)
        alpha = torch.exp(m - new_m)
        l = l * alpha + probs.sum(dim=-1, keepdim=True)
        acc = acc * alpha + torch.matmul(probs, v32[:, k_start:k_end])
        m = new_m
        del scores, probs
    return (acc / l).reshape(QUERY_HEADS, probes, HEAD_DIM)[None].contiguous()


def row_error_stats(torch: Any, out: Any, ref: Any) -> ErrorStats:
    """Per-row relative L2 error statistics of ``out`` (bf16) versus ``ref`` (fp32)."""

    if tuple(out.shape) != tuple(ref.shape) or out.ndim != 4:
        raise RuntimeError("output and reference shapes differ")
    heads = int(out.shape[1])
    errors = []
    finite = True
    for head in range(heads):
        actual = out[0, head].float()
        expected = ref[0, head]
        finite = finite and bool(torch.isfinite(actual).all().item())
        difference = (actual - expected).norm(dim=-1)
        denominator = expected.norm(dim=-1).clamp_min(ROW_NORM_EPSILON)
        errors.append(difference / denominator)
    err = torch.stack(errors)  # [heads, N]
    rows = int(err.numel())
    if not finite or not bool(torch.isfinite(err).all().item()):
        inf = float("inf")
        return ErrorStats(False, rows, inf, inf, inf, inf, 1.0)
    flat = err.reshape(-1)
    tail_rank = min(rows, max(1, int(math.ceil(TAIL_QUANTILE * rows))))
    tail = float(torch.kthvalue(flat, tail_rank).values.item())
    return ErrorStats(
        finite=True,
        rows=rows,
        mean=float(flat.mean().item()),
        head_max_mean=float(err.mean(dim=1).max().item()),
        tail=tail,
        max=float(flat.max().item()),
        bad=float((flat > BAD_ROW_THRESHOLD).float().mean().item()),
    )


def hardware_inventory(torch: Any) -> dict[str, Any]:
    if not bool(torch.cuda.is_available()):
        raise RuntimeFailure(FailureCategory.DEPENDENCY, "CUDA is unavailable")
    count = int(torch.cuda.device_count())
    if count != 1:
        raise RuntimeFailure(
            FailureCategory.CONTRACT, f"exactly one visible GPU is required, found {count}"
        )
    name = str(torch.cuda.get_device_name(0))
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(0))
    properties = torch.cuda.get_device_properties(0)
    total_memory_bytes = int(properties.total_memory)
    if capability < MIN_COMPUTE_CAPABILITY:
        raise RuntimeFailure(
            FailureCategory.CONTRACT,
            f"GPU compute capability {capability} is below the task minimum "
            f"{MIN_COMPUTE_CAPABILITY}: {name}",
        )
    if total_memory_bytes < MIN_DEVICE_MEMORY_BYTES:
        raise RuntimeFailure(
            FailureCategory.CONTRACT,
            f"GPU memory {total_memory_bytes} bytes is below the task minimum "
            f"{MIN_DEVICE_MEMORY_BYTES} bytes: {name}",
        )
    return {
        "name": name,
        "count": count,
        "capability": list(capability),
        "uuid": str(getattr(properties, "uuid", "")),
        "total_memory_bytes": total_memory_bytes,
        "min_compute_capability": list(MIN_COMPUTE_CAPABILITY),
        "min_device_memory_bytes": MIN_DEVICE_MEMORY_BYTES,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
    }

from kernel_fidelity_v3 import LIMIT_KEYS, validate_limits, passes as fidelity_passes

def load_numerical_policy(path, *, diagnostic=False):
    policy,digest=read_json(path)
    if policy.get("protocol")!="minference-kernel-energy-fidelity-v1" or policy.get("schema_version")!=1:
        raise ArchiveError("kernel-only numerical policy required; old gross screen rejected")
    if not diagnostic and policy.get("status")!="frozen":raise ArchiveError("policy not frozen")
    if policy.get("uniform_probe_rows")!=512 or policy.get("tail_probe_rows")!=64:raise ArchiveError("probe scope differs")
    if set(policy.get("limits",{}))!={"default"}:raise ArchiveError("one independent global budget required")
    validate_limits(policy["limits"]["default"])
    return policy,digest

def case_limits(policy,group):return dict(policy["limits"]["default"])
def numeric_passes(uniform,tail,limits):return fidelity_passes(uniform,tail,limits)


def score_from_cases(cases, *, required_trials=TRIALS_PER_SCORED_CASE):
    seen, medians = set(), []
    if not cases:
        raise ValueError("score requires cases")
    for case in cases:
        cid = case.get("case_id")
        if not isinstance(cid, str) or not cid or cid in seen:
            raise ValueError("duplicate/empty score case")
        seen.add(cid)
        ns, times = case.get("trial_ns"), case.get("trial_seconds")
        if not isinstance(ns, list) or not isinstance(times, list) or len(ns) != required_trials or len(times) != required_trials:
            raise ValueError("score requires paired real lengths and timings")
        medians.append(statistics.median(effective_tflops(n, t) for n, t in zip(ns, times)))
    return geometric_mean(medians)


def _probe_rows(torch, n, nonce, fixture_id, device):
    payload = f"{TRIAL_RANDOMNESS_PROTOCOL}\0{nonce}\0{fixture_id}".encode()
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)
    generator = torch.Generator(device=device).manual_seed(seed)
    uniform = torch.randperm(n, generator=generator, device=device)[:min(n, PROBE_ROWS)]
    tail = torch.arange(max(0, n - TAIL_ROWS), n, device=device)
    # The uniform and question-tail statistics are kept separate; overlap is OK.
    return uniform, tail, torch.cat((uniform, tail))


def _checked_call(torch, tensors, entrypoint, clock, *, measured):
    q, k, v = (tensors[name] for name in ("q", "k", "v"))
    snapshots = tuple(t.clone() for t in (q, k, v))
    out = torch.full_like(q, float("nan"))
    captured = torch.empty_like(q)
    invoke = lambda: invoke_candidate(entrypoint, (q, k, v, SOFTMAX_SCALE, out))
    if measured:
        timing = measure_call(clock, invoke, lambda: captured.copy_(out))
    else:
        # Input snapshots and NaN initialization were queued on the default
        # stream. Establish their completion before invoking another stream.
        clock.synchronize()
        with clock.stream_context(clock.caller_stream):
            invoke()
            captured.copy_(out)
        clock.synchronize()
        timing = None
    if any(not bool(torch.equal(t.view(torch.uint8), s.view(torch.uint8))) for t, s in zip((q, k, v), snapshots)):
        raise RuntimeFailure(FailureCategory.INPUT_MUTATION, "candidate mutated a QKV input")
    if not bool(torch.isfinite(captured).all()):
        raise RuntimeFailure(FailureCategory.INCOMPLETE_OUTPUT, "candidate left a nonfinite or unwritten output")
    return captured, timing


def run_suite(candidate_path: Path, *, manifest_path: Path, activation_root: Path,
              numerical_policy_path: Path, suite_type="final", nonce=None,
              before_candidate_import=None, diagnostic=False, case_ids=None,
              require_hardware=True):
    """Open all private FDs before privilege drop, then stream one fixture.

    Every timed trial gets an independent same-shape layer-2 warm-up. Its output
    is discarded. No synthetic perturbations or seed-derived acceptance gates
    occur. Probe references, verification and fixture I/O are outside the timer.
    """
    import torch
    manifest, manifest_sha = read_json(manifest_path)
    validate_manifest(manifest)
    policy, policy_sha = load_numerical_policy(numerical_policy_path, diagnostic=diagnostic)
    inventory = hardware_inventory(torch) if require_hardware else {}
    if torch.__version__ != "2.8.0+cu129":
        raise RuntimeFailure(FailureCategory.DEPENDENCY, "formal runtime requires Torch 2.8.0+cu129")
    _use_exact_fp32(torch)
    clock = TrustedClock.capture(torch)
    nonce = secrets.randbits(128) if nonce is None else nonce
    if type(nonce) is not int or not 0 <= nonce < 2**128:
        raise RuntimeFailure(FailureCategory.CONTRACT, "invalid private row nonce")
    groups = manifest["case_groups"]
    if case_ids is not None:
        wanted = set(case_ids)
        groups = [g for g in groups if g["case_id"] in wanted]
        if len(groups) != len(wanted) or not groups:
            raise ArchiveError("unknown requested groups")
    handles = {}
    # Resolve/bind trusted helpers before executing candidate source.
    reader, probe, statistics_fn, checked = read_tensors, probe_dense_reference, row_error_stats, _checked_call
    try:
        required = {fid for group in groups for key in ("warmups", "trials") for fid in group[key]}
        for fid in sorted(required):
            handles[fid] = open_handle(activation_root, fid, manifest["fixtures"][fid])
        if before_candidate_import is not None:
            before_candidate_import()
        module, entrypoint, candidate_sha = load_candidate(Path(candidate_path))
        results = []
        torch.cuda.reset_peak_memory_stats()
        for group in groups:
            limits, rows = case_limits(policy, group), []
            for wid, tid in zip(group["warmups"], group["trials"]):
                warm = reader(torch, handles[wid], "cuda")
                discarded, _ = checked(torch, warm, entrypoint, clock, measured=False)
                del discarded, warm
                values = reader(torch, handles[tid], "cuda")
                f = manifest["fixtures"][tid]
                uniform_rows, tail_rows, all_rows = _probe_rows(torch, f["actual_n"], nonce, tid, values["q"].device)
                reference = probe(torch, values["q"], values["k"], values["v"], SOFTMAX_SCALE, all_rows)
                out, measurement = checked(torch, values, entrypoint, clock, measured=True)
                selected = out.index_select(2, all_rows)
                split = int(uniform_rows.numel())
                uniform = statistics_fn(torch, selected[:, :, :split], reference[:, :, :split]).as_dict()
                tail = statistics_fn(torch, selected[:, :, split:], reference[:, :, split:]).as_dict()
                passed = numeric_passes(uniform, tail, limits)
                trial = {"fixture_id": tid, "prompt_id": f["prompt_id"],
                         "input_token_sha256": f["input_token_sha256"], "actual_n": f["actual_n"],
                         "seconds": measurement.seconds, "host_seconds": measurement.host_seconds,
                         "event_seconds": measurement.event_seconds,
                         "uniform_stats": uniform, "tail_stats": tail, "limits": limits,
                         "numeric_passed": passed, "full_write_ok": True, "input_immutable_ok": True}
                rows.append(trial)
                del values, reference, selected, out, all_rows, uniform_rows, tail_rows
                if not passed and not diagnostic:
                    raise RuntimeFailure(FailureCategory.CORRECTNESS, "absolute numerical screening failed", case_id=group["case_id"])
            case = {k: group[k] for k in ("case_id", "task", "context_budget", "layer")}
            case.update(trial_ids=list(group["trials"]), trial_ns=[r["actual_n"] for r in rows],
                        trial_seconds=[r["seconds"] for r in rows],
                        trial_host_seconds=[r["host_seconds"] for r in rows],
                        trial_event_seconds=[r["event_seconds"] for r in rows], trials=rows,
                        numeric_passed=all(r["numeric_passed"] for r in rows))
            case["median_tflops"] = statistics.median(effective_tflops(n, t) for n, t in zip(case["trial_ns"], case["trial_seconds"]))
            results.append(case)
        score = score_from_cases(results)
        return {"schema_version": 3, "protocol": PROTOCOL, "suite_type": suite_type,
                "manifest_sha256": manifest_sha, "numerical_policy_sha256": policy_sha,
                "candidate_sha256": candidate_sha, "ok": all(r["numeric_passed"] for r in results),
                "score_enabled": policy.get("status") == "frozen" and not diagnostic,
                "score": score, "raw_primary_tflops": score, "hardware": inventory,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()), "cases": results,
                "evaluation_kind": "kernel-only-v3"}
    finally:
        for handle in handles.values():
            handle.close()
