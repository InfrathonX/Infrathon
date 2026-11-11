#!/usr/bin/env python3
"""Trusted runtime for the standalone dynamic-grouped FP8 GEMM task, v4.

The kernel ABI (block-scaled FP8 quantization, the masked and prefix-sum
layouts, the BF16-source ``calc_diff`` metric) is a clean-room port of
DeepGEMM's grouped-GEMM test at ``559d79f``; no DeepGEMM code is imported or
required.  v4 keeps the v3 ABI and replaces the scored workload:

* four workload families modelling one rank of mixture-of-experts serving:
  prefill (prefix-sum, large batches), decode (both layouts, skewed per-group
  rows, some stale hints), latency (one group, or very few rows) and sequence
  (consecutive calls of one forward pass, timed per round with no
  synchronization between calls);
* a support domain with M_max up to 4096 (masked) or 131072 (prefix-sum) and
  ``G * M_max * (K + 2 * N) <= 2^34``; any ``0 <= M_g <= M_max``;
  ``expected_m_hint`` is scheduling information only;
* an exact reference (FP64 product of the dequantized FP8 operands) checked on
  every checked group, with the BF16-source metric applied only where it is
  statistically meaningful;
* zero-weight edge cases that exercise the whole domain;
* guard bands around every tensor and byte-exact (edge) or fingerprinted
  (scored) input checks;
* a decoy pre-call: the untimed call before each measurement uses the same
  shapes and hint but different row counts and tensors;
* timing and checking through references captured before any candidate code
  is imported, plus a sentinel that fails the run when a critical attribute
  has been replaced, and a physical-plausibility bound on case throughput.

Every per-call row count is a pure function of the case seed, so the Judge
recomputes it.  Public and Judge controllers provide the case manifests.
"""

from __future__ import annotations

import builtins
import hashlib
import importlib.util
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

# --------------------------------------------------------------------------
# Trusted references.  Captured when this module is imported, which both
# controllers do before loading any candidate module.
# --------------------------------------------------------------------------
_perf_counter = time.perf_counter
_Event = torch.cuda.Event
_event_record = torch.cuda.Event.record
_event_elapsed = torch.cuda.Event.elapsed_time
_cuda_synchronize = torch.cuda.synchronize
_current_stream = torch.cuda.current_stream
_isfinite = torch.isfinite
_torch_equal = torch.equal
_median = statistics.median
_fsum = math.fsum

MASKED_ENTRYPOINT = "grouped_fp8_gemm_nt_masked"
PSUM_ENTRYPOINT = "grouped_fp8_gemm_nt_psum"
ENTRYPOINTS = (MASKED_ENTRYPOINT, PSUM_ENTRYPOINT)
LAYOUTS = ("masked", "psum")
LAYOUT_ENTRYPOINT = {
    "masked": MASKED_ENTRYPOINT,
    "psum": PSUM_ENTRYPOINT,
}
# Workload families: one rank of mixture-of-experts serving.
FAMILIES = ("prefill", "decode", "latency", "sequence")
DISTRIBUTIONS = ("uniform", "single", "dirichlet")
SCALE_BLOCK = 128
PSUM_ALIGNMENT = 128
GENERATOR_PROVENANCE = "moe-serving-synthetic-v4"
SCHEMA_VERSION = 4

# Support domain.
MIN_GROUPS, MAX_GROUPS = 1, 64
MIN_M_MAX = 128
MAX_M_MAX = {"masked": 4096, "psum": 131072}
MIN_NK, MAX_NK = 128, 8192
# G * M_max * (K + 2 * N): the bytes of a, out and their row scales are bounded
# by this, so every legal call fits the declared device memory.
MAX_CAPACITY_BYTES = 1 << 34
MAX_HINT = 1 << 20

# Sequence cases: consecutive calls of one MoE forward pass on one rank.
SEQUENCE_LAYERS = 58
SEQUENCE_WEIGHT_POOL = 8
SEQUENCE_A_POOL = 2
SEQUENCE_OUT_POOL = 8

# Correctness.  The exact reference is the FP64 product of the dequantized
# FP8 operands; a correct kernel differs from it only by its final BF16
# rounding (about 1e-6, 2.4e-6 worst case on 128-element slices).  The
# BF16-source metric compares against the unquantized BF16 source GEMM and is
# dominated by quantization noise (about 6.9e-4), so it is only meaningful on
# slices with enough elements: 1.7-1.9 % of exactly correct 128-element slices
# exceed 1e-3, while 16,384-element slices peak near 7.3e-4.
EXACT_ERROR_LIMIT = 5.0e-5
SOURCE_ERROR_LIMIT = 1.0e-3
SOURCE_CHECK_MIN_ELEMENTS = 16_384
# Backwards-compatible alias used by the portable oracle protocol test.
CORRECTNESS_LIMIT = SOURCE_ERROR_LIMIT

# Memory safety.
GUARD_BYTES = 1 << 20
GUARD_FILL = 0xA5
OUT_POISON_BITS = 0x7FC1  # a BF16 quiet NaN with a payload kernels do not produce
FP8_NAN_BYTE = 0x7F  # E4M3FN NaN; fills invalid A rows, gaps and capacity
FINGERPRINT_STRIDES = ((1, 0), (3, 1), (7, 2))

# Edge-case size bounds (enforced by the validator, not part of the ABI).
EDGE_MAX_GROUP_NK = 1 << 28

# Physical plausibility: no implementation can exceed the dense FP8 tensor
# throughput of the device.  Conservative upper bounds by compute capability
# (H100 SXM dense FP8 is 1,979 TFLOPS; B200 is 4,500).  Unlisted capabilities
# are not bounded.  This is a tamper backstop, not a performance threshold.
FP8_DENSE_PEAK_BOUND_TFLOPS: dict[tuple[int, int], float] = {
    (8, 9): 1_500.0,
    (9, 0): 2_500.0,
    (10, 0): 7_000.0,
    (10, 3): 7_000.0,
    (12, 0): 2_500.0,
}


# --------------------------------------------------------------------------
# Gate-attributed failures.
# --------------------------------------------------------------------------
class GateFailure(RuntimeError):
    gate = "correctness_ok"

    def __init__(self, message: str, *, detail: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.detail = dict(detail or {})


class CorrectnessFailure(GateFailure):
    gate = "correctness_ok"


class EdgeCorrectnessFailure(GateFailure):
    gate = "edge_correctness_ok"


class MemorySafetyFailure(GateFailure):
    gate = "memory_safety_ok"


class HarnessIntegrityFailure(GateFailure):
    gate = "harness_integrity_ok"


# --------------------------------------------------------------------------
# Small numeric helpers.
# --------------------------------------------------------------------------
def finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise RuntimeError(f"{label} must be finite and positive")
    return number


def geometric_mean(values: Iterable[float]) -> float:
    numbers = [finite_positive(value, "geometric-mean input") for value in values]
    if not numbers:
        raise ValueError("geometric mean requires at least one value")
    return math.exp(_fsum(math.log(value) for value in numbers) / len(numbers))


def effective_tflops(m_sizes: Sequence[int], n: int, k: int, seconds: float) -> float:
    measured = finite_positive(seconds, "measured seconds")
    if (
        not m_sizes
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in m_sizes
        )
        or sum(m_sizes) <= 0
        or isinstance(n, bool)
        or not isinstance(n, int)
        or n <= 0
        or isinstance(k, bool)
        or not isinstance(k, int)
        or k <= 0
    ):
        raise RuntimeError("effective dimensions must be positive integers")
    return finite_positive(2.0 * sum(m_sizes) * n * k / measured / 1.0e12, "TFLOPS")


def _align(value: int, alignment: int = PSUM_ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def group_ends(m_sizes: Sequence[int]) -> list[int]:
    """Prefix-sum layout: group g starts at align(end[g-1], 128), 0 for g = 0."""
    ends: list[int] = []
    start = 0
    for size in m_sizes:
        end = start + int(size)
        ends.append(end)
        start = _align(end)
    return ends


def group_starts(m_sizes: Sequence[int]) -> list[int]:
    ends = group_ends(m_sizes)
    return [end - int(size) for size, end in zip(m_sizes, ends, strict=True)]


# --------------------------------------------------------------------------
# Case validation.
# --------------------------------------------------------------------------
SCORED_KEYS = frozenset(
    {
        "case_id", "family", "layout", "g", "m_max", "expected_m", "n", "k", "n2", "k2",
        "dist", "alpha", "hint_scale", "rows_low", "rows_high", "seed",
    }
)
EDGE_KEYS = frozenset(
    {"case_id", "layout", "category", "g", "m_max", "n", "k", "hint", "m_sizes", "seed", "sanitize"}
)


def _int_field(raw: Mapping[str, Any], name: str, *, minimum: int = 1) -> int:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _positive_number(raw: Mapping[str, Any], name: str) -> float:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _validate_nk(name: str, value: int) -> None:
    if not MIN_NK <= value <= MAX_NK or value % SCALE_BLOCK:
        raise ValueError(f"{name} must be a multiple of 128 in [{MIN_NK}, {MAX_NK}]")


def _validate_capacity(g: int, m_max: int, n: int, k: int) -> None:
    if g * m_max * (k + 2 * n) > MAX_CAPACITY_BYTES:
        raise ValueError("G * M_max * (K + 2 * N) exceeds the support domain")


def _validate_common(case: Mapping[str, Any]) -> tuple[str, int, int, int, int]:
    if not isinstance(case.get("case_id"), str) or not case["case_id"]:
        raise ValueError("case_id must be a nonempty string")
    layout = case.get("layout")
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}")
    g = _int_field(case, "g")
    m_max = _int_field(case, "m_max")
    n = _int_field(case, "n")
    k = _int_field(case, "k")
    if not MIN_GROUPS <= g <= MAX_GROUPS:
        raise ValueError(f"g must be in [{MIN_GROUPS}, {MAX_GROUPS}]")
    limit = MAX_M_MAX[str(layout)]
    if not MIN_M_MAX <= m_max <= limit or m_max % PSUM_ALIGNMENT:
        raise ValueError(f"m_max must be a multiple of 128 in [{MIN_M_MAX}, {limit}] for {layout}")
    _validate_nk("n", n)
    _validate_nk("k", k)
    _validate_capacity(g, m_max, n, k)
    _int_field(case, "seed", minimum=0)
    return str(layout), g, m_max, n, k


def _validate_layout_capacity(layout: str, g: int, m_max: int, m_sizes: Sequence[int]) -> None:
    if len(m_sizes) != g:
        raise ValueError("m_sizes must have one entry per group")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 or v > m_max for v in m_sizes):
        raise ValueError("every group size must be an integer in [0, m_max]")
    if layout == "psum" and group_ends(m_sizes)[-1] > g * m_max:
        raise ValueError("the aligned prefix-sum layout does not fit in G * m_max rows")


def case_hint(case: Mapping[str, Any]) -> int:
    """The scheduling hint of a scored case: an estimate, possibly far off."""
    return max(1, int(1.2 * int(case["expected_m"]) * float(case["hint_scale"])))


def validate_scored_case(raw: Mapping[str, Any]) -> dict[str, Any]:
    missing = sorted(SCORED_KEYS - set(raw))
    if missing:
        raise ValueError(f"scored case is missing fields: {missing}")
    extra = sorted(set(raw) - SCORED_KEYS)
    if extra:
        raise ValueError(f"scored case has unknown fields: {extra}")
    case = dict(raw)
    layout, g, m_max, _n, _k = _validate_common(case)
    family = case.get("family")
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")
    if family == "prefill" and layout != "psum":
        raise ValueError("prefill cases use the prefix-sum layout")
    expected_m = _int_field(case, "expected_m")
    _positive_number(case, "hint_scale")
    if case_hint(case) > MAX_HINT:
        raise ValueError(f"the case hint exceeds {MAX_HINT}")
    if family == "sequence":
        n2 = _int_field(case, "n2")
        k2 = _int_field(case, "k2")
        _validate_nk("n2", n2)
        _validate_nk("k2", k2)
        _validate_capacity(g, m_max, n2, k2)
    elif case.get("n2") is not None or case.get("k2") is not None:
        raise ValueError("only sequence cases carry a second shape")
    dist = case.get("dist")
    if dist not in DISTRIBUTIONS:
        raise ValueError(f"dist must be one of {DISTRIBUTIONS}")
    if dist == "single":
        low = _int_field(case, "rows_low")
        high = _int_field(case, "rows_high")
        if g != 1 or low > high or high > m_max:
            raise ValueError("a single-group case needs G = 1 and 1 <= rows_low <= rows_high <= m_max")
    elif case.get("rows_low") is not None or case.get("rows_high") is not None:
        raise ValueError("only single-group cases carry a row band")
    if dist == "dirichlet":
        _positive_number(case, "alpha")
        if expected_m > m_max:
            raise ValueError("expected_m exceeds m_max")
    elif case.get("alpha") is not None:
        raise ValueError("only Dirichlet cases carry alpha")
    if dist == "uniform":
        worst = math.ceil(1.3 * expected_m) - 1
        if worst > m_max:
            raise ValueError("m_max cannot hold the uniform row distribution")
        _validate_layout_capacity(layout, g, m_max, [worst] * g)
    return case


def validate_edge_case(raw: Mapping[str, Any]) -> dict[str, Any]:
    missing = sorted(EDGE_KEYS - set(raw))
    if missing:
        raise ValueError(f"edge case is missing fields: {missing}")
    case = dict(raw)
    layout, g, m_max, n, k = _validate_common(case)
    if not isinstance(case.get("category"), str) or not case["category"]:
        raise ValueError("category must be a nonempty string")
    hint = _int_field(case, "hint")
    if hint > MAX_HINT:
        raise ValueError(f"hint must be in [1, {MAX_HINT}]")
    if not isinstance(case.get("sanitize"), bool):
        raise ValueError("sanitize must be a boolean")
    m_sizes = case.get("m_sizes")
    if not isinstance(m_sizes, list):
        raise ValueError("m_sizes must be a list")
    _validate_layout_capacity(layout, g, m_max, m_sizes)
    if g * n * k > EDGE_MAX_GROUP_NK:
        raise ValueError("edge case exceeds the edge-suite size bound")
    return case


# Kept for callers that only distinguish the two schemas by their fields.
def validate_case(raw: Mapping[str, Any]) -> dict[str, Any]:
    if "m_sizes" in raw:
        return validate_edge_case(raw)
    return validate_scored_case(raw)


def draw_m_sizes(case: Mapping[str, Any], seed: int) -> list[int]:
    """Per-group row counts of one call, a pure function of the case and the seed.

    ``uniform``: int(expected_m * U(0.7, 1.3)) per group.  ``single``: one group
    with a row count drawn from [rows_low, rows_high].  ``dirichlet``: the call's
    rows (expected_m * G) split by Dirichlet(alpha) weights, each group clipped to
    m_max with the remainder moved to the heaviest groups that still have room.
    """
    generator = random.Random(seed)
    g, mean, m_max = int(case["g"]), int(case["expected_m"]), int(case["m_max"])
    dist = case["dist"]
    if dist == "uniform":
        sizes = [int(mean * generator.uniform(0.7, 1.3)) for _ in range(g)]
    elif dist == "single":
        sizes = [generator.randint(int(case["rows_low"]), int(case["rows_high"]))]
    else:
        total = max(1, mean * g)
        weights = [generator.gammavariate(float(case["alpha"]), 1.0) for _ in range(g)]
        scale = sum(weights)
        if scale <= 0.0:
            weights, scale = [1.0] * g, float(g)
        sizes = [min(m_max, int(weight / scale * total)) for weight in weights]
        missing = total - sum(sizes)
        order = sorted(range(g), key=lambda index: weights[index], reverse=True)
        while missing > 0:
            progressed = False
            for index in order:
                room = min(missing, m_max - sizes[index])
                if room > 0:
                    sizes[index] += room
                    missing -= room
                    progressed = True
                if missing == 0:
                    break
            if not progressed:
                break
    _validate_layout_capacity(str(case["layout"]), g, m_max, sizes)
    return sizes


SEQUENCE_HINT_SALT = 0x2D2D2D2D


def sequence_hints(case: Mapping[str, Any]) -> list[int]:
    """One scheduling hint per pooled layer, fixed for the whole case."""
    generator = random.Random(int(case["seed"]) ^ SEQUENCE_HINT_SALT)
    base = 1.2 * int(case["expected_m"]) * float(case["hint_scale"])
    return [max(1, int(base * generator.uniform(0.7, 1.3))) for _ in range(SEQUENCE_WEIGHT_POOL)]


def sequence_schedule(case: Mapping[str, Any], round_seed: int) -> list[tuple[int, int, list[int], int, int]]:
    """The calls of one sequence round: (linear, layer, m_sizes, a_index, out_index)."""
    generator = random.Random(round_seed)
    calls: list[tuple[int, int, list[int], int, int]] = []
    for layer in range(SEQUENCE_LAYERS):
        for linear in (0, 1):
            sizes = draw_m_sizes(case, generator.getrandbits(48))
            a_index = generator.randrange(SEQUENCE_A_POOL)
            out_index = generator.randrange(SEQUENCE_OUT_POOL)
            calls.append((linear, layer, sizes, a_index, out_index))
    return calls


# --------------------------------------------------------------------------
# Hardware.
# --------------------------------------------------------------------------
# Capability envelope: what the task contract actually needs, not the GPU model
# it was qualified on.  The GPU name and UUID are recorded as provenance only;
# they never gate admission.  Keep these two constants identical in the Judge
# verifier (tests assert the pair).
MIN_COMPUTE_CAPABILITY = (8, 9)
MIN_DEVICE_MEMORY_BYTES = 56 * 1024**3


def hardware_inventory(torch_module: Any = torch) -> dict[str, Any]:
    if not torch_module.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    count = int(torch_module.cuda.device_count())
    if count != 1:
        raise RuntimeError(f"exactly one visible GPU is required, found {count}")
    capability = tuple(int(value) for value in torch_module.cuda.get_device_capability(0))
    name = str(torch_module.cuda.get_device_name(0))
    properties = torch_module.cuda.get_device_properties(0)
    total_memory_bytes = int(properties.total_memory)
    if capability < MIN_COMPUTE_CAPABILITY:
        raise RuntimeError(
            f"GPU compute capability {capability} is below the task minimum "
            f"{MIN_COMPUTE_CAPABILITY}: {name}"
        )
    if total_memory_bytes < MIN_DEVICE_MEMORY_BYTES:
        raise RuntimeError(
            f"GPU memory {total_memory_bytes} bytes is below the task minimum "
            f"{MIN_DEVICE_MEMORY_BYTES} bytes: {name}"
        )
    return {
        "name": name,
        "count": count,
        "capability": list(capability),
        "uuid": str(getattr(properties, "uuid", "")),
        "total_memory_bytes": total_memory_bytes,
        "min_compute_capability": list(MIN_COMPUTE_CAPABILITY),
        "min_device_memory_bytes": MIN_DEVICE_MEMORY_BYTES,
        "torch_version": str(torch_module.__version__),
        "torch_cuda_version": str(torch_module.version.cuda),
    }


def plausibility_bound(capability: Sequence[int] | None) -> float | None:
    if not capability or len(capability) != 2:
        return None
    return FP8_DENSE_PEAK_BOUND_TFLOPS.get((int(capability[0]), int(capability[1])))


# --------------------------------------------------------------------------
# Harness integrity sentinel.
# --------------------------------------------------------------------------
_SENTINEL_ATTRIBUTES: tuple[tuple[Any, str], ...] = (
    (time, "perf_counter"),
    (time, "perf_counter_ns"),
    (time, "monotonic"),
    (time, "time"),
    (torch.cuda, "Event"),
    (torch.cuda.Event, "record"),
    (torch.cuda.Event, "elapsed_time"),
    (torch.cuda.Event, "synchronize"),
    (torch.cuda.Event, "query"),
    (torch.cuda, "synchronize"),
    (torch.cuda, "current_stream"),
    (torch.cuda, "Stream"),
    (torch, "isfinite"),
    (torch, "equal"),
    (torch, "matmul"),
    (torch, "randn"),
    (torch, "empty"),
    (torch, "full"),
    (torch, "tensor"),
    (torch, "manual_seed"),
    (torch.cuda, "manual_seed"),
    (torch.Tensor, "fill_"),
    (torch.Tensor, "copy_"),
    (torch.Tensor, "zero_"),
    (torch.Tensor, "double"),
    (torch.Tensor, "float"),
    (torch.Tensor, "to"),
    (torch.Tensor, "sum"),
    (torch.Tensor, "square"),
    (torch.Tensor, "item"),
    (torch.Tensor, "all"),
    (torch.Tensor, "eq"),
    (torch.Tensor, "ne"),
    (torch.Tensor, "view"),
    (torch.Tensor, "reshape"),
    (torch.Tensor, "clone"),
    (torch.Tensor, "data_ptr"),
    (torch.Tensor, "__mul__"),
    (torch.Tensor, "__sub__"),
    (torch.Tensor, "__getitem__"),
    (statistics, "median"),
    (math, "fsum"),
    (math, "log"),
    (math, "exp"),
    (math, "isfinite"),
    (math, "isclose"),
    (math, "sqrt"),
    (builtins, "max"),
    (builtins, "min"),
    (builtins, "sum"),
    (builtins, "float"),
    (builtins, "int"),
    (builtins, "len"),
    (builtins, "abs"),
    (builtins, "isinstance"),
    (builtins, "sorted"),
    (builtins, "print"),
    (builtins, "open"),
)


def _sentinel_flags() -> dict[str, Any]:
    matmul = torch.backends.cuda.matmul
    return {
        "allow_tf32": bool(matmul.allow_tf32),
        "allow_bf16_reduced_precision_reduction": bool(
            matmul.allow_bf16_reduced_precision_reduction
        ),
        "allow_fp16_reduced_precision_reduction": bool(
            matmul.allow_fp16_reduced_precision_reduction
        ),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "default_dtype": str(torch.get_default_dtype()),
        "deterministic": bool(torch.are_deterministic_algorithms_enabled()),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
    }


class HarnessSentinel:
    """Fails the run when a timing, checking or reduction primitive was replaced."""

    def __init__(self) -> None:
        self._expected = tuple(
            (owner, name, getattr(owner, name)) for owner, name in _SENTINEL_ATTRIBUTES
        )
        self._flags = _sentinel_flags()
        self._modules = {
            name: module
            for name in ("torch", "time", "math", "statistics", "builtins", "json")
            if (module := sys.modules.get(name)) is not None
        }
        self.checks = 0

    def check(self, where: str) -> None:
        self.checks += 1
        for owner, name, expected in self._expected:
            if getattr(owner, name, None) is not expected:
                label = getattr(owner, "__name__", type(owner).__name__)
                raise HarnessIntegrityFailure(
                    f"{label}.{name} was replaced ({where})",
                    detail={"attribute": f"{label}.{name}", "where": where},
                )
        flags = _sentinel_flags()
        if flags != self._flags:
            changed = sorted(key for key in flags if flags[key] != self._flags.get(key))
            raise HarnessIntegrityFailure(
                f"global numeric configuration changed: {changed} ({where})",
                detail={"flags": changed, "where": where},
            )
        for name, module in self._modules.items():
            if sys.modules.get(name) is not module:
                raise HarnessIntegrityFailure(
                    f"module {name} was replaced in sys.modules ({where})",
                    detail={"module": name, "where": where},
                )


# --------------------------------------------------------------------------
# Candidate loading.
# --------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_candidate(
    path: Path,
) -> tuple[ModuleType, dict[str, Callable[..., Any]], str]:
    candidate = path.resolve()
    if not candidate.is_file() or candidate.is_symlink():
        raise RuntimeError(f"candidate is not a regular file: {candidate}")
    source_sha256 = _sha256(candidate)
    module_name = f"grouped_fp8_candidate_{source_sha256[:24]}"
    spec = importlib.util.spec_from_file_location(module_name, candidate)
    if spec is None or spec.loader is None:
        raise RuntimeError("candidate module could not be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    entrypoints: dict[str, Callable[..., Any]] = {}
    for layout, name in LAYOUT_ENTRYPOINT.items():
        value = getattr(module, name, None)
        if not callable(value):
            raise RuntimeError(f"candidate must define callable {name}")
        entrypoints[layout] = value
    return module, entrypoints, source_sha256


# --------------------------------------------------------------------------
# Guarded allocation and input integrity.
# --------------------------------------------------------------------------
@dataclass
class _Region:
    name: str
    raw: Any
    nbytes: int


class GuardedArena:
    """Every tensor lives in the middle of a buffer with filled guard bands."""

    def __init__(self, guards: bool) -> None:
        self.guards = guards
        self.regions: list[_Region] = []

    def empty(self, name: str, shape: Sequence[int], dtype: Any) -> Any:
        element = torch.empty((), dtype=dtype).element_size()
        nbytes = math.prod(int(item) for item in shape) * element
        if not self.guards:
            return torch.empty(tuple(shape), dtype=dtype, device="cuda")
        raw = torch.empty((GUARD_BYTES + nbytes + GUARD_BYTES,), dtype=torch.uint8, device="cuda")
        raw[:GUARD_BYTES].fill_(GUARD_FILL)
        raw[GUARD_BYTES + nbytes :].fill_(GUARD_FILL)
        view = raw[GUARD_BYTES : GUARD_BYTES + nbytes].view(dtype).view(tuple(shape))
        if not view.is_contiguous():  # pragma: no cover - defensive
            raise RuntimeError("guarded view is not contiguous")
        self.regions.append(_Region(name, raw, nbytes))
        return view

    def violated(self) -> list[str]:
        problems: list[str] = []
        for region in self.regions:
            head = region.raw[:GUARD_BYTES]
            tail = region.raw[GUARD_BYTES + region.nbytes :]
            if not bool(head.eq(GUARD_FILL).all().item()):
                problems.append(f"{region.name}: write before the tensor")
            if not bool(tail.eq(GUARD_FILL).all().item()):
                problems.append(f"{region.name}: write past the tensor")
        return problems


def fingerprint(tensor: Any) -> tuple[int, ...]:
    """Cheap, allocation-free summary of a tensor's bytes (sums of int32 words)."""
    flat = tensor.reshape(-1)
    nbytes = flat.numel() * flat.element_size()
    if nbytes % 4:
        words = flat.view(torch.uint8).to(torch.int32)
    else:
        words = flat.view(torch.uint8).view(torch.int32)
    values: list[int] = [int(words.numel())]
    for stride, offset in FINGERPRINT_STRIDES:
        values.append(int(words[offset::stride].sum(dtype=torch.int64).item()))
    return tuple(values)


# --------------------------------------------------------------------------
# Workload generation.
# --------------------------------------------------------------------------
def _per_token_cast_to_fp8(value: Any) -> tuple[Any, Any]:
    rows, k = (int(item) for item in value.shape)
    view = value.view(rows, k // SCALE_BLOCK, SCALE_BLOCK)
    amax = view.abs().float().amax(dim=2).clamp(1.0e-4)
    scale = amax / 448.0
    quantized = (view * (1.0 / scale.unsqueeze(2))).to(torch.float8_e4m3fn).view(rows, k)
    return quantized.contiguous(), scale.contiguous()


def _per_block_cast_to_fp8(value: Any) -> tuple[Any, Any]:
    n, k = (int(item) for item in value.shape)
    view = value.view(n // SCALE_BLOCK, SCALE_BLOCK, k // SCALE_BLOCK, SCALE_BLOCK)
    amax = view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1.0e-4)
    scale = amax / 448.0
    quantized = (view * (1.0 / scale)).to(torch.float8_e4m3fn)
    return quantized.view(n, k).contiguous(), scale.view(n // SCALE_BLOCK, k // SCALE_BLOCK).contiguous()


def _dequant_rows(quantized: Any, scale: Any) -> Any:
    rows, k = (int(item) for item in quantized.shape)
    return (
        quantized.double().view(rows, k // SCALE_BLOCK, SCALE_BLOCK)
        * scale.double().unsqueeze(2)
    ).view(rows, k)


def _dequant_blocks(quantized: Any, scale: Any) -> Any:
    n, k = (int(item) for item in quantized.shape)
    return (
        quantized.double().view(n // SCALE_BLOCK, SCALE_BLOCK, k // SCALE_BLOCK, SCALE_BLOCK)
        * scale.double().view(n // SCALE_BLOCK, 1, k // SCALE_BLOCK, 1)
    ).view(n, k)


@dataclass
class Invocation:
    layout: str
    g: int
    m_max: int
    n: int
    k: int
    hint: int
    m_sizes: list[int]
    arena: GuardedArena
    a: Any
    b: Any
    a_scale: Any
    b_scale: Any
    metadata: Any
    out: Any
    output_slices: list[Any]
    exact_refs: list[Any] = field(default_factory=list)
    source_refs: list[Any] = field(default_factory=list)
    pristine: dict[str, Any] = field(default_factory=dict)
    fingerprints: dict[str, tuple[int, ...]] = field(default_factory=dict)

    @property
    def inputs(self) -> dict[str, Any]:
        return {
            "a": self.a,
            "b": self.b,
            "a_scale": self.a_scale,
            "b_scale": self.b_scale,
            "metadata": self.metadata,
        }


def prepare_invocation(
    layout: str,
    g: int,
    m_max: int,
    n: int,
    k: int,
    m_sizes: Sequence[int],
    hint: int,
    seed: int,
    *,
    references: bool,
    guards: bool = True,
    pristine: bool = False,
    fingerprints: bool = False,
) -> Invocation:
    """Generate one invocation group by group, with NaN in every invalid A row."""
    sizes = [int(value) for value in m_sizes]
    _validate_layout_capacity(layout, g, m_max, sizes)
    arena = GuardedArena(guards)
    k_blocks = k // SCALE_BLOCK
    n_blocks = n // SCALE_BLOCK
    rows_total = g * m_max
    if layout == "masked":
        a = arena.empty("a", (g, m_max, k), torch.float8_e4m3fn)
        a_scale = arena.empty("a_scale", (g, m_max, k_blocks), torch.float32)
        out = arena.empty("out", (g, m_max, n), torch.bfloat16)
        metadata_values = sizes
    else:
        a = arena.empty("a", (rows_total, k), torch.float8_e4m3fn)
        a_scale = arena.empty("a_scale", (rows_total, k_blocks), torch.float32)
        out = arena.empty("out", (rows_total, n), torch.bfloat16)
        metadata_values = group_ends(sizes)
    a.view(torch.uint8).fill_(FP8_NAN_BYTE)
    a_scale.fill_(float("nan"))
    out.view(torch.int16).fill_(OUT_POISON_BITS)
    b = arena.empty("b", (g, n, k), torch.float8_e4m3fn)
    b_scale = arena.empty("b_scale", (g, n_blocks, k_blocks), torch.float32)
    metadata = arena.empty("metadata", (g,), torch.int32)
    metadata.copy_(torch.tensor(metadata_values, dtype=torch.int32))

    starts = group_starts(sizes)
    output_slices: list[Any] = []
    exact_refs: list[Any] = []
    source_refs: list[Any] = []
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    for group, rows in enumerate(sizes):
        b_source = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
        b_quantized, b_block_scale = _per_block_cast_to_fp8(b_source)
        b[group].copy_(b_quantized)
        b_scale[group].copy_(b_block_scale)
        if layout == "masked":
            row_slice = (group, slice(0, rows))
            output_slices.append(out[group, :rows])
        else:
            row_slice = (slice(starts[group], starts[group] + rows),)
            output_slices.append(out[starts[group] : starts[group] + rows])
        if rows == 0:
            exact_refs.append(None)
            source_refs.append(None)
            del b_source, b_quantized, b_block_scale
            continue
        a_source = torch.randn((rows, k), dtype=torch.bfloat16, device="cuda")
        a_quantized, a_row_scale = _per_token_cast_to_fp8(a_source)
        a[row_slice].copy_(a_quantized)
        a_scale[row_slice].copy_(a_row_scale)
        if references:
            exact = _dequant_rows(a_quantized, a_row_scale) @ _dequant_blocks(
                b_quantized, b_block_scale
            ).t()
            exact_refs.append(exact.float())
            source_refs.append(torch.matmul(a_source, b_source.t()))
            del exact
        else:
            exact_refs.append(None)
            source_refs.append(None)
        del a_source, a_quantized, a_row_scale, b_source, b_quantized, b_block_scale

    invocation = Invocation(
        layout=layout,
        g=g,
        m_max=m_max,
        n=n,
        k=k,
        hint=int(hint),
        m_sizes=sizes,
        arena=arena,
        a=a,
        b=b,
        a_scale=a_scale,
        b_scale=b_scale,
        metadata=metadata,
        out=out,
        output_slices=output_slices,
        exact_refs=exact_refs,
        source_refs=source_refs,
    )
    if pristine:
        invocation.pristine = {name: tensor.clone() for name, tensor in invocation.inputs.items()}
    if fingerprints:
        invocation.fingerprints = {
            name: fingerprint(tensor) for name, tensor in invocation.inputs.items()
        }
    return invocation


def invoke(entrypoints: Mapping[str, Callable[..., Any]], prepared: Invocation) -> None:
    entrypoint = entrypoints[prepared.layout]
    returned = entrypoint(
        prepared.a,
        prepared.b,
        prepared.a_scale,
        prepared.b_scale,
        prepared.metadata,
        prepared.hint,
        prepared.out,
    )
    if returned is not None:
        raise CorrectnessFailure(f"{LAYOUT_ENTRYPOINT[prepared.layout]} must return None")


def poison_valid_output(prepared: Invocation) -> None:
    for output in prepared.output_slices:
        if output.numel():
            output.view(torch.int16).fill_(OUT_POISON_BITS)


# --------------------------------------------------------------------------
# Checks.
# --------------------------------------------------------------------------
def _diff_terms(value: Any, reference: Any) -> tuple[float, float]:
    value64 = value.double()
    reference64 = reference.double()
    cross = float((value64 * reference64).sum().item())
    squares = float((value64.square() + reference64.square()).sum().item())
    return cross, squares


def normalized_error(cross: float, squares: float) -> float:
    if squares == 0.0:
        return 0.0
    return 1.0 - 2.0 * cross / squares


@dataclass
class OutputCheck:
    max_exact_error: float = 0.0
    max_source_error: float = 0.0
    source_checked_elements: int = 0
    pooled_elements: int = 0


def check_output(prepared: Invocation, failure: type[GateFailure]) -> OutputCheck:
    result = OutputCheck()
    pooled_cross = 0.0
    pooled_squares = 0.0
    pooled_elements = 0
    for group, output in enumerate(prepared.output_slices):
        if output.numel() == 0:
            continue
        if not bool(_isfinite(output).all().item()):
            raise failure(
                f"group {group}: valid output contains an unwritten or non-finite value",
                detail={"kind": "unwritten_or_nonfinite", "group": group},
            )
        exact = prepared.exact_refs[group]
        source = prepared.source_refs[group]
        if exact is None or source is None:
            raise RuntimeError("invocation was prepared without references")
        cross, squares = _diff_terms(output, exact)
        exact_error = normalized_error(cross, squares)
        if not math.isfinite(exact_error) or exact_error >= EXACT_ERROR_LIMIT:
            raise failure(
                f"group {group}: error {exact_error:.3e} against the exact FP8 reference "
                f"exceeds {EXACT_ERROR_LIMIT:.1e}",
                detail={"kind": "exact_mismatch", "group": group, "error": exact_error},
            )
        result.max_exact_error = max(result.max_exact_error, exact_error)
        cross, squares = _diff_terms(output, source)
        elements = int(output.numel())
        if elements >= SOURCE_CHECK_MIN_ELEMENTS:
            source_error = normalized_error(cross, squares)
            if not math.isfinite(source_error) or source_error >= SOURCE_ERROR_LIMIT:
                raise failure(
                    f"group {group}: upstream metric {source_error:.3e} exceeds "
                    f"{SOURCE_ERROR_LIMIT:.1e}",
                    detail={"kind": "source_mismatch", "group": group, "error": source_error},
                )
            result.max_source_error = max(result.max_source_error, source_error)
            result.source_checked_elements += elements
        else:
            pooled_cross += cross
            pooled_squares += squares
            pooled_elements += elements
    if pooled_elements >= SOURCE_CHECK_MIN_ELEMENTS:
        source_error = normalized_error(pooled_cross, pooled_squares)
        if not math.isfinite(source_error) or source_error >= SOURCE_ERROR_LIMIT:
            raise failure(
                f"pooled small groups: upstream metric {source_error:.3e} exceeds "
                f"{SOURCE_ERROR_LIMIT:.1e}",
                detail={"kind": "source_mismatch_pooled", "error": source_error},
            )
        result.max_source_error = max(result.max_source_error, source_error)
        result.source_checked_elements += pooled_elements
    result.pooled_elements = pooled_elements
    return result


def check_memory(prepared: Invocation, where: str) -> None:
    problems = prepared.arena.violated()
    if problems:
        raise MemorySafetyFailure(
            f"out-of-bounds write detected ({where}): {problems[:4]}",
            detail={"kind": "guard_band", "where": where, "regions": problems[:8]},
        )
    if prepared.pristine:
        changed = [
            name
            for name, tensor in prepared.inputs.items()
            if not _torch_equal(
                tensor.reshape(-1).view(torch.uint8),
                prepared.pristine[name].reshape(-1).view(torch.uint8),
            )
        ]
        if changed:
            raise MemorySafetyFailure(
                f"candidate modified its inputs ({where}): {changed}",
                detail={"kind": "input_modified", "where": where, "inputs": changed},
            )
    if prepared.fingerprints:
        changed = [
            name
            for name, tensor in prepared.inputs.items()
            if fingerprint(tensor) != prepared.fingerprints[name]
        ]
        if changed:
            raise MemorySafetyFailure(
                f"candidate modified its inputs ({where}): {changed}",
                detail={"kind": "input_modified", "where": where, "inputs": changed},
            )


def classify_exception(exc: BaseException) -> str:
    """Map an exception raised inside a candidate call to the gate it fails."""
    if isinstance(exc, GateFailure):
        return exc.gate
    text = f"{type(exc).__name__}: {exc}".lower()
    for marker in (
        "illegal memory access",
        "misaligned address",
        "out of bounds",
        "illegal address",
        "unspecified launch failure",
    ):
        if marker in text:
            return "memory_safety_ok"
    return "correctness_ok"


# --------------------------------------------------------------------------
# Timing.
# --------------------------------------------------------------------------
def _measure_once(
    entrypoints: Mapping[str, Callable[..., Any]], prepared: Invocation
) -> tuple[float, float, float]:
    start_event = _Event(enable_timing=True)
    end_event = _Event(enable_timing=True)
    _cuda_synchronize()
    wall_start = _perf_counter()
    _event_record(start_event, _current_stream())
    invoke(entrypoints, prepared)
    _event_record(end_event, _current_stream())
    _cuda_synchronize()
    host_wall = finite_positive(_perf_counter() - wall_start, "host wall time")
    event_seconds = finite_positive(
        float(_event_elapsed(start_event, end_event)) / 1000.0, "CUDA event time"
    )
    return max(host_wall, event_seconds), host_wall, event_seconds


# --------------------------------------------------------------------------
# Suites.
# --------------------------------------------------------------------------
DECOY_SEED_SALT = 0x3C3C3C3C
WARMUP_SEED_SALT = 0x5A5A5A5A
TRIAL_SEED_STRIDE = 104729


def trial_seed(case_seed: int, trial: int) -> int:
    return int(case_seed) + (trial + 1) * TRIAL_SEED_STRIDE


def _positive_count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def run_single_case(
    entrypoints: Mapping[str, Callable[..., Any]],
    raw_case: Mapping[str, Any],
    *,
    trials: int,
    warmups: int,
    sentinel: HarnessSentinel,
    bound_tflops: float | None,
) -> dict[str, Any]:
    """One prefill, decode or latency case: warm-up, then timed single calls."""
    case = validate_scored_case(raw_case)
    if case["family"] == "sequence":
        raise ValueError("sequence cases run through run_sequence_case")
    _positive_count(trials, "trials")
    _positive_count(warmups, "warmups")
    layout = str(case["layout"])
    g, m_max, n, k = (int(case[name]) for name in ("g", "m_max", "n", "k"))
    hint = case_hint(case)
    seed = int(case["seed"])

    warm_seed = seed ^ WARMUP_SEED_SALT
    warm = prepare_invocation(
        layout, g, m_max, n, k, draw_m_sizes(case, warm_seed), hint, warm_seed,
        references=True, fingerprints=True,
    )
    for _ in range(warmups):
        poison_valid_output(warm)
        invoke(entrypoints, warm)
        sentinel.check("warm-up call")
    _cuda_synchronize()
    checked = check_output(warm, CorrectnessFailure)
    check_memory(warm, "warm-up")
    max_exact_error = checked.max_exact_error
    max_source_error = checked.max_source_error
    del warm
    torch.cuda.empty_cache()

    trial_us: list[float] = []
    trial_host_us: list[float] = []
    trial_event_us: list[float] = []
    trial_tflops: list[float] = []
    trial_m_sizes: list[list[int]] = []
    trial_valid_m: list[int] = []
    for trial in range(trials):
        measured_seed = trial_seed(seed, trial)
        # Decoy pre-call: same shapes and hint, different row counts, data and
        # tensors.  It absorbs compilation keyed by shape or hint, and it is
        # released before the measured tensors exist.
        decoy_seed = measured_seed ^ DECOY_SEED_SALT
        decoy = prepare_invocation(
            layout, g, m_max, n, k, draw_m_sizes(case, decoy_seed), hint, decoy_seed,
            references=False,
        )
        invoke(entrypoints, decoy)
        _cuda_synchronize()
        sentinel.check("decoy call")
        check_memory(decoy, "decoy call")
        del decoy
        m_sizes = draw_m_sizes(case, measured_seed)
        prepared = prepare_invocation(
            layout, g, m_max, n, k, m_sizes, hint, measured_seed,
            references=True, fingerprints=True,
        )
        measured, host_wall, event_seconds = _measure_once(entrypoints, prepared)
        sentinel.check("measured call")
        checked = check_output(prepared, CorrectnessFailure)
        check_memory(prepared, "measured call")
        max_exact_error = max(max_exact_error, checked.max_exact_error)
        max_source_error = max(max_source_error, checked.max_source_error)
        value = effective_tflops(m_sizes, n, k, measured)
        trial_us.append(measured * 1.0e6)
        trial_host_us.append(host_wall * 1.0e6)
        trial_event_us.append(event_seconds * 1.0e6)
        trial_tflops.append(value)
        trial_m_sizes.append(list(m_sizes))
        trial_valid_m.append(sum(m_sizes))
        del prepared
    torch.cuda.empty_cache()

    case_tflops = finite_positive(_median(trial_tflops), "case median TFLOPS")
    if bound_tflops is not None and case_tflops > bound_tflops:
        raise HarnessIntegrityFailure(
            f"case {case['case_id']} reports {case_tflops:.1f} TFLOPS, above the "
            f"physical FP8 bound {bound_tflops:.0f} for this device",
            detail={"kind": "implausible_throughput", "case_id": case["case_id"]},
        )
    return {
        "case_id": str(case["case_id"]),
        "family": str(case["family"]),
        "layout": layout,
        "entrypoint": LAYOUT_ENTRYPOINT[layout],
        "g": g,
        "m_max": m_max,
        "expected_m": int(case["expected_m"]),
        "expected_m_hint": hint,
        "n": n,
        "k": k,
        "dist": str(case["dist"]),
        "trial_us": trial_us,
        "trial_host_us": trial_host_us,
        "trial_event_us": trial_event_us,
        "trial_tflops": trial_tflops,
        "trial_m_sizes": trial_m_sizes,
        "trial_valid_m": trial_valid_m,
        "median_us": finite_positive(_median(trial_us), "case median us"),
        "tflops": case_tflops,
        "max_exact_error": max_exact_error,
        "max_source_error": max_source_error,
        "decoy_precalls": trials,
    }


def _sequence_weights(arena: GuardedArena, label: str, g: int, n: int, k: int) -> tuple[Any, Any]:
    b = arena.empty(f"{label}.b", (g, n, k), torch.float8_e4m3fn)
    b_scale = arena.empty(f"{label}.b_scale", (g, n // SCALE_BLOCK, k // SCALE_BLOCK), torch.float32)
    for group in range(g):
        source = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
        quantized, block_scale = _per_block_cast_to_fp8(source)
        b[group].copy_(quantized)
        b_scale[group].copy_(block_scale)
        del source, quantized, block_scale
    return b, b_scale


def _sequence_activations(
    arena: GuardedArena, label: str, layout: str, g: int, m_max: int, k: int
) -> tuple[Any, Any]:
    rows = g * m_max
    if layout == "masked":
        a = arena.empty(f"{label}.a", (g, m_max, k), torch.float8_e4m3fn)
        a_scale = arena.empty(f"{label}.a_scale", (g, m_max, k // SCALE_BLOCK), torch.float32)
    else:
        a = arena.empty(f"{label}.a", (rows, k), torch.float8_e4m3fn)
        a_scale = arena.empty(f"{label}.a_scale", (rows, k // SCALE_BLOCK), torch.float32)
    flat = a.view(rows, k)
    flat_scale = a_scale.view(rows, k // SCALE_BLOCK)
    chunk = max(1, (1 << 27) // k)
    for start in range(0, rows, chunk):
        stop = min(rows, start + chunk)
        source = torch.randn((stop - start, k), dtype=torch.bfloat16, device="cuda")
        quantized, row_scale = _per_token_cast_to_fp8(source)
        flat[start:stop].copy_(quantized)
        flat_scale[start:stop].copy_(row_scale)
        del source, quantized, row_scale
    return a, a_scale


def _sequence_slice(layout: str, tensor: Any, sizes: Sequence[int], group: int) -> Any:
    count = int(sizes[group])
    if layout == "masked":
        return tensor[group, :count]
    start = group_starts(sizes)[group]
    return tensor[start : start + count]


def run_sequence_case(
    entrypoints: Mapping[str, Callable[..., Any]],
    raw_case: Mapping[str, Any],
    *,
    rounds: int,
    sentinel: HarnessSentinel,
    bound_tflops: float | None,
) -> dict[str, Any]:
    """Consecutive calls of one MoE forward pass, timed per round without
    synchronization between calls.

    Each round is SEQUENCE_LAYERS layers x two linears.  Weights come from a
    pool of SEQUENCE_WEIGHT_POOL layers (fixed across rounds, as model weights
    are); each call draws fresh row counts and picks an activation buffer and an
    output buffer from small pools.  All output buffers are poisoned before
    each round; after it, the last call written to each output buffer is
    checked against the exact reference, so which calls are checked depends on
    the hidden seed.
    """
    case = validate_scored_case(raw_case)
    if case["family"] != "sequence":
        raise ValueError("run_sequence_case needs a sequence case")
    _positive_count(rounds, "rounds")
    layout = str(case["layout"])
    entry = entrypoints[layout]
    g, m_max = int(case["g"]), int(case["m_max"])
    shapes = ((int(case["n"]), int(case["k"])), (int(case["n2"]), int(case["k2"])))
    seed = int(case["seed"])
    hints = sequence_hints(case)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    arena = GuardedArena(True)
    weights: list[list[tuple[Any, Any]]] = []
    activations: list[list[tuple[Any, Any]]] = []
    outputs: list[list[Any]] = []
    for linear, (n, k) in enumerate(shapes):
        weights.append([_sequence_weights(arena, f"w{linear}.{layer}", g, n, k) for layer in range(SEQUENCE_WEIGHT_POOL)])
        activations.append(
            [_sequence_activations(arena, f"a{linear}.{index}", layout, g, m_max, k) for index in range(SEQUENCE_A_POOL)]
        )
        out_shape = (g, m_max, n) if layout == "masked" else (g * m_max, n)
        outputs.append([arena.empty(f"out{linear}.{index}", out_shape, torch.bfloat16) for index in range(SEQUENCE_OUT_POOL)])
    calls_per_round = SEQUENCE_LAYERS * 2
    metadata = arena.empty("metadata", (calls_per_round, g), torch.int32)
    inputs: dict[str, Any] = {"metadata": metadata}
    for linear in range(2):
        for layer, (b, b_scale) in enumerate(weights[linear]):
            inputs[f"w{linear}.{layer}.b"] = b
            inputs[f"w{linear}.{layer}.b_scale"] = b_scale
        for index, (a, a_scale) in enumerate(activations[linear]):
            inputs[f"a{linear}.{index}.a"] = a
            inputs[f"a{linear}.{index}.a_scale"] = a_scale

    def load_round(calls: Sequence[tuple[int, int, list[int], int, int]]) -> None:
        rows = [
            sizes if layout == "masked" else group_ends(sizes)
            for _linear, _layer, sizes, _a, _out in calls
        ]
        metadata.copy_(torch.tensor(rows, dtype=torch.int32))
        for linear in range(2):
            for out in outputs[linear]:
                out.view(torch.int16).fill_(OUT_POISON_BITS)

    def execute(calls: Sequence[tuple[int, int, list[int], int, int]]) -> None:
        for index, (linear, layer, _sizes, a_index, out_index) in enumerate(calls):
            b, b_scale = weights[linear][layer % SEQUENCE_WEIGHT_POOL]
            a, a_scale = activations[linear][a_index]
            returned = entry(
                a, b, a_scale, b_scale, metadata[index], hints[layer % SEQUENCE_WEIGHT_POOL],
                outputs[linear][out_index],
            )
            if returned is not None:
                raise CorrectnessFailure(f"{LAYOUT_ENTRYPOINT[layout]} must return None")

    def check_inputs(where: str, expected: Mapping[str, tuple[int, ...]]) -> None:
        problems = arena.violated()
        if problems:
            raise MemorySafetyFailure(
                f"out-of-bounds write detected ({where}): {problems[:4]}",
                detail={"kind": "guard_band", "where": where, "regions": problems[:8]},
            )
        changed = [name for name, tensor in inputs.items() if fingerprint(tensor) != expected[name]]
        if changed:
            raise MemorySafetyFailure(
                f"candidate modified its inputs ({where}): {changed[:8]}",
                detail={"kind": "input_modified", "where": where, "inputs": changed[:8]},
            )

    def verify(calls: Sequence[tuple[int, int, list[int], int, int]]) -> tuple[int, float]:
        last: dict[tuple[int, int], int] = {}
        for index, (linear, _layer, _sizes, _a, out_index) in enumerate(calls):
            last[(linear, out_index)] = index
        worst = 0.0
        for index in sorted(last.values()):
            linear, layer, sizes, a_index, out_index = calls[index]
            b, b_scale = weights[linear][layer % SEQUENCE_WEIGHT_POOL]
            a, a_scale = activations[linear][a_index]
            out = outputs[linear][out_index]
            for group, count in enumerate(sizes):
                if count == 0:
                    continue
                got = _sequence_slice(layout, out, sizes, group)
                if not bool(_isfinite(got).all().item()):
                    raise CorrectnessFailure(
                        f"sequence call {index}, group {group}: valid output contains an unwritten or non-finite value",
                        detail={"kind": "unwritten_or_nonfinite", "call": index, "group": group},
                    )
                exact = _dequant_rows(
                    _sequence_slice(layout, a, sizes, group), _sequence_slice(layout, a_scale, sizes, group)
                ) @ _dequant_blocks(b[group], b_scale[group]).t()
                cross, squares = _diff_terms(got, exact)
                error = normalized_error(cross, squares)
                del exact
                if not math.isfinite(error) or error >= EXACT_ERROR_LIMIT:
                    raise CorrectnessFailure(
                        f"sequence call {index}, group {group}: error {error:.3e} against the exact "
                        f"FP8 reference exceeds {EXACT_ERROR_LIMIT:.1e}",
                        detail={"kind": "exact_mismatch", "call": index, "group": group, "error": error},
                    )
                worst = max(worst, error)
        return len(last), worst

    _cuda_synchronize()
    expected = {name: fingerprint(tensor) for name, tensor in inputs.items()}
    warm_calls = sequence_schedule(case, seed ^ WARMUP_SEED_SALT)
    load_round(warm_calls)
    expected["metadata"] = fingerprint(metadata)
    execute(warm_calls)
    _cuda_synchronize()
    sentinel.check("sequence warm-up round")
    check_inputs("sequence warm-up round", expected)
    _checked, max_exact_error = verify(warm_calls)

    round_us: list[float] = []
    round_host_us: list[float] = []
    round_event_us: list[float] = []
    round_tflops: list[float] = []
    round_flops: list[float] = []
    round_valid_m: list[list[int]] = []
    checked_calls: list[int] = []
    for ordinal in range(rounds):
        calls = sequence_schedule(case, trial_seed(seed, ordinal))
        load_round(calls)
        _cuda_synchronize()
        expected["metadata"] = fingerprint(metadata)
        start_event = _Event(enable_timing=True)
        end_event = _Event(enable_timing=True)
        _cuda_synchronize()
        wall_start = _perf_counter()
        _event_record(start_event, _current_stream())
        execute(calls)
        _event_record(end_event, _current_stream())
        _cuda_synchronize()
        host_wall = finite_positive(_perf_counter() - wall_start, "host wall time")
        event_seconds = finite_positive(
            float(_event_elapsed(start_event, end_event)) / 1000.0, "CUDA event time"
        )
        seconds = max(host_wall, event_seconds)
        sentinel.check("sequence round")
        check_inputs("sequence round", expected)
        count, worst = verify(calls)
        max_exact_error = max(max_exact_error, worst)
        flops = _fsum(
            2.0 * sum(sizes) * shapes[linear][0] * shapes[linear][1]
            for linear, _layer, sizes, _a, _out in calls
        )
        round_us.append(seconds * 1.0e6)
        round_host_us.append(host_wall * 1.0e6)
        round_event_us.append(event_seconds * 1.0e6)
        round_flops.append(flops)
        round_tflops.append(finite_positive(flops / seconds / 1.0e12, "round TFLOPS"))
        round_valid_m.append([sum(sizes) for _linear, _layer, sizes, _a, _out in calls])
        checked_calls.append(count)
    del weights, activations, outputs, inputs, metadata, arena
    torch.cuda.empty_cache()

    case_tflops = finite_positive(_median(round_tflops), "case median TFLOPS")
    if bound_tflops is not None and case_tflops > bound_tflops:
        raise HarnessIntegrityFailure(
            f"case {case['case_id']} reports {case_tflops:.1f} TFLOPS, above the "
            f"physical FP8 bound {bound_tflops:.0f} for this device",
            detail={"kind": "implausible_throughput", "case_id": case["case_id"]},
        )
    return {
        "case_id": str(case["case_id"]),
        "family": "sequence",
        "layout": layout,
        "entrypoint": LAYOUT_ENTRYPOINT[layout],
        "g": g,
        "m_max": m_max,
        "expected_m": int(case["expected_m"]),
        "n": shapes[0][0],
        "k": shapes[0][1],
        "n2": shapes[1][0],
        "k2": shapes[1][1],
        "dist": str(case["dist"]),
        "calls_per_round": calls_per_round,
        "hints": hints,
        "round_us": round_us,
        "round_host_us": round_host_us,
        "round_event_us": round_event_us,
        "round_flops": round_flops,
        "round_tflops": round_tflops,
        "round_valid_m": round_valid_m,
        "checked_calls": checked_calls,
        "median_us": finite_positive(_median(round_us), "case median us"),
        "tflops": case_tflops,
        "max_exact_error": max_exact_error,
        "max_source_error": 0.0,
    }


def run_scored_case(
    entrypoints: Mapping[str, Callable[..., Any]],
    raw_case: Mapping[str, Any],
    *,
    trials: int,
    warmups: int,
    rounds: int,
    sentinel: HarnessSentinel,
    bound_tflops: float | None,
) -> dict[str, Any]:
    if raw_case.get("family") == "sequence":
        return run_sequence_case(
            entrypoints, raw_case, rounds=rounds, sentinel=sentinel, bound_tflops=bound_tflops
        )
    return run_single_case(
        entrypoints, raw_case, trials=trials, warmups=warmups, sentinel=sentinel,
        bound_tflops=bound_tflops,
    )


def run_edge_case(
    entrypoints: Mapping[str, Callable[..., Any]],
    raw_case: Mapping[str, Any],
    *,
    sentinel: HarnessSentinel,
    guards: bool = True,
) -> dict[str, Any]:
    """Two untimed calls on one fresh, guarded invocation; raise on any failure."""
    case = validate_edge_case(raw_case)
    layout = str(case["layout"])
    g, m_max, n, k = (int(case[name]) for name in ("g", "m_max", "n", "k"))
    prepared = prepare_invocation(
        layout, g, m_max, n, k, case["m_sizes"], int(case["hint"]), int(case["seed"]),
        references=True, guards=guards, pristine=guards,
    )
    max_exact_error = 0.0
    max_source_error = 0.0
    for attempt in range(2):
        if attempt:
            poison_valid_output(prepared)
        invoke(entrypoints, prepared)
        _cuda_synchronize()
        sentinel.check(f"edge call {attempt + 1}")
        check_memory(prepared, f"edge call {attempt + 1}")
        checked = check_output(prepared, EdgeCorrectnessFailure)
        max_exact_error = max(max_exact_error, checked.max_exact_error)
        max_source_error = max(max_source_error, checked.max_source_error)
    del prepared
    torch.cuda.empty_cache()
    return {
        "case_id": str(case["case_id"]),
        "layout": layout,
        "category": str(case["category"]),
        "ok": True,
        "max_exact_error": max_exact_error,
        "max_source_error": max_source_error,
    }


def aggregate_case_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Geometric mean within each family; the primary score is the geometric
    mean of the family scores present (all four on the Judge)."""
    if not results:
        raise ValueError("cannot aggregate an empty case list")
    family_values: dict[str, list[float]] = {}
    for result in results:
        value = finite_positive(result.get("tflops"), "case TFLOPS")
        family = str(result.get("family"))
        if family not in FAMILIES:
            raise ValueError("case result has an invalid family")
        family_values.setdefault(family, []).append(value)
    family_tflops = {
        family: geometric_mean(family_values[family]) for family in FAMILIES if family in family_values
    }
    return {
        "family_tflops": family_tflops,
        "primary_tflops": geometric_mean(family_tflops.values()),
    }


GATE_ORDER = (
    "edge_correctness_ok",
    "memory_safety_ok",
    "harness_integrity_ok",
    "correctness_ok",
)


def run_suite(
    candidate_path: Path,
    scored_cases: Sequence[Mapping[str, Any]],
    edge_cases: Sequence[Mapping[str, Any]] = (),
    *,
    trials: int,
    warmups: int,
    rounds: int = 1,
    require_hardware: bool = True,
    stop_on_edge_failure: bool = True,
) -> dict[str, Any]:
    """Import one candidate, run the edge cases, then the scored cases.

    Returns a schema-4 result.  Gate failures do not raise: the result carries
    ``status = "failed"``, the failing gate and a private message, so the
    caller can report which gate failed.  Infrastructure errors still raise.
    """
    scored = [validate_scored_case(case) for case in scored_cases]
    edges = [validate_edge_case(case) for case in edge_cases]
    ids = [str(case["case_id"]) for case in (*scored, *edges)]
    if len(set(ids)) != len(ids):
        raise ValueError("suite case IDs must be unique")
    _positive_count(trials, "trials")
    _positive_count(warmups, "warmups")
    _positive_count(rounds, "rounds")

    hardware = hardware_inventory() if require_hardware else {}
    bound = plausibility_bound(hardware.get("capability")) if hardware else None
    sentinel = HarnessSentinel()
    gates = {name: None for name in GATE_ORDER}
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generator_provenance": GENERATOR_PROVENANCE,
        "entrypoints": list(ENTRYPOINTS),
        "hardware": hardware,
        "trials_per_case": trials,
        "warmups": warmups,
        "rounds_per_sequence": rounds,
        "protocol": {
            "decoy_precall": True,
            "decoy_rows": "redrawn",
            "sequence_layers": SEQUENCE_LAYERS,
            "sequence_weight_pool": SEQUENCE_WEIGHT_POOL,
            "sequence_a_pool": SEQUENCE_A_POOL,
            "sequence_out_pool": SEQUENCE_OUT_POOL,
            "guard_bytes": GUARD_BYTES,
            "exact_error_limit": EXACT_ERROR_LIMIT,
            "source_error_limit": SOURCE_ERROR_LIMIT,
            "source_check_min_elements": SOURCE_CHECK_MIN_ELEMENTS,
            "plausibility_bound_tflops": bound,
        },
        "edge": {"cases_total": len(edges), "cases_passed": 0, "cases_failed": 0, "results": []},
        "cases": [],
    }

    def fail(gate: str, message: str, detail: Mapping[str, Any]) -> dict[str, Any]:
        gates[gate] = False
        payload["status"] = "failed"
        payload["failed_gate"] = gate
        payload["private_error"] = message
        payload["failure_detail"] = dict(detail)
        payload["gates"] = {name: value for name, value in gates.items()}
        payload["sentinel_checks"] = sentinel.checks
        return payload

    try:
        _module, entrypoints, source_sha256 = load_candidate(Path(candidate_path))
    except Exception as exc:
        payload["candidate_sha256"] = None
        return fail("correctness_ok", f"candidate import failed: {type(exc).__name__}: {exc}", {"kind": "import"})
    payload["candidate_sha256"] = source_sha256
    try:
        sentinel.check("candidate import")
    except HarnessIntegrityFailure as exc:
        return fail(exc.gate, str(exc), exc.detail)

    edge_failures: list[dict[str, Any]] = []
    first_edge_gate: str | None = None
    aborted = False
    for case in edges:
        try:
            row = run_edge_case(entrypoints, case, sentinel=sentinel)
            payload["edge"]["results"].append(row)
            payload["edge"]["cases_passed"] += 1
        except Exception as exc:  # noqa: BLE001 - classified below
            gate = classify_exception(exc)
            if gate == "correctness_ok":
                gate = "edge_correctness_ok"
            detail = dict(getattr(exc, "detail", {}) or {})
            row = {
                "case_id": str(case["case_id"]),
                "layout": str(case["layout"]),
                "category": str(case["category"]),
                "ok": False,
                "gate": gate,
                "error": f"{type(exc).__name__}: {str(exc)[:400]}",
                "detail": detail,
            }
            payload["edge"]["results"].append(row)
            payload["edge"]["cases_failed"] += 1
            edge_failures.append(row)
            first_edge_gate = first_edge_gate or gate
            cuda_fault = gate == "memory_safety_ok" and not isinstance(exc, GateFailure)
            if cuda_fault or gate == "harness_integrity_ok":
                aborted = True
                break
    pending_edge_failure: tuple[str, str, dict[str, Any]] | None = None
    if edge_failures:
        # Edge correctness is the gate of record unless memory safety or harness
        # integrity failed, which outranks plain wrong answers.
        gate = first_edge_gate or "edge_correctness_ok"
        for row in edge_failures:
            if row["gate"] in ("harness_integrity_ok", "memory_safety_ok"):
                gate = row["gate"]
                break
        message = (
            f"{len(edge_failures)} of {len(edges)} edge cases failed; first: "
            f"{edge_failures[0]['case_id']}: {edge_failures[0]['error']}"
        )
        pending_edge_failure = (gate, message, {"kind": "edge", "failed": len(edge_failures)})
        if stop_on_edge_failure or aborted:
            # Stop when asked to, and always after an aborted edge loop (a CUDA
            # fault or a harness-integrity failure poisons the process).
            return fail(*pending_edge_failure)
    elif edges:
        gates["edge_correctness_ok"] = True

    results: list[dict[str, Any]] = []
    for case in scored:
        try:
            results.append(
                run_scored_case(
                    entrypoints,
                    case,
                    trials=trials,
                    warmups=warmups,
                    rounds=rounds,
                    sentinel=sentinel,
                    bound_tflops=bound,
                )
            )
        except Exception as exc:  # noqa: BLE001 - classified below
            gate = classify_exception(exc)
            payload["cases"] = results
            return fail(
                gate,
                f"case {case['case_id']} ({case['family']}, {case['layout']}) failed: {type(exc).__name__}: {exc}",
                {"case_id": str(case["case_id"]), **dict(getattr(exc, "detail", {}) or {})},
            )
    payload["cases"] = results
    if results:
        payload.update(aggregate_case_results(results))
    if pending_edge_failure is not None:
        return fail(*pending_edge_failure)
    gates["correctness_ok"] = True
    gates["memory_safety_ok"] = True
    gates["harness_integrity_ok"] = True
    payload["status"] = "passed"
    payload["failed_gate"] = None
    payload["private_error"] = ""
    payload["gates"] = gates
    payload["sentinel_checks"] = sentinel.checks
    return payload


def run_sanitize_cases(
    candidate_path: Path,
    edge_cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Plain allocations (no guard bands) for compute-sanitizer; one call per case."""
    cases = [validate_edge_case(case) for case in edge_cases]
    sentinel = HarnessSentinel()
    _module, entrypoints, source_sha256 = load_candidate(Path(candidate_path))
    rows: list[dict[str, Any]] = []
    for case in cases:
        prepared = prepare_invocation(
            str(case["layout"]), int(case["g"]), int(case["m_max"]), int(case["n"]),
            int(case["k"]), case["m_sizes"], int(case["hint"]), int(case["seed"]),
            references=True, guards=False,
        )
        invoke(entrypoints, prepared)
        _cuda_synchronize()
        sentinel.check("sanitizer call")
        checked = check_output(prepared, EdgeCorrectnessFailure)
        rows.append(
            {
                "case_id": str(case["case_id"]),
                "layout": str(case["layout"]),
                "max_exact_error": checked.max_exact_error,
            }
        )
        del prepared
    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_sha256": source_sha256,
        "status": "passed",
        "cases": rows,
    }
