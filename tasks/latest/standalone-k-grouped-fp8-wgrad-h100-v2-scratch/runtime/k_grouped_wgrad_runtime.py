#!/usr/bin/env python3
"""Trusted runtime for the standalone K-grouped FP8 wgrad GEMM task, v3.

The kernel ABI (per-channel FP8 quantization with one scale per 128-row K
block and column, K groups concatenated without gaps, the FP32 addend C, the
BF16-source ``calc_diff`` metric) is a clean-room port of DeepGEMM's own
K-grouped contiguous test at ``559d79f``; no DeepGEMM code is imported or
required.  v3 keeps the v2 ABI and replaces the scored workload:

* four workload families modelling one rank of a mixture-of-experts training
  step (the weight gradient of the expert linears): large batch (deep K per
  expert), many groups (dozens to 128 local experts, many small or empty),
  small batch (few tokens per expert) and sequence (one backward pass through
  58 layers, timed per round with no synchronization between calls);
* expert token counts split by Dirichlet weights at a fixed K_total, and a
  scheduling hint that can be stale or badly wrong;
* a support domain with ``K_total <= 262144`` and ``G * M * N <= 2^31 - 1``;
* four correctness layers: complete writes; an element-wise bound against the
  exact FP64 product of the dequantized FP8 operands; a per-group normalized
  error against the same exact product (both computed in K chunks, so the
  check's memory does not grow with K); and the BF16-source metric where it is
  statistically meaningful;
* zero-weight edge cases that exercise the whole domain;
* guard bands around every tensor and byte-exact (edge) or fingerprinted
  (scored) input checks;
* a decoy pre-call: the untimed call before each measurement uses the same
  shapes and hint but different group sizes and tensors;
* 63-bit case seeds and SHA-256-derived per-call seeds;
* timing and checking through references captured before any candidate code
  is imported, plus a sentinel that fails the run when a critical attribute
  has been replaced, and a physical-plausibility bound on case throughput.

Every per-call group size is a pure function of the case seed, so the Judge
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
_matmul = torch.matmul
_median = statistics.median
_fsum = math.fsum

WGRAD_ENTRYPOINT = "k_grouped_fp8_gemm_tn_contiguous"
ENTRYPOINTS = (WGRAD_ENTRYPOINT,)
# Workload families: one rank of a mixture-of-experts training step.
FAMILIES = ("large", "many_groups", "small_batch", "sequence")
DISTRIBUTIONS = ("uniform", "dirichlet")
SCALE_BLOCK = 128
K_ALIGNMENT = 128
GENERATOR_PROVENANCE = "moe-training-synthetic-v3"
SCHEMA_VERSION = 4

# Support domain.
MIN_GROUPS, MAX_GROUPS = 1, 128
MIN_MN, MAX_MN = 1, 8192
MAX_TOTAL_K = 262_144
MAX_OUTPUT_ELEMENTS = (1 << 31) - 1

# Sequence cases: one backward pass of the expert linears on one rank.  Each
# layer issues the second linear's weight gradient, then the first's.
SEQUENCE_LAYERS = 58
SEQUENCE_OUT_POOL = 2

# The exact check dequantizes and multiplies in K chunks of this many rows.
CHECK_CHUNK_ROWS = 8192
FINGERPRINT_STRIDES = ((1, 0), (3, 1), (7, 2))
MAX_HINT = 1 << 20
MAX_SEED = (1 << 63) - 1

# Correctness.  E is the exact FP64 product of the dequantized FP8 operands
# plus C; R[g, m, n] = sqrt(sum_k (a s_a)^2 (b s_b)^2) is the element-wise
# product scale, which grows like sqrt(K) exactly as FP32 accumulation error
# and a dropped K block do.  CPU simulation (see the v2 iteration plan):
# correct FP32 kernels stay below 3.6e-6 of R, pessimistic tensor-core
# accumulation models below 6.6e-3, while a dropped K block, a zeroed row, a
# tile left at C or a single element left at C all exceed 0.196.  The
# per-group normalized error separates reduced-precision shortcuts (BF16
# rounding of the block dot, the scale product or the dequantized operands)
# from FP32 accumulation; its limit is calibrated on H100 (RUN_RESULTS.md):
# FP8 tensor-core kernels and the BF16-MMA baseline stay at or below 3.3e-8,
# while FP16 accumulation (with C rounded to FP16) and BF16 rounding of the
# block dot, the scale product or the dequantized operands reach 8.4e-7 to
# 2.1e-6 on the deep-K scored groups.
# The upstream metric compares against the unquantized BF16 source GEMM and
# is dominated by quantization noise, so it is only applied to slices with
# enough elements (a correct 1x3 output already reads 1.006e-3).
ELEMENT_RATIO_LIMIT = 0.05
ELEMENT_ADDEND_RELATIVE = 2.0**-20
EXACT_ERROR_LIMIT = 2.0e-7
SOURCE_ERROR_LIMIT = 1.0e-3
SOURCE_CHECK_MIN_ELEMENTS = 16_384
# Backwards-compatible alias used by the portable protocol tests.
CORRECTNESS_LIMIT = SOURCE_ERROR_LIMIT

# Memory safety.
GUARD_BYTES = 1 << 20
GUARD_FILL = 0xA5
OUT_POISON_BITS = 0x7FC1A5A5  # an FP32 quiet NaN with a payload kernels do not produce

# Edge-case size bounds (enforced by the validator, not part of the ABI).
EDGE_MAX_OUTPUT_ELEMENTS = 1 << 26
EDGE_MAX_INPUT_BYTES = 1 << 28

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


def effective_tflops(ks: Sequence[int], m: int, n: int, seconds: float) -> float:
    measured = finite_positive(seconds, "measured seconds")
    if (
        not ks
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in ks)
        or sum(ks) <= 0
        or isinstance(m, bool)
        or not isinstance(m, int)
        or m <= 0
        or isinstance(n, bool)
        or not isinstance(n, int)
        or n <= 0
    ):
        raise RuntimeError("effective dimensions must be positive integers")
    return finite_positive(2.0 * sum(ks) * m * n / measured / 1.0e12, "TFLOPS")


def _align(value: int, alignment: int = K_ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def max_dynamic_k(expected_k: int) -> int:
    return _align(int(expected_k * 1.3))


def case_hint(case: Mapping[str, Any]) -> int:
    """The scheduling hint of a scored case: an estimate, possibly far off."""
    mean = int(case["total_k"]) / int(case["g"])
    return max(1, min(MAX_HINT, int(1.2 * mean * float(case["hint_scale"]))))


def derive_seed(case_seed: int, role: str, trial: int) -> int:
    """Per-call torch / Python seed: 63 bits of SHA-256(case seed, role, trial)."""
    digest = hashlib.sha256(f"kgw-v2:{int(case_seed)}:{role}:{int(trial)}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & MAX_SEED


# --------------------------------------------------------------------------
# Case validation.
# --------------------------------------------------------------------------
SCORED_KEYS = frozenset(
    {"case_id", "family", "model", "g", "m", "n", "m2", "n2", "dist", "total_k", "alpha", "hint_scale", "seed"}
)
EDGE_KEYS = frozenset({"case_id", "category", "g", "m", "n", "ks", "hint", "seed", "sanitize"})


def _int_field(raw: Mapping[str, Any], name: str, *, minimum: int = 1, maximum: int | None = None) -> int:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def _validate_shape(case: Mapping[str, Any]) -> tuple[int, int, int]:
    if not isinstance(case.get("case_id"), str) or not case["case_id"]:
        raise ValueError("case_id must be a nonempty string")
    g = _int_field(case, "g", minimum=MIN_GROUPS, maximum=MAX_GROUPS)
    m = _int_field(case, "m", minimum=MIN_MN, maximum=MAX_MN)
    n = _int_field(case, "n", minimum=MIN_MN, maximum=MAX_MN)
    if g * m * n > MAX_OUTPUT_ELEMENTS:
        raise ValueError("G * M * N exceeds the disclosed output capacity")
    _int_field(case, "seed", minimum=0, maximum=MAX_SEED)
    return g, m, n


def validate_ks(ks: Any, g: int) -> list[int]:
    if not isinstance(ks, list) or len(ks) != g:
        raise ValueError("ks must be a list with one entry per group")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 or value % K_ALIGNMENT
        for value in ks
    ):
        raise ValueError("every group K must be 0 or a positive multiple of 128")
    if sum(ks) > MAX_TOTAL_K:
        raise ValueError(f"K_total exceeds {MAX_TOTAL_K}")
    return [int(value) for value in ks]


def _positive_number(raw: Mapping[str, Any], name: str) -> float:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def validate_scored_case(raw: Mapping[str, Any]) -> dict[str, Any]:
    missing = sorted(SCORED_KEYS - set(raw))
    if missing:
        raise ValueError(f"scored case is missing fields: {missing}")
    extra = sorted(set(raw) - SCORED_KEYS)
    if extra:
        raise ValueError(f"scored case has unknown fields: {extra}")
    case = dict(raw)
    g, m, n = _validate_shape(case)
    family = case.get("family")
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")
    if not isinstance(case.get("model"), str) or not case["model"]:
        raise ValueError("model must be a nonempty string")
    total_k = _int_field(case, "total_k", minimum=K_ALIGNMENT, maximum=MAX_TOTAL_K)
    if total_k % K_ALIGNMENT:
        raise ValueError("total_k must be a multiple of 128")
    _positive_number(case, "hint_scale")
    if family == "sequence":
        m2 = _int_field(case, "m2", minimum=MIN_MN, maximum=MAX_MN)
        n2 = _int_field(case, "n2", minimum=MIN_MN, maximum=MAX_MN)
        if g * m2 * n2 > MAX_OUTPUT_ELEMENTS:
            raise ValueError("G * M2 * N2 exceeds the disclosed output capacity")
    elif case.get("m2") is not None or case.get("n2") is not None:
        raise ValueError("only sequence cases carry a second shape")
    dist = case.get("dist")
    if dist not in DISTRIBUTIONS:
        raise ValueError(f"dist must be one of {DISTRIBUTIONS}")
    if dist == "dirichlet":
        _positive_number(case, "alpha")
    else:
        if case.get("alpha") is not None:
            raise ValueError("only Dirichlet cases carry alpha")
        if family == "sequence":
            raise ValueError("sequence cases use the Dirichlet distribution")
        if total_k % g or (total_k // g) % K_ALIGNMENT:
            raise ValueError("a uniform case needs total_k = G * expected_k with expected_k a multiple of 128")
        if g * max_dynamic_k(total_k // g) > MAX_TOTAL_K:
            raise ValueError("G * aligned maximum dynamic K exceeds the disclosed capacity")
    return case


def validate_edge_case(raw: Mapping[str, Any]) -> dict[str, Any]:
    missing = sorted(EDGE_KEYS - set(raw))
    if missing:
        raise ValueError(f"edge case is missing fields: {missing}")
    case = dict(raw)
    g, m, n = _validate_shape(case)
    if not isinstance(case.get("category"), str) or not case["category"]:
        raise ValueError("category must be a nonempty string")
    _int_field(case, "hint", minimum=1, maximum=MAX_HINT)
    if not isinstance(case.get("sanitize"), bool):
        raise ValueError("sanitize must be a boolean")
    ks = validate_ks(case.get("ks"), g)
    if g * m * n > EDGE_MAX_OUTPUT_ELEMENTS or sum(ks) * (m + n) > EDGE_MAX_INPUT_BYTES:
        raise ValueError("edge case exceeds the edge-suite size bound")
    return case


# Kept for callers that only distinguish the two schemas by their fields.
def validate_case(raw: Mapping[str, Any]) -> dict[str, Any]:
    if "ks" in raw:
        return validate_edge_case(raw)
    return validate_scored_case(raw)


def draw_ks(case: Mapping[str, Any], seed: int) -> list[int]:
    """Per-group K of one call, a pure function of the case and the seed.

    ``uniform``: align(max(1, int(expected_k * U(0.7, 1.3))), 128) per group,
    with expected_k = total_k / G.  ``dirichlet``: the call's total_k / 128 K
    blocks split by Dirichlet(alpha) weights with largest-remainder rounding,
    so K_total is fixed and any group can be empty.
    """
    generator = random.Random(seed)
    g, total = int(case["g"]), int(case["total_k"])
    if case["dist"] == "uniform":
        expected = total // g
        limit = max_dynamic_k(expected)
        ks = [_align(max(1, int(expected * generator.uniform(0.7, 1.3)))) for _ in range(g)]
        if any(value < K_ALIGNMENT or value > limit for value in ks):
            raise RuntimeError("generated dynamic K is outside the allocated range")
    else:
        blocks = total // K_ALIGNMENT
        weights = [generator.gammavariate(float(case["alpha"]), 1.0) for _ in range(g)]
        scale = _fsum(weights)
        if scale <= 0.0:
            weights, scale = [1.0] * g, float(g)
        shares = [weight / scale * blocks for weight in weights]
        counts = [int(share) for share in shares]
        order = sorted(range(g), key=lambda index: (counts[index] - shares[index], index))
        for index in order[: blocks - sum(counts)]:
            counts[index] += 1
        ks = [count * K_ALIGNMENT for count in counts]
    if sum(ks) > MAX_TOTAL_K:
        raise RuntimeError("generated K_total exceeds the support domain")
    return ks


SEQUENCE_KS_SALT = "sequence-ks"


def sequence_schedule(case: Mapping[str, Any], round_seed: int) -> list[tuple[int, int, list[int], int]]:
    """The calls of one sequence round: (linear, layer, ks, out_index).

    Layers run in backward order; each layer issues the second linear, then the
    first.  Every call draws its own group sizes.
    """
    generator = random.Random(round_seed)
    calls: list[tuple[int, int, list[int], int]] = []
    for layer in reversed(range(SEQUENCE_LAYERS)):
        for linear in (1, 0):
            ks = draw_ks(case, generator.getrandbits(62))
            calls.append((linear, layer, ks, generator.randrange(SEQUENCE_OUT_POOL)))
    return calls


# --------------------------------------------------------------------------
# Hardware.
# --------------------------------------------------------------------------
# Capability envelope: what the task contract actually needs, not the GPU model
# it was qualified on.  The GPU name and UUID are recorded as provenance only;
# they never gate admission.  Keep these two constants identical in the Judge
# verifier (tests assert the pair).
MIN_COMPUTE_CAPABILITY = (8, 9)
MIN_DEVICE_MEMORY_BYTES = 48 * 1024**3


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
    (torch.Tensor, "sqrt"),
    (torch.Tensor, "abs"),
    (torch.Tensor, "amax"),
    (torch.Tensor, "max"),
    (torch.Tensor, "repeat_interleave"),
    (torch.Tensor, "item"),
    (torch.Tensor, "all"),
    (torch.Tensor, "any"),
    (torch.Tensor, "eq"),
    (torch.Tensor, "ne"),
    (torch.Tensor, "gt"),
    (torch.Tensor, "view"),
    (torch.Tensor, "reshape"),
    (torch.Tensor, "clone"),
    (torch.Tensor, "data_ptr"),
    (torch.Tensor, "__mul__"),
    (torch.Tensor, "__add__"),
    (torch.Tensor, "__sub__"),
    (torch.Tensor, "__truediv__"),
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
            for name in ("torch", "time", "math", "statistics", "builtins", "json", "hashlib")
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
    module_name = f"k_grouped_wgrad_candidate_{source_sha256[:24]}"
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
    value = getattr(module, WGRAD_ENTRYPOINT, None)
    if not callable(value):
        raise RuntimeError(f"candidate must define callable {WGRAD_ENTRYPOINT}")
    entrypoints = {WGRAD_ENTRYPOINT: value}
    return module, entrypoints, source_sha256


# --------------------------------------------------------------------------
# Guarded allocation.
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


# --------------------------------------------------------------------------
# Workload generation.
# --------------------------------------------------------------------------
def _per_channel_cast_to_fp8(value: Any) -> tuple[Any, Any]:
    """[K, columns] -> FP8 E4M3FN plus one FP32 scale per (128-row block, column)."""
    rows, columns = (int(item) for item in value.shape)
    view = value.view(rows // SCALE_BLOCK, SCALE_BLOCK, columns)
    amax = view.abs().float().amax(dim=1).clamp(1.0e-4)
    scale = amax / 448.0
    quantized = (view * (1.0 / scale.unsqueeze(1))).to(torch.float8_e4m3fn).view(rows, columns)
    return quantized.contiguous(), scale.contiguous()


def _cast_into(source: Any, quantized_out: Any, scale_out: Any) -> None:
    """Per-channel cast of [K, columns] into preallocated tensors, in K chunks."""
    rows = int(source.shape[0])
    for start in range(0, rows, CHECK_CHUNK_ROWS):
        stop = min(rows, start + CHECK_CHUNK_ROWS)
        quantized, scale = _per_channel_cast_to_fp8(source[start:stop])
        quantized_out[start:stop].copy_(quantized)
        scale_out[start // SCALE_BLOCK : stop // SCALE_BLOCK].copy_(scale)
        del quantized, scale


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


def _dequant(quantized: Any, scale: Any) -> Any:
    """FP64 dequantization: exact, FP8 (4 significant bits) times an FP32 scale."""
    return quantized.double() * scale.double().repeat_interleave(SCALE_BLOCK, dim=0)


@dataclass
class Invocation:
    g: int
    m: int
    n: int
    ks: list[int]
    hint: int
    arena: GuardedArena
    a: Any
    b: Any
    a_scale: Any
    b_scale: Any
    metadata: Any
    c: Any
    out: Any
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
            "ks": self.metadata,
            "c": self.c,
        }

    @property
    def starts(self) -> list[int]:
        starts: list[int] = []
        total = 0
        for size in self.ks:
            starts.append(total)
            total += size
        return starts


def prepare_invocation(
    g: int,
    m: int,
    n: int,
    ks: Sequence[int],
    hint: int,
    seed: int,
    *,
    references: bool,
    guards: bool = True,
    pristine: bool = True,
    fingerprints: bool = False,
) -> Invocation:
    """Generate one invocation group by group: BF16 sources, per-channel FP8
    casts, FP32 C.  Peak temporary memory is one group's BF16 sources.

    ``pristine`` keeps byte copies of every input (edge cases); ``fingerprints``
    keeps cheap summaries instead (scored cases, whose inputs can be large).
    """
    sizes = validate_ks(list(ks), g)
    total_k = sum(sizes)
    blocks = total_k // SCALE_BLOCK
    arena = GuardedArena(guards)
    a = arena.empty("a", (total_k, m), torch.float8_e4m3fn)
    b = arena.empty("b", (total_k, n), torch.float8_e4m3fn)
    a_scale = arena.empty("a_scale", (blocks, m), torch.float32)
    b_scale = arena.empty("b_scale", (blocks, n), torch.float32)
    metadata = arena.empty("ks", (g,), torch.int32)
    c = arena.empty("c", (g, m, n), torch.float32)
    out = arena.empty("out", (g, m, n), torch.float32)
    metadata.copy_(torch.tensor(sizes, dtype=torch.int32))
    out.view(torch.int32).fill_(OUT_POISON_BITS)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    c.copy_(torch.randn((g, m, n), dtype=torch.float32, device="cuda") * 32.0)
    source_refs: list[Any] = []
    start = 0
    for group, size in enumerate(sizes):
        end = start + size
        if size:
            a_source = torch.randn((size, m), dtype=torch.bfloat16, device="cuda")
            b_source = torch.randn((size, n), dtype=torch.bfloat16, device="cuda")
            if references:
                reference = c[group].clone()
                for lo in range(0, size, CHECK_CHUNK_ROWS):
                    hi = min(size, lo + CHECK_CHUNK_ROWS)
                    reference += _matmul(a_source[lo:hi].t(), b_source[lo:hi]).float()
                source_refs.append(reference)
            _cast_into(a_source, a[start:end], a_scale[start // SCALE_BLOCK : end // SCALE_BLOCK])
            _cast_into(b_source, b[start:end], b_scale[start // SCALE_BLOCK : end // SCALE_BLOCK])
            del a_source, b_source
        elif references:
            source_refs.append(c[group].clone())
        start = end

    invocation = Invocation(
        g=g,
        m=m,
        n=n,
        ks=sizes,
        hint=int(hint),
        arena=arena,
        a=a,
        b=b,
        a_scale=a_scale,
        b_scale=b_scale,
        metadata=metadata,
        c=c,
        out=out,
        source_refs=source_refs,
    )
    if pristine:
        invocation.pristine = {name: tensor.clone() for name, tensor in invocation.inputs.items()}
    elif fingerprints:
        invocation.fingerprints = {name: fingerprint(tensor) for name, tensor in invocation.inputs.items()}
    return invocation


def invoke(entrypoints: Mapping[str, Callable[..., Any]], prepared: Invocation) -> None:
    entrypoint = entrypoints[WGRAD_ENTRYPOINT]
    returned = entrypoint(
        prepared.a,
        prepared.b,
        prepared.a_scale,
        prepared.b_scale,
        prepared.metadata,
        prepared.c,
        prepared.hint,
        prepared.out,
    )
    if returned is not None:
        raise CorrectnessFailure(f"{WGRAD_ENTRYPOINT} must return None")


def poison_output(prepared: Invocation) -> None:
    prepared.out.view(torch.int32).fill_(OUT_POISON_BITS)


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
    max_element_ratio: float = 0.0
    max_exact_error: float = 0.0
    max_source_error: float = 0.0
    source_checked_elements: int = 0
    pooled_elements: int = 0


def _exact_group(prepared: Invocation, group: int) -> tuple[Any, Any]:
    """(E, R) for one group from the verified-unchanged inputs, in K chunks."""
    start = prepared.starts[group]
    size = prepared.ks[group]
    c64 = prepared.c[group].double()
    if size == 0:
        return c64, torch.zeros_like(c64)
    exact = c64
    squares = torch.zeros((prepared.m, prepared.n), dtype=torch.float32, device=c64.device)
    for lo in range(start, start + size, CHECK_CHUNK_ROWS):
        hi = min(start + size, lo + CHECK_CHUNK_ROWS)
        a = _dequant(prepared.a[lo:hi], prepared.a_scale[lo // SCALE_BLOCK : hi // SCALE_BLOCK])
        b = _dequant(prepared.b[lo:hi], prepared.b_scale[lo // SCALE_BLOCK : hi // SCALE_BLOCK])
        exact = exact + _matmul(a.t(), b)
        squares += _matmul(a.square().float().t(), b.square().float())
        del a, b
    return exact, squares.sqrt().double()


def check_output(prepared: Invocation, failure: type[GateFailure], *, source: bool = True) -> OutputCheck:
    result = OutputCheck()
    out = prepared.out
    if not bool(_isfinite(out).all().item()):
        bad = (~_isfinite(out)).view(prepared.g, -1).any(dim=1)
        group = int(bad.nonzero()[0].item()) if bool(bad.any().item()) else -1
        raise failure(
            f"group {group}: output contains an unwritten or non-finite value",
            detail={"kind": "unwritten_or_nonfinite", "group": group},
        )
    pooled_cross = 0.0
    pooled_squares = 0.0
    pooled_elements = 0
    for group in range(prepared.g):
        output = out[group]
        exact, scale = _exact_group(prepared, group)
        c_abs = prepared.c[group].double().abs()
        difference = (output.double() - exact).abs()
        allowed = ELEMENT_RATIO_LIMIT * scale + ELEMENT_ADDEND_RELATIVE * c_abs
        ratio = float((difference / (scale + ELEMENT_ADDEND_RELATIVE * c_abs + 1e-300)).max().item())
        result.max_element_ratio = max(result.max_element_ratio, ratio)
        if bool(difference.gt(allowed).any().item()):
            flat = int((difference - allowed).argmax().item())
            raise failure(
                f"group {group}: element ({flat // prepared.n}, {flat % prepared.n}) differs from "
                f"the exact FP8 product by {ratio:.3e} of its scale (limit {ELEMENT_RATIO_LIMIT})",
                detail={
                    "kind": "element_mismatch",
                    "group": group,
                    "row": flat // prepared.n,
                    "column": flat % prepared.n,
                    "ratio": ratio,
                },
            )
        cross, squares = _diff_terms(output, exact)
        exact_error = normalized_error(cross, squares)
        if not math.isfinite(exact_error) or exact_error >= EXACT_ERROR_LIMIT:
            raise failure(
                f"group {group}: normalized error {exact_error:.3e} against the exact FP8 "
                f"product exceeds {EXACT_ERROR_LIMIT:.1e}",
                detail={"kind": "precision_mismatch", "group": group, "error": exact_error},
            )
        result.max_exact_error = max(result.max_exact_error, exact_error)
        del exact, scale, c_abs, difference, allowed
        if not source:
            continue
        reference = prepared.source_refs[group] if prepared.source_refs else None
        if reference is None:
            raise RuntimeError("invocation was prepared without references")
        cross, squares = _diff_terms(output, reference)
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
    elif prepared.fingerprints:
        changed = [
            name for name, tensor in prepared.inputs.items()
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
def _positive_count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _check_bound(case: Mapping[str, Any], case_tflops: float, bound_tflops: float | None) -> None:
    if bound_tflops is not None and case_tflops > bound_tflops:
        raise HarnessIntegrityFailure(
            f"case {case['case_id']} reports {case_tflops:.1f} TFLOPS, above the "
            f"physical FP8 bound {bound_tflops:.0f} for this device",
            detail={"kind": "implausible_throughput", "case_id": case["case_id"]},
        )


def run_single_case(
    entrypoints: Mapping[str, Callable[..., Any]],
    raw_case: Mapping[str, Any],
    *,
    trials: int,
    warmups: int,
    sentinel: HarnessSentinel,
    bound_tflops: float | None,
) -> dict[str, Any]:
    """One large, many-groups or small-batch case: warm-up, then timed single calls."""
    case = validate_scored_case(raw_case)
    if case["family"] == "sequence":
        raise ValueError("sequence cases run through run_sequence_case")
    _positive_count(trials, "trials")
    _positive_count(warmups, "warmups")
    g, m, n = (int(case[name]) for name in ("g", "m", "n"))
    hint = case_hint(case)
    seed = int(case["seed"])

    warm = prepare_invocation(
        g, m, n, draw_ks(case, derive_seed(seed, "warmup-ks", 0)), hint,
        derive_seed(seed, "warmup", 0), references=True, pristine=False, fingerprints=True,
    )
    for _ in range(warmups):
        poison_output(warm)
        invoke(entrypoints, warm)
        sentinel.check("warm-up call")
    _cuda_synchronize()
    check_memory(warm, "warm-up")
    checked = check_output(warm, CorrectnessFailure)
    max_element_ratio = checked.max_element_ratio
    max_exact_error = checked.max_exact_error
    max_source_error = checked.max_source_error
    del warm
    torch.cuda.empty_cache()

    trial_us: list[float] = []
    trial_host_us: list[float] = []
    trial_event_us: list[float] = []
    trial_tflops: list[float] = []
    trial_ks: list[list[int]] = []
    trial_total_k: list[int] = []
    for trial in range(trials):
        # Decoy pre-call: same shapes and hint, different group sizes, data and
        # tensors.  It absorbs compilation keyed by shape or hint, and it is
        # released before the measured tensors exist.
        decoy = prepare_invocation(
            g, m, n, draw_ks(case, derive_seed(seed, "decoy-ks", trial)), hint,
            derive_seed(seed, "decoy", trial), references=False, pristine=False,
        )
        invoke(entrypoints, decoy)
        _cuda_synchronize()
        sentinel.check("decoy call")
        check_memory(decoy, "decoy call")
        del decoy
        ks = draw_ks(case, derive_seed(seed, "ks", trial))
        prepared = prepare_invocation(
            g, m, n, ks, hint, derive_seed(seed, "measured", trial),
            references=True, pristine=False, fingerprints=True,
        )
        measured, host_wall, event_seconds = _measure_once(entrypoints, prepared)
        sentinel.check("measured call")
        check_memory(prepared, "measured call")
        checked = check_output(prepared, CorrectnessFailure)
        max_element_ratio = max(max_element_ratio, checked.max_element_ratio)
        max_exact_error = max(max_exact_error, checked.max_exact_error)
        max_source_error = max(max_source_error, checked.max_source_error)
        value = effective_tflops(ks, m, n, measured)
        trial_us.append(measured * 1.0e6)
        trial_host_us.append(host_wall * 1.0e6)
        trial_event_us.append(event_seconds * 1.0e6)
        trial_tflops.append(value)
        trial_ks.append(list(ks))
        trial_total_k.append(sum(ks))
        del prepared
        torch.cuda.empty_cache()

    case_tflops = finite_positive(_median(trial_tflops), "case median TFLOPS")
    _check_bound(case, case_tflops, bound_tflops)
    return {
        "case_id": str(case["case_id"]),
        "family": str(case["family"]),
        "model": str(case["model"]),
        "entrypoint": WGRAD_ENTRYPOINT,
        "g": g,
        "m": m,
        "n": n,
        "total_k": int(case["total_k"]),
        "dist": str(case["dist"]),
        "expected_k_hint": hint,
        "trial_us": trial_us,
        "trial_host_us": trial_host_us,
        "trial_event_us": trial_event_us,
        "trial_tflops": trial_tflops,
        "trial_ks": trial_ks,
        "trial_total_k": trial_total_k,
        "median_us": finite_positive(_median(trial_us), "case median us"),
        "tflops": case_tflops,
        "max_element_ratio": max_element_ratio,
        "max_exact_error": max_exact_error,
        "max_source_error": max_source_error,
        "decoy_precalls": trials,
    }


def run_sequence_case(
    entrypoints: Mapping[str, Callable[..., Any]],
    raw_case: Mapping[str, Any],
    *,
    rounds: int,
    sentinel: HarnessSentinel,
    bound_tflops: float | None,
) -> dict[str, Any]:
    """One backward pass through SEQUENCE_LAYERS layers, timed per round with no
    synchronization between calls.

    Each linear has one set of operands (A, B and their scales), one addend C
    and SEQUENCE_OUT_POOL output buffers.  K_total is fixed per case, so every
    call reuses A and B with fresh group sizes.  Output buffers are poisoned
    before each round; after it, the last call written to each buffer is
    checked against the exact reference, so which calls are checked depends on
    the hidden seed.
    """
    case = validate_scored_case(raw_case)
    if case["family"] != "sequence":
        raise ValueError("run_sequence_case needs a sequence case")
    _positive_count(rounds, "rounds")
    entry = entrypoints[WGRAD_ENTRYPOINT]
    g, total_k = int(case["g"]), int(case["total_k"])
    shapes = ((int(case["m"]), int(case["n"])), (int(case["m2"]), int(case["n2"])))
    hint = case_hint(case)
    seed = int(case["seed"])

    linears = []
    for linear, (m, n) in enumerate(shapes):
        base = prepare_invocation(
            g, m, n, draw_ks(case, derive_seed(seed, f"sequence-base-{linear}", 0)), hint,
            derive_seed(seed, f"sequence-data-{linear}", 0), references=False, pristine=False,
        )
        outs = [base.out] + [
            base.arena.empty(f"out{index}", (g, m, n), torch.float32) for index in range(1, SEQUENCE_OUT_POOL)
        ]
        linears.append((base, outs))
    calls_per_round = SEQUENCE_LAYERS * 2
    meta_arena = GuardedArena(True)
    metadata = meta_arena.empty("ks", (calls_per_round, g), torch.int32)

    def inputs() -> dict[str, Any]:
        table: dict[str, Any] = {"metadata": metadata}
        for linear, (base, _outs) in enumerate(linears):
            for name in ("a", "b", "a_scale", "b_scale", "c"):
                table[f"{linear}.{name}"] = getattr(base, name)
        return table

    def load_round(calls: Sequence[tuple[int, int, list[int], int]]) -> None:
        metadata.copy_(torch.tensor([ks for _linear, _layer, ks, _out in calls], dtype=torch.int32))
        for _base, outs in linears:
            for out in outs:
                out.view(torch.int32).fill_(OUT_POISON_BITS)

    def execute(calls: Sequence[tuple[int, int, list[int], int]]) -> None:
        for index, (linear, _layer, _ks, out_index) in enumerate(calls):
            base, outs = linears[linear]
            returned = entry(
                base.a, base.b, base.a_scale, base.b_scale, metadata[index], base.c, hint, outs[out_index]
            )
            if returned is not None:
                raise CorrectnessFailure(f"{WGRAD_ENTRYPOINT} must return None")

    def check_inputs(where: str, expected: Mapping[str, tuple[int, ...]]) -> None:
        problems = meta_arena.violated() + [problem for base, _outs in linears for problem in base.arena.violated()]
        if problems:
            raise MemorySafetyFailure(
                f"out-of-bounds write detected ({where}): {problems[:4]}",
                detail={"kind": "guard_band", "where": where, "regions": problems[:8]},
            )
        changed = [name for name, tensor in inputs().items() if fingerprint(tensor) != expected[name]]
        if changed:
            raise MemorySafetyFailure(
                f"candidate modified its inputs ({where}): {changed[:8]}",
                detail={"kind": "input_modified", "where": where, "inputs": changed[:8]},
            )

    def verify(calls: Sequence[tuple[int, int, list[int], int]]) -> tuple[int, float]:
        last: dict[tuple[int, int], int] = {}
        for index, (linear, _layer, _ks, out_index) in enumerate(calls):
            last[(linear, out_index)] = index
        worst = 0.0
        for index in sorted(last.values()):
            linear, _layer, ks, out_index = calls[index]
            base, outs = linears[linear]
            view = Invocation(
                g=base.g, m=base.m, n=base.n, ks=list(ks), hint=hint, arena=base.arena, a=base.a, b=base.b,
                a_scale=base.a_scale, b_scale=base.b_scale, metadata=metadata[index], c=base.c,
                out=outs[out_index],
            )
            try:
                checked = check_output(view, CorrectnessFailure, source=False)
            except CorrectnessFailure as exc:
                raise CorrectnessFailure(
                    f"sequence call {index}: {exc}", detail={"call": index, **dict(exc.detail)}
                ) from exc
            worst = max(worst, checked.max_exact_error)
        return len(last), worst

    _cuda_synchronize()
    expected = {name: fingerprint(tensor) for name, tensor in inputs().items()}
    warm_calls = sequence_schedule(case, derive_seed(seed, "sequence-warmup", 0))
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
    checked_calls: list[int] = []
    for ordinal in range(rounds):
        calls = sequence_schedule(case, derive_seed(seed, "sequence-round", ordinal))
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
        event_seconds = finite_positive(float(_event_elapsed(start_event, end_event)) / 1000.0, "CUDA event time")
        seconds = max(host_wall, event_seconds)
        sentinel.check("sequence round")
        check_inputs("sequence round", expected)
        count, worst = verify(calls)
        max_exact_error = max(max_exact_error, worst)
        flops = _fsum(2.0 * sum(ks) * shapes[linear][0] * shapes[linear][1] for linear, _layer, ks, _out in calls)
        round_us.append(seconds * 1.0e6)
        round_host_us.append(host_wall * 1.0e6)
        round_event_us.append(event_seconds * 1.0e6)
        round_flops.append(flops)
        round_tflops.append(finite_positive(flops / seconds / 1.0e12, "round TFLOPS"))
        checked_calls.append(count)
    del linears, metadata, meta_arena
    torch.cuda.empty_cache()

    case_tflops = finite_positive(_median(round_tflops), "case median TFLOPS")
    _check_bound(case, case_tflops, bound_tflops)
    return {
        "case_id": str(case["case_id"]),
        "family": "sequence",
        "model": str(case["model"]),
        "entrypoint": WGRAD_ENTRYPOINT,
        "g": g,
        "m": shapes[0][0],
        "n": shapes[0][1],
        "m2": shapes[1][0],
        "n2": shapes[1][1],
        "total_k": total_k,
        "dist": str(case["dist"]),
        "expected_k_hint": hint,
        "calls_per_round": calls_per_round,
        "round_us": round_us,
        "round_host_us": round_host_us,
        "round_event_us": round_event_us,
        "round_flops": round_flops,
        "round_tflops": round_tflops,
        "checked_calls": checked_calls,
        "median_us": finite_positive(_median(round_us), "case median us"),
        "tflops": case_tflops,
        "max_element_ratio": 0.0,
        "max_exact_error": max_exact_error,
        "max_source_error": 0.0,
    }


def run_scored_case(
    entrypoints: Mapping[str, Callable[..., Any]],
    raw_case: Mapping[str, Any],
    *,
    trials: int,
    warmups: int,
    rounds: int = 1,
    sentinel: HarnessSentinel,
    bound_tflops: float | None,
) -> dict[str, Any]:
    if raw_case.get("family") == "sequence":
        return run_sequence_case(entrypoints, raw_case, rounds=rounds, sentinel=sentinel, bound_tflops=bound_tflops)
    return run_single_case(
        entrypoints, raw_case, trials=trials, warmups=warmups, sentinel=sentinel, bound_tflops=bound_tflops
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
    prepared = prepare_invocation(
        int(case["g"]), int(case["m"]), int(case["n"]), case["ks"], int(case["hint"]),
        int(case["seed"]), references=True, guards=guards, pristine=True,
    )
    max_element_ratio = 0.0
    max_exact_error = 0.0
    max_source_error = 0.0
    for attempt in range(2):
        if attempt:
            poison_output(prepared)
        invoke(entrypoints, prepared)
        _cuda_synchronize()
        sentinel.check(f"edge call {attempt + 1}")
        check_memory(prepared, f"edge call {attempt + 1}")
        checked = check_output(prepared, EdgeCorrectnessFailure)
        max_element_ratio = max(max_element_ratio, checked.max_element_ratio)
        max_exact_error = max(max_exact_error, checked.max_exact_error)
        max_source_error = max(max_source_error, checked.max_source_error)
    del prepared
    return {
        "case_id": str(case["case_id"]),
        "category": str(case["category"]),
        "ok": True,
        "max_element_ratio": max_element_ratio,
        "max_exact_error": max_exact_error,
        "max_source_error": max_source_error,
    }


def aggregate_case_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Within each family, the geometric mean per model, then across models, so
    every model counts equally regardless of its case count; the primary score
    is the geometric mean of the family scores present (all four on the Judge)."""
    if not results:
        raise ValueError("cannot aggregate an empty case list")
    values: dict[str, dict[str, list[float]]] = {}
    for result in results:
        value = finite_positive(result.get("tflops"), "case TFLOPS")
        family = str(result.get("family"))
        model = str(result.get("model") or "")
        if family not in FAMILIES or not model:
            raise ValueError("case result has an invalid family or model")
        values.setdefault(family, {}).setdefault(model, []).append(value)
    family_tflops = {
        family: geometric_mean(geometric_mean(cases) for cases in values[family].values())
        for family in FAMILIES
        if family in values
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
            "decoy_ks": "redrawn",
            "sequence_layers": SEQUENCE_LAYERS,
            "sequence_out_pool": SEQUENCE_OUT_POOL,
            "check_chunk_rows": CHECK_CHUNK_ROWS,
            "guard_bytes": GUARD_BYTES,
            "element_ratio_limit": ELEMENT_RATIO_LIMIT,
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
        torch.cuda.empty_cache()
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
                f"case {case['case_id']} ({case['family']}) failed: {type(exc).__name__}: {exc}",
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
    """Plain allocations (no guard bands) for compute-sanitizer; one call per case.

    The Judge runs this pass before any native call, so an out-of-bounds kernel
    is stopped by memcheck before it reaches the hardware.  A wrong answer is
    reported against its own gate rather than as a memory-safety failure.
    """
    cases = [validate_edge_case(case) for case in edge_cases]
    sentinel = HarnessSentinel()
    rows: list[dict[str, Any]] = []
    payload: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "candidate_sha256": None, "cases": rows}
    try:
        _module, entrypoints, source_sha256 = load_candidate(Path(candidate_path))
    except Exception as exc:
        return payload | {"status": "failed", "failed_gate": "correctness_ok",
                          "private_error": f"candidate import failed: {type(exc).__name__}: {exc}"}
    payload["candidate_sha256"] = source_sha256
    for case in cases:
        try:
            prepared = prepare_invocation(
                int(case["g"]), int(case["m"]), int(case["n"]), case["ks"], int(case["hint"]),
                int(case["seed"]), references=True, guards=False, pristine=True,
            )
            invoke(entrypoints, prepared)
            _cuda_synchronize()
            sentinel.check("sanitizer call")
            check_memory(prepared, "sanitizer call")
            checked = check_output(prepared, EdgeCorrectnessFailure)
        except Exception as exc:  # noqa: BLE001 - classified below
            gate = classify_exception(exc)
            if gate == "correctness_ok":
                gate = "edge_correctness_ok"
            return payload | {
                "status": "failed",
                "failed_gate": gate,
                "private_error": f"sanitizer case {case['case_id']}: {type(exc).__name__}: {str(exc)[:400]}",
            }
        rows.append(
            {
                "case_id": str(case["case_id"]),
                "max_element_ratio": checked.max_element_ratio,
                "max_exact_error": checked.max_exact_error,
            }
        )
        del prepared
    return payload | {"status": "passed"}
