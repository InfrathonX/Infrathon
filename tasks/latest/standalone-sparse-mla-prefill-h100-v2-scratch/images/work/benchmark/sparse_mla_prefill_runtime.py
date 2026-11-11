#!/usr/bin/env python3
"""Public test runtime for the standalone BF16 sparse-MLA prefill task.

The public controller provides the case manifests (``public_cases.json`` and
``public_edge_cases.json``).  Public cases cover the same kinds of workload as the
hidden ones: single calls with causal indices and ``topk_length``, calls at the
head count of one tensor-parallel rank, shards with short ``topk_length``, chunks
over several requests, and streams of consecutive calls with different shapes.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import torch

# --------------------------------------------------------------------------
# Local references, bound before any candidate module is loaded.
# --------------------------------------------------------------------------
_perf_counter = time.perf_counter
_Event = torch.cuda.Event
_event_record = torch.cuda.Event.record
_event_elapsed = torch.cuda.Event.elapsed_time
_cuda_synchronize = torch.cuda.synchronize
_current_stream = torch.cuda.current_stream
_torch_equal = torch.equal
_logsumexp = torch.logsumexp
_median = statistics.median
_fsum = math.fsum

PREFILL_ENTRYPOINT = "sparse_mla_prefill"
ENTRYPOINTS = (PREFILL_ENTRYPOINT,)
SCHEMA_VERSION = 3
DEVICE = "cuda"

# --------------------------------------------------------------------------
# Geometry and the public cases.
# --------------------------------------------------------------------------
# Four deployed geometries (HF config.json of the four models).
FAMILIES = ("v32", "glm5", "v4_flash", "v4_pro")
FAMILY_GEOMETRY: dict[str, tuple[int, int]] = {
    "v32": (576, 128),
    "glm5": (576, 64),
    "v4_flash": (512, 64),
    "v4_pro": (512, 128),
}
MIN_H_Q, H_Q_MULTIPLE = 16, 16
# Public groups.  ``basic`` holds the original public shapes with inputs A (every
# index valid and distinct) or B (invalid indices, topk_length, finite sinks).
GROUP_TIER: dict[str, str] = {
    "basic": "evidence",
    "trace": "evidence",
    "replay": "evidence",
    "multi": "evidence",
    "pattern": "evidence",
    "sweep": "sweep",
    "heads": "sweep",
    "length": "sweep",
}
GROUPS = tuple(GROUP_TIER)
TIERS = ("evidence", "sweep")
STREAM_GROUPS = ("trace", "replay")
SCORED_SINK_MODES = ("none", "finite")
LENGTH_DIVISORS = (2, 4, 8)
# Index patterns of the ``pattern`` group: where a learned indexer's top-k keys sit.
# window: most keys in the last few thousand positions; sink_window: the first keys,
# a local window and the rest anywhere; blocks: rows of one 64-row tile share a set
# of 64-key blocks; sorted: causal keys in ascending order.
INDEX_PATTERNS = ("window", "sink_window", "blocks", "sorted")
PATTERN_WINDOW_FACTOR = 2      # the local window holds this many times topk keys
PATTERN_SINK_KEYS = 64
PATTERN_BLOCK_KEYS = 64
PATTERN_ROW_TILE = 64
MAX_REQUESTS = 128
MAX_STREAM_CALLS = 256
# A stream starts with STREAM_WARM_CALLS untimed calls at shapes next to, never
# equal to, its own (half keep each size's residue mod 16, half change it); every
# later call is timed, compilation included.  It stops once its timed seconds
# pass STREAM_TIME_LIMIT_SECONDS and scores the work it finished.
STREAM_WARM_CALLS = 16
STREAM_CHECK_EVERY = 8
STREAM_TIME_LIMIT_SECONDS = 8.0
# Slots past topk_length hold a stale, valid-looking index this often.
STALE_TAIL_FRACTION = 0.5
PUBLIC_SM_SCALE = 0.5
D_V = 512
H_KV = 1


# Correctness limits.
OUT_ABS_TOL = 8.0e-4
OUT_REL_TOL = 3.01 / 128
OUT_COS_TOL = 7.0e-6
LOGIT_ABS_TOL = 1.0e-6
LOGIT_REL_TOL = 2.01 / 65536
LOGIT_COS_TOL = 1.0e-7

# Support domain (the task contract).
MIN_TOPK, MAX_TOPK, TOPK_MULTIPLE = 128, 2048, 128
MAX_S_Q = 70_000
MAX_S_KV = 1 << 20
MAX_Q_ELEMENTS = 1 << 32          # s_q * h_q * d_qk
MAX_TOPK_POSITIONS = 1 << 25      # s_q * topk
# Natively run cases keep q and out at or below 3 GiB each.
NATIVE_MAX_TENSOR_BYTES = 3 << 30
LARGE_OFFSET_ELEMENTS = 1 << 31
MAX_SEED = (1 << 63) - 1
REFERENCE_CHUNK_BYTES = 1 << 30
COMPARE_STEP_ELEMENTS = 1 << 24

# Invalid-index candidates; ``None`` is materialised as ``s_kv``.
INVALID_INDEX_CANDIDATES: tuple[int | None, ...] = (
    -2147483648,
    -123456,
    -1,
    None,
    114514,
    1919810,
    2147480000,
    2147483647,
)

# Content switches of a generated invocation.
BIAS_MODES = ("default", "zero")
INDEX_MODES = ("default", "valid_pad", "sprinkle", "causal")
INVALID_ROW_MODES = ("none", "default", "sparse", "some", "all")
INVALID_ROW_THRESHOLD = {"default": -2.0, "sparse": -1.0, "some": -0.5}
SPRINKLE_FRACTION = 0.125
SINK_MODES = ("none", "finite", "default", "extreme")
LENGTH_MODES = ("none", "default", "edge", "half")
EDGE_LENGTH_BOUNDARY = (0, 1, 63, 64, 65, "topk-1", "topk")
INPUT_CONTENT: dict[str, dict[str, Any]] = {
    "A": {"bias": "default", "index_mode": "default", "invalid_rows": "none", "duplicates": False,
          "scramble": False, "sink_mode": "default", "length_mode": "none"},
    "B": {"bias": "default", "index_mode": "sprinkle", "invalid_rows": "none", "duplicates": False,
          "scramble": False, "sink_mode": "finite", "length_mode": "half"},
}

# Memory safety.
GUARD_MIN_BYTES = 1 << 20
GUARD_ROWS = 128
GUARD_ALIGN = 4096
GUARD_FILL_BYTE = 0xA5  # BF16 0xA5A5 ~ -2.9e-16 and FP32 0xA5A5A5A5 ~ -2.9e-16: finite
SOFT_INDEX = 0          # a valid key index for any s_kv >= 1
SOFT_LENGTH = 0         # a valid topk_length
L2_FLUSH_BYTES = 256 << 20
EDGE_CALLS = 2


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


def effective_flops(positions: int, h_q: int, d_qk: int) -> float:
    """Effective FLOPs: 2 * total_topk * h_q * (d_qk + d_v)."""
    for name, value in (("positions", positions), ("h_q", h_q), ("d_qk", d_qk)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeError(f"effective {name} must be a positive integer")
    return 2.0 * positions * h_q * (d_qk + D_V)


def effective_tflops(positions: int, h_q: int, d_qk: int, seconds: float) -> float:
    measured = finite_positive(seconds, "measured seconds")
    return finite_positive(effective_flops(positions, h_q, d_qk) / measured / 1.0e12, "TFLOPS")


def derive_seed(case_seed: int, role: str, trial: int) -> int:
    """Per-call seed: 63 bits of SHA-256(case seed, role, trial)."""
    digest = hashlib.sha256(f"smla-v2:{int(case_seed)}:{role}:{int(trial)}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & MAX_SEED


def length_boundary(topk: int) -> list[int]:
    values = []
    for item in EDGE_LENGTH_BOUNDARY:
        value = topk - 1 if item == "topk-1" else topk if item == "topk" else int(item)
        values.append(max(0, min(topk, value)))
    return values


# --------------------------------------------------------------------------
# Case validation.
# --------------------------------------------------------------------------
SCORED_KEYS = frozenset({
    "case_id", "group", "family", "h_q", "topk", "s_q", "s_kv", "sm_scale", "sink_mode", "seed",
})
SCORED_GROUP_KEYS: dict[str, frozenset[str]] = {
    "basic": frozenset({"inputs"}),
    "sweep": frozenset(),
    "heads": frozenset(),
    "length": frozenset({"length_divisor", "ragged"}),
    "multi": frozenset({"request_lengths"}),
    "trace": frozenset({"calls"}),
    "replay": frozenset({"calls"}),
    "pattern": frozenset({"index_pattern"}),
}
SCORED_DERIVED_KEYS = frozenset({"tier", "d_qk", "scored", "stream"})
EDGE_KEYS = frozenset({
    "case_id", "category", "family", "s_q", "s_kv", "topk", "sm_scale", "bias",
    "index_mode", "invalid_rows", "duplicates", "scramble", "sink_mode",
    "length_mode", "seed", "sanitize", "native",
})
EDGE_OPTIONAL_KEYS = frozenset({"h_q"})
EDGE_DERIVED_KEYS = frozenset({"d_qk", "scored"})


def _int_field(raw: Mapping[str, Any], name: str, *, minimum: int = 1, maximum: int | None = None) -> int:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def tensor_bytes(s_q: int, h_q: int, d_qk: int) -> tuple[int, int]:
    """(q bytes, out bytes) of one invocation."""
    return 2 * s_q * h_q * d_qk, 2 * s_q * h_q * D_V


def validate_shape(family: Any, s_q: Any, s_kv: Any, topk: Any, h_q: Any = None) -> tuple[int, int]:
    """Check one (family, s_q, s_kv, topk[, h_q]) against the contract; return (d_qk, h_q).

    ``h_q`` defaults to the family's head count; a given one must be a multiple of
    16 no larger than it.
    """
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")
    d_qk, family_heads = FAMILY_GEOMETRY[str(family)]
    if h_q is None:
        h_q = family_heads
    elif isinstance(h_q, bool) or not isinstance(h_q, int):
        raise ValueError("h_q must be an integer")
    elif h_q % H_Q_MULTIPLE or not MIN_H_Q <= h_q <= family_heads:
        raise ValueError(f"h_q must be a multiple of {H_Q_MULTIPLE} in [{MIN_H_Q}, {family_heads}]")
    raw = {"s_q": s_q, "s_kv": s_kv, "topk": topk}
    s_q = _int_field(raw, "s_q", minimum=1, maximum=MAX_S_Q)
    s_kv = _int_field(raw, "s_kv", minimum=1, maximum=MAX_S_KV)
    topk = _int_field(raw, "topk", minimum=MIN_TOPK, maximum=MAX_TOPK)
    if topk % TOPK_MULTIPLE:
        raise ValueError("topk must be a multiple of 128 in [128, 2048]")
    if s_q * h_q * d_qk > MAX_Q_ELEMENTS:
        raise ValueError("s_q * h_q * d_qk exceeds 2**32")
    if s_q * topk > MAX_TOPK_POSITIONS:
        raise ValueError("s_q * topk exceeds 2**25")
    return d_qk, h_q


def validate_scored_case(raw: Mapping[str, Any]) -> dict[str, Any]:
    """One public scored case: a group, a geometry, the sizes and the content kind."""
    if not isinstance(raw, Mapping):
        raise ValueError("scored case must be an object")
    group = raw.get("group")
    if group not in GROUPS:
        raise ValueError(f"group must be one of {GROUPS}")
    wanted = SCORED_KEYS | SCORED_GROUP_KEYS[str(group)]
    missing = sorted(wanted - set(raw))
    if missing:
        raise ValueError(f"scored case is missing fields: {missing}")
    extra = sorted(set(raw) - wanted - SCORED_DERIVED_KEYS)
    if extra:
        raise ValueError(f"scored case has unexpected fields: {extra}")
    case = {key: raw[key] for key in wanted}
    if not isinstance(case["case_id"], str) or not case["case_id"]:
        raise ValueError("case_id must be a nonempty string")
    s_q = _int_field(case, "s_q", minimum=1, maximum=MAX_S_Q)
    s_kv = _int_field(case, "s_kv", minimum=1, maximum=MAX_S_KV)
    d_qk, h_q = validate_shape(case["family"], s_q, s_kv, case["topk"], case["h_q"])
    scale = case["sm_scale"]
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(float(scale)) or scale <= 0:
        raise ValueError("sm_scale must be finite and positive")
    case["sm_scale"] = float(scale)
    _int_field(case, "seed", minimum=0, maximum=MAX_SEED)
    if group == "basic":
        if case["inputs"] not in INPUT_CONTENT:
            raise ValueError(f"inputs must be one of {tuple(INPUT_CONTENT)}")
    elif case["sink_mode"] not in SCORED_SINK_MODES:
        raise ValueError(f"sink_mode must be one of {SCORED_SINK_MODES}")
    if group in ("sweep", "heads", "pattern") and s_q > s_kv:
        raise ValueError("a causal single call needs s_q <= s_kv")
    if group == "length" and (case["length_divisor"] not in LENGTH_DIVISORS or not isinstance(case["ragged"], bool)):
        raise ValueError(f"length_divisor must be one of {LENGTH_DIVISORS} and ragged a boolean")
    if group == "pattern" and case["index_pattern"] not in INDEX_PATTERNS:
        raise ValueError(f"index_pattern must be one of {INDEX_PATTERNS}")
    if group == "multi":
        lengths = case["request_lengths"]
        if (
            not isinstance(lengths, list) or not 2 <= len(lengths) <= MAX_REQUESTS
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in lengths)
            or sum(lengths) != s_kv or len(lengths) > s_q
        ):
            raise ValueError(f"request_lengths must list 2 to {MAX_REQUESTS} positive integers summing to s_kv")
    stream = group in STREAM_GROUPS
    if stream:
        calls = case["calls"]
        if not isinstance(calls, list) or not 1 <= len(calls) <= MAX_STREAM_CALLS:
            raise ValueError(f"calls must list 1 to {MAX_STREAM_CALLS} calls")
        parsed = []
        for call in calls:
            if (
                not isinstance(call, list) or len(call) != 2
                or any(isinstance(item, bool) or not isinstance(item, int) for item in call)
                or not 1 <= call[0] <= call[1]
            ):
                raise ValueError("each call is [s_q, s_kv] with 1 <= s_q <= s_kv")
            validate_shape(case["family"], call[0], call[1], case["topk"], h_q)
            parsed.append([int(call[0]), int(call[1])])
        if max(item[0] for item in parsed) != s_q or max(item[1] for item in parsed) != s_kv:
            raise ValueError("a stream case's s_q and s_kv are the largest of its calls")
        case["calls"] = parsed
    q_bytes, out_bytes = tensor_bytes(s_q, h_q, d_qk)
    if max(q_bytes, out_bytes) > NATIVE_MAX_TENSOR_BYTES:
        raise ValueError("a scored case keeps q and out at or below the native tensor cap")
    case.update(tier=GROUP_TIER[str(group)], d_qk=d_qk, h_q=h_q, scored=True, stream=stream)
    return case


def validate_edge_case(raw: Mapping[str, Any]) -> dict[str, Any]:
    missing = sorted(EDGE_KEYS - set(raw))
    if missing:
        raise ValueError(f"edge case is missing fields: {missing}")
    extra = sorted(set(raw) - EDGE_KEYS - EDGE_OPTIONAL_KEYS - EDGE_DERIVED_KEYS)
    if extra:
        raise ValueError(f"edge case has unexpected fields: {extra}")
    case = {key: raw[key] for key in EDGE_KEYS}
    for name in ("case_id", "category"):
        if not isinstance(case[name], str) or not case[name]:
            raise ValueError(f"{name} must be a nonempty string")
    d_qk, h_q = validate_shape(case["family"], case["s_q"], case["s_kv"], case["topk"], raw.get("h_q"))
    scale = case["sm_scale"]
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise ValueError("sm_scale must be a number")
    if not math.isfinite(float(scale)) or float(scale) <= 0.0:
        raise ValueError("sm_scale must be finite and positive")
    case["sm_scale"] = float(scale)
    for name, allowed in (
        ("bias", BIAS_MODES),
        ("index_mode", INDEX_MODES),
        ("invalid_rows", INVALID_ROW_MODES),
        ("sink_mode", SINK_MODES),
        ("length_mode", LENGTH_MODES),
    ):
        if case[name] not in allowed:
            raise ValueError(f"{name} must be one of {allowed}")
    for flag in ("duplicates", "scramble", "sanitize", "native"):
        if not isinstance(case[flag], bool):
            raise ValueError(f"{flag} must be a boolean")
    _int_field(case, "seed", minimum=0, maximum=MAX_SEED)
    q_bytes, out_bytes = tensor_bytes(int(case["s_q"]), h_q, d_qk)
    if case["native"]:
        if max(q_bytes, out_bytes) > NATIVE_MAX_TENSOR_BYTES:
            raise ValueError("a natively run edge case keeps q and out at or below the native tensor cap")
    else:
        if not case["sanitize"]:
            raise ValueError("a case that never runs natively must be a sanitizer case")
        if int(case["s_q"]) * h_q * d_qk <= LARGE_OFFSET_ELEMENTS:
            raise ValueError("a sanitizer-only case must hold more than 2**31 q elements")
    case["d_qk"] = d_qk
    case["h_q"] = h_q
    case["scored"] = False
    return case


def sanitize_order(cases: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The sanitizer subset in run order: sanitizer-only cases first (they are
    the ones that must not reach the hardware natively), then the others in
    manifest order."""
    flagged = [case for case in cases if case.get("sanitize") is True]
    return [case for case in flagged if case.get("native") is not True] + [
        case for case in flagged if case.get("native") is True
    ]


# --------------------------------------------------------------------------
# Hardware.
# --------------------------------------------------------------------------
# Capability envelope: what the task contract actually needs, not the GPU model
# it was qualified on.  The GPU name and UUID are recorded as provenance only;
# they never gate admission.
MIN_COMPUTE_CAPABILITY = (8, 0)
MIN_DEVICE_MEMORY_BYTES = 32 * 1024**3


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


def use_exact_fp32() -> None:
    """Freeze the reference at true FP32; TF32 must never move it."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_candidate(path: Path) -> tuple[ModuleType, Callable[..., Any], str]:
    candidate = path.resolve()
    if not candidate.is_file() or candidate.is_symlink():
        raise RuntimeError(f"candidate is not a regular file: {candidate}")
    source_sha256 = _sha256(candidate)
    module_name = f"sparse_mla_prefill_candidate_{source_sha256[:24]}"
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
    value = getattr(module, PREFILL_ENTRYPOINT, None)
    if not callable(value):
        raise RuntimeError(f"candidate must define callable {PREFILL_ENTRYPOINT}")
    return module, value, source_sha256


# --------------------------------------------------------------------------
# Guarded allocation with soft-landing fills.
# --------------------------------------------------------------------------
def guard_bytes(row_bytes: int) -> int:
    """max(1 MiB, 128 rows), rounded up to 4 KiB so every view stays aligned.

    A kernel that tiles s_q (or s_kv) by 64 or 128 rows without masking the
    tail overruns by less than 128 rows, so the overrun lands in the band.
    """
    wanted = max(GUARD_MIN_BYTES, GUARD_ROWS * int(row_bytes))
    return -(-wanted // GUARD_ALIGN) * GUARD_ALIGN


@dataclass
class _Region:
    name: str
    raw: Any
    guard: int
    nbytes: int
    dtype: Any
    fill: int  # a byte for floating tensors, an element value for int32 tensors


def _fill_guard(view: Any, dtype: Any, fill: int) -> None:
    if dtype == torch.int32:
        view.view(torch.int32).fill_(int(fill))
    else:
        view.fill_(int(fill))


def _guard_intact(view: Any, dtype: Any, fill: int) -> bool:
    if dtype == torch.int32:
        return bool(view.view(torch.int32).eq(int(fill)).all().item())
    return bool(view.eq(int(fill)).all().item())


class GuardedArena:
    """Every tensor lives in the middle of a buffer with filled guard bands.

    BF16 and FP32 tensors get 0xA5 bytes (finite in both dtypes).  The int32
    metadata gets a valid element value: the bands around ``indices`` hold
    the key index 0 and those around ``topk_length`` the length 0, so an
    out-of-bounds metadata read lands on real KV memory (a wrong answer)
    instead of a wild address.  Without guards (the compute-sanitizer mode)
    every tensor is its own allocation so memcheck sees exact bounds.
    """

    def __init__(self, guards: bool, device: Any = None) -> None:
        self.guards = guards
        self.device = DEVICE if device is None else device
        self.regions: list[_Region] = []

    def empty(
        self, name: str, shape: Sequence[int], dtype: Any, *, row_bytes: int,
        fill: int = GUARD_FILL_BYTE,
    ) -> Any:
        element = torch.empty((), dtype=dtype).element_size()
        nbytes = math.prod(int(item) for item in shape) * element
        if not self.guards:
            return torch.empty(tuple(shape), dtype=dtype, device=self.device)
        guard = guard_bytes(row_bytes)
        raw = torch.empty((guard + nbytes + guard,), dtype=torch.uint8, device=self.device)
        _fill_guard(raw[:guard], dtype, fill)
        _fill_guard(raw[guard + nbytes :], dtype, fill)
        view = raw[guard : guard + nbytes].view(dtype).view(tuple(shape))
        if not view.is_contiguous():  # pragma: no cover - defensive
            raise RuntimeError("guarded view is not contiguous")
        self.regions.append(_Region(name, raw, guard, nbytes, dtype, fill))
        return view

    def violated(self) -> list[str]:
        problems: list[str] = []
        for region in self.regions:
            if not _guard_intact(region.raw[: region.guard], region.dtype, region.fill):
                problems.append(f"{region.name}: write before the tensor")
            if not _guard_intact(region.raw[region.guard + region.nbytes :], region.dtype, region.fill):
                problems.append(f"{region.name}: write past the tensor")
        return problems


# --------------------------------------------------------------------------
# Workload generation.
# --------------------------------------------------------------------------
def scored_spec(case: Mapping[str, Any]) -> dict[str, Any]:
    """A basic cell's trial is its shape with its input kind."""
    return {
        "case_id": str(case["case_id"]), "family": str(case["family"]), "h_q": int(case["h_q"]),
        "s_q": int(case["s_q"]), "s_kv": int(case["s_kv"]), "topk": int(case["topk"]),
        "sm_scale": float(case["sm_scale"]), **INPUT_CONTENT[str(case["inputs"])],
    }


def _paddings(index_mode: str, s_kv: int) -> list[int]:
    if index_mode == "valid_pad":
        return [0]
    return [s_kv if value is None else int(value) for value in INVALID_INDEX_CANDIDATES]


def randperm_batch(
    batch: int,
    perm_range: int,
    perm_size: int,
    paddings: Sequence[int],
    generator: Any,
    device: Any,
) -> Any:
    """Row-chunked random permutations.

    Each row holds distinct values in ``[0, perm_range)`` in random order;
    slots that cannot be filled (``perm_range < perm_size``) take one of
    ``paddings``.
    """
    span = max(int(perm_range), int(perm_size))
    result = torch.empty((batch, perm_size), dtype=torch.int32, device=device)
    rows_per_chunk = max(1, (64 << 20) // (4 * span))
    for start in range(0, batch, rows_per_chunk):
        stop = min(batch, start + rows_per_chunk)
        scores = torch.rand((stop - start, span), dtype=torch.float32, device=device, generator=generator)
        if perm_range < span:
            scores[:, perm_range:] = float("-inf")
        result[start:stop] = scores.topk(perm_size, dim=-1, sorted=True).indices.to(torch.int32)
        del scores
    overflow = result >= perm_range
    if bool(overflow.any().item()):
        if len(paddings) == 1:
            result[overflow] = int(paddings[0])
        else:
            table = torch.tensor(list(paddings), dtype=torch.int32, device=device)
            picks = torch.randint(
                0, len(paddings), (int(overflow.sum().item()),), dtype=torch.int64,
                device=device, generator=generator,
            )
            result[overflow] = table.index_select(0, picks)
    return result


def generate_into(
    spec: Mapping[str, Any],
    seed: int,
    arena: GuardedArena,
) -> dict[str, Any]:
    """Generate one invocation's inputs from a content spec into ``arena``."""
    device = arena.device
    family = str(spec["family"])
    d_qk, h_q = spec_geometry(spec)
    s_q, s_kv, topk = int(spec["s_q"]), int(spec["s_kv"]), int(spec["topk"])
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) & MAX_SEED)
    host = torch.Generator()
    host.manual_seed(derive_seed(seed, "host", 0))

    # q and kv: randn/10 + a per-tensor bias in [-0.05, 0.05), clamped to +-10.
    q_bias = (float(torch.rand((), generator=host).item()) - 0.5) / 10.0
    kv_bias = (float(torch.rand((), generator=host).item()) - 0.5) / 10.0
    if spec["bias"] == "zero":
        q_bias = kv_bias = 0.0
    q = arena.empty("q", (s_q, h_q, d_qk), torch.bfloat16, row_bytes=2 * h_q * d_qk)
    q.normal_(generator=generator).div_(10.0).add_(q_bias).clamp_(-10.0, 10.0)
    kv = arena.empty("kv", (s_kv, H_KV, d_qk), torch.bfloat16, row_bytes=2 * H_KV * d_qk)
    kv.normal_(generator=generator).div_(10.0).add_(kv_bias).clamp_(-10.0, 10.0)

    index_mode = str(spec["index_mode"])
    causal_lengths = None
    if index_mode == "causal":
        indices, causal_lengths = causal_indices(s_q, topk, s_kv, max(0, s_kv - s_q), generator, device)
    else:
        paddings = _paddings(index_mode, s_kv)
        indices = randperm_batch(s_q, s_kv, topk, paddings, generator, device)
    invalid_pool = _paddings("default", s_kv)
    if index_mode == "sprinkle":
        # Invalid indices interleaved with valid ones (candidate
        # values): a kernel must mask per slot, not stop at the first hole.
        hit = torch.rand((s_q, topk), device=device, generator=generator) < SPRINKLE_FRACTION
        table = torch.tensor(invalid_pool, dtype=torch.int32, device=device)
        picks = torch.randint(
            0, len(invalid_pool), (s_q, topk), dtype=torch.int64, device=device, generator=generator
        )
        indices = torch.where(hit, table[picks], indices)
        del hit, picks
    if bool(spec["duplicates"]) and topk > 1:
        # Valid keys may repeat inside one row; a kernel that deduplicates its
        # gather is wrong.
        picks = torch.randint(0, topk, (s_q, topk), dtype=torch.int64, device=device, generator=generator)
        take = torch.rand((s_q, topk), device=device, generator=generator) < 0.25
        indices = torch.where(take, indices.gather(1, picks), indices)
        del picks, take
    if bool(spec["scramble"]) and topk > 1:
        # By default every valid index comes before the padding; scrambling
        # interleaves them.
        order = torch.rand((s_q, topk), device=device, generator=generator).argsort(dim=-1)
        indices = indices.gather(1, order)
        del order
    invalid_rows = str(spec["invalid_rows"])
    if invalid_rows != "none":
        sentinel = int(invalid_pool[int(torch.randint(0, len(invalid_pool), (1,), generator=host).item())])
        if invalid_rows == "all":
            rows = torch.ones(s_q, dtype=torch.bool, device=device)
        else:
            rows = torch.randn(s_q, device=device, generator=generator) < INVALID_ROW_THRESHOLD[invalid_rows]
            rows[0] = True
        indices = torch.where(rows.unsqueeze(1), torch.full_like(indices, sentinel), indices)
        del rows
    index_view = arena.empty(
        "indices", (s_q, H_KV, topk), torch.int32, row_bytes=4 * H_KV * topk, fill=SOFT_INDEX
    )
    index_view.copy_(indices.view(s_q, H_KV, topk))
    del indices

    attn_sink = None
    sink_mode = str(spec["sink_mode"])
    if sink_mode != "none":
        values = torch.randn((h_q,), dtype=torch.float32, device=device, generator=generator)
        if sink_mode in ("default", "extreme"):
            selector = torch.randn((h_q,), dtype=torch.float32, device=device, generator=generator)
            values[selector < -0.5] = float("-inf")
            values[selector > 0.5] = float("+inf")
            if sink_mode == "extreme":
                for position, value in enumerate((float("-inf"), float("+inf"), 100.0, -100.0)):
                    if position < h_q:
                        values[position] = value
        attn_sink = arena.empty("attn_sink", (h_q,), torch.float32, row_bytes=4 * h_q)
        attn_sink.copy_(values)

    topk_length = None
    length_mode = str(spec["length_mode"])
    if causal_lengths is not None:
        # Causal rows always carry their valid counts, whatever the length mode.
        topk_length = arena.empty("topk_length", (s_q,), torch.int32, row_bytes=4, fill=SOFT_LENGTH)
        topk_length.copy_(causal_lengths)
    elif length_mode != "none":
        if length_mode == "half":
            values = torch.randint(
                -(-topk // 2), topk + 1, (s_q,), dtype=torch.int32, device=device, generator=generator
            )
        else:
            values = torch.randint(
                0, max(topk + 1, 64), (s_q,), dtype=torch.int32, device=device, generator=generator
            ).clamp_max_(topk)
            if length_mode == "edge":
                for position, value in enumerate(length_boundary(topk)):
                    if position < s_q:
                        values[position] = value
        topk_length = arena.empty("topk_length", (s_q,), torch.int32, row_bytes=4, fill=SOFT_LENGTH)
        topk_length.copy_(values)
    return {
        "q": q, "kv": kv, "indices": index_view, "attn_sink": attn_sink,
        "topk_length": topk_length,
    }


@dataclass
class Invocation:
    case_id: str
    family: str
    s_q: int
    s_kv: int
    topk: int
    h_q: int
    d_qk: int
    sm_scale: float
    arena: GuardedArena
    q: Any
    kv: Any
    indices: Any
    attn_sink: Any
    topk_length: Any
    out: Any
    max_logits: Any
    lse: Any
    positions: int
    pristine: dict[str, Any] = field(default_factory=dict)

    @property
    def inputs(self) -> dict[str, Any]:
        named = {"q": self.q, "kv": self.kv, "indices": self.indices}
        if self.attn_sink is not None:
            named["attn_sink"] = self.attn_sink
        if self.topk_length is not None:
            named["topk_length"] = self.topk_length
        return named

    @property
    def outputs(self) -> tuple[Any, Any, Any]:
        return self.out, self.max_logits, self.lse


def prepare_invocation(
    spec: Mapping[str, Any],
    seed: int,
    *,
    guards: bool = True,
    pristine: bool = True,
    device: Any = None,
) -> Invocation:
    """Generate one invocation from a validated spec (edge case or scored trial)."""
    arena = GuardedArena(guards, device)
    tensors = generate_into(spec, seed, arena)
    family = str(spec["family"])
    d_qk, h_q = spec_geometry(spec)
    s_q = int(spec["s_q"])
    out = arena.empty("out", (s_q, h_q, D_V), torch.bfloat16, row_bytes=2 * h_q * D_V)
    max_logits = arena.empty("max_logits", (s_q, h_q), torch.float32, row_bytes=4 * h_q)
    lse = arena.empty("lse", (s_q, h_q), torch.float32, row_bytes=4 * h_q)
    topk = int(spec["topk"])
    topk_length = tensors["topk_length"]
    positions = s_q * topk if topk_length is None else int(topk_length.sum().item())
    invocation = Invocation(
        case_id=str(spec["case_id"]), family=family, s_q=s_q, s_kv=int(spec["s_kv"]),
        topk=topk, h_q=h_q, d_qk=d_qk, sm_scale=float(spec["sm_scale"]), arena=arena,
        q=tensors["q"], kv=tensors["kv"], indices=tensors["indices"],
        attn_sink=tensors["attn_sink"], topk_length=topk_length, out=out,
        max_logits=max_logits, lse=lse, positions=positions,
    )
    poison_outputs(invocation)
    if pristine:
        invocation.pristine = {name: tensor.clone() for name, tensor in invocation.inputs.items()}
    return invocation


def invoke(entrypoint: Callable[..., Any], prepared: Invocation) -> None:
    returned = entrypoint(
        prepared.q,
        prepared.kv,
        prepared.indices,
        prepared.sm_scale,
        prepared.attn_sink,
        prepared.topk_length,
        prepared.out,
        prepared.max_logits,
        prepared.lse,
    )
    if returned is not None:
        raise CorrectnessFailure(f"{PREFILL_ENTRYPOINT} must return None")


def spec_geometry(spec: Mapping[str, Any]) -> tuple[int, int]:
    """(d_qk, h_q): the family's d_qk, and the spec's head count when it names one."""
    d_qk, family_heads = FAMILY_GEOMETRY[str(spec["family"])]
    h_q = spec.get("h_q")
    return d_qk, family_heads if h_q is None else int(h_q)


# --------------------------------------------------------------------------
# Serving inputs for the public scored suite (everything except basic).
# --------------------------------------------------------------------------
def stale_tail(indices: Any, lengths: Any, bound: Any, generator: Any) -> Any:
    """Slots past topk_length keep a stale but valid-looking index (a reused buffer)
    STALE_TAIL_FRACTION of the time; the others keep what they hold (-1)."""
    slots = torch.arange(indices.shape[1], device=indices.device).unsqueeze(0)
    tail = slots >= lengths.to(torch.int64).unsqueeze(1)
    stale = tail & (torch.rand(indices.shape, device=indices.device, generator=generator) < STALE_TAIL_FRACTION)
    values = (
        torch.rand(indices.shape, device=indices.device, generator=generator) * bound.to(torch.float32)
    ).floor().to(torch.int32)
    return torch.where(stale, values, indices)


def causal_indices(
    s_q: int, topk: int, s_kv: int, base: int, generator: Any, device: Any,
) -> tuple[Any, Any]:
    """Row i sits at position base + i and selects keys in [0, min(base + i, s_kv - 1)].

    A row with fewer allowed keys than topk keeps them, distinct and in random
    order, in its first slots, and its topk_length is that count; longer rows hold
    topk keys and topk_length = topk.  A row far from the start (at least 4 * topk
    allowed keys) draws with replacement, so a valid key may repeat, which the
    contract allows; nearer rows draw distinct keys.  O(s_q * topk) work.
    """
    position = int(base) + torch.arange(s_q, device=device, dtype=torch.int64)
    allowed = (position + 1).clamp(max=int(s_kv))
    lengths = allowed.clamp(max=topk).to(torch.int32)
    indices = torch.full((s_q, topk), -1, dtype=torch.int32, device=device)
    wide = allowed >= 4 * topk
    if bool(wide.any().item()):
        rows = wide.nonzero().squeeze(1)
        limit = allowed.index_select(0, rows).unsqueeze(1)
        draws = torch.rand((int(rows.numel()), topk), device=device, generator=generator)
        picked = (draws * limit.to(torch.float32)).floor().to(torch.int64)
        indices[rows] = torch.minimum(picked, limit - 1).to(torch.int32)
    narrow = (~wide).nonzero().squeeze(1)
    span = 4 * topk
    columns = torch.arange(span, device=device).unsqueeze(0)
    slots = torch.arange(topk, device=device).unsqueeze(0)
    for start in range(0, int(narrow.numel()), 1024):
        rows = narrow[start:start + 1024]
        scores = torch.rand((int(rows.numel()), span), device=device, generator=generator)
        scores.masked_fill_(columns >= allowed.index_select(0, rows).unsqueeze(1), -1.0)
        picked = scores.topk(topk, dim=-1).indices.to(torch.int32)
        picked.masked_fill_(slots >= lengths.index_select(0, rows).unsqueeze(1), -1)
        indices[rows] = picked
        del scores, picked
    return stale_tail(indices, lengths, allowed.unsqueeze(1), generator), lengths


def shard_indices(
    s_q: int, topk: int, s_kv: int, divisor: int, generator: Any, device: Any, ragged: bool = False,
) -> tuple[Any, Any]:
    """One decode-context-parallel shard: about topk / divisor local keys packed at the
    front of each row (topk_length counts them), drawn from the whole shard.  Rows hold
    topk / divisor keys within 10 %, or, when ``ragged``, anywhere from 1 to twice that
    (a shard that owns an uneven share of each row's keys)."""
    target = max(1, topk // int(divisor))
    if ragged:
        upper = min(topk, 2 * target)
        lengths = torch.randint(1, upper + 1, (s_q,), device=device, generator=generator).to(torch.int32)
    else:
        jitter = max(1, target // 10)
        lengths = (
            target + torch.randint(-jitter, jitter + 1, (s_q,), device=device, generator=generator)
        ).clamp(1, topk).to(torch.int32)
    draws = torch.rand((s_q, topk), device=device, generator=generator)
    indices = (draws * float(s_kv)).floor().clamp(max=s_kv - 1).to(torch.int32)
    slots = torch.arange(topk, device=device).unsqueeze(0)
    indices.masked_fill_(slots >= lengths.to(torch.int64).unsqueeze(1), -1)
    bound = torch.full((s_q, 1), int(s_kv), dtype=torch.int64, device=device)
    return stale_tail(indices, lengths, bound, generator), lengths


def pattern_indices(
    s_q: int, topk: int, s_kv: int, base: int, pattern: str, generator: Any, device: Any,
) -> tuple[Any, Any]:
    """Causal indices with the locality of a learned indexer's top-k (INDEX_PATTERNS).

    Rows near the start (fewer than 4 * topk allowed keys) keep the distinct causal
    draw of ``causal_indices``; every other row holds topk keys at or before its
    position, which may repeat, as the contract allows.  ``sorted`` sorts the valid
    slots of every row ascending and leaves the stale tail as it is.
    """
    indices, lengths = causal_indices(s_q, topk, s_kv, base, generator, device)
    position = int(base) + torch.arange(s_q, device=device, dtype=torch.int64)
    allowed = (position + 1).clamp(max=int(s_kv))
    if pattern == "sorted":
        slots = torch.arange(topk, device=device).unsqueeze(0)
        valid = slots < lengths.to(torch.int64).unsqueeze(1)
        largest = torch.iinfo(torch.int32).max
        ordered = torch.where(valid, indices, torch.full_like(indices, largest)).sort(dim=1).values
        return torch.where(valid, ordered, indices), lengths
    rows = (allowed >= 4 * topk).nonzero().squeeze(1)
    count = int(rows.numel())
    if count == 0:
        return indices, lengths
    limit = allowed.index_select(0, rows).unsqueeze(1)
    anywhere = (torch.rand((count, topk), device=device, generator=generator) * limit.to(torch.float32)).floor()
    window = PATTERN_WINDOW_FACTOR * topk
    local = limit - 1 - (torch.rand((count, topk), device=device, generator=generator) * window).floor()
    choice = torch.rand((count, topk), device=device, generator=generator)
    if pattern == "window":
        picked = torch.where(choice < 0.75, local, anywhere)
    elif pattern == "sink_window":
        sink = (torch.rand((count, topk), device=device, generator=generator) * PATTERN_SINK_KEYS).floor()
        picked = torch.where(choice < 0.125, sink, torch.where(choice < 0.625, local, anywhere))
    elif pattern == "blocks":
        tiles = rows // PATTERN_ROW_TILE
        tile_count = (s_q + PATTERN_ROW_TILE - 1) // PATTERN_ROW_TILE
        per_tile = max(4, (2 * topk) // PATTERN_BLOCK_KEYS)
        # A tile's blocks end at or before its first row, so every row of the tile may read them.
        first = int(base) + torch.arange(tile_count, device=device, dtype=torch.int64) * PATTERN_ROW_TILE + 1
        first = first.clamp(max=int(s_kv)).unsqueeze(1)
        span = (first - PATTERN_BLOCK_KEYS).clamp(min=1).to(torch.float32)
        starts = (torch.rand((tile_count, per_tile), device=device, generator=generator) * span).floor()
        which = (torch.rand((count, topk), device=device, generator=generator) * per_tile).floor().to(torch.int64)
        offset = (torch.rand((count, topk), device=device, generator=generator) * PATTERN_BLOCK_KEYS).floor()
        picked = starts.index_select(0, tiles).gather(1, which) + offset
    else:
        raise ValueError(f"unknown index pattern {pattern!r}")
    indices[rows] = torch.minimum(picked.clamp(min=0).to(torch.int64), limit - 1).to(torch.int32)
    return indices, lengths


def multi_request_rows(s_q: int, request_lengths: Sequence[int]) -> list[int]:
    """Query rows per request: an even split, each request's rows its last tokens."""
    count = len(request_lengths)
    base, extra = divmod(int(s_q), count)
    rows = [base + (1 if number < extra else 0) for number in range(count)]
    return [max(1, min(int(length), share)) for length, share in zip(request_lengths, rows)]


def multi_request_indices(
    s_q: int, topk: int, request_lengths: Sequence[int], generator: Any, device: Any,
) -> tuple[Any, Any]:
    """A prefill chunk over several requests packed into one KV workspace.  Each row
    selects causally inside its own request only; returns rows for sum(rows)."""
    parts, lengths = [], []
    offset = 0
    for length, rows in zip(request_lengths, multi_request_rows(s_q, request_lengths)):
        index, valid = causal_indices(rows, topk, int(length), int(length) - rows, generator, device)
        parts.append(torch.where(index >= 0, index + offset, index))
        lengths.append(valid)
        offset += int(length)
    return torch.cat(parts), torch.cat(lengths)


def _fill_activation(tensor: Any, generator: Any, host: Any) -> None:
    bias = (float(torch.rand((), generator=host).item()) - 0.5) / 10.0
    tensor.normal_(generator=generator).div_(10.0).add_(bias).clamp_(-10.0, 10.0)


def scored_rows(case: Mapping[str, Any]) -> int:
    """Query rows of one call of a non-stream case (a multi chunk may hold fewer)."""
    if case["group"] == "multi":
        return sum(multi_request_rows(int(case["s_q"]), case["request_lengths"]))
    return int(case["s_q"])


def prepare_scored_invocation(
    case: Mapping[str, Any],
    s_q: int,
    s_kv: int,
    seed: int,
    *,
    kv_pool: Any = None,
    pristine: bool = True,
) -> Invocation:
    """One call of a validated scored case at (s_q, s_kv) on fresh tensors.

    Basic cells use inputs A or B.  Every other group uses causal,
    shard or multi-request indices with topk_length, a stale tail, the case's
    sm_scale and sink mode.  A stream passes its KV pool and
    each call reads its prefix.
    """
    if case["group"] == "basic":
        return prepare_invocation(scored_spec(case), seed, pristine=pristine)
    device = DEVICE
    d_qk, h_q, topk = int(case["d_qk"]), int(case["h_q"]), int(case["topk"])
    arena = GuardedArena(True, device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) & MAX_SEED)
    host = torch.Generator()
    host.manual_seed(derive_seed(seed, "host", 0))
    group = str(case["group"])
    if group == "multi":
        indices, lengths = multi_request_indices(s_q, topk, case["request_lengths"], generator, device)
    elif group == "length":
        indices, lengths = shard_indices(s_q, topk, s_kv, int(case["length_divisor"]), generator, device,
                                         bool(case["ragged"]))
    elif group == "pattern":
        indices, lengths = pattern_indices(s_q, topk, s_kv, s_kv - s_q, str(case["index_pattern"]),
                                           generator, device)
    else:
        indices, lengths = causal_indices(s_q, topk, s_kv, s_kv - s_q, generator, device)
    rows = int(indices.shape[0])
    q = arena.empty("q", (rows, h_q, d_qk), torch.bfloat16, row_bytes=2 * h_q * d_qk)
    _fill_activation(q, generator, host)
    if kv_pool is None:
        kv = arena.empty("kv", (s_kv, H_KV, d_qk), torch.bfloat16, row_bytes=2 * H_KV * d_qk)
        _fill_activation(kv, generator, host)
    else:
        kv = kv_pool[:s_kv]
    index_view = arena.empty("indices", (rows, H_KV, topk), torch.int32, row_bytes=4 * H_KV * topk,
                             fill=SOFT_INDEX)
    index_view.copy_(indices.view(rows, H_KV, topk))
    topk_length = arena.empty("topk_length", (rows,), torch.int32, row_bytes=4, fill=SOFT_LENGTH)
    topk_length.copy_(lengths)
    del indices, lengths
    attn_sink = None
    if case["sink_mode"] == "finite":
        attn_sink = arena.empty("attn_sink", (h_q,), torch.float32, row_bytes=4 * h_q)
        attn_sink.copy_(torch.randn((h_q,), dtype=torch.float32, device=device, generator=generator))
    out = arena.empty("out", (rows, h_q, D_V), torch.bfloat16, row_bytes=2 * h_q * D_V)
    max_logits = arena.empty("max_logits", (rows, h_q), torch.float32, row_bytes=4 * h_q)
    lse = arena.empty("lse", (rows, h_q), torch.float32, row_bytes=4 * h_q)
    invocation = Invocation(
        case_id=str(case["case_id"]), family=str(case["family"]), s_q=rows, s_kv=int(s_kv),
        topk=topk, h_q=h_q, d_qk=d_qk, sm_scale=float(case["sm_scale"]), arena=arena,
        q=q, kv=kv, indices=index_view, attn_sink=attn_sink, topk_length=topk_length,
        out=out, max_logits=max_logits, lse=lse, positions=int(topk_length.sum().item()),
    )
    poison_outputs(invocation)
    if pristine:
        invocation.pristine = {
            name: tensor.clone() for name, tensor in invocation.inputs.items()
            if not (name == "kv" and kv_pool is not None)
        }
    return invocation


def poison_outputs(prepared: Invocation) -> None:
    for tensor in prepared.outputs:
        tensor.fill_(float("nan"))


# --------------------------------------------------------------------------
# Reference and checks.
# --------------------------------------------------------------------------
def reference_chunk_rows(topk: int, d_qk: int, h_q: int) -> int:
    per_row = topk * (d_qk * 4 + h_q * 8) + h_q * (d_qk + D_V) * 4
    return max(1, min(1024, REFERENCE_CHUNK_BYTES // max(1, per_row)))


def reference_chunks(
    q: Any,
    kv: Any,
    indices: Any,
    sm_scale: float,
    attn_sink: Any,
    topk_length: Any,
) -> Iterator[tuple[int, int, Any, Any, Any]]:
    """Row chunks of the FP32 reference.

    Yields ``(start, stop, out, max_logits, lse)`` for rows ``[start, stop)``
    so even the s_q = 70000 corner never holds a full FP32 reference.
    """
    s_q, h_q, d_qk = (int(value) for value in q.shape)
    s_kv = int(kv.shape[0])
    topk = int(indices.shape[-1])
    slots = torch.arange(topk, device=q.device)
    kv_flat = kv.view(s_kv, d_qk)
    chunk = reference_chunk_rows(topk, d_qk, h_q)
    for start in range(0, s_q, chunk):
        stop = min(start + chunk, s_q)
        rows = stop - start
        picked = indices[start:stop].reshape(rows, topk).to(torch.int64)
        if topk_length is not None:
            expired = slots.unsqueeze(0) >= topk_length[start:stop].to(torch.int64).unsqueeze(1)
            picked = picked.masked_fill(expired, -1)
        invalid = (picked < 0) | (picked >= s_kv)
        gathered = kv_flat.index_select(0, picked.masked_fill(invalid, 0).reshape(-1)).reshape(
            rows, topk, d_qk
        ).float()
        logits = q[start:stop].float() @ gathered.transpose(1, 2)
        logits *= float(sm_scale)
        logits.masked_fill_(invalid.unsqueeze(1), float("-inf"))
        raw_lse = _logsumexp(logits, dim=-1)
        max_logits = logits.amax(dim=-1)
        if attn_sink is None:
            merged = raw_lse
        else:
            merged = _logsumexp(
                torch.stack([raw_lse, attn_sink.broadcast_to(raw_lse.shape)], dim=0), dim=0
            )
        merged = merged.masked_fill(merged == float("-inf"), float("+inf"))
        weights = torch.exp(logits - merged.unsqueeze(-1))
        del logits
        out = weights @ gathered[..., :D_V]
        lse = raw_lse.masked_fill(raw_lse == float("-inf"), float("+inf"))
        del gathered, weights, picked, invalid
        yield start, stop, out, max_logits, lse


def reference_outputs(
    q: Any, kv: Any, indices: Any, sm_scale: float, attn_sink: Any, topk_length: Any
) -> tuple[Any, Any, Any]:
    """The whole FP32 reference at once (tests and small cases only)."""
    parts = list(reference_chunks(q, kv, indices, sm_scale, attn_sink, topk_length))
    return (
        torch.cat([part[2] for part in parts]),
        torch.cat([part[3] for part in parts]),
        torch.cat([part[4] for part in parts]),
    )


@dataclass
class ComparisonResult:
    ok: bool
    cos_diff: float
    max_abs: float
    max_rel: float
    fail_elements: int
    nan_elements: int
    anomaly_ok: bool
    detail: str


class StreamingComparison:
    """Chunked tolerance comparison.

    Element-wise ``|a-r| < abs_tol or |a-r|/(|r|+1e-6) < rel_tol`` on finite
    elements, identical +inf / -inf / NaN positions, and a whole-tensor cosine
    difference ``1 - 2<a,r>/(|a|^2+|r|^2)`` accumulated across chunks (the
    combined-norm form short-circuits only when both tensors are zero).
    """

    def __init__(self, label: str, abs_tol: float, rel_tol: float, cos_tol: float) -> None:
        self.label = label
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.cos_tol = cos_tol
        self.dot_parts: list[float] = []
        self.norm_parts: list[float] = []
        self.max_abs = 0.0
        self.max_rel = 0.0
        self.fail_elements = 0
        self.nan_elements = 0
        self.total = 0
        self.anomaly_detail: list[str] = []

    def update(self, answer: Any, reference: Any) -> None:
        if tuple(answer.shape) != tuple(reference.shape):
            self.anomaly_detail.append(f"{self.label}: shape mismatch")
            return
        flat_answer = answer.reshape(-1)
        flat_reference = reference.reshape(-1)
        count = int(flat_answer.numel())
        self.total += count
        for start in range(0, count, COMPARE_STEP_ELEMENTS):
            stop = min(start + COMPARE_STEP_ELEMENTS, count)
            piece_answer = flat_answer[start:stop].float().clone()
            piece_reference = flat_reference[start:stop].float().clone()
            answer_nan = torch.isnan(piece_answer)
            self.nan_elements += int(answer_nan.sum().item())
            for value in (float("+inf"), float("-inf"), float("nan")):
                if value != value:
                    reference_mask = torch.isnan(piece_reference)
                    answer_mask = answer_nan
                else:
                    reference_mask = piece_reference == value
                    answer_mask = piece_answer == value
                if not bool(_torch_equal(reference_mask, answer_mask)):
                    if len(self.anomaly_detail) < 4:
                        self.anomaly_detail.append(
                            f"{self.label}: {value} positions differ (candidate "
                            f"{int(answer_mask.sum().item())}, reference {int(reference_mask.sum().item())})"
                        )
                piece_reference = piece_reference.masked_fill(reference_mask, 0.0)
                piece_answer = piece_answer.masked_fill(answer_mask, 0.0)
            wide_answer = piece_answer.double()
            wide_reference = piece_reference.double()
            self.dot_parts.append(float((wide_answer * wide_reference).sum().item()))
            self.norm_parts.append(
                float((wide_answer * wide_answer + wide_reference * wide_reference).sum().item())
            )
            raw_abs = (piece_answer - piece_reference).abs()
            raw_rel = raw_abs / (piece_reference.abs() + 1.0e-6)
            rel_err = raw_rel.masked_fill(raw_abs < self.abs_tol, 0.0)
            abs_err = raw_abs.masked_fill(raw_rel < self.rel_tol, 0.0)
            passed = (abs_err < self.abs_tol) | (rel_err < self.rel_tol)
            self.fail_elements += int((~passed).sum().item())
            if raw_abs.numel():
                self.max_abs = max(self.max_abs, float(raw_abs.max().item()))
                self.max_rel = max(self.max_rel, float(raw_rel.max().item()))
            del piece_answer, piece_reference, wide_answer, wide_reference
            del raw_abs, raw_rel, rel_err, abs_err, passed, answer_nan

    def result(self) -> ComparisonResult:
        dot = _fsum(self.dot_parts)
        norm = _fsum(self.norm_parts)
        cos_diff = 0.0 if norm <= 1.0e-12 else 1.0 - 2.0 * dot / norm
        common = (self.max_abs, self.max_rel, self.fail_elements, self.nan_elements)
        if self.nan_elements:
            return ComparisonResult(
                False, cos_diff, *common, False,
                f"{self.label}: {self.nan_elements}/{self.total} elements are unwritten or NaN",
            )
        if self.anomaly_detail:
            return ComparisonResult(False, cos_diff, *common, False, "; ".join(self.anomaly_detail))
        if not math.isfinite(cos_diff):
            return ComparisonResult(False, cos_diff, *common, True, f"{self.label}: cosine diff is not finite")
        if self.fail_elements:
            return ComparisonResult(
                False, cos_diff, *common, True,
                f"{self.label}: {self.fail_elements}/{self.total} elements outside "
                f"(abs<{self.abs_tol:g} or rel<{self.rel_tol:g}); "
                f"max_abs={self.max_abs:.6g} max_rel={self.max_rel:.6g}",
            )
        if abs(cos_diff) > self.cos_tol:
            return ComparisonResult(
                False, cos_diff, *common, True,
                f"{self.label}: cosine diff {cos_diff:.6g} exceeds {self.cos_tol:g}",
            )
        return ComparisonResult(True, cos_diff, *common, True, "")


def compare_tensors(
    ans: Any, ref: Any, *, label: str, abs_tol: float, rel_tol: float, cos_tol: float
) -> ComparisonResult:
    comparison = StreamingComparison(label, abs_tol, rel_tol, cos_tol)
    comparison.update(ans, ref)
    return comparison.result()


ACCURACY_FIELDS = (
    "out_max_abs", "out_max_rel", "out_cos_diff",
    "max_logits_max_abs", "max_logits_max_rel", "max_logits_cos_diff",
    "lse_max_abs", "lse_max_rel", "lse_cos_diff",
)
COS_LIMITS = {"out": OUT_COS_TOL, "max_logits": LOGIT_COS_TOL, "lse": LOGIT_COS_TOL}


BASE_SM_SCALE = 0.5


def logit_abs_tol(sm_scale: float) -> float:
    """Absolute limit for ``max_logits`` and ``lse`` at a given ``sm_scale``.

    ``abs 1e-6`` is calibrated at ``sm_scale = 0.5``.  The
    FP32 rounding error of a logit is proportional to ``sm_scale``, so for larger
    scales (the extra edge cases use 4.0) the absolute limit grows with it; at
    0.5 and below it is exactly ``abs 1e-6``.  The relative and cosine
    limits are unchanged.
    """
    return LOGIT_ABS_TOL * max(1.0, float(sm_scale) / BASE_SM_SCALE)


def check_outputs(prepared: Invocation, failure: type[GateFailure]) -> dict[str, float]:
    """Compare all three outputs against the chunked FP32 reference."""
    logit_abs = logit_abs_tol(prepared.sm_scale)
    comparisons = {
        "out": StreamingComparison("out", OUT_ABS_TOL, OUT_REL_TOL, OUT_COS_TOL),
        "max_logits": StreamingComparison("max_logits", logit_abs, LOGIT_REL_TOL, LOGIT_COS_TOL),
        "lse": StreamingComparison("lse", logit_abs, LOGIT_REL_TOL, LOGIT_COS_TOL),
    }
    for start, stop, out, max_logits, lse in reference_chunks(
        prepared.q, prepared.kv, prepared.indices, prepared.sm_scale,
        prepared.attn_sink, prepared.topk_length,
    ):
        comparisons["out"].update(prepared.out[start:stop], out)
        comparisons["max_logits"].update(prepared.max_logits[start:stop], max_logits)
        comparisons["lse"].update(prepared.lse[start:stop], lse)
        del out, max_logits, lse
    metrics: dict[str, float] = {}
    first_failure: tuple[str, ComparisonResult] | None = None
    for label, comparison in comparisons.items():
        result = comparison.result()
        metrics[f"{label}_max_abs"] = result.max_abs
        metrics[f"{label}_max_rel"] = result.max_rel
        metrics[f"{label}_cos_diff"] = result.cos_diff
        if not result.ok and first_failure is None:
            first_failure = (label, result)
    if first_failure is not None:
        label, result = first_failure
        kind = "unwritten_or_nan" if result.nan_elements else (
            "inf_position" if not result.anomaly_ok else
            "element_mismatch" if result.fail_elements else "cosine"
        )
        raise failure(
            result.detail,
            detail={"kind": kind, "tensor": label, **{k: v for k, v in metrics.items() if math.isfinite(v)}},
        )
    return metrics


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
            if name in prepared.pristine and not _torch_equal(
                tensor.reshape(-1).view(torch.uint8),
                prepared.pristine[name].reshape(-1).view(torch.uint8),
            )
        ]
        if changed:
            raise MemorySafetyFailure(
                f"candidate modified its inputs ({where}): {changed}",
                detail={"kind": "input_modified", "where": where, "inputs": changed},
            )


def _merge_metrics(into: dict[str, float], new: Mapping[str, float]) -> None:
    for key, value in new.items():
        current = into.get(key)
        if key.endswith("_cos_diff"):
            if current is None or abs(value) > abs(current):
                into[key] = float(value)
        elif current is None or value > current:
            into[key] = float(value)


# Fatal CUDA errors leave the context unusable: the run stops at once and the
# faulting kernel is never called again.
_FATAL_MARKERS = (
    "illegal memory access",
    "misaligned address",
    "out of bounds",
    "illegal address",
    "unspecified launch failure",
    "operation not supported on global/shared address space",
    "illegal instruction",
    "invalid program counter",
    "device-side assert",
    "hardware stack error",
    "warp out-of-range",
    "context is destroyed",
    "cudaerrorinvalidpc",
    "cudaerrorillegalinstruction",
    "launch failure",
)
_MEMORY_MARKERS = (
    "illegal memory access",
    "misaligned address",
    "out of bounds",
    "illegal address",
    "operation not supported on global/shared address space",
)


def classify_exception(exc: BaseException) -> str:
    """Map an exception raised inside a candidate call to the gate it fails."""
    if isinstance(exc, GateFailure):
        return exc.gate
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(marker in text for marker in _MEMORY_MARKERS):
        return "memory_safety_ok"
    return "correctness_ok"


def is_cuda_fault(exc: BaseException) -> bool:
    if isinstance(exc, GateFailure):
        return False
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _FATAL_MARKERS)


def context_alive() -> bool:
    """False when the CUDA context can no longer run work (a sticky error)."""
    if not torch.cuda.is_available():
        return True
    try:
        probe = torch.empty((1,), dtype=torch.float32, device=DEVICE)
        probe.fill_(1.0)
        _cuda_synchronize()
        return bool(probe.item() == 1.0)
    except Exception:  # noqa: BLE001 - any failure means the context is gone
        return False


def fatal_fault(exc: BaseException) -> bool:
    if isinstance(exc, GateFailure):
        return False
    return is_cuda_fault(exc) or not context_alive()


# --------------------------------------------------------------------------
# Timing.
# --------------------------------------------------------------------------
class L2Flusher:
    """Writes a buffer larger than any GPU L2 before each timed call."""

    def __init__(self, nbytes: int = L2_FLUSH_BYTES, device: Any = None) -> None:
        self.buffer = torch.empty((nbytes,), dtype=torch.uint8, device=DEVICE if device is None else device)

    def flush(self) -> None:
        self.buffer.zero_()


def _measure_once(entrypoint: Callable[..., Any], prepared: Invocation) -> tuple[float, float, float]:
    start_event = _Event(enable_timing=True)
    end_event = _Event(enable_timing=True)
    _cuda_synchronize()
    wall_start = _perf_counter()
    _event_record(start_event, _current_stream())
    invoke(entrypoint, prepared)
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
def check_written(prepared: Invocation) -> None:
    """Every output element written (no NaN); the check of an unchecked call."""
    for label, tensor in zip(("out", "max_logits", "lse"), prepared.outputs):
        if bool(torch.isnan(tensor).any().item()):
            raise CorrectnessFailure(f"{label}: unwritten or NaN elements",
                                     detail={"kind": "unwritten_or_nan", "tensor": label})


def stream_warm_shapes(calls: Sequence[Sequence[int]]) -> list[tuple[int, int]]:
    """STREAM_WARM_CALLS shapes next to the stream's own and equal to none of them.

    Even warm-up calls shift both sizes by a multiple of 16 (same residue mod 16),
    odd ones by 1-15 (another residue), so both integer-divisibility
    specializations compile before timing.
    """
    shapes = {(int(s_q), int(s_kv)) for s_q, s_kv in calls}
    largest = max(int(s_kv) for _, s_kv in calls)
    warm: list[tuple[int, int]] = []
    for number in range(STREAM_WARM_CALLS):
        s_q, s_kv = (int(value) for value in calls[(number * 7) % len(calls)])
        step = 16 if number % 2 == 0 else 1 + number % 15
        for _ in range(64):
            if (s_q, s_kv) not in shapes:
                break
            s_q = s_q - step if s_q > step else s_q + step
            s_kv = min(largest, max(s_q, s_kv - step if s_kv > step else s_kv + step))
        warm.append((s_q, s_kv))
    return warm


def _case_record(case: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case_id": str(case["case_id"]),
        "group": str(case["group"]),
        "tier": str(case["tier"]),
        "family": str(case["family"]),
        "scored": True,
        "stream": bool(case["stream"]),
        "entrypoint": PREFILL_ENTRYPOINT,
        "s_q": int(case["s_q"]),
        "s_kv": int(case["s_kv"]),
        "topk": int(case["topk"]),
        "h_q": int(case["h_q"]),
        "d_qk": int(case["d_qk"]),
        "sm_scale": float(case["sm_scale"]),
    }


def _run_single_case(
    entrypoint: Callable[..., Any], case: Mapping[str, Any], *, trials: int, warmups: int,
    flusher: L2Flusher, check: bool,
) -> dict[str, Any]:
    seed = int(case["seed"])
    s_q, s_kv = int(case["s_q"]), int(case["s_kv"])
    warm = prepare_scored_invocation(case, s_q, s_kv, derive_seed(seed, "warmup", 0))
    for _ in range(warmups):
        poison_outputs(warm)
        invoke(entrypoint, warm)
    _cuda_synchronize()
    check_memory(warm, "warm-up")
    check_written(warm)
    del warm
    torch.cuda.empty_cache()
    metrics: dict[str, float] = {}
    trial_us: list[float] = []
    trial_tflops: list[float] = []
    trial_positions: list[int] = []
    for trial in range(trials):
        prepared = prepare_scored_invocation(case, s_q, s_kv, derive_seed(seed, "measured", trial))
        flusher.flush()
        measured, _host_wall, _event_seconds = _measure_once(entrypoint, prepared)
        check_memory(prepared, "measured call")
        if check and trial == 0:
            _merge_metrics(metrics, check_outputs(prepared, CorrectnessFailure))
        else:
            check_written(prepared)
        trial_us.append(measured * 1.0e6)
        trial_tflops.append(effective_tflops(prepared.positions, prepared.h_q, prepared.d_qk, measured))
        trial_positions.append(int(prepared.positions))
        del prepared
        torch.cuda.empty_cache()
    return {
        **_case_record(case),
        "trial_us": trial_us,
        "trial_tflops": trial_tflops,
        "trial_positions": trial_positions,
        "median_us": finite_positive(_median(trial_us), "case median us"),
        "tflops": finite_positive(_median(trial_tflops), "case median TFLOPS"),
        "checked_calls": 1 if check else 0,
        **{key: float(value) for key, value in metrics.items()},
    }


def _run_stream_case(
    entrypoint: Callable[..., Any], case: Mapping[str, Any], *, flusher: L2Flusher, check: bool,
) -> dict[str, Any]:
    """Consecutive calls over one KV pool; see STREAM_WARM_CALLS."""
    seed = int(case["seed"])
    calls = [tuple(call) for call in case["calls"]]
    d_qk, h_q = int(case["d_qk"]), int(case["h_q"])
    arena = GuardedArena(True, DEVICE)
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(derive_seed(seed, "kv_pool", 0))
    host = torch.Generator()
    host.manual_seed(derive_seed(seed, "kv_pool_host", 0))
    pool = arena.empty("kv_pool", (int(case["s_kv"]), H_KV, d_qk), torch.bfloat16, row_bytes=2 * H_KV * d_qk)
    _fill_activation(pool, generator, host)
    pristine_pool = pool.clone()
    for number, (s_q, s_kv) in enumerate(stream_warm_shapes(calls)):
        warm = prepare_scored_invocation(case, s_q, s_kv, derive_seed(seed, "warmup", number),
                                         kv_pool=pool, pristine=False)
        invoke(entrypoint, warm)
        _cuda_synchronize()
        check_memory(warm, "stream warm-up")
        check_written(warm)
        del warm
    metrics: dict[str, float] = {}
    seconds = work = 0.0
    checked = run = 0
    for number, (s_q, s_kv) in enumerate(calls):
        prepared = prepare_scored_invocation(case, s_q, s_kv, derive_seed(seed, "call", number), kv_pool=pool)
        flusher.flush()
        measured, _host_wall, _event_seconds = _measure_once(entrypoint, prepared)
        check_memory(prepared, f"stream call {number}")
        if check and (number % STREAM_CHECK_EVERY == 0 or number == len(calls) - 1):
            _merge_metrics(metrics, check_outputs(prepared, CorrectnessFailure))
            checked += 1
        else:
            check_written(prepared)
        seconds += measured
        work += effective_flops(int(prepared.positions), h_q, d_qk)
        run += 1
        del prepared
        if seconds > STREAM_TIME_LIMIT_SECONDS:
            break
    if not bool(_torch_equal(pool.reshape(-1).view(torch.uint8), pristine_pool.reshape(-1).view(torch.uint8))):
        raise MemorySafetyFailure("candidate modified its inputs (stream KV pool)",
                                  detail={"kind": "input_modified", "inputs": ["kv"]})
    del pool, pristine_pool, arena
    torch.cuda.empty_cache()
    return {
        **_case_record(case),
        "calls_total": len(calls),
        "calls_run": run,
        "truncated": run < len(calls),
        "stream_seconds": finite_positive(seconds, "stream seconds"),
        "tflops": finite_positive(work / seconds / 1.0e12, "stream TFLOPS"),
        "checked_calls": checked,
        **{key: float(value) for key, value in metrics.items()},
    }


def run_scored_case(
    entrypoint: Callable[..., Any],
    raw_case: Mapping[str, Any],
    *,
    trials: int,
    warmups: int,
    flusher: L2Flusher,
    check: bool = True,
) -> dict[str, Any]:
    case = validate_scored_case(raw_case)
    if isinstance(trials, bool) or not isinstance(trials, int) or trials <= 0:
        raise ValueError("trials must be a positive integer")
    if isinstance(warmups, bool) or not isinstance(warmups, int) or warmups <= 0:
        raise ValueError("warmups must be a positive integer")
    if case["stream"]:
        return _run_stream_case(entrypoint, case, flusher=flusher, check=check)
    return _run_single_case(entrypoint, case, trials=trials, warmups=warmups, flusher=flusher, check=check)


def run_edge_case(
    entrypoint: Callable[..., Any],
    raw_case: Mapping[str, Any],
    *,
    guards: bool = True,
    calls: int = EDGE_CALLS,
) -> dict[str, Any]:
    """Untimed calls on one fresh invocation; raise on any failure."""
    case = validate_edge_case(raw_case)
    prepared = prepare_invocation(case, int(case["seed"]), guards=guards, pristine=True)
    metrics: dict[str, float] = {}
    for attempt in range(calls):
        if attempt:
            poison_outputs(prepared)
        invoke(entrypoint, prepared)
        _cuda_synchronize()
        check_memory(prepared, f"edge call {attempt + 1}")
        _merge_metrics(metrics, check_outputs(prepared, EdgeCorrectnessFailure))
    del prepared
    return {
        "case_id": str(case["case_id"]),
        "category": str(case["category"]),
        "ok": True,
        **{key: float(value) for key, value in metrics.items()},
    }


def aggregate_case_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """group -> tier -> family -> primary, each a geometric mean.

    A group's score is the geometric mean of its cases in one family; a tier's is
    the geometric mean of its groups; a family scores sqrt(evidence x sweep), or the
    one tier it has (partial public runs); the primary score is the geometric mean
    of the families present.
    """
    scored = [result for result in results if bool(result.get("scored"))]
    if not scored:
        raise ValueError("cannot aggregate without at least one scored case")
    cells: dict[str, dict[str, list[float]]] = {}
    for result in scored:
        value = finite_positive(result.get("tflops"), "case TFLOPS")
        family = str(result.get("family"))
        group = str(result.get("group"))
        if family not in FAMILIES or group not in GROUPS:
            raise ValueError("case result has invalid family or group")
        cells.setdefault(family, {}).setdefault(group, []).append(value)
    group_tflops: dict[str, float] = {}
    tier_tflops: dict[str, float] = {}
    family_tflops: dict[str, float] = {}
    for family in FAMILIES:
        if family not in cells:
            continue
        groups = {group: geometric_mean(values) for group, values in cells[family].items()}
        for group, value in groups.items():
            group_tflops[f"{family}:{group}"] = value
        tiers = {}
        for tier in TIERS:
            members = [value for group, value in groups.items() if GROUP_TIER[group] == tier]
            if members:
                tiers[tier] = geometric_mean(members)
                tier_tflops[f"{family}:{tier}"] = tiers[tier]
        family_tflops[family] = geometric_mean(tiers.values())
    return {
        "family_tflops": family_tflops,
        "tier_tflops": tier_tflops,
        "group_tflops": group_tflops,
        "primary_tflops": geometric_mean(family_tflops.values()),
    }


GATE_ORDER = (
    "edge_correctness_ok",
    "memory_safety_ok",
    "correctness_ok",
)


def protocol_record() -> dict[str, Any]:
    return {
        "edge_calls": EDGE_CALLS,
        "guard_min_bytes": GUARD_MIN_BYTES,
        "guard_rows": GUARD_ROWS,
        "soft_landing": {"indices": SOFT_INDEX, "topk_length": SOFT_LENGTH},
        "l2_flush_bytes": L2_FLUSH_BYTES,
        "stream_warm_calls": STREAM_WARM_CALLS,
        "stream_check_every": STREAM_CHECK_EVERY,
        "stream_time_limit_seconds": STREAM_TIME_LIMIT_SECONDS,
        "stale_tail_fraction": STALE_TAIL_FRACTION,
        "out_tolerance": [OUT_ABS_TOL, OUT_REL_TOL, OUT_COS_TOL],
        "logit_tolerance": [LOGIT_ABS_TOL, LOGIT_REL_TOL, LOGIT_COS_TOL],
    }


def run_suite(
    candidate_path: Path,
    scored_cases: Sequence[Mapping[str, Any]],
    edge_cases: Sequence[Mapping[str, Any]] = (),
    *,
    trials: int,
    warmups: int,
    require_hardware: bool = True,
    run_scored_after_edge_failure: bool = False,
    check: bool = True,
) -> dict[str, Any]:
    """Import one candidate, run the native edge cases, then the scored cases.

    Edge cases that answer wrongly do not stop the edge loop, so the failure
    count is complete; a memory-safety or fatal CUDA
    failure stops the run at once, and a faulting kernel is never called
    again.  Returns a schema-3 result; gate failures do not raise.
    Infrastructure errors still raise.
    """
    scored = [validate_scored_case(case) for case in scored_cases]
    edges = [validate_edge_case(case) for case in edge_cases]
    ids = [str(case["case_id"]) for case in (*scored, *edges)]
    if len(set(ids)) != len(ids):
        raise ValueError("suite case IDs must be unique")
    native_edges = [case for case in edges if case["native"]]

    use_exact_fp32()
    hardware = hardware_inventory() if require_hardware else {}
    gates: dict[str, bool | None] = {name: None for name in GATE_ORDER}
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "entrypoints": list(ENTRYPOINTS),
        "hardware": hardware,
        "trials_per_case": trials,
        "warmups": warmups,
        "protocol": protocol_record(),
        "edge": {
            "cases_total": len(native_edges),
            "cases_skipped_sanitize_only": len(edges) - len(native_edges),
            "cases_passed": 0,
            "cases_failed": 0,
            "results": [],
        },
        "cases": [],
    }

    def fail(gate: str, message: str, detail: Mapping[str, Any]) -> dict[str, Any]:
        gates[gate] = False
        payload["status"] = "failed"
        payload["failed_gate"] = gate
        payload["private_error"] = message
        payload["failure_detail"] = dict(detail)
        payload["gates"] = dict(gates)
        return payload

    try:
        _module, entrypoint, source_sha256 = load_candidate(Path(candidate_path))
    except Exception as exc:
        payload["candidate_sha256"] = None
        return fail("correctness_ok", f"candidate import failed: {type(exc).__name__}: {exc}", {"kind": "import"})
    payload["candidate_sha256"] = source_sha256

    edge_failures: list[dict[str, Any]] = []
    for case in native_edges:
        try:
            row = run_edge_case(entrypoint, case)
            payload["edge"]["results"].append(row)
            payload["edge"]["cases_passed"] += 1
        except Exception as exc:  # noqa: BLE001 - classified below
            fatal = fatal_fault(exc)
            gate = classify_exception(exc)
            if gate == "correctness_ok":
                gate = "memory_safety_ok" if fatal else "edge_correctness_ok"
            row = {
                "case_id": str(case["case_id"]),
                "category": str(case["category"]),
                "ok": False,
                "gate": gate,
                "fatal": fatal,
                "error": f"{type(exc).__name__}: {str(exc)[:400]}",
                "detail": dict(getattr(exc, "detail", {}) or {}),
            }
            payload["edge"]["results"].append(row)
            payload["edge"]["cases_failed"] += 1
            edge_failures.append(row)
            if fatal or gate == "memory_safety_ok":
                break
        torch.cuda.empty_cache()
    pending_edge_failure: tuple[str, str, dict[str, Any]] | None = None
    if edge_failures:
        # Memory safety outranks plain wrong answers.
        gate = "edge_correctness_ok"
        for row in edge_failures:
            if row["gate"] == "memory_safety_ok":
                gate = row["gate"]
                break
        message = (
            f"{len(edge_failures)} of {len(native_edges)} edge cases failed; first: "
            f"{edge_failures[0]['case_id']}: {edge_failures[0]['error']}"
        )
        pending_edge_failure = (gate, message, {"kind": "edge", "failed": len(edge_failures)})
        stopped = gate != "edge_correctness_ok" or any(row.get("fatal") for row in edge_failures)
        if stopped or not run_scored_after_edge_failure:
            return fail(*pending_edge_failure)
    elif native_edges:
        gates["edge_correctness_ok"] = True

    flusher = L2Flusher() if scored else None
    results: list[dict[str, Any]] = []
    for case in scored:
        try:
            results.append(
                run_scored_case(
                    entrypoint, case, trials=trials, warmups=warmups, flusher=flusher, check=check,
                )
            )
        except Exception as exc:  # noqa: BLE001 - classified below
            fatal = fatal_fault(exc)
            gate = classify_exception(exc)
            if fatal and gate == "correctness_ok":
                gate = "memory_safety_ok"
            payload["cases"] = results
            return fail(
                gate,
                f"case {case['case_id']} ({case['family']}/{case['group']}) failed: "
                f"{type(exc).__name__}: {exc}",
                {"case_id": str(case["case_id"]), "fatal": fatal,
                 **dict(getattr(exc, "detail", {}) or {})},
            )
        torch.cuda.empty_cache()
    payload["cases"] = results
    if results:
        payload.update(aggregate_case_results(results))
    if pending_edge_failure is not None:
        return fail(*pending_edge_failure)
    gates["correctness_ok"] = True
    gates["memory_safety_ok"] = True
    payload["status"] = "passed"
    payload["failed_gate"] = None
    payload["private_error"] = ""
    payload["gates"] = gates
    return payload


# Statuses of one case in the compute-sanitizer pass.
SANITIZE_PASSED = "passed"
# The call completed without a fatal error but answered wrongly or raised a
# non-fatal Python error.  Numbers computed under memcheck instrumentation are
# not trusted to fail a kernel (a correct kernel can be
# wrong only under memcheck), so the case's numeric verdict is deferred to its
# native run.
SANITIZE_DEFERRED = "deferred"
# A fatal CUDA error (context destroyed).  The pass stops here; the memcheck
# report decides whether it was an invalid access.
SANITIZE_FATAL = "fatal"
SANITIZE_MEMORY = "memory_safety"
SANITIZE_STATUSES = (SANITIZE_PASSED, SANITIZE_DEFERRED, SANITIZE_FATAL, SANITIZE_MEMORY)


def run_sanitize_cases(
    candidate_path: Path,
    edge_cases: Sequence[Mapping[str, Any]],
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Plain allocations (no guard bands) for compute-sanitizer; one call per case.

    Runs the cases flagged ``sanitize`` in ``sanitize_order``, one process, one
    case at a time.  It stops at the first fatal CUDA error or input mutation
    and never calls the kernel again after that.
    ``progress`` receives one line before and after each case so the caller can
    tell which case was in flight if the process dies.
    """
    cases = [validate_edge_case(case) for case in edge_cases]
    cases = [validate_edge_case(case) for case in sanitize_order(cases)]
    use_exact_fp32()
    rows: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "candidate_sha256": None,
        "order": [str(case["case_id"]) for case in cases],
        "cases": rows,
    }
    say = progress or (lambda _line: None)
    try:
        _module, entrypoint, source_sha256 = load_candidate(Path(candidate_path))
    except Exception as exc:
        return payload | {"status": "failed", "failed_gate": "correctness_ok",
                          "private_error": f"candidate import failed: {type(exc).__name__}: {exc}"}
    payload["candidate_sha256"] = source_sha256
    stop_status: str | None = None
    for index, case in enumerate(cases):
        say(f"SANITIZE_BEGIN {index} {case['case_id']}")
        row: dict[str, Any] = {"case_id": str(case["case_id"]), "category": str(case["category"])}
        try:
            result = run_edge_case(entrypoint, case, guards=False, calls=1)
            row.update(status=SANITIZE_PASSED, ok=True, **{
                key: value for key, value in result.items() if key in ACCURACY_FIELDS
            })
        except MemorySafetyFailure as exc:
            row.update(status=SANITIZE_MEMORY, ok=False, error=str(exc)[:400])
        except Exception as exc:  # noqa: BLE001 - classified below
            fatal = fatal_fault(exc)
            row.update(
                status=SANITIZE_FATAL if fatal else SANITIZE_DEFERRED,
                ok=False,
                error=f"{type(exc).__name__}: {str(exc)[:400]}",
            )
        rows.append(row)
        say(f"SANITIZE_END {index} {case['case_id']} {row['status']}")
        if row["status"] in (SANITIZE_FATAL, SANITIZE_MEMORY):
            stop_status = row["status"]
            break
        torch.cuda.empty_cache()
    if stop_status == SANITIZE_MEMORY:
        return payload | {"status": "failed", "failed_gate": "memory_safety_ok",
                          "private_error": f"sanitizer case {rows[-1]['case_id']}: {rows[-1]['error']}"}
    if stop_status == SANITIZE_FATAL:
        return payload | {"status": "fatal", "private_error":
                          f"sanitizer case {rows[-1]['case_id']}: {rows[-1]['error']}"}
    return payload | {"status": "completed"}
