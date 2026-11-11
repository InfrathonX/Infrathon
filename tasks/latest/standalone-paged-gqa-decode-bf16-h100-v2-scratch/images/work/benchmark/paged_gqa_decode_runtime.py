#!/usr/bin/env python3
"""Public test runtime for the paged-GQA decode task.

Generates the public cases, calls the candidate, times it and checks every
output against an FP32 reference over the exact BF16 inputs.  The public
controller provides the case manifests.
"""

from __future__ import annotations

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
_einsum = torch.einsum
_softmax = torch.softmax
_median = statistics.median
_fsum = math.fsum

DECODE_ENTRYPOINT = "paged_gqa_decode_bf16"
ENTRYPOINTS = (DECODE_ENTRYPOINT,)
GENERATOR_PROVENANCE = "paged-gqa-decode-v4-public"
SCHEMA_VERSION = 4

HEAD_DIM = 128
SOFTMAX_SCALE = 1.0 / math.sqrt(float(HEAD_DIM))

# Four scored families.  decode, spec and long cases are single calls; a stream
# case is a sequence of consecutive decode steps of one server.
FAMILIES = ("decode", "spec", "long", "stream")
CALL_FAMILIES = ("decode", "spec", "long")
# Shared prefixes are tracked in blocks of ``block_tokens`` tokens per case: 512
# for request-trace prefix caching, down to 16 for sequences that share pages
# (parallel sampling, beam search, a shared few-shot prompt).  The page size
# divides the block size.
BLOCK_SIZES = (16, 32, 64, 128, 256, 512)
SCORED_FILLS = ("randn", "nan")
# "outlier" draws Q, K and V from N(0, 1) + N(0, 100) * Bernoulli(0.001), the
# distribution of outlier features in large models.
SCORED_VALUES = ("randn", "outlier")
OUTLIER_PROBABILITY = 0.001
OUTLIER_SCALE = 10.0
MAX_STREAM_STEPS = 1024
MAX_STREAM_REQUESTS = 4096
STREAM_WARMUPS = 16
STREAM_CHECK_EVERY = 64
# Candidate time (sum of measured calls) allowed per sequence case.
STREAM_TIME_LIMIT_SECONDS = 120.0
MAX_SCORED_POOL_PAGES = 1 << 23
MAX_SCORED_KV_BYTES = 12 * (1 << 30)
MAX_STREAM_TABLE_ELEMENTS = 1 << 28
MAX_SCORED_WIDTH_FACTOR = 4

# Support domain (edge cases may use any of it).
MAX_B = 256
MAX_S_Q = 16
MAX_H_Q = 128
MAX_L_MAX = 1 << 20
PAGE_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
TABLE_MODES = ("perm", "identity", "reversed", "random", "shared", "tail_perm")
FILLS = ("randn", "nan", "inf")
VALUES = ("randn", "large_logits", "negative_logits")
LARGE_LOGIT_SCALE = 30.0
NEGATIVE_LOGIT_MAGNITUDE = 64.0
# A pool this small keeps every P // B mis-stride inside the 1 MiB page-table
# guard (4-byte entries), where the soft-landing fill is a valid page id.
MAX_POOL_PAGES = 1 << 18
MAX_WIDTH_FACTOR = 4
MAX_OUTPUT_ELEMENTS = 1 << 26
MAX_CASE_KV_BYTES = 2_684_354_560  # 2.5 GiB of K plus V for natively run cases
MAX_SANITIZE_ONLY_KV_BYTES = 10 * (1 << 30)
MAX_SEED = (1 << 63) - 1

# Correctness gates (max <= 2x + floor,
# mean <= 1.5x of the BF16 reordered reference, compared with <=, so a zero
# limit demands an exact answer), plus the v1 normalized-L2 backstop.
MAX_ABS_MULT = 2.0
MEAN_ABS_MULT = 1.5
MAX_ABS_FLOOR = 1.0e-5
NORM_L2_TOL = 5.0e-3

# Memory safety.  GUARD_BYTES is a multiple of every element size used here.
GUARD_BYTES = 1 << 20
GUARD_FILL_BYTE = 0xA5  # BF16 0xA5A5 is about -2.9e-16: finite, harmless when read
SOFT_PAGE_ID = 0
OUT_POISON = float("nan")



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


def pairs_for_lengths(s_q: int, lengths: Sequence[int]) -> int:
    """Eligible query-key pairs under bottom-right causal masking.

    Row ``i`` of a request with cache length ``L`` sees ``L - S_q + i + 1``
    keys, so a request contributes ``S_q*L - S_q*(S_q-1)/2`` pairs.
    """
    if isinstance(s_q, bool) or not isinstance(s_q, int) or s_q <= 0:
        raise RuntimeError("s_q must be a positive integer")
    total = 0
    for length in lengths:
        value = int(length)
        if value < s_q:
            raise RuntimeError("cache length below s_q cannot be scored")
        total += s_q * value - (s_q * (s_q - 1)) // 2
    return total


def effective_tflops(total_pairs: int, h_q: int, seconds: float) -> float:
    """Audit formula: 4 * HEAD_DIM * H_q * pairs (QK^T and P.V, 2 FLOPs each)."""
    measured = finite_positive(seconds, "measured seconds")
    for name, value in (("total pairs", total_pairs), ("h_q", h_q)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeError(f"effective {name} must be a positive integer")
    flops = 4.0 * HEAD_DIM * h_q * total_pairs
    return finite_positive(flops / measured / 1.0e12, "TFLOPS")


def derive_seed(case_seed: int, role: str, trial: int) -> int:
    """Per-call seed: 63 bits of SHA-256(case seed, role, trial)."""
    digest = hashlib.sha256(f"pgqa-v2:{int(case_seed)}:{role}:{int(trial)}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & MAX_SEED


def blocks_for(length: int, page_size: int) -> int:
    return -(-int(length) // int(page_size))


def kv_cache_bytes(pool_pages: int, page_size: int, h_kv: int) -> int:
    """K plus V bytes of one BF16 page pool."""
    return 4 * int(pool_pages) * int(page_size) * int(h_kv) * HEAD_DIM


# --------------------------------------------------------------------------
# Case validation.
# --------------------------------------------------------------------------
CALL_KEYS = frozenset({
    "case_id", "family", "s_q", "h_q", "h_kv", "page_size", "block_tokens", "lengths", "blocks",
    "table_width", "pool_pages", "fill", "values", "seed",
})
STREAM_KEYS = frozenset({
    "case_id", "family", "s_q", "h_q", "h_kv", "page_size", "block_tokens", "steps", "requests",
    "table_width", "pool_pages", "values", "seed",
})
EDGE_KEYS = frozenset({
    "case_id", "category", "s_q", "b", "l_max", "h_q", "h_kv", "page_size", "lengths",
    "table_mode", "pool_pages", "table_width", "fill", "values", "seed", "sanitize", "native",
})
# Fields the validators add.  Validation is idempotent: a validated case may be
# validated again; the derived fields are recomputed, never trusted.
SCORED_DERIVED_KEYS = frozenset({"b", "l_max", "total_pairs", "softmax_scale", "scored"})
EDGE_DERIVED_KEYS = frozenset({"softmax_scale", "scored"})


def _int_field(raw: Mapping[str, Any], name: str, *, minimum: int = 1, maximum: int | None = None) -> int:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def _heads_and_page(case: Mapping[str, Any]) -> tuple[int, int, int, int]:
    s_q = _int_field(case, "s_q", minimum=1, maximum=MAX_S_Q)
    h_q = _int_field(case, "h_q", minimum=1, maximum=MAX_H_Q)
    h_kv = _int_field(case, "h_kv", minimum=1, maximum=h_q)
    if h_q % h_kv:
        raise ValueError("H_q must be divisible by H_kv")
    page_size = _int_field(case, "page_size", minimum=1)
    if page_size not in PAGE_SIZES:
        raise ValueError(f"page_size must be one of {PAGE_SIZES}")
    block_tokens = _int_field(case, "block_tokens", minimum=1)
    if block_tokens not in BLOCK_SIZES or block_tokens % page_size:
        raise ValueError(f"block_tokens must be one of {BLOCK_SIZES} and a multiple of page_size")
    if case.get("values") not in SCORED_VALUES:
        raise ValueError(f"values must be one of {SCORED_VALUES}")
    return s_q, h_q, h_kv, page_size


def _block_list(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value
    ):
        raise ValueError(f"{label} must be a list of non-negative integers")
    if len(set(value)) != len(value):
        raise ValueError(f"{label} repeats a block")
    return [int(item) for item in value]


def call_page_demand(lengths: Sequence[int], blocks: Sequence[Sequence[int]], page_size: int,
                     block_tokens: int) -> int:
    """Physical pages one call needs: each distinct shared block once, plus
    every request's private tail."""
    per_block = block_tokens // page_size
    distinct = set()
    private = 0
    for length, ids in zip(lengths, blocks):
        distinct.update(ids)
        private += blocks_for(int(length) - len(ids) * block_tokens, page_size)
    return len(distinct) * per_block + private


def _validate_call_case(raw: Mapping[str, Any]) -> dict[str, Any]:
    missing = sorted(CALL_KEYS - set(raw))
    if missing:
        raise ValueError(f"scored case is missing fields: {missing}")
    extra = sorted(set(raw) - CALL_KEYS - SCORED_DERIVED_KEYS)
    if extra:
        raise ValueError(f"scored case has unexpected fields: {extra}")
    case = {key: raw[key] for key in CALL_KEYS}
    family = case["family"]
    if family not in CALL_FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")
    s_q, h_q, h_kv, page_size = _heads_and_page(case)
    if (family == "spec") != (s_q > 1):
        raise ValueError("spec cases have S_q > 1; decode and long cases have S_q = 1")
    lengths = case["lengths"]
    if (
        not isinstance(lengths, list)
        or not 1 <= len(lengths) <= MAX_B
        or any(isinstance(value, bool) or not isinstance(value, int) for value in lengths)
        or any(not s_q <= value <= MAX_L_MAX for value in lengths)
    ):
        raise ValueError(f"lengths must be 1 to {MAX_B} integers in [s_q, {MAX_L_MAX}]")
    raw_blocks = case["blocks"]
    if not isinstance(raw_blocks, list) or len(raw_blocks) != len(lengths):
        raise ValueError("blocks must hold one list per request")
    blocks = [_block_list(item, "blocks entry") for item in raw_blocks]
    block_tokens = int(case["block_tokens"])
    for length, ids in zip(lengths, blocks):
        if len(ids) * block_tokens > length:
            raise ValueError("a request's shared blocks exceed its length")
    b, l_max = len(lengths), max(lengths)
    needed = blocks_for(l_max, page_size)
    _int_field(case, "table_width", minimum=needed, maximum=MAX_SCORED_WIDTH_FACTOR * needed)
    demand = call_page_demand(lengths, blocks, page_size, block_tokens)
    pool = _int_field(case, "pool_pages", minimum=demand, maximum=MAX_SCORED_POOL_PAGES)
    if pool > 2 * demand + 4096:
        raise ValueError("pool_pages is far larger than the call needs")
    if kv_cache_bytes(pool, page_size, h_kv) > MAX_SCORED_KV_BYTES:
        raise ValueError("case KV pool exceeds the scored allocation cap")
    if b * s_q * h_q * HEAD_DIM > MAX_OUTPUT_ELEMENTS:
        raise ValueError("case output exceeds the size bound")
    if case["fill"] not in SCORED_FILLS:
        raise ValueError(f"fill must be one of {SCORED_FILLS}")
    _int_field(case, "seed", minimum=0, maximum=MAX_SEED)
    if not isinstance(case["case_id"], str) or not case["case_id"]:
        raise ValueError("case_id must be a nonempty string")
    case.update(
        lengths=[int(value) for value in lengths], blocks=blocks, b=b, l_max=l_max,
        total_pairs=pairs_for_lengths(s_q, lengths), softmax_scale=SOFTMAX_SCALE, scored=True,
    )
    return case


@dataclass
class StreamPlan:
    """Exact schedule of one sequence case."""

    active: list[list[int]]          # per step: request indices, in request order
    starts: list[int]
    inputs: list[int]
    blocks: list[list[int]]
    finals: list[int]                # length at each request's last step
    total_pairs: int
    peak_pages: int


def stream_plan(case: Mapping[str, Any]) -> StreamPlan:
    """Expand a sequence case.  Request r is active at steps [start, end) with
    length input + (step - start).  Pages are reserved for a request's final
    length when it joins and released when it leaves (shared blocks when their
    last user leaves); ``peak_pages`` is the most pages ever held at once."""
    steps = int(case["steps"])
    s_q, page_size = int(case["s_q"]), int(case["page_size"])
    block_tokens = int(case["block_tokens"])
    per_block = block_tokens // page_size
    starts, ends, inputs, blocks, finals = [], [], [], [], []
    for raw in case["requests"]:
        if not isinstance(raw, list) or len(raw) != 4:
            raise ValueError("a stream request is [start, end, input_length, blocks]")
        start, end, length = raw[0], raw[1], raw[2]
        for value in (start, end, length):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("stream request fields must be integers")
        if not 0 <= start < end <= steps or length < s_q:
            raise ValueError("stream request window or length is invalid")
        ids = _block_list(raw[3], "stream request blocks")
        final = length + (end - start - 1)
        if final > MAX_L_MAX or len(ids) * block_tokens > length:
            raise ValueError("stream request exceeds the length domain")
        starts.append(start); ends.append(end); inputs.append(length); blocks.append(ids); finals.append(final)
    joins: dict[int, list[int]] = {}
    leaves: dict[int, list[int]] = {}
    for index, (start, end) in enumerate(zip(starts, ends)):
        joins.setdefault(start, []).append(index)
        leaves.setdefault(end, []).append(index)
    active: list[list[int]] = []
    live: set[int] = set()
    refs: dict[int, int] = {}
    held = peak = total = 0
    for step in range(steps):
        for index in leaves.get(step, []):
            live.discard(index)
            held -= blocks_for(finals[index] - len(blocks[index]) * block_tokens, page_size)
            for block in blocks[index]:
                refs[block] -= 1
                if not refs[block]:
                    del refs[block]
                    held -= per_block
        for index in joins.get(step, []):
            live.add(index)
            held += blocks_for(finals[index] - len(blocks[index]) * block_tokens, page_size)
            for block in blocks[index]:
                if block not in refs:
                    refs[block] = 0
                    held += per_block
                refs[block] += 1
        peak = max(peak, held)
        now = sorted(live)
        if not 1 <= len(now) <= MAX_B:
            raise ValueError(f"every stream step needs 1 to {MAX_B} active requests")
        active.append(now)
        total += pairs_for_lengths(s_q, [inputs[i] + step - starts[i] for i in now])
    return StreamPlan(active, starts, inputs, blocks, finals, total, peak)


def _validate_stream_case(raw: Mapping[str, Any]) -> dict[str, Any]:
    missing = sorted(STREAM_KEYS - set(raw))
    if missing:
        raise ValueError(f"stream case is missing fields: {missing}")
    extra = sorted(set(raw) - STREAM_KEYS - SCORED_DERIVED_KEYS)
    if extra:
        raise ValueError(f"stream case has unexpected fields: {extra}")
    case = {key: raw[key] for key in STREAM_KEYS}
    if case["family"] != "stream":
        raise ValueError("a stream case has family 'stream'")
    s_q, h_q, h_kv, page_size = _heads_and_page(case)
    _int_field(case, "steps", minimum=1, maximum=MAX_STREAM_STEPS)
    requests = case["requests"]
    if not isinstance(requests, list) or not 1 <= len(requests) <= MAX_STREAM_REQUESTS:
        raise ValueError(f"requests must hold 1 to {MAX_STREAM_REQUESTS} entries")
    plan = stream_plan(case)
    l_max = max(plan.finals)
    needed = blocks_for(l_max, page_size)
    width = _int_field(case, "table_width", minimum=needed, maximum=MAX_SCORED_WIDTH_FACTOR * needed)
    if len(requests) * width > MAX_STREAM_TABLE_ELEMENTS:
        raise ValueError("stream page tables exceed the size bound")
    pool = _int_field(case, "pool_pages", minimum=plan.peak_pages, maximum=MAX_SCORED_POOL_PAGES)
    if pool > 2 * plan.peak_pages + 4096:
        raise ValueError("pool_pages is far larger than the stream needs")
    if kv_cache_bytes(pool, page_size, h_kv) > MAX_SCORED_KV_BYTES:
        raise ValueError("stream KV pool exceeds the scored allocation cap")
    if MAX_B * s_q * h_q * HEAD_DIM > MAX_OUTPUT_ELEMENTS:
        raise ValueError("stream output exceeds the size bound")
    _int_field(case, "seed", minimum=0, maximum=MAX_SEED)
    if not isinstance(case["case_id"], str) or not case["case_id"]:
        raise ValueError("case_id must be a nonempty string")
    case.update(
        b=max(len(now) for now in plan.active), l_max=l_max, total_pairs=plan.total_pairs,
        softmax_scale=SOFTMAX_SCALE, scored=True,
    )
    return case


def validate_scored_case(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one scored case (a single call or a sequence)."""
    if not isinstance(raw, Mapping):
        raise ValueError("scored case must be an object")
    if raw.get("family") == "stream":
        return _validate_stream_case(raw)
    return _validate_call_case(raw)


def _validate_geometry(case: Mapping[str, Any]) -> tuple[int, int, int, int, int, int]:
    s_q = _int_field(case, "s_q", minimum=1, maximum=MAX_S_Q)
    b = _int_field(case, "b", minimum=1, maximum=MAX_B)
    h_q = _int_field(case, "h_q", minimum=1, maximum=MAX_H_Q)
    h_kv = _int_field(case, "h_kv", minimum=1, maximum=h_q)
    if h_q % h_kv:
        raise ValueError("H_q must be divisible by H_kv")
    page_size = _int_field(case, "page_size", minimum=1)
    if page_size not in PAGE_SIZES:
        raise ValueError(f"page_size must be one of {PAGE_SIZES}")
    l_max = _int_field(case, "l_max", minimum=s_q, maximum=MAX_L_MAX)
    return s_q, b, h_q, h_kv, page_size, l_max


def validate_edge_case(raw: Mapping[str, Any]) -> dict[str, Any]:
    missing = sorted(EDGE_KEYS - set(raw))
    if missing:
        raise ValueError(f"edge case is missing fields: {missing}")
    extra = sorted(set(raw) - EDGE_KEYS - EDGE_DERIVED_KEYS)
    if extra:
        raise ValueError(f"edge case has unexpected fields: {extra}")
    case = {key: raw[key] for key in EDGE_KEYS}
    if not isinstance(case["case_id"], str) or not case["case_id"]:
        raise ValueError("case_id must be a nonempty string")
    if not isinstance(case["category"], str) or not case["category"]:
        raise ValueError("category must be a nonempty string")
    s_q, b, h_q, h_kv, page_size, l_max = _validate_geometry(case)
    lengths = case["lengths"]
    if (
        not isinstance(lengths, list)
        or len(lengths) != b
        or any(isinstance(value, bool) or not isinstance(value, int) for value in lengths)
        or any(not s_q <= value <= l_max for value in lengths)
    ):
        raise ValueError("lengths must be b integers in [s_q, l_max]")
    needed = blocks_for(l_max, page_size)
    width = _int_field(case, "table_width", minimum=needed, maximum=MAX_WIDTH_FACTOR * needed)
    pool = _int_field(case, "pool_pages", minimum=1, maximum=MAX_POOL_PAGES)
    mode = case["table_mode"]
    if mode not in TABLE_MODES:
        raise ValueError(f"table_mode must be one of {TABLE_MODES}")
    if mode != "random" and pool < b * width:
        raise ValueError("a one-to-one table needs pool_pages >= b * table_width")
    if case["fill"] not in FILLS:
        raise ValueError(f"fill must be one of {FILLS}")
    if case["values"] not in VALUES:
        raise ValueError(f"values must be one of {VALUES}")
    _int_field(case, "seed", minimum=0, maximum=MAX_SEED)
    for flag in ("sanitize", "native"):
        if not isinstance(case[flag], bool):
            raise ValueError(f"{flag} must be a boolean")
    if not case["native"] and not case["sanitize"]:
        raise ValueError("a case that never runs natively must be a sanitizer case")
    if b * s_q * h_q * HEAD_DIM > MAX_OUTPUT_ELEMENTS:
        raise ValueError("edge output exceeds the edge-suite size bound")
    limit = MAX_CASE_KV_BYTES if case["native"] else MAX_SANITIZE_ONLY_KV_BYTES
    if kv_cache_bytes(pool, page_size, h_kv) > limit:
        raise ValueError("edge KV pool exceeds the edge-suite size bound")
    case["softmax_scale"] = SOFTMAX_SCALE
    case["scored"] = False
    return case


# --------------------------------------------------------------------------
# Hardware.
# --------------------------------------------------------------------------
# Capability envelope: what the task contract actually needs, not the GPU model
# it was qualified on.  The GPU name and UUID are recorded as provenance only;
# they never gate admission.  Keep these two constants identical in the Judge
# verifier (tests assert the pair).
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
    """Freeze the reference at true FP32; TF32 must never move the oracle."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


class _NoChecks:
    """The public runner does not run the Judge's integrity checks."""

    checks = 0

    def check(self, where: str) -> None:
        return None


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
    module_name = f"paged_gqa_decode_candidate_{source_sha256[:24]}"
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
    value = getattr(module, DECODE_ENTRYPOINT, None)
    if not callable(value):
        raise RuntimeError(f"candidate must define callable {DECODE_ENTRYPOINT}")
    return module, value, source_sha256


# --------------------------------------------------------------------------
# Guarded allocation with soft-landing fills.
# --------------------------------------------------------------------------
@dataclass
class _Region:
    name: str
    raw: Any
    nbytes: int
    dtype: Any
    fill: Any  # a byte for floating tensors, an element value for int32 tensors


def _fill_guard(view: Any, dtype: Any, fill: Any) -> None:
    if dtype == torch.int32:
        view.view(torch.int32).fill_(int(fill))
    else:
        view.fill_(int(fill))


def _guard_intact(view: Any, dtype: Any, fill: Any) -> bool:
    if dtype == torch.int32:
        return bool(view.view(torch.int32).eq(int(fill)).all().item())
    return bool(view.eq(int(fill)).all().item())


class GuardedArena:
    """Every tensor lives in the middle of a buffer with filled guard bands.

    BF16 tensors get 0xA5 bytes (a finite BF16 value).  Int32 metadata gets a
    valid element value: the page table's bands hold a valid page id and the
    lengths' bands a valid length, so an out-of-bounds metadata read lands on
    real cache memory (a wrong answer) instead of a wild address.
    """

    def __init__(self, guards: bool) -> None:
        self.guards = guards
        self.regions: list[_Region] = []

    def empty(self, name: str, shape: Sequence[int], dtype: Any, *, fill: Any = GUARD_FILL_BYTE) -> Any:
        element = torch.empty((), dtype=dtype).element_size()
        nbytes = math.prod(int(item) for item in shape) * element
        if not self.guards:
            return torch.empty(tuple(shape), dtype=dtype, device="cuda")
        raw = torch.empty((GUARD_BYTES + nbytes + GUARD_BYTES,), dtype=torch.uint8, device="cuda")
        _fill_guard(raw[:GUARD_BYTES], dtype, fill)
        _fill_guard(raw[GUARD_BYTES + nbytes :], dtype, fill)
        view = raw[GUARD_BYTES : GUARD_BYTES + nbytes].view(dtype).view(tuple(shape))
        if not view.is_contiguous():  # pragma: no cover - defensive
            raise RuntimeError("guarded view is not contiguous")
        self.regions.append(_Region(name, raw, nbytes, dtype, fill))
        return view

    def violated(self) -> list[str]:
        problems: list[str] = []
        for region in self.regions:
            if not _guard_intact(region.raw[:GUARD_BYTES], region.dtype, region.fill):
                problems.append(f"{region.name}: write before the tensor")
            if not _guard_intact(region.raw[GUARD_BYTES + region.nbytes :], region.dtype, region.fill):
                problems.append(f"{region.name}: write past the tensor")
        return problems


# --------------------------------------------------------------------------
# Workload generation.
# --------------------------------------------------------------------------
@dataclass
class Invocation:
    case_id: str
    s_q: int
    b: int
    l_max: int
    h_q: int
    h_kv: int
    page_size: int
    lengths: list[int]
    arena: GuardedArena
    q: Any
    k_cache: Any
    v_cache: Any
    page_table: Any
    cache_seqlens: Any
    out: Any
    total_pairs: int
    pristine: dict[str, Any] = field(default_factory=dict)

    @property
    def inputs(self) -> dict[str, Any]:
        return {
            "q": self.q,
            "k_cache": self.k_cache,
            "v_cache": self.v_cache,
            "page_table": self.page_table,
            "cache_seqlens": self.cache_seqlens,
        }


def precall_lengths(case: Mapping[str, Any], seed: int) -> list[int]:
    """Untimed-call lengths: each request keeps its shared blocks and is shortened
    by up to 20%, so the batch, table width and pool stay the same."""
    generator = random.Random(seed)
    s_q = int(case["s_q"])
    out = []
    for length, ids in zip(case["lengths"], case["blocks"]):
        floor = max(s_q, len(ids) * int(case["block_tokens"]))
        cut = int(int(length) * 0.2 * generator.random())
        out.append(max(floor, int(length) - cut))
    return out


def build_call_table(case: Mapping[str, Any], lengths: Sequence[int], generator: Any) -> Any:
    """[B, width] int32 page table on the host.  Each distinct shared block owns
    block_tokens // page_size physical pages that every request listing it reads;
    every request's tail gets its own pages; physical ids come from one random
    permutation of the pool (a fragmented allocator); entries past a request's
    last page are arbitrary valid ids."""
    page_size, width, pool = int(case["page_size"]), int(case["table_width"]), int(case["pool_pages"])
    block_tokens = int(case["block_tokens"])
    per_block = block_tokens // page_size
    order: dict[int, int] = {}
    for ids in case["blocks"]:
        for block in ids:
            order.setdefault(int(block), len(order))
    perm = torch.randperm(pool, generator=generator)
    shared = perm[: len(order) * per_block].view(len(order), per_block)
    cursor = len(order) * per_block
    table = torch.randint(0, pool, (len(lengths), width), generator=generator, dtype=torch.int64)
    for row, (length, ids) in enumerate(zip(lengths, case["blocks"])):
        private = blocks_for(int(length) - len(ids) * block_tokens, page_size)
        parts = []
        if ids:
            parts.append(shared[[order[int(block)] for block in ids]].reshape(-1))
        parts.append(perm[cursor : cursor + private])
        cursor += private
        pages = torch.cat(parts)
        table[row, : pages.numel()] = pages
    if cursor > pool:
        raise RuntimeError("call layout exceeds its page pool")
    return table.to(torch.int32)


def prepare_call(case: Mapping[str, Any], lengths: Sequence[int], seed: int, *,
                 guards: bool = True, pristine: bool = True) -> "Invocation":
    """Generate one scored single call (measured, precall or warm-up lengths)."""
    host_generator = torch.Generator()
    host_generator.manual_seed(derive_seed(seed, "layout", 0))
    table_host = build_call_table(case, lengths, host_generator)
    return _materialize(
        case_id=str(case["case_id"]), s_q=int(case["s_q"]), h_q=int(case["h_q"]),
        h_kv=int(case["h_kv"]), page_size=int(case["page_size"]), pool=int(case["pool_pages"]),
        lengths=[int(value) for value in lengths], table_host=table_host, seed=seed,
        fill=str(case["fill"]), values=str(case["values"]), guards=guards, pristine=pristine,
    )


def _build_table(spec: Mapping[str, Any], generator: Any) -> Any:
    b, width, pool = int(spec["b"]), int(spec["table_width"]), int(spec["pool_pages"])
    mode = str(spec["table_mode"])
    count = b * width
    if mode == "identity":
        flat = torch.arange(count, dtype=torch.int64)
    elif mode == "reversed":
        flat = torch.arange(pool - 1, pool - 1 - count, -1, dtype=torch.int64)
    elif mode == "tail_perm":
        flat = (pool - count) + torch.randperm(count, generator=generator)
    elif mode == "random":
        flat = torch.randint(0, pool, (count,), generator=generator)
    else:  # perm and shared start from a one-to-one draw over the whole pool
        flat = torch.randperm(pool, generator=generator)[:count]
    table = flat.view(b, width).clone()
    if mode == "shared":
        # Each odd request reuses the full pages of its predecessor's shared
        # prefix (one physical block, several logical blocks).  Shared tokens therefore hold identical K/V.
        page_size = int(spec["page_size"])
        lengths = spec["lengths"]
        for row in range(1, b, 2):
            shared_tokens = min(int(lengths[row - 1]), int(lengths[row])) // 2
            pages = shared_tokens // page_size
            if pages:
                table[row, :pages] = table[row - 1, :pages]
    return table.to(torch.int32)


def _visible_slots(table: Any, lengths: Sequence[int], pool: int, page_size: int) -> Any:
    """[pool, page_size] bool: slots some query of some request may attend."""
    visible = torch.zeros((pool, page_size), dtype=torch.bool)
    for row, length in enumerate(lengths):
        positions = torch.arange(int(length))
        pages = table[row, positions // page_size].long()
        visible[pages, positions % page_size] = True
    return visible


def prepare_invocation(
    spec: Mapping[str, Any],
    seed: int,
    *,
    guards: bool = True,
    pristine: bool = True,
) -> Invocation:
    """Generate one invocation from a validated edge-case spec."""
    host_generator = torch.Generator()
    host_generator.manual_seed(derive_seed(seed, "layout", 0))
    table_host = _build_table(spec, host_generator)
    return _materialize(
        case_id=str(spec["case_id"]), s_q=int(spec["s_q"]), h_q=int(spec["h_q"]),
        h_kv=int(spec["h_kv"]), page_size=int(spec["page_size"]), pool=int(spec["pool_pages"]),
        lengths=[int(value) for value in spec["lengths"]], table_host=table_host, seed=seed,
        fill=str(spec["fill"]), values=str(spec["values"]), guards=guards, pristine=pristine,
    )


def _add_outliers(tensor: Any, generator: Any) -> None:
    """In place: x + N(0, 100) * Bernoulli(0.001), chunked to bound scratch."""
    flat = tensor.view(-1)
    chunk = 1 << 26
    for start in range(0, flat.numel(), chunk):
        part = flat[start : start + chunk]
        spikes = torch.randn(part.shape, dtype=torch.float32, device=part.device, generator=generator)
        keep = torch.rand(part.shape, dtype=torch.float32, device=part.device, generator=generator)
        spikes.mul_((keep < OUTLIER_PROBABILITY).float() * OUTLIER_SCALE)
        part.add_(spikes.to(part.dtype))
        del spikes, keep


def _materialize(
    *, case_id: str, s_q: int, h_q: int, h_kv: int, page_size: int, pool: int,
    lengths: list[int], table_host: Any, seed: int, fill: str, values: str,
    guards: bool, pristine: bool,
) -> Invocation:
    b, width = len(lengths), int(table_host.shape[1])
    arena = GuardedArena(guards)
    q = arena.empty("q", (b, s_q, h_q, HEAD_DIM), torch.bfloat16)
    k_cache = arena.empty("k_cache", (pool, page_size, h_kv, HEAD_DIM), torch.bfloat16)
    v_cache = arena.empty("v_cache", (pool, page_size, h_kv, HEAD_DIM), torch.bfloat16)
    page_table = arena.empty("page_table", (b, width), torch.int32, fill=SOFT_PAGE_ID)
    cache_seqlens = arena.empty("cache_seqlens", (b,), torch.int32, fill=s_q)
    out = arena.empty("out", (b, s_q, h_q, HEAD_DIM), torch.bfloat16)

    device_generator = torch.Generator(device="cuda")
    device_generator.manual_seed(derive_seed(seed, "values", 0))
    if values == "negative_logits":
        # Extreme negative logits: every visible logit is far below any finite
        # masking value (here about -4.6e4).
        q.fill_(NEGATIVE_LOGIT_MAGNITUDE)
        k_cache.fill_(-NEGATIVE_LOGIT_MAGNITUDE)
    else:
        q.copy_(torch.randn(q.shape, dtype=torch.bfloat16, device="cuda", generator=device_generator))
        if values == "large_logits":
            q.mul_(LARGE_LOGIT_SCALE)
        k_cache.copy_(torch.randn(k_cache.shape, dtype=torch.bfloat16, device="cuda", generator=device_generator))
    v_cache.copy_(torch.randn(v_cache.shape, dtype=torch.bfloat16, device="cuda", generator=device_generator))
    if values == "outlier":
        for tensor in (q, k_cache, v_cache):
            _add_outliers(tensor, device_generator)
    if fill != "randn":
        hidden = ~_visible_slots(table_host, lengths, pool, page_size).to("cuda")
        value = float("nan") if fill == "nan" else float("inf")
        k_cache[hidden] = value
        v_cache[hidden] = value
        del hidden
    page_table.copy_(table_host.to("cuda"))
    cache_seqlens.copy_(torch.tensor(lengths, dtype=torch.int32).to("cuda"))
    out.fill_(OUT_POISON)

    invocation = Invocation(
        case_id=case_id, s_q=s_q, b=b, l_max=max(lengths), h_q=h_q, h_kv=h_kv,
        page_size=page_size, lengths=lengths, arena=arena, q=q, k_cache=k_cache,
        v_cache=v_cache, page_table=page_table, cache_seqlens=cache_seqlens, out=out,
        total_pairs=pairs_for_lengths(s_q, lengths),
    )
    if pristine:
        invocation.pristine = {name: tensor.clone() for name, tensor in invocation.inputs.items()}
    return invocation


def invoke(entrypoint: Callable[..., Any], prepared: Invocation) -> None:
    returned = entrypoint(
        prepared.q,
        prepared.k_cache,
        prepared.v_cache,
        prepared.page_table,
        prepared.cache_seqlens,
        SOFTMAX_SCALE,
        prepared.out,
    )
    if returned is not None:
        raise CorrectnessFailure(f"{DECODE_ENTRYPOINT} must return None")


def poison_output(prepared: Invocation) -> None:
    prepared.out.fill_(OUT_POISON)


# --------------------------------------------------------------------------
# Reference and checks.
# --------------------------------------------------------------------------
def _depage(cache: Any, table_row: Any, length: int, page_size: int) -> Any:
    """Linearise one request's cache into logical order: [length, H_kv, D]."""
    blocks = blocks_for(length, page_size)
    physical = table_row[:blocks].long()
    linear = cache[physical].reshape(blocks * page_size, cache.shape[2], HEAD_DIM)
    return linear[:length]


@dataclass
class Reference:
    golden: Any
    max_abs_limit: float
    mean_abs_limit: float


def reference_outputs(prepared: Invocation) -> Reference:
    """Per-request FP32 de-paged attention over the exact BF16 inputs, and the
    limits a BF16 reordered pass sets .  Only visible keys are ever read, so hidden
    slots (NaN included) cannot reach the reference."""
    s_q, h_q, h_kv = prepared.s_q, prepared.h_q, prepared.h_kv
    rho = h_q // h_kv
    device = prepared.q.device
    golden = torch.empty((prepared.b, s_q, h_q, HEAD_DIM), dtype=torch.float32, device=device)
    bf16_max_abs = 0.0
    bf16_mean_sum = 0.0
    for row in range(prepared.b):
        length = int(prepared.lengths[row])
        keys = _depage(prepared.k_cache, prepared.page_table[row], length, prepared.page_size)
        values = _depage(prepared.v_cache, prepared.page_table[row], length, prepared.page_size)
        query = prepared.q[row].view(s_q, h_kv, rho, HEAD_DIM)
        positions = torch.arange(length, device=device)
        limit = (length - s_q + torch.arange(s_q, device=device)).unsqueeze(-1)
        masked = (positions.unsqueeze(0) > limit).unsqueeze(0).unsqueeze(0)

        scores = _einsum("qhrd,khd->hrqk", query.float(), keys.float()) * SOFTMAX_SCALE
        weights = _softmax(scores.masked_fill(masked, float("-inf")), dim=-1)
        reference = _einsum("hrqk,khd->qhrd", weights, values.float()).reshape(s_q, h_q, HEAD_DIM)
        golden[row] = reference
        del scores, weights

        scores16 = _einsum("qhrd,khd->hrqk", query, keys).float() * SOFTMAX_SCALE
        weights16 = _softmax(scores16.masked_fill(masked, float("-inf")), dim=-1)
        approx = _einsum("hrqk,khd->qhrd", weights16.to(torch.bfloat16), values).float()
        delta = (approx.reshape(s_q, h_q, HEAD_DIM) - reference).abs()
        bf16_max_abs = max(bf16_max_abs, float(delta.max().item()))
        bf16_mean_sum += float(delta.sum().item())
        del scores16, weights16, approx, delta, reference, keys, values
    rounding_floor = float((2.0 * (((golden + 0.3) - 0.3) - golden).abs().max()).item())
    max_abs_limit = MAX_ABS_MULT * bf16_max_abs + max(MAX_ABS_FLOOR, rounding_floor)
    mean_abs_limit = MEAN_ABS_MULT * (bf16_mean_sum / golden.numel())
    if not (math.isfinite(max_abs_limit) and math.isfinite(mean_abs_limit)):
        raise RuntimeError("reference tolerance limits are not finite")
    if max_abs_limit <= 0.0 or mean_abs_limit < 0.0:
        raise RuntimeError("reference tolerance limits are invalid")
    return Reference(golden, max_abs_limit, mean_abs_limit)


def check_output(
    prepared: Invocation, reference: Reference, failure: type[GateFailure]
) -> dict[str, float]:
    out = prepared.out
    finite = _isfinite(out)
    if not bool(finite.all().item()):
        bad = int((~finite).sum().item())
        raise failure(
            f"out contains {bad} unwritten, NaN or infinite elements",
            detail={"kind": "unwritten_or_nonfinite", "elements": bad},
        )
    delta = (out.float() - reference.golden).abs()
    max_abs = float(delta.max().item())
    mean_abs = float(delta.mean().item())
    denominator = max(float(reference.golden.square().sum().sqrt().item()), 1.0e-12)
    norm_l2 = float(delta.square().sum().sqrt().item()) / denominator
    del delta
    metrics = {
        "out_max_abs": max_abs,
        "out_mean_abs": mean_abs,
        "out_norm_l2": norm_l2,
        "out_max_abs_limit": reference.max_abs_limit,
        "out_mean_abs_limit": reference.mean_abs_limit,
    }
    if not max_abs <= reference.max_abs_limit:
        raise failure(
            f"out max abs error {max_abs:.6g} exceeds differential limit {reference.max_abs_limit:.6g}",
            detail={"kind": "max_abs", **metrics},
        )
    if not mean_abs <= reference.mean_abs_limit:
        raise failure(
            f"out mean abs error {mean_abs:.6g} exceeds differential limit {reference.mean_abs_limit:.6g}",
            detail={"kind": "mean_abs", **metrics},
        )
    if not norm_l2 <= NORM_L2_TOL:
        raise failure(
            f"out normalized L2 error {norm_l2:.6g} exceeds {NORM_L2_TOL:g}",
            detail={"kind": "norm_l2", **metrics},
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


def _merge_metrics(into: dict[str, float], new: Mapping[str, float]) -> None:
    # Keep each differential error paired with the limit from the same call.
    for metric in ("out_max_abs", "out_mean_abs"):
        limit = f"{metric}_limit"
        if metric not in into or new[metric] > into[metric]:
            into[metric] = float(new[metric])
            into[limit] = float(new[limit])
    if "out_norm_l2" not in into or new["out_norm_l2"] > into["out_norm_l2"]:
        into["out_norm_l2"] = float(new["out_norm_l2"])


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
        "operation not supported on global/shared address space",
    ):
        if marker in text:
            return "memory_safety_ok"
    return "correctness_ok"


def is_cuda_fault(exc: BaseException) -> bool:
    return classify_exception(exc) == "memory_safety_ok" and not isinstance(exc, GateFailure)


# --------------------------------------------------------------------------
# Timing.
# --------------------------------------------------------------------------
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
def _check_bound(case_id: str, value: float, bound_tflops: float | None) -> None:
    if bound_tflops is not None and value > bound_tflops:
        raise HarnessIntegrityFailure(
            f"case {case_id} reports {value:.1f} TFLOPS, above the "
            f"physical BF16 bound {bound_tflops:.0f} for this device",
            detail={"kind": "implausible_throughput", "case_id": case_id},
        )


def run_call_case(
    entrypoint: Callable[..., Any],
    raw_case: Mapping[str, Any],
    *,
    trials: int,
    warmups: int,
    sentinel: Any,
    bound_tflops: float | None,
) -> dict[str, Any]:
    """One scored single call: warm-up, then per trial a precall and a measured call."""
    case = validate_scored_case(raw_case)
    if case["family"] not in CALL_FAMILIES:
        raise ValueError("run_call_case needs a decode, spec or long case")
    if isinstance(trials, bool) or not isinstance(trials, int) or trials <= 0:
        raise ValueError("trials must be a positive integer")
    if isinstance(warmups, bool) or not isinstance(warmups, int) or warmups <= 0:
        raise ValueError("warmups must be a positive integer")
    seed = int(case["seed"])

    warm = prepare_call(case, precall_lengths(case, derive_seed(seed, "warmup-lengths", 0)),
                        derive_seed(seed, "warmup", 0))
    for _ in range(warmups):
        poison_output(warm)
        invoke(entrypoint, warm)
        sentinel.check("warm-up call")
    _cuda_synchronize()
    check_memory(warm, "warm-up")
    metrics: dict[str, float] = {}
    _merge_metrics(metrics, check_output(warm, reference_outputs(warm), CorrectnessFailure))
    del warm
    torch.cuda.empty_cache()

    trial_us: list[float] = []
    trial_host_us: list[float] = []
    trial_event_us: list[float] = []
    trial_tflops: list[float] = []
    trial_pairs: list[int] = []
    for trial in range(trials):
        # Untimed pre-call: same batch, heads, page size, table width and pool;
        # different lengths, data and tensors.  It absorbs compilation and
        # autotuning and is released before the measured tensors exist.
        precall = prepare_call(case, precall_lengths(case, derive_seed(seed, "precall-lengths", trial)),
                             derive_seed(seed, "precall", trial))
        invoke(entrypoint, precall)
        _cuda_synchronize()
        sentinel.check("pre-call")
        check_memory(precall, "pre-call")
        del precall
        torch.cuda.empty_cache()
        prepared = prepare_call(case, case["lengths"], derive_seed(seed, "measured", trial))
        measured, host_wall, event_seconds = _measure_once(entrypoint, prepared)
        sentinel.check("measured call")
        check_memory(prepared, "measured call")
        _merge_metrics(metrics, check_output(prepared, reference_outputs(prepared), CorrectnessFailure))
        value = effective_tflops(prepared.total_pairs, prepared.h_q, measured)
        trial_us.append(measured * 1.0e6)
        trial_host_us.append(host_wall * 1.0e6)
        trial_event_us.append(event_seconds * 1.0e6)
        trial_tflops.append(value)
        trial_pairs.append(int(prepared.total_pairs))
        del prepared
        torch.cuda.empty_cache()

    case_tflops = finite_positive(_median(trial_tflops), "case median TFLOPS")
    _check_bound(str(case["case_id"]), case_tflops, bound_tflops)
    median_us = finite_positive(_median(trial_us), "case median us")
    kv_bytes = 4 * int(case["h_kv"]) * HEAD_DIM * sum(case["lengths"])
    return {
        "case_id": str(case["case_id"]),
        "family": str(case["family"]),
        "scored": True,
        "entrypoint": DECODE_ENTRYPOINT,
        "s_q": int(case["s_q"]),
        "b": int(case["b"]),
        "l_max": int(case["l_max"]),
        "h_q": int(case["h_q"]),
        "h_kv": int(case["h_kv"]),
        "page_size": int(case["page_size"]),
        "table_width": int(case["table_width"]),
        "pool_pages": int(case["pool_pages"]),
        "softmax_scale": SOFTMAX_SCALE,
        "trial_us": trial_us,
        "trial_host_us": trial_host_us,
        "trial_event_us": trial_event_us,
        "trial_tflops": trial_tflops,
        "trial_pairs": trial_pairs,
        "median_us": median_us,
        "tflops": case_tflops,
        "kv_gbps": kv_bytes / (median_us * 1.0e-6) / 1.0e9,
        "untimed_precalls": trials,
        **{key: float(value) for key, value in metrics.items()},
    }


def _stream_tables(case: Mapping[str, Any], plan: StreamPlan, generator: Any) -> Any:
    """[requests, width] int32 host table: every request's pages, reserved for its
    final length when it joins.  A fragmented free list (a random permutation of
    the pool) hands out pages; leaving requests return theirs, and later requests
    reuse them.  Shared blocks keep one set of pages while any user is live."""
    page_size, width, pool = int(case["page_size"]), int(case["table_width"]), int(case["pool_pages"])
    block_tokens = int(case["block_tokens"])
    per_block = block_tokens // page_size
    count = len(plan.starts)
    ends = [0] * count
    for step, now in enumerate(plan.active):
        for index in now:
            ends[index] = step + 1
    free = torch.randperm(pool, generator=generator)
    top = pool
    table = torch.randint(0, pool, (count, width), generator=generator, dtype=torch.int64)
    owned: dict[int, Any] = {}
    block_pages: dict[int, Any] = {}
    refs: dict[int, int] = {}
    joins: dict[int, list[int]] = {}
    leaves: dict[int, list[int]] = {}
    for index in range(count):
        joins.setdefault(plan.starts[index], []).append(index)
        leaves.setdefault(ends[index], []).append(index)

    def release(pages: Any) -> None:
        nonlocal top
        free[top : top + pages.numel()] = pages
        top += pages.numel()

    def take(number: int) -> Any:
        nonlocal top
        if number > top:
            raise RuntimeError("stream layout exceeds its page pool")
        top -= number
        return free[top : top + number].clone()

    for step in range(len(plan.active)):
        for index in leaves.get(step, []):
            release(owned.pop(index))
            for block in plan.blocks[index]:
                refs[block] -= 1
                if not refs[block]:
                    del refs[block]
                    release(block_pages.pop(block))
        for index in joins.get(step, []):
            parts = []
            for block in plan.blocks[index]:
                if block not in block_pages:
                    block_pages[block] = take(per_block)
                    refs[block] = 0
                refs[block] += 1
                parts.append(block_pages[block])
            private = take(blocks_for(plan.finals[index] - len(plan.blocks[index]) * block_tokens, page_size))
            owned[index] = private
            parts.append(private)
            pages = torch.cat(parts)
            table[index, : pages.numel()] = pages
    return table.to(torch.int32)


def run_stream_case(
    entrypoint: Callable[..., Any],
    raw_case: Mapping[str, Any],
    *,
    sentinel: Any,
    bound_tflops: float | None,
) -> dict[str, Any]:
    """A sequence of consecutive decode steps of one server.

    The page pool and every request's pages persist across steps; each step
    gathers the live requests' rows of a fixed-width table, so B and every
    length change between calls while the table width stays fixed.  After
    STREAM_WARMUPS untimed calls on nearby shapes, every step is timed on fresh
    queries and compilation during these calls counts.  The golden check runs
    every STREAM_CHECK_EVERY steps and on the last step."""
    case = validate_scored_case(raw_case)
    if case["family"] != "stream":
        raise ValueError("run_stream_case needs a stream case")
    seed = int(case["seed"])
    plan = stream_plan(case)
    s_q, h_q, h_kv = int(case["s_q"]), int(case["h_q"]), int(case["h_kv"])
    page_size, pool = int(case["page_size"]), int(case["pool_pages"])
    host_generator = torch.Generator()
    host_generator.manual_seed(derive_seed(seed, "layout", 0))
    table_all = _stream_tables(case, plan, host_generator).to("cuda")

    pool_arena = GuardedArena(True)
    k_cache = pool_arena.empty("k_cache", (pool, page_size, h_kv, HEAD_DIM), torch.bfloat16)
    v_cache = pool_arena.empty("v_cache", (pool, page_size, h_kv, HEAD_DIM), torch.bfloat16)
    device_generator = torch.Generator(device="cuda")
    device_generator.manual_seed(derive_seed(seed, "values", 0))
    k_cache.copy_(torch.randn(k_cache.shape, dtype=torch.bfloat16, device="cuda", generator=device_generator))
    v_cache.copy_(torch.randn(v_cache.shape, dtype=torch.bfloat16, device="cuda", generator=device_generator))
    outliers = case["values"] == "outlier"
    if outliers:
        _add_outliers(k_cache, device_generator)
        _add_outliers(v_cache, device_generator)
    pool_pristine = {"k_cache": k_cache.clone(), "v_cache": v_cache.clone()}

    def call(rows: list[int], lengths: list[int], call_seed: int) -> Invocation:
        arena = GuardedArena(True)
        b = len(rows)
        q = arena.empty("q", (b, s_q, h_q, HEAD_DIM), torch.bfloat16)
        page_table = arena.empty("page_table", (b, int(case["table_width"])), torch.int32, fill=SOFT_PAGE_ID)
        cache_seqlens = arena.empty("cache_seqlens", (b,), torch.int32, fill=s_q)
        out = arena.empty("out", (b, s_q, h_q, HEAD_DIM), torch.bfloat16)
        generator = torch.Generator(device="cuda")
        generator.manual_seed(call_seed)
        q.copy_(torch.randn(q.shape, dtype=torch.bfloat16, device="cuda", generator=generator))
        if outliers:
            _add_outliers(q, generator)
        page_table.copy_(table_all.index_select(0, torch.tensor(rows, dtype=torch.int64, device="cuda")))
        cache_seqlens.copy_(torch.tensor(lengths, dtype=torch.int32).to("cuda"))
        out.fill_(OUT_POISON)
        invocation = Invocation(
            case_id=str(case["case_id"]), s_q=s_q, b=b, l_max=max(lengths), h_q=h_q, h_kv=h_kv,
            page_size=page_size, lengths=lengths, arena=arena, q=q, k_cache=k_cache,
            v_cache=v_cache, page_table=page_table, cache_seqlens=cache_seqlens, out=out,
            total_pairs=pairs_for_lengths(s_q, lengths),
        )
        invocation.pristine = {"q": q.clone(), "page_table": page_table.clone(),
                               "cache_seqlens": cache_seqlens.clone()}
        return invocation

    def check_step(invocation: Invocation, where: str, full: bool) -> None:
        problems = invocation.arena.violated() + (pool_arena.violated() if full else [])
        if problems:
            raise MemorySafetyFailure(
                f"out-of-bounds write detected ({where}): {problems[:4]}",
                detail={"kind": "guard_band", "where": where, "regions": problems[:8]},
            )
        changed = [name for name, saved in invocation.pristine.items()
                   if not _torch_equal(invocation.inputs[name].reshape(-1).view(torch.uint8),
                                       saved.reshape(-1).view(torch.uint8))]
        if full:
            changed += [name for name, saved in pool_pristine.items()
                        if not _torch_equal(invocation.inputs[name].reshape(-1).view(torch.uint8),
                                            saved.reshape(-1).view(torch.uint8))]
        if changed:
            raise MemorySafetyFailure(
                f"candidate modified its inputs ({where}): {changed}",
                detail={"kind": "input_modified", "where": where, "inputs": changed},
            )

    # Untimed warm-up on nearby shapes: half keep the first step's batch size
    # modulo 16, half shift it; lengths and rows are drawn from the stream.
    draw = random.Random(derive_seed(seed, "stream-warmup", 0))
    first = len(plan.active[0])
    count = len(plan.starts)
    for ordinal in range(STREAM_WARMUPS):
        shift = 16 * draw.randint(-1, 1) if ordinal % 2 == 0 else draw.choice((-3, -2, -1, 1, 2, 3))
        b = min(MAX_B, max(1, first + shift))
        rows = [draw.randrange(count) for _ in range(b)]
        lengths = [draw.randint(s_q, plan.finals[row]) for row in rows]
        invocation = call(rows, lengths, derive_seed(seed, "stream-warmup-q", ordinal))
        invoke(entrypoint, invocation)
        _cuda_synchronize()
        sentinel.check("stream warm-up call")
        check_step(invocation, "stream warm-up call", ordinal == 0)
        del invocation

    step_us: list[float] = []
    metrics: dict[str, float] = {}
    total_seconds = 0.0
    total_pairs = 0
    checks = 0
    last = len(plan.active) - 1
    for step, rows in enumerate(plan.active):
        lengths = [plan.inputs[row] + step - plan.starts[row] for row in rows]
        invocation = call(rows, lengths, derive_seed(seed, "stream-q", step))
        measured, _host, _event = _measure_once(entrypoint, invocation)
        sentinel.check("stream call")
        full = step % STREAM_CHECK_EVERY == 0 or step == last
        check_step(invocation, "stream call", full)
        if full:
            _merge_metrics(metrics, check_output(invocation, reference_outputs(invocation), CorrectnessFailure))
            checks += 1
        total_seconds += measured
        total_pairs += invocation.total_pairs
        step_us.append(measured * 1.0e6)
        del invocation
        if total_seconds > STREAM_TIME_LIMIT_SECONDS:
            raise CorrectnessFailure(
                f"sequence {case['case_id']} exceeded its {STREAM_TIME_LIMIT_SECONDS:.0f} s "
                f"candidate-time limit after {step + 1} of {len(plan.active)} steps",
                detail={"kind": "stream_time_limit", "steps_done": step + 1},
            )
    if total_pairs != int(case["total_pairs"]):
        raise RuntimeError("stream pair accounting differs from the plan")
    value = effective_tflops(total_pairs, h_q, total_seconds)
    _check_bound(str(case["case_id"]), value, bound_tflops)
    del table_all, pool_pristine
    return {
        "case_id": str(case["case_id"]),
        "family": "stream",
        "scored": True,
        "entrypoint": DECODE_ENTRYPOINT,
        "s_q": s_q,
        "b": int(case["b"]),
        "l_max": int(case["l_max"]),
        "h_q": h_q,
        "h_kv": h_kv,
        "page_size": page_size,
        "table_width": int(case["table_width"]),
        "pool_pages": pool,
        "softmax_scale": SOFTMAX_SCALE,
        "steps": len(plan.active),
        "step_us": step_us,
        "total_pairs": total_pairs,
        "seconds": total_seconds,
        "tflops": value,
        "max_step_us": max(step_us),
        "median_step_us": _median(step_us),
        "warmup_calls": STREAM_WARMUPS,
        "golden_checks": checks,
        **{key: float(item) for key, item in metrics.items()},
    }


def run_scored_case(
    entrypoint: Callable[..., Any],
    raw_case: Mapping[str, Any],
    *,
    trials: int,
    warmups: int,
    sentinel: Any,
    bound_tflops: float | None,
) -> dict[str, Any]:
    if raw_case.get("family") == "stream":
        return run_stream_case(entrypoint, raw_case, sentinel=sentinel, bound_tflops=bound_tflops)
    return run_call_case(entrypoint, raw_case, trials=trials, warmups=warmups,
                         sentinel=sentinel, bound_tflops=bound_tflops)


def run_edge_case(
    entrypoint: Callable[..., Any],
    raw_case: Mapping[str, Any],
    *,
    sentinel: Any,
    guards: bool = True,
    calls: int = 2,
) -> dict[str, Any]:
    """Untimed calls on one fresh invocation; raise on any failure."""
    case = validate_edge_case(raw_case)
    prepared = prepare_invocation(case, int(case["seed"]), guards=guards, pristine=True)
    reference: Reference | None = None
    metrics: dict[str, float] = {}
    for attempt in range(calls):
        if attempt:
            poison_output(prepared)
        invoke(entrypoint, prepared)
        _cuda_synchronize()
        sentinel.check(f"edge call {attempt + 1}")
        check_memory(prepared, f"edge call {attempt + 1}")
        if reference is None:
            reference = reference_outputs(prepared)
        _merge_metrics(metrics, check_output(prepared, reference, EdgeCorrectnessFailure))
    del prepared, reference
    return {
        "case_id": str(case["case_id"]),
        "category": str(case["category"]),
        "ok": True,
        **{key: float(value) for key, value in metrics.items()},
    }


def aggregate_case_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Geometric mean inside each family, then over the families present."""
    scored = [result for result in results if bool(result.get("scored"))]
    if not scored:
        raise ValueError("cannot aggregate without at least one scored case")
    family_values: dict[str, list[float]] = {}
    for result in scored:
        value = finite_positive(result.get("tflops"), "case TFLOPS")
        family = str(result.get("family"))
        if family not in FAMILIES:
            raise ValueError("case result has an invalid family")
        family_values.setdefault(family, []).append(value)
    family_tflops = {name: geometric_mean(family_values[name]) for name in FAMILIES if name in family_values}
    return {
        "family_tflops": family_tflops,
        "primary_tflops": geometric_mean(family_tflops.values()),
    }


GATE_ORDER = (
    "edge_correctness_ok",
    "memory_safety_ok",
    "correctness_ok",
)


def run_suite(
    candidate_path: Path,
    scored_cases: Sequence[Mapping[str, Any]],
    edge_cases: Sequence[Mapping[str, Any]] = (),
    *,
    trials: int,
    warmups: int,
    require_hardware: bool = True,
    stop_on_edge_failure: bool = True,
) -> dict[str, Any]:
    """Import one candidate, run the native edge cases, then the scored cases.

    Returns a schema-4 result.  Gate failures do not raise: the result carries
    ``status = "failed"``, the failing gate and a private message.  A CUDA
    fault stops the run at once, so a faulting kernel is never called again.
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
    bound = None
    sentinel = _NoChecks()
    gates: dict[str, bool | None] = {name: None for name in GATE_ORDER}
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generator_provenance": GENERATOR_PROVENANCE,
        "entrypoints": list(ENTRYPOINTS),
        "hardware": hardware,
        "trials_per_case": trials,
        "warmups": warmups,
        "protocol": {
            "untimed_precall": True,
            "stream_warmups": STREAM_WARMUPS,
            "stream_check_every": STREAM_CHECK_EVERY,
            "stream_time_limit_seconds": STREAM_TIME_LIMIT_SECONDS,
            "edge_calls": 2,
            "guard_bytes": GUARD_BYTES,
            "soft_landing": {"page_table": SOFT_PAGE_ID, "cache_seqlens": "s_q"},
            "max_abs_mult": MAX_ABS_MULT,
            "mean_abs_mult": MEAN_ABS_MULT,
            "max_abs_floor": MAX_ABS_FLOOR,
            "norm_l2_tol": NORM_L2_TOL,
        },
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
        payload["sentinel_checks"] = sentinel.checks
        return payload

    try:
        _module, entrypoint, source_sha256 = load_candidate(Path(candidate_path))
    except Exception as exc:
        payload["candidate_sha256"] = None
        return fail("correctness_ok", f"candidate import failed: {type(exc).__name__}: {exc}", {"kind": "import"})
    payload["candidate_sha256"] = source_sha256
    try:
        sentinel.check("candidate import")
    except GateFailure as exc:
        return fail(exc.gate, str(exc), exc.detail)

    edge_failures: list[dict[str, Any]] = []
    aborted = False
    for case in native_edges:
        try:
            row = run_edge_case(entrypoint, case, sentinel=sentinel)
            payload["edge"]["results"].append(row)
            payload["edge"]["cases_passed"] += 1
        except Exception as exc:  # noqa: BLE001 - classified below
            gate = classify_exception(exc)
            if gate == "correctness_ok":
                gate = "edge_correctness_ok"
            row = {
                "case_id": str(case["case_id"]),
                "category": str(case["category"]),
                "ok": False,
                "gate": gate,
                "error": f"{type(exc).__name__}: {str(exc)[:400]}",
                "detail": dict(getattr(exc, "detail", {}) or {}),
            }
            payload["edge"]["results"].append(row)
            payload["edge"]["cases_failed"] += 1
            edge_failures.append(row)
            if is_cuda_fault(exc) or gate == "harness_integrity_ok":
                aborted = True
                break
        torch.cuda.empty_cache()
    pending_edge_failure: tuple[str, str, dict[str, Any]] | None = None
    if edge_failures:
        # Memory safety and harness integrity outrank plain wrong answers.
        gate = "edge_correctness_ok"
        for row in edge_failures:
            if row["gate"] in ("harness_integrity_ok", "memory_safety_ok"):
                gate = row["gate"]
                break
        message = (
            f"{len(edge_failures)} of {len(native_edges)} edge cases failed; first: "
            f"{edge_failures[0]['case_id']}: {edge_failures[0]['error']}"
        )
        pending_edge_failure = (gate, message, {"kind": "edge", "failed": len(edge_failures)})
        if stop_on_edge_failure or aborted:
            return fail(*pending_edge_failure)
    elif native_edges:
        gates["edge_correctness_ok"] = True

    results: list[dict[str, Any]] = []
    for case in scored:
        try:
            results.append(
                run_scored_case(
                    entrypoint, case, trials=trials, warmups=warmups,
                    sentinel=sentinel, bound_tflops=bound,
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
    payload["sentinel_checks"] = sentinel.checks
    return payload


def run_sanitize_cases(
    candidate_path: Path,
    edge_cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Plain allocations (no guard bands) for compute-sanitizer; one call per case.

    Runs every case flagged ``sanitize``.  The Judge runs this pass before any
    native call, so an out-of-bounds kernel is stopped by memcheck before it
    reaches the hardware.  A wrong answer is reported against its own gate.
    """
    cases = [validate_edge_case(case) for case in edge_cases]
    cases = [case for case in cases if case["sanitize"]]
    use_exact_fp32()
    sentinel = _NoChecks()
    rows: list[dict[str, Any]] = []
    payload: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "candidate_sha256": None, "cases": rows}
    try:
        _module, entrypoint, source_sha256 = load_candidate(Path(candidate_path))
    except Exception as exc:
        return payload | {"status": "failed", "failed_gate": "correctness_ok",
                          "private_error": f"candidate import failed: {type(exc).__name__}: {exc}"}
    payload["candidate_sha256"] = source_sha256
    for case in cases:
        try:
            row = run_edge_case(entrypoint, case, sentinel=sentinel, guards=False, calls=1)
        except Exception as exc:  # noqa: BLE001 - classified below
            gate = classify_exception(exc)
            if gate == "correctness_ok":
                gate = "edge_correctness_ok"
            return payload | {
                "status": "failed",
                "failed_gate": gate,
                "private_error": f"sanitizer case {case['case_id']}: {type(exc).__name__}: {str(exc)[:400]}",
            }
        rows.append(row)
        torch.cuda.empty_cache()
    return payload | {"status": "passed"}
