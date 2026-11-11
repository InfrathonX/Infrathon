#!/usr/bin/env python3
"""Private verifier for standalone BF16 paged GQA decode (scored suite v4).

Order of evaluation: protected hashes, evaluation profile, the two root-only
manifests (scored serving cases and edge cases), source admission, hardware, the
compute-sanitizer memcheck subset in a separate unprivileged process (first,
so an out-of-bounds kernel never reaches the hardware natively), and then the
unprivileged worker (native edge cases, then the scored cases).  Every
aggregate the worker reports is recomputed here from its trial records, and
every case's work (counted query-key pairs) is recomputed from the manifest,
including the full request schedule of every sequence case.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pwd
import re
import shutil
import signal
import stat
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


sys.path.insert(0, str(Path(__file__).resolve().parent))
from source_admission import validate_submission  # noqa: E402


JUDGE_ROOT = Path("/opt/standalone_paged_gqa_decode_judge")
RUNTIME_ROOT = Path("/opt/standalone_paged_gqa_decode_worker")
WORKER = JUDGE_ROOT / "candidate_worker.py"
SANITIZER = Path("/usr/local/cuda/compute-sanitizer/compute-sanitizer")
EVALUATION_CONTEXT = Path("/tmp/.sforge-evaluation-context.json")
FINAL_MANIFEST = JUDGE_ROOT / "private/final_cases.json"
EDGE_MANIFEST = JUDGE_ROOT / "private/edge_cases.json"
RESULT_NAME = "standalone_paged_gqa_decode_v4_results.json"
FINAL_SUITE_ID = "final-holdout-v4"
EDGE_SUITE_ID = "edge-holdout-v4"
EVALUATION_POLICY = "full-final-shared-v4"
SELECTION_PROTOCOL_ID = "paged-gqa-shared-final-v4"
GENERATOR_PROVENANCE = "flash-attention-0251105a-derived-paged-gqa-decode-v2-judge-port"
SCHEMA_VERSION = 4
PROFILE_CODES = {"auto": 1.0, "final": 2.0}
DECODE_ENTRYPOINT = "paged_gqa_decode_bf16"
CASE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,95}$")
CATEGORY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
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
SANITIZER_EXIT_CODE = 86
# compute-sanitizer prints this when it cannot attach (for example under a GPU
# virtualization hook); the target then runs uninstrumented, which proves nothing.
SANITIZER_UNAVAILABLE_MARKER = "No attachable process found"
# The deployment, never the candidate, selects the sanitizer mode (Judge image ENV
# or the controller's Judge environment). "required" is the task contract;
# "skip-platform-incompatible" exists for platforms whose GPU virtualization hook
# stops compute-sanitizer from attaching to any CUDA process (PyroMind's HAMi
# libvgpu.so, 2026-10-05 and 2026-10-06); results then record that the pass did
# not run.
SANITIZER_MODES = ("required", "skip-platform-incompatible")
SANITIZER_ARGUMENTS = (
    "--tool=memcheck",
    f"--error-exitcode={SANITIZER_EXIT_CODE}",
    "--target-processes=application-only",
    "--destroy-on-device-error=context",
    "--force-blocking-launches",
    "--check-api-memory-access=no",
    "--print-limit=20",
    "--kernel-name-exclude",
    "kns=nvjet",
)

# The disclosed workload grid and support domain.  These tables are duplicated
# from the trusted runtime on purpose: the Judge must not depend on an
# importable copy of it, and task-local tests assert the two agree.
HEAD_DIM = 128
SOFTMAX_SCALE = 1.0 / math.sqrt(float(HEAD_DIM))
FAMILIES = ("decode", "spec", "long", "stream")
CALL_FAMILIES = ("decode", "spec", "long")
BLOCK_SIZES = (16, 32, 64, 128, 256, 512)
SCORED_FILLS = ("randn", "nan")
SCORED_VALUES = ("randn", "outlier")
OUTLIER_PROBABILITY = 0.001
OUTLIER_SCALE = 10.0
MAX_STREAM_STEPS = 1024
MAX_STREAM_REQUESTS = 4096
STREAM_WARMUPS = 16
STREAM_CHECK_EVERY = 64
STREAM_TIME_LIMIT_SECONDS = 120.0
MAX_SCORED_POOL_PAGES = 1 << 23
MAX_SCORED_KV_BYTES = 12 * (1 << 30)
MAX_STREAM_TABLE_ELEMENTS = 1 << 28
MAX_SCORED_WIDTH_FACTOR = 4
MAX_B = 256
MAX_S_Q = 16
MAX_H_Q = 128
MAX_L_MAX = 1 << 20
PAGE_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
TABLE_MODES = ("perm", "identity", "reversed", "random", "shared", "tail_perm")
FILLS = ("randn", "nan", "inf")
VALUES = ("randn", "large_logits", "negative_logits")
MAX_POOL_PAGES = 1 << 18
MAX_WIDTH_FACTOR = 4
MAX_OUTPUT_ELEMENTS = 1 << 26
MAX_CASE_KV_BYTES = 2_684_354_560
MAX_SANITIZE_ONLY_KV_BYTES = 10 * (1 << 30)
MAX_SEED = (1 << 63) - 1
# A case that never runs natively exists only to exercise 64-bit cache offsets.
LARGE_OFFSET_ELEMENTS = 1 << 31

# Correctness and memory-safety protocol; keep identical to the runtime.
MAX_ABS_MULT = 2.0
MEAN_ABS_MULT = 1.5
MAX_ABS_FLOOR = 1.0e-5
NORM_L2_TOL = 5.0e-3
GUARD_BYTES = 1 << 20
SOFT_LANDING = {"page_table": 0, "cache_seqlens": "s_q"}
EDGE_CALLS = 2
ACCURACY_FIELDS = (
    "out_max_abs",
    "out_mean_abs",
    "out_norm_l2",
    "out_max_abs_limit",
    "out_mean_abs_limit",
)

FINAL_CASE_COUNT = 1114
FINAL_SEED_MINIMUM = 1 << 32  # every public seed is below this
# The frozen private edge instance; keep identical to
# scripts/generate_edge_manifest.py and the frozen judge/private/edge_cases.json.
EDGE_CASE_COUNT = 168
EDGE_SANITIZE_COUNT = 12
EDGE_SANITIZE_ONLY_COUNT = 1
EXPECTED_TRIALS = 3
EXPECTED_WARMUPS = 1


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _blocks_for(length: int, page_size: int) -> int:
    return -(-int(length) // int(page_size))


def _kv_cache_bytes(pool_pages: int, page_size: int, h_kv: int) -> int:
    return 4 * int(pool_pages) * int(page_size) * int(h_kv) * HEAD_DIM


# Hazards the private instance must keep exercising (plan section 3.4).  A
# regenerated manifest that silently drops one fails suite_ok.
EDGE_REQUIRED_COVERAGE: dict[str, Callable[[Mapping[str, Any]], bool]] = {
    "batch of one": lambda case: case["b"] == 1,
    "non-power-of-two S_q": lambda case: not _is_power_of_two(case["s_q"]),
    "non-power-of-two GQA group": lambda case: not _is_power_of_two(case["h_q"] // case["h_kv"]),
    "page size 1": lambda case: case["page_size"] == 1,
    "page size 256": lambda case: case["page_size"] == 256,
    "pool larger than B x width with random page ids": lambda case: (
        case["table_mode"] == "random" and case["pool_pages"] > case["b"] * case["table_width"]
    ),
    "shared prefix pages": lambda case: case["table_mode"] == "shared",
    "page table wider than needed": lambda case: (
        case["table_width"] > _blocks_for(case["l_max"], case["page_size"])
    ),
    "NaN in hidden cache slots": lambda case: case["fill"] == "nan",
    "extreme negative logits": lambda case: case["values"] == "negative_logits",
    "context above 2^17 tokens": lambda case: case["l_max"] > (1 << 17),
}

GATE_NAMES = (
    "profile_ok",
    "suite_ok",
    "source_ok",
    "hardware_ok",
    "edge_correctness_ok",
    "correctness_ok",
    "memory_safety_ok",
    "harness_integrity_ok",
)
RUNTIME_GATES = (
    "edge_correctness_ok",
    "correctness_ok",
    "memory_safety_ok",
    "harness_integrity_ok",
)

# Physical plausibility bounds; keep identical to the runtime (tests assert).
BF16_DENSE_PEAK_BOUND_TFLOPS: dict[tuple[int, int], float] = {
    (8, 0): 400.0,
    (8, 6): 250.0,
    (8, 9): 450.0,
    (9, 0): 1_200.0,
    (10, 0): 3_000.0,
    (10, 3): 3_000.0,
    (12, 0): 1_200.0,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise RuntimeError(f"{label} must be finite and positive")
    return number


def geometric_mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("geometric mean requires values")
    numbers = [finite_positive(value, "geometric mean input") for value in values]
    return math.exp(math.fsum(math.log(value) for value in numbers) / len(numbers))


def _strict_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(
            raw.decode("utf-8", errors="strict"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is malformed") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return parsed


def _read_root_only_json(
    path: Path,
    label: str,
    *,
    max_bytes: int,
    expected_uid: int = 0,
    exact_mode: int | None = None,
) -> tuple[dict[str, Any], bytes]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RuntimeError(f"{label} is unavailable") from exc
    permissions = stat.S_IMODE(metadata.st_mode)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != expected_uid
        or permissions & 0o077
        or not permissions & 0o400
        or (exact_mode is not None and permissions != exact_mode)
        or metadata.st_size <= 0
        or metadata.st_size > max_bytes
    ):
        raise RuntimeError(f"{label} does not satisfy the root-only file contract")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"{label} is unreadable") from exc
    if len(raw) != metadata.st_size:
        raise RuntimeError(f"{label} changed while being read")
    return _strict_json_bytes(raw, label), raw


def load_evaluation_profile(
    path: Path = EVALUATION_CONTEXT,
    *,
    expected_uid: int = 0,
) -> tuple[str, str, str, int | None]:
    context, _raw = _read_root_only_json(
        path,
        "SForge evaluation context",
        max_bytes=16 * 1024,
        expected_uid=expected_uid,
        exact_mode=0o600,
    )
    round_id = context.get("round_id")
    if not isinstance(round_id, str) or not round_id:
        raise RuntimeError("SForge evaluation context lacks round_id")
    # Host-side per-version replay may override the submission's own
    # FINAL_VERSION so one archive can be scored per retained implementation.
    # The context file is root-only and written by the trusted host, and the
    # override is refused on interactive rounds so an agent-visible result can
    # never come from a version other than the one its submission selected.
    version_override = context.get("candidate_version")
    if version_override is not None and (
        isinstance(version_override, bool)
        or not isinstance(version_override, int)
        or not 1 <= version_override <= 999
    ):
        raise RuntimeError("SForge evaluation candidate_version is invalid")
    # Every admitted round uses the complete frozen final suite and the
    # private edge suite.  Round identity still determines candidate-override
    # authorization; do not rewrite agent-/auto- as a host-side replay.
    for interactive in ("auto-", "agent-"):
        if round_id.startswith(interactive) and len(round_id) > len(interactive):
            if version_override is not None:
                raise RuntimeError(
                    "candidate_version override is host-side only"
                )
            return "final", FINAL_SUITE_ID, round_id, None
    # Only trusted host-side evaluations may override FINAL_VERSION.
    for host_side in ("manual-", "candidate-"):
        if round_id.startswith(host_side) and len(round_id) > len(host_side):
            return "final", FINAL_SUITE_ID, round_id, version_override
    raise RuntimeError("SForge evaluation round_id has an unknown profile")


def _integer(
    value: Any, label: str, *, minimum: int = 1, maximum: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RuntimeError(f"{label} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise RuntimeError(f"{label} must be <= {maximum}")
    return value


def _heads_and_page(case: Mapping[str, Any], case_id: str) -> tuple[int, int, int, int]:
    s_q = _integer(case.get("s_q"), f"{case_id}.s_q", maximum=MAX_S_Q)
    h_q = _integer(case.get("h_q"), f"{case_id}.h_q", maximum=MAX_H_Q)
    h_kv = _integer(case.get("h_kv"), f"{case_id}.h_kv", maximum=h_q)
    if h_q % h_kv:
        raise RuntimeError(f"{case_id}: H_q must be divisible by H_kv")
    page_size = _integer(case.get("page_size"), f"{case_id}.page_size")
    if page_size not in PAGE_SIZES:
        raise RuntimeError(f"{case_id}: page_size outside the domain")
    block_tokens = _integer(case.get("block_tokens"), f"{case_id}.block_tokens")
    if block_tokens not in BLOCK_SIZES or block_tokens % page_size:
        raise RuntimeError(f"{case_id}: block_tokens outside the domain")
    if case.get("values") not in SCORED_VALUES:
        raise RuntimeError(f"{case_id}: values outside the domain")
    return s_q, h_q, h_kv, page_size


def _block_ids(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value
    ) or len(set(value)) != len(value):
        raise RuntimeError(f"{label} must be distinct non-negative integers")
    return list(value)


def _pairs(s_q: int, lengths: Sequence[int]) -> int:
    return sum(s_q * int(length) - (s_q * (s_q - 1)) // 2 for length in lengths)


def _validate_call(raw_case: Mapping[str, Any], case_id: str) -> dict[str, Any]:
    if set(raw_case) != CALL_KEYS:
        raise RuntimeError(f"final case {case_id} fields differ from schema v4")
    family = raw_case.get("family")
    if family not in CALL_FAMILIES:
        raise RuntimeError(f"final case {case_id} family is invalid")
    s_q, h_q, h_kv, page_size = _heads_and_page(raw_case, case_id)
    if (family == "spec") != (s_q > 1):
        raise RuntimeError(f"final case {case_id}: S_q does not match its family")
    lengths = raw_case.get("lengths")
    if (
        not isinstance(lengths, list)
        or not 1 <= len(lengths) <= MAX_B
        or any(isinstance(value, bool) or not isinstance(value, int) for value in lengths)
        or any(not s_q <= value <= MAX_L_MAX for value in lengths)
    ):
        raise RuntimeError(f"final case {case_id}: lengths are outside the domain")
    blocks = raw_case.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != len(lengths):
        raise RuntimeError(f"final case {case_id}: blocks must hold one list per request")
    block_tokens = int(raw_case["block_tokens"])
    distinct: set[int] = set()
    private = 0
    for length, ids in zip(lengths, blocks):
        ids = _block_ids(ids, f"{case_id}.blocks")
        if len(ids) * block_tokens > length:
            raise RuntimeError(f"final case {case_id}: shared blocks exceed a request length")
        distinct.update(ids)
        private += _blocks_for(length - len(ids) * block_tokens, page_size)
    demand = len(distinct) * (block_tokens // page_size) + private
    l_max = max(lengths)
    needed = _blocks_for(l_max, page_size)
    _integer(raw_case.get("table_width"), f"{case_id}.table_width", minimum=needed,
             maximum=MAX_SCORED_WIDTH_FACTOR * needed)
    pool = _integer(raw_case.get("pool_pages"), f"{case_id}.pool_pages", minimum=demand,
                    maximum=min(MAX_SCORED_POOL_PAGES, 2 * demand + 4096))
    if _kv_cache_bytes(pool, page_size, h_kv) > MAX_SCORED_KV_BYTES:
        raise RuntimeError(f"final case {case_id} exceeds the scored allocation cap")
    if len(lengths) * s_q * h_q * HEAD_DIM > MAX_OUTPUT_ELEMENTS:
        raise RuntimeError(f"final case {case_id} output exceeds the size bound")
    if raw_case.get("fill") not in SCORED_FILLS:
        raise RuntimeError(f"final case {case_id}: fill outside the domain")
    case = dict(raw_case)
    case.update(b=len(lengths), l_max=l_max, total_pairs=_pairs(s_q, lengths))
    return case


def _validate_stream(raw_case: Mapping[str, Any], case_id: str) -> dict[str, Any]:
    """Expand the request schedule independently of the runtime."""
    if set(raw_case) != STREAM_KEYS or raw_case.get("family") != "stream":
        raise RuntimeError(f"final case {case_id} fields differ from schema v4")
    s_q, h_q, h_kv, page_size = _heads_and_page(raw_case, case_id)
    steps = _integer(raw_case.get("steps"), f"{case_id}.steps", maximum=MAX_STREAM_STEPS)
    requests = raw_case.get("requests")
    if not isinstance(requests, list) or not 1 <= len(requests) <= MAX_STREAM_REQUESTS:
        raise RuntimeError(f"final case {case_id}: requests are invalid")
    block_tokens = int(raw_case["block_tokens"])
    per_block = block_tokens // page_size
    active = [0] * steps
    pairs = 0
    finals = []
    events: dict[int, list[tuple[int, int, list[int], int]]] = {}
    for raw in requests:
        if not isinstance(raw, list) or len(raw) != 4:
            raise RuntimeError(f"final case {case_id}: a request is [start, end, input, blocks]")
        start, end, length = raw[0], raw[1], raw[2]
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (start, end, length)):
            raise RuntimeError(f"final case {case_id}: request fields must be integers")
        if not 0 <= start < end <= steps or length < s_q:
            raise RuntimeError(f"final case {case_id}: request window or length is invalid")
        ids = _block_ids(raw[3], f"{case_id}.requests")
        final = length + end - start - 1
        if final > MAX_L_MAX or len(ids) * block_tokens > length:
            raise RuntimeError(f"final case {case_id}: request exceeds the length domain")
        finals.append(final)
        private = _blocks_for(final - len(ids) * block_tokens, page_size)
        events.setdefault(start, []).append((+1, private, ids, 0))
        events.setdefault(end, []).append((-1, private, ids, 0))
        for step in range(start, end):
            active[step] += 1
        # sum over steps of S_q * (length + step - start) - S_q(S_q-1)/2
        span = end - start
        pairs += s_q * (length * span + span * (span - 1) // 2) - span * ((s_q * (s_q - 1)) // 2)
    if any(not 1 <= count <= MAX_B for count in active):
        raise RuntimeError(f"final case {case_id}: every step needs 1 to {MAX_B} requests")
    refs: dict[int, int] = {}
    held = peak = 0
    for step in range(steps + 1):
        # departures first, then arrivals (the runtime's allocation order)
        for sign in (-1, +1):
            for kind, private, ids, _unused in events.get(step, []):
                if kind != sign:
                    continue
                held += sign * private
                for block in ids:
                    if sign > 0:
                        if block not in refs:
                            refs[block] = 0
                            held += per_block
                        refs[block] += 1
                    else:
                        refs[block] -= 1
                        if not refs[block]:
                            del refs[block]
                            held -= per_block
        peak = max(peak, held)
    l_max = max(finals)
    needed = _blocks_for(l_max, page_size)
    width = _integer(raw_case.get("table_width"), f"{case_id}.table_width", minimum=needed,
                     maximum=MAX_SCORED_WIDTH_FACTOR * needed)
    if len(requests) * width > MAX_STREAM_TABLE_ELEMENTS:
        raise RuntimeError(f"final case {case_id}: stream tables exceed the size bound")
    pool = _integer(raw_case.get("pool_pages"), f"{case_id}.pool_pages", minimum=peak,
                    maximum=min(MAX_SCORED_POOL_PAGES, 2 * peak + 4096))
    if _kv_cache_bytes(pool, page_size, h_kv) > MAX_SCORED_KV_BYTES:
        raise RuntimeError(f"final case {case_id} exceeds the scored allocation cap")
    case = dict(raw_case)
    case.update(b=max(active), l_max=l_max, total_pairs=pairs,
                golden_checks=sum(1 for step in range(steps)
                                  if step % STREAM_CHECK_EVERY == 0 or step == steps - 1))
    return case


def validate_final_manifest(payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """The frozen v4 serving suite: four families, every case re-derived."""
    if set(payload) != {"schema_version", "suite_id", "cases"}:
        raise RuntimeError("final manifest fields differ from schema v4")
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("suite_id") != FINAL_SUITE_ID:
        raise RuntimeError("final manifest identity differs")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != FINAL_CASE_COUNT:
        raise RuntimeError(f"final manifest must contain exactly {FINAL_CASE_COUNT} cases")
    cases: list[dict[str, Any]] = []
    ids: set[str] = set()
    seeds: set[int] = set()
    for raw_case in raw_cases:
        if not isinstance(raw_case, Mapping):
            raise RuntimeError("final case must be an object")
        case_id = raw_case.get("case_id")
        if not isinstance(case_id, str) or CASE_ID_RE.fullmatch(case_id) is None or case_id in ids:
            raise RuntimeError("final case_id is invalid or repeated")
        ids.add(case_id)
        seed = _integer(raw_case.get("seed"), f"{case_id}.seed", minimum=FINAL_SEED_MINIMUM, maximum=MAX_SEED)
        if seed in seeds:
            raise RuntimeError("final case seeds are not unique")
        seeds.add(seed)
        if raw_case.get("family") == "stream":
            cases.append(_validate_stream(raw_case, case_id))
        else:
            cases.append(_validate_call(raw_case, case_id))
    if {case["family"] for case in cases} != set(FAMILIES):
        raise RuntimeError("final cases must exercise all four families")
    return tuple(cases)


def validate_edge_case(raw_case: Any) -> dict[str, Any]:
    """The runtime's edge-case domain (``validate_edge_case``), checked independently."""
    if not isinstance(raw_case, Mapping) or set(raw_case) != EDGE_KEYS:
        raise RuntimeError("edge case fields differ from the edge schema")
    case = dict(raw_case)
    case_id = case.get("case_id")
    if not isinstance(case_id, str) or CASE_ID_RE.fullmatch(case_id) is None:
        raise RuntimeError("edge case_id is invalid")
    category = case.get("category")
    if not isinstance(category, str) or CATEGORY_RE.fullmatch(category) is None:
        raise RuntimeError(f"{case_id}: category is invalid")
    s_q = _integer(case.get("s_q"), f"{case_id}.s_q", maximum=MAX_S_Q)
    b = _integer(case.get("b"), f"{case_id}.b", maximum=MAX_B)
    h_q = _integer(case.get("h_q"), f"{case_id}.h_q", maximum=MAX_H_Q)
    h_kv = _integer(case.get("h_kv"), f"{case_id}.h_kv", maximum=h_q)
    if h_q % h_kv:
        raise RuntimeError(f"{case_id}: H_q must be divisible by H_kv")
    page_size = _integer(case.get("page_size"), f"{case_id}.page_size")
    if page_size not in PAGE_SIZES:
        raise RuntimeError(f"{case_id}: page_size outside the domain")
    l_max = _integer(case.get("l_max"), f"{case_id}.l_max", minimum=s_q, maximum=MAX_L_MAX)
    lengths = case.get("lengths")
    if (
        not isinstance(lengths, list)
        or len(lengths) != b
        or any(isinstance(value, bool) or not isinstance(value, int) for value in lengths)
        or any(not s_q <= value <= l_max for value in lengths)
    ):
        raise RuntimeError(f"{case_id}: lengths must be b integers in [s_q, l_max]")
    needed = _blocks_for(l_max, page_size)
    width = _integer(
        case.get("table_width"), f"{case_id}.table_width",
        minimum=needed, maximum=MAX_WIDTH_FACTOR * needed,
    )
    pool = _integer(case.get("pool_pages"), f"{case_id}.pool_pages", maximum=MAX_POOL_PAGES)
    if case.get("table_mode") not in TABLE_MODES:
        raise RuntimeError(f"{case_id}: table_mode outside the domain")
    if case["table_mode"] != "random" and pool < b * width:
        raise RuntimeError(f"{case_id}: a one-to-one table needs pool_pages >= b * table_width")
    if case.get("fill") not in FILLS or case.get("values") not in VALUES:
        raise RuntimeError(f"{case_id}: fill or values outside the domain")
    _integer(case.get("seed"), f"{case_id}.seed", minimum=0, maximum=MAX_SEED)
    for flag in ("sanitize", "native"):
        if not isinstance(case.get(flag), bool):
            raise RuntimeError(f"{case_id}: {flag} flag is invalid")
    if b * s_q * h_q * HEAD_DIM > MAX_OUTPUT_ELEMENTS:
        raise RuntimeError(f"{case_id}: output exceeds the edge-suite size bound")
    if case["native"]:
        if _kv_cache_bytes(pool, page_size, h_kv) > MAX_CASE_KV_BYTES:
            raise RuntimeError(f"{case_id}: KV pool exceeds the native size bound")
    else:
        # Only the large-offset case may skip native execution, and only
        # because its pool is too large to run natively on every Judge.
        if not case["sanitize"]:
            raise RuntimeError(f"{case_id}: a case that never runs natively must be a sanitizer case")
        if _kv_cache_bytes(pool, page_size, h_kv) > MAX_SANITIZE_ONLY_KV_BYTES:
            raise RuntimeError(f"{case_id}: KV pool exceeds the sanitizer-only size bound")
        if pool * page_size * h_kv * HEAD_DIM <= LARGE_OFFSET_ELEMENTS:
            raise RuntimeError(f"{case_id}: a sanitizer-only case must exceed 2^31 cache elements")
    return case


def validate_edge_manifest(payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    if set(payload) != {"schema_version", "suite_id", "cases"}:
        raise RuntimeError("edge manifest fields differ from the edge schema")
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("suite_id") != EDGE_SUITE_ID:
        raise RuntimeError("edge manifest identity differs")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != EDGE_CASE_COUNT:
        raise RuntimeError(f"edge manifest must contain exactly {EDGE_CASE_COUNT} cases")
    ids: set[str] = set()
    cases: list[dict[str, Any]] = []
    for raw_case in raw_cases:
        case = validate_edge_case(raw_case)
        if case["case_id"] in ids:
            raise RuntimeError("edge case IDs are not unique")
        ids.add(case["case_id"])
        cases.append(case)
    if sum(1 for case in cases if case["sanitize"]) != EDGE_SANITIZE_COUNT:
        raise RuntimeError(f"edge manifest must flag exactly {EDGE_SANITIZE_COUNT} sanitizer cases")
    if sum(1 for case in cases if not case["native"]) != EDGE_SANITIZE_ONLY_COUNT:
        raise RuntimeError(
            f"edge manifest must hold exactly {EDGE_SANITIZE_ONLY_COUNT} sanitizer-only case"
        )
    missing = [
        label
        for label, covered in EDGE_REQUIRED_COVERAGE.items()
        if not any(covered(case) for case in cases if case["native"])
    ]
    if missing:
        raise RuntimeError(f"edge manifest no longer exercises: {missing}")
    return tuple(cases)


def _load_manifest(
    path: Path, label: str, validator, *, expected_uid: int, max_bytes: int
) -> tuple[tuple[dict[str, Any], ...], str]:
    payload, raw = _read_root_only_json(path, label, max_bytes=max_bytes, expected_uid=expected_uid)
    return validator(payload), hashlib.sha256(raw).hexdigest()


def load_final_manifest(
    path: Path = FINAL_MANIFEST, *, expected_uid: int = 0
) -> tuple[tuple[dict[str, Any], ...], str]:
    return _load_manifest(
        path, "final holdout manifest", validate_final_manifest,
        expected_uid=expected_uid, max_bytes=32 * 1024 * 1024,
    )


def load_edge_manifest(
    path: Path = EDGE_MANIFEST, *, expected_uid: int = 0
) -> tuple[tuple[dict[str, Any], ...], str]:
    return _load_manifest(
        path, "edge holdout manifest", validate_edge_manifest,
        expected_uid=expected_uid, max_bytes=2 * 1024 * 1024,
    )


def locate_submission(app_dir: Path) -> Path:
    roots = (
        app_dir / "workdir/paged_gqa_decode_lab",
        app_dir / "environment/workdir/paged_gqa_decode_lab",
        app_dir,
    )
    matches: list[Path] = []
    for root in roots:
        try:
            root_mode = root.lstat().st_mode
        except OSError:
            continue
        if (
            stat.S_ISDIR(root_mode)
            and not stat.S_ISLNK(root_mode)
            and (root / "candidates").is_dir()
            and (root / "FINAL_VERSION").exists()
        ):
            absolute = root.absolute()
            if absolute not in matches:
                matches.append(absolute)
    if len(matches) != 1:
        raise RuntimeError(f"expected one submission root, found {len(matches)}")
    return matches[0]


def _protected_path(kind: str, relative: str) -> Path:
    if kind == "judge":
        return JUDGE_ROOT / relative
    if kind == "worker":
        return RUNTIME_ROOT / relative
    raise ValueError(f"unknown protected path kind {kind!r}")


def verify_protected_hashes(manifest: Path) -> list[str]:
    expected_labels = {
        "judge:candidate_worker.py",
        "judge:run_judge.py",
        "judge:source_admission.py",
        "judge:verify_paged_gqa.py",
        "worker:paged_gqa_decode_runtime.py",
    }
    problems: list[str] = []
    seen: set[str] = set()
    if not manifest.is_file() or manifest.is_symlink():
        return ["protected hash manifest is missing"]
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return ["protected hash manifest is unreadable"]
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            expected, label = line.split(None, 1)
            kind, relative = label.split(":", 1)
        except ValueError:
            problems.append("malformed protected hash record")
            continue
        if label in seen:
            problems.append(f"duplicate protected hash record: {label}")
            continue
        seen.add(label)
        if label not in expected_labels:
            problems.append(f"unexpected protected hash record: {label}")
            continue
        if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
            problems.append(f"invalid protected digest for {label}")
            continue
        path = _protected_path(kind, relative)
        if not path.is_file() or path.is_symlink():
            problems.append(f"missing protected file: {label}")
        elif sha256_file(path) != expected:
            problems.append(f"protected file changed: {label}")
    missing = sorted(expected_labels - seen)
    if missing:
        problems.append(f"protected hash records are missing: {missing}")
    return problems


def _agent_account() -> pwd.struct_passwd:
    return pwd.getpwnam("agent")


def _drop_to_agent() -> None:
    account = _agent_account()
    os.setgroups([])
    os.setgid(account.pw_gid)
    os.setuid(account.pw_uid)
    os.umask(0o077)


def _signal_session(session_id: int, requested_signal: signal.Signals) -> None:
    process_groups: set[int] = set()
    try:
        entries = tuple(Path("/proc").iterdir())
    except OSError:
        entries = ()
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            if os.getsid(pid) == session_id:
                process_groups.add(os.getpgid(pid))
        except (OSError, ProcessLookupError):
            continue
    for process_group in sorted(process_groups, reverse=True):
        try:
            os.killpg(process_group, requested_signal)
        except (OSError, ProcessLookupError):
            pass


def _terminate_session(process: subprocess.Popen[bytes]) -> None:
    _signal_session(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    _signal_session(process.pid, signal.SIGKILL)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def _candidate_environment(cache_dir: Path) -> dict[str, str]:
    allowed = (
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
        "NVIDIA_DRIVER_CAPABILITIES",
        "NVIDIA_VISIBLE_DEVICES",
        "PATH",
    )
    env = {key: os.environ[key] for key in allowed if key in os.environ}
    env.update(
        {
            "HOME": "/home/agent",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "TMPDIR": str(cache_dir),
            "TRITON_CACHE_DIR": str(cache_dir / "triton"),
        }
    )
    return env


def _strict_worker_result(path: Path, *, expected_uid: int) -> dict[str, Any]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RuntimeError("worker result file is unavailable") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != expected_uid
        or stat.S_IMODE(metadata.st_mode) & 0o077
        or metadata.st_size <= 0
        or metadata.st_size > 16 * 1024 * 1024
    ):
        raise RuntimeError("worker result file is invalid")
    return _strict_json_bytes(path.read_bytes(), "worker result")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


# Capability envelope: what the task contract actually needs, not the GPU model
# it was qualified on.  The GPU name and UUID are recorded as provenance only;
# they never gate admission.  Keep these two constants identical in the Judge
# verifier (tests assert the pair).
MIN_COMPUTE_CAPABILITY = (8, 0)
MIN_DEVICE_MEMORY_BYTES = 32 * 1024**3


def _trusted_hardware() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    count = int(torch.cuda.device_count())
    if count != 1:
        raise RuntimeError(f"exactly one visible GPU is required, found {count}")
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(0))
    name = str(torch.cuda.get_device_name(0))
    properties = torch.cuda.get_device_properties(0)
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
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
    }


def _prepare_run_dirs(run_root: Path, name: str) -> tuple[pwd.struct_passwd, Path, Path]:
    account = _agent_account()
    result_dir = run_root / f"{name}_result"
    result_dir.mkdir(mode=0o700)
    cache_dir = run_root / f"{name}_cache"
    cache_dir.mkdir(mode=0o700)
    (cache_dir / "triton").mkdir(mode=0o700)
    for path in (result_dir, cache_dir, cache_dir / "triton"):
        os.chown(path, account.pw_uid, account.pw_gid)
    return account, result_dir, cache_dir


class UnprivilegedTimeout(RuntimeError):
    """A trusted child process ran past its wall-clock budget and was killed."""


def _run_unprivileged(
    command: list[str],
    *,
    stdin_payload: bytes,
    run_root: Path,
    log_name: str,
    env: dict[str, str],
    timeout: float,
) -> int:
    log_path = run_root / log_name
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            cwd=run_root,
            env=env,
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            preexec_fn=_drop_to_agent,
        )
        try:
            process.communicate(input=stdin_payload, timeout=timeout)
            returncode = process.returncode
            _signal_session(process.pid, signal.SIGKILL)
        except subprocess.TimeoutExpired:
            _terminate_session(process)
            raise UnprivilegedTimeout(f"{log_name.split('.')[0]} timed out") from None
    return int(returncode)


def run_worker(
    candidate_path: Path,
    run_root: Path,
    cases: Sequence[Mapping[str, Any]],
    edge_cases: Sequence[Mapping[str, Any]],
    *,
    trials: int,
    warmups: int,
    timeout: float,
) -> dict[str, Any]:
    """Native run: every edge case (the runtime skips sanitizer-only ones), then the grid."""
    account, result_dir, cache_dir = _prepare_run_dirs(run_root, "worker")
    manifest_bytes = json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "mode": "suite",
            "suite_id": FINAL_SUITE_ID,
            "cases": [
                {key: case[key] for key in (STREAM_KEYS if case["family"] == "stream" else CALL_KEYS)}
                for case in cases
            ],
            "edge_suite_id": EDGE_SUITE_ID,
            "edge_cases": list(edge_cases),
        },
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    result_path = result_dir / "result.json"
    command = [
        sys.executable,
        str(WORKER),
        "--mode",
        "suite",
        "--candidate",
        str(candidate_path),
        "--output",
        str(result_path),
        "--trials",
        str(trials),
        "--warmups",
        str(warmups),
    ]
    returncode = _run_unprivileged(
        command,
        stdin_payload=manifest_bytes,
        run_root=run_root,
        log_name="candidate_worker.log",
        env=_candidate_environment(cache_dir),
        timeout=timeout,
    )
    if returncode != 0:
        raise RuntimeError("candidate worker exited without a structured result")
    return _strict_worker_result(result_path, expected_uid=account.pw_uid)


def run_sanitizer(
    candidate_path: Path,
    run_root: Path,
    sanitize_cases: Sequence[Mapping[str, Any]],
    *,
    timeout: float,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Run the flagged edge cases once each under compute-sanitizer memcheck."""
    if not SANITIZER.is_file():
        raise RuntimeError("compute-sanitizer is missing from the Judge image")
    account, result_dir, cache_dir = _prepare_run_dirs(run_root, "sanitizer")
    payload = json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "mode": "sanitize",
            "edge_suite_id": EDGE_SUITE_ID,
            "edge_cases": list(sanitize_cases),
        },
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    result_path = result_dir / "result.json"
    env = _candidate_environment(cache_dir)
    # Plain cudaMalloc per tensor, so memcheck sees each tensor's exact bounds.
    env["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    command = [
        str(SANITIZER),
        *SANITIZER_ARGUMENTS,
        sys.executable,
        str(WORKER),
        "--mode",
        "sanitize",
        "--candidate",
        str(candidate_path),
        "--output",
        str(result_path),
    ]
    try:
        returncode = _run_unprivileged(
            command,
            stdin_payload=payload,
            run_root=run_root,
            log_name="sanitizer.log",
            env=env,
            timeout=timeout,
        )
    except UnprivilegedTimeout:
        # A pass that cannot finish establishes nothing about memory safety.
        return {"returncode": None, "cases": len(sanitize_cases), "status": "timeout",
                "report": _sanitizer_report(run_root / "sanitizer.log")}
    record: dict[str, Any] = {"returncode": returncode, "cases": len(sanitize_cases)}
    report = _sanitizer_report(run_root / "sanitizer.log")
    try:
        raw_log = (run_root / "sanitizer.log").read_text(encoding="utf-8", errors="replace")
    except OSError:
        raw_log = ""
    if SANITIZER_UNAVAILABLE_MARKER in raw_log:
        # Required mode fails closed: an uninstrumented pass is not a memcheck.
        record.update(status="unavailable", report=report)
        return record
    # An invalid access can also surface as the application's own CUDA error
    # exit, so the report text decides as well as the exit code.
    invalid_access = any(
        marker in line
        for line in report
        for marker in ("Invalid __global__", "Invalid __shared__", "Invalid __local__", "Out-of-range")
    )
    if returncode == SANITIZER_EXIT_CODE or invalid_access:
        record.update(status="memory_error", report=report)
        return record
    if returncode != 0:
        record.update(status="failed", report=report)
        return record
    try:
        result = _strict_worker_result(result_path, expected_uid=account.pw_uid)
    except RuntimeError as exc:
        record.update(status="failed", failed_gate="memory_safety_ok", report=report,
                      private_error=f"sanitizer pass left no structured result: {exc}")
        return record
    record["result"] = result
    if result.get("status") != "passed":
        gate = result.get("failed_gate")
        record["status"] = "failed"
        record["failed_gate"] = gate if gate in RUNTIME_GATES else "memory_safety_ok"
        record["private_error"] = str(result.get("private_error") or "")
        return record
    try:
        _validate_sanitizer_result(
            result,
            sanitize_cases,
            expected_sha256 if expected_sha256 is not None else sha256_file(candidate_path),
        )
    except RuntimeError as exc:
        # A pass that does not account for every flagged case proves nothing.
        record["status"] = "failed"
        record["failed_gate"] = "memory_safety_ok"
        record["private_error"] = f"sanitizer result rejected: {exc}"
        return record
    record["status"] = "passed"
    return record


def _sanitizer_report(log_path: Path) -> list[str]:
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    keep = [line for line in lines if line.startswith("=========")]
    return keep[:40] if keep else lines[-20:]


def _validate_hardware_record(result: Mapping[str, Any]) -> None:
    hardware = result.get("hardware")
    if not isinstance(hardware, Mapping):
        raise RuntimeError("runtime result lacks hardware inventory")
    capability = hardware.get("capability")
    total_memory = hardware.get("total_memory_bytes")
    if (
        hardware.get("count") != 1
        or not isinstance(capability, list)
        or len(capability) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) for v in capability)
        or tuple(int(v) for v in capability) < MIN_COMPUTE_CAPABILITY
        or isinstance(total_memory, bool)
        or not isinstance(total_memory, int)
        or total_memory < MIN_DEVICE_MEMORY_BYTES
    ):
        raise RuntimeError("runtime hardware inventory violates the capability envelope")


def plausibility_bound(capability: Sequence[int] | None) -> float | None:
    if not capability or len(capability) != 2:
        return None
    return BF16_DENSE_PEAK_BOUND_TFLOPS.get((int(capability[0]), int(capability[1])))


def _exact_int(row: Mapping[str, Any], name: str, expected: int, label: str) -> None:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise RuntimeError(f"{label} differs for {name}")


def _error_metric(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise RuntimeError(f"{label} is outside the numerical contract")
    return number


def _validate_accuracy_metrics(row: Mapping[str, Any], label: str) -> None:
    """Re-check the differential and normalized-L2 gates from the reported metrics.

    The mean limit may be zero (an exact answer is then required); the max
    limit always carries the rounding floor.
    """
    max_abs = _error_metric(row.get("out_max_abs"), f"{label}.out_max_abs")
    mean_abs = _error_metric(row.get("out_mean_abs"), f"{label}.out_mean_abs")
    norm_l2 = _error_metric(row.get("out_norm_l2"), f"{label}.out_norm_l2")
    max_abs_limit = _error_metric(row.get("out_max_abs_limit"), f"{label}.out_max_abs_limit")
    mean_abs_limit = _error_metric(row.get("out_mean_abs_limit"), f"{label}.out_mean_abs_limit")
    if max_abs_limit < MAX_ABS_FLOOR:
        raise RuntimeError(f"{label}.out_max_abs_limit lacks the rounding floor")
    if not max_abs <= max_abs_limit:
        raise RuntimeError(f"{label}.out_max_abs exceeds its reported limit")
    if not mean_abs <= mean_abs_limit:
        raise RuntimeError(f"{label}.out_mean_abs exceeds its reported limit")
    if not norm_l2 <= NORM_L2_TOL:
        raise RuntimeError(f"{label}.out_norm_l2 exceeds {NORM_L2_TOL:g}")


def _numeric_list(value: Any, label: str, expected_length: int) -> list[float]:
    if not isinstance(value, list) or len(value) != expected_length:
        raise RuntimeError(f"{label} has the wrong length")
    return [finite_positive(item, label) for item in value]


def _validate_case_result(
    row: Mapping[str, Any],
    expected: Mapping[str, Any],
    expected_trials: int,
    bound: float | None,
) -> float:
    case_id = str(expected["case_id"])
    for name in ("case_id", "family"):
        if row.get(name) != expected[name]:
            raise RuntimeError(f"case identity differs for {case_id}")
    if row.get("scored") is not True:
        raise RuntimeError(f"case {case_id} is not marked scored")
    if row.get("entrypoint") != DECODE_ENTRYPOINT:
        raise RuntimeError(f"entrypoint identity differs for {case_id}")
    label = f"case {case_id} geometry"
    for name in ("s_q", "b", "l_max", "h_q", "h_kv", "page_size", "table_width", "pool_pages"):
        _exact_int(row, name, int(expected[name]), label)
    scale = row.get("softmax_scale")
    if isinstance(scale, bool) or not isinstance(scale, float) or scale != SOFTMAX_SCALE:
        raise RuntimeError(f"scored softmax scale differs for {case_id}")
    h_q = int(expected["h_q"])
    pairs = int(expected["total_pairs"])

    if expected["family"] == "stream":
        steps = int(expected["steps"])
        _exact_int(row, "steps", steps, label)
        _exact_int(row, "total_pairs", pairs, label)
        _exact_int(row, "warmup_calls", STREAM_WARMUPS, label)
        _exact_int(row, "golden_checks", int(expected["golden_checks"]), label)
        step_us = _numeric_list(row.get("step_us"), f"{case_id}.step_us", steps)
        seconds = math.fsum(step_us) * 1.0e-6
        if seconds > STREAM_TIME_LIMIT_SECONDS:
            raise RuntimeError(f"sequence {case_id} exceeded its candidate-time limit")
        observed_seconds = finite_positive(row.get("seconds"), f"{case_id}.seconds")
        if not math.isclose(observed_seconds, seconds, rel_tol=1.0e-9, abs_tol=1.0e-12):
            raise RuntimeError(f"sequence time differs for {case_id}")
        computed_case = 4.0 * HEAD_DIM * h_q * pairs / seconds / 1.0e12
        for name, value in (("max_step_us", max(step_us)), ("median_step_us", float(statistics.median(step_us)))):
            observed = finite_positive(row.get(name), f"{case_id}.{name}")
            if not math.isclose(observed, value, rel_tol=1.0e-9, abs_tol=1.0e-12):
                raise RuntimeError(f"sequence {name} differs for {case_id}")
    else:
        _exact_int(row, "decoy_precalls", expected_trials, label)
        trial_us = _numeric_list(row.get("trial_us"), f"{case_id}.trial_us", expected_trials)
        host_us = _numeric_list(row.get("trial_host_us"), f"{case_id}.trial_host_us", expected_trials)
        event_us = _numeric_list(row.get("trial_event_us"), f"{case_id}.trial_event_us", expected_trials)
        trial_tflops = _numeric_list(row.get("trial_tflops"), f"{case_id}.trial_tflops", expected_trials)
        trial_pairs = row.get("trial_pairs")
        if not isinstance(trial_pairs, list) or trial_pairs != [pairs] * expected_trials:
            raise RuntimeError(f"counted pairs differ from the manifest for {case_id}")
        for ordinal in range(expected_trials):
            if not math.isclose(
                trial_us[ordinal], max(host_us[ordinal], event_us[ordinal]), rel_tol=1.0e-9, abs_tol=1.0e-9
            ):
                raise RuntimeError(f"trial clock differs from max(host, event) for {case_id}")
            computed = 4.0 * HEAD_DIM * h_q * pairs / (trial_us[ordinal] * 1.0e-6) / 1.0e12
            if not math.isclose(trial_tflops[ordinal], computed, rel_tol=1.0e-9, abs_tol=1.0e-12):
                raise RuntimeError(f"trial TFLOPS differs for {case_id}")
        computed_median_us = float(statistics.median(trial_us))
        observed_median_us = finite_positive(row.get("median_us"), f"{case_id}.median_us")
        if not math.isclose(observed_median_us, computed_median_us, rel_tol=1.0e-9, abs_tol=1.0e-12):
            raise RuntimeError(f"median time differs for {case_id}")
        computed_case = float(statistics.median(trial_tflops))
    observed_case = finite_positive(row.get("tflops"), f"{case_id}.tflops")
    if not math.isclose(observed_case, computed_case, rel_tol=1.0e-9, abs_tol=1.0e-12):
        raise RuntimeError(f"case TFLOPS differs for {case_id}")
    if bound is not None and computed_case > bound:
        raise RuntimeError(f"case TFLOPS exceeds the physical bound for {case_id}")
    _validate_accuracy_metrics(row, case_id)
    return computed_case


def _validate_edge_rows(
    rows: Any, expected_edges: Sequence[Mapping[str, Any]], label: str
) -> None:
    """Rows must answer the expected cases one-to-one, in manifest order."""
    if not isinstance(rows, list) or len(rows) != len(expected_edges):
        raise RuntimeError(f"{label} results differ from the edge manifest")
    for row, case in zip(rows, expected_edges):
        if (
            not isinstance(row, Mapping)
            or row.get("case_id") != case["case_id"]
            or row.get("category") != case["category"]
        ):
            raise RuntimeError(f"{label} results differ from the edge manifest order")
        if row.get("ok") is not True:
            raise RuntimeError(f"{label} case {case['case_id']} did not pass")
        _validate_accuracy_metrics(row, str(case["case_id"]))


def _validate_edge_record(edge: Any, expected_edges: Sequence[Mapping[str, Any]]) -> int:
    if not isinstance(edge, Mapping):
        raise RuntimeError("runtime result lacks the edge record")
    native = [case for case in expected_edges if case["native"]]
    for name, value in (
        ("cases_total", len(native)),
        ("cases_skipped_sanitize_only", len(expected_edges) - len(native)),
        ("cases_passed", len(native)),
        ("cases_failed", 0),
    ):
        _exact_int(edge, name, value, "edge-case record")
    _validate_edge_rows(edge.get("results"), native, "edge-case")
    return len(native)


def _validate_sanitizer_result(
    result: Mapping[str, Any],
    sanitize_cases: Sequence[Mapping[str, Any]],
    expected_sha256: str,
) -> None:
    if result.get("schema_version") != SCHEMA_VERSION or result.get("status") != "passed":
        raise RuntimeError("sanitizer result identity differs")
    if result.get("candidate_sha256") != expected_sha256:
        raise RuntimeError("sanitizer pass evaluated a different candidate source")
    _validate_edge_rows(result.get("cases"), sanitize_cases, "sanitizer")


def _validate_protocol(protocol: Any, bound: float | None) -> None:
    expected = {
        "decoy_precall": True,
        "stream_warmups": STREAM_WARMUPS,
        "stream_check_every": STREAM_CHECK_EVERY,
        "stream_time_limit_seconds": STREAM_TIME_LIMIT_SECONDS,
        "edge_calls": EDGE_CALLS,
        "guard_bytes": GUARD_BYTES,
        "soft_landing": SOFT_LANDING,
        "max_abs_mult": MAX_ABS_MULT,
        "mean_abs_mult": MEAN_ABS_MULT,
        "max_abs_floor": MAX_ABS_FLOOR,
        "norm_l2_tol": NORM_L2_TOL,
        "plausibility_bound_tflops": bound,
    }
    if not isinstance(protocol, Mapping) or any(
        type(protocol.get(name)) is not type(value) or protocol.get(name) != value
        for name, value in expected.items()
    ):
        raise RuntimeError("runtime measurement protocol differs")


def validate_runtime_result(
    result: Mapping[str, Any],
    expected_cases: Sequence[Mapping[str, Any]],
    expected_edges: Sequence[Mapping[str, Any]],
    *,
    expected_trials: int,
    expected_warmups: int | None = None,
) -> tuple[float, dict[str, float]]:
    if result.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("runtime result schema differs")
    if result.get("generator_provenance") != GENERATOR_PROVENANCE:
        raise RuntimeError("runtime generator provenance differs")
    if result.get("status") != "passed":
        raise RuntimeError("runtime result is not a passing result")
    gates = result.get("gates")
    if not isinstance(gates, Mapping) or any(gates.get(name) is not True for name in RUNTIME_GATES):
        raise RuntimeError("runtime gates are not all satisfied")
    _validate_hardware_record(result)
    if result.get("trials_per_case") != expected_trials:
        raise RuntimeError("runtime trial count differs")
    if expected_warmups is not None and result.get("warmups") != expected_warmups:
        raise RuntimeError("runtime warmup count differs")
    hardware = result.get("hardware") or {}
    bound = plausibility_bound(hardware.get("capability"))
    _validate_protocol(result.get("protocol"), bound)
    raw_entrypoints = result.get("entrypoints")
    if not isinstance(raw_entrypoints, list) or raw_entrypoints != [DECODE_ENTRYPOINT]:
        raise RuntimeError("runtime entrypoint set differs")
    native_edges = _validate_edge_record(result.get("edge"), expected_edges)

    if len(expected_cases) != FINAL_CASE_COUNT:
        raise RuntimeError(f"selected manifest must contain exactly {FINAL_CASE_COUNT} cases")
    raw_cases = result.get("cases")
    if not isinstance(raw_cases, list):
        raise RuntimeError("runtime result lacks cases")
    by_id: dict[str, Mapping[str, Any]] = {}
    for item in raw_cases:
        if not isinstance(item, Mapping):
            raise RuntimeError("runtime case is not an object")
        case_id = item.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in by_id:
            raise RuntimeError("runtime case IDs are invalid")
        by_id[case_id] = item
    expected_ids = {str(case["case_id"]) for case in expected_cases}
    if len(expected_ids) != len(expected_cases) or set(by_id) != expected_ids:
        raise RuntimeError("runtime case set differs from the selected manifest")

    # One sentinel check after import, one per edge call; per single-call case
    # one per warm-up and two per trial (decoy and measured call); per sequence
    # one per warm-up call and one per step.  The runtime may check more.
    minimum_checks = 1 + EDGE_CALLS * native_edges
    for case in expected_cases:
        if case["family"] == "stream":
            minimum_checks += STREAM_WARMUPS + int(case["steps"])
        else:
            minimum_checks += (expected_warmups or 1) + 2 * expected_trials
    checks = result.get("sentinel_checks")
    if isinstance(checks, bool) or not isinstance(checks, int) or checks < minimum_checks:
        raise RuntimeError("runtime harness sentinel did not run on every call")

    family_values: dict[str, list[float]] = {family: [] for family in FAMILIES}
    for expected in expected_cases:
        value = _validate_case_result(by_id[str(expected["case_id"])], expected, expected_trials, bound)
        family_values[str(expected["family"])].append(value)
    families = {family: geometric_mean(values) for family, values in family_values.items()}
    primary = geometric_mean([families[family] for family in FAMILIES])

    raw_families = result.get("family_tflops")
    if not isinstance(raw_families, Mapping) or set(raw_families) != set(families):
        raise RuntimeError("runtime family aggregate set differs")
    for label, expected_value in families.items():
        observed = finite_positive(raw_families.get(label), f"{label} runtime aggregate")
        if not math.isclose(observed, expected_value, rel_tol=1.0e-9, abs_tol=1.0e-12):
            raise RuntimeError(f"runtime family aggregate differs for {label}")
    observed_primary = finite_positive(result.get("primary_tflops"), "runtime primary score")
    if not math.isclose(observed_primary, primary, rel_tol=1.0e-9, abs_tol=1.0e-12):
        raise RuntimeError("runtime primary score differs")
    return primary, families


def _numeric_summary(detail: Mapping[str, Any]) -> dict[str, float]:
    numeric: dict[str, float] = {}
    for key, value in detail.items():
        if isinstance(value, bool):
            numeric[key] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if math.isfinite(number):
                numeric[key] = number
    return numeric


def write_result(log_dir: Path, detail: Mapping[str, Any]) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_dir.chmod(0o700)
    payload = dict(detail)
    score = float(payload.get("score_tflops", 0.0))
    payload["reward"] = score
    _atomic_json(log_dir / RESULT_NAME, payload)
    numeric = _numeric_summary(payload)
    numeric["reward"] = score
    _atomic_json(log_dir / "reward.json", numeric)
    temporary = log_dir / f".reward.txt.tmp.{os.getpid()}"
    temporary.write_text(f"{score}\n", encoding="utf-8")
    os.replace(temporary, log_dir / "reward.txt")


def failure_detail(
    message: str,
    *,
    final_version: int = 0,
    versions: int = 0,
    profile: str | None = None,
    suite_id: str | None = None,
) -> dict[str, Any]:
    detail: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_policy": EVALUATION_POLICY,
        "selection_protocol_id": SELECTION_PROTOCOL_ID,
        **{name: 0.0 for name in GATE_NAMES},
        "failed_gate_code": 0.0,
        "profile_code": PROFILE_CODES.get(profile or "", 0.0),
        "score_tflops": 0.0,
        **{f"{family}_tflops": 0.0 for family in FAMILIES},
        "edge_cases_total": 0.0,
        "edge_cases_failed": 0.0,
        "sanitizer_ran": 0.0,
        "final_version": float(final_version),
        "num_candidate_versions": float(versions),
        "private_error": message,
    }
    if profile is not None:
        detail["evaluation_profile"] = profile
    if suite_id is not None:
        detail["suite_id"] = suite_id
    return detail


def _mark_failed_gate(detail: dict[str, Any], gate: str) -> None:
    detail[gate] = 0.0
    if gate in GATE_NAMES and not detail.get("failed_gate_code"):
        detail["failed_gate_code"] = float(GATE_NAMES.index(gate) + 1)
        detail["failed_gate"] = gate


def _apply_runtime_gates(detail: dict[str, Any], runtime_result: Mapping[str, Any]) -> None:
    gates = runtime_result.get("gates")
    if not isinstance(gates, Mapping):
        return
    for name in RUNTIME_GATES:
        detail[name] = 1.0 if gates.get(name) is True else 0.0
    # edge_cases_total stays the private manifest's size; the runtime counts
    # only the natively run cases.
    edge = runtime_result.get("edge")
    if isinstance(edge, Mapping):
        value = edge.get("cases_failed")
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            detail["edge_cases_failed"] = float(value)


def _sanitizer_failure_message(sanitizer: Mapping[str, Any]) -> str:
    status = sanitizer.get("status")
    if status == "memory_error":
        return "compute-sanitizer memcheck reported invalid memory accesses"
    if status == "timeout":
        return "the compute-sanitizer pass did not finish within its time limit"
    if status == "unavailable":
        return (
            "compute-sanitizer could not attach to the worker; a deployment where it cannot "
            "attach must select the skip-platform-incompatible sanitizer mode"
        )
    return str(sanitizer.get("private_error") or "the compute-sanitizer pass did not complete")


def evaluate(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    protected = verify_protected_hashes(JUDGE_ROOT / "protected_hashes.txt")
    if protected:
        return 1, failure_detail("protected Judge/runtime hash failure") | {
            "protected_failures": protected
        }

    if args.sanitizer_mode not in SANITIZER_MODES:
        return 1, failure_detail("invalid compute-sanitizer mode")
    try:
        profile, suite_id, round_id, version_override = load_evaluation_profile()
    except Exception as exc:
        return 1, failure_detail(f"{type(exc).__name__}: {exc}")
    base = failure_detail("evaluation did not complete", profile=profile, suite_id=suite_id)
    base["profile_ok"] = 1.0
    base["round_id"] = round_id
    base["sanitizer_mode"] = args.sanitizer_mode
    if version_override is not None:
        base["candidate_version_override"] = float(version_override)

    try:
        cases, manifest_sha256 = load_final_manifest()
        edge_cases, edge_manifest_sha256 = load_edge_manifest()
        if {str(case["case_id"]) for case in cases} & {str(case["case_id"]) for case in edge_cases}:
            raise RuntimeError("scored and edge case IDs overlap")
        base["suite_ok"] = 1.0
        base["manifest_sha256"] = manifest_sha256
        base["edge_manifest_sha256"] = edge_manifest_sha256
        base["case_count"] = float(len(cases))
        base["edge_cases_total"] = float(len(edge_cases))
        base["edge_sanitize_cases"] = float(sum(1 for case in edge_cases if case["sanitize"]))
    except Exception as exc:
        base["private_error"] = f"{type(exc).__name__}: {exc}"
        _mark_failed_gate(base, "suite_ok")
        return 1, base

    submission_root = locate_submission(args.app_dir.resolve())
    admission = validate_submission(submission_root, version_override)
    final_version = admission.get("final_version")
    versions = admission.get("num_candidate_versions")
    version_count = versions if isinstance(versions, int) and not isinstance(versions, bool) else 0
    selected = final_version if isinstance(final_version, int) and not isinstance(final_version, bool) else 0
    base["final_version"] = float(selected)
    base["num_candidate_versions"] = float(version_count)
    if not admission.get("ok") or selected <= 0:
        base["private_error"] = "submission source admission failed"
        base["admission"] = admission
        _mark_failed_gate(base, "source_ok")
        return 1, base
    base["source_ok"] = 1.0

    if args.trials != EXPECTED_TRIALS or args.warmups != EXPECTED_WARMUPS:
        base["private_error"] = (
            f"verifier requires exactly {EXPECTED_TRIALS} trials and "
            f"{EXPECTED_WARMUPS} warmup"
        )
        return 1, base
    total_timeout = float(args.total_timeout)
    if not math.isfinite(total_timeout) or total_timeout <= 60.0:
        base["private_error"] = "invalid verifier total timeout"
        return 1, base

    started = time.monotonic()
    deadline = started + total_timeout
    run_root = Path(tempfile.mkdtemp(prefix="standalone-paged-gqa-v4-", dir="/tmp"))
    run_root.chmod(0o755)
    try:
        try:
            hardware = _trusted_hardware()
        except Exception as exc:
            base["private_error"] = f"{type(exc).__name__}: {exc}"
            _mark_failed_gate(base, "hardware_ok")
            return 1, base
        base["hardware_ok"] = 1.0
        selected_source = Path(str(admission["selected_path"]))
        candidate_dir = run_root / "selected_candidate"
        candidate_dir.mkdir(mode=0o755)
        candidate_path = candidate_dir / "submission.py"
        shutil.copyfile(selected_source, candidate_path)
        candidate_path.chmod(0o444)
        candidate_dir.chmod(0o555)

        # The sanitizer subset runs first, before any native call: memcheck stops
        # an out-of-bounds kernel before the access reaches the hardware, where
        # a wild pointer can wedge the shared host driver (2026-10-06).
        if args.sanitizer_mode == "required":
            sanitize_cases = [case for case in edge_cases if case["sanitize"]]
            remaining = deadline - time.monotonic() - 60.0
            if remaining <= 0.0:
                raise RuntimeError("Judge total deadline exhausted before the sanitizer pass")
            sanitizer = run_sanitizer(
                candidate_path,
                run_root,
                sanitize_cases,
                timeout=min(float(args.sanitizer_timeout), remaining),
                expected_sha256=str(admission["selected_sha256"]),
            )
            base["sanitizer"] = sanitizer
            if sanitizer.get("status") != "passed":
                gate = str(sanitizer.get("failed_gate") or "memory_safety_ok")
                gate = gate if gate in GATE_NAMES else "memory_safety_ok"
                _mark_failed_gate(base, gate)
                if gate == "edge_correctness_ok":
                    # The pass stops at its first wrong answer.
                    base["edge_cases_failed"] = 1.0
                base["private_error"] = _sanitizer_failure_message(sanitizer)
                return 1, base
            base["sanitizer_ran"] = 1.0
        else:
            base["sanitizer"] = {
                "status": "not_run",
                "mode": args.sanitizer_mode,
                "reason": "compute-sanitizer cannot attach to CUDA processes under this "
                "deployment's GPU virtualization hook",
            }

        remaining = deadline - time.monotonic() - 30.0
        if remaining <= 0.0:
            raise RuntimeError("Judge total deadline exhausted")
        runtime_result = run_worker(
            candidate_path,
            run_root,
            cases,
            edge_cases,
            trials=args.trials,
            warmups=args.warmups,
            timeout=min(float(args.candidate_timeout), remaining),
        )
        _apply_runtime_gates(base, runtime_result)
        if runtime_result.get("candidate_sha256") not in (None, admission["selected_sha256"]):
            raise RuntimeError("runtime evaluated a different candidate source")
        if runtime_result.get("status") != "passed":
            gate = str(runtime_result.get("failed_gate") or "correctness_ok")
            _mark_failed_gate(base, gate if gate in GATE_NAMES else "correctness_ok")
            base["private_error"] = str(runtime_result.get("private_error") or "runtime gate failure")
            base["private_runtime_result"] = runtime_result
            return 1, base
        if runtime_result.get("candidate_sha256") != admission["selected_sha256"]:
            raise RuntimeError("runtime evaluated a different candidate source")
        score, families = validate_runtime_result(
            runtime_result,
            cases,
            edge_cases,
            expected_trials=args.trials,
            expected_warmups=args.warmups,
        )

        detail = dict(base)
        detail.update(
            {
                "score_tflops": score,
                **{f"{family}_tflops": families[family] for family in FAMILIES},
                "selected_source_sha256": admission["selected_sha256"],
                "score_method": "geomean over the decode, spec, long and stream family geomeans",
                "wall_time_seconds": time.monotonic() - started,
                "hardware": hardware,
                "private_runtime_result": runtime_result,
            }
        )
        detail["private_error"] = ""
        return 0, detail
    except Exception as exc:
        base["private_error"] = f"{type(exc).__name__}: {exc}"
        if not base.get("failed_gate_code"):
            _mark_failed_gate(base, "correctness_ok")
        return 1, base
    finally:
        shutil.rmtree(run_root, ignore_errors=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-dir", type=Path, default=Path(os.environ.get("APP_DIR", "/app")))
    parser.add_argument(
        "--log-dir", type=Path, default=Path(os.environ.get("LOG_DIR", "/logs/verifier"))
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=int(os.environ.get("STANDALONE_PAGED_GQA_VERIFY_TRIALS", "3")),
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=int(os.environ.get("STANDALONE_PAGED_GQA_VERIFY_WARMUPS", "1")),
    )
    parser.add_argument(
        "--candidate-timeout",
        type=int,
        default=int(os.environ.get("STANDALONE_PAGED_GQA_CANDIDATE_TIMEOUT", "7200")),
    )
    parser.add_argument(
        "--sanitizer-timeout",
        type=int,
        default=int(os.environ.get("STANDALONE_PAGED_GQA_SANITIZER_TIMEOUT", "780")),
    )
    parser.add_argument(
        "--sanitizer-mode",
        default=os.environ.get("STANDALONE_PAGED_GQA_SANITIZER_MODE", "required"),
    )
    parser.add_argument(
        "--total-timeout",
        type=int,
        default=int(os.environ.get("STANDALONE_PAGED_GQA_TOTAL_TIMEOUT", "8100")),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rc, detail = evaluate(args)
    except Exception as exc:
        detail = failure_detail(f"{type(exc).__name__}: {exc}")
        rc = 1
    write_result(args.log_dir.resolve(), detail)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
