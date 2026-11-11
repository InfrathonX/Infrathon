#!/usr/bin/env python3
"""Private verifier for standalone BF16 sparse-MLA prefill H100 v2.

Order of evaluation: protected hashes, evaluation profile, the two root-only
manifests (scored suite and edge cases), source admission, hardware, the
compute-sanitizer memcheck subset in one separate unprivileged process (first,
so an out-of-bounds kernel is stopped before any native call), and then the
unprivileged worker (native edge cases, then the scored suite).  Every
aggregate the worker reports is recomputed here from its trial records.

The sanitizer process runs its cases sequentially and stops at the first
fatal CUDA error; nothing restarts it.  Only an explicit memcheck report of an
invalid ``__global__``/``__shared__``/``__local__`` access or a misaligned
address is a memory-safety failure.  A fatal error without such a report (a
tool-level error such as ``cudaErrorInvalidPc``, seen with historical Triton
kernels that pass natively) makes the pass *inconclusive*: the case in flight
and every later sanitizer case are then judged natively with soft-landing
guard bands, on numerics and guards only.
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


JUDGE_ROOT = Path("/opt/standalone_sparse_mla_prefill_judge")
RUNTIME_ROOT = Path("/opt/standalone_sparse_mla_prefill_worker")
WORKER = JUDGE_ROOT / "candidate_worker.py"
SANITIZER = Path("/usr/local/cuda/compute-sanitizer/compute-sanitizer")
EVALUATION_CONTEXT = Path("/tmp/.sforge-evaluation-context.json")
FINAL_MANIFEST = JUDGE_ROOT / "private/final_cases.json"
EDGE_MANIFEST = JUDGE_ROOT / "private/edge_cases.json"
RESULT_NAME = "standalone_sparse_mla_prefill_v2_results.json"
FINAL_SUITE_ID = "final-holdout-v4"
EDGE_SUITE_ID = "edge-holdout-v3"
EVALUATION_POLICY = "full-final-shared-v3"
SELECTION_PROTOCOL_ID = "sparse-mla-shared-final-v3"
GENERATOR_PROVENANCE = "flashmla-ba89a34-derived-sparse-prefill-v2-judge-port"
SCHEMA_VERSION = 3
# The scored final manifest (scripts/generate_final_manifest.py) is schema 5.
FINAL_SCHEMA_VERSION = 5
PROFILE_CODES = {"auto": 1.0, "final": 2.0}
PREFILL_ENTRYPOINT = "sparse_mla_prefill"
CASE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,95}$")
CATEGORY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
SCORED_KEYS = frozenset({"case_id", "group", "family", "h_q", "s_q", "s_kv", "sink_mode", "gated", "seed"})
SCORED_GROUP_KEYS: dict[str, frozenset[str]] = {
    "anchor": frozenset(), "sweep": frozenset(), "heads": frozenset(),
    "length": frozenset({"length_divisor", "ragged"}), "multi": frozenset({"request_lengths"}),
    "trace": frozenset({"calls"}), "replay": frozenset({"calls"}), "pattern": frozenset({"index_pattern"}),
}
EDGE_OPTIONAL_KEYS = frozenset({"h_q"})
EDGE_KEYS = frozenset({
    "case_id", "category", "family", "s_q", "s_kv", "topk", "sm_scale", "bias",
    "index_mode", "invalid_rows", "duplicates", "scramble", "sink_mode",
    "length_mode", "seed", "sanitize", "native",
})

# --------------------------------------------------------------------------
# compute-sanitizer.
# --------------------------------------------------------------------------
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
# Padding after every allocation: without it an overrun that lands in the next
# allocation is not reported (2026-10-06 probe: a one-element read past q went
# unflagged whenever q's size was a 2 MiB multiple).
SANITIZER_PADDING_BYTES = 4 << 20
# torch and cuBLAS kernels are not instrumented: only the candidate's kernels
# are (source admission keeps candidate kernel names out of these patterns).
SANITIZER_KERNEL_EXCLUDES = (
    "regex=^_Z", "kns=xmma", "kns=gemm", "kns=nvjet", "kns=cutlass", "kns=cublas",
)
SANITIZER_ARGUMENTS = (
    "--tool=memcheck",
    f"--error-exitcode={SANITIZER_EXIT_CODE}",
    "--target-processes=application-only",
    "--destroy-on-device-error=context",
    "--force-blocking-launches",
    "--check-api-memory-access=no",
    "--padding",
    str(SANITIZER_PADDING_BYTES),
    "--print-limit=20",
    *(item for pattern in SANITIZER_KERNEL_EXCLUDES for item in ("--kernel-name-exclude", pattern)),
)
# Only these memcheck reports are a memory-safety failure.
SANITIZER_ACCESS_MARKERS = (
    "invalid __global__",
    "invalid __shared__",
    "invalid __local__",
    "misaligned",
)
SANITIZE_STATUSES = ("passed", "deferred", "fatal", "memory_safety", "harness_integrity")

# --------------------------------------------------------------------------
# The disclosed scored suite and support domain.  These tables are duplicated
# from the trusted runtime on purpose: the Judge must not depend on an
# importable copy of it, and task-local tests assert the two agree.
# --------------------------------------------------------------------------
FAMILIES = ("v32", "glm5", "v4_flash", "v4_pro")
FAMILY_GEOMETRY: dict[str, tuple[int, int]] = {
    "v32": (576, 128),
    "glm5": (576, 64),
    "v4_flash": (512, 64),
    "v4_pro": (512, 128),
}
FAMILY_TOPK = {"v32": 2048, "glm5": 2048, "v4_flash": 512, "v4_pro": 1024}
FAMILY_MODEL = {
    "v32": "DeepSeek-V3.2", "glm5": "GLM-5",
    "v4_flash": "DeepSeek-V4-Flash", "v4_pro": "DeepSeek-V4-Pro",
}
FAMILY_MAX_CONTEXT = {"v32": 163_840, "glm5": 202_752, "v4_flash": 1 << 20, "v4_pro": 1 << 20}
FAMILY_SM_SCALE = {"v32": 192 ** -0.5, "glm5": 256 ** -0.5, "v4_flash": 512 ** -0.5, "v4_pro": 512 ** -0.5}
ANCHOR_S_KV: dict[str, tuple[int, ...]] = {
    "v32": (8192, 32768, 65536, 98304, 131072),
    "glm5": (8192, 32768, 65536, 131072),
    "v4_flash": (8192, 32768, 49152, 65536),
    "v4_pro": (8192, 32768, 49152, 65536),
}
ANCHOR_S_Q = 4096
ANCHOR_SM_SCALE = 0.5
GROUP_TIER = {
    "anchor": "evidence", "trace": "evidence", "replay": "evidence", "multi": "evidence",
    "pattern": "evidence", "sweep": "sweep", "heads": "sweep", "length": "sweep",
}
GROUPS = tuple(GROUP_TIER)
TIERS = ("evidence", "sweep")
STREAM_GROUPS = ("trace", "replay")
SCORED_SINK_MODES = ("upstream", "none", "finite")
LENGTH_DIVISORS = (2, 4, 8)
INDEX_PATTERNS = ("window", "sink_window", "blocks", "sorted")
MAX_REQUESTS = 128
MAX_STREAM_CALLS = 256
STREAM_WARM_CALLS = 16
STREAM_CHECK_EVERY = 8
STREAM_TIME_LIMIT_SECONDS = 8.0
STALE_TAIL_FRACTION = 0.5
MIN_H_Q, H_Q_MULTIPLE = 16, 16
# The frozen scored instance: case count per group.
FINAL_GROUP_COUNTS = {
    "anchor": 17, "trace": 80, "replay": 60, "multi": 100, "pattern": 120,
    "sweep": 400, "heads": 120, "length": 100,
}
D_V = 512
MIN_TOPK, MAX_TOPK, TOPK_MULTIPLE = 128, 2048, 128
MAX_S_Q = 70_000
MAX_S_KV = 1 << 20
MAX_Q_ELEMENTS = 1 << 32
MAX_TOPK_POSITIONS = 1 << 25
NATIVE_MAX_TENSOR_BYTES = 3 << 30
LARGE_OFFSET_ELEMENTS = 1 << 31
MAX_SEED = (1 << 63) - 1
BIAS_MODES = ("upstream", "zero")
INDEX_MODES = ("upstream", "valid_pad", "sprinkle", "causal")
INVALID_ROW_MODES = ("none", "upstream", "sparse", "some", "all")
SINK_MODES = ("none", "finite", "upstream", "extreme")
LENGTH_MODES = ("none", "upstream", "edge", "half")

# Correctness and memory-safety protocol; keep identical to the runtime.
OUT_ABS_TOL = 8.0e-4
OUT_REL_TOL = 3.01 / 128
OUT_COS_TOL = 7.0e-6
LOGIT_ABS_TOL = 1.0e-6
LOGIT_REL_TOL = 2.01 / 65536
LOGIT_COS_TOL = 1.0e-7
COS_LIMITS = {"out": OUT_COS_TOL, "max_logits": LOGIT_COS_TOL, "lse": LOGIT_COS_TOL}
GUARD_MIN_BYTES = 1 << 20
GUARD_ROWS = 128
SOFT_LANDING = {"indices": 0, "topk_length": 0}
L2_FLUSH_BYTES = 256 << 20
EDGE_CALLS = 2

FINAL_CASE_COUNT = sum(FINAL_GROUP_COUNTS.values())
FINAL_SEED_MINIMUM = 1 << 32  # every public seed is below this
# The frozen private edge instance; keep identical to
# scripts/generate_edge_manifest.py and the frozen judge/private/edge_cases.json.
EDGE_CASE_COUNT = 470
EDGE_SANITIZE_COUNT = 17
EDGE_SANITIZE_ONLY_COUNT = 1
EDGE_CATEGORY_COUNTS = {
    "upstream_correctness": 90,
    "upstream_features": 168,
    "upstream_all_invalid": 12,
    "upstream_empty_blocks": 12,
    "upstream_large_s_q": 1,
    "scored_semantics": 17,
    "scored_precision": 17,
    "odd_sizes": 112,
    "v1_fixture": 8,
    "causal": 20,
    "large_kv": 4,
    "tp_heads": 9,
}
EXPECTED_TRIALS = 3
EXPECTED_WARMUPS = 1


def _scored_shape(case: Mapping[str, Any]) -> tuple[str, int] | None:
    family = case.get("family")
    if (
        case.get("s_q") == ANCHOR_S_Q
        and family in FAMILIES
        and case.get("h_q") in (None, FAMILY_GEOMETRY[family][1])
        and case.get("topk") == FAMILY_TOPK[family]
        and case.get("s_kv") in ANCHOR_S_KV[family]
    ):
        return str(family), int(case["s_kv"])
    return None


def _all_scored_shapes(predicate: Callable[[Mapping[str, Any]], bool]) -> Callable[[Sequence[Mapping[str, Any]]], bool]:
    expected = {(family, s_kv) for family in FAMILIES for s_kv in ANCHOR_S_KV[family]}

    def covered(cases: Sequence[Mapping[str, Any]]) -> bool:
        return expected <= {
            shape for case in cases if predicate(case) and (shape := _scored_shape(case)) is not None
        }

    return covered


def _power_of_two(value: float) -> bool:
    mantissa, _exponent = math.frexp(value)
    return mantissa == 0.5


# Hazards the private instance must keep exercising (plan section 3.4).  A
# regenerated manifest that silently drops one fails suite_ok.
EDGE_REQUIRED_COVERAGE: dict[str, Callable[[Sequence[Mapping[str, Any]]], bool]] = {
    "s_q = 1": lambda cases: any(case["s_q"] == 1 and case["native"] for case in cases),
    "s_q not divisible by 8": lambda cases: any(case["s_q"] % 8 and case["native"] for case in cases),
    "s_kv below topk (invalid padding)": lambda cases: any(
        case["s_kv"] < case["topk"] and case["index_mode"] != "valid_pad" for case in cases
    ),
    "rows with no valid key": lambda cases: any(case["invalid_rows"] != "none" for case in cases),
    "interleaved invalid slots": lambda cases: any(case["index_mode"] == "sprinkle" for case in cases),
    "duplicated keys": lambda cases: any(case["duplicates"] for case in cases),
    "+-inf sinks": lambda cases: any(case["sink_mode"] in ("upstream", "extreme") for case in cases),
    "boundary topk_length": lambda cases: any(case["length_mode"] == "edge" for case in cases),
    "non-power-of-two sm_scale": lambda cases: any(
        not _power_of_two(float(case["sm_scale"])) for case in cases
    ),
    "every scored shape with edge semantics": _all_scored_shapes(
        lambda case: case["index_mode"] == "sprinkle" and case["length_mode"] == "edge"
        and case["sink_mode"] == "extreme" and not _power_of_two(float(case["sm_scale"]))
    ),
    "every scored shape with zero bias": _all_scored_shapes(lambda case: case["bias"] == "zero"),
    "s_q above 65535 under the sanitizer": lambda cases: any(
        case["s_q"] > 65535 and case["sanitize"] for case in cases
    ),
    "an anchor shape of every DeepSeek family under the sanitizer": lambda cases: {
        shape[0] for case in cases if case["sanitize"] and (shape := _scored_shape(case)) is not None
    } == {"v32", "v4_flash", "v4_pro"},
    "causal indices in every family": lambda cases: {
        case["family"] for case in cases if case["index_mode"] == "causal"
    } == set(FAMILIES),
    "keys near the context limit of every family": lambda cases: {
        case["family"] for case in cases if case["s_kv"] >= FAMILY_MAX_CONTEXT[case["family"]] - 1
    } == set(FAMILIES),
    "tensor-parallel head counts": lambda cases: any(
        case.get("h_q") not in (None, FAMILY_GEOMETRY[case["family"]][1]) for case in cases
    ),
    "a 2**20-key case under the sanitizer": lambda cases: any(
        case["sanitize"] and case["s_kv"] >= (1 << 20) - 1 for case in cases
    ),
    "a tensor-parallel head count under the sanitizer": lambda cases: any(
        case["sanitize"] and case.get("h_q") not in (None, FAMILY_GEOMETRY[case["family"]][1])
        for case in cases
    ),
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
    (9, 0): 1_000.0,
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




def _scored_heads(case_id: str, group: str, family: str, raw: Any) -> int:
    family_heads = FAMILY_GEOMETRY[family][1]
    h_q = _integer(raw, f"{case_id}.h_q", minimum=MIN_H_Q, maximum=family_heads)
    if h_q % H_Q_MULTIPLE:
        raise RuntimeError(f"final case {case_id} h_q is not a multiple of {H_Q_MULTIPLE}")
    if (group == "heads") == (h_q == family_heads):
        raise RuntimeError(f"final case {case_id}: only heads cases change the head count")
    return h_q


def validate_final_manifest(payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """The frozen scored instance: FINAL_GROUP_COUNTS cases, every anchor shape once,
    distinct 63-bit seeds, every shape inside the contract."""
    if set(payload) != {"schema_version", "suite_id", "cases"}:
        raise RuntimeError("final manifest fields differ from schema v5")
    if payload.get("schema_version") != FINAL_SCHEMA_VERSION or payload.get("suite_id") != FINAL_SUITE_ID:
        raise RuntimeError("final manifest identity differs")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != FINAL_CASE_COUNT:
        raise RuntimeError(f"final manifest must contain exactly {FINAL_CASE_COUNT} cases")

    cases: list[dict[str, Any]] = []
    ids: set[str] = set()
    seeds: set[int] = set()
    anchors: set[tuple[str, int]] = set()
    counts: dict[str, int] = {}
    for raw_case in raw_cases:
        if not isinstance(raw_case, Mapping) or raw_case.get("group") not in GROUPS:
            raise RuntimeError("final case group is invalid")
        group = str(raw_case["group"])
        if set(raw_case) != SCORED_KEYS | SCORED_GROUP_KEYS[group]:
            raise RuntimeError("final case fields differ from schema v5")
        case_id = raw_case.get("case_id")
        if not isinstance(case_id, str) or CASE_ID_RE.fullmatch(case_id) is None or case_id in ids:
            raise RuntimeError("final case_id is invalid or repeated")
        ids.add(case_id)
        family = raw_case.get("family")
        if family not in FAMILIES:
            raise RuntimeError(f"final case {case_id} family is invalid")
        family = str(family)
        h_q = _scored_heads(case_id, group, family, raw_case.get("h_q"))
        d_qk, topk = FAMILY_GEOMETRY[family][0], FAMILY_TOPK[family]
        s_q = _integer(raw_case.get("s_q"), f"{case_id}.s_q", maximum=MAX_S_Q)
        cap = FAMILY_MAX_CONTEXT[family] * (5 if group == "multi" else 1)
        s_kv = _integer(raw_case.get("s_kv"), f"{case_id}.s_kv", maximum=min(MAX_S_KV, cap))
        if s_q * h_q * d_qk > MAX_Q_ELEMENTS or s_q * topk > MAX_TOPK_POSITIONS:
            raise RuntimeError(f"final case {case_id} exceeds the contract caps")
        if 2 * s_q * h_q * max(d_qk, D_V) > NATIVE_MAX_TENSOR_BYTES:
            raise RuntimeError(f"final case {case_id} q or out exceeds the native size bound")
        sink_mode = raw_case.get("sink_mode")
        gated = raw_case.get("gated")
        if sink_mode not in SCORED_SINK_MODES or not isinstance(gated, bool):
            raise RuntimeError(f"final case {case_id} sink_mode or gated is invalid")
        if (group == "anchor") != (sink_mode == "upstream"):
            raise RuntimeError(f"final case {case_id}: only anchors use upstream sinks")
        if group in ("anchor", "multi", "pattern", *STREAM_GROUPS) and not gated:
            raise RuntimeError(f"final case {case_id} must be checked")
        seed = _integer(raw_case.get("seed"), f"{case_id}.seed", minimum=FINAL_SEED_MINIMUM, maximum=MAX_SEED)
        if seed in seeds:
            raise RuntimeError("final case seeds are not unique")
        seeds.add(seed)
        case: dict[str, Any] = {key: raw_case[key] for key in SCORED_KEYS}
        case.update(family=family, h_q=h_q, s_q=s_q, s_kv=s_kv)
        if group == "anchor":
            if s_q != ANCHOR_S_Q or s_kv not in ANCHOR_S_KV[family] or (family, s_kv) in anchors:
                raise RuntimeError(f"final case {case_id} is not a distinct anchor shape")
            anchors.add((family, s_kv))
        elif group in ("sweep", "heads", "pattern") and s_q > s_kv:
            raise RuntimeError(f"final case {case_id}: a causal single call needs s_q <= s_kv")
        if group == "length":
            if raw_case.get("length_divisor") not in LENGTH_DIVISORS:
                raise RuntimeError(f"final case {case_id} length_divisor is invalid")
            case["length_divisor"] = int(raw_case["length_divisor"])
            if not isinstance(raw_case.get("ragged"), bool):
                raise RuntimeError(f"final case {case_id} ragged is invalid")
            case["ragged"] = bool(raw_case["ragged"])
        if group == "pattern":
            if raw_case.get("index_pattern") not in INDEX_PATTERNS:
                raise RuntimeError(f"final case {case_id} index_pattern is invalid")
            case["index_pattern"] = str(raw_case["index_pattern"])
        if group == "multi":
            lengths = raw_case.get("request_lengths")
            if (
                not isinstance(lengths, list) or not 2 <= len(lengths) <= MAX_REQUESTS
                or any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in lengths)
                or sum(lengths) != s_kv or len(lengths) > s_q
            ):
                raise RuntimeError(f"final case {case_id} request_lengths are invalid")
            case["request_lengths"] = [int(item) for item in lengths]
        if group in STREAM_GROUPS:
            calls = raw_case.get("calls")
            if not isinstance(calls, list) or not 1 <= len(calls) <= MAX_STREAM_CALLS:
                raise RuntimeError(f"final case {case_id} calls are invalid")
            parsed = []
            for call in calls:
                if (
                    not isinstance(call, list) or len(call) != 2
                    or any(isinstance(item, bool) or not isinstance(item, int) for item in call)
                    or not 1 <= call[0] <= call[1] <= s_kv or call[0] > s_q
                ):
                    raise RuntimeError(f"final case {case_id} has an invalid call")
                parsed.append([int(call[0]), int(call[1])])
            if max(item[0] for item in parsed) != s_q or max(item[1] for item in parsed) != s_kv:
                raise RuntimeError(f"final case {case_id}: s_q and s_kv are the largest call")
            case["calls"] = parsed
        counts[group] = counts.get(group, 0) + 1
        cases.append(case)
    if counts != FINAL_GROUP_COUNTS:
        raise RuntimeError("final manifest group counts differ from the frozen design")
    if anchors != {(family, s_kv) for family in FAMILIES for s_kv in ANCHOR_S_KV[family]}:
        raise RuntimeError("final cases must cover every anchor shape exactly once")
    if {case["family"] for case in cases} != set(FAMILIES):
        raise RuntimeError("final cases must cover every family")
    return tuple(cases)


def validate_edge_case(raw_case: Any) -> dict[str, Any]:
    """The runtime's edge-case domain (``validate_edge_case``), checked independently."""
    if not isinstance(raw_case, Mapping) or not EDGE_KEYS <= set(raw_case) <= EDGE_KEYS | EDGE_OPTIONAL_KEYS:
        raise RuntimeError("edge case fields differ from schema v3")
    case = dict(raw_case)
    case_id = case.get("case_id")
    if not isinstance(case_id, str) or CASE_ID_RE.fullmatch(case_id) is None:
        raise RuntimeError("edge case_id is invalid")
    category = case.get("category")
    if not isinstance(category, str) or CATEGORY_RE.fullmatch(category) is None:
        raise RuntimeError(f"{case_id}: category is invalid")
    family = case.get("family")
    if family not in FAMILIES:
        raise RuntimeError(f"{case_id}: family outside the domain")
    d_qk, h_q = FAMILY_GEOMETRY[str(family)]
    if "h_q" in case:
        h_q = _integer(case.get("h_q"), f"{case_id}.h_q", minimum=MIN_H_Q, maximum=h_q)
        if h_q % H_Q_MULTIPLE:
            raise RuntimeError(f"{case_id}: h_q must be a multiple of {H_Q_MULTIPLE}")
    s_q = _integer(case.get("s_q"), f"{case_id}.s_q", maximum=MAX_S_Q)
    _integer(case.get("s_kv"), f"{case_id}.s_kv", maximum=MAX_S_KV)
    topk = _integer(case.get("topk"), f"{case_id}.topk", minimum=MIN_TOPK, maximum=MAX_TOPK)
    if topk % TOPK_MULTIPLE:
        raise RuntimeError(f"{case_id}: topk must be a multiple of {TOPK_MULTIPLE}")
    if s_q * h_q * d_qk > MAX_Q_ELEMENTS or s_q * topk > MAX_TOPK_POSITIONS:
        raise RuntimeError(f"{case_id}: shape exceeds the contract caps")
    scale = case.get("sm_scale")
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not math.isfinite(float(scale))
        or float(scale) <= 0.0
    ):
        raise RuntimeError(f"{case_id}: sm_scale must be finite and positive")
    for name, allowed in (
        ("bias", BIAS_MODES),
        ("index_mode", INDEX_MODES),
        ("invalid_rows", INVALID_ROW_MODES),
        ("sink_mode", SINK_MODES),
        ("length_mode", LENGTH_MODES),
    ):
        if case.get(name) not in allowed:
            raise RuntimeError(f"{case_id}: {name} outside the domain")
    for flag in ("duplicates", "scramble", "sanitize", "native"):
        if not isinstance(case.get(flag), bool):
            raise RuntimeError(f"{case_id}: {flag} flag is invalid")
    _integer(case.get("seed"), f"{case_id}.seed", minimum=0, maximum=MAX_SEED)
    if case["native"]:
        if 2 * s_q * h_q * max(d_qk, D_V) > NATIVE_MAX_TENSOR_BYTES:
            raise RuntimeError(f"{case_id}: q or out exceeds the native size bound")
    else:
        # Only the large-offset case may skip native execution, and only
        # because its q and out exceed 2**31 elements.
        if not case["sanitize"]:
            raise RuntimeError(f"{case_id}: a case that never runs natively must be a sanitizer case")
        if s_q * h_q * d_qk <= LARGE_OFFSET_ELEMENTS:
            raise RuntimeError(f"{case_id}: a sanitizer-only case must exceed 2^31 q elements")
    return case


def validate_edge_manifest(payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    if set(payload) != {"schema_version", "suite_id", "cases"}:
        raise RuntimeError("edge manifest fields differ from schema v3")
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
    counts: dict[str, int] = {}
    for case in cases:
        counts[case["category"]] = counts.get(case["category"], 0) + 1
    if counts != EDGE_CATEGORY_COUNTS:
        raise RuntimeError("edge manifest category counts differ from the frozen design")
    if sum(1 for case in cases if case["sanitize"]) != EDGE_SANITIZE_COUNT:
        raise RuntimeError(f"edge manifest must flag exactly {EDGE_SANITIZE_COUNT} sanitizer cases")
    if sum(1 for case in cases if not case["native"]) != EDGE_SANITIZE_ONLY_COUNT:
        raise RuntimeError(
            f"edge manifest must hold exactly {EDGE_SANITIZE_ONLY_COUNT} sanitizer-only case"
        )
    missing = [label for label, covered in EDGE_REQUIRED_COVERAGE.items() if not covered(cases)]
    if missing:
        raise RuntimeError(f"edge manifest no longer exercises: {missing}")
    return tuple(cases)


def sanitize_order(cases: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Sanitizer-only cases first, then the other flagged cases in manifest order
    (the runtime's ``sanitize_order``; tests assert the two agree)."""
    flagged = [case for case in cases if case.get("sanitize") is True]
    return [case for case in flagged if case.get("native") is not True] + [
        case for case in flagged if case.get("native") is True
    ]


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
        expected_uid=expected_uid, max_bytes=1024 * 1024,
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
        app_dir / "workdir/sparse_mla_prefill_lab",
        app_dir / "environment/workdir/sparse_mla_prefill_lab",
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
        "judge:verify_sparse_mla.py",
        "worker:sparse_mla_prefill_runtime.py",
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
    promoted: Sequence[str] = (),
) -> dict[str, Any]:
    """Native run: every native edge case plus the promoted sanitizer-only ones, then the suite."""
    account, result_dir, cache_dir = _prepare_run_dirs(run_root, "worker")
    manifest_bytes = json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "mode": "suite",
            "suite_id": FINAL_SUITE_ID,
            "cases": list(cases),
            "edge_suite_id": EDGE_SUITE_ID,
            "edge_cases": list(edge_cases),
            "promoted_sanitize_only": list(promoted),
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


def _read_log(log_path: Path) -> str:
    try:
        return log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _sanitizer_report(raw_log: str) -> list[str]:
    lines = raw_log.splitlines()
    keep = [line for line in lines if line.startswith("=========")]
    return keep[:40] if keep else lines[-20:]


def explicit_access_report(raw_log: str) -> bool:
    """True only for memcheck's own invalid-access or misaligned-address reports."""
    return any(
        marker in line.lower()
        for line in raw_log.splitlines()
        if line.startswith("=========")
        for marker in SANITIZER_ACCESS_MARKERS
    )


def _progress(raw_log: str) -> tuple[list[str], dict[str, str]]:
    """Case IDs the worker began, and the status it ended each one with."""
    begun: list[str] = []
    ended: dict[str, str] = {}
    for line in raw_log.splitlines():
        parts = line.strip().split()
        if len(parts) == 3 and parts[0] == "SANITIZE_BEGIN":
            begun.append(parts[2])
        elif len(parts) == 4 and parts[0] == "SANITIZE_END":
            ended[parts[2]] = parts[3]
    return begun, ended


def _inconclusive(
    record: dict[str, Any], order: Sequence[str], index: int, sanitize_only: set[str],
    deferred: Sequence[str], reason: str,
) -> dict[str, Any]:
    """Stop at ``order[index]``: that case and every later one are judged natively."""
    uncovered = list(order[index:])
    record.update(
        status="inconclusive",
        inconclusive_case_id=order[index] if index < len(order) else None,
        inconclusive_reason=reason,
        deferred_case_ids=list(deferred),
        uncovered_case_ids=uncovered,
        promote_case_ids=[
            case_id for case_id in order if case_id in sanitize_only
            and (case_id in uncovered or case_id in deferred)
        ],
    )
    return record


def run_sanitizer(
    candidate_path: Path,
    run_root: Path,
    sanitize_cases: Sequence[Mapping[str, Any]],
    *,
    timeout: float,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Run the flagged edge cases once each under compute-sanitizer memcheck.

    One process, sequential, stopped by the runtime at the first fatal CUDA
    error and never restarted.  Statuses: ``passed`` (every case completed;
    numeric verdicts of ``deferred`` cases move to the native run),
    ``inconclusive`` (a fatal error without an invalid-access report; the case
    in flight and the later ones run natively), and the failures
    ``memory_error``, ``failed``, ``timeout`` and ``unavailable``.
    """
    if not SANITIZER.is_file():
        raise RuntimeError("compute-sanitizer is missing from the Judge image")
    ordered = sanitize_order(sanitize_cases)
    order = [str(case["case_id"]) for case in ordered]
    sanitize_only = {str(case["case_id"]) for case in ordered if not case["native"]}
    expected = expected_sha256 if expected_sha256 is not None else sha256_file(candidate_path)
    account, result_dir, cache_dir = _prepare_run_dirs(run_root, "sanitizer")
    payload = json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "mode": "sanitize",
            "edge_suite_id": EDGE_SUITE_ID,
            "edge_cases": [dict(case) for case in ordered],
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
    record: dict[str, Any] = {"cases": len(ordered), "order": order}
    log_path = run_root / "sanitizer.log"
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
        raw_log = _read_log(log_path)
        record.update(returncode=None, report=_sanitizer_report(raw_log))
        if explicit_access_report(raw_log):
            return record | {"status": "memory_error"}
        _begun, ended = _progress(raw_log)
        fatal = [case_id for case_id in order if ended.get(case_id) == "fatal"]
        if fatal:
            # The pass had already stopped at a fatal error; only its exit hung.
            deferred = [case_id for case_id in order if ended.get(case_id) == "deferred"]
            return _inconclusive(record, order, order.index(fatal[0]), sanitize_only, deferred,
                                 "fatal CUDA error without an invalid-access report; exit timed out")
        # A pass that cannot finish establishes nothing about memory safety.
        return record | {"status": "timeout"}
    raw_log = _read_log(log_path)
    record.update(returncode=returncode, report=_sanitizer_report(raw_log))
    if SANITIZER_UNAVAILABLE_MARKER in raw_log:
        # Required mode fails closed: an uninstrumented pass is not a memcheck.
        return record | {"status": "unavailable"}
    if explicit_access_report(raw_log):
        # Decided by the report text, whatever the exit code.
        return record | {"status": "memory_error"}
    try:
        result = _strict_worker_result(result_path, expected_uid=account.pw_uid)
    except RuntimeError as exc:
        if returncode == 0:
            return record | {"status": "failed", "failed_gate": "memory_safety_ok",
                             "private_error": f"sanitizer pass left no structured result: {exc}"}
        # The process died (a fatal error took the interpreter with it):
        # inconclusive from the first case that did not end cleanly.
        _begun, ended = _progress(raw_log)
        for case_id in order:
            gate = {"memory_safety": "memory_safety_ok",
                    "harness_integrity": "harness_integrity_ok"}.get(ended.get(case_id, ""))
            if gate is not None:
                return record | {"status": "failed", "failed_gate": gate,
                                 "private_error": f"sanitizer case {case_id} ended with {ended[case_id]}"}
        index = next(
            (position for position, case_id in enumerate(order)
             if ended.get(case_id) not in ("passed", "deferred")),
            len(order),
        )
        deferred = [case_id for case_id in order if ended.get(case_id) == "deferred"]
        return _inconclusive(record, order, index, sanitize_only, deferred,
                             f"the sanitizer process exited ({returncode}) without a result and "
                             "without an invalid-access report")
    record["result"] = result
    if result.get("schema_version") != SCHEMA_VERSION or result.get("candidate_sha256") not in (None, expected):
        return record | {"status": "failed", "failed_gate": "memory_safety_ok",
                         "private_error": "sanitizer pass evaluated a different candidate source"}
    status = result.get("status")
    if status == "failed":
        gate = result.get("failed_gate")
        return record | {
            "status": "failed",
            "failed_gate": gate if gate in RUNTIME_GATES else "memory_safety_ok",
            "private_error": str(result.get("private_error") or ""),
        }
    rows = result.get("cases")
    if (
        result.get("candidate_sha256") != expected
        or result.get("order") != order
        or not isinstance(rows, list)
        or not rows and status == "fatal"
        or len(rows) > len(order)
        or any(
            not isinstance(row, Mapping)
            or row.get("case_id") != order[index]
            or row.get("status") not in SANITIZE_STATUSES
            for index, row in enumerate(rows)
        )
    ):
        return record | {"status": "failed", "failed_gate": "memory_safety_ok",
                         "private_error": "sanitizer result does not account for the flagged cases in order"}
    deferred = [str(row["case_id"]) for row in rows if row["status"] == "deferred"]
    if status == "fatal" and rows[-1]["status"] == "fatal" and all(
        row["status"] in ("passed", "deferred") for row in rows[:-1]
    ):
        return _inconclusive(record, order, len(rows) - 1, sanitize_only, deferred,
                             "fatal CUDA error without an invalid-access report")
    if status != "completed" or len(rows) != len(order) or any(
        row["status"] not in ("passed", "deferred") for row in rows
    ):
        return record | {"status": "failed", "failed_gate": "memory_safety_ok",
                         "private_error": "sanitizer result status is inconsistent"}
    for row in rows:
        if row["status"] == "passed":
            try:
                _validate_accuracy_metrics(row, f"sanitizer {row['case_id']}")
            except RuntimeError as exc:
                return record | {"status": "failed", "failed_gate": "memory_safety_ok",
                                 "private_error": f"sanitizer result rejected: {exc}"}
    record.update(
        status="passed",
        deferred_case_ids=deferred,
        uncovered_case_ids=[],
        promote_case_ids=[case_id for case_id in deferred if case_id in sanitize_only],
    )
    return record


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


ACCURACY_FIELDS = tuple(
    f"{tensor}_{metric}" for tensor in ("out", "max_logits", "lse")
    for metric in ("max_abs", "max_rel", "cos_diff")
)


def _validate_accuracy_metrics(row: Mapping[str, Any], label: str) -> None:
    """Re-check the reported metrics: finite, and every cosine difference within its limit."""
    for name in ACCURACY_FIELDS:
        value = row.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise RuntimeError(f"{label}.{name} must be a finite number")
        if not name.endswith("_cos_diff") and float(value) < 0.0:
            raise RuntimeError(f"{label}.{name} must be non-negative")
    for tensor, limit in COS_LIMITS.items():
        if abs(float(row[f"{tensor}_cos_diff"])) > limit:
            raise RuntimeError(f"{label}.{tensor}_cos_diff exceeds {limit:g}")


def _numeric_list(value: Any, label: str, expected_length: int) -> list[float]:
    if not isinstance(value, list) or len(value) != expected_length:
        raise RuntimeError(f"{label} has the wrong length")
    return [finite_positive(item, label) for item in value]


def _expected_stream_checks(calls_total: int, calls_run: int) -> int:
    return sum(
        1 for number in range(calls_run)
        if number % STREAM_CHECK_EVERY == 0 or number == calls_total - 1
    )


def _validate_case_result(
    row: Mapping[str, Any],
    expected: Mapping[str, Any],
    expected_trials: int,
    bound: float | None,
) -> float:
    case_id = str(expected["case_id"])
    family = str(expected["family"])
    group = str(expected["group"])
    stream = group in STREAM_GROUPS
    for name in ("case_id", "family", "group"):
        if row.get(name) != expected[name]:
            raise RuntimeError(f"case identity differs for {case_id}")
    label = f"case {case_id} geometry"
    d_qk, h_q, topk = FAMILY_GEOMETRY[family][0], int(expected["h_q"]), FAMILY_TOPK[family]
    if row.get("scored") is not True or row.get("stream") is not stream:
        raise RuntimeError(f"case {case_id} is not marked scored or has the wrong kind")
    if row.get("gated") is not bool(expected["gated"]) or row.get("tier") != GROUP_TIER[group]:
        raise RuntimeError(f"case {case_id} gate or tier differs")
    if row.get("entrypoint") != PREFILL_ENTRYPOINT:
        raise RuntimeError(f"entrypoint identity differs for {case_id}")
    if row.get("model") != FAMILY_MODEL[family]:
        raise RuntimeError(f"model identity differs for {case_id}")
    for name, value in (
        ("s_q", int(expected["s_q"])),
        ("s_kv", int(expected["s_kv"])),
        ("topk", topk),
        ("h_q", h_q),
        ("d_qk", d_qk),
    ):
        _exact_int(row, name, value, label)
    wanted_scale = ANCHOR_SM_SCALE if group == "anchor" else FAMILY_SM_SCALE[family]
    scale = row.get("sm_scale")
    if isinstance(scale, bool) or not isinstance(scale, float) or scale != wanted_scale:
        raise RuntimeError(f"scored sm_scale differs for {case_id}")
    per_position = 2.0 * h_q * (d_qk + D_V)

    if stream:
        calls = expected["calls"]
        _exact_int(row, "calls_total", len(calls), label)
        _exact_int(row, "warmup_calls", STREAM_WARM_CALLS, label)
        run = row.get("calls_run")
        if isinstance(run, bool) or not isinstance(run, int) or not 1 <= run <= len(calls):
            raise RuntimeError(f"stream call count is invalid for {case_id}")
        if row.get("truncated") is not (run < len(calls)):
            raise RuntimeError(f"stream truncation flag differs for {case_id}")
        seconds = finite_positive(row.get("stream_seconds"), f"{case_id}.stream_seconds")
        if run < len(calls) and seconds <= STREAM_TIME_LIMIT_SECONDS:
            raise RuntimeError(f"stream {case_id} stopped before its time limit")
        positions = row.get("stream_positions")
        ceiling = sum(call[0] for call in calls[:run]) * topk
        if isinstance(positions, bool) or not isinstance(positions, int) or not 0 < positions <= ceiling:
            raise RuntimeError(f"stream positions are invalid for {case_id}")
        _exact_int(row, "checked_calls", _expected_stream_checks(len(calls), run), label)
        computed_case = per_position * positions / seconds / 1.0e12
    else:
        _exact_int(row, "decoy_precalls", expected_trials, label)
        _exact_int(row, "checked_calls", 1 if expected["gated"] else 0, label)
        trial_us = _numeric_list(row.get("trial_us"), f"{case_id}.trial_us", expected_trials)
        host_us = _numeric_list(row.get("trial_host_us"), f"{case_id}.trial_host_us", expected_trials)
        event_us = _numeric_list(row.get("trial_event_us"), f"{case_id}.trial_event_us", expected_trials)
        trial_tflops = _numeric_list(row.get("trial_tflops"), f"{case_id}.trial_tflops", expected_trials)
        trial_positions = row.get("trial_positions")
        if not isinstance(trial_positions, list) or len(trial_positions) != expected_trials:
            raise RuntimeError(f"position telemetry differs for {case_id}")
        ceiling = int(expected["s_q"]) * topk
        for ordinal, value in enumerate(trial_positions):
            if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= ceiling:
                raise RuntimeError(f"counted positions are invalid for {case_id}")
            if group == "anchor" and value != ceiling:
                raise RuntimeError(f"anchor positions differ for {case_id}")
            if not math.isclose(
                trial_us[ordinal], max(host_us[ordinal], event_us[ordinal]), rel_tol=1.0e-9, abs_tol=1.0e-9
            ):
                raise RuntimeError(f"trial clock differs from max(host, event) for {case_id}")
            computed = per_position * value / (trial_us[ordinal] * 1.0e-6) / 1.0e12
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
    if stream or expected["gated"]:
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


def native_edge_cases(
    expected_edges: Sequence[Mapping[str, Any]], promoted: Sequence[str] = ()
) -> list[Mapping[str, Any]]:
    chosen = set(promoted)
    return [case for case in expected_edges if case["native"] or case["case_id"] in chosen]


def _validate_edge_record(
    edge: Any, expected_edges: Sequence[Mapping[str, Any]], promoted: Sequence[str]
) -> int:
    if not isinstance(edge, Mapping):
        raise RuntimeError("runtime result lacks the edge record")
    native = native_edge_cases(expected_edges, promoted)
    for name, value in (
        ("cases_total", len(native)),
        ("cases_skipped_sanitize_only", len(expected_edges) - len(native)),
        ("cases_passed", len(native)),
        ("cases_failed", 0),
    ):
        _exact_int(edge, name, value, "edge-case record")
    if edge.get("cases_promoted") != sorted(promoted):
        raise RuntimeError("edge-case record differs for the promoted sanitizer-only cases")
    _validate_edge_rows(edge.get("results"), native, "edge-case")
    return len(native)


def expected_protocol(bound: float | None) -> dict[str, Any]:
    return {
        "decoy_precall": True,
        "edge_calls": EDGE_CALLS,
        "guard_min_bytes": GUARD_MIN_BYTES,
        "guard_rows": GUARD_ROWS,
        "soft_landing": SOFT_LANDING,
        "l2_flush_bytes": L2_FLUSH_BYTES,
        "stream_warm_calls": STREAM_WARM_CALLS,
        "stream_check_every": STREAM_CHECK_EVERY,
        "stream_time_limit_seconds": STREAM_TIME_LIMIT_SECONDS,
        "stale_tail_fraction": STALE_TAIL_FRACTION,
        "out_tolerance": [OUT_ABS_TOL, OUT_REL_TOL, OUT_COS_TOL],
        "logit_tolerance": [LOGIT_ABS_TOL, LOGIT_REL_TOL, LOGIT_COS_TOL],
        "plausibility_bound_tflops": bound,
    }


def _validate_protocol(protocol: Any, bound: float | None) -> None:
    expected = expected_protocol(bound)
    if not isinstance(protocol, Mapping) or set(protocol) != set(expected) or any(
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
    promoted: Sequence[str] = (),
) -> tuple[float, dict[str, float], dict[str, float]]:
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
    if not isinstance(raw_entrypoints, list) or raw_entrypoints != [PREFILL_ENTRYPOINT]:
        raise RuntimeError("runtime entrypoint set differs")
    native_edges = _validate_edge_record(result.get("edge"), expected_edges, promoted)

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

    # One sentinel check after import, one per edge call; per single case one per
    # warm-up and two per trial (decoy and measured call); per stream one per warm-up
    # call and one per timed call.  The runtime may check more.
    singles = [case for case in expected_cases if case["group"] not in STREAM_GROUPS]
    stream_calls = 0
    for case in expected_cases:
        if case["group"] in STREAM_GROUPS:
            run = by_id[str(case["case_id"])].get("calls_run")
            stream_calls += STREAM_WARM_CALLS + (run if isinstance(run, int) and not isinstance(run, bool) else 0)
    minimum_checks = 1 + EDGE_CALLS * native_edges + len(singles) * (
        (expected_warmups or 1) + 2 * expected_trials
    ) + stream_calls
    checks = result.get("sentinel_checks")
    if isinstance(checks, bool) or not isinstance(checks, int) or checks < minimum_checks:
        raise RuntimeError("runtime harness sentinel did not run on every call")

    cells: dict[str, dict[str, list[float]]] = {family: {} for family in FAMILIES}
    for expected in expected_cases:
        value = _validate_case_result(by_id[str(expected["case_id"])], expected, expected_trials, bound)
        cells[str(expected["family"])].setdefault(str(expected["group"]), []).append(value)
    groups: dict[str, float] = {}
    tiers: dict[str, float] = {}
    families: dict[str, float] = {}
    for family in FAMILIES:
        by_group = {group: geometric_mean(values) for group, values in cells[family].items()}
        for group, value in by_group.items():
            groups[f"{family}:{group}"] = value
        family_tiers = []
        for tier in TIERS:
            members = [value for group, value in by_group.items() if GROUP_TIER[group] == tier]
            if not members:
                raise RuntimeError(f"family {family} lacks the {tier} tier")
            tiers[f"{family}:{tier}"] = geometric_mean(members)
            family_tiers.append(tiers[f"{family}:{tier}"])
        families[family] = geometric_mean(family_tiers)
    primary = geometric_mean([families[family] for family in FAMILIES])

    for key, expected_values in (("family_tflops", families), ("tier_tflops", tiers), ("group_tflops", groups)):
        raw = result.get(key)
        if not isinstance(raw, Mapping) or set(raw) != set(expected_values):
            raise RuntimeError(f"runtime {key} set differs")
        for label, expected_value in expected_values.items():
            observed = finite_positive(raw.get(label), f"{label} runtime aggregate")
            if not math.isclose(observed, expected_value, rel_tol=1.0e-9, abs_tol=1.0e-12):
                raise RuntimeError(f"runtime {key} differs for {label}")
    observed_primary = finite_positive(result.get("primary_tflops"), "runtime primary score")
    if not math.isclose(observed_primary, primary, rel_tol=1.0e-9, abs_tol=1.0e-12):
        raise RuntimeError("runtime primary score differs")
    return primary, families, {"tiers": tiers, "groups": groups}


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
        "sanitizer_inconclusive": 0.0,
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
    run_root = Path(tempfile.mkdtemp(prefix="standalone-sparse-mla-v2-", dir="/tmp"))
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
        promoted: list[str] = []
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
            if sanitizer.get("status") not in ("passed", "inconclusive"):
                gate = str(sanitizer.get("failed_gate") or "memory_safety_ok")
                gate = gate if gate in GATE_NAMES else "memory_safety_ok"
                _mark_failed_gate(base, gate)
                base["private_error"] = _sanitizer_failure_message(sanitizer)
                return 1, base
            base["sanitizer_ran"] = 1.0
            base["sanitizer_inconclusive"] = 1.0 if sanitizer["status"] == "inconclusive" else 0.0
            promoted = [str(case_id) for case_id in sanitizer.get("promote_case_ids", [])]
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
            promoted=promoted,
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
        score, families, breakdown = validate_runtime_result(
            runtime_result,
            cases,
            edge_cases,
            expected_trials=args.trials,
            expected_warmups=args.warmups,
            promoted=promoted,
        )

        detail = dict(base)
        detail.update(
            {
                "score_tflops": score,
                **{f"{family}_tflops": families[family] for family in FAMILIES},
                "selected_source_sha256": admission["selected_sha256"],
                "score_method": (
                    "geomean over v32, glm5, v4_flash, v4_pro of sqrt(evidence x sweep), each tier "
                    "the geomean of its groups and each group the geomean of its cases' TFLOPS"
                ),
                "wall_time_seconds": time.monotonic() - started,
                "hardware": hardware,
                "tier_tflops": breakdown["tiers"],
                "group_tflops": breakdown["groups"],
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
        default=int(os.environ.get("STANDALONE_SPARSE_MLA_VERIFY_TRIALS", "3")),
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=int(os.environ.get("STANDALONE_SPARSE_MLA_VERIFY_WARMUPS", "1")),
    )
    parser.add_argument(
        "--candidate-timeout",
        type=int,
        default=int(os.environ.get("STANDALONE_SPARSE_MLA_CANDIDATE_TIMEOUT", "3600")),
    )
    parser.add_argument(
        "--sanitizer-timeout",
        type=int,
        default=int(os.environ.get("STANDALONE_SPARSE_MLA_SANITIZER_TIMEOUT", "780")),
    )
    parser.add_argument(
        "--sanitizer-mode",
        default=os.environ.get("STANDALONE_SPARSE_MLA_SANITIZER_MODE", "required"),
    )
    parser.add_argument(
        "--total-timeout",
        type=int,
        default=int(os.environ.get("STANDALONE_SPARSE_MLA_TOTAL_TIMEOUT", "4800")),
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
