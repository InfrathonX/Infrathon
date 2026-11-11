#!/usr/bin/env python3
"""Trusted runtime for the LMCache paged-KV block transfer H100 task.

The Judge worker runs this file; the Work image's public runtime is derived
from it by ``scripts/make_public_runtime.py``.  It owns the frozen ABI, case
validation, the independent bit-exact reference, the poison-canary protocol,
the timing protocol (including the overlap foreground), and the score.  A
candidate never imports it.

v2 workloads (``docs``: see the task README): one engine pool per case,
sized from real per-GPU model shapes; block-id tables replayed from serving
traces; large and small calls scored as separate families; and an overlap
family that charges a transfer for the foreground time it costs.

The operator is upstream LMCache's ``multi_layer_block_kv_transfer``
(``csrc/cuda/mp_mem_kernels.cu`` @ f9addd2e4e074f21f26cbda84c3abd932d32ef33):
a block-granular gather/scatter between a serving engine's paged per-layer KV
buffers and LMCache's contiguous token-major chunk objects
(``[kv, NL, T, NH*HS]``, "2LTD").  Two layout families are frozen:

- ``split``: per-layer ``[2, NB, NH, BS, HS]``  (``NL_X_TWO_NB_NH_BS_HS``)
- ``fused``: per-layer ``[NB, NH, BS, CS]``, CS = 2*HS, kv_size == 1
  (``NL_X_NB_NH_BS_CS``)

This is raw byte movement.  The destination must be **bit-exact** — BF16 bit
patterns include NaN payloads, infinities, and signed zeros, and tolerance is
zero.  Sources, block-id tables, pointer arrays, skipped prefix blocks, and
every untargeted destination byte must be untouched.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

ENTRYPOINT = "lmcache_paged_kv_block_transfer"

# --------------------------------------------------------------------------
# Frozen geometry constants
# --------------------------------------------------------------------------
BS = 16            # engine page size in tokens (vLLM default)
HEAD_SIZES = (64, 128)  # true head sizes in elements (gpt-oss uses 64)
ESIZE = 2          # bf16 storage bytes
POISON = 0x7C7C    # int16 poison pattern for destinations and scratch
SCRATCH_BYTES = 256 * 1024 * 1024

# Settling barrier between preparing a trial's buffers and timing it.
# Refilling multi-gigabyte pinned buffers leaves deferred work behind (CPU
# writeback of the freshly written pinned pages, allocator bookkeeping).  On a
# CPU-quota-limited evaluation container that work drains slowly enough to
# overlap the timed region and inflate it 2-4.5x at random; on an unthrottled
# host it drains before the timer starts, which is why the defect is invisible
# outside a container.  Measured in the work image with --cpus=8 on one trace
# case: no barrier -> [95.8, 43.3, 50.7, 55.9, 49.5] ms; 0.2 s -> a flat
# [20.6, 20.7, 20.6, 20.7, 20.6].
#
# The barrier keeps the GPU busy rather than sleeping.  A plain sleep drains
# the same work but lets the device fall out of its boost state: measured on
# one device case, sleep gave [702.5, 1289.0, 1287.8, 1270.2, 1270.4] GB/s --
# still an outlier and ~4% low -- while an equally long busy barrier gave
# [1323.6, 1327.9, 1325.5, 1326.3, 1325.9].  Trace cases are flat either way.
SETTLE_SECONDS = 0.10
# v2 refills pinned chunks with device-to-host copies, not CPU writes, so far
# less deferred host work is left behind: on 2026-11-08 (H100 SXM, 32 CPUs)
# host cases measured identically at 0.25 s and 0.10 s (e.g. 52.3 / 53.0 /
# 52.3 / 52.3 GB/s both ways, trial CV <= 0.05 %).
# Device-only cases refill nothing on the host; a short busy barrier is
# enough to keep the device clocked.
DEVICE_SETTLE_SECONDS = 0.05

DIR_STORE = 0      # engine paged buffers -> chunk objects (upstream D2H)
DIR_RETRIEVE = 1   # chunk objects -> engine paged buffers (upstream H2D)
LAYOUT_SPLIT = 0
LAYOUT_FUSED = 1

LAYOUT_NAMES = {LAYOUT_SPLIT: "split", LAYOUT_FUSED: "fused"}
LAYOUT_IDS = {name: value for value, name in LAYOUT_NAMES.items()}
DIRECTION_NAMES = {DIR_STORE: "store", DIR_RETRIEVE: "retrieve"}
DIRECTION_IDS = {name: value for value, name in DIRECTION_NAMES.items()}

# ABI validity ranges (edge fixtures probe the extremes).  NL reaches 128 so
# a deep model's full layer stack (80-126 layers) fits in one call.
NL_RANGE = (1, 128)
NH_RANGE = (1, 32)
NB_RANGE = (1, 65536)
OBJECTS_RANGE = (1, 4)
DEFAULT_TRIALS = 3

# Physical bandwidth envelope: a case whose implied bandwidth exceeds
# envelope * margin is physically impossible and invalidates the whole run —
# this is the backstop against a native candidate forging its own timing.
# The envelope is measured on the evaluation device *in the same run* by the
# trusted probe in measure_envelopes() (contiguous D2D copy payload and pinned
# H2D/D2H cudaMemcpyAsync), before the candidate library is loaded.  It is
# never frozen from one host: the qualification host (H100 NVL, PCIe Gen5 x16)
# measured 1763 GB/s device / 57.6 GB/s host, kept below as provenance only.
ENVELOPE_MARGIN = 1.10
ENVELOPE_PROTOCOL = "same-run-cudamemcpy-zerocopy-duplex-probe-v3"
ENVELOPE_PROBE_BYTES = 256 * 1024 * 1024
ENVELOPE_PROBE_REPEATS = 5
# v2: the host ceiling is the PCIe link, not the copy engine.  cudaMemcpy
# through the copy engines and an SM-driven zero-copy stream over pinned
# memory (what the seed and any UVA candidate actually do) reach different
# fractions of the link on different platforms -- on a PCIe Gen3 host the
# copy engine measured 8 GB/s while the zero-copy kernel reached 12 GB/s, so
# the v1 copy-engine-only probe rejected a correct candidate.  The host
# envelope is now the best of both mechanisms in either direction.
ENVELOPE_ZEROCOPY_SOURCE = r"""
#include <cuda_runtime.h>
#include <stdint.h>
// One 16-byte element per thread and one block per 256 elements: over a
// PCIe link the rate is set by how many reads are in flight, and a small
// grid-stride grid measured about half of what this shape reaches (7 vs
// 12.3 GB/s on a Gen3 x16 host); the wide launch is what candidates do.
__global__ void zerocopy_stream_kernel(const uint4* __restrict__ src,
                                       uint4* __restrict__ dst, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) dst[i] = src[i];
}
extern "C" int envelope_zerocopy_stream(const void* src, void* dst,
                                        size_t bytes, void* stream) {
  size_t n = bytes / sizeof(uint4);
  unsigned int blocks = (unsigned int)((n + 255) / 256);
  zerocopy_stream_kernel<<<blocks, 256, 0, (cudaStream_t)stream>>>(
      (const uint4*)src, (uint4*)dst, n);
  return (int)cudaGetLastError();
}
"""
# Sanity floors for the probe: every GPU and PCIe link the task admits clears
# them by a wide margin, so a probe below them means the measurement is broken.
ENVELOPE_MIN_DEVICE_GBPS = 100.0
ENVELOPE_MIN_HOST_GBPS = 4.0
QUALIFICATION_ENVELOPE_GBPS = {"device": 1763.0, "host": 57.6}  # H100 NVL provenance, not a gate

# Weighted geometric mean across the four scored groups (device is the
# geometric mean of its large-call and small-call families; host also folds in
# the full-duplex family).
# Large calls reach 94-99% of the copy envelopes with a well-tuned kernel
# (2026-11-07 pilot); small calls and overlap are where the room is, so the
# device group no longer carries most of the weight.
GROUP_WEIGHTS = {"device": 0.35, "host": 0.25, "trace": 0.10, "overlap": 0.30}

# Overlap cases: the trusted foreground runs this many times the transfer's
# own duration, so it covers the whole transfer; the foreground time lost per
# second of transfer is clamped to [0, MAX_OVERLAP_LOSS] and the case scores
# GB/s * (1 - loss).
OVERLAP_COVER = 3.0
MAX_OVERLAP_LOSS = 0.95

# Recorded in every result and checked by the Judge verifier.
RECORD_CONSTANTS = {
    "bs": BS,
    "head_sizes": list(HEAD_SIZES),
    "esize": ESIZE,
    "poison": POISON,
    "scratch_bytes": SCRATCH_BYTES,
    "envelope_margin": ENVELOPE_MARGIN,
    "envelope_protocol": ENVELOPE_PROTOCOL,
    "envelope_probe_bytes": ENVELOPE_PROBE_BYTES,
    "small_call_bytes": 16 * 1024 * 1024,
    "overlap_cover": OVERLAP_COVER,
    "max_overlap_loss": MAX_OVERLAP_LOSS,
}

# --------------------------------------------------------------------------
# Frozen compile command (identical for the public runner and the Judge).
# Only the target architecture varies: it is the native architecture of the
# GPU allocated to the run, so the same source builds on every admitted card.
# --------------------------------------------------------------------------
NVCC = "/usr/local/cuda/bin/nvcc"


def native_arch(torch: Any) -> str:
    """``sm_XY`` (``sm_XYa`` on Hopper and newer) for the visible GPU."""
    major, minor = (int(v) for v in torch.cuda.get_device_capability(0))
    return f"sm_{major}{minor}" + ("a" if major >= 9 else "")


def compile_argv(source: Path, output: Path, include_dir: Path, arch: str) -> list[str]:
    if not re.fullmatch(r"sm_[0-9]{2,3}a?", arch):
        raise ValueError(f"unexpected nvcc architecture {arch!r}")
    return [
        NVCC,
        "-O3",
        "-std=c++17",
        f"-arch={arch}",
        "--shared",
        "--cudart=shared",
        "-Xcompiler",
        "-fPIC",
        f"-I{include_dir}",
        "-o",
        str(output),
        str(source),
    ]


def compile_candidate(
    source: Path, out_dir: Path, include_dir: Path, *, arch: str, timeout: float = 300.0
) -> tuple[Path, str]:
    """Compile one submission with the frozen command; return (.so, log)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "submission.so"
    process = subprocess.run(
        compile_argv(Path(source), output, Path(include_dir), arch),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    log = (process.stdout or "") + (process.stderr or "")
    if process.returncode != 0 or not output.is_file():
        raise RuntimeError(f"candidate compilation failed:\n{log[-4000:]}")
    return output, log


# Undefined dynamic symbols a compiled submission may import.  Default-deny:
# a symbol must be on one of these lists or the library never loads.  The CUDA
# runtime API surface is an *exact* allowlist — cudaMalloc*, cudaFree, private
# stream/event creation, device-wide synchronisation, graphs/capture, peer
# copies, host callbacks, and the whole driver API (cu*) are simply absent.
ALLOWED_LIBC_SYMBOLS = frozenset(
    {
        "__cxa_atexit", "__cxa_finalize", "__gmon_start__",
        "_ITM_registerTMCloneTable", "_ITM_deregisterTMCloneTable",
        "abort", "free", "malloc", "memcpy", "memmove", "memset", "strlen",
        "__stack_chk_fail", "__assert_fail",
        "__gxx_personality_v0", "_Unwind_Resume",
    }
)
ALLOWED_CUDA_SYMBOLS = frozenset(
    {
        # launches
        "cudaLaunchKernel", "cudaLaunchKernelExC", "cudaLaunchCooperativeKernel",
        # async data movement on a caller-provided stream
        "cudaMemcpyAsync", "cudaMemcpy2DAsync", "cudaMemcpy3DAsync",
        "cudaMemsetAsync", "cudaMemset2DAsync",
        # error queries
        "cudaGetLastError", "cudaPeekAtLastError",
        "cudaGetErrorString", "cudaGetErrorName",
        # occupancy / attribute introspection and shared-memory opt-in
        "cudaFuncSetAttribute", "cudaFuncGetAttributes",
        "cudaFuncSetCacheConfig",
        "cudaOccupancyMaxActiveBlocksPerMultiprocessor",
        "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
        "cudaDeviceGetAttribute", "cudaGetDevice", "cudaGetDeviceProperties",
        "cudaGetDeviceProperties_v2",
    }
)
# nvcc registration glue emitted for every fatbinary under --cudart=shared.
ALLOWED_UNDEFINED_PREFIXES = ("__cuda", "_ZSt", "_ZNSt", "_Znwm", "_ZdlPv")


def inspect_binary(library: Path) -> dict[str, Any]:
    """Undefined-symbol audit of a compiled submission, before it is loaded.

    Runs readelf on the file only; no candidate code executes here.
    """
    process = subprocess.run(
        ["readelf", "--dyn-syms", "--wide", str(library)],
        capture_output=True,
        text=True,
        timeout=60.0,
        check=True,
    )
    undefined: list[str] = []
    for line in process.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 8 and parts[6] == "UND" and parts[7]:
            name = parts[7].split("@", 1)[0]
            if name:
                undefined.append(name)
    problems: list[str] = []
    for name in sorted(set(undefined)):
        if name in ALLOWED_LIBC_SYMBOLS or name in ALLOWED_CUDA_SYMBOLS:
            continue
        if name.startswith(ALLOWED_UNDEFINED_PREFIXES):
            continue
        problems.append(f"undefined symbol outside the allowlist: {name}")
    return {
        "ok": not problems,
        "undefined_symbols": sorted(set(undefined)),
        "problems": problems,
    }


# --------------------------------------------------------------------------
# ABI (must match transfer_abi.h exactly)
# --------------------------------------------------------------------------
class TransferParams(ctypes.Structure):
    _fields_ = [
        ("paged_layer_ptrs", ctypes.c_uint64),
        ("chunk_ptrs", ctypes.c_uint64 * 4),
        ("block_ids", ctypes.c_uint64),
        ("num_objects", ctypes.c_int32),
        ("total_blocks", ctypes.c_int32),
        ("blocks_per_object", ctypes.c_int32),
        ("skip_prefix_n_blocks", ctypes.c_int32),
        ("direction", ctypes.c_int32),
        ("layout", ctypes.c_int32),
        ("kv_size", ctypes.c_int32),
        ("nl", ctypes.c_int32),
        ("nb", ctypes.c_int32),
        ("bs", ctypes.c_int32),
        ("nh", ctypes.c_int32),
        ("hs", ctypes.c_int32),
        ("element_size", ctypes.c_int32),
        ("chunk_tokens", ctypes.c_int32),
        ("chunk_side_is_host", ctypes.c_int32),
        ("_pad0", ctypes.c_int32),
        ("device_scratch", ctypes.c_uint64),
        ("device_scratch_bytes", ctypes.c_int64),
    ]


assert ctypes.sizeof(TransferParams) == 128, ctypes.sizeof(TransferParams)


# --------------------------------------------------------------------------
# Trusted timing primitives, captured before any candidate library is loaded
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TrustedClock:
    perf_counter: Callable[[], float]
    synchronize: Callable[..., Any]
    current_stream: Callable[..., Any]
    event: Callable[..., Any]

    @classmethod
    def capture(cls, torch: Any) -> "TrustedClock":
        return cls(
            perf_counter=time.perf_counter,
            synchronize=torch.cuda.synchronize,
            current_stream=torch.cuda.current_stream,
            event=torch.cuda.Event,
        )


def finite_positive(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise RuntimeError(f"{label} is not a finite positive number: {value!r}")
    return number


def geometric_mean(values: Iterable[float]) -> float:
    items = [finite_positive(value, "geometric mean term") for value in values]
    if not items:
        raise RuntimeError("geometric mean of an empty sequence")
    return math.exp(math.fsum(math.log(item) for item in items) / len(items))

# --------------------------------------------------------------------------
# Wave and case descriptors
# --------------------------------------------------------------------------
# A wave is one ABI call.  Its block-id table is explicit: the Judge's tables
# come from replaying a serving trace through a paged block manager, so they
# carry the run structure of a live engine pool.  Every trial rotates the
# whole table by one offset (mod NB), which keeps that structure, keeps the
# ids unique, and moves every address between trials.
WAVE_KEYS = {
    "layout", "nl", "nh", "hs", "chunk_tokens", "num_objects", "direction",
    "skip", "nb", "ids",
}
CASE_KEYS = {"case_id", "family", "arm", "foreground", "stratum", "tier", "waves", "trials", "seed"}
POOL_KEYS = ("layout", "nl", "nh", "hs", "nb")
# A direct case repeats one call class; the skipped prefix may differ per call
# because it is whatever part of each request the GPU already holds.
DIRECT_KEYS = ("layout", "nl", "nh", "hs", "chunk_tokens", "num_objects",
               "direction", "nb")

# A duplex case stores and retrieves at the same time on two streams, over
# disjoint blocks of one pool: the full-duplex PCIe link the LMCache paper
# uses for parallel store and load (§5.1).
DUPLEX_KEYS = ("layout", "nl", "nh", "hs", "chunk_tokens", "num_objects", "nb")
ARMS = ("device", "host", "trace", "overlap", "duplex")
FAMILIES = {
    "device-large": "device",
    "device-small": "device",
    "host-large": "host",
    "host-small": "host",
    "host-duplex": "duplex",
    "trace": "trace",
    "overlap": "overlap",
}
GROUP_FAMILIES = {
    "device": ("device-large", "device-small"),
    "host": ("host-large", "host-small", "host-duplex"),
    "trace": ("trace",),
    "overlap": ("overlap",),
}
# Within a family every shape stratum counts once, and each tier (the paper's
# evaluation models, other models) counts once, so neither the number of
# cases per shape nor the number of extrapolated shapes moves the score.
TIERS = ("paper", "other", "public", "edge")
STRATUM_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,40}$")
# A call that moves less than this is a small call: one chunk of a deep
# tensor-parallel split, a request's tail chunk, a chunk stored during decode.
# Its time is dominated by fixed per-call costs, so it is scored separately
# and cannot be averaged away by large calls.
SMALL_CALL_BYTES = 16 * 1024 * 1024
FOREGROUNDS = ("gemm", "hbm")
MAX_DIRECT_WAVES = 128
MAX_TRACE_WAVES = 64
MAX_CASE_PAYLOAD_BYTES = 4 * 1024**3
MAX_POOL_BYTES = 44 * 1000**3
HOST_ARENA_BYTES = 4 * 1024**3 + 64 * 1024**2
ARENA_ALIGN = 2 * 1024 * 1024


def _int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def validate_wave(raw: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != WAVE_KEYS:
        raise ValueError(f"wave fields differ from schema: {sorted(raw)}")
    layout = str(raw["layout"])
    if layout not in LAYOUT_IDS:
        raise ValueError(f"unknown layout {layout!r}")
    direction = str(raw["direction"])
    if direction not in DIRECTION_IDS:
        raise ValueError(f"unknown direction {direction!r}")
    nl = _int(raw["nl"], "nl")
    nh = _int(raw["nh"], "nh")
    hs = _int(raw["hs"], "hs")
    nb = _int(raw["nb"], "nb")
    chunk_tokens = _int(raw["chunk_tokens"], "chunk_tokens")
    num_objects = _int(raw["num_objects"], "num_objects")
    skip = _int(raw["skip"], "skip")
    if not NL_RANGE[0] <= nl <= NL_RANGE[1]:
        raise ValueError(f"nl outside {NL_RANGE}: {nl}")
    if not NH_RANGE[0] <= nh <= NH_RANGE[1]:
        raise ValueError(f"nh outside {NH_RANGE}: {nh}")
    if hs not in HEAD_SIZES:
        raise ValueError(f"hs outside {HEAD_SIZES}: {hs}")
    if not NB_RANGE[0] <= nb <= NB_RANGE[1]:
        raise ValueError(f"nb outside {NB_RANGE}: {nb}")
    if not OBJECTS_RANGE[0] <= num_objects <= OBJECTS_RANGE[1]:
        raise ValueError(f"num_objects outside {OBJECTS_RANGE}: {num_objects}")
    if chunk_tokens <= 0 or chunk_tokens % BS != 0:
        raise ValueError(f"chunk_tokens must be a positive multiple of {BS}")
    total_blocks = num_objects * (chunk_tokens // BS)
    if total_blocks > nb:
        raise ValueError("total blocks exceed the paged pool")
    if not 0 <= skip <= total_blocks:
        raise ValueError("skip outside [0, total_blocks]")
    ids = raw["ids"]
    if not isinstance(ids, list) or len(ids) != total_blocks:
        raise ValueError("ids must list exactly one block id per flat block")
    ids = [_int(value, "block id") for value in ids]
    if min(ids) < 0 or max(ids) >= nb or len(set(ids)) != len(ids):
        raise ValueError("block ids must be unique and inside the pool")
    return {
        "layout": layout,
        "nl": nl,
        "nh": nh,
        "hs": hs,
        "chunk_tokens": chunk_tokens,
        "num_objects": num_objects,
        "direction": direction,
        "skip": skip,
        "nb": nb,
        "ids": ids,
    }


def wave_geometry(wave: Mapping[str, Any]) -> dict[str, int]:
    layout_id = LAYOUT_IDS[str(wave["layout"])]
    kv = 2 if layout_id == LAYOUT_SPLIT else 1
    width = int(wave["hs"]) if layout_id == LAYOUT_SPLIT else 2 * int(wave["hs"])
    blocks_per_object = int(wave["chunk_tokens"]) // BS
    total_blocks = int(wave["num_objects"]) * blocks_per_object
    return {
        "layout_id": layout_id,
        "kv": kv,
        "hs": width,
        "hidden": int(wave["nh"]) * width,
        "blocks_per_object": blocks_per_object,
        "total_blocks": total_blocks,
    }


def wave_logical_bytes(wave: Mapping[str, Any]) -> int:
    geometry = wave_geometry(wave)
    moved_blocks = geometry["total_blocks"] - int(wave["skip"])
    return (
        moved_blocks * geometry["kv"] * int(wave["nl"]) * BS
        * int(wave["nh"]) * geometry["hs"] * ESIZE
    )


def wave_object_bytes(wave: Mapping[str, Any]) -> int:
    geometry = wave_geometry(wave)
    return geometry["kv"] * int(wave["nl"]) * int(wave["chunk_tokens"]) * geometry["hidden"] * ESIZE


def pool_bytes(wave: Mapping[str, Any]) -> int:
    geometry = wave_geometry(wave)
    return (
        geometry["kv"] * int(wave["nl"]) * int(wave["nb"]) * int(wave["nh"])
        * BS * geometry["hs"] * ESIZE
    )


def case_payload_bytes(case: Mapping[str, Any]) -> int:
    return sum(wave_object_bytes(wave) * int(wave["num_objects"]) for wave in case["waves"])


def case_logical_bytes(case: Mapping[str, Any]) -> int:
    return sum(wave_logical_bytes(wave) for wave in case["waves"])


def case_host_chunks(case: Mapping[str, Any]) -> bool:
    return str(case["arm"]) in ("host", "trace", "overlap", "duplex")


def validate_case(raw: Mapping[str, Any], *, scored: bool) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) - {"scored"} != CASE_KEYS:
        raise ValueError(f"case fields differ from schema: {sorted(raw)}")
    case_id = str(raw["case_id"])
    family = str(raw["family"])
    arm = str(raw["arm"])
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    if family not in FAMILIES or FAMILIES[family] != arm:
        raise ValueError(f"case {case_id}: family {family!r} does not belong to arm {arm!r}")
    foreground = raw["foreground"]
    if arm == "overlap":
        if foreground not in FOREGROUNDS:
            raise ValueError(f"case {case_id}: overlap needs a foreground in {FOREGROUNDS}")
    elif foreground is not None:
        raise ValueError(f"case {case_id}: only overlap cases name a foreground")
    waves_raw = raw["waves"]
    if not isinstance(waves_raw, list) or not waves_raw:
        raise ValueError(f"case {case_id} has no waves")
    waves = [validate_wave(wave) for wave in waves_raw]
    keys = POOL_KEYS if arm == "trace" else (DUPLEX_KEYS if arm == "duplex" else DIRECT_KEYS)
    if len({tuple(wave[key] for key in keys) for wave in waves}) != 1:
        raise ValueError(f"case {case_id}: waves must share {keys}")
    if arm == "duplex":
        stores = {b for w in waves if w["direction"] == "store" for b in w["ids"]}
        loads = {b for w in waves if w["direction"] == "retrieve" for b in w["ids"]}
        if not stores or not loads:
            raise ValueError(f"case {case_id}: a duplex case needs stores and retrieves")
        if stores & loads:
            raise ValueError(f"case {case_id}: duplex stores and retrieves must use disjoint blocks")
    stratum = str(raw["stratum"])
    tier = str(raw["tier"])
    if STRATUM_RE.fullmatch(stratum) is None or tier not in TIERS:
        raise ValueError(f"case {case_id}: stratum or tier differs from the schema")
    limit = MAX_TRACE_WAVES if arm == "trace" else MAX_DIRECT_WAVES
    if len(waves) > limit:
        raise ValueError(f"case {case_id} exceeds {limit} waves")
    trials = _int(raw["trials"], "trials")
    if not 1 <= trials <= 32:
        raise ValueError("trials outside [1, 32]")
    seed = _int(raw["seed"], "seed")
    case = {
        "case_id": case_id,
        "family": family,
        "arm": arm,
        "foreground": foreground,
        "stratum": stratum,
        "tier": tier,
        "waves": waves,
        "trials": trials,
        "seed": seed,
        "scored": bool(scored),
    }
    if case_payload_bytes(case) > MAX_CASE_PAYLOAD_BYTES:
        raise ValueError(f"case {case_id}: chunk payload exceeds {MAX_CASE_PAYLOAD_BYTES} bytes")
    if case_host_chunks(case) and case_payload_bytes(case) > HOST_ARENA_BYTES - 4 * ARENA_ALIGN * len(waves):
        raise ValueError(f"case {case_id}: host chunks exceed the pinned arena")
    if pool_bytes(waves[0]) > MAX_POOL_BYTES:
        raise ValueError(f"case {case_id}: paged pool exceeds {MAX_POOL_BYTES} bytes")
    if scored and family in ("device-large", "device-small", "host-large", "host-small"):
        small = family.endswith("-small")
        for wave in waves:
            if (wave_logical_bytes(wave) < SMALL_CALL_BYTES) != small:
                raise ValueError(f"case {case_id}: a wave's size does not match family {family}")
    return case


def case_envelope_gbps(case: Mapping[str, Any], envelopes: Mapping[str, Any]) -> float:
    """Envelope for one case from the same-run probe record."""
    arm = str(case["arm"])
    key = "device_gbps" if arm == "device" else ("duplex_gbps" if arm == "duplex" else "host_gbps")
    return float(envelopes[key])


def validate_envelopes(envelopes: Any) -> dict[str, Any]:
    """Reject a probe record that is malformed or below the sanity floors."""
    if not isinstance(envelopes, Mapping) or envelopes.get("protocol") != ENVELOPE_PROTOCOL:
        raise RuntimeError("bandwidth envelope record differs")
    if envelopes.get("probe_bytes") != ENVELOPE_PROBE_BYTES or envelopes.get("repeats") != ENVELOPE_PROBE_REPEATS:
        raise RuntimeError("bandwidth envelope probe parameters differ")
    if envelopes.get("margin") != ENVELOPE_MARGIN:
        raise RuntimeError("bandwidth envelope margin differs")
    values: dict[str, float] = {}
    for key in (
        "device_gbps", "host_gbps", "h2d_gbps", "d2h_gbps",
        "zerocopy_h2d_gbps", "zerocopy_d2h_gbps", "duplex_gbps",
    ):
        value = envelopes.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0:
            raise RuntimeError(f"bandwidth envelope {key} is not a finite positive number")
        values[key] = float(value)
    if values["host_gbps"] != max(
        values["h2d_gbps"], values["d2h_gbps"],
        values["zerocopy_h2d_gbps"], values["zerocopy_d2h_gbps"],
    ):
        raise RuntimeError("bandwidth envelope host value differs from its probes")
    if values["device_gbps"] < ENVELOPE_MIN_DEVICE_GBPS or values["host_gbps"] < ENVELOPE_MIN_HOST_GBPS:
        raise RuntimeError("bandwidth envelope probe is below the sanity floor")
    if values["duplex_gbps"] < values["host_gbps"]:
        raise RuntimeError("bandwidth envelope duplex value is below the one-way host value")
    return {**dict(envelopes), **values}


def measure_envelopes(torch: Any, device: Any) -> dict[str, Any]:
    """Trusted same-run probe of the device-copy and pinned host-copy envelopes.

    Runs before the candidate library is loaded, with plain torch copies:
    contiguous D2D payload bandwidth and pinned-memory H2D/D2H bandwidth, best
    of ENVELOPE_PROBE_REPEATS.  These are the physical ceilings the scored
    arms are checked against, measured where the candidate is measured.
    """
    size = ENVELOPE_PROBE_BYTES
    source = torch.empty(size, dtype=torch.uint8, device=device)
    target = torch.empty(size, dtype=torch.uint8, device=device)
    host = torch.empty(size, dtype=torch.uint8).pin_memory()
    source.random_(0, 256)
    torch.cuda.synchronize(device)

    def best_gbps(copy: Callable[[], Any]) -> float:
        best = 0.0
        for _ in range(ENVELOPE_PROBE_REPEATS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            copy()
            end.record()
            torch.cuda.synchronize(device)
            seconds = float(start.elapsed_time(end)) / 1000.0
            if seconds > 0.0:
                best = max(best, size / seconds / 1.0e9)
        return best

    device_gbps = best_gbps(lambda: target.copy_(source))
    h2d_gbps = best_gbps(lambda: source.copy_(host, non_blocking=True))
    d2h_gbps = best_gbps(lambda: host.copy_(source, non_blocking=True))

    # SM-driven zero-copy stream over the same pinned buffer, through the
    # frozen compile command (native arch) and a trusted kernel.  Pinned
    # memory is UVA-addressable from the device, exactly as candidates use it.
    zerocopy = _load_zerocopy_probe(torch)
    stream = torch.cuda.current_stream(device)

    def zerocopy_copy(src: int, dst: int) -> None:
        status = zerocopy(
            ctypes.c_void_p(src), ctypes.c_void_p(dst),
            ctypes.c_size_t(size), ctypes.c_void_p(stream.cuda_stream),
        )
        if status != 0:
            raise RuntimeError(f"zero-copy envelope probe launch failed: cudaError {status}")

    zerocopy_h2d_gbps = best_gbps(lambda: zerocopy_copy(host.data_ptr(), source.data_ptr()))
    zerocopy_d2h_gbps = best_gbps(lambda: zerocopy_copy(source.data_ptr(), host.data_ptr()))

    # Full-duplex PCIe: one H2D and one D2H copy on two streams at once.  The
    # duplex envelope is the larger of that and the best one-way host value,
    # so a link without duplex gain never rejects a correct candidate.
    host_in = torch.empty(size, dtype=torch.uint8).pin_memory()
    host_in.fill_(7)
    upload = torch.cuda.Stream(device)
    download = torch.cuda.Stream(device)
    duplex_best = 0.0
    for _ in range(ENVELOPE_PROBE_REPEATS):
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        upload.wait_event(start)
        download.wait_event(start)
        with torch.cuda.stream(upload):
            target.copy_(host_in, non_blocking=True)
        with torch.cuda.stream(download):
            host.copy_(source, non_blocking=True)
        torch.cuda.current_stream(device).wait_stream(upload)
        torch.cuda.current_stream(device).wait_stream(download)
        end.record()
        torch.cuda.synchronize(device)
        seconds = float(start.elapsed_time(end)) / 1000.0
        if seconds > 0.0:
            duplex_best = max(duplex_best, 2 * size / seconds / 1.0e9)
    del host_in
    del source, target, host
    torch.cuda.synchronize(device)
    return validate_envelopes(
        {
            "protocol": ENVELOPE_PROTOCOL,
            "probe_bytes": size,
            "repeats": ENVELOPE_PROBE_REPEATS,
            "margin": ENVELOPE_MARGIN,
            "device_gbps": device_gbps,
            "h2d_gbps": h2d_gbps,
            "d2h_gbps": d2h_gbps,
            "zerocopy_h2d_gbps": zerocopy_h2d_gbps,
            "zerocopy_d2h_gbps": zerocopy_d2h_gbps,
            "host_gbps": max(h2d_gbps, d2h_gbps, zerocopy_h2d_gbps, zerocopy_d2h_gbps),
            "duplex_gbps": max(duplex_best, h2d_gbps, d2h_gbps, zerocopy_h2d_gbps, zerocopy_d2h_gbps),
        }
    )


_ZEROCOPY_PROBE: Callable[..., int] | None = None


def _load_zerocopy_probe(torch: Any) -> Callable[..., int]:
    """Compile (once per process) and load the trusted zero-copy probe."""
    global _ZEROCOPY_PROBE
    if _ZEROCOPY_PROBE is not None:
        return _ZEROCOPY_PROBE
    build_dir = Path(tempfile.mkdtemp(prefix="envelope-probe-"))
    source = build_dir / "envelope_probe.cu"
    source.write_text(ENVELOPE_ZEROCOPY_SOURCE, encoding="utf-8")
    output = build_dir / "envelope_probe.so"
    process = subprocess.run(
        compile_argv(source, output, Path(__file__).resolve().parent, native_arch(torch)),
        capture_output=True,
        text=True,
        timeout=300.0,
    )
    if process.returncode != 0:
        raise RuntimeError(f"zero-copy envelope probe did not compile: {process.stderr[-2000:]}")
    library = ctypes.CDLL(str(output))
    function = library.envelope_zerocopy_stream
    function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    function.restype = ctypes.c_int
    _ZEROCOPY_PROBE = function
    return function


# --------------------------------------------------------------------------
# Candidate loading
# --------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_candidate(library_path: Path) -> Callable[..., None]:
    library = ctypes.CDLL(str(library_path))
    function = getattr(library, ENTRYPOINT)
    function.argtypes = [ctypes.POINTER(TransferParams), ctypes.c_void_p]
    function.restype = None
    return function


# Capability envelope: what the task contract actually needs, not the GPU model
# it was qualified on.  The GPU name, UUID and PCIe link are recorded as
# provenance only; they never gate admission.  Keep these two constants
# identical in the Judge verifier (tests assert the pair).
MIN_COMPUTE_CAPABILITY = (8, 0)
MIN_DEVICE_MEMORY_BYTES = 64 * 1024**3


def _pcie_provenance(index: int) -> dict[str, Any]:
    """Best-effort PCIe link generation/width (host-platform provenance)."""
    try:
        import pynvml  # noqa: PLC0415

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        return {
            "pcie_link_gen": int(pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle)),
            "pcie_link_width": int(pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle)),
        }
    except Exception:  # pragma: no cover - provenance only, never a gate
        return {"pcie_link_gen": None, "pcie_link_width": None}


def hardware_inventory(torch: Any) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device is visible")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"exactly one CUDA device must be visible, found {torch.cuda.device_count()}"
        )
    index = int(torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(index)
    capability = (int(properties.major), int(properties.minor))
    total_memory_bytes = int(properties.total_memory)
    if capability < MIN_COMPUTE_CAPABILITY:
        raise RuntimeError(
            f"GPU compute capability {capability} is below the task minimum "
            f"{MIN_COMPUTE_CAPABILITY}: {properties.name}"
        )
    if total_memory_bytes < MIN_DEVICE_MEMORY_BYTES:
        raise RuntimeError(
            f"GPU memory {total_memory_bytes} bytes is below the task minimum "
            f"{MIN_DEVICE_MEMORY_BYTES} bytes: {properties.name}"
        )
    return {
        "device_name": str(properties.name),
        "uuid": str(getattr(properties, "uuid", "")),
        "capability": f"sm_{capability[0]}{capability[1]}",
        "compute_capability": list(capability),
        "min_compute_capability": list(MIN_COMPUTE_CAPABILITY),
        "min_device_memory_bytes": MIN_DEVICE_MEMORY_BYTES,
        "total_memory_bytes": total_memory_bytes,
        **_pcie_provenance(index),
        "multi_processor_count": int(properties.multi_processor_count),
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
    }

# --------------------------------------------------------------------------
# Case execution
# --------------------------------------------------------------------------
def _derive_seed(*parts: int) -> int:
    digest = hashlib.sha256(
        b"lmcache-kv-transfer-v2:" + b":".join(str(part).encode() for part in parts)
    ).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFF_FFFF_FFFF_FFFF


def _poison_scratch(torch: Any, scratch: Any, generator: Any) -> None:
    scratch.random_(0, 256, generator=generator)


def _generator(torch: Any, device: Any, seed: int) -> Any:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


class HostArena:
    """One pinned host allocation for the whole suite, carved per case.

    Pinned allocations are rounded up and cached by the allocator and never
    returned, so allocating per case would grow host memory with every new
    size.  One arena, allocated before the candidate loads, avoids that and
    keeps every case's host chunks on pages that are already faulted in.
    """

    def __init__(self, torch: Any, nbytes: int) -> None:
        self.torch = torch
        self.buffer = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        self.offset = 0

    def reset(self) -> None:
        self.offset = 0

    def take(self, shape: Sequence[int]) -> Any:
        count = 1
        for dim in shape:
            count *= int(dim)
        nbytes = count * ESIZE
        start = (self.offset + ARENA_ALIGN - 1) // ARENA_ALIGN * ARENA_ALIGN
        if start + nbytes > self.buffer.numel():
            raise RuntimeError("host chunks exceed the pinned arena")
        self.offset = start + nbytes
        return self.buffer[start:start + nbytes].view(self.torch.int16).view(*shape)


def _object_rows(torch: Any, geometry: Mapping[str, int], objects: Sequence[Any],
                 layer: int, nh: int) -> Any:
    """Flat blocks of one layer from chunk objects: [kv, F, NH, BS, W]."""
    kv, width = geometry["kv"], geometry["hs"]
    rows = torch.cat(
        [obj[:, layer].reshape(kv, geometry["blocks_per_object"], BS, nh, width)
         for obj in objects],
        dim=1,
    )
    return rows.permute(0, 1, 3, 2, 4)


def _select_blocks(layer_tensor: Any, layout_id: int, ids: Any) -> Any:
    """Selected pool blocks of one layer: [kv, F, NH, BS, W]."""
    if layout_id == LAYOUT_SPLIT:
        return layer_tensor[:, ids]
    return layer_tensor[ids].unsqueeze(0)


class PoolCase:
    """One engine pool shared by every wave of a case.

    The pool is filled per layer from a seeded generator, so its expected
    contents can be regenerated layer by layer instead of being cloned: a
    real pool is tens of gigabytes and a clone would not fit.  Verification
    replays the case's waves in order on one regenerated layer at a time,
    which checks every store destination, every retrieved block, every
    skipped block and every untargeted pool byte bit for bit.
    """

    def __init__(self, torch: Any, case: Mapping[str, Any], device: Any,
                 arena: HostArena | None) -> None:
        self.torch = torch
        self.device = device
        self.case = case
        self.waves = list(case["waves"])
        first = self.waves[0]
        self.geometry0 = wave_geometry(first)
        self.nl, self.nh, self.nb = int(first["nl"]), int(first["nh"]), int(first["nb"])
        kv, width = self.geometry0["kv"], self.geometry0["hs"]
        if self.geometry0["layout_id"] == LAYOUT_SPLIT:
            self.pool_shape = (kv, self.nb, self.nh, BS, width)
        else:
            self.pool_shape = (self.nb, self.nh, BS, width)
        self.pool = [torch.empty(self.pool_shape, dtype=torch.int16, device=device)
                     for _ in range(self.nl)]
        self.ptr_array = torch.tensor([t.data_ptr() for t in self.pool],
                                      dtype=torch.int64, device=device)
        self.host_chunks = case_host_chunks(case)
        if self.host_chunks and arena is None:
            raise RuntimeError("host chunks need the pinned arena")
        if arena is not None:
            arena.reset()
        self.chunks: list[list[Any]] = []
        self.base_ids: list[Any] = []
        self.ids: list[Any] = []
        largest = 0
        for wave in self.waves:
            geometry = wave_geometry(wave)
            shape = (geometry["kv"], self.nl, int(wave["chunk_tokens"]), geometry["hidden"])
            objects = []
            for _ in range(int(wave["num_objects"])):
                if self.host_chunks:
                    objects.append(arena.take(shape))
                else:
                    objects.append(torch.empty(shape, dtype=torch.int16, device=device))
            largest = max(largest, wave_object_bytes(wave) // ESIZE)
            self.chunks.append(objects)
            base = torch.tensor(wave["ids"], dtype=torch.int64, device=device)
            self.base_ids.append(base)
            self.ids.append(torch.empty_like(base))
        self.staging = (torch.empty(largest, dtype=torch.int16, device=device)
                        if self.host_chunks else None)
        self.seed = 0

    # -- per-trial contents -------------------------------------------------
    def _layer_seed(self, layer: int) -> int:
        return _derive_seed(self.seed, 0xA11, layer)

    def _object_seed(self, wave: int, obj: int) -> int:
        return _derive_seed(self.seed, 0xC0, wave, obj)

    def _fill_layer(self, target: Any, layer: int) -> None:
        target.random_(-32768, 32768, generator=_generator(self.torch, self.device,
                                                           self._layer_seed(layer)))

    def _source_object(self, wave: int, obj: int, shape: Sequence[int]) -> Any:
        out = self.torch.empty(tuple(shape), dtype=self.torch.int16, device=self.device)
        out.random_(-32768, 32768, generator=_generator(self.torch, self.device,
                                                        self._object_seed(wave, obj)))
        return out

    def refresh(self, seed: int) -> None:
        """New pool contents, new source payloads, new rotation, poisoned
        store destinations; reuses every allocation."""
        torch = self.torch
        self.seed = int(seed)
        for layer, tensor in enumerate(self.pool):
            self._fill_layer(tensor, layer)
        rotation = _derive_seed(self.seed, 0xB0B) % self.nb
        for base, ids in zip(self.base_ids, self.ids):
            ids.copy_(torch.remainder(base + rotation, self.nb))
        for index, wave in enumerate(self.waves):
            store = DIRECTION_IDS[str(wave["direction"])] == DIR_STORE
            for obj, tensor in enumerate(self.chunks[index]):
                if not self.host_chunks:
                    if store:
                        tensor.fill_(POISON)
                    else:
                        tensor.random_(-32768, 32768, generator=_generator(
                            torch, self.device, self._object_seed(index, obj)))
                    continue
                staging = self.staging[: tensor.numel()].view(tensor.shape)
                if store:
                    staging.fill_(POISON)
                else:
                    staging.random_(-32768, 32768, generator=_generator(
                        torch, self.device, self._object_seed(index, obj)))
                tensor.copy_(staging)
        torch.cuda.synchronize(self.device)

    def table_digest(self) -> str:
        tables = hashlib.sha256()
        for ids in self.ids:
            tables.update(ids.cpu().numpy().tobytes())
        tables.update(self.ptr_array.cpu().numpy().tobytes())
        return tables.hexdigest()

    def params_list(self, scratch: Any, retrieve_scratch: Any = None) -> list[TransferParams]:
        packed = []
        for index, wave in enumerate(self.waves):
            arena = scratch
            if retrieve_scratch is not None and wave["direction"] == "retrieve":
                arena = retrieve_scratch
            geometry = wave_geometry(wave)
            params = TransferParams()
            params.paged_layer_ptrs = self.ptr_array.data_ptr()
            chunks = self.chunks[index]
            for slot in range(4):
                params.chunk_ptrs[slot] = chunks[slot].data_ptr() if slot < len(chunks) else 0
            params.block_ids = self.ids[index].data_ptr()
            params.num_objects = int(wave["num_objects"])
            params.total_blocks = geometry["total_blocks"]
            params.blocks_per_object = geometry["blocks_per_object"]
            params.skip_prefix_n_blocks = int(wave["skip"])
            params.direction = DIRECTION_IDS[str(wave["direction"])]
            params.layout = geometry["layout_id"]
            params.kv_size = geometry["kv"]
            params.nl = self.nl
            params.nb = self.nb
            params.bs = BS
            params.nh = self.nh
            params.hs = geometry["hs"]
            params.element_size = ESIZE
            params.chunk_tokens = int(wave["chunk_tokens"])
            params.chunk_side_is_host = 1 if self.host_chunks else 0
            params.device_scratch = arena.data_ptr()
            params.device_scratch_bytes = int(arena.numel())
            packed.append(params)
        return packed

    # -- verification -----------------------------------------------------------
    def verify(self) -> int:
        """Number of differing elements over every destination and source."""
        torch = self.torch
        device = self.device
        bad = 0
        sources: dict[int, list[Any]] = {}
        stored: dict[int, list[Any]] = {}
        for index, wave in enumerate(self.waves):
            geometry = wave_geometry(wave)
            shape = (geometry["kv"], self.nl, int(wave["chunk_tokens"]), geometry["hidden"])
            if DIRECTION_IDS[str(wave["direction"])] == DIR_RETRIEVE:
                expected_objects = []
                for obj, tensor in enumerate(self.chunks[index]):
                    want = self._source_object(index, obj, shape)
                    got = tensor.to(device, non_blocking=False)
                    if not torch.equal(got, want):
                        bad += int((got != want).sum().item())
                    expected_objects.append(want)
                sources[index] = expected_objects
            else:
                stored[index] = [tensor.to(device) for tensor in self.chunks[index]]
        expected = torch.empty(self.pool_shape, dtype=torch.int16, device=device)
        layout_id = self.geometry0["layout_id"]
        for layer in range(self.nl):
            self._fill_layer(expected, layer)
            for index, wave in enumerate(self.waves):
                geometry = wave_geometry(wave)
                ids = self.ids[index]
                skip = int(wave["skip"])
                if index in sources:
                    rows = _object_rows(torch, geometry, sources[index], layer, self.nh)
                    if skip < geometry["total_blocks"]:
                        if layout_id == LAYOUT_SPLIT:
                            expected[:, ids[skip:]] = rows[:, skip:]
                        else:
                            expected[ids[skip:]] = rows[0, skip:]
                    continue
                selected = _select_blocks(expected, layout_id, ids)
                want = selected.permute(0, 1, 3, 2, 4).reshape(
                    geometry["kv"], geometry["total_blocks"] * BS, geometry["hidden"])
                if skip:
                    want = want.clone()
                    want[:, : skip * BS] = POISON
                bpo = geometry["blocks_per_object"]
                for obj, got in enumerate(stored[index]):
                    piece = want[:, obj * bpo * BS:(obj + 1) * bpo * BS]
                    have = got[:, layer]
                    if not torch.equal(have, piece):
                        bad += int((have != piece).sum().item())
            if not torch.equal(self.pool[layer], expected):
                bad += int((self.pool[layer] != expected).sum().item())
        del expected, sources, stored
        return bad


class Foreground:
    """Trusted concurrent work on a second stream: a serving engine's compute
    (bf16 GEMM) or its HBM traffic (a large device copy)."""

    def __init__(self, torch: Any, kind: str, device: Any) -> None:
        self.torch = torch
        self.kind = kind
        self.stream = torch.cuda.Stream(device)
        if kind == "gemm":
            self.a = torch.randn(8192, 8192, dtype=torch.bfloat16, device=device)
            self.b = torch.randn(8192, 8192, dtype=torch.bfloat16, device=device)
            self.c = torch.empty(8192, 8192, dtype=torch.bfloat16, device=device)
        elif kind == "hbm":
            self.src = torch.empty(1024**3, dtype=torch.uint8, device=device)
            self.src.random_(0, 256)
            self.dst = torch.empty_like(self.src)
        else:
            raise ValueError(f"unknown foreground {kind!r}")

    def step(self) -> None:
        if self.kind == "gemm":
            self.torch.mm(self.a, self.b, out=self.c)
        else:
            self.dst.copy_(self.src)

    def launch(self, clock: TrustedClock, iterations: int) -> tuple[Any, Any]:
        start = clock.event(enable_timing=True)
        end = clock.event(enable_timing=True)
        with self.torch.cuda.stream(self.stream):
            start.record(self.stream)
            for _ in range(iterations):
                self.step()
            end.record(self.stream)
        return start, end

    def alone(self, clock: TrustedClock, iterations: int) -> float:
        clock.synchronize()
        start, end = self.launch(clock, iterations)
        clock.synchronize()
        return finite_positive(float(start.elapsed_time(end)) / 1000.0, "foreground time")


def _invoke_params(
    function: Callable[..., None],
    packed: Sequence[TransferParams],
    stream_handle: int,
) -> None:
    handle = ctypes.c_void_p(stream_handle)
    for params in packed:
        function(ctypes.byref(params), handle)


class DuplexStreams:
    """Two trusted streams for duplex cases: stores on one, retrieves on the
    other.  Concurrent calls never share a scratch arena: retrieve calls get
    their own, as each transfer stream of a real cache server has its own
    staging buffer."""

    def __init__(self, torch: Any, device: Any) -> None:
        self.torch = torch
        self.store = torch.cuda.Stream(device)
        self.retrieve = torch.cuda.Stream(device)
        self.scratch = torch.empty(SCRATCH_BYTES, dtype=torch.uint8, device=device)


def _launch(function: Callable[..., None], case_set: PoolCase, packed: Sequence[TransferParams],
            clock: TrustedClock, duplex: DuplexStreams | None) -> None:
    """Issue every call: on the current stream, or for a duplex case store calls
    on one stream and retrieve calls on another, interleaved so both run at once.
    The current stream waits for both, so its next event marks the end."""
    stream = clock.current_stream()
    if duplex is None:
        _invoke_params(function, packed, stream.cuda_stream)
        return
    begin = clock.event()
    begin.record(stream)
    duplex.store.wait_event(begin)
    duplex.retrieve.wait_event(begin)
    stores = [p for p, w in zip(packed, case_set.waves) if w["direction"] == "store"]
    loads = [p for p, w in zip(packed, case_set.waves) if w["direction"] == "retrieve"]
    store_handle = ctypes.c_void_p(duplex.store.cuda_stream)
    load_handle = ctypes.c_void_p(duplex.retrieve.cuda_stream)
    for index in range(max(len(stores), len(loads))):
        if index < len(stores):
            function(ctypes.byref(stores[index]), store_handle)
        if index < len(loads):
            function(ctypes.byref(loads[index]), load_handle)
    stream.wait_stream(duplex.store)
    stream.wait_stream(duplex.retrieve)


def _settle(clock: TrustedClock, spin: Any, seconds: float) -> None:
    """Drain the preparation's deferred work with the GPU kept clocked."""
    deadline = clock.perf_counter() + seconds
    while clock.perf_counter() < deadline:
        for _ in range(50):
            spin.add_(1.0)
    clock.synchronize()


def _checked_pass(torch: Any, clock: TrustedClock, function: Callable[..., None],
                  case_set: PoolCase, scratch: Any, generator: Any, seed: int,
                  label: str, duplex: DuplexStreams | None = None) -> None:
    """One untimed, fully verified pass."""
    case_set.refresh(seed)
    digest = case_set.table_digest()
    _poison_scratch(torch, scratch, generator)
    if duplex is not None:
        _poison_scratch(torch, duplex.scratch, generator)
    packed = case_set.params_list(scratch, duplex.scratch if duplex is not None else None)
    clock.synchronize()
    _launch(function, case_set, packed, clock, duplex)
    clock.synchronize()
    # Tables first: verification indexes the pool with them.
    if case_set.table_digest() != digest:
        raise RuntimeError(f"{label}: block tables or pointer arrays were modified")
    bad = case_set.verify()
    if bad:
        raise RuntimeError(f"{label}: {bad} element(s) differ")


def _timed_pass(torch: Any, clock: TrustedClock, function: Callable[..., None],
                case_set: PoolCase, scratch: Any, generator: Any, spin: Any,
                seed: int, label: str, foreground: Foreground | None = None,
                iterations: int = 0, duplex: DuplexStreams | None = None) -> dict[str, Any]:
    """Refresh, settle, time one pass (optionally against a foreground), verify."""
    case_set.refresh(seed)
    digest = case_set.table_digest()
    _poison_scratch(torch, scratch, generator)
    if duplex is not None:
        _poison_scratch(torch, duplex.scratch, generator)
    packed = case_set.params_list(scratch, duplex.scratch if duplex is not None else None)
    stream = clock.current_stream()
    start_event = clock.event(enable_timing=True)
    end_event = clock.event(enable_timing=True)
    clock.synchronize()
    _settle(clock, spin, SETTLE_SECONDS if case_set.host_chunks else DEVICE_SETTLE_SECONDS)
    fg_events = None
    if foreground is not None:
        fg_events = foreground.launch(clock, iterations)
    wall_start = clock.perf_counter()
    start_event.record(stream)
    _launch(function, case_set, packed, clock, duplex)
    end_event.record(stream)
    end_event.synchronize()
    host_wall = finite_positive(clock.perf_counter() - wall_start, "host wall time")
    clock.synchronize()
    event_seconds = finite_positive(
        float(start_event.elapsed_time(end_event)) / 1000.0, "CUDA event time")
    record: dict[str, Any] = {"host_wall": host_wall, "event": event_seconds,
                              "seconds": max(host_wall, event_seconds)}
    if fg_events is not None:
        fg_start, fg_end = fg_events
        record["fg_seconds"] = finite_positive(
            float(fg_start.elapsed_time(fg_end)) / 1000.0, "foreground time")
        record["fg_to_transfer_end"] = float(fg_start.elapsed_time(end_event)) / 1000.0
    if case_set.table_digest() != digest:
        raise RuntimeError(f"{label}: block tables or pointer arrays were modified")
    bad = case_set.verify()
    if bad:
        raise RuntimeError(f"{label}: {bad} element(s) differ")
    return record


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def overlap_loss(fg_overlap: float, fg_alone: float, transfer: float) -> float:
    """Foreground time lost per second of overlapped transfer, clamped to
    [0, MAX_OVERLAP_LOSS]."""
    loss = (fg_overlap - fg_alone) / transfer
    return min(max(loss, 0.0), MAX_OVERLAP_LOSS)


def run_scored_case(
    torch: Any,
    clock: TrustedClock,
    function: Callable[..., None],
    case: Mapping[str, Any],
    device: Any,
    scratch: Any,
    generator: Any,
    arena: HostArena | None,
    envelopes: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Decoy pass, then timed and verified trials; overlap cases also time
    the transfer alone and the foreground alone."""
    case_id = str(case["case_id"])
    trials = int(case["trials"])
    seed = int(case["seed"])
    case_set = PoolCase(torch, case, device, arena)
    spin = torch.zeros(256, dtype=torch.float32, device=device)
    duplex = DuplexStreams(torch, device) if case["arm"] == "duplex" else None
    # Decoy warm-up: same waves, a different seed lineage -> different pool,
    # payloads and rotation.  Verified, so a cached answer fails here.
    _checked_pass(torch, clock, function, case_set, scratch, generator,
                  _derive_seed(seed, 3), f"case {case_id} decoy", duplex)
    logical = case_logical_bytes(case)
    row: dict[str, Any] = {
        "case_id": case_id,
        "family": str(case["family"]),
        "arm": str(case["arm"]),
        "foreground": case["foreground"],
        "stratum": str(case["stratum"]),
        "tier": str(case["tier"]),
        "scored": True,
        "waves": len(case["waves"]),
        "trials": trials,
        "logical_bytes": logical,
    }
    foreground = None
    iterations = 0
    if case["arm"] == "overlap":
        alone = [
            _timed_pass(torch, clock, function, case_set, scratch, generator, spin,
                        _derive_seed(seed, 5, trial), f"case {case_id} alone {trial}")
            for trial in range(trials)
        ]
        alone_seconds = [item["seconds"] for item in alone]
        foreground = Foreground(torch, str(case["foreground"]), device)
        foreground.alone(clock, 3)
        per_step = foreground.alone(clock, 5) / 5.0
        iterations = max(4, int(math.ceil(OVERLAP_COVER * _median(alone_seconds) / per_step)))
        row["alone_seconds"] = alone_seconds
    records: list[dict[str, Any]] = []
    fg_alone: list[float] = []
    for trial in range(trials):
        label = f"case {case_id} trial {trial}"
        trial_seed = _derive_seed(seed, 4, trial + 1)
        if foreground is None:
            records.append(_timed_pass(torch, clock, function, case_set, scratch,
                                       generator, spin, trial_seed, label, duplex=duplex))
            continue
        for _attempt in range(4):
            record = _timed_pass(torch, clock, function, case_set, scratch, generator,
                                 spin, trial_seed, label, foreground, iterations)
            if record["fg_seconds"] >= record["fg_to_transfer_end"]:
                break
            iterations *= 2
        else:
            raise RuntimeError(f"{label}: the foreground never covered the transfer")
        record["fg_iterations"] = iterations
        record["fg_alone"] = foreground.alone(clock, iterations)
        records.append(record)
        fg_alone.append(record["fg_alone"])
    elapsed = [item["seconds"] for item in records]
    median = _median(elapsed)
    gbps = logical / median / 1.0e9
    if envelopes is not None:
        envelope = case_envelope_gbps(case, envelopes) * ENVELOPE_MARGIN
        if gbps > envelope:
            raise RuntimeError(
                f"case {case_id}: {gbps:.1f} GB/s exceeds the physical envelope "
                f"({envelope:.1f} GB/s) — measurement or candidate is not trustworthy"
            )
    row.update({
        "trial_seconds": elapsed,
        "host_wall_seconds": [item["host_wall"] for item in records],
        "event_seconds": [item["event"] for item in records],
        "median_seconds": median,
        "gbps": gbps,
    })
    score = gbps
    if foreground is not None:
        losses = [
            overlap_loss(item["fg_seconds"], item["fg_alone"], item["seconds"])
            for item in records
        ]
        row.update({
            "fg_overlap_seconds": [item["fg_seconds"] for item in records],
            "fg_alone_seconds": fg_alone,
            "fg_iterations": [item["fg_iterations"] for item in records],
            "fg_loss": losses,
            "fg_loss_median": _median(losses),
        })
        score = gbps * (1.0 - _median(losses))
        del foreground
    row["case_score"] = score
    del case_set
    torch.cuda.empty_cache()
    return row


def run_edge_case(
    torch: Any,
    clock: TrustedClock,
    function: Callable[..., None],
    case: Mapping[str, Any],
    device: Any,
    scratch: Any,
    generator: Any,
    arena: HostArena | None,
) -> dict[str, Any]:
    """Correctness only.  Never timed, never scored."""
    case_set = PoolCase(torch, case, device, arena)
    duplex = DuplexStreams(torch, device) if case["arm"] == "duplex" else None
    _checked_pass(torch, clock, function, case_set, scratch, generator,
                  _derive_seed(int(case["seed"]), 1), f"edge {case['case_id']}", duplex)
    del case_set
    torch.cuda.empty_cache()
    return {
        "case_id": str(case["case_id"]),
        "arm": str(case["arm"]),
        "scored": False,
        "waves": len(case["waves"]),
        "correct": True,
    }


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
def family_score(rows: Sequence[Mapping[str, Any]]) -> float:
    """Tier-balanced geometric mean of stratum geometric means."""
    strata: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        strata.setdefault((str(row["tier"]), str(row["stratum"])), []).append(float(row["case_score"]))
    tiers: dict[str, list[float]] = {}
    for (tier, _stratum), values in strata.items():
        tiers.setdefault(tier, []).append(math.log(geometric_mean(values)))
    return math.exp(math.fsum(math.fsum(logs) / len(logs) for logs in tiers.values()) / len(tiers))


def aggregate(
    results: Sequence[Mapping[str, Any]], *, require_full_groups: bool = True
) -> dict[str, Any]:
    scored = [row for row in results if row.get("scored")]
    if not scored:
        raise RuntimeError("no scored case produced a result")
    per_family: dict[str, list[Mapping[str, Any]]] = {}
    for row in scored:
        per_family.setdefault(str(row["family"]), []).append(row)
    family_scores = {name: family_score(rows) for name, rows in sorted(per_family.items())}
    groups: dict[str, float] = {}
    for group, families in GROUP_FAMILIES.items():
        present = [family_scores[name] for name in families if name in family_scores]
        if require_full_groups and len(present) != len(families):
            missing = [name for name in families if name not in family_scores]
            raise RuntimeError(f"no scored case for family(ies) {missing}")
        if present:
            groups[group] = geometric_mean(present)
    if require_full_groups:
        score = math.exp(math.fsum(
            GROUP_WEIGHTS[name] * math.log(groups[name]) for name in GROUP_WEIGHTS))
    else:
        score = geometric_mean(groups.values())
    return {
        "family_scores": family_scores,
        "group_scores": groups,
        "group_weights": dict(GROUP_WEIGHTS),
        "score_gbps": score,
        "scored_cases": len(scored),
    }


# --------------------------------------------------------------------------
# Suites
# --------------------------------------------------------------------------
PUBLIC_SUITE_ID = "lmcache-kv-transfer-public-v3"
FINAL_SUITE_ID = "lmcache-kv-transfer-final-holdout-v2"
EDGE_SEED_BASE = 0x00ED_0002


def load_case_table(path: Path, *, scored: bool = True) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise ValueError("case table schema differs")
    cases = [validate_case(case, scored=scored) for case in payload["cases"]]
    if not cases:
        raise ValueError("case table contains no case")
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("case table contains duplicate case ids")
    return {"suite_id": str(payload["suite_id"]), "cases": cases}


def load_manifest(path: Path) -> dict[str, Any]:
    table = load_case_table(path)
    missing = sorted(set(FAMILIES) - {case["family"] for case in table["cases"]})
    if missing:
        raise ValueError(f"manifest does not cover family(ies) {missing}")
    return table


# --------------------------------------------------------------------------
# Edge fixtures — correctness only, run first in every suite
# --------------------------------------------------------------------------
def edge_ids(pattern: str, count: int, nb: int, seed: int) -> list[int]:
    """Deterministic block-id tables with the shapes a live pool produces."""
    import random as _random

    rng = _random.Random(seed)
    if pattern == "contig":
        start = rng.randrange(0, nb - count + 1)
        return list(range(start, start + count))
    if pattern == "reverse":
        start = rng.randrange(0, nb - count + 1)
        return list(range(start + count - 1, start - 1, -1))
    if pattern == "ends":
        return [nb - 1 - i // 2 if i % 2 else i // 2 for i in range(count)]
    if pattern == "runs":
        free = list(range(nb))
        rng.shuffle(free)
        ids: list[int] = []
        used: set[int] = set()
        for start in free:
            if len(ids) >= count:
                break
            length = rng.choice((1, 1, 2, 3, 5, 8))
            step = rng.choice((1, -1))
            run = [start + step * k for k in range(length)]
            run = [b for b in run if 0 <= b < nb and b not in used][: count - len(ids)]
            ids.extend(run)
            used.update(run)
        return ids
    return rng.sample(range(nb), count)


def _edge(case_id: str, arm: str, *, pattern: str = "perm", waves: int = 1,
          **overrides: Any) -> dict[str, Any]:
    base = {
        "layout": "split",
        "nl": 4,
        "nh": 4,
        "hs": 128,
        "chunk_tokens": 64,
        "num_objects": 2,
        "direction": "retrieve",
        "skip": 0,
        "nb": 64,
    }
    base.update(overrides)
    count = base["num_objects"] * base["chunk_tokens"] // BS
    seed = _derive_seed(EDGE_SEED_BASE, int.from_bytes(
        hashlib.sha256(case_id.encode()).digest()[:4], "little"))
    wave_list = []
    for index in range(waves):
        wave = dict(base)
        wave["ids"] = edge_ids(pattern, count, base["nb"], _derive_seed(seed, index))
        wave_list.append(wave)
    if arm in ("trace", "duplex"):
        for index, wave in enumerate(wave_list):
            wave["direction"] = "store" if index % 2 else "retrieve"
    if arm == "duplex":
        # Disjoint blocks: stores use the low half of the pool, retrieves the high half.
        half = base["nb"] // 2
        for wave in wave_list:
            offset = 0 if wave["direction"] == "store" else half
            wave["ids"] = [offset + b % half for b in wave["ids"]]
            if len(set(wave["ids"])) != len(wave["ids"]):
                wave["ids"] = list(range(offset, offset + len(wave["ids"])))
    family = {"trace": "trace", "duplex": "host-duplex"}.get(arm, f"{arm}-small")
    return validate_case(
        {
            "case_id": case_id,
            "family": family,
            "arm": arm,
            "foreground": None,
            "stratum": "edge",
            "tier": "edge",
            "waves": wave_list,
            "trials": 1,
            "seed": seed,
        },
        scored=False,
    )


def edge_fixtures() -> list[dict[str, Any]]:
    return [
        _edge("edge-skip-all", "device", skip=8),
        _edge("edge-skip-almost-all", "device", skip=7, direction="store"),
        _edge("edge-skip-partial-fused", "device", layout="fused", skip=3),
        _edge("edge-single-block", "device", chunk_tokens=16, num_objects=1),
        _edge("edge-three-objects", "device", num_objects=3, chunk_tokens=128, nb=128),
        _edge("edge-tail-chunk", "device", chunk_tokens=48, num_objects=1, direction="store"),
        _edge("edge-single-layer", "device", nl=1),
        _edge("edge-deepest-stack", "device", nl=128, nb=32, num_objects=1),
        _edge("edge-single-head", "device", nh=1, pattern="reverse"),
        _edge("edge-most-heads", "device", nh=32, nb=16, num_objects=1),
        _edge("edge-head-64-split", "device", hs=64, pattern="runs", direction="store"),
        _edge("edge-head-64-fused", "device", hs=64, layout="fused", pattern="runs"),
        _edge("edge-pool-ends", "device", pattern="ends", nb=4096, direction="store"),
        _edge("edge-whole-pool", "device", pattern="contig", nb=8),
        _edge("edge-repeated-blocks", "device", pattern="contig", nb=8, waves=3),
        _edge("edge-largest-pool", "device", layout="fused", nl=1, nh=1, hs=128,
              chunk_tokens=1024, num_objects=4, nb=65536, pattern="runs"),
        _edge("edge-object-beyond-scratch", "device", nl=16, nh=8, chunk_tokens=4128,
              num_objects=1, nb=300),
        _edge("edge-host-single", "host", num_objects=1, nh=3),
        _edge("edge-host-store-fused", "host", layout="fused", direction="store"),
        _edge("edge-host-skip", "host", skip=5, pattern="reverse"),
        _edge("edge-host-head-64", "host", hs=64, nl=18, nh=2, direction="store"),
        _edge("edge-host-beyond-scratch", "host", layout="fused", nl=16, nh=8,
              chunk_tokens=4224, num_objects=1, nb=512),
        _edge("edge-trace-overwrite", "trace", pattern="contig", nb=8, waves=4),
        _edge("edge-duplex", "duplex", pattern="perm", nb=128, waves=4),
        _edge("edge-duplex-skip-fused", "duplex", layout="fused", skip=3, pattern="runs",
              nb=256, waves=6),
    ]


# --------------------------------------------------------------------------
# Suite driver
# --------------------------------------------------------------------------
def run_suite(
    torch: Any,
    library_path: Path,
    cases: Sequence[Mapping[str, Any]],
    *,
    suite_id: str,
    with_edge_fixtures: bool = True,
    require_full_groups: bool = True,
) -> dict[str, Any]:
    """Run one suite end to end and return a JSON-serialisable record.

    The clock is captured, the hardware inventoried, the envelope probed and
    the pinned arena allocated before the candidate library is loaded.
    ``library_path`` must already have passed source admission, frozen-command
    compilation, and the pre-load binary audit.
    """
    inventory = hardware_inventory(torch)
    clock = TrustedClock.capture(torch)
    device = torch.device("cuda", torch.cuda.current_device())
    envelopes = measure_envelopes(torch, device)
    scratch = torch.empty(SCRATCH_BYTES, dtype=torch.uint8, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(_derive_seed(0x5C4A7C4))
    needs_arena = any(case_host_chunks(case) for case in cases) or with_edge_fixtures
    arena = HostArena(torch, HOST_ARENA_BYTES) if needs_arena else None
    library_digest = sha256_file(Path(library_path))
    function = load_candidate(Path(library_path))

    results: list[dict[str, Any]] = []
    if with_edge_fixtures:
        for fixture in edge_fixtures():
            results.append(
                run_edge_case(torch, clock, function, fixture, device, scratch, generator, arena)
            )
    for case in cases:
        results.append(
            run_scored_case(torch, clock, function, case, device, scratch, generator,
                            arena, envelopes)
        )
    summary = aggregate(results, require_full_groups=require_full_groups)
    return {
        "suite_id": suite_id,
        "library_sha256": library_digest,
        "entrypoint": ENTRYPOINT,
        "hardware": inventory,
        "envelopes": envelopes,
        "constants": RECORD_CONSTANTS,
        "edge_fixtures": [row for row in results if not row.get("scored")],
        "cases": [row for row in results if row.get("scored")],
        **summary,
    }
