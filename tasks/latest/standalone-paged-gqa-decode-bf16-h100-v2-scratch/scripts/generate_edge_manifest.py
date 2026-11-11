#!/usr/bin/env python3
"""Generate the zero-weight edge-case manifests for standalone paged GQA decode.

Edge cases are correctness and memory-safety gates over the whole declared
support domain: any batch 1-256, any S_q 1-16, any GQA group size with
H_q <= 128, page sizes 1-256 (powers of two), page tables up to 4x wider than
needed, page pools larger than ``B x width`` with repeated (shared) page ids,
and cache slots that no query may see holding NaN or Inf.  They are untimed
and never enter the score.  Every value list below is taken from an upstream
test or deployment default (FlashAttention @ 0251105a, vLLM @ 6bbad6a,
FlashInfer @ ea8135c, SGLang @ f3b5a28) or from a v1 edge fixture; the
comments cite the source, and the task-authored additions say so.

One seeded generator produces two frozen instances with the same categories
and different draws:

* ``judge/private/edge_cases.json`` (``edge-holdout-v4``, 168 cases,
  Judge-only, staged like the final holdout);
* ``images/work/benchmark/public_edge_cases.json`` (``edge-public-v4``,
  44 cases, shipped to Work for ``run_public_tests.sh --edge``).

v4 appends the ``long_context`` category (8 private, 4 public cases): context
lengths above the v2 bound of 32768, up to the 1,048,576 of the v4 support
domain, at page sizes 1 to 256.  The draw is seeded from the v2 suite
identities, so the first 160 / 40 cases are byte-identical to v2.

The ``sanitize`` category (12 private, 4 public cases, ``"sanitize": true``)
is the subset the Judge runs under ``compute-sanitizer --tool memcheck``
before any native call: the layouts most likely to produce a wild pointer.
One private sanitizer case is ``"native": false``: its K/V pool holds more
than 2**31 elements (SGLang's int32-offset regression), too large to run
natively in the edge suite, so it runs only under the sanitizer.

The ``v1_contract`` category stays inside the v1 support domain (heads
32/8, 64/8, 128/1; pages 16, 64, 128; S_q 1 or 16; a one-to-one page table of
exactly ``B x ceil(L_max/ps)`` pages; N(0,1) everywhere), so a failure there
is a bug v1 should already have rejected.  It opens with the two cross
combinations two historical kernels failed on (2026-10-05 audit).

    scripts/generate_edge_manifest.py --check     # regenerate and compare
    scripts/generate_edge_manifest.py --write     # rewrite both instances

``--write`` mints new suite identities when the output changes; the
``frozen_instance`` and ``sha256`` records in ``portable/*.json`` must then be
refreshed.  Both modes print a summary (cases per category, sanitizer subset,
K/V bytes).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

TASK_ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "standalone_paged_gqa_decode_bf16_h100_scratch_v2"
PRIVATE_SUITE_ID = "edge-holdout-v4"
PUBLIC_SUITE_ID = "edge-public-v4"
# The draw keeps the v2 seeds, so the cases v2 already had are unchanged.
SEED_SUITE_IDS = {PRIVATE_SUITE_ID: "edge-holdout-v2", PUBLIC_SUITE_ID: "edge-public-v2"}
PRIVATE_PATH = TASK_ROOT / "judge/private/edge_cases.json"
PUBLIC_PATH = TASK_ROOT / "images/work/benchmark/public_edge_cases.json"
SCHEMA_VERSION = 4

# Batch sizes.  FA3 hopper/test_flash_attn.py::test_flash_attn_kvcache batch 5
# (:744); FA2 tests/test_flash_attn.py::test_flash_attn_kvcache batch 2
# (:1878-1936); vLLM tests/kernels/attention/test_triton_decode_attention.py
# B 3 and 5 (:16-41); FlashInfer tests/attention/test_batch_decode_kernels.py
# batch 12 and 17 (:285-290).  1, 7, 9 and the 2^k +/- 1 neighbours 31, 33,
# 127, 129, 255 up to the contract maximum 256 are task-authored (the
# infrastructure review's "1, and sizes not divisible by 8").
BATCHES = (1, 2, 3, 5, 7, 9, 12, 17, 31, 33, 127, 129, 255, 256)
SMALL_BATCHES = BATCHES[:10]  # up to 33

# Cache lengths.  FA3 (seqlen_q, seqlen_k) pairs (1, 339), (3, 799), (16, 20000)
# (:700-706) with cache_seqlens = randint(1, seqlen_k + 1) (:810), hence 1-3 at
# the low end; vLLM seq_lens 18, 37, 463, 523, 1328, 2011
# (tests/kernels/attention/test_flash_attn.py:25-34) and 1025, 1027
# (test_triton_decode_attention.py:16-41); FlashInfer kv_len 54, 97
# (:285-290); SGLang seq_lens 5, 100, 128, 500
# (test/registered/attention/test_triton_attention_kernels.py:851-863).
# 7, 9, 17 and the 2^k - 1 / 2^k / 2^k + 1 values 127-129, 4095, 16383, 32767
# and the contract maximum 32768 are task-authored (infrastructure review and
# the v1 edge fixtures edge-page-boundary / edge-tiny-batch).
UPSTREAM_LENGTHS = (
    1, 2, 3, 5, 7, 9, 17, 18, 37, 54, 97, 100, 127, 128, 129, 339, 463, 500, 523,
    799, 1025, 1027, 1328, 2011, 4095, 16383, 20000, 32767, 32768,
)
# The lengths category sweeps these groups; "s_q" and "s_q+1" are the contract
# corners L_b = S_q and L_b = S_q + 1 (FA3's randint lower bound), and the
# all-ones group at S_q = 1 makes every output row V itself (a zero tolerance).
LENGTH_GROUPS: tuple[tuple[Any, ...], ...] = (
    ("s_q",),
    ("s_q+1",),
    (1,),
    (1, 2, 3, 5, 7, 9),
    (17, 18, 37, 54),
    (97, 100, 127, 128, 129),
    (339, 463, 500, 523),
    (799, 1025, 1027, 1328),
    (2011, 4095),
    (16383,),
    (20000,),
    (32767,),
    (32768,),
)

# Query lengths.  FA3 seqlen_q 1, 3, 16 (:700-706); FlashInfer multi-token
# decode q_len_per_req 2, 3 (:1195-1198); vLLM query_len 1 and 5
# (test_flash_attn.py:25-34); FA3 hopper/benchmark_split_kv.py query_seqlen
# 1, 4, 8, 16 (:47-77).  7, 9 and 15 are the odd neighbours of 8 and 16
# (task-authored).
S_Q_VALUES = (1, 2, 3, 4, 5, 7, 8, 9, 15, 16)

# (H_q, H_kv) at head_dim 128.  FA3 hopper/benchmark_split_kv.py model_configs
# (:47-77): Llama-3.1-8B 32/8, Llama-3.1-70B 64/8, Qwen-2.5-7B 28/4,
# Llama-3.2-3B 24/8, Qwen-2.5-1.5B 12/2, Gemma-2-27B 32/16, Mistral Large 96/8,
# Llama-3.1-405B 128/8, Nemotron-4-15B 48/8.  FA3 test nheads 6 with nheads_k
# 6, 3, 1 (MHA / GQA / MQA, :747-751).  vLLM NUM_HEADS (4, 4), (8, 2)
# (test_flash_attn.py:25-34) and H_Q 32 with H_KV 32
# (test_triton_decode_attention.py:16-41).  SGLang grouped decode (16, 1),
# (64, 1), (128, 1), (128, 2) (:851-863).  Tensor-parallel splits of the
# benchmark table (its tp_degree, :87-88): 8/1 (Llama-3.1-70B at TP8); 16/1 is
# Llama-3.1-405B at TP8 and SGLang's (16, 1).
HEAD_CONFIGS = (
    (32, 8), (64, 8), (28, 4), (24, 8), (12, 2), (32, 16), (96, 8), (128, 8), (48, 8),
    (6, 6), (6, 3), (6, 1),
    (4, 4), (8, 2), (32, 32),
    (16, 1), (64, 1), (128, 1), (128, 2),
    (8, 1),
)

# Page sizes.  FA3 page_size 1, 4, 128 (:682); FA2 paged_kv_block_size 256
# (:1878-1936); FlashInfer page_size 1, 8, 16 (:285-290); vLLM Triton decode
# PAGE_SIZE 1, 16 (:16-41) and DEFAULT_BLOCK_SIZE 16 (vllm/config/cache.py:71);
# SGLang default page_size 1 (python/sglang/srt/server_args.py:77);
# PagedAttention (SOSP'23, arXiv 2309.06180, section 7.2) block sizes 16-128,
# hence 32 and 64.  2 completes the power-of-two range (task-authored).
PAGE_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)

# Page-table layouts:
#   v1        one-to-one permutation of exactly B x ceil(L_max/ps) pages (the
#             v1 generator, the only layout v1 ever produced);
#   fa3       FA3 _generate_block_kvcache (:1054-1077): num_blocks =
#             ceil(L/ps) * B * 3, page_table = randperm(num_blocks) split by
#             batch, so every row is three times as wide as needed;
#   wide      the widest table the contract admits (4x), every entry valid;
#   vllm      vLLM NUM_BLOCKS 2048 and 32768 (test_flash_attn.py:25-34),
#             CACHE_SIZE 16384 (test_triton_decode_attention.py:16-41) with
#             block_tables = randint(0, num_blocks) (:96-98, :156-158): a pool
#             far larger than B x width and page ids drawn with repetition;
#   tail      one-to-one pages taken from the top of a pool larger than
#             B x width (large page ids at native sizes);
#   shared    odd requests reuse their predecessor's prefix pages (several
#             logical blocks map to one physical block, PagedAttention 4.4);
#   identity, reversed   the v1 fixtures edge-identity-table and
#             edge-reversed-table.
LAYOUTS = ("v1", "fa3", "wide", "vllm", "tail", "shared", "identity", "reversed")
LAYOUT_WEIGHTS = (5, 3, 1, 3, 2, 1, 1, 1)
VLLM_POOL_PAGES = (2048, 16384, 32768)

# Value hazards (values, fill).  large_logits: the v1 fixture
# edge-extreme-logits (q scaled by 30); negative_logits: FlashInfer
# test_paged_decode_extreme_negative_logits (:1462-1515, issues
# #4267/#4450/#4451/#4452); fill nan / inf: the slots no query may see hold
# NaN in production (vLLM v0.6.0 csrc/attention/attention_kernels.cu:420-430,
# issue #641), which FA3 never reads (hopper/paged_kv.h:148, :261-270).
HAZARDS = (
    ("large_logits", "randn"),
    ("negative_logits", "randn"),
    ("randn", "nan"),
    ("randn", "inf"),
    ("negative_logits", "nan"),
    ("large_logits", "inf"),
)

# The v1 support domain: v1 runtime HEAD_CONFIGS, PAGE_SIZES and FAMILY_S_Q.
# The first two crosses are the combinations two historical kernels failed on
# in the 2026-10-05 audit (64/8 with page 16 at q16: half the heads left
# unwritten; 32/8 with page 64 at q16: wrong KV mapping and writes past out).
V1_HEADS = ((32, 8), (64, 8), (128, 1))
V1_PAGE_SIZES = (16, 64, 128)
V1_S_Q = (1, 16)
V1_CROSS_FAILURES = ((64, 8, 16, 16), (32, 8, 64, 16))
V1_GRID = V1_CROSS_FAILURES + tuple(
    (h_q, h_kv, page_size, s_q)
    for h_q, h_kv in V1_HEADS
    for page_size in V1_PAGE_SIZES
    for s_q in V1_S_Q
    if (h_q, h_kv, page_size, s_q) not in V1_CROSS_FAILURES
)

# (category, private cases, public cases); private sums to 168, public to 44.
CATEGORY_COUNTS: tuple[tuple[str, int, int], ...] = (
    ("v1_contract", 30, 6),
    ("batch", 16, 4),
    ("lengths", 20, 5),
    ("s_q", 14, 4),
    ("heads", 24, 5),
    ("page_size", 14, 4),
    ("values", 12, 3),
    ("layout", 18, 5),
    ("sanitize", 12, 4),
    ("long_context", 8, 4),
)

# Long contexts (v4): single requests and mixed batches far above the v2 bound,
# with tables of exactly the needed width, shared prefix pages, tail pools and
# NaN or Inf in hidden slots.  Deterministic shapes; missing lengths are drawn.
LONG_PRIVATE: tuple[dict[str, Any], ...] = (
    {"s_q": 1, "b": 1, "h_q": 8, "h_kv": 1, "page_size": 1, "l_max": 131071, "layout": "v1", "lengths": (131071,)},
    {"s_q": 5, "b": 2, "h_q": 7, "h_kv": 1, "page_size": 16, "l_max": 262143, "layout": "v1", "fill": "nan"},
    {"s_q": 16, "b": 1, "h_q": 16, "h_kv": 2, "page_size": 64, "l_max": 524287, "layout": "v1", "lengths": (524287,)},
    {"s_q": 1, "b": 1, "h_q": 8, "h_kv": 1, "page_size": 256, "l_max": 1048576, "layout": "v1", "lengths": (1048576,)},
    {"s_q": 3, "b": 4, "h_q": 12, "h_kv": 1, "page_size": 2, "l_max": 65537, "layout": "shared"},
    {"s_q": 1, "b": 1, "h_q": 10, "h_kv": 2, "page_size": 32, "l_max": 200003, "layout": "v1", "fill": "inf", "lengths": (200003,)},
    {"s_q": 2, "b": 1, "h_q": 28, "h_kv": 4, "page_size": 8, "l_max": 98305, "layout": "tail", "fill": "inf", "lengths": (98305,)},
    {"s_q": 1, "b": 4, "h_q": 8, "h_kv": 1, "page_size": 16, "l_max": 131072, "layout": "v1", "values": "large_logits", "lengths": (131072, 33, 4097, 1025)},
)
LONG_PUBLIC: tuple[dict[str, Any], ...] = (
    {"s_q": 1, "b": 1, "h_q": 4, "h_kv": 1, "page_size": 1, "l_max": 70001, "layout": "v1", "lengths": (70001,)},
    {"s_q": 1, "b": 1, "h_q": 14, "h_kv": 2, "page_size": 32, "l_max": 300007, "layout": "v1", "fill": "nan", "lengths": (300007,)},
    {"s_q": 7, "b": 2, "h_q": 20, "h_kv": 5, "page_size": 4, "l_max": 40961, "layout": "shared"},
    {"s_q": 1, "b": 1, "h_q": 5, "h_kv": 1, "page_size": 128, "l_max": 786433, "layout": "v1", "lengths": (786433,)},
)
LONG_CASE_KV_CAP = 1 << 30

# Sanitizer subsets: deterministic shapes (lengths drawn where absent) covering
# the layouts most likely to produce a wild pointer.  Private, in order: B = 1;
# vLLM's Triton decode test (B 3, L 1025/1027, 32/8, page 1, CACHE_SIZE 16384,
# randint ids); FA3's kvcache test 6/3 with page 4 at (3, 799) and its 3x
# table; group 7 at S_q 16 (Qwen-2.5-7B); group 12 at S_q 5 (Mistral Large);
# page 256 crossings (FA2's block size); the 4x table; NaN in every hidden slot
# of a vLLM pool; FlashInfer's extreme negative logits; shared prefix pages;
# the 64/8 with page 16 q16 cross combination; and the large-offset pool
# (sanitizer only): pages at the top of a pool of 135,000 pages of 16 x 8 x 128,
# 2.21e9 K elements, the int32-offset overflow of SGLang PR #28788
# (test_triton_attention_kernels.py:866-873).
SANITIZE_PRIVATE: tuple[dict[str, Any], ...] = (
    {"s_q": 1, "b": 1, "h_q": 32, "h_kv": 8, "page_size": 16, "l_max": 2011, "layout": "v1", "lengths": (2011,)},
    {"s_q": 1, "b": 3, "h_q": 32, "h_kv": 8, "page_size": 1, "l_max": 1027, "layout": "vllm", "pool": 16384, "lengths": (1025, 1027, 1027)},
    {"s_q": 3, "b": 5, "h_q": 6, "h_kv": 3, "page_size": 4, "l_max": 799, "layout": "fa3"},
    {"s_q": 16, "b": 4, "h_q": 28, "h_kv": 4, "page_size": 16, "l_max": 523, "layout": "tail"},
    {"s_q": 5, "b": 3, "h_q": 96, "h_kv": 8, "page_size": 128, "l_max": 1328, "layout": "fa3"},
    {"s_q": 2, "b": 2, "h_q": 6, "h_kv": 6, "page_size": 256, "l_max": 1025, "layout": "v1", "lengths": (257, 1025)},
    {"s_q": 1, "b": 5, "h_q": 64, "h_kv": 8, "page_size": 16, "l_max": 463, "layout": "wide"},
    {"s_q": 9, "b": 5, "h_q": 24, "h_kv": 8, "page_size": 8, "l_max": 500, "layout": "vllm", "pool": 2048, "fill": "nan"},
    {"s_q": 4, "b": 5, "h_q": 32, "h_kv": 8, "page_size": 16, "l_max": 339, "layout": "fa3", "values": "negative_logits"},
    {"s_q": 1, "b": 6, "h_q": 32, "h_kv": 8, "page_size": 16, "l_max": 1328, "layout": "shared"},
    {"s_q": 16, "b": 5, "h_q": 64, "h_kv": 8, "page_size": 16, "l_max": 1031, "layout": "v1"},
    {"s_q": 1, "b": 2, "h_q": 32, "h_kv": 8, "page_size": 16, "l_max": 1000, "layout": "large_offset", "pool": 135_000, "native": False},
)
# The public instance gets its own native sanitizer cases (same hazard classes):
# B = 1 in a vLLM pool at page 1; FA3's 6/1 at page 1, (1, 339), 3x table;
# group 7 at S_q 3 with NaN hidden slots in a tail pool; shared prefix pages
# with group 6 at page 8 and extreme negative logits.
SANITIZE_PUBLIC: tuple[dict[str, Any], ...] = (
    {"s_q": 1, "b": 1, "h_q": 16, "h_kv": 1, "page_size": 1, "l_max": 2011, "layout": "vllm", "pool": 16384},
    {"s_q": 1, "b": 5, "h_q": 6, "h_kv": 1, "page_size": 1, "l_max": 339, "layout": "fa3"},
    {"s_q": 3, "b": 3, "h_q": 28, "h_kv": 4, "page_size": 16, "l_max": 799, "layout": "tail", "fill": "nan"},
    {"s_q": 2, "b": 4, "h_q": 12, "h_kv": 2, "page_size": 8, "l_max": 463, "layout": "shared", "values": "negative_logits"},
)

# Size caps (bytes of K plus V).  Every native case stays at or below 64 MiB
# and the native suite as a whole below 3 GB, so it runs in a few minutes:
# drawn vLLM-style pools are cut to 16 MiB (still at least twice B x width) and
# drawn context lengths lean short (half of the draws are log-uniform).  The
# single sanitizer-only case must exceed 2**31 K elements and stay within the
# runtime's 10 GiB sanitizer-only bound.
CASE_KV_CAP = 64 << 20
VLLM_POOL_KV_CAP = 16 << 20
NATIVE_KV_TOTAL_CAP = 6_000_000_000
LARGE_OFFSET_MIN_K_ELEMENTS = 1 << 31
DRAW_ATTEMPTS = 10_000


def _load_runtime():
    path = TASK_ROOT / "runtime/paged_gqa_decode_runtime.py"
    spec = importlib.util.spec_from_file_location("edge_generator_runtime", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def default_seed(suite_id: str) -> int:
    digest = hashlib.sha256(f"{suite_id}|{TASK_ID}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _blocks(length: int, page_size: int) -> int:
    return -(-length // page_size)


def _table(
    rng: random.Random, runtime, layout: str, b: int, needed: int, page_size: int,
    h_kv: int, pool: int | None,
) -> tuple[str, int, int] | None:
    """(table_mode, table_width, pool_pages) for one layout, or None if it cannot fit."""
    if layout in ("v1", "shared", "identity", "reversed"):
        mode = "perm" if layout == "v1" else layout
        return mode, needed, b * needed
    if layout == "fa3":
        return "perm", 3 * needed, 3 * b * needed
    if layout == "wide":
        width = runtime.MAX_WIDTH_FACTOR * needed
        return "perm", width, b * width
    if layout == "tail":
        count = b * needed
        return "tail_perm", needed, count + rng.randint(1, max(1, 2 * count))
    if layout == "vllm":
        if pool is None:
            page_bytes = runtime.kv_cache_bytes(1, page_size, h_kv)
            pool = min(
                rng.choice(VLLM_POOL_PAGES), VLLM_POOL_KV_CAP // page_bytes, runtime.MAX_POOL_PAGES
            )
            if pool < 2 * b * needed:
                return None
        return "random", needed, pool
    if layout == "large_offset":
        assert pool is not None
        return "tail_perm", needed, pool
    raise ValueError(layout)


def _case(
    rng: random.Random, runtime, category: str, *, s_q: int, b: int, h_q: int, h_kv: int,
    page_size: int, l_max: int, lengths: list[int], layout: str, values: str = "randn",
    fill: str = "randn", pool: int | None = None, sanitize: bool = False, native: bool = True,
    kv_cap: int = CASE_KV_CAP,
) -> dict[str, Any] | None:
    table = _table(rng, runtime, layout, b, _blocks(l_max, page_size), page_size, h_kv, pool)
    if table is None:
        return None
    mode, width, pool_pages = table
    case = {
        "case_id": "pending",
        "category": category,
        "s_q": s_q,
        "b": b,
        "l_max": l_max,
        "h_q": h_q,
        "h_kv": h_kv,
        "page_size": page_size,
        "lengths": list(lengths),
        "table_mode": mode,
        "pool_pages": pool_pages,
        "table_width": width,
        "fill": fill,
        "values": values,
        "seed": 0,
        "sanitize": sanitize,
        "native": native,
    }
    if native and runtime.kv_cache_bytes(pool_pages, page_size, h_kv) > kv_cap:
        return None
    try:
        runtime.validate_edge_case(case)
    except ValueError:
        return None
    return case


def _lengths(
    rng: random.Random, b: int, s_q: int, l_max: int, page_size: int, pinned: tuple[int, ...] = ()
) -> list[int]:
    """B lengths in [S_q, L_max]: the pinned values first, then a mix of the
    contract corners (S_q, S_q + 1, L_max), page boundaries k*ps - 1 / k*ps /
    k*ps + 1, and FA3's uniform randint draw."""
    values = [value for value in pinned if s_q <= value <= l_max][:b]
    while len(values) < b:
        roll = rng.random()
        if roll < 0.1:
            value = s_q
        elif roll < 0.2:
            value = s_q + 1
        elif roll < 0.3:
            value = l_max
        elif roll < 0.5:
            value = rng.randint(1, max(1, l_max // page_size)) * page_size + rng.choice((-1, 0, 1))
        else:
            value = rng.randint(s_q, l_max)
        if s_q <= value <= l_max:
            values.append(value)
    rng.shuffle(values)
    return values


def _layout(rng: random.Random) -> str:
    return rng.choices(LAYOUTS, weights=LAYOUT_WEIGHTS)[0]


def _hazard(rng: random.Random) -> tuple[str, str]:
    """Background (values, fill) mix for categories that do not sweep hazards."""
    roll = rng.random()
    fill = "nan" if roll < 0.2 else "inf" if roll < 0.27 else "randn"
    roll = rng.random()
    values = "large_logits" if roll < 0.08 else "negative_logits" if roll < 0.12 else "randn"
    return values, fill


def _l_max(rng: random.Random, s_q: int, top: int) -> int:
    """A context length in [S_q, top]: uniform or log-uniform, half each."""
    top = max(s_q, top)
    if rng.random() < 0.5:
        return rng.randint(s_q, top)
    return min(top, max(s_q, round(math.exp(rng.uniform(math.log(s_q), math.log(top))))))


def _draw(build: Callable[[], dict[str, Any] | None]) -> dict[str, Any]:
    for _ in range(DRAW_ATTEMPTS):
        case = build()
        if case is not None:
            return case
    raise RuntimeError("could not draw a case within the edge-suite size bounds")


def _category_case(
    rng: random.Random, runtime, category: str, ordinal: int, rotation: int
) -> dict[str, Any]:
    def generic(*, s_q: int, b: int, heads: tuple[int, int], page_size: int, l_max: int,
                pinned: tuple[int, ...] = (), layout: str | None = None,
                hazard: tuple[str, str] | None = None) -> dict[str, Any] | None:
        chosen = layout or _layout(rng)
        values, fill = hazard or _hazard(rng)
        return _case(
            rng, runtime, category, s_q=s_q, b=b, h_q=heads[0], h_kv=heads[1],
            page_size=page_size, l_max=l_max,
            lengths=_lengths(rng, b, s_q, l_max, page_size, pinned),
            layout=chosen, values=values, fill=fill,
        )

    if category == "v1_contract":
        if ordinal < len(V1_GRID):
            h_q, h_kv, page_size, s_q = V1_GRID[(rotation + ordinal) % len(V1_GRID)]
        else:
            (h_q, h_kv), page_size, s_q = (
                rng.choice(V1_HEADS), rng.choice(V1_PAGE_SIZES), rng.choice(V1_S_Q)
            )

        def build():
            b = rng.choice(BATCHES)
            if ordinal < len(V1_GRID):
                # The grid sweep uses multi-page contexts, as the 2026-10-05
                # cross-combination failures did.
                l_max = rng.randint(max(s_q, 64), 523 if b >= 100 else 4095)
            else:
                l_max = _l_max(rng, s_q, 523 if b >= 100 else 32768)
            # v1 layouts and values only: N(0,1) everywhere, one-to-one tables.
            layout = rng.choice(("v1", "v1", "v1", "identity", "reversed"))
            values = "large_logits" if rng.random() < 0.15 else "randn"
            return _case(
                rng, runtime, category, s_q=s_q, b=b, h_q=h_q, h_kv=h_kv,
                page_size=page_size, l_max=l_max,
                lengths=_lengths(rng, b, s_q, l_max, page_size), layout=layout,
                values=values,
            )
    elif category == "batch":
        b = BATCHES[(rotation + ordinal) % len(BATCHES)]

        def build():
            s_q = rng.choice(S_Q_VALUES)
            top = 523 if b >= 100 else 2011 if b >= 17 else 4095
            return generic(
                s_q=s_q, b=b, heads=rng.choice(HEAD_CONFIGS),
                page_size=rng.choice(PAGE_SIZES), l_max=_l_max(rng, s_q, top),
            )
    elif category == "lengths":
        group = LENGTH_GROUPS[(rotation + ordinal) % len(LENGTH_GROUPS)]

        def build():
            heads = rng.choice(HEAD_CONFIGS)
            page_size = rng.choice(PAGE_SIZES)
            if group in (("s_q",), ("s_q+1",)):
                s_q = rng.choice(S_Q_VALUES)
                base = s_q if group == ("s_q",) else s_q + 1
                b = rng.choice(SMALL_BATCHES)
                l_max = base if rng.random() < 0.5 else base + rng.randint(1, 64)
                lengths = [base] * b
            elif group == (1,):
                s_q, b = 1, rng.choice((2, 3, 5, 9, 17, 33))
                l_max = rng.choice((1, 2, 17, 128))
                lengths = [1] * b
            else:
                s_q = rng.choice([value for value in S_Q_VALUES if value <= min(group)])
                if max(group) >= 16383:
                    b = rng.choice((1, 2))
                    lengths = list(group) + [rng.randint(s_q, max(group)) for _ in range(b - 1)]
                else:
                    b = len(group) + rng.randint(0, 3)
                    lengths = list(group) + [rng.choice(group) for _ in range(b - len(group))]
                rng.shuffle(lengths)
                l_max = min(32768, max(group) + rng.choice((0, 0, 1, page_size)))
            layout = _layout(rng)
            values, fill = _hazard(rng)
            return _case(
                rng, runtime, category, s_q=s_q, b=b, h_q=heads[0], h_kv=heads[1],
                page_size=page_size, l_max=l_max, lengths=lengths, layout=layout,
                values=values, fill=fill,
            )
    elif category == "s_q":
        s_q = S_Q_VALUES[(rotation + ordinal) % len(S_Q_VALUES)]

        def build():
            return generic(
                s_q=s_q, b=rng.choice(SMALL_BATCHES), heads=rng.choice(HEAD_CONFIGS),
                page_size=rng.choice(PAGE_SIZES), l_max=_l_max(rng, s_q, 2011),
                pinned=(s_q, s_q + 1),
            )
    elif category == "heads":
        heads = HEAD_CONFIGS[(rotation + ordinal) % len(HEAD_CONFIGS)]

        def build():
            s_q = rng.choice(S_Q_VALUES)
            return generic(
                s_q=s_q, b=rng.choice(SMALL_BATCHES), heads=heads,
                page_size=rng.choice(PAGE_SIZES), l_max=_l_max(rng, s_q, 2011),
            )
    elif category == "page_size":
        page_size = PAGE_SIZES[(rotation + ordinal) % len(PAGE_SIZES)]

        def build():
            s_q = rng.choice(S_Q_VALUES)
            boundary = rng.randint(1, max(1, min(16, 4096 // page_size))) * page_size
            pinned = (page_size - 1, page_size, page_size + 1, boundary - 1, boundary, boundary + 1)
            l_max = boundary + 1 + rng.choice((0, 0, rng.randint(0, page_size)))
            return generic(
                s_q=s_q, b=rng.choice((6, 7, 9, 12, 17)), heads=rng.choice(HEAD_CONFIGS),
                page_size=page_size, l_max=max(s_q, l_max), pinned=pinned,
            )
    elif category == "values":
        hazard = HAZARDS[(rotation + ordinal) % len(HAZARDS)]

        def build():
            s_q = rng.choice(S_Q_VALUES)
            return generic(
                s_q=s_q, b=rng.choice(SMALL_BATCHES), heads=rng.choice(HEAD_CONFIGS),
                page_size=rng.choice(PAGE_SIZES), l_max=_l_max(rng, s_q, 2011),
                layout=rng.choice(("v1", "fa3", "vllm", "tail", "shared", "wide")),
                hazard=hazard,
            )
    elif category == "layout":
        layout = LAYOUTS[(rotation + ordinal) % len(LAYOUTS)]

        def build():
            s_q = rng.choice(S_Q_VALUES)
            return generic(
                s_q=s_q, b=rng.choice(SMALL_BATCHES), heads=rng.choice(HEAD_CONFIGS),
                page_size=rng.choice(PAGE_SIZES), l_max=_l_max(rng, s_q, 2011),
                layout=layout,
            )
    else:
        raise ValueError(category)
    return _draw(build)


def _sanitize_case(rng: random.Random, runtime, spec: dict[str, Any]) -> dict[str, Any]:
    s_q, b, l_max = spec["s_q"], spec["b"], spec["l_max"]
    lengths = list(spec.get("lengths") or _lengths(rng, b, s_q, l_max, spec["page_size"], (l_max,)))
    native = bool(spec.get("native", True))
    case = _case(
        rng, runtime, "sanitize", s_q=s_q, b=b, h_q=spec["h_q"], h_kv=spec["h_kv"],
        page_size=spec["page_size"], l_max=l_max, lengths=lengths, layout=spec["layout"],
        values=spec.get("values", "randn"), fill=spec.get("fill", "randn"),
        pool=spec.get("pool"), sanitize=True, native=native,
    )
    if case is None:
        raise RuntimeError(f"sanitizer case does not fit the edge-suite bounds: {spec}")
    if not native:
        k_elements = case["pool_pages"] * case["page_size"] * case["h_kv"] * runtime.HEAD_DIM
        first_page = case["pool_pages"] - case["b"] * case["table_width"]
        first_offset = first_page * case["page_size"] * case["h_kv"] * runtime.HEAD_DIM
        if k_elements <= LARGE_OFFSET_MIN_K_ELEMENTS or first_offset < LARGE_OFFSET_MIN_K_ELEMENTS:
            raise RuntimeError("the large-offset case must place every page beyond 2**31 elements")
    return case


def _long_case(rng: random.Random, runtime, spec: dict[str, Any]) -> dict[str, Any]:
    s_q, b, l_max = spec["s_q"], spec["b"], spec["l_max"]
    lengths = list(spec.get("lengths") or _lengths(rng, b, s_q, l_max, spec["page_size"], (l_max,)))
    case = _case(
        rng, runtime, "long_context", s_q=s_q, b=b, h_q=spec["h_q"], h_kv=spec["h_kv"],
        page_size=spec["page_size"], l_max=l_max, lengths=lengths, layout=spec["layout"],
        values=spec.get("values", "randn"), fill=spec.get("fill", "randn"),
        pool=spec.get("pool"), kv_cap=LONG_CASE_KV_CAP,
    )
    if case is None:
        raise RuntimeError(f"long-context case does not fit the edge-suite bounds: {spec}")
    return case


# The public instance never uses a scored (H_q, H_kv, page size) combination;
# such a draw moves to the page size here.
PUBLIC_PAGE_SIZE_SWAP = {
    (32, 8, 16): 32,
    (28, 4, 16): 8,
    (64, 8, 64): 32,
    (96, 8, 128): 64,
    (16, 1, 1): 2,
}


def _swap_public_page_size(case: dict[str, Any]) -> None:
    swap = PUBLIC_PAGE_SIZE_SWAP.get((case["h_q"], case["h_kv"], case["page_size"]))
    if swap is None:
        return
    old_size = case["page_size"]
    factor = case["table_width"] // _blocks(case["l_max"], old_size)
    width = factor * _blocks(case["l_max"], swap)
    if case["table_mode"] in ("tail_perm", "random"):
        case["pool_pages"] = max(
            -(-case["pool_pages"] * old_size // swap), 2 * case["b"] * width
        )
    else:
        case["pool_pages"] = case["b"] * width
    case["table_width"] = width
    case["page_size"] = swap


def generate(suite_id: str, *, public: bool, seed: int | None = None) -> dict[str, Any]:
    runtime = _load_runtime()
    rng = random.Random(default_seed(SEED_SUITE_IDS.get(suite_id, suite_id)) if seed is None else seed)
    prefix = "public-edge" if public else "edge"
    sanitize_specs = SANITIZE_PUBLIC if public else SANITIZE_PRIVATE
    cases: list[dict[str, Any]] = []
    seeds: set[int] = set()
    index = 0
    for category, private_count, public_count in CATEGORY_COUNTS:
        count = public_count if public else private_count
        # The private instance sweeps every list from its start, so a full
        # sweep is guaranteed; the public one starts at a drawn offset.
        rotation = rng.randrange(1 << 16) if public else 0
        for ordinal in range(count):
            if category == "sanitize":
                case = _sanitize_case(rng, runtime, sanitize_specs[ordinal])
            elif category == "long_context":
                case = _long_case(rng, runtime, (LONG_PUBLIC if public else LONG_PRIVATE)[ordinal])
            else:
                case = _category_case(rng, runtime, category, ordinal, rotation)
            data_seed = rng.getrandbits(63)
            while data_seed in seeds:
                data_seed = rng.getrandbits(63)
            seeds.add(data_seed)
            index += 1
            case["case_id"] = f"{prefix}-{index:03d}-{category.replace('_', '-')}"
            case["seed"] = data_seed
            if public:
                _swap_public_page_size(case)
            runtime.validate_edge_case(case)
            cases.append(case)
    native_bytes = sum(
        runtime.kv_cache_bytes(case["pool_pages"], case["page_size"], case["h_kv"])
        for case in cases if case["native"]
    )
    if native_bytes > NATIVE_KV_TOTAL_CAP:
        raise RuntimeError(f"native edge K/V total {native_bytes} exceeds {NATIVE_KV_TOTAL_CAP}")
    if public and not all(case["native"] for case in cases):
        raise RuntimeError("every public edge case must run natively")
    return {"schema_version": SCHEMA_VERSION, "suite_id": suite_id, "cases": cases}


def encode(payload: dict[str, Any]) -> bytes:
    """Strict JSON with one case per line, so diffs and reviews stay readable."""
    rows = [
        json.dumps(case, sort_keys=True, allow_nan=False, separators=(", ", ": "))
        for case in payload["cases"]
    ]
    lines = [
        "{",
        f'  "schema_version": {json.dumps(payload["schema_version"])},',
        f'  "suite_id": {json.dumps(payload["suite_id"])},',
        '  "cases": [',
        *(f"    {row}," for row in rows[:-1]),
        f"    {rows[-1]}",
        "  ]",
        "}",
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def summary(payload: dict[str, Any]) -> list[str]:
    runtime = sys.modules.get("edge_generator_runtime") or _load_runtime()
    cases = payload["cases"]

    def kv(case: dict[str, Any]) -> int:
        return runtime.kv_cache_bytes(case["pool_pages"], case["page_size"], case["h_kv"])

    native = [case for case in cases if case["native"]]
    sanitize = [case for case in cases if case["sanitize"]]
    counts = Counter(case["category"] for case in cases)
    lines = [
        f"{payload['suite_id']}: {len(cases)} cases, {len(native)} native, "
        f"{len(sanitize)} sanitize ({sum(not case['native'] for case in cases)} sanitizer-only)",
        "  categories: " + ", ".join(f"{name}={counts[name]}" for name, *_ in CATEGORY_COUNTS),
        f"  native K+V: total {sum(map(kv, native)) / 1e9:.3f} GB, "
        f"max {max(map(kv, native)) / (1 << 20):.1f} MiB, "
        f"{sum(kv(case) > CASE_KV_CAP for case in native)} above {CASE_KV_CAP >> 20} MiB",
    ]
    for case in sanitize:
        lines.append(
            f"  sanitize {case['case_id']}: s_q={case['s_q']} b={case['b']} "
            f"heads={case['h_q']}/{case['h_kv']} ps={case['page_size']} l_max={case['l_max']} "
            f"{case['table_mode']} W={case['table_width']} P={case['pool_pages']} "
            f"fill={case['fill']} values={case['values']} native={case['native']} "
            f"K+V={kv(case) / (1 << 20):.1f} MiB"
        )
    return lines


TARGETS: tuple[tuple[str, Path, bool], ...] = (
    (PRIVATE_SUITE_ID, PRIVATE_PATH, False),
    (PUBLIC_SUITE_ID, PUBLIC_PATH, True),
)


def _committed(path: Path, public: bool) -> Path | None:
    """The instance --check compares: the repository copy, else (a tree without it, such as
    the public tree) the Judge-only instance assets/fetch.sh places under
    <INFRATHON_ASSETS_ROOT>/<task directory>/judge/; None when neither exists."""
    if public or path.exists():
        return path
    assets_root = os.environ.get("INFRATHON_ASSETS_ROOT") or "~/.cache/infrathon/assets"
    fetched = Path(assets_root).expanduser() / TASK_ROOT.name / "judge" / path.name
    return fetched if fetched.is_file() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    drift = False
    for suite_id, path, public in TARGETS:
        payload = generate(suite_id, public=public)
        encoded = encode(payload)
        source = path if args.write else _committed(path, public)
        if source is None:
            print(f"{suite_id}: {path.relative_to(TASK_ROOT)} not present "
                  "(Judge-only dataset asset; run assets/fetch.sh), not compared")
            continue
        current = source.read_bytes() if source.exists() else b""
        if current != encoded:
            drift = True
            print(f"{path.relative_to(TASK_ROOT)}: differs from the seeded generator")
            if args.write:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(encoded)
                if not public:
                    path.chmod(0o600)
        digest = hashlib.sha256(source.read_bytes() if source.exists() else encoded).hexdigest()
        print(f"{suite_id}: {path.relative_to(TASK_ROOT)} sha256={digest}")
        for line in summary(payload):
            print(line)
    if args.check and drift:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
