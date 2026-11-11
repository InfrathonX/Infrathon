#!/usr/bin/env python3
"""Generate the zero-weight edge-case manifests for standalone sparse-MLA prefill.

Edge cases are untimed correctness and memory-safety hard gates; they never
enter the score.  Categories (plan section 3.4; U = upstream test lists of
FlashMLA ``tests/test_flash_mla_sparse_prefill.py`` at ``ba89a34``, the last
commit with Hopper support, restricted to the three deployed geometries):

=====================  =======  ===============================================
category               private  content
=====================  =======  ===============================================
upstream_correctness        90  U1: 10 (s_kv, topk) x s_q {1, 62, 213}, no sink/length
upstream_features          168  U2: 7 (s_kv, topk) x s_q {62, 213} x sink x length
upstream_all_invalid        12  U3: (1,128,128) (1,256,256) (1234,4321,2048*) (4096,2048,2048)
upstream_empty_blocks       12  U4: (s_kv, topk) in {(32,2048), (64,2048*)} x s_q {1, 1024}
upstream_large_s_q           1  U5: (70000, 256, 256) V4-Flash, sanitizer only
scored_semantics            17  L1: every anchor shape with the edge semantics
scored_precision            17  L2: anchor shapes with zero q/kv bias
odd_sizes                  112  O: odd s_q / s_kv sweep over the four families
v1_fixture                   8  V: the eight v1 edge fixtures with new seeds
causal                      20  C: causal indices with stale slots past topk_length
large_kv                     4  K: keys at each family's context limit
tp_heads                     9  T: tensor-parallel head counts (16 to 64)
=====================  =======  ===============================================

``*`` marks upstream corners adapted to the task envelope (topk 4096 and 8192
become 2048).  The geometry order is upstream's loop order (d_qk 512 then 576,
h_q 128 then 64); the GLM-5 geometry (576, 64) enters through the anchor,
odd-size and serving categories.

The public instance is a separate, smaller set (``PUBLIC_CATEGORY_COUNTS``)
that shares no (family, s_q, s_kv, topk) with the private instance or the
scored suite.

The ``sanitize`` flag marks the subset the Judge runs first under
``compute-sanitizer --tool memcheck`` (17 private, 7 public).  The one
``"native": false`` case is upstream's s_q = 70000 corner: its q and out hold
more than 2**31 elements, so it runs only under the sanitizer unless the
sanitizer pass cannot establish it.

    scripts/generate_edge_manifest.py --check     # regenerate and compare
    scripts/generate_edge_manifest.py --write     # rewrite both instances

The private instance is read from and written to
``<INFRATHON_ASSETS_ROOT>/<task directory>/judge/edge_cases.json``.

``--write`` mints new suite identities when the output changes; the
``frozen_instance`` and ``sha256`` records in ``portable/*.json`` must then be
refreshed (``scripts/refresh_portable_manifests.py``).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

TASK_ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "standalone_sparse_mla_prefill_h100_scratch_v2"
PRIVATE_SUITE_ID = "edge-holdout-v3"
PUBLIC_SUITE_ID = "edge-public-v3"
# The private instance is a Judge-only asset: it is written to and checked against
# <INFRATHON_ASSETS_ROOT>/<task directory>/judge/edge_cases.json, never the repository.
PRIVATE_PATH = Path("judge/edge_cases.json")
PUBLIC_PATH = TASK_ROOT / "images/work/benchmark/public_edge_cases.json"
SCHEMA_VERSION = 3

# Upstream loop order over the deployed geometries: d_qk in [512, 576], h_q in
# [128, 64]; (576, 64) is skipped.
UPSTREAM_FAMILY_ORDER = ("v4_pro", "v4_flash", "v32")
# correctness_cases (:62-89)
UPSTREAM_PAIRS10 = (
    (128, 128), (256, 256), (512, 512),
    (592, 128), (1840, 256), (1592, 384), (1521, 512),
    (95, 128), (153, 256), (114, 384),
)
UPSTREAM_CORRECTNESS_S_Q = (1, 62, 213)
# correctness_cases_with_features (:91-111), without the 15f13e5 duplicate loop
UPSTREAM_PAIRS7 = UPSTREAM_PAIRS10[3:]
UPSTREAM_FEATURES_S_Q = (62, 213)
# corner_cases (:113-144); the starred values are adapted to topk <= 2048.
UPSTREAM_ALL_INVALID = ((1, 128, 128), (1, 256, 256), (1234, 4321, 2048), (4096, 2048, 2048))
UPSTREAM_EMPTY_BLOCKS = ((32, 2048), (64, 2048))
UPSTREAM_EMPTY_BLOCKS_S_Q = (1, 1024)
UPSTREAM_LARGE_S_Q = ("v4_flash", 70000, 256, 256)
UPSTREAM_SM_SCALE = 0.5

# Odd-size sweep (the 2026-10-06 probe's O set).
PRIVATE_ODD_S_Q = (
    1, 2, 3, 5, 7, 9, 15, 17, 31, 33, 63, 65, 127, 129, 255, 257, 511, 513,
    767, 769, 1023, 1025, 1535, 1537, 2047, 2049, 3071, 4095,
)
PUBLIC_ODD_S_Q = (4, 11, 19, 47, 71, 131, 223, 449, 1021, 2039)
ODD_TOPK = (128, 256, 384, 640, 896, 1024, 1152, 1408, 1664, 1920, 2048)
ODD_CONTENTS: dict[str, dict[str, Any]] = {
    # Upstream performance-style content at an odd shape.
    "scored_like": dict(bias="upstream", index_mode="upstream", invalid_rows="none",
                        duplicates=False, scramble=False, sink_mode="upstream", length_mode="none"),
    # Every semantic hazard at once.
    "nasty": dict(bias="upstream", index_mode="sprinkle", invalid_rows="some",
                  duplicates=True, scramble=True, sink_mode="extreme", length_mode="edge"),
    # Neither optional argument, zero bias, padded invalid tail scrambled in.
    "bare": dict(bias="zero", index_mode="upstream", invalid_rows="none",
                 duplicates=False, scramble=True, sink_mode="none", length_mode="none"),
}
ODD_CONTENT_NAMES = tuple(ODD_CONTENTS)
# (family, s_q) -> s_kv for the odd cases that also run under the sanitizer.
PRIVATE_ODD_SANITIZE = {("v32", 7): 1, ("v4_pro", 129): 131071}
PUBLIC_ODD_SANITIZE = {11: 7}

# L1: the scored shapes with the edge semantics.
SCORED_SEMANTICS = dict(bias="upstream", index_mode="sprinkle", invalid_rows="sparse",
                        duplicates=True, scramble=True, sink_mode="extreme", length_mode="edge")

# V: the eight v1 edge fixtures (v1 runtime EDGE_CASES), re-expressed in the v2
# content vocabulary; the v1 family names of the (512, 64) and (512, 128) geometries
# are the v2 v4_flash and v4_pro.
V1_FIXTURES: tuple[dict[str, Any], ...] = (
    dict(name="edge-all-invalid", family="v32", s_q=8, s_kv=128, topk=128, sm_scale=0.5,
         index_mode="upstream", invalid_rows="all", scramble=True, duplicates=False,
         sink_mode="extreme", length_mode="edge"),
    dict(name="edge-scrambled-oob", family="v32", s_q=62, s_kv=592, topk=128, sm_scale=0.5,
         index_mode="upstream", invalid_rows="some", scramble=True, duplicates=True,
         sink_mode="extreme", length_mode="edge"),
    dict(name="edge-irregular", family="v32", s_q=211, s_kv=1521, topk=512, sm_scale=576 ** -0.55,
         index_mode="upstream", invalid_rows="none", scramble=True, duplicates=True,
         sink_mode="finite", length_mode="edge"),
    dict(name="edge-peaked-nosink", family="v4_flash", s_q=213, s_kv=1592, topk=384, sm_scale=4.0,
         index_mode="valid_pad", invalid_rows="none", scramble=True, duplicates=True,
         sink_mode="none", length_mode="none"),
    dict(name="edge-length-nosink", family="v4_flash", s_q=128, s_kv=2048, topk=512, sm_scale=0.5,
         index_mode="valid_pad", invalid_rows="none", scramble=False, duplicates=False,
         sink_mode="none", length_mode="edge"),
    dict(name="edge-oversubscribed", family="v4_pro", s_q=256, s_kv=64, topk=2048, sm_scale=0.5,
         index_mode="upstream", invalid_rows="some", scramble=True, duplicates=True,
         sink_mode="extreme", length_mode="edge"),
    dict(name="edge-tiny-kv", family="v4_pro", s_q=17, s_kv=95, topk=128, sm_scale=1.0,
         index_mode="upstream", invalid_rows="some", scramble=True, duplicates=False,
         sink_mode="none", length_mode="none"),
    dict(name="edge-wide-index", family="v32", s_q=64, s_kv=131072, topk=2048, sm_scale=0.5,
         index_mode="valid_pad", invalid_rows="none", scramble=True, duplicates=True,
         sink_mode="finite", length_mode="edge"),
)
V1_SANITIZE = ("edge-wide-index",)

# Public instance.  Every shape differs from the private instance and the
# scored suite; a test asserts that.
PUBLIC_CELLS = (
    ("v32", 1536, 1536, 12288), ("v32", 1536, 1024, 73728),
    ("v4_flash", 2048, 384, 12288), ("v4_flash", 2048, 768, 57344),
    ("v4_pro", 1536, 768, 12288), ("v4_pro", 1536, 1536, 57344),
)
PUBLIC_PAIRS = (
    (136, 128), (264, 256), (520, 512),
    (600, 128), (1848, 256), (1600, 384), (1529, 512),
    (97, 128), (149, 256), (119, 384),
)
PUBLIC_CORRECTNESS_S_Q = (1, 59)
PUBLIC_FEATURES_S_Q = (47, 197)
PUBLIC_ALL_INVALID = ((1, 130, 128), (1, 250, 256), (1111, 4099, 2048), (3072, 2040, 2048))
PUBLIC_EMPTY_BLOCKS = ((40, 2048), (56, 2048))
PUBLIC_EMPTY_BLOCKS_S_Q = (1, 768)
# (v1 fixture, s_q, s_kv): four fixtures at public shapes.
PUBLIC_FIXTURES = (
    ("edge-all-invalid", 9, 132), ("edge-irregular", 207, 1517),
    ("edge-oversubscribed", 240, 60), ("edge-wide-index", 72, 131000),
)

# (category, count); generation order.
CATEGORY_COUNTS: tuple[tuple[str, int], ...] = (
    ("upstream_correctness", 90),
    ("upstream_features", 168),
    ("upstream_all_invalid", 12),
    ("upstream_empty_blocks", 12),
    ("upstream_large_s_q", 1),
    ("scored_semantics", 17),
    ("scored_precision", 17),
    ("odd_sizes", 112),
    ("v1_fixture", 8),
    ("causal", 20),
    ("large_kv", 4),
    ("tp_heads", 9),
)
PUBLIC_CATEGORY_COUNTS: tuple[tuple[str, int], ...] = (
    ("correctness", 20),
    ("features", 14),
    ("all_invalid", 4),
    ("empty_blocks", 4),
    ("semantics", 6),
    ("precision", 3),
    ("odd_sizes", 10),
    ("fixture", 4),
    ("causal", 4),
    ("large_kv", 1),
    ("tp_heads", 3),
)
PRIVATE_CASE_COUNT = sum(count for _name, count in CATEGORY_COUNTS)
PUBLIC_CASE_COUNT = sum(count for _name, count in PUBLIC_CATEGORY_COUNTS)
PRIVATE_SANITIZE_COUNT = 17
PUBLIC_SANITIZE_COUNT = 7

# Upstream-shape sanitizer picks: (category, family, s_q, s_kv, topk[, sink, length]).
PRIVATE_UPSTREAM_SANITIZE = {
    ("upstream_correctness", "v32", 1, 95, 128),
    ("upstream_correctness", "v4_flash", 213, 114, 384),
    ("upstream_features", "v4_pro", 62, 153, 256, True, True),
    ("upstream_all_invalid", "v32", 1, 128, 128),
    ("upstream_all_invalid", "v4_flash", 4096, 2048, 2048),
    ("upstream_empty_blocks", "v4_pro", 1, 32, 2048),
    ("upstream_empty_blocks", "v32", 1024, 64, 2048),
}
# L1 cases that also run under the sanitizer: s_kv = 8192 of every family.
PRIVATE_SEMANTICS_SANITIZE = {("v32", 8192), ("v4_flash", 8192), ("v4_pro", 8192)}


def _load_runtime():
    path = TASK_ROOT / "runtime/sparse_mla_prefill_runtime.py"
    spec = importlib.util.spec_from_file_location("edge_generator_runtime", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def default_seed(suite_id: str) -> int:
    digest = hashlib.sha256(f"{suite_id}|{TASK_ID}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


UPSTREAM_CONTENT: dict[str, Any] = dict(
    bias="upstream", index_mode="upstream", invalid_rows="none", duplicates=False,
    scramble=False, sink_mode="none", length_mode="none",
)


def _case(category: str, family: str, s_q: int, s_kv: int, topk: int, sm_scale: float,
          content: dict[str, Any], *, sanitize: bool = False, native: bool = True) -> dict[str, Any]:
    return {
        "category": category, "family": family, "s_q": int(s_q), "s_kv": int(s_kv),
        "topk": int(topk), "sm_scale": float(sm_scale), **dict(content),
        "sanitize": bool(sanitize), "native": bool(native),
    }


def upstream_cases() -> list[dict[str, Any]]:
    picks = PRIVATE_UPSTREAM_SANITIZE
    cases: list[dict[str, Any]] = []
    for family in UPSTREAM_FAMILY_ORDER:
        for s_kv, topk in UPSTREAM_PAIRS10:
            for s_q in UPSTREAM_CORRECTNESS_S_Q:
                key = ("upstream_correctness", family, s_q, s_kv, topk)
                cases.append(_case("upstream_correctness", family, s_q, s_kv, topk,
                                   UPSTREAM_SM_SCALE, UPSTREAM_CONTENT, sanitize=key in picks))
    for family in UPSTREAM_FAMILY_ORDER:
        for s_kv, topk in UPSTREAM_PAIRS7:
            for s_q in UPSTREAM_FEATURES_S_Q:
                for sink in (False, True):
                    for length in (False, True):
                        content = dict(UPSTREAM_CONTENT, sink_mode="upstream" if sink else "none",
                                       length_mode="upstream" if length else "none")
                        key = ("upstream_features", family, s_q, s_kv, topk, sink, length)
                        cases.append(_case("upstream_features", family, s_q, s_kv, topk,
                                           UPSTREAM_SM_SCALE, content, sanitize=key in picks))
    corner = dict(UPSTREAM_CONTENT, sink_mode="upstream", length_mode="upstream")
    for family in UPSTREAM_FAMILY_ORDER:
        for s_q, s_kv, topk in UPSTREAM_ALL_INVALID:
            key = ("upstream_all_invalid", family, s_q, s_kv, topk)
            cases.append(_case("upstream_all_invalid", family, s_q, s_kv, topk, UPSTREAM_SM_SCALE,
                               dict(corner, invalid_rows="upstream"), sanitize=key in picks))
    for family in UPSTREAM_FAMILY_ORDER:
        for s_kv, topk in UPSTREAM_EMPTY_BLOCKS:
            for s_q in UPSTREAM_EMPTY_BLOCKS_S_Q:
                key = ("upstream_empty_blocks", family, s_q, s_kv, topk)
                cases.append(_case("upstream_empty_blocks", family, s_q, s_kv, topk,
                                   UPSTREAM_SM_SCALE, corner, sanitize=key in picks))
    family, s_q, s_kv, topk = UPSTREAM_LARGE_S_Q
    cases.append(_case("upstream_large_s_q", family, s_q, s_kv, topk, UPSTREAM_SM_SCALE,
                       corner, sanitize=True, native=False))
    return cases


def scored_shape_cases(runtime) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    sanitize = PRIVATE_SEMANTICS_SANITIZE
    for family, s_kv in runtime.anchor_grid():
        d_qk, _h_q = runtime.FAMILY_GEOMETRY[family]
        cases.append(_case("scored_semantics", family, runtime.ANCHOR_S_Q, s_kv,
                           runtime.FAMILY_TOPK[family], d_qk ** -0.55, SCORED_SEMANTICS,
                           sanitize=(family, s_kv) in sanitize))
    for family, s_kv in runtime.anchor_grid():
        cases.append(_case("scored_precision", family, runtime.ANCHOR_S_Q, s_kv,
                           runtime.FAMILY_TOPK[family], runtime.ANCHOR_SM_SCALE,
                           dict(runtime.SCORED_CONTENT, bias="zero")))
    return cases


def odd_cases(runtime) -> list[dict[str, Any]]:
    scales = lambda d_qk: (0.5, d_qk ** -0.55, 1.0, 4.0)  # noqa: E731
    cases: list[dict[str, Any]] = []
    for fi, family in enumerate(runtime.FAMILIES):
        d_qk = runtime.FAMILY_GEOMETRY[family][0]
        for i, s_q in enumerate(PRIVATE_ODD_S_Q):
            topk = ODD_TOPK[(i * 5 + fi * 3) % len(ODD_TOPK)]
            kv_opts = (1, 7, 95, 129, 1153, 4097, 65537, 131071, topk - 1, topk + 1, 3 * topk + 5)
            s_kv = kv_opts[(i * 7 + fi) % len(kv_opts)]
            override = PRIVATE_ODD_SANITIZE.get((family, s_q))
            content = ODD_CONTENTS[ODD_CONTENT_NAMES[(i + fi) % 3]]
            cases.append(_case("odd_sizes", family, s_q, override or s_kv, topk,
                               scales(d_qk)[(i + 2 * fi) % 4], content,
                               sanitize=override is not None))
    return cases


CAUSAL_CONTENT: dict[str, Any] = dict(
    bias="upstream", index_mode="causal", invalid_rows="none", duplicates=False,
    scramble=False, sink_mode="none", length_mode="none",
)
# (s_q, s_kv): a one-row first chunk, a first chunk shorter than topk, a mid-context
# chunk, a first chunk longer than topk, and a short chunk deep in a long context.
PRIVATE_CAUSAL = ((1, 1), (63, 63), (777, 4097), (2049, 2049), (300, 131073))
PUBLIC_CAUSAL = ((59, 59), (511, 4099), (1500, 1500), (97, 131075))
PRIVATE_LARGE_KV_S_Q = 64
PUBLIC_LARGE_KV = ("v4_pro", 48, 1048573)
# (family, heads per rank): the TP2, TP4 and TP8 splits that keep a multiple of 16.
PRIVATE_TP_HEADS = (
    ("v32", 32), ("v32", 16), ("glm5", 32), ("glm5", 16),
    ("v4_flash", 32), ("v4_flash", 16), ("v4_pro", 64), ("v4_pro", 32), ("v4_pro", 16),
)
PUBLIC_TP_HEADS = (("v32", 48), ("v4_flash", 48), ("v4_pro", 80))
TP_HEADS_SHAPE = (213, 1153)
PUBLIC_TP_HEADS_SHAPE = (197, 1155)
PRIVATE_NEW_SANITIZE = {("causal", "glm5", 63), ("large_kv", "v4_flash", PRIVATE_LARGE_KV_S_Q),
                        ("tp_heads", "v4_pro", 16)}


def serving_cases(runtime, *, public: bool) -> list[dict[str, Any]]:
    families = runtime.FAMILIES
    cases: list[dict[str, Any]] = []
    causal = PUBLIC_CAUSAL if public else PRIVATE_CAUSAL
    for fi, family in enumerate(families):
        topk = runtime.FAMILY_TOPK[family]
        scale = runtime.FAMILY_SM_SCALE[family]
        shapes = (causal[fi],) if public else causal
        for i, (s_q, s_kv) in enumerate(shapes):
            content = dict(CAUSAL_CONTENT, sink_mode="finite" if (i + fi) % 2 else "none")
            sanitize = (public and fi == 1) or (not public and ("causal", family, s_q) in PRIVATE_NEW_SANITIZE)
            cases.append(_case("causal", family, s_q, s_kv, topk, scale, content, sanitize=sanitize))
    corner = dict(UPSTREAM_CONTENT, sink_mode="upstream", length_mode="upstream")
    large = [PUBLIC_LARGE_KV] if public else [
        (family, PRIVATE_LARGE_KV_S_Q, runtime.FAMILY_MAX_CONTEXT[family] - 1) for family in families
    ]
    for family, s_q, s_kv in large:
        cases.append(_case("large_kv", family, s_q, s_kv, runtime.FAMILY_TOPK[family], UPSTREAM_SM_SCALE,
                           corner, sanitize=not public and ("large_kv", family, s_q) in PRIVATE_NEW_SANITIZE))
    s_q, s_kv = PUBLIC_TP_HEADS_SHAPE if public else TP_HEADS_SHAPE
    for family, h_q in (PUBLIC_TP_HEADS if public else PRIVATE_TP_HEADS):
        d_qk = runtime.FAMILY_GEOMETRY[family][0]
        case = _case("tp_heads", family, s_q, s_kv, runtime.FAMILY_TOPK[family], d_qk ** -0.55,
                     SCORED_SEMANTICS, sanitize=not public and ("tp_heads", family, h_q) in PRIVATE_NEW_SANITIZE)
        case["h_q"] = h_q
        cases.append(case)
    return cases


def v1_fixture_cases() -> list[dict[str, Any]]:
    cases = []
    for fixture in V1_FIXTURES:
        content = {key: fixture[key] for key in (
            "index_mode", "invalid_rows", "scramble", "duplicates", "sink_mode", "length_mode"
        )}
        case = _case("v1_fixture", fixture["family"], fixture["s_q"], fixture["s_kv"],
                     fixture["topk"], fixture["sm_scale"], dict(content, bias="upstream"),
                     sanitize=fixture["name"] in V1_SANITIZE)
        case["_name"] = fixture["name"]
        cases.append(case)
    return cases


def public_cases(runtime) -> list[dict[str, Any]]:
    families = runtime.FAMILIES
    cases: list[dict[str, Any]] = []
    for i, (s_kv, topk) in enumerate(PUBLIC_PAIRS):
        for j, s_q in enumerate(PUBLIC_CORRECTNESS_S_Q):
            cases.append(_case("correctness", families[(i + j) % len(families)], s_q, s_kv, topk,
                               UPSTREAM_SM_SCALE, UPSTREAM_CONTENT,
                               sanitize=(i, j) == (7, 0)))
    combos = ((True, True), (True, False), (False, True), (False, False))
    for i, (s_kv, topk) in enumerate(PUBLIC_PAIRS[3:]):
        for j, s_q in enumerate(PUBLIC_FEATURES_S_Q):
            sink, length = combos[(i + 2 * j) % 4]
            content = dict(UPSTREAM_CONTENT, sink_mode="upstream" if sink else "none",
                           length_mode="upstream" if length else "none")
            cases.append(_case("features", families[(i + j) % len(families)], s_q, s_kv, topk,
                               UPSTREAM_SM_SCALE, content))
    corner = dict(UPSTREAM_CONTENT, sink_mode="upstream", length_mode="upstream")
    for i, (s_q, s_kv, topk) in enumerate(PUBLIC_ALL_INVALID):
        cases.append(_case("all_invalid", families[i % len(families)], s_q, s_kv, topk, UPSTREAM_SM_SCALE,
                           dict(corner, invalid_rows="upstream"), sanitize=i == 0))
    for i, (s_kv, topk) in enumerate(PUBLIC_EMPTY_BLOCKS):
        for j, s_q in enumerate(PUBLIC_EMPTY_BLOCKS_S_Q):
            cases.append(_case("empty_blocks", families[(i + j) % len(families)], s_q, s_kv, topk,
                               UPSTREAM_SM_SCALE, corner, sanitize=(i, j) == (1, 1)))
    for i, (family, s_q, topk, s_kv) in enumerate(PUBLIC_CELLS):
        d_qk = runtime.FAMILY_GEOMETRY[family][0]
        cases.append(_case("semantics", family, s_q, s_kv, topk, d_qk ** -0.55, SCORED_SEMANTICS,
                           sanitize=i == 2))
    for family, s_q, topk, s_kv in PUBLIC_CELLS[::2]:
        cases.append(_case("precision", family, s_q, s_kv, topk, UPSTREAM_SM_SCALE,
                           dict(UPSTREAM_CONTENT, bias="zero", sink_mode="upstream")))
    scales = lambda d_qk: (0.5, d_qk ** -0.55, 1.0, 4.0)  # noqa: E731
    for i, s_q in enumerate(PUBLIC_ODD_S_Q):
        family = families[i % len(families)]
        topk = ODD_TOPK[(i * 7 + 3) % len(ODD_TOPK)]
        kv_opts = (1, 9, 93, 131, 1155, 4099, 65539, 131069, topk - 3, topk + 3, 3 * topk + 7)
        override = PUBLIC_ODD_SANITIZE.get(s_q)
        s_kv = override or kv_opts[(i * 5 + 2) % len(kv_opts)]
        d_qk = runtime.FAMILY_GEOMETRY[family][0]
        cases.append(_case("odd_sizes", family, s_q, s_kv, topk, scales(d_qk)[(i + 1) % 4],
                           ODD_CONTENTS[ODD_CONTENT_NAMES[(i + 1) % 3]],
                           sanitize=override is not None))
    fixtures = {fixture["name"]: fixture for fixture in V1_FIXTURES}
    for name, s_q, s_kv in PUBLIC_FIXTURES:
        fixture = fixtures[name]
        content = {key: fixture[key] for key in (
            "index_mode", "invalid_rows", "scramble", "duplicates", "sink_mode", "length_mode"
        )}
        cases.append(_case("fixture", fixture["family"], s_q, s_kv, fixture["topk"],
                           fixture["sm_scale"], dict(content, bias="upstream"),
                           sanitize=name in V1_SANITIZE))
    return cases


def _public_names(case: dict[str, Any]) -> dict[str, Any]:
    """The Work runtime calls the Judge's ``upstream`` content mode ``default``."""
    return {key: "default" if value == "upstream" else value for key, value in case.items()}


def generate(suite_id: str, *, public: bool, seed: int | None = None) -> dict[str, Any]:
    runtime = _load_runtime()
    rng = random.Random(default_seed(suite_id) if seed is None else seed)
    prefix = "public-edge" if public else "edge"
    drafts = public_cases(runtime) if public else [
        *upstream_cases(),
        *scored_shape_cases(runtime),
        *odd_cases(runtime),
        *v1_fixture_cases(),
        *serving_cases(runtime, public=False),
    ]
    if public:
        drafts += serving_cases(runtime, public=True)
    expected = dict(PUBLIC_CATEGORY_COUNTS if public else CATEGORY_COUNTS)
    counts = Counter(case["category"] for case in drafts)
    if dict(counts) != expected:
        raise RuntimeError(f"category counts differ from the plan: {dict(counts)}")
    cases: list[dict[str, Any]] = []
    seeds: set[int] = set()
    for index, draft in enumerate(drafts, start=1):
        data_seed = rng.getrandbits(63)
        while data_seed in seeds:
            data_seed = rng.getrandbits(63)
        seeds.add(data_seed)
        name = draft.pop("_name", None)
        suffix = name if name else draft["category"].replace("_", "-")
        case = {"case_id": f"{prefix}-{index:03d}-{suffix}", **draft, "seed": data_seed}
        runtime.validate_edge_case(case)
        cases.append(_public_names(case) if public else case)
    sanitize = sum(case["sanitize"] for case in cases)
    wanted = PUBLIC_SANITIZE_COUNT if public else PRIVATE_SANITIZE_COUNT
    if sanitize != wanted:
        raise RuntimeError(f"{suite_id}: {sanitize} sanitizer cases, expected {wanted}")
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
    native = [case for case in cases if case["native"]]
    counts = Counter(case["category"] for case in cases)
    q_bytes = [
        runtime.tensor_bytes(case["s_q"], runtime.spec_geometry(case)[1], runtime.spec_geometry(case)[0])[0]
        for case in native
    ]
    lines = [
        f"{payload['suite_id']}: {len(cases)} cases, {len(native)} native, "
        f"{sum(case['sanitize'] for case in cases)} sanitize "
        f"({sum(not case['native'] for case in cases)} sanitizer-only); "
        f"{sum(case['s_q'] == 1 for case in cases)} with s_q = 1, "
        f"{sum(case['s_q'] % 8 != 0 for case in cases)} with s_q not divisible by 8",
        "  categories: " + ", ".join(f"{name}={count}" for name, count in counts.items()),
        f"  native q: total {sum(q_bytes) / 1e9:.3f} GB, max {max(q_bytes) / (1 << 20):.1f} MiB",
    ]
    for case in runtime.sanitize_order(cases):
        lines.append(
            f"  sanitize {case['case_id']}: {case['family']} s_q={case['s_q']} s_kv={case['s_kv']} "
            f"topk={case['topk']} native={case['native']}"
        )
    return lines


TARGETS: tuple[tuple[str, Path, bool], ...] = (
    (PRIVATE_SUITE_ID, PRIVATE_PATH, False),
    (PUBLIC_SUITE_ID, PUBLIC_PATH, True),
)


def _resolve(path: Path, public: bool) -> Path:
    """The public instance lives in the Work build context; the private one under
    <INFRATHON_ASSETS_ROOT>/<task directory>/judge/."""
    if public:
        return path
    assets_root = os.environ.get("INFRATHON_ASSETS_ROOT") or "~/.cache/infrathon/assets"
    return Path(assets_root).expanduser() / TASK_ROOT.name / path


def _committed(path: Path, public: bool) -> Path | None:
    """The instance --check compares; None when the Judge-only asset is absent."""
    resolved = _resolve(path, public)
    return resolved if public or resolved.is_file() else None


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
        source = _resolve(path, public) if args.write else _committed(path, public)
        if source is None:
            print(f"{suite_id}: {path} not present "
                  "(Judge-only dataset asset; run assets/fetch.sh), not compared")
            continue
        label = source.relative_to(TASK_ROOT) if public else source
        current = source.read_bytes() if source.exists() else b""
        if current != encoded:
            drift = True
            print(f"{label}: differs from the seeded generator")
            if args.write:
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_bytes(encoded)
                if not public:
                    source.chmod(0o600)
        digest = hashlib.sha256(source.read_bytes() if source.exists() else encoded).hexdigest()
        print(f"{suite_id}: {label} sha256={digest}")
        for line in summary(payload):
            print(line)
    if args.check and drift:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
