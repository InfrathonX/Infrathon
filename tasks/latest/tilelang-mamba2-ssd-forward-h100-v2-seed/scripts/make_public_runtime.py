#!/usr/bin/env python3
"""Derive the public Work runtime from the trusted runtime.

The Work copy keeps the operator, the generator, the reference, the comparison,
the timing protocol, the operator record and the score byte for byte, and swaps
in the public geometries, the public cases, the public edge fixtures and the
public captured layer.  It drops the Judge's device-memory floor and every word
that names the hidden suite.

usage: scripts/make_public_runtime.py [--check]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parents[1]
SOURCE = TASK_ROOT / "runtime/mamba2_ssd_runtime.py"
TARGET = TASK_ROOT / "images/work/benchmark/mamba2_ssd_runtime.py"

PUBLIC_MODEL = '''# --------------------------------------------------------------------------
# Workload model
# --------------------------------------------------------------------------
# A geometry is a Mamba-2 head layout (heads, head width, state width, groups).
# The public geometries are real deployed layouts or synthetic ones; none of
# them is a hidden-suite geometry.

GEOMETRIES: dict[str, dict[str, int]] = {
    "h48-p64-n256-g1": {"h": 48, "p": 64, "n": 256, "g": 1},
    "h96-p80-n128-g4": {"h": 96, "p": 80, "n": 128, "g": 4},
    "h40-p64-n64-g1": {"h": 40, "p": 64, "n": 64, "g": 1},
    "h256-p48-n128-g16": {"h": 256, "p": 48, "n": 128, "g": 16},
    "h48-p64-n128-g1": {"h": 48, "p": 64, "n": 128, "g": 1},
    "h64-p128-n256-g4": {"h": 64, "p": 128, "n": 256, "g": 4},
    "h24-p64-n128-g1": {"h": 24, "p": 64, "n": 128, "g": 1},
    "h80-p64-n64-g1": {"h": 80, "p": 64, "n": 64, "g": 1},
    "h24-p32-n64-g1": {"h": 24, "p": 32, "n": 64, "g": 1},
    "h48-p32-n128-g1": {"h": 48, "p": 32, "n": 128, "g": 1},
    # Correctness-only edge geometries.
    "E1": {"h": 6, "p": 16, "n": 16, "g": 1},
    "E2": {"h": 16, "p": 32, "n": 64, "g": 4},
    "E3": {"h": 12, "p": 48, "n": 32, "g": 3},
    "E4": {"h": 200, "p": 16, "n": 32, "g": 200},
    "E5": {"h": 7, "p": 256, "n": 64, "g": 1},
    "E6": {"h": 20, "p": 64, "n": 256, "g": 5},
}

# Workload families.  ``batch``: training and padded batches, batch >= 2.
# ``serving``: batch 1 at real request lengths.  ``long``: batch 1, long
# sequences.  ``real``: a layer captured from a real model on real text, timed
# once.
SCORE_FAMILIES: tuple[str, ...] = ("batch", "serving", "long", "real")
TIERS: tuple[str, ...] = ("recipe", "sweep", "real")
# Case count per family of the full public suite.
FINAL_FAMILY_COUNTS: dict[str, int] = {"batch": 6, "serving": 13, "long": 4, "real": 1}
FINAL_CASE_COUNT = sum(FINAL_FAMILY_COUNTS.values())
REAL_TRIALS = 1
MIN_LONG_LENGTH = 60000

# No tensor reaches 2**31 elements (the support domain says so).
MAX_TENSOR_ELEMENTS = 2**31 - 1
MAX_BATCH = 16

FAMILY_WEIGHTS: dict[str, float] = {family: 1.0 / len(SCORE_FAMILIES) for family in SCORE_FAMILIES}'''

PUBLIC_EDGES = '''# Correctness-only fixtures, never timed or scored: lengths shorter than one
# chunk and straddling chunk boundaries, ragged long tails, slow and fast decay,
# dt pinned to one, D or B removed, a full batch and one head per group.
EDGE_FIXTURES: tuple[dict[str, Any], ...] = (
    {"case_id": "public-edge-len1", "geometry": "E1", "batch": 1, "seqlen": 1, "seed": 0xA001},
    {"case_id": "public-edge-len2", "geometry": "E3", "batch": 2, "seqlen": 2, "seed": 0xA002},
    {"case_id": "public-edge-len31", "geometry": "E2", "batch": 1, "seqlen": 31, "seed": 0xA003},
    {"case_id": "public-edge-len127", "geometry": "E1", "batch": 2, "seqlen": 127, "seed": 0xA004},
    {"case_id": "public-edge-len128", "geometry": "E3", "batch": 1, "seqlen": 128, "seed": 0xA005},
    {"case_id": "public-edge-len191", "geometry": "E2", "batch": 2, "seqlen": 191, "seed": 0xA006},
    {"case_id": "public-edge-len255", "geometry": "E1", "batch": 1, "seqlen": 255, "seed": 0xA007},
    {"case_id": "public-edge-len385", "geometry": "E2", "batch": 1, "seqlen": 385, "seed": 0xA008},
    {"case_id": "public-edge-len769", "geometry": "E5", "batch": 1, "seqlen": 769, "seed": 0xA009},
    {"case_id": "public-edge-slow-decay", "geometry": "E3", "batch": 2, "seqlen": 1500, "regime": "slow_decay", "seed": 0xA00A},
    {"case_id": "public-edge-slow-decay-g5", "geometry": "E6", "batch": 1, "seqlen": 2100, "regime": "slow_decay", "seed": 0xA00B},
    {"case_id": "public-edge-slow-decay-g4", "geometry": "E2", "batch": 1, "seqlen": 4000, "regime": "slow_decay", "seed": 0xA00C},
    {"case_id": "public-edge-fast-decay", "geometry": "E5", "batch": 2, "seqlen": 1000, "regime": "fast_decay", "seed": 0xA00D},
    {"case_id": "public-edge-unit-dt", "geometry": "E1", "batch": 1, "seqlen": 700, "regime": "unit_dt", "seed": 0xA00E},
    {"case_id": "public-edge-no-residual", "geometry": "E3", "batch": 1, "seqlen": 950, "d_mode": "zero", "seed": 0xA00F},
    {"case_id": "public-edge-residual-only", "geometry": "E6", "batch": 1, "seqlen": 1200, "bc_mode": "zero", "seed": 0xA010},
    {"case_id": "public-edge-full-batch", "geometry": "E1", "batch": 16, "seqlen": 300, "seed": 0xA011},
    {"case_id": "public-edge-one-head-per-group", "geometry": "E4", "batch": 1, "seqlen": 333, "seed": 0xA012},
    {"case_id": "public-edge-long-tail-g3", "geometry": "E3", "batch": 1, "seqlen": 7777, "seed": 0xA013},
    {"case_id": "public-edge-long-tail-g4", "geometry": "E2", "batch": 1, "seqlen": 13001, "seed": 0xA014},
    {"case_id": "public-edge-long-tail-g5", "geometry": "E6", "batch": 1, "seqlen": 24601, "seed": 0xA015},
)


'''

PUBLIC_SUITE = '''

# --------------------------------------------------------------------------
# Public suite
# --------------------------------------------------------------------------
PUBLIC_SUITE_ID = "tilelang-ssd-public-v3"
PUBLIC_SEED_BASE = 0x0066_0301
PUBLIC_LAYER_ROOT = Path(__file__).resolve().parent / "public_layer"

# (name, family, geometry, batch, L).  Serving lengths are real request lengths
# from public serving traces.
PUBLIC_CASES: tuple[tuple[str, str, str, int, int], ...] = (
    ("P1", "batch", "h48-p64-n256-g1", 8, 1280),
    ("P2", "batch", "h96-p80-n128-g4", 4, 7000),
    ("P3", "serving", "h40-p64-n64-g1", 1, 24000),
    ("P4", "batch", "h256-p48-n128-g16", 2, 5000),
    ("P5", "batch", "h48-p64-n128-g1", 8, 6144),
    ("P6", "batch", "h64-p128-n256-g4", 16, 1393),
    ("P7", "batch", "h24-p64-n128-g1", 2, 8222),
    ("P8", "serving", "h48-p64-n128-g1", 1, 6166),
    ("P9", "serving", "h48-p64-n128-g1", 1, 12535),
    ("P10", "serving", "h48-p64-n256-g1", 1, 3170),
    ("P11", "serving", "h48-p64-n256-g1", 1, 10128),
    ("P12", "serving", "h80-p64-n64-g1", 1, 3175),
    ("P13", "serving", "h80-p64-n64-g1", 1, 1632),
    ("P14", "serving", "h24-p64-n128-g1", 1, 6242),
    ("P15", "serving", "h24-p64-n128-g1", 1, 10318),
    ("P16", "serving", "h64-p128-n256-g4", 1, 5424),
    ("P17", "serving", "h64-p128-n256-g4", 1, 15919),
    ("P18", "serving", "h96-p80-n128-g4", 1, 6346),
    ("P19", "serving", "h96-p80-n128-g4", 1, 16890),
    ("P20", "long", "h48-p64-n128-g1", 1, 65500),
    ("P21", "long", "h48-p64-n256-g1", 1, 131000),
    ("P22", "long", "h24-p32-n64-g1", 1, 240000),
    ("P23", "long", "h64-p128-n256-g4", 1, 100003),
    ("P24", "real", "h48-p32-n128-g1", 1, 8123),
)
PUBLIC_LAYER = "public-real-layer16"


def public_suite() -> list[dict[str, Any]]:
    """The public cases with disclosed seeds; every trial's first one is compared."""
    return [
        validate_case(
            {
                "case_id": f"public-{name.lower()}",
                "geometry": geometry,
                "family": family,
                "tier": "real" if family == "real" else "sweep",
                "batch": batch,
                "seqlen": seqlen,
                "regime": "mixed",
                "seed": PUBLIC_SEED_BASE + index,
                "check_trial": 0,
                "layer": PUBLIC_LAYER if family == "real" else None,
            },
            scored=True,
        )
        for index, (name, family, geometry, batch, seqlen) in enumerate(PUBLIC_CASES)
    ]
'''

BANNED = (
    "mamba2_2p7b", "codestral", "falcon_h1", "Falcon", "Codestral", "Mamba-2-2.7B", "scored_grid",
    "SCORED_", "tilelang-ssd-final", "eighteen", "73 GiB", "73 *", "decoy", "Decoy",
    "PyroMind", "HAMi", "terminal Judge", "deadline Judge", "same shapes", "Nemotron", "nemotron",
    "Granite", "granite", "Bamba", "Zamba", "zamba", "Judge-only",
)


def _replace_between(text: str, start: str, end: str, new: str, *, keep_end: bool = True) -> str:
    i = text.index(start)
    j = text.index(end, i)
    if not keep_end:
        j += len(end)
    return text[:i] + new + text[j:]


def build() -> str:
    text = SOURCE.read_text(encoding="utf-8")
    a = text.index('"""Trusted runtime')
    b = text.index('"""', a + 3) + 3
    text = text[:a] + '"""Public test runtime for the TileLang Mamba-2 SSD forward task, v3."""' + text[b:]
    text = _replace_between(
        text,
        "# --------------------------------------------------------------------------\n# Workload model",
        "FAMILY_WEIGHTS: dict[str, float] = {family: 1.0 / len(SCORE_FAMILIES) for family in SCORE_FAMILIES}",
        PUBLIC_MODEL,
        keep_end=False,
    )
    text = _replace_between(text, "# Correctness-only fixtures.", "def edge_fixtures()", PUBLIC_EDGES)
    # The Work GPU only has to meet the capability floor.
    text = re.sub(r"MIN_DEVICE_MEMORY_BYTES = .*\n", "MIN_DEVICE_MEMORY_BYTES = 0\n", text)
    text = text.replace('FINAL_SUITE_ID = "tilelang-ssd-final-v3"\n', "")
    text = text.replace(
        "# Captured layers\n# --------------------------------------------------------------------------\n"
        "# A real case reads one layer captured from a real model on real text\n"
        "# (fixtures/recipe/real_layers/).",
        "# Captured layers\n# --------------------------------------------------------------------------\n"
        "# A real case reads one layer captured from a real model on real text.",
    )
    text = re.sub(r"\bdecoy\b", "precall", text)
    text = text.replace("Decoy", "Pre-call").replace("``precall``", "``precall``")
    text = text.replace("the final suite", "the full public suite").replace("final suite", "full public suite")
    text = text.rstrip("\n") + "\n" + PUBLIC_SUITE
    for banned in BANNED:
        assert banned not in text, banned
    return text


def main(argv: list[str]) -> int:
    text = build()
    if "--check" in argv:
        if TARGET.read_text(encoding="utf-8") != text:
            print(f"{TARGET} is stale; run scripts/make_public_runtime.py")
            return 1
        return 0
    TARGET.write_text(text, encoding="utf-8")
    print(f"wrote {TARGET.relative_to(TASK_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
