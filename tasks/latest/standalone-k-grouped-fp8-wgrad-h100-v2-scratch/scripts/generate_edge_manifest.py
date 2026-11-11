#!/usr/bin/env python3
"""Generate the zero-weight edge-case manifests for standalone K-grouped wgrad v3.

Edge cases are correctness and memory-safety gates over the whole declared
support domain (any ``1 <= M, N <= 8192``, ``1 <= G <= 128``, every group K
either 0 or a positive multiple of 128 with ``K_total <= 262144``, any positive
hint).  They are untimed and never enter the score.

One seeded generator produces two frozen instances with the same categories
and different draws:

* ``judge/private/edge_cases.json`` (``edge-holdout-v3``, 166 cases,
  Judge-only, staged like the final holdout);
* ``images/work/benchmark/public_edge_cases.json`` (``edge-public-v3``,
  43 cases, shipped to Work for ``run_public_tests.sh --edge``).

The sanitizer subset (8 private, 2 public cases, ``"sanitize": true``) is
also run under ``compute-sanitizer --tool memcheck`` by the Judge.

The categories follow the infrastructure review that motivated v2: sizes of
one and other tiny sizes (3, 7, 9, ...), sizes that are not multiples of 8,
16 or 128, multiples of 128 with an odd number of 128-blocks, dimension
limits, group counts that are not powers of two, empty groups and an empty
problem, unbalanced and odd-block K, hints far from the actual K, and tails
on the last group where any unmasked overrun leaves the tensor.  v3 adds
``deep_k``: one to four groups with K_total up to the v3 bound of 262144, where
any 32-bit product of a K offset and a row stride reaches 2^31.  The
``v1_contract`` category stays inside the v1 support domain, so a failure
there is a bug v1 should already have rejected.

    scripts/generate_edge_manifest.py --check     # regenerate and compare
    scripts/generate_edge_manifest.py --write     # rewrite both instances

``--write`` mints new suite identities when the output changes; the
``frozen_instance`` records in ``portable/*.json`` must then be refreshed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

TASK_ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "standalone_k_grouped_fp8_wgrad_h100_scratch_v2"
PRIVATE_SUITE_ID = "edge-holdout-v3"
PUBLIC_SUITE_ID = "edge-public-v3"
PRIVATE_PATH = TASK_ROOT / "judge/private/edge_cases.json"
PUBLIC_PATH = TASK_ROOT / "images/work/benchmark/public_edge_cases.json"

TINY_MN = (1, 2, 3, 5, 7, 9, 15, 17, 31, 33)
ODD_MID_MN = (127, 129, 255, 257, 1000, 1023, 1025, 2880, 4095, 7167, 8191)
TILE_EDGE_MN = (63, 64, 65, 127, 128, 129, 255, 256, 257)
SMALL_MN = (1, 3, 7, 9, 64, 100, 128, 129, 256, 257, 384, 640)
ODD_BLOCK_M = tuple(128 * blocks for blocks in (5, 7, 9, 11, 13, 15, 17, 63)) + (768, 1280, 1792)
ODD_BLOCK_N = tuple(128 * blocks for blocks in (9, 11, 13, 15, 17, 19, 23, 63)) + (1280, 1792)
EDGE_GROUPS = (1, 2, 3, 5, 7, 31, 33, 63, 64, 127, 128)
V1_EXPECTED_K = (128, 256, 384, 512, 640, 1024, 1536, 2048)

# (category, private cases, public cases); private sums to 166, public to 43.
CATEGORY_COUNTS: tuple[tuple[str, int, int], ...] = (
    ("tiny_mn", 20, 5),
    ("odd_mid_mn", 24, 6),
    ("v1_contract", 30, 4),
    ("dim_limits", 10, 3),
    ("group_counts", 16, 4),
    ("k_patterns", 24, 6),
    ("hint_mismatch", 10, 3),
    ("tail_overflow", 10, 3),
    ("tile_boundaries", 8, 2),
    ("sanitize_small", 8, 4),
    ("deep_k", 6, 3),
)
# deep_k: K_total in the v3 range above the v2 bound of 49152.
DEEP_K_TOTALS = (262_144, 196_608, 131_072, 98_304, 65_536 + 128 * 7, 229_376)
PUBLIC_DEEP_K_TOTALS = (245_760, 163_840, 81_920)

# Deterministic sanitizer subset: small enough to run under memcheck quickly,
# chosen so every overrun class reaches past an allocation.  (g, m, n, ks, hint)
SANITIZE_CASES: tuple[tuple[int, int, int, list[int], int], ...] = (
    (3, 7, 9, [128, 0, 384], 154),
    (1, 1, 1, [256], 1),
    (2, 129, 257, [384, 128], 307),
    (2, 640, 1152, [256, 384], 384),
    (2, 33, 65, [0, 0], 128),
    (1, 3, 100, [1536], 1843),
    (1, 3, 8191, [256], 1 << 20),
    (4, 65, 127, [128, 0, 640, 128], 1),
)
# The public instance gets its own small sanitizer cases (same overrun classes).
PUBLIC_SANITIZE_CASES: tuple[tuple[int, int, int, list[int], int], ...] = (
    (2, 5, 17, [256, 0], 1),
    (1, 9, 129, [384], 460),
    (48, 17, 33, [128] * 48, 154),
    (2, 1, 8190, [128, 128], 1 << 20),
)
# Public-only draws: these code paths stay visible without reusing private values.
PUBLIC_EDGE_GROUPS = (40, 72, 96, 3)
PUBLIC_K_PATTERNS = ("with_empty", "many_empty", "first_empty", "all_empty", "deep_single", "unbalanced")
PUBLIC_DEEP_K = (40_960, 24_576, 128 * 193)
PUBLIC_UNBALANCED_K = 8_960
# Names visible in the public manifest.
PUBLIC_CATEGORY_NAMES = {"v1_contract": "basic_shapes"}


def _load_runtime():
    path = TASK_ROOT / "runtime/k_grouped_wgrad_runtime.py"
    spec = importlib.util.spec_from_file_location("edge_generator_runtime", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def default_seed(suite_id: str) -> int:
    digest = hashlib.sha256(f"{suite_id}|{TASK_ID}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _align(value: int) -> int:
    return (value + 127) // 128 * 128


def _fits(g: int, m: int, n: int, ks: list[int], runtime) -> bool:
    return (
        len(ks) == g
        and g * m * n <= runtime.EDGE_MAX_OUTPUT_ELEMENTS
        and sum(ks) * (m + n) <= runtime.EDGE_MAX_INPUT_BYTES
        and sum(ks) <= runtime.MAX_TOTAL_K
    )


def _upstream_ks(rng: random.Random, g: int, expected: int) -> list[int]:
    return [_align(max(1, int(expected * rng.uniform(0.7, 1.3)))) for _ in range(g)]


def _ks(rng: random.Random, pattern: str, g: int, scale: int = 4) -> list[int]:
    """Group sizes in 128-row blocks times 128; ``scale`` caps the typical block count."""
    if pattern == "small":
        return [128 * rng.randint(1, scale) for _ in range(g)]
    if pattern == "with_empty":
        ks = [128 * rng.randint(1, scale) for _ in range(g)]
        if g > 1:
            ks[rng.randrange(g)] = 0
        return ks
    if pattern == "many_empty":
        return [0 if rng.random() < 0.6 else 128 * rng.randint(1, scale) for _ in range(g)]
    if pattern == "first_empty":
        return [0] + [128 * rng.randint(1, scale) for _ in range(g - 1)]
    if pattern == "last_empty":
        return [128 * rng.randint(1, scale) for _ in range(g - 1)] + [0]
    if pattern == "all_empty":
        return [0] * g
    if pattern == "odd_blocks":
        return [128 * rng.choice((1, 3, 5, 7, 9, 11)) for _ in range(g)]
    if pattern == "single_block":
        return [128] * g
    raise ValueError(pattern)


def _draw(rng: random.Random, runtime, build) -> tuple[int, int, int, list[int], int]:
    for _ in range(10_000):
        g, m, n, ks, hint = build()
        if _fits(g, m, n, ks, runtime):
            return g, m, n, ks, hint
    raise RuntimeError("could not draw a case within the edge-suite size bound")


def _hint(rng: random.Random, ks: list[int]) -> int:
    nonzero = [value for value in ks if value]
    mean = sum(nonzero) / len(nonzero) if nonzero else 128
    mode = rng.choice(("normal", "normal", "random", "low", "high"))
    if mode == "low":
        return 1
    if mode == "high":
        return 1 << 20
    if mode == "random":
        return rng.randint(1, 1 << 16)
    return max(1, int(1.2 * mean))


def _category_case(
    rng: random.Random, runtime, category: str, ordinal: int, *, public: bool = False
) -> tuple[int, int, int, list[int], int]:
    if category == "tiny_mn":
        def build():
            g = rng.choice((1, 2, 3, 4, 5, 8, 16))
            m = rng.choice(TINY_MN)
            n = rng.choice(TINY_MN + (64, 100, 129, 1000))
            if ordinal % 2:
                m, n = n, m
            ks = _ks(rng, rng.choice(("small", "with_empty", "odd_blocks")), g, scale=8)
            return g, m, n, ks, _hint(rng, ks)
    elif category == "odd_mid_mn":
        def build():
            g = rng.choice((1, 2, 3, 4, 6, 8))
            m = rng.choice(ODD_MID_MN)
            n = rng.choice(ODD_MID_MN + SMALL_MN)
            if ordinal % 2:
                m, n = n, m
            ks = _ks(rng, rng.choice(("small", "with_empty", "odd_blocks")), g, scale=6)
            return g, m, n, ks, _hint(rng, ks)
    elif category == "v1_contract":
        def build():
            expected = rng.choice(V1_EXPECTED_K)
            g = rng.randint(2, 32)
            if g * _align(int(expected * 1.3)) > 49_152:  # the v1 bound
                g = 2
            m = rng.choice(ODD_BLOCK_M)
            n = rng.choice(ODD_BLOCK_N)
            return g, m, n, _upstream_ks(rng, g, expected), int(expected * 1.2)
    elif category == "dim_limits":
        def build():
            g = rng.choice((1, 1, 2, 3))
            big = rng.choice((8192, 8191))
            small = rng.choice((1, 3, 7, 100, 129, 640))
            m, n = (big, small) if ordinal % 2 else (small, big)
            ks = _ks(rng, rng.choice(("small", "odd_blocks", "with_empty")), g, scale=8)
            return g, m, n, ks, _hint(rng, ks)
    elif category == "group_counts":
        groups = PUBLIC_EDGE_GROUPS if public else EDGE_GROUPS
        g_value = groups[ordinal % len(groups)]

        def build():
            m = rng.choice(SMALL_MN + TINY_MN)
            n = rng.choice(SMALL_MN + TINY_MN)
            ks = _ks(rng, rng.choice(("small", "with_empty", "many_empty", "single_block")), g_value, scale=3)
            return g_value, m, n, ks, _hint(rng, ks)
    elif category == "k_patterns":
        pattern = PUBLIC_K_PATTERNS[ordinal] if public else (
            "with_empty", "many_empty", "first_empty", "last_empty", "all_empty",
            "deep_single", "unbalanced", "odd_blocks",
        )[ordinal % 8]

        def build():
            if pattern == "deep_single":
                k = rng.choice(PUBLIC_DEEP_K if public else (49_152, 32_768, 16_384, 128 * 257))
                m = rng.choice((1, 3, 7, 64, 129, 1000, 2880))
                n = rng.choice((1, 9, 100, 257, 1025))
                return 1, m, n, [k], _hint(rng, [k])
            g = rng.choice((2, 3, 4, 5, 7, 9))
            m = rng.choice(SMALL_MN + ODD_MID_MN[:6])
            n = rng.choice(SMALL_MN + ODD_MID_MN[:6])
            if pattern == "unbalanced":
                ks = [128] * g
                ks[rng.randrange(g)] = PUBLIC_UNBALANCED_K if public else 10_752
            else:
                ks = _ks(rng, pattern, g, scale=8)
            return g, m, n, ks, _hint(rng, ks)
    elif category == "hint_mismatch":
        def build():
            g = rng.choice((2, 3, 4, 6, 8))
            m = rng.choice(SMALL_MN + ODD_MID_MN[:5])
            n = rng.choice(SMALL_MN + ODD_MID_MN[:5])
            if ordinal % 2:
                ks = [128 * rng.randint(8, 40) for _ in range(g)]
                return g, m, n, ks, 1
            ks = [128 * rng.randint(1, 2) for _ in range(g)]
            return g, m, n, ks, 1 << 20
    elif category == "tail_overflow":
        def build():
            g = 1 if ordinal % 2 else rng.choice((2, 3))
            m = rng.choice((65, 129, 193, 257, 513, 1025, 2049, 4097))
            n = rng.choice((65, 129, 257, 513, 1025, 2049, 4097, 8191))
            ks = _ks(rng, rng.choice(("small", "last_empty", "odd_blocks")), g, scale=6)
            if sum(ks) == 0:
                ks[0] = 128
            return g, m, n, ks, _hint(rng, ks)
    elif category == "deep_k":
        total = (PUBLIC_DEEP_K_TOTALS if public else DEEP_K_TOTALS)[ordinal]

        def build():
            g = rng.choice((1, 2, 3, 4))
            blocks = total // 128
            cuts = sorted(rng.sample(range(1, blocks), g - 1)) if g > 1 else []
            edges = [0, *cuts, blocks]
            ks = [128 * (edges[i + 1] - edges[i]) for i in range(g)]
            if g > 2 and ordinal % 2:
                ks[rng.randrange(g)] = 0
                ks[0] += total - sum(ks)
            limit = max(2, (1 << 28) // total)
            m = rng.choice([value for value in (1, 3, 65, 129, 257, 383, 640, 1000) if value < limit])
            n = rng.choice([value for value in (1, 7, 100, 129, 255, 384) if value + m <= limit])
            return g, m, n, ks, _hint(rng, ks)
    elif category == "tile_boundaries":
        def build():
            g = rng.choice((1, 2, 3, 4))
            # Deterministic sweep: M and N together cover every boundary value.
            m = TILE_EDGE_MN[ordinal % len(TILE_EDGE_MN)]
            n = TILE_EDGE_MN[(len(TILE_EDGE_MN) - 1 - ordinal) % len(TILE_EDGE_MN)]
            ks = _ks(rng, rng.choice(("small", "odd_blocks")), g, scale=6)
            return g, m, n, ks, _hint(rng, ks)
    else:
        raise ValueError(category)
    return _draw(rng, runtime, build)


def generate(suite_id: str, *, public: bool, seed: int | None = None) -> dict[str, Any]:
    runtime = _load_runtime()
    rng = random.Random(default_seed(suite_id) if seed is None else seed)
    prefix = "public-edge" if public else "edge"
    private_shapes = (
        {(case["g"], case["m"], case["n"]) for case in generate(PRIVATE_SUITE_ID, public=False)["cases"]}
        if public
        else set()
    )
    cases: list[dict[str, Any]] = []
    seeds: set[int] = set()
    index = 0
    for category, private_count, public_count in CATEGORY_COUNTS:
        for ordinal in range(public_count if public else private_count):
            if category == "sanitize_small":
                table = PUBLIC_SANITIZE_CASES if public else SANITIZE_CASES
                g, m, n, ks, hint = table[ordinal]
                ks = list(ks)
            else:
                g, m, n, ks, hint = _category_case(rng, runtime, category, ordinal, public=public)
                while (g, m, n) in private_shapes:
                    g, m, n, ks, hint = _category_case(rng, runtime, category, ordinal, public=public)
            data_seed = rng.getrandbits(63)
            while data_seed in seeds:
                data_seed = rng.getrandbits(63)
            seeds.add(data_seed)
            index += 1
            label = PUBLIC_CATEGORY_NAMES.get(category, category) if public else category
            case = {
                "case_id": f"{prefix}-{index:03d}-{label.replace('_', '-')}",
                "category": label,
                "g": g,
                "m": m,
                "n": n,
                "ks": ks,
                "hint": hint,
                "seed": data_seed,
                "sanitize": category == "sanitize_small",
            }
            runtime.validate_edge_case(case)
            cases.append(case)
    return {"schema_version": 4, "suite_id": suite_id, "cases": cases}


def encode(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


TARGETS: tuple[tuple[str, Path, bool], ...] = (
    (PRIVATE_SUITE_ID, PRIVATE_PATH, False),
    (PUBLIC_SUITE_ID, PUBLIC_PATH, True),
)


def _committed(path: Path, public: bool) -> Path:
    """The instance --check compares: the repository copy, else (a tree without it, such as
    the public tree) the Judge-only instance assets/fetch.sh places under
    <INFRATHON_ASSETS_ROOT>/<task directory>/judge/."""
    if public or path.exists():
        return path
    assets_root = os.environ.get("INFRATHON_ASSETS_ROOT") or "~/.cache/infrathon/assets"
    fetched = Path(assets_root).expanduser() / TASK_ROOT.name / "judge" / path.name
    return fetched if fetched.is_file() else path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    drift = False
    for suite_id, path, public in TARGETS:
        encoded = encode(generate(suite_id, public=public))
        source = path if args.write else _committed(path, public)
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
    if args.check and drift:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
