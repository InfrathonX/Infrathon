#!/usr/bin/env python3
"""Generate the zero-weight edge-case manifests for standalone v4.

Edge cases are correctness and memory-safety gates over the whole declared
support domain (any ``0 <= M_g <= M_max``, ``1 <= G <= 64``, 128-aligned
``128 <= N, K <= 8192``, 128-aligned ``128 <= M_max <= 4096``, any positive
hint).  They are untimed and never enter the score.

One seeded generator produces two frozen instances with the same categories
and different draws:

* the Judge-only instance (``edge-holdout-v4``, 188 cases; an Infrathon asset,
  ``--private-path`` names where it lives), staged like the final holdout;
* ``images/work/benchmark/public_edge_cases.json`` (``edge-public-v4``,
  32 cases, shipped to Work for ``run_public_tests.sh --edge``).

A handful of small cases per instance carry ``"sanitize": true``; the Judge
also runs those under ``compute-sanitizer --tool memcheck``.

The classic categories follow the infrastructure review that motivated v3: sizes of
one, sizes that are not multiples of the usual tile widths (odd counts of
128-blocks), group counts that are not powers of two, empty and full groups,
tile-boundary row counts, hints that are far from the actual rows, and tight
capacities where any overrun leaves the tensor.

    scripts/generate_edge_manifest.py --check [--private-path P]   # regenerate and compare
    scripts/generate_edge_manifest.py --write --private-path P     # rewrite both instances

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
TASK_ID = "standalone_dynamic_grouped_fp8_h100_scratch_v3"
PRIVATE_SUITE_ID = "edge-holdout-v4"
PUBLIC_SUITE_ID = "edge-public-v4"
PRIVATE_PATH = TASK_ROOT / "judge/private/edge_cases.json"
PUBLIC_PATH = TASK_ROOT / "images/work/benchmark/public_edge_cases.json"
LAYOUTS = ("masked", "psum")

TINY_NK = (128, 256, 384)
ODD_BLOCK_NK = tuple(128 * blocks for blocks in (3, 5, 7, 9, 11, 13, 17, 23, 33, 57, 63))
COMMON_NK = (1024, 2048, 3072, 4096, 6144, 7168)
EDGE_GROUPS = (1, 2, 3, 5, 7, 31, 33, 63, 64)
TILE_EDGE_ROWS = (1, 63, 64, 65, 127, 128, 129, 255, 256, 257)
M_MAX_CHOICES = (128, 256, 384, 512, 1024, 2048, 4096)
# Public-only draws: these code paths stay visible without reusing private values.
PUBLIC_EDGE_GROUPS = (64, 20)
PUBLIC_ROW_PATTERN_M_MAX = (4096, 2048)
# Names visible in the public manifest.
PUBLIC_LABEL, PUBLIC_SANITIZE_LABEL = "contract", "sanitize"

# (category, private cases per layout, public cases per layout)
CATEGORY_COUNTS: tuple[tuple[str, int, int], ...] = (
    ("tiny_blocks", 8, 1),
    ("odd_n_blocks", 10, 1),
    ("odd_k_blocks", 10, 1),
    ("odd_both", 8, 1),
    ("max_nk", 6, 1),
    ("group_counts", 10, 1),
    ("row_patterns", 12, 1),
    ("hint_mismatch", 6, 1),
    ("tight_capacity", 6, 1),
    ("sanitize_small", 3, 2),
    # v4: the extended prefix-sum domain and the serving-shaped extremes.
    ("large_offset", 2, 1),
    ("single_group", 4, 1),
    ("hot_group", 3, 1),
    ("all_empty", 2, 1),
    ("hint_extreme", 3, 1),
    ("psum_large_m_max", 4, 1),
)
# Categories that only exist for the prefix-sum layout (its M_max goes past 4096).
PSUM_ONLY = frozenset({"large_offset", "psum_large_m_max"})
# Bound on the classic categories so the private suite stays fast; the v4
# categories are bounded only by the support domain.
EDGE_MAX_ROWS_TIMES_WIDTH = 1 << 28


def _load_runtime():
    path = TASK_ROOT / "runtime/grouped_fp8_runtime.py"
    spec = importlib.util.spec_from_file_location("edge_generator_runtime", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def default_seed(suite_id: str) -> int:
    digest = hashlib.sha256(f"{suite_id}|{TASK_ID}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _fits(g: int, m_max: int, n: int, k: int, runtime, *, classic: bool = True) -> bool:
    return (
        g * n * k <= runtime.EDGE_MAX_GROUP_NK
        and g * m_max * (k + 2 * n) <= runtime.MAX_CAPACITY_BYTES
        and (not classic or g * m_max * max(n, k) <= EDGE_MAX_ROWS_TIMES_WIDTH)
    )


def _row_pattern(rng: random.Random, pattern: str, g: int, m_max: int) -> list[int]:
    if pattern == "zeros_but_one":
        sizes = [0] * g
        sizes[rng.randrange(g)] = rng.randint(1, m_max)
        return sizes
    if pattern == "all_empty_but_last":
        return [0] * (g - 1) + [rng.randint(1, m_max)]
    if pattern == "ones":
        return [1] * g
    if pattern == "full":
        return [m_max] * g
    if pattern == "full_minus_one":
        return [m_max - 1] * g
    if pattern == "tile_edges":
        rows = [value for value in TILE_EDGE_ROWS if value <= m_max]
        offset = rng.randrange(len(rows))
        return [rows[(offset + index) % len(rows)] for index in range(g)]
    if pattern == "uniform":
        return [rng.randint(0, m_max) for _ in range(g)]
    if pattern == "upstream_tiny":
        expected = rng.choice((2, 3, 7, 9))
        return [int(expected * rng.uniform(0.7, 1.3)) for _ in range(g)]
    if pattern == "mixed_extremes":
        return [rng.choice((0, 1, m_max, m_max - 1, rng.randint(0, m_max))) for _ in range(g)]
    if pattern == "last_full":
        return [rng.randint(0, m_max) for _ in range(g - 1)] + [m_max]
    if pattern == "sparse_single_rows":
        return [1 if rng.random() < 0.5 else 0 for _ in range(g)]
    raise ValueError(pattern)


ROW_PATTERNS = (
    "zeros_but_one",
    "all_empty_but_last",
    "ones",
    "full",
    "full_minus_one",
    "tile_edges",
    "uniform",
    "upstream_tiny",
    "mixed_extremes",
    "last_full",
    "sparse_single_rows",
)


def _hint(rng: random.Random, sizes: list[int], m_max: int, mode: str) -> int:
    if mode == "low":
        return 1
    if mode == "high":
        return m_max
    if mode == "random":
        return rng.randint(1, 4096)
    mean = sum(sizes) / max(1, len(sizes))
    return max(1, int(round(1.2 * mean)))


def _shape(
    rng: random.Random,
    runtime,
    *,
    g_choices,
    m_max_choices,
    n_choices,
    k_choices,
    classic: bool = True,
) -> tuple[int, int, int, int]:
    for _ in range(10_000):
        g = rng.choice(g_choices)
        m_max = rng.choice(m_max_choices)
        n = rng.choice(n_choices)
        k = rng.choice(k_choices)
        if _fits(g, m_max, n, k, runtime, classic=classic):
            return g, m_max, n, k
    raise RuntimeError("could not draw a shape within the edge-suite size bound")


def _category_case(
    rng: random.Random, runtime, category: str, ordinal: int, *, public: bool = False,
    layout: str = "masked",
) -> tuple[int, int, int, int, list[int], int]:
    small_m_max = (128, 256, 384, 512)
    if category in V4_CATEGORIES:
        return _v4_case(rng, runtime, category, ordinal, public=public, layout=layout)
    if category == "tiny_blocks":
        g, m_max, n, k = _shape(
            rng, runtime, g_choices=(1, 2, 3, 4, 5, 8, 16), m_max_choices=small_m_max,
            n_choices=TINY_NK, k_choices=TINY_NK + (128,),
        )
        pattern = ROW_PATTERNS[ordinal % len(ROW_PATTERNS)]
    elif category == "odd_n_blocks":
        g, m_max, n, k = _shape(
            rng, runtime, g_choices=(2, 3, 4, 6, 8, 12), m_max_choices=small_m_max,
            n_choices=ODD_BLOCK_NK, k_choices=COMMON_NK,
        )
        pattern = rng.choice(("uniform", "last_full", "tile_edges", "mixed_extremes", "full"))
    elif category == "odd_k_blocks":
        g, m_max, n, k = _shape(
            rng, runtime, g_choices=(2, 3, 4, 6, 8, 12), m_max_choices=small_m_max,
            n_choices=COMMON_NK, k_choices=ODD_BLOCK_NK,
        )
        pattern = rng.choice(("uniform", "last_full", "tile_edges", "mixed_extremes", "full"))
    elif category == "odd_both":
        g, m_max, n, k = _shape(
            rng, runtime, g_choices=(1, 2, 3, 5, 7), m_max_choices=small_m_max,
            n_choices=ODD_BLOCK_NK, k_choices=ODD_BLOCK_NK,
        )
        pattern = rng.choice(ROW_PATTERNS)
    elif category == "max_nk":
        if ordinal % 2:
            n_choices, k_choices = (8192,), ODD_BLOCK_NK + TINY_NK
        else:
            n_choices, k_choices = ODD_BLOCK_NK + TINY_NK, (8192,)
        g, m_max, n, k = _shape(
            rng, runtime, g_choices=(1, 2, 3), m_max_choices=(128, 256, 512),
            n_choices=n_choices, k_choices=k_choices,
        )
        pattern = rng.choice(("uniform", "last_full", "full"))
    elif category == "group_counts":
        groups = PUBLIC_EDGE_GROUPS if public else EDGE_GROUPS
        g_value = groups[ordinal % len(groups)]
        g, m_max, n, k = _shape(
            rng, runtime, g_choices=(g_value,), m_max_choices=(128, 256),
            n_choices=TINY_NK + ODD_BLOCK_NK[:6], k_choices=TINY_NK + ODD_BLOCK_NK[:6],
        )
        pattern = rng.choice(("uniform", "upstream_tiny", "sparse_single_rows", "mixed_extremes"))
    elif category == "row_patterns":
        pattern = ROW_PATTERNS[ordinal % len(ROW_PATTERNS)]
        if public:
            m_max_choices = (PUBLIC_ROW_PATTERN_M_MAX[ordinal % len(PUBLIC_ROW_PATTERN_M_MAX)],)
        else:
            m_max_choices = (4096,) if ordinal % 6 == 5 else M_MAX_CHOICES[:5]
        g, m_max, n, k = _shape(
            rng, runtime, g_choices=(3, 4, 5, 7, 9), m_max_choices=m_max_choices,
            n_choices=TINY_NK + ODD_BLOCK_NK[:5], k_choices=TINY_NK + ODD_BLOCK_NK[:5],
        )
    elif category == "hint_mismatch":
        g, m_max, n, k = _shape(
            rng, runtime, g_choices=(2, 4, 6, 8), m_max_choices=(512, 1024, 2048),
            n_choices=TINY_NK + ODD_BLOCK_NK[:4], k_choices=TINY_NK + ODD_BLOCK_NK[:4],
        )
        pattern = "uniform" if ordinal % 2 else "full_minus_one"
    elif category == "tight_capacity":
        g, m_max, n, k = _shape(
            rng, runtime, g_choices=(1, 1, 2, 3), m_max_choices=(128, 256, 384),
            n_choices=ODD_BLOCK_NK[:6], k_choices=ODD_BLOCK_NK[:6] + TINY_NK,
        )
        pattern = "full" if ordinal % 2 else "last_full"
    elif category == "sanitize_small":
        g, m_max, n, k = _shape(
            rng, runtime, g_choices=(1, 2, 3), m_max_choices=(128, 256),
            n_choices=(384, 640, 1152), k_choices=(128, 384, 640),
        )
        pattern = ("last_full", "tile_edges", "mixed_extremes")[ordinal % 3]
    else:
        raise ValueError(category)
    sizes = _row_pattern(rng, pattern, g, m_max)
    if category == "hint_mismatch":
        hint_mode = "low" if ordinal % 2 else "high"
    else:
        hint_mode = rng.choice(("normal", "normal", "random", "low", "high"))
    return g, m_max, n, k, sizes, _hint(rng, sizes, m_max, hint_mode)


V4_CATEGORIES = frozenset(
    {"large_offset", "single_group", "hot_group", "all_empty", "hint_extreme", "psum_large_m_max"}
)


def _v4_case(
    rng: random.Random, runtime, category: str, ordinal: int, *, public: bool, layout: str
) -> tuple[int, int, int, int, list[int], int]:
    masked_m_max = (128, 256, 1024, 4096)
    if category == "large_offset":
        # Rows past the 2^31-byte offset of A: 64-bit addressing is required.
        g, m_max = 8, 65536
        n = 384 if public else (128, 256)[ordinal % 2]
        k = 7168 if public else (7168, 8192)[ordinal % 2]
        sizes = [rng.randint(m_max - 4096, m_max) for _ in range(g)]
        sizes[-1] = m_max - rng.randint(0, 127)
        return g, m_max, n, k, sizes, _hint(rng, sizes, m_max, "normal")
    if category == "psum_large_m_max":
        g, m_max, n, k = _shape(
            rng, runtime, g_choices=(2, 3, 4, 5), m_max_choices=((8192, 16384, 32768, 131072)[ordinal % 4],),
            n_choices=TINY_NK + ODD_BLOCK_NK[:3], k_choices=TINY_NK + ODD_BLOCK_NK[:3], classic=False,
        )
        pattern = ("uniform", "last_full", "mixed_extremes", "zeros_but_one")[ordinal % 4]
        sizes = _row_pattern(rng, pattern, g, m_max)
        return g, m_max, n, k, sizes, _hint(rng, sizes, m_max, rng.choice(("normal", "low", "high")))
    large = (8192, 16384) if layout == "psum" else masked_m_max
    if category == "single_group":
        g, m_max, n, k = _shape(
            rng, runtime, g_choices=(1,), m_max_choices=large,
            n_choices=TINY_NK + ODD_BLOCK_NK[:6], k_choices=TINY_NK + ODD_BLOCK_NK[:6], classic=False,
        )
        sizes = [(1, m_max, m_max - 1, rng.randint(1, m_max))[ordinal % 4]]
        return g, m_max, n, k, sizes, _hint(rng, sizes, m_max, rng.choice(("normal", "low", "high")))
    if category == "hot_group":
        g, m_max, n, k = _shape(
            rng, runtime, g_choices=(8, 16, 32), m_max_choices=large,
            n_choices=TINY_NK + ODD_BLOCK_NK[:3], k_choices=TINY_NK + ODD_BLOCK_NK[:3], classic=False,
        )
        sizes = [rng.randint(0, 3) for _ in range(g)]
        sizes[rng.randrange(g)] = m_max - rng.randint(0, 64)
        return g, m_max, n, k, sizes, _hint(rng, sizes, m_max, "normal")
    if category == "all_empty":
        g, m_max, n, k = _shape(
            rng, runtime, g_choices=(1, 7, 32, 64), m_max_choices=(128, 1024),
            n_choices=TINY_NK + ODD_BLOCK_NK[:3], k_choices=TINY_NK + ODD_BLOCK_NK[:3],
        )
        return g, m_max, n, k, [0] * g, rng.randint(1, 512)
    if category == "hint_extreme":
        g, m_max, n, k = _shape(
            rng, runtime, g_choices=(2, 4, 8), m_max_choices=large,
            n_choices=TINY_NK + ODD_BLOCK_NK[:3], k_choices=TINY_NK + ODD_BLOCK_NK[:3], classic=False,
        )
        if ordinal % 2:
            sizes = [rng.randint(m_max // 2, m_max) for _ in range(g)]
            return g, m_max, n, k, sizes, 1
        sizes = [rng.randint(0, 4) for _ in range(g)]
        return g, m_max, n, k, sizes, runtime.MAX_HINT
    raise ValueError(category)


def generate(suite_id: str, *, public: bool, seed: int | None = None) -> dict[str, Any]:
    runtime = _load_runtime()
    rng = random.Random(default_seed(suite_id) if seed is None else seed)
    prefix = "public-edge" if public else "edge"
    private_shapes = (
        {(case["g"], case["n"], case["k"]) for case in generate(PRIVATE_SUITE_ID, public=False)["cases"]}
        if public
        else set()
    )
    cases: list[dict[str, Any]] = []
    seeds: set[int] = set()
    for layout in LAYOUTS:
        index = 0
        for category, private_count, public_count in CATEGORY_COUNTS:
            if layout == "masked" and category in PSUM_ONLY:
                continue
            for ordinal in range(public_count if public else private_count):
                # One public case per layout: the layouts take different variants.
                variant = ordinal + (1 if public and layout == "psum" and public_count == 1 else 0)
                g, m_max, n, k, sizes, hint = _category_case(
                    rng, runtime, category, variant, public=public, layout=layout
                )
                while (g, n, k) in private_shapes:
                    g, m_max, n, k, sizes, hint = _category_case(
                        rng, runtime, category, variant, public=public, layout=layout
                    )
                data_seed = rng.randrange(10_000, 2**31 - 1)
                while data_seed in seeds:
                    data_seed = rng.randrange(10_000, 2**31 - 1)
                seeds.add(data_seed)
                index += 1
                label = category
                if public:
                    label = PUBLIC_SANITIZE_LABEL if category == "sanitize_small" else PUBLIC_LABEL
                case = {
                    "case_id": f"{prefix}-{layout}-{index:03d}-{label}",
                    "layout": layout,
                    "category": label,
                    "g": g,
                    "m_max": m_max,
                    "n": n,
                    "k": k,
                    "hint": hint,
                    "m_sizes": sizes,
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


def _label(path: Path) -> str:
    try:
        return str(path.relative_to(TASK_ROOT))
    except ValueError:
        return str(path)


def _committed(path: Path, public: bool) -> Path:
    """The instance --check compares: the repository copy, else (a tree without it, such as
    the public tree) the Judge-only instance assets/fetch.sh places under
    <INFRATHON_ASSETS_ROOT>/<task directory>/judge/."""
    if public or path.exists() or path != PRIVATE_PATH:
        return path
    assets_root = os.environ.get("INFRATHON_ASSETS_ROOT") or "~/.cache/infrathon/assets"
    fetched = Path(assets_root).expanduser() / TASK_ROOT.name / "judge" / path.name
    return fetched if fetched.is_file() else path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    parser.add_argument(
        "--private-path",
        type=Path,
        default=PRIVATE_PATH,
        help="where the Judge-only instance lives (default: judge/private/edge_cases.json)",
    )
    args = parser.parse_args(argv)
    drift = False
    targets = ((PRIVATE_SUITE_ID, args.private_path.absolute(), False), (PUBLIC_SUITE_ID, PUBLIC_PATH, True))
    for suite_id, path, public in targets:
        encoded = encode(generate(suite_id, public=public))
        source = path if args.write else _committed(path, public)
        current = source.read_bytes() if source.exists() else b""
        if current != encoded:
            drift = True
            print(f"{_label(path)}: differs from the seeded generator")
            if args.write:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(encoded)
                if not public:
                    path.chmod(0o600)
        digest = hashlib.sha256(source.read_bytes() if source.exists() else encoded).hexdigest()
        print(f"{suite_id}: {_label(path)} sha256={digest}")
    if args.check and drift:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
