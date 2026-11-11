#!/usr/bin/env python3
"""Draw the mode-0600 Judge-only manifest of the scored cases.

Two domains.  Video cases are full multi-head attention built from token lattices
(text prefix + tokens per latent frame x latent frames) and sequence-parallel head splits of
Wan 2.1, LTX-2, Kandinsky 5, CogVideoX and HunyuanVideo.  LLM cases are causal grouped-query
attention built from Qwen3, Qwen3.5/3.6 and Llama 3.1 attention configurations and their
tensor-parallel head splits, at pre-training, long-context and fine-tuning lengths.
`recipe` cases are geometries a paper or an upstream training recipe actually uses; `sweep`
cases extrapolate along the same rules; `real` cases are captured layers
(fixtures/recipe/real_layers/).  What a redraw changes is the seeds, the fine-tuning lengths
and batch sizes, and which half of each family's sweep cases runs the golden.

The holdout is frozen, not drawn per build.  This generator mints a **new** instance from a
seeded ``random.Random``; the default seed is the first eight bytes of
``sha256("<suite_id>|<task_id>")`` read big-endian.  ``--redraw`` is the required
acknowledgement that the draw is a new suite identity whose ``frozen_instance`` record has to
be updated before any score is compared against it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

SUITE_ID = "final-holdout-v3"
TASK_ID = "attn_qat_nvfp4_training_attention_h100_seed_v1"
FROZEN_INSTANCE = "judge/private/final_cases.json"
CASE_KEYS = {
    "case_id", "family", "tier", "b", "h", "h_kv", "d", "s_q", "s_kv", "causal", "seed",
    "gated", "layer",
}
EXPECTED_CASE_COUNT = 287
WAN_480P, WAN_720P, LTX2, KANDINSKY5 = 1560, 3600, 390, 1536
COGVIDEOX, COGVIDEOX_TEXT = 1350, 226
HUNYUAN, HUNYUAN_TEXT = 2040, 256
KANDINSKY5_TEXT_TOKENS = 512
# Must match the public tests (runtime PUBLIC_LENGTHS); fine-tuning draws avoid them.
PUBLIC_LENGTHS = frozenset(
    {2728, 2784, 3584, 6120, 7168, 9216, 9696, 9984, 12384, 14336, 18016, 20480, 26208, 27280,
     28672, 40960, 57440}
)

# (family, d, heads, text prefix, lattice, latent frames, s_kv or None for self attention)
VIDEO_GRID = (
    ("wan13b", 128, (12, 6, 3), 0, WAN_480P, (5, 9, 13, 17, 20, 21), None),
    ("wan14b", 128, (40, 20, 10), 0, WAN_480P, (5, 9, 13, 17, 21), None),
    ("wan14b", 128, (20, 10), 0, WAN_720P, (9, 13, 17, 21), None),
    ("ltx2", 128, (32, 16), 0, LTX2, (3, 5, 7, 9, 11, 13), None),
    ("kandinsky5", 64, (32, 8), 0, KANDINSKY5, (7, 11, 16, 21, 26, 31), None),
    ("kandinsky5", 64, (32, 8), 0, KANDINSKY5, (11, 21, 31), KANDINSKY5_TEXT_TOKENS),
    ("cogvideox", 64, (48, 30, 15, 12), COGVIDEOX_TEXT, COGVIDEOX, (5, 9, 13), None),
    ("cogvideox", 64, (48, 30, 15, 12), COGVIDEOX_TEXT, COGVIDEOX, (7, 11), None),
    ("hunyuan", 128, (12, 6, 3), HUNYUAN_TEXT, HUNYUAN, (5, 9, 17, 25, 33), None),
)
# (family, heads, s_q, s_kv): the video geometries a paper or an upstream recipe uses.
VIDEO_RECIPES = frozenset(
    {
        ("wan13b", 3, 31200, 31200),
        ("wan13b", 12, 31200, 31200),
        ("wan13b", 12, 32760, 32760),
        ("wan14b", 20, 75600, 75600),
        ("ltx2", 32, 4290, 4290),
        ("kandinsky5", 32, 47616, 47616),
        ("kandinsky5", 32, 47616, KANDINSKY5_TEXT_TOKENS),
        ("cogvideox", 48, 17776, 17776),
        ("cogvideox", 30, 17776, 17776),
    }
)

# LLM attention geometries (h, h_kv) at D = 128: Qwen3-8B / Llama-3.1-8B 32/8, Qwen3-14B 40/8
# with its TP2 and TP4 splits 20/4 and 10/2, Qwen3-32B / Llama-3.1-70B 64/8 with TP4 and TP8
# splits 16/2 and 8/1, Qwen3-30B-A3B 32/4.
LLM_GEOMETRIES = ((32, 8), (40, 8), (20, 4), (10, 2), (64, 8), (16, 2), (8, 1), (32, 4))
# Pre-training: packed sequences, micro-batches of 16k tokens.
PRETRAIN_LENGTHS = ((1024, 16), (2048, 8), (4096, 4), (8192, 2))
# Long-context continued pre-training: one sequence per rank.
LONG_GEOMETRIES = ((32, 8), (40, 8), (64, 8), (16, 2), (8, 1), (32, 4))
LONG_LENGTHS = (16384, 32768)
LONGEST_GEOMETRIES = ((16, 2), (8, 1))
LONGEST_LENGTH = 65536
# Fine-tuning: dynamically padded batches (length a multiple of 64), batch x length <= 32k.
SFT_CASES = 36
SFT_LENGTH_RANGE = (1024, 12288)
SFT_BATCHES = (1, 2, 4)
# Qwen3.6-27B (24/4) and its TP2 / TP4 splits, Qwen3.5-4B (16/4) and its TP2 split; the
# full-attention layers of these hybrid models use D = 256.
D256_GEOMETRIES = ((24, 4), (12, 2), (6, 1), (16, 4), (8, 2))
D256_LENGTHS = ((4096, 2), (8192, 1), (16384, 1), (32768, 1))
# The Attn-QAT paper's Qwen3-14B fine-tuning configuration (sequence length 8192).
LLM_RECIPE = ("llm_pretrain", 1, 40, 8, 8192)

REAL_WAN13B_BLOCKS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27)
REAL_WAN13B_SIGMAS = (10, 50, 90)
REAL_WAN14B_BLOCKS = (0, 39)
REAL_COGVIDEOX_BLOCKS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27)
REAL_QWEN3_14B_LAYERS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39)
REAL_QWEN3_14B_TEXTS = (1, 2)
REAL_QWEN35_4B_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31)
REAL_LLM_LENGTH = 8192


def default_seed(suite_id: str = SUITE_ID, task_id: str = TASK_ID) -> int:
    """The seed a redraw uses when none is given, derived from the suite identity alone."""
    digest = hashlib.sha256(f"{suite_id}|{task_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


DEFAULT_SEED = default_seed()


def _row(family, tier, b, h, h_kv, d, s_q, s_kv, causal, layer=None):
    return {"family": family, "tier": tier, "b": b, "h": h, "h_kv": h_kv, "d": d, "s_q": s_q,
            "s_kv": s_kv, "causal": causal, "layer": layer}


def _sft_rows(chooser: random.Random) -> list[dict]:
    """Fine-tuning cases: a drawn geometry, padded length and batch size per case."""
    rows, seen = [], set()
    while len(rows) < SFT_CASES:
        h, h_kv = chooser.choice(LLM_GEOMETRIES)
        s = chooser.randrange(SFT_LENGTH_RANGE[0] // 64, SFT_LENGTH_RANGE[1] // 64 + 1) * 64
        b = chooser.choice([x for x in SFT_BATCHES if x * s <= 32768])
        if s in PUBLIC_LENGTHS or s & (s - 1) == 0 or (h, h_kv, s) in seen:
            continue
        seen.add((h, h_kv, s))
        rows.append(_row("llm_sft", "sweep", b, h, h_kv, 128, s, s, True))
    return rows


def grid_cases(chooser: random.Random | None = None) -> list[dict]:
    """Every scored case without its seed, gate flag or id."""
    chooser = chooser if chooser is not None else random.Random(DEFAULT_SEED)
    rows = []
    for family, d, heads, prefix, lattice, frames, s_kv in VIDEO_GRID:
        for h in heads:
            for t in frames:
                s_q = prefix + lattice * t
                kv = s_q if s_kv is None else s_kv
                tier = "recipe" if (family, h, s_q, kv) in VIDEO_RECIPES else "sweep"
                rows.append(_row(family, tier, 1, h, h, d, s_q, kv, False))
    for block in REAL_WAN13B_BLOCKS:
        for sigma in REAL_WAN13B_SIGMAS:
            rows.append(_row("real_video", "real", 1, 12, 12, 128, 32760, 32760, False,
                             f"wan13b-b{block:02d}-s{sigma:03d}"))
    for block in REAL_WAN14B_BLOCKS:
        rows.append(_row("real_video", "real", 1, 20, 20, 128, 75600, 75600, False,
                         f"wan14b-b{block:02d}-s050"))
    for index, block in enumerate(REAL_COGVIDEOX_BLOCKS):
        sigma = (10, 90)[index % 2]
        rows.append(_row("real_video", "real", 1, 30, 30, 64, 17776, 17776, False,
                         f"cogvideox2b-b{block:02d}-s{sigma:03d}"))

    family, b, h, h_kv, s = LLM_RECIPE
    rows.append(_row(family, "recipe", b, h, h_kv, 128, s, s, True))
    for h, h_kv in LLM_GEOMETRIES:
        for s, b in PRETRAIN_LENGTHS:
            rows.append(_row("llm_pretrain", "sweep", b, h, h_kv, 128, s, s, True))
    for h, h_kv in LONG_GEOMETRIES:
        for s in LONG_LENGTHS:
            rows.append(_row("llm_long", "sweep", 1, h, h_kv, 128, s, s, True))
    for h, h_kv in LONGEST_GEOMETRIES:
        rows.append(_row("llm_long", "sweep", 1, h, h_kv, 128, LONGEST_LENGTH, LONGEST_LENGTH, True))
    rows.extend(_sft_rows(chooser))
    for h, h_kv in D256_GEOMETRIES:
        for s, b in D256_LENGTHS:
            rows.append(_row("llm_d256", "sweep", b, h, h_kv, 256, s, s, True))
    for layer in REAL_QWEN3_14B_LAYERS:
        for text in REAL_QWEN3_14B_TEXTS:
            rows.append(_row("real_llm", "real", 1, 40, 8, 128, REAL_LLM_LENGTH, REAL_LLM_LENGTH,
                             True, f"qwen3-14b-l{layer:02d}-x{text:02d}"))
    for layer in REAL_QWEN35_4B_LAYERS:
        rows.append(_row("real_llm", "real", 1, 16, 4, 256, REAL_LLM_LENGTH, REAL_LLM_LENGTH,
                         True, f"qwen35-4b-l{layer:02d}-x01"))
    return rows


def generate_manifest(rng: random.Random | None = None) -> dict:
    """Draw the fine-tuning cases, the seeds and the hidden half of each family's sweep cases
    that runs the golden."""
    chooser = rng if rng is not None else random.Random(DEFAULT_SEED)
    rows = grid_cases(chooser)
    gated_sweep: set[int] = set()
    for family in sorted({row["family"] for row in rows}):
        sweep = [i for i, row in enumerate(rows) if row["family"] == family and row["tier"] == "sweep"]
        gated_sweep.update(chooser.sample(sweep, (len(sweep) + 1) // 2))
    seeds: set[int] = set()
    cases = []
    for index, row in enumerate(rows):
        seed = chooser.randrange(10_000, 2**31 - 1)
        while seed in seeds:
            seed = chooser.randrange(10_000, 2**31 - 1)
        seeds.add(seed)
        gated = row["tier"] != "sweep" or index in gated_sweep
        cases.append({"case_id": f"holdout-{index + 1:03d}", **row, "seed": seed, "gated": gated})
    if len(cases) != EXPECTED_CASE_COUNT or any(set(x) != CASE_KEYS for x in cases):
        raise RuntimeError(f"manifest schema differs ({len(cases)} cases)")
    return {"schema_version": 2, "suite_id": SUITE_ID, "cases": cases}


def main(argv=None):
    """Write the manifest atomically, mode 0600, refusing to replace an existing one."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            "seed for the draw; default %(default)s, the first eight bytes of "
            "sha256('<suite_id>|<task_id>') read big-endian"
        ),
    )
    p.add_argument(
        "--redraw",
        action="store_true",
        help=(
            "acknowledge that this draw is a NEW suite identity: the frozen "
            f"instance {FROZEN_INSTANCE} is the default build input and the "
            "instance every recorded score used, so a redraw must be recorded "
            "in portable/final-suite-manifest.json (frozen_instance) before "
            "any score is compared against it"
        ),
    )
    a = p.parse_args(argv)
    path = a.output.absolute()
    if not a.redraw:
        raise SystemExit(
            f"refusing to draw: {FROZEN_INSTANCE} is the frozen default build "
            "input; pass --redraw (with an explicit --seed) to mint a new suite identity"
        )
    if path.exists() or path.is_symlink():
        raise FileExistsError("refusing to replace manifest")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    data = (
        json.dumps(generate_manifest(random.Random(a.seed)), indent=2, sort_keys=True)
        + "\n"
    ).encode()
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{os.urandom(4).hex()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    path.chmod(0o600)
    print(
        json.dumps(
            {
                "seed": a.seed,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "suite_identity": "new",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
