#!/usr/bin/env python3
"""Draw the Judge-only final suite (final-holdout-v3) of the quantized attention task.

The suite models attention calls of real deployments in four families:

* ``video``: non-causal self attention of video diffusion transformers, at their
  resolutions, clip lengths (latent frames), sequence-parallel head splits and CFG batch;
* ``image``: non-causal joint attention of image diffusion transformers, at several
  resolutions and aspect ratios;
* ``llm``: causal prefill of language models, per tensor-parallel rank, over context bands
  from 2K to 256K tokens, each band holding hidden, non-aligned lengths;
* ``real``: attention layers captured from real models (see ``REAL_SOURCES``).

Each synthetic case is ``recipe`` (a published default configuration) or ``sweep`` (the same
model's other supported settings).  Every recipe and real case runs the golden check; a
hidden half of each family's sweep cases does.

The draw is deterministic: a ``random.Random`` seeded from ``sha256("<suite>|<task>")``, so
every host reproduces the same instance.  The output is Judge-only; it must never enter a
Work build context.  Usage: generate_final_manifest.py --output PATH [--seed N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

SUITE_ID = "final-holdout-v3"
TASK_ID = "sageattention_quantized_attention_bf16_h100_seed_v2"
CASE_KEYS = ("case_id", "family", "tier", "model", "b", "h_q", "h_kv", "d", "l",
             "is_causal", "regime", "seed", "gated", "layer")
MAX_L = 262144

# Public geometries and lengths (images/work/benchmark/public_runner.py); no Judge case may
# use either.
PUBLIC_GEOMETRIES = frozenset({(20, 20, 128, False), (16, 16, 64, False), (16, 16, 128, False),
                               (24, 6, 128, True), (12, 4, 64, True), (16, 4, 128, True)})
PUBLIC_LENGTHS = frozenset({5184, 24192, 61441, 96001, 6697, 20276, 44017, 2050, 5001, 4700,
                            4160, 14336, 65001, 190001, 3001, 7168, 32001})

# (model, H, D, batches, {resolution: tokens per latent frame}, latent frames, sequence-
#  parallel degrees, text tokens, recipe (resolution, frames, sp))
VIDEO = (
    ("wan2.1-1.3b", 12, 128, (1, 2), {"480p": 1560}, (5, 9, 13, 17, 21), (1, 2, 4), 0, ("480p", 21, 1)),
    ("wan2.1-14b", 40, 128, (1, 2), {"480p": 1560, "720p": 3600}, (5, 9, 13, 17, 21), (1, 4, 8), 0, ("720p", 21, 1)),
    ("hunyuanvideo", 24, 128, (1,), {"480p": 1590, "540p": 2040, "720p": 3600}, (9, 17, 25, 33), (1, 2, 4, 8), 256, ("720p", 33, 1)),
    ("mochi-1", 24, 128, (1, 2), {"480p": 1590}, (7, 14, 21, 28), (1, 2, 4), 256, ("480p", 28, 1)),
    ("cogvideox-2b", 30, 64, (2,), {"480p": 1350}, (5, 9, 13), (1, 2), 226, ("480p", 13, 1)),
    ("cogvideox-5b", 48, 64, (2,), {"480p": 1350, "768p": 4080}, (4, 7, 11, 13, 21), (1, 2, 4), 226, ("768p", 11, 1)),
    ("ltx-video", 32, 64, (1, 2), {"512p": 384, "720p": 920}, (9, 17, 25, 33), (1,), 128, ("512p", 17, 1)),
)
IMAGE_RES = ((512, 512), (768, 768), (1024, 1024), (1024, 768), (768, 1344), (1344, 768),
             (1440, 1440), (2048, 2048))
# (model, H, D, text tokens, largest side, batches, recipe batch)
IMAGE = (
    ("sd3.5-medium", 24, 64, 333, 1440, (1, 2), 2),
    ("sd3.5-large", 38, 64, 333, 1440, (1, 2), 2),
    ("flux.1-dev", 24, 128, 512, 2048, (1, 4), 1),
)
# (model, Hq per rank, Hkv per rank, context window, recipe bands)
LLM = (
    ("llama-3.1-8b", 32, 8, 131072, (16384, 131072)),
    ("llama-3-8b-262k", 32, 8, 262144, (262144,)),
    ("llama-3.2-3b", 24, 8, 131072, ()),
    ("qwen3-1.7b", 16, 8, 40960, ()),
    ("qwen2.5-7b", 28, 4, 131072, ()),
    ("qwen2.5-14b", 40, 8, 131072, ()),
    ("llama-3.1-70b-tp2", 32, 4, 131072, ()),
    ("llama-3.1-70b-tp4", 16, 2, 131072, ()),
    ("llama-3.1-70b-tp8", 8, 1, 131072, (131072,)),
    ("qwen3-235b-a22b-tp2", 32, 2, 131072, ()),
    ("qwen3-235b-a22b-tp4", 16, 1, 131072, ()),
    ("mistral-large-2-tp4", 24, 2, 131072, ()),
    ("mistral-large-2-tp8", 12, 1, 131072, ()),
    ("llama-3.1-405b-tp8", 16, 1, 131072, ()),
)
BANDS = (2048, 4096, 8192, 16384, 32768, 65536, 131072, 196608, 262144)
DRAWS_PER_BAND = 4

# Real captured layers: source -> (B, Hq, Hkv, D, L, causal, captured layer numbers).  The
# recipes are fixtures/recipe/real_layers/RECIPE_<source>.json.
REAL_SOURCES = {
    "cogvideox2b-s00": (2, 30, 30, 64, 17776, False, tuple(range(30))),
    "llama31-16k": (1, 32, 8, 128, 16384, True, tuple(range(32))),
    "cogvideox2b-s10": (2, 30, 30, 64, 17776, False, tuple(range(10))),
    "cogvideox2b-s40": (2, 30, 30, 64, 17776, False, tuple(range(10))),
    "wan13b-480p": (1, 12, 12, 128, 32760, False, tuple(range(15))),
    "wan14b-720p": (1, 40, 40, 128, 75600, False, tuple(range(8))),
    "llama31-64k": (1, 32, 8, 128, 65536, True, (0, 10, 20, 31)),
    "llama31-128k": (1, 32, 8, 128, 131072, True, (0, 10, 20, 31)),
    "llama3-262k": (1, 32, 8, 128, 262144, True, (0, 31)),
    "sd35m-1024": (2, 24, 24, 64, 4429, False, tuple(range(24))),
}


def default_seed(suite_id: str = SUITE_ID, task_id: str = TASK_ID) -> int:
    """``int.from_bytes(sha256(f"{suite_id}|{task_id}").digest()[:8], "big")``: no stored state."""
    return int.from_bytes(hashlib.sha256(f"{suite_id}|{task_id}".encode("utf-8")).digest()[:8], "big")


DEFAULT_SEED = default_seed()


def _synthetic(rng: random.Random) -> list[dict]:
    cases: list[dict] = []

    def add(family, tier, model, b, hq, hkv, d, l, causal):
        cases.append({"family": family, "tier": tier, "model": model, "b": b, "h_q": hq, "h_kv": hkv,
                      "d": d, "l": l, "is_causal": causal, "layer": None})

    for model, h, d, batches, res, frames, sps, text, recipe in VIDEO:
        for name, per in res.items():
            for t in frames:
                if model == "cogvideox-5b" and (name == "480p") != (t == 13):
                    continue  # CogVideoX-5B is 480p at 49 frames; CogVideoX1.5-5B is 768p
                for sp in sps:
                    for b in batches:
                        tier = "recipe" if (name, t, sp, b) == (*recipe, batches[0]) else "sweep"
                        add("video", tier, model, b, h // sp, h // sp, d, per * t + text, False)
    for model, h, d, text, largest, batches, recipe_b in IMAGE:
        for w, hh in IMAGE_RES:
            if max(w, hh) > largest:
                continue
            for b in batches:
                tier = "recipe" if (w, hh, b) == (1024, 1024, recipe_b) else "sweep"
                add("image", tier, model, b, h, h, d, (w // 16) * (hh // 16) + text, False)
    for model, hq, hkv, ctx, recipe_bands in LLM:
        for band in BANDS:
            if band > ctx:
                continue
            for i in range(DRAWS_PER_BAND):
                while True:
                    l = band - rng.randrange(1, band // 8)
                    if l % 128 and l not in PUBLIC_LENGTHS:
                        break
                b = 4 if (band <= 8192 and i == 0) else 1
                tier = "recipe" if (band in recipe_bands and i == 0) else "sweep"
                add("llm", tier, model, b, hq, hkv, 128, l, True)
    return cases


def _real() -> list[dict]:
    cases = []
    for source, (b, hq, hkv, d, l, causal, layers) in REAL_SOURCES.items():
        for layer in layers:
            cases.append({"family": "real", "tier": "real", "model": source, "b": b, "h_q": hq,
                          "h_kv": hkv, "d": d, "l": l, "is_causal": causal,
                          "layer": f"{source}/{layer:02d}"})
    return cases


def generate_manifest(rng: random.Random | None = None) -> dict:
    rng = rng if rng is not None else random.Random(default_seed())
    synthetic = _synthetic(rng)
    rng.shuffle(synthetic)
    # A hidden half of each family's sweep cases runs the golden check.
    for family in ("video", "image", "llm"):
        sweep = [c for c in synthetic if c["family"] == family and c["tier"] == "sweep"]
        for i, case in enumerate(sweep):
            case["gated"] = i % 2 == 0
        for case in synthetic:
            if case["family"] == family and case["tier"] == "recipe":
                case["gated"] = True
    cases = _real() + synthetic
    seeds: set[int] = set()
    for index, case in enumerate(cases):
        seed = rng.randrange(10_000, 2**31 - 1)
        while seed in seeds:
            seed = rng.randrange(10_000, 2**31 - 1)
        seeds.add(seed)
        case["seed"] = seed
        case["case_id"] = f"final-{index:04d}"
        case.setdefault("gated", True)
        case["regime"] = "real" if case["family"] == "real" else rng.choice(("norm", "raw"))
        if (case["h_q"], case["h_kv"], case["d"], case["is_causal"]) in PUBLIC_GEOMETRIES and case["family"] != "real":
            raise RuntimeError(f"case {case['case_id']} uses a public geometry")
        if case["l"] in PUBLIC_LENGTHS or not 0 < case["l"] <= MAX_L:
            raise RuntimeError(f"case {case['case_id']} has a forbidden length")
    return {"schema_version": 1, "suite_id": SUITE_ID,
            "cases": [{k: c[k] for k in CASE_KEYS} for c in cases]}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="seed for the draw; default: the first eight bytes of sha256('<suite_id>|<task_id>')")
    p.add_argument("--redraw", action="store_true",
                   help="acknowledge that writing a manifest mints (or re-mints) the Judge-only suite instance; "
                        "the frozen instance's SHA-256 is recorded in portable/final-suite-manifest.json")
    a = p.parse_args(argv)
    if not a.redraw:
        raise SystemExit("refusing to draw: the Judge-only final suite is frozen; pass --redraw to write an instance")
    path = a.output.absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError("refusing to replace a manifest")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    data = (json.dumps(generate_manifest(random.Random(a.seed)), indent=1, sort_keys=True) + "\n").encode()
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    print(json.dumps({"seed": a.seed, "sha256": hashlib.sha256(data).hexdigest(), "suite_id": SUITE_ID}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
