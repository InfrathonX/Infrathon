#!/usr/bin/env python3
"""Convert a frozen suite plan into the mode-0600 Judge-only manifest of TileLang SSD v3.

The plan (a JSON file kept outside the repository with the other Judge-only
inputs) lists every scored case: its geometry, batch, length, workload family
(``batch``, ``serving``, ``long`` or ``real``) and tier (``recipe``, ``sweep``
or ``real``).  Lengths of the serving cases are real request lengths drawn from
the Mooncake traces and capped by each model's context limit; the plan is the
single source of truth for them, and this script only converts it.  Real cases
take their captured layers, in order, from ``fixtures/recipe/real_layers/
REAL_LAYERS.json``.

What a draw adds is one 63-bit seed per case and, per synthetic case, which of
its timed trials is compared against the FP32 reference.  The draw comes from
a seeded ``random.Random`` whose default seed is the first eight bytes of
``sha256("<suite_id>|<task_id>")`` read big-endian, so another host converting
the same plan regenerates the identical instance.  ``--redraw`` is the required
acknowledgement that a new draw is a new suite identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Mapping, Sequence


SUITE_ID = "tilelang-ssd-final-v3"
SCHEMA_VERSION = 3
TASK_ID = "tilelang_mamba2_ssd_forward_h100_seed_v2"
TASK_ROOT = Path(__file__).resolve().parents[1]
FROZEN_INSTANCE = "judge/private/final_cases.json"
RECIPE = TASK_ROOT / "fixtures/recipe/real_layers/REAL_LAYERS.json"
FAMILY_COUNTS = {"batch": 42, "serving": 54, "long": 18, "real": 24}
TRIALS = 3
# Disclosed public seeds (the Work runtime's PUBLIC_SEED_BASE + index) and the
# edge-fixture seeds; a final seed must never coincide with one of them.
PUBLIC_SEED_BASE = 0x0066_0301
PUBLIC_CASE_COUNT = 24
SEED_MAX = 0x7FFF_FFFF_FFFF_FFFF
# The geometry of a real case names the capture source of its layers.
REAL_SOURCES = {"granite_h_micro": "granite_h_micro", "granite_h_small": "bamba_9b_v2"}


def default_seed(suite_id: str = SUITE_ID, task_id: str = TASK_ID) -> int:
    """``int.from_bytes(sha256(f"{suite_id}|{task_id}").digest()[:8], "big")``."""
    digest = hashlib.sha256(f"{suite_id}|{task_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


DEFAULT_SEED = default_seed()


def plan_from_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """The plan a manifest was converted from (its cases without seeds or checks)."""
    return {
        "judge": [
            {
                "family": case["family"],
                "tier": case["tier"],
                "geometry": case["geometry"],
                "b": case["batch"],
                "L": case["seqlen"],
            }
            for case in manifest["cases"]
        ]
    }


def real_layer_names(recipe: Mapping[str, Any], source: str) -> list[str]:
    entry = recipe["sources"][source]
    return [f"{entry['name_prefix']}{int(layer):02d}" for layer in entry["layers"]]


def generate_manifest(
    plan: Mapping[str, Any], recipe: Mapping[str, Any], rng: random.Random | None = None
) -> dict[str, Any]:
    """Return the manifest of the plan's Judge cases with seeds and checked trials."""
    chooser = rng if rng is not None else random.Random(DEFAULT_SEED)
    raw_cases = plan["judge"]
    counts: dict[str, int] = {}
    for raw in raw_cases:
        counts[str(raw["family"])] = counts.get(str(raw["family"]), 0) + 1
    if counts != FAMILY_COUNTS:
        raise SystemExit(f"the plan's Judge family counts {counts} differ from {FAMILY_COUNTS}")

    layers = {geometry: iter(real_layer_names(recipe, source)) for geometry, source in REAL_SOURCES.items()}
    reserved = {PUBLIC_SEED_BASE + index for index in range(PUBLIC_CASE_COUNT)}
    reserved |= set(range(0xE001, 0xE019))
    seeds: set[int] = set()
    cases: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_cases):
        family = str(raw["family"])
        geometry = str(raw["geometry"])
        seed = chooser.randint(1, SEED_MAX)
        while seed in seeds or seed in reserved:
            seed = chooser.randint(1, SEED_MAX)
        seeds.add(seed)
        check_trial = chooser.randrange(TRIALS)
        layer = None
        if family == "real":
            layer = next(layers[geometry])
            check_trial = 0
        cases.append(
            {
                "case_id": f"final-{index:03d}-{family}",
                "geometry": geometry,
                "family": family,
                "tier": str(raw["tier"]),
                "batch": int(raw["b"]),
                "seqlen": int(raw["L"]),
                "regime": "mixed",
                "seed": seed,
                "check_trial": check_trial,
                "layer": layer,
            }
        )
    for geometry, remaining in layers.items():
        if next(remaining, None) is not None:
            raise SystemExit(f"the plan uses fewer {geometry} layers than the recipe captures")
    return {"schema_version": SCHEMA_VERSION, "suite_id": SUITE_ID, "cases": cases}


def _atomic_write_private(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace existing manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{os.urandom(4).hex()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True, help="the frozen Judge-only suite plan")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            "seed for the draw; default %(default)s, the first eight bytes of "
            "sha256('<suite_id>|<task_id>') read big-endian"
        ),
    )
    parser.add_argument(
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
    args = parser.parse_args(argv)
    if not args.redraw:
        raise SystemExit(
            f"refusing to draw: {FROZEN_INSTANCE} is the frozen default build "
            "input; pass --redraw (with an explicit --seed) to mint a new suite identity"
        )
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
    output = args.output.absolute()
    _atomic_write_private(output, generate_manifest(plan, recipe, random.Random(args.seed)))
    print(
        json.dumps(
            {
                "seed": args.seed,
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "suite_identity": "new",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
