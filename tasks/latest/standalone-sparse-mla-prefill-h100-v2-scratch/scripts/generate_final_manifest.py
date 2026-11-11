#!/usr/bin/env python3
"""Draw the root-only scored final manifest for sparse-MLA prefill.

The scored suite covers four deployed geometries (DeepSeek-V3.2, GLM-5,
DeepSeek-V4-Flash, DeepSeek-V4-Pro) in eight groups, about 250 cases per family:

* ``anchor``: FlashMLA's own sparse-prefill performance cases (s_q = 4096) and
  the same template at the GLM-5 geometry, with upstream inputs;
* ``trace``: streams of the prefill calls of consecutive requests from three
  production request traces, chunked at 2048 to 16384 query rows;
* ``replay``: streams of one whole-context chunked prefill, up to each model's
  ``max_position_embeddings``;
* ``multi``: one chunk over 2 to 128 requests packed into one KV workspace;
* ``pattern``: causal single calls whose indices carry a learned indexer's
  locality (``index_pattern``);
* ``sweep``: causal single calls stratified over query-chunk and context sizes;
* ``heads``: causal single calls at the head count of one tensor-parallel rank;
* ``length``: decode-context-parallel shards whose rows hold about topk/2, /4 or
  /8 local keys, evenly or raggedly (``ragged``).

The plan (group, family, heads and sizes of every case, the trace windows and the
request lengths) is a Judge-only input: ``--plan`` names the ``judge`` list of a
plan file written on the host from Judge-only request traces.  This script adds
what the plan does not fix - the 63-bit case seeds, the sink mode of each
non-anchor case and the hidden half of the sweep-tier cases whose first measured
trial is checked against the reference - from a seeded ``random.Random``.  The
default seed is the first eight bytes of ``sha256("<suite_id>|<task_id>")`` read
big-endian, so the draw needs no stored state; ``--redraw`` acknowledges that the
output is a new suite identity whose ``frozen_instance`` record must be updated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Mapping, Sequence


SUITE_ID = "final-holdout-v4"
TASK_ID = "standalone_sparse_mla_prefill_h100_scratch_v2"
SCHEMA_VERSION = 5
FROZEN_INSTANCE = "judge/final_cases.json under the asset root"
FAMILIES = ("v32", "glm5", "v4_flash", "v4_pro")
FAMILY_HEADS = {"v32": 128, "glm5": 64, "v4_flash": 64, "v4_pro": 128}
GROUPS = ("anchor", "trace", "replay", "multi", "pattern", "sweep", "heads", "length")
STREAM_GROUPS = ("trace", "replay")
# Sweep-tier groups check the first measured trial of a hidden half of their cases.
HALF_GATED_GROUPS = ("sweep", "heads", "length")
SEED_RANGE = (1 << 32, 1 << 63)
EXPECTED_CASE_COUNT = 997
BASE_KEYS = ("case_id", "group", "family", "h_q", "s_q", "s_kv", "sink_mode", "gated", "seed")


def default_seed(suite_id: str = SUITE_ID, task_id: str = TASK_ID) -> int:
    digest = hashlib.sha256(f"{suite_id}|{task_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


DEFAULT_SEED = default_seed()


def _case(plan: Mapping[str, Any], number: int, seed: int, sink_mode: str, gated: bool) -> dict[str, Any]:
    group = str(plan["group"])
    family = str(plan["family"])
    if group not in GROUPS or family not in FAMILIES:
        raise ValueError(f"plan row {number} has an unknown group or family")
    case: dict[str, Any] = {
        "case_id": f"final-{number:04d}-{group}-{family.replace('_', '-')}",
        "group": group,
        "family": family,
        "h_q": int(plan["h_q"]),
        "s_q": int(plan["s_q"]),
        "s_kv": int(plan["s_kv"]),
        "sink_mode": sink_mode,
        "gated": gated,
        "seed": seed,
    }
    if group == "length":
        divisor = str(plan["length"]).split("/")
        if len(divisor) != 2 or divisor[0] != "topk":
            raise ValueError(f"plan row {number}: length must read topk/N")
        case["length_divisor"] = int(divisor[1])
        case["ragged"] = bool(plan["ragged"])
    elif group == "pattern":
        case["index_pattern"] = str(plan["index_pattern"])
    elif group == "multi":
        case["request_lengths"] = [int(value) for value in plan["request_lengths"]]
        case["s_kv"] = sum(case["request_lengths"])
    elif group in STREAM_GROUPS:
        case["calls"] = [[int(s_q), int(s_kv)] for s_q, s_kv in plan["call_list"]]
        case["s_q"] = max(call[0] for call in case["calls"])
        case["s_kv"] = max(call[1] for call in case["calls"])
    if group != "heads" and case["h_q"] != FAMILY_HEADS[family]:
        raise ValueError(f"plan row {number}: only heads cases change the head count")
    return case


def generate_manifest(plan_rows: Sequence[Mapping[str, Any]], rng: random.Random | None = None) -> dict[str, Any]:
    chooser = rng if rng is not None else random.Random(DEFAULT_SEED)
    seeds: set[int] = set()
    half = {group: [number for number, row in enumerate(plan_rows) if row["group"] == group]
            for group in HALF_GATED_GROUPS}
    gated_rows: set[int] = set()
    for group, numbers in half.items():
        gated_rows.update(chooser.sample(numbers, (len(numbers) + 1) // 2))
    cases = []
    for number, row in enumerate(plan_rows):
        seed = chooser.randrange(*SEED_RANGE)
        while seed in seeds:
            seed = chooser.randrange(*SEED_RANGE)
        seeds.add(seed)
        group = str(row["group"])
        if group == "anchor":
            sink_mode, gated = "upstream", True
        else:
            sink_mode = chooser.choice(("none", "finite"))
            gated = group not in HALF_GATED_GROUPS or number in gated_rows
        cases.append(_case(row, number, seed, sink_mode, gated))
    if len(cases) != EXPECTED_CASE_COUNT:
        raise RuntimeError(f"plan has {len(cases)} cases, expected {EXPECTED_CASE_COUNT}")
    if len({case["case_id"] for case in cases}) != len(cases):
        raise RuntimeError("duplicate case IDs")
    for case in cases:
        if not set(BASE_KEYS) <= set(case):
            raise RuntimeError("generated case lacks a required field")
    return {"schema_version": SCHEMA_VERSION, "suite_id": SUITE_ID, "cases": cases}


def _atomic_write_private(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace existing manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    encoded = (json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
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
    parser.add_argument("--plan", type=Path, required=True, help="plan file with a 'judge' case list")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--redraw", action="store_true",
                        help="acknowledge that the output is a new suite identity")
    args = parser.parse_args(argv)
    if not args.redraw:
        raise SystemExit("refusing to draw without --redraw: the output is a new suite identity")
    plan = json.loads(args.plan.read_text())
    output = args.output.absolute()
    _atomic_write_private(output, generate_manifest(plan["judge"], random.Random(args.seed)))
    print(json.dumps({"seed": args.seed, "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                      "size": output.stat().st_size, "suite_identity": "new"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
