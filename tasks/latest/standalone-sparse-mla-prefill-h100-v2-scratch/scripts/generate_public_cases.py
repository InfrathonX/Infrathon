#!/usr/bin/env python3
"""Write the public scored cases, ``images/work/benchmark/public_cases.json``.

``--plan`` names the ``public`` list of a plan file.  The public cases use the
three DeepSeek geometries at head counts and top-k values that no hidden case
uses, a synthetic request trace, and sizes of their own.  The six original public
shapes stay as the ``basic`` group with inputs A and B.  Seeds are below 2**32
(every hidden seed is at least 2**32) and come from a fixed ``random.Random``.

Usage: generate_public_cases.py --plan PLAN.json [--check]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = TASK_ROOT / "images/work/benchmark/public_cases.json"
SUITE_ID = "sparse-mla-public-v4"
SCHEMA_VERSION = 1
SEED = 20261108
GEOMETRY_FAMILY = {"d576": "v32", "d512_h64": "v4_flash", "d512_h128": "v4_pro",
                   "v32": "v32", "v4_flash": "v4_flash", "v4_pro": "v4_pro"}
BASIC_INPUTS = ("A", "B", "A", "B", "A", "B")
EXPECTED_COUNT = 108


def build(plan_rows: list[dict]) -> dict:
    rng = random.Random(SEED)
    cases = []
    basic = 0
    for number, row in enumerate(plan_rows):
        family = GEOMETRY_FAMILY[row["family"]]
        group = "basic" if row["group"] == "anchor" else row["group"]
        case = {
            "case_id": f"public-{number:03d}-{group}",
            "group": group,
            "family": family,
            "h_q": int(row["h_q"]),
            "topk": int(row["topk"]),
            "s_q": int(row["s_q"]),
            "s_kv": int(row["s_kv"]),
            "sm_scale": 0.5 if group == "basic" else float(row["sm_scale"]),
            "sink_mode": "finite" if group == "basic" else rng.choice(("none", "finite")),
            "seed": rng.randrange(1, 1 << 32),
        }
        if group == "basic":
            case["inputs"] = BASIC_INPUTS[basic]
            basic += 1
        elif group == "length":
            case["length_divisor"] = int(str(row["length"]).split("/")[1])
            case["ragged"] = bool(row["ragged"])
        elif group == "pattern":
            case["index_pattern"] = str(row["index_pattern"])
        elif group == "multi":
            case["request_lengths"] = [int(value) for value in row["request_lengths"]]
            case["s_kv"] = sum(case["request_lengths"])
        elif group in ("trace", "replay"):
            case["calls"] = [[int(a), int(b)] for a, b in row["call_list"]]
            case["s_q"] = max(call[0] for call in case["calls"])
            case["s_kv"] = max(call[1] for call in case["calls"])
        cases.append(case)
    if len(cases) != EXPECTED_COUNT or basic != len(BASIC_INPUTS):
        raise RuntimeError(f"plan has {len(cases)} public cases ({basic} basic)")
    return {"schema_version": SCHEMA_VERSION, "suite_id": SUITE_ID, "cases": cases}


def encode(payload: dict) -> bytes:
    rows = [json.dumps(case, sort_keys=True, separators=(", ", ": ")) for case in payload["cases"]]
    lines = ["{", f'  "schema_version": {payload["schema_version"]},', f'  "suite_id": "{payload["suite_id"]}",',
             '  "cases": [', *(f"    {row}," for row in rows[:-1]), f"    {rows[-1]}", "  ]", "}"]
    return ("\n".join(lines) + "\n").encode()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    encoded = encode(build(json.loads(args.plan.read_text())["public"]))
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_bytes() != encoded:
            print(f"{OUTPUT.relative_to(TASK_ROOT)} differs from the plan")
            return 1
        return 0
    OUTPUT.write_bytes(encoded)
    print(f"wrote {OUTPUT.relative_to(TASK_ROOT)} ({len(encoded)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
