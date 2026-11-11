#!/usr/bin/env python3
"""Convert a frozen suite plan into the scored manifests of paged GQA decode v4.

The v4 scored workload models one tensor-parallel rank of a serving engine
during decode, in four families: ``decode`` (one decode step of a busy server,
S_q = 1), ``spec`` (2-16 query tokens per request at small batch sizes),
``long`` (contexts up to 1,048,576 tokens) and ``stream`` (consecutive decode
steps of one server, requests joining and leaving between calls).  Every
single-call case lists its per-request cache lengths and shared-prefix blocks
(``block_tokens`` tokens each: 512 for trace prefix caching, 16 or the page
size for the parallel-sampling, beam-search and shared-prompt workloads of the
PagedAttention paper) and whether Q, K and V carry outliers;
every stream case lists its requests as ``[start, end, input_length, blocks]``.
The plan that produces them (model configurations, request traces, server
simulation) lives outside the repository, in the Judge-only asset
``<INFRATHON_ASSETS_ROOT>/<task>/judge/source/`` (generator, traces and the
frozen plan ``paged-gqa-plan-v2.json``); this tool only converts and checks
it, so the frozen plan stays the single source of truth.  Converting that plan
reproduces ``judge/final_cases.json`` and ``public_cases.json`` byte for byte
(``tests/test_private_inputs.py``).

Outputs:

* the root-only Judge manifest (``--private-output``), suite
  ``final-holdout-v4``; it must never enter the repository or any Work build
  context (it belongs in ``<INFRATHON_ASSETS_ROOT>/<task>/judge/``);
* ``images/work/benchmark/public_cases.json`` (``--write-public``), the
  public cases shipped to Work;
* the ``cases`` lists of ``portable/{public,interactive,final}-suite-manifest.json``
  (``--write-portable``): a public summary of each public case, and only opaque
  case IDs and families for the hidden suite.

Hidden seeds are at least ``FINAL_SEED_MINIMUM``; public seeds are below it.

    scripts/generate_final_manifest.py --plan PLAN.json --private-output OUT.json
    scripts/generate_final_manifest.py --plan PLAN.json --write-public --write-portable
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


TASK_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 4
SUITE_ID = "final-holdout-v4"
PUBLIC_SUITE_ID = "public-v4"
TASK_ID = "standalone_paged_gqa_decode_bf16_h100_scratch_v2"
PUBLIC_PATH = TASK_ROOT / "images/work/benchmark/public_cases.json"
PORTABLE = TASK_ROOT / "portable"
FAMILIES = ("decode", "spec", "long", "stream")
FINAL_CASE_COUNT = 1114
PUBLIC_CASE_COUNT = 127
FINAL_SEED_MINIMUM = 1 << 32
CALL_KEYS = ("case_id", "family", "s_q", "h_q", "h_kv", "page_size", "block_tokens", "lengths", "blocks",
             "table_width", "pool_pages", "fill", "values", "seed")
STREAM_KEYS = ("case_id", "family", "s_q", "h_q", "h_kv", "page_size", "block_tokens", "steps", "requests",
               "table_width", "pool_pages", "values", "seed")


def _load_runtime():
    path = TASK_ROOT / "runtime/paged_gqa_decode_runtime.py"
    spec = importlib.util.spec_from_file_location("manifest_generator_runtime", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def convert_case(raw: Mapping[str, Any], case_id: str) -> dict[str, Any]:
    keys = STREAM_KEYS if raw.get("family") == "stream" else CALL_KEYS
    case = {key: raw[key] for key in keys if key != "case_id"}
    return {"case_id": case_id, **case}


def convert(plan: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runtime = _load_runtime()
    out: dict[str, list[dict[str, Any]]] = {}
    for split, prefix, count in (("judge", "final", FINAL_CASE_COUNT), ("public", "public", PUBLIC_CASE_COUNT)):
        raw_cases = plan[split]
        if len(raw_cases) != count:
            raise SystemExit(f"plan {split} holds {len(raw_cases)} cases, expected {count}")
        cases = []
        for index, raw in enumerate(raw_cases):
            case = convert_case(raw, f"{prefix}-{index:04d}-{raw['family']}")
            runtime.validate_scored_case(case)
            cases.append(case)
        out[split] = cases
    judge, public = out["judge"], out["public"]
    for case in judge:
        if case["seed"] < FINAL_SEED_MINIMUM:
            raise SystemExit(f"{case['case_id']}: hidden seeds must be >= {FINAL_SEED_MINIMUM}")
    for case in public:
        if case["seed"] >= FINAL_SEED_MINIMUM:
            raise SystemExit(f"{case['case_id']}: public seeds must be < {FINAL_SEED_MINIMUM}")
    for split, cases in (("judge", judge), ("public", public)):
        families = Counter(case["family"] for case in cases)
        if set(families) != set(FAMILIES):
            raise SystemExit(f"{split} must exercise all four families, has {dict(families)}")
    heads_judge = {(case["h_q"], case["h_kv"]) for case in judge}
    heads_public = {(case["h_q"], case["h_kv"]) for case in public}
    if heads_judge & heads_public:
        raise SystemExit(f"public and hidden head configurations overlap: {heads_judge & heads_public}")
    return judge, public


def encode(payload: Mapping[str, Any]) -> bytes:
    rows = [json.dumps(case, sort_keys=True, allow_nan=False, separators=(",", ":"))
            for case in payload["cases"]]
    lines = ["{", f'  "schema_version": {payload["schema_version"]},',
             f'  "suite_id": {json.dumps(payload["suite_id"])},', '  "cases": [',
             *(f"    {row}," for row in rows[:-1]), f"    {rows[-1]}", "  ]", "}"]
    return ("\n".join(lines) + "\n").encode("utf-8")


def public_summary(case: Mapping[str, Any]) -> dict[str, Any]:
    summary = {"case_id": case["case_id"], "family": case["family"], "role": "performance",
               "s_q": case["s_q"], "h_q": case["h_q"], "h_kv": case["h_kv"],
               "page_size": case["page_size"], "seed": case["seed"]}
    if case["family"] == "stream":
        summary.update(steps=case["steps"], requests=len(case["requests"]))
    else:
        summary.update(b=len(case["lengths"]), l_max=max(case["lengths"]))
    return summary


def write_portable(judge: Sequence[Mapping[str, Any]], public: Sequence[Mapping[str, Any]]) -> None:
    hidden = [{"case_id": case["case_id"], "family": case["family"], "role": "performance",
               "weight": "family geometric mean"} for case in judge]
    for name, cases in (("final", hidden), ("interactive", hidden),
                        ("public", [public_summary(case) for case in public])):
        path = PORTABLE / f"{name}-suite-manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["cases"] = cases
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--write-public", action="store_true")
    parser.add_argument("--write-portable", action="store_true")
    parser.add_argument("--force", action="store_true", help="replace an existing private output")
    args = parser.parse_args(argv)
    judge, public = convert(json.loads(args.plan.read_text(encoding="utf-8")))
    private = {"schema_version": SCHEMA_VERSION, "suite_id": SUITE_ID, "cases": judge}
    public_payload = {"schema_version": SCHEMA_VERSION, "suite_id": PUBLIC_SUITE_ID, "cases": public}
    for split, cases in (("judge", judge), ("public", public)):
        print(f"{split}: {len(cases)} cases {dict(Counter(case['family'] for case in cases))}")
    if args.private_output is not None:
        target = args.private_output
        if target.resolve().is_relative_to(TASK_ROOT.resolve()):
            raise SystemExit("the hidden manifest must not be written inside the task tree")
        if target.exists() and not args.force:
            raise SystemExit(f"refusing to replace {target} (use --force for a new suite identity)")
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        data = encode(private)
        temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
        os.replace(temporary, target)
        print(f"wrote {target} sha256={hashlib.sha256(data).hexdigest()} size={len(data)}")
    if args.write_public:
        data = encode(public_payload)
        PUBLIC_PATH.write_bytes(data)
        print(f"wrote {PUBLIC_PATH.relative_to(TASK_ROOT)} sha256={hashlib.sha256(data).hexdigest()}")
    if args.write_portable:
        write_portable(judge, public)
        print("updated portable case lists")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
