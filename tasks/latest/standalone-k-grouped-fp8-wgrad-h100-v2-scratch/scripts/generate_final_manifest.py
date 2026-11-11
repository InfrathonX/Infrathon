#!/usr/bin/env python3
"""Convert a frozen suite plan into the scored manifests of standalone K-grouped wgrad v3.

The v3 scored workload models one rank of a mixture-of-experts training step
(the weight gradient of the expert linears) in four families: large batch,
many groups, small batch and sequence.  The plan is a frozen JSON file with a
``public`` and a ``judge`` case list (see the scale-up notes); this tool only
converts it, so the frozen plan stays the single source of truth.

Outputs:

* the root-only Judge manifest (``--private-output``), suite ``final-holdout-v3``;
  it must never enter the repository or any Work build context;
* ``images/work/benchmark/public_cases.json`` (``--write-public``), the public
  cases shipped to Work;
* the ``cases`` lists of ``portable/{public,interactive,final}-suite-manifest.json``
  (``--write-portable``): full public cases, and only opaque case IDs and
  families for the hidden suite.

Judge seeds are offset to at least ``FINAL_SEED_MINIMUM`` and public seeds are
renumbered below it, so the two seed sets are structurally disjoint.

    scripts/generate_final_manifest.py --plan PLAN.json --private-output OUT.json
    scripts/generate_final_manifest.py --plan PLAN.json --write-public --write-portable
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


TASK_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 4
SUITE_ID = "final-holdout-v3"
PUBLIC_SUITE_ID = "public-v3"
TASK_ID = "standalone_k_grouped_fp8_wgrad_h100_scratch_v2"
PUBLIC_PATH = TASK_ROOT / "images/work/benchmark/public_cases.json"
PORTABLE = TASK_ROOT / "portable"
FAMILIES = ("large", "many_groups", "small_batch", "sequence")
FINAL_CASE_COUNT = 1082
PUBLIC_CASE_COUNT = 123
FINAL_SEED_MINIMUM = 1 << 32
PUBLIC_SEED_BASE = 1000
PUBLIC_SEED_STRIDE = 101
JUDGE_TRIALS, JUDGE_ROUNDS, PUBLIC_TRIALS, PUBLIC_ROUNDS = 3, 3, 3, 1
# The legacy public grid (former public cases plus the former scored stress
# shape) keeps the uniform per-group draw and is filed by its mean K per group.
LEGACY_LARGE_MIN_K = 1024


def convert_case(raw: Mapping[str, Any], *, case_id: str, seed: int) -> dict[str, Any]:
    family = str(raw["family"])
    if family == "legacy":
        g, k_per_group = int(raw["g"]), int(raw["k_per_group"])
        return {
            "case_id": case_id,
            "family": "large" if k_per_group >= LEGACY_LARGE_MIN_K else "small_batch",
            "model": "legacy-grid",
            "g": g,
            "m": int(raw["m"]),
            "n": int(raw["n"]),
            "m2": None,
            "n2": None,
            "dist": "uniform",
            "total_k": g * k_per_group,
            "alpha": None,
            "hint_scale": 1.0,
            "seed": int(seed),
        }
    if family not in FAMILIES:
        raise SystemExit(f"unknown plan family {family!r}")
    sequence = family == "sequence"
    return {
        "case_id": case_id,
        "family": family,
        "model": str(raw["model"]),
        "g": int(raw["g"]),
        "m": int(raw["l1"][0]) if sequence else int(raw["m"]),
        "n": int(raw["l1"][1]) if sequence else int(raw["n"]),
        "m2": int(raw["l2"][0]) if sequence else None,
        "n2": int(raw["l2"][1]) if sequence else None,
        "dist": "dirichlet",
        "total_k": int(raw["total"]),
        "alpha": float(raw["alpha"]),
        "hint_scale": float(raw.get("hint_scale", 1.0)),
        "seed": int(seed),
    }


def convert_plan(plan: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    judge_raw, public_raw = plan.get("judge"), plan.get("public")
    if not isinstance(judge_raw, list) or not isinstance(public_raw, list):
        raise SystemExit("the plan needs judge and public case lists")
    if len(judge_raw) != FINAL_CASE_COUNT or len(public_raw) != PUBLIC_CASE_COUNT:
        raise SystemExit(f"the plan must hold {FINAL_CASE_COUNT} judge and {PUBLIC_CASE_COUNT} public cases")
    judge = []
    for index, raw in enumerate(judge_raw):
        seed = FINAL_SEED_MINIMUM + int(raw["seed"])
        judge.append(convert_case(raw, case_id=f"final-{index:04d}-{raw['family']}", seed=seed))
    public = []
    for index, raw in enumerate(public_raw):
        case = convert_case(raw, case_id="pending", seed=PUBLIC_SEED_BASE + PUBLIC_SEED_STRIDE * index)
        case["case_id"] = f"public-{index:03d}-{case['family']}"
        public.append(case)
    seeds = [case["seed"] for case in judge]
    if len(set(seeds)) != len(seeds):
        raise SystemExit("judge seeds must be unique")
    shapes = lambda cases: {  # noqa: E731
        shape for case in cases for shape in ((case["m"], case["n"]), (case["m2"], case["n2"])) if shape[0]
    }
    overlap = shapes(judge) & shapes(public)
    if overlap:
        raise SystemExit(f"public and Judge share (M, N) shapes: {sorted(overlap)}")
    return judge, public


def _encoded(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _atomic_write_private(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace existing manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{os.urandom(4).hex()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(_encoded(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def final_manifest(judge: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "suite_id": SUITE_ID, "cases": [dict(case) for case in judge]}


def public_manifest(public: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "suite_id": PUBLIC_SUITE_ID, "cases": [dict(case) for case in public]}


def _case_line(case: Mapping[str, Any]) -> str:
    return "    " + json.dumps(dict(case), sort_keys=True, allow_nan=False)


def _replace_case_list(path: Path, lines: Sequence[str]) -> None:
    text = path.read_text(encoding="utf-8")
    match = re.search(r'\n  "cases": \[\n(?:.*\n)*?  \]\n\}\n?$', text)
    if match is None:
        raise SystemExit(f"{path.name}: could not locate the trailing cases list")
    body = ",\n".join(lines)
    path.write_text(text[: match.start()] + '\n  "cases": [\n' + body + "\n  ]\n}\n", encoding="utf-8")


def _weight(cases: Sequence[Mapping[str, Any]], case: Mapping[str, Any]) -> dict[str, int]:
    """Geometric weight of one case: a family is 1/F of the score, a model 1/M of
    its family, and a case 1/C of its model's cases in that family."""
    families = sorted({item["family"] for item in cases})
    models = sorted({item["model"] for item in cases if item["family"] == case["family"]})
    count = sum(1 for item in cases if item["family"] == case["family"] and item["model"] == case["model"])
    return {"numerator": 1, "denominator": len(families) * len(models) * count}


def write_portable(judge: Sequence[Mapping[str, Any]], public: Sequence[Mapping[str, Any]]) -> None:
    hidden = [
        {
            "case_id": case["case_id"],
            "family": case["family"],
            "role": "performance",
            "seed": "frozen_root_only",
            "shape": "frozen_root_only",
            "trials": JUDGE_ROUNDS if case["family"] == "sequence" else JUDGE_TRIALS,
            "measurement": "rounds" if case["family"] == "sequence" else "trials",
            "weight": _weight(judge, case),
        }
        for case in judge
    ]
    for name in ("final-suite-manifest.json", "interactive-suite-manifest.json"):
        _replace_case_list(PORTABLE / name, [_case_line(case) for case in hidden])
    shown = [
        dict(case, role="performance",
             trials=PUBLIC_ROUNDS if case["family"] == "sequence" else PUBLIC_TRIALS,
             measurement="rounds" if case["family"] == "sequence" else "trials",
             weight=_weight(public, case))
        for case in public
    ]
    _replace_case_list(PORTABLE / "public-suite-manifest.json", [_case_line(case) for case in shown])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True, help="frozen suite plan JSON")
    parser.add_argument("--private-output", type=Path, help="root-only Judge manifest to create")
    parser.add_argument("--write-public", action="store_true", help=f"rewrite {PUBLIC_PATH.name}")
    parser.add_argument("--write-portable", action="store_true", help="rewrite the portable case lists")
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    judge, public = convert_plan(plan)
    report: dict[str, Any] = {"judge_cases": len(judge), "public_cases": len(public)}
    if args.private_output is not None:
        output = args.private_output.absolute()
        _atomic_write_private(output, final_manifest(judge))
        report["private_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
        report["private_size"] = output.stat().st_size
    if args.write_public:
        PUBLIC_PATH.write_bytes(_encoded(public_manifest(public)))
        report["public_sha256"] = hashlib.sha256(PUBLIC_PATH.read_bytes()).hexdigest()
    if args.write_portable:
        write_portable(judge, public)
        report["portable"] = "rewritten"
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
