#!/usr/bin/env python3
"""Rewrite the portable suite documents of TileLang SSD v3 from the executable sources.

    scripts/write_portable_suites.py <frozen final_cases.json>

It rewrites ``portable/{public,interactive,final}-suite-manifest.json``,
``portable/selection-protocol.json`` and the suite, metric and definition parts
of the task JSON's ``portable`` block.  Public rows come from the Work runtime;
final rows carry each case's identity, family, tier and weight but withhold its
geometry, shape, seed and checked trial, which stay in the Judge-only frozen
instance.  Run ``scripts/refresh_portable_manifests.py`` afterwards to rebind
the digests.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parents[1]
PORTABLE = TASK_ROOT / "portable"
TASK_JSON = TASK_ROOT / "tasks/tilelang_mamba2_ssd_forward_h100_seed_v2.json"
FROZEN = "judge/private/final_cases.json"
TRIALS = 3
WARMUPS = 1
QUICK_CASES = ("public-p1", "public-p20")


def _load(name: str, relative: str):
    path = TASK_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


def _weight(family: str, counts: dict[str, int], families: int) -> dict[str, int]:
    share = Fraction(1, families * counts[family])
    return {"numerator": share.numerator, "denominator": share.denominator}


def _edge_rows(runtime) -> list[dict]:
    rows = []
    for fixture in runtime.edge_fixtures():
        geometry = runtime.GEOMETRIES[fixture["geometry"]]
        row = {key: value for key, value in fixture.items() if key not in ("scored", "check_trial", "layer", "tier")}
        row.update(geometry)
        row.update(
            role="correctness", measurement_mode="correctness", timing="not_measured",
            weight={"numerator": 0, "denominator": 1},
        )
        rows.append(dict(sorted(row.items())))
    return rows


def _dump(path: Path, payload: dict) -> None:
    """One case per line, everything else indented, keys sorted."""
    cases = payload.get("cases")
    body = dict(payload)
    if cases is not None:
        body["cases"] = "__CASES__"
    text = json.dumps(body, indent=2, sort_keys=True)
    if cases is not None:
        lines = ",\n".join("    " + json.dumps(case, sort_keys=True) for case in cases)
        text = text.replace('"__CASES__"', "[\n" + lines + "\n  ]")
    path.write_text(text + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    frozen_path = Path(argv[0])
    frozen_bytes = frozen_path.read_bytes()
    frozen = json.loads(frozen_bytes)
    runtime = _load("portable_writer_runtime", "runtime/mamba2_ssd_runtime.py")
    work = _load("portable_writer_work", "images/work/benchmark/mamba2_ssd_runtime.py")
    cases = runtime.validate_manifest_cases(frozen["cases"])
    families = runtime.SCORE_FAMILIES

    # ---- public --------------------------------------------------------------------
    public = work.public_suite()
    public_rows = _edge_rows(work)
    for case in public:
        geometry = work.GEOMETRIES[case["geometry"]]
        row = {key: value for key, value in case.items() if key not in ("scored",)}
        row.update(geometry)
        row.update(
            role="performance",
            measurement_mode="real_capture_single_trial" if case["family"] == "real" else "fresh_inputs_warmup_precall",
            quick_suite=case["case_id"] in QUICK_CASES,
            trials={"full": TRIALS, "quick": TRIALS} if case["family"] != "real" else {"full": 1, "quick": 1},
            warmups=WARMUPS,
            weight=_weight(case["family"], work.FINAL_FAMILY_COUNTS, len(families)),
        )
        public_rows.append(dict(sorted(row.items())))
    public_doc = json.loads((PORTABLE / "public-suite-manifest.json").read_text(encoding="utf-8"))
    public_doc.update(
        suite_id=work.PUBLIC_SUITE_ID,
        cases=public_rows,
        case_counts={
            "total": len(public_rows),
            "scored_performance": len(public),
            "zero_weight_hard_gate": len(public_rows) - len(public),
        },
        selection_semantics=(
            f"the full suite runs the {len(work.EDGE_FIXTURES)} public edge fixtures first and all must pass, "
            f"then the {len(public)} public cases in the four workload families; quick runs P1 and P20 and one "
            "runs P1 without the edge fixtures; --case and --family select cases and --perf-only skips the "
            "reference; the public runner defaults to 3 trials and 1 warm-up (the real layer is timed once) "
            "and applies the TileLang-only operator record; public timing is candidate-only feedback and "
            "never enters a score"
        ),
    )
    _dump(PORTABLE / "public-suite-manifest.json", public_doc)

    # ---- final and interactive ----------------------------------------------------
    final_rows = _edge_rows(runtime)
    for case in cases:
        final_rows.append(
            {
                "case_id": case["case_id"],
                "family": case["family"],
                "tier": case["tier"],
                "geometry": "withheld_root_only",
                "shape": "withheld_root_only",
                "seed": "private_root_only",
                "checked_trial": "withheld_root_only",
                "regime": "mixed",
                "role": "performance",
                "measurement_mode": (
                    "real_capture_single_trial" if case["family"] == "real" else "fresh_inputs_decoy_warmup"
                ),
                "trials": runtime.REAL_TRIALS if case["family"] == "real" else TRIALS,
                "warmups": WARMUPS,
                "weight": _weight(case["family"], runtime.FINAL_FAMILY_COUNTS, len(families)),
            }
        )
    counts = {
        "total": len(final_rows),
        "scored_performance": len(cases),
        "zero_weight_hard_gate": len(final_rows) - len(cases),
    }
    frozen_record = {
        "path": FROZEN,
        "sha256": hashlib.sha256(frozen_bytes).hexdigest(),
        "provenance": (
            "v3 qualification draw 2026-11-07: the default seeded draw of scripts/generate_final_manifest.py "
            "(seed derived from the suite identity) over the frozen Judge-only suite plan"
        ),
    }
    for role in ("final", "interactive"):
        path = PORTABLE / f"{role}-suite-manifest.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc.update(suite_id=runtime.FINAL_SUITE_ID, cases=final_rows, case_counts=counts, frozen_instance=frozen_record)
        if role == "final":
            doc["budget_gates"] = {
                "warmup_wall_cap_seconds": str(runtime.WARMUP_WALL_CAP_SECONDS),
                "workspace_multiple": str(runtime.WORKSPACE_MULTIPLE),
                "workspace_cap_bytes": str(runtime.WORKSPACE_CAP_BYTES),
            }
            doc["draw_policy"]["redraw"] = (
                "scripts/generate_final_manifest.py --plan PLAN --redraw --seed <int> draws one reproducible "
                "63-bit seed and one checked trial per case from a seeded random.Random (default seed: "
                "int.from_bytes(sha256(\"<suite_id>|<task_id>\").digest()[:8], \"big\")); the cases come from "
                "the frozen Judge-only plan. A redraw is a NEW suite identity: frozen_instance must be "
                "re-recorded and no earlier score is comparable to it."
            )
            doc["draw_policy"]["shapes"] = (
                "withheld: 138 cases on 15 deployed geometries (including per-rank tensor-parallel layouts) "
                "in four workload families; serving lengths are real request lengths; only the families, "
                "tiers, counts and weights are disclosed here"
            )
            doc["selection_semantics"] = (
                f"all {counts['total']} cases execute; the {counts['zero_weight_hard_gate']} edge fixtures are "
                f"zero-weight hard gates; the {counts['scored_performance']} scored rows are geometrically "
                "averaged within each of the batch, serving, long and real families and the score is the "
                "equal-weight geometric mean of the four family means; every decoy and edge call runs under "
                "the TileLang-only operator record"
            )
        _dump(path, doc)

    # ---- selection protocol -------------------------------------------------------
    selection_path = PORTABLE / "selection-protocol.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["edge_fixtures"]["case_count"] = len(runtime.EDGE_FIXTURES)
    selection["family_weights"] = {family: "1/4" for family in families}
    for role in ("final", "interactive"):
        selection[role].update(
            case_count=len(cases), scored_case_count=len(cases),
            hard_gate_case_count=len(runtime.EDGE_FIXTURES), suite_id=runtime.FINAL_SUITE_ID,
            trials=TRIALS, real_trials=runtime.REAL_TRIALS, warmups=WARMUPS,
        )
    selection["public"] = {
        "full": {"case_count": len(public), "case_ids": "all_public_cases", "trials": TRIALS},
        "one": {"case_count": 1, "case_id": "public-p1"},
        "quick": {"case_count": len(QUICK_CASES), "case_ids": list(QUICK_CASES), "trials": TRIALS},
        "warmups": WARMUPS,
    }
    selection["historical_protocol"] = (
        "v1 scored three bands with slot-drawn holdout lengths on a partly synthetic geometry set; v2 scored "
        "the 18 public upstream benchmark shapes (batch 8 or 4, power-of-two lengths) on three deployed "
        "geometries; v3 scores 138 withheld cases on 15 deployed geometries in four workload families "
        "(batch, serving at batch 1, long context, captured real layers). v1, v2 and v3 scores are not comparable."
    )
    selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # ---- task JSON portable block -------------------------------------------------
    text = TASK_JSON.read_text(encoding="utf-8")
    task = json.loads(text)
    portable = task["portable"]
    definition = portable["task_core"]["definition"]
    for stale in ("families", "scored_lengths", "scored_batch", "min_scored_length"):
        definition.pop(stale, None)
    definition["geometries"] = {name: dict(value) for name, value in runtime.GEOMETRIES.items()}
    definition["workload_families"] = list(families)
    definition["final_family_counts"] = dict(runtime.FINAL_FAMILY_COUNTS)
    definition["max_batch"] = runtime.MAX_BATCH
    definition["min_long_length"] = runtime.MIN_LONG_LENGTH
    definition["max_tensor_elements"] = runtime.MAX_TENSOR_ELEMENTS
    invariants = portable["task_core"]["invariants"]
    portable["task_core"]["invariants"] = [
        line.replace("the 18 disclosed edge fixtures", f"the {len(runtime.EDGE_FIXTURES)} edge fixtures")
        for line in invariants
    ]
    roles = portable["evaluation_suites"]["tilelang-ssd-v2"]["roles"]
    roles["public"]["case_count"] = len(public_rows)
    roles["interactive"]["case_count"] = len(final_rows)
    roles["final"]["case_count"] = len(final_rows)
    metric = portable["metrics"]["family-geomean-effective-tflops"]["aggregation"]
    metric["protocol_id"] = "workload-family-geomean-v3"
    metric["spec"].update(
        formula="equal_weight_geometric_mean_over_workload_families_of_geometric_mean_over_cases",
        families=list(families),
        family_weights=["1/4"] * len(families),
    )
    TASK_JSON.write_text(json.dumps(task, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"public {len(public_rows)} rows, final {len(final_rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
