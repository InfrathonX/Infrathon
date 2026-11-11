#!/usr/bin/env python3
"""Public tests for the TileLang Mamba-2 SSD forward task, v3.

Candidate-visible feedback only.  Every run applies the runtime TileLang-only
operator record, so a call that does compute in host torch fails here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

BENCHMARK_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCHMARK_ROOT))

import mamba2_ssd_runtime as runtime
from source_admission import validate_submission

QUICK_CASES = ("public-p1", "public-p20")
SMOKE_CASE = "public-p1"


def _select_cases(suite: str, case: str | None, family: str | None) -> list[dict[str, Any]]:
    """`one` is a single case, `quick` two cases, `full` all of them; `--case` and
    `--family` narrow the selection."""
    cases = runtime.public_suite()
    if case is not None:
        wanted = case.lower()
        wanted = wanted if wanted.startswith("public-") else f"public-{wanted}"
        selected = [row for row in cases if row["case_id"] == wanted]
        if not selected:
            raise SystemExit(f"unknown public case {case!r}")
        return selected
    if family is not None:
        return [row for row in cases if row["family"] == family]
    if suite == "full":
        return cases
    if suite == "quick":
        return [row for row in cases if row["case_id"] in QUICK_CASES]
    if suite == "one":
        return [row for row in cases if row["case_id"] == SMOKE_CASE]
    raise ValueError(f"unknown suite {suite!r}")


def _print_admission(report: Mapping[str, Any]) -> None:
    for warning in report.get("warnings", ()):
        print(f"  warning: {warning}")
    for error in report.get("errors", ()):
        print(f"  error:   {error}")


def _print_edge(rows: Sequence[Mapping[str, Any]]) -> None:
    print()
    print("Edge fixtures (correctness only, never scored)")
    print(f"  {'fixture':34s} {'geometry':8s} {'batch':>5s} {'L':>6s} {'rel L2':>10s} {'max|err|/scale':>15s}")
    for row in rows:
        print(
            f"  {row['case_id']:34s} {row['geometry']:8s} {row['batch']:5d} {row['seqlen']:6d} "
            f"{row['rel_l2']:10.3e} {row['max_abs_scaled']:15.3e}"
        )


def _print_cases(rows: Sequence[Mapping[str, Any]]) -> None:
    print()
    print("Public cases")
    print(
        f"  {'case':12s} {'family':8s} {'batch':>5s} {'L':>7s} {'H':>4s} {'P':>4s} {'N':>4s} {'G':>3s} "
        f"{'ms':>9s} {'TFLOPS':>9s} {'rel L2':>10s} {'warm s':>8s} {'ws GiB':>7s}"
    )
    for row in rows:
        # A captured layer is judged on its own trial; the warm-up ran on generated tensors.
        value = row.get("real_rel_l2", row.get("rel_l2"))
        rel = f"{value:10.3e}" if value is not None else f"{'unchecked':>10s}"
        print(
            f"  {row['case_id']:12s} {row['family']:8s} {row['batch']:5d} {row['seqlen']:7d} "
            f"{row['heads']:4d} {row['headdim']:4d} {row['dstate']:4d} {row['groups']:3d} "
            f"{row['median_seconds'] * 1e3:9.3f} {row['tflops']:9.2f} {rel} "
            f"{row['warmup_wall_seconds']:8.1f} {row['workspace_bytes'] / 2**30:7.2f}"
        )
    ops = sorted({op for row in rows for op in row.get("torch_ops", [])})
    print()
    print("TileLang-only operator record: passed (host torch dispatched only allocations and views)")
    print(f"  torch operators seen: {', '.join(ops) if ops else 'none'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lab-root", "--workdir", dest="lab_root", type=Path,
        default=Path("/app/workdir/mamba2_ssd_lab"),
    )
    parser.add_argument("--candidate-version", type=int)
    parser.add_argument(
        "--suite", choices=("one", "quick", "full"), default="full",
        help="one: a single smoke case; quick: two cases; full: every public case",
    )
    parser.add_argument("--quick", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case", help="run one public case, e.g. P7")
    parser.add_argument("--family", choices=runtime.SCORE_FAMILIES, help="run one workload family")
    parser.add_argument(
        "--perf-only", action="store_true",
        help="time only: skip the FP32 reference and the comparison (finiteness is still checked)",
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--skip-edge-fixtures", action="store_true")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args(argv)

    print("Source admission")
    suite = "quick" if args.quick else args.suite
    report = validate_submission(args.lab_root, args.candidate_version, workspace_mode=True)
    _print_admission(report)
    if not report["ok"]:
        print("\nadmission failed; nothing was run")
        return 2
    version = report["candidate_version"]
    print(f"  admitted candidates/kernel_v{version:03d} ({report['selected_sha256'][:16]}...)")

    import torch

    cases = _select_cases(suite, args.case, args.family)
    full = suite == "full" and args.case is None and args.family is None
    try:
        result = runtime.run_suite(
            torch,
            Path(report["selected_path"]),
            cases,
            suite_id=runtime.PUBLIC_SUITE_ID,
            trials=args.trials,
            warmups=args.warmups,
            with_edge_fixtures=full and not args.skip_edge_fixtures and not args.perf_only,
            require_full_suite=full,
            real_layer_root=runtime.PUBLIC_LAYER_ROOT,
            perf_only=args.perf_only,
        )
    except Exception as error:
        gate = getattr(error, "gate", None)
        prefix = f"FAILED ({gate})" if gate else "FAILED"
        print(f"\n{prefix}: {type(error).__name__}: {error}")
        return 1

    if result["edge_fixtures"]:
        _print_edge(result["edge_fixtures"])
    _print_cases(result["cases"])
    print()
    for family, value in result["family_tflops"].items():
        print(f"  {family:14s} {value:8.2f} TFLOPS")
    label = "public aggregate" if full else "aggregate of the selected cases"
    print(f"  {label:16s} {result['score_tflops']:8.2f} TFLOPS")
    if args.perf_only:
        print("  (--perf-only: no output was compared against the reference)")
    print()

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
