#!/usr/bin/env python3
"""Public tests for the paged-KV block transfer task."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

BENCHMARK_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCHMARK_ROOT))

import kv_transfer_runtime as runtime
from source_admission import validate_submission

SMOKE_CASE = "public-mistral-nemo-12b-tp8-fused-synt-device-large-0"
QUICK_CASES = (
    SMOKE_CASE,
    "public-mistral-nemo-12b-tp2-split-synt-device-small-0",
    "public-mistral-nemo-12b-tp8-fused-synt-host-large-0",
    "public-qwen3-8b-tp4-fused-synt-host-small-0",
    "public-mistral-nemo-12b-tp8-fused-synt-host-duplex-0",
    "public-qwen3-8b-tp8-fused-synt-trace-0",
    "public-mistral-nemo-12b-tp8-fused-synt-overlap-0",
)


def _select_cases(suite: str) -> list[dict[str, Any]]:
    cases = runtime.public_suite()
    if suite == "full":
        return cases
    if suite == "quick":
        return [case for case in cases if case["case_id"] in QUICK_CASES]
    if suite == "one":
        return [case for case in cases if case["case_id"] == SMOKE_CASE]
    raise ValueError(f"unknown suite {suite!r}")


def _print_admission(report: Mapping[str, Any]) -> None:
    for warning in report.get("warnings", ()):
        print(f"  warning: {warning}")
    for error in report.get("errors", ()):
        print(f"  error:   {error}")


def _print_edge(rows: Sequence[Mapping[str, Any]]) -> None:
    print()
    print("Public edge cases (bit-exact correctness only)")
    for row in rows:
        print(f"  {row['case_id']:34s} arm={row['arm']:7s} waves={row['waves']}")


def _print_cases(rows: Sequence[Mapping[str, Any]]) -> None:
    print()
    print("Public cases (bit-exact, every wave of every trial verified)")
    print(
        f"  {'case':62s} {'family':13s} {'waves':>5s} {'MiB':>8s} "
        f"{'median ms':>10s} {'GB/s':>9s} {'score':>9s}"
    )
    for row in rows:
        extra = ""
        if row.get("fg_loss_median") is not None:
            extra = f"  foreground loss {row['fg_loss_median']:.2f}"
        print(
            f"  {row['case_id']:62s} {row['family']:13s} {row['waves']:5d} "
            f"{row['logical_bytes'] / 2**20:8.1f} "
            f"{row['median_seconds'] * 1e3:10.3f} {row['gbps']:9.2f} "
            f"{row['case_score']:9.2f}{extra}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lab-root", "--workdir", dest="lab_root", type=Path,
        default=Path("/app/workdir/kv_transfer_lab"),
    )
    parser.add_argument("--candidate-version", type=int)
    parser.add_argument(
        "--suite", choices=("one", "quick", "full"), default="full",
        help="one: one case; quick: one case per family; full: every public case",
    )
    parser.add_argument("--quick", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--trials", type=int, default=3,
        help="trials per case",
    )
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

    print("\nFrozen compilation")
    build_dir = args.lab_root / "build" / f"kernel_v{version:03d}"
    try:
        build_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        import tempfile

        build_dir = Path(tempfile.mkdtemp(prefix="kv-transfer-build-"))
    import torch

    try:
        library, compile_log = runtime.compile_candidate(
            Path(report["selected_path"]),
            build_dir,
            BENCHMARK_ROOT,
            arch=runtime.native_arch(torch),
        )
    except Exception as error:
        print(f"\nCOMPILE FAILED: {error}")
        return 1
    print(f"  built {library}")
    binary = runtime.inspect_binary(library)
    if not binary["ok"]:
        print("\nBINARY AUDIT FAILED:")
        for problem in binary["problems"]:
            print(f"  {problem}")
        return 1
    print("  undefined-symbol audit passed")

    if not 1 <= args.trials <= 32:
        print("trials must be in [1, 32]")
        return 2
    cases = [
        {**case, "trials": args.trials, "scored": True}
        for case in _select_cases(suite)
    ]
    cases = [runtime.validate_case(case, scored=True) for case in cases]
    try:
        result = runtime.run_suite(
            torch,
            library,
            cases,
            suite_id=runtime.PUBLIC_SUITE_ID,
            with_edge_fixtures=not args.skip_edge_fixtures,
            require_full_groups=suite == "full",
        )
    except Exception as error:
        print(f"\nFAILED: {type(error).__name__}: {error}")
        return 1

    if result["edge_fixtures"]:
        _print_edge(result["edge_fixtures"])
    _print_cases(result["cases"])
    print()
    for family, value in result["family_scores"].items():
        print(f"  {family:13s} {value:9.2f}")
    for group, value in result["group_scores"].items():
        weight = runtime.GROUP_WEIGHTS.get(group)
        print(f"  {group:13s} {value:9.2f}  (weight {weight:.2f})")
    covered = len(result["family_scores"])
    total = len(runtime.FAMILIES)
    label = "public aggregate" if covered == total else f"partial ({covered}/{total} families)"
    print(f"  {label:16s} {result['score_gbps']:9.2f}")

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
