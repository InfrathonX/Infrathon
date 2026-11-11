#!/usr/bin/env python3
"""Candidate-only public feedback for the standalone K-grouped FP8 wgrad v3 task.

Suites:
  one       one public case (fast compile/launch check)
  quick     two public cases (one large batch, one small batch)
  full      all public cases (3 trials; sequence cases 1 round)
  family    the public cases of one family (--family)
  edge      the public edge cases (correctness, guard bands, unchanged inputs)
  sanitize  the public sanitizer subset under compute-sanitizer memcheck

Output carries only public cases and their results; hidden cases are never run
here.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

from source_admission import validate_submission


BENCHMARK_ROOT = Path(__file__).resolve().parent
PUBLIC_EDGE_MANIFEST = BENCHMARK_ROOT / "public_edge_cases.json"
PUBLIC_CASE_MANIFEST = BENCHMARK_ROOT / "public_cases.json"
SANITIZER = Path("/usr/local/cuda/compute-sanitizer/compute-sanitizer")
SANITIZER_EXIT_CODE = 86
SANITIZER_ARGUMENTS = (
    "--tool=memcheck",
    f"--error-exitcode={SANITIZER_EXIT_CODE}",
    "--target-processes=application-only",
    "--destroy-on-device-error=context",
    "--force-blocking-launches",
    "--check-api-memory-access=no",
    "--print-limit=20",
    "--kernel-name-exclude",
    "kns=nvjet",
)

# compute-sanitizer prints this when it cannot attach to the CUDA process; the
# target then keeps running without instrumentation, so stop the whole tree.
# Seen on platforms whose GPU virtualization hook is preloaded into every process.
SANITIZER_UNAVAILABLE_MARKER = "No attachable process found"
SANITIZE_TIMEOUT_SECONDS = 600
EXIT_SANITIZER_UNAVAILABLE = 2

FAMILIES = ("large", "many_groups", "small_batch", "sequence")
FULL_TRIALS, FULL_ROUNDS = 3, 1


def _public_cases() -> tuple[dict[str, Any], ...]:
    payload = json.loads(PUBLIC_CASE_MANIFEST.read_text(encoding="utf-8"))
    return tuple(dict(case) for case in payload["cases"])


PUBLIC_CASES = _public_cases()


def _first(predicate) -> str:
    return next(case["case_id"] for case in PUBLIC_CASES if predicate(case))


# A small-batch case and a skewed large-batch case.
ONE_CASE_ID = _first(lambda c: c["family"] == "small_batch" and c["dist"] == "dirichlet")
QUICK_CASE_IDS = (ONE_CASE_ID, _first(lambda c: c["family"] == "large" and c["dist"] == "dirichlet"))


def cases_for_suite(name: str, family: str | None = None) -> list[dict[str, Any]]:
    by_id = {case["case_id"]: case for case in PUBLIC_CASES}
    if name == "one":
        return [dict(by_id[ONE_CASE_ID])]
    if name == "quick":
        return [dict(by_id[case_id]) for case_id in QUICK_CASE_IDS]
    if name == "full":
        return [dict(case) for case in PUBLIC_CASES]
    if name == "family":
        if family not in FAMILIES:
            raise ValueError(f"--family must be one of {FAMILIES}")
        return [dict(case) for case in PUBLIC_CASES if case["family"] == family]
    raise ValueError(f"unknown public suite: {name}")


def public_edge_cases(*, sanitize_only: bool = False) -> list[dict[str, Any]]:
    payload = json.loads(PUBLIC_EDGE_MANIFEST.read_text(encoding="utf-8"))
    cases = list(payload["cases"])
    return [case for case in cases if case["sanitize"]] if sanitize_only else cases


def _load_trusted_runtime() -> Any:
    """Load the public runtime, then remove its ordinary candidate import route."""
    import k_grouped_wgrad_runtime as runtime

    sys.path = [
        entry for entry in sys.path if Path(entry or ".").resolve() != BENCHMARK_ROOT
    ]
    sys.modules.pop("k_grouped_wgrad_runtime", None)
    return runtime


def _print_header(status: str) -> None:
    print(f"STATUS                    {status}")
    print("CANDIDATE_ONLY            YES")
    print("HIDDEN_CASES              NOT_RUN")


def _print_scored(payload: dict[str, Any]) -> None:
    print(
        "CASE                          FAMILY        G     M     N  TOTAL_K  HINT     "
        "MEDIAN_US     TFLOPS   EXACT_ERR  SOURCE_ERR"
    )
    for case in payload["cases"]:
        shape = f"{case['m']:>5} {case['n']:>5}"
        print(
            f"{case['case_id']:<29} {case['family']:<12} {case['g']:>3} {shape} "
            f"{case['total_k']:>8} {case['expected_k_hint']:>5} "
            f"{case['median_us']:>12.2f} {case['tflops']:>10.3f} "
            f"{case['max_exact_error']:>11.3e} {case['max_source_error']:>11.3e}"
        )
    print()
    for family, value in payload.get("family_tflops", {}).items():
        print(f"FAMILY_{family.upper()}_TFLOPS  {value:.6f}")
    if "primary_tflops" in payload:
        print(f"PUBLIC_PRIMARY_TFLOPS    {payload['primary_tflops']:.6f}")


def _print_edge(payload: dict[str, Any]) -> None:
    edge = payload["edge"]
    print(
        f"EDGE_CASES               {edge['cases_passed']} passed, "
        f"{edge['cases_failed']} failed, {edge['cases_total']} total"
    )
    for row in edge["results"]:
        if not row["ok"]:
            print(f"EDGE_FAIL  {row['case_id']}  [{row['gate']}]  {row['error']}")


def _kill_session(session_id: int) -> None:
    """SIGKILL every process in the sanitizer's session (the launcher and its target)."""
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            fields = open(f"/proc/{entry}/stat", encoding="utf-8").read().rsplit(")", 1)[1].split()
            if int(fields[3]) == session_id:
                os.kill(int(entry), signal.SIGKILL)
        except (OSError, ValueError, IndexError):
            continue


def _run_sanitize_outer(args: argparse.Namespace) -> int:
    if not SANITIZER.is_file():
        print("SANITIZER_ERROR: compute-sanitizer is not installed", file=sys.stderr)
        return 1
    env = dict(os.environ)
    env["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    command = [
        str(SANITIZER),
        *SANITIZER_ARGUMENTS,
        sys.executable,
        str(Path(__file__).resolve()),
        "--lab-root",
        str(args.lab_root),
        "--suite",
        "sanitize-inner",
        "--candidate-version",
        str(args.candidate_version),
    ]
    outcome = "finished"
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(
            command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        deadline = time.monotonic() + SANITIZE_TIMEOUT_SECONDS
        while process.poll() is None:
            time.sleep(1.0)
            log.seek(0)
            if SANITIZER_UNAVAILABLE_MARKER in log.read().decode("utf-8", errors="replace"):
                outcome = "unavailable"
                break
            if time.monotonic() > deadline:
                outcome = "timeout"
                break
        if process.poll() is None:
            _kill_session(process.pid)
            process.wait()
        log.seek(0)
        output = log.read().decode("utf-8", errors="replace")
    lines = output.splitlines()
    report = [line for line in lines if line.startswith("=========")]
    if outcome == "unavailable":
        _print_header("candidate-only; compute-sanitizer unavailable on this machine")
        print("MEMCHECK                  UNAVAILABLE (compute-sanitizer could not attach to the CUDA process)")
        return EXIT_SANITIZER_UNAVAILABLE
    if outcome == "timeout":
        _print_header("candidate-only; sanitizer subset timed out")
        print(f"MEMCHECK                  TIMEOUT after {SANITIZE_TIMEOUT_SECONDS} s")
        for line in report[:60]:
            print(line)
        return 1
    if process.returncode == 0:
        _print_header("candidate-only; sanitizer subset clean")
        print("MEMCHECK                  PASS")
        print("\n".join(line for line in lines if line.startswith("SANITIZE_CASE")))
        return 0
    _print_header("candidate-only; sanitizer subset failed")
    invalid_access = any("Invalid __" in line or "Out-of-range" in line for line in report)
    memcheck_failed = process.returncode == SANITIZER_EXIT_CODE or invalid_access
    print("MEMCHECK                  " + ("FAIL" if memcheck_failed else "ERROR (the run failed before memcheck found an invalid access)"))
    for line in report[:60]:
        print(line)
    if not memcheck_failed:
        print(output[-4000:], file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lab-root", type=Path, default=Path("/app/workdir/k_grouped_wgrad_lab"))
    parser.add_argument(
        "--suite",
        choices=("one", "quick", "full", "family", "edge", "sanitize", "sanitize-inner"),
        default="full",
    )
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--candidate-version", type=int, required=True)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    stage = "source admission"
    try:
        admission = validate_submission(args.lab_root, args.candidate_version, workspace_mode=True)
        if not admission["ok"]:
            for error in admission["errors"]:
                print(f"ADMISSION_ERROR: {error}", file=sys.stderr)
            raise RuntimeError("public source admission failed")
        print("SOURCE_ADMISSION          PASS")
        candidate = Path(str(admission["selected_path"]))
        if args.suite == "sanitize":
            return _run_sanitize_outer(args)
        stage = "candidate import, JIT, launch, or public comparison"
        runtime = _load_trusted_runtime()
        globals().pop("_load_trusted_runtime", None)
        if args.suite == "sanitize-inner":
            result = runtime.run_sanitize_cases(candidate, public_edge_cases(sanitize_only=True))
            for row in result["cases"]:
                print(
                    f"SANITIZE_CASE {row['case_id']} element_ratio={row['max_element_ratio']:.3e} "
                    f"exact_error={row['max_exact_error']:.3e}"
                )
            return 0
        if args.suite == "edge":
            result = runtime.run_suite(
                candidate, [], public_edge_cases(), trials=1, warmups=1,
                stop_on_edge_failure=False,
            )
        else:
            full = args.suite in ("full", "family")
            result = runtime.run_suite(
                candidate, cases_for_suite(args.suite, args.family), [],
                trials=FULL_TRIALS if full else 1, warmups=1, rounds=FULL_ROUNDS,
            )
        payload = {
            **result,
            "suite": args.suite,
            "candidate_version": args.candidate_version,
            "num_candidate_versions": admission["num_candidate_versions"],
            "candidate_only": True,
            "hidden_cases_run": False,
        }
        if args.json_output:
            args.json_output.write_text(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n",
                encoding="utf-8",
            )
        if result.get("status") == "passed":
            _print_header("candidate-only; all public gates passed")
        else:
            _print_header(f"candidate-only; FAILED {result.get('failed_gate')}")
            print(f"PUBLIC_FAILURE: {result.get('private_error')}")
        if args.suite == "edge":
            _print_edge(payload)
        elif payload.get("cases"):
            _print_scored(payload)
        return 0 if result.get("status") == "passed" else 1
    except Exception as exc:
        _print_header("candidate-only; FAILED")
        print(f"PUBLIC_FAILURE_STAGE: {stage}", file=sys.stderr)
        print(f"PUBLIC_FAILURE: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
