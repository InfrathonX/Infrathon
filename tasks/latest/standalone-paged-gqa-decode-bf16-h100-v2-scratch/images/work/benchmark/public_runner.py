#!/usr/bin/env python3
"""Candidate-only public feedback for the paged-GQA decode task.

Suites:
  one       one public decode case (fast compile/launch check)
  quick     one decode, one speculative and one long-context case, one trial each
  full      all public cases, three trials each
  decode | spec | long | stream
            the public cases of one family, three trials each
  edge      the public edge cases (correctness, guard bands, unchanged inputs)
  sanitize  the public sanitizer subset under compute-sanitizer memcheck

Output carries only disclosed geometry, correctness summaries, per-case edge
results and aggregates; hidden cases are never run here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

from source_admission import validate_submission


BENCHMARK_ROOT = Path(__file__).resolve().parent
PUBLIC_EDGE_MANIFEST = BENCHMARK_ROOT / "public_edge_cases.json"
PUBLIC_EDGE_SUITE_ID = "edge-public-v4"
PUBLIC_EDGE_SCHEMA_VERSION = 4
PUBLIC_CASES_MANIFEST = BENCHMARK_ROOT / "public_cases.json"
PUBLIC_CASES_SUITE_ID = "public-v4"
PUBLIC_CASES_SCHEMA_VERSION = 4
PUBLIC_PAYLOAD_SCHEMA_VERSION = 4
FAMILIES = ("decode", "spec", "long", "stream")

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

def _load_public_cases() -> tuple[dict[str, Any], ...]:
    payload = json.loads(PUBLIC_CASES_MANIFEST.read_text(encoding="utf-8"))
    if payload.get("suite_id") != PUBLIC_CASES_SUITE_ID:
        raise RuntimeError(f"public case manifest must be suite {PUBLIC_CASES_SUITE_ID}")
    if payload.get("schema_version") != PUBLIC_CASES_SCHEMA_VERSION:
        raise RuntimeError(f"public case manifest must be schema {PUBLIC_CASES_SCHEMA_VERSION}")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise RuntimeError("public case manifest has no cases")
    return tuple(dict(case) for case in cases)


PUBLIC_CASES = _load_public_cases()
ONE_CASE_ID = "public-0012-decode"
# A decode step with a group size of 5, a speculative case at S_q = 5 and a
# single long-context request of about 63K tokens.
QUICK_CASE_IDS = ("public-0012-decode", "public-0044-spec", "public-0084-long")

# Mirrors the runtime's GATE_ORDER (asserted by a task-local test).
PUBLIC_GATES = (
    "edge_correctness_ok",
    "memory_safety_ok",
    "correctness_ok",
)
PUBLIC_EDGE_FIELDS = (
    "case_id",
    "category",
    "s_q",
    "b",
    "l_max",
    "h_q",
    "h_kv",
    "page_size",
    "table_mode",
    "table_width",
    "pool_pages",
    "fill",
    "values",
    "sanitize",
)
PUBLIC_METRIC_FIELDS = (
    "out_max_abs",
    "out_max_abs_limit",
    "out_mean_abs",
    "out_mean_abs_limit",
    "out_norm_l2",
)
PUBLIC_CASE_FIELDS = (
    "case_id",
    "family",
    "s_q",
    "b",
    "l_max",
    "h_q",
    "h_kv",
    "page_size",
    "table_width",
    "pool_pages",
    "softmax_scale",
    "tflops",
    *PUBLIC_METRIC_FIELDS,
)
PUBLIC_CALL_FIELDS = ("median_us", "kv_gbps")
PUBLIC_STREAM_FIELDS = ("steps", "seconds", "max_step_us", "median_step_us")
PUBLIC_FLOAT_FIELDS = ("median_us", "kv_gbps", "seconds", "max_step_us", "median_step_us", "tflops",
                       "softmax_scale", *PUBLIC_METRIC_FIELDS)
MAX_PUBLIC_ERROR_CHARS = 600


def cases_for_suite(name: str) -> list[dict[str, Any]]:
    by_id = {case["case_id"]: case for case in PUBLIC_CASES}
    if name == "one":
        return [dict(by_id[ONE_CASE_ID])]
    if name == "quick":
        return [dict(by_id[case_id]) for case_id in QUICK_CASE_IDS]
    if name == "full":
        return [dict(case) for case in PUBLIC_CASES]
    if name in FAMILIES:
        return [dict(case) for case in PUBLIC_CASES if case["family"] == name]
    raise ValueError(f"unknown public suite: {name}")


def public_edge_cases(*, sanitize_only: bool = False) -> list[dict[str, Any]]:
    """The disclosed edge manifest; the runtime validates every field again."""
    payload = json.loads(PUBLIC_EDGE_MANIFEST.read_text(encoding="utf-8"))
    if payload.get("suite_id") != PUBLIC_EDGE_SUITE_ID:
        raise RuntimeError(f"public edge manifest must be suite {PUBLIC_EDGE_SUITE_ID}")
    if payload.get("schema_version") != PUBLIC_EDGE_SCHEMA_VERSION:
        raise RuntimeError(f"public edge manifest must be schema {PUBLIC_EDGE_SCHEMA_VERSION}")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise RuntimeError("public edge manifest has no cases")
    if any(case.get("native") is not True for case in cases):
        raise RuntimeError("every public edge case must run natively")
    return [dict(case) for case in cases if case["sanitize"] or not sanitize_only]


def _load_trusted_runtime() -> Any:
    """Load the public runtime, then remove its ordinary candidate import route."""
    import paged_gqa_decode_runtime as runtime

    sys.path = [
        entry for entry in sys.path if Path(entry or ".").resolve() != BENCHMARK_ROOT
    ]
    sys.modules.pop("paged_gqa_decode_runtime", None)
    return runtime


def _disclosed_fields(
    record: Mapping[str, Any], fields: Sequence[str]
) -> dict[str, Any]:
    missing = [field for field in fields if field not in record]
    if missing:
        raise RuntimeError(f"trusted public result is missing fields: {missing}")
    return {field: record[field] for field in fields}


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"public value {label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"public value {label} must be finite")
    return number


def _public_aggregate(
    value: object, *, label: str, allowed_keys: tuple[str, ...]
) -> dict[str, float]:
    if not isinstance(value, dict) or not value:
        raise RuntimeError(f"public aggregate {label} must be a nonempty object")
    if not set(value).issubset(allowed_keys):
        raise RuntimeError(f"public aggregate {label} contains an unknown key")
    result: dict[str, float] = {}
    for key, raw in value.items():
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise RuntimeError(f"public aggregate {label}.{key} must be numeric")
        number = float(raw)
        if not math.isfinite(number):
            raise RuntimeError(f"public aggregate {label}.{key} must be finite")
        result[str(key)] = number
    return result


def _public_gate(value: object) -> str | None:
    if value is None:
        return None
    if value not in PUBLIC_GATES:
        raise RuntimeError(f"trusted public result names an unknown gate: {value!r}")
    return str(value)


def _public_edge_row(row: Mapping[str, Any], case: Mapping[str, Any]) -> dict[str, Any]:
    """Disclosed manifest geometry plus this case's own pass/fail and metrics."""
    public = _disclosed_fields(case, PUBLIC_EDGE_FIELDS)
    public["ok"] = row.get("ok") is True
    if public["ok"]:
        for field in PUBLIC_METRIC_FIELDS:
            public[field] = _finite(row.get(field), f"{case['case_id']}.{field}")
        return public
    public["gate"] = _public_gate(row.get("gate"))
    public["error"] = str(row.get("error", ""))[:MAX_PUBLIC_ERROR_CHARS]
    detail = row.get("detail")
    kind = detail.get("kind") if isinstance(detail, Mapping) else None
    public["failure_kind"] = kind if isinstance(kind, str) else None
    return public


def _public_case(record: Mapping[str, Any]) -> dict[str, Any]:
    fields = (*PUBLIC_CASE_FIELDS,
              *(PUBLIC_STREAM_FIELDS if record.get("family") == "stream" else PUBLIC_CALL_FIELDS))
    return {
        field: (_finite(value, f"{record.get('case_id')}.{field}") if field in PUBLIC_FLOAT_FIELDS else value)
        for field, value in _disclosed_fields(record, fields).items()
    }


def _public_payload(
    result: Mapping[str, Any],
    *,
    suite: str,
    candidate_version: int,
    num_candidate_versions: int,
    edge_cases: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Allowlist the public result: no trial lists, seeds, hashes or internals."""
    passed = result.get("status") == "passed"
    gates = result.get("gates")
    gates = gates if isinstance(gates, Mapping) else {}
    manifest = {str(case["case_id"]): case for case in edge_cases}
    edge = result.get("edge") if isinstance(result.get("edge"), Mapping) else {}
    edge_rows = []
    for row in edge.get("results", []):
        case = manifest.get(str(row.get("case_id")))
        if case is None:
            raise RuntimeError("trusted public result reports an undisclosed edge case")
        edge_rows.append(_public_edge_row(row, case))
    payload: dict[str, Any] = {
        "schema_version": PUBLIC_PAYLOAD_SCHEMA_VERSION,
        "suite": suite,
        "candidate_version": candidate_version,
        "num_candidate_versions": num_candidate_versions,
        "candidate_only": True,
        "hidden_cases_run": False,
        "source_admission_ok": True,
        "status": "passed" if passed else "failed",
        "failed_gate": None if passed else _public_gate(result.get("failed_gate")),
        "failure": "" if passed else str(result.get("private_error", ""))[:MAX_PUBLIC_ERROR_CHARS],
        "gates": {
            gate: (None if gates.get(gate) is None else bool(gates.get(gate)))
            for gate in PUBLIC_GATES
        },
        "trials_per_case": int(result["trials_per_case"]),
        "warmups": int(result["warmups"]),
        "edge": {
            "cases_total": len(edge_cases),
            "cases_passed": sum(row["ok"] for row in edge_rows),
            "cases_failed": sum(not row["ok"] for row in edge_rows),
            "cases_not_run": len(edge_cases) - len(edge_rows),
            "results": edge_rows,
        },
        "cases": [_public_case(record) for record in result.get("cases", [])],
    }
    if "family_tflops" in result:
        payload["family_tflops"] = _public_aggregate(
            result["family_tflops"], label="family_tflops", allowed_keys=FAMILIES
        )
        payload["primary_tflops"] = _finite(result["primary_tflops"], "primary_tflops")
    return payload


def _print_header(status: str) -> None:
    print(f"STATUS                    {status}")
    print("CANDIDATE_ONLY            YES")
    print("HIDDEN_CASES              NOT_RUN")


def _gate_text(value: bool | None) -> str:
    return "-" if value is None else ("PASS" if value else "FAIL")


def _print_result(payload: Mapping[str, Any]) -> None:
    if payload["status"] == "passed":
        _print_header("candidate-only; all public gates passed")
    else:
        _print_header(f"candidate-only; FAILED {payload['failed_gate']}")
    for gate in PUBLIC_GATES:
        print(f"{gate.upper():<26}{_gate_text(payload['gates'][gate])}")
    if payload["failure"]:
        print(f"PUBLIC_FAILURE: {payload['failure']}")
    edge = payload["edge"]
    if edge["cases_total"]:
        print()
        print(
            f"EDGE_CASES                {edge['cases_passed']} passed, "
            f"{edge['cases_failed']} failed, {edge['cases_not_run']} not run, "
            f"{edge['cases_total']} total"
        )
        print(
            "EDGE_CASE                          CATEGORY           S_Q    B  L_MAX  HEADS"
            "   PS  TABLE      FILL   RESULT       NORM_L2"
        )
        for row in edge["results"]:
            heads = f"{row['h_q']}/{row['h_kv']}"
            result = "PASS" if row["ok"] else f"FAIL [{row['gate']}]"
            norm = f"{row['out_norm_l2']:>13.3e}" if row["ok"] else ""
            print(
                f"{row['case_id']:<34} {row['category']:<18} {row['s_q']:>3} "
                f"{row['b']:>4} {row['l_max']:>6} {heads:>6} {row['page_size']:>4}  "
                f"{row['table_mode']:<10} {row['fill']:<6} {result:<12}{norm}"
            )
        for row in edge["results"]:
            if not row["ok"]:
                print(f"EDGE_FAIL  {row['case_id']}  [{row['gate']}]  {row['error']}")
    if payload["cases"]:
        print()
        print(
            "CASE                FAMILY  S_Q    B    L_MAX  HEADS   PS      TIME_US     TFLOPS"
            "    KV_GB/S       NORM_L2"
        )
        for case in payload["cases"]:
            heads = f"{case['h_q']}/{case['h_kv']}"
            if case["family"] == "stream":
                time_us = case["seconds"] * 1.0e6
                gbps = "-"
            else:
                time_us = case["median_us"]
                gbps = f"{case['kv_gbps']:.1f}"
            print(
                f"{case['case_id']:<19} {case['family']:<7} {case['s_q']:>3} {case['b']:>4} "
                f"{case['l_max']:>8} {heads:>6} {case['page_size']:>4} "
                f"{time_us:>12.1f} {case['tflops']:>10.3f} {gbps:>10} "
                f"{case['out_norm_l2']:>13.3e}"
            )
    if "primary_tflops" in payload:
        print()
        for family, value in sorted(payload["family_tflops"].items()):
            print(f"FAMILY_{family.upper()}_TFLOPS  {value:.6f}")
        print(f"PUBLIC_PRIMARY_TFLOPS    {payload['primary_tflops']:.6f}")


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
    cases = [line for line in lines if line.startswith("SANITIZE_CASE")]
    failures = [line for line in lines if line.startswith("SANITIZE_FAIL")]
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
        print("\n".join(cases))
        return 0
    _print_header("candidate-only; sanitizer subset failed")
    invalid_access = any("Invalid __" in line or "Out-of-range" in line for line in report)
    if process.returncode == SANITIZER_EXIT_CODE or invalid_access:
        print("MEMCHECK                  FAIL")
        for line in report[:60]:
            print(line)
        print("\n".join(failures))
        return 1
    if failures:
        # Memcheck saw no invalid access; a case failed its correctness check.
        print("MEMCHECK                  no invalid access reported")
        print("SANITIZE_CASES            FAIL")
        print("\n".join([*cases, *failures]))
        return 1
    print("MEMCHECK                  ERROR (the run failed before memcheck found an invalid access)")
    for line in report[:60]:
        print(line)
    print(output[-4000:], file=sys.stderr)
    return 1


def _run_sanitize_inner(runtime: Any, candidate: Path, cases: Sequence[Mapping[str, Any]]) -> int:
    """Runs inside compute-sanitizer: plain allocations, one call per case."""
    result = runtime.run_sanitize_cases(candidate, list(cases))
    for row in result.get("cases", []):
        print(
            f"SANITIZE_CASE {row['case_id']} category={row['category']} "
            f"max_abs={row['out_max_abs']:.3e} norm_l2={row['out_norm_l2']:.3e}"
        )
    if result.get("status") != "passed":
        print(f"SANITIZE_FAIL [{result.get('failed_gate')}] {result.get('private_error', '')}")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--lab-root", type=Path, default=Path("/app/workdir/paged_gqa_decode_lab")
    )
    parser.add_argument(
        "--suite",
        choices=("one", "quick", "full", *FAMILIES, "edge", "sanitize", "sanitize-inner"),
        default="full",
    )
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
        stage = "public edge manifest"
        edge_cases: list[dict[str, Any]] = []
        if args.suite == "edge":
            edge_cases = public_edge_cases()
        elif args.suite == "sanitize-inner":
            edge_cases = public_edge_cases(sanitize_only=True)
        stage = "candidate import, JIT, launch, or public comparison"
        runtime = _load_trusted_runtime()
        globals().pop("_load_trusted_runtime", None)
        if args.suite == "sanitize-inner":
            return _run_sanitize_inner(runtime, candidate, edge_cases)
        if args.suite == "edge":
            result = runtime.run_suite(
                candidate, [], edge_cases, trials=1, warmups=1, stop_on_edge_failure=False
            )
        else:
            trials = 1 if args.suite in ("one", "quick") else 3
            result = runtime.run_suite(
                candidate, cases_for_suite(args.suite), [], trials=trials, warmups=1
            )
        payload = _public_payload(
            result,
            suite=args.suite,
            candidate_version=args.candidate_version,
            num_candidate_versions=int(admission["num_candidate_versions"]),
            edge_cases=edge_cases,
        )
        if args.json_output:
            args.json_output.write_text(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        _print_result(payload)
        print(json.dumps(payload, sort_keys=True, allow_nan=False))
        return 0 if payload["status"] == "passed" else 1
    except Exception as exc:
        _print_header("candidate-only; FAILED")
        print(f"PUBLIC_FAILURE_STAGE: {stage}", file=sys.stderr)
        print(f"PUBLIC_FAILURE: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
