#!/usr/bin/env python3
"""Candidate-only public feedback for the standalone sparse-MLA prefill task.

Suites:
  one         one public case (fast compile/launch check)
  quick       one public case per group, streams included
  full        all public cases in public_cases.json
  edge        the public edge cases (correctness, guard bands, unchanged inputs)
  sanitize    the public sanitizer subset under compute-sanitizer memcheck

``--case ID`` (repeatable) runs only the named public cases; ``--perf-only`` skips
the reference comparison and only checks that every output element is written.

Output carries only disclosed shapes, correctness summaries, per-case edge
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
PUBLIC_CASE_MANIFEST = BENCHMARK_ROOT / "public_cases.json"
PUBLIC_CASE_SUITE_ID = "sparse-mla-public-v4"
PUBLIC_EDGE_SUITE_ID = "edge-public-v3"
PUBLIC_EDGE_SCHEMA_VERSION = 3
PUBLIC_PAYLOAD_SCHEMA_VERSION = 2

SANITIZER = Path("/usr/local/cuda/compute-sanitizer/compute-sanitizer")
SANITIZER_EXIT_CODE = 86
SANITIZER_PADDING_BYTES = 4 << 20
SANITIZER_KERNEL_EXCLUDES = (
    "regex=^_Z", "kns=xmma", "kns=gemm", "kns=nvjet", "kns=cutlass", "kns=cublas",
)
SANITIZER_ARGUMENTS = (
    "--tool=memcheck",
    f"--error-exitcode={SANITIZER_EXIT_CODE}",
    "--target-processes=application-only",
    "--destroy-on-device-error=context",
    "--force-blocking-launches",
    "--check-api-memory-access=no",
    "--padding",
    str(SANITIZER_PADDING_BYTES),
    "--print-limit=20",
    *(item for pattern in SANITIZER_KERNEL_EXCLUDES for item in ("--kernel-name-exclude", pattern)),
)
SANITIZER_ACCESS_MARKERS = ("invalid __global__", "invalid __shared__", "invalid __local__", "misaligned")

# compute-sanitizer prints this when it cannot attach to the CUDA process; the
# target then keeps running without instrumentation, so stop the whole tree.
# Seen on platforms whose GPU virtualization hook is preloaded into every process.
SANITIZER_UNAVAILABLE_MARKER = "No attachable process found"
SANITIZE_TIMEOUT_SECONDS = 900
EXIT_SANITIZER_UNAVAILABLE = 2
EXIT_SANITIZER_INCONCLUSIVE = 3

PUBLIC_FAMILIES = ("v32", "glm5", "v4_flash", "v4_pro")
PUBLIC_GROUPS = ("basic", "trace", "replay", "multi", "pattern", "sweep", "heads", "length")
PUBLIC_TIERS = ("evidence", "sweep")
ONE_CASE_ID = "public-002-basic"


def public_cases() -> list[dict[str, Any]]:
    """The disclosed public scored cases; the runtime validates every field again."""
    payload = json.loads(PUBLIC_CASE_MANIFEST.read_text(encoding="utf-8"))
    if payload.get("suite_id") != PUBLIC_CASE_SUITE_ID:
        raise RuntimeError(f"public case manifest must be suite {PUBLIC_CASE_SUITE_ID}")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases or any(not isinstance(case, dict) for case in cases):
        raise RuntimeError("public case manifest must list case objects")
    return cases

# Mirrors the runtime's GATE_ORDER (asserted by a task-local test).
PUBLIC_GATES = (
    "edge_correctness_ok",
    "memory_safety_ok",
    "correctness_ok",
)
PUBLIC_EDGE_FIELDS = (
    "case_id",
    "category",
    "family",
    "s_q",
    "s_kv",
    "topk",
    "sm_scale",
    "bias",
    "index_mode",
    "invalid_rows",
    "duplicates",
    "scramble",
    "sink_mode",
    "length_mode",
    "sanitize",
)
PUBLIC_EDGE_OPTIONAL_FIELDS = ("h_q",)
PUBLIC_METRIC_FIELDS = tuple(
    f"{tensor}_{metric}" for tensor in ("out", "max_logits", "lse")
    for metric in ("max_abs", "max_rel", "cos_diff")
)
PUBLIC_CASE_FIELDS = (
    "case_id",
    "group",
    "tier",
    "family",
    "s_q",
    "s_kv",
    "topk",
    "h_q",
    "d_qk",
    "sm_scale",
    "tflops",
    "checked_calls",
)
# Present only for one kind of case (single calls, streams, or checked cases).
PUBLIC_CASE_OPTIONAL_FIELDS = (
    "median_us", "calls_total", "calls_run", "truncated", "stream_seconds", *PUBLIC_METRIC_FIELDS,
)
MAX_PUBLIC_ERROR_CHARS = 600


def cases_for_suite(name: str, case_ids: Sequence[str] = ()) -> list[dict[str, Any]]:
    cases = public_cases()
    by_id = {case["case_id"]: case for case in cases}
    if case_ids:
        unknown = [case_id for case_id in case_ids if case_id not in by_id]
        if unknown:
            raise ValueError(f"unknown public case IDs: {unknown}")
        return [dict(by_id[case_id]) for case_id in case_ids]
    if name == "one":
        return [dict(by_id[ONE_CASE_ID])]
    if name == "quick":
        first: dict[str, dict[str, Any]] = {}
        for case in cases:
            first.setdefault(str(case["group"]), case)
        return [dict(case) for case in first.values()]
    if name == "full":
        return [dict(case) for case in cases]
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
    import sparse_mla_prefill_runtime as runtime

    sys.path = [
        entry for entry in sys.path if Path(entry or ".").resolve() != BENCHMARK_ROOT
    ]
    sys.modules.pop("sparse_mla_prefill_runtime", None)
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
    """Disclosed manifest fields plus this case's own pass/fail and metrics."""
    public = _disclosed_fields(case, PUBLIC_EDGE_FIELDS)
    public.update({field: case[field] for field in PUBLIC_EDGE_OPTIONAL_FIELDS if field in case})
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
        "cases": [_public_case_row(record) for record in result.get("cases", [])],
    }
    if "family_tflops" in result:
        payload["family_tflops"] = _public_aggregate(
            result["family_tflops"], label="family_tflops", allowed_keys=PUBLIC_FAMILIES
        )
        payload["tier_tflops"] = _public_aggregate(
            result["tier_tflops"], label="tier_tflops",
            allowed_keys=tuple(f"{family}:{tier}" for family in PUBLIC_FAMILIES for tier in PUBLIC_TIERS),
        )
        payload["group_tflops"] = _public_aggregate(
            result["group_tflops"], label="group_tflops",
            allowed_keys=tuple(f"{family}:{group}" for family in PUBLIC_FAMILIES for group in PUBLIC_GROUPS),
        )
        payload["primary_tflops"] = _finite(result["primary_tflops"], "primary_tflops")
    return payload


def _public_case_row(record: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(_disclosed_fields(record, PUBLIC_CASE_FIELDS))
    row.update({field: record[field] for field in PUBLIC_CASE_OPTIONAL_FIELDS if field in record})
    for field in ("median_us", "tflops", "sm_scale", "stream_seconds", *PUBLIC_METRIC_FIELDS):
        if field in row:
            row[field] = _finite(row[field], f"{record.get('case_id')}.{field}")
    return row


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
            "EDGE_CASE                                CATEGORY               FAMILY     S_Q   S_KV  TOPK"
            "  SINK     LENGTH   RESULT        OUT_COS_DIFF"
        )
        for row in edge["results"]:
            result = "PASS" if row["ok"] else f"FAIL [{row['gate']}]"
            cos = f"{row['out_cos_diff']:>13.3e}" if row["ok"] else ""
            print(
                f"{row['case_id']:<40} {row['category']:<22} {row['family']:<8} {row['s_q']:>6} "
                f"{row['s_kv']:>6} {row['topk']:>5}  {row['sink_mode']:<8} {row['length_mode']:<8} "
                f"{result:<13}{cos}"
            )
        for row in edge["results"]:
            if not row["ok"]:
                print(f"EDGE_FAIL  {row['case_id']}  [{row['gate']}]  {row['error']}")
    if payload["cases"]:
        print()
        for row in payload["cases"]:
            extra = (
                f"calls {row['calls_run']}/{row['calls_total']}{' TRUNCATED' if row['truncated'] else ''}"
                if "calls_run" in row else f"median_us {row['median_us']:.1f}"
            )
            print(f"CASE {row['case_id']:<24} {row['family']:<9} h_q={row['h_q']:<4} topk={row['topk']:<5} "
                  f"s_q={row['s_q']:<6} s_kv={row['s_kv']:<8} {row['tflops']:8.2f} TFLOPS  {extra}")
    if "primary_tflops" in payload:
        print()
        for group, value in sorted(payload["group_tflops"].items()):
            print(f"GROUP_{group.upper().replace(':', '_')}_TFLOPS  {value:.6f}")
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


def _explicit_access_report(report: Sequence[str]) -> bool:
    return any(marker in line.lower() for line in report for marker in SANITIZER_ACCESS_MARKERS)


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
    if _explicit_access_report(report):
        _print_header("candidate-only; sanitizer subset failed")
        print("MEMCHECK                  FAIL (invalid memory access reported)")
        for line in report[:60]:
            print(line)
        print("\n".join([*cases, *failures]))
        return 1
    if outcome == "timeout":
        _print_header("candidate-only; sanitizer subset timed out")
        print(f"MEMCHECK                  TIMEOUT after {SANITIZE_TIMEOUT_SECONDS} s")
        for line in report[:60]:
            print(line)
        return 1
    fatal = [line for line in cases if " status=fatal" in line]
    hard = [line for line in failures if "[memory_safety_ok]" in line or "[correctness_ok]" in line]
    if hard:
        _print_header("candidate-only; sanitizer subset failed")
        print("MEMCHECK                  no invalid access reported")
        print("\n".join([*cases, *failures]))
        return 1
    if fatal or (process.returncode not in (0, SANITIZER_EXIT_CODE) and not failures):
        _print_header("candidate-only; sanitizer subset inconclusive")
        print("MEMCHECK                  INCONCLUSIVE (a fatal CUDA error without an invalid-access report;")
        print("                          the pass stopped there; run --edge to check those cases natively)")
        for line in report[:60]:
            print(line)
        print("\n".join([*cases, *failures]))
        return EXIT_SANITIZER_INCONCLUSIVE
    deferred = [line for line in cases if " status=deferred" in line]
    _print_header("candidate-only; sanitizer subset clean" if not deferred
                  else "candidate-only; no invalid access, some answers wrong under memcheck")
    print("MEMCHECK                  PASS (no invalid access reported)")
    if deferred:
        print("NOTE                      cases marked deferred answered wrongly or raised under memcheck;")
        print("                          run --edge to check their numerics natively")
    print("\n".join(cases))
    return 0


def _run_sanitize_inner(runtime: Any, candidate: Path, cases: Sequence[Mapping[str, Any]]) -> int:
    """Runs inside compute-sanitizer: plain allocations, one call per case, stop at a fatal error."""

    def progress(line: str) -> None:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    result = runtime.run_sanitize_cases(candidate, list(cases), progress=progress)
    for row in result.get("cases", []):
        detail = (
            f"out_cos_diff={row['out_cos_diff']:.3e}" if row.get("ok")
            else f"error={str(row.get('error', ''))[:200]}"
        )
        print(f"SANITIZE_CASE {row['case_id']} category={row['category']} status={row['status']} {detail}")
    sys.stdout.flush()
    if result.get("status") == "failed":
        print(f"SANITIZE_FAIL [{result.get('failed_gate')}] {result.get('private_error', '')}")
        return 1
    if result.get("status") == "fatal":
        print(f"SANITIZE_FATAL {result.get('private_error', '')}")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--lab-root", type=Path, default=Path("/app/workdir/sparse_mla_prefill_lab")
    )
    parser.add_argument(
        "--suite",
        choices=("one", "quick", "full", "edge", "sanitize", "sanitize-inner"),
        default="full",
    )
    parser.add_argument("--candidate-version", type=int, required=True)
    parser.add_argument("--case", action="append", default=[], metavar="ID",
                        help="run only this public case (repeatable)")
    parser.add_argument("--perf-only", action="store_true",
                        help="skip the reference comparison; still check complete writes")
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
            result = runtime.run_suite(candidate, [], edge_cases, trials=1, warmups=1)
        else:
            trials = 3 if args.suite == "full" or args.case else 1
            result = runtime.run_suite(
                candidate, cases_for_suite(args.suite, args.case), [], trials=trials, warmups=1,
                check=not args.perf_only,
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
