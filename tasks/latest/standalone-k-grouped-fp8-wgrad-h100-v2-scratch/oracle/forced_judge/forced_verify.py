#!/usr/bin/env python3
"""Forced-oracle verifier for standalone K-grouped FP8 wgrad v2 (LOCAL ONLY).

Drives the Judge's own ``verify_wgrad.py`` (imported unchanged: protected
hashes, profile, both root-only manifests, hardware envelope, the
compute-sanitizer command, the unprivileged worker, every runtime record
check and the scored aggregation are the Judge's own code) with these
deviations, each recorded under ``forced`` in the private result:

1. source admission: ``forced_admission.validate_submission`` keeps the
   Judge's decision and admits a rejected submission anyway (for DeepGEMM:
   ``import deep_gemm``); the original errors are recorded;
2. per-case isolation with strict timeouts.  Upstream DeepGEMM's SM90
   kernel does not support most edge shapes: some raise, some fault the
   context, and some spin on the GPU.  In the Judge, one hang ends the whole
   worker without a result.  Here every sanitizer-flagged case runs alone
   under the Judge's compute-sanitizer command (FORCED_SANITIZER_CASE_TIMEOUT),
   every edge case runs alone in the Judge's worker
   (FORCED_EDGE_CASE_TIMEOUT), and the scored grid runs in one more Judge
   worker with no edge cases, with the Judge's trials, warm-ups and timeout;
3. a sanitizer or edge failure is recorded instead of ending the evaluation.

If every gate passes, the result is the Judge's normal valid result (the
combined edge record is checked by the Judge's own validate_runtime_result).
If a gate fails, the result is a normal trusted gate failure (score 0,
failing gate named, as ``run_judge.py`` expects), and the scored-grid numbers
are recorded under ``forced_scored_*`` after the Judge's own per-case
validation and aggregation.  The LLM integrity review is skipped by the host
(TRUSTED_CONTROL=1), not here.
"""

from __future__ import annotations

import copy
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

JUDGE_ROOT = Path("/opt/standalone_k_grouped_wgrad_judge")
sys.path.insert(0, str(JUDGE_ROOT))

import verify_wgrad as V  # noqa: E402
import forced_admission  # noqa: E402

EDGE_CASE_TIMEOUT = float(os.environ.get("FORCED_EDGE_CASE_TIMEOUT", "240"))
SANITIZER_CASE_TIMEOUT = float(os.environ.get("FORCED_SANITIZER_CASE_TIMEOUT", "300"))


def _log_tail(path: Path, limit: int = 600) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines[-4:])[-limit:]


def _subroot(run_root: Path, name: str) -> Path:
    path = run_root / name
    path.mkdir(mode=0o755)
    path.chmod(0o755)
    return path


def _validated(
    runtime_result: Mapping[str, Any],
    cases: Any,
    edge_cases: Any,
    edge_record: Mapping[str, Any],
    *,
    trials: int,
    warmups: int,
    edges_as_run: bool,
) -> tuple[float, dict[str, float], dict[str, float]]:
    """The Judge's validate_runtime_result on the scored run plus the edge record.

    With ``edges_as_run`` the edge-record equality is not required (used only
    for the forced_scored_* numbers); every scored-case check stays.
    """
    result = copy.deepcopy(dict(runtime_result))
    result["edge"] = copy.deepcopy(dict(edge_record))
    gates = dict(result.get("gates") or {})
    gates["edge_correctness_ok"] = True
    result["gates"] = gates
    if not edges_as_run:
        return V.validate_runtime_result(
            result, cases, edge_cases, expected_trials=trials, expected_warmups=warmups
        )
    saved = V._validate_edge_record
    V._validate_edge_record = lambda _edge, _expected: None
    try:
        return V.validate_runtime_result(
            result, cases, edge_cases, expected_trials=trials, expected_warmups=warmups
        )
    finally:
        V._validate_edge_record = saved


def evaluate(args: Any) -> tuple[int, dict[str, Any]]:
    protected = V.verify_protected_hashes(JUDGE_ROOT / "protected_hashes.txt")
    if protected:
        return 1, V.failure_detail("protected Judge/runtime hash failure") | {
            "protected_failures": protected
        }
    if args.sanitizer_mode not in V.SANITIZER_MODES:
        return 1, V.failure_detail("invalid compute-sanitizer mode")
    try:
        profile, suite_id, round_id, version_override = V.load_evaluation_profile()
    except Exception as exc:
        return 1, V.failure_detail(f"{type(exc).__name__}: {exc}")
    base = V.failure_detail("evaluation did not complete", profile=profile, suite_id=suite_id)
    base["profile_ok"] = 1.0
    base["round_id"] = round_id
    forced: dict[str, Any] = {
        "mode": "forced-oracle",
        "bypassed": [],
        "isolation": "per-case",
        "edge_case_timeout_seconds": EDGE_CASE_TIMEOUT,
        "sanitizer_case_timeout_seconds": SANITIZER_CASE_TIMEOUT,
        "notes": "LOCAL forced-oracle Judge image; not an admissible submission result",
    }
    base["forced"] = forced
    if version_override is not None:
        base["candidate_version_override"] = float(version_override)

    try:
        cases, manifest_sha256 = V.load_final_manifest()
        edge_cases, edge_manifest_sha256 = V.load_edge_manifest()
        base["suite_ok"] = 1.0
        base["manifest_sha256"] = manifest_sha256
        base["edge_manifest_sha256"] = edge_manifest_sha256
        base["case_count"] = float(len(cases))
        base["edge_cases_total"] = float(len(edge_cases))
    except Exception as exc:
        base["private_error"] = f"{type(exc).__name__}: {exc}"
        V._mark_failed_gate(base, "suite_ok")
        return 1, base

    submission_root = V.locate_submission(args.app_dir.resolve())
    admission = forced_admission.validate_submission(submission_root, version_override)
    final_version = admission.get("final_version")
    versions = admission.get("num_candidate_versions")
    version_count = versions if isinstance(versions, int) and not isinstance(versions, bool) else 0
    selected = final_version if isinstance(final_version, int) and not isinstance(final_version, bool) else 0
    base["final_version"] = float(selected)
    base["num_candidate_versions"] = float(version_count)
    if not admission.get("ok") or selected <= 0:
        base["private_error"] = "submission source admission failed"
        base["admission"] = admission
        V._mark_failed_gate(base, "source_ok")
        return 1, base
    if admission.get("forced_bypass"):
        forced["bypassed"].append("source_ok")
        forced["original_admission"] = admission.get("original_decision")
    base["source_ok"] = 1.0

    if args.trials <= 0 or args.warmups <= 0:
        base["private_error"] = "invalid trial or warmup count"
        return 1, base
    total_timeout = float(args.total_timeout)
    if not math.isfinite(total_timeout) or total_timeout <= 60.0:
        base["private_error"] = "invalid verifier total timeout"
        return 1, base

    started = time.monotonic()
    deadline = started + total_timeout
    run_root = Path(tempfile.mkdtemp(prefix="standalone-kgrouped-wgrad-v2-forced-", dir="/tmp"))
    run_root.chmod(0o755)
    failures: list[str] = []  # in the order the normal Judge would hit them
    try:
        try:
            hardware = V._trusted_hardware()
        except Exception as exc:
            base["private_error"] = f"{type(exc).__name__}: {exc}"
            V._mark_failed_gate(base, "hardware_ok")
            return 1, base
        base["hardware_ok"] = 1.0
        candidate_dir = run_root / "selected_candidate"
        candidate_dir.mkdir(mode=0o755)
        candidate_path = candidate_dir / "submission.py"
        shutil.copyfile(Path(str(admission["selected_path"])), candidate_path)
        candidate_path.chmod(0o444)
        candidate_dir.chmod(0o555)

        # 1. compute-sanitizer memcheck, one flagged case per process.
        sanitizer_rows: list[dict[str, Any]] = []
        sanitizer_failed_gate: str | None = None
        if args.sanitizer_mode == "required":
            for index, case in enumerate(c for c in edge_cases if c["sanitize"]):
                remaining = deadline - time.monotonic() - 120.0
                if remaining <= 0.0:
                    sanitizer_rows.append({"case_id": case["case_id"], "status": "not_run_deadline"})
                    continue
                sub = _subroot(run_root, f"sanitizer_{index:02d}")
                t0 = time.monotonic()
                try:
                    record = V.run_sanitizer(
                        candidate_path, sub, [case],
                        timeout=min(SANITIZER_CASE_TIMEOUT, remaining),
                    )
                except Exception as exc:  # noqa: BLE001
                    record = {"status": "error", "private_error": f"{type(exc).__name__}: {exc}"}
                row = {
                    "case_id": case["case_id"],
                    "g": case["g"], "m": case["m"], "n": case["n"],
                    "status": record.get("status"),
                    "failed_gate": record.get("failed_gate"),
                    "returncode": record.get("returncode"),
                    "private_error": str(record.get("private_error") or "")[:400],
                    "report": list(record.get("report") or [])[:8],
                    "seconds": round(time.monotonic() - t0, 2),
                }
                sanitizer_rows.append(row)
                if record.get("status") != "passed" and sanitizer_failed_gate is None:
                    gate = str(record.get("failed_gate") or "memory_safety_ok")
                    sanitizer_failed_gate = gate if gate in V.GATE_NAMES else "memory_safety_ok"
            base["sanitizer"] = {
                "status": "passed" if sanitizer_failed_gate is None else "failed",
                "isolation": "per-case",
                "cases": sanitizer_rows,
                "passed": sum(1 for r in sanitizer_rows if r["status"] == "passed"),
                "total": len(sanitizer_rows),
            }
            if sanitizer_failed_gate is None:
                base["sanitizer_ran"] = 1.0
            else:
                failures.append(sanitizer_failed_gate)
                forced["sanitizer_failure_recorded_not_fatal"] = True
        else:
            base["sanitizer"] = {"status": "not_run", "mode": args.sanitizer_mode}

        # 2. every edge case alone in the Judge's worker.
        edge_rows: list[dict[str, Any]] = []
        edge_meta: list[dict[str, Any]] = []
        for index, case in enumerate(edge_cases):
            remaining = deadline - time.monotonic() - 900.0  # keep room for the scored grid
            row: dict[str, Any]
            t0 = time.monotonic()
            if remaining <= 0.0:
                row = {"case_id": case["case_id"], "category": case["category"], "ok": False,
                       "gate": "edge_correctness_ok", "error": "forced: not run (deadline)"}
            else:
                sub = _subroot(run_root, f"edge_{index:03d}")
                try:
                    result = V.run_worker(
                        candidate_path, sub, [], [case],
                        trials=args.trials, warmups=args.warmups,
                        timeout=min(EDGE_CASE_TIMEOUT, remaining),
                    )
                    rows = (result.get("edge") or {}).get("results") or []
                    if len(rows) == 1 and isinstance(rows[0], Mapping):
                        row = dict(rows[0])
                    else:
                        row = {"case_id": case["case_id"], "category": case["category"], "ok": False,
                               "gate": str(result.get("failed_gate") or "correctness_ok"),
                               "error": str(result.get("private_error") or "no edge row")[:400]}
                except V.UnprivilegedTimeout:
                    row = {"case_id": case["case_id"], "category": case["category"], "ok": False,
                           "gate": "edge_correctness_ok",
                           "error": f"forced: worker killed after the {EDGE_CASE_TIMEOUT:.0f} s per-case timeout (hang)"}
                except Exception as exc:  # noqa: BLE001 - worker died without a result
                    row = {"case_id": case["case_id"], "category": case["category"], "ok": False,
                           "gate": "edge_correctness_ok",
                           "error": f"forced: {type(exc).__name__}: {exc}; log: "
                           + _log_tail(sub / "candidate_worker.log")}
                shutil.rmtree(sub, ignore_errors=True)
            edge_rows.append(row)
            edge_meta.append({
                "case_id": case["case_id"], "category": case["category"],
                "g": case["g"], "m": case["m"], "n": case["n"], "k_total": sum(case["ks"]),
                "ok": bool(row.get("ok") is True), "gate": row.get("gate"),
                "error": str(row.get("error") or "")[:300],
                "seconds": round(time.monotonic() - t0, 2),
            })
        passed = sum(1 for r in edge_rows if r.get("ok") is True)
        edge_record = {
            "cases_total": len(edge_cases),
            "cases_passed": passed,
            "cases_failed": len(edge_cases) - passed,
            "results": edge_rows,
        }
        base["edge_cases_failed"] = float(len(edge_cases) - passed)
        base["forced_edge_cases"] = edge_meta
        base["forced_edge_cases_passed"] = float(passed)
        edge_failed = passed != len(edge_cases)
        if edge_failed:
            gate = "edge_correctness_ok"
            for row in edge_rows:
                if row.get("ok") is not True and row.get("gate") in ("harness_integrity_ok", "memory_safety_ok"):
                    gate = str(row["gate"])
                    break
            failures.append(gate)
            forced["edge_failure_recorded_not_fatal"] = True

        # 3. the scored grid in one Judge worker with the Judge's protocol.
        remaining = deadline - time.monotonic() - 30.0
        if remaining <= 0.0:
            raise RuntimeError("Judge total deadline exhausted before the scored grid")
        sub = _subroot(run_root, "scored")
        runtime_result = V.run_worker(
            candidate_path, sub, cases, [],
            trials=args.trials, warmups=args.warmups,
            timeout=min(float(args.candidate_timeout), remaining),
        )
        if runtime_result.get("candidate_sha256") not in (None, admission["selected_sha256"]):
            raise RuntimeError("runtime evaluated a different candidate source")
        scored_passed = runtime_result.get("status") == "passed"
        gates = runtime_result.get("gates") if isinstance(runtime_result.get("gates"), Mapping) else {}
        for name in ("correctness_ok", "memory_safety_ok", "harness_integrity_ok"):
            base[name] = 1.0 if gates.get(name) is True else 0.0
        base["edge_correctness_ok"] = 0.0 if edge_failed else 1.0
        if not scored_passed:
            gate = str(runtime_result.get("failed_gate") or "correctness_ok")
            failures.append(gate if gate in V.GATE_NAMES else "correctness_ok")
        if sanitizer_failed_gate is not None:
            base["memory_safety_ok"] = 0.0
        for row in edge_rows:
            if row.get("ok") is not True and row.get("gate") in ("memory_safety_ok", "harness_integrity_ok"):
                base[str(row["gate"])] = 0.0

        if not failures and runtime_result.get("candidate_sha256") == admission["selected_sha256"]:
            score, families, bands = _validated(
                runtime_result, cases, edge_cases, edge_record,
                trials=args.trials, warmups=args.warmups, edges_as_run=False,
            )
            detail = dict(base)
            detail.update(
                {
                    "score_tflops": score,
                    "coarse_tflops": families["coarse"],
                    "fine_tflops": families["fine"],
                    "selected_source_sha256": admission["selected_sha256"],
                    "score_method": "sqrt(coarse family geomean * fine family geomean)",
                    "wall_time_seconds": time.monotonic() - started,
                    "hardware": hardware,
                    "band_tflops": bands,
                    "private_runtime_result": runtime_result,
                }
            )
            detail["private_error"] = ""
            return 0, detail

        base["private_runtime_result"] = runtime_result
        if scored_passed:
            try:
                score, families, bands = _validated(
                    runtime_result, cases, edge_cases, edge_record,
                    trials=args.trials, warmups=args.warmups, edges_as_run=True,
                )
                base["forced_scored_score_tflops"] = score
                base["forced_scored_coarse_tflops"] = families["coarse"]
                base["forced_scored_fine_tflops"] = families["fine"]
                base["forced_scored_band_tflops"] = bands
                base["forced_scored_validated"] = 1.0
            except Exception as exc:  # noqa: BLE001
                base["forced_scored_validated"] = 0.0
                base["forced_scored_error"] = f"{type(exc).__name__}: {exc}"
        else:
            base["forced_scored_validated"] = 0.0
            base["forced_scored_error"] = str(runtime_result.get("private_error") or "")
        base["forced_scored_cases"] = [
            {k: row.get(k) for k in ("case_id", "family", "band", "g", "m", "n", "tflops", "median_us")}
            for row in (runtime_result.get("cases") or [])
            if isinstance(row, Mapping)
        ]
        for gate in failures:
            V._mark_failed_gate(base, gate)
        zero = {name for name in V.RUNTIME_GATES if base.get(name) == 0.0}
        base["forced_failed_gates"] = sorted(set(failures) | zero, key=V.GATE_NAMES.index)
        base["private_error"] = "forced: gate(s) failed: " + ", ".join(base["forced_failed_gates"])
        base["wall_time_seconds"] = time.monotonic() - started
        return 1, base
    except Exception as exc:
        base["private_error"] = f"{type(exc).__name__}: {exc}"
        if not base.get("failed_gate_code"):
            V._mark_failed_gate(base, failures[0] if failures else "correctness_ok")
        return 1, base
    finally:
        shutil.rmtree(run_root, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    V.evaluate = evaluate
    return V.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
