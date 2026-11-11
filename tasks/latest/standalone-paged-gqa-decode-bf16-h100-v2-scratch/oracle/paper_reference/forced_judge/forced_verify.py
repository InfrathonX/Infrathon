#!/usr/bin/env python3
"""Forced-oracle verifier for standalone paged GQA decode v2 (LOCAL ONLY).

Drives the Judge's own ``verify_paged_gqa.py`` (imported unchanged, so the
protected-hash check, profile, both root-only manifests, hardware envelope,
compute-sanitizer pass, worker, every runtime record check and the scored
aggregation are the Judge's own code) with exactly three deviations, each
recorded under ``forced`` in the private result:

1. source admission: ``forced_admission.validate_submission`` keeps the
   Judge's decision and admits a rejected submission anyway (the original
   errors are recorded);
2. the compute-sanitizer pass runs exactly as in the Judge, but its failure
   is recorded instead of ending the evaluation;
3. the worker is ``forced_worker.py``: the Judge's worker with
   ``stop_on_edge_failure=False``, so the scored grid is measured even after
   a failing edge case.

If every gate passes, the result is the Judge's normal valid result.  If a
gate fails, the result is a normal trusted gate failure (score 0, failing
gate named, as ``run_judge.py`` expects), and the scored-case numbers that
were still measured are recorded under ``forced_scored_*`` keys after the
Judge's own per-case validation and aggregation.  The LLM integrity review
is not part of this file; the host skips it with TRUSTED_CONTROL=1.
"""

from __future__ import annotations

import copy
import shutil
import sys
import tempfile
import time
import math
from pathlib import Path
from typing import Any, Mapping

JUDGE_ROOT = Path("/opt/standalone_paged_gqa_decode_judge")
sys.path.insert(0, str(JUDGE_ROOT))

import verify_paged_gqa as V  # noqa: E402
import forced_admission  # noqa: E402

FORCED_WORKER = JUDGE_ROOT / "forced_worker.py"
FAMILY_A, FAMILY_B = V.FAMILIES  # ("q1", "q16")


def _scored_only(
    runtime_result: Mapping[str, Any],
    cases: Any,
    edge_cases: Any,
    *,
    trials: int,
    warmups: int,
) -> tuple[float, dict[str, float], dict[str, float]]:
    """The Judge's validate_runtime_result with the edge record taken as given.

    Every scored-case record check (identity, trial clocks, TFLOPS recompute,
    error limits, plausibility bound, sentinel count, family/band/primary
    aggregates) is the Judge's own; only the edge-record equality and the
    runtime pass/fail status are not required here.
    """
    result = copy.deepcopy(dict(runtime_result))
    result["status"] = "passed"
    result["gates"] = {name: True for name in V.RUNTIME_GATES}
    saved = V._validate_edge_record

    def _edges_as_run(edge: Any, _expected: Any) -> int:
        # The sentinel lower bound counts two checks per edge case; a failed
        # edge case stops after its first call, so count only the passed ones.
        rows = edge.get("results") if isinstance(edge, Mapping) else None
        if not isinstance(rows, list):
            return 0
        return sum(1 for row in rows if isinstance(row, Mapping) and row.get("ok") is True)

    V._validate_edge_record = _edges_as_run
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
    base["sanitizer_mode"] = args.sanitizer_mode
    forced: dict[str, Any] = {
        "mode": "forced-oracle",
        "bypassed": [],
        "notes": "LOCAL forced-oracle Judge image; not an admissible submission result",
    }
    base["forced"] = forced
    if version_override is not None:
        base["candidate_version_override"] = float(version_override)

    try:
        cases, manifest_sha256 = V.load_final_manifest()
        edge_cases, edge_manifest_sha256 = V.load_edge_manifest()
        if {str(c["case_id"]) for c in cases} & {str(c["case_id"]) for c in edge_cases}:
            raise RuntimeError("scored and edge case IDs overlap")
        base["suite_ok"] = 1.0
        base["manifest_sha256"] = manifest_sha256
        base["edge_manifest_sha256"] = edge_manifest_sha256
        base["case_count"] = float(len(cases))
        base["edge_cases_total"] = float(len(edge_cases))
        base["edge_sanitize_cases"] = float(sum(1 for c in edge_cases if c["sanitize"]))
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

    if args.trials != V.EXPECTED_TRIALS or args.warmups != V.EXPECTED_WARMUPS:
        base["private_error"] = "verifier trial/warm-up counts differ from the Judge's"
        return 1, base
    total_timeout = float(args.total_timeout)
    if not math.isfinite(total_timeout) or total_timeout <= 60.0:
        base["private_error"] = "invalid verifier total timeout"
        return 1, base

    started = time.monotonic()
    deadline = started + total_timeout
    run_root = Path(tempfile.mkdtemp(prefix="standalone-paged-gqa-v2-forced-", dir="/tmp"))
    run_root.chmod(0o755)
    failures: list[str] = []  # gates in the order the normal Judge would hit them
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

        sanitizer_failed_gate: str | None = None
        if args.sanitizer_mode == "required":
            sanitize_cases = [c for c in edge_cases if c["sanitize"]]
            remaining = deadline - time.monotonic() - 60.0
            if remaining <= 0.0:
                raise RuntimeError("Judge total deadline exhausted before the sanitizer pass")
            sanitizer = V.run_sanitizer(
                candidate_path,
                run_root,
                sanitize_cases,
                timeout=min(float(args.sanitizer_timeout), remaining),
                expected_sha256=str(admission["selected_sha256"]),
            )
            base["sanitizer"] = sanitizer
            if sanitizer.get("status") == "passed":
                base["sanitizer_ran"] = 1.0
            else:
                gate = str(sanitizer.get("failed_gate") or "memory_safety_ok")
                sanitizer_failed_gate = gate if gate in V.GATE_NAMES else "memory_safety_ok"
                failures.append(sanitizer_failed_gate)
                forced["sanitizer_failure_recorded_not_fatal"] = V._sanitizer_failure_message(sanitizer)
        else:
            base["sanitizer"] = {"status": "not_run", "mode": args.sanitizer_mode}

        remaining = deadline - time.monotonic() - 30.0
        if remaining <= 0.0:
            raise RuntimeError("Judge total deadline exhausted")
        V.WORKER = FORCED_WORKER
        forced["worker"] = "forced_worker.py (stop_on_edge_failure=False)"
        runtime_result = V.run_worker(
            candidate_path,
            run_root,
            cases,
            edge_cases,
            trials=args.trials,
            warmups=args.warmups,
            timeout=min(float(args.candidate_timeout), remaining),
        )
        if runtime_result.get("candidate_sha256") not in (None, admission["selected_sha256"]):
            raise RuntimeError("runtime evaluated a different candidate source")
        base["private_runtime_result"] = runtime_result
        V._apply_runtime_gates(base, runtime_result)
        edge = runtime_result.get("edge") if isinstance(runtime_result.get("edge"), Mapping) else {}
        forced["edge_failures"] = [
            {k: row.get(k) for k in ("case_id", "category", "gate", "error")}
            for row in (edge.get("results") or [])
            if isinstance(row, Mapping) and row.get("ok") is not True
        ]

        if (
            runtime_result.get("status") == "passed"
            and sanitizer_failed_gate is None
            and runtime_result.get("candidate_sha256") == admission["selected_sha256"]
        ):
            score, families, bands = V.validate_runtime_result(
                runtime_result, cases, edge_cases,
                expected_trials=args.trials, expected_warmups=args.warmups,
            )
            detail = dict(base)
            detail.update(
                {
                    "score_tflops": score,
                    f"{FAMILY_A}_tflops": families[FAMILY_A],
                    f"{FAMILY_B}_tflops": families[FAMILY_B],
                    "selected_source_sha256": admission["selected_sha256"],
                    "score_method": f"sqrt({FAMILY_A} family geomean * {FAMILY_B} family geomean)",
                    "wall_time_seconds": time.monotonic() - started,
                    "hardware": hardware,
                    "band_tflops": bands,
                }
            )
            detail["private_error"] = ""
            return 0, detail

        # A gate failed.  Name it the way the Judge would and keep the score 0;
        # record the still-measured scored grid separately.
        runtime_gate = runtime_result.get("failed_gate")
        if runtime_result.get("status") != "passed":
            failures.append(str(runtime_gate) if runtime_gate in V.GATE_NAMES else "correctness_ok")
        try:
            score, families, bands = _scored_only(
                runtime_result, cases, edge_cases, trials=args.trials, warmups=args.warmups
            )
            base["forced_scored_score_tflops"] = score
            base[f"forced_scored_{FAMILY_A}_tflops"] = families[FAMILY_A]
            base[f"forced_scored_{FAMILY_B}_tflops"] = families[FAMILY_B]
            base["forced_scored_band_tflops"] = bands
            base["forced_scored_validated"] = 1.0
            # The scored grid itself passed every scored-case check.
            base["correctness_ok"] = 1.0
        except Exception as exc:  # noqa: BLE001
            base["forced_scored_validated"] = 0.0
            base["forced_scored_error"] = f"{type(exc).__name__}: {exc}"
        base["forced_scored_cases"] = [
            {k: row.get(k) for k in ("case_id", "family", "cell", "tflops", "median_us")}
            for row in (runtime_result.get("cases") or [])
            if isinstance(row, Mapping)
        ]
        if sanitizer_failed_gate is not None:
            base["memory_safety_ok"] = 0.0
        for gate in failures:
            V._mark_failed_gate(base, gate)
        base["forced_failed_gates"] = sorted(set(failures), key=V.GATE_NAMES.index)
        base["private_error"] = str(runtime_result.get("private_error") or "forced: a gate failed")
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
