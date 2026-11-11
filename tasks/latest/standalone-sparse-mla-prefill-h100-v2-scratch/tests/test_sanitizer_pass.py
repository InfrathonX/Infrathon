"""The Judge's compute-sanitizer pass: configuration, ordering, outcome classification, modes.

The pass runs in one process, sequentially, and the runtime stops it at the
first fatal CUDA error; nothing restarts it.  Only memcheck's own
invalid-access or misaligned reports fail memory safety.  A fatal error
without one is *inconclusive*: the case in flight and every later sanitizer
case run natively with guard bands instead.  Numbers computed under memcheck
never fail a case on their own (their verdict moves to the native run).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from conftest import load_module
from suite_fixtures import accuracy as _accuracy, build_edge_manifest, build_final_manifest


verifier = load_module("sanitizer_pass_verifier", "images/judge/verify_sparse_mla.py")
worker = load_module("sanitizer_pass_worker", "images/judge/candidate_worker.py")
CANDIDATE_SHA = "c" * 64


def _sanitize_cases() -> list[dict[str, Any]]:
    return [case for case in verifier.validate_edge_manifest(build_edge_manifest()) if case["sanitize"]]


def _ordered(cases=None) -> list[dict[str, Any]]:
    return list(verifier.sanitize_order(_sanitize_cases() if cases is None else cases))


def _row(case: dict[str, Any], status: str = "passed") -> dict[str, Any]:
    row = {"case_id": case["case_id"], "category": case["category"], "status": status, "ok": status == "passed"}
    if status == "passed":
        row.update(_accuracy())
    else:
        row["error"] = "RuntimeError: something"
    return row


def _result(rows: list[dict[str, Any]], status: str = "completed", *, sha: str = CANDIDATE_SHA,
            order: list[str] | None = None) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "candidate_sha256": sha,
        "order": order if order is not None else [case["case_id"] for case in _ordered()],
        "status": status,
        "cases": rows,
    }


def _passing_result() -> dict[str, Any]:
    return _result([_row(case) for case in _ordered()])


def _run(monkeypatch, tmp_path: Path, *, returncode: int | None, log: str, result: dict | None = None,
         hang: bool = False):
    result_dir = tmp_path / "sanitizer_result"
    result_dir.mkdir()
    cache_dir = tmp_path / "sanitizer_cache"
    cache_dir.mkdir()
    account = types.SimpleNamespace(pw_uid=os.getuid(), pw_gid=os.getgid())
    monkeypatch.setattr(verifier, "SANITIZER", Path(__file__))
    monkeypatch.setattr(verifier, "_prepare_run_dirs", lambda run_root, name: (account, result_dir, cache_dir))
    monkeypatch.setattr(verifier, "_candidate_environment", lambda cache: {})
    seen: dict[str, Any] = {}

    def fake_run(command, *, stdin_payload, run_root, log_name, env, timeout):
        seen.update(command=command, env=env, payload=json.loads(stdin_payload))
        (run_root / log_name).write_text(log, encoding="utf-8")
        if result is not None:
            path = result_dir / "result.json"
            path.write_text(json.dumps(result), encoding="utf-8")
            path.chmod(0o600)
        if hang:
            raise verifier.UnprivilegedTimeout("sanitizer timed out")
        return returncode

    monkeypatch.setattr(verifier, "_run_unprivileged", fake_run)
    record = verifier.run_sanitizer(
        tmp_path / "candidate.py", tmp_path, _sanitize_cases(), timeout=10, expected_sha256=CANDIDATE_SHA
    )
    return record, seen


def _large_id() -> str:
    return next(case["case_id"] for case in _ordered() if not case["native"])


def test_the_sanitizer_configuration_is_the_hard_safety_rule() -> None:
    arguments = verifier.SANITIZER_ARGUMENTS
    assert "--tool=memcheck" in arguments
    assert "--destroy-on-device-error=context" in arguments
    assert "--force-blocking-launches" in arguments
    padding = int(arguments[arguments.index("--padding") + 1])
    assert padding >= 4 << 20
    excludes = [arguments[i + 1] for i, item in enumerate(arguments) if item == "--kernel-name-exclude"]
    assert excludes == ["regex=^_Z", "kns=xmma", "kns=gemm", "kns=nvjet", "kns=cutlass", "kns=cublas"]
    assert verifier.SANITIZER_ACCESS_MARKERS == (
        "invalid __global__", "invalid __shared__", "invalid __local__", "misaligned",
    )


def test_clean_pass_is_accepted_and_runs_the_large_case_first(monkeypatch, tmp_path: Path) -> None:
    record, seen = _run(monkeypatch, tmp_path, returncode=0, log="========= ERROR SUMMARY: 0 errors\n",
                        result=_passing_result())
    assert record["status"] == "passed"
    assert record["deferred_case_ids"] == [] and record["promote_case_ids"] == []
    assert seen["env"]["PYTORCH_NO_CUDA_MEMORY_CACHING"] == "1"
    command = seen["command"]
    assert command[0] == str(verifier.SANITIZER)
    assert tuple(command[1 : 1 + len(verifier.SANITIZER_ARGUMENTS)]) == verifier.SANITIZER_ARGUMENTS
    assert command[command.index("--mode") + 1] == "sanitize"
    assert "--trials" not in command
    payload_cases = seen["payload"]["edge_cases"]
    assert [case["case_id"] for case in payload_cases] == [case["case_id"] for case in _ordered()]
    assert payload_cases[0]["native"] is False and payload_cases[0]["s_q"] == 70000
    assert seen["payload"]["mode"] == "sanitize" and seen["payload"]["edge_suite_id"] == "edge-holdout-v3"


@pytest.mark.parametrize("damage", ["missing_case", "reordered", "other_candidate", "bad_metric",
                                    "bad_status", "order"])
def test_a_completed_result_must_account_for_every_flagged_case(monkeypatch, tmp_path: Path, damage: str) -> None:
    result = _passing_result()
    if damage == "missing_case":
        result["cases"].pop()
    elif damage == "reordered":
        result["cases"].reverse()
    elif damage == "other_candidate":
        result["candidate_sha256"] = "d" * 64
    elif damage == "bad_metric":
        result["cases"][1]["out_cos_diff"] = 1.0
    elif damage == "bad_status":
        result["cases"][2]["status"] = "ok"
    else:
        result["order"] = list(reversed(result["order"]))
    record, _ = _run(monkeypatch, tmp_path, returncode=0, log="", result=result)
    assert record["status"] == "failed"
    assert record["failed_gate"] == "memory_safety_ok"


@pytest.mark.parametrize("report", [
    "========= Invalid __global__ read of size 2 bytes",
    "========= Invalid __shared__ write of size 16 bytes",
    "========= Invalid __local__ read of size 4 bytes",
    "========= Invalid __global__ read of size 16 bytes\n=========     Address 0x7f00 is misaligned",
    "========= Misaligned Shared or Local Address",
])
@pytest.mark.parametrize("returncode", [verifier.SANITIZER_EXIT_CODE, 1, 0])
def test_explicit_access_reports_are_memory_errors_whatever_the_exit_code_or_result(
    monkeypatch, tmp_path: Path, report: str, returncode: int
) -> None:
    log = report + "\ntorch.AcceleratorError: CUDA error: unspecified launch failure\n"
    record, _ = _run(monkeypatch, tmp_path, returncode=returncode, log=log, result=_passing_result())
    assert record["status"] == "memory_error"


def test_candidate_output_cannot_fake_or_hide_a_report() -> None:
    # Only memcheck's own "=========" lines count.
    assert not verifier.explicit_access_report("Invalid __global__ read (printed by the candidate)\n")
    assert verifier.explicit_access_report("========= Invalid __global__ write of size 4 bytes\n")


def test_a_tool_level_fatal_error_is_inconclusive_and_moves_the_rest_to_native(monkeypatch, tmp_path: Path) -> None:
    ordered = _ordered()
    rows = [_row(ordered[0]), _row(ordered[1], "deferred"), _row(ordered[2]), _row(ordered[3], "fatal")]
    log = (
        "SANITIZE_BEGIN 3 x\n"
        "========= Program hit cudaErrorInvalidPc (error 718) due to \"invalid program counter\"\n"
        "========= ERROR SUMMARY: 1 error\n"
    )
    record, _ = _run(monkeypatch, tmp_path, returncode=verifier.SANITIZER_EXIT_CODE, log=log,
                     result=_result(rows, "fatal"))
    assert record["status"] == "inconclusive"
    assert record["inconclusive_case_id"] == ordered[3]["case_id"]
    assert record["uncovered_case_ids"] == [case["case_id"] for case in ordered[3:]]
    assert record["deferred_case_ids"] == [ordered[1]["case_id"]]
    # The large case ran (and passed) first, so nothing sanitizer-only is promoted.
    assert ordered[0]["native"] is False and record["promote_case_ids"] == []


def test_an_inconclusive_large_case_is_promoted_to_the_native_run(monkeypatch, tmp_path: Path) -> None:
    ordered = _ordered()
    record, _ = _run(monkeypatch, tmp_path, returncode=verifier.SANITIZER_EXIT_CODE,
                     log="========= Program hit cudaErrorIllegalInstruction\n",
                     result=_result([_row(ordered[0], "fatal")], "fatal"))
    assert record["status"] == "inconclusive"
    assert record["promote_case_ids"] == [_large_id()]
    assert len(record["uncovered_case_ids"]) == len(ordered)


def test_numeric_failures_under_memcheck_are_deferred_not_failed(monkeypatch, tmp_path: Path) -> None:
    ordered = _ordered()
    rows = [_row(case, "deferred" if index in (0, 4) else "passed") for index, case in enumerate(ordered)]
    # A launch error is reported as a CUDA API error: exit code 86 without an access report.
    record, _ = _run(monkeypatch, tmp_path, returncode=verifier.SANITIZER_EXIT_CODE,
                     log="========= Program hit cudaErrorInvalidValue on CUDA API call to cuLaunchKernel\n",
                     result=_result(rows))
    assert record["status"] == "passed"
    assert record["deferred_case_ids"] == [ordered[0]["case_id"], ordered[4]["case_id"]]
    assert record["promote_case_ids"] == [_large_id()]


def test_a_process_that_dies_without_a_result_is_inconclusive_from_the_case_in_flight(
    monkeypatch, tmp_path: Path
) -> None:
    ordered = _ordered()
    log = "".join(
        f"SANITIZE_BEGIN {i} {case['case_id']}\nSANITIZE_END {i} {case['case_id']} passed\n"
        for i, case in enumerate(ordered[:5])
    ) + f"SANITIZE_BEGIN 5 {ordered[5]['case_id']}\nSegmentation fault\n"
    record, _ = _run(monkeypatch, tmp_path, returncode=139, log=log)
    assert record["status"] == "inconclusive"
    assert record["inconclusive_case_id"] == ordered[5]["case_id"]
    assert record["uncovered_case_ids"] == [case["case_id"] for case in ordered[5:]]


def test_a_dead_process_whose_markers_show_an_input_mutation_still_fails(monkeypatch, tmp_path: Path) -> None:
    ordered = _ordered()
    log = f"SANITIZE_BEGIN 0 {ordered[0]['case_id']}\nSANITIZE_END 0 {ordered[0]['case_id']} memory_safety\n"
    record, _ = _run(monkeypatch, tmp_path, returncode=1, log=log)
    assert record["status"] == "failed" and record["failed_gate"] == "memory_safety_ok"


def test_a_clean_exit_without_a_result_is_not_a_pass(monkeypatch, tmp_path: Path) -> None:
    record, _ = _run(monkeypatch, tmp_path, returncode=0, log="========= ERROR SUMMARY: 0 errors\n")
    assert record["status"] == "failed" and record["failed_gate"] == "memory_safety_ok"
    assert "no structured result" in record["private_error"]


@pytest.mark.parametrize("gate, expected", [
    ("memory_safety_ok", "memory_safety_ok"),
    ("correctness_ok", "correctness_ok"),
    ("harness_integrity_ok", "harness_integrity_ok"),
    ("suite_ok", "memory_safety_ok"),
    (None, "memory_safety_ok"),
])
def test_runtime_failures_inside_the_pass_keep_their_gate(monkeypatch, tmp_path: Path, gate, expected) -> None:
    result = {"schema_version": 3, "candidate_sha256": CANDIDATE_SHA, "cases": [], "status": "failed",
              "failed_gate": gate, "private_error": "sanitizer case edge-001: candidate modified its inputs"}
    record, _ = _run(monkeypatch, tmp_path, returncode=0, log="========= ERROR SUMMARY: 0 errors\n",
                     result=result)
    assert record["status"] == "failed" and record["failed_gate"] == expected


def test_a_pass_that_times_out_is_a_failure(monkeypatch, tmp_path: Path) -> None:
    record, _ = _run(monkeypatch, tmp_path, returncode=None, log="========= COMPUTE-SANITIZER\n", hang=True)
    assert record["status"] == "timeout"
    assert record["report"] == ["========= COMPUTE-SANITIZER"]


def test_a_hung_exit_after_a_fatal_error_is_inconclusive_not_a_timeout(monkeypatch, tmp_path: Path) -> None:
    ordered = _ordered()
    log = (f"SANITIZE_BEGIN 0 {ordered[0]['case_id']}\nSANITIZE_END 0 {ordered[0]['case_id']} passed\n"
           f"SANITIZE_BEGIN 1 {ordered[1]['case_id']}\nSANITIZE_END 1 {ordered[1]['case_id']} fatal\n")
    record, _ = _run(monkeypatch, tmp_path, returncode=None, log=log, hang=True)
    assert record["status"] == "inconclusive"
    assert record["inconclusive_case_id"] == ordered[1]["case_id"]


def test_run_sanitizer_detects_the_attach_failure(monkeypatch, tmp_path: Path) -> None:
    record, _ = _run(monkeypatch, tmp_path, returncode=0,
                     log="========= Error: No attachable process found. compute-sanitizer timed-out.\n")
    assert record["status"] == "unavailable"


def test_a_missing_sanitizer_binary_is_never_a_pass(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(verifier, "SANITIZER", tmp_path / "absent")
    with pytest.raises(RuntimeError, match="compute-sanitizer is missing"):
        verifier.run_sanitizer(tmp_path / "candidate.py", tmp_path, [], timeout=10.0)


# --------------------------------------------------------------------------
# evaluate(): the pass runs first, and its outcome decides whether and how native runs happen.
# --------------------------------------------------------------------------
def _evaluate_with(monkeypatch, tmp_path: Path, *, mode: str, sanitizer_record=None, worker_result=None):
    context = tmp_path / "context.json"
    context.write_text(json.dumps({"round_id": "agent-1"}))
    context.chmod(0o600)
    original_profile = verifier.load_evaluation_profile
    monkeypatch.setattr(verifier, "load_evaluation_profile",
                        lambda: original_profile(context, expected_uid=os.getuid()))
    final = json.dumps(build_final_manifest()).encode()
    edges = json.dumps(build_edge_manifest()).encode()
    final_cases = verifier.validate_final_manifest(json.loads(final))
    edge_cases = verifier.validate_edge_manifest(json.loads(edges))
    monkeypatch.setattr(verifier, "load_final_manifest", lambda: (final_cases, hashlib.sha256(final).hexdigest()))
    monkeypatch.setattr(verifier, "load_edge_manifest", lambda: (edge_cases, hashlib.sha256(edges).hexdigest()))
    monkeypatch.setattr(verifier, "verify_protected_hashes", lambda path: [])
    monkeypatch.setattr(verifier, "locate_submission", lambda path: tmp_path)
    candidate = tmp_path / "candidate"
    candidate.write_text("# CPU fixture")
    monkeypatch.setattr(verifier, "validate_submission", lambda root, override: {
        "ok": True, "final_version": 1, "num_candidate_versions": 1,
        "selected_path": str(candidate), "selected_sha256": CANDIDATE_SHA,
    })
    monkeypatch.setattr(verifier, "_trusted_hardware", lambda: {"count": 1})
    run_root = tmp_path / "run"
    run_root.mkdir()
    monkeypatch.setattr(verifier.tempfile, "mkdtemp", lambda **kwargs: str(run_root))
    calls: list[tuple[str, Any]] = []

    def native(source, root, cases, edges, **kwargs):
        calls.append(("worker", (tuple(cases), tuple(edges), kwargs)))
        if worker_result is not None:
            return worker_result
        return {
            "status": "passed", "candidate_sha256": CANDIDATE_SHA,
            "gates": {name: True for name in verifier.RUNTIME_GATES},
            "edge": {"cases_total": len(edge_cases) - 1, "cases_failed": 0},
        }

    monkeypatch.setattr(verifier, "run_worker", native)
    monkeypatch.setattr(verifier, "validate_runtime_result",
                        lambda *a, **k: (100.0, {family: 100.0 for family in verifier.FAMILIES},
                                         {"tiers": {}, "groups": {}}))

    def sanitizer(source, root, cases, **kwargs):
        calls.append(("sanitizer", (tuple(cases), kwargs)))
        return sanitizer_record

    monkeypatch.setattr(verifier, "run_sanitizer", sanitizer)
    args = verifier.parse_args(["--app-dir", str(tmp_path), "--sanitizer-mode", mode])
    rc, detail = verifier.evaluate(args)
    return rc, detail, calls, edge_cases


def test_required_mode_runs_the_sanitizer_first_and_records_it(monkeypatch, tmp_path: Path) -> None:
    rc, detail, calls, edge_cases = _evaluate_with(
        monkeypatch, tmp_path, mode="required",
        sanitizer_record={"status": "passed", "promote_case_ids": []},
    )
    assert rc == 0 and [name for name, _ in calls] == ["sanitizer", "worker"]
    sanitized, sanitizer_kwargs = calls[0][1]
    assert sanitized == tuple(case for case in edge_cases if case["sanitize"])
    assert len(sanitized) == verifier.EDGE_SANITIZE_COUNT
    assert sanitizer_kwargs["expected_sha256"] == CANDIDATE_SHA
    # The native worker receives the whole edge suite and no promoted case.
    assert calls[1][1][1] == edge_cases and calls[1][1][2]["promoted"] == []
    assert detail["sanitizer_ran"] == 1.0 and detail["sanitizer_inconclusive"] == 0.0
    assert detail["memory_safety_ok"] == 1.0
    assert detail["edge_cases_total"] == float(verifier.EDGE_CASE_COUNT)
    assert detail["edge_sanitize_cases"] == float(verifier.EDGE_SANITIZE_COUNT)
    assert detail["sanitizer_mode"] == "required"
    assert detail["score_tflops"] == 100.0 and detail["v4_flash_tflops"] == 100.0
    assert detail["glm5_tflops"] == 100.0


def test_an_inconclusive_pass_continues_natively_with_the_promoted_cases(monkeypatch, tmp_path: Path) -> None:
    large = next(case["case_id"] for case in verifier.validate_edge_manifest(build_edge_manifest())
                 if not case["native"])
    rc, detail, calls, _ = _evaluate_with(
        monkeypatch, tmp_path, mode="required",
        sanitizer_record={"status": "inconclusive", "promote_case_ids": [large]},
    )
    assert rc == 0 and [name for name, _ in calls] == ["sanitizer", "worker"]
    assert calls[1][1][2]["promoted"] == [large]
    assert detail["sanitizer_ran"] == 1.0 and detail["sanitizer_inconclusive"] == 1.0


@pytest.mark.parametrize("status, message", [
    ("timeout", "time limit"),
    ("memory_error", "invalid memory accesses"),
    ("unavailable", "could not attach"),
])
def test_failed_passes_stop_the_judge_before_any_native_call(monkeypatch, tmp_path: Path, status, message) -> None:
    rc, detail, calls, _ = _evaluate_with(monkeypatch, tmp_path, mode="required",
                                          sanitizer_record={"status": status, "report": []})
    assert rc == 1 and [name for name, _ in calls] == ["sanitizer"]
    assert detail["failed_gate"] == "memory_safety_ok"
    assert detail["failed_gate_code"] == float(verifier.GATE_NAMES.index("memory_safety_ok") + 1)
    assert message in detail["private_error"]
    assert detail["sanitizer_ran"] == 0.0 and detail["correctness_ok"] == 0.0


def test_a_failure_inside_the_pass_keeps_its_own_gate(monkeypatch, tmp_path: Path) -> None:
    record = {"status": "failed", "failed_gate": "harness_integrity_ok",
              "private_error": "sanitizer case edge-001: time.perf_counter was replaced"}
    rc, detail, calls, _ = _evaluate_with(monkeypatch, tmp_path, mode="required", sanitizer_record=record)
    assert rc == 1 and [name for name, _ in calls] == ["sanitizer"]
    assert detail["failed_gate"] == "harness_integrity_ok" and "perf_counter" in detail["private_error"]


def test_platform_mode_skips_the_pass_and_says_so(monkeypatch, tmp_path: Path) -> None:
    rc, detail, calls, edge_cases = _evaluate_with(monkeypatch, tmp_path, mode="skip-platform-incompatible")
    assert rc == 0 and [name for name, _ in calls] == ["worker"]
    assert calls[0][1][1] == edge_cases and calls[0][1][2]["promoted"] == []
    assert detail["sanitizer_ran"] == 0.0
    assert detail["sanitizer"]["status"] == "not_run"
    assert detail["memory_safety_ok"] == 1.0


def test_unknown_sanitizer_mode_fails_closed(monkeypatch, tmp_path: Path) -> None:
    rc, detail, calls, _ = _evaluate_with(monkeypatch, tmp_path, mode="off")
    assert rc == 1 and calls == []
    assert detail["private_error"] == "invalid compute-sanitizer mode"


def test_native_edge_failure_reports_counts_against_the_private_manifest(monkeypatch, tmp_path: Path) -> None:
    runtime_failure = {
        "status": "failed", "candidate_sha256": CANDIDATE_SHA, "failed_gate": "edge_correctness_ok",
        "private_error": "7 of 469 edge cases failed; first: edge-007-upstream-correctness",
        "gates": {"edge_correctness_ok": False, "memory_safety_ok": None,
                  "harness_integrity_ok": None, "correctness_ok": None},
        "edge": {"cases_total": 469, "cases_skipped_sanitize_only": 1, "cases_passed": 462,
                 "cases_failed": 7, "results": []},
    }
    rc, detail, calls, _ = _evaluate_with(monkeypatch, tmp_path, mode="required",
                                          sanitizer_record={"status": "passed", "promote_case_ids": []},
                                          worker_result=runtime_failure)
    assert rc == 1 and [name for name, _ in calls] == ["sanitizer", "worker"]
    assert detail["failed_gate"] == "edge_correctness_ok"
    assert detail["edge_cases_failed"] == 7.0
    assert detail["edge_cases_total"] == float(verifier.EDGE_CASE_COUNT)
    assert detail["private_runtime_result"] == runtime_failure


def test_a_runtime_result_from_another_candidate_is_rejected(monkeypatch, tmp_path: Path) -> None:
    rc, detail, _calls, _ = _evaluate_with(
        monkeypatch, tmp_path, mode="skip-platform-incompatible",
        worker_result={"status": "passed", "candidate_sha256": "d" * 64,
                       "gates": {name: True for name in verifier.RUNTIME_GATES}, "edge": {}},
    )
    assert rc == 1 and "different candidate source" in detail["private_error"]
    assert detail["score_tflops"] == 0.0


# --------------------------------------------------------------------------
# candidate_worker.py --mode sanitize.
# --------------------------------------------------------------------------
def _run_worker_main(monkeypatch, tmp_path: Path, manifest: dict, argv: list[str]):
    calls: list[tuple[str, Any]] = []

    def sanitize(candidate, cases, *, progress):
        progress("SANITIZE_BEGIN 0 x")
        calls.append(("sanitize", cases))
        return {"status": "completed"}

    fake_runtime = types.SimpleNamespace(
        run_sanitize_cases=sanitize,
        run_suite=lambda candidate, cases, edges, **kwargs: calls.append(("suite", (cases, edges, kwargs)))
        or {"status": "passed"},
    )
    monkeypatch.setattr(worker, "_load_trusted_runtime", lambda: fake_runtime, raising=False)
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(json.dumps(manifest).encode())))
    output = tmp_path / "result.json"
    rc = worker.main([*argv, "--candidate", str(tmp_path / "candidate.py"), "--output", str(output)])
    return rc, calls, output


def test_worker_sanitize_mode_runs_only_the_flagged_cases_and_prints_progress(monkeypatch, tmp_path, capsys) -> None:
    cases = _ordered()
    manifest = {"schema_version": 3, "mode": "sanitize", "edge_suite_id": "edge-holdout-v3", "edge_cases": cases}
    rc, calls, output = _run_worker_main(monkeypatch, tmp_path, manifest, ["--mode", "sanitize"])
    assert rc == 0 and calls == [("sanitize", cases)]
    assert json.loads(output.read_text()) == {"status": "completed"}
    assert "SANITIZE_BEGIN 0 x" in capsys.readouterr().out


@pytest.mark.parametrize("damage", ["unflagged_case", "mode_mismatch", "public_suite", "schema", "promoted"])
def test_worker_sanitize_mode_refuses_mismatched_manifests(monkeypatch, tmp_path: Path, damage: str) -> None:
    cases = _ordered()
    manifest = {"schema_version": 3, "mode": "sanitize", "edge_suite_id": "edge-holdout-v3", "edge_cases": cases}
    if damage == "unflagged_case":
        cases[0]["sanitize"] = False
    elif damage == "mode_mismatch":
        manifest["mode"] = "suite"
    elif damage == "public_suite":
        manifest["edge_suite_id"] = "edge-public-v3"
    elif damage == "promoted":
        manifest["promoted_sanitize_only"] = [cases[0]["case_id"]]
    else:
        manifest["schema_version"] = 1
    with pytest.raises(ValueError):
        _run_worker_main(monkeypatch, tmp_path, manifest, ["--mode", "sanitize"])


def test_worker_keeps_a_failing_result_with_non_finite_metrics(monkeypatch, tmp_path: Path) -> None:
    failing = {"status": "failed", "failed_gate": "edge_correctness_ok",
               "failure_detail": {"kind": "cosine", "out_cos_diff": float("inf"), "out_max_abs": float("nan")}}
    fake_runtime = types.SimpleNamespace(run_sanitize_cases=lambda candidate, cases, progress: failing)
    monkeypatch.setattr(worker, "_load_trusted_runtime", lambda: fake_runtime, raising=False)
    manifest = {"schema_version": 3, "mode": "sanitize", "edge_suite_id": "edge-holdout-v3",
                "edge_cases": _ordered()}
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(json.dumps(manifest).encode())))
    output = tmp_path / "result.json"
    assert worker.main(["--mode", "sanitize", "--candidate", str(tmp_path / "c.py"), "--output", str(output)]) == 0
    written = verifier._strict_json_bytes(output.read_bytes(), "worker result")
    assert written["failed_gate"] == "edge_correctness_ok"
    assert written["failure_detail"] == {"kind": "cosine", "out_cos_diff": "inf", "out_max_abs": "nan"}
