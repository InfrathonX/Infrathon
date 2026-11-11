"""The Judge's compute-sanitizer pass: ordering, outcome classification, deployment modes."""

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
from test_scoring import build_edge_manifest, build_final_manifest


verifier = load_module("sanitizer_pass_verifier", "images/judge/verify_paged_gqa.py")
worker = load_module("sanitizer_pass_worker", "images/judge/candidate_worker.py")
CANDIDATE_SHA = "c" * 64


def _sanitize_cases() -> list[dict[str, Any]]:
    return [case for case in verifier.validate_edge_manifest(build_edge_manifest()) if case["sanitize"]]


def _passing_result(cases, *, sha: str = CANDIDATE_SHA) -> dict[str, Any]:
    return {
        "schema_version": verifier.SCHEMA_VERSION,
        "candidate_sha256": sha,
        "status": "passed",
        "cases": [
            {
                "case_id": case["case_id"], "category": case["category"], "ok": True,
                "out_max_abs": 0.0, "out_max_abs_limit": 1.0e-5, "out_mean_abs": 0.0,
                "out_mean_abs_limit": 0.0, "out_norm_l2": 0.0,
            }
            for case in cases
        ],
    }


def _run(monkeypatch, tmp_path: Path, *, returncode: int, log: str, result: dict | None = None,
         cases: list[dict[str, Any]] | None = None):
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
        return returncode

    monkeypatch.setattr(verifier, "_run_unprivileged", fake_run)
    cases = _sanitize_cases() if cases is None else cases
    record = verifier.run_sanitizer(
        tmp_path / "candidate.py", tmp_path, cases, timeout=10, expected_sha256=CANDIDATE_SHA
    )
    return record, seen


def test_clean_pass_is_accepted(monkeypatch, tmp_path: Path) -> None:
    cases = _sanitize_cases()
    record, seen = _run(monkeypatch, tmp_path, returncode=0, log="========= ERROR SUMMARY: 0 errors\n",
                        result=_passing_result(cases), cases=cases)
    assert record["status"] == "passed"
    assert seen["env"]["PYTORCH_NO_CUDA_MEMORY_CACHING"] == "1"
    command = seen["command"]
    assert command[0] == str(verifier.SANITIZER)
    assert tuple(command[1 : 1 + len(verifier.SANITIZER_ARGUMENTS)]) == verifier.SANITIZER_ARGUMENTS
    assert "--tool=memcheck" in command and f"--error-exitcode={verifier.SANITIZER_EXIT_CODE}" in command
    assert command[command.index("--mode") + 1] == "sanitize"
    assert "--trials" not in command
    assert seen["payload"] == {
        "schema_version": 4, "mode": "sanitize", "edge_suite_id": "edge-holdout-v4", "edge_cases": cases,
    }
    # The sanitizer-only large-offset case is part of the pass.
    assert any(not case["native"] for case in seen["payload"]["edge_cases"])


@pytest.mark.parametrize("damage", ["missing_case", "reordered", "not_ok", "other_candidate", "over_limit"])
def test_a_passing_result_must_account_for_every_flagged_case(monkeypatch, tmp_path: Path, damage: str) -> None:
    cases = _sanitize_cases()
    result = _passing_result(cases)
    if damage == "missing_case":
        result["cases"].pop()
    elif damage == "reordered":
        result["cases"].reverse()
    elif damage == "not_ok":
        result["cases"][2]["ok"] = False
    elif damage == "other_candidate":
        result["candidate_sha256"] = "d" * 64
    else:
        result["cases"][1]["out_norm_l2"] = 1.0
    record, _ = _run(monkeypatch, tmp_path, returncode=0, log="", result=result, cases=cases)
    assert record["status"] == "failed"
    assert record["failed_gate"] == "memory_safety_ok"
    assert record["private_error"].startswith("sanitizer result rejected")


@pytest.mark.parametrize("returncode", [verifier.SANITIZER_EXIT_CODE, 1])
def test_invalid_access_is_a_memory_error_whatever_the_exit_code(monkeypatch, tmp_path: Path, returncode: int) -> None:
    log = (
        "========= Invalid __global__ read of size 2 bytes\n"
        "=========     Address 0x7f0a5a5a5a5a is out of bounds\n"
        "torch.AcceleratorError: CUDA error: unspecified launch failure\n"
    )
    record, _ = _run(monkeypatch, tmp_path, returncode=returncode, log=log)
    assert record["status"] == "memory_error"
    assert any("Invalid __global__" in line for line in record["report"])


def test_application_failure_without_invalid_access_still_fails(monkeypatch, tmp_path: Path) -> None:
    record, _ = _run(
        monkeypatch, tmp_path, returncode=1,
        log="========= Target application returned an error\n========= ERROR SUMMARY: 0 errors\n",
    )
    assert record["status"] == "failed"


def test_a_clean_exit_without_a_result_is_not_a_pass(monkeypatch, tmp_path: Path) -> None:
    record, _ = _run(monkeypatch, tmp_path, returncode=0, log="========= ERROR SUMMARY: 0 errors\n")
    assert record["status"] == "failed" and record["failed_gate"] == "memory_safety_ok"
    assert "no structured result" in record["private_error"]


@pytest.mark.parametrize("gate, expected", [
    ("edge_correctness_ok", "edge_correctness_ok"),
    ("correctness_ok", "correctness_ok"),
    ("harness_integrity_ok", "harness_integrity_ok"),
    ("suite_ok", "memory_safety_ok"),
    (None, "memory_safety_ok"),
])
def test_runtime_failures_inside_the_pass_keep_their_gate(monkeypatch, tmp_path: Path, gate, expected) -> None:
    result = {"schema_version": verifier.SCHEMA_VERSION, "candidate_sha256": CANDIDATE_SHA, "cases": [], "status": "failed",
              "failed_gate": gate, "private_error": "sanitizer case edge-007-nan_slots: out contains NaN"}
    record, _ = _run(monkeypatch, tmp_path, returncode=0, log="========= ERROR SUMMARY: 0 errors\n",
                     result=result)
    assert record["status"] == "failed" and record["failed_gate"] == expected


def test_a_pass_that_times_out_is_a_memory_safety_failure_not_a_correctness_failure(
    monkeypatch, tmp_path: Path
) -> None:
    result_dir = tmp_path / "sanitizer_result"
    result_dir.mkdir()
    cache_dir = tmp_path / "sanitizer_cache"
    cache_dir.mkdir()
    account = types.SimpleNamespace(pw_uid=os.getuid(), pw_gid=os.getgid())
    monkeypatch.setattr(verifier, "SANITIZER", Path(__file__))
    monkeypatch.setattr(verifier, "_prepare_run_dirs", lambda run_root, name: (account, result_dir, cache_dir))
    monkeypatch.setattr(verifier, "_candidate_environment", lambda cache: {})

    def hung(command, *, stdin_payload, run_root, log_name, env, timeout):
        (run_root / log_name).write_text("========= COMPUTE-SANITIZER\n", encoding="utf-8")
        raise verifier.UnprivilegedTimeout("sanitizer timed out")

    monkeypatch.setattr(verifier, "_run_unprivileged", hung)
    record = verifier.run_sanitizer(tmp_path / "candidate.py", tmp_path, [{"case_id": "x"}], timeout=10)
    assert record["status"] == "timeout"
    assert record["report"] == ["========= COMPUTE-SANITIZER"]
    # The candidate worker's own timeout is still an ordinary runtime error.
    assert issubclass(verifier.UnprivilegedTimeout, RuntimeError)


def test_run_sanitizer_detects_the_attach_failure(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(verifier, "SANITIZER", tmp_path / "compute-sanitizer")
    (tmp_path / "compute-sanitizer").write_text("")
    monkeypatch.setattr(verifier, "_prepare_run_dirs", lambda run_root, name: (None, tmp_path, tmp_path))
    monkeypatch.setattr(verifier, "_candidate_environment", lambda cache: {})

    def fake_run(command, *, run_root, log_name, **kwargs):
        (run_root / log_name).write_text("========= Error: No attachable process found. compute-sanitizer timed-out.\n")
        return 0

    monkeypatch.setattr(verifier, "_run_unprivileged", fake_run)
    record = verifier.run_sanitizer(tmp_path / "candidate.py", tmp_path, [], timeout=10.0)
    assert record["status"] == "unavailable"


def test_a_missing_sanitizer_binary_is_never_a_pass(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(verifier, "SANITIZER", tmp_path / "absent")
    with pytest.raises(RuntimeError, match="compute-sanitizer is missing"):
        verifier.run_sanitizer(tmp_path / "candidate.py", tmp_path, [], timeout=10.0)


# --------------------------------------------------------------------------
# evaluate(): the pass runs first, and its outcome decides whether native runs happen.
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
                        lambda *a, **k: (100.0, {"decode": 100.0, "spec": 100.0, "long": 100.0, "stream": 100.0}))

    def sanitizer(source, root, cases, **kwargs):
        calls.append(("sanitizer", (tuple(cases), kwargs)))
        return sanitizer_record

    monkeypatch.setattr(verifier, "run_sanitizer", sanitizer)
    args = verifier.parse_args(["--app-dir", str(tmp_path), "--sanitizer-mode", mode])
    rc, detail = verifier.evaluate(args)
    return rc, detail, calls, edge_cases


def test_required_mode_runs_the_sanitizer_first_and_records_it(monkeypatch, tmp_path: Path) -> None:
    rc, detail, calls, edge_cases = _evaluate_with(monkeypatch, tmp_path, mode="required",
                                                   sanitizer_record={"status": "passed"})
    assert rc == 0 and [name for name, _ in calls] == ["sanitizer", "worker"]
    sanitized, sanitizer_kwargs = calls[0][1]
    assert sanitized == tuple(case for case in edge_cases if case["sanitize"])
    assert len(sanitized) == verifier.EDGE_SANITIZE_COUNT
    assert sanitizer_kwargs["expected_sha256"] == CANDIDATE_SHA
    # The native worker receives the whole edge suite; the runtime skips the
    # sanitizer-only case itself and reports it as skipped.
    assert calls[1][1][1] == edge_cases
    assert detail["sanitizer_ran"] == 1.0 and detail["memory_safety_ok"] == 1.0
    assert detail["edge_cases_total"] == float(verifier.EDGE_CASE_COUNT)
    assert detail["edge_sanitize_cases"] == float(verifier.EDGE_SANITIZE_COUNT)
    assert detail["sanitizer_mode"] == "required"


def test_required_mode_timeout_fails_memory_safety_before_any_native_call(monkeypatch, tmp_path: Path) -> None:
    rc, detail, calls, _ = _evaluate_with(monkeypatch, tmp_path, mode="required",
                                          sanitizer_record={"status": "timeout", "report": []})
    assert rc == 1 and [name for name, _ in calls] == ["sanitizer"]
    assert detail["failed_gate"] == "memory_safety_ok"
    assert detail["correctness_ok"] == 0.0
    assert "time limit" in detail["private_error"]
    assert detail["sanitizer_ran"] == 0.0


def test_invalid_access_stops_the_judge_before_the_native_worker(monkeypatch, tmp_path: Path) -> None:
    rc, detail, calls, _ = _evaluate_with(
        monkeypatch, tmp_path, mode="required",
        sanitizer_record={"status": "memory_error", "report": ["========= Invalid __global__ write"]},
    )
    assert rc == 1 and [name for name, _ in calls] == ["sanitizer"]
    assert detail["failed_gate"] == "memory_safety_ok"
    assert detail["failed_gate_code"] == float(verifier.GATE_NAMES.index("memory_safety_ok") + 1)
    assert "invalid memory accesses" in detail["private_error"]
    assert detail["edge_cases_failed"] == 0.0


def test_wrong_answer_under_the_sanitizer_keeps_its_own_gate(monkeypatch, tmp_path: Path) -> None:
    record = {"status": "failed", "failed_gate": "edge_correctness_ok",
              "private_error": "sanitizer case edge-002-group_seven: out max abs error 3 exceeds limit"}
    rc, detail, calls, _ = _evaluate_with(monkeypatch, tmp_path, mode="required", sanitizer_record=record)
    assert rc == 1 and [name for name, _ in calls] == ["sanitizer"]
    assert detail["failed_gate"] == "edge_correctness_ok"
    assert detail["memory_safety_ok"] == 0.0 and "exceeds limit" in detail["private_error"]
    # The pass stops at its first wrong answer; the agent sees at least one failure.
    assert detail["edge_cases_failed"] == 1.0


def test_required_mode_fails_closed_when_the_sanitizer_cannot_attach(monkeypatch, tmp_path: Path) -> None:
    rc, detail, calls, _ = _evaluate_with(monkeypatch, tmp_path, mode="required",
                                          sanitizer_record={"status": "unavailable", "report": []})
    assert rc == 1 and [name for name, _ in calls] == ["sanitizer"]
    assert detail["failed_gate"] == "memory_safety_ok"
    assert "could not attach" in detail["private_error"]
    assert "skip-platform-incompatible" in detail["private_error"]


def test_platform_mode_skips_the_pass_and_says_so(monkeypatch, tmp_path: Path) -> None:
    rc, detail, calls, edge_cases = _evaluate_with(monkeypatch, tmp_path, mode="skip-platform-incompatible")
    assert rc == 0 and [name for name, _ in calls] == ["worker"]
    assert calls[0][1][1] == edge_cases
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
        "private_error": "7 of 159 edge cases failed; first: edge-007-nan_slots",
        "gates": {"edge_correctness_ok": False, "memory_safety_ok": None,
                  "harness_integrity_ok": None, "correctness_ok": None},
        "edge": {"cases_total": 159, "cases_skipped_sanitize_only": 1, "cases_passed": 152,
                 "cases_failed": 7, "results": []},
    }
    rc, detail, calls, _ = _evaluate_with(monkeypatch, tmp_path, mode="required",
                                          sanitizer_record={"status": "passed"},
                                          worker_result=runtime_failure)
    assert rc == 1 and [name for name, _ in calls] == ["sanitizer", "worker"]
    assert detail["failed_gate"] == "edge_correctness_ok"
    assert detail["edge_cases_failed"] == 7.0
    # The total stays the private manifest's size, not the runtime's native count.
    assert detail["edge_cases_total"] == float(verifier.EDGE_CASE_COUNT)
    assert detail["sanitizer_ran"] == 1.0
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
    fake_runtime = types.SimpleNamespace(
        run_sanitize_cases=lambda candidate, cases: calls.append(("sanitize", cases)) or {"status": "passed"},
        run_suite=lambda candidate, cases, edges, **kwargs: calls.append(("suite", (cases, edges, kwargs)))
        or {"status": "passed"},
    )
    monkeypatch.setattr(worker, "_load_trusted_runtime", lambda: fake_runtime, raising=False)
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(json.dumps(manifest).encode())))
    output = tmp_path / "result.json"
    rc = worker.main([*argv, "--candidate", str(tmp_path / "candidate.py"), "--output", str(output)])
    return rc, calls, output


def test_worker_sanitize_mode_runs_only_the_flagged_cases(monkeypatch, tmp_path: Path) -> None:
    cases = _sanitize_cases()
    manifest = {"schema_version": 4, "mode": "sanitize", "edge_suite_id": "edge-holdout-v4", "edge_cases": cases}
    rc, calls, output = _run_worker_main(monkeypatch, tmp_path, manifest, ["--mode", "sanitize"])
    assert rc == 0 and calls == [("sanitize", cases)]
    assert json.loads(output.read_text()) == {"status": "passed"}


@pytest.mark.parametrize("damage", ["unflagged_case", "mode_mismatch", "public_suite", "schema"])
def test_worker_sanitize_mode_refuses_mismatched_manifests(monkeypatch, tmp_path: Path, damage: str) -> None:
    cases = _sanitize_cases()
    manifest = {"schema_version": 4, "mode": "sanitize", "edge_suite_id": "edge-holdout-v4", "edge_cases": cases}
    if damage == "unflagged_case":
        cases[0]["sanitize"] = False
    elif damage == "mode_mismatch":
        manifest["mode"] = "suite"
    elif damage == "public_suite":
        manifest["edge_suite_id"] = "edge-public-v2"
    else:
        manifest["schema_version"] = 1
    with pytest.raises(ValueError):
        _run_worker_main(monkeypatch, tmp_path, manifest, ["--mode", "sanitize"])


def test_worker_keeps_a_failing_result_with_non_finite_metrics(monkeypatch, tmp_path: Path) -> None:
    failing = {"status": "failed", "failed_gate": "edge_correctness_ok",
               "failure_detail": {"kind": "norm_l2", "out_norm_l2": float("inf"), "out_max_abs": float("nan")}}
    fake_runtime = types.SimpleNamespace(run_sanitize_cases=lambda candidate, cases: failing)
    monkeypatch.setattr(worker, "_load_trusted_runtime", lambda: fake_runtime, raising=False)
    manifest = {"schema_version": 4, "mode": "sanitize", "edge_suite_id": "edge-holdout-v4",
                "edge_cases": _sanitize_cases()}
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(json.dumps(manifest).encode())))
    output = tmp_path / "result.json"
    assert worker.main(["--mode", "sanitize", "--candidate", str(tmp_path / "c.py"), "--output", str(output)]) == 0
    written = verifier._strict_json_bytes(output.read_bytes(), "worker result")
    assert written["failed_gate"] == "edge_correctness_ok"
    assert written["failure_detail"] == {"kind": "norm_l2", "out_norm_l2": "inf", "out_max_abs": "nan"}
