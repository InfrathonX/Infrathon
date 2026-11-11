"""The Judge's compute-sanitizer pass classifies outcomes from the exit code and the report."""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from conftest import load_module, judge_only_file


verifier = load_module("sanitizer_pass_verifier", "images/judge/verify_wgrad.py")


def _run(monkeypatch, tmp_path: Path, *, returncode: int, log: str, result: str | None = None):
    result_dir = tmp_path / "sanitizer_result"
    result_dir.mkdir()
    cache_dir = tmp_path / "sanitizer_cache"
    cache_dir.mkdir()
    account = types.SimpleNamespace(pw_uid=__import__("os").getuid(), pw_gid=__import__("os").getgid())
    monkeypatch.setattr(verifier, "SANITIZER", Path(__file__))
    monkeypatch.setattr(verifier, "_prepare_run_dirs", lambda run_root, name: (account, result_dir, cache_dir))
    monkeypatch.setattr(verifier, "_candidate_environment", lambda cache: {})
    seen = {}

    def fake_run(command, *, stdin_payload, run_root, log_name, env, timeout):
        seen["command"] = command
        seen["env"] = env
        (run_root / log_name).write_text(log, encoding="utf-8")
        if result is not None:
            path = result_dir / "result.json"
            path.write_text(result, encoding="utf-8")
            path.chmod(0o600)
        return returncode

    monkeypatch.setattr(verifier, "_run_unprivileged", fake_run)
    record = verifier.run_sanitizer(tmp_path / "candidate.py", tmp_path, [{"case_id": "x"}], timeout=10)
    return record, seen


def test_clean_pass_is_accepted(monkeypatch, tmp_path: Path) -> None:
    record, seen = _run(
        monkeypatch, tmp_path, returncode=0,
        log="========= ERROR SUMMARY: 0 errors\n",
        result='{"schema_version": 4, "status": "passed", "cases": []}',
    )
    assert record["status"] == "passed"
    assert seen["env"]["PYTORCH_NO_CUDA_MEMORY_CACHING"] == "1"
    assert seen["command"][0] == str(verifier.SANITIZER)
    assert "--tool=memcheck" in seen["command"]
    assert "--mode" in seen["command"] and "sanitize" in seen["command"]


@pytest.mark.parametrize("returncode", [verifier.SANITIZER_EXIT_CODE, 1])
def test_invalid_access_is_a_memory_error_whatever_the_exit_code(monkeypatch, tmp_path: Path, returncode: int) -> None:
    log = (
        "========= Invalid __global__ write of size 16 bytes\n"
        "=========     and is 1281 bytes after the nearest allocation\n"
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


def test_a_pass_that_times_out_is_a_memory_safety_failure_not_a_correctness_failure(
    monkeypatch, tmp_path: Path
) -> None:
    result_dir = tmp_path / "sanitizer_result"
    result_dir.mkdir()
    cache_dir = tmp_path / "sanitizer_cache"
    cache_dir.mkdir()
    account = types.SimpleNamespace(pw_uid=__import__("os").getuid(), pw_gid=__import__("os").getgid())
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


def _evaluate_with(monkeypatch, tmp_path: Path, *, mode: str, sanitizer_record=None):
    """Drive evaluate() past a passing runtime result to the sanitizer decision."""
    import hashlib
    import json
    import os

    root = Path(__file__).resolve().parents[1]
    context = tmp_path / "context.json"
    context.write_text(json.dumps({"round_id": "agent-1"}))
    context.chmod(0o600)
    original_profile = verifier.load_evaluation_profile
    monkeypatch.setattr(verifier, "load_evaluation_profile",
                        lambda: original_profile(context, expected_uid=os.getuid()))
    raw = judge_only_file("judge/private/final_cases.json").read_bytes()
    raw_edges = judge_only_file("judge/private/edge_cases.json").read_bytes()
    monkeypatch.setattr(verifier, "load_final_manifest",
                        lambda: (tuple(json.loads(raw)["cases"]), hashlib.sha256(raw).hexdigest()))
    monkeypatch.setattr(verifier, "load_edge_manifest",
                        lambda: (tuple(json.loads(raw_edges)["cases"]), hashlib.sha256(raw_edges).hexdigest()))
    monkeypatch.setattr(verifier, "verify_protected_hashes", lambda path: [])
    monkeypatch.setattr(verifier, "locate_submission", lambda path: tmp_path)
    candidate = tmp_path / "candidate"
    candidate.write_text("# CPU fixture")
    monkeypatch.setattr(verifier, "validate_submission", lambda root, override: {
        "ok": True, "final_version": 1, "num_candidate_versions": 1,
        "selected_path": str(candidate), "selected_sha256": "c" * 64,
    })
    monkeypatch.setattr(verifier, "_trusted_hardware", lambda: {"count": 1})
    run_root = tmp_path / "run"
    run_root.mkdir()
    monkeypatch.setattr(verifier.tempfile, "mkdtemp", lambda **kwargs: str(run_root))
    gates = {name: True for name in verifier.RUNTIME_GATES}
    calls = []

    def worker(*args, **kwargs):
        calls.append("worker")
        return {
            "status": "passed", "candidate_sha256": "c" * 64, "gates": gates,
            "edge": {"cases_total": 166, "cases_failed": 0},
        }

    monkeypatch.setattr(verifier, "run_worker", worker)
    monkeypatch.setattr(verifier, "validate_runtime_result",
                        lambda *a, **k: (100.0, {family: 100.0 for family in verifier.FAMILIES}))

    def sanitizer(*args, **kwargs):
        calls.append("sanitizer")
        return sanitizer_record

    monkeypatch.setattr(verifier, "run_sanitizer", sanitizer)
    args = verifier.parse_args(["--app-dir", str(tmp_path), "--sanitizer-mode", mode])
    rc, detail = verifier.evaluate(args)
    return rc, detail, calls


def test_required_mode_runs_the_sanitizer_first_and_records_it(monkeypatch, tmp_path: Path) -> None:
    rc, detail, calls = _evaluate_with(monkeypatch, tmp_path, mode="required",
                                       sanitizer_record={"status": "passed"})
    assert rc == 0 and calls == ["sanitizer", "worker"]
    assert detail["sanitizer_ran"] == 1.0 and detail["memory_safety_ok"] == 1.0


def test_required_mode_timeout_fails_memory_safety_before_any_native_call(monkeypatch, tmp_path: Path) -> None:
    rc, detail, calls = _evaluate_with(monkeypatch, tmp_path, mode="required",
                                       sanitizer_record={"status": "timeout", "report": []})
    assert rc == 1 and calls == ["sanitizer"]
    assert detail["failed_gate"] == "memory_safety_ok"
    assert detail["correctness_ok"] == 0.0
    assert "time limit" in detail["private_error"]
    assert detail["sanitizer_ran"] == 0.0


def test_invalid_access_stops_the_judge_before_the_native_worker(monkeypatch, tmp_path: Path) -> None:
    rc, detail, calls = _evaluate_with(monkeypatch, tmp_path, mode="required",
                                       sanitizer_record={"status": "memory_error", "report": ["========= Invalid __global__ write"]})
    assert rc == 1 and calls == ["sanitizer"]
    assert detail["failed_gate"] == "memory_safety_ok"
    assert "invalid memory accesses" in detail["private_error"]


def test_wrong_answer_under_the_sanitizer_keeps_its_own_gate(monkeypatch, tmp_path: Path) -> None:
    record = {"status": "failed", "failed_gate": "edge_correctness_ok",
              "private_error": "sanitizer case edge-001: element mismatch"}
    rc, detail, calls = _evaluate_with(monkeypatch, tmp_path, mode="required", sanitizer_record=record)
    assert rc == 1 and calls == ["sanitizer"]
    assert detail["failed_gate"] == "edge_correctness_ok"
    assert detail["memory_safety_ok"] == 0.0 and "element mismatch" in detail["private_error"]


def test_required_mode_fails_closed_when_the_sanitizer_cannot_attach(monkeypatch, tmp_path: Path) -> None:
    rc, detail, calls = _evaluate_with(monkeypatch, tmp_path, mode="required",
                                       sanitizer_record={"status": "unavailable", "report": []})
    assert rc == 1 and calls == ["sanitizer"]
    assert detail["failed_gate"] == "memory_safety_ok"
    assert "could not attach" in detail["private_error"]


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


def test_platform_mode_skips_the_pass_and_says_so(monkeypatch, tmp_path: Path) -> None:
    rc, detail, calls = _evaluate_with(monkeypatch, tmp_path, mode="skip-platform-incompatible")
    assert rc == 0 and calls == ["worker"]
    assert detail["sanitizer_ran"] == 0.0
    assert detail["sanitizer"]["status"] == "not_run"
    assert detail["memory_safety_ok"] == 1.0


def test_unknown_sanitizer_mode_fails_closed(monkeypatch, tmp_path: Path) -> None:
    rc, detail, calls = _evaluate_with(monkeypatch, tmp_path, mode="off")
    assert rc == 1 and calls == []
    assert detail["private_error"] == "invalid compute-sanitizer mode"
