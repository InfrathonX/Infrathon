from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _profiles() -> list[Path]:
    """The experiment-*.yaml run profiles the tree carries (possibly none)."""
    return sorted(ROOT.glob("experiment-*.yaml"))


def _value(pattern: str, text: str, default: str | None = None) -> str | None:
    match = re.search(pattern, text, re.M)
    return match.group(1) if match else default


def _task_json() -> tuple[Path, dict]:
    paths = sorted((ROOT / "tasks").glob("*.json"))
    assert len(paths) == 1, f"expected one task JSON, found {paths}"
    return paths[0], json.loads(paths[0].read_text(encoding="utf-8"))


def test_task_uses_valid_then_score_and_raw_throughput_direction() -> None:
    _, task = _task_json()
    judge = task["judge"]

    assert judge["parser"] == "structured_json"
    assert judge["selection"] == "valid_then_score"
    assert judge["score_direction"] == "maximize"
    assert int(judge["eval_timeout"]) >= 2400
    assert judge["private_artifact_path"].startswith("/logs/verifier/")


def test_formal_experiment_has_unlimited_manual_submissions_and_cooldown() -> None:
    launcher_path = ROOT / "scripts" / "run_fixed_budget.sh"
    assert launcher_path.is_file()
    launcher = launcher_path.read_text(encoding="utf-8")

    assert "--max-submissions" not in launcher
    assert 'SUBMISSION_COOLDOWN="${SUBMISSION_COOLDOWN:-300}"' in launcher
    assert "--disable-auto-eval" in launcher
    for path in _profiles():
        experiment = path.read_text(encoding="utf-8")
        assert "max_submissions:" not in experiment
        assert "submission_cooldown: 300" in experiment
        assert "disable_auto_eval: true" in experiment


def test_formal_budget_profiles_are_exact() -> None:
    launcher = (ROOT / "scripts" / "run_fixed_budget.sh").read_text(
        encoding="utf-8"
    )
    tmux_launcher = (ROOT / "scripts" / "launch_fixed_budget_tmux.sh").read_text(
        encoding="utf-8"
    )

    # No default model: the run names it (launch/concurrent-serving/run.sh --model).
    assert 'MODEL="${MODEL:-}"' in launcher
    assert 'REASONING_EFFORT="${REASONING_EFFORT:-max}"' in launcher
    assert 'TASK_TIMEOUT="${TASK_TIMEOUT:-21600}"' in launcher
    assert 'MODEL_VALUE="${MODEL:-}"' in tmux_launcher
    assert 'REASONING_VALUE="${REASONING_EFFORT:-max}"' in tmux_launcher
    assert 'TIMEOUT_VALUE="${TASK_TIMEOUT:-21600}"' in tmux_launcher
    # Each run profile the tree carries matches its tmux wrapper, when present
    # (a wrapper that leaves effort or budget unset gets the defaults above).
    for path in _profiles():
        experiment = path.read_text(encoding="utf-8")
        model = _value(r"^\s*model: (\S+)$", experiment)
        effort = _value(r"^\s*SFORGE_CODEX_REASONING_EFFORT: (\S+)$", experiment)
        timeout = _value(r"^\s*timeout: (\d+)$", experiment)
        assert model and effort and timeout, path.name
        assert "disable_auto_eval: true" in experiment
        stem = path.stem.removeprefix("experiment-").replace("-", "_")
        wrapper_path = ROOT / "scripts" / f"launch_{stem}_tmux.sh"
        if not wrapper_path.is_file():
            continue
        wrapper = wrapper_path.read_text(encoding="utf-8")
        assert _value(r'^export MODEL="(?:\$\{MODEL:-)?([^"}]+)\}?"$', wrapper) == model
        assert _value(r'^export REASONING_EFFORT="(\w+)"$', wrapper, "max") == effort
        assert _value(r'^export TASK_TIMEOUT="(\d+)"$', wrapper, "21600") == timeout


def test_formal_runner_validates_model_effort_and_budget_independently() -> None:
    """Model, effort, and budget must not be welded into one whitelist.

    The old combined MODEL:EFFORT:TIMEOUT case forced a source change for
    every new duration and rejected short validation runs outright. The run
    manifest attests the exact triple, so each axis is validated on its own.
    """

    runner = (ROOT / "scripts" / "run_fixed_budget.sh").read_text(
        encoding="utf-8"
    )

    assert 'case "${MODEL}:${REASONING_EFFORT}:${TASK_TIMEOUT}" in' not in runner
    assert "unsupported formal profile" not in runner
    assert 'case "${MODEL}" in' in runner
    assert '[[ "${MODEL}" =~ ^[A-Za-z0-9][A-Za-z0-9_./:-]*$ ]]' in runner
    assert 'case "${REASONING_EFFORT}" in' in runner
    assert "high|max|xhigh|ultra" in runner
    assert "TASK_TIMEOUT must be minute-aligned and at least 3600 seconds" in runner
    assert '[[ "${TASK_TIMEOUT}" == "21600" ]]' not in runner


def test_agent_contract_explains_sf_feedback_policy() -> None:
    _, task = _task_json()
    query = task["work"]["agent_query"].lower()
    instruction_path = ROOT / "images" / "work" / "AGENT_TASK.md"
    assert instruction_path.is_file()
    instruction = instruction_path.read_text(encoding="utf-8").lower()

    # v2.4 keeps the task query architecture-first and leaves iterative-eval
    # mechanics in the complete instruction document.
    assert "workflow-level" in query
    assert "online monitoring" in query
    assert "sforge-submit" not in query
    assert "sforge-submit" in instruction
    assert "count-unlimited" in instruction
    assert "cooldown in `run_budget.json`" in instruction
    assert "300-second cooldown" not in instruction


def test_packaging_invalid_candidate_is_structured_exit_zero(tmp_path: Path) -> None:
    """Candidate-invalid is data, unlike an abnormal verifier process exit."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    migration_root = tmp_path / "migration"
    migration_root.mkdir()
    eval_root = tmp_path / "evals"
    eval_root.mkdir()
    scratch = eval_root / "judge-unit"
    scratch.mkdir()
    nonce_source = tmp_path / "nonce.jsonl"
    nonce_source.write_text("{}\n", encoding="utf-8")
    baseline_path = tmp_path / "hidden-baseline.json"
    baseline_bytes = (
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "suite": "hidden",
                "profile": "hidden-pressure-192",
                "workload_nonce_sha256": (
                    "c525ae31a2bb1d3ed0232c345f9fc978fd52574d415aaa2777bead17f2fac20a"
                ),
                "legs": {
                    "baseline": {
                        "valid_steps_per_min": 83.73333333333333,
                        "model_provenance": {
                            "profile_id": "fixture-profile",
                            "profile_sha256": "a" * 64,
                            "model_revision": (
                                "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
                            ),
                            "runtime_image_id": (
                                "sha256:1d8369b9e3c91aa8cba5c7aeeef26e3c4afd455e5d2ce9aebe6f30f6dd9344bd"
                            ),
                            "tensor_parallel_size": 2,
                            "kv_cache_dtype": "fp8",
                            "backend_serving_contract_sha256": (
                                "18d27d97c736f90e1d74822e80ce1f212685e26627b31a08e47569a1457f52c6"
                            ),
                        },
                    }
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()
    baseline_path.write_bytes(baseline_bytes)
    log_dir = tmp_path / "logs"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "judge" / "run_judge.py"),
            "--workspace",
            str(workspace),
            "--log-dir",
            str(log_dir),
        ],
        env={
            **os.environ,
            "KVBENCH_HOST_EVAL_DIR": str(scratch),
            "KVBENCH_EVAL_ROOT": str(eval_root),
            "KVBENCH_MIGRATION_ROOT": str(migration_root),
            "KVBENCH_HIDDEN_NONCE_SOURCE": str(nonce_source),
            "KVBENCH_HIDDEN_BASELINE_PATH": str(baseline_path),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    start = completed.stdout.index(">>>>> Start Structured Result")
    end = completed.stdout.index(">>>>> End Structured Result")
    result = json.loads(
        completed.stdout[
            start + len(">>>>> Start Structured Result") : end
        ].strip()
    )
    assert result["valid"] is False
    assert result["score"] == 0.0
    assert result["metrics"]["evaluation_status"] == "candidate_invalid"
    private_path = log_dir / "kvbench-private-evidence.json"
    assert private_path.is_file()
    private = json.loads(private_path.read_text(encoding="utf-8"))
    assert private["commitments"]["hidden_baseline_identity"] == {
        "file_sha256": hashlib.sha256(baseline_bytes).hexdigest(),
        "valid_steps_per_min": 83.73333333333333,
        "backend_serving_contract_sha256": (
            "18d27d97c736f90e1d74822e80ce1f212685e26627b31a08e47569a1457f52c6"
        ),
        "workload_nonce_sha256": (
            "c525ae31a2bb1d3ed0232c345f9fc978fd52574d415aaa2777bead17f2fac20a"
        ),
    }
