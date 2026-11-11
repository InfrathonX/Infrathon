from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "judge/run_public.py"
DIRECT_RATE = 103.3
BASELINE_SHA256 = "b" * 64


def _load_adapter(name: str = "kvbench_public_judge_adapter"):
    spec = importlib.util.spec_from_file_location(name, ADAPTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def public_judge():
    return _load_adapter()


def profile(public_judge) -> dict[str, object]:
    return {
        "name": public_judge.PUBLIC_PROFILE,
        "workflows": 192,
        "concurrency": 192,
        "max_steps": 100,
        "measurement_warmup_sec": 300,
        "measurement_window_sec": 600,
    }


def workflows() -> list[dict[str, object]]:
    return [
        {
            "task_id": f"private-task-{index}",
            "run_id": f"private-run-{index}",
            "total_executed_steps": 8,
            "prompt": f"private prompt {index}",
        }
        for index in range(192)
    ]


def candidate(public_judge, *, rate: float = 165.0) -> dict[str, object]:
    return {
        "profile": public_judge.PUBLIC_PROFILE,
        "profile_config": profile(public_judge),
        "workflows": workflows(),
        "valid_steps_per_min": rate,
        "request_success_rate": 0.995,
        "valid_step_ratio": 0.97,
        "backend_tokens_per_valid_step": 4800.0,
        "backend_audit": {
            "p95_gateway_wait_sec": 0.12,
            "p99_gateway_wait_sec": 0.25,
        },
        "vllm_metrics": {
            "counter_deltas": [
                {"name": "vllm:prefix_cache_hits_total", "delta": 250.0},
                {"name": "vllm:prefix_cache_queries_total", "delta": 1000.0},
            ]
        },
        "pressure_validation": {
            "status": "pressurized",
            "valid": True,
            "occupancy_evidence": {
                "observed_kv_occupancy": {
                    "kv_p95": 0.94,
                    "time_fraction": 0.41,
                }
            },
            "contention_evidence": {
                "waiting_at_high_kv": {"valid": True, "sample_count": 246},
                "preemption": {"preemptions_delta": 2.0},
            },
        },
    }


def evaluation(public_judge, *, rate: float = 165.0, valid: bool = True):
    return {
        "valid": valid,
        "qualification_passed": rate > DIRECT_RATE,
        "strictly_better_than_v2_1_direct": rate
        > DIRECT_RATE,
        "direct_valid_steps_per_min": DIRECT_RATE,
        "candidate_valid_steps_per_min": rate,
        "speedup": rate / DIRECT_RATE,
        "change_percent": (rate / DIRECT_RATE - 1.0) * 100.0,
        "all_invalid_reasons": [] if valid else ["request_success_below_floor"],
        "backend_serving_contract_sha256": (
            public_judge.EXPECTED_BACKEND_SERVING_CONTRACT_SHA256
        ),
    }


def baseline(public_judge) -> dict[str, object]:
    return {
        "valid_steps_per_min": DIRECT_RATE,
        "backend_tokens_per_valid_step": 4700.0,
        "profile": public_judge.PUBLIC_PROFILE,
        "profile_config": profile(public_judge),
        "model_provenance": {
            "profile_id": "qwen3.6-27b-compatibility-profile",
            "profile_sha256": "a" * 64,
            "model_revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
            "runtime_image_id": public_judge.COMMON.EXPECTED_RUNTIME_IMAGE_ID,
            "tensor_parallel_size": 2,
            "kv_cache_dtype": "fp8",
            "backend_serving_contract_sha256": (
                public_judge.EXPECTED_BACKEND_SERVING_CONTRACT_SHA256
            ),
        },
    }


def baseline_bundle(public_judge) -> dict[str, object]:
    return {
        "path": "/private/public-direct.json",
        "public_baseline_identity": {
            "file_sha256": BASELINE_SHA256,
            "valid_steps_per_min": DIRECT_RATE,
            "backend_serving_contract_sha256": (
                public_judge.EXPECTED_BACKEND_SERVING_CONTRACT_SHA256
            ),
            "workload_nonce_sha256": (
                public_judge.PUBLIC_NONCE_COMMITMENT_SHA256
            ),
        },
        "leg": baseline(public_judge),
    }


def report(
    public_judge,
    *,
    tree: str = "a" * 64,
    valid: bool = True,
    rate: float = 165.0,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": public_judge.PUBLIC_RUNNER_KIND,
        "status": "complete" if valid else "candidate_invalid",
        "phase": "quiescent",
        "completed_at_utc": "2026-08-01T00:00:00Z",
        "direct_leg_executed": False,
        "candidate_tree_sha256": tree,
        "preflight_backend_serving_contract_sha256": (
            public_judge.EXPECTED_BACKEND_SERVING_CONTRACT_SHA256
        ),
        "workload_contract": {
            "suite": "public",
            "profile": public_judge.PUBLIC_PROFILE,
            "workflows": 192,
            "concurrency": 192,
            "max_steps": 100,
            "measurement_warmup_sec": 300,
            "measurement_window_sec": 600,
            "workload_nonce_sha256": (
                public_judge.PUBLIC_NONCE_COMMITMENT_SHA256
            ),
        },
        "baseline_evidence": {
            "path": "/private/public-direct.json",
            "file_sha256": BASELINE_SHA256,
            "workload_nonce_sha256": (
                public_judge.PUBLIC_NONCE_COMMITMENT_SHA256
            ),
            "direct_valid_steps_per_min": DIRECT_RATE,
            "backend_serving_contract_sha256": (
                public_judge.EXPECTED_BACKEND_SERVING_CONTRACT_SHA256
            ),
        },
        "leg": candidate(public_judge, rate=rate),
        "evaluation": evaluation(public_judge, rate=rate, valid=valid),
        "errors": [],
        "cleanup": {
            "attempted": True,
            "stack_stop_succeeded": True,
            "quiescent": True,
            "reconciled_before_start": False,
            "residue": {
                "compose_container_ids": [],
                "sandbox_container_ids": [],
            },
        },
    }


def test_public_contract_and_safe_aggregate_projection(public_judge) -> None:
    evidence = report(public_judge)
    cand, evaluated, status = public_judge._validate_complete_report(
        evidence, "a" * 64, 0, baseline_bundle(public_judge)
    )
    result = public_judge.public_structured_result(
        cand, evaluated, baseline(public_judge), status=status
    )

    assert result["valid"] is True
    assert result["score"] == 165.0
    metrics = result["metrics"]
    assert metrics["workflows"] == 192
    assert metrics["concurrency"] == 192
    assert metrics["max_steps"] == 100
    assert metrics["measurement_warmup_sec"] == 300
    assert metrics["measurement_window_sec"] == 600
    assert metrics["prefix_cache_hit_rate"] == pytest.approx(0.25)
    assert metrics["backend_work_ratio_to_frozen_direct"] == pytest.approx(
        4800.0 / 4700.0
    )
    assert metrics["gateway_wait_p95_sec"] == pytest.approx(0.12)
    assert metrics["gateway_wait_p99_sec"] == pytest.approx(0.25)
    assert metrics["kv_cache_usage_p95"] == pytest.approx(0.94)
    assert metrics["waiting_at_high_kv_sample_count"] == 246

    visible = json.dumps(result, sort_keys=True)
    for secret in (
        "private-task-0",
        "private-run-0",
        "private prompt 0",
        public_judge.PUBLIC_NONCE_COMMITMENT_SHA256,
        "/private/",
    ):
        assert secret not in visible


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("workflows", 191),
        ("concurrency", 191),
        ("max_steps", 99),
        ("measurement_warmup_sec", 1),
        ("measurement_window_sec", 599),
    ],
)
def test_terminal_evidence_rejects_public_profile_drift(
    public_judge, field: str, value: int
) -> None:
    evidence = report(public_judge)
    evidence["workload_contract"][field] = value
    with pytest.raises(RuntimeError, match="terminal evidence"):
        public_judge._validate_complete_report(
            evidence, "a" * 64, 0, baseline_bundle(public_judge)
        )

    evidence = report(public_judge)
    evidence["leg"]["profile_config"][field] = value
    with pytest.raises(RuntimeError, match="profile shape"):
        public_judge._validate_complete_report(
            evidence, "a" * 64, 0, baseline_bundle(public_judge)
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "baseline_sha",
        "baseline_nonce",
        "baseline_contract",
        "baseline_rate",
        "baseline_path",
        "baseline_extra",
        "nonce",
        "contract",
        "cleanup",
        "returncode",
        "tree",
    ],
)
def test_terminal_evidence_rejects_identity_or_cleanup_drift(
    public_judge, mutation: str
) -> None:
    evidence = report(public_judge)
    tree = "a" * 64
    returncode = 0
    if mutation == "baseline_sha":
        evidence["baseline_evidence"]["file_sha256"] = "0" * 64
    elif mutation == "baseline_nonce":
        evidence["baseline_evidence"]["workload_nonce_sha256"] = "0" * 64
    elif mutation == "baseline_contract":
        evidence["baseline_evidence"][
            "backend_serving_contract_sha256"
        ] = "0" * 64
    elif mutation == "baseline_rate":
        evidence["baseline_evidence"]["direct_valid_steps_per_min"] = 103.4
    elif mutation == "baseline_path":
        evidence["baseline_evidence"]["path"] = "/different/public.json"
    elif mutation == "baseline_extra":
        evidence["baseline_evidence"]["unexpected"] = True
    elif mutation == "nonce":
        evidence["workload_contract"]["workload_nonce_sha256"] = "0" * 64
    elif mutation == "contract":
        evidence["preflight_backend_serving_contract_sha256"] = "0" * 64
    elif mutation == "cleanup":
        evidence["cleanup"]["quiescent"] = False
    elif mutation == "returncode":
        returncode = 3
    elif mutation == "tree":
        tree = "b" * 64
    with pytest.raises(RuntimeError, match="terminal evidence"):
        public_judge._validate_complete_report(
            evidence, tree, returncode, baseline_bundle(public_judge)
        )


def test_candidate_invalid_is_authoritative_and_unknown_reason_is_redacted(
    public_judge,
) -> None:
    evidence = report(public_judge, valid=False, rate=120.0)
    evidence["evaluation"]["all_invalid_reasons"] = [
        "request_success_below_floor",
        "secret-private-reason-value",
    ]
    cand, evaluated, status = public_judge._validate_complete_report(
        evidence, "a" * 64, 3, baseline_bundle(public_judge)
    )
    result = public_judge.public_structured_result(
        cand, evaluated, baseline(public_judge), status=status
    )

    assert result["valid"] is False
    assert result["score"] == 0.0
    assert result["errors"] == 0
    assert result["metrics"]["evaluation_status"] == "candidate_invalid"
    assert result["metrics"]["candidate_valid_steps_per_min"] == 120.0
    assert result["metrics"]["invalid_reasons"] == [
        "request_success_below_floor",
        "runner_validation_failed",
    ]
    assert "secret-private-reason-value" not in json.dumps(result)


def test_frozen_public_baseline_bytes_and_contract_are_pinned(
    public_judge, monkeypatch, tmp_path: Path
) -> None:
    frozen = {
        "schema_version": 1,
        "status": "complete",
        "suite": "public",
        "profile": public_judge.PUBLIC_PROFILE,
        "workload_nonce_sha256": public_judge.PUBLIC_NONCE_COMMITMENT_SHA256,
        "legs": {"baseline": baseline(public_judge)},
    }
    path = tmp_path / "public-direct.json"
    raw = (json.dumps(frozen, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    monkeypatch.setattr(public_judge, "PUBLIC_BASELINE_PATH", path)
    monkeypatch.setattr(
        public_judge.COMMON,
        "file_sha256",
        lambda _path: pytest.fail("loader must hash its already-read bytes"),
    )
    monkeypatch.setattr(
        public_judge.COMMON,
        "_strict_json_object",
        lambda _path: pytest.fail("loader must parse its already-read bytes"),
    )
    bundle = public_judge._load_public_baseline_bundle()
    assert bundle["leg"]["valid_steps_per_min"] == 103.3
    assert bundle["public_baseline_identity"] == {
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "valid_steps_per_min": DIRECT_RATE,
        "backend_serving_contract_sha256": (
            public_judge.EXPECTED_BACKEND_SERVING_CONTRACT_SHA256
        ),
        "workload_nonce_sha256": public_judge.PUBLIC_NONCE_COMMITMENT_SHA256,
    }

    path.write_text(json.dumps({**frozen, "suite": "hidden"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="contract changed"):
        public_judge._load_public_baseline()


def test_public_baseline_requires_actual_or_exact_legacy_contract(
    public_judge, monkeypatch, tmp_path: Path
) -> None:
    frozen = {
        "schema_version": 1,
        "status": "complete",
        "suite": "public",
        "profile": public_judge.PUBLIC_PROFILE,
        "workload_nonce_sha256": public_judge.PUBLIC_NONCE_COMMITMENT_SHA256,
        "legs": {"baseline": baseline(public_judge)},
    }
    path = tmp_path / "public-direct.json"
    monkeypatch.setattr(public_judge, "PUBLIC_BASELINE_PATH", path)

    frozen["legs"]["baseline"]["model_provenance"][
        "backend_serving_contract_sha256"
    ] = "0" * 64
    path.write_text(json.dumps(frozen), encoding="utf-8")
    with pytest.raises(RuntimeError, match="backend serving contract"):
        public_judge._load_public_baseline_bundle()

    # Evidence recorded before the digest existed is attributed to the
    # retained v1 revision, not restamped with the shipped revision.
    frozen["legs"]["baseline"]["model_provenance"] = {
        **public_judge.COMMON.LEGACY_MODEL_PROVENANCE,
    }
    path.write_text(json.dumps(frozen), encoding="utf-8")
    assert (
        public_judge._load_public_baseline_bundle()["public_baseline_identity"][
            "backend_serving_contract_sha256"
        ]
        == public_judge.COMMON.LEGACY_BACKEND_SERVING_CONTRACT_SHA256
    )

    frozen["legs"]["baseline"]["model_provenance"]["profile_id"] = (
        "unknown-legacy-profile"
    )
    path.write_text(json.dumps(frozen), encoding="utf-8")
    with pytest.raises(RuntimeError, match="legacy serving provenance"):
        public_judge._load_public_baseline_bundle()


def test_public_private_envelope_uses_bundle_without_rereading_path(
    public_judge, monkeypatch, tmp_path: Path
) -> None:
    bundle = baseline_bundle(public_judge)
    monkeypatch.setattr(
        public_judge.COMMON,
        "file_sha256",
        lambda _path: pytest.fail("envelope must not reread public baseline"),
    )
    output = tmp_path / "public-private.json"
    public_judge._write_private_envelope(
        output,
        started_at_utc="2026-08-01T00:00:00Z",
        classification="infrastructure_error",
        candidate_sha256=None,
        runner_metadata={},
        runner_evidence=None,
        public_result=public_judge.infrastructure_failure(),
        private_errors=[],
        baseline_bundle=bundle,
    )

    private = json.loads(output.read_text(encoding="utf-8"))
    assert private["commitments"] == {
        "public_baseline_identity": bundle["public_baseline_identity"]
    }


def test_public_end_guard_detects_drift_without_replacing_commitment(
    public_judge, tmp_path: Path
) -> None:
    path = tmp_path / "public-direct.json"
    path.write_bytes(b"original public bytes")
    identity = {
        "file_sha256": hashlib.sha256(b"original public bytes").hexdigest(),
        "valid_steps_per_min": DIRECT_RATE,
        "backend_serving_contract_sha256": (
            public_judge.EXPECTED_BACKEND_SERVING_CONTRACT_SHA256
        ),
        "workload_nonce_sha256": public_judge.PUBLIC_NONCE_COMMITMENT_SHA256,
    }
    bundle = {
        "path": str(path),
        "public_baseline_identity": identity,
        "leg": baseline(public_judge),
    }
    public_judge._assert_public_baseline_snapshot_current(bundle)
    path.write_bytes(b"replacement public bytes")
    with pytest.raises(RuntimeError, match="bytes drifted"):
        public_judge._assert_public_baseline_snapshot_current(bundle)
    assert bundle["public_baseline_identity"] == identity


def test_secure_snapshot_rejects_links_and_preserves_executable(
    public_judge, tmp_path: Path
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    launch = workspace / "launch.sh"
    launch.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    launch.chmod(0o755)
    (workspace / "gateway.py").write_text("print('ok')\n", encoding="utf-8")
    destination = tmp_path / "snapshot"

    digest = public_judge.secure_copy_submission(workspace, destination)
    assert digest == public_judge.COMMON.tree_sha256(destination)
    assert destination.stat().st_mode & 0o777 == workspace.stat().st_mode & 0o777
    assert os.access(destination / "launch.sh", os.X_OK)

    linked_workspace = tmp_path / "linked"
    linked_workspace.mkdir()
    linked_launch = linked_workspace / "launch.sh"
    linked_launch.write_text("#!/bin/sh\n", encoding="utf-8")
    linked_launch.chmod(0o755)
    (linked_workspace / "escape").symlink_to("/etc/passwd")
    with pytest.raises(public_judge.PublicSubmissionError, match="submission_symlink"):
        public_judge.secure_copy_submission(
            linked_workspace, tmp_path / "linked-snapshot"
        )


def test_supervisor_gives_runner_a_bounded_cleanup_window(
    public_judge, monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(public_judge, "PUBLIC_WARMUP_SEC", 0)
    monkeypatch.setattr(public_judge, "PUBLIC_MEASUREMENT_SEC", 0)
    monkeypatch.setenv("KVBENCH_PUBLIC_RUNNER_TIMEOUT_SEC", "1")
    monkeypatch.setenv("KVBENCH_PUBLIC_CLEANUP_GRACE_SEC", "2")
    program = """
import signal
import time

def stop(_signum, _frame):
    print('trusted-cleanup-finished', flush=True)
    raise SystemExit(0)

signal.signal(signal.SIGTERM, stop)
time.sleep(30)
"""
    runner_log = tmp_path / "runner.log"
    outcome = public_judge._run_supervised(
        [sys.executable, "-c", program],
        environment=dict(os.environ),
        runner_log=runner_log,
    )

    assert set(outcome) == {
        "returncode",
        "timed_out",
        "cancel_signal",
        "termination_sent",
        "forced_kill",
        "runtime_seconds",
        "force_reconcile",
    }
    assert outcome["timed_out"] is True
    assert outcome["termination_sent"] is True
    assert outcome["forced_kill"] is False
    assert outcome["force_reconcile"] is None
    assert outcome["returncode"] == 0
    assert "trusted-cleanup-finished" in runner_log.read_text(encoding="utf-8")
    assert runner_log.stat().st_mode & 0o777 == 0o600


def test_default_timeout_layers_leave_outer_supervision_margin(public_judge) -> None:
    assert public_judge.DEFAULT_PUBLIC_RUNNER_TIMEOUT_SEC > 900
    assert public_judge.DEFAULT_PUBLIC_CLEANUP_GRACE_SEC > 0
    assert (
        public_judge.DEFAULT_PUBLIC_RUNNER_TIMEOUT_SEC
        + public_judge.DEFAULT_PUBLIC_CLEANUP_GRACE_SEC
        <= public_judge.PUBLIC_OUTER_TIMEOUT_SEC
        - public_judge.PUBLIC_OUTER_MARGIN_SEC
    )


def test_main_emits_once_and_keeps_full_evidence_private(
    public_judge, monkeypatch, tmp_path: Path, capsys
) -> None:
    migration = tmp_path / "migration"
    (migration / "scripts").mkdir(parents=True)
    canonical_runner = migration / "scripts/run_public_candidate.py"
    canonical_runner.write_text("# frozen runner\n", encoding="utf-8")
    installed_runner = tmp_path / "installed-runner.py"
    installed_runner.write_text("# frozen runner\n", encoding="utf-8")
    host_root = tmp_path / "evals"
    host_scratch = host_root / "judge-test"
    host_scratch.mkdir(parents=True)
    nonce = tmp_path / "nonce-source"
    nonce.write_text("private raw nonce source\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    log_dir = tmp_path / "logs"
    tree = "c" * 64

    monkeypatch.setattr(public_judge, "INSTALLED_PUBLIC_RUNNER", installed_runner)
    monkeypatch.setattr(
        public_judge,
        "_validate_runtime_paths",
        lambda: (migration, host_scratch, nonce),
    )
    monkeypatch.setattr(
        public_judge,
        "_load_public_baseline_bundle",
        lambda: baseline_bundle(public_judge),
    )
    monkeypatch.setattr(
        public_judge,
        "_assert_public_baseline_snapshot_current",
        lambda _bundle: None,
    )
    monkeypatch.setattr(public_judge, "secure_copy_submission", lambda *_: tree)

    def fake_run(command, *, environment, runner_log):
        del environment
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps(report(public_judge, tree=tree)), encoding="utf-8")
        runner_log.write_text("private runner log\n", encoding="utf-8")
        runner_log.chmod(0o600)
        return {
            "returncode": 0,
            "timed_out": False,
            "cancel_signal": None,
            "termination_sent": False,
            "forced_kill": False,
            "runtime_seconds": 1.0,
            "force_reconcile": None,
        }

    monkeypatch.setattr(public_judge, "_run_supervised", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(ADAPTER),
            "--workspace",
            str(workspace),
            "--log-dir",
            str(log_dir),
        ],
    )

    assert public_judge.main() == 0
    stdout = capsys.readouterr().out
    assert stdout.count(">>>>> Start Structured Result") == 1
    assert stdout.count(">>>>> End Structured Result") == 1
    assert "private-task-0" not in stdout
    assert "private raw nonce" not in stdout
    artifact = log_dir / "kvbench-public-private-evidence.json"
    private = json.loads(artifact.read_text(encoding="utf-8"))
    assert private["runner_evidence"]["leg"]["workflows"][0]["task_id"] == (
        "private-task-0"
    )
    assert private["commitments"] == {
        "public_baseline_identity": baseline_bundle(public_judge)[
            "public_baseline_identity"
        ]
    }
    assert artifact.stat().st_mode & 0o777 == 0o600
    assert list(host_scratch.iterdir()) == []
