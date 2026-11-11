from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


TASK_ID = "concurrent_agent_serving_optimizer_qwen36_swe192_v24"
JUDGE_NAME = f"kvbench_qwen36.judge.{TASK_ID}.manual-1"
WORK_NAME = f"sforge.run.{TASK_ID}.run-1"
VALIDATION_STACK = (
    Path(__file__).resolve().parents[1]
    / "runtime/frozen/harbor_run_scripts/"
    "concurrent-agent-serving-optimizer-qwen36-27b-swe192-v2.1-gpu45/"
    "gpu67-v2.1-validation-stack.sh"
)


def _bind_targets(plan: dict[str, object]) -> dict[str, tuple[str, str]]:
    targets: dict[str, tuple[str, str]] = {}
    for source, value in plan.items():
        if isinstance(value, str):
            target, _, mode = value.partition(":")
        else:
            assert isinstance(value, dict), (source, value)
            target = str(value.get("bind", ""))
            mode = str(value.get("mode", ""))
        assert target.startswith("/"), (source, value)
        targets[target] = (str(source), mode)
    return targets


def _configure_judge_files(runtime_module, monkeypatch, tmp_path: Path) -> None:
    nonce_source = tmp_path / "nonce-source.jsonl"
    hidden_baseline = tmp_path / "hidden-direct.json"
    public_baseline = tmp_path / "public-direct.json"
    for path in (nonce_source, hidden_baseline, public_baseline):
        path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv(runtime_module.NONCE_SOURCE_ENV, str(nonce_source))
    monkeypatch.setenv(runtime_module.HIDDEN_BASELINE_ENV, str(hidden_baseline))
    monkeypatch.setenv(runtime_module.PUBLIC_BASELINE_ENV, str(public_baseline))


def test_name_classifier_is_fail_closed(runtime_module) -> None:
    assert runtime_module.is_judge_container(JUDGE_NAME) is True
    assert runtime_module.is_judge_container(WORK_NAME) is False
    assert runtime_module.is_judge_container("sforge.judge.kvbench.1") is False
    assert runtime_module.is_judge_container("kvbench-agent") is False
    assert runtime_module.is_judge_container("") is False


def test_work_container_receives_no_host_or_hidden_mounts(runtime_module) -> None:
    plan = runtime_module.mount_plan(WORK_NAME)

    assert plan == {}


def test_work_container_scrubs_accidentally_forwarded_trusted_environment(
    runtime_module, monkeypatch, tmp_path: Path
) -> None:
    public_direct_rate = 137.25
    selected_public_baseline = tmp_path / "selected-public-direct.json"
    selected_public_baseline.write_text(
        json.dumps(
            {
                "status": "complete",
                "suite": "public",
                "profile": "public-swe192-distinct",
                "workload_nonce_sha256": (
                    runtime_module.EXPECTED_PUBLIC_NONCE_COMMITMENT
                ),
                "legs": {
                    "baseline": {
                        "mode": "baseline",
                        "valid_steps_per_min": public_direct_rate,
                        "model_provenance": {
                            "backend_serving_contract_sha256": (
                                runtime_module.BACKEND_SERVING_CONTRACT_SHA256
                            )
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    private_environment = {
        runtime_module.VLLM_BASE_URL_ENV: "http://host-backend.invalid:18000",
        runtime_module.HIDDEN_BASELINE_ENV: "/host/hidden.json",
        runtime_module.PUBLIC_BASELINE_ENV: "/host/public.json",
        runtime_module.NONCE_SOURCE_ENV: "/host/nonce-source",
        runtime_module.EVAL_ROOT_ENV: "/host/evaluations",
        "KVBENCH_BACKEND_SERVING_CONTRACT_SHA256": "a" * 64,
        "KVBENCH_HIDDEN_NONCE_SOURCE": "/run/private/nonce",
        "KVBENCH_MEASUREMENT_LOCK_PATH": "/run/private/measurement.lock",
        "KVBENCH_HOST_EVAL_DIR": "/host/private/eval",
        "KVBENCH_EVAL_ROOT": "/host/private/eval-root",
        "KVBENCH_MIGRATION_ROOT": "/host/private/migration",
        "SFORGE_KVBENCH_MIGRATION_ROOT": "/host/private/migration",
    }
    incoming_environment = {"SAFE_AGENT_SETTING": "preserved", **private_environment}
    observed: dict[str, object] = {}

    class FakeContainers:
        def create(self, image, **kwargs):
            observed["image"] = image
            observed.update(kwargs)
            return SimpleNamespace(id="container-id", name=kwargs["name"])

    class FakeNetworks:
        def get(self, name):
            observed["network_lookup"] = name
            return SimpleNamespace(attrs={"Internal": True})

    backend = SimpleNamespace(
        client=SimpleNamespace(containers=FakeContainers(), networks=FakeNetworks())
    )
    monkeypatch.setenv(runtime_module.NETWORK_ENV, "mlsysforge-work-internal")
    monkeypatch.setenv(
        runtime_module.PUBLIC_BASELINE_ENV, str(selected_public_baseline)
    )

    runtime_module.isolated_create_container(
        backend,
        "work-image",
        WORK_NAME,
        environment=incoming_environment,
    )

    assert observed["environment"] == {
        "SAFE_AGENT_SETTING": "preserved",
        "KVBENCH_PUBLIC_DIRECT_RATE": format(public_direct_rate, ".17g"),
    }
    assert observed["network"] == "mlsysforge-work-internal"
    assert observed["network_lookup"] == "mlsysforge-work-internal"
    assert observed["labels"] == {"mlsysforge.kvbench.role": "untrusted-work"}
    # Scrubbing operates on a copy; callers retain the exact environment they
    # supplied for their own bookkeeping.
    assert set(private_environment) <= set(incoming_environment)


def test_work_container_accepts_only_readonly_agent_runtime_mounts(
    runtime_module, monkeypatch, tmp_path: Path
) -> None:
    selected_public_baseline = tmp_path / "selected-public-direct.json"
    selected_public_baseline.write_text(
        json.dumps(
            {
                "status": "complete",
                "suite": "public",
                "profile": "public-swe192-distinct",
                "workload_nonce_sha256": (
                    runtime_module.EXPECTED_PUBLIC_NONCE_COMMITMENT
                ),
                "legs": {
                    "baseline": {
                        "mode": "baseline",
                        "valid_steps_per_min": 137.25,
                        "model_provenance": {
                            "backend_serving_contract_sha256": (
                                runtime_module.BACKEND_SERVING_CONTRACT_SHA256
                            )
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    observed: dict[str, object] = {}

    class FakeContainers:
        def create(self, image, **kwargs):
            observed["image"] = image
            observed.update(kwargs)
            return SimpleNamespace(id="container-id", name=kwargs["name"])

    class FakeNetworks:
        def get(self, name):
            return SimpleNamespace(attrs={"Internal": True})

    backend = SimpleNamespace(
        client=SimpleNamespace(containers=FakeContainers(), networks=FakeNetworks())
    )
    monkeypatch.setenv(runtime_module.NETWORK_ENV, "mlsysforge-work-internal")
    monkeypatch.setenv(
        runtime_module.PUBLIC_BASELINE_ENV, str(selected_public_baseline)
    )

    runtime_module.isolated_create_container(
        backend,
        "work-image",
        WORK_NAME,
        mounts=[("/host/agent_runtimes/codex-0.148.0", "/opt/agent-runtime", "ro")],
    )
    assert observed["volumes"] == {
        "/host/agent_runtimes/codex-0.148.0": {
            "bind": "/opt/agent-runtime",
            "mode": "ro",
        }
    }

    with pytest.raises(RuntimeError, match="only read-only"):
        runtime_module.isolated_create_container(
            backend,
            "work-image",
            WORK_NAME,
            mounts=[("/host/agent_runtimes/codex-0.148.0", "/opt/agent-runtime", "rw")],
        )


def test_judge_container_refuses_caller_supplied_mounts(runtime_module) -> None:
    backend = SimpleNamespace(client=SimpleNamespace())
    with pytest.raises(RuntimeError, match="trusted task mount plan"):
        runtime_module.isolated_create_container(
            backend,
            "kvbench.judge:local",
            JUDGE_NAME,
            mounts=[("/host/agent_runtimes/codex-0.148.0", "/opt/agent-runtime", "ro")],
        )


def test_only_judge_gets_minimal_runtime_mounts(
    runtime_module, monkeypatch, tmp_path: Path
) -> None:
    _configure_judge_files(runtime_module, monkeypatch, tmp_path)
    plan = runtime_module.mount_plan(JUDGE_NAME)
    assert isinstance(plan, dict) and plan
    targets = _bind_targets(plan)

    assert "/var/run/docker.sock" in targets
    assert targets["/var/run/docker.sock"][1] in {"rw", ""}
    nonce_targets = [target for target in targets if "nonce" in target.lower()]
    assert nonce_targets, targets
    assert all(targets[target][1] == "ro" for target in nonce_targets)
    for baseline_target in (
        runtime_module.CONTAINER_HIDDEN_BASELINE,
        runtime_module.CONTAINER_PUBLIC_BASELINE,
    ):
        assert baseline_target in targets
        assert targets[baseline_target][1] == "ro"
    assert any("harbor-kv-qwen36" in target for target in targets)
    scratch = str(runtime_module.scratch_dir_for(JUDGE_NAME))
    assert scratch in targets
    assert targets[scratch][1] == "rw"

    # Never mount a broad host directory into the privileged Judge.  The
    # runtime, frozen evidence, nonce source, migration bundle, and per-eval
    # candidate scratch must each be explicit.
    forbidden_sources = {"/", "/home", str(Path.home()), "~"}
    assert not (set(plan) & forbidden_sources)


def test_work_and_judge_mount_plans_are_strictly_segregated(
    runtime_module, monkeypatch, tmp_path: Path
) -> None:
    _configure_judge_files(runtime_module, monkeypatch, tmp_path)
    work = runtime_module.mount_plan(WORK_NAME)
    judge = runtime_module.mount_plan(JUDGE_NAME)

    assert work == {}
    assert judge != work
    serialized_judge = repr(judge).lower()
    for sensitive in (
        "docker.sock",
        "nonce",
        "hidden-direct.json",
        "public-direct.json",
        "harbor-kv-qwen36",
    ):
        assert sensitive in serialized_judge
        assert sensitive not in repr(work).lower()


def test_replay_attestation_runs_before_create_and_after_cleanup_fail_closed(
    runtime_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    nonce_source = tmp_path / "nonce-source.jsonl"
    nonce_source.write_text("{}\n", encoding="utf-8")
    eval_root = tmp_path / "evals"
    monkeypatch.setenv(runtime_module.NONCE_SOURCE_ENV, str(nonce_source))
    monkeypatch.setenv(runtime_module.EVAL_ROOT_ENV, str(eval_root))
    monkeypatch.setenv(
        runtime_module.VLLM_BASE_URL_ENV, "http://host-backend.invalid:18000"
    )
    monkeypatch.setenv(runtime_module.REPLAY_PREFLIGHT_ENV, str(tmp_path / "baseline.json"))
    monkeypatch.setenv(runtime_module.REPLAY_PREFLIGHT_SHA_ENV, "a" * 64)
    monkeypatch.setenv(runtime_module.REPLAY_PREFLIGHT_HELPER_SHA_ENV, "b" * 64)

    verification_count = 0

    def verify() -> None:
        nonlocal verification_count
        verification_count += 1
        events.append(f"verify-{verification_count}")
        if verification_count == 2:
            raise RuntimeError("live artifacts changed")

    monkeypatch.setattr(runtime_module, "_verify_replay_live_artifacts", verify)
    monkeypatch.setattr(
        runtime_module,
        "_validate_judge_host_contract",
        lambda: events.append("host-contract"),
    )
    monkeypatch.setattr(runtime_module, "mount_plan", lambda _name: {})
    monkeypatch.setattr(
        runtime_module,
        "_original_cleanup_container",
        lambda _self, _handle, _logger=None: events.append("cleanup"),
    )

    class Containers:
        def create(self, _image: str, **kwargs: object) -> SimpleNamespace:
            events.append("create")
            return SimpleNamespace(id="judge-1", name=str(kwargs["name"]))

    backend = SimpleNamespace(client=SimpleNamespace(containers=Containers()))
    handle = runtime_module.isolated_create_container(
        backend,
        "kvbench.judge:local",
        JUDGE_NAME,
    )
    with pytest.raises(RuntimeError, match="live artifacts changed"):
        runtime_module.isolated_cleanup_container(backend, handle)

    assert events == [
        "verify-1",
        "host-contract",
        "create",
        "cleanup",
        "verify-2",
    ]
    assert not runtime_module._replay_guarded_container_ids


def test_validation_stack_requires_explicit_owner_label() -> None:
    source = VALIDATION_STACK.read_text(encoding="utf-8")

    assert (
        '${KV_GPU67_V21_OWNER_TOKEN:?set an explicit Docker ownership label}'
        in source
    )
    assert "${KV_GPU67_V21_OWNER_TOKEN:-" not in source


def test_validation_stack_rejects_legacy_controller_override_before_docker(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    launch = candidate / "launch.sh"
    launch.write_text("#!/bin/sh\n", encoding="utf-8")
    launch.chmod(0o755)
    legacy = "harbor-kv/bench-controller:qwen36-swe192-v24-public-warm300"
    completed = subprocess.run(
        [str(VALIDATION_STACK), "check"],
        env={
            **os.environ,
            "KV_AUTHOR_REFERENCE_GATEWAY_DIR": str(candidate),
            "KV_GPU67_V21_OWNER_TOKEN": "controller-allowlist-test",
            "KVBENCH_VLLM_BASE_URL": "http://127.0.0.1:18000",
            "KV_GPU67_CONTROLLER_IMAGE": legacy,
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert completed.returncode == 4
    assert completed.stdout.strip() == (
        f"unapproved validation controller image: {legacy}"
    )


def test_validation_stack_has_one_controller_image_allowlist_entry() -> None:
    source = VALIDATION_STACK.read_text(encoding="utf-8")
    approved = "harbor-kv/bench-controller:qwen36-swe192-v24-url-v1"

    assert source.count(approved) == 1
    assert "qwen36-swe192-v24-public-warm300" not in source
    assert '[[ "$controller_image" == "$frozen_controller_image" ]]' in source
