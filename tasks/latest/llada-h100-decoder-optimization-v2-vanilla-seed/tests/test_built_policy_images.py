from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from conftest import load_module


def image_policy(task_root):
    return load_module(
        "built_image_policy_under_test",
        task_root / "scripts/built_image_policy.py",
    )


def test_built_judge_hash_and_work_absence_use_executable_image_checks(
    task_root, monkeypatch
):
    module = image_policy(task_root)
    expected = hashlib.sha256(
        (task_root / "workload_policy.json").read_bytes()
    ).hexdigest()
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if "sha256sum" in command:
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    f"{expected}  /opt/decoder_judge/private_data/"
                    "workload_policy.json\n"
                ),
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module.validate_judge_policy_bytes(
        engine="docker",
        image="judge@sha256:" + "a" * 64,
        container_path="/opt/decoder_judge/private_data/workload_policy.json",
        expected_sha256=expected,
    )
    module.validate_work_policy_absent(
        engine="docker",
        image="work@sha256:" + "b" * 64,
        policy_filename="workload_policy.json",
    )
    assert len(calls) == 2
    assert all(call[0][:4] == ["docker", "run", "--rm", "--network"] for call in calls)
    assert all(call[1]["timeout"] == 120 for call in calls)


def test_built_image_policy_rejects_hash_drift_and_work_exposure(task_root, monkeypatch):
    module = image_policy(task_root)
    expected = "a" * 64

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=("b" * 64) + "  /opt/decoder_judge/private_data/memory_policy.json\n",
            stderr="",
        ),
    )
    with pytest.raises(ValueError, match="differ"):
        module.validate_judge_policy_bytes(
            engine="docker",
            image="judge",
            container_path="/opt/decoder_judge/private_data/memory_policy.json",
            expected_sha256=expected,
        )

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="/workspace/private/memory_policy.json\n",
            stderr="",
        ),
    )
    with pytest.raises(ValueError, match="exposes forbidden"):
        module.validate_work_policy_absent(
            engine="docker", image="work", policy_filename="memory_policy.json"
        )


def test_built_memory_validator_executes_launcher_formula(task_root, monkeypatch):
    module = image_policy(task_root)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(
            returncode=0,
            stdout="81\t4096\t87040\n",
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module.validate_launcher_memory_contract(
        launcher_lib=task_root / "scripts/lib.sh",
        policy=task_root / "memory_policy.json",
        expected_cap_gib=81.0,
        expected_headroom_mib=4096,
        expected_admission_mib=87040,
    )
    assert calls[0][0:2] == ["bash", "-c"]
    with pytest.raises(ValueError, match="differs"):
        module.validate_launcher_memory_contract(
            launcher_lib=task_root / "scripts/lib.sh",
            policy=task_root / "memory_policy.json",
            expected_cap_gib=80.0,
            expected_headroom_mib=4096,
            expected_admission_mib=87040,
        )
