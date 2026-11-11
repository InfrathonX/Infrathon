from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
PAIR_RUNNER = (
    ROOT
    / "runtime/frozen/harbor_run_scripts/"
    "concurrent-agent-serving-optimizer-qwen36-27b-swe192-v2.1-gpu45/"
    "run-gpu67-v2.1-pair.py"
)


def _load_pair_runner():
    spec = importlib.util.spec_from_file_location(
        "kvbench_secret_transport_contract", PAIR_RUNNER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hidden_nonce_is_stdin_only_and_redacted_on_failure(monkeypatch) -> None:
    runner = _load_pair_runner()
    canary = "a" * 63 + "b"
    observed: dict[str, object] = {}

    def successful_run(command, **kwargs):
        observed["command"] = command
        observed["input"] = kwargs.get("input")
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"result": {"valid_steps_per_min": 1.0}}),
        )

    monkeypatch.setattr(runner.subprocess, "run", successful_run)
    runner.controller_run(
        "controller-id",
        mode="candidate",
        nonce=canary,
        timeout_sec=10,
        suite="hidden",
        profile="hidden-pressure-192",
    )

    argv = [str(value) for value in observed["command"]]
    assert argv[:3] == ["docker", "exec", "-i"]
    assert canary not in "\n".join(argv)
    assert canary in str(observed["input"])

    def failed_run(command, **kwargs):
        return SimpleNamespace(returncode=9, stdout=f"controller echoed {canary}")

    monkeypatch.setattr(runner.subprocess, "run", failed_run)
    with pytest.raises(RuntimeError) as caught:
        runner.controller_run(
            "controller-id",
            mode="candidate",
            nonce=canary,
            timeout_sec=10,
            suite="hidden",
            profile="hidden-pressure-192",
        )
    assert canary not in str(caught.value)
    assert "<redacted>" in str(caught.value)


def test_formal_runtime_disables_token_bearing_access_log(runtime_module) -> None:
    import uvicorn

    assert uvicorn.run is runtime_module.private_uvicorn_run
    assert runtime_module._original_uvicorn_run is not runtime_module.private_uvicorn_run
