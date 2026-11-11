"""The hardware gate is a capability envelope, never a GPU model name."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

TASK_ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runtime = _load(next((TASK_ROOT / "runtime").glob("*_runtime.py")), "task_runtime_under_test")
verifier = _load(next((TASK_ROOT / "images/judge").glob("verify_*.py")), "task_verifier_under_test")


def _fake_torch(*, name: str, capability: tuple[int, int], total_memory: int, count: int = 1):
    props = types.SimpleNamespace(total_memory=total_memory, uuid="GPU-fake")
    cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: count,
        get_device_capability=lambda index=0: capability,
        get_device_name=lambda index=0: name,
        get_device_properties=lambda index=0: props,
    )
    return types.SimpleNamespace(cuda=cuda, __version__="0.0", version=types.SimpleNamespace(cuda="0.0"))


def test_runtime_and_verifier_share_one_envelope() -> None:
    assert runtime.MIN_COMPUTE_CAPABILITY == verifier.MIN_COMPUTE_CAPABILITY
    assert runtime.MIN_DEVICE_MEMORY_BYTES == verifier.MIN_DEVICE_MEMORY_BYTES
    assert runtime.MIN_COMPUTE_CAPABILITY >= (8, 0)


@pytest.mark.parametrize(
    "name, capability",
    [
        ("NVIDIA H100 NVL", (9, 0)),
        ("NVIDIA H200", (9, 0)),
        ("NVIDIA A100-SXM4-80GB", (8, 0)),
        ("NVIDIA RTX A6000", (8, 6)),
        ("NVIDIA H20", (9, 0)),
        ("NVIDIA B200", (10, 0)),
    ],
)
def test_any_gpu_meeting_the_envelope_is_admitted(name: str, capability: tuple[int, int]) -> None:
    if capability < runtime.MIN_COMPUTE_CAPABILITY:
        pytest.skip("below this task's compute-capability floor")
    torch = _fake_torch(name=name, capability=capability, total_memory=runtime.MIN_DEVICE_MEMORY_BYTES)
    inventory = runtime.hardware_inventory(torch)
    assert inventory["name"] == name
    assert inventory["capability"] == list(capability)
    assert inventory["total_memory_bytes"] == runtime.MIN_DEVICE_MEMORY_BYTES
    verifier._validate_hardware_record({"hardware": inventory})


def test_gpu_name_never_gates() -> None:
    inventory = runtime.hardware_inventory(
        _fake_torch(name="Some Future Accelerator", capability=(12, 0), total_memory=2**40)
    )
    verifier._validate_hardware_record({"hardware": inventory})


def test_below_capability_floor_is_rejected() -> None:
    major, minor = runtime.MIN_COMPUTE_CAPABILITY
    too_old = (major, minor - 1) if minor else (major - 1, 9)
    with pytest.raises(RuntimeError, match="compute capability"):
        runtime.hardware_inventory(_fake_torch(name="NVIDIA H100", capability=too_old, total_memory=2**40))
    with pytest.raises(RuntimeError, match="capability envelope"):
        verifier._validate_hardware_record(
            {"hardware": {"name": "NVIDIA H100", "count": 1, "capability": list(too_old), "total_memory_bytes": 2**40}}
        )


def test_below_memory_floor_is_rejected() -> None:
    small = runtime.MIN_DEVICE_MEMORY_BYTES - 1
    with pytest.raises(RuntimeError, match="GPU memory"):
        runtime.hardware_inventory(_fake_torch(name="NVIDIA H100", capability=(9, 0), total_memory=small))
    with pytest.raises(RuntimeError, match="capability envelope"):
        verifier._validate_hardware_record(
            {"hardware": {"name": "NVIDIA H100", "count": 1, "capability": [9, 0], "total_memory_bytes": small}}
        )


def test_multiple_visible_gpus_are_rejected() -> None:
    with pytest.raises(RuntimeError, match="exactly one"):
        runtime.hardware_inventory(_fake_torch(name="NVIDIA H100", capability=(9, 0), total_memory=2**40, count=2))


def test_no_model_name_gate_remains_in_trusted_sources() -> None:
    for path in [next((TASK_ROOT / "runtime").glob("*_runtime.py")), *(TASK_ROOT / "images/judge").glob("*.py")]:
        text = path.read_text()
        assert '"h100" not in' not in text.lower(), path
        assert "capability != (9, 0)" not in text and "capability[0] != 9" not in text, path
