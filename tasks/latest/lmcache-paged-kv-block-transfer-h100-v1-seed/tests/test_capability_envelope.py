"""The hardware gate is a capability envelope, never a GPU model name."""

from __future__ import annotations

import importlib.util
import re
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


if "tilelang" not in sys.modules:
    try:
        import tilelang  # noqa: F401
    except Exception:
        sys.modules["tilelang"] = types.SimpleNamespace(__version__="0.0-test-stub")

RUNTIME_PATH = next((TASK_ROOT / "runtime").glob("*_runtime.py"))
VERIFIER_PATH = next((TASK_ROOT / "images/judge").glob("verify_*.py"))
sys.path.insert(0, str(TASK_ROOT / "runtime"))
sys.path.insert(0, str(TASK_ROOT / "images/judge"))
sys.path.insert(0, str(TASK_ROOT / "images/work/benchmark"))
runtime = _load(RUNTIME_PATH, RUNTIME_PATH.stem)
verifier = _load(VERIFIER_PATH, "task_verifier_under_test")

DEVICE_COUNT = 2 if "moe_ag" in RUNTIME_PATH.name else 1


def _fake_torch(*, name: str, capability: tuple[int, int], total_memory: int, count: int = DEVICE_COUNT):
    props = types.SimpleNamespace(
        total_memory=total_memory, uuid="GPU-fake", major=capability[0], minor=capability[1],
        name=name, multi_processor_count=100,
    )
    cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: count,
        current_device=lambda: 0,
        get_device_capability=lambda index=0: capability,
        get_device_name=lambda index=0: name,
        get_device_properties=lambda index=0: props,
        can_device_access_peer=lambda a, b: True,
    )
    return types.SimpleNamespace(cuda=cuda, __version__="0.0", version=types.SimpleNamespace(cuda="0.0"))


def _capability_of(inventory: dict) -> tuple[int, int]:
    if "devices" in inventory:
        inventory = inventory["devices"][0]
    value = inventory.get("compute_capability", inventory.get("capability"))
    if isinstance(value, str):
        match = re.fullmatch(r"sm_(\d+)(\d)", value)
        assert match, value
        return int(match.group(1)), int(match.group(2))
    return tuple(int(v) for v in value)


def _name_of(inventory: dict) -> str:
    if "devices" in inventory:
        inventory = inventory["devices"][0]
    return inventory.get("name", inventory.get("device_name"))


def _validate_record(inventory: dict) -> None:
    check = getattr(verifier, "_validate_hardware_record", None)
    if check is not None:
        check({"hardware": inventory})


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
    assert _name_of(inventory) == name
    assert _capability_of(inventory) == capability
    _validate_record(inventory)


def test_gpu_name_never_gates() -> None:
    inventory = runtime.hardware_inventory(
        _fake_torch(name="Some Future Accelerator", capability=(12, 0), total_memory=2**40)
    )
    _validate_record(inventory)


def test_below_capability_floor_is_rejected() -> None:
    major, minor = runtime.MIN_COMPUTE_CAPABILITY
    too_old = (major, minor - 1) if minor else (major - 1, 9)
    with pytest.raises(Exception, match="compute capability"):
        runtime.hardware_inventory(_fake_torch(name="NVIDIA H100", capability=too_old, total_memory=2**40))


def test_below_memory_floor_is_rejected() -> None:
    small = runtime.MIN_DEVICE_MEMORY_BYTES - 1
    with pytest.raises(Exception, match="GPU.*memory"):
        runtime.hardware_inventory(_fake_torch(name="NVIDIA H100", capability=(9, 0), total_memory=small))


def test_verifier_rejects_records_below_the_envelope() -> None:
    check = getattr(verifier, "_validate_hardware_record", None)
    if check is None:
        pytest.skip("verifier validates hardware by calling the runtime inventory directly")
    good = runtime.hardware_inventory(_fake_torch(name="NVIDIA H100", capability=(9, 0), total_memory=2**40))
    check({"hardware": good})
    major, minor = runtime.MIN_COMPUTE_CAPABILITY
    too_old = (major, minor - 1) if minor else (major - 1, 9)
    bad = dict(good)
    if isinstance(good.get("capability"), str):
        bad["capability"] = f"sm_{too_old[0]}{too_old[1]}"
    else:
        bad["capability"] = list(too_old)
    with pytest.raises(RuntimeError, match="capability envelope"):
        check({"hardware": bad})
    bad = dict(good)
    bad["total_memory_bytes"] = runtime.MIN_DEVICE_MEMORY_BYTES - 1
    with pytest.raises(RuntimeError, match="capability envelope"):
        check({"hardware": bad})


def test_wrong_device_count_is_rejected() -> None:
    with pytest.raises(Exception, match="exactly"):
        runtime.hardware_inventory(
            _fake_torch(name="NVIDIA H100", capability=(9, 0), total_memory=2**40, count=DEVICE_COUNT + 1)
        )


def test_no_model_name_gate_remains_in_trusted_sources() -> None:
    for path in [RUNTIME_PATH, *(TASK_ROOT / "images/judge").glob("*.py")]:
        text = path.read_text()
        assert '"h100" not in' not in text.lower(), path
        assert '"h100" in' not in text.lower(), path
        assert "capability != (9, 0)" not in text and "capability[0] != 9" not in text, path
        assert "!= [9, 0]" not in text and '!= "sm_90"' not in text, path
