"""The hardware gate is a capability envelope, never a GPU model name or UUID."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

TASK_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = TASK_ROOT / "runtime"
JUDGE_DIR = TASK_ROOT / "images" / "judge"


def _stub_torch() -> None:
    """The verifier imports evaluation.py, which imports torch and sdv2_core.

    The CPU test venv has no torch and no CUDA.  Install minimal stand-ins so
    the trusted sources can be imported and inspected; the real interpreters
    (Judge and Work images) always shadow these with the pinned torch.
    """
    if "torch" not in sys.modules:
        try:
            import torch  # noqa: F401
        except Exception:
            torch = types.ModuleType("torch")
            torch.__version__ = "0.0-test-stub"
            torch.bfloat16 = object()
            torch.uint16 = object()
            torch.Tensor = type("Tensor", (), {})
            torch.cuda = types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
            torch.version = types.SimpleNamespace(cuda="0.0-test-stub")
            sys.modules["torch"] = torch
    if "sdv2_core.sink" not in sys.modules:
        try:
            import sdv2_core.sink  # noqa: F401
        except Exception:
            package = types.ModuleType("sdv2_core")
            package.__path__ = []
            sink = types.ModuleType("sdv2_core.sink")
            sink.RunLedger = type("RunLedger", (), {})
            package.sink = sink
            sys.modules["sdv2_core"] = package
            sys.modules["sdv2_core.sink"] = sink


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(RUNTIME_DIR))
sys.path.insert(0, str(JUDGE_DIR))
_stub_torch()
hardware = _load(RUNTIME_DIR / "hardware.py", "sdv2_hardware_under_test")
verifier = _load(JUDGE_DIR / "verify_sdv2_pipeline.py", "sdv2_verifier_under_test")

DEVICE_COUNT = hardware.DEVICE_COUNT


def _fake_torch(*, name: str, capability: tuple[int, int], total_memory: int, count: int = DEVICE_COUNT):
    props = types.SimpleNamespace(
        total_memory=total_memory, uuid=f"GPU-fake-{name}", major=capability[0], minor=capability[1],
        name=name, multi_processor_count=100,
    )
    cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: count,
        get_device_capability=lambda index=0: capability,
        get_device_name=lambda index=0: name,
        get_device_properties=lambda index=0: props,
        can_device_access_peer=lambda a, b: True,
    )
    return types.SimpleNamespace(cuda=cuda, __version__="0.0", version=types.SimpleNamespace(cuda="0.0"))


def _inventory(**kwargs) -> dict:
    return hardware.hardware_inventory(_fake_torch(**kwargs))


def test_runtime_and_verifier_share_one_envelope() -> None:
    assert hardware.MIN_COMPUTE_CAPABILITY == verifier.MIN_COMPUTE_CAPABILITY
    assert hardware.MIN_DEVICE_MEMORY_BYTES == verifier.MIN_DEVICE_MEMORY_BYTES
    assert hardware.MIN_HOST_MEMORY_BYTES == verifier.MIN_HOST_MEMORY_BYTES
    assert hardware.MIN_COMPUTE_CAPABILITY >= (8, 0)
    assert hardware.DEVICE_COUNT == 2


@pytest.mark.parametrize(
    "name, capability",
    [
        ("NVIDIA H100 NVL", (9, 0)),
        ("NVIDIA H200", (9, 0)),
        ("NVIDIA A100-SXM4-80GB", (8, 0)),
        ("NVIDIA RTX A6000", (8, 6)),
        ("NVIDIA H20", (9, 0)),
        ("NVIDIA B200", (10, 0)),
        ("Some Future Accelerator", (12, 0)),
    ],
)
def test_any_pair_meeting_the_envelope_is_admitted(name: str, capability: tuple[int, int]) -> None:
    if capability < hardware.MIN_COMPUTE_CAPABILITY:
        pytest.skip("below this task's compute-capability floor")
    inventory = _inventory(
        name=name, capability=capability, total_memory=hardware.MIN_DEVICE_MEMORY_BYTES
    )
    assert inventory["count"] == 2 and len(inventory["devices"]) == 2
    assert {device["name"] for device in inventory["devices"]} == {name}
    assert tuple(inventory["devices"][0]["capability"]) == capability
    verifier._validate_hardware_record({"hardware": inventory})


def test_below_capability_floor_is_rejected() -> None:
    major, minor = hardware.MIN_COMPUTE_CAPABILITY
    too_old = (major, minor - 1) if minor else (major - 1, 9)
    with pytest.raises(RuntimeError, match="compute capability"):
        _inventory(name="NVIDIA H100 NVL", capability=too_old, total_memory=2**40)


def test_below_device_memory_floor_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="memory"):
        _inventory(
            name="NVIDIA H100 NVL",
            capability=(9, 0),
            total_memory=hardware.MIN_DEVICE_MEMORY_BYTES - 1,
        )


def test_wrong_device_count_is_rejected() -> None:
    for count in (1, 3):
        with pytest.raises(RuntimeError, match="exactly"):
            _inventory(name="NVIDIA H100 NVL", capability=(9, 0), total_memory=2**40, count=count)


def test_below_host_memory_floor_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(hardware, "host_memory_bytes", lambda: hardware.MIN_HOST_MEMORY_BYTES - 1)
    with pytest.raises(RuntimeError, match="host memory"):
        _inventory(name="NVIDIA H100 NVL", capability=(9, 0), total_memory=2**40)


def test_host_memory_respects_the_cgroup_limit(tmp_path, monkeypatch) -> None:
    limit = tmp_path / "memory.max"
    limit.write_text("1024\n", encoding="utf-8")
    monkeypatch.setattr(hardware, "_CGROUP_LIMITS", (str(limit),))
    assert hardware.host_memory_bytes() == 1024


def test_verifier_rejects_records_below_the_envelope() -> None:
    good = _inventory(name="NVIDIA H100 NVL", capability=(9, 0), total_memory=2**40)
    verifier._validate_hardware_record({"hardware": good})
    major, minor = hardware.MIN_COMPUTE_CAPABILITY
    too_old = (major, minor - 1) if minor else (major - 1, 9)
    for mutate in (
        lambda record: record["devices"][0].update(capability=list(too_old)),
        lambda record: record["devices"][0].update(
            total_memory_bytes=hardware.MIN_DEVICE_MEMORY_BYTES - 1
        ),
        lambda record: record.update(host_memory_bytes=hardware.MIN_HOST_MEMORY_BYTES - 1),
        lambda record: record.update(count=1),
        lambda record: record.update(devices=record["devices"][:1]),
    ):
        import copy

        bad = copy.deepcopy(good)
        mutate(bad)
        with pytest.raises(RuntimeError, match="capability envelope"):
            verifier._validate_hardware_record({"hardware": bad})


def test_gpu_name_and_uuid_are_provenance_only() -> None:
    inventory = _inventory(name="Some Future Accelerator", capability=(12, 0), total_memory=2**40)
    assert inventory["devices"][0]["uuid"].startswith("GPU-fake")
    verifier._validate_hardware_record({"hardware": inventory})


def test_no_model_name_gate_remains_in_trusted_sources() -> None:
    sources = [
        RUNTIME_DIR / "hardware.py",
        RUNTIME_DIR / "scoring.py",
        RUNTIME_DIR / "evaluation.py",
        *JUDGE_DIR.glob("*.py"),
        TASK_ROOT / "images/work/benchmark/public_runner.py",
        TASK_ROOT / "scripts/preflight.py",
    ]
    for path in sources:
        text = path.read_text()
        lowered = text.lower()
        assert '"h100" in' not in lowered and '"h100" not in' not in lowered, path
        assert "NVIDIA H100 NVL" not in text, path
        assert "capability != (9, 0)" not in text and "!= [9, 0]" not in text, path


def test_launcher_uses_a_capability_floor_not_a_model_name() -> None:
    lib = (TASK_ROOT / "scripts/lib.sh").read_text()
    assert "assert_gpu_idle()" in lib and "assert_h100_idle" not in lib
    assert "*H100*" not in lib
    assert 'GPU_MIN_FREE_MIB:-${H100_MIN_FREE_MIB:-26624}' in lib
    for script in (TASK_ROOT / "scripts").glob("*.sh"):
        assert "assert_h100_idle" not in script.read_text(), script


def test_task_json_declares_the_same_envelope() -> None:
    import json

    spec = json.loads(
        (TASK_ROOT / "tasks/streamdiffusionv2_dual_h100_pipeline_runtime_seed_v2.json").read_text()
    )
    portable = spec["portable"]
    profile = portable["execution_profiles"][portable["defaults"]["execution_profile_id"]]
    assert profile["count"] == hardware.DEVICE_COUNT
    assert profile["min_memory_bytes"] == hardware.MIN_DEVICE_MEMORY_BYTES
    assert profile["min_host_memory_bytes"] == hardware.MIN_HOST_MEMORY_BYTES
    assert profile["min_compute_capability"] == "%d.%d" % hardware.MIN_COMPUTE_CAPABILITY
    assert portable["metrics"]["primary"]["comparability"] == "same_measurement_profile"
    assert portable["metrics"]["primary"]["baseline_policy"] == "in_run_seed_reference"
