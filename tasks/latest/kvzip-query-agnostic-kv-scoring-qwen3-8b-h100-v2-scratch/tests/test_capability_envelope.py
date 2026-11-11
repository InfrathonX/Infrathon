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
        set_device=lambda index: None,
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


def test_expandable_segments_are_switched_on_for_the_sxm_build() -> None:
    """The 80 GB H100 SXM build keeps expandable segments on: with them off, the
    trusted Phase B decode ran out of memory on a 79.18 GiB card from reserved
    fragments; a caller's own setting is still overridden."""
    import os

    assert runtime.ALLOCATOR_CONF == "expandable_segments:True"
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"].split(",")[-1] == runtime.ALLOCATOR_CONF
    assert runtime._apply_allocator_conf({}) == "expandable_segments:True"
    assert runtime._apply_allocator_conf(
        {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"}
    ) == "max_split_size_mb:128,expandable_segments:True"
    assert runtime._apply_allocator_conf(
        {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False"}
    ) == "expandable_segments:True"


def test_allocator_policy_is_applied_before_torch_is_imported() -> None:
    source = RUNTIME_PATH.read_text(encoding="utf-8")
    assert source.index("_apply_allocator_conf()") < source.index('MODEL_ID = "Qwen/Qwen3-8B"')
    assert "\nimport torch" not in source, "the runtime must keep importing torch lazily"
    for entry in (
        TASK_ROOT / "images/judge/verify_kv_scoring.py",
        TASK_ROOT / "images/judge/candidate_worker.py",
        TASK_ROOT / "images/work/benchmark/public_runner.py",
    ):
        text = entry.read_text(encoding="utf-8")
        runtime_import = text.index("from kv_scoring_runtime import")
        torch_import = text.find("import torch")
        assert torch_import == -1 or runtime_import < torch_import, entry


def test_hardware_inventory_records_the_allocator_policy() -> None:
    inventory = runtime.hardware_inventory(
        _fake_torch(name="NVIDIA H100 NVL", capability=(9, 0), total_memory=2**40)
    )
    assert inventory["allocator_conf"].endswith(runtime.ALLOCATOR_CONF)


def test_no_model_name_gate_remains_in_trusted_sources() -> None:
    for path in [RUNTIME_PATH, *(TASK_ROOT / "images/judge").glob("*.py")]:
        text = path.read_text()
        assert '"h100" not in' not in text.lower(), path
        assert '"h100" in' not in text.lower(), path
        assert "capability != (9, 0)" not in text and "capability[0] != 9" not in text, path
        assert "!= [9, 0]" not in text and '!= "sm_90"' not in text, path


def test_shard_device_binding(monkeypatch) -> None:
    bound = []
    torch = _fake_torch(name="NVIDIA H100", capability=(9, 0), total_memory=2**40, count=8)
    torch.cuda.set_device = bound.append
    monkeypatch.setenv(runtime.SHARD_DEVICE_ENV, "5")
    assert runtime.bind_device(torch) == 5 and bound == [5]
    assert runtime.hardware_inventory(torch)["device_index"] == 5
    for bad in ("8", "-1", "GPU-0", ""):
        monkeypatch.setenv(runtime.SHARD_DEVICE_ENV, bad)
        with pytest.raises(RuntimeError, match="names no visible GPU"):
            runtime.bind_device(torch)
    monkeypatch.delenv(runtime.SHARD_DEVICE_ENV)
    with pytest.raises(RuntimeError, match="exactly"):
        runtime.bind_device(torch)


def _device_ops(current: int = 2, reserved: dict | None = None, peak: dict | None = None):
    reserved = reserved or {}; peak = peak or {}
    return types.SimpleNamespace(current_device=lambda: current, device_count=lambda: 4,
                                 reserved=lambda j: reserved.get(j, 0), max_reserved=lambda j: peak.get(j, 0))


def test_candidate_must_stay_on_its_own_gpu() -> None:
    runtime._own_device_only(_device_ops(reserved={2: 1 << 30}, peak={2: 2 << 30}), 2)
    with pytest.raises(RuntimeError, match="changed the current CUDA device"):
        runtime._own_device_only(_device_ops(current=0), 2)
    with pytest.raises(RuntimeError, match="used CUDA device 3"):
        runtime._own_device_only(_device_ops(reserved={3: 2 << 20}), 2)
    with pytest.raises(RuntimeError, match="used CUDA device 0"):
        runtime._own_device_only(_device_ops(peak={0: 2 << 20}), 2)  # freed again, but the peak remains


def test_trusted_ops_capture_device_guards_before_candidate_import() -> None:
    torch = pytest.importorskip("torch")
    ops = runtime.TrustedOps.capture(torch)
    assert ops.current_device is torch.cuda.current_device and ops.device_count is torch.cuda.device_count
    assert ops.reserved is torch.cuda.memory_reserved and ops.max_reserved is torch.cuda.max_memory_reserved
    assert ops.max_memory is torch.cuda.max_memory_allocated and ops.normal_ is torch.Tensor.normal_
