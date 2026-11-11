"""The hardware gate is a capability envelope, never a GPU model name.

The trusted runtime used to refuse every scored cohort whose device name did
not contain ``H100`` and whose compute capability was not exactly ``(9, 0)``.
That was a qualification fact promoted to a TaskCore hard gate: an H200 or a
Blackwell part failed before any training step ran, even though nothing in the
objective, the oracle, the metric or the admission policy depends on the GPU
model.  These tests pin the replacement: a compute-capability floor, a device
memory floor, the device count the workload really needs, and a peak-memory
gate that is relative to the paired reference and to the card's own capacity.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

from conftest import LAB_ROOT, TASK_ROOT
from runtime import distributed_probe


CONTROLLER_PATH = TASK_ROOT / "images/judge/verifier/e2e_formal_controller.py"

# The fleet this task must admit once the model-name gate is gone.  Every entry
# is at or above the compute-capability floor; the memory floor is checked
# separately because it is a real, honest requirement of the scored rows.
FLEET = [
    ("NVIDIA H100 NVL", (9, 0)),
    ("NVIDIA H100 80GB HBM3", (9, 0)),
    ("NVIDIA H200", (9, 0)),
    ("NVIDIA H20", (9, 0)),
    ("NVIDIA A100-SXM4-80GB", (8, 0)),
    ("NVIDIA B200", (10, 0)),
    ("Some Future Accelerator", (12, 0)),
]


@pytest.fixture()
def controller():
    name = f"mlsysforge_capability_envelope_{id(object())}"
    spec = importlib.util.spec_from_file_location(name, CONTROLLER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


def _properties(name: str, capability: tuple[int, int], total_memory: int):
    return types.SimpleNamespace(
        name=name,
        major=capability[0],
        minor=capability[1],
        total_memory=total_memory,
        uuid="GPU-00000000-0000-0000-0000-000000000000",
    )


def _drive_probe(
    monkeypatch,
    *,
    name: str = "NVIDIA H100 NVL",
    capability: tuple[int, int] = (9, 0),
    total_memory: int | None = None,
    device_count: int = 2,
    bf16: bool = True,
):
    """Run the trusted runtime's environment validation against a fake device.

    Every collective and CUDA handle the function touches is replaced, so the
    only thing under test is the admission predicate itself.
    """

    if total_memory is None:
        total_memory = distributed_probe.MIN_DEVICE_MEMORY_BYTES
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(
        distributed_probe, "_TRUSTED_CUDA_DEVICE_COUNT", lambda: device_count
    )
    monkeypatch.setattr(distributed_probe, "_TRUSTED_CUDA_IS_AVAILABLE", lambda: True)
    monkeypatch.setattr(
        distributed_probe, "_TRUSTED_CUDA_IS_BF16_SUPPORTED", lambda: bf16
    )
    monkeypatch.setattr(
        distributed_probe, "_TRUSTED_CUDA_SET_DEVICE", lambda index: None
    )
    monkeypatch.setattr(
        distributed_probe,
        "_TRUSTED_CUDA_GET_DEVICE_PROPERTIES",
        lambda device: _properties(name, capability, total_memory),
    )
    monkeypatch.setattr(
        distributed_probe,
        "_TRUSTED_DIST_INIT_PROCESS_GROUP",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(distributed_probe, "_TRUSTED_DIST_GET_WORLD_SIZE", lambda: 2)
    monkeypatch.setattr(distributed_probe, "_TRUSTED_DIST_GET_RANK", lambda: 0)

    def _all_gather(values, value):
        values[0] = dict(value)
        values[1] = dict(value, rank=1, local_rank=1, pid=int(value["pid"]) + 1)

    monkeypatch.setattr(
        distributed_probe, "_TRUSTED_DIST_ALL_GATHER_OBJECT", _all_gather
    )
    return distributed_probe._validate_distributed_environment()


def test_runtime_and_controller_share_one_envelope(controller) -> None:
    assert (
        distributed_probe.MIN_COMPUTE_CAPABILITY == controller.MIN_COMPUTE_CAPABILITY
    )
    assert (
        distributed_probe.MIN_DEVICE_MEMORY_BYTES
        == controller.MIN_DEVICE_MEMORY_BYTES
    )
    assert distributed_probe.MIN_COMPUTE_CAPABILITY >= (8, 0)
    assert controller.DEVICE_MEMORY_HEADROOM_BYTES > 0
    # A floor that already exceeds the smallest card the workload fits would be
    # a disguised model gate; a zero floor would be a lie about the workload.
    assert 0 < distributed_probe.MIN_DEVICE_MEMORY_BYTES < 200 * 1024**3


@pytest.mark.parametrize("name, capability", FLEET)
def test_any_gpu_meeting_the_envelope_is_admitted(
    monkeypatch, name: str, capability: tuple[int, int]
) -> None:
    total_memory = distributed_probe.MIN_DEVICE_MEMORY_BYTES + 1024**3
    _, _, _, identities = _drive_probe(
        monkeypatch, name=name, capability=capability, total_memory=total_memory
    )
    assert sorted(item["rank"] for item in identities) == [0, 1]
    for item in identities:
        # Name, capability and total memory are recorded as provenance.
        assert item["device_name"] == name
        assert item["compute_capability"] == f"{capability[0]}.{capability[1]}"
        assert item["total_memory_bytes"] == total_memory


def test_below_capability_floor_is_rejected(monkeypatch) -> None:
    major, minor = distributed_probe.MIN_COMPUTE_CAPABILITY
    too_old = (major, minor - 1) if minor else (major - 1, 9)
    with pytest.raises(RuntimeError, match="compute capability"):
        _drive_probe(monkeypatch, name="NVIDIA H100 NVL", capability=too_old)


def test_below_memory_floor_is_rejected(monkeypatch) -> None:
    with pytest.raises(RuntimeError, match="below the task minimum"):
        _drive_probe(
            monkeypatch,
            name="NVIDIA H100 NVL",
            total_memory=distributed_probe.MIN_DEVICE_MEMORY_BYTES - 1,
        )


def test_native_bf16_is_still_required(monkeypatch) -> None:
    with pytest.raises(RuntimeError, match="BF16"):
        _drive_probe(monkeypatch, bf16=False)


def test_two_ranks_remain_part_of_the_task_definition(monkeypatch) -> None:
    # The scored throughput is summed over two ranks; one or three GPUs is a
    # different workload, not a slower one.
    for count in (1, 3):
        with pytest.raises(RuntimeError, match="exactly 2 CUDA devices"):
            _drive_probe(monkeypatch, device_count=count)


def _identity(
    *,
    capability: str = "8.0",
    total_memory: int | None = None,
    name: str = "Some Future Accelerator",
) -> list[dict[str, object]]:
    if total_memory is None:
        total_memory = distributed_probe.MIN_DEVICE_MEMORY_BYTES
    return [
        {
            "rank": rank,
            "pid": 1000 + rank,
            "device_name": name,
            "compute_capability": capability,
            "total_memory_bytes": total_memory,
        }
        for rank in (0, 1)
    ]


def test_controller_validates_the_envelope_and_returns_the_device_capacity(
    controller,
) -> None:
    floor = controller.MIN_DEVICE_MEMORY_BYTES
    identities = _identity(total_memory=floor + 7)
    assert controller._validate_hardware_record(identities) == floor + 7
    # The smaller of the two cards decides the ceiling.
    identities[1]["total_memory_bytes"] = floor + 3
    assert controller._validate_hardware_record(identities) == floor + 3


@pytest.mark.parametrize("name, capability", FLEET)
def test_controller_hardware_record_never_reads_the_model_name(
    controller, name: str, capability: tuple[int, int]
) -> None:
    controller._validate_hardware_record(
        _identity(name=name, capability=f"{capability[0]}.{capability[1]}")
    )


@pytest.mark.parametrize(
    "mutation",
    [
        {"compute_capability": "7.5"},
        {"compute_capability": "not-a-capability"},
        {"total_memory_bytes": 1},
        {"total_memory_bytes": "80GiB"},
    ],
)
def test_controller_rejects_records_below_the_envelope(
    controller, mutation: dict[str, object]
) -> None:
    identities = _identity()
    identities[1].update(mutation)
    with pytest.raises(RuntimeError, match="capability envelope|compute capability"):
        controller._validate_hardware_record(identities)


def test_controller_rejects_a_record_missing_the_device_capacity(controller) -> None:
    identities = _identity()
    del identities[0]["total_memory_bytes"]
    with pytest.raises(RuntimeError, match="capability envelope"):
        controller._validate_hardware_record(identities)


def test_no_gpu_model_name_gate_remains_in_the_trusted_sources() -> None:
    sources = [LAB_ROOT / "runtime/distributed_probe.py"]
    sources += sorted((TASK_ROOT / "images/judge/verifier").glob("*.py"))
    sources.append(TASK_ROOT / "images/judge/run_judge.py")
    sources.append(TASK_ROOT / "scripts/preflight.py")
    for path in sources:
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        assert '"h100" not in' not in lowered, path
        assert '"h100" in' not in lowered, path
        assert "nvidia h100" not in lowered, path
        assert "!= (9, 0)" not in text and "== (9, 0)" not in text, path
        assert "!= [9, 0]" not in text and '!= "sm_90"' not in text, path
