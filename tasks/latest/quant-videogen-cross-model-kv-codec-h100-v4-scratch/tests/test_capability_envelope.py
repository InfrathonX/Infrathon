"""The hardware gate is a capability envelope, never a GPU model name."""

from __future__ import annotations

import inspect
import json
import math
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
JUDGE = ROOT / "judge"
LIB = ROOT / "scripts/lib.sh"
TASK_JSON = ROOT / "tasks/quant_videogen_cross_model_kv_codec_h100_scratch_v4.json"
sys.path.insert(0, str(JUDGE))

import evaluate_hidden  # noqa: E402
from conftest import release_file  # noqa: E402


MIB = 1024**2
# The release floor: 79 GiB, like the other 80 GB-class tasks.  The official
# codec peaks at about 75.4e9 bytes per card (LongCat), so H100 SXM 80 GB
# (85.0e9 bytes to torch) and H100 NVL both qualify; only the BF16 goldens,
# frozen data generated on H100 NVL, needed more (HY 97.4e9 bytes).
FLOOR = 79 * 1024**3
JUDGE_CONFIG = {
    "judge": {"min_gpu_count": 1, "max_gpu_count": 8, "min_device_memory_bytes": FLOOR},
}

FLEET = [
    ("NVIDIA H100 NVL", (9, 0), 95_830),
    ("NVIDIA H100 80GB HBM3", (9, 0), 81_559),
    ("NVIDIA A100-SXM4-80GB", (8, 0), 81_920),
    ("NVIDIA H20", (9, 0), 97_871),
    ("NVIDIA H200", (9, 0), 143_771),
    ("NVIDIA B200", (10, 0), 183_359),
]


def _rows(*, name: str = "NVIDIA H100 NVL", capability=(9, 0), total: int = 95_830 * MIB,
          count: int = 8, used: int = 600 * MIB) -> list[dict]:
    return [
        {
            "index": index,
            "name": name,
            "uuid": f"{index:08x}-1111-2222-3333-444444444444",
            "capability": list(capability),
            "total_memory_bytes": total,
            "free_memory_bytes": total - used,
            "pci_bus_id": f"00000000:{0x40 + index:02x}:00.0",
        }
        for index in range(count)
    ]


def _lib_shell_value(name: str) -> str:
    result = subprocess.run(
        ["bash", "-c", f'source "{LIB}"; printf "%s" "${{{name}}}"'],
        text=True,
        capture_output=True,
        env={**os.environ, "REPOSITORY_ROOT": str(ROOT.parents[2])},
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def _lib_has(function: str) -> bool:
    result = subprocess.run(
        ["bash", "-c", f'source "{LIB}"; declare -F {function} >/dev/null'],
        capture_output=True,
        env={**os.environ, "REPOSITORY_ROOT": str(ROOT.parents[2])},
        check=False,
    )
    return result.returncode == 0


def _run_assert_gpu_idle(
    tmp_path: Path, *, name: str, capability: str, total_mib: int
) -> subprocess.CompletedProcess[str]:
    executable = tmp_path / "nvidia-smi"
    executable.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
query=""
for argument in "$@"; do
  case "$argument" in
    --query-gpu=*) query="${{argument#--query-gpu=}}" ;;
  esac
done
case "$query" in
  name,compute_cap,memory.total,memory.used,utilization.gpu)
    echo "{name}, {capability}, {total_mib}, 0, 0" ;;
  *) echo "unsupported fake query: $query" >&2; exit 2 ;;
esac
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return subprocess.run(
        ["bash", "-c", f'source "{LIB}"; assert_gpu_idle 0 judge'],
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "REPOSITORY_ROOT": str(ROOT.parents[2]),
        },
        check=False,
    )


def _release_judge() -> dict:
    return json.loads(release_file().read_text(encoding="utf-8"))["judge"]


def test_envelope_is_declared_once_and_shared_by_the_launcher() -> None:
    floor = _release_judge()["min_device_memory_bytes"]
    assert evaluate_hidden.MIN_COMPUTE_CAPABILITY >= (8, 0)
    major, minor = evaluate_hidden.MIN_COMPUTE_CAPABILITY
    assert _lib_shell_value("GPU_MIN_COMPUTE_CAPABILITY") == f"{major}.{minor}"
    # The launcher compares whole MiB: the smallest MiB count at the floor.
    assert floor == FLOOR
    assert int(_lib_shell_value("GPU_MIN_TOTAL_MIB")) == math.ceil(floor / MIB)


def test_task_json_publishes_the_same_envelope() -> None:
    task = json.loads(TASK_JSON.read_text(encoding="utf-8"))
    portable = task["portable"]
    profile = portable["execution_profiles"][portable["defaults"]["execution_profile_id"]]
    major, minor = evaluate_hidden.MIN_COMPUTE_CAPABILITY
    assert profile["min_compute_capability"] == f"{major}.{minor}"
    assert profile["min_memory_bytes"] == _release_judge()["min_device_memory_bytes"]


@pytest.mark.parametrize("name, capability, total_mib", FLEET)
def test_any_gpu_fleet_meeting_the_envelope_is_admitted(
    name: str, capability: tuple[int, int], total_mib: int
) -> None:
    total = total_mib * MIB
    if capability < evaluate_hidden.MIN_COMPUTE_CAPABILITY or total < FLOOR:
        with pytest.raises(evaluate_hidden.InfrastructureFailure, match="envelope"):
            evaluate_hidden.judge_fleet(_rows(name=name, capability=capability, total=total), JUDGE_CONFIG, {})
        return
    gpus, fleet = evaluate_hidden.judge_fleet(_rows(name=name, capability=capability, total=total), JUDGE_CONFIG, {})
    assert fleet["name"] == name and fleet["count"] == 8 and len(gpus) == 8
    assert fleet["compute_capability"] == list(capability) and fleet["total_memory_bytes"] == total
    assert [gpu.index for gpu in gpus] == list(range(8))
    assert all(gpu.uuid.startswith("GPU-") for gpu in gpus)


def test_80_gb_class_devices_qualify_and_smaller_ones_do_not() -> None:
    # An H100 SXM 80 GB reports 85,019,590,656 bytes to torch.
    gpus, _ = evaluate_hidden.judge_fleet(_rows(name="NVIDIA H100 80GB HBM3", total=85_019_590_656), JUDGE_CONFIG, {})
    assert len(gpus) == 8
    with pytest.raises(evaluate_hidden.InfrastructureFailure, match="memory"):
        evaluate_hidden.judge_fleet(_rows(name="NVIDIA L40S", total=48_305_799_168), JUDGE_CONFIG, {})
    with pytest.raises(evaluate_hidden.InfrastructureFailure, match="memory"):
        evaluate_hidden.judge_fleet(_rows(total=FLOOR - 1), JUDGE_CONFIG, {})
    gpus, _ = evaluate_hidden.judge_fleet(_rows(total=FLOOR, used=0), JUDGE_CONFIG, {})
    assert len(gpus) == 8


def test_gpu_name_never_gates() -> None:
    _, fleet = evaluate_hidden.judge_fleet(
        _rows(name="Some Future Accelerator", capability=(12, 0), total=2**40), JUDGE_CONFIG, {}
    )
    assert fleet["name"] == "Some Future Accelerator"


def test_below_capability_floor_is_rejected() -> None:
    major, minor = evaluate_hidden.MIN_COMPUTE_CAPABILITY
    too_old = (major, minor - 1) if minor else (major - 1, 9)
    with pytest.raises(evaluate_hidden.InfrastructureFailure, match="compute capability"):
        evaluate_hidden.judge_fleet(_rows(capability=too_old, total=2**40), JUDGE_CONFIG, {})


@pytest.mark.parametrize("count", [1, 2, 7, 8])
def test_one_to_eight_identical_gpus_are_admitted(count: int) -> None:
    gpus, fleet = evaluate_hidden.judge_fleet(_rows(count=count), JUDGE_CONFIG, {})
    assert fleet["count"] == count == len(gpus)


@pytest.mark.parametrize("count", [0, 9])
def test_a_gpu_count_outside_the_release_range_is_rejected(count: int) -> None:
    with pytest.raises(evaluate_hidden.InfrastructureFailure, match="GPUs"):
        evaluate_hidden.judge_fleet(_rows(count=count), JUDGE_CONFIG, {})


def test_the_release_can_require_a_minimum_gpu_count() -> None:
    config = {"judge": {**JUDGE_CONFIG["judge"], "min_gpu_count": 4}}
    with pytest.raises(evaluate_hidden.InfrastructureFailure, match="4..8"):
        evaluate_hidden.judge_fleet(_rows(count=2), config, {})


def test_mixed_duplicate_or_busy_devices_are_rejected() -> None:
    mixed = _rows(count=2)
    mixed[1]["name"] = "NVIDIA H200"
    with pytest.raises(evaluate_hidden.InfrastructureFailure, match="not identical"):
        evaluate_hidden.judge_fleet(mixed, JUDGE_CONFIG, {})
    duplicate = _rows(count=2)
    duplicate[1]["uuid"] = duplicate[0]["uuid"]
    with pytest.raises(evaluate_hidden.InfrastructureFailure, match="duplicate GPU UUID"):
        evaluate_hidden.judge_fleet(duplicate, JUDGE_CONFIG, {})
    busy = _rows(count=3)
    busy[2]["free_memory_bytes"] = busy[2]["total_memory_bytes"] - 3 * 1024**3
    with pytest.raises(evaluate_hidden.InfrastructureFailure, match=r"\[2\] are not idle"):
        evaluate_hidden.judge_fleet(busy, JUDGE_CONFIG, {})


def test_a_runtime_cuda_visible_devices_only_narrows_the_slots() -> None:
    rows = _rows(count=8)
    gpus, fleet = evaluate_hidden.judge_fleet(rows, JUDGE_CONFIG, {"CUDA_VISIBLE_DEVICES": "6,2"})
    assert [gpu.index for gpu in gpus] == [2, 6] and fleet["count"] == 2


def test_the_inventory_probe_runs_in_a_child_with_the_identity_mapping(monkeypatch) -> None:
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["env"] = kwargs["env"]
        payload = {"devices": _rows(count=2), "torch": "x"}
        return types.SimpleNamespace(returncode=0, stdout="warning line\n" + json.dumps(payload) + "\n", stderr="")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(evaluate_hidden.subprocess, "run", fake_run)
    rows = evaluate_hidden.probe_devices()
    assert len(rows) == 2 and seen["command"][1:3] == ["-I", "-c"]
    assert "CUDA_VISIBLE_DEVICES" not in seen["env"]
    assert seen["env"]["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert "torch.cuda" in seen["command"][3] and "mem_get_info" in seen["command"][3]
    # The controller process itself never imports torch for the inventory.
    for function in (evaluate_hidden.probe_devices, evaluate_hidden.judge_fleet, evaluate_hidden.evaluator_hardware):
        assert "import torch" not in inspect.getsource(function)


@pytest.mark.parametrize(
    "result",
    [
        types.SimpleNamespace(returncode=1, stdout="", stderr="CUDA error"),
        types.SimpleNamespace(returncode=0, stdout="not json\n", stderr=""),
        types.SimpleNamespace(returncode=0, stdout='{"devices": 3}\n', stderr=""),
    ],
)
def test_a_failed_inventory_probe_is_infrastructure(monkeypatch, result) -> None:
    monkeypatch.setattr(evaluate_hidden.subprocess, "run", lambda *a, **k: result)
    with pytest.raises(evaluate_hidden.InfrastructureFailure, match="inventory"):
        evaluate_hidden.probe_devices()


@pytest.mark.parametrize("name, total_mib", [("NVIDIA H200", 143_771), ("NVIDIA H100 80GB HBM3", 81_559),
                                             ("NVIDIA H100 NVL", 95_830)])
def test_launcher_admits_devices_inside_the_envelope(tmp_path: Path, name: str, total_mib: int) -> None:
    if not _lib_has("assert_gpu_idle"):
        pytest.skip("the launcher library has no per-GPU idle check")
    result = _run_assert_gpu_idle(tmp_path, name=name, capability="9.0", total_mib=total_mib)
    assert result.returncode == 0, result.stderr
    assert name in result.stdout


@pytest.mark.parametrize(
    "name, capability, total_mib",
    [
        ("NVIDIA RTX A6000", "8.6", 49_140),
        ("NVIDIA A100-PCIE-40GB", "8.0", 40_960),
        ("NVIDIA L40S", "8.9", 46_068),
        ("Tesla V100-SXM2-32GB", "7.0", 143_771),
    ],
)
def test_launcher_rejects_devices_outside_the_envelope(
    tmp_path: Path, name: str, capability: str, total_mib: int
) -> None:
    if not _lib_has("assert_gpu_idle"):
        pytest.skip("the launcher library has no per-GPU idle check")
    result = _run_assert_gpu_idle(
        tmp_path, name=name, capability=capability, total_mib=total_mib
    )
    assert result.returncode != 0
    assert "compute capability" in result.stderr or "floor" in result.stderr


def test_no_model_name_gate_remains_in_the_trusted_or_launch_surface() -> None:
    for path in [*JUDGE.glob("*.py"), LIB]:
        text = path.read_text(encoding="utf-8")
        assert not re.search(r'"h100"\s+not\s+in', text, flags=re.IGNORECASE), path
        assert "capability != (9, 0)" not in text, path
        assert "*H100*" not in text, path
