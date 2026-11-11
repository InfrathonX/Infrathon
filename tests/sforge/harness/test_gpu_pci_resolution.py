from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from sforge.harness import host_topology as ht
from sforge.harness.host_topology import InventoryGpu, TopologyError


#: The real NVML child script, kept before any test replaces it.
REAL_NVML_SCRIPT = ht._NVML_INVENTORY_SCRIPT


def _uuid(index: int) -> str:
    return f"GPU-{index:08x}-1111-2222-3333-{index:012x}"


def _fake_nvidia_smi(directory: Path, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / "nvidia-smi"
    script.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    script.chmod(0o755)
    return directory


@pytest.fixture(autouse=True)
def _no_real_gpu_inventory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No test here may reach the real driver, even by mistake: a failing
    nvidia-smi comes first on PATH and the NVML child exits before it loads
    libnvidia-ml.  Tests that need a working inventory replace both."""

    stub = _fake_nvidia_smi(
        tmp_path / "no-gpu-bin", "echo 'real nvidia-smi is off limits' >&2; exit 97"
    )
    monkeypatch.setenv("PATH", f"{stub}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(
        ht, "_NVML_INVENTORY_SCRIPT", "import sys; sys.exit('NVML is off limits')"
    )


def _forbid_inventories(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(timeout: float = 0):
        raise AssertionError("procfs should have resolved this GPU")

    monkeypatch.setattr(ht, "nvml_inventory", forbidden)
    monkeypatch.setattr(ht, "nvidia_smi_inventory", forbidden)


def test_procfs_information_is_parsed(fake_host) -> None:
    host = fake_host()

    gpus = ht.read_procfs_gpus(host.procfs)

    assert sorted(gpus) == [
        "0000:46:00.0", "0000:49:00.0", "0000:4e:00.0", "0000:4f:00.0",
        "0000:c5:00.0", "0000:c6:00.0", "0000:c9:00.0", "0000:d0:00.0",
    ]
    assert gpus["0000:4e:00.0"].uuid == _uuid(2)
    assert gpus["0000:4e:00.0"].minor == 2


def test_uuid_resolves_through_procfs_without_nvml(fake_host, monkeypatch) -> None:
    host = fake_host()
    _forbid_inventories(monkeypatch)

    assert ht.resolve_gpu_pcis(
        [_uuid(2), _uuid(7)], procfs=host.procfs
    ) == ("0000:4e:00.0", "0000:d0:00.0")


def test_uninitialised_procfs_uuid_falls_back_to_nvml(fake_host, monkeypatch) -> None:
    host = fake_host(procfs_uuids=False)
    assert ht.read_procfs_gpus(host.procfs)["0000:46:00.0"].uuid is None
    calls: list[str] = []

    def nvml(timeout: float = 0):
        calls.append("nvml")
        return (InventoryGpu(5, _uuid(5), "0000:c6:00.0"),)

    monkeypatch.setattr(ht, "nvml_inventory", nvml)
    monkeypatch.setattr(ht, "nvidia_smi_inventory", lambda timeout=0: ())

    assert ht.resolve_gpu_pcis([_uuid(5)], procfs=host.procfs) == ("0000:c6:00.0",)
    # The mapping is static per process: a second lookup is cached.
    assert ht.resolve_gpu_pcis([_uuid(5)], procfs=host.procfs) == ("0000:c6:00.0",)
    assert calls == ["nvml"]


def test_indices_are_nvml_indices_not_procfs_minors(fake_host, monkeypatch) -> None:
    host = fake_host()
    # NVML index 0 is deliberately a different card than device minor 0.
    monkeypatch.setattr(
        ht,
        "nvml_inventory",
        lambda timeout=0: (
            InventoryGpu(0, _uuid(3), "0000:4f:00.0"),
            InventoryGpu(1, _uuid(0), "0000:46:00.0"),
        ),
    )

    assert ht.resolve_gpu_pcis(["0", "1"], procfs=host.procfs) == (
        "0000:4f:00.0",
        "0000:46:00.0",
    )


def test_nvml_failure_falls_back_to_one_nvidia_smi_query(fake_host, monkeypatch) -> None:
    host = fake_host()

    def broken_nvml(timeout: float = 0):
        raise TopologyError("libnvidia-ml missing")

    queries: list[list[str]] = []

    def fake_command(argv, timeout):
        queries.append(list(argv))
        assert timeout == ht.RESOLVE_TIMEOUT_SECONDS
        return (
            f"0, {_uuid(0)}, 00000000:46:00.0\n"
            f"3, {_uuid(3)}, 00000000:4F:00.0\n"
        )

    monkeypatch.setattr(ht, "nvml_inventory", broken_nvml)
    monkeypatch.setattr(ht, "_run_inventory_command", fake_command)

    assert ht.resolve_gpu_pcis(["3", "0"], procfs=host.procfs) == (
        "0000:4f:00.0",
        "0000:46:00.0",
    )
    assert queries == [
        ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader"]
    ]


def test_unresolvable_gpu_names_the_explicit_escape_hatch(fake_host, monkeypatch) -> None:
    host = fake_host()

    def broken(timeout: float = 0):
        raise TopologyError("no driver")

    monkeypatch.setattr(ht, "nvml_inventory", broken)
    monkeypatch.setattr(ht, "nvidia_smi_inventory", broken)

    with pytest.raises(TopologyError, match="explicit CPULIST@NODES") as raised:
        ht.resolve_gpu_pcis([_uuid(99)], procfs=host.procfs)
    message = str(raised.value)
    assert "procfs: no match" in message
    assert "nvml: no driver" in message
    assert "nvidia-smi: no driver" in message


def test_pci_addresses_pass_through_and_garbage_is_refused(fake_host, monkeypatch) -> None:
    host = fake_host()
    _forbid_inventories(monkeypatch)

    assert ht.resolve_gpu_pcis(["00000000:C9:00.0"], procfs=host.procfs) == (
        "0000:c9:00.0",
    )
    with pytest.raises(TopologyError, match="not a GPU UUID"):
        ht.resolve_gpu_pcis(["all"], procfs=host.procfs)


def test_nvidia_smi_output_is_normalized(monkeypatch) -> None:
    monkeypatch.setattr(
        ht,
        "_run_inventory_command",
        lambda argv, timeout: f"\n2, {_uuid(2)}, 00000000:4E:00.0\n",
    )

    assert ht.nvidia_smi_inventory() == (InventoryGpu(2, _uuid(2), "0000:4e:00.0"),)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (f"exec {shutil.which('sleep') or '/bin/sleep'} 30", "timed out after 0.5s$"),
        (None, "could not run"),
        ("echo 'NVML fail' >&2; exit 9", "exited 9: NVML fail"),
        ("echo '0, x'", "malformed"),
    ],
)
def test_nvidia_smi_failures_are_topology_errors(
    tmp_path: Path, monkeypatch, body: str | None, message: str
) -> None:
    # A stand-in nvidia-smi on PATH (or none at all) for each failure.
    directory = tmp_path / "smi"
    if body is None:
        directory.mkdir()
    else:
        _fake_nvidia_smi(directory, body)
    monkeypatch.setenv("PATH", str(directory))

    with pytest.raises(TopologyError, match=message):
        ht.nvidia_smi_inventory(timeout=0.5)


def test_a_child_that_survives_the_kill_is_abandoned(monkeypatch) -> None:
    # A child stuck in the driver (uninterruptible sleep) outlives SIGKILL;
    # subprocess.run would then wait for it without a bound.
    events: list[object] = []

    class StuckChild:
        stdout = io.StringIO()
        stderr = io.StringIO()
        returncode = None

        def __init__(self, argv, **kwargs) -> None:
            events.append(("spawn", argv[0]))

        def communicate(self, timeout=None):
            events.append(("communicate", timeout))
            raise subprocess.TimeoutExpired("nvidia-smi", timeout)

        def kill(self) -> None:
            events.append("kill")

        def wait(self, timeout=None):
            events.append(("wait", timeout))
            assert timeout is not None, "an unbounded wait would hang the caller"
            raise subprocess.TimeoutExpired("nvidia-smi", timeout)

    monkeypatch.setattr(ht.subprocess, "Popen", StuckChild)

    with pytest.raises(TopologyError, match="did not exit and was abandoned"):
        ht._run_inventory_command(["nvidia-smi"], 0.5)
    assert events == [
        ("spawn", "nvidia-smi"),
        ("communicate", 0.5),
        "kill",
        ("wait", ht.INVENTORY_REAP_SECONDS),
    ]
    assert StuckChild.stdout.closed and StuckChild.stderr.closed


def test_nvml_runs_in_an_isolated_child_process(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_command(argv, timeout):
        seen["argv"] = list(argv)
        seen["timeout"] = timeout
        return f'[{{"index": 4, "uuid": "{_uuid(4)}", "pci": "00000000:C5:00.0"}}]'

    monkeypatch.setattr(ht, "_NVML_INVENTORY_SCRIPT", REAL_NVML_SCRIPT)
    monkeypatch.setattr(ht, "_run_inventory_command", fake_command)

    assert ht.nvml_inventory(timeout=7.0) == (InventoryGpu(4, _uuid(4), "0000:c5:00.0"),)
    assert seen["argv"][:3] == [sys.executable, "-I", "-c"]
    assert "nvmlDeviceGetPciInfo_v3" in seen["argv"][3]
    assert "nvmlShutdown" in seen["argv"][3]
    assert seen["timeout"] == 7.0


def test_a_stalled_nvml_child_is_killed_at_the_timeout(monkeypatch) -> None:
    # A real child process that stalls like a wedged driver call.
    monkeypatch.setattr(ht, "_NVML_INVENTORY_SCRIPT", "import time; time.sleep(30)")

    with pytest.raises(TopologyError, match="timed out after 0.5s"):
        ht.nvml_inventory(timeout=0.5)


def test_malformed_nvml_child_output_is_refused(monkeypatch) -> None:
    monkeypatch.setattr(ht, "_NVML_INVENTORY_SCRIPT", "print('not json')")

    with pytest.raises(TopologyError, match="malformed"):
        ht.nvml_inventory(timeout=10)


def test_the_stand_in_driver_refuses_by_default() -> None:
    # The autouse guard: an unpatched inventory fails instead of reaching a GPU.
    with pytest.raises(TopologyError, match="off limits"):
        ht.nvidia_smi_inventory(timeout=10)
    with pytest.raises(TopologyError, match="off limits"):
        ht.nvml_inventory(timeout=10)


def test_resolution_caches_stay_out_of_the_replay_evaluator_identity(fake_host) -> None:
    from sforge.harness import candidate_replay

    host = fake_host()
    names = candidate_replay._loaded_sforge_module_names()
    assert "sforge.harness.host_topology" in names
    before = candidate_replay._loaded_evaluator_code_sha256(names)

    assert ht.resolve_gpu_pcis([_uuid(2)], procfs=host.procfs) == ("0000:4e:00.0",)

    # The host GPU id is cached for the process, but never hashed into it.
    assert ht._resolution_cache
    assert candidate_replay._loaded_evaluator_code_sha256(names) == before
