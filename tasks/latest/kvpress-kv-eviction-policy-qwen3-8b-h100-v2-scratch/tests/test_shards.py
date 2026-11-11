from __future__ import annotations

from pathlib import Path

import pytest

import judge_shards as js


def _gpu(index: int, bus: str, name: str = "NVIDIA H100 NVL", memory: int = 95830, cc=(9, 0)) -> js.Gpu:
    return js.Gpu(index, f"GPU-0000000{index}-0000-0000-0000-000000000000", name, f"00000000:{bus}:00.0", memory, cc)


def test_plan_is_deterministic_balanced_and_complete() -> None:
    ids = [f"f{i:05d}" for i in range(997)]
    costs = {x: 1000 + (i * 7919) % 25000 for i, x in enumerate(ids)}
    for n in (1, 2, 3, 4, 8):
        plan = js.plan_shards(ids, costs, n)
        assert plan == js.plan_shards(ids, costs, n)
        assert sorted(x for shard in plan for x in shard) == sorted(ids)
        loads = [sum(costs[x] for x in shard) for shard in plan]
        assert max(loads) - min(loads) <= max(costs.values())
        order = {x: i for i, x in enumerate(ids)}
        assert all(shard == sorted(shard, key=order.__getitem__) for shard in plan)


def test_plan_rejects_duplicates() -> None:
    with pytest.raises(ValueError):
        js.plan_shards(["a", "a"], {"a": 1}, 2)


def test_visible_filter_by_index_and_uuid() -> None:
    gpus = [_gpu(0, "46"), _gpu(1, "49"), _gpu(2, "4e")]
    assert js._visible_filter(gpus, {}) == gpus
    assert js._visible_filter(gpus, {"CUDA_VISIBLE_DEVICES": "2,0"}) == [gpus[2], gpus[0]]
    assert js._visible_filter(gpus, {"CUDA_VISIBLE_DEVICES": gpus[1].uuid}) == [gpus[1]]
    with pytest.raises(RuntimeError):
        js._visible_filter(gpus, {"CUDA_VISIBLE_DEVICES": "5"})


def test_fleet_must_be_identical_and_within_envelope() -> None:
    fleet = js.validate_fleet([_gpu(0, "46"), _gpu(1, "49")], min_capability=(8, 0), min_memory_bytes=46 << 30)
    assert fleet["count"] == 2 and len(fleet["devices"]) == 2
    with pytest.raises(RuntimeError, match="not identical"):
        js.validate_fleet([_gpu(0, "46"), _gpu(1, "49", name="NVIDIA H100 80GB HBM3", memory=81559)], min_capability=(8, 0), min_memory_bytes=1)
    with pytest.raises(RuntimeError, match="memory"):
        js.validate_fleet([_gpu(0, "46", memory=40000)], min_capability=(8, 0), min_memory_bytes=46 << 30)
    with pytest.raises(RuntimeError, match="capability"):
        js.validate_fleet([_gpu(0, "46", cc=(7, 5))], min_capability=(8, 0), min_memory_bytes=1)
    with pytest.raises(RuntimeError):
        js.validate_fleet([], min_capability=(8, 0), min_memory_bytes=1)


def _sysfs(tmp_path: Path, local: dict[str, str]) -> tuple[Path, Path]:
    pci = tmp_path / "pci"; cpu = tmp_path / "cpu"
    for bus, cpus in local.items():
        (pci / f"0000:{bus}:00.0").mkdir(parents=True); (pci / f"0000:{bus}:00.0" / "local_cpulist").write_text(cpus + "\n")
    for c in range(32):  # 16 cores with SMT siblings c and c+16
        (cpu / f"cpu{c}" / "topology").mkdir(parents=True)
        (cpu / f"cpu{c}" / "topology" / "thread_siblings_list").write_text(f"{c % 16},{c % 16 + 16}\n")
    return pci, cpu


def test_cpu_slices_are_numa_local_whole_cores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pci, cpu = _sysfs(tmp_path, {"46": "0-7,16-23", "49": "0-7,16-23", "c5": "8-15,24-31"})
    original = js._split_even
    monkeypatch.setattr(js, "_split_even", lambda cpus, parts, cpu_sysfs=cpu: original(cpus, parts, cpu_sysfs))
    gpus = [_gpu(0, "46"), _gpu(1, "49"), _gpu(2, "c5")]
    slices, mode = js.cpu_slices(gpus, allowed=range(32), sysfs=pci, quota=None)
    assert mode == "numa-local"
    assert slices[0] == [0, 1, 2, 3, 16, 17, 18, 19] and slices[1] == [4, 5, 6, 7, 20, 21, 22, 23]
    assert len(slices[2]) == len(slices[0]) and set(slices[2]) <= set(range(8, 16)) | set(range(24, 32))
    assert not set(slices[0]) & set(slices[1])


def test_cpu_slices_fall_back_to_even_split(tmp_path: Path) -> None:
    pci, _cpu = _sysfs(tmp_path, {})
    slices, mode = js.cpu_slices([_gpu(0, "46"), _gpu(1, "49")], allowed=range(8), sysfs=pci, quota=None)
    assert mode == "even-split" and sorted(slices[0] + slices[1]) == list(range(8))
    with pytest.raises(RuntimeError):
        js.cpu_slices([_gpu(i, f"4{i}") for i in range(3)], allowed=range(2), sysfs=pci, quota=None)


def test_shard_environment_pins_one_device() -> None:
    env = js.shard_environment({"PATH": "/bin", "CUDA_VISIBLE_DEVICES": "0,1"}, _gpu(1, "49"), [4, 5, 6, 7])
    assert "CUDA_VISIBLE_DEVICES" not in env and env["SFORGE_JUDGE_DEVICE"] == "1" and env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["OMP_NUM_THREADS"] == "4" and env["PATH"] == "/bin"


def test_cpu_quota_reads_cgroup_v2_and_v1(tmp_path: Path) -> None:
    (tmp_path / "cpu.max").write_text("1600000 100000\n")
    assert js.cpu_quota(tmp_path) == 16.0
    (tmp_path / "cpu.max").write_text("max 100000\n")
    assert js.cpu_quota(tmp_path) is None
    v1 = tmp_path / "v1"; (v1 / "cpu").mkdir(parents=True)
    (v1 / "cpu" / "cpu.cfs_quota_us").write_text("800000\n"); (v1 / "cpu" / "cpu.cfs_period_us").write_text("100000\n")
    assert js.cpu_quota(v1) == 8.0
    (v1 / "cpu" / "cpu.cfs_quota_us").write_text("-1\n")
    assert js.cpu_quota(v1) is None


def test_cpu_slices_are_capped_by_the_quota_share(tmp_path: Path) -> None:
    pci = tmp_path / "pci"
    gpus = [_gpu(0, "46"), _gpu(1, "49")]
    wide, _ = js.cpu_slices(gpus, allowed=range(32), sysfs=pci, quota=None)
    capped, mode = js.cpu_slices(gpus, allowed=range(32), sysfs=pci, quota=16.0)
    assert all(len(x) == 8 for x in capped) and mode.endswith("+quota-8")
    assert all(c == w[:8] for c, w in zip(capped, wide))
    single, _ = js.cpu_slices([_gpu(0, "46")], allowed=range(32), sysfs=pci, quota=8.0)
    assert len(single[0]) == 8
    tiny, _ = js.cpu_slices(gpus, allowed=range(32), sysfs=pci, quota=1.0)
    assert all(len(x) == 1 for x in tiny)
