"""Sharded Judge: plan, slot isolation, fail-fast, aggregation, GPU-count independence.

Everything here runs on the CPU with fake devices and fake case runners; no
test starts a GPU process.
"""

from __future__ import annotations

import importlib.util
import json
import math
import random
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

from conftest import HARDWARE_TYPES, quality_file, release_file


ROOT = Path(__file__).resolve().parents[1]
JUDGE = ROOT / "judge"
sys.path.insert(0, str(JUDGE))

import evaluate_hidden as eh  # noqa: E402
import judge_shards as js  # noqa: E402


DIMS = eh.VBENCH_DIMENSIONS
SHA = "0" * 64
LONGCAT_INDICES = (12, 185, 320, 593, 893, 894)
HY_NAMES = [f"{i:02d}" for i in range(1, 11)] + ["qvg"]
PAPER_POSE = "w-8,s-8,a-8,d-8,up-8,down-8,w-8,s-8,a-7"
# A zero point of q that fits make_release's synthetic BF16 readings and gates.
SYNTHETIC_QUALITY_REFERENCE = {
    "longcat": {"psnr": 27.0, "ssim": 0.91, "lpips": 0.06, "vbench": {d: 89.7 for d in DIMS}},
    "hy_worldplay_8b": {"psnr": 28.0, "ssim": 0.91, "lpips": 0.06, "vbench": {d: 94.7 for d in DIMS}},
}


def _shipped_quality_reference(name: str, hardware: str = "h100-94gb-400w") -> dict:
    """The zero point of q one device class's quality file freezes (skips without the Judge-only asset)."""
    quality = json.loads(quality_file(hardware).read_text(encoding="utf-8"))
    return quality["models"][name]["quality_reference"]


def _quality_doc(release: dict, hardware: str = "h100-94gb-400w") -> dict:
    """Move make_release's embedded references into a device-class quality file."""
    models = {name: {"quality_reference": model.pop("quality_reference")} for name, model in release["models"].items()}
    return {"schema_version": 1, "hardware": hardware, "models": models,
            "oracle_calibration": {"score_band": [1.0, 2.0]}}


def _gates(psnr: float, ssim: float, lpips: float, drop: float) -> dict:
    return {"psnr_min": psnr, "ssim_min": ssim, "lpips_max": lpips, "vbench_allowed_drop": {d: drop for d in DIMS}}


def make_release(status: str = "calibrated", **judge: object) -> dict:
    """The spec's 6 + 11 case table with placeholder digests and gates."""
    cases = []
    for index in LONGCAT_INDICES:
        cases.append({
            "case_id": f"lc-{index:03d}", "model": "longcat", "seed": 0, "prompt_index": index,
            "prompt": f"prompt line {index}", "segments": 10,
            "init_video": f"/eval/hidden/longcat/{index:03d}/init.mp4", "init_video_sha256": SHA,
            "bf16_frames": f"/eval/hidden/longcat/{index:03d}/bf16-10seg.frames.pt", "bf16_frames_sha256": SHA,
            "frame_count": 273, "fidelity_frames": [73, 93], "bf16_vbench": {d: 90.0 for d in DIMS},
            "bf16_kv_bytes": None if status == "calibrating" else 1_000_000 * index,
        })
    for position, name in enumerate(HY_NAMES):
        suffix = ".jpeg" if name in ("05", "06") else ".png"
        cases.append({
            "case_id": f"hy-{name}", "model": "hy_worldplay_8b", "seed": 0,
            "image": f"/eval/hidden/hy/{name}/image{suffix}", "image_sha256": SHA,
            "prompt": f"caption {name}", "video_length": 285, "pose": PAPER_POSE,
            "bf16_frames": f"/eval/hidden/hy/{name}/bf16-285.frames.pt", "bf16_frames_sha256": SHA,
            "frame_count": 285, "fidelity_frames": [29, 45], "bf16_vbench": {d: 95.0 for d in DIMS},
            "bf16_kv_bytes": None if status == "calibrating" else 2_000_000 * (position + 1),
        })
    release = {
        "status": status,
        "protocol_schema_version": 2,
        "paper_protocol_sha256": SHA,
        "metric_implementation_sha256": {"fidelity": SHA, "vbench": SHA},
        "prompt_file": "/eval/hidden/prompts/MovieGenVideoBench_extended.txt",
        "prompt_file_sha256": SHA,
        "judge": {
            "max_gpu_count": 8, "min_gpu_count": 1, "slot_uids": list(range(23001, 23009)),
            "master_port_base": 29600, "threads_per_slot": 4, "start_stagger_seconds": 20,
            "case_overhead_seconds": 120, "controller_overhead_seconds": 300,
            "min_device_memory_bytes": 99_000_000_000, **judge,
        },
        "models": {
            "longcat": {
                "fps": 15, "stage_seconds": 3000, "fidelity_metric_seconds": 600, "vbench_metric_seconds": 1200,
                "profile_key": "persistent_kv", "frames_file": "final.frames.pt",
                "gates": {"mean": _gates(26.0, 0.90, 0.07, 0.5), "case_floor": _gates(25.0, 0.87, 0.10, 1.0)},
                "quality_reference": json.loads(json.dumps(SYNTHETIC_QUALITY_REFERENCE["longcat"])),
            },
            "hy_worldplay_8b": {
                "fps": 24, "stage_seconds": 900, "fidelity_metric_seconds": 600, "vbench_metric_seconds": 900,
                "profile_key": "persistent_vision_kv", "frames_file": "gen.frames.pt",
                "gates": {"mean": _gates(27.0, 0.90, 0.08, 0.5), "case_floor": _gates(26.0, 0.87, 0.11, 1.0)},
                "quality_reference": json.loads(json.dumps(SYNTHETIC_QUALITY_REFERENCE["hy_worldplay_8b"])),
            },
        },
        "cases": cases,
    }
    return eh.validate_release(release)


def fake_gpu(index: int, *, free: int = 99_000_000_000) -> js.Gpu:
    return js.Gpu(index, f"GPU-{index:08d}-0000-0000-0000-000000000000", "NVIDIA H100 NVL",
                  f"00000000:{0x40 + index:02x}:00.0", 100_000_000_000, free, (9, 0))


def make_slots(release: dict, count: int) -> list[eh.Slot]:
    gpus = [fake_gpu(i) for i in range(count)]
    return eh.build_slots(release, gpus, [list(range(4 * i, 4 * i + 4)) for i in range(count)])


def fake_row(case: dict, *, wall: float = 100.0) -> dict:
    """Deterministic per-case readings (independent of slot and GPU count)."""
    order = int("".join(ch for ch in case["case_id"] if ch.isdigit()) or "99")
    bf16 = case["bf16_kv_bytes"] or 1_000_000 * (order + 1)
    ratio = 6.0 + (order % 5) * 0.5
    return {
        "compression_ratio": bf16 / int(bf16 / ratio),
        "stored_kv_bytes": int(bf16 / ratio),
        "bf16_kv_bytes": bf16,
        "other_device_allocated_bytes": 0,
        "wall_seconds": wall,
        "fidelity": {"psnr": 28.0 + (order % 3) * 0.1, "ssim": 0.92, "lpips": 0.05},
        "vbench": {d: case["bf16_vbench"][d] - 0.1 for d in DIMS},
    }


class FakeControl:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.calls: list[tuple[str, tuple[int, ...]]] = []

    def kill_uids(self, uids) -> None:
        with self.lock:
            self.calls.append((threading.current_thread().name, tuple(uids)))


class FakeRunner:
    """Records what each case would run with; never starts a process."""

    def __init__(self, release: dict, *, durations: dict | None = None, fail: dict | None = None,
                 hang: set | None = None, scale: float = 0.0) -> None:
        self.release = release
        self.durations = durations or {}
        self.fail = fail or {}
        self.hang = hang or set()
        self.scale = scale
        self.lock = threading.Lock()
        self.calls: list[dict] = []
        self.active: dict[int, str] = {}
        self.overlaps: list[tuple[int, str, str]] = []
        self._stages = tempfile.TemporaryDirectory(prefix="qvg-fake-stages-")
        self.root = Path(self._stages.name)

    def run_case(self, case: dict, slot: eh.Slot, control: dict, state: eh.RunState) -> dict:
        with self.lock:
            if slot.index in self.active:
                self.overlaps.append((slot.index, self.active[slot.index], case["case_id"]))
            self.active[slot.index] = case["case_id"]
        stage = Path(tempfile.mkdtemp(prefix=f"slot{slot.index}-", dir=self.root))
        try:
            command = eh.runner_command(self.release, case, stage=stage, master_port=control["master_port"],
                                        device_index=slot.gpu.index)
            with self.lock:
                self.calls.append({
                    "case_id": case["case_id"], "slot": slot.index, "uid": slot.uid, "device": slot.gpu.index,
                    "port": control["master_port"], "stage": str(stage), "command": command,
                    "environment": eh.candidate_environment(stage, slot),
                    "metrics": eh.metric_commands(self.release, case, candidate_frames=stage / "f.pt",
                                                  output=stage, device_index=slot.gpu.index),
                    "start": time.monotonic(),
                })
            if case["case_id"] in self.hang:
                # A case the cancel kills: it fails only because its process died.
                state.cancel.wait(10)
                raise eh.CandidateFailure("candidate execution returned exit -9")
            seconds = self.durations.get(case["case_id"], 0.0) * self.scale
            if seconds:
                time.sleep(seconds)
            if case["case_id"] in self.fail:
                raise self.fail[case["case_id"]]
            return fake_row(case)
        finally:
            with self.lock:
                self.active.pop(slot.index, None)


def run(release: dict, gpu_count: int, runner: FakeRunner, *, state_factory=eh.RunState):
    slots = make_slots(release, gpu_count)
    plan = eh.schedule_plan(release, gpu_count)
    control = FakeControl()
    controls, rows, state = eh.run_suite(release, slots, plan, runner, control=control,
                                         registry=eh.ChildRegistry(), state_factory=state_factory)
    return slots, plan, control, controls, rows, state


class InstantStagger(eh.RunState):
    def wait(self, seconds: float) -> bool:
        return self.cancel.is_set()


class ScaledStagger(eh.RunState):
    scale = 1e-4

    def wait(self, seconds: float) -> bool:
        return super().wait(seconds * self.scale)


# --------------------------------------------------------------------------
# judge_shards
# --------------------------------------------------------------------------


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


def test_plan_rejects_duplicates_and_bad_offsets() -> None:
    with pytest.raises(ValueError):
        js.plan_shards(["a", "a"], {"a": 1}, 2)
    with pytest.raises(ValueError):
        js.plan_shards(["a"], {"a": 1}, 2, initial=[0])
    with pytest.raises(ValueError):
        js.plan_shards(["a"], {"a": 1}, 0)


def test_suite_plan_is_deterministic_and_puts_one_longcat_per_slot() -> None:
    release = make_release()
    plan = eh.schedule_plan(release, 8)
    assert plan == eh.schedule_plan(make_release(), 8)
    longcat = [[x for x in shard if x.startswith("lc-")] for shard in plan]
    assert [len(x) for x in longcat] == [1, 1, 1, 1, 1, 1, 0, 0]
    assert sorted(x for shard in plan for x in shard) == sorted(c["case_id"] for c in release["cases"])
    # Inside a slot the release order is kept: the LongCat case runs first.
    assert all(shard[0].startswith("lc-") for shard in plan[:6])
    single = eh.schedule_plan(release, 1)
    assert single == [[c["case_id"] for c in release["cases"]]]


def test_visible_filter_by_index_and_uuid() -> None:
    gpus = [fake_gpu(0), fake_gpu(1), fake_gpu(2)]
    assert js._visible_filter(gpus, {}) == gpus
    assert js._visible_filter(gpus, {"CUDA_VISIBLE_DEVICES": "2,0"}) == [gpus[0], gpus[2]]
    assert js._visible_filter(gpus, {"CUDA_VISIBLE_DEVICES": gpus[1].uuid}) == [gpus[1]]
    with pytest.raises(RuntimeError):
        js._visible_filter(gpus, {"CUDA_VISIBLE_DEVICES": "5"})


def test_probe_rows_normalize_the_uuid_and_reject_mig_or_garbage() -> None:
    row = {"index": 3, "name": "NVIDIA H100 NVL", "uuid": "1234-abcd", "capability": [9, 0],
           "total_memory_bytes": 100, "free_memory_bytes": 90, "pci_bus_id": "00000000:4e:00.0"}
    gpu = js.gpu_from_probe(row)
    assert gpu.uuid == "GPU-1234-abcd" and gpu.index == 3 and gpu.compute_capability == (9, 0)
    assert js.gpu_from_probe({**row, "uuid": "GPU-1234-abcd"}).uuid == "GPU-1234-abcd"
    for bad in ({**row, "uuid": "MIG-1"}, {**row, "uuid": ""}, {**row, "index": True},
                {**row, "capability": [9]}, {k: v for k, v in row.items() if k != "total_memory_bytes"}):
        with pytest.raises(RuntimeError):
            js.gpu_from_probe(bad)


def _sysfs(tmp_path: Path, local: dict[str, str]) -> tuple[Path, Path]:
    pci = tmp_path / "pci"; cpu = tmp_path / "cpu"
    for bus, cpus in local.items():
        (pci / f"0000:{bus}:00.0").mkdir(parents=True); (pci / f"0000:{bus}:00.0" / "local_cpulist").write_text(cpus + "\n")
    for c in range(32):  # 16 cores with SMT siblings c and c+16
        (cpu / f"cpu{c}" / "topology").mkdir(parents=True)
        (cpu / f"cpu{c}" / "topology" / "thread_siblings_list").write_text(f"{c % 16},{c % 16 + 16}\n")
    return pci, cpu


def _bus_gpu(index: int, bus: str) -> js.Gpu:
    return js.Gpu(index, f"GPU-{index}", "NVIDIA H100 NVL", f"00000000:{bus}:00.0", 100, 100, (9, 0))


def test_cpu_slices_are_numa_local_whole_cores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pci, cpu = _sysfs(tmp_path, {"46": "0-7,16-23", "49": "0-7,16-23", "c5": "8-15,24-31"})
    original = js._split_even
    monkeypatch.setattr(js, "_split_even", lambda cpus, parts, cpu_sysfs=cpu: original(cpus, parts, cpu_sysfs))
    gpus = [_bus_gpu(0, "46"), _bus_gpu(1, "49"), _bus_gpu(2, "c5")]
    slices, mode = js.cpu_slices(gpus, allowed=range(32), sysfs=pci, quota=None)
    assert mode == "numa-local"
    assert slices[0] == [0, 1, 2, 3, 16, 17, 18, 19] and slices[1] == [4, 5, 6, 7, 20, 21, 22, 23]
    assert len(slices[2]) == len(slices[0]) and set(slices[2]) <= set(range(8, 16)) | set(range(24, 32))
    assert not set(slices[0]) & set(slices[1])


def test_cpu_slices_fall_back_to_even_split_and_respect_the_quota(tmp_path: Path) -> None:
    pci, _cpu = _sysfs(tmp_path, {})
    slices, mode = js.cpu_slices([_bus_gpu(0, "46"), _bus_gpu(1, "49")], allowed=range(8), sysfs=pci, quota=None)
    assert mode == "even-split" and sorted(slices[0] + slices[1]) == list(range(8))
    with pytest.raises(RuntimeError):
        js.cpu_slices([_bus_gpu(i, f"4{i}") for i in range(3)], allowed=range(2), sysfs=pci, quota=None)
    gpus = [_bus_gpu(i, f"4{i}") for i in range(8)]
    capped, mode = js.cpu_slices(gpus, allowed=range(64), sysfs=pci, quota=32.0)
    assert all(len(x) == 4 for x in capped) and mode.endswith("+quota-4")


def test_every_slot_gets_exactly_threads_per_slot_cpus() -> None:
    assert js.slot_cpus([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9]], 4) == [[0, 1, 2, 3], [6, 7, 8, 9]]
    with pytest.raises(RuntimeError, match="fewer than 4"):
        js.slot_cpus([[0, 1, 2, 3], [4, 5]], 4)


def test_cpu_quota_reads_cgroup_v2_and_v1(tmp_path: Path) -> None:
    (tmp_path / "cpu.max").write_text("3200000 100000\n")
    assert js.cpu_quota(tmp_path) == 32.0
    (tmp_path / "cpu.max").write_text("max 100000\n")
    assert js.cpu_quota(tmp_path) is None
    v1 = tmp_path / "v1"; (v1 / "cpu").mkdir(parents=True)
    (v1 / "cpu" / "cpu.cfs_quota_us").write_text("800000\n"); (v1 / "cpu" / "cpu.cfs_period_us").write_text("100000\n")
    assert js.cpu_quota(v1) == 8.0


def test_shard_environment_keeps_every_gpu_visible_and_binds_by_index() -> None:
    env = js.shard_environment({"PATH": "/bin", "CUDA_VISIBLE_DEVICES": "0,1", "CUDA_DEVICE_SM_LIMIT": "100"},
                               fake_gpu(5), [4, 5, 6, 7])
    assert "CUDA_VISIBLE_DEVICES" not in env
    # PyroMind's SM limit of 100 starts HAMi's utilization watcher, which makes
    # every allocation ~1 s with eight slots; 0 (no limit) turns it off.
    assert env["CUDA_DEVICE_SM_LIMIT"] == "0"
    assert env["QVG_DEVICE_INDEX"] == "5" and env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == "4" and env["PATH"] == "/bin"
    assert js.taskset_command("/usr/bin/taskset", [4, 5], ["python3", "x.py"]) == [
        "/usr/bin/taskset", "--cpu-list", "4,5", "python3", "x.py"]


# --------------------------------------------------------------------------
# Sharded execution
# --------------------------------------------------------------------------


def test_every_case_gets_a_distinct_port_uid_dir_and_device() -> None:
    release = make_release()
    runner = FakeRunner(release)
    slots, plan, _control, controls, rows, state = run(release, 8, runner, state_factory=InstantStagger)
    assert state.first_error is None and set(rows) == {c["case_id"] for c in release["cases"]}
    assert not runner.overlaps  # a slot never runs two cases at once
    by_case = {call["case_id"]: call for call in runner.calls}
    assert len({call["port"] for call in runner.calls}) == len(release["cases"])
    assert len({call["stage"] for call in runner.calls}) == len(release["cases"])
    for order, case in enumerate(release["cases"]):
        call = by_case[case["case_id"]]
        slot = slots[call["slot"]]
        assert case["case_id"] in plan[slot.index]
        assert call["port"] == 29600 + order == controls[case["case_id"]]["master_port"]
        assert call["uid"] == slot.uid == 23001 + slot.index
        assert call["device"] == slot.gpu.index
        command = call["command"]
        assert command[command.index("--master-port") + 1] == str(29600 + order)
        assert command[command.index("--device-index") + 1] == str(slot.gpu.index)
        environment = call["environment"]
        assert "CUDA_VISIBLE_DEVICES" not in environment
        assert environment["QVG_DEVICE_INDEX"] == str(slot.gpu.index)
        assert environment["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        assert environment["OMP_NUM_THREADS"] == environment["MKL_NUM_THREADS"] == "4"
        for metric in call["metrics"]:
            assert metric[metric.index("--device") + 1] == f"cuda:{slot.gpu.index}"
        control = controls[case["case_id"]]
        assert control["status"] == "ok" and control["gpu_uuid"] == slot.gpu.uuid and control["uid"] == slot.uid
        assert control["cpus"] == list(slot.cpus)
    # Distinct slots never share a UID, a device or a CPU.
    assert len({s.uid for s in slots}) == len({s.gpu.index for s in slots}) == 8
    assert not set().union(*[set(a.cpus) & set(b.cpus) for a in slots for b in slots if a is not b])


def test_slot_cleanup_kills_only_its_own_uid_and_global_kills_bracket_the_run() -> None:
    release = make_release()
    slots = make_slots(release, 3)
    plan = eh.schedule_plan(release, 3)
    control = FakeControl()

    class Runner(eh.CaseRunner):
        def create_stage(self, slot):
            return Path(tempfile.mkdtemp(prefix=f"slot{slot.index}-"))

        def stage_inputs(self, case, stage, slot):
            pass

        def run_candidate(self, command, *, stage, slot, log, timeout, state=None):
            case = next(c for c in release["cases"] if c["case_id"] == log.parent.name)
            run_dir = stage / "output" / eh.RUN_DIRS[case["model"]].format(seed=case["seed"])
            run_dir.mkdir(parents=True)
            row = fake_row(case)
            key = release["models"][case["model"]]["profile_key"]
            (run_dir / "profile.json").write_text(
                json.dumps({key: {"stored_kv_bytes": row["stored_kv_bytes"], "bf16_kv_bytes": row["bf16_kv_bytes"],
                                  "calls": eh.expected_compression_calls(case)},
                            "wall_seconds": 10.0, "other_device_allocated_bytes": 0,
                            "cuda_device_index": slot.gpu.index}))
            time.sleep(0.01)

        def freeze_output(self, output):
            pass

        def check_frames(self, case, frames):
            pass

        def run_metrics(self, case, slot, frames, logs, state):
            row = fake_row(case)
            return row["fidelity"], row["vbench"]

    with tempfile.TemporaryDirectory() as logs:
        runner = Runner(release, tools=eh.Tools("setpriv", "pkill", "taskset"), control=control,
                        registry=eh.ChildRegistry(), log_root=Path(logs), scratch=Path(logs), prompts=[])
        _controls, rows, state = eh.run_suite(release, slots, plan, runner, control=control,
                                              registry=eh.ChildRegistry(), state_factory=InstantStagger)
    assert state.first_error is None and len(rows) == len(release["cases"])
    everyone = tuple(slot.uid for slot in slots)
    calls = [uids for _thread, uids in control.calls]
    assert calls[0] == everyone and calls[-1] == everyone
    middle = calls[1:-1]
    assert all(len(uids) == 1 for uids in middle)
    for slot, shard in zip(slots, plan):
        # Before and after each of its own cases, never anyone else's.
        assert middle.count((slot.uid,)) == 2 * len(shard)


def test_first_candidate_failure_cancels_every_slot_and_decides_the_outcome() -> None:
    release = make_release()
    failing = "lc-185"
    hang = {c["case_id"] for c in release["cases"]} - {failing}
    runner = FakeRunner(release, fail={failing: eh.CandidateFailure("candidate execution returned exit 1")},
                        hang=hang, durations={failing: 1.0}, scale=0.05)
    _slots, plan, control, controls, rows, state = run(release, 8, runner, state_factory=InstantStagger)
    assert state.first_error["kind"] == "candidate_failure" and state.first_error["case_id"] == failing
    assert controls[failing]["status"] == "candidate_failure"
    others = [controls[c]["status"] for c in controls if c != failing]
    assert others and set(others) == {"cancelled"}
    assert not rows
    # The cancel killed every slot UID while the other slots were still running.
    everyone = tuple(range(23001, 23009))
    assert [uids for _t, uids in control.calls].count(everyone) == 3  # start, cancel, end
    with pytest.raises(eh.CandidateFailure):
        eh.raise_first_error(state.first_error)


@pytest.mark.parametrize("error", [eh.InfrastructureFailure("a trusted metric returned an error"), ValueError("bug")])
def test_first_infrastructure_failure_is_infrastructure(error: Exception) -> None:
    release = make_release()
    # The first case of an HY-only slot, so it starts while the others hang.
    failing = eh.schedule_plan(release, 8)[7][0]
    hang = {c["case_id"] for c in release["cases"]} - {failing}
    runner = FakeRunner(release, fail={failing: error}, hang=hang)
    *_rest, controls, rows, state = run(release, 8, runner, state_factory=InstantStagger)
    assert state.first_error["kind"] == "infrastructure_failure" and state.first_error["case_id"] == failing
    assert controls[failing]["status"] == "infrastructure_failure"
    assert {controls[c]["status"] for c in controls if c != failing} == {"cancelled"}
    with pytest.raises(eh.InfrastructureFailure):
        eh.raise_first_error(state.first_error)


def test_a_failure_after_the_cancel_never_overrides_the_first() -> None:
    state = eh.RunState(lambda: None)
    assert state.fail("candidate_failure", "lc-012", 0, "first") is True
    assert state.fail("infrastructure_failure", "hy-01", 1, "caused by the cancel") is False
    assert state.first_error["kind"] == "candidate_failure" and state.cancel.is_set()


def test_one_and_eight_gpus_give_the_identical_score_and_rows() -> None:
    release = make_release()
    results = {}
    for count in (1, 2, 3, 8):
        *_rest, controls, rows, state = run(release, count, FakeRunner(release), state_factory=InstantStagger)
        assert state.first_error is None
        result = eh.finalize(release, rows, controls, {})
        results[count] = (result["score"], result["valid"], {k: v["row"] for k, v in result["cases"].items()})
    assert results[1] == results[2] == results[3] == results[8]
    assert results[8][1] is True and results[8][0] > 1.0


def test_wall_time_stays_inside_the_controller_bound_for_random_durations() -> None:
    release = make_release()
    rng = random.Random(20261006)
    scale = ScaledStagger.scale
    for count in (4, 8):
        durations = {c["case_id"]: rng.uniform(0.1, 1.0) * eh.model_unit_seconds(release, c["model"])
                     for c in release["cases"]}
        runner = FakeRunner(release, durations=durations, scale=scale)
        started = time.monotonic()
        *_rest, rows, state = run(release, count, runner, state_factory=ScaledStagger)
        wall = time.monotonic() - started
        assert state.first_error is None and len(rows) == len(release["cases"])
        bound = eh.controller_budget_seconds(release, count) - release["judge"]["controller_overhead_seconds"]
        assert wall <= bound * scale + 0.5


# --------------------------------------------------------------------------
# Aggregation and gates
# --------------------------------------------------------------------------


def _rows(release: dict) -> dict:
    return {case["case_id"]: fake_row(case) for case in release["cases"]}


def test_ratio_is_the_byte_sum_ratio_not_the_mean_of_case_ratios() -> None:
    release = make_release()
    rows = _rows(release)
    summary = eh.aggregate(release, rows)
    for name in eh.MODELS:
        members = [rows[c["case_id"]] for c in release["cases"] if c["model"] == name]
        stored = sum(r["stored_kv_bytes"] for r in members)
        bf16 = sum(r["bf16_kv_bytes"] for r in members)
        model = summary["models"][name]
        assert model["stored_kv_bytes"] == stored and model["bf16_kv_bytes"] == bf16
        assert model["compression_ratio"] == bf16 / stored
        mean_of_ratios = sum(r["bf16_kv_bytes"] / r["stored_kv_bytes"] for r in members) / len(members)
        assert not math.isclose(model["compression_ratio"], mean_of_ratios, rel_tol=1e-9)
        assert model["case_count"] == (6 if name == "longcat" else 11)
    expected = math.sqrt(summary["models"]["longcat"]["compression_ratio"]
                         * summary["models"]["hy_worldplay_8b"]["compression_ratio"])
    assert summary["compression_score"] == expected and summary["passed"] is True
    weighted = [summary["models"][name]["compression_ratio"] * (1 + summary["models"][name]["quality"]["q"])
                for name in eh.MODELS]
    assert summary["score"] == pytest.approx(math.sqrt(weighted[0] * weighted[1]), rel=1e-12)


def test_a_heavy_case_dominates_the_byte_sum_like_the_paper_total() -> None:
    release = make_release()
    rows = _rows(release)
    # One LongCat case stores far more than the rest: its ratio pulls R down
    # in proportion to its bytes, which a mean of ratios would hide.
    rows["lc-012"].update(stored_kv_bytes=rows["lc-012"]["bf16_kv_bytes"], compression_ratio=1.0)
    rows["lc-894"].update(bf16_kv_bytes=10 * rows["lc-894"]["bf16_kv_bytes"])
    before = eh.aggregate(release, _rows(release))["models"]["longcat"]["compression_ratio"]
    after = eh.aggregate(release, rows)["models"]["longcat"]["compression_ratio"]
    assert after != before


def test_mean_gate_compares_mean_vbench_with_mean_bf16_vbench_minus_the_drop() -> None:
    release = make_release()
    rows = _rows(release)
    drop = release["models"]["hy_worldplay_8b"]["gates"]["mean"]["vbench_allowed_drop"]["image_quality"]
    hy = [c for c in release["cases"] if c["model"] == "hy_worldplay_8b"]
    assert len(hy) == 11 and drop == 0.5
    # Exactly at the boundary on average (cases above and below it, all
    # values exact binary fractions) passes.
    deltas = [0.25 if position % 2 else -0.25 for position in range(10)] + [0.0]
    for case, delta in zip(hy, deltas):
        rows[case["case_id"]]["vbench"]["image_quality"] = case["bf16_vbench"]["image_quality"] - drop + delta
    summary = eh.aggregate(release, rows)
    assert summary["models"]["hy_worldplay_8b"]["checks"]["mean_vbench_image_quality"] is True
    for case in hy:
        rows[case["case_id"]]["vbench"]["image_quality"] -= 0.01
    summary = eh.aggregate(release, rows)
    assert summary["models"]["hy_worldplay_8b"]["checks"]["mean_vbench_image_quality"] is False
    assert summary["passed"] is False
    # The per-case floors (twice the drop) still hold for every case.
    assert all(summary["cases"][c["case_id"]]["checks"]["vbench_image_quality"] for c in hy)


def test_one_case_below_its_floor_invalidates_even_when_the_mean_passes() -> None:
    release = make_release()
    rows = _rows(release)
    floor = release["models"]["longcat"]["gates"]["case_floor"]["psnr_min"]
    rows["lc-320"]["fidelity"]["psnr"] = floor - 0.01
    for case in release["cases"]:
        if case["model"] == "longcat" and case["case_id"] != "lc-320":
            rows[case["case_id"]]["fidelity"]["psnr"] = 40.0
    summary = eh.aggregate(release, rows)
    assert summary["models"]["longcat"]["checks"]["mean_psnr"] is True
    assert summary["cases"]["lc-320"]["checks"]["psnr"] is False
    assert summary["models"]["longcat"]["passed"] is False and summary["passed"] is False
    result = eh.finalize(release, rows, {}, {})
    assert result["valid"] is False and result["score"] == 0.0 and result["candidate_invalid"] is True
    assert result["reason"] == "quality_gates" and result["raw_score"] > 1.0


def test_mean_fidelity_gate_fails_although_every_case_clears_its_floor() -> None:
    release = make_release()
    rows = _rows(release)
    gates = release["models"]["longcat"]["gates"]
    for case in release["cases"]:
        if case["model"] == "longcat":
            rows[case["case_id"]]["fidelity"]["lpips"] = (gates["mean"]["lpips_max"] + gates["case_floor"]["lpips_max"]) / 2
    summary = eh.aggregate(release, rows)
    assert summary["models"]["longcat"]["checks"]["mean_lpips"] is False
    assert all(summary["cases"][c["case_id"]]["checks"]["lpips"] for c in release["cases"] if c["model"] == "longcat")
    assert summary["passed"] is False


def test_the_rows_must_cover_the_release_exactly() -> None:
    release = make_release()
    rows = _rows(release)
    rows.pop("hy-qvg")
    with pytest.raises(eh.InfrastructureFailure, match="exactly once"):
        eh.aggregate(release, rows)
    rows = _rows(release)
    rows["hy-11"] = rows["hy-10"]
    with pytest.raises(eh.InfrastructureFailure, match="exactly once"):
        eh.aggregate(release, rows)


# --------------------------------------------------------------------------
# Quality-weighted score
# --------------------------------------------------------------------------


def _shipped_mean_inputs(name: str) -> tuple[dict, dict, dict]:
    """Case-mean BF16 VBench, mean allowed drops and mean gates of the shipped suite (public manifest)."""
    manifest = json.loads((ROOT / "portable/final-suite-manifest.json").read_text(encoding="utf-8"))
    cases = [c for c in manifest["cases"] if c["workload"] == name]
    bf16 = {d: statistics.fmean(c["goldens"]["bf16_vbench"][d] for c in cases) for d in DIMS}
    gate = manifest["workload_gates"][name]["mean"]
    drop = {d: float(gate["vbench_allowed_drop"][d]) for d in DIMS}
    return bf16, drop, {k: float(gate[k]) for k in ("psnr_min", "ssim_min", "lpips_max")}


def _terms(name: str, psnr: float, ssim: float, lpips: float, vbench: dict,
           hardware: str = "h100-94gb-400w") -> dict:
    bf16, drop, _gate = _shipped_mean_inputs(name)
    return eh.quality_terms({"psnr": psnr, "ssim": ssim, "lpips": lpips}, vbench, bf16, drop,
                            _shipped_quality_reference(name, hardware))


@pytest.mark.parametrize("hardware", HARDWARE_TYPES)
def test_the_reference_codec_scores_q_zero_and_clears_the_shipped_mean_gates(hardware: str) -> None:
    for name in eh.MODELS:
        reference = _shipped_quality_reference(name, hardware)
        terms = _terms(name, reference["psnr"], reference["ssim"], reference["lpips"], reference["vbench"], hardware)
        assert terms["f"] == 0.0 and terms["v"] == 0.0 and terms["q"] == 0.0
        # The zero point lies inside the admitted region: the reference passes every mean gate.
        _bf16, _drop, gate = _shipped_mean_inputs(name)
        assert reference["psnr"] >= gate["psnr_min"] and reference["ssim"] >= gate["ssim_min"]
        assert reference["lpips"] <= gate["lpips_max"]
        assert 0.0 < terms["reference_worst_vbench_margin"] < 1.0


def test_the_worked_example_of_the_score_definition() -> None:
    """Worked example: a real valid submission at 19.62x compression scores 28.92 (H100 NVL reference)."""
    longcat = _terms("longcat", 27.144947942097982, 0.9318036953608195, 0.048375577107071875, {
        "aesthetic_quality": 61.15581691265106, "background_consistency": 94.94542140586704,
        "image_quality": 71.53803065000727, "subject_consistency": 94.23779915360844})
    hy = _terms("hy_worldplay_8b", 34.37534150210294, 0.9746179797432639, 0.023606637386943807, {
        "aesthetic_quality": 73.63522486253218, "background_consistency": 97.8881679637484,
        "image_quality": 73.58401237865004, "subject_consistency": 96.94245008706436})
    assert longcat["f"] == pytest.approx(0.3939, abs=5e-5) and longcat["v"] == pytest.approx(0.4972, abs=5e-5)
    assert longcat["q"] == pytest.approx(0.4456, abs=5e-5)
    assert longcat["worst_vbench_dimension"] == "image_quality"
    assert longcat["reference_worst_vbench_margin"] == pytest.approx(0.329, abs=5e-4)
    assert longcat["vbench_margin"]["aesthetic_quality"] == 1.0  # above BF16 earns nothing
    assert hy["f"] == pytest.approx(0.4457, abs=5e-5) and hy["v"] == pytest.approx(0.5581, abs=5e-5)
    assert hy["q"] == pytest.approx(0.5019, abs=5e-5)
    assert hy["reference_worst_vbench_margin"] == pytest.approx(0.376, abs=5e-4)
    r_lc, r_hy = 18.76088529385833, 20.527413362865524
    assert math.sqrt(r_lc * r_hy) == pytest.approx(19.62, abs=5e-3)
    assert math.sqrt(r_lc * (1 + longcat["q"]) * r_hy * (1 + hy["q"])) == pytest.approx(28.92, abs=5e-3)


def test_fidelity_gains_are_capped_at_the_ideal_and_negative_below_the_reference() -> None:
    reference = _shipped_quality_reference("longcat")
    perfect = _terms("longcat", 100.0, 1.0, 0.0, reference["vbench"])
    assert perfect["fidelity_gain"] == {"psnr": 1.0, "ssim": 1.0, "lpips": 1.0} and perfect["f"] == 1.0
    at_ceiling = _terms("longcat", eh.PSNR_CEILING_DB, reference["ssim"], reference["lpips"], reference["vbench"])
    assert at_ceiling["fidelity_gain"]["psnr"] == 1.0 and at_ceiling["f"] == pytest.approx(1 / 3)
    worse = _terms("longcat", reference["psnr"] - 0.1, reference["ssim"] - 0.005, reference["lpips"] * 1.05,
                   reference["vbench"])
    assert all(gain < 0 for gain in worse["fidelity_gain"].values()) and worse["q"] < 0


def test_the_tightest_vbench_margin_decides_v_and_readings_above_bf16_earn_nothing() -> None:
    bf16, drop, _gate = _shipped_mean_inputs("hy_worldplay_8b")
    reference = _shipped_quality_reference("hy_worldplay_8b")
    above = {d: bf16[d] + 5.0 for d in DIMS}
    best = _terms("hy_worldplay_8b", reference["psnr"], reference["ssim"], reference["lpips"], above)
    assert all(m == 1.0 for m in best["vbench_margin"].values())
    assert best["v"] == pytest.approx(1.0 - best["reference_worst_vbench_margin"])
    # Spending half of one dimension's allowed drop caps v there, whatever the others do.
    one_down = dict(above, image_quality=bf16["image_quality"] - drop["image_quality"] / 2)
    capped = _terms("hy_worldplay_8b", reference["psnr"], reference["ssim"], reference["lpips"], one_down)
    assert capped["worst_vbench_dimension"] == "image_quality"
    assert capped["worst_vbench_margin"] == pytest.approx(0.5)
    assert capped["v"] == pytest.approx(0.5 - capped["reference_worst_vbench_margin"])


def test_quality_scales_the_compression_and_a_worse_codec_scores_below_its_ratio() -> None:
    release = make_release()
    rows = _rows(release)
    better = eh.aggregate(release, rows)
    assert all(better["models"][name]["quality"]["q"] > 0 for name in eh.MODELS)
    assert better["score"] > better["compression_score"]
    # Still inside every gate, but below the reference on every metric.
    for case in release["cases"]:
        row = rows[case["case_id"]]
        psnr = SYNTHETIC_QUALITY_REFERENCE[case["model"]]["psnr"] - 0.5
        row["fidelity"].update(psnr=psnr, ssim=0.905, lpips=0.065)
        row["vbench"] = {d: value - 0.3 for d, value in row["vbench"].items()}
    worse = eh.aggregate(release, rows)
    assert worse["passed"] is True and worse["compression_score"] == better["compression_score"]
    assert all(worse["models"][name]["quality"]["q"] < 0 for name in eh.MODELS)
    assert worse["score"] < worse["compression_score"]
    result = eh.finalize(release, rows, {}, {})
    assert result["valid"] is True and result["score"] == worse["score"]
    assert result["compression_score"] == worse["compression_score"]


def _load_run_judge():
    spec = importlib.util.spec_from_file_location("qvg_h100_scratch_v4_run_judge_shards", JUDGE / "run_judge.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_calibrating_release_runs_everything_but_issues_no_verdict() -> None:
    release = make_release("calibrating")
    assert all(case["bf16_kv_bytes"] is None for case in release["cases"])
    runner = FakeRunner(release)
    *_rest, controls, rows, state = run(release, 8, runner, state_factory=InstantStagger)
    assert state.first_error is None and len(runner.calls) == len(release["cases"])
    result = eh.finalize(release, rows, controls, {"release_status": release["status"]})
    assert result["outcome"] == "evaluated" and result["reason"] == "uncalibrated_release"
    assert result["valid"] is False and result["score"] == 0.0
    assert result["candidate_invalid"] is False and result["infrastructure_invalid"] is True
    assert result["gates_passed"] is True and result["raw_score"] > 1.0
    for case in release["cases"]:
        record = result["cases"][case["case_id"]]
        assert record["row"]["fidelity"] and record["row"]["vbench"] and record["control"]["status"] == "ok"
        assert record["row"]["bf16_kv_bytes"] > 0
    run_judge = _load_run_judge()
    payload = run_judge.public_payload(run_judge.validate_private_result(json.loads(json.dumps(result))))
    assert payload["valid"] is False and payload["score"] == 0.0
    assert payload["metrics"]["evaluation_status"] == "uncalibrated"
    assert payload["infrastructure_invalid"] is True and payload["candidate_invalid"] is False


def test_the_quality_reference_is_device_class_data() -> None:
    release = make_release()
    rows = _rows(release)
    summary = eh.aggregate(release, rows)
    # Moving the zero point moves q: it is read from the quality data, not the controller.
    release["models"]["longcat"]["quality_reference"]["psnr"] += 0.5
    moved = eh.aggregate(release, rows)
    assert moved["models"]["longcat"]["quality"]["q"] < summary["models"]["longcat"]["quality"]["q"]
    assert moved["models"]["hy_worldplay_8b"]["quality"] == summary["models"]["hy_worldplay_8b"]["quality"]
    assert moved["compression_score"] == summary["compression_score"]


def test_the_quality_file_supplies_the_zero_point_of_the_shared_release() -> None:
    embedded = make_release()
    rows = _rows(embedded)
    shared = make_release()
    quality = _quality_doc(shared)
    assert all("quality_reference" not in model for model in shared["models"].values())
    applied = eh.validate_release(eh.apply_quality(shared, quality))
    assert eh.aggregate(applied, rows) == eh.aggregate(embedded, rows)


def test_load_release_reads_the_quality_file(tmp_path: Path) -> None:
    shared = make_release()
    quality = _quality_doc(shared, "h100-80gb-700w")
    (tmp_path / "release.json").write_text(json.dumps(shared), encoding="utf-8")
    (tmp_path / "quality.json").write_text(json.dumps(quality), encoding="utf-8")
    bare = eh.load_release(tmp_path / "release.json")
    with pytest.raises(eh.InfrastructureFailure, match="without a quality file"):
        eh.require_quality(bare)
    loaded = eh.load_release(tmp_path / "release.json", quality_path=tmp_path / "quality.json")
    eh.require_quality(loaded)
    assert loaded["models"]["longcat"]["quality_reference"] == quality["models"]["longcat"]["quality_reference"]


@pytest.mark.parametrize("edit, message", [
    (lambda r, q: q.pop("hardware"), "must hold"),
    (lambda r, q: q.update(extra=1), "must hold"),
    (lambda r, q: q.update(schema_version=2), "schema 1"),
    (lambda r, q: q.update(hardware=""), "hardware type"),
    (lambda r, q: q["models"].pop("longcat"), "exactly the models"),
    (lambda r, q: q["models"]["longcat"].update(gates={}), "exactly quality_reference"),
    (lambda r, q: r["models"]["longcat"].update(quality_reference=q["models"]["longcat"]["quality_reference"]),
     "release itself carries"),
    (lambda r, q: q["models"]["hy_worldplay_8b"]["quality_reference"].update(ssim=1.0), "inside the fidelity ideals"),
])
def test_a_malformed_quality_file_is_refused(edit, message) -> None:
    release = make_release()
    quality = _quality_doc(release)
    edit(release, quality)
    with pytest.raises(eh.InfrastructureFailure, match=message):
        eh.apply_quality(release, quality)


@pytest.mark.parametrize("edit, message", [
    (lambda ref: ref.pop("lpips"), "exactly psnr, ssim, lpips and vbench"),
    (lambda ref: ref.update(ssim=1.0), "inside the fidelity ideals"),
    (lambda ref: ref.update(psnr=eh.PSNR_CEILING_DB), "inside the fidelity ideals"),
    (lambda ref: ref.update(lpips=0.0), "inside the fidelity ideals"),
    (lambda ref: ref["vbench"].pop("image_quality"), "VBench dimensions"),
    (lambda ref: ref.update(psnr="27"), "finite number"),
])
def test_a_malformed_quality_reference_is_rejected(edit, message) -> None:
    release = make_release()
    edit(release["models"]["hy_worldplay_8b"]["quality_reference"])
    with pytest.raises(eh.InfrastructureFailure, match=message):
        eh.validate_release(release)


def test_only_a_calibrating_release_may_score_without_the_quality_reference() -> None:
    release = make_release()
    del release["models"]["longcat"]["quality_reference"]
    eh.validate_release(release)  # the shared release itself carries no reference
    with pytest.raises(eh.InfrastructureFailure, match="quality_reference for \\['longcat'\\]"):
        eh.require_quality(release)
    calibrating = make_release("calibrating")
    for name in eh.MODELS:
        calibrating["models"][name].pop("quality_reference")
    eh.validate_release(calibrating)
    eh.require_quality(calibrating)
    rows = _rows(calibrating)
    summary = eh.aggregate(calibrating, rows)
    assert all(summary["models"][name]["quality"] is None for name in eh.MODELS)
    assert summary["score"] == summary["compression_score"]
    result = eh.finalize(calibrating, rows, {}, {})
    assert result["valid"] is False and result["reason"] == "uncalibrated_release"


def test_the_mean_vbench_drops_must_be_positive() -> None:
    release = make_release()
    release["models"]["longcat"]["gates"]["mean"]["vbench_allowed_drop"]["image_quality"] = 0.0
    with pytest.raises(eh.InfrastructureFailure, match="must be positive"):
        eh.validate_release(release)


def test_a_calibrated_release_needs_every_frozen_bf16_byte_count() -> None:
    release = make_release()
    release["cases"][3]["bf16_kv_bytes"] = None
    with pytest.raises(eh.InfrastructureFailure, match="bf16_kv_bytes"):
        eh.validate_release(release)


def test_the_shipped_release_passes_the_controller_schema() -> None:
    release = json.loads(release_file().read_text(encoding="utf-8"))
    unpinned = [
        (case["case_id"], key)
        for case in release["cases"]
        for key, value in case.items()
        if key.endswith("_sha256") and value is None
    ]
    # Golden digests stay null until the goldens exist; the controller
    # refuses such a release, so substitute them to check everything else.
    for case in release["cases"]:
        for key in [k for k, v in case.items() if k.endswith("_sha256") and v is None]:
            case[key] = SHA
    release = eh.validate_release(release)
    assert [c["case_id"] for c in release["cases"]] == [f"lc-{i:03d}" for i in LONGCAT_INDICES] + [f"hy-{n}" for n in HY_NAMES]
    assert [c["prompt_index"] for c in release["cases"] if c["model"] == "longcat"] == list(LONGCAT_INDICES)
    assert all(c["segments"] == 10 and c["frame_count"] == 273 for c in release["cases"] if c["model"] == "longcat")
    assert all(c["pose"] == PAPER_POSE and c["frame_count"] == 285 for c in release["cases"] if c["model"] == "hy_worldplay_8b")
    if release["status"] == "calibrated":
        assert not unpinned
    plan = eh.schedule_plan(release, 8)
    assert sorted(x for shard in plan for x in shard) == sorted(c["case_id"] for c in release["cases"])


# --------------------------------------------------------------------------
# The controller's top-level flow, with every OS/GPU phase faked
# --------------------------------------------------------------------------


@pytest.fixture
def faked_phases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    release = make_release(start_stagger_seconds=0)
    path = tmp_path / "release.json"
    path.write_text(json.dumps(release), encoding="utf-8")
    seen: dict = {"kills": FakeControl(), "failures": {}}
    monkeypatch.setattr(eh, "verify_frozen_files", lambda files: seen.__setitem__("frozen", files))
    monkeypatch.setattr(eh, "prompt_lines", lambda release: [])
    monkeypatch.setattr(eh, "isolation_prerequisites", lambda release: eh.Tools("setpriv", "pkill", "taskset"))
    monkeypatch.setattr(eh, "evaluator_hardware",
                        lambda release: ([fake_gpu(i) for i in range(seen.get("gpus", 8))], {"count": seen.get("gpus", 8)}))
    monkeypatch.setattr(js, "cpu_slices", lambda gpus: ([list(range(4 * i, 4 * i + 4)) for i in range(len(gpus))], "even-split"))
    monkeypatch.setattr(eh, "admit_submission", lambda submission: {"valid": True})
    monkeypatch.setattr(eh, "ProcessControl", lambda pkill: seen["kills"])

    class Runner(eh.CaseRunner):
        def preflight(self, slots):
            return [{"slot": s.index, "ok": True} for s in slots]

        def run_case(self, case, slot, control, state):
            if case["case_id"] in seen["failures"]:
                raise seen["failures"][case["case_id"]]
            return fake_row(case)

    monkeypatch.setattr(eh, "CaseRunner", Runner)
    return release, path, tmp_path / "logs" / "qvg-private.json", seen


def test_evaluate_runs_every_case_and_records_the_schema_5_result(faked_phases) -> None:
    release, path, output, seen = faked_phases
    context: dict = {}
    result = eh.evaluate(Path("/workspace/submission"), path, output, expected_gpu_count=8, context=context)
    assert result["result_schema_version"] == 5 and result["outcome"] == "evaluated"
    assert result["valid"] is True and result["score"] == result["raw_score"] > result["compression_score"] > 1.0
    assert set(result["cases"]) == {c["case_id"] for c in release["cases"]}
    for record in result["cases"].values():
        assert record["control"]["status"] == "ok" and record["passed"] is True and record["row"]["fidelity"]
    assert result["schedule"]["gpu_count"] == 8 and len(result["schedule"]["plan"]) == 8
    assert result["schedule"]["controller_budget_seconds"] == eh.controller_budget_seconds(release, 8)
    assert len(result["slots"]) == 8 and len(result["slot_preflight"]) == 8
    assert result["release_status"] == "calibrated" and len(result["release_sha256"]) == 64
    assert {p.name for p in seen["frozen"]} >= {"paper_protocol.json", "compare_frames.py", "run_vbench.py"}
    assert len(seen["frozen"]) == 3 + 1 + 6 * 2 + 11 * 2
    everyone = tuple(range(23001, 23009))
    calls = [uids for _thread, uids in seen["kills"].calls]
    assert calls[0] == everyone and calls[-1] == everyone
    json.dumps(result, allow_nan=False)


def test_evaluate_fails_with_the_first_errors_kind_and_keeps_the_records(faked_phases) -> None:
    release, path, output, seen = faked_phases
    seen["failures"]["hy-04"] = eh.InfrastructureFailure("a trusted metric returned an error (vbench.log)")
    context: dict = {}
    with pytest.raises(eh.InfrastructureFailure, match="vbench"):
        eh.evaluate(Path("/workspace/submission"), path, output, context=context)
    assert context["first_error"]["case_id"] == "hy-04"
    assert context["cases"]["hy-04"]["control"]["status"] == "infrastructure_failure"
    detail = eh.failure_result("infrastructure_failure", "x", context)
    assert detail["infrastructure_invalid"] is True and detail["score"] == 0.0
    json.dumps(detail, allow_nan=False)


def test_evaluate_refuses_a_gpu_count_that_changed_after_the_adapter_counted(faked_phases) -> None:
    _release, path, output, seen = faked_phases
    seen["gpus"] = 7
    with pytest.raises(eh.InfrastructureFailure, match="GPU count changed"):
        eh.evaluate(Path("/workspace/submission"), path, output, expected_gpu_count=8, context={})


def test_evaluate_scores_identically_on_one_gpu(faked_phases) -> None:
    _release, path, output, seen = faked_phases
    eight = eh.evaluate(Path("/workspace/submission"), path, output, context={})
    seen["gpus"] = 1
    one = eh.evaluate(Path("/workspace/submission"), path, output, context={})
    assert one["score"] == eight["score"] and one["schedule"]["gpu_count"] == 1
    assert {k: v["row"] for k, v in one["cases"].items()} == {k: v["row"] for k, v in eight["cases"].items()}


def test_progress_snapshots_keep_per_case_records_for_a_killed_controller(tmp_path: Path) -> None:
    release = make_release()
    failing = "lc-012"
    hang = {c["case_id"] for c in release["cases"]} - {failing}
    snapshots: list[dict] = []
    runner = FakeRunner(release, fail={failing: eh.CandidateFailure("exit 1")}, hang=hang)
    slots = make_slots(release, 8)
    eh.run_suite(release, slots, eh.schedule_plan(release, 8), runner, control=FakeControl(),
                 registry=eh.ChildRegistry(), state_factory=InstantStagger, progress=snapshots.append)
    assert snapshots and all(json.dumps(snapshot, allow_nan=False) for snapshot in snapshots)
    assert any(record["status"] == "running" for snapshot in snapshots for record in snapshot["cases"].values())
    last = snapshots[-1]
    assert last["first_error"]["case_id"] == failing
    assert {record["status"] for record in last["cases"].values()} <= {"candidate_failure", "cancelled"}
    assert last["cases"][failing]["master_port"] == 29600 and last["cases"][failing]["gpu_uuid"].startswith("GPU-")
    path = tmp_path / eh.PROGRESS_NAME
    eh._write_json(path, last)
    run_judge = _load_run_judge()
    assert run_judge._progress(tmp_path)["first_error"]["case_id"] == failing
    assert run_judge._progress(tmp_path / "absent") is None
