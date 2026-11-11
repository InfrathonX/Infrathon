from __future__ import annotations

import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from conftest import release_file


ROOT = Path(__file__).resolve().parents[1]
JUDGE = ROOT / "judge"
sys.path.insert(0, str(JUDGE))

import evaluate_hidden  # noqa: E402
import judge_shards  # noqa: E402


TASK_ID = "quant_videogen_cross_model_kv_codec_h100_scratch_v4"
DIMS = evaluate_hidden.VBENCH_DIMENSIONS
POSE = "w-8,s-8,a-8,d-8,up-8,down-8,w-8,s-8,a-7"


def load_run_judge():
    spec = importlib.util.spec_from_file_location(
        "qvg_h100_scratch_v4_run_judge", JUDGE / "run_judge.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def slot(index: int = 3, *, cpus=(12, 13, 14, 15)) -> evaluate_hidden.Slot:
    gpu = judge_shards.Gpu(index, f"GPU-{index:08d}", "NVIDIA H100 NVL", f"00000000:{0x40 + index:02x}:00.0",
                           100_000_000_000, 99_000_000_000, (9, 0))
    return evaluate_hidden.Slot(index, gpu, 23001 + index, tuple(cpus))


MODELS = {
    "longcat": {"fps": 15, "stage_seconds": 3000, "fidelity_metric_seconds": 600, "vbench_metric_seconds": 1200,
                "profile_key": "persistent_kv", "frames_file": "final.frames.pt"},
    "hy_worldplay_8b": {"fps": 24, "stage_seconds": 900, "fidelity_metric_seconds": 600, "vbench_metric_seconds": 900,
                        "profile_key": "persistent_vision_kv", "frames_file": "gen.frames.pt"},
}
RELEASE = {"models": MODELS}
LONGCAT = {
    "case_id": "lc-185", "model": "longcat", "seed": 0, "prompt_index": 185, "prompt": "a hidden prompt",
    "segments": 10, "init_video": "/eval/hidden/longcat/185/init.mp4",
    "bf16_frames": "/eval/hidden/longcat/185/bf16-10seg.frames.pt", "frame_count": 273,
    "fidelity_frames": [73, 93], "bf16_vbench": {d: 90.0 for d in DIMS}, "bf16_kv_bytes": 4096,
}
HY = {
    "case_id": "hy-05", "model": "hy_worldplay_8b", "seed": 0, "prompt": "a hidden caption",
    "image": "/eval/hidden/hy/05/image.jpeg", "video_length": 285, "pose": POSE,
    "bf16_frames": "/eval/hidden/hy/05/bf16-285.frames.pt", "frame_count": 285,
    "fidelity_frames": [29, 45], "bf16_vbench": {d: 90.0 for d in DIMS}, "bf16_kv_bytes": None,
}


# --------------------------------------------------------------------------
# Candidate identity, environment and command lines
# --------------------------------------------------------------------------


@pytest.mark.parametrize("uid", [23001, 23005, 23008])
def test_candidate_command_drops_to_the_slot_uid_and_all_caps(uid: int) -> None:
    command = evaluate_hidden.candidate_command("/usr/bin/setpriv", uid, ["python3", "runner.py"])
    assert command == [
        "/usr/bin/setpriv",
        f"--reuid={uid}",
        f"--regid={uid}",
        "--clear-groups",
        "--no-new-privs",
        "--inh-caps=-all",
        "--ambient-caps=-all",
        "--bounding-set=-all",
        "--pdeathsig=KILL",
        "--",
        "python3",
        "runner.py",
    ]


def test_candidate_environment_drops_credentials_and_binds_the_slot_device(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("HTTPS_PROXY", "http://secret")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("OMP_NUM_THREADS", "64")
    environment = evaluate_hidden.candidate_environment(tmp_path, slot(5, cpus=(20, 21, 22, 23)))
    assert "OPENAI_API_KEY" not in environment
    assert "HTTPS_PROXY" not in environment
    # Every GPU stays visible; the slot binds its own by PCI-ordered index.
    assert "CUDA_VISIBLE_DEVICES" not in environment
    assert environment["QVG_DEVICE_INDEX"] == "5"
    assert environment["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert environment["OMP_NUM_THREADS"] == environment["MKL_NUM_THREADS"] == "4"
    assert environment["HOME"] == str(tmp_path / "home")
    assert environment["TMPDIR"] == str(tmp_path / "tmp")
    assert environment["HF_HUB_OFFLINE"] == "1"
    # HAMi's utilization watcher stays off for every slot process.
    assert environment["CUDA_DEVICE_SM_LIMIT"] == "0"


def test_candidate_environment_carries_the_deployment_reserve_switch(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QVG_CUDA_RESERVE", "1")
    monkeypatch.setenv("QVG_CUDA_RESERVE_HEADROOM_BYTES", "4294967296")
    environment = evaluate_hidden.candidate_environment(tmp_path, slot(1, cpus=(4, 5, 6, 7)))
    assert environment["QVG_CUDA_RESERVE"] == "1"
    assert environment["QVG_CUDA_RESERVE_HEADROOM_BYTES"] == "4294967296"


def test_root_metrics_see_every_gpu_and_run_on_the_slot_device(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-xyz")
    monkeypatch.setenv("CUDA_DEVICE_MEMORY_SHARED_CACHE", "/tmp/hami.cache")
    environment = evaluate_hidden.metric_environment(slot(2, cpus=(8, 9, 10, 11)))
    assert "CUDA_VISIBLE_DEVICES" not in environment
    assert environment["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID" and environment["QVG_DEVICE_INDEX"] == "2"
    assert environment["CUDA_DEVICE_MEMORY_SHARED_CACHE"] == "/tmp/hami.cache"
    fidelity, vbench = evaluate_hidden.metric_commands(
        RELEASE, LONGCAT, candidate_frames=Path("/tmp/c/final.frames.pt"), output=Path("/tmp/m"), device_index=2
    )
    assert fidelity == [
        sys.executable, "/opt/mlsysforge/metrics/compare_frames.py",
        "--reference", "/eval/hidden/longcat/185/bf16-10seg.frames.pt",
        "--candidate", "/tmp/c/final.frames.pt",
        "--frame-range", "73", "93",
        "--output", "/tmp/m/fidelity.json",
        "--device", "cuda:2",
    ]
    assert vbench == [
        sys.executable, "/opt/mlsysforge/metrics/run_vbench.py",
        "--frames", "/tmp/c/final.frames.pt",
        "--prompt", "a hidden prompt",
        "--fps", "15",
        "--output-dir", "/tmp/m/vbench",
        "--name", "lc-185",
        "--device", "cuda:2",
    ]


def test_longcat_runner_command_matches_the_runner_interface(tmp_path: Path) -> None:
    command = evaluate_hidden.runner_command(RELEASE, LONGCAT, stage=tmp_path, master_port=29601, device_index=6)
    assert command == [
        sys.executable, "/opt/qvg-runners/run_longcat.py",
        "--mode", "candidate",
        "--prompt-file", str(tmp_path / "input/prompt.txt"),
        "--prompt-index", "1",
        "--seed", "0",
        "--output", str(tmp_path / "output"),
        "--init-video", str(tmp_path / "input/init.mp4"),
        "--segments", "10",
        "--master-port", "29601",
        "--device-index", "6",
    ]


def test_hy_runner_command_passes_pose_port_and_device(tmp_path: Path) -> None:
    command = evaluate_hidden.runner_command(RELEASE, HY, stage=tmp_path, master_port=29610, device_index=0)
    assert command == [
        sys.executable, "/opt/qvg-runners/run_hy_worldplay_8b.py",
        "--mode", "candidate",
        "--seed", "0",
        "--video-length", "285",
        "--prompt", "a hidden caption",
        "--image-path", str(tmp_path / "input/image.jpeg"),
        "--pose", POSE,
        "--output", str(tmp_path / "output"),
        "--master-port", "29610",
        "--device-index", "0",
    ]


def _load_runner(name: str):
    spec = importlib.util.spec_from_file_location(f"qvg_v4_{name}", JUDGE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_release_case_command_round_trips_through_the_runner_parsers(tmp_path: Path) -> None:
    """End to end on CPU: the frozen release -> controller argv -> runner argv/env."""
    release = json.loads(release_file().read_text(encoding="utf-8"))
    runners = {"longcat": _load_runner("run_longcat"), "hy_worldplay_8b": _load_runner("run_hy_worldplay_8b")}
    for position, case in enumerate(release["cases"]):
        port = release["judge"]["master_port_base"] + position
        command = evaluate_hidden.runner_command(release, case, stage=tmp_path, master_port=port, device_index=5)
        module = runners[case["model"]]
        args = module.parse_args(command[2:])
        assert (args.master_port, args.device_index, args.seed, args.mode) == (port, 5, case["seed"], "candidate")
        env = module.build_env(args, {"CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
        assert env["QVG_DEVICE_INDEX"] == "5" and env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        assert env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == "4"
        argv = module.build_command(args)
        assert f"--master_port={port}" in argv
        if case["model"] == "longcat":
            assert args.segments == case["segments"] and args.prompt_index == 1
            assert argv[argv.index("--num_segments") + 1] == str(case["segments"])
            assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
        else:
            assert args.pose == case["pose"] and args.video_length == case["video_length"]
            assert argv[argv.index("--pose") + 1] == case["pose"]
            assert argv[argv.index("--prompt") + 1] == case["prompt"]
            assert "PYTORCH_CUDA_ALLOC_CONF" not in env


def test_runner_cli_accepts_the_controller_arguments() -> None:
    """The runners own their CLI; this pins that they parse what we pass."""
    for runner, flags in (
        ("run_longcat.py", ("--master-port", "--device-index", "--segments")),
        ("run_hy_worldplay_8b.py", ("--master-port", "--device-index", "--pose")),
    ):
        text = (JUDGE / runner).read_text(encoding="utf-8")
        for flag in flags:
            assert f'"{flag}"' in text, (runner, flag)
        assert "--master_port=29501" not in text and "--master_port=29502" not in text


# --------------------------------------------------------------------------
# Per-case stage and staged inputs
# --------------------------------------------------------------------------


class _Chowns:
    def __init__(self) -> None:
        self.calls: dict[str, tuple[int, int]] = {}

    def __call__(self, path, uid, gid, *args, **kwargs) -> None:
        self.calls[str(path)] = (uid, gid)


def _runner(tmp_path: Path, *, prompts=(), cls=evaluate_hidden.CaseRunner):
    return cls(
        RELEASE,
        tools=evaluate_hidden.Tools("setpriv", "pkill", "taskset"),
        control=types.SimpleNamespace(kill_uids=lambda uids: None),
        registry=evaluate_hidden.ChildRegistry(),
        log_root=tmp_path / "logs",
        scratch=tmp_path / "scratch",
        prompts=list(prompts),
    )


def test_each_case_stage_is_traversable_only_by_its_slot(tmp_path: Path, monkeypatch) -> None:
    chowns = _Chowns()
    monkeypatch.setattr(evaluate_hidden.os, "chown", chowns)
    runner = _runner(tmp_path)
    first, second = runner.create_stage(slot(1)), runner.create_stage(slot(1))
    try:
        assert first != second and first.name.startswith("qvg-slot1-")
        assert chowns.calls[str(first)] == (0, 23002) and stat.S_IMODE(first.stat().st_mode) == 0o710
        assert chowns.calls[str(first / "input")] == (0, 23002)
        assert stat.S_IMODE((first / "input").stat().st_mode) == 0o550
        for name in ("output", "home", "tmp", "cache"):
            assert chowns.calls[str(first / name)] == (23002, 23002)
            assert stat.S_IMODE((first / name).stat().st_mode) == 0o700
    finally:
        for path in (first, second):
            for child in path.iterdir():
                child.chmod(0o700)
            evaluate_hidden.shutil.rmtree(path)


def test_inputs_are_staged_per_case_and_checked_against_the_release(tmp_path: Path, monkeypatch) -> None:
    chowns = _Chowns()
    monkeypatch.setattr(evaluate_hidden.os, "chown", chowns)
    sources = tmp_path / "hidden"
    sources.mkdir()
    (sources / "init.mp4").write_bytes(b"mp4")
    (sources / "image.jpeg").write_bytes(b"jpeg")
    longcat = {**LONGCAT, "init_video": str(sources / "init.mp4"), "prompt_index": 2}
    hy = {**HY, "image": str(sources / "image.jpeg")}
    runner = _runner(tmp_path, prompts=["line one", "a hidden prompt"])
    stage = tmp_path / "stage"
    (stage / "input").mkdir(parents=True)
    runner.stage_inputs(longcat, stage, slot(0))
    assert (stage / "input/prompt.txt").read_text() == "a hidden prompt\n"
    assert (stage / "input/init.mp4").read_bytes() == b"mp4"
    assert chowns.calls[str(stage / "input/init.mp4")] == (0, 23001)
    assert stat.S_IMODE((stage / "input/init.mp4").stat().st_mode) == 0o440
    runner.stage_inputs(hy, stage, slot(0))
    assert (stage / "input/image.jpeg").read_bytes() == b"jpeg"
    with pytest.raises(evaluate_hidden.InfrastructureFailure, match="prompt"):
        runner.stage_inputs({**longcat, "prompt_index": 1}, stage, slot(0))
    (sources / "link.png").symlink_to(sources / "image.jpeg")
    with pytest.raises(evaluate_hidden.InfrastructureFailure, match="unsafe"):
        runner.stage_inputs({**hy, "image": str(sources / "link.png")}, stage, slot(0))


def test_isolation_requires_one_account_per_slot(monkeypatch, tmp_path: Path) -> None:
    release = {"judge": {"slot_uids": [23001, 23002, 23003]}}
    accounts = {23001 + i: types.SimpleNamespace(pw_uid=23001 + i, pw_gid=23001 + i) for i in range(3)}
    monkeypatch.setattr(evaluate_hidden.os, "geteuid", lambda: 0)
    monkeypatch.setattr(evaluate_hidden.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(evaluate_hidden, "PRIVATE_PATHS", ())
    # Accounts are checked by UID/GID; their names are provenance only.
    monkeypatch.setattr(evaluate_hidden.pwd, "getpwuid", lambda uid: accounts[uid])
    tools = evaluate_hidden.isolation_prerequisites(release)
    assert tools == evaluate_hidden.Tools("/usr/bin/setpriv", "/usr/bin/pkill", "/usr/bin/taskset")
    accounts[23002] = types.SimpleNamespace(pw_uid=23002, pw_gid=100)
    with pytest.raises(evaluate_hidden.InfrastructureFailure, match="unexpected IDs"):
        evaluate_hidden.isolation_prerequisites(release)
    del accounts[23002]
    with pytest.raises(evaluate_hidden.InfrastructureFailure, match="missing"):
        evaluate_hidden.isolation_prerequisites(release)
    accounts[23002] = types.SimpleNamespace(pw_uid=23002, pw_gid=23002)
    monkeypatch.setattr(evaluate_hidden.shutil, "which", lambda name: None if name == "taskset" else name)
    with pytest.raises(evaluate_hidden.InfrastructureFailure, match="tools"):
        evaluate_hidden.isolation_prerequisites(release)
    monkeypatch.setattr(evaluate_hidden.shutil, "which", lambda name: name)
    monkeypatch.setattr(evaluate_hidden, "PRIVATE_PATHS", (tmp_path,))
    with pytest.raises(evaluate_hidden.InfrastructureFailure, match="candidate-readable"):
        evaluate_hidden.isolation_prerequisites(release)


def test_judge_image_creates_one_literal_account_per_release_slot_uid() -> None:
    """Literal UID == GID accounts, named as the PyroMind qvg-v4 Dockerfile.judge names them."""
    release = json.loads(release_file().read_text(encoding="utf-8"))
    dockerfile = (ROOT / "images/judge/Dockerfile").read_text(encoding="utf-8")
    created = {
        int(uid): (gid, name)
        for uid, gid, name in re.findall(
            r"useradd --uid (\d+) --gid (\d+) --no-create-home --home-dir /nonexistent "
            r"--shell /usr/sbin/nologin ([a-z0-9-]+)",
            dockerfile,
        )
    }
    uids = release["judge"]["slot_uids"]
    assert sorted(created) == sorted(uids)
    for slot, uid in enumerate(uids):
        assert created[uid] == (str(uid), evaluate_hidden.slot_account_name(slot))
        assert f"groupadd --gid {uid} {evaluate_hidden.slot_account_name(slot)}" in dockerfile
    assert "$" not in "".join(line for line in dockerfile.splitlines() if "useradd" in line)


def test_slot_preflight_rejects_every_isolation_or_binding_defect() -> None:
    good = {
        "uid": 23004, "gid": 23004, "groups": [], "cuda_visible_devices": None, "current_device": 3,
        "uuid": "00000003", "sum": float(64 * 1024 * 1024), "other_reserved_bytes": 0,
        "private_listable": [], "readable_missing": [],
    }
    assert evaluate_hidden.slot_preflight_problem(good, slot(3)) is None
    for change, message in (
        ({"uid": 23001}, "identity"),
        ({"groups": [0]}, "groups"),
        ({"cuda_visible_devices": "3"}, "CUDA_VISIBLE_DEVICES"),
        ({"current_device": 0}, "bind"),
        ({"uuid": "00000000"}, "different GPU"),
        ({"other_reserved_bytes": 2 << 20}, "another GPU"),
        ({"private_listable": ["/eval"]}, "controller-private"),
        ({"readable_missing": ["/models/hidden/longcat"]}, "unreadable"),
    ):
        assert message in evaluate_hidden.slot_preflight_problem({**good, **change}, slot(3))
    assert evaluate_hidden.slot_preflight_problem(None, slot(3)) == "the probe did not complete"


# --------------------------------------------------------------------------
# Candidate outputs
# --------------------------------------------------------------------------


def _profile(tmp_path: Path, stats: dict, **top) -> Path:
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"persistent_kv": {"calls": 10, **stats}, "wall_seconds": 12.5,
                                "other_device_allocated_bytes": 0, "cuda_device_index": 3, **top}),
                    encoding="utf-8")
    return path


def test_profile_row_reads_the_accumulated_counters(tmp_path: Path) -> None:
    row = evaluate_hidden.profile_row(
        _profile(tmp_path, {"stored_kv_bytes": 512, "bf16_kv_bytes": 4096, "calls": 10,
                            "kv_compression_ratio": 8.0, "last_call": {"stored_kv_bytes": 51}}),
        "persistent_kv", LONGCAT, device_index=3,
    )
    assert row == {"compression_ratio": 8.0, "stored_kv_bytes": 512, "bf16_kv_bytes": 4096,
                   "other_device_allocated_bytes": 0, "wall_seconds": 12.5, "compression_calls": 10}


@pytest.mark.parametrize(
    "stats, top, message",
    [
        ({"stored_kv_bytes": 512, "bf16_kv_bytes": 4000}, {}, "frozen release"),
        ({"stored_kv_bytes": 8192, "bf16_kv_bytes": 4096}, {}, "not compressed"),
        ({"stored_kv_bytes": 512, "bf16_kv_bytes": 4096, "kv_compression_ratio": 9.0}, {}, "inconsistent"),
        ({"stored_kv_bytes": 512, "bf16_kv_bytes": 4096}, {"other_device_allocated_bytes": 2 << 20}, "outside its slot"),
        ({"stored_kv_bytes": 512, "bf16_kv_bytes": 4096}, {"other_device_allocated_bytes": None}, "other-device"),
        ({"stored_kv_bytes": 0, "bf16_kv_bytes": 4096}, {}, "positive integer"),
        ({"stored_kv_bytes": True, "bf16_kv_bytes": 4096}, {}, "positive integer"),
        ({"stored_kv_bytes": 512, "bf16_kv_bytes": 4096}, {"wall_seconds": float("nan")}, "JSON contract"),
        ({"stored_kv_bytes": 512, "bf16_kv_bytes": 4096}, {"cuda_device_index": 0}, "other than its slot"),
        ({"stored_kv_bytes": 512, "bf16_kv_bytes": 4096}, {"cuda_device_index": None}, "other than its slot"),
        ({"stored_kv_bytes": 512, "bf16_kv_bytes": 4096, "calls": 1}, {}, "every compression call"),
        ({"stored_kv_bytes": 512, "bf16_kv_bytes": 4096, "calls": None}, {}, "every compression call"),
    ],
)
def test_a_profile_contract_violation_is_a_candidate_failure(tmp_path: Path, stats, top, message) -> None:
    with pytest.raises(evaluate_hidden.CandidateFailure, match=message):
        evaluate_hidden.profile_row(_profile(tmp_path, stats, **top), "persistent_kv", LONGCAT, device_index=3)


def test_expected_compression_calls_follow_the_trusted_pipelines() -> None:
    # One quantize_kv_cache per LongCat segment; one HY context pass per
    # 4-latent chunk after the first (285 frames = 72 latents = 18 chunks).
    assert evaluate_hidden.expected_compression_calls(LONGCAT) == 10
    assert evaluate_hidden.expected_compression_calls({**LONGCAT, "segments": 1}) == 1
    assert evaluate_hidden.expected_compression_calls(HY) == 17
    assert evaluate_hidden.expected_compression_calls({**HY, "video_length": 125}) == 7


def test_an_unfrozen_bf16_byte_count_is_recorded_during_calibration(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"persistent_vision_kv": {"stored_kv_bytes": 7, "bf16_kv_bytes": 70, "calls": 17},
                                "wall_seconds": 1.0, "other_device_allocated_bytes": 0,
                                "cuda_device_index": 5}), encoding="utf-8")
    assert evaluate_hidden.profile_row(path, "persistent_vision_kv", HY, device_index=5)["bf16_kv_bytes"] == 70
    with pytest.raises(evaluate_hidden.CandidateFailure, match="incomplete"):
        evaluate_hidden.profile_row(path, "persistent_kv", HY, device_index=5)


def test_frames_must_have_the_release_shape(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    case = {**LONGCAT, "frame_count": 2, "bf16_frames": str(tmp_path / "golden.pt")}
    torch.save(torch.zeros((2, 480, 832, 3), dtype=torch.uint8), tmp_path / "golden.pt")
    torch.save(torch.zeros((2, 480, 832, 3), dtype=torch.uint8), tmp_path / "good.pt")
    evaluate_hidden.check_frames(case, tmp_path / "good.pt")
    torch.save(torch.zeros((3, 480, 832, 3), dtype=torch.uint8), tmp_path / "long.pt")
    with pytest.raises(evaluate_hidden.CandidateFailure, match="shape"):
        evaluate_hidden.check_frames(case, tmp_path / "long.pt")
    torch.save(torch.zeros((2, 480, 832, 3), dtype=torch.float32), tmp_path / "float.pt")
    with pytest.raises(evaluate_hidden.CandidateFailure, match="ABI"):
        evaluate_hidden.check_frames(case, tmp_path / "float.pt")
    with pytest.raises(evaluate_hidden.CandidateFailure, match="missing"):
        evaluate_hidden.check_frames(case, tmp_path / "absent.pt")
    torch.save(torch.zeros((2, 480, 640, 3), dtype=torch.uint8), tmp_path / "golden.pt")
    with pytest.raises(evaluate_hidden.InfrastructureFailure, match="reference"):
        evaluate_hidden.check_frames(case, tmp_path / "good.pt")


def test_metric_readings_must_be_complete_and_finite() -> None:
    reading = evaluate_hidden._fidelity_reading({"psnr": 30, "ssim": 0.9, "lpips": 0.1, "evaluation_start_frame": 72})
    assert reading == {"psnr": 30.0, "ssim": 0.9, "lpips": 0.1, "evaluation_start_frame": 72}
    with pytest.raises(evaluate_hidden.InfrastructureFailure):
        evaluate_hidden._fidelity_reading({"psnr": 30, "ssim": 0.9})
    assert evaluate_hidden._vbench_reading({"dimensions": {d: 50 for d in DIMS}}) == {d: 50.0 for d in DIMS}
    with pytest.raises(evaluate_hidden.InfrastructureFailure):
        evaluate_hidden._vbench_reading({"dimensions": {d: 50 for d in DIMS[:-1]}})


# --------------------------------------------------------------------------
# Real child processes (CPU only): attribution, timeouts, fail-fast
# --------------------------------------------------------------------------


def _alive(pid: int) -> bool:
    """True while ``pid`` exists and is not a zombie."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (FileNotFoundError, ProcessLookupError):
        return False


def _passthrough_tools(tmp_path: Path) -> evaluate_hidden.Tools:
    """taskset/setpriv stand-ins that drop their own options and exec the rest."""
    taskset = tmp_path / "taskset"
    taskset.write_text('#!/usr/bin/env bash\nshift 2\nexec "$@"\n')
    setpriv = tmp_path / "setpriv"
    setpriv.write_text('#!/usr/bin/env bash\nwhile [ "$1" != "--" ]; do shift; done\nshift\nexec "$@"\n')
    for path in (taskset, setpriv):
        path.chmod(0o755)
    return evaluate_hidden.Tools(str(setpriv), "pkill", str(taskset))


def _process_runner(tmp_path: Path, cgroup: Path):
    class Runner(evaluate_hidden.CaseRunner):
        candidate_cwd = tmp_path
        cgroup_root = cgroup

    runner = _runner(tmp_path, cls=Runner)
    runner.tools = _passthrough_tools(tmp_path)
    return runner


def test_candidate_exit_is_a_candidate_failure_unless_the_cgroup_oom_killer_fired(tmp_path: Path) -> None:
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    (cgroup / "memory.events").write_text("low 0\nhigh 0\nmax 4\noom 1\noom_kill 1\n")
    runner = _process_runner(tmp_path, cgroup)
    assert runner.oom_baseline == 1
    stage = tmp_path / "stage"
    log = tmp_path / "candidate.log"
    failing = [sys.executable, "-c", "import os, sys; print(os.environ['QVG_DEVICE_INDEX']); sys.exit(3)"]
    with pytest.raises(evaluate_hidden.CandidateFailure, match="exit 3"):
        runner.run_candidate(failing, stage=stage, slot=slot(3), log=log, timeout=60)
    assert log.read_text().strip() == "3" and stat.S_IMODE(log.stat().st_mode) == 0o600
    (cgroup / "memory.events").write_text("oom 2\noom_kill 2\n")
    with pytest.raises(evaluate_hidden.InfrastructureFailure, match="OOM"):
        runner.run_candidate(failing, stage=stage, slot=slot(3), log=log, timeout=60)


def test_a_stage_timeout_kills_the_candidate_and_is_a_candidate_failure(tmp_path: Path) -> None:
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    runner = _process_runner(tmp_path, cgroup)
    assert runner.oom_baseline is None
    pid_file = tmp_path / "pid"
    sleeper = [sys.executable, "-c",
               f"import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(60)"]
    started = time.monotonic()
    with pytest.raises(evaluate_hidden.CandidateFailure, match="time limit"):
        runner.run_candidate(sleeper, stage=tmp_path, slot=slot(0), log=tmp_path / "c.log", timeout=1)
    assert time.monotonic() - started < 20
    assert not _alive(int(pid_file.read_text()))


def test_a_cancel_before_launch_never_starts_the_candidate(tmp_path: Path) -> None:
    runner = _process_runner(tmp_path, tmp_path / "no-cgroup")
    state = evaluate_hidden.RunState(lambda: None)
    state.cancel.set()
    marker = tmp_path / "started"
    command = [sys.executable, "-c", f"open({str(marker)!r}, 'w').write('1')"]
    with pytest.raises(evaluate_hidden.InfrastructureFailure, match="cancelled"):
        runner.run_candidate(command, stage=tmp_path, slot=slot(0), log=tmp_path / "c.log", timeout=60, state=state)
    assert not marker.exists()


def test_a_cancel_during_the_stage_kills_the_candidate_at_once(tmp_path: Path) -> None:
    runner = _process_runner(tmp_path, tmp_path / "no-cgroup")
    state = evaluate_hidden.RunState(lambda: None)
    pid_file = tmp_path / "pid"
    sleeper = [sys.executable, "-c",
               f"import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(60)"]
    threading.Timer(0.5, state.cancel.set).start()
    started = time.monotonic()
    with pytest.raises(evaluate_hidden.InfrastructureFailure, match="cancelled"):
        runner.run_candidate(sleeper, stage=tmp_path, slot=slot(0), log=tmp_path / "c.log", timeout=600, state=state)
    assert time.monotonic() - started < 10
    assert not _alive(int(pid_file.read_text()))


def test_a_session_is_never_registered_after_the_cancel() -> None:
    registry = evaluate_hidden.ChildRegistry()
    cancel = threading.Event()
    assert registry.add_session_unless(101, cancel) is True
    cancel.set()
    assert registry.add_session_unless(102, cancel) is False
    assert registry._sessions == {101}


def test_a_slot_staging_across_another_slots_failure_returns_the_verdict_promptly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Slot 1 passes its per-case cancel check, then slot 0 fails while slot 1
    is still staging; slot 1's candidate must not run out its stage budget."""
    release = {
        "models": MODELS,
        "judge": {"start_stagger_seconds": 0, "master_port_base": 29600},
        "cases": [dict(LONGCAT, case_id="lc-a"), dict(LONGCAT, case_id="lc-b")],
    }
    pid_file = tmp_path / "late.pid"
    late = [sys.executable, "-c",
            f"import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(60)"]

    class Runner(evaluate_hidden.CaseRunner):
        candidate_cwd = tmp_path
        cgroup_root = tmp_path / "no-cgroup"

        def case_log_dir(self, case):
            path = tmp_path / "logs" / case["case_id"]
            path.mkdir(parents=True, exist_ok=True)
            return path

        def create_stage(self, slot):
            path = tmp_path / f"stage-{slot.index}"
            path.mkdir(exist_ok=True)
            return path

        def stage_inputs(self, case, stage, slot):
            if slot.index == 0:
                raise evaluate_hidden.CandidateFailure("candidate execution returned exit 1")
            assert state.cancel.wait(10)

    monkeypatch.setattr(evaluate_hidden, "runner_command", lambda *args, **kwargs: late)
    runner = _runner(tmp_path, cls=Runner)
    runner.release = release
    runner.tools = _passthrough_tools(tmp_path)
    slots = [slot(0, cpus=(0, 1)), slot(1, cpus=(2, 3))]
    state = evaluate_hidden.RunState(lambda: None)
    started = time.monotonic()
    controls, rows = evaluate_hidden.execute_plan(release, slots, [["lc-a"], ["lc-b"]], runner, state)
    assert time.monotonic() - started < 10
    assert state.first_error["kind"] == "candidate_failure" and state.first_error["case_id"] == "lc-a"
    assert controls["lc-b"]["status"] == "cancelled" and not rows
    if pid_file.exists():
        assert not _alive(int(pid_file.read_text()))


def _start_stamp(path: Path, *, sleep: float, code: int = 0) -> list[str]:
    return [sys.executable, "-c",
            f"import os, sys, time; open({str(path)!r}, 'w').write(f'{{os.getpid()}} {{time.time()}}'); "
            f"time.sleep({sleep}); sys.exit({code})"]


def test_root_metrics_run_concurrently(tmp_path: Path) -> None:
    evaluate_hidden.supervise_root(
        [(_start_stamp(tmp_path / "a", sleep=1.0), tmp_path / "a.log", 30),
         (_start_stamp(tmp_path / "b", sleep=1.0), tmp_path / "b.log", 30)],
        env=os.environ, registry=evaluate_hidden.ChildRegistry(),
    )
    # Both started before either could have finished.
    a, b = (float((tmp_path / name).read_text().split()[1]) for name in ("a", "b"))
    assert abs(a - b) < 0.5


def test_a_failed_metric_stops_its_sibling_at_once(tmp_path: Path) -> None:
    started = time.monotonic()
    with pytest.raises(evaluate_hidden.InfrastructureFailure, match="returned an error"):
        evaluate_hidden.supervise_root(
            [(_start_stamp(tmp_path / "a", sleep=0.1, code=1), tmp_path / "a.log", 30),
             (_start_stamp(tmp_path / "b", sleep=60), tmp_path / "b.log", 30)],
            env=os.environ, registry=evaluate_hidden.ChildRegistry(),
        )
    assert time.monotonic() - started < 10
    assert not _alive(int((tmp_path / "b").read_text().split()[0]))


def test_a_metric_timeout_or_a_cancel_is_infrastructure(tmp_path: Path) -> None:
    with pytest.raises(evaluate_hidden.InfrastructureFailure, match="time limit"):
        evaluate_hidden.supervise_root(
            [(_start_stamp(tmp_path / "a", sleep=60), tmp_path / "a.log", 1)],
            env=os.environ, registry=evaluate_hidden.ChildRegistry(),
        )
    state = evaluate_hidden.RunState(lambda: None)
    threading.Timer(0.3, state.cancel.set).start()
    with pytest.raises(evaluate_hidden.InfrastructureFailure, match="cancelled"):
        evaluate_hidden.supervise_root(
            [(_start_stamp(tmp_path / "b", sleep=60), tmp_path / "b.log", 30)],
            env=os.environ, registry=evaluate_hidden.ChildRegistry(), state=state,
        )


def test_the_adapter_kills_the_controller_session_and_every_slot_uid_on_timeout(
    tmp_path: Path, monkeypatch
) -> None:
    run_judge = load_run_judge()
    killed: list[tuple[int, ...]] = []
    monkeypatch.setattr(run_judge, "_pkill_uids", lambda uids: killed.append(tuple(uids)))
    pid_file = tmp_path / "metric.pid"
    controller = [
        sys.executable, "-c",
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], process_group=0)\n"
        f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
        "time.sleep(60)\n",
    ]
    uids = tuple(range(23001, 23009))
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        run_judge.run_controller(controller, log=tmp_path / "controller.log", timeout=2, slot_uids=uids)
    assert time.monotonic() - started < 30
    assert killed == [uids]
    pid = int(pid_file.read_text())
    deadline = time.monotonic() + 5
    while _alive(pid):
        assert time.monotonic() < deadline, "a trusted child of the controller survived the adapter timeout"
        time.sleep(0.05)


def test_the_adapter_returns_the_controller_exit_code(tmp_path: Path) -> None:
    run_judge = load_run_judge()
    code = run_judge.run_controller([sys.executable, "-c", "raise SystemExit(2)"],
                                    log=tmp_path / "c.log", timeout=30, slot_uids=(23001,))
    assert code == 2


# --------------------------------------------------------------------------
# Private result and public projection
# --------------------------------------------------------------------------


def _evaluated(**overrides) -> dict:
    value = {
        "result_schema_version": 5,
        "task_id": TASK_ID,
        "outcome": "evaluated",
        "valid": True,
        "candidate_invalid": False,
        "infrastructure_invalid": False,
        "score": 12.5,
        "reason": None,
        "models": {"longcat": {"compression_ratio": 9.0}, "hy_worldplay_8b": {"compression_ratio": 7.0}},
        "cases": {"lc-012": {"row": {"psnr": 1}}, "hy-qvg": {"control": {"gpu_uuid": "GPU-1"}}},
        "first_error": None,
        "release_sha256": "secret",
    }
    value.update(overrides)
    return value


def test_public_payload_contains_only_aggregate_result() -> None:
    run_judge = load_run_judge()
    payload = run_judge.public_payload(run_judge.validate_private_result(_evaluated()))
    encoded = json.dumps(payload, sort_keys=True)
    assert payload["valid"] is True
    assert payload["score"] == 12.5
    assert payload["result_schema_version"] == 5 and payload["metrics"]["evaluation_status"] == "completed"
    for forbidden in ("models", "gates", "release", "psnr", "longcat", "worldplay", "hy-", "lc-", "case",
                      "gpu", "slot", "uuid"):
        assert forbidden not in encoded.lower()


def test_public_candidate_error_is_generic() -> None:
    run_judge = load_run_judge()
    private = _evaluated(outcome="candidate_failure", valid=False, candidate_invalid=True, score=0.0,
                         error="secret workload lc-185 and exact gate details",
                         first_error={"case_id": "lc-185", "kind": "candidate_failure"})
    payload = run_judge.public_payload(run_judge.validate_private_result(private))
    assert payload["error"] == "the protected evaluation did not accept this submission"
    assert "secret" not in json.dumps(payload) and "lc-185" not in json.dumps(payload)


def test_valid_absolute_scores_are_preserved_but_invalid_scores_stay_zero() -> None:
    run_judge = load_run_judge()
    assert run_judge.validate_private_result(_evaluated(score=5.0))["score"] == 5.0
    rejected = _evaluated(valid=False, candidate_invalid=True, score=5.0)
    with pytest.raises(ValueError, match="invalid result has a non-zero score"):
        run_judge.validate_private_result(rejected)
    with pytest.raises(ValueError, match="schema"):
        run_judge.validate_private_result(_evaluated(result_schema_version=3))
    # Schema 4 scored plain compression; its score must not pass as the quality-weighted one.
    with pytest.raises(ValueError, match="schema"):
        run_judge.validate_private_result(_evaluated(result_schema_version=4))
    # The quality factor can take a valid score below the compression ratio, even below 1x.
    assert run_judge.validate_private_result(_evaluated(score=0.9))["score"] == 0.9
    with pytest.raises(ValueError, match="not positive"):
        run_judge.validate_private_result(_evaluated(score=0.0))


@pytest.mark.parametrize(
    "overrides",
    [
        {"valid": True, "infrastructure_invalid": False},
        {"valid": False, "candidate_invalid": True, "infrastructure_invalid": False, "score": 0.0},
        {"valid": False, "candidate_invalid": False, "infrastructure_invalid": False, "score": 0.0},
    ],
)
def test_an_uncalibrated_result_can_never_be_a_verdict(overrides: dict) -> None:
    run_judge = load_run_judge()
    with pytest.raises(ValueError):
        run_judge.validate_private_result(_evaluated(reason="uncalibrated_release", **overrides))
    accepted = run_judge.validate_private_result(_evaluated(
        reason="uncalibrated_release", valid=False, candidate_invalid=False, infrastructure_invalid=True, score=0.0,
    ))
    payload = run_judge.public_payload(accepted)
    assert payload["metrics"]["evaluation_status"] == "uncalibrated" and payload["valid"] is False


def test_controller_failures_keep_their_per_case_records_private(tmp_path: Path, monkeypatch) -> None:
    def failing_evaluate(submission, release, output, *, quality_path, expected_gpu_count, context):
        context["cases"] = {"lc-012": {"control": {"status": "candidate_failure", "slot": 0}}}
        context["first_error"] = {"kind": "candidate_failure", "case_id": "lc-012"}
        raise evaluate_hidden.CandidateFailure("candidate execution returned exit 1")

    monkeypatch.setattr(evaluate_hidden, "evaluate", failing_evaluate)
    output = tmp_path / "private.json"
    monkeypatch.setattr(sys, "argv", ["evaluate_hidden.py", "--submission", "s", "--release", "r",
                                      "--output", str(output), "--gpu-count", "8"])
    with pytest.raises(SystemExit) as exit_info:
        evaluate_hidden.main()
    assert exit_info.value.code == 2
    private = json.loads(output.read_text())
    assert private["outcome"] == "candidate_failure" and private["result_schema_version"] == 5
    assert private["cases"]["lc-012"]["control"]["status"] == "candidate_failure"
    assert private["first_error"]["case_id"] == "lc-012"
    run_judge = load_run_judge()
    payload = run_judge.public_payload(run_judge.validate_private_result(private))
    assert "lc-012" not in json.dumps(payload)
