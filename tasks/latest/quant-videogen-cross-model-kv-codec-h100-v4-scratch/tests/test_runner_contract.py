"""The trusted runners and the upstream accounting they rely on (CPU only).

The sharded Judge runs several cases at once with every GPU visible, so each
runner takes its GPU (``--device-index``) and torchrun port (``--master-port``)
from the controller and hands the GPU to the upstream entrypoint through
``QVG_DEVICE_INDEX``.  The score sums persistent-cache bytes over every
LongCat segment and every HY context pass.  Nothing here launches torchrun:
the runners are driven through ``parse_args``/``build_command``/``build_env``
and the upstream helpers are executed from their source with fake torch
surfaces and a fake ``payload_nbytes`` that accepts CPU tensors.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
JUDGE = ROOT / "judge"
LONGCAT = ROOT / "upstream/experiments/LongCat"
LONGCAT_MAIN = LONGCAT / "run_long_t2v.py"
LONGCAT_DIT = LONGCAT / "longcat_video/modules/longcat_video_dit.py"
HY = ROOT / "upstream/experiments/HY-WorldPlay/hyvideo"
HY_MAIN = HY / "generate.py"
HY_ATTENTION = HY / "models/transformers/modules/attention.py"
HY_TRANSFORMER = HY / "models/transformers/worldplay_1_5_transformer.py"
PAPER_POSE = "w-8,s-8,a-8,d-8,up-8,down-8,w-8,s-8,a-7"


TASK_ROOT = Path(__file__).resolve().parents[1]


def load_runner(name: str):
    spec = importlib.util.spec_from_file_location(f"qvg_v4_{name}", JUDGE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def longcat():
    return load_runner("run_longcat")


@pytest.fixture(scope="module")
def hy():
    return load_runner("run_hy_worldplay_8b")


def longcat_argv(*extra: str) -> list[str]:
    return [
        "--mode", "candidate", "--prompt-file", "/stage/input/prompt.txt", "--prompt-index", "185",
        "--seed", "0", "--output", "/stage/output", "--init-video", "/stage/input/init.mp4",
        "--master-port", "29605", "--device-index", "5", *extra,
    ]


def hy_argv(*extra: str) -> list[str]:
    return [
        "--mode", "candidate", "--output", "/stage/output", "--image-path", "/stage/input/image.jpeg",
        "--prompt", "A quiet street.", "--master-port", "29611", "--device-index", "3", *extra,
    ]


def flag(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


# --------------------------------------------------------------------------
# Runner command line
# --------------------------------------------------------------------------


@pytest.mark.parametrize("runner, argv", [("run_longcat", longcat_argv), ("run_hy_worldplay_8b", hy_argv)])
@pytest.mark.parametrize("dropped", ["--master-port", "--device-index"])
def test_runners_require_port_and_device(runner, argv, dropped: str) -> None:
    full = argv()
    position = full.index(dropped)
    with pytest.raises(SystemExit):
        load_runner(runner).parse_args(full[:position] + full[position + 2:])


@pytest.mark.parametrize("runner, argv", [("run_longcat", longcat_argv), ("run_hy_worldplay_8b", hy_argv)])
@pytest.mark.parametrize("name, value", [
    ("--master-port", "1023"), ("--master-port", "65536"), ("--master-port", "0"), ("--master-port", "port"),
    ("--device-index", "-1"), ("--device-index", "cuda:1"),
])
def test_runners_range_check_port_and_device(runner, argv, name: str, value: str) -> None:
    full = argv()
    full[full.index(name) + 1] = value
    with pytest.raises(SystemExit):
        load_runner(runner).parse_args(full)


@pytest.mark.parametrize("runner, argv", [("run_longcat", longcat_argv), ("run_hy_worldplay_8b", hy_argv)])
@pytest.mark.parametrize("port", ["1024", "65535"])
def test_port_bounds_are_inclusive(runner, argv, port: str) -> None:
    full = argv()
    full[full.index("--master-port") + 1] = port
    module = load_runner(runner)
    assert f"--master_port={port}" in module.build_command(module.parse_args(full))


def test_longcat_command_uses_the_case_port_and_segments(longcat) -> None:
    args = longcat.parse_args(longcat_argv("--segments", "10"))
    command = longcat.build_command(args)
    assert command[:4] == ["torchrun", "--nproc_per_node=1", "--master_addr=127.0.0.1", "--master_port=29605"]
    assert command[4] == "/opt/qvg-src/experiments/LongCat/run_long_t2v.py"
    assert not any(part.startswith("--master_port=2950") for part in command)
    assert flag(command, "--workload") == "480p_long_gen"
    assert flag(command, "--num_segments") == "10"
    assert flag(command, "--prompt_idx") == "185"
    assert flag(command, "--init_video_path") == "/stage/input/init.mp4"
    assert flag(command, "--quant_type") == "triton-nstages-kmeans-int2"
    assert "--no_offload_kv_cache" in command and "--profile" in command
    # The device travels only through the environment.
    assert not any("device" in part for part in command)


def test_longcat_segments_default_to_upstream_and_are_positive(longcat) -> None:
    assert flag(longcat.build_command(longcat.parse_args(longcat_argv())), "--num_segments") == "10"
    with pytest.raises(SystemExit):
        longcat.parse_args(longcat_argv("--segments", "0"))


def test_longcat_bf16_and_make_init_commands(longcat) -> None:
    bf16 = longcat.parse_args(longcat_argv("--segments", "10"))
    bf16.mode = "bf16"
    assert flag(longcat.build_command(bf16), "--quant_type") == "none"
    init = longcat.parse_args(longcat_argv("--make-init"))
    command = longcat.build_command(init)
    assert flag(command, "--workload") == "480p_init"
    assert flag(command, "--num_frames") == "73"
    assert "--num_segments" not in command and "--init_video_path" not in command


def test_longcat_continuation_requires_an_init_video(longcat, tmp_path: Path, monkeypatch) -> None:
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("x\n", encoding="utf-8")
    monkeypatch.setattr(longcat, "CHECKPOINT_DIR", tmp_path)
    argv = longcat_argv()
    argv[argv.index("--prompt-file") + 1] = str(prompt)
    args = longcat.parse_args(argv)
    with pytest.raises(SystemExit, match="--init-video"):
        longcat.check_inputs(args)
    args.make_init = True
    longcat.check_inputs(args)


def test_hy_command_uses_the_case_port_and_paper_pose_by_default(hy) -> None:
    command = hy.build_command(hy.parse_args(hy_argv()))
    assert command[:4] == ["torchrun", "--nproc_per_node=1", "--master_addr=127.0.0.1", "--master_port=29611"]
    assert command[4] == "/opt/qvg-src/experiments/HY-WorldPlay/hyvideo/generate.py"
    assert flag(command, "--pose") == PAPER_POSE
    assert flag(command, "--video_length") == "285"
    assert flag(command, "--image_path") == "/stage/input/image.jpeg"
    assert flag(command, "--qvg_quantize_kv") == "true"
    assert not any("device" in part for part in command)


def test_hy_pose_defaults_follow_video_length(hy) -> None:
    assert hy.parse_args(hy_argv("--video-length", "33")).pose == "w-8"
    assert hy.parse_args(hy_argv("--video-length", "45")).pose == "w-8,s-3"
    with pytest.raises(SystemExit):
        hy.parse_args(hy_argv("--video-length", "289"))


def test_hy_explicit_pose_is_passed_through(hy) -> None:
    pose = "d-8,d-8,a-8,a-8,left-8,right-8,up-8,down-8,w-7"
    assert flag(hy.build_command(hy.parse_args(hy_argv("--pose", pose))), "--pose") == pose


@pytest.mark.parametrize("pose", [
    "/stage/input/pose.json",                       # upstream would open a JSON file
    "w-8,s-8,a-8,d-8,up-8,down-8,w-8,s-8,a-6",      # 70 of 71 latent steps
    "w-8,s-8,a-8,d-8,up-8,down-8,w-8,s-8,a-8",      # 72 of 71
    "w-8,s-8,a-8,d-8,up-8,down-8,w-8,s-8,a-6.5",    # fractional span
    "w-8, s-8,a-8,d-8,up-8,down-8,w-8,s-8,a-7",     # whitespace
    "jump-71",
    "w-0,w-71",
    "",
])
def test_hy_rejects_malformed_poses(hy, pose: str) -> None:
    with pytest.raises(SystemExit):
        hy.parse_args(hy_argv("--pose", pose))


# --------------------------------------------------------------------------
# Runner environment
# --------------------------------------------------------------------------


BASE_ENV = {
    "PATH": "/usr/bin",
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "HOME": "/stage/home",
}


@pytest.mark.parametrize("mode", ["candidate", "qvg-reference", "bf16"])
def test_longcat_env_binds_the_device_and_expandable_segments_in_every_mode(longcat, mode: str) -> None:
    args = longcat.parse_args(longcat_argv())
    args.mode = mode
    env = longcat.build_env(args, {**BASE_ENV, "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:64"})
    assert env["QVG_DEVICE_INDEX"] == "5"
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == "4"
    assert env["HOME"] == "/stage/home"
    assert "CUDA_VISIBLE_DEVICES" not in env


@pytest.mark.parametrize("mode", ["candidate", "qvg-reference", "bf16"])
def test_hy_env_binds_the_device_and_leaves_the_allocator_alone(hy, mode: str) -> None:
    args = hy.parse_args(hy_argv())
    args.mode = mode
    env = hy.build_env(args, BASE_ENV)
    assert env["QVG_DEVICE_INDEX"] == "3"
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert "PYTORCH_CUDA_ALLOC_CONF" not in env
    assert env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == "4"
    assert "CUDA_VISIBLE_DEVICES" not in env
    inherited = hy.build_env(args, {**BASE_ENV, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    assert inherited["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


@pytest.mark.parametrize("runner, argv", [("run_longcat", longcat_argv), ("run_hy_worldplay_8b", hy_argv)])
def test_reserve_mode_switches_to_the_plain_allocator(runner, argv) -> None:
    """PyroMind (QVG_CUDA_RESERVE=1): one up-front reservation needs the plain allocator."""
    module = load_runner(runner)
    args = module.parse_args(argv())
    env = module.build_env(args, {**BASE_ENV, "QVG_CUDA_RESERVE": "1",
                                  "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:False"
    assert env["QVG_CUDA_RESERVE"] == "1"


@pytest.mark.parametrize("script", ["upstream/experiments/LongCat/run_long_t2v.py",
                                    "upstream/experiments/HY-WorldPlay/hyvideo/generate.py"])
def test_entry_scripts_reserve_right_after_binding_the_device(script) -> None:
    source = (TASK_ROOT / script).read_text(encoding="utf-8")
    tree = ast.parse(source)
    helper = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "qvg_reserve_device_memory"]
    assert len(helper) == 1
    text = ast.get_source_segment(source, helper[0])
    for needle in ('os.environ.get("QVG_CUDA_RESERVE") != "1"', "torch.cuda.mem_get_info", "torch.empty(size",
                   "torch.cuda.empty_cache = lambda: None", "reset_peak_memory_stats"):
        assert needle in text, needle
    lines = source.splitlines()
    bind = next(i for i, line in enumerate(lines) if "torch.cuda.set_device(" in line and "QVG" not in line or
                line.strip() in ("torch.cuda.set_device(device_index)", "torch.cuda.set_device(DEVICE_INDEX)"))
    assert "qvg_reserve_device_memory(" in lines[bind + 1]


@pytest.mark.parametrize("runner, argv", [("run_longcat", longcat_argv), ("run_hy_worldplay_8b", hy_argv)])
def test_runners_never_set_cuda_visible_devices(runner, argv) -> None:
    module = load_runner(runner)
    args = module.parse_args(argv())
    assert "CUDA_VISIBLE_DEVICES" not in module.build_env(args, BASE_ENV)
    # Whatever the controller passes is neither rewritten nor removed.
    assert module.build_env(args, {**BASE_ENV, "CUDA_VISIBLE_DEVICES": "0,1"})["CUDA_VISIBLE_DEVICES"] == "0,1"


@pytest.mark.parametrize("runner, argv", [("run_longcat", longcat_argv), ("run_hy_worldplay_8b", hy_argv)])
def test_runner_pythonpath_by_mode(runner, argv) -> None:
    module = load_runner(runner)
    args = module.parse_args(argv())
    assert module.build_env(args, BASE_ENV)["PYTHONPATH"].split(":")[:2] == ["/workspace/submission", "/opt/qvg-src"]
    args.mode = "qvg-reference"
    golden = module.build_env(args, {**BASE_ENV, "QVG_GOLDEN_CODEC_DIR": "/opt/qvg-golden-codec-test"})
    assert golden["PYTHONPATH"].split(":")[:2] == ["/opt/qvg-golden-codec-test", "/opt/qvg-src"]


def test_longcat_main_runs_torchrun_once_with_the_built_command(longcat, tmp_path: Path, monkeypatch) -> None:
    prompt = tmp_path / "prompt.txt"
    init = tmp_path / "init.mp4"
    prompt.write_text("x\n", encoding="utf-8")
    init.write_bytes(b"\0")
    monkeypatch.setattr(longcat, "CHECKPOINT_DIR", tmp_path)
    for key, value in BASE_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    calls = []
    monkeypatch.setattr(longcat.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    argv = longcat_argv("--segments", "10")
    for name, value in (("--prompt-file", prompt), ("--init-video", init), ("--output", tmp_path / "out")):
        argv[argv.index(name) + 1] = str(value)
    longcat.main(argv)
    [(command, kwargs)] = calls
    assert command == longcat.build_command(longcat.parse_args(argv))
    assert kwargs["check"] is True and kwargs["cwd"] == longcat.QVG_ROOT
    assert kwargs["env"]["QVG_DEVICE_INDEX"] == "5" and kwargs["env"]["OMP_NUM_THREADS"] == "4"
    assert (tmp_path / "out").is_dir()


def test_hy_main_runs_torchrun_once_with_the_built_command(hy, tmp_path: Path, monkeypatch) -> None:
    image = tmp_path / "image.jpeg"
    image.write_bytes(b"\0")
    (tmp_path / "base/transformer/480p_i2v").mkdir(parents=True)
    (tmp_path / "action.safetensors").write_bytes(b"\0")
    monkeypatch.setattr(hy, "BASE_MODEL", tmp_path / "base")
    monkeypatch.setattr(hy, "ACTION_CKPT", tmp_path / "action.safetensors")
    for key, value in BASE_ENV.items():
        monkeypatch.setenv(key, value)
    calls = []
    monkeypatch.setattr(hy.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    argv = hy_argv()
    for name, value in (("--image-path", image), ("--output", tmp_path / "out")):
        argv[argv.index(name) + 1] = str(value)
    hy.main(argv)
    [(command, kwargs)] = calls
    assert command == hy.build_command(hy.parse_args(argv))
    assert flag(command, "--model_path") == str(tmp_path / "base")
    assert kwargs["env"]["QVG_DEVICE_INDEX"] == "3" and kwargs["check"] is True


# --------------------------------------------------------------------------
# Upstream device binding and the other-device guard
# --------------------------------------------------------------------------


def _source_nodes(path: Path, names: set[str], *, class_name: str | None = None) -> list[ast.stmt]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    scope = tree.body
    if class_name is not None:
        [cls] = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name]
        scope = cls.body
    picked: list[ast.stmt] = []
    found: set[str] = set()
    for node in scope:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            picked.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign):
            targets = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if targets & names:
                picked.append(node)
                found |= targets & names
    assert found == names, f"{path.name} is missing {sorted(names - found)}"
    return picked


def load_source(path: Path, names: set[str], namespace: dict, *, class_name: str | None = None) -> dict:
    module = ast.Module(body=_source_nodes(path, names, class_name=class_name), type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def _function_source(path: Path, name: str) -> str:
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f"{path.name} does not define {name}()")


def test_longcat_binds_qvg_device_index_with_the_upstream_fallback() -> None:
    body = _function_source(LONGCAT_MAIN, "init_distributed_environment")
    assert "local_rank = rank % num_gpus" in body
    assert "device_index = int(os.environ.get('QVG_DEVICE_INDEX', local_rank))" in body
    assert "torch.cuda.set_device(device_index)" in body
    assert body.index("set_device(device_index)") < body.index("dist.init_process_group(")
    generate = _function_source(LONGCAT_MAIN, "generate")
    assert "pipe.to(device_index)" in _function_source(LONGCAT_MAIN, "get_model_and_pipe")
    assert "torch.cuda.max_memory_allocated(device_index)" in generate
    # Rank-0 writers stay gated on the rank, not on the GPU index.
    assert "local_rank == 0 and args.profile" in generate


def test_longcat_profile_reports_the_cumulative_cache_and_other_devices() -> None:
    generate = _function_source(LONGCAT_MAIN, "generate")
    assert "'persistent_kv': getattr(pipe.dit, 'kv_stats', None)" in generate
    assert "'other_device_allocated_bytes': other_device_reserved_bytes(device_index)" in generate
    assert "'cuda_device_index': device_index" in generate
    assert "last_kv_stats" not in generate


def test_hy_binds_qvg_device_index_before_the_device_mesh() -> None:
    statements = [ast.unparse(node) for node in ast.parse(HY_MAIN.read_text(encoding="utf-8")).body]
    binding = "DEVICE_INDEX = int(os.environ.get('QVG_DEVICE_INDEX', os.environ.get('LOCAL_RANK', '0')))"
    assert binding in statements
    set_device = statements.index("torch.cuda.set_device(DEVICE_INDEX)")
    mesh = next(i for i, text in enumerate(statements) if "initialize_parallel_state(" in text)
    assert statements.index(binding) < set_device < mesh
    assert sum("set_device" in text for text in statements) == 1
    body = _function_source(HY_MAIN, "generate_video")
    assert "'other_device_allocated_bytes': other_device_reserved_bytes(DEVICE_INDEX)" in body
    assert "'persistent_vision_kv': qvg_cache_stats()" in body


class FakeCuda:
    def __init__(self, count: int, reserved: dict[int, int], peak: dict[int, int]) -> None:
        self.count, self.reserved, self.peak = count, reserved, peak

    def device_count(self) -> int:
        return self.count

    def memory_reserved(self, index: int) -> int:
        return self.reserved.get(index, 0)

    def max_memory_reserved(self, index: int) -> int:
        return self.peak.get(index, 0)


@pytest.mark.parametrize("path", [LONGCAT_MAIN, HY_MAIN], ids=["longcat", "hy"])
def test_other_device_bytes_cover_every_other_gpu_now_and_at_peak(path: Path) -> None:
    def measure(reserved: dict[int, int], peak: dict[int, int], own: int = 2) -> int:
        cuda = FakeCuda(8, reserved, peak)
        namespace = load_source(path, {"other_device_reserved_bytes"}, {"torch": SimpleNamespace(cuda=cuda)})
        return namespace["other_device_reserved_bytes"](own)

    assert measure({2: 80 << 30}, {2: 90 << 30}) == 0
    assert measure({2: 1, 5: 2 << 20}, {5: 2 << 20}) == 2 << 20
    # Freed and released again, but the allocator's peak remembers it.
    assert measure({}, {0: 4 << 20}) == 4 << 20
    # A reset peak cannot hide memory that is still reserved.
    assert measure({7: 3 << 20}, {}) == 3 << 20
    assert measure({0: 1, 1: 2, 3: 4}, {}, own=2) == 7


# --------------------------------------------------------------------------
# LongCat: bytes summed over every quantize_kv_cache call (one per segment)
# --------------------------------------------------------------------------


def fake_payload_nbytes(value) -> int:
    """payload_nbytes without the CUDA requirement: unique storages, once each."""
    seen: set[tuple[int, int]] = set()

    def visit(item) -> int:
        if isinstance(item, torch.Tensor):
            storage = item.untyped_storage()
            key = (storage.data_ptr(), storage.nbytes())
            if key in seen:
                return 0
            seen.add(key)
            return storage.nbytes()
        if isinstance(item, dict):
            return sum(visit(child) for child in item.values())
        if isinstance(item, (list, tuple)):
            return sum(visit(child) for child in item)
        return 0

    return visit(value)


class _Event:
    def __init__(self, enable_timing: bool = False) -> None:
        pass

    def record(self) -> None:
        pass

    def elapsed_time(self, other: "_Event") -> float:
        return 0.0


def longcat_dit(codec):
    """A stand-in DiT carrying the real quantize_kv_cache/_record_kv_stats."""
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(Event=_Event, synchronize=lambda: None),
        no_grad=contextlib.nullcontext,
        random=SimpleNamespace(fork_rng=lambda devices: contextlib.nullcontext()),
    )
    namespace = {
        "torch": fake_torch,
        "cprint": lambda *args, **kwargs: None,
        "payload_nbytes": fake_payload_nbytes,
        "get_quantize_fn": lambda quant_type, config: None,
        "compress_kv_cache": lambda k, v, quant_type, config, quantize_fn: codec(k, v),
        "_onload_kv_cache": None,
        "_offload_kv_cache": None,
    }
    load_source(LONGCAT_DIT, {"_kv_ratio", "kv_call_stats", "accumulate_kv_stats"}, namespace)
    load_source(LONGCAT_DIT, {"quantize_kv_cache", "_record_kv_stats"}, namespace,
                class_name="LongCatVideoTransformer3DModel")
    dit_type = type("DiT", (), {
        "quantize_kv_cache": namespace["quantize_kv_cache"],
        "_record_kv_stats": namespace["_record_kv_stats"],
    })
    return dit_type, namespace


def segment_cache(layers: int = 3, tokens: int = 8) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    return {
        index: (torch.randn(1, 2, tokens, 4).to(torch.bfloat16), torch.randn(1, 2, tokens, 4).to(torch.bfloat16))
        for index in range(layers)
    }


def int8_codec(k, v):
    return {"data": k.to(torch.int8)}, {"data": v.to(torch.int8)}


def run_segments(codec_for_segment, quant_type: str = "int8", segments: int = 10):
    caches = []
    dit_type, _ = longcat_dit(lambda k, v: codec_for_segment(len(caches) - 1)(k, v))
    dit = dit_type()
    dit.quant_config = SimpleNamespace(quant_type=quant_type)
    for _ in range(segments):
        caches.append(segment_cache())
        dit.quantize_kv_cache(caches[-1])
    return dit


def test_longcat_sums_every_segment() -> None:
    dit = run_segments(lambda segment: int8_codec)
    per_call_bf16 = 3 * 2 * (1 * 2 * 8 * 4) * 2
    assert dit.last_kv_stats == {
        "bf16_kv_bytes": per_call_bf16,
        "stored_kv_bytes": per_call_bf16 // 2,
        "kv_compression_ratio": 2.0,
        "layers": 3,
    }
    assert dit.kv_stats == {
        "bf16_kv_bytes": 10 * per_call_bf16,
        "stored_kv_bytes": 10 * per_call_bf16 // 2,
        "kv_compression_ratio": 2.0,
        "calls": 10,
        "last_call": dit.last_kv_stats,
    }


def test_longcat_compressing_only_the_last_segment_does_not_pay() -> None:
    identity = lambda k, v: (k, v)  # noqa: E731
    dit = run_segments(lambda segment: int8_codec if segment == 9 else identity)
    assert dit.kv_stats["last_call"]["kv_compression_ratio"] == 2.0
    assert dit.kv_stats["kv_compression_ratio"] == pytest.approx(10 / 9.5)
    assert dit.kv_stats["calls"] == 10


def test_longcat_aliases_count_once_per_call_but_in_every_call() -> None:
    def shared(k, v):
        packed = k.to(torch.int8)
        return packed, packed  # K and V share one storage

    dit = run_segments(lambda segment: shared, segments=4)
    per_call_bf16 = 3 * 2 * (1 * 2 * 8 * 4) * 2
    assert dit.last_kv_stats["stored_kv_bytes"] == per_call_bf16 // 4
    assert dit.kv_stats["stored_kv_bytes"] == 4 * per_call_bf16 // 4
    assert dit.kv_stats["kv_compression_ratio"] == 4.0


def no_codec(k, v):
    raise AssertionError("the codec must not run in BF16 mode")


def test_longcat_bf16_mode_is_well_defined() -> None:
    dit = run_segments(lambda segment: no_codec, quant_type="none")
    per_call = 3 * 2 * (1 * 2 * 8 * 4) * 2
    assert dit.kv_stats == {
        "bf16_kv_bytes": 10 * per_call,
        "stored_kv_bytes": 10 * per_call,
        "kv_compression_ratio": 1.0,
        "calls": 10,
        "last_call": {"bf16_kv_bytes": per_call, "stored_kv_bytes": per_call,
                      "kv_compression_ratio": 1.0, "layers": 3},
    }


def test_longcat_bf16_mode_returns_the_cache_untouched() -> None:
    dit_type, _ = longcat_dit(no_codec)
    dit = dit_type()
    dit.quant_config = SimpleNamespace(quant_type="none")
    cache = segment_cache()
    tensors = [tensor for pair in cache.values() for tensor in pair]
    assert dit.quantize_kv_cache(cache) is cache
    assert all(a is b for a, b in zip(tensors, (t for pair in cache.values() for t in pair)))


def test_longcat_zero_storage_payload_fails_the_run() -> None:
    with pytest.raises(ValueError, match="zero bytes"):
        run_segments(lambda segment: (lambda k, v: ({"shape": 1}, {"shape": 1})), segments=1)


# --------------------------------------------------------------------------
# HY: bytes summed over every cache_vision context pass
# --------------------------------------------------------------------------


HY_ACCOUNTING = {
    "_QVG_LAYER_STATS", "_QVG_OPEN_LAYERS", "_QVG_TOTALS", "_qvg_ratio", "_qvg_record_layer",
    "_qvg_pass_stats", "qvg_close_context_pass", "qvg_cache_stats", "_qvg_encode_context",
}


@pytest.fixture
def hy_accounting(monkeypatch):
    codec = types.ModuleType("mlsysforge_codec")
    codec.encode = int8_codec
    monkeypatch.setitem(sys.modules, "mlsysforge_codec", codec)
    namespace = load_source(HY_ATTENTION, HY_ACCOUNTING, {"torch": torch, "payload_nbytes": fake_payload_nbytes})
    namespace["codec"] = codec
    return namespace


def context_pass(namespace, *, layers: int, tokens: int, recent: int) -> None:
    for block_idx in range(layers):
        raw_k = torch.randn(1, tokens, 2, 4).to(torch.bfloat16)  # [B, L, H, D]
        raw_v = torch.randn(1, tokens, 2, 4).to(torch.bfloat16)
        namespace["_qvg_encode_context"]((raw_k, raw_v, None, None, None, recent), block_idx)
    namespace["qvg_close_context_pass"]()


def bf16_pass(namespace, *, layers: int, tokens: int) -> None:
    # Mirrors the BF16 branch of sequence_parallel_attention_vision.
    for block_idx in range(layers):
        key = torch.randn(2, 2, tokens, 4).to(torch.bfloat16)
        value = torch.randn(2, 2, tokens, 4).to(torch.bfloat16)
        namespace["_qvg_record_layer"](block_idx, fake_payload_nbytes((key, value)), (key, value), 0)
    namespace["qvg_close_context_pass"]()


def test_hy_sums_every_context_pass_and_keeps_the_window_out(hy_accounting) -> None:
    token_bytes = 2 * 4 * 2  # heads * dim * bf16
    contexts = [4, 12, 20, 20]  # the newest 4 tokens stay BF16 in every pass
    for tokens in contexts:
        context_pass(hy_accounting, layers=3, tokens=tokens, recent=4)
    stats = hy_accounting["qvg_cache_stats"]()
    old = sum(tokens - 4 for tokens in contexts)
    assert stats["bf16_kv_bytes"] == 3 * 2 * old * token_bytes
    assert stats["stored_kv_bytes"] == stats["bf16_kv_bytes"] // 2
    assert stats["kv_compression_ratio"] == 2.0
    assert stats["bf16_window_bytes"] == 3 * 2 * 4 * token_bytes * len(contexts)
    assert stats["calls"] == len(contexts)
    assert stats["last_call"] == {
        "bf16_kv_bytes": 3 * 2 * 16 * token_bytes,
        "stored_kv_bytes": 3 * 16 * token_bytes,
        "kv_compression_ratio": 2.0,
        "bf16_window_bytes": 3 * 2 * 4 * token_bytes,
        "layers": 3,
    }


def test_hy_first_pass_with_only_the_window_counts_zero_bytes(hy_accounting) -> None:
    context_pass(hy_accounting, layers=2, tokens=4, recent=4)
    stats = hy_accounting["qvg_cache_stats"]()
    assert stats["bf16_kv_bytes"] == stats["stored_kv_bytes"] == 0
    assert stats["kv_compression_ratio"] == 1.0
    assert stats["calls"] == 1 and stats["last_call"]["layers"] == 2


def test_hy_compressing_only_the_last_pass_does_not_pay(hy_accounting) -> None:
    identity = lambda k, v: (k, v)  # noqa: E731
    for index in range(4):
        hy_accounting["codec"].encode = int8_codec if index == 3 else identity
        context_pass(hy_accounting, layers=2, tokens=12, recent=4)
    stats = hy_accounting["qvg_cache_stats"]()
    assert stats["last_call"]["kv_compression_ratio"] == 2.0
    assert stats["kv_compression_ratio"] == pytest.approx(4 / 3.5)


def test_hy_aliases_count_once_per_pass_but_in_every_pass(hy_accounting) -> None:
    shared = torch.zeros(64, dtype=torch.int8)
    hy_accounting["codec"].encode = lambda k, v: ({"data": shared}, {"data": shared[:32]})
    for _ in range(3):
        context_pass(hy_accounting, layers=4, tokens=12, recent=4)
    stats = hy_accounting["qvg_cache_stats"]()
    assert stats["last_call"]["stored_kv_bytes"] == 64
    assert stats["stored_kv_bytes"] == 3 * 64


def test_hy_bf16_mode_is_well_defined(hy_accounting) -> None:
    for tokens in (4, 8, 8):
        bf16_pass(hy_accounting, layers=3, tokens=tokens)
    stats = hy_accounting["qvg_cache_stats"]()
    expected = 3 * 2 * (2 * 2 * 4 * 2) * (4 + 8 + 8)
    assert stats["bf16_kv_bytes"] == stats["stored_kv_bytes"] == expected
    assert stats["kv_compression_ratio"] == 1.0
    assert stats["bf16_window_bytes"] == 0
    assert stats["calls"] == 3


def test_hy_close_without_a_pass_counts_nothing(hy_accounting) -> None:
    hy_accounting["qvg_close_context_pass"]()
    assert hy_accounting["qvg_cache_stats"]() == {
        "bf16_kv_bytes": 0, "stored_kv_bytes": 0, "kv_compression_ratio": 1.0,
        "bf16_window_bytes": 0, "calls": 0, "last_call": None,
    }


def test_hy_stats_close_an_unclosed_pass_once(hy_accounting) -> None:
    for block_idx in range(2):
        raw = torch.randn(1, 12, 2, 4).to(torch.bfloat16)
        hy_accounting["_qvg_encode_context"]((raw, raw.clone(), None, None, None, 4), block_idx)
    first = hy_accounting["qvg_cache_stats"]()
    assert first["calls"] == 1
    assert hy_accounting["qvg_cache_stats"]() == first


def test_hy_layer_rows_keep_the_latest_pass_tensors(hy_accounting) -> None:
    """Rows are overwritten, not cleared: cache lifetimes match the v3 path."""
    bf16_pass(hy_accounting, layers=2, tokens=4)
    rows = hy_accounting["_QVG_LAYER_STATS"]
    assert sorted(rows) == [0, 1] and all(row["codec_payload"] is not None for row in rows.values())
    assert not hy_accounting["_QVG_OPEN_LAYERS"]
    assert all(not isinstance(value, torch.Tensor) for value in hy_accounting["_QVG_TOTALS"].values())


def test_hy_zero_storage_payload_fails_the_run(hy_accounting) -> None:
    hy_accounting["codec"].encode = lambda k, v: ({"shape": 1}, {"shape": 1})
    with pytest.raises(ValueError, match="zero bytes"):
        context_pass(hy_accounting, layers=1, tokens=12, recent=4)


def test_hy_transformer_closes_every_cache_vision_pass() -> None:
    tree = ast.parse(HY_TRANSFORMER.read_text(encoding="utf-8"))
    imports = [node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == "modules.attention"]
    assert any(alias.name == "qvg_close_context_pass" for node in imports for alias in node.names)
    [model] = [node for node in tree.body if isinstance(node, ast.ClassDef)
               and any(isinstance(item, ast.FunctionDef) and item.name == "forward_vision"
                       and any(arg.arg == "qvg_recent_latents" for arg in item.args.args) for item in node.body)]
    [forward] = [item for item in model.body if isinstance(item, ast.FunctionDef) and item.name == "forward_vision"]
    exits = [node for node in ast.walk(forward) if isinstance(node, ast.If)
             and ast.unparse(node.test) == "cache_vision"
             and any(isinstance(child, ast.Return) for child in node.body)]
    assert len(exits) == 1
    body = [ast.unparse(child) for child in exits[0].body]
    assert body == ["qvg_close_context_pass()", "return _kv_cache_new"]
