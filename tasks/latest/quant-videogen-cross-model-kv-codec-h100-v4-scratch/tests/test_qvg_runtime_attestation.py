from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ENTRY = ROOT / "scripts/sforge_runtime_entry.py"
RUNTIME_IDENTITY = ROOT / "scripts/runtime_identity.py"
UUIDS = [f"GPU-{index:08x}-0000-4000-8000-{index:012x}" for index in range(1, 10)]
ADAPTER_ENVIRONMENT = (
    "QVG_JUDGE_GPU_UUIDS",
    "QVG_REPLAY_MODEL_LOCK_PATH",
    "QVG_REPLAY_MODEL_LOCK_SHA256",
    "QVG_REPLAY_VERIFY_MODELS_SHA256",
    "QVG_REPLAY_RUNTIME_IDENTITY_PATH",
    "QVG_REPLAY_RUNTIME_IDENTITY_SHA256",
    "QVG_REPLAY_RUNTIME_IDENTITY_HELPER_SHA256",
    "QVG_REPLAY_GPU_INDICES",
    "QVG_REPLAY_GPU_UUIDS",
    "SFORGE_ISOLATED_DOCKER_NETWORK",
)


class _DeviceRequest:
    def __init__(self, **kwargs: object) -> None:
        self.device_ids = kwargs.get("device_ids")


class _DockerContainerHandle:
    def __init__(self, container: object) -> None:
        self.container = container
        self.id = str(getattr(container, "id"))
        self.name = str(getattr(container, "name"))


class _DockerBackend:
    def create_container(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("adapter did not replace create_container")

    def cleanup_container(self, *args: object, **kwargs: object) -> None:
        return None


class _CodexAgent:
    install_cmds: list[str] = []

    def format_run_cmd(self, *args: object, **kwargs: object) -> object:
        return kwargs


def _load_adapter(monkeypatch: pytest.MonkeyPatch, artifact_root: Path):
    monkeypatch.setenv("QVG_ARTIFACT_ROOT", str(artifact_root))
    for name in ADAPTER_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    docker = types.ModuleType("docker")
    docker_types = types.ModuleType("docker.types")
    docker_types.DeviceRequest = _DeviceRequest
    docker.types = docker_types
    sforge = types.ModuleType("sforge")
    harness = types.ModuleType("sforge.harness")
    run_agent = types.ModuleType("sforge.harness.run_agent")
    run_agent.generate_evolve_prompt = lambda *args, **kwargs: kwargs
    harness.run_agent = run_agent
    agent = types.ModuleType("sforge.harness.agent")
    codex = types.ModuleType("sforge.harness.agent.codex")
    codex.CodexAgent = _CodexAgent
    backend = types.ModuleType("sforge.harness.backend")
    # The adapter forwards the single-allocation handoff contract (allocation
    # and validated binding labels); the stubs keep those helpers inert here.
    backend.AcceleratorAllocation = object
    backend.HANDOFF_BINDING_ANNOTATIONS = frozenset()
    docker_backend = types.ModuleType("sforge.harness.backend.docker_backend")
    docker_backend.DockerBackend = _DockerBackend
    docker_backend.DockerContainerHandle = _DockerContainerHandle
    docker_backend._validated_docker_labels = dict
    docker_backend._validated_handoff_binding_labels = dict
    cli = types.ModuleType("sforge.cli")
    cli.main = lambda: None
    codex_search = types.ModuleType("codex_search")
    codex_search.enable_native_web_search = lambda value: value
    for name, module in {
        "docker": docker,
        "docker.types": docker_types,
        "sforge": sforge,
        "sforge.harness": harness,
        "sforge.harness.run_agent": run_agent,
        "sforge.harness.agent": agent,
        "sforge.harness.agent.codex": codex,
        "sforge.harness.backend": backend,
        "sforge.harness.backend.docker_backend": docker_backend,
        "sforge.cli": cli,
        "codex_search": codex_search,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location(
        f"qvg_runtime_attestation_test_{id(monkeypatch)}", RUNTIME_ENTRY
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_replay_attestation_runs_before_create_and_after_cleanup_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _load_adapter(monkeypatch, tmp_path)
    monkeypatch.setenv(adapter.REPLAY_MODEL_LOCK_ENV, str(tmp_path / "model-lock.json"))
    monkeypatch.setenv(adapter.REPLAY_MODEL_LOCK_SHA_ENV, "a" * 64)
    monkeypatch.setenv(adapter.REPLAY_VERIFY_MODELS_SHA_ENV, "b" * 64)
    monkeypatch.setenv(adapter.REPLAY_RUNTIME_IDENTITY_PATH_ENV, str(tmp_path / "runtime.json"))
    monkeypatch.setenv(adapter.REPLAY_RUNTIME_IDENTITY_SHA_ENV, "c" * 64)
    monkeypatch.setenv(adapter.REPLAY_RUNTIME_IDENTITY_HELPER_SHA_ENV, "d" * 64)
    monkeypatch.setenv(adapter.REPLAY_GPU_INDICES_ENV, "4 5")
    monkeypatch.setenv(adapter.REPLAY_GPU_UUIDS_ENV, f"{UUIDS[0]} {UUIDS[1]}")
    monkeypatch.setenv(adapter.JUDGE_GPUS_ENV, f"{UUIDS[1]} {UUIDS[0]}")
    events: list[str] = []
    verification_count = 0

    def verify() -> None:
        nonlocal verification_count
        verification_count += 1
        events.append(f"verify-{verification_count}")
        if verification_count == 2:
            raise RuntimeError("live models changed")

    monkeypatch.setattr(adapter, "_verify_replay_live_artifacts", verify)
    monkeypatch.setattr(adapter, "judge_volumes", lambda: {})
    monkeypatch.setattr(
        adapter,
        "_original_cleanup_container",
        lambda _self, _handle, _logger=None: events.append("cleanup"),
    )

    class Containers:
        def create(self, _image: str, **kwargs: object) -> SimpleNamespace:
            events.append("create")
            return SimpleNamespace(id="judge-1", name=str(kwargs["name"]))

    backend = SimpleNamespace(client=SimpleNamespace(containers=Containers()))
    handle = adapter.mounted_create_container(
        backend,
        "qvg.judge.task:local",
        "sforge.judge.qvg.test",
        environment={},
    )
    with pytest.raises(RuntimeError, match="live models changed"):
        adapter.mounted_cleanup_container(backend, handle)

    assert events == ["verify-1", "create", "cleanup", "verify-2"]
    assert not adapter._replay_guarded_container_ids


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("boot", "boot_id", "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        ("docker", "daemon_id", "NEW-DAEMON"),
        ("gpus", "driver_version", "999.99"),
        ("gpus", "power_limit", "650.00"),
    ],
)
def test_runtime_identity_rejects_changed_host_contract(
    section: str, field: str, replacement: str
) -> None:
    spec = importlib.util.spec_from_file_location(
        "qvg_host_runtime_identity_test", RUNTIME_IDENTITY
    )
    assert spec is not None and spec.loader is not None
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    observed = {
        "boot": {
            "boot_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "kernel": {"release": "6.8.0", "version": "one", "machine": "x86_64"},
        },
        "docker": {"daemon_id": "DAEMON", "default_runtime": "runc"},
        "gpus": [
            {
                "index": "4",
                "uuid": "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "name": "NVIDIA H100 NVL",
                "driver_version": "570.00",
                "power_limit": "600.00",
            }
        ],
    }
    baseline = helper.build_identity_document(observed)
    changed = json.loads(json.dumps(observed))
    if section == "gpus":
        changed[section][0][field] = replacement
    else:
        changed[section][field] = replacement
    current = helper.build_identity_document(changed)

    assert baseline["identity_sha256"] != current["identity_sha256"]
    with pytest.raises(RuntimeError, match="host runtime identity changed"):
        helper.verify_same_identity(baseline, current)


class _Containers:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def create(self, image: str, **kwargs: object) -> SimpleNamespace:
        self.calls.append((image, kwargs))
        return SimpleNamespace(id=f"container-{len(self.calls)}", name=str(kwargs["name"]))


def _backend() -> tuple[SimpleNamespace, _Containers]:
    containers = _Containers()
    network = SimpleNamespace(attrs={"Internal": True})
    client = SimpleNamespace(
        containers=containers, networks=SimpleNamespace(get=lambda _name: network)
    )
    return SimpleNamespace(client=client), containers


def _mountless_adapter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    adapter = _load_adapter(monkeypatch, tmp_path)
    monkeypatch.setattr(adapter, "judge_volumes", lambda: {})
    monkeypatch.setattr(adapter, "work_volumes", lambda: {})
    return adapter


def _create_judge(adapter, environment: dict[str, str], allocation=None):
    backend, containers = _backend()
    adapter.mounted_create_container(
        backend,
        "qvg.judge.task:local",
        "sforge.judge.qvg.test",
        environment=environment,
        accelerator_allocation=allocation,
    )
    (_, kwargs), = containers.calls
    return kwargs


def test_judge_gets_every_listed_gpu_in_one_device_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _mountless_adapter(monkeypatch, tmp_path)
    monkeypatch.setenv(adapter.JUDGE_GPUS_ENV, " ".join(UUIDS[:8]))
    kwargs = _create_judge(adapter, {"CUDA_VISIBLE_DEVICES": "0", "KEEP": "1"})
    (request,) = kwargs["device_requests"]
    assert request.device_ids == UUIDS[:8]
    environment = kwargs["environment"]
    assert environment["NVIDIA_VISIBLE_DEVICES"] == ",".join(UUIDS[:8])
    # Every Judge GPU stays visible to the controller; no index mask survives.
    assert "CUDA_VISIBLE_DEVICES" not in environment
    assert environment["KEEP"] == "1"
    assert kwargs["network_mode"] == "none"


def test_judge_list_accepts_semicolons_and_an_equal_allocation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _mountless_adapter(monkeypatch, tmp_path)
    monkeypatch.setenv(adapter.JUDGE_GPUS_ENV, f"{UUIDS[0]};{UUIDS[1]}")
    allocation = SimpleNamespace(vendor="nvidia", device_ids=(UUIDS[1], UUIDS[0]))
    kwargs = _create_judge(adapter, {}, allocation)
    assert kwargs["device_requests"][0].device_ids == UUIDS[:2]


@pytest.mark.parametrize(
    "raw",
    [
        "GPU-not-a-uuid",
        f"{UUIDS[0]} {UUIDS[0]}",
        " ".join(UUIDS[:9]),
        "0 1",
    ],
)
def test_judge_list_must_name_one_to_eight_distinct_whole_gpus(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, raw: str
) -> None:
    adapter = _mountless_adapter(monkeypatch, tmp_path)
    monkeypatch.setenv(adapter.JUDGE_GPUS_ENV, raw)
    with pytest.raises(RuntimeError, match="QVG_JUDGE_GPU_UUIDS"):
        _create_judge(adapter, {})


def test_judge_rejects_a_pin_or_allocation_that_disagrees_with_the_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _mountless_adapter(monkeypatch, tmp_path)
    monkeypatch.setenv(adapter.JUDGE_GPUS_ENV, f"{UUIDS[0]} {UUIDS[1]}")
    with pytest.raises(RuntimeError, match="disagrees"):
        _create_judge(adapter, {"NVIDIA_VISIBLE_DEVICES": UUIDS[0]})
    allocation = SimpleNamespace(vendor="nvidia", device_ids=(UUIDS[0],))
    with pytest.raises(RuntimeError, match="accelerator allocation"):
        _create_judge(adapter, {}, allocation)


def test_judge_without_the_list_falls_back_to_one_pinned_device(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _mountless_adapter(monkeypatch, tmp_path)
    kwargs = _create_judge(
        adapter, {"NVIDIA_VISIBLE_DEVICES": UUIDS[2], "CUDA_VISIBLE_DEVICES": "0"}
    )
    assert kwargs["device_requests"][0].device_ids == [UUIDS[2]]
    assert "CUDA_VISIBLE_DEVICES" not in kwargs["environment"]
    with pytest.raises(RuntimeError, match="exactly one"):
        _create_judge(adapter, {"NVIDIA_VISIBLE_DEVICES": f"{UUIDS[0]},{UUIDS[1]}"})


def test_work_keeps_exactly_one_gpu_outside_the_judge_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _mountless_adapter(monkeypatch, tmp_path)
    monkeypatch.setenv("SFORGE_ISOLATED_DOCKER_NETWORK", "qvg-internal")
    monkeypatch.setenv(adapter.JUDGE_GPUS_ENV, f"{UUIDS[0]} {UUIDS[1]}")

    def create_work(environment: dict[str, str]) -> dict[str, object]:
        backend, containers = _backend()
        adapter.mounted_create_container(
            backend,
            "qvg.work.task:local",
            "sforge.run.qvg.test",
            environment=environment,
        )
        (_, kwargs), = containers.calls
        return kwargs

    kwargs = create_work({"NVIDIA_VISIBLE_DEVICES": UUIDS[2], "CUDA_VISIBLE_DEVICES": "0"})
    assert kwargs["device_requests"][0].device_ids == [UUIDS[2]]
    assert kwargs["environment"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert kwargs["network"] == "qvg-internal"
    with pytest.raises(RuntimeError, match="exactly one"):
        create_work({"NVIDIA_VISIBLE_DEVICES": f"{UUIDS[2]},{UUIDS[3]}"})
    with pytest.raises(RuntimeError, match="one of the Judge GPUs"):
        create_work({"NVIDIA_VISIBLE_DEVICES": UUIDS[1]})


def test_replay_attestation_must_cover_exactly_the_judge_gpus(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _mountless_adapter(monkeypatch, tmp_path)
    monkeypatch.setenv(adapter.REPLAY_MODEL_LOCK_ENV, str(tmp_path / "model-lock.json"))
    monkeypatch.setenv(adapter.JUDGE_GPUS_ENV, f"{UUIDS[0]} {UUIDS[1]}")
    monkeypatch.setenv(adapter.REPLAY_GPU_INDICES_ENV, "4")
    monkeypatch.setenv(adapter.REPLAY_GPU_UUIDS_ENV, UUIDS[0])
    verified: list[str] = []
    monkeypatch.setattr(adapter, "_verify_replay_live_artifacts", lambda: verified.append("x"))
    with pytest.raises(RuntimeError, match="does not cover the Judge GPUs"):
        _create_judge(adapter, {})
    assert verified == []


@pytest.mark.parametrize(
    ("indices", "uuids"),
    [
        ("", ""),
        ("4", f"{UUIDS[0]} {UUIDS[1]}"),
        ("4 4", f"{UUIDS[0]} {UUIDS[1]}"),
        ("04", UUIDS[0]),
        ("4", "GPU-not-a-uuid"),
    ],
)
def test_replay_gpu_selection_is_paired_canonical_and_distinct(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, indices: str, uuids: str
) -> None:
    adapter = _load_adapter(monkeypatch, tmp_path)
    monkeypatch.setenv(adapter.REPLAY_GPU_INDICES_ENV, indices)
    monkeypatch.setenv(adapter.REPLAY_GPU_UUIDS_ENV, uuids)
    with pytest.raises(RuntimeError, match="incomplete QVG replay attestation contract"):
        adapter._replay_gpu_selection()


def test_replay_gpu_selection_pairs_indices_with_uuids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _load_adapter(monkeypatch, tmp_path)
    monkeypatch.setenv(adapter.REPLAY_GPU_INDICES_ENV, "0 7")
    monkeypatch.setenv(adapter.REPLAY_GPU_UUIDS_ENV, f"{UUIDS[0]} {UUIDS[7]}")
    assert adapter._replay_gpu_selection() == (["0", "7"], [UUIDS[0], UUIDS[7]])
