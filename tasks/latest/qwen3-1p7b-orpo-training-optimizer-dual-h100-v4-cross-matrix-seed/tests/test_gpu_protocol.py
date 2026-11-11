from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import TASK_ROOT


ENTRY_PATH = TASK_ROOT / "scripts/sforge_dual_h100_entry.py"


class _DeviceRequest:
    def __init__(self, **kwargs: object) -> None:
        self.driver = kwargs.get("driver")
        self.device_ids = kwargs.get("device_ids")
        self.capabilities = kwargs.get("capabilities")


class _DockerContainerHandle:
    def __init__(self, container: object) -> None:
        self.container = container
        self.id = getattr(container, "id", "")


class _DockerBackend:
    def create_container(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("adapter did not replace create_container")

    def cleanup_container(self, *args: object, **kwargs: object) -> None:
        pass


class _CodexAgent:
    install_cmds: list[str] = []

    def format_run_cmd(self, *args: object, **kwargs: object) -> object:
        return kwargs


def _load_adapter(monkeypatch: pytest.MonkeyPatch):
    """Import the entry module with tiny SForge/Docker protocol doubles."""

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
    docker_backend = types.ModuleType("sforge.harness.backend.docker_backend")
    docker_backend.DockerBackend = _DockerBackend
    docker_backend.DockerContainerHandle = _DockerContainerHandle
    cli = types.ModuleType("sforge.cli")
    cli.main = lambda: None

    modules = {
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
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = f"orpo_dual_h100_adapter_test_{id(monkeypatch)}"
    spec = importlib.util.spec_from_file_location(module_name, ENTRY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeNetwork:
    def __init__(self, internal: object) -> None:
        self.attrs = {"Internal": internal}
        self.reload_count = 0

    def reload(self) -> None:
        self.reload_count += 1


class _FakeNetworks:
    def __init__(self, network: _FakeNetwork) -> None:
        self.network = network
        self.requests: list[str] = []

    def get(self, name: str) -> _FakeNetwork:
        self.requests.append(name)
        return self.network


class _FakeContainers:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def create(self, image: str, **kwargs: object) -> SimpleNamespace:
        self.calls.append((image, kwargs))
        return SimpleNamespace(id=f"container-{len(self.calls)}")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_preflight_requires_an_explicit_gpu_pair_without_killing_processes():
    source = (TASK_ROOT / "scripts/preflight.py").read_text()
    # P4: no host GPU pair may be a default.  The operator names the pair, and
    # the flag is argparse-required so an unset pair is a usage error, never a
    # silent run against the authoring host's cards.
    assert 'default="6,7"' not in source
    assert '"--gpu-indices",\n        required=True,' in source
    # The preflight enforces the capability envelope, never a GPU model name:
    # any card at or above the floor is admitted, and the name it prints is
    # provenance in the JSON report.
    assert "NVIDIA H100 NVL" not in source
    assert "MIN_COMPUTE_CAPABILITY = (8, 0)" in source
    assert "MIN_DEVICE_MEMORY_BYTES" in source
    assert "is below the task minimum" in source
    assert "below the task floor" in source
    assert "kill(" not in source
    assert "terminate(" not in source
    assert "docker stop" not in source


def test_archive_evaluation_is_enabled_and_uses_the_dual_h100_adapter():
    source = (TASK_ROOT / "scripts/eval_archive.sh").read_text()
    assert "intentionally disabled" not in source
    assert "sforge_dual_h100_entry.py" in source
    assert "scripts/preflight.py" in source
    assert '--task "${TASK_ID}"' in source


def test_documentation_requires_same_pair_for_all_roles():
    path = TASK_ROOT / "docs/DESIGN.md"
    if not path.is_file():
        pytest.skip("maintainer design notes are not shipped")
    design = path.read_text()
    assert "physical GPU 6 and 7" in design
    assert "official TRL" in design
    assert "positive control" in design


def test_adapter_declares_managed_serial_lifecycle(monkeypatch: pytest.MonkeyPatch):
    adapter = _load_adapter(monkeypatch)
    doc = (adapter.__doc__ or "").lower()
    assert "waits for idle devices" in doc
    assert "pauses the exact run's work container" in doc
    assert "resumes work" in doc
    assert adapter.MODEL_REVISION == "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    assert adapter.MODEL_CONFIG_SHA256 == (
        "1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197"
    )
    assert adapter.PUBLIC_PARTITION_MANIFEST_SHA256 == (
        "ba9816131647aefe0d813fc7c7d3706894988943bc58e92c472667f722528c20"
    )
    assert set(adapter.PRIVATE_PARTITION_MANIFEST_SHA256) == {
        "private_cross_512",
        "private_cross_1024",
        "private_cross_2048",
    }


class _ManagedWork:
    def __init__(self) -> None:
        self.status = "running"
        self.pause_count = 0
        self.unpause_count = 0

    def reload(self) -> None:
        pass

    def pause(self) -> None:
        self.pause_count += 1
        self.status = "paused"

    def unpause(self) -> None:
        self.unpause_count += 1
        self.status = "running"


def test_managed_judge_pauses_and_resumes_exact_work_on_success_and_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    adapter = _load_adapter(monkeypatch)
    work = _ManagedWork()
    requested: list[str] = []

    class Containers:
        def get(self, name: str) -> _ManagedWork:
            requested.append(name)
            return work

    backend = SimpleNamespace(
        backend_name="docker",
        client=SimpleNamespace(containers=Containers()),
    )
    monkeypatch.setattr(adapter, "MANAGED_JUDGE_LOCK_PATH", tmp_path / "gpu.lock")
    monkeypatch.setattr(adapter, "_query_target_gpu_idle", lambda _uuids: True)
    # P4: the adapter has no default pair, so the run must name its own.
    monkeypatch.setenv(
        "ORPO_GPU_UUIDS",
        "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa,"
        "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
    )

    with adapter._exclusive_managed_judge(
        backend=backend,
        task_id=adapter.TARGET_TASK_ID,
        run_id="managed-test",
    ):
        assert work.status == "paused"
    assert work.status == "running"

    with pytest.raises(RuntimeError, match="candidate-independent fixture"):
        with adapter._exclusive_managed_judge(
            backend=backend,
            task_id=adapter.TARGET_TASK_ID,
            run_id="managed-test",
        ):
            raise RuntimeError("candidate-independent fixture")
    assert work.status == "running"
    assert work.pause_count == 2
    assert work.unpause_count == 2
    assert requested == [
        f"sforge.run.{adapter.TARGET_TASK_ID}.managed-test",
        f"sforge.run.{adapter.TARGET_TASK_ID}.managed-test",
    ]
    assert adapter.MODEL_MOUNT_ROOT == "/models/Qwen3-1.7B-repo"


_PAIR_A = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa,GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
_PAIR_B = "GPU-cccccccc-cccc-cccc-cccc-cccccccccccc,GPU-dddddddd-dddd-dddd-dddd-dddddddddddd"


def test_managed_judge_lease_is_per_gpu_pair(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    adapter = _load_adapter(monkeypatch)
    monkeypatch.setattr(adapter, "MANAGED_JUDGE_LOCK_PATH", tmp_path / "gpu.lock")
    a = adapter._managed_judge_lock_path(_PAIR_A.split(","))
    assert a == adapter._managed_judge_lock_path(list(reversed(_PAIR_A.upper().split(","))))
    assert a != adapter._managed_judge_lock_path(_PAIR_B.split(","))
    assert a.parent == tmp_path and a.name.startswith("gpu-")


def test_managed_judge_idle_wait_starts_after_the_pair_lease(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Queueing behind another Judge on the same pair is not gpu_busy; other pairs never queue."""
    import fcntl
    import threading
    import time

    adapter = _load_adapter(monkeypatch)
    monkeypatch.setattr(adapter, "MANAGED_JUDGE_LOCK_PATH", tmp_path / "gpu.lock")
    monkeypatch.setattr(adapter, "_query_target_gpu_idle", lambda _uuids: True)
    monkeypatch.setenv(adapter.MANAGED_WORK_IDLE_TIMEOUT_ENV, "1")
    monkeypatch.setenv("ORPO_GPU_UUIDS", _PAIR_A)
    backend = SimpleNamespace(
        backend_name="docker",
        client=SimpleNamespace(containers=SimpleNamespace(get=lambda _name: _ManagedWork())),
    )
    holder = adapter._managed_judge_lock_path(_PAIR_A.split(",")).open("a+")
    fcntl.flock(holder.fileno(), fcntl.LOCK_EX)
    threading.Timer(1.5, lambda: fcntl.flock(holder.fileno(), fcntl.LOCK_UN)).start()
    started = time.monotonic()
    with adapter._exclusive_managed_judge(
        backend=backend, task_id=adapter.TARGET_TASK_ID, run_id="queued"
    ):
        assert time.monotonic() - started >= 1.4
    holder.close()

    other = adapter._managed_judge_lock_path(_PAIR_A.split(",")).open("a+")
    fcntl.flock(other.fileno(), fcntl.LOCK_EX)
    monkeypatch.setenv("ORPO_GPU_UUIDS", _PAIR_B)
    started = time.monotonic()
    with adapter._exclusive_managed_judge(
        backend=backend, task_id=adapter.TARGET_TASK_ID, run_id="other-pair"
    ):
        assert time.monotonic() - started < 1.0
    other.close()
    assert adapter.MODEL_MOUNT == (
        "/models/Qwen3-1.7B-repo/snapshots/"
        "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    )


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa,GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb,GPU-cccccccc-cccc-cccc-cccc-cccccccccccc",
        "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa,GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "6,7",
        "all,GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "MIG-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa,GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
    ],
)
def test_adapter_rejects_non_exact_gpu_uuid_pairs(
    monkeypatch: pytest.MonkeyPatch, raw: str
):
    adapter = _load_adapter(monkeypatch)
    with pytest.raises(RuntimeError):
        adapter._gpu_uuids(raw)


def test_adapter_reads_two_distinct_explicit_uuids_from_environment(
    monkeypatch: pytest.MonkeyPatch,
):
    adapter = _load_adapter(monkeypatch)
    expected = [
        "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
    ]
    monkeypatch.setenv("ORPO_GPU_UUIDS", ", ".join(expected))
    assert adapter._gpu_uuids() == expected


def test_model_source_checks_revision_config_hash_and_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    adapter = _load_adapter(monkeypatch)
    repository = tmp_path / "models--Qwen--Qwen3-1.7B"
    snapshot = repository / "snapshots" / adapter.MODEL_REVISION
    snapshot.mkdir(parents=True)
    (repository / "blobs").mkdir()
    monkeypatch.setattr(adapter, "REQUIRED_MODEL_FILES", ("config.json",))
    config = {
        "model_type": "qwen3",
        "hidden_size": 2048,
        "num_hidden_layers": 28,
        "vocab_size": 151936,
        "tie_word_embeddings": True,
    }
    config_bytes = json.dumps(config, sort_keys=True).encode()
    (snapshot / "config.json").write_bytes(config_bytes)
    monkeypatch.setattr(adapter, "MODEL_CONFIG_SHA256", _sha256(config_bytes))
    monkeypatch.setenv("ORPO_MODEL_HOST_PATH", str(repository))
    assert adapter._model_source() == repository.resolve()

    config["vocab_size"] += 1
    bad_bytes = json.dumps(config, sort_keys=True).encode()
    (snapshot / "config.json").write_bytes(bad_bytes)
    monkeypatch.setattr(adapter, "MODEL_CONFIG_SHA256", _sha256(bad_bytes))
    with pytest.raises(RuntimeError, match="config identity"):
        adapter._model_source()

    wrong_repository = tmp_path / "mutable-main"
    (wrong_repository / "blobs").mkdir(parents=True)
    monkeypatch.setenv("ORPO_MODEL_HOST_PATH", str(wrong_repository))
    with pytest.raises(RuntimeError, match="cache root"):
        adapter._model_source()


def test_model_source_rejects_snapshot_symlinks_that_escape_repo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    adapter = _load_adapter(monkeypatch)
    repository = tmp_path / "models--Qwen--Qwen3-1.7B"
    snapshot = repository / "snapshots" / adapter.MODEL_REVISION
    snapshot.mkdir(parents=True)
    (repository / "blobs").mkdir()
    outside_config = tmp_path / "outside-config.json"
    outside_config.write_text("{}", encoding="utf-8")
    (snapshot / "config.json").symlink_to(outside_config)
    monkeypatch.setattr(adapter, "REQUIRED_MODEL_FILES", ("config.json",))
    monkeypatch.setenv("ORPO_MODEL_HOST_PATH", str(repository))
    with pytest.raises(RuntimeError, match="link escapes"):
        adapter._model_source()


def test_dataset_source_checks_root_partition_manifest_and_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    adapter = _load_adapter(monkeypatch)
    root_manifest = b'{"schema_version":1}\n'
    (tmp_path / "freeze_manifest.json").write_bytes(root_manifest)
    public = tmp_path / "public"
    public.mkdir()
    manifest = {
        "partition": "public",
        "provenance": {
            "dataset": {
                "repo_id": "trl-lib/tldr-preference",
                "revision": adapter.DATASET_REVISION,
                "config": "default",
            },
            "tokenizer": {
                "repo_id": "Qwen/Qwen3-1.7B",
                "revision": adapter.MODEL_REVISION,
            },
        },
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode()
    records = b'{"record_id":"public:0"}\n'
    tensors = b"safe-tensor-fixture"
    (public / "manifest.json").write_bytes(manifest_bytes)
    (public / "records.jsonl").write_bytes(records)
    (public / "tensors.safetensors").write_bytes(tensors)
    monkeypatch.setattr(
        adapter, "PUBLIC_DATASET_ROOT_MANIFEST_SHA256", _sha256(root_manifest)
    )
    monkeypatch.setattr(
        adapter, "PUBLIC_PARTITION_MANIFEST_SHA256", _sha256(manifest_bytes)
    )
    monkeypatch.setattr(
        adapter,
        "PUBLIC_PARTITION_ARTIFACT_SHA256",
        {
            "records.jsonl": _sha256(records),
            "tensors.safetensors": _sha256(tensors),
        },
    )
    monkeypatch.setenv("ORPO_DATA_HOST_ROOT", str(tmp_path))
    assert adapter._dataset_source("public") == public.resolve()

    (public / "tensors.safetensors").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        adapter._dataset_source("public")


def test_work_and_judge_use_two_uuid_device_request_and_disjoint_data_mounts(
    monkeypatch: pytest.MonkeyPatch,
):
    adapter = _load_adapter(monkeypatch)
    gpu_uuids = [
        "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
    ]
    network = _FakeNetwork(True)
    networks = _FakeNetworks(network)
    containers = _FakeContainers()
    backend = SimpleNamespace(
        client=SimpleNamespace(networks=networks, containers=containers)
    )
    monkeypatch.setenv("ORPO_GPU_UUIDS", ",".join(gpu_uuids))
    monkeypatch.setenv("SFORGE_ISOLATED_DOCKER_NETWORK", "orpo-internal")
    monkeypatch.setattr(adapter, "_model_source", lambda: Path("/host/model"))
    monkeypatch.setattr(
        adapter,
        "_dataset_source",
        lambda partition: Path(f"/host/data/{partition}"),
    )

    supplied_environment = {
        "CUDA_VISIBLE_DEVICES": "9",
        "HF_HUB_OFFLINE": "0",
        "UNRELATED": "kept",
    }
    work_handle = adapter.dual_h100_create_container(
        backend,
        "orpo.work.task:local",
        "orpo.work.container",
        environment=supplied_environment,
        cpu_limit=16,
        mem_limit="96g",
        cap_drop=["NET_RAW"],
        user="agent",
    )
    assert work_handle.id == "container-1"
    assert supplied_environment["CUDA_VISIBLE_DEVICES"] == "9"
    work_image, work = containers.calls[0]
    assert work_image == "orpo.work.task:local"
    assert work["network"] == "orpo-internal"
    assert "network_mode" not in work
    assert work["volumes"] == {
        "/host/model": {"bind": "/models/Qwen3-1.7B-repo", "mode": "ro"},
        "/host/data/public": {
            "bind": "/datasets/tldr-preference/public",
            "mode": "ro",
        },
    }
    assert "/host/data/private" not in work["volumes"]
    request = work["device_requests"][0]
    assert request.driver == "nvidia"
    assert request.device_ids == gpu_uuids
    assert request.capabilities == [["gpu"]]
    assert work["security_opt"] == ["no-new-privileges:true"]
    assert work["nano_cpus"] == 16_000_000_000
    assert work["mem_limit"] == "96g"
    assert work["cap_drop"] == ["NET_RAW"]
    assert work["user"] == "agent"
    assert "privileged" not in work
    work_env = work["environment"]
    assert work_env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert work_env["NVIDIA_VISIBLE_DEVICES"] == ",".join(gpu_uuids)
    assert work_env["ORPO_MODEL_PATH"] == (
        "/models/Qwen3-1.7B-repo/snapshots/"
        "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    )
    assert work_env["ORPO_DATA_PARTITION"] == "public"
    assert work_env["HF_HUB_OFFLINE"] == "1"
    assert work_env["TRANSFORMERS_OFFLINE"] == "1"
    assert work_env["HF_DATASETS_OFFLINE"] == "1"
    assert work_env["PIP_NO_INDEX"] == "1"
    assert work_env["UNRELATED"] == "kept"

    judge_handle = adapter.dual_h100_create_container(
        backend,
        "orpo.judge.task:local",
        "orpo.judge.container",
    )
    assert judge_handle.id == "container-2"
    _, judge = containers.calls[1]
    assert judge["network_mode"] == "none"
    assert "network" not in judge
    assert judge["volumes"] == {
        "/host/model": {"bind": "/models/Qwen3-1.7B-repo", "mode": "ro"},
        "/host/data/private": {
            "bind": "/datasets/tldr-preference",
            "mode": "ro",
        },
    }
    assert "/host/data/public" not in judge["volumes"]
    assert judge["device_requests"][0].device_ids == gpu_uuids
    assert judge["environment"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert judge["environment"]["ORPO_DATA_PARTITION"] == "private"


def test_replay_attestation_runs_before_create_and_after_cleanup_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
):
    adapter = _load_adapter(monkeypatch)
    monkeypatch.setenv(adapter.REPLAY_MODEL_TREE_SHA_ENV, "a" * 64)
    monkeypatch.setenv(adapter.REPLAY_PRIVATE_DATA_TREE_SHA_ENV, "b" * 64)
    monkeypatch.setenv(adapter.REPLAY_MODEL_HELPER_SHA_ENV, "c" * 64)
    monkeypatch.setenv(adapter.REPLAY_RUNTIME_IDENTITY_PATH_ENV, "/runtime.json")
    monkeypatch.setenv(adapter.REPLAY_RUNTIME_IDENTITY_SHA_ENV, "d" * 64)
    monkeypatch.setenv(adapter.REPLAY_RUNTIME_IDENTITY_HELPER_SHA_ENV, "e" * 64)
    monkeypatch.setenv(adapter.REPLAY_GPU_INDICES_ENV, "6,7")
    events: list[str] = []
    verification_count = 0

    def verify() -> None:
        nonlocal verification_count
        verification_count += 1
        events.append(f"verify-{verification_count}")
        if verification_count == 2:
            raise RuntimeError("live ORPO assets changed")

    monkeypatch.setattr(adapter, "_verify_replay_live_artifacts", verify)
    monkeypatch.setattr(adapter, "_model_source", lambda: Path("/host/model"))
    monkeypatch.setattr(
        adapter,
        "_dataset_source",
        lambda partition: Path(f"/host/data/{partition}"),
    )
    monkeypatch.setattr(
        adapter,
        "_gpu_uuids",
        lambda: [
            "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        ],
    )
    monkeypatch.setattr(
        adapter,
        "_original_cleanup_container",
        lambda _self, _handle, _logger=None: events.append("cleanup"),
    )
    containers = _FakeContainers()
    backend = SimpleNamespace(client=SimpleNamespace(containers=containers))

    handle = adapter.dual_h100_create_container(
        backend,
        "orpo.judge.task:local",
        "orpo.judge.container",
    )
    events.append("create")
    with pytest.raises(RuntimeError, match="live ORPO assets changed"):
        adapter.dual_h100_cleanup_container(backend, handle)

    assert events == ["verify-1", "create", "cleanup", "verify-2"]
    assert not adapter._replay_guarded_container_ids


def test_work_refuses_missing_or_non_internal_network(
    monkeypatch: pytest.MonkeyPatch,
):
    adapter = _load_adapter(monkeypatch)
    backend = SimpleNamespace(
        client=SimpleNamespace(networks=_FakeNetworks(_FakeNetwork(True)))
    )
    monkeypatch.delenv("SFORGE_ISOLATED_DOCKER_NETWORK", raising=False)
    with pytest.raises(RuntimeError, match="required for the Work"):
        adapter._work_network(backend)

    for internal in (False, "true", 1, None):
        fake_network = _FakeNetwork(internal)
        backend = SimpleNamespace(
            client=SimpleNamespace(networks=_FakeNetworks(fake_network))
        )
        monkeypatch.setenv("SFORGE_ISOLATED_DOCKER_NETWORK", "not-internal")
        with pytest.raises(RuntimeError, match="non-internal"):
            adapter._work_network(backend)
        assert fake_network.reload_count == 1


def test_codex_wrappers_follow_the_task_internet_flag(
    monkeypatch: pytest.MonkeyPatch,
):
    """internet: true (the owner policy) keeps live web search and a prompt
    without network text; the launcher's "false" restores offline mode."""

    adapter = _load_adapter(monkeypatch)
    install_check = adapter.CodexAgent.install_cmds[0]
    assert "codex --version" in install_check
    assert "0.147.0" not in install_check
    assert "SFORGE_CODEX_CLI_VERSION" not in install_check
    assert adapter.CodexAgent.format_run_cmd is adapter.task_internet_format_run_cmd
    assert (
        adapter.run_agent.generate_evolve_prompt
        is adapter.task_internet_generate_prompt
    )
    assert not hasattr(adapter, "offline_format_run_cmd")
    assert not hasattr(adapter, "offline_generate_prompt")

    def wrapped(requested: bool) -> tuple[object, object]:
        formatted = adapter.task_internet_format_run_cmd(object(), internet=requested)
        prompt = adapter.task_internet_generate_prompt(object(), internet=requested)
        return formatted["internet"], prompt["internet"]

    for value in (None, "true", " TRUE "):
        if value is None:
            monkeypatch.delenv(adapter.WORK_INTERNET_ENV, raising=False)
        else:
            monkeypatch.setenv(adapter.WORK_INTERNET_ENV, value)
        assert wrapped(True) == (True, True)
        assert wrapped(False) == (False, False)
    monkeypatch.setenv(adapter.WORK_INTERNET_ENV, "false")
    assert wrapped(True) == (False, False)
    monkeypatch.setenv(adapter.WORK_INTERNET_ENV, "yes")
    with pytest.raises(RuntimeError, match=adapter.WORK_INTERNET_ENV):
        wrapped(True)


def test_launcher_wires_the_task_internet_flag():
    source = (TASK_ROOT / "scripts/launch_fixed_budget_after_gpu67_idle.sh").read_text(encoding="utf-8")
    assert '"${TASK_ROOT}/tasks/${TASK_ID}.json"' in source
    assert "PROXY_EGRESS_ARGS=(--allow-public-egress)" in source
    assert '"${PROXY_EGRESS_ARGS[@]}"' in source
    assert 'export MLSYSFORGE_WORK_INTERNET="${TASK_INTERNET}"' in source
    assert "--internal" in source and "--enable-internet" in source
    assert (
        "llada-h100-decoder-optimization-v2-vanilla-seed/scripts/"
        "restricted_connect_proxy.py"
    ) in source
