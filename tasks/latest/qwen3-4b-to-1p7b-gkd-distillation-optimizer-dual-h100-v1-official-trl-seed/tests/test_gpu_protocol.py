from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import MAINTAINER_DOCS, TASK_ROOT


ENTRY_PATH = TASK_ROOT / "scripts/sforge_dual_h100_entry.py"
PROBE_PATH = TASK_ROOT / "images/work/workdir/gkd_lab/runtime/distributed_probe.py"
PUBLIC_PROBE_PATH = TASK_ROOT / "images/work/public_tests/public_train_probe.py"


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

    module_name = f"gkd_dual_h100_adapter_test_{id(monkeypatch)}"
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


def _load_preflight():
    path = TASK_ROOT / "scripts/preflight.py"
    spec = importlib.util.spec_from_file_location("gkd_preflight_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("mode", "is_dir", "accepted"),
    [
        (0o711, True, True),   # traversable but unlistable: the frozen root
        (0o755, True, True),
        (0o700, True, False),  # root-only: candidate code cannot traverse
        (0o644, False, True),
        (0o600, False, False),
    ],
)
def test_preflight_rejects_data_only_the_privileged_role_could_read(
    tmp_path: Path, mode: int, is_dir: bool, accepted: bool
):
    """Candidate code runs de-privileged; the reference runs as root.

    A frozen tree only root can read fails on exactly one side, deep inside a
    formal run, which reads as a candidate defect rather than a deployment
    mistake. Preflight has to catch it before any GPU work starts.
    """

    preflight = _load_preflight()
    target = tmp_path / "artifact"
    if is_dir:
        target.mkdir()
    else:
        target.write_bytes(b"payload")
    target.chmod(mode)
    if accepted:
        preflight.require_deprivileged_reachable(target, "artifact")
        return
    with pytest.raises(SystemExit) as failure:
        preflight.require_deprivileged_reachable(target, "artifact")
    assert "de-privileged" in str(failure.value)


def test_preflight_takes_the_pair_from_the_operator_without_killing_processes():
    source = (TASK_ROOT / "scripts/preflight.py").read_text()
    assert 'os.environ.get("GKD_GPU_INDICES", "")' in source
    assert "name the GPU pair" in source
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


def test_no_launcher_bakes_in_a_physical_gpu_pair():
    """The pair is an operator input, never a property of the task.

    Containers always receive the selected pair as CUDA devices 0,1, so a
    hard-coded index or UUID would only serve to freeze the task onto one
    machine's silicon.
    """

    uuid_pattern = re.compile(
        r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        re.IGNORECASE,
    )
    index_pattern = re.compile(r"--gpu-indices[= ]+\"?[0-9]+,[0-9]+")
    for path in sorted((TASK_ROOT / "scripts").glob("*")):
        if path.suffix not in {".sh", ".py"}:
            continue
        source = path.read_text(encoding="utf-8")
        assert not uuid_pattern.search(source), f"{path.name} pins a physical GPU UUID"
        assert not index_pattern.search(source), f"{path.name} pins physical GPU indices"


def test_archive_evaluation_is_enabled_and_uses_the_dual_h100_adapter():
    source = (TASK_ROOT / "scripts/eval_archive.sh").read_text()
    assert "intentionally disabled" not in source
    assert "sforge_dual_h100_entry.py" in source
    assert "scripts/preflight.py" in source
    assert '--task "${TASK_ID}"' in source


@pytest.mark.skipif(not MAINTAINER_DOCS, reason="maintainer docs are not in the public tree")
def test_documentation_requires_same_pair_for_all_roles():
    design = (TASK_ROOT / "docs/DESIGN.md").read_text()
    assert "dual-H100" in design
    assert "official TRL" in design
    assert "positive control" in design
    readme = (TASK_ROOT / "README.md").read_text()
    assert "GKD_GPU_INDICES=" in readme
    assert "GKD_GPU_UUIDS" in readme
    assert "pins no GPU pair of its own" in readme
    assert "CUDA devices `0,1`" in readme


def test_distributed_probe_arguments_follow_the_gkd_protocol(
    monkeypatch: pytest.MonkeyPatch,
):
    from runtime import distributed_probe

    monkeypatch.setenv("GKD_STUDENT_MODEL_PATH", "/env/student")
    monkeypatch.setenv("GKD_TEACHER_MODEL_PATH", "/env/teacher")
    monkeypatch.setattr(
        sys,
        "argv",
        ["distributed_probe", "--kind", "reference", "--output", "/tmp/out.json"],
    )
    args = distributed_probe.parse_args()
    assert args.student_model_path == Path("/env/student")
    assert args.teacher_model_path == Path("/env/teacher")
    assert args.lab_root == Path("/app/workdir/gkd_lab")
    assert args.seed == 20260822
    assert args.batch == 1
    assert args.beta == 0.5
    assert args.temperature == 0.9

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "distributed_probe",
            "--kind",
            "candidate",
            "--candidate-version",
            "7",
            "--output",
            "/tmp/out.json",
            "--model-path",
            "/cli/student",
            "--teacher-model-path",
            "/cli/teacher",
            "--pair-batch",
            "3",
            "--beta",
            "0.25",
            "--temperature",
            "1.3",
        ],
    )
    args = distributed_probe.parse_args()
    assert args.student_model_path == Path("/cli/student")
    assert args.teacher_model_path == Path("/cli/teacher")
    assert args.batch == 3
    assert args.beta == 0.25
    assert args.temperature == 1.3


@pytest.mark.parametrize(
    "extra",
    [
        ["--beta", "1.5"],
        ["--beta", "-0.1"],
        ["--temperature", "0.0"],
        ["--temperature", "-1.0"],
        ["--batch", "0"],
        ["--candidate-version", "0"],
        ["--sequence-length", "4096"],
        ["--data-path", "/tmp/data"],
        ["--expected-manifest-sha256", "a" * 64],
    ],
)
def test_distributed_probe_rejects_out_of_contract_arguments(
    monkeypatch: pytest.MonkeyPatch, extra: list[str]
):
    from runtime import distributed_probe

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "distributed_probe",
            "--kind",
            "reference",
            "--output",
            "/tmp/out.json",
            *extra,
        ],
    )
    with pytest.raises(SystemExit):
        distributed_probe.parse_args()


def test_distributed_probe_payload_contract_includes_teacher_evidence():
    source = PROBE_PATH.read_text()
    assert '"teacher_model_optimization": teacher_model_optimization' in source
    assert '"objective_contract"' in source
    assert '"teacher_witness"' in source
    assert '"execution": "per_rank_replicated_no_grad_eval"' in source
    assert '"inside_timed_window": True' in source
    assert '"wrapped_in_fsdp": False' in source
    assert '"batch_per_rank": args.batch' in source
    assert '"row_order_on_disk": "single_sequence_per_record"' in source
    assert '"labels_contract": "completion_only"' in source
    assert "pair_batch_per_rank" not in source
    assert "nll_target" not in source
    assert "chosen_first" not in source


def test_public_train_probe_forwards_the_gkd_two_model_protocol(
    monkeypatch: pytest.MonkeyPatch,
):
    source = PUBLIC_PROBE_PATH.read_text()
    assert '"runtime.distributed_probe"' in source
    assert '"--student-model-path"' in source
    assert '"--teacher-model-path"' in source
    assert '"--beta"' in source
    assert '"--temperature"' in source
    assert "GKD_STUDENT_MODEL_PATH" in source
    assert "GKD_TEACHER_MODEL_PATH" in source

    spec = importlib.util.spec_from_file_location(
        f"gkd_public_train_probe_test_{id(monkeypatch)}", PUBLIC_PROBE_PATH
    )
    assert spec is not None and spec.loader is not None
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)

    monkeypatch.setattr(
        sys,
        "argv",
        ["public_train_probe", "--pair-batch", "2", "--candidate-version", "3"],
    )
    args = probe.parse_args()
    assert args.batch == 2
    assert args.candidate_version == 3
    assert args.beta == 0.5
    assert args.temperature == 0.9
    assert args.seed == 20260822

    for extra in (["--beta", "2.0"], ["--temperature", "0"], ["--batch", "0"]):
        monkeypatch.setattr(sys, "argv", ["public_train_probe", *extra])
        with pytest.raises(SystemExit):
            probe.parse_args()


def test_adapter_declares_managed_serial_lifecycle(monkeypatch: pytest.MonkeyPatch):
    adapter = _load_adapter(monkeypatch)
    doc = (adapter.__doc__ or "").lower()
    assert "waits for idle devices" in doc
    assert "pauses the exact run's work container" in doc
    assert "resumes work" in doc
    assert adapter.STUDENT_MODEL_REVISION == "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    assert adapter.STUDENT_MODEL_CONFIG_SHA256 == (
        "1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197"
    )
    assert adapter.TEACHER_MODEL_REVISION == "1cfa9a7208912126459214e8b04321603b3df60c"
    assert adapter.TEACHER_MODEL_CONFIG_SHA256 == (
        "8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a"
    )
    assert adapter.PUBLIC_PARTITION_MANIFEST_SHA256 == (
        "e653d0a59a07d7341197263a28424f85940ef633cce1e3a93891095cf2e04956"
    )
    assert set(adapter.PRIVATE_PARTITION_MANIFEST_SHA256) == {
        "private_gkd_512",
        "private_gkd_1024",
        "private_gkd_2048",
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
    monkeypatch.setenv(
        "GKD_GPU_UUIDS",
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
    assert adapter.STUDENT_MODEL_MOUNT_ROOT == "/models/Qwen3-1.7B-repo"
    assert adapter.STUDENT_MODEL_MOUNT == (
        "/models/Qwen3-1.7B-repo/snapshots/"
        "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    )
    assert adapter.TEACHER_MODEL_MOUNT_ROOT == "/models/Qwen3-4B-repo"
    assert adapter.TEACHER_MODEL_MOUNT == (
        "/models/Qwen3-4B-repo/snapshots/"
        "1cfa9a7208912126459214e8b04321603b3df60c"
    )


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "GPU-22c98c90-b7be-5cb7-47cc-1f097071187b",
        "GPU-22c98c90-b7be-5cb7-47cc-1f097071187b,GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb,GPU-cccccccc-cccc-cccc-cccc-cccccccccccc",
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
    monkeypatch.setenv("GKD_GPU_UUIDS", ", ".join(expected))
    assert adapter._gpu_uuids() == expected


def test_student_model_source_checks_revision_config_hash_and_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    adapter = _load_adapter(monkeypatch)
    repository = tmp_path / "models--Qwen--Qwen3-1.7B"
    snapshot = repository / "snapshots" / adapter.STUDENT_MODEL_REVISION
    snapshot.mkdir(parents=True)
    (repository / "blobs").mkdir()
    monkeypatch.setattr(adapter, "STUDENT_REQUIRED_MODEL_FILES", ("config.json",))
    config = {
        "model_type": "qwen3",
        "hidden_size": 2048,
        "num_hidden_layers": 28,
        "vocab_size": 151936,
        "tie_word_embeddings": True,
    }
    config_bytes = json.dumps(config, sort_keys=True).encode()
    (snapshot / "config.json").write_bytes(config_bytes)
    monkeypatch.setattr(adapter, "STUDENT_MODEL_CONFIG_SHA256", _sha256(config_bytes))
    monkeypatch.setenv("GKD_STUDENT_MODEL_HOST_PATH", str(repository))
    assert adapter._student_model_source() == repository.resolve()

    config["vocab_size"] += 1
    bad_bytes = json.dumps(config, sort_keys=True).encode()
    (snapshot / "config.json").write_bytes(bad_bytes)
    monkeypatch.setattr(adapter, "STUDENT_MODEL_CONFIG_SHA256", _sha256(bad_bytes))
    with pytest.raises(RuntimeError, match="config identity"):
        adapter._student_model_source()

    wrong_repository = tmp_path / "mutable-main"
    (wrong_repository / "blobs").mkdir(parents=True)
    monkeypatch.setenv("GKD_STUDENT_MODEL_HOST_PATH", str(wrong_repository))
    with pytest.raises(RuntimeError, match="cache root"):
        adapter._student_model_source()


def test_teacher_model_source_checks_revision_config_hash_and_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    adapter = _load_adapter(monkeypatch)
    repository = tmp_path / "models--Qwen--Qwen3-4B"
    snapshot = repository / "snapshots" / adapter.TEACHER_MODEL_REVISION
    snapshot.mkdir(parents=True)
    (repository / "blobs").mkdir()
    monkeypatch.setattr(adapter, "TEACHER_REQUIRED_MODEL_FILES", ("config.json",))
    config = {
        "model_type": "qwen3",
        "hidden_size": 2560,
        "num_hidden_layers": 36,
        "vocab_size": 151936,
        "tie_word_embeddings": True,
    }
    config_bytes = json.dumps(config, sort_keys=True).encode()
    (snapshot / "config.json").write_bytes(config_bytes)
    monkeypatch.setattr(adapter, "TEACHER_MODEL_CONFIG_SHA256", _sha256(config_bytes))
    monkeypatch.setenv("GKD_TEACHER_MODEL_HOST_PATH", str(repository))
    assert adapter._teacher_model_source() == repository.resolve()

    config["hidden_size"] = 2048
    bad_bytes = json.dumps(config, sort_keys=True).encode()
    (snapshot / "config.json").write_bytes(bad_bytes)
    monkeypatch.setattr(adapter, "TEACHER_MODEL_CONFIG_SHA256", _sha256(bad_bytes))
    with pytest.raises(RuntimeError, match="config identity"):
        adapter._teacher_model_source()

    wrong_repository = tmp_path / "mutable-main"
    (wrong_repository / "blobs").mkdir(parents=True)
    monkeypatch.setenv("GKD_TEACHER_MODEL_HOST_PATH", str(wrong_repository))
    with pytest.raises(RuntimeError, match="cache root"):
        adapter._teacher_model_source()


def test_model_source_rejects_snapshot_symlinks_that_escape_repo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    adapter = _load_adapter(monkeypatch)
    repository = tmp_path / "models--Qwen--Qwen3-1.7B"
    snapshot = repository / "snapshots" / adapter.STUDENT_MODEL_REVISION
    snapshot.mkdir(parents=True)
    (repository / "blobs").mkdir()
    outside_config = tmp_path / "outside-config.json"
    outside_config.write_text("{}", encoding="utf-8")
    (snapshot / "config.json").symlink_to(outside_config)
    monkeypatch.setattr(adapter, "STUDENT_REQUIRED_MODEL_FILES", ("config.json",))
    monkeypatch.setenv("GKD_STUDENT_MODEL_HOST_PATH", str(repository))
    with pytest.raises(RuntimeError, match="link escapes"):
        adapter._student_model_source()


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
                "revision": adapter.STUDENT_MODEL_REVISION,
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
        adapter, "DATASET_ROOT_MANIFEST_SHA256", _sha256(root_manifest)
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
    monkeypatch.setenv("GKD_DATA_HOST_ROOT", str(tmp_path))
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
    monkeypatch.setenv("GKD_GPU_UUIDS", ",".join(gpu_uuids))
    monkeypatch.setenv("SFORGE_ISOLATED_DOCKER_NETWORK", "gkd-internal")
    monkeypatch.setattr(
        adapter, "_student_model_source", lambda: Path("/host/student-model")
    )
    monkeypatch.setattr(
        adapter, "_teacher_model_source", lambda: Path("/host/teacher-model")
    )
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
        "gkd.work.task:local",
        "gkd.work.container",
        environment=supplied_environment,
        cpu_limit=16,
        mem_limit="96g",
        cap_drop=["NET_RAW"],
        user="agent",
    )
    assert work_handle.id == "container-1"
    assert supplied_environment["CUDA_VISIBLE_DEVICES"] == "9"
    work_image, work = containers.calls[0]
    assert work_image == "gkd.work.task:local"
    assert work["network"] == "gkd-internal"
    assert "network_mode" not in work
    assert work["volumes"] == {
        "/host/student-model": {"bind": "/models/Qwen3-1.7B-repo", "mode": "ro"},
        "/host/teacher-model": {"bind": "/models/Qwen3-4B-repo", "mode": "ro"},
        "/host/data/public": {
            "bind": "/datasets/tldr-gkd/public",
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
    assert work_env["GKD_STUDENT_MODEL_PATH"] == (
        "/models/Qwen3-1.7B-repo/snapshots/"
        "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    )
    assert work_env["GKD_TEACHER_MODEL_PATH"] == (
        "/models/Qwen3-4B-repo/snapshots/"
        "1cfa9a7208912126459214e8b04321603b3df60c"
    )
    assert work_env["GKD_DATA_PATH"] == "/datasets/tldr-gkd"
    assert work_env["GKD_DATA_PARTITION"] == "public"
    assert work_env["HF_HUB_OFFLINE"] == "1"
    assert work_env["TRANSFORMERS_OFFLINE"] == "1"
    assert work_env["HF_DATASETS_OFFLINE"] == "1"
    assert work_env["PIP_NO_INDEX"] == "1"
    assert work_env["UNRELATED"] == "kept"

    judge_handle = adapter.dual_h100_create_container(
        backend,
        "gkd.judge.task:local",
        "gkd.judge.container",
    )
    assert judge_handle.id == "container-2"
    _, judge = containers.calls[1]
    assert judge["network_mode"] == "none"
    assert "network" not in judge
    assert judge["volumes"] == {
        "/host/student-model": {"bind": "/models/Qwen3-1.7B-repo", "mode": "ro"},
        "/host/teacher-model": {"bind": "/models/Qwen3-4B-repo", "mode": "ro"},
        "/host/data/private": {
            "bind": "/datasets/tldr-gkd",
            "mode": "ro",
        },
    }
    assert "/host/data/public" not in judge["volumes"]
    assert judge["device_requests"][0].device_ids == gpu_uuids
    assert judge["environment"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert judge["environment"]["GKD_DATA_PARTITION"] == "private"


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
            raise RuntimeError("live GKD assets changed")

    monkeypatch.setattr(adapter, "_verify_replay_live_artifacts", verify)
    monkeypatch.setattr(
        adapter, "_student_model_source", lambda: Path("/host/student-model")
    )
    monkeypatch.setattr(
        adapter, "_teacher_model_source", lambda: Path("/host/teacher-model")
    )
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
        "gkd.judge.task:local",
        "gkd.judge.container",
    )
    events.append("create")
    with pytest.raises(RuntimeError, match="live GKD assets changed"):
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
    source = (TASK_ROOT / "scripts/launch_fixed_budget_after_gpu_pair_idle.sh").read_text(encoding="utf-8")
    assert '"${TASK_ROOT}/tasks/${TASK_ID}.json"' in source
    assert "PROXY_EGRESS_ARGS=(--allow-public-egress)" in source
    assert '"${PROXY_EGRESS_ARGS[@]}"' in source
    assert 'export MLSYSFORGE_WORK_INTERNET="${TASK_INTERNET}"' in source
    assert "--internal" in source and "--enable-internet" in source
    assert (
        "llada-h100-decoder-optimization-v2-vanilla-seed/scripts/"
        "restricted_connect_proxy.py"
    ) in source
