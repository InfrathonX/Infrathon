"""scripts/sforge_local_entry_v3.py: the local Docker Work/Judge setup PyroMind's controller does."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "minference_sparse_prefill_h100_seed_v3"
WORK_GPU = "GPU-a0c3447e-5ddc-48ce-e1c0-9c27711d6900"
JUDGE_GPU = "GPU-da213503-a9e4-73bd-3b65-cbe49aced2a1"


@pytest.fixture
def entry(tmp_path, monkeypatch):
    """Import the entry (it patches SForge globally) and undo every patch afterwards."""
    from sforge import cli
    from sforge.harness import judge_server, run_agent, run_evaluation
    from sforge.harness.agent.codex import CodexAgent
    from sforge.harness.backend.docker_backend import DockerBackend
    saved = [(DockerBackend, "create_container"), (DockerBackend, "start_container"),
             (run_evaluation, "judge_submission"), (judge_server, "judge_submission"),
             (cli, "judge_submission"), (CodexAgent, "format_run_cmd"), (run_agent, "generate_evolve_prompt")]
    originals = [(owner, name, owner.__dict__[name] if isinstance(owner, type) else getattr(owner, name))
                 for owner, name in saved]
    for name in ("model", "work_model", "public", "private"):
        (tmp_path / name).mkdir()
    monkeypatch.setenv("MINFERENCE_MODEL_DIR", str(tmp_path / "model"))
    monkeypatch.setenv("MINFERENCE_WORK_MODEL_DIR", str(tmp_path / "work_model"))
    monkeypatch.setenv("MINFERENCE_PUBLIC_PROMPTS_DIR", str(tmp_path / "public"))
    monkeypatch.setenv("MINFERENCE_PRIVATE_PROMPTS_DIR", str(tmp_path / "private"))
    monkeypatch.setenv("MINFERENCE_JUDGE_GPU_UUID", JUDGE_GPU)
    spec = importlib.util.spec_from_file_location("minference_local_entry", ROOT / "scripts/sforge_local_entry_v3.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module, tmp_path
    finally:
        for owner, name, value in originals:
            setattr(owner, name, value)
        sys.modules.pop(spec.name, None)


class FakeClient:
    def __init__(self, internal=True):
        self.created = []
        self.networks = SimpleNamespace(get=lambda name: SimpleNamespace(attrs={"Internal": internal},
                                                                          reload=lambda: None))
        self.containers = SimpleNamespace(create=self._create)

    def _create(self, image, **kwargs):
        self.created.append(kwargs)
        return SimpleNamespace(id="%064x" % len(self.created), name=kwargs["name"])


def backend(client):
    from sforge.harness.backend.docker_backend import DockerBackend
    return DockerBackend(client)


def test_judge_gets_exactly_the_two_bootstrap_mounts_offline_on_its_gpu(entry):
    module, tmp_path = entry
    from sforge.harness.backend import AcceleratorAllocation
    client = FakeClient()
    handle = backend(client).create_container(
        "sha256:" + "a" * 64, f"minference_sparse_prefill.judge.{TASK_ID}.agent-1", network_mode="none",
        user="root", accelerator_allocation=AcceleratorAllocation("nvidia", (JUDGE_GPU,)))
    created = client.created[0]
    assert created["network_mode"] == "none" and created["user"] == "root"
    assert created["volumes"] == {
        str(tmp_path / "model"): {"bind": module.ASSETS.MOUNTS["model"], "mode": "ro"},
        str(tmp_path / "private"): {"bind": module.ASSETS.MOUNTS["private-performance"], "mode": "ro"}}
    assert created["device_requests"][0]["DeviceIDs"] == [JUDGE_GPU]
    assert module._roles[handle.id] == "judge"


def test_judge_refuses_sforge_mounts_and_a_missing_allocation(entry):
    from sforge.harness.backend import AcceleratorAllocation
    name = f"minference_sparse_prefill.judge.{TASK_ID}.agent-1"
    with pytest.raises(RuntimeError, match="no SForge mounts"):
        backend(FakeClient()).create_container("img", name, mounts=[("/x", "/y", "ro")],
                                               accelerator_allocation=AcceleratorAllocation("nvidia", (JUDGE_GPU,)))
    with pytest.raises(RuntimeError, match="exact GPU allocation"):
        backend(FakeClient()).create_container("img", name)


def test_work_gets_internal_network_its_gpu_and_the_public_trees(entry, monkeypatch):
    module, tmp_path = entry
    monkeypatch.setenv("SFORGE_ISOLATED_DOCKER_NETWORK", "run-internal")
    client = FakeClient()
    handle = backend(client).create_container(
        "minference_sparse_prefill.work.x:local", f"sforge.run.{TASK_ID}.run-1",
        environment={"NVIDIA_VISIBLE_DEVICES": WORK_GPU}, mounts=[("/runtime", "/opt/agent-runtime", "ro")])
    created = client.created[0]
    assert created["network"] == "run-internal"
    assert created["volumes"] == {
        "/runtime": {"bind": "/opt/agent-runtime", "mode": "ro"},
        str(tmp_path / "work_model"): {"bind": module.WORK_MODEL_MOUNT, "mode": "ro"},
        str(tmp_path / "public"): {"bind": module.WORK_PROMPTS_MOUNT, "mode": "ro"}}
    assert created["device_requests"][0]["DeviceIDs"] == [WORK_GPU]
    assert module._roles[handle.id] == "work"
    monkeypatch.setenv("SFORGE_ISOLATED_DOCKER_NETWORK", "")
    with pytest.raises(RuntimeError, match="is required"):
        backend(FakeClient()).create_container("img", f"sforge.run.{TASK_ID}.run-2",
                                               environment={"NVIDIA_VISIBLE_DEVICES": WORK_GPU})


def test_other_containers_keep_the_stock_path(entry):
    client = FakeClient()
    backend(client).create_container("img", "sforge.review.something")
    assert "volumes" not in client.created[0] and "network" not in client.created[0]


def test_judge_submission_gets_the_judge_gpu_and_a_final_round(entry, monkeypatch):
    module, _ = entry
    calls = []
    monkeypatch.setattr(module, "_stock_judge_submission", lambda *a, **k: calls.append(k))
    task = SimpleNamespace(task_id=TASK_ID)
    module.local_judge_submission(task_spec=task, round_id="agent-1")
    assert calls[-1]["accelerator_allocation"].device_ids == (JUDGE_GPU,)
    assert calls[-1]["round_id"] == "agent-1" and "evaluation_phase" not in calls[-1]
    monkeypatch.setenv("MINFERENCE_FINAL_EVAL", "1")
    module.local_judge_submission(task_spec=task, round_id="manual-1")
    assert calls[-1]["evaluation_phase"] == "final" and calls[-1]["round_id"] == "final-1"
    module.local_judge_submission(task_spec=SimpleNamespace(task_id="other"), round_id="manual-1")
    assert calls[-1] == {"task_spec": SimpleNamespace(task_id="other"), "round_id": "manual-1"}
