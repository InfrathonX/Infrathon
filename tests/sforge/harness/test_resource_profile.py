"""One CPU/memory size per task for PyroMind Sandboxes and local Docker containers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import docker
import pytest

from sforge import cli
from sforge.harness.benchmark import load_benchmark
from sforge.harness.config import SForgeConfig, load_config
from sforge.harness.task_spec import load_all_tasks
from sforge.harness import resource_profile as rp
from sforge.pyromind.deployment import Deployment

REPO = Path(__file__).resolve().parents[3]
PAGED = "standalone_paged_gqa_decode_bf16_h100_scratch_v1"
QVG = "quant_videogen_cross_model_kv_codec_h100_scratch_v3"
FLUX = "flux_moe_ag_scatter_grouped_gemm_dual_h100_seed_v1"

pytestmark = pytest.mark.skipif(not (REPO / "deploy/pyromind").is_dir(),
                                reason="no deploy/pyromind in this checkout")


def _deployments():
    return sorted((REPO / "deploy/pyromind").glob("*/deployment.json"))


def _task(task_id):
    deployment = Deployment.load(rp._index(REPO.resolve())[task_id], REPO)
    tasks_dir = deployment.local_path("task_directory") / "tasks"
    return load_all_tasks(tasks_dir, load_benchmark(tasks_dir))[0]


def _sizes(task_id):
    deployment = Deployment.load(rp._index(REPO.resolve())[task_id], REPO)
    return {role: deployment.resources_for_role(name)
            for role, name in (("work", "worker"), ("judge", "judge"))}


@pytest.fixture(autouse=True)
def _no_operator_pinning(monkeypatch):
    for name in ("SFORGE_CPUSET", "SFORGE_WORK_CPUSET", "SFORGE_JUDGE_CPUSET",
                 "SFORGE_CPUSET_UNDERSIZE", "SFORGE_RESOURCE_PROFILE"):
        monkeypatch.delenv(name, raising=False)


def test_every_deployment_names_one_task_and_one_task_directory():
    index = rp._index(REPO.resolve())
    assert len(index) == len(_deployments())
    for task_id, path in index.items():
        deployment = Deployment.load(path, REPO)
        declared = {json.loads(item.read_text()).get("task_id") for item in
                    (deployment.local_path("task_directory") / "tasks").glob("*.json")}
        assert declared == {task_id} == {deployment.config["task_id"]}
    names = {Deployment.load(path, REPO).config["name"] for path in _deployments()}
    assert set(rp.LOCAL_EXEMPT) <= names


def test_docker_memory_is_the_same_byte_count():
    assert rp.docker_memory("16Gi") == "16g"
    assert docker.utils.parse_bytes(rp.docker_memory("16Gi")) == 16 * 1024 ** 3
    for value in ("16G", "1.5Gi", "16", "0Gi", 16):
        with pytest.raises(rp.ResourceProfileError):
            rp.docker_memory(value)


def test_profile_mirrors_the_deployment_for_every_role():
    profile = rp.profile_for_task(PAGED)
    sizes = _sizes(PAGED)
    assert (profile.work.cpu, profile.work.memory) == (sizes["work"]["cpu"],
                                                       rp.docker_memory(sizes["work"]["memory"]))
    assert (profile.judge.cpu, profile.judge.memory) == (sizes["judge"]["cpu"],
                                                         rp.docker_memory(sizes["judge"]["memory"]))
    assert (profile.review.cpu, profile.review.memory, profile.review.gpu) == (2, "4g", 0)
    assert profile.source == "deploy/pyromind/paged-gqa/deployment.json" and len(profile.sha256) == 64
    assert rp.profile_for_task("no_such_task") is None


def test_local_docker_takes_the_profile_over_task_script_flags():
    task = _task(PAGED)
    config = SForgeConfig(resource_profile="pyromind", work_cpu_limit=8, work_mem_limit="32g")
    sized, record = rp.apply_local_profile(config, task)
    sizes = _sizes(PAGED)
    assert sized.work_cpu_limit == sizes["work"]["cpu"]
    assert sized.judge_mem_limit == rp.docker_memory(sizes["judge"]["memory"])
    assert record["status"] == "applied" and record["deployment"] == "paged-gqa"
    assert record["roles"]["work"]["replaced"] == {"cpu": 8, "memory": "32g"}
    assert record["roles"]["work"]["replaced_origin"] == "config"
    assert record["roles"]["judge"]["replaced"] == {"cpu": task.judge.cpu_limit,
                                                    "memory": task.judge.mem_limit}
    assert record["roles"]["judge"]["replaced_origin"] == "task"
    assert "cpuset_undersize" not in record and sized.cpuset_undersize == "refuse"
    assert "paged-gqa" in rp.summary(record)


@pytest.mark.parametrize("config", [
    SForgeConfig(resource_profile="native", work_cpu_limit=8),
    SForgeConfig(resource_profile="pyromind", backend="k8s", work_cpu_limit=8),
])
def test_native_mode_and_other_backends_keep_their_limits(config):
    sized, record = rp.apply_local_profile(config, _task(PAGED))
    assert sized is config and record is None


def test_exempt_and_unprofiled_tasks_keep_their_limits():
    config = SForgeConfig(resource_profile="pyromind", judge_cpu_limit=16, judge_mem_limit="96g")
    sized, record = rp.apply_local_profile(config, _task(FLUX))
    assert sized is config
    assert record["status"] == "exempt" and "local_entry.py" in record["reason"]
    unprofiled = SimpleNamespace(task_id="miles_task", work=None, judge=None)
    assert rp.apply_local_profile(config, unprofiled) == (config, None)
    assert rp.review_limits(config, _task(FLUX)) is None


def test_review_containers_take_the_profile_review_size():
    config = SForgeConfig(resource_profile="pyromind")
    cpu, memory, record = rp.review_limits(config, _task(PAGED))
    assert (cpu, memory) == (2.0, "4g") and record["review"] == {"cpu": 2, "memory": "4g"}
    assert rp.review_sandbox_limits(config)(_task(PAGED)) == (2.0, "4g")
    assert rp.review_sandbox_limits(SForgeConfig())(_task(PAGED)) is None


def test_judge_registration_overlay_passes_through_when_not_applied():
    task = _task(PAGED)
    assert rp.judge_overlay(SForgeConfig(), task, 8, "32g") == (8, "32g", None, None)
    cpu, memory, undersize, record = rp.judge_overlay(
        SForgeConfig(resource_profile="pyromind"), task, 8, "32g")
    sizes = _sizes(PAGED)["judge"]
    assert (cpu, memory) == (sizes["cpu"], rp.docker_memory(sizes["memory"]))
    assert undersize is None and record["roles"]["judge"]["replaced"] == {"cpu": 8, "memory": "32g"}


@pytest.mark.parametrize("task_id, expected", [(PAGED, "allow"), (QVG, "refuse")])
def test_over_slice_quota_is_pinned_only_when_the_slice_holds_native_cpus(
        monkeypatch, task_id, expected):
    monkeypatch.setattr(rp, "_gpu_local_slice", lambda config, role, gpus: 4 * gpus)
    task = _task(task_id)
    profile = rp.profile_for_task(task_id)
    if task_id == PAGED:
        # Native 8 CPUs fit an 8-CPU slice even when the profile asks for more.
        monkeypatch.setattr(rp, "_gpu_local_slice", lambda config, role, gpus: 8 * gpus)
        profile = rp.TaskProfile(**{**profile.__dict__,
                                    "work": rp.RoleSize(16, "32g", 1), "judge": rp.RoleSize(16, "32g", 1)})
    decision = rp.undersize_decision(SForgeConfig(cpuset="auto"), task, profile, ("work", "judge"))
    assert decision["value"] == expected
    assert set(decision["roles"]) == {"work", "judge"}


def test_an_explicit_operator_undersize_choice_wins(monkeypatch):
    monkeypatch.setattr(rp, "_gpu_local_slice", lambda config, role, gpus: 8 * gpus)
    monkeypatch.setenv("SFORGE_CPUSET_UNDERSIZE", "refuse")
    task = _task(QVG)
    assert rp.undersize_decision(SForgeConfig(cpuset="auto"), task,
                                 rp.profile_for_task(QVG), ("work", "judge")) is None


def test_only_load_config_defaults_to_the_profile(monkeypatch):
    assert SForgeConfig().resource_profile == "native"
    assert load_config().resource_profile == "pyromind"
    monkeypatch.setenv("SFORGE_RESOURCE_PROFILE", "native")
    assert load_config().resource_profile == "native"
    with pytest.raises(ValueError, match="SFORGE_RESOURCE_PROFILE"):
        SForgeConfig(resource_profile="cloud")


def _run_args(task_id, **values):
    return SimpleNamespace(experiment=None, task=[task_id], run_id="parity-run", silent=True,
                           stagger=None, judge_url="http://127.0.0.1:1", agent="codex",
                           model="m", timeout=60, eval_interval=60, **values)


@pytest.mark.parametrize("mode", ["pyromind", "native"])
def test_sforge_run_records_and_uses_the_profile(monkeypatch, tmp_path, mode):
    task = _task(PAGED)
    config = SForgeConfig(log_dir=tmp_path, resource_profile=mode, work_cpu_limit=8,
                          work_mem_limit="32g", judge_cpu_limit=8, judge_mem_limit="32g")
    seen = []
    monkeypatch.setattr(cli, "_make_config", lambda args: config)
    monkeypatch.setattr(cli, "_resolve_tasks", lambda args, config: [task])
    monkeypatch.setattr(cli, "create_backend_from_config", lambda config: object())
    monkeypatch.setattr(cli, "_run_single_task",
                        lambda ts, args, task_config, *rest, **kwargs: seen.append(task_config))
    cli.cmd_run(_run_args(PAGED))
    recorded = json.loads((tmp_path / "runs/parity-run" / PAGED / "run_config.json").read_text())
    if mode == "native":
        assert (seen[0].work_cpu_limit, seen[0].judge_mem_limit) == (8, "32g")
        assert "resource_profile" not in recorded
        return
    sizes = _sizes(PAGED)
    assert seen[0].work_cpu_limit == sizes["work"]["cpu"]
    assert recorded["judge_mem_limit"] == rp.docker_memory(sizes["judge"]["memory"])
    assert recorded["resource_profile"]["roles"]["work"]["replaced"] == {"cpu": 8, "memory": "32g"}


def test_sforge_eval_judges_at_the_profile_size(monkeypatch, tmp_path):
    task = _task(PAGED)
    archive = tmp_path / "a.tar.gz"
    archive.write_bytes(b"archive")
    config = SForgeConfig(log_dir=tmp_path, resource_profile="pyromind",
                          judge_cpu_limit=8, judge_mem_limit="32g")
    seen = []
    monkeypatch.setattr(cli, "_make_config", lambda args: config)
    monkeypatch.setattr(cli, "_resolve_task", lambda args, config: task)
    monkeypatch.setattr(cli, "create_backend_from_config", lambda config: object())
    monkeypatch.setattr(cli, "judge_submission", lambda **kwargs: seen.append(kwargs["config"]))
    monkeypatch.setattr(cli, "_print_eval_report", lambda report, include_json: None)
    cli.cmd_eval(SimpleNamespace(archive=str(archive), run_id="ev", timeout=None, silent=True,
                                 json=False, standalone_final_handoff=False,
                                 handoff_source_run_id=None, final_round_id=None))
    sizes = _sizes(PAGED)["judge"]
    assert (seen[0].judge_cpu_limit, seen[0].judge_mem_limit) == (sizes["cpu"], rp.docker_memory(sizes["memory"]))
    record = json.loads((tmp_path / "runs/ev" / PAGED / "submissions/manual-1/resource_profile.json").read_text())
    assert record["roles"]["judge"]["applied"]["cpu"] == sizes["cpu"]


def test_judge_server_registration_applies_the_profile(tmp_path):
    from sforge.harness.judge_server import JudgeState

    task = _task(PAGED)
    state = JudgeState(SForgeConfig(log_dir=tmp_path, resource_profile="pyromind"))
    state.tasks[task.task_id] = task
    token = state.register_session(task.task_id, "srv-run", judge_cpu_limit=8, judge_mem_limit="32g")
    sizes = _sizes(PAGED)["judge"]
    expected = (sizes["cpu"], rp.docker_memory(sizes["memory"]))
    assert (state.tokens[token]["judge_cpu_limit"], state.tokens[token]["judge_mem_limit"]) == expected
    limits = state.run_resource_limits["srv-run"]
    assert (limits["judge_cpu_limit"], limits["judge_mem_limit"]) == expected
    assert (tmp_path / "runs/srv-run" / PAGED / "judge_resource_profile.json").is_file()
    native = JudgeState(SForgeConfig(log_dir=tmp_path / "native"))
    native.tasks[task.task_id] = task
    token = native.register_session(task.task_id, "native-run", judge_cpu_limit=8, judge_mem_limit="32g")
    assert (native.tokens[token]["judge_cpu_limit"], native.tokens[token]["judge_mem_limit"]) == (8, "32g")
