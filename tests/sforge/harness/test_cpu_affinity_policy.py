from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from sforge.harness import cpu_affinity
from sforge.harness.config import SForgeConfig, load_config
from sforge.harness.cpu_affinity import (
    ContainerCpuState,
    cpu_affinity_identity,
    parse_policy_value,
    pin_before_start,
    plan_cpu_binding,
    policy_for_role,
    policy_from_env,
)
from sforge.harness.host_topology import TopologyError


PINNING_ENV = (
    "SFORGE_CPUSET",
    "SFORGE_WORK_CPUSET",
    "SFORGE_JUDGE_CPUSET",
    "SFORGE_CPUSET_MEMS",
    "SFORGE_CPUSET_RESERVE",
    "SFORGE_CPUSET_UNDERSIZE",
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in (*PINNING_ENV, "SFORGE_BACKEND"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, ("off", None, None)),
        ("", ("off", None, None)),
        ("  ", ("off", None, None)),
        ("off", ("off", None, None)),
        ("auto", ("auto", None, None)),
        (" auto\n", ("auto", None, None)),
        ("8-11,40-43", ("explicit", frozenset({8, 9, 10, 11, 40, 41, 42, 43}), None)),
        ("8-11,40-43@0", ("explicit", frozenset({8, 9, 10, 11, 40, 41, 42, 43}), frozenset({0}))),
        ("16@0-1", ("explicit", frozenset({16}), frozenset({0, 1}))),
    ],
)
def test_policy_grammar(value, expected) -> None:
    assert parse_policy_value(value) == expected


@pytest.mark.parametrize(
    "value", ["foo", "AUTO", "Off", "8-", "@0", "8-11@", "8@0@1", "11-8", "8,,9", "8 9"]
)
def test_policy_grammar_rejects_garbage(value: str) -> None:
    with pytest.raises(ValueError, match="off, auto or CPULIST"):
        parse_policy_value(value, name="SFORGE_CPUSET")


def test_default_config_has_no_policy() -> None:
    config = SForgeConfig()

    assert policy_for_role("work", config) is None
    assert policy_for_role("judge", config) is None
    assert cpu_affinity_identity(config) == {}


def test_role_value_overrides_the_global_one() -> None:
    config = SForgeConfig(cpuset="auto", judge_cpuset="8-11,40-43@0")

    work = policy_for_role("work", config)
    judge = policy_for_role("judge", config)
    assert (work.mode, work.source_env, work.value) == ("auto", "SFORGE_CPUSET", "auto")
    assert (judge.mode, judge.source_env, judge.value) == (
        "explicit",
        "SFORGE_JUDGE_CPUSET",
        "8-11,40-43@0",
    )


def test_role_off_disables_only_that_role() -> None:
    config = SForgeConfig(cpuset="auto", judge_cpuset="off")

    assert policy_for_role("work", config) is not None
    assert policy_for_role("judge", config) is None
    assert cpu_affinity_identity(config) == {"work": "auto:gpu-local-l3-v1:mems=bind"}


def test_role_value_alone_enables_only_that_role() -> None:
    config = SForgeConfig(work_cpuset="auto")

    assert cpu_affinity_identity(config) == {"work": "auto:gpu-local-l3-v1:mems=bind"}


def test_explicit_value_is_canonicalized() -> None:
    policy = policy_for_role("judge", SForgeConfig(cpuset=" 43,40-42,8,9-11@0 "))

    assert policy.value == "8-11,40-43@0"
    assert policy.cpus == frozenset({8, 9, 10, 11, 40, 41, 42, 43})
    assert policy.nodes == frozenset({0})


def test_load_config_maps_every_pinning_variable(clean_env) -> None:
    clean_env.setenv("SFORGE_CPUSET", "auto")
    clean_env.setenv("SFORGE_WORK_CPUSET", "off")
    clean_env.setenv("SFORGE_JUDGE_CPUSET", "8-11,40-43")
    clean_env.setenv("SFORGE_CPUSET_MEMS", "none")
    clean_env.setenv("SFORGE_CPUSET_RESERVE", "0,32")
    clean_env.setenv("SFORGE_CPUSET_UNDERSIZE", "allow")

    config = load_config()

    assert (
        config.cpuset,
        config.work_cpuset,
        config.judge_cpuset,
        config.cpuset_mems_mode,
        config.cpuset_reserve,
        config.cpuset_undersize,
    ) == ("auto", "off", "8-11,40-43", "none", "0,32", "allow")
    assert cpu_affinity_identity(config) == {
        "judge": "explicit:cpus=8:nodes=local:mems=none:undersize=allow"
    }


def test_load_config_without_variables_leaves_pinning_off(clean_env) -> None:
    config = load_config()

    assert config.cpuset is None
    assert config.cpuset_mems_mode == "bind"
    assert config.cpuset_undersize == "refuse"
    assert cpu_affinity_identity(config) == {}


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("SFORGE_CPUSET", "yes", "SFORGE_CPUSET must be off, auto or CPULIST"),
        ("SFORGE_JUDGE_CPUSET", "8-", "SFORGE_JUDGE_CPUSET must be"),
        ("SFORGE_CPUSET_MEMS", "preferred", "SFORGE_CPUSET_MEMS must be one of bind/none"),
        ("SFORGE_CPUSET_UNDERSIZE", "shrink", "SFORGE_CPUSET_UNDERSIZE must be one of"),
        ("SFORGE_CPUSET_RESERVE", "0-", "SFORGE_CPUSET_RESERVE must be a CPULIST"),
    ],
)
def test_load_config_rejects_bad_values(clean_env, name, value, message) -> None:
    clean_env.setenv(name, value)

    with pytest.raises(ValueError, match=message):
        load_config()


def test_global_value_is_validated_even_when_both_roles_override_it() -> None:
    with pytest.raises(ValueError, match="SFORGE_CPUSET must be"):
        SForgeConfig(cpuset="bogus", work_cpuset="auto", judge_cpuset="auto")


def test_explicit_nodes_contradict_mems_none() -> None:
    with pytest.raises(ValueError, match="drop @NODELIST or use bind"):
        SForgeConfig(cpuset="8-11@0", cpuset_mems_mode="none")


def test_non_docker_backend_with_a_policy_fails_closed(clean_env) -> None:
    with pytest.raises(ValueError, match="requires the Docker backend, not 'k8s'"):
        SForgeConfig(backend="k8s", cpuset="auto")

    clean_env.setenv("SFORGE_BACKEND", "k8s")
    clean_env.setenv("SFORGE_JUDGE_CPUSET", "auto")
    with pytest.raises(ValueError, match="requires the Docker backend"):
        load_config()

    # An explicit off on the other role, or no policy at all, stays valid.
    assert SForgeConfig(backend="k8s", cpuset="off").backend == "k8s"
    assert SForgeConfig(backend="k8s").cpuset is None


def test_dataclass_replace_revalidates() -> None:
    config = SForgeConfig(cpuset="auto")

    with pytest.raises(ValueError):
        replace(config, cpuset_mems_mode="interleave")
    assert replace(config, judge_cpu_limit=8).cpuset == "auto"


@pytest.mark.parametrize(
    ("config", "role", "expected"),
    [
        (SForgeConfig(cpuset="auto"), "judge", "auto:gpu-local-l3-v1:mems=bind"),
        (
            SForgeConfig(cpuset="auto", cpuset_reserve="0,16,32,48", cpuset_mems_mode="none"),
            "work",
            "auto:gpu-local-l3-v1:mems=none:reserve=4",
        ),
        (
            SForgeConfig(cpuset="auto", cpuset_undersize="allow"),
            "judge",
            "auto:gpu-local-l3-v1:mems=bind:undersize=allow",
        ),
        (
            SForgeConfig(judge_cpuset="24-27,56-59@1"),
            "judge",
            "explicit:cpus=8:nodes=1:mems=bind",
        ),
        (
            SForgeConfig(cpuset="8-11,40-43", cpuset_reserve="0"),
            "work",
            "explicit:cpus=8:nodes=local:mems=bind",
        ),
    ],
)
def test_policy_id_is_normalized_without_cpu_ids(config, role, expected) -> None:
    policy = policy_for_role(role, config)

    assert policy.policy_id == expected
    # Counts only: no CPU list, range or node list survives normalization.
    assert not re.search(r"\d[-,@]\d", policy.policy_id)


def test_replays_on_different_gpus_share_one_identity() -> None:
    first = policy_for_role("judge", SForgeConfig(judge_cpuset="8-11,40-43@0"))
    second = policy_for_role("judge", SForgeConfig(judge_cpuset="24-27,56-59@1"))

    assert first.policy_id == second.policy_id


def test_policy_from_env_matches_config_resolution(clean_env) -> None:
    clean_env.setenv("SFORGE_CPUSET", "auto")
    clean_env.setenv("SFORGE_WORK_CPUSET", "0-3,32-35@0")

    assert policy_from_env("judge").policy_id == "auto:gpu-local-l3-v1:mems=bind"
    assert policy_from_env("work") == policy_for_role(
        "work", SForgeConfig(cpuset="auto", work_cpuset="0-3,32-35@0")
    )
    assert policy_from_env(None).source_env == "SFORGE_CPUSET"
    assert policy_from_env("judge", {"SFORGE_CPUSET": "off"}) is None
    with pytest.raises(ValueError, match="unknown CPU pinning role"):
        policy_from_env("game")


# ---------------------------------------------------------------------------
# Explicit values against a (fake) host topology
# ---------------------------------------------------------------------------


def _explicit(value: str, host, **overrides):
    config = SForgeConfig(judge_cpuset=value, **overrides)
    return policy_for_role(
        "judge", config, sysfs_root=host.sysfs, procfs_root=host.procfs
    )


def test_explicit_value_binds_memory_to_the_cpus_nodes(fake_host) -> None:
    host = fake_host()

    plan = plan_cpu_binding(_explicit("8-11,40-43", host), ContainerCpuState(nano_cpus=8_000_000_000))

    assert (plan.outcome, plan.cpuset_cpus, plan.cpuset_mems) == ("pinned", "8-11,40-43", "0")
    assert plan.algorithm is None and plan.topology_fingerprint is None


def test_explicit_value_crossing_nodes_is_refused_in_bind_mode(fake_host) -> None:
    host = fake_host()

    with pytest.raises(TopologyError, match="remote to the bound memory nodes 0"):
        plan_cpu_binding(_explicit("8,16@0", host), ContainerCpuState())

    # Without @ the memory follows both CPUs' nodes; with mems=none nothing binds.
    plan = plan_cpu_binding(_explicit("8,16", host), ContainerCpuState())
    assert (plan.cpuset_mems, plan.cross_node) == ("0-1", True)
    plan = plan_cpu_binding(
        _explicit("8,16", host, cpuset_mems_mode="none"), ContainerCpuState()
    )
    assert plan.cpuset_mems is None


def test_explicit_value_must_name_online_cpus_and_existing_nodes(fake_host) -> None:
    host = fake_host()

    with pytest.raises(TopologyError, match="not online"):
        plan_cpu_binding(_explicit("60-70", host), ContainerCpuState())
    with pytest.raises(TopologyError, match="do not exist"):
        plan_cpu_binding(_explicit("8@5", host), ContainerCpuState())


def test_explicit_value_pins_a_gpu_less_container(fake_host) -> None:
    host = fake_host()

    plan = plan_cpu_binding(
        _explicit("20-23@1", host),
        ContainerCpuState(visible_devices="void", nano_cpus=4_000_000_000),
    )

    assert (plan.outcome, plan.cpuset_cpus, plan.cpuset_mems) == ("pinned", "20-23", "1")


# ---------------------------------------------------------------------------
# pin_before_start
# ---------------------------------------------------------------------------


class _Untouchable:
    """Any attribute access fails: proves the off path never looks."""

    def __getattribute__(self, name: str):
        raise AssertionError(f"backend touched while pinning is off: {name}")


def test_pin_before_start_is_a_no_op_when_off(tmp_path: Path) -> None:
    record = tmp_path / "cpu_affinity.json"

    result = pin_before_start(
        _Untouchable(),
        _Untouchable(),
        role="judge",
        config=SForgeConfig(cpuset="auto", judge_cpuset="off"),
        record_path=record,
    )

    assert result is None
    assert not record.exists()


def test_pin_before_start_fails_closed_without_backend_support(tmp_path: Path) -> None:
    backend = SimpleNamespace(backend_name="fake")

    with pytest.raises(NotImplementedError, match="fake backend cannot apply CPU pinning"):
        pin_before_start(
            backend,
            SimpleNamespace(id="c", name="n"),
            role="work",
            config=SForgeConfig(cpuset="auto"),
            record_path=tmp_path / "cpu_affinity_work.json",
        )
    assert not (tmp_path / "cpu_affinity_work.json").exists()


def test_pin_before_start_writes_the_record(tmp_path: Path, caplog) -> None:
    calls: list[object] = []

    class _Backend:
        backend_name = "docker"

        def apply_cpu_binding(self, handle, policy):
            calls.append((handle.id, policy.role, policy.value))
            return {
                "outcome": "pinned",
                "cpuset_cpus": "8-11,40-43",
                "cpuset_mems": "0",
                "cpu_budget": 8,
            }

    record_path = tmp_path / "nested" / "cpu_affinity.json"
    logger = cpu_affinity.logging.getLogger("test.cpu_affinity")
    with caplog.at_level("INFO", logger="test.cpu_affinity"):
        record = pin_before_start(
            _Backend(),
            SimpleNamespace(id="abc", name="bench.judge.task.1"),
            role="judge",
            config=SForgeConfig(cpuset="auto"),
            record_path=record_path,
            logger=logger,
        )

    assert calls == [("abc", "judge", "auto")]
    persisted = json.loads(record_path.read_text(encoding="utf-8"))
    assert persisted == record
    assert persisted["schema_version"] == 1
    assert persisted["role"] == "judge"
    assert persisted["container_id"] == "abc"
    assert persisted["container_name"] == "bench.judge.task.1"
    assert persisted["policy_id"] == "auto:gpu-local-l3-v1:mems=bind"
    assert persisted["policy"] == {
        "value": "auto",
        "source_env": "SFORGE_CPUSET",
        "mems_mode": "bind",
        "reserve": "",
        "undersize": "refuse",
    }
    assert persisted["cpuset_cpus"] == "8-11,40-43"
    assert "CPU pinning (judge): pinned cpus=8-11,40-43 mems=0" in caplog.text
    assert not list(record_path.parent.glob(".*.tmp"))


def test_pin_before_start_propagates_backend_failures(tmp_path: Path) -> None:
    class _Backend:
        backend_name = "docker"

        def apply_cpu_binding(self, handle, policy):
            raise RuntimeError("update rejected")

    with pytest.raises(RuntimeError, match="update rejected"):
        pin_before_start(
            _Backend(),
            SimpleNamespace(id="abc", name="n"),
            role="work",
            config=SForgeConfig(work_cpuset="auto"),
            record_path=tmp_path / "cpu_affinity_work.json",
        )
    assert not (tmp_path / "cpu_affinity_work.json").exists()


# ---------------------------------------------------------------------------
# Work call site (run_agent)
# ---------------------------------------------------------------------------


def _work_task():
    from sforge.harness.task_spec import JudgeSpec, TaskSpec, WorkSpec

    return TaskSpec(
        task_id="pin-work",
        name="pin work",
        base_image="base",
        platform="linux/amd64",
        cwd="/workspace",
        submit_paths=["."],
        submit_exclude=[],
        work=WorkSpec(specs_dir="/tmp/specs", agent_query="", image_tag="prebuilt"),
        judge=JudgeSpec(
            eval_cmd="true", eval_timeout=60, parser="structured_json", image_tag="prebuilt"
        ),
        benchmark_name="test",
    )


class _WorkBackend:
    backend_name = "fake"

    def __init__(self) -> None:
        self.events: list[object] = []
        self.create_kwargs: dict[str, object] = {}

    def image_exists(self, image_key: str) -> bool:
        return True

    def remove_container_by_name(self, name: str) -> None:
        pass

    def create_container(self, image, name, **kwargs):
        self.events.append("create")
        self.create_kwargs = dict(kwargs)
        return SimpleNamespace(id="work-id", name=name)

    def start_container(self, handle) -> None:
        self.events.append("start")
        raise RuntimeError("stop the run after start")

    def cleanup_container(self, handle, logger=None) -> None:
        self.events.append(("cleanup", handle.id))


def _run_work(backend, config, monkeypatch):
    import requests

    import sforge.harness.run_agent as run_agent_module
    from sforge.harness.agent.factory import create_agent

    class _Registration:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, object]:
            return {"token": "session-token"}

    def no_history(*args, **kwargs):
        raise requests.RequestException("no judge in this test")

    monkeypatch.setattr(run_agent_module.requests, "post", lambda *a, **k: _Registration())
    monkeypatch.setattr(run_agent_module.requests, "get", no_history)
    return run_agent_module.run_agent(
        _work_task(),
        create_agent("claude-code", config),
        config,
        backend,
        run_id="run-1",
    )


def test_work_pin_failure_never_starts_and_removes_the_container(
    tmp_path: Path, monkeypatch
) -> None:
    class Backend(_WorkBackend):
        def apply_cpu_binding(self, handle, policy):
            self.events.append(("pin", policy.role, policy.value))
            raise TopologyError("GPU reports no NUMA node")

    backend = Backend()
    config = SForgeConfig(log_dir=tmp_path, work_cpuset="auto", judge_cpuset="off")

    result = _run_work(backend, config, monkeypatch)

    assert backend.events == ["create", ("pin", "work", "auto"), ("cleanup", "work-id")]
    assert "GPU reports no NUMA node" in result.agent_output
    assert not (tmp_path / "runs/run-1/pin-work/cpu_affinity_work.json").exists()


def test_work_pin_is_recorded_before_start(tmp_path: Path, monkeypatch) -> None:
    class Backend(_WorkBackend):
        def apply_cpu_binding(self, handle, policy):
            self.events.append(("pin", policy.role, policy.value))
            return {"outcome": "pinned", "cpuset_cpus": "0-3,32-35", "cpuset_mems": "0"}

    backend = Backend()
    config = SForgeConfig(log_dir=tmp_path, cpuset="auto")

    _run_work(backend, config, monkeypatch)

    assert backend.events == [
        "create", ("pin", "work", "auto"), "start", ("cleanup", "work-id"),
    ]
    record = json.loads(
        (tmp_path / "runs/run-1/pin-work/cpu_affinity_work.json").read_text("utf-8")
    )
    assert (record["role"], record["container_id"], record["cpuset_cpus"]) == (
        "work", "work-id", "0-3,32-35",
    )


def test_work_creation_is_unchanged_when_pinning_is_off(
    tmp_path: Path, monkeypatch
) -> None:
    backend = _WorkBackend()   # no apply_cpu_binding at all

    _run_work(backend, SForgeConfig(log_dir=tmp_path), monkeypatch)

    assert backend.events == ["create", "start", ("cleanup", "work-id")]
    assert set(backend.create_kwargs) == {
        "environment", "extra_hosts", "cap_drop", "cpu_limit", "mem_limit",
    }
    assert not (tmp_path / "runs/run-1/pin-work/cpu_affinity_work.json").exists()
