from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from sforge.harness import host_topology as ht
from sforge.harness.backend.docker_backend import DockerBackend, DockerContainerHandle
from sforge.harness.backend.k8s_backend import K8sBackend
from sforge.harness.config import SForgeConfig
from sforge.harness.cpu_affinity import pin_before_start, policy_for_role
from sforge.harness.host_topology import InventoryGpu, TopologyError


H100_FINGERPRINT = (
    "sha256:3a51f52c156b199240f55ffd06408eedaf5f55c6c6789f180120acc4eb568bc8"
)
GIB = 1_000_000_000


def _uuid(index: int) -> str:
    return f"GPU-{index:08x}-1111-2222-3333-{index:012x}"


def _gpu_request(*device_ids: str, count: int = 0) -> dict[str, object]:
    return {
        "Driver": "nvidia",
        "Count": count,
        "DeviceIDs": list(device_ids) or None,
        "Capabilities": [["gpu"]],
        "Options": {},
    }


def _cdi_request(*device_ids: str) -> dict[str, object]:
    return {
        "Driver": "cdi",
        "Count": 0,
        "DeviceIDs": list(device_ids),
        "Capabilities": None,
        "Options": {},
    }


class _FakeContainer:
    """docker-py Container stand-in: attrs, reload() and update()."""

    def __init__(
        self,
        *,
        device_requests=None,
        env=(),
        nano_cpus: int = 0,
        cpu_quota: int = 0,
        cpu_period: int = 0,
        cpuset_cpus: str = "",
        cpuset_mems: str = "",
        status: str = "created",
        pid: int = 0,
        persist: bool = True,
        nano_cpus_after_update: int | None = None,
    ) -> None:
        self.id = "c" * 64
        self.name = "bench.judge.task.submission"
        self.status = status
        self.attrs = {
            "State": {"Status": status, "Running": status == "running", "Pid": pid},
            "HostConfig": {
                "DeviceRequests": device_requests,
                "NanoCpus": nano_cpus,
                "CpuQuota": cpu_quota,
                "CpuPeriod": cpu_period,
                "CpusetCpus": cpuset_cpus,
                "CpusetMems": cpuset_mems,
            },
            "Config": {"Env": list(env)},
        }
        self.persist = persist
        self.nano_cpus_after_update = nano_cpus_after_update
        self.updates: list[dict[str, str]] = []
        self.reloads = 0
        self.started = False

    def reload(self) -> None:
        self.reloads += 1

    def update(self, **kwargs) -> dict[str, object]:
        self.updates.append(dict(kwargs))
        host_config = self.attrs["HostConfig"]
        if self.persist:
            if "cpuset_cpus" in kwargs:
                host_config["CpusetCpus"] = kwargs["cpuset_cpus"]
            if "cpuset_mems" in kwargs:
                host_config["CpusetMems"] = kwargs["cpuset_mems"]
        if self.nano_cpus_after_update is not None:
            host_config["NanoCpus"] = self.nano_cpus_after_update
        return {"Warnings": []}

    def start(self) -> None:
        self.started = True


def _backend() -> DockerBackend:
    return DockerBackend(client=SimpleNamespace())


class _FakeHostDockerBackend(DockerBackend):
    """The real Docker pin step, reading the fake host instead of /sys."""

    def __init__(self, host) -> None:
        super().__init__(client=SimpleNamespace())
        self._host = host

    def apply_cpu_binding(self, handle, policy):
        return super().apply_cpu_binding(
            handle,
            replace(policy, sysfs_root=self._host.sysfs, procfs_root=self._host.procfs),
        )


def _policy(host, role: str = "judge", **config):
    config.setdefault("cpuset", "auto")
    return policy_for_role(
        role, SForgeConfig(**config), sysfs_root=host.sysfs, procfs_root=host.procfs
    )


def _apply(container: _FakeContainer, policy) -> dict[str, object]:
    return _backend().apply_cpu_binding(DockerContainerHandle(container), policy)


def test_device_request_gpu_is_pinned_to_its_slice(fake_host) -> None:
    host = fake_host()
    container = _FakeContainer(
        device_requests=[_gpu_request(_uuid(2))],
        env=[f"NVIDIA_VISIBLE_DEVICES={_uuid(2)}", "PATH=/usr/bin"],
        nano_cpus=8 * GIB,
    )

    record = _apply(container, _policy(host))

    assert container.updates == [{"cpuset_cpus": "8-11,40-43", "cpuset_mems": "0"}]
    assert container.reloads >= 2
    assert container.started is False
    assert record == {
        "outcome": "pinned",
        "gpu_source": "device_requests",
        "gpus": [{"id": _uuid(2), "pci": "0000:4e:00.0", "numa_node": 0}],
        "slice_cpus": "8-11,40-43",
        "cpuset_cpus": "8-11,40-43",
        "cpuset_mems": "0",
        "cpu_count": 8,
        "nano_cpus": 8 * GIB,
        "cpu_budget": 8,
        "trimmed_to_quota": False,
        "undersized": False,
        "cross_node": False,
        "algorithm": "gpu-local-l3-v1",
        "topology_fingerprint": H100_FINGERPRINT,
    }


def test_environment_only_gpu_is_pinned(fake_host) -> None:
    host = fake_host()
    container = _FakeContainer(
        env=[f"NVIDIA_VISIBLE_DEVICES={_uuid(6)}"], nano_cpus=8 * GIB
    )

    record = _apply(container, _policy(host))

    assert container.updates == [{"cpuset_cpus": "24-27,56-59", "cpuset_mems": "1"}]
    assert record["gpu_source"] == "nvidia_visible_devices"


def test_nvml_index_in_the_environment_is_resolved(fake_host, monkeypatch) -> None:
    host = fake_host()
    monkeypatch.setattr(
        ht,
        "nvml_inventory",
        lambda timeout=0: (InventoryGpu(3, _uuid(3), "0000:4f:00.0"),),
    )
    container = _FakeContainer(env=["NVIDIA_VISIBLE_DEVICES=3"], nano_cpus=8 * GIB)

    record = _apply(container, _policy(host))

    assert record["cpuset_cpus"] == "12-15,44-47"


def test_dual_gpu_container_gets_the_pair_union(fake_host) -> None:
    host = fake_host()
    container = _FakeContainer(
        device_requests=[_gpu_request(_uuid(4), _uuid(5))],
        env=[f"NVIDIA_VISIBLE_DEVICES={_uuid(5)},{_uuid(4)}", "CUDA_VISIBLE_DEVICES=0,1"],
        nano_cpus=16 * GIB,
    )

    record = _apply(container, _policy(host))

    assert container.updates == [{"cpuset_cpus": "16-23,48-55", "cpuset_mems": "1"}]
    assert [gpu["pci"] for gpu in record["gpus"]] == ["0000:c5:00.0", "0000:c6:00.0"]
    assert record["cpu_count"] == 16


def test_image_default_visible_devices_do_not_override_the_request(fake_host) -> None:
    host = fake_host()
    container = _FakeContainer(
        device_requests=[_gpu_request(_uuid(0))],
        env=["NVIDIA_VISIBLE_DEVICES=all"],
        nano_cpus=8 * GIB,
    )

    assert _apply(container, _policy(host))["cpuset_cpus"] == "0-3,32-35"


def test_request_and_environment_naming_different_gpus_is_refused(fake_host) -> None:
    host = fake_host()
    container = _FakeContainer(
        device_requests=[_gpu_request(_uuid(0))],
        env=[f"NVIDIA_VISIBLE_DEVICES={_uuid(1)}"],
        nano_cpus=8 * GIB,
    )

    with pytest.raises(TopologyError, match="disagree with NVIDIA_VISIBLE_DEVICES"):
        _apply(container, _policy(host))
    assert container.updates == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"device_requests": [_gpu_request(count=-1)]}, "by count or 'all'"),
        ({"env": ["NVIDIA_VISIBLE_DEVICES=all"]}, "NVIDIA_VISIBLE_DEVICES=all"),
        (
            {"env": ["NVIDIA_VISIBLE_DEVICES=MIG-00000001-1111-2222-3333-000000000001"]},
            "MIG device",
        ),
        ({"device_requests": [_cdi_request("nvidia.com/gpu=2")]}, "cannot map to NVIDIA GPUs"),
        (
            {"device_requests": [{"Driver": "other", "Count": 1, "Capabilities": [["compute"]]}]},
            "cannot map to NVIDIA GPUs",
        ),
    ],
)
def test_ambiguous_gpu_sets_are_refused_in_auto_mode(fake_host, kwargs, message) -> None:
    host = fake_host()
    container = _FakeContainer(nano_cpus=8 * GIB, **kwargs)

    with pytest.raises(TopologyError, match=message):
        _apply(container, _policy(host))
    assert container.updates == []


@pytest.mark.parametrize(
    "env", [["NVIDIA_VISIBLE_DEVICES=void"], ["NVIDIA_VISIBLE_DEVICES="], []]
)
def test_gpu_less_container_stays_unpinned_in_auto_mode(tmp_path: Path, env) -> None:
    # No sysfs at all: a GPU-less container never reads the topology.
    policy = policy_for_role(
        "work",
        SForgeConfig(cpuset="auto"),
        sysfs_root=tmp_path / "missing-sys",
        procfs_root=tmp_path / "missing-proc",
    )
    container = _FakeContainer(env=env, nano_cpus=16 * GIB)

    record = _apply(container, policy)

    assert container.updates == []
    assert record["outcome"] == "unpinned_no_gpu"
    assert record["cpuset_cpus"] is None


def test_unrecognized_device_requests_fail_closed_only_in_auto_mode(fake_host) -> None:
    host = fake_host()
    # A CDI request names a GPU in a form auto pinning cannot resolve; with no
    # NVIDIA_VISIBLE_DEVICES it must not be taken for a GPU-less container.
    cdi = [_cdi_request("nvidia.com/gpu=2")]
    auto = _FakeContainer(device_requests=cdi, nano_cpus=8 * GIB)
    with pytest.raises(TopologyError, match="explicit CPULIST@NODES"):
        _apply(auto, _policy(host))
    assert auto.updates == []

    # An explicit policy never classifies the GPUs, so it still pins.
    explicit = _FakeContainer(device_requests=cdi, nano_cpus=8 * GIB)
    assert _apply(explicit, _policy(host, cpuset="8-11,40-43@0"))["outcome"] == "pinned"
    assert explicit.updates == [{"cpuset_cpus": "8-11,40-43", "cpuset_mems": "0"}]

    # A request that names no device at all is not a GPU and is ignored.
    inert = _FakeContainer(
        device_requests=[{"Driver": "other", "Count": 0, "DeviceIDs": None}],
        nano_cpus=8 * GIB,
    )
    assert _apply(inert, _policy(host))["outcome"] == "unpinned_no_gpu"

    # quant-videogen's shape: no driver name, but the gpu capability.
    driverless = _FakeContainer(
        device_requests=[
            {"Driver": "", "Count": 0, "DeviceIDs": [_uuid(2)], "Capabilities": [["gpu"]]}
        ],
        nano_cpus=8 * GIB,
    )
    assert _apply(driverless, _policy(host))["cpuset_cpus"] == "8-11,40-43"


def test_task_owned_cpuset_is_left_alone_by_auto(fake_host) -> None:
    host = fake_host()
    container = _FakeContainer(
        device_requests=[_gpu_request(_uuid(4), _uuid(5))],
        nano_cpus=20 * GIB,
        cpuset_cpus="16-35",
    )

    record = _apply(container, _policy(host))

    assert container.updates == []
    assert (record["outcome"], record["cpuset_cpus"], record["cpu_count"]) == (
        "task_owned", "16-35", 20,
    )


def test_explicit_value_must_match_a_task_owned_cpuset(fake_host) -> None:
    host = fake_host()

    matching = _FakeContainer(cpuset_cpus="16-19,48-51")
    record = _apply(matching, _policy(host, cpuset="48-51,16-19"))
    assert (record["outcome"], matching.updates) == ("task_owned", [])

    conflicting = _FakeContainer(cpuset_cpus="16-19,48-51")
    with pytest.raises(TopologyError, match="conflicts with the task-owned cpuset"):
        _apply(conflicting, _policy(host, cpuset="8-11,40-43"))
    assert conflicting.updates == []


def test_preset_mems_without_cpus_is_refused(fake_host) -> None:
    host = fake_host()
    container = _FakeContainer(device_requests=[_gpu_request(_uuid(0))], cpuset_mems="1")

    with pytest.raises(TopologyError, match="presets CpusetMems=1"):
        _apply(container, _policy(host))


def test_slice_larger_than_the_budget_is_trimmed(fake_host) -> None:
    host = fake_host()
    container = _FakeContainer(device_requests=[_gpu_request(_uuid(2))], nano_cpus=4 * GIB)

    record = _apply(container, _policy(host))

    assert container.updates == [{"cpuset_cpus": "8-9,40-41", "cpuset_mems": "0"}]
    assert (record["slice_cpus"], record["cpu_count"], record["trimmed_to_quota"]) == (
        "8-11,40-43", 4, True,
    )


def test_larger_host_slice_is_trimmed_to_the_quota(fake_host) -> None:
    gpus = [("0000:17:00.0", 0), ("0000:3d:00.0", 0), ("0000:9b:00.0", 1), ("0000:bb:00.0", 1)]
    host = fake_host(nodes=(32, 32), l3_cores=8, gpus=gpus)
    container = _FakeContainer(env=[f"NVIDIA_VISIBLE_DEVICES={_uuid(1)}"], nano_cpus=8 * GIB)

    record = _apply(container, _policy(host))

    assert record["slice_cpus"] == "16-31,80-95"
    assert record["cpuset_cpus"] == "16-19,80-83"
    assert record["trimmed_to_quota"] is True


def test_cfs_quota_counts_as_the_budget(fake_host) -> None:
    host = fake_host()
    container = _FakeContainer(
        device_requests=[_gpu_request(_uuid(2))], cpu_quota=600_000, cpu_period=100_000
    )

    record = _apply(container, _policy(host))

    assert (record["cpu_budget"], record["cpuset_cpus"]) == (6, "8-10,40-42")


def test_no_quota_pins_the_whole_slice(fake_host) -> None:
    host = fake_host()
    container = _FakeContainer(device_requests=[_gpu_request(_uuid(2))])

    record = _apply(container, _policy(host))

    assert (record["cpu_budget"], record["cpuset_cpus"]) == (None, "8-11,40-43")


def test_budget_larger_than_the_slice_is_refused_by_default(fake_host) -> None:
    host = fake_host()
    container = _FakeContainer(device_requests=[_gpu_request(_uuid(2))], nano_cpus=16 * GIB)

    with pytest.raises(TopologyError, match="SFORGE_CPUSET_UNDERSIZE=allow"):
        _apply(container, _policy(host))
    assert container.updates == []


def test_undersize_allow_pins_the_smaller_slice_and_records_it(fake_host) -> None:
    host = fake_host()
    container = _FakeContainer(device_requests=[_gpu_request(_uuid(2))], nano_cpus=16 * GIB)

    record = _apply(container, _policy(host, cpuset_undersize="allow"))

    assert container.updates == [{"cpuset_cpus": "8-11,40-43", "cpuset_mems": "0"}]
    assert (record["undersized"], record["cpu_budget"], record["nano_cpus"]) == (
        True, 16, 16 * GIB,
    )


def test_mems_none_sets_cpus_only(fake_host) -> None:
    host = fake_host()
    container = _FakeContainer(device_requests=[_gpu_request(_uuid(2))], nano_cpus=8 * GIB)

    record = _apply(container, _policy(host, cpuset_mems_mode="none"))

    assert container.updates == [{"cpuset_cpus": "8-11,40-43"}]
    assert record["cpuset_mems"] is None


def test_reserve_shrinks_the_auto_slice(fake_host) -> None:
    host = fake_host()
    container = _FakeContainer(device_requests=[_gpu_request(_uuid(0))], nano_cpus=6 * GIB)

    record = _apply(container, _policy(host, cpuset_reserve="0,32"))

    assert record["cpuset_cpus"] == "1-3,33-35"


def test_explicit_value_is_pinned_as_given(fake_host) -> None:
    host = fake_host()
    container = _FakeContainer(
        device_requests=[_gpu_request(_uuid(0))], nano_cpus=8 * GIB
    )

    record = _apply(container, _policy(host, judge_cpuset="24-27,56-59@1"))

    assert container.updates == [{"cpuset_cpus": "24-27,56-59", "cpuset_mems": "1"}]
    assert record["gpus"] == [] and record["algorithm"] is None


@pytest.mark.parametrize("status", ["running", "exited", "paused"])
def test_only_a_created_container_can_be_pinned(fake_host, status: str) -> None:
    host = fake_host()
    container = _FakeContainer(
        device_requests=[_gpu_request(_uuid(2))],
        nano_cpus=8 * GIB,
        status=status,
        pid=4242 if status != "exited" else 0,
    )

    with pytest.raises(RuntimeError, match="created, not yet started"):
        _apply(container, _policy(host))
    assert container.updates == []


def test_exact_lookup_and_frozen_handles_are_refused(fake_host) -> None:
    host = fake_host()
    container = _FakeContainer(device_requests=[_gpu_request(_uuid(2))], nano_cpus=8 * GIB)

    looked_up = DockerContainerHandle(container)
    looked_up._resolved_by_exact_lookup = True
    with pytest.raises(RuntimeError, match="exact re-lookup cannot authorize CPU pinning"):
        _backend().apply_cpu_binding(looked_up, _policy(host))

    frozen = DockerContainerHandle(container)
    frozen._snapshot_freeze_metadata = {"mechanism": "docker_stop_timeout_0"}
    with pytest.raises(RuntimeError, match="frozen for final snapshot"):
        _backend().apply_cpu_binding(frozen, _policy(host))
    assert container.updates == []


def test_update_that_does_not_persist_fails_verification(fake_host) -> None:
    host = fake_host()
    container = _FakeContainer(
        device_requests=[_gpu_request(_uuid(2))], nano_cpus=8 * GIB, persist=False
    )

    with pytest.raises(RuntimeError, match="did not persist the requested CPU pinning"):
        _apply(container, _policy(host))


def test_quota_change_during_update_fails_verification(fake_host) -> None:
    host = fake_host()
    container = _FakeContainer(
        device_requests=[_gpu_request(_uuid(2))],
        nano_cpus=8 * GIB,
        nano_cpus_after_update=0,
    )

    with pytest.raises(RuntimeError, match="nano_cpus=0"):
        _apply(container, _policy(host))


def test_pin_before_start_writes_the_docker_record(fake_host, tmp_path: Path) -> None:
    host = fake_host()
    container = _FakeContainer(device_requests=[_gpu_request(_uuid(7))], nano_cpus=8 * GIB)
    record_path = tmp_path / "cpu_affinity.json"

    record = pin_before_start(
        _FakeHostDockerBackend(host),
        DockerContainerHandle(container),
        role="judge",
        config=SForgeConfig(cpuset="auto"),
        record_path=record_path,
    )

    persisted = json.loads(record_path.read_text(encoding="utf-8"))
    assert persisted == record
    assert set(persisted) == {
        "schema_version", "role", "container_id", "container_name", "policy",
        "policy_id", "outcome", "gpu_source", "gpus", "slice_cpus", "cpuset_cpus",
        "cpuset_mems", "cpu_count", "nano_cpus", "cpu_budget", "trimmed_to_quota",
        "undersized", "cross_node", "algorithm", "topology_fingerprint",
    }
    assert (persisted["outcome"], persisted["cpuset_cpus"], persisted["cpuset_mems"]) == (
        "pinned", "28-31,60-63", "1",
    )
    assert persisted["container_id"] == "c" * 64
    assert persisted["policy_id"] == "auto:gpu-local-l3-v1:mems=bind"


def test_non_docker_backends_fail_closed_by_default(fake_host) -> None:
    backend = object.__new__(K8sBackend)

    with pytest.raises(NotImplementedError, match="does not support CPU pinning"):
        backend.apply_cpu_binding(SimpleNamespace(id="pod"), _policy(fake_host()))


def test_docker_create_container_is_unchanged_by_the_feature() -> None:
    created: dict[str, object] = {}

    def create(image: str, **kwargs):
        created.update(kwargs)
        return SimpleNamespace(id="container-id", name=kwargs["name"])

    backend = DockerBackend(client=SimpleNamespace(containers=SimpleNamespace(create=create)))
    backend.create_container("image:tag", "judge", cpu_limit=8, mem_limit="32g")

    assert created == {
        "name": "judge",
        "detach": True,
        "command": "tail -f /dev/null",
        "environment": {},
        "nano_cpus": 8 * GIB,
        "mem_limit": "32g",
    }
