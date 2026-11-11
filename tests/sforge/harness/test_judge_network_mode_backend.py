from __future__ import annotations

from types import SimpleNamespace

import pytest

from sforge.harness.backend.docker_backend import DockerBackend
from sforge.harness.backend.k8s_backend import K8sBackend


class _CreatedContainers:
    def __init__(self) -> None:
        self.args: tuple[object, ...] = ()
        self.kwargs: dict[str, object] = {}

    def create(self, *args, **kwargs):
        self.args = args
        self.kwargs = dict(kwargs)
        return SimpleNamespace(id="container-id", name="judge")


def test_docker_network_none_is_an_actual_container_create_kwarg() -> None:
    containers = _CreatedContainers()
    backend = DockerBackend(client=SimpleNamespace(containers=containers))

    backend.create_container(
        "judge:immutable",
        "judge-network-none",
        network_mode="none",
    )

    assert containers.args == ("judge:immutable",)
    assert containers.kwargs["network_mode"] == "none"
    assert "extra_hosts" not in containers.kwargs


def test_docker_network_none_rejects_a_host_gateway_mapping() -> None:
    backend = DockerBackend(
        client=SimpleNamespace(containers=_CreatedContainers())
    )

    with pytest.raises(ValueError, match="extra hosts|host gateway|network"):
        backend.create_container(
            "judge:immutable",
            "judge-network-conflict",
            network_mode="none",
            extra_hosts={"host.docker.internal": "host-gateway"},
        )


def test_k8s_backend_fails_closed_for_unsupported_network_none() -> None:
    backend = object.__new__(K8sBackend)

    with pytest.raises(NotImplementedError, match="network.*none|none.*network"):
        backend.create_container(
            "judge:immutable",
            "judge-network-none",
            network_mode="none",
        )
