from __future__ import annotations

from types import SimpleNamespace

import docker.types
import pytest

from sforge.harness.backend import AcceleratorAllocation
from sforge.harness.backend.docker_backend import DockerBackend
from sforge.harness.backend.k8s_backend import K8sBackend


class _CreatedContainers:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}

    def create(self, image: str, **kwargs):
        self.kwargs = dict(kwargs)
        return SimpleNamespace(id="container-id", name=kwargs["name"])


def test_allocation_normalizes_vendor_and_freezes_device_ids() -> None:
    gpu_id = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    allocation = AcceleratorAllocation(" NVIDIA ", [gpu_id])

    assert allocation.vendor == "nvidia"
    assert allocation.device_ids == (gpu_id,)


@pytest.mark.parametrize(
    "device_id",
    [
        "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "MIG-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "MIG-GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee/1/2",
    ],
)
def test_allocation_accepts_canonical_nvidia_uuid_forms(device_id: str) -> None:
    assert AcceleratorAllocation("nvidia", (device_id,)).device_ids == (
        device_id,
    )


@pytest.mark.parametrize(
    ("vendor", "device_ids", "message"),
    [
        ("", ("0",), "vendor"),
        ("nvidia", (), "contain a device ID"),
        (
            "nvidia",
            (
                "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            ),
            "duplicate",
        ),
        ("nvidia", ("",), "unsafe"),
        ("nvidia", ("GPU-ok\n--privileged",), "unsafe"),
        ("nvidia", ("../device",), "unsafe"),
        ("nvidia", ("all",), "not an index or alias"),
        ("nvidia", ("ALL",), "not an index or alias"),
        ("nvidia", ("none",), "not an index or alias"),
        ("nvidia", ("None",), "not an index or alias"),
        ("nvidia", ("void",), "not an index or alias"),
        ("nvidia", ("VOID",), "not an index or alias"),
        ("nvidia", ("gpu",), "not an index or alias"),
        ("nvidia", ("0",), "not an index or alias"),
        ("nvidia", ("00",), "not an index or alias"),
        ("nvidia", ("GPU-not-a-uuid",), "not an index or alias"),
        (
            "nvidia",
            ("GPU-AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE",),
            "canonical GPU UUID",
        ),
    ],
)
def test_allocation_rejects_invalid_values(
    vendor: str,
    device_ids: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        AcceleratorAllocation(vendor, device_ids)


def test_docker_omits_device_requests_without_an_allocation() -> None:
    containers = _CreatedContainers()
    backend = DockerBackend(client=SimpleNamespace(containers=containers))

    backend.create_container("image:tag", "work")

    assert "device_requests" not in containers.kwargs


def test_docker_passes_exact_nvidia_ids_without_a_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    containers = _CreatedContainers()
    backend = DockerBackend(client=SimpleNamespace(containers=containers))
    request_kwargs: dict[str, object] = {}
    request_sentinel = object()

    def device_request(**kwargs):
        request_kwargs.update(kwargs)
        return request_sentinel

    monkeypatch.setattr(docker.types, "DeviceRequest", device_request)

    backend.create_container(
        "image:tag",
        "work",
        accelerator_allocation=AcceleratorAllocation(
            "nvidia",
            ("GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",),
        ),
    )

    assert request_kwargs == {
        "driver": "nvidia",
        "device_ids": ["GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"],
        "capabilities": [["gpu"]],
    }
    assert containers.kwargs["device_requests"] == [request_sentinel]


def test_docker_fails_closed_for_an_unsupported_vendor() -> None:
    backend = DockerBackend(
        client=SimpleNamespace(containers=_CreatedContainers())
    )

    with pytest.raises(ValueError, match="only NVIDIA.*amd"):
        backend.create_container(
            "image:tag",
            "work",
            accelerator_allocation=AcceleratorAllocation("amd", ("0",)),
        )


def test_k8s_rejects_physical_device_pinning_with_logical_guidance() -> None:
    backend = object.__new__(K8sBackend)

    with pytest.raises(
        NotImplementedError,
        match="physical accelerator device pinning.*logical accelerator resource",
    ):
        backend.create_container(
            "image:tag",
            "work",
            accelerator_allocation=AcceleratorAllocation(
                "nvidia",
                ("GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",),
            ),
        )
