from __future__ import annotations

from types import SimpleNamespace

import pytest

from sforge.harness.backend import (
    EVALUATION_CONTRACT_SHA256_ANNOTATION,
    HANDOFF_ALLOCATION_KEY_ANNOTATION,
    HANDOFF_DOCKER_PAUSE_PROTOCOL,
    HANDOFF_LEASE_ID_ANNOTATION,
    HANDOFF_PROTOCOL_ANNOTATION,
    HANDOFF_RUN_ID_ANNOTATION,
)
from sforge.harness.backend.docker_backend import DockerBackend


class _CreatedContainers:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}

    def create(self, image: str, **kwargs):
        self.kwargs = dict(kwargs)
        return SimpleNamespace(id="container-id", name=kwargs["name"])


def test_docker_maps_validated_annotations_to_labels() -> None:
    containers = _CreatedContainers()
    backend = DockerBackend(client=SimpleNamespace(containers=containers))

    backend.create_container(
        "image:tag",
        "work",
        annotations={
            "sforge.run-id": "run-123",
            "sforge/lease-nonce": "nonce:456",
        },
    )

    assert containers.kwargs["labels"] == {
        "sforge.run-id": "run-123",
        "sforge/lease-nonce": "nonce:456",
    }


def test_docker_omits_labels_without_annotations() -> None:
    containers = _CreatedContainers()
    backend = DockerBackend(client=SimpleNamespace(containers=containers))

    backend.create_container("image:tag", "work")

    assert "labels" not in containers.kwargs


def test_docker_validates_and_maps_complete_handoff_bindings() -> None:
    containers = _CreatedContainers()
    backend = DockerBackend(client=SimpleNamespace(containers=containers))
    bindings = {
        HANDOFF_RUN_ID_ANNOTATION: "run-123",
        HANDOFF_LEASE_ID_ANNOTATION: "lease-456",
        HANDOFF_ALLOCATION_KEY_ANNOTATION: "allocation-sha256:" + "b" * 64,
        HANDOFF_PROTOCOL_ANNOTATION: HANDOFF_DOCKER_PAUSE_PROTOCOL,
        EVALUATION_CONTRACT_SHA256_ANNOTATION: "sha256:" + "c" * 64,
    }

    backend.create_container(
        "image:tag",
        "work",
        annotations=bindings,
    )

    assert containers.kwargs["labels"] == bindings


def test_docker_rejects_partial_reserved_handoff_bindings() -> None:
    containers = _CreatedContainers()
    backend = DockerBackend(client=SimpleNamespace(containers=containers))

    with pytest.raises(ValueError, match="missing trusted handoff"):
        backend.create_container(
            "image:tag",
            "work",
            annotations={HANDOFF_RUN_ID_ANNOTATION: "run-123"},
        )

    assert containers.kwargs == {}


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        (
            HANDOFF_ALLOCATION_KEY_ANNOTATION,
            "allocation-789",
            "allocation key",
        ),
        (HANDOFF_PROTOCOL_ANNOTATION, "docker-stop-v1", "protocol"),
        (
            EVALUATION_CONTRACT_SHA256_ANNOTATION,
            "sha256:not-a-digest",
            "evaluation contract",
        ),
    ],
)
def test_docker_rejects_malformed_reserved_handoff_bindings(
    key: str,
    value: str,
    message: str,
) -> None:
    containers = _CreatedContainers()
    backend = DockerBackend(client=SimpleNamespace(containers=containers))
    bindings = {
        HANDOFF_RUN_ID_ANNOTATION: "run-123",
        HANDOFF_LEASE_ID_ANNOTATION: "lease-456",
        HANDOFF_ALLOCATION_KEY_ANNOTATION: "allocation-sha256:" + "b" * 64,
        HANDOFF_PROTOCOL_ANNOTATION: HANDOFF_DOCKER_PAUSE_PROTOCOL,
        EVALUATION_CONTRACT_SHA256_ANNOTATION: "sha256:" + "c" * 64,
    }
    bindings[key] = value

    with pytest.raises(ValueError, match=message):
        backend.create_container(
            "image:tag",
            "work",
            annotations=bindings,
        )

    assert containers.kwargs == {}


@pytest.mark.parametrize(
    "annotations",
    [
        {"bad key": "value"},
        {"safe.key": "line\nbreak"},
        {"safe.key": 123},
    ],
)
def test_docker_rejects_invalid_annotations(annotations: dict) -> None:
    containers = _CreatedContainers()
    backend = DockerBackend(client=SimpleNamespace(containers=containers))

    with pytest.raises(ValueError, match="annotation"):
        backend.create_container(
            "image:tag",
            "work",
            annotations=annotations,
        )

    assert containers.kwargs == {}
