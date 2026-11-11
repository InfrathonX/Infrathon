from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from sforge import cli
from sforge.harness.backend.base import ResolvedImage
from sforge.harness.backend.docker_backend import DockerBackend
from sforge.harness.backend.k8s_backend import _sanitize_k8s_name


_DIGEST = "sha256:" + "a" * 64


def test_docker_resolver_pins_the_full_local_image_id() -> None:
    looked_up: list[str] = []

    class _Images:
        def get(self, image: str):
            looked_up.append(image)
            return SimpleNamespace(id=_DIGEST)

    backend = DockerBackend(client=SimpleNamespace(images=_Images()))

    resolved = backend.resolve_image_for_execution("judge:local")

    assert looked_up == ["judge:local"]
    assert resolved.backend == "docker"
    assert resolved.requested_ref == "judge:local"
    assert resolved.execution_ref == _DIGEST
    assert resolved.content_identity == _DIGEST
    assert resolved.identity_kind == "docker-image-id"


def test_replay_builds_before_resolving_the_execution_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class _Images:
        def get(self, image: str):
            events.append(f"inspect:{image}")
            return SimpleNamespace(id=_DIGEST)

    backend = DockerBackend(client=SimpleNamespace(images=_Images()))
    monkeypatch.setattr(
        cli,
        "build_judge_image",
        lambda *args, **kwargs: events.append("build"),
    )
    task_spec = SimpleNamespace(judge_image_key="judge:local")

    resolved = cli._resolve_replay_judge_image(
        backend,
        task_spec,
        SimpleNamespace(),
    )

    assert events == ["build", "inspect:judge:local"]
    assert resolved.execution_ref == _DIGEST


def test_docker_resume_verifies_the_pinned_id_without_resolving_the_tag() -> None:
    looked_up: list[str] = []

    class _Images:
        def get(self, image: str):
            looked_up.append(image)
            if image != _DIGEST:
                raise AssertionError("resume must not resolve the moved tag")
            return SimpleNamespace(id=_DIGEST)

    backend = DockerBackend(client=SimpleNamespace(images=_Images()))
    resolved = ResolvedImage(
        backend="docker",
        requested_ref="judge:local",
        execution_ref=_DIGEST,
        content_identity=_DIGEST,
        identity_kind="docker-image-id",
    )

    backend.verify_resolved_image_for_execution(resolved)

    assert looked_up == [_DIGEST]


@pytest.mark.parametrize("image_id", ["", "abc123", "sha256:abc", "b" * 64])
def test_docker_resolver_rejects_noncanonical_image_ids(image_id: str) -> None:
    backend = DockerBackend(
        client=SimpleNamespace(
            images=SimpleNamespace(
                get=lambda image: SimpleNamespace(id=image_id)
            )
        )
    )

    with pytest.raises(RuntimeError, match="immutable Docker image ID"):
        backend.resolve_image_for_execution("judge:local")


def test_long_k8s_replay_names_keep_a_hash_of_the_full_submission_id() -> None:
    prefix = "mlsysforge.judge." + "very-long-task-name-" * 4
    first = _sanitize_k8s_name(prefix + "replay-candidate-a-curve-one")
    second = _sanitize_k8s_name(prefix + "replay-candidate-a-curve-two")

    assert first != second
    assert len(first) <= 63
    assert len(second) <= 63
    assert re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", first)
    assert re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", second)
