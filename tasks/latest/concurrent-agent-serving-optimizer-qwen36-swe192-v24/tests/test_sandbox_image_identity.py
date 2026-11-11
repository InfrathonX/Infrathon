"""The sandbox worker accepts a staged SWE-bench image by either content address.

The classic Docker image store reports the image config digest as the image ID (``image_digest`` in the corpus
manifest); the containerd image store reports the digest of the pulled manifest (``official_manifest_digest``).
Any other ID, or a manifest-digest ID without a recorded manifest digest, is still refused.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

TASK_ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = (
    TASK_ROOT
    / "runtime/frozen/harbor_tasks/concurrent-agent-serving-optimizer-qwen36-27b-swe192-v2.1-gpu45/environment"
)
CONFIG = "sha256:" + "1" * 64
MANIFEST = "sha256:" + "2" * 64
OTHER = "sha256:" + "3" * 64


class _Reached(Exception):
    """Raised once the identity check passed and container creation began."""


@pytest.fixture()
def worker(monkeypatch):
    docker = types.ModuleType("docker")
    docker.DockerClient = object
    errors = types.ModuleType("docker.errors")
    for name in ("APIError", "ImageNotFound", "NotFound"):
        setattr(errors, name, type(name, (Exception,), {}))
    models = types.ModuleType("docker.models")
    containers = types.ModuleType("docker.models.containers")
    containers.Container = object
    dtypes = types.ModuleType("docker.types")
    dtypes.Ulimit = lambda **kwargs: kwargs
    for name, module in {
        "docker": docker,
        "docker.errors": errors,
        "docker.models": models,
        "docker.models.containers": containers,
        "docker.types": dtypes,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("kvbench_sandbox_worker_identity", ENVIRONMENT / "sandbox-worker/worker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _create(worker, image_id: str, manifest: str | None) -> None:
    def create(**_kwargs):
        raise _Reached

    worker.docker_client = types.SimpleNamespace(
        images=types.SimpleNamespace(get=lambda _name: types.SimpleNamespace(id=image_id)),
        containers=types.SimpleNamespace(create=create),
    )
    request = {
        "sandbox_id": "a" * 32,
        "image": "docker.io/swebench/sweb.eval.x86_64.example:latest",
        "image_digest": CONFIG,
        "base_commit": "b" * 40,
        "runtime_head_commit": "c" * 40,
    }
    if manifest is not None:
        request["image_manifest_digest"] = manifest
    worker.create_runtime(request)


@pytest.mark.parametrize(
    ("image_id", "manifest"),
    [(CONFIG, None), (CONFIG, MANIFEST), (MANIFEST, MANIFEST)],
)
def test_either_recorded_content_address_is_accepted(worker, image_id: str, manifest: str | None) -> None:
    with pytest.raises(_Reached):
        _create(worker, image_id, manifest)


@pytest.mark.parametrize(
    ("image_id", "manifest"),
    [(OTHER, None), (OTHER, MANIFEST), (MANIFEST, None), (MANIFEST, OTHER)],
)
def test_any_other_identity_is_refused(worker, image_id: str, manifest: str | None) -> None:
    with pytest.raises(RuntimeError, match="digest mismatch"):
        _create(worker, image_id, manifest)


def test_malformed_manifest_digest_is_refused(worker) -> None:
    with pytest.raises(ValueError, match="manifest digest"):
        _create(worker, CONFIG, "sha256:short")


def test_controller_sends_the_recorded_manifest_digest() -> None:
    source = (ENVIRONMENT / "bench-controller/runner/sandbox.py").read_text()
    assert '"image_manifest_digest": spec.runtime_manifest_digest,' in source
