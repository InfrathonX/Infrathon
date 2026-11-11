"""CPU mock checks for trusted live identity bootstrap; no GPU/SDK is called."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sforge.harness import private_judge_bootstrap as bootstrap, run_evaluation
from sforge.harness.backend import AcceleratorAllocation, ExecResult, StreamingExecResult
from sforge.harness.backend.base import ContainerBackend, ResolvedImage
from sforge.harness.backend.docker_backend import DockerBackend, DockerContainerHandle
from sforge.harness.config import SForgeConfig
from sforge.harness.constants import START_TEST_OUTPUT, END_TEST_OUTPUT
from sforge.harness.task_spec import BenchmarkMeta, make_task_spec

GPU = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
POOL = (GPU, "GPU-bbbbbbbb-bbbb-cccc-dddd-eeeeeeeeeeee",
        "GPU-cccccccc-bbbb-cccc-dddd-eeeeeeeeeeee", "GPU-dddddddd-bbbb-cccc-dddd-eeeeeeeeeeee")
IMAGE = "sha256:" + "a" * 64
CID = "c" * 64
ALLOCATION = AcceleratorAllocation("nvidia", POOL)


def receipt():
    return {"Id": CID, "Image": IMAGE, "State": {"Running": True}, "Config": {"User": "root", "Env": []},
        "HostConfig": {"NetworkMode": "none", "Privileged": False, "Devices": [], "CapAdd": [], "DeviceRequests": [
            {"Driver": "nvidia", "DeviceIDs": list(POOL), "Count": 0, "Capabilities": [["gpu"]]}]}}


def test_default_backend_runtime_attestation_is_explicitly_unsupported():
    with pytest.raises(NotImplementedError, match="cannot attest"):
        ContainerBackend.authoritative_container_receipt(SimpleNamespace(backend_name="remote"), SimpleNamespace(id=CID))


def test_docker_receipt_uses_reloaded_actual_attrs_and_returns_detached_copy():
    events = []
    actual = receipt()
    container = SimpleNamespace(id=CID, name="judge", attrs=actual, reload=lambda: events.append("reload"))
    backend = DockerBackend(client=SimpleNamespace())
    result = backend.authoritative_container_receipt(DockerContainerHandle(container))
    assert events == ["reload"] and result == actual
    result["HostConfig"]["DeviceRequests"][0]["DeviceIDs"] = ["another"]
    assert actual["HostConfig"]["DeviceRequests"][0]["DeviceIDs"] == list(POOL)


@pytest.mark.parametrize("field,value", [("Id", "another"), ("Image", "tag:mutable"), ("State", {"Running": False})])
def test_docker_receipt_rejects_wrong_or_non_running_actual_identity(field, value):
    actual = receipt(); actual[field] = value
    backend = DockerBackend(client=SimpleNamespace())
    container = SimpleNamespace(id=CID, name="judge", attrs=actual, reload=lambda: None)
    with pytest.raises(RuntimeError, match="exact running handle"):
        backend.authoritative_container_receipt(DockerContainerHandle(container))


@pytest.mark.parametrize("change", ["image", "handle", "running", "user", "network", "uuid", "one-gpu", "many-gpus", "capabilities", "allocation", "order"])
def test_bootstrap_rejects_requested_or_inexact_image_device_identity(change):
    actual = receipt(); allocation = ALLOCATION
    if change == "image": actual["Image"] = "sha256:" + "b" * 64
    elif change == "handle": actual["Id"] = "d" * 64
    elif change == "running": actual["State"]["Running"] = False
    elif change == "user": actual["Config"]["User"] = "agent"
    elif change == "network": actual["HostConfig"]["NetworkMode"] = "bridge"
    elif change == "uuid": actual["HostConfig"]["DeviceRequests"][0]["DeviceIDs"] = ["0"]
    elif change == "one-gpu": actual["HostConfig"]["DeviceRequests"][0]["DeviceIDs"] = [GPU]
    elif change == "many-gpus": actual["HostConfig"]["DeviceRequests"][0]["DeviceIDs"] *= 2
    elif change == "capabilities": actual["HostConfig"]["DeviceRequests"][0]["Capabilities"] = [["gpu", "other"]]
    elif change == "order": allocation = AcceleratorAllocation("nvidia", tuple(reversed(POOL)))
    else: allocation = None
    with pytest.raises(ValueError):
        bootstrap.validate_receipt(actual, handle_id=CID, execution_image=IMAGE, accelerator_allocation=allocation)


def test_trusted_cache_exact_sha_and_private_permissions(tmp_path):
    path = tmp_path / "dense.json"; raw = b"trusted-only exact bytes\n"
    path.write_bytes(raw); path.chmod(0o600)
    sha = hashlib.sha256(raw).hexdigest()
    context = {bootstrap.CACHE_PATH_KEY: str(path), bootstrap.CACHE_SHA_KEY: sha}
    assert bootstrap.trusted_cache(context) == (raw, sha)
    assert not context


@pytest.mark.parametrize("case", ["missing-sha", "wrong-sha", "public-mode", "symlink", "missing-file"])
def test_untrusted_or_missing_declared_dense_cache_cannot_fall_back(tmp_path, case):
    path = tmp_path / "dense.json"; path.write_bytes(b"cache"); path.chmod(0o600)
    sha = hashlib.sha256(b"cache").hexdigest()
    if case == "public-mode": path.chmod(0o644)
    elif case == "symlink":
        link = tmp_path / "link"; link.symlink_to(path); path = link
    elif case == "missing-file": path.unlink()
    context = {bootstrap.CACHE_PATH_KEY: str(path), bootstrap.CACHE_SHA_KEY: "b" * 64 if case == "wrong-sha" else sha}
    if case == "missing-sha": context.pop(bootstrap.CACHE_SHA_KEY)
    with pytest.raises((ValueError, FileNotFoundError)): bootstrap.trusted_cache(context)


class Backend:
    backend_name = "docker"

    def __init__(self, *, fail=False):
        self.events = []; self.copied = {}; self.fail = fail
        # Synthetic trusted on-started asset installer handoff, not an actual
        # container/data proof. The generic bootstrap test then exercises the
        # remaining real protocol ordering and private-feedback boundary.
        self.handle = SimpleNamespace(id=CID, name="judge", minference_external_context={
            "EXTERNAL_ASSET_BINDINGS_SHA256": "c" * 64,
            "EXTERNAL_ASSET_RECEIPT_SHA256": "d" * 64,
        })

    def resolve_image_for_execution(self, ref):
        self.events.append(("resolve", ref))
        return ResolvedImage("docker", ref, IMAGE, IMAGE, "docker-image-id")

    def create_container(self, image, name, **kwargs):
        self.events.append(("create", image, kwargs)); return self.handle

    def start_container(self, handle): self.events.append(("start",))

    def authoritative_container_receipt(self, handle):
        self.events.append(("actual-receipt",))
        if self.fail: raise ValueError("synthetic-private-detail-MUST-NOT-BE-PUBLIC")
        return receipt()

    def copy_to_container(self, handle, path, destination):
        self.events.append(("copy", str(destination)))
        self.copied[str(destination)] = path.read_bytes()

    def exec_run(self, handle, command, **kwargs):
        if bootstrap.INSTALL in command:
            assert command[:3] == ["python3", "-I", "-c"]
            self.events.append(("install",))
            payload = json.loads(self.copied["/tmp/.sforge-private-bootstrap.json"])
            values = {"FINAL_MANIFEST_SHA256": "f" * 64}
            if payload["phase"] == "final": values["QUALITY_CONTEXT_SHA256"] = "q" * 64
            # SHA must be real hexadecimal; keep the fixture clearly synthetic.
            if "QUALITY_CONTEXT_SHA256" in values: values["QUALITY_CONTEXT_SHA256"] = "b" * 64
            return ExecResult(output=json.dumps(values))
        return ExecResult()

    def exec_run_with_timeout(self, handle, command, timeout):
        self.events.append(("evaluate",))
        result = {"valid": True, "score": 1.0, "passed": 1, "total_tests": 1}
        output = "\n".join([START_TEST_OUTPUT, ">>>>> Start Structured Result", json.dumps(result), ">>>>> End Structured Result", END_TEST_OUTPUT])
        return StreamingExecResult(output=output, exit_code=0, elapsed_seconds=.01)

    def cleanup_container(self, handle, logger=None): self.events.append(("cleanup",))


def task(tmp_path, *, protocol=bootstrap.PROTOCOL):
    data = {"task_id": "bootstrap-test", "name": "bootstrap test", "base_image": "python", "platform": "linux/amd64",
        "cwd": "/workspace", "submit_paths": ["."], "work": {"image_tag": "prebuilt"},
        "judge": {"image_tag": "prebuilt", "eval_cmd": "python verifier.py", "eval_timeout": 60,
            "parser": "structured_json", "private_bootstrap": protocol}}
    path = tmp_path / "task.json"; path.write_text(json.dumps(data))
    return make_task_spec(path, BenchmarkMeta(name="test", base_images={"python": {}}))


@pytest.mark.parametrize("phase", ["interactive", "final"])
def test_bootstrap_live_receipt_precedes_candidate_execution_and_stays_private(tmp_path, phase):
    backend = Backend(); directory = tmp_path / "result"
    report = run_evaluation.judge_submission(task(tmp_path), b"", SForgeConfig(log_dir=tmp_path), backend,
        judge_image_ref=IMAGE, accelerator_allocation=ALLOCATION, log_dir=directory, evaluation_phase=phase)
    assert report.valid
    events = [event[0] for event in backend.events]
    assert events.index("start") < events.index("actual-receipt") < events.index("install") < events.index("evaluate")
    assert backend.events[1][1] == IMAGE
    assert backend.events[1][2]["network_mode"] == "none"
    context = json.loads(backend.copied[str(run_evaluation.EVALUATION_CONTEXT_PATH)])
    assert context["private"]["EVALUATION_PHASE"] == "final"
    assert context["private"]["REQUESTED_EVALUATION_PHASE"] == phase
    payload = json.loads(backend.copied["/tmp/.sforge-private-bootstrap.json"])
    assert payload["phase"] == "final"
    assert payload["pool_gpu_uuids"] == list(POOL)
    helper = backend.copied[str(bootstrap.ROOT / "quad-runtime-binding.py")]
    assert hashlib.sha256(helper).hexdigest() == payload["binding_helper_sha256"]
    assert helper == Path(bootstrap.runtime_binding.__file__).read_bytes()
    assert context["private"]["FINAL_MANIFEST_SHA256"] == "f" * 64
    private = directory / "private_runtime_bootstrap.json"
    assert private.stat().st_mode & 0o777 == 0o600
    public = (directory / "report.json").read_text()
    assert GPU not in public and "QUALITY_CONTEXT_SHA256" not in public and "private_runtime_bootstrap" not in public


def test_bad_live_receipt_fails_before_candidate_exec_without_private_error_leak(tmp_path):
    backend = Backend(fail=True); directory = tmp_path / "result"
    report = run_evaluation.judge_submission(task(tmp_path), b"", SForgeConfig(log_dir=tmp_path), backend,
        judge_image_ref=IMAGE, accelerator_allocation=ALLOCATION, log_dir=directory)
    assert not report.valid
    assert not any(event[0] == "evaluate" for event in backend.events)
    assert backend.events[-1] == ("cleanup",)
    assert "synthetic-private-detail" not in (directory / "report.json").read_text()


@pytest.mark.parametrize("name", ["pyromind", "k8s"])
def test_remote_without_runtime_attestation_fails_before_container_creation(tmp_path, name):
    backend = Backend(); backend.backend_name = name
    report = run_evaluation.judge_submission(task(tmp_path), b"", SForgeConfig(log_dir=tmp_path), backend,
        judge_image_ref=IMAGE, accelerator_allocation=ALLOCATION, log_dir=tmp_path / "result")
    assert not report.valid
    assert backend.events == [("cleanup",)]
    assert "actual live image/physical-GPU" in report.raw_output


@pytest.mark.parametrize("phase", ["interactive", "final"])
def test_both_scored_sources_require_identical_configured_cache_binding(tmp_path, phase):
    backend = Backend(); private = {bootstrap.CACHE_PATH_KEY: "/missing/private-terminal-cache",
        bootstrap.CACHE_SHA_KEY: "b" * 64}
    with pytest.raises((ValueError, OSError)):
        bootstrap.install_private_bootstrap(backend, backend.handle, protocol=bootstrap.PROTOCOL, phase=phase,
            execution_image=IMAGE, accelerator_allocation=ALLOCATION, private_context=private, log_dir=tmp_path)
    assert not any(event[0] == "install" for event in backend.events)


def test_public_or_legacy_task_never_requests_private_runtime_receipt(tmp_path):
    backend = Backend(fail=True)
    for protocol, phase in ((None, "final"), (bootstrap.PROTOCOL, "public")):
        bootstrap.install_private_bootstrap(backend, backend.handle, protocol=protocol, phase=phase,
            execution_image=IMAGE, accelerator_allocation=ALLOCATION, private_context={}, log_dir=tmp_path)
    assert not backend.events


def test_unknown_bootstrap_is_not_admitted_by_task_loader(tmp_path):
    with pytest.raises(ValueError, match="unknown judge.private_bootstrap"): task(tmp_path, protocol="unknown")


def test_terminal_cache_copies_exact_bytes_and_removes_private_transport(tmp_path):
    backend = Backend(); cache = tmp_path / "trusted.json"; raw = b"exact private cache\n"
    cache.write_bytes(raw); cache.chmod(0o600)
    context = {bootstrap.CACHE_PATH_KEY: str(cache), bootstrap.CACHE_SHA_KEY: hashlib.sha256(raw).hexdigest()}
    bootstrap.install_private_bootstrap(backend, backend.handle, protocol=bootstrap.PROTOCOL, phase="final",
        execution_image=IMAGE, accelerator_allocation=ALLOCATION, private_context=context, log_dir=tmp_path)
    assert backend.copied[str(bootstrap.ROOT / "dense-quality.json")] == raw
    assert not list(tmp_path.glob(".private-bootstrap-*"))
    assert bootstrap.CACHE_PATH_KEY not in context and bootstrap.CACHE_SHA_KEY not in context


def test_installer_nonroot_guard_survives_inherited_pythonoptimize(tmp_path):
    # Harmless CPU fixture: the installer must stop at its UID guard before
    # opening any private path. This remains meaningful when CI itself is root.
    source = "import os\nos.geteuid=lambda:1000\n" + bootstrap.INSTALL
    result = subprocess.run([sys.executable, "-I", "-c", source, str(tmp_path / "absent.json")],
        env={**os.environ, "PYTHONOPTIMIZE": "1"}, capture_output=True, timeout=10)
    assert result.returncode != 0
    assert b"AssertionError" in result.stderr and b"FileNotFoundError" not in result.stderr


def test_installer_file_guard_survives_inherited_pythonoptimize(tmp_path):
    # Replace only the trusted absolute root in this CPU fixture. The actual
    # ordinary file intentionally has an unsafe mode, never a private asset.
    root = tmp_path / "assets"; root.mkdir()
    (root / "final_manifest.json").write_bytes(b"{}")
    (root / "final_manifest.json").chmod(0o644)
    payload = tmp_path / "transport.json"; payload.write_text(json.dumps({"phase": "interactive"}))
    source = "import os\nos.geteuid=lambda:0\n" + bootstrap.INSTALL.replace(
        'root=Path("/opt/minference_prefill_judge/private")', f"root=Path({str(root)!r})")
    result = subprocess.run([sys.executable, "-I", "-c", source, str(payload)],
        env={**os.environ, "PYTHONOPTIMIZE": "1"}, capture_output=True, timeout=10)
    assert result.returncode != 0 and b"AssertionError" in result.stderr
