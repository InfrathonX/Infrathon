#!/usr/bin/env python3
"""Task-owned SForge adapter adding the read-only Qwen3 model bind and the multi-GPU Judge.

Work gets exactly one device, pinned by ``NVIDIA_VISIBLE_DEVICES``.  The Judge
gets every whole-GPU UUID listed in ``KV_EVICTION_JUDGE_GPU_UUIDS`` (spaces or
semicolons; the harness splits its extra-env on commas, so the list travels in
this host variable) as one DeviceRequest, and the verifier shards the suite
across them.  Without that variable the Judge falls back to the single pinned
device.  When the coordinator supplies an allocation it must name exactly the
same devices; there is no fallback to an index.

Work containers are additionally placed on the launcher's ``--internal``
Docker network (``SFORGE_ISOLATED_DOCKER_NETWORK``, required, fail-closed):
the only reachable peer there is the restricted CONNECT sidecar that
run_fixed_budget.sh starts, so ``--enable-internet`` grants no direct public
egress.  The Codex command and evolve prompt follow the task JSON via
MLSYSFORGE_WORK_INTERNET ("false" forces the offline form; "true" or unset
passes the caller's value through).
Judge containers keep the stock network placement.
"""
from __future__ import annotations
import json, os, re
from pathlib import Path
from typing import Any, Literal
from docker.types import DeviceRequest
from sforge.harness import run_agent
from sforge.harness.agent.codex import CodexAgent
from sforge.harness.backend import AcceleratorAllocation, HANDOFF_BINDING_ANNOTATIONS
from sforge.harness.backend.docker_backend import (
    DockerBackend,
    DockerContainerHandle,
    _validated_docker_labels,
    _validated_handoff_binding_labels,
)

TASK_ID = "kvpress_kv_eviction_policy_qwen3_8b_h100_scratch_v2"
JUDGE_GPUS_ENV = "KV_EVICTION_JUDGE_GPU_UUIDS"
UUID_PATTERN = re.compile(r"GPU-[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}")
MODEL_MOUNT = "/opt/models/qwen3-8b"
Role = Literal["work", "judge"]
NETWORK_ENV = "SFORGE_ISOLATED_DOCKER_NETWORK"

def _isolated_work_network(backend: DockerBackend) -> str:
    """Require the launcher-created internal network; refuse anything else."""
    network_name = os.environ.get(NETWORK_ENV, "").strip()
    if not network_name: raise RuntimeError(f"{NETWORK_ENV} is required for Work")
    try:
        network = backend.client.networks.get(network_name); network.reload()
    except Exception as exc:
        raise RuntimeError(f"isolated Docker network does not exist: {network_name}") from exc
    if network.attrs.get("Internal") is not True: raise RuntimeError(f"refusing non-internal Work network: {network_name}")
    return network_name

def _classify_role(image: str, name: str, image_tags: tuple[str, ...] = ()) -> Role:
    # candidate replay pins the Judge image to its immutable sha256 ID; classify by its tags then
    probe = " ".join(image_tags) if image.startswith("sha256:") else image
    image_work = f".work.{TASK_ID}" in probe; image_judge = f".judge.{TASK_ID}" in probe
    if image_work == image_judge: raise RuntimeError("cannot classify container image")
    image_role: Role = "work" if image_work else "judge"
    name_work = name.startswith(f"sforge.run.{TASK_ID}.")
    name_judge = f".judge.{TASK_ID}." in name or name.startswith(f"sforge.game.{TASK_ID}.")
    if name_work != name_judge:
        name_role: Role = "work" if name_work else "judge"
        if name_role != image_role: raise RuntimeError("container role mismatch between image and name")
    return image_role

def _model() -> tuple[str, str]:
    raw = os.environ.get("KV_EVICTION_MODEL_REPO", "").strip()
    if not raw: raise RuntimeError("KV_EVICTION_MODEL_REPO is required")
    repo = Path(raw).resolve(strict=True)
    if not repo.is_dir() or repo.is_symlink() or not (repo / "refs/main").is_file(): raise RuntimeError("invalid model repo")
    revision = (repo / "refs/main").read_text().strip()
    if re.fullmatch(r"[0-9a-f]{40,64}", revision) is None: raise RuntimeError("invalid model revision")
    snapshot = repo / "snapshots" / revision; config = snapshot / "config.json"
    if not snapshot.is_dir() or snapshot.is_symlink() or not config.is_file(): raise RuntimeError("model snapshot missing")
    data = json.loads(config.read_text())
    if data.get("model_type") != "qwen3" or data.get("num_hidden_layers") != 36: raise RuntimeError("model config differs")
    return f"{repo}:{MODEL_MOUNT}:ro", revision

def _device_request(requested: str, accelerator_allocation: AcceleratorAllocation | None) -> DeviceRequest:
    """One exact device: the env pin and, when present, the coordinator allocation must agree."""
    if accelerator_allocation is not None:
        if accelerator_allocation.vendor != "nvidia":
            raise RuntimeError(f"task adapter supports only NVIDIA allocations, not {accelerator_allocation.vendor!r}")
        device_ids = list(accelerator_allocation.device_ids)
        if device_ids != [requested]:
            raise RuntimeError("accelerator allocation does not name the single pinned NVIDIA_VISIBLE_DEVICES device")
    return DeviceRequest(driver="nvidia", device_ids=[requested], capabilities=[["gpu"]])

def _judge_uuids() -> list[str]:
    raw = os.environ.get(JUDGE_GPUS_ENV, "").replace(";", " ").split()
    if any(UUID_PATTERN.fullmatch(x) is None for x in raw): raise RuntimeError(f"{JUDGE_GPUS_ENV} must list whole-GPU UUIDs")
    if len(set(raw)) != len(raw) or len(raw) > 8: raise RuntimeError(f"{JUDGE_GPUS_ENV} must name 1-8 distinct GPUs")
    return raw

def _judge_device_request(uuids: list[str], accelerator_allocation: AcceleratorAllocation | None) -> DeviceRequest:
    """Every listed Judge GPU in one request; a coordinator allocation must name the same set."""
    if accelerator_allocation is not None:
        if accelerator_allocation.vendor != "nvidia" or sorted(accelerator_allocation.device_ids) != sorted(uuids):
            raise RuntimeError("accelerator allocation does not name the listed Judge GPUs")
    return DeviceRequest(driver="nvidia", device_ids=list(uuids), capabilities=[["gpu"]])

def judge_cpuset(uuids: list[str]) -> str:
    """Host CPUs for a Judge on ``uuids``: each NUMA node's whole cores are split
    evenly across *all* of that node's GPUs in index order and the Judge gets the
    blocks of its own GPUs, so concurrent Judges on different GPUs never share a
    CPU (inside the container the verifier slices this set per shard).  Empty
    when sysfs gives no locality."""
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("judge_shards_host", Path(__file__).resolve().parents[1] / "images/judge/judge_shards.py")
    if spec is None or spec.loader is None: return ""
    js = sys.modules.get(spec.name)
    if js is None:
        js = importlib.util.module_from_spec(spec); sys.modules[spec.name] = js; spec.loader.exec_module(js)
    by_node: dict[tuple[int, ...], list[Any]] = {}
    for gpu in js.visible_gpus({}):
        by_node.setdefault(tuple(sorted(js._local_cpus(gpu, Path("/sys/bus/pci/devices")))), []).append(gpu)
    blocks: dict[str, list[int]] = {}
    for local, members in by_node.items():
        if local:
            for gpu, part in zip(members, js._split_even(list(local), len(members))): blocks[gpu.uuid.removeprefix("GPU-")] = part
    if any(u.removeprefix("GPU-") not in blocks for u in uuids): return ""
    return ",".join(str(c) for c in sorted({c for u in uuids for c in blocks[u.removeprefix("GPU-")]}))

def create_container(self: DockerBackend, image: str, name: str, *, command: str = "tail -f /dev/null", environment: dict[str, str] | None = None, extra_hosts: dict[str, str] | None = None, cap_drop: list[str] | None = None, cpu_limit: int | None = None, mem_limit: str | None = None, user: str | None = None, annotations: dict[str, str] | None = None, mounts: list[tuple[str, str, str]] | None = None, accelerator_allocation: AcceleratorAllocation | None = None) -> DockerContainerHandle:
    image_tags = tuple(self._client.images.get(image).tags) if image.startswith("sha256:") else ()
    role = _classify_role(image, name, image_tags); model_volume, revision = _model(); env = dict(environment or {})
    judge_uuids = _judge_uuids() if role == "judge" else []
    if judge_uuids:
        env.update(NVIDIA_VISIBLE_DEVICES=",".join(judge_uuids)); env.pop("CUDA_VISIBLE_DEVICES", None)
        device_request = _judge_device_request(judge_uuids, accelerator_allocation)
    else:
        requested = env.get("NVIDIA_VISIBLE_DEVICES", "").strip()
        if not requested or "," in requested: raise RuntimeError("exactly one NVIDIA_VISIBLE_DEVICES value required")
        env.update(CUDA_VISIBLE_DEVICES="0"); device_request = _device_request(requested, accelerator_allocation)
    env.update(KV_EVICTION_MODEL_DIR=f"{MODEL_MOUNT}/snapshots/{revision}", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    volumes = [model_volume]
    if mounts:
        if role != "work": raise RuntimeError("Judge accepts no SForge mounts")
        for host, container, mode in mounts:
            if mode != "ro": raise RuntimeError("Work accepts read-only mounts only")
            volumes.append(f"{host}:{container}:ro")
    kwargs: dict[str, Any] = {"volumes": volumes, "device_requests": [device_request], "security_opt": ["no-new-privileges:true"]}
    if role == "work": kwargs["network"] = _isolated_work_network(self)
    if annotations is not None:
        labels = _validated_docker_labels(annotations)
        if HANDOFF_BINDING_ANNOTATIONS.intersection(labels):
            labels = _validated_handoff_binding_labels(labels)
        if labels: kwargs["labels"] = labels
    if judge_uuids and (cpuset := judge_cpuset(judge_uuids)): kwargs["cpuset_cpus"] = cpuset
    if cpu_limit is not None: kwargs["nano_cpus"] = int(cpu_limit * 1e9)
    if mem_limit is not None: kwargs["mem_limit"] = mem_limit
    if extra_hosts: kwargs["extra_hosts"] = extra_hosts
    if cap_drop: kwargs["cap_drop"] = cap_drop
    if user: kwargs["user"] = user
    return DockerContainerHandle(self.client.containers.create(image, name=name, detach=True, command=command, environment=env, **kwargs))

DockerBackend.create_container = create_container

# The launcher always passes --enable-internet because SForge's own
# internet=false isolation needs host iptables, which this host does not
# grant; the internal network plus the CONNECT sidecar is the enforcement
# boundary.  The task JSON's ``internet`` value reaches this entry through
# MLSYSFORGE_WORK_INTERNET ("true"/"false", exported by the launcher).
# "false" keeps the offline Codex flags and prompt; "true" (or unset) passes
# the caller's value through, so Codex runs with web_search="live" and the
# prompt carries no network note.
WORK_INTERNET_ENV = "MLSYSFORGE_WORK_INTERNET"


def _task_internet(requested: Any) -> bool:
    """Resolve the Codex/prompt internet flag from MLSYSFORGE_WORK_INTERNET."""
    value = os.environ.get(WORK_INTERNET_ENV, "").strip().lower()
    if value == "false":
        return False
    if value in {"", "true"}:
        return bool(requested)
    raise RuntimeError(f"{WORK_INTERNET_ENV} must be 'true' or 'false'")


_original_format_run_cmd = CodexAgent.format_run_cmd


def task_internet_format_run_cmd(
    self: CodexAgent, *args: Any, **kwargs: Any
) -> str:
    """Build the Codex command with the task JSON's internet setting."""
    kwargs["internet"] = _task_internet(kwargs.get("internet", True))
    return _original_format_run_cmd(self, *args, **kwargs)


CodexAgent.format_run_cmd = task_internet_format_run_cmd

_original_generate_prompt = run_agent.generate_evolve_prompt


def task_internet_generate_prompt(*args: Any, **kwargs: Any) -> str:
    """Build the evolve prompt with the task JSON's internet setting."""
    kwargs["internet"] = _task_internet(kwargs.get("internet", True))
    return _original_generate_prompt(*args, **kwargs)


run_agent.generate_evolve_prompt = task_internet_generate_prompt

from sforge.cli import main
if __name__ == "__main__": main()
