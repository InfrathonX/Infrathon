#!/usr/bin/env python3
"""Task-local SForge binding for the Miles OPCD task (v3).

Wraps the external SForge CLI (`serve`, `run`, `eval`) and replaces the Docker
backend's container creation so that:
  * Work and Judge get the leased GPU pair (OPCD_GPU_UUIDS) as CUDA 0,1, read-only model
    and initial-checkpoint mounts, and a per-role host scratch at /scratch;
  * Work runs as `agent` on an internal-only network (it reaches the model API, the Judge
    and, when the task JSON sets "internet": true, public hosts only through the run's relay)
    with the host agent runtime mounted read-only;
  * the Judge runs as root with no network, never receives API credentials, and pauses
    Work for its whole duration because both roles share the GPU pair.
No agent, model or Codex version is chosen here: those come from the launch layer.
"""
from __future__ import annotations

import fcntl
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
TASK_ID = "miles_qwen3_opcd_2xh100_v3"
BENCH = "miles_opcd"
THREAD_LOCK = threading.Lock()


def host_path(name):
    """A host location resolved by the launcher (scripts/run_task.py, scripts/miles_common/host_config.py)."""
    value = os.environ.get(name, "")
    require(value, f"{name} is not set: start the task through scripts/run_task.py or scripts/qualify.py, "
                   "which resolve it from the host config")
    return Path(value)


def model_mounts():
    hf = host_path("OPCD_HF_CACHE")
    # v3: the teacher is the frozen initial Qwen3-1.7B snapshot, so the 4B model is not mounted
    return {"/models/qwen3-1.7b": hf / "models--Qwen--Qwen3-1.7B"}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


# "true"/"false" from the task JSON's internet field, exported by scripts/run_task.py (the same
# contract as the other H100 task entries).
WORK_INTERNET_ENV = "MLSYSFORGE_WORK_INTERNET"


def work_internet(requested=True):
    """The launcher always passes --enable-internet (SForge's own offline mode needs sudo iptables); the
    internal network plus the run relay is the boundary. "false" forces the offline Codex flags, prompt and
    shell proxy rules; "true" or unset passes the caller's value through."""
    value = os.environ.get(WORK_INTERNET_ENV, "").strip().lower()
    if value == "false":
        return False
    require(value in ("", "true"), f"{WORK_INTERNET_ENV} must be 'true' or 'false'")
    return bool(requested)


def gpu_uuids():
    uuids = tuple(u for u in os.environ.get("OPCD_GPU_UUIDS", "").split(",") if u)
    require(len(uuids) == 2, "OPCD_GPU_UUIDS must name the leased GPU pair")
    return uuids


def run_id():
    rid = os.environ.get("OPCD_RUN_ID", "")
    require(rid, "OPCD_RUN_ID missing")
    return rid


def role_of(name):
    if name == f"sforge.run.{TASK_ID}.{run_id()}":
        return "work"
    if name.startswith(f"{BENCH}.judge.{TASK_ID}."):
        return "judge"
    raise RuntimeError(f"container {name!r} is outside this task/run namespace")


def scratch_for(role):
    root = Path(os.environ["OPCD_RUN_ROOT"]) / f"{role}-scratch"
    root.mkdir(parents=True, exist_ok=True)
    os.chmod(root, 0o777)
    return root


def create_container(self, image, name, *, command="tail -f /dev/null", environment=None, extra_hosts=None,
                     cap_drop=None, cpu_limit=None, mem_limit=None, network_mode=None, user=None,
                     annotations=None, mounts=None, accelerator_allocation=None):
    from docker.types import DeviceRequest
    from sforge.harness.backend.docker_backend import DockerContainerHandle

    role = role_of(name)
    env = dict(environment or {})
    if role == "judge":
        env = {k: v for k, v in env.items()
               if not any(x in k.upper() for x in ("TOKEN", "SECRET", "API_KEY", "PASSWORD"))
               and not k.startswith(("SFORGE_AGENT_", "SFORGE_CODEX_", "OPENAI_", "CODEX_"))}
    env.update({"CUDA_VISIBLE_DEVICES": "0,1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1", "PIP_NO_INDEX": "1", "WANDB_MODE": "disabled",
                "RAY_USAGE_STATS_ENABLED": "0", "OPCD_RUN_ID": run_id()})
    internet = work_internet()
    if role == "work" and internet:
        env.pop("PIP_NO_INDEX")  # Work may install from the package index; the Judge stays offline.
    volumes = {str(src.resolve()): {"bind": dst, "mode": "ro"} for dst, src in model_mounts().items()}
    volumes[str(host_path("OPCD_MEGATRON_INIT").resolve())] = {"bind": "/data/megatron", "mode": "ro"}
    volumes[str(scratch_for(role))] = {"bind": "/scratch", "mode": "rw"}
    kwargs = dict(user="agent" if role == "work" else "root", hostname="opcd-" + role,
                  nano_cpus=24_000_000_000, mem_limit="256g", shm_size="64g",
                  security_opt=["no-new-privileges:true"], cap_drop=["NET_RAW", "SYS_PTRACE", "SYS_ADMIN"],
                  labels={"opcd.task": TASK_ID, "opcd.run": run_id(), "opcd.role": role},
                  device_requests=[DeviceRequest(driver="nvidia", device_ids=list(gpu_uuids()), capabilities=[["gpu"]])])
    if role == "work":
        runtime = Path(os.environ["SFORGE_AGENT_RUNTIME_DIR"]).resolve()
        for source, target, mode in mounts or []:
            require(Path(source).resolve() == runtime and target == "/opt/agent-runtime" and mode == "ro",
                    "Work accepts only the read-only host agent runtime mount")
            volumes[str(runtime)] = {"bind": target, "mode": "ro"}
        if internet:
            # Login shells keep the relay CONNECT proxy (public hosts) and add Work's own addresses to NO_PROXY.
            volumes[str(Path(__file__).resolve().parent / "public-egress.profile.sh")] = {
                "bind": "/etc/profile.d/zz-opcd-public-egress.sh", "mode": "ro"}
        else:
            volumes[str(Path(__file__).resolve().parent / "no-proxy.profile.sh")] = {
                "bind": "/etc/profile.d/zz-opcd-no-proxy.sh", "mode": "ro"}
        network = os.environ["OPCD_INTERNAL_NETWORK"]
        net = self.client.networks.get(network)
        net.reload()
        require(net.attrs.get("Internal") is True, "Work network must be internal")
        kwargs["network"] = network
    else:
        require(not mounts, "Judge accepts no caller mounts")
        kwargs["network_mode"] = "none"
    container = self.client.containers.create(image, name=name, command=command, detach=True,
                                              environment=env, volumes=volumes, **kwargs)
    return DockerContainerHandle(container)


def gpu_idle():
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"], text=True)
    used = {r.split(",")[0].strip(): int(r.split(",")[1]) for r in rows.splitlines() if "," in r}
    owners = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True)
    return all(used.get(u, 10 ** 9) <= 1024 for u in gpu_uuids()) and not set(owners.split()).intersection(gpu_uuids())


@contextmanager
def exclusive_judge(backend):
    """Pause Work (same GPU pair) for the whole Judge; wait for its GPU work to stop first."""
    try:
        work = backend.client.containers.get(f"sforge.run.{TASK_ID}.{run_id()}")
    except Exception:
        work = None
    with THREAD_LOCK, open(f"/tmp/opcd-judge-{run_id()}.lock", "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        paused = False
        try:
            deadline = time.monotonic() + 900
            while True:
                if work is not None:
                    work.reload()
                if work is not None and work.status == "running" and not paused:
                    if gpu_idle():
                        work.pause()
                        paused = True
                        continue
                elif gpu_idle():
                    break
                require(time.monotonic() < deadline, "the GPU pair stayed busy: stop Work GPU processes before submitting")
                time.sleep(2)
            yield
        finally:
            if paused:
                work.reload()
                if work.status == "paused":
                    work.unpause()
            fcntl.flock(lock, fcntl.LOCK_UN)


def install():
    from sforge.harness.backend.docker_backend import DockerBackend
    from sforge.harness.agent.codex import CodexAgent
    from sforge.harness import run_agent, judge_server

    DockerBackend.create_container = create_container
    original_judge = judge_server.judge_submission

    def serialized(*args, **kwargs):
        require(getattr(kwargs.get("task_spec"), "task_id", None) == TASK_ID, "foreign task Judge request")
        with exclusive_judge(kwargs["backend"]):
            return original_judge(*args, **kwargs)

    judge_server.judge_submission = serialized
    original_format = CodexAgent.format_run_cmd

    def task_format(self, *args, **kwargs):
        # Task internet true: Codex web_search "live" (set by the harness); false: the old offline run.
        kwargs["internet"] = work_internet(kwargs.get("internet", True))
        cmd = original_format(self, *args, **kwargs)
        # A ChatGPT-login run prefixes codex exec with the relay's CONNECT proxy. Codex 0.156 ignores
        # shell_environment_policy.exclude in exec mode and replays a captured shell environment, so
        # snapshots are off and every login shell sources the profile.d network script: offline,
        # no-proxy.profile.sh keeps the proxy out of Ray, SGLang and sforge-submit; with internet,
        # public-egress.profile.sh keeps it and adds Work's own addresses to NO_PROXY.
        return cmd.replace("codex exec", "codex exec -c " + shlex.quote("features.shell_snapshot=false"), 1)

    CodexAgent.format_run_cmd = task_format
    original_prompt = run_agent.generate_evolve_prompt

    def task_prompt(*args, **kwargs):
        kwargs["internet"] = work_internet(kwargs.get("internet", True))
        return original_prompt(*args, **kwargs)

    run_agent.generate_evolve_prompt = task_prompt


def main():
    install()
    # scripts/run_task.py picks the agent (MLSYSFORGE_AGENT=codex|claude-code, scripts/miles_common/agent.py): for
    # Claude Code, the harness's local bindings (token file, tools, WebSearch/WebFetch when the task allows internet).
    from sforge.pyromind.claude_local import install_from_environment

    install_from_environment(work_internet, sys.argv)
    from sforge.cli import main as sforge_main

    sforge_main()


if __name__ == "__main__":
    main()
