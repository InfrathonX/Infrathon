#!/usr/bin/env python3
"""Task-owned SForge adapter for the eight-GPU Work and Judge containers.

Work and every Judge container use the eight GPUs in FLUXMOE_GPU_UUIDS (one
NVSwitch node), each seeing them as CUDA devices 0-7.  The Judge pauses Work
while it measures (shared mode; FLUXMOE_JUDGE_GPU_UUIDS stays unset, since a
separate Judge group would need another eight GPUs on the same node).  Replay
and the standalone archive Judge use FLUXMOE_GPU_UUIDS as well.
"""
from __future__ import annotations
import os,re,subprocess,sys,threading,time
from contextlib import contextmanager
from fcntl import LOCK_EX,LOCK_UN,flock
from pathlib import Path
from typing import Any,Iterator
from docker.types import DeviceRequest
from sforge.harness import run_agent
from sforge.harness.agent.codex import CodexAgent
from sforge.harness.backend.docker_backend import DockerBackend,DockerContainerHandle

GPU_COUNT=8
GPU_UUID_ENV="FLUXMOE_GPU_UUIDS"; GPU_INDEX_ENV="FLUXMOE_GPU_INDICES"
JUDGE_GPU_UUID_ENV="FLUXMOE_JUDGE_GPU_UUIDS"; PAUSE_WORK_FOR_JUDGE_ENV="FLUXMOE_PAUSE_WORK_FOR_JUDGE"
NETWORK_ENV="SFORGE_ISOLATED_DOCKER_NETWORK"
MANAGED_JUDGE_LOCK_PATH=Path("/tmp/flux-moe-ag-dual-h100-managed-judge.lock")
MANAGED_WORK_IDLE_TIMEOUT_ENV="FLUXMOE_MANAGED_WORK_IDLE_TIMEOUT_SECONDS"
MANAGED_WORK_IDLE_TIMEOUT_SECONDS=900.0; MANAGED_GPU_IDLE_MAX_USED_MIB=256
TARGET_TASK_ID="flux_moe_ag_scatter_grouped_gemm_dual_h100_seed_v1"
_GPU_UUID_RE=re.compile(r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_RUN_ID_RE=re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}")
_managed_judge_thread_lock=threading.Lock()

class ManagedJudgeLifecycleError(RuntimeError):
    def __init__(self,failure_class): super().__init__(f"managed Judge lifecycle failed: {failure_class}"); self.failure_class=failure_class

def _resolve_gpu_uuids_from_indices(raw: str) -> str:
    values=[x.strip() for x in raw.split(",") if x.strip()]
    if len(values)!=GPU_COUNT or len(set(values))!=GPU_COUNT: raise RuntimeError(f"{GPU_INDEX_ENV} must name exactly {GPU_COUNT} distinct physical GPU indices")
    try: indices=[int(x) for x in values]
    except ValueError as exc: raise RuntimeError(f"{GPU_INDEX_ENV} accepts only integer indices") from exc
    listing=subprocess.run(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader,nounits"],check=True,capture_output=True,text=True).stdout
    by_index={int(line.partition(",")[0].strip()):line.partition(",")[2].strip() for line in listing.splitlines() if line.strip()}
    if any(index not in by_index for index in indices): raise RuntimeError(f"{GPU_INDEX_ENV} names a missing GPU")
    return ",".join(by_index[index] for index in indices)

def _gpu_uuids(raw: str|None=None,*,env_name: str=GPU_UUID_ENV):
    if raw is None:
        raw=os.environ.get(GPU_UUID_ENV,"").strip() or _resolve_gpu_uuids_from_indices(os.environ.get(GPU_INDEX_ENV,""))
    values=[x.strip() for x in raw.split(",") if x.strip()]
    if len(values)!=GPU_COUNT or any(_GPU_UUID_RE.fullmatch(x) is None for x in values) or len({x.casefold() for x in values})!=GPU_COUNT: raise RuntimeError(f"{env_name} must contain {GPU_COUNT} distinct full GPU UUIDs")
    return values

def _separate_judge_gpu_uuids():
    """The Judge pair of a separate-mode run, or None in shared mode."""
    raw=os.environ.get(JUDGE_GPU_UUID_ENV,"").strip()
    if not raw: return None
    judge=_gpu_uuids(raw,env_name=JUDGE_GPU_UUID_ENV)
    if {x.casefold() for x in judge}&{x.casefold() for x in _gpu_uuids()}:
        raise RuntimeError(f"{JUDGE_GPU_UUID_ENV} overlaps the Work pair in {GPU_UUID_ENV}; pass --judge-gpus with two other GPUs, or --shared-gpus")
    return judge

def _judge_gpu_uuids():
    judge=_separate_judge_gpu_uuids()
    return _gpu_uuids() if judge is None else judge

def _pause_work_for_judge():
    value=os.environ.get(PAUSE_WORK_FOR_JUDGE_ENV,"").strip()
    if value in {"","1"}: return True
    if value=="0": return False
    raise RuntimeError(f"{PAUSE_WORK_FOR_JUDGE_ENV} must be 0 or 1")

def _query_target_gpu_idle(gpu_uuids):
    try:
        apps=subprocess.run(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader,nounits"],check=True,capture_output=True,text=True,timeout=10)
        usage=subprocess.run(["nvidia-smi","--query-gpu=uuid,memory.used","--format=csv,noheader,nounits"],check=True,capture_output=True,text=True,timeout=10)
    except (OSError,subprocess.SubprocessError) as exc: raise ManagedJudgeLifecycleError("gpu_probe_failed") from exc
    targets=set(gpu_uuids)
    if any(line.split(",")[0].strip() in targets for line in apps.stdout.splitlines() if line.strip()): return False
    observed={}
    try:
        for line in usage.stdout.splitlines():
            fields=[x.strip() for x in line.split(",")]
            if len(fields)==2 and fields[0] in targets: observed[fields[0]]=int(fields[1])
    except ValueError as exc: raise ManagedJudgeLifecycleError("gpu_probe_failed") from exc
    if set(observed)!=targets: raise ManagedJudgeLifecycleError("gpu_probe_failed")
    return all(x<=MANAGED_GPU_IDLE_MAX_USED_MIB for x in observed.values())

def _managed_idle_timeout():
    try: value=float(os.environ.get(MANAGED_WORK_IDLE_TIMEOUT_ENV,str(MANAGED_WORK_IDLE_TIMEOUT_SECONDS)))
    except ValueError as exc: raise ManagedJudgeLifecycleError("invalid_idle_timeout") from exc
    if not 1<=value<=3600: raise ManagedJudgeLifecycleError("invalid_idle_timeout")
    return value

def _container_status(container): container.reload(); return str(getattr(container,"status",""))
def _resume_work(container,*,after_judge=False):
    if _container_status(container)=="paused": container.unpause()
    status=_container_status(container)
    if status!="running":
        # The deadline freeze may have stopped the paused container while the
        # Judge round completed; that result is authoritative, not a failure.
        if after_judge and status in {"exited","dead","removing"}: return
        raise ManagedJudgeLifecycleError("work_resume_failed")

def _managed_judge_lock_path(gpu_uuids):
    # Host-wide on purpose: two Flux Judges on different pairs of one NVSwitch host fail together in
    # the symmetric-memory rendezvous ("CUDA driver error: out of memory"), alone they pass.
    return MANAGED_JUDGE_LOCK_PATH

@contextmanager
def _exclusive_managed_judge(*,backend: Any,task_id: str,run_id: str)->Iterator[None]:
    if getattr(backend,"backend_name",None)!="docker" or getattr(backend,"client",None) is None: raise ManagedJudgeLifecycleError("unsupported_backend")
    if task_id!=TARGET_TASK_ID or _RUN_ID_RE.fullmatch(run_id) is None: raise ManagedJudgeLifecycleError("invalid_run_identity")
    timeout=_managed_idle_timeout(); name=f"sforge.run.{task_id}.{run_id}"
    separate_judge=_separate_judge_gpu_uuids()
    if separate_judge is not None and not _pause_work_for_judge():
        # Separate pairs with FLUXMOE_PAUSE_WORK_FOR_JUDGE=0, Work keeps
        # running.  The host-wide lock still keeps this Judge apart from every
        # other Flux Judge (symmetric memory); the idle wait covers the Judge
        # pair only.  As in shared mode a Judge only starts while Work is
        # running: once the run has frozen or removed Work,
        # a Judge still queued here refuses instead of meeting the launcher's
        # final Judge on the Judge pair.
        with _managed_judge_thread_lock,_managed_judge_lock_path(separate_judge).open("a+") as lock:
            flock(lock.fileno(),LOCK_EX)
            deadline=time.monotonic()+timeout
            try:
                try: work=backend.client.containers.get(name)
                except Exception as exc: raise ManagedJudgeLifecycleError("work_not_found") from exc
                while True:
                    if _container_status(work)!="running": raise ManagedJudgeLifecycleError("work_not_running")
                    if _query_target_gpu_idle(separate_judge): break
                    if time.monotonic()>=deadline: raise ManagedJudgeLifecycleError("gpu_busy")
                    time.sleep(1)
                yield
            finally: flock(lock.fileno(),LOCK_UN)
        return
    # Shared pair (separate_judge is None): probe and pause on that one pair, as
    # always.  Separate pairs (FLUXMOE_PAUSE_WORK_FOR_JUDGE=1, the default):
    # Work is paused only once both its pair and the Judge pair are idle, so no Work
    # symmetric-memory rendezvous can overlap the Judge's.
    probe=_gpu_uuids() if separate_judge is None else [*_gpu_uuids(),*separate_judge]
    with _managed_judge_thread_lock,_managed_judge_lock_path(probe).open("a+") as lock:
        flock(lock.fileno(),LOCK_EX)
        # The idle wait starts once this pair is ours, so queueing is never reported as gpu_busy.
        deadline=time.monotonic()+timeout
        try:
            try: work=backend.client.containers.get(name)
            except Exception as exc: raise ManagedJudgeLifecycleError("work_not_found") from exc
            paused=False
            try:
                while True:
                    if _container_status(work)!="running": raise ManagedJudgeLifecycleError("work_not_running")
                    if time.monotonic()>=deadline: raise ManagedJudgeLifecycleError("gpu_busy")
                    if not _query_target_gpu_idle(probe): time.sleep(1); continue
                    work.pause(); paused=True
                    if _container_status(work)!="paused": raise ManagedJudgeLifecycleError("work_pause_failed")
                    if _query_target_gpu_idle(probe): break
                    _resume_work(work); paused=False; time.sleep(1)
                yield
            finally:
                if paused:_resume_work(work,after_judge=True)
        finally: flock(lock.fileno(),LOCK_UN)

def _install_managed_judge_wrapper():
    from sforge.harness import judge_server
    original=judge_server.judge_submission
    if getattr(original,"_flux_moe_ag_managed_serialization",False): return
    def serialized(*args,**kwargs):
        spec=kwargs.get("task_spec"); run_id=kwargs.get("run_id")
        if getattr(spec,"task_id",None)!=TARGET_TASK_ID or run_id is None:return original(*args,**kwargs)
        with _exclusive_managed_judge(backend=kwargs.get("backend"),task_id=spec.task_id,run_id=run_id): return original(*args,**kwargs)
    serialized._flux_moe_ag_managed_serialization=True; judge_server.judge_submission=serialized

def _is_judge(image,name): return ".judge." in image or ".judge." in name
def _work_network(self):
    name=os.environ.get(NETWORK_ENV,"").strip()
    if not name: raise RuntimeError(f"{NETWORK_ENV} is required")
    network=self.client.networks.get(name); network.reload()
    if network.attrs.get("Internal") is not True: raise RuntimeError("Work network must be internal")
    return name

def dual_h100_create_container(self: DockerBackend,image: str,name: str,*,command: str="tail -f /dev/null",environment=None,extra_hosts=None,cap_drop=None,cpu_limit=None,mem_limit=None,user=None,annotations=None,mounts=None):
    del annotations
    judge=_is_judge(image,name); gpu_uuids=_judge_gpu_uuids() if judge else _gpu_uuids(); container_environment=dict(environment or {}); container_environment.update(CUDA_VISIBLE_DEVICES=",".join(str(i) for i in range(GPU_COUNT)),NVIDIA_VISIBLE_DEVICES=",".join(gpu_uuids))
    volumes={}
    if mounts:
        if judge: raise RuntimeError("Judge rejects external mounts")
        for host_path,container_path,mode in mounts:
            if mode!="ro": raise RuntimeError("Work mounts must be read-only")
            volumes[host_path]={"bind":container_path,"mode":mode}
    kwargs={"volumes":volumes,"device_requests":[DeviceRequest(driver="nvidia",device_ids=gpu_uuids,capabilities=[["gpu"]])],"security_opt":["no-new-privileges:true"]}
    if judge: kwargs["network_mode"]="none"
    else: kwargs["network"]=_work_network(self)
    if cpu_limit is not None:kwargs["nano_cpus"]=int(cpu_limit*1e9)
    if mem_limit is not None:kwargs["mem_limit"]=mem_limit
    if cap_drop:kwargs["cap_drop"]=cap_drop
    if extra_hosts:kwargs["extra_hosts"]=extra_hosts
    if user:kwargs["user"]=user
    return DockerContainerHandle(self.client.containers.create(image,name=name,detach=True,command=command,environment=container_environment,**kwargs))

DockerBackend.create_container=dual_h100_create_container
CodexAgent.install_cmds=['test "$(node --version)" = v20.18.0 && command -v codex >/dev/null && codex --version >/dev/null']
# The launcher always passes --enable-internet (SForge's internet=false isolation
# needs sudo iptables, unavailable here); the internal network plus the borrowed
# llada sidecar is the enforcement boundary.  The task JSON's `internet` value
# reaches this entry as MLSYSFORGE_WORK_INTERNET ("true"/"false"): "false" keeps
# the offline Codex flags and prompt; "true" (or unset) passes the caller's value
# through, so Codex runs with web_search="live" and the prompt has no network note.
WORK_INTERNET_ENV="MLSYSFORGE_WORK_INTERNET"
def _task_internet(requested):
    value=os.environ.get(WORK_INTERNET_ENV,"").strip().lower()
    if value=="false": return False
    if value in {"","true"}: return bool(requested)
    raise RuntimeError(f"{WORK_INTERNET_ENV} must be 'true' or 'false'")
_original_format_run_cmd=CodexAgent.format_run_cmd
def task_internet_format_run_cmd(self,*args,**kwargs): kwargs["internet"]=_task_internet(kwargs.get("internet",True)); return _original_format_run_cmd(self,*args,**kwargs)
CodexAgent.format_run_cmd=task_internet_format_run_cmd
_original_generate_prompt=run_agent.generate_evolve_prompt
def task_internet_generate_prompt(*args,**kwargs): kwargs["internet"]=_task_internet(kwargs.get("internet",True)); return _original_generate_prompt(*args,**kwargs)
run_agent.generate_evolve_prompt=task_internet_generate_prompt
# The launcher picks the Agent (MLSYSFORGE_AGENT=codex|claude-code); Claude Code
# bindings: sforge/harness/agent/claude_local.py.
from sforge.harness.agent.claude_local import install_from_environment as _install_agent
from sforge.cli import main
if __name__=="__main__": _install_managed_judge_wrapper(); _install_agent(_task_internet,sys.argv); main()
