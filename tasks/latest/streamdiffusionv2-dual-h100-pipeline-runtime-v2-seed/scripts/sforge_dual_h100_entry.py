#!/usr/bin/env python3
"""Task-owned SForge adapter for a shared dual-H100 Work/Judge pair."""
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

GPU_UUID_ENV="SDV2_GPU_UUIDS"; GPU_INDEX_ENV="SDV2_GPU_INDICES"
NETWORK_ENV="SFORGE_ISOLATED_DOCKER_NETWORK"
MANAGED_JUDGE_LOCK_PATH=Path("/tmp/streamdiffusionv2-pipeline-dual-h100-managed-judge.lock")
MANAGED_WORK_IDLE_TIMEOUT_ENV="SDV2_MANAGED_WORK_IDLE_TIMEOUT_SECONDS"
MANAGED_WORK_IDLE_TIMEOUT_SECONDS=900.0; MANAGED_GPU_IDLE_MAX_USED_MIB=256
TARGET_TASK_ID="streamdiffusionv2_dual_h100_pipeline_runtime_seed_v2"
_GPU_UUID_RE=re.compile(r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_RUN_ID_RE=re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}")
_managed_judge_thread_lock=threading.Lock()

class ManagedJudgeLifecycleError(RuntimeError):
    def __init__(self,failure_class): super().__init__(f"managed Judge lifecycle failed: {failure_class}"); self.failure_class=failure_class

def _resolve_gpu_uuids_from_indices(raw: str) -> str:
    values=[x.strip() for x in raw.split(",") if x.strip()]
    if len(values)!=2 or len(set(values))!=2: raise RuntimeError(f"{GPU_INDEX_ENV} must name exactly two distinct physical GPU indices")
    try: indices=[int(x) for x in values]
    except ValueError as exc: raise RuntimeError(f"{GPU_INDEX_ENV} accepts only integer indices") from exc
    listing=subprocess.run(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader,nounits"],check=True,capture_output=True,text=True).stdout
    by_index={int(line.partition(",")[0].strip()):line.partition(",")[2].strip() for line in listing.splitlines() if line.strip()}
    if any(index not in by_index for index in indices): raise RuntimeError(f"{GPU_INDEX_ENV} names a missing GPU")
    return ",".join(by_index[index] for index in indices)

def _gpu_uuids(raw: str|None=None):
    if raw is None:
        raw=os.environ.get(GPU_UUID_ENV,"").strip() or _resolve_gpu_uuids_from_indices(os.environ.get(GPU_INDEX_ENV,""))
    values=[x.strip() for x in raw.split(",") if x.strip()]
    if len(values)!=2 or any(_GPU_UUID_RE.fullmatch(x) is None for x in values) or len({x.casefold() for x in values})!=2: raise RuntimeError(f"{GPU_UUID_ENV} must contain two distinct full GPU UUIDs")
    return values

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
    # One lease per GPU pair: Judges on different pairs of the same host run in parallel.
    first,second=sorted(x.lower() for x in gpu_uuids)
    return MANAGED_JUDGE_LOCK_PATH.with_name(f"{MANAGED_JUDGE_LOCK_PATH.stem}-{first}-{second}.lock")

@contextmanager
def _exclusive_managed_judge(*,backend: Any,task_id: str,run_id: str)->Iterator[None]:
    if getattr(backend,"backend_name",None)!="docker" or getattr(backend,"client",None) is None: raise ManagedJudgeLifecycleError("unsupported_backend")
    if task_id!=TARGET_TASK_ID or _RUN_ID_RE.fullmatch(run_id) is None: raise ManagedJudgeLifecycleError("invalid_run_identity")
    timeout=_managed_idle_timeout(); name=f"sforge.run.{task_id}.{run_id}"
    with _managed_judge_thread_lock,_managed_judge_lock_path(_gpu_uuids()).open("a+") as lock:
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
                    if not _query_target_gpu_idle(_gpu_uuids()): time.sleep(1); continue
                    work.pause(); paused=True
                    if _container_status(work)!="paused": raise ManagedJudgeLifecycleError("work_pause_failed")
                    if _query_target_gpu_idle(_gpu_uuids()): break
                    _resume_work(work); paused=False; time.sleep(1)
                yield
            finally:
                if paused:_resume_work(work,after_judge=True)
        finally: flock(lock.fileno(),LOCK_UN)

def _install_managed_judge_wrapper():
    from sforge.harness import judge_server
    original=judge_server.judge_submission
    if getattr(original,"_sdv2_pipeline_managed_serialization",False): return
    def serialized(*args,**kwargs):
        spec=kwargs.get("task_spec"); run_id=kwargs.get("run_id")
        if getattr(spec,"task_id",None)!=TARGET_TASK_ID or run_id is None:return original(*args,**kwargs)
        with _exclusive_managed_judge(backend=kwargs.get("backend"),task_id=spec.task_id,run_id=run_id): return original(*args,**kwargs)
    serialized._sdv2_pipeline_managed_serialization=True; judge_server.judge_submission=serialized

def _is_judge(image,name): return ".judge." in image or ".judge." in name
def _work_network(self):
    name=os.environ.get(NETWORK_ENV,"").strip()
    if not name: raise RuntimeError(f"{NETWORK_ENV} is required")
    network=self.client.networks.get(name); network.reload()
    if network.attrs.get("Internal") is not True: raise RuntimeError("Work network must be internal")
    return name

def dual_h100_create_container(self: DockerBackend,image: str,name: str,*,command: str="tail -f /dev/null",environment=None,extra_hosts=None,cap_drop=None,cpu_limit=None,mem_limit=None,user=None,annotations=None,mounts=None):
    del annotations
    judge=_is_judge(image,name); gpu_uuids=_gpu_uuids(); container_environment=dict(environment or {}); container_environment.update(CUDA_VISIBLE_DEVICES="0,1",NVIDIA_VISIBLE_DEVICES=",".join(gpu_uuids))
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
import sforge.cli as _sforge_cli

def _install_candidate_eval_round_wrapper():
    """Make archive evals exercise the candidate- holdout route.

    Upstream's standalone ``eval`` command assigns ``manual-N``.  The task's
    acceptance path deliberately uses an authoritative ``candidate-`` round,
    matching candidate replay and testing the exact private profile channel.
    """
    requested=os.environ.get("SDV2_EVAL_ROUND_ID","").strip()
    if len(sys.argv)<2 or sys.argv[1]!="eval" or not requested:
        return
    if re.fullmatch(r"(?:candidate|auto)-[A-Za-z0-9][A-Za-z0-9_.-]*",requested) is None:
        raise RuntimeError("SDV2_EVAL_ROUND_ID must be a candidate- or auto- round id")
    original=_sforge_cli.judge_submission
    def candidate_round(*args,**kwargs):
        kwargs["round_id"]=requested
        return original(*args,**kwargs)
    _sforge_cli.judge_submission=candidate_round

if __name__=="__main__":
    # The launcher picks the Agent (MLSYSFORGE_AGENT=codex|claude-code); Claude
    # Code bindings: sforge/harness/agent/claude_local.py.
    from sforge.harness.agent.claude_local import install_from_environment
    _install_managed_judge_wrapper()
    _install_candidate_eval_round_wrapper()
    install_from_environment(_task_internet,sys.argv)
    _sforge_cli.main()
