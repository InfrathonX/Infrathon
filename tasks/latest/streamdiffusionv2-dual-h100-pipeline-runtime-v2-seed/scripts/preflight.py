#!/usr/bin/env python3
"""Read-only two-GPU NVLink, P2P, idle, and symmetric-memory preflight."""
from __future__ import annotations
import argparse,csv,io,json,os,signal,subprocess,sys,tempfile,time
from pathlib import Path

# Capability envelope, mirrored from runtime/hardware.py; GPU model names are
# provenance and never gate. The trusted worker re-checks this from torch.
MIN_DEVICE_MEMORY_MIB=26*1024

SYMMETRIC_PROBE=r'''import os,time,torch,torch.distributed as dist
import torch.distributed._symmetric_memory as symm
r=int(os.environ["RANK"]); d=torch.device("cuda",r); torch.cuda.set_device(d)
dist.init_process_group("nccl",device_id=d); name=dist.group.WORLD.group_name; symm.enable_symm_mem_for_group(name)
start=time.perf_counter(); x=symm.empty((4*1024*1024,),dtype=torch.uint8,device=d); h=symm.rendezvous(x,group=name); peer=h.get_buffer(1-r,x.shape,torch.uint8); x.fill_(r+1); h.barrier(channel=0); torch.cuda.synchronize()
assert int(peer[0].item())==2-r; elapsed=time.perf_counter()-start; assert elapsed<3.0,elapsed
dist.destroy_process_group()
'''

def command(*args): return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT)
def topology_link(topology,source,target):
    for line in topology.splitlines():
        fields=line.split()
        if fields and fields[0]==f"GPU{source}":
            if len(fields)>1+target:return fields[1+target]
    raise SystemExit("cannot resolve selected GPU topology link")

def torch_probe(indices):
    env=dict(os.environ); env["CUDA_VISIBLE_DEVICES"]=",".join(str(x) for x in indices)
    for key in tuple(env):
        if key.lower() in {"http_proxy","https_proxy","all_proxy","no_proxy"}:env.pop(key,None)
    p2p="import json,torch; assert torch.cuda.device_count()==2; print(json.dumps({'p2p':[torch.cuda.can_device_access_peer(0,1),torch.cuda.can_device_access_peer(1,0)]}))"
    p2p_result=json.loads(subprocess.check_output([sys.executable,"-c",p2p],env=env,text=True,timeout=60))
    if p2p_result["p2p"]!=[True,True]:raise SystemExit("bidirectional torch CUDA peer access is unavailable")
    with tempfile.TemporaryDirectory(prefix="sdv2-pipeline-preflight-") as raw:
        script=Path(raw)/"probe.py"; script.write_text(SYMMETRIC_PROBE)
        command=[sys.executable,"-m","torch.distributed.run","--standalone","--nproc-per-node=2",str(script)]
        started=time.perf_counter()
        process=subprocess.Popen(command,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
        try:
            stdout,stderr=process.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM)
            try:
                stdout,stderr=process.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                stdout,stderr=process.communicate()
            raise SystemExit(f"symmetric-memory rendezvous exceeded controller timeout; stdout={stdout[-1000:]!r}; stderr={stderr[-1000:]!r}")
        completed=subprocess.CompletedProcess(command,process.returncode,stdout,stderr)
        if completed.returncode:raise SystemExit(f"symmetric-memory rendezvous failed: {completed.stderr[-1000:]}")
        return {"p2p":p2p_result["p2p"],"symmetric_memory_bytes":4*1024*1024,"wall_seconds":time.perf_counter()-started,"protocol_budget_seconds":3.0}

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--gpu-indices",default=os.environ.get("SDV2_GPU_INDICES","")); p.add_argument("--expected-gpu-uuids",default=os.environ.get("SDV2_GPU_UUIDS","")); p.add_argument("--max-used-mib",type=int,default=1024); p.add_argument("--allow-non-nvlink",action="store_true"); a=p.parse_args(argv)
    try: indices=[int(x) for x in a.gpu_indices.split(",")]
    except ValueError as exc:raise SystemExit("GPU indices must be integers") from exc
    if len(indices)!=2 or len(set(indices))!=2:raise SystemExit("exactly two distinct GPU indices are required")
    raw=command("nvidia-smi","--query-gpu=index,uuid,name,memory.total,memory.used","--format=csv,noheader,nounits"); rows={}
    for row in csv.reader(io.StringIO(raw),skipinitialspace=True):rows[int(row[0])]={"index":int(row[0]),"uuid":row[1],"name":row[2],"memory_total_mib":int(row[3]),"memory_used_mib":int(row[4])}
    if any(i not in rows for i in indices):raise SystemExit("selected GPU index is unavailable")
    selected=[rows[i] for i in indices]; observed=tuple(x["uuid"] for x in selected)
    expected=tuple(x.strip() for x in a.expected_gpu_uuids.split(",") if x.strip())
    if expected and observed!=expected:raise SystemExit("selected GPU UUIDs differ from pin")
    for gpu in selected:
        if gpu["memory_total_mib"]<MIN_DEVICE_MEMORY_MIB:raise SystemExit(f"GPU {gpu['index']} has {gpu['memory_total_mib']} MiB, below the {MIN_DEVICE_MEMORY_MIB} MiB task minimum")
        if gpu["memory_used_mib"]>a.max_used_mib:raise SystemExit(f"GPU {gpu['index']} is busy")
    owners=[row for row in csv.reader(io.StringIO(command("nvidia-smi","--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory","--format=csv,noheader,nounits")),skipinitialspace=True) if row and row[0] in set(observed)]
    if owners:raise SystemExit(f"selected GPU pair has active compute owners: {owners}")
    topology=command("nvidia-smi","topo","-m"); links=[topology_link(topology,indices[0],indices[1]),topology_link(topology,indices[1],indices[0])]
    if not a.allow_non_nvlink and not all(x.startswith("NV") for x in links):raise SystemExit(f"selected GPUs are not an NVLink pair: {links}")
    probe=torch_probe(indices)
    print(json.dumps({"gpus":selected,"topology_link":links,"topology":topology,"torch_probe":probe},indent=2)); return 0
if __name__=="__main__":raise SystemExit(main())
