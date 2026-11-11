"""MInference controller bootstrap; public mounts only, private runtime failclosed.

It does not assert that the SDK provides authoritative
immutable-image/single-physical-GPU evidence.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import re
import sys

TASK_DIRECTORY='minference-sparse-prefill-h100-v3-seed'
TASK_ID='minference_sparse_prefill_h100_seed_v3'
WORKDIR='/app/workdir/minference_prefill_lab'
JUDGE_ROOT='/opt/minference_prefill_judge'
GPU=re.compile(r'GPU-[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}\Z')


def actual_runtime(backend,handle,*,gpu_count):
    """Live native backend identity plus observation; never create-time hints."""
    receipt=backend.authoritative_container_receipt(handle)
    if (not isinstance(receipt,dict) or receipt.get('Id')!=handle.id
        or not re.fullmatch('[0-9a-f]{64}',str(handle.id))
        or not re.fullmatch('sha256:[0-9a-f]{64}',str(receipt.get('Image')))
        or receipt.get('State',{}).get('Running') is not True):
        raise ValueError('actual native Docker image/container identity required; unsupported platform has no fallback')
    host=receipt.get('HostConfig',{});requests=host.get('DeviceRequests') or []
    if (host.get('Privileged') is not False or host.get('Devices') not in (None,[])
        or len(requests)!=1 or requests[0].get('Driver') not in ('','nvidia')
        or requests[0].get('Capabilities')!=[['gpu']] or type(requests[0].get('Count')) is not int
        or requests[0]['Count']!=0):
        raise ValueError('actual GPU request/isolation differs')
    pool=requests[0].get('DeviceIDs')
    if not isinstance(pool,list) or len(pool)!=gpu_count or len(set(pool))!=gpu_count or any(not GPU.fullmatch(str(g)) for g in pool):
        raise ValueError('exact physical GPU UUIDs required')
    result=backend.exec_run(handle,['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],user='root')
    observed=[line.strip().lower().removeprefix('gpu-') for line in result.output.splitlines() if line.strip()]
    if result.exit_code!=0 or len(observed)!=gpu_count or set(observed)!={g.lower().removeprefix('gpu-') for g in pool}:
        raise ValueError('actual visible physical GPUs differ from native receipt')
    return receipt,pool


def _module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value;spec.loader.exec_module(value);return value


def private_payload_metadata(task_root):
    task_root=Path(task_root)
    if task_root.name!=TASK_DIRECTORY:raise ValueError('MInference task directory mismatch')
    files={p:hashlib.sha256((task_root/p).read_bytes()).hexdigest() for p in (
        'images/judge/protected_hashes.txt','qualification/external-assets-contract.json',
        'qualification/numerical_policy.json')}
    return {'task_id':TASK_ID,'mode':'preinstalled-protected-payload',
        'metadata_sha256':files,
        'files':[{'source':p,'sha256':value,'size':(task_root/p).stat().st_size,
                  'installation':'preinstalled-source-metadata'} for p,value in sorted(files.items())],
        'qualified':False,
        'private_runtime_status':'authoritative-single-GPU-kernel-runtime-required'}


def install_private_judge(backend,handle,task_root):
    metadata=private_payload_metadata(task_root)
    receipt,_=actual_runtime(backend,handle,gpu_count=1)
    task_root=Path(task_root)
    assets=_module('minference_private_assets',task_root/'scripts/external_assets_on_demand_v3.py')
    contract=assets.validate_contract(json.loads((task_root/'qualification/external-assets-contract.json').read_bytes()))
    mounts={m['Destination']:m for m in receipt.get('Mounts',[])}
    if len(mounts)!=2 or set(mounts)!=set(assets.MOUNTS.values()):
        raise ValueError('native private bootstrap requires exact two RO kernel assets; no archive/cache fifth mount')
    bindings={'schema_version':1,'protocol':assets.PROTOCOL,'assets':[]}
    for row in contract['assets']:
        role=row['role'];mount=mounts[assets.MOUNTS[role]]
        bindings['assets'].append({**row,'mount_path':assets.MOUNTS[role],
            'source_directory':mount['Source'],'read_only':True,
            'visibility':'public' if role=='public-performance' else 'private',
            'consumers':['worker','judge'] if role=='public-performance' else ['judge'],
            'storage_identity':assets.STORAGE_IDENTITY})
    bindings=assets.validate_bindings(bindings,contract)
    assets.validate_inspection(receipt,bindings,receipt['Image'],contract=contract)
    code="import sys,json;sys.path.insert(0,'/opt/minference_prefill_judge/scripts');import external_assets_on_demand_v3 as A;value=A.strict_json(sys.argv[1]);inspection=A.strict_json(sys.argv[2]);r=A.install(value,inspection,image_id=sys.argv[3],bindings_sha256=A.digest(A.canonical(value)));print(json.dumps({'status':r['status'],'bindings_sha256':r['bindings_sha256'],'receipt_sha256':A.digest(A.canonical(r))}))"
    result=backend.exec_run(handle,['python3','-I','-c',code,assets.canonical(bindings).decode(),
        assets.canonical(receipt).decode(),receipt['Image']],user='root')
    if result.exit_code!=0:raise RuntimeError('native private two-asset bootstrap failed')
    proof=json.loads(result.output)
    if (set(proof)!={'status','bindings_sha256','receipt_sha256'} or proof.get('status')!='ready'
        or any(not isinstance(proof.get(k),str) or re.fullmatch('[0-9a-f]{64}',proof[k]) is None
               for k in ('bindings_sha256','receipt_sha256'))):
        raise RuntimeError('private asset bootstrap incomplete')
    # Host-only handle metadata; caller must pass these exact dynamic hashes
    # into root evaluation context. Never expose them to candidate feedback.
    handle.minference_external_context={'EXTERNAL_ASSET_BINDINGS_SHA256':proof['bindings_sha256'],
        'EXTERNAL_ASSET_RECEIPT_SHA256':proof['receipt_sha256']}
    return metadata


def private_evaluation_context(task_root,run_id,output):return {}

def install_worker_bindings(backend,handle,task_root,standard_worker_rows):
    task_root=Path(task_root)
    if task_root.name!=TASK_DIRECTORY:raise ValueError('MInference task directory mismatch')
    assets=_module('minference_worker_asset_validator',task_root/'scripts/external_assets_on_demand_v3.py')
    projection=_module('minference_worker_projection',Path(__file__).with_name('worker_asset_projection.py'))
    contract_raw=(task_root/'qualification/public-assets-contract.json').read_bytes()
    native=projection.project_worker_assets(standard_worker_rows,
        contract=json.loads(contract_raw),validator=assets)
    receipt,pool=actual_runtime(backend,handle,gpu_count=1)
    context={'image_id':receipt['Image'],'gpu_uuid':pool[0],'purpose':'public-diagnostic'}
    installer=r'''import hashlib,json,os,pathlib,stat,subprocess,sys
assert os.geteuid()==0
sys.path.insert(0,'/workspace/benchmark')
import public_assets_v3 as P
import external_assets_v3 as A
contract_raw,contract=P.read_image_file(P.CONTRACT)
assert hashlib.sha256(contract_raw).hexdigest()==sys.argv[2]
value=A.validate_bindings(A.strict_json(sys.argv[1]),contract,purpose='public-diagnostic')
raw=A.canonical(value)
target=P.BINDINGS
if target.exists():
    old,_=P.read_image_file(target);assert old==raw
else:
    fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o444)
    try:
        view=memoryview(raw)
        while view:
            n=os.write(fd,view);assert n>0;view=view[n:]
        os.fsync(fd);os.fchown(fd,0,0);os.fchmod(fd,0o444)
    finally:os.close(fd)
context=A.strict_json(sys.argv[3])
assert set(context)=={'image_id','gpu_uuid','purpose'} and context['purpose']=='public-diagnostic'
context_target=pathlib.Path('/workspace/benchmark/on-demand-context.json')
context_raw=A.canonical(context)
if context_target.exists():
    previous,_=P.read_image_file(context_target);assert previous==context_raw
else:
    fd=os.open(context_target,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o444)
    try:
        view=memoryview(context_raw)
        while view:
            n=os.write(fd,view);assert n>0;view=view[n:]
        os.fsync(fd);os.fchown(fd,0,0);os.fchmod(fd,0o444)
    finally:os.close(fd)
probe="import sys,json;sys.path.insert(0,'/workspace/benchmark');from public_assets_v3 import verify_public_assets;print(json.dumps(verify_public_assets()))"
cp=subprocess.run(['runuser','-u','agent','--','python3','-I','-c',probe],capture_output=True,timeout=120)
assert cp.returncode==0
result=json.loads(cp.stdout)
assert result['execution_uid']==1000 and len(result['assets'])==2
print(json.dumps({'status':'worker-public-two-role-bootstrap-ready','native_bindings_sha256':hashlib.sha256(raw).hexdigest(),'public_context_sha256':hashlib.sha256(context_raw).hexdigest(),'assets':result['assets'],'qualified':False,'context_scope':'live native Docker identity and one observed GPU; public diagnostic only'}))
'''
    result=backend.exec_run(handle,['python3','-I','-c',installer,
        assets.canonical(native).decode(),hashlib.sha256(contract_raw).hexdigest(),
        assets.canonical(context).decode()],user='root')
    if result.exit_code!=0:raise RuntimeError('MInference Worker bindings/bootstrap failed')
    return json.loads(result.output)
