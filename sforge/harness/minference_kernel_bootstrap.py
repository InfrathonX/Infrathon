"""Trusted single-GPU kernel context; deliberately independent of full-QA v3."""
import hashlib,json,re
from pathlib import Path,PurePosixPath

PROTOCOL='minference-kernel-prefill-v3'
GPU=re.compile(r'GPU-[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}\Z')
SHA=re.compile(r'[0-9a-f]{64}\Z')
def validate_receipt(receipt,handle_id,image,allocation):
    h=receipt.get('HostConfig',{});req=h.get('DeviceRequests') or []
    if receipt.get('Id')!=handle_id or receipt.get('Image')!=image or not re.fullmatch(r'sha256:[0-9a-f]{64}',image) or receipt.get('State',{}).get('Running') is not True:
        raise ValueError('actual immutable kernel image/container receipt differs')
    if receipt.get('Config',{}).get('User') not in ('root','0','0:0') or h.get('NetworkMode')!='none' or h.get('Privileged') is not False or h.get('Devices') not in (None,[]) or h.get('CapAdd') not in (None,[]):
        raise ValueError('kernel Judge isolation differs')
    if len(req)!=1 or req[0].get('Driver')!='nvidia' or type(req[0].get('Count')) is not int or req[0]['Count']!=0 or req[0].get('Capabilities')!=[['gpu']] or req[0].get('DeviceIDs')!=list(allocation.device_ids) or allocation.vendor!='nvidia' or len(allocation.device_ids)!=1 or not GPU.fullmatch(allocation.device_ids[0]):
        raise ValueError('exact one physical NVIDIA GPU required')
    return receipt

def install(backend,handle,*,phase,execution_image,accelerator_allocation,private_context,log_dir):
    from sforge.harness.private_judge_bootstrap import _copy_private
    if phase not in ('interactive','final'):raise ValueError('scored kernel phase required')
    receipt=validate_receipt(backend.authoritative_container_receipt(handle),handle.id,execution_image,accelerator_allocation)
    hashes=getattr(handle,'minference_external_context',None)
    keys={'EXTERNAL_ASSET_BINDINGS_SHA256','EXTERNAL_ASSET_RECEIPT_SHA256'}
    if not isinstance(hashes,dict) or set(hashes)!=keys or any(not isinstance(v,str) or not SHA.fullmatch(v) for v in hashes.values()):
        raise ValueError('kernel assets must be installed by trusted task bootstrap first')
    for k,v in hashes.items():
        if k in private_context and private_context[k]!=v:raise ValueError('conflicting host asset identity')
    if any(k.startswith('MINFERENCE_DENSE_CACHE') for k in private_context):raise ValueError('kernel context cannot consume old model-answer cache')
    private_context.update(hashes)
    root=PurePosixPath('/opt/minference_prefill_judge/private')
    _copy_private(backend,handle,(json.dumps([receipt],sort_keys=True)+'\n').encode(),root/'container-inspect.json',log_dir)
    code="import hashlib,pathlib; p=pathlib.Path('/opt/minference_prefill_judge/private/final_manifest.json');s=p.lstat();assert s.st_uid==0 and s.st_nlink==1 and s.st_mode&511==256;print(hashlib.sha256(p.read_bytes()).hexdigest())"
    result=backend.exec_run(handle,['python3','-I','-c',code],user='root');digest=result.output.strip()
    if result.exit_code or not SHA.fullmatch(digest):raise RuntimeError('kernel manifest proof failed')
    private_context['FINAL_MANIFEST_SHA256']=digest
    private_context['EVALUATION_KIND']='kernel-only-v3'
    p=Path(log_dir)/'private_kernel_bootstrap.json';p.write_text(json.dumps({'protocol':PROTOCOL,'inspection':receipt,'bindings':dict(private_context)},indent=2)+'\n');p.chmod(0o600)
