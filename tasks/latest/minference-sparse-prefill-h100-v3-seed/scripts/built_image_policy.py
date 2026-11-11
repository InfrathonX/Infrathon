"""CPU-only final filesystem/software checks for source-only full-RULER images.

External model/text bytes and GPU/answer quality are independently checked at
runtime. A successful image policy receipt is never release qualification.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import uuid

IMAGE_ID=re.compile(r'sha256:[0-9a-f]{64}\Z')
JUDGE_ROOT='/opt/minference_prefill_judge'
EXPECTED_SOFTWARE_IMAGE_ID='sha256:603d5163cef94886eade0d7b55e8b39ecc29f6ff1ad68c76556d573e16613189'
CLEAN_BUILD_RECEIPT_SHA256='52264c8f08f3d17b046dd0295b0f8d86eed927a4795d7a60277d7651481c45ac'
CLEAN_EQUIVALENCE_RECEIPT_SHA256='828024029d60860e08d1d9938e0785207d2358c7952677f89b9efd6cde42ef0a'
OFFICIAL_TORCH_IMAGE_ID='sha256:cf5aa3f7045a68c10d80f546746591c5ccae6a33729e5e32625ff76bd2c036fe'
ABSENT_MODULES=('minference','sgl_kernel','sglang','vllm','vllm_flash_attn','flash_attn_interface','flashinfer','xformers')
EXTERNAL_QA=set()

def require(ok,message):
    if not ok:raise ValueError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def _run(command,label,timeout=120):
    try:r=subprocess.run(command,capture_output=True,text=True,timeout=timeout,check=False)
    except (OSError,subprocess.TimeoutExpired) as exc:raise ValueError(label+' could not complete') from exc
    require(r.returncode==0,label+' failed: '+r.stderr[:2048]);return r
def validate_layer_ancestry(image,base,*,image_id,base_image_id):
    for row,digest in ((image,image_id),(base,base_image_id)):
        require(IMAGE_ID.fullmatch(str(digest)) and isinstance(row,dict) and row.get('Id')==digest,'actual immutable image identity differs')
        rf=row.get('RootFS');require(isinstance(rf,dict) and rf.get('Type')=='layers' and isinstance(rf.get('Layers'),list) and rf['Layers'] and all(IMAGE_ID.fullmatch(str(v)) for v in rf['Layers']),'complete immutable image layer list required')
    require(image['RootFS']['Layers'][:len(base['RootFS']['Layers'])]==base['RootFS']['Layers'],'exact shared software ancestor prefix differs')
    return {'schema_version':1,'protocol':'sforge-docker-layer-ancestry-v1','status':'passed','image_id':image_id,'base_image_id':base_image_id,'image_layers':list(image['RootFS']['Layers']),'base_layers':list(base['RootFS']['Layers']),'exact_base_layer_prefix':True}
def validate_image_ancestry(*,engine,image_id,base_image_id,rebuilt_parent_id=None):
    require(IMAGE_ID.fullmatch(str(image_id)) and IMAGE_ID.fullmatch(str(base_image_id)),'immutable image/base content IDs required before inspection')
    require(base_image_id in (EXPECTED_SOFTWARE_IMAGE_ID,rebuilt_parent_id),'full-RULER software parent must be the verified clean603d parent or a verified rebuild of it')
    rows=json.loads(_run([engine,'image','inspect',image_id,base_image_id],'image/base inspection').stdout)
    require(isinstance(rows,list) and len(rows)==2,'actual two image inspection records required')
    return validate_layer_ancestry(*rows,image_id=image_id,base_image_id=base_image_id)

REBUILT_PARENT_PROTOCOL='minference-rebuilt-software-parent-v1'
def validate_rebuilt_software_parent(verification_receipt,*,engine='docker'):
    """Accept a parent rebuilt from images/parent/Dockerfile and checked by
    scripts/verify_software_parent.py --receipt: official Torch base layers, then
    package files identical to the recorded clean parent's manifest."""
    root=Path(__file__).resolve().parents[1]
    raw=Path(verification_receipt).read_bytes();receipt=json.loads(raw)
    clean_manifest=json.loads((root/'references/software-parent/original-file-equivalence-receipt.json').read_bytes())['clean_file_manifest_sha256']
    require(receipt.get('protocol')==REBUILT_PARENT_PROTOCOL and receipt.get('status')=='identical' and receipt.get('qualified') is False,'rebuilt software parent verification differs')
    require(IMAGE_ID.fullmatch(str(receipt.get('image_id'))) and receipt['image_id']!=EXPECTED_SOFTWARE_IMAGE_ID,'rebuilt software parent identity differs')
    require(receipt.get('official_base_image_id')==OFFICIAL_TORCH_IMAGE_ID and receipt.get('clean_file_manifest_sha256')==clean_manifest,'rebuilt software parent references differ')
    require(receipt.get('dockerfile_sha256')==sha((root/'images/parent/Dockerfile').read_bytes()),'rebuilt software parent was not built from this images/parent/Dockerfile')
    require(type(receipt.get('compared_package_files')) is int and receipt['compared_package_files']>0,'rebuilt software parent comparison is empty')
    rows=json.loads(_run([engine,'image','inspect',receipt['image_id'],OFFICIAL_TORCH_IMAGE_ID],'rebuilt parent inspection').stdout)
    require(isinstance(rows,list) and len(rows)==2 and rows[0].get('Id')==receipt['image_id'],'rebuilt software parent is not present')
    layers=rows[0]['RootFS']['Layers'];official=rows[1]['RootFS']['Layers']
    require(layers==receipt.get('rootfs_layers') and layers[:len(official)]==official,'rebuilt software parent RootFS differs from its verification')
    return {'clean_image_id':receipt['image_id'],'official_torch_image_id':OFFICIAL_TORCH_IMAGE_ID,'rebuilt_parent_verification_sha256':sha(raw),'recorded_clean_parent_id':EXPECTED_SOFTWARE_IMAGE_ID,'clean_file_manifest_sha256':clean_manifest,'compared_package_files':receipt['compared_package_files'],'exact_equal_rootfs_layers':layers,'historical_judge_layers_inherited':False,'qualified':False}

def validate_software_parent_evidence(build_receipt,equivalence_receipt):
    """Accept only the independently measured clean software epoch, before Docker."""
    from stage_thin_images_v3 import read,parse
    build_raw=read(Path(build_receipt));equivalence_raw=read(Path(equivalence_receipt))
    require(sha(build_raw)==CLEAN_BUILD_RECEIPT_SHA256 and sha(equivalence_raw)==CLEAN_EQUIVALENCE_RECEIPT_SHA256,'clean software provenance raw SHA differs')
    build=parse(build_raw);equivalence=parse(equivalence_raw)
    require(build.get('status')=='build-completed' and type(build.get('returncode')) is int and build['returncode']==0 and build.get('image_id')==EXPECTED_SOFTWARE_IMAGE_ID and build.get('official_base_image_id')==OFFICIAL_TORCH_IMAGE_ID and build.get('model_qkv_or_prompts_copied') is False and build.get('python_import_check_cuda_visible_devices')=='' and build.get('python_import_check_nvidia_visible_devices')=='void' and build.get('persistent_cuda_visible_devices_override_added') is False,'clean software build provenance differs')
    require(equivalence.get('protocol')=='minference-clean-software-masked-derivative-proof-v1' and equivalence.get('masked_image_id')==EXPECTED_SOFTWARE_IMAGE_ID and equivalence.get('masked_build_receipt_sha256')==sha(build_raw) and equivalence.get('actual_rootfs_layers_equal') is True and equivalence.get('actual_configuration_environment_equal') is True and equivalence.get('actual_forbidden_paths_absence_derived_from_identical_rootfs') is True and equivalence.get('software_files_rescanned_on_masked_image') is False and equivalence.get('model_qkv_or_prompts_copied') is False,'actual clean software derivative proof differs')
    layers=equivalence.get('actual_equal_rootfs_layers')
    require(isinstance(layers,list) and len(layers)==37 and all(IMAGE_ID.fullmatch(str(v)) for v in layers),'complete 37-entry software RootFS proof required')
    return {'clean_image_id':EXPECTED_SOFTWARE_IMAGE_ID,'official_torch_image_id':OFFICIAL_TORCH_IMAGE_ID,'build_receipt_sha256':sha(build_raw),'masked_derivative_certificate_sha256':sha(equivalence_raw),'original_measured_file_equivalence_receipt_sha256':equivalence['original_file_equivalence_receipt_sha256'],'original_measured_software_files':5327,'masked_software_files_rescanned':False,'exact_equal_rootfs_layers':layers,'historical_judge_layers_inherited':False,'qualified':False}

def _authenticate(row,*,image,name,nonce,user):
    require(isinstance(row,dict) and re.fullmatch('[0-9a-f]{64}',str(row.get('Id'))) and row.get('Name')=='/'+name and row.get('Image')==image,'CPU policy container identity differs')
    c=row.get('Config',{});h=row.get('HostConfig',{})
    require(c.get('Labels',{}).get('mlsysforge.full-ruler-image-policy')==nonce and c.get('User')==user,'CPU policy container ownership/user differs')
    selected={e.split('=',1)[0]:e.split('=',1)[1] for e in c.get('Env',[]) if e.startswith(('NVIDIA_VISIBLE_DEVICES=','CUDA_VISIBLE_DEVICES='))}
    require(selected=={'NVIDIA_VISIBLE_DEVICES':'void','CUDA_VISIBLE_DEVICES':''} and h.get('Runtime')=='runc' and h.get('NetworkMode')=='none' and h.get('DeviceRequests') in (None,[]) and h.get('Devices') in (None,[]) and h.get('CapAdd') in (None,[]) and h.get('Privileged') is False and row.get('Mounts')==[],'CPU policy must have no GPU/network/mounts/privilege')
    require(h.get('Memory')==4*1024**3 and h.get('NanoCpus')==2_000_000_000,'CPU policy resource bound differs')
    return row['Id']
def _python(engine,image,code,*,user='0:0',label):
    require(IMAGE_ID.fullmatch(str(image)),'immutable image required')
    nonce=uuid.uuid4().hex;name='minference-image-policy-'+nonce;cid=None;attempted=False;owned=False
    def inspect():
        rows=json.loads(_run([engine,'inspect',name],'owned CPU policy inspection',30).stdout)
        require(isinstance(rows,list) and len(rows)==1,'owned CPU policy inspection absent')
        return _authenticate(rows[0],image=image,name=name,nonce=nonce,user=user)
    try:
        attempted=True
        created=_run([engine,'create','--pull','never','--name',name,'--label','mlsysforge.full-ruler-image-policy='+nonce,
            '--runtime','runc','--network','none','--cpus','2','--memory','4g','--user',user,
            '--env','NVIDIA_VISIBLE_DEVICES=void','--env','CUDA_VISIBLE_DEVICES=',
            '--entrypoint','sleep',image,'600'],label+' create',30).stdout.strip()
        cid=inspect();owned=True;require(created==cid,'create response does not bind actual owned CID')
        _run([engine,'start',cid],label+' start',30)
        output=_run([engine,'exec',cid,'python3','-I','-c',code],label,240)
        require(inspect()==cid,'CPU policy container changed during execution');return output
    finally:
        # Lost/truncated create response is reconciled only by the unique name
        # and exact label/image/resource authentication, never a broad ps scan.
        if attempted and not owned:
            try:cid=inspect();owned=True
            except ValueError:pass
        if owned:
            require(inspect()==cid,'owned cleanup identity differs')
            _run([engine,'stop','--time','1',cid],label+' own stop',30)
            _run([engine,'rm',cid],label+' own remove',30)

def _read_stage(source_root,stage_receipt,prequalification):
    from stage_thin_images_v3 import read,parse,MAX_CONTEXT,module,protected_map
    root=Path(source_root);receipt_path=Path(stage_receipt);raw=read(receipt_path);receipt=parse(raw)
    require(receipt.get('protocol')=='minference-kernel-thin-image-context-v3' and receipt.get('qualified') is False and receipt.get('gpu_execution') is False and receipt.get('external_assets_authenticated') is False,'source staging receipt scope differs')
    require(receipt.get('source_root')==str(root) and receipt.get('stage')==str(receipt_path.parent) and receipt.get('prequalification') is prequalification,'staging source/policy epoch differs')
    files=receipt.get('files');require(isinstance(files,dict) and 0<sum(v['size'] for v in files.values())<=MAX_CONTEXT,'bounded active contexts required')
    for n,entry in files.items():
        require(not Path(n).is_absolute() and '..' not in Path(n).parts and Path(n).name!='prompts.jsonl' and Path(n).suffix not in ('.bin','.safetensors'),'payload or unsafe context member rejected')
        payload=read(receipt_path.parent/n);require(len(payload)==entry['size'] and sha(payload)==entry['sha256'],'staged source changed')
    generated=module(root,'scripts/generate_protected_hashes.py','image_protected36')
    targets=set(generated.PROTECTED_LOCAL_PATHS)|set(generated.DYNAMIC_PATHS)
    listed=protected_map(read(receipt_path.parent/'installed/protected_hashes.txt'),targets)
    require(len(targets)==25 and targets-EXTERNAL_QA=={JUDGE_ROOT+'/'+n[len('installed/'):] for n in files if n.startswith('installed/') and n!='installed/protected_hashes.txt'},'installed34 projection differs')
    for target,rel in generated.PROTECTED_LOCAL_PATHS.items():require(sha(read(root/rel))==listed[target],'current protected source differs from actual staged bytes')
    require(prequalification or (receipt['numerical_policy_status']=='frozen' and receipt['execution_bindings_complete'] is True),'formal image policy cannot accept a draft')
    return raw,receipt,listed

SOFTWARE_CODE="""from importlib.metadata import version
import torch,triton,transformers,importlib.util
assert torch.__version__=='2.8.0+cu129' and torch.version.cuda=='12.9'
assert triton.__version__=='3.4.0' and transformers.__version__=='4.57.1'
assert version('flash_attn')=='2.8.3' and version('einops')=='0.8.1'
assert all(importlib.util.find_spec(n) is None for n in ABSENT)
""".replace('ABSENT',repr(ABSENT_MODULES))

def _image_check(*,engine,image,role,source_root,stage_receipt,prequalification):
    stage_raw,record,listed=_read_stage(source_root,stage_receipt,prequalification)
    files=record['files'];expected={};stage=Path(stage_receipt).parent
    if role=='judge':
        expected={JUDGE_ROOT+'/'+n[len('installed/'):]:v['sha256'] for n,v in files.items() if n.startswith('installed/')}
        code="""import hashlib,json,os,stat
from pathlib import Path
root=Path('/opt/minference_prefill_judge')
for target,digest in EXPECTED.items():
 p=Path(target);s=p.lstat();assert p.is_file() and not p.is_symlink() and s.st_nlink==1 and s.st_uid==0
 assert stat.S_IMODE(s.st_mode)==(0o500 if p.name=='candidate_worker.py' else 0o400),target
 assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,target
for target in QA_EXTERNAL:assert not Path(target).exists(),target
for relative in ('external-assets/kernel-prompts','model/d10aef7999a2b5ba950ab3974312feeedbfe0b77'):
 p=root/relative;assert p.stat().st_uid==0 and stat.S_IMODE(p.stat().st_mode)==0o500 and not list(p.iterdir()),relative
assert not any((root/'private'/n).exists() for n in ('dense-quality.json','quality-context.json','container-inspect.json','external-assets-bindings.json'))
for d,_,ns in os.walk(root):assert not any(n=='prompts.jsonl' or n.endswith(('.safetensors','.bin')) for n in ns),d
for relative in ('external-assets','private','model'):
 p=root/relative;assert p.stat().st_uid==0 and stat.S_IMODE(p.stat().st_mode)==0o500
""".replace('EXPECTED',repr(expected)).replace('QA_EXTERNAL',repr(EXTERNAL_QA))
    else:
        for n,v in files.items():
            if n.startswith('work/benchmark/'):expected['/workspace/benchmark/'+n[len('work/benchmark/'):]]=v['sha256']
            elif n.startswith('work/workdir/'):expected['/app/workdir/'+n[len('work/workdir/'):]]=v['sha256']
        expected['/app/workdir/minference_prefill_lab/AGENT_TASK.md']=files['work/AGENT_TASK.md']['sha256']
        for name in ('sforge-submit','sforge-save-version'):expected['/usr/local/bin/'+name]=files['work/commands/'+name]['sha256']
        code="""import hashlib,json,os,stat
from pathlib import Path
assert os.geteuid()==1000
for name in ('sforge-submit','sforge-save-version'):
 p=Path('/usr/local/bin')/name;assert p.stat().st_uid==0 and stat.S_IMODE(p.stat().st_mode)==0o755,name
for target,digest in EXPECTED.items():assert hashlib.sha256(Path(target).read_bytes()).hexdigest()==digest,target
assert not Path('/opt/minference_prefill_judge').exists()
for target in ('/workspace/benchmark/public-prompts','/workspace/benchmark/model'):
 p=Path(target);assert p.stat().st_uid==0 and stat.S_IMODE(p.stat().st_mode)==0o555 and not list(p.iterdir())
for n in ('runtime','activations','numerical_policy.json','public-performance-plan.json','candidate_worker.py'):assert not Path('/workspace/benchmark',n).exists()
for base in ('/workspace/benchmark','/app/workdir'):
 for d,_,ns in os.walk(base):
  assert not any(n=='prompts.jsonl' or n.endswith(('.safetensors','.bin')) for n in ns),d
  assert not set(ns)&{'protected_hashes.txt','quality-policy.json','final_manifest.json','verify_minference_prefill.py','run_judge.py'},d
""".replace('EXPECTED',repr(expected))
    _python(engine,image,SOFTWARE_CODE,user='0:0',label=role+' software presence/absence')
    _python(engine,image,code,user='agent' if role=='work' else '0:0',label=role+' source-only filesystem')
    if role=='judge':
        _python(engine,image,"""from pathlib import Path
for target in ('/opt/minference_prefill_judge/private','/opt/minference_prefill_judge/external-assets','/opt/minference_prefill_judge/model'):
 try:list(Path(target).iterdir())
 except PermissionError:pass
 else:raise AssertionError('candidate-readable future private root')
""",user='agent',label='future private path denial')
    require(_read_stage(source_root,stage_receipt,prequalification)[0]==stage_raw,'stage/current sources changed during CPU image check')
    return {'status':'passed','scope':'thin-image-software-source-metadata-only','image_id':image,'stage_receipt_sha256':sha(stage_raw),'prequalification':prequalification,'qualified':False,'protected_file_count':25,'baked_protected_files':25,'external_assets_authenticated':False,'gpu_execution':False}
def validate_work_image(*,engine,image,image_only=True,source_root=None,stage_receipt=None,prequalification=False):
    require(image_only and source_root is not None and stage_receipt is not None,'new thin source stage receipt required; assets are checked separately')
    return _image_check(engine=engine,image=image,role='work',source_root=source_root,stage_receipt=stage_receipt,prequalification=prequalification)
def validate_judge_image(*,engine,image,image_only=True,source_root=None,stage_receipt=None,prequalification=False):
    require(image_only and source_root is not None and stage_receipt is not None,'new thin source stage receipt required; assets are checked separately')
    return _image_check(engine=engine,image=image,role='judge',source_root=source_root,stage_receipt=stage_receipt,prequalification=prequalification)
