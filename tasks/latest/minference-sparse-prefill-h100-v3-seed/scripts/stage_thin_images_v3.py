#!/usr/bin/env python3
"""Stage only bounded source/metadata for the full-RULER on-demand images.

This author build helper never copies an asset payload, authenticates a GPU,
or qualifies a release. --prequalification permits a disclosed draft policy;
the executable Judge still requires a frozen policy before it can score.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import uuid

PREFIX='/opt/minference_prefill_judge/'
MAX_FILE=4*1024**2
MAX_CONTEXT=16*1024**2
QA_EXTERNAL={PREFIX+'private/quality-prompts/'+n for n in ('manifest.json','prompts.jsonl')}
SOURCE_KEYS={'quality_worker_sha256':'scripts/full_ruler/quality_worker_full_ruler_v5.py',
 'scorer_sha256':'scripts/full_ruler/ruler_official_constants.py',
 'model_recipe_sha256':'activations/recipe/RULER_V3_RECIPE.json',
 'candidate_api_sha256':'scripts/full_ruler/candidate_attention_fa2_prototype.py',
 'quality_contract_sha256':'scripts/full_ruler/quality_contract_v3.py',
 'reference_parser_sha256':'scripts/full_ruler/ruler_complete_reference_v5.py'}

def require(ok,message):
    if not ok:raise ValueError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def identity(i):return (i.st_dev,i.st_ino,i.st_mode,i.st_uid,i.st_gid,i.st_nlink,i.st_size,i.st_mtime_ns,i.st_ctime_ns)
def canonical(raw):return json.dumps(raw,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def read(path):
    path=Path(path)
    require(path.is_absolute() and '..' not in path.parts and not any(p.is_symlink() for p in (path,*path.parents)),'canonical source path required')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        before=os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink==1 and 0<before.st_size<=MAX_FILE,'bounded single-link source/metadata required')
        with os.fdopen(fd,'rb',closefd=False) as stream:raw=stream.read(MAX_FILE+1)
        require(len(raw)==before.st_size and identity(before)==identity(os.fstat(fd))==identity(path.lstat()),'source changed during read')
        return raw
    finally:os.close(fd)
def parse(raw):
    def unique(items):
        result={}
        for k,v in items:require(k not in result,'duplicate metadata key');result[k]=v
        return result
    return json.loads(raw,object_pairs_hook=unique,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite metadata')))
def module(root,relative,name):
    p=root/relative;read(p)
    spec=importlib.util.spec_from_file_location(name,p);value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value;spec.loader.exec_module(value);return value
def write(path,raw):
    path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
def protected_map(raw,expected):
    result={}
    for line in raw.decode().splitlines():
        require(re.fullmatch(r'[0-9a-f]{64}  /opt/minference_prefill_judge/[^\s]+',line),'malformed protected list')
        digest,path=line.split('  ');require(path not in result,'duplicate protected path');result[path]=digest
    require(set(result)==expected and len(result)==25,'exact protected25 source set required')
    return result
def add(files,target,raw):
    require(target not in files,'duplicate context projection: '+target)
    p=Path(target)
    require(not p.is_absolute() and '..' not in p.parts and p.suffix not in ('.bin','.safetensors') and p.name!='prompts.jsonl','asset payload forbidden in build context')
    require(0<len(raw)<=MAX_FILE,'bounded source/metadata context member required');files[target]=raw
def tree(root,relative,target,files):
    base=root/relative
    require(base.is_dir() and not any(p.is_symlink() for p in (base,*base.parents)),'source directory alias rejected')
    for p in sorted(base.rglob('*')):
        require(not p.is_symlink(),'source tree symlink rejected')
        if p.is_dir() or '__pycache__' in p.parts or p.suffix=='.pyc':continue
        add(files,str(Path(target)/p.relative_to(base)),read(p))

def stage(args):
    root=args.source_root
    require(root.is_absolute() and '..' not in root.parts and not any(p.is_symlink() for p in (root,*root.parents)),'source root must be canonical')
    generated=module(root,'scripts/generate_protected_hashes.py','thin_source_protected')
    expected=set(generated.PROTECTED_LOCAL_PATHS)|set(generated.DYNAMIC_PATHS)
    require(len(expected)==25,'kernel-only protected25 required')
    assets=module(root,'scripts/external_assets_on_demand_v3.py','thin_external_assets')
    archive=module(root,'runtime/single_prompt_archive_v3.py','thin_plan_validator')
    module(root,'runtime/activation_archive_v3.py','activation_archive_v3')
    module(root,'runtime/kernel_fidelity_v3.py','kernel_fidelity_v3')
    runtime=module(root,'runtime/minference_prefill_runtime.py','thin_numerical_validator')
    numeric_raw=read(args.numerical_policy);numeric=parse(numeric_raw)
    runtime.load_numerical_policy(args.numerical_policy,diagnostic=args.prequalification)
    judge_raw=read(args.judge_asset_contract);judge=assets.validate_contract(parse(judge_raw))
    jr={x['role']:x for x in judge['assets']}
    plans={}
    for partition,path,row in [('final',args.final_plan,jr['private-performance'])]:
        raw=read(path);plan=parse(raw);archive.validate_plan(plan)
        require(plan['partition']==partition,'fixed plan partition differs')
        entries={x['path']:x for x in row['files']}
        require(plan['input_bindings']=={'prompt_manifest_sha256':row['native_manifest_sha256'],
            'prompts_sha256':entries['prompts.jsonl']['sha256'],'recipe_sha256':sha(read(root/'activations/recipe/RULER_V3_RECIPE.json'))},'native text/model recipe binding differs')
        plans[partition]=raw
    dynamic={PREFIX+'private/final_manifest.json':plans['final'],PREFIX+'private/numerical_policy.json':numeric_raw,
        PREFIX+'private/external-assets-contract.json':judge_raw}
    listed_raw=read(root/'images/judge/protected_hashes.txt')
    listed=protected_map(listed_raw,expected)
    for target,raw in dynamic.items():require(listed[target]==sha(raw),'protected metadata differs')
    files={}
    for role in ('work','judge'):
        docker=read(root/'images'/role/'Dockerfile').decode()
        require('ARG ON_DEMAND_SOFTWARE_IMAGE\nFROM ${ON_DEMAND_SOFTWARE_IMAGE}' in docker and (role=='work' or '--from=installed_sources' in docker) and 'test ! -e /opt/minference_prefill_judge' in docker and 'uninstall' not in docker,'new source-only FA2 software Dockerfile required')
    commands=parse(read(root/'images/work/commands/source-manifest.json'))
    require(commands.get('protocol')=='sforge-public-generated-commands-v1' and set(commands.get('commands',{}))=={'sforge-submit','sforge-save-version'},'explicit public command provenance required')
    for name,entry in commands['commands'].items():
        raw=read(root/'images/work/commands'/name)
        require(sha(raw)==entry.get('sha256') and len(raw)==entry.get('size') and raw.startswith(b'#!/bin/bash\n'),'public command bytes differ from generated source')
    tree(root,'images/work','work',files);tree(root,'images/judge','judge',files)
    # Use explicit mappings rather than flattening scripts/full_ruler imports.
    for target,relative in generated.PROTECTED_LOCAL_PATHS.items():
        raw=read(root/relative);require(sha(raw)==listed[target],'protected source is stale: '+relative)
        dest=str(Path('installed')/Path(target).relative_to(PREFIX))
        add(files,dest,raw)
    for target,raw in dynamic.items():add(files,str(Path('installed')/Path(target).relative_to(PREFIX)),raw)
    # images/judge context has the authoritative list; do not stage an old list.
    files['judge/protected_hashes.txt']=listed_raw
    add(files,'installed/protected_hashes.txt',listed_raw)
    require((root/'oracle/candidates/kernel_v001/submission.py').read_bytes()==(root/'images/work/workdir/minference_prefill_lab/candidates/kernel_v001/submission.py').read_bytes(),'Work/oracle seed bytes differ')
    require(sum(map(len,files.values()))<=MAX_CONTEXT,'all active build contexts exceed16MiB')
    # No output is created until every prerequisite and byte has passed.
    base=args.build_root
    require(base.is_absolute() and '..' not in base.parts and not any(p.is_symlink() for p in (base,*base.parents)),'build root must be canonical')
    base.mkdir(parents=True,exist_ok=True,mode=0o700)
    require(base.stat().st_uid==os.geteuid(),'build root owner differs');os.chmod(base,0o700)
    output=base/('thin-'+uuid.uuid4().hex);output.mkdir(mode=0o700)
    for name,raw in sorted(files.items()):write(output/name,raw)
    record={'schema_version':2,'protocol':'minference-kernel-thin-image-context-v3','status':'source-metadata-staged',
      'stage':str(output),'prequalification':args.prequalification,'qualified':False,
      'protected_file_count':len(expected),'baked_protected_files':len(expected),'external_qa_protected_files':0,
      'evaluation_kind':'kernel-only-v3','gpu_count':1,'submit_final_same_contract':True,
      'total_bytes':sum(map(len,files.values())),'file_count':len(files),'files':{n:{'sha256':sha(b),'size':len(b)} for n,b in sorted(files.items())},
      'source_root':str(root),'external_assets_authenticated':False,'raw_tensor_files_staged':0,'model_files_staged':0,'terminal_prompt_payloads_staged':0,'gpu_execution':False,
      'numerical_policy_status':numeric['status'],
      'execution_bindings_complete':True,
      'protected_list_authority':'prequalification-source-metadata-only' if args.protected_hashes is None else 'provided-protected-source-manifest'}
    write(output/'staging-receipt.json',canonical(record)+b'\n');return record

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','build-root','final-plan','judge-asset-contract','numerical-policy'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--protected-hashes',type=Path)
    p.add_argument('--prequalification',action='store_true');return p
def main(argv=None):
    args=parser().parse_args(argv);print(json.dumps(stage(args),sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
