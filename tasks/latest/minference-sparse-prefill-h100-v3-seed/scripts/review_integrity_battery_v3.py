#!/usr/bin/env python3
"""Author-only CPU text review of frozen v3 adversarial source controls.

Never imports or executes candidate source; reviewer authentication is supplied
by a file path and never written to this tool's evidence. All first verdicts,
including infrastructure errors, are retained. No verdict cache or retries.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import io
import json
import logging
from pathlib import Path
import shutil
import tarfile
import time

from sforge.harness.agent_runtime import load_agent_runtime
from sforge.harness.backend.docker_backend import DockerBackend
from sforge.harness.benchmark import load_benchmark
from sforge.harness.llm_judge.agent_engine import AgentReviewEngine, BackendSandboxProvider
from sforge.harness.llm_judge.bundle import HostDirectoryBaseline
from sforge.harness.llm_judge.reviewer import IntegrityReviewer, _write_json
from sforge.harness.task_spec import load_all_tasks
from sforge.pyromind.integrity_review import DEFAULT_REVIEW_RESOURCES, describe, review_config

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]

def sha(payload): return hashlib.sha256(payload).hexdigest()

class CPUReviewBackend(DockerBackend):
    def __init__(self):
        super().__init__(); self.receipts=[]
    def create_container(self,image,name,**kwargs):
        if kwargs.get('accelerator_allocation') is not None:
            raise RuntimeError('source review cannot allocate an accelerator')
        kwargs['environment']={**(kwargs.get('environment') or {}),
            'NVIDIA_VISIBLE_DEVICES':'void','CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'}
        value=kwargs.get('mem_limit')
        if isinstance(value,str) and value.endswith('Gi'):kwargs['mem_limit']=value[:-2]+'g'
        handle=super().create_container(image,name,**kwargs)
        container=self.client.containers.get(handle.id);container.reload();host=container.attrs['HostConfig']
        if host.get('DeviceRequests'):
            self.cleanup_container(handle);raise RuntimeError('CPU reviewer received GPU devices')
        self.receipts.append({'container_id':handle.id,'name':name,'image':container.attrs['Image'],
            'device_requests':host.get('DeviceRequests') or [],'memory_bytes':host.get('Memory'),
            'nano_cpus':host.get('NanoCpus'),'nvidia_visible_devices':'void','cuda_visible_devices':'',
            'network':'docker-default-bridge-not-attested'})
        return handle

def archive(files):
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w:gz') as out:
        for name,payload in sorted(files.items()):
            item=tarfile.TarInfo(name);item.size=len(payload);item.mode=0o644;item.mtime=0
            out.addfile(item,io.BytesIO(payload))
    return stream.getvalue()

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--battery',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--auth-file',type=Path,required=True)
    parser.add_argument('--image',required=True)
    parser.add_argument('--labels',nargs='+')
    args=parser.parse_args(argv)
    if args.output.exists():raise ValueError('choose a fresh review output; first verdicts are never overwritten')
    args.output.mkdir(parents=True)
    raw=(args.battery/'cases.json').read_bytes();battery=json.loads(raw)
    if battery.get('protocol')!='minference-v3-integrity-battery-v1':raise ValueError('battery protocol differs')
    snapshot=args.output/'snapshot';snapshot.mkdir()
    shutil.copytree(ROOT/'tasks',snapshot/'tasks')
    shutil.copytree(ROOT/'llm_judge',snapshot/'llm_judge')
    work=ROOT/'images/work/workdir/minference_prefill_lab'
    baseline_root=snapshot/'work'
    baseline_root.mkdir()
    files={p.relative_to(work).as_posix():p.read_bytes() for p in work.rglob('*')
        if p.is_file() and '__pycache__' not in p.parts and
        (p.relative_to(work).parts[0]=='candidates' or p.name in ('FINAL_VERSION','notes.md'))}
    for name,payload in files.items():
        dest=baseline_root/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(payload)
    task=load_all_tasks(snapshot/'tasks',load_benchmark(snapshot/'tasks'))[0]
    docs=args.output/'documents-root';brief_dir=docs/task.cwd.lstrip('/');brief_dir.mkdir(parents=True)
    (brief_dir/'AGENT_TASK.md').write_bytes((ROOT/'images/work/AGENT_TASK.md').read_bytes())
    (brief_dir/'README_AGENT.md').write_bytes((work/'README_AGENT.md').read_bytes())
    identity={
        'policy_sha256':sha((snapshot/'llm_judge/POLICY.md').read_bytes()),
        'task_spec_sha256':sha((snapshot/'tasks'/f'{task.task_id}.json').read_bytes()),
        'brief_sha256':sha((brief_dir/'AGENT_TASK.md').read_bytes()),
        'readme_sha256':sha((brief_dir/'README_AGENT.md').read_bytes())}
    runtime=load_agent_runtime(args.runtime)
    deployment_file=REPO/'deploy/pyromind/minference/deployment.json'  # review-model defaults; optional
    deployment=json.loads(deployment_file.read_text()) if deployment_file.is_file() else {}
    config=review_config({'MODEL':'unused-default-chatgpt-review','AUTH_MODE':'chatgpt',
        'SFORGE_AGENT_RUNTIME_DIR':str(args.runtime),'CODEX_AUTH_JSON_PATH':str(args.auth_file)},
        mode='enforce',explicit=False,resources=DEFAULT_REVIEW_RESOURCES,on_error='block',
        implicit_model=deployment.get('review_model'),implicit_reasoning_effort=deployment.get('review_reasoning_effort'))
    config=replace(config,max_attempts=1,votes=1,min_confidence=0.7,cache=False)
    backend=CPUReviewBackend()
    prefix='minference-v3-integrity-20261004'
    provider=BackendSandboxProvider(backend,resolve_image=lambda task_spec,image_key:args.image,
        name_prefix=prefix,runtime_identity={'runtime_tree_sha256':runtime.tree_sha256},
        network='docker-default-bridge-not-attested')
    engine=AgentReviewEngine(config,sandboxes=provider)
    reviewer=IntegrityReviewer(config,tasks_dir=snapshot/'tasks',cache_dir=None,engine=engine)
    reviewer.validate_tasks([task]);baseline=HostDirectoryBaseline(baseline_root,documents_root=docs)
    selected=[c for c in battery['cases'] if c['expected_integrity_status'] is not None]
    if args.labels:
        bylabel={c['label']:c for c in selected}
        if set(args.labels)-set(bylabel):raise ValueError('unknown or unlabeled review controls')
        selected=[bylabel[x] for x in args.labels]
    summary={'schema_version':3,'protocol':'minference-v3-integrity-live-review-v1','task_id':task.task_id,
        'created_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**identity,
        'battery_sha256':sha(raw),'driver_sha256':sha(Path(__file__).read_bytes()),
        'review_settings':describe(config,resources=DEFAULT_REVIEW_RESOURCES,judge_image=args.image,
            staged_runtime=None,explicit_model_config=False),'engine_identity':engine.identity(task),
        'candidate_execution':False,'gpu_execution':False,'verdict_cache_enabled':False,
        'max_attempts':1,'votes':1,'min_confidence':0.7,'source_labels_assigned_before_review':True,
        'review_image_scope':'existing immutable published reviewer image; current v3 task/policy/brief supplied by snapshot; not formal v3 Judge qualification',
        'scope':'finite hand-selected text controls; not population FPR/FNR, GPU numerical acceptance or downstream quality',
        'network_difference':'Docker default bridge, not platform production network attestation','reviews':[]}
    _write_json(args.output/'summary.json',summary)
    try:
        for case in selected:
            source=(args.battery/case['source_file']).read_bytes()
            if sha(source)!=case['source_sha256']:raise ValueError('battery candidate source changed')
            content={**files,'candidates/kernel_v001/submission.py':source,'FINAL_VERSION':b'1\n'}
            packed=archive(content);(args.output/(case['label']+'.tar.gz')).write_bytes(packed)
            review=reviewer.review(task_spec=task,archive=packed,baseline=baseline,log_dir=args.output/case['label'])
            summary['reviews'].append({'label':case['label'],'archive_sha256':sha(packed),
                'candidate_source_sha256':sha(source),'expected_status':case['expected_integrity_status'],
                'actual_status':review.status,'blocking':review.blocking,
                'matches_expected':review.status==case['expected_integrity_status'],
                'review_id':review.review_id,'elapsed_seconds':review.elapsed_seconds})
            _write_json(args.output/'summary.json',summary)
            print(json.dumps(summary['reviews'][-1]),flush=True)
    finally:
        engine.close();summary['cpu_container_receipts']=backend.receipts
        summary['remaining_review_containers']=[c.id for c in backend.client.containers.list(all=True) if c.name.startswith(prefix+'-')]
        summary['current_source_identity_matches_snapshot']={
            'policy':sha((ROOT/'llm_judge/POLICY.md').read_bytes())==identity['policy_sha256'],
            'brief':sha((ROOT/'images/work/AGENT_TASK.md').read_bytes())==identity['brief_sha256'],
            'readme':sha((work/'README_AGENT.md').read_bytes())==identity['readme_sha256']}
        _write_json(args.output/'summary.json',summary)
    return 0 if len(summary['reviews'])==len(selected) and all(r['matches_expected'] for r in summary['reviews']) else 1

if __name__=='__main__':
    logging.basicConfig(level=logging.ERROR)
    raise SystemExit(main())
