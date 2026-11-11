#!/usr/bin/env python3
"""Run the standard SForge Judge API on one explicitly allocated local GPU.

No model QA or alternate kernel-only shortcut: both interactive/final call
judge_submission, including archive admission, task bootstrap, result parsing
and private artifact collection. GPU allocation is externally authorized.
"""
import argparse,hashlib,importlib.util,json,os,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[2];sys.path.insert(0,str(REPO))

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--image',required=True);p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--gpu',required=True);p.add_argument('--asset-bindings',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=('interactive','final'),default='final')
    p.add_argument('--validate-only',action='store_true');a=p.parse_args(argv)
    assert re.fullmatch('sha256:[0-9a-f]{64}',a.image) and re.fullmatch('GPU-[0-9a-f-]{36}',a.gpu)
    assert a.output.is_absolute() and not a.output.exists();a.output.mkdir(mode=0o700)
    spec=importlib.util.spec_from_file_location('kernel_assets',ROOT/'scripts/external_assets_on_demand_v3.py');assets=importlib.util.module_from_spec(spec);spec.loader.exec_module(assets)
    contract=json.loads((ROOT/'qualification/external-assets-contract.json').read_bytes())
    bindings=assets.validate_bindings(json.loads(a.asset_bindings.read_bytes()),contract)
    from sforge.harness.task_spec import make_task_spec
    from sforge.harness.benchmark import load_benchmark
    task=make_task_spec(ROOT/'tasks/minference_sparse_prefill_h100_seed_v3.json',load_benchmark(ROOT/'tasks'))
    receipt={'protocol':'minference-standard-single-GPU-local-v3','phase':a.phase,'image_id':a.image,'gpu_uuid':a.gpu,
        'archive_sha256':hashlib.sha256(a.archive.read_bytes()).hexdigest(),'qualified':False,'valid':False}
    if a.validate_only:
        receipt['status']='CPU metadata validated; no container/GPU';(a.output/'receipt.json').write_text(json.dumps(receipt));return 0
    clients=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    if any(line.split(',')[0].strip().lower()==a.gpu.lower() for line in clients.splitlines()):raise RuntimeError('allocated GPU has another CUDA client; no foreign process changed')
    import docker
    from sforge.harness.backend.docker_backend import DockerBackend
    from sforge.harness.backend.base import AcceleratorAllocation
    from sforge.harness.config import SForgeConfig
    from sforge.harness.run_evaluation import judge_submission
    # The task keeps the controller bootstrap in scripts/bootstrap/ (public tree) or deploy/pyromind/minference-v3/.
    bootstrap_path=next(p for p in (ROOT/'scripts/bootstrap/bootstrap.py',REPO/'deploy/pyromind/minference-v3/bootstrap.py') if p.is_file())
    bs=importlib.util.spec_from_file_location('kernel_task_bootstrap',bootstrap_path);bootstrap=importlib.util.module_from_spec(bs);bs.loader.exec_module(bootstrap)
    class Backend(DockerBackend):
        def create_container(self,*args,**kwargs):
            kwargs['mounts']=[(r['source_directory'],r['mount_path'],'ro') for r in bindings['assets']]
            return super().create_container(*args,**kwargs)
        def start_container(self,handle):
            super().start_container(handle)
            bootstrap.install_private_judge(self,handle,ROOT)
        def cleanup_container(self,handle,logger=None):
            try:
                code="import json,pathlib; r=pathlib.Path('/sys/fs/cgroup'); print(json.dumps({'peak_bytes':int((r/'memory.peak').read_text()),'events':(r/'memory.events').read_text()}))"
                sample=self.exec_run(handle,['python3','-I','-c',code],user='root')
                if sample.exit_code==0:
                    usage=json.loads(sample.output);usage.update(image_id=a.image,gpu_uuid=a.gpu,host_limit_bytes=48*1024**3,cpu_limit=8)
                    (a.output/'resource-usage.json').write_text(json.dumps(usage,indent=2)+'\n')
            except Exception as error:
                (a.output/'resource-usage.json').write_text(json.dumps({'status':'unavailable','error_type':type(error).__name__}))
            finally:super().cleanup_container(handle,logger)
    backend=Backend(docker.from_env());config=SForgeConfig(judge_cpu_limit=8,judge_mem_limit='48g',judge_network_mode='none')
    report=judge_submission(task,a.archive.read_bytes(),config,backend,submission_id='kernel-local-'+a.phase,
        log_dir=a.output/'logs',judge_image_ref=a.image,accelerator_allocation=AcceleratorAllocation('nvidia',(a.gpu,)),
        evaluation_phase=a.phase,round_id='agent-1' if a.phase=='interactive' else 'final-1',timeout=7200)
    result=report.model_dump() if hasattr(report,'model_dump') else __import__('dataclasses').asdict(report)
    receipt.update(status='standard-judge-complete',valid=bool(result.get('valid')),report=result)
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2,default=str)+'\n');(a.output/'receipt.json').chmod(0o600)
    print(json.dumps({'valid':receipt['valid'],'phase':a.phase,'score':result.get('score')}))
    return 0 if receipt['valid'] else 1
if __name__=='__main__':raise SystemExit(main())
