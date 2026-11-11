#!/usr/bin/env python3
"""Local installed sforge-submit HTTP -> live source review -> kernel Judge.

Only the allocated single GPU is used by Judge. The CLI and source reviewer
have no GPU. This validates local transport, not remote platform publication.
"""
import argparse,hashlib,importlib.util,json,logging,os,socket,subprocess,sys,threading,time,uuid
from pathlib import Path
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[2];sys.path.insert(0,str(REPO))

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def main():
 p=argparse.ArgumentParser();p.add_argument('--judge-image',required=True);p.add_argument('--work-image',required=True)
 p.add_argument('--review-image',required=True);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--auth-file',type=Path,required=True)
 p.add_argument('--gpu',required=True);p.add_argument('--asset-bindings',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--wait-container');a=p.parse_args()
 assert a.output.is_absolute() and not a.output.exists();a.output.mkdir(mode=0o700)
 import docker,uvicorn
 from sforge.harness.backend.docker_backend import DockerBackend
 from sforge.harness.backend.base import AcceleratorAllocation
 from sforge.harness.config import SForgeConfig
 from sforge.harness.run_evaluation import judge_submission
 from sforge.harness.judge_server import create_app
 from sforge.harness.llm_judge.agent_engine import AgentReviewEngine,BackendSandboxProvider
 from sforge.harness.llm_judge.bundle import HostDirectoryBaseline
 from sforge.harness.agent_runtime import load_agent_runtime
 from sforge.pyromind.integrity_review import review_config,DEFAULT_REVIEW_RESOURCES
 assets=load('cli_assets',ROOT/'scripts/external_assets_on_demand_v3.py');contract=json.loads((ROOT/'qualification/external-assets-contract.json').read_bytes())
 bindings=assets.validate_bindings(json.loads(a.asset_bindings.read_bytes()),contract)
 bootstrap=load('cli_kernel_bootstrap',next(p for p in (ROOT/'scripts/bootstrap/bootstrap.py',REPO/'deploy/pyromind/minference-v3/bootstrap.py') if p.is_file()))
 class Backend(DockerBackend):
  def create_container(self,*args,**kwargs):
   kwargs['mounts']=[(r['source_directory'],r['mount_path'],'ro') for r in bindings['assets']]
   return super().create_container(*args,**kwargs)
  def start_container(self,h):super().start_container(h);bootstrap.install_private_judge(self,h,ROOT)
  def cleanup_container(self,h,logger=None):
   try:
    result=self.exec_run(h,['python3','-I','-c',"import json,pathlib;r=pathlib.Path('/sys/fs/cgroup');print(json.dumps({'peak_bytes':int((r/'memory.peak').read_text()),'events':(r/'memory.events').read_text()}))"],user='root')
    if result.exit_code==0:(a.output/'resource-usage.json').write_text(result.output)
   except Exception as error:(a.output/'resource-usage.json').write_text(json.dumps({'status':'unavailable','error_type':type(error).__name__}))
   finally:super().cleanup_container(h,logger)
 backend=Backend(docker.from_env());rid='kernel-submit-cli-'+uuid.uuid4().hex[:12];receipt={'protocol':'minference-standard-single-GPU-local-v3','phase':'interactive','transport':'installed-sforge-submit-HTTP-with-enforced-live-review','image_id':a.judge_image,'gpu_uuid':a.gpu,'qualified':False,'valid':False}
 def evaluator(**kwargs):
  if a.wait_container:
   try:
    prior=backend.client.containers.get(a.wait_container);prior.reload()
    req=prior.attrs['HostConfig']['DeviceRequests']
    assert prior.attrs['Image']==a.judge_image and prior.name.startswith('minference_sparse_prefill.judge.minference_sparse_prefill_h100_seed_v3.kernel-local-final') and len(req)==1 and req[0]['DeviceIDs']==[a.gpu]
    while True:
     try:
      prior.reload()
      if not prior.attrs['State']['Running']:break
     except docker.errors.NotFound:break
     time.sleep(1)
   except docker.errors.NotFound:pass
  assert a.gpu not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
  receipt['archive_sha256']=hashlib.sha256(kwargs['archive']).hexdigest()
  kwargs.update(backend=backend,judge_image_ref=a.judge_image,accelerator_allocation=AcceleratorAllocation('nvidia',(a.gpu,)),evaluation_phase='interactive',timeout=7200)
  report=judge_submission(**kwargs);receipt.update(valid=report.valid,report=report.to_dict(),status='standard-judge-complete')
  (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2,default=str)+'\n')
  return report
 review_driver=load('cli_cpu_review_backend',ROOT/'scripts/review_integrity_battery_v3.py')
 deployment_file=REPO/'deploy/pyromind/minference/deployment.json'  # review-model defaults; optional
 deploy=json.loads(deployment_file.read_bytes()) if deployment_file.is_file() else {}
 rc=review_config({'MODEL':'unused-default-chatgpt-review','AUTH_MODE':'chatgpt','SFORGE_AGENT_RUNTIME_DIR':str(a.runtime),'CODEX_AUTH_JSON_PATH':str(a.auth_file)},mode='enforce',explicit=False,resources=DEFAULT_REVIEW_RESOURCES,on_error='block',implicit_model=deploy.get('review_model'),implicit_reasoning_effort=deploy.get('review_reasoning_effort'))
 rc=replace(rc,max_attempts=1,votes=1,cache=False,min_confidence=.7)
 config=SForgeConfig(tasks_dir=ROOT/'tasks',log_dir=a.output/'logs',judge_cpu_limit=8,judge_mem_limit='48g',judge_network_mode='none',llm_judge=rc)
 app=create_app(config,submission_evaluator=evaluator);state=app.state.judge_state;state.backend=backend
 cpu=review_driver.CPUReviewBackend();runtime=load_agent_runtime(a.runtime)
 provider=BackendSandboxProvider(cpu,resolve_image=lambda task,key:a.review_image,name_prefix='minference-cli-review-'+rid,runtime_identity={'runtime_tree_sha256':runtime.tree_sha256},network='local-Docker-bridge')
 state.configure_integrity_review(AgentReviewEngine(rc,sandboxes=provider))
 docs=a.output/'documents';docs.mkdir();(docs/'AGENT_TASK.md').write_bytes((ROOT/'images/work/AGENT_TASK.md').read_bytes());(docs/'README_AGENT.md').write_bytes((ROOT/'images/work/workdir/minference_prefill_lab/README_AGENT.md').read_bytes())
 state.set_run_baseline(rid,HostDirectoryBaseline(ROOT/'images/work/workdir/minference_prefill_lab',documents_root=docs))
 token=state.register_session('minference_sparse_prefill_h100_seed_v3',rid,judge_cpu_limit=8,judge_mem_limit='48g',max_agent_submissions=1,submission_cooldown=0)
 sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
 server=uvicorn.Server(uvicorn.Config(app,log_level='error',access_log=False));thread=threading.Thread(target=lambda:server.run(sockets=[sock]),daemon=True);thread.start()
 while not server.started:time.sleep(.05)
 c=None;nonce=uuid.uuid4().hex
 try:
  c=backend.client.containers.run(a.work_image,['sleep','86400'],detach=True,user='agent',name='minference-cli-'+nonce[:10],labels={'mlsysforge.kernel-owner':nonce},network_mode='host',nano_cpus=1_000_000_000,mem_limit='1g',environment={'NVIDIA_VISIBLE_DEVICES':'void','CUDA_VISIBLE_DEVICES':'','SFORGE_JUDGE_URL':f'http://127.0.0.1:{port}','SFORGE_TOKEN':token,'SFORGE_PATCH_DIR':'/app/workdir/minference_prefill_lab','SFORGE_SUBMIT_PATHS':'FINAL_VERSION notes.md candidates','SFORGE_SUBMIT_POLL_ATTEMPTS':'1440'})
  c.reload();assert c.attrs['Image']==a.work_image and not c.attrs['HostConfig']['DeviceRequests']
  receipt['cli_container_id']=c.id
  with (a.output/'cli.stdout.txt').open('wb') as out,(a.output/'cli.stderr.txt').open('wb') as err:
   result=subprocess.run(['docker','exec','--user','agent',c.id,'sforge-submit','--details'],stdout=out,stderr=err,timeout=7500)
  receipt['cli_returncode']=result.returncode
  observations=[]
  for sid,s in state.submissions.items():
   review=s.get('integrity_review');observations.append({'submission_id':sid,'status':s.get('status'),'integrity_review':review.history_summary() if review is not None else None})
  (a.output/'server-submissions.json').write_text(json.dumps(observations,indent=2,default=str))
  receipt['live_review_passed']=len(observations)==1 and observations[0]['integrity_review'] is not None and observations[0]['integrity_review']['status']=='pass'
  receipt['valid']=receipt['valid'] and result.returncode==0 and receipt['live_review_passed']
 finally:
  if c:
   c.reload();assert c.attrs['Image']==a.work_image and c.attrs['Config']['Labels']['mlsysforge.kernel-owner']==nonce;c.stop(timeout=1);c.remove();receipt['owned_cli_container_removed']=True
  server.should_exit=True;thread.join(timeout=15);state.close_integrity_review();receipt['review_cpu_containers']=cpu.receipts
  (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2,default=str)+'\n');(a.output/'receipt.json').chmod(0o600)
 print(json.dumps({'valid':receipt['valid'],'phase':'interactive','cli_returncode':receipt.get('cli_returncode'),'live_review_passed':receipt.get('live_review_passed')}))
 return 0 if receipt['valid'] else 1
if __name__=='__main__':logging.basicConfig(level=logging.ERROR);raise SystemExit(main())
