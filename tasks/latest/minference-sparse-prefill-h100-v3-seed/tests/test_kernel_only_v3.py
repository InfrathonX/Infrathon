import importlib.util,json,math,sys,tarfile,io
from pathlib import Path
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'runtime'),str(ROOT/'scripts'),str(ROOT/'images/judge')]
from kernel_fidelity_v3 import energy_stats,passes
from source_admission import validate_source_text

def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
L={'nrmse':.25,'head_max_nrmse':.5,'tail_nrmse':.25,'tail_head_max_nrmse':.5}
def test_exact_and_rounding_reference():
 r=torch.randn(1,32,64,128)
 assert passes(energy_stats(torch,r,r),energy_stats(torch,r.to(torch.bfloat16),r),L)
@pytest.mark.parametrize('mutation',['zero','scaled','one-head-missing','nan'])
def test_wrong_outputs_cannot_hide_in_averages(mutation):
 r=torch.ones(1,32,64,128);a=r.clone()
 if mutation=='zero':a.zero_()
 if mutation=='scaled':a*=.6
 if mutation=='one-head-missing':a[:,0]=0
 if mutation=='nan':a[:,:,0,0]=float('nan')
 s=energy_stats(torch,a,r);assert not passes(s,s,L)
def test_zero_reference_not_a_free_pass():
 r=torch.zeros(1,32,64,128)
 assert passes(energy_stats(torch,r,r),energy_stats(torch,r,r),L)
 s=energy_stats(torch,r+.001,r);assert not passes(s,s,L)
def test_tail_independent_gate():
 r=torch.ones(1,32,64,128);u=energy_stats(torch,r,r);t=energy_stats(torch,r*.6,r)
 assert not passes(u,t,L)
def test_exact_protected_kernel_sources():
 from generate_protected_hashes import PROTECTED_LOCAL_PATHS,DYNAMIC_PATHS,PROTECTED_FILE_COUNT
 assert PROTECTED_FILE_COUNT==25
 assert all('full_ruler' not in p and 'quality-prompts' not in p for p in list(PROTECTED_LOCAL_PATHS)+list(DYNAMIC_PATHS))
def test_two_roles_no_model_qa():
 from external_assets_on_demand_v3 import validate_contract
 v=validate_contract(json.loads((ROOT/'qualification/external-assets-contract.json').read_bytes()))
 assert {r['role'] for r in v['assets']}=={'model','private-performance'}
 bad=json.loads(json.dumps(v));bad['assets'].append({'role':'terminal-quality'})
 with pytest.raises(ValueError):validate_contract(bad)
def test_starter_static_policy_and_mirror():
 a=(ROOT/'oracle/candidates/kernel_v001/submission.py').read_bytes();assert a==(ROOT/'images/work/workdir/minference_prefill_lab/candidates/kernel_v001/submission.py').read_bytes()
 assert validate_source_text(a.decode(),'submission.py')==[]
def records():
 rows=[]
 for t in ('niah_single_2','niah_multikey_1','vt','fwe'):
  for n in (32768,65536,131072):
   for layer in (1,4,15,27):
    ns=[n-128,n-129,n-130];sec=[.01,.012,.009];v=[8192*x*(x+1)/s/1e12 for x,s in zip(ns,sec)]
    rows.append({'task':t,'context_budget':n,'layer':layer,'trial_ns':ns,'trial_seconds':sec,'median_tflops':__import__('statistics').median(v)})
 score=math.exp(sum(math.log(r['median_tflops']) for r in rows)/48)
 from run_judge import GATES
 v={'protocol':'minference-real-ruler-prefill-v3','evaluation_kind':'kernel-only-v3','case_count':48,'boundary_case_count':11,'score_enabled':1,'provisional':False,'raw_primary_tflops':score,'reward':score,**{k:1 for k in GATES}}
 return dict(v),{**v,'cases':rows}
def test_no_qa_fields_required_for_same_kernel_score():
 from run_judge import build_structured_result
 r,d=records();v=build_structured_result(r,d,verifier_completed=True,verifier_returncode=0);assert v['valid'] and 'qa_ok' not in v['metrics']
@pytest.mark.parametrize('damage',['missing-group','missing-boundaries','duplicates','inflated-score','old-qa-protocol','worker-failure'])
def test_actual_adapter_rejects_invalid_or_old_results(damage):
 from run_judge import build_structured_result
 r,d=records();rc=0
 if damage=='missing-group':d['cases'].pop()
 if damage=='missing-boundaries':d.pop('boundary_case_count')
 if damage=='duplicates':d['cases'][0]=d['cases'][1]
 if damage=='inflated-score':r['reward']*=2
 if damage=='old-qa-protocol':d.pop('evaluation_kind')
 if damage=='worker-failure':rc=1
 assert not build_structured_result(r,d,verifier_completed=True,verifier_returncode=rc)['valid']
def test_declared_resource_and_task_bootstrap():
 d=json.loads((ROOT/'tasks/minference_sparse_prefill_h100_seed_v3.json').read_bytes())
 assert d['judge']['private_bootstrap']=='minference-kernel-prefill-v3' and d['judge']['cpu_limit']==8 and d['judge']['mem_limit']=='48g'
 assert 'downstream_quality_scope' not in d['portable']['task_core']['definition']

@pytest.mark.parametrize('prefix',['GPU-',''])
def test_control_capture_accepts_torch_uuid_representation(prefix):
 from types import SimpleNamespace
 from run_on_demand_integrity_battery_v3 import validate_capture,RUNTIME_FILES
 gpu='GPU-6e5ab7c8-39bf-df56-533a-729fb2bcc77b';image='sha256:'+'a'*64
 sources={'runtime/'+f:'b'*64 for f in RUNTIME_FILES}
 sources.update({'activations/recipe/capture_activations.py':'c'*64,'runtime/on_demand_suite_v3.py':'d'*64})
 bindings={'image_id':image,'model':{'revision':'test'},'attention_backend':'flash_attention_2','flash_attention_version':'2.8.3',
  'capture_helper_sha256':sources['activations/recipe/capture_activations.py'],'on_demand_suite_sha256':sources['runtime/on_demand_suite_v3.py'],
  'on_demand_sources':{f:sources['runtime/'+f] for f in RUNTIME_FILES}}
 plan={'protocol':'test','partition':'final','bindings':bindings,'fixtures':{}}
 single=SimpleNamespace(PROTOCOL='test',validate_bundle=lambda b,p:None)
 bundle={'purpose':'private-judge','gpu_uuid':prefix+gpu.removeprefix('GPU-')}
 validate_capture(bundle,plan,single,sources,image_id=image,gpu_uuid=gpu,recipe={'model':bindings['model']})
 bundle['gpu_uuid']='GPU-'+'0'*36
 with pytest.raises(ValueError):validate_capture(bundle,plan,single,sources,image_id=image,gpu_uuid=gpu,recipe={'model':bindings['model']})


@pytest.mark.parametrize('damage',[None,'missing','duplicate','head-omission','wrong-rows','numeric-flag','global-budget','nan'])
def test_boundary_coverage_and_independent_fidelity(damage):
 from kernel_fidelity_v3 import BOUNDARY_LENGTHS,validate_boundary_checks
 checks=[]
 for n in BOUNDARY_LENGTHS:
  stat={'finite':True,'nrmse':.01,'head_max_nrmse':.01,'head_nrmse':[.01]*32}
  checks.append({'n':n,'numeric_passed':True,'full_write_ok':True,'input_immutable_ok':True,
    'uniform_stats':{**stat,'rows':32*n},'tail_stats':{**stat,'rows':32*min(n,64)}})
 if damage=='missing':checks.pop()
 if damage=='duplicate':checks[-1]=checks[0]
 if damage=='head-omission':checks[0]['uniform_stats']['head_nrmse']=[]
 if damage=='wrong-rows':checks[0]['uniform_stats']['rows']=1
 if damage=='numeric-flag':checks[0]['numeric_passed']=False
 if damage=='global-budget':checks[0]['uniform_stats']['nrmse']=.26
 if damage=='nan':checks[0]['uniform_stats']['nrmse']=float('nan')
 if damage is None:assert validate_boundary_checks(checks,L)==checks
 else:
  with pytest.raises(ValueError):validate_boundary_checks(checks,L)

@pytest.mark.parametrize('phase',['interactive','final'])
def test_root_success_records_reach_actual_adapter(phase):
 verifier=module('kernel_root_success_test',ROOT/'images/judge/verify_minference_prefill.py')
 from run_judge import build_structured_result
 _,d=records()
 derived={**d,'hardware':{},'candidate_sha256':'a'*64,'manifest_sha256':'b'*64,'numerical_policy_sha256':'c'*64,
          'length_geomean_tflops':{},'family_geomean_tflops':{}}
 reward,detail,rc=verifier.build_result_records(derived,final_version=1,num_candidate_versions=1,evaluation_phase=phase)
 actual=build_structured_result(reward,detail,verifier_completed=True,verifier_returncode=rc)
 assert actual['valid'] and math.isclose(actual['score'],derived['raw_primary_tflops'],rel_tol=1e-9)

def test_retain_literal_worker_report_and_reject_changed_bytes(tmp_path):
 verifier=module('kernel_report_retention_test',ROOT/'images/judge/verify_minference_prefill.py')
 source=tmp_path/'worker.json';destination=tmp_path/'retained.json'
 raw=b'{ "measured": true, "cases": 48 }\n';source.write_bytes(raw)
 parsed,digest=verifier._retain_worker_report(source,destination,expected_payload={'measured':True,'cases':48})
 assert destination.read_bytes()==raw and destination.stat().st_mode&0o777==0o600
 assert digest==__import__('hashlib').sha256(raw).hexdigest()
 source.write_bytes(b'{"measured":true,"cases":47}')
 with pytest.raises(RuntimeError):verifier._retain_worker_report(source,destination,expected_payload=parsed)
 assert destination.read_bytes()==raw

@pytest.mark.parametrize('name',['evaluate_local_v3.py','evaluate_submit_cli_local_v3.py'])
@pytest.mark.parametrize('probe_fails',[False,True])
def test_native_cleanup_honors_framework_logger_and_probe_is_optional(tmp_path,name,probe_fails):
 import ast
 from types import SimpleNamespace
 class FakeBackend:
  def exec_run(self,*args,**kwargs):
   if probe_fails:raise RuntimeError('resource observation unavailable')
   return SimpleNamespace(exit_code=0,output=json.dumps({'peak_bytes':1000,'events':'oom 0'}))
  def cleanup_container(self,handle,logger=None):self.cleaned=(handle,logger)
 tree=ast.parse((ROOT/'scripts'/name).read_text())
 cls=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='Backend')
 ns={'__name__':'native_cleanup_contract','DockerBackend':FakeBackend,'json':json,
     'a':SimpleNamespace(output=tmp_path,image='sha256:'+'a'*64,gpu='test-allocation')}
 exec(compile(ast.Module(body=[cls],type_ignores=[]),name,'exec'),ns)
 backend=ns['Backend']();logger=object();backend.cleanup_container('owned-handle',logger)
 assert backend.cleaned==('owned-handle',logger)
 usage=json.loads((tmp_path/'resource-usage.json').read_bytes())
 assert usage['status']=='unavailable' if probe_fails else usage['peak_bytes']==1000
