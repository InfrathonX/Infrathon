import hashlib
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from sforge.pyromind.deployment import Deployment, validate_mount_separation, qualification_contract_digest, private_payload_identity
from sforge.pyromind.qualification import (positive_result, _qualify_task,
    native_requalification_spec, install_checkpoint_capture, validate_checkpoint_archive, check_native_report)
from sforge.pyromind.runner import (private_context_environment, task_submission_evaluator,
    configure_judge_runtime, cancel_task_evaluator, close_task_evaluator)


@pytest.mark.parametrize('record,expected', [
    ({'valid':True,'score':1},True),
    ({'valid':False,'score':1},False),
    ({'valid':True,'score':0},False),
    ({'valid':True,'score':True},False),
    ({'valid':True,'score':float('nan')},False),
    ({'valid':True,'score':2,'metrics':{'evaluation_status':'infrastructure_error'}},False),
])
def test_positive_control_requires_a_real_success(record, expected):
    assert positive_result(record) is expected


def test_private_binding_cannot_be_shared_or_unpinned(tmp_path):
    manifest='a'*64
    files=[{'path':'fixture.bin','size':3,'sha256':'b'*64}]
    binding={'mount_path':'/opt/private/fixtures','object_prefix':f'sforge-assets/v2/judge-private/fixtures/{manifest}/files',
             'manifest_sha256':manifest,'files':files,'read_only':True,'visibility':'private','consumers':['judge']}
    identity={'manifest_sha256':manifest,'files_sha256':hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
    path=tmp_path/'bindings.json';path.write_text(json.dumps({'schema_version':1,'assets':[binding]}))
    config={'private_asset_requirements':[{'mount_path':binding['mount_path']}],
            'private_asset_identities':{binding['mount_path']:identity}}
    deployment=Deployment(tmp_path/'profile.json',tmp_path,config)
    assert deployment.judge_asset_bindings(path)==[binding]
    with pytest.raises(ValueError,match='authorized role'):
        deployment.asset_bindings(path)
    binding['object_prefix']=f'sforge-assets/v2/shared/fixtures/{manifest}/files'
    path.write_text(json.dumps({'schema_version':1,'assets':[binding]}))
    with pytest.raises(ValueError,match='judge-private'):
        deployment.judge_asset_bindings(path)
    with pytest.raises(ValueError,match='overlap'):
        validate_mount_separation([{'mount_path':'/opt'}],[{'mount_path':'/opt/private'}])


@pytest.mark.parametrize('change',['none','description','image','resource','private','environment','non-release'])
def test_agent_launch_refuses_stale_gpu_receipts(tmp_path,monkeypatch,change):
    config={'task_id':'fixture','requires_gpu_qualification':False,'images_file':'images.json',
            'task_directory':'task','resources':{'gpu':1,'cpu':4,'memory':'8Gi'},'runtime_expectations':{'torch':'locked'}}
    images={'worker':'repo@sha256:'+'a'*64,'judge':'repo@sha256:'+'b'*64}
    metadata={'files':[{'destination':'/private/verifier.py','sha256':'c'*64}]}
    receipt={'status':'passed','task_id':'fixture','images':dict(images),
             'resources_each_role':dict(config['resources']),'runtime_expectations':config['runtime_expectations'],
             'runtime_contract_sha256':qualification_contract_digest(config),
             'private_payload_manifest_sha256':hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest()}
    if change=='description':config['evaluation_description']='now verified'
    if change=='image':images['judge']='repo@sha256:'+'d'*64
    if change=='resource':config['resources']['cpu']=8
    if change=='private':metadata['files'][0]['sha256']='d'*64
    if change=='environment':config['judge_environment']={'UNTESTED_MODE':'1'}
    if change=='non-release':receipt['status']='non-release-passed'
    (tmp_path/'images.json').write_text(json.dumps(images))
    (tmp_path/'gpu-qualification.json').write_text(json.dumps(receipt))
    monkeypatch.setattr(Deployment,'bootstrap',lambda self:SimpleNamespace(private_payload_metadata=lambda root:metadata))
    deployment=Deployment(tmp_path/'deployment.json',tmp_path,config)
    deployment.assert_launch_ready(preflight=True)
    if change in ('none','description'):deployment.assert_launch_ready()
    else:
        with pytest.raises(ValueError,match='stale'):deployment.assert_launch_ready()


@pytest.mark.parametrize('change', ['path', 'hash', 'mode', 'destination', 'projection', 'version', 'missing'])
def test_portable_private_receipt_binds_installed_content(tmp_path, monkeypatch, change):
    metadata = {'files': [{'source': '/old/controller/../inputs/final.json',
                          'destination': '/private/final.json', 'mode': 0o400, 'sha256': 'a' * 64}],
                'projection': {'target': '/private/data'}}
    config = {'task_id': 'test', 'requires_gpu_qualification': False, 'images_file': 'images.json',
              'task_directory': 'task', 'resources': {}, 'runtime_expectations': {}}
    images = {'judge': 'image@sha256:pin'}
    receipt = {'status': 'passed', 'task_id': 'test', 'images': images, 'resources_each_role': {},
               'runtime_expectations': {}, 'runtime_contract_sha256': qualification_contract_digest(config),
               'private_payload_identity': private_payload_identity(metadata),
               'private_payload_manifest_sha256': hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()}
    metadata['files'][0]['source'] = '/new/controller/inputs/final.json'
    if change == 'hash': metadata['files'][0]['sha256'] = 'b' * 64
    if change == 'mode': metadata['files'][0]['mode'] = 0o444
    if change == 'destination': metadata['files'][0]['destination'] = '/different/final.json'
    if change == 'projection': metadata['projection']['target'] = '/different/data'
    if change == 'version': receipt['private_payload_identity']['schema_version'] = 2
    if change == 'missing': receipt['private_payload_identity'] = None
    (tmp_path / 'images.json').write_text(json.dumps(images))
    (tmp_path / 'gpu-qualification.json').write_text(json.dumps(receipt))
    monkeypatch.setattr(Deployment, 'bootstrap', lambda self: SimpleNamespace(private_payload_metadata=lambda root: metadata))
    deployment = Deployment(tmp_path / 'deployment.json', tmp_path, config)
    if change == 'path': deployment.assert_launch_ready()
    else:
        with pytest.raises(ValueError, match='stale'): deployment.assert_launch_ready()


def test_context_hook_uses_private_channel_and_command_hook_keeps_public_route():
    module=SimpleNamespace(private_evaluation_context=lambda *args:{'COHORT_SECRET':'private-value'},
                           evaluation_command=lambda root,cmd:'trusted-wrapper '+cmd)
    assert private_context_environment(module,Path('/task'),'run',Path('/out')) == {
        'SFORGE_PRIVATE_CONTEXT_COHORT_SECRET':'private-value'}
    seen=[]
    evaluate=task_submission_evaluator(module,Path('/task'),lambda **kwargs:seen.append(kwargs))
    task=SimpleNamespace(judge=SimpleNamespace(eval_cmd='original-judge'))
    evaluate(task_spec=task)
    evaluate(task_spec=task,eval_cmd='original-public')
    assert [item['eval_cmd'] for item in seen]==['trusted-wrapper original-judge','original-public']


def test_controller_evaluator_factory_owns_native_lifecycle_and_keeps_public_command():
    observed = []
    def default(**_kwargs):
        raise AssertionError('The generic Judge must not start before native prewarm')
    def factory(task_root, fallback, **context):
        assert task_root == Path('/task') and fallback is default
        assert context['run_id'] == 'owned-run'
        assert context['controller_origin'] == 'http://controller:8081'
        def selected(**request):
            observed.append(request)
            return 'native-report'
        return selected
    selected = task_submission_evaluator(SimpleNamespace(submission_evaluator=factory),
        Path('/task'), default, run_id='owned-run', controller_origin='http://controller:8081')
    assert selected(eval_cmd='original-public',mounted_archive='saved-version') == 'native-report'
    assert observed == [{'eval_cmd':'original-public','mounted_archive':'saved-version'}]


def test_invalid_controller_evaluator_factory_fails_before_judge_allocation():
    module = SimpleNamespace(submission_evaluator=lambda *_args, **_kwargs: None)
    with pytest.raises(ValueError, match='callable'):
        task_submission_evaluator(module,Path('/task'),lambda **_kwargs: None)


@pytest.mark.parametrize('failure', [None, 'unsafe', 'missing-mirror'])
def test_native_factory_must_validate_real_worker_projection_before_acceptance(failure):
    observed = []
    class Selected:
        def __call__(self, **_kwargs):
            raise AssertionError('Projection validation must precede any submission')
        def assert_private_projection(self, root):
            observed.append(('projection', root))
            if failure == 'unsafe': raise ValueError('Private evidence overlaps Worker artifacts')
        def cancel(self): observed.append('cancel')
        def close(self, **kwargs):
            observed.append(('close',kwargs));return {'cleaned': True}
    evaluator = Selected()
    module = SimpleNamespace(submission_evaluator=lambda *_args, **_kwargs: evaluator)
    context = {} if failure == 'missing-mirror' else {'mirror': SimpleNamespace(output=Path('/actual/run'))}
    if failure:
        with pytest.raises(ValueError):
            task_submission_evaluator(module,Path('/task'),lambda **_kwargs: None,**context)
        assert observed[-2:] == ['cancel',('close',{'timeout':300})]
    else:
        assert task_submission_evaluator(module,Path('/task'),lambda **_kwargs: None,**context) is evaluator
        assert observed == [('projection',Path('/actual/run'))]


def test_native_evaluator_cleanup_requires_confirmation_and_omits_private_receipt():
    observed = []
    evaluator = SimpleNamespace(
        cancel=lambda: observed.append('cancel'),
        close=lambda **kwargs: observed.append(kwargs) or {
            'cleaned': True, 'private_capability': 'must-stay-private'})
    cancel_task_evaluator(evaluator)
    result = close_task_evaluator(evaluator,timeout=300)
    assert observed == ['cancel', {'timeout': 300}]
    assert result == {'cleanup_hook_used': True, 'cleaned': True}
    assert 'private_capability' not in result
    evaluator.close = lambda **_kwargs: {'cleaned': False, 'remaining': ['sb-owned']}
    with pytest.raises(RuntimeError, match='not independently confirmed'):
        close_task_evaluator(evaluator)


def test_existing_evaluator_without_owned_lifecycle_retains_default_cleanup():
    evaluator = lambda **kwargs: kwargs
    cancel_task_evaluator(evaluator)
    assert close_task_evaluator(evaluator) == {'cleanup_hook_used': False}


@pytest.mark.parametrize('collector_fails',[False,True])
def test_private_diagnostics_stay_private_and_never_prevent_cleanup(tmp_path,monkeypatch,collector_fails):
    import sforge.pyromind.runner as runner
    calls=[]
    def collector(backend,handle,root,directory):
        calls.append('diagnostics')
        assert directory == tmp_path/'.private/diagnostics/sb-fixture'
        assert (directory.stat().st_mode & 0o777) == 0o700
        assert ((tmp_path/'.private').stat().st_mode & 0o777) == 0o700
        if collector_fails:raise RuntimeError('private output must not enter error record')
        (directory/'worker.log').write_text('private details')
    backend=SimpleNamespace(cleanup_container=lambda *args:calls.append('delete'))
    module=SimpleNamespace(install_private_judge=lambda *args:None,collect_private_diagnostics=collector)
    deployment=SimpleNamespace(resources_for_role=lambda role:{})
    monkeypatch.setattr(runner,'capture_host_memory',lambda *args:calls.append('usage'))
    configure_judge_runtime(backend,deployment,module,tmp_path,tmp_path,[],{})
    backend.cleanup_container(SimpleNamespace(id='sb-fixture'))
    assert calls==['diagnostics','usage','delete']
    assert not list(runner.public_artifact_paths(tmp_path))
    if collector_fails:
        assert json.loads((tmp_path/'.private/diagnostics/sb-fixture/capture-unavailable.json').read_text())=={'error':'RuntimeError'}


def test_cpu_judge_bootstrap_retains_private_checks_and_skips_all_gpu_work(tmp_path, monkeypatch):
    import sforge.pyromind.runner as runner
    import sforge.pyromind.gpu_runtime as gpu_runtime
    calls = []
    config = {'execution_kind': 'cpu', 'resources': {'gpu': 0, 'cpu': 4, 'memory': '8Gi'},
              'judge_resources': {'gpu': 0, 'cpu': 8, 'memory': '16Gi'},
              'runtime_expectations': {'python': '3.12', 'architecture': 'x86_64'}}
    deployment = Deployment(tmp_path / 'deployment.json', tmp_path, config)
    metadata = {'files': []}
    backend = SimpleNamespace(cleanup_container=lambda *args: None,
                              _checked=lambda *args: pytest.fail('CPU Judge called nvidia-smi'),
                              _event=lambda event, *args, **kwargs: calls.append(event))
    module = SimpleNamespace(install_private_judge=lambda *args: calls.append('private') or metadata,
                             validate_judge=lambda *args: calls.append('task') or {'valid': True})
    for name in ('prepare_hami_runtime', 'capture_gpu_topology', 'verify_distributed_runtime'):
        monkeypatch.setattr(gpu_runtime, name, lambda *args, **kwargs: pytest.fail('GPU initialization on CPU'))
    monkeypatch.setattr(runner, 'verify_asset_mounts', lambda *args, **kwargs: calls.append('assets'))
    monkeypatch.setattr(runner, 'checked', lambda *args, **kwargs: 'cpu runtime checked')
    configure_judge_runtime(backend, deployment, module, tmp_path, tmp_path, [], metadata)
    backend.on_started(SimpleNamespace(id='cpu-judge'))
    assert calls == ['assets', 'assets', 'private', 'task', 'cpu_verified']
    record = json.loads((tmp_path / 'judge-bootstrap/cpu-judge.json').read_text())
    assert record['gpu'] is None and record['platform_gpu_runtime'] is None
    assert record['private_payload_manifest'] == metadata
    assert not (tmp_path / 'gpu-topology').exists()


@pytest.mark.parametrize('failure', ['none', 'collector', 'memory'])
def test_worker_failure_cleanup_collects_only_worker_diagnostics(tmp_path, monkeypatch, failure):
    import sforge.pyromind.runner as runner
    calls = []
    def collect(backend, handle, task_root, directory):
        calls.append('worker-log')
        assert directory == tmp_path / '.private/diagnostics/sb-worker'
        if failure == 'collector': raise RuntimeError('private details')
        (directory / 'worker.log').write_text('private native traceback')
    def memory(*args):
        calls.append('memory')
        if failure == 'memory': raise RuntimeError('private details')
    backend = SimpleNamespace(cleanup_container=lambda handle, logger=None: calls.append('delete'))
    module = SimpleNamespace(collect_worker_diagnostics=collect,
                             collect_private_diagnostics=lambda *args: pytest.fail('Wrong role collector'))
    monkeypatch.setattr(runner, 'capture_host_memory', memory)
    runner.configure_runtime_cleanup(backend, module, Path('/task'), tmp_path, role='worker')
    backend.cleanup_container(SimpleNamespace(id='sb-worker'))
    assert calls == ['worker-log', 'memory', 'delete']
    assert not any('private' in str(path.relative_to(tmp_path)) for path in runner.public_artifact_paths(tmp_path))


def test_native_mode_requires_task_owned_exact_contract():
    module=SimpleNamespace(qualification_runtime_spec=lambda:{'mode':'non-release-qualification',
        'release_qualification_permitted':False,'allowed_execution_blockers':['native-pending']},
        qualification_evaluation_context=lambda:None)
    def ordinary_gate(**kwargs):raise ValueError('ordinary native pending')
    deployment=SimpleNamespace(config={'execution_blockers':['native-pending']},assert_launch_ready=ordinary_gate)
    args=SimpleNamespace(native_requalification=False,rounds=['final'],model_validation=Path('/receipt'))
    with pytest.raises(ValueError,match='ordinary native pending'):native_requalification_spec(deployment,module,args)
    args.native_requalification=True
    assert native_requalification_spec(deployment,module,args)['mode']=='non-release-qualification'
    args.rounds=['interactive','final']
    with pytest.raises(ValueError,match='final-only'):native_requalification_spec(deployment,module,args)
    args.rounds=['final'];deployment.config['execution_blockers'].append('wrong-model')
    with pytest.raises(ValueError,match='exact gates'):native_requalification_spec(deployment,module,args)


def checkpoint_tar():
    output=io.BytesIO()
    with tarfile.open(fileobj=output,mode='w') as bundle:
        root=tarfile.TarInfo('checkpoints');root.type=tarfile.DIRTYPE;bundle.addfile(root)
        entry=tarfile.TarInfo('checkpoints/deadline.json');data=b'{"deadline":123}'
        entry.size=len(data);bundle.addfile(entry,io.BytesIO(data))
    return output.getvalue()


@pytest.mark.parametrize('broken',[False,True])
def test_checkpoint_capture_validates_and_persists_before_delete(tmp_path,broken):
    calls=[]
    backend=SimpleNamespace(on_started=lambda handle:calls.append('started'),
        cleanup_container=lambda *args:calls.append('deleted'),_checked=lambda *args:None,
        copy_from_container=lambda *args:(calls.append('copied') or (b'' if broken else checkpoint_tar())))
    install_checkpoint_capture(backend,tmp_path,{'checkpoint_mount':'/logs/verifier/checkpoints'})
    handle=SimpleNamespace(id='sb-fixture')
    backend.on_started(handle)
    if broken:
        with pytest.raises(RuntimeError,match='capture failed'):backend.cleanup_container(handle)
        assert not (tmp_path/'.private/checkpoints/sb-fixture.json').exists()
    else:
        backend.cleanup_container(handle)
        proof=json.loads((tmp_path/'.private/checkpoints/sb-fixture.json').read_text())
        assert proof['copied_before_delete'] is True
        assert 'checkpoints/deadline.json' in proof['verified_archive_members']
        assert (tmp_path/'.private/checkpoints/sb-fixture.tar').read_bytes()==checkpoint_tar()
    assert calls==['started','copied','deleted']


@pytest.mark.parametrize('changed',[{}, {'generation_count':519}, {'qualification':False}])
def test_nonrelease_success_requires_full_original_workload(tmp_path,changed):
    detail={'qualification':True,'release_evidence':False,'generation_count':520,
            'records':[{}]*520,'rows':{str(i):{} for i in range(12)},'row_commitments':[{}]*12}
    detail.update(changed)
    report=tmp_path/'report.tar.gz'
    with tarfile.open(report,'w:gz') as bundle:
        data=json.dumps(detail).encode();entry=tarfile.TarInfo('decoder_private.json');entry.size=len(data)
        bundle.addfile(entry,io.BytesIO(data))
    spec={'success_contract':{'private_flags':{'qualification':True,'release_evidence':False},
                             'generation_count':520,'record_count':520,'row_count':12,'row_commitment_count':12}}
    if changed:
        with pytest.raises(RuntimeError,match='full workload'):check_native_report(report,spec)
    else:assert check_native_report(report,spec)['release_evidence'] is False


@pytest.mark.parametrize('invalid_final',[False,True])
@pytest.mark.parametrize('final_round',['manual-1','final-1'])
@pytest.mark.parametrize('execution_kind', ['gpu', 'cpu'])
@pytest.mark.parametrize('unknown_create', [False, True])
@pytest.mark.pyromind_sdk
def test_qualification_runs_separate_original_rounds_and_keeps_evidence_private(tmp_path,monkeypatch,invalid_final,final_round,execution_kind,unknown_create):
    import sforge.pyromind.qualification as q
    import sforge.harness.backend.pyromind_backend as bm
    import sforge.harness.benchmark as benchmark
    import sforge.harness.task_spec as ts
    import sforge.harness.run_evaluation as evaluation
    profile=tmp_path/'profile.json';profile.write_text('{}')
    original_images={'judge':'repo@sha256:'+'a'*64}
    images=tmp_path/'images.json';images.write_text(json.dumps(original_images))
    def evaluation_context(*args):
        # Source files may change after loading; evidence must bind the actual
        # loaded configuration and image selection, including failed rounds.
        profile.write_text(json.dumps({'qualification_status':'updated elsewhere'}))
        images.write_text(json.dumps({'judge':'later@sha256:'+'b'*64}))
        return {'COHORT_SECRET':'private-secret'}
    private=SimpleNamespace(private_payload_metadata=lambda root:{'files':[]},
                            private_evaluation_context=evaluation_context)
    deployment=SimpleNamespace(path=profile,config={'name':'fixture','task_id':'fixture','final_round_id':final_round},sandbox_ready_timeout_seconds=900,
        resources={'gpu':1,'gpu_card':'H100','cpu':4,'memory':'8Gi'},
        assert_launch_ready=lambda **kwargs:None,
        local_path=lambda key:images if key=='images_file' else tmp_path,
        asset_bindings=lambda path:[],judge_asset_bindings=lambda path:[],bootstrap=lambda:private)
    if execution_kind == 'cpu':
        deployment.config.update(execution_kind='cpu', judge_resources={'gpu': 0, 'cpu': 8, 'memory': '16Gi'})
    profile.write_text(json.dumps(deployment.config))
    original_source_sha=hashlib.sha256(profile.read_bytes()).hexdigest()
    deployment.evaluation_round_id=lambda mode:Deployment.evaluation_round_id(deployment,mode)
    deployment.resources_for_role=lambda role:deployment.config.get(role+'_resources',deployment.resources)
    monkeypatch.setattr(q.Deployment,'load',lambda *args:deployment)
    monkeypatch.setattr(q,'configure_judge_runtime',lambda *args:None)
    monkeypatch.setattr(benchmark,'load_benchmark',lambda *args:None)
    task=SimpleNamespace(task_id='fixture')
    monkeypatch.setattr(ts,'load_all_tasks',lambda *args:[task])
    backend_instances=[]
    class Backend:
        def __init__(self,**kwargs):
            self.options = kwargs
            self.active={};self.deleted=[];self.events=kwargs['event_path'];self.next_id=0
            self.client=SimpleNamespace(list=lambda:list(self.active.values()))
            backend_instances.append(self)
        def create_container(self,image,name,**kwargs):
            self.next_id+=1;handle=SimpleNamespace(id=f'sb-{self.next_id}',name=name)
            self.active[handle.id]=handle
            with self.events.open('a') as stream:stream.write(json.dumps({'event':'created','id':handle.id,'name':name})+'\n')
            return handle
        def cleanup_container(self,handle,logger=None):
            self.active.pop(handle.id,None);self.deleted.append(handle.id)
        def exec_run_with_timeout(self,*args,**kwargs):raise AssertionError('No shell needed in fixture')
        def reconcile_creation_intents(self):
            return {'remaining_sandbox_ids': [],
                    'unresolved_creation_names': ['q-owned-delayed'] if unknown_create else [],
                    'cleanup_errors': ['unresolved-create:q-owned-delayed'] if unknown_create else []}
    monkeypatch.setattr(bm,'PyroMindBackend',Backend)
    rounds=[]
    def evaluate(**kwargs):
        rounds.append(kwargs['round_id'])
        assert kwargs['config'].judge_extra_env=={'SFORGE_PRIVATE_CONTEXT_COHORT_SECRET':'private-secret'}
        assert kwargs['config'].judge_cpu_limit == (8 if execution_kind == 'cpu' else 4)
        assert kwargs['config'].judge_mem_limit == ('16Gi' if execution_kind == 'cpu' else '8Gi')
        backend=kwargs['backend'];handle=backend.create_container(kwargs['judge_image_ref'],kwargs['submission_id'])
        backend.cleanup_container(handle)
        valid=not(invalid_final and kwargs['round_id']==final_round)
        return SimpleNamespace(to_dict=lambda:{'valid':valid,'score':1 if valid else None})
    monkeypatch.setattr(evaluation,'judge_submission',evaluate)
    candidate=tmp_path/'candidate.tar.gz'
    with tarfile.open(candidate,'w:gz') as bundle:
        info=tarfile.TarInfo('candidate.py');data=b'x=1\n';info.size=len(data);bundle.addfile(info,io.BytesIO(data))
    args=SimpleNamespace(deployment=profile,candidate_archive=candidate,assets_file=None,
        judge_assets_file=None,rounds=['interactive','final'],artifacts=tmp_path/'out')
    if invalid_final:
        with pytest.raises(RuntimeError,match='positive-control evaluation failed'):_qualify_task(args)
    elif unknown_create:
        with pytest.raises(RuntimeError,match='Qualification cleanup is incomplete'):_qualify_task(args)
    else:_qualify_task(args)
    output=next((tmp_path/'out').iterdir());state=json.loads((output/'run.json').read_text())
    assert rounds==['agent-1',final_round]
    assert backend_instances[0].deleted==['sb-1','sb-2']
    assert state['remaining_sandbox_ids'] == (None if unknown_create else [])
    assert state['status'] == ('cleanup-failed' if unknown_create else 'failed' if invalid_final else 'passed')
    assert state['unresolved_creation_names'] == (['q-owned-delayed'] if unknown_create else [])
    if unknown_create:
        assert 'unresolved-create:q-owned-delayed' in state['cleanup_errors']
    assert state['worker_created'] is False and state['shared_artifacts'] is False
    assert json.loads((output/'deployment.json').read_text())==deployment.config
    assert json.loads((output/'images.json').read_text())==original_images
    assert state['deployment_sha256']==original_source_sha
    assert state['deployment_sha256']!=hashlib.sha256(profile.read_bytes()).hexdigest()
    assert state['deployment_snapshot_sha256']==hashlib.sha256((output/'deployment.json').read_bytes()).hexdigest()
    assert state['images_snapshot_sha256']==hashlib.sha256((output/'images.json').read_bytes()).hexdigest()
    assert state['runtime_contract_sha256']==qualification_contract_digest(deployment.config)
    assert state['resources'] == deployment.resources_for_role('judge')
    assert backend_instances[0].options['gpu_count'] == (0 if execution_kind == 'cpu' else 1)
    assert (output.stat().st_mode & 0o777)==0o700
    assert 'private-secret' not in (output/'run.json').read_text()
