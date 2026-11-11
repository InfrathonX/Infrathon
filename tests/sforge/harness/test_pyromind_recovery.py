"""Saved-code recovery must preserve provenance, submit boundaries and time."""
import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import pwd
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace

import pytest

from sforge.harness.version_store import VersionStore, VersionStoreCandidateError
from sforge.pyromind.recovery import load_recovery, remaining_budget, restore_worker
from sforge.pyromind.runner import initialize_task_store, submit_wait_diagnostics, supervisor_source


def bundle(files):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w:gz') as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
    return output.getvalue()


@pytest.fixture
def parent_run(tmp_path):
    now = time.time()
    parent = tmp_path/'parent-run'; parent.mkdir()
    state = {'run_id': parent.name, 'status': 'failed', 'task_id': 'task',
             'model': 'example-model', 'reasoning_effort': 'max', 'runtime_version': '0.159.3',
             'worker_image': 'worker@sha256:'+'a'*64, 'judge_image': 'judge@sha256:'+'b'*64,
             'budget_seconds': 43200, 'agent_started_at': now-3600,
             'agent_progress': {'started_at': now-3600, 'deadline_at': now+39600},
             'remaining_sandbox_ids': [], 'cleanup_errors': []}
    (parent/'run.json').write_text(json.dumps(state))
    (parent/'history.json').write_text(json.dumps({'run_id':parent.name, 'best_score':30.3,
        'best_round':'agent-4', 'private_report': 'must not be copied',
        'entries':[{'round':'agent-4','score':30.3,'valid':True,'summary':'v004 passed',
                    'private_report':'must not be copied'}]}))
    initialize_task_store(parent/'version-store', parent.name, 'task')
    store = VersionStore.open(parent/'version-store')
    task = SimpleNamespace(task_id='task', cwd=str(tmp_path/'workspace'),
                           submit_paths=['kernels','FINAL_VERSION','notes.md'], submit_exclude=['auth.json'])
    files = {'kernels/flash_kmeans_v004.py':b'best implementation', 'FINAL_VERSION':b'4\n',
             'notes.md':b'v004 best; v008 unfinished'}
    best = store.record_archive(task_id='task', run_id=parent.name, source='agent', round_id='agent-4',
        label=None, archive=bundle(files), submit_paths=task.submit_paths, submit_exclude=task.submit_exclude)
    latest = store.record_archive(task_id='task', run_id=parent.name, source='checkpoint', round_id='checkpoint-15',
        label='v008 retained', archive=bundle({**files,'kernels/flash_kmeans_v008.py':b'new experiment'}),
        submit_paths=task.submit_paths, submit_exclude=task.submit_exclude)
    kwargs = {'task':task, 'model':{'MODEL':'example-model','REASONING_EFFORT':'max'},
              'runtime':SimpleNamespace(version='0.159.3'), 'worker_image':state['worker_image'],
              'judge_image':state['judge_image'], 'now':now}
    return SimpleNamespace(parent=parent, state=state, store=store, latest=latest, best=best,
                           task=task, kwargs=kwargs, now=now)


def test_latest_checkpoint_preserves_best_and_unjudged_experiment_without_parent_mutation(parent_run):
    fixture=parent_run
    before={str(path):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in fixture.parent.rglob('*') if path.is_file()}
    plan=load_recovery(fixture.parent, **fixture.kwargs)
    assert plan.parent_version_id==fixture.latest.version_id
    assert plan.budget_seconds==43200
    assert plan.original_deadline_at==fixture.state['agent_progress']['deadline_at']
    assert plan.metadata(now=fixture.now)['remaining_seconds_at_recovery']==pytest.approx(39600)
    assert plan.metadata(now=fixture.now)['conversation_restored'] is False
    with tarfile.open(fileobj=io.BytesIO(plan.archive)) as archive:
        assert archive.extractfile('FINAL_VERSION').read()==b'4\n'
        assert archive.extractfile('kernels/flash_kmeans_v004.py').read()==b'best implementation'
        assert archive.extractfile('kernels/flash_kmeans_v008.py').read()==b'new experiment'
    assert 'private_report' not in json.dumps(plan.public_history)
    assert plan.public_history['entries'][0]['summary']=='v004 passed'
    assert 'original Codex conversation was not restored' in plan.guidance()
    after={str(path):hashlib.sha256(path.read_bytes()).hexdigest()
           for path in fixture.parent.rglob('*') if path.is_file()}
    assert after==before


def test_explicit_version_selects_earlier_judged_archive(parent_run):
    plan=load_recovery(parent_run.parent, version_id=parent_run.best.version_id, **parent_run.kwargs)
    assert plan.parent_version_id==parent_run.best.version_id
    assert 'kernels/flash_kmeans_v008.py' not in plan.members


def test_foreign_version_is_not_accepted(parent_run):
    with pytest.raises(ValueError,match='not found'):
        load_recovery(parent_run.parent, version_id='other-run-version', **parent_run.kwargs)


def test_archive_sha_and_size_must_match_immutable_manifest(parent_run):
    parent_run.latest.archive_path.write_bytes(parent_run.latest.archive_path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='SHA-256'):
        load_recovery(parent_run.parent, **parent_run.kwargs)


def test_restore_rechecks_current_submit_allowlist(parent_run):
    parent_run.task.submit_paths=['FINAL_VERSION']
    with pytest.raises(VersionStoreCandidateError,match='outside the submit contract'):
        load_recovery(parent_run.parent, **parent_run.kwargs)


@pytest.mark.parametrize('field,value',[
    ('task_id','other-task'), ('model','other-model'), ('reasoning_effort','high'),
    ('runtime_version','0.156.1'), ('worker_image','new-worker@sha256:'+'c'*64),
    ('judge_image','new-judge@sha256:'+'d'*64),
])
def test_recovery_rejects_changed_task_model_runtime_and_images(parent_run,field,value):
    parent_run.state[field]=value
    (parent_run.parent/'run.json').write_text(json.dumps(parent_run.state))
    with pytest.raises(ValueError,match=field):
        load_recovery(parent_run.parent, **parent_run.kwargs)


def test_original_deadline_expired_is_not_a_fresh_budget(parent_run):
    kwargs={**parent_run.kwargs, 'now':parent_run.state['agent_progress']['deadline_at']+1}
    with pytest.raises(ValueError,match='expired'):
        load_recovery(parent_run.parent, **kwargs)
    with pytest.raises(ValueError,match='expired'):
        remaining_budget(parent_run.now,now=parent_run.now)


def test_repeated_recovery_retains_first_run_deadline(parent_run):
    state=parent_run.state
    state['recovery']={'original_agent_started_at':state['agent_progress']['started_at'],
                       'original_deadline_at':state['agent_progress']['deadline_at'], 'recovery_count':1}
    state['agent_progress']={'started_at':parent_run.now-10,
                             'deadline_at':state['recovery']['original_deadline_at']}
    (parent_run.parent/'run.json').write_text(json.dumps(state))
    plan=load_recovery(parent_run.parent,**parent_run.kwargs)
    assert plan.recovery_count==2
    assert plan.original_deadline_at==state['recovery']['original_deadline_at']


@pytest.mark.parametrize('changed',[
    {'status':'agent-running'}, {'cleanup_errors':['DeletionFailed']},
    {'remaining_sandbox_ids':['still-running']},
])
def test_recovery_cannot_duplicate_unfinished_parent(parent_run,changed):
    (parent_run.parent/'run.json').write_text(json.dumps({**parent_run.state,**changed}))
    with pytest.raises(ValueError):
        load_recovery(parent_run.parent, **parent_run.kwargs)


@pytest.mark.parametrize('field',['cleanup_errors','remaining_sandbox_ids'])
@pytest.mark.parametrize('value',[None,False,{},'[]'])
def test_recovery_requires_explicit_typed_cleanup_completion(parent_run,field,value):
    parent_run.state[field]=value
    (parent_run.parent/'run.json').write_text(json.dumps(parent_run.state))
    with pytest.raises(ValueError,match='confirmed cleanup'):
        load_recovery(parent_run.parent, **parent_run.kwargs)


@pytest.mark.parametrize('field',['cleanup_errors','remaining_sandbox_ids'])
def test_recovery_rejects_failed_run_still_inside_cleanup_window(parent_run,field):
    parent_run.state.pop(field)
    (parent_run.parent/'run.json').write_text(json.dumps(parent_run.state))
    with pytest.raises(ValueError,match='confirmed cleanup'):
        load_recovery(parent_run.parent, **parent_run.kwargs)


def test_worker_restore_reads_back_files_and_keeps_original_task_files(parent_run,tmp_path,monkeypatch):
    plan=load_recovery(parent_run.parent,**parent_run.kwargs)
    workspace=Path(parent_run.task.cwd);workspace.mkdir()
    (workspace/'AGENT_TASK.md').write_text('original task design')
    remote=tmp_path/'uploaded.tar.gz';writes=[];deleted=[]
    def copy(_worker,source,_destination):remote.write_bytes(source.read_bytes())
    backend=SimpleNamespace(copy_to_container=copy,
        write_to_container=lambda worker,data,path:writes.append((str(path),data)),
        client=SimpleNamespace(delete_file=lambda *args:deleted.append(args)))
    monkeypatch.setattr(pwd,'getpwnam',lambda name:SimpleNamespace(pw_uid=os.getuid(),pw_gid=os.getgid()))
    def checked(backend,worker,command,**kwargs):
        if command[:2]!=['python3','-c']:return ''
        cfg=json.loads(command[3]);cfg['archive_path']=str(remote)
        monkeypatch.setattr(sys,'argv',['restore',json.dumps(cfg)])
        output=io.StringIO()
        with redirect_stdout(output):exec(command[2],{})
        return output.getvalue()
    evidence=restore_worker(plan,backend,SimpleNamespace(id='worker'),parent_run.task,checked)
    assert evidence['archive_sha256']==plan.archive_sha256
    assert (workspace/'FINAL_VERSION').read_bytes()==b'4\n'
    assert (workspace/'kernels/flash_kmeans_v008.py').read_bytes()==b'new experiment'
    assert (workspace/'AGENT_TASK.md').read_text()=='original task design'
    for record in evidence['files']:
        assert record['sha256']==hashlib.sha256((workspace/record['path']).read_bytes()).hexdigest()
    assert writes[0][0]=='/home/agent/recovery-history.json'
    assert deleted==[('worker','/run/sforge-agent/recovery.tar.gz')]


@pytest.mark.parametrize('recovery',[False,True])
def test_supervisor_deadline_uses_original_absolute_bound_not_a_fresh_budget(recovery):
    tree=ast.parse(supervisor_source())
    names={'started','deadline_at','remaining','deadline','status'}
    nodes=[node for node in tree.body if isinstance(node,ast.Assign)
           and any(isinstance(target,ast.Name) and target.id in names for target in node.targets)]
    cfg={'seconds':43200}
    if recovery:cfg['deadline_at']=10400
    namespace={'cfg':cfg,'time':SimpleNamespace(time=lambda:10000,monotonic=lambda:200)}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<budget>','exec'),namespace)
    assert namespace['deadline']==(600 if recovery else 43400)
    assert namespace['status']['deadline_at']==(10400 if recovery else 53200)
    assert namespace['status']['budget_seconds']==43200


def test_failure_diagnostics_export_ids_only_not_commands_transcripts_or_result_body():
    document={'schema_version':1,'active':None,'active_tools':{'hash':{
        'session_id':'session-123','tool_use_id':'call_123','tool_name':'exec_command','started_at':10,
        'command':'secret content','transcript_path':'private-session.jsonl','output':'private output'}},
        'last_completion':{'outcome':'completed','tool_use_id':'call_122','finished_at':8,'output':'private output'}}
    result=submit_wait_diagnostics(document)
    assert result['pending_tools']==[{'session_id':'session-123','tool_use_id':'call_123',
                                      'tool_name':'exec_command','started_at':10}]
    assert 'private' not in json.dumps(result) and 'secret' not in json.dumps(result)


def test_failure_diagnostics_accept_real_generated_hook_pending_active_and_completed_timestamps(tmp_path):
    from sforge.harness.submit_wait import generate_submit_wait_hook_script

    hook=tmp_path/'submit-wait-hook.py'
    hook.write_text(generate_submit_wait_hook_script())
    state_dir=tmp_path/'wait-state'
    fake_submit=tmp_path/'sforge-submit'
    fake_submit.write_text('#!'+sys.executable+'\nprint("private result must stay out of diagnostics")\n')
    fake_submit.chmod(0o755)
    environment={'PATH':str(tmp_path)+os.pathsep+os.defpath,
                 'SFORGE_SUBMIT_WAIT_STATE_DIR':str(state_dir)}
    event={'session_id':'01a0f8de-b002-74b1-bb63-0fb2b78d9a6d','tool_use_id':'ordinary-call',
           'tool_name':'Bash','tool_input':{'command':'printf ready'},'cwd':str(tmp_path)}

    def invoke(phase, payload):
        result=subprocess.run([sys.executable,str(hook),phase],input=json.dumps(payload),
                              capture_output=True,text=True,env=environment,check=True,timeout=10)
        return json.loads(result.stdout)

    invoke('pre',event)
    pending=json.loads((state_dir/'state.json').read_text())
    result=submit_wait_diagnostics(pending)
    assert result['pending_tools'][0]['started_at']==next(iter(pending['active_tools'].values()))['started_at']
    assert result['pending_tools'][0]['started_at'].endswith('Z')
    invoke('post',{**event,'tool_response':{'exit_code':0,'output':'ready'}})

    event={**event,'tool_use_id':'submit-call','tool_input':{'command':'sforge-submit'}}
    assert invoke('pre',event)['hookSpecificOutput']['permissionDecision']=='allow'
    active=json.loads((state_dir/'state.json').read_text())
    result=submit_wait_diagnostics(active)
    assert result['active_submission']['started_at']==active['active']['started_at']
    assert result['active_submission']['result_ready'] is True
    assert result['active_submission']['timeout_ms']==active['active']['timeout_ms']
    assert 'result_id' not in json.dumps(result) and 'private result' not in json.dumps(result)

    invoke('post',{**event,'tool_response':{'exit_code':0,'output':'delivered'}})
    completed=json.loads((state_dir/'state.json').read_text())
    result=submit_wait_diagnostics(completed)
    assert result['active_submission'] is None
    assert result['last_completion']['finished_at']==completed['last_completion']['finished_at']
    assert result['last_completion']['outcome']=='completed'


@pytest.mark.parametrize('timestamp',[
    '2026-10-01T19:08:58.123Z','2024-02-29T00:00:00.000Z',10,10.5,
])
def test_failure_diagnostics_accept_canonical_utc_and_legacy_numeric_timestamps(timestamp):
    document={'schema_version':1,'active':None,'active_tools':{},
              'last_completion':{'finished_at':timestamp}}
    assert submit_wait_diagnostics(document)['last_completion']['finished_at']==timestamp


@pytest.mark.parametrize('timestamp',[
    '2026-02-30T19:08:58.123Z','0000-10-01T19:08:58.123Z',
    '2026-10-01T24:08:58.123Z','2026-10-01T19:08:60.123Z',
    '2026-10-01T19:08:58Z','2026-10-01T19:08:58.123456Z',
    '2026-10-01T19:08:58.123+00:00','2026-10-01T19:08:58.123Z\n',
    True,False,0,-1,float('nan'),float('inf'),[],{},
])
def test_failure_diagnostics_reject_noncanonical_or_invalid_timestamps(timestamp):
    document={'schema_version':1,'active':None,'active_tools':{},
              'last_completion':{'finished_at':timestamp}}
    with pytest.raises(ValueError,match='timestamp'):
        submit_wait_diagnostics(document)


@pytest.mark.parametrize('timeout',[True,False,None,0,-1,172800001,1.5,'1000',float('nan')])
def test_failure_diagnostics_timeout_matches_generated_hook_positive_integer_limit(timeout):
    document={'schema_version':1,'active':{'timeout_ms':timeout},'active_tools':{}}
    with pytest.raises(ValueError,match='timeout'):
        submit_wait_diagnostics(document)


@pytest.mark.parametrize('timeout',[1,21660000,172800000])
def test_failure_diagnostics_accept_supported_timeout_bounds(timeout):
    document={'schema_version':1,'active':{'timeout_ms':timeout},'active_tools':{}}
    assert submit_wait_diagnostics(document)['active_submission']['timeout_ms']==timeout
